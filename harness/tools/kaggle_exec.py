"""
Kaggle-backed execution of a candidate's real run -- an alternative to
execute.py's local implementation, same (candidate_dir) -> ExecutionResult
contract, so orchestrator.py only needs to pick which one to call
(config.EXECUTION_BACKEND). Built after real local CPU timing (DECISIONS.md)
showed this laptop needs 2-6+ hours for a GNN candidate; Kaggle's free GPU
tier removes that ceiling (9h hard cap per kernel run, ~30-40 GPU-hours/week
quota -- checked directly against Kaggle's own docs/community reports, not
assumed).

Two real constraints shaped this design, both things a fixed CLI-invocation
model (like the local runner's `python main.py --stage X`) can't just
assume still holds once execution moves to a remote kernel:

1. A Kaggle kernel's `kernel-metadata.json` has one `code_file` field, not a
   list -- verified against the Kaggle API's real schema before designing
   around it, not assumed. So a candidate's qm8_data.py and main.py are
   concatenated into one self-contained script per push. This is a
   deterministic string transform done here, never by the LLM -- same
   principle as the local runner deciding *how* to invoke unknown code.

2. Each Kaggle kernel run is a fresh, independent remote filesystem. Locally,
   `--stage train` (writes model.pt/scaler.pt) and `--stage evaluate` (reads
   them back) share a disk because they're two subprocess calls in the same
   candidate_dir. Two separate Kaggle pushes would NOT share a filesystem --
   evaluate would find nothing to load. Solved by running both stages in
   ONE kernel push: the generated script `exec()`s the candidate's own
   main.py body twice in the same process, patching `sys.argv` before each
   pass -- so train's output files are still on disk (same process, same
   /kaggle/working/) when the evaluate pass reads them. No Kaggle Datasets
   upload needed.

Everything else here (the exact status-string format, how output files come
back) was verified live against a real Kaggle account with a throwaway
smoke-test kernel before writing the parsing logic -- not guessed from docs,
which turned out to not even match: `kernels status` prints a sentence like
`... has status "KernelWorkerStatus.COMPLETE"`, not the bare "complete" the
public docs describe, hence the case-insensitive substring match below
rather than an exact comparison. `kernels output` downloads each file the
script wrote (e.g. results.json) under its own name, plus a `<slug>.log`
file that is a JSON array of {stream_name, time, data} entries -- not plain
text -- reconstructed into stdout/stderr text by `_parse_log`.
"""

import json
import os
import re
import time
from pathlib import Path

from harness.tools.execute import ExecutionResult
from harness.tools.subprocess_utils import TimedOut, run_with_timeout
from harness.trace import Trace

PUSH_TIMEOUT_SECONDS = 120
STATUS_CHECK_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 30
# Comfortably under Kaggle's own ~9h hard cap per kernel run (verified via
# Kaggle's own community docs, not assumed) -- this is OUR patience limit,
# not a Kaggle-side setting. Raised from 3h to 6h for headroom in case the
# Planner picks an ambitious architecture/hyperparameter search again
# (DECISIONS.md) -- still leaves ~3h margin below Kaggle's real hard cap.
RUN_TIMEOUT_SECONDS = 6 * 60 * 60
OUTPUT_PULL_TIMEOUT_SECONDS = 120

_KAGGLE_BIN = Path(__file__).parent.parent.parent / ".venv" / "bin" / "kaggle"


def _slug(run_id: str, candidate_number: int, round_num: int) -> str:
    """
    Kaggle kernel slugs allow only lowercase letters, digits, and dashes.
    Includes round_num so each retry round gets its OWN kernel rather than
    re-pushing to the same one -- re-pushing was found to risk `kernels
    status` briefly reporting the PREVIOUS round's terminal status before
    Kaggle finishes transitioning the kernel to the new push, which could
    read a stale "complete"/"error" for code that hasn't actually run yet
    (DECISIONS.md). A few extra kernels accumulating in the account across
    retries is harmless -- Kaggle doesn't limit or charge by kernel count.
    """
    raw = f"qm8-{run_id}-c{candidate_number}-r{round_num}"
    return re.sub(r"[^a-z0-9-]+", "-", raw.lower()).strip("-")


def _build_combined_script(candidate_dir: Path) -> str:
    qm8_data_src = (candidate_dir / "qm8_data.py").read_text()
    main_src = (candidate_dir / "main.py").read_text()
    # qm8_data's functions/classes are inlined above main_src in the same
    # file below, so main.py's own `from qm8_data import ...` line would
    # just be a redundant self-import -- stripped rather than left as a
    # confusing no-op. Two passes: a multi-line parenthesized import (e.g.
    # `from qm8_data import (\n    load_qm8,\n)`) first, since the single-
    # line pattern below only strips its opening line and leaves the
    # continuation lines behind as orphaned, syntactically invalid code --
    # a real gap found by inspection, not yet triggered live.
    main_src = re.sub(r"^from qm8_data import\s*\(.*?\)\s*$", "", main_src, flags=re.MULTILINE | re.DOTALL)
    main_src = re.sub(r"^(from qm8_data import .*|import qm8_data.*)$", "", main_src, flags=re.MULTILINE)

    extra_packages = []
    requirements_path = candidate_dir / "requirements.txt"
    if requirements_path.exists():
        for line in requirements_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                extra_packages.append(line)
    # rdkit/torch_geometric are not part of Kaggle's stock image (torch
    # itself is, GPU-enabled already) -- pip install is idempotent, so
    # listing them even if a future Kaggle image already includes them
    # just no-ops rather than errors.
    packages = sorted(set(extra_packages) | {"torch_geometric", "rdkit"})

    install_block = (
        "import subprocess as _subprocess, sys as _sys\n"
        f"_subprocess.check_call([_sys.executable, '-m', 'pip', 'install', '-q'] + {packages!r})\n\n"
    )

    # Runs the candidate's own `if __name__ == '__main__':` dispatch TWICE,
    # in the same process -- see module docstring for why this (not two
    # separate kernel pushes) is how train's output reaches evaluate.
    #
    # exec()'d against this script's own globals() (NOT a fresh/empty
    # namespace) -- the whole point of concatenating qm8_data.py above is
    # that load_qm8() etc. end up defined right there, and main.py's own
    # code (e.g. stage_train()) needs to actually see them. A separate
    # namespace dict would isolate main.py's exec from those definitions,
    # causing a NameError -- caught for real by testing end-to-end against
    # a live Kaggle run with a dummy candidate before ever trusting this
    # with the real, hours-long GNN run (DECISIONS.md). This script is
    # already running as __main__ on Kaggle, so globals() already has
    # __name__ == "__main__" -- no need to fake it either.
    run_block = f"""
import sys as _sys

_main_code = compile({main_src!r}, "main.py", "exec")

_sys.argv = ["main.py", "--stage", "train"]
exec(_main_code, globals())

_sys.argv = ["main.py", "--stage", "evaluate"]
exec(_main_code, globals())
"""

    return install_block + qm8_data_src + "\n\n" + run_block


def _write_push_dir(candidate_dir: Path, candidate_number: int, round_num: int, slug: str, username: str) -> Path:
    # A SIBLING of candidate_dir, not nested inside it -- reviewer.py's
    # read_candidate_files does a recursive rglob() over candidate_dir for
    # every re-review, so anything placed inside it becomes Reviewer input.
    # The pushed script here is a large, mostly-duplicate concatenation of
    # qm8_data.py + main.py -- real to keep for auditability (rule 6: every
    # run inspectable), just not something the Reviewer should be re-reading.
    # Scoped per round (not just per candidate) so an earlier round's pushed
    # script is never overwritten -- consistent with "nothing gets deleted."
    push_dir = candidate_dir.parent / f"candidate_{candidate_number}_kaggle_push_round{round_num}"
    push_dir.mkdir(exist_ok=True)
    (push_dir / "script.py").write_text(_build_combined_script(candidate_dir))
    metadata = {
        "id": f"{username}/{slug}",
        "title": slug,
        "code_file": "script.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,  # qm8_data.py downloads QM8's raw files from S3
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (push_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return push_dir


def _parse_log(log_path: Path) -> str:
    """
    Reconstructs plain stdout/stderr text from Kaggle's real log format
    (verified live: a JSON array of {stream_name, time, data} entries, in
    time order) -- so callers (Reviewer re-review, trace previews) see
    readable text/tracebacks, not a raw JSON blob.
    """
    if not log_path.exists():
        return "(no kernel log was downloaded)"
    entries = json.loads(log_path.read_text())
    stdout = "".join(e.get("data", "") for e in entries if e.get("stream_name") == "stdout")
    stderr = "".join(e.get("data", "") for e in entries if e.get("stream_name") == "stderr")
    return stdout + (f"\n--- stderr ---\n{stderr}" if stderr else "")


def _log(trace: Trace | None, event_type: str, **fields) -> None:
    if trace is not None:
        trace.log_event(stage="kaggle_exec", event_type=event_type, **fields)


def execute_candidate_kaggle(
    candidate_dir: Path, run_id: str, candidate_number: int, round_num: int,
    trace: Trace | None = None,
) -> ExecutionResult:
    """
    Same (candidate_dir) -> ExecutionResult contract as execute.py's local
    execute_candidate(). Pushes ONE combined kernel that runs both train and
    evaluate (see module docstring), polls until it finishes, then pulls
    results.json and the run log back into candidate_dir -- so downstream
    code (Reviewer re-review, orchestrator's read_candidate_files) doesn't
    need to know which backend actually ran the candidate.

    `trace`, if given, logs push/poll milestones under stage "kaggle_exec" --
    previously nothing about a Kaggle run (push outcome, how long polling
    took, a silent timeout) was ever written to the trace at all; only the
    caller's post-hoc `execute_result` summary existed. Real GPU-hours
    against Kaggle's weekly quota is exactly the kind of "budget spent"
    CLAUDE.md rule 6 says must be externalized (DECISIONS.md).
    """
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    # Both credentials and the actual binary are checked up front -- before
    # this fix, a missing KAGGLE_KEY or a venv never reinstalled after
    # `kaggle` was added to requirements.txt would raise an uncaught
    # FileNotFoundError from subprocess.Popen deep inside run_with_timeout,
    # which propagates all the way to run_pipeline's top-level crash handler
    # (killing the whole pipeline) instead of failing just this one round
    # like every other execution failure does.
    if not username or not key:
        return ExecutionResult(
            success=False, stage_failed="train",
            output="KAGGLE_USERNAME and/or KAGGLE_KEY is not set in .env -- cannot push to Kaggle.",
            results=None,
        )
    if not _KAGGLE_BIN.exists():
        return ExecutionResult(
            success=False, stage_failed="train",
            output=f"{_KAGGLE_BIN} does not exist -- is `kaggle` installed in the venv "
                   "(pip install -r requirements.txt)?",
            results=None,
        )

    slug = _slug(run_id, candidate_number, round_num)
    push_dir = _write_push_dir(candidate_dir, candidate_number, round_num, slug, username)

    try:
        push_result = run_with_timeout(
            # --accelerator NvidiaTeslaT4 is required, not optional: verified
            # live that an API-pushed kernel with only `enable_gpu: true`
            # defaults to a Tesla P100 (compute capability sm_60), which
            # Kaggle's own stock-image torch build (2.10.0+cu128, min
            # supported sm_70) cannot run at all -- "CUDA error: no kernel
            # image is available for execution on the device". Confirmed via
            # a throwaway diagnostic kernel: identical script, P100 crashes
            # on a plain GPU matmul, T4 runs it fine. This is a known,
            # documented Kaggle API quirk (their own product-feedback board),
            # not something fixable from inside the candidate's own code.
            [str(_KAGGLE_BIN), "kernels", "push", "-p", str(push_dir), "--accelerator", "NvidiaTeslaT4"],
            cwd=push_dir, timeout=PUSH_TIMEOUT_SECONDS,
        )
    except TimedOut:
        _log(trace, "kaggle_push", round=round_num, success=False, reason="timed_out")
        return ExecutionResult(success=False, stage_failed="train",
                                output=f"Kaggle push timed out after {PUSH_TIMEOUT_SECONDS}s", results=None)
    if push_result.returncode != 0:
        _log(trace, "kaggle_push", round=round_num, success=False, reason="nonzero_exit")
        return ExecutionResult(
            success=False, stage_failed="train",
            output="Kaggle push failed:\n" + push_result.stdout + push_result.stderr,
            results=None,
        )
    _log(trace, "kaggle_push", round=round_num, success=True, kernel_ref=f"{username}/{slug}")

    kernel_ref = f"{username}/{slug}"
    poll_started = time.monotonic()
    deadline = poll_started + RUN_TIMEOUT_SECONDS
    status_text = ""
    finished = False
    while time.monotonic() < deadline:
        time.sleep(POLL_INTERVAL_SECONDS)
        try:
            status_result = run_with_timeout(
                [str(_KAGGLE_BIN), "kernels", "status", kernel_ref],
                cwd=push_dir, timeout=STATUS_CHECK_TIMEOUT_SECONDS,
            )
        except TimedOut:
            continue  # one stuck status check shouldn't abandon the whole run
        status_text = (status_result.stdout + status_result.stderr).lower()
        if "complete" in status_text or "error" in status_text or "cancel" in status_text:
            finished = True
            break

    poll_duration_s = round(time.monotonic() - poll_started)

    if not finished:
        # Best-effort cleanup: the Kaggle API has no verified "cancel a
        # running kernel" call (confirmed against the real CLI's own
        # --help), only `delete`, whose effect on an IN-PROGRESS run is
        # unverified. Attempted anyway since it can only help, never hurt --
        # without it, a timed-out kernel just keeps running on Kaggle's side
        # up to their own ~9h cap, silently spending GPU-hours from the
        # weekly quota with nothing here even trying to stop it.
        try:
            run_with_timeout(
                [str(_KAGGLE_BIN), "kernels", "delete", "-y", kernel_ref],
                cwd=push_dir, timeout=STATUS_CHECK_TIMEOUT_SECONDS,
            )
            cleanup_note = "attempted `kernels delete` as best-effort cleanup (Kaggle has no verified cancel API)"
        except TimedOut:
            cleanup_note = "cleanup attempt itself timed out -- kernel may still be running on Kaggle"
        _log(trace, "kaggle_poll_timeout", round=round_num, poll_duration_s=poll_duration_s, cleanup=cleanup_note)
        return ExecutionResult(
            success=False, stage_failed="train_or_evaluate",
            output=(f"Kaggle run did not finish within {RUN_TIMEOUT_SECONDS}s "
                    f"(last status: {status_text!r}). {cleanup_note}."),
            results=None,
        )

    _log(trace, "kaggle_poll_complete", round=round_num, poll_duration_s=poll_duration_s, status=status_text.strip())

    # Pulled into a SIBLING dir, not candidate_dir directly -- same reasoning
    # as push_dir above. The real output files (results.json, and whatever
    # else the candidate's own diagnostic_plan produced) are copied into
    # candidate_dir explicitly below (only on success -- see below); the raw
    # `.log` (a JSON blob, not human/Reviewer-readable -- see _parse_log)
    # stays out of candidate_dir either way.
    output_dir = candidate_dir.parent / f"candidate_{candidate_number}_kaggle_output_round{round_num}"
    output_dir.mkdir(exist_ok=True)
    try:
        run_with_timeout(
            [str(_KAGGLE_BIN), "kernels", "output", kernel_ref, "-p", str(output_dir)],
            cwd=output_dir, timeout=OUTPUT_PULL_TIMEOUT_SECONDS,
        )
    except TimedOut:
        return ExecutionResult(success=False, stage_failed="train_or_evaluate",
                                output="Kaggle output download timed out", results=None)

    log_output = _parse_log(output_dir / f"{slug}.log")

    if "complete" not in status_text:
        # Ran both stages in one process (see module docstring), so on
        # failure we genuinely can't tell from the outside which of the two
        # raised -- the real traceback in log_output is the actual answer;
        # "train_or_evaluate" says so honestly instead of guessing one.
        #
        # Output files are NOT copied into candidate_dir on this path --
        # previously they always were, before this check, which meant a
        # results.json written successfully by evaluate right before some
        # LATER step in the same script crashed (ending the kernel in
        # "error" status) would still land in candidate_dir. The function's
        # own return value correctly said results=None, but orchestrator.py's
        # read_candidate_files() reads candidate_dir directly regardless,
        # so that stale file could still reach write_final_report's
        # python_sandbox as if it belonged to a successful run (DECISIONS.md).
        return ExecutionResult(success=False, stage_failed="train_or_evaluate", output=log_output, results=None)

    for f in output_dir.iterdir():
        if f.is_file() and f.suffix != ".log":
            (candidate_dir / f.name).write_bytes(f.read_bytes())

    results_path = candidate_dir / "results.json"
    if not results_path.exists():
        return ExecutionResult(
            success=False, stage_failed="evaluate",
            output=log_output + "\n(Kaggle run completed but wrote no results.json)",
            results=None,
        )

    return ExecutionResult(
        success=True, stage_failed=None, output=log_output,
        results=json.loads(results_path.read_text()),
    )
