"""
Wires the three roles and the deterministic tools into the actual pipeline
described in ARCHITECTURE.md. Plain Python control flow -- no framework,
per CLAUDE.md's "custom minimal loop" decision: the pipeline shape (plan ->
build/review/execute loop -> maybe repeat -> report) is simple enough that
a loop and a few functions express it exactly as well as a graph-based
framework would, while staying fully auditable (rule 3).

Candidate count is inherently capped at MAX_CANDIDATES (2) by construction,
not by an explicit counter -- candidate 1 always runs, and there is at most
one possible candidate 2 (designed from candidate 1's real result), so
there is no code path that could ever produce a candidate 3.
"""

import json
import time
from pathlib import Path

from harness.config import MAX_REVIEW_FIX_ROUNDS
from harness.roles.planner import design_candidate_2, draft_blueprint, write_final_report
from harness.roles.reviewer import review_candidate
from harness.roles.software_engineer import implement_candidate
from harness.tools.execute import execute_candidate
from harness.trace import Trace


def _build_review_execute_loop(
    candidate: dict, constraints: list[str], candidate_number: int, run_id: str, trace: Trace,
    diagnostic_plan: str = "",
) -> tuple[Path, dict | None, str | None]:
    """
    ARCHITECTURE.md steps 3a-3d: write, review, install+execute; a review
    FAIL sends it back to the Software Engineer with the Reviewer's notes,
    a real execution failure sends it back to the Reviewer first (with the
    actual error, so it can tell "code wrong" from "check wrong" --
    Deep Thought failure #7) which then re-issues fix notes. One shared
    counter for both failure types, capped at MAX_REVIEW_FIX_ROUNDS,
    tracked here as external state (failure #6), not left to either agent.

    Returns (candidate_dir, results_or_None, failure_reason_or_None).
    """
    reviewer_notes = None
    execution_error = None
    candidate_dir = None

    for round_num in range(1, MAX_REVIEW_FIX_ROUNDS + 1):
        try:
            candidate_dir = implement_candidate(
                candidate=candidate,
                constraints=constraints,
                candidate_number=candidate_number,
                run_id=run_id,
                trace=trace,
                diagnostic_plan=diagnostic_plan,
                reviewer_notes=reviewer_notes,
            )
        except RuntimeError as e:
            # The Software Engineer's tool loop ran out of turns (e.g. it
            # kept re-verifying already-working code instead of stopping --
            # DECISIONS.md). This is a recoverable failure for THIS round,
            # same as a review or execution failure -- it must not crash
            # the whole pipeline, which is what happened before this fix.
            trace.log_event(
                stage=f"candidate_{candidate_number}", event_type="implement_failed",
                round=round_num, error=str(e),
            )
            reviewer_notes = (
                "Your previous attempt ran out of turns before finishing -- it "
                "was still re-verifying already-working code instead of stopping. "
                "Be decisive: write the code, run baseline once, and the moment "
                "it succeeds, stop immediately with no further tool calls."
            )
            execution_error = None
            continue

        passed, notes = review_candidate(
            candidate_dir=candidate_dir,
            constraints=constraints,
            trace=trace,
            execution_error=execution_error,
        )
        trace.log_event(
            stage=f"candidate_{candidate_number}", event_type="review_result",
            round=round_num, passed=passed, notes=notes,
        )

        if not passed:
            # execution_error is deliberately NOT cleared here. If this
            # review round was checking code against a real runtime
            # failure, that context must survive until execution is
            # actually re-attempted -- clearing it here meant a Software
            # Engineer attempt that made no real change (e.g. the empty-
            # response bug just fixed above) could get a fresh PASS on
            # unchanged, still-broken code, because the next review had no
            # idea a timeout ever happened. Confirmed happening in a real
            # run: same hyperparameters, same code, PASS on round 3 after
            # FAIL on round 2 citing that exact timeout. See DECISIONS.md.
            reviewer_notes = notes
            continue

        # Review passed -- about to actually re-attempt execution, so any
        # stale prior failure is no longer relevant either way.
        execution_error = None

        result = execute_candidate(candidate_dir)
        trace.log_event(
            stage=f"candidate_{candidate_number}", event_type="execute_result",
            round=round_num, success=result.success, stage_failed=result.stage_failed,
            # The actual error text -- without this, diagnosing a real
            # failure means guessing from timestamps (found the hard way:
            # had to infer a likely timeout from elapsed time alone,
            # DECISIONS.md).
            output_preview=result.output[:2000],
        )

        if result.success:
            return candidate_dir, result.results, None

        # Real execution failure -- back to the Reviewer with the actual
        # error first (ARCHITECTURE.md step 3d -> 3b), not straight to the
        # Software Engineer, so it can distinguish a real bug from a bad check.
        reviewer_notes = None
        execution_error = f"Stage '{result.stage_failed}' failed:\n{result.output[:3000]}"

    return candidate_dir, None, f"exceeded MAX_REVIEW_FIX_ROUNDS={MAX_REVIEW_FIX_ROUNDS}"


def run_pipeline(task_description: str, background_docs: str = "") -> Path:
    """The full pipeline, ARCHITECTURE.md steps 1-5. Returns the trace file's path."""
    run_id = time.strftime("%Y%m%d_%H%M%S")
    trace = Trace(run_id=run_id)

    try:
        return _run_pipeline_steps(task_description, background_docs, run_id, trace)
    except Exception as e:
        # A real multi-hour, real-money run crashing with a bare traceback
        # and no record of where it died is a genuine gap -- happened
        # several times today (max_turns, JSON decode error, insufficient
        # credits, an f-string bug). This doesn't handle the failure, it
        # just guarantees the trace file itself says what happened before
        # the exception propagates and the caller/user sees it.
        trace.log_event(stage="pipeline", event_type="pipeline_crashed",
                         error_type=type(e).__name__, error=str(e))
        raise


def _run_pipeline_steps(task_description: str, background_docs: str, run_id: str, trace: Trace) -> Path:
    blueprint = draft_blueprint(task_description, background_docs, run_id, trace)

    all_results = [_run_one_candidate(blueprint["candidate_1"], blueprint, 1, run_id, trace)]

    candidate_1_dir, candidate_1_results = all_results[0]["dir"], all_results[0]["results"]
    if candidate_1_results is not None:
        candidate_2 = design_candidate_2(
            blueprint=blueprint,
            candidate_1_result=candidate_1_results,
            candidate_1_files=_read_candidate_files(candidate_1_dir),
            trace=trace,
        )
        if candidate_2 is not None:
            all_results.append(_run_one_candidate(candidate_2, blueprint, 2, run_id, trace))

    # Flat, number-prefixed across all candidates so write_final_report's
    # single python_sandbox can tell candidate_1_predictions.csv apart from
    # candidate_2_predictions.csv without a filename collision.
    all_candidate_files = {}
    for r in all_results:
        for name, content in _read_candidate_files(r["dir"]).items():
            all_candidate_files[f"candidate_{r['number']}_{name}"] = content

    report = write_final_report(
        blueprint=blueprint,
        results=[{"candidate_number": r["number"], "candidate": r["candidate"],
                  "results": r["results"], "failure": r["failure"]} for r in all_results],
        candidate_files=all_candidate_files,
        run_id=run_id,
        trace=trace,
    )
    trace.log_event(stage="pipeline", event_type="run_complete", report_chars=len(report))

    return trace.path


def _run_one_candidate(candidate: dict, blueprint: dict, number: int, run_id: str, trace: Trace) -> dict:
    # Candidate 2 may specify its own diagnostic_plan (it can reasonably want
    # different analysis than candidate 1, e.g. a GNN's per-atom
    # contributions vs. a tree model's feature importances); fall back to
    # the blueprint's original plan if it didn't.
    diagnostic_plan = candidate.get("diagnostic_plan") or blueprint.get("diagnostic_plan", "")
    candidate_dir, results, failure = _build_review_execute_loop(
        candidate=candidate, constraints=blueprint["constraints"],
        candidate_number=number, run_id=run_id, trace=trace,
        diagnostic_plan=diagnostic_plan,
    )
    return {"number": number, "candidate": candidate, "dir": candidate_dir,
            "results": results, "failure": failure}


_BINARY_EXTENSIONS = {".pkl", ".pickle", ".pt", ".pth", ".joblib", ".npy", ".npz", ".h5"}


def _read_candidate_files(candidate_dir: Path | None) -> dict[str, str]:
    """
    Reads back whatever the Software Engineer actually produced -- no fixed
    filenames assumed, since its diagnostic output is now its own design
    choice (DECISIONS.md), not a format we prescribe. Excludes qm8_data.py
    (pre-seeded infrastructure, not the candidate's own work), __pycache__,
    and binary files (e.g. a saved model.pkl) -- reading one as text and
    writing it back via python_sandbox's input_files would corrupt it, and
    neither downstream stage needs the raw model object, only its outputs.
    """
    if candidate_dir is None:
        return {}
    return {
        f.name: f.read_text(errors="replace")
        for f in candidate_dir.iterdir()
        if f.is_file()
        and f.name != "qm8_data.py"
        and f.suffix not in _BINARY_EXTENSIONS
        and "__pycache__" not in f.parts
    }
