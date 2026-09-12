"""
Deterministic execution of a candidate's real run -- no LLM involved, per
ARCHITECTURE.md's "execution is always deterministic" design. This is the
LOCAL implementation (DECISIONS.md: Kaggle deferred to an optional later
upgrade, since candidate 1 is classical ML and genuinely CPU-tractable).

Runs install -> train -> evaluate against the fixed CLI contract every
Software Engineer candidate is required to implement
(`python main.py --stage {baseline,train,evaluate}`). Swapping this for a
Kaggle-backed implementation later only means writing a new function with
the same (candidate_dir) -> ExecutionResult shape -- the orchestrator's
loop logic doesn't need to change.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from harness.tools.shell import install_dependencies, venv_env
from harness.tools.subprocess_utils import TimedOut, run_with_timeout

TRAIN_TIMEOUT_SECONDS = 20 * 60
EVALUATE_TIMEOUT_SECONDS = 5 * 60


@dataclass
class ExecutionResult:
    success: bool
    stage_failed: str | None  # "install" | "train" | "evaluate" | None if success
    output: str  # combined stdout+stderr of whichever stage ran (or failed)
    results: dict | None  # parsed results.json, only present on success


def _run_stage(candidate_dir: Path, stage: str, timeout: int):
    return run_with_timeout(
        ["python", "main.py", "--stage", stage],
        cwd=candidate_dir,
        timeout=timeout,
        env=venv_env(),
    )


def execute_candidate(candidate_dir: Path) -> ExecutionResult:
    """
    The real run: install this candidate's requirements.txt, then
    `--stage train`, then `--stage evaluate`. Stops at the first failure
    and reports which stage it was -- the caller (orchestrator) feeds this
    straight back into the Reviewer per ARCHITECTURE.md step 3d -> 3b.
    """
    install_ok, install_output = install_dependencies(candidate_dir)
    if not install_ok:
        return ExecutionResult(success=False, stage_failed="install", output=install_output, results=None)

    try:
        train_result = _run_stage(candidate_dir, "train", TRAIN_TIMEOUT_SECONDS)
    except TimedOut:
        return ExecutionResult(
            success=False, stage_failed="train",
            output=f"(train timed out after {TRAIN_TIMEOUT_SECONDS}s)", results=None,
        )
    if train_result.returncode != 0:
        return ExecutionResult(
            success=False, stage_failed="train",
            output=train_result.stdout + "\n--- stderr ---\n" + train_result.stderr, results=None,
        )

    try:
        eval_result = _run_stage(candidate_dir, "evaluate", EVALUATE_TIMEOUT_SECONDS)
    except TimedOut:
        return ExecutionResult(
            success=False, stage_failed="evaluate",
            output=f"(evaluate timed out after {EVALUATE_TIMEOUT_SECONDS}s)", results=None,
        )
    if eval_result.returncode != 0:
        return ExecutionResult(
            success=False, stage_failed="evaluate",
            output=eval_result.stdout + "\n--- stderr ---\n" + eval_result.stderr, results=None,
        )

    results_path = candidate_dir / "results.json"
    if not results_path.exists():
        return ExecutionResult(
            success=False, stage_failed="evaluate",
            output=eval_result.stdout + "\n(evaluate exited 0 but wrote no results.json)", results=None,
        )

    return ExecutionResult(
        success=True, stage_failed=None,
        output=train_result.stdout + "\n" + eval_result.stdout,
        results=json.loads(results_path.read_text()),
    )
