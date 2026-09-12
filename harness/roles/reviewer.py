"""
Reviewer agent -- never writes code. A mandatory gate: the Software
Engineer's code cannot reach execution without a Reviewer pass (Deep
Thought failure #5: optional collaboration doesn't reliably happen even
when the role exists).

Reads every file the Software Engineer produced directly (deterministically,
via Python) and hands them to the model in one prompt, rather than giving
it its own read_file tool loop -- a candidate's whole project is small
enough (a handful of KB) to just include outright, and the Reviewer's job
is to look at everything anyway, so a selective-fetch tool loop would only
add turns and cost, not capability. Deliberate simplification from
ARCHITECTURE.md's original "tools: read_file" wording -- see DECISIONS.md.

Checks three kinds of things:
1. A standing checklist that applies to every candidate regardless of what
   the blueprint says (the fixed CLI contract, real hyperparameter tuning,
   correct use of the provided QM8 loader) -- these are fixed requirements
   stated in the Software Engineer's own system prompt, and Deep Thought
   failure #1 is exactly "don't trust a vibe check that stated requirements
   were followed."
2. The blueprint's own stated constraints (candidate-specific).
3. General code correctness/completeness.
"""

import json
import re
from pathlib import Path

from harness.config import REVIEWER_MODEL, REVIEWER_TEMP
from harness.llm_client import call_model
from harness.trace import Trace

STANDING_CHECKLIST = [
    "main.py is runnable exactly as `python main.py --stage baseline`, "
    "`python main.py --stage train`, and `python main.py --stage evaluate` "
    "(these three stages must exist under these exact names)",
    "The `train` stage performs a real cross-validated hyperparameter search "
    "over multiple configurations (e.g. GridSearchCV/RandomizedSearchCV) -- "
    "a single fixed, hand-picked configuration is a FAIL",
    "QM8 is loaded via `from qm8_data import load_qm8` -- the code must not "
    "reimplement QM8 loading or use torch_geometric.datasets.QM8/deepchem",
    "The `evaluate` stage writes its final metric to results.json in the "
    "candidate's own directory, not only to stdout",
    "requirements.txt lists only packages genuinely imported and not already "
    "pre-installed (torch, torch_geometric, rdkit, pandas, numpy, "
    "scikit-learn are already available)",
]

# Pre-verified harness infrastructure, not the Software Engineer's own work
# -- reviewing qm8_data.py would be reviewing us, not the candidate.
_EXCLUDED_FILES = {"qm8_data.py"}


def _read_candidate_files(candidate_dir: Path) -> dict[str, str]:
    files = {}
    for path in sorted(candidate_dir.rglob("*")):
        if path.is_file() and path.name not in _EXCLUDED_FILES and "__pycache__" not in path.parts:
            files[str(path.relative_to(candidate_dir))] = path.read_text(errors="replace")
    return files


def review_candidate(
    candidate_dir: Path,
    constraints: list[str],
    trace: Trace,
    execution_error: str | None = None,
) -> tuple[bool, str]:
    """
    Returns (passed, notes). `execution_error` is given only on a re-review
    after a real execution failure (ARCHITECTURE.md step 3d -> 3b) -- when
    present, the Reviewer is explicitly asked to distinguish "the code is
    wrong" from "the check/constraint itself is wrong" (Deep Thought
    failure #7: don't let this become an infinite loop fixing the wrong thing).
    """
    files = _read_candidate_files(candidate_dir)
    files_text = "\n\n".join(f"--- {name} ---\n{content}" for name, content in files.items())

    system_prompt = f"""You are the Reviewer agent. You never write code -- only judge it. \
Check the Software Engineer's submitted files below against two checklists. \
Every item must be checked explicitly; do not give a vibe-check pass.

STANDING CHECKLIST (applies to every candidate, always):
{json.dumps(STANDING_CHECKLIST, indent=2)}

BLUEPRINT CONSTRAINTS (specific to this candidate):
{json.dumps(constraints, indent=2)}

Submitted files:
{files_text}"""

    if execution_error:
        system_prompt += f"""

This code already passed a previous review but FAILED when actually executed:
{execution_error}

Determine whether the CODE is wrong (a real bug to fix) or whether a CHECK/
CONSTRAINT itself is unreasonable or was misapplied -- don't let this become \
an infinite loop fixing the wrong thing. State which in your notes explicitly."""

    system_prompt += """

Respond with exactly one fenced ```json block:
{"verdict": "PASS" or "FAIL", "notes": "specific, actionable feedback -- cite \
exactly which checklist item(s) failed and why, so the Software Engineer can \
fix it without guessing"}"""

    # No tool loop here (this call has no tools), so the format/parse retry
    # is a small local loop -- same fix as the Planner functions' fenced-
    # block validation, applied here after a real run crashed on
    # json.JSONDecodeError with no retry at all (DECISIONS.md).
    messages = [{"role": "system", "content": system_prompt}]
    verdict = None
    for _ in range(3):
        response = call_model(
            model=REVIEWER_MODEL,
            messages=messages,
            temperature=REVIEWER_TEMP,
            trace=trace,
            stage="reviewer",
        )
        reply = response.choices[0].message.content or ""
        match = re.search(r"```json\n(.*?)\n```", reply, re.DOTALL)
        error = None
        if not match:
            error = "missing a fenced ```json block"
        else:
            try:
                parsed = json.loads(match.group(1))
                # Valid JSON with the wrong keys is a real, separate failure
                # mode from invalid JSON syntax -- e.g. {"result": "PASS"}
                # instead of {"verdict": ..., "notes": ...} would otherwise
                # raise an uncaught KeyError below instead of retrying.
                if "verdict" not in parsed or "notes" not in parsed:
                    error = "was valid JSON but missing the required 'verdict' and/or 'notes' keys"
                else:
                    verdict = parsed
            except json.JSONDecodeError as e:
                error = f"contained invalid JSON ({e})"
        if verdict is not None:
            break
        messages.append({"role": "assistant", "content": reply})
        messages.append({
            "role": "user",
            "content": (
                f"Your response {error}. You MUST respond with exactly one "
                "fenced ```json block, valid JSON, with no other text "
                "outside it. Please provide your complete final answer now, "
                "in that exact format."
            ),
        })

    if verdict is None:
        raise ValueError(f"Reviewer's response {error} after 3 attempts:\n" + reply)

    return verdict["verdict"] == "PASS", verdict["notes"]
