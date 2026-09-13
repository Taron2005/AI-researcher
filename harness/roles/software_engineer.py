"""
Software Engineer agent -- writes the actual project files for one
candidate, per the blueprint. Never touches anything outside its own
workspace/run_<id>/candidate_N/ directory (files.py's path validation
enforces this). See ARCHITECTURE.md for its exact tools/inputs/outputs and
where it sits in the pipeline.

Candidate directories are scoped by run_id, not a fixed workspace/candidate_N
path -- otherwise a new pipeline run would silently overwrite a previous
run's real code, which violates the "nothing gets deleted/overwritten"
policy (DECISIONS.md) just as much as an explicit `rm` would.
"""

import json
from pathlib import Path

from harness.config import SWE_MAX_TOOL_TURNS, SWE_MODEL, SWE_TEMP
from harness.tool_loop import run_tool_loop
from harness.tools.files import READ_FILE_SCHEMA, WRITE_FILE_SCHEMA
from harness.tools.files import read_file as _read_file
from harness.tools.files import write_file as _write_file
from harness.tools.shell import LOCAL_RUN_SCHEMA
from harness.tools.shell import local_run as _local_run
from harness.trace import Trace

WORKSPACE = Path(__file__).parent.parent.parent / "workspace"
QM8_DATA_MODULE = Path(__file__).parent.parent / "qm8_data.py"

TOOLS = [WRITE_FILE_SCHEMA, READ_FILE_SCHEMA, LOCAL_RUN_SCHEMA]


def implement_candidate(
    candidate: dict,
    constraints: list[str],
    candidate_number: int,
    run_id: str,
    trace: Trace,
    diagnostic_plan: str = "",
    reviewer_notes: str | None = None,
) -> Path:
    """
    Writes (or fixes) candidate_number's project files under
    workspace/run_<run_id>/candidate_<candidate_number>/. Returns the
    candidate's directory once the model reports it's done.

    `diagnostic_plan` is the Planner's OWN decision (blueprint.json) on what
    analysis beyond the headline metric is needed -- deliberately not
    prescribed by us (DECISIONS.md: the user wanted the model driving what
    research to do, not told exactly what to compute).

    `reviewer_notes` is None on the first attempt for this candidate, and
    the Reviewer's specific feedback on any retry (ARCHITECTURE.md step 3a
    <-> 3b loop). This function doesn't track the retry count itself --
    that's the orchestrator's job (external state, per Deep Thought
    failure #6), this just does one attempt.
    """
    candidate_dir = WORKSPACE / f"run_{run_id}" / f"candidate_{candidate_number}"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # qm8_data.py is solved, verified infrastructure (DECISIONS.md) -- seeded
    # directly rather than making the agent rediscover QM8 loading, and
    # copied (not imported from harness/) so this candidate's folder stays
    # self-contained and portable to Kaggle, which never sees harness/.
    (candidate_dir / "qm8_data.py").write_text(QM8_DATA_MODULE.read_text())

    system_prompt = f"""You are the Software Engineer agent. Implement this candidate \
exactly as specified -- do not redesign it:
{json.dumps(candidate, indent=2)}

Constraints from the blueprint (the Reviewer checks every one of these):
{json.dumps(constraints, indent=2)}

You have no search tool -- you cannot look up new evidence yourself, so every \
design/implementation decision must be justified by what's ALREADY stated \
above, not invented. If the candidate names an architecture family (e.g. \
"SchNet-style"), implement the actual, standard version of that architecture \
as its own cited literature describes it -- do not silently substitute a \
different or heavier variant (e.g. a full per-edge weight matrix instead of \
SchNet's real elementwise filter) with no justification. Any detail the \
blueprint leaves unspecified should default to the simplest choice \
consistent with what IS stated, not your own unstated preference. If you \
genuinely believe a deviation is necessary (e.g. the specified approach is \
infeasible), say so explicitly in your final summary with your reasoning --\
never make an undisclosed departure from what was actually asked for.

qm8_data.py is already in your directory -- import it directly:
    from qm8_data import load_qm8
This returns QM8Molecule objects (smiles, atomic_numbers, positions, \
labels) with QM8's real original 3D coordinates and correctly-disambiguated \
labels already handled. Do not reimplement QM8 loading yourself.

Your deliverable MUST be runnable exactly as:
    python main.py --stage baseline
    python main.py --stage train
    python main.py --stage evaluate
This is a fixed contract -- the harness runs your code this way with no \
further reasoning about how to invoke it, so these three stages must exist \
under these exact names. "baseline" is a fast, cheap run (e.g. a tiny data \
subset) to catch bugs before the real run; "train" fits the model with a \
single, reasonable, hand-picked configuration -- do not run a hyperparameter \
search (grid/random search, k-fold CV, or multiple configs). Pick sensible \
values and train once; this keeps the real run's compute/time cost bounded \
and predictable, which matters more here than squeezing out extra accuracy.

"train" MUST use early stopping: track validation performance every epoch, \
and stop once it hasn't improved for a fixed patience window (e.g. 10-15 \
epochs), keeping the best checkpoint rather than the last one. Do not fix a \
large epoch count and run it unconditionally regardless of whether the model \
has already converged -- that wastes real compute/time for no benefit. This \
is a training safeguard, not a hyperparameter search: pick one reasonable \
patience value yourself, don't tune it.

"evaluate" MUST write results.json with your final metric(s), so the harness \
can read the outcome back -- that file existing is the one fixed \
requirement. Beyond that, implement your OWN diagnostic plan from the \
blueprint:
{diagnostic_plan or "(the blueprint did not specify one -- use your own judgment on what analysis, if any, is needed to understand this model's behavior beyond the headline metric)"}
Decide what files/fields that requires and implement it yourself -- this is \
your call about what's scientifically useful to measure, not a fixed format \
we're prescribing.

Write requirements.txt listing any package beyond what's already available \
(torch, torch_geometric, rdkit, pandas, numpy, scikit-learn are \
pre-installed -- list only genuinely additional packages, e.g. xgboost).

Use local_run to sanity-check your own code (does it parse, does \
`python main.py --stage baseline` complete without error) before finishing. \
This is a correctness check, not a tuning session -- once baseline runs \
without crashing and produces a plausible-looking metric, STOP iterating on \
it. Do not adjust hyperparameters, try alternative model classes, or \
otherwise polish it at this stage; that is out of scope for a sanity check \
and only costs turns you need for train/evaluate. A working, un-tuned \
baseline plus finished train/evaluate stages is the goal -- not a polished \
baseline with train/evaluate missing.

Only torch, torch_geometric, rdkit, pandas, numpy, and scikit-learn are \
actually installed right now -- anything you add to requirements.txt is NOT \
yet installed when you self-check (installation happens later, after the \
Reviewer approves your code). Do not try to pip install anything yourself; \
if your code needs a package beyond the six listed above, verify it with \
`python3 -c "import ast; ast.parse(open('main.py').read())"` (syntax only) \
rather than actually running it, and trust it will be installed before \
the real run.

STOP CONDITION -- read this carefully, it determines when you are done: \
once `python main.py --stage baseline` has run successfully ONE time \
without error, and main.py contains working train/evaluate logic, you are \
DONE. Do not re-run baseline again, do not test individual functions in \
isolation, do not re-read files you already wrote to double check them, do \
not verify error-handling edge cases. Every one of those is a real thing a \
careful engineer might do with unlimited time, but you do not have \
unlimited turns -- one successful baseline run is sufficient evidence the \
code works. The moment it succeeds, immediately respond with a short \
summary and no further tool calls. Continuing to verify past that point has \
directly caused past attempts to run out of turns before finishing.

When your implementation is complete and self-checked, respond with a \
short summary and no further tool calls."""

    if reviewer_notes:
        system_prompt += (
            f"\n\nThe Reviewer found problems with your previous attempt:\n{reviewer_notes}\n"
            "Fix these specifically -- don't restart from scratch."
        )

    messages = [{"role": "system", "content": system_prompt}]

    run_tool_loop(
        model=SWE_MODEL,
        temperature=SWE_TEMP,
        messages=messages,
        tools=TOOLS,
        dispatch={
            "write_file": lambda path, content: _write_file(candidate_dir, path, content),
            "read_file": lambda path: _read_file(candidate_dir, path),
            "local_run": lambda command: _local_run(candidate_dir, command),
        },
        online=True,
        trace=trace,
        stage=f"swe_candidate_{candidate_number}",
        max_turns=SWE_MAX_TOOL_TURNS,
    )

    return candidate_dir
