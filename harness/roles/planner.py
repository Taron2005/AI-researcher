"""
Planner/Research agent -- opens the pipeline (blueprint.md + blueprint.json),
returns mid-run to design candidate 2 from candidate 1's real result, and
closes the pipeline with the final write-up. See ARCHITECTURE.md for the
full picture of when each of these three functions gets called and by what.

Three functions, one shared identity, one shared parsing convention: the
Planner's tools never write files directly (it only has search_papers and
python_sandbox -- see ARCHITECTURE.md's tool-ownership table), so every
function here ends by extracting two fenced code blocks (```markdown and
```json) out of the model's final answer and writing them to disk itself.
That keeps "the LLM decides content, the orchestrator/role code decides
where it lands on disk" a strict rule, not just a convention.
"""

import json
import re
from pathlib import Path

from harness.config import PLANNER_MODEL, PLANNER_TEMP_DRAFT, PLANNER_TEMP_JUDGMENT
from harness.tool_loop import run_tool_loop
from harness.tools.python_sandbox import TOOL_SCHEMA as PYTHON_SANDBOX_SCHEMA
from harness.tools.python_sandbox import python_sandbox
from harness.tools.search_papers import TOOL_SCHEMA as SEARCH_PAPERS_SCHEMA
from harness.tools.search_papers import search_papers
from harness.trace import Trace

WORKSPACE = Path(__file__).parent.parent.parent / "workspace"

TOOLS = [SEARCH_PAPERS_SCHEMA, PYTHON_SANDBOX_SCHEMA]
DISPATCH = {"search_papers": search_papers, "python_sandbox": python_sandbox}

BLUEPRINT_JSON_SCHEMA = """{
  "candidate_1": {
    "name": "short name, e.g. 'ECFP fingerprints + gradient boosted trees'",
    "approach": "what it does and why, 2-4 sentences",
    "libraries": ["rdkit", "..."]
  },
  "constraints": ["specific, checkable requirement, e.g. 'must not discard 3D coordinates'"],
  "evaluation_protocol": {"split": "how train/val/test is split, and why", "metric": "e.g. 'MAE averaged across all QM8 properties'"},
  "diagnostic_plan": "YOUR OWN decision on what analysis, beyond the headline metric, is needed to actually understand this model's behavior and failure modes -- e.g. per-molecule error data, feature importance, error-vs-property-type breakdown, whatever you judge is scientifically warranted. Be specific enough that the Software Engineer can implement exactly what you decided, not generic.",
  "validation_checkpoint": "the cheap check to run before the expensive real run",
  "budget_split": "how much effort goes to architecture exploration vs. tuning",
  "candidate_2_strategy": {
    "condition": "under what result from candidate 1 a second candidate would be worth trying",
    "hypothesis": "what kind of approach it would likely be, at a high level -- NOT a fixed spec"
  },
  "sources": [{"title": "...", "url": "..."}]
}"""


def _extract_fenced_blocks(text: str) -> tuple[str, dict]:
    """
    Pulls the ```markdown and ```json fenced blocks out of the model's final
    answer. Raises clearly rather than silently writing an empty/partial
    blueprint -- every later stage reads from these files, so a silent
    failure here would surface confusingly far downstream instead. By the
    time this runs, `_validate_two_fenced_blocks` (passed to run_tool_loop)
    should already have forced a retry on a malformed response, so this
    raising is a last resort, not the primary defense.
    """
    md_match = re.search(r"```markdown\n(.*?)\n```", text, re.DOTALL)
    json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if not md_match or not json_match:
        raise ValueError(
            "Planner's final answer is missing a ```markdown or ```json fenced "
            "block -- cannot write blueprint files. Raw answer:\n" + text
        )
    return md_match.group(1), json.loads(json_match.group(1))


def _validate_two_fenced_blocks(text: str) -> str | None:
    """
    Passed to run_tool_loop as its `validate` callback -- see that function's
    docstring for why this exists (a real run once got a degenerate 2-token
    response instead of the required format; this forces a retry instead of
    crashing the whole pipeline on it, DECISIONS.md).
    """
    has_md = bool(re.search(r"```markdown\n(.*?)\n```", text, re.DOTALL))
    json_match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if not has_md or not json_match:
        return (
            "Your response did not contain the required format. You MUST "
            "respond with exactly two fenced code blocks: one ```markdown "
            "block and one ```json block, with no other text outside them. "
            "Please provide your complete final answer now, in that exact "
            "format."
        )
    try:
        parsed = json.loads(json_match.group(1))
    except json.JSONDecodeError as e:
        # Checking the block merely EXISTS isn't enough -- invalid JSON
        # inside it would otherwise pass this check and then crash later
        # in _extract_fenced_blocks' own json.loads(), uncaught.
        return f"Your ```json block was not valid JSON ({e}). Please provide your complete final answer again, as valid JSON."
    missing = [k for k in ("candidate_1", "constraints") if k not in parsed]
    if missing:
        return f"Your JSON is missing required key(s): {missing}. Please provide your complete final answer again, including them."
    return None


def _validate_one_json_block(text: str) -> str | None:
    """
    For the candidate-2 decision. Checks the block exists, is valid JSON,
    AND has the right keys for its own "proceed" value -- valid-but-wrong-
    shape JSON (e.g. missing "candidate_2" when proceed=true) would
    otherwise raise an uncaught KeyError downstream instead of retrying,
    same class of gap just fixed in reviewer.py.
    """
    match = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if not match:
        return (
            "Your response did not contain the required format. You MUST "
            "respond with exactly one fenced ```json block, with no other "
            "text outside it. Please provide your complete final answer "
            "now, in that exact format."
        )
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        return f"Your ```json block was not valid JSON ({e}). Please provide your complete final answer again, as valid JSON."
    if "proceed" not in parsed:
        return "Your JSON is missing the required 'proceed' key. Please provide your complete final answer again."
    if parsed["proceed"] and "candidate_2" not in parsed:
        return "You set proceed=true but omitted 'candidate_2'. Please provide your complete final answer again, including it."
    return None


def draft_blueprint(task_description: str, background_docs: str, run_id: str, trace: Trace) -> dict:
    """
    Stage 1 (ARCHITECTURE.md step 2). Writes
    workspace/run_<run_id>/blueprint.md and blueprint.json (run-scoped so a
    new run never overwrites a previous run's real blueprint -- same
    reasoning as candidate directories, DECISIONS.md). Returns the parsed
    blueprint.json content.
    """
    system_prompt = f"""You are the Planner/Research agent for a small AI research \
team studying the QM8 molecular property dataset (quantum chemistry: TDDFT/CC2 \
excitation energies and oscillator strengths on small organic molecules).

Be exploratory and thorough. Use search_papers and python_sandbox before \
committing to an approach -- do not rely on memory alone for anything you \
would call "state of the art" or "best practice"; cite what you actually \
found. Use python_sandbox to inspect the real QM8 dataset (check label \
distributions, confirm 3D coordinates are present) before proposing an \
approach -- do not propose blind. Load it with:
    from harness.qm8_data import load_qm8
    molecules = load_qm8()
This returns a list of QM8Molecule objects, each a plain dataclass with four \
fields: `smiles` (str), `atomic_numbers` (list[int]), `positions` \
(list of (x,y,z) tuples -- the real original 3D coordinates), and `labels` \
(a plain dict of 16 entries, e.g. `{{"E1-CC2": 0.43, "E2-CC2": 0.43, ...}}` -- \
NOT a numpy array or a fixed-order vector). Do not try \
torch_geometric.datasets.QM8 (does not exist) or deepchem (pulls in \
TensorFlow); this loader is the verified fix.

Your job: decide candidate 1's actual approach yourself, and specify it in \
full detail, plus a strategy note for a possible candidate 2 -- NOT a second \
fixed architecture; candidate 2, if it happens, is designed later from \
candidate 1's real result. This is your research plan -- the modeling \
approach, the evaluation methodology, and what you'll do with two attempts \
are your call, made from what you actually find, not a template to fill in.

Context you should weigh in making that call, not a rule to follow: this is \
a resource-constrained, two-attempt design (see budget_split below) -- \
QM8's labels are known to be 3D-structure-dependent, so a model that ignores \
spatial coordinates may be limited, but a cheap, well-tuned baseline can \
also beat an expensive, undertuned sophisticated model if the budget for \
tuning matters more than architecture choice. Decide what candidate 1 should \
be, and what makes a candidate 2 worth attempting, based on your own \
research and reasoning about this specific trade-off -- not a fixed formula.

When you are done, respond with NO further tool calls. Your final message \
must contain exactly two fenced code blocks:
1. A ```markdown block: the full blueprint in prose (approach, why, \
sources, evaluation protocol, validation checkpoint, budget split, \
candidate-2 strategy note).
2. A ```json block matching this schema exactly:
{BLUEPRINT_JSON_SCHEMA}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task:\n{task_description}\n\nBackground:\n{background_docs}"},
    ]

    final_text = run_tool_loop(
        model=PLANNER_MODEL,
        temperature=PLANNER_TEMP_DRAFT,
        messages=messages,
        tools=TOOLS,
        dispatch=DISPATCH,
        online=True,
        trace=trace,
        stage="planner_blueprint",
        validate=_validate_two_fenced_blocks,
    )

    blueprint_md, blueprint_json = _extract_fenced_blocks(final_text)

    run_dir = WORKSPACE / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "blueprint.md").write_text(blueprint_md)
    (run_dir / "blueprint.json").write_text(json.dumps(blueprint_json, indent=2))

    return blueprint_json


def design_candidate_2(blueprint: dict, candidate_1_result: dict, candidate_1_files: dict, trace: Trace) -> dict | None:
    """
    Stage 3e (ARCHITECTURE.md). Decides, from candidate 1's REAL result,
    whether a second candidate is warranted -- and if so, designs its
    actual specifics. Returns a candidate_2 dict (same shape as
    blueprint["candidate_1"], optionally with its own "diagnostic_plan") to
    merge into blueprint.json, or None if the Planner decides candidate 1
    is the final answer.

    `candidate_1_files` is {filename: content} -- whatever candidate 1's own
    Software Engineer decided to produce (per its own diagnostic_plan, not
    a fixed set we prescribe) -- passed through to python_sandbox's
    input_files so this decision can inspect real error patterns if such
    data exists, not just the headline metric. See python_sandbox.py's
    docstring for why this is the harness's job to populate, not the
    model's.
    """
    system_prompt = f"""You are the Planner/Research agent, returning after candidate \
1 actually ran on real data. Candidate 1's real result is below, and its actual \
files are readable by python_sandbox under these names: {list(candidate_1_files.keys())}.

The original blueprint's candidate-2 strategy note: {json.dumps(blueprint.get("candidate_2_strategy", {}))}

Decide, using whatever investigation you judge necessary (python_sandbox on \
candidate 1's actual files if that helps, search_papers if literature \
context would help), whether a second candidate is genuinely warranted. This \
is your call to make and your methodology to design -- don't just restate \
the headline metric as justification either way; ground your reasoning in \
whatever you actually find."""

    system_prompt += """

When done, respond with NO further tool calls. Your final message must \
contain exactly one fenced ```json block:
- If a second candidate is warranted: {"proceed": true, "candidate_2": {"name": ..., "approach": ..., "libraries": [...], "diagnostic_plan": "your own choice, may differ from candidate 1's"}, "reasoning": "why, grounded in the real result"}
- If candidate 1 is the final answer: {"proceed": false, "reasoning": "why not, grounded in the real result"}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Candidate 1 result:\n{json.dumps(candidate_1_result, indent=2)}\n\n"
                f"Files available to python_sandbox via input_files: {list(candidate_1_files.keys())}"
            ),
        },
    ]

    def python_sandbox_with_files(code: str) -> str:
        return python_sandbox(code, input_files=candidate_1_files)

    final_text = run_tool_loop(
        model=PLANNER_MODEL,
        temperature=PLANNER_TEMP_DRAFT,
        messages=messages,
        tools=TOOLS,
        dispatch={"search_papers": search_papers, "python_sandbox": python_sandbox_with_files},
        online=True,
        trace=trace,
        stage="planner_candidate_2",
        validate=_validate_one_json_block,
    )

    json_match = re.search(r"```json\n(.*?)\n```", final_text, re.DOTALL)
    if not json_match:
        raise ValueError("Planner's candidate-2 decision is missing a ```json block:\n" + final_text)
    decision = json.loads(json_match.group(1))

    return decision["candidate_2"] if decision.get("proceed") else None


def _validate_one_markdown_block(text: str) -> str | None:
    """Same idea as _validate_two_fenced_blocks, for the final report."""
    if re.search(r"```markdown\n(.*?)\n```", text, re.DOTALL):
        return None
    return (
        "Your response did not contain the required format. You MUST respond "
        "with exactly one fenced ```markdown block containing the full report, "
        "with no other text outside it. Please provide your complete final "
        "answer now, in that exact format."
    )


def write_final_report(
    blueprint: dict, results: list[dict], candidate_files: dict, run_id: str, trace: Trace,
) -> str:
    """
    Stage 4 (ARCHITECTURE.md). Has both search_papers (verify comparison
    claims -- rule 2, and DECISIONS.md's fabricated-citation incident) and
    python_sandbox (inspect whatever diagnostic files each candidate's
    Software Engineer actually decided to produce -- since that's now the
    Software Engineer's own choice, not a fixed filename we prescribe, this
    stage can't just assume e.g. "feature_importance.json" exists).

    `candidate_files` is {filename: content} across ALL candidates, with
    filenames prefixed by candidate number (candidate_1_predictions.csv,
    candidate_2_predictions.csv, ...) to avoid collisions -- passed through
    to python_sandbox's input_files the same way design_candidate_2 does it.
    """
    system_prompt = f"""You are the Planner/Research agent, writing the final \
research report. Report which candidates were tried, what won, what was \
rejected and why -- a decision trail, not just a final number.

Compare the result against real published QM8 baselines -- use search_papers \
to verify any specific number you cite (e.g. "MPNN reports MAE 0.0143 on \
QM8"). Do NOT state a benchmark comparison number from memory; if you can't \
verify a specific figure, say so explicitly rather than stating an \
unverified number as fact. Be honest about any candidate that failed or \
underperformed, and why, and about any caveat that makes a comparison less \
than perfectly apples-to-apples (e.g. a different number of target tasks).

Each candidate's Software Engineer decided its own diagnostic approach (see \
each candidate's "diagnostic_plan" in the blueprint) and produced whatever \
files it judged useful -- available to python_sandbox under these names: \
{list(candidate_files.keys())}. Use python_sandbox to actually load and \
inspect them (they were not pasted into this prompt directly -- some may be \
large). Decide yourself what's worth reporting from what you find -- a \
report that only restates the aggregate metric, when richer diagnostic data \
was available and unexamined, is incomplete.

When you are done, respond with NO further tool calls. Your final message \
must contain exactly one fenced ```markdown block containing the full report."""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Blueprint:\n{json.dumps(blueprint, indent=2)}\n\n"
                f"Results from every attempted candidate:\n{json.dumps(results, indent=2)}"
            ),
        },
    ]

    def python_sandbox_with_files(code: str) -> str:
        return python_sandbox(code, input_files=candidate_files)

    final_text = run_tool_loop(
        model=PLANNER_MODEL,
        temperature=PLANNER_TEMP_JUDGMENT,
        messages=messages,
        tools=[SEARCH_PAPERS_SCHEMA, PYTHON_SANDBOX_SCHEMA],
        dispatch={"search_papers": search_papers, "python_sandbox": python_sandbox_with_files},
        online=True,
        trace=trace,
        stage="planner_writeup",
        validate=_validate_one_markdown_block,
    )

    md_match = re.search(r"```markdown\n(.*?)\n```", final_text, re.DOTALL)
    if not md_match:
        raise ValueError("Planner's final report is missing a ```markdown block:\n" + final_text)
    report_md = md_match.group(1)

    writeup_dir = WORKSPACE.parent / "writeup" / f"run_{run_id}"
    writeup_dir.mkdir(parents=True, exist_ok=True)
    (writeup_dir / "report.md").write_text(report_md)

    return report_md
