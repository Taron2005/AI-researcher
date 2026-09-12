# Architecture — QM8 Agentic Research Harness

This is the complete technical picture of the system: every role, every tool,
every input/output, every loop and cap. For *why* each choice was made (what
was rejected, what evidence was checked), see [DECISIONS.md](DECISIONS.md).
For the standing rules this project follows, see [CLAUDE.md](CLAUDE.md). This
file only answers "what exists and how does it fit together."

## The one-sentence version

A plain Python orchestrator runs three LLM-backed roles as **sequential
stages** (never as nested tool-calls of each other) plus a handful of
**deterministic, non-LLM tools** — the roles think and write, the tools
execute and report facts back.

## Is anything "a tool" for anything else?

No agent is a tool that another agent calls. There is no nested
LLM-calls-LLM relationship anywhere in this system. The three roles —
Planner, Software Engineer, Reviewer — are separate stages that our own
`orchestrator.py` drives with plain Python control flow (function calls,
loops, if-statements). The orchestrator calls one role, gets a result
(a file, or a verdict), and decides in code what to do next.

"Tools" in the strict sense (something an LLM calls mid-conversation via
OpenRouter's function-calling) only exist *inside* one role's own turn.
Web search is a partial exception, explained right after this list: it's
not a custom function-calling tool, it's OpenRouter's built-in `:online`
search enabled per-call — see DECISIONS.md for why.

- Planner can use web search (native `:online`) and call `search_papers`,
  `python_sandbox` — the latter two are real function-calling tools *of
  the Planner*, invoked by its own reasoning.
- Software Engineer can call `read_file`/`write_file`, `local_run`, and
  use web search (native `:online`, sparingly).
- Reviewer has no tools at all — no write, no execution, no search, and (a
  deliberate simplification made once it was actually built; see
  DECISIONS.md) no `read_file` tool loop either. The orchestrator reads
  every file in the candidate's directory itself and hands them to the
  Reviewer in one prompt — a candidate's whole project is small enough to
  include outright, and the Reviewer looks at everything anyway, so a
  selective-fetch tool loop would only cost turns, not add capability.

None of these tools belong to more than one role, and none of the three
roles is itself in another role's tool list.

## How many LLMs, exactly

**Three distinct role identities** (three different system prompts, three
different models). Some are invoked more than once per run:

| Role | Model | Invoked when | Times per run |
|---|---|---|---|
| Planner | `google/gemini-3.5-flash-lite` | (1) blueprint drafting — fully specifies candidate 1 only, plus a strategy note for candidate 2, (2) after candidate 1's real result, designing candidate 2's actual specifics if warranted (not just a yes/no vote), (3) final write-up | 1 + up to 1 + 1 → **up to 3** (capped at `MAX_CANDIDATES` = 2 candidates total either way) |
| Software Engineer | `deepseek/deepseek-v4-pro` | writing/fixing one candidate's code | up to `MAX_REVIEW_FIX_ROUNDS` (5) **per candidate** → up to 10 total |
| Reviewer | `openai/gpt-5-mini` | gating one Software Engineer draft | same cadence as Software Engineer → up to 10 total |

Worst-case ceiling for one full run: **≈23 LLM calls** (1 blueprint + 2×10
build/review rounds + 1 candidate decision + 1 write-up). In practice far
fewer — most drafts should pass review well before the 5-round cap, and the
Planner may stop after candidate 1 if it's clearly good enough.

Two of the three Planner calls use tools and the draft temperature (0.6):
blueprint drafting, and the candidate-2 design call — both are genuine
design work, drawing on literature search and dataset/result inspection.
Only the final write-up (temperature 0.2) is a plain completion — by then
the orchestrator hands it everything it needs directly in the prompt (every
candidate's real result, the blueprint, a trace summary), so there's
nothing left to look up, only to synthesize faithfully.

## Full pipeline, step by step

```
 1. Orchestrator starts a run, opens traces/run_<id>.jsonl

 2. PLANNER (blueprint drafting, temp 0.6)
    in:  QM8 task description + the 3 background docs
    tools: native `:online` search, search_papers, python_sandbox
    out: workspace/blueprint.md  (prose: candidate 1's full approach,
         sources, evaluation protocol, validation checkpoint, budget
         split, and a strategy note for candidate 2 — conditions under
         which it's worth trying and what kind of thing it might be,
         NOT a fixed second architecture)
         workspace/blueprint.json (same facts as structured fields the
         orchestrator can check in code — candidate 1's spec, the
         candidate-2 strategy note, explicit constraints list, etc.)

 3. Candidate 1 always runs. Candidate 2 is designed from candidate 1's
    real result, not pre-specified (see 3e) — so this isn't a `for`
    loop over a fixed list, it's candidate 1, then a decision, then
    maybe candidate 2:

    3a. SOFTWARE ENGINEER  (temp 0.2)
        in:  this candidate's description + blueprint.json's constraints
             + (on a retry) the Reviewer's specific fix notes
        tools: read_file/write_file (scoped to workspace/candidate_N/),
               local_run (quick syntax/logic self-check, NOT the real
               training run), native `:online` search (sparingly)
        out: workspace/candidate_N/*.py, requirements.txt

    3b. REVIEWER  (temp 0.2)   — mandatory gate, cannot be skipped
        in:  every file in workspace/candidate_N/ (read directly by the
             orchestrator, handed to the Reviewer in one prompt — not a
             tool call) + blueprint.json's constraint checklist + a
             STANDING checklist that applies to every candidate regardless
             of the blueprint (fixed CLI contract, real hyperparameter
             tuning present, correct use of the provided QM8 loader,
             results.json written, requirements.txt sane) — fixed
             requirements shouldn't depend on the Planner remembering to
             restate them per candidate
        tools: none
        out: PASS, or FAIL + specific notes citing which checklist item(s) failed

        FAIL and under MAX_REVIEW_FIX_ROUNDS (5)?
            -> back to 3a with the notes attached
        FAIL and round cap reached?
            -> candidate marked failed, move to next candidate (or to
               step 5 if this was the last one)
        PASS -> continue:

    3c. INSTALL  (deterministic, no LLM)
        batch `pip install -r requirements.txt`, targeting the pinned
        Kaggle image. A real version conflict's resolver error goes back
        into 3a/3b, not to a new agent.

    3d. KAGGLE EXECUTE  (deterministic, no LLM)
        push workspace/candidate_N/ to Kaggle, run it as
        `python main.py --stage {baseline,train,evaluate}` (the fixed
        CLI contract — no LLM ever decides how to invoke the code),
        poll, pull back stdout/stderr/traceback/metrics.

        Failure? -> raw traceback goes back into 3b with an added
                    "is the code wrong or is the check wrong" item,
                    then 3a. Same retry cap as above.
        Success? -> record the real metric, continue.

    3e. PLANNER (candidate-2 design, temp 0.6) — only runs after
        candidate 1; skipped entirely once a real candidate 2 has been
        attempted, since MAX_CANDIDATES (2) is reached
        in:  candidate 1's real Kaggle result (not just the headline
             metric — see tools below) + blueprint.md's strategy note
             + the original literature/sources from step 2
        tools: native `:online` search, search_papers, python_sandbox — the last one
             now with READ-ONLY access to candidate 1's actual result
             files in workspace/ (predictions, per-molecule errors,
             metrics), so this decision can be grounded in real error
             patterns (e.g. does error correlate with molecule size,
             is the baseline already saturating) instead of just a
             single number. Still no write access to workspace/ — that
             stays the Software Engineer's alone.
        out: either "stop here, candidate 1 is the final answer," or a
             new workspace/blueprint.json entry fully specifying
             candidate 2's actual architecture/approach, informed by
             what candidate 1's real result showed
             -> if a candidate 2 was specified, back to 3a for it

 4. PLANNER (final write-up, temp 0.2)
    in:  blueprint + every attempted candidate's real result + a trace
         summary
    tools: none
    out: writeup/report.md — what was tried, what won, what was
         rejected and why, compared against real published QM8 numbers

 5. Trace file is complete. Run done.
```

## Deterministic (non-LLM) components

| Component | What it does | Why it isn't an LLM agent |
|---|---|---|
| `install` | Batch-installs a candidate's `requirements.txt` | A package resolver's constraint-solving beats an LLM guessing compatible versions (DECISIONS.md) |
| `kaggle_execute` | Pushes/polls/pulls a Kaggle kernel run via the fixed CLI contract | Removes the ambiguity an LLM would otherwise have to resolve about *how* to invoke unknown code |
| Budget/retry counters | Tracks candidate count and review-fix rounds | External state an LLM can't forget (Deep Thought failure #6) |
| `trace.py` | Appends one JSON line per event to the run's log file | Every LLM call and every deterministic step logs here — this is the "run traces" deliverable |

## Tool inventory (who has what)

| Tool | Owner | What it does | Touches `workspace/`? |
|---|---|---|---|
| native `:online` search | Planner, Software Engineer | OpenRouter's built-in web search (per-call model setting, not a function-calling tool — see DECISIONS.md) | No |
| `search_papers` | Planner only | arXiv + Semantic Scholar direct API lookups | No |
| `python_sandbox` | Planner only | Exploratory-only code execution — dataset stats, loader sanity checks, a tiny throwaway baseline fit (blueprint drafting); analyzing a completed candidate's real results/errors (candidate-2 design) | Read-only, and only during candidate-2 design (to inspect candidate 1's results). Never writes to `workspace/` — that stays the Software Engineer's alone |
| `read_file` / `write_file` | Software Engineer only (read+write) | File access scoped to the current candidate's folder. Reviewer has no tool at all — the orchestrator reads every file itself and includes them directly in the Reviewer's prompt | Yes |
| `local_run` | Software Engineer only | Quick local syntax/logic self-check (e.g. does it parse, does a tiny dry run complete) — cheap, not the real GPU training run | Yes (runs the candidate's own files) |

## Artifacts — what's produced where, and who reads it next

| Artifact | Produced by | Consumed by |
|---|---|---|
| `workspace/blueprint.md`, `blueprint.json` | Planner (step 2) | Software Engineer, Reviewer, Planner's later calls |
| `workspace/candidate_N/*.py`, `requirements.txt` | Software Engineer (3a) | Reviewer, install, kaggle_execute |
| Reviewer verdict (pass/fail + notes) | Reviewer (3b) | Orchestrator (routing decision), Software Engineer (on fail) |
| Kaggle stdout/stderr/metrics | kaggle_execute (3d) | Reviewer (on failure), Planner's candidate-decision call, final write-up |
| `traces/run_<id>.jsonl` | Every stage, via `trace.py` | The "run traces" deliverable; also whoever debugs a run |
| `writeup/report.md` | Planner (step 4) | The "research write-up" deliverable |

## Budget/caps (all enforced by the orchestrator, none left to an LLM to self-track)

- `MAX_CANDIDATES = 2` — cheap baseline, then one 3D-aware GNN
- `MAX_REVIEW_FIX_ROUNDS = 5` — per candidate, across both the build↔review
  loop and the execute-failure↔review loop
- Kaggle time cap — TBD when `kaggle_exec.py` is actually built
