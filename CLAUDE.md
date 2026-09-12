# Project: QM8 Agentic Research (Deep Origin hiring challenge)

## Context

Task: build a team of AI agents that autonomously does ML research on the QM8 dataset
(quantum-chemistry property prediction — TDDFT/CC2 excitation energies and oscillator
strengths on ~22k small molecules, part of MoleculeNet). Deadline: 2026-09-14.

Deliverables to khsmbatyan@deeporigin.com:
1. Research write-up + code (the QM8 findings)
2. Agent harness code (the system that did the research)
3. Run traces (logs of what the agents actually did)
4. A line on what was done by hand vs. by the agents

Background docs already in this repo, read before proposing architecture/strategy changes:
- [do_challenge_notes.md](do_challenge_notes.md) — the source paper's benchmark, the four
  measured success factors (structure selection, spatial-relational nets, position
  non-invariance, strategic submitting), and how they do/don't transfer to QM8.
- [01-coscientist.md](01-coscientist.md) — the minimal single-planner/tool-call agent
  pattern (Boiko et al., Nature 2023) this harness's orchestration style is based on.
- [deep_thought_architecture.md](deep_thought_architecture.md) — the paper's own
  multi-agent system (roles, pipeline, 9 documented failure modes) this project's agent
  roles and guardrails are modeled on.
- [DECISIONS.md](DECISIONS.md) — the *why* behind every architecture choice below:
  which paper each role/tool came from, what was deliberately dropped, and what
  trade-offs were accepted knowingly. Check this before questioning why something is
  designed a certain way — it's probably already been reasoned through there.
- [ARCHITECTURE.md](ARCHITECTURE.md) — the complete *what/how*: every role, every
  tool and who owns it, full step-by-step pipeline, inputs/outputs, artifacts, and
  budget caps in one place. Read this before implementing a new piece, to see exactly
  where it fits and what it should and shouldn't have access to.

## Collaboration contract — how we work together

These rules exist because this is a hiring evaluation the user must be able to defend
line-by-line, not a vibe-coded demo. Follow them by default; don't ask permission to
follow them, only to deviate from them.

1. **No unilateral decisions.** Architecture choices, library/model/framework picks,
   anything with a real tradeoff — propose it with the alternatives considered and why,
   then wait for the user to choose. This applies to every stage (data handling, agent
   roles, modeling approach, evaluation), not just the big upfront ones. If something
   genuinely has no reasonable alternative (e.g. fixing a typo), just do it.
2. **Don't stop at the first answer.** When researching a method, library, or SOTA
   result, check more than one source and more than initial training-data recall —
   arXiv/Semantic Scholar for papers, official docs/GitHub for library capabilities.
   Cite what was actually checked (link or arXiv ID), especially for any claim that
   goes into the write-up as "state of the art" or "best practice."
3. **Readable over clever.** Small, single-purpose files. No file should need heavy
   scrolling to understand what it's for. Comments only explain non-obvious *why*
   (a constraint, a workaround, a subtlety) — never restate what the code already says.
   Every module should be understandable by the user without needing to ask "what does
   this do" — if a design alternative was rejected, say so briefly (in the write-up or
   a short note), not as a leftover commented-out block of code.
4. **No overengineering.** No framework, abstraction, or config knob that isn't earned
   by an actual current need. Prefer plain Python over a heavier orchestration
   framework where a loop and a few functions do the same job legibly (see harness
   architecture decision below). Add complexity only when a concrete requirement forces
   it, not for hypothetical future flexibility.
5. **Cost-conscious by default.** OpenRouter is the default LLM provider. Prefer
   cheap/open models for high-volume or low-difficulty agent calls (search
   summarization, boilerplate code, formatting); reserve stronger/pricier models for
   steps where correctness actually matters (final modeling decisions, the write-up).
   Model choice per agent role is itself a decision point under rule 1, not a default
   to assume.
6. **Every run must be inspectable.** Every agent step (prompt, tool call, output,
   tokens, latency, cost) is appended to a structured JSONL trace log — this is the
   "run traces" deliverable and the primary debugging tool. No agent state should live
   only in an LLM's own context if it's something a human needs to audit afterward
   (e.g. budget spent, decisions made) — externalize it, per Deep Thought failure #6.
7. **Secrets never in code or chat.** `OPENROUTER_API_KEY` and any other credential
   live in a gitignored `.env`, loaded at runtime. Never hardcoded, never echoed back
   in full in logs or conversation.
8. **Sandbox any model-generated code before executing it.** Per the Coscientist
   trust-boundary lesson: an LLM's code is untrusted input. It runs in an isolated
   environment (container or restricted subprocess) with the smallest package/network
   surface that still does the job; only stdout/stderr/tracebacks cross back into the
   agent's context.

## Current settled decisions (revisit only by explicit discussion)

- **Data/modeling library:** PyTorch Geometric, for its 3D-aware GNN baselines --
  SchNet, DimeNet++, PaiNN -- matching DO Challenge success factors 2 and 3.
  **Correction (2026-09-10):** this decision originally justified PyG by also
  claiming it has a "native QM8/MoleculeNet loader" -- that was never checked
  against PyG's real source and is false (PyG has QM7b and QM9, not QM8; its
  MoleculeNet class covers esol/freesolv/lipo/pcba/muv/hiv/bace/bbbp/tox21/
  toxcast/sider/clintox, not QM8 either). Found only by actually running the
  code. QM8 loading is solved separately now in `harness/qm8_data.py` -- see
  DECISIONS.md for the full investigation (DeepChem's real loader exists but
  drags in TensorFlow just to import; the fix downloads QM8's actual raw files
  directly and parses them with RDKit, verified correct empirically). PyG
  itself is still used for GNN modeling once data is loaded -- that part of
  the original decision holds.
- **Harness orchestration:** custom minimal loop (Coscientist-style: one planner,
  explicit tool calls, tool output fed back as a message), not a heavier framework
  (LangGraph/AutoGen/CrewAI) — the pipeline shape needed (plan → build → evaluate,
  capped retries) doesn't need a graph framework's machinery, and a plain loop keeps
  every step auditable per rule 3.
- **Run tracing:** JSONL structured log is the source of truth and the submitted
  artifact. Langfuse (free tier) may sit on top of the same events as a dev-time viewer
  — optional, not a submission dependency. W&B deferred until/unless the modeling stage
  actually needs training-run comparison plots.

### Agent roles (finalized 2026-09-06)

Pruned hard from Deep Thought's ~9-entity roster — see rationale below and the
failure-mode mapping further down.

- **Planner/Research agent** (Coscientist-style single agent + tools). Tools:
  OpenRouter's native `:online` web search (no custom tool file — see "Web search"
  decision below), `search_papers`, and a sandboxed Python tool for *exploratory-only* work
  (dataset statistics, confirming the PyG QM8 loader behaves as expected, a tiny
  sanity-check fit) — never the real training run, both for cost control and so this
  stage can't quietly become the coding stage. No separate Scientist agent (QM8's task
  isn't ambiguous enough to need a rephrasing step — its function, producing an
  explicit plan, is just this agent's own first output). No separate `DOCUMENTATION`
  tool (PyG/RDKit are mainstream, well-documented libraries, not an obscure DSL like
  Coscientist's ECL/SLL case — a generic web search covers it). Output: a written
  **blueprint file** (chosen approach + why, evaluation protocol, sources, a validation
  checkpoint before any expensive run, and how much budget goes to architecture
  exploration vs. tuning) — the artifact every later stage is held accountable to.
- **Software Engineer agent**: writes the actual project files (data loading,
  featurization, model, training, evaluation scripts) per the blueprint, plus a
  `requirements.txt` covering any package beyond the pinned base environment. Tools:
  file read/write, `run_shell`, `run_python`, OpenRouter's native `:online` search
  (sparingly, only for a genuinely unfamiliar API detail — enabled per-call, not a
  separate tool file).
- **Reviewer agent**: never writes code. A **mandatory orchestrator gate** — code
  cannot reach execution without a Reviewer pass (Deep Thought failure #5: optional
  collaboration doesn't reliably happen even when the role exists). Checks the
  dependency manifest against actual imports, an explicit checklist item pulled
  literally from the blueprint's stated constraints (failure #1: don't trust a vibe
  check for spec compliance), and general correctness/completeness. Re-invoked after
  any evaluation failure with an added "is the code wrong or is the check wrong" item
  (failure #7).
- **No dedicated Installer or Evaluation LLM agents.** Both are deterministic tools:
  - *Install*: the SWE agent's `requirements.txt` is installed in one batch call
    (`pip`/`uv`), not reactively per import error. A real version conflict is better
    resolved by `pip`/`uv`'s actual constraint solver than by an LLM guessing
    compatible versions from memory — the resolver's precise error goes back into the
    existing SWE↔Reviewer loop rather than needing a new agent identity.
  - *Evaluate*: a plain tool runs the script and returns raw stdout/stderr/traceback
    (Coscientist's proven pattern), which the Reviewer re-reads on failure rather than
    a separate Evaluation agent producing its own diagnosis.
  - *Base environment* (PyTorch + matching CUDA build, PyTorch Geometric's C++
    extensions, RDKit) is pinned and tested by hand, once, up front — not resolved live
    by any agent. This is the fragile part (CUDA/wheel-index matching); it's a fixed
    problem, not a dynamic one, and a bad fit for both an LLM and a cold resolver.
- **"Creativity Level" (GBS) is not real.** Confirmed by reading arXiv 2504.19912
  Appendix E directly: the paper defines it ("determines how creative or conservative
  the responses should be") but never specifies an implementation, and no LLM provider
  exposes creativity as an API parameter — only temperature does. We use two real
  levers per role: temperature (a number) and a short system-prompt stance sentence.
  No invented setting that doesn't correspond to anything an API actually accepts.
- **Context management**: no Observation Manager subsystem (that existed for Deep
  Thought's much longer 9-agent chains). Bounded turns per agent session; handoff
  between stages goes through a file (blueprint doc, current code tree, last eval
  result), not a replayed full chat history.

### Pipeline shape & execution (finalized 2026-09-06)

- **Three agent roles total**, one of them reused twice: Planner/Research opens the
  pipeline (produces `blueprint.md` + `blueprint.json`) and is invoked again at the end
  to write the final research write-up from the accumulated blueprint + results + trace
  summary — no separate fourth "Reporter" identity. Software Engineer and Reviewer run
  the build↔review loop in between.
- **Handoff between stages is a file, not an LLM-authored prompt.** The Planner does
  not write the next stage's prompt text itself — it produces `blueprint.md` (prose:
  approach, sources, evaluation protocol, validation checkpoint, budget split) and
  `blueprint.json` (the same key facts as structured fields the orchestrator can gate
  on mechanically, e.g. "baseline must run before further iteration" — externalizing
  failure-mode #3/#8/#9 checks as code, not as something an LLM has to remember to
  enforce on itself). The orchestrator's own fixed template combines this with each
  stage's system prompt.
- **Execution is always deterministic, no LLM-based execution fallback.** The Software
  Engineer agent's system prompt fixes its deliverable's interface up front (e.g. must
  be runnable as `python main.py --stage {baseline,train,evaluate}`). Given that
  contract there's no ambiguity for the runner to resolve, so it never needs an LLM to
  decide *how* to invoke the code — only the SWE agent reasons about its own code.
  Failures return raw stdout/stderr/traceback straight into the Reviewer→SWE loop.
- **Compute target: Kaggle**, not Colab, for the real training runs — chosen because
  Kaggle has a real API (`kaggle kernels push`/pull) the orchestrator can drive
  programmatically (submit, poll, pull results), keeping execution fully automated.
  Colab has no headless-execution API; using it would have meant a manual hand-off
  point instead. Needs a Kaggle API token in `.env` alongside `OPENROUTER_API_KEY`.
  Implication for later: the pinned base environment (PyTorch Geometric, RDKit) must
  target Kaggle's kernel image specifically (an install step inside the pushed
  script), since Kaggle doesn't ship these by default — exact mechanics TBD when the
  execution tool is actually built.

### Research/iteration loop (finalized 2026-09-06)

Single-shot "plan once, build until correct, run once" would be agentic *engineering*
of a predetermined plan, not agentic *research* — QM8 doesn't force strategic,
resource-aware behavior the way DO Challenge's label budget does, so it has to be
imposed deliberately (per [do_challenge_notes.md](do_challenge_notes.md) §5's closing
point). Checked this against real precedent before deciding, per rule 2:
**AIDE** (arXiv:[2502.13138](https://arxiv.org/pdf/2502.13138), WeCo AI — the scaffold
behind the top MLE-bench score) structures ML-engineering agents as Solution
Generator → Evaluator → Solution Selector over a small tree of candidates, not a
single fixed plan. An independent evaluation of Sakana AI's comparable "AI
Scientist-v2" (arXiv:[2502.14297](https://arxiv.org/abs/2502.14297)) found it
repeatedly got stuck cycling through broken code without fixing the actual bug —
this is Deep Thought's failure #7, and it's exactly why our mandatory Reviewer gate
and hard retry cap already exist; that part of the design is validated, not obsolete.

Adopted, without adding agents or new infrastructure:
- The Planner's blueprint fully specifies **candidate 1 only** (a cheap baseline),
  plus a brief strategy note — what kind of thing candidate 2 might be and under what
  conditions (e.g. "if the baseline's errors show position-sensitivity, prioritize a
  3D-aware GNN next") — not a second fixed architecture. Candidate 2's actual design
  is decided later, from real evidence (see below). This matters: fixing both
  candidates upfront, with the mid-run Planner call only voting yes/no on the second
  one, would be a static A/B test with an early-stop option — not the "propose a
  specific improvement based on feedback" mechanism AIDE was cited for adopting in the
  first place. Caught by the user re-reading this section on 2026-09-10: the original
  version of this bullet list had quietly regressed to the weaker design despite
  citing AIDE's adaptive one.
- The orchestrator tracks a **self-imposed experiment budget** (candidate count,
  Kaggle time cap) as external state — the deliberately-imposed constraint the notes
  called for, tracked as a counter per failure #6, not left to an LLM to remember.
- Each candidate reuses the existing SWE→Reviewer→Kaggle-execute loop as-is.
- The Planner is invoked a third time, after candidate 1's real result, to do the
  actual "Selector" work: decide whether a second candidate is warranted at all, and
  if so, design its specifics informed by candidate 1's real outcome (not just its
  headline metric — `python_sandbox` gets read-only access to candidate 1's actual
  results here, so the Planner can look at real error patterns, not just a number).
  Because this is genuine design work, not a judgment call, it uses the same tools
  (native `:online` search, `search_papers`, `python_sandbox`) and the same draft-style
  temperature (0.6, not the judgment 0.2) as the initial blueprint call — it's a
  second instance of the same "Planner drafts a plan" capability, not a lighter,
  tool-less variant.
- The final write-up (Planner, once more, judgment temperature 0.2) reports which
  candidates were tried and why — a decision trail, not just a final number.

### Failure-mode → mitigation mapping (Deep Thought's 9 failures, §5 of
[deep_thought_architecture.md](deep_thought_architecture.md))

| # | Failure | Our mitigation |
|---|---|---|
| 1 | Constraint acknowledged but violated | Reviewer checklist has a literal item pulled from the blueprint's stated constraints |
| 2 | Context collapse past ~20-50k tokens | Bounded sessions, file-based handoff between stages |
| 3 | Working solution never actually run | Orchestrator blocks further "improve it" rounds until one execution attempt has happened |
| 4 | No plan across multiple submissions | N/A for QM8 — no submission-budget mechanic |
| 5 | SWE skips the Reviewer | Reviewer is a mandatory orchestrator gate, not optional |
| 6 | Forgets its own budget | Any self-imposed budget is a counter in orchestrator code, not in a prompt |
| 7 | Infinite bug-fix loop / fixes the check instead | Hard retry cap; Reviewer explicitly asked "code wrong or check wrong" on re-review |
| 8 | Fancy architecture, no tuning budget | Blueprint states the architecture-exploration-vs-tuning split up front |
| 9 | Full budget spent before validation | Blueprint must specify a cheap validation checkpoint before the expensive run |

### Model assignment (finalized 2026-09-06)

Chosen by judgment (user explicitly opted to skip an empirical pilot given time
pressure), from real OpenRouter pricing (fetched live from
`https://openrouter.ai/api/v1/models`, not recalled from training data) plus the most
task-relevant benchmark evidence found: mlebench.com's current leaderboard (an actively
tracked, more credible source than several SEO-aggregator "2026 leaderboard" blog posts
that turned up first in search and were not trusted — those claimed numbers like
"Claude Sonnet 5: 95% SWE-bench Verified" with no verifiable primary source).

| Role | Model | Temp | Why |
|---|---|---|---|
| Planner/Research | `google/gemini-3.5-flash-lite` | 0.6 (blueprint drafting), 0.2 (candidate-selector decision, final write-up) | Deep Thought's real deployment used Gemini-Flash-tier for its whole research subgroup — reliable tool-calling at low cost, matching this role's actual job. Updated 2026-09-06 from `3.1-flash-lite` to the newest release still in the cheap "flash-lite" tier — `3.6`/`3.7`/`3.8-flash` are a full tier up (no flash-lite variant exists yet for those generations), 3x the price for no verified benefit to this role; `3.5-flash-lite` is a modest +20%/+67% (prompt/completion) over `3.1-flash-lite` for a newer generation, with identical (i.e. equally absent) benchmark coverage either way. |
| Software Engineer | `deepseek/deepseek-v4-pro` | 0.2 | Cheapest completion-token price of the real candidates (code-writing is completion-heavy); DeepSeek-R1 (same lineage, prior generation) placed 2nd on mlebench.com's actual ML-engineering leaderboard (36.4% medal rate) — the most task-relevant evidence found for any candidate. |
| Reviewer | `openai/gpt-5-mini` | 0.2 | Different provider family than the SWE model, deliberately — mirrors Deep Thought's actual config (different providers for SWE vs. Reviewer) so a model can't share its own blind spots with the code it's checking. Plain GPT-5 placed 3rd on mlebench.com (35.1%) — strongest evidence found for any GPT-5-tier model. |

**Caveat, tightened after direct verification (see
[DECISIONS.md](DECISIONS.md#benchmark-verification-for-model-selection--whats-actually-evidence-vs-inference)
for the full check):** none of the three exact model IDs above have a direct entry on
any benchmark actually verified so far (MLE-bench and Aider were fetched and read
directly; BFCL and Artificial Analysis couldn't be scraped — JS-rendered, needs manual
check). The supporting evidence is real but one level removed: it's for each model's
flagship/predecessor sibling (Gemini-3-Pro-Preview, deepseek-r1/Deepseek-V3.2-Speciale,
bare gpt-5), not for the specific mini/flash-lite/pro-tier variants chosen. This is a
reasoned judgment call resting on family-lineage inference plus verified pricing, not
a verified benchmark result for these exact models. Each model ID is a single config
constant specifically so swapping one out later, if it underperforms in practice, is
a one-line change, not a redesign.

## Open, not yet decided

- Project directory/file structure — not yet created; propose before writing any code.
- Exact mechanics of the Kaggle execution tool: how the orchestrator packages the SWE
  agent's multi-file project into a kernel push, what the Kaggle-image install step
  looks like, how results/artifacts/logs are pulled back into the local trace log.
