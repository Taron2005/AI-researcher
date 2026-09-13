# Design decisions and rationale

This is the "why," not the "what" — [CLAUDE.md](CLAUDE.md) has the current settled
architecture as facts to follow; this file explains where each piece came from, what
was rejected, and what trade-offs were accepted knowingly. Written so it can be read
back later (including by someone else) without needing to reconstruct the reasoning
from chat history, and reusable as raw material for the final write-up's methodology
section.

Sources used: [01-coscientist.md](01-coscientist.md) (Boiko et al., Nature 2023 /
arXiv:2304.05332), [deep_thought_architecture.md](deep_thought_architecture.md) +
[do_challenge_notes.md](do_challenge_notes.md) (Smbatyan et al., arXiv:2504.19912 —
verified against the actual paper PDF, not just the condensed notes, for the GBS/model
roster details), and two systems found via targeted research once the initial design
felt too single-shot: AIDE (arXiv:[2502.13138](https://arxiv.org/pdf/2502.13138)) and
an independent evaluation of Sakana AI's AI Scientist-v2
(arXiv:[2502.14297](https://arxiv.org/abs/2502.14297)).

## Why mix two systems instead of adopting either one whole

Coscientist is a single Planner agent with four text tools — cheap, transparent, one
agent's entire reasoning visible in one transcript, but no review step and no
multi-role division of labor (see "What they did *not* do" in
[01-coscientist.md](01-coscientist.md): "No multi-agent debate/critic/tournament").
Deep Thought is the opposite: up to 9 named agents/sub-agents across research,
engineering, installation, and evaluation groups, with documented mixed results — its
own §5 lists 9 measured failure modes.

Neither was adopted whole. Coscientist doesn't have enough structure for a project
with a real "did the code actually work" question; Deep Thought has more structure
than this task's shape needs and — per its own failure-mode analysis — that extra
structure didn't reliably prevent problems (failure #5: agents skip the collaboration
even when the role exists; the 6-agent Research group was "almost never invoked" by
strong models "without much benefit" even when it was). So: take Coscientist's minimal,
transparent, tool-calling loop as the *mechanism*, and take only the specific Deep
Thought roles that its own failure-mode evidence says earn their cost.

**On "simpler might mean lower quality" (the concern raised before writing this
down):** the simplification isn't just a speed trade-off against the deadline, it's
also evidence-backed on its own terms — Deep Thought's failure #8 states "a simple,
well-tuned model beat most fancy, under-tuned ones" in their own results, and the
6-agent Research group added measurable benefit only for weaker models. So fewer,
well-executed steps isn't purely a risk we're accepting for speed — the papers'
own data points the same direction. The actual risk being accepted is narrower: if
QM8 modeling turns out to need some capability only a dropped role provided (e.g. a
dedicated docs-retrieval step, if PyTorch Geometric's API turns out less
self-documenting in practice than assumed), we'd have to add it back mid-project
rather than having it from day one. Judged acceptable because every dropped piece
below has a specific, stated reason, not a blanket "less is easier."

## Role-by-role lineage

| Our role | Adapted from | What was kept | What changed |
|---|---|---|---|
| **Planner/Research** | Coscientist's Planner (mechanism) + Deep Thought's optional Scientist agent (function) | Coscientist: one agent, explicit tool calls, tool output fed back as a message. Deep Thought's Scientist: turns the raw task into a concrete plan/"blueprint" before anything is built. | Merged into one role instead of two — Deep Thought treats "turn task into blueprint" as a separate optional agent; here it's just the Planner's own first output, since QM8's task isn't ambiguous enough to need a dedicated rephrasing step. Also reused a second and third time (mid-run candidate-comparison decision, final write-up) rather than being single-use. |
| **Software Engineer** | Deep Thought's Software Engineer agent, directly | The role and its job (write the actual project files) — this one has no Coscientist analogue; Coscientist's Planner writes its own code inline, there's no separate coding identity to compare against. | Scope narrowed: writes to a fixed CLI contract (`python main.py --stage ...`) so execution can stay fully deterministic (see below) — not part of Deep Thought's original description. |
| **Reviewer** | Deep Thought's Reviewer agent | Never writes code; reviews the SWE agent's output before it can proceed. | Made into a hard orchestrator gate rather than an optional call, specifically because Deep Thought's own failure #5 found optional collaboration doesn't reliably happen even when the role exists. This one **wasn't in the original plan** — see below. |

### On the Reviewer specifically

Not part of the first sketch of this pipeline. Added after reading Deep Thought's
failure-mode table and recognizing that failure #1 (a stated constraint gets
acknowledged then violated anyway — e.g. their documented case of agents reasoning
"this should be invariant" in direct contradiction of an explicit spec) and failure #5
(optional review doesn't happen) are both about the same missing piece: nothing was
checking the Software Engineer's output against the plan's stated constraints. Once
that was clear, this stopped being optional.

## What was deliberately dropped, and why

**From Coscientist:**
- `DOCUMENTATION` tool (a separate docs-retrieval step) — Coscientist needed this
  because OT-2/ECL were genuinely obscure APIs/DSLs the model had never seen. PyTorch
  Geometric and RDKit are mainstream, well-documented libraries; a plain web search
  covers this without a dedicated tool.
- `EXPERIMENT` (physical hardware/robot execution) — not applicable, QM8 has no lab
  hardware component.
- The manual "parse exactly one command per turn" text convention — this was a 2023
  workaround for chat completions with no native tool-calling. Modern function-calling
  APIs (structured JSON tool calls) solve this natively; no need to reproduce a text
  parser for a problem the API layer already handles.

**From Deep Thought:**
- **ML Engineer agent** (optional SWE advisor) — folded into the Planner's blueprint
  (architecture choice + rationale lives there) and the Reviewer's checklist, rather
  than kept as a separate advisory identity.
- **Installer agent** and **Evaluation agent** — replaced with deterministic tools.
  This is where Coscientist's own principle was applied *against* Deep Thought's
  design: Coscientist's stated rule is "LLM tools vs non-LLM tools — search/docs may
  use a model, code and hardware execution must not." Deep Thought's Installer/
  Evaluation agents are LLM-driven; given our dependency set is fixed in advance (not
  discovered live, since PyTorch Geometric + RDKit + a Kaggle target were already
  decided), a deterministic tool does this job at least as reliably and without an
  extra model call — and for real version conflicts specifically, a package resolver's
  constraint-solving is more trustworthy than an LLM guessing compatible versions from
  memory.
- **Research agent group** (Research Manager + Web Searcher + Assistant + Summarizer +
  Ranking + Critic — 6 sub-agents) — collapsed into two direct tools (`web_search`,
  `search_papers`) called straight by the Planner. Justified by the paper's own
  finding, not just a hunch: strong primary models "almost never invoked this group,"
  and weaker models that did invoke it saw "not much benefit."
- **Observation Manager / Context Window Management** — not built. This subsystem
  exists in Deep Thought because its longer 9-agent chains needed active compression.
  Our chain is 3 roles with bounded jobs; mitigated instead with capped session length
  and file-based handoff between stages (the blueprint doc, the code tree, the last
  result) — cheaper, and targeted at the same documented risk (failure #2: tool-calling
  degrades past ~20-50k tokens) without the extra subsystem.
- **Scientific Idea Tournament** (several idea-generator agents across providers/
  temperatures, blind-compared pairwise, Elo-ranked) — not adopted as such. Our
  candidate-comparison loop (below) is sequential and ranks candidates by their real
  Kaggle execution result, not by an LLM's blind pairwise judgment of un-executed ideas.
  Slower (one candidate at a time instead of parallel generation) but the ranking
  signal is ground truth instead of a model's guess — a deliberate trade of speed for
  reliability.
- **Generation Behavior Settings' "Creativity Level"** — the paper defines it in
  Appendix E but never specifies an implementation, and no LLM provider exposes
  creativity as a real parameter (only temperature is). Not reproduced as a fake
  setting; using temperature + a system-prompt stance sentence per role instead.
- **Provider Abstraction Layer** (their custom code for switching a session between
  OpenAI/Anthropic/Gemini mid-conversation) — not needed; OpenRouter's single API
  gives this for free.

## The candidate-comparison loop, and why 2

Not from either paper — added after re-examining whether "plan once, build until
correct, run once" would actually demonstrate *research* rather than *engineering a
predetermined plan* (QM8 doesn't force strategic behavior the way DO Challenge's label
budget does). Modeled on AIDE (the scaffold behind the top MLE-bench score): Solution
Generator → Evaluator → Solution Selector over a small set of candidates, not a single
fixed plan.

**Why exactly 2, and why it's a config value, not a hardcoded shape:** driven by
Kaggle's free-tier GPU quota and the Sep 14 deadline — not a belief that 2 is
architecturally correct. The candidate count is a number the orchestrator reads from
config, not a structural assumption baked into the loop logic, specifically so it can
become 3 or 4 later (more compute, more time, or just wanting a stronger comparison)
without changing how the loop itself works — the Planner→SWE→Reviewer→Execute→Planner
cycle is the same regardless of how many times it repeats.

## Tools — currently minimal, intentionally

Current tool list: OpenRouter's native `:online` web search (no custom file — see the
dedicated entry below), `search_papers` (arXiv + Semantic Scholar), a sandboxed
`python` tool for the Planner's exploratory-only work, file read/write,
`run_shell`/`run_python`, and a package-install tool. This is deliberately the minimum
that makes the current pipeline work, not a guess at what might be needed later —
per CLAUDE.md rule 4, a new tool gets added only when a concrete step in the pipeline
actually needs it, not preemptively.

## Benchmark verification for model selection — what's actually evidence vs. inference

The user asked directly whether the three model picks (`google/gemini-3.1-flash-lite`,
`deepseek/deepseek-v4-pro`, `openai/gpt-5-mini`) are benchmark-backed, and pushed for
exact sources, currency, and results rather than a general impression. Full honest
accounting of what was checked, 2026-09-06:

**MLE-bench** (github.com/openai/mle-bench official README — fetched and read
directly, not a search summary). Leaderboard is currently frozen ("not taking any new
submissions" as of 2026-04-24) with entries through 2026-03-06. **None of our three
exact model IDs appear in it.** What's actually there: `Gemini-3-Pro-Preview` dominates
the top of the table (six entries, 50.67–64.44%); `Deepseek-V3.2-Speciale` scored
56.44% (2025-12-16); bare `gpt-5` scored 35.11–39.56% and `gpt-5-codex` 48.44%;
`deepseek-r1` (older than V4) scored 29.33–36.44%. `Gemini-3-Flash-Preview` appears
once at 62.66%, but in the "Additional Submissions... not directly comparable"
section, not the main table. `gpt-5-mini` appears only as an undifferentiated
sub-component of one ensemble architecture's footnote — never as its own data point.
**Conclusion:** the earlier justification ("DeepSeek-R1 placed 2nd," "GPT-5 placed
3rd") was accurate as far as it went, but citing it as support for the *mini/flash-lite/
pro-tier-of-a-different-version* models actually chosen conflates "the flagship
sibling did well" with "this specific budget variant is verified" — that's a lineage
inference, not a direct benchmark result. There's a structural reason for the gap:
MLE-bench runs are 24-hour, real-GPU, multi-seed evaluations, expensive enough that
almost all submissions use flagship models — absence of budget-tier data here reflects
cost of testing, not measured weakness.

**Aider LLM Leaderboard** (aider.chat — fetched directly). Last updated 2025-11-20,
~10 months stale relative to now. Contains zero Gemini 3.x or DeepSeek V4 entries, and
no `gpt-5-mini` (only bare `gpt-5` at various reasoning efforts, 81.3–88.0%). No
usable evidence for any of the three picks.

**Berkeley Function-Calling Leaderboard** and **Artificial Analysis Intelligence
Index** — both are JavaScript-rendered pages; the fetch tool available here returns
the page shell, not the populated table (BFCL confirmed "Last Updated 2026-04-12" but
no scores extracted; Artificial Analysis confirmed 643 models in its database but
none of our three found in the retrievable excerpt). This is a tool limitation, not a
completed check — genuinely unresolved, left for manual verification in a real browser.

**Where this leaves the three picks:** not benchmark-verified at the exact model
version, on any source checked so far. They rest on family/lineage inference (the
named model's sibling/predecessor performs well) plus real, verified OpenRouter
pricing — not on a direct score for the specific IDs in use. This is a materially
weaker evidence standard than the CLAUDE.md model-assignment section's original
phrasing implied, and CLAUDE.md's caveat has been tightened to say so plainly.

## QM8 has no PyTorch Geometric loader -- a foundational claim was wrong (2026-09-10)

The very first settled decision in this project (data/modeling library) justified
PyTorch Geometric partly by claiming it has a "native QM8/MoleculeNet loader." That
claim was never actually verified against PyG's real source at the time -- it was
stated with confidence, accepted, and stood unchallenged for days. Found wrong only
because `python_sandbox` actually tried `from torch_geometric.datasets import QM8`
during a real Planner run and got `ImportError`. This is exactly the failure mode
rule 2 exists to prevent, and it happened at the single most foundational decision
point in the whole project. Worth naming plainly rather than quietly patching.

**Investigation, each step verified hands-on, not assumed:**
1. Checked `torch_geometric.datasets` directly: only `QM7b` and `QM9` exist as named
   classes. Checked `MoleculeNet.names`: `['esol', 'freesolv', 'lipo', 'pcba', 'muv',
   'hiv', 'bace', 'bbbp', 'tox21', 'toxcast', 'sider', 'clintox']` -- QM8 is absent
   from PyG entirely, confirmed by reading the installed package's own source
   (version 2.8.0.post1, current), not a stale-version issue.
2. DeepChem has a real `load_qm8()`. Considered using it (user's first choice), but
   `import deepchem` unconditionally pulls in TensorFlow (confirmed by trying it --
   `ModuleNotFoundError` cascades through `deepchem.hyper` -> `deepchem.trans` ->
   TensorFlow), a large, slow-to-install dependency for a project using none of it.
   Its default featurizer (`CoulombMatrix(26)`) also isn't raw 3D positions anyway --
   fetched DeepChem's actual `qm8_datasets.py` source from GitHub (no need to execute
   the package) to find the real answer without paying the import cost.
3. That source revealed the real raw files: `gdb8.tar.gz` (structures) and `qm8.csv`
   (labels), both on a public S3 bucket. Downloaded and inspected both directly.
   Verified empirically, not assumed: SDF molecule 0 is methane (5 atoms, formula
   CH4); its RDKit canonical SMILES ("C") exactly matches CSV row 0's SMILES field
   canonicalized the same way -- confirming a **plain positional join** (molecule N
   <-> label row N) is correct, with no id-based matching needed. Also confirmed the
   SDF's embedded conformer is read as-is by `Chem.SDMolSupplier` with no
   conformer-generation step -- these are QM8's real original 3D coordinates (the
   geometry the CC2/PBE0/CAM excitation-energy labels were actually computed on),
   not a re-embedded substitute like DeepChem's `RDKitConformerFeaturizer` would give
   (that one explicitly runs ETKDG to generate a *new* conformer -- would have silently
   broken the label-geometry correspondence, exactly the position-non-invariance trap
   [do_challenge_notes.md](do_challenge_notes.md) warns about, this time in the loading
   layer rather than the modeling choice).
4. Caught a second, separate real bug while inspecting the raw label file: its header
   contains a genuine duplicate -- `E1-PBE0,E2-PBE0,f1-PBE0,f2-PBE0` appears twice,
   verbatim, with different numeric values each time (confirmed by comparing actual
   row values, not just column names). Researched why: PBE0 is computed at two
   different basis sets, def2-SVP and def2-TZVP (confirmed against the original QM8
   paper's methodology, Ramakrishnan et al. 2015). A naive `csv.DictReader` read
   silently collapses duplicate keys to the last-seen value, which would have quietly
   discarded 4 real label columns with no error. Verified `pandas.read_csv` instead
   auto-suffixes the second occurrence (`E1-PBE0.1` etc.) by testing it directly against
   the real file -- this also matches the column naming convention used in published
   QM8 benchmark code, so it's the standard fix, not an improvised one.

**Result:** `harness/qm8_data.py` -- downloads and caches the raw files once, parses
with RDKit + pandas, no DeepChem/TensorFlow dependency. Loads 21,747 of 21,786
molecules (39 fail RDKit sanitization due to invalid valence states in the raw SDF --
a known issue with GDB-derived structures, ~0.18% of the dataset, logged explicitly
rather than silently dropped). PyTorch Geometric is still used downstream for GNN
modeling once data is loaded in this format -- only the loading claim was wrong, not
the choice to use PyG for modeling.

**Why this matters beyond just fixing a bug:** this was exactly the kind of "fixed,
solved-once" infrastructure problem CLAUDE.md's rule 8 already argues shouldn't be left
for an agent to rediscover live (same reasoning as pinning the base environment by
hand). Had this not been caught now, the Software Engineer agent would have hit the
same `ImportError` during a real, budget-capped run and had to burn review-fix rounds
rediscovering all four of the findings above under time pressure, instead of spending
that budget on actual modeling decisions.

## Workspace paths made run-scoped (2026-09-12)

Direct consequence of the "nothing gets deleted" policy above, caught while wiring
the orchestrator together: `blueprint.md`/`blueprint.json`, `candidate_N/`, and
`report.md` were all at fixed paths (`workspace/blueprint.md`,
`workspace/candidate_1/`, `writeup/report.md`) with no run identifier. A second
pipeline run would have silently overwritten the first run's real files -- exactly
what the no-delete policy was meant to prevent, just via overwrite instead of `rm`.
Fixed by threading `run_id` through `draft_blueprint`, `implement_candidate`, and
`write_final_report` so every run gets its own `workspace/run_<id>/` and
`writeup/run_<id>/` subtree.

## Full codebase review after the autonomy-shift run's bugs (2026-09-12)

User asked for a systematic pass over the whole codebase for the same *classes* of
bug just found (unhandled API response shapes, missing retries, resource leaks), not
just the two specific incidents. Went through every file; real findings, each fixed:

1. **Orphaned worker processes on timeout, confirmed live, not theoretical.**
   `subprocess.run(timeout=...)` only kills the direct child on timeout -- a child
   that spawns its own workers (e.g. scikit-learn's `n_jobs=-1` via joblib/loky)
   leaves them orphaned. Checked `ps aux` after the earlier timeout and found real
   `loky` worker processes still running at 35% CPU, 17+ minutes of accumulated CPU
   time, well after the parent had been killed. Affected every timed subprocess call
   in the codebase (`python_sandbox`, `local_run`, `install_dependencies`,
   `execute_candidate`'s train/evaluate) -- fixed once in a new shared
   `harness/tools/subprocess_utils.py` (`run_with_timeout`, using `start_new_session`
   + `os.killpg` to kill the whole process group) rather than patched per call site.
   Verified the fix works with a real test: a sandboxed snippet that spawns a
   `sleep 30` child was confirmed killed alongside the timed-out parent, not orphaned.
2. **`install_dependencies` had no timeout handling at all** -- a hung `pip install`
   would have raised `subprocess.TimeoutExpired` uncaught and crashed the pipeline.
   Never triggered in practice (installs have been fast) but a real latent gap.
3. **Non-atomic download/extraction in `qm8_data.py`.** A killed-mid-download process
   (has genuinely happened today) would leave a partial `gdb8.tar.gz` that a later
   run's `.exists()` check mistakes for complete, then fails or silently works with
   truncated data. Fixed with the standard pattern: download to a temp path + rename
   only on success, and a `.download_complete` marker file (not "does the target file
   exist") as the actual completion signal. Verified the migration path works
   against the real existing cache (harmless one-time re-extraction, marker created).
4. **`search_papers`'s per-source exception handling was too narrow** -- caught
   `requests.RequestException` only, but a source outage returning an HTML error page
   instead of XML/JSON fails at parsing (`xml.etree.ParseError`, `json.JSONDecodeError`),
   neither a `RequestException` subclass. Broadened to catch `Exception` per source
   (one source's failure, whatever kind, should never lose the other's real results).
5. **No retry for genuinely transient API errors** in `llm_client.py` -- confirmed the
   real `openai` SDK exception hierarchy first rather than guessing, then added retry
   (3 attempts, exponential backoff) for exactly `RateLimitError`, `APIConnectionError`,
   `InternalServerError` -- deliberately NOT the broader `APIStatusError`, which also
   covers permanent errors (401, 402 -- hit for real earlier) that retrying can't fix.
6. **`run_pipeline` had no top-level error handling** -- any unhandled exception
   (a new API error shape, a provider outage) would crash a multi-hour, real-money run
   with a bare traceback and no record in the trace of where it died. Added a
   top-level catch that logs a `pipeline_crashed` event (exception type + message)
   before re-raising -- doesn't handle the failure, guarantees the trace explains it.
7. **Valid JSON with the wrong keys, an uncaught `KeyError` waiting to happen.** The
   Reviewer's and `design_candidate_2`'s retry loops (added earlier) only checked for
   a fenced block and valid JSON *syntax* -- `{"result": "PASS"}` instead of
   `{"verdict": ..., "notes": ...}` would pass those checks and then crash on
   `verdict["verdict"]`. Same fix pattern extended to check required keys are
   present, not just that the JSON parses; applied to `reviewer.py`,
   `design_candidate_2`'s validator, and `draft_blueprint`'s validator (which had the
   identical gap: checked the block existed via regex, never checked it actually
   parsed before `_extract_fenced_blocks` later called `json.loads()` on it uncaught).

## Two real bugs from the autonomy-shift run, both root-caused not patched (2026-09-12)

The first run under the new autonomy design hit a genuine problem: candidate 1's
`train` stage timed out (confirmed: `TRAIN_TIMEOUT_SECONDS=1200`, ~1200s elapsed
between review-pass and execute-result), the Reviewer correctly diagnosed it and gave
a specific fix ("reduce N_HP_ITER to 4-6, N_HP_CV to 2"), but the NEXT round somehow
passed review on **completely unchanged code** (`N_HP_ITER=10, N_HP_CV=3` on disk,
verified directly) -- meaning it was about to repeat the exact same timeout.

**Root cause, found by live diagnostic, not guessed**: fetched the raw API response
for a real `deepseek-v4-pro` call and found it returns a **separate `reasoning`
field** distinct from `content`. The Software Engineer's "fix" attempt had
`completion_tokens=202` but empty `content` and no `tool_calls` -- it almost
certainly put its output entirely in `reasoning` and never committed to an action
(stuck mid-thought). `run_tool_loop` only checked `tool_calls`/`content` and treated
empty-content-no-tool-calls as a legitimate "I'm done" signal, so the SWE's turn
silently did nothing, leaving the broken code in place.

This also explains something asked about directly: why two `reviewer` `llm_call`
events appeared back-to-back for the same round. The Reviewer hit the *same*
empty-response glitch once, but its own local retry loop (added earlier for a
different reason -- invalid JSON) happened to also catch this, because an empty
response fails its "find a JSON block" regex too. That's why the Reviewer self-healed
and the Software Engineer didn't: only the Reviewer had a check that incidentally
covered this case.

**Fixes, both minimal, no new abstractions:**
1. `run_tool_loop` (`harness/tool_loop.py`) now treats an empty `content` with no
   tool calls as an error requiring retry, using the exact same retry mechanism
   already built for `validate` -- one more condition, not a new mechanism. This
   fixes it for every caller (Software Engineer, all three Planner functions) at
   once, not per-caller.
2. `orchestrator.py`'s `_build_review_execute_loop` no longer clears
   `execution_error` when a review fails (it was being reset unconditionally on
   every review FAIL). The problem: if the Software Engineer's fix attempt does
   nothing real (exactly what the bug above caused), the *next* review has no idea a
   runtime failure ever happened and may pass the same still-broken code. Now
   `execution_error` persists across failed review rounds until execution is
   actually re-attempted, one line removed (`, execution_error = notes, None` ->
   `reviewer_notes = notes`), not an added mechanism.

Also added: `execute_result` trace events now include `output_preview` (the actual
failure text, truncated) -- previously only `success`/`stage_failed` were logged, so
diagnosing the timeout required inferring it from elapsed-time arithmetic between
events instead of just reading what happened.

## Autonomy shift: the model decides what research to do, we stopped prescribing content (2026-09-12)

Direct user pushback on the three depth upgrades below (which is now the "before" state
this entry describes changing): "I want the model to do its research and make
decisions, not us tell him what to do." Fair, and consistent with the honest answer
given when asked -- these fixes, and really the whole pipeline since the beginning,
had concentrated more decision-making in prescriptive prompts than in genuine agent
judgment. Reworked the three Planner-facing functions to separate two different kinds
of requirement that had been blurred together:

- **Structural/interface contracts** (kept, because the orchestrator mechanically
  needs them to function): the fixed CLI (`--stage baseline/train/evaluate`), the
  requirement that SOME results file exists so the harness knows the run succeeded,
  real hyperparameter tuning (an engineering-quality bar, not a research-direction
  choice), using the verified `load_qm8()` loader (solved infrastructure, not a
  research question).
- **Content/methodology prescriptions** (removed): "candidate 1 MUST be classical ML,"
  the exact `predictions.csv`/`feature_importance.json`/`error_analysis` file
  requirements, `design_candidate_2`'s required use of `python_sandbox` on a
  specifically-named file, the specific published MPNN number fed to it as context.

**What replaced them**: a new `diagnostic_plan` field in `blueprint.json` -- the
Planner's OWN decision about what analysis is needed to understand the model's
behavior, specific enough for the Software Engineer to implement, but decided by the
Planner, not by us. The Software Engineer's evaluate-stage instructions now say
"results.json must exist; beyond that, implement your own diagnostic_plan" instead of
naming exact files. `design_candidate_2` now says "decide, using whatever
investigation you judge necessary" instead of "you MUST use python_sandbox on
predictions.csv." Candidate 2 can specify its own `diagnostic_plan`, independent of
candidate 1's (a GNN might reasonably want different diagnostics than a tree model).

**Consequence this forced**: since the Software Engineer's diagnostic output format is
now its own choice, `write_final_report` could no longer assume fixed filenames like
`feature_importance.json` existed to read directly. Gave it `python_sandbox` access
too (previously only `search_papers`) so it can inspect whatever files each candidate
actually produced, the same way `design_candidate_2` already does -- replaced the
old `_read_feature_importance` (fixed filename) with a general `_read_candidate_files`
helper (reads back everything, excluding pre-seeded infrastructure and binary files,
which would corrupt if round-tripped as text through python_sandbox's input_files).

**Also fixed in the same pass** (found while the old prescriptive version's background
run crashed): `review_candidate` had zero retry-on-bad-format handling and crashed
outright on a real `json.JSONDecodeError` -- same class of bug as the Planner's
fenced-block issue fixed earlier, same fix applied (a small local retry loop, since
the Reviewer has no tools and doesn't go through `run_tool_loop`).

**Known trade-off, stated plainly**: this is a deliberate bet that Gemini-3.5-flash-
lite and DeepSeek-v4-pro can design competent research methodology on their own,
which was explicitly flagged as a risk before making this change -- these are
cost-tier models, not frontier reasoning models, and a fully open-ended "decide what
to investigate" prompt is higher-variance than a prescribed checklist. Worth watching
the next real run closely for whether depth/quality holds up or regresses.

## Three depth upgrades: real error analysis, feature importance, evidence-based candidate-2 decision (2026-09-12)

User pushed back on the first successful run: "just training a model" isn't a research
finding on its own -- it's a competently-executed baseline, but the central claim
("2D features were enough, we didn't need 3D") rested on the *absence* of a failure
signal in an aggregate metric, not on an actual test of the hypothesis. Agreed, and
implemented three concrete upgrades rather than accepting the first result as final:

1. **Per-molecule data, not just aggregate metrics.** The Software Engineer's evaluate
   stage previously wrote only `results.json` (aggregate MAE/R² per property) -- with
   only that, nothing downstream (not `design_candidate_2`, not a human reading the
   report) can ask "does error correlate with molecule size" or "which properties fail
   together," only look at one number. Now required to also write `predictions.csv`
   (one row per test molecule: smiles, size, actual vs. predicted per property) and
   `feature_importance.json` (top 15 features by importance). `results.json` also
   gets a required `error_analysis` field computed from predictions.csv, not restated
   from the aggregate.
2. **`design_candidate_2` given real evidence and required to use it.** Previously
   decided from the aggregate metric alone (confirmed in the actual first run's trace:
   `tools=[]`, zero tool calls, decided directly from candidate_1_result). Now (a)
   REQUIRED to load predictions.csv via python_sandbox before deciding, not just
   read the summary number; (b) given the actual published comparison (MPNN's own
   0.0314 MAE on f2-CC2, from the same MoleculeNet table verified earlier) so it
   reasons against real evidence that even a 3D-aware model struggles on this property
   type -- framed as "this makes the question genuinely open, worth testing," not as
   a directive toward either outcome. The goal is a real evidence-based decision, not
   forcing candidate 2 to happen regardless of what the data says.
3. **`write_final_report` required to discuss error analysis and feature importance**,
   not just report the aggregate metric and stop -- orchestrator now also reads
   `feature_importance.json` (via a new `_read_feature_importance` helper) and passes
   it through, alongside `results.json`'s new `error_analysis` field.

Orchestrator also fixed to pass `.csv` files (not just `.py`/`.json`) into
`design_candidate_2`'s available file set -- `predictions.csv` would otherwise have
been silently excluded by the existing suffix filter.

## First successful full run, and a real fabricated benchmark claim caught in review (2026-09-12)

The fourth attempt at `run.py` completed cleanly end to end: Reviewer passed candidate 1
on the first try, execution succeeded first try, no candidate 2 attempted, total cost
$0.1446, 12.3 minutes wall time. Real result: mean test MAE 0.0123, mean R² 0.805
across all 16 QM8 properties (candidate 1: RDKit descriptors/Morgan fingerprints +
tuned gradient-boosted trees). Weakest properties were the "f2" oscillator strengths
(R² 0.53-0.59) -- worth noting since DO Challenge's own success factors flag 3D-awareness
as mattering most for exactly this kind of property, making the Planner's "skip
candidate 2" decision worth scrutinizing rather than accepting at face value.

Scrutinizing it surfaced a real problem: the write-up's benchmark comparison
("MPNN/SchNet on QM8 typically report MAE 0.010-0.014, source: MoleculeNet") was
checked against the actual paper (arXiv 1703.00564, downloaded and read directly,
Table 9) and found to be **not real** -- `write_final_report` had zero tools at that
point, so this specific, load-bearing citation came from the model's training-data
memory, not a real check. This is exactly the failure rule 2 exists to prevent, at the
one stage that hadn't been tool-equipped.

**Fix**: gave `write_final_report` `search_papers` access (via `run_tool_loop`,
matching the other two Planner functions) and told it explicitly not to state a
benchmark number from memory -- verify via `search_papers` or say it can't confirm one.
Regenerating with this fix produced an *honest* report: it tried `search_papers` twice,
couldn't find the specific number (arXiv/Semantic Scholar only return title/abstract
snippets, not full-text table data -- a real, understood ceiling on that tool, not a
bug), and correctly declined to state a number rather than guess. Real improvement
over fabricating one, even though it didn't find the actual figure.

Since the real figure was already found by hand (downloading and reading the full
paper), the final `report.md` for this run was patched manually with the verified
table (MPNN 0.0143, DTNN 0.0169, GC 0.0148, KRR 0.0195, our candidate 0.0123 --
**better than the paper's own MPNN result**) plus an important caveat stated plainly:
MoleculeNet's canonical QM8 benchmark uses 12 tasks, our loader correctly preserves
all 16 (the genuine duplicate-basis-set PBE0 columns), so the aggregate comparison
isn't perfectly apples-to-apples even though the per-task comparison (which does line
up) supports the same conclusion. This whole investigation is exactly the kind of
manual verification step that belongs in the "what was done by hand" line for the
final submission, not hidden as if the agents did it unassisted.

## Second run: degenerate model output, fixed with Coscientist's own documented pattern (2026-09-12)

Second full-run attempt failed differently: the Planner's blueprint call raised
`ValueError: missing fenced block` because the model's final response was two tokens
of garbage (" средиų") instead of the required format. Trace review showed this came
right after several turns of the Planner genuinely struggling with `QM8Molecule.labels`'
actual type -- it guessed numpy array (`.shape` -- wrong), then dict-like via
`.keys()` (wrong, it's a dataclass), then tried `np.array()`/`np.isnan()` on it
(wrong again) -- four wasted turns before finally discovering it's a plain dict, right
before the response degenerated.

Two fixes: (1) told the Planner's prompt the exact structure up front (`labels` is a
plain dict of 16 entries, explicitly not a numpy array) rather than making it
rediscover this through trial and error -- same "don't make an agent rediscover
solved infrastructure" principle as the QM8 loader itself. (2) The real fix for the
crash: added a `validate` callback to `run_tool_loop` (`harness/tool_loop.py`) that
checks the final response's format and, if wrong, feeds a corrective message back and
retries instead of returning bad text -- this is [01-coscientist.md](01-coscientist.md)'s
own documented pattern ("if the model emits [an invalid response], inject a message
telling it to follow the format"), which this project cited as a "decision to steal
later" back when the coscientist notes were first read, but hadn't actually
implemented until a real run hit exactly the failure it was meant to prevent. Applied
to all three Planner functions that require a specific output format
(`draft_blueprint`, `design_candidate_2` via the same `validate` mechanism;
`write_final_report` via a small local retry loop since it has no tools and doesn't
go through `run_tool_loop` at all).

## First full pipeline run crashed on over-verification, not a real bug (2026-09-12)

The first genuine end-to-end run (`run.py`, real everything) crashed with
`RuntimeError: Tool loop exceeded max_turns=20`. Full trace review showed the actual
work finished by turn 13 of 20: `main.py` written (turn 10), `requirements.txt`
written (turn 11), `python main.py --stage baseline` ran successfully with no errors
(turn 13). The remaining seven turns (14-20) were pure re-verification with zero
further edits -- testing individual functions in isolation, re-checking error-handling
behavior, re-reading files it had already written, running `ls -la` again -- the
model never recognized it was actually done. Same root cause as the earlier
"over-tuning" bug (SWE agent over-iterating instead of progressing), just manifesting
as excessive re-verification instead of hyperparameter tuning.

**Two separate fixes, not one:**
1. Added an explicit STOP CONDITION to the Software Engineer's prompt: once `--stage
   baseline` succeeds ONE time, stop immediately -- no re-running it, no testing
   functions in isolation, no re-reading already-written files. Named the actual
   failure mode directly in the prompt ("continuing to verify past that point has
   directly caused past attempts to run out of turns before finishing") rather than a
   generic "be efficient."
2. **Separately**, the orchestrator crashed the *entire pipeline* on this instead of
   treating it as a recoverable per-round failure -- a real robustness gap regardless
   of the prompt fix, since any hard failure in `implement_candidate` had the same
   problem. Fixed: `_build_review_execute_loop` now catches this specific
   `RuntimeError`, logs it, and retries with a nudge, respecting the existing
   `MAX_REVIEW_FIX_ROUNDS` cap -- consistent with how a review or execution failure is
   already handled, rather than a special case that bypasses the retry loop entirely.

## Local execution first, Kaggle deferred to an optional upgrade (2026-09-12)

With ~2 days left and Kaggle execution completely unbuilt and untested, treating it
as a hard prerequisite for a full pipeline run would put the single riskiest,
least-verified piece of the whole system on the critical path before anything else
gets validated end-to-end. Reconsidered given a fact that was true from the start but
under-weighted: **Kaggle's GPU is only actually needed for candidate 2 (a GNN)** --
candidate 1 (RDKit fingerprints + gradient-boosted trees) is classical ML, already
verified running locally on CPU in seconds.

**Decision**: build a plain local execute step (`python main.py --stage train` then
`--stage evaluate`, deterministic, no LLM) and run the entire pipeline end-to-end
locally first -- Planner, both SWE<->Reviewer loops, candidate-2 decision, write-up,
all for real, all today. Kaggle becomes an optional upgrade attempted only if time
remains, most valuable if candidate 2 turns out to need more compute than local CPU
can handle in a reasonable time. If Kaggle doesn't get built in time, local CPU
execution for both candidates is an honestly-documented choice in the final write-up,
not a failure -- far better than an unfinished system blocked on the riskiest piece.

The execute step is one function (`execute_candidate(candidate_dir) -> results`) so
swapping local for Kaggle later doesn't touch the orchestrator's own loop logic --
this was already the plan (ARCHITECTURE.md's kaggle_execute step), just implemented
against local compute first instead of Kaggle.

## Reviewer built without a tool loop, and a standing checklist added (2026-09-12)

ARCHITECTURE.md originally gave the Reviewer a `read_file` tool, matching the
Software Engineer's shape (an agent that fetches what it needs, turn by turn). Once
actually building it, this was simplified: a candidate's whole project is a
handful of KB (main.py, requirements.txt), and the Reviewer's job is to look at
everything anyway to do a real review, not selectively investigate. Giving it a
fetch tool would only add turns and cost, not capability -- so the orchestrator
reads every file in the candidate's directory directly (deterministic Python) and
includes them all in one prompt. Same principle already used to justify not giving
Installer/Evaluation their own agent identities: agentic control isn't free even
when nothing about the task calls for it.

Also added a **standing checklist** (`STANDING_CHECKLIST` in `reviewer.py`) —
fixed requirements that apply to every candidate regardless of what the blueprint
says: the CLI contract, real hyperparameter tuning present (not a single fixed
config, per the tuning decision above), correct use of the provided QM8 loader,
`results.json` written, a sane `requirements.txt`. These were already stated in the
Software Engineer's own system prompt, but Deep Thought failure #1 is exactly "an
agent acknowledges a stated constraint and then violates it anyway" -- the Reviewer
checking these explicitly, every time, doesn't depend on the Planner remembering to
restate them per candidate in `blueprint.json`.

## Hyperparameter tuning added to the train stage, and run-logging policy fixed (2026-09-12)

User caught two real gaps by reviewing the actual generated candidate 1 code and this
project's own testing habits:

1. **No tuning.** Candidate 1's `train` stage used one fixed, hand-picked
   configuration (`n_estimators=200, max_depth=5`, ...) -- no search at all, despite
   this project's own documented lesson (Deep Thought failure #8: an untuned fancy
   model loses to a tuned simple one). Given ~2 days left, chose the light option: the
   Software Engineer's system prompt now requires a small cross-validated search
   (e.g. `GridSearchCV`/`RandomizedSearchCV` over 3-5 configs) as part of `train`,
   not a separate tuning agent or infrastructure -- proportionate to the remaining
   time, still directly closes the documented gap. Noted for when the Reviewer role is
   built: this should be a standing checklist item Reviewer always checks, not
   something dependent on the Planner remembering to state it in blueprint.json,
   since it's a fixed requirement for every candidate, not candidate-specific.
**Reversed (2026-09-13), with ~1 day left to the deadline.** A real run's
Software Engineer designed a 5-config × 3-fold-CV × 60-epoch grid search for
`train` (15 full trainings) -- technically satisfying the requirement, but
costing ~3+ hours of real Kaggle GPU time for one candidate. With the deadline
this close, the user chose to prioritize *getting any complete result at all*
over a properly-tuned one: the tuning requirement is removed entirely, not
just capped. `train` now fits a single, reasonable, hand-picked configuration
-- `software_engineer.py`'s system prompt and `reviewer.py`'s
`STANDING_CHECKLIST` both updated to match (the checklist item is deleted, not
softened). This knowingly re-opens the exact risk item 1 above was written to
close (an untuned model may underperform a tuned one, Deep Thought failure #8)
-- accepted deliberately, under time pressure, not overlooked.

2. **Deleting test/debug runs was the wrong call.** Every dev-time test run today
   (blueprint drafts, SWE attempts, tool tests) was cleaned up after serving its
   immediate debugging purpose. User correctly pointed out this throws away exactly
   the raw material a write-up or the "run traces" deliverable should be able to draw
   on later. **Policy from now on: nothing gets deleted.** Every run's trace and
   generated code is kept, distinguishable by run ID, whether it's a debug test or a
   real pipeline run. Runs deleted before this point are genuinely gone.

## Copied qm8_data.py resolved its cache to the wrong directory (2026-09-11)

After the previous two fixes, a real SWE run got stuck repeatedly timing out on
`python3 main.py --stage baseline` without ever completing, spending many turns
checking `ls -lh data/qm8/` and inspecting `qm8_data.DATA_DIR` trying to diagnose why.
Root cause: `qm8_data.py`'s `DATA_DIR = Path(__file__).parent.parent / "data" / "qm8"`
is relative to wherever the file itself lives. That's correct at
`harness/qm8_data.py` (resolves to the project's real `data/qm8` cache), but this
file gets copied into `workspace/candidate_N/qm8_data.py` for Kaggle portability
(DECISIONS.md, QM8 loader section) -- from there the same relative computation
resolves to `workspace/data/qm8`, a different, empty location. Every candidate run
was silently trying to re-download the ~8.7MB archive and re-parse all 21,786 SDF
records from scratch (~20-30s) instead of reusing the cache this project already
built, which is almost certainly what was actually timing out inside the 60s
`local_run` window.

**Fixed**: `DATA_DIR` now checks a `QM8_DATA_DIR` environment variable first, falling
back to the `__file__`-relative default. `local_run`'s subprocess environment
(`harness/tools/shell.py`) sets `QM8_DATA_DIR` explicitly to the one real project-
level cache, so every candidate's self-check reuses it regardless of where its copy
of `qm8_data.py` physically lives. Kaggle execution (not yet built) will need its own
answer to this -- a fresh Kaggle kernel has no access to this local cache at all, so
that environment will download once per kernel run regardless; not a regression, just
a separate concern for `kaggle_exec.py` later.

## SWE agent over-iterated instead of progressing, found from real token counts (2026-09-10)

After fixing the venv issue below, the same test still hit `max_turns=15` -- but this
time doing real, legitimate-looking work: it got a genuinely working baseline running
(real MAE numbers, `load_qm8()` succeeding, 21,747 molecules loaded) and recovered
from one real traceback. The problem was scope, not confusion: it kept rewriting the
entire `main.py` to tune hyperparameters (`n_estimators`/`max_depth`, then tried
`HistGradientBoostingRegressor`) during what was only supposed to be a "does it run"
sanity check, never reaching the `train`/`evaluate` stages before running out of turns.

Checked actual prompt token counts across the run rather than guessing whether more
turns would help: 1,162 -> 18,721 by turn 15, with several individual completions in
the 4,000-8,700 token range (deepseek-v4-pro appears to produce quite verbose
reasoning/commentary around tool calls, at least in this observed run -- a real,
first-hand behavioral data point, not something found in any benchmark). At ~18.7k
tokens by turn 15, simply raising the turn cap to "fix" this would risk running
straight into Deep Thought's documented ~20-50k tool-calling degradation threshold
rather than avoiding it.

**Fixed the actual behavior, not just the budget:** the system prompt now explicitly
states self-checking is for correctness, not tuning -- stop iterating once baseline
runs without crashing, and prioritize finishing train/evaluate over polishing
baseline. `SWE_MAX_TOOL_TURNS` raised modestly (15 -> 20) as a buffer, not a fix on
its own.

## local_run ran against the wrong Python entirely (2026-09-10)

First real end-to-end test of `implement_candidate()` hit the `max_turns=15` cap
without finishing -- not stuck on the actual candidate implementation, but on
package availability confusion. `local_run` executes via `subprocess.run(command,
shell=True, ...)` with no explicit `env`, so bare `python3`/`pip3` in the SWE agent's
commands resolved to the **system** Python (`/usr/bin/python3`), not this project's
`.venv` -- meaning none of the pinned base packages (torch, torch_geometric, rdkit,
pandas) were visible to it at all. The agent burned its entire turn budget (~$0.19)
diagnosing "missing" packages that were actually installed, just in a different
Python than the one it was accidentally invoking, including trying
`pip3 install --break-system-packages` against the system interpreter.

**Fixed two ways:**
1. `local_run` and `install_dependencies` (`harness/tools/shell.py`) now build an
   explicit `env` with the venv's `bin/` prepended to `PATH` and `VIRTUAL_ENV` set --
   the same effect as `source .venv/bin/activate`, applied per-subprocess rather than
   relying on inherited shell state.
2. Added `scikit-learn` to the pinned base environment (`requirements.txt`) -- it's
   needed by nearly every classical-ML candidate, and its absence was part of what
   sent the agent down the self-install rabbit hole in the first place.
3. Software Engineer's system prompt now explicitly states which six packages are
   actually installed *during self-checks* versus what's in `requirements.txt` (not
   installed until after Reviewer approval) -- removes the ambiguity that caused the
   agent to reasonably but wastefully try installing things itself.

## Semantic Scholar client-side throttling added (2026-09-10)

The key-approval email states the 1 req/sec limit (cumulative across all endpoints)
is the caller's responsibility to respect -- it rejects bursts rather than smoothing
them. Confirmed our header format was already correct (tried x-api-key, Authorization
Bearer, and a query param -- identical 429 response body each time, ruling out a
formatting issue) before concluding the persistent 429s were likely from our own
rapid-fire diagnostic testing plus a possible longer IP-level cooldown from earlier
anonymous-tier testing today, not a code bug. Added a real module-level throttle
(`_MIN_SECONDS_BETWEEN_REQUESTS = 1.1`) in `search_papers.py` regardless, since it's
explicitly our responsibility per the email and costs nothing when the key isn't even
being rate-limited.

## "Cheap baseline" wasn't concrete enough, found by running the real Planner (2026-09-10)

First real end-to-end test of `draft_blueprint()` (not a mock, an actual OpenRouter
run) picked SchNet -- already a full 3D-aware GNN -- as candidate 1, with DimeNet++
(an even more sophisticated one) as the candidate-2 hypothesis. That's not the
progression this project already agreed to (a classical-ML baseline first, a 3D-aware
GNN second, motivated by evidence) -- the system prompt said "a cheap, fast-to-
implement baseline" without saying what that concretely rules out, and the model
filled the gap with "the standard well-known architecture for this data type," which
is a reasonable reading of vague wording, just not the one intended.

Fixed by making the prompt concrete: candidate 1 must be descriptors/fingerprints
(e.g. via RDKit) into a classical model (gradient-boosted trees or a small MLP),
explicitly not a GNN, explicitly framed as "candidate 2 is where a GNN belongs, if
the evidence says so." Treated as a bug fix against an already-settled design (the
progression was already decided), not a new architecture decision -- so fixed directly
rather than brought back for another round of discussion.

## arXiv query behavior, found by testing while building search_papers.py (2026-09-10)

Two real bugs caught by actually running the tool against a real query, not by
inspecting the code: (1) arXiv's API defaults to sorting by **submission date**, not
relevance — an unqualified query returned recent-but-irrelevant papers until
`sortBy=relevance&sortOrder=descending` was added explicitly; this isn't prominently
documented by arXiv itself. (2) Multi-word queries are matched as an **OR of
individual words**, not a phrase or semantic match, and arXiv has no way to weight a
rare term (e.g. "QM8") over a common one (e.g. "energy", "network") — so
`"QM8 excitation energy GNN"` returned Dark Energy Survey papers ahead of anything
QM8-related, while `"QM8 SchNet"` (two genuinely distinctive terms) returned the real
SchNet paper immediately. Fixed by adding the sort parameters and rewriting the tool's
own description (which the calling model reads) to steer toward 1-2 distinctive
keywords instead of a natural-language phrase — there's no way to fully engineer
around arXiv's matching being word-frequency-blind without building real semantic
re-ranking ourselves, which would be overkill for what this tool needs to do.

This is also a concrete, tested argument (not just the earlier general suggestion) for
getting a free Semantic Scholar API key: unlike arXiv, Semantic Scholar does real
embedding-based semantic search (the same keyword-vs-similarity distinction
[01-coscientist.md](01-coscientist.md) describes for their OT-2 docs search) — it
would likely handle a natural-language query like "QM8 excitation energy GNN" as well
or better than arXiv is designed to. Currently blocked on real testing by the same
anonymous 429 rate limit hit earlier in this project.

## Web search: OpenRouter's native `:online`, not a custom tool (2026-09-10)

Original plan was a `web_search.py` tool file backed by a third-party search API
(e.g. Tavily), matching `search_papers.py`'s shape — an explicit named function the
model chooses to call. Checked OpenRouter's own docs before building it and found
built-in web search: appending `:online` to a model ID (or a plugin parameter) enables
search grounding, using each model's own native provider capability where available
(Google, OpenAI, Anthropic, Perplexity — this covers our Planner's Gemini model
directly) or Exa's API for everything else (our SWE's DeepSeek falls here, at
$0.007/request, still billed through the same `OPENROUTER_API_KEY`). Responses come
back with structured `url_citation` annotations (URL, title, excerpt) already attached.

**Decision: use it, drop the custom tool file.** No new API key, no new code to write
or maintain, and citations arrive built-in rather than needing to be parsed or
constructed by us — directly useful for the "cite sources" requirement (rule 2).
Trade-off accepted knowingly: `web_search` stops being structurally identical to
`search_papers`/`python_sandbox` (an explicit function the model calls from a `tools`
list) and becomes a per-call model-string/parameter setting instead — a different
mechanism under the hood, invisible to the Planner's own reasoning either way.
`search_papers.py` and `python_sandbox.py` still need custom implementations — arXiv/
Semantic Scholar are structured academic APIs, not general web search, and sandboxed
execution obviously isn't something a search feature provides.

## Candidate 2 planned after candidate 1's real result, not upfront (2026-09-10)

Caught by the user reading [ARCHITECTURE.md](ARCHITECTURE.md) after it was written:
the original design had the Planner's blueprint specify *both* candidates upfront,
with the mid-run Planner call only voting yes/no on whether to run the second one.
That's weaker than it looked when written — it's a fixed A/B test with an early-stop
option, not real iteration. It also silently contradicted the precedent used to
justify having a candidate loop at all: AIDE (arXiv:2502.13138, cited above) proposes
each next solution *from the previous one's actual results*, not from a plan written
before any evidence existed. Citing AIDE to justify the loop's existence, then not
adopting its actual mechanism, is exactly the kind of gap this document exists to
catch — and this time the user caught it, not a paper or a benchmark check.

**Fix**: the blueprint now fully specifies only candidate 1, plus a brief strategy
note (what candidate 2 might be, under what conditions — not a fixed architecture).
The mid-run Planner call, informed by candidate 1's real Kaggle result, now designs
candidate 2's actual specifics rather than casting a yes/no vote. Concretely this
changed three things in CLAUDE.md's pipeline description: (1) this call now uses the
same tools as blueprint drafting (native `:online` search, `search_papers`,
`python_sandbox`) instead of none, since it's doing real design work; (2) it uses the draft temperature
(0.6) instead of the judgment temperature (0.2), for the same reason; (3)
`python_sandbox` gained read-only access to a completed candidate's actual result
files in `workspace/`, so this decision can be grounded in real error patterns
(e.g. does error correlate with molecule size, is the baseline already saturating)
rather than only the headline metric number.

No new agents, no new infrastructure — same Planner role, same invocation point in
the pipeline, just given the right inputs and capability to do what it was already
supposed to be doing.

## GLM-5.3-Flash investigation (2026-09-06) — considered and rejected for Software Engineer

User brought two chart images from Z.ai (GLM's maker) claiming strong GLM-5.3/
GLM-5.3-Flash performance across ~15 benchmarks (Terminal Bench, DeepSWE, NL2Repo,
CyberGym, AutomationBench, Agents' Last Exam, GDPval-AA, etc.), proposing it as a
much cheaper ($0.075/$0.250 per M vs. DeepSeek-V4-Pro's $0.657/$1.314) alternative for
Software Engineer. Investigated rather than accepted at face value:

- **Tier ambiguity**: one of the two images labels its strong column plain "GLM-5.3"
  (the $1.40/$4.40 flagship), not "GLM-5.3-Flash" — the same tier-confusion risk
  already caught once with `gemini-3.1-flash-lite` vs. `-flash`. If that chart is the
  flagship, it says nothing about the actual flash-tier candidate.
- **Source**: both images are Z.ai's own self-reported marketing material — no error
  bars/methodology shown (contrast MLE-bench's `±SEM` over multiple seeds), no
  independent corroboration found despite checking. Attempted to verify the one
  benchmark on the chart recognized as an independent project (Terminal-Bench,
  tbench.ai / github.com/laude-institute/terminal-bench) — blocked by the same
  JS-rendering limitation hit on BFCL and Artificial Analysis; the GitHub repo has no
  raw results file, only leaderboard-submission instructions. Also noted: the live
  Terminal-Bench site is already on v4.0, while the chart cites v2.1/v3.0 — an older
  snapshot even if it could be read.
- **Reading image 2 (the one that does isolate GLM-5.3-Flash) bar-by-bar, not by
  overall impression**: mixed, not dominant. 1st on GDPval-AA v2, close 2nd on
  AutomationBench — but 3rd–4th on Terminal Bench 2.1 and DeepSWE v1.1 (behind
  GPT-5.6 Terra and Gemini 3.7 Flash both times), and last of the models shown on
  Agents' Last Exam. A plausible-looking, non-cherry-perfect result, which is mildly
  reassuring about the chart's honesty, but doesn't establish an advantage for our
  specific coding task.

**Decision: kept `deepseek/deepseek-v4-pro`.** Not on a finding that GLM is worse —
no independent evidence either way was found — but because (a) DeepSeek's supporting
evidence (MLE-bench lineage) is the one piece in this entire investigation that's
independently verified and methodologically transparent, and (b) QM8's scale means
total token spend for the whole harness run is likely low-single-digit dollars either
way — the 9x per-token price ratio matters far less in absolute terms at this project's
scale than it would in production. Flagged as revisitable if DeepSeek underperforms
in practice, not treated as a closed question.

## Cost/quality philosophy

Every role-pruning decision above optimizes the same thing: fewer LLM calls per unit
of actual research progress, without dropping anything the papers' own evidence says
mattered. This is also why OpenRouter is the provider (§ CLAUDE.md rule 5) and why
model assignment per role — next decision — will weight cheap/fast models for
high-volume, low-difficulty calls and reserve stronger models for where correctness
is actually load-bearing (writing code, judging code, the final write-up).

## search_papers timeout raised from 15s to 25s

A real run (`traces/run_20260912_215046.jsonl`) hit a genuine arXiv `Read timed out`
on `requests.get(..., timeout=REQUEST_TIMEOUT_SECONDS)` — not a bug, just arXiv's
public API being slow under real load. The harness already degrades gracefully here
(`search_papers()`'s per-source try/except turns it into a `{"source": "arxiv",
"error": ...}` entry rather than crashing, and the Planner just retried its query on
the next turn and succeeded), so this wasn't blocking anything.

Raised anyway, from 15s to 25s: unlike the subprocess timeouts (`train`'s 20 minutes,
calibrated from watching real compute duration), 15s for a generic REST call was never
measured against anything — it was a guess. 25s reduces how often this specific
transient hiccup surfaces, at negligible cost: if a source is genuinely down rather
than just slow, the call fails either way, just a few seconds later, and the existing
per-source degradation still covers it. Applies to both `_search_arxiv()` and
`_search_semantic_scholar()`, which share the one constant.

## GNN-on-CPU is genuinely slow, and a total-failure report used to fabricate its cause

A full run (`traces/run_20260912_215046.jsonl`) gave two pieces of real evidence:

**Compute:** the Planner autonomously chose a SchNet-style GNN for candidate 1. Direct
measurement (timing the worst-case hyperparameter config on the full 17,397-molecule
training set) showed ~65-77s/epoch on this CPU-only laptop (confirmed no usable GPU:
Ryzen 5 3500U's Vega 8 Mobile is not on AMD's own current ROCm support matrix,
checked directly against `rocm.docs.amd.com`, not assumed). Extrapolated, one full
candidate realistically needs 2-6+ hours, not the 20-minute `TRAIN_TIMEOUT_SECONDS`
cap — round 2 hit that timeout for real. This is why the user is moving execution to
a GPU-equipped machine (their brother's computer) via a clean `git`
push — README.md added with setup/run commands for that handoff.

**A real, more interesting failure mode:** by round 5, the Software Engineer had
already adapted well to the round-2 timeout on its own initiative — it redesigned
candidate 1 from a fully-connected-graph/full-dataset approach down to a 100-molecule
CV subset with radius-cutoff graphs, a dramatically cheaper design. It crashed on a
real bug (`AttributeError: 'list' object has no attribute 'y'` in `_compute_stats`)
right as it was converging on something workable, and ran out of
`MAX_REVIEW_FIX_ROUNDS=5`.

Because candidate 1 never produced a result, `design_candidate_2()` was never invoked
(the orchestrator only calls it `if candidate_1_results is not None`) — the Planner's
own blueprint-stated fallback plan never got a chance, even though the final report
correctly noted this fallback existed.

Worse: `_build_review_execute_loop`'s final return, on exhausting all rounds, used to
be a bare `f"exceeded MAX_REVIEW_FIX_ROUNDS={MAX_REVIEW_FIX_ROUNDS}"` — discarding the
actual `execution_error`/`reviewer_notes` that were sitting right there in scope.
`write_final_report` only ever sees this string, never the trace, so with no real
cause attached it had nothing to report except a guess — and it guessed, confidently:
"the candidate failed... primarily due to environment-specific dependency edge cases,
data pipeline hurdles..., or runtime instability" — none of which is what actually
happened (a timeout, then a real AttributeError). Same class of problem as the earlier
fabricated-benchmark-citation incident, now for failure diagnosis instead of a SOTA
number.

**Fixed:** that final return now attaches whatever real `execution_error` or
`reviewer_notes` was last known before the round cap was hit, so `write_final_report`
has an actual cause to cite instead of inventing one. The candidate-2-never-triggered
gap and the round-cap size are known, separate issues — flagged, not fixed here (user
chose to prioritize the report-honesty fix first).

## Kaggle execution built (2026-09-12) — real GPU compute, `config.EXECUTION_BACKEND`

Given the above (2-6+ hours for a GNN on local CPU, ~2 days to the deadline), the
user chose to build Kaggle execution now rather than wait on an unknown-spec machine.
`harness/tools/kaggle_exec.py` is the result — same `(candidate_dir) ->
ExecutionResult` contract as `execute.py`'s local implementation, selected by one
config constant (`config.EXECUTION_BACKEND = "local" | "kaggle"`, defaulted to
`"kaggle"` per explicit user choice).

Two real constraints were discovered by verifying against the actual Kaggle API and
a real live account (not assumed from docs, per rule 2 — the docs themselves turned
out to be incomplete/wrong on some of this):

1. **A kernel takes one `code_file`, not a folder of files** (confirmed against the
   real `kernel-metadata.json` schema). Fix: concatenate the candidate's
   `qm8_data.py` + `main.py` into one self-contained script per push — a
   deterministic string transform in the harness, never the LLM's job, matching the
   existing principle that only code decides *how* to invoke a candidate.
2. **Each Kaggle kernel run is a fresh, independent remote filesystem.** Local
   execution runs `--stage train` and `--stage evaluate` as two subprocess calls
   sharing one disk; two separate Kaggle pushes would NOT share a filesystem, so
   `evaluate` would find no `model.pt` to load. Fix: run BOTH stages inside ONE
   kernel push — the generated script `exec()`s the candidate's own
   `--stage`-dispatch body twice in the same process, patching `sys.argv` before
   each pass, so files written during the "train" pass are still on disk (same
   process, same `/kaggle/working/`) for the "evaluate" pass to read. No Kaggle
   Datasets upload needed. This also means a Kaggle-executed candidate's failure
   can't always be attributed to specifically "train" or "evaluate" from the outside
   (`stage_failed="train_or_evaluate"` in that case) — the real traceback in the
   captured output is the actual answer, reported honestly rather than guessed.

**Verified live before trusting any of this**, using a real Kaggle account
(`taronbabayan`) rather than relying on public docs alone:
- Credentials in `.env` work against the real API (`kaggle kernels list --mine`
  returned real kernels from the account).
- `kernels status` prints `... has status "KernelWorkerStatus.COMPLETE"` — NOT the
  bare `"complete"` the public docs describe. Parsing uses a case-insensitive
  substring match ("complete"/"error"/"cancel") rather than an exact comparison,
  specifically because of this mismatch.
- `kernels output -p <dir>` downloads each file the script wrote (e.g.
  `results.json`) under its own name, plus a `<slug>.log` file that is a JSON array
  of `{stream_name, time, data}` entries — not plain text. `_parse_log` reconstructs
  readable stdout/stderr from it.
- The core cross-stage-persistence mechanism (exec main.py's dispatch twice with
  patched `sys.argv`, relying on the same-process/same-disk assumption) was tested
  end-to-end with a fast dummy candidate (a `train` stage writing a file, an
  `evaluate` stage reading it back and writing `results.json`) before ever pointing
  this at the real, multi-hour GNN candidate — confirmed working for real, not
  assumed to work from the design reasoning alone.
- That live test caught two more real bugs the design reasoning alone had missed:
  (a) the account needed Kaggle's own phone-number verification before
  `enable_internet`/`enable_gpu` did anything at all — without it, the flags are
  silently ignored and the kernel gets no network (`Temporary failure in name
  resolution` on `pip install`), a real external account requirement, not a bug in
  this code, fixed by the user verifying their phone on kaggle.com; (b) `main.py`'s
  body was originally `exec()`'d against a fresh, empty namespace
  (`{"__name__": "__main__"}`) instead of the outer script's own `globals()` --
  `qm8_data.py`'s definitions (e.g. `load_qm8`) live in that outer scope, so the
  fresh namespace couldn't see them, raising `NameError: name 'load_qm8' is not
  defined` inside `stage_train()`. Fixed by exec'ing against `globals()` directly
  (the outer script is already running as `__main__` on Kaggle, so `__name__` was
  already correct -- no fake namespace needed at all). Re-tested after each fix;
  the final version passed end-to-end for real (`train`→`evaluate` handoff via a
  real written/read-back file, `results.json` parsed correctly, real `pip install`
  over real internet).

**Kept out of `candidate_N/`**, to avoid polluting the Reviewer's re-review context
(`reviewer.py`'s `_read_candidate_files` does a recursive `rglob()` over the whole
candidate directory): the generated push script and the raw `.log` file live in
sibling directories (`candidate_N_kaggle_push/`, `candidate_N_kaggle_output/`), kept
for auditability (rule 6) but never swept into what the Reviewer or the final
write-up's `python_sandbox` sees. Only the candidate's own real output files (e.g.
`results.json`, whatever the diagnostic_plan produced) get copied into `candidate_N/`
itself.

Timeout: `RUN_TIMEOUT_SECONDS` raised to 6h (from an initial 3h) for headroom against
an ambitious Planner architecture choice — still comfortably under Kaggle's own ~9h
hard cap per kernel run.

## Kaggle assigns an incompatible GPU by default -- P100 vs the stock image's torch

First real pipeline run against the new Kaggle backend hit a genuine, reproducible
crash on the very first execution attempt: `torch.AcceleratorError: CUDA error: no
kernel image is available for execution on the device`. Root-caused with a live
diagnostic kernel (matching this project's established pattern — never guessed):

- `nvidia-smi` showed the assigned GPU was a **Tesla P100** (compute capability 6.0,
  "sm_60").
- Kaggle's own stock image ships **torch 2.10.0+cu128**, whose own startup warning
  states its minimum supported compute capability is **sm_70** — Pascal-generation
  GPUs (P100) were dropped entirely by this torch build.
- Confirmed this is a known, documented Kaggle API bug (their own product-feedback
  board), not our code: kernels pushed via the API with only `enable_gpu: true`
  default to a P100, while Kaggle's own notebook UI defaults to a T4 — the API and
  UI defaults diverge, and the API's default is the one that's actually broken
  against Kaggle's own current image.
- **Fix, verified live**: `kaggle kernels push` has a real `--accelerator` flag (not
  documented in `kernel-metadata.json`'s schema — it's a push-command argument).
  Pushed the exact same diagnostic script twice: once plain (assigned P100, crashed
  on a plain GPU matmul) and once with `--accelerator NvidiaTeslaT4` (assigned a real
  T4 x2, ran the same matmul fine, torch_geometric imported fine). `kaggle_exec.py`'s
  push call now always passes `--accelerator NvidiaTeslaT4`.

This was caught immediately (first real Kaggle execution in the actual pipeline, not
just the earlier dummy-candidate smoke test, which never touched GPU-dependent code)
— a reminder that a smoke test only verifies what it actually exercises; the dummy
candidate's `train`/`evaluate` functions did no GPU work, so it couldn't have caught
this.

## No early stopping, and an undisclosed architecture blow-up (2026-09-13)

The next run (post tuning-removal) surfaced two more real issues, both from reading
the actual generated code rather than just watching metrics:

1. **No early stopping.** The Software Engineer's `train` ran a fixed `CFG_EPOCHS =
   80` unconditionally, tracking the best checkpoint but never stopping early. Real
   measured rate on Kaggle's T4: ~84.3s/epoch, consistent across epochs — so a full
   run cost ~1.87 hours regardless of whether the model had already converged, and
   the live val_mae curve (0.0607 → 0.0456 → 0.0380 → 0.0371 → 0.0338 → 0.0321 across
   epochs 1-6) showed exactly the decelerating-improvement pattern that makes most of
   those 80 epochs likely wasted time. **Fixed**: `software_engineer.py`'s prompt now
   requires patience-based early stopping (keep the best checkpoint, stop once
   validation stops improving for a fixed window) — a training safeguard, explicitly
   distinguished from the just-removed hyperparameter search (one fixed patience
   value, not tuned).
2. **An undisclosed architecture deviation inflated the model 34x.** That same
   candidate's 6,485,648 parameters traced to one specific choice: its
   `ContinuousFilter` layer outputs a full 128×128 weight matrix per edge (applied via
   `torch.bmm`) instead of SchNet's real elementwise filter vector — a legitimate
   published variant (Gilmer et al. 2017's MPNN "edge network"), but never disclosed,
   cited, or justified anywhere; the blueprint said "SchNet-style" and cited the real
   SchNet paper, not this. This project's own earlier candidate, same hidden_dim=128/
   layers=3/gaussians=50 settings but the standard elementwise filter, had 190,288
   parameters — 34x smaller for what the blueprint called the same architecture
   family. With hyperparameter tuning now removed and no CV, an unexplained 34x
   parameter blow-up on a ~17k-molecule training set is a real, avoidable overfitting
   risk nobody actually chose to accept — the Reviewer's checklist doesn't (and
   shouldn't) police architecture size, so nothing was positioned to catch this except
   reading the generated code directly. **Fixed**: the Software Engineer's prompt now
   states it has no search tool of its own, so every design decision must be
   justified by what the blueprint ALREADY establishes (including its cited sources)
   -- if the blueprint names an architecture family, implement the actual published
   version, not an unstated heavier/different variant; any genuine deviation must be
   disclosed with reasoning in the final summary, never silent.

## Execution failures triggered a wasted, blind Software Engineer rewrite (2026-09-13)

User noticed a real inefficiency while watching a live run: a `stage_evaluate`
`AttributeError` wasn't visible to the Software Engineer until several rounds after
the execution that produced it. Root cause, found by re-reading
`_build_review_execute_loop`'s actual control flow: the loop unconditionally called
`implement_candidate` FIRST every round, regardless of why the previous round ended.
After a real execution failure, `reviewer_notes` is `None` (there's no fix guidance
yet -- only the Reviewer has the mechanism to interpret `execution_error`), so that
next `implement_candidate` call happened completely blind: no notes, no idea anything
broke, free to change anything or nothing. Only AFTER that wasted round did
`review_candidate` finally see the real error and produce actual fix notes -- one
full round (a costly multi-turn Software Engineer tool loop) spent for zero benefit.

**Fixed** with a `need_implement` flag: `implement_candidate` is now skipped on any
round that follows a real execution failure, going straight to `review_candidate`
with the real `execution_error` -- reviewing the SAME code that actually failed
(more accurate than reviewing a blindly-mutated variant), and producing real fix
notes immediately rather than after a wasted extra round. The round number at which
the Software Engineer itself receives informed fix notes doesn't necessarily change
(the Reviewer still needs its own round to translate the error into notes) -- what's
eliminated is the pointless blind rewrite attempt in between, which cost real turns
and money for a code change that could not have been informed by anything.

## Unguarded dispatch lookup crashed on a hallucinated tool name (2026-09-13)

Resuming the interrupted run to design/build candidate 2 crashed for real:
`KeyError: 'edit_file'` in `run_tool_loop`. The Software Engineer (only ever given
`write_file`/`read_file`/`local_run`) called a tool named `edit_file` that doesn't
exist in its own toolset -- plausibly hallucinated by analogy to a common tool name
pattern. `dispatch[tool_call.function.name]` was a bare, unguarded lookup sitting
OUTSIDE the `try/except` that already wraps the actual tool call (`fn(**args)`) --
so an unknown tool name crashed the whole pipeline instead of getting the same
"broken tool call becomes text feedback, not a crashed harness" treatment this file's
own docstring already describes as its guiding principle, just not applied to this
specific failure point. **Fixed**: check `tool_call.function.name in dispatch` first;
on a miss, feed back which tools are actually available as a normal tool-result
message instead of crashing, letting the model self-correct on its next turn.

## Full-codebase audit fixes, batch 1 (2026-09-13)

User asked for a full review of the last complete run's logs/traces/decisions,
specifically flagging: truncated log previews, a recurring `offset`/`limit` tool
error, empty `reply_preview` fields, the Planner's research depth, and whether every
stage actually gets the context it needs. Each was checked against the real trace
before being treated as a finding (several turned out to be real, confirmed bugs, not
just questions):

1. **`evaluation_protocol` never reached the Software Engineer.** Written to
   `blueprint.json` by the Planner, read by nothing downstream -- the SWE had no way
   to know what split/metric the Planner actually specified and had to reinvent one.
   **Fixed**: threaded through `_run_one_candidate` -> `_build_review_execute_loop` ->
   `implement_candidate`, included in its system prompt.
2. **`diagnostic_plan` reached the Software Engineer but never the Reviewer.** The
   Reviewer had no way to check whether the SWE's `evaluate` stage actually reflected
   the diagnostic analysis it was asked to produce. **Fixed**: passed through to
   `review_candidate`, given as judgment context (not a rigid checklist item, since
   it's candidate-specific free text).
3. **`read_file`'s `offset`/`limit` mismatch, confirmed 4 real failures in one run.**
   The model (plausibly generalizing from other coding-assistant tools) repeatedly
   called `read_file(path, offset=N, limit=M)`; the harness's real signature only
   takes `path`, so every attempt raised `TypeError`, wasting a turn each time.
   **Fixed**: the dispatch lambda now accepts and silently ignores extra kwargs
   (`lambda path, **_ignored: ...`), and the tool's description now states files are
   always returned in full, since candidate projects are small enough that partial
   reads were never actually needed.
4. **Reviewer's `STANDING_CHECKLIST` never got an item for the new mandatory early
   stopping requirement** (added in the previous fix batch) -- a candidate with no
   early-stop logic at all could still pass review. **Fixed**: added as a standing
   checklist item.
5. **The Planner's research loop was real but thin.** Confirmed from the actual
   trace: `draft_blueprint` used only 2 tool calls total (one `search_papers`, one
   `python_sandbox`) out of an 8-turn budget, and didn't retry after a search hit a
   real arXiv rate-limit/timeout. **Fixed**: prompt now requires at least 2-3
   genuinely different `search_papers` queries and explicit retry-on-failure/weak-
   result guidance, rather than accepting the first attempt as sufficient.
6. **Kaggle execution, six sub-fixes, all in `kaggle_exec.py`:**
   - No pre-flight check for `KAGGLE_KEY` or the `kaggle` binary's existence --
     either missing case previously raised an uncaught exception that crashed the
     whole pipeline instead of failing just one round. Now checked up front.
   - Kernel slugs were reused across retry rounds for the same candidate, risking
     `kernels status` reading a stale terminal status from the PREVIOUS round's
     kernel before Kaggle finished transitioning to the new push. Slugs (and the
     push/output directories) now include the round number, so every retry gets its
     own kernel.
   - The import-stripping regex only handled single-line imports; a multi-line
     parenthesized `from qm8_data import (...)` would leave orphaned, syntactically
     invalid continuation lines in the generated script. Fixed with an added
     DOTALL pass for the parenthesized case first -- verified against both forms
     directly (both now produce script text that actually compiles).
   - Output files (including `results.json`) were copied into `candidate_dir`
     unconditionally, before checking whether the run actually succeeded -- a
     `results.json` written by `evaluate` just before a LATER crash in the same
     script would still land in `candidate_dir` and could reach the final report as
     if it belonged to a successful run. Now only copied on a confirmed "complete"
     status.
   - On our own `RUN_TIMEOUT_SECONDS` being exceeded, nothing ever tried to stop the
     still-running remote kernel -- it kept consuming GPU-hours up to Kaggle's own
     9h cap. The Kaggle CLI has no verified "cancel" call (confirmed against its own
     `--help`), only `delete` with an unverified effect on an in-progress run --
     attempted anyway as best-effort cleanup, since it can only help.
   - `execute_candidate_kaggle` took no `trace` and logged nothing itself; only the
     caller's post-hoc summary existed. Now logs push outcome and poll
     duration/result under a new "kaggle_exec" stage.
7. **Trace preview truncation (300/200 chars) was cutting real diagnostic content**
   with no other record for `local_run`/`read_file`/`search_papers` results or
   intermediate LLM text. Raised to 2000/1000 chars respectively -- a plain constant
   change, not new infrastructure.

Every changed function's call sites were cross-checked directly (`grep` across the
whole `harness/` tree) after editing, not assumed consistent -- the multi-line-import
regex fix and the offset/limit tolerance fix were each additionally verified with a
standalone script before being considered done, not just read over.

## Prompt-only "retry on failed search" didn't hold; enforced in code instead (2026-09-13)

The very next real run reproduced the exact failure the prompt fix above was meant to
prevent: one `search_papers` call hit a real arXiv timeout, returned zero real
results, and the Planner finalized the blueprint anyway -- despite the prompt
explicitly saying to retry with different keywords on a failed/weak search. Same
lesson as the fenced-block format issue from earlier in this project: a soft prompt
instruction asking a cheap model to follow a multi-step meta-behavior isn't reliably
followed just by asking.

**Fixed the same way the format issue was fixed** -- enforced in code via the
`validate` callback already used for format-checking, not left as a prompt-only
request. `draft_blueprint` now wraps `search_papers` in a small closure that tracks
call count and whether any call returned at least one real (non-error) result;
`_validate_blueprint_response` (composing the existing format check with this new
one) refuses to accept a final answer until at least one search has actually
succeeded, injecting a corrective message otherwise -- the identical retry mechanism
Coscientist's own pattern already established for malformed output, just applied to
"zero real research grounding" as another kind of invalid final answer. Verified with
an isolated logic test (an all-error result correctly blocks, a real result correctly
unblocks) before being trusted. The now-unused module-level `DISPATCH` constant
(replaced by this function's own tracked dispatch dict) was removed rather than left
as dead code.

## arXiv wasn't slow, it was being hammered with zero throttling (2026-09-13)

User asked to actually diagnose the recurring arXiv timeouts/errors rather than keep
bumping `REQUEST_TIMEOUT_SECONDS` blindly (15->25 earlier tonight was exactly that --
a guess, not a measurement). Measured real latency directly instead:

- `http://export.arxiv.org/api/query` (the URL this code used) 301-redirects every
  single request to `https://` -- confirmed live with `allow_redirects=False`. Every
  call was paying for two full connections, not one.
- arXiv's own Terms of Use (info.arxiv.org/help/api/tou.html) state a hard limit:
  **one request every 3 seconds, single connection at a time**. This code had ZERO
  throttling for arXiv -- Semantic Scholar already had one (`_MIN_SECONDS_BETWEEN_REQUESTS`),
  arXiv never did.
- Real measured data: successful requests were fast (sub-1s to a few seconds) --
  the failures were `429`s and full-timeout hangs, not slow-but-working responses.
  A bigger timeout cannot fix a rate limit; this confirmed the actual problem before
  touching any timeout value.
- Semantic Scholar, tested independently and directly: 100% success, 0.97-1.71s
  latency across 3 real queries. Not the source of any problem -- confirmed working,
  not assumed.

**Fixed**: `ARXIV_API` now points directly at `https://...` (skips the redirect);
added `_MIN_SECONDS_BETWEEN_ARXIV_REQUESTS = 3.5` (arXiv's documented 3s minimum plus
the same small safety margin Semantic Scholar's 1.1s already uses over its own 1/sec
limit); added explicit 429 handling for arXiv matching the graceful-degradation
pattern Semantic Scholar already had, so a rate limit now produces a clear
`{"source": "arxiv", "error": "rate-limited (429)"}` instead of a generic
`requests.HTTPError` string.

**Verified live, and found something to be honest about**: even with all three fixes
in place, arXiv is STILL returning 429 for every real request right now. This is not
the code being broken -- it's almost certainly this IP hitting arXiv's own documented
"temporary IP-based blocking" for excessive usage, from tonight's cumulative testing
(many pipeline runs, each with multiple searches, plus repeated diagnostic calls just
now). A code fix cannot lift an existing block; it needs real time to expire on
arXiv's side. In the meantime `search_papers()` as a whole tool remains genuinely
useful -- Semantic Scholar keeps returning real, on-topic results (verified: actual
SchNet, MoleculeNet, and GNN papers came back correctly) even while arXiv is down,
which is exactly what the existing "one source's failure doesn't lose the other's
results" design is for.

## Removed the GNN nudge from the task framing; required real alternatives (2026-09-13)

User pushed back on a real tension flagged during review: `run.py`'s `TASK_DESCRIPTION`
stated "success factors... spatial-relational architectures (GNNs, 3D-aware models)
since the labels are 3D-structure-dependent" -- honest domain framing, not fabricated,
but every real run so far had converged on a GNN, and there was no way to tell whether
that was genuine independent reasoning or just following an unsubtle hint. User's own
words: "what if GNN is not needed, boosting algorithms will handle for example -- also
the GNN choosing must be its decision too."

**Fixed**: `TASK_DESCRIPTION` now states only neutral facts (dataset contents,
including that 3D coordinates ARE present in the raw data -- a fact, not a directive
to use them) and explicitly says no architecture family is assumed to work best.
Paired with a new, code-enforced requirement (`alternatives_considered`, added to
`BLUEPRINT_JSON_SCHEMA` and the required-keys check in `_validate_two_fenced_blocks`,
same enforcement pattern as the search-effort fix above): the Planner must now state
at least one real alternative architecture family it weighed against candidate 1 and
specifically why it wasn't picked, grounded in what it actually found -- not a generic
tradeoff statement. This doesn't change what the Planner is allowed to choose, only
makes the choice itself, and the reasoning against a real alternative, visibly and
auditably its own -- directly serving the actual hiring ask's "let them do research"
framing, not just "let them pick a model."

## A fresh code review caught a real edge case in yesterday's search-effort fix

A follow-up review of the search-effort enforcement found a genuine bug before it
could bite: `search_successes` incremented on `any("error" not in r for r in
results)`, which is `False` for an EMPTY results list -- meaning a fully legitimate
search that reached both sources cleanly but found zero matching papers would be
wrongly treated as a failure, forcing retries until `MAX_TOOL_TURNS=8` was exhausted
and crashing the whole pipeline with `RuntimeError`. **Fixed**: count error entries
instead (`search_papers` always queries exactly 2 sources, contributing either real
paper entries or exactly one `{"error": ...}` per failed source) -- `error_count < 2`
correctly means "at least one source responded without failing, whatever it found,"
verified against all four real scenarios (clean-empty, partial failure with real
results, partial failure with a clean-empty other side, total failure) before being
trusted. Also tightened the fix's own comment, which had overclaimed enforcing the
prompt's full "2-3 distinct queries" ask when the code only ever checked for one
successful search -- mechanically verifying query keywords were "genuinely
different" isn't worth the complexity it would add.

## arXiv temporarily disabled; fixed a latent bug this surfaced (2026-09-13)

Given arXiv's temporary block hasn't cleared and the deadline doesn't allow waiting
it out, user asked to just disable arXiv for now rather than keep eating its timeout/
429 cost on every search. Added `ARXIV_ENABLED = False` (a plain module constant,
flip back to `True` once arXiv responds normally again) and made `search_papers()`
build its source list from it.

Turning off a source exposed a real bug in the search-effort validator from earlier
tonight: `error_count < 2` hardcoded the assumption that exactly 2 sources are always
queried. With only Semantic Scholar active, a single failure of that ONE source would
still satisfy `1 < 2` and be wrongly reported as success -- meaning the entire
search-effort gate could silently no-op with arXiv off. **Fixed** by moving success
determination into `search_papers.py` itself as `search_succeeded()`, which computes
the threshold from the actual number of active sources rather than a hardcoded
constant -- the module that owns `ARXIV_ENABLED` is the only place that can correctly
answer "how many sources were even tried." `planner.py`'s tracked wrapper now just
calls this shared helper. Verified against all real scenarios both with arXiv enabled
(the original 4 cases) and disabled (3 new cases, including the one that would have
silently broken) before trusting it.

The currently-running pipeline (already past `draft_blueprint`, mid candidate-1 build)
has the old module loaded in memory and will still attempt arXiv once more during the
final write-up -- harmless (gracefully degrades to Semantic Scholar, same as always),
just not benefiting from the toggle. Not worth restarting and losing real progress
over; the fix is for the next run.

## A 2.3MB pickle file crashed the Reviewer with a 637k-token prompt (2026-09-13)

The next real run crashed for real: `openai.BadRequestError: ... maximum context
length is 400000 tokens. However, you requested about 637168 tokens`. Root cause,
found by inspecting the actual candidate directory rather than guessing: the
Software Engineer had written a 2.3MB `models/baseline_models.pkl`, and
`reviewer.py`'s file-reading helper read it as text with `errors="replace"`,
turning 2.3MB of binary data into hundreds of thousands of garbage tokens stuffed
into the Reviewer's prompt.

The deeper problem: there were TWO separately-maintained private copies of
essentially the same "read back everything the candidate produced" logic --
`orchestrator.py`'s (non-recursive, excluded binary files) and `reviewer.py`'s
(recursive, did NOT exclude binary files). They had already silently diverged,
and the copy that actually ran into a real binary file was the one missing the
exclusion. This is exactly the failure mode duplicated logic invites.

**Fixed** by deleting both private copies and adding one shared
`read_candidate_files()` in `harness/tools/files.py` (a real utility module already
shared by both, not a new dependency), recursive AND binary-excluding, used by both
`orchestrator.py` and `reviewer.py` now. Caught a second real issue while doing this:
recursive traversal returns relative-path keys like `"models/summary.json"`, which
would break `python_sandbox`'s `input_files` writer (`write_text()` doesn't create
parent directories, so a literal "/" in a key would raise `FileNotFoundError` the
moment any candidate organizes output into a subdirectory) -- fixed by flattening
"/" to "_" in the keys at the source, once, rather than requiring every caller to
remember to sanitize them.

Verified, not assumed: a standalone test recreated the exact crash scenario (a
2.3MB random-bytes file in a `models/` subdirectory, alongside a real JSON file and
the excluded `qm8_data.py`) and confirmed the binary file is excluded, the legitimate
subdirectory file comes through with its flattened key, and `python_sandbox` writes
that key correctly with no subdirectory-creation error.
