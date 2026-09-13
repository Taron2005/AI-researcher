# QM8 Agentic Research — Submission Overview

A short, skimmable summary of what this is. For setup/run instructions see
[README.md](README.md); for the full design reasoning see
[CLAUDE.md](CLAUDE.md), [ARCHITECTURE.md](ARCHITECTURE.md), and
[DECISIONS.md](DECISIONS.md) (a chronological log of every real decision,
correction, and bug found while building this).

## What this is

A three-agent research team — Planner, Software Engineer, Reviewer — that
autonomously researches the QM8 quantum-chemistry dataset (TDDFT/CC2
excitation energies and oscillator strengths on ~22k small molecules). The
Planner decides the actual research: what model architecture to try and why,
what to measure beyond the headline metric, whether a second attempt is
warranted, and what it should be — informed by real evidence (dataset
inspection, literature search, and, for a second candidate, the first
candidate's actual results) rather than a template we handed it.

## Why a team of agents, not one

- **Planner/Research** — decides the approach, using real tools (`search_papers`
  against arXiv/Semantic Scholar, a sandboxed Python tool to inspect the real
  dataset) rather than memory alone. Reused at three points in the pipeline:
  drafting the initial research plan, deciding on a second candidate from the
  first one's real result, and writing the final report.
- **Software Engineer** — implements exactly what the Planner specified, with
  no search tool of its own — every implementation decision has to be
  justified by what the blueprint already states, not invented.
- **Reviewer** — never writes code; a mandatory gate that checks a fixed
  engineering checklist (does it run, does it use the provided data loader,
  does it produce real output) plus the blueprint's own stated constraints,
  before any candidate reaches real execution.

This split (and the retry/budget logic around it) is modeled on real published
patterns — Coscientist's minimal tool-calling loop, Deep Thought's documented
failure modes, and AIDE's Solution-Generator → Evaluator → Selector loop for
ML research — not invented from scratch. See DECISIONS.md for what was
checked and why each pattern was or wasn't adopted.

## What's genuinely autonomous vs. what's fixed infrastructure

**Fixed by the harness** (engineering process, not research content): the CLI
contract every candidate must implement, cost/time bounds (no hyperparameter
search, required early stopping, turn/round caps), the Reviewer's mechanical
checklist, and a hard rule that any benchmark comparison must be verified via
a real search, never stated from memory.

**Decided by the agents, every run**: the actual model architecture, its
hyperparameters, what diagnostic analysis to run beyond the headline metric,
whether a second candidate is worth attempting, what that candidate should be,
and what the final report concludes. The task brief given to the Planner
deliberately states no assumption about which architecture family will work
best (an earlier version nudged toward 3D-aware GNNs; every run then picked
one, so the hint was removed to let the choice be genuinely open).

Verified concretely, not assumed, across two different real runs:
- **Independent architecture choice with real alternatives documented**: with
  the neutral task framing, the Planner chose classical ML (RDKit/ECFP
  fingerprints + LightGBM) over a 3D-aware GNN, and its blueprint explicitly
  states what it rejected and why: *"We strongly considered a continuous-filter
  convolutional network leveraging the provided 3D coordinates. However, given
  resource constraints and the extensive hyperparameter search required...
  we rejected it for Candidate 1 in favor of a reliable, high-velocity
  classical baseline."* — a real, specific tradeoff, not a generic dismissal.
- **Evidence-driven iteration**: in a separate run, the Planner reloaded
  candidate 1's actual per-property error breakdown before designing
  candidate 2, and the resulting design (angle-aware features, separate task
  heads for the harder oscillator-strength targets) directly targeted the
  real weak point that data showed — not a generic "try something different."

## Real result (see `writeup/` and `traces/` for the full record)

Latest run — **RDKit/ECFP fingerprints + LightGBM**, chosen and justified by
the Planner itself (see above), trained and evaluated on real GPU compute
(Kaggle):

| Metric | Value |
|---|---|
| Overall test MAE (all 16 properties) | **0.01637** |
| Error uncorrelated with molecule size | r = 0.041 |
| Median absolute error | 0.00782 |

The Planner's own write-up declined to cite a specific published benchmark
number here, because its literature-verification searches hit real API rate
limits during this run — stating that honestly rather than citing an
unverified figure from memory (a real, previously-caught failure mode this
project specifically guards against; see DECISIONS.md).

A separate run with a SchNet-style 3D GNN (chosen and justified on its own
merits, back when candidate 2 was designed from real per-property evidence
rather than a template) achieved 0.01166 overall MAE, which — compared
against the real, hand-verified MoleculeNet benchmark (Wu et al. 2018,
arXiv:1703.00564, Table 9): MPNN 0.0143, GC 0.0148, DTNN 0.0169, KRR 0.0195 —
beat all of them, with the honest caveat that MoleculeNet's official QM8
benchmark uses 12 tasks, while this project's loader correctly includes all
16 real properties (the genuine duplicate PBE0 basis-set columns), so it
isn't perfectly apples-to-apples.

## What was done by hand

I (with AI pair-programming assistance via Claude Code) designed and built
the multi-agent harness itself — the three agent roles, their tools, the
orchestration/retry logic, budget caps, the QM8 data loader, and the Kaggle
GPU execution pipeline — plus all debugging along the way. Specifically, by hand:

- The system architecture: three agent roles, what's fixed engineering
  process (the CLI contract, cost/time bounds, the Reviewer's checklist)
  versus genuinely open to the agents (architecture, hyperparameters,
  whether/how to iterate, conclusions).
- Which LLM powers each role and at what temperature.
- The task brief given to the Planner — including catching and removing an
  earlier draft's unintentional bias toward 3D-aware GNN architectures, so
  the architecture choice would be genuinely open rather than hinted at.
- All harness code and every agent's system prompt (written to state
  constraints and context, not prescribe research content).
- The QM8 data loader — a verified, automated data-access utility (it
  downloads and parses the real public dataset itself at runtime; nothing
  about the data's content is hand-curated or pre-processed), built after
  catching a real bug a naive parser would have hit (a genuine duplicate
  column in the raw label file).
- The Kaggle GPU execution pipeline, and all debugging of real failures
  encountered while building this (documented chronologically in
  DECISIONS.md).
- External verification done outside any pipeline run to make correct
  engineering decisions — e.g. downloading and reading the actual
  MoleculeNet paper to check real benchmark numbers, live-testing Kaggle's
  API behavior directly rather than assuming documented behavior held.
- Reading and verifying the agents' output after the fact — checking a
  report's claims against real sources, confirming a result wasn't
  fabricated — closer to a PI reviewing a report before it goes out than to
  writing it.
- Choosing which of many real runs (see `traces/`) to include in this
  submission.

Within that harness, the AI agents autonomously decided everything about the
actual research: what model architecture to build and why, what
hyperparameters to use, whether a second candidate was worth attempting and
what it should be, what diagnostics to run, and what to conclude — none of
this was specified or scripted by me.

## Run traces

Every agent step (prompt, tool call, output, tokens, cost) is logged to
`traces/run_<id>.jsonl`, one JSON object per line, in real time as the run
happens — not reconstructed after the fact. `workspace/run_<id>/` holds each
candidate's actual generated code; `writeup/run_<id>/report.md` is the
Planner's own final write-up for that run.

The two runs cited above:
- `run_20260913_133643` — the LightGBM/ECFP run with the documented
  alternatives-considered reasoning.
- `run_20260913_022029` — the GNN run with the evidence-driven candidate-2
  iteration.
