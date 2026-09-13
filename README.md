# QM8 Agentic Research Harness

A small team of AI agents (Planner, Software Engineer, Reviewer) that autonomously
researches the QM8 quantum-chemistry dataset -- the Planner decides its own modeling
approach from real investigation (papers, dataset inspection), not a prescribed
methodology. See CLAUDE.md / ARCHITECTURE.md / DECISIONS.md for the full design.

## Setup

```bash
git clone <repo-url>
cd "ML Engineering Deep Dive"   # or whatever you named the clone

python3.12 -m venv .venv
source .venv/bin/activate
```

**Install PyTorch first**, matching your hardware -- check for an NVIDIA GPU with
`nvidia-smi`:

```bash
# No NVIDIA GPU (CPU-only):
pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU: check your CUDA version, then pick the matching build from
# https://pytorch.org/get-started/locally/ -- e.g. for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then everything else:

```bash
pip install -r requirements.txt
```

**Configure secrets** (never commit `.env` -- it's gitignored):

```bash
cp .env.example .env
```

Edit `.env` and set:
- `OPENROUTER_API_KEY` (required -- get one at https://openrouter.ai/keys)
- `KAGGLE_USERNAME` and `KAGGLE_KEY` (required -- candidates train/evaluate on
  real Kaggle GPU compute by default; get these from
  https://www.kaggle.com/settings under "API" -> "Create New Token", which
  downloads a `kaggle.json` containing both values)
- `SEMANTIC_SCHOLAR_API_KEY` (optional -- raises an otherwise-tight anonymous
  rate limit)

## Run

```bash
python run.py
```

This runs the full pipeline end to end: the Planner drafts a research blueprint,
the Software Engineer implements each candidate, the Reviewer gates every
candidate before it runs, each candidate actually trains/evaluates on a real
Kaggle GPU kernel (`harness/tools/kaggle_exec.py`), and the Planner writes the
final report. A single candidate typically takes anywhere from ~15 minutes to
a few hours depending on what architecture the Planner chooses -- this is
normal, not a hang; check the trace file for live progress.

Output per run (timestamped, never overwritten):
- `workspace/run_<id>/` -- blueprint + each candidate's code and real results
- `traces/run_<id>.jsonl` -- full structured log of every agent step
- `writeup/run_<id>/report.md` -- the final research write-up

No local GPU is needed -- training happens on Kaggle, not on your machine. To
run candidates locally instead (e.g. no Kaggle account), set
`EXECUTION_BACKEND = "local"` in `harness/config.py`; local CPU-only training
can take multiple hours depending on what the Planner chooses (verified
directly: roughly 2-6 hours for a mid-size GNN candidate on a CPU-only laptop),
so raising `TRAIN_TIMEOUT_SECONDS` in `harness/tools/execute.py` (currently 20
minutes) may be necessary in that mode.
