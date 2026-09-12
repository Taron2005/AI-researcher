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

Edit `.env` and set `OPENROUTER_API_KEY` (required -- get one at
https://openrouter.ai/keys). `SEMANTIC_SCHOLAR_API_KEY` is optional (raises an
otherwise-tight anonymous rate limit).

## Run

```bash
python run.py
```

This runs the full pipeline end to end: Planner drafts a research blueprint, the
Software Engineer implements and locally executes each candidate, the Reviewer
gates every candidate before it runs, and the Planner writes the final report.

Output per run (timestamped, never overwritten):
- `workspace/run_<id>/` -- blueprint + each candidate's code
- `traces/run_<id>.jsonl` -- full structured log of every agent step
- `writeup/run_<id>/report.md` -- the final research write-up

## Hardware note

Candidate models are chosen by the Planner itself and can include GNNs -- real
measurement on a CPU-only laptop found one epoch of a mid-size GNN config takes
roughly a minute on QM8's ~17k-molecule training set, meaning a full run (hyperparameter
search + final training) can take **2-6+ hours on CPU**. If you have a working NVIDIA
GPU (confirm with `nvidia-smi` and that the CUDA-build `torch` above installed
correctly), this should be dramatically faster. If you're CPU-only and a run keeps
failing with a training timeout, raise `TRAIN_TIMEOUT_SECONDS` in
`harness/tools/execute.py` (currently 20 minutes) to match what your hardware actually
needs.
