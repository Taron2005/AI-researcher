"""
Entry point: `python run.py` runs the full pipeline end to end (ARCHITECTURE.md).

The task description here is what the Planner sees as its starting brief --
it does NOT include our own design docs (CLAUDE.md/DECISIONS.md/
ARCHITECTURE.md), since those are about how *we* built the harness, not
QM8 chemistry. The Planner has its own tools (search_papers, python_sandbox)
to find domain literature and inspect the real data itself.
"""

from harness.orchestrator import run_pipeline

TASK_DESCRIPTION = """Analyze the QM8 dataset and produce a research result: propose a
modeling approach, implement and evaluate it, and report findings honestly.

QM8 is a quantum-chemistry molecular property dataset (part of MoleculeNet):
~22,000 small organic molecules (up to 8 heavy atoms, drawn from GDB-8). Each
molecule is labeled with 16 electronic spectra properties (S0->S1 and S0->S2
excitation energies and oscillator strengths) computed via TDDFT and CC2
quantum chemistry methods, and the raw data includes each molecule's SMILES
string, atomic numbers, and 3D atomic coordinates. The dataset is fully
labeled -- there is no labeling budget constraint.

Decide the modeling approach yourself, based on what you actually find by
inspecting the data and checking the literature -- nothing here should be
read as implying which architecture family (a 3D-aware GNN, classical ML on
molecular descriptors/fingerprints, or anything else) will work best; that's
for you to determine, not something to assume going in."""

if __name__ == "__main__":
    trace_path = run_pipeline(task_description=TASK_DESCRIPTION)
    run_id = trace_path.stem.removeprefix("run_")
    print(f"\nRun complete. Trace: {trace_path}")
    # Was a hardcoded "writeup/report.md" -- stale since write_final_report
    # scopes its output per run_id to avoid overwriting a previous run's
    # real write-up (same reasoning as candidate directories, DECISIONS.md).
    print(f"Write-up: writeup/run_{run_id}/report.md")
