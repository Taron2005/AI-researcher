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
~22,000 small organic molecules (up to 8 heavy atoms, drawn from GDB-8), each
with a 3D conformation, labeled with electronic spectra properties (S0->S1 and
S0->S2 excitation energies and oscillator strengths) computed via TDDFT and
CC2 quantum chemistry methods. The dataset is fully labeled -- there is no
labeling budget constraint. Success factors known from related literature on
similar agentic ML tasks: strategic model/architecture selection, using
spatial-relational architectures (GNNs, 3D-aware models) since the labels are
3D-structure-dependent, and not discarding position/orientation information
during featurization."""

if __name__ == "__main__":
    trace_path = run_pipeline(task_description=TASK_DESCRIPTION)
    print(f"\nRun complete. Trace: {trace_path}")
    print("Write-up: writeup/report.md")
