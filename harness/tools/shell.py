"""
Software-Engineer-only tool: a quick, timeout-bounded local self-check --
does the code parse, does a tiny dry run complete -- NOT the real training
run (that's the deterministic Kaggle execution step the orchestrator drives
separately; see ARCHITECTURE.md). Sandboxing matches python_sandbox.py: a
restricted subprocess (CLAUDE.md rule 8 explicitly allows this over a
container), scoped to the candidate's own directory, with a hard timeout.

Also provides install_dependencies() -- deliberately NOT exposed to the LLM
(no TOOL_SCHEMA). Called directly by the orchestrator once the Reviewer
approves a candidate's code, per DECISIONS.md's "batch install, not
reactive per-error" design: a real version conflict is better resolved by
pip's own constraint solver than by an LLM guessing compatible versions.
"""

import os
from pathlib import Path

from harness.tools.subprocess_utils import TimedOut, run_with_timeout

LOCAL_RUN_TIMEOUT_SECONDS = 60
INSTALL_TIMEOUT_SECONDS = 300

# The harness's own pinned virtual environment -- local_run must resolve
# bare `python`/`python3`/`pip`/`pip3` to THIS, not whatever the system
# shell's ambient PATH happens to find. Without this, a candidate's
# self-check silently runs against the wrong Python entirely (found by
# actually running it: the SWE agent burned its whole turn budget confused
# about why torch/pandas/rdkit "weren't installed" -- they were, just not
# in the Python it was accidentally invoking). See DECISIONS.md.
_VENV_BIN = Path(__file__).parent.parent.parent / ".venv" / "bin"

# The one real QM8 cache (see harness/qm8_data.py) -- pointed to explicitly
# so a candidate's copy of qm8_data.py (a __file__-relative path away from
# this project root once copied) reuses it instead of re-downloading and
# re-parsing the whole dataset on every local self-check. See DECISIONS.md.
_QM8_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "qm8"


def venv_env() -> dict:
    return {
        **os.environ,
        "PATH": f"{_VENV_BIN}:{os.environ.get('PATH', '')}",
        "VIRTUAL_ENV": str(_VENV_BIN.parent),
        "QM8_DATA_DIR": str(_QM8_DATA_DIR),
    }

LOCAL_RUN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "local_run",
        "description": (
            "Run a shell command in your candidate's own project directory -- "
            "for a quick sanity check (does it parse, does a tiny dry run "
            "complete), NOT the real training run. Times out after 60s."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command, e.g. 'python main.py --stage baseline'"},
            },
            "required": ["command"],
        },
    },
}


def local_run(base_dir: Path, command: str) -> str:
    try:
        result = run_with_timeout(
            command,
            shell=True,
            cwd=base_dir,
            timeout=LOCAL_RUN_TIMEOUT_SECONDS,
            env=venv_env(),
        )
    except TimedOut:
        return f"(command timed out after {LOCAL_RUN_TIMEOUT_SECONDS}s)"

    output = result.stdout
    if result.stderr:
        output += f"\n--- stderr ---\n{result.stderr}"
    return output.strip() or f"(no output, exit code {result.returncode})"


def install_dependencies(base_dir: Path, python_executable: str | None = None) -> tuple[bool, str]:
    """
    Batch-installs the candidate's requirements.txt in one call. Not an LLM
    tool. Defaults to the harness's own venv -- this is for LOCAL pre-flight
    validation (so local_run/"baseline" can be verified before spending
    Kaggle time), separate from whatever install step targets Kaggle's own
    image once kaggle_exec.py exists (that one installs into a remote
    environment this function knows nothing about).

    Returns (success, output) -- success is the subprocess's actual return
    code, not a guess from string-matching "error" in the output (pip's
    own text doesn't reliably contain that word on failure, and can
    contain it harmlessly on success, e.g. in a package description).
    """
    python_executable = python_executable or str(_VENV_BIN / "python")
    requirements_path = base_dir / "requirements.txt"
    if not requirements_path.exists():
        return True, "(no requirements.txt found -- skipping install)"

    try:
        result = run_with_timeout(
            [python_executable, "-m", "pip", "install", "-r", str(requirements_path)],
            cwd=base_dir,
            timeout=INSTALL_TIMEOUT_SECONDS,
        )
    except TimedOut:
        # Previously uncaught here -- a hung pip install would crash the
        # whole pipeline instead of being reported as a normal failure.
        return False, f"(pip install timed out after {INSTALL_TIMEOUT_SECONDS}s)"

    output = result.stdout
    if result.stderr:
        output += f"\n--- stderr ---\n{result.stderr}"
    return result.returncode == 0, output
