"""
Shared subprocess helper: runs a command with a hard timeout that actually
kills the whole process tree, not just the direct child.

`subprocess.run(..., timeout=...)`'s own timeout handling only kills the
immediate child process -- if that child spawns its own worker processes
(e.g. scikit-learn's `n_jobs=-1` via joblib/loky), those become orphaned
and keep running indefinitely after the timeout fires. This is not
theoretical: a real timed-out `train` stage left loky worker processes
running at 35% CPU for 17+ minutes after the parent had already been
killed (found by checking `ps aux` after a run, not by reading the code).
See DECISIONS.md. Every subprocess call in this codebase with a timeout
(python_sandbox, local_run, install_dependencies, execute_candidate's
train/evaluate) had this same latent bug -- fixed once, here, rather than
patched separately in each.
"""

import os
import signal
import subprocess


class TimedOut(Exception):
    """Raised instead of subprocess.TimeoutExpired -- same meaning, this module's own type."""

    def __init__(self, timeout: float):
        self.timeout = timeout
        super().__init__(f"timed out after {timeout}s")


def run_with_timeout(
    cmd, cwd, timeout: float, env: dict | None = None, shell: bool = False,
) -> subprocess.CompletedProcess:
    """
    Same shape as `subprocess.run(cmd, cwd=cwd, timeout=timeout,
    capture_output=True, text=True, env=env)`, except on timeout it kills
    the ENTIRE process group (the command and anything it spawned), not
    just the command itself. Raises `TimedOut` instead of
    `subprocess.TimeoutExpired`.
    """
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        shell=shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        # Own process group/session, so on timeout we can kill it and every
        # descendant together (os.killpg) instead of only the process we
        # directly started.
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()  # reap the now-dead process so it doesn't linger as a zombie
        raise TimedOut(timeout)

    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
