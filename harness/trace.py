"""
Structured run log — the "run traces" deliverable (CLAUDE.md rule 6).

One JSON object per line, appended as the run happens. Deliberately plain:
no logging framework, no database — a run is a handful of agent calls and
tool calls, and a human (or us, debugging) should be able to read this file
top to bottom in a text editor.

Every role/tool call in the harness should produce exactly one log_event()
call. Nothing about "what happened" should live only inside an LLM's own
context — if it matters for an audit later, it's in this file.
"""

import json
import time
from pathlib import Path

TRACE_DIR = Path(__file__).parent.parent / "traces"


class Trace:
    """One run's log file. Create one per orchestrator run."""

    def __init__(self, run_id: str):
        TRACE_DIR.mkdir(exist_ok=True)
        self.path = TRACE_DIR / f"run_{run_id}.jsonl"

    def log_event(self, stage: str, event_type: str, **fields):
        """
        stage: which part of the pipeline this is ("planner", "swe",
               "reviewer", "install", "execute").
        event_type: what kind of thing happened ("llm_call", "tool_call",
               "gate_result", "budget_update", etc).
        fields: anything relevant — model, prompt summary, tokens, cost,
               latency_s, output summary, error, etc. Kept as free-form
               kwargs rather than a fixed schema so each stage logs what's
               actually relevant to it, without a shared class hierarchy.
        """
        record = {
            "timestamp": time.time(),
            "stage": stage,
            "event_type": event_type,
            **fields,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
