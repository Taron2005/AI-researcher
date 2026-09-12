"""
Software-Engineer-only tools: read/write files, scoped to one candidate's
own directory in workspace/. Path validation rejects anything that would
escape that directory (e.g. "../../harness/config.py") rather than silently
resolving it -- this is the sandboxing boundary for the one role that
actually produces the deliverable code (CLAUDE.md rule 8).

The schemas below take a plain `path` parameter -- what the model reasons
about. The actual base directory is bound in by roles/software_engineer.py
via a closure when it wires up the dispatch table (see that file), not
hardcoded here -- this module is a plain, reusable primitive, not aware of
"candidates" or "workspace" as concepts.
"""

from pathlib import Path

WRITE_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write (or overwrite) a file in your candidate's own project directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path, e.g. 'main.py' or 'requirements.txt'"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
}

READ_FILE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from your candidate's own project directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path, e.g. 'main.py'"},
            },
            "required": ["path"],
        },
    },
}


def _resolve_safe_path(base_dir: Path, relative_path: str) -> Path:
    """
    Resolves `relative_path` against `base_dir` and refuses anything that
    escapes it. Raises rather than silently clamping, so a bad path shows
    up as a clear tool error the model can react to, not a confusing
    wrong-file read/write.
    """
    resolved = (base_dir / relative_path).resolve()
    if not resolved.is_relative_to(base_dir.resolve()):
        raise ValueError(f"Path '{relative_path}' escapes the candidate directory -- refused.")
    return resolved


def write_file(base_dir: Path, path: str, content: str) -> str:
    target = _resolve_safe_path(base_dir, path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} chars to {path}"


def read_file(base_dir: Path, path: str) -> str:
    target = _resolve_safe_path(base_dir, path)
    if not target.exists():
        return f"(file '{path}' does not exist yet)"
    return target.read_text()
