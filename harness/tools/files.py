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

Also hosts read_candidate_files() -- not an LLM tool, a plain helper shared
by orchestrator.py and reviewer.py. It used to be two separately-maintained
private copies of the same logic, which had already silently diverged (one
excluded binary files, the other didn't) -- exactly how a 2.3MB .pkl file
ended up read as text into the Reviewer's prompt, ballooning it to 637k
tokens and crashing the pipeline (DECISIONS.md). One shared, correct
implementation instead of two that can drift apart again.
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
        "description": (
            "Read a file from your candidate's own project directory. Always "
            "returns the WHOLE file -- there is no partial/paginated read, "
            "since candidate files are always small enough to read in full."
        ),
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


# Pre-seeded infrastructure, not the Software Engineer's own work -- reading
# it back would be reviewing/re-feeding us, not the candidate.
_EXCLUDED_FROM_CANDIDATE_READ = {"qm8_data.py"}

# Reading one of these as text with errors="replace" doesn't just produce
# garbage -- confirmed live, it can be hundreds of thousands of tokens of
# garbage (a 2.3MB .pkl blew a Reviewer call up to 637k tokens, over
# GPT-5-mini's 400k limit). Neither the Reviewer nor the final write-up
# needs the raw binary object, only whatever real metrics/text the
# candidate's own diagnostic_plan produced.
_BINARY_EXTENSIONS = {".pkl", ".pickle", ".pt", ".pth", ".joblib", ".npy", ".npz", ".h5"}


def read_candidate_files(candidate_dir: Path | None) -> dict[str, str]:
    """
    Reads back every real file a candidate's Software Engineer produced,
    recursively -- a real run wrote a model into a `models/` subdirectory,
    which a non-recursive read would have missed entirely. Keys are the
    relative path with "/" flattened to "_" (e.g. "models/summary.json" ->
    "models_summary.json"), not a real relative path -- these keys go
    straight into python_sandbox's `input_files`, which writes each one
    with a plain `write_text()` that does NOT create parent directories, so
    a real "/" in the key would raise FileNotFoundError the moment any
    candidate organizes its output into a subdirectory. Flattening here,
    once, at the source, means every caller gets safe keys automatically
    instead of each having to remember to sanitize them.
    """
    if candidate_dir is None:
        return {}
    files = {}
    for path in sorted(candidate_dir.rglob("*")):
        if (
            path.is_file()
            and path.name not in _EXCLUDED_FROM_CANDIDATE_READ
            and path.suffix not in _BINARY_EXTENSIONS
            and "__pycache__" not in path.parts
        ):
            flat_key = str(path.relative_to(candidate_dir)).replace("/", "_")
            files[flat_key] = path.read_text(errors="replace")
    return files
