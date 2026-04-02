"""
mcp/mcp_storage.py — Storage domain module for the MCP router.

Public interface:
    STORAGE_TOOLS                              list[dict]  MCP tool definitions
    dispatch_storage_tool(name, args)      -> dict     MCP result or raises

Tools:
    save_to_file   — write text to a file (dangerous)
    read_file      — read file contents with offset support
    list_files     — list project files matching a glob pattern
    grep_files     — search file contents by regex
    edit_file      — replace text fragment in a file (dangerous)

Raises:
    KeyError   — unknown tool name
    ValueError — invalid / missing arguments
    RuntimeError — file write/read failure
"""

import difflib
import re
from pathlib import Path

_PROJECT_ROOT = Path(".").resolve()

_SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".mypy_cache"}

# ---------------------------------------------------------------------------
# Tool definitions (MCP-compatible schema)
# ---------------------------------------------------------------------------

STORAGE_TOOLS: list[dict] = [
    {
        "name": "save_to_file",
        "title": "Save text to file",
        "description": (
            "Save a text string to a file in the current working directory. "
            "Use this to persist tool output (e.g. weather summaries) to disk."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Text content to write to the file",
                },
                "filename": {
                    "type": "string",
                    "description": "Target filename (default: 'forecast.txt')",
                    "default": "forecast.txt",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "read_file",
        "title": "Read file contents",
        "description": (
            "Read the contents of a file in the project directory (including subdirectories). "
            "If the file is not found, suggests the most similar filename. "
            "Supports offset to read from a specific line number."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Path to file relative to project root (e.g. 'agent.py' or 'mcp/server.py')",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: 200)",
                    "default": 200,
                },
                "offset": {
                    "type": "integer",
                    "description": "Line number to start reading from, 0-based (default: 0)",
                    "default": 0,
                },
            },
            "required": ["filename"],
        },
    },
    {
        "name": "list_files",
        "title": "List project files",
        "description": (
            "List files in the project directory matching a glob pattern. "
            "Use to explore project structure before reading specific files. "
            "Returns file paths with sizes. Skips .git, __pycache__, .venv."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match (default: '*.py')",
                    "default": "*.py",
                },
                "directory": {
                    "type": "string",
                    "description": "Subdirectory to search in (default: '.' = project root)",
                    "default": ".",
                },
            },
        },
    },
    {
        "name": "grep_files",
        "title": "Search file contents",
        "description": (
            "Search for a regex pattern across project files. "
            "Returns matching lines with file path and line numbers. "
            "Use to find where a function, class, or pattern is used."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for",
                },
                "file_glob": {
                    "type": "string",
                    "description": "File glob filter (default: '*.py')",
                    "default": "*.py",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return (default: 20)",
                    "default": 20,
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "edit_file",
        "title": "Edit file contents",
        "description": (
            "Replace a specific text fragment in a project file. "
            "Returns a unified diff of the change. "
            "DANGEROUS: modifies files on disk, requires user approval."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Path to file relative to project root",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text fragment to find and replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text",
                },
            },
            "required": ["filename", "old_text", "new_text"],
        },
    },
]

# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------

def dispatch_storage_tool(tool_name: str, arguments: dict) -> dict:
    """
    Dispatch a tools/call request to the appropriate storage tool.

    Returns an MCP-compatible result dict:
        {"content": [{"type": "text", "text": "..."}], "data": {...}}

    Raises:
        KeyError    if tool_name is not a known storage tool
        ValueError  if required arguments are missing or invalid
        RuntimeError on file write errors
    """
    if tool_name == "read_file":
        filename = (arguments.get("filename") or "").strip()
        if not filename:
            raise ValueError("Missing required argument: 'filename'")
        max_lines = arguments.get("max_lines", 200)
        offset = arguments.get("offset", 0)
        try:
            max_lines = int(max_lines)
            if max_lines < 1:
                raise ValueError("max_lines must be >= 1")
            offset = int(offset)
            if offset < 0:
                raise ValueError("offset must be >= 0")
        except (TypeError, ValueError) as e:
            raise ValueError(str(e)) from e
        return _read_file(filename, max_lines, offset)

    if tool_name == "save_to_file":
        content = arguments.get("content")
        if content is None:
            raise ValueError("Missing required argument: 'content'")
        if not isinstance(content, str):
            raise ValueError("'content' must be a string")
        filename = (arguments.get("filename") or "forecast.txt").strip()
        if not filename:
            filename = "forecast.txt"
        return _save_to_file(filename, content)

    if tool_name == "list_files":
        pattern = (arguments.get("pattern") or "*.py").strip()
        directory = (arguments.get("directory") or ".").strip()
        return _list_files(pattern, directory)

    if tool_name == "grep_files":
        pattern = (arguments.get("pattern") or "").strip()
        if not pattern:
            raise ValueError("Missing required argument: 'pattern'")
        file_glob = (arguments.get("file_glob") or "*.py").strip()
        max_results = arguments.get("max_results", 20)
        try:
            max_results = int(max_results)
            if max_results < 1:
                raise ValueError("max_results must be >= 1")
        except (TypeError, ValueError) as e:
            raise ValueError(str(e)) from e
        return _grep_files(pattern, file_glob, max_results)

    if tool_name == "edit_file":
        filename = (arguments.get("filename") or "").strip()
        if not filename:
            raise ValueError("Missing required argument: 'filename'")
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        if old_text is None:
            raise ValueError("Missing required argument: 'old_text'")
        if new_text is None:
            raise ValueError("Missing required argument: 'new_text'")
        return _edit_file(filename, old_text, new_text)

    raise KeyError(f"Unknown storage tool: {tool_name!r}")


# ---------------------------------------------------------------------------
# Internal storage logic
# ---------------------------------------------------------------------------

def _check_inside_project(path: Path) -> None:
    """Raise ValueError if resolved path escapes project root."""
    resolved = path.resolve()
    if not resolved.is_relative_to(_PROJECT_ROOT):
        raise ValueError(f"Access denied: path {str(path)!r} is outside the project root")


def _should_skip(path: Path) -> bool:
    """Return True if path is inside a directory we want to skip."""
    return any(part in _SKIP_DIRS for part in path.parts)


def _read_file(filename: str, max_lines: int, offset: int = 0) -> dict:
    """Read file contents, supporting subdirectories and offset."""
    if ".." in Path(filename).parts:
        raise ValueError("Path traversal ('..') is not allowed")

    path = Path(filename)
    _check_inside_project(path)

    if path.is_file():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            raise RuntimeError(f"Failed to read file {filename!r}: {e}") from e

        total = len(lines)
        selected = lines[offset:offset + max_lines]
        truncated = (offset + max_lines) < total
        text = "\n".join(f"{offset + i + 1:4d} | {line}" for i, line in enumerate(selected))
        if truncated:
            text += f"\n\n... (showing lines {offset + 1}-{offset + len(selected)} of {total})"

        return {
            "content": [{"type": "text", "text": text}],
            "data": {
                "filename": filename,
                "total_lines": total,
                "offset": offset,
                "lines_returned": len(selected),
                "truncated": truncated,
            },
        }

    # File not found — suggest similar names (recursive search)
    candidates = [
        str(f) for f in Path(".").rglob("*")
        if f.is_file() and not _should_skip(f) and not f.name.startswith(".")
    ]
    matches = difflib.get_close_matches(filename, candidates, n=3, cutoff=0.4)

    if matches:
        suggestions = ", ".join(matches)
        text = (
            f"Файл '{filename}' не найден.\n"
            f"Похожие файлы: {suggestions}\n"
            f"Попробуйте запросить один из них."
        )
    else:
        text = f"Файл '{filename}' не найден, и похожих имён не обнаружено."

    return {
        "content": [{"type": "text", "text": text}],
        "data": {"filename": filename, "found": False, "suggestions": matches},
    }


def _list_files(pattern: str, directory: str) -> dict:
    """List files matching a glob pattern under the given directory."""
    base = Path(directory)
    _check_inside_project(base)

    results = []
    for f in sorted(base.rglob(pattern)):
        if not f.is_file() or _should_skip(f):
            continue
        try:
            size = f.stat().st_size
        except OSError:
            size = 0
        results.append(f"{str(f):<50s}  {size:>8d} bytes")
        if len(results) >= 50:
            break

    if results:
        text = f"Found {len(results)} file(s) matching '{pattern}' in '{directory}':\n\n"
        text += "\n".join(results)
    else:
        text = f"No files matching '{pattern}' found in '{directory}'."

    return {
        "content": [{"type": "text", "text": text}],
        "data": {"pattern": pattern, "directory": directory, "count": len(results)},
    }


def _grep_files(pattern: str, file_glob: str, max_results: int) -> dict:
    """Search for a regex pattern across project files."""
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}") from e

    matches = []
    files_searched = 0
    for filepath in sorted(Path(".").rglob(file_glob)):
        if not filepath.is_file() or _should_skip(filepath):
            continue
        files_searched += 1
        try:
            lines = filepath.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(f"{filepath}:{lineno}: {line.strip()}")
                if len(matches) >= max_results:
                    break
        if len(matches) >= max_results:
            break

    if matches:
        text = f"Found {len(matches)} match(es) for /{pattern}/ in {files_searched} file(s):\n\n"
        text += "\n".join(matches)
        if len(matches) >= max_results:
            text += f"\n\n... (truncated at {max_results} results)"
    else:
        text = f"No matches for /{pattern}/ in {files_searched} file(s) matching '{file_glob}'."

    return {
        "content": [{"type": "text", "text": text}],
        "data": {
            "pattern": pattern,
            "file_glob": file_glob,
            "match_count": len(matches),
            "files_searched": files_searched,
        },
    }


def _edit_file(filename: str, old_text: str, new_text: str) -> dict:
    """Replace the first occurrence of old_text with new_text in a file."""
    if ".." in Path(filename).parts:
        raise ValueError("Path traversal ('..') is not allowed")

    path = Path(filename)
    _check_inside_project(path)

    if not path.is_file():
        raise ValueError(f"File not found: {filename!r}")

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Failed to read file {filename!r}: {e}") from e

    if old_text not in content:
        raise ValueError(
            f"old_text not found in {filename!r}. "
            "Make sure the text matches exactly (including whitespace)."
        )

    new_content = content.replace(old_text, new_text, 1)

    # Generate unified diff
    old_lines = content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = "".join(difflib.unified_diff(old_lines, new_lines, fromfile=filename, tofile=filename))

    try:
        path.write_text(new_content, encoding="utf-8")
    except OSError as e:
        raise RuntimeError(f"Failed to write file {filename!r}: {e}") from e

    text = f"edit_file: Modified {filename}\n\n{diff}"
    return {
        "content": [{"type": "text", "text": text}],
        "data": {"filename": filename, "diff": diff},
    }


def _save_to_file(filename: str, content: str) -> dict:
    """Write content to filename in cwd, return MCP result."""
    try:
        path = Path(filename)
        path.write_text(content, encoding="utf-8")
        byte_count = len(content.encode("utf-8"))
    except OSError as e:
        raise RuntimeError(f"Failed to write file {filename!r}: {e}") from e

    text = (
        f"save_to_file: Saving result to {filename}\n"
        f"Saved {byte_count} bytes to {path.resolve()}"
    )
    return {
        "content": [{"type": "text", "text": text}],
        "data": {
            "filename": filename,
            "path": str(path.resolve()),
            "bytes": byte_count,
        },
    }
