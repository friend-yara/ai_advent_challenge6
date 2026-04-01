"""
mcp/mcp_storage.py — Storage domain module for the MCP router.

Public interface:
    STORAGE_TOOLS                              list[dict]  MCP tool definitions
    dispatch_storage_tool(name, args)      -> dict     MCP result or raises

Raises:
    KeyError   — unknown tool name
    ValueError — invalid / missing arguments
    RuntimeError — file write/read failure
"""

import difflib
from pathlib import Path

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
            "Read the contents of a file in the project root directory. "
            "If the file is not found, suggests the most similar filename. "
            "Only reads files in the current directory (no subdirectories)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Name of the file to read (in project root)",
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to return (default: 200)",
                    "default": 200,
                },
            },
            "required": ["filename"],
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
        try:
            max_lines = int(max_lines)
            if max_lines < 1:
                raise ValueError("max_lines must be >= 1")
        except (TypeError, ValueError) as e:
            raise ValueError(str(e)) from e
        return _read_file(filename, max_lines)

    if tool_name != "save_to_file":
        raise KeyError(f"Unknown storage tool: {tool_name!r}")

    content = arguments.get("content")
    if content is None:
        raise ValueError("Missing required argument: 'content'")
    if not isinstance(content, str):
        raise ValueError("'content' must be a string")

    filename = (arguments.get("filename") or "forecast.txt").strip()
    if not filename:
        filename = "forecast.txt"

    return _save_to_file(filename, content)


# ---------------------------------------------------------------------------
# Internal storage logic
# ---------------------------------------------------------------------------

def _read_file(filename: str, max_lines: int) -> dict:
    """Read file contents from cwd, or suggest similar filenames."""
    # Security: reject path traversal and subdirectory access
    if "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError("Only filenames in the project root are allowed (no paths)")

    path = Path(filename)
    if path.is_file():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as e:
            raise RuntimeError(f"Failed to read file {filename!r}: {e}") from e

        total = len(lines)
        truncated = total > max_lines
        output_lines = lines[:max_lines]
        text = "\n".join(output_lines)
        if truncated:
            text += f"\n\n... (showing {max_lines} of {total} lines)"

        return {
            "content": [{"type": "text", "text": text}],
            "data": {
                "filename": filename,
                "total_lines": total,
                "truncated": truncated,
            },
        }

    # File not found — suggest similar names
    candidates = [
        f.name for f in Path(".").iterdir()
        if f.is_file() and not f.name.startswith(".")
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
