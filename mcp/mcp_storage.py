"""
mcp/mcp_storage.py — Storage domain module for the MCP router.

Public interface:
    STORAGE_TOOLS                              list[dict]  MCP tool definitions
    dispatch_storage_tool(name, args)      -> dict     MCP result or raises

Raises:
    KeyError   — unknown tool name
    ValueError — invalid / missing arguments
    RuntimeError — file write failure
"""

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
    }
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
