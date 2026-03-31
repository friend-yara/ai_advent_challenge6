"""
mcp/mcp_git.py — Git domain module for the MCP router.

Public interface:
    GIT_TOOLS                          list[dict]  MCP tool definitions
    dispatch_git_tool(name, args)      -> dict     MCP result or raises

Raises:
    KeyError   — unknown tool name
    ValueError — invalid / missing arguments
    RuntimeError — git command failure
"""

import subprocess

# ---------------------------------------------------------------------------
# Tool definitions (MCP-compatible schema)
# ---------------------------------------------------------------------------

GIT_TOOLS: list[dict] = [
    {
        "name": "git_branch",
        "title": "Git branches",
        "description": (
            "Show current git branch. "
            "Set list_all=true to include all local and remote branches."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "list_all": {
                    "type": "boolean",
                    "description": "If true, list all branches (local + remote)",
                    "default": False,
                },
            },
            "required": [],
        },
    },
    {
        "name": "git_log",
        "title": "Git log",
        "description": "Show recent git commits (one-line format).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "description": "Number of commits to show",
                    "default": 10,
                },
            },
            "required": [],
        },
    },
    {
        "name": "git_diff",
        "title": "Git diff",
        "description": (
            "Show current changes in the working directory. "
            "Set staged=true to show only staged changes."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "If true, show only staged (--cached) changes",
                    "default": False,
                },
            },
            "required": [],
        },
    },
    {
        "name": "git_diff_branch",
        "title": "Git diff between branches",
        "description": (
            "Show diff between two branches or refs. "
            "Useful for reviewing PR changes locally."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "base": {
                    "type": "string",
                    "description": "Base branch/ref (e.g. 'main')",
                    "default": "main",
                },
                "head": {
                    "type": "string",
                    "description": "Head branch/ref (e.g. 'feature-x'). Defaults to HEAD.",
                    "default": "HEAD",
                },
                "name_only": {
                    "type": "boolean",
                    "description": "If true, show only changed file names",
                    "default": False,
                },
            },
            "required": [],
        },
    },
]

# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------

def dispatch_git_tool(tool_name: str, arguments: dict) -> dict:
    """
    Dispatch a tools/call request to the appropriate git tool.

    Returns an MCP-compatible result dict:
        {"content": [{"type": "text", "text": "..."}]}

    Raises:
        KeyError    if tool_name is not a known git tool
        ValueError  if arguments are invalid
        RuntimeError on git command failure
    """
    if tool_name == "git_branch":
        return _git_branch(arguments.get("list_all", False))
    if tool_name == "git_log":
        return _git_log(arguments.get("count", 10))
    if tool_name == "git_diff":
        return _git_diff(arguments.get("staged", False))
    if tool_name == "git_diff_branch":
        return _git_diff_branch(
            arguments.get("base", "main"),
            arguments.get("head", "HEAD"),
            arguments.get("name_only", False),
        )
    raise KeyError(f"Unknown git tool: {tool_name!r}")


# ---------------------------------------------------------------------------
# Internal git helpers
# ---------------------------------------------------------------------------

_TIMEOUT = 10
_MAX_DIFF_CHARS = 4000


def _run_git(*args: str) -> str:
    """Run a git command and return stdout. Raises RuntimeError on failure."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        timeout=_TIMEOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def _git_branch(list_all: bool) -> dict:
    """Return current branch and optionally all branches."""
    current = _run_git("branch", "--show-current")
    text = f"Current branch: {current}"
    if list_all:
        all_branches = _run_git("branch", "-a", "--no-color")
        text += f"\n\nAll branches:\n{all_branches}"
    return {"content": [{"type": "text", "text": text}]}


def _git_log(count: int) -> dict:
    """Return last N commits in one-line format."""
    count = max(1, min(count, 100))
    log = _run_git("log", f"--oneline", f"-{count}", "--no-color")
    if not log:
        log = "(no commits)"
    return {"content": [{"type": "text", "text": log}]}


def _git_diff(staged: bool) -> dict:
    """Return working directory or staged diff."""
    args = ["diff", "--no-color"]
    if staged:
        args.append("--staged")
    try:
        diff = _run_git(*args)
    except RuntimeError:
        diff = ""
    if not diff:
        label = "staged" if staged else "working directory"
        return {"content": [{"type": "text", "text": f"No changes in {label}."}]}
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + "\n\n... (truncated)"
    return {"content": [{"type": "text", "text": diff}]}


def _git_diff_branch(base: str, head: str, name_only: bool) -> dict:
    """Return diff between two branches/refs."""
    args = ["diff", f"{base}...{head}", "--no-color"]
    if name_only:
        args.append("--name-only")
    try:
        diff = _run_git(*args)
    except RuntimeError as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}]}
    if not diff:
        return {"content": [{"type": "text", "text": f"No differences between {base} and {head}."}]}
    if len(diff) > _MAX_DIFF_CHARS:
        diff = diff[:_MAX_DIFF_CHARS] + "\n\n... (truncated)"
    return {"content": [{"type": "text", "text": diff}]}
