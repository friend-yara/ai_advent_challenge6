#!/usr/bin/env python3
"""
mcp/server.py — MCP router: single endpoint, multiple domain modules.

Serves all tools via one /mcp endpoint by delegating to domain modules.

Usage:
    python mcp/server.py                   # 127.0.0.1:8000
    python mcp/server.py --port 9000
    python mcp/server.py --host 0.0.0.0 --port 8080

MCP endpoint: http://127.0.0.1:8000/mcp

Extending — add a new domain module (e.g. mcp/mcp_notes.py):
    1. Create mcp/mcp_notes.py with NOTES_TOOLS and dispatch_notes_tool()
    2. Import them below
    3. Add to ALL_TOOLS and _DISPATCH (2 lines)
"""

import os
import sys

# Ensure project root is on sys.path when run directly as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.base import (
    ERR_INVALID_PARAMS,
    ERR_METHOD_NOT_FOUND,
    ERR_SERVER_ERROR,
    MCPBaseHandler,
    run_server,
)
from mcp.mcp_scheduler import SCHEDULER_TOOLS, dispatch_scheduler_tool, init_scheduler
from mcp.mcp_storage import STORAGE_TOOLS, dispatch_storage_tool
from mcp.mcp_weather import WEATHER_TOOLS, dispatch_weather_tool

# ---------------------------------------------------------------------------
# Tool registry — add new domain modules here
# ---------------------------------------------------------------------------

ALL_TOOLS: list[dict] = [
    *WEATHER_TOOLS,
    *SCHEDULER_TOOLS,
    *STORAGE_TOOLS,
]

_DISPATCH: dict[str, object] = {
    **{t["name"]: dispatch_weather_tool   for t in WEATHER_TOOLS},
    **{t["name"]: dispatch_scheduler_tool for t in SCHEDULER_TOOLS},
    **{t["name"]: dispatch_storage_tool   for t in STORAGE_TOOLS},
}

# ---------------------------------------------------------------------------
# Router handler
# ---------------------------------------------------------------------------

class RouterMCPHandler(MCPBaseHandler):
    """
    Single MCP handler that routes tools/call to the appropriate domain module.
    Protocol mechanics (sessions, initialize, tools/list) are handled by MCPBaseHandler.
    """

    SERVER_NAME    = "local-mcp-router"
    SERVER_VERSION = "0.1"
    INSTRUCTIONS   = (
        "Local MCP router. "
        "Available tools: weather forecast (get_forecast), "
        "forecast summary (summarize_forecast), "
        "scheduler reminders (reminder), "
        "file storage (save_to_file)."
    )
    TOOLS = ALL_TOOLS   # used by run_server() for startup display

    def _get_tools(self) -> list[dict]:
        return ALL_TOOLS

    def _dispatch_tool(self, req_id, tool_name: str,
                       arguments: dict, session_id: str):
        dispatch_fn = _DISPATCH.get(tool_name)
        if dispatch_fn is None:
            self._send_rpc_error(req_id, ERR_METHOD_NOT_FOUND,
                                 f"Unknown tool: {tool_name!r}")
            return
        try:
            result = dispatch_fn(tool_name, arguments)
            self._send_rpc_ok(req_id, result)
        except KeyError as e:
            self._send_rpc_error(req_id, ERR_METHOD_NOT_FOUND, str(e))
        except ValueError as e:
            self._send_rpc_error(req_id, ERR_INVALID_PARAMS, str(e))
        except Exception as e:
            self._send_rpc_error(req_id, ERR_SERVER_ERROR, f"Tool error: {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_scheduler()
    run_server(RouterMCPHandler)
