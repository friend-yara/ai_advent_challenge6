#!/usr/bin/env python3
"""
mcp/server.py — Multi-server MCP launcher.

Starts four domain MCP servers on separate ports:
  WeatherMCPHandler   → 127.0.0.1:8001  (get_forecast, summarize_forecast)
  SchedulerMCPHandler → 127.0.0.1:8002  (reminder)
  GitMCPHandler       → 127.0.0.1:8004  (git_branch, git_log, git_diff)
  StorageMCPHandler   → 127.0.0.1:8003  (save_to_file)

Usage:
    python mcp/server.py

All servers except storage run in daemon threads; storage runs in the main thread.
Press Ctrl+C to stop all servers.
"""

import os
import sys
import threading
from http.server import HTTPServer

# Ensure project root is on sys.path when run directly as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.base import (
    ERR_INVALID_PARAMS,
    ERR_METHOD_NOT_FOUND,
    ERR_SERVER_ERROR,
    MCPBaseHandler,
)
from mcp.mcp_scheduler import SCHEDULER_TOOLS, dispatch_scheduler_tool, init_scheduler
from mcp.mcp_storage import STORAGE_TOOLS, dispatch_storage_tool
from mcp.mcp_git import GIT_TOOLS, dispatch_git_tool
from mcp.mcp_weather import WEATHER_TOOLS, dispatch_weather_tool

# ---------------------------------------------------------------------------
# Domain handlers
# ---------------------------------------------------------------------------


class WeatherMCPHandler(MCPBaseHandler):
    """MCP handler for weather forecast tools."""

    SERVER_NAME    = "local-mcp-weather"
    SERVER_VERSION = "0.1"
    INSTRUCTIONS   = "Weather forecast server. Tools: get_forecast, summarize_forecast."
    TOOLS          = WEATHER_TOOLS

    def _get_tools(self) -> list[dict]:
        return WEATHER_TOOLS

    def _dispatch_tool(self, req_id, tool_name: str,
                       arguments: dict, session_id: str):
        try:
            result = dispatch_weather_tool(tool_name, arguments)
            self._send_rpc_ok(req_id, result)
        except KeyError as e:
            self._send_rpc_error(req_id, ERR_METHOD_NOT_FOUND, str(e))
        except ValueError as e:
            self._send_rpc_error(req_id, ERR_INVALID_PARAMS, str(e))
        except Exception as e:
            self._send_rpc_error(req_id, ERR_SERVER_ERROR, f"Tool error: {e}")


class SchedulerMCPHandler(MCPBaseHandler):
    """MCP handler for scheduler/reminder tools."""

    SERVER_NAME    = "local-mcp-scheduler"
    SERVER_VERSION = "0.1"
    INSTRUCTIONS   = "Scheduler server. Tools: reminder."
    TOOLS          = SCHEDULER_TOOLS

    def _get_tools(self) -> list[dict]:
        return SCHEDULER_TOOLS

    def _dispatch_tool(self, req_id, tool_name: str,
                       arguments: dict, session_id: str):
        try:
            result = dispatch_scheduler_tool(tool_name, arguments)
            self._send_rpc_ok(req_id, result)
        except KeyError as e:
            self._send_rpc_error(req_id, ERR_METHOD_NOT_FOUND, str(e))
        except ValueError as e:
            self._send_rpc_error(req_id, ERR_INVALID_PARAMS, str(e))
        except Exception as e:
            self._send_rpc_error(req_id, ERR_SERVER_ERROR, f"Tool error: {e}")


class GitMCPHandler(MCPBaseHandler):
    """MCP handler for git information tools."""

    SERVER_NAME    = "local-mcp-git"
    SERVER_VERSION = "0.1"
    INSTRUCTIONS   = "Git information server. Tools: git_branch, git_log, git_diff."
    TOOLS          = GIT_TOOLS

    def _get_tools(self) -> list[dict]:
        return GIT_TOOLS

    def _dispatch_tool(self, req_id, tool_name: str,
                       arguments: dict, session_id: str):
        try:
            result = dispatch_git_tool(tool_name, arguments)
            self._send_rpc_ok(req_id, result)
        except KeyError as e:
            self._send_rpc_error(req_id, ERR_METHOD_NOT_FOUND, str(e))
        except ValueError as e:
            self._send_rpc_error(req_id, ERR_INVALID_PARAMS, str(e))
        except Exception as e:
            self._send_rpc_error(req_id, ERR_SERVER_ERROR, f"Tool error: {e}")


class StorageMCPHandler(MCPBaseHandler):
    """MCP handler for file storage tools."""

    SERVER_NAME    = "local-mcp-storage"
    SERVER_VERSION = "0.1"
    INSTRUCTIONS   = "File storage server. Tools: save_to_file."
    TOOLS          = STORAGE_TOOLS

    def _get_tools(self) -> list[dict]:
        return STORAGE_TOOLS

    def _dispatch_tool(self, req_id, tool_name: str,
                       arguments: dict, session_id: str):
        try:
            result = dispatch_storage_tool(tool_name, arguments)
            self._send_rpc_ok(req_id, result)
        except KeyError as e:
            self._send_rpc_error(req_id, ERR_METHOD_NOT_FOUND, str(e))
        except ValueError as e:
            self._send_rpc_error(req_id, ERR_INVALID_PARAMS, str(e))
        except Exception as e:
            self._send_rpc_error(req_id, ERR_SERVER_ERROR, f"Tool error: {e}")


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------

_HOST = "127.0.0.1"

_SERVERS = [
    (WeatherMCPHandler,   8001),
    (SchedulerMCPHandler, 8002),
    (GitMCPHandler,       8004),
    (StorageMCPHandler,   8003),
]


def _start_server(handler_class: type, port: int) -> None:
    """Start an HTTPServer in the current thread (blocking)."""
    server = HTTPServer((_HOST, port), handler_class)
    server.serve_forever()


if __name__ == "__main__":
    init_scheduler()

    print("Starting MCP domain servers:")
    for handler_class, port in _SERVERS:
        url = f"http://{_HOST}:{port}/mcp"
        tool_names = ", ".join(t["name"] for t in handler_class.TOOLS)
        print(f"  {handler_class.SERVER_NAME}: {url}  [{tool_names}]")

    # Start weather and scheduler as daemon threads
    for handler_class, port in _SERVERS[:-1]:
        t = threading.Thread(
            target=_start_server,
            args=(handler_class, port),
            daemon=True,
            name=f"{handler_class.SERVER_NAME}-thread",
        )
        t.start()

    # Run storage in main thread (blocking — keeps the process alive)
    print("Press Ctrl+C to stop.")
    try:
        _start_server(StorageMCPHandler, 8003)
    except KeyboardInterrupt:
        print("\nStopped.")
