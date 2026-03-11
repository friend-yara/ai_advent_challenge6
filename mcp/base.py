#!/usr/bin/env python3
"""
mcp/base.py — Reusable MCP Streamable HTTP base handler.

Provides the full MCP protocol scaffold (2024-11-05) so individual
server modules only need to declare their tools and handle tool calls.

Usage — implement a new MCP server:

    from mcp.base import MCPBaseHandler, run_server

    class MyHandler(MCPBaseHandler):
        SERVER_NAME = "my-server"
        SERVER_VERSION = "0.1"
        INSTRUCTIONS = "What this server does."

        def _get_tools(self) -> list[dict]:
            return [MY_TOOL_DEFINITION]

        def _dispatch_tool(self, req_id, tool_name: str,
                           arguments: dict, session_id: str):
            if tool_name == "my_tool":
                result = do_something(arguments)
                self._send_rpc_ok(req_id, {"content": [{"type": "text", "text": result}]})
            else:
                self._send_rpc_error(req_id, ERR_METHOD_NOT_FOUND,
                                     f"Unknown tool: {tool_name!r}")

    if __name__ == "__main__":
        run_server(MyHandler)
"""

import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer

# ---------------------------------------------------------------------------
# JSON-RPC 2.0 error codes
# ---------------------------------------------------------------------------

ERR_INVALID_REQUEST  = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS   = -32602
ERR_SERVER_ERROR     = -32000

# ---------------------------------------------------------------------------
# MCP protocol constants
# ---------------------------------------------------------------------------

MCP_PROTOCOL_VERSION = "2024-11-05"

# ---------------------------------------------------------------------------
# Session store  (module-level, process lifetime, shared across all handlers)
# ---------------------------------------------------------------------------

# sid -> True  means the session completed the notifications/initialized step
_sessions: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def rpc_ok(req_id, result: dict | list) -> dict:
    """Build a JSON-RPC 2.0 success response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_err(req_id, code: int, message: str) -> dict:
    """Build a JSON-RPC 2.0 error response."""
    return {"jsonrpc": "2.0", "id": req_id,
            "error": {"code": code, "message": message}}


# ---------------------------------------------------------------------------
# Base handler
# ---------------------------------------------------------------------------

class MCPBaseHandler(BaseHTTPRequestHandler):
    """
    Reusable MCP Streamable HTTP request handler.

    Subclasses must override:
      SERVER_NAME    : str  — reported in initialize response
      SERVER_VERSION : str  — reported in initialize response
      INSTRUCTIONS   : str  — natural-language description for LLM clients
      _get_tools()   — return list of MCP tool definition dicts
      _dispatch_tool(req_id, tool_name, arguments, session_id)
                     — handle a tools/call request and send the response

    The base class handles all protocol mechanics: session lifecycle,
    initialize, notifications/initialized, tools/list, routing, and
    all HTTP/JSON-RPC plumbing.
    """

    SERVER_NAME:    str = "local-mcp-server"
    SERVER_VERSION: str = "0.1"
    INSTRUCTIONS:   str = "Local MCP server."

    # ------------------------------------------------------------------
    # Logging — silenced; servers can override to enable
    # ------------------------------------------------------------------

    def log_message(self, format, *args):  # noqa: A002
        pass

    # ------------------------------------------------------------------
    # HTTP entry point
    # ------------------------------------------------------------------

    def do_POST(self):
        if self.path != "/mcp":
            self._send_json(404, {"error": "Not found"})
            return

        # Enforce Accept header per MCP spec
        accept = self.headers.get("Accept", "")
        if "application/json" not in accept and "text/event-stream" not in accept:
            self._send_rpc_error(
                None, ERR_INVALID_REQUEST,
                "Not Acceptable: Client must accept application/json "
                "or text/event-stream",
                status=406,
            )
            return

        # Read body
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            self._send_rpc_error(None, ERR_INVALID_REQUEST,
                                 f"Invalid JSON: {e}")
            return

        method     = body.get("method", "")
        req_id     = body.get("id")
        session_id = self.headers.get("Mcp-Session-Id", "").strip()
        params     = body.get("params") or {}

        if method == "initialize":
            self._handle_initialize(req_id, params)
        elif method == "notifications/initialized":
            self._handle_notifications_initialized(session_id)
        elif method == "tools/list":
            self._handle_tools_list(req_id, session_id)
        elif method == "tools/call":
            self._handle_tools_call(req_id, session_id, params)
        else:
            self._send_rpc_error(req_id, ERR_METHOD_NOT_FOUND,
                                 f"Method not found: {method!r}")

    # ------------------------------------------------------------------
    # MCP protocol handlers
    # ------------------------------------------------------------------

    def _handle_initialize(self, req_id, params: dict):
        """Create a new session and return server capabilities."""
        sid = uuid.uuid4().hex
        _sessions[sid] = False   # pending notifications/initialized

        result = {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {
                "name": self.SERVER_NAME,
                "version": self.SERVER_VERSION,
            },
            "instructions": self.INSTRUCTIONS,
        }
        self._send_rpc_ok(req_id, result,
                          extra_headers={"Mcp-Session-Id": sid})

    def _handle_notifications_initialized(self, session_id: str):
        """Mark session as fully initialized. Respond 202 with no body."""
        if session_id:
            _sessions[session_id] = True
        self.send_response(202)
        self.end_headers()

    def _handle_tools_list(self, req_id, session_id: str):
        """Return the list of available tools from the subclass."""
        if self._require_session(req_id, session_id):
            return
        self._send_rpc_ok(req_id, {"tools": self._get_tools()})

    def _handle_tools_call(self, req_id, session_id: str, params: dict):
        """Validate session and delegate to subclass _dispatch_tool."""
        if self._require_session(req_id, session_id):
            return
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}
        self._dispatch_tool(req_id, tool_name, arguments, session_id)

    # ------------------------------------------------------------------
    # Subclass interface (must override)
    # ------------------------------------------------------------------

    def _get_tools(self) -> list[dict]:
        """Return list of MCP tool definition dicts for tools/list."""
        return []

    def _dispatch_tool(self, req_id, tool_name: str,
                       arguments: dict, session_id: str):
        """Handle a tools/call request. Subclass must send response."""
        self._send_rpc_error(req_id, ERR_METHOD_NOT_FOUND,
                             f"Unknown tool: {tool_name!r}")

    # ------------------------------------------------------------------
    # Session guard
    # ------------------------------------------------------------------

    def _require_session(self, req_id, session_id: str) -> bool:
        """
        Verify session exists and is initialized.
        Sends error and returns True if invalid; returns False if valid.
        """
        if not session_id:
            self._send_rpc_error(req_id, ERR_INVALID_REQUEST,
                                 "Mcp-Session-Id header required")
            return True
        if session_id not in _sessions:
            self._send_rpc_error(req_id, ERR_INVALID_REQUEST,
                                 "Invalid or expired session. Re-initialize.")
            return True
        if not _sessions[session_id]:
            self._send_rpc_error(req_id, ERR_INVALID_REQUEST,
                                 "Session not initialized. "
                                 "Send notifications/initialized first.")
            return True
        return False

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _send_rpc_ok(self, req_id, result,
                     extra_headers: dict | None = None):
        """Send a 200 JSON-RPC success response."""
        self._send_json(200, rpc_ok(req_id, result),
                        extra_headers=extra_headers)

    def _send_rpc_error(self, req_id, code: int, message: str,
                        status: int = 200):
        """Send a JSON-RPC error response."""
        self._send_json(status, rpc_err(req_id, code, message))

    def _send_json(self, status: int, data: dict,
                   extra_headers: dict | None = None):
        """Serialise *data* as JSON and write the full HTTP response."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_server(handler_class: type, default_host: str = "127.0.0.1",
               default_port: int = 8000):
    """
    Parse --host / --port CLI args and start an HTTPServer.
    Call from the module's __main__ block.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Local MCP server ({handler_class.SERVER_NAME})"
    )
    parser.add_argument("--host", default=default_host,
                        help=f"Bind address (default: {default_host})")
    parser.add_argument("--port", type=int, default=default_port,
                        help=f"Port (default: {default_port})")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), handler_class)
    url = f"http://{args.host}:{args.port}/mcp"
    # Get tool names without instantiating the handler
    try:
        tools = [t["name"] for t in handler_class.TOOLS]
    except AttributeError:
        tools = []
    print(f"MCP server '{handler_class.SERVER_NAME}' running at {url}")
    print(f"Tools: {', '.join(tools) or '(none)'}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
