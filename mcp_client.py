#!/usr/bin/env python3
"""
mcp_client.py — MCP (Model Context Protocol) client over Streamable HTTP.

Implements the 2024-11-05 MCP Streamable HTTP transport using only the
`requests` library (already a project dependency — no new packages needed).

Session lifecycle per call (stateless — no session is reused):
  1. POST <url>  initialize           → captures Mcp-Session-Id from headers
  2. POST <url>  notifications/initialized  (session header, no id)  → 202
  3. POST <url>  tools/list or tools/call  (session header)

Server specs are loaded from *.yaml files in a configurable directory
(default: tools/). Each file must contain at minimum: name, url.

Tool schemas are fetched and cached at load() time so the orchestrator can
pass them to the LLM as OpenAI function definitions without extra round-trips.

Usage:
    mcp = MCPClient("tools/")
    mcp.load()

    # Inspect tools
    tools = mcp.list_tools("vkusvill")
    llm_tools = mcp.all_tools_for_llm()

    # Call a tool
    result = mcp.call_tool("weather", "get_forecast", {"place": "London", "days": 3})
    print(result)
"""

import sys
from pathlib import Path

import requests

try:
    import yaml
except ImportError as e:
    raise SystemExit("ERROR: Install YAML support: pip install pyyaml") from e

# MCP protocol version we advertise
_PROTOCOL_VERSION = "2024-11-05"
_CLIENT_INFO = {"name": "llm-agent-cli", "version": "0.1"}
_TIMEOUT = 15   # seconds per HTTP request


class MCPClient:
    """
    Minimal MCP client for the Streamable HTTP transport.

    Loads server specs from a directory of *.yaml files.
    Tool schemas are fetched and cached at load() time for LLM function calling.
    Each individual call (list_tools, call_tool) opens a fresh stateless session.
    """

    def __init__(self, servers_dir: str | Path = "tools"):
        """Initialize with path to server spec directory."""
        self.servers_dir = Path(servers_dir)
        self._servers: dict[str, dict] = {}          # name -> {name, url, description}
        self._tools_cache: dict[str, list[dict]] = {} # server_name -> tools list

    def load(self):
        """
        Scan servers_dir for *.yaml files, load each as a server spec,
        and pre-fetch tool schemas from reachable servers.
        Servers that are unreachable at startup are skipped with a WARN.
        """
        self._servers = {}
        self._tools_cache = {}

        if not self.servers_dir.is_dir():
            print(
                f"[WARN] MCP servers dir not found: {self.servers_dir}",
                file=sys.stderr,
            )
            return

        for path in sorted(self.servers_dir.glob("*.yaml")):
            spec = self._load_file(path)
            if spec is None:
                continue
            self._servers[spec["name"]] = spec

            # Pre-fetch tools (skip unreachable servers silently)
            try:
                tools = self._fetch_tools(spec["url"])
                self._tools_cache[spec["name"]] = tools
            except requests.exceptions.ConnectionError:
                print(
                    f"[WARN] MCP server '{spec['name']}' unreachable at startup "
                    f"— tools not cached ({spec['url']})",
                    file=sys.stderr,
                )
                self._tools_cache[spec["name"]] = []
            except Exception as e:
                print(
                    f"[WARN] Could not fetch tools from '{spec['name']}': {e}",
                    file=sys.stderr,
                )
                self._tools_cache[spec["name"]] = []

    def _load_file(self, path: Path) -> dict | None:
        """Parse a single server spec YAML file. Returns None on error."""
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            print(f"[WARN] Cannot parse MCP spec {path}: {e}", file=sys.stderr)
            return None
        if "name" not in data or "url" not in data:
            print(
                f"[WARN] MCP spec {path} missing required fields (name, url)",
                file=sys.stderr,
            )
            return None
        return {
            "name": str(data["name"]),
            "url": str(data["url"]),
            "description": str(data.get("description", "")),
        }

    # ---------------- Public API ----------------

    def list_servers(self) -> list[dict]:
        """Return all loaded server specs sorted by name."""
        return sorted(self._servers.values(), key=lambda s: s["name"])

    def get_server(self, name: str) -> dict | None:
        """Return server spec by name, or None if not found."""
        return self._servers.get(name)

    def list_tools(self, server_name: str) -> list[dict]:
        """
        Connect to the named MCP server and return its tool list (fresh session).
        Returns empty list on any error (error already printed to stderr).
        """
        spec = self._servers.get(server_name)
        if spec is None:
            known = ", ".join(self._servers.keys()) or "(none loaded)"
            print(
                f"ERROR: MCP server '{server_name}' not found. "
                f"Known servers: {known}"
            )
            return []

        try:
            return self._fetch_tools(spec["url"])
        except requests.exceptions.ConnectionError as e:
            print(f"ERROR: Cannot connect to MCP server '{server_name}': {e}")
            return []
        except requests.exceptions.Timeout:
            print(f"ERROR: Timeout connecting to MCP server '{server_name}'")
            return []
        except RuntimeError as e:
            print(f"ERROR: MCP error from '{server_name}': {e}")
            return []
        except Exception as e:
            print(f"ERROR: Unexpected error from MCP server '{server_name}': {e}")
            return []

    def call_tool(
        self, server_name: str, tool_name: str, arguments: dict
    ) -> dict:
        """
        Call a tool on the named MCP server and return the full MCP result dict.

        Opens a fresh MCP session (initialize → notifications/initialized →
        tools/call). Returns the raw result dict from the server:
            {"content": [{"type": "text", "text": "..."}], "data": {...}}

        Callers extract content[0]["text"] for LLM injection, or inspect
        data["status"] for structured status checks (e.g. reminder poller).

        Raises RuntimeError if the server returns an error.
        Raises requests.exceptions.ConnectionError if the server is unreachable.
        """
        spec = self._servers.get(server_name)
        if spec is None:
            raise RuntimeError(
                f"MCP server '{server_name}' not found in loaded servers"
            )

        url = spec["url"]
        session_id = self._initialize(url)
        self._notify_initialized(url, session_id)
        return self._tools_call(url, session_id, tool_name, arguments)

    def all_tools_for_llm(self) -> list[dict]:
        """
        Return all cached tools as OpenAI Responses API function definitions.

        Each entry has the standard OpenAI format plus an internal
        '_mcp_server' key (stripped before sending to the API) so the
        orchestrator can route tool calls back to the right server.
        """
        result: list[dict] = []
        for server_name, tools in self._tools_cache.items():
            for t in tools:
                result.append({
                    "type": "function",
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get(
                        "inputSchema",
                        {"type": "object", "properties": {}},
                    ),
                    "_mcp_server": server_name,  # internal routing key
                })
        return result

    def find_tool_server(self, tool_name: str) -> str | None:
        """Return the server name that provides the named tool, or None."""
        for server_name, tools in self._tools_cache.items():
            for t in tools:
                if t["name"] == tool_name:
                    return server_name
        return None

    def tools_summary(self) -> str:
        """One-line summary of cached tools across all servers."""
        total = sum(len(t) for t in self._tools_cache.values())
        if total == 0:
            return "0 tool(s) available"
        names = [t["name"] for tools in self._tools_cache.values() for t in tools]
        return f"{total} tool(s): {', '.join(names)}"

    def summary(self) -> str:
        """One-line summary of loaded servers."""
        n = len(self._servers)
        names = ", ".join(self._servers.keys()) if self._servers else "(none)"
        return f"{n} server(s): {names}"

    # ---------------- MCP protocol internals ----------------

    def _post(self, url: str, payload: dict, session_id: str | None = None
              ) -> requests.Response:
        """Send a single MCP JSON-RPC POST request."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if session_id:
            headers["Mcp-Session-Id"] = session_id
        return requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT)

    def _initialize(self, url: str) -> str:
        """
        Send MCP initialize. Returns session ID from Mcp-Session-Id header.
        Raises RuntimeError if server returns error or no session ID.
        """
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _CLIENT_INFO,
            },
        }
        resp = self._post(url, payload)
        resp.raise_for_status()

        session_id = resp.headers.get("Mcp-Session-Id", "").strip()
        if not session_id:
            raise RuntimeError(
                "Server did not return Mcp-Session-Id header in initialize response"
            )

        data = resp.json()
        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"initialize failed: {msg}")

        return session_id

    def _notify_initialized(self, url: str, session_id: str):
        """
        Send notifications/initialized to complete the MCP handshake.
        Notification (no 'id') — server responds with 202.
        """
        payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        resp = self._post(url, payload, session_id=session_id)
        if resp.status_code not in (200, 202, 204):
            print(
                f"[WARN] notifications/initialized returned HTTP {resp.status_code}",
                file=sys.stderr,
            )

    def _fetch_tools(self, url: str) -> list[dict]:
        """Open a session, fetch tools/list, return the tools array."""
        session_id = self._initialize(url)
        self._notify_initialized(url, session_id)
        return self._tools_list(url, session_id)

    def _tools_list(self, url: str, session_id: str) -> list[dict]:
        """Send tools/list and return the tools array."""
        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
        resp = self._post(url, payload, session_id=session_id)
        resp.raise_for_status()

        data = resp.json()
        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"tools/list failed: {msg}")

        result = data.get("result", {})
        return result.get("tools", [])

    def _tools_call(
        self, url: str, session_id: str, tool_name: str, arguments: dict
    ) -> dict:
        """Send tools/call and return the result dict from JSON-RPC response."""
        payload = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }
        resp = self._post(url, payload, session_id=session_id)
        resp.raise_for_status()

        data = resp.json()
        if "error" in data:
            err = data["error"]
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"tools/call '{tool_name}' failed: {msg}")

        return data.get("result", {})
