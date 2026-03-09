#!/usr/bin/env python3
"""
mcp_client.py — MCP (Model Context Protocol) client over Streamable HTTP.

Implements the 2024-11-05 MCP Streamable HTTP transport using only the
`requests` library (already a project dependency — no new packages needed).

Session lifecycle per call (stateless — no session is reused):
  1. POST <url>  initialize           → captures Mcp-Session-Id from headers
  2. POST <url>  notifications/initialized  (session header, no id)  → 202
  3. POST <url>  tools/list           (session header)  → tools array

Server specs are loaded from *.yaml files in a configurable directory
(default: tools/). Each file must contain at minimum: name, url.

Usage:
    mcp = MCPClient("tools/")
    mcp.load()
    tools = mcp.list_tools("vkusvill")
    for t in tools:
        print(t["name"], t.get("description", ""))
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
    Each call to list_tools() opens a fresh session (initialize → tools/list).
    No persistent sessions, no background threads, no extra dependencies.
    """

    def __init__(self, servers_dir: str | Path = "tools"):
        """Initialize with path to server spec directory."""
        self.servers_dir = Path(servers_dir)
        self._servers: dict[str, dict] = {}   # name -> {name, url, description}

    def load(self):
        """
        Scan servers_dir for *.yaml files and load each as a server spec.
        Skips files with missing required fields (warns to stderr).
        """
        self._servers = {}
        if not self.servers_dir.is_dir():
            print(
                f"[WARN] MCP servers dir not found: {self.servers_dir}",
                file=sys.stderr,
            )
            return
        for path in sorted(self.servers_dir.glob("*.yaml")):
            spec = self._load_file(path)
            if spec is not None:
                self._servers[spec["name"]] = spec

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
        Connect to the named MCP server and return its tool list.

        Each tool dict contains at minimum: name, description.
        Optionally: title, inputSchema, and any other server-provided fields.

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

        url = spec["url"]
        try:
            session_id = self._initialize(url)
            self._notify_initialized(url, session_id)
            return self._tools_list(url, session_id)
        except requests.exceptions.ConnectionError as e:
            print(f"ERROR: Cannot connect to MCP server '{server_name}': {e}")
            return []
        except requests.exceptions.Timeout:
            print(
                f"ERROR: Timeout connecting to MCP server '{server_name}' "
                f"({url})"
            )
            return []
        except RuntimeError as e:
            print(f"ERROR: MCP error from '{server_name}': {e}")
            return []
        except Exception as e:
            print(f"ERROR: Unexpected error from MCP server '{server_name}': {e}")
            return []

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
        Send MCP initialize request.
        Returns the session ID from the Mcp-Session-Id response header.
        Raises RuntimeError if the server returns an error or no session ID.
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
        This is a notification (no 'id' field) — server responds with 202.
        """
        payload = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        }
        resp = self._post(url, payload, session_id=session_id)
        # 202 Accepted is the expected response; anything else is tolerated
        # (some servers may return 200 with no body)
        if resp.status_code not in (200, 202, 204):
            print(
                f"[WARN] notifications/initialized returned "
                f"HTTP {resp.status_code}",
                file=sys.stderr,
            )

    def _tools_list(self, url: str, session_id: str) -> list[dict]:
        """
        Send tools/list request and return the tools array.
        Raises RuntimeError on JSON-RPC error responses.
        """
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
