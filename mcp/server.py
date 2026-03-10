#!/usr/bin/env python3
"""
mcp/server.py — Minimal local MCP server over HTTP.

Implements MCP Streamable HTTP transport (spec 2024-11-05).
Exposes a single tool: get_forecast (weather via Open-Meteo).

Usage:
    python mcp/server.py                  # 127.0.0.1:8000
    python mcp/server.py --port 9000
    python mcp/server.py --host 0.0.0.0 --port 8080

MCP endpoint: http://127.0.0.1:8000/mcp
"""

import argparse
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import quote as url_quote

import requests

# ---------------------------------------------------------------------------
# MCP protocol constants
# ---------------------------------------------------------------------------

_PROTOCOL_VERSION = "2024-11-05"
_SERVER_INFO = {"name": "local-weather-mcp", "version": "0.1"}

# Tool definition returned by tools/list
_TOOL_GET_FORECAST = {
    "name": "get_forecast",
    "title": "Weather forecast",
    "description": (
        "Get weather forecast by city/place and number of days. "
        "Returns daily temperature_min and temperature_max."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "place": {
                "type": "string",
                "description": "City or place name",
            },
            "days": {
                "type": "integer",
                "description": "Number of forecast days",
                "default": 3,
            },
        },
        "required": ["place"],
    },
}

# ---------------------------------------------------------------------------
# Session store  (module-level, process lifetime)
# ---------------------------------------------------------------------------

# sid -> True means session is fully initialized (notifications/initialized received)
_sessions: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Weather logic
# ---------------------------------------------------------------------------

def _get_forecast(place: str, days: int) -> dict:
    """
    Fetch weather forecast for *place* for *days* days.

    Returns list of dicts: [{date, temperature_min, temperature_max}, ...]
    Raises ValueError if the place is not found.
    Raises RuntimeError on Open-Meteo API errors.
    """
    # 1. Geocode
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={url_quote(place)}&count=1"
    )
    geo_resp = requests.get(geo_url, timeout=10)
    geo_resp.raise_for_status()
    geo_data = geo_resp.json()

    results = geo_data.get("results") or []
    if not results:
        raise ValueError(f"Place not found: {place!r}")

    lat = results[0]["latitude"]
    lon = results[0]["longitude"]
    resolved_name = results[0].get("name", place)
    country = results[0].get("country", "")

    # 2. Forecast
    fc_url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=temperature_2m_max,temperature_2m_min"
        f"&forecast_days={days}"
        "&timezone=auto"
    )
    fc_resp = requests.get(fc_url, timeout=10)
    fc_resp.raise_for_status()
    fc_data = fc_resp.json()

    daily = fc_data.get("daily", {})
    times = daily.get("time", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])

    if not times:
        raise RuntimeError("Open-Meteo returned empty forecast data")

    forecast = []
    for date, tmax, tmin in zip(times, t_max, t_min):
        forecast.append({
            "date": date,
            "temperature_min": tmin,
            "temperature_max": tmax,
        })

    return {
        "place": resolved_name,
        "country": country,
        "latitude": lat,
        "longitude": lon,
        "forecast": forecast,
    }


# ---------------------------------------------------------------------------
# JSON-RPC helpers
# ---------------------------------------------------------------------------

def _ok(req_id, result: dict | list) -> dict:
    """Build a JSON-RPC success response."""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id, code: int, message: str) -> dict:
    """Build a JSON-RPC error response."""
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


# JSON-RPC 2.0 error codes
_ERR_INVALID_REQUEST = -32600
_ERR_METHOD_NOT_FOUND = -32601
_ERR_INVALID_PARAMS = -32602
_ERR_SERVER_ERROR = -32000


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class MCPHandler(BaseHTTPRequestHandler):
    """Handles POST /mcp — full MCP Streamable HTTP transport."""

    # Silence default access log; we print our own
    def log_message(self, format, *args):  # noqa: A002
        print(f"[MCP] {self.address_string()} {format % args}", file=sys.stdout)

    def do_POST(self):
        if self.path != "/mcp":
            self._send_json(404, {"error": "Not found"})
            return

        # Enforce Accept header per MCP spec
        accept = self.headers.get("Accept", "")
        if "application/json" not in accept and "text/event-stream" not in accept:
            self._send_rpc_error(
                None,
                _ERR_INVALID_REQUEST,
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
            self._send_rpc_error(None, _ERR_INVALID_REQUEST,
                                 f"Invalid JSON: {e}")
            return

        method = body.get("method", "")
        req_id = body.get("id")           # None for notifications
        session_id = self.headers.get("Mcp-Session-Id", "").strip()
        params = body.get("params") or {}

        # Route
        if method == "initialize":
            self._handle_initialize(req_id, params)
        elif method == "notifications/initialized":
            self._handle_notifications_initialized(session_id)
        elif method == "tools/list":
            self._handle_tools_list(req_id, session_id)
        elif method == "tools/call":
            self._handle_tools_call(req_id, session_id, params)
        else:
            self._send_rpc_error(
                req_id, _ERR_METHOD_NOT_FOUND,
                f"Method not found: {method!r}",
            )

    # ------------------------------------------------------------------
    # MCP method handlers
    # ------------------------------------------------------------------

    def _handle_initialize(self, req_id, params: dict):
        """Create a new session, return capabilities."""
        sid = uuid.uuid4().hex
        _sessions[sid] = False   # not yet initialized

        result = {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": _SERVER_INFO,
            "instructions": (
                "Local weather MCP server. "
                "Use get_forecast to get weather by city name."
            ),
        }
        extra_headers = {"Mcp-Session-Id": sid}
        self._send_rpc_ok(req_id, result, extra_headers=extra_headers)

    def _handle_notifications_initialized(self, session_id: str):
        """Mark session as fully initialized. Respond 202 with no body."""
        if not session_id or session_id not in _sessions:
            # Tolerate missing session on notifications (some clients don't send it)
            if session_id:
                _sessions[session_id] = True
            self.send_response(202)
            self.end_headers()
            return
        _sessions[session_id] = True
        self.send_response(202)
        self.end_headers()

    def _handle_tools_list(self, req_id, session_id: str):
        """Return the list of available tools."""
        err = self._require_session(req_id, session_id)
        if err:
            return
        self._send_rpc_ok(req_id, {"tools": [_TOOL_GET_FORECAST]})

    def _handle_tools_call(self, req_id, session_id: str, params: dict):
        """Execute a tool call and return the result."""
        err = self._require_session(req_id, session_id)
        if err:
            return

        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}

        if tool_name != "get_forecast":
            self._send_rpc_error(
                req_id, _ERR_METHOD_NOT_FOUND,
                f"Unknown tool: {tool_name!r}",
            )
            return

        place = arguments.get("place", "").strip()
        if not place:
            self._send_rpc_error(
                req_id, _ERR_INVALID_PARAMS,
                "Missing required argument: 'place'",
            )
            return

        days = arguments.get("days", 3)
        try:
            days = int(days)
            if not (1 <= days <= 16):
                raise ValueError("days must be between 1 and 16")
        except (TypeError, ValueError) as e:
            self._send_rpc_error(req_id, _ERR_INVALID_PARAMS, str(e))
            return

        try:
            data = _get_forecast(place, days)
        except ValueError as e:
            # Place not found
            self._send_rpc_error(req_id, _ERR_INVALID_PARAMS, str(e))
            return
        except Exception as e:
            self._send_rpc_error(req_id, _ERR_SERVER_ERROR,
                                 f"Forecast error: {e}")
            return

        # Format result as MCP content array (text block)
        lines = [
            f"Weather forecast for {data['place']}"
            + (f", {data['country']}" if data['country'] else ""),
            f"Coordinates: {data['latitude']}, {data['longitude']}",
            "",
        ]
        for day in data["forecast"]:
            lines.append(
                f"{day['date']}: "
                f"min {day['temperature_min']}°C, "
                f"max {day['temperature_max']}°C"
            )

        result = {
            "content": [
                {"type": "text", "text": "\n".join(lines)}
            ],
            "data": data,   # also return structured data
        }
        self._send_rpc_ok(req_id, result)

    # ------------------------------------------------------------------
    # Session guard
    # ------------------------------------------------------------------

    def _require_session(self, req_id, session_id: str) -> bool:
        """
        Verify session exists and is initialized.
        Sends error response and returns True if session is invalid.
        Returns False if session is valid (caller may proceed).
        """
        if not session_id:
            self._send_rpc_error(
                req_id, _ERR_INVALID_REQUEST,
                "Mcp-Session-Id header required for this request",
            )
            return True
        if session_id not in _sessions:
            self._send_rpc_error(
                req_id, _ERR_INVALID_REQUEST,
                "Invalid or expired session. Please re-initialize.",
            )
            return True
        if not _sessions[session_id]:
            self._send_rpc_error(
                req_id, _ERR_INVALID_REQUEST,
                "Session not initialized. Send notifications/initialized first.",
            )
            return True
        return False

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _send_rpc_ok(self, req_id, result, extra_headers: dict | None = None):
        """Send a 200 JSON-RPC success response."""
        self._send_json(200, _ok(req_id, result),
                        extra_headers=extra_headers)

    def _send_rpc_error(self, req_id, code: int, message: str,
                        status: int = 200):
        """Send a JSON-RPC error response."""
        self._send_json(status, _err(req_id, code, message))

    def _send_json(self, status: int, data: dict,
                   extra_headers: dict | None = None):
        """Serialize *data* as JSON and write the full HTTP response."""
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
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Parse CLI args and start the MCP server."""
    parser = argparse.ArgumentParser(
        description="Local MCP server — weather forecast tool"
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port (default: 8000)")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), MCPHandler)
    url = f"http://{args.host}:{args.port}/mcp"
    print(f"MCP server running at {url}")
    print(f"Tools: get_forecast")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
