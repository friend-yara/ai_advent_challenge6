#!/usr/bin/env python3
"""
mcp/mcp_weather.py — Local MCP server: weather forecast tool.

Implements MCP Streamable HTTP transport (spec 2024-11-05) via MCPBaseHandler.
Exposes one tool: get_forecast (weather data via Open-Meteo, no API key needed).

Usage:
    python mcp/mcp_weather.py                  # 127.0.0.1:8000
    python mcp/mcp_weather.py --port 9000
    python mcp/mcp_weather.py --host 0.0.0.0 --port 8080

MCP endpoint: http://127.0.0.1:8000/mcp
"""

from urllib.parse import quote as url_quote

import requests

try:
    from mcp.base import (
        ERR_INVALID_PARAMS, ERR_METHOD_NOT_FOUND, ERR_SERVER_ERROR,
        MCPBaseHandler, run_server,
    )
except ImportError:
    # When run directly as `python mcp/mcp_weather.py` the package root
    # may not be on sys.path; fall back to a sibling-module import.
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from mcp.base import (
        ERR_INVALID_PARAMS, ERR_METHOD_NOT_FOUND, ERR_SERVER_ERROR,
        MCPBaseHandler, run_server,
    )

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

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
# Weather logic
# ---------------------------------------------------------------------------

def _get_forecast(place: str, days: int) -> dict:
    """
    Fetch weather forecast for *place* for *days* days via Open-Meteo.

    Returns dict with place, country, latitude, longitude, forecast list.
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

    lat           = results[0]["latitude"]
    lon           = results[0]["longitude"]
    resolved_name = results[0].get("name", place)
    country       = results[0].get("country", "")

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

    daily  = fc_data.get("daily", {})
    times  = daily.get("time", [])
    t_max  = daily.get("temperature_2m_max", [])
    t_min  = daily.get("temperature_2m_min", [])

    if not times:
        raise RuntimeError("Open-Meteo returned empty forecast data")

    forecast = [
        {"date": date, "temperature_min": tmin, "temperature_max": tmax}
        for date, tmax, tmin in zip(times, t_max, t_min)
    ]

    return {
        "place": resolved_name,
        "country": country,
        "latitude": lat,
        "longitude": lon,
        "forecast": forecast,
    }


# ---------------------------------------------------------------------------
# MCP handler
# ---------------------------------------------------------------------------

class WeatherMCPHandler(MCPBaseHandler):
    """MCP handler exposing the get_forecast weather tool."""

    SERVER_NAME    = "local-weather-mcp"
    SERVER_VERSION = "0.1"
    INSTRUCTIONS   = (
        "Local weather MCP server. "
        "Use get_forecast to get weather by city name."
    )
    TOOLS = [_TOOL_GET_FORECAST]   # class-level, used by run_server for display

    def _get_tools(self) -> list[dict]:
        return self.TOOLS

    def _dispatch_tool(self, req_id, tool_name: str,
                       arguments: dict, session_id: str):
        if tool_name != "get_forecast":
            self._send_rpc_error(req_id, ERR_METHOD_NOT_FOUND,
                                 f"Unknown tool: {tool_name!r}")
            return

        place = arguments.get("place", "").strip()
        if not place:
            self._send_rpc_error(req_id, ERR_INVALID_PARAMS,
                                 "Missing required argument: 'place'")
            return

        days = arguments.get("days", 3)
        try:
            days = int(days)
            if not (1 <= days <= 16):
                raise ValueError("days must be between 1 and 16")
        except (TypeError, ValueError) as e:
            self._send_rpc_error(req_id, ERR_INVALID_PARAMS, str(e))
            return

        try:
            data = _get_forecast(place, days)
        except ValueError as e:
            self._send_rpc_error(req_id, ERR_INVALID_PARAMS, str(e))
            return
        except Exception as e:
            self._send_rpc_error(req_id, ERR_SERVER_ERROR,
                                 f"Forecast error: {e}")
            return

        # Format as MCP content array
        lines = [
            f"Weather forecast for {data['place']}"
            + (f", {data['country']}" if data["country"] else ""),
            f"Coordinates: {data['latitude']}, {data['longitude']}",
            "",
        ]
        for day in data["forecast"]:
            lines.append(
                f"{day['date']}: "
                f"min {day['temperature_min']}°C, "
                f"max {day['temperature_max']}°C"
            )

        self._send_rpc_ok(req_id, {
            "content": [{"type": "text", "text": "\n".join(lines)}],
            "data": data,
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_server(WeatherMCPHandler)
