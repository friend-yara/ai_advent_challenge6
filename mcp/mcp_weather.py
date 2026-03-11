"""
mcp/mcp_weather.py — Weather domain module for the MCP router.

Public interface:
    WEATHER_TOOLS                          list[dict]  MCP tool definitions
    dispatch_weather_tool(name, args)      -> dict     MCP result or raises

Raises:
    KeyError   — unknown tool name
    ValueError — invalid / missing arguments
    RuntimeError — upstream API failure
"""

from urllib.parse import quote as url_quote

import requests

# ---------------------------------------------------------------------------
# Tool definitions (MCP-compatible schema)
# ---------------------------------------------------------------------------

WEATHER_TOOLS: list[dict] = [
    {
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
]

# ---------------------------------------------------------------------------
# Public dispatch entry point
# ---------------------------------------------------------------------------

def dispatch_weather_tool(tool_name: str, arguments: dict) -> dict:
    """
    Dispatch a tools/call request to the appropriate weather tool.

    Returns an MCP-compatible result dict:
        {"content": [{"type": "text", "text": "..."}], "data": {...}}

    Raises:
        KeyError    if tool_name is not a known weather tool
        ValueError  if required arguments are missing or invalid
        RuntimeError on upstream API errors
    """
    if tool_name != "get_forecast":
        raise KeyError(f"Unknown weather tool: {tool_name!r}")

    place = arguments.get("place", "")
    if isinstance(place, str):
        place = place.strip()
    if not place:
        raise ValueError("Missing required argument: 'place'")

    days = arguments.get("days", 3)
    try:
        days = int(days)
        if not (1 <= days <= 16):
            raise ValueError("days must be between 1 and 16")
    except (TypeError, ValueError) as e:
        raise ValueError(str(e)) from e

    data = _get_forecast(place, days)

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

    return {
        "content": [{"type": "text", "text": "\n".join(lines)}],
        "data": data,
    }


# ---------------------------------------------------------------------------
# Internal weather logic (Open-Meteo, no API key required)
# ---------------------------------------------------------------------------

def _get_forecast(place: str, days: int) -> dict:
    """
    Fetch weather forecast via Open-Meteo geocoding + forecast APIs.

    Returns dict: {place, country, latitude, longitude, forecast: [...]}.
    Raises ValueError if place is not found.
    Raises RuntimeError on API errors.
    """
    # 1. Geocode
    geo_url = (
        "https://geocoding-api.open-meteo.com/v1/search"
        f"?name={url_quote(place)}&count=1"
    )
    geo_resp = requests.get(geo_url, timeout=10)
    geo_resp.raise_for_status()
    results = (geo_resp.json().get("results") or [])
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
    daily = fc_resp.json().get("daily", {})

    times = daily.get("time", [])
    t_max = daily.get("temperature_2m_max", [])
    t_min = daily.get("temperature_2m_min", [])

    if not times:
        raise RuntimeError("Open-Meteo returned empty forecast data")

    return {
        "place": resolved_name,
        "country": country,
        "latitude": lat,
        "longitude": lon,
        "forecast": [
            {"date": d, "temperature_min": mn, "temperature_max": mx}
            for d, mx, mn in zip(times, t_max, t_min)
        ],
    }
