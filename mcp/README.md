# Local MCP Server — Weather Forecast

Minimal MCP server over HTTP (Streamable HTTP transport, spec 2024-11-05).  
Implements one tool: **get_forecast** — weather forecast via Open-Meteo (free, no API key).

---

## Start the server

```bash
# Default: 127.0.0.1:8000
python mcp/mcp_weather.py

# Custom host/port
python mcp/mcp_weather.py --host 0.0.0.0 --port 9000
```

## MCP endpoint

```
http://127.0.0.1:8000/mcp
```

---

## Tool: get_forecast

| Field | Value |
|---|---|
| name | `get_forecast` |
| title | Weather forecast |
| description | Get weather forecast by city/place and number of days |

**Arguments:**

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `place` | string | yes | — | City or place name |
| `days` | integer | no | 3 | Number of forecast days (1–16) |

**Returns:** daily `temperature_min` and `temperature_max` in °C.

---

## Connect to the agent

Add `tools/weather.yaml` (already included in this project):

```yaml
name: weather
url: http://127.0.0.1:8000/mcp
description: Локальный MCP-сервер — прогноз погоды (get_forecast)
```

Then in the agent CLI (start server first):

```
/tool list              → shows "weather" in the list
/tool list weather      → connects and prints get_forecast tool info
```

---

## curl examples

All requests require `Accept: application/json, text/event-stream`.

### 1. initialize

```bash
curl -s -D - -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {},
      "clientInfo": {"name": "test", "version": "0.1"}
    }
  }'
```

Response headers contain `Mcp-Session-Id: <sid>` — save this for subsequent requests.

### 2. notifications/initialized

```bash
curl -s -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: <sid>" \
  -d '{"jsonrpc": "2.0", "method": "notifications/initialized"}'
# → 202
```

### 3. tools/list

```bash
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: <sid>" \
  -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}'
```

### 4. tools/call — get_forecast for London

```bash
curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: <sid>" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "get_forecast",
      "arguments": {"place": "London", "days": 3}
    }
  }'
```

Example response:

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Weather forecast for London, United Kingdom\nCoordinates: 51.50853, -0.12574\n\n2026-03-10: min 9.0°C, max 13.5°C\n2026-03-11: min 7.5°C, max 12.2°C\n2026-03-12: min 7.0°C, max 11.8°C"
      }
    ],
    "data": {
      "place": "London",
      "country": "United Kingdom",
      "latitude": 51.50853,
      "longitude": -0.12574,
      "forecast": [
        {"date": "2026-03-10", "temperature_min": 9.0, "temperature_max": 13.5},
        {"date": "2026-03-11", "temperature_min": 7.5, "temperature_max": 12.2},
        {"date": "2026-03-12", "temperature_min": 7.0, "temperature_max": 11.8}
      ]
    }
  }
}
```

### One-liner: full session in a shell script

```bash
SID=$(curl -sI -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"sh","version":"0.1"}}}' \
  | awk 'tolower($0) ~ /mcp-session-id:/ {print $2}' | tr -d '\r')

curl -so /dev/null -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

curl -s -X POST http://127.0.0.1:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "Mcp-Session-Id: $SID" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_forecast","arguments":{"place":"London","days":3}}}'
```
