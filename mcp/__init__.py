# MCP server package
#
# Structure:
#   base.py          — MCPBaseHandler + session store + JSON-RPC helpers
#   server.py        — MCP router: single /mcp endpoint, delegates to domain modules
#   mcp_weather.py   — Weather domain module: WEATHER_TOOLS + dispatch_weather_tool
#
# To add a new domain module:
#   1. Create mcp/mcp_<name>.py with <NAME>_TOOLS and dispatch_<name>_tool()
#   2. Import and register in mcp/server.py (3 lines)
#   3. Add an entry in tools/<name>.yaml pointing to http://127.0.0.1:8000/mcp
