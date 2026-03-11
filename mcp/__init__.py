# MCP server package
#
# Structure:
#   base.py          — MCPBaseHandler + session store + JSON-RPC helpers
#   mcp_weather.py   — Weather forecast server (get_forecast via Open-Meteo)
#
# To add a new server:
#   1. Create mcp/mcp_<name>.py
#   2. Subclass MCPBaseHandler, override _get_tools() and _dispatch_tool()
#   3. Add an entry in tools/<name>.yaml pointing to the server URL
