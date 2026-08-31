# server_loop.py
# ------------------------------------------------------------
# Minimal MCP server entry point: registers ONLY loop_tools.py's
# functions (get_image, get_pos, move, get_move_history) - kept
# deliberately minimal, 4 tools total, by explicit team direction.
#
# Run, either way:
#   confocal-mcp                      (installed console script - see pyproject.toml)
#   python -m mcp_server.server_loop  (from a source checkout)
#
# Point an MCP client at whichever form you use, e.g.:
#   { "mcpServers": { "confocal": { "command": "confocal-mcp" } } }
# ------------------------------------------------------------

from mcp.server.mcpserver import MCPServer

from mcp_server import loop_tools as tools

mcp = MCPServer("ConfocalOrchestrator-Loop")

mcp.add_tool(tools.get_image)
mcp.add_tool(tools.get_pos)
mcp.add_tool(tools.move)
mcp.add_tool(tools.get_move_history)


def main() -> None:
    """Console entry point (``confocal-mcp``): serve the 4 tools over stdio."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
