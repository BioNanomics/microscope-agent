# baumer_ti2_server.py
# ------------------------------------------------------------
# Minimal MCP server entry point: registers ONLY baumer_ti2_tools.py's
# functions (get_image, get_pos, move, get_move_history, estop) - kept
# deliberately minimal, 5 tools total, by explicit team direction.
#
# Run, either way:
#   baumer-ti2-mcp                    (installed console script - see pyproject.toml;
#                                      the old name confocal-mcp still works)
#   python -m mcp_server.baumer_ti2.baumer_ti2_server  (from a source checkout)
#
# Point an MCP client at whichever form you use, e.g.:
#   { "mcpServers": { "baumer-ti2": { "command": "baumer-ti2-mcp" } } }
# ------------------------------------------------------------

from mcp.server.mcpserver import MCPServer

from acquisition import estop

from mcp_server.baumer_ti2 import baumer_ti2_tools as tools

mcp = MCPServer("ConfocalOrchestrator-Loop")

mcp.add_tool(tools.get_image)
mcp.add_tool(tools.get_pos)
mcp.add_tool(tools.move)
mcp.add_tool(tools.get_move_history)
mcp.add_tool(tools.estop)


def main() -> None:
    """Console entry point (``baumer-ti2-mcp``): serve the 5 tools over stdio."""
    # The STOP panel, on screen for as long as this server runs.
    estop.launch_panel()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
