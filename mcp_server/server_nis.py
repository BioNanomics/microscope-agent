# server_nis.py
# ------------------------------------------------------------
# MCP server for driving the microscope through NIS-Elements (see
# nis_tools.py). Separate from server_loop.py on purpose: that server's
# tool set is fixed by team direction and has no NIS-Elements in it, and
# this one needs the NIS bridge job running inside NIS.
#
# Run, either way:
#   confocal-mcp-nis                  (installed console script)
#   python -m mcp_server.server_nis   (from a source checkout)
#
#   { "mcpServers": { "confocal-nis": { "command": "confocal-mcp-nis" } } }
# ------------------------------------------------------------

from mcp.server.mcpserver import MCPServer

from acquisition import estop
from mcp_server import loop_tools, nis_tools

mcp = MCPServer("ConfocalOrchestrator-NIS")

mcp.add_tool(nis_tools.nis_status)
mcp.add_tool(nis_tools.nis_move_relative)
mcp.add_tool(nis_tools.nis_change_objective)
mcp.add_tool(nis_tools.nis_capture)
mcp.add_tool(nis_tools.nis_run_experiment)
mcp.add_tool(nis_tools.nis_finish_experiment)
mcp.add_tool(loop_tools.estop)          # the same e-stop, reachable from this server too


def main() -> None:
    """Console entry point (``confocal-mcp-nis``): serve the NIS tools over stdio."""
    estop.launch_panel()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
