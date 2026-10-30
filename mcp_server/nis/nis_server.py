# nis_server.py
# ------------------------------------------------------------
# MCP server for driving the microscope through NIS-Elements (see
# nis_tools.py). Separate from baumer_ti2_server.py on purpose: that server's
# tool set is fixed by team direction and has no NIS-Elements in it, and
# this one needs the NIS bridge job running inside NIS.
#
# Run, either way:
#   nis-mcp                           (installed console script; the old name
#                                      confocal-mcp-nis still works)
#   python -m mcp_server.nis.nis_server   (from a source checkout)
#
#   { "mcpServers": { "nis": { "command": "nis-mcp" } } }
# ------------------------------------------------------------

from mcp.server.mcpserver import MCPServer

from acquisition import estop, nis_monitor
from mcp_server.baumer_ti2 import baumer_ti2_tools
from mcp_server.nis import nis_tools

mcp = MCPServer("ConfocalOrchestrator-NIS")

mcp.add_tool(nis_tools.nis_status)
mcp.add_tool(nis_tools.nis_move_relative)
mcp.add_tool(nis_tools.nis_change_objective)
mcp.add_tool(nis_tools.nis_list_optical_configs)
mcp.add_tool(nis_tools.nis_capture)
mcp.add_tool(nis_tools.nis_run_experiment)
mcp.add_tool(nis_tools.nis_finish_experiment)
mcp.add_tool(baumer_ti2_tools.estop)          # the same e-stop, reachable from this server too


def main() -> None:
    """Console entry point (``nis-mcp``): serve the NIS tools over stdio."""
    estop.launch_panel()
    nis_monitor.launch()            # live view of what this server does
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
