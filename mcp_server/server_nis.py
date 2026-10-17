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

from acquisition import estop, nis_monitor
from mcp_server import loop_tools, nis_tools

mcp = MCPServer("ConfocalOrchestrator-NIS")

mcp.add_tool(nis_tools.nis_status)
mcp.add_tool(nis_tools.nis_move_relative)
mcp.add_tool(nis_tools.nis_change_objective)
mcp.add_tool(nis_tools.nis_list_optical_configs)
# nis_capture / nis_run_experiment / nis_finish_experiment stay out until the bridge can
# run them on NIS's main thread (they crashed NIS from the worker thread, 2026-10-16).
mcp.add_tool(loop_tools.estop)          # the same e-stop, reachable from this server too


def main() -> None:
    """Console entry point (``confocal-mcp-nis``): serve the NIS tools over stdio."""
    estop.launch_panel()
    nis_monitor.launch()            # live view of what this server does
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
