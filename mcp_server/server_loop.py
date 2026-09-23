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

import os
import subprocess
import sys

from mcp.server.mcpserver import MCPServer

from mcp_server import loop_tools as tools

mcp = MCPServer("ConfocalOrchestrator-Loop")

mcp.add_tool(tools.get_image)
mcp.add_tool(tools.get_pos)
mcp.add_tool(tools.move)
mcp.add_tool(tools.get_move_history)
mcp.add_tool(tools.estop)


def _launch_estop_panel() -> None:
    """Put the STOP button on screen for as long as this server runs.

    Detached, not a child we wait on: the panel must survive this process
    hanging, and a server that cannot draw a window (headless, no display)
    must still serve tools. Any failure here is reported to stderr - never
    stdout, which is the JSON-RPC channel - and never prevents startup.

    Launched by default because the one time it was needed, the operator
    was hunting for a terminal while the stage was moving. A safety
    control you have to remember to start is one you will not have.
    """
    if os.environ.get("CONFOCAL_NO_ESTOP_PANEL"):
        return
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
        subprocess.Popen([sys.executable, "-m", "acquisition.estop_panel"],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=flags)
    except Exception as exc:
        print(f"[confocal-mcp] could not start the e-stop panel ({exc}). "
              f"Stop manually with: python -m acquisition.estop engage", file=sys.stderr)


def main() -> None:
    """Console entry point (``confocal-mcp``): serve the 5 tools over stdio."""
    _launch_estop_panel()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
