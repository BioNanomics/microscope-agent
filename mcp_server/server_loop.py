# server_loop.py
# ------------------------------------------------------------
# Minimal MCP server entry point: registers ONLY loop_tools.py's
# functions (get_image, get_pos, move, get_move_history) - kept
# deliberately minimal, 4 tools total, by explicit team direction.
#
# Run (from the repo root, with .venv activated):
#   python -m mcp_server.server_loop
# ------------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.mcpserver import MCPServer

from mcp_server import loop_tools as tools

mcp = MCPServer("ConfocalOrchestrator-Loop")

mcp.add_tool(tools.get_image)
mcp.add_tool(tools.get_pos)
mcp.add_tool(tools.move)
mcp.add_tool(tools.get_move_history)


if __name__ == "__main__":
    mcp.run(transport="stdio")
