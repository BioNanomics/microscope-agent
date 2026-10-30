# _nis_settings.py
# ------------------------------------------------------------
# This repo's settings for the nis-elements-mcp package
# (packages/nis-elements-mcp). The package keeps its event log and
# captures under NIS_LOG_DIR / NIS_CAPTURE_DIR (default ~/nis-mcp/...);
# this repo keeps them where they always were: logs/nis and
# data/NIS/captures under the data folder (acquisition/paths.py).
#
# The forwarders that lead into the package (acquisition/nis_*.py,
# acquisition/backends/nis_bridge.py, mcp_server/nis/) import this first.
# A variable that is already set wins, so an MCP client config can still
# point them elsewhere. Child processes (the monitor window) inherit them.
# ------------------------------------------------------------

import os

from acquisition import paths

os.environ.setdefault("NIS_LOG_DIR", str(paths.logs_dir() / "nis"))
os.environ.setdefault("NIS_CAPTURE_DIR", os.environ.get("CONFOCAL_NIS_CAPTURE_DIR")
                      or str(paths.data_root() / "data" / "NIS" / "captures"))
