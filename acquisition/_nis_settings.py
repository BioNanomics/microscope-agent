# _nis_settings.py
# ------------------------------------------------------------
# This repo's settings for the nis-elements-mcp package
# (packages/nis-elements-mcp, see its settings.py). The forwarders that
# lead into the package (acquisition/estop*.py, acquisition/nis_*.py,
# acquisition/backends/nis_bridge.py, mcp_server/nis/) import this first.
# A variable that is already set wins, so an MCP client config can still
# override any of them. Child processes (STOP panel, monitor) inherit them.
#
#   * Logs and captures stay where they always were: logs/nis and
#     data/NIS/captures under the data folder (acquisition/paths.py).
#   * The e-stop is the package default, ~/.nis-mcp/ESTOP, because the
#     bridge job inside NIS uses that default and cannot see this repo's
#     settings. A stop still engaged in the old file (~/.confocal-mcp/ESTOP,
#     before 2026-10-30) is moved there, so a switch never loses a stop.
#   * The old CONFOCAL_* names of the package's switches still work.
# ------------------------------------------------------------

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

from acquisition import paths
from nis_elements_mcp import settings


def use_repo_folders() -> None:
    """Logs in logs/nis and captures in data/NIS/captures, unless already set."""
    os.environ.setdefault("NIS_LOG_DIR", str(paths.logs_dir() / "nis"))
    os.environ.setdefault("NIS_CAPTURE_DIR", os.environ.get("CONFOCAL_NIS_CAPTURE_DIR")
                          or str(paths.data_root() / "data" / "NIS" / "captures"))


use_repo_folders()


def use_old_names() -> None:
    """Honour the switches' names from before the package (CONFOCAL_*)."""
    for old, new in (("CONFOCAL_NO_ESTOP_PANEL", "NIS_NO_ESTOP_PANEL"),
                     ("CONFOCAL_NO_NIS_MONITOR", "NIS_NO_MONITOR")):
        if os.environ.get(old):
            os.environ.setdefault(new, os.environ[old])
    if os.environ.get("CONFOCAL_NIS_BRIDGE_URL"):
        url = urlsplit(os.environ["CONFOCAL_NIS_BRIDGE_URL"])
        os.environ.setdefault("NIS_BRIDGE_HOST", url.hostname or "127.0.0.1")
        os.environ.setdefault("NIS_BRIDGE_PORT", str(url.port or 8766))


use_old_names()

OLD_ESTOP_FILE = Path.home() / ".confocal-mcp" / "ESTOP"


def carry_over_old_estop(old: Path = OLD_ESTOP_FILE) -> None:
    """Move a stop engaged in the old e-stop file to the current one.

    If the old file cannot be removed it stays, and the stop is engaged
    again on the next start: failing towards stopped, never towards moving.
    """
    try:
        if not old.exists():
            return
        new = settings.estop_file()
        if not new.exists():
            new.parent.mkdir(parents=True, exist_ok=True)
            new.write_bytes(old.read_bytes())
        old.unlink()
        print(f"[estop] moved the engaged e-stop from {old} to {new}", file=sys.stderr)
    except OSError as exc:
        print(f"[estop] could not move the old e-stop file {old} ({exc}); "
              f"delete it by hand once {settings.estop_file()} is engaged", file=sys.stderr)


carry_over_old_estop()
