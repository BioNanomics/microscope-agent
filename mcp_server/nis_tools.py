# nis_tools.py
# ------------------------------------------------------------
# MCP tools that drive the microscope THROUGH NIS-Elements, via the NIS
# bridge (acquisition/nis_bridge/bridge_job.py, running as a JOBS task;
# client in acquisition/backends/nis_bridge.py). Served by their own
# server, mcp_server/server_nis.py - kept apart from loop_tools.py, which
# by team direction has exactly its agreed tools and no NIS-Elements.
#
# WHAT THIS ADDS over loop_tools: things only NIS can do - the AX laser
# acquisition set up in NIS (capture, saved ND experiments) and NIS's own
# objective change.
#
# SAFETY, same shape as loop_tools:
#   * every tool that moves or acquires requires confirm=True
#     (PermissionError otherwise), exactly like backend="sdk" there
#   * the e-stop is checked in the client before a request is sent, and
#     again inside NIS by the bridge
#   * XY moves are relative and capped at 1 mm; there is no Z tool
#   * objective changes are limited to 4x/10x by the bridge
#   * captures are written only under the data folder, by file name - a
#     chat prompt cannot choose an arbitrary path
#   * nis_finish_experiment and estop never need confirm: they only stop
# ------------------------------------------------------------

from __future__ import annotations

import os
import re
from pathlib import Path

from acquisition.backends.nis_bridge import NISBridge
from acquisition.paths import data_root


def _require_confirm(confirm: bool, what: str) -> None:
    if not confirm:
        raise PermissionError(
            f"{what} drives real microscope hardware through NIS-Elements and "
            "requires confirm=True. Refusing to proceed without explicit confirmation.")


def _bridge() -> NISBridge:
    return NISBridge()


def capture_dir() -> Path:
    """Where nis_capture writes: CONFOCAL_NIS_CAPTURE_DIR, else <data>/data/nis_captures."""
    d = Path(os.environ.get("CONFOCAL_NIS_CAPTURE_DIR") or data_root() / "data" / "nis_captures")
    d.mkdir(parents=True, exist_ok=True)
    return d


def nis_status() -> dict:
    """Read-only snapshot from NIS-Elements: XY and Z position (um), current
    objective, the objective at each nosepiece position, whether an ND
    experiment is running, and whether the e-stop is engaged. Safe to call
    at any time; needs the NIS bridge job to be running."""
    return _bridge().status()


def nis_move_relative(dx_um: float, dy_um: float, confirm: bool = False) -> dict:
    """Move the XY stage by (dx_um, dy_um) microns RELATIVE to where it is,
    through NIS-Elements. At most 1000 um per axis per call. Returns the
    position before and after - treat 'after_um' as ground truth.
    Requires confirm=True. Z cannot be moved from here."""
    _require_confirm(confirm, "nis_move_relative")
    return _bridge().move_relative(dx_um, dy_um)


def nis_change_objective(name: str, confirm: bool = False) -> dict:
    """Switch objective in NIS-Elements. `name` must be the exact name
    nis_status lists under nosepiece_objectives, and only 4x or 10x
    objectives are accepted (long working distance - they cannot reach
    the sample). Returns the objective and nosepiece position before and
    after. Requires confirm=True."""
    _require_confirm(confirm, "nis_change_objective")
    return _bridge().change_objective(name)


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,80}$")


def nis_capture(filename: str, confirm: bool = False) -> dict:
    """Capture one image with the current NIS-Elements acquisition settings
    (lasers, channels, scan) and save it as an ND2 file in the NIS capture
    folder. `filename` is a plain name such as "aml18_check" - no folders;
    ".nd2" is added if missing and an existing file is never overwritten.
    Returns the saved path. Requires confirm=True."""
    _require_confirm(confirm, "nis_capture")
    stem = filename[:-4] if filename.lower().endswith(".nd2") else filename
    if not _SAFE_NAME.match(stem) or ".." in stem:
        raise ValueError("filename must be a plain name (letters, digits, space, _ . -), no folders")
    return _bridge().capture(str(capture_dir() / f"{stem}.nd2"))


def nis_run_experiment(experiment: str, confirm: bool = False) -> dict:
    """Start an ND experiment that was set up and SAVED in NIS-Elements'
    ND Acquisition window, by its saved name. It runs with exactly the
    saved settings (time loop, channels, save-to-file path). The call may
    not return until the experiment ends. Refused if one is already
    running. Requires confirm=True."""
    _require_confirm(confirm, "nis_run_experiment")
    return _bridge().nd_run(experiment)


def nis_finish_experiment() -> dict:
    """End the running ND experiment after its current time loop; NIS saves
    the frames captured so far. Never needs confirm - it only stops."""
    return _bridge().nd_finish()
