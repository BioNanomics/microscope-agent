# estop.py
# ------------------------------------------------------------
# Emergency stop for every process that can move this microscope.
#
#   python -m acquisition.estop engage        # STOP EVERYTHING, NOW
#   python -m acquisition.estop status
#   python -m acquisition.estop release       # deliberate, separate action
#
# WHY THIS EXISTS. On 2026-09-21 a mosaic raster with a bad focus plane
# began driving Z toward the sample. Stopping it took several minutes and
# three separate interventions, because every mechanism available was the
# wrong shape:
#
#   * stopping the background SHELL left its python CHILD running, still
#     issuing moves - the process list, not the task list, was the truth
#   * a tool call the operator REJECTED had already spawned its process,
#     which kept looping Z moves regardless of the rejection
#   * a "STOP file" checked once per mosaic tile is not a stop; between
#     checks it still issued dozens of moves
#   * the Ti2 SDK has NO abort/halt command (searched: only ZESCAPE, which
#     is a retract, not a stop), so there is nothing to "cancel" with
#
# The lesson is that an e-stop must sit BELOW whatever is misbehaving. So
# this is enforced inside nis_sdk's move primitives themselves: no caller,
# no matter how confused, can move an axis while the stop is engaged,
# because the check happens after the caller has already decided to move.
#
# WHY A FILE. It must work across processes that do not know about each
# other - that was the whole failure. A file is visible to every process,
# survives the death of whoever set it, needs no server or port, and can
# be set by a human from any shell (or by creating the file by hand) when
# the agent itself is the thing malfunctioning.
#
# WHY A FIXED PATH, NOT data_root(). data_root() resolves relative to the
# working directory, and an MCP server launched by a desktop client gets a
# different cwd than a terminal. Two processes would then disagree about
# where the e-stop lives, which is precisely the failure this must not
# have. ESTOP_PATH is absolute and identical for every process on the
# machine.
#
# FAIL SAFE: if the state cannot be determined (unreadable directory, odd
# filesystem error), check() treats the stop as ENGAGED. An e-stop that
# fails open is not a safety device. The cost of the opposite error is a
# refused move and a clear message, which is recoverable; the cost of
# failing open is a driven objective.
#
# WHAT IT DOES NOT DO: it cannot un-issue a setpoint the controller has
# already accepted. engage() therefore also calls halt(), which writes the
# CURRENT position back as the target - with setpoint-based motion that is
# the only available way to stop an axis already in motion.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

#: Absolute and machine-wide. Never derive this from the working
#: directory - see the module header.
ESTOP_PATH = Path(os.environ.get("CONFOCAL_ESTOP_FILE",
                                 Path.home() / ".confocal-mcp" / "ESTOP"))


class EStopEngaged(RuntimeError):
    """Raised by any motion primitive while the e-stop is engaged."""


def is_engaged() -> bool:
    """True if motion is currently forbidden.

    Any error resolving the state counts as engaged - see FAIL SAFE.
    """
    try:
        return ESTOP_PATH.exists()
    except OSError:
        return True


def details() -> dict | None:
    """Who engaged the stop, when, and why - or None if it is clear."""
    try:
        if not ESTOP_PATH.exists():
            return None
        return json.loads(ESTOP_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Present but unreadable still means engaged; say so without a reason.
        return {"reason": "(e-stop file present but unreadable)"}


def check() -> None:
    """Raise EStopEngaged if motion is forbidden. Called by every move."""
    if is_engaged():
        info = details() or {}
        raise EStopEngaged(
            "E-STOP ENGAGED - motion refused. "
            f"engaged_at={info.get('engaged_at', '?')} "
            f"by={info.get('by', '?')} reason={info.get('reason', '?')}. "
            f"Release with: python -m acquisition.estop release  "
            f"(or delete {ESTOP_PATH})"
        )


def engage(reason: str = "manual", halt_stage: bool = True) -> dict:
    """Forbid all motion immediately, and try to halt anything in flight.

    The flag is written FIRST and the hardware halt attempted second: if
    halting raises (SDK missing, COM busy, no hardware), the prohibition is
    already in force. Doing it the other way round would leave a window
    where a failed halt also meant no flag.
    """
    info = {
        "engaged_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "by": f"pid {os.getpid()}",
        "reason": reason,
    }
    ESTOP_PATH.parent.mkdir(parents=True, exist_ok=True)
    ESTOP_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")

    if halt_stage:
        try:
            info["halt"] = halt()
        except Exception as exc:                      # hardware may be absent
            info["halt"] = f"not halted ({type(exc).__name__}: {exc})"
    return info


def release() -> None:
    """Allow motion again. Deliberately a separate, explicit action."""
    try:
        ESTOP_PATH.unlink()
    except FileNotFoundError:
        pass


def halt() -> str:
    """Stop axes already in motion by re-commanding their current position.

    The Ti2 exposes no abort: a move is a setpoint write and the stage
    servos to it. Writing the position it is at right now is therefore the
    only way to stop an axis mid-travel.

    Deliberately bypasses nis_sdk's XY_Move/Z_Move - those now refuse to
    run while the stop is engaged, and a stop primitive that the stop
    itself blocks would be useless.
    """
    from acquisition.backends.nis_sdk import NISSdk, XY_COUNTS_PER_UM, Z_COUNTS_PER_UM

    sdk = NISSdk()

    def freeze(m):
        x, y, z = m.iXPOSITION, m.iYPOSITION, m.iZPOSITION
        m.iXPOSITION, m.iYPOSITION, m.iZPOSITION = x, y, z
        return (x / XY_COUNTS_PER_UM, y / XY_COUNTS_PER_UM, z / Z_COUNTS_PER_UM)

    x, y, z = sdk._thread.call(freeze)
    return f"halted at x={x:.1f} y={y:.1f} z={z:.2f} um"


def main() -> None:
    ap = argparse.ArgumentParser(description="Emergency stop for microscope motion.")
    ap.add_argument("action", choices=("engage", "release", "status"))
    ap.add_argument("--reason", default="manual")
    ap.add_argument("--no-halt", action="store_true",
                    help="set the flag only; do not try to halt the stage")
    args = ap.parse_args()

    if args.action == "engage":
        info = engage(args.reason, halt_stage=not args.no_halt)
        print("E-STOP ENGAGED")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print(f"\nflag file: {ESTOP_PATH}")
        print("All motion is now refused. Release with: python -m acquisition.estop release")
    elif args.action == "release":
        release()
        print(f"e-stop released - motion permitted again ({ESTOP_PATH} removed)")
    else:
        if is_engaged():
            print("ENGAGED - motion is refused")
            for k, v in (details() or {}).items():
                print(f"  {k}: {v}")
            sys.exit(2)
        print(f"clear - motion permitted (no {ESTOP_PATH})")


if __name__ == "__main__":
    main()
