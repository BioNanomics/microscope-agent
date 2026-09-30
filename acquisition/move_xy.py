# move_xy.py
# ------------------------------------------------------------
# Drive the XY stage by hand, from a terminal.
#
#   python -m acquisition.move_xy                     # just show where we are
#   python -m acquisition.move_xy --by 500 0          # relative, microns
#   python -m acquisition.move_xy --to -1848 -10021   # absolute, microns
#   python -m acquisition.move_xy --by 20000 0 --yes  # skip the confirmation
#
# XY ONLY, ON PURPOSE. Z is what drives the objective into the sample, and
# it is not reachable from here - use the focus knob. This mirrors the
# same choice in the MCP tools.
#
# LONG MOVES ARE SPLIT AUTOMATICALLY. nis_sdk refuses any single step over
# MAX_XY_STEP_UM (5 mm) so that a fat-fingered coordinate cannot become a
# stage-length dash. That guard is worth keeping, but it makes a legitimate
# 15 mm move fail halfway with the stage somewhere unintended - which is
# exactly what happened on 2026-09-21, leaving the stage 8 mm from where
# anyone thought it was. So this walks the distance in legal hops and
# reports where it actually ended up.
#
# ANYTHING OVER --confirm-above ASKS FIRST, because the difference between
# a 200 um nudge and a 20000 um traverse is one keystroke, and only one of
# them can put the objective under the dish holder.
#
# The e-stop is checked by nis_sdk on every hop, so engaging it mid-move
# stops this at the next hop rather than at the end of the journey.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from acquisition import estop

#: Kept under nis_sdk's 5000 um hard limit so a rounding error at the
#: boundary cannot trip the very guard we are working within.
HOP_UM = 4000.0

#: Moves longer than this ask for confirmation first.
CONFIRM_ABOVE_UM = 2000.0


def move_to(sdk, x: float, y: float, quiet: bool = False) -> tuple[float, float]:
    """Absolute move in microns, split into legal hops."""
    while True:
        cx, cy = sdk.XY_GetPosition()
        dx, dy = x - cx, y - cy
        dist = float(np.hypot(dx, dy))
        if dist < 1.0:
            return cx, cy
        f = min(1.0, HOP_UM / dist)
        sdk.XY_Move(cx + dx * f, cy + dy * f)
        time.sleep(0.4)
        if not quiet and dist > HOP_UM:
            nx, ny = sdk.XY_GetPosition()
            print(f"    ... at ({nx:.1f}, {ny:.1f}), {dist - HOP_UM:.0f} um to go", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Move the XY stage manually (microns).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--to", nargs=2, type=float, metavar=("X", "Y"), help="absolute target")
    g.add_argument("--by", nargs=2, type=float, metavar=("DX", "DY"), help="relative offset")
    ap.add_argument("--yes", action="store_true", help="do not ask about long moves")
    ap.add_argument("--confirm-above", type=float, default=CONFIRM_ABOVE_UM)
    args = ap.parse_args()

    if estop.is_engaged():
        info = estop.details() or {}
        print("E-STOP IS ENGAGED - the stage will not move.")
        print(f"  reason: {info.get('reason', '?')}  at {info.get('engaged_at', '?')}")
        print("  release with: python -m acquisition.estop release")
        raise SystemExit(2)

    from acquisition.backends.nis_sdk import NISSdk
    sdk = NISSdk()
    x0, y0 = sdk.XY_GetPosition()
    z0 = sdk.Z_GetPosition()
    print(f"current: X={x0:.2f}  Y={y0:.2f}   (Z={z0:.2f}, not touched)")

    if not args.to and not args.by:
        return

    tx, ty = (args.to[0], args.to[1]) if args.to else (x0 + args.by[0], y0 + args.by[1])
    dist = float(np.hypot(tx - x0, ty - y0))
    print(f"target : X={tx:.2f}  Y={ty:.2f}   (moving {dist:.1f} um)")

    if dist > args.confirm_above and not args.yes:
        # Interactive only: piped/automated use must pass --yes explicitly
        # rather than have a prompt silently read EOF and proceed.
        if not sys.stdin.isatty():
            print("long move needs --yes when not run interactively."); raise SystemExit(3)
        if input(f"  move {dist:.0f} um? [y/N] ").strip().lower() not in ("y", "yes"):
            print("  cancelled - nothing moved."); return

    try:
        x1, y1 = move_to(sdk, tx, ty)
    except estop.EStopEngaged as exc:
        cx, cy = sdk.XY_GetPosition()
        print(f"\nSTOPPED BY E-STOP at ({cx:.2f}, {cy:.2f}) - target not reached.")
        print(f"  {exc}")
        raise SystemExit(2)

    err = float(np.hypot(x1 - tx, y1 - ty))
    print(f"arrived: X={x1:.2f}  Y={y1:.2f}   (moved {np.hypot(x1-x0, y1-y0):.1f} um, "
          f"{err:.2f} um from target)")


if __name__ == "__main__":
    main()
