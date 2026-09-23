# mosaic.py
# ------------------------------------------------------------
# Whole-dish time-lapse by tiling: raster a grid of fields with the stage,
# stitch them into one large image, and repeat the grid on an interval.
#
#   # plan only - prints the grid, moves nothing
#   python -m acquisition.orchestration.mosaic --grid 6x6 --dry-run
#
#   # one mosaic now, to check focus and framing before committing
#   python -m acquisition.orchestration.mosaic --grid 6x6 --rounds 1
#
#   # overnight: a mosaic every 20 min for 12 h
#   python -m acquisition.orchestration.mosaic --grid 6x6 --interval-min 20 --hours 12
#
# WHY TILING RATHER THAN A SECOND CAMERA: the Baumer has no lens of its
# own - it is a bare sensor fed by the microscope optics, so it cannot
# image a whole dish directly. The stage, however, travels +/-57 mm in X
# and +/-37.5 mm in Y, far more than a 35 mm dish. So the wide view is
# assembled from many narrow ones.
#
# STITCHING IS BY STAGE COORDINATES, NOT FEATURE MATCHING. The stage is
# accurate (commanded 150.00 um -> reported 150.00) and the image scale
# was measured (1.4901 um/px), so every tile's position on the canvas is
# computed, not searched for. That matters for a living sample: feature
# matching on an organism that moves between tiles can align to the
# organism instead of the substrate and warp the mosaic. Geometry cannot.
#
# THE STAGE->IMAGE MAPPING IS MEASURED, NOT ASSUMED. calibrate_axes()
# moves in X and then in Y and phase-correlates each, recovering a full
# 2x2 matrix. This gets the SIGNS right - image Y usually points down
# while stage Y points up, so assuming a sign is a coin flip that
# silently mirrors the mosaic - and absorbs any small camera rotation.
# Pass --scale to skip it only if you already know the mapping.
#
# LIGHT: Physarum is negatively phototactic, and a tiled run illuminates
# any given spot only for the moment its tile is taken rather than
# continuously. That is gentler than parking on one field for hours - a
# real advantage of this approach for an overnight run, not just a
# side effect.
#
# FOCUS IS THE KNOWN WEAKNESS: Z is held fixed across the whole grid, so
# a non-level agar surface will drift out of focus toward the edges.
# Run --rounds 1 first and look at the corner tiles before trusting an
# overnight run. Each tile's focus score is written to tiles.csv so the
# drift is measurable rather than a surprise in the morning.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from acquisition.paths import data_root

#: Per-call stage limit in nis_sdk. Longer hops are broken into chunks.
_MAX_STEP_UM = 4500.0

#: Seconds to let the stage settle before grabbing. The stage reports the
#: move complete before vibration has died away; a frame taken too early
#: is motion-blurred in a way no later processing can undo.
SETTLE_S = 0.8


def _move_safe(sdk, x: float, y: float) -> tuple[float, float]:
    """Absolute move, split into legal-sized hops.

    nis_sdk refuses a single step over MAX_XY_STEP_UM, which the jump from
    the end of one grid back to the start of the next will exceed.
    """
    while True:
        cx, cy = sdk.XY_GetPosition()
        dx, dy = x - cx, y - cy
        dist = float(np.hypot(dx, dy))
        if dist < 1.0:
            return cx, cy
        if dist <= _MAX_STEP_UM:
            sdk.XY_Move(x, y)
            return sdk.XY_GetPosition()
        f = _MAX_STEP_UM / dist
        sdk.XY_Move(cx + dx * f, cy + dy * f)


#: A computed focus Z may never depart from the reference focus by more
#: than this. The sample tilt actually measured across a 15.6 mm span was
#: 306 um, so 400 um is generous for any real surface while still being
#: far short of the objective's working distance. See FocusPlane.
MAX_FOCUS_DEVIATION_UM = 400.0


class FocusPlane:
    """z = z_ref + b*(x-x0) + c*(y-y0), fitted to measured focus points.

    THIS CLASS EXISTS BECAUSE A NAIVE FIT DROVE THE OBJECTIVE AT THE DISH.
    On 2026-09-21 a plane was fitted to TWO points with
    `lstsq` on [1, x, y]. Two points cannot determine three coefficients,
    so lstsq returned its minimum-norm solution - and because it minimises
    a^2+b^2+c^2, and x,y are ~10^4 while z is ~4.6x10^3, the cheapest way
    to satisfy both equations was to collapse the intercept to ZERO and
    use enormous gradients: z = 0 + 0.209*x + 1.470*y. At the far corner
    of the grid that evaluates to 10734 um against a true focus of 4573 -
    about 6 mm of objective travel toward the sample. The raster was
    stopped at tile 15 of 48 before reaching it.
    So this class refuses the conditions that produced that:
      * at least THREE points, and not collinear - anything less cannot
        define a plane, and pretending otherwise is what caused the fault
      * coordinates are centred before fitting, so the free parameter is
        the mean measured focus rather than an intercept at x=y=0 tens of
        millimetres outside the sample
      * every evaluated Z is hard-clamped to MAX_FOCUS_DEVIATION_UM around
        the reference, so no arithmetic error downstream can command a
        large move even if the fit is somehow still wrong
    """

    def __init__(self, points: list[tuple[float, float, float]],
                 max_dev: float = MAX_FOCUS_DEVIATION_UM):
        if len(points) < 3:
            raise ValueError(
                f"a focus plane needs at least 3 measured points, got {len(points)}. "
                "Two points cannot define a plane - fitting them is what drove the "
                "objective 6 mm off focus on 2026-09-21. Use a fixed Z instead.")
        pts = np.asarray(points, dtype=float)
        self.x0, self.y0 = pts[:, 0].mean(), pts[:, 1].mean()
        dx, dy = pts[:, 0] - self.x0, pts[:, 1] - self.y0
        # Collinear points leave the perpendicular gradient undetermined,
        # which is the same degenerate case by another route.
        spread = np.linalg.svd(np.column_stack([dx, dy]), compute_uv=False)
        if spread.min() < 1e-6 * max(spread.max(), 1.0):
            raise ValueError(
                "focus points are collinear - they cannot determine tilt "
                "perpendicular to the line joining them. Add a point off that line.")
        A = np.column_stack([np.ones_like(dx), dx, dy])
        coef, *_ = np.linalg.lstsq(A, pts[:, 2], rcond=None)
        self.z_ref, self.b, self.c = (float(v) for v in coef)
        self.max_dev = float(max_dev)
        self.residuals = [abs(self.z(px, py) - pz) for px, py, pz in points]

    def z(self, x: float, y: float) -> float:
        raw = self.z_ref + self.b * (x - self.x0) + self.c * (y - self.y0)
        return float(np.clip(raw, self.z_ref - self.max_dev, self.z_ref + self.max_dev))

    def would_clamp(self, x: float, y: float) -> bool:
        raw = self.z_ref + self.b * (x - self.x0) + self.c * (y - self.y0)
        return abs(raw - self.z_ref) > self.max_dev

    def __str__(self) -> str:
        return (f"z = {self.z_ref:.2f} {self.b:+.5f}*(x-{self.x0:.0f}) "
                f"{self.c:+.5f}*(y-{self.y0:.0f})  clamped to +/-{self.max_dev:.0f} um")


def _move_z(sdk, z: float, limit: float = 45.0) -> float:
    """Z in steps under nis_sdk's 50 um per-call cap."""
    for _ in range(200):
        cz = sdk.Z_GetPosition()
        d = z - cz
        if abs(d) < 0.3:
            return cz
        sdk.Z_Move(cz + max(-limit, min(limit, d)))
        time.sleep(0.2)
    return sdk.Z_GetPosition()


def _grab_gray(cam) -> np.ndarray:
    p = cam.capture()
    a = np.asarray(Image.open(p)).mean(axis=2).astype(np.float32)
    p.unlink(missing_ok=True)
    return a


def calibrate_axes(cam, sdk, step_um: float = 150.0) -> np.ndarray:
    """Measure M where (image shift in px) = M @ (stage delta in um).

    Returns a 2x2 matrix. Inverting the sign of M gives where a tile
    belongs on the canvas: if moving the stage +X slides features left in
    the image, then the tile taken at larger X shows sample further right.
    """
    x0, y0 = sdk.XY_GetPosition()
    base = _grab_gray(cam)
    win = cv2.createHanningWindow((base.shape[1], base.shape[0]), cv2.CV_32F)

    cols = []
    for axis in (0, 1):
        tx = x0 + (step_um if axis == 0 else 0.0)
        ty = y0 + (step_um if axis == 1 else 0.0)
        _move_safe(sdk, tx, ty)
        time.sleep(SETTLE_S + 0.7)
        xa, ya = sdk.XY_GetPosition()
        actual = (xa - x0) if axis == 0 else (ya - y0)
        moved = _grab_gray(cam)
        (dx, dy), resp = cv2.phaseCorrelate(base, moved, win)
        if abs(actual) < 1.0 or resp < 0.05:
            raise SystemExit(
                f"axis {'XY'[axis]} calibration failed (moved {actual:.2f} um, "
                f"confidence {resp:.3f}) - too little texture in view, or the stage did not move"
            )
        cols.append([dx / actual, dy / actual])
        print(f"  axis {'XY'[axis]}: moved {actual:+.2f} um -> image ({dx:+.2f}, {dy:+.2f}) px, "
              f"confidence {resp:.3f}")
        _move_safe(sdk, x0, y0)
        time.sleep(SETTLE_S)

    M = np.array(cols).T                       # columns are the X and Y responses
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    print(f"  => {1.0 / scale:.4f} um/px" if scale else "  => degenerate mapping")
    return M


def grid_positions(cx: float, cy: float, nx: int, ny: int,
                   step_x: float, step_y: float) -> list[tuple[float, float]]:
    """Serpentine raster - each row reverses, so the stage never makes a
    long dash back to the start of the next row. Less travel, less
    vibration, less time per mosaic."""
    xs = (np.arange(nx) - (nx - 1) / 2.0) * step_x + cx
    ys = (np.arange(ny) - (ny - 1) / 2.0) * step_y + cy
    out = []
    for j, y in enumerate(ys):
        row = xs if j % 2 == 0 else xs[::-1]
        out.extend((float(x), float(y)) for x in row)
    return out


def stitch(tiles: list[tuple[np.ndarray, float, float]], M: np.ndarray,
           ref: tuple[float, float]) -> np.ndarray:
    """Paste tiles onto one canvas using their stage coordinates."""
    Minv = -M                                   # canvas offset is the negated image shift
    offs = [Minv @ np.array([sx - ref[0], sy - ref[1]]) for _, sx, sy in tiles]
    h, w = tiles[0][0].shape[:2]
    ox = np.array([o[0] for o in offs]); oy = np.array([o[1] for o in offs])
    ox -= ox.min(); oy -= oy.min()
    # Size from the ROUNDED placements, not the raw maxima: a tile whose
    # offset is x.6 is written at x+1, which overruns a canvas sized on
    # int(x). Rounding first makes the bound exact.
    xi_all = np.round(ox).astype(int); yi_all = np.round(oy).astype(int)
    canvas = np.zeros((int(yi_all.max()) + h, int(xi_all.max()) + w, 3), np.uint8)
    for (img, _, _), xi, yi in zip(tiles, xi_all, yi_all):
        canvas[yi:yi + h, xi:xi + w] = img
    return canvas


def run(grid: tuple[int, int], overlap: float, rounds: int, interval_s: float,
        exposure_us: float, mosaic_scale: float, keep_tiles: bool,
        scale_um_px: float | None, tag: str | None,
        focus_points: list[tuple[float, float, float]] | None = None,
        centre: tuple[float, float] | None = None) -> Path:
    from acquisition.backends.baumer_genicam import BaumerGenICam
    from acquisition.backends.nis_sdk import NISSdk

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = data_root() / "data" / (f"mosaic_{stamp}" + (f"_{tag}" if tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    sdk = NISSdk()
    cam = BaumerGenICam()
    cx = cy = 0.0
    try:
        cam.set_settings(exposure_time_us=exposure_us, gain=1.0)
        # AUTO WHITE BALANCE MUST BE OFF FOR A MOSAIC. Left on Continuous,
        # the camera re-neutralises every tile independently - a tile of
        # bare agar and a tile of organism then get different corrections,
        # and the stitched result has visible colour seams at every tile
        # boundary that no amount of post-processing can unpick. It resets
        # itself to Continuous across power cycles, so force it per run
        # rather than trusting whatever it was left in.
        try:
            node = cam._acquirer.remote_device.node_map.BalanceWhiteAuto
            if node.value != "Off":
                print(f"BalanceWhiteAuto was {node.value!r} - forcing 'Off' for consistent tiles")
                node.value = "Off"
        except Exception as exc:
            print(f"WARNING: could not force BalanceWhiteAuto off ({exc}); "
                  "tiles may not colour-match", file=sys.stderr)
        print(f"camera: {cam.get_settings()}")
        if centre is not None:
            cx, cy = centre
            _move_safe(sdk, cx, cy)
            cx, cy = sdk.XY_GetPosition()
        else:
            cx, cy = sdk.XY_GetPosition()
        print(f"grid centre: ({cx:.1f}, {cy:.1f}) um")

        plane = None
        z_fixed = sdk.Z_GetPosition()
        if focus_points:
            plane = FocusPlane(focus_points)
            print(f"focus plane {plane}  (from {len(focus_points)} points, "
                  f"max residual {max(plane.residuals):.1f} um)")
        else:
            print(f"FIXED focus: Z held at {z_fixed:.2f} um for every tile "
                  f"(no --focus-plane given)")

        if scale_um_px:
            M = np.array([[1.0 / scale_um_px, 0.0], [0.0, 1.0 / scale_um_px]])
            print(f"using supplied scale {scale_um_px} um/px (axes assumed square and unflipped)")
        else:
            print("calibrating stage->image mapping...")
            M = calibrate_axes(cam, sdk)

        probe = _grab_gray(cam)
        h, w = probe.shape
        umpx = 1.0 / float(np.hypot(M[0, 0], M[1, 0]))
        step_x = w * umpx * (1.0 - overlap)
        step_y = h * umpx * (1.0 - overlap)
        nx, ny = grid
        pts = grid_positions(cx, cy, nx, ny, step_x, step_y)
        print(f"{nx}x{ny} tiles, step {step_x:.0f} x {step_y:.0f} um "
              f"-> covers {nx * step_x / 1000:.2f} x {ny * step_y / 1000:.2f} mm")

        (out / "mosaic.json").write_text(json.dumps({
            "grid": [nx, ny], "overlap": overlap, "um_per_px": round(umpx, 4),
            "step_um": [round(step_x, 1), round(step_y, 1)],
            "centre_um": [cx, cy], "exposure_us": exposure_us,
            "stage_to_image_matrix": M.tolist(),
            "covers_mm": [round(nx * step_x / 1000, 3), round(ny * step_y / 1000, 3)],
            # The saved mosaic_NNN.png is resized by this factor, so its
            # pixels are um_per_px / mosaic_scale - make_movie needs it
            # to draw a scale bar that is true on the stitched image.
            "mosaic_scale": mosaic_scale,
        }, indent=2), encoding="utf-8")

        # An out-of-band kill switch. Stopping a background shell does NOT
        # always take its python child with it - on 2026-09-21 a stopped
        # raster kept driving the stage afterwards. Creating this file makes
        # the run halt itself at the next tile boundary, without needing to
        # find and kill a process.
        stop_file = out / "STOP"
        print(f"to halt this run at any point, create: {stop_file}")

        fields = ("round", "tile", "wall_clock", "x_um", "y_um", "z_um",
                  "mean", "focus", "error")
        log = (out / "tiles.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()

        t_start = time.monotonic()
        for r in range(rounds):
            target = t_start + r * interval_s
            if time.monotonic() < target:
                time.sleep(target - time.monotonic())
            r_dir = out / f"round_{r:03d}"
            if keep_tiles:
                r_dir.mkdir(exist_ok=True)
            print(f"\n[round {r + 1}/{rounds}] {datetime.datetime.now():%H:%M:%S}", flush=True)

            tiles = []
            for i, (tx, ty) in enumerate(pts):
                row = {k: "" for k in fields}
                row.update(round=r, tile=i,
                           wall_clock=datetime.datetime.now().isoformat(timespec="seconds"))
                try:
                    if stop_file.exists():
                        print(f"  STOP FILE {stop_file.name} present - halting", flush=True)
                        raise KeyboardInterrupt
                    ax, ay = _move_safe(sdk, tx, ty)
                    if plane is not None:
                        _move_z(sdk, plane.z(ax, ay))
                    row["z_um"] = round(sdk.Z_GetPosition(), 2)
                    time.sleep(SETTLE_S)
                    p = cam.capture()
                    img = np.asarray(Image.open(p))
                    if keep_tiles:
                        p.replace(r_dir / f"t{i:03d}.png")
                    else:
                        p.unlink(missing_ok=True)
                    g = img.mean(axis=2)
                    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
                           - 4 * g[1:-1, 1:-1])
                    row.update(x_um=round(ax, 1), y_um=round(ay, 1),
                               mean=round(float(g.mean()), 2), focus=round(float(lap.var()), 2))
                    tiles.append((img, ax, ay))
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"  tile {i}: {row['error']}", flush=True)
                writer.writerow(row); log.flush()

            if tiles:
                canvas = stitch(tiles, M, (cx, cy))
                if mosaic_scale != 1.0:
                    canvas = cv2.resize(
                        canvas, (int(canvas.shape[1] * mosaic_scale),
                                 int(canvas.shape[0] * mosaic_scale)),
                        interpolation=cv2.INTER_AREA)
                dest = out / f"mosaic_{r:03d}.png"
                Image.fromarray(canvas).save(dest)
                print(f"  {len(tiles)}/{len(pts)} tiles -> {dest.name} "
                      f"({canvas.shape[1]}x{canvas.shape[0]})", flush=True)

        log.close()
    finally:
        try:
            _move_safe(sdk, cx, cy)
            print(f"returned to grid centre ({cx:.1f}, {cy:.1f}) um")
        except Exception as exc:
            print(f"WARNING: could not return to centre: {exc}", file=sys.stderr)
        cam.close()
        print("released")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="6x6", help="tiles as NXxNY (default 6x6)")
    ap.add_argument("--overlap", type=float, default=0.10, help="tile overlap fraction")
    ap.add_argument("--rounds", type=int, help="number of mosaics (default: from --hours)")
    ap.add_argument("--hours", type=float, help="run length; with --interval-min sets rounds")
    ap.add_argument("--interval-min", type=float, default=20.0)
    ap.add_argument("--exposure-us", type=float, default=10000.0)
    ap.add_argument("--mosaic-scale", type=float, default=0.5,
                    help="downsample the stitched mosaic (tiles are kept full-res)")
    ap.add_argument("--no-tiles", action="store_true", help="do not keep individual tiles")
    ap.add_argument("--scale", type=float, help="known um/px; skips axis calibration")
    ap.add_argument("--tag")
    ap.add_argument("--focus-plane",
                    help="measured focus references as 'x,y,z;x,y,z;...' (um). Z is "
                         "interpolated per tile from a least-squares plane through them")
    ap.add_argument("--centre", help="grid centre as 'x,y' (um); default is the current position")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, move nothing")
    args = ap.parse_args()

    fpts = None
    if args.focus_plane:
        fpts = [tuple(float(v) for v in grp.split(","))
                for grp in args.focus_plane.split(";") if grp.strip()]
        if any(len(p) != 3 for p in fpts):
            ap.error("--focus-plane entries must each be x,y,z")
    ctr = tuple(float(v) for v in args.centre.split(",")) if args.centre else None

    nx, ny = (int(v) for v in args.grid.lower().split("x"))
    if args.rounds:
        rounds = args.rounds
    elif args.hours:
        rounds = max(1, int(round(args.hours * 60 / args.interval_min)))
    else:
        rounds = 1

    if args.dry_run:
        umpx = args.scale or 1.4901
        sx = 1920 * umpx * (1 - args.overlap) / 1000
        sy = 1200 * umpx * (1 - args.overlap) / 1000
        print(f"{nx}x{ny} = {nx * ny} tiles, step {sx:.3f} x {sy:.3f} mm")
        print(f"covers {nx * sx:.2f} x {ny * sy:.2f} mm at {umpx} um/px")
        print(f"{rounds} rounds every {args.interval_min} min "
              f"= {rounds * args.interval_min / 60:.1f} h")
        print(f"~{nx * ny * 3:.0f} MB of tiles per round, ~{rounds * nx * ny * 3 / 1000:.1f} GB total")
        return

    run((nx, ny), args.overlap, rounds, args.interval_min * 60.0, args.exposure_us,
        args.mosaic_scale, not args.no_tiles, args.scale, args.tag, fpts, ctr)


if __name__ == "__main__":
    main()
