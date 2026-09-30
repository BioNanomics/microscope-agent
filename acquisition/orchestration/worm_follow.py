# worm_follow.py
# ------------------------------------------------------------
# Follow one crawling worm across a dish: take a small mosaic (3x3 by
# default) around it, find the worm in the stitched image, re-centre the
# next mosaic on it, and repeat back-to-back. The result is the worm's
# path across the dish, one point every ~15 s, plus a mosaic per round.
#
#   # detection check on a saved image, moves nothing
#   python -m acquisition.orchestration.worm_follow --detect-only img.png --um-per-px 1.49
#
#   # one round, to check focus, exposure and that the worm is found
#   python -m acquisition.orchestration.worm_follow --rounds 1
#
#   # follow for an hour
#   python -m acquisition.orchestration.worm_follow --minutes 60 --tag celegans
#
# START WITH THE WORM IN THE MIDDLE OF THE FIELD. The first round takes
# the qualifying object nearest the grid centre; after that each round
# takes the one nearest the last position. Debris the size of an adult
# worm is rare, so size plus proximity is enough to stay locked on.
#
# WHY A SMALL MOSAIC AND NOT THE WHOLE DISH. A crawling adult covers
# 0.1-0.2 mm/s. A whole-dish round takes ~4.5 min, in which the worm can
# cross the dish, and tiles one row apart are ~25 s apart, so it is cut
# at seams or appears twice. A 3x3 round is ~13 s: the worm moves at most
# ~2 mm, so it stays inside the 7.7 x 4.8 mm block and is found again.
#
# WHY A DEADBAND ON RE-CENTRING. Worms react to vibration (tap response).
# The stage only jumps when the worm has moved more than --deadband-mm
# from the block centre, so a dwelling worm is not shaken every round.
#
# SAFETY: every re-centre is clamped to --limit-mm around the dish centre
# (default: where the run started), so a misdetection cannot walk the
# stage off the glass window. Z is fixed - the worm crawls on the agar
# surface, and a 3x3 block is small enough that tilt barely matters.
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

from acquisition.orchestration.mosaic import (SETTLE_S, _move_safe, calibrate_axes,
                                               grid_positions)
from acquisition.paths import data_root

#: Stage->image mapping measured at 4x on 2026-09-24 (night-2 Physarum
#: run, 1.4877 um/px). Same camera, same objective, so reused rather than
#: re-measured - calibration needs texture, and clean agar has little.
M_4X = np.array([[-0.6721844559295096, 0.0007000999870820124],
                 [0.0007312652036724406, 0.6711263094676038]])

#: Detection runs on the stitched block shrunk by this factor (~6 um/px
#: at 4x). An adult is ~50 um wide, so still ~8 px across.
DETECT_SCALE = 0.25


def force_camera_autos_off(cam) -> None:
    """Any auto colour/exposure function drifts a series mid-run
    (2026-09-23: ColorTransformationAuto turned mosaics green). Camera
    Explorer can switch them back on, so force them per run."""
    nm = cam._acquirer.remote_device.node_map
    for name in ("BalanceWhiteAuto", "ColorTransformationAuto", "ExposureAuto", "GainAuto"):
        try:
            node = getattr(nm, name)
            if node.value != "Off":
                print(f"{name} was {node.value!r} - forcing 'Off'")
                node.value = "Off"
        except Exception:
            pass
    # Explorer also leaves its own colour pipeline behind: on 2026-09-29 it
    # had Gamma 1.45, an adapted colour matrix that zeroed green, and red/
    # blue gains of 5.9/4.4 - every 4x frame was solid 255. Restore the
    # linear, neutral state measured on clear agar on 2026-09-23.
    try:
        nm.Gamma.value = 1.0
        for s in nm.ColorTransformationValueSelector.symbolics:
            nm.ColorTransformationValueSelector.value = s
            nm.ColorTransformationValue.value = 1.0 if s[-2] == s[-1] else 0.0
        for s, v in (("Red", 1.689), ("GreenRed", 1.0), ("GreenBlue", 1.0), ("Blue", 2.884)):
            nm.GainSelector.value = s
            nm.Gain.value = v
        nm.GainSelector.value = "All"
        print("camera colour: Gamma 1.0, identity matrix, gains R 1.689 / G 1.0 / B 2.884")
    except Exception as exc:
        print(f"WARNING: could not restore camera colour state ({exc})", file=sys.stderr)


def stitch_with_origin(tiles, M, ref):
    """mosaic.stitch, but also returns the canvas origin offset so canvas
    pixels can be mapped back to stage coordinates."""
    offs = np.array([-M @ np.array([sx - ref[0], sy - ref[1]]) for _, sx, sy in tiles])
    omin = offs.min(axis=0)
    pos = np.round(offs - omin).astype(int)
    h, w = tiles[0][0].shape[:2]
    canvas = np.zeros((pos[:, 1].max() + h, pos[:, 0].max() + w, 3), np.uint8)
    for (img, _, _), (xi, yi) in zip(tiles, pos):
        canvas[yi:yi + h, xi:xi + w] = img
    return canvas, omin, (w, h)


def canvas_to_stage(px, py, M, ref, omin, tile_wh) -> tuple[float, float]:
    """Stage position that would put canvas pixel (px, py) at the image
    centre. A feature at stage-centre S appears in a tile taken at stage s
    at c + M(s - S); with the tile's canvas offset -M(s-ref) - omin this
    reduces to S = ref - M^-1 (P + omin - c), independent of the tile."""
    c = np.array(tile_wh, float) / 2.0
    S = np.array(ref) - np.linalg.solve(M, np.array([px, py]) + omin - c)
    return float(S[0]), float(S[1])


def find_worms(rgb: np.ndarray, um_per_px: float, min_area_mm2: float,
               max_area_mm2: float, dark_ratio: float) -> tuple[list[dict], np.ndarray]:
    """Dark objects of worm size against a locally-estimated background.

    Returns candidates (full-resolution centroid, area, length) and the
    shrunk mask for inspection. The background is a morphological CLOSE
    with a kernel wider than a worm: that fills in anything dark and thin
    while following slow illumination falloff and vignetting, so the
    threshold is relative to the local agar rather than a global level.
    """
    small = cv2.resize(rgb, None, fx=DETECT_SCALE, fy=DETECT_SCALE, interpolation=cv2.INTER_AREA)
    g = small.mean(axis=2).astype(np.float32)
    # Canvas outside every tile is zero. The camera is rotated ~0.06 deg
    # to the stage, so the stitched block has zero slivers a few px wide
    # along its edges; shrunk, they turn into thin grey lines that read as
    # dark objects. Judge validity at full resolution (a 2 px sliver is
    # only half a shrunk pixel, never dark enough to test as empty), keep
    # only shrunk pixels that were entirely inside tiles, then erode.
    full = rgb.any(axis=2).astype(np.float32)
    valid = cv2.resize(full, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA) > 0.999
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    upx = um_per_px / DETECT_SCALE
    k = int(round(150 / upx)) | 1               # 150 um: 3x a worm's width
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    mask = ((g < dark_ratio * np.maximum(bg, 1)) & valid).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    px_mm2 = (upx / 1000.0) ** 2
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA] * px_mm2
        if not (min_area_mm2 <= area <= max_area_mm2):
            continue
        comp = (lab == i).astype(np.uint8)
        # Skeleton-free length estimate: area / mean width, with width from
        # the distance transform. Curled worms still read as long.
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        width_um = max(2.0 * float(dt.max()) * upx, 1.0)
        length_mm = area / (width_um / 1000.0)
        out.append({"x": cents[i][0] / DETECT_SCALE, "y": cents[i][1] / DETECT_SCALE,
                    "area_mm2": area, "width_um": width_um, "length_mm": length_mm})
    return out, mask * 255


def pick(cands: list[dict], near_px: tuple[float, float], max_px: float,
         area_mm2: float | None = None) -> dict | None:
    """Nearest candidate within max_px. With area_mm2 (the tracked worm's
    last area), candidates more than 2x bigger or smaller are ignored, so
    the track does not hop onto a larva or another adult passing close by
    - plates are crowded (2026-09-29: a second adult 2.5 mm away)."""
    best = None
    for c in cands:
        if area_mm2 and not (0.5 <= c["area_mm2"] / area_mm2 <= 2.0):
            continue
        d = float(np.hypot(c["x"] - near_px[0], c["y"] - near_px[1]))
        if d <= max_px and (best is None or d < best["dist_px"]):
            best = {**c, "dist_px": d}
    return best


def detect_only(path: Path, umpx: float, args) -> None:
    rgb = np.asarray(Image.open(path).convert("RGB"))
    cands, mask = find_worms(rgb, umpx, args.min_area_mm2, args.max_area_mm2, args.dark_ratio)
    h, w = rgb.shape[:2]
    for c in cands:
        print(f"  ({c['x']:.0f}, {c['y']:.0f}) px  area {c['area_mm2']:.4f} mm2  "
              f"width {c['width_um']:.0f} um  length ~{c['length_mm']:.2f} mm")
    best = pick(cands, (w / 2, h / 2), float("inf"))
    print(f"{len(cands)} candidate(s); picked: {best and (round(best['x']), round(best['y']))}")
    Image.fromarray(mask).save(path.with_name(path.stem + "_mask.png"))


#: Setup focus sweep never leaves this band around the starting Z. The 4x
#: has a 20 mm working distance, so this is far from any contact; it only
#: has to cover the parfocal offset from the 10x (a few tens of um).
SETUP_Z_RANGE_UM = 150.0


def _focus_score(rgb: np.ndarray) -> float:
    g = cv2.resize(rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).mean(axis=2)
    return float(cv2.Laplacian(g.astype(np.float32), cv2.CV_32F).var())


def setup(args) -> Path:
    """Put the microscope in the state a run starts from: 4x objective,
    exposure set once from the agar, focus found by a capped Z sweep, and
    the worm moved to the middle of the field. Saves every sweep frame and
    a final frame so the result can be checked by eye before a run."""
    from acquisition import estop
    from acquisition.backends.baumer_genicam import BaumerGenICam
    from acquisition.backends.nis_sdk import NISSdk
    from acquisition.orchestration.mosaic import _move_z

    estop.check()                                 # the nosepiece write below has no guard of its own
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = data_root() / "data" / f"wormsetup_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    sdk = NISSdk()
    cam = BaumerGenICam()

    def grab() -> np.ndarray:
        p = cam.capture()
        a = np.asarray(Image.open(p)); p.unlink(missing_ok=True)
        return a

    try:
        z_start = sdk.Z_GetPosition()
        nose = sdk._thread.call(lambda m: int(m.iNOSEPIECE))
        if nose != args.nosepiece:
            print(f"nosepiece {nose} -> {args.nosepiece} (Z {z_start:.2f} um before)", flush=True)
            estop.check()
            sdk._thread.call(lambda m: setattr(m, "iNOSEPIECE", args.nosepiece))
            deadline = time.monotonic() + 20
            while sdk._thread.call(lambda m: int(m.iNOSEPIECE)) != args.nosepiece:
                if time.monotonic() > deadline:
                    raise SystemExit("nosepiece did not reach the requested position in 20 s")
                time.sleep(0.25)
            time.sleep(1.0)
        z0 = sdk.Z_GetPosition()
        print(f"nosepiece {args.nosepiece}, Z {z0:.2f} um", flush=True)

        force_camera_autos_off(cam)
        exp = args.exposure_us
        for _ in range(10):
            cam.set_settings(exposure_time_us=exp, gain=1.0)
            grab()                                # first frame after a change can be stale
            p90 = float(np.percentile(grab().mean(axis=2), 90))
            print(f"exposure {exp / 1000:.2f} ms -> agar 90th percentile {p90:.0f}/255", flush=True)
            if 170 <= p90 <= 230:
                break
            # A clipped frame says nothing about how far over it is, so
            # halve until it reads; the response is linear after that.
            exp = exp / 2 if p90 >= 250 else exp * 200.0 / max(p90, 1.0)
            exp = float(np.clip(exp, 100.0, 60000.0))

        def sweep(centre: float, half: float, step: float, tag: str) -> float:
            scores = []
            for z in np.arange(centre - half, centre + half + 0.1, step):
                z = float(np.clip(z, z0 - SETUP_Z_RANGE_UM, z0 + SETUP_Z_RANGE_UM))
                _move_z(sdk, z)
                time.sleep(0.4)
                img = grab()
                s = _focus_score(img)
                print(f"  {tag} Z {sdk.Z_GetPosition():.1f} um  focus {s:.1f}", flush=True)
                Image.fromarray(cv2.resize(img, None, fx=0.4, fy=0.4)).save(
                    out / f"{tag}_z{z:.0f}.jpg", quality=85)
                scores.append((s, z))
            # A flat curve (e.g. every frame clipped - 2026-09-29 sent Z to
            # the top of the range on all-zero scores) means no information:
            # stay where the sweep was centred rather than at an end.
            vals = [s for s, _ in scores]
            if max(vals) < 1.0 or max(vals) < 1.2 * min(vals):
                print(f"  {tag} sweep is flat - keeping Z {centre:.1f} um", flush=True)
                return centre
            return max(scores)[1]

        z_best = sweep(z0, SETUP_Z_RANGE_UM, 25.0, "coarse")
        _move_z(sdk, z_best)

        M = M_4X
        umpx = 1.0 / float(np.hypot(M[0, 0], M[1, 0]))
        img = grab()
        h, w = img.shape[:2]
        cands, _ = find_worms(img, umpx, args.min_area_mm2, args.max_area_mm2, args.dark_ratio)
        best = pick(cands, (w / 2, h / 2), float("inf"))
        if best:
            # A feature at image pixel q is centred by moving the stage by
            # -M^-1 (q - c) - same geometry as canvas_to_stage.
            x, y = sdk.XY_GetPosition()
            d = np.linalg.solve(M, np.array([best["x"] - w / 2, best["y"] - h / 2]))
            tx, ty = x - d[0], y - d[1]
            print(f"worm at ({best['x']:.0f}, {best['y']:.0f}) px, area {best['area_mm2']:.3f} mm2 "
                  f"-> centring: stage ({x:.1f}, {y:.1f}) -> ({tx:.1f}, {ty:.1f}) um", flush=True)
            _move_safe(sdk, tx, ty)
            time.sleep(SETTLE_S)
        else:
            print(f"NO WORM FOUND in the 4x field ({len(cands)} candidates) - stage not moved",
                  flush=True)

        z_fine = sweep(z_best, 20.0, 5.0, "fine")
        _move_z(sdk, z_fine)
        time.sleep(0.4)
        final = grab()
        cands, _ = find_worms(final, umpx, args.min_area_mm2, args.max_area_mm2, args.dark_ratio)
        b = pick(cands, (w / 2, h / 2), float("inf"))
        mark = final.copy()
        if b:
            cv2.circle(mark, (int(b["x"]), int(b["y"])), int(700 / umpx), (255, 0, 0), 4)
        Image.fromarray(final).save(out / "final.png")
        Image.fromarray(cv2.resize(mark, None, fx=0.5, fy=0.5)).save(out / "final_marked.jpg", quality=90)
        x, y = sdk.XY_GetPosition()
        summary = {"nosepiece": args.nosepiece, "z_um": sdk.Z_GetPosition(), "z_before_um": z_start,
                   "xy_um": [x, y], "exposure_us": exp, "worm_found": bool(b),
                   "worm_px": [b["x"], b["y"]] if b else None}
        (out / "setup.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"READY: {summary}\nimages in {out}", flush=True)
    finally:
        cam.close()
    return out


def run(args) -> Path:
    from acquisition.backends.baumer_genicam import BaumerGenICam
    from acquisition.backends.nis_sdk import NISSdk

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = data_root() / "data" / (f"wormfollow_{stamp}" + (f"_{args.tag}" if args.tag else ""))
    (out / "crops").mkdir(parents=True, exist_ok=True)
    stop_file = out / "STOP"

    sdk = NISSdk()
    cam = BaumerGenICam()
    cx = cy = 0.0
    try:
        force_camera_autos_off(cam)
        cam.set_settings(exposure_time_us=args.exposure_us, gain=1.0)
        cx, cy = sdk.XY_GetPosition()
        dish = tuple(float(v) for v in args.dish_centre.split(",")) if args.dish_centre else (cx, cy)
        z = sdk.Z_GetPosition()
        print(f"start ({cx:.1f}, {cy:.1f}) um, Z fixed at {z:.2f} um; "
              f"stage confined to {args.limit_mm} mm around ({dish[0]:.0f}, {dish[1]:.0f})")

        if args.calibrate:
            M = calibrate_axes(cam, sdk)
        else:
            M = M_4X
            print("using the 4x stage->image mapping measured 2026-09-24 (pass --calibrate to re-measure)")
        umpx = 1.0 / float(np.hypot(M[0, 0], M[1, 0]))

        probe = cam.capture()
        img = np.asarray(Image.open(probe)); probe.unlink(missing_ok=True)
        h, w = img.shape[:2]
        bright = float(np.percentile(img.mean(axis=2), 90))
        print(f"camera {cam.get_settings()}  - probe frame 90th percentile {bright:.0f}/255")
        if bright > 245 or bright < 60:
            print("WARNING: agar is " + ("clipped" if bright > 245 else "dark")
                  + " - adjust --exposure-us or the lamp", file=sys.stderr)

        step = (w * umpx * (1 - args.overlap), h * umpx * (1 - args.overlap))
        (out / "run.json").write_text(json.dumps({
            "grid": args.grid, "search_grid": args.search_grid, "overlap": args.overlap,
            "um_per_px": round(umpx, 4), "stage_to_image_matrix": M.tolist(),
            "step_um": [round(s, 1) for s in step], "z_um": z, "exposure_us": args.exposure_us,
            "dish_centre_um": dish, "limit_mm": args.limit_mm, "deadband_mm": args.deadband_mm,
            "mosaic_scale": args.mosaic_scale, "min_area_mm2": args.min_area_mm2,
            "max_area_mm2": args.max_area_mm2, "dark_ratio": args.dark_ratio,
        }, indent=2), encoding="utf-8")
        print(f"to halt this run at any point, create: {stop_file}")

        fields = ("round", "wall_clock", "t_s", "grid", "centre_x_um", "centre_y_um",
                  "found", "worm_x_um", "worm_y_um", "area_mm2", "length_mm",
                  "n_candidates", "recentred", "error")
        log = (out / "track.csv").open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(log, fieldnames=fields)
        writer.writeheader()

        last = None                               # last worm position, stage um
        last_area = None
        misses = 0
        t0 = time.monotonic()
        r = 0
        while (args.rounds is None or r < args.rounds) and \
                (args.minutes is None or time.monotonic() - t0 < args.minutes * 60):
            if args.interval_s:
                target = t0 + r * args.interval_s
                if time.monotonic() < target:
                    time.sleep(target - time.monotonic())
            gspec = args.search_grid if misses else args.grid
            nx, ny = (int(v) for v in gspec.lower().split("x"))
            row = {k: "" for k in fields}
            row.update(round=r, wall_clock=datetime.datetime.now().isoformat(timespec="seconds"),
                       t_s=round(time.monotonic() - t0, 1), grid=gspec,
                       centre_x_um=round(cx, 1), centre_y_um=round(cy, 1), found=0)
            try:
                tiles = []
                for tx, ty in grid_positions(cx, cy, nx, ny, *step):
                    if stop_file.exists():
                        print("STOP file present - halting", flush=True)
                        raise KeyboardInterrupt
                    ax, ay = _move_safe(sdk, tx, ty)
                    time.sleep(SETTLE_S)
                    p = cam.capture()
                    tiles.append((np.asarray(Image.open(p)), ax, ay))
                    p.unlink(missing_ok=True)
                canvas, omin, twh = stitch_with_origin(tiles, M, (cx, cy))
                cands, _ = find_worms(canvas, umpx, args.min_area_mm2, args.max_area_mm2,
                                      args.dark_ratio)
                row["n_candidates"] = len(cands)
                # Where the previous worm position (or, first round, the
                # block centre) sits on this canvas. Gate generously: a
                # worm reversing and sprinting can do ~0.3 mm/s.
                anchor = last or (cx, cy)
                ap = -M @ (np.array(anchor) - np.array((cx, cy))) - omin + np.array(twh) / 2
                gate_px = (args.gate_mm * 1000 if last else args.first_gate_mm * 1000) / umpx
                if misses:
                    gate_px *= 2
                best = pick(cands, (ap[0], ap[1]), gate_px, last_area)

                if best:
                    wx, wy = canvas_to_stage(best["x"], best["y"], M, (cx, cy), omin, twh)
                    last, misses, last_area = (wx, wy), 0, best["area_mm2"]
                    row.update(found=1, worm_x_um=round(wx, 1), worm_y_um=round(wy, 1),
                               area_mm2=round(best["area_mm2"], 4),
                               length_mm=round(best["length_mm"], 2))
                    half = int(args.crop_mm * 1000 / umpx / 2)
                    bx, by = int(best["x"]), int(best["y"])
                    crop = canvas[max(0, by - half):by + half, max(0, bx - half):bx + half]
                    Image.fromarray(crop).save(out / "crops" / f"crop_{r:04d}.jpg", quality=92)
                else:
                    misses += 1

                small = cv2.resize(canvas, None, fx=args.mosaic_scale, fy=args.mosaic_scale,
                                   interpolation=cv2.INTER_AREA)
                if best:
                    cv2.circle(small, (int(best["x"] * args.mosaic_scale),
                                       int(best["y"] * args.mosaic_scale)),
                               int(0.7 * 1000 / umpx * args.mosaic_scale), (255, 0, 0), 3)
                Image.fromarray(small).save(out / f"round_{r:04d}.jpg", quality=90)

                # Re-centre, only past the deadband, clamped to the dish.
                if last and np.hypot(last[0] - cx, last[1] - cy) > args.deadband_mm * 1000:
                    nxp, nyp = last
                    dx, dy = nxp - dish[0], nyp - dish[1]
                    rad = float(np.hypot(dx, dy)); lim = args.limit_mm * 1000
                    if rad > lim:
                        nxp, nyp = dish[0] + dx * lim / rad, dish[1] + dy * lim / rad
                        print(f"  re-centre clamped to the {args.limit_mm} mm limit", flush=True)
                    cx, cy = nxp, nyp
                    row["recentred"] = 1
                state = (f"worm at ({row['worm_x_um']}, {row['worm_y_um']}) um, "
                         f"{row['area_mm2']} mm2" if best else f"NOT FOUND (miss {misses}, "
                         f"{len(cands)} candidates) - next round searches {args.search_grid}")
                print(f"[{r:04d}] {datetime.datetime.now():%H:%M:%S} {gspec} {state}"
                      + ("  -> re-centred" if row["recentred"] else ""), flush=True)
            except KeyboardInterrupt:
                writer.writerow(row); log.flush()
                break
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{r:04d}] ERROR {row['error']}", flush=True)
                if "e-stop" in str(exc).lower() or "estop" in str(exc).lower():
                    writer.writerow(row); log.flush()
                    break
            writer.writerow(row); log.flush()
            r += 1
        log.close()
        plot_track(out)
    finally:
        try:
            _move_safe(sdk, cx, cy)
        except Exception as exc:
            print(f"WARNING: could not park at the last centre: {exc}", file=sys.stderr)
        cam.close()
        print(f"released - output in {out}")
    return out


def plot_track(out: Path) -> None:
    """track.png: the path drawn the way the mosaics show it (stage +X
    moves features left, +Y moves them down, so X runs right-to-left and
    Y bottom-to-top), coloured blue -> yellow by time, with a 1 mm bar."""
    with (out / "track.csv").open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["found"] == "1"]
    if not rows:
        return
    x = np.array([float(r["worm_x_um"]) for r in rows]) / 1000
    y = np.array([float(r["worm_y_um"]) for r in rows]) / 1000
    t = np.array([float(r["t_s"]) for r in rows]) / 60
    size, pad = 900, 70
    span = max(np.ptp(x), np.ptp(y), 2.0)
    s = (size - 2 * pad) / span
    cxm, cym = (x.max() + x.min()) / 2, (y.max() + y.min()) / 2
    pts = np.column_stack([size / 2 - (x - cxm) * s, size / 2 - (y - cym) * s]).astype(np.int32)
    img = np.full((size, size, 3), 255, np.uint8)
    cols = cv2.applyColorMap(np.linspace(0, 255, len(pts)).astype(np.uint8)[:, None],
                             cv2.COLORMAP_VIRIDIS)[:, 0]
    for i in range(1, len(pts)):
        cv2.line(img, tuple(pts[i - 1]), tuple(pts[i]), [int(v) for v in cols[i]], 2, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[0]), 9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(img, tuple(pts[-1]), 7, [int(v) for v in cols[-1]], -1, cv2.LINE_AA)
    cv2.line(img, (pad, size - 30), (pad + int(s), size - 30), (0, 0, 0), 3)
    cv2.putText(img, "1 mm", (pad, size - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    d = np.hypot(np.diff(x), np.diff(y)).sum()
    cv2.putText(img, f"{len(rows)} points, {t[-1]:.1f} min, {d:.1f} mm travelled "
                     f"(o = start, colour = time)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out / "track.png"), img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--grid", default="3x3", help="block while locked on (default 3x3)")
    ap.add_argument("--search-grid", default="5x5", help="block after a miss (default 5x5)")
    ap.add_argument("--overlap", type=float, default=0.10)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--minutes", type=float)
    ap.add_argument("--interval-s", type=float, default=0.0,
                    help="seconds between round starts; 0 = back to back (default)")
    ap.add_argument("--exposure-us", type=float, default=1600.0,
                    help="4x at lamp 1072: 1.6 ms puts agar at ~200/255 (2026-09-29)")
    ap.add_argument("--deadband-mm", type=float, default=0.5)
    ap.add_argument("--gate-mm", type=float, default=3.0,
                    help="max distance from the last position to accept a detection")
    ap.add_argument("--first-gate-mm", type=float, default=2.0,
                    help="first round: max distance from the block centre")
    ap.add_argument("--limit-mm", type=float, default=10.0,
                    help="never centre farther than this from the dish centre")
    ap.add_argument("--dish-centre", help="'x,y' um; default is the start position")
    ap.add_argument("--min-area-mm2", type=float, default=0.012)
    ap.add_argument("--max-area-mm2", type=float, default=0.25)
    ap.add_argument("--dark-ratio", type=float, default=0.85,
                    help="a pixel is worm if darker than this fraction of local agar")
    ap.add_argument("--crop-mm", type=float, default=1.8)
    ap.add_argument("--mosaic-scale", type=float, default=0.5)
    ap.add_argument("--calibrate", action="store_true", help="re-measure the stage->image mapping")
    ap.add_argument("--tag")
    ap.add_argument("--detect-only", type=Path, help="run detection on an image and exit")
    ap.add_argument("--um-per-px", type=float, default=1.4877, help="for --detect-only")
    ap.add_argument("--setup", action="store_true",
                    help="switch objective, set exposure, focus and centre the worm, then exit")
    ap.add_argument("--nosepiece", type=int, default=1, help="for --setup (1 = 4x)")
    args = ap.parse_args()
    if args.setup:
        setup(args)
        return
    if args.detect_only:
        detect_only(args.detect_only, args.um_per_px, args)
        return
    if args.rounds is None and args.minutes is None:
        args.rounds = 1
    run(args)


if __name__ == "__main__":
    main()
