# oat_approach.py
# ------------------------------------------------------------
# Is the plasmodium moving toward the food? Measure it from a mosaic run.
#
#   python -m analysis.oat_approach data/mosaic_<stamp> --oat-box 0,3350,1200,7300
#
# For every stitched round (mosaic_NNN.png) this segments plasmodium and
# reports how much of it lies within each distance band of the oat, plus
# the area-weighted mean distance to the oat. A plasmodium heading for the
# food shows up as area moving into the near bands and the mean distance
# falling, which is a number rather than an impression from the movie.
#
# THE OAT IS LOCATED BY HAND, ONCE. It is a flake that does not move, and
# in brightfield it is the only thing that is truly black (grey ~0-2 on a
# 0-255 scale; plasmodium bottoms out around 10). But the dish rim is just
# as black, so --oat-box says where to look: pixels <= OAT_MAX_GREY inside
# that box and inside the dish, in round 0, are the oat for every round.
#
# THRESHOLDS WERE MEASURED on 2026-09-23 (16.7 ms, gamma 1.0, neutral
# gains; quarter-size mosaic): agar ~219, plasmodium fans and trunk
# 10-40, an out-of-focus smear 110-180, oat 0-1, bare glass 250-255.
# PLASMODIUM_MAX_GREY sits in the gap so the smear is not counted as
# organism. Re-check them if the exposure or lamp changes.
#
# ONLY THE AGAR BLOCK IS MEASURED - organism cannot be anywhere else, so
# glass, pen marks and the rim are excluded by construction. What is NOT
# excluded is the dark shadow along the block's top edge near the oat,
# which is as dark and as textured as real fans. It does not change
# between rounds, so it offsets the near bands but not their trend: read
# this output as change over time, not as absolute coverage.
#
# DISTANCES are straight-line over the agar, from the oat's edge. Most of
# this dish's oat lies beyond the glass window; only its visible strip is
# the reference, so distances run from where the oat enters the view.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from analysis.make_movie import _mosaic_rows

#: Work at this fraction of the stitched size - ~12 um/px, plenty for mm-scale fronts.
WORK_SCALE = 0.25
OAT_MAX_GREY = 4
PLASMODIUM_MAX_GREY = 90
#: Bare glass beside the agar block reads 250-255; agar ~220.
GLASS_MIN_GREY = 246
#: Band edges in mm from the oat's edge.
BANDS_MM = (0, 3, 6, 9, 12, 15, 20, 30)
#: Ordinal blue ramp, near = dark (BGR for cv2).
BAND_COLOURS = [(0x6b, 0x36, 0x0d), (0x95, 0x4f, 0x18), (0xbf, 0x6a, 0x25),
                (0xe5, 0x87, 0x39), (0xe7, 0x98, 0x55), (0xec, 0xa7, 0x6d),
                (0xef, 0xb6, 0x86)]


def _grey(path: Path) -> np.ndarray:
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(g, None, fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)


def _hull(mask: np.ndarray) -> np.ndarray:
    pts = np.column_stack(np.nonzero(mask)[::-1]).astype(np.int32)
    out = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out


def dish_mask(g: np.ndarray) -> np.ndarray:
    """The glass window: convex hull of all bare-glass pixels, pulled in so
    the dark gradient at the rim is not counted as plasmodium."""
    return cv2.erode(_hull(g >= GLASS_MIN_GREY), np.ones((41, 41), np.uint8))


def agar_mask(g: np.ndarray, dish: np.ndarray) -> np.ndarray:
    """The agar block. Plasmodium only grows on agar, so everything outside
    it - bare glass, the block's shadowed edge, pen marks on the dish - is
    excluded. Everything in the dish that is not bare glass is agar,
    organism or oat; the largest such region, as a convex outline, is the
    block (the network cuts the visible agar into pieces, hence the hull)."""
    m = ((g < GLASS_MIN_GREY) & (dish > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return cv2.erode(_hull(lab == big), np.ones((21, 21), np.uint8))


def oat_mask(g: np.ndarray, dish: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = (int(v * WORK_SCALE) for v in box)
    m = np.zeros_like(g, dtype=np.uint8)
    m[y0:y1, x0:x1] = (g[y0:y1, x0:x1] <= OAT_MAX_GREY) & (dish[y0:y1, x0:x1] > 0)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        raise SystemExit("no black object found inside --oat-box")
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return (lab == big).astype(np.uint8)


def run(series_dir: Path, box: tuple[int, int, int, int]) -> Path:
    meta = json.loads((series_dir / "mosaic.json").read_text(encoding="utf-8"))
    um_px = meta["um_per_px"] / meta.get("mosaic_scale", 0.5) / WORK_SCALE
    rows = _mosaic_rows(series_dir)
    if not rows:
        raise SystemExit(f"no stitched rounds in {series_dir}")

    g0 = _grey(series_dir / rows[0]["filename"])
    dish = dish_mask(g0)
    oat = oat_mask(g0, dish, box)
    agar = agar_mask(g0, dish)
    # distance (mm) of every pixel from the oat's edge
    dist_mm = cv2.distanceTransform((1 - oat).astype(np.uint8), cv2.DIST_L2, 5) * um_px / 1000
    valid = (agar > 0) & (oat == 0)
    px_mm2 = (um_px / 1000) ** 2
    print(f"oat area {oat.sum() * px_mm2:.2f} mm^2, agar block {agar.sum() * px_mm2:.1f} mm^2, "
          f"{um_px:.1f} um/px")

    edges = list(zip(BANDS_MM[:-1], BANDS_MM[1:]))
    fields = (["round", "t_min", "total_mm2"] + [f"band_{a}_{b}mm_mm2" for a, b in edges]
              + ["mean_dist_mm", "nearest_mm"])
    out = series_dir / "oat_approach.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            g = g0 if i == 0 else _grey(series_dir / r["filename"])
            if g.shape != g0.shape:
                g = cv2.resize(g, g0.shape[::-1], interpolation=cv2.INTER_AREA)
            plas = (g <= PLASMODIUM_MAX_GREY) & (g > OAT_MAX_GREY) & valid
            d = dist_mm[plas]
            row = {"round": i, "t_min": round(r["t_seconds"] / 60, 1),
                   "total_mm2": round(plas.sum() * px_mm2, 2),
                   "mean_dist_mm": round(float(d.mean()), 3) if d.size else "",
                   "nearest_mm": round(float(d.min()), 3) if d.size else ""}
            for a, b in edges:
                row[f"band_{a}_{b}mm_mm2"] = round(((d >= a) & (d < b)).sum() * px_mm2, 2)
            w.writerow(row)
            print(f"round {i:3d}  t={row['t_min']:6.1f} min  total {row['total_mm2']:7.2f} mm^2  "
                  f"mean dist {row['mean_dist_mm']} mm")

    # One check image, so the oat, dish and bands can be verified by eye.
    check = cv2.cvtColor(g0, cv2.COLOR_GRAY2BGR)
    for (a, b), col in zip(edges, BAND_COLOURS):
        ring = (dist_mm >= a) & (dist_mm < b) & valid & (g0 <= PLASMODIUM_MAX_GREY) & (g0 > OAT_MAX_GREY)
        check[ring] = col
    check[oat > 0] = (0, 0, 220)
    for m, col in ((dish, (0, 180, 0)), (agar, (0, 200, 255))):
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(check, cnt, -1, col, 3)
    cv2.imwrite(str(series_dir / "oat_approach_check.png"), check)
    print(f"wrote {out} and oat_approach_check.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("series_dir", type=Path)
    ap.add_argument("--oat-box", required=True,
                    help="x0,y0,x1,y1 in round-0 stitched-mosaic pixels bounding the oat")
    args = ap.parse_args()
    box = tuple(int(v) for v in args.oat_box.split(","))
    run(args.series_dir, box)


if __name__ == "__main__":
    main()
