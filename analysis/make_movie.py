# make_movie.py
# ------------------------------------------------------------
# Turn a timelapse series into a watchable MP4.
#
#   python -m analysis.make_movie data/timelapse_20260921_163741_oats
#   python -m analysis.make_movie <dir> --fps 24 --out front.mp4
#   python -m analysis.make_movie <dir> --scale 0.5 --no-overlay
#
# A folder of 500 PNGs is data; a 20-second movie is something a person
# can actually look at and see an organism move. This is the step that
# turns one into the other.
#
# SCALE BAR: the pixel size is not guessable from the image - it depends
# on the objective and the C-mount adapter. UM_PER_PX below was MEASURED
# on 2026-09-21 by moving the stage a known 150.00 um and phase-
# correlating the before/after frames (shift 100.66 px, confidence 0.95,
# dy 0.18 px so the camera is square to the stage axes). It is valid ONLY
# for nosepiece position 1 with that adapter - change the objective and
# it is wrong. Pass --um-per-px to override, or --no-overlay to omit the
# bar rather than draw a scale you do not trust.
#
# TIMESTAMPS come from frames.csv's t_seconds, not from frame index x
# interval: a run that dropped or lagged a frame would otherwise show a
# time that drifts from reality, and the whole point of the overlay is to
# let a viewer judge rate.
#
# WHY mp4v: cv2's ffmpeg build on Windows reliably has it. H.264 gives
# smaller files but is not always present in the wheel, and a movie that
# fails to write is worse than one that is a few MB larger.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

#: Measured, not assumed - see module header. Objective-specific.
UM_PER_PX = 1.4901


def _rows(series_dir: Path) -> list[dict]:
    with (series_dir / "frames.csv").open(encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("filename") and not r.get("error")]
    return [r for r in rows if (series_dir / r["filename"]).exists()]


def wb_gains(img: np.ndarray) -> np.ndarray:
    """Per-channel gains that make the brightest UNCLIPPED pixels neutral.

    In transmitted brightfield the bright background is light that missed
    the specimen, so it is a legitimate white reference - but only where
    it is not clipped. Clipped pixels are (255,255,255) by definition, so
    including them forces the gains to 1.0 and silently reports "already
    balanced" no matter how strong the cast really is. That mistake was
    made once on this data; hence the explicit mask.

    Measured on this rig 2026-09-21: the camera's native response with
    BalanceWhiteAuto off is markedly blue (background R=145 G=168 B=237),
    needing roughly 1.26/1.09/0.77.
    """
    chans = [img[:, :, i].astype(np.float32) for i in range(3)]
    g = img.mean(axis=2)
    unclipped = (img < 250).all(axis=2)
    if unclipped.sum() < 1000:
        return np.ones(3, dtype=np.float32)
    ref = unclipped & (g > np.percentile(g[unclipped], 92))
    if ref.sum() < 200:
        return np.ones(3, dtype=np.float32)
    means = np.array([c[ref].mean() for c in chans], dtype=np.float32)
    if (means <= 1).any():
        return np.ones(3, dtype=np.float32)
    return means.mean() / means


def _scale_bar(img: np.ndarray, um_per_px: float) -> None:
    """Draw a scale bar whose length is a round number of microns.

    Picks the largest 'nice' length that stays under a quarter of the
    frame width, so the bar is meaningful at any magnification instead of
    a fixed pixel count that means something different on every objective.
    """
    h, w = img.shape[:2]
    max_um = (w * 0.25) * um_per_px
    nice = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    target = max([n for n in nice if n <= max_um], default=nice[0])
    px = int(round(target / um_per_px))

    x1, y2 = w - 40, h - 40
    x0 = x1 - px
    cv2.rectangle(img, (x0, y2 - 9), (x1, y2), (255, 255, 255), -1)
    cv2.rectangle(img, (x0, y2 - 9), (x1, y2), (0, 0, 0), 1)
    label = f"{target} um" if target < 1000 else f"{target / 1000:g} mm"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, label, (x0 + (px - tw) // 2, y2 - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(img, label, (x0 + (px - tw) // 2, y2 - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def _stamp(img: np.ndarray, t_s: float, i: int, n: int) -> None:
    hms = f"{int(t_s) // 3600:d}:{(int(t_s) % 3600) // 60:02d}:{int(t_s) % 60:02d}"
    txt = f"t + {hms}   ({i + 1}/{n})"
    cv2.putText(img, txt, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(img, txt, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def build(series_dir: Path, out: Path, fps: int, scale: float,
          um_per_px: float, overlay: bool, white_balance: bool = True) -> Path:
    rows = _rows(series_dir)
    if not rows:
        raise SystemExit(f"no usable frames in {series_dir}")

    first = cv2.imread(str(series_dir / rows[0]["filename"]))
    if first is None:
        raise SystemExit(f"could not read {rows[0]['filename']}")

    # ONE set of gains for the whole movie, from the first frame. Computing
    # them per frame would re-neutralise every frame independently, so the
    # colour would shift as the organism covers more or less of the field -
    # visible as flicker, and fatal to any comparison across time.
    gains = wb_gains(first) if white_balance else np.ones(3, dtype=np.float32)
    if white_balance:
        print(f"white balance gains (B,G,R) = "
              f"{gains[0]:.3f}, {gains[1]:.3f}, {gains[2]:.3f}")
    h, w = first.shape[:2]
    if scale != 1.0:
        w, h = int(w * scale), int(h * scale)
    w -= w % 2; h -= h % 2          # some encoders reject odd dimensions

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit("could not open the video writer - is the mp4v codec available?")

    n = len(rows)
    try:
        for i, r in enumerate(rows):
            img = cv2.imread(str(series_dir / r["filename"]))
            if img is None:
                continue                      # skip, do not abort the whole movie
            if white_balance:
                img = np.clip(img.astype(np.float32) * gains[None, None, :],
                              0, 255).astype(np.uint8)
            if (img.shape[1], img.shape[0]) != (w, h):
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            if overlay:
                _stamp(img, float(r["t_seconds"]), i, n)
                _scale_bar(img, um_per_px / scale)
            writer.write(img)
            if i % 50 == 0:
                print(f"  {i + 1}/{n}", flush=True)
    finally:
        writer.release()

    span = float(rows[-1]["t_seconds"]) - float(rows[0]["t_seconds"])
    print(f"\n{n} frames spanning {span / 3600:.2f} h -> {n / fps:.1f} s of video")
    print(f"time compression: {span / (n / fps):.0f}x")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("series_dir", type=Path)
    ap.add_argument("--out", type=Path, help="default: <series_dir>/movie.mp4")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.0, help="resize factor (0.5 = half size)")
    ap.add_argument("--um-per-px", type=float, default=UM_PER_PX)
    ap.add_argument("--no-overlay", action="store_true", help="omit timestamp and scale bar")
    ap.add_argument("--no-white-balance", action="store_true",
                    help="keep the camera's raw (blue-biased) colour")
    args = ap.parse_args()

    out = args.out or (args.series_dir / "movie.mp4")
    build(args.series_dir, out, args.fps, args.scale, args.um_per_px,
          not args.no_overlay, not args.no_white_balance)


if __name__ == "__main__":
    main()
