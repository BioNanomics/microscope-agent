# timelapse.py
# ------------------------------------------------------------
# Fixed-position brightfield time-lapse from the Baumer camera.
#
#   python -m acquisition.orchestration.timelapse --minutes 60 --interval 10
#   python -m acquisition.orchestration.timelapse --frames 12 --interval 5 --tag test
#
# WHY 10 s BY DEFAULT: written for Physarum polycephalum, whose
# cytoplasmic shuttle streaming reverses on a ~100-130 s contraction
# rhythm. A 10 s interval puts ~10-13 samples in each cycle - enough to
# fit the oscillation - while an hour of it still spans ~30 cycles, so a
# periodogram has something to work with. Slower front advance falls out
# of the same series for free.
#
# THE CAMERA IS OPENED ONCE for the whole run, not per frame. Connecting
# costs seconds and a GenICam camera can only be held by one process at a
# time, so a per-frame open/close would both blow the interval budget and
# leave a window where anything else could grab the device mid-run.
#
# ILLUMINATION IS LEFT ALONE. The DIA lamp stays at whatever it was set
# to, steady, for the run's whole duration - shuttering or dimming
# between frames would save the sample some light but makes every frame's
# exposure history different, which is precisely what ruins a
# quantitative intensity series. Set the lamp before starting.
#
# DRIFT: frames are scheduled against a fixed start time, not by sleeping
# `interval` after each capture. Capture takes real time (exposure +
# fetch + PNG encode), so sleep-after-capture accumulates a lag that
# grows all run - fatal for a periodogram. A frame whose slot has already
# passed is taken immediately and flagged `late` in the log rather than
# skipped.
#
# ROBUSTNESS: one failed frame does not end the run - it is recorded with
# its error in the CSV and the series continues on schedule. A run that
# dies at frame 200 of 360 still leaves 200 usable frames plus their
# metadata.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from acquisition.paths import data_root

#: Per-frame columns. `mean`/`p1`/`p50`/`p99`/`sat` are written at capture
#: time because they are what tells you mid-run whether the series is
#: still well exposed and in focus, without reopening 300 PNGs.
_FIELDS = ("frame", "t_seconds", "wall_clock", "filename", "exposure_us",
           "gain", "mean", "p1", "p50", "p99", "sat_frac", "focus", "late_s", "error")


def _series_dir(tag: str | None) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"timelapse_{stamp}" + (f"_{tag}" if tag else "")
    return data_root() / "data" / name


def _stats(path: Path) -> dict:
    """Exposure and focus numbers for one frame.

    `focus` is the variance of the Laplacian - the standard cheap
    focus proxy. Only ever compare it between frames of one series at one
    illumination; it is not an absolute scale.
    """
    a = np.asarray(Image.open(path))
    g = a.mean(axis=2)
    lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1])
    return {
        "mean": round(float(g.mean()), 3),
        "p1": round(float(np.percentile(g, 1)), 2),
        "p50": round(float(np.percentile(g, 50)), 2),
        "p99": round(float(np.percentile(g, 99)), 2),
        "sat_frac": round(float((a >= 254).mean()), 6),
        "focus": round(float(lap.var()), 2),
    }


def run(frames: int, interval: float, exposure_us: float, gain: float,
        tag: str | None = None) -> Path:
    from acquisition.backends.baumer_genicam import BaumerGenICam

    out = _series_dir(tag)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "frames.csv"

    cam = BaumerGenICam()
    try:
        settings = cam.set_settings(exposure_time_us=exposure_us, gain=gain)
        print(f"[timelapse] camera: {settings}", flush=True)
        print(f"[timelapse] {frames} frames every {interval}s "
              f"(~{frames * interval / 60:.1f} min) -> {out}", flush=True)

        # One throwaway grab so the first logged frame is taken under the
        # exposure just set, not the one before it.
        cam.capture().unlink(missing_ok=True)

        start = time.monotonic()
        wall_start = datetime.datetime.now()

        with log_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_FIELDS)
            writer.writeheader()

            for i in range(frames):
                target = start + i * interval
                late = time.monotonic() - target
                if late < 0:
                    time.sleep(-late)
                    late = 0.0

                row = {f: "" for f in _FIELDS}
                row.update(frame=i,
                           t_seconds=round(time.monotonic() - start, 3),
                           wall_clock=datetime.datetime.now().isoformat(timespec="seconds"),
                           exposure_us=exposure_us, gain=gain,
                           late_s=round(late, 3))
                try:
                    src = cam.capture()
                    dest = out / f"f{i:05d}.png"
                    shutil.move(str(src), dest)
                    row["filename"] = dest.name
                    row.update(_stats(dest))
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"

                writer.writerow(row)
                fh.flush()   # so the CSV is readable mid-run, not just at the end

                if row["error"]:
                    print(f"[timelapse] frame {i}: {row['error']}", flush=True)
                elif i % 10 == 0 or i == frames - 1:
                    print(f"[timelapse] {i + 1}/{frames} t={row['t_seconds']:.0f}s "
                          f"mean={row['mean']} focus={row['focus']} "
                          f"sat={row['sat_frac']}", flush=True)

        elapsed = (datetime.datetime.now() - wall_start).total_seconds()
        print(f"[timelapse] done in {elapsed / 60:.1f} min -> {out}", flush=True)
    finally:
        cam.close()

    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--minutes", type=float, help="run length; with --interval sets frame count")
    g.add_argument("--frames", type=int, help="explicit frame count")
    p.add_argument("--interval", type=float, default=10.0, help="seconds between frames (default 10)")
    p.add_argument("--exposure-us", type=float, default=40_000.0)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--tag", help="appended to the output directory name")
    args = p.parse_args()

    if args.frames:
        frames = args.frames
    elif args.minutes:
        frames = int(round(args.minutes * 60 / args.interval)) + 1
    else:
        p.error("pass --minutes or --frames")

    run(frames, args.interval, args.exposure_us, args.gain, args.tag)


if __name__ == "__main__":
    sys.exit(main())
