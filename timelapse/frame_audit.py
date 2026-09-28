# frame_audit.py
# ------------------------------------------------------------
# "Was that a filming error or biology?" - audit an existing frame
# sequence around a suspected event.
#
# For every consecutive pair of frames it reports:
#   dt_s         - seconds between the two frames, and its ratio to the
#                  median interval. A ratio well above 1 means the
#                  acquisition stalled: the "sudden" change in the video
#                  is really a missing stretch of time.
#   mean_delta   - relative change in global mean intensity. A large
#                  value with little local change = illumination/exposure
#                  changed, not the specimen.
#   shift_px     - global XY translation (phase correlation). Non-zero =
#                  the stage or the sample moved.
#   diff_score / area_delta - the same quantities change_detector.py
#                  uses live, so the values seen at a known event give the
#                  thresholds to configure the scheduler with.
#
# Timestamps come from, in order of preference:
#   --history frame_history.jsonl   (this repo's own capture log)
#   --timestamps file.csv           (one ISO-8601 or epoch-seconds per line,
#                                    same order as the frames; e.g. exported
#                                    from the ND2's metadata)
#   file mtime                      (fallback - only trustworthy if the files
#                                    were written as they were captured)
#
# Usage:
#   python -m timelapse.frame_audit FRAME_DIR [--glob '*.png'] [--around 25h --window 1h]
#   python -m timelapse.frame_audit --history logs/frame_history.jsonl
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from timelapse.change_detector import estimate_shift, foreground_fraction, load_gray, otsu_threshold

_DURATION_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(s|m|h|d)?$")
_DURATION_UNIT_S = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, None: 1.0}


def parse_duration(text: str) -> float:
    """'25h' -> 90000.0, '90m' -> 5400.0, '30' -> 30.0 (seconds)."""
    m = _DURATION_RE.match(text.strip())
    if not m:
        raise ValueError(f"bad duration {text!r} (expected e.g. '25h', '90m', '30s')")
    return float(m.group(1)) * _DURATION_UNIT_S[m.group(2)]


def parse_timestamp(text: str) -> float:
    """ISO-8601 or epoch seconds -> epoch seconds."""
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        return datetime.fromisoformat(text).timestamp()


@dataclass
class FrameRecord:
    index: int
    path: Path
    t_s: float  # epoch seconds


@dataclass
class PairAudit:
    index: int            # index of the later frame
    t_rel_s: float        # seconds since the first frame
    dt_s: float
    dt_ratio: float       # dt / median dt
    mean_delta: float     # (mean_b - mean_a) / mean_a
    shift_px: float
    diff_score: float
    area_delta: float
    flags: list[str]


def frames_from_dir(frame_dir: Path, glob: str) -> list[FrameRecord]:
    paths = sorted(frame_dir.glob(glob))
    if not paths:
        raise FileNotFoundError(f"no files matching {glob!r} in {frame_dir}")
    return [FrameRecord(i, p, p.stat().st_mtime) for i, p in enumerate(paths)]


def frames_from_history(history_path: Path) -> list[FrameRecord]:
    records = []
    with open(history_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(FrameRecord(len(records), Path(rec["image"]), parse_timestamp(rec["captured_at"])))
    if not records:
        raise ValueError(f"{history_path} has no frame records")
    return records


def apply_timestamps(frames: list[FrameRecord], timestamps_path: Path) -> list[FrameRecord]:
    with open(timestamps_path) as f:
        rows = [row[0] for row in csv.reader(f) if row and row[0].strip() and not row[0].startswith("#")]
    if len(rows) != len(frames):
        raise ValueError(f"{timestamps_path} has {len(rows)} timestamps but there are {len(frames)} frames")
    return [FrameRecord(fr.index, fr.path, parse_timestamp(ts)) for fr, ts in zip(frames, rows)]


def audit(
    frames: list[FrameRecord],
    gap_ratio: float = 1.5,
    mean_delta_threshold: float = 0.10,
    shift_threshold_px: float = 4.0,
    diff_threshold: float = 3.0,
    area_threshold: float = 0.05,
) -> list[PairAudit]:
    """Score every consecutive pair. `frames` must be in time order.

    Two passes: the first computes raw pairwise quantities, the second
    normalizes the pixel-difference by the *median* pairwise difference of
    the whole sequence (the quiet-stretch noise level) - so diff_score is
    "how many times noisier than a typical interval", the same units the
    live ChangeDetector reports. Frame-to-frame rather than
    against-a-rolling-baseline on purpose: an audit wants to localize an
    event to one interval, not smear it over the baseline window."""
    if len(frames) < 2:
        raise ValueError("need at least 2 frames to audit")
    times = np.array([fr.t_s for fr in frames])
    dts = np.diff(times)
    median_dt = float(np.median(dts))

    raw = []  # (dt, mean_delta, shift, raw_diff, area_delta, size_changed)
    prev = load_gray(frames[0].path)
    for i in range(1, len(frames)):
        cur = load_gray(frames[i].path)
        mean_a, mean_b = float(prev.mean()), float(cur.mean())
        mean_delta = (mean_b - mean_a) / mean_a if mean_a > 0 else 0.0
        size_changed = cur.shape != prev.shape
        if size_changed:
            shift, raw_diff, area_delta = float("nan"), float("nan"), float("nan")
        else:
            dx, dy = estimate_shift(prev, cur)
            shift = float(np.hypot(dx, dy))
            raw_diff = float(np.mean(np.abs(cur - prev)))
            thr = otsu_threshold(prev)
            area_delta = foreground_fraction(cur, thr) - foreground_fraction(prev, thr)
        raw.append((float(dts[i - 1]), mean_delta, shift, raw_diff, area_delta, size_changed))
        prev = cur

    diffs = np.array([r[3] for r in raw])
    noise = float(np.nanmedian(diffs)) if np.isfinite(diffs).any() else 1.0
    noise = max(noise, 1e-6)

    t0 = frames[0].t_s
    results: list[PairAudit] = []
    for i, (dt, mean_delta, shift, raw_diff, area_delta, size_changed) in enumerate(raw, start=1):
        dt_ratio = dt / median_dt if median_dt > 0 else 1.0
        diff_score = raw_diff / noise
        flags = []
        if dt_ratio > gap_ratio:
            flags.append(f"GAP x{dt_ratio:.1f}")
        if dt <= 0:
            flags.append("NON-MONOTONIC TIME")
        if size_changed:
            flags.append("SIZE CHANGED")
        if abs(mean_delta) > mean_delta_threshold:
            flags.append(f"INTENSITY {mean_delta:+.0%}")
        if shift > shift_threshold_px:
            flags.append(f"SHIFT {shift:.0f}px")
        non_bio = any(f.startswith(("SHIFT", "INTENSITY", "SIZE")) for f in flags)
        if not non_bio and (diff_score > diff_threshold or abs(area_delta) > area_threshold):
            flags.append("CHANGE")
        results.append(PairAudit(
            index=i, t_rel_s=frames[i].t_s - t0, dt_s=dt, dt_ratio=dt_ratio,
            mean_delta=mean_delta, shift_px=shift, diff_score=diff_score,
            area_delta=float(area_delta), flags=flags,
        ))
    return results


def _fmt_rel(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_table(results: list[PairAudit], out=None) -> None:
    out = out or sys.stdout
    print(f"{'idx':>5} {'t_rel':>9} {'dt_s':>8} {'dt_x':>5} {'mean%':>7} {'shift':>6} {'diff':>6} {'area%':>7}  flags", file=out)
    for r in results:
        print(
            f"{r.index:>5} {_fmt_rel(r.t_rel_s):>9} {r.dt_s:>8.1f} {r.dt_ratio:>5.2f} "
            f"{r.mean_delta * 100:>+7.1f} {r.shift_px:>6.1f} {r.diff_score:>6.2f} {r.area_delta * 100:>+7.2f}  "
            f"{', '.join(r.flags)}",
            file=out,
        )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__ or "Audit a frame sequence for acquisition gaps and non-biological changes.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("frame_dir", nargs="?", type=Path, help="directory of frames (sorted by filename)")
    src.add_argument("--history", type=Path, help="this repo's logs/frame_history.jsonl")
    ap.add_argument("--glob", default="*.png")
    ap.add_argument("--timestamps", type=Path, help="CSV of one timestamp per frame (ISO-8601 or epoch seconds)")
    ap.add_argument("--around", help="only report pairs near this offset from the first frame, e.g. 25h")
    ap.add_argument("--window", default="1h", help="half-width of --around window (default 1h)")
    ap.add_argument("--gap-ratio", type=float, default=1.5)
    ap.add_argument("--mean-delta", type=float, default=0.10)
    ap.add_argument("--shift-px", type=float, default=4.0)
    ap.add_argument("--json", type=Path, help="also write full results as JSON here")
    ap.add_argument("--flagged-only", action="store_true")
    args = ap.parse_args(argv)

    frames = frames_from_history(args.history) if args.history else frames_from_dir(args.frame_dir, args.glob)
    if args.timestamps:
        frames = apply_timestamps(frames, args.timestamps)
    results = audit(frames, args.gap_ratio, args.mean_delta, args.shift_px)

    dts = [r.dt_s for r in results]
    print(f"{len(frames)} frames, {len(results)} intervals, median interval {np.median(dts):.1f}s, "
          f"span {_fmt_rel(results[-1].t_rel_s)}")

    shown = results
    if args.around:
        center, half = parse_duration(args.around), parse_duration(args.window)
        shown = [r for r in results if abs(r.t_rel_s - center) <= half]
    if args.flagged_only:
        shown = [r for r in shown if r.flags]
    print_table(shown)

    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], default=str, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
