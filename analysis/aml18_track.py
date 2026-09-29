# aml18_track.py
# ------------------------------------------------------------
# Second step after aml18_survey on a time-lapse: link the worms found in
# each frame into tracks, measure how far and how fast each one moved,
# check whether the neuron signal fades over the run, and render the
# outlined frames as a movie.
#
#   python -m analysis.aml18_track D:\...\CelegansAML-18\Timelapse_1h.nd2
#
# Reads <file>_survey/worms.csv and survey_tNNN.png, writes into the same
# folder: tracks.csv (one row per worm per frame, with a track id),
# track_summary.csv (one row per track), tracks.png (all paths on the
# first frame), signal.csv (neuron-channel brightness per frame) and
# survey_movie.mp4.
#
# LINKING. Frames are a minute apart, so a roaming worm can move several
# mm between them while a feeding one barely moves. Each worm is joined
# to the nearest unclaimed worm of the previous frame within --max-step-mm
# and of similar length (0.5-2x), closest pairs first. Anything further
# starts a new track. So a fast worm crossing the field may be split into
# pieces, and a worm leaving the 3.9 mm field simply ends - tracks are a
# lower bound on how long each worm was followed, not identities.
#
# FADING. The 99.9th percentile of the neuron channel per frame tracks the
# brightest neurons in view. Worms entering and leaving change it too, so
# only a steady decline over the whole run points to photobleaching.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def link(rows: list[dict], um_px: float, max_step_mm: float) -> list[dict]:
    frames = sorted({int(r["t_index"]) for r in rows})
    by_t = {t: [r for r in rows if int(r["t_index"]) == t] for t in frames}
    next_id = 1
    prev: list[dict] = []
    for t in frames:
        cur = by_t[t]
        pairs = []
        for i, c in enumerate(cur):
            for j, p in enumerate(prev):
                d_mm = np.hypot(float(c["x_px"]) - float(p["x_px"]),
                                float(c["y_px"]) - float(p["y_px"])) * um_px / 1000
                ratio = float(c["length_mm"]) / max(float(p["length_mm"]), 1e-6)
                if d_mm <= max_step_mm and 0.5 <= ratio <= 2.0:
                    pairs.append((d_mm, i, j))
        used_c, used_p = set(), set()
        for d_mm, i, j in sorted(pairs):
            if i in used_c or j in used_p:
                continue
            cur[i]["track"] = prev[j]["track"]
            cur[i]["step_mm"] = round(d_mm, 3)
            used_c.add(i); used_p.add(j)
        for i, c in enumerate(cur):
            if i not in used_c:
                c["track"] = next_id
                c["step_mm"] = ""
                next_id += 1
        prev = cur
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for tid in sorted({r["track"] for r in rows}):
        tr = sorted((r for r in rows if r["track"] == tid), key=lambda r: int(r["t_index"]))
        t0, t1 = float(tr[0]["t_s"]), float(tr[-1]["t_s"])
        dist = sum(float(r["step_mm"]) for r in tr if r["step_mm"] != "")
        lengths = [float(r["length_mm"]) for r in tr]
        stages = [r["stage_estimate"].split()[0] for r in tr]
        out.append({
            "track": tid, "frames": len(tr), "start_min": round(t0 / 60, 1),
            "end_min": round(t1 / 60, 1), "distance_mm": round(dist, 2),
            "mean_speed_mm_per_min": round(dist / ((t1 - t0) / 60), 3) if t1 > t0 else "",
            "median_length_mm": round(float(np.median(lengths)), 3),
            "stage": max(set(stages), key=stages.count),
        })
    return out


def draw_tracks(first_png: Path, rows: list[dict], dest: Path) -> None:
    img = cv2.imread(str(first_png))
    if img is None:
        return
    rng = np.random.default_rng(1)
    for tid in sorted({r["track"] for r in rows}):
        tr = sorted((r for r in rows if r["track"] == tid), key=lambda r: int(r["t_index"]))
        pts = np.array([[2 * int(r["x_px"]), 2 * int(r["y_px"])] for r in tr], np.int32)
        col = tuple(int(v) for v in rng.integers(80, 256, 3))
        if len(pts) > 1:
            cv2.polylines(img, [pts], False, col, 3, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 7, col, -1)
        cv2.putText(img, str(tid), tuple(pts[-1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    cv2.imwrite(str(dest), img)


def movie(folder: Path, fps: int) -> Path | None:
    import imageio_ffmpeg
    frames = sorted(folder.glob("survey_t*.png"))
    if len(frames) < 2:
        return None
    first = cv2.imread(str(frames[0]))
    h, w = (first.shape[0] // 2) * 2, (first.shape[1] // 2) * 2
    dest = folder / "survey_movie.mp4"
    wr = imageio_ffmpeg.write_frames(
        str(dest), (w, h), fps=fps, codec="libx264", quality=None, macro_block_size=2,
        output_params=["-crf", "22", "-movflags", "+faststart"])  # write_frames already outputs yuv420p
    wr.send(None)
    for f in frames:
        img = cv2.imread(str(f))[:h, :w]
        wr.send(np.ascontiguousarray(img[..., ::-1]))
    wr.close()
    return dest


def signal_per_frame(nd2_path: Path, channel: str) -> list[dict]:
    from analysis.aml18_survey import load
    frames, _ = load(nd2_path)
    return [{"t_min": round(t / 60, 2),
             "p99_9": round(float(np.percentile(ch[channel], 99.9)), 1),
             "max": round(float(ch[channel].max()), 1)} for t, ch in frames]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nd2", type=Path)
    ap.add_argument("--survey", type=Path, help="survey folder (default <file>_survey)")
    ap.add_argument("--max-step-mm", type=float, default=1.0)
    ap.add_argument("--channel", default="RFP")
    ap.add_argument("--fps", type=int, default=6)
    a = ap.parse_args()
    folder = a.survey or a.nd2.with_name(a.nd2.stem + "_survey")

    import nd2
    with nd2.ND2File(a.nd2) as f:
        um_px = float(f.voxel_size().x)
    with (folder / "worms.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rows = link(rows, um_px, a.max_step_mm)
    with (folder / "tracks.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["track", "step_mm"] + [k for k in rows[0]
                                                                   if k not in ("track", "step_mm")])
        wr.writeheader(); wr.writerows(rows)
    summ = summarise(rows)
    with (folder / "track_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(summ[0]))
        wr.writeheader(); wr.writerows(summ)
    draw_tracks(folder / "survey_t000.png", rows, folder / "tracks.png")

    sig = signal_per_frame(a.nd2, a.channel)
    with (folder / "signal.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(sig[0]))
        wr.writeheader(); wr.writerows(sig)
    mv = movie(folder, a.fps)

    long_tracks = [s for s in summ if s["frames"] >= 3]
    print(f"{len(summ)} tracks, {len(long_tracks)} followed for 3+ frames")
    for s in sorted(summ, key=lambda s: -s["frames"])[:15]:
        print(f"  track {s['track']:>3}: {s['stage']:<6} {s['frames']:>3} frames "
              f"({s['start_min']}-{s['end_min']} min), {s['distance_mm']} mm, "
              f"{s['mean_speed_mm_per_min']} mm/min")
    first, last = sig[0]["p99_9"], sig[-1]["p99_9"]
    print(f"{a.channel} brightest-neuron level: {first} -> {last} "
          f"({(last / first - 1) * 100 if first else 0:+.0f}%) over {sig[-1]['t_min']} min")
    print(f"wrote tracks.csv, track_summary.csv, tracks.png, signal.csv"
          + (f", {mv.name}" if mv else "") + f" in {folder}")


if __name__ == "__main__":
    main()
