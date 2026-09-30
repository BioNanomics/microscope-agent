# aml18_survey.py
# ------------------------------------------------------------
# Population survey of AML18 worms (pan-neuronal nuclear GFP + tagRFP)
# from a single NIS-Elements .nd2 frame: find every worm in the
# transmitted-light (TD) channel, measure its length along the body,
# estimate its life stage, and use the neuronal fluorescence to tell the
# head (nerve ring = brightest neuron cluster) from the tail.
#
#   python -m analysis.aml18_survey D:\...\CelegansAML-18\Test1.nd2
#   python -m analysis.aml18_survey file.nd2 --neuron-channel GFP
#
# Writes <file>_survey/ (or --out): worms.csv, survey.png (annotated
# overlay); for a time-lapse, survey_tNNN.png per time point and counts.csv.
#
# WHICH FLUORESCENCE CHANNEL. The neuron channel must be captured at the
# same instant as TD, or it shows where a crawling worm used to be. On
# the AX, TD is read out with the 561 nm laser, so RFP lines up with TD
# and a sequentially-scanned GFP does not (2026-09-29, Test1.nd2: 11.7 s
# frames, GFP offset from every worm). Default is RFP for that reason;
# use --neuron-channel GFP only for simultaneous acquisitions.
#
# WORMS VS EGGS. Both are dark in TD. Eggs are compact (~50 x 30 um);
# worms are long and thin. Objects are kept only if their body length
# (longest path through the skeleton) is > --min-length-um and at least
# 4x their width.
#
# LIFE STAGE is a rough guide from body length alone (N2 at 20 C:
# L1 ~0.25, L2 ~0.36, L3 ~0.49, L4 ~0.62-0.9, adult ~1.0-1.3 mm). A
# curled or partly hidden worm reads short, so treat it as indicative.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import nd2
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.morphology import remove_small_objects, skeletonize

STAGES = [(0.30, "L1"), (0.42, "L2"), (0.56, "L3"), (0.90, "L4"), (99.0, "adult")]


def load(path: Path) -> tuple[list[tuple[float, dict[str, np.ndarray]]], float]:
    """Every time point as (seconds, {channel: 2-D image}).

    A Z-stack is collapsed per time point: fluorescence by maximum
    projection (a neuron is counted wherever it is in focus), TD by
    taking the single sharpest plane (a projection of brightfield smears
    every plane's blur together). XY multipoint files are not handled.
    """
    with nd2.ND2File(path) as f:
        extra = set(f.sizes) - {"T", "Z", "C", "Y", "X"}
        if extra:
            raise SystemExit(f"unsupported dimensions {extra} in {f.sizes}")
        arr = f.asarray().astype(np.float32)
        dims = list(f.sizes)
        names = [c.channel.name for c in f.metadata.channels]
        um_px = float(f.voxel_size().x)
        try:
            times = [e.get("Time [s]", i) for i, e in enumerate(f.events())][: f.sizes.get("T", 1)]
        except Exception:
            times = []
    # Put the array in a fixed T, Z, C, Y, X order, adding missing axes.
    for d in ("T", "Z", "C"):
        if d not in dims:
            arr = arr[np.newaxis]
            dims.insert(0, d)
    arr = np.transpose(arr, [dims.index(d) for d in ("T", "Z", "C", "Y", "X")])
    frames = []
    for t in range(arr.shape[0]):
        ch = {}
        for i, name in enumerate(names):
            stack = arr[t, :, i]                                # Z, Y, X
            if name == "TD":
                sharp = [float(cv2.Laplacian(z, cv2.CV_32F).var()) for z in stack]
                ch[name] = np.ascontiguousarray(stack[int(np.argmax(sharp))])
            else:
                ch[name] = np.ascontiguousarray(stack.max(axis=0))
        frames.append((float(times[t]) if t < len(times) else float(t), ch))
    return frames, um_px


def segment_worms(td: np.ndarray, um_px: float, dark_ratio: float) -> np.ndarray:
    """Dark, thin objects relative to local background (closing removes
    anything narrower than the kernel, leaving the agar)."""
    k = int(round(160 / um_px)) | 1                          # 160 um: 3x an adult's width
    bg = cv2.morphologyEx(td, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3)
    mask = td < dark_ratio * bg
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    return remove_small_objects(mask, max_size=int(2000 / um_px ** 2))  # <= 2000 um2 is debris


def longest_path(skel: np.ndarray) -> tuple[np.ndarray, float]:
    """Pixels of the longest geodesic path through a skeleton, and its
    length in pixels (diagonal steps count sqrt 2)."""
    ys, xs = np.nonzero(skel)
    idx = -np.ones(skel.shape, int)
    idx[ys, xs] = np.arange(len(ys))
    rows, cols, w = [], [], []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y2, x2 = ys + dy, xs + dx
        ok = (y2 >= 0) & (y2 < skel.shape[0]) & (x2 >= 0) & (x2 < skel.shape[1])
        ok[ok] = skel[y2[ok], x2[ok]]
        rows += list(idx[ys[ok], xs[ok]]); cols += list(idx[y2[ok], x2[ok]])
        w += [np.hypot(dy, dx)] * int(ok.sum())
    n = len(ys)
    g = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    d0 = dijkstra(g, directed=False, indices=0)
    a = int(np.nanargmax(np.where(np.isinf(d0), -1, d0)))
    da, pred = dijkstra(g, directed=False, indices=a, return_predecessors=True)
    b = int(np.nanargmax(np.where(np.isinf(da), -1, da)))
    path = [b]
    while path[-1] != a:
        path.append(pred[path[-1]])
    return np.column_stack([ys[path], xs[path]]), float(da[b])


def neuron_mask(ch: np.ndarray, um_px: float) -> tuple[np.ndarray, float]:
    """Fluorescent neurons: well above the (mostly zero) background."""
    bg = cv2.medianBlur(ch.astype(np.float32), 5) if ch.max() > 0 else ch
    noise = float(np.percentile(ch, 99))                    # >99% of the frame is empty agar
    thr = max(3.0 * noise, noise + 5.0)
    return (cv2.GaussianBlur(ch, (0, 0), 1.0) > thr), thr


def survey_frame(ch: dict[str, np.ndarray], um_px: float, neuron_channel: str,
                 min_length_um: float, dark_ratio: float, title: str):
    """One frame: worm rows and the annotated overlay (BGR)."""
    if "TD" not in ch or neuron_channel not in ch:
        raise SystemExit(f"need TD and {neuron_channel}; file has {list(ch)}")
    td, neu = ch["TD"], ch[neuron_channel]

    worms = segment_worms(td, um_px, dark_ratio)
    nmask, thr = neuron_mask(neu, um_px)
    lab, n = ndi.label(worms)
    dist = ndi.distance_transform_edt(worms)
    near = ndi.distance_transform_edt(~worms) <= 25 / um_px   # neurons may sit just outside the TD edge

    rows = []
    shape = td.shape

    def add_worm(obj: np.ndarray, width_map: np.ndarray | None, source: str) -> bool:
        skel = skeletonize(obj)
        if skel.sum() < 3:
            return False
        path_px, length_px = longest_path(skel)
        length_um = length_px * um_px
        width_um = 2 * float(np.median(width_map[skel])) * um_px if width_map is not None else None
        if length_um < min_length_um or (width_um and length_um < 4 * width_um):
            return False                                      # egg or debris
        ys, xs = np.nonzero(obj)
        edge = bool(ys.min() == 0 or xs.min() == 0 or ys.max() == shape[0] - 1
                    or xs.max() == shape[1] - 1)
        # Neuron signal belonging to this worm: inside the body or within
        # 25 um of it.
        zone = ndi.binary_dilation(obj, iterations=int(25 / um_px))
        if source == "TD":
            zone &= near
        sig = np.where(zone & nmask, neu, 0)
        total = float(sig.sum())
        if total == 0:
            # Every AML18 worm has glowing neurons; a dark TD object with
            # none is agar texture or debris (Loop 16:13 worm 4).
            return False
        # Head = the end whose first 15% of body length holds more signal.
        seg = max(3, int(0.15 * len(path_px)))
        r = int(60 / um_px)

        def end_signal(pts):
            m = np.zeros(shape, np.uint8)
            for y, x in pts:
                cv2.circle(m, (int(x), int(y)), r, 1, -1)
            return float(sig[m.astype(bool)].sum())

        s_a, s_b = end_signal(path_px[:seg]), end_signal(path_px[-seg:])
        if max(s_a, s_b) < 1.5 * min(s_a, s_b) + 1:
            head = None                                       # no clear winner
        else:
            head = tuple(path_px[0] if s_a > s_b else path_px[-1])
            # An end at the image border is where the frame cut the worm,
            # not a real end - its neurons may be off-frame, so the
            # comparison is meaningless (Test1 worm 6 was called at the cut).
            hy, hx = head
            if min(hy, hx) <= 3 or hy >= shape[0] - 4 or hx >= shape[1] - 4:
                head = None
        stage = next(name for lim, name in STAGES if length_um / 1000 < lim)
        cy, cx = ndi.center_of_mass(obj)
        rows.append({
            "worm": len(rows) + 1, "found_by": source, "x_px": round(cx), "y_px": round(cy),
            "length_mm": round(length_um / 1000, 3),
            "width_um": round(width_um, 1) if width_um else "",
            "stage_estimate": stage + (" (cut by edge)" if edge else ""),
            "neuron_signal": round(total), "neuron_px": int((sig > 0).sum()),
            "head_x_px": head[1] if head else "", "head_y_px": head[0] if head else "",
            "head_confidence": round(max(s_a, s_b) / (min(s_a, s_b) + 1), 1),
            "_path": path_px, "_obj": obj,
        })
        return True

    for i in range(1, n + 1):
        add_worm(lab == i, dist, "TD")

    # Worms seen only by their neurons: small larvae are too faint and
    # thin in 4x brightfield (Timelapse1: ~6 glowing larvae unoutlined),
    # but their neuron chain is a clear line of signal. Join each chain
    # into one object and keep it if it is worm-shaped and not already
    # part of a brightfield worm. Their length follows the neurons, so it
    # can read a little short of the body.
    claimed = np.zeros(shape, bool)
    for w in rows:
        claimed |= w["_obj"]
    claimed = ndi.binary_dilation(claimed, iterations=int(40 / um_px))
    glow = ndi.binary_dilation(nmask, iterations=max(1, int(8 / um_px)))
    glow = cv2.morphologyEx(glow.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))).astype(bool)
    glab, gn = ndi.label(glow)
    for i in range(1, gn + 1):
        obj = glab == i
        if (obj & claimed).any() or (obj & nmask).sum() < 8:
            continue
        add_worm(obj, None, "neurons")

    # Annotated overlay: TD grey, neuron channel magenta, worm outlines cyan.
    g = np.clip((td - np.percentile(td, 0.5)) / (np.percentile(td, 99.5) - np.percentile(td, 0.5)), 0, 1)
    f = np.clip(neu / max(np.percentile(neu[nmask], 95) if nmask.any() else 1, 1), 0, 1)
    img = np.dstack([g * 0.7 + f, g * 0.7, g * 0.7 + f])
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1].copy()   # BGR
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    for w in rows:
        cnts, _ = cv2.findContours(w["_obj"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        colour = (255, 255, 0) if w["found_by"] == "TD" else (0, 165, 255)
        cv2.drawContours(img, [c * 2 for c in cnts], -1, colour, 2 if w["found_by"] != "TD" else 1)
        if w["head_x_px"] != "":
            cv2.circle(img, (2 * w["head_x_px"], 2 * w["head_y_px"]), 14, (0, 255, 255), 3)
        lbl = f"{w['worm']}: {w['stage_estimate'].split()[0]} {w['length_mm']:.2f}mm"
        p = (2 * w["x_px"] + 30, 2 * w["y_px"])
        cv2.putText(img, lbl, p, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, lbl, p, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    px = int(round(500 / um_px)) * 2
    h, wd = img.shape[:2]
    cv2.rectangle(img, (wd - 40 - px, h - 50), (wd - 40, h - 38), (255, 255, 255), -1)
    cv2.putText(img, "500 um", (wd - 40 - px, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, f"{title}: {len(rows)} worms | cyan = brightfield outline, orange = found by neurons, yellow ring = head "
                     f"({neuron_channel} neurons, magenta)", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return rows, img, thr


def analyse(path: Path, neuron_channel: str, min_length_um: float, dark_ratio: float,
            out: Path | None = None) -> Path:
    frames, um_px = load(path)
    out = out or path.with_name(path.stem + "_survey")
    out.mkdir(exist_ok=True)
    multi = len(frames) > 1
    all_rows, counts = [], []
    print(f"{path.name}: {um_px:.3f} um/px, {len(frames)} time point(s), neuron channel {neuron_channel}")
    for ti, (t_s, ch) in enumerate(frames):
        title = f"{path.name}" + (f" t={t_s / 60:.1f} min ({ti + 1}/{len(frames)})" if multi else "")
        rows, img, thr = survey_frame(ch, um_px, neuron_channel, min_length_um, dark_ratio, title)
        dest = out / (f"survey_t{ti:03d}.png" if multi else "survey.png")
        # cv2.imwrite returns False instead of raising - a file held open
        # elsewhere silently kept the previous run's image on 2026-09-29.
        if not cv2.imwrite(str(dest), img):
            raise SystemExit(f"could not write {dest} - is it open in another program?")
        for w in rows:
            w.update(t_index=ti, t_s=round(t_s, 1))
        all_rows += rows
        stages = [w["stage_estimate"].split()[0] for w in rows]
        counts.append({"t_index": ti, "t_s": round(t_s, 1), "worms": len(rows),
                       **{st: stages.count(st) for _, st in STAGES},
                       "heads_found": sum(1 for w in rows if w["head_x_px"] != "")})
        if not multi:
            print(f"threshold {thr:.0f}; {len(rows)} worms")
            for w in rows:
                head = "head found" if w["head_x_px"] != "" else "head unclear"
                print(f"  #{w['worm']:>2} {w['stage_estimate']:<22} {w['length_mm']:.2f} mm, "
                      + (f"{w['width_um']:.0f} um wide, " if w["width_um"] != "" else "found by neurons, ")
                      + f"neuron px {w['neuron_px']:>4}, {head} "
                      f"(ratio {w['head_confidence']})")
        else:
            c = counts[-1]
            print(f"  t={t_s / 60:5.1f} min: {len(rows)} worms "
                  + " ".join(f"{st}:{c[st]}" for _, st in STAGES) + f"  heads {c['heads_found']}")

    keys = ["t_index", "t_s"] + [k for k in (all_rows[0] if all_rows else {"worm": 0})
                                 if not k.startswith("_") and k not in ("t_index", "t_s")]
    with (out / "worms.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader(); wr.writerows(all_rows)
    if multi:
        with (out / "counts.csv").open("w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(counts[0]))
            wr.writeheader(); wr.writerows(counts)
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nd2", type=Path)
    ap.add_argument("--neuron-channel", default="RFP")
    ap.add_argument("--min-length-um", type=float, default=150.0)
    ap.add_argument("--dark-ratio", type=float, default=0.85)
    ap.add_argument("--out", type=Path, help="output folder (default: <file>_survey)")
    a = ap.parse_args()
    analyse(a.nd2, a.neuron_channel, a.min_length_um, a.dark_ratio, a.out)


if __name__ == "__main__":
    main()
