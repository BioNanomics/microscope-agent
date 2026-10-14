# worm_short.py
# ------------------------------------------------------------
# YouTube Short (1080x1920, 30 fps, H.264 + original music) from an AML18
# C. elegans time-lapse that aml18_survey has already been run on.
#
#   python -m analysis.worm_short D:\...\CelegansAML-18\Timelapse_1h.nd2
#   python -m analysis.worm_short file.nd2 --stars 2,43,5       # pick the worms
#   python -m analysis.worm_short file.nd2 --preview            # 7 stills, seconds
#
# Reads the .nd2 (TD + neuron channel) and <file>_survey/worms.csv; writes
# <file folder>/shorts/<stem>_short.mp4 (+ .wav).
#
# SEQUENCE: "lights off" (neurons only, on black) -> brightfield fades in
# -> the whole run in 17 s with trails for up to three "star" worms and a
# live distance-crawled chart -> the speed comparison -> every track at
# once -> end card. Music is synthesised here (no samples), with a soft
# tick for every frame of the time-lapse.
#
# PICTURE = TD brightfield screened with the neuron channel as a pink glow.
# Default RFP, because on the AX it is read out with TD and lines up with
# the worms; a sequentially scanned GFP lands offset (see aml18_survey.py).
#
# TRACKS are re-linked from worms.csv with aml18_track.link (default 0.5 mm
# max step, stricter than aml18_track's 1 mm, so crowded L1s are not chained
# into long false tracks). Each track's position then gets a 3-frame median,
# which removes one-frame centroid jumps (a curled or touching worm
# segmented oddly) before distances are summed.
#
# STARS: by default the three adult tracks followed for the most frames,
# named by speed - fastest "Explorer", slowest "Homebody", middle
# "Wanderer". --stars picks the track ids instead (same naming rule).
#
# THE SPEED CLAIM. Positions are sampled once per frame (1 min here), so
# "distance crawled" undercounts the real path and the Explorer/Homebody
# ratio depends on the clean-up: 6.5x from raw steps, 13.9x after the
# median, for Timelapse_1h. The Short states the smaller one as "at least
# Nx". The chart compares worms fairly; it is not an absolute speed.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from analysis.aml18_track import link, summarise
from analysis.short_video import (FONT_B, FONT_I, FONT_R, FPS, GREY, SR, VH, VW, ShortWriter, canvas,
                                  centred, font, write_wav)

BG = (10, 10, 14)
PINK = (255, 90, 210)
GLOW_RGB = np.array([1.0, 0.25, 0.85])
ROLES = {"Explorer": (255, 205, 60), "Wanderer": (90, 220, 255), "Homebody": (120, 255, 140)}
PANEL_Y = 380
P = VW                                     # square panel

S_DARK, S_FADE, S_TL, S_HOLD, S_MAP, S_END = 3.5, 3.0, 17.0, 3.5, 5.0, 3.5
T_FADE = S_DARK
T_TL = T_FADE + S_FADE
T_HOLD = T_TL + S_TL
T_MAP = T_HOLD + S_HOLD
T_END = T_MAP + S_MAP
DURATION = T_END + S_END


# ---- picture -------------------------------------------------------------------
def load(nd2_path: Path, channel: str):
    import nd2
    with nd2.ND2File(nd2_path) as f:
        um_px = float(f.voxel_size().x)
        names = [c.channel.name for c in f.metadata.channels]
        stack = f.asarray()                               # T, C, Y, X
        t_s = np.array([e["Time [s]"] for e in f.events()])
    return stack[:, names.index("TD")], stack[:, names.index(channel)], um_px, t_s


def make_composites(td, neuron):
    def glow(t):
        r = np.clip((neuron[t].astype(np.float32) - 6) / 250, 0, 1) ** 0.6
        return np.clip(cv2.GaussianBlur(r, (0, 0), 1.2) + 0.6 * cv2.GaussianBlur(r, (0, 0), 6), 0, 1)

    def bright(t):
        x = td[t].astype(np.float32)
        lo, hi = np.percentile(x, [0.5, 99.8])
        g = np.clip((x - lo) / (hi - lo), 0, 1) ** 1.3 * 0.85
        return np.repeat(g[..., None], 3, -1)

    def comp(t, light=1.0):
        rgb = 1 - (1 - bright(t) * light) * (1 - np.clip(glow(t)[..., None] * GLOW_RGB * (1.6 - 0.6 * light), 0, 1))
        return cv2.resize((np.clip(rgb, 0, 1) * 255).astype(np.uint8), (P, P), interpolation=cv2.INTER_AREA)

    return [comp(t) for t in range(len(td))], comp(0, light=0.0)


# ---- tracks ----------------------------------------------------------------------
def build_tracks(worms_csv: Path, um_px: float, k: float, dt_min: float, max_step_mm: float):
    """{track: [(minute, x, y, step_mm)]} smoothed, plus raw mm/min per track and the summary."""
    rows = link(list(csv.DictReader(worms_csv.open(encoding="utf-8"))), um_px, max_step_mm)
    summary = {s["track"]: s for s in summarise(rows)}
    raw: dict = {}
    for r in rows:
        raw.setdefault(r["track"], []).append((int(r["t_index"]) * dt_min, float(r["x_px"]) * k,
                                               float(r["y_px"]) * k,
                                               float(r["step_mm"]) if r["step_mm"] != "" else 0.0))
    tracks, raw_rate = {}, {}
    for tid, v in raw.items():
        v.sort()
        if len(v) > 1:
            raw_rate[tid] = sum(p[3] for p in v) / (v[-1][0] - v[0][0])
        xs = np.array([p[1] for p in v]); ys = np.array([p[2] for p in v])
        if len(v) >= 3:
            xs = np.array([np.median(xs[max(0, i - 1):i + 2]) for i in range(len(v))])
            ys = np.array([np.median(ys[max(0, i - 1):i + 2]) for i in range(len(v))])
        st = np.r_[0, np.hypot(np.diff(xs), np.diff(ys)) / k * um_px / 1000]
        tracks[tid] = [(p[0], x, y, s) for p, x, y, s in zip(v, xs, ys, st)]
    return tracks, raw_rate, summary


def pick_stars(tracks, summary, ids):
    if not ids:
        adults = [t for t, s in summary.items() if s["stage"] == "adult" and s["frames"] >= 10]
        ids = sorted(adults, key=lambda t: -summary[t]["frames"])[:3]
    if len(ids) < 2:
        raise SystemExit("need at least two tracks to compare (pass --stars)")
    rate = {t: sum(p[3] for p in tracks[t]) / (tracks[t][-1][0] - tracks[t][0][0]) for t in ids}
    order = sorted(ids, key=lambda t: -rate[t])
    names = ["Explorer", "Wanderer", "Homebody"] if len(order) == 3 else ["Explorer", "Homebody"]
    return {t: (n, ROLES[n]) for t, n in zip(order, names)}, rate


def pos_at(tr, tm):
    ts = [p[0] for p in tr]
    if tm < ts[0] or tm > ts[-1]:
        return None
    i = int(np.searchsorted(ts, tm, side="right")) - 1
    if i >= len(tr) - 1:
        return tr[-1][1], tr[-1][2]
    a, b = tr[i], tr[i + 1]
    u = (tm - a[0]) / (b[0] - a[0])
    return a[1] + u * (b[1] - a[1]), a[2] + u * (b[2] - a[2])


# ---- soundtrack --------------------------------------------------------------------
def soundtrack(n_ticks: int) -> np.ndarray:
    n = int(SR * DURATION); t = np.arange(n) / SR
    rng = np.random.default_rng(3)
    mix = np.zeros((n, 2))

    def note(m):
        return 440.0 * 2 ** ((m - 69) / 12)

    def add(sig, at, pan=0.5):
        i0 = int(at * SR)
        if i0 >= n:
            return
        L = min(len(sig), n - i0)
        mix[i0:i0 + L, 0] += sig[:L] * (1 - pan)
        mix[i0:i0 + L, 1] += sig[:L] * pan

    def marimba(m, lvl, decay=0.45):
        tt = np.arange(int(SR * decay * 5)) / SR
        f = note(m)
        s = np.sin(2 * np.pi * f * tt) + 0.25 * np.sin(2 * np.pi * 4 * f * tt) * np.exp(-tt / 0.05)
        return s * np.exp(-tt / decay) * np.clip(tt / 0.003, 0, 1) * lvl

    def pad(ms, a, b, lvl):
        e = np.clip(np.minimum((t - a) / 1.5, (b - t) / 1.5), 0, 1)
        e = 0.5 - 0.5 * np.cos(np.pi * e)
        for m in ms:
            for det, pan in ((-0.1, 0.25), (0.1, 0.75)):
                s = np.sin(2 * np.pi * note(m) * 2 ** (det / 12) * t + rng.uniform(0, 6.3)) * e * lvl / len(ms)
                mix[:, 0] += s * (1 - pan); mix[:, 1] += s * pan

    # dark opening: low drone + glassy shimmer, then the "lights on" swell
    pad([38, 45, 50], 0, T_TL + 1, 0.45)
    sh = np.clip(t / 2.5, 0, 1) * np.clip((T_TL - t) / 1.5, 0, 1)
    for f_, pan in ((1174.7, 0.2), (1396.9, 0.8), (1760.0, 0.5)):
        s = np.sin(2 * np.pi * f_ * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t + pan * 4)) * sh * 0.025
        mix[:, 0] += s * (1 - pan); mix[:, 1] += s * pan
    pad([50, 57, 62, 66], T_FADE, T_TL + 0.8, 0.5)
    # time-lapse: D-major pentatonic marimba at 112 bpm over a I-vi-IV-V loop
    beat = 60 / 112
    melody = [74, 76, 78, 81, 78, 76, 74, 71, 74, 78, 81, 83, 81, 78, 76, 74]
    chords = [[50, 57, 62, 66], [47, 54, 59, 62], [43, 50, 55, 59], [45, 52, 57, 61]]
    t_, i = T_TL, 0
    while t_ < T_MAP:
        bar = int((t_ - T_TL) / (4 * beat)) % 4
        if i % 2 == 0 or rng.random() < 0.6:
            add(marimba(melody[i % 16] - (12 if i % 8 == 7 else 0), 0.16), t_, pan=0.3 + 0.4 * (i % 2))
        if i % 4 == 0:
            add(marimba(chords[bar][0] - 12, 0.30, decay=0.9), t_, pan=0.5)
        if i % 8 == 0:
            pad(chords[bar][1:], t_, t_ + 8 * beat * 0.5 + 0.3, 0.30)
        t_ += beat / 2; i += 1
    for k in range(n_ticks):                                # a soft tick per time-lapse frame
        tt = np.arange(int(SR * 0.04)) / SR
        add(rng.standard_normal(len(tt)) * np.exp(-tt / 0.006) * 0.05, T_TL + S_TL * k / (n_ticks - 1))
    # every-path map + end: rising arpeggio into a warm resolve
    for k, m in enumerate([62, 66, 69, 74, 78, 81, 86]):
        add(marimba(m, 0.13, decay=0.8), T_MAP + 0.25 * k, pan=0.2 + 0.1 * k)
    pad([50, 57, 62, 66, 69, 74], T_MAP + 1.5, DURATION + 2, 0.75)
    add(marimba(86, 0.12, decay=2.0), T_END + 0.2, pan=0.6)
    # reverb (FFT convolution with decaying noise), fades, -1 dBFS
    ir_len = int(SR * 1.6)
    ir = rng.standard_normal((ir_len, 2)) * np.exp(-np.arange(ir_len) / (SR * 0.4))[:, None]
    ir[0] = 0
    nfft = 1 << int(np.ceil(np.log2(n + ir_len)))
    wet = np.stack([np.fft.irfft(np.fft.rfft(mix[:, c], nfft) * np.fft.rfft(ir[:, c], nfft), nfft)[:n]
                    for c in range(2)], axis=1)
    wet *= np.abs(mix).max() / (np.abs(wet).max() + 1e-9)
    out = 0.75 * mix + 0.35 * wet
    out *= (np.clip(t / 0.2, 0, 1) * np.clip((DURATION - t) / 2.0, 0, 1))[:, None]
    return out * 10 ** (-1 / 20) / np.abs(out).max()


# ---- the Short -----------------------------------------------------------------------
def make(nd2_path: Path, survey: Path, out: Path, channel: str, max_step_mm: float,
         star_ids: list[int], preview: bool) -> None:
    td, neuron, um_px, t_s = load(nd2_path, channel)
    nt, k = len(td), P / td.shape[-1]
    dt_min = (t_s[-1] - t_s[0]) / (nt - 1) / 60
    run_min = dt_min * (nt - 1)
    run_txt = "1 hour" if round(run_min) == 60 else f"{run_min:.0f} min"
    every_txt = "1 photo every minute" if round(dt_min, 1) == 1 else f"1 photo every {dt_min:.1f} min"
    print(f"{nt} frames, {dt_min:.2f} min apart, {um_px:.2f} um/px; compositing", flush=True)
    frames, dark0 = make_composites(td, neuron)

    tracks, raw_rate, summary = build_tracks(survey / "worms.csv", um_px, k, dt_min, max_step_mm)
    stars, rate = pick_stars(tracks, summary, star_ids)
    by_name = {n: t for t, (n, _) in stars.items()}
    fast, slow = by_name["Explorer"], by_name["Homebody"]
    faster = min(rate[fast] / rate[slow], raw_rate[fast] / raw_rate[slow])
    for t, (n, _) in stars.items():
        print(f"{n}: track {t}, {tracks[t][0][0]:.0f}-{tracks[t][-1][0]:.0f} min, "
              f"{rate[t]:.3f} mm/min (raw {raw_rate[t]:.3f})")
    print(f"Explorer/Homebody: {raw_rate[fast] / raw_rate[slow]:.1f}x raw, "
          f"{rate[fast] / rate[slow]:.1f}x filtered -> 'at least {int(faster)}x'")
    others = [v for t, v in tracks.items() if t not in stars and len(v) >= 5]
    crawl = {t: (np.array([p[0] for p in tracks[t]]), np.cumsum([p[3] for p in tracks[t]])) for t in stars}
    y_max = max(1.0, np.ceil(max(c[-1] for _, c in crawl.values()) * 2) / 2)

    def trails(img, tm, labels=True, show_others=False):
        lay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        if show_others:
            for tr in others:
                pts = [(x, y) for m, x, y, _ in tr if m <= tm]
                if len(pts) > 1:
                    d.line(pts, fill=(255, 255, 255, 110), width=3, joint="curve")
        for tid, (_, col) in stars.items():
            pts = [(x, y) for m, x, y, _ in tracks[tid] if m <= tm]
            cur = pos_at(tracks[tid], tm)
            if cur is not None:
                pts.append(cur)
            if len(pts) > 1:
                d.line(pts, fill=col + (255,), width=7, joint="curve")
        blur = lay.filter(ImageFilter.GaussianBlur(6))
        img.paste(blur, (0, 0), blur)
        img.paste(lay, (0, 0), lay)
        d = ImageDraw.Draw(img)
        for tid, (name, col) in stars.items():
            tr = tracks[tid]
            cur = pos_at(tr, tm)
            if cur is None:
                if tm > tr[-1][0]:                         # track ended: hollow marker
                    x, y = tr[-1][1], tr[-1][2]
                    d.ellipse([x - 10, y - 10, x + 10, y + 10], outline=col, width=4)
                continue
            x, y = cur
            rad = 13 + 3 * np.sin(tm * 2.0)
            d.ellipse([x - rad, y - rad, x + rad, y + rad], fill=col, outline=(0, 0, 0), width=3)
            if labels:
                f = font(FONT_B, 36)
                w = d.textlength(name, font=f)
                lx = min(max(x - w / 2, 10), P - w - 10)
                ly = y - 72 if y > 90 else y + 26
                d.rounded_rectangle([lx - 12, ly - 4, lx + w + 12, ly + 46], radius=14, fill=(0, 0, 0))
                d.text((lx, ly), name, font=f, fill=col)

    def scale_bar(d):
        px = 500 / um_px * k
        x1, y = P - 36, PANEL_Y + P - 34
        d.rectangle([x1 - px, y - 9, x1, y], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        f = font(FONT_B, 30)
        d.text((x1 - px / 2 - d.textlength("0.5 mm", font=f) / 2, y - 46), "0.5 mm", font=f,
               fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))

    cx0, cx1 = 120, VW - 70
    cy0, cy1 = PANEL_Y + P + 175, VH - 100

    def chart(img, tm):
        d = ImageDraw.Draw(img)
        f_ax, f_lb = font(FONT_R, 30), font(FONT_B, 38)
        d.text((cx0 - 60, cy0 - 120), "Distance crawled (mm)", font=f_lb, fill=(235, 235, 235))
        for v in np.arange(0, y_max + 0.01, 0.5):
            y = cy1 - (cy1 - cy0) * v / y_max
            d.line([(cx0, y), (cx1, y)], fill=(46, 46, 54), width=2)
            t = f"{v:g}"
            d.text((cx0 - 18 - d.textlength(t, font=f_ax), y - 20), t, font=f_ax, fill=GREY)
        for m in np.linspace(0, run_min, 5):
            x = cx0 + (cx1 - cx0) * m / run_min
            t = f"{m:.0f} min"
            d.text((x - d.textlength(t, font=f_ax) / 2, cy1 + 8), t, font=f_ax, fill=GREY)
        for tid, (_, col) in stars.items():
            ms, cs = crawl[tid]
            keep = ms <= tm
            if not keep.any():
                continue
            pts = [(cx0 + (cx1 - cx0) * m / run_min, cy1 - (cy1 - cy0) * c / y_max)
                   for m, c in zip(ms[keep], cs[keep])]
            if len(pts) > 1:
                d.line(pts, fill=col, width=6, joint="curve")
            x, y = pts[-1]
            d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=col)
            d.text((x + 14, y - 22), f"{cs[keep][-1]:.1f}", font=font(FONT_B, 32), fill=col,
                   stroke_width=3, stroke_fill=BG)

    def panel_at(tm):
        u = tm / dt_min
        i = min(int(u), nt - 1); j = min(i + 1, nt - 1); u -= i
        return frames[i] if u <= 0 or i == j else cv2.addWeighted(frames[i], 1 - u, frames[j], u, 0)

    def with_panel(arr):
        c = canvas(BG)
        c.paste(arr if isinstance(arr, Image.Image) else Image.fromarray(arr), (0, PANEL_Y))
        return c, ImageDraw.Draw(c)

    def span(a, b):
        return range(int(round(a * FPS)), int(round(b * FPS)))

    wav = None
    if not preview:
        wav = out.with_suffix(".wav")
        write_wav(soundtrack(nt), wav)
    stills = {int(s * FPS) for s in (1.5, T_FADE + 2.0, T_TL + 6, T_TL + 14, T_HOLD + 2, T_MAP + 3, T_END + 2)}
    dim = (frames[0].astype(np.float32) * 0.45).astype(np.uint8)
    with ShortWriter(out, wav, preview, stills) as w:
        for i in span(0, T_FADE):                                   # 1. lights off
            if not w.wants(i):
                continue
            c, d = with_panel((dark0 * min(1, i / FPS)).astype(np.uint8))
            centred(d, 95, "Lights off.", font(FONT_B, 100))
            if i / FPS > 1.2:
                centred(d, 235, "Every glowing dot is a neuron.", font(FONT_R, 58), fill=PINK)
            w.emit(c, i)
        for i in span(T_FADE, T_TL):                                # 2. lights on
            if not w.wants(i):
                continue
            u = np.clip((i / FPS - T_FADE) / 1.5, 0, 1); u = 0.5 - 0.5 * np.cos(np.pi * u)
            c, d = with_panel(cv2.addWeighted(dark0, 1 - u, frames[0], u, 0))
            centred(d, 80, "C. elegans", font(FONT_I, 100))
            centred(d, 215, "1 mm worms with 302 neurons —", font(FONT_R, 50), fill=(225, 225, 225))
            centred(d, 280, "engineered so each one glows", font(FONT_R, 50), fill=PINK)
            scale_bar(d); chart(c, -1)
            w.emit(c, i)
        for i in span(T_TL, T_HOLD):                                # 3. the run
            if not w.wants(i):
                continue
            tm = min(run_min, (i / FPS - T_TL) / S_TL * run_min)
            pan = Image.fromarray(panel_at(tm)); trails(pan, tm)
            c, d = with_panel(pan)
            centred(d, 80, f"Same worms, {run_txt}", font(FONT_B, 84))
            centred(d, 205, f"minute {tm:4.1f}", font(FONT_R, 60), fill=ROLES["Explorer"])
            centred(d, 290, every_txt, font(FONT_R, 40), fill=GREY)
            scale_bar(d); chart(c, tm)
            w.emit(c, i)
            if not preview and i % 60 == 0:
                print("timelapse", i, flush=True)
        for i in span(T_HOLD, T_MAP):                               # 4. the comparison
            if not w.wants(i):
                continue
            pan = Image.fromarray(frames[-1]); trails(pan, run_min)
            c, d = with_panel(pan)
            centred(d, 70, "The Explorer moved", font(FONT_B, 76), fill=ROLES["Explorer"])
            centred(d, 180, f"at least {int(faster)}× faster than the Homebody", font(FONT_B, 50))
            if i / FPS - T_HOLD > 1.2:
                centred(d, 285, "Same species. Same plate. Different personalities?", font(FONT_I, 40),
                        fill=GREY)
            scale_bar(d); chart(c, run_min)
            w.emit(c, i)
        for i in span(T_MAP, DURATION):                             # 5. every path + 6. end card
            if not w.wants(i):
                continue
            u = np.clip((i / FPS - T_MAP) / 3.0, 0, 1)
            pan = Image.fromarray(dim); trails(pan, run_min * u, labels=False, show_others=True)
            c, d = with_panel(pan)
            if i / FPS < T_END:
                centred(d, 80, f"Every path in {run_txt}", font(FONT_B, 84))
                centred(d, 205, f"{len(others) + len(stars)} worms followed for 5+ frames", font(FONT_R, 50),
                        fill=(225, 225, 225))
            else:
                centred(d, 80, f"{run_txt} in {S_TL:.0f} seconds", font(FONT_B, 84))
                centred(d, 205, "Nikon AX confocal · brightfield + RFP neurons", font(FONT_R, 44),
                        fill=(225, 225, 225))
                centred(d, 275, f"AML18 C. elegans · {every_txt.replace('1 photo', '1 frame')}",
                        font(FONT_R, 44), fill=GREY)
            scale_bar(d); chart(c, run_min)
            w.emit(c, i)
    print("wrote", out if not preview else f"preview stills in {out.parent}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nd2", type=Path)
    ap.add_argument("--survey", type=Path, help="survey folder (default <file>_survey)")
    ap.add_argument("--out", type=Path, help="default: <file folder>/shorts/<stem>_short.mp4")
    ap.add_argument("--channel", default="RFP", help="neuron channel for the glow")
    ap.add_argument("--max-step-mm", type=float, default=0.5)
    ap.add_argument("--stars", help="comma-separated track ids to feature (default: auto, see header)")
    ap.add_argument("--preview", action="store_true", help="save a few stills instead of rendering")
    a = ap.parse_args()
    survey = a.survey or a.nd2.with_name(a.nd2.stem + "_survey")
    out = a.out or a.nd2.parent / "shorts" / f"{a.nd2.stem}_short.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    stars = [int(s) for s in a.stars.split(",")] if a.stars else []
    make(a.nd2, survey, out.resolve(), a.channel, a.max_step_mm, stars, a.preview)


if __name__ == "__main__":
    main()
