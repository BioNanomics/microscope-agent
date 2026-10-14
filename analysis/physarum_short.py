# physarum_short.py
# ------------------------------------------------------------
# YouTube Short (1080x1920, 30 fps, H.264 + music) from the whole-dish
# Physarum mosaic runs: the time-lapse with a live growth curve drawn
# underneath, then a wipe from the last frame to the green new-growth map.
#
#   python -m analysis.physarum_short data/physarum_whole_dish_combined_20260923
#   python -m analysis.physarum_short <combined_dir> --preview      # 6 stills, seconds
#
# INPUTS, all in <combined_dir> (written by its combined_all_script.py):
#   combined_all_script.py           - run list + alignment into one pixel frame
#   oat_approach_all.csv             - on-agar area per round (the curve)
#   growth_start_to_end_colour.jpg   - new growth in green (the reveal)
# Output: <combined_dir>/shorts/physarum_short_growth.mp4 (+ .wav).
#
# ALIGNMENT is not redone here: the part of combined_all_script.py before
# "# ---- movie setup" (ECC warps of every run into the overnight run's
# frame, agar/oat masks) is executed as-is, so the short shows exactly the
# frames the analysis measured. The growth map is that same frame resized
# under a caption header, so its bottom H*width/W rows are pixel-aligned
# with the panel and the wipe needs no registration.
#
# THE CURVE is total_mm2 smoothed over 7 rounds (35 min) so the single-
# round segmentation flicker does not read as growth. Gaps between runs
# are drawn as straight lines.
#
# THIS IS THE 23-25 SEP 2026 EXPERIMENT'S SHORT: the phase captions (by
# hour), the oat label position and the default reveal caption describe
# that dish. Another experiment needs those changed, not just new paths.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from analysis.make_movie import wb_gains
from analysis.make_soundtrack import render
from analysis.short_video import (FONT_B, FONT_I, FONT_R, FPS, GREY, SR, VH, VW, ShortWriter, canvas,
                                  centred, font, write_wav)

YELLOW = (255, 214, 64)
GREEN = (80, 230, 90)
S_HOOK, S_INTRO, S_TL, S_REVEAL, S_END = 3, 3, 22, 9, 4
DURATION = S_HOOK + S_INTRO + S_TL + S_REVEAL + S_END
REVEAL_AT = S_HOOK + S_INTRO + S_TL
PANEL_Y = 400
A_LO, A_HI = 110, 180                     # curve y range, mm^2


def phase(t_h: float) -> str:
    if t_h < 9:
        return "Exploring in every direction"
    if t_h < 20:
        return "Spreading toward the food"
    if t_h < 24:
        return "Pulling back & rebuilding"
    return "Holding its network overnight"


def load_alignment(combined: Path) -> dict:
    """Run the analysis script's setup (runs, warps, frame size) into a namespace."""
    script = combined / "combined_all_script.py"
    ns = {"__file__": str(script), "__name__": "combined_setup"}
    exec(script.read_text().split("# ---- movie setup")[0], ns)
    return ns


def soundtrack() -> np.ndarray:
    """The Physarum theme, plus a soft tremolo swell into the reveal."""
    mix = render(DURATION)
    t = np.arange(len(mix)) / SR
    sw = np.clip((t - (REVEAL_AT - 2.0)) / 2.0, 0, 1) * np.exp(-np.clip(t - REVEAL_AT, 0, None) / 1.5)
    for f, pan in ((523.25, 0.3), (659.25, 0.7), (783.99, 0.5), (1046.5, 0.5)):
        s = np.sin(2 * np.pi * f * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 5.5 * t)) * sw * 0.06
        mix[:, 0] += s * (1 - pan); mix[:, 1] += s * pan
    return mix * 10 ** (-1 / 20) / np.abs(mix).max()


def make(combined: Path, out: Path, caption: str, preview: bool) -> None:
    ns = load_alignment(combined)
    RUNS, REF, W, H, UM_PX, to_ref = ns["RUNS"], ns["REF"], ns["W"], ns["H"], ns["UM_PX"], ns["to_ref"]

    ref0 = cv2.imread(str(REF / "mosaic_000.png"))
    gains = wb_gains(ref0)
    panel_h = int(H * VW / W)
    um_panel = UM_PX * W / VW

    def dish_frame(run, fname):
        img = cv2.imread(str(run / fname))
        img = np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)
        img = cv2.resize(img, (img.shape[1] * W // ref0.shape[1], img.shape[0] * W // ref0.shape[1]),
                         interpolation=cv2.INTER_AREA)
        img = to_ref(img, run)
        return cv2.cvtColor(cv2.resize(img, (VW, panel_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)

    def oats_label(d):
        tip = (40, PANEL_Y + int(0.50 * panel_h))
        base = (95, PANEL_Y + int(0.66 * panel_h))
        d.line([base, tip], fill=(0, 0, 0), width=10)
        d.line([base, tip], fill=YELLOW, width=5)
        d.polygon([tip, (tip[0] - 14, tip[1] + 30), (tip[0] + 18, tip[1] + 24)], fill=YELLOW)
        d.text((12, PANEL_Y + int(0.68 * panel_h)), "OATS", font=font(FONT_B, 46), fill=YELLOW,
               stroke_width=4, stroke_fill=(0, 0, 0))

    def scale_bar(d):
        px = int(round(5000 / um_panel))
        x1, y = VW - 40, PANEL_Y + panel_h - 30
        d.rectangle([x1 - px, y - 10, x1, y], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
        f = font(FONT_B, 30)
        d.text((x1 - px / 2 - d.textlength("5 mm", font=f) / 2, y - 48), "5 mm", font=f,
               fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))

    # ---- growth curve ----
    rows = list(csv.DictReader((combined / "oat_approach_all.csv").open()))
    CT = np.array([float(r["t_h"]) for r in rows])
    CA = np.array([float(r["total_mm2"]) for r in rows])
    CA_S = np.convolve(np.pad(CA, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    peak = int(np.argmax(CA_S))
    x0, x1 = 120, VW - 60
    y0, y1 = PANEL_Y + panel_h + 150, VH - 110

    def cx(t):
        return x0 + (x1 - x0) * t / CT[-1]

    def cy(a):
        return y1 - (y1 - y0) * (a - A_LO) / (A_HI - A_LO)

    def curve(img, t_h, show_peak):
        d = ImageDraw.Draw(img)
        f_ax, f_lb = font(FONT_R, 30), font(FONT_B, 38)
        d.text((x0 - 60, y0 - 112), "Slime mould on the agar (mm²)", font=f_lb, fill=(235, 235, 235))
        for a in (120, 140, 160, 180):
            y = cy(a)
            d.line([(x0, y), (x1, y)], fill=(48, 48, 54), width=2)
            d.text((x0 - 14 - d.textlength(str(a), font=f_ax), y - 20), str(a), font=f_ax, fill=GREY)
        for h in (0, 10, 20, 30):
            d.text((cx(h) - 12, y1 + 8), f"{h}h", font=f_ax, fill=GREY)
        n = int(np.searchsorted(CT, t_h, side="right"))
        if n >= 2:
            pts = [(cx(t), cy(a)) for t, a in zip(CT[:n], CA_S[:n])]
            glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
            ImageDraw.Draw(glow).line(pts, fill=YELLOW + (150,), width=16, joint="curve")
            glow = glow.filter(ImageFilter.GaussianBlur(8))
            img.paste(glow, (0, 0), glow)
            d = ImageDraw.Draw(img)
            d.line(pts, fill=YELLOW, width=6, joint="curve")
            x, y = pts[-1]
            d.ellipse([x - 12, y - 12, x + 12, y + 12], fill=(255, 255, 255), outline=YELLOW, width=4)
            lab = f"{CA_S[n - 1]:.0f} mm²"
            d.text((min(x + 18, x1 - d.textlength(lab, font=f_lb)), y - 58), lab, font=f_lb,
                   fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        if show_peak and n > peak:
            x, y = cx(CT[peak]), cy(CA_S[peak])
            d.line([(x, y - 16), (x, y - 50)], fill=GREEN, width=3)
            t = f"peak {CA_S[peak]:.0f} mm² at {CT[peak]:.0f} h"
            d.text((x - d.textlength(t, font=f_ax) / 2, y - 92), t, font=f_ax, fill=GREEN)

    # ---- rounds, in time order across runs ----
    rounds, t0 = [], None
    for run in RUNS:
        starts = {}
        for r in csv.DictReader((run / "tiles.csv").open()):
            starts.setdefault(int(r["round"]), datetime.datetime.fromisoformat(r["wall_clock"]))
        for rnd in sorted(starts):
            fn = f"mosaic_{rnd:03d}.png"
            if (run / fn).exists():
                t0 = t0 or starts[rnd]
                rounds.append((run, fn, (starts[rnd] - t0).total_seconds() / 3600))
    total_h = rounds[-1][2]
    print(f"{len(rounds)} rounds, {total_h:.1f} h; area peak {CA_S[peak]:.0f} mm² at {CT[peak]:.1f} h")

    gm = Image.open(combined / "growth_start_to_end_colour.jpg").convert("RGB")
    gm = gm.crop((0, gm.height - round(H * gm.width / W), gm.width, gm.height))
    growth = np.asarray(gm.resize((VW, panel_h), Image.LANCZOS))

    wav = None
    if not preview:
        wav = out.with_suffix(".wav")
        write_wav(soundtrack(), wav)
    stills = {2 * FPS, 5 * FPS, (S_HOOK + S_INTRO + 15) * FPS, (REVEAL_AT + 1) * FPS,
              (REVEAL_AT + 6) * FPS, (DURATION - 2) * FPS}
    with ShortWriter(out, wav, preview, stills) as w:
        # 1. hook + 2. intro over the first frame
        first = dish_frame(*rounds[0][:2])
        for i in range((S_HOOK + S_INTRO) * FPS):
            if not w.wants(i):
                continue
            c = canvas(); c.paste(Image.fromarray(first), (0, PANEL_Y)); d = ImageDraw.Draw(c)
            if i < S_HOOK * FPS:
                centred(d, 110, "This is alive.", font(FONT_B, 104))
                centred(d, 255, "A single cell. No brain.", font(FONT_R, 62), fill=(220, 220, 220))
            else:
                centred(d, 90, "Slime mould", font(FONT_B, 92))
                centred(d, 205, "Physarum polycephalum", font(FONT_I, 60), fill=YELLOW)
                centred(d, 290, f"{total_h:.0f} hours under a microscope", font(FONT_R, 54),
                        fill=(220, 220, 220))
            oats_label(d); scale_bar(d); curve(c, 0, False)
            w.emit(c, i)

        # 3. time-lapse with the live curve
        base = (S_HOOK + S_INTRO) * FPS
        n_tl = S_TL * FPS
        cache: dict = {}
        for i in range(n_tl):
            if not w.wants(base + i) and i != n_tl - 1:
                continue
            k = min(len(rounds) - 1, int(i * len(rounds) / n_tl))
            if k not in cache:
                cache = {k: dish_frame(*rounds[k][:2])}
            t_h = rounds[k][2]
            c = canvas(); c.paste(Image.fromarray(cache[k]), (0, PANEL_Y)); d = ImageDraw.Draw(c)
            centred(d, 110, phase(t_h), font(FONT_B, 70))
            centred(d, 215, f"hour {t_h:4.1f}", font(FONT_R, 60), fill=YELLOW)
            oats_label(d); scale_bar(d); curve(c, t_h, True)
            w.emit(c, base + i)
            if not preview and i % 60 == 0:
                print(f"timelapse {i}/{n_tl}", flush=True)
        last = cache[k]

        # 4. wipe from the last frame to the new-growth map
        base = REVEAL_AT * FPS
        for i in range(S_REVEAL * FPS):
            if not w.wants(base + i):
                continue
            p = min(1.0, i / (2.0 * FPS)); p = 0.5 - 0.5 * np.cos(np.pi * p)
            xw = int(p * VW)
            panel = last.copy(); panel[:, :xw] = growth[:, :xw]
            c = canvas(); c.paste(Image.fromarray(panel), (0, PANEL_Y)); d = ImageDraw.Draw(c)
            if 0 < xw < VW:
                d.line([(xw, PANEL_Y), (xw, PANEL_Y + panel_h)], fill=(255, 255, 255), width=5)
            centred(d, 100, "So what actually grew?", font(FONT_B, 76))
            if i > 1.5 * FPS:
                centred(d, 215, "Green = brand-new slime mould", font(FONT_B, 54), fill=GREEN)
            if i > 3.5 * FPS:
                centred(d, 290, caption, font(FONT_R, 48), fill=(220, 220, 220))
            curve(c, total_h, True)
            w.emit(c, base + i)

        # 5. end card
        base = (REVEAL_AT + S_REVEAL) * FPS
        for i in range(S_END * FPS):
            if not w.wants(base + i):
                continue
            c = canvas(); c.paste(Image.fromarray(growth), (0, PANEL_Y)); d = ImageDraw.Draw(c)
            centred(d, 90, f"{total_h:.0f} hours in {DURATION} seconds", font(FONT_B, 76))
            centred(d, 205, "It never stopped rebuilding itself.", font(FONT_R, 52), fill=(220, 220, 220))
            centred(d, 275, "Nikon Ti2 · 4x · 1 photo-mosaic every 5 min", font(FONT_R, 42), fill=GREY)
            curve(c, total_h, True)
            w.emit(c, base + i)
    print("wrote", out if not preview else f"preview stills in {out.parent}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("combined_dir", type=Path)
    ap.add_argument("--out", type=Path, help="default: <combined_dir>/shorts/physarum_short_growth.mp4")
    ap.add_argument("--caption", default="+20 mm² grew  ·  17 mm² pulled back",
                    help="reveal caption (the default is the 23-25 Sep figures)")
    ap.add_argument("--preview", action="store_true", help="save a few stills instead of rendering")
    a = ap.parse_args()
    out = a.out or a.combined_dir / "shorts" / "physarum_short_growth.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    make(a.combined_dir.resolve(), out.resolve(), a.caption, a.preview)


if __name__ == "__main__":
    main()
