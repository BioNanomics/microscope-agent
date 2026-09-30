# physarum_rhythm.py
# ------------------------------------------------------------
# Quantify a fixed-position brightfield time-lapse of a Physarum
# plasmodium: the contraction rhythm, and how the organism's footprint
# changes over the run.
#
#   python -m analysis.physarum_rhythm data/timelapse_20260921_150051_physarum
#   python -m analysis.physarum_rhythm <dir> --json out.json
#
# THE MEASUREMENT: in transmitted light a plasmodium's local brightness
# tracks its local thickness - a thicker strand absorbs/scatters more, so
# it reads darker. Physarum drives cytoplasm through its network with a
# peristaltic contraction wave, so thickness at a fixed point rises and
# falls, and transmitted intensity oscillates with it. The period of that
# oscillation is the quantity of interest; the textbook figure for
# P. polycephalum is ~100-130 s at room temperature.
#
# WHY A PER-PIXEL PERIODOGRAM AND NOT JUST THE FRAME MEAN: neighbouring
# regions of one plasmodium oscillate at a common period but *out of
# phase* - that phase gradient is the peristaltic wave. Averaging the
# whole frame first lets antiphase regions cancel, which can bury a
# strong local rhythm in a flat global trace. So the period is taken from
# the median of per-pixel dominant frequencies inside the mask, and the
# global trace is reported alongside as a sanity check, not as the
# primary estimate.
#
# DETRENDING: the slow drift from the organism advancing across the field
# (and any lamp drift) is a large low-frequency component that would
# otherwise dominate every periodogram. Each pixel's series is linearly
# detrended and Hann-windowed before the FFT, and frequencies below
# MIN_PERIOD_S/MAX_PERIOD_S are excluded from the peak search.
#
# SCALE: areas are reported in mm^2 when the series folder carries a
# calibration.json with a measured `um_per_px`, and in PIXELS otherwise -
# never by assuming a magnification. acquisition/orchestration's runs
# since 2026-09-21 write that file; older series predate it and will
# report pixels, which is the honest answer for them.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

#: Period search window, seconds. Wide enough to contain the textbook
#: ~100-130 s without presupposing it - a result landing hard against
#: either edge should be read as "outside the searched band", not as a
#: measurement.
MIN_PERIOD_S = 30.0
MAX_PERIOD_S = 600.0

#: Spatial downsample for the per-pixel analysis. The rhythm is a
#: large-scale thickness wave, not a fine texture, so full resolution buys
#: nothing here and costs ~25x the memory (a full-res float stack of a
#: 360-frame run is several GB).
DOWNSAMPLE = 5


def _um_per_px(series_dir: Path) -> float | None:
    """Measured pixel size for this series, or None if it was never measured.

    Returning None rather than a default is deliberate: a plausible-looking
    wrong scale silently turns every area into a wrong physical number,
    which is worse than an honest pixel count.
    """
    path = series_dir / "calibration.json"
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("um_per_px")
        return float(value) if value else None
    except (ValueError, OSError):
        return None


def _load_series(series_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (times_s, small_stack, filenames) for the frames that exist.

    Frames whose row carries an `error`, or whose file is missing, are
    skipped rather than faked - a gap in the series is better than an
    interpolated frame in a periodogram.
    """
    rows = list(csv.DictReader((series_dir / "frames.csv").open(encoding="utf-8")))
    times, frames, names = [], [], []
    for r in rows:
        if r.get("error") or not r.get("filename"):
            continue
        path = series_dir / r["filename"]
        if not path.exists():
            continue
        g = np.asarray(Image.open(path)).mean(axis=2).astype(np.float32)
        frames.append(g[::DOWNSAMPLE, ::DOWNSAMPLE])
        times.append(float(r["t_seconds"]))
        names.append(r["filename"])
    if len(frames) < 8:
        raise SystemExit(f"only {len(frames)} usable frames in {series_dir} - too few to analyse")
    return np.asarray(times), np.stack(frames), names


#: Band a plasmodium contraction rhythm is expected to fall in. Used ONLY
#: to score which Otsu region is the organism (see _regions), never to
#: constrain the reported period - that is searched over the full
#: MIN/MAX_PERIOD_S window so a result outside this band can still come out.
_BIO_BAND_S = (60.0, 200.0)


def _regions(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two Otsu classes of the time-averaged frame, as (bright, dark).

    Otsu on the mean image rather than per frame: a per-frame threshold
    would itself flicker with the contraction rhythm, writing the signal
    being measured into the very region it is measured over.

    WHICH ONE IS THE ORGANISM IS NOT DECIDED HERE, deliberately. A thick
    pigmented plasmodium usually reads darker than bare agar in
    transmitted light, but that depends on the illumination, the
    substrate and how thin the organism has spread - and getting it
    backwards would measure the substrate with full confidence. analyse()
    scores both regions for rhythmicity instead and lets the oscillation
    identify the living one.
    """
    mean_img = stack.mean(axis=0)
    u8 = cv2.normalize(mean_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    bright = th > 0
    return bright, ~bright


def _dominant_period(series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel dominant period (s) and its spectral power fraction.

    `series` is (T, N) - time along axis 0. Returns (period, strength),
    each length N. `strength` is the peak's share of total in-band power:
    a clean oscillation concentrates power in one bin, noise spreads it,
    so it is the natural weight for "how periodic is this pixel".
    """
    T = series.shape[0]
    t = np.arange(T, dtype=np.float64)

    # Linear detrend per pixel (closed form - avoids scipy).
    tm = t.mean()
    denom = ((t - tm) ** 2).sum()
    slope = ((t - tm)[:, None] * (series - series.mean(axis=0))).sum(axis=0) / denom
    detr = series - (slope[None, :] * (t - tm)[:, None] + series.mean(axis=0))

    win = np.hanning(T)[:, None]
    spec = np.abs(np.fft.rfft(detr * win, axis=0)) ** 2
    freqs = np.fft.rfftfreq(T, d=dt)

    band = (freqs >= 1.0 / MAX_PERIOD_S) & (freqs <= 1.0 / MIN_PERIOD_S)
    if not band.any():
        raise SystemExit("period search band contains no FFT bins - run longer or sample faster")

    spec_b, freqs_b = spec[band], freqs[band]
    idx = spec_b.argmax(axis=0)
    peak = spec_b[idx, np.arange(spec_b.shape[1])]
    total = spec_b.sum(axis=0)
    strength = np.divide(peak, total, out=np.zeros_like(peak), where=total > 0)
    return 1.0 / freqs_b[idx], strength


def _score_region(flat: np.ndarray, mflat: np.ndarray, dt: float) -> dict:
    """Periodogram summary for one region, plus a rhythmicity score.

    `rhythmic_fraction` - the share of the region's pixels that both
    oscillate cleanly (peak holds >=15% of in-band power) and do so
    within _BIO_BAND_S - is what distinguishes a contracting organism
    from substrate that merely drifts or flickers.
    """
    period, strength = _dominant_period(flat[:, mflat], dt)
    lo, hi = _BIO_BAND_S
    rhythmic = (strength >= 0.15) & (period >= lo) & (period <= hi)
    good = strength >= np.percentile(strength, 75)
    return {
        "pixels": int(mflat.sum()),
        "mean_intensity": round(float(flat[:, mflat].mean()), 2),
        "period_s": round(float(np.median(period[good])), 1),
        "period_iqr_s": [round(float(np.percentile(period[good], 25)), 1),
                         round(float(np.percentile(period[good], 75)), 1)],
        "rhythmic_fraction": round(float(rhythmic.mean()), 3),
        "_period": period, "_strength": strength, "_good": good,
    }


def analyse(series_dir: Path) -> dict:
    times, stack, names = _load_series(series_dir)
    dt = float(np.median(np.diff(times)))
    umpx = _um_per_px(series_dir)
    bright, dark = _regions(stack)

    T = stack.shape[0]
    flat = stack.reshape(T, -1)

    scored = {name: _score_region(flat, m.reshape(-1), dt)
              for name, m in (("bright", bright), ("dark", dark))}

    # The organism is whichever region actually oscillates in the
    # biological band - not whichever is brighter. If the two scores are
    # close, that is reported rather than papered over: it means the
    # segmentation did not separate organism from substrate, and the
    # period below should not be trusted without looking at the frames.
    order = sorted(scored, key=lambda k: scored[k]["rhythmic_fraction"], reverse=True)
    organism, other = order[0], order[1]
    margin = scored[organism]["rhythmic_fraction"] - scored[other]["rhythmic_fraction"]

    sel = scored[organism]
    mflat = (bright if organism == "bright" else dark).reshape(-1)
    period, strength, good = sel["_period"], sel["_strength"], sel["_good"]
    period_est = sel["period_s"]

    # Global trace (sanity check - see module header on why it is not primary).
    trace = flat[:, mflat].mean(axis=1)

    # Footprint over time, from a fixed threshold so area changes reflect
    # the organism, not a moving threshold.
    u8m = cv2.normalize(stack.mean(axis=0), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    thr, _ = cv2.threshold(u8m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = float(stack.min()), float(stack.max())
    level = lo + (thr / 255.0) * (hi - lo)
    side = (stack > level) if organism == "bright" else (stack < level)
    area = side.reshape(T, -1).sum(axis=1).astype(float)
    area_px = area * (DOWNSAMPLE ** 2)

    return {
        "series_dir": str(series_dir),
        "frames_used": T,
        "duration_s": float(times[-1] - times[0]),
        "interval_s": dt,
        "organism_region": organism,
        "organism_identified_by": "rhythmicity",
        "rhythmic_margin": round(float(margin), 3),
        "region_scores": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                          for k, v in scored.items()},
        "mask_pixels_downsampled": int(mflat.sum()),
        "period_s": round(period_est, 1),
        "period_iqr_s": [round(float(np.percentile(period[good], 25)), 1),
                         round(float(np.percentile(period[good], 75)), 1)],
        "cycles_observed": round(float(times[-1] - times[0]) / period_est, 1),
        "oscillating_pixel_fraction": round(float((strength > 0.15).mean()), 3),
        "area_start_px": round(area_px[0], 0),
        "area_end_px": round(area_px[-1], 0),
        "area_change_pct": round(100.0 * (area_px[-1] - area_px[0]) / area_px[0], 2),
        "um_per_px": umpx,
        "area_start_mm2": round(area_px[0] * umpx ** 2 / 1e6, 4) if umpx else None,
        "area_end_mm2": round(area_px[-1] * umpx ** 2 / 1e6, 4) if umpx else None,
        "area_change_mm2": (round((area_px[-1] - area_px[0]) * umpx ** 2 / 1e6, 4)
                            if umpx else None),
        "growth_rate_mm2_per_h": (
            round((area_px[-1] - area_px[0]) * umpx ** 2 / 1e6
                  / max((times[-1] - times[0]) / 3600.0, 1e-9), 4) if umpx else None),
        "trace_t_s": [round(float(x), 1) for x in times],
        "trace_intensity": [round(float(x), 4) for x in trace],
        "trace_area_px": [round(float(x), 0) for x in area_px],
        "period_map_percentiles": {
            str(p): round(float(np.percentile(period[good], p)), 1)
            for p in (5, 25, 50, 75, 95)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("series_dir", type=Path)
    ap.add_argument("--json", type=Path, help="also write the full result here")
    args = ap.parse_args()

    res = analyse(args.series_dir)
    if args.json:
        args.json.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(f"frames            {res['frames_used']} over {res['duration_s'] / 60:.1f} min "
          f"@ {res['interval_s']:.1f}s")
    rs = res["region_scores"]
    print(f"organism region   {res['organism_region']} "
          f"(rhythmic fraction {rs[res['organism_region']]['rhythmic_fraction']} vs "
          f"{min(v['rhythmic_fraction'] for v in rs.values())}, "
          f"margin {res['rhythmic_margin']})")
    if res["rhythmic_margin"] < 0.05:
        print("  WARNING: the two regions are near-equally rhythmic - segmentation")
        print("  did not separate organism from substrate. Treat the period below")
        print("  as unverified and look at the frames.")
    print(f"contraction period {res['period_s']} s  "
          f"(IQR {res['period_iqr_s'][0]}-{res['period_iqr_s'][1]} s, "
          f"{res['cycles_observed']} cycles observed)")
    print(f"oscillating pixels {res['oscillating_pixel_fraction'] * 100:.1f}% of the plasmodium mask")
    if res["um_per_px"]:
        print(f"footprint          {res['area_change_pct']:+.2f}%  "
              f"({res['area_start_mm2']:.4f} -> {res['area_end_mm2']:.4f} mm2, "
              f"{res['area_change_mm2']:+.4f} mm2)")
        print(f"growth rate        {res['growth_rate_mm2_per_h']:+.4f} mm2/h "
              f"(scale {res['um_per_px']} um/px, measured)")
    else:
        print(f"footprint          {res['area_change_pct']:+.2f}% "
              f"({res['area_start_px']:.0f} -> {res['area_end_px']:.0f} px) "
              f"- no calibration.json, so pixels only")


if __name__ == "__main__":
    main()
