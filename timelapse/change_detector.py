# change_detector.py
# ------------------------------------------------------------
# Cheap, model-free "did something happen?" score for one new frame.
#
# WHY NO LLM HERE: the fast loop of an adaptive time-lapse may run every
# few seconds for days. A vision-model call per frame is far too slow and
# expensive for that; it is also unnecessary - "a lot of pixels changed
# where the specimen is" is a numpy question. The model is only consulted
# when this detector fires (see scheduler.py), to decide what to do about
# it, not whether it happened.
#
# WHAT IT MEASURES (all on a small grayscale copy of the frame):
#   diff_score  - mean |frame - baseline| / baseline noise level, where
#                 the baseline is a running median of the last N frames.
#                 Robust to a single bad frame; drifts with slow change.
#   area_delta  - fractional change in the foreground-mask area (Otsu
#                 threshold on the baseline, applied to the new frame).
#                 A retraction/expansion of a specimen shows up here even
#                 when the per-pixel diff is spread thin.
#   shift_px    - global XY translation between baseline and frame (phase
#                 correlation). Large shift = the stage or the sample
#                 moved, which is NOT biology - the scheduler treats it as
#                 a separate event class.
#
# THRESHOLDS are configuration, not science - tune them on a recorded run
# (frame_audit.py prints the same quantities for an existing sequence, so
# the values seen at a known event give the starting point).
# ------------------------------------------------------------

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

ANALYSIS_MAX_DIMENSION = 256


def load_gray(path: str | Path, max_dimension: int = ANALYSIS_MAX_DIMENSION) -> np.ndarray:
    """Load an image as a float32 grayscale array, downscaled so its long
    edge is at most `max_dimension`. Small on purpose - the metrics below
    are about gross change, and a 256px thumbnail makes every one of them
    sub-millisecond."""
    with Image.open(path) as img:
        img = img.convert("L")
        img.thumbnail((max_dimension, max_dimension))
        return np.asarray(img, dtype=np.float32)


def otsu_threshold(gray: np.ndarray) -> float:
    """Otsu's threshold on a grayscale array (0-255 range). Pure numpy."""
    hist, edges = np.histogram(gray, bins=256, range=(0.0, 256.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    sum_bg = np.cumsum(hist * centers)
    mean_bg = np.divide(sum_bg, weight_bg, out=np.zeros_like(sum_bg), where=weight_bg > 0)
    mean_fg = np.divide(sum_bg[-1] - sum_bg, weight_fg, out=np.zeros_like(sum_bg), where=weight_fg > 0)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return float(centers[int(np.argmax(between))])


def foreground_fraction(gray: np.ndarray, threshold: float) -> float:
    """Fraction of pixels on the darker-than-threshold side. Works for
    brightfield specimens on a bright background; for fluorescence
    (bright specimen, dark background) callers pass `invert=True` to
    ChangeDetector."""
    return float((gray < threshold).mean())


def phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(dx, dy) in pixels that shifts `a` onto `b`, via phase correlation.
    Integer precision - good enough to tell "the stage moved" from "it
    didn't". Both arrays must have the same shape."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    cross = fa * np.conj(fb)
    denom = np.abs(cross)
    denom[denom == 0] = 1.0
    corr = np.fft.ifft2(cross / denom).real
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy, dx = peak
    h, w = a.shape
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    return float(-dx), float(-dy)


def estimate_shift(a: np.ndarray, b: np.ndarray, min_improvement: float = 0.5) -> tuple[float, float]:
    """Like phase_correlation_shift, but returns (0, 0) unless applying the
    candidate shift actually makes `b` match `a` much better - specifically
    unless it cuts mean|a - b| to below `min_improvement` of the unshifted
    value. Phase correlation happily returns a peak for a specimen that
    changed *shape* in place (a shrinking blob, say) with no real
    translation; this check rejects those."""
    dx, dy = phase_correlation_shift(a, b)
    if dx == 0.0 and dy == 0.0:
        return 0.0, 0.0
    unshifted = float(np.mean(np.abs(a - b)))
    if unshifted == 0.0:
        return 0.0, 0.0
    b_back = np.roll(b, shift=(int(-dy), int(-dx)), axis=(0, 1))
    shifted = float(np.mean(np.abs(a - b_back)))
    if shifted < unshifted * min_improvement:
        return dx, dy
    return 0.0, 0.0


@dataclass
class ChangeScore:
    diff_score: float
    area_delta: float
    shift_px: float
    foreground_fraction: float
    baseline_frames: int
    interesting: bool
    reason: str

    def as_dict(self) -> dict:
        return {
            "diff_score": round(self.diff_score, 3),
            "area_delta": round(self.area_delta, 4),
            "shift_px": round(self.shift_px, 1),
            "foreground_fraction": round(self.foreground_fraction, 4),
            "baseline_frames": self.baseline_frames,
            "interesting": self.interesting,
            "reason": self.reason,
        }


@dataclass
class ChangeDetector:
    """Scores each new frame against a running median of the previous
    `baseline_frames` frames. Feed it frames in time order via score().

    diff_threshold: diff_score above this is "interesting". Units are
        multiples of the baseline's own frame-to-frame noise, so ~3 means
        "three times noisier than a quiet stretch".
    area_threshold: |area_delta| above this fraction is "interesting"
        (0.05 = the specimen's footprint changed by 5%).
    shift_threshold_px: shift above this (in analysis-thumbnail pixels)
        is flagged as a stage/sample move rather than biology.
    invert: True for bright-on-dark images (fluorescence).
    """

    baseline_frames: int = 5
    diff_threshold: float = 3.0
    area_threshold: float = 0.05
    shift_threshold_px: float = 4.0
    invert: bool = False
    _history: deque = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.baseline_frames)

    def reset(self) -> None:
        self._history.clear()

    def score_array(self, gray: np.ndarray) -> ChangeScore:
        if self.invert:
            gray = 255.0 - gray
        n = len(self._history)
        if n == 0:
            self._history.append(gray)
            thr = otsu_threshold(gray)
            return ChangeScore(0.0, 0.0, 0.0, foreground_fraction(gray, thr), 0, False, "first frame")

        stack = np.stack(self._history)
        baseline = np.median(stack, axis=0)
        # Baseline noise: how much the baseline frames disagree with each
        # other. With a single frame there is no estimate; use a floor.
        noise = float(np.mean(np.abs(stack - baseline))) if n > 1 else 0.0
        noise = max(noise, 1.0)

        if gray.shape != baseline.shape:
            # Camera settings changed mid-run (binning, ROI). Restart.
            self.reset()
            self._history.append(gray)
            return ChangeScore(0.0, 0.0, 0.0, 0.0, 0, True, "frame size changed - baseline reset")

        diff_score = float(np.mean(np.abs(gray - baseline))) / noise
        thr = otsu_threshold(baseline)
        fg_base = foreground_fraction(baseline, thr)
        fg_new = foreground_fraction(gray, thr)
        area_delta = fg_new - fg_base
        dx, dy = estimate_shift(baseline, gray)
        shift_px = float(np.hypot(dx, dy))

        reasons = []
        if shift_px > self.shift_threshold_px:
            reasons.append(f"xy shift {shift_px:.1f}px (stage/sample moved?)")
        if abs(area_delta) > self.area_threshold:
            reasons.append(f"foreground area {'grew' if area_delta > 0 else 'shrank'} by {abs(area_delta):.1%}")
        if diff_score > self.diff_threshold:
            reasons.append(f"pixel change {diff_score:.1f}x baseline noise")

        self._history.append(gray)
        return ChangeScore(
            diff_score=diff_score,
            area_delta=float(area_delta),
            shift_px=shift_px,
            foreground_fraction=fg_new,
            baseline_frames=n,
            interesting=bool(reasons),
            reason="; ".join(reasons) if reasons else "quiet",
        )

    def score(self, path: str | Path) -> ChangeScore:
        return self.score_array(load_gray(path))
