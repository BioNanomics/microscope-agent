# scheduler.py
# ------------------------------------------------------------
# Adaptive time-lapse: image slowly, look for change cheaply, image fast
# when something is happening.
#
#   slow mode   - one capture every `slow_interval_s`
#   burst mode  - one capture every `burst_interval_s` for `burst_duration_s`
#                 after the ChangeDetector fires; each further "interesting"
#                 frame inside the burst extends it
#   back to slow when the burst window runs out
#
# WHY: a fixed-interval time-lapse spends its light and disk budget evenly
# on the boring hours and the interesting minutes alike, and the event you
# care about (a retraction that looked instantaneous at a 20-minute
# interval, say) falls between two frames. This loop puts the fast frames
# where the change is.
#
# WHAT DECIDES: timelapse.change_detector - numpy, no model call, so the
# slow loop costs nothing per frame. An optional `on_trigger` hook runs
# when a burst starts or extends; that is the place to ask a vision
# model "is this worth a longer look?" (return "extend" / "ignore"), or
# to notify someone. The hook is off the fast path: it is called once per
# trigger, not once per frame. timelapse.model_trigger.ModelTrigger is
# the Claude implementation of that hook (--model-trigger on the CLI).
#
# WHAT IT DOES NOT DO: move the stage, refocus, or change exposure. It
# only calls get_image(). A stage/sample shift the detector sees is
# logged as a "shift" event and does NOT start a burst - that is not
# biology, and burst-imaging a drifted field is wasted light.
#
# HARD CAPS (max_captures, max_runtime_s) always end the run, whatever
# mode it is in. On real hardware the run is a single up-front approval
# of the whole plan (see main()): the per-call confirm=True that
# get_image() requires is supplied by this scheduler after that one
# approval, never by anything the model says. Keep it that way.
#
# Log: one JSON line per event in logs/timelapse_events.jsonl (under the
# CONFOCAL_MCP_DATA_DIR data root, same as frame_history.jsonl), so a run
# can be replayed or audited with frame_audit.py --history afterwards.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from acquisition.paths import logs_dir as _logs_dir
from timelapse.change_detector import ChangeDetector, ChangeScore, load_gray

CaptureFn = Callable[[], dict]                       # -> get_image() metadata dict
TriggerFn = Callable[[dict, dict, ChangeScore], str | None]  # (event, metadata, score) -> "extend" | "ignore" | None


@dataclass
class TimelapseConfig:
    backend: str = "mock"
    slow_interval_s: float = 60.0
    burst_interval_s: float = 5.0
    burst_duration_s: float = 120.0
    max_captures: int = 500
    max_runtime_s: float = 3600.0
    max_bursts: int | None = None
    baseline_frames: int = 5
    diff_threshold: float = 3.0
    area_threshold: float = 0.05
    shift_threshold_px: float = 4.0
    invert: bool = False
    preview_max_dimension: int = 256   # passed to get_image - small, the model isn't looking
    events_path: Path | None = None    # default: <data root>/logs/timelapse_events.jsonl

    def validate(self) -> None:
        if self.backend not in ("mock", "sdk"):
            raise ValueError(f"backend must be 'mock' or 'sdk', got {self.backend!r}")
        if self.burst_interval_s <= 0 or self.slow_interval_s <= 0:
            raise ValueError("intervals must be > 0")
        if self.burst_interval_s > self.slow_interval_s:
            raise ValueError("burst_interval_s must not exceed slow_interval_s")
        if self.max_captures < 1 or self.max_runtime_s <= 0:
            raise ValueError("max_captures and max_runtime_s must be positive")

    def detector(self) -> ChangeDetector:
        return ChangeDetector(
            baseline_frames=self.baseline_frames,
            diff_threshold=self.diff_threshold,
            area_threshold=self.area_threshold,
            shift_threshold_px=self.shift_threshold_px,
            invert=self.invert,
        )


def _default_capture(config: TimelapseConfig) -> CaptureFn:
    """Capture via loop_tools.get_image(). For backend='sdk' this supplies
    confirm=True - main() has already obtained the one-time human approval
    for the whole run before this is ever constructed."""
    from mcp_server import loop_tools

    def capture() -> dict:
        metadata, _preview = loop_tools.get_image(
            backend=config.backend,
            confirm=(config.backend == "sdk"),
            max_dimension=config.preview_max_dimension,
        )
        return metadata
    return capture


@dataclass
class RunSummary:
    captures: int = 0
    bursts: int = 0
    burst_captures: int = 0
    shifts: int = 0
    elapsed_s: float = 0.0
    stop_reason: str = ""
    events_path: str = ""
    frames: list[dict] = field(default_factory=list)  # per-capture event dicts


class AdaptiveTimelapse:
    def __init__(
        self,
        config: TimelapseConfig,
        capture: CaptureFn | None = None,
        on_trigger: TriggerFn | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
    ):
        config.validate()
        self.config = config
        self.capture = capture or _default_capture(config)
        self.on_trigger = on_trigger
        self.clock = clock
        self.sleep = sleep
        self.log = log
        self.detector = config.detector()
        self.events_path = Path(config.events_path) if config.events_path else _logs_dir() / "timelapse_events.jsonl"
        self.mode = "slow"
        self.burst_until: float | None = None
        self.summary = RunSummary(events_path=str(self.events_path))
        self._t0: float | None = None
        self._previous_image: str | None = None

    # -- state helpers -------------------------------------------------
    def _elapsed(self) -> float:
        return self.clock() - (self._t0 if self._t0 is not None else self.clock())

    def _interval(self) -> float:
        return self.config.burst_interval_s if self.mode == "burst" else self.config.slow_interval_s

    def _emit(self, event: dict) -> None:
        event = {"at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                 "elapsed_s": round(self._elapsed(), 3), "mode": self.mode, **event}
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.events_path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
        if event["event"] != "capture":
            self.log(f"[timelapse +{event['elapsed_s']:.0f}s] {event['event']}: {event.get('detail', '')}")

    # -- one iteration ---------------------------------------------------
    def step(self) -> dict:
        """Capture one frame, score it, update mode. Returns the capture event."""
        metadata = self.capture()
        score = self.detector.score_array(load_gray(metadata["image"]))
        now = self.clock()
        self.summary.captures += 1
        if self.mode == "burst":
            self.summary.burst_captures += 1

        event = {"event": "capture", "frame_id": metadata.get("frame_id"), "image": metadata.get("image"),
                 "previous_image": self._previous_image,
                 "position": metadata.get("position"), "score": score.as_dict()}
        self._emit(event)
        self.summary.frames.append(event)
        self._previous_image = metadata.get("image")

        is_shift = score.shift_px > self.config.shift_threshold_px
        if is_shift:
            self.summary.shifts += 1
            self._emit({"event": "shift", "frame_id": metadata.get("frame_id"),
                        "detail": f"{score.shift_px:.1f}px - stage/sample moved, not starting a burst"})
        elif score.interesting:
            if self.mode == "slow":
                if self.config.max_bursts is not None and self.summary.bursts >= self.config.max_bursts:
                    self._emit({"event": "burst_skipped", "detail": "max_bursts reached"})
                else:
                    self.mode = "burst"
                    self.summary.bursts += 1
                    self.burst_until = now + self.config.burst_duration_s
                    self._emit({"event": "burst_start", "frame_id": metadata.get("frame_id"), "detail": score.reason})
                    self._consult(event, metadata, score)
            else:
                self.burst_until = now + self.config.burst_duration_s
                self._emit({"event": "burst_extend", "frame_id": metadata.get("frame_id"), "detail": score.reason})
                self._consult(event, metadata, score)

        if self.mode == "burst" and self.burst_until is not None and now >= self.burst_until:
            self._end_burst("burst window elapsed")
        return event

    def _consult(self, event: dict, metadata: dict, score: ChangeScore) -> None:
        if self.on_trigger is None:
            return
        decision = self.on_trigger(event, metadata, score)
        if decision == "ignore":
            self._end_burst("on_trigger said ignore")
        elif decision == "extend" and self.burst_until is not None:
            self.burst_until += self.config.burst_duration_s
            self._emit({"event": "burst_extend", "detail": "on_trigger said extend"})

    def _end_burst(self, why: str) -> None:
        self.mode = "slow"
        self.burst_until = None
        self._emit({"event": "burst_end", "detail": why})

    # -- the loop --------------------------------------------------------
    def run(self) -> RunSummary:
        self._t0 = self.clock()
        self._emit({"event": "start", "detail": json.dumps(asdict(self.config), default=str)})
        next_at = self._t0
        try:
            while True:
                if self.summary.captures >= self.config.max_captures:
                    self.summary.stop_reason = "max_captures"
                    break
                if self._elapsed() >= self.config.max_runtime_s:
                    self.summary.stop_reason = "max_runtime"
                    break
                wait = next_at - self.clock()
                if wait > 0:
                    self.sleep(wait)
                self.step()
                # Schedule from the intended slot, not from "now", so slow
                # captures don't drift; but never queue up a backlog.
                next_at = max(next_at + self._interval(), self.clock())
                if self.mode == "burst" and self.burst_until is not None:
                    next_at = min(next_at, self.burst_until)
        except KeyboardInterrupt:
            self.summary.stop_reason = "interrupted"
        finally:
            if self.mode == "burst":
                self._end_burst("run ended")
            self.summary.elapsed_s = self._elapsed()
            self._emit({"event": "stop", "detail": f"{self.summary.stop_reason}; {self.summary.captures} captures, "
                                                     f"{self.summary.bursts} bursts, {self.summary.shifts} shifts"})
        return self.summary


# -- CLI ----------------------------------------------------------------------

def _plan_text(config: TimelapseConfig) -> str:
    slow_only = config.max_runtime_s / config.slow_interval_s
    return (
        f"  backend           {config.backend}{'  (REAL HARDWARE)' if config.backend == 'sdk' else ''}\n"
        f"  slow interval     {config.slow_interval_s:g}s  (~{slow_only:.0f} captures if nothing happens)\n"
        f"  burst             every {config.burst_interval_s:g}s for {config.burst_duration_s:g}s per trigger"
        f"{'' if config.max_bursts is None else f', at most {config.max_bursts} bursts'}\n"
        f"  hard caps         {config.max_captures} captures, {config.max_runtime_s:g}s runtime\n"
        f"  thresholds        diff>{config.diff_threshold:g}x noise, |area|>{config.area_threshold:.0%}, "
        f"shift>{config.shift_threshold_px:g}px{', inverted (bright-on-dark)' if config.invert else ''}\n"
    )


def _approve_real_hardware(config: TimelapseConfig) -> bool:
    print("\n[HARDWARE GATE] This run will fire the REAL camera repeatedly, unattended, with this plan:")
    print(_plan_text(config))
    answer = input("Approve this whole acquisition plan? [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Adaptive time-lapse: slow captures, burst mode on detected change.")
    ap.add_argument("--backend", choices=["mock", "sdk"], default="mock")
    ap.add_argument("--slow", type=float, default=60.0, help="slow-mode interval, seconds")
    ap.add_argument("--burst", type=float, default=5.0, help="burst-mode interval, seconds")
    ap.add_argument("--burst-duration", type=float, default=120.0, help="seconds of burst per trigger")
    ap.add_argument("--max-captures", type=int, default=500)
    ap.add_argument("--max-runtime", type=float, default=3600.0, help="seconds")
    ap.add_argument("--max-bursts", type=int, default=None)
    ap.add_argument("--baseline-frames", type=int, default=5)
    ap.add_argument("--diff-threshold", type=float, default=3.0)
    ap.add_argument("--area-threshold", type=float, default=0.05)
    ap.add_argument("--shift-px", type=float, default=4.0)
    ap.add_argument("--invert", action="store_true", help="bright specimen on dark background (fluorescence)")
    ap.add_argument("--events", type=Path, default=None, help="events JSONL path (default: logs/timelapse_events.jsonl)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit without capturing")
    ap.add_argument("--model-trigger", action="store_true",
                    help="ask Claude at each trigger whether to extend or end the burst (needs API credentials)")
    ap.add_argument("--model-max-calls", type=int, default=50, help="cap on model calls per run")
    args = ap.parse_args(argv)

    config = TimelapseConfig(
        backend=args.backend, slow_interval_s=args.slow, burst_interval_s=args.burst,
        burst_duration_s=args.burst_duration, max_captures=args.max_captures, max_runtime_s=args.max_runtime,
        max_bursts=args.max_bursts, baseline_frames=args.baseline_frames, diff_threshold=args.diff_threshold,
        area_threshold=args.area_threshold, shift_threshold_px=args.shift_px, invert=args.invert,
        events_path=args.events,
    )
    config.validate()
    if args.dry_run:
        print(_plan_text(config))
        return 0
    if config.backend == "sdk" and not _approve_real_hardware(config):
        print("Not approved. Nothing captured.")
        return 2

    on_trigger = None
    if args.model_trigger:
        from timelapse.model_trigger import ModelTrigger
        on_trigger = ModelTrigger(max_calls=args.model_max_calls)
    summary = AdaptiveTimelapse(config, on_trigger=on_trigger).run()
    print(f"done: {summary.stop_reason}; {summary.captures} captures ({summary.burst_captures} in {summary.bursts} bursts), "
          f"{summary.shifts} shifts, {summary.elapsed_s:.0f}s. Events: {summary.events_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
