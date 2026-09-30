import json

import pytest

from timelapse.scheduler import AdaptiveTimelapse, TimelapseConfig


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        assert s >= 0
        self.t += s


class FrameSource:
    """Serves frames in order; the scheduler's capture() pulls the next one.
    After the list runs out, keeps serving the last frame."""

    def __init__(self, clock, paths):
        self.clock = clock
        self.paths = list(paths)
        self.i = 0
        self.captured_at = []

    def __call__(self):
        path = self.paths[min(self.i, len(self.paths) - 1)]
        self.i += 1
        self.captured_at.append(self.clock())
        return {"image": str(path), "frame_id": self.i, "position": {"x": 0, "y": 0, "z": 0}}


def _config(tmp_path, **kw):
    base = dict(backend="mock", slow_interval_s=60.0, burst_interval_s=5.0, burst_duration_s=30.0,
                max_captures=40, max_runtime_s=10_000.0, events_path=tmp_path / "events.jsonl")
    base.update(kw)
    return TimelapseConfig(**base)


def _events(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_quiet_run_stays_slow_and_stops_at_max_captures(write_frame, tmp_path):
    clock = FakeClock()
    frames = [write_frame(f"q{i}.png", seed=i) for i in range(12)]
    src = FrameSource(clock, frames)
    cfg = _config(tmp_path, max_captures=10)
    summary = AdaptiveTimelapse(cfg, capture=src, clock=clock, sleep=clock.sleep, log=lambda s: None).run()
    assert summary.stop_reason == "max_captures"
    assert summary.captures == 10
    assert summary.bursts == 0
    gaps = [b - a for a, b in zip(src.captured_at, src.captured_at[1:])]
    assert all(g == 60.0 for g in gaps), gaps


def test_change_starts_burst_then_returns_to_slow(write_frame, tmp_path):
    clock = FakeClock()
    quiet = [write_frame(f"q{i}.png", seed=i) for i in range(6)]
    shrunk = [write_frame(f"s{i}.png", radius=10.0, seed=100 + i) for i in range(20)]
    src = FrameSource(clock, quiet + shrunk)
    cfg = _config(tmp_path, max_captures=25)
    summary = AdaptiveTimelapse(cfg, capture=src, clock=clock, sleep=clock.sleep, log=lambda s: None).run()

    kinds = [e["event"] for e in _events(cfg.events_path)]
    assert "burst_start" in kinds and "burst_end" in kinds
    assert summary.bursts >= 1
    # First burst starts at the 7th capture (first shrunk frame) - then 5s spacing.
    gaps = [b - a for a, b in zip(src.captured_at, src.captured_at[1:])]
    assert gaps[5] == 60.0            # slow gap into the first shrunk frame
    assert gaps[6] == 5.0             # burst spacing right after trigger
    assert summary.burst_captures > 0
    # After the specimen settles at its new size the baseline catches up
    # and the loop must be back on the slow interval by the end.
    assert gaps[-1] == 60.0, gaps


def test_stage_shift_does_not_start_burst(write_frame, tmp_path):
    clock = FakeClock()
    quiet = [write_frame(f"q{i}.png", seed=i) for i in range(6)]
    shifted = [write_frame(f"m{i}.png", center=(78, 64), seed=200 + i) for i in range(4)]
    src = FrameSource(clock, quiet + shifted)
    cfg = _config(tmp_path, max_captures=10)
    summary = AdaptiveTimelapse(cfg, capture=src, clock=clock, sleep=clock.sleep, log=lambda s: None).run()
    assert summary.shifts >= 1
    assert summary.bursts == 0


def test_on_trigger_ignore_ends_burst_immediately(write_frame, tmp_path):
    clock = FakeClock()
    quiet = [write_frame(f"q{i}.png", seed=i) for i in range(6)]
    shrunk = [write_frame(f"s{i}.png", radius=10.0, seed=100 + i) for i in range(6)]
    src = FrameSource(clock, quiet + shrunk)
    cfg = _config(tmp_path, max_captures=12)
    seen = []

    def on_trigger(event, metadata, score):
        seen.append(score.reason)
        return "ignore"

    summary = AdaptiveTimelapse(cfg, capture=src, on_trigger=on_trigger,
                                clock=clock, sleep=clock.sleep, log=lambda s: None).run()
    assert seen, "hook never called"
    assert summary.burst_captures == 0
    gaps = [b - a for a, b in zip(src.captured_at, src.captured_at[1:])]
    assert all(g == 60.0 for g in gaps), gaps


def test_max_runtime_stops_run(write_frame, tmp_path):
    clock = FakeClock()
    src = FrameSource(clock, [write_frame("q.png")])
    cfg = _config(tmp_path, max_captures=1000, max_runtime_s=300.0)
    summary = AdaptiveTimelapse(cfg, capture=src, clock=clock, sleep=clock.sleep, log=lambda s: None).run()
    assert summary.stop_reason == "max_runtime"
    assert summary.captures == 6  # t=0,60,...,300 -> the 300s check fires before a 7th


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        _config(tmp_path, burst_interval_s=120.0).validate()
    with pytest.raises(ValueError):
        _config(tmp_path, backend="nope").validate()


def test_cli_dry_run_prints_plan(capsys):
    from timelapse.scheduler import main
    assert main(["--dry-run", "--backend", "sdk", "--slow", "30"]) == 0
    out = capsys.readouterr().out
    assert "REAL HARDWARE" in out and "30s" in out


# -- background consults --------------------------------------------------------

class ManualExecutor:
    """Executor whose futures complete only when the test says so."""

    def __init__(self):
        self.futures = []

    def submit(self, fn, *args):
        from concurrent.futures import Future
        f = Future()
        f._fn_args = (fn, args)
        self.futures.append(f)
        return f

    def resolve(self, i, value=None, exc=None):
        f = self.futures[i]
        if exc is not None:
            f.set_exception(exc)
        else:
            f.set_result(value)

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _shrink_run(write_frame, tmp_path, max_captures):
    clock = FakeClock()
    quiet = [write_frame(f"q{i}.png", seed=i) for i in range(6)]
    shrunk = [write_frame(f"s{i}.png", radius=10.0, seed=100 + i) for i in range(30)]
    src = FrameSource(clock, quiet + shrunk)
    ex = ManualExecutor()
    calls = []
    tl = AdaptiveTimelapse(_config(tmp_path, max_captures=max_captures), capture=src,
                           on_trigger=lambda e, m, s: calls.append(e["frame_id"]),
                           clock=clock, sleep=clock.sleep, log=lambda s: None,
                           consult_in_background=True, executor=ex)
    return tl, src, ex, clock


def test_background_consult_does_not_block_burst_timing(write_frame, tmp_path):
    tl, src, ex, clock = _shrink_run(write_frame, tmp_path, max_captures=9)
    tl.run()
    gaps = [b - a for a, b in zip(src.captured_at, src.captured_at[1:])]
    assert gaps[6] == 5.0 and gaps[7] == 5.0       # burst spacing held while the consult was pending
    assert len(ex.futures) >= 1                      # the hook was submitted, never awaited


def test_background_extend_is_applied_at_next_step(write_frame, tmp_path):
    tl, src, ex, clock = _shrink_run(write_frame, tmp_path, max_captures=100)
    tl._t0 = clock()
    for _ in range(7):
        tl.step()                                    # 6 quiet + first shrunk -> burst_start, consult pending
    assert tl.mode == "burst" and len(ex.futures) == 1
    until_before = tl.burst_until
    ex.resolve(0, "extend")
    clock.t += 5.0
    tl.step()                                        # next frame also fires the detector - must not undo the extension
    assert tl.burst_until == until_before + tl.config.burst_duration_s
    kinds = [json.loads(l)["event"] for l in tl.config.events_path.read_text().splitlines()]
    assert kinds.count("burst_extend") >= 1


def test_background_ignore_ends_burst_and_late_reply_is_logged(write_frame, tmp_path):
    tl, src, ex, clock = _shrink_run(write_frame, tmp_path, max_captures=100)
    tl._t0 = clock()
    for _ in range(7):
        tl.step()
    assert tl.mode == "burst"
    ex.resolve(0, "ignore")
    clock.t += 5.0
    tl.step()
    events = [json.loads(l) for l in tl.config.events_path.read_text().splitlines()]
    assert any(e["event"] == "burst_end" and "ignore" in e["detail"] for e in events)

    # Keep stepping until the specimen's new size is the baseline and the
    # loop has settled back to slow on its own.
    for _ in range(12):
        clock.t += 5.0
        tl.step()
    assert tl.mode == "slow"
    # Consults still outstanding from those bursts now answer "extend" -
    # too late to matter, and logged as such.
    for i, f in enumerate(ex.futures):
        if not f.done():
            ex.resolve(i, "extend")
    clock.t += 60.0
    tl.step()
    assert tl.mode == "slow"
    events = [json.loads(l) for l in tl.config.events_path.read_text().splitlines()]
    assert any(e["event"] == "consult_late" for e in events)


def test_background_consult_exception_is_logged_not_raised(write_frame, tmp_path):
    tl, src, ex, clock = _shrink_run(write_frame, tmp_path, max_captures=100)
    tl._t0 = clock()
    for _ in range(7):
        tl.step()
    ex.resolve(0, exc=RuntimeError("model down"))
    clock.t += 5.0
    tl.step()                                        # must not raise
    events = [json.loads(l) for l in tl.config.events_path.read_text().splitlines()]
    assert any(e["event"] == "consult_failed" and "model down" in e["detail"] for e in events)
