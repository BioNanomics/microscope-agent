# Edge cases of the timelapse package: validation, CLI, error paths, and the
# real model call (against a fake SDK).
import json
import runpy
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from timelapse import change_detector as cd, frame_audit as fa, model_trigger as mt, scheduler as sc
from timelapse.change_detector import ChangeScore

ROOT = Path(__file__).parent.parent


# ---- change_detector ---------------------------------------------------------------

def test_otsu_of_an_empty_image_is_zero():
    assert cd.otsu_threshold(np.zeros((0, 0))) == 0.0


def test_phase_correlation_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        cd.phase_correlation_shift(np.zeros((8, 8)), np.zeros((8, 9)))


def test_estimate_shift_rejects_a_peak_that_does_not_improve_the_match():
    rng = np.random.default_rng(0)
    a, b = rng.random((32, 32)) * 255, rng.random((32, 32)) * 255          # unrelated noise
    assert cd.estimate_shift(a, b) == (0.0, 0.0)


def test_estimate_shift_of_identical_frames_is_zero(monkeypatch):
    a = np.random.default_rng(1).random((16, 16)) * 255
    assert cd.estimate_shift(a, a.copy()) == (0.0, 0.0)
    # even if the correlator reported a peak, frames with no difference at all moved nowhere
    monkeypatch.setattr(cd, "phase_correlation_shift", lambda x, y: (3.0, 1.0))
    assert cd.estimate_shift(a, a.copy()) == (0.0, 0.0)


def test_inverted_detector_and_path_scoring(write_frame):
    det = cd.ChangeDetector(invert=True)
    first = det.score(write_frame("a.png", seed=1))
    assert first.interesting is False and first.baseline_frames == 0


# ---- frame_audit ---------------------------------------------------------------------

def test_parse_duration_and_timestamp_variants():
    assert fa.parse_duration("90m") == 5400 and fa.parse_duration("30") == 30 and fa.parse_duration("1.5h") == 5400
    with pytest.raises(ValueError, match="bad duration"):
        fa.parse_duration("soon")
    assert fa.parse_timestamp("1700000000") == 1700000000.0
    assert fa.parse_timestamp("2026-10-15T14:00:00") == pytest.approx(
        __import__("datetime").datetime(2026, 10, 15, 14).timestamp())


def test_frames_from_dir_needs_matching_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="no files matching"):
        fa.frames_from_dir(tmp_path, "*.png")


def test_frames_from_history_skips_blank_lines_and_rejects_empty(tmp_path):
    h = tmp_path / "frames.jsonl"
    h.write_text('\n{"image": "a.png", "captured_at": "100"}\n\n{"image": "b.png", "captured_at": "160"}\n')
    frames = fa.frames_from_history(h)
    assert [(f.index, f.path.name, f.t_s) for f in frames] == [(0, "a.png", 100.0), (1, "b.png", 160.0)]
    h.write_text("\n\n")
    with pytest.raises(ValueError, match="no frame records"):
        fa.frames_from_history(h)


def test_apply_timestamps_checks_the_count_and_ignores_comments(tmp_path, write_frame):
    frames = fa.frames_from_dir(write_frame("a.png").parent, "*.png")
    ts = tmp_path / "ts.csv"
    ts.write_text("# header\n100\n")
    assert fa.apply_timestamps(frames, ts)[0].t_s == 100.0
    ts.write_text("100\n200\n")
    with pytest.raises(ValueError, match="2 timestamps but there are 1 frames"):
        fa.apply_timestamps(frames, ts)


def test_audit_needs_two_frames(write_frame):
    f = fa.FrameRecord(0, write_frame("a.png"), 0.0)
    with pytest.raises(ValueError, match="at least 2 frames"):
        fa.audit([f])


def test_audit_flags_size_changes_and_time_going_backwards(write_frame):
    frames = [fa.FrameRecord(0, write_frame("a.png", size=64), 100.0),
              fa.FrameRecord(1, write_frame("b.png", size=96), 90.0),
              fa.FrameRecord(2, write_frame("c.png", size=96), 120.0)]
    r = fa.audit(frames)
    assert "SIZE CHANGED" in r[0].flags and "NON-MONOTONIC TIME" in r[0].flags
    assert np.isnan(r[0].shift_px)


def test_audit_cli_flagged_only_from_history(write_frame, tmp_path, capsys):
    a, b, c = (write_frame(f"{n}.png", seed=i) for i, n in enumerate("abc"))
    h = tmp_path / "h.jsonl"
    h.write_text("".join(json.dumps({"image": str(p), "captured_at": str(t)}) + "\n"
                         for p, t in ((a, 0), (b, 60), (c, 600))))
    assert fa.main(["--history", str(h), "--flagged-only"]) == 0
    out = capsys.readouterr().out
    assert "3 frames, 2 intervals" in out and "GAP" in out


def test_audit_cli_with_a_timestamp_file(write_frame, tmp_path, capsys):
    write_frame("a.png", seed=1), write_frame("b.png", seed=2)
    ts = tmp_path / "ts.csv"
    ts.write_text("0\n30\n")
    assert fa.main([str(tmp_path), "--timestamps", str(ts)]) == 0
    assert "2 frames, 1 intervals" in capsys.readouterr().out


def test_audit_module_runs_as_a_script(write_frame, tmp_path, monkeypatch, capsys):
    write_frame("a.png", seed=1), write_frame("b.png", seed=2)
    monkeypatch.setattr(sys, "argv", ["frame_audit", str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("timelapse.frame_audit", run_name="__main__")
    assert exc.value.code == 0


# ---- model_trigger ---------------------------------------------------------------------

def test_parse_decision_with_broken_json_is_no_opinion():
    assert mt.parse_decision('see {"decision": extend,} ok') == (None, 'see {"decision": extend,} ok')


class FakeAnthropic:
    stop_reason = "end_turn"
    blocks = [NS(type="thinking"), NS(type="text", text='{"decision": '), NS(type="text", text='"extend"}')]
    seen = {}

    def __init__(self):
        self.messages = NS(create=lambda **kw: (FakeAnthropic.seen.update(kw),
                                                NS(stop_reason=FakeAnthropic.stop_reason,
                                                   content=FakeAnthropic.blocks))[1])


def test_anthropic_ask_joins_text_blocks_and_uses_adaptive_thinking(monkeypatch):
    import anthropic
    import dotenv
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    out = mt.anthropic_ask("system", [{"type": "text", "text": "hi"}])
    assert out == '{"decision": "extend"}'
    assert FakeAnthropic.seen["system"] == "system" and FakeAnthropic.seen["thinking"] == {"type": "adaptive"}
    assert FakeAnthropic.seen["model"] == mt.MODEL


def test_anthropic_ask_returns_nothing_on_a_refusal(monkeypatch):
    import anthropic
    import dotenv
    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.setattr(FakeAnthropic, "stop_reason", "refusal")
    assert mt.anthropic_ask("s", []) == ""


# ---- scheduler ------------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    dict(backend="nope"), dict(slow_interval_s=0), dict(burst_interval_s=-1),
    dict(burst_interval_s=90.0, slow_interval_s=60.0), dict(max_captures=0), dict(max_runtime_s=0)])
def test_config_validation_refuses_nonsense(bad):
    with pytest.raises(ValueError):
        sc.TimelapseConfig(**bad).validate()


def score(interesting=True, shift=0.0):
    return ChangeScore(diff_score=9.0, area_delta=0.2, shift_px=shift, foreground_fraction=0.3,
                       baseline_frames=5, interesting=interesting, reason="changed")


class Rig:
    """A timelapse whose detector returns scripted scores."""

    def __init__(self, tmp_path, write_frame, scores, **cfg):
        self.t = 1000.0
        self.frame = str(write_frame("f.png"))
        self.scores = list(scores)
        config = sc.TimelapseConfig(slow_interval_s=60, burst_interval_s=5, burst_duration_s=30,
                                    events_path=tmp_path / "ev.jsonl", **cfg)
        self.n = 0
        self.tl = sc.AdaptiveTimelapse(config, capture=self.capture, clock=lambda: self.t,
                                       sleep=self.sleep, log=lambda s: None)
        self.tl.detector = NS(score_array=lambda g: self.scores.pop(0) if self.scores else score(False))

    def capture(self):
        self.n += 1
        return {"image": self.frame, "frame_id": self.n, "position": {"x": 0, "y": 0, "z": 0}}

    def sleep(self, s):
        self.t += s

    def events(self):
        return [json.loads(l)["event"] for l in (self.tl.events_path).read_text().splitlines()]


def test_max_bursts_caps_how_many_bursts_start(tmp_path, write_frame):
    rig = Rig(tmp_path, write_frame, [score()], max_bursts=1)
    rig.tl.step()
    assert rig.tl.mode == "burst"
    rig.tl._end_burst("test")
    rig.scores.append(score())
    rig.tl.step()
    assert rig.tl.mode == "slow" and "burst_skipped" in rig.events() and rig.tl.summary.bursts == 1


def test_a_shift_never_starts_a_burst(tmp_path, write_frame):
    rig = Rig(tmp_path, write_frame, [score(shift=20.0)])
    rig.tl.step()
    assert rig.tl.mode == "slow" and rig.tl.summary.shifts == 1 and "shift" in rig.events()


def test_a_trigger_with_no_opinion_changes_nothing(tmp_path, write_frame):
    rig = Rig(tmp_path, write_frame, [score()])
    calls = []
    rig.tl.on_trigger = lambda e, m, s: calls.append(1)
    rig.tl.step()
    assert calls == [1] and rig.tl.mode == "burst" and rig.tl.burst_until == 1030.0


def test_background_consult_applies_a_reply_on_a_later_step(tmp_path, write_frame):
    rig = Rig(tmp_path, write_frame, [score(), score(False)])
    rig.tl.on_trigger = lambda e, m, s: "ignore"
    rig.tl.consult_in_background = True
    rig.tl.step()
    rig.tl._executor.shutdown(wait=True)                    # let the worker thread finish
    rig.tl.step()
    assert rig.tl.mode == "slow" and "burst_end" in rig.events()


def test_a_failing_background_consult_is_reported(tmp_path, write_frame):
    rig = Rig(tmp_path, write_frame, [score(), score(False)])

    def boom(e, m, s):
        raise RuntimeError("model down")

    rig.tl.on_trigger = boom
    rig.tl.consult_in_background = True
    rig.tl.step()
    rig.tl._executor.shutdown(wait=True)
    rig.tl.step()
    assert "consult_failed" in rig.events()


def test_a_reply_after_the_burst_ended_is_ignored(tmp_path, write_frame):
    rig = Rig(tmp_path, write_frame, [score(False)])
    rig.tl.mode = "slow"
    rig.tl._apply_decision("extend", 3, late_ok=True)
    assert "consult_late" in rig.events()
    rig.tl._apply_decision("extend", 3, late_ok=False)       # not late_ok: silent


def test_run_stops_cleanly_on_ctrl_c_and_abandons_pending_replies(tmp_path, write_frame):
    rig = Rig(tmp_path, write_frame, [score()])
    rig.tl.on_trigger = lambda e, m, s: "extend"

    class Manual:
        def __init__(self):
            self.futures = []

        def submit(self, fn, *a):
            f = Future()
            self.futures.append(f)
            return f

        def shutdown(self, wait=True, cancel_futures=False):
            pass

    rig.tl.consult_in_background = True
    rig.tl._executor = Manual()
    seen = []

    def capture():
        seen.append(1)
        if len(seen) > 1:
            raise KeyboardInterrupt
        return rig.capture()

    rig.tl.capture = capture
    summary = rig.tl.run()
    assert summary.stop_reason == "interrupted"
    assert {"consult_abandoned", "burst_end", "stop"} <= set(rig.events())


def test_run_stops_at_the_runtime_cap(tmp_path, write_frame):
    rig = Rig(tmp_path, write_frame, [], max_runtime_s=130)
    assert rig.tl.run().stop_reason == "max_runtime"


def test_default_capture_goes_through_get_image(monkeypatch, tmp_path):
    calls = []
    from mcp_server import loop_tools
    monkeypatch.setattr(loop_tools, "get_image", lambda **kw: (calls.append(kw), ({"image": "x.png"}, None))[1])
    cap = sc._default_capture(sc.TimelapseConfig(backend="mock"))
    assert cap() == {"image": "x.png"} and calls[0]["confirm"] is False and calls[0]["backend"] == "mock"
    sc._default_capture(sc.TimelapseConfig(backend="sdk"))()
    assert calls[1]["confirm"] is True


def test_plan_text_mentions_hardware_bursts_and_inversion():
    t = sc._plan_text(sc.TimelapseConfig(backend="sdk", max_bursts=3, invert=True))
    assert "REAL HARDWARE" in t and "at most 3 bursts" in t and "inverted" in t


def test_cli_dry_run_prints_the_plan(capsys):
    assert sc.main(["--dry-run", "--slow", "30", "--burst", "3"]) == 0
    assert "hard caps" in capsys.readouterr().out


def test_cli_real_hardware_needs_whole_plan_approval(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert sc.main(["--backend", "sdk"]) == 2
    assert "Nothing captured" in capsys.readouterr().out


def test_cli_runs_a_short_mock_session(tmp_path, monkeypatch, capsys):
    from mcp_server import loop_tools
    from tests.conftest import make_blob_frame, save_frame
    frame = save_frame(make_blob_frame(), tmp_path / "f.png")
    monkeypatch.setattr(loop_tools, "get_image", lambda **kw: ({"image": str(frame), "frame_id": 1, "position": {}}, None))
    events = tmp_path / "ev.jsonl"
    assert sc.main(["--max-captures", "2", "--slow", "0.01", "--burst", "0.01", "--events", str(events)]) == 0
    assert "done: max_captures; 2 captures" in capsys.readouterr().out and events.exists()


def test_cli_wires_up_the_model_trigger(tmp_path, monkeypatch):
    seen = {}

    class FakeTrigger:
        def __init__(self, max_calls):
            seen["max_calls"] = max_calls

        def __call__(self, *a):
            return None

    from mcp_server import loop_tools
    from tests.conftest import make_blob_frame, save_frame
    frame = save_frame(make_blob_frame(), tmp_path / "f.png")
    monkeypatch.setattr(loop_tools, "get_image", lambda **kw: ({"image": str(frame), "frame_id": 1, "position": {}}, None))
    monkeypatch.setattr(mt, "ModelTrigger", FakeTrigger)
    sc.main(["--max-captures", "1", "--slow", "0.01", "--burst", "0.01", "--model-trigger",
             "--model-max-calls", "7", "--events", str(tmp_path / "e.jsonl")])
    assert seen["max_calls"] == 7


def test_scheduler_module_runs_as_a_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["scheduler", "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("timelapse.scheduler", run_name="__main__")
    assert exc.value.code == 0
