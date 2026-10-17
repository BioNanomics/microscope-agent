import csv
import runpy
import sys
from pathlib import Path

import pytest
from PIL import Image

from acquisition import estop, move_xy
from acquisition.orchestration import timelapse
from tests.fakes import fake_ti2  # noqa: F401  (fixture)
from tests.fakes_camera import install_fake_camera_stack


@pytest.fixture
def clear_estop(tmp_path, monkeypatch):
    monkeypatch.setattr(estop, "ESTOP_PATH", tmp_path / "ESTOP")
    monkeypatch.setattr(move_xy.time, "sleep", lambda s: None)


# ---- move_xy ------------------------------------------------------------

class FlatStage:
    def __init__(self):
        self.pos, self.moves = (0.0, 0.0), []

    def XY_GetPosition(self):
        return self.pos

    def XY_Move(self, x, y):
        self.moves.append((x, y))
        self.pos = (x, y)


def test_move_to_splits_long_moves_into_legal_hops(clear_estop, capsys):
    s = FlatStage()
    end = move_xy.move_to(s, 15000, 0)
    assert end == (15000.0, 0.0)
    assert len(s.moves) == 4 and all(b[0] - a[0] <= move_xy.HOP_UM + 1e-6
                                     for a, b in zip([(0, 0)] + s.moves, s.moves))
    assert "um to go" in capsys.readouterr().out


def test_move_to_quiet_and_already_there(clear_estop, capsys):
    s = FlatStage()
    assert move_xy.move_to(s, 0.5, 0.0) == (0.0, 0.0) and s.moves == []
    move_xy.move_to(s, 9000, 0, quiet=True)
    assert "to go" not in capsys.readouterr().out


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["move_xy", *argv])
    move_xy.main()


def test_main_only_reports_position(fake_ti2, clear_estop, monkeypatch, capsys):
    sdk, m = fake_ti2
    run_main(monkeypatch)
    assert "current: X=100.00  Y=200.00" in capsys.readouterr().out
    assert m.iXPOSITION == 1000


def test_main_relative_move_in_hops_through_the_sdk(fake_ti2, clear_estop, monkeypatch, capsys):
    sdk, m = fake_ti2
    run_main(monkeypatch, "--by", "9000", "0", "--yes")
    assert m.iXPOSITION == round((100 + 9000) * 10)
    assert "arrived" in capsys.readouterr().out


def test_main_absolute_move(fake_ti2, clear_estop, monkeypatch):
    sdk, m = fake_ti2
    run_main(monkeypatch, "--to", "300", "400")
    assert (m.iXPOSITION, m.iYPOSITION) == (3000, 4000)


def test_main_long_move_needs_yes_when_not_interactive(fake_ti2, clear_estop, monkeypatch, capsys):
    sdk, m = fake_ti2
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--by", "5000", "0")
    assert exc.value.code == 3 and m.iXPOSITION == 1000


@pytest.mark.parametrize("answer,moved", [("y", True), ("n", False)])
def test_main_long_move_asks_when_interactive(fake_ti2, clear_estop, monkeypatch, capsys, answer, moved):
    sdk, m = fake_ti2
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    run_main(monkeypatch, "--by", "3000", "0")
    assert (m.iXPOSITION != 1000) is moved
    assert ("cancelled" in capsys.readouterr().out) is (not moved)


def test_main_refuses_when_estop_is_engaged(fake_ti2, clear_estop, monkeypatch, capsys):
    sdk, m = fake_ti2
    estop.engage("test")
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--by", "10", "0")
    assert exc.value.code == 2 and "E-STOP IS ENGAGED" in capsys.readouterr().out


def test_main_reports_a_stop_that_lands_mid_move(fake_ti2, clear_estop, monkeypatch, capsys):
    sdk, m = fake_ti2
    real_move = sdk.NISSdk.XY_Move
    calls = []

    def move_then_stop(self, x, y):
        real_move(self, x, y)
        calls.append(1)
        estop.engage("mid-move")

    monkeypatch.setattr(sdk.NISSdk, "XY_Move", move_then_stop)
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--by", "9000", "0", "--yes")
    assert exc.value.code == 2 and len(calls) == 1
    assert "STOPPED BY E-STOP" in capsys.readouterr().out


# ---- raw_move_xy (a hardware experiment script, run top to bottom) -------

def test_raw_move_script_runs_against_a_fake_microscope(fake_ti2, monkeypatch, capsys):
    import os
    sdk, m = fake_ti2
    exits = []

    def fake_exit(code):
        exits.append(code)
        raise SystemExit(code)

    monkeypatch.setattr(os, "_exit", fake_exit)
    monkeypatch.setattr(sys, "coinit_flags", 0, raising=False)
    sys.modules["pythoncom"].CoInitializeEx = lambda flag: None
    sys.modules["pythoncom"].COINIT_MULTITHREADED = 0
    with pytest.raises(SystemExit):
        runpy.run_path(str(Path(__file__).parent.parent / "acquisition" / "raw_move_xy.py"), run_name="raw_move_xy")
    out = capsys.readouterr().out
    assert exits == [0] and "watchdog: halted" in out and "exit" in out
    assert (m.iXPOSITION, m.iYPOSITION) == (0, 0)


# ---- timelapse ------------------------------------------------------------

@pytest.fixture
def lapse(tmp_path, monkeypatch):
    mod, acq, log = install_fake_camera_stack(monkeypatch, tmp_path, pixel_format="RGB8")
    monkeypatch.setattr(timelapse, "data_root", lambda: tmp_path)
    monkeypatch.setattr(timelapse.time, "sleep", lambda s: None)
    yield mod, acq, tmp_path
    sys.modules.pop("acquisition.backends.baumer_genicam", None)


def rows(out):
    with (out / "frames.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_series_dir_name_carries_the_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(timelapse, "data_root", lambda: tmp_path)
    assert timelapse._series_dir("oats").name.endswith("_oats")
    assert "_oats" not in timelapse._series_dir(None).name


def test_stats_describe_exposure_and_focus(tmp_path):
    p = tmp_path / "f.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(p)
    s = timelapse._stats(p)
    assert s["mean"] == 255.0 and s["sat_frac"] == 1.0 and s["focus"] == 0.0


def test_run_writes_frames_and_csv(lapse, capsys):
    mod, acq, tmp = lapse
    out = timelapse.run(frames=3, interval=0, exposure_us=1600, gain=2.0, tag="t")
    r = rows(out)
    assert [x["frame"] for x in r] == ["0", "1", "2"]
    assert all((out / x["filename"]).exists() and x["error"] == "" for x in r)
    assert acq.remote_device.node_map.ExposureTime.value == 1600 and acq.stopped
    assert "done in" in capsys.readouterr().out


def test_a_failed_frame_is_recorded_and_the_run_continues(lapse, monkeypatch, capsys):
    mod, acq, tmp = lapse
    real = mod.BaumerGenICam.capture
    state = {"n": 0}

    def flaky(self):
        state["n"] += 1
        if state["n"] == 3:                            # 1 = throwaway, 2 = frame 0, 3 = frame 1
            raise RuntimeError("fetch timed out")
        return real(self)

    monkeypatch.setattr(mod.BaumerGenICam, "capture", flaky)
    out = timelapse.run(frames=3, interval=0, exposure_us=1600, gain=1.0)
    r = rows(out)
    assert r[1]["error"] == "RuntimeError: fetch timed out" and r[1]["filename"] == ""
    assert r[0]["error"] == r[2]["error"] == ""
    assert "frame 1: RuntimeError" in capsys.readouterr().out


def test_camera_is_closed_even_if_setup_fails(lapse, monkeypatch):
    mod, acq, tmp = lapse
    with pytest.raises(ValueError):
        timelapse.run(frames=1, interval=0, exposure_us=1, gain=1.0)     # outside the camera's range
    assert acq.stopped


def test_late_frames_are_flagged(lapse, monkeypatch):
    mod, acq, tmp = lapse
    clock = iter(x * 5.0 for x in range(1000))
    monkeypatch.setattr(timelapse.time, "monotonic", lambda: next(clock))
    out = timelapse.run(frames=2, interval=1, exposure_us=1600, gain=1.0)
    assert float(rows(out)[1]["late_s"]) > 0


def test_sleeps_until_the_scheduled_slot(lapse, monkeypatch):
    mod, acq, tmp = lapse
    slept = []
    monkeypatch.setattr(timelapse.time, "sleep", lambda s: slept.append(s))
    timelapse.run(frames=2, interval=30, exposure_us=1600, gain=1.0)
    assert slept and slept[0] > 0


def test_main_frame_count_from_minutes_or_frames(lapse, monkeypatch):
    calls = []
    monkeypatch.setattr(timelapse, "run", lambda *a: calls.append(a))
    monkeypatch.setattr(sys, "argv", ["tl", "--minutes", "1", "--interval", "10"])
    timelapse.main()
    monkeypatch.setattr(sys, "argv", ["tl", "--frames", "4", "--tag", "x"])
    timelapse.main()
    assert calls[0][0] == 7 and calls[1][0] == 4 and calls[1][-1] == "x"


def test_main_needs_minutes_or_frames(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tl"])
    with pytest.raises(SystemExit) as exc:
        timelapse.main()
    assert exc.value.code == 2
