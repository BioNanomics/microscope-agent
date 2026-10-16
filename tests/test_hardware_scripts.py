# The hardware experiment scripts (estop_inflight_test, read_during_move_test)
# and every module's `python -m` entry point, run against the fake Ti2 and
# the mock stage. Their 60 s / 20 s safety timers and os._exit calls are
# replaced so nothing waits and the test process is never killed.
import importlib
import os
import runpy
import sys
import threading
import time
from pathlib import Path

import pytest

from acquisition import estop
from acquisition.backends import nis_mock
from tests.fakes import fake_ti2  # noqa: F401  (fixture)

ROOT = Path(__file__).parent.parent


class InstantTimer:
    """threading.Timer that runs its function at start() (or never, if told to)."""
    fire = True
    started = []

    def __init__(self, interval, fn, *a):
        self.interval, self.fn, self.a = interval, fn, a

    def start(self):
        InstantTimer.started.append(self.interval)
        if InstantTimer.fire and self.interval < 10:      # the 20 s and 60 s safety timers never fire
            self.fn(*self.a)

    def is_alive(self):
        return False


@pytest.fixture
def scripts(fake_ti2, monkeypatch, tmp_path):
    sdk, m = fake_ti2
    InstantTimer.fire, InstantTimer.started = True, []
    monkeypatch.setattr(threading, "Timer", InstantTimer)
    real_sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda s: real_sleep(0.001))
    exits = []
    monkeypatch.setattr(os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(estop, "ESTOP_PATH", tmp_path / "ESTOP")
    monkeypatch.setattr(sys, "coinit_flags", 0, raising=False)
    sys.modules.pop("acquisition.estop_inflight_test", None)
    mod = importlib.import_module("acquisition.estop_inflight_test")
    yield mod, m, exits
    sys.modules.pop("acquisition.estop_inflight_test", None)


def sticky(m, attr, setter):
    """Make `attr` on the fake microscope store through `setter`."""
    store = {"v": getattr(m, attr)}
    m.__class__ = type("Sticky", (type(m),), {
        attr: property(lambda self: store["v"], lambda self, v: store.__setitem__("v", setter(store["v"], v)))})
    return store


# ---- estop_inflight_test ------------------------------------------------------

def test_log_names_the_thread(scripts, capsys):
    mod, m, _ = scripts
    mod.log("hello")
    assert "hello" in capsys.readouterr().out


def test_z_test_reports_a_move_the_halt_did_not_stop(scripts, capsys):
    mod, m, _ = scripts
    start = m.iZPOSITION
    mod.test_z()
    out = capsys.readouterr().out
    assert m.iZPOSITION == start - 100_000 and "halt did not stop it" in out


def test_z_test_reports_a_halt_that_reverses_the_move(scripts, capsys):
    mod, m, _ = scripts
    sticky(m, "iZPOSITION", lambda old, new: old)               # every write is ignored: back at the start
    mod.test_z()
    assert "Z back at start" in capsys.readouterr().out


def test_z_test_reports_a_partial_move(scripts, capsys):
    mod, m, _ = scripts
    sticky(m, "iZPOSITION", lambda old, new: (old + new) // 2)  # only half way
    mod.test_z()
    assert "halt works on Z" in capsys.readouterr().out


def test_z_test_logs_a_failed_write_and_a_failed_halt(scripts, capsys):
    mod, m, _ = scripts

    def boom(old, new):
        raise RuntimeError("controller busy")

    sticky(m, "iZPOSITION", boom)
    mod.test_z()
    out = capsys.readouterr().out
    assert "set iZPOSITION failed" in out and "write-back failed: controller busy" in out


def test_z_test_will_not_move_below_the_travel_range(scripts, capsys):
    mod, m, _ = scripts
    m.ZPosition.Lower = m.iZPOSITION                               # no room to drop 1 mm
    start = m.iZPOSITION
    mod.test_z()
    assert m.iZPOSITION == start and "not moving" in capsys.readouterr().out


def test_z_test_refuses_under_estop(scripts):
    mod, m, _ = scripts
    estop.engage("t")
    with pytest.raises(estop.EStopEngaged):
        mod.test_z()


def test_xy_test_is_stopped_by_the_estop_engaging_mid_move(scripts, capsys):
    mod, m, _ = scripts
    mod.test_xy()
    out = capsys.readouterr().out
    assert "stopped by e-stop" in out and "kept moving" in out and estop.is_engaged()


def test_xy_test_reports_a_move_that_nothing_stopped(scripts, capsys):
    mod, m, _ = scripts
    InstantTimer.fire = False
    mod.test_xy()
    assert "WITHOUT being stopped" in capsys.readouterr().out and m.iXPOSITION == round((100 + 20000) * 10)


def test_xy_test_refuses_under_estop(scripts):
    mod, m, _ = scripts
    estop.engage("t")
    with pytest.raises(estop.EStopEngaged):
        mod.test_xy()


def run_inflight(argv):
    sys.argv = ["estop_inflight_test", *argv]
    runpy.run_path(str(ROOT / "acquisition" / "estop_inflight_test.py"), run_name="__main__")


def test_inflight_entry_point_needs_z_or_xy(scripts, monkeypatch):
    monkeypatch.setattr(sys, "argv", sys.argv)
    for argv in ([], ["q"], ["z", "extra"]):
        with pytest.raises(SystemExit) as exc:
            run_inflight(argv)
        assert exc.value.code == 2


def test_inflight_entry_point_runs_the_chosen_test(scripts, monkeypatch, capsys):
    mod, m, exits = scripts
    monkeypatch.setattr(sys, "argv", sys.argv)
    run_inflight(["z"])
    assert exits == [0] and "RESULT: Z" in capsys.readouterr().out and 60.0 in InstantTimer.started
    estop.release()
    run_inflight(["xy"])
    assert exits == [0, 0] and "RESULT: kept moving" in capsys.readouterr().out


# ---- read_during_move_test --------------------------------------------------------

def test_read_during_move_script_runs_and_logs_events(scripts, monkeypatch, capsys):
    mod, m, exits = scripts
    sys.modules.pop("acquisition.read_during_move_test", None)
    rd = importlib.import_module("acquisition.read_during_move_test")
    try:
        out = capsys.readouterr().out
        assert exits == [0] and "set iXPOSITION done" in out and "read done" in out
        assert (m.iXPOSITION, m.iYPOSITION) == (0, 0)
        assert [c[0] for c in m.commands] == ["SET_MOVING_STATE_NOTIFICATION", "GET_MOVING_STATE_NOTIFICATION"]
        assert "GET_MOVING_STATE_NOTIFICATION failed" in out                    # the fake refuses the read-back
        ev = rd.Events()
        ev.OnGeneralCallback(3, object())
        ev.OnDedicatedCallback(25, "busy")
        ev.OnMetaDataCallback(0x10)
        ev.OnEnabledCallback(0x20)
        out = capsys.readouterr().out
        assert "XYStageLogicalLimitsReached" in out and "Ti2_IsBusy" in out and "metadata 0x10" in out
    finally:
        sys.modules.pop("acquisition.read_during_move_test", None)


# ---- python -m entry points of the other modules ----------------------------------

@pytest.fixture
def entry(monkeypatch, tmp_path):
    monkeypatch.setattr(nis_mock.time, "sleep", lambda s: None)
    monkeypatch.setattr(estop, "ESTOP_PATH", tmp_path / "ESTOP")
    monkeypatch.setenv("CONFOCAL_NO_ESTOP_PANEL", "1")
    monkeypatch.setenv("CONFOCAL_MCP_DATA_DIR", str(tmp_path))
    from acquisition import paths
    paths.data_root.cache_clear()
    yield
    paths.data_root.cache_clear()


def test_nis_mock_demo(entry, capsys):
    runpy.run_module("acquisition.backends.nis_mock", run_name="__main__")
    out = capsys.readouterr().out
    assert "Starting position" in out and "Expected error on out-of-range move" in out


def test_stage_positions_demo(entry, capsys):
    runpy.run_module("acquisition.orchestration.stage_positions", run_name="__main__")
    out = capsys.readouterr().out
    assert "sample_edge" in out and "Summary of all saved positions" in out


def test_nis_sdk_demo(fake_ti2, entry, capsys):
    runpy.run_module("acquisition.backends.nis_sdk", run_name="__main__")
    assert "Current position: (100.0, 200.0) 4619.0" in capsys.readouterr().out


def test_estop_cli_entry_point(entry, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["estop", "status"])
    runpy.run_module("acquisition.estop", run_name="__main__")
    assert "clear" in capsys.readouterr().out


def test_orchestration_entry_points_reach_their_argument_parsers(entry, monkeypatch):
    for module in ("acquisition.orchestration.timelapse",):
        monkeypatch.setattr(sys, "argv", ["x"])                       # no --minutes/--frames: the parser refuses
        with pytest.raises(SystemExit) as exc:
            runpy.run_module(module, run_name="__main__")
        assert exc.value.code == 2
    for module in ("acquisition.orchestration.mosaic", "acquisition.orchestration.worm_follow",
                   "acquisition.move_xy"):
        monkeypatch.setattr(sys, "argv", ["x", "--help"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_module(module, run_name="__main__")
        assert exc.value.code == 0
