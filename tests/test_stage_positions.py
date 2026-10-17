import pytest

from acquisition.backends import nis_mock
from acquisition.backends.nis_mock import MockNIS
from acquisition.orchestration import stage_positions as sp


@pytest.fixture
def mgr(tmp_path, monkeypatch):
    monkeypatch.setattr(nis_mock.time, "sleep", lambda s: None)
    nis = MockNIS(100, 200, 10)
    return sp.StagePositionManager(nis_module=nis, positions_file=tmp_path / "p.json"), nis


def test_validate_position_limits():
    sp.validate_position(0, 0)
    sp.validate_position(57_000, -36_500)
    with pytest.raises(ValueError, match="X position"):
        sp.validate_position(57_001, 0)
    with pytest.raises(ValueError, match="Y position"):
        sp.validate_position(0, 36_501)


def test_to_plain_float_unwraps_numpy():
    import numpy as np
    v = sp.to_plain_float(np.float32(1.5))
    assert type(v) is float and v == 1.5


def test_save_list_and_persist(mgr, tmp_path):
    m, nis = mgr
    assert m.save_current("a") == {"x": 100.0, "y": 200.0, "z": 10.0}
    assert m.list_positions() == {"a": {"x": 100.0, "y": 200.0, "z": 10.0}}
    again = sp.StagePositionManager(nis_module=nis, positions_file=tmp_path / "p.json")
    assert "a" in again.list_positions()            # survived a restart


def test_save_current_refuses_an_impossible_reading(mgr):
    m, nis = mgr
    nis._x = 99_999.0
    with pytest.raises(ValueError):
        m.save_current("bad")
    assert m.list_positions() == {}


def test_define_position_validates_and_overwrites(mgr):
    m, _ = mgr
    m.define_position("p", 1, 2, 3)
    m.define_position("p", 4, 5, 6)
    assert m.list_positions()["p"] == {"x": 4.0, "y": 5.0, "z": 6.0}
    with pytest.raises(ValueError):
        m.define_position("far", 90_000, 0, 0)
    assert "far" not in m.list_positions()


def test_go_to_moves_xy_then_z(mgr):
    m, nis = mgr
    m.define_position("p", 500, 600, 20)
    m.go_to("p")
    assert nis.XY_GetPosition() == (500.0, 600.0) and nis.Z_GetPosition() == 20.0


def test_go_to_unknown_label(mgr):
    m, _ = mgr
    with pytest.raises(KeyError, match="No saved position"):
        m.go_to("nope")


def test_go_to_refuses_hand_edited_out_of_range_position(mgr):
    m, nis = mgr
    m._positions["bad"] = {"x": 90_000.0, "y": 0.0, "z": 0.0}
    with pytest.raises(ValueError):
        m.go_to("bad")
    assert nis.XY_GetPosition() == (100.0, 200.0)


def test_go_to_returns_xy_when_z_step_is_too_big(mgr):
    m, nis = mgr
    m.define_position("far_z", 500, 600, 10 + 51)          # beyond the 50 um per-call cap
    with pytest.raises(ValueError, match="per-call safety limit"):
        m.go_to("far_z")
    assert nis.XY_GetPosition() == (100.0, 200.0)           # moved back


def test_delete(mgr):
    m, _ = mgr
    m.define_position("p", 1, 2, 3)
    m.delete("p")
    assert m.list_positions() == {}
    with pytest.raises(KeyError):
        m.delete("p")


def test_load_positions_from_yaml(mgr, tmp_path):
    m, _ = mgr
    y = tmp_path / "proto.yaml"
    y.write_text("positions:\n  - {label: A, x: 1, y: 2, z: 3}\n  - {label: B, x: 4, y: 5, z: 6}\n")
    assert m.load_positions_from_yaml(y) == {"A": {"x": 1.0, "y": 2.0, "z": 3.0},
                                             "B": {"x": 4.0, "y": 5.0, "z": 6.0}}
    assert set(m.list_positions()) == {"A", "B"}


def test_print_summary(mgr, capsys):
    m, _ = mgr
    m.print_summary()
    assert "No saved positions" in capsys.readouterr().out
    m.define_position("alpha", 1, 2, 3)
    out = capsys.readouterr()
    m.print_summary()
    out = capsys.readouterr().out
    assert "alpha" in out and "x (um)" in out and "1.00" in out


def test_backend_selection(tmp_path):
    assert sp.StagePositionManager(positions_file=tmp_path / "a.json")._nis is sp.nis
    with pytest.raises(ValueError, match="Unknown backend"):
        sp.StagePositionManager(backend="bogus", positions_file=tmp_path / "b.json")
