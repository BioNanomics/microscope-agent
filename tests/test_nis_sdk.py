import pytest

from acquisition import estop
from tests.fakes import fake_ti2  # noqa: F401  (fixture)


@pytest.fixture
def clear_estop(tmp_path, monkeypatch):
    monkeypatch.setattr(estop, "ESTOP_PATH", tmp_path / "ESTOP")
    return tmp_path / "ESTOP"


def test_positions_are_converted_to_microns(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    s = sdk.NISSdk()
    assert s.XY_GetPosition() == (100.0, 200.0) and s.Z_GetPosition() == 4619.0


def test_all_instances_share_one_com_thread(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    assert sdk.NISSdk()._thread is sdk.NISSdk()._thread


def test_xy_move_writes_counts(fake_ti2, clear_estop, capsys):
    sdk, m = fake_ti2
    sdk.NISSdk().XY_Move(150.5, 250.3)
    assert (m.iXPOSITION, m.iYPOSITION) == (1505, 2503)
    assert "XY_Move" in capsys.readouterr().out


def test_xy_move_step_cap_and_travel_range(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    s = sdk.NISSdk()
    with pytest.raises(ValueError, match="per-call safety limit"):
        s.XY_Move(100 + 5001, 200)
    m.XPosition.Higher = 1500                                   # soft limit at 150 um
    with pytest.raises(ValueError, match="X target"):
        s.XY_Move(200, 200)
    m.YPosition.Lower = 3000
    with pytest.raises(ValueError, match="Y target"):
        s.XY_Move(100, 250)
    assert (m.iXPOSITION, m.iYPOSITION) == (1000, 2000)


def test_xy_move_warns_when_readback_differs(fake_ti2, clear_estop, capsys):
    sdk, m = fake_ti2
    m.__class__ = type("Sticky", (type(m),), {
        "iXPOSITION": property(lambda self: 1000, lambda self, v: None)})
    sdk.NISSdk().XY_Move(110, 200)
    assert "WARNING" in capsys.readouterr().out


def test_z_move_writes_counts_and_enforces_limits(fake_ti2, clear_estop, capsys):
    sdk, m = fake_ti2
    s = sdk.NISSdk()
    s.Z_Move(4619 + 10)
    assert m.iZPOSITION == 462900
    with pytest.raises(ValueError, match="per-call safety limit"):
        s.Z_Move(4629 + 51)
    m.ZPosition.Higher = 462000
    with pytest.raises(ValueError, match="Z target"):
        s.Z_Move(4629 + 5)
    assert m.iZPOSITION == 462900


def test_z_move_warns_when_readback_differs(fake_ti2, clear_estop, capsys):
    sdk, m = fake_ti2
    m.__class__ = type("Sticky", (type(m),), {
        "iZPOSITION": property(lambda self: 461900, lambda self, v: None)})
    sdk.NISSdk().Z_Move(4620)
    assert "WARNING" in capsys.readouterr().out


def test_estop_blocks_every_motion_primitive(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    s = sdk.NISSdk()
    estop.engage("test")
    for call in (lambda: s.XY_Move(101, 200), lambda: s.Z_Move(4620), lambda: s.nudge_pfs_offset(10)):
        with pytest.raises(estop.EStopEngaged):
            call()
    assert (m.iXPOSITION, m.iYPOSITION, m.iZPOSITION, m.iPFS_OFFSET) == (1000, 2000, 461900, 18966)


def test_pfs_status(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    st = sdk.NISSdk().pfs_status()
    assert st["enabled"] is True and st["offset"] == 18966 and st["offset_higher"] == 40000
    assert st["objective_magnification"] == 10 and st["working_distance_mm"] == 4.0


def test_pfs_nudge_applies_within_cap(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    assert sdk.NISSdk().nudge_pfs_offset(100) == {"offset_before": 18966, "offset_after": 19066}


def test_pfs_nudge_refusals(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    s = sdk.NISSdk()
    with pytest.raises(ValueError, match="safety limit"):           # 2% of 40000 = 800
        s.nudge_pfs_offset(801)
    m.iPFS_OFFSET = 100
    with pytest.raises(ValueError, match="outside the reported"):
        s.nudge_pfs_offset(-200)
    m.objective.IsPFSEnabled = False
    try:
        with pytest.raises(PermissionError, match="PFS is not currently enabled"):
            s.nudge_pfs_offset(10)
    finally:
        m.objective.IsPFSEnabled = True
    assert m.iPFS_OFFSET == 100


def test_optical_configuration_snapshot_skips_unreadable(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    cfg = sdk.NISSdk().get_optical_configuration()
    assert cfg == {"iDIA_LAMP_Switch": 1, "iLIGHTPATH": 2, "iNOSEPIECE": 3, "iXPOSITION": 5}


def test_apply_configuration_skips_motion_unless_asked(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    s = sdk.NISSdk()
    cfg = {"iLIGHTPATH": 4, "iDIA_LAMP_Switch": 1, "iNOSEPIECE": 5, "iXPOSITION": 7, "iZPOSITION": 9}
    assert s.apply_optical_configuration(cfg) == {"iLIGHTPATH": 4, "iDIA_LAMP_Switch": 1}
    assert (m.iNOSEPIECE, m.iXPOSITION) == (1, 1000)
    out = s.apply_optical_configuration({"iNOSEPIECE": 5}, include_motion=True)
    assert out == {"iNOSEPIECE": 5} and m.iNOSEPIECE == 5


def test_apply_configuration_records_none_for_a_property_that_raises(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    m.__class__ = type("Locked", (type(m),), {
        "iLIGHTPATH": property(lambda self: 1, lambda self, v: (_ for _ in ()).throw(RuntimeError("no")))})
    assert sdk.NISSdk().apply_optical_configuration({"iLIGHTPATH": 3}) == {"iLIGHTPATH": None}


def test_com_thread_reraises_job_errors(fake_ti2, clear_estop):
    sdk, m = fake_ti2
    t = sdk.NISSdk()._thread
    with pytest.raises(ZeroDivisionError):
        t.call(lambda mic: 1 / 0)
    assert t.call(lambda mic: 5) == 5                      # thread survives
