import pytest

from acquisition.backends import nis_mock
from acquisition.backends.nis_mock import MockNIS


@pytest.fixture(autouse=True)
def fast(monkeypatch, tmp_path):
    monkeypatch.setattr(nis_mock.time, "sleep", lambda s: None)
    monkeypatch.setattr(nis_mock, "CAPTURE_DIR", tmp_path / "caps")
    monkeypatch.setattr(nis_mock, "SAMPLE_FRAME_PATH", tmp_path / "missing.png")


def test_start_position_and_absolute_moves():
    n = MockNIS(1, 2, 3)
    assert n.XY_GetPosition() == (1.0, 2.0) and n.Z_GetPosition() == 3.0
    n.XY_Move(1000, -500)
    n.Z_Move(25)
    assert n.XY_GetPosition() == (1000.0, -500.0) and n.Z_GetPosition() == 25.0


def test_relative_moves():
    n = MockNIS()
    n.XY_MoveRelative(250, 250)
    n.Z_MoveRelative(-0.0)
    n.Z_MoveRelative(5)
    assert n.XY_GetPosition() == (250.0, 250.0) and n.Z_GetPosition() == 5.0


@pytest.mark.parametrize("x,y", [(60_000, 0), (-60_000, 0), (0, 40_000), (0, -40_000)])
def test_xy_outside_travel_is_refused(x, y):
    n = MockNIS(x=0 if abs(x) > 57_000 else x, y=0)
    with pytest.raises(ValueError, match="outside the stage travel limit"):
        n.XY_Move(x, y)
    assert n.XY_GetPosition() == (0.0 if abs(x) > 57_000 else float(x), 0.0)


def test_xy_step_cap():
    n = MockNIS()
    with pytest.raises(ValueError, match="per-call safety limit"):
        n.XY_Move(5001, 0)
    assert n.XY_GetPosition() == (0.0, 0.0)


def test_z_limits_and_step_cap():
    n = MockNIS(z=5)
    with pytest.raises(ValueError, match="outside the stage travel limit"):
        n.Z_Move(-1)
    with pytest.raises(ValueError, match="outside the stage travel limit"):
        n.Z_Move(10_001)
    with pytest.raises(ValueError, match="per-call safety limit"):
        n.Z_Move(5 + 51)
    assert n.Z_GetPosition() == 5.0


def test_abort_is_never_requested():
    n = MockNIS()
    assert n.shouldAbort() is False and n.ctx.shouldAbort() is False


def test_capture_makes_unique_files_with_placeholder():
    n = MockNIS()
    a, b = n.capture(), n.capture()
    assert a != b and a.exists() and b.exists()
    assert a.suffix == ".png" and a.name.endswith("_0001.png") and b.name.endswith("_0002.png")


def test_capture_copies_the_sample_frame_when_present(tmp_path, monkeypatch):
    from PIL import Image
    sample = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(sample)
    monkeypatch.setattr(nis_mock, "SAMPLE_FRAME_PATH", sample)
    out = MockNIS().capture()
    assert Image.open(out).size == (8, 8)
