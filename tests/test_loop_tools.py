import sys
import types

import pytest
from PIL import Image

from acquisition import estop
from acquisition.backends import nis_mock
from acquisition.orchestration import stage_positions
from mcp_server import loop_tools as lt
from tests.fakes import fake_ti2  # noqa: F401  (fixture)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(lt, "MOVE_HISTORY_PATH", tmp_path / "logs" / "move_history.jsonl")
    monkeypatch.setattr(lt, "FRAME_HISTORY_PATH", tmp_path / "logs" / "frame_history.jsonl")
    monkeypatch.setattr(lt, "_stage_revision", 0)
    monkeypatch.setattr(lt, "_last_known_position", None)
    monkeypatch.setattr(lt, "_frame_counter", 0)
    monkeypatch.setattr(lt, "_camera", None)
    monkeypatch.setattr(estop, "ESTOP_PATH", tmp_path / "ESTOP")
    monkeypatch.setattr(nis_mock.time, "sleep", lambda s: None)
    monkeypatch.setattr(nis_mock, "CAPTURE_DIR", tmp_path / "caps")
    monkeypatch.setattr(nis_mock, "SAMPLE_FRAME_PATH", tmp_path / "none.png")
    monkeypatch.setattr(stage_positions, "nis", nis_mock.MockNIS(10, 20, 30))


def test_unknown_backend_everywhere():
    with pytest.raises(ValueError):
        lt._get_backend("x")
    with pytest.raises(ValueError):
        lt._require_confirm_for_sdk("x", True)
    for call in (lambda: lt.get_pos("x"), lambda: lt.move(0, 0, "x"), lambda: lt.move_focus(0, "x")):
        with pytest.raises(ValueError):
            call()


def test_get_pos_mock_and_revision():
    a = lt.get_pos()
    b = lt.get_pos()
    assert a["position"] == {"x": 10.0, "y": 20.0, "z": 30.0} and a["backend"] == "mock"
    assert a["stage_revision"] == b["stage_revision"] == 1            # unchanged: same revision
    lt.move(11, 20)
    assert lt.get_pos()["stage_revision"] == 2


def test_move_returns_actual_position_and_logs_history():
    r = lt.move(110, 120)
    assert r["position"] == {"x": 110.0, "y": 120.0, "z": 30.0}
    h = lt.get_move_history()
    assert h["total_moves"] == 1 and h["history"][0]["requested"] == {"x": 110.0, "y": 120.0}


def test_move_history_limit_and_empty():
    assert lt.get_move_history() == {"history": [], "returned": 0, "total_moves": 0}
    for i in range(5):
        lt.move(10 + i, 20)
    h = lt.get_move_history(limit=2)
    assert h["returned"] == 2 and h["total_moves"] == 5
    assert [r["requested"]["x"] for r in h["history"]] == [13.0, 14.0]


def test_move_focus_mock_and_step_cap():
    r = lt.move_focus(40)
    assert r["position"]["z"] == 40.0
    assert lt.get_move_history()["history"][0]["requested"] == {"z": 40.0}
    with pytest.raises(ValueError, match="per-call safety limit"):
        lt.move_focus(40 + 51)


def test_sdk_backend_needs_confirm_for_moves(fake_ti2):
    with pytest.raises(PermissionError):
        lt.move(0, 0, backend="sdk")
    with pytest.raises(PermissionError):
        lt.move_focus(0, backend="sdk")


def test_sdk_backend_moves_through_the_fake_ti2(fake_ti2):
    sdk, m = fake_ti2
    assert lt.get_pos("sdk")["position"] == {"x": 100.0, "y": 200.0, "z": 4619.0}
    assert lt.move(150, 250, backend="sdk", confirm=True)["position"]["x"] == 150.0
    assert lt.move_focus(4629, backend="sdk", confirm=True)["position"]["z"] == 4629.0


def test_sdk_move_refused_by_estop(fake_ti2):
    sdk, m = fake_ti2
    estop.engage("test")
    with pytest.raises(estop.EStopEngaged):
        lt.move(150, 250, backend="sdk", confirm=True)
    assert m.iXPOSITION == 1000


def test_validate_crop():
    lt._validate_crop({"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5})
    for bad in ({"x": 0}, {"x": -0.1, "y": 0, "width": 1, "height": 1},
                {"x": 0, "y": 0, "width": "a", "height": 1},
                {"x": 0.6, "y": 0, "width": 0.5, "height": 1},
                {"x": 0, "y": 0.6, "width": 1, "height": 0.5}):
        with pytest.raises(ValueError):
            lt._validate_crop(bad)


def test_preview_crop_and_downscale(tmp_path):
    p = tmp_path / "big.png"
    Image.new("RGB", (400, 200), (9, 9, 9)).save(p)
    whole = lt._make_preview_image(p, max_dimension=100)
    cropped = lt._make_preview_image(p, crop={"x": 0, "y": 0, "width": 0.5, "height": 1.0}, max_dimension=100)
    import io
    assert Image.open(io.BytesIO(whole.data)).size == (100, 50)
    assert Image.open(io.BytesIO(cropped.data)).size == (100, 100)


def test_get_image_mock_records_frame_history_and_crop():
    meta, preview = lt.get_image(backend="mock", crop={"x": 0, "y": 0, "width": 0.5, "height": 0.5})
    assert meta["frame_id"] == 1 and meta["crop"]["width"] == 0.5
    assert lt.get_frame(1)["image"] == meta["image"]
    meta2, _ = lt.get_image(backend="mock")
    assert meta2["frame_id"] == 2


def test_get_frame_unknown_id_and_blank_lines():
    with pytest.raises(KeyError):
        lt.get_frame(99)                                           # no history file yet
    lt.get_image(backend="mock")
    with open(lt.FRAME_HISTORY_PATH, "a") as f:
        f.write("\n")
    with pytest.raises(KeyError):
        lt.get_frame(99)


def test_get_image_sdk_uses_the_camera_and_applies_settings(fake_ti2, tmp_path, monkeypatch):
    img_path = tmp_path / "cam.png"
    Image.new("RGB", (8, 8)).save(img_path)
    seen = {}

    class Cam:
        def set_settings(self, **kw):
            seen["settings"] = kw

        def capture(self):
            return img_path

    made = []
    mod = types.ModuleType("acquisition.backends.baumer_genicam")
    mod.BaumerGenICam = lambda: (made.append(1), Cam())[1]
    monkeypatch.setitem(sys.modules, "acquisition.backends.baumer_genicam", mod)
    meta, _ = lt.get_image(confirm=True, exposure_time_us=1600, gain=2.0)
    assert seen["settings"] == {"exposure_time_us": 1600, "gain": 2.0}
    assert meta["backend"] == "sdk" and meta["image"] == str(img_path)
    lt.get_image(confirm=True)
    assert made == [1]                                            # the camera connection is reused


def test_get_image_sdk_without_settings_leaves_camera_alone(fake_ti2, tmp_path, monkeypatch):
    img_path = tmp_path / "cam.png"
    Image.new("RGB", (8, 8)).save(img_path)
    cam = types.SimpleNamespace(capture=lambda: img_path,
                                set_settings=lambda **k: pytest.fail("settings must not be touched"))
    monkeypatch.setattr(lt, "_camera", cam)
    lt.get_image(confirm=True)


def test_estop_tool_engage_status_and_unknown():
    assert lt.estop("status")["engaged"] is False
    out = lt.estop("engage", reason="test")
    assert out["engaged"] is True and out["detail"]["reason"] == "test"
    s = lt.estop()
    assert s["engaged"] is True and s["flag_file"]
    with pytest.raises(ValueError, match="Releasing is a human action"):
        lt.estop("release")
