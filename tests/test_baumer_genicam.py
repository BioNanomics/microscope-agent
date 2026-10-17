import sys

import pytest
from PIL import Image

from tests.fakes_camera import install_fake_camera_stack


@pytest.fixture(autouse=True)
def cleanup_module():
    yield
    sys.modules.pop("acquisition.backends.baumer_genicam", None)


def test_find_cti_files_lists_only_producers(monkeypatch, tmp_path):
    mod, _, _ = install_fake_camera_stack(monkeypatch, tmp_path)
    found = mod.find_cti_files()
    assert len(found) == 1 and found[0].lower().endswith(".cti")
    monkeypatch.setenv("GENICAM_GENTL64_PATH", str(tmp_path / "does-not-exist"))
    assert mod.find_cti_files() == []


def test_no_producer_is_a_clear_error(monkeypatch, tmp_path):
    mod, _, _ = install_fake_camera_stack(monkeypatch, tmp_path)
    monkeypatch.setenv("GENICAM_GENTL64_PATH", "")
    with pytest.raises(RuntimeError, match="No GenTL producer"):
        mod.BaumerGenICam()


def test_no_camera_found_resets_harvester(monkeypatch, tmp_path):
    mod, _, log = install_fake_camera_stack(monkeypatch, tmp_path, devices=0)
    with pytest.raises(RuntimeError, match="no camera was found"):
        mod.BaumerGenICam()
    assert log["reset"] is True


def test_connect_starts_acquisition_and_close_stops_it(monkeypatch, tmp_path):
    mod, acq, log = install_fake_camera_stack(monkeypatch, tmp_path)
    cam = mod.BaumerGenICam()
    assert acq.started and len(log["files"]) == 1
    cam.close()
    assert acq.stopped and log["reset"]


def test_get_settings(monkeypatch, tmp_path):
    mod, _, _ = install_fake_camera_stack(monkeypatch, tmp_path)
    s = mod.BaumerGenICam().get_settings()
    assert s["exposure_time_us"] == 10000.0 and s["exposure_time_range_us"] == [28.0, 60_000_000.0]
    assert s["gain_range"] == [1.0, 251.0] and s["pixel_format"] == "Mono8"
    assert s["balance_white_auto"] == "Off"


def test_get_settings_without_white_balance(monkeypatch, tmp_path):
    mod, _, _ = install_fake_camera_stack(monkeypatch, tmp_path, with_white_balance=False)
    assert mod.BaumerGenICam().get_settings()["balance_white_auto"] is None


def test_set_settings_turns_autos_off_and_writes(monkeypatch, tmp_path):
    mod, acq, _ = install_fake_camera_stack(monkeypatch, tmp_path)
    nm = acq.remote_device.node_map
    out = mod.BaumerGenICam().set_settings(exposure_time_us=1600, gain=2.0)
    assert nm.ExposureTime.value == 1600 and nm.Gain.value == 2.0
    assert nm.ExposureAuto.value == "Off" and nm.GainAuto.value == "Off"
    assert out["exposure_time_us"] == 1600


def test_set_settings_touches_only_what_was_passed(monkeypatch, tmp_path):
    mod, acq, _ = install_fake_camera_stack(monkeypatch, tmp_path)
    nm = acq.remote_device.node_map
    mod.BaumerGenICam().set_settings(gain=3.0)
    assert nm.Gain.value == 3.0 and nm.ExposureTime.value == 10000.0
    assert nm.ExposureAuto.value == "Continuous"


def test_set_settings_without_auto_nodes(monkeypatch, tmp_path):
    mod, acq, _ = install_fake_camera_stack(monkeypatch, tmp_path, with_auto=False)
    mod.BaumerGenICam().set_settings(exposure_time_us=500, gain=2.0)
    assert acq.remote_device.node_map.ExposureTime.value == 500


def test_set_settings_never_clamps(monkeypatch, tmp_path):
    mod, acq, _ = install_fake_camera_stack(monkeypatch, tmp_path)
    cam = mod.BaumerGenICam()
    with pytest.raises(ValueError, match="exposure"):
        cam.set_settings(exposure_time_us=1)
    with pytest.raises(ValueError, match="gain"):
        cam.set_settings(gain=999)
    assert acq.remote_device.node_map.ExposureTime.value == 10000.0


@pytest.mark.parametrize("fmt", ["Mono8", "RGB8", "BGR8", "BayerRG8"])
def test_capture_saves_an_rgb_png_for_each_pixel_format(monkeypatch, tmp_path, fmt):
    mod, acq, _ = install_fake_camera_stack(monkeypatch, tmp_path, pixel_format=fmt)
    cam = mod.BaumerGenICam()
    first, second = cam.capture(), cam.capture()
    assert first != second and first.suffix == ".png"
    img = Image.open(first)
    assert img.mode == "RGB" and img.size == (acq.w, acq.h)


def test_capture_drains_stale_frames_first(monkeypatch, tmp_path):
    mod, acq, _ = install_fake_camera_stack(monkeypatch, tmp_path)
    mod.BaumerGenICam().capture()
    assert acq._stale == 0


def test_capture_rejects_unknown_pixel_format(monkeypatch, tmp_path):
    mod, _, _ = install_fake_camera_stack(monkeypatch, tmp_path, pixel_format="Weird16")
    with pytest.raises(RuntimeError, match="Unsupported camera PixelFormat"):
        mod.BaumerGenICam().capture()
