import os
from pathlib import Path

import pytest

from acquisition.backends import nis_mock
from mcp_server import loop_tools


def test_get_image_sdk_default_requires_confirm():
    with pytest.raises(PermissionError):
        loop_tools.get_image()


def test_get_image_mock_returns_metadata_and_preview(write_frame, monkeypatch):
    frame = write_frame("sample.png")
    monkeypatch.setattr(nis_mock, "SAMPLE_FRAME_PATH", frame)
    metadata, preview = loop_tools.get_image(backend="mock", max_dimension=64)
    assert metadata["backend"] == "mock"
    assert Path(metadata["image"]).exists()
    assert Path(metadata["image"]).is_relative_to(Path(os.environ["CONFOCAL_MCP_DATA_DIR"]))
    assert set(metadata["position"]) == {"x", "y", "z"}
    assert metadata["frame_id"] >= 1
    assert preview.format == "jpeg"
    assert loop_tools.get_frame(metadata["frame_id"])["image"] == metadata["image"]


def test_get_image_mock_rejects_bad_crop():
    with pytest.raises(ValueError):
        loop_tools.get_image(backend="mock", crop={"x": 0.8, "y": 0.0, "width": 0.5, "height": 0.5})


def test_get_image_unknown_backend():
    with pytest.raises(ValueError):
        loop_tools.get_image(backend="nope")
