# Runtime data (captures, history logs) is anchored to CONFOCAL_MCP_DATA_DIR
# at *import* time (acquisition/paths.py), so it must be set before any
# test module imports mcp_server.loop_tools or acquisition.backends.nis_mock.
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_DATA_DIR = tempfile.mkdtemp(prefix="confocal-mcp-tests-")
os.environ["CONFOCAL_MCP_DATA_DIR"] = _DATA_DIR


def make_blob_frame(
    size: int = 128,
    center: tuple[float, float] = (64, 64),
    radius: float = 20.0,
    background: float = 200.0,
    foreground: float = 60.0,
    noise: float = 3.0,
    seed: int = 0,
) -> np.ndarray:
    """Brightfield-like frame: a dark disc on a bright background, plus
    Gaussian noise. uint8 array of shape (size, size)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
    frame = np.full((size, size), background, dtype=np.float32)
    frame[disc] = foreground
    frame += rng.normal(0.0, noise, frame.shape).astype(np.float32)
    return np.clip(frame, 0, 255).astype(np.uint8)


def save_frame(array: np.ndarray, path: Path) -> Path:
    Image.fromarray(array, mode="L").convert("RGB").save(path)
    return path


@pytest.fixture
def blob_frame():
    return make_blob_frame


@pytest.fixture
def write_frame(tmp_path):
    def _write(name: str, **kwargs) -> Path:
        return save_frame(make_blob_frame(**kwargs), tmp_path / name)
    return _write
