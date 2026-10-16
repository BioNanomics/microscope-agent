# A simulated microscope for mosaic.py and worm_follow.py: a textured agar
# "world" with one dark worm on it, a stage that moves the camera's window
# across it (stage +X slides features left, +Y moves them down - the same
# signs as worm_follow.M_4X), and a camera that renders the window with
# exposure-dependent brightness and defocus blur. Real OpenCV, no hardware.
import sys
import types
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from acquisition import estop

UM_PER_PX = 1.0 / 0.6717            # what worm_follow.M_4X implies
WORLD = 2400
Z_FOCUS = 4619.0


class _Node:
    def __init__(self, value="Off", symbolics=None):
        self.value, self.symbolics = value, symbolics


class RigNodeMap:
    """The GenICam nodes force_camera_autos_off and mosaic.run touch."""

    def __init__(self, white_balance="Off", full=True):
        self.BalanceWhiteAuto = _Node(white_balance)
        self.ColorTransformationAuto = _Node("Once")
        self.ExposureAuto = _Node("Off")
        self.GainAuto = _Node("Continuous")
        if full:
            self.Gamma = _Node(0.5)
            self.ColorTransformationValueSelector = _Node("Gain00", ["Gain00", "Gain01", "Gain10", "Gain11"])
            self.ColorTransformationValue = _Node(0.3)
            self.GainSelector = _Node("All")
            self.Gain = _Node(1.0)


class RigStage:
    def __init__(self, rig):
        self.rig = rig
        self.x = self.y = 0.0
        self.z = Z_FOCUS
        self.moves, self.z_moves, self.frozen = [], [], False
        self.micro = types.SimpleNamespace(iNOSEPIECE=2)
        self._thread = types.SimpleNamespace(call=lambda fn: fn(self.micro))

    def XY_GetPosition(self):
        return self.x, self.y

    def XY_Move(self, x, y):
        estop.check()
        if np.hypot(x - self.x, y - self.y) > 5000:
            raise ValueError("per-call safety limit")
        self.moves.append((x, y))
        if not self.frozen:
            self.x, self.y = float(x), float(y)

    def Z_GetPosition(self):
        return self.z

    def Z_Move(self, z):
        estop.check()
        if abs(z - self.z) > 50:
            raise ValueError("per-call safety limit")
        self.z_moves.append(z)
        self.z = float(z)


class RigCamera:
    def __init__(self, rig, w=320, h=240, white_balance="Off", full_nodes=True):
        self.rig, self.w, self.h = rig, w, h
        self.exposure_us, self.settings_calls, self.closed = 1600.0, [], False
        self.captures = 0
        self.on_capture = None
        self._acquirer = types.SimpleNamespace(
            remote_device=types.SimpleNamespace(node_map=RigNodeMap(white_balance, full_nodes)))

    def set_settings(self, exposure_time_us=None, gain=None):
        self.settings_calls.append((exposure_time_us, gain))
        if exposure_time_us is not None:
            self.exposure_us = exposure_time_us
        return self.get_settings()

    def get_settings(self):
        return {"exposure_time_us": self.exposure_us, "gain": 1.0}

    def frame(self) -> np.ndarray:
        st = self.rig.stage
        cx = WORLD / 2 + st.x / UM_PER_PX
        cy = WORLD / 2 - st.y / UM_PER_PX
        x0 = int(np.clip(round(cx - self.w / 2), 0, WORLD - self.w))
        y0 = int(np.clip(round(cy - self.h / 2), 0, WORLD - self.h))
        img = self.rig.world[y0:y0 + self.h, x0:x0 + self.w].copy()
        defocus = abs(st.z - Z_FOCUS)
        if defocus > 2:
            img = cv2.GaussianBlur(img, (0, 0), defocus / 8)
        img = img * (self.exposure_us / 1600.0)
        return np.clip(img, 0, 255).astype(np.uint8)

    def capture(self) -> Path:
        self.captures += 1
        if self.on_capture:
            self.on_capture(self.captures)
        folder = self.rig.tmp / "rig-captures"
        folder.mkdir(exist_ok=True)
        path = folder / f"c{self.captures:05d}.png"
        Image.fromarray(self.frame()).save(path)
        return path

    def close(self):
        self.closed = True


class SceneRig:
    def __init__(self, tmp_path, worm_um=(60.0, 30.0), worm=True, seed=1, **camera_kw):
        self.tmp = tmp_path
        rng = np.random.default_rng(seed)
        noise = cv2.GaussianBlur(rng.normal(0, 1, (WORLD, WORLD)).astype(np.float32), (0, 0), 2.0)
        agar = 190.0 + 6.0 * noise / noise.std()
        world = np.stack([agar] * 3, axis=-1).astype(np.float32)
        self.worm_um = worm_um
        if worm:
            wx = int(WORLD / 2 + worm_um[0] / UM_PER_PX)
            wy = int(WORLD / 2 - worm_um[1] / UM_PER_PX)
            cv2.ellipse(world, (wx, wy), (45, 11), 20, 0, 360, (45, 45, 45), -1)
        self.world = np.clip(world, 0, 255)
        self.stage = RigStage(self)
        self.camera = RigCamera(self, **camera_kw)

    def install(self, monkeypatch):
        """Make `from acquisition.backends.X import ...` hand out this rig."""
        cam = types.ModuleType("acquisition.backends.baumer_genicam")
        cam.BaumerGenICam = lambda: self.camera
        sdk = types.ModuleType("acquisition.backends.nis_sdk")
        sdk.NISSdk = lambda: self.stage
        monkeypatch.setitem(sys.modules, "acquisition.backends.baumer_genicam", cam)
        monkeypatch.setitem(sys.modules, "acquisition.backends.nis_sdk", sdk)
        return self
