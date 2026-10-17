# A stand-in for the `nd2` reader and a synthetic worm scene, so the analysis
# scripts run without ND2 files: brightfield (TD) with a dark worm on agar,
# and a fluorescence channel (RFP) with glowing neurons at the worm's head.
import sys
import types

import cv2
import numpy as np

UM_PX = 3.0


def worm_td(size=256, p0=(60, 128), p1=(200, 140), width=10, value=70.0, seed=0, blur=0.0):
    rng = np.random.default_rng(seed)
    img = 190.0 + rng.normal(0, 1.5, (size, size)).astype(np.float32)
    cv2.line(img, p0, p1, float(value), width, cv2.LINE_AA)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    return img.astype(np.float32)


def neurons(size=256, centres=((62, 128), (70, 129), (78, 129)), amp=400.0, sigma=2.5):
    yy, xx = np.mgrid[0:size, 0:size]
    out = np.zeros((size, size), np.float32)
    for cx, cy in centres:
        out += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return out


class FakeND2File:
    config = {}

    def __init__(self, path):
        self.path = path
        c = FakeND2File.config
        self._arr, self._dims = c["array"], c["dims"]
        self.sizes = dict(zip(c["dims"], c["array"].shape))
        self.sizes.update(c.get("extra_sizes", {}))
        self.metadata = types.SimpleNamespace(channels=[
            types.SimpleNamespace(channel=types.SimpleNamespace(name=n)) for n in c["channels"]])
        self._um_px, self._times = c.get("um_px", UM_PX), c.get("times")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def asarray(self):
        return self._arr

    def voxel_size(self):
        return types.SimpleNamespace(x=self._um_px, y=self._um_px, z=1.0)

    def events(self):
        if self._times is None:
            raise RuntimeError("no events in this file")
        return [{"Time [s]": t} for t in self._times]


def install_nd2(monkeypatch, array, dims, channels, **config):
    FakeND2File.config = dict(array=array, dims=dims, channels=channels, **config)
    mod = types.ModuleType("nd2")
    mod.ND2File = FakeND2File
    monkeypatch.setitem(sys.modules, "nd2", mod)
    return mod


def two_channel_stack(t=2, z=3, size=256, **kw):
    """T, Z, C, Y, X: brightfield sharpest at the middle plane, neurons at the head."""
    arr = np.zeros((t, z, 2, size, size), np.float32)
    for ti in range(t):
        for zi in range(z):
            arr[ti, zi, 0] = worm_td(size, seed=ti, blur=0.0 if zi == z // 2 else 4.0, **kw)
            arr[ti, zi, 1] = neurons(size) if zi == z // 2 else 0.3 * neurons(size)
    return arr
