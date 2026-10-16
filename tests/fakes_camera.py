# Fake Baumer / GenICam stack: cv2 and harvesters stand-ins plus a .cti
# folder, so acquisition.backends.baumer_genicam imports and runs on a plain
# CI runner with no camera and no opencv.
import importlib
import sys
import types


class _Node:
    def __init__(self, value, lo=None, hi=None):
        self.value, self.min, self.max = value, lo, hi


class FakeNodeMap:
    def __init__(self, pixel_format="Mono8", with_auto=True, with_white_balance=True):
        self.ExposureTime = _Node(10000.0, 28.0, 60_000_000.0)
        self.Gain = _Node(1.0, 1.0, 251.0)
        self.PixelFormat = _Node(pixel_format)
        if with_auto:
            self.ExposureAuto = _Node("Continuous")
            self.GainAuto = _Node("Continuous")
        if with_white_balance:
            self.BalanceWhiteAuto = _Node("Off")


class _Buffer:
    def __init__(self, data, h, w):
        comp = types.SimpleNamespace(height=h, width=w, data=data)
        self.payload = types.SimpleNamespace(components=[comp])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeAcquirer:
    def __init__(self, node_map, h=4, w=6):
        import numpy as np
        self.remote_device = types.SimpleNamespace(node_map=node_map)
        self.h, self.w, self.started, self.stopped = h, w, False, False
        self._np = np
        self._stale = 2                       # frames queued before the capture: must be drained

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def destroy(self):
        self.destroyed = True

    def fetch(self, timeout=0):
        if timeout == 0.01:
            if self._stale <= 0:
                raise TimeoutError("no more frames")
            self._stale -= 1
            return _Buffer(self._np.zeros(1, dtype="uint8"), 1, 1)
        fmt = self.remote_device.node_map.PixelFormat.value
        n = self.h * self.w * (3 if fmt in ("BGR8", "RGB8") else 1)
        return _Buffer(self._np.arange(n, dtype="uint8"), self.h, self.w)


def install_fake_camera_stack(monkeypatch, tmp_path, pixel_format="Mono8", devices=1, **node_kw):
    """Fake cv2, harvesters and a .cti folder, then import a fresh
    baumer_genicam. Returns (module, acquirer, harvester_log)."""
    import numpy as np
    cv2 = types.ModuleType("cv2")
    cv2.COLOR_BGR2RGB, cv2.COLOR_GRAY2RGB, cv2.COLOR_BayerBG2RGB = 1, 3, 4

    def cvt(a, code):
        if code == cv2.COLOR_BGR2RGB:
            return a[..., ::-1].copy()
        return np.stack([a, a, a], axis=-1)

    cv2.cvtColor = cvt
    acquirer = FakeAcquirer(FakeNodeMap(pixel_format, **node_kw))
    log = {"files": [], "reset": False}

    class Harvester:
        device_info_list = [object()] * devices

        def add_file(self, f):
            log["files"].append(f)

        def update(self):
            pass

        def create(self, i):
            return acquirer

        def reset(self):
            log["reset"] = True

    harv_core = types.ModuleType("harvesters.core")
    harv_core.Harvester = Harvester
    harv = types.ModuleType("harvesters")
    harv.core = harv_core
    for name, mod in (("cv2", cv2), ("harvesters", harv), ("harvesters.core", harv_core)):
        monkeypatch.setitem(sys.modules, name, mod)
    cti_dir = tmp_path / "gentl"
    cti_dir.mkdir(exist_ok=True)
    (cti_dir / "producer.CTI").write_text("")
    (cti_dir / "readme.txt").write_text("")
    monkeypatch.setenv("GENICAM_GENTL64_PATH", str(cti_dir))
    sys.modules.pop("acquisition.backends.baumer_genicam", None)
    mod = importlib.import_module("acquisition.backends.baumer_genicam")
    monkeypatch.setattr(mod, "CAPTURE_DIR", tmp_path / "caps")
    monkeypatch.setattr(mod, "FALLBACK_CTI_DIRS", [])
    return mod, acquirer, log
