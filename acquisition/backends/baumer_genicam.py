# baumer_genicam.py
# ------------------------------------------------------------
# Real image capture from the Baumer VCXU-23C camera via GenICam/GenTL,
# confirmed working end-to-end 2026-08-10 using the `harvesters` Python
# library against the camera's own bgapi2_usb.cti GenTL producer
# (installed with Baumer Camera Explorer).
#
# NOT the confocal (N-SPARC) detector - the Baumer VCXU-23C is a separate
# industrial color camera found already installed on the microscope PC
# (Baumer Camera Explorer + Toshiba Teli GenICam SDK). Originally added
# just to exercise the rest of the pipeline (focus_check, dashboard)
# against real frames while N-SPARC capture was blocked on NIS-Elements
# licensing/Jobs work (see docs/microscope-notes.md's "Image Capture"
# investigation for that history).
#
# 2026-08-17: now the PRIMARY capture path by deliberate decision - the
# team decided against capturing through NIS-Elements/Jobs at all (see
# mcp_server/loop_tools.py's header comment), so this module's capture()
# is what mcp_server.loop_tools.get_image() actually calls. Whatever is
# optically coupled to wherever this camera is physically mounted is
# what gets captured - that's a hardware/alignment decision made outside
# this code, not something this module controls.
#
# Requires `pip install harvesters opencv-python` (confirmed against
# harvesters==1.4.3, genicam==1.5.1; opencv-python added 2026-08-17,
# Bayer demosaicing only - see capture() below). A camera can only be
# held by one application at a time - close Baumer Camera Explorer (or
# any other GenICam consumer) before using this.
# ------------------------------------------------------------

import os
from datetime import datetime
from pathlib import Path

import cv2
from harvesters.core import Harvester
from PIL import Image

# Same directory nis_mock.MockNIS.capture() writes to, so frames from
# either backend land in one place.
CAPTURE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "captures"

# GenTL producer folders confirmed present on the microscope PC,
# 2026-08-10. Checked in addition to GENICAM_GENTL64_PATH (not instead
# of) because a machine-level env var change doesn't propagate to
# already-running processes/shells until they're relaunched - confirmed
# hitting this exact issue while testing, so the env var alone isn't
# reliable here.
FALLBACK_CTI_DIRS = [
    r"C:\Program Files\Baumer Camera Explorer",
    r"C:\Program Files\TOSHIBA TELI\TeliCamSDK\TeliCamApi\bin\x64",
]


def find_cti_files() -> list[str]:
    """Return every .cti GenTL producer file findable via GENICAM_GENTL64_PATH
    plus FALLBACK_CTI_DIRS (see module comment re: stale env vars)."""
    env_dirs = [d for d in os.environ.get("GENICAM_GENTL64_PATH", "").split(os.pathsep) if d]
    dirs = list(dict.fromkeys(env_dirs + FALLBACK_CTI_DIRS))
    cti_files = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if name.lower().endswith(".cti"):
                cti_files.append(os.path.join(d, name))
    return cti_files


class BaumerGenICam:
    """Real capture() from a GenICam camera (confirmed against the Baumer
    VCXU-23C over USB3), matching nis_mock.MockNIS.capture()'s shape:
    writes a timestamped PNG to CAPTURE_DIR and returns its Path.

    Only implements capture() - no stage control. Pair with
    nis_sdk.NISSdk for XY/Z if a caller needs both.
    """

    def __init__(self):
        cti_files = find_cti_files()
        if not cti_files:
            raise RuntimeError(
                "No GenTL producer (.cti) files found - check "
                "GENICAM_GENTL64_PATH or FALLBACK_CTI_DIRS in this module."
            )
        self._harvester = Harvester()
        for cti in cti_files:
            self._harvester.add_file(cti)
        self._harvester.update()
        if not self._harvester.device_info_list:
            self._harvester.reset()
            raise RuntimeError(
                f"GenTL producer(s) loaded ({len(cti_files)}) but no camera "
                "was found - check it's connected and powered, and not "
                "already open in another application (e.g. Baumer Camera "
                "Explorer can only be open in one place at a time)."
            )
        self._acquirer = self._harvester.create(0)
        self._acquirer.start()
        self._capture_count = 0

    def get_settings(self) -> dict:
        """Return the camera's current exposure/gain/pixel format, plus
        each numeric setting's valid range, read live from the GenICam
        node map - so a caller can see what a capture will actually use
        without guessing.

        Confirmed live 2026-08-17 against the VCXU-23C: ExposureTime is
        in microseconds (range 28 to 60,000,000); Gain's range (1.0 to
        251.1875) doesn't look dB-shaped, so it's reported here as a
        plain, unit-less "gain" factor rather than assumed to be dB -
        treat it as the camera's own arbitrary scale, not a physical
        unit. Both ExposureAuto/GainAuto were already "Off" (manual
        control) by default on this camera.
        """
        node_map = self._acquirer.remote_device.node_map
        exposure = node_map.ExposureTime
        gain = node_map.Gain
        return {
            "exposure_time_us": exposure.value,
            "exposure_time_range_us": [exposure.min, exposure.max],
            "gain": gain.value,
            "gain_range": [gain.min, gain.max],
            "pixel_format": node_map.PixelFormat.value,
        }

    def set_settings(self, exposure_time_us: float | None = None, gain: float | None = None) -> dict:
        """Set exposure time (microseconds) and/or gain via the GenICam
        node map. Only touches whichever parameter is actually passed -
        omit one to leave it unchanged. Forces ExposureAuto/GainAuto to
        "Off" first if the camera has them and a matching manual value
        was requested, since a manual write would otherwise be silently
        overridden by auto-exposure/auto-gain on the next frame.

        Raises ValueError if a requested value is outside the camera's
        own reported valid range (same pattern as nis_sdk.py's stage
        moves) - values are never silently clamped.

        Takes effect on the next capture() call - this only writes the
        node map, it doesn't grab a frame itself.
        """
        node_map = self._acquirer.remote_device.node_map

        if exposure_time_us is not None:
            if hasattr(node_map, "ExposureAuto"):
                node_map.ExposureAuto.value = "Off"
            exposure = node_map.ExposureTime
            if not (exposure.min <= exposure_time_us <= exposure.max):
                raise ValueError(
                    f"Requested exposure {exposure_time_us:.1f} us is outside "
                    f"the camera's reported range ({exposure.min:.1f} to {exposure.max:.1f} us)."
                )
            exposure.value = exposure_time_us

        if gain is not None:
            if hasattr(node_map, "GainAuto"):
                node_map.GainAuto.value = "Off"
            gain_node = node_map.Gain
            if not (gain_node.min <= gain <= gain_node.max):
                raise ValueError(
                    f"Requested gain {gain:.2f} is outside the camera's "
                    f"reported range ({gain_node.min:.2f} to {gain_node.max:.2f})."
                )
            gain_node.value = gain

        return self.get_settings()

    def capture(self) -> Path:
        """Grab one live frame, demosaic it to RGB, and save it as a
        timestamped PNG in CAPTURE_DIR, same filename convention as
        MockNIS.capture()."""
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        self._capture_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = CAPTURE_DIR / f"capture_{timestamp}_{self._capture_count:04d}.png"

        # Drain any backlog of already-queued frames before grabbing the
        # real one. The acquirer streams continuously for this object's
        # whole lifetime (started once in __init__, never stopped between
        # capture() calls, for connection-reuse performance - see the
        # class docstring) - if nothing calls fetch() for a while, filled
        # buffers just pile up in the completion queue instead of being
        # replaced by newer frames. A later fetch() then returns whatever
        # stale frame is at the front of that backlog, not a fresh one -
        # confirmed 2026-08-21: a real ~27mm stage move and several
        # minutes apart still returned a visibly identical old frame
        # until this drain was added. A short per-fetch timeout here
        # means "give up as soon as the backlog is empty", not "wait for
        # a new frame" - once it raises/times out, the queue is caught up
        # to live and the real fetch below gets a genuinely current one.
        while True:
            try:
                with self._acquirer.fetch(timeout=0.01):
                    pass
            except Exception:
                break

        with self._acquirer.fetch(timeout=5) as buffer:
            component = buffer.payload.components[0]
            mosaic = component.data.reshape(component.height, component.width)
            # VCXU-23C reports BayerRG8 (confirmed 2026-08-10). cv2's
            # Bayer color codes are named for OpenCV's own row/col
            # convention, which doesn't always match a camera's declared
            # GenICam PixelFormat 1:1 - COLOR_BayerRG2RGB is the literal-
            # name match used here, but NOT yet independently verified
            # against a known-color reference target. If captured colors
            # come out wrong/channel-swapped once there's real light on
            # the sensor, try COLOR_BayerGR2RGB / COLOR_BayerBG2RGB /
            # COLOR_BayerGB2RGB instead - re-confirm before trusting this
            # for actual analysis.
            rgb = cv2.cvtColor(mosaic, cv2.COLOR_BayerRG2RGB)
            Image.fromarray(rgb, mode="RGB").save(dest)

        return dest

    def close(self) -> None:
        self._acquirer.stop()
        self._acquirer.destroy()
        self._harvester.reset()


if __name__ == "__main__":
    cam = BaumerGenICam()
    try:
        print("Connected. Capturing one frame...")
        path = cam.capture()
        print(f"Saved: {path}")
    finally:
        cam.close()
