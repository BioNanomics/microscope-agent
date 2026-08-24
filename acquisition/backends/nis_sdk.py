# nis_sdk.py
# ------------------------------------------------------------
# Ti2 ActiveX SDK backend for stage control - real hardware via
# win32com.client.Dispatch(NkTi2Ax.NikonTi2AxAutoConnectMicroscope.CLSID),
# the same connection pattern confirmed working in
# acquisition/calibration/nikon_connection_test.py against the Ti2-E Device Simulator.
#
# ConfocalOrchestrator has two stage-control backends, both exposing the
# same shape of interface so orchestration/stage_positions.py can swap
# between them via its `backend` parameter - see nis_mock.py for the
# other one ("mock").
# This is the "sdk" backend: direct ActiveX bindings, now that Nikon has
# approved SDK access (see docs/microscope-notes.md's "SDK Status") -
# confirmed end-to-end against the Ti2-E Device Simulator, 2026-07-27.
#
# CONFIRMED PROPERTIES (from .venv/Lib/site-packages/NkTi2Ax.py, the
# generated bindings for the SDK's own type library - the same file that
# defines iTURRET1POS/Turret1Pos, confirmed working in calibration/nikon_connection_test.py):
#   iXPOSITION / iYPOSITION / iZPOSITION - direct properties, readable and
#   writable, same shape as iTURRET1POS.
#   XPosition / YPosition / ZPosition - child settings objects (.Value/
#   .Lower/.Higher), same shape as Turret1Pos. Read-verified against the
#   Ti2-E Device Simulator via acquisition/calibration/nikon_stage_test.py -
#   both forms returned identical values.
#
# UNITS (inferred, not stated anywhere explicit - the bindings just
# declare a plain integer VARIANT, no unit metadata): cross-referencing

# the simulator's reported Lower/Higher travel limits against
# docs/microscope-notes.md's documented hardware spec ("Stroke X:
# +/-57mm, Y: +/-36.5mm ... Focusing: min increment 0.01um, 10mm stroke"):
#   X: Lower/Higher = +/-570000  -> 0.1um/count exactly reproduces +/-57mm
#   Z: Lower/Higher = 0..1000000 -> 0.01um/count exactly reproduces the
#      10mm stroke, and matches the doc's stated 0.01um min focus increment
#   Y: Lower/Higher = +/-375000  -> 0.1um/count gives +/-37.5mm, close to
#      but not exactly the documented +/-36.5mm - most likely the
#      simulator's configured soft limit isn't identical to the real
#      hardware's exact stroke, not a different unit (X and Z both match
#      their spec exactly at these scales). Re-confirm against the real
#      microscope if Y positions come out visibly wrong.
# So: X/Y properties are in units of 0.1um ("decimicrons"), Z is in units
# of 0.01um ("centimicrons"). XY_GetPosition/XY_Move/Z_GetPosition/Z_Move
# below convert to/from plain microns at their boundary so callers
# (StagePositionManager, run_protocol.py) never see raw counts.
# ------------------------------------------------------------

import queue
import threading

import pythoncom
import win32com.client
import NkTi2Ax

# Raw-count-per-micron scale factors confirmed above.
XY_COUNTS_PER_UM = 10.0
Z_COUNTS_PER_UM = 100.0

# Per-call safety caps (2026-08-10 incident: a hung MCP call turned out to
# have actually executed a ~7mm XY move against real hardware with no
# limit checking at all). These bound how far a SINGLE XY_Move/Z_Move call
# can travel, regardless of caller - a mistaken or hallucinated large
# target now fails loudly instead of silently reaching the hardware. Z is
# far tighter than XY: a big blind Z move risks crashing the objective
# into the sample, which XY moves don't. Deliberately not exposed as a
# parameter - raise these here, in code, if a real workflow needs bigger
# single-call steps, rather than letting a caller opt out per-call.
MAX_XY_STEP_UM = 5000.0
MAX_Z_STEP_UM = 50.0

# Direct scalar properties making up an "optical configuration" (objective,
# filters, light path, illumination) - confirmed present in NkTi2Ax.py's
# INikonTi2AxMicroscope interface, same source as the stage properties
# above. Used by get_optical_configuration()/apply_optical_configuration()
# below - see acquisition/orchestration/imaging_profile.py for the
# save/list/apply-by-experiment-name layer on top of these two methods.
OPTICAL_CONFIG_PROPERTIES = [
    "iNOSEPIECE",       # objective (turret slot 1-6)
    "iDIC_PRISM",
    "iDIC_POLARIZER",
    "iANALYZER_POS",
    "iANALYZER_SLOT",
    "iLIGHTPATH",
    "iCONDENSER",
    "iOPTZOOM",
    "iTURRET1POS", "iTURRET1SHUTTER",
    "iTURRET2POS", "iTURRET2SHUTTER",
    "iDLED1_POS", "iDLED1_SWITCH",
    "iDLED2_POS", "iDLED2_SWITCH",
    "iDLED3_POS", "iDLED3_SWITCH",
    "iDLED4_POS", "iDLED4_SWITCH",
]

# PFS offset units aren't calibrated to microns (unlike XY/Z - see the
# module docstring), so nudge_pfs_offset's cap is expressed as a fraction
# of the SDK's own reported valid range (PfsOffset.Lower/.Higher) rather
# than an assumed physical distance.
PFS_MAX_OFFSET_STEP_FRACTION = 0.02


def to_plain_float(value) -> float:
    """Convert a numpy scalar (or anything float-like) to a plain Python float.

    Matches the same convention used in calibration/nis_jobs_connection_test.py/orchestration/stage_positions.py -
    values passed to a COM property setter must be plain Python numbers,
    not numpy types.
    """
    return float(value)


class _ComThread(threading.Thread):
    """Owns the one and only COM connection to AutoConnectMicroscope.

    win32com objects are thread-affine (STA) - the MCP server dispatches
    each tool call via anyio.to_thread.run_sync, which runs it on whatever
    worker thread from its pool happens to be free. Creating a fresh
    Dispatch() on a different thread per call (the old behaviour here)
    meant multiple STA apartments independently racing for the same
    single-session hardware connection, with none of them pumping Windows
    messages - this is what caused calls to hang/deadlock instead of
    failing cleanly. Routing every call through one persistent thread that
    owns the connection for the server's whole lifetime avoids that.
    """

    def __init__(self):
        super().__init__(daemon=True, name="Ti2SdkComThread")
        self._jobs: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self.start()
        self._ready.wait()

    def run(self) -> None:
        pythoncom.CoInitialize()
        self._microscope = win32com.client.Dispatch(
            NkTi2Ax.NikonTi2AxAutoConnectMicroscope.CLSID
        )
        self._ready.set()
        while True:
            fn, result_queue = self._jobs.get()
            try:
                result_queue.put(("ok", fn(self._microscope)))
            except Exception as e:
                result_queue.put(("error", e))
            # Services any pending STA messages (e.g. completion callbacks
            # from the hardware) between jobs - without this, a later call
            # can hang waiting on a message this thread never processes.
            pythoncom.PumpWaitingMessages()

    def call(self, fn):
        """Run fn(microscope) on this thread and return its result."""
        result_queue: queue.Queue = queue.Queue()
        self._jobs.put((fn, result_queue))
        status, value = result_queue.get()
        if status == "error":
            raise value
        return value


_com_thread: "_ComThread | None" = None
_com_thread_lock = threading.Lock()


def _get_com_thread() -> _ComThread:
    global _com_thread
    with _com_thread_lock:
        if _com_thread is None:
            _com_thread = _ComThread()
        return _com_thread


class NISSdk:
    """Ti2 SDK (ActiveX) backend: real stage control via NkTi2Ax's
    iXPOSITION/iYPOSITION/iZPOSITION properties, matching the
    XY_GetPosition/XY_Move/Z_GetPosition/Z_Move shape used by MockNIS so
    StagePositionManager can use this backend interchangeably with 'mock'.

    Every instance shares the same underlying COM connection (see
    _ComThread above) - constructing NISSdk() repeatedly (once per MCP
    tool call, as acquisition_tools.py does) is cheap and does not open a
    new connection each time.
    """

    def __init__(self):
        self._thread = _get_com_thread()

    def XY_GetPosition(self) -> tuple[float, float]:
        """Return the current stage (x, y) position in microns."""
        x, y = self._thread.call(
            lambda m: (m.iXPOSITION / XY_COUNTS_PER_UM, m.iYPOSITION / XY_COUNTS_PER_UM)
        )
        return to_plain_float(x), to_plain_float(y)

    def XY_Move(self, x: float, y: float) -> None:
        """Move the stage to an absolute (x, y) position, in microns.

        Raises ValueError without moving anything if the step is larger
        than MAX_XY_STEP_UM, or if the target is outside the stage's own
        reported travel range (XPosition/YPosition .Lower/.Higher).
        """
        before = self.XY_GetPosition()
        x, y = to_plain_float(x), to_plain_float(y)
        step_um = ((x - before[0]) ** 2 + (y - before[1]) ** 2) ** 0.5
        if step_um > MAX_XY_STEP_UM:
            raise ValueError(
                f"Requested XY move of {step_um:.1f} um exceeds the "
                f"{MAX_XY_STEP_UM:.0f} um per-call safety limit. Break large "
                f"moves into smaller confirmed steps."
            )
        x_counts = round(x * XY_COUNTS_PER_UM)
        y_counts = round(y * XY_COUNTS_PER_UM)

        def move(m):
            x_lo, x_hi = m.XPosition.Lower, m.XPosition.Higher
            y_lo, y_hi = m.YPosition.Lower, m.YPosition.Higher
            if not (x_lo <= x_counts <= x_hi):
                raise ValueError(
                    f"X target {x:.2f} um is outside the stage's reported "
                    f"travel range ({x_lo / XY_COUNTS_PER_UM:.1f} to {x_hi / XY_COUNTS_PER_UM:.1f} um)."
                )
            if not (y_lo <= y_counts <= y_hi):
                raise ValueError(
                    f"Y target {y:.2f} um is outside the stage's reported "
                    f"travel range ({y_lo / XY_COUNTS_PER_UM:.1f} to {y_hi / XY_COUNTS_PER_UM:.1f} um)."
                )
            m.iXPOSITION = x_counts
            m.iYPOSITION = y_counts

        self._thread.call(move)
        after = self.XY_GetPosition()
        print(
            f"[NISSdk] XY_Move: requested ({x:.2f}, {y:.2f}) um "
            f"[counts ({x_counts}, {y_counts})] - before {before} - after {after} um"
        )
        if round(after[0] * XY_COUNTS_PER_UM) != x_counts or round(after[1] * XY_COUNTS_PER_UM) != y_counts:
            print(f"[NISSdk] WARNING: XY_Move readback does not match requested counts!")

    def Z_GetPosition(self) -> float:
        """Return the current focus (z) position in microns."""
        z = self._thread.call(lambda m: m.iZPOSITION / Z_COUNTS_PER_UM)
        return to_plain_float(z)

    def Z_Move(self, z: float) -> None:
        """Move focus to an absolute z position in microns.

        Raises ValueError without moving anything if the step is larger
        than MAX_Z_STEP_UM, or if the target is outside the stage's own
        reported travel range (ZPosition .Lower/.Higher). A big blind Z
        move risks crashing the objective into the sample - prefer
        nudge_pfs_offset() for routine focus adjustment when PFS is
        engaged, and reserve this for deliberate, small, checked steps.
        """
        before = self.Z_GetPosition()
        z = to_plain_float(z)
        step_um = abs(z - before)
        if step_um > MAX_Z_STEP_UM:
            raise ValueError(
                f"Requested Z move of {step_um:.1f} um exceeds the "
                f"{MAX_Z_STEP_UM:.0f} um per-call safety limit. Break large "
                f"focus changes into smaller confirmed steps."
            )
        z_counts = round(z * Z_COUNTS_PER_UM)

        def move(m):
            lo, hi = m.ZPosition.Lower, m.ZPosition.Higher
            if not (lo <= z_counts <= hi):
                raise ValueError(
                    f"Z target {z:.2f} um is outside the stage's reported "
                    f"travel range ({lo / Z_COUNTS_PER_UM:.1f} to {hi / Z_COUNTS_PER_UM:.1f} um)."
                )
            m.iZPOSITION = z_counts

        self._thread.call(move)
        after = self.Z_GetPosition()
        print(
            f"[NISSdk] Z_Move: requested {z:.2f} um [counts {z_counts}] - "
            f"before {before} um - after {after} um"
        )
        if round(after * Z_COUNTS_PER_UM) != z_counts:
            print(f"[NISSdk] WARNING: Z_Move readback does not match requested counts!")

    def pfs_status(self) -> dict:
        """Return Nikon PFS (Perfect Focus System) status: whether it's
        currently enabled/locked, its raw status code, current offset, and
        the offset's SDK-reported valid range - plus the active objective's
        model/magnification/working distance, since working distance is
        directly relevant to how much Z headroom actually exists.

        IsPFSEnabled is a property of the *objective* (INikonTi2AxObjective,
        obtained via microscope.Objective(iNOSEPIECE)), not the microscope
        object itself - confirmed from NkTi2Ax.py's interface definitions.
        """

        def read(m):
            objective = m.Objective(m.iNOSEPIECE)
            return {
                "enabled": bool(objective.IsPFSEnabled),
                "status": m.iPFS_STATUS,
                "offset": m.iPFS_OFFSET,
                "offset_lower": m.PfsOffset.Lower,
                "offset_higher": m.PfsOffset.Higher,
                # Diagnostic fields on the PfsOffset INikonTi2AxSetting
                # object, not yet used elsewhere - checking these because a
                # nudge_pfs_offset() write silently didn't take effect
                # (offset_before == offset_after), and "Control" in
                # particular may indicate whether the SDK currently has
                # write access to this setting vs. e.g. the NIS-Elements UI
                # holding it.
                "offset_control": m.PfsOffset.Control,
                "offset_enabled": m.PfsOffset.Enabled,
                "offset_unit": m.PfsOffset.Unit,
                "offset_scale": m.PfsOffset.Scale,
                "objective_model": objective.Model,
                "objective_magnification": objective.Magnification,
                # CONFIRMED millimeters, 2026-08-10 - cross-referenced
                # against C:\ProgramData\Laboratory Imaging\Platform\AX
                # Confocal\Objectives.xml, NIS-Elements' own objective
                # catalog, which reports this same 10x objective's
                # WorkingDistance as 4000 in explicit microns - matching
                # this property's raw value of 4.0 at a 1000x ratio.
                "working_distance_mm": objective.WorkingDistance,
            }

        return self._thread.call(read)

    def nudge_pfs_offset(self, delta_counts: float) -> dict:
        """Adjust the PFS focus-lock offset by a small relative amount.

        Units are raw PFS-offset counts, not calibrated to microns (unlike
        XY/Z - see the module docstring). Requires PFS to already be
        enabled/locked: this only fine-tunes an existing lock, it never
        bootstraps focus from an arbitrary Z position, so it can't cause
        the kind of blind, uncalibrated jump Z_Move can. Each call is
        capped to PFS_MAX_OFFSET_STEP_FRACTION of the SDK's own reported
        valid offset range.

        TODO(unconfirmed, 2026-08-10): writing iPFS_OFFSET while PFS is
        actively enabled/locked has been observed to silently no-op on
        real hardware - offset_before == offset_after, no exception - on
        two separate real-microscope tests (delta=100, PFS enabled/locked,
        offset mid-range at 18966/40000, PfsOffset.Control=1 and .Enabled=1
        so no obvious permission lockout visible from the SDK's own
        diagnostic fields). The write may need PFS to be in a different
        mode, a different property/method entirely, or there may be an
        interlock not exposed by NkTi2Ax's type library. Confirm the
        correct procedure with Nikon's SDK docs or a Ti2 SDK-experienced
        contact (see docs/microscope-notes.md) before relying on this -
        do not attempt to fix by further trial-and-error against real
        hardware. Failure mode observed so far is safe (no motion, no
        error) - not a functional feature yet, but not a hazard either.
        """
        delta_counts = to_plain_float(delta_counts)

        def do_nudge(m):
            if not m.Objective(m.iNOSEPIECE).IsPFSEnabled:
                raise PermissionError(
                    "PFS is not currently enabled/locked - nudge_pfs_offset "
                    "only fine-tunes an existing focus lock, it will not "
                    "engage PFS or move focus from an arbitrary starting "
                    "position. Enable PFS from the NIS-Elements UI first."
                )
            lo, hi = m.PfsOffset.Lower, m.PfsOffset.Higher
            max_step = (hi - lo) * PFS_MAX_OFFSET_STEP_FRACTION
            if abs(delta_counts) > max_step:
                raise ValueError(
                    f"Requested PFS offset nudge of {delta_counts:.1f} exceeds "
                    f"the safety limit of {max_step:.1f} "
                    f"({PFS_MAX_OFFSET_STEP_FRACTION * 100:.0f}% of the "
                    f"reported valid range {lo} to {hi})."
                )
            current = m.iPFS_OFFSET
            target = current + delta_counts
            if not (lo <= target <= hi):
                raise ValueError(
                    f"Target PFS offset {target:.1f} is outside the reported "
                    f"valid range {lo} to {hi}."
                )
            m.iPFS_OFFSET = round(target)
            return {"offset_before": current, "offset_after": m.iPFS_OFFSET}

        return self._thread.call(do_nudge)

    # ── Optical configuration snapshot/replay ────────────────────────────
    # Not related to XY/Z stage safety - these are device/optics settings
    # (objective, filters, light path, illumination), not physical stage
    # travel, so none of the move-safety caps above apply here. Applying a
    # config does still physically move things (turret rotation, filter
    # wheels) - it's not purely passive, just not stage-collision risk the
    # way an uncapped Z move is.

    def get_optical_configuration(self) -> dict:
        """Snapshot the current optical/device configuration - objective
        (nosepiece), DIC prism/polarizer, analyzer, light path, condenser,
        zoom, both filter turrets, and the 4 D-LEDI illumination channels -
        as a dict of raw SDK property values. Save this (e.g. to a JSON
        file) and pass it to apply_optical_configuration() later to
        reproduce the same setup.

        Property semantics (e.g. whether iDLED1_POS is an intensity
        percent or something else) are NOT independently calibrated the
        way XY/Z/PFS units were (see this module's docstring for that
        methodology) - this assumes reading then writing back the same
        raw value reproduces the same NIS-Elements UI state, which is
        reasonable since these are exactly the properties NIS's own
        "Optical Configuration" presets are built from, but it hasn't
        been verified against a known physical reference the way stage
        position was.
        """

        def read(m):
            return {name: getattr(m, name) for name in OPTICAL_CONFIG_PROPERTIES}

        return self._thread.call(read)

    def apply_optical_configuration(self, config: dict) -> dict:
        """Write back a config dict from get_optical_configuration().

        Only keys in OPTICAL_CONFIG_PROPERTIES are applied - unknown keys
        (e.g. from a config saved by a newer version of this code) are
        silently ignored rather than erroring. Returns the read-back
        value of every property actually applied.

        Switching the objective (iNOSEPIECE) changes working distance,
        which may invalidate prior Z-safety assumptions for whatever
        position you're at - this method never touches Z itself, that's
        on the caller to handle deliberately afterward if needed.
        """

        def write(m):
            applied = {}
            for name, value in config.items():
                if name not in OPTICAL_CONFIG_PROPERTIES:
                    continue
                setattr(m, name, value)
                applied[name] = getattr(m, name)
            return applied

        return self._thread.call(write)


if __name__ == "__main__":
    sdk = NISSdk()
    print("Current position:", sdk.XY_GetPosition(), sdk.Z_GetPosition())
