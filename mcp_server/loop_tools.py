# loop_tools.py
# ------------------------------------------------------------
# Minimal MCP surface for the normal agent control loop:
#   get_image()         - capture + the stage position that goes with it
#   get_pos()            - a lightweight, on-demand position sync primitive
#   move()               - move XY, return the ACTUAL resulting position
#   get_move_history()   - every point move() has visited this session
#
# Exactly 4 tools, by explicit direction - do not add more without
# checking first. Calibration, historical-frame lookup, etc. should be
# done by the model reasoning over get_image()'s embedded picture, not
# by adding a dedicated tool per capability.
#
# NO NIS-ELEMENTS ANYWHERE IN THIS FILE (explicit team decision - manager
# does not want capture going through NIS): get_image() captures via
# acquisition.backends.baumer_genicam.BaumerGenICam - a GenICam/GenTL
# camera reached directly through the `harvesters` library, no NIS-
# Elements process, no Jobs, no nis_ar.exe. get_pos()/move() are also
# NIS-Elements-independent for real hardware (backend="sdk" talks to the
# Ti2 ActiveX SDK directly - does not require NIS-Elements to be
# running); backend="mock" falls back to nis_mock.MockNIS for offline
# dev, which also has no NIS-Elements dependency (see stage_positions.py -
# `import nis` only ever succeeds inside NIS-Elements' own bundled
# Python, never from this repo's .venv).
#
# WHY get_move_history(): move() already returns the resulting position
# of ONE move - this answers "where has the stage actually been", e.g.
# "did I already image this area" or "what path did I take to get here"
# without the model needing to remember/re-derive it turn over turn.
# Every move() call appends one record (requested target, actual
# resulting position, timestamps, backend) to logs/move_history.jsonl -
# append-only, one JSON object per line, so concurrent/repeated runs
# never need a read-modify-write of the whole file.
#
# WHY THIS SHAPE (design discussion with the team):
# Stage position is volatile physical state - if it's pushed into the
# model's context continuously (e.g. injected into the system prompt),
# the model's belief about "where the stage is" can go stale between
# inference and action (a human bumps the joystick, another process
# moves it, etc.), and the model acts on a position that's no longer
# true. The fix is to stop treating position as ambient context and
# instead attach it as a return value on the calls that already touch
# it - get_image() and move() both return the position they observed,
# so the model rarely needs to call get_pos() at all. get_pos() is
# there mainly as an explicit sync primitive: call it when a human/
# other process may have moved the stage, when enough time has passed
# that cached state might be stale, or when you want position without
# paying for a full image capture.
#
# WHY NO get_time(): wall-clock time is something the agent/runtime
# already knows - no reason to ask the microscope for it. What matters
# is DEVICE/EVENT time (when was this physical fact true), so that's
# attached as metadata on every call instead (measured_at/captured_at,
# ISO-8601 wall clock; monotonic_ms, so "218ms after this image" style
# reasoning isn't thrown off by clock corrections).
#
# stage_revision: a monotonic counter, incremented whenever get_pos()/
# move()/get_image() observes a position that differs (to 0.01 um) from
# the last position this process saw - see _note_position(). This is
# pull-based, not push-based: an external actor (joystick, another
# controller) moving the stage is only detected the next time one of
# these tools is actually called, not the instant it happens. That's
# enough to let the model tell "the image/position I'm holding is still
# current" from "something moved the stage since I last looked" without
# a full get_pos() round-trip every time - compare the stage_revision
# already attached to an old result against a fresh one. An optional
# expected_revision on move() (to reject a move if the stage changed
# since the agent last looked) is left for when that need actually shows
# up, same as before.
#
# frame_id: a monotonic per-process counter on every get_image() capture
# (see _next_frame_id()), analogous to stage_revision but for images -
# lets a caller refer back to "the image from frame 42" unambiguously.
# Each capture's metadata is also appended to FRAME_HISTORY_PATH (see
# _append_frame_history), and get_frame(frame_id) resolves a frame_id
# back to that record - but get_frame() is a plain Python function, NOT
# a 5th MCP tool: team direction is to keep the model-facing surface at
# exactly 4 tools, so history/lookup logic can grow underneath without
# growing what the model itself can call.
#
# WHY move() IS XY-ONLY (not x/y/z): a blind absolute Z move must never
# be reachable from a chat prompt (risk of crashing the objective into
# the sample). Z is out of scope for this tool entirely.
#
# WHY get_image() REQUIRES confirm=True: unlike get_pos()/move(), it
# fires a real camera - same safety-gate pattern used for anything that
# touches real hardware, even when (as here) there's no backend="mock"
# equivalent to fall back to.
#
# WHY get_image() RETURNS [metadata, image] INSTEAD OF JUST A PATH: MCP
# tool results can embed real image content (base64 + mime type), not
# just text - returning a bare file path means the model can't actually
# see the picture. The embedded preview is a downscaled JPEG (see
# _make_preview_image) - full-res PNGs are ~2-3MB, far more than a
# vision model needs to interpret the image, and wasteful of context
# budget per capture. The full-resolution file still saves to disk at
# metadata["image"] for anything that needs full quality.
# ------------------------------------------------------------

import io
import json
import threading
from datetime import datetime
from pathlib import Path
from time import monotonic

from PIL import Image as _PILImage

from acquisition.backends.baumer_genicam import BaumerGenICam
from acquisition.orchestration.stage_positions import to_plain_float
from mcp.server.mcpserver import Image as MCPImage

# Long-edge size for the JPEG preview embedded in get_image()'s MCP
# response - see that function for why this exists.
PREVIEW_MAX_DIMENSION = 1024

REPO_ROOT = Path(__file__).resolve().parent.parent
MOVE_HISTORY_PATH = REPO_ROOT / "logs" / "move_history.jsonl"
FRAME_HISTORY_PATH = REPO_ROOT / "logs" / "frame_history.jsonl"


def _get_backend(backend: str):
    """Return the stage backend named by `backend` ("mock" or "sdk").

    "mock" returns the same persistent, module-level `nis` object that
    acquisition.orchestration.stage_positions.StagePositionManager uses
    (either MockNIS, or the real `nis` module if this happens to be
    running inside NIS-Elements' own Python environment) - NOT a fresh
    MockNIS() per call, so repeated calls agree on simulated state.

    No safety gate here - only for use by read-only callers. Move/write
    calls must call _require_confirm_for_sdk() first.
    """
    if backend == "mock":
        from acquisition.orchestration.stage_positions import nis
        return nis
    elif backend == "sdk":
        from acquisition.backends.nis_sdk import NISSdk
        return NISSdk()
    raise ValueError(f"Unknown backend '{backend}'. Expected 'mock' or 'sdk'.")


def _require_confirm_for_sdk(backend: str, confirm: bool) -> None:
    """Raise PermissionError if `backend` is "sdk" and `confirm` is not True.

    Safety gate for every tool that can move real hardware - must be
    called before any stage motion, never bypassed with a silent
    fallback to mock.
    """
    if backend not in ("mock", "sdk"):
        raise ValueError(f"Unknown backend '{backend}'. Expected 'mock' or 'sdk'.")
    if backend == "sdk" and not confirm:
        raise PermissionError(
            "backend='sdk' controls real microscope hardware and requires "
            "confirm=True. Refusing to proceed without explicit confirmation."
        )


# Persistent camera connection, opened lazily on the first get_image()
# call and reused after that - matches nis_sdk.py's pattern for the
# stage connection (constructing BaumerGenICam() is not cheap: it opens
# the GenTL producer, enumerates devices, and starts continuous
# acquisition - see that class's __init__). A camera can only be held
# open by one process at a time, so this also means: close any other
# GenICam consumer (Baumer Camera Explorer, etc.) before the first call.
_camera: "BaumerGenICam | None" = None
_camera_lock = threading.Lock()


def _get_camera() -> BaumerGenICam:
    global _camera
    with _camera_lock:
        if _camera is None:
            _camera = BaumerGenICam()
        return _camera


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


# Process-local revision state - see this file's header comment
# ("stage_revision" / "frame_id") for why these exist and why they're
# pull-based rather than backed by a hardware change-notification.
_revision_lock = threading.Lock()
_stage_revision = 0
_last_known_position: dict | None = None
_frame_counter = 0


def _note_position(position: dict) -> int:
    """Bump _stage_revision if `position` differs (to 0.01 um) from the
    last position this process observed, then return the current
    revision. Called from every get_pos()/move()/get_image() so any of
    them can surface "did the stage move since the caller last looked".
    """
    global _stage_revision, _last_known_position
    rounded = {axis: round(value, 2) for axis, value in position.items()}
    with _revision_lock:
        if rounded != _last_known_position:
            _stage_revision += 1
            _last_known_position = rounded
        return _stage_revision


def _next_frame_id() -> int:
    global _frame_counter
    with _revision_lock:
        _frame_counter += 1
        return _frame_counter


def _validate_crop(crop: dict) -> None:
    """Raise ValueError if `crop` isn't a well-formed {"x","y","width",
    "height"} box - each a 0.0-1.0 fraction of the full-res frame, with
    the box not running past the frame's edge. Fractions (not pixels) so
    the caller never needs to know the camera's actual sensor
    resolution - see get_image()'s docstring.
    """
    missing = {"x", "y", "width", "height"} - crop.keys()
    if missing:
        raise ValueError(f"crop is missing required key(s): {sorted(missing)}")
    for key in ("x", "y", "width", "height"):
        value = crop[key]
        if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"crop['{key}'] must be a number between 0.0 and 1.0, got {value!r}")
    if crop["x"] + crop["width"] > 1.0:
        raise ValueError("crop['x'] + crop['width'] must not exceed 1.0")
    if crop["y"] + crop["height"] > 1.0:
        raise ValueError("crop['y'] + crop['height'] must not exceed 1.0")


def _make_preview_image(
    path: Path,
    crop: dict | None = None,
    max_dimension: int = PREVIEW_MAX_DIMENSION,
) -> MCPImage:
    """Downscale a full-res capture (optionally cropped first) to a small
    JPEG for embedding in the MCP response's content, so the model can
    actually see the picture in context instead of only getting a file
    path it can't view. The full-res PNG stays on disk (the `image` path
    in get_image()'s return dict) for anything that needs full quality -
    this preview, cropped or not, is only for the model's own visual
    interpretation.

    crop: optional {"x","y","width","height"} fractions of the full-res
    frame (see _validate_crop) - lets a caller spend preview resolution
    on a small region of interest instead of the whole frame, e.g. after
    spotting something in a wide low-res shot. None (default) previews
    the whole frame, unchanged from before this parameter existed.

    max_dimension: long-edge cap in pixels for the returned JPEG -
    defaults to PREVIEW_MAX_DIMENSION, but a crop commonly wants a
    higher cap (already a small region, so more pixels are still cheap)
    or a survey shot a lower one.
    """
    with _PILImage.open(path) as img:
        img = img.convert("RGB")
        if crop is not None:
            width, height = img.size
            box = (
                round(crop["x"] * width),
                round(crop["y"] * height),
                round((crop["x"] + crop["width"]) * width),
                round((crop["y"] + crop["height"]) * height),
            )
            img = img.crop(box)
        img.thumbnail((max_dimension, max_dimension))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return MCPImage(data=buf.getvalue(), format="jpeg")


def _append_move_history(record: dict) -> None:
    """Append one move record as a single JSON line to MOVE_HISTORY_PATH.

    Append-only by design (see this file's header) - never reads or
    rewrites prior entries, so this stays cheap regardless of how long
    the history grows.
    """
    MOVE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MOVE_HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def _append_frame_history(record: dict) -> None:
    """Append one get_image() metadata record as a single JSON line to
    FRAME_HISTORY_PATH - same append-only pattern as
    _append_move_history, for the same reason. Lets get_frame() resolve
    a frame_id back to its full-res path/position/stage_revision later,
    without keeping every frame in the model's context.
    """
    FRAME_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FRAME_HISTORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_frame(frame_id: int) -> dict:
    """Look up a previously captured frame's metadata record (image path,
    position, stage_revision, timestamps) by the frame_id get_image()
    returned for it.

    NOT registered as an MCP tool (see server_loop.py) - by explicit team
    direction the chat-facing surface stays at 4 tools. This is a plain
    importable function for harness/analysis code that needs to resolve
    "frame 42" to its full-res file and the position it was captured at,
    without re-reading FRAME_HISTORY_PATH by hand.

    Raises KeyError if frame_id was never captured.
    """
    if FRAME_HISTORY_PATH.exists():
        with open(FRAME_HISTORY_PATH, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["frame_id"] == frame_id:
                    return record
    raise KeyError(f"No captured frame with frame_id {frame_id}.")


def get_pos(backend: str = "mock") -> dict:
    """Return the current stage (x, y, z) position, in microns - a cheap,
    on-demand sync primitive rather than something to call before every
    move.

    Reach for this when: a human may have moved the stage (joystick),
    another controller/process may have moved it, enough time has passed
    that a previously-observed position might be stale, or you want
    position without paying for a full get_image() capture. During
    normal operation, get_image() and move() already return position as
    part of their result, so most loops don't need this at all.

    backend: "mock" (default, safe) or "sdk" (real hardware). Read-only -
    no confirmation required for either backend.

    Also returns stage_revision - see this file's header comment - so a
    caller holding an older result can tell whether the stage has moved
    since then without diffing raw coordinates itself.
    """
    nis = _get_backend(backend)
    x, y = nis.XY_GetPosition()
    z = nis.Z_GetPosition()
    position = {"x": to_plain_float(x), "y": to_plain_float(y), "z": to_plain_float(z)}
    return {
        "position": position,
        "stage_revision": _note_position(position),
        "measured_at": _now_iso(),
        "monotonic_ms": int(monotonic() * 1000),
        "backend": backend,
    }


def move(x: float, y: float, backend: str = "mock", confirm: bool = False) -> dict:
    """Move the XY stage to an absolute (x, y) position, in microns, and
    return the ACTUAL resulting position - not merely "success". A real
    move commonly lands slightly off the requested target, so callers
    should treat the returned position as ground truth, not an echo of
    the input.

    Z is intentionally not accepted here - see this file's header
    comment for why absolute Z stays out of the chat-reachable surface.

    backend: "mock" (default, safe) or "sdk" (real hardware - requires
    confirm=True, same gate as every other move tool in this repo).
    """
    _require_confirm_for_sdk(backend, confirm)
    nis = _get_backend(backend)

    started_at = _now_iso()
    nis.XY_Move(x, y)
    new_x, new_y = nis.XY_GetPosition()
    z = nis.Z_GetPosition()
    completed_at = _now_iso()

    position = {"x": to_plain_float(new_x), "y": to_plain_float(new_y), "z": to_plain_float(z)}
    result = {
        "position": position,
        "stage_revision": _note_position(position),
        "started_at": started_at,
        "completed_at": completed_at,
        "backend": backend,
    }
    _append_move_history({
        "requested": {"x": to_plain_float(x), "y": to_plain_float(y)},
        **result,
    })
    return result


def get_image(
    confirm: bool = False,
    exposure_time_us: float | None = None,
    gain: float | None = None,
    crop: dict | None = None,
    max_dimension: int | None = None,
) -> list:
    """Grab one frame from the Baumer GenICam camera (see
    acquisition.backends.baumer_genicam.BaumerGenICam) and return it
    together with the exact stage position associated with it - read
    immediately after the frame is captured, so the two stay coupled
    without the model needing a separate get_pos() call.

    Returns [metadata_dict, image] - the metadata dict (frame_id,
    position, stage_revision, timestamps, full-res file path) as one
    content block, plus an actual viewable image as a second content
    block (a downscaled JPEG preview - see _make_preview_image -
    embedded directly in the MCP response, not just a path the model
    can't see). The full-resolution PNG is still saved to disk at
    metadata_dict["image"] for anything that needs full quality
    (analysis, calibration, etc.) - the embedded preview is only for the
    model's own visual interpretation.

    frame_id is a per-process monotonic counter identifying this
    specific capture; stage_revision is the same counter get_pos()/
    move() use - see this file's header comment - so a caller can tell
    whether the stage has moved since this particular image was taken.

    No NIS-Elements involved - connects directly to the camera via its
    GenTL producer (see BaumerGenICam/find_cti_files), independent of any
    NIS-Elements process. Real hardware only (no mock equivalent - there's
    nothing to simulate a camera trigger against) - requires confirm=True,
    same safety-gate pattern as every other real-hardware-touching tool in
    this repo. Position is read via backend="sdk" (Ti2 ActiveX SDK), the
    same NIS-Elements-independent hardware path move()/get_pos() use.

    exposure_time_us, gain: optional - if given, applied via
    BaumerGenICam.set_settings() before capturing (raises ValueError if
    outside the camera's own reported valid range - see that method's
    docstring for confirmed ranges/units, notably that "gain" is the
    camera's own unit-less scale, not dB). Omit either to leave it at
    whatever the camera is already set to (persists across calls, since
    the camera connection - and therefore its settings - is reused, not
    reopened, between get_image() calls).

    The saved image is real RGB (demosaiced from the sensor's raw
    BayerRG8 via BaumerGenICam.capture()) - see that method's docstring
    for the one unconfirmed detail (which Bayer color code is actually
    correct for this camera).

    crop: optional {"x","y","width","height"}, each a 0.0-1.0 fraction of
    the full frame, restricting the embedded preview to that region -
    e.g. after a wide shot shows something interesting near the right
    edge, crop={"x":0.6,"y":0.2,"width":0.3,"height":0.3} previews just
    that area instead of resending the whole frame. The full-res file on
    disk is always the complete, uncropped frame - crop only changes
    what's embedded for the model to look at. Raises ValueError if the
    box isn't within [0, 1] or runs past the frame's edge. Omit for the
    previous behavior (whole-frame preview).

    max_dimension: long-edge cap in pixels for the embedded preview,
    overriding PREVIEW_MAX_DIMENSION for this call - e.g. a smaller
    value for a coarse survey shot, or a larger one when cropping (an
    already-small region can afford more pixels). Omit to use the
    default.
    """
    if not confirm:
        raise PermissionError(
            "get_image fires a real camera and requires confirm=True. "
            "Refusing to proceed without explicit confirmation."
        )
    if crop is not None:
        _validate_crop(crop)

    camera = _get_camera()
    if exposure_time_us is not None or gain is not None:
        camera.set_settings(exposure_time_us=exposure_time_us, gain=gain)
    image_path = camera.capture()
    pos = get_pos(backend="sdk")

    metadata = {
        "image": str(image_path),
        "frame_id": _next_frame_id(),
        "position": pos["position"],
        "stage_revision": pos["stage_revision"],
        "captured_at": pos["measured_at"],
        "monotonic_ms": pos["monotonic_ms"],
        "crop": crop,
    }
    _append_frame_history(metadata)
    preview = _make_preview_image(
        image_path,
        crop=crop,
        max_dimension=max_dimension if max_dimension is not None else PREVIEW_MAX_DIMENSION,
    )
    return [metadata, preview]


def get_move_history(limit: int = 50) -> dict:
    """Return the most recent points move() has actually moved the stage
    to, oldest-first, each with its requested target, actual resulting
    position, and timestamps - so "where has this session already been"
    doesn't need to be remembered/re-derived turn over turn.

    limit: max number of most-recent records to return (default 50) -
    the log file itself is never truncated, only what's returned here.

    Returns {"history": [...], "returned": N, "total_moves": M} - total_moves
    lets the caller tell "you're seeing the last 50 of 300" apart from
    "you're seeing everything there is".
    """
    if not MOVE_HISTORY_PATH.exists():
        return {"history": [], "returned": 0, "total_moves": 0}

    with open(MOVE_HISTORY_PATH, "r") as f:
        lines = [line for line in f if line.strip()]

    records = [json.loads(line) for line in lines[-limit:]]
    return {"history": records, "returned": len(records), "total_moves": len(lines)}
