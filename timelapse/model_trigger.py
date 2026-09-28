# model_trigger.py
# ------------------------------------------------------------
# The "ask the model" seam for the adaptive time-lapse: an on_trigger
# hook for scheduler.AdaptiveTimelapse that shows Claude the frame before
# and the frame that fired the detector, plus the detector's own reason,
# and asks one question: is this worth a longer look?
#
#   "extend"  - keep burst-imaging for another window
#   "ignore"  - this is not interesting (dust, a bubble, a drift), end
#               the burst now and go back to the slow interval
#   None      - no opinion; the scheduler carries on with its own timer
#
# WHY THE MODEL IS HERE AND NOWHERE ELSE: the detector runs on every
# frame and is free. The model runs only when the detector fires - a
# handful of times per run - so its cost and latency never touch the
# fast loop, and every decision it makes is written down with a reason
# (see .decisions and the scheduler's event log).
#
# WHY IT CAN NEVER MAKE THINGS WORSE: every failure path returns None.
# No key, no network, a refusal, an unparseable answer - the scheduler
# behaves exactly as it would with no hook. The model can shorten or
# lengthen a burst; it cannot capture, move, or refocus anything.
#
# THE API CALL IS ONE INJECTABLE FUNCTION (`ask`), so the whole class is
# testable without a key or a network: tests pass a fake `ask` and
# assert on what it was given and how its answer was interpreted.
# ------------------------------------------------------------

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from PIL import Image

from timelapse.change_detector import ChangeScore

MODEL = "claude-opus-5"           # same as harness/agent.py and harness/mcp_agent.py
MAX_TOKENS = 1024
PREVIEW_MAX_DIMENSION = 512       # small on purpose: the question is "did the specimen change", not fine detail

SYSTEM_PROMPT = """\
You are watching a live-cell time-lapse on a microscope. A change detector has
just flagged the latest frame as different from the frames before it, and the
scheduler has switched to fast burst imaging. You see the frame from just before
the trigger and the frame that fired it, with the detector's numeric reason.

Decide whether continuing fast imaging is worth the light exposure:
- "extend": the specimen itself is doing something (moving, growing, retracting,
  changing shape or internal structure). Keep imaging fast.
- "ignore": the change is not the specimen - debris, a bubble, a focus drift,
  an illumination change, noise, or a stage shift. Stop the burst.

Reply with only a JSON object: {"decision": "extend" | "ignore", "reason": "<one sentence>"}.
"""

AskFn = Callable[[str, list[dict]], str]   # (system_prompt, user_content_blocks) -> model text


def _jpeg_b64(path: str | Path, max_dimension: int = PREVIEW_MAX_DIMENSION) -> str:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((max_dimension, max_dimension))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _image_block(path: str | Path) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _jpeg_b64(path)}}


def build_user_content(event: dict, score: ChangeScore) -> list[dict]:
    """The user turn: before-frame (if there is one), trigger frame, and the
    detector's numbers. Frames are downscaled JPEGs - the full-res files
    stay on disk, exactly as get_image() does for its own preview."""
    blocks: list[dict] = []
    previous = event.get("previous_image")
    if previous and Path(previous).exists():
        blocks.append({"type": "text", "text": "Frame before the trigger:"})
        blocks.append(_image_block(previous))
    blocks.append({"type": "text", "text": "Frame that fired the detector:"})
    blocks.append(_image_block(event["image"]))
    blocks.append({"type": "text", "text": (
        f"Detector reason: {score.reason}. "
        f"Pixel change {score.diff_score:.1f}x baseline noise; "
        f"foreground area changed by {score.area_delta:+.1%}; "
        f"whole-field shift {score.shift_px:.1f}px. "
        f"Stage position: {event.get('position')}."
    )})
    return blocks


def parse_decision(text: str) -> tuple[str | None, str]:
    """Pull {"decision": ..., "reason": ...} out of the model's reply.
    Tolerates prose around the JSON. Anything unrecognizable -> (None, text)."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None, text.strip()
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, text.strip()
    decision = str(obj.get("decision", "")).strip().lower()
    reason = str(obj.get("reason", "")).strip()
    if decision not in ("extend", "ignore"):
        return None, reason or text.strip()
    return decision, reason


def anthropic_ask(system_prompt: str, content: list[dict]) -> str:
    """The real call. Constructed lazily so importing this module never
    needs the SDK or a key - only actually asking does. Credentials come
    from the environment / .env / `ant auth login`, same as the harnesses."""
    import anthropic
    from dotenv import load_dotenv
    load_dotenv()
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    if response.stop_reason == "refusal":
        return ""
    return "".join(block.text for block in response.content if block.type == "text")


@dataclass
class ModelTrigger:
    """on_trigger hook for AdaptiveTimelapse. Call it like the scheduler
    does: trigger(event, metadata, score) -> "extend" | "ignore" | None.

    max_calls: hard cap on model calls per run - after it, the hook is a
        no-op (returns None). Cost control for an unattended run.
    min_interval_s: don't ask again within this many seconds of the last
        ask - a burst that keeps extending should not spam the model.
    """

    ask: AskFn = anthropic_ask
    max_calls: int = 50
    min_interval_s: float = 30.0
    clock: Callable[[], float] = time.monotonic
    log: Callable[[str], None] = print
    calls: int = 0
    decisions: list[dict] = field(default_factory=list)
    _last_ask_at: float | None = None

    def __call__(self, event: dict, metadata: dict, score: ChangeScore) -> str | None:
        now = self.clock()
        if self.calls >= self.max_calls:
            self._record(event, None, f"skipped: max_calls={self.max_calls} reached")
            return None
        if self._last_ask_at is not None and now - self._last_ask_at < self.min_interval_s:
            self._record(event, None, f"skipped: asked {now - self._last_ask_at:.0f}s ago")
            return None

        self.calls += 1
        self._last_ask_at = now
        try:
            content = build_user_content(event, score)
            text = self.ask(SYSTEM_PROMPT, content)
        except Exception as exc:  # noqa: BLE001 - any failure here must be non-fatal
            self._record(event, None, f"model call failed: {type(exc).__name__}: {exc}")
            return None
        decision, reason = parse_decision(text)
        self._record(event, decision, reason if decision else f"unparseable reply: {reason[:120]}")
        return decision

    def _record(self, event: dict, decision: str | None, reason: str) -> None:
        entry = {"frame_id": event.get("frame_id"), "image": event.get("image"),
                 "decision": decision, "reason": reason}
        self.decisions.append(entry)
        self.log(f"[model] frame {entry['frame_id']}: {decision or 'no opinion'} - {reason}")
