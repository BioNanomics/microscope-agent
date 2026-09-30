# timelapse/model_trigger.py with a fake model: what the model is shown,
# how its answer is read, and that every failure path is a harmless None.
import json

import numpy as np

from timelapse.change_detector import ChangeScore
from timelapse.model_trigger import ModelTrigger, SYSTEM_PROMPT, build_user_content, parse_decision


def _score(reason="foreground area shrank by 5.7%"):
    return ChangeScore(diff_score=6.2, area_delta=-0.057, shift_px=0.0, foreground_fraction=0.05,
                       baseline_frames=5, interesting=True, reason=reason)


def _event(write_frame, with_previous=True):
    after = write_frame("after.png", radius=10.0, seed=2)
    before = write_frame("before.png", seed=1) if with_previous else None
    return {"event": "capture", "frame_id": 7, "image": str(after),
            "previous_image": str(before) if before else None, "position": {"x": 1, "y": 2, "z": 3}}


class FakeAsk:
    def __init__(self, reply):
        self.reply = reply
        self.received = []

    def __call__(self, system_prompt, content):
        self.received.append((system_prompt, content))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# -- what the model is shown ---------------------------------------------------

def test_prompt_contains_before_and_after_images_and_detector_numbers(write_frame):
    content = build_user_content(_event(write_frame), _score())
    kinds = [b["type"] for b in content]
    assert kinds == ["text", "image", "text", "image", "text"]
    for b in content:
        if b["type"] == "image":
            assert b["source"]["media_type"] == "image/jpeg" and len(b["source"]["data"]) > 100
    tail = content[-1]["text"]
    assert "shrank" in tail and "6.2x" in tail and "-5.7%" in tail


def test_prompt_without_previous_frame_has_one_image(write_frame):
    content = build_user_content(_event(write_frame, with_previous=False), _score())
    assert [b["type"] for b in content] == ["text", "image", "text"]


# -- how the answer is read ----------------------------------------------------

def test_parse_decision_variants():
    assert parse_decision('{"decision": "extend", "reason": "tube thickening"}') == ("extend", "tube thickening")
    assert parse_decision('Sure. {"decision":"IGNORE","reason":"bubble"} ok') == ("ignore", "bubble")
    assert parse_decision('{"decision": "maybe"}')[0] is None
    assert parse_decision("no json here")[0] is None
    assert parse_decision("{broken")[0] is None
    assert parse_decision("")[0] is None


def test_trigger_returns_model_decision_and_records_reason(write_frame):
    ask = FakeAsk(json.dumps({"decision": "extend", "reason": "front advancing"}))
    trig = ModelTrigger(ask=ask, log=lambda s: None)
    assert trig(_event(write_frame), {}, _score()) == "extend"
    assert ask.received[0][0] == SYSTEM_PROMPT
    assert trig.decisions[-1]["decision"] == "extend"
    assert trig.decisions[-1]["reason"] == "front advancing"
    assert trig.calls == 1


# -- every failure is a harmless None -----------------------------------------

def test_model_exception_is_swallowed(write_frame):
    trig = ModelTrigger(ask=FakeAsk(RuntimeError("no network")), log=lambda s: None)
    assert trig(_event(write_frame), {}, _score()) is None
    assert "no network" in trig.decisions[-1]["reason"]


def test_unparseable_reply_is_no_opinion(write_frame):
    trig = ModelTrigger(ask=FakeAsk("I think it's interesting!"), log=lambda s: None)
    assert trig(_event(write_frame), {}, _score()) is None


def test_max_calls_cap_and_min_interval(write_frame):
    clock = FakeClock()
    ask = FakeAsk('{"decision": "extend", "reason": "x"}')
    trig = ModelTrigger(ask=ask, max_calls=2, min_interval_s=30.0, clock=clock, log=lambda s: None)
    ev = _event(write_frame)
    assert trig(ev, {}, _score()) == "extend"      # call 1
    assert trig(ev, {}, _score()) is None           # too soon
    clock.t = 31.0
    assert trig(ev, {}, _score()) == "extend"      # call 2
    clock.t = 62.0
    assert trig(ev, {}, _score()) is None           # cap reached
    assert len(ask.received) == 2
    assert "max_calls" in trig.decisions[-1]["reason"]


# -- wired into the scheduler --------------------------------------------------

def test_scheduler_passes_previous_image_and_honours_ignore(write_frame, tmp_path):
    from tests.test_scheduler import FakeClock as SchedClock, FrameSource, _config
    from timelapse.scheduler import AdaptiveTimelapse
    clock = SchedClock()
    quiet = [write_frame(f"q{i}.png", seed=i) for i in range(6)]
    shrunk = [write_frame(f"s{i}.png", radius=10.0, seed=100 + i) for i in range(6)]
    src = FrameSource(clock, quiet + shrunk)
    ask = FakeAsk('{"decision": "ignore", "reason": "bubble"}')
    trig = ModelTrigger(ask=ask, min_interval_s=0.0, log=lambda s: None)
    summary = AdaptiveTimelapse(_config(tmp_path, max_captures=12), capture=src, on_trigger=trig,
                                clock=clock, sleep=clock.sleep, log=lambda s: None).run()
    assert trig.calls >= 1
    _, content = ask.received[0]
    assert [b["type"] for b in content][:2] == ["text", "image"]   # a "before" frame was available
    assert summary.burst_captures == 0                             # ignore ended the burst at once
