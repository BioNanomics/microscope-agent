# The harnesses' real-hardware approval gate is the load-bearing safety
# property of harness/agent.py and harness/mcp_agent.py (see their header
# comments). These tests pin it down without a model or hardware:
#
#   - the model never sees a "confirm" parameter in any tool schema
#   - a real-hardware call (get_image on sdk, move on sdk) stops at the
#     gate; a decline returns an error result and executes nothing;
#     an approval injects confirm=True itself
#   - a mock call never touches the gate at all
#
# The gate's input() prompt is monkeypatched with a recorder, and the
# underlying tool is replaced with a stub that records what it was called
# with, so the assertions are about the harness's decisions only.
import asyncio
import json

import pytest

from harness import agent, mcp_agent
from mcp_server import loop_tools


class FakeImage:
    def to_image_content(self):
        class C:
            mime_type = "image/jpeg"
            data = "AAAA"
        return C()


@pytest.fixture
def gate(monkeypatch):
    """Replace both harnesses' gate with a recorder whose answer we control."""
    calls = []
    state = {"answer": False}

    def sync_gate(name, tool_input):
        calls.append((name, dict(tool_input)))
        return state["answer"]

    async def async_gate(name, tool_input):
        calls.append((name, dict(tool_input)))
        return state["answer"]

    monkeypatch.setattr(agent, "_confirm_real_hardware_action", sync_gate)
    monkeypatch.setattr(mcp_agent, "_confirm_real_hardware_action", async_gate)
    state["calls"] = calls
    return state


@pytest.fixture
def stub_tools(monkeypatch):
    """Stub loop_tools so nothing real runs; record kwargs each was called with."""
    seen = {}

    def get_image(**kw):
        seen["get_image"] = kw
        return {"frame_id": 1, "image": "x.png", "backend": kw.get("backend", "sdk")}, FakeImage()

    def move(**kw):
        seen["move"] = kw
        return {"position": {"x": kw["x"], "y": kw["y"], "z": 0}}

    monkeypatch.setattr(loop_tools, "get_image", get_image)
    monkeypatch.setattr(loop_tools, "move", move)
    return seen


# -- schemas -----------------------------------------------------------------

def _all_property_names(schema: dict) -> set[str]:
    names = set()
    for key, sub in schema.get("properties", {}).items():
        names.add(key)
        if isinstance(sub, dict):
            names |= _all_property_names(sub)
    return names


def test_agent_tool_schemas_never_expose_confirm():
    for tool in agent.TOOLS:
        assert "confirm" not in _all_property_names(tool["input_schema"]), tool["name"]


def test_mcp_agent_strips_confirm_from_real_server_schemas():
    from mcp_server.server_loop import mcp
    mcp_tools = asyncio.run(mcp.list_tools())
    assert any("confirm" in t.input_schema.get("properties", {}) for t in mcp_tools), \
        "precondition: the server itself does expose confirm (the harness must strip it)"
    for t in mcp_agent._mcp_tools_to_anthropic_tools(mcp_tools):
        assert "confirm" not in t["input_schema"]["properties"], t["name"]


def test_strip_confirm_does_not_mutate_input():
    schema = {"type": "object", "properties": {"confirm": {"type": "boolean"}, "x": {"type": "number"}}}
    out = mcp_agent._strip_confirm(schema)
    assert "confirm" not in out["properties"]
    assert "confirm" in schema["properties"]


# -- harness/agent.py (in-process) --------------------------------------------

def test_agent_mock_capture_skips_gate(gate, stub_tools):
    content, is_error = agent._execute_tool("get_image", {"backend": "mock"})
    assert gate["calls"] == []
    assert is_error is False
    assert "confirm" not in stub_tools["get_image"]
    assert [c["type"] for c in content] == ["text", "image"]


def test_agent_real_capture_declined_executes_nothing(gate, stub_tools):
    gate["answer"] = False
    content, is_error = agent._execute_tool("get_image", {})
    assert gate["calls"] == [("get_image", {})]
    assert is_error is True
    assert "get_image" not in stub_tools
    assert "declined" in content[0]["text"].lower()


def test_agent_real_capture_approved_injects_confirm(gate, stub_tools):
    gate["answer"] = True
    _, is_error = agent._execute_tool("get_image", {"exposure_time_us": 5000})
    assert is_error is False
    assert stub_tools["get_image"]["confirm"] is True
    assert stub_tools["get_image"]["exposure_time_us"] == 5000


def test_agent_model_cannot_smuggle_confirm_on_mock(gate, stub_tools):
    # Even if the model somehow sent confirm=True, a mock call must not be
    # promoted to real hardware: backend decides, not the flag.
    agent._execute_tool("get_image", {"backend": "mock", "confirm": True})
    assert gate["calls"] == []
    assert stub_tools["get_image"]["backend"] == "mock"


def test_agent_move_gate_matches_backend(gate, stub_tools):
    agent._execute_tool("move", {"x": 1, "y": 2})                       # backend defaults to mock
    assert gate["calls"] == [] and "confirm" not in stub_tools["move"]

    gate["answer"] = False
    _, is_error = agent._execute_tool("move", {"x": 1, "y": 2, "backend": "sdk"})
    assert is_error is True and gate["calls"] == [("move", {"x": 1, "y": 2, "backend": "sdk"})]

    gate["answer"] = True
    _, is_error = agent._execute_tool("move", {"x": 3, "y": 4, "backend": "sdk"})
    assert is_error is False and stub_tools["move"]["confirm"] is True


# -- harness/mcp_agent.py (over a fake MCP client) -----------------------------

class FakeMCPClient:
    """Records call_tool arguments; returns a fixed text result."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, dict(arguments)))

        class Block:
            type = "text"
            text = json.dumps({"ok": True})

        class Result:
            content = [Block()]
            is_error = False
        return Result()


def _run(coro):
    return asyncio.run(coro)


def test_mcp_agent_mock_capture_skips_gate(gate):
    client = FakeMCPClient()
    _, is_error = _run(mcp_agent._execute_tool(client, "get_image", {"backend": "mock"}))
    assert gate["calls"] == []
    assert is_error is False
    assert client.calls == [("get_image", {"backend": "mock"})]


def test_mcp_agent_real_capture_declined_never_reaches_server(gate):
    gate["answer"] = False
    client = FakeMCPClient()
    _, is_error = _run(mcp_agent._execute_tool(client, "get_image", {}))
    assert is_error is True
    assert client.calls == []


def test_mcp_agent_real_capture_approved_injects_confirm(gate):
    gate["answer"] = True
    client = FakeMCPClient()
    _, is_error = _run(mcp_agent._execute_tool(client, "get_image", {"gain": 2.0}))
    assert is_error is False
    assert client.calls == [("get_image", {"gain": 2.0, "confirm": True})]


def test_mcp_agent_move_gate_matches_backend(gate):
    client = FakeMCPClient()
    _run(mcp_agent._execute_tool(client, "move", {"x": 1, "y": 2, "backend": "mock"}))
    assert gate["calls"] == [] and client.calls[-1][1].get("confirm") is None

    gate["answer"] = False
    _, is_error = _run(mcp_agent._execute_tool(client, "move", {"x": 1, "y": 2, "backend": "sdk"}))
    assert is_error is True and len(client.calls) == 1

    gate["answer"] = True
    _run(mcp_agent._execute_tool(client, "move", {"x": 1, "y": 2, "backend": "sdk"}))
    assert client.calls[-1] == ("move", {"x": 1, "y": 2, "backend": "sdk", "confirm": True})
