# The harness conversation loops (run_turn / main) and the tool plumbing around
# the approval gate, driven by a scripted stand-in for the Anthropic client.
# No API key, no network, no hardware.
import asyncio
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import anthropic
import pytest
from PIL import Image

from acquisition import estop
from acquisition.backends import nis_mock
from acquisition.orchestration import stage_positions
from harness import agent, mcp_agent
from mcp_server import loop_tools

ROOT = Path(__file__).parent.parent


def tool_use(name, input_, id_="t1"):
    return NS(type="tool_use", id=id_, name=name, input=input_)


def text(t):
    return NS(type="text", text=t)


@pytest.fixture(autouse=True)
def mock_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_tools, "MOVE_HISTORY_PATH", tmp_path / "move.jsonl")
    monkeypatch.setattr(loop_tools, "FRAME_HISTORY_PATH", tmp_path / "frame.jsonl")
    monkeypatch.setattr(estop, "ESTOP_PATH", tmp_path / "ESTOP")
    monkeypatch.setattr(nis_mock.time, "sleep", lambda s: None)
    monkeypatch.setattr(nis_mock, "CAPTURE_DIR", tmp_path / "caps")
    monkeypatch.setattr(nis_mock, "SAMPLE_FRAME_PATH", tmp_path / "none.png")
    monkeypatch.setattr(stage_positions, "nis", nis_mock.MockNIS(1, 2, 3))


# ---- shared: scripted model ----------------------------------------------------

class ScriptedMessages:
    def __init__(self, scripted, is_async):
        self.scripted, self.calls, self._async = list(scripted), [], is_async

    def create(self, **kw):
        self.calls.append(kw)
        out = self.scripted.pop(0)
        if self._async:
            async def done():
                return out
            return done()
        return out


def sync_client(*responses):
    return NS(messages=ScriptedMessages([NS(content=r) for r in responses], False))


def async_client(*responses):
    return NS(messages=ScriptedMessages([NS(content=r) for r in responses], True))


# ---- agent.py -------------------------------------------------------------------

def test_agent_gate_prompt_accepts_only_yes(monkeypatch):
    for answer, ok in (("y", True), ("YES", True), ("n", False), ("", False), ("sure", False)):
        monkeypatch.setattr("builtins.input", lambda prompt, a=answer: a)
        assert agent._confirm_real_hardware_action("move", {"x": 1}) is ok


def test_agent_executes_read_tools_and_reports_errors():
    content, err = agent._execute_tool("get_pos", {})
    assert not err and json.loads(content[0]["text"])["position"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    content, err = agent._execute_tool("move", {"x": 5, "y": 6})
    assert not err and json.loads(content[0]["text"])["position"]["x"] == 5.0
    content, err = agent._execute_tool("get_move_history", {"limit": 1})
    assert not err and json.loads(content[0]["text"])["returned"] == 1
    content, err = agent._execute_tool("fly", {})
    assert err and "Unknown tool: fly" in content[0]["text"]
    content, err = agent._execute_tool("move", {"x": 99_999, "y": 0})          # outside the mock's travel
    assert err and content[0]["text"].startswith("ValueError")
    content, err = agent._execute_tool("get_image", {"backend": "nope"})
    assert err


def test_agent_mock_capture_returns_text_and_an_image_block():
    content, err = agent._execute_tool("get_image", {"backend": "mock", "max_dimension": 64})
    assert not err and [b["type"] for b in content] == ["text", "image"]
    assert content[1]["source"]["media_type"] == "image/jpeg"


def test_agent_run_turn_executes_tools_then_returns_the_final_text():
    client = sync_client([tool_use("get_pos", {})], [text("the stage is at 1, 2")])
    messages = [{"role": "user", "content": "where is it"}]
    assert agent.run_turn(client, messages) == "the stage is at 1, 2"
    roles = [m["role"] for m in messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    result = messages[2]["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1" and "is_error" not in result
    assert client.messages.calls[0]["tools"] == agent.TOOLS and client.messages.calls[0]["model"] == agent.MODEL


def test_agent_run_turn_marks_failed_tools_as_errors():
    client = sync_client([tool_use("fly", {})], [text("sorry")])
    messages = [{"role": "user", "content": "go"}]
    agent.run_turn(client, messages)
    assert messages[2]["content"][0]["is_error"] is True


def test_agent_run_turn_prunes_old_images(monkeypatch):
    client = sync_client(*([[tool_use("get_image", {"backend": "mock", "max_dimension": 32}, f"t{i}")] for i in range(4)]
                           + [[text("done")]]))
    messages = [{"role": "user", "content": "shoot"}]
    agent.run_turn(client, messages)
    with_images = [m for m in messages if m["role"] == "user" and isinstance(m["content"], list)
                   and any(b["type"] == "tool_result" and any(c["type"] == "image" for c in b["content"]) for b in m["content"])]
    assert len(with_images) == agent.KEEP_LAST_N_IMAGES
    first = messages[2]["content"][0]["content"]                       # an old frame: text survives, picture gone
    assert [c["type"] for c in first] == ["text"] and "frame_id" in first[0]["text"]


def test_agent_run_turn_stops_at_the_round_cap(monkeypatch, capsys):
    monkeypatch.setattr(agent, "MAX_TOOL_ROUNDS", 2)
    client = sync_client(*([[tool_use("get_pos", {}), text("still going")]] * 2))
    assert agent.run_turn(client, [{"role": "user", "content": "loop"}]) == "still going"
    assert "hit MAX_TOOL_ROUNDS" in capsys.readouterr().out


def scripted_input(monkeypatch, lines):
    it = iter(lines)

    def fake(prompt=""):
        v = next(it)
        if isinstance(v, BaseException):
            raise v
        return v

    monkeypatch.setattr("builtins.input", fake)


def test_agent_main_loop(monkeypatch, capsys):
    monkeypatch.setattr(agent.anthropic, "Anthropic", lambda: "client")
    seen = []
    monkeypatch.setattr(agent, "run_turn", lambda c, m: (seen.append(list(m)), "hello back")[1])
    scripted_input(monkeypatch, ["", "hi", "/exit"])
    agent.main()
    assert len(seen) == 1 and seen[0][0] == {"role": "user", "content": "hi"}
    assert "Claude: hello back" in capsys.readouterr().out


def api_errors():
    """The two API exceptions, built without the HTTP client library so the
    tests do not depend on which one the SDK uses."""
    out = []
    for cls, msg in ((anthropic.APIStatusError, "overloaded"), (anthropic.APIConnectionError, "unreachable")):
        exc = cls.__new__(cls)
        Exception.__init__(exc, msg)
        out.append(exc)
    return out


def test_agent_main_survives_api_errors_and_eof(monkeypatch, capsys):
    monkeypatch.setattr(agent.anthropic, "Anthropic", lambda: "client")
    errors = iter(api_errors())

    def fail(c, m):
        raise next(errors)

    monkeypatch.setattr(agent, "run_turn", fail)
    scripted_input(monkeypatch, ["one", "two", EOFError()])
    agent.main()
    out = capsys.readouterr().out
    assert "[API error]" in out and "[connection error]" in out


# ---- mcp_agent.py ----------------------------------------------------------------

def test_mcp_gate_prompt_accepts_only_yes(monkeypatch):
    for answer, ok in (("y", True), (" Yes ", True), ("no", False), ("", False)):
        monkeypatch.setattr("builtins.input", lambda prompt, a=answer: a)
        assert asyncio.run(mcp_agent._confirm_real_hardware_action("move", {})) is ok


def test_mcp_content_conversion_handles_text_image_and_unknown():
    blocks = mcp_agent._mcp_content_to_anthropic_content([
        NS(type="text", text="hi"),
        NS(type="image", mime_type="image/png", data="QUJD"),
        NS(type="audio")])
    assert blocks[0] == {"type": "text", "text": "hi"}
    assert blocks[1]["source"] == {"type": "base64", "media_type": "image/png", "data": "QUJD"}
    assert "unsupported MCP content type: audio" in blocks[2]["text"]


def test_mcp_tools_convert_to_anthropic_tools_without_confirm():
    tools = mcp_agent._mcp_tools_to_anthropic_tools([
        NS(name="move", description=None, input_schema={"properties": {"x": {}, "confirm": {}}})])
    assert tools == [{"name": "move", "description": "", "input_schema": {"properties": {"x": {}}}}]


def test_mcp_execute_tool_reports_protocol_errors():
    from mcp.shared.exceptions import MCPError

    class Boom:
        async def call_tool(self, name, args):
            raise MCPError(-32601, "no such tool")

    content, err = asyncio.run(mcp_agent._execute_tool(Boom(), "nope", {}))
    assert err and content[0]["text"].startswith("MCPError")


def test_mcp_run_turn_round_trip_and_error_marking():
    class Server:
        async def call_tool(self, name, args):
            return NS(content=[NS(type="text", text="ok")], is_error=(name == "bad"))

    client = async_client([tool_use("get_pos", {}, "a"), tool_use("bad", {}, "b")], [text("all done")])
    messages = [{"role": "user", "content": "go"}]
    out = asyncio.run(mcp_agent.run_turn(client, Server(), [], messages))
    assert out == "all done"
    results = messages[2]["content"]
    assert "is_error" not in results[0] and results[1]["is_error"] is True


def test_mcp_run_turn_stops_at_the_round_cap(monkeypatch, capsys):
    monkeypatch.setattr(mcp_agent, "MAX_TOOL_ROUNDS", 1)

    class Server:
        async def call_tool(self, name, args):
            return NS(content=[NS(type="text", text="ok")], is_error=False)

    client = async_client([tool_use("get_pos", {}), text("partial")])
    assert asyncio.run(mcp_agent.run_turn(client, Server(), [], [{"role": "user", "content": "x"}])) == "partial"
    assert "hit MAX_TOOL_ROUNDS" in capsys.readouterr().out


def fake_mcp(monkeypatch):
    class FakeClient:
        protocol_version = "2026-01-01"

        def __init__(self, transport):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def list_tools(self):
            return NS(tools=[NS(name="get_pos", description="d", input_schema={"properties": {}})])

    seen = {}
    monkeypatch.setattr(mcp_agent, "Client", FakeClient)
    monkeypatch.setattr(mcp_agent, "stdio_client", lambda params: seen.setdefault("params", params))
    monkeypatch.setattr(mcp_agent.anthropic, "AsyncAnthropic", lambda: "client")
    return seen


def test_mcp_main_loop(monkeypatch, capsys):
    seen = fake_mcp(monkeypatch)
    turns = []

    async def turn(client, mcp_client, tools, messages):
        turns.append((tools, list(messages)))
        return "hello back"

    monkeypatch.setattr(mcp_agent, "run_turn", turn)
    scripted_input(monkeypatch, ["", "hi", "exit"])
    asyncio.run(mcp_agent.main())
    assert seen["params"].args == ["-m", "mcp_server.server_loop"]
    assert turns[0][0][0]["name"] == "get_pos" and len(turns) == 1
    assert "Claude: hello back" in capsys.readouterr().out


def test_mcp_main_survives_api_errors_and_eof(monkeypatch, capsys):
    fake_mcp(monkeypatch)
    errors = iter(api_errors())

    async def fail(*a):
        raise next(errors)

    monkeypatch.setattr(mcp_agent, "run_turn", fail)
    scripted_input(monkeypatch, ["one", "two", KeyboardInterrupt()])
    asyncio.run(mcp_agent.main())
    out = capsys.readouterr().out
    assert "[API error]" in out and "[connection error]" in out


# ---- the modules' own sanity-check blocks ------------------------------------------

def test_context_module_self_check_runs(capsys):
    runpy.run_path(str(ROOT / "harness" / "context.py"), run_name="__main__")
    assert "All sanity checks passed." in capsys.readouterr().out


def test_harness_modules_start_from_main(monkeypatch):
    called = []
    monkeypatch.setattr("builtins.input", lambda prompt="": "/exit")
    monkeypatch.setattr(anthropic, "Anthropic", lambda: "client")
    runpy.run_path(str(ROOT / "harness" / "agent.py"), run_name="__main__")
    fake_mcp(monkeypatch)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda: "client")
    import mcp.client as mc, mcp.client.stdio as ms
    class Fake:
        protocol_version = "x"
        def __init__(self, t): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def list_tools(self): return NS(tools=[])
    monkeypatch.setattr(mc, "Client", Fake)
    monkeypatch.setattr(ms, "stdio_client", lambda p: p)
    runpy.run_path(str(ROOT / "harness" / "mcp_agent.py"), run_name="__main__")
