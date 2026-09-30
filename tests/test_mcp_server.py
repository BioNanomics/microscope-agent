# End-to-end over the real protocol: spawn `python -m mcp_server.server_loop`
# as a subprocess (exactly what Claude Desktop/Code and harness/mcp_agent.py
# do), talk MCP to it over stdio, and exercise every tool against the mock
# backends. No hardware, no model. Guards two things the in-process tests
# cannot: that the server actually starts and serves on a core-only
# install, and that the model-facing tool surface is exactly the agreed
# set - nothing added by accident.
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client

EXPECTED_TOOLS = {"get_image", "get_pos", "move", "get_move_history", "estop"}


def _server_params(data_dir: Path, frame: Path) -> StdioServerParameters:
    env = dict(os.environ)
    env.update({
        "CONFOCAL_MCP_DATA_DIR": str(data_dir),
        "CONFOCAL_MOCK_FRAME_PATH": str(frame),
        "CONFOCAL_NO_ESTOP_PANEL": "1",   # headless: don't try to open the STOP window
    })
    return StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server_loop"], env=env)


def _text(result) -> dict:
    block = next(b for b in result.content if b.type == "text")
    return json.loads(block.text)


async def _session(data_dir: Path, frame: Path):
    async with Client(stdio_client(_server_params(data_dir, frame))) as client:
        tools = (await client.list_tools()).tools
        names = {t.name for t in tools}

        pos = _text(await client.call_tool("get_pos", {"backend": "mock"}))
        moved = _text(await client.call_tool("move", {"x": 250.0, "y": -100.0, "backend": "mock"}))
        history = _text(await client.call_tool("get_move_history", {"limit": 5}))

        img = await client.call_tool("get_image", {"backend": "mock", "max_dimension": 64})
        img_meta = _text(img)
        img_types = [b.type for b in img.content]

        gated = await client.call_tool("get_image", {})            # backend defaults to sdk, no confirm
        gated_move = await client.call_tool("move", {"x": 0, "y": 0, "backend": "sdk"})
        estop = _text(await client.call_tool("estop", {"action": "status"}))

        return dict(names=names, pos=pos, moved=moved, history=history, img_meta=img_meta,
                    img_types=img_types, gated=gated, gated_move=gated_move, estop=estop)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("mcp-data")
    sys.path.insert(0, str(Path(__file__).parent))
    from conftest import make_blob_frame, save_frame
    frame = save_frame(make_blob_frame(), data_dir / "frame.png")
    return asyncio.run(_session(data_dir, frame))


def test_tool_surface_is_exactly_the_agreed_set(run):
    assert run["names"] == EXPECTED_TOOLS


def test_mock_stage_round_trip(run):
    assert run["pos"]["backend"] == "mock"
    assert run["moved"]["position"]["x"] == 250.0
    assert run["moved"]["position"]["y"] == -100.0
    assert run["history"]["total_moves"] >= 1
    last = run["history"]["history"][-1]
    assert last["requested"] == {"x": 250.0, "y": -100.0}
    assert last["position"]["x"] == 250.0


def test_mock_capture_returns_metadata_and_embedded_image(run):
    assert run["img_meta"]["backend"] == "mock"
    assert Path(run["img_meta"]["image"]).exists()
    assert run["img_meta"]["position"]["x"] == 250.0   # position is coupled to the capture
    assert "image" in run["img_types"], run["img_types"]


def test_real_hardware_refused_without_confirm_over_protocol(run):
    assert run["gated"].is_error is True
    assert run["gated_move"].is_error is True


def test_estop_status_reports_over_protocol(run):
    assert "engaged" in run["estop"]
    assert run["estop"]["engaged"] is False
