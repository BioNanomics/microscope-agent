# mcp_agent.py
# ------------------------------------------------------------
# Interactive harness loop - Plan A: calls Claude directly (Anthropic
# API), but drives tools through a REAL MCP client talking to
# mcp_server/server_loop.py as a separate subprocess over stdio, using
# the official `mcp` SDK (mcp.client.Client). This is the MCP-protocol
# counterpart to harness/agent.py (Plan B, which imports loop_tools.py's
# functions directly, in-process, with no MCP involved at all).
#
# Both harnesses are independent and permanent - neither replaces the
# other. Use this one when the harness and the microscope server should
# be separate processes (possibly on separate machines) talking real
# MCP, or when you want to exercise server_loop.py itself, the same
# server Claude Desktop/Code connect to. Use agent.py when you want the
# simplest possible path with no protocol/subprocess overhead.
#
# For a walkthrough of how MCP fits in here, what the current MCP
# protocol version this SDK speaks actually looks like, and why several
# design choices below exist, see docs/mcp_harness.md.
#
# WHY mcp.client.Client (not the older ClientSession/stdio_client
# pattern some docs still show): this repo's installed `mcp` SDK is
# 2.0.0, speaking MCP protocol version 2026-07-28 - a materially newer
# spec than the 2025-06-18 version most existing docs/examples (written
# for `mcp` 1.x) describe. `Client` is the current SDK's unified,
# high-level wrapper: it accepts a Transport (or an in-process
# Server/MCPServer, or a URL) and handles the handshake/version
# negotiation itself. Verified directly against the installed package's
# source (.venv/Lib/site-packages/mcp/client/client.py) rather than
# assumed from training data or from stale examples - see
# docs/mcp_harness.md for specifics of what changed.
#
# WHY A SEPARATE SUBPROCESS (stdio), not an in-process Client(mcp):
# Client can connect to an in-process MCPServer object directly with no
# subprocess or protocol framing at all - but that would just be Plan B
# again with extra ceremony. The whole point of Plan A is a real client/
# server boundary: server_loop.py runs as its own process (the same one
# Claude Desktop/Code would launch), reachable over stdio exactly the
# way any external MCP client reaches it.
#
# WHY "confirm" IS STRIPPED FROM EVERY TOOL SCHEMA BEFORE CLAUDE SEES IT,
# AND WHY THE HARNESS INJECTS IT ITSELF: identical reasoning to
# agent.py's header comment - loop_tools.get_image()/move() gate real
# hardware behind confirm=True on the assumption that a human already
# approved the call (true inside Claude Desktop/Code, via their own
# permission-prompt UI; NOT automatically true here). _strip_confirm()
# removes the "confirm" property from every tool's input_schema (it
# always carries a default and is never in "required" - verified
# directly against server_loop.py's actual list_tools() output, not
# assumed), so the model cannot request it. _confirm_real_hardware_action()
# is the harness's own live terminal gate; only on approval does the
# harness add "confirm": True to the arguments dict it sends to
# call_tool() - the model's own tool_use.input is never trusted for that
# field, even if it tried to set one (stripped, so it can't).
#
# Run (from the repo root, with .venv activated, ANTHROPIC_API_KEY set
# in a .env file at the repo root, in the environment, or via
# `ant auth login`):
#   python -m harness.mcp_agent
# ------------------------------------------------------------

import asyncio
import json
import sys

import anthropic
from dotenv import load_dotenv
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.shared.exceptions import MCPError

from harness.context import prune_context_images

load_dotenv()  # reads .env (repo root, gitignored) into the process environment

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # pragma: no cover - stdout isn't a reconfigurable TextIOWrapper
    pass

MODEL = "claude-opus-5"
MAX_TOKENS = 16000
MAX_TOOL_ROUNDS = 25
KEEP_LAST_N_IMAGES = 2

SYSTEM_PROMPT = """\
You are an agent operating a Nikon Ti2-E microscope stage and a Baumer \
camera through an MCP server (mcp_server/server_loop.py), reached via \
tools named get_image, get_pos, move, get_move_history.

Safety rules:
- Always use backend="mock" (the default) unless the user has clearly \
asked you to control the real, physical microscope. Only pass \
backend="sdk" when real hardware action is actually intended.
- Every real-hardware action (get_image always; move when \
backend="sdk") pauses for a live human approval at the terminal before \
it executes - expect that pause, and explain to the user what you're \
about to do and why before calling it, so the approval makes sense to \
whoever is watching the terminal.
- move() only accepts XY - there is no way to move Z from this tool \
surface, by design (crash risk into the sample). Don't suggest workarounds.
- Position is not something you should assume is still current after a \
few turns have passed - get_pos()/move()/get_image() all return \
stage_revision; a later stage_revision than one you saw earlier means \
the stage moved since then (possibly by someone else).
- Older captured images are pruned from your context after a couple of \
turns - their metadata (position, frame_id, timestamps) stays in the \
text history even after the pixels are gone, so rely on that text \
rather than assuming you can still see an old frame.
"""


def _strip_confirm(input_schema: dict) -> dict:
    """Return a copy of an MCP tool's input_schema with "confirm" removed
    from its properties - see this file's header comment for why.
    """
    schema = dict(input_schema)
    properties = dict(schema.get("properties", {}))
    properties.pop("confirm", None)
    schema["properties"] = properties
    return schema


def _mcp_tools_to_anthropic_tools(mcp_tools: list) -> list[dict]:
    """Convert MCP Tool objects (from Client.list_tools()) to Anthropic's
    tool definition shape. MCP's input_schema is already JSON Schema in
    the same shape Anthropic's input_schema expects, so this is mostly a
    field rename plus _strip_confirm() - not a schema translation.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": _strip_confirm(tool.input_schema),
        }
        for tool in mcp_tools
    ]


def _text_content(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


async def _confirm_real_hardware_action(tool_name: str, tool_input: dict) -> bool:
    """Live human approval gate for anything that touches real hardware.
    Runs input() in a worker thread (asyncio.to_thread) so it doesn't
    block the event loop, even though this REPL has nothing else running
    concurrently in practice.
    """
    print(f"\n[HARDWARE GATE] Claude wants to call {tool_name}({tool_input}) on the REAL microscope.")
    answer = await asyncio.to_thread(input, "Allow this real-hardware action? [y/N]: ")
    return answer.strip().lower() in ("y", "yes")


def _mcp_content_to_anthropic_content(mcp_content: list) -> list[dict]:
    """Convert a CallToolResult's content blocks (TextContent/ImageContent/
    ...) to Anthropic tool_result content blocks. Only text and image are
    handled - loop_tools.py's 4 tools never return audio/resource blocks.
    """
    blocks = []
    for block in mcp_content:
        if block.type == "text":
            blocks.append({"type": "text", "text": block.text})
        elif block.type == "image":
            blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": block.mime_type, "data": block.data},
            })
        else:
            blocks.append({"type": "text", "text": f"[unsupported MCP content type: {block.type}]"})
    return blocks


async def _execute_tool(mcp_client: Client, name: str, tool_input: dict) -> tuple[list[dict], bool]:
    """Run one tool call through the real MCP client and return
    (content_blocks, is_error) in the shape Anthropic's tool_result
    content expects.
    """
    tool_input = dict(tool_input)
    try:
        if name == "get_image":
            if not await _confirm_real_hardware_action(name, tool_input):
                return _text_content("User declined this real-hardware capture. Not executed."), True
            tool_input["confirm"] = True
        elif name == "move" and tool_input.get("backend", "mock") == "sdk":
            if not await _confirm_real_hardware_action(name, tool_input):
                return _text_content("User declined this real-hardware move. Not executed."), True
            tool_input["confirm"] = True

        result = await mcp_client.call_tool(name, tool_input)
        return _mcp_content_to_anthropic_content(result.content), result.is_error
    except MCPError as exc:
        # Protocol-level failure (unknown tool, malformed call, etc.) -
        # tool-originated errors (bad move() input, etc.) already come
        # back as a normal CallToolResult with is_error=True, not here.
        return _text_content(f"MCPError: {exc}"), True


async def run_turn(anthropic_client: anthropic.AsyncAnthropic, mcp_client: Client, tools: list[dict], messages: list[dict]) -> str:
    """Run one user turn to completion: call Claude, execute any tool
    calls through the real MCP client, prune old images from `messages`
    between rounds, and repeat until Claude stops calling tools (or
    MAX_TOOL_ROUNDS is hit). Mutates `messages` in place. Returns the
    final assistant text.
    """
    response = None
    for _ in range(MAX_TOOL_ROUNDS):
        response = await anthropic_client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools,
            thinking={"type": "adaptive"},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
        if not tool_use_blocks:
            break

        tool_results = []
        for block in tool_use_blocks:
            content, is_error = await _execute_tool(mcp_client, block.name, block.input)
            result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
            if is_error:
                result["is_error"] = True
            tool_results.append(result)
        messages.append({"role": "user", "content": tool_results})

        messages[:] = prune_context_images(messages, keep_first_n=0, keep_last_n=KEEP_LAST_N_IMAGES)
    else:
        print("[warning] hit MAX_TOOL_ROUNDS without a final answer for this turn.")

    return "\n".join(block.text for block in response.content if block.type == "text")


async def main() -> None:
    anthropic_client = anthropic.AsyncAnthropic()
    messages: list[dict] = []

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server_loop"],
    )

    print("Connecting to mcp_server/server_loop.py over MCP (stdio)...")
    async with Client(stdio_client(server_params)) as mcp_client:
        print(f"Connected - MCP protocol version: {mcp_client.protocol_version}")
        mcp_tools = (await mcp_client.list_tools()).tools
        tools = _mcp_tools_to_anthropic_tools(mcp_tools)
        print(f"Tools available: {', '.join(t['name'] for t in tools)}")

        print("\nMicroscope harness - real MCP client (backend defaults to mock; real hardware needs live approval).")
        print("Type /exit to quit.\n")

        while True:
            try:
                user_input = await asyncio.to_thread(input, "You: ")
                user_input = user_input.strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input in ("/exit", "exit", "quit"):
                break

            messages.append({"role": "user", "content": user_input})
            try:
                final_text = await run_turn(anthropic_client, mcp_client, tools, messages)
            except anthropic.APIStatusError as exc:
                print(f"\n[API error] {exc}\n")
                continue
            except anthropic.APIConnectionError as exc:
                print(f"\n[connection error] {exc}\n")
                continue
            print(f"\nClaude: {final_text}\n")


if __name__ == "__main__":
    asyncio.run(main())
