# agent.py
# ------------------------------------------------------------
# Minimal interactive harness loop: calls Claude directly (Anthropic
# API), driving mcp_server/loop_tools.py's 4 functions as tools. This is
# Plan B from the harness design discussion - it does NOT go through
# mcp_server/server_loop.py's MCP/stdio protocol at all. Since this
# module lives in the same codebase as loop_tools.py, it just imports
# and calls those functions directly as plain Python. server_loop.py is
# untouched and still works independently for Claude Desktop/Code - this
# is a second, separate way to drive the same underlying tool functions,
# not a replacement.
#
# WHY A MANUAL LOOP (not the Anthropic SDK's beta tool_runner): full
# control is needed over two things the tool_runner doesn't expose - (1)
# pruning old images out of the message history between rounds (see
# harness/context.py), and (2) a live human approval gate before any
# real-hardware action (see next paragraph) - and avoiding a beta SDK
# dependency for the first version of this loop.
#
# WHY "confirm" IS NOT A MODEL-SETTABLE TOOL PARAMETER HERE: loop_tools'
# get_image()/move() gate real hardware behind confirm=True, on the
# assumption that whatever's calling them already got a human's
# approval - true when the caller is Claude Desktop/Code (their own
# permission-prompt UI is that approval step), but NOT automatically
# true here, since this loop calls Python functions directly with no
# such UI. So confirm is deliberately left out of every tool's
# input_schema below - the model cannot set it - and _confirm_real_hardware_action()
# is the harness's own live approval gate: it prints the proposed call
# and requires a human "y" at the terminal before confirm=True is ever
# passed to the real function. A declined action is reported back to
# the model as a normal (if_error) tool result, not a crash, so it can
# adjust its plan.
#
# WHY backend DEFAULTS TO "mock" IN THE SYSTEM PROMPT: real hardware
# should be something the model reaches for deliberately, not a default
# it falls into - matches loop_tools.py's own backend="mock" defaults.
#
# Run (from the repo root, with .venv activated, ANTHROPIC_API_KEY set
# in a .env file at the repo root, in the environment, or via
# `ant auth login`):
#   python -m harness.agent
# ------------------------------------------------------------

import json
import sys

import anthropic
from dotenv import load_dotenv

from harness.context import prune_context_images
from mcp_server import loop_tools

load_dotenv()  # reads .env (repo root, gitignored) into the process environment

# Windows terminals commonly default stdout to cp1252, which cannot encode
# characters Claude's responses routinely use (arrows, em-dashes, etc.) -
# without this, printing such a response raises UnicodeEncodeError and
# crashes the loop mid-conversation. errors="replace" is a second line of
# defense in case some other exotic character still isn't representable.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:  # pragma: no cover - stdout isn't a reconfigurable TextIOWrapper
    pass

MODEL = "claude-opus-5"
MAX_TOKENS = 16000
MAX_TOOL_ROUNDS = 25  # safety cap on tool-call rounds within a single user turn

# How many of the most recent image-bearing tool results stay as real
# images in context; older ones keep their text (position, frame_id,
# etc.) but lose the pixels - see harness/context.py.
KEEP_LAST_N_IMAGES = 2

SYSTEM_PROMPT = """\
You are an agent operating a Nikon Ti2-E microscope stage and a Baumer \
camera through four tools: get_image, get_pos, move, get_move_history.

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

TOOLS = [
    {
        "name": "get_image",
        "description": (
            "Capture one frame from the real camera, paired with the exact "
            "stage position it was taken at. Always touches real hardware - "
            "pauses for human approval before executing. Optionally crop and/or "
            "resize the embedded preview."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "exposure_time_us": {
                    "type": ["number", "null"],
                    "description": "Camera exposure time in microseconds. Omit to keep the current setting.",
                },
                "gain": {
                    "type": ["number", "null"],
                    "description": "Camera gain (camera's own unit-less scale, not dB). Omit to keep the current setting.",
                },
                "crop": {
                    "type": ["object", "null"],
                    "description": (
                        "Restrict the embedded preview to a region of interest: "
                        "x, y, width, height, each a 0.0-1.0 fraction of the full frame. "
                        "Omit to preview the whole frame."
                    ),
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "width": {"type": "number"},
                        "height": {"type": "number"},
                    },
                },
                "max_dimension": {
                    "type": ["integer", "null"],
                    "description": "Long-edge cap in pixels for the embedded preview. Omit for the default.",
                },
            },
        },
    },
    {
        "name": "get_pos",
        "description": (
            "Current stage (x, y, z) position, in microns. Cheap read-only sync "
            "primitive - get_image/move already return position, so this is mainly "
            "for checking whether the stage moved since you last looked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "backend": {
                    "type": "string",
                    "enum": ["mock", "sdk"],
                    "description": 'Defaults to "mock" if omitted.',
                },
            },
        },
    },
    {
        "name": "move",
        "description": (
            "Move the XY stage to an absolute (x, y) position, in microns. Returns "
            "the actual resulting position. backend=\"sdk\" (real hardware) pauses "
            "for human approval before executing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "backend": {
                    "type": "string",
                    "enum": ["mock", "sdk"],
                    "description": 'Defaults to "mock" if omitted.',
                },
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "get_move_history",
        "description": "Every point move() has actually sent the stage to this session.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": ["integer", "null"],
                    "description": "Max number of most-recent records to return. Omit for the default (50).",
                },
            },
        },
    },
]


def _text_content(text: str) -> list[dict]:
    return [{"type": "text", "text": text}]


def _confirm_real_hardware_action(tool_name: str, tool_input: dict) -> bool:
    """Live human approval gate for anything that touches real hardware -
    see this file's header comment for why this exists and why it can't
    be satisfied by the model itself.
    """
    print(f"\n[HARDWARE GATE] Claude wants to call {tool_name}({tool_input}) on the REAL microscope.")
    answer = input("Allow this real-hardware action? [y/N]: ").strip().lower()
    return answer in ("y", "yes")


def _execute_tool(name: str, tool_input: dict) -> tuple[list[dict], bool]:
    """Run one tool call and return (content_blocks, is_error) in the
    shape Anthropic's tool_result content expects.
    """
    tool_input = dict(tool_input)
    try:
        if name == "get_image":
            if not _confirm_real_hardware_action(name, tool_input):
                return _text_content("User declined this real-hardware capture. Not executed."), True
            metadata, image = loop_tools.get_image(confirm=True, **tool_input)
            image_content = image.to_image_content()
            return [
                {"type": "text", "text": json.dumps(metadata)},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_content.mime_type,
                        "data": image_content.data,
                    },
                },
            ], False

        if name == "get_pos":
            return _text_content(json.dumps(loop_tools.get_pos(**tool_input))), False

        if name == "move":
            if tool_input.get("backend", "mock") == "sdk":
                if not _confirm_real_hardware_action(name, tool_input):
                    return _text_content("User declined this real-hardware move. Not executed."), True
                tool_input["confirm"] = True
            return _text_content(json.dumps(loop_tools.move(**tool_input))), False

        if name == "get_move_history":
            return _text_content(json.dumps(loop_tools.get_move_history(**tool_input))), False

        return _text_content(f"Unknown tool: {name}"), True
    except (ValueError, PermissionError, KeyError) as exc:
        return _text_content(f"{type(exc).__name__}: {exc}"), True


def run_turn(client: anthropic.Anthropic, messages: list[dict]) -> str:
    """Run one user turn to completion: call Claude, execute any tool
    calls, prune old images from `messages` between rounds, and repeat
    until Claude stops calling tools (or MAX_TOOL_ROUNDS is hit).
    Mutates `messages` in place. Returns the final assistant text.
    """
    response = None
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            thinking={"type": "adaptive"},
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
        if not tool_use_blocks:
            break

        tool_results = []
        for block in tool_use_blocks:
            content, is_error = _execute_tool(block.name, block.input)
            result = {"type": "tool_result", "tool_use_id": block.id, "content": content}
            if is_error:
                result["is_error"] = True
            tool_results.append(result)
        messages.append({"role": "user", "content": tool_results})

        messages[:] = prune_context_images(messages, keep_first_n=0, keep_last_n=KEEP_LAST_N_IMAGES)
    else:
        print("[warning] hit MAX_TOOL_ROUNDS without a final answer for this turn.")

    return "\n".join(block.text for block in response.content if block.type == "text")


def main() -> None:
    client = anthropic.Anthropic()
    messages: list[dict] = []

    print("Microscope harness (backend defaults to mock; real hardware needs live approval).")
    print("Type /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input in ("/exit", "exit", "quit"):
            break

        messages.append({"role": "user", "content": user_input})
        try:
            final_text = run_turn(client, messages)
        except anthropic.APIStatusError as exc:
            print(f"\n[API error] {exc}\n")
            continue
        except anthropic.APIConnectionError as exc:
            print(f"\n[connection error] {exc}\n")
            continue
        print(f"\nClaude: {final_text}\n")


if __name__ == "__main__":
    main()
