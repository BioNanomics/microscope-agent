# The MCP client harness (`harness/mcp_agent.py`)

This document explains `harness/mcp_agent.py` in depth: what MCP actually
is here, why the code is shaped the way it is, how it differs from
`harness/agent.py`, and how to run and troubleshoot it. For a short
summary see the README; this file is the deep dive.

## Why two harnesses exist

This repo has two independent, permanent ways to drive the microscope
tools with Claude:

| | `harness/agent.py` ("Plan B") | `harness/mcp_agent.py` ("Plan A", this doc) |
|---|---|---|
| How it calls the tools | Imports `mcp_server/loop_tools.py`'s functions directly, in-process | Talks real MCP protocol to `mcp_server/server_loop.py`, running as a **separate subprocess** |
| Needs the `mcp` SDK client | No | Yes (`mcp.client`) |
| Works if the harness and the microscope server run on different machines | No | Yes (with a different transport than stdio - see below) |
| Exercises the same server Claude Desktop/Code connect to | No | Yes |
| Complexity | Lower | Higher (protocol/subprocess plumbing) |

Neither is "the real one" - they're both real, tested implementations,
kept side by side on purpose. Pick `agent.py` when you just want the
simplest thing that works on one machine. Pick `mcp_agent.py` when you
specifically want the MCP protocol boundary - e.g. testing
`server_loop.py` itself, or a future setup where the harness and the
microscope PC aren't the same machine.

## What MCP actually is, briefly

MCP (Model Context Protocol) is a JSON-RPC-based protocol for a
*client* (something that wants to use tools) to talk to a *server*
(something that exposes tools, resources, and prompts). It says nothing
about how you talk to an LLM - that's a completely separate concern.
Concretely, in this repo:

```
harness/mcp_agent.py                    mcp_server/server_loop.py
        (MCP client)   <--- stdio --->        (MCP server)
              |                                     |
              |                                     +-- loop_tools.get_image
        talks to Claude                             +-- loop_tools.get_pos
        (Anthropic API,                             +-- loop_tools.move
         unrelated to MCP)                           +-- loop_tools.get_move_history
```

`mcp_agent.py` spawns `python -m mcp_server.server_loop` as a child
process and exchanges newline-delimited JSON-RPC messages with it over
that process's stdin/stdout - the exact same mechanism Claude
Desktop/Code use when you configure this project as an MCP server for
them. Separately, and with no connection to MCP at all,
`mcp_agent.py` also calls the Anthropic API directly to actually talk
to Claude, exactly like `agent.py` does.

## The MCP SDK version actually matters here

This repo pins `mcp==2.0.0` in `requirements.txt`. That's not a detail
to skim past: this package speaks **MCP protocol version `2026-07-28`**,
which is materially different from the `2025-06-18` version most MCP
tutorials, blog posts, and even some official examples were written
against (`mcp` 1.x). The old 1.x pattern - manually opening a
`ClientSession` over `stdio_client(StdioServerParameters(...))`, calling
`.initialize()` yourself - is deprecated in this SDK; a lot of what it
did (the `initialize` handshake, `ping`, client→server progress,
`resources/subscribe`) is either replaced or removed outright in the
newer protocol.

This code was written by reading the actual installed package source
(`.venv/Lib/site-packages/mcp/client/client.py`) rather than from
memory or from older docs, specifically because guessing here would
have produced code against a protocol version this SDK doesn't
implement the same way. The current, correct pattern is the unified
`Client` wrapper:

```python
from mcp.client import Client
from mcp.client.stdio import stdio_client, StdioServerParameters

params = StdioServerParameters(command=sys.executable, args=["-m", "mcp_server.server_loop"])
async with Client(stdio_client(params)) as client:
    print(client.protocol_version)   # "2026-07-28"
    tools = await client.list_tools()
    result = await client.call_tool("get_pos", {"backend": "mock"})
```

`Client` also accepts an in-process `Server`/`MCPServer` object directly
(no subprocess, no JSON-RPC framing at all) or a URL string (Streamable
HTTP transport) - `stdio_client(...)` is only one of three ways to
construct the `server=` argument. If this harness ever needs to talk to
a microscope server running on a different machine, swapping the
transport is the only change required; nothing else in this file
depends on stdio specifically.

## Why a manual loop, not the SDK's tool_runner

The Anthropic Python SDK has a beta `client.beta.messages.tool_runner()`
helper that automates the whole "call model → run tool → feed result
back" cycle. `mcp_agent.py` doesn't use it, for the same two reasons
`agent.py` doesn't:

1. **Image pruning.** Between tool-call rounds, this harness strips old
   images out of the message history (see "Context management" below).
   The tool_runner doesn't expose a hook for rewriting history between
   rounds.
2. **The real-hardware approval gate** (next section) needs to intercept
   specific tool calls *before* they execute and possibly refuse to run
   them - again, not something the tool_runner's per-turn hooks are
   built for as cleanly as owning the loop outright.

There's a third, smaller reason: the tool_runner is beta, and a manual
loop has no such dependency.

## The real-hardware approval gate

`loop_tools.get_image()`/`move()` (in `mcp_server/loop_tools.py`) gate
any real-hardware action behind a `confirm=True` parameter, on the
assumption that whatever is calling them already obtained a human's
approval. That assumption is true when the caller is Claude
Desktop/Code - their own permission-prompt UI *is* that approval step.
It is **not** automatically true here: this harness calls tools over raw
MCP with no such UI in between.

So this file does two things together, and both matter:

1. **`_strip_confirm()`** removes the `"confirm"` property from every
   tool's `input_schema` before it's shown to Claude - verified directly
   against `server_loop.py`'s real `list_tools()` output (see
   `_mcp_tools_to_anthropic_tools`'s docstring), not assumed. Claude
   literally cannot request `confirm=True`; the field doesn't exist in
   the tool definition it sees.
2. **`_confirm_real_hardware_action()`** is the harness's own live gate:
   before calling `get_image` (always real hardware) or `move` with
   `backend="sdk"`, it prints the proposed call and blocks on a real
   terminal prompt (`input()`, run via `asyncio.to_thread` so it doesn't
   block the event loop). Only on a human "y" does the harness itself
   add `"confirm": True` to the arguments dict it sends to
   `call_tool()`. A decline returns a normal `is_error` tool result to
   Claude - not a crash - so it can react and adjust its plan instead of
   the whole session dying.

This is the load-bearing safety property of this file. If you ever
modify `_execute_tool()`, keep this ordering: gate first, only inject
`confirm` after a live "yes", never let the model's own `tool_use.input`
supply that field.

## Context management (image pruning)

Same mechanism as `agent.py`, reused directly from `harness/context.py`
rather than reimplemented: after every round of tool execution,
`prune_context_images(messages, keep_first_n=0, keep_last_n=2)` strips
image content out of all but the 2 most recent image-bearing tool
results, keeping their text (position, `frame_id`, timestamps) intact.
See `harness/context.py`'s own header comment and `docs/mcp_harness.md`'s
sibling discussion in the README for the reasoning; nothing about that
mechanism is MCP-specific, which is exactly why it could be reused
unchanged here.

## Converting between MCP and Anthropic content shapes

MCP tool results (`CallToolResult.content`) are typed content blocks -
`TextContent`, `ImageContent`, etc. - not plain Python dicts. Anthropic's
`tool_result` content blocks have their own, similar-but-distinct shape.
`_mcp_content_to_anthropic_content()` does this conversion:

| MCP `ContentBlock` | Anthropic tool_result block |
|---|---|
| `TextContent(text=...)` | `{"type": "text", "text": ...}` |
| `ImageContent(data=..., mime_type=...)` | `{"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": data}}` |
| anything else (audio, resource links, ...) | a text block noting the unsupported type - `loop_tools.py`'s 4 tools never actually produce these, so this is a defensive fallback, not a real code path |

Conveniently, MCP's `Tool.input_schema` is already plain JSON Schema in
the same shape Anthropic's tool `input_schema` expects, so
`_mcp_tools_to_anthropic_tools()` is mostly a field rename (`name`,
`description`, `input_schema`) plus `_strip_confirm()` - not a schema
translation.

## A bug this work found and fixed

Building this harness surfaced a real, pre-existing bug:
`acquisition/backends/nis_mock.py`'s `XY_Move`/`Z_Move` used a plain
`print(...)` (stdout) to log simulated moves. When `server_loop.py` runs
as an MCP stdio server, **stdout is the JSON-RPC channel** -
`stage_positions.py` already documented this exact constraint for its
own startup message, but `nis_mock.py`'s move logging never followed
it. The symptom: calling `move()` with `backend="mock"` through a real
MCP client corrupted the JSON-RPC stream ("Failed to parse JSONRPC
message from server"), even though the exact same call worked fine
through `agent.py` (Plan B), which never touches stdio at all. Fixed by
routing those two `print()` calls to `sys.stderr` instead, matching the
convention already established elsewhere in this codebase. This is a
good example of why Plan A is worth having even if you mostly use Plan
B: it exercises a code path (the real stdio/JSON-RPC boundary) that
in-process calls never touch.

## Running it

```
python -m harness.mcp_agent
```

Requires `ANTHROPIC_API_KEY` (`.env` at the repo root, a real
environment variable, or `ant auth login`) - same resolution as
`agent.py`. On startup it prints the negotiated MCP protocol version and
the tool list it discovered from the server, then behaves like a normal
chat REPL (`/exit` to quit). Real-hardware calls pause for a live "y/N"
approval, same as `agent.py`.

## What's been verified end-to-end

Both with the mock backend (no real hardware needed) and real Claude
API calls:

- `list_tools()` correctly discovers all 4 tools, with `confirm` absent
  from every schema Claude sees.
- `call_tool()` round-trips correctly for `get_pos`, `move`, and
  `get_move_history` - including error cases (unknown tool name, a
  declined real-hardware action) coming back as normal `is_error` tool
  results rather than crashing the session.
- A full multi-turn conversation (move → check history → check position
  again) correctly tracked `stage_revision` across turns through the
  real MCP boundary, matching the behavior already verified for Plan B.
- The `nis_mock.py` stdout fix: confirmed the exact call sequence that
  previously corrupted the JSON-RPC stream now completes cleanly.

Not yet exercised: `get_image()` (always real-hardware, no mock path
exists for it) and the real-hardware approval gate against actual
hardware (only unit-tested with a scripted decline so far, on both
harnesses).
