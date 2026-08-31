# EAA integration (isolated)

This folder wraps the main project's existing, **unmodified** MCP server
(`../mcp_server/server_loop.py`) as a tool inside
[EAA (Experiment Automation Agents)](https://github.com/AdvancedPhotonSource/EAA).

Everything EAA-related lives here - its own venv, its own dependencies,
its own `.env`, its own runtime state (checkpoint/transcript databases,
memory if ever enabled). Nothing outside this folder was modified to
make this work. See `REMOVAL.md` to fully remove this integration.

## Why a separate folder, separate venv

- `eaa-core` pulls in a large, different dependency stack (LangGraph,
  its own `mcp` 1.29.1 client, OpenAI SDK, etc.) that has nothing to do
  with the main project's dependencies (GenICam camera bindings, Ti2
  SDK bindings, `mcp` 2.0.0). Keeping them in separate venvs means
  neither can break the other.
- `eaa-core`'s GitHub repository carries **no license** (verified
  directly against the source, twice) - isolating it in one folder
  means it can be deleted entirely, cleanly, if that's ever a problem.

## Setup

```
cd eaa_integration
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Fill in `.env` (already gitignored) with a real OpenAI key:
```
OPENAI_API_KEY=sk-...
```

EAA has no native Anthropic/Claude client - only OpenAI-compatible
endpoints (plus two DOE-specific gateways, AskSage and Argo, that
aren't relevant here). This is a **separate key** from the main
project's `ANTHROPIC_API_KEY` - the two harnesses in `../harness/`
still use Claude directly and are unaffected by any of this.

## Run

Terminal mode (default, matches the rest of this project's harnesses):
```
.venv\Scripts\python run_session.py
```

WebUI mode (`--webui` flag) - a browser dashboard instead of the terminal:
```
.venv\Scripts\python run_session.py --webui
```
Then open **http://127.0.0.1:8008**. Confirmed working: real HTML served
with this project's title, correctly wired to the runtime API on port
8010. Important behavior shift, verified directly against the source
(`BaseTaskManager.get_user_input`): in WebUI mode, **every message and
every tool-approval prompt happens in the browser, not the terminal** -
the terminal just runs the process and prints log lines. `--webui` is
opt-in specifically so the already-verified terminal flow keeps working
unchanged when you don't pass it.

This spawns `../mcp_server/server_loop.py` as a subprocess, using the
**main project's own `.venv`** (not this folder's) - `server_loop.py`
needs `mcp`/`PIL`/`harvesters`/`pywin32`, none of which are installed
here. Make sure the main project's `.venv` exists and has its
dependencies installed first (`../requirements.txt`).

## Stopping a session

- **Clean exit:** type `/exit` at the prompt (in the browser chat box in
  `--webui` mode). EAA also has `/return`, `/skill`, etc.
- **Ctrl+C:** terminal mode - first Ctrl+C interrupts the current turn
  (EAA catches it by design), a second within 3 s force-quits. `--webui`
  mode - one Ctrl+C force-quits, since the terminal is the only control
  surface there.

`run_session.py` installs its own signal handler and ends every run with
a child-process tree kill + `os._exit()`. Without that, two things bite
on Windows: EAA's `MCPTool.__del__` deadlocks in anyio at interpreter
shutdown (the process won't close), and the MCP server - launched through
the main venv's redirector `python.exe` - gets orphaned, leaving a stray
`server_loop.py` (and its camera/NIS handles) running. If you ever do
find a leftover, `taskkill /F /T /IM python.exe` clears it, or kill just
the `server_loop.py` PID from Task Manager.

## Safety: how this differs from `../harness/agent.py` and `../harness/mcp_agent.py`

Those two harnesses only pause for a live human approval on
**real-hardware actions specifically** (`get_image`, or `move` with
`backend="sdk"`) - safe mock/read calls go through automatically.

This integration is coarser, deliberately, as a first version:
`MCPTool(..., require_approval=True)` gates **every single tool call**
through this connection - including harmless mock reads like `get_pos`.
Every tool call pauses for a `[y/N]` terminal prompt. Simpler, safer by
default, but naggier. If that becomes a problem, the fix is a thin
proxy MCP server in this same folder that replicates the selective gate
the other two harnesses already use - not a change to `server_loop.py`
itself.

The task manager's system prompt also explicitly tells the model to
default to `backend="mock"` unless real hardware is clearly intended
(see `CONFOCAL_SAFETY_NOTE` in `run_session.py`) - the same convention
the other two harnesses use.

## A real bug this found: BaseTaskManager auto-registers 13 extra tools

`BaseTaskManager` doesn't only expose whatever you pass via `tools=` -
by default it also auto-registers built-in tools for bash execution,
Python execution, file read/write/edit, `uv`, image rendering, and
launching subagents (confirmed directly via
`tool_executor.list_tool_schemas()`, not assumed). None of that was
requested or wanted here - this integration should only ever expose the
4 microscope tools. Worse, several of those built-ins use **dotted**
names (e.g. `simple_python_eval_tool.evaluate_python_expression`),
which OpenAI's function-calling API rejects outright (tool names must
match `^[a-zA-Z0-9_-]+$`) - so the default configuration doesn't just
over-expose capability, it actively **crashes** against OpenAI models.

`run_session.py` fixes this by explicitly calling every `disable_*`
method on `task_manager.tool_manager` right after construction, then
asserts the final tool list is exactly the 4 microscope tools before
`run_conversation()` ever starts. If you ever add a new EAA version and
this integration starts silently exposing tools again, that assertion
will fail loudly instead of quietly widening what the model can touch.

## What's been verified

- `eaa-core==1.0.0` installs cleanly in its own venv, pulling in its own
  `mcp==1.29.1` with zero conflict with the main project's `mcp==2.0.0`.
- `MCPTool` successfully spawns `server_loop.py` as a real subprocess
  (via the main project's venv) and connects to it.
- `ConfocalTaskManager`'s resolved `assistant_system_message` correctly
  contains EAA's own default prompt **plus** the appended microscope
  safety note - not a replacement.
- The approval gate (`require_approval=True`) resolves to a real
  terminal `[y/N]` prompt in this (non-WebUI) configuration.
- **A full live conversation, with a real `OPENAI_API_KEY` and
  `gpt-4o-mini`, end to end**: the model correctly chose to call
  `get_pos(backend="mock")`, the approval prompt fired and was
  approved, the real MCP call executed against the real `server_loop.py`
  subprocess, and the model gave a correct final answer using the real
  returned data (`x=0.0, y=0.0, z=0.0`, matching the mock stage's actual
  state).

Not yet tested: `backend="sdk"` (real hardware) through this specific
integration - test more on mock first, same as the other two harnesses,
before pointing this at the real stage.
