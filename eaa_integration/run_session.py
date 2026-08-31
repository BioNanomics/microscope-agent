# run_session.py
# ------------------------------------------------------------
# Entry point for the EAA (Experiment Automation Agents) integration.
# Everything EAA-related lives in this folder - see REMOVAL.md for how
# to fully remove this integration later.
#
# WHAT THIS DOES: wraps the EXISTING, UNMODIFIED microscope MCP server
# (../mcp_server/server_loop.py) as an EAA tool via MCPTool, and starts
# an EAA BaseTaskManager terminal chat session on top of it. The MCP
# server itself is spawned as a subprocess using the MAIN project's own
# .venv (it needs mcp/PIL/harvesters/pywin32, none of which are
# installed here) - this folder's .venv only holds eaa-core's own
# dependencies (langgraph, openai, etc.), so neither project's
# dependencies bleed into the other's.
#
# LLM BACKEND: OpenAI (EAA has no native Anthropic/Claude client - see
# https://advancedphotonsource.github.io/EAA/llm-config/ - only
# OpenAIConfig, AskSageConfig, ArgoConfig exist). This is a SEPARATE key
# from the main project's ANTHROPIC_API_KEY - set OPENAI_API_KEY in this
# folder's own .env, not the main repo's .env.
#
# REAL-HARDWARE SAFETY: require_approval=True on the MCPTool below gates
# EVERY tool call (get_pos, get_move_history, move, get_image - all of
# them, not just real-hardware ones) behind a human approval prompt.
# This is coarser than the per-call gate in harness/agent.py and
# harness/mcp_agent.py (which only pause for real-hardware actions,
# letting safe mock/read calls through freely) - a deliberate, simpler
# choice for this first version. See docs/mcp_harness.md in the main
# repo for how the finer-grained gate works, if this ever needs to be
# tightened to just real-hardware calls.
#
# Even with require_approval=True, this defaults to backend="mock" -
# the system prompt below tells the model to only use backend="sdk"
# when explicitly asked, same as the other two harnesses.
#
# Run (from this folder, with eaa_integration/.venv activated, and
# OPENAI_API_KEY set in this folder's .env):
#   .venv\Scripts\python run_session.py
# ------------------------------------------------------------

import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from eaa_core.api.llm_config import OpenAIConfig
from eaa_core.gui.html import launch_html_webui_subprocess
from eaa_core.task_manager.base import BaseTaskManager
from eaa_core.tool.mcp_client import MCPTool

WEBUI_HOST = "127.0.0.1"
WEBUI_RUNTIME_PORT = 8010  # backend API the browser app talks to
WEBUI_BROWSER_PORT = 8008  # what you actually open in a browser

THIS_DIR = Path(__file__).resolve().parent
MAIN_REPO_ROOT = THIS_DIR.parent
MAIN_VENV_PYTHON = MAIN_REPO_ROOT / ".venv" / "Scripts" / "python.exe"

load_dotenv(dotenv_path=THIS_DIR / ".env")

# Change this to whatever model your OpenAI key actually has access to.
MODEL = "gpt-4o-mini"

CONFOCAL_SAFETY_NOTE = """\
You are controlling a Nikon Ti2-E microscope stage and Baumer camera
through an MCP server exposing get_image, get_pos, move, and
get_move_history. Always default to backend="mock" unless the user has
clearly asked you to control the real physical microscope - only use
backend="sdk" (or call get_image, which is always real hardware) when
that's explicitly intended. Every single tool call here pauses for a
human approval before executing, regardless of backend - expect that,
and explain what you're about to do before calling a tool so the
approval makes sense to whoever is watching. move() only accepts XY -
there is no way to move Z through this tool surface, by design.
"""


# ------------------------------------------------------------
# Shutdown handling
# ------------------------------------------------------------
# Why this is not just "let main() return":
#
#  1. EAA's graph loop CATCHES KeyboardInterrupt itself (see
#     BaseTaskManager.invoke_graph_with_interruption_recovery) and turns
#     it into "interrupt this turn, go back to the prompt" - so a single
#     Ctrl+C never reaches us and never exits. The clean way out is the
#     `/exit` command (type it in the terminal, or in the browser box in
#     --webui mode).
#
#  2. On ANY exit path, EAA's MCPTool.__del__ runs a fresh
#     loop.run_until_complete(disconnect()) at interpreter shutdown. On
#     Windows that deadlocks inside anyio's task-group __aexit__
#     (CancelledError awaiting _on_completed_fut) - the hang that makes
#     this process impossible to close normally. os._exit() below skips
#     __del__/atexit entirely, so that code never runs.
#
#  3. The MCP server is spawned through the main project's *virtualenv*
#     python.exe, a ~270 KB redirector stub that re-execs the real
#     interpreter as a child. mcp's stdio client only ever signals the
#     stub, orphaning the real server_loop.py (and the camera / NIS SDK
#     handles it holds). --webui's HTML server subprocess has the same
#     shape. taskkill /T walks the whole tree before we go.

_last_sigint = [float("-inf")]


def _terminate_child_processes(timeout: float = 10.0) -> None:
    """Kill every process this session spawned, whole tree, best effort."""
    if sys.platform != "win32":
        return
    me = os.getpid()
    try:
        listing = subprocess.run(
            [
                "powershell", "-NoProfile", "-NonInteractive", "-Command",
                f"(Get-CimInstance Win32_Process -Filter 'ParentProcessId={me}')"
                ".ProcessId -join ' '",
            ],
            capture_output=True, text=True, timeout=timeout,
        ).stdout
    except Exception:
        listing = ""
    for token in listing.split():
        if token.isdigit() and token != str(me):
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", token],
                    capture_output=True, timeout=timeout,
                )
            except Exception:
                pass


def _hard_exit(code: int = 0) -> None:
    """Kill children, then os._exit past EAA's hanging __del__ cleanup."""
    _terminate_child_processes()
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


def _install_interrupt_handler(hard_on_first: bool) -> None:
    """Route Ctrl+C to a real exit.

    hard_on_first=True (--webui): the terminal is the only control
    surface, so one Ctrl+C force-quits immediately.

    hard_on_first=False (terminal): the first Ctrl+C still passes through
    to EAA (interrupt the current turn); a second within 3 s force-quits.
    """

    def handler(signum, frame):
        now = time.monotonic()
        if hard_on_first or (now - _last_sigint[0]) < 3.0:
            print(
                "\nForce-quitting - killing the microscope MCP server and exiting.",
                file=sys.stderr,
            )
            _hard_exit(130)
        _last_sigint[0] = now
        print(
            "\nInterrupting current turn. Press Ctrl+C again within 3 s to force-quit "
            "(or type /exit).",
            file=sys.stderr,
        )
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handler)
    if hasattr(signal, "SIGBREAK"):  # Windows Ctrl+Break
        signal.signal(signal.SIGBREAK, handler)


class ConfocalTaskManager(BaseTaskManager):
    """BaseTaskManager with the microscope safety note appended to EAA's
    own default system prompt - NOT a replacement of it. Overriding this
    method (rather than passing a constructor kwarg) is deliberate:
    assistant_system_message is resolved once, early in
    BaseTaskManager.__init__ (see base.py), by calling
    self.get_default_system_prompt() if it isn't already set - Python's
    normal method dispatch means that call reaches this override
    automatically, so EAA's own prompt template (which includes its
    skills-formatting logic) is preserved and just extended, not
    discarded.
    """

    def get_default_system_prompt(self) -> str:
        base_prompt = super().get_default_system_prompt()
        return f"{base_prompt}\n\n{CONFOCAL_SAFETY_NOTE}"


def _run_session(task_manager: "ConfocalTaskManager", use_webui: bool) -> None:
    """Own the conversation loop. Returns only on a clean `/exit`."""
    if not use_webui:
        task_manager.run_conversation()
        return

    # WebUI mode: start the backend runtime API, then the browser-facing
    # static app as a non-blocking subprocess (so this process can still
    # run the actual conversation graph) - the browser is only a remote
    # frontend; task_manager.run_conversation() still owns the loop.
    task_manager.start_webui_runtime()
    webui_process = launch_html_webui_subprocess(
        runtime_url=f"http://{WEBUI_HOST}:{WEBUI_RUNTIME_PORT}",
        host=WEBUI_HOST,
        port=WEBUI_BROWSER_PORT,
        title="Confocal Microscope Agent",
    )
    print(f"\nWebUI running - open http://{WEBUI_HOST}:{WEBUI_BROWSER_PORT} in your browser.")
    print("All messages and tool approvals now happen in the browser, not this terminal.\n")
    try:
        task_manager.run_conversation()
    finally:
        webui_process.terminate()


def main() -> None:
    use_webui = "--webui" in sys.argv
    # One Ctrl+C force-quits in --webui mode (the terminal is the only
    # control surface there); in terminal mode a second Ctrl+C within 3 s
    # does, and the first is still EAA's "interrupt this turn".
    _install_interrupt_handler(hard_on_first=use_webui)
    # In WebUI mode, EAA routes ALL user input (messages AND tool
    # approvals) through the browser instead of the terminal - confirmed
    # directly against BaseTaskManager.get_user_input(): when a
    # runtime_controller exists (i.e. use_webui=True), it calls
    # runtime_controller.request_input(...) and never touches
    # terminal input() at all. --webui is opt-in (not the default) so
    # the already-verified terminal flow keeps working unchanged unless
    # you explicitly ask for the dashboard.
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set. Fill it in at "
            f"{THIS_DIR / '.env'} (uncomment the line, add your real key) "
            "and try again.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not MAIN_VENV_PYTHON.exists():
        print(
            f"Main project venv not found at {MAIN_VENV_PYTHON} - the main "
            "repo's .venv must exist and have its dependencies installed "
            "(see ../requirements.txt) before this will work.",
            file=sys.stderr,
        )
        sys.exit(1)

    llm_config = OpenAIConfig(
        model=MODEL,
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
    )

    mcp_tool = MCPTool(
        {
            "mcpServers": {
                "confocal": {
                    "command": str(MAIN_VENV_PYTHON),
                    "args": ["-m", "mcp_server.server_loop"],
                    "cwd": str(MAIN_REPO_ROOT),
                }
            }
        },
        require_approval=True,
    )

    task_manager = ConfocalTaskManager(
        llm_config=llm_config,
        tools=[mcp_tool],
        skill_dirs=[],
        name="confocal_agent",
        checkpoint_db_path=str(THIS_DIR / "checkpoint.sqlite"),
        transcript_db_path=str(THIS_DIR / "transcript.sqlite"),
        use_webui=use_webui,
        webui_runtime_host=WEBUI_HOST,
        webui_runtime_port=WEBUI_RUNTIME_PORT,
    )

    # BaseTaskManager auto-registers a set of built-in tools (bash/python
    # execution, file read/write, uv, subagents, ...) alongside whatever
    # you pass via tools= - confirmed by inspecting
    # tool_executor.list_tool_schemas() directly, not assumed. This
    # integration should only ever expose the 4 microscope tools (wrapped
    # via mcp_tool above), so disable every built-in explicitly. This
    # also fixes a real crash: several of those built-ins use dotted
    # names (e.g. "simple_python_eval_tool.evaluate_python_expression"),
    # which OpenAI's function-calling API rejects outright (names must
    # match ^[a-zA-Z0-9_-]+$, no dots) - our own 4 tools are named
    # cleanly and were never the problem.
    task_manager.tool_manager.disable_bash_coding_tool()
    task_manager.tool_manager.disable_python_coding_tool()
    task_manager.tool_manager.disable_simple_python_eval_tool()
    task_manager.tool_manager.disable_file_system_tool()
    task_manager.tool_manager.disable_image_rendering_tool()
    task_manager.tool_manager.disable_uv_tool()
    task_manager.tool_manager.disable_subagent_tool()
    task_manager.tool_manager.disable_workspace_tool()
    task_manager.tool_manager.disable_image_captioning()

    remaining = [s["function"]["name"] for s in task_manager.tool_executor.list_tool_schemas()]
    assert remaining == ["get_image", "get_pos", "move", "get_move_history"], (
        f"Expected only the 4 microscope tools, but tool list is: {remaining}"
    )

    try:
        _run_session(task_manager, use_webui)
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        # Always the last thing: kill the MCP server / WebUI subprocess
        # trees and os._exit past EAA's deadlocking __del__ cleanup.
        _hard_exit(0)


if __name__ == "__main__":
    main()
