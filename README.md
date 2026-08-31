# microscope-agent

A minimal MCP server for controlling a real Nikon Ti2-E microscope + Baumer
GenICam camera from an AI agent (Claude Desktop, Claude Code, or any other
MCP client).

## Tools

Exactly 4, by design - kept deliberately minimal:

- **`get_pos(backend)`** - current stage (x, y, z) position, in microns.
  A cheap, on-demand sync primitive, not something to call before every
  move (`get_image`/`move` already return position as part of their result).
- **`move(x, y, backend, confirm)`** - move the XY stage to an absolute
  position, returns the *actual* resulting position. XY-only, on purpose
  (no absolute Z move is reachable from chat - crash risk into the sample).
- **`get_image(confirm, exposure_time_us, gain, crop, max_dimension)`** -
  capture a real frame from the Baumer camera, paired with the exact
  stage position it was taken at. Returns both the metadata and an
  embedded image preview, so the model can actually see the picture, not
  just a file path. `crop` (optional `{"x","y","width","height"}`
  fractions of the full frame) restricts the embedded preview to a
  region of interest instead of resending the whole frame; the full-res
  file on disk is always uncropped. `max_dimension` overrides the
  preview's default long-edge cap for that call.

`get_pos`/`move`/`get_image` all return `stage_revision`, a monotonic
counter bumped whenever a call observes the stage at a different
position than the last one this process saw - so a model holding an
older result can tell "has the stage moved since then" (e.g. someone
touched the joystick) without diffing raw coordinates itself.
`get_image` additionally returns `frame_id`, a per-process counter
identifying that specific capture.
- **`get_move_history(limit)`** - every point `move()` has actually sent
  the stage to this session, so "where have we already been" doesn't
  need to be re-derived from conversation history.

`backend` is `"mock"` (default, safe, simulated) or `"sdk"` (real
hardware - `move()`/`get_image()` require `confirm=True` for anything
that touches real hardware). See `mcp_server/loop_tools.py`'s header
comment for the full design rationale (why no `get_time()`, why no
`stage_revision` yet, etc.).

No NIS-Elements involved anywhere - the camera is reached directly via
GenICam/GenTL, and the stage via the Ti2 ActiveX SDK, both independent
of whether NIS-Elements software is even running.

## Install

This is a normal installable package (`confocal-mcp`) with a console entry
point. Dependencies are split into groups so a machine only pulls what it needs:

| Group | Pulls in | For |
|---|---|---|
| *(core)* | `mcp`, `Pillow`, `PyYAML` | the MCP server against the **mock** stage |
| `camera` | `harvesters`, `genicam`, `opencv-python` | real Baumer camera capture |
| `sdk` | `pywin32` | real Ti2 stage control (Windows) |
| `harness` | `anthropic`, `python-dotenv` | the standalone Claude loops in `harness/` |
| `all` | everything above | a full workstation |

The real hardware backends are imported lazily, so a core-only install runs
fine anywhere (CI, a laptop) - it just can't touch hardware.

### From a source checkout

```
python -m venv .venv
.venv\Scripts\pip install -e ".[camera,sdk]"     # real hardware
.venv\Scripts\pip install -r requirements.txt    # == -e ".[all]"
```

### As a standalone tool (isolated, no repo checkout)

```
uv tool install "git+https://github.com/BioNanomics/microscope-agent[camera,sdk]"
# or: pipx install "git+https://github.com/BioNanomics/microscope-agent[camera,sdk]"
```

### Real stage backend (one extra step)

`acquisition/backends/nis_sdk.py` imports `NkTi2Ax`, the Nikon Ti2 SDK's
generated Python bindings - machine-generated (via `pywin32`'s
`gencache`/`makepy` against the installed SDK), not pip-installable, and not in
this repo. If it isn't already in `site-packages/`, let it regenerate against an
installed Ti2 SDK, or copy it from another working environment on the same
machine. Not needed for `backend="mock"`.

## Run

```
confocal-mcp                       # installed console script
python -m mcp_server.server_loop   # equivalent, from a source checkout
```

Point an MCP client at it - Claude Desktop's `claude_desktop_config.json`, or a
project-level `.mcp.json` for Claude Code:

```json
{ "mcpServers": { "confocal": { "command": "confocal-mcp" } } }
```

Data (captures, move/frame history logs) is written under the current working
directory when the server runs from an installed package - launch it from a
stable location.

## Harness loop (`harness/agent.py`)

A second, independent way to drive the same tools - a small interactive
CLI that calls Claude (Anthropic API) directly with tool use, importing
`mcp_server/loop_tools.py`'s functions in-process rather than going
through `server_loop.py`'s MCP/stdio protocol. `server_loop.py` is
unaffected either way - use whichever fits: MCP for Claude Desktop/Code,
this loop for a standalone script.

Requires `ANTHROPIC_API_KEY` set - either in a `.env` file at the repo
root (copy the commented-out line in `.env`, fill in your real key; this
file is gitignored and loaded automatically by `harness/agent.py`), as a
regular environment variable, or via `ant auth login`. Real hardware
(`backend="sdk"`, or any `get_image` call) always pauses for a live
"y/N" approval at the terminal before executing, regardless of what the
model requests - see `harness/agent.py`'s header comment for why. Old
captured images are pruned from the model's context after a couple of
turns (`harness/context.py`) so a long session doesn't keep resending
every frame it has ever captured.

```
python -m harness.agent
```

## MCP client harness (`harness/mcp_agent.py`)

A third way to drive the same tools - the real-MCP-protocol counterpart
to `harness/agent.py`. Instead of importing `loop_tools.py` in-process,
this one spawns `server_loop.py` as a separate subprocess and talks to
it exactly the way Claude Desktop/Code do: real MCP over stdio, using
the current MCP SDK (`mcp==2.0.0`, protocol version `2026-07-28`). Same
`.env`/`ANTHROPIC_API_KEY` setup, same live "y/N" real-hardware approval
gate, same image pruning via `harness/context.py` - just reached through
an actual client/server boundary instead of a direct function call.

```
python -m harness.mcp_agent
```

See **`docs/mcp_harness.md`** for the full write-up: how the current MCP
protocol differs from what most tutorials show, why `confirm` is
stripped from every tool schema before Claude ever sees it, the MCP↔
Anthropic content-block conversion, and a real stdout/JSON-RPC bug this
work found and fixed in `nis_mock.py`.

## Hardware

- **Stage/focus**: Nikon Ti2-E via the Ti2 ActiveX SDK (`nis_sdk.py`) -
  real hardware, no NIS-Elements process required.
- **Camera**: Baumer VCXU-23C via GenICam/GenTL (`baumer_genicam.py`) -
  a separate industrial camera, not the confocal N-SPARC detector.
  Only one application can hold it open at a time - close Baumer Camera
  Explorer (or any other GenICam consumer) before running this.

## Versioning

Versions follow [SemVer](https://semver.org/); see `CHANGELOG.md`. The current
version is importable as `mcp_server.__version__` (from installed package
metadata).

## License

MIT - see `LICENSE`.