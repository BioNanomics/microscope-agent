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
- **`get_image(confirm, exposure_time_us, gain)`** - capture a real frame
  from the Baumer camera, paired with the exact stage position it was
  taken at. Returns both the metadata and an embedded image preview, so
  the model can actually see the picture, not just a file path.

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

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

One extra step on a new machine: `acquisition/backends/nis_sdk.py`
imports `NkTi2Ax`, the Nikon Ti2 SDK's generated Python bindings - this
file is machine-generated (via `pywin32`'s `gencache`/`makepy` against
the installed SDK), not pip-installable, and isn't included in this
repo. If it doesn't already exist in `.venv/Lib/site-packages/`, either
let it regenerate against an installed Ti2 SDK, or copy it from another
working `.venv` on the same machine.

## Run

```
python -m mcp_server.server_loop
```

Then point an MCP client (Claude Desktop's `claude_desktop_config.json`,
or a project-level `.mcp.json` for Claude Code) at this command.

## Hardware

- **Stage/focus**: Nikon Ti2-E via the Ti2 ActiveX SDK (`nis_sdk.py`) -
  real hardware, no NIS-Elements process required.
- **Camera**: Baumer VCXU-23C via GenICam/GenTL (`baumer_genicam.py`) -
  a separate industrial camera, not the confocal N-SPARC detector.
  Only one application can hold it open at a time - close Baumer Camera
  Explorer (or any other GenICam consumer) before running this.