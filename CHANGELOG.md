# Changelog

All notable changes to this project are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- NIS bridge (not yet run inside NIS - see `docs/nis-bridge.md`):
  `acquisition/nis_bridge/bridge_job.py`, a JOBS Python task that serves NIS
  macro calls (status, relative XY move, 4x/10x objective change, ND2
  capture, saved ND experiment run/finish) on 127.0.0.1:8766;
  `acquisition/backends/nis_bridge.py`, its client and command line; and
  `mcp_server/server_nis.py` (`confocal-mcp-nis`), a separate MCP server
  with confirm-gated tools. The e-stop is enforced in both client and
  bridge; `server_loop` is unchanged.
- `acquisition/calibration/nis_port_probe.py` - minimal JOBS task showing
  that a port bound inside `nis_ar.exe` is reachable from outside.
- `analysis/make_soundtrack.py` - original ambient soundtrack synthesised
  with numpy (no samples, so no licensing questions), stretched to any
  length. `--movie run.mp4` sizes it to the movie and muxes it on as AAC
  with the video stream copied, writing `run_music.mp4`. At 39 s it
  reproduces the Physarum Short's track exactly.
- `acquisition/calibration/ti2_inventory.py` - read-only inventory of every
  device the Ti2 reports, with each one's valid range, unit, and `Control`
  value. Field names come from the COM type library at runtime and rows are
  grouped by the microscope's own `Control` value, so there is no device list
  to keep in sync. `Control` is the practical guide to what can be driven:
  `-1` devices (the D-LEDI channels, the DIA/EPI/AUX shutters, Intensilight,
  TIRF, LAPP) reliably ignore writes; `>= 0` usually accepts, with measured
  exceptions (`iDIC_PRISM`, `iTURRET2SHUTTER`). Note `Enabled` is *not* a
  fitted-hardware flag - it reads True for all 88 devices.
- `acquisition/calibration/ti2_config.py` - `save` / `show` / `diff` / `apply`
  for the full device configuration from the command line. `apply` requires
  `--confirm`, skips stage/objective/TIRF unless `--include-motion`, and polls
  the read-back rather than reading once (several devices report the old value
  for up to a second after a successful write).
- `get_image(backend=...)`: `backend="mock"` returns a simulated frame via
  `MockNIS.capture()` with no camera attached and no `confirm`, so capture
  logic can be developed off the microscope PC. `backend="sdk"` (the
  default) is unchanged and still requires `confirm=True`. The metadata dict
  now carries `backend`. Both harnesses skip the hardware gate for mock
  captures.
- `CONFOCAL_MOCK_FRAME_PATH`: PNG the mock capture serves (defaults to the
  old `data/analysis/nd2_sample/frame_0.png` location).
- `timelapse/` package (not MCP tools): `change_detector` (model-free
  per-frame change score), `frame_audit` (CLI: gaps / intensity jumps /
  stage shifts vs. specimen change in an existing sequence), `scheduler`
  (adaptive slow/burst acquisition loop with hard caps and a one-time
  real-hardware approval), `model_trigger` (Claude behind the scheduler's
  trigger hook: extend or end a burst from the before/after frames, with
  call caps and fail-safe "no opinion"; runs on a background thread so
  burst timing never waits on the model). See `docs/adaptive_timelapse.md`.
- `tests/` (pytest, mock only) and a GitHub Actions CI workflow. Includes a
  protocol-level test that spawns `mcp_server.server_loop` as a subprocess,
  asserts the exact tool set, and drives every tool over MCP stdio. Also
  tests of both harnesses' real-hardware approval gate: no `confirm` in any
  model-facing schema, sdk calls stop at the gate and a decline executes
  nothing, mock calls never prompt. And tests of `harness/context.py`'s
  image pruning.
- `numpy` (2.4.x, the last line that supports Python 3.11) is now a core dependency; new `test` optional group (pytest).

### Fixed
- README and code comments said the MCP surface was 4 tools; it has been 5
  since `estop` was added. The README's tool list now includes `estop`.
- `get_optical_configuration()` recorded no illumination state. It walked
  `OPTICAL_CONFIG_PROPERTIES`, a hardcoded list that omitted
  `iDIA_LAMP_Switch`/`iDIA_LAMP_Pos` entirely - so a saved configuration never
  captured whether the transmitted lamp was on, the one setting that decides
  whether a camera on the camera port sees anything. The `iDLED*` names it did
  list are ignored by this microscope, whose D-LEDI is not driven through the
  Ti2 body. Both it and `apply_optical_configuration()` now use the SDK's own
  `DataGet`, which returns all ~90 properties in one call, and the hardcoded
  list is gone. `apply_optical_configuration()` gains `include_motion=False`:
  the snapshot is now the full device set, so without that guard restoring a
  lamp setting would also drive the stage.

- Runtime data no longer falls back to an unwritable working directory. MCP
  clients choose the working directory their servers are launched with, and
  Claude Desktop on Windows uses `C:\WINDOWS\system32` - so with
  `CONFOCAL_MCP_DATA_DIR` unset, the first `get_image()` failed with an
  "access is denied" `OSError` that read as a camera fault, and under an
  elevated process would instead have written captures into a system
  directory. `acquisition/paths.py` now rejects a working directory that is
  inside the Windows directory or that it cannot create a file in, falling
  back to `~/.confocal-mcp` with a warning on stderr. Setting
  `CONFOCAL_MCP_DATA_DIR` is still honoured as-is, and a normal source
  checkout still resolves to the checkout directory.
- The `analysis/` scripts imported `scipy`, `scikit-image`, `nd2`,
  `imageio-ffmpeg` and `opencv-python`, none of which the package declared, so
  a fresh install could not run them. They are now the `analysis` optional
  group, which `all` (and so `requirements.txt`) includes.
- `python -m acquisition.orchestration.stage_positions` ended by loading
  `protocols/example_protocol.yaml`, which is not in this repo, so the demo
  always failed. That step is removed. Code comments that pointed to files in
  ConfocalOrchestrator (`docs/microscope-notes.md`, `run_protocol.py`, the
  calibration tests) now say so, and those citing old names
  (`mcp_server/server.py`, `acquisition_tools.py`) use the current ones.

## [0.1.0] - 2026-08-31

First packaged release.

### Added
- Installable package `confocal-mcp` with a `confocal-mcp` console entry point
  (`mcp_server.server_loop:main`). MCP client config becomes
  `{"command": "confocal-mcp"}` after `pip install` / `uv tool install` /
  `pipx install`.
- Optional dependency groups: `camera` (harvesters, genicam, opencv), `sdk`
  (pywin32), `harness` (anthropic, python-dotenv), `all`. A core install
  (server + mock stage) pulls no hardware dependencies and runs anywhere.
- `CONFOCAL_MCP_DATA_DIR` environment variable. Without it, runtime data
  (captures, move/frame history, saved positions) is written under the current
  working directory instead of next to the installed package
  (`acquisition/paths.py`).
- `mcp_server.__version__`, resolved from installed package metadata.
- MIT license.

### Changed
- The camera stack (`cv2` + `harvesters`) is imported lazily in
  `loop_tools._get_camera()`, so `import mcp_server.loop_tools` works on a
  core-only install - matching the existing lazy import of `nis_sdk`.
- `BaumerGenICam.capture()` dispatches on the camera's declared `PixelFormat`
  (BGR8 / RGB8 / Mono8 / Bayer*) instead of assuming BayerRG8, which crashed
  whenever the camera was left in a 3-channel format.
- `requirements.txt` is now `-e .[all]`; exact pins live in `pyproject.toml`.

---

The MCP surface itself - `get_pos`, `move` (XY only), `get_image`,
`get_move_history`, with `backend` = `mock` | `sdk` and a `confirm=True` gate on
anything that touches real hardware - predates this changelog. See `README.md`.

[Unreleased]: https://github.com/BioNanomics/microscope-agent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/BioNanomics/microscope-agent/releases/tag/v0.1.0
