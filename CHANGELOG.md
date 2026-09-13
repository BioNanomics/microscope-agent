# Changelog

All notable changes to this project are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
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

### Fixed
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
