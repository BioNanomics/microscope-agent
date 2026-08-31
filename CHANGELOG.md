# Changelog

All notable changes to this project are recorded here.
Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
