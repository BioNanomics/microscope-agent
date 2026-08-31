# paths.py
# ------------------------------------------------------------
# Where the server writes runtime data (captured frames, move/frame history
# logs). Anchored to the *working directory*, not to __file__ - once this is
# installed as a package, __file__ lives in site-packages and must never be
# written to.
#
#   CONFOCAL_MCP_DATA_DIR   if set, the base directory to use
#   (unset)                 the current working directory
#
# Resolved once at import time, from wherever the server process was launched -
# so launch it from a stable location (or set the env var).
# ------------------------------------------------------------

import os
from pathlib import Path


def data_root() -> Path:
    """Base directory for all runtime data. See module docstring."""
    env = os.environ.get("CONFOCAL_MCP_DATA_DIR")
    return Path(env).expanduser().resolve() if env else Path.cwd()


def captures_dir() -> Path:
    """Directory for saved full-resolution captures."""
    return data_root() / "data" / "captures"


def logs_dir() -> Path:
    """Directory for append-only history logs (move_history, frame_history)."""
    return data_root() / "logs"
