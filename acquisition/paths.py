# paths.py
# ------------------------------------------------------------
# Where the server writes runtime data (captured frames, move/frame history
# logs, saved stage positions). Anchored to the *working directory*, not to
# __file__ - once this is installed as a package, __file__ lives in
# site-packages and must never be written to.
#
#   CONFOCAL_MCP_DATA_DIR   if set, the base directory to use - always
#                           honoured as-is, no second-guessing
#   (unset)                 the current working directory, if it is a usable
#                           place to write; otherwise FALLBACK_DATA_ROOT
#
# WHY THE CWD IS NOT TRUSTED BLINDLY: an MCP client picks the working
# directory its servers are launched with, and the caller has no say in it.
# Claude Desktop on Windows launches them in C:\WINDOWS\system32, so a plain
# Path.cwd() sends captures to C:\WINDOWS\system32\data\captures - which
# fails with a bare "access is denied" OSError at the first capture, a long
# way from this module and looking for all the world like a camera fault.
# Running elevated is worse than the error: the write then *succeeds*, and
# scatters image files through a system directory.
#
# So an unset env var resolves the cwd through _usable_data_root(), which
# rejects anything under the Windows directory and anything it cannot
# actually create a file in, and falls back to a per-user directory with a
# warning on stderr (never stdout - that is the JSON-RPC channel).
#
# Resolved once per process and cached: every caller reads it into a
# module-level constant at import time anyway. Tests that manipulate the env
# var must call data_root.cache_clear() to see the change.
# ------------------------------------------------------------

import functools
import os
import sys
import tempfile
from pathlib import Path

#: Used when the working directory is not a usable data root. Under the
#: user's home, so it is writable without elevation and easy to find.
FALLBACK_DATA_ROOT = Path.home() / ".confocal-mcp"


def _is_system_dir(path: Path) -> bool:
    """True if `path` is inside the Windows directory.

    Checked separately from writability because an elevated process *can*
    write to the Windows system directory - that is precisely the case
    worth refusing rather than permitting.
    """
    system_root = os.environ.get("SystemRoot")
    if not system_root:
        return False
    try:
        return path == Path(system_root) or Path(system_root) in path.parents
    except OSError:
        return False


def _is_writable(path: Path) -> bool:
    """True if a file can actually be created in `path`.

    Probes with a real file rather than os.access(), which on Windows
    reports only the read-only attribute and happily returns True for
    directories that ACLs deny.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path, prefix=".confocal-write-test-"):
            pass
        return True
    except OSError:
        return False


def _usable_data_root(cwd: Path) -> Path:
    """`cwd` if it is a sane place to write runtime data, else the fallback."""
    if not _is_system_dir(cwd) and _is_writable(cwd):
        return cwd

    print(
        f"[confocal-mcp] Working directory {cwd} is not usable for runtime data "
        f"(system directory or not writable); using {FALLBACK_DATA_ROOT} instead. "
        f"Set CONFOCAL_MCP_DATA_DIR to choose the location explicitly.",
        file=sys.stderr,
    )
    return FALLBACK_DATA_ROOT


@functools.lru_cache(maxsize=1)
def data_root() -> Path:
    """Base directory for all runtime data. See module docstring."""
    env = os.environ.get("CONFOCAL_MCP_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _usable_data_root(Path.cwd())


def captures_dir() -> Path:
    """Directory for saved full-resolution captures."""
    return data_root() / "data" / "captures"


def logs_dir() -> Path:
    """Directory for append-only history logs (move_history, frame_history)."""
    return data_root() / "logs"
