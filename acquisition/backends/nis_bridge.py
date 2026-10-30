# nis_bridge.py
# ------------------------------------------------------------
# Moved to the nis-elements-mcp package: nis_elements_mcp.nis_bridge_client
# (packages/nis-elements-mcp/src/nis_elements_mcp/nis_bridge_client.py).
# This old name stays so existing imports and `python -m acquisition.backends.nis_bridge`
# keep working. It is the same module object, not a copy, so settings
# and test patches made through either name reach the same code.
# ------------------------------------------------------------

import sys

from acquisition import _nis_settings  # noqa: F401  (this repo's log and capture folders)

if __name__ == "__main__":
    import runpy
    runpy.run_module("nis_elements_mcp.nis_bridge_client", run_name="__main__")
else:
    from nis_elements_mcp import nis_bridge_client as _impl
    sys.modules[__name__] = _impl
