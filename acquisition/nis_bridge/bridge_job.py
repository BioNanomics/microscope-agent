# bridge_job.py
# ------------------------------------------------------------
# Moved to the nis-elements-mcp package: nis_elements_mcp.nis_bridge_job
# (packages/nis-elements-mcp/src/nis_elements_mcp/nis_bridge_job.py).
# This old name stays so existing imports and `python -m acquisition.nis_bridge.bridge_job`
# keep working. It is the same module object, not a copy, so settings
# and test patches made through either name reach the same code.
#
# DO NOT PASTE THIS FILE INTO NIS. Paste the package file instead:
#   packages/nis-elements-mcp/src/nis_elements_mcp/nis_bridge_job.py
# ------------------------------------------------------------

import sys

if __name__ == "__main__":
    import runpy
    runpy.run_module("nis_elements_mcp.nis_bridge_job", run_name="__main__")
else:
    from nis_elements_mcp import nis_bridge_job as _impl
    sys.modules[__name__] = _impl
