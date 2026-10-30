# estop_panel.py
# ------------------------------------------------------------
# Moved to the nis-elements-mcp package: nis_elements_mcp.estop_panel
# (packages/nis-elements-mcp/src/nis_elements_mcp/estop_panel.py).
# This old name stays so existing imports and `python -m acquisition.estop_panel`
# keep working. It is the same module object, not a copy, so settings
# and test patches made through either name reach the same code.
# ------------------------------------------------------------

import sys

from acquisition import _nis_settings  # noqa: F401  (this repo's settings; moves an old e-stop)

if __name__ == "__main__":
    import runpy
    runpy.run_module("nis_elements_mcp.estop_panel", run_name="__main__")
else:
    from nis_elements_mcp import estop_panel as _impl
    sys.modules[__name__] = _impl
