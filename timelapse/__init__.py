# timelapse/
# ------------------------------------------------------------
# Offline-testable pieces of the adaptive time-lapse idea:
#
#   change_detector.py  - scores a new frame against a rolling baseline
#                         (no model call, no hardware - numpy + Pillow)
#   frame_audit.py      - audits an existing frame sequence for acquisition
#                         gaps, global intensity jumps and XY shifts, to
#                         separate "filming error" from "biological change"
#   scheduler.py        - the slow-loop / burst-mode acquisition loop that
#                         drives loop_tools.get_image() using the detector
#
# None of this is an MCP tool. The model-facing surface stays at 4 tools
# (see mcp_server/loop_tools.py's header); this package sits *beside* the
# tools and calls get_image() itself.
# ------------------------------------------------------------
