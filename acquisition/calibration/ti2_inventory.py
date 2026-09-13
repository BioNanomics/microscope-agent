# ti2_inventory.py
# ------------------------------------------------------------
# What this Ti2 reports about its own devices, straight from the SDK.
#
#   python -m acquisition.calibration.ti2_inventory
#   python -m acquisition.calibration.ti2_inventory --json out.json
#   python -m acquisition.calibration.ti2_inventory --columns Name,Value,Control
#
# WHY THIS EXISTS: the Ti2 COM interface exposes ~88 writable i* properties
# covering every device a Ti2 *could* have - TIRF stages, LAPP branches,
# filter wheels, four D-LEDI channels, Intensilight, and so on. A write the
# microscope will not act on is silently ignored: no exception, no error,
# the read-back simply keeps the old value. So write-then-read-back cannot
# tell "device not under Ti2 control" from "value refused" from "write
# worked and something else reset it" - which is the hole a long dark-image
# hunt falls into.
#
# INikonTi2AxMicroscope.Settings is the metadata side of those properties:
# a tuple of INikonTi2AxSetting objects, one per device.
#
#   Control   The field that predicts whether a write will take. Measured
#             against this Ti2-E: Control >= 0 accepted, Control = -1 was
#             refused, for 7 of the 8 devices actually written to during
#             the session that produced this module. The exception was
#             Turret2Shutter (Control=1, refused at the time), so treat
#             this as a strong signal rather than a guarantee.
#   Enabled   NOT a fitted-hardware flag, despite the name: it reads True
#             for all 88 devices, including every one that refuses writes.
#             Read it as "the SDK models this device".
#   Lower/Higher  valid range, so "which iLIGHTPATH values exist?" is a
#                 lookup (1-4) rather than a sweep, and "iDIA_LAMP_Pos =
#                 2100 out of what?" is answerable (0-2100, i.e. maximum).
#   Unit/Scale    units where the SDK declares them (um for the stages).
#
# Field names are discovered from the COM type library at runtime, and rows
# are grouped by the Control value the microscope itself reports - nothing
# about the device set is written down here. That is deliberate: a
# hardcoded property list (OPTICAL_CONFIG_PROPERTIES in nis_sdk.py) is what
# silently dropped every illumination setting from
# get_optical_configuration() in the first place, and a hardcoded list here
# would rot the same way against a different Ti2 or a newer SDK.
#
# Read-only and side-effect free - it moves nothing and changes no state,
# so it is safe to run against live hardware mid-experiment.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
from typing import Any

#: Zero-argument accessor methods on INikonTi2AxSetting that are safe to call.
#: An allowlist rather than "call everything callable", because the same
#: interface exposes mutators (SetLongName/SetShortName) and methods needing
#: arguments (ConvertDev2Phys, ConvertPhys2Dev, GetConvertParams). Plain
#: properties are not listed anywhere - those come from the type library.
_SAFE_ACCESSOR_METHODS = ("LongName", "ShortName")

#: Shown by default. Any field present on the Setting can be asked for with
#: --columns; this is a display choice, not a claim about what exists.
_DEFAULT_COLUMNS = ("Name", "Value", "Lower", "Higher", "Unit", "Enabled")


def _setting_fields(setting: Any) -> list[str]:
    """Field names for one Setting, read from the COM type library.

    win32com's generated wrapper records every readable property of the
    interface in ``_prop_map_get_``, so reading that gives whatever this
    SDK version actually exposes - including fields added after this module
    was written.
    """
    cls = type(setting)
    names = sorted(getattr(cls, "_prop_map_get_", {}) or {})
    names += [m for m in _SAFE_ACCESSOR_METHODS if hasattr(cls, m)]
    return names


def _read_field(setting: Any, name: str) -> Any:
    """Read one field, calling it if it is one of the accessor methods.

    Returns None on failure - one field that raises should not cost us the
    other eleven.
    """
    try:
        value = getattr(setting, name)
    except Exception:
        return None
    if callable(value):
        try:
            value = value()
        except Exception:
            return None
    return value


def _read_settings(microscope: Any) -> list[dict]:
    """Snapshot every Setting. Runs on the COM thread - see inventory()."""
    try:
        settings = microscope.Settings
    except Exception as exc:
        return [{"_error": "%s: %s" % (type(exc).__name__, exc)}]
    rows = []
    for index, setting in enumerate(settings):
        row = {"index": index}
        for name in _setting_fields(setting):
            row[name] = _read_field(setting, name)
        rows.append(row)
    return rows


def inventory(sdk: Any = None) -> list[dict]:
    """Every device Setting the microscope reports, as a list of dicts.

    Pass an existing NISSdk to reuse its COM thread; omit it to build one.
    NISSdk is imported lazily so this module stays importable without
    pywin32 or the Ti2 SDK installed (same pattern as the lazy backend
    imports in mcp_server/loop_tools.py).
    """
    if sdk is None:
        from acquisition.backends.nis_sdk import NISSdk
        sdk = NISSdk()
    return sdk._thread.call(_read_settings)


def _control_label(control: Any) -> str:
    """Section heading for one Control value - see this module's header."""
    if control is None:
        return "Control unknown"
    if control < 0:
        return "Control = %s  (writes refused on every device tested)" % control
    return "Control = %s  (writes took effect)" % control


def format_inventory(rows: list[dict], columns: list[str] | None = None) -> str:
    """Render inventory() grouped by the Control value the microscope reports."""
    if rows and "_error" in rows[0]:
        return "could not read Settings: " + rows[0]["_error"]

    columns = list(columns or _DEFAULT_COLUMNS)
    widths = {
        col: max([len(col)] + [len(str(r.get(col, "") or "")) for r in rows])
        for col in columns
    }

    def row_text(values: dict) -> str:
        return "  " + "  ".join(
            str(values.get(col, "") if values.get(col) is not None else "").ljust(widths[col])
            for col in columns
        )

    lines = []
    for control in sorted({r.get("Control") for r in rows}, key=lambda c: (c is None, c)):
        group = [r for r in rows if r.get("Control") == control]
        lines.append("[%s]  %d devices" % (_control_label(control), len(group)))
        lines.append(row_text({col: col for col in columns}))
        for row in sorted(group, key=lambda r: str(r.get("Name") or "")):
            lines.append(row_text(row))
        lines.append("")

    discovered = sorted(set(rows[0]) - {"index"}) if rows else []
    lines.append("%d devices. Fields discovered from the type library: %s"
                 % (len(rows), ", ".join(discovered)))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory the Ti2's device settings (read-only).")
    parser.add_argument("--json", metavar="PATH",
                        help="also write the full raw inventory to this JSON file")
    parser.add_argument("--columns", metavar="A,B,C",
                        help="comma-separated fields to show instead of the default")
    args = parser.parse_args()

    rows = inventory()
    columns = args.columns.split(",") if args.columns else None
    print(format_inventory(rows, columns=columns))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, default=str)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
