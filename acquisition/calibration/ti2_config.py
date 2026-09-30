# ti2_config.py
# ------------------------------------------------------------
# Save and restore the Ti2's device configuration from the command line.
#
#   python -m acquisition.calibration.ti2_config save
#   python -m acquisition.calibration.ti2_config save --out protocols/brightfield.json
#   python -m acquisition.calibration.ti2_config show protocols/brightfield.json
#   python -m acquisition.calibration.ti2_config diff protocols/brightfield.json
#   python -m acquisition.calibration.ti2_config apply protocols/brightfield.json --confirm
#
# HOW THE SNAPSHOT IS TAKEN: INikonTi2AxMicroscope.DataGet fills an
# INikonTi2AxData object with all 90 device properties in one COM call, and
# it uses the writable i* names (iLIGHTPATH, iDIA_LAMP_Switch, ...) - so a
# saved file can be written straight back with setattr, no name mapping.
#
#   data = microscope.DataGet(None, 0)
#
# Pass None for the first argument: it is declared [in,out] (VT_BYREF|
# VT_DISPATCH) and win32com returns the filled object. Do not try to
# construct the object yourself - the NikonTi2AxData coclass is not
# registered on this workstation (CoCreateInstance gives "Class not
# registered"), which is a dead end DataGet sidesteps entirely.
#
# WHY NOT NISSdk.get_optical_configuration(): it loops over
# OPTICAL_CONFIG_PROPERTIES, a hand-maintained list in nis_sdk.py that
# omits iDIA_LAMP_Switch and iDIA_LAMP_Pos. A configuration saved through
# it does not record whether the transmitted lamp was on - the one setting
# that decides whether the camera sees anything at all.
#
# WHY NOT DataSet FOR RESTORE: DataSet(newVal, ...) writes the whole device
# set in one call, so restoring a lamp setting would also drive the stage,
# Z and the nosepiece as an unavoidable side effect. apply() writes
# per-property instead, which allows skipping motion devices and reporting
# each write individually. --bulk opts into DataSet when that is what you
# actually want.
#
# Control: each Setting reports a Control value. It is a useful filter but
# NOT a guarantee, and the asymmetry matters:
#
#   Control = -1  reliably refuses. The D-LEDI channels, the DIA/EPI/AUX
#                 shutters, Intensilight, TIRF and LAPP all sit here, and
#                 every write to them was silently ignored. Worth skipping
#                 rather than issuing writes that vanish without an error.
#   Control >= 0  usually accepts, but not always. Measured exceptions on
#                 this Ti2-E: iDIC_PRISM (Control=0) and iTURRET2SHUTTER
#                 (Control=1) both refuse writes that never take, even
#                 given seconds to settle. Something - an interlock, or
#                 NIS-Elements holding those devices - overrides them.
#
# So treat a Control >= 0 "not-applied" as real, and check the device
# rather than assuming the report is a timing artifact.
#
# SAFETY: save/show/diff are read-only. apply moves real hardware, needs
# --confirm, and skips the motion devices unless --include-motion.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import datetime
import json
import re
import time
from pathlib import Path
from typing import Any

from acquisition.calibration.ti2_inventory import inventory

#: Some devices do not report a new value the instant the write returns -
#: the DIC prism and turret shutters are mechanical, the DIA lamp settles.
#: Reading back immediately made apply report DiaLampPos, DicPrism and
#: Turret2Shutter as failures when all three had in fact applied. Poll
#: instead of trusting one immediate read.
SETTLE_SECONDS = 1.5
SETTLE_POLL_SECONDS = 0.25

#: Devices that physically move something substantial; skipped by apply
#: unless --include-motion. Matched as substrings of the i* property name.
_MOTION_PROPERTIES = ("XPOSITION", "YPOSITION", "ZPOSITION", "ZEscape",
                      "ZReset", "XReset", "YReset", "NOSEPIECE", "Tirf")


def _is_motion(prop: str) -> bool:
    return any(key.lower() in prop.lower() for key in _MOTION_PROPERTIES)


def _normalise(name: str) -> str:
    """Fold a property name to a comparable key.

    Only needed to join DataGet's i* names to the friendly names Settings
    reports (iDIA_LAMP_Switch <-> DiaLampSwitch), so that a snapshot can
    carry each property's Control value and valid range alongside it.
    """
    stripped = name[1:] if name.startswith("i") else name
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def _data_properties(data: Any) -> list[str]:
    """Property names on the INikonTi2AxData object, from the type library."""
    return sorted(getattr(type(data), "_prop_map_get_", {}) or {})


def save_config(sdk: Any = None) -> dict:
    """Snapshot every device property via DataGet, plus Settings metadata.

    Read-only. Values come from one DataGet call; Control/range/unit are
    joined on from INikonTi2AxMicroscope.Settings so the saved file records
    what can be written back and within what limits.
    """
    if sdk is None:
        from acquisition.backends.nis_sdk import NISSdk
        sdk = NISSdk()

    def snapshot(microscope: Any) -> dict:
        data = microscope.DataGet(None, 0)
        values = {}
        for prop in _data_properties(data):
            try:
                values[prop] = getattr(data, prop)
            except Exception:
                continue
        return values

    values = sdk._thread.call(snapshot)
    meta = {_normalise(str(row.get("Name"))): row for row in inventory(sdk) if row.get("Name")}

    properties = {}
    for prop, value in sorted(values.items()):
        row = meta.get(_normalise(prop), {})
        properties[prop] = {
            "value": value,
            "control": row.get("Control"),
            "lower": row.get("Lower"),
            "higher": row.get("Higher"),
            "unit": row.get("Unit"),
            "friendly_name": row.get("Name"),
        }
    return {
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "property_count": len(properties),
        "properties": properties,
    }


def _writable(entry: dict) -> bool:
    control = entry.get("control")
    return control is not None and control >= 0


def format_config(config: dict, only_writable: bool = False) -> str:
    props = config.get("properties", {})
    lines = ["saved_at: %s   (%d properties)" % (config.get("saved_at", "?"), len(props)), ""]
    lines.append("  %-24s %10s  %8s  %s" % ("PROPERTY", "VALUE", "CONTROL", "RANGE"))
    for prop, entry in sorted(props.items()):
        if only_writable and not _writable(entry):
            continue
        rng = "" if entry.get("lower") is None else "%s .. %s" % (entry["lower"], entry["higher"])
        lines.append("  %-24s %10s  %8s  %s"
                     % (prop, entry.get("value"), entry.get("control"), rng))
    return "\n".join(lines)


def diff_config(config: dict, sdk: Any = None) -> list[dict]:
    """Properties whose live value differs from the saved one. Read-only."""
    live = save_config(sdk)["properties"]
    rows = []
    for prop, entry in sorted(config.get("properties", {}).items()):
        now = live.get(prop, {}).get("value")
        if now != entry.get("value"):
            rows.append({
                "property": prop, "saved": entry.get("value"), "now": now,
                "control": entry.get("control"), "motion": _is_motion(prop),
            })
    return rows


def _write_and_settle(microscope: Any, prop: str, value: Any) -> tuple[Any, Any]:
    """setattr, then poll the read-back until it matches or SETTLE_SECONDS.

    Returns (before, readback). Polling rather than one immediate read is
    the whole point - see SETTLE_SECONDS.
    """
    before = getattr(microscope, prop)
    setattr(microscope, prop, value)
    deadline = time.monotonic() + SETTLE_SECONDS
    readback = getattr(microscope, prop)
    while readback != value and time.monotonic() < deadline:
        time.sleep(SETTLE_POLL_SECONDS)
        readback = getattr(microscope, prop)
    return before, readback


def apply_config(config: dict, sdk: Any = None, include_motion: bool = False,
                 bulk: bool = False) -> dict:
    """Write a saved config back. Real hardware - see this module's header."""
    if sdk is None:
        from acquisition.backends.nis_sdk import NISSdk
        sdk = NISSdk()

    props = config.get("properties", {})

    def run(microscope: Any) -> dict:
        if bulk:
            data = microscope.DataGet(None, 0)
            for prop, entry in props.items():
                try:
                    setattr(data, prop, entry.get("value"))
                except Exception:
                    continue
            microscope.DataSet(data, 0, 1)
            return {"_bulk": {"status": "DataSet issued", "properties": len(props)}}

        report: dict[str, dict] = {}
        for prop, entry in sorted(props.items()):
            value = entry.get("value")
            if not _writable(entry):
                report[prop] = {"status": "skipped", "why": "control=%s" % entry.get("control")}
                continue
            if not include_motion and _is_motion(prop):
                report[prop] = {"status": "skipped", "why": "motion device"}
                continue
            try:
                if getattr(microscope, prop) == value:
                    report[prop] = {"status": "unchanged", "value": value}
                    continue
                before, readback = _write_and_settle(microscope, prop, value)
                report[prop] = {
                    "status": "written" if readback == value else "not-applied",
                    "from": before, "to": value, "readback": readback,
                }
            except Exception as exc:
                report[prop] = {"status": "error",
                                "error": "%s: %s" % (type(exc).__name__, exc)}
        return report

    return sdk._thread.call(run)


def _default_out() -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("protocols") / ("optical_config_%s.json" % stamp)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save and restore the Ti2 device configuration.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_save = sub.add_parser("save", help="snapshot the configuration (read-only)")
    p_save.add_argument("--out", type=Path, default=None)

    p_show = sub.add_parser("show", help="print a saved configuration file")
    p_show.add_argument("path", type=Path)
    p_show.add_argument("--writable", action="store_true",
                        help="only properties this SDK can drive (control >= 0)")

    p_diff = sub.add_parser("diff", help="compare a saved file against the live scope")
    p_diff.add_argument("path", type=Path)

    p_apply = sub.add_parser("apply", help="write a saved configuration back")
    p_apply.add_argument("path", type=Path)
    p_apply.add_argument("--confirm", action="store_true",
                         help="required: this moves real hardware")
    p_apply.add_argument("--include-motion", action="store_true",
                         help="also restore stage/objective/TIRF positions")
    p_apply.add_argument("--bulk", action="store_true",
                         help="use DataSet to write everything in one call "
                              "(ignores --include-motion; moves the stage)")

    args = parser.parse_args()

    if args.command == "save":
        config = save_config()
        out = args.out or _default_out()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
        print(format_config(config))
        print("\nwrote %s" % out)

    elif args.command == "show":
        config = json.loads(args.path.read_text(encoding="utf-8"))
        print(format_config(config, only_writable=args.writable))

    elif args.command == "diff":
        config = json.loads(args.path.read_text(encoding="utf-8"))
        rows = diff_config(config)
        if not rows:
            print("no differences - live scope matches %s" % args.path)
        else:
            print("  %-24s %10s %10s %8s  %s"
                  % ("PROPERTY", "SAVED", "NOW", "CONTROL", "RESTORABLE?"))
            for row in rows:
                if row["control"] is None or row["control"] < 0:
                    verdict = "no (control=%s)" % row["control"]
                elif row["motion"]:
                    verdict = "only with --include-motion"
                else:
                    verdict = "yes"
                print("  %-24s %10s %10s %8s  %s"
                      % (row["property"], row["saved"], row["now"], row["control"], verdict))
            print("\n%d propert%s differ" % (len(rows), "y" if len(rows) == 1 else "ies"))

    elif args.command == "apply":
        if not args.confirm:
            raise SystemExit("apply moves real hardware - re-run with --confirm")
        config = json.loads(args.path.read_text(encoding="utf-8"))
        report = apply_config(config, include_motion=args.include_motion, bulk=args.bulk)
        counts: dict[str, int] = {}
        for prop, result in sorted(report.items()):
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            if result["status"] in ("written", "not-applied", "error", "DataSet issued"):
                print("  %-24s %s" % (prop, result))
        print("\n" + ", ".join("%s: %d" % kv for kv in sorted(counts.items())))


if __name__ == "__main__":
    main()
