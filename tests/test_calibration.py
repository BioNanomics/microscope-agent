import ctypes
import http.server
import json
import sys
import threading
import time
import types
import urllib.request
from types import SimpleNamespace as NS

import pytest

from acquisition import estop
from acquisition.calibration import ti2_config, ti2_inventory
from tests.fakes import fake_ti2  # noqa: F401  (fixture)
from tests.test_nis_bridge import FakeNIS


def make_setting(name, control, value=1, lower=0, higher=10, unit="u", enabled=True, long_name=None):
    """A COM-style Setting: readable fields named in _prop_map_get_, plus accessor methods."""
    attrs = {"_prop_map_get_": {"Name": 1, "Value": 1, "Lower": 1, "Higher": 1, "Unit": 1, "Enabled": 1,
                                "Control": 1, "Broken": 1},
             "Name": name, "Value": value, "Lower": lower, "Higher": higher, "Unit": unit,
             "Enabled": enabled, "Control": control,
             "Broken": property(lambda self: (_ for _ in ()).throw(RuntimeError("unreadable"))),
             "LongName": lambda self: long_name or name.upper(),
             "ShortName": lambda self: (_ for _ in ()).throw(RuntimeError("no short name"))}
    return type("Setting", (), attrs)()


@pytest.fixture
def scope(fake_ti2):
    sdk, m = fake_ti2
    m.Settings = [make_setting("DiaLampSwitch", 1), make_setting("LightPath", 0, value=2),
                  make_setting("XPosition", 1, value=5), make_setting("Locked", -1),
                  make_setting("Mystery", None)]
    return sdk, m


# ---- ti2_inventory -------------------------------------------------------------

def test_setting_fields_come_from_the_type_library():
    fields = ti2_inventory._setting_fields(make_setting("A", 1))
    assert "Name" in fields and "LongName" in fields and "ShortName" in fields


def test_read_field_calls_accessors_and_tolerates_failures():
    s = make_setting("A", 1)
    assert ti2_inventory._read_field(s, "Name") == "A"
    assert ti2_inventory._read_field(s, "LongName") == "A".upper()
    assert ti2_inventory._read_field(s, "ShortName") is None
    assert ti2_inventory._read_field(s, "Broken") is None
    assert ti2_inventory._read_field(s, "Missing") is None


def test_inventory_snapshots_every_setting(scope):
    rows = ti2_inventory.inventory()
    assert [r["Name"] for r in rows] == ["DiaLampSwitch", "LightPath", "XPosition", "Locked", "Mystery"]
    assert rows[0]["index"] == 0 and rows[0]["Broken"] is None and rows[1]["Value"] == 2


def test_inventory_reuses_a_given_sdk(scope):
    sdk, _ = scope
    assert len(ti2_inventory.inventory(sdk.NISSdk())) == 5


def test_inventory_reports_an_unreadable_settings_list(fake_ti2):
    sdk, m = fake_ti2
    m.__class__ = type("NoSettings", (type(m),), {"Settings": property(lambda self: (_ for _ in ()).throw(RuntimeError("busy")))})
    rows = ti2_inventory.inventory()
    assert rows[0]["_error"] == "RuntimeError: busy"
    assert ti2_inventory.format_inventory(rows) == "could not read Settings: RuntimeError: busy"


def test_format_inventory_groups_by_control(scope):
    text = ti2_inventory.format_inventory(ti2_inventory.inventory())
    assert "Control = 1  (writes took effect)" in text and "writes refused" in text
    assert "Control unknown" in text and "5 devices. Fields discovered" in text
    custom = ti2_inventory.format_inventory(ti2_inventory.inventory(), columns=["Name"])
    assert "DiaLampSwitch" in custom and "Unit" not in custom.splitlines()[1]


def test_inventory_main_prints_and_writes_json(scope, tmp_path, monkeypatch, capsys):
    out = tmp_path / "inv.json"
    monkeypatch.setattr(sys, "argv", ["inv", "--json", str(out), "--columns", "Name,Control"])
    ti2_inventory.main()
    assert "wrote" in capsys.readouterr().out and len(json.loads(out.read_text())) == 5
    monkeypatch.setattr(sys, "argv", ["inv"])
    ti2_inventory.main()
    assert "devices" in capsys.readouterr().out


# ---- ti2_config ----------------------------------------------------------------

def test_motion_and_name_helpers():
    assert ti2_config._is_motion("iXPOSITION") and ti2_config._is_motion("iNOSEPIECE") and ti2_config._is_motion("iZESCAPE")
    assert not ti2_config._is_motion("iLIGHTPATH")
    assert ti2_config._normalise("iDIA_LAMP_Switch") == ti2_config._normalise("DiaLampSwitch")


def test_save_config_joins_values_with_settings_metadata(scope):
    cfg = ti2_config.save_config()
    assert cfg["property_count"] == 4 and "iBROKEN" not in cfg["properties"]
    lamp = cfg["properties"]["iDIA_LAMP_Switch"]
    assert lamp["value"] == 1 and lamp["control"] == 1 and lamp["friendly_name"] == "DiaLampSwitch"
    assert lamp["lower"] == 0 and lamp["higher"] == 10
    assert cfg["properties"]["iLIGHTPATH"]["control"] == 0


def test_format_config_filters_writable(scope):
    cfg = ti2_config.save_config()
    both = ti2_config.format_config(cfg)
    only = ti2_config.format_config(cfg, only_writable=True)
    assert "iDIA_LAMP_Switch" in both and "iDIA_LAMP_Switch" in only
    assert "iNOSEPIECE" in both and "iNOSEPIECE" not in only            # no matching setting: control unknown
    assert "0 .. 10" in both


def test_diff_config_lists_what_changed(scope):
    sdk, m = scope
    cfg = ti2_config.save_config()
    assert ti2_config.diff_config(cfg) == []
    cfg["properties"]["iLIGHTPATH"]["value"] = 99
    cfg["properties"]["iXPOSITION"]["value"] = 77
    rows = {r["property"]: r for r in ti2_config.diff_config(cfg)}
    assert rows["iLIGHTPATH"]["saved"] == 99 and rows["iLIGHTPATH"]["now"] == 2 and not rows["iLIGHTPATH"]["motion"]
    assert rows["iXPOSITION"]["motion"] is True


def entry(value, control=1):
    return {"value": value, "control": control}


@pytest.fixture
def quick(monkeypatch):
    monkeypatch.setattr(ti2_config.time, "sleep", lambda s: None)
    monkeypatch.setattr(ti2_config, "SETTLE_SECONDS", 0.01)


def test_apply_config_reports_each_outcome(scope, quick):
    sdk, m = scope
    m.__class__ = type("Ignores", (type(m),), {
        "iPFS_STATUS": property(lambda self: 7, lambda self, v: None)})        # accepts the write, keeps the value
    cfg = {"properties": {
        "iLIGHTPATH": entry(4),                         # written
        "iDIA_LAMP_Switch": entry(0),                   # unchanged (already 0)
        "iPFS_STATUS": entry(9),                        # not-applied
        "iNOSEPIECE": entry(5),                         # motion: skipped
        "iXPOSITION": entry(7, control=-1),             # not writable: skipped
        "iMISSING": entry(1)}}                          # error
    report = ti2_config.apply_config(cfg, sdk.NISSdk())
    assert report["iLIGHTPATH"]["status"] == "written" and m.iLIGHTPATH == 4
    assert report["iDIA_LAMP_Switch"]["status"] == "unchanged"
    assert report["iPFS_STATUS"]["status"] == "not-applied"
    assert report["iNOSEPIECE"] == {"status": "skipped", "why": "motion device"}
    assert report["iXPOSITION"]["why"] == "control=-1"
    assert report["iMISSING"]["status"] == "error" and m.iNOSEPIECE == 1


def test_apply_config_with_motion_allowed(scope, quick):
    sdk, m = scope
    report = ti2_config.apply_config({"properties": {"iNOSEPIECE": entry(5)}}, include_motion=True)
    assert report["iNOSEPIECE"]["status"] == "written" and m.iNOSEPIECE == 5


def test_apply_config_waits_for_a_slow_axis_to_settle(scope, monkeypatch):
    sdk, m = scope
    reads = {"n": 0}

    def slow_get(self):
        reads["n"] += 1
        return 4 if reads["n"] > 4 else 1

    m.__class__ = type("Slow", (type(m),), {"iLIGHTPATH": property(slow_get, lambda self, v: None)})
    slept = []
    monkeypatch.setattr(ti2_config.time, "sleep", lambda s: slept.append(s))
    report = ti2_config.apply_config({"properties": {"iLIGHTPATH": entry(4)}})
    assert report["iLIGHTPATH"]["status"] == "written" and slept


def test_apply_config_bulk_uses_dataset(scope):
    sdk, m = scope
    sent = []
    m.DataSet = lambda data, a, b: sent.append((data, a, b))
    report = ti2_config.apply_config({"properties": {"iLIGHTPATH": entry(4), "iBROKEN": entry(1)}}, bulk=True)
    assert report == {"_bulk": {"status": "DataSet issued", "properties": 2}} and len(sent) == 1


def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["ti2_config", *argv])
    ti2_config.main()


def test_config_cli_save_show_diff_apply(scope, quick, tmp_path, monkeypatch, capsys):
    path = tmp_path / "cfg.json"
    run_main(monkeypatch, "save", "--out", str(path))
    assert path.exists() and "wrote" in capsys.readouterr().out
    run_main(monkeypatch, "show", str(path), "--writable")
    assert "PROPERTY" in capsys.readouterr().out
    run_main(monkeypatch, "diff", str(path))
    assert "no differences" in capsys.readouterr().out

    cfg = json.loads(path.read_text())
    cfg["properties"]["iLIGHTPATH"]["value"] = 4          # restorable
    cfg["properties"]["iXPOSITION"]["value"] = 77         # motion
    cfg["properties"]["iXPOSITION"]["control"] = 1
    cfg["properties"]["iDIA_LAMP_Switch"]["value"] = 5
    cfg["properties"]["iDIA_LAMP_Switch"]["control"] = -1  # not restorable
    path.write_text(json.dumps(cfg))
    run_main(monkeypatch, "diff", str(path))
    out = capsys.readouterr().out
    assert "yes" in out and "only with --include-motion" in out and "no (control=-1)" in out and "3 properties differ" in out
    cfg["properties"].pop("iXPOSITION")
    cfg["properties"].pop("iDIA_LAMP_Switch")
    path.write_text(json.dumps(cfg))
    run_main(monkeypatch, "diff", str(path))
    assert "1 property differ" in capsys.readouterr().out

    with pytest.raises(SystemExit, match="--confirm"):
        run_main(monkeypatch, "apply", str(path))
    run_main(monkeypatch, "apply", str(path), "--confirm")
    out = capsys.readouterr().out
    assert "iLIGHTPATH" in out and "written: 1" in out


def test_config_cli_default_output_path(scope, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    run_main(monkeypatch, "save")
    assert list((tmp_path / "protocols").glob("optical_config_*.json"))


# ---- nis_port_probe (runs inside NIS; here against the fake DLL) -----------------

@pytest.fixture
def probe(monkeypatch, tmp_path):
    dll = FakeNIS()
    limjob = types.ModuleType("limjob")                       # NIS's job API; the probe annotates with its types
    limjob.Image = limjob.JobParam = limjob.MacroParam = limjob.RunContext = object
    monkeypatch.setitem(sys.modules, "limjob", limjob)
    monkeypatch.setattr(ctypes, "cdll", NS(g5_regprocs=dll), raising=False)
    sys.modules.pop("acquisition.calibration.nis_port_probe", None)
    import importlib
    mod = importlib.import_module("acquisition.calibration.nis_port_probe")
    monkeypatch.setattr(mod, "ESTOP", str(tmp_path / "ESTOP"))
    yield mod, dll
    sys.modules.pop("acquisition.calibration.nis_port_probe", None)


def test_probe_status(probe):
    mod, dll = probe
    s = mod.status()
    assert s["xy_um"] == [100.0, 200.0] and s["z_um"] == 4619.0 and s["nosepiece_position"] == 1
    assert s["nosepiece_objectives"] == {i: n for i, n in dll.objectives.items()}


def test_probe_move_is_capped_and_blocked_by_estop(probe, tmp_path):
    mod, dll = probe
    r = mod.move(50, -20)
    assert r["after_um"] == [150.0, 180.0]
    assert "refused" in mod.move(1000.1, 0)["error"]
    (tmp_path / "ESTOP").write_text("{}")
    assert "e-stop" in mod.move(1, 0)["error"] and "e-stop" in mod.objective("Plan Fluor 10x Ph1 DLL")["error"]


def test_probe_objective_allow_list(probe):
    mod, dll = probe
    assert "allowed" in mod.objective("Plan Apo 60x Oil")["error"]
    r = mod.objective("Plan Fluor 10x Ph1 DLL")
    assert r["after"] == "Plan Fluor 10x Ph1 DLL" and r["rc"] == 1


def test_probe_serves_the_three_endpoints_from_run(probe, monkeypatch, tmp_path):
    mod, dll = probe
    servers = []

    class Rec(http.server.ThreadingHTTPServer):
        def __init__(self, *a, **k):
            super().__init__(("127.0.0.1", 0), a[1])
            servers.append(self)

    monkeypatch.setattr(mod.http.server, "ThreadingHTTPServer", Rec)
    done = threading.Event()
    ctx = NS(shouldAbort=lambda: done.is_set())
    t = threading.Thread(target=mod.run, args=((), None, None, ctx), daemon=True)
    t.start()
    deadline = time.time() + 5
    while not servers and time.time() < deadline:
        time.sleep(0.01)
    assert servers, "the probe never started its server"
    base = f"http://127.0.0.1:{servers[0].server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=5) as r:
            return json.loads(r.read())

    assert get("/status")["xy_um"] == [100.0, 200.0]
    assert get("/move?dx=10&dy=0")["after_um"][0] == 110.0
    assert "allowed" in get("/objective?name=Plan%20Apo%2060x%20Oil")["error"]
    assert "use /status" in get("/nothing")["error"]
    dll.StgGetPosXY = lambda *a: (_ for _ in ()).throw(RuntimeError("dll fell over"))
    assert "RuntimeError" in get("/status")["error"]               # a failing call is reported, not fatal
    done.set()
    t.join(5)
    assert not t.is_alive()
