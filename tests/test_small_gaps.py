# The last small branches: server entry points, package version fallback,
# bridge error paths, and a few `python -m` blocks.
import contextlib
import http.server
import importlib
import json
import queue
import runpy
import sys
import threading
import types
import urllib.request
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from acquisition import estop, nis_monitor, paths
from acquisition.backends import nis_bridge as client_mod
from acquisition.nis_bridge import bridge_job as bj
from acquisition.orchestration import stage_positions, worm_follow
from harness import context
from mcp_server import nis_tools
from tests.fakes import fake_ti2  # noqa: F401  (fixture)
from tests.fakes_camera import install_fake_camera_stack
from tests.test_nis_bridge import FakeNIS, bridge, estop_file, raw  # noqa: F401  (fixtures)


# ---- MCP server entry points -------------------------------------------------------

def test_nis_server_main_opens_the_panel_and_monitor_then_serves(monkeypatch):
    from mcp_server import server_nis
    calls = []
    monkeypatch.setattr(server_nis.estop, "launch_panel", lambda: calls.append("panel"))
    monkeypatch.setattr(server_nis.nis_monitor, "launch", lambda: calls.append("monitor"))
    monkeypatch.setattr(server_nis.mcp, "run", lambda transport: calls.append(("run", transport)))
    server_nis.main()
    assert calls == ["panel", "monitor", ("run", "stdio")]


def test_loop_server_main_opens_the_panel_then_serves(monkeypatch):
    from mcp_server import server_loop
    calls = []
    monkeypatch.setattr(server_loop.estop, "launch_panel", lambda: calls.append("panel"))
    monkeypatch.setattr(server_loop.mcp, "run", lambda transport: calls.append(("run", transport)))
    server_loop.main()
    assert calls == ["panel", ("run", "stdio")]


@pytest.mark.parametrize("module", ["mcp_server.server_loop", "mcp_server.server_nis"])
def test_server_modules_start_when_run_with_dash_m(module, monkeypatch):
    from mcp.server.mcpserver import MCPServer
    ran = []
    monkeypatch.setattr(MCPServer, "run", lambda self, transport: ran.append(transport))
    monkeypatch.setattr(estop, "launch_panel", lambda: None)
    monkeypatch.setattr(nis_monitor, "launch", lambda: None)
    runpy.run_module(module, run_name="__main__")
    assert ran == ["stdio"]


def test_the_nis_server_registers_the_nis_tools_in_process():
    import asyncio
    from mcp_server import server_nis
    names = {t.name for t in asyncio.run(server_nis.mcp.list_tools())}
    assert names == {"nis_status", "nis_move_relative", "nis_change_objective", "nis_capture",
                     "nis_run_experiment", "nis_finish_experiment", "nis_list_optical_configs", "estop"}


def test_package_version_falls_back_when_not_installed(monkeypatch):
    import importlib.metadata as md
    import mcp_server

    def missing(name):
        raise md.PackageNotFoundError(name)

    monkeypatch.setattr(md, "version", missing)
    try:
        importlib.reload(mcp_server)
        assert mcp_server.__version__ == "0.0.0+unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(mcp_server)


# ---- NIS tools / client / bridge -----------------------------------------------------

def test_status_and_finish_tools_reach_the_bridge(bridge):
    nis, url, c = bridge
    assert nis_tools.nis_status()["xy_um"] == [100.0, 200.0]
    nis.nd_running = True
    assert nis_tools.nis_finish_experiment()["nd_running"] is False


def test_client_wraps_a_non_json_error_body(tmp_path):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"<html>oops</html>")

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = client_mod.NISBridge(f"http://127.0.0.1:{srv.server_address[1]}").status()
    finally:
        srv.shutdown()
        srv.server_close()
    assert out["http_status"] == 500 and "error" in out


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["nis_bridge", *argv])
    client_mod.main()


def test_cli_exits_with_the_reason_when_the_bridge_is_down_or_stopped(monkeypatch, estop_file):
    monkeypatch.setenv("CONFOCAL_NIS_BRIDGE_URL", "http://127.0.0.1:9")
    with pytest.raises(SystemExit, match="not reachable"):
        run_cli(monkeypatch, "status")
    estop_file.write_text("{}")
    with pytest.raises(SystemExit, match="E-STOP"):
        run_cli(monkeypatch, "move", "1", "0")


def test_cli_prints_the_bridge_answer(bridge, monkeypatch, capsys):
    nis, url, c = bridge
    run_cli(monkeypatch, "--url", url, "status")
    assert json.loads(capsys.readouterr().out)["xy_um"] == [100.0, 200.0]


def test_cli_module_runs_as_a_script(bridge, monkeypatch, capsys):
    nis, url, c = bridge
    monkeypatch.setattr(sys, "argv", ["nis_bridge", "--url", url, "status"])
    runpy.run_module("acquisition.backends.nis_bridge", run_name="__main__")
    assert "xy_um" in capsys.readouterr().out


def test_bridge_treats_an_unreadable_estop_file_as_engaged(monkeypatch):
    class Broken:
        def exists(self):
            raise PermissionError("denied")

    monkeypatch.setattr(bj, "ESTOP", Broken())
    assert bj._estop_engaged() is True


def test_objective_name_must_be_on_the_nosepiece_exactly_once(bridge):
    nis, url, c = bridge
    nis.objectives[5] = nis.objectives[2]                       # the same 10x in two positions
    r = c.change_objective("Plan Fluor 10x Ph1 DLL")
    assert "exactly once" in r["error"] and "Stg_SetNosepiecePosition" not in nis.calls


def test_bridge_rejects_a_body_that_is_not_json(bridge):
    nis, url, c = bridge
    req = urllib.request.Request(url + "/move", data=b"{not json", method="POST")
    try:
        urllib.request.urlopen(req, timeout=5)
        status = 200
    except urllib.error.HTTPError as e:
        status, body = e.code, json.loads(e.read())
    assert status == 400 and "not JSON" in body["error"]
    assert nis.moved() == []


def test_a_route_that_raises_is_reported_and_the_bridge_survives(bridge, monkeypatch):
    nis, url, c = bridge
    monkeypatch.setitem(bj.ROUTES, ("GET", "/status"), lambda b, body: 1 / 0)
    assert "ZeroDivisionError" in c.status()["error"]
    monkeypatch.undo()
    assert c.status()["xy_um"] == [100.0, 200.0]


def test_a_request_the_worker_never_answers_times_out_with_504():
    jobs, stop = queue.Queue(), threading.Event()                  # nobody consumes the queue
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), bj.make_handler(jobs, stop, timeout_s=0.1))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        code, out = raw(f"http://127.0.0.1:{srv.server_address[1]}", "POST", "/move", {"dx": 1})
    finally:
        srv.shutdown()
        srv.server_close()
    assert code == 504 and "timed out" in out["error"]


# ---- paths / monitor / context ----------------------------------------------------------

def test_system_dir_check_survives_an_os_error(monkeypatch):
    class Odd:
        def __eq__(self, other):
            raise OSError("odd filesystem")

        @property
        def parents(self):
            raise OSError("odd filesystem")

    monkeypatch.setenv("SystemRoot", "C:\\Windows")
    assert paths._is_system_dir(Odd()) is False


def test_monitor_probe_is_false_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert nis_monitor.monitor_running() is False


def test_monitor_module_prints_one_snapshot_as_a_script(monkeypatch, capsys):
    monkeypatch.setenv("CONFOCAL_NIS_BRIDGE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(sys, "argv", ["nis_monitor", "--once"])
    runpy.run_module("acquisition.nis_monitor", run_name="__main__")
    assert "bridge:" in capsys.readouterr().out


def test_strip_images_copies_plain_string_content():
    msg = {"role": "user", "content": "just text"}
    out = context._strip_images(msg)
    assert out == msg and out is not msg


# ---- camera / stage backends ------------------------------------------------------------

def test_camera_demo_captures_one_frame(monkeypatch, tmp_path, capsys):
    mod, acq, _ = install_fake_camera_stack(monkeypatch, tmp_path)
    try:
        runpy.run_module("acquisition.backends.baumer_genicam", run_name="__main__")
    finally:
        sys.modules.pop("acquisition.backends.baumer_genicam", None)
    assert "Saved:" in capsys.readouterr().out and acq.stopped


def test_stage_positions_can_drive_the_sdk_backend(fake_ti2, tmp_path):
    m = stage_positions.StagePositionManager(backend="sdk", positions_file=tmp_path / "p.json")
    assert m.save_current("here") == {"x": 100.0, "y": 200.0, "z": 4619.0}


def test_force_camera_autos_off_ignores_nodes_that_raise():
    class Bad:
        @property
        def value(self):
            raise RuntimeError("node not available")

    nm = NS(BalanceWhiteAuto=Bad(), ColorTransformationAuto=Bad(), ExposureAuto=Bad(), GainAuto=Bad())
    cam = NS(_acquirer=NS(remote_device=NS(node_map=nm)))
    worm_follow.force_camera_autos_off(cam)                       # no exception


# ---- the NIS port probe's timeout and the calibration/panel scripts -------------------------

def test_probe_reports_a_nis_that_never_answers(monkeypatch, tmp_path):
    import ctypes
    limjob = types.ModuleType("limjob")
    limjob.Image = limjob.JobParam = limjob.MacroParam = limjob.RunContext = object
    monkeypatch.setitem(sys.modules, "limjob", limjob)
    monkeypatch.setattr(ctypes, "cdll", NS(g5_regprocs=FakeNIS()), raising=False)
    sys.modules.pop("acquisition.calibration.nis_port_probe", None)
    mod = importlib.import_module("acquisition.calibration.nis_port_probe")

    class NeverAnswers(queue.Queue):
        def get(self, timeout=None):
            raise queue.Empty

    monkeypatch.setattr(mod, "queue", NS(Queue=NeverAnswers, Empty=queue.Empty))
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), mod.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/status", timeout=5) as r:
            assert "timed out waiting for NIS" in json.loads(r.read())["error"]
    finally:
        srv.shutdown()
        srv.server_close()
        sys.modules.pop("acquisition.calibration.nis_port_probe", None)


@pytest.mark.parametrize("module", ["acquisition.calibration.ti2_config", "acquisition.calibration.ti2_inventory"])
def test_calibration_modules_run_as_scripts(module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["x", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module(module, run_name="__main__")
    assert exc.value.code == 0 and "usage" in capsys.readouterr().out


def test_estop_panel_runs_as_a_script(monkeypatch):
    class TclError(Exception):
        pass

    def no_display():
        raise TclError("no display")

    tk = types.ModuleType("tkinter")
    tk.Tk, tk.TclError = no_display, TclError
    tk.messagebox = types.ModuleType("tkinter.messagebox")
    monkeypatch.setitem(sys.modules, "tkinter", tk)
    monkeypatch.setitem(sys.modules, "tkinter.messagebox", tk.messagebox)
    monkeypatch.setattr(sys, "argv", ["estop_panel"])
    with contextlib.suppress(SystemExit):                          # no display -> exit 1; already open -> returns
        runpy.run_module("acquisition.estop_panel", run_name="__main__")
