# The last small branches: server entry points, package version fallback,
# and a few `python -m` blocks.
import contextlib
import importlib
import runpy
import sys
import types
from types import SimpleNamespace as NS

import pytest

from acquisition import estop, paths
from acquisition.orchestration import stage_positions, worm_follow
from harness import context
from tests.fakes import fake_ti2  # noqa: F401  (fixture)
from tests.fakes_camera import install_fake_camera_stack


# ---- MCP server entry points -------------------------------------------------------

def test_loop_server_main_opens_the_panel_then_serves(monkeypatch):
    from mcp_server.baumer_ti2 import baumer_ti2_server
    calls = []
    monkeypatch.setattr(baumer_ti2_server.estop, "launch_panel", lambda: calls.append("panel"))
    monkeypatch.setattr(baumer_ti2_server.mcp, "run", lambda transport: calls.append(("run", transport)))
    baumer_ti2_server.main()
    assert calls == ["panel", ("run", "stdio")]


def test_server_module_starts_when_run_with_dash_m(monkeypatch):
    module = "mcp_server.baumer_ti2.baumer_ti2_server"
    from mcp.server.mcpserver import MCPServer
    ran = []
    monkeypatch.setattr(MCPServer, "run", lambda self, transport: ran.append(transport))
    monkeypatch.setattr(estop, "launch_panel", lambda: None)
    monkeypatch.delitem(sys.modules, module, raising=False)       # run it as a fresh `python -m` would
    runpy.run_module(module, run_name="__main__")
    assert ran == ["stdio"]


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


# ---- paths / context ----------------------------------------------------------

def test_system_dir_check_survives_an_os_error(monkeypatch):
    class Odd:
        def __eq__(self, other):
            raise OSError("odd filesystem")

        @property
        def parents(self):
            raise OSError("odd filesystem")

    monkeypatch.setenv("SystemRoot", "C:\\Windows")
    assert paths._is_system_dir(Odd()) is False


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


# ---- the calibration/panel scripts -------------------------

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
    monkeypatch.delitem(sys.modules, 'acquisition.estop_panel', raising=False)       # run it as a fresh `python -m` would
    with contextlib.suppress(SystemExit):                          # no display -> exit 1; already open -> returns
        runpy.run_module("acquisition.estop_panel", run_name="__main__")
