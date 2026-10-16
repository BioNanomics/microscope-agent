import json
import sys

import numpy as np
import pytest
from PIL import Image

from acquisition import estop
from acquisition.orchestration import mosaic
from tests.fakes_scene import RigStage, SceneRig, UM_PER_PX


@pytest.fixture(autouse=True)
def quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(mosaic.time, "sleep", lambda s: None)
    monkeypatch.setattr(mosaic, "data_root", lambda: tmp_path / "root")
    monkeypatch.setattr(estop, "ESTOP_PATH", tmp_path / "ESTOP")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return SceneRig(tmp_path).install(monkeypatch)


# ---- grid and stitch ------------------------------------------------------

def test_grid_positions_is_a_serpentine_around_the_centre():
    pts = mosaic.grid_positions(100, 200, 3, 2, 10, 20)
    assert pts == [(90.0, 190.0), (100.0, 190.0), (110.0, 190.0),
                   (110.0, 210.0), (100.0, 210.0), (90.0, 210.0)]


def test_stitch_places_tiles_by_stage_offset():
    a = np.full((10, 10, 3), 50, np.uint8)
    b = np.full((10, 10, 3), 200, np.uint8)
    M = np.array([[-1.0, 0.0], [0.0, 1.0]])              # +X slides features left
    canvas = mosaic.stitch([(a, 0.0, 0.0), (b, 10.0, 0.0)], M, (0.0, 0.0))
    assert canvas.shape == (10, 20, 3)
    assert canvas[0, 0, 0] == 50 and canvas[0, 19, 0] == 200          # the tile at larger X shows sample further right


# ---- focus plane (the 2026-09-21 incident) --------------------------------

def test_focus_plane_needs_three_points():
    with pytest.raises(ValueError, match="at least 3"):
        mosaic.FocusPlane([(0, 0, 4600), (1000, 0, 4600)])


def test_focus_plane_refuses_collinear_points():
    with pytest.raises(ValueError, match="collinear"):
        mosaic.FocusPlane([(0, 0, 4600), (1000, 0, 4601), (2000, 0, 4602)])


def test_focus_plane_recovers_a_tilted_plane_with_no_residual():
    f = lambda x, y: 4600 + 0.002 * x - 0.001 * y
    pts = [(x, y, f(x, y)) for x, y in ((0, 0), (10000, 0), (0, 8000), (10000, 8000))]
    plane = mosaic.FocusPlane(pts)
    assert plane.z(5000, 4000) == pytest.approx(f(5000, 4000), abs=1e-6)
    assert max(plane.residuals) < 1e-6 and "clamped to +/-400" in str(plane)


def test_focus_plane_never_commands_more_than_the_clamp():
    plane = mosaic.FocusPlane([(0, 0, 4600), (1000, 0, 4601), (0, 1000, 4602)], max_dev=50)
    assert plane.would_clamp(1e7, 0) and not plane.would_clamp(10, 10)
    assert plane.z(1e7, 0) == pytest.approx(plane.z_ref + 50)
    assert plane.z(-1e7, 0) == pytest.approx(plane.z_ref - 50)


# ---- moves ------------------------------------------------------------------

def test_move_safe_walks_a_long_jump_in_legal_hops(rig):
    s = rig.stage
    assert mosaic._move_safe(s, 12000, 0) == (12000.0, 0.0)
    assert len(s.moves) == 3 and all(abs(b - a) <= 4500 + 1e-6 for a, b in zip([0] + [m[0] for m in s.moves], [m[0] for m in s.moves]))


def test_move_safe_is_a_noop_when_already_there(rig):
    assert mosaic._move_safe(rig.stage, 0.4, 0.0) == (0.0, 0.0) and rig.stage.moves == []


def test_move_z_steps_under_the_cap(rig):
    s = rig.stage
    z = mosaic._move_z(s, s.z + 120)
    assert z == pytest.approx(s.z) and all(abs(b - a) <= 45 + 1e-6 for a, b in zip([4619.0] + s.z_moves, s.z_moves))
    assert len(s.z_moves) == 3


def test_move_z_gives_up_on_a_stuck_axis(rig, monkeypatch):
    s = rig.stage
    monkeypatch.setattr(s, "Z_Move", lambda z: None)                 # accepted, never arrives
    start = s.z
    assert mosaic._move_z(s, start + 10) == start


# ---- calibration ------------------------------------------------------------

def test_calibrate_axes_measures_the_stage_to_image_mapping(rig, capsys):
    M = mosaic.calibrate_axes(rig.camera, rig.stage)
    assert M[0, 0] == pytest.approx(-1 / UM_PER_PX, abs=0.01) and M[1, 1] == pytest.approx(1 / UM_PER_PX, abs=0.01)
    assert abs(M[0, 1]) < 0.01 and abs(M[1, 0]) < 0.01
    assert rig.stage.XY_GetPosition() == (0.0, 0.0)                   # returned home
    assert "um/px" in capsys.readouterr().out


def test_calibrate_axes_stops_when_the_stage_does_not_move(rig):
    rig.stage.frozen = True
    with pytest.raises(SystemExit, match="calibration failed"):
        mosaic.calibrate_axes(rig.camera, rig.stage)


# ---- run --------------------------------------------------------------------

def run_kw(**over):
    kw = dict(grid=(2, 2), overlap=0.1, rounds=1, interval_s=0.0, exposure_us=1600.0,
              mosaic_scale=0.5, keep_tiles=True, scale_um_px=UM_PER_PX, tag="t")
    kw.update(over)
    return kw


def test_run_writes_tiles_csv_json_and_a_stitched_mosaic(rig, tmp_path):
    out = mosaic.run(**run_kw())
    assert (out / "tiles.csv").exists() and (out / "mosaic_000.png").exists()
    info = json.loads((out / "mosaic.json").read_text())
    assert info["grid"] == [2, 2] and info["um_per_px"] == pytest.approx(UM_PER_PX, abs=1e-3)
    assert len(list((out / "round_000").glob("t*.png"))) == 4
    assert rig.camera.closed and rig.stage.XY_GetPosition() == (0.0, 0.0)
    assert Image.open(out / "mosaic_000.png").size[0] > 100
    rows = (out / "tiles.csv").read_text().splitlines()
    assert len(rows) == 5 and all(r.endswith(",") for r in rows[1:])        # no error text


def test_run_calibrates_when_no_scale_is_given(rig, capsys):
    mosaic.run(**run_kw(scale_um_px=None, grid=(1, 1), keep_tiles=False, mosaic_scale=1.0))
    assert "calibrating stage->image mapping" in capsys.readouterr().out


def test_run_follows_a_focus_plane_and_a_given_centre(rig):
    pts = [(0, 0, 4619.0), (300, 0, 4625.0), (0, 300, 4619.0)]
    out = mosaic.run(**run_kw(grid=(1, 1), focus_points=pts, centre=(100.0, -50.0)))
    assert rig.stage.z_moves, "Z should follow the plane"
    assert (out / "mosaic_000.png").exists()


def test_run_fixed_focus_leaves_z_alone(rig):
    mosaic.run(**run_kw(grid=(1, 1)))
    assert rig.stage.z_moves == []


def test_run_forces_balance_white_off_and_survives_a_missing_node(tmp_path, monkeypatch, capsys):
    rig = SceneRig(tmp_path, white_balance="Continuous").install(monkeypatch)
    mosaic.run(**run_kw(grid=(1, 1)))
    assert rig.camera._acquirer.remote_device.node_map.BalanceWhiteAuto.value == "Off"
    assert "forcing 'Off'" in capsys.readouterr().out
    del rig.camera._acquirer.remote_device.node_map.BalanceWhiteAuto
    mosaic.run(**run_kw(grid=(1, 1)))
    assert "could not force BalanceWhiteAuto off" in capsys.readouterr().err


def test_a_failed_tile_is_logged_and_the_round_continues(rig):
    rig.camera.on_capture = lambda n: (_ for _ in ()).throw(RuntimeError("fetch timed out")) if n == 3 else None
    out = mosaic.run(**run_kw())
    text = (out / "tiles.csv").read_text()
    assert text.count("RuntimeError: fetch timed out") == 1
    assert (out / "mosaic_000.png").exists()


def test_stop_file_halts_the_run_and_still_parks_and_releases(rig, tmp_path):
    def stop_after_first(n):
        if n == 3:        # 1 = probe, 2 = first tile; 3 = second tile is refused at its start
            for d in (tmp_path / "root" / "data").glob("mosaic_*"):
                (d / "STOP").write_text("")
    rig.camera.on_capture = stop_after_first
    with pytest.raises(KeyboardInterrupt):
        mosaic.run(**run_kw(rounds=3))
    assert rig.camera.closed and rig.stage.XY_GetPosition() == (0.0, 0.0)


def test_parking_failure_is_reported_not_raised(rig, monkeypatch, capsys):
    real = mosaic._move_safe
    calls = []

    def flaky(sdk, x, y):
        calls.append(1)
        if len(calls) > 2:               # the two tile moves work, the park at the end fails
            raise RuntimeError("stage gone")
        return real(sdk, x, y)

    monkeypatch.setattr(mosaic, "_move_safe", flaky)
    mosaic.run(**run_kw(grid=(2, 1), keep_tiles=False))
    assert "could not return to centre" in capsys.readouterr().err and rig.camera.closed


def test_estop_during_a_run_is_recorded_per_tile(rig, capsys):
    estop.engage("test")
    out = mosaic.run(**run_kw(grid=(2, 1)))
    assert (out / "tiles.csv").read_text().count("EStopEngaged") == 2
    assert not list(out.glob("mosaic_000.png")) and rig.stage.moves == []
    assert rig.camera.closed


def test_rounds_are_paced_by_the_interval(rig, monkeypatch):
    slept = []
    monkeypatch.setattr(mosaic.time, "sleep", lambda s: slept.append(s))
    mosaic.run(**run_kw(grid=(1, 1), rounds=2, interval_s=600.0))
    assert any(s > 100 for s in slept)


# ---- command line -----------------------------------------------------------

def run_main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["mosaic", *argv])
    mosaic.main()


def test_dry_run_prints_the_plan_and_moves_nothing(rig, monkeypatch, capsys):
    run_main(monkeypatch, "--dry-run", "--grid", "3x2", "--hours", "1", "--interval-min", "20")
    out = capsys.readouterr().out
    assert "3x2 = 6 tiles" in out and "3 rounds every 20.0 min" in out
    assert rig.stage.moves == [] and rig.camera.captures == 0


def test_main_hands_parsed_arguments_to_run(monkeypatch):
    seen = []
    monkeypatch.setattr(mosaic, "run", lambda *a: seen.append(a))
    run_main(monkeypatch, "--grid", "4x3", "--rounds", "2", "--no-tiles", "--scale", "1.5",
             "--focus-plane", "0,0,4600;100,0,4601;0,100,4602", "--centre", "10,20", "--tag", "x")
    (grid, overlap, rounds, interval, exposure, mscale, keep, scale, tag, fpts, centre), = seen
    assert grid == (4, 3) and rounds == 2 and keep is False and scale == 1.5 and tag == "x"
    assert fpts == [(0.0, 0.0, 4600.0), (100.0, 0.0, 4601.0), (0.0, 100.0, 4602.0)]
    assert centre == (10.0, 20.0) and interval == 20 * 60


def test_main_defaults_to_one_round(monkeypatch):
    seen = []
    monkeypatch.setattr(mosaic, "run", lambda *a: seen.append(a))
    run_main(monkeypatch)
    assert seen[0][2] == 1 and seen[0][9] is None and seen[0][10] is None


def test_main_rejects_a_focus_point_that_is_not_xyz(monkeypatch):
    with pytest.raises(SystemExit) as exc:
        run_main(monkeypatch, "--focus-plane", "0,0;1,1,1")
    assert exc.value.code == 2
