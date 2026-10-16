import csv
import json
import sys
import types

import numpy as np
import pytest
from PIL import Image

from acquisition import estop
from acquisition.orchestration import mosaic, worm_follow as wf
from tests.fakes_scene import RigNodeMap, SceneRig, UM_PER_PX, Z_FOCUS

AREA = dict(min_area_mm2=0.001, max_area_mm2=0.02)       # the rig's worm is ~0.004 mm2


@pytest.fixture(autouse=True)
def quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(mosaic.time, "sleep", lambda s: None)       # the same time module wf uses
    monkeypatch.setattr(wf, "data_root", lambda: tmp_path / "root")
    monkeypatch.setattr(estop, "ESTOP_PATH", tmp_path / "ESTOP")


@pytest.fixture
def rig(tmp_path, monkeypatch):
    return SceneRig(tmp_path).install(monkeypatch)


def parse(monkeypatch, *argv):
    """The real command line's defaults, with `run`/`setup`/`detect_only` captured
    (only while parsing, so the real functions stay available afterwards)."""
    seen = []
    with monkeypatch.context() as m:
        m.setattr(wf, "run", lambda a: seen.append(("run", a)))
        m.setattr(wf, "setup", lambda a: seen.append(("setup", a)))
        m.setattr(wf, "detect_only", lambda *a: seen.append(("detect", a)))
        m.setattr(sys, "argv", ["worm_follow", *argv])
        wf.main()
    return seen


def track_args(monkeypatch, *argv):
    kind, args = parse(monkeypatch, "--grid", "1x1", "--search-grid", "2x1", "--min-area-mm2", "0.001",
                       "--max-area-mm2", "0.02", "--crop-mm", "0.3", "--deadband-mm", "0.01", *argv)[0]
    return args


def rows(out):
    with (out / "track.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def only_run_dir(tmp_path):
    return next((tmp_path / "root" / "data").glob("wormfollow_*"))


# ---- camera state ------------------------------------------------------------

def test_force_camera_autos_off_and_restores_colour(rig, capsys):
    nm = rig.camera._acquirer.remote_device.node_map
    wf.force_camera_autos_off(rig.camera)
    assert nm.ColorTransformationAuto.value == nm.GainAuto.value == "Off"
    assert nm.Gamma.value == 1.0 and nm.GainSelector.value == "All"
    assert nm.ColorTransformationValue.value in (0.0, 1.0)
    assert "identity matrix" in capsys.readouterr().out


def test_force_camera_autos_off_survives_a_camera_without_colour_nodes(tmp_path, monkeypatch, capsys):
    rig = SceneRig(tmp_path, full_nodes=False).install(monkeypatch)
    wf.force_camera_autos_off(rig.camera)
    assert rig.camera._acquirer.remote_device.node_map.GainAuto.value == "Off"
    assert "could not restore camera colour state" in capsys.readouterr().err


# ---- geometry ------------------------------------------------------------------

def test_canvas_to_stage_inverts_the_stitch_geometry():
    tiles = [(np.zeros((120, 160, 3), np.uint8), x, y) for x, y in ((0.0, 0.0), (200.0, 0.0), (0.0, 150.0))]
    ref = (0.0, 0.0)
    canvas, omin, twh = wf.stitch_with_origin(tiles, wf.M_4X, ref)
    assert twh == (160, 120) and canvas.shape[0] > 120 and canvas.shape[1] > 160
    # the centre of the first tile is the stage position it was taken at
    first_centre = -wf.M_4X @ np.array([0.0, 0.0]) - omin + np.array(twh) / 2
    sx, sy = wf.canvas_to_stage(first_centre[0], first_centre[1], wf.M_4X, ref, omin, twh)
    assert (sx, sy) == pytest.approx((0.0, 0.0), abs=1e-6)
    second_centre = -wf.M_4X @ np.array([200.0, 0.0]) - omin + np.array(twh) / 2
    sx, sy = wf.canvas_to_stage(second_centre[0], second_centre[1], wf.M_4X, ref, omin, twh)
    assert (sx, sy) == pytest.approx((200.0, 0.0), abs=1e-6)


# ---- detection ------------------------------------------------------------------

def test_find_worms_finds_the_worm_where_the_scene_put_it(rig):
    img = np.asarray(Image.open(rig.camera.capture()))
    cands, mask = wf.find_worms(img, UM_PER_PX, **AREA, dark_ratio=0.85)
    assert len(cands) == 1 and mask.dtype == np.uint8 and mask.max() == 255
    c = cands[0]
    # worm at stage (+60, +30) um from a tile centred at the origin: right of and above centre
    assert c["x"] == pytest.approx(160 + 60 / UM_PER_PX, abs=6) and c["y"] == pytest.approx(120 - 30 / UM_PER_PX, abs=6)
    assert 0.001 < c["area_mm2"] < 0.02 and c["length_mm"] > 0.05 and c["width_um"] > 1


def test_find_worms_ignores_objects_outside_the_size_window(rig):
    img = np.asarray(Image.open(rig.camera.capture()))
    assert wf.find_worms(img, UM_PER_PX, 0.1, 1.0, 0.85)[0] == []
    assert wf.find_worms(img, UM_PER_PX, 1e-6, 1e-5, 0.85)[0] == []


def test_find_worms_on_empty_agar_finds_nothing(tmp_path, monkeypatch):
    rig = SceneRig(tmp_path, worm=False).install(monkeypatch)
    img = np.asarray(Image.open(rig.camera.capture()))
    assert wf.find_worms(img, UM_PER_PX, **AREA, dark_ratio=0.85)[0] == []


def test_pick_takes_the_nearest_within_range():
    cands = [{"x": 10, "y": 0, "area_mm2": 0.01}, {"x": 100, "y": 0, "area_mm2": 0.01}]
    assert wf.pick(cands, (0, 0), 50)["x"] == 10 and wf.pick(cands, (0, 0), 50)["dist_px"] == 10
    assert wf.pick(cands, (95, 0), 50)["x"] == 100
    assert wf.pick(cands, (500, 0), 50) is None and wf.pick([], (0, 0), 50) is None


def test_pick_does_not_hop_onto_a_worm_of_a_different_size():
    cands = [{"x": 5, "y": 0, "area_mm2": 0.002}, {"x": 40, "y": 0, "area_mm2": 0.011}]
    assert wf.pick(cands, (0, 0), 100, area_mm2=0.01)["x"] == 40           # the larva beside it is skipped
    assert wf.pick(cands, (0, 0), 100, area_mm2=0.0001) is None            # both more than 2x too big
    assert wf.pick(cands, (0, 0), 100)["x"] == 5                           # no history: nearest wins


def test_detect_only_writes_a_mask_and_reports(rig, tmp_path, monkeypatch, capsys):
    path = tmp_path / "shot.png"
    Image.fromarray(rig.camera.frame()).save(path)
    kind, (p, umpx, args) = parse(monkeypatch, "--detect-only", str(path), "--min-area-mm2", "0.001",
                                  "--max-area-mm2", "0.02", "--um-per-px", str(UM_PER_PX))[0]
    assert kind == "detect" and p == path
    wf.detect_only(p, umpx, args)
    assert (tmp_path / "shot_mask.png").exists()
    out = capsys.readouterr().out
    assert "1 candidate(s)" in out and "area" in out


def test_focus_score_prefers_the_sharper_image(rig):
    sharp = rig.camera.frame()
    rig.stage.z = Z_FOCUS + 60
    blurred = rig.camera.frame()
    assert wf._focus_score(sharp) > 5 * wf._focus_score(blurred)


# ---- setup ----------------------------------------------------------------------

def setup_args(monkeypatch, *argv):
    return parse(monkeypatch, "--setup", "--min-area-mm2", "0.001", "--max-area-mm2", "0.02", *argv)[0][1]


def test_setup_switches_objective_focuses_and_centres_on_the_worm(rig, tmp_path, monkeypatch, capsys):
    rig.stage.z = Z_FOCUS + 30                                    # start out of focus
    out = wf.setup(setup_args(monkeypatch))
    info = json.loads((out / "setup.json").read_text())
    assert info["nosepiece"] == 1 and rig.stage.micro.iNOSEPIECE == 1
    assert info["worm_found"] is True and info["z_before_um"] == Z_FOCUS + 30
    assert abs(rig.stage.z - Z_FOCUS) <= 10                       # the sweep found the sharp plane
    assert np.hypot(rig.stage.x - 60, rig.stage.y - 30) < 15      # and the worm was centred
    assert (out / "final.png").exists() and (out / "final_marked.jpg").exists()
    assert list(out.glob("coarse_z*.jpg")) and list(out.glob("fine_z*.jpg"))
    assert rig.camera.closed and "READY" in capsys.readouterr().out


def test_setup_leaves_the_objective_alone_when_it_is_already_right(rig, monkeypatch, capsys):
    rig.stage.micro.iNOSEPIECE = 1
    wf.setup(setup_args(monkeypatch))
    assert "nosepiece 2 ->" not in capsys.readouterr().out


def test_setup_halves_exposure_down_from_a_clipped_start_and_raises_it_from_dark(rig, monkeypatch):
    wf.setup(setup_args(monkeypatch, "--exposure-us", "8000"))
    assert 700 <= rig.camera.exposure_us <= 2200
    wf.setup(setup_args(monkeypatch, "--exposure-us", "300"))
    assert 1000 <= rig.camera.exposure_us <= 2200


def test_setup_reports_no_worm(tmp_path, monkeypatch, capsys):
    rig = SceneRig(tmp_path, worm=False).install(monkeypatch)
    out = wf.setup(setup_args(monkeypatch))
    assert json.loads((out / "setup.json").read_text())["worm_found"] is False
    assert "NO WORM FOUND" in capsys.readouterr().out and rig.stage.moves == []


def test_setup_keeps_z_when_the_sweep_is_flat(rig, monkeypatch, capsys):
    rig.camera.blur_per_um = 0
    wf.setup(setup_args(monkeypatch))
    assert "sweep is flat" in capsys.readouterr().out


def test_setup_refuses_under_estop(rig, monkeypatch):
    estop.engage("test")
    with pytest.raises(estop.EStopEngaged):
        wf.setup(setup_args(monkeypatch))
    assert rig.stage.micro.iNOSEPIECE == 2 and rig.stage.moves == []


def test_setup_gives_up_if_the_turret_never_arrives(rig, monkeypatch):
    class Micro:
        iNOSEPIECE = 2

        def __setattr__(self, name, value):
            pass                                                   # accepts the write, never turns

    rig.stage.micro = Micro()
    clock = iter(range(0, 10_000, 5))
    monkeypatch.setattr(mosaic.time, "monotonic", lambda: next(clock))
    with pytest.raises(SystemExit, match="did not reach"):
        wf.setup(setup_args(monkeypatch))
    assert rig.camera.closed


# ---- run ------------------------------------------------------------------------

def test_run_tracks_the_worm_and_recentres_on_it(rig, tmp_path, monkeypatch, capsys):
    out = wf.run(track_args(monkeypatch, "--rounds", "3", "--tag", "t"))
    r = rows(out)
    assert [x["found"] for x in r] == ["1", "1", "1"] and r[0]["recentred"] == "1"
    assert float(r[0]["worm_x_um"]) == pytest.approx(60, abs=15) and float(r[0]["worm_y_um"]) == pytest.approx(30, abs=15)
    assert len(list((out / "crops").glob("crop_*.jpg"))) == 3 and (out / "round_0002.jpg").exists()
    assert (out / "track.png").exists() and (out / "run.json").exists()
    assert np.hypot(rig.stage.x - 60, rig.stage.y - 30) < 15          # parked on the worm
    assert rig.camera.closed and "released" in capsys.readouterr().out


def test_run_widens_the_search_after_a_miss(tmp_path, monkeypatch):
    rig = SceneRig(tmp_path, worm=False).install(monkeypatch)
    out = wf.run(track_args(monkeypatch, "--rounds", "2"))
    r = rows(out)
    assert [x["grid"] for x in r] == ["1x1", "2x1"] and [x["found"] for x in r] == ["0", "0"]
    assert not (out / "track.png").exists()


def test_run_clamps_recentring_to_the_dish_limit(rig, monkeypatch, capsys):
    out = wf.run(track_args(monkeypatch, "--rounds", "1", "--dish-centre", "0,0", "--limit-mm", "0.02"))
    assert "clamped to the 0.02 mm limit" in capsys.readouterr().out
    assert np.hypot(rig.stage.x, rig.stage.y) <= 21


def test_run_records_a_camera_error_and_carries_on(rig, monkeypatch, capsys):
    rig.camera.on_capture = lambda n: (_ for _ in ()).throw(RuntimeError("fetch timed out")) if n in (2, 3) else None
    out = wf.run(track_args(monkeypatch, "--rounds", "2"))
    r = rows(out)
    assert [x["error"] for x in r] == ["RuntimeError: fetch timed out"] * 2
    assert "ERROR RuntimeError" in capsys.readouterr().out


def test_run_ends_the_series_on_an_estop(rig, monkeypatch):
    estop.engage("test")
    out = wf.run(track_args(monkeypatch, "--rounds", "5", "--grid", "2x1"))
    r = rows(out)
    assert len(r) == 1 and "E-STOP" in r[0]["error"] and rig.camera.closed


def test_run_halts_on_a_stop_file(rig, tmp_path, monkeypatch, capsys):
    def stop_at_second_round(n):
        if n == 3:                       # 1 = probe, 2 = round 0 tile, 3 = round 1 (checked before it)
            (only_run_dir(tmp_path) / "STOP").write_text("")
    rig.camera.on_capture = stop_at_second_round
    out = wf.run(track_args(monkeypatch, "--rounds", "9"))
    assert len(rows(out)) <= 3 and "STOP file present" in capsys.readouterr().out and rig.camera.closed


def test_run_paces_rounds_by_interval_and_stops_on_minutes(rig, monkeypatch):
    slept = []
    monkeypatch.setattr(mosaic.time, "sleep", lambda s: slept.append(s))
    wf.run(track_args(monkeypatch, "--rounds", "2", "--interval-s", "600"))
    assert any(s > 100 for s in slept)
    clock = iter(float(i) * 30 for i in range(10_000))
    monkeypatch.setattr(mosaic.time, "monotonic", lambda: next(clock))
    out = wf.run(track_args(monkeypatch, "--minutes", "1", "--tag", "m"))
    assert 1 <= len(rows(out)) <= 3


def test_run_can_remeasure_the_axes(rig, monkeypatch, capsys):
    wf.run(track_args(monkeypatch, "--rounds", "1", "--calibrate"))
    assert "um/px" in capsys.readouterr().out


@pytest.mark.parametrize("exposure,word", [("4000", "clipped"), ("300", "dark")])
def test_run_warns_when_the_agar_is_clipped_or_dark(rig, monkeypatch, capsys, exposure, word):
    wf.run(track_args(monkeypatch, "--rounds", "1", "--exposure-us", exposure))
    assert f"agar is {word}" in capsys.readouterr().err


def test_run_parks_even_if_parking_fails(rig, monkeypatch, capsys):
    real = wf._move_safe
    calls = []

    def flaky(sdk, x, y):
        calls.append(1)
        if len(calls) >= 2:
            raise RuntimeError("stage gone")
        return real(sdk, x, y)

    monkeypatch.setattr(wf, "_move_safe", flaky)
    wf.run(track_args(monkeypatch, "--rounds", "1", "--grid", "2x1"))
    assert "could not park at the last centre" in capsys.readouterr().err and rig.camera.closed


# ---- track plot --------------------------------------------------------------------

def write_track(path, found_rows):
    with (path / "track.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["found", "worm_x_um", "worm_y_um", "t_s"])
        w.writeheader()
        w.writerows(found_rows)


def test_plot_track_draws_found_points_only(tmp_path):
    write_track(tmp_path, [{"found": 0, "worm_x_um": "", "worm_y_um": "", "t_s": 0},
                           {"found": 1, "worm_x_um": 0, "worm_y_um": 0, "t_s": 60},
                           {"found": 1, "worm_x_um": 500, "worm_y_um": 300, "t_s": 120},
                           {"found": 1, "worm_x_um": 900, "worm_y_um": 100, "t_s": 180}])
    wf.plot_track(tmp_path)
    assert Image.open(tmp_path / "track.png").size == (900, 900)


def test_plot_track_with_nothing_found_writes_nothing(tmp_path):
    write_track(tmp_path, [{"found": 0, "worm_x_um": "", "worm_y_um": "", "t_s": 0}])
    wf.plot_track(tmp_path)
    assert not (tmp_path / "track.png").exists()


# ---- command line -------------------------------------------------------------------

def test_main_dispatches_setup_detect_and_run(monkeypatch, tmp_path):
    assert parse(monkeypatch, "--setup")[0][0] == "setup"
    assert parse(monkeypatch, "--detect-only", str(tmp_path / "x.png"))[0][0] == "detect"
    kind, args = parse(monkeypatch)[0]
    assert kind == "run" and args.rounds == 1 and args.grid == "3x3" and args.search_grid == "5x5"
    kind, args = parse(monkeypatch, "--minutes", "5")[0]
    assert args.rounds is None and args.minutes == 5
