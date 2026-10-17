# analysis/make_movie.py, oat_approach.py, physarum_rhythm.py and live_view.py
# against small synthetic Physarum series.
import csv
import datetime
import json
import os
import runpy
import sys
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from analysis import live_view, make_movie as mm, oat_approach as oa, physarum_rhythm as pr


def can_write_mp4(tmp_path):
    w = cv2.VideoWriter(str(tmp_path / "probe.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    ok = w.isOpened()
    w.release()
    return ok


@pytest.fixture
def mp4(tmp_path):
    if not can_write_mp4(tmp_path):
        pytest.skip("this OpenCV build cannot write mp4v video")


def write_png(path, value=120, size=(48, 96), bgr=None):
    img = np.full((*size, 3), value, np.uint8)
    if bgr is not None:
        img[:] = bgr
    cv2.imwrite(str(path), img)
    return path


# ---- make_movie ------------------------------------------------------------------------------

def mosaic_series(tmp_path, rounds=3, missing=(), scale=None):
    d = tmp_path / "mosaic_run"
    d.mkdir()
    t0 = datetime.datetime(2026, 9, 21, 10, 0, 0)
    with (d / "tiles.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "tile", "wall_clock"])
        for r in range(rounds):
            for tile in range(2):
                w.writerow([r, tile, (t0 + datetime.timedelta(minutes=5 * r, seconds=tile)).isoformat()])
    for r in range(rounds):
        if r not in missing:
            write_png(d / f"mosaic_{r:03d}.png", 100 + r * 20)
    meta = {"um_per_px": 3.0}
    if scale is not None:
        meta["mosaic_scale"] = scale
    (d / "mosaic.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def frames_series(tmp_path, n=4):
    d = tmp_path / "tl"
    d.mkdir()
    with (d / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "t_seconds", "filename", "error"])
        w.writeheader()
        for i in range(n):
            w.writerow({"frame": i, "t_seconds": i * 10, "filename": f"f{i:05d}.png", "error": ""})
            write_png(d / f"f{i:05d}.png", 80 + 10 * i)
        w.writerow({"frame": n, "t_seconds": n * 10, "filename": "", "error": "fetch timed out"})
        w.writerow({"frame": n + 1, "t_seconds": (n + 1) * 10, "filename": "gone.png", "error": ""})
    return d


def test_mosaic_rows_are_timed_from_the_first_round(tmp_path):
    d = mosaic_series(tmp_path, rounds=3, missing=(1,))
    assert mm._mosaic_rows(d) == [{"filename": "mosaic_000.png", "t_seconds": 0.0},
                                  {"filename": "mosaic_002.png", "t_seconds": 600.0}]
    (d / "tiles.csv").write_text("round,tile,wall_clock\n", encoding="utf-8")
    assert mm._mosaic_rows(d) == []


def test_mosaic_pixel_size_defaults_to_the_old_half_scale(tmp_path):
    assert mm.mosaic_um_per_px(tmp_path) is None
    assert mm.mosaic_um_per_px(mosaic_series(tmp_path)) == 6.0
    other = tmp_path / "b"
    other.mkdir()
    (other / "mosaic.json").write_text(json.dumps({"um_per_px": 3.0, "mosaic_scale": 0.25}))
    assert mm.mosaic_um_per_px(other) == 12.0


def test_rows_skip_failed_and_missing_frames(tmp_path):
    rows = mm._rows(frames_series(tmp_path))
    assert [r["filename"] for r in rows] == ["f00000.png", "f00001.png", "f00002.png", "f00003.png"]


def test_rows_prefer_the_mosaic_rounds_when_there_are_any(tmp_path):
    assert [r["filename"] for r in mm._rows(mosaic_series(tmp_path))] == [
        "mosaic_000.png", "mosaic_001.png", "mosaic_002.png"]


def blue_cast(h=100, w=100):
    """Agar with the camera's blue-biased cast (BGR 237/168/145) and a little texture."""
    noise = np.random.default_rng(0).integers(-6, 7, (h, w, 1))
    return np.clip(np.array([237, 168, 145]) + noise, 0, 249).astype(np.uint8)


def test_white_balance_neutralises_the_bright_background():
    img = blue_cast()
    g = mm.wb_gains(img)
    assert g[0] < 1 < g[2]                                      # blue turned down, red turned up
    balanced = (img.astype(float) * g).reshape(-1, 3).mean(axis=0)
    assert np.ptp(balanced) < 0.1 * balanced.mean()


def test_white_balance_refuses_to_trust_clipped_or_tiny_references():
    assert np.array_equal(mm.wb_gains(np.full((100, 100, 3), 255, np.uint8)), np.ones(3))   # all clipped
    small = np.full((10, 10, 3), 200, np.uint8)
    assert np.array_equal(mm.wb_gains(small), np.ones(3))                                  # under 1000 pixels
    img = np.full((100, 100, 3), 100, np.uint8)
    img[:3, :3] = 240                                                                      # a handful of bright pixels
    assert np.array_equal(mm.wb_gains(img), np.ones(3))
    no_blue = blue_cast()
    no_blue[:, :, 0] = 0
    assert np.array_equal(mm.wb_gains(no_blue), np.ones(3))                                 # a dead channel


def test_scale_bar_uses_a_round_number_of_microns():
    for um_per_px, label in ((1.0, "20 um"), (4.0, "100 um"), (60.0, "2 mm"), (0.01, "10 um")):
        img = np.full((200, 400, 3), 128, np.uint8)
        before = img.copy()
        mm._scale_bar(img, um_per_px)
        assert not np.array_equal(img, before), label


def test_timestamp_overlay_draws_something():
    img = np.full((100, 300, 3), 50, np.uint8)
    mm._stamp(img, 3725.0, 4, 10)
    assert (img != 50).any()


def test_build_writes_a_video_from_a_timelapse(tmp_path, mp4, capsys):
    d = frames_series(tmp_path)
    out = mm.build(d, tmp_path / "out.mp4", fps=5, scale=1.0, um_per_px=1.5, overlay=True)
    assert out.exists() and out.stat().st_size > 0
    text = capsys.readouterr().out
    assert "white balance gains" in text and "4 frames spanning" in text


def test_build_variants(tmp_path, mp4):
    d = frames_series(tmp_path)
    mm.build(d, tmp_path / "a.mp4", 5, 0.5, 1.5, overlay=False, white_balance=False)
    assert (tmp_path / "a.mp4").exists()
    m = mosaic_series(tmp_path)
    mm.build(m, tmp_path / "b.mp4", 5, 1.0, 6.0, overlay=True)
    assert (tmp_path / "b.mp4").exists()


def test_build_skips_an_unreadable_frame_but_not_a_missing_first_one(tmp_path, mp4):
    d = frames_series(tmp_path)
    (d / "f00002.png").write_bytes(b"not an image")
    assert mm.build(d, tmp_path / "c.mp4", 5, 1.0, 1.5, overlay=False).exists()
    (d / "f00000.png").write_bytes(b"not an image")
    with pytest.raises(SystemExit, match="could not read"):
        mm.build(d, tmp_path / "d.mp4", 5, 1.0, 1.5, overlay=False)


def test_build_needs_frames_and_a_working_codec(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "frames.csv").write_text("frame,t_seconds,filename,error\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no usable frames"):
        mm.build(empty, tmp_path / "x.mp4", 5, 1.0, 1.5, True)
    d = frames_series(tmp_path)

    class NoCodec:
        def __init__(self, *a):
            pass

        def isOpened(self):
            return False

    monkeypatch.setattr(mm.cv2, "VideoWriter", NoCodec)
    with pytest.raises(SystemExit, match="could not open the video writer"):
        mm.build(d, tmp_path / "y.mp4", 5, 1.0, 1.5, True)


def test_movie_main_picks_the_pixel_size_and_flags(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(mm, "build", lambda *a: seen.append(a))
    d = mosaic_series(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_movie", str(d), "--fps", "10", "--no-overlay", "--no-white-balance"])
    mm.main()
    assert seen[0][1] == d / "movie.mp4" and seen[0][4] == 6.0 and seen[0][5:] == (False, False)
    monkeypatch.setattr(sys, "argv", ["make_movie", str(d), "--um-per-px", "2.5", "--out", str(tmp_path / "o.mp4")])
    mm.main()
    assert seen[1][4] == 2.5 and seen[1][1] == tmp_path / "o.mp4"
    plain = frames_series(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_movie", str(plain)])
    mm.main()
    assert seen[2][4] == mm.UM_PER_PX


# ---- oat_approach ------------------------------------------------------------------------------

def dish_image(growth=40, size=800):
    """Glass disc, an agar block on it, a black oat on the agar and a plasmodium patch."""
    img = np.zeros((size, size), np.uint8)
    cv2.circle(img, (400, 400), 340, 250, -1)               # bare glass
    cv2.circle(img, (400, 400), 200, 150, -1)               # agar
    cv2.circle(img, (300, 400), 20, 0, -1)                  # the oat
    cv2.circle(img, (420, 400), growth, 60, -1)             # plasmodium
    return img


def oat_series(tmp_path, rounds=2):
    d = tmp_path / "oat_run"
    d.mkdir()
    t0 = datetime.datetime(2026, 9, 21, 10, 0, 0)
    with (d / "tiles.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["round", "tile", "wall_clock"])
        for r in range(rounds):
            w.writerow([r, 0, (t0 + datetime.timedelta(minutes=10 * r)).isoformat()])
    for r in range(rounds):
        img = dish_image(growth=40 + 30 * r)
        if r == 1:
            img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)   # a later round at another size
        cv2.imwrite(str(d / f"mosaic_{r:03d}.png"), img)
    (d / "mosaic.json").write_text(json.dumps({"um_per_px": 2.0, "mosaic_scale": 0.5}), encoding="utf-8")
    return d


def test_masks_find_the_dish_agar_and_oat():
    g = cv2.resize(dish_image(), None, fx=oa.WORK_SCALE, fy=oa.WORK_SCALE, interpolation=cv2.INTER_AREA)
    dish = oa.dish_mask(g)
    assert dish[100, 100] == 1 and dish[2, 2] == 0
    agar = oa.agar_mask(g, dish)
    assert agar[100, 100] == 1 and agar[100, 190] == 0
    oat = oa.oat_mask(g, dish, (250, 350, 350, 450))
    assert oat[100, 75] == 1 and oat.sum() < 200
    with pytest.raises(SystemExit, match="no black object"):
        oa.oat_mask(g, dish, (600, 600, 700, 700))


def test_run_measures_how_far_the_plasmodium_is_from_the_oat(tmp_path, capsys):
    d = oat_series(tmp_path)
    out = oa.run(d, (250, 350, 350, 450))
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert [r["round"] for r in rows] == ["0", "1"] and (d / "oat_approach_check.png").exists()
    assert float(rows[1]["total_mm2"]) > float(rows[0]["total_mm2"]) > 0
    assert float(rows[0]["nearest_mm"]) < float(rows[0]["mean_dist_mm"])
    assert any(float(rows[0][k]) > 0 for k in rows[0] if k.startswith("band_"))
    assert "oat area" in capsys.readouterr().out


def test_run_without_stitched_rounds_stops(tmp_path):
    d = oat_series(tmp_path)
    for p in d.glob("mosaic_*.png"):
        p.unlink()
    with pytest.raises(SystemExit, match="no stitched rounds"):
        oa.run(d, (250, 350, 350, 450))


def test_oat_main_parses_the_box(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(oa, "run", lambda d, box: seen.append((d, box)))
    monkeypatch.setattr(sys, "argv", ["oat", str(tmp_path), "--oat-box", "1,2,3,4"])
    oa.main()
    assert seen == [(tmp_path, (1, 2, 3, 4))]


# ---- physarum_rhythm -------------------------------------------------------------------------------

def rhythm_series(tmp_path, n=60, dt=10.0, period=120.0, calibrated=True, osc=True):
    """Left half: dark organism pulsing with `period`; right half: steady bright agar."""
    d = tmp_path / "rhythm"
    d.mkdir()
    rng = np.random.default_rng(0)
    with (d / "frames.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "t_seconds", "filename", "error"])
        w.writeheader()
        for i in range(n):
            t = i * dt
            img = np.full((100, 100), 200.0)
            img[:, :50] = 80 + (18 * np.sin(2 * np.pi * t / period) if osc else 0)
            img[:, :50] += np.linspace(-12, 12, 50)               # a real image is never two flat values
            img[:, 50:] += np.linspace(-12, 12, 50)
            img += rng.normal(0, 0.6, img.shape)
            Image.fromarray(np.clip(np.stack([img] * 3, -1), 0, 255).astype(np.uint8)).save(d / f"f{i:05d}.png")
            w.writerow({"frame": i, "t_seconds": t, "filename": f"f{i:05d}.png", "error": ""})
    if calibrated:
        (d / "calibration.json").write_text(json.dumps({"um_per_px": 1.5}), encoding="utf-8")
    return d


def test_pixel_size_comes_only_from_a_measurement(tmp_path):
    assert pr._um_per_px(tmp_path) is None
    (tmp_path / "calibration.json").write_text("{not json")
    assert pr._um_per_px(tmp_path) is None
    (tmp_path / "calibration.json").write_text(json.dumps({"um_per_px": 0}))
    assert pr._um_per_px(tmp_path) is None
    (tmp_path / "calibration.json").write_text(json.dumps({"um_per_px": 1.49}))
    assert pr._um_per_px(tmp_path) == 1.49


def test_series_loading_skips_bad_rows_and_demands_enough_frames(tmp_path):
    d = rhythm_series(tmp_path)
    with (d / "frames.csv").open("a", encoding="utf-8") as f:
        f.write("60,600,,boom\n61,610,missing.png,\n")
    times, stack, names = pr._load_series(d)
    assert len(times) == 60 and stack.shape == (60, 20, 20) and names[0] == "f00000.png"
    few = tmp_path / "few"
    few.mkdir()
    (few / "frames.csv").write_text("frame,t_seconds,filename,error\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="too few to analyse"):
        pr._load_series(few)


def test_dominant_period_finds_the_pulse_and_rejects_noise():
    t = np.arange(60) * 10.0
    pulse = np.sin(2 * np.pi * t / 120.0)
    noise = np.random.default_rng(1).normal(0, 1, 60)
    period, strength = pr._dominant_period(np.column_stack([pulse, noise]), 10.0)
    assert period[0] == pytest.approx(120.0, rel=0.1) and strength[0] > 0.5 > strength[1]


def test_dominant_period_needs_a_search_band():
    with pytest.raises(SystemExit, match="no FFT bins"):
        pr._dominant_period(np.random.default_rng(0).normal(size=(10, 3)), 5000.0)


def test_regions_split_bright_from_dark(tmp_path):
    _, stack, _ = pr._load_series(rhythm_series(tmp_path))
    bright, dark = pr._regions(stack)
    assert dark[:, :8].all() and bright[:, 12:].all() and not (bright & dark).any()


def test_the_pulsing_region_is_identified_as_the_organism(tmp_path):
    res = pr.analyse(rhythm_series(tmp_path))
    assert res["organism_region"] == "dark" and res["organism_identified_by"] == "rhythmicity"
    assert res["period_s"] == pytest.approx(120.0, rel=0.15) and res["rhythmic_margin"] > 0.3
    assert res["frames_used"] == 60 and res["interval_s"] == 10.0 and res["um_per_px"] == 1.5
    assert res["area_start_mm2"] is not None and res["growth_rate_mm2_per_h"] is not None
    assert len(res["trace_t_s"]) == len(res["trace_intensity"]) == len(res["trace_area_px"]) == 60
    assert set(res["period_map_percentiles"]) == {"5", "25", "50", "75", "95"}
    assert res["region_scores"]["dark"]["rhythmic_fraction"] > res["region_scores"]["bright"]["rhythmic_fraction"]


def test_without_calibration_areas_stay_in_pixels(tmp_path):
    res = pr.analyse(rhythm_series(tmp_path, calibrated=False))
    assert res["um_per_px"] is None and res["area_start_mm2"] is None and res["growth_rate_mm2_per_h"] is None
    assert res["area_start_px"] > 0


def test_rhythm_main_reports_and_writes_json(tmp_path, monkeypatch, capsys):
    d = rhythm_series(tmp_path)
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", ["physarum_rhythm", str(d), "--json", str(out)])
    pr.main()
    text = capsys.readouterr().out
    assert "contraction period" in text and "growth rate" in text and "measured" in text
    assert json.loads(out.read_text())["organism_region"] == "dark"


def test_rhythm_main_warns_when_the_regions_cannot_be_told_apart(tmp_path, monkeypatch, capsys):
    d = rhythm_series(tmp_path, calibrated=False, osc=False)         # nothing pulses anywhere
    monkeypatch.setattr(sys, "argv", ["physarum_rhythm", str(d)])
    pr.main()
    text = capsys.readouterr().out
    assert "WARNING" in text and "no calibration.json" in text


# ---- live_view ---------------------------------------------------------------------------------------------

def touch(path, age_s):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), (9, 9, 9)).save(path)
    t = time.time() - age_s
    os.utime(path, (t, t))
    return path


def test_newest_image_ignores_the_published_copy(tmp_path):
    assert live_view.newest_image(tmp_path) is None
    touch(tmp_path / "a.png", 100)
    new = touch(tmp_path / "sub" / "b.png", 10)
    touch(tmp_path / "latest.jpg", 0)
    assert live_view.newest_image(tmp_path) == new


def test_newest_run_looks_only_at_mosaic_and_timelapse_folders(tmp_path):
    assert live_view.newest_run(tmp_path) is None
    old = tmp_path / "mosaic_a"
    new = tmp_path / "timelapse_b"
    other = tmp_path / "notes"
    for p, age in ((old, 100), (new, 10), (other, 0)):
        p.mkdir()
        os.utime(p, (time.time() - age,) * 2)
    (tmp_path / "mosaic_file").write_text("x")
    assert live_view.newest_run(tmp_path) == new


class Stop(BaseException):
    pass


def run_serve(monkeypatch, watch, out, ticks):
    calls = []

    def sleep(s):
        calls.append(s)
        if len(calls) >= ticks:
            raise Stop

    monkeypatch.setattr(live_view.time, "sleep", sleep)
    with pytest.raises(Stop):
        live_view.serve(watch, out, fps=2.0, width=32)
    return calls


def test_serve_publishes_a_frame_only_once_its_size_has_settled(tmp_path, monkeypatch, capsys):
    watch = tmp_path / "mosaic_run"
    touch(watch / "mosaic_000.png", 5)
    out = tmp_path / "live"
    calls = run_serve(monkeypatch, watch, out, ticks=3)
    assert calls == [0.5, 0.5, 0.5]
    assert Image.open(out / "latest.jpg").size[0] <= 32
    page = (out / "index.html").read_text(encoding="utf-8")
    assert "mosaic_run" in page and "500" in page and "__DIR__" not in page
    assert "OPEN THIS" in capsys.readouterr().out


def test_serve_retries_a_truncated_image(tmp_path, monkeypatch):
    watch = tmp_path / "timelapse_run"
    watch.mkdir()
    (watch / "f.png").write_bytes(b"\x89PNG not finished")
    run_serve(monkeypatch, watch, tmp_path / "live", ticks=4)
    assert not (tmp_path / "live" / "latest.jpg").exists()


def test_live_main_needs_a_run_to_watch(tmp_path, monkeypatch):
    monkeypatch.setattr(live_view, "data_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["live_view"])
    with pytest.raises(SystemExit, match="no run folder"):
        live_view.main()


def test_live_main_serves_the_given_or_newest_run(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(live_view, "serve", lambda *a: seen.append(a))
    monkeypatch.setattr(sys, "argv", ["live_view", str(tmp_path), "--fps", "4", "--width", "640"])
    live_view.main()
    assert seen[0] == (tmp_path, tmp_path / "_live", 4.0, 640)
    newest = tmp_path / "data" / "mosaic_x"
    newest.mkdir(parents=True)
    monkeypatch.setattr(live_view, "data_root", lambda: tmp_path)
    monkeypatch.setattr(sys, "argv", ["live_view"])
    live_view.main()
    assert seen[1][0] == newest


def test_analysis_scripts_run_with_dash_m(monkeypatch):
    for module in ("analysis.make_movie", "analysis.oat_approach", "analysis.physarum_rhythm", "analysis.live_view"):
        monkeypatch.setattr(sys, "argv", ["x", "--help"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_module(module, run_name="__main__")
        assert exc.value.code == 0


class FakeEntry:
    name = "x.png"

    def stat(self):
        raise OSError("vanished")


def test_a_file_that_vanishes_mid_scan_is_skipped(tmp_path, monkeypatch):
    keep = touch(tmp_path / "keep.png", 5)
    monkeypatch.setattr(live_view.Path, "rglob", lambda self, pattern: [FakeEntry(), keep])
    assert live_view.newest_image(tmp_path) == keep


def test_serve_keeps_watching_after_an_error(tmp_path, monkeypatch):
    def boom(root):
        raise RuntimeError("disk hiccup")

    monkeypatch.setattr(live_view, "newest_image", boom)
    calls = run_serve(monkeypatch, tmp_path, tmp_path / "live", ticks=3)
    assert len(calls) == 3
