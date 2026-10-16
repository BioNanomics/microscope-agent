# analysis/aml18_survey.py and analysis/aml18_track.py on a synthetic worm scene
# (a dark worm on agar, glowing neurons at its head) behind a fake ND2 reader.
import csv
import importlib
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.fakes_nd2 import UM_PX, install_nd2, neurons, two_channel_stack, worm_td

PATH = Path("scene.nd2")


@pytest.fixture
def survey(monkeypatch):
    install_nd2(monkeypatch, two_channel_stack(), list("TZCYX"), ["TD", "RFP"], times=[0.0, 600.0])
    sys.modules.pop("analysis.aml18_survey", None)
    mod = importlib.import_module("analysis.aml18_survey")
    yield mod
    sys.modules.pop("analysis.aml18_survey", None)


def channels(td, rfp):
    return {"TD": td, "RFP": rfp}


# ---- loading -------------------------------------------------------------------------

def test_load_collapses_z_and_reads_times(survey):
    frames, um_px = survey.load(PATH)
    assert um_px == UM_PX and [t for t, _ in frames] == [0.0, 600.0]
    td, rfp = frames[0][1]["TD"], frames[0][1]["RFP"]
    expected = two_channel_stack()[0, 1]
    assert np.allclose(td, expected[0])                            # the sharp (middle) plane
    assert np.allclose(rfp, expected[1])                           # max projection of the stack


def test_load_falls_back_to_frame_numbers_without_event_times(monkeypatch, survey):
    install_nd2(monkeypatch, two_channel_stack(), list("TZCYX"), ["TD", "RFP"], times=None)
    assert [t for t, _ in survey.load(PATH)[0]] == [0.0, 1.0]


def test_load_adds_missing_axes(monkeypatch, survey):
    install_nd2(monkeypatch, worm_td()[None], ["C", "Y", "X"], ["TD"])
    frames, _ = survey.load(PATH)
    assert len(frames) == 1 and frames[0][1]["TD"].shape == (256, 256)


def test_load_refuses_multipoint_files(monkeypatch, survey):
    install_nd2(monkeypatch, two_channel_stack(), list("TZCYX"), ["TD", "RFP"], extra_sizes={"P": 4})
    with pytest.raises(SystemExit, match="unsupported dimensions"):
        survey.load(PATH)


# ---- segmentation and measurement -------------------------------------------------------

def test_segment_worms_keeps_the_worm_and_drops_debris(survey):
    td = worm_td()
    cv2.circle(td, (200, 40), 3, 60.0, -1)                          # a speck well under 2000 um2
    mask = survey.segment_worms(td, UM_PX, 0.85)
    assert mask[128, 100] and not mask[40, 200] and mask.dtype == bool


def test_longest_path_length_counts_diagonals(survey):
    skel = np.zeros((20, 20), bool)
    skel[5, 2:12] = True                                            # 9 steps of 1
    path, length = survey.longest_path(skel)
    assert length == pytest.approx(9.0) and len(path) == 10
    diag = np.eye(10, dtype=bool)
    assert survey.longest_path(diag)[1] == pytest.approx(9 * np.sqrt(2))


def test_neuron_mask_threshold_sits_above_the_background(survey):
    mask, thr = survey.neuron_mask(neurons(), UM_PX)
    assert thr >= 5.0 and mask[128, 62] and not mask[10, 10]
    flat, thr = survey.neuron_mask(np.zeros((32, 32), np.float32), UM_PX)
    assert not flat.any() and thr == 5.0


# ---- one frame ------------------------------------------------------------------------------

def test_survey_frame_finds_the_worm_and_its_head(survey):
    rows, img, thr = survey.survey_frame(channels(worm_td(), neurons()), UM_PX, "RFP", 150, 0.85, "t")
    assert len(rows) == 1 and img.shape == (512, 512, 3) and img.dtype == np.uint8
    w = rows[0]
    assert w["found_by"] == "TD" and w["stage_estimate"] == "L3" and 0.4 < w["length_mm"] < 0.5
    assert abs(w["head_x_px"] - 62) < 8 and abs(w["head_y_px"] - 128) < 8        # the end with the neurons
    assert w["neuron_px"] > 0 and w["width_um"] > 0


def test_survey_frame_needs_both_channels(survey):
    with pytest.raises(SystemExit, match="need TD and RFP"):
        survey.survey_frame({"TD": worm_td()}, UM_PX, "RFP", 150, 0.85, "t")


def test_a_dark_object_with_no_neurons_is_not_a_worm(survey):
    rows, _, _ = survey.survey_frame(channels(worm_td(), np.zeros((256, 256), np.float32)), UM_PX, "RFP", 150, 0.85, "t")
    assert rows == []


def test_short_objects_and_eggs_are_dropped(survey):
    short = worm_td(p0=(100, 128), p1=(120, 128))                   # 60 um long
    assert survey.survey_frame(channels(short, neurons()), UM_PX, "RFP", 150, 0.85, "t")[0] == []
    egg = np.full((256, 256), 190.0, np.float32)
    cv2.circle(egg, (128, 128), 18, 60.0, -1)                       # as wide as it is long
    rows, _, _ = survey.survey_frame(channels(egg, neurons(centres=((128, 128),))), UM_PX, "RFP", 50, 0.85, "t")
    assert rows == []


def test_neurons_at_both_ends_give_no_head_call(survey):
    both = neurons(centres=((62, 128), (70, 129), (190, 139), (198, 140)))
    rows, _, _ = survey.survey_frame(channels(worm_td(), both), UM_PX, "RFP", 150, 0.85, "t")
    assert rows[0]["head_x_px"] == ""


def test_a_head_at_the_frame_edge_is_not_called(survey):
    td = worm_td(p0=(0, 128), p1=(140, 140))
    rows, _, _ = survey.survey_frame(channels(td, neurons(centres=((1, 128), (6, 128), (12, 128)))),
                                     UM_PX, "RFP", 150, 0.85, "t")
    assert rows and rows[0]["head_x_px"] == "" and "cut by edge" in rows[0]["stage_estimate"]


def test_larvae_seen_only_by_their_neurons_are_found(survey):
    flat = np.full((256, 256), 190.0, np.float32)
    chain = neurons(centres=[(60 + 8 * i, 128) for i in range(10)], amp=300.0, sigma=2.0)
    rows, img, _ = survey.survey_frame(channels(flat, chain), UM_PX, "RFP", 100, 0.85, "t")
    assert len(rows) == 1 and rows[0]["found_by"] == "neurons" and rows[0]["width_um"] == ""


@pytest.mark.parametrize("p1_x,stage", [(77, "L1"), (130, "L2"), (170, "L3"), (243, "L4")])
def test_stage_follows_body_length(survey, p1_x, stage):
    td = worm_td(p0=(10, 128), p1=(p1_x, 128))
    rows, _, _ = survey.survey_frame(channels(td, neurons(centres=((14, 128), (22, 128), (30, 128)))),
                                     UM_PX, "RFP", 150, 0.85, "t")
    assert len(rows) == 1 and rows[0]["stage_estimate"] == stage


# ---- whole file -----------------------------------------------------------------------------------

def test_analyse_a_time_series_writes_images_and_tables(survey, tmp_path, capsys):
    out = survey.analyse(PATH, "RFP", 150, 0.85, out=tmp_path / "out")
    assert (out / "survey_t000.png").exists() and (out / "survey_t001.png").exists()
    rows = list(csv.DictReader((out / "worms.csv").open(encoding="utf-8")))
    assert [r["t_index"] for r in rows] == ["0", "1"] and "_path" not in rows[0]
    counts = list(csv.DictReader((out / "counts.csv").open(encoding="utf-8")))
    assert counts[0]["worms"] == "1" and counts[0]["heads_found"] == "1"
    assert "t=  0.0 min" in capsys.readouterr().out


def test_analyse_a_single_image_reports_each_worm(monkeypatch, survey, tmp_path, capsys):
    install_nd2(monkeypatch, two_channel_stack(t=1), list("TZCYX"), ["TD", "RFP"], times=[0.0])
    out = survey.analyse(PATH, "RFP", 150, 0.85)                    # default folder next to the file
    try:
        assert (out / "survey.png").exists() and not (out / "counts.csv").exists()
        assert "head found" in capsys.readouterr().out and out.name == "scene_survey"
    finally:
        import shutil
        shutil.rmtree(out, ignore_errors=True)


def test_analyse_with_no_worms_still_writes_a_table(monkeypatch, survey, tmp_path):
    empty = np.full((1, 1, 2, 64, 64), 190.0, np.float32)
    empty[:, :, 1] = 0
    install_nd2(monkeypatch, empty, list("TZCYX"), ["TD", "RFP"], times=[0.0])
    out = survey.analyse(PATH, "RFP", 150, 0.85, out=tmp_path / "o")
    assert (out / "worms.csv").read_text(encoding="utf-8").startswith("t_index,t_s,worm")


def test_analyse_stops_if_the_image_cannot_be_written(survey, tmp_path, monkeypatch):
    monkeypatch.setattr(survey.cv2, "imwrite", lambda *a: False)
    with pytest.raises(SystemExit, match="could not write"):
        survey.analyse(PATH, "RFP", 150, 0.85, out=tmp_path / "o")


def test_survey_main_parses_the_command_line(survey, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(survey, "analyse", lambda *a: seen.append(a))
    monkeypatch.setattr(sys, "argv", ["aml18_survey", "f.nd2", "--neuron-channel", "GFP", "--min-length-um", "99",
                                      "--dark-ratio", "0.7", "--out", str(tmp_path)])
    survey.main()
    assert seen == [(Path("f.nd2"), "GFP", 99.0, 0.7, tmp_path)]


# ---- aml18_track --------------------------------------------------------------------------------------

@pytest.fixture
def track(survey):
    sys.modules.pop("analysis.aml18_track", None)
    mod = importlib.import_module("analysis.aml18_track")
    yield mod
    sys.modules.pop("analysis.aml18_track", None)


def row(t, x, y, length, stage="L3 ", t_s=None):
    return {"t_index": t, "t_s": t * 60 if t_s is None else t_s, "x_px": x, "y_px": y,
            "length_mm": length, "stage_estimate": stage}


def test_link_follows_a_worm_and_starts_new_tracks_for_strangers(track):
    rows = [row(0, 100, 100, 0.5), row(0, 800, 800, 0.5),
            row(1, 110, 100, 0.5),                       # moved 10 px (~15 um): same worm
            row(1, 1500, 100, 0.5),                      # far from everything: a new worm
            row(2, 112, 100, 1.5)]                       # 3x longer: not the same worm
    out = track.link(rows, um_px=1.5, max_step_mm=1.0)
    assert out[2]["track"] == out[0]["track"] and out[2]["step_mm"] == pytest.approx(0.015)
    assert len({r["track"] for r in out}) == 4
    assert out[0]["step_mm"] == "" and out[4]["track"] not in (out[0]["track"], out[2]["track"])


def test_link_gives_each_worm_at_most_one_match(track):
    rows = [row(0, 100, 100, 0.5), row(1, 105, 100, 0.5), row(1, 108, 100, 0.5)]
    out = track.link(rows, um_px=1.5, max_step_mm=1.0)
    assert out[1]["track"] == out[0]["track"] and out[2]["track"] != out[0]["track"]


def test_summarise_gives_distance_speed_and_stage(track):
    rows = track.link([row(0, 0, 0, 0.5, "L2 x"), row(1, 100, 0, 0.5, "L3 y"), row(2, 200, 0, 0.5, "L3 y"),
                       row(0, 900, 900, 0.4, "L1 ")], 2.0, 1.0)
    s = {r["track"]: r for r in track.summarise(rows)}
    first = s[min(s)]
    assert first["frames"] == 3 and first["distance_mm"] == 0.4 and first["stage"] == "L3"
    assert first["mean_speed_mm_per_min"] == pytest.approx(0.2) and first["median_length_mm"] == 0.5
    lone = s[max(s)]
    assert lone["frames"] == 1 and lone["mean_speed_mm_per_min"] == ""


def test_draw_tracks_writes_an_overlay_or_nothing(track, tmp_path):
    base = tmp_path / "survey_t000.png"
    cv2.imwrite(str(base), np.full((200, 200, 3), 90, np.uint8))
    rows = track.link([row(0, 20, 20, 0.5), row(1, 30, 25, 0.5), row(0, 80, 80, 0.5)], 1.5, 1.0)
    dest = tmp_path / "tracks.png"
    track.draw_tracks(base, rows, dest)
    assert dest.exists()
    track.draw_tracks(tmp_path / "missing.png", rows, tmp_path / "none.png")
    assert not (tmp_path / "none.png").exists()


def fake_ffmpeg(monkeypatch):
    sent = []

    class Writer:
        def send(self, frame):
            sent.append(frame)

        def close(self):
            sent.append("closed")

    mod = types.ModuleType("imageio_ffmpeg")
    mod.write_frames = lambda dest, size, **kw: (sent.append(("open", dest, size, kw["fps"])), Writer())[1]
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", mod)
    return sent


def test_movie_needs_two_frames_and_writes_even_dimensions(track, tmp_path, monkeypatch):
    sent = fake_ffmpeg(monkeypatch)
    assert track.movie(tmp_path, 6) is None
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"survey_t{i:03d}.png"), np.full((101, 99, 3), i * 40, np.uint8))
    dest = track.movie(tmp_path, 6)
    assert dest == tmp_path / "survey_movie.mp4"
    assert sent[0] == ("open", str(dest), (98, 100), 6) and sent[-1] == "closed"
    assert sum(isinstance(s, np.ndarray) for s in sent) == 3


def test_signal_per_frame_reports_the_brightest_neurons(track):
    sig = track.signal_per_frame(PATH, "RFP")
    assert [s["t_min"] for s in sig] == [0.0, 10.0] and sig[0]["max"] > 300 and sig[0]["p99_9"] > 0


def test_track_main_end_to_end(survey, track, tmp_path, monkeypatch, capsys):
    folder = survey.analyse(PATH, "RFP", 150, 0.85, out=tmp_path / "scene_survey")
    fake_ffmpeg(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["aml18_track", str(PATH), "--survey", str(folder), "--fps", "4"])
    track.main()
    out = capsys.readouterr().out
    assert "tracks" in out and "RFP brightest-neuron level" in out
    for name in ("tracks.csv", "track_summary.csv", "tracks.png", "signal.csv"):
        assert (folder / name).exists(), name
    summary = list(csv.DictReader((folder / "track_summary.csv").open(encoding="utf-8")))
    assert summary[0]["frames"] == "2" and summary[0]["stage"] == "L3"


def test_modules_run_as_scripts(survey, track, monkeypatch):
    import runpy
    for mod in ("analysis.aml18_survey", "analysis.aml18_track"):
        monkeypatch.setattr(sys, "argv", ["x", "--help"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_module(mod, run_name="__main__")
        assert exc.value.code == 0
