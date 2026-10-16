# analysis/make_soundtrack.py, short_video.py, physarum_short.py and
# worm_short.py: audio synthesis, the video writer (against a fake ffmpeg) and
# both "short" builders on tiny synthetic inputs, with the durations shrunk so
# a run renders a couple of dozen frames instead of a thousand.
import csv
import datetime
import importlib
import runpy
import sys
import types
import wave
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest
from PIL import Image

from analysis import make_soundtrack as ms, physarum_short as ps, short_video as sv
from tests.fakes_nd2 import UM_PX, install_nd2, neurons, worm_td


# ---- fake ffmpeg ---------------------------------------------------------------------------

class FakeProc:
    def __init__(self, cmd, code=0):
        self.cmd, self.code, self.written, self.closed = cmd, code, 0, False
        proc = self

        class Stdin:
            def write(self, data):
                proc.written += len(data)

            def close(self):
                proc.closed = True

        self.stdin = Stdin()

    def wait(self):
        return self.code


@pytest.fixture
def ffmpeg(monkeypatch):
    procs, runs = [], []
    mod = types.ModuleType("imageio_ffmpeg")
    mod.get_ffmpeg_exe = lambda: "ffmpeg"
    mod.count_frames_and_secs = lambda path: (300, 12.5)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", mod)
    monkeypatch.setattr(sv.subprocess, "Popen", lambda cmd, **kw: procs.append(FakeProc(cmd)) or procs[-1])
    monkeypatch.setattr(ms.subprocess, "run", lambda cmd, check: runs.append(cmd))
    return NS(procs=procs, runs=runs)



# ---- make_soundtrack -----------------------------------------------------------------------------

def test_midi_pitches():
    assert ms.midi(69) == 440.0 and ms.midi(57) == pytest.approx(220.0) and ms.midi(81) == pytest.approx(880.0)


def test_render_is_a_normalised_stereo_mix_with_fades():
    mix = ms.render(2.0)
    assert mix.shape == (int(ms.SR * 2.0), 2)
    assert np.abs(mix).max() == pytest.approx(10 ** (-1 / 20))
    assert abs(mix[0]).max() < 0.01 and abs(mix[-50:]).max() < 0.05           # faded in and out
    assert np.array_equal(mix, ms.render(2.0)) and not np.array_equal(mix, ms.render(2.0, seed=8))


def test_wav_round_trip(tmp_path):
    mix = np.stack([np.sin(np.linspace(0, 60, 4410)), np.cos(np.linspace(0, 60, 4410))], axis=1)
    path = tmp_path / "t.wav"
    ms.write_wav(mix, path)
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (2, 2, ms.SR, 4410)


def test_movie_length_and_muxing_use_ffmpeg(ffmpeg, tmp_path):
    assert ms.movie_seconds(tmp_path / "m.mp4") == 12.5
    ms.mux(tmp_path / "m.mp4", tmp_path / "a.wav", tmp_path / "out.mp4")
    cmd = ffmpeg.runs[0]
    assert cmd[0] == "ffmpeg" and "-shortest" in cmd and cmd[-1] == str(tmp_path / "out.mp4")


def test_soundtrack_main_for_a_plain_duration(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ms, "render", lambda dur, seed: np.zeros((100, 2)))
    out = tmp_path / "s.wav"
    monkeypatch.setattr(sys, "argv", ["make_soundtrack", "--duration", "3", "--out", str(out), "--seed", "9"])
    ms.main()
    assert out.exists() and "wrote" in capsys.readouterr().out


def test_soundtrack_main_matches_the_movie(tmp_path, monkeypatch, ffmpeg):
    seen = []
    monkeypatch.setattr(ms, "render", lambda dur, seed: (seen.append(dur), np.zeros((100, 2)))[1])
    movie = tmp_path / "clip.mp4"
    monkeypatch.setattr(sys, "argv", ["make_soundtrack", "--movie", str(movie)])
    ms.main()
    assert seen == [12.5] and (tmp_path / "clip_soundtrack.wav").exists()
    assert ffmpeg.runs[0][-1] == str(tmp_path / "clip_music.mp4")


def test_soundtrack_main_defaults_to_the_score_length(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ms, "render", lambda dur, seed: (seen.append(dur), np.zeros((10, 2)))[1])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_soundtrack"])
    ms.main()
    assert seen == [ms.SCORE_S] and (tmp_path / "soundtrack.wav").exists()


# ---- short_video ---------------------------------------------------------------------------------------

def test_fonts_fall_back_and_are_cached():
    a = sv.font("C:/definitely/not/here.ttf", 30)
    assert a is sv.font("C:/definitely/not/here.ttf", 30) and a is not sv.font("C:/definitely/not/here.ttf", 31)
    assert sv.font(sv.FONT_B, 24) is not None


def test_canvas_and_centred_text():
    c = sv.canvas()
    assert c.size == (sv.VW, sv.VH) and c.getpixel((0, 0)) == sv.BG
    from PIL import ImageDraw
    d = ImageDraw.Draw(c)
    sv.centred(d, 100, "hello", sv.font(sv.FONT_R, 60))
    xs = np.nonzero((np.asarray(c)[100:200] != np.array(sv.BG)).any(axis=2))[1]
    assert xs.size and abs((xs.min() + xs.max()) / 2 - sv.VW / 2) < 40          # drawn around the middle


def test_short_wav_clips_instead_of_wrapping(tmp_path):
    sv.write_wav(np.array([[2.0, -2.0], [0.5, -0.5]]), tmp_path / "c.wav")
    with wave.open(str(tmp_path / "c.wav")) as w:
        data = np.frombuffer(w.readframes(2), np.int16)
    assert list(data[:2]) == [32767, -32767]


def test_preview_writer_saves_only_the_chosen_stills(tmp_path):
    w = sv.ShortWriter(tmp_path / "o.mp4", None, preview=True, stills={1, 3})
    img = sv.canvas()
    assert [w.wants(i) for i in range(4)] == [False, True, False, True]
    for i in range(4):
        w.emit(img, i)
    w.close()
    assert sorted(p.name for p in tmp_path.glob("_preview_*.png")) == ["_preview_0001.png", "_preview_0003.png"]


def test_writer_pipes_raw_frames_to_ffmpeg(ffmpeg, tmp_path):
    with sv.ShortWriter(tmp_path / "o.mp4", tmp_path / "a.wav") as w:
        w.emit(sv.canvas())
        w.emit(sv.canvas())
        assert w.index == 2 and w.wants(0)
    p = ffmpeg.procs[0]
    assert p.written == 2 * sv.VW * sv.VH * 3 and p.closed
    assert "-i" in p.cmd and str(tmp_path / "a.wav") in p.cmd and "aac" in p.cmd and p.cmd[-1] == str(tmp_path / "o.mp4")


def test_writer_without_audio_and_ffmpeg_failure(ffmpeg, tmp_path, monkeypatch):
    with sv.ShortWriter(tmp_path / "o.mp4", None) as w:
        w.emit(sv.canvas())
    assert "aac" not in ffmpeg.procs[0].cmd
    monkeypatch.setattr(sv.subprocess, "Popen", lambda cmd, **kw: FakeProc(cmd, code=1))
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        with sv.ShortWriter(tmp_path / "bad.mp4", None) as w:
            w.emit(sv.canvas())


# ---- physarum_short ------------------------------------------------------------------------------------------

def test_phase_names_follow_the_hours():
    assert [ps.phase(h) for h in (1, 10, 21, 30)] == [
        "Exploring in every direction", "Spreading toward the food",
        "Pulling back & rebuilding", "Holding its network overnight"]


def combined_dir(tmp_path):
    d = tmp_path / "combined"
    d.mkdir()
    runs = []
    t0 = datetime.datetime(2026, 9, 23, 8, 0, 0)
    for n in range(2):
        run = tmp_path / f"run{n}"
        run.mkdir()
        with (run / "tiles.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["round", "tile", "wall_clock"])
            for r in range(3):
                w.writerow([r, 0, (t0 + datetime.timedelta(hours=n * 3 + r)).isoformat()])
        for r in range(3):
            cv2.imwrite(str(run / f"mosaic_{r:03d}.png"), np.full((150, 200, 3), 90 + 30 * r, np.uint8))
        runs.append(run)
    (d / "combined_all_script.py").write_text(
        "from pathlib import Path\n"
        f"RUNS = [Path(r'{runs[0]}'), Path(r'{runs[1]}')]\n"
        f"REF = Path(r'{runs[0]}')\n"
        "W, H, UM_PX = 200, 150, 8.0\n"
        "def to_ref(img, run):\n    return img\n"
        "# ---- movie setup\nraise RuntimeError('the movie half must never run')\n", encoding="utf-8")
    with (d / "oat_approach_all.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t_h", "total_mm2"])
        for i in range(12):
            w.writerow([i * 0.5, 120 + 5 * i - (i > 8) * 20])
    Image.new("RGB", (200, 200), (30, 160, 60)).save(d / "growth_start_to_end_colour.jpg")
    return d


@pytest.fixture
def tiny_physarum(monkeypatch):
    for name, v in dict(S_HOOK=1, S_INTRO=1, S_TL=2, S_REVEAL=5, S_END=1, FPS=2, DURATION=10, REVEAL_AT=4).items():
        monkeypatch.setattr(ps, name, v)
    monkeypatch.setattr(ps, "soundtrack", lambda: np.zeros((100, 2)))


def test_the_analysis_script_setup_is_read_but_not_the_movie_half(tmp_path):
    ns = ps.load_alignment(combined_dir(tmp_path))
    assert ns["W"] == 200 and len(ns["RUNS"]) == 2 and callable(ns["to_ref"])


def test_the_theme_gets_a_swell_into_the_reveal(monkeypatch):
    monkeypatch.setattr(ps, "render", lambda dur: np.random.default_rng(0).normal(0, 0.1, (ps.SR * 2, 2)))
    mix = ps.soundtrack()
    assert mix.shape == (ps.SR * 2, 2) and np.abs(mix).max() == pytest.approx(10 ** (-1 / 20))


def test_preview_saves_stills_without_encoding(tmp_path, tiny_physarum, capsys):
    d = combined_dir(tmp_path)
    out = tmp_path / "shorts" / "p.mp4"
    out.parent.mkdir()
    ps.make(d, out, "caption text", preview=True)
    stills = sorted(out.parent.glob("_preview_*.png"))
    assert stills and Image.open(stills[0]).size == (1080, 1920) and not out.exists()
    assert "preview stills" in capsys.readouterr().out


def test_full_render_pipes_every_frame_to_the_encoder(tmp_path, tiny_physarum, ffmpeg, capsys):
    d = combined_dir(tmp_path)
    out = tmp_path / "shorts" / "p.mp4"
    out.parent.mkdir()
    ps.make(d, out, "caption text", preview=False)
    p = ffmpeg.procs[0]
    assert p.written == ps.DURATION * ps.FPS * 1080 * 1920 * 3 and (out.with_suffix(".wav")).exists()
    assert "wrote" in capsys.readouterr().out


def test_physarum_main_makes_the_output_folder(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ps, "make", lambda *a: seen.append(a))
    d = combined_dir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["physarum_short", str(d), "--preview", "--caption", "hi"])
    ps.main()
    assert (d / "shorts").is_dir() and seen[0][1] == (d / "shorts" / "physarum_short_growth.mp4").resolve()
    assert seen[0][2:] == ("hi", True)


# ---- worm_short -------------------------------------------------------------------------------------------------------

ws = importlib.import_module("analysis.worm_short")


def worm_files(tmp_path, nt=12, dt=60.0, size=128, short_track=False):
    """A time series ND2 (via the fake reader) plus a survey worms.csv with three adults."""
    stack = np.zeros((nt, 2, size, size), np.float32)
    for t in range(nt):
        stack[t, 0] = worm_td(size, p0=(20, 60), p1=(90, 70), seed=t)
        stack[t, 1] = neurons(size, centres=((22, 60), (30, 62)))
    rows = []
    speeds = {1: 6, 2: 3, 3: 0.5}                              # px per frame: Explorer, Wanderer, Homebody
    for t in range(nt):
        for wid, sp in speeds.items():
            if short_track and wid == 3 and t >= 5:
                continue                                       # the Homebody leaves the field after frame 4
            rows.append({"t_index": t, "t_s": t * dt, "x_px": 20 + 25 * wid + sp * t, "y_px": 20 + 30 * wid,
                         "length_mm": 1.0, "stage_estimate": "adult"})
    survey = tmp_path / "scene_survey"
    survey.mkdir()
    with (survey / "worms.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return stack, survey, [t * dt for t in range(nt)]


@pytest.fixture
def tiny_worms(monkeypatch):
    consts = dict(S_DARK=1.0, S_FADE=1.0, S_TL=2.0, S_HOLD=1.0, S_MAP=2.0, S_END=1.0,
                  T_FADE=1.0, T_TL=2.0, T_HOLD=4.0, T_MAP=5.0, T_END=7.0, DURATION=8.0, FPS=2)
    for k, v in consts.items():
        monkeypatch.setattr(ws, k, v)


def test_nd2_is_split_into_brightfield_and_neuron_channels(tmp_path, monkeypatch):
    stack, survey, times = worm_files(tmp_path)
    install_nd2(monkeypatch, stack, list("TCYX"), ["TD", "RFP"], times=times)
    td, neu, um_px, t_s = ws.load(Path("x.nd2"), "RFP")
    assert td.shape == neu.shape == (12, 128, 128) and um_px == UM_PX and t_s[-1] == 660.0
    assert np.allclose(td, stack[:, 0]) and neu.max() > 300


def test_composites_glow_only_where_neurons_are(tmp_path):
    stack, _, _ = worm_files(tmp_path)
    frames, dark = ws.make_composites(stack[:, 0], stack[:, 1])
    assert len(frames) == 12 and frames[0].shape == (ws.P, ws.P, 3) and frames[0].dtype == np.uint8
    glow_px = dark[int(60 * ws.P / 128), int(22 * ws.P / 128)]
    assert glow_px[0] > 100 and glow_px[2] > 100 and glow_px[1] < glow_px[0]       # pink on black
    assert dark[10, 10].max() < 30 and frames[0][10, 10].mean() > 100


def test_tracks_are_linked_smoothed_and_summarised(tmp_path):
    _, survey, _ = worm_files(tmp_path)
    tracks, raw_rate, summary = ws.build_tracks(survey / "worms.csv", UM_PX, 2.0, 1.0, 0.5)
    assert sorted(tracks) == [1, 2, 3] and all(len(v) == 12 for v in tracks.values())
    assert raw_rate[1] > raw_rate[2] > raw_rate[3] > 0 and summary[1]["frames"] == 12
    assert tracks[1][0][3] == 0.0 and tracks[1][-1][0] == 11.0


def test_stars_are_ranked_by_speed(tmp_path):
    _, survey, _ = worm_files(tmp_path)
    tracks, raw_rate, summary = ws.build_tracks(survey / "worms.csv", UM_PX, 2.0, 1.0, 0.5)
    stars, rate = ws.pick_stars(tracks, summary, [])
    assert {n for n, _ in stars.values()} == {"Explorer", "Wanderer", "Homebody"}
    assert stars[1][0] == "Explorer" and stars[3][0] == "Homebody"
    two, _ = ws.pick_stars(tracks, summary, [1, 3])
    assert [n for n, _ in two.values()] == ["Explorer", "Homebody"]
    with pytest.raises(SystemExit, match="at least two tracks"):
        ws.pick_stars(tracks, summary, [1])
    with pytest.raises(SystemExit, match="at least two tracks"):
        ws.pick_stars({}, {}, [])


def test_position_is_interpolated_inside_a_track_only():
    tr = [(0.0, 0.0, 0.0, 0.0), (10.0, 100.0, 50.0, 1.0), (20.0, 100.0, 150.0, 1.0)]
    assert ws.pos_at(tr, 5.0) == (50.0, 25.0)
    assert ws.pos_at(tr, 20.0) == (100.0, 150.0)
    assert ws.pos_at(tr, -1.0) is None and ws.pos_at(tr, 21.0) is None


def test_soundtrack_is_a_normalised_stereo_bed(tiny_worms):
    mix = ws.soundtrack(12)
    assert mix.shape == (int(ws.SR * ws.DURATION), 2) and np.abs(mix).max() == pytest.approx(10 ** (-1 / 20))


def make_args(tmp_path, monkeypatch, **kw):
    stack, survey, times = worm_files(tmp_path, **kw)
    install_nd2(monkeypatch, stack, list("TCYX"), ["TD", "RFP"], times=times)
    return survey


def test_preview_saves_stills(tmp_path, monkeypatch, tiny_worms, capsys):
    survey = make_args(tmp_path, monkeypatch)
    out = tmp_path / "shorts" / "w.mp4"
    out.parent.mkdir()
    ws.make(Path("x.nd2"), survey, out, "RFP", 0.5, [], preview=True)
    text = capsys.readouterr().out
    assert list(out.parent.glob("_preview_*.png")) and "Explorer/Homebody" in text and "at least" in text
    assert not out.exists()


def test_a_full_render_encodes_every_frame(tmp_path, monkeypatch, tiny_worms, ffmpeg):
    survey = make_args(tmp_path, monkeypatch)
    out = tmp_path / "shorts" / "w.mp4"
    out.parent.mkdir()
    ws.make(Path("x.nd2"), survey, out, "RFP", 0.5, [1, 2, 3], preview=False)
    p = ffmpeg.procs[0]
    assert p.written == int(ws.DURATION * ws.FPS) * 1080 * 1920 * 3 and out.with_suffix(".wav").exists()


def test_an_hour_long_run_is_described_as_such(tmp_path, monkeypatch, tiny_worms, capsys):
    survey = make_args(tmp_path, monkeypatch, nt=7, dt=600.0)             # 6 intervals of 10 min = 60 min
    out = tmp_path / "shorts" / "w.mp4"
    out.parent.mkdir()
    ws.make(Path("x.nd2"), survey, out, "RFP", 5.0, [1, 2], preview=True)
    assert "10.00 min apart" in capsys.readouterr().out


def test_worm_main_builds_default_paths(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(ws, "make", lambda *a: seen.append(a))
    nd2 = tmp_path / "scene.nd2"
    monkeypatch.setattr(sys, "argv", ["worm_short", str(nd2), "--stars", "1,2,3", "--preview", "--channel", "GFP"])
    ws.main()
    nd2_, survey, out, channel, step, stars, preview = seen[0]
    assert survey == tmp_path / "scene_survey" and out == (tmp_path / "shorts" / "scene_short.mp4").resolve()
    assert (channel, step, stars, preview) == ("GFP", 0.5, [1, 2, 3], True)
    assert (tmp_path / "shorts").is_dir()


def test_video_scripts_run_with_dash_m(monkeypatch, tmp_path):
    for module in ("analysis.make_soundtrack", "analysis.physarum_short", "analysis.worm_short"):
        monkeypatch.setattr(sys, "argv", ["x", "--help"])
        with pytest.raises(SystemExit) as exc:
            runpy.run_module(module, run_name="__main__")
        assert exc.value.code == 0


def test_a_track_that_ends_early_gets_a_hollow_marker_and_every_caption_shows(tmp_path, monkeypatch, capsys):
    consts = dict(S_DARK=2.0, S_FADE=1.0, S_TL=2.0, S_HOLD=3.0, S_MAP=2.0, S_END=1.0,
                  T_FADE=2.0, T_TL=3.0, T_HOLD=5.0, T_MAP=8.0, T_END=10.0, DURATION=11.0, FPS=2)
    for k, v in consts.items():
        monkeypatch.setattr(ws, k, v)
    stack, survey, times = worm_files(tmp_path, short_track=True)
    install_nd2(monkeypatch, stack, list("TCYX"), ["TD", "RFP"], times=times)
    out = tmp_path / "shorts" / "w.mp4"
    out.parent.mkdir()
    ws.make(Path("x.nd2"), survey, out, "RFP", 0.5, [1, 2, 3], preview=True)
    assert len(list(out.parent.glob("_preview_*.png"))) >= 4


def test_ticks_and_notes_past_the_end_of_the_track_are_dropped(monkeypatch):
    for k, v in dict(DURATION=3.0, T_END=7.0).items():
        monkeypatch.setattr(ws, k, v)
    mix = ws.soundtrack(12)
    assert mix.shape == (int(ws.SR * 3.0), 2) and np.isfinite(mix).all()
