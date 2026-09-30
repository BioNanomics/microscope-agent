import json

from timelapse.frame_audit import audit, frames_from_dir, apply_timestamps, main, parse_duration


def _sequence(write_frame, tmp_path, n=12, event_at=8, gap_at=None, shift_at=None, dark_at=None):
    """n frames 600s apart. Optional: blob shrinks at event_at, a 3x time
    gap before gap_at, an XY shift at shift_at, a global dimming at dark_at."""
    frames, ts = [], []
    t = 1_700_000_000.0
    for i in range(n):
        kw = {"seed": i}
        if event_at is not None and i >= event_at:
            kw["radius"] = 10.0
        if shift_at is not None and i >= shift_at:
            kw["center"] = (78, 64)
        if dark_at is not None and i >= dark_at:
            kw["background"] = 120.0
            kw["foreground"] = 30.0
        frames.append(write_frame(f"frame_{i:03d}.png", **kw))
        if gap_at is not None and i == gap_at:
            t += 1200.0
        ts.append(t)
        t += 600.0
    ts_file = tmp_path / "timestamps.csv"
    ts_file.write_text("\n".join(str(x) for x in ts) + "\n")
    return frames, ts_file


def test_shrink_event_flagged_as_change_not_gap(write_frame, tmp_path):
    _sequence(write_frame, tmp_path, event_at=8)
    frames = apply_timestamps(frames_from_dir(tmp_path, "*.png"), tmp_path / "timestamps.csv")
    results = audit(frames)
    by_index = {r.index: r for r in results}
    assert "CHANGE" in by_index[8].flags
    assert not any(f.startswith("GAP") for f in by_index[8].flags)
    assert all(not r.flags for r in results if r.index not in (8,)), [r.flags for r in results]


def test_time_gap_flagged(write_frame, tmp_path):
    _sequence(write_frame, tmp_path, event_at=None, gap_at=5)
    frames = apply_timestamps(frames_from_dir(tmp_path, "*.png"), tmp_path / "timestamps.csv")
    results = audit(frames)
    r = next(r for r in results if r.index == 5)  # the gap is *before* frame 5
    assert r.dt_ratio > 2.5
    assert any(f.startswith("GAP") for f in r.flags)


def test_shift_and_dimming_flagged_as_non_biological(write_frame, tmp_path):
    _sequence(write_frame, tmp_path, event_at=None, shift_at=6, dark_at=9)
    frames = apply_timestamps(frames_from_dir(tmp_path, "*.png"), tmp_path / "timestamps.csv")
    results = audit(frames)
    by_index = {r.index: r for r in results}
    assert any(f.startswith("SHIFT") for f in by_index[6].flags)
    assert any(f.startswith("INTENSITY") for f in by_index[9].flags)
    assert "CHANGE" not in by_index[6].flags
    assert "CHANGE" not in by_index[9].flags


def test_cli_around_window_and_json(write_frame, tmp_path, capsys):
    _sequence(write_frame, tmp_path, event_at=8)
    out_json = tmp_path / "audit.json"
    rc = main([str(tmp_path), "--timestamps", str(tmp_path / "timestamps.csv"),
               "--around", "80m", "--window", "10m", "--json", str(out_json)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "12 frames" in text
    assert "CHANGE" in text
    data = json.loads(out_json.read_text())
    assert len(data) == 11


def test_parse_duration():
    assert parse_duration("25h") == 90000.0
    assert parse_duration("90m") == 5400.0
    assert parse_duration("30") == 30.0
