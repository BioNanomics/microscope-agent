import numpy as np

from timelapse.change_detector import ChangeDetector, otsu_threshold, phase_correlation_shift


def _feed_quiet(det, blob_frame, n=6):
    for i in range(n):
        det.score_array(blob_frame(seed=i).astype(np.float32))


def test_first_frame_is_never_interesting(blob_frame):
    det = ChangeDetector()
    sc = det.score_array(blob_frame().astype(np.float32))
    assert sc.interesting is False
    assert sc.baseline_frames == 0


def test_quiet_sequence_stays_quiet(blob_frame):
    det = ChangeDetector()
    _feed_quiet(det, blob_frame)
    sc = det.score_array(blob_frame(seed=99).astype(np.float32))
    assert sc.interesting is False, sc
    assert sc.diff_score < 2.0


def test_shrinking_blob_triggers_area_delta(blob_frame):
    det = ChangeDetector()
    _feed_quiet(det, blob_frame)
    sc = det.score_array(blob_frame(radius=10.0, seed=7).astype(np.float32))
    assert sc.interesting is True
    assert sc.area_delta < -0.04
    assert "shrank" in sc.reason


def test_stage_shift_is_reported_as_shift(blob_frame):
    det = ChangeDetector()
    _feed_quiet(det, blob_frame)
    sc = det.score_array(blob_frame(center=(76, 64), seed=7).astype(np.float32))
    assert sc.interesting is True
    assert sc.shift_px > 8
    assert "shift" in sc.reason


def test_frame_size_change_resets_baseline(blob_frame):
    det = ChangeDetector()
    _feed_quiet(det, blob_frame)
    sc = det.score_array(blob_frame(size=64).astype(np.float32))
    assert "reset" in sc.reason
    assert sc.baseline_frames == 0


def test_otsu_splits_bimodal():
    arr = np.concatenate([np.full(500, 50.0), np.full(500, 200.0)])
    thr = otsu_threshold(arr)
    assert 50 < thr < 200


def test_phase_correlation_recovers_integer_shift(blob_frame):
    a = blob_frame(seed=1).astype(np.float32)
    b = np.roll(a, shift=(3, -5), axis=(0, 1))
    dx, dy = phase_correlation_shift(a, b)
    assert (dx, dy) == (-5.0, 3.0)
