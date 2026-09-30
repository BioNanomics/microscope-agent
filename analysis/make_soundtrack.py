# make_soundtrack.py
# ------------------------------------------------------------
# Original ambient soundtrack for a time-lapse movie, synthesised with
# numpy - no samples and no third-party audio, so there are no licensing
# questions when a movie is posted or presented.
#
#   python -m analysis.make_soundtrack --duration 39 --out track.wav
#   python -m analysis.make_soundtrack --movie run.mp4          # -> run_music.mp4
#
# With --movie the track is made exactly as long as the movie and muxed
# onto it (video stream copied untouched, audio AAC), using the ffmpeg
# that imageio-ffmpeg bundles.
#
# THE ARRANGEMENT is the one written for the 2026-09-25 Physarum Short:
# a quiet opening, a chord progression that builds with a rising
# arpeggio, a sparse held middle, a lighter second rise and a warm major
# resolve at the end. It was composed on a 39 s timeline; other lengths
# stretch that timeline, so sections keep their proportions. Notes keep
# their own length, so very long movies (minutes) get sparser, not
# slower-sounding - it suits roughly 20-120 s.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 44100
SCORE_S = 39.0            # the timeline the arrangement was written on


def midi(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def render(duration: float, seed: int = 7) -> np.ndarray:
    """Stereo float mix in [-1, 1], `duration` seconds long."""
    n = int(SR * duration)
    t = np.arange(n) / SR
    k = duration / SCORE_S                   # score seconds -> real seconds
    rng = np.random.default_rng(seed)

    def env_between(a, b, fade):
        e = np.minimum(np.clip((t - a * k) / fade, 0, 1), np.clip((b * k - t) / fade, 0, 1))
        return 0.5 - 0.5 * np.cos(np.pi * e)

    def pad(notes, a, b, level, bright=0.3):
        """Soft pad: each note = detuned sines + weak harmonics, slow swell."""
        out = np.zeros((n, 2))
        e = env_between(a, b, fade=2.0)
        for note in notes:
            f = midi(note)
            for det, pan in ((-0.12, 0.3), (0.0, 0.5), (0.12, 0.7)):
                ff = f * 2 ** (det / 12)
                ph = rng.uniform(0, 2 * np.pi)
                s = (np.sin(2 * np.pi * ff * t + ph)
                     + bright * 0.35 * np.sin(2 * np.pi * 2 * ff * t + ph)
                     + bright * 0.12 * np.sin(2 * np.pi * 3 * ff * t + ph))
                out[:, 0] += s * (1 - pan)
                out[:, 1] += s * pan
        return out * (e * level / (len(notes) * 3))[:, None]

    def pluck(note, at, level, pan=0.5, decay=1.2):
        out = np.zeros((n, 2))
        i0 = int(at * k * SR)
        if i0 >= n:
            return out
        L = min(n - i0, int(SR * decay * 4))
        tt = np.arange(L) / SR
        f = midi(note)
        s = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * 2 * f * tt)
             + 0.1 * np.sin(2 * np.pi * 3 * f * tt))
        s *= np.exp(-tt / decay) * np.clip(tt / 0.005, 0, 1) * level
        out[i0:i0 + L, 0] += s * (1 - pan)
        out[i0:i0 + L, 1] += s * pan
        return out

    mix = np.zeros((n, 2))
    # A minor world resolving to C major: Am, F, C, G
    AM, F, C, G = [57, 64, 69, 72], [53, 60, 65, 69], [48, 55, 64, 67], [55, 62, 67, 71]
    CMAJ = [48, 55, 60, 64, 67, 72]

    mix += pad([45, 57, 64], 0.0, 7.0, 0.55, bright=0.15)              # opening
    for s0, ch in ((6.0, AM), (8.75, F), (11.5, C), (14.25, G)):         # progression
        mix += pad(ch, s0 - 0.5, s0 + 3.4, 0.7 if s0 < 11 else 0.85,
                   bright=0.3 if s0 < 11 else 0.55)
    for i, at in enumerate(np.arange(6.0, 11.0, 0.6875)):               # root pulse
        mix += pluck(45 if i % 2 == 0 else 52, at, 0.10, pan=0.4, decay=0.6)
    arp = [69, 72, 76, 79, 81, 84, 81, 79]
    for i, at in enumerate(np.arange(11.0, 17.2, 0.34375)):             # rising arpeggio
        note = arp[i % len(arp)] + (12 if at > 14.5 else 0)
        mix += pluck(note - 12, at, 0.10 + 0.05 * (at - 11) / 6, pan=0.3 + 0.4 * (i % 2), decay=0.9)
    mix += pad([45, 52, 60, 64], 16.5, 27.5, 0.5, bright=0.12)          # held middle
    for at, note in ((18.5, 76), (21.0, 72), (23.5, 71), (25.5, 69)):
        mix += pluck(note, at, 0.07, pan=0.6, decay=2.0)
    mix += pad(F, 26.8, 31.0, 0.6, bright=0.35)                          # second rise
    mix += pad(G, 30.5, 34.5, 0.6, bright=0.35)
    for i, at in enumerate(np.arange(27.2, 34.0, 0.4125)):
        mix += pluck([72, 76, 79, 76][i % 4], at, 0.08, pan=0.35 + 0.3 * (i % 2), decay=0.8)
    mix += pad(CMAJ, 33.8, 40.5, 0.9, bright=0.4)                        # resolve
    for dt, note, lv in ((0.0, 60, 0.12), (0.15, 64, 0.10), (0.3, 67, 0.10)):
        mix += pluck(note, 34.0 + dt, lv, decay=3.0)

    # Stereo reverb: convolve with decaying noise (FFT).
    ir_len = int(SR * 2.2)
    ir = rng.standard_normal((ir_len, 2)) * np.exp(-np.arange(ir_len) / (SR * 0.55))[:, None]
    ir[0] = 0
    nfft = 1 << int(np.ceil(np.log2(n + ir_len)))
    wet = np.stack([np.fft.irfft(np.fft.rfft(mix[:, c], nfft) * np.fft.rfft(ir[:, c], nfft), nfft)[:n]
                    for c in range(2)], axis=1)
    wet *= np.abs(mix).max() / (np.abs(wet).max() + 1e-9)
    out = 0.72 * mix + 0.38 * wet

    # Master: short fade-in, 2.5 s fade-out, normalise to -1 dBFS.
    out *= (np.clip(t / 0.3, 0, 1) * np.clip((duration - t) / 2.5, 0, 1))[:, None]
    return out * 10 ** (-1 / 20) / np.abs(out).max()


def write_wav(mix: np.ndarray, path: Path) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((mix * 32767).astype(np.int16).tobytes())


def movie_seconds(movie: Path) -> float:
    import imageio_ffmpeg
    frames, secs = imageio_ffmpeg.count_frames_and_secs(str(movie))
    return float(secs)


def mux(movie: Path, wav: Path, out: Path) -> None:
    import imageio_ffmpeg
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-i", str(movie), "-i", str(wav), "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, help="seconds (default: the movie's length, else 39)")
    ap.add_argument("--movie", type=Path, help="add the track to this movie -> <name>_music.mp4")
    ap.add_argument("--out", type=Path, help="output .wav (default beside the movie, or soundtrack.wav)")
    ap.add_argument("--seed", type=int, default=7, help="changes pad phases and reverb, not the notes")
    a = ap.parse_args()

    dur = a.duration or (movie_seconds(a.movie) if a.movie else SCORE_S)
    wav = a.out or (a.movie.with_name(a.movie.stem + "_soundtrack.wav") if a.movie
                    else Path("soundtrack.wav"))
    mix = render(dur, a.seed)
    write_wav(mix, wav)
    print(f"wrote {wav} ({dur:.1f} s)")
    if a.movie:
        dest = a.movie.with_name(a.movie.stem + "_music.mp4")
        mux(a.movie, wav, dest)
        print(f"wrote {dest}")


if __name__ == "__main__":
    main()
