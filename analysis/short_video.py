# short_video.py
# ------------------------------------------------------------
# Shared pieces for the vertical (1080x1920) YouTube Shorts made by
# physarum_short.py and worm_short.py: canvas, fonts, centred text, and a
# writer that pipes RGB frames straight into ffmpeg (x264) with a WAV
# soundtrack muxed on as AAC in the same pass.
#
# WHY PIPE INTO FFMPEG rather than cv2.VideoWriter: cv2's Windows build has
# no reliable H.264 (see make_movie.py), and the MSMF H.264 it sometimes
# offers writes ~40 Mbit/s - the first Physarum Short was 200 MB. libx264
# at CRF 19 looks the same at a few MB, and taking the audio in the same
# ffmpeg run avoids a second mux step. The ffmpeg used is the one
# imageio-ffmpeg bundles (in the `analysis` dependency group).
#
# PREVIEW: with preview=True the writer saves the frames whose index is in
# `stills` as PNGs instead of encoding anything, so a layout can be checked
# in seconds before a render that may take most of an hour.
# ------------------------------------------------------------

from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VW, VH, FPS = 1080, 1920, 30
SR = 44100
BG = (12, 12, 14)
GREY = (170, 170, 170)

# Segoe UI ships with Windows; elsewhere fall back to PIL's scalable default.
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"
FONT_R = "C:/Windows/Fonts/segoeui.ttf"
FONT_I = "C:/Windows/Fonts/segoeuii.ttf"
_fonts: dict = {}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, size)
    if key not in _fonts:
        try:
            _fonts[key] = ImageFont.truetype(path, size)
        except OSError:
            _fonts[key] = ImageFont.load_default(size)
    return _fonts[key]


def canvas(bg=BG) -> Image.Image:
    return Image.new("RGB", (VW, VH), bg)


def centred(draw: ImageDraw.ImageDraw, y: float, text: str, f, fill=(255, 255, 255)) -> None:
    draw.text(((VW - draw.textlength(text, font=f)) / 2, y), text, font=f, fill=fill)


def write_wav(mix: np.ndarray, path: Path) -> None:
    """Stereo float mix in [-1, 1] -> 16-bit WAV."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())


class ShortWriter:
    """Frames in, finished MP4 (H.264 + AAC) out.

        with ShortWriter(out, wav) as w:
            w.emit(img)        # PIL RGB image, VW x VH
    """

    def __init__(self, out: Path, wav: Path | None, preview: bool = False,
                 stills: set[int] = frozenset(), still_dir: Path | None = None):
        self.out, self.preview, self.stills = Path(out), preview, set(stills)
        self.still_dir = Path(still_dir or self.out.parent)
        self.index = 0
        self._ff = None
        if preview:
            return
        import imageio_ffmpeg
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{VW}x{VH}", "-r", str(FPS), "-i", "-"]
        if wav:
            cmd += ["-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(self.out)]
        self._ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def wants(self, index: int) -> bool:
        """False for frames a preview would skip, so callers can avoid drawing them."""
        return not self.preview or index in self.stills

    def emit(self, img: Image.Image, index: int | None = None) -> None:
        if index is not None:
            self.index = index
        if self.preview:
            if self.index in self.stills:
                img.save(self.still_dir / f"_preview_{self.index:04d}.png")
        else:
            self._ff.stdin.write(np.asarray(img.convert("RGB")).tobytes())
        self.index += 1

    def close(self) -> None:
        if self._ff:
            self._ff.stdin.close()
            if self._ff.wait() != 0:
                raise RuntimeError(f"ffmpeg failed writing {self.out}")
            self._ff = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
