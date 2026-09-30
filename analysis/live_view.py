# live_view.py
# ------------------------------------------------------------
# Watch an acquisition as it happens, without touching the camera.
#
#   python -m analysis.live_view                 # newest run, auto-detected
#   python -m analysis.live_view data/mosaic_20260921_194543_region
#   python -m analysis.live_view --fps 2 --width 1100
#
# Then open the printed file:// link in a browser and leave it there.
#
# WHY NOT AN ACTUAL LIVE FEED: a GenICam camera can be held by exactly one
# process. Anything that opened the camera to stream preview frames would
# take it away from the run that is acquiring - the failure this repo hits
# repeatedly (AccessDeniedException) whenever two things want the device.
#
# So this never opens the camera. It watches the FILES the running
# acquisition is already writing and republishes the newest one as a fixed
# filename that a browser can poll. The acquisition does not know this
# exists and cannot be affected by it: worst case, the viewer shows a
# slightly stale frame.
#
# WHY A COPY RATHER THAN POINTING THE PAGE AT THE TILE ITSELF: the newest
# file changes name constantly, and a browser cannot discover that from a
# file:// page. Republishing to one stable name (latest.jpg) makes the
# page a two-line poll. JPEG rather than PNG because it is a preview -
# it re-encodes in milliseconds where a 3 MB PNG does not.
#
# PARTIAL WRITES: a file picked up mid-write decodes as a truncated or
# corrupt image. Each candidate is therefore only published once its size
# has stopped changing between polls, and any decode error is swallowed
# and retried rather than killing the viewer.
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import time
from pathlib import Path

from PIL import Image

from acquisition.paths import data_root

_PAGE = """<!doctype html>
<meta charset="utf-8"><title>microscope live view</title>
<style>
 :root { color-scheme: dark; }
 body { margin:0; background:#111; color:#ddd; font:13px system-ui, sans-serif;
        display:flex; flex-direction:column; height:100vh; }
 header { padding:8px 12px; background:#1b1b1b; border-bottom:1px solid #333;
          display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
 #wrap { flex:1; display:flex; align-items:center; justify-content:center; overflow:hidden; }
 img { max-width:100%; max-height:100%; object-fit:contain; }
 .k { color:#888; }
 b { color:#fff; font-weight:600; }
</style>
<header>
  <span class="k">watching</span> <b>__DIR__</b>
  <span class="k">frame</span> <b id="n">-</b>
  <span class="k">updated</span> <b id="t">-</b>
  <span class="k">poll</span> <b>__MS__ ms</b>
</header>
<div id="wrap"><img id="v" alt="waiting for the first frame..."></div>
<script>
let n = 0;
function tick() {
  const img = document.getElementById('v');
  const next = new Image();
  next.onload = () => {
    img.src = next.src;
    document.getElementById('n').textContent = ++n;
    document.getElementById('t').textContent = new Date().toLocaleTimeString();
  };
  next.src = 'latest.jpg?c=' + Date.now();
}
setInterval(tick, __MS__); tick();
</script>
"""


def newest_image(root: Path) -> Path | None:
    best, best_m = None, -1.0
    for p in root.rglob("*.png"):
        if p.name == "latest.jpg":
            continue
        try:
            m = p.stat().st_mtime
        except OSError:
            continue
        if m > best_m:
            best, best_m = p, m
    return best


def newest_run(base: Path) -> Path | None:
    runs = [p for p in base.glob("*") if p.is_dir()
            and (p.name.startswith("mosaic_") or p.name.startswith("timelapse_"))]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def serve(watch: Path, out: Path, fps: float, width: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    period = max(0.2, 1.0 / max(fps, 0.1))
    page = out / "index.html"
    page.write_text(
        _PAGE.replace("__DIR__", watch.name).replace("__MS__", str(int(period * 1000))),
        encoding="utf-8")
    print(f"watching : {watch}")
    print(f"OPEN THIS: {page.resolve().as_uri()}")
    print("(Ctrl-C to stop - this never touches the camera)")

    published, last_size = None, -1
    while True:
        try:
            src = newest_image(watch)
            if src is not None and src != published:
                size = src.stat().st_size
                if size == last_size and size > 0:
                    try:
                        im = Image.open(src)
                        im.thumbnail((width, width))
                        im.convert("RGB").save(out / "latest.jpg", quality=85)
                        published = src
                        print(f"  {time.strftime('%H:%M:%S')}  {src.name}", flush=True)
                    except Exception:
                        last_size = -1          # truncated - wait and retry
                else:
                    last_size = size
        except Exception:
            pass
        time.sleep(period)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("watch", nargs="?", type=Path,
                    help="run folder to watch (default: newest under data/)")
    ap.add_argument("--out", type=Path, help="where to write the viewer (default: <watch>/_live)")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=1200)
    args = ap.parse_args()

    watch = args.watch or newest_run(data_root() / "data")
    if watch is None or not watch.exists():
        raise SystemExit("no run folder found to watch - pass one explicitly")
    serve(watch, args.out or (watch / "_live"), args.fps, args.width)


if __name__ == "__main__":
    main()
