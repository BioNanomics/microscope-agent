# nis_bridge.py
# ------------------------------------------------------------
# Client for the NIS bridge (acquisition/nis_bridge/bridge_job.py), the
# JOBS task that exposes NIS-Elements macro calls on 127.0.0.1:8766 while
# it runs. Used by the NIS MCP server; also a command line for testing by
# hand:
#
#   python -m acquisition.backends.nis_bridge status
#   python -m acquisition.backends.nis_bridge move 100 0
#   python -m acquisition.backends.nis_bridge objective "Plan Fluor 10x Ph1 DLL"
#   python -m acquisition.backends.nis_bridge capture D:\...\check.nd2
#   python -m acquisition.backends.nis_bridge nd-run "C. elegans 3h"
#   python -m acquisition.backends.nis_bridge nd-finish
#   python -m acquisition.backends.nis_bridge shutdown
#
# The e-stop is checked HERE as well as in the bridge: a refused request
# never leaves this process, so a stopped microscope does not depend on
# the NIS side having been updated. Stopping an ND run (nd-finish) and
# shutting the bridge down are always allowed - they only end things.
#
# CONFOCAL_NIS_BRIDGE_URL overrides the address (default
# http://127.0.0.1:8766).
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

from acquisition import estop

DEFAULT_URL = "http://127.0.0.1:8766"


class BridgeUnavailable(RuntimeError):
    """The bridge job is not running (or not reachable)."""


class NISBridge:
    def __init__(self, url: str | None = None, timeout_s: float = 30.0):
        self.url = (url or os.environ.get("CONFOCAL_NIS_BRIDGE_URL") or DEFAULT_URL).rstrip("/")
        self.timeout_s = timeout_s

    def _call(self, method: str, path: str, body: dict | None = None,
              timeout_s: float | None = None) -> dict:
        data = json.dumps(body or {}).encode() if method == "POST" else None
        req = urllib.request.Request(self.url + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s or self.timeout_s) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                return {"http_status": e.code, **json.loads(e.read())}
            except ValueError:
                return {"http_status": e.code, "error": str(e)}
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            raise BridgeUnavailable(
                f"NIS bridge not reachable at {self.url} ({e}). Is the bridge job "
                "running in NIS-Elements? See docs/nis-bridge.md.") from e

    # Read-only
    def status(self) -> dict:
        return self._call("GET", "/status")

    # Motion / acquisition: e-stop checked before anything is sent
    def move_relative(self, dx_um: float, dy_um: float) -> dict:
        estop.check()
        return self._call("POST", "/move", {"dx": dx_um, "dy": dy_um})

    def change_objective(self, name: str) -> dict:
        estop.check()
        return self._call("POST", "/objective", {"name": name})

    def capture(self, path: str) -> dict:
        estop.check()
        return self._call("POST", "/capture", {"path": path}, timeout_s=300)

    def nd_run(self, experiment: str, timeout_s: float = 7200.0) -> dict:
        # ND_RunExperiment may block until the run ends (unverified), so
        # the answer can take as long as the run itself.
        estop.check()
        return self._call("POST", "/nd/run", {"experiment": experiment}, timeout_s=timeout_s)

    # Always allowed: these only end things
    def nd_finish(self) -> dict:
        return self._call("POST", "/nd/finish")

    def shutdown(self) -> dict:
        return self._call("POST", "/shutdown")


def main() -> None:
    ap = argparse.ArgumentParser(description="Talk to the NIS bridge job by hand.")
    ap.add_argument("--url", help=f"bridge address (default {DEFAULT_URL})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    m = sub.add_parser("move", help="relative XY move in um (max 1000 per axis)")
    m.add_argument("dx", type=float); m.add_argument("dy", type=float)
    o = sub.add_parser("objective", help="exact name as listed by status (4x/10x only)")
    o.add_argument("name")
    c = sub.add_parser("capture", help="capture one image and save it as .nd2")
    c.add_argument("path")
    r = sub.add_parser("nd-run", help="load a saved ND experiment by name and run it")
    r.add_argument("experiment")
    sub.add_parser("nd-finish", help="end the running ND experiment after this loop")
    sub.add_parser("shutdown", help="stop the bridge job")
    a = ap.parse_args()

    b = NISBridge(a.url)
    try:
        out = {"status": lambda: b.status(),
               "move": lambda: b.move_relative(a.dx, a.dy),
               "objective": lambda: b.change_objective(a.name),
               "capture": lambda: b.capture(a.path),
               "nd-run": lambda: b.nd_run(a.experiment),
               "nd-finish": lambda: b.nd_finish(),
               "shutdown": lambda: b.shutdown()}[a.cmd]()
    except (estop.EStopEngaged, BridgeUnavailable) as e:
        raise SystemExit(str(e))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
