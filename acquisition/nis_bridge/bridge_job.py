# IMPORTANT: 'limjob' must be imported like this (not from nor as)
try:
    import limjob
except ImportError:          # off the microscope (tests): the module is only needed for type hints
    limjob = None

# bridge_job.py
# ------------------------------------------------------------
# The NIS-Elements side of the NIS bridge: paste this file into a JOBS
# Python task and run the job. While the job runs, NIS answers a small
# JSON-over-HTTP API on 127.0.0.1:8766, which acquisition/backends/
# nis_bridge.py (and through it the NIS MCP server) talks to.
#
#   GET  /status                      positions, objective, ND run state
#   POST /move        {"dx", "dy"}    relative XY move, um
#   POST /objective   {"name"}        ChangeObjective, allow-listed names only
#   POST /capture     {"path"}        Capture, save as ND2, close the document
#   POST /nd/run      {"experiment"}  ND_LoadExperiment + ND_RunExperiment(0)
#   POST /nd/finish                   ND_FinishExperiment (stop after this loop)
#   POST /shutdown                    stop serving and end the job
#
# WHY A BRIDGE AND NOT THE MCP SERVER ITSELF INSIDE NIS. Nothing has to be
# installed into NIS's own Python (Nikon pins its packages), a crash in our
# code cannot take NIS down with it, and the MCP server keeps the e-stop,
# confirm gate and tests it already has. This file only translates HTTP
# requests into NIS macro calls.
#
# HOW NIS IS CALLED. NIS macro functions are exported by g5_regprocs.dll
# and called through ctypes - the way NIS's own limpy.macro calls
# WaitText. Signatures are from the NIS 6.20 macro reference
# (Docs/nis/eng_ar on the microscope PC). A macro char* is wchar_t* in
# this build (limpy passes WaitText's text as c_wchar_p).
#
# THREADING. NIS is not known to be thread-safe, so every NIS call runs on
# the job's own thread in run(); the HTTP threads only queue requests and
# wait for the answer. A call that takes long (ND_RunExperiment may block
# until the run ends - unverified) delays the requests queued behind it.
#
# SAFETY, enforced here as well as in the client:
#   * motion and acquisition only on POST, so a browser cannot trigger them
#   * nothing moves or starts while the e-stop file exists (the same file
#     acquisition/estop.py uses)
#   * XY moves are relative and capped at MAX_STEP_UM; Z is never moved
#   * objective changes only to 4x/10x names (ALLOWED_MAGNIFICATIONS):
#     working distances of 20 and 4 mm cannot reach the dish
#   * serves on 127.0.0.1 only
# ------------------------------------------------------------
import ctypes as ct
import http.server
import json
import os
import queue
import re
import threading
import time
from pathlib import Path

PORT = 8766                     # 8765 is the plain port probe (nis_port_probe.py)
VERSION = 1
MAX_STEP_UM = 1000.0
ALLOWED_MAGNIFICATIONS = (4, 10)
# "4x" / "10x" as a whole magnification - not the 4x inside "14x" or "40x"
_MAG = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*[xX](?![a-zA-Z0-9])")
ESTOP = Path(os.environ.get("CONFOCAL_ESTOP_FILE",
                            Path.home() / ".confocal-mcp" / "ESTOP"))

MOVE_RELATIVE = 1               # StgMoveXY 'relative' argument
ND2_ALL_LAYERS = 14             # ImageSaveAs ImType: all layers (ND2 file)
QUERYSAVE_NO = 2                # CloseCurrentDocument: close without asking


class Macro:
    """ctypes view of the NIS macro functions this bridge uses."""

    SIGNATURES = {
        "StgGetPosXY": [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)],
        "StgGetPosZ": [ct.POINTER(ct.c_double), ct.c_int],
        "StgMoveXY": [ct.c_double, ct.c_double, ct.c_int],
        "Stg_GetNosepiecePosition": [],
        "Stg_GetNosepiecePositions": [],
        "Stg_GetNosepieceObjectiveName": [ct.c_int, ct.c_wchar_p, ct.c_int],
        "GetCurrentObjName": [ct.c_wchar_p],
        "ChangeObjective": [ct.c_wchar_p],
        "Capture": [],
        "ImageSaveAs": [ct.c_wchar_p, ct.c_int, ct.c_int],
        "CloseCurrentDocument": [ct.c_int],
        "ND_LoadExperiment": [ct.c_wchar_p],
        "ND_RunExperiment": [ct.c_int],
        "ND_IsInExperimentCapture": [],
        "ND_FinishExperiment": [],
    }

    def __init__(self, dll=None):
        self.dll = dll if dll is not None else ct.cdll.g5_regprocs
        for name, args in self.SIGNATURES.items():
            fn = getattr(self.dll, name)
            fn.argtypes, fn.restype = args, ct.c_int

    def __getattr__(self, name):
        return getattr(self.dll, name)


def _estop_engaged() -> bool:
    try:
        return ESTOP.exists()
    except OSError:
        return True             # cannot tell -> treat as engaged, like estop.py


class Bridge:
    """The operations, as plain methods returning JSON-able dicts."""

    def __init__(self, macro: Macro):
        self.m = macro

    def status(self) -> dict:
        x, y, z = ct.c_double(), ct.c_double(), ct.c_double()
        rc_xy = self.m.StgGetPosXY(ct.byref(x), ct.byref(y))
        rc_z = self.m.StgGetPosZ(ct.byref(z), 0)
        cur = ct.create_unicode_buffer(255)
        self.m.GetCurrentObjName(cur)
        names = {}
        for i in range(0, self.m.Stg_GetNosepiecePositions() + 1):  # index base unverified
            b = ct.create_unicode_buffer(255)
            if self.m.Stg_GetNosepieceObjectiveName(i, b, 255) == 1 and b.value:
                names[str(i)] = b.value
        return {"version": VERSION, "pid": os.getpid(),
                "xy_um": [x.value, y.value], "xy_rc": rc_xy,
                "z_um": z.value, "z_rc": rc_z,
                "nosepiece_position": self.m.Stg_GetNosepiecePosition(),
                "current_objective": cur.value, "nosepiece_objectives": names,
                "nd_running": bool(self.m.ND_IsInExperimentCapture()),
                "estop_engaged": _estop_engaged()}

    def _refuse_if_stopped(self, what: str) -> dict | None:
        if _estop_engaged():
            return {"error": f"e-stop engaged - {what} refused", "estop_engaged": True}
        return None

    def move(self, dx: float, dy: float) -> dict:
        if (r := self._refuse_if_stopped("move")):
            return r
        if max(abs(dx), abs(dy)) > MAX_STEP_UM:
            return {"error": f"step over {MAX_STEP_UM} um refused"}
        before = self.status()["xy_um"]
        rc = self.m.StgMoveXY(float(dx), float(dy), MOVE_RELATIVE)
        return {"rc": rc, "before_um": before, "after_um": self.status()["xy_um"]}

    def objective(self, name: str) -> dict:
        if (r := self._refuse_if_stopped("objective change")):
            return r
        mags = {float(m) for m in _MAG.findall(name)}
        if len(mags) != 1 or mags.pop() not in ALLOWED_MAGNIFICATIONS:
            return {"error": f"only {ALLOWED_MAGNIFICATIONS}x objectives are allowed, got {name!r}"}
        before = self.status()
        rc = self.m.ChangeObjective(name)
        after = self.status()
        return {"rc": rc, "before": before["current_objective"], "after": after["current_objective"],
                "nosepiece_before": before["nosepiece_position"],
                "nosepiece_after": after["nosepiece_position"]}

    def capture(self, path: str) -> dict:
        if (r := self._refuse_if_stopped("capture")):
            return r
        if not path.lower().endswith(".nd2"):
            return {"error": "path must end in .nd2"}
        if os.path.exists(path):
            return {"error": f"{path} already exists - refusing to overwrite"}
        rc_cap = self.m.Capture()
        rc_save = self.m.ImageSaveAs(path, ND2_ALL_LAYERS, 0)
        rc_close = self.m.CloseCurrentDocument(QUERYSAVE_NO)
        return {"rc_capture": rc_cap, "rc_save": rc_save, "rc_close": rc_close,
                "path": path, "saved": os.path.exists(path)}

    def nd_run(self, experiment: str) -> dict:
        if (r := self._refuse_if_stopped("ND run")):
            return r
        if self.m.ND_IsInExperimentCapture():
            return {"error": "an ND experiment is already running"}
        rc_load = self.m.ND_LoadExperiment(experiment)
        if rc_load != 1:
            return {"error": f"ND_LoadExperiment({experiment!r}) returned {rc_load}", "rc_load": rc_load}
        t0 = time.time()
        rc_run = self.m.ND_RunExperiment(0)          # 0: save to disk, do not open
        return {"rc_load": rc_load, "rc_run": rc_run,
                "call_returned_after_s": round(time.time() - t0, 1),
                "nd_running": bool(self.m.ND_IsInExperimentCapture())}

    def nd_finish(self) -> dict:
        # Always allowed, e-stop or not: it only ends a run early.
        rc = self.m.ND_FinishExperiment()
        return {"rc": rc, "nd_running": bool(self.m.ND_IsInExperimentCapture())}


# Routes: (method, path) -> function(bridge, body) -> dict
ROUTES = {
    ("GET", "/status"): lambda b, body: b.status(),
    ("POST", "/move"): lambda b, body: b.move(float(body.get("dx", 0)), float(body.get("dy", 0))),
    ("POST", "/objective"): lambda b, body: b.objective(str(body.get("name", ""))),
    ("POST", "/capture"): lambda b, body: b.capture(str(body.get("path", ""))),
    ("POST", "/nd/run"): lambda b, body: b.nd_run(str(body.get("experiment", ""))),
    ("POST", "/nd/finish"): lambda b, body: b.nd_finish(),
}


def make_handler(jobs: "queue.Queue", stop: threading.Event, timeout_s: float = 7200.0):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _answer(self, code: int, out: dict) -> None:
            body = (json.dumps(out) + "\n").encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _dispatch(self, method: str) -> None:
            path = self.path.split("?")[0]
            if method == "POST" and path == "/shutdown":
                stop.set()
                return self._answer(200, {"stopping": True})
            fn = ROUTES.get((method, path))
            if fn is None:
                return self._answer(404, {"error": f"no route {method} {path}",
                                          "routes": [f"{m} {p}" for m, p in ROUTES]})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
            except ValueError:
                return self._answer(400, {"error": "body is not JSON"})
            box = queue.Queue()
            jobs.put((fn, body, box))
            try:
                self._answer(200, box.get(timeout=timeout_s))
            except queue.Empty:
                self._answer(504, {"error": "timed out waiting for NIS"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def log_message(self, *args):      # keep the JOBS log readable
            pass

    return Handler


def serve(bridge: Bridge, port: int = PORT, should_abort=lambda: False,
          stop: threading.Event | None = None, on_ready=None) -> None:
    """Serve until /shutdown, should_abort() or `stop` is set. NIS calls run
    here, on the calling thread; HTTP requests are handled on others."""
    jobs = queue.Queue()
    stop = stop or threading.Event()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), make_handler(jobs, stop))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"NIS bridge v{VERSION} listening on 127.0.0.1:{srv.server_address[1]}, pid {os.getpid()}")
    if on_ready:
        on_ready(srv.server_address[1])
    try:
        while not stop.is_set() and not should_abort():
            try:
                fn, body, box = jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                box.put(fn(bridge, body))
            except Exception as e:
                box.put({"error": f"{type(e).__name__}: {e}"})
    finally:
        srv.shutdown()
        srv.server_close()
        print("NIS bridge stopped")


def run(imgs: "tuple[limjob.Image]", Job: "limjob.JobParam", macro: "limjob.MacroParam",
        ctx: "limjob.RunContext"):
    """JOBS entry point: serve until Abort or POST /shutdown. The hints are
    the JOBS template's own, as strings so this file imports off NIS."""
    serve(Bridge(Macro()), should_abort=getattr(ctx, "shouldAbort", lambda: False))
