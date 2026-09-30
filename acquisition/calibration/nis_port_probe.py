# IMPORTANT: 'limjob' must be imported like this (not from nor as)
import limjob

# A JOBS Python task that serves a tiny HTTP control API from inside NIS for
# 10 minutes (or until Abort). From a terminal on this PC:
#   curl http://127.0.0.1:8765/status
#   curl "http://127.0.0.1:8765/move?dx=200&dy=0"        relative, um
#   curl "http://127.0.0.1:8765/objective?name=<exact name from /status>"
#
# NIS macro functions are called through ctypes on g5_regprocs.dll, the way
# Nikon's own limpy.macro calls WaitText. Signatures are from the NIS 6.20
# macro reference (Docs/nis/eng_ar); macro char* is wchar_t* in this build.
#
# NIS calls are made only on the job's own thread (the run() loop); the HTTP
# thread just queues requests, since NIS is not known to be thread-safe.
#
# SAFETY: moves are relative and capped at MAX_STEP_UM per call; nothing
# moves while the e-stop file exists (same file acquisition/estop.py uses);
# Z is never moved; objective changes are limited to 4x and 10x, whose long
# working distances cannot reach the dish.
import ctypes as ct, http.server, json, os, queue, threading, time, urllib.parse

PORT, RUN_S, MAX_STEP_UM = 8765, 600, 1000.0
ESTOP = r"C:\Users\AX Confocal\.confocal-mcp\ESTOP"
ALLOWED_OBJECTIVES = ("4x", "10x")

nis = ct.cdll.g5_regprocs
for fn, args in {"StgGetPosXY": [ct.POINTER(ct.c_double)] * 2,
                 "StgGetPosZ": [ct.POINTER(ct.c_double), ct.c_int],
                 "StgMoveXY": [ct.c_double, ct.c_double, ct.c_int],
                 "Stg_GetNosepiecePosition": [],
                 "Stg_GetNosepiecePositions": [],
                 "Stg_GetNosepieceObjectiveName": [ct.c_int, ct.c_wchar_p, ct.c_int],
                 "GetCurrentObjName": [ct.c_wchar_p],
                 "ChangeObjective": [ct.c_wchar_p]}.items():
    getattr(nis, fn).argtypes, getattr(nis, fn).restype = args, ct.c_int


def status():
    x, y, z = ct.c_double(), ct.c_double(), ct.c_double()
    rc_xy = nis.StgGetPosXY(ct.byref(x), ct.byref(y))
    rc_z = nis.StgGetPosZ(ct.byref(z), 0)
    cur = ct.create_unicode_buffer(255); nis.GetCurrentObjName(cur)
    names = {}
    for i in range(0, nis.Stg_GetNosepiecePositions() + 1):   # index base unknown: try 0..n
        b = ct.create_unicode_buffer(255)
        if nis.Stg_GetNosepieceObjectiveName(i, b, 255) == 1 and b.value:
            names[i] = b.value
    return {"xy_um": [x.value, y.value], "xy_rc": rc_xy, "z_um": z.value, "z_rc": rc_z,
            "nosepiece_position": nis.Stg_GetNosepiecePosition(),
            "current_objective": cur.value, "nosepiece_objectives": names}


def move(dx, dy):
    if os.path.exists(ESTOP):
        return {"error": "e-stop engaged - not moving"}
    if max(abs(dx), abs(dy)) > MAX_STEP_UM:
        return {"error": f"step over {MAX_STEP_UM} um refused"}
    before = status()["xy_um"]
    rc = nis.StgMoveXY(dx, dy, 1)                                # 1 = MOVE_RELATIVE
    return {"rc": rc, "before_um": before, "after_um": status()["xy_um"]}


def objective(name):
    if os.path.exists(ESTOP):
        return {"error": "e-stop engaged - not changing objective"}
    if not any(k in name for k in ALLOWED_OBJECTIVES):
        return {"error": f"only {ALLOWED_OBJECTIVES} objectives allowed here"}
    before = status()
    rc = nis.ChangeObjective(name)
    after = status()
    return {"rc": rc, "before": before["current_objective"], "after": after["current_objective"],
            "nosepiece_before": before["nosepiece_position"],
            "nosepiece_after": after["nosepiece_position"]}


jobs = queue.Queue()


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = dict(urllib.parse.parse_qsl(u.query))
        call = {"/status": lambda: status(),
                "/move": lambda: move(float(q.get("dx", 0)), float(q.get("dy", 0))),
                "/objective": lambda: objective(q.get("name", ""))}.get(u.path)
        if call is None:
            out = {"error": "use /status, /move?dx=&dy=, /objective?name="}
        else:
            box = queue.Queue(); jobs.put((call, box))
            try:
                out = box.get(timeout=60)
            except queue.Empty:
                out = {"error": "timed out waiting for NIS"}
        body = (json.dumps(out, indent=1) + "\n").encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(body)


def run(imgs: tuple[limjob.Image], Job: limjob.JobParam, macro: limjob.MacroParam, ctx: limjob.RunContext):
    s = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    print(f"listening on 127.0.0.1:{PORT}, pid {os.getpid()}")
    abort = getattr(ctx, "shouldAbort", lambda: False)
    t0 = time.time()
    try:
        while time.time() - t0 < RUN_S and not abort():
            try:
                call, box = jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                box.put(call())
            except Exception as e:
                box.put({"error": f"{type(e).__name__}: {e}"})
    finally:
        s.shutdown(); s.server_close()
        print("server stopped")
