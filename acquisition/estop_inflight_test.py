# estop_inflight_test.py
# ------------------------------------------------------------
# Can the e-stop stop a move that is ALREADY running?
#
#   .venv/Scripts/python -m acquisition.estop_inflight_test z    # Test A
#   .venv/Scripts/python -m acquisition.estop_inflight_test xy   # Test B
#
# The e-stop flag reliably refuses NEW moves. What is unknown is what
# happens to a move the controller has already accepted. XY reads are
# cached for the whole move (read_during_move_test.py), so writing the
# "current" XY back reverses the stage instead of freezing it.
#
# TEST A (Z). The Ti2 pushed Z-position events every ~500 ms during a
# move, which XY never did, so Z reads may be live. One raw Z move of
# Z_TEST_DROP_UM DOWNWARD - away from the sample, the only direction this
# test ever moves on its own - with a reader thread logging Z every 50 ms
# and a halt that writes the current Z back after HALT_AFTER_S. If Z reads
# are live, Z should stop partway. If they are cached, the write-back
# commands the start position, which is where we came from: the worst
# case of this test is going back up to where it began, never beyond.
# Afterwards Z is left where it stopped; restore focus by hand.
#
# TEST B (XY). No halt is possible, so the stop is the flag checked
# between move_xy's hops. A multi-hop move of XY_TEST_BY_UM in +X runs on
# a worker thread; the flag is engaged (no halt) after ENGAGE_AFTER_S.
# Reported: how long the stage kept moving after the engage, and how far
# it travelled in total. The stop is left ENGAGED - a human releases it.
#
# Raw COM, not NISSdk, for Test A: NISSdk caps a Z step at 50 um, which
# is over before any halt could land. Test B goes through NISSdk and
# move_xy on purpose, since that is the path real moves take.
# ------------------------------------------------------------

import os
import sys
import threading
import time

# Multi-threaded COM, so the reader and halt threads share the microscope
# object with the thread that is blocked in the move.
sys.coinit_flags = 0  # COINIT_MULTITHREADED

import pythoncom
import win32com.client
import NkTi2Ax

from acquisition import estop

Z_COUNTS_PER_UM = 100.0
Z_TEST_DROP_UM = 1000.0
HALT_AFTER_S = 0.2

XY_TEST_BY_UM = 20000.0
ENGAGE_AFTER_S = 1.5


def log(msg):
    now = time.time()
    stamp = f"{time.strftime('%H:%M:%S', time.localtime(now))}.{int(now % 1 * 1e6):06d}"
    print(f"{stamp}  [{threading.current_thread().name:>6}]  {msg}", flush=True)


def test_z():
    m = win32com.client.Dispatch(NkTi2Ax.NikonTi2AxAutoConnectMicroscope.CLSID)
    z0 = m.iZPOSITION
    lo = m.ZPosition.Lower
    target = z0 - round(Z_TEST_DROP_UM * Z_COUNTS_PER_UM)
    log(f"start Z {z0 / Z_COUNTS_PER_UM:.2f} um, target {target / Z_COUNTS_PER_UM:.2f} um (down)")
    if target < lo:
        log(f"target below travel range ({lo / Z_COUNTS_PER_UM:.1f} um) - not moving")
        return
    estop.check()

    stop = threading.Event()

    def reader():
        pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
        while not stop.is_set():
            log(f"Z read {m.iZPOSITION / Z_COUNTS_PER_UM:.2f}")
            time.sleep(0.05)

    def halter():
        pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
        z = m.iZPOSITION
        log(f"HALT: read Z {z / Z_COUNTS_PER_UM:.2f}, writing it back")
        try:
            m.iZPOSITION = z
            log("HALT: write-back accepted")
        except Exception as e:
            log(f"HALT: write-back failed: {e}")

    threading.Thread(target=reader, name="reader", daemon=True).start()
    time.sleep(0.2)  # a few idle reads first, as a baseline
    threading.Timer(HALT_AFTER_S, halter).start()

    t0 = time.perf_counter()
    log("set iZPOSITION start")
    try:
        m.iZPOSITION = target
        log(f"set iZPOSITION done after {time.perf_counter() - t0:.2f} s")
    except Exception as e:
        log(f"set iZPOSITION failed after {time.perf_counter() - t0:.2f} s: {e}")

    time.sleep(1.5)  # let whatever is still moving settle
    stop.set()
    z1 = m.iZPOSITION
    travelled = (z0 - z1) / Z_COUNTS_PER_UM
    log(f"final Z {z1 / Z_COUNTS_PER_UM:.2f} um - moved {travelled:.2f} of {Z_TEST_DROP_UM:.0f} um down")
    if abs(travelled) < 1:
        log("RESULT: Z back at start - halt read a stale position (or the move never ran)")
    elif travelled < Z_TEST_DROP_UM - 1:
        log("RESULT: Z stopped partway - halt works on Z")
    else:
        log("RESULT: Z reached target - halt did not stop it")


def test_xy():
    from acquisition.backends.nis_sdk import NISSdk
    from acquisition.move_xy import move_to

    sdk = NISSdk()
    x0, y0 = sdk.XY_GetPosition()
    log(f"start ({x0:.1f}, {y0:.1f}), moving +{XY_TEST_BY_UM:.0f} um in X")
    estop.check()

    at_engage = {}

    def engager():
        # Flag first: NISSdk has one COM thread, busy for the whole hop in
        # flight, so reading the position before engaging would delay the
        # engage by up to a hop - the very latency being measured.
        estop.engage("estop_inflight_test", halt_stage=False)
        at_engage["t"] = time.perf_counter()
        log("ENGAGED (flag only)")

    threading.Timer(ENGAGE_AFTER_S, engager).start()
    t0 = time.perf_counter()
    try:
        move_to(sdk, x0 + XY_TEST_BY_UM, y0)
        log("move finished WITHOUT being stopped")
    except estop.EStopEngaged:
        log("move refused its next hop - stopped by e-stop")
    t_end = time.perf_counter()

    x1, y1 = sdk.XY_GetPosition()
    log(f"final ({x1:.1f}, {y1:.1f}) after {t_end - t0:.2f} s")
    if at_engage:
        log(f"RESULT: kept moving {t_end - at_engage['t']:.2f} s after engage "
            f"(travelled {x1 - x0:.0f} of {XY_TEST_BY_UM:.0f} um in total)")
    log("e-stop left ENGAGED - release by hand: python -m acquisition.estop release")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("z", "xy"):
        print(__doc__ or "usage: python -m acquisition.estop_inflight_test z|xy")
        raise SystemExit(2)
    threading.current_thread().name = "main"
    # Hard stop in case anything hangs.
    threading.Timer(60.0, lambda: (log("60s safety timeout"), os._exit(1))).start()
    (test_z if sys.argv[1] == "z" else test_xy)()
    log("exit")
    os._exit(0)
