import os
import sys
import threading
import time

# Multi-threaded COM, so the watchdog thread can use the same microscope
# object as the main thread. Must be set before pythoncom is imported.
sys.coinit_flags = 0  # COINIT_MULTITHREADED

import pythoncom
import win32com.client
import NkTi2Ax


def log(msg):
    now = time.time()
    print(f"{time.strftime('%H:%M:%S', time.localtime(now))}.{int(now % 1 * 1e6):06d}  {msg}", flush=True)


def watchdog():
    pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
    log("watchdog: 500ms elapsed, halting")
    x, y = m.iXPOSITION, m.iYPOSITION
    m.iXPOSITION, m.iYPOSITION = x, y
    log(f"watchdog: halted at ({x}, {y})")


log("dispatch start")
m = win32com.client.Dispatch(NkTi2Ax.NikonTi2AxAutoConnectMicroscope.CLSID)
log("dispatch done")
log(f"start position ({m.iXPOSITION}, {m.iYPOSITION})")

# Armed after the connect, so the halt always has a microscope to talk to.
timer = threading.Timer(0.5, watchdog)
timer.start()
log("watchdog armed (500ms)")

log("set iXPOSITION start")
m.iXPOSITION = 0
log("set iXPOSITION done")

log("set iYPOSITION start")
m.iYPOSITION = 0
log("set iYPOSITION done")

# Track progress until the halt, then for 1s after to see where it settles.
halted_at = None
while halted_at is None or time.time() - halted_at < 1.0:
    log(f"position ({m.iXPOSITION}, {m.iYPOSITION})")
    if halted_at is None and not timer.is_alive():
        halted_at = time.time()
    time.sleep(0.05)

log("exit")
os._exit(0)
