# Does reading iXPOSITION block while a move is in progress, or return a
# cached value? And does the SDK push position/moving events during a move?
#
# Moves XY to (0, 0) - X then Y, with blocking writes on the main thread -
# while a reader thread times every read and an event sink logs callbacks.
import os
import sys
import threading
import time

# Multi-threaded COM: the reader thread shares the microscope object, and
# events fire on the SDK's own thread without needing a message loop.
sys.coinit_flags = 0  # COINIT_MULTITHREADED

import pythoncom
import win32com.client
import NkTi2Ax

TARGET = (0, 0)
XY_STAGE_MASK = 0x2  # MIC_ACCESSORY_MASK_XYSTAGE

GENERAL_EVENTS = {0: "DataSet_PositionChanged", 1: "RemCtrl_PositionChanged",
                  3: "XYStageLogicalLimitsReached", 20: "DataSetReady",
                  21: "DataSetFinished", 22: "DataSetAborted"}
DEDICATED_EVENTS = {25: "Ti2_IsBusy", 38: "MovingStatus_Changed"}


def log(msg):
    now = time.time()
    stamp = f"{time.strftime('%H:%M:%S', time.localtime(now))}.{int(now % 1 * 1e6):06d}"
    print(f"{stamp}  [{threading.current_thread().name:>6}]  {msg}", flush=True)


class Events:
    def OnGeneralCallback(self, event, data):
        d = win32com.client.Dispatch(data, None, NkTi2Ax.INikonTi2AxData)
        log(f"EVENT general {event} {GENERAL_EVENTS.get(event, '?')} "
            f"mask=0x{d.uiDataUsageMask:x} data=({d.iXPOSITION}, {d.iYPOSITION})")

    def OnDedicatedCallback(self, event, data):
        log(f"EVENT dedicated {event} {DEDICATED_EVENTS.get(event, '?')} data={data!r}")

    def OnMetaDataCallback(self, mask):
        log(f"EVENT metadata 0x{mask:x}")

    def OnEnabledCallback(self, mask):
        log(f"EVENT enabled 0x{mask:x}")


def reader(stop):
    pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
    while not stop.is_set():
        t0 = time.perf_counter()
        log("read start")
        x, y = m.iXPOSITION, m.iYPOSITION
        log(f"read done ({x}, {y}) took {(time.perf_counter() - t0) * 1000:.1f} ms")
        time.sleep(0.05)


# Hard stop in case anything hangs.
threading.Timer(20.0, lambda: (log("20s safety timeout"), os._exit(1))).start()

threading.current_thread().name = "main"
log("dispatch start")
m = win32com.client.DispatchWithEvents(NkTi2Ax.NikonTi2AxAutoConnectMicroscope.CLSID, Events)
log("dispatch done")

for cmd, arg in (("SET_MOVING_STATE_NOTIFICATION", str(XY_STAGE_MASK)),
                 ("GET_MOVING_STATE_NOTIFICATION", "")):
    try:
        log(f"{cmd} -> {m.DedicatedCommand(cmd, arg)!r}")
    except Exception as e:
        log(f"{cmd} failed: {e}")

log(f"start position ({m.iXPOSITION}, {m.iYPOSITION})")

stop = threading.Event()
threading.Thread(target=reader, args=(stop,), name="reader").start()
time.sleep(0.2)  # a few idle reads first, as a baseline

log("set iXPOSITION start")
m.iXPOSITION = TARGET[0]
log("set iXPOSITION done")

log("set iYPOSITION start")
m.iYPOSITION = TARGET[1]
log("set iYPOSITION done")

time.sleep(0.5)  # a few reads after the move
stop.set()
time.sleep(0.2)
log("exit")
os._exit(0)
