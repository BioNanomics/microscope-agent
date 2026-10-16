# Fake hardware for the tests: a Ti2 ActiveX microscope and the modules
# nis_sdk imports at the top (pythoncom, win32com, NkTi2Ax), so nis_sdk -
# and everything built on it - can be imported and run on a plain CI runner.
import importlib
import sys
import types

import pytest


class _Setting:
    def __init__(self, lower, higher):
        self.Lower, self.Higher = lower, higher
        self.Control, self.Enabled, self.Unit, self.Scale = 1, 1, "u", 1.0


class _Objective:
    IsPFSEnabled = True
    Model = "Fake 10x"
    Magnification = 10
    WorkingDistance = 4.0


class _Data:
    """What DataGet fills in: _prop_map_get_ names the readable properties."""
    _prop_map_get_ = {"iLIGHTPATH": 1, "iDIA_LAMP_Switch": 1, "iXPOSITION": 1,
                      "iNOSEPIECE": 1, "iBROKEN": 1}
    iLIGHTPATH = 2
    iDIA_LAMP_Switch = 1
    iXPOSITION = 5
    iNOSEPIECE = 3

    @property
    def iBROKEN(self):
        raise RuntimeError("unreadable")


class FakeMicroscope:
    """Counts: X/Y 0.1 um, Z 0.01 um (see nis_sdk). Writes land at once."""

    def __init__(self):
        self.iXPOSITION = 1000          # 100.0 um
        self.iYPOSITION = 2000          # 200.0 um
        self.iZPOSITION = 461900        # 4619.00 um
        self.XPosition = _Setting(-570000, 570000)
        self.YPosition = _Setting(-375000, 375000)
        self.ZPosition = _Setting(0, 1000000)
        self.PfsOffset = _Setting(0, 40000)
        self.iPFS_OFFSET = 18966
        self.iPFS_STATUS = 7
        self.iNOSEPIECE = 1
        self.objective = _Objective()
        self.iLIGHTPATH = 1
        self.iDIA_LAMP_Switch = 0

    def Objective(self, n):
        return self.objective

    def DataGet(self, data, flag):
        return _Data()


@pytest.fixture
def fake_ti2(monkeypatch):
    """Import a fresh acquisition.backends.nis_sdk bound to a FakeMicroscope.
    Yields (nis_sdk module, microscope)."""
    micro = FakeMicroscope()
    pythoncom = types.ModuleType("pythoncom")
    pythoncom.CoInitialize = lambda: None
    pythoncom.PumpWaitingMessages = lambda: None
    win32com = types.ModuleType("win32com")
    client = types.ModuleType("win32com.client")
    client.Dispatch = lambda clsid: micro
    win32com.client = client
    nk = types.ModuleType("NkTi2Ax")
    nk.NikonTi2AxAutoConnectMicroscope = types.SimpleNamespace(CLSID="fake-clsid")
    for name, mod in (("pythoncom", pythoncom), ("win32com", win32com),
                      ("win32com.client", client), ("NkTi2Ax", nk)):
        monkeypatch.setitem(sys.modules, name, mod)
    from acquisition import estop
    monkeypatch.setattr(estop, "launch_panel", lambda: None)      # no STOP window on a test run
    sys.modules.pop("acquisition.backends.nis_sdk", None)
    sdk = importlib.import_module("acquisition.backends.nis_sdk")
    yield sdk, micro
    sys.modules.pop("acquisition.backends.nis_sdk", None)
