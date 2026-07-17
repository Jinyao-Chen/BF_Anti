# tests/test_rc_mux.py
from rc_mux import RCMux, TrackCmd


class FakeMSP:
    def __init__(self):
        self.calls = []

    def set_rc(self, roll=1500, pitch=1500, throttle=1000, yaw=1500,
               aux1=1500, aux2=1500, aux3=1500, aux4=1500):
        self.calls.append((roll, pitch, throttle, yaw, aux1, aux2, aux3, aux4))


class FakeCRSF:
    def __init__(self):
        self.ch = [1500] * 16
        self.ch[2] = 1000  # 油门低位
        self._age = 0.0

    def channels(self):
        return list(self.ch)

    def age(self, now=None):
        return self._age


def make_mux(**kw):
    msp, crsf = FakeMSP(), FakeCRSF()
    mux = RCMux(msp, crsf=crsf, switch_channel=7, **kw)
    return mux, msp, crsf


def test_passthrough_forwards_rpyt_and_aux():
    mux, msp, crsf = make_mux()
    crsf.ch[0] = 1600
    crsf.ch[2] = 1234
    crsf.ch[4] = 2000
    mux._tick(10.0)
    assert msp.calls[-1] == (1600, 1500, 1234, 1500, 2000, 1500, 1500, 1500)
    assert mux.mode == "PASSTHROUGH"


def test_crsf_silent_stops_all_output():
    mux, msp, crsf = make_mux()
    mux._tick(10.0)
    n = len(msp.calls)
    crsf._age = 1.0
    mux._tick(10.02)
    assert len(msp.calls) == n
    assert mux.mode == "SILENT"


def test_recovery_returns_to_passthrough():
    mux, msp, crsf = make_mux()
    crsf._age = 1.0
    mux._tick(10.0)
    crsf._age = 0.01
    mux._tick(10.5)
    assert mux.mode == "PASSTHROUGH"
    assert len(msp.calls) == 1
