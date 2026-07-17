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


# --- Switch engage / takeover / inhibit (Task 5) ---


def test_switch_high_at_startup_does_not_engage():
    mux, msp, crsf = make_mux()
    fired = []
    mux.on_engage = lambda thr: fired.append(thr)
    crsf.ch[7] = 2000
    mux._tick(10.0)
    mux._tick(10.02)
    assert fired == []
    assert mux.mode == "PASSTHROUGH"


def test_engage_after_low_then_high_captures_throttle():
    mux, msp, crsf = make_mux()
    fired = []
    mux.on_engage = lambda thr: fired.append(thr)
    crsf.ch[7] = 1000
    mux._tick(10.0)
    crsf.ch[2] = 1540
    crsf.ch[7] = 2000
    mux._tick(10.02)
    assert fired == [1540]
    assert mux.engaged_throttle == 1540


def _engaged_mux():
    mux, msp, crsf = make_mux()
    crsf.ch[7] = 1000
    mux._tick(10.0)
    crsf.ch[2] = 1540
    crsf.ch[7] = 2000
    mux._tick(10.02)
    return mux, msp, crsf


def test_rpy_takeover_reverts_and_inhibits():
    mux, msp, crsf = _engaged_mux()
    dis = []
    mux.on_disengage = lambda: dis.append(1)
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.02))
    mux._tick(10.04)
    assert mux.mode == "AUTO"
    crsf.ch[0] = 1900
    mux._tick(10.06)
    assert mux.mode == "PASSTHROUGH" and dis == [1]
    # switch still high, must not re-engage (inhibit)
    crsf.ch[0] = 1500
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.06))
    mux._tick(10.08)
    assert mux.mode == "PASSTHROUGH"
    # switch low->high re-enables
    crsf.ch[7] = 1000
    mux._tick(10.10)
    crsf.ch[7] = 2000
    mux._tick(10.12)
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.12))
    mux._tick(10.14)
    assert mux.mode == "AUTO"


def test_throttle_takeover_uses_delta_not_center():
    mux, msp, crsf = _engaged_mux()
    mux.publish_cmd(TrackCmd(pitch=1500, yaw=1500, throttle=1550, t=10.02))
    mux._tick(10.04)
    assert mux.mode == "AUTO"
    crsf.ch[2] = 1300  # delta 240 > 150 -> takeover
    mux._tick(10.06)
    assert mux.mode == "PASSTHROUGH"


def test_silent_while_engaged_disengages_and_needs_low():
    mux, msp, crsf = _engaged_mux()
    dis = []
    mux.on_disengage = lambda: dis.append(1)
    crsf._age = 1.0
    mux._tick(11.0)
    assert mux.mode == "SILENT" and dis == [1]
    crsf._age = 0.01
    mux._tick(11.5)
    assert mux.mode == "PASSTHROUGH"
    mux._tick(11.52)
    assert mux.mode == "PASSTHROUGH"  # switch still high, must not re-engage


# --- Cmd substitution / ramp / stale / bench (Task 6) ---


def test_stale_cmd_falls_back_to_passthrough_without_disengage():
    mux, msp, crsf = _engaged_mux()
    dis = []
    mux.on_disengage = lambda: dis.append(1)
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.02))
    mux._tick(10.04)
    assert mux.mode == "AUTO"
    mux._tick(11.0)  # cmd age 0.98s > 0.5 -> vision hung
    assert mux.mode == "PASSTHROUGH"
    assert dis == []
    r, p, t, y = msp.calls[-1][:4]
    assert (r, p, t, y) == (1500, 1500, 1540, 1500)


def test_fresh_cmd_substitutes_fully_after_ramp():
    mux, msp, crsf = _engaged_mux()
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.02))
    mux._tick(10.04)
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.64))
    mux._tick(10.64)
    roll, pitch, thr, yaw, aux1 = msp.calls[-1][:5]
    assert (pitch, yaw, thr) == (1700, 1300, 1600)
    assert aux1 == 1500


def test_bench_toggle_arms_and_disarms():
    msp = FakeMSP()
    mux = RCMux(msp, crsf=None, base_throttle=1100)
    assert mux.bench_toggle() is True
    mux._tick(10.0)
    assert mux.mode == "BENCH"
    assert msp.calls[-1] == (1500, 1500, 1000, 1500, 2000, 1500, 1500, 1500)
    mux.publish_cmd(TrackCmd(pitch=1600, yaw=1400, throttle=1200, t=10.0))
    mux._tick(10.0)
    mux.publish_cmd(TrackCmd(pitch=1600, yaw=1400, throttle=1200, t=10.6))
    mux._tick(10.6)
    assert msp.calls[-1][:4] == (1500, 1600, 1200, 1400)
    assert mux.bench_toggle() is False
    mux._tick(10.7)
    assert msp.calls[-1] == (1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)
