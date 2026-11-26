"""Three-position pilot mode: FORCE MANUAL / AUTO / STRIKE.

The companion must never arm, must never engage while the pilot is in the low
position or disarmed, and must be able to engage automatically after a manual
arm edge when a target is already visible.
"""
from rc_mux import RCMux, TrackCmd


class FakeFC:
    def __init__(self):
        self.calls = []

    def send_rc_16(self, channels):
        self.calls.append(tuple(channels))

    def frames(self):
        return self.calls


class Source:
    def __init__(self):
        self.ch = [1500] * 16
        self.ch[2] = 1200       # throttle
        self.ch[4] = 1000       # ARM low
        self.ch[6] = 2000       # MSP override high (pilot)
        self.ch[7] = 1500       # mode mid (AUTO)
        self.age = 0.0

    def __call__(self):
        return tuple(self.ch), self.age


def make_mux(source, **kwargs):
    mux = RCMux(
        FakeFC(),
        mode_channel=7,
        arm_channel=4,
        channel_source=source,
        rc_stale_timeout=0.5,
        stick_takeover=False,
        **kwargs,
    )
    mux.RAMP_TIME = 0.001
    # Observe the boot state (disarmed, mode LOW) so the first manual arm is
    # an edge and the C7 "must have seen LOW" condition is satisfied.
    saved_arm = source.ch[4]
    saved_mode = source.ch[7]
    source.ch[4] = 1000
    source.ch[7] = 1000
    mux._tick(0.0)
    source.ch[4] = saved_arm
    source.ch[7] = saved_mode
    return mux


def test_force_manual_never_engages_even_with_request():
    source = Source()
    source.ch[7] = 1000
    mux = make_mux(source)
    entered = []
    mux.on_engage = lambda thr: entered.append(thr)
    mux.request_auto(True)
    mux.publish_cmd(TrackCmd(roll=1600, pitch=1700, throttle=1500, yaw=1300, t=10.0))
    for i in range(10):
        mux._tick(10.0 + i * 0.02)
    assert mux.mode_position() == 0
    assert mux._auto is False
    assert entered == []
    assert mux.msp.frames()  # still sends pilot passthrough
    assert mux.msp.frames()[-1][0:4] == (1500, 1500, 1200, 1500)


def test_auto_engages_only_after_manual_arm_edge():
    source = Source()
    mux = make_mux(source, base_throttle=1200)
    entered = []
    mux.on_engage = lambda thr: entered.append(thr)
    mux.request_auto(True)
    mux.publish_cmd(TrackCmd(roll=1600, pitch=1700, throttle=1500, yaw=1300, t=10.0))

    # Mode is mid at boot, but ARM is low: no engagement and no motion.
    mux._tick(10.0)
    assert mux.mode_position() == 1
    assert mux._auto is False

    # Pilot arms.  The rising edge is consent; AUTO may now engage.
    source.ch[4] = 2000
    mux._tick(10.02)
    assert mux._auto is True
    assert entered and entered[0] == 1200
    mux._tick(10.04)
    assert mux.mode == "AUTO"
    ch = mux.msp.frames()[-1]
    assert ch[0:4] == (1600, 1700, 1500, 1300)
    assert ch[4] == 2000      # ARM passthrough, never commanded by companion
    assert ch[6] == 2000      # override passthrough


def test_force_manual_switch_aborts_immediately_and_can_reengage():
    source = Source()
    source.ch[4] = 2000
    mux = make_mux(source)
    mux.request_auto(True)
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.0))
    mux._tick(10.0)
    assert mux._auto is True

    source.ch[7] = 1000
    mux._tick(10.02)
    assert mux._auto is False
    assert mux.mode_position() == 0
    assert mux.msp.frames()[-1][0:4] == (1500, 1500, 1200, 1500)

    # Back to mid: the safety gate is still green, so AUTO re-engages.
    source.ch[7] = 1500
    mux._tick(10.04)
    assert mux._auto is True


def test_disarm_kills_auto_and_rearm_reengages():
    source = Source()
    source.ch[4] = 2000
    mux = make_mux(source)
    mux.request_auto(True)
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.0))
    mux._tick(10.0)
    assert mux._auto is True

    source.ch[4] = 1000
    mux._tick(10.02)
    assert mux._auto is False
    assert mux.msp.frames()[-1][4] == 1000

    source.ch[4] = 2000
    mux._tick(10.04)
    assert mux._auto is True


def test_strike_position_is_reported_and_engages():
    source = Source()
    source.ch[4] = 2000
    source.ch[7] = 2000
    mux = make_mux(source)
    mux.request_auto(True)
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.0))
    mux._tick(10.0)
    assert mux.mode_position() == 2
    assert mux.mode_name() == "STRIKE"
    assert mux._auto is True


def test_stale_rc_returns_to_manual_and_inhibits():
    source = Source()
    source.ch[4] = 2000
    mux = make_mux(source)
    mux.request_auto(True)
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.0))
    mux._tick(10.0)
    assert mux._auto is True
    source.age = 0.6
    mux._tick(10.02)
    assert mux._auto is False
    assert mux.mode == "FC_STALE"
    assert mux.mode_position() == 0


def test_force_manual_low_edge_resets_latched_safety():
    source = Source()
    source.ch[4] = 2000
    mux = make_mux(source)
    calls = []
    mux.on_disengage = lambda: calls.append(1)
    mux.request_auto(False)
    mux._tick(10.0)
    source.ch[7] = 1000
    mux._tick(10.02)          # low edge = pilot reset
    source.ch[7] = 1000
    mux._tick(10.04)          # already low: no repeated reset
    assert calls == [1]


def test_manual_disarm_edge_resets_latched_safety():
    source = Source()
    source.ch[4] = 2000
    mux = make_mux(source)
    calls = []
    mux.on_disengage = lambda: calls.append(1)
    mux.request_auto(False)
    mux._tick(10.0)
    source.ch[4] = 1000
    mux._tick(10.02)          # disarm edge = pilot reset
    source.ch[4] = 1000
    mux._tick(10.04)          # already disarmed: no repeated reset
    assert calls == [1]
