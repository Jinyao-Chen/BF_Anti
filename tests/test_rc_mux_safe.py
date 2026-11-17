# tests/test_rc_mux_safe.py
"""FC-mode RC mux safety: no guessed AUX, no frame when the RC cache is stale."""
from rc_mux import RCMux, TrackCmd


class FakeFC:
    """Link double that records both the 16ch writer and the 8ch compat writer."""

    def __init__(self):
        self.calls = []

    def set_rc(self, roll=1500, pitch=1500, throttle=1000, yaw=1500,
               aux1=1500, aux2=1500, aux3=1500, aux4=1500):
        self.calls.append(("set_rc", (roll, pitch, throttle, yaw,
                                      aux1, aux2, aux3, aux4)))

    def send_rc_16(self, channels):
        self.calls.append(("rc16", tuple(channels)))

    def frames(self):
        return [payload for _kind, payload in self.calls]


class LegacyFC:
    """Old RCMux-compatible object without send_rc_16."""

    def __init__(self):
        self.calls = []

    def set_rc(self, roll=1500, pitch=1500, throttle=1000, yaw=1500,
               aux1=1500, aux2=1500, aux3=1500, aux4=1500):
        self.calls.append((roll, pitch, throttle, yaw, aux1, aux2, aux3, aux4))


def pilot_channels():
    channels = [1500] * 16
    channels[2] = 1200       # throttle
    channels[4] = 1000       # AUX1 / ARM low
    channels[5] = 1500       # AUX2 / ANGLE mid
    channels[6] = 1100       # AUX3 / override low
    channels[7] = 1000       # AUX4 / engage low
    return channels


class Source:
    def __init__(self, channels="default", age=0.0):
        self.channels = pilot_channels() if channels == "default" else (
            list(channels) if channels is not None else None)
        self.age = age

    def __call__(self):
        return (tuple(self.channels) if self.channels is not None else None,
                self.age)


def make_mux(source, msp=None, **kwargs):
    return RCMux(msp or FakeFC(), switch_channel=7, channel_source=source,
                 rc_stale_timeout=kwargs.pop("rc_stale_timeout", 0.5), **kwargs)


def test_stale_cache_sends_no_msp_rc_frame():
    msp = FakeFC()
    mux = make_mux(Source(age=0.0), msp)
    mux._tick(10.0)                       # fresh: one frame
    assert len(msp.calls) == 1
    source = Source(channels=pilot_channels(), age=0.0)
    mux = make_mux(source, msp)
    mux._tick(10.0)
    n = len(msp.calls)
    source.age = 0.6                      # stale > rc_stale_timeout
    mux._tick(10.02)
    assert len(msp.calls) == n            # nothing sent at all
    assert mux.mode == "FC_STALE"


def test_none_cache_and_short_cache_send_nothing():
    for source in (Source(channels=None), Source(channels=[1500] * 4)):
        msp = FakeFC()
        mux = make_mux(source, msp)
        mux._tick(10.0)
        assert msp.calls == []
        assert mux.mode == "FC_STALE"


def test_legacy_writer_also_silent_when_stale():
    msp = LegacyFC()
    mux = make_mux(Source(age=5.0), msp)
    mux._tick(10.0)
    assert msp.calls == []


def test_stale_while_engaged_disengages_and_never_replays_cmd():
    source = Source()
    msp = FakeFC()
    mux = make_mux(source, msp)
    disengaged = []
    mux.on_disengage = lambda: disengaged.append(1)
    mux._tick(10.0)                       # switch low seen
    source.channels[2] = 1400
    source.channels[7] = 2000
    mux._tick(10.02)                      # engage
    assert mux._auto is True
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1500, t=10.02))
    mux._tick(10.04)
    assert mux.mode == "AUTO"
    n = len(msp.calls)
    source.age = 0.6
    mux._tick(10.06)
    assert len(msp.calls) == n            # stale path must not send the command
    assert disengaged == [1]
    assert mux._auto is False
    assert mux.mode == "FC_STALE"


def test_aux_is_pilot_passthrough_never_hardcoded_override():
    source = Source()
    source.channels[6] = 1100             # pilot override switch LOW
    source.channels[7] = 2000             # engage switch HIGH (already on at boot)
    msp = FakeFC()
    mux = make_mux(source, msp)
    for index in range(20):
        mux._tick(10.0 + index * 0.02)
    assert len(msp.calls) == 20
    for channels in msp.frames():
        assert channels[4:8] == (1000, 1500, 1100, 2000)
        assert channels[6] == 1100        # old code hardcoded 2000 here
        assert channels[4] == 1000        # never commands ARM high


def test_auto_uses_cmd_but_keeps_pilot_aux():
    source = Source()
    msp = FakeFC()
    mux = make_mux(source, msp)
    mux._tick(10.0)
    source.channels[2] = 1400
    source.channels[7] = 2000
    mux._tick(10.02)
    assert mux._auto is True
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, roll=1520, t=10.04))
    mux._tick(10.04)                     # starts the 0.5 s blend
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, roll=1520, t=10.60))
    mux._tick(10.64)                     # after the 0.5 s ramp
    channels = msp.frames()[-1]
    assert (channels[0], channels[1], channels[2], channels[3]) == (1520, 1700, 1600, 1300)
    assert channels[4:8] == (1000, 1500, 1100, 2000)
    assert channels[8:] == tuple(source.channels[8:])  # AUX5..12 passthrough too


def test_publish_none_falls_back_to_pilot_immediately():
    source = Source()
    msp = FakeFC()
    mux = make_mux(source, msp)
    disengaged = []
    mux.on_disengage = lambda: disengaged.append(1)
    mux._tick(10.0)
    source.channels[2] = 1400
    source.channels[7] = 2000
    mux._tick(10.02)
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.02))
    mux._tick(10.04)
    assert mux.mode == "AUTO"
    mux.publish_cmd(None)
    mux._tick(10.06)                     # same tick: pilot values on the wire
    channels = msp.frames()[-1]
    assert tuple(channels[0:4]) == tuple(source.channels[0:4])
    assert channels[4:8] == tuple(source.channels[4:8])
    assert mux.mode == "PASSTHROUGH"
    assert disengaged == []              # still engaged; safety gate only paused it


def test_pilot_takeover_still_works_in_fc_mode():
    source = Source()
    msp = FakeFC()
    mux = make_mux(source, msp)
    mux._tick(10.0)
    source.channels[2] = 1400
    source.channels[7] = 2000
    mux._tick(10.02)
    mux.publish_cmd(TrackCmd(pitch=1700, yaw=1300, throttle=1600, t=10.02))
    mux._tick(10.04)
    assert mux.mode == "AUTO"
    source.channels[0] = 1800            # pilot rolls hard -> takeover
    mux._tick(10.06)
    assert mux.mode == "PASSTHROUGH"
    assert mux._auto is False
    assert msp.frames()[-1][4:8] == (1000, 1500, 1100, 2000)
