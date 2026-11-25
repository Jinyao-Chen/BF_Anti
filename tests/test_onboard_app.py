"""Onboard app wiring: fakes for FC/camera/detector/mux/video."""
import time
from dataclasses import replace

from onboard.app import OnboardApp
from onboard.config import PowerConfig, load_config
from onboard.safety import SafetyDecision
from onboard.types import Detection, FcState


class FakeState:
    def __init__(self, state):
        self._state = state
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def snapshot(self, now=None):
        return self._state

    def rc_snapshot(self):
        return (self._state.rc, 0.0)


class FakeDetector:
    def __init__(self, detection=None):
        self.detection = detection
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def latest(self, max_age_s, now=None):
        return self.detection

    def healthy(self, now):
        return True


class FakeCamera:
    def __init__(self):
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def latest(self, max_age_s, now=None):
        return None, 0.0


class FakeMux:
    def __init__(self):
        self.cmds = []
        self.on_engage = None
        self.on_disengage = None
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def publish_cmd(self, cmd):
        self.cmds.append(cmd)


class FakeVideo:
    def __init__(self):
        self.submits = 0

    def start(self):
        pass

    def stop(self):
        pass

    def submit(self, *args):
        self.submits += 1


def fresh_state(now):
    return FcState(
        altitude_m=10.0,
        vario_mps=0.0,
        rpy_deg=(0.0, 0.0, 0.0),
        rc=(1500, 1500, 1500, 1500, 2000, 2000, 2000, 2000) + (1500,) * 8,
        armed=True,
        att_t=now,
        alt_t=now,
        rc_t=now,
        status_t=now,
        link_t=now,
    )


def build_app(detection=None, state=None, cfg=None):
    cfg = cfg or load_config("config/onboard.example.json")
    now = time.monotonic()
    st = state or fresh_state(now)
    fake_state = FakeState(st)
    fake_det = FakeDetector(detection)
    fake_mux = FakeMux()
    fake_video = FakeVideo()
    app = OnboardApp(
        cfg, link=object(), state=fake_state, detector=fake_det,
        camera=FakeCamera(), mux=fake_mux, video=fake_video,
    )
    return app, fake_mux, fake_video, now


def test_step_once_publishes_none_without_target():
    app, mux, video, now = build_app()
    out, decision = app.step_once(now)
    assert out.channels is None
    assert mux.cmds and mux.cmds[-1] is None
    assert isinstance(decision, SafetyDecision)
    assert video.submits == 1


def test_engaged_with_confirmed_target_publishes_track_cmd():
    now = time.monotonic()
    det = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now)
    app, mux, _, _ = build_app(det, state=fresh_state(now))
    app._engaged = True
    app.step_once(now)
    app.detector.detection = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now + 0.02)
    out, decision = app.step_once(now + 0.02)
    assert out.confirmed and out.channels is not None
    assert decision.allow_auto, decision.reason
    cmd = mux.cmds[-1]
    assert cmd is not None
    assert 1000 <= cmd.throttle <= 2000
    assert 1000 <= cmd.roll <= 2000


def test_rc_shutdown_channel_sets_poweroff_request_once():
    now = time.monotonic()
    cfg = replace(
        load_config("config/onboard.example.json"),
        power=PowerConfig(aux_channel=6, hold_s=0.05, max_rc_age_s=0.3),
    )
    st = fresh_state(now)
    st.armed = False
    rc = list(st.rc)
    rc[2] = 1000
    rc[6] = 1000
    st.rc = tuple(rc)
    app, _, _, _ = build_app(state=st, cfg=cfg)

    # First sample is low: arm the edge detector.
    app.step_once(now)
    assert app._poweroff_requested is False

    rc[6] = 2000
    st.rc = tuple(rc)
    st.rc_t = now + 0.01
    st.status_t = now + 0.01
    app.step_once(now + 0.01)
    assert app._poweroff_requested is False
    app.step_once(now + 0.10)
    assert app._poweroff_requested is True
    # run() consumes the flag before issuing the command; the guard itself is
    # one-shot until the runtime retries or the switch cycles.
    app._poweroff_requested = False
    app.step_once(now + 0.20)
    assert app._poweroff_requested is False


def test_stale_rc_immediately_closes_gate():
    now = time.monotonic()
    det = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now)
    st = fresh_state(now)
    st.rc_t = now - 2.0
    app, mux, _, _ = build_app(det, state=st)
    app._engaged = True
    app.step_once(now)
    out, decision = app.step_once(now + 0.02)
    assert not decision.allow_auto
    assert decision.critical
    assert mux.cmds[-1] is None


def test_fc_angle_mode_with_rate_profile_is_denied():
    now = time.monotonic()
    st = fresh_state(now)
    st.flight_mode_flags = 1  # Betaflight ANGLE_MODE is bit 0
    app, _, _, _ = build_app(state=st)
    app._engaged = True
    _, decision = app.step_once(now)
    assert not decision.allow_auto
    assert decision.reason == "att_mode_mismatch"


def test_request_stop_still_publishes_one_last_passthrough():
    app, mux, _, _ = build_app()
    app.request_stop("sigterm")
    app.mux = FakeMux()
    app.stop("sigterm")
    assert app.mux.cmds == [None]
    # Idempotent: a second stop must not sleep/publish again.
    app.stop("sigterm")
    assert app.mux.cmds == [None]
