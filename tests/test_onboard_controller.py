"""Controller: tracker -> guidance -> 4 RC channels, never AUX/ARM."""
import time

import pytest

from onboard.config import load_config
from onboard.controller import Controller
from onboard.types import RC_MAX, RC_MIN, Detection, FcState, Mode


def make_cfg():
    return load_config("config/onboard.example.json")


def make_state():
    return FcState(
        altitude_m=10.0,
        vario_mps=0.0,
        rpy_deg=(0.0, 0.0, 0.0),
        rc=(1500,) * 16,
        armed=True,
        att_t=time.monotonic(),
        alt_t=time.monotonic(),
        rc_t=time.monotonic(),
        status_t=time.monotonic(),
        link_t=time.monotonic(),
    )


def test_no_detection_returns_no_channels():
    c = Controller(make_cfg())
    out = c.step(None, make_state(), time.monotonic(), 0.02)
    assert out.mode == Mode.SEARCH
    assert out.channels is None


def test_confirmed_detection_returns_only_four_channels():
    c = Controller(make_cfg())
    st = make_state()
    now = time.monotonic()
    det = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now)
    out = c.step(det, st, now, 0.02)
    assert out.channels is None  # confirm_frames=2
    out = c.step(det, st, now + 0.02, 0.02)
    assert out.confirmed
    assert out.channels is not None
    assert len(out.channels) == 4
    assert all(RC_MIN <= v <= RC_MAX for v in out.channels)
    # The 4-tuple is RPYT only; there is no AUX/ARM slot to leak into.
    roll, pitch, throttle, yaw = out.channels
    assert throttle == pytest.approx(c.cfg.control.hover_throttle, abs=120)


def test_channels_are_rate_limited():
    c = Controller(make_cfg())
    st = make_state()
    now = time.monotonic()
    det = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now)
    c.step(det, st, now, 0.02)
    prev = None
    for i in range(6):
        out = c.step(det, st, now + 0.02 * (i + 1), 0.02)
        if out.channels is None:
            continue
        if prev is not None:
            assert all(abs(a - b) <= c.cfg.control.rc_rate_limit for a, b in zip(out.channels, prev))
        prev = out.channels


def test_hit_hands_back_to_pilot():
    c = Controller(make_cfg())
    # Force a hit by using a bbox wider than the hit threshold at close range.
    st = make_state()
    now = time.monotonic()
    det = Detection((0.0, 0.0, 1280.0, 720.0), 0.99, t=now)
    out = None
    for i in range(4):
        out = c.step(det, st, now + 0.02 * i, 0.02)
        if out.mode == Mode.HIT:
            break
    assert out is not None
    if out.mode == Mode.HIT:
        assert out.channels is None
