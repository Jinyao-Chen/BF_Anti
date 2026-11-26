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


def test_repeated_stale_detection_is_not_fresh():
    c = Controller(make_cfg())
    st = make_state()
    now = time.monotonic()
    det = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now)
    c.step(det, st, now, 0.02)
    # Reusing the same detection timestamp at 50 Hz must not look like a new
    # measurement: the tracker coasts instead of confirming a false lock.
    out = c.step(det, st, now + 0.02, 0.02)
    assert not out.confirmed
    assert out.reason == "coast"
    det2 = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now + 0.04)
    out = c.step(det2, st, now + 0.04, 0.02)
    assert out.confirmed


def test_confirmed_detection_returns_only_four_channels():
    c = Controller(make_cfg())
    st = make_state()
    now = time.monotonic()
    det = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now)
    out = c.step(det, st, now, 0.02)
    assert out.channels is None  # confirm_frames=2
    det2 = Detection((601.0, 320.0, 681.0, 400.0), 0.9, t=now + 0.02)
    out = c.step(det2, st, now + 0.02, 0.02)
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
        di = Detection((600.0 + i, 320.0, 680.0 + i, 400.0), 0.9, t=now + 0.02 * (i + 1))
        out = c.step(di, st, now + 0.02 * (i + 1), 0.02)
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
        di = Detection(det.bbox, 0.99, t=now + 0.02 * i)
        out = c.step(di, st, now + 0.02 * i, 0.02)
        if out.mode == Mode.HIT:
            break
    assert out is not None
    if out.mode == Mode.HIT:
        assert out.channels is None

from dataclasses import replace

from guidance.p_visual_guidance import PStrikeCommand


def test_detection_age_prediction_shifts_box_forward():
    c = Controller(make_cfg())
    now = time.monotonic()
    # Two real measurements establish +100 px/s horizontal velocity.
    c.tracker.update((500.0, 300.0, 540.0, 340.0), dt=0.05, timestamp=now)
    c.tracker.update((510.0, 300.0, 550.0, 340.0), dt=0.05, timestamp=now + 0.05)
    bbox = (510.0, 300.0, 550.0, 340.0)
    shifted = c._predict_bbox(bbox, 0.10)
    assert shifted[0] > bbox[0]
    assert shifted[2] > bbox[2]
    # The box size is not changed by the shift.
    assert (shifted[2] - shifted[0]) == pytest.approx(bbox[2] - bbox[0])


def test_rate_mode_maps_attitude_error_to_angular_rate():
    cfg = make_cfg()
    cfg = replace(cfg, control=replace(cfg.control, att_mode="rate",
                                        att_rate_gain=6.0, rate_max_dps=250.0,
                                        rate_full_dps=600.0,
                                        rc_slew_rate_per_s=0.0))
    c = Controller(cfg)
    cmd = PStrikeCommand(roll_cmd_deg=0.0, pitch_cmd_deg=10.0,
                         yaw_rate_dps=0.0, throttle_pwm=1500.0)
    st = replace(make_state(), rpy_deg=(0.0, 0.0, 0.0))
    ch = c._rpyt(cmd, st)
    # 6 * 10 = 60 dps -> 50 RC above centre, and pitch is inverted for BF.
    assert ch[1] == pytest.approx(1450, abs=3)
    assert len(ch) == 4


def test_angle_mode_still_uses_angle_scaling():
    cfg = make_cfg()
    cfg = replace(cfg, control=replace(cfg.control, att_mode="angle",
                                        angle_limit_deg=60.0, rc_rate_limit=1000,
                                        rc_slew_rate_per_s=0.0))
    c = Controller(cfg)
    cmd = PStrikeCommand(roll_cmd_deg=0.0, pitch_cmd_deg=10.0,
                         yaw_rate_dps=0.0, throttle_pwm=1500.0)
    st = replace(make_state(), rpy_deg=(0.0, 0.0, 0.0))
    ch = c._rpyt(cmd, st)
    assert ch[1] == pytest.approx(1500 - 10 * 500 / 60, abs=3)
