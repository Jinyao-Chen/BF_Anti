"""Regression tests for P0/P1 visual servoing pipeline fixes."""
from __future__ import annotations

import os

import numpy as np
import pytest

from guidance.p_visual_guidance import PVisualGuidance
from onboard.config import SafetyConfig, load_config
from onboard.controller import Controller
from onboard.detector.postprocess import Letterbox, parse_hailo_nms
from onboard.fc.state import FcStateEstimator
from onboard.safety import SafetyInputs, SafetySupervisor
from onboard.types import Detection, FcState
from perception.camera_geometry import CameraIntrinsics, pixel_to_camera_ray
from perception.target_kf import TargetKF
from rc_mux import RCMux


def test_controller_deduplicates_repeated_detection_timestamp():
    cfg = load_config("config/onboard.example.json")
    ctrl = Controller(cfg)
    st = FcState(altitude_m=10.0, rpy_deg=(0.0, 0.0, 0.0))
    t0 = 100.0
    det = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=t0)
    out1 = ctrl.step(det, st, now=t0, dt=0.02)
    assert out1.confirmed is False
    out2 = ctrl.step(det, st, now=t0 + 0.02, dt=0.02)
    assert out2.confirmed is False
    det2 = Detection((602.0, 320.0, 682.0, 400.0), 0.9, t=t0 + 0.04)
    out3 = ctrl.step(det2, st, now=t0 + 0.04, dt=0.02)
    assert out3.confirmed is True


def test_fc_state_attitude_interpolation_unwraps_yaw():
    est = FcStateEstimator(link=object(), cfg=object())
    est._att_hist.append((10.0, (0.0, -10.0, 350.0)))
    est._att_hist.append((10.1, (10.0, -20.0, 10.0)))
    r, p, y = est.attitude_at(10.05)
    assert abs(r - 5.0) < 1e-6
    assert abs(p - (-15.0)) < 1e-6
    assert abs(y - 0.0) < 1e-6 or abs(y - 360.0) < 1e-6


def test_rc_mux_prevents_msp_override_echo_latchup():
    class FakeLink:
        def __init__(self):
            self.sent = []
        def send_rc_16(self, ch):
            self.sent.append(tuple(ch))

    ch = [1500, 1500, 1430, 1500] + [1000, 1500, 1000, 1000] + [1500] * 8
    link = FakeLink()
    mux = RCMux(
        link,
        mode_channel=7,
        arm_channel=4,
        stick_takeover=False,
        base_throttle=1430,
        channel_source=lambda: (tuple(ch), 0.01),
    )
    ch[4] = 2000
    ch[7] = 1000
    mux._tick(10.0)
    ch[0], ch[1], ch[2], ch[3] = 1850, 1900, 1950, 1200
    ch[7] = 1500
    mux.request_auto(False)
    mux._tick(10.02)
    last = link.sent[-1]
    assert last[:4] == (1500, 1500, 1430, 1500)


def test_safety_supervisor_catches_att_mode_mismatch():
    cfg = load_config("config/onboard.example.json")
    sup = SafetySupervisor(cfg.safety)
    inp = SafetyInputs(
        now=10.0, armed=True, override_on=True, engage_on=True,
        rc_age_s=0.01, att_age_s=0.01, alt_age_s=0.01, status_age_s=0.01,
        link_age_s=0.01, altitude_m=10.0, range_m=15.0, loop_dt_s=0.02,
        target_confirmed=True, target_age_s=0.02,
        angle_mode_active=False, expected_att_mode="angle",
    )
    dec = sup.evaluate(inp)
    assert dec.allow_auto is False
    assert dec.reason == "att_mode_mismatch"


def test_p_visual_guidance_reset_clears_all_history_and_extrinsics_yaw():
    g = PVisualGuidance(camera_tilt_deg=44.0, camera_yaw_offset_deg=10.0, lift_ff=True)
    assert g.extrinsics.yaw_offset_deg == 10.0
    for i in range(5):
        g.update(
            (600.0 + i * 2, 320.0, 660.0 + i * 2, 380.0),
            (0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (10.0, -25.0, 0.0),
            10.0 + i * 0.04, 0.04, fresh=True,
        )
    assert len(g._az_hist) > 0 and len(g._range_hist) > 0
    g.reset()
    assert len(g._az_hist) == 0
    assert len(g._range_hist) == 0
    assert g._lam_dps is None
    assert g._z_int == 0.0


def test_parse_hailo_nms_5x5_disambiguation():
    params = Letterbox(src_w=640, src_h=640, dst_w=640, dst_h=640, scale_x=1.0, scale_y=1.0, pad_x=0, pad_y=0)
    arr = np.array([
        [0.1, 0.1, 0.2, 0.2, 0.90],
        [0.3, 0.3, 0.4, 0.4, 0.85],
        [0.5, 0.5, 0.6, 0.6, 0.80],
        [0.7, 0.1, 0.8, 0.2, 0.75],
        [0.1, 0.7, 0.2, 0.8, 0.70],
    ], dtype=np.float64)
    dets = parse_hailo_nms(
        arr, params, conf_threshold=0.5, iou_threshold=0.45,
        max_detections=10, t=1.0, normalized=True,
    )
    assert len(dets) == 5
    assert abs(dets[0].conf - 0.90) < 1e-6
    # The same five boxes in the tensor layout (5, M=5) must decode to the
    # same result; the old code unconditionally transposed (5, M).
    dets_t = parse_hailo_nms(
        arr.T, params, conf_threshold=0.5, iou_threshold=0.45,
        max_detections=10, t=1.0, normalized=True,
    )
    assert len(dets_t) == 5
    assert abs(dets_t[0].conf - 0.90) < 1e-6
    assert dets[0].bbox == pytest.approx(dets_t[0].bbox)


def test_target_kf_preserves_init_vars_and_recovers_after_rejects():
    kf = TargetKF(init_pos_var=2.0, init_vel_var=5.0, init_acc_var=11.0, gate_chi2=5.0)
    kf.reset(np.array([0.0, 0.0, 10.0]))
    assert np.allclose(np.diag(kf.P), [2.0] * 3 + [5.0] * 3 + [11.0] * 3)
    for i in range(5):
        kf.predict(1.0 + i * 0.05)
        ok = kf.update_position(np.array([500.0, 500.0, 500.0]))
    assert ok is True
    assert np.allclose(kf.position, [500.0, 500.0, 500.0])


def test_camera_intrinsics_optional_undistortion():
    intr_ideal = CameraIntrinsics.from_fov(1280, 720, 70.0)
    intr_dist = CameraIntrinsics(
        width=1280, height=720, fx=intr_ideal.fx, fy=intr_ideal.fy,
        cx=intr_ideal.cx, cy=intr_ideal.cy, dist_coeffs=(-0.25, 0.05, 0.0, 0.0, 0.0),
    )
    r_ideal = pixel_to_camera_ray((100.0, 100.0), intr_ideal)
    r_undist = pixel_to_camera_ray((100.0, 100.0), intr_dist)
    assert abs(np.linalg.norm(r_undist) - 1.0) < 1e-9
    assert not np.allclose(r_ideal, r_undist)


def test_real_profiles_have_safe_throttle_margin():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("real_pn8_rate", "real_pn8_angle", "real_conservative"):
        cfg = load_config(
            os.path.join(root, "config", "onboard.example.json"),
            os.path.join(root, "config", "profiles", name + ".json"),
        )
        p = cfg.guidance.params
        assert p["hover_pwm"] <= 1500.0
        assert p["hover_pwm"] - p["thr_min_pwm"] >= 200.0
        assert p["lift_ff"] is True


def test_lift_ff_uses_measured_total_tilt():
    g = PVisualGuidance(
        camera_tilt_deg=40.0, lift_ff=True, hover_pwm=1430.0,
        thr_min_pwm=1000.0, thr_max_pwm=2000.0,
    )
    level = g._lift_ff_throttle(1430.0, 0.0, 0.0)
    assert level == pytest.approx(1430.0)
    tilted = g._lift_ff_throttle(1430.0, 0.0, 60.0)
    # 1000 + (1430-1000)/sqrt(cos 60) = 1608.1
    assert tilted == pytest.approx(1000.0 + 430.0 / np.sqrt(0.5), rel=1e-6)
    # The old formula ignored roll entirely and would return 1430 here.
    assert tilted > level + 150.0
    g.lift_ff = False
    assert g._lift_ff_throttle(1180.0, 0.0, 60.0) == pytest.approx(1180.0)


def test_battery_debounce_denies_but_does_not_latch_early():
    sup = SafetySupervisor(
        SafetyConfig(min_battery_v=13.6, battery_debounce_s=1.0), True, True,
        battery_debounce_s=1.0,
    )
    base = dict(
        now=10.0, armed=True, override_on=True, engage_on=True,
        rc_age_s=0.01, att_age_s=0.01, alt_age_s=0.01, status_age_s=0.01,
        link_age_s=0.01, altitude_m=10.0, range_m=12.0, loop_dt_s=0.02,
        target_confirmed=True, target_age_s=0.02, battery_v=13.0,
    )
    dec = sup.evaluate(SafetyInputs(**base))
    assert not dec.allow_auto
    assert not dec.critical
    assert dec.reason == "battery_low_pending"
    dec = sup.evaluate(SafetyInputs(**{**base, "now": 11.1}))
    assert dec.critical and dec.reason == "CRITICAL:battery_low"


def test_rc_mux_override_fallback_never_drops_below_hover_floor():
    class FakeLink:
        def __init__(self):
            self.sent = []
            self.last = [1500, 1500, 1000, 1500] + [1500] * 12
        def send_rc_16(self, ch):
            self.last = [int(v) for v in ch]
            self.sent.append(tuple(self.last))

    link = FakeLink()
    mux = RCMux(
        link, mode_channel=7, arm_channel=4, stick_takeover=False,
        base_throttle=1430, channel_source=lambda: (tuple(link.last), 0.01),
    )
    # Manual phase: pilot throttle is 1000 (ground).
    link.last[7] = 1000
    mux._tick(10.0)
    # Pilot flips CH8 high while still on the ground; never echo 1000 back.
    link.last[7] = 1500
    mux.request_auto(False)
    mux._tick(10.02)
    assert link.sent[-1][2] == 1430


def test_fc_poll_staggers_aux_queries_instead_of_six_at_once():
    from onboard.config import FcConfig

    class CountingLink:
        def __init__(self):
            self.calls = 0
            self.generation = 1
            self.errors = 0
            self.last_rx_t = 0.0

        def query(self, cmd, is_v2=False, timeout=None):
            self.calls += 1
            return None

    link = CountingLink()
    est = FcStateEstimator(link, FcConfig(telemetry_hz=50.0))
    counts = []
    for _ in range(40):
        before = link.calls
        est._poll_once()
        counts.append(link.calls - before)
    # Attitude every tick + staggered secondary queries; a bad LCM tick can
    # overlap three, but the old six-query pileup is gone.
    assert max(counts) <= 4
    assert counts[0] <= 2
