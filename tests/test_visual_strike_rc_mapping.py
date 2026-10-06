"""RC mapping / teardown regression tests for the visual strike SITL script."""

import numpy as np

from guidance.visual_strike_guidance import StrikeCommand, StrikePhase
from sim.run_intercept_mission import RcState
from sim.run_visual_strike import strike_to_rc
from sim.run_visual_strike_yolo import KalmanFilterUV


def test_strike_to_rc_inverts_pitch_for_betaflight_angle_mode():
    """制导层抬头为正；Betaflight ANGLE 通道正方向是低头，必须取反。"""
    nose_down = StrikeCommand(
        phase=StrikePhase.APPROACH, pitch_cmd_deg=-30.0, throttle_cmd=0.5
    )
    nose_up = StrikeCommand(
        phase=StrikePhase.APPROACH, pitch_cmd_deg=30.0, throttle_cmd=0.5
    )
    assert strike_to_rc(nose_down, max_angle_deg=60.0)[1] > 1500
    assert strike_to_rc(nose_up, max_angle_deg=60.0)[1] < 1500


def test_rc_state_idle_disarms_and_drops_throttle():
    rc = RcState()
    rc.set_aux1(2000)
    rc.set_throttle(1600)
    rc.idle()
    channels = rc.snapshot()
    assert channels[2] == 1000  # throttle low
    assert channels[4] == 1000  # AUX1 arm low


def test_kf_uv_rejects_large_jump():
    kf = KalmanFilterUV()
    kf.reset(np.array([100.0, 200.0]))
    kf.update(np.array([102.0, 198.0]), 0.05)
    pos, accepted = kf.update(np.array([260.0, 50.0]), 0.05)
    assert kf.last_rejected is True
    assert accepted is False
    # 预测位置应仍在上一帧附近，而不是跳到假框位置。
    assert abs(pos[0] - 101.5) < 5.0
    assert abs(pos[1] - 198.5) < 5.0
