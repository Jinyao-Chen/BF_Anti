import numpy as np

from guidance.intercept_guidance import GuidancePhase, InterceptGuidance
from perception.target_estimator import TargetState


def make_target(position, velocity=(0.0, 0.0, 0.0)):
    return TargetState(
        position_enu=np.array(position, dtype=float),
        velocity_enu=np.array(velocity, dtype=float),
        range_m=float(np.linalg.norm(position)),
        az_deg=0.0,
        el_deg=0.0,
        timestamp=0.0,
    )


def test_approach_for_far_target_ahead_same_altitude():
    g = InterceptGuidance(camera_tilt_deg=44.0)
    cmd = g.update(make_target((0.0, 20.0, 0.0)), np.zeros(3), np.zeros(3), timestamp=1.0)
    assert cmd.valid
    assert cmd.phase == GuidancePhase.APPROACH
    assert abs(cmd.yaw_cmd_deg) < 1e-6
    assert abs(cmd.pitch_cmd_deg + 44.0) < 1e-6


def test_terminal_for_close_target():
    g = InterceptGuidance(camera_tilt_deg=44.0)
    cmd = g.update(make_target((0.0, 1.0, 0.0)), np.zeros(3), np.zeros(3), timestamp=1.0)
    assert cmd.valid
    assert cmd.phase == GuidancePhase.TERMINAL
    assert abs(cmd.pitch_cmd_deg + 44.0) < 1e-6


def test_track_for_mid_range_target():
    g = InterceptGuidance(camera_tilt_deg=44.0, approach_range_m=8.0, terminal_range_m=2.0)
    cmd = g.update(make_target((0.0, 5.0, 0.0)), np.zeros(3), np.zeros(3), timestamp=1.0)
    assert cmd.valid
    assert cmd.phase == GuidancePhase.TRACK


def test_yaw_points_at_target_to_the_right():
    g = InterceptGuidance(camera_tilt_deg=44.0)
    cmd = g.update(make_target((10.0, 10.0, 0.0)), np.zeros(3), np.zeros(3), timestamp=1.0)
    assert cmd.valid
    assert abs(cmd.yaw_cmd_deg - 45.0) < 1e-6
    assert abs(cmd.pitch_cmd_deg + 44.0) < 1e-6
