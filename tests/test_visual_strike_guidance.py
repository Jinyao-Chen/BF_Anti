import math

import numpy as np

from guidance.visual_strike_guidance import (
    ImageTrackReference,
    InterceptTrajectory,
    StrikePhase,
    VisualStrikeGuidance,
)


def test_image_reference_starts_at_measurement_and_converges_smoothly():
    ref = ImageTrackReference(1280, 720, max_vel_px_s=600, max_accel_px_s2=2000, max_jerk_px_s3=10000)
    ref.reset(100.0, 200.0)
    ref.set_goal(640.0, 360.0)

    max_vel = 0.0
    max_accel = 0.0
    first = ref.update(0.02, 100.0, 200.0)
    assert abs(first.position[0] - 100.0) < 1.0
    assert abs(first.position[1] - 200.0) < 1.0

    for _ in range(300):
        state = ref.update(0.02, 100.0, 200.0)
        max_vel = max(max_vel, float(np.max(np.abs(state.velocity[:2]))))
        max_accel = max(max_accel, float(np.max(np.abs(state.acceleration[:2]))))

    assert max_vel <= 600.01
    assert max_accel <= 2000.01
    assert abs(ref.state.position[0] - 640.0) < 1.0
    assert abs(ref.state.position[1] - 360.0) < 1.0


def test_image_reference_step_goal_does_not_jump_reference():
    ref = ImageTrackReference(1280, 720, max_vel_px_s=900, max_accel_px_s2=4000, max_jerk_px_s3=30000)
    ref.reset(100.0, 100.0)
    ref.set_goal(1200.0, 700.0)
    state = ref.update(0.02, 100.0, 100.0)
    assert state.position[0] < 110.0
    assert state.position[1] < 110.0


def test_intercept_trajectory_leads_crossing_target():
    traj = InterceptTrajectory(nav_constant=3.5, max_lead_time_s=2.0)
    own = np.zeros(3)
    target_pos = np.array([0.0, 20.0, 0.0])
    target_vel = np.array([5.0, 0.0, 0.0])  # 向东横穿
    state = traj.update(target_pos, target_vel, own, np.zeros(3))
    assert state.aim_point_enu[0] > target_pos[0]
    assert state.range_m > 0.0
    assert state.t_go_s > 0.0
    assert np.all(np.isfinite(state.accel_pn_enu))


def test_visual_guidance_pitch_equals_elevation_minus_camera_tilt():
    g = VisualStrikeGuidance(camera_tilt_deg=44.0)
    el = math.radians(44.0)
    target = np.array([0.0, 10.0 * math.cos(el), 10.0 * math.sin(el)])
    own = np.zeros(3)
    pix = g.project_enu_to_pixel(target, own, (0.0, 0.0, 0.0))
    assert pix is not None
    bbox = (pix[0] - 20.0, pix[1] - 20.0, pix[0] + 20.0, pix[1] + 20.0)
    cmd = g.update(target, np.zeros(3), own, np.zeros(3), (0.0, 0.0, 0.0), timestamp=1.0,
                   bbox_xyxy=bbox)
    assert cmd.valid
    assert abs(cmd.pitch_cmd_deg) < 5.0  # 目标仰角 44°，相机上仰 44° -> 机体俯仰约 0
    assert abs(cmd.u_ref_px - pix[0]) < 2.0
    assert abs(cmd.v_ref_px - pix[1]) < 2.0


def test_visual_guidance_terminal_hit():
    g = VisualStrikeGuidance(camera_tilt_deg=44.0, hit_radius_m=0.8)
    target = np.array([0.0, 0.5, 0.0])
    pix = g.project_enu_to_pixel(target, np.zeros(3), (0.0, 0.0, 0.0))
    assert pix is not None
    bbox = (pix[0] - 10.0, pix[1] - 10.0, pix[0] + 10.0, pix[1] + 10.0)
    cmd = g.update(target, np.zeros(3), np.zeros(3), np.zeros(3), (0.0, 0.0, 0.0),
                   timestamp=1.0, bbox_xyxy=bbox)
    assert cmd.phase == StrikePhase.TERMINAL
    assert cmd.hit
    assert math.isfinite(cmd.throttle_cmd)


def test_visual_guidance_invalid_target_returns_search():
    g = VisualStrikeGuidance()
    cmd = g.update(None, None, np.zeros(3), np.zeros(3), (0.0, 0.0, 0.0), timestamp=1.0)
    assert cmd.phase == StrikePhase.SEARCH
    assert not cmd.valid
