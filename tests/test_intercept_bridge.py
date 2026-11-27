"""Gazebo 拦截桥单元测试：坐标/姿态换算、44° 相机链路、RC 映射。"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guidance.intercept_guidance import GuidancePhase  # noqa: E402
from perception.camera_geometry import (  # noqa: E402
    CameraExtrinsics,
    CameraIntrinsics,
    desired_body_pitch_for_elevation,
    rotation_earth_from_body,
)
from sim.gazebo_target_bridge import (  # noqa: E402
    InterceptBridge,
    PoseState,
    gazebo_xyz_to_enu,
    project_target_to_bbox,
    quat_to_rotation_enu_body,
    rpy_from_rotation_enu_body,
)


def _pose(own_enu, target_enu, rpy=(0.0, 0.0, 0.0), t=1.0, vel=None) -> PoseState:
    return PoseState(
        own_position_enu=np.asarray(own_enu, dtype=float),
        own_velocity_enu=np.zeros(3) if vel is None else np.asarray(vel, dtype=float),
        own_rpy_deg=tuple(float(v) for v in rpy),
        target_position_enu=np.asarray(target_enu, dtype=float),
        timestamp=t,
    )


def test_gazebo_xyz_to_enu_matches_fc_convention():
    # 世界坐标 (30, 0, 10) 的靶机在 ENU 里应是 北=30, 东=0, 上=10
    assert gazebo_xyz_to_enu(30.0, 0.0, 10.0) == pytest.approx([0.0, 30.0, 10.0])
    # 2026-09-28 用真实相机画面标定：Gazebo 的 +y 是**西**，不是东。
    # 旧实现 det=-1（反射阵）导致方位镜像；14 组样本实测 Δu 从 +260 px 降到 +2.3 px。
    assert gazebo_xyz_to_enu(0.0, 5.0, 1.0) == pytest.approx([-5.0, 0.0, 1.0])
    assert gazebo_xyz_to_enu(0.0, -5.0, 1.0) == pytest.approx([5.0, 0.0, 1.0])


def test_quat_to_rotation_is_proper_rotation():
    """姿态换算必须是真旋转 (det=+1)。旧实现 det=-1：俯仰对、方位镜像。"""
    import math as _m
    for yaw, pitch, roll in [(0, 0, 0), (25, 0, 0), (0, 15, 0), (0, 0, 20), (-30, 10, -12)]:
        cy, sy = _m.cos(_m.radians(yaw) / 2), _m.sin(_m.radians(yaw) / 2)
        cp, sp = _m.cos(_m.radians(pitch) / 2), _m.sin(_m.radians(pitch) / 2)
        cr, sr = _m.cos(_m.radians(roll) / 2), _m.sin(_m.radians(roll) / 2)
        q = (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
             cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)
        r = quat_to_rotation_enu_body(*q)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-9)
        assert np.abs(r.T @ r - np.eye(3)).max() < 1e-9


def test_rotation_rpy_round_trip():
    for yaw, pitch, roll in [(0, 0, 0), (30, 0, 0), (0, 20, 0), (0, 0, 15), (-40, 10, -25)]:
        r = rotation_earth_from_body(yaw, pitch, roll)
        got = rpy_from_rotation_enu_body(r)
        assert got[0] == pytest.approx(roll, abs=1e-6)
        assert got[1] == pytest.approx(pitch, abs=1e-6)
        assert got[2] == pytest.approx(yaw, abs=1e-6)


def test_rotation_earth_from_body_is_orthonormal():
    r = rotation_earth_from_body(35.0, -12.0, 20.0)
    assert r.T @ r == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-9)


def test_body_pitch_aims_optical_axis_at_same_altitude_target():
    # 同高正前方目标：elevation=0 -> 机体俯仰 -44° 让光轴水平
    assert desired_body_pitch_for_elevation(0.0, tilt_deg=44.0) == pytest.approx(-44.0)
    # 目标在仰角 +10° -> 机体 -34°
    assert desired_body_pitch_for_elevation(10.0, tilt_deg=44.0) == pytest.approx(-34.0)


def test_projected_bbox_centers_on_optical_axis_target():
    intrinsics = CameraIntrinsics.from_fov(1280, 720, hfov_deg=70.0)
    extrinsics = CameraExtrinsics(tilt_deg=44.0)
    # 目标正好落在光轴上：机体水平时，光轴指向 ENU 的 (东0, 北cos44, 上sin44)
    own = [0.0, 0.0, 10.0]
    los = np.array([0.0, 6.0 * np.cos(np.deg2rad(44.0)), 6.0 * np.sin(np.deg2rad(44.0))])
    target = np.asarray(own) + los
    pose = _pose(own, target, rpy=(0.0, 0.0, 0.0))
    bbox = project_target_to_bbox(pose, intrinsics, extrinsics, target_width_m=1.0)
    assert bbox is not None
    u = (bbox[0] + bbox[2]) * 0.5
    v = (bbox[1] + bbox[3]) * 0.5
    assert u == pytest.approx(intrinsics.cx, abs=1.0)
    assert v == pytest.approx(intrinsics.cy, abs=1.0)
    # bbox 宽度应反映 6m 距离
    assert (bbox[2] - bbox[0]) == pytest.approx(intrinsics.fx / 6.0, rel=0.05)


def test_same_altitude_target_centers_with_pitch_minus_tilt():
    """44° 相机看同高正前方目标：机体压 -44° 时光轴正对目标（像素居中）。"""
    intrinsics = CameraIntrinsics.from_fov(1280, 720, hfov_deg=70.0)
    extrinsics = CameraExtrinsics(tilt_deg=44.0)
    own = [0.0, 0.0, 10.0]
    target = [0.0, 6.0, 10.0]
    pose = _pose(own, target, rpy=(0.0, -44.0, 0.0))
    bbox = project_target_to_bbox(pose, intrinsics, extrinsics, target_width_m=1.0)
    assert bbox is not None
    u = (bbox[0] + bbox[2]) * 0.5
    v = (bbox[1] + bbox[3]) * 0.5
    assert u == pytest.approx(intrinsics.cx, abs=1.5)
    assert v == pytest.approx(intrinsics.cy, abs=1.5)


def test_camera_chain_recovers_target_position_and_terminal_pitch():
    bridge = InterceptBridge(camera_tilt_deg=44.0, use_camera=True, target_width_m=1.0)
    own = [0.0, 0.0, 10.0]
    target = [0.0, 1.5, 10.0]           # 正北 1.5m，同高 -> TERMINAL
    pose = _pose(own, target, rpy=(0.0, -44.0, 0.0), t=0.0)
    cmd = bridge.guidance.update(bridge.target_state(pose), own, np.zeros(3), 0.0)
    assert cmd.phase == GuidancePhase.TERMINAL
    # 同高目标 elevation=0 -> 终端机体俯仰 = -44°
    assert cmd.pitch_cmd_deg == pytest.approx(-44.0, abs=0.5)

    # 视觉链路估计出的目标位置要落在真值附近
    est = bridge.target_state(pose)
    assert est is not None
    assert est.position_enu == pytest.approx(target, abs=0.15)


@pytest.mark.parametrize("range_m,phase", [(20.0, GuidancePhase.APPROACH), (5.0, GuidancePhase.TRACK)])
def test_phase_selection_matches_range(range_m, phase):
    bridge = InterceptBridge(camera_tilt_deg=44.0, use_camera=False)
    own = [0.0, 0.0, 10.0]
    target = [0.0, range_m, 10.0]
    cmd = bridge.guidance.update(bridge.target_state(_pose(own, target)), own, np.zeros(3), 0.0)
    assert cmd.phase == phase
    assert cmd.valid


def test_rc_frame_mapping_scales_pitch_to_rc_range():
    bridge = InterceptBridge(camera_tilt_deg=44.0, use_camera=False, max_angle_deg=45.0)
    own = [0.0, 0.0, 10.0]
    target = [0.0, 1.5, 10.0]
    cmd, frame = bridge.step(_pose(own, target))
    assert cmd.phase == GuidancePhase.TERMINAL
    # Betaflight pitch high = nose down: 1500 + (44°) * (500/45) ≈ 1989.
    assert frame["pitch"] == pytest.approx(1989, abs=3)
    assert 1000 <= frame["throttle"] <= 2000
    assert frame["roll"] == 1500
    assert frame["yaw"] == 1500


def test_rc_frame_clamped_and_searched_when_no_target():
    from sim.gazebo_target_bridge import _rc_frame_to_udp_frame

    bridge = InterceptBridge(use_camera=True)
    frame = _rc_frame_to_udp_frame({"roll": 900, "pitch": 2100, "throttle": 1500, "yaw": 500})
    assert frame[0] == 1000 and frame[1] == 2000 and frame[3] == 1000
    assert len(frame) == 16 and frame[4] == 2000

    # 相机看不到目标 -> SEARCH，油门保持悬停，指令无效
    own = [0.0, 0.0, 10.0]
    target = [0.0, -6.0, 10.0]  # 目标在正后方，44° 前视相机看不到
    cmd, _ = bridge.step(_pose(own, target, rpy=(0.0, -44.0, 0.0)))
    assert cmd.phase == GuidancePhase.SEARCH
    assert cmd.valid is False
