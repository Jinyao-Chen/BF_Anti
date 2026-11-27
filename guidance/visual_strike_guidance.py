"""44° 上仰相机下的纯视觉拦截/打击制导原型。

三层结构
--------
1. **Track trajectory（参考轨迹）**：`InterceptTrajectory` 由目标 ENU 状态
   计算拦截瞄准点、视线角、接近速度、t_go 和 PN 横向加速度。这是“把目标
   放在未来命中点”的外层轨迹。
2. **Image reference governor（画面参考逐步插值）**：`ImageTrackReference`
   用 PX4 风格 jerk-limited 参考模型，把画面内参考点从“当前检测框中心”
   一步步插值到投影出来的瞄准点。误差是 `参考点 - 当前检测`，而不是
   `最终中心 - 当前检测`，所以不会在捕获/切换阶段产生阶跃误差。
3. **Visual servo + thrust（视觉反馈 + 推力）**：由图误差生成偏航/俯仰
   修正（或角速率），并沿瞄准方向生成接近加速度、推力归一化油门。

为什么不是“像素误差 -> 姿态”直接比例
--------------------------------------
44° 相机光轴比机头高 44°。要让目标在画面中心，机体俯仰约为
``目标仰角 - 44°``；这个姿态本身就会让推力产生前向分量。控制器必须把
“画面中心保持”和“接近/命中”统一起来：参考轨迹负责未来命中点，画面
参考负责平滑地跟踪这个点，视觉误差只做修正。直接比例控制在捕获、
丢失重捕和终端都会产生阶跃误差，是参数难调的主要原因之一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from modeling.reference_trajectory import PositionReference3D, ReferenceState
from perception.camera_geometry import (
    CameraExtrinsics,
    CameraIntrinsics,
    earth_los_to_az_el,
    pixel_to_earth_los,
    rotation_earth_from_body,
)

G = 9.80665


class StrikePhase(str, Enum):
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    TRACK = "TRACK"
    TERMINAL = "TERMINAL"
    ABORT = "ABORT"


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _clamp(value: float, lo: float, hi: float) -> float:
    v = float(value)
    lo_f, hi_f = float(lo), float(hi)
    if not np.isfinite(v):
        return 0.0 if (lo_f <= 0.0 <= hi_f) else lo_f
    return max(lo_f, min(hi_f, v))


@dataclass
class StrikeCommand:
    """视觉打击制导输出。"""

    phase: StrikePhase
    valid: bool = False
    hit: bool = False

    # 画面参考（像素）
    u_ref_px: float = float("nan")
    v_ref_px: float = float("nan")
    u_meas_px: float = float("nan")
    v_meas_px: float = float("nan")

    # 期望视线/姿态
    az_ref_deg: float = 0.0
    el_ref_deg: float = 0.0
    pitch_cmd_deg: float = 0.0
    roll_cmd_deg: float = 0.0
    yaw_cmd_deg: float = 0.0
    pitch_rate_dps: float = 0.0
    yaw_rate_dps: float = 0.0

    # 推力
    throttle_cmd: float = 0.5

    # 目标/命中信息
    target_range_m: float = float("inf")
    closing_speed_mps: float = 0.0
    t_go_s: float = float("inf")
    aim_point_enu: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel_enu: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class InterceptTrajectoryState:
    aim_point_enu: np.ndarray
    los_az_deg: float
    los_el_deg: float
    range_m: float
    closing_speed_mps: float
    t_go_s: float
    accel_pn_enu: np.ndarray


class InterceptTrajectory:
    """外层拦截轨迹：瞄准点、视线角、PN 横向加速度。"""

    def __init__(
        self,
        nav_constant: float = 3.5,
        max_lead_time_s: float = 2.0,
        min_closing_speed_mps: float = 0.8,
        max_speed_mps: float = 12.0,
    ) -> None:
        self.nav_constant = float(nav_constant)
        self.max_lead_time_s = float(max_lead_time_s)
        self.min_closing_speed_mps = float(min_closing_speed_mps)
        self.max_speed_mps = float(max_speed_mps)

    def update(
        self,
        target_position_enu: np.ndarray,
        target_velocity_enu: np.ndarray,
        own_position_enu: np.ndarray,
        own_velocity_enu: np.ndarray,
    ) -> InterceptTrajectoryState:
        target_pos = np.asarray(target_position_enu, dtype=float)
        target_vel = np.asarray(target_velocity_enu, dtype=float)
        own_pos = np.asarray(own_position_enu, dtype=float)
        own_vel = np.asarray(own_velocity_enu, dtype=float)

        rel = target_pos - own_pos
        rng = float(np.linalg.norm(rel))
        if rng < 1e-6:
            return InterceptTrajectoryState(
                aim_point_enu=target_pos.copy(),
                los_az_deg=0.0,
                los_el_deg=0.0,
                range_m=0.0,
                closing_speed_mps=0.0,
                t_go_s=0.0,
                accel_pn_enu=np.zeros(3),
            )

        los = rel / rng
        v_rel = target_vel - own_vel  # 目标相对本机 (dr/dt)
        closing = float(-np.dot(los, v_rel))
        closing_eff = max(closing, self.min_closing_speed_mps)
        t_go = min(rng / max(closing_eff, 1e-6), self.max_lead_time_s)

        # 未来瞄准点：目标匀速外推，终端 t_go 自然收敛到 0。
        aim = target_pos + target_vel * t_go
        aim_rel = aim - own_pos
        aim_range = float(np.linalg.norm(aim_rel))
        if aim_range < 1e-6:
            az = el = 0.0
        else:
            az, el = earth_los_to_az_el(aim_rel / aim_range)

        # 三维 PN：omega = (r x v_rel)/|r|²，a_pn = N*Vc*(omega x los)。
        omega = np.cross(rel, v_rel) / max(float(np.dot(rel, rel)), 1e-12)
        a_pn = self.nav_constant * closing_eff * np.cross(omega, los)

        return InterceptTrajectoryState(
            aim_point_enu=aim,
            los_az_deg=float(az),
            los_el_deg=float(el),
            range_m=rng,
            closing_speed_mps=closing,
            t_go_s=float(t_go),
            accel_pn_enu=a_pn,
        )


class ImageTrackReference:
    """图像平面参考点逐步插值器。

    复位后参考点等于当前检测框中心；此后用 jerk-limited 参考模型逐步移动到
    ``goal``（通常是投影出来的拦截瞄准点）。检测丢失时参考点按当前速度外推
    （coast），重捕时用新测量重新初始化，避免误差阶跃。
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        max_vel_px_s: float = 900.0,
        max_accel_px_s2: float = 4000.0,
        max_jerk_px_s3: float = 30000.0,
        safety_factor: float = 0.7,
    ) -> None:
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self._ref = PositionReference3D(
            max_vel=max_vel_px_s,
            max_accel=max_accel_px_s2,
            max_jerk=max_jerk_px_s3,
            safety_factor=safety_factor,
        )
        self.goal = np.array([image_width * 0.5, image_height * 0.5], dtype=float)
        self.feedforward_velocity = np.zeros(2, dtype=float)
        self.valid = False

    def reset(self, u_px: float, v_px: float,
              du_px_s: float = 0.0, dv_px_s: float = 0.0) -> None:
        self._ref.reset([u_px, v_px, 0.0], [du_px_s, dv_px_s, 0.0])
        self.goal = np.array([u_px, v_px], dtype=float)
        self.feedforward_velocity = np.array([du_px_s, dv_px_s], dtype=float)
        self.valid = True

    def set_goal(self, u_px: float, v_px: float,
                 du_px_s: float = 0.0, dv_px_s: float = 0.0) -> None:
        margin = 2.0
        self.goal = np.array([
            _clamp(u_px, margin, self.image_width - margin),
            _clamp(v_px, margin, self.image_height - margin),
        ], dtype=float)
        self.feedforward_velocity = np.array([du_px_s, dv_px_s], dtype=float)

    def update(self, dt: float,
               u_meas_px: Optional[float] = None,
               v_meas_px: Optional[float] = None) -> ReferenceState:
        if not self._ref.initialized:
            if u_meas_px is None or v_meas_px is None:
                self.reset(self.image_width * 0.5, self.image_height * 0.5)
            else:
                self.reset(float(u_meas_px), float(v_meas_px))
        self._ref.set_target(
            [self.goal[0], self.goal[1], 0.0],
            [self.feedforward_velocity[0], self.feedforward_velocity[1], 0.0],
        )
        if u_meas_px is None or v_meas_px is None:
            current = self._ref.position
        else:
            current = np.array([u_meas_px, v_meas_px, 0.0], dtype=float)
        return self._ref.update(dt, current_position=current)

    @property
    def state(self) -> ReferenceState:
        return self._ref.state


class VisualStrikeGuidance:
    """视觉打击控制律（PBVS + track trajectory + image reference）。"""

    def __init__(
        self,
        camera_tilt_deg: float = 44.0,
        image_size: tuple[int, int] = (1280, 720),
        hfov_deg: float = 70.0,
        approach_range_m: float = 8.0,
        terminal_range_m: float = 2.0,
        hit_radius_m: float = 0.8,
        min_safe_altitude_m: float = 0.0,
        hover_throttle: float = 0.52,
        throttle_per_g: float = 0.45,
        max_angle_deg: float = 60.0,
        yaw_error_gain_dps_per_deg: float = 2.5,
        pitch_error_gain_dps_per_deg: float = 2.5,
        image_attitude_gain: float = 0.8,
        speed_gain: float = 0.8,
        velocity_gain: float = 1.5,
    ) -> None:
        self.camera_tilt_deg = float(camera_tilt_deg)
        self.intrinsics = CameraIntrinsics.from_fov(*image_size, hfov_deg=hfov_deg)
        self.extrinsics = CameraExtrinsics(tilt_deg=camera_tilt_deg)
        self.trajectory = InterceptTrajectory()
        self.image_ref = ImageTrackReference(*image_size)
        self.approach_range_m = float(approach_range_m)
        self.terminal_range_m = float(terminal_range_m)
        self.hit_radius_m = float(hit_radius_m)
        self.min_safe_altitude_m = float(min_safe_altitude_m)
        self.hover_throttle = float(hover_throttle)
        self.throttle_per_g = float(throttle_per_g)
        self.max_angle_deg = float(max_angle_deg)
        self.yaw_error_gain = float(yaw_error_gain_dps_per_deg)
        self.pitch_error_gain = float(pitch_error_gain_dps_per_deg)
        self.image_attitude_gain = float(image_attitude_gain)
        self.speed_gain = float(speed_gain)
        self.velocity_gain = float(velocity_gain)
        self._last_uv: Optional[tuple[float, float]] = None
        self._last_timestamp: Optional[float] = None
        self._last_image_velocity = np.zeros(2, dtype=float)
        self._lost_since: Optional[float] = None
        self.lost_timeout_s = 1.0

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def project_enu_to_pixel(
        self,
        point_enu: np.ndarray,
        own_position_enu: np.ndarray,
        own_rpy_deg: Sequence[float],
    ) -> Optional[tuple[float, float]]:
        rel = np.asarray(point_enu, dtype=float) - np.asarray(own_position_enu, dtype=float)
        r_eb = rotation_earth_from_body(*own_rpy_deg)
        rel_body = r_eb.T @ rel
        r_bc = self.extrinsics.rotation_body_from_camera()
        rel_cam = r_bc.T @ rel_body
        if rel_cam[2] <= 1e-6:
            return None
        u = self.intrinsics.fx * rel_cam[0] / rel_cam[2] + self.intrinsics.cx
        v = self.intrinsics.fy * rel_cam[1] / rel_cam[2] + self.intrinsics.cy
        return float(u), float(v)

    def _phase_for_range(self, rng: float, altitude_m: float) -> StrikePhase:
        if altitude_m < self.min_safe_altitude_m and rng > self.terminal_range_m * 2.0:
            return StrikePhase.ABORT
        if rng > self.approach_range_m:
            return StrikePhase.APPROACH
        if rng > self.terminal_range_m:
            return StrikePhase.TRACK
        return StrikePhase.TERMINAL

    def _throttle_for_accel(self, accel_enu: np.ndarray) -> float:
        thrust_vec = np.asarray(accel_enu, dtype=float) + np.array([0.0, 0.0, G])
        thrust_norm = float(np.linalg.norm(thrust_vec)) / G
        throttle = self.hover_throttle + (thrust_norm - 1.0) * self.throttle_per_g
        return _clamp(throttle, 0.2, 1.0)

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def update(
        self,
        target_position_enu: Optional[np.ndarray],
        target_velocity_enu: Optional[np.ndarray],
        own_position_enu: np.ndarray,
        own_velocity_enu: np.ndarray,
        own_rpy_deg: Sequence[float],
        timestamp: float,
        dt: float = 0.02,
        bbox_xyxy: Optional[Sequence[float]] = None,
        valid: bool = True,
        target_accel_enu: Optional[np.ndarray] = None,
    ) -> StrikeCommand:
        own_pos = np.asarray(own_position_enu, dtype=float)
        own_vel = np.asarray(own_velocity_enu, dtype=float)

        if not valid or target_position_enu is None:
            self._lost_since = timestamp if self._lost_since is None else self._lost_since
            return StrikeCommand(
                phase=StrikePhase.SEARCH,
                throttle_cmd=self.hover_throttle,
                valid=False,
            )
        self._lost_since = None

        target_pos = np.asarray(target_position_enu, dtype=float)
        target_vel = (
            np.zeros(3)
            if target_velocity_enu is None
            else np.asarray(target_velocity_enu, dtype=float)
        )
        traj = self.trajectory.update(target_pos, target_vel, own_pos, own_vel)

        # 目标图像速度估计（用于画面参考的前馈，避免移动目标时参考滞后）。
        u_meas = v_meas = float("nan")
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
            u_meas = (x1 + x2) * 0.5
            v_meas = (y1 + y2) * 0.5
            if self._last_uv is not None and self._last_timestamp is not None:
                dt_uv = max(timestamp - self._last_timestamp, 1e-6)
                self._last_image_velocity = (
                    np.array([u_meas, v_meas]) - np.array(self._last_uv)
                ) / dt_uv
                self._last_image_velocity = np.clip(
                    self._last_image_velocity, -2000.0, 2000.0
                )
            self._last_uv = (u_meas, v_meas)
            self._last_timestamp = timestamp

        # 将拦截瞄准点投影到画面，作为参考目标；投影失败则用画面中心。
        aim_px = self.project_enu_to_pixel(traj.aim_point_enu, own_pos, own_rpy_deg)
        if aim_px is None:
            aim_px = (self.intrinsics.cx, self.intrinsics.cy)
        self.image_ref.set_goal(
            aim_px[0], aim_px[1],
            self._last_image_velocity[0], self._last_image_velocity[1],
        )
        img_state = self.image_ref.update(
            max(float(dt), 1e-4),
            None if not np.isfinite(u_meas) else u_meas,
            None if not np.isfinite(v_meas) else v_meas,
        )
        u_ref, v_ref = float(img_state.position[0]), float(img_state.position[1])
        du_ref, dv_ref = float(img_state.velocity[0]), float(img_state.velocity[1])

        # 参考点/测量点像素 -> 地球系视线 -> 期望视线角。
        az_ref, el_ref = self._pixel_to_az_el(u_ref, v_ref, own_rpy_deg)
        if np.isfinite(u_meas) and np.isfinite(v_meas):
            az_meas, el_meas = self._pixel_to_az_el(u_meas, v_meas, own_rpy_deg)
        else:
            az_meas, el_meas = az_ref, el_ref

        yaw_err = _wrap180(az_ref - az_meas)
        pitch_err = el_ref - el_meas
        # 前馈角速率来自参考点图像速度，经相机焦距近似成视线角速率。
        yaw_rate_ff = np.rad2deg(np.arctan2(du_ref, self.intrinsics.fx))
        # Image v increases downward -> dv_ref > 0 means lower elevation (pitch down).
        pitch_rate_ff = -np.rad2deg(np.arctan2(dv_ref, self.intrinsics.fy))

        yaw_rate_cmd = self.yaw_error_gain * yaw_err + yaw_rate_ff
        pitch_rate_cmd = self.pitch_error_gain * pitch_err + pitch_rate_ff
        pitch_cmd = _clamp(
            el_ref - self.camera_tilt_deg + self.image_attitude_gain * pitch_err,
            -self.max_angle_deg,
            self.max_angle_deg,
        )
        yaw_cmd = _wrap180(az_ref + self.image_attitude_gain * yaw_err)
        roll_cmd = 0.0

        # 速度/推力：沿瞄准点的接近速度 + PN 横向加速度。
        speed_cmd = _clamp(traj.range_m * self.speed_gain, 1.0, 10.0)
        los_aim = traj.aim_point_enu - own_pos
        los_aim_norm = float(np.linalg.norm(los_aim))
        if los_aim_norm > 1e-6:
            los_aim = los_aim / los_aim_norm
        else:
            los_aim = np.array([0.0, 1.0, 0.0])
        v_des = target_vel + los_aim * speed_cmd
        a_cmd = traj.accel_pn_enu + self.velocity_gain * (v_des - own_vel)
        if target_accel_enu is not None:
            a_cmd = a_cmd + np.asarray(target_accel_enu, dtype=float) * 0.5
        throttle = self._throttle_for_accel(a_cmd)

        phase = self._phase_for_range(traj.range_m, float(own_pos[2]))
        if phase == StrikePhase.ABORT:
            throttle = max(throttle, min(1.0, self.hover_throttle + 0.2))

        hit = bool(
            traj.range_m <= self.hit_radius_m
            or (
                phase == StrikePhase.TERMINAL
                and traj.closing_speed_mps > 0.5
                and traj.t_go_s <= 0.25
            )
        )

        return StrikeCommand(
            phase=phase,
            valid=True,
            hit=hit,
            u_ref_px=u_ref,
            v_ref_px=v_ref,
            u_meas_px=u_meas,
            v_meas_px=v_meas,
            az_ref_deg=float(az_ref),
            el_ref_deg=float(el_ref),
            pitch_cmd_deg=float(pitch_cmd),
            roll_cmd_deg=float(roll_cmd),
            yaw_cmd_deg=float(yaw_cmd),
            pitch_rate_dps=float(pitch_rate_cmd),
            yaw_rate_dps=float(yaw_rate_cmd),
            throttle_cmd=float(throttle),
            target_range_m=float(traj.range_m),
            closing_speed_mps=float(traj.closing_speed_mps),
            t_go_s=float(traj.t_go_s),
            aim_point_enu=traj.aim_point_enu.copy(),
            accel_enu=a_cmd.copy(),
        )

    def _pixel_to_az_el(
        self,
        u_px: float,
        v_px: float,
        own_rpy_deg: Sequence[float],
    ) -> tuple[float, float]:
        los_e = pixel_to_earth_los(
            (u_px, v_px),
            self.intrinsics,
            self.extrinsics,
            yaw_deg=own_rpy_deg[2],
            pitch_deg=own_rpy_deg[1],
            roll_deg=own_rpy_deg[0],
        )
        return earth_los_to_az_el(los_e)


__all__ = [
    "ImageTrackReference",
    "InterceptTrajectory",
    "InterceptTrajectoryState",
    "StrikeCommand",
    "StrikePhase",
    "VisualStrikeGuidance",
]
