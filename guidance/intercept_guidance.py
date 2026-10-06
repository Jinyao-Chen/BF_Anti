"""反无人机拦截制导：44° 上仰相机、三维比例导引与终端俯冲。

设计要点
--------
1. **相机几何**：相机固连机体并相对机头抬高 ``camera_tilt_deg``。同一高度
   的目标要让光轴居中，机体俯仰角应为 ``elevation - camera_tilt``。
2. **三维 PN**：用向量形式计算视线角速度与横向加速度，而不是把像素误差
   直接映射成姿态角。``a_pn = N * Vc * (omega x los)``。
3. **速度/推力**：外环生成期望速度与加速度，推力大小按
   ``T/mg = norm(a_cmd + g_up) / g`` 归一化，天然补偿倾斜时的重力分量。
4. **状态机**：SEARCH -> APPROACH -> TRACK -> TERMINAL，并带 ABORT 安全分支。
5. **命中**：距离小于命中半径，或终端阶段接近速度足够且剩余时间极小。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class GuidancePhase(str, Enum):
    SEARCH = "SEARCH"
    APPROACH = "APPROACH"
    TRACK = "TRACK"
    TERMINAL = "TERMINAL"
    ABORT = "ABORT"


@dataclass
class GuidanceCommand:
    """给飞控/融合层的制导指令。

    角度为机体姿态指令（度），油门为归一化 ``[0, 1]``；``accel_enu`` 为
    ENU 下的期望加速度（m/s²），便于调试和记录。
    """

    phase: GuidancePhase
    roll_cmd_deg: float = 0.0
    pitch_cmd_deg: float = 0.0
    yaw_cmd_deg: float = 0.0
    throttle_cmd: float = 0.5
    desired_speed_mps: float = 0.0
    target_range_m: float = float("inf")
    az_deg: float = 0.0
    el_deg: float = 0.0
    los_rate_az_dps: float = 0.0
    los_rate_el_dps: float = 0.0
    accel_enu: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    t_go_s: float = float("inf")
    hit: bool = False
    valid: bool = False


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros(3, dtype=float)


def los_az_el_from_relative(relative_enu: np.ndarray) -> Tuple[float, float]:
    """相对位置 ENU -> ``(方位角, 俯仰角)``，单位度。

    方位角为罗盘角：0=北，90=东；俯仰角抬头为正。
    """
    rel = np.asarray(relative_enu, dtype=float)
    east, north, up = float(rel[0]), float(rel[1]), float(rel[2])
    horiz = float(np.hypot(east, north))
    az = float(np.rad2deg(np.arctan2(east, north)) % 360.0)
    el = float(np.rad2deg(np.arctan2(up, horiz))) if horiz > 1e-9 else 90.0
    return az, el


def _heading_left(yaw_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """罗盘航向单位向量与机体左向单位向量（ENU）。"""
    psi = np.deg2rad(yaw_deg)
    heading = np.array([np.sin(psi), np.cos(psi), 0.0])
    left = np.array([-np.cos(psi), np.sin(psi), 0.0])
    return heading, left


class InterceptGuidance:
    """三段式拦截制导 + 终端俯冲。

    参数
    ----
    camera_tilt_deg
        相机相对机头上仰角，默认 44°。
    approach_range_m / terminal_range_m
        分段距离阈值。``> approach`` 为 APPROACH，``> terminal`` 为 TRACK，
        否则为 TERMINAL。
    """

    def __init__(
        self,
        camera_tilt_deg: float = 44.0,
        approach_range_m: float = 8.0,
        terminal_range_m: float = 2.0,
        min_pitch_deg: float = -75.0,
        max_pitch_deg: float = 45.0,
        max_roll_deg: float = 60.0,
        max_yaw_deg: float = 180.0,
        nav_constant: float = 3.5,
        min_closing_speed_mps: float = 0.8,
        max_speed_mps: float = 8.0,
        max_terminal_speed_mps: float = 14.0,
        max_accel_mps2: float = 10.0,
        velocity_gain: float = 1.6,
        hover_throttle: float = 0.52,
        throttle_per_g: float = 0.45,
        hit_radius_m: float = 0.8,
        min_safe_altitude_m: float = 0.0,
        speed_schedule: str = "proportional",
    ) -> None:
        if approach_range_m <= terminal_range_m:
            raise ValueError("approach_range_m must be > terminal_range_m")
        self.camera_tilt_deg = float(camera_tilt_deg)
        self.approach_range_m = float(approach_range_m)
        self.terminal_range_m = float(terminal_range_m)
        self.min_pitch_deg = float(min_pitch_deg)
        self.max_pitch_deg = float(max_pitch_deg)
        self.max_roll_deg = float(max_roll_deg)
        self.max_yaw_deg = float(max_yaw_deg)
        self.nav_constant = float(nav_constant)
        self.min_closing_speed_mps = float(min_closing_speed_mps)
        self.max_speed_mps = float(max_speed_mps)
        self.max_terminal_speed_mps = float(max_terminal_speed_mps)
        self.max_accel_mps2 = float(max_accel_mps2)
        self.velocity_gain = float(velocity_gain)
        self.hover_throttle = float(hover_throttle)
        self.throttle_per_g = float(throttle_per_g)
        self.hit_radius_m = float(hit_radius_m)
        self.min_safe_altitude_m = float(min_safe_altitude_m)
        self.speed_schedule = str(speed_schedule)

        self.last_az_deg: Optional[float] = None
        self.last_el_deg: Optional[float] = None
        self.last_timestamp: Optional[float] = None

    def reset(self) -> None:
        self.last_az_deg = None
        self.last_el_deg = None
        self.last_timestamp = None

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def update(
        self,
        target,
        own_position_enu: np.ndarray,
        own_velocity_enu: Optional[np.ndarray],
        timestamp: float,
        dt: Optional[float] = None,
        target_accel_enu: Optional[np.ndarray] = None,
    ) -> GuidanceCommand:
        if target is None or not getattr(target, "valid", False):
            self.last_az_deg = None
            self.last_el_deg = None
            self.last_timestamp = None
            return GuidanceCommand(
                phase=GuidancePhase.SEARCH,
                throttle_cmd=self.hover_throttle,
                valid=False,
            )

        own_pos = np.asarray(own_position_enu, dtype=float)
        own_vel = np.zeros(3) if own_velocity_enu is None else np.asarray(own_velocity_enu, dtype=float)
        target_pos = np.asarray(target.position_enu, dtype=float)
        target_vel = np.asarray(getattr(target, "velocity_enu", np.zeros(3)), dtype=float)

        rel = target_pos - own_pos
        rng = float(np.linalg.norm(rel))
        if rng < 1e-6:
            return GuidanceCommand(
                phase=GuidancePhase.TERMINAL,
                pitch_cmd_deg=_clamp(-self.camera_tilt_deg, self.min_pitch_deg, self.max_pitch_deg),
                throttle_cmd=min(1.0, self.hover_throttle + self.throttle_per_g * 0.5),
                target_range_m=0.0,
                hit=True,
                valid=True,
            )

        los = rel / rng
        az_deg, el_deg = los_az_el_from_relative(rel)
        dt = self._resolve_dt(timestamp, dt)
        az_rate, el_rate = self._los_rates(az_deg, el_deg, dt)

        # 相对速度与接近速度（自己 - 目标）。
        v_rel = own_vel - target_vel
        vc = float(-np.dot(los, v_rel))
        vc_eff = max(vc, self.min_closing_speed_mps)

        # 三维比例导引：omega = (r x v_rel) / |r|²；a_pn = N * Vc * (omega x los)。
        omega = np.cross(rel, v_rel) / max(float(np.dot(rel, rel)), 1e-12)
        a_pn = self.nav_constant * vc_eff * np.cross(omega, los)

        # APN：目标加速度前馈（0.5*a_t 是经典 augmented PN 项）。
        if target_accel_enu is not None:
            a_pn = a_pn + 0.5 * np.asarray(target_accel_enu, dtype=float)[:3]

        phase = self._phase_for_range(rng, own_pos[2])
        speed_cmd = self._speed_command(phase, rng)
        v_des = los * speed_cmd
        a_cmd = a_pn + self.velocity_gain * (v_des - own_vel)

        # 安全：低于最低高度且离目标还不近时，优先拉起复飞。
        if own_pos[2] < self.min_safe_altitude_m and rng > max(self.terminal_range_m * 2.0, 3.0):
            phase = GuidancePhase.ABORT
            a_cmd = np.array([0.0, 0.0, 3.0], dtype=float)
            speed_cmd = 0.0

        # 限幅总加速度。
        a_norm = float(np.linalg.norm(a_cmd))
        if a_norm > self.max_accel_mps2:
            a_cmd = a_cmd * (self.max_accel_mps2 / a_norm)

        # 几何瞄准：让相机光轴对准目标。
        pitch_aim = _clamp(el_deg - self.camera_tilt_deg, self.min_pitch_deg, self.max_pitch_deg)
        yaw_aim = _wrap180(az_deg)
        lead_pitch, lead_yaw = self._lead_bias(
            target_pos, target_vel, own_pos, vc, az_deg, el_deg
        )

        if phase == GuidancePhase.APPROACH:
            pitch_cmd = pitch_aim
            yaw_cmd = yaw_aim
        elif phase == GuidancePhase.TERMINAL:
            # 终端把光轴压向目标，同时用横向加速度修正 roll。
            pitch_cmd = _clamp(pitch_aim + lead_pitch, self.min_pitch_deg, self.max_pitch_deg)
            yaw_cmd = _wrap180(yaw_aim + lead_yaw)
        elif phase == GuidancePhase.ABORT:
            pitch_cmd = _clamp(10.0, self.min_pitch_deg, self.max_pitch_deg)
            yaw_cmd = yaw_aim
        else:  # TRACK
            pitch_cmd = _clamp(pitch_aim + 0.5 * lead_pitch, self.min_pitch_deg, self.max_pitch_deg)
            yaw_cmd = _wrap180(yaw_aim + 0.5 * lead_yaw)

        roll_cmd = self._roll_for_lateral_accel(a_cmd, yaw_cmd)

        # 推力归一化：T/mg = |a_cmd + g_up| / g。
        thrust_vec = a_cmd + np.array([0.0, 0.0, 9.80665])
        thrust_norm = float(np.linalg.norm(thrust_vec)) / 9.80665
        throttle = self.hover_throttle + (thrust_norm - 1.0) * self.throttle_per_g
        if phase == GuidancePhase.TERMINAL:
            throttle = max(throttle, min(1.0, self.hover_throttle + 0.25))
        if phase == GuidancePhase.ABORT:
            throttle = max(throttle, min(1.0, self.hover_throttle + 0.15))
        throttle = _clamp(throttle, 0.25, 1.0)

        t_go = rng / max(vc_eff, 1e-6)
        hit = bool(rng <= self.hit_radius_m or (phase == GuidancePhase.TERMINAL and vc > 0.5 and t_go <= 0.25))

        return GuidanceCommand(
            phase=phase,
            roll_cmd_deg=float(roll_cmd),
            pitch_cmd_deg=float(pitch_cmd),
            yaw_cmd_deg=float(_clamp(yaw_cmd, -self.max_yaw_deg, self.max_yaw_deg)),
            throttle_cmd=float(throttle),
            desired_speed_mps=float(speed_cmd),
            target_range_m=float(rng),
            az_deg=float(az_deg),
            el_deg=float(el_deg),
            los_rate_az_dps=float(az_rate),
            los_rate_el_dps=float(el_rate),
            accel_enu=a_cmd.copy(),
            t_go_s=float(t_go),
            hit=hit,
            valid=True,
        )

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _resolve_dt(self, timestamp: float, dt: Optional[float]) -> float:
        if dt is not None:
            return max(float(dt), 1e-6)
        if self.last_timestamp is None:
            return 1e-6
        return max(float(timestamp) - self.last_timestamp, 1e-6)

    def _los_rates(self, az_deg: float, el_deg: float, dt: float) -> Tuple[float, float]:
        az_rate = 0.0
        el_rate = 0.0
        if self.last_az_deg is not None and self.last_el_deg is not None:
            az_rate = _wrap180(az_deg - self.last_az_deg) / dt
            el_rate = (el_deg - self.last_el_deg) / dt
        self.last_az_deg = az_deg
        self.last_el_deg = el_deg
        self.last_timestamp = float(self.last_timestamp + dt) if self.last_timestamp is not None else dt
        return az_rate, el_rate

    def _phase_for_range(self, rng: float, own_up_m: float) -> GuidancePhase:
        if own_up_m < self.min_safe_altitude_m:
            return GuidancePhase.ABORT
        if rng > self.approach_range_m:
            return GuidancePhase.APPROACH
        if rng > self.terminal_range_m:
            return GuidancePhase.TRACK
        return GuidancePhase.TERMINAL

    def _speed_command(self, phase: GuidancePhase, rng: float) -> float:
        if self.speed_schedule == "constant":
            # 撞机式打击：全程保持最大接近速度，只在终端用更大的上限。
            if phase == GuidancePhase.TERMINAL:
                return self.max_terminal_speed_mps
            if phase in (GuidancePhase.APPROACH, GuidancePhase.TRACK):
                return self.max_speed_mps
            return 0.0
        if phase == GuidancePhase.APPROACH:
            return _clamp(0.6 * (rng - self.terminal_range_m), 1.0, self.max_speed_mps)
        if phase == GuidancePhase.TRACK:
            return _clamp(0.8 * rng, 1.5, self.max_speed_mps)
        if phase == GuidancePhase.TERMINAL:
            return _clamp(3.0 + 0.5 * rng, 3.0, self.max_terminal_speed_mps)
        return 0.0

    def _lead_bias(
        self,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        own_pos: np.ndarray,
        vc: float,
        az_deg: float,
        el_deg: float,
    ) -> Tuple[float, float]:
        """目标运动前馈：预测命中点，返回相对当前视线的偏置角（度）。"""
        t_go = max(float(np.linalg.norm(target_pos - own_pos)), 0.0) / max(vc, self.min_closing_speed_mps)
        t_go = _clamp(t_go, 0.0, 3.0)
        predicted = target_pos + target_vel * t_go
        az_pred, el_pred = los_az_el_from_relative(predicted - own_pos)
        lead_yaw = _clamp(_wrap180(az_pred - az_deg), -25.0, 25.0)
        lead_pitch = _clamp(el_pred - el_deg, -20.0, 20.0)
        return lead_pitch, lead_yaw

    def _roll_for_lateral_accel(self, a_cmd_enu: np.ndarray, yaw_deg: float) -> float:
        """把 ENU 横向加速度需求转成机体滚转指令（小角度）。"""
        _, left = _heading_left(yaw_deg)
        a_right = -float(np.dot(np.asarray(a_cmd_enu, dtype=float)[:2], left[:2]))  # 机体右向 = -left
        roll = np.rad2deg(np.arctan2(a_right, 9.80665))
        return _clamp(roll, -self.max_roll_deg, self.max_roll_deg)
