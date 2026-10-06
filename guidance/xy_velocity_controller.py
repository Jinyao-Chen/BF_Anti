"""伴机侧 XY 速度外环（固件只跑 ALT_HOLD + 姿态内环）。

为什么放在伴机侧
----------------
upstream 的固件位置环（`positionControl`）在 SITL 的 LiftDrag 被控对象上
相位裕度不足：P-only 能悬停但阶跃后振荡增长，加 D 反而更快发散（见
`docs/interception/` 调试记录）。伴机侧外环让我们用自己可控的阻尼/相位
补偿设计，改代码不需要重编固件，而且这正是纯视觉制导最终要用的接口：
制导输出 ENU 加速度/速度 → 本模块转成 roll/pitch 指令 → 固件姿态内环。

接口
----
```
ctrl = XYVelocityController(...)
cmd = ctrl.update(target_pos_enu, target_vel_enu,
                  own_pos_enu, own_vel_enu, yaw_deg, dt,
                  accel_ff_enu=pn_accel_enu)
# cmd.roll_deg / cmd.pitch_deg 为"航空约定"：pitch 抬头为正、roll 右滚为正
```
RC 映射约定见 `strike_to_rc`：`ch[0]=1500+roll*scale`，
`ch[1]=1500-pitch*scale`（Betaflight ANGLE 的 pitch 正方向是低头）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

G = 9.80665


@dataclass
class XYCommand:
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    v_des_enu: np.ndarray
    a_cmd_enu: np.ndarray
    v_err_enu: np.ndarray
    err_int: np.ndarray
    saturated: bool = False


def _clamp_norm(vec: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    norm = float(np.linalg.norm(vec))
    if limit > 0.0 and norm > limit and norm > 1e-9:
        return vec * (limit / norm), True
    return vec, False


class XYVelocityController:
    """位置→速度→加速度→滚转/俯仰的两级外环。

    参数
    ----
    kp_pos_mps_per_m
        位置误差到速度指令的增益（1/s）。
    kp_vel_mps2_per_mps
        速度误差到加速度指令的增益（1/s）。
    max_vel_mps / max_accel_mps2
        速度指令与加速度指令限幅。
    max_angle_deg
        输出倾角限幅。
    vel_lpf_hz
        测量速度低通，抑制 GPS/估计器噪声；0 关闭。
    roll_sign / pitch_sign
        方向标定系数（+1/-1），用于适配固件/RC 的实际响应符号。
    """

    def __init__(
        self,
        kp_pos_mps_per_m: float = 0.8,
        kp_vel_mps2_per_mps: float = 1.5,
        max_vel_mps: float = 5.0,
        max_accel_mps2: float = 4.0,
        max_angle_deg: float = 30.0,
        ki_pos_mps2_per_m: float = 0.0,
        int_limit_m_s: float = 2.0,
        vel_lpf_hz: float = 4.0,
        roll_sign: float = 1.0,
        pitch_sign: float = 1.0,
        yaw_from_velocity: bool = False,
        yaw_rate_limit_dps: float = 60.0,
    ) -> None:
        self.kp_pos = float(kp_pos_mps_per_m)
        self.kp_vel = float(kp_vel_mps2_per_mps)
        self.max_vel = float(max_vel_mps)
        self.max_accel = float(max_accel_mps2)
        self.max_angle = float(max_angle_deg)
        self.ki_pos = float(ki_pos_mps2_per_m)
        self.int_limit = float(int_limit_m_s)
        self.vel_lpf_hz = float(vel_lpf_hz)
        self.roll_sign = float(roll_sign)
        self.pitch_sign = float(pitch_sign)
        self.yaw_from_velocity = bool(yaw_from_velocity)
        self.yaw_rate_limit = float(yaw_rate_limit_dps)

        self._vel_filt: Optional[np.ndarray] = None
        self._err_int = np.zeros(2, dtype=float)

    def reset(self) -> None:
        self._vel_filt = None
        self._err_int = np.zeros(2, dtype=float)

    def _filter_velocity(self, own_vel_enu: np.ndarray, dt: float) -> np.ndarray:
        vel = np.asarray(own_vel_enu, dtype=float).copy()
        if self.vel_lpf_hz <= 0.0 or dt <= 0.0:
            return vel
        if self._vel_filt is None:
            self._vel_filt = vel
        else:
            alpha = min(1.0, 2.0 * np.pi * self.vel_lpf_hz * dt)
            self._vel_filt = self._vel_filt + alpha * (vel - self._vel_filt)
        return self._vel_filt.copy()

    def update(
        self,
        target_pos_enu: Sequence[float],
        target_vel_enu: Sequence[float],
        own_pos_enu: Sequence[float],
        own_vel_enu: Sequence[float],
        yaw_deg: float,
        dt: float,
        accel_ff_enu: Optional[Sequence[float]] = None,
    ) -> XYCommand:
        target_pos = np.asarray(target_pos_enu, dtype=float)[:2]
        target_vel = np.asarray(target_vel_enu, dtype=float)[:2]
        own_pos = np.asarray(own_pos_enu, dtype=float)[:2]
        own_vel = self._filter_velocity(np.asarray(own_vel_enu, dtype=float)[:2], dt)

        # 位置环：位置误差 → 期望速度（P + 可选 I，带目标速度前馈）。
        err = target_pos - own_pos

        def build_v_cmd() -> np.ndarray:
            return target_vel + self.kp_pos * err + self.ki_pos * self._err_int

        if self.ki_pos > 0.0 and dt > 0.0:
            self._err_int = self._err_int + err * dt
            self._err_int = np.clip(self._err_int, -self.int_limit, self.int_limit)

        v_cmd = build_v_cmd()
        v_des, v_sat = _clamp_norm(v_cmd, self.max_vel)

        # 抗积分饱和：速度指令饱和且误差与指令同向时，撤回本次积分累计。
        if v_sat and self.ki_pos > 0.0 and dt > 0.0 and float(np.dot(err, v_cmd)) > 0.0:
            self._err_int = self._err_int - err * dt
            v_cmd = build_v_cmd()
            v_des, v_sat = _clamp_norm(v_cmd, self.max_vel)

        # 速度环：速度误差 → 加速度（带制导加速度前馈）。
        ff = np.zeros(2) if accel_ff_enu is None else np.asarray(accel_ff_enu, dtype=float)[:2]
        v_err = v_des - own_vel
        a_cmd = ff + self.kp_vel * v_err
        a_cmd, a_sat = _clamp_norm(a_cmd, self.max_accel)

        # ENU 横向加速度 → 机体俯仰/滚转（ψ 为罗盘航向，0=北，顺时针为正）。
        psi = np.deg2rad(float(yaw_deg))
        forward = np.array([np.sin(psi), np.cos(psi)])
        right = np.array([np.cos(psi), -np.sin(psi)])
        a_fwd = float(np.dot(a_cmd, forward))
        a_right = float(np.dot(a_cmd, right))

        pitch_cmd = np.rad2deg(np.arctan2(a_fwd, G)) * self.pitch_sign
        roll_cmd = np.rad2deg(np.arctan2(a_right, G)) * self.roll_sign
        pitch_cmd = max(-self.max_angle, min(self.max_angle, pitch_cmd))
        roll_cmd = max(-self.max_angle, min(self.max_angle, roll_cmd))

        yaw_cmd = 0.0
        if self.yaw_from_velocity:
            speed = float(np.linalg.norm(v_des))
            if speed > 0.5:
                heading = np.rad2deg(np.arctan2(v_des[0], v_des[1]))
                yaw_cmd = (heading - float(yaw_deg) + 180.0) % 360.0 - 180.0
                yaw_cmd = max(-self.yaw_rate_limit, min(self.yaw_rate_limit, yaw_cmd))

        return XYCommand(
            roll_deg=float(roll_cmd),
            pitch_deg=float(pitch_cmd),
            yaw_deg=float(yaw_cmd),
            v_des_enu=np.array([v_des[0], v_des[1], 0.0]),
            a_cmd_enu=np.array([a_cmd[0], a_cmd[1], 0.0]),
            v_err_enu=np.array([v_err[0], v_err[1], 0.0]),
            err_int=np.array([self._err_int[0], self._err_int[1]]),
            saturated=bool(v_sat or a_sat),
        )


__all__ = ["XYCommand", "XYVelocityController", "G"]
