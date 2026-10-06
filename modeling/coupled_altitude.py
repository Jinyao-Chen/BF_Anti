"""水平机动 -> 姿态 -> 混控 -> 高度环 的 Simulink 式耦合模型。

这个模块不是重新实现一个飞控，而是把固件里真实存在的几个块按信号流串起来，
用来回答一个具体问题：**水平飞行时的高度损失，到底来自哪里？**

块图
----
::

    target_n ---------> 位置->速度 ----> 速度->倾角 ----> 姿态内环 ----> 混控
                                                |               |
                                                |               v
    target_up ------> 高度 PID <---- 高度估计 <-- 垂向动力学 <-- 推力/倾角耦合
                          |
                          v
                      倾角补偿

模型分为两代
------------
``legacy_*`` / ``ideal`` / ``gravity_only``
    对应 release260904 修复前的历史状态：默认混控会在差动饱和时压低总距，
    旧固件用 ``1/cos^2`` 做倾角补偿。``ring_mixer`` 与 ``full`` 保留历史
    拟合参数，用来复现第 13 节的日志归因。

``current``
    对应当前 release260904：``mixer.c`` 在 ALT_HOLD/POS_HOLD/GPS_RESCUE 下
    优先保总距、只按需缩差动；``altitudeControl()`` 用精确的
    ``1/(cos(roll)cos(pitch))``；offboard 速度环使用 config 化增益，导航
    加减速上限为 2.0 m/s²；水平映射用 ``atan(a/g)``。当前模型中的
    ``mixer_loss=0`` 表示“控制链路不再吃总距”，残余掉高应归因于
    Gazebo LiftDrag 等 plant 气动损失，而不是控制器。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .firmware_params import (
    OFFBOARD_MAX_ACCEL_MPS2,
    SITL_ACTUATOR_TAU_S,
    SITL_ESTIMATOR_TAU_S,
    SITL_FINAL_ALTITUDE_CLI,
    SITL_FINAL_GAINS,
    SITL_RING_HZ,
    SITL_VZ_ESTIMATOR_TAU_S,
)


# --------------------------------------------------------------------------- #
# 参数
# --------------------------------------------------------------------------- #
@dataclass
class CoupledAltitudeParams:
    """耦合模型参数。

    默认值对应 **当前 release260904**（总距优先混控 + 精确倾角补偿 +
    最终 offboard/高度配置）。历史场景在 :func:`case_params` 中显式覆盖。
    """

    # 机体
    mass: float = 1.5
    gravity: float = 9.80665

    # 油门 -> 推力映射（归一化油门 0~1；悬停油门约 0.53）
    hover_throttle: float = 0.53
    thrust_exponent: float = 1.0
    max_thrust_ratio: float = 1.9

    # 水平目标与位置->速度外环
    target_north: float = 30.0
    target_up: float = 10.0
    cruise: float = 5.0
    pos_vel_gain: float = 0.5       # desired_v = gain * distance_error [1/s]
    max_angle_deg: float = 15.0     # ap_max_angle
    max_nav_accel_mps2: float = OFFBOARD_MAX_ACCEL_MPS2

    # 速度误差 -> 倾角指令（对应固件 offboard 分支的 config 化缩放）
    vel_p_gain: float = 2.0         # deg per (m/s)
    vel_d_gain: float = 3.0         # deg per (m/s)
    vel_f_gain: float = 3.0         # deg per (m/s)
    accel_d_gain: float = 1.0       # deg per (m/s^2)，对应 ap_offboard_accel_ff=100
    accel_lpf_tau_s: float = 0.10   # 测量加速度低通
    tilt_lpf_tau_s: float = 0.05    # 倾角指令低通
    horizontal_map: str = "atan"    # linear / atan

    # 姿态内环等效二阶模型：wn=0 表示理想无延迟跟随
    attitude_wn_hz: float = 0.0
    attitude_zeta: float = 0.08
    attitude_delay_s: float = 0.06

    # 混控模式：
    #   ideal          —— 完全保总距
    #   legacy         —— 默认 Betaflight 压低总距行为
    #   thrust_priority—— 当前固件：优先保总距，差动按需缩放；只有差动权限
    #                     低于 mixer_min_authority 时才产生残余总距损失
    mixer_mode: str = "thrust_priority"
    mixer_loss: float = 0.0
    mixer_diff_scale_deg: float = 6.0
    mixer_min_authority: float = 0.35
    mixer_sat_power: float = 2.0
    mixer_tau_s: float = 0.05       # 混控损失的等效时间常数，避免离散抖振

    # 推力/旋翼转速等效一阶滞后；SITL 的力控旋翼在饱和后不能瞬间恢复
    thrust_tau_s: float = SITL_ACTUATOR_TAU_S

    # 高度/垂速估计一阶滞后（必须分开：高度 KF 慢，D 反馈滤波快）
    est_tau_s: float = SITL_ESTIMATOR_TAU_S
    vz_est_tau_s: Optional[float] = None  # None -> 沿用 est_tau_s（历史场景兼容）
    # 垂速 lead：用 IMU 垂向加速度把估计垂速向前预测，补偿估计滞后
    altitude_vel_lead_s: float = 0.0

    # 固件高度环内部增益（见 autopilot_multirotor.c；SITL_FINAL_GAINS 是
    # ap_altitude_p/i/d/f = 50/50/20/0 对应的 PWM 工程量）
    #
    # altitude_structure：
    #   firmware_pid    —— 当前固件结构：P/I 直接作用于高度误差 -> PWM，
    #                      D 作用于垂速误差 -> PWM，F 作用于目标垂速 -> PWM。
    #   legacy_cascade  —— 第 13 节历史拟合用的“位置 P -> 期望垂速 -> 速度
    #                      PID”结构；它与固件实现不同，只用于复现历史归因。
    altitude_structure: str = "firmware_pid"
    altitude_kp: float = SITL_FINAL_GAINS.kp_pwm_per_cm
    altitude_ki: float = SITL_FINAL_GAINS.ki_pwm_per_cm_s
    altitude_kd: float = SITL_FINAL_GAINS.kd_pwm_per_cm_s
    altitude_kf: float = SITL_FINAL_GAINS.kf_pwm_per_cm_s
    altitude_i_limit: float = 150.0
    altitude_vel_max_cms: float = 500.0
    target_vz_cms: float = 0.0
    hover_offset_pwm: float = SITL_FINAL_ALTITUDE_CLI["ap_hover_throttle"] - 1000.0

    # 倾角补偿模式：
    #   none / actual_cos / actual_sq / commanded_cos / commanded_sq
    #   legacy_sq      —— 修复前固件的 1/(cos*cos)
    #   exact_cos      —— 当前固件的 1/(cos(roll)cos(pitch))
    #   firmware       —— legacy_sq 的兼容别名
    #   firmware_exact —— exact_cos 的兼容别名
    tilt_comp: str = "exact_cos"

    # 垂向阻尼（SITL 模型的小阻尼）
    vertical_drag: float = 0.05

    # Plant 前飞升力损失模型。
    #   plant_lift_loss_accel_gain: 额外向下加速度 [m/s² per (m/s)²]，
    #       由 identify_lift_loss 从 SITL 日志辨识；这是当前使用的物理模型。
    #   plant_lift_loss_gain / max: 旧的比例型损失，保留用于历史实验。
    plant_lift_loss_accel_gain: float = 0.0
    plant_lift_loss_gain: float = 0.0
    plant_lift_loss_max: float = 0.5

    # 控制器侧速度²前馈（对应固件 ap_altitude_lift_ff）。
    #   altitude_lift_ff_gain: 额外总距对应的等效垂直加速度
    #       [m/s² per (m/s)²]；0 表示关闭。
    #   lift_ff_couple_horizontal: 推总距时同步减小倾角，避免水平超速。
    altitude_lift_ff_gain: float = 0.0
    lift_ff_couple_horizontal: bool = True

    # 积分器
    dt: float = 0.004


@dataclass
class SimulationResult:
    t: np.ndarray
    north: np.ndarray
    up: np.ndarray
    vn: np.ndarray
    vu: np.ndarray
    pitch_deg: np.ndarray
    pitch_cmd_deg: np.ndarray
    throttle_norm: np.ndarray
    thrust_ratio: np.ndarray
    mixer_factor: np.ndarray
    z_est: np.ndarray
    vz_est: np.ndarray
    accel_u: np.ndarray
    accel_n: np.ndarray
    plant_lift_loss: np.ndarray
    plant_lift_loss_accel: np.ndarray
    lift_ff_accel: np.ndarray

    def metrics(self) -> Dict[str, float]:
        min_idx = int(np.argmin(self.up))
        return {
            "min_up_m": float(self.up.min()),
            "min_up_time_s": float(self.t[min_idx]),
            "max_pitch_deg": float(np.max(np.abs(self.pitch_deg))),
            "max_pitch_cmd_deg": float(np.max(np.abs(self.pitch_cmd_deg))),
            "min_mixer_factor": float(self.mixer_factor.min()),
            "max_throttle_norm": float(self.throttle_norm.max()),
            "min_accel_u_mps2": float(self.accel_u.min()),
            "max_plant_lift_loss": float(self.plant_lift_loss.max()),
            "max_plant_lift_loss_accel_mps2": float(self.plant_lift_loss_accel.max()),
            "max_lift_ff_accel_mps2": float(self.lift_ff_accel.max()),
            "final_up_m": float(self.up[-1]),
        }


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def tilt_gravity_deficit(theta_deg: float) -> float:
    """倾角造成的纯重力垂向加速度损失 ``g*(1-cos(theta))``，单位 m/s²。"""
    theta = math.radians(theta_deg)
    return 9.80665 * (1.0 - math.cos(theta))


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _apply_horizontal_map(mode: str, cmd_deg: float) -> float:
    """把 offboard 速度环输出的“角度当量”映射为实际倾角指令。

    当前固件把速度环输出解释为 ``a/g``（角度当量，单位 deg），实际倾角为
    ``atan(a/g)``；修复前固件直接把它当线性角度使用。
    """
    if mode == "linear":
        return cmd_deg
    if mode == "atan":
        return math.degrees(math.atan(math.radians(cmd_deg)))
    raise ValueError(f"unknown horizontal_map mode: {mode}")


def _tilt_multiplier(mode: str, theta_actual: float, theta_cmd: float) -> float:
    """固件风格的倾角补偿系数。

    ``theta_actual`` / ``theta_cmd`` 单位为弧度，且定义为“低头为正”。
    """
    ca = max(math.cos(theta_actual), 0.6)
    cc = max(math.cos(theta_cmd), 0.6)
    if mode == "none":
        return 1.0
    if mode == "actual_cos":
        return 1.0 / ca
    if mode == "actual_sq":
        return 1.0 / (ca * ca)
    if mode == "commanded_cos":
        return 1.0 / cc
    if mode == "commanded_sq":
        return 1.0 / (cc * cc)
    if mode in ("legacy_sq", "firmware"):
        # 修复前固件：tiltCos = max(min(measured, commanded), 0.6)，再平方取倒数
        tc = max(min(ca, cc), 0.6)
        return 1.0 / (tc * tc)
    if mode in ("exact_cos", "firmware_exact"):
        # 当前固件：tiltCos = max(min(measured, commanded), 0.6)，取倒数
        tc = max(min(ca, cc), 0.6)
        return 1.0 / tc
    raise ValueError(f"unknown tilt_comp mode: {mode}")


def _mixer_factor(
    mode: str,
    diff_deg: float,
    diff_scale_deg: float,
    loss: float,
    min_authority: float,
    power: float,
) -> float:
    """由姿态指令误差估计总距推力保留系数。

    ``legacy`` 复现默认 Betaflight 混控“差动越大、总距被压得越低”。
    ``thrust_priority`` 对应当前固件：差动先缩放到可容纳范围，总距优先；
    只有差动需求超过 ``1 / min_authority`` 的权限时才有残余损失。
    """
    sat = _clip(diff_deg / max(diff_scale_deg, 1e-6), 0.0, 1.0)
    if mode == "ideal":
        return 1.0
    if mode == "legacy":
        return _clip(1.0 - loss * sat ** power, 0.05, 1.0)
    if mode == "thrust_priority":
        # sat=min_authority 时差动刚好能被最小姿态权限容纳；超过后才损失总距。
        over = _clip((sat - min_authority) / max(1.0 - min_authority, 1e-6), 0.0, 1.0)
        return _clip(1.0 - loss * over ** power, 0.05, 1.0)
    raise ValueError(f"unknown mixer_mode: {mode}")


# --------------------------------------------------------------------------- #
# 仿真
# --------------------------------------------------------------------------- #
def simulate(
    params: Optional[CoupledAltitudeParams] = None,
    duration: float = 30.0,
    target_north: Optional[float] = None,
    target_up: Optional[float] = None,
) -> SimulationResult:
    """运行块图仿真，返回完整时间序列。"""
    p = params if params is not None else CoupledAltitudeParams()
    dt = p.dt
    steps = int(round(duration / dt)) + 1
    t = np.linspace(0.0, duration, steps)
    tgt_n = p.target_north if target_north is None else float(target_north)
    tgt_u = p.target_up if target_up is None else float(target_up)

    # 状态
    north = 0.0
    vn = 0.0
    up = tgt_u
    vu = 0.0
    theta = 0.0
    q = 0.0
    z_est = tgt_u
    vz_est = 0.0
    alt_i = 0.0
    prev_accel_n = 0.0
    accel_lpf = 0.0
    tilt_lpf = 0.0
    prev_target_v = 0.0
    thrust_eff = 1.0
    pitch_cmd_lag = 0.0
    mixer_factor = 1.0
    prev_accel_u = 0.0

    out = {
        "t": t,
        "north": np.zeros(steps),
        "up": np.zeros(steps),
        "vn": np.zeros(steps),
        "vu": np.zeros(steps),
        "pitch_deg": np.zeros(steps),
        "pitch_cmd_deg": np.zeros(steps),
        "throttle_norm": np.zeros(steps),
        "thrust_ratio": np.zeros(steps),
        "mixer_factor": np.zeros(steps),
        "z_est": np.zeros(steps),
        "vz_est": np.zeros(steps),
        "accel_u": np.zeros(steps),
        "accel_n": np.zeros(steps),
        "plant_lift_loss": np.zeros(steps),
        "plant_lift_loss_accel": np.zeros(steps),
        "lift_ff_accel": np.zeros(steps),
    }

    for i in range(steps):
        # ---- 1. 水平位置外环 -> 期望速度（带物理可实现的加速度限幅） ----
        desired_v = _clip(p.pos_vel_gain * (tgt_n - north), -p.cruise, p.cruise)
        max_dv = max(p.max_nav_accel_mps2, 0.0) * dt
        desired_v = prev_target_v + _clip(desired_v - prev_target_v, -max_dv, max_dv)
        prev_target_v = desired_v

        # 当前水平速度与控制器速度²前馈（对应 ap_altitude_lift_ff）。
        speed_mps = abs(vn)
        lift_ff_accel = max(0.0, p.altitude_lift_ff_gain) * speed_mps * speed_mps

        # ---- 2. 速度误差 -> 倾角指令（低头为正） ----
        if p.accel_lpf_tau_s > 1e-6:
            accel_lpf += (prev_accel_n - accel_lpf) * dt / p.accel_lpf_tau_s
        else:
            accel_lpf = prev_accel_n
        raw_cmd_deg = (
            p.vel_p_gain * (desired_v - vn)
            + p.vel_d_gain * (-vn)
            + p.vel_f_gain * desired_v
            + p.accel_d_gain * accel_lpf
        )
        if p.tilt_lpf_tau_s > 1e-6:
            tilt_lpf += (raw_cmd_deg - tilt_lpf) * dt / p.tilt_lpf_tau_s
        else:
            tilt_lpf = raw_cmd_deg
        # 增加总距前馈时同步减小倾角指令：推力矢量分配近似保持水平加速度
        # a_h = (g + a_z_ff) * tan(theta) 不变。
        if p.lift_ff_couple_horizontal and lift_ff_accel > 1e-9:
            tilt_lpf *= p.gravity / (p.gravity + lift_ff_accel)
        pitch_cmd_deg = _clip(
            _apply_horizontal_map(p.horizontal_map, tilt_lpf),
            -p.max_angle_deg,
            p.max_angle_deg,
        )
        pitch_cmd = math.radians(pitch_cmd_deg)

        # 指令执行延迟：用一阶滞后代替整数延迟队列，避免步长敏感
        if p.attitude_delay_s > 1e-6:
            pitch_cmd_lag += (pitch_cmd - pitch_cmd_lag) * dt / p.attitude_delay_s
            pitch_cmd_delayed = pitch_cmd_lag
        else:
            pitch_cmd_delayed = pitch_cmd

        # ---- 3. 姿态内环：二阶振铃 ----
        if p.attitude_wn_hz > 0.0:
            wn = 2.0 * math.pi * p.attitude_wn_hz
            theta_ddot = wn * wn * (pitch_cmd_delayed - theta) - 2.0 * p.attitude_zeta * wn * q
            q += theta_ddot * dt
            theta += q * dt
        else:
            theta = pitch_cmd_delayed
            q = 0.0

        # ---- 4. 混控：差动饱和 -> 总距保留系数 ----
        diff_deg = math.degrees(abs(pitch_cmd_delayed - theta))
        mixer_factor_target = _mixer_factor(
            mode=p.mixer_mode,
            diff_deg=diff_deg,
            diff_scale_deg=p.mixer_diff_scale_deg,
            loss=p.mixer_loss,
            min_authority=p.mixer_min_authority,
            power=p.mixer_sat_power,
        )
        if p.mixer_tau_s > 1e-6:
            mixer_factor += (mixer_factor_target - mixer_factor) * dt / p.mixer_tau_s
        else:
            mixer_factor = mixer_factor_target

        # ---- 5. 高度/垂速估计（各自的一阶滞后） ----
        vz_tau = p.est_tau_s if p.vz_est_tau_s is None else p.vz_est_tau_s
        if p.est_tau_s > 1e-6:
            z_est += (up - z_est) * dt / p.est_tau_s
        else:
            z_est = up
        if vz_tau > 1e-6:
            vz_est += (vu - vz_est) * dt / vz_tau
        else:
            vz_est = vu

        # ---- 6. 高度环：两种结构 ----
        alt_err_cm = (tgt_u - z_est) * 100.0
        # itermRelax：固件只在误差 >5m 时把积分削弱到 0.1
        relax = 1.0 if abs(alt_err_cm) < 500.0 else 0.1
        alt_i += alt_err_cm * p.altitude_ki * relax * dt
        alt_i = _clip(alt_i, -p.altitude_i_limit, p.altitude_i_limit)
        # 用 IMU 垂向加速度做 lead，补偿估计滞后；lead 项限幅避免噪声放大。
        lead_cms = _clip(p.altitude_vel_lead_s * prev_accel_u * 100.0, -250.0, 250.0)
        vz_lead_cms = vz_est * 100.0 + lead_cms
        # 固件的 dBoost：垂速超过 5 m/s 后增大 D 阻尼
        abs_vz_cms = abs(vz_lead_cms)
        d_boost = 1.0
        if abs_vz_cms > 500.0:
            ratio = abs_vz_cms / 500.0
            d_boost = (3.0 * ratio - 2.0) / ratio

        if p.altitude_structure == "firmware_pid":
            # 当前固件：P/I 是 PWM 项，D 作用于垂速误差，F 作用于目标垂速。
            altitude_p = alt_err_cm * p.altitude_kp
            velocity_error_cms = p.target_vz_cms - vz_lead_cms
            altitude_d = velocity_error_cms * p.altitude_kd * d_boost
            altitude_f = p.target_vz_cms * p.altitude_kf
            az_cmd = altitude_p + alt_i + altitude_d + altitude_f
        elif p.altitude_structure == "legacy_cascade":
            # 历史拟合结构；保留用于复现第 13 节，不代表当前固件实现。
            vz_cmd = _clip(
                alt_err_cm * p.altitude_kp,
                -p.altitude_vel_max_cms,
                p.altitude_vel_max_cms,
            )
            velocity_error_cms = vz_cmd - vz_lead_cms
            az_cmd = p.altitude_kd * d_boost * velocity_error_cms + alt_i + p.altitude_kf * vz_cmd
        else:
            raise ValueError(f"unknown altitude_structure: {p.altitude_structure}")

        # 固件只对悬停前馈做倾角补偿，不对 PID 项整体乘 tiltMultiplier。
        tilt_mul = _tilt_multiplier(p.tilt_comp, theta, pitch_cmd_delayed)
        tilt_hover_offset = p.hover_offset_pwm * tilt_mul
        # 速度²前馈：额外总距 PWM = a_z_ff / (g / hoverOffset)
        lift_ff_pwm = lift_ff_accel * p.hover_offset_pwm / p.gravity
        throttle_offset = az_cmd + tilt_hover_offset + lift_ff_pwm
        new_throttle = 1000.0 + throttle_offset
        # 固件里还有一个 tilt hover floor，只在低于目标时生效。
        if tilt_mul > 1.0 and alt_err_cm > 0.0:
            new_throttle = max(new_throttle, 1000.0 + tilt_hover_offset)
        new_throttle = _clip(new_throttle, 1000.0, 2000.0)
        throttle_norm = (new_throttle - 1000.0) / 1000.0

        # ---- 7. 油门 -> 推力，再经过混控与 plant 损失、倾角投影 ----
        thrust_ratio = (throttle_norm / p.hover_throttle) ** p.thrust_exponent
        thrust_ratio = _clip(thrust_ratio, 0.0, p.max_thrust_ratio)
        plant_lift_loss = _clip(
            p.plant_lift_loss_gain * vn * vn,
            0.0,
            p.plant_lift_loss_max,
        )
        plant_lift_loss_accel = max(0.0, p.plant_lift_loss_accel_gain) * vn * vn
        thrust_ratio_target = thrust_ratio * mixer_factor * (1.0 - plant_lift_loss)
        if p.thrust_tau_s > 1e-6:
            thrust_eff += (thrust_ratio_target - thrust_eff) * dt / p.thrust_tau_s
        else:
            thrust_eff = thrust_ratio_target
        accel_u = (
            p.gravity * thrust_eff * math.cos(theta)
            - p.gravity
            - p.vertical_drag * vu
            - plant_lift_loss_accel
        )
        accel_n = p.gravity * thrust_eff * math.sin(theta)
        prev_accel_n = accel_n
        prev_accel_u = accel_u

        # ---- 8. 积分 ----
        north += vn * dt
        vn += accel_n * dt
        up += vu * dt
        vu += accel_u * dt
        # 防止数值上冲出地面；真值是穿过去的，这里只用于观察趋势
        # up = max(up, -2.0)

        out["north"][i] = north
        out["up"][i] = up
        out["vn"][i] = vn
        out["vu"][i] = vu
        out["pitch_deg"][i] = math.degrees(theta)
        out["pitch_cmd_deg"][i] = pitch_cmd_deg
        out["throttle_norm"][i] = throttle_norm
        out["thrust_ratio"][i] = thrust_eff
        out["mixer_factor"][i] = mixer_factor
        out["z_est"][i] = z_est
        out["vz_est"][i] = vz_est
        out["accel_u"][i] = accel_u
        out["accel_n"][i] = accel_n
        out["plant_lift_loss"][i] = plant_lift_loss
        out["plant_lift_loss_accel"][i] = plant_lift_loss_accel
        out["lift_ff_accel"][i] = lift_ff_accel

    return SimulationResult(**out)


# --------------------------------------------------------------------------- #
# 典型场景
# --------------------------------------------------------------------------- #
def _legacy_ring_mixer_params() -> CoupledAltitudeParams:
    """第 13 节从旧 SITL 日志拟合出的历史参数。"""
    p = CoupledAltitudeParams()
    p.attitude_wn_hz = 1.4
    p.attitude_zeta = 0.071
    p.attitude_delay_s = 0.06

    # 旧混控：差动饱和时压低总距；旧固件用 1/cos^2
    p.mixer_mode = "legacy"
    p.mixer_loss = 0.566
    p.mixer_diff_scale_deg = 2.05
    p.thrust_tau_s = 0.277
    p.tilt_comp = "legacy_sq"

    # 旧 offboard / 高度参数
    p.altitude_structure = "legacy_cascade"
    p.horizontal_map = "linear"
    p.max_nav_accel_mps2 = 5.0
    p.vel_p_gain = 2.0
    p.vel_d_gain = 3.0
    p.vel_f_gain = 3.0
    p.accel_d_gain = 4.0
    p.altitude_kp = 0.2
    p.altitude_ki = 0.049
    p.altitude_kd = 0.797
    p.altitude_kf = 0.1
    p.est_tau_s = 0.0
    return p


def case_params(case: str) -> CoupledAltitudeParams:
    """返回典型场景参数组合。"""
    if case == "ideal":
        p = CoupledAltitudeParams()
        p.attitude_wn_hz = 0.0
        p.mixer_mode = "ideal"
        p.mixer_loss = 0.0
        p.est_tau_s = 0.0
        p.tilt_comp = "actual_cos"
        p.max_nav_accel_mps2 = OFFBOARD_MAX_ACCEL_MPS2
        p.horizontal_map = "atan"
        p.altitude_structure = "firmware_pid"
        return p
    if case == "gravity_only":
        # 理想姿态/估计，只关闭倾角补偿，检验“重力补偿不足”假设
        p = case_params("ideal")
        p.tilt_comp = "none"
        return p
    if case == "ring_mixer":
        # 历史场景：姿态振铃 + 旧混控压低总距，但估计理想、倾角补偿按旧固件
        return _legacy_ring_mixer_params()
    if case == "full":
        # 历史全链路：再加高度估计滞后，复现第 13 节的 SITL 掉高
        p = _legacy_ring_mixer_params()
        p.est_tau_s = 1.129
        return p
    if case == "current":
        # 当前 release260904：总距优先 + 精确倾角补偿 + config 化 offboard
        p = CoupledAltitudeParams()
        p.attitude_wn_hz = SITL_RING_HZ
        p.attitude_zeta = 0.071
        p.attitude_delay_s = 0.06
        p.mixer_mode = "thrust_priority"
        p.mixer_loss = 0.0
        p.mixer_min_authority = 0.35
        p.thrust_tau_s = SITL_ACTUATOR_TAU_S
        p.tilt_comp = "exact_cos"
        p.horizontal_map = "atan"
        p.max_nav_accel_mps2 = OFFBOARD_MAX_ACCEL_MPS2
        p.est_tau_s = SITL_ESTIMATOR_TAU_S
        p.vz_est_tau_s = SITL_VZ_ESTIMATOR_TAU_S
        p.altitude_structure = "firmware_pid"
        return p
    if case == "current_liftdrag":
        # 当前固件 + 从 5/10 m/s SITL 日志辨识的 LiftDrag 损失（无前馈）。
        p = case_params("current")
        p.plant_lift_loss_accel_gain = 0.00532
        return p
    if case == "current_ff":
        # 当前固件 + 相同 plant 损失 + 速度²前馈 + 水平推力耦合。
        p = case_params("current_liftdrag")
        p.altitude_lift_ff_gain = 0.00532
        p.lift_ff_couple_horizontal = True
        return p
    raise ValueError(f"unknown case: {case}")


CASES = (
    "ideal",
    "gravity_only",
    "ring_mixer",
    "full",
    "current",
    "current_liftdrag",
    "current_ff",
)


def run_cases(duration: float = 30.0) -> Dict[str, Dict[str, float]]:
    return {case: simulate(case_params(case), duration=duration).metrics() for case in CASES}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="水平机动-姿态-混控-高度耦合模型")
    parser.add_argument("--case", choices=CASES, default="current")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--target-north", type=float, default=30.0)
    parser.add_argument("--target-up", type=float, default=10.0)
    parser.add_argument("--all", action="store_true", help="打印全部场景对比")
    parser.add_argument("--vel-lead", type=float, default=None, help="覆盖垂速 lead 时间 s")
    parser.add_argument("--output", default="", help="可选 JSON 输出路径")
    args = parser.parse_args(argv)

    def make(case: str) -> CoupledAltitudeParams:
        params = case_params(case)
        if args.vel_lead is not None:
            params.altitude_vel_lead_s = args.vel_lead
        return params

    if args.all:
        result = {
            case: simulate(
                make(case), duration=args.duration,
                target_north=args.target_north, target_up=args.target_up,
            ).metrics()
            for case in CASES
        }
    else:
        result = {
            args.case: simulate(
                make(args.case), duration=args.duration,
                target_north=args.target_north, target_up=args.target_up,
            ).metrics()
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
