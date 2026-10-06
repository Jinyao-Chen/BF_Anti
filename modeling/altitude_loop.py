"""高度回路的一维理论模型与自动整定（固件等效结构）。

固件 ``altitudeControl()`` 的真实结构是：

.. code-block:: c

    altitudeP = altitudeErrorCm * altitudeKp;
    altitudeI += altitudeErrorCm * altitudeKi * itermRelax * dt;
    velocityError = targetVerticalVelocity - verticalVelocity;
    altitudeD = velocityError * altitudeKd * dBoost;
    altitudeF = targetVerticalVelocity * altitudeKf;
    throttleOffset = altitudeP + altitudeI + altitudeD + altitudeF + tiltHover;

也就是说 P/I 直接作用于高度误差（PWM 工程量），D 作用于垂速误差，
不是“位置 P -> 期望垂速 -> 速度 PI”的串级结构。本模块用固件等效的
PWM 工程量增益做一维辨识/整定，因此输出可以直接写回：

* ``ap_altitude_p = altitudeKp / 0.005``
* ``ap_altitude_i = altitudeKi / 0.002``
* ``ap_altitude_d = altitudeKd / 0.01``
* ``ap_altitude_f = altitudeKf / (0.1/30)``

真实系统还有执行机构（电机 + 桨叶 + 机体）的一阶滞后和高度/垂速估计滞后；
``--sitl`` 预设使用 release260904 已辨识的 0.28 s / 0.5 s 值。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from .firmware_params import (
    ALTITUDE_I_LIMIT,
    SITL_ACTUATOR_TAU_S,
    SITL_ESTIMATOR_TAU_S,
    SITL_FINAL_GAINS,
    SITL_VZ_ESTIMATOR_TAU_S,
    cli_from_altitude_kd,
    cli_from_altitude_ki,
    cli_from_altitude_kp,
    cli_from_altitude_kf,
)


@dataclass
class AltitudePlantParams:
    """一维高度 plant 参数。"""

    mass: float = 1.4
    gravity: float = 9.80665
    # 油门 -> 推力斜率；hover_throttle_norm=0.53 对应 ap_hover_throttle=1530
    hover_throttle_norm: float = 0.53
    # 总距推力一阶滞后：真实 Iris 约 80 ms；SITL 辨识约 280 ms
    thrust_tau_s: float = 0.08
    # 高度/垂速估计的等效滞后（SITL KF/滤波相位损失）。
    # 垂速走 altitude_d_lpf（默认 300 -> 3 Hz），必须与高度滞后分开。
    measurement_tau_s: float = 0.05
    vz_measurement_tau_s: Optional[float] = None
    thrust_min_n: float = 0.0
    thrust_max_n: float = 41.0
    max_accel_mps2: float = 3.0
    # 常值向下扰动加速度，用于模拟水平机动时的平均升力损失/阻力配平需求。
    disturbance_accel_mps2: float = 0.0
    altitude_i_limit_pwm: float = ALTITUDE_I_LIMIT
    target_vz_cms: float = 0.0
    pwm_range_min: float = 1000.0
    pwm_range_max: float = 2000.0

    @property
    def pwm_to_accel_mps2(self) -> float:
        """1 PWM 总距偏移在悬停点附近的等效垂向加速度 [m/s²]。"""
        return self.gravity / (self.hover_throttle_norm * (self.pwm_range_max - self.pwm_range_min))

    @property
    def max_accel_pwm(self) -> float:
        return self.max_accel_mps2 / self.pwm_to_accel_mps2


def sitl_params() -> AltitudePlantParams:
    """release260904 SITL 已辨识参数。"""
    return AltitudePlantParams(
        mass=1.5,
        hover_throttle_norm=0.53,
        thrust_tau_s=SITL_ACTUATOR_TAU_S,
        measurement_tau_s=SITL_ESTIMATOR_TAU_S,
        vz_measurement_tau_s=SITL_VZ_ESTIMATOR_TAU_S,
        max_accel_mps2=3.0,
        thrust_max_n=1.5 * 9.80665 * 1.9,
    )


@dataclass
class AltitudeGains:
    """固件内部高度环增益（PWM 工程量）。"""

    kp: float  # [PWM/cm]
    ki: float  # [PWM/(cm*s)]
    kd: float  # [PWM/(cm/s)]
    kf: float = 0.0  # [PWM/(cm/s)]

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_betaflight(self) -> Dict[str, float]:
        """固件内部增益 -> CLI ``ap_altitude_*``（与 autopilot_multirotor.c 一致）。"""
        return {
            "ap_altitude_p": round(cli_from_altitude_kp(self.kp), 4),
            "ap_altitude_i": round(cli_from_altitude_ki(self.ki), 4),
            "ap_altitude_d": round(cli_from_altitude_kd(self.kd), 4),
            "ap_altitude_f": round(cli_from_altitude_kf(self.kf), 4),
        }


def simulate_altitude(
    gains: AltitudeGains,
    params: Optional[AltitudePlantParams] = None,
    z_des: float = 2.0,
    duration: float = 10.0,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """仿真高度阶跃，返回 ``(t, z, T)``。"""
    p = params if params is not None else AltitudePlantParams()
    steps = int(duration / dt) + 1
    t = np.linspace(0.0, duration, steps)
    z = np.zeros(steps)
    vz = np.zeros(steps)
    thrust = np.full(steps, p.mass * p.gravity)
    integral = 0.0
    z_meas = 0.0
    vz_meas = 0.0
    vz_tau = p.measurement_tau_s if p.vz_measurement_tau_s is None else p.vz_measurement_tau_s
    alpha_z = dt / (p.measurement_tau_s + dt)
    alpha_vz = dt / (vz_tau + dt)

    for i in range(1, steps):
        # 测量滞后（高度与垂速分别一阶）
        z_meas += alpha_z * (z[i - 1] - z_meas)
        vz_meas += alpha_vz * (vz[i - 1] - vz_meas)

        alt_err_cm = (z_des - z_meas) * 100.0
        p_term = gains.kp * alt_err_cm
        integral += gains.ki * alt_err_cm * dt
        integral = float(np.clip(integral, -p.altitude_i_limit_pwm, p.altitude_i_limit_pwm))
        vel_err_cms = p.target_vz_cms - vz_meas * 100.0
        d_term = gains.kd * vel_err_cms
        f_term = gains.kf * p.target_vz_cms
        az_cmd_pwm = p_term + integral + d_term + f_term

        # 加速度/推力限幅 + 反算抗饱和
        az_cmd_clamped = float(np.clip(az_cmd_pwm, -p.max_accel_pwm, p.max_accel_pwm))
        if abs(az_cmd_clamped - az_cmd_pwm) > 1e-9:
            integral = float(
                np.clip(
                    az_cmd_clamped - p_term - d_term - f_term,
                    -p.altitude_i_limit_pwm,
                    p.altitude_i_limit_pwm,
                )
            )

        thrust_cmd = p.mass * (p.gravity + az_cmd_clamped * p.pwm_to_accel_mps2)
        thrust_cmd = max(p.thrust_min_n, min(p.thrust_max_n, thrust_cmd))
        # 推力一阶滞后
        if p.thrust_tau_s > 1e-6:
            thrust_dot = (thrust_cmd - thrust[i - 1]) / p.thrust_tau_s
            thrust[i] = thrust[i - 1] + thrust_dot * dt
            thrust[i] = max(p.thrust_min_n, min(p.thrust_max_n, thrust[i]))
        else:
            thrust[i] = thrust_cmd

        # 垂直动力学（disturbance_accel_mps2 为向下为正的常值扰动）
        az = thrust[i] / p.mass - p.gravity - p.disturbance_accel_mps2
        vz[i] = vz[i - 1] + az * dt
        z[i] = z[i - 1] + vz[i] * dt

    return t, z, thrust


def altitude_cost(
    gains: AltitudeGains,
    params: Optional[AltitudePlantParams] = None,
    z_des: float = 2.0,
    duration: float = 10.0,
) -> float:
    """ITAE + 超调惩罚 + 控制量惩罚 + 稳态误差。"""
    t, z, thrust = simulate_altitude(gains, params, z_des=z_des, duration=duration)
    err = z_des - z
    itae = float(np.trapz(t * np.abs(err), t))
    overshoot = max(0.0, float(np.max(z)) - z_des)
    effort = float(np.mean(np.abs(np.diff(thrust)))) * 1e-3
    ss = abs(float(z[-1]) - z_des)
    return itae + 5.0 * overshoot + effort + 20.0 * ss


def tune_altitude(
    params: Optional[AltitudePlantParams] = None,
    z_des: float = 2.0,
    duration: float = 10.0,
    seed: int = 0,
) -> AltitudeGains:
    """在物理约束下优化固件等效高度 PID 增益。

    目标函数同时评估标称阶跃和带常值向下扰动的阶跃，并惩罚过大增益；
    否则纯阶跃优化会退化成 ``I≈0`` 的高 P/D 解，无法配平水平机动引起的
    垂向升力损失。
    """
    p = params if params is not None else AltitudePlantParams()
    disturbed = replace(p, disturbance_accel_mps2=0.8)
    # kp [PWM/cm], ki [PWM/(cm*s)], kd [PWM/(cm/s)]
    # 限制在 Betaflight CLI 合理可写范围，避免优化器给出 ap_altitude_p=300
    # 这类虽然在本模型里可行、但没有工程裕度的解。
    bounds = [(0.05, 0.6), (0.005, 0.3), (0.02, 1.0)]

    def objective(x: Sequence[float]) -> float:
        kp, ki, kd = [float(v) for v in x]
        gains = AltitudeGains(kp=kp, ki=ki, kd=kd)
        nominal = altitude_cost(gains, p, z_des, duration)
        disturbed_cost = altitude_cost(gains, disturbed, z_des, duration)
        # 惩罚过大 P/D/I，把解拉向有噪声/延迟裕度的区域。
        gain_penalty = 0.5 * (kp + kd) + 1.0 * ki
        return nominal + 2.0 * disturbed_cost + gain_penalty

    result = differential_evolution(
        objective,
        bounds,
        seed=seed,
        maxiter=80,
        popsize=12,
        tol=1e-6,
        polish=True,
    )
    kp, ki, kd = [float(v) for v in result.x]
    return AltitudeGains(kp=kp, ki=ki, kd=kd)


def evaluate_altitude(
    gains: AltitudeGains,
    params: Optional[AltitudePlantParams] = None,
    z_des: float = 2.0,
    duration: float = 10.0,
) -> Dict[str, float]:
    t, z, thrust = simulate_altitude(gains, params, z_des, duration)
    return {
        "cost": float(altitude_cost(gains, params, z_des, duration)),
        "overshoot_m": float(max(0.0, np.max(z) - z_des)),
        "steady_state_error_m": float(abs(z[-1] - z_des)),
        "settling_time_s": _settling_time(t, z, z_des),
        "max_thrust_n": float(np.max(thrust)),
        "min_thrust_n": float(np.min(thrust)),
    }


def _settling_time(t: np.ndarray, z: np.ndarray, z_des: float) -> float:
    band = 0.02 * max(abs(z_des), 1e-9)
    inside = np.where(np.abs(z - z_des) <= band)[0]
    return float(t[inside[0]]) if inside.size else float(t[-1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="固件等效高度 PID 整定")
    parser.add_argument("--sitl", action="store_true", help="使用 release260904 SITL 辨识的 0.28s/0.5s 滞后")
    parser.add_argument("--thrust-tau", type=float, default=None)
    parser.add_argument("--measurement-tau", type=float, default=None)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output", default="config/theory_altitude.json")
    args = parser.parse_args(argv)

    params = sitl_params() if args.sitl else AltitudePlantParams()
    if args.thrust_tau is not None:
        params.thrust_tau_s = args.thrust_tau
    if args.measurement_tau is not None:
        params.measurement_tau_s = args.measurement_tau

    gains = tune_altitude(params, duration=args.duration)
    metrics = evaluate_altitude(gains, params, duration=args.duration)
    out = {
        "structure": "firmware_pid",
        "gains": gains.as_dict(),
        "betaflight": gains.to_betaflight(),
        "validated_release_gains": SITL_FINAL_GAINS.to_cli(),
        "plant": asdict(params),
        "metrics": metrics,
        "note": (
            "betaflight 是理论整定值，validated_release_gains 是 release260904 "
            "已通过 SITL 回归的最终值；两者都使用固件真实缩放。"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
