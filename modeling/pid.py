"""离散 PID：抗积分饱和、微分低通滤波、输出限幅。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PIDGains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0

    def as_tuple(self) -> Tuple[float, float, float]:
        return (float(self.kp), float(self.ki), float(self.kd))


def _clamp(value: float, limits: Tuple[float, float]) -> float:
    lower, upper = limits
    return max(lower, min(upper, value))


class PID:
    """并联式 PID。

    * 积分项向前欧拉，带独立限幅；
    * 输出饱和时做反算抗饱和，保证积分不会继续累积；
    * 微分项对测量值求导并做一阶低通，避免设定值阶跃冲击。
    """

    def __init__(
        self,
        gains: PIDGains,
        dt: float,
        output_limits: Tuple[float, float] = (-np.inf, np.inf),
        integral_limits: Tuple[float, float] = (-np.inf, np.inf),
        derivative_tau: float = 0.02,
        setpoint_weight: float = 1.0,
    ) -> None:
        self.gains = gains
        self.dt = float(dt)
        self.output_limits = output_limits
        self.integral_limits = integral_limits
        self.derivative_tau = max(float(derivative_tau), 1e-6)
        self.setpoint_weight = float(setpoint_weight)
        self.integral = 0.0
        self.prev_measurement: Optional[float] = None
        self.derivative_state = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_measurement = None
        self.derivative_state = 0.0

    def update(self, setpoint: float, measurement: float) -> float:
        error = float(setpoint) - float(measurement)
        p_term = self.gains.kp * error

        # 微分：对测量值求导，一阶低通。
        if self.prev_measurement is None:
            raw_d = 0.0
        else:
            raw_d = -(float(measurement) - self.prev_measurement) / self.dt
        alpha = self.dt / (self.derivative_tau + self.dt)
        self.derivative_state += alpha * (raw_d - self.derivative_state)
        d_term = self.gains.kd * self.derivative_state
        self.prev_measurement = float(measurement)

        # 积分：向前欧拉 + 限幅。
        self.integral = _clamp(
            self.integral + self.gains.ki * error * self.dt,
            self.integral_limits,
        )
        unsat = p_term + self.integral + d_term
        saturated = _clamp(unsat, self.output_limits)

        # 反算抗饱和：把积分拉回，使输出刚好等于饱和值。
        if saturated != unsat:
            self.integral = _clamp(
                saturated - p_term - d_term,
                self.integral_limits,
            )
            saturated = _clamp(p_term + self.integral + d_term, self.output_limits)
        return float(saturated)
