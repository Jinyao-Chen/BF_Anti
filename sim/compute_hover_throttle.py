#!/usr/bin/env python3
"""由质量与旋翼参数计算悬停油门，避免 SITL 配置手算出错。

用法::

    python3 sim/compute_hover_throttle.py --mass 1.75 \
        --motor-constant 8.54858e-06 --max-omega 1100

输出归一化命令与 Betaflight PWM（默认 1000..2000 线性）。
"""

from __future__ import annotations

import argparse
import math

G = 9.80665


def compute_hover_throttle(
    mass_kg: float,
    motor_constant: float,
    max_omega: float,
    motor_count: int = 4,
    pwm_min: float = 1000.0,
    pwm_max: float = 2000.0,
) -> tuple[float, float, float]:
    """返回 (omega_rad_s, normalized_cmd, pwm)。"""
    if mass_kg <= 0.0 or motor_constant <= 0.0 or max_omega <= 0.0:
        raise ValueError("mass, motor_constant and max_omega must be positive")
    omega = math.sqrt(mass_kg * G / (motor_count * motor_constant))
    normalized = omega / max_omega
    pwm = pwm_min + normalized * (pwm_max - pwm_min)
    return omega, normalized, pwm


def main() -> int:
    parser = argparse.ArgumentParser(description="计算悬停油门")
    parser.add_argument("--mass", type=float, required=True, help="总质量 kg")
    parser.add_argument("--motor-constant", type=float, required=True,
                        help="推力系数 cT，thrust = cT * omega^2 N")
    parser.add_argument("--max-omega", type=float, required=True, help="最大转速 rad/s")
    parser.add_argument("--motors", type=int, default=4)
    parser.add_argument("--pwm-min", type=float, default=1000.0)
    parser.add_argument("--pwm-max", type=float, default=2000.0)
    args = parser.parse_args()

    omega, normalized, pwm = compute_hover_throttle(
        args.mass, args.motor_constant, args.max_omega, args.motors,
        args.pwm_min, args.pwm_max,
    )
    print(f"hover omega      : {omega:.2f} rad/s")
    print(f"normalized cmd    : {normalized:.4f}")
    print(f"Betaflight PWM    : {pwm:.1f}")
    print(f"CLI               : set ap_hover_throttle = {pwm:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
