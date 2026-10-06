#!/usr/bin/env python3
"""固定靶机 + 简化质点飞机模型的拦截制导闭环验证。

这个脚本不依赖 Gazebo，用来先验证 GuidanceCommand -> TrackCmd 的
制导逻辑能否把一架简化飞机引导到固定靶机附近。Gazebo 接入是下一步。

用法:
  python3 sim/guidance_kinematic.py --target-north 30 --duration 30
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guidance.intercept_guidance import InterceptGuidance, los_az_el_from_relative  # noqa: E402
from perception.target_estimator import TargetState  # noqa: E402

G = 9.80665


def wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-east", type=float, default=0.0)
    ap.add_argument("--target-north", type=float, default=30.0)
    ap.add_argument("--target-up", type=float, default=0.0)
    ap.add_argument("--start-east", type=float, default=0.0)
    ap.add_argument("--start-north", type=float, default=0.0)
    ap.add_argument("--start-up", type=float, default=0.0)
    ap.add_argument("--start-yaw", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--hit-radius", type=float, default=0.5)
    ap.add_argument("--max-yaw-rate-dps", type=float, default=120.0)
    ap.add_argument("--max-pitch-rate-dps", type=float, default=180.0)
    ap.add_argument("--drag", type=float, default=0.015)
    ap.add_argument("--log", default=None, help="可选 CSV 输出路径")
    args = ap.parse_args()

    guidance = InterceptGuidance(camera_tilt_deg=44.0)
    target_pos = np.array([args.target_east, args.target_north, args.target_up], dtype=float)

    own_pos = np.array([args.start_east, args.start_north, args.start_up], dtype=float)
    own_yaw = float(args.start_yaw)
    own_pitch = 0.0
    speed = 0.0

    t = 0.0
    min_range = float("inf")
    hit_time = None
    phase_history = []
    writer = None
    if args.log:
        writer = open(args.log, "w")
        writer.write("t,east,north,up,yaw,pitch,speed,range,phase,pitch_cmd,yaw_cmd,throttle_cmd\n")

    try:
        while t < args.duration:
            rel = target_pos - own_pos
            rng = float(np.linalg.norm(rel))
            az, el = los_az_el_from_relative(rel)
            vel = np.array(
                [speed * math.sin(math.radians(own_yaw)),
                 speed * math.cos(math.radians(own_yaw)),
                 0.0]
            )
            target = TargetState(
                position_enu=target_pos.copy(),
                velocity_enu=np.zeros(3),
                range_m=rng,
                az_deg=az,
                el_deg=el,
                timestamp=t,
            )
            cmd = guidance.update(target, own_pos, vel, timestamp=t, dt=args.dt)
            phase_history.append(cmd.phase.value)

            min_range = min(min_range, rng)
            if writer is not None:
                writer.write(
                    f"{t:.3f},{own_pos[0]:.4f},{own_pos[1]:.4f},{own_pos[2]:.4f},"
                    f"{own_yaw:.3f},{own_pitch:.3f},{speed:.3f},{rng:.4f},"
                    f"{cmd.phase.value},{cmd.pitch_cmd_deg:.3f},"
                    f"{cmd.yaw_cmd_deg:.3f},{cmd.throttle_cmd:.3f}\n"
                )

            if rng <= args.hit_radius:
                hit_time = t
                break

            # 简化姿态响应：一阶跟踪制导姿态指令
            yaw_err = wrap180(cmd.yaw_cmd_deg - own_yaw)
            own_yaw += clamp(yaw_err, -args.max_yaw_rate_dps * args.dt,
                             args.max_yaw_rate_dps * args.dt)
            own_yaw = wrap180(own_yaw)

            pitch_err = cmd.pitch_cmd_deg - own_pitch
            own_pitch += clamp(pitch_err, -args.max_pitch_rate_dps * args.dt,
                               args.max_pitch_rate_dps * args.dt)

            # 简化质点纵向动力学：俯冲产生前向加速度，阻力随速度增加
            a_fwd = -G * math.tan(math.radians(own_pitch)) - args.drag * speed * abs(speed)
            speed = max(0.0, speed + a_fwd * args.dt)

            own_pos = own_pos + np.array(
                [speed * math.sin(math.radians(own_yaw)),
                 speed * math.cos(math.radians(own_yaw)),
                 0.0]
            ) * args.dt
            t += args.dt

    finally:
        if writer is not None:
            writer.close()

    print("=== guidance kinematic result ===")
    print(f"final range : {min_range:.3f} m")
    print(f"hit/time    : {hit_time if hit_time is not None else 'no hit'}")
    print(f"final pos   : {own_pos}")
    print(f"final yaw   : {own_yaw:.1f} deg")
    print(f"final pitch : {own_pitch:.1f} deg")
    print(f"final speed : {speed:.2f} m/s")
    print(f"phases      : {' -> '.join(dict.fromkeys(phase_history))}")
    if args.log:
        print(f"log         : {args.log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
