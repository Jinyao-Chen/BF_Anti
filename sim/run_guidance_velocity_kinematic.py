#!/usr/bin/env python3
"""制导 + 伴机侧速度外环：轻量级运动学闭环验证。

不依赖 Gazebo/YOLO，用带一阶加速度滞后的三维双积分模型验证：

1. `InterceptGuidance`（3D APN + 提前量）输出 ENU 加速度指令；
2. `XYVelocityController` 把加速度限幅并转成 roll/pitch（本脚本只取限幅后的
   加速度，用于运动学积分）；
3. 目标按场景运动（对头/侧向/尾追/机动），统计脱靶量、命中时间与加速度需求。

这一层用于在接入视觉之前把制导律与执行链的量级、符号、限幅先跑对。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from guidance.intercept_guidance import InterceptGuidance  # noqa: E402
from perception.target_estimator import TargetState  # noqa: E402

G = 9.80665


@dataclass
class Scenario:
    name: str
    own0: np.ndarray
    own_vel0: np.ndarray
    tgt0: np.ndarray
    tgt_vel0: np.ndarray
    tgt_accel0: np.ndarray
    weave_amp_mps: float = 0.0   # 侧向正弦机动幅值（E 轴），0=不机动
    weave_omega: float = 0.6     # 机动角频率 rad/s


def make_scenarios() -> list[Scenario]:
    return [
        Scenario("head-on", np.array([0.0, 0.0, 10.0]), np.array([0.0, 3.0, 0.0]),
                 np.array([0.0, 80.0, 12.0]), np.array([0.0, -6.0, 0.0]), np.zeros(3)),
        Scenario("crossing", np.array([0.0, 0.0, 10.0]), np.array([2.0, 0.0, 0.0]),
                 np.array([60.0, 60.0, 12.0]), np.array([-6.0, 0.0, 0.0]), np.zeros(3)),
        Scenario("tail-chase", np.array([0.0, 0.0, 10.0]), np.array([0.0, 4.0, 0.0]),
                 np.array([0.0, 40.0, 12.0]), np.array([0.0, 6.0, 0.0]), np.zeros(3)),
        Scenario("maneuvering", np.array([0.0, 0.0, 10.0]), np.array([0.0, 2.0, 0.0]),
                 np.array([0.0, 70.0, 12.0]), np.array([0.0, -5.0, 0.0]),
                 np.zeros(3), weave_amp_mps=4.0, weave_omega=0.6),
    ]


def run_scenario(sc: Scenario, args: argparse.Namespace) -> dict:
    guidance = InterceptGuidance(
        camera_tilt_deg=44.0,
        approach_range_m=args.approach_m,
        terminal_range_m=args.terminal_m,
        nav_constant=args.nav_constant,
        max_speed_mps=args.max_speed,
        max_terminal_speed_mps=args.max_terminal_speed,
        max_accel_mps2=args.max_accel,
        velocity_gain=args.velocity_gain,
        speed_schedule=args.speed_schedule,
        hit_radius_m=args.hit_radius,
        min_safe_altitude_m=0.0,
    )

    own_pos = sc.own0.astype(float).copy()
    own_vel = sc.own_vel0.astype(float).copy()
    tgt_pos = sc.tgt0.astype(float).copy()
    tgt_vel = sc.tgt_vel0.astype(float).copy()
    a_real = np.zeros(3)
    rows: list[list] = []
    min_range = float("inf")
    hit_time: Optional[float] = None

    dt = args.dt
    steps = int(args.duration / dt)
    for k in range(steps):
        t = k * dt
        # 目标运动：常加速度 或 侧向正弦机动（速度有界，更接近真实规避）。
        if sc.weave_amp_mps > 0.0:
            weave_v = sc.weave_amp_mps * math.sin(sc.weave_omega * t)
            weave_a = sc.weave_amp_mps * sc.weave_omega * math.cos(sc.weave_omega * t)
            tgt_vel = sc.tgt_vel0 + np.array([weave_v, 0.0, 0.0])
            cur_tgt_accel = np.array([weave_a, 0.0, 0.0])
            tgt_pos = sc.tgt0 + sc.tgt_vel0 * t + np.array([
                sc.weave_amp_mps / max(sc.weave_omega, 1e-3) * (1.0 - math.cos(sc.weave_omega * t)),
                0.0, 0.0])
        else:
            tgt_vel = tgt_vel + sc.tgt_accel0 * dt
            tgt_pos = tgt_pos + tgt_vel * dt
            cur_tgt_accel = sc.tgt_accel0
        rel = tgt_pos - own_pos
        rng = float(np.linalg.norm(rel))
        min_range = min(min_range, rng)

        target = TargetState(
            position_enu=tgt_pos.copy(),
            velocity_enu=tgt_vel.copy(),
            range_m=rng,
            az_deg=float(np.degrees(math.atan2(rel[0], rel[1])) % 360.0),
            el_deg=float(np.degrees(math.atan2(rel[2], math.hypot(rel[0], rel[1])))),
            timestamp=t,
        )
        gcmd = guidance.update(target, own_pos, own_vel, timestamp=t, dt=dt,
                               target_accel_enu=cur_tgt_accel)
        a_cmd = np.array(gcmd.accel_enu, dtype=float)
        # 执行链：一阶加速度滞后 + 倾角限幅（用控制器换算出的最大可用加速度）。
        a_cmd = np.clip(a_cmd, -args.max_accel, args.max_accel)
        a_real = a_real + (a_cmd - a_real) * min(1.0, dt / args.actuator_tau)
        own_vel = own_vel + a_real * dt
        own_pos = own_pos + own_vel * dt

        rows.append([
            t, *own_pos, *own_vel, *tgt_pos, *tgt_vel, rng,
            float(gcmd.desired_speed_mps), float(gcmd.t_go_s),
            int(gcmd.hit), float(np.linalg.norm(a_cmd)),
        ])
        if rng <= args.hit_radius and hit_time is None:
            hit_time = t
            break
        if own_pos[2] < 0.0:
            break

    return {
        "name": sc.name,
        "hit": hit_time is not None,
        "hit_time": hit_time,
        "min_range": min_range,
        "rows": rows,
        "steps": len(rows) * dt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="制导+速度外环运动学闭环")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--nav-constant", type=float, default=4.0)
    parser.add_argument("--velocity-gain", type=float, default=1.6)
    parser.add_argument("--speed-schedule", choices=("proportional","constant"), default="proportional")
    parser.add_argument("--max-speed", type=float, default=12.0)
    parser.add_argument("--max-terminal-speed", type=float, default=18.0)
    parser.add_argument("--max-accel", type=float, default=12.0)
    parser.add_argument("--max-angle", type=float, default=60.0)
    parser.add_argument("--approach-m", type=float, default=25.0)
    parser.add_argument("--terminal-m", type=float, default=3.0)
    parser.add_argument("--hit-radius", type=float, default=0.8)
    parser.add_argument("--actuator-tau", type=float, default=0.25)
    parser.add_argument("--log", default="/tmp/guidance_kin.csv")
    return run_all(parser.parse_args())


def run_all(args: argparse.Namespace) -> int:
    all_rows: list[list] = []
    print(f"{'scenario':<12} {'hit':<5} {'t_hit':>7} {'miss':>8} {'v_end':>7}")
    for sc in make_scenarios():
        res = run_scenario(sc, args)
        print(
            f"{res['name']:<12} {str(res['hit']):<5} "
            f"{(res['hit_time'] if res['hit_time'] is not None else float('nan')):7.2f} "
            f"{res['min_range']:8.3f} "
            f"{res['rows'][-1][6] if res['rows'] else float('nan'):7.2f}"
        )
        for row in res["rows"]:
            all_rows.append([res["name"], *row])
    if all_rows:
        with open(args.log, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "scenario", "t", "own_e", "own_n", "own_u", "own_ve", "own_vn", "own_vu",
                "tgt_e", "tgt_n", "tgt_u", "tgt_ve", "tgt_vn", "tgt_vu",
                "range", "v_des", "t_go", "hit", "a_cmd",
            ])
            writer.writerows(all_rows)
        print(f"wrote {args.log} ({len(all_rows)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
