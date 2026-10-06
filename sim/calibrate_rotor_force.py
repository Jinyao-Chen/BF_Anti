#!/usr/bin/env python3
"""RotorForcePlugin 推力标定：由机体加速度反推实际悬停命令。

用一个单 link、无控制器的标定 world（质量/参数与 force-only Iris 一致），
直接向 `/calib/telemetry` 发布固定归一化电机命令，记录 z(t)，二次拟合得到
垂向加速度 a(cmd)。再线性拟合 a(cmd) = 0 得到实际悬停命令，输出对应 PWM。

用法::

    python3 sim/calibrate_rotor_force.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))

import gz.transport13 as gzt  # noqa: E402
from gz.msgs10.float_v_pb2 import Float_V  # noqa: E402
from gz.msgs10.pose_v_pb2 import Pose_V  # noqa: E402

G = 9.80665


def fit_accel(times: np.ndarray, heights: np.ndarray) -> float:
    """z = c0 + c1*t + 0.5*a*t^2 的最小二乘 a。"""
    if len(times) < 8:
        return float("nan")
    t0 = times - times[0]
    d = t0 * t0 / 2.0
    A = np.column_stack([np.ones_like(t0), t0, d])
    coeff, *_ = np.linalg.lstsq(A, heights, rcond=None)
    return float(coeff[2])


def run_calibration(args: argparse.Namespace) -> int:
    world = os.path.join(ROOT, "sim", "worlds", "rotor_force_calib.sdf")
    plugin_lib = os.path.expanduser(
        "~/betaflight/src/platform/SIMULATOR/gazebo/build/lib"
    )
    env = os.environ.copy()
    env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = plugin_lib + ":" + env.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")
    proc = subprocess.Popen(
        ["gz", "sim", "-r", "-s", "-v", "2", world],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3.0)
        node = gzt.Node()
        pub = node.advertise("/calib/telemetry", Float_V)
        poses: list[tuple[float, float]] = []
        msg_holder = {"msg": None}

        def on_pose(msg: Pose_V) -> None:
            for p in msg.pose:
                if p.name == "test_body":
                    poses.append((time.perf_counter(), float(p.position.z)))
                    msg_holder["msg"] = msg
                    return

        if not node.subscribe(Pose_V, "/world/rotor_force_calib/pose/info", on_pose):
            raise RuntimeError("failed to subscribe pose/info")
        time.sleep(0.5)

        results = []
        for cmd in args.commands:
            msg = Float_V()
            msg.data.extend([cmd, cmd, cmd, cmd, 0.0, 0.0, 0.0, 0.0])
            # 先稳定 0.3 s，再记录 0.8 s
            deadline = time.perf_counter() + 0.3
            while time.perf_counter() < deadline:
                pub.publish(msg)
                time.sleep(0.01)
            poses.clear()
            t_start = time.perf_counter()
            deadline = t_start + 0.8
            while time.perf_counter() < deadline:
                pub.publish(msg)
                time.sleep(0.01)
            data = np.array([(t - t_start, z) for t, z in poses if t >= t_start], dtype=float)
            if len(data) < 8:
                print(f"cmd={cmd:.3f} no pose samples", flush=True)
                continue
            accel = fit_accel(data[:, 0], data[:, 1])
            results.append((cmd, accel))
            print(f"cmd={cmd:.3f} samples={len(data)} z0={data[0,1]:.3f} z1={data[-1,1]:.3f} accel={accel:+.3f} m/s^2",
                  flush=True)

        if len(results) >= 2:
            cmds = np.array([r[0] for r in results])
            accels = np.array([r[1] for r in results])
            slope, intercept = np.polyfit(cmds, accels, 1)
            hover_cmd = -intercept / slope if abs(slope) > 1e-9 else float("nan")
            hover_pwm = 1000.0 + 1000.0 * hover_cmd
            print("=== calibration fit ===", flush=True)
            print(f"a(cmd) = {slope:.3f} * cmd + {intercept:.3f}", flush=True)
            print(f"hover cmd = {hover_cmd:.4f}", flush=True)
            print(f"hover PWM = {hover_pwm:.1f}", flush=True)
            print(f"set ap_hover_throttle = {hover_pwm:.0f}", flush=True)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="RotorForcePlugin 推力标定")
    parser.add_argument(
        "--commands",
        type=float,
        nargs="+",
        default=[0.58, 0.60, 0.62, 0.64, 0.66, 0.68],
    )
    return run_calibration(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
