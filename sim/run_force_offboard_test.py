#!/usr/bin/env python3
"""force-only world 的 offboard 参考轨迹验证。

force world 的 RotorForcePlugin 已验证能产生推力；本脚本按“先手动 RC
爬到安全高度，再切 MSP offboard 位置目标”的顺序，避开地面接触/初始
位置估计不准的问题，验证：

1. 参考轨迹 p_ref/v_ref/a_ref 是否连续；
2. 水平 offboard 是否真正产生运动；
3. 0.05 s 日志与 3D 轨迹图。

用法：
    # 终端 1
    BF_ANTI_WORLD=sim/worlds/bf_anti_intercept_force_flat.sdf \
        ./sim/launch_sitl.sh --headless
    # 终端 2
    python3 sim/run_force_offboard_test.py --cruise 3 --climb-alt 4 \
        --aim-north 15 --duration 15 --log /tmp/force_offboard.csv \
        --plot /tmp/force_offboard.png
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time
import types
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import GazeboPoseSource  # noqa: E402
from run_altitude_diag import (  # noqa: E402
    read_altitude,
    read_attitude,
    read_motors,
    read_position_state_ext,
    read_target,
    write_csv,
)

_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402


def manual_climb(link, rc, target_alt_m: float, timeout_s: float = 20.0) -> bool:
    """手动 RC 油门 PD 控制爬到目标高度，避免大超调。

    force world 的悬停油门约 1644；用 1644 作为基础，叠加高度/垂速 PD，
    比固定 1700 油门+到达后再收敛稳定得多。
    """
    rc._hold_throttle = 1642
    deadline = time.time() + timeout_s
    stable_since = None
    while time.time() < deadline:
        alt = read_altitude(link)
        if alt is None:
            time.sleep(0.05)
            continue
        alt_m, vario_mps = alt
        error = target_alt_m - alt_m
        throttle = 1642.0 + 20.0 * error - 60.0 * vario_mps
        rc._hold_throttle = int(max(1500, min(1720, throttle)))
        if abs(error) < 0.35:
            stable_since = stable_since or time.time()
            if time.time() - stable_since >= 1.0:
                return True
        else:
            stable_since = None
        time.sleep(0.05)
    return False


def hover_stabilize(link, rc, target_alt_m: float, timeout_s: float = 12.0) -> bool:
    """在目标高度附近用简单 P 控制把油门稳定住。"""
    deadline = time.time() + timeout_s
    stable_since = None
    while time.time() < deadline:
        alt = read_altitude(link)
        if alt is None:
            time.sleep(0.05)
            continue
        alt_m = alt[0]
        throttle = int(max(1500, min(1800, 1700 - 120.0 * (alt_m - target_alt_m))))
        rc._hold_throttle = throttle
        if abs(alt_m - target_alt_m) < 0.5:
            stable_since = stable_since or time.time()
            if time.time() - stable_since >= 1.0:
                return True
        else:
            stable_since = None
        time.sleep(0.05)
    return False


def run(args: argparse.Namespace) -> int:
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = offboard.RcLink()
    link = offboard.MspLink()
    running = [True]
    threading.Thread(target=offboard.rc_thread, args=(udp, rc, running), daemon=True).start()

    source = GazeboPoseSource(args.world, args.interceptor, args.target)
    rows: List[list] = []
    try:
        print("arming...", flush=True)
        if not offboard.arm(link, rc, timeout=60.0):
            print("ERROR: failed to arm", flush=True)
            return 2
        print("armed; manual RC climb...", flush=True)
        if not manual_climb(link, rc, args.climb_alt):
            print("WARNING: manual climb timeout; continuing", flush=True)
        # manual_climb 已经在目标高度附近停留过 1 s；不要再调用基于 1700
        # 油门的旧 hover_stabilize，否则会把机体重新推高。
        current_alt = read_altitude(link)
        aim_up = args.aim_up if args.aim_up >= 0.0 else (
            current_alt[0] if current_alt is not None and np.isfinite(current_alt[0])
            else args.climb_alt
        )
        print(f"switching to offboard aim=({args.aim_east:.1f},{args.aim_north:.1f},{aim_up:.1f})",
              flush=True)
        rc._hold_throttle = 1642  # 标定后的 force world 悬停值，offboard 会接管
        offboard.send_target(
            link,
            types.SimpleNamespace(cruise=args.cruise, timeout_ms=0),
            args.aim_east,
            args.aim_north,
            aim_up,
        )

        start = time.monotonic()
        period = max(args.period, 0.02)
        while time.monotonic() - start < args.duration:
            state = read_position_state_ext(link)
            if state is None:
                time.sleep(period)
                continue
            pos, vel, ref_pos, ref_vel, ref_acc = state
            attitude = read_attitude(link)
            altitude = read_altitude(link)
            motors = read_motors(link)
            target_cmd = read_target(link)
            gazebo = source.pose()
            truth = gazebo.own_position_enu if gazebo else (float("nan"),) * 3
            truth_rpy = gazebo.own_rpy_deg if gazebo else (float("nan"),) * 3
            rows.append([
                time.monotonic() - start,
                *pos, *vel,
                *ref_pos, *ref_vel, *ref_acc,
                *attitude,
                *altitude,
                *motors,
                *target_cmd[2], target_cmd[0], target_cmd[1],
                *truth, *truth_rpy,
                *((0,) * 8),
            ])
            if len(rows) % 10 == 0:
                print(
                    f"t={rows[-1][0]:5.1f} pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:5.2f}) "
                    f"ref=({ref_pos[0]:6.2f},{ref_pos[1]:6.2f},{ref_pos[2]:5.2f}) "
                    f"vref=({ref_vel[0]:5.2f},{ref_vel[1]:5.2f}) alt={altitude[0]:5.2f}",
                    flush=True,
                )
            time.sleep(period)
    finally:
        rc.idle()
        running[0] = False
        udp.close()
        if rows:
            write_csv(args.log, rows)
            print(f"wrote {args.log} ({len(rows)} samples)", flush=True)
            if args.plot:
                from plot_trajectory import plot_csv
                plot_csv(args.log, args.plot, title="force-world offboard reference")
                print(f"wrote {args.plot}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="force-only world offboard 参考轨迹验证")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--interceptor", default="iris_with_Betaflight")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--cruise", type=float, default=3.0)
    parser.add_argument("--climb-alt", type=float, default=4.0)
    parser.add_argument("--aim-east", type=float, default=0.0)
    parser.add_argument("--aim-north", type=float, default=15.0)
    parser.add_argument("--aim-up", type=float, default=-1.0,
                        help="offboard 目标高度；<0 表示保持切换时的当前高度")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--period", type=float, default=0.05)
    parser.add_argument("--log", default="/tmp/force_offboard.csv")
    parser.add_argument("--plot", default="/tmp/force_offboard.png")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
