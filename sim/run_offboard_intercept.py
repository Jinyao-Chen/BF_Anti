#!/usr/bin/env python3
"""固定靶机 offboard 闭环拦截验证。

使用 release260904 已经验证过的 MSP2 offboard 位置控制器，把固定靶机位置
作为最终位置目标，飞过去并做命中判定。相比直接 RC 姿态控制，这条链路在
SITL 中已经稳定，适合作为“可复现的拦截基线”。

流程：
  1. 复用官方 offboard_position_example 的 arming / RC 线程；
  2. 从 Gazebo 读取固定靶机 ENU 位置；
  3. 先垂直爬升到安全高度；
  4. 发送靶机位置为目标，cruise 可配置；
  5. 轮询 MSP2_BETAFLIGHT_POSITION_STATE，距离 < hit_radius 判定命中。
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import struct
import sys
import threading
import time
import types
from typing import Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import GazeboPoseSource  # noqa: E402

_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402

MSP2_POSITION_STATE = 0x3015


def read_position_state(link) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """读取固件实时 ENU 位置/速度（米、米/秒）。"""
    payload = link.query(MSP2_POSITION_STATE, is_v2=True)
    if not payload or len(payload) < 27:
        return None
    # u8 active, u8 reached, u8 offboard_velocity_active + 3 floats position
    pos = np.array(struct.unpack_from("<3f", payload, 3), dtype=float)
    vel = np.array(struct.unpack_from("<3f", payload, 15), dtype=float)
    return pos, vel


def wait_for_target_pose(source: GazeboPoseSource, timeout: float = 10.0) -> Optional[np.ndarray]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        pose = source.pose()
        if pose is not None:
            return pose.target_position_enu.copy()
        time.sleep(0.1)
    return None


def climb_to(
    link,
    rc,
    target_alt: float,
    timeout: float = 60.0,
) -> bool:
    """先垂直爬升到 target_alt，并等到高度稳定后再切水平目标。

    只判断“当前高度超过目标”会在高速爬升中提前放行，随后叠加水平目标会
    造成高度环大超调和积分饱和；这里改成 reached 或连续 1.5 s 在 ±0.3 m
    带内才返回。
    """
    setpoint = types.SimpleNamespace(cruise=1.0, timeout_ms=0)
    offboard.send_target(link, setpoint, 0.0, 0.0, target_alt)
    start = time.time()
    stable_since = None
    while time.time() - start < timeout:
        state = offboard.read_target_state(link)
        alt = offboard.read_altitude(link)
        if state is not None:
            _, reached, _ = state
            if reached:
                return True
        if alt is not None and abs(alt - target_alt) < 0.3:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= 1.5:
                return True
        else:
            stable_since = None
        time.sleep(0.2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="固定靶机 offboard 闭环拦截验证")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--target-width", type=float, default=1.0)
    parser.add_argument("--cruise", type=float, default=2.0, help="水平巡航速度 m/s")
    parser.add_argument("--climb-alt", type=float, default=10.0, help="先爬升到的安全高度")
    parser.add_argument("--hit-radius", type=float, default=1.0, help="命中判定半径 m")
    parser.add_argument(
        "--beyond",
        type=float,
        default=3.0,
        help="把 offboard 目标点放到靶机后方多少米，避免在 1.5 m 到达区提前刹停",
    )
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = offboard.RcLink()
    link = offboard.MspLink()
    running = [True]
    rc_thread = threading.Thread(target=offboard.rc_thread, args=(udp, rc, running), daemon=True)
    rc_thread.start()

    source = GazeboPoseSource(args.world, "iris_with_Betaflight", args.target)
    target_enu = wait_for_target_pose(source)
    if target_enu is None:
        print("ERROR: could not read target_drone pose from Gazebo", flush=True)
        running[0] = False
        return 2
    print(f"target ENU: ({target_enu[0]:.2f}, {target_enu[1]:.2f}, {target_enu[2]:.2f}) m", flush=True)

    rows = []
    hit = False
    min_dist = float("inf")
    try:
        print("arming...", flush=True)
        if not offboard.arm(link, rc, timeout=40.0):
            print("ERROR: failed to arm", flush=True)
            return 3
        print("armed; spooling up", flush=True)
        rc.start_spool()
        time.sleep(1.0)
        print(f"climbing to {args.climb_alt:.1f} m ...", flush=True)
        if not climb_to(link, rc, args.climb_alt):
            print("WARNING: climb did not reach setpoint; continuing", flush=True)

        current = read_position_state(link)
        own_now = np.zeros(3) if current is None else current[0]
        direction = target_enu - own_now
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm < 1e-6:
            direction = np.array([0.0, 1.0, 0.0])
        else:
            direction = direction / direction_norm
        # 目标点放到靶机后方，保证飞行器会“穿过”靶机而不是在到达区提前刹停。
        aim_point = target_enu + direction * float(args.beyond)
        print(
            f"moving through target at cruise={args.cruise:.1f} m/s; "
            f"aim=({aim_point[0]:.2f},{aim_point[1]:.2f},{aim_point[2]:.2f})",
            flush=True,
        )
        setpoint = types.SimpleNamespace(cruise=args.cruise, timeout_ms=0)
        offboard.send_target(link, setpoint, float(aim_point[0]), float(aim_point[1]), float(aim_point[2]))

        start = time.monotonic()
        closest = float("inf")
        increasing_count = 0
        while time.monotonic() - start < args.duration:
            state = read_position_state(link)
            if state is None:
                time.sleep(0.1)
                continue
            pos, vel = state
            dist = float(np.linalg.norm(pos - target_enu))
            min_dist = min(min_dist, dist)
            if dist > closest + 0.02:
                increasing_count += 1
            else:
                increasing_count = 0
            closest = min(closest, dist)
            rows.append((time.monotonic() - start, *pos, *vel, dist))
            print(
                f"t={time.monotonic() - start:5.1f} pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:5.2f}) "
                f"dist={dist:5.2f}m",
                flush=True,
            )
            if dist <= args.hit_radius:
                hit = True
                break
            # 已经穿过靶机且距离开始持续增大：停止等待，按最小距离评估。
            if increasing_count >= 5 and min_dist < args.hit_radius + 2.0:
                break
            time.sleep(0.2)
    finally:
        rc.idle()
        running[0] = False
        rc_thread.join(timeout=1.0)
        udp.close()
        if args.log and rows:
            with open(args.log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t", "east", "north", "up", "ve", "vn", "vu", "distance_m"])
                writer.writerows(rows)
            print(f"wrote {args.log}", flush=True)

    print("=== offboard intercept summary ===")
    print(f"hit       : {hit}")
    print(f"min dist  : {min_dist:.3f} m")
    print(f"cruise    : {args.cruise:.2f} m/s")
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
