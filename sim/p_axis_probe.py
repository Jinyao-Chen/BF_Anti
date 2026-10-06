#!/usr/bin/env python3
"""P 律执行层符号标定：RC 杆量 → 姿态 → 世界（ENU）加速度方向。

纯 P 视觉制导的每个通道符号都必须实测：姿态符号对了不代表世界加速度方向对
（44° 上仰相机 + LiftDrag 机体的推力方向与机头 x 轴的映射需要实测确认）。

流程：解锁 → 爬到 --climb-alt → 水平悬停 → 依次给 pitch / roll / yaw 杆量，
记录 MSP_ATTITUDE 与 Gazebo 真值位置/速度，输出每个通道的
"姿态变化 + 世界速度变化方向(方位角)"。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import socket
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import GazeboPoseSource  # noqa: E402
from run_intercept_mission import (  # noqa: E402
    MSP_ATTITUDE,
    RcState,
    climb_to,
    rc_sender,
    spool_up,
    wait_until_armed,
)

_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402

RC_MID = 1500


def read_attitude(link):
    import struct
    payload = link.query(MSP_ATTITUDE, is_v2=False)
    if payload and len(payload) >= 6:
        raw = struct.unpack_from("<3h", payload, 0)
        return tuple(float(v) / 10.0 for v in raw)
    return (float("nan"),) * 3


def az_deg(vec) -> float:
    return math.degrees(math.atan2(float(vec[0]), float(vec[1]))) % 360.0


def probe_stick(rc, source, link, axis: str, delta: int, seconds: float, rows: list) -> dict:
    """给一个轴的杆量，返回姿态与世界速度的变化。"""
    pose0 = None
    deadline = time.monotonic() + 0.6
    while time.monotonic() < deadline:
        pose0 = source.pose() or pose0
        time.sleep(0.05)
    att0 = read_attitude(link)
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        roll = RC_MID + (delta if axis == "roll" else 0)
        pitch = RC_MID + (delta if axis == "pitch" else 0)
        yaw = RC_MID + (delta if axis == "yaw" else 0)
        rc.set_rpyt(roll, pitch, int(rc.snapshot()[2]), yaw)
        pose = source.pose()
        att = read_attitude(link)
        rows.append([axis, time.monotonic() - t0, *att,
                     *(pose.own_position_enu if pose else (float("nan"),) * 3),
                     *(pose.own_velocity_enu if pose else (float("nan"),) * 3)])
        time.sleep(0.05)
    att1 = read_attitude(link)
    pose1 = source.pose()
    dv = (np.asarray(pose1.own_velocity_enu) - np.asarray(pose0.own_velocity_enu))
    datt = tuple(att1[i] - att0[i] for i in range(3))
    out = {
        "axis": axis, "delta_rc": delta,
        "d_roll": datt[0], "d_pitch": datt[1], "d_yaw": datt[2],
        "dv_e": float(dv[0]), "dv_n": float(dv[1]), "dv_u": float(dv[2]),
        "dv_az_deg": az_deg(dv) if float(np.linalg.norm(dv[:2])) > 0.3 else float("nan"),
        "dv_horiz": float(np.linalg.norm(dv[:2])),
        "yaw0": att0[2],
    }
    print(f"  [{axis:>5s}] Δatt(r,p,y)=({datt[0]:+6.1f},{datt[1]:+6.1f},{datt[2]:+6.1f})  "
          f"Δv_ENU=({dv[0]:+5.2f},{dv[1]:+5.2f},{dv[2]:+5.2f}) "
          f"az={out['dv_az_deg']:.0f}° |dv|={out['dv_horiz']:.2f} yaw0={att0[2]:.0f}", flush=True)
    # 回中，等姿态稳下来
    rc.set_rpyt(RC_MID, RC_MID, int(rc.snapshot()[2]), RC_MID)
    time.sleep(1.5)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="RC 轴 → 姿态 → 世界加速度 标定")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--interceptor", default="iris_with_Betaflight")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--climb-alt", type=float, default=5.0)
    parser.add_argument("--delta", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=2.5)
    parser.add_argument("--throttle", type=int, default=1670, help="悬停油门 PWM（轻机体 ~1410）")
    parser.add_argument("--log", default="/tmp/p_axis_probe.csv")
    args = parser.parse_args()

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = RcState()
    running = [True]
    threading.Thread(target=rc_sender, args=(udp, rc, running), daemon=True).start()
    link = offboard.MspLink()
    source = GazeboPoseSource(args.world, args.interceptor, args.target)
    rows: list[list] = []
    results = []
    try:
        print("arming...", flush=True)
        if not wait_until_armed(link, rc):
            print("ERROR: 解锁失败", flush=True)
            return 2
        spool_up(rc, target=args.throttle, duration=2.0)
        print(f"climb to {args.climb_alt} m ...", flush=True)
        if not climb_to(rc, source, link, args.climb_alt, timeout=25.0):
            print("WARNING: climb timeout", flush=True)
        rc.set_rpyt(RC_MID, RC_MID, args.throttle, RC_MID)
        time.sleep(1.0)
        print(f"hover att={read_attitude(link)}", flush=True)
        for axis in ("pitch", "roll", "yaw"):
            results.append(probe_stick(rc, source, link, axis, args.delta, args.seconds, rows))
    finally:
        rc.set_throttle(1000)
        rc.set_aux1(1000)
        time.sleep(0.5)
        running[0] = False
        udp.close()
        if rows:
            with open(args.log, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["axis", "t", "roll", "pitch", "yaw",
                                 "pos_e", "pos_n", "pos_u", "vel_e", "vel_n", "vel_u"])
                writer.writerows(rows)
            print(f"wrote {args.log} ({len(rows)} rows)", flush=True)
    print("=== axis probe summary (rc +%d = 通道值 %d) ===" % (args.delta, RC_MID + args.delta))
    for r in results:
        print(f"  {r['axis']:<6} Δatt=({r['d_roll']:+6.1f},{r['d_pitch']:+6.1f},{r['d_yaw']:+6.1f}) "
              f"Δv_az={r['dv_az_deg']:.0f}° |Δv|={r['dv_horiz']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
