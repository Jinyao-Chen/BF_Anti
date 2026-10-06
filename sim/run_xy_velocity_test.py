#!/usr/bin/env python3
"""伴机侧 XY 速度外环 SITL 验证。

用法::

    # 终端 1
    BF_ROOT=/tmp/bf_merge ./sim/launch_sitl.sh --headless
    # 终端 2
    python3 sim/run_xy_velocity_test.py --step-e 2.0 --climb-alt 10 \
        --log /tmp/xy_vel.csv --plot /tmp/xy_vel.png

固件侧要求 `ap_offboard_xy_enable = OFF`（默认）：offboard 目标只驱动
ALT_HOLD 垂直通道，XY 由本脚本通过 RC roll/pitch 控制。
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
from typing import List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from guidance.xy_velocity_controller import XYVelocityController  # noqa: E402

_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402

MSP_ATTITUDE = 108
MSP2_POSITION_STATE = 0x3015
RC_MID = 1500


def read_state(link) -> Optional[dict]:
    """读固件 ENU 位置/速度与姿态。"""
    payload = link.query(MSP2_POSITION_STATE, is_v2=True)
    out: dict = {}
    if payload and len(payload) >= 27:
        out["pos"] = np.array(struct.unpack_from("<3f", payload, 3), dtype=float)
        out["vel"] = np.array(struct.unpack_from("<3f", payload, 15), dtype=float)
    att = link.query(MSP_ATTITUDE, is_v2=False)
    if att and len(att) >= 6:
        r, p, y = struct.unpack_from("<3h", att, 0)
        out["att"] = (r / 10.0, p / 10.0, y / 10.0)
    return out or None


def main() -> int:
    parser = argparse.ArgumentParser(description="伴机侧 XY 速度外环验证")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--climb-alt", type=float, default=10.0)
    parser.add_argument("--step-e", type=float, default=2.0)
    parser.add_argument("--step-n", type=float, default=0.0)
    parser.add_argument("--hover-s", type=float, default=8.0)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--kp-pos", type=float, default=0.8)
    parser.add_argument("--ki-pos", type=float, default=0.0)
    parser.add_argument("--shape", choices=("step", "ramp", "circle"), default="step")
    parser.add_argument("--ramp-speed", type=float, default=1.5, help="ramp 目标速度 m/s")
    parser.add_argument("--ramp-dir", type=float, default=0.0, help="ramp 方向，罗盘角 0=北")
    parser.add_argument("--radius", type=float, default=3.0, help="circle 半径 m")
    parser.add_argument("--omega", type=float, default=0.5, help="circle 角速度 rad/s")
    parser.add_argument("--kp-vel", type=float, default=1.5)
    parser.add_argument("--max-vel", type=float, default=5.0)
    parser.add_argument("--max-accel", type=float, default=4.0)
    parser.add_argument("--max-angle", type=float, default=25.0)
    parser.add_argument("--roll-sign", type=float, default=1.0)
    parser.add_argument("--pitch-sign", type=float, default=1.0)
    parser.add_argument("--yaw-from-velocity", action="store_true")
    parser.add_argument("--log", default="/tmp/xy_vel.csv")
    parser.add_argument("--plot", default="")
    args = parser.parse_args()

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = offboard.RcLink()
    running = [True]
    rc_thread = threading.Thread(target=offboard.rc_thread, args=(udp, rc, running), daemon=True)
    rc_thread.start()
    link = offboard.MspLink()

    print("arming...", flush=True)
    if not offboard.arm(link, rc, timeout=40.0):
        print("ERROR: failed to arm", flush=True)
        return 2
    print("armed; switching to companion XY RC stream", flush=True)

    # 停止 offboard 的 throttle+AUX1 发送线程，改用全通道发送器。
    running[0] = False
    rc_thread.join(timeout=1.0)
    channels = [RC_MID] * 16
    channels[2] = 1545          # ALT_HOLD 悬停油门中心
    channels[4] = 2000          # AUX1 / ARM
    channels[5] = 2000          # AUX2 / ALT_HOLD (merged box id 4)
    channels[6] = 1000          # AUX3 unused
    channels[7] = 2000          # AUX4 / ANGLE (merged box id 1)
    running[0] = True

    def sender() -> None:
        while running[0]:
            udp.sendto(
                struct.pack("<d" + "H" * 16, time.time(), *channels),
                (offboard.UDP_IP, offboard.UDP_PORT),
            )
            time.sleep(0.02)

    threading.Thread(target=sender, daemon=True).start()

    # offboard 目标只用于 ALT_HOLD 的高度指令（固件 XY 已关闭）。
    offboard.send_target(
        link, types.SimpleNamespace(cruise=1.0, timeout_ms=0), 0.0, 0.0, args.climb_alt
    )
    print(f"climbing to {args.climb_alt:.1f} m (ALT_HOLD, companion XY idle)...", flush=True)
    time.sleep(args.hover_s)

    state = read_state(link)
    if state is None:
        print("ERROR: no position state", flush=True)
        running[0] = False
        return 3
    base = state["pos"].copy()
    target = base.copy()
    target[0] += args.step_e
    target[1] += args.step_n
    target[2] = float(args.climb_alt)
    print(f"hover base={np.round(base, 2).tolist()} -> step target={np.round(target, 2).tolist()}", flush=True)

    ctrl = XYVelocityController(
        kp_pos_mps_per_m=args.kp_pos,
        ki_pos_mps2_per_m=args.ki_pos,
        kp_vel_mps2_per_mps=args.kp_vel,
        max_vel_mps=args.max_vel,
        max_accel_mps2=args.max_accel,
        max_angle_deg=args.max_angle,
        roll_sign=args.roll_sign,
        pitch_sign=args.pitch_sign,
        yaw_from_velocity=args.yaw_from_velocity,
    )

    rows: List[list] = []
    period = 1.0 / max(1.0, args.rate)
    start = time.monotonic()
    last = start
    while time.monotonic() - start < args.duration:
        loop_start = time.monotonic()
        state = read_state(link)
        if state is None:
            time.sleep(period)
            continue
        now = time.monotonic()
        dt = max(now - last, 1e-3)
        last = now
        t = now - start
        target_vel = np.zeros(3)
        if args.shape == "step":
            target_now = target
        elif args.shape == "ramp":
            d = np.deg2rad(args.ramp_dir)
            target_now = base.copy()
            target_now[0] += args.ramp_speed * np.sin(d) * t
            target_now[1] += args.ramp_speed * np.cos(d) * t
            target_now[2] = float(args.climb_alt)
            target_vel[0] = args.ramp_speed * np.sin(d)
            target_vel[1] = args.ramp_speed * np.cos(d)
        else:  # circle
            target_now = base.copy()
            target_now[0] += args.radius * np.sin(args.omega * t)
            target_now[1] += args.radius * (1.0 - np.cos(args.omega * t))
            target_now[2] = float(args.climb_alt)
            target_vel[0] = args.radius * args.omega * np.cos(args.omega * t)
            target_vel[1] = args.radius * args.omega * np.sin(args.omega * t)
        cmd = ctrl.update(
            target_now, target_vel, state["pos"], state["vel"], state["att"][2], dt
        )
        target = target_now
        scale = 500.0 / max(1e-6, args.max_angle)
        channels[0] = int(max(1000, min(2000, round(RC_MID + cmd.roll_deg * scale))))
        channels[1] = int(max(1000, min(2000, round(RC_MID - cmd.pitch_deg * scale))))
        channels[3] = int(
            max(1000, min(2000, round(RC_MID + cmd.yaw_deg * (500.0 / 60.0))))
        )
        t = now - start
        rows.append(
            [
                t,
                *state["pos"],
                *state["vel"],
                *state["att"],
                cmd.roll_deg,
                cmd.pitch_deg,
                cmd.yaw_deg,
                channels[0],
                channels[1],
                float(np.linalg.norm(state["pos"][:2] - target_now[:2])),
            ]
        )
        if len(rows) % max(1, int(args.rate / 2)) == 0:
            print(
                f"t={t:5.2f} pos=({state['pos'][0]:6.2f},{state['pos'][1]:6.2f},{state['pos'][2]:5.2f}) "
                f"vel=({state['vel'][0]:5.2f},{state['vel'][1]:5.2f}) "
                f"cmd=({cmd.roll_deg:5.1f},{cmd.pitch_deg:5.1f}) err={rows[-1][-1]:5.2f}",
                flush=True,
            )
        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))

    running[0] = False
    channels[2] = 1000
    channels[4] = 1000
    time.sleep(0.5)
    udp.close()

    if rows:
        with open(args.log, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["t", "e", "n", "u", "ve", "vn", "vu", "roll", "pitch", "yaw",
                 "roll_cmd", "pitch_cmd", "yaw_cmd", "rc_roll", "rc_pitch", "err"]
            )
            writer.writerows(rows)
        print(f"wrote {args.log} ({len(rows)} samples)", flush=True)
        if args.plot:
            from plot_trajectory import plot_csv

            plot_csv(args.log, args.plot)
            print(f"wrote {args.plot}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
