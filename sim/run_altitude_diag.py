#!/usr/bin/env python3
"""水平机动掉高诊断数据采集。

这不是主验证脚本，而是给 ``modeling/diagnose_altitude.py`` 生成输入 CSV：
在验证过的 offboard 链路上飞一条 5 m/s 水平腿，同时记录

* 固件位置/速度估计（MSP2_POSITION_STATE）；
* 姿态估计（MSP_ATTITUDE）；
* 高度/垂速估计（MSP_ALTITUDE）；
* 4 个电机输出（MSP_MOTOR）；
* Gazebo 真值位姿（pose/info）。

输出列与 ``modeling.diagnose_altitude`` 兼容。用法::

    # 终端 1
    ./sim/launch_sitl.sh --headless

    # 终端 2
    python3 sim/run_altitude_diag.py --cruise 5 --log /tmp/altitude_diag.csv

    python3 -m modeling.diagnose_altitude --log /tmp/altitude_diag.csv
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
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import GazeboPoseSource  # noqa: E402

_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402

from run_offboard_intercept import climb_to, wait_for_target_pose  # noqa: E402

MSP_ATTITUDE = 108
MSP_ALTITUDE = 109
MSP_MOTOR = 104
MSP_DEBUG = 254
MSP2_GET_POS = 0x3012


def read_attitude(link) -> Tuple[float, float, float]:
    """MSP_ATTITUDE -> (roll, pitch, yaw) 度。"""
    payload = link.query(MSP_ATTITUDE, is_v2=False)
    if payload and len(payload) >= 6:
        return tuple(v / 10.0 for v in struct.unpack_from("<3h", payload, 0))  # type: ignore[return-value]
    return (float("nan"),) * 3  # type: ignore[return-value]


def read_altitude(link) -> Tuple[float, float]:
    """MSP_ALTITUDE -> (altitude_m, vario_mps)。"""
    payload = link.query(MSP_ALTITUDE, is_v2=False)
    if payload and len(payload) >= 6:
        alt_cm, vario_cms = struct.unpack_from("<ih", payload, 0)
        return alt_cm / 100.0, vario_cms / 100.0
    return float("nan"), float("nan")


def read_motors(link) -> Tuple[int, int, int, int]:
    payload = link.query(MSP_MOTOR, is_v2=False)
    if payload and len(payload) >= 8:
        return struct.unpack_from("<4H", payload, 0)  # type: ignore[return-value]
    return (0, 0, 0, 0)


def read_debug(link) -> Tuple[int, ...]:
    """MSP_DEBUG -> 8 个 int16。debug_mode 未设置时返回全 0。"""
    payload = link.query(MSP_DEBUG, is_v2=False)
    if payload and len(payload) >= 16:
        return struct.unpack_from("<8h", payload, 0)
    return (0,) * 8


def read_target(link) -> Tuple[int, int, Tuple[float, float, float]]:
    payload = link.query(MSP2_GET_POS)
    if payload and len(payload) >= 16:
        active, reached = struct.unpack_from("<BB", payload, 0)
        target = struct.unpack_from("<3f", payload, 4)
        return active, reached, target  # type: ignore[return-value]
    return 0, 0, (float("nan"),) * 3  # type: ignore[return-value]


def read_position_state_ext(link):
    """读取 MSP2_POSITION_STATE，包含固件参考轨迹（新版固件）。

    返回 ``(pos, vel, ref_pos, ref_vel, ref_accel)``；旧固件没有尾部
    参考字段时用 NaN 填充。
    """
    payload = link.query(0x3015, is_v2=True)
    if not payload or len(payload) < 27:
        return None
    pos = np.array(struct.unpack_from("<3f", payload, 3), dtype=float)
    vel = np.array(struct.unpack_from("<3f", payload, 15), dtype=float)
    nan3 = (float("nan"),) * 3
    # Payload layout: 3 flags + pos(12) + vel(12) + target-velocity(12) +
    # reference position(12) + reference velocity(12) + reference acceleration(12).
    if len(payload) >= 27 + 12 + 36:
        ref_pos = np.array(struct.unpack_from("<3f", payload, 39), dtype=float)
        ref_vel = np.array(struct.unpack_from("<3f", payload, 51), dtype=float)
        ref_acc = np.array(struct.unpack_from("<3f", payload, 63), dtype=float)
    else:
        ref_pos = np.array(nan3, dtype=float)
        ref_vel = np.array(nan3, dtype=float)
        ref_acc = np.array(nan3, dtype=float)
    return pos, vel, ref_pos, ref_vel, ref_acc


def write_csv(path: str, rows: List[list]) -> None:
    header = [
        "t", "east", "north", "up", "ve", "vn", "vu",
        "ref_e", "ref_n", "ref_u", "ref_ve", "ref_vn", "ref_vu",
        "ref_ae", "ref_an", "ref_au",
        "roll", "pitch", "yaw", "msp_alt", "msp_vario",
        "m0", "m1", "m2", "m3",
        "tx", "ty", "tz", "tactive", "treached",
        "truth_e", "truth_n", "truth_u", "truth_roll", "truth_pitch", "truth_yaw",
        "dbg0", "dbg1", "dbg2", "dbg3", "dbg4", "dbg5", "dbg6", "dbg7",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = offboard.RcLink()
    link = offboard.MspLink()
    running = [True]
    rc_thread = threading.Thread(
        target=offboard.rc_thread, args=(udp, rc, running), daemon=True
    )
    rc_thread.start()

    source = GazeboPoseSource(args.world, "iris_with_Betaflight", args.target)
    target_enu = wait_for_target_pose(source)
    if target_enu is None:
        print("ERROR: cannot read target pose from Gazebo", flush=True)
        running[0] = False
        return 2
    print(f"target ENU: ({target_enu[0]:.2f}, {target_enu[1]:.2f}, {target_enu[2]:.2f}) m", flush=True)

    rows: List[list] = []
    try:
        print("arming...", flush=True)
        if not offboard.arm(link, rc, timeout=40.0):
            print("ERROR: failed to arm", flush=True)
            return 3
        rc.start_spool()
        time.sleep(1.0)
        print(f"climbing to {args.climb_alt:.1f} m ...", flush=True)
        if not climb_to(link, rc, args.climb_alt):
            print("WARNING: climb did not reach setpoint; continuing", flush=True)

        # Stabilise at the target altitude before starting the horizontal leg.
        # The climb phase can overshoot by more than a metre; starting the
        # aggressive 15-degree acceleration while high/low contaminates the
        # drop measurement and makes before/after comparisons unrepeatable.
        state = read_position_state_ext(link)
        leg_alt = float(args.climb_alt)
        if state is None:
            print("WARNING: no position state; skipping settle phase", flush=True)
        else:
            own_hold = state[0]
            # Hold the altitude the climb actually reached instead of
            # commanding a step back to climb_alt.  This removes the climb
            # overshoot from the initial condition of the diagnostic leg.
            leg_alt = float(own_hold[2])
            offboard.send_target(
                link,
                types.SimpleNamespace(cruise=0.5, timeout_ms=0),
                float(own_hold[0]),
                float(own_hold[1]),
                leg_alt,
            )
            settle_start = time.monotonic()
            while time.monotonic() - settle_start < args.settle:
                time.sleep(0.1)
            state = read_position_state_ext(link)
            if state is not None:
                leg_alt = float(state[0][2])
                print(
                    f"settled at ({state[0][0]:.2f},{state[0][1]:.2f},{state[0][2]:.2f}) m "
                    f"after {args.settle:.1f} s; leg altitude {leg_alt:.2f} m",
                    flush=True,
                )

        state = read_position_state_ext(link)
        own = np.zeros(3) if state is None else state[0]
        direction = target_enu - own
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
        aim = target_enu + direction * float(args.beyond)
        # Altitude diagnostic legs are level: use the altitude actually held
        # after the settle phase, rather than the climb setpoint, so the leg
        # starts without a vertical step.
        aim[2] = leg_alt
        # Offset the leg to the east so it flies past, not through, the fixed
        # target model.  The distance metric still uses the real target pose.
        aim[0] += float(args.lateral_offset)
        print(
            f"diagnostic leg at cruise={args.cruise:.1f} m/s; "
            f"aim=({aim[0]:.2f},{aim[1]:.2f},{aim[2]:.2f})",
            flush=True,
        )
        offboard.send_target(
            link,
            types.SimpleNamespace(cruise=args.cruise, timeout_ms=0),
            float(aim[0]),
            float(aim[1]),
            float(aim[2]),
        )

        start = time.monotonic()
        while time.monotonic() - start < args.duration:
            state = read_position_state_ext(link)
            if state is None:
                time.sleep(0.05)
                continue
            pos, vel, ref_pos, ref_vel, ref_acc = state
            attitude = read_attitude(link)
            altitude = read_altitude(link)
            motors = read_motors(link)
            debug = read_debug(link)
            target = read_target(link)
            gazebo = source.pose()
            truth = gazebo.own_position_enu if gazebo else (float("nan"),) * 3
            truth_rpy = gazebo.own_rpy_deg if gazebo else (float("nan"),) * 3
            rows.append(
                [
                    time.monotonic() - start,
                    *pos,
                    *vel,
                    *ref_pos,
                    *ref_vel,
                    *ref_acc,
                    *attitude,
                    *altitude,
                    *motors,
                    *target[2],
                    target[0],
                    target[1],
                    *truth,
                    *truth_rpy,
                    *debug,
                ]
            )
            if len(rows) % 10 == 0:
                print(
                    f"t={rows[-1][0]:5.1f} pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:6.2f}) "
                    f"pitch={attitude[1]:6.1f} alt={altitude[0]:6.2f} "
                    f"motors={motors} truth_z={truth[2]:6.2f}",
                    flush=True,
                )
            if np.linalg.norm(pos - target_enu) < args.hit_radius:
                break
            time.sleep(args.period)
    finally:
        rc.idle()
        running[0] = False
        rc_thread.join(timeout=1.0)
        udp.close()
        if rows:
            write_csv(args.log, rows)
            print(f"wrote {args.log} ({len(rows)} samples)", flush=True)
            if args.plot:
                from plot_trajectory import plot_csv
                plot_csv(args.log, args.plot)
                print(f"wrote {args.plot}", flush=True)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="水平机动掉高诊断采集")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--cruise", type=float, default=5.0, help="水平巡航速度 m/s")
    parser.add_argument("--climb-alt", type=float, default=10.0)
    parser.add_argument("--settle", type=float, default=4.0,
                        help="水平腿开始前的悬停稳定时间 s")
    parser.add_argument("--beyond", type=float, default=6.0)
    parser.add_argument("--lateral-offset", type=float, default=5.0,
                        help="aim point east offset, avoids flying through the target model")
    parser.add_argument("--hit-radius", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=25.0)
    parser.add_argument("--period", type=float, default=0.05,
                        help="采样周期 s（默认 0.05 s = 20 Hz，避免步长过大）")
    parser.add_argument("--log", default="/tmp/altitude_diag.csv")
    parser.add_argument("--plot", default="", help="可选：输出 3D 轨迹 PNG 路径")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
