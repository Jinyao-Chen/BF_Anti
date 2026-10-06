#!/usr/bin/env python3
"""44° 上仰相机终端修正的 Gazebo 验证。

思路
----
固定靶机在 10 m 高，44° 上仰相机在水平巡航时看不到它。本脚本先用稳定的
offboard 位置控制器飞到一个“靶机下方 5 m、后方 5 m”的低位航点；此时目标
相对本机的仰角约 45°，正好进入 44° 相机视场。然后用相机针孔模型 + 目标
估计器从合成 bbox 估计目标位置，并用该估计值发起终端 offboard 目标，最后
用真值距离判定命中。

检测 bbox 是合成的，但使用的相机外参、内参、目标尺寸测距和 alpha-beta
滤波与真实视觉链路完全一致，因此可以作为 camera-in-the-loop 的几何闭环。
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
from typing import Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import GazeboPoseSource, InterceptBridge, PoseState  # noqa: E402
from run_offboard_intercept import read_position_state, wait_for_target_pose  # noqa: E402

_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402

MSP_ATTITUDE = 108


def read_attitude(link) -> Tuple[float, float, float]:
    """MSP_ATTITUDE -> (roll, pitch, yaw) 度。"""
    payload = link.query(MSP_ATTITUDE, is_v2=False)
    if not payload or len(payload) < 6:
        return (0.0, 0.0, 0.0)
    raw = struct.unpack_from("<3h", payload, 0)
    return tuple(float(v) / 10.0 for v in raw)  # type: ignore[return-value]


def send_offboard(link, x: float, y: float, z: float, cruise: float) -> None:
    setpoint = types.SimpleNamespace(cruise=cruise, timeout_ms=0)
    offboard.send_target(link, setpoint, float(x), float(y), float(z))


def wait_reached(link, timeout: float = 40.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        state = offboard.read_target_state(link)
        if state is not None and state[1]:
            return True
        time.sleep(0.2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="44° 相机终端修正 offboard 验证")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--target-width", type=float, default=1.0)
    parser.add_argument("--camera-tilt", type=float, default=44.0)
    parser.add_argument("--cruise", type=float, default=2.0)
    parser.add_argument("--climb-alt", type=float, default=10.0)
    parser.add_argument("--low-standoff", type=float, default=5.0, help="低位航点相对靶机的水平/垂直偏移")
    parser.add_argument("--hit-radius", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = offboard.RcLink()
    link = offboard.MspLink()
    running = [True]
    threading.Thread(target=offboard.rc_thread, args=(udp, rc, running), daemon=True).start()

    source = GazeboPoseSource(args.world, "iris_with_Betaflight", args.target)
    target_enu = wait_for_target_pose(source)
    if target_enu is None:
        print("ERROR: no target pose", flush=True)
        running[0] = False
        return 2

    bridge = InterceptBridge(
        camera_tilt_deg=args.camera_tilt,
        target_width_m=args.target_width,
        use_camera=True,
    )

    rows = []
    hit = False
    min_dist = float("inf")
    camera_seen = 0
    camera_est = None
    try:
        print("arming...", flush=True)
        if not offboard.arm(link, rc, timeout=40.0):
            print("ERROR: arm failed", flush=True)
            return 3
        rc.start_spool()
        time.sleep(1.0)

        print(f"climbing to {args.climb_alt:.1f} m ...", flush=True)
        send_offboard(link, 0.0, 0.0, args.climb_alt, 1.0)
        wait_reached(link, timeout=60.0)

        # 低位航点：在靶机水平后方 low_standoff 米、下方 low_standoff 米。
        own = read_position_state(link)
        own_pos = np.zeros(3) if own is None else own[0]
        horiz = target_enu[:2] - own_pos[:2]
        if float(np.linalg.norm(horiz)) < 1e-6:
            horiz = np.array([0.0, 1.0])
        horiz = horiz / float(np.linalg.norm(horiz))
        low_point = target_enu - np.array([horiz[0], horiz[1], 0.0]) * args.low_standoff
        low_point[2] = target_enu[2] - args.low_standoff
        print(f"low standoff point: ({low_point[0]:.2f},{low_point[1]:.2f},{low_point[2]:.2f})", flush=True)
        send_offboard(link, low_point[0], low_point[1], low_point[2], args.cruise)
        wait_reached(link, timeout=60.0)
        time.sleep(1.0)

        print("searching target through 44deg camera ...", flush=True)
        start = time.monotonic()
        while time.monotonic() - start < args.duration:
            pos_state = read_position_state(link)
            if pos_state is None:
                time.sleep(0.1)
                continue
            pos, vel = pos_state
            rpy = read_attitude(link)
            truth_dist = float(np.linalg.norm(pos - target_enu))
            min_dist = min(min_dist, truth_dist)
            pose = PoseState(
                own_position_enu=pos,
                own_velocity_enu=vel,
                own_rpy_deg=rpy,
                target_position_enu=target_enu,
                target_velocity_enu=np.zeros(3),
                timestamp=time.monotonic(),
            )
            cam_state = bridge.target_state(pose)
            visible = cam_state is not None and cam_state.valid
            if visible:
                camera_seen += 1
                camera_est = cam_state.position_enu.copy()
                est_err = float(np.linalg.norm(camera_est - target_enu))
                rows.append((time.monotonic() - start, *pos, rpy[1], truth_dist, cam_state.range_m, est_err))
                print(
                    f"t={time.monotonic() - start:5.1f} CAM visible range={cam_state.range_m:5.2f} "
                    f"est_err={est_err:5.2f} truth_dist={truth_dist:5.2f}",
                    flush=True,
                )
                # 连续看到几帧后，用相机估计位置发起终端目标，并附加一个穿靶偏置。
                if camera_seen >= 3:
                    direction = camera_est - pos
                    direction_norm = float(np.linalg.norm(direction))
                    direction = direction / direction_norm if direction_norm > 1e-6 else np.array([0.0, 1.0, 0.0])
                    aim = camera_est + direction * 3.0
                    print(
                        f"camera handoff -> aim=({aim[0]:.2f},{aim[1]:.2f},{aim[2]:.2f})",
                        flush=True,
                    )
                    send_offboard(link, aim[0], aim[1], aim[2], args.cruise)
                    break
            else:
                rows.append((time.monotonic() - start, *pos, rpy[1], truth_dist, -1.0, -1.0))
                if int((time.monotonic() - start) * 2) % 2 == 0:
                    print(
                        f"t={time.monotonic() - start:5.1f} CAM searching alt={pos[2]:5.2f} "
                        f"truth_dist={truth_dist:5.2f}",
                        flush=True,
                    )
            time.sleep(0.2)

        if camera_seen < 3:
            print("ERROR: 44deg camera did not acquire target at low standoff", flush=True)
            return 4

        # 终端阶段：继续监控真值距离，命中即结束。
        print("terminal offboard phase ...", flush=True)
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.duration:
            pos_state = read_position_state(link)
            if pos_state is None:
                time.sleep(0.1)
                continue
            pos, _ = pos_state
            dist = float(np.linalg.norm(pos - target_enu))
            min_dist = min(min_dist, dist)
            print(f"terminal t={time.monotonic() - t0:5.1f} dist={dist:5.2f} alt={pos[2]:5.2f}", flush=True)
            if dist <= args.hit_radius:
                hit = True
                break
            time.sleep(0.2)
    finally:
        rc.idle()
        running[0] = False
        udp.close()
        if args.log and rows:
            with open(args.log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["t", "east", "north", "up", "pitch_deg", "truth_dist", "cam_range", "cam_est_err"])
                writer.writerows(rows)
            print(f"wrote {args.log}", flush=True)

    print("=== camera handoff summary ===")
    print(f"hit         : {hit}")
    print(f"camera_seen : {camera_seen}")
    print(f"min dist    : {min_dist:.3f} m")
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
