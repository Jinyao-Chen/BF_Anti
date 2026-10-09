#!/usr/bin/env python3
"""纯视觉打击闭环（伴机侧）：相机 → YOLO/合成 bbox → 3D KF → APN → 速度外环。

控制链::

    camera → bbox(track) → 图像/视线 → TargetEstimator(测距+视线)
           → TargetKF(位置/速度/加速度) → InterceptGuidance(APN, 恒定接近速度)
           → XYVelocityController(accel→roll/pitch) → RC(ANGLE + ALT_HOLD)

固件侧: `ap_offboard_xy_enable=OFF`，offboard 目标只给 ALT_HOLD 高度；
`ANGLE` 由 AUX4 打开（见 sitl_gazebo_config.txt）。

用法::

    # 终端 1：SITL + Gazebo
    BF_ROOT=/tmp/bf_merge ./sim/launch_sitl.sh --headless
    # 终端 2：无 YOLO 的合成 bbox 版本（先验证视觉-制导-控制链路）
    python3 sim/run_vision_strike_companion.py --synthetic --duration 40 \
        --log /tmp/vision_strike.csv
    # 真实 YOLO：
    python3 sim/run_vision_strike_companion.py --yolo-host <host> --yolo-port 9300
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import socket
import struct
import sys
import threading
import time
import types
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import GazeboPoseSource, project_target_to_bbox  # noqa: E402
from guidance.intercept_guidance import InterceptGuidance  # noqa: E402
from guidance.xy_velocity_controller import XYVelocityController  # noqa: E402
from perception.camera_geometry import CameraExtrinsics, CameraIntrinsics  # noqa: E402
from perception.target_estimator import TargetEstimator, TargetState  # noqa: E402
from perception.target_kf import TargetKF  # noqa: E402

_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402

MSP_ATTITUDE = 108
MSP2_POSITION_STATE = 0x3015
RC_MID = 1500


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def read_own(link) -> Optional[dict]:
    payload = link.query(MSP2_POSITION_STATE, is_v2=True)
    att = link.query(MSP_ATTITUDE, is_v2=False)
    out: dict = {}
    if payload and len(payload) >= 27:
        out["pos"] = np.array(struct.unpack_from("<3f", payload, 3), dtype=float)
        out["vel"] = np.array(struct.unpack_from("<3f", payload, 15), dtype=float)
    if att and len(att) >= 6:
        r, p, y = struct.unpack_from("<3h", att, 0)
        out["rpy"] = (r / 10.0, p / 10.0, y / 10.0)
    return out if len(out) == 3 else None


def main() -> int:
    parser = argparse.ArgumentParser(description="纯视觉打击闭环（伴机侧）")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--interceptor", default="iris_with_Betaflight")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--tilt", type=float, default=44.0)
    parser.add_argument("--hfov", type=float, default=68.75)
    parser.add_argument("--target-width", type=float, default=0.8)
    parser.add_argument("--climb-alt", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--hit-radius", type=float, default=1.0)
    parser.add_argument("--synthetic", action="store_true",
                        help="用真值投影生成 bbox（无 YOLO），验证感知-制导-控制链路")
    parser.add_argument("--yolo-host", default=os.environ.get("YOLO_HOST", "127.0.0.1"),
                        help="YOLO 服务地址；默认取 $YOLO_HOST，否则本机 127.0.0.1")
    parser.add_argument("--yolo-port", type=int, default=9300)
    parser.add_argument("--nav-constant", type=float, default=4.0)
    parser.add_argument("--max-speed", type=float, default=10.0)
    parser.add_argument("--max-terminal-speed", type=float, default=14.0)
    parser.add_argument("--max-accel", type=float, default=8.0)
    parser.add_argument("--max-angle", type=float, default=40.0)
    parser.add_argument("--log", default="/tmp/vision_strike.csv")
    args = parser.parse_args()

    from run_visual_strike_yolo import RemoteYolo, choose_bbox  # noqa: E402

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = offboard.RcLink()
    running = [True]
    th = threading.Thread(target=offboard.rc_thread, args=(udp, rc, running), daemon=True)
    th.start()
    link = offboard.MspLink()
    print("arming...", flush=True)
    if not offboard.arm(link, rc, timeout=40.0):
        print("ERROR: failed to arm", flush=True)
        return 2

    running[0] = False
    th.join(timeout=1.0)
    channels = [RC_MID] * 16
    channels[2] = 1545
    channels[4] = 2000   # ARM
    channels[5] = 2000   # ALT_HOLD
    channels[6] = 1000
    channels[7] = 2000   # ANGLE
    running[0] = True

    def sender() -> None:
        while running[0]:
            udp.sendto(struct.pack("<d" + "H" * 16, time.time(), *channels),
                       (offboard.UDP_IP, offboard.UDP_PORT))
            time.sleep(0.02)

    threading.Thread(target=sender, daemon=True).start()

    offboard.send_target(link, types.SimpleNamespace(cruise=1.0, timeout_ms=0),
                         0.0, 0.0, args.climb_alt)
    time.sleep(1.0)

    source = GazeboPoseSource(args.world, args.interceptor, args.target)
    try:
        from gazebo_camera_source import GazeboCameraSource
        camera = GazeboCameraSource(args.world, args.interceptor)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: camera source unavailable: {exc}", flush=True)
        running[0] = False
        return 2
    yolo = RemoteYolo(args.yolo_host, args.yolo_port)

    frame0 = None
    deadline = time.time() + 10.0
    while time.time() < deadline and frame0 is None:
        frame0 = camera.frame()
        time.sleep(0.05)
    if frame0 is None:
        print("ERROR: no camera frame", flush=True)
        running[0] = False
        return 2
    h, w = frame0.shape[:2]
    intr = CameraIntrinsics.from_fov(w, h, hfov_deg=args.hfov)
    extr = CameraExtrinsics(tilt_deg=args.tilt)
    estimator = TargetEstimator(intr, extr, target_width_m=args.target_width)
    kf = TargetKF()
    guidance = InterceptGuidance(
        camera_tilt_deg=args.tilt,
        approach_range_m=25.0,
        terminal_range_m=3.0,
        nav_constant=args.nav_constant,
        max_speed_mps=args.max_speed,
        max_terminal_speed_mps=args.max_terminal_speed,
        max_accel_mps2=args.max_accel,
        hit_radius_m=args.hit_radius,
        speed_schedule="constant",
    )
    ctrl = XYVelocityController(
        kp_pos_mps_per_m=0.0,
        kp_vel_mps2_per_mps=0.0,
        max_accel_mps2=args.max_accel,
        max_angle_deg=args.max_angle,
        roll_sign=-1.0,
        pitch_sign=-1.0,
    )

    rows: list[list] = []
    hit = False
    min_truth_range = float("inf")
    last_alt_cmd = 0.0
    alt_cmd = args.climb_alt
    period = 1.0 / max(1.0, args.rate)
    start = time.monotonic()
    while time.monotonic() - start < args.duration:
        loop_start = time.monotonic()
        own = read_own(link)
        if own is None:
            time.sleep(period)
            continue
        now = time.monotonic()
        pose = source.pose()
        frame = camera.frame()
        bbox = None
        if args.synthetic and pose is not None:
            bbox = project_target_to_bbox(pose, intr, extr, args.target_width)
        elif frame is not None:
            dets = yolo.detect(frame)
            bbox = choose_bbox(dets, min_conf=0.25)

        truth_range = float("inf")
        if pose is not None:
            truth_range = float(np.linalg.norm(pose.target_position_enu - pose.own_position_enu))
            min_truth_range = min(min_truth_range, truth_range)

        est = None
        if bbox is not None:
            est = estimator.update(
                bbox, own["pos"], now,
                yaw_deg=own["rpy"][2], pitch_deg=own["rpy"][1], roll_deg=own["rpy"][0],
            )
            kf.predict(now)
            kf.update_position(est.position_enu, variance=max(0.25, (0.15 * est.range_m) ** 2))
        else:
            kf.predict(now)

        if kf.initialized:
            # 垂直通道：把 ALT_HOLD 的高度目标跟到目标高度（限幅+限速由 nav ramp 负责）。
            if now - last_alt_cmd > 0.5:
                want = float(np.clip(kf.position[2] + 0.5, 2.0, 60.0))
                alt_cmd = alt_cmd + float(np.clip(want - alt_cmd, -2.0, 2.0))
                link.send(offboard.mspv2_command(
                    offboard.MSP2_SET_POS,
                    struct.pack("<B3f2fH", 0, 0.0, 0.0, alt_cmd, 1.0, 0.5, 0),
                ))
                last_alt_cmd = now
            target = TargetState(
                position_enu=kf.position,
                velocity_enu=kf.velocity,
                range_m=float(np.linalg.norm(kf.position - own["pos"])),
                az_deg=0.0, el_deg=0.0, timestamp=now,
            )
            gcmd = guidance.update(
                target, own["pos"], own["vel"], timestamp=now, dt=period,
                target_accel_enu=kf.acceleration,
            )
            cmd = ctrl.update(
                target.position_enu, target.velocity_enu, own["pos"], own["vel"],
                own["rpy"][2], period, accel_ff_enu=gcmd.accel_enu,
            )
            scale = 500.0 / max(1e-6, args.max_angle)
            channels[0] = int(max(1000, min(2000, round(RC_MID + cmd.roll_deg * scale))))
            channels[1] = int(max(1000, min(2000, round(RC_MID - cmd.pitch_deg * scale))))
            # 航向：把目标方位误差转成角速率（ANGLE 模式下 yaw 通道是角速率）。
            yaw_err = wrap180(target_az(kf.position - own["pos"]) - own["rpy"][2])
            channels[3] = int(max(1000, min(2000, round(RC_MID - 1.5 * yaw_err * 500.0 / 120.0))))
            est_range = float(np.linalg.norm(kf.position - own["pos"]))
            closing = float(-np.dot(kf.position - own["pos"],
                                    own["vel"] - kf.velocity) / max(est_range, 1e-6))
            if est_range < args.hit_radius and closing > 0.3:
                hit = True
        else:
            gcmd = None
            cmd = None
            est_range, closing = float("inf"), 0.0

        rows.append([
            now - start,
            *(bbox if bbox is not None else (float("nan"),) * 4),
            *own["pos"], *own["vel"], *own["rpy"],
            *kf.position, *kf.velocity, *kf.acceleration,
            est_range, closing, truth_range, int(hit),
        ])
        if len(rows) % max(1, int(args.rate)) == 0:
            print(f"t={now - start:5.1f} est_r={est_range:6.2f} truth_r={truth_range:6.2f} "
                  f"hit={hit} pos=({own['pos'][0]:6.2f},{own['pos'][1]:6.2f},{own['pos'][2]:5.2f})", flush=True)
        if hit or truth_range <= 0.8:
            hit = True
            break
        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))

    running[0] = False
    channels[2] = 1000
    channels[4] = 1000
    rc.idle()
    time.sleep(0.5)
    udp.close()
    yolo.close()

    if rows:
        with open(args.log, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "t", "x1", "y1", "x2", "y2",
                "own_e", "own_n", "own_u", "own_ve", "own_vn", "own_vu",
                "roll", "pitch", "yaw",
                "kf_e", "kf_n", "kf_u", "kf_ve", "kf_vn", "kf_vu",
                "kf_ae", "kf_an", "kf_au",
                "est_range", "closing", "truth_range", "hit",
            ])
            writer.writerows(rows)
        print(f"wrote {args.log} ({len(rows)} samples)", flush=True)
    print(f"=== vision strike summary: hit={hit} min_truth_range={min_truth_range:.3f} m")
    return 0 if hit else 1


def target_az(rel: np.ndarray) -> float:
    return math.degrees(math.atan2(rel[0], rel[1])) % 360.0


if __name__ == "__main__":
    raise SystemExit(main())
