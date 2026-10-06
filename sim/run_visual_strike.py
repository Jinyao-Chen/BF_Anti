#!/usr/bin/env python3
"""44° 上仰相机 + 纯视觉 YOLO 接口的打击控制律 SITL 验证。

本脚本把 ``VisualStrikeGuidance`` 接到 Gazebo 真值上，并用相机投影生成
合成 bbox（与真实 YOLO 输出同接口）。因此它验证的是：

* 44° 相机外参/内参 -> 画面参考点 -> 逐步插值；
* track trajectory（拦截瞄准点、PN、t_go）；
* 视觉反馈把被打击无人机保持在画面中心；
* 终端命中判定；
* 0.05 s 日志与三维轨迹图输出。

真实 YOLO 只需要把 ``project_target_to_bbox`` 换成检测器输出
``(x1, y1, x2, y2)``，制导层不需要改动。
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
from typing import Optional


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import GazeboPoseSource, PoseState, project_target_to_bbox  # noqa: E402
from gazebo_camera_source import GazeboCameraSource  # noqa: E402
from guidance.visual_strike_guidance import StrikeCommand, VisualStrikeGuidance  # noqa: E402
from perception.camera_geometry import CameraExtrinsics, CameraIntrinsics  # noqa: E402
from run_intercept_mission import (
    RcState,
    climb_to,
    rc_sender,
    read_truth_pose,
    spool_up,
    wait_until_armed,
)  # noqa: E402

RC_MIN, RC_MAX, RC_MID = 1000, 2000, 1500


def strike_to_rc(cmd: StrikeCommand, max_angle_deg: float, current_yaw_deg: float = 0.0,
                 max_yaw_rate_dps: float = 200.0, prev: Optional[list[int]] = None,
                 rate_limit: int = 60, althold: bool = False,
                 yaw_sign: float = 1.0) -> list[int]:
    """StrikeCommand -> 16 通道 RC；ANGLE(+可选 ALT_HOLD)，AUX1/2 高。"""
    def clamp_axis(value: float, lo: int = 1000, hi: int = 2000) -> int:
        return int(max(lo, min(hi, round(value))))

    scale = 500.0 / max(1e-6, max_angle_deg)
    yaw_rate_scale = 500.0 / max(1e-6, max_yaw_rate_dps)
    channels = [RC_MID] * 16
    channels[0] = clamp_axis(RC_MID + cmd.roll_cmd_deg * scale)
    # 制导层的 pitch_cmd 用航空约定（抬头正），Betaflight ANGLE 通道正方向是
    # 低头，因此 RC 增量取反；否则会朝目标反方向加速。
    channels[1] = clamp_axis(RC_MID - cmd.pitch_cmd_deg * scale)
    # ALT_HOLD 模式下油门杆量以 ap_hover_throttle 为中心；配置里是 1545。
    channels[2] = 1545 if althold else clamp_axis(1000.0 + cmd.throttle_cmd * 1000.0)
    # 对 ANGLE 模式来说 yaw 通道是角速率杆量；yaw_sign 适配 yaw_motors_reversed。
    channels[3] = clamp_axis(RC_MID + yaw_sign * cmd.yaw_rate_dps * yaw_rate_scale)
    channels[4] = 2000  # ARM
    channels[5] = 2000  # ANGLE / tracking override
    channels[6] = 2000 if althold else 1000  # AUX3 / ALT_HOLD

    if prev is not None:
        for axis in (0, 1, 3, 2):
            delta = channels[axis] - prev[axis]
            if abs(delta) > rate_limit:
                channels[axis] = prev[axis] + (rate_limit if delta > 0 else -rate_limit)
    return channels


def search_for_target(
    rc: RcState,
    source: GazeboPoseSource,
    link,
    guidance: VisualStrikeGuidance,
    intr: CameraIntrinsics,
    extr: CameraExtrinsics,
    target_width_m: float,
    timeout_s: float = 15.0,
    althold: bool = False,
) -> Optional[PoseState]:
    """固定俯冲搜索角 + 左右扫描，直到合成相机看到目标。"""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        pose = read_truth_pose(link, source)
        if pose is None:
            time.sleep(0.05)
            continue
        bbox = project_target_to_bbox(pose, intr, extr, target_width_m)
        if bbox is not None:
            return pose
        elapsed = time.monotonic() - start
        yaw_off = 60.0 * math.sin(2.0 * math.pi * elapsed / 3.0)
        rc.set_rpyt(
            RC_MID,
            int(RC_MID + 50.0 * 500.0 / 60.0),  # Betaflight: 正 pitch 通道 = 低头
            1545 if althold else 1650,
            int(RC_MID + yaw_off * 500.0 / 60.0),
        )
        time.sleep(0.02)
    return None


def sitl_reachable(host: str = "127.0.0.1", port: int = 5761, timeout: float = 1.0) -> bool:
    """非阻塞预检：MspLink 的 _connect() 会一直重试，所以缺 SITL 时先给出明确提示。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="44° 视觉打击控制律 SITL 验证")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--interceptor", default="iris_with_Betaflight")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--tilt", type=float, default=44.0)
    parser.add_argument("--target-width", type=float, default=0.5, help="x500 机体宽度 m")
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--hfov", type=float, default=70.0)
    parser.add_argument("--release-alt", type=float, default=15.0)
    parser.add_argument("--max-angle", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=20.0, help="控制/日志频率 Hz，默认 20 = 0.05 s")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--search-timeout", type=float, default=20.0)
    parser.add_argument("--althold", action="store_true",
                        help="AUX3 打开 ALT_HOLD，用固件高度环代替手动推力")
    parser.add_argument("--yaw-gain", type=float, default=2.5, help="视觉偏航误差增益 dps/deg")
    parser.add_argument("--pitch-gain", type=float, default=2.5, help="视觉俯仰误差增益 dps/deg")
    parser.add_argument("--image-gain", type=float, default=0.8, help="画面误差到姿态的前馈增益")
    parser.add_argument("--log", default="/tmp/visual_strike.csv")
    parser.add_argument("--plot", default="/tmp/visual_strike.png")
    parser.add_argument("--capture", default=None, help="acquire 后保存一帧 Gazebo 相机 PNG")
    args = parser.parse_args()

    if not sitl_reachable():
        print("ERROR: SITL MSP 127.0.0.1:5761 is not reachable.", flush=True)
        print("Start Gazebo/SITL first in another terminal, for example:", flush=True)
        print("  cd ~/BF_Anti && BF_ANTI_CONFIG=$PWD/sim/config/sitl_gazebo_visual_strike.txt ./sim/launch_sitl.sh", flush=True)
        print("then rerun this script.", flush=True)
        return 2

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = RcState()
    running = [True]
    threading.Thread(target=rc_sender, args=(udp, rc, running), daemon=True).start()

    from run_intercept_mission import offboard  # 复用 MSP 连接
    link = offboard.MspLink()
    source = GazeboPoseSource(args.world, args.interceptor, args.target)
    camera = GazeboCameraSource(args.world, args.interceptor) if args.capture else None

    intr = CameraIntrinsics.from_fov(args.image_width, args.image_height, hfov_deg=args.hfov)
    extr = CameraExtrinsics(tilt_deg=args.tilt)
    guidance = VisualStrikeGuidance(
        camera_tilt_deg=args.tilt,
        image_size=(args.image_width, args.image_height),
        hfov_deg=args.hfov,
        hit_radius_m=0.8,
        yaw_error_gain_dps_per_deg=args.yaw_gain,
        pitch_error_gain_dps_per_deg=args.pitch_gain,
        image_attitude_gain=args.image_gain,
    )

    rows: list[list] = []
    hit = False
    min_range = float("inf")
    phases: list[str] = []
    try:
        print("arming...", flush=True)
        if not wait_until_armed(link, rc):
            print("ERROR: failed to arm", flush=True)
            return 2
        print("armed; spooling up", flush=True)
        spool_up(rc, target=1545, duration=2.5)
        print(f"climbing to {args.release_alt:.1f} m ...", flush=True)
        if not climb_to(rc, source, link, args.release_alt, timeout=30.0):
            print("WARNING: climb timeout", flush=True)
        if args.althold:
            print("enabling ALT_HOLD (AUX3 high)", flush=True)
            rc.set_aux3(2000)
        print("searching with 44deg camera...", flush=True)
        pose = search_for_target(rc, source, link, guidance, intr, extr,
                                 args.target_width, args.search_timeout,
                                 althold=args.althold)
        if pose is None:
            print("ERROR: target not acquired", flush=True)
            return 3
        print("target acquired; visual strike loop", flush=True)

        if camera is not None:
            deadline = time.time() + 3.0
            frame = None
            while time.time() < deadline:
                frame = camera.frame()
                if frame is not None:
                    break
                time.sleep(0.05)
            if frame is None:
                print("WARNING: no Gazebo camera frame received", flush=True)
            else:
                import cv2
                cv2.imwrite(args.capture, frame)
                print(f"wrote {args.capture} shape={frame.shape}", flush=True)

        start = time.monotonic()
        period = 1.0 / max(1.0, args.rate)
        prev_channels = [RC_MID] * 16
        last_cmd: Optional[StrikeCommand] = None
        while time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            pose = read_truth_pose(link, source)
            if pose is None:
                time.sleep(period)
                continue
            bbox = project_target_to_bbox(pose, intr, extr, args.target_width)
            target_vel = pose.target_velocity_enu
            cmd = guidance.update(
                pose.target_position_enu,
                target_vel,
                pose.own_position_enu,
                pose.own_velocity_enu,
                pose.own_rpy_deg,
                timestamp=time.monotonic(),
                dt=period,
                bbox_xyxy=bbox,
                valid=bbox is not None,
            )
            last_cmd = cmd
            channels = strike_to_rc(cmd, args.max_angle, pose.own_rpy_deg[2],
                                    prev=prev_channels, althold=args.althold)
            prev_channels = channels
            rc.set_channels(channels)
            min_range = min(min_range, cmd.target_range_m)
            if not phases or phases[-1] != cmd.phase.value:
                phases.append(cmd.phase.value)
            bbox_center = (float("nan"), float("nan")) if bbox is None else (
                (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
            rows.append([
                time.monotonic() - start,
                cmd.phase.value,
                cmd.target_range_m,
                *pose.own_position_enu,
                *pose.own_velocity_enu,
                *pose.target_position_enu,
                *pose.target_velocity_enu,
                bbox_center[0], bbox_center[1],
                cmd.u_ref_px, cmd.v_ref_px,
                cmd.az_ref_deg, cmd.el_ref_deg,
                cmd.pitch_cmd_deg, cmd.roll_cmd_deg, cmd.yaw_cmd_deg,
                cmd.pitch_rate_dps, cmd.yaw_rate_dps,
                cmd.throttle_cmd,
                int(cmd.hit),
            ])
            if int((time.monotonic() - start) * 5) % 5 == 0:
                print(
                    f"t={time.monotonic() - start:5.2f} {cmd.phase.value:<8} "
                    f"range={cmd.target_range_m:5.2f} vc={cmd.closing_speed_mps:5.2f} "
                    f"u_ref={cmd.u_ref_px:6.1f} u_meas={cmd.u_meas_px:6.1f} "
                    f"pitch={cmd.pitch_cmd_deg:6.1f} thr={cmd.throttle_cmd:4.2f} hit={cmd.hit}",
                    flush=True,
                )
            if cmd.hit:
                hit = True
                break
            time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
    finally:
        rc.idle()
        time.sleep(1.0)
        running[0] = False
        udp.close()
        if rows:
            with open(args.log, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    "t", "phase", "range", "truth_e", "truth_n", "truth_u",
                    "truth_ve", "truth_vn", "truth_vu",
                    "target_e", "target_n", "target_u",
                    "target_ve", "target_vn", "target_vu",
                    "u_meas", "v_meas", "u_ref", "v_ref",
                    "az_ref", "el_ref", "pitch_cmd", "roll_cmd", "yaw_cmd",
                    "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd", "hit",
                ])
                writer.writerows(rows)
            print(f"wrote {args.log} ({len(rows)} samples)", flush=True)
            if args.plot:
                from plot_trajectory import plot_csv
                plot_csv(args.log, args.plot)
                print(f"wrote {args.plot}", flush=True)

    print("=== visual strike summary ===")
    print(f"hit        : {hit}")
    print(f"min range  : {min_range:.3f} m")
    print(f"phases     : {' -> '.join(phases)}")
    if last_cmd is not None:
        print(f"final pitch/roll/yaw: {last_cmd.pitch_cmd_deg:.1f}/{last_cmd.roll_cmd_deg:.1f}/{last_cmd.yaw_cmd_deg:.1f}")
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
