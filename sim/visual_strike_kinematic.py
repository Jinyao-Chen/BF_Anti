#!/usr/bin/env python3
"""44° 视觉打击控制律的轻量级运动学闭环验证。

不依赖 Gazebo，用一阶姿态 + 推力/重力运动学飞机模型验证：

1. 44° 相机投影合成 bbox（与真实 YOLO 输出同接口）；
2. 画面参考逐步插值：u_ref/v_ref 从当前框中心移动到瞄准点投影；
3. track trajectory：拦截瞄准点、PN、t_go；
4. 视觉反馈 + 推力：把被打击无人机保持画面中心并完成命中；
5. 输出 0.05 s 日志与 3D 轨迹图。

这个脚本的作用是给真实 Gazebo/YOLO 链路提供可复现的控制律基线。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import PoseState, project_target_to_bbox  # noqa: E402
from guidance.visual_strike_guidance import VisualStrikeGuidance  # noqa: E402
from perception.camera_geometry import CameraExtrinsics, CameraIntrinsics  # noqa: E402

G = 9.80665


def wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def write_video(
    rows: list[list],
    path: str,
    image_width: int,
    image_height: int,
    hfov_deg: float,
    target_width_m: float,
    fps: int = 25,
) -> bool:
    """把运动学闭环回放成 MP4：左侧 E-N 轨迹，右侧 44° 相机画面与 bbox/参考点。"""
    try:
        import cv2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as exc:  # pragma: no cover - 环境缺库时仅跳过视频
        print(f"WARNING: cannot write video ({exc})", flush=True)
        return False

    if not rows:
        return False

    focal_px = (image_width * 0.5) / math.tan(math.radians(hfov_deg) * 0.5)
    east = [r[1] for r in rows]
    north = [r[2] for r in rows]
    te = [r[7] for r in rows]
    tn = [r[8] for r in rows]
    e_lo, e_hi = min(min(east), min(te)), max(max(east), max(te))
    n_lo, n_hi = min(min(north), min(tn)), max(max(north), max(tn))
    e_pad = max(5.0, 0.1 * (e_hi - e_lo + 1e-6))
    n_pad = max(5.0, 0.1 * (n_hi - n_lo + 1e-6))

    dpi = 100
    fig = plt.figure(figsize=(12.8, 4.8), dpi=dpi)
    ax_map = fig.add_subplot(1, 2, 1)
    ax_cam = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.10, wspace=0.18)

    ax_map.set_xlim(e_lo - e_pad, e_hi + e_pad)
    ax_map.set_ylim(n_lo - n_pad, n_hi + n_pad)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_xlabel("East (m)")
    ax_map.set_ylabel("North (m)")
    ax_map.grid(True, color="#d0d0d0", lw=0.5)
    trail_own, = ax_map.plot([], [], "-", color="#1f77b4", lw=2.0, label="interceptor")
    trail_tgt, = ax_map.plot([], [], "--", color="#d62728", lw=1.5, label="target")
    own_dot, = ax_map.plot([], [], "o", color="#1f77b4", ms=6)
    tgt_dot, = ax_map.plot([], [], "s", color="#d62728", ms=6)
    hit_dot, = ax_map.plot([], [], "*", color="gold", ms=16, markeredgecolor="black", lw=0.5)
    ax_map.legend(loc="upper left", fontsize=8)

    canvas_h = int(round(fig.get_figheight() * dpi))
    canvas_w = int(round(fig.get_figwidth() * dpi))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (canvas_w, canvas_h))
    if not writer.isOpened():
        print(f"WARNING: OpenCV VideoWriter could not open {path}", flush=True)
        plt.close(fig)
        return False

    try:
        for i, r in enumerate(rows):
            t = r[0]
            rng = r[28]
            if r[31]:
                phase = "TERMINAL/HIT"
            elif rng < 6.0:
                phase = "TERMINAL"
            elif rng < 15.0:
                phase = "TRACK"
            else:
                phase = "APPROACH"
            own_e, own_n = r[1], r[2]
            tgt_e, tgt_n = r[7], r[8]
            u_meas, v_meas, u_ref, v_ref = r[16], r[17], r[18], r[19]
            vc = r[29]

            trail_own.set_data(east[: i + 1], north[: i + 1])
            trail_tgt.set_data(te[: i + 1], tn[: i + 1])
            own_dot.set_data([own_e], [own_n])
            tgt_dot.set_data([tgt_e], [tgt_n])
            if r[31]:
                hit_dot.set_data([own_e], [own_n])
            ax_map.set_title(f"E-N trajectory  t={t:.2f}s  {phase}", fontsize=10)

            ax_cam.clear()
            ax_cam.set_xlim(0, image_width)
            ax_cam.set_ylim(image_height, 0)
            ax_cam.set_facecolor("#101010")
            ax_cam.set_xticks([])
            ax_cam.set_yticks([])
            ax_cam.set_title(f"camera 44 deg  range={rng:5.1f} m  vc={vc:5.1f} m/s", fontsize=10)
            for gx in np.linspace(0, image_width, 5):
                ax_cam.axvline(gx, color="#303030", lw=0.5)
            for gy in np.linspace(0, image_height, 5):
                ax_cam.axhline(gy, color="#303030", lw=0.5)

            if math.isfinite(u_meas) and math.isfinite(v_meas) and rng > 0.05:
                w_px = max(4.0, focal_px * target_width_m / rng)
                h_px = w_px * 0.6
                ax_cam.add_patch(Rectangle(
                    (u_meas - w_px * 0.5, v_meas - h_px * 0.5), w_px, h_px,
                    fill=False, edgecolor="#39ff14", lw=2.0,
                ))
            if math.isfinite(u_ref) and math.isfinite(v_ref):
                ax_cam.plot([u_ref - 18, u_ref + 18], [v_ref, v_ref], color="red", lw=1.5)
                ax_cam.plot([u_ref, u_ref], [v_ref - 18, v_ref + 18], color="red", lw=1.5)
                img_err = math.hypot(u_ref - u_meas, v_ref - v_meas) if math.isfinite(u_meas) else float("nan")
                ax_cam.text(
                    16.0, 30.0,
                    f"phase {phase}\nrange {rng:5.1f} m\nimg err {img_err:5.1f} px",
                    color="white", fontsize=10, va="top", family="monospace",
                )

            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # 末帧多停 0.5 s，便于肉眼确认命中。
        frame_bgr = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[:, :, :3], cv2.COLOR_RGB2BGR)
        for _ in range(max(1, int(0.5 * fps))):
            writer.write(frame_bgr)
    finally:
        writer.release()
        plt.close(fig)

    print(f"wrote {path}", flush=True)
    return True


def run(args: argparse.Namespace) -> int:
    guidance = VisualStrikeGuidance(
        camera_tilt_deg=args.camera_tilt,
        image_size=(args.image_width, args.image_height),
        hfov_deg=args.hfov,
        hit_radius_m=args.hit_radius,
        min_safe_altitude_m=args.min_safe_alt,
    )
    intr = CameraIntrinsics.from_fov(args.image_width, args.image_height, hfov_deg=args.hfov)
    extr = CameraExtrinsics(tilt_deg=args.camera_tilt)

    own_pos = np.array([args.start_east, args.start_north, args.start_up], dtype=float)
    own_vel = np.zeros(3, dtype=float)
    target_pos = np.array([args.target_east, args.target_north, args.target_up], dtype=float)
    target_vel = np.array([args.target_vel_east, args.target_vel_north, args.target_vel_up], dtype=float)

    yaw = float(args.start_yaw)
    rel0 = target_pos - own_pos
    horiz0 = float(np.hypot(rel0[0], rel0[1]))
    el0 = math.degrees(math.atan2(rel0[2], horiz0)) if horiz0 > 1e-9 else 90.0
    pitch = el0 - args.camera_tilt  # 初始让相机光轴对准目标
    roll = 0.0
    rows: list[list] = []
    hit = False
    hit_time: Optional[float] = None
    min_range = float("inf")
    image_errors: list[float] = []
    phases: list[str] = []

    t = 0.0
    steps = int(args.duration / args.dt)
    for _ in range(steps):
        target_pos = target_pos + target_vel * args.dt
        rel = target_pos - own_pos
        rng = float(np.linalg.norm(rel))
        min_range = min(min_range, rng)

        pose = PoseState(
            own_position_enu=own_pos.copy(),
            own_velocity_enu=own_vel.copy(),
            own_rpy_deg=(roll, pitch, yaw),
            target_position_enu=target_pos.copy(),
            target_velocity_enu=target_vel.copy(),
            timestamp=t,
        )
        bbox = project_target_to_bbox(pose, intr, extr, args.target_width)

        cmd = guidance.update(
            target_pos,
            target_vel,
            own_pos,
            own_vel,
            (roll, pitch, yaw),
            timestamp=t,
            dt=args.dt,
            bbox_xyxy=bbox,
            valid=bbox is not None,
        )
        phases.append(cmd.phase.value)

        u_meas = (bbox[0] + bbox[2]) * 0.5 if bbox is not None else float("nan")
        v_meas = (bbox[1] + bbox[3]) * 0.5 if bbox is not None else float("nan")
        if bbox is not None:
            image_errors.append(math.hypot(cmd.u_ref_px - u_meas, cmd.v_ref_px - v_meas))

        # 一阶姿态响应；pitch 负 = 低头，产生向北加速度。
        tau_att = max(args.att_tau, 1e-3)
        pitch += (cmd.pitch_cmd_deg - pitch) * min(1.0, args.dt / tau_att)
        roll += (cmd.roll_cmd_deg - roll) * min(1.0, args.dt / tau_att)
        yaw = wrap180(yaw + cmd.yaw_rate_dps * args.dt)

        # 简化推力/重力模型：总距产生垂直加速度，俯仰产生水平加速度。
        # 高度环单独验证；这里先固定高度，专注水平视觉制导与命中。
        a_up = 0.0
        a_north = -G * math.tan(math.radians(pitch))
        a_east = G * math.tan(math.radians(roll))
        accel = np.array([a_east, a_north, a_up], dtype=float)
        # 简单线性阻力，避免速度无界。
        accel = accel - args.drag * own_vel
        own_vel = own_vel + accel * args.dt
        own_pos = own_pos + own_vel * args.dt

        rows.append([
            t,
            *own_pos,
            *own_vel,
            *target_pos,
            *target_vel,
            roll, pitch, yaw,
            u_meas, v_meas, cmd.u_ref_px, cmd.v_ref_px,
            cmd.az_ref_deg, cmd.el_ref_deg,
            cmd.pitch_cmd_deg, cmd.roll_cmd_deg, cmd.yaw_cmd_deg,
            cmd.pitch_rate_dps, cmd.yaw_rate_dps,
            cmd.throttle_cmd,
            cmd.target_range_m, cmd.closing_speed_mps, cmd.t_go_s,
            1 if cmd.hit else 0,
        ])

        if cmd.hit or rng <= args.hit_radius:
            hit = True
            hit_time = t
            break
        t += args.dt

    with open(args.log, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "t", "truth_e", "truth_n", "truth_u", "truth_ve", "truth_vn", "truth_vu",
            "target_e", "target_n", "target_u", "target_ve", "target_vn", "target_vu",
            "roll", "pitch", "yaw",
            "u_meas", "v_meas", "u_ref", "v_ref",
            "az_ref", "el_ref", "pitch_cmd", "roll_cmd", "yaw_cmd",
            "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd",
            "range", "closing_speed", "t_go", "hit",
        ])
        writer.writerows(rows)

    if args.plot:
        from plot_trajectory import plot_csv
        plot_csv(args.log, args.plot, title="visual strike kinematic")
        print(f"wrote {args.plot}", flush=True)

    if args.video:
        fps = min(30, max(10, int(round(1.0 / max(args.dt, 1e-3)))))
        write_video(
            rows, args.video, args.image_width, args.image_height,
            args.hfov, args.target_width, fps=fps,
        )

    mean_img_err = float(np.mean(image_errors)) if image_errors else float("nan")
    final_img_err = image_errors[-1] if image_errors else float("nan")
    print("=== visual strike kinematic summary ===")
    print(f"hit            : {hit}")
    print(f"hit time       : {hit_time if hit_time is not None else 'no hit'}")
    print(f"min range      : {min_range:.3f} m")
    print(f"final range    : {np.linalg.norm(target_pos - own_pos):.3f} m")
    print(f"mean image err : {mean_img_err:.2f} px")
    print(f"final image err: {final_img_err:.2f} px")
    print(f"phases         : {' -> '.join(dict.fromkeys(phases))}")
    return 0 if hit else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="44° 视觉打击控制律运动学验证")
    parser.add_argument("--target-east", type=float, default=0.0)
    parser.add_argument("--target-north", type=float, default=50.0)
    parser.add_argument("--target-up", type=float, default=0.0)
    parser.add_argument("--target-vel-east", type=float, default=0.0)
    parser.add_argument("--target-vel-north", type=float, default=0.0)
    parser.add_argument("--target-vel-up", type=float, default=0.0)
    parser.add_argument("--start-east", type=float, default=0.0)
    parser.add_argument("--start-north", type=float, default=0.0)
    parser.add_argument("--start-up", type=float, default=0.0)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--camera-tilt", type=float, default=44.0)
    parser.add_argument("--target-width", type=float, default=0.5)
    parser.add_argument("--image-width", type=int, default=1280)
    parser.add_argument("--image-height", type=int, default=720)
    parser.add_argument("--hfov", type=float, default=70.0)
    parser.add_argument("--hit-radius", type=float, default=0.8)
    parser.add_argument("--min-safe-alt", type=float, default=-10.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--att-tau", type=float, default=0.25)
    parser.add_argument("--drag", type=float, default=0.02)
    parser.add_argument("--log", default="/tmp/visual_strike_kinematic.csv")
    parser.add_argument("--plot", default="/tmp/visual_strike_kinematic.png")
    parser.add_argument("--video", default=None, help="输出命中回放 MP4，例如 /tmp/vs_kin.mp4")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
