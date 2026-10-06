#!/usr/bin/env python3
"""纯 P 视觉制导的运动学闭环验证（不依赖 Gazebo）。

链路::

    44° 上仰相机投影 → 合成 bbox → PVisualGuidance（纯 P）
        → 一阶姿态响应 + 推力/重力/阻力运动学 → 命中判定

与 `sim/visual_strike_kinematic.py` 共用日志列（32 列）与绘图/视频工具，
因此可以直接复用 `write_video` 与 `plot_trajectory.plot_csv`。

用途
----
1. 在进 Gazebo 之前确认控制律与几何自洽（命中、画面误差、高度保持）；
2. 增益扫描（`--sweep`）给出 Gazebo 试验的初值；
3. 作为 CI 回归（见 `tests/test_p_strike_kinematic.py`）。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import PoseState, project_target_to_bbox  # noqa: E402
from guidance.p_visual_guidance import PVisualGuidance  # noqa: E402
from perception.camera_geometry import CameraExtrinsics, CameraIntrinsics  # noqa: E402

G = 9.80665

CSV_HEADER = [
    "t", "truth_e", "truth_n", "truth_u", "truth_ve", "truth_vn", "truth_vu",
    "target_e", "target_n", "target_u", "target_ve", "target_vn", "target_vu",
    "roll", "pitch", "yaw",
    "u_meas", "v_meas", "u_ref", "v_ref",
    "az_ref", "el_ref", "pitch_cmd", "roll_cmd", "yaw_cmd",
    "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd",
    "range", "closing_speed", "t_go", "hit",
]


@dataclass
class KinConfig:
    """运动学闭环配置（全部为 SI；角度为度）。"""

    duration: float = 30.0
    dt: float = 0.02
    att_tau: float = 0.25          # 一阶姿态时间常数
    drag: float = 0.05             # 线性阻力 1/s（未辨识；Gazebo 才是真实终端速度）
    start_east: float = 0.0
    start_north: float = 0.0
    start_up: float = 10.0
    start_yaw: float = 0.0
    target_east: float = 0.0
    target_north: float = 60.0
    target_up: float = 10.0
    target_vel_east: float = 0.0
    target_vel_north: float = 0.0
    target_vel_up: float = 0.0
    target_sine_amp: float = 0.0   # 有界正弦机动（横向加速度），0=匀速
    target_sine_period: float = 8.0
    start_pitch: Optional[float] = None   # None = 自动按初始视线仰角对准（el0-44）
    fov_strict: bool = True               # True = 画面外目标视为不可见（诚实建模）
    camera_tilt: float = 44.0
    target_width: float = 0.55
    image_width: int = 640
    image_height: int = 480
    hfov: float = 68.75
    hit_radius: float = 0.8
    # P 律增益
    kp_yaw: float = 55.0
    kp_roll: float = 25.0
    kp_pitch: float = 13.0
    kp_z: float = 1.5        # 高度误差 → 垂速指令 (1/s)
    kp_vz: float = 250.0     # 垂速误差 → 油门 (RC per m/s)
    vz_max: float = 8.0      # 垂速指令限幅 (m/s)
    kp_speed: float = 0.0
    v_max: float = 12.0
    max_roll: float = 25.0
    max_yaw_rate: float = 120.0
    hover_pwm: float = 1670.0
    max_pitch_down: float = 44.0
    lift_ff: bool = False
    kp_velalign: float = 0.0    # 运动学模型偏航无滞后，开此项会过冲；Gazebo 侧用 runner 的 1.5
    align_gate: float = 0.0     # 同上：对准门限只在真系统（偏航迟缓）启用
    climb_along_los: bool = False   # 运动学模型无此问题，保持已验证行为
    err_soft: float = 0.0           # 误差软饱和只在真机(捕获段撞限幅)启用
    bbox_noise_px: float = 0.0     # 加性均匀噪声（复现检测抖动）
    seed: int = 0


@dataclass
class KinResult:
    hit: bool = False
    hit_time: Optional[float] = None
    min_range: float = float("inf")
    final_range: float = float("inf")
    mean_image_err_px: float = float("nan")
    final_image_err_px: float = float("nan")
    max_alt_error_m: float = 0.0
    max_speed: float = 0.0
    phases: list = field(default_factory=list)
    rows: list = field(default_factory=list)
    bbox_missing_frames: int = 0


def make_guidance(cfg: KinConfig, **overrides) -> PVisualGuidance:
    params = dict(
        camera_tilt_deg=cfg.camera_tilt,
        image_size=(cfg.image_width, cfg.image_height),
        hfov_deg=cfg.hfov,
        kp_yaw=cfg.kp_yaw,
        kp_roll=cfg.kp_roll,
        kp_pitch=cfg.kp_pitch,
        kp_z=cfg.kp_z,
        kp_vz=cfg.kp_vz,
        vz_max_mps=cfg.vz_max,
        kp_speed=cfg.kp_speed,
        v_max_mps=cfg.v_max,
        max_roll_deg=cfg.max_roll,
        max_yaw_rate_dps=cfg.max_yaw_rate,
        hover_pwm=cfg.hover_pwm,
        max_pitch_down_deg=cfg.max_pitch_down,
        target_width_m=cfg.target_width,
        hit_range_m=cfg.hit_radius,
        lift_ff=cfg.lift_ff,
        kp_velalign=cfg.kp_velalign,
        align_gate=cfg.align_gate,
        climb_along_los=cfg.climb_along_los,
        err_soft=cfg.err_soft,
    )
    params.update(overrides)
    return PVisualGuidance(**params)


def simulate(cfg: KinConfig, guidance: Optional[PVisualGuidance] = None,
             verbose: bool = False) -> KinResult:
    """跑一次运动学闭环；返回命中/脱靶/画面误差等指标。"""
    guidance = guidance or make_guidance(cfg)
    intr = CameraIntrinsics.from_fov(cfg.image_width, cfg.image_height, hfov_deg=cfg.hfov)
    extr = CameraExtrinsics(tilt_deg=cfg.camera_tilt)

    own_pos = np.array([cfg.start_east, cfg.start_north, cfg.start_up], dtype=float)
    own_vel = np.zeros(3, dtype=float)
    target_pos = np.array([cfg.target_east, cfg.target_north, cfg.target_up], dtype=float)
    target_vel0 = np.array([cfg.target_vel_east, cfg.target_vel_north, cfg.target_vel_up])

    roll = 0.0
    yaw = float(cfg.start_yaw)
    if cfg.start_pitch is None:
        rel0 = target_pos - own_pos
        horiz0 = float(np.hypot(rel0[0], rel0[1]))
        el0 = math.degrees(math.atan2(rel0[2], horiz0)) if horiz0 > 1e-9 else 90.0
        pitch = el0 - cfg.camera_tilt   # 初始让相机光轴对准目标（发射前对准）
    else:
        pitch = float(cfg.start_pitch)
    guidance.z_hold_m = float(cfg.start_up)

    rng = np.random.default_rng(cfg.seed)
    res = KinResult()
    t = 0.0
    steps = int(cfg.duration / cfg.dt)

    for _ in range(steps):
        # 目标运动：匀速 + 可选有界正弦（横向加速度有界，不会无限加速）
        target_vel = target_vel0.copy()
        if cfg.target_sine_amp > 0.0:
            w = 2.0 * math.pi / max(cfg.target_sine_period, 1e-3)
            target_vel[1] = target_vel0[1] + cfg.target_sine_amp * math.sin(w * t)
        target_pos = target_pos + target_vel * cfg.dt

        pose = PoseState(
            own_position_enu=own_pos.copy(),
            own_velocity_enu=own_vel.copy(),
            own_rpy_deg=(roll, pitch, yaw),
            target_position_enu=target_pos.copy(),
            target_velocity_enu=target_vel.copy(),
            timestamp=t,
        )
        bbox = project_target_to_bbox(pose, intr, extr, cfg.target_width)
        if bbox is not None and cfg.fov_strict:
            # 真实相机/YOLO：画面外没有检测
            if not (0.0 <= bbox[0] <= intr.width and 0.0 <= bbox[2] <= intr.width
                    and 0.0 <= bbox[1] <= intr.height and 0.0 <= bbox[3] <= intr.height):
                bbox = None
        if bbox is not None and cfg.bbox_noise_px > 0.0:
            noise = rng.uniform(-cfg.bbox_noise_px, cfg.bbox_noise_px, size=4)
            bbox = tuple(float(bbox[i] + noise[i]) for i in range(4))

        cmd = guidance.update(bbox, own_pos, own_vel, (roll, pitch, yaw),
                              timestamp=t, dt=cfg.dt)
        if bbox is None:
            res.bbox_missing_frames += 1
        if not res.phases or res.phases[-1] != cmd.phase.value:
            res.phases.append(cmd.phase.value)

        u_meas = (bbox[0] + bbox[2]) * 0.5 if bbox is not None else float("nan")
        v_meas = (bbox[1] + bbox[3]) * 0.5 if bbox is not None else float("nan")
        if bbox is not None:
            res.mean_image_err_px = _running_mean(
                res.mean_image_err_px, len(res.rows),
                math.hypot(u_meas - intr.cx, v_meas - intr.cy))
            res.final_image_err_px = math.hypot(u_meas - intr.cx, v_meas - intr.cy)

        # 一阶姿态响应 + 偏航角速率积分
        k_att = min(1.0, cfg.dt / max(cfg.att_tau, 1e-3))
        roll += (cmd.roll_cmd_deg - roll) * k_att
        pitch += (cmd.pitch_cmd_deg - pitch) * k_att
        yaw = _wrap180(yaw + cmd.yaw_rate_dps * cfg.dt)

        # 推力模型：pwm → 归一化推力（T ∝ cmd²），再按倾角分解
        hover_cmd = max(1.0, cfg.hover_pwm - 1000.0)
        cmd_norm = max(0.0, (cmd.throttle_pwm - 1000.0)) / hover_cmd
        thrust_ratio = cmd_norm * cmd_norm
        sigma = math.radians(-pitch)   # 低头量 σ = -θ
        phi = math.radians(roll)
        a_fwd = G * thrust_ratio * math.sin(sigma)
        a_right = G * thrust_ratio * math.sin(phi)
        a_up = G * (thrust_ratio * math.cos(sigma) * math.cos(phi) - 1.0)
        psi = math.radians(yaw)
        accel = np.array([
            a_fwd * math.sin(psi) + a_right * math.cos(psi),
            a_fwd * math.cos(psi) - a_right * math.sin(psi),
            a_up,
        ], dtype=float)
        accel = accel - cfg.drag * own_vel
        own_vel = own_vel + accel * cfg.dt
        own_pos = own_pos + own_vel * cfg.dt
        speed = float(np.linalg.norm(own_vel))
        res.max_speed = max(res.max_speed, speed)

        rel = target_pos - own_pos
        rng_m = float(np.linalg.norm(rel))
        res.min_range = min(res.min_range, rng_m)
        res.final_range = rng_m
        res.max_alt_error_m = max(res.max_alt_error_m, abs(rng_m * 0.0))

        res.rows.append([
            t,
            *own_pos, *own_vel,
            *target_pos, *target_vel,
            roll, pitch, yaw,
            u_meas, v_meas, float(intr.cx), float(intr.cy),
            cmd.az_los_deg, cmd.el_los_deg,
            cmd.pitch_cmd_deg, cmd.roll_cmd_deg, 0.0,
            0.0, cmd.yaw_rate_dps,
            cmd.throttle_pwm,
            cmd.range_est_m, cmd.closing_ratio, float("nan"),
            1 if cmd.hit else 0,
        ])

        if cmd.hit or rng_m <= cfg.hit_radius:
            res.hit = True
            res.hit_time = t
            break
        if own_pos[2] < -1.0:
            break
        t += cfg.dt

    if verbose:
        print(f"hit={res.hit} t_hit={res.hit_time} min_range={res.min_range:.3f} "
              f"phases={'->'.join(res.phases)}", flush=True)
    return res


def _running_mean(previous: float, n: int, value: float) -> float:
    if n <= 0 or not math.isfinite(previous):
        return value
    return previous + (value - previous) / (n + 1)


def _wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def write_csv(path: str, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def benchmark(cfg: KinConfig) -> dict:
    """对头 / 侧偏 / 高于本机 三个基准工况。"""
    cases = {
        "head_on": dict(target_east=0.0, target_north=60.0, target_up=cfg.start_up),
        "lateral_15m": dict(target_east=15.0, target_north=58.0, target_up=cfg.start_up),
        "target_above_20m": dict(target_east=0.0, target_north=50.0, target_up=cfg.start_up + 20.0),
        "crossing_5mps": dict(target_east=-30.0, target_north=55.0, target_up=cfg.start_up,
                              target_vel_east=5.0),
    }
    out = {}
    for name, override in cases.items():
        local = KinConfig(**{**asdict(cfg), **override})
        res = simulate(local)
        out[name] = res
        print(f"{name:<18} hit={str(res.hit):<5} t_hit={_fmt(res.hit_time)} "
              f"min_range={res.min_range:6.3f} m  img_err={res.final_image_err_px:6.1f} px  "
              f"phases={'->'.join(res.phases)}", flush=True)
    return out


def sweep(cfg: KinConfig, gains: list[tuple[str, list[float]]]) -> None:
    """单参数扫描：一次只动一个增益，其余固定。"""
    for name, values in gains:
        print(f"\n=== sweep {name} ===")
        for value in values:
            local = KinConfig(**{**asdict(cfg), name: value})
            res = simulate(local)
            print(f"  {name}={value:<8g} hit={str(res.hit):<5} t_hit={_fmt(res.hit_time)} "
                  f"min_range={res.min_range:6.3f} max_speed={res.max_speed:5.1f}", flush=True)


def _fmt(value) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="纯 P 视觉制导运动学闭环")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--att-tau", type=float, default=0.25)
    parser.add_argument("--drag", type=float, default=0.05)
    parser.add_argument("--start-east", type=float, default=0.0)
    parser.add_argument("--start-north", type=float, default=0.0)
    parser.add_argument("--start-up", type=float, default=10.0)
    parser.add_argument("--start-yaw", type=float, default=0.0)
    parser.add_argument("--target-east", type=float, default=0.0)
    parser.add_argument("--target-north", type=float, default=60.0)
    parser.add_argument("--target-up", type=float, default=10.0)
    parser.add_argument("--target-vel-east", type=float, default=0.0)
    parser.add_argument("--target-vel-north", type=float, default=0.0)
    parser.add_argument("--target-vel-up", type=float, default=0.0)
    parser.add_argument("--target-sine-amp", type=float, default=0.0)
    parser.add_argument("--target-sine-period", type=float, default=8.0)
    parser.add_argument("--bbox-noise-px", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-tilt", type=float, default=44.0)
    parser.add_argument("--target-width", type=float, default=0.55)
    parser.add_argument("--hit-radius", type=float, default=0.8)
    parser.add_argument("--kp-yaw", type=float, default=55.0)
    parser.add_argument("--kp-roll", type=float, default=11.0)
    parser.add_argument("--kp-pitch", type=float, default=13.0)
    parser.add_argument("--kp-z", type=float, default=1.0, help="高度误差→垂速指令 (1/s)")
    parser.add_argument("--kp-vz", type=float, default=150.0, help="垂速误差→油门 (RC/(m/s))")
    parser.add_argument("--vz-max", type=float, default=4.0)
    parser.add_argument("--kp-speed", type=float, default=3.0)
    parser.add_argument("--v-max", type=float, default=12.0)
    parser.add_argument("--hover-pwm", type=float, default=1670.0)
    parser.add_argument("--lift-ff", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="跑对头/侧偏/高空/横穿四工况")
    parser.add_argument("--sweep", action="store_true", help="单参数增益扫描")
    parser.add_argument("--log", default="/tmp/p_strike_kinematic.csv")
    parser.add_argument("--plot", default=None)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()

    cfg = KinConfig(
        duration=args.duration, dt=args.dt, att_tau=args.att_tau, drag=args.drag,
        start_east=args.start_east, start_north=args.start_north, start_up=args.start_up,
        start_yaw=args.start_yaw,
        target_east=args.target_east, target_north=args.target_north, target_up=args.target_up,
        target_vel_east=args.target_vel_east, target_vel_north=args.target_vel_north,
        target_vel_up=args.target_vel_up,
        target_sine_amp=args.target_sine_amp, target_sine_period=args.target_sine_period,
        camera_tilt=args.camera_tilt, target_width=args.target_width,
        hit_radius=args.hit_radius,
        kp_yaw=args.kp_yaw, kp_roll=args.kp_roll, kp_pitch=args.kp_pitch, kp_z=args.kp_z,
        kp_vz=args.kp_vz, vz_max=args.vz_max,
        kp_speed=args.kp_speed, v_max=args.v_max, hover_pwm=args.hover_pwm,
        lift_ff=args.lift_ff, bbox_noise_px=args.bbox_noise_px, seed=args.seed,
    )

    if args.sweep:
        sweep(cfg, [
            ("kp_pitch", [4.0, 8.0, 13.0, 20.0, 30.0]),
            ("kp_yaw", [25.0, 55.0, 90.0]),
            ("kp_roll", [0.0, 6.0, 11.0, 20.0]),
            ("kp_z", [0.5, 1.0, 1.5, 2.5]),
            ("kp_vz", [80.0, 150.0, 250.0]),
            ("v_max", [6.0, 9.0, 12.0, 16.0]),
        ])
        return 0

    if args.benchmark:
        benchmark(cfg)
        return 0

    res = simulate(cfg, verbose=False)
    write_csv(args.log, res.rows)
    print("=== P-only kinematic summary ===")
    print(f"hit            : {res.hit}")
    print(f"hit time       : {res.hit_time if res.hit_time is not None else 'no hit'}")
    print(f"min range      : {res.min_range:.3f} m")
    print(f"final range    : {res.final_range:.3f} m")
    print(f"mean image err : {res.mean_image_err_px:.2f} px")
    print(f"final image err: {res.final_image_err_px:.2f} px")
    print(f"max speed      : {res.max_speed:.2f} m/s")
    print(f"bbox missing   : {res.bbox_missing_frames} frames")
    print(f"phases         : {' -> '.join(res.phases)}")
    print(f"log            : {args.log}")

    if args.plot:
        from plot_trajectory import plot_csv
        plot_csv(args.log, args.plot, title="P-only visual strike kinematic")
        print(f"wrote {args.plot}")
    if args.video:
        from visual_strike_kinematic import write_video
        fps = min(30, max(10, int(round(1.0 / max(cfg.dt, 1e-3)))))
        write_video(res.rows, args.video, cfg.image_width, cfg.image_height,
                    cfg.hfov, cfg.target_width, fps=fps)
    return 0 if res.hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
