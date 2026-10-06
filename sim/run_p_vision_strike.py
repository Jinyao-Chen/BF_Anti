#!/usr/bin/env python3
"""纯 P 视觉制导打击闭环（Gazebo SITL）。

链路::

    Gazebo 相机 / 合成 bbox → PVisualGuidance（纯 P）
        → 16 通道 RC（ANGLE + 手动油门，或 ANGLE + ALT_HOLD）
        → Betaflight SITL（Gazebo Harmonic + BetaflightPlugin）

任务分段（全部只用视觉量 + 计时器）::

    ARM → CLIMB(P 高度环) → STAGE(固定扫描俯仰 + 偏航扫掠) → LOCK(P 律)
        → HIT(bbox 尺寸判据) → 安全收尾

真值（GazeboPoseSource）只用于两件事：① `--synthetic` 合成 bbox；
② 评分（真值最近距离）。命中判据完全来自机载可观测量。

用法::

    # 终端 1
    BF_ROOT=~/betaflight BF_ANTI_WORLD=$PWD/sim/worlds/bf_anti_p_strike.sdf \
        ./sim/launch_sitl.sh --headless
    # 终端 2
    python3 sim/run_p_vision_strike.py --synthetic --duration 90 \
        --climb-alt 12 --log /tmp/p_strike.csv --plot /tmp/p_strike.png
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

from bbox_tracker import BboxTracker  # noqa: E402
from gazebo_target_bridge import GazeboPoseSource, project_target_to_bbox  # noqa: E402
from guidance.p_visual_guidance import PVisualGuidance, p_cmd_to_rc  # noqa: E402
from perception.camera_geometry import CameraExtrinsics, rotation_earth_from_body  # noqa: E402
from run_intercept_mission import (
    MSP_ATTITUDE,
    RcState,
    rc_sender,
    spool_up,
    wait_until_armed,
)  # noqa: E402
from run_visual_strike import sitl_reachable  # noqa: E402

_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402

MSP2_POSITION_STATE = 0x3015

CSV_HEADER = [
    "t", "phase", "u_px", "v_px", "s_px", "e_u", "e_v",
    "range_est", "z_t_est", "closing", "az_los", "el_los",
    "roll_cmd", "pitch_cmd", "yaw_rate_cmd", "throttle_pwm",
    "rc_roll", "rc_pitch", "rc_thr", "rc_yaw",
    "own_e", "own_n", "own_u", "own_ve", "own_vn", "own_vu",
    "roll", "pitch", "yaw",
    "truth_e", "truth_n", "truth_u",
    "truth_range", "hit", "sat",
    "own_truth_e", "own_truth_n", "own_truth_u",
    "truth_roll", "truth_pitch", "truth_yaw",
    "raw_u", "raw_v", "raw_s", "det_age", "track_reason", "cam_age",
    "vc_mps", "u_rate", "pn_roll",
]


class AsyncDetector(threading.Thread):
    """后台抓帧 + 推理，控制回路只取最近结果（带时效）。

    动机：同步调用 YOLO 时回路周期中位 106.6 ms(9.4 Hz)，离散相位滞后升高，
    同一组 P 会过冲/振荡；异步后回到 ~16 Hz。检测比控制慢是正常架构。
    """

    def __init__(self, yolo, camera, min_conf: float = 0.10) -> None:
        super().__init__(daemon=True)
        self.yolo = yolo
        self.camera = camera
        self.min_conf = float(min_conf)
        self._bbox = None
        self._t = 0.0
        self._frame = None
        self._pair = (None, None)      # (frame, bbox) 同一帧，供显示/录像
        self._bbox_vel = None          # 框中心速度(px/**秒**)，用于补偿检测延迟
        self._prev_bbox = None
        self._prev_det_t = 0.0
        self._raw = None               # 本帧原始检测（未过跟踪器），供诊断
        self._det_t = 0.0              # 产出 _raw 的墙钟时刻
        self._lock = threading.Lock()
        self._stop = False
        self.frames = 0
        self.dets = 0
        self.errors = 0
        self.last_error = ""

    def run(self) -> None:
        # choose_bbox 在 main() 里是**局部** import（有意保持 YOLO 为可选依赖），
        # 线程里看不到 -> 曾导致每帧 NameError、检测全程为 0（静默失效）。
        # 这里惰性导入，既拿得到又不污染模块级依赖。
        from run_visual_strike_yolo import choose_bbox
        while not self._stop:
            try:
                frame = self.camera.frame()
                if frame is not None:
                    with self._lock:
                        self.frames += 1
                        self._frame = frame          # 供录像/显示复用，避免双线程抢帧
                    b = choose_bbox(self.yolo.detect(frame), min_conf=self.min_conf)
                    now_det = time.monotonic()
                    with self._lock:
                        self._bbox = b
                        # 帧与框必须**成对**保存：异步后主循环若拿"最新帧"配"旧框"，
                        # 框就会落在飞机后面（用户反馈"yolo框跟不住"）。
                        self._pair = (frame, b)
                        if (b is not None and self._prev_bbox is not None
                                and self._prev_det_t > 0.0):
                            # 速度必须折成 **px/s**：此前存的是"每次检测的位移"，
                            # 却被当作 px/s 乘秒使用（检测 13 Hz vs 控制 20 Hz）→ 外推
                            # 少了约 13 倍。主循环也按同一约定（det_age）使用。
                            dtd = min(0.5, max(0.02, now_det - self._prev_det_t))
                            cur_u = (b[0] + b[2]) / 2.0
                            cur_v = (b[1] + b[3]) / 2.0
                            prv_u = (self._prev_bbox[0] + self._prev_bbox[2]) / 2.0
                            prv_v = (self._prev_bbox[1] + self._prev_bbox[3]) / 2.0
                            self._bbox_vel = ((cur_u - prv_u) / dtd, (cur_v - prv_v) / dtd)
                        if b is not None:
                            self._prev_bbox = b
                            self._prev_det_t = now_det
                            self._raw = b
                            self._det_t = now_det
                        self._t = now_det
                        if b is not None:
                            self.dets += 1
            except Exception as exc:                 # 不再静默吞异常
                with self._lock:
                    self.errors += 1
                    self.last_error = "%s: %s" % (type(exc).__name__, exc)
            time.sleep(0.005)

    def last_frame(self):
        with self._lock:
            return self._frame

    def latest_pair(self, max_age: float = 1.0):
        """返回 (帧, 该帧的框) —— 显示专用，保证框与画面同帧对齐。"""
        with self._lock:
            if self._pair[0] is None:
                return None, None
            if (time.monotonic() - self._t) > max_age or self._pair[1] is None:
                return self._pair[0], None
            return self._pair

    def age(self) -> float:
        with self._lock:
            return time.monotonic() - self._t if self._t else 0.0

    def bbox_velocity(self):
        with self._lock:
            return self._bbox_vel

    def raw_bbox(self):
        """最近一次**原始**检测（未过跟踪器）与它的墙钟延迟，供诊断/记录。"""
        with self._lock:
            if self._raw is None:
                return None, float("inf")
            return self._raw, time.monotonic() - self._det_t

    def predict(self, dt: float, max_shift_px: float = 90.0):
        """按检测延迟外推中心（补偿"框落后于机体运动"）。

        dt 用秒；_bbox_vel 已是 px/s。外推量限幅：像素速度本身带噪声，终端
        一秒能到上千 px，不封顶会把框甩出画面。
        """
        with self._lock:
            if self._bbox is None or self._bbox_vel is None:
                return self._bbox
            vx, vy = self._bbox_vel
            b = self._bbox
            dx = float(np.clip(vx * dt, -max_shift_px, max_shift_px))
            dy = float(np.clip(vy * dt, -max_shift_px, max_shift_px))
            return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)

    def latest(self, max_age: float = 1.0):
        with self._lock:
            if self._bbox is not None and (time.monotonic() - self._t) <= max_age:
                return self._bbox
            return None

    def stop(self) -> None:
        self._stop = True


def read_own(link) -> Optional[dict]:
    """飞控可观测量：MSP2 位置状态 + MSP 姿态。

    **航向标定（2026-09-30 实测）**：本 SITL 构建的 MSP_ATTITUDE 三个字段缩放不一致 ——
    roll/pitch 是 0.1deg，而 yaw 走的是
    src/main/msp/msp.c:1473 sbufWriteU16(dst, DECIDEGREES_TO_DEGREES(attitude.values.yaw))
    即**度**；固件航向还与 Gazebo 罗盘方位反号。按"统一 /10"解读会得到 1/10 且反号的航向。
    用 4 组日志（含 +-180deg 打转样本）拟合：yaw_msp_div10 = -0.100 * yaw_true，
    残差 p90 = 1.1deg、max = 2.4deg。这里按实测复原：yaw_true = -y。

    为什么必须修：guidance 的视线方位 az_los = f(像素, 姿态, 航向)，以及"丢失后按方位 cue
    回头"都依赖它。航向错 1/10 时，实测出现"速度矢量与视线平行"的 1.8 m 平行脱靶
    （预测脱靶距离 pm 从 25 m 到 1.5 m 全程钉在 1.7-1.9 m），打转时更是全错。
    """
    payload = link.query(MSP2_POSITION_STATE, is_v2=True)
    att = link.query(MSP_ATTITUDE, is_v2=False)
    out: dict = {}
    if payload and len(payload) >= 27:
        out["pos"] = np.array(struct.unpack_from("<3f", payload, 3), dtype=float)
        out["vel"] = np.array(struct.unpack_from("<3f", payload, 15), dtype=float)
    if att and len(att) >= 6:
        r, p, y = struct.unpack_from("<3h", att, 0)
        # yaw 字段是**度**（不是 0.1 度）且与本工程罗盘方位反号：见上方标定。
        out["rpy"] = (r / 10.0, p / 10.0, -float(y))
    return out if len(out) == 3 else None


def mask_own_horizontal(own: dict, mode: str) -> dict:
    """模拟"真机飞控给不出水平状态"（无 GPS / 无光流）的一种输入降级。

    真机上 IMU 姿态与气压高度一定拿得到，水平位置/速度则取决于有没有 GPS 或光流。
    本函数按 --mask-own-horiz 把水平分量清零，用来在仿真里预演无 GPS 真机：

      none  原样（当前仿真配置）
      pos   水平位置清零
      vel   水平速度清零
      both  两者都清零（最接近无 GPS/无光流的真机）

    注意：制导律只用 own_pos[2]（高度）与 own_vel（速度矢量对齐 / 超速治理 /
    沿视线爬升限幅）；水平位置在制导律里没有任何用处，所以 pos 单独清零近似无操作，
    真正起作用的是 vel。这里的 mask 只作用于制导律的入参，日志、HUD 与
    "相机陈旧"诊断仍用飞控原始状态，免得把仿真侧的诊断一起打瘸。
    """
    if mode == "none":
        return own
    out = dict(own)
    pos = np.array(own["pos"], dtype=float).copy()
    vel = np.array(own["vel"], dtype=float).copy()
    if mode in ("pos", "both"):
        pos[0] = 0.0
        pos[1] = 0.0
    if mode in ("vel", "both"):
        vel[0] = 0.0
        vel[1] = 0.0
    out["pos"] = pos
    out["vel"] = vel
    return out


def wrap180_deg(angle: float) -> float:
    return (float(angle) + 180.0) % 360.0 - 180.0


def render_projected_view(own_pos, own_rpy, target_pos, target_width_m,
                          intr, extr, image_size, tilt_deg):
    """相机几何精确的投影视图（与制导律共用内参/外参）。

    用途：本机 gz-sim 的相机渲染不跟随嵌套模型（机体俯仰 40° 画面仍不变），
    此时用它作为可视化回退——画出地平线、地面距离标尺与目标（按真实投影尺寸）。
    """
    import cv2
    import numpy as np

    w, h = int(image_size[0]), int(image_size[1])
    roll, pitch, yaw = (float(own_rpy[0]), float(own_rpy[1]), float(own_rpy[2]))
    img = np.full((h, w, 3), 205, np.uint8)          # 天空
    cam_axis_el = pitch + float(tilt_deg)            # 光轴仰角（度）

    horizon_v = intr.cy - intr.fy * np.tan(np.deg2rad(cam_axis_el))
    if horizon_v < h * 3:
        v0 = int(np.clip(horizon_v, -2 * h, 3 * h))
        cv2.rectangle(img, (0, v0), (w, h), (150, 150, 150), -1)   # 地面
    # 注意：不对背景做整图旋转——前景（目标/检测框）是按含滚转的完整投影矩阵算的，
    # 再转一次背景会让地平线与前景错位。
    heading = np.array([np.sin(np.deg2rad(yaw)), np.cos(np.deg2rad(yaw))])
    r_eb = rotation_earth_from_body(roll, pitch, yaw)
    r_bc = extr.rotation_body_from_camera()

    def project(point_enu):
        rel = np.asarray(point_enu, dtype=float) - np.asarray(own_pos, dtype=float)
        rel_cam = r_bc.T @ (r_eb.T @ rel)
        if rel_cam[2] <= 1e-3:
            return None
        return (float(intr.fx * rel_cam[0] / rel_cam[2] + intr.cx),
                float(intr.fy * rel_cam[1] / rel_cam[2] + intr.cy))

    # 地面距离标尺（每 10 m 一条横线，沿航向）
    for d in range(10, 121, 10):
        for lateral in (-6.0, 0.0, 6.0):
            left = np.array([-heading[1], heading[0]])
            p = np.array([float(own_pos[0]) + heading[0] * d + left[0] * lateral,
                          float(own_pos[1]) + heading[1] * d + left[1] * lateral,
                          0.0])
            uv = project(p)
            if uv is not None and -50 < uv[0] < w + 50 and -50 < uv[1] < h + 50:
                color = (120, 120, 120) if lateral else (90, 90, 90)
                cv2.circle(img, (int(uv[0]), int(uv[1])), 2, color, -1)

    # 目标（按真实投影尺寸画实心体，模拟真机轮廓）
    if target_pos is not None and np.all(np.isfinite(target_pos)):
        uv = project(target_pos)
        rng = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - np.asarray(own_pos, dtype=float)))
        if uv is not None:
            wpx = max(3.0, float(intr.fx) * float(target_width_m) / max(rng, 1e-3))
            x, y = int(uv[0]), int(uv[1])
            cv2.rectangle(img, (int(x - wpx / 2), int(y - wpx / 2)),
                          (int(x + wpx / 2), int(y + wpx / 2)), (60, 60, 60), -1)
            cv2.rectangle(img, (int(x - wpx / 2), int(y - wpx / 2)),
                          (int(x + wpx / 2), int(y + wpx / 2)), (30, 30, 30), 2)
    return img


def draw_view(frame, bbox, hud_lines, hit=False, lost=False):
    """在相机画面上叠加：检测框 + 画面中心 + HUD。

    文字用 ASCII（cv2 默认 Hershey 字体不含中文字形）。
    """
    import cv2

    img = frame.copy()
    h, w = img.shape[:2]
    bar_h = min(h, 12 + 16 * len(hud_lines))
    strip = img.copy()
    cv2.rectangle(strip, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(strip, 0.55, img, 0.45, 0, img)

    for i, line in enumerate(hud_lines):
        color = (0, 0, 255) if hit else (80, 255, 80)
        cv2.putText(img, line, (8, 20 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)

    # 画面中心（制导律的瞄准点：把目标压到这里）
    cv2.drawMarker(img, (w // 2, h // 2), (220, 220, 220), cv2.MARKER_CROSS, 18, 1)

    if bbox is not None:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255) if hit else (0, 255, 0), 2)
        cv2.putText(img, "DET %dpx" % (x2 - x1), (max(0, x1), max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(img, ((x1 + x2) // 2, (y1 + y2) // 2), (0, 255, 0),
                       cv2.MARKER_TILTED_CROSS, 12, 1)
    elif lost:
        cv2.putText(img, "NO DETECTION", (w // 2 - 80, h // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA)
    return img


def build_guidance(args, image_size) -> PVisualGuidance:
    return PVisualGuidance(
        camera_tilt_deg=args.tilt,
        image_size=image_size,
        hfov_deg=args.hfov,
        kp_yaw=args.kp_yaw,
        kp_roll=args.kp_roll,
        kp_pitch=args.kp_pitch,
        kp_z=args.kp_z,
        kp_vz=args.kp_vz,
        vz_max_mps=args.vz_max,
        kp_speed=args.kp_speed,
        v_max_mps=args.v_max,
        hover_pwm=args.hover_pwm,
        max_roll_deg=args.max_roll,
        max_pitch_down_deg=args.max_pitch_down,
        max_yaw_rate_dps=args.max_yaw_rate,
        target_width_m=args.target_width,
        hit_range_m=args.hit_radius,
        hit_dwell=args.hit_dwell,
        lift_ff=args.lift_ff,
        close_gain_deg=args.close_gain,
        meas_lpf_alpha=args.meas_lpf,
        lost_timeout_s=args.lost_timeout,
        kp_velalign=args.kp_velalign,
        pn_gain=args.pn_gain,
        pn_min_speed_mps=args.pn_min_speed,
        speed_source=args.speed_source,
        velalign_min_speed=args.velalign_min_speed,
        velalign_soft_deg=args.velalign_soft,
        range_lpf_tau_s=args.range_lpf,
        range_lpf_min_range_m=args.range_lpf_min,
        closing_max_mps=args.closing_max,
        vmax_close_range_m=args.vmax_close_range,
        vmax_close_floor_mps=args.vmax_close_floor,
        align_gate=args.align_gate,
        roll_align_gate=args.roll_align_gate,
        align_gate_lpf_s=args.align_gate_lpf,
        err_soft=args.err_soft,
        align_pitch_limit_deg=args.align_pitch,
        climb_along_los=bool(args.climb_along_los),
        climb_margin_mps=args.climb_margin,
        kp_launch=args.kp_launch,
        launch_cap_m=args.launch_cap,
        aim_below_deg=args.aim_below,
        vz_time_to_go=bool(args.vz_time_to_go),
        tau_min_s=args.tau_min,
        tau_max_s=args.tau_max,
        vz_lpf_tau_s=args.vz_lpf,
        ki_z=args.ki_z,
        ki_z_limit_m=args.ki_z_limit,
        pop_up_pitch_deg=args.pop_up_pitch,
        pop_up_z_m=args.pop_up_z,
        angle_limit_deg=args.angle_limit,
        yaw_full_rate_dps=args.yaw_full_rate,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="纯 P 视觉制导打击（Gazebo SITL）")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--interceptor", default="iris_with_Betaflight")
    parser.add_argument("--inner-model", default="iris_with_standoffs",
                        help="外层模型内嵌的机体名（轻量机体为 light_quad；顶层相机用空串）")
    parser.add_argument("--cam-link", default="base_link",
                        help="相机所在 link（顶层 fixed-joint 相机为 cam_link）")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--mask-own-horiz", choices=("none", "pos", "vel", "both"),
                        default="none",
                        help="预演无 GPS/无光流真机：把飞控给的水平位置/速度对制导律清零"
                             "（none=原样；pos/vel/both）。制导律只用 pos[2] 与 vel，"
                             "所以起作用的是 vel")
    parser.add_argument("--tilt", type=float, default=44.0)
    parser.add_argument("--hfov", type=float, default=68.75)
    parser.add_argument("--target-width", type=float, default=0.55)
    parser.add_argument("--climb-alt", type=float, default=12.0)
    parser.add_argument("--stage-pitch", type=float, default=25.0,
                        help="搜索阶段固定低头角（44° 相机需低头才看得到同高度目标，"
                             "18-30° 可兼顾视野与温和加速）")
    parser.add_argument("--kp-speed-search", type=float, default=3.0,
                        help="扫描阶段超速收平俯仰的 P 增益 (deg per m/s)")
    parser.add_argument("--kp-yaw-search", type=float, default=3.0,
                        help="有方位 cue 时的偏航率 P 增益 (dps per deg)")
    # MSP_ATTITUDE 约定标定（sim/p_axis_probe.py 实测，2026-09-28）：
    #   ch1=1700 → MSP pitch +11.1° 而真值 pitch -11.1° ⟹ MSP pitch 与 FLU 反号
    #   ch0=1700 → MSP roll +11.1° 且世界加速度向左 ⟹ MSP roll 与 FLU 反号
    #   （BetaflightPlugin 的 IMU 以 3.141593 0 0 安装 ⟹ 机体系 y/z 反号）
    # 偏航：本 SITL 构建的 MSP_ATTITUDE yaw 字段是**度**且反号（见 read_own 内标定），
    # 已在 read_own 里复原为 Gazebo 罗盘方位（残差 p90 1.1deg）。因此 az_los / 方位 cue
    # 都是可用的：实测 az_los-az_truth 中位 -1.0deg、p90 5deg。
    # 2026-09-28 在**修正后的**坐标系下重标（141 组样本，RMS 判据）：
    #   roll_sign=+1 RMS 0.28°  | pitch_sign=-1 RMS 0.22°  | yaw 无法仿射标定
    #   （加偏置后 RMS 仍 51°，固件航向估计在飘）⟹ 速度矢量对齐项默认关。
    parser.add_argument("--msp-roll-sign", type=float, default=1.0)
    parser.add_argument("--msp-pitch-sign", type=float, default=-1.0)
    parser.add_argument("--search-sweep", type=float, default=25.0, help="搜索偏航角速率幅值 dps")
    parser.add_argument("--reacquire-angle", type=float, default=60.0,
                        help="|方位 cue 误差| 超过该值即进入重捕：大角速率回头 + 抬平机头")
    parser.add_argument("--reacquire-rate", type=float, default=200.0,
                        help="重捕时的偏航角速率上限 dps（180° 回头约 0.9 s）")
    parser.add_argument("--reacquire-pitch", type=float, default=0.0,
                        help="重捕时的扫描俯仰 (deg，负=低头)；0=抬平不再加速")
    parser.add_argument("--search-period", type=float, default=6.0)
    parser.add_argument("--acquire-timeout", type=float, default=45.0)
    parser.add_argument("--acquire-frames", type=int, default=3,
                        help="连续 N 帧有效检测才进入 LOCK（防单帧假框）")
    parser.add_argument("--lost-timeout", type=float, default=0.5,
                        help="COAST 保持上一条姿态指令的时长（终端出画后继续压机头）")
    parser.add_argument("--lost-frames", type=int, default=8,
                        help="连续 N 帧无检测则回到搜索/扫描（要大于跟踪器 coast 帧数，"
                             "否则终端漏检会过早丢锁；但也不能太长，否则白白丢掉重捕时机）")
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--hit-radius", type=float, default=0.8)
    parser.add_argument("--hit-dwell", type=int, default=1)
    parser.add_argument("--hit-hold", type=float, default=0.30,
                        help="真值进入命中圈后继续记录的时长 s（取真实最近点）")
    parser.add_argument("--vertical", choices=("throttle", "althold"), default="throttle")
    # P 律增益
    parser.add_argument("--kp-yaw", type=float, default=55.0)
    parser.add_argument("--kp-roll", type=float, default=25.0)
    parser.add_argument("--kp-pitch", type=float, default=13.0)
    parser.add_argument("--kp-z", type=float, default=1.5)
    parser.add_argument("--kp-vz", type=float, default=250.0)
    parser.add_argument("--vz-max", type=float, default=8.0)
    # 实测：`--climb-along-los 1` 会把爬升限速，延迟"高度到手后才能低头加速"这一步 →
    # 前 8 s 速度只到 3.0 m/s（对照 5.4 m/s）、距离 28.8 m 就丢锁（24.5 m 脱靶）。
    # 20° 相机下必须**先拿高度**（el→0 才允许 −20° 低头、拿到 0.36g），故默认关闭。
    parser.add_argument("--climb-along-los", type=int, default=0,
                        help="1=垂速按几何限幅（沿视线爬升，实测会拖慢加速），0=默认")
    parser.add_argument("--climb-margin", type=float, default=1.5)
    parser.add_argument("--att-rate-mode", type=int, default=0,
                        help="1=pitch/roll 走角速率（ACRO）+伴机侧姿态级联 P，绕开 ANGLE 内环折损")
    parser.add_argument("--att-rate-gain", type=float, default=6.0,
                        help="姿态误差(deg) → 角速率(dps) 的伴随机侧 P 增益")
    parser.add_argument("--rate-max", type=float, default=250.0, help="角速率指令限幅 dps")
    parser.add_argument("--rate-full", type=float, default=600.0,
                        help="满杆对应的角速率 dps（BF 默认 ~670，取 600）")
    parser.add_argument("--kp-launch", type=float, default=0.0,
                        help="起飞段前向加速偏置 deg/m（按高度差额外低头，压掉爬行段）")
    parser.add_argument("--launch-cap", type=float, default=12.0)
    parser.add_argument("--pop-up-pitch", type=float, default=0.0,
                        help="抬头爬升段的俯仰角(deg, 正=抬头)；0=关闭")
    # 2026-09-29 设为默认：垂直用 (高度差/剩余飞行时间)，使航迹角跟随视线角。
    # 实测 |航迹角-视线角| 中位 8.7°（kp_z 法为陡爬先拔高），脱靶 0.586 m。
    parser.add_argument("--vz-time-to-go", type=int, default=1,
                        help="1=垂直通道用(高度差/剩余飞行时间)作垂速指令（命中一致）")
    parser.add_argument("--tau-min", type=float, default=0.20)
    parser.add_argument("--tau-max", type=float, default=3.0,
                        help="剩余时间估计上限(s)：防视觉测距噪声把 τ 估成大数导致爬行自锁")
    parser.add_argument("--vz-lpf", type=float, default=0.0,
                        help="竖向设定值 LPF 时间常数(s)，0=不滤波")
    # 用户要求加"一小点 Ki"：0.05 使俯仰 std 6.4°->4.6°、油门峰峰 397->349，
    # 脱靶不变（0.586->0.609 m）。原纯 P 契约据此放宽为"默认惰性、显式启用"。
    parser.add_argument("--ki-z", type=float, default=0.05,
                        help="竖向小积分增益(1/s^2)；原设计契约是纯 P，默认 0")
    parser.add_argument("--ki-z-limit", type=float, default=4.0)
    parser.add_argument("--aim-below", type=float, default=0.0,
                        help="从下方向上打：垂直通道瞄准视线下方该角度(deg)，0=瞄准目标高度")
    parser.add_argument("--pop-up-z", type=float, default=3.0,
                        help="高度差小于该值即结束抬头段、交回常规律")
    parser.add_argument("--kp-speed", type=float, default=0.0)
    parser.add_argument("--roll-align-gate", type=float, default=0.0,
                        help="横向对准门（默认关）：|e_u| 超过此值后衰减滚转指令，"
                             "避免起飞/回头段“航向未对准却先压坡度”造成横向漂移"
                             "（0.10 ≈ 3.4 度）")
    parser.add_argument("--align-gate", type=float, default=0.15,
                        help="|e_u| 大于该值时限制低头（先对准再闭合）")
    parser.add_argument("--align-pitch", type=float, default=8.0)
    parser.add_argument("--err-soft", type=float, default=0.35,
                        help="图像误差软饱和宽度（0=纯线性）；抑制捕获段撞限幅")
    parser.add_argument("--align-gate-lpf", type=float, default=0.15,
                        help="对准门限判据所用的 e_u 滤波时间常数(s)")
    # 2026-09-29：东向速度符号修正后该项才第一次真正可用（此前 az_vel 的东向分量
    # 是镜像的）。实测 4/4 命中(0.559-0.766 m) 且 TRACK 帧 99%（关闭时 0/2 命中、
    # TRACK 仅 69-70%）：把速度矢量对到视线上，横向偏差从 0.49-0.85 m 降到 0.02-0.47 m。
    parser.add_argument("--pn-gain", type=float, default=0.0,
                        help="比例导引横向项增益 N（0=关闭，默认纯 P）。N>0 时用"
                             "a=N*Vc*λ̇（λ̇ 取跟踪器框中心像素速率 / fx，Vc 取视觉接近率）"
                             "折算滚转指令，可替代依赖飞控速度的 velalign ——"
                             "但**启用即超出纯 P 契约**（引入视线角速率项）")
    parser.add_argument("--pn-min-speed", type=float, default=2.5,
                        help="比例导引项的低速门（m/s）：速度低于此值不产生横向加速度指令"
                             "（起飞/悬停时对准交给偏航通道）")
    parser.add_argument("--speed-source", choices=("fc", "closing"), default="fc",
                        help="超速治理/爬升限幅的速度来源：fc=飞控速度（默认）；"
                             "closing=视觉接近率（无 GPS/光流真机的替代）")
    parser.add_argument("--kp-velalign", type=float, default=1.5,
                        help="速度矢量-视线夹角 → 滚转 的比例增益 (deg/deg)")
    # 实测：该项线性使用时能顶到 ±270°（1.5 deg/deg x 180°），把 roll 打满 ±30°
    # （坏样本 54% 帧饱和、横滚速率 ±330°/s 的极限环；好样本该项只有 ~5°）。
    parser.add_argument("--velalign-min-speed", type=float, default=2.5,
                        help="速度矢量对齐项起效的最低水平速度 m/s（低速时速度方向无意义）")
    parser.add_argument("--velalign-soft", type=float, default=10.0,
                        help="速度矢量对齐项软饱和宽度 deg；该项最大输出 = kp*soft")
    parser.add_argument("--range-lpf", type=float, default=0.25,
                        help="用于 τ 的视觉测距低通时间常数 s（0=关闭）")
    parser.add_argument("--range-lpf-min", type=float, default=12.0,
                        help="小于该距离不滤波（近距离滤波会让 τ 滞后 ~2 m，竖向跟不上）")
    parser.add_argument("--closing-max", type=float, default=25.0,
                        help="接近率估计上限 m/s（防 bbox 抖动把接近率估成 90 m/s）")
    parser.add_argument("--vmax-close-range", type=float, default=12.0,
                        help="终端限速作用距离 m（0=关闭）；转弯半径正比 v^2")
    parser.add_argument("--vmax-close-floor", type=float, default=3.0,
                        help="终端限速下限 m/s")
    parser.add_argument("--v-max", type=float, default=12.0)
    parser.add_argument("--hover-pwm", type=float, default=1670.0)
    parser.add_argument("--max-roll", type=float, default=25.0)
    parser.add_argument("--max-pitch-down", type=float, default=44.0)
    parser.add_argument("--max-yaw-rate", type=float, default=120.0)
    parser.add_argument("--angle-limit", type=float, default=60.0)
    parser.add_argument("--yaw-full-rate", type=float, default=300.0)
    parser.add_argument("--lift-ff", action="store_true")
    parser.add_argument("--close-gain", type=float, default=0.0)
    parser.add_argument("--meas-lpf", type=float, default=1.0, help="1.0 = 关闭测量低通")
    # 感知
    parser.add_argument("--synthetic", action="store_true",
                        help="真值投影合成 bbox（与 YOLO 同接口），先验证制导-控制链路")
    parser.add_argument("--yolo-host", default="GPU_HOST")
    parser.add_argument("--yolo-port", type=int, default=9300)
    parser.add_argument("--async-yolo", type=int, default=1,
                        help="1=检测放后台线程（控制回路不被推理阻塞）")
    parser.add_argument("--min-conf", type=float, default=0.10,
                        help="YOLO 置信度阈值（配合跟踪器可放低，减少漏检）")
    parser.add_argument("--confirm-frames", type=int, default=2)
    parser.add_argument("--coast-frames", type=int, default=8,
                        help="漏检时按预测外推的最大帧数（终端框撑满时靠它续住）")
    parser.add_argument("--gate-px", type=float, default=90.0,
                        help="跳框门限：与预测位置差超过该像素数的框不采信")
    # 实时可视化 / 录像
    parser.add_argument("--show", action="store_true",
                        help="弹出实时窗口：相机画面 + 检测框 + HUD")
    parser.add_argument("--video", default=None, help="同时录制 mp4（带检测框与 HUD）")
    parser.add_argument("--show-scale", type=float, default=1.6, help="窗口放大倍数")
    parser.add_argument("--view", choices=("auto", "camera", "projected"), default="auto",
                        help="可视化源：auto=实拍可用则用实拍，否则用几何投影视图")
    parser.add_argument("--video-fps", type=float, default=20.0)
    # 记录
    parser.add_argument("--log", default="/tmp/p_strike.csv")
    parser.add_argument("--plot", default=None)
    parser.add_argument("--capture", default=None, help="保存一帧相机 PNG")
    parser.add_argument("--dump-dir", default=None,
                        help="按 --dump-hz 落原始相机帧(JPEG) + dump.csv(t, truth_range)，"
                             "供离线做检测器 A/B（不画框，保持像素原始）")
    parser.add_argument("--dump-hz", type=float, default=10.0)
    args = parser.parse_args()

    if not sitl_reachable():
        print("ERROR: SITL MSP 127.0.0.1:5761 不可达。先启动 SITL+Gazebo：", flush=True)
        print("  BF_ROOT=~/betaflight BF_ANTI_WORLD=$PWD/sim/worlds/bf_anti_p_strike.sdf \\", flush=True)
        print("      ./sim/launch_sitl.sh --headless", flush=True)
        return 2

    # 相机（先起，用真实分辨率建内参）
    camera = None
    try:
        from gazebo_camera_source import GazeboCameraSource
        camera = GazeboCameraSource(args.world, args.interceptor,
                                    inner_model=args.inner_model, link=args.cam_link)
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: 相机不可用({exc})，将只用合成 bbox", flush=True)

    frame = None
    if camera is not None:
        deadline = time.time() + 10.0
        while time.time() < deadline and frame is None:
            frame = camera.frame()
            time.sleep(0.05)
    image_size = (frame.shape[1], frame.shape[0]) if frame is not None else (640, 480)
    print(f"image_size={image_size}", flush=True)
    if args.capture and frame is not None:
        import cv2
        cv2.imwrite(args.capture, frame)
        print(f"wrote {args.capture}", flush=True)

    guidance = build_guidance(args, image_size)
    intr = guidance.intrinsics
    extr = CameraExtrinsics(tilt_deg=args.tilt)
    source = GazeboPoseSource(args.world, args.interceptor, args.target)
    if args.mask_own_horiz != "none":
        print("[mask] 制导律入参屏蔽水平状态: %s  （预演无 GPS/无光流真机）"
              % args.mask_own_horiz, flush=True)

    yolo = None
    if not args.synthetic:
        from run_visual_strike_yolo import RemoteYolo, choose_bbox
        yolo = RemoteYolo(args.yolo_host, args.yolo_port)

    viewer = None
    writer = None
    if args.show or args.video:
        try:
            import cv2
            viewer = "show" if args.show else "video"
            if args.video:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                size = (int(image_size[0]), int(image_size[1]))
                writer = cv2.VideoWriter(args.video, fourcc, float(args.video_fps), size)
                if not writer.isOpened():
                    print(f"WARNING: 无法写入 {args.video}", flush=True)
                    writer = None
            if args.show and not os.environ.get("DISPLAY"):
                print("WARNING: 无 DISPLAY，--show 不可用；改用 --video 录制", flush=True)
                viewer = "video" if writer is not None else None
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: 可视化不可用: {exc}", flush=True)
            viewer, writer = None, None

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = RcState()
    running = [True]
    threading.Thread(target=rc_sender, args=(udp, rc, running), daemon=True).start()
    link = offboard.MspLink()

    print("arming...", flush=True)
    if not wait_until_armed(link, rc, timeout=40.0):
        print("ERROR: 解锁失败", flush=True)
        running[0] = False
        return 2
    print("armed; spool up", flush=True)
    spool_up(rc, target=int(args.hover_pwm), duration=2.0)

    guidance.z_hold_m = float(args.climb_alt)
    alt_sp = float(args.climb_alt)
    if args.vertical == "althold":
        print("ALT_HOLD on (AUX3)", flush=True)
        rc.set_aux3(2000)
        offboard.send_target(link, types.SimpleNamespace(cruise=2.0, timeout_ms=0),
                             0.0, 0.0, args.climb_alt)
    print(f"climb to {args.climb_alt:.1f} m ...", flush=True)

    # ---- 异步检测：YOLO 推理(经隧道约 40-60 ms)不得阻塞控制回路 ----
    # 实测：同步调用时回路周期中位 106.6 ms(9.4 Hz)，此时同一组 P 就出现过冲
    # (e_u std 0.073、滚转过冲 0.90°)；异步后回路回到 ~16 Hz，同一组 P 仅
    # 0.044 / 0.21°。故周期变长（而非 P 过大）才是"晃"的根因。
    detector = None
    if yolo is not None and args.async_yolo:
        detector = AsyncDetector(yolo, camera, min_conf=args.min_conf)
        detector.start()
        print("[det] 异步检测已启用（控制回路不再等推理）", flush=True)

    rows: list[list] = []
    prev_channels = [1500] * 16
    period = 1.0 / max(1.0, args.rate)
    start = time.monotonic()
    locked = False
    lock_time: Optional[float] = None
    hit = False
    hit_t = None
    min_truth_range = float("inf")
    last_status = 0.0
    last_alt_cmd = 0.0
    rejected_frames = 0
    streak = 0
    lost_streak = 0
    tracker = BboxTracker(confirm_frames=args.confirm_frames,
                          max_coast_frames=args.coast_frames, gate_px=args.gate_px)
    disp_bbox = None          # 显示用的框（与 frame 同帧）
    last_track_t = 0.0        # 上一次跟踪器更新的时刻（算真实控制周期）
    raw_bbox, raw_age = None, float("nan")
    cam_ref = None            # 上一次比较用的帧（灰度）
    cam_pos_ref = None        # 上一次比较时机体位姿
    cam_check_t = 0.0
    cam_stale = False
    loop_period_avg = period          # 回路周期指数平滑（看门狗用）
    cam_checked = False
    final_cmd = None
    # 离线检测器 A/B 用的原始帧录制（默认关闭，不影响制导链路）
    dump_rows = []
    last_dump_t = -1e9

    try:
        while time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            now = time.monotonic()
            own = read_own(link)
            if own is None:
                time.sleep(period)
                continue
            # 制导律入参（可能与飞控原始状态不同：见 --mask-own-horiz）
            own_cmd = mask_own_horizontal(own, args.mask_own_horiz)
            pose = source.pose()
            # 每周期必须重新取帧：真 YOLO 吃的是这张图（此前只在循环外取过一次，
            # 导致检测框整场冻结在首帧；合成模式不看 frame 所以一直没暴露）。
            # 有异步检测线程时由它独占抓帧（同时两处调 camera.frame() 会让检测线程挂掉），
            # 主循环复用最近一帧做录制/显示。
            if detector is not None:
                # 显示用"帧+框"配对（同帧），否则异步下框会落在飞机后面
                latest, disp_bbox = detector.latest_pair(max_age=1.0)
                if latest is not None:
                    frame = latest
            elif camera is not None:
                latest = camera.frame()
                if latest is not None:
                    frame = latest

            bbox = None
            if args.synthetic and pose is not None:
                bbox = project_target_to_bbox(pose, intr, extr, args.target_width)
                # 相机只能看到画面内；真值投影允许 ±2 倍溢出，必须自己闸门，
                # 否则会把画面外的目标当成检测 → 锁定到假框 → 直接飞走（已实测）。
                if bbox is not None and not (
                        0.0 <= bbox[0] <= intr.width and 0.0 <= bbox[2] <= intr.width
                        and 0.0 <= bbox[1] <= intr.height and 0.0 <= bbox[3] <= intr.height):
                    bbox = None
                    rejected_frames += 1
            elif detector is not None:
                # 控制用最新框，并按检测延迟外推中心（补偿"框落后于运动"）
                b = detector.latest(max_age=1.0)
                bbox = detector.predict(min(0.12, detector.age())) if b is not None else None
                raw_bbox, raw_age = detector.raw_bbox()
            elif yolo is not None and frame is not None:
                bbox = choose_bbox(yolo.detect(frame), min_conf=args.min_conf)
            if raw_bbox is None and bbox is not None:
                raw_bbox, raw_age = bbox, 0.0

            # 检测调理（确认 / 门限关联 / 短时外推）：真 YOLO 逐次方差的主因是
            # 单帧漏检或跳框，这里只做**测量调理**，制导律仍是纯 P。
            dt_ctrl = max(1e-3, now - last_track_t) if last_track_t > 0.0 else period
            track = tracker.update(bbox, dt=dt_ctrl, timestamp=now)
            last_track_t = now
            bbox = track.bbox
            # 外推/门控帧不是本帧检测：制导律必须知道，否则"冻结像素 + 实时姿态"
            # 会让俯仰几何偏置正反馈到限幅（实测坏样本 0.4 s 顶到 +25°）。
            fresh_meas = track.fresh

            truth_range = float("inf")
            kf_target_pos = None
            if pose is not None:
                truth_range = float(np.linalg.norm(pose.target_position_enu - pose.own_position_enu))
                min_truth_range = min(min_truth_range, truth_range)
                # 投影视图用真值目标位置（仅用于可视化；制导只用视觉量）
                kf_target_pos = pose.target_position_enu

            if args.dump_dir is not None and frame is not None and pose is not None:
                el_dump = now - start
                if el_dump - last_dump_t >= 1.0 / max(1e-3, args.dump_hz):
                    last_dump_t = el_dump
                    import cv2 as _cv2
                    try:
                        os.makedirs(args.dump_dir, exist_ok=True)
                        _cv2.imwrite(os.path.join(args.dump_dir, "t%010.3f.jpg" % el_dump),
                                     frame, [int(_cv2.IMWRITE_JPEG_QUALITY), 95])
                        dump_rows.append((round(el_dump, 3), round(truth_range, 4)))
                    except Exception as exc:  # noqa: BLE001
                        print("WARNING: dump frame failed: %r" % (exc,), flush=True)
                        args.dump_dir = None

            elapsed = now - start
            # 捕获判据：由跟踪器保证"连续 N 帧确认"（tracker 负责 streak 计数）
            streak = track.streak
            if not locked and track.confirmed and bbox is not None:
                locked = True
                lock_time = elapsed
                print(f"t={elapsed:5.2f} LOCK bbox={tuple(round(float(v),1) for v in bbox)}"
                      f" s={bbox[2]-bbox[0]:.1f}px", flush=True)
            elif locked and streak == 0:
                lost_streak += 1
                if lost_streak >= args.lost_frames:
                    locked = False
                    lost_streak = 0
                    last_az = guidance.az_last_deg
                    print(f"t={elapsed:5.2f} LOST -> SEARCH (last az={last_az})", flush=True)
            if bbox is not None:
                lost_streak = 0

            # MSP 姿态 → FLU/航空约定（符号实测标定，见 --msp-*-sign 注释）
            own_rpy = (args.msp_roll_sign * own["rpy"][0],
                       args.msp_pitch_sign * own["rpy"][1],
                       own["rpy"][2])

            if locked:
                # 框中心像素速率（px/s）：由跟踪器在**真有检测**的那几帧更新，
                # 供 --pn-gain>0 的比例导引横向项使用；外推帧沿用它最后一次的值。
                u_rate = None
                if detector is not None:
                    bvec = detector.bbox_velocity()
                    if bvec is not None:
                        u_rate = float(bvec[0])
                cmd = guidance.update(bbox, own_cmd["pos"], own_cmd["vel"], own_rpy, now, period,
                                      fresh=fresh_meas, u_rate_px_s=u_rate)
                if args.vertical == "althold" and now - last_alt_cmd > 0.05:
                    if cmd.valid:
                        target_z = float(cmd.z_target_est_m)
                        vz = max(-args.vz_max, min(args.vz_max, args.kp_z * (target_z - own["pos"][2])))
                        alt_sp = alt_sp + float(np.clip(vz, -args.vz_max, args.vz_max)) * (now - last_alt_cmd)
                        alt_sp = float(np.clip(alt_sp, 2.0, 80.0))
                        link.send(offboard.mspv2_command(
                            offboard.MSP2_SET_POS,
                            struct.pack("<B3f2fH", 0, 0.0, 0.0, alt_sp, 2.0, 0.5, 0)))
                    last_alt_cmd = now
            else:
                # 扫描：在最后一次已知方位附近往返（有 cue 时不需要大范围扫）
                sweep_center = guidance.az_last_deg
                speed = float(np.linalg.norm(own_cmd["vel"][:2]))
                stage = args.stage_pitch
                if sweep_center is None:
                    sweep = args.search_sweep * math.sin(
                        2.0 * math.pi * elapsed / max(1e-3, args.search_period))
                else:
                    yaw_err = wrap180_deg(sweep_center - own["rpy"][2])
                    if abs(yaw_err) > args.reacquire_angle:
                        # 目标已经绕到身后：必须**大角速率回头**并**抬平机头**，
                        # 否则原逻辑把回扫速率钉在 ±search_sweep(25 dps) -> 转 180°
                        # 要 7 s，其间以 8-10 m/s 继续飞离（实测"拉不回来"）。
                        sweep = max(-args.reacquire_rate,
                                    min(args.reacquire_rate, args.kp_yaw_search * 2.0 * yaw_err))
                        stage = args.reacquire_pitch
                    else:
                        sweep = max(-args.search_sweep, min(args.search_sweep,
                                                            args.kp_yaw_search * yaw_err))
                # 扫描时低头会产生前向加速度；超速时按比例收平扫描俯仰，
                # 否则会在捕获前就加速飞过目标（早期实测：飞出 500 m）。
                if speed > args.v_max:
                    stage = max(0.0, stage - args.kp_speed_search * (speed - args.v_max))
                cmd = guidance.search_command(own_cmd["pos"], sweep, stage, own_cmd["vel"])
                if elapsed > args.acquire_timeout:
                    print("ERROR: 超时未捕获目标", flush=True)
                    break

            if args.att_rate_mode:
                k = args.att_rate_gain
                rr = max(-args.rate_max, min(args.rate_max,
                                            k * (cmd.roll_cmd_deg - own_rpy[0])))
                pr = max(-args.rate_max, min(args.rate_max,
                                            k * (cmd.pitch_cmd_deg - own_rpy[1])))
                channels = p_cmd_to_rc(cmd, angle_limit_deg=args.angle_limit,
                                       yaw_full_rate_dps=args.yaw_full_rate,
                                       prev=prev_channels, rate_limit=60,
                                       roll_rate_dps=rr, pitch_rate_dps=pr,
                                       rate_full_dps=args.rate_full, angle_on=False)
            else:
                channels = p_cmd_to_rc(cmd, angle_limit_deg=args.angle_limit,
                                       yaw_full_rate_dps=args.yaw_full_rate,
                                       prev=prev_channels, rate_limit=60)
            if args.vertical == "althold":
                channels[2] = int(args.hover_pwm)
            prev_channels = channels
            rc.set_channels(channels)
            final_cmd = cmd

            rows.append([
                elapsed, cmd.phase.value,
                cmd.u_px, cmd.v_px, cmd.s_px, cmd.e_u, cmd.e_v,
                cmd.range_est_m, cmd.z_target_est_m, cmd.closing_ratio,
                cmd.az_los_deg, cmd.el_los_deg,
                cmd.roll_cmd_deg, cmd.pitch_cmd_deg, cmd.yaw_rate_dps, cmd.throttle_pwm,
                channels[0], channels[1], channels[2], channels[3],
                *own["pos"], *own["vel"], *own_rpy,
                *(pose.target_position_enu if pose is not None else (float("nan"),) * 3),
                truth_range, int(cmd.hit), int(cmd.saturated),
                *(pose.own_position_enu if pose is not None else (float("nan"),) * 3),
                *(pose.own_rpy_deg if pose is not None else (float("nan"),) * 3),
                (raw_bbox[0] if raw_bbox is not None else float("nan")),
                (raw_bbox[1] if raw_bbox is not None else float("nan")),
                ((raw_bbox[2] - raw_bbox[0]) if raw_bbox is not None else float("nan")),
                raw_age, track.reason,
                (camera.age_s if camera is not None else float("nan")),
                cmd.vc_mps, cmd.u_rate_px_s, cmd.pn_roll_deg,
            ])

            # ---- 判断 Gazebo 实拍是否跟随机体（本机 gz-sim 会定格）----
            if (viewer is not None or writer is not None) and frame is not None:
                import cv2
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16)
                # 每 2 s 与 2 s 前的帧比较一次；只有这段时间里机体确实移动了(>1.5 m)
                # 却画面不变，才判定"相机定格"。避免起飞瞬间误判。
                if now - cam_check_t > 2.0:
                    if cam_ref is not None and not cam_checked:
                        diff = float(np.abs(gray - cam_ref).mean())
                        moved = float(np.linalg.norm(own["pos"][:2] - cam_pos_ref[:2])) \
                            if cam_pos_ref is not None else 0.0
                        if moved > 1.5:
                            # 阈值放到 0.05：只有"逐像素几乎完全相同"才算定格。
                            # 真实渲染下慢速爬升的平均帧间差约 0.26，不能误判。
                            cam_stale = diff < 0.05
                            cam_checked = True
                            print("[view] %s (2 s 内位移 %.1f m, 帧间差 %.2f)" % (
                                "Gazebo 相机画面定格 → 改用几何投影视图" if cam_stale
                                else "使用实拍画面", moved, diff), flush=True)
                    cam_ref = gray.copy()
                    cam_pos_ref = own["pos"].copy()
                    cam_check_t = now

            # ---- 实时画面（相机/投影视图 + 检测框 + HUD）----
            if frame is not None and (viewer is not None or writer is not None):
                est_r = cmd.range_est_m if np.isfinite(cmd.range_est_m) else float("nan")
                speed = float(np.linalg.norm(own["vel"][:2]))
                hud = [
                    "P-ONLY VISUAL STRIKE   phase=%s%s"
                    % (cmd.phase.value, "   *** HIT ***" if cmd.hit else ""),
                    "truth_r=%6.2fm  est_r=%6.2fm  bbox=%s"
                    % (truth_range, est_r, "yes" if bbox is not None else "--"),
                    "e_u=%+.2f  e_v=%+.2f  el_los=%+.1fdeg"
                    % (cmd.e_u, cmd.e_v, cmd.el_los_deg),
                    "alt=%5.1fm  spd=%4.1fm/s  thr=%4.0f  sat=%d"
                    % (own["pos"][2], speed, cmd.throttle_pwm, int(cmd.saturated)),
                    "cmd roll=%+5.1f pitch=%+6.1f yaw_rate=%+5.1f"
                    % (cmd.roll_cmd_deg, cmd.pitch_cmd_deg, cmd.yaw_rate_dps),
                    "RC r=%d p=%d t=%d y=%d"
                    % (channels[0], channels[1], channels[2], channels[3]),
                ]
                use_cam = (args.view == "camera") or (args.view == "auto" and not cam_stale)
                if use_cam:
                    base_img = frame
                else:
                    # 前景必须与检测框用同一套位姿（真值），否则两者会差几度而错位：
                    # 检测框来自真值投影，而 MSP 姿态/位置与之存在偏差。
                    if pose is not None:
                        view_pos, view_rpy = pose.own_position_enu, np.asarray(pose.own_rpy_deg)
                    else:
                        view_pos, view_rpy = own["pos"], np.asarray(own_rpy)
                    base_img = render_projected_view(
                        view_pos, view_rpy, kf_target_pos, args.target_width,
                        intr, extr, image_size, args.tilt)
                hud.append("view=%s" % ("camera" if use_cam else "projected"))
                show_bbox = bbox if detector is None else disp_bbox
                vis = draw_view(base_img, show_bbox, hud, hit=cmd.hit, lost=bbox is None)
                if writer is not None:
                    writer.write(vis)
                if viewer == "show":
                    import cv2
                    if args.show_scale != 1.0:
                        vis = cv2.resize(vis, None, fx=args.show_scale, fy=args.show_scale)
                    cv2.imshow("BF_Anti  P-only visual strike", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        print("quit by user", flush=True)
                        break

            if elapsed - last_status > 1.0:
                last_status = elapsed
                loop_hz = 1.0 / max(1e-6, loop_period_avg)
                det_txt = ""
                if detector is not None:
                    # 看门狗：检测线程若静默失效（frames 不涨 / errors 增长 / dets 恒 0），
                    # 这一行会立刻暴露，避免整场 SEARCH 却毫无提示。
                    det_txt = (f" det[fr={detector.frames} bbox={detector.dets} "
                               f"err={detector.errors}{' ' + detector.last_error[:60] if detector.errors else ''}]")
                print(f"t={elapsed:5.1f} {cmd.phase.value:<8} "
                      f"est_r={cmd.range_est_m:6.2f} truth_r={truth_range:6.2f} "
                      f"el={cmd.el_los_deg:6.1f} pitch={cmd.pitch_cmd_deg:6.1f} "
                      f"thr={cmd.throttle_pwm:4.0f} bbox={'Y' if bbox is not None else '-'} "
                      f"alt={own['pos'][2]:5.1f} spd={np.linalg.norm(own['vel']):4.1f} "
                      f"loop={loop_hz:4.1f}Hz{det_txt}", flush=True)

            if cmd.hit:
                hit = True
                print(f"t={elapsed:5.2f} HIT (bbox 判据) est_r={cmd.range_est_m:.2f}", flush=True)
                break
            # 命中：不要立刻退出——继续飞一小段记录**真实最近点**。
            # 否则报出来的只是"首次进入命中圈那一帧"的采样值，20 Hz / 8 m/s 下
            # 每帧距离就差 0.4 m，把 0.23 m 的近失报成 0.7 m。
            if truth_range <= args.hit_radius and hit_t is None:
                hit_t = elapsed
                print(f"t={elapsed:5.2f} 真值进入 {args.hit_radius} m（继续 {args.hit_hold}s 记最近点）",
                      flush=True)
            if hit_t is not None and (elapsed - hit_t >= args.hit_hold
                                      or truth_range > args.hit_radius * 1.5):
                break
            _lp = time.monotonic() - loop_start
            time.sleep(max(0.0, period - _lp))
            # 注意：用**完整周期**(含休眠)做 EMA，否则状态行会报出 38 Hz 而实际是 20 Hz
            loop_period_avg = 0.8 * loop_period_avg + 0.2 * max(
                1e-4, time.monotonic() - loop_start)
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    finally:
        running[0] = False
        channels_end = [1500] * 16
        channels_end[2] = 1000
        rc.set_channels(channels_end)
        rc.idle()
        time.sleep(0.5)
        udp.close()
        if yolo is not None:
            yolo.close()
        if writer is not None:
            writer.release()
            print(f"wrote {args.video}", flush=True)
        if viewer == "show":
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass

    if rows:
        with open(args.log, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        print(f"wrote {args.log} ({len(rows)} samples)", flush=True)

    if args.dump_dir is not None:
        dump_csv = os.path.join(args.dump_dir, "dump.csv")
        with open(dump_csv, "w", newline="", encoding="utf-8") as handle:
            w = csv.writer(handle)
            w.writerow(["t", "truth_range", "file"])
            for t_d, r_d in dump_rows:
                w.writerow([t_d, r_d, "t%010.3f.jpg" % t_d])
        print(f"wrote {dump_csv} ({len(dump_rows)} frames)", flush=True)

    print("=== P-only visual strike summary ===")
    print(f"locked         : {locked} (t={lock_time if lock_time is None else round(lock_time,2)})")
    print(f"hit (bbox 判据) : {hit}")
    print(f"min truth range: {min_truth_range:.3f} m")
    print(f"rejected boxes : {rejected_frames} (画面外/无效)")
    print(f"mask own horiz : {args.mask_own_horiz}")
    if final_cmd is not None:
        print(f"final phase    : {final_cmd.phase.value}")
        print(f"final est range: {final_cmd.range_est_m:.3f} m")

    if args.plot and rows:
        from plot_trajectory import plot_csv
        try:
            plot_csv(args.log, args.plot, title="P-only visual strike (Gazebo)")
            print(f"wrote {args.plot}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: plot failed: {exc}", flush=True)
    return 0 if (hit or min_truth_range <= args.hit_radius) else 1


if __name__ == "__main__":
    raise SystemExit(main())
