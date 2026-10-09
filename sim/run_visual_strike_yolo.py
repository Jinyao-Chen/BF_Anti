#!/usr/bin/env python3
"""真实 Gazebo 相机 + 远程 nvidia0 YOLO + ALT_HOLD 的视觉打击闭环。

与 ``run_visual_strike.py`` 的区别：
* 目标图像来自 Gazebo ``camera_sensor`` 真实渲染帧，而不是真值投影；
* 检测来自 --yolo-host（默认 $YOLO_HOST，否则 127.0.0.1；远程 GPU 常走 SSH 隧道）的 YOLO 服务；
* 目标 3D 状态由 ``TargetEstimator`` 从 YOLO bbox + 本机位姿估计；
* 录像保存真实相机帧 + 检测框 + 阶段/距离 HUD；
* 控制仍走 RC ANGLE + AUX3 ALT_HOLD。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import socket
import struct
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_camera_source import GazeboCameraSource  # noqa: E402
from gazebo_target_bridge import GazeboPoseSource, PoseState  # noqa: E402
from guidance.visual_strike_guidance import (  # noqa: E402
    ImageTrackReference,
    StrikeCommand,
    StrikePhase,
    VisualStrikeGuidance,
)
from perception.camera_geometry import CameraExtrinsics, CameraIntrinsics  # noqa: E402
from perception.target_estimator import TargetEstimator  # noqa: E402
from run_intercept_mission import (
    MSP2_POSITION_STATE,
    MSP_ATTITUDE,
    RcState,
    climb_to,
    rc_sender,
    spool_up,
    wait_until_armed,
)  # noqa: E402
from run_visual_strike import RC_MID, sitl_reachable, strike_to_rc  # noqa: E402

RC_MIN, RC_MAX = 1000, 2000
TARGET_NAMES = {"copter_drone", "motor_drone"}


def _msp_ensure_bounded(link, timeout: float = 0.3) -> bool:
    """有界重连：避免 MspLink._connect() 的无限重试阻塞控制环。"""
    sock = getattr(link, "_sock", None)
    if sock is not None:
        return True
    try:
        host = getattr(link, "host", "127.0.0.1")
        port = int(getattr(link, "port", 5761))
        s = socket.create_connection((host, port), timeout=timeout)
        s.settimeout(0.5)
        link._sock = s
        return True
    except OSError:
        return False


def _msp_read(link, cmd: int, is_v2: bool):
    if not _msp_ensure_bounded(link):
        return None
    try:
        return link.query(cmd, is_v2=is_v2, reconnect=False)
    except (OSError, ConnectionResetError):
        try:
            link._close()
        except OSError:
            pass
        return None


def read_own_pose(link, source, last_target=None):
    """只依赖 MSP 读本机状态；Gazebo pose 只用于靶机真值（缺失时沿用上一帧）。"""
    pos_payload = _msp_read(link, MSP2_POSITION_STATE, is_v2=True)
    if not pos_payload or len(pos_payload) < 27:
        return None
    pos = np.array(struct.unpack_from("<3f", pos_payload, 3), dtype=float)
    vel = np.array(struct.unpack_from("<3f", pos_payload, 15), dtype=float)
    att_payload = _msp_read(link, MSP_ATTITUDE, is_v2=False)
    if att_payload and len(att_payload) >= 6:
        raw = struct.unpack_from("<3h", att_payload, 0)
        rpy = (float(raw[0]) / 10.0, -float(raw[1]) / 10.0, float(raw[2]) / 10.0)
    else:
        rpy = (0.0, 0.0, 0.0)
    tgt_pos = np.zeros(3, dtype=float)
    tgt_vel = np.zeros(3, dtype=float)
    gz = source.pose()
    if gz is not None:
        tgt_pos = np.array(gz.target_position_enu, dtype=float)
        tgt_vel = np.array(gz.target_velocity_enu, dtype=float)
        last_target = (tgt_pos.copy(), tgt_vel.copy())
    elif last_target is not None:
        tgt_pos, tgt_vel = last_target
    return PoseState(
        own_position_enu=pos,
        own_velocity_enu=vel,
        own_rpy_deg=rpy,
        target_position_enu=tgt_pos,
        target_velocity_enu=tgt_vel,
        timestamp=time.monotonic(),
    ), last_target


class RemoteYolo:
    """4-byte length-prefixed JPEG -> JSON detections client."""

    def __init__(self, host: str, port: int, timeout: float = 0.5) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: Optional[socket.socket] = None

    def _connect(self) -> None:
        if self.sock is None:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            self.sock.settimeout(self.timeout)

    def _recv_exact(self, n: int) -> Optional[bytes]:
        assert self.sock is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def detect(self, frame: np.ndarray) -> list[dict]:
        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return []
        data = jpg.tobytes()
        for attempt in range(2):
            try:
                self._connect()
                assert self.sock is not None
                self.sock.sendall(struct.pack("!I", len(data)) + data)
                hdr = self._recv_exact(4)
                if hdr is None:
                    raise ConnectionResetError("no response header")
                n = struct.unpack("!I", hdr)[0]
                payload = self._recv_exact(n)
                if payload is None:
                    raise ConnectionResetError("short response")
                return json.loads(payload).get("dets", [])
            except (OSError, ConnectionResetError, json.JSONDecodeError):
                if self.sock is not None:
                    try:
                        self.sock.close()
                    except OSError:
                        pass
                    self.sock = None
                if attempt == 1:
                    return []
        return []

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None


def choose_bbox(dets: list[dict], min_conf: float = 0.25, min_px: float = 3.0):
    best = None
    for d in dets:
        if d.get("name") not in TARGET_NAMES:
            continue
        conf = float(d.get("conf", 0.0))
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = [float(v) for v in d.get("xyxy", [0, 0, 0, 0])]
        if x2 - x1 < min_px or y2 - y1 < min_px:
            continue
        if best is None or conf > float(best.get("conf", 0.0)):
            best = d
    if best is None:
        return None
    return (float(best["xyxy"][0]), float(best["xyxy"][1]),
            float(best["xyxy"][2]), float(best["xyxy"][3]))


class KalmanFilterUV:
    """画面框中心 (u,v) 的恒速 Kalman 滤波 + 马氏距离门。

    状态 x = [u, v, du, dv]^T；量测 z = [u, v]^T。
    """

    def __init__(self, q: float = 40.0, r: float = 9.0) -> None:
        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float) * 500.0
        self.q = float(q)
        self.r = float(r)
        self.initialized = False
        self.last_rejected = False

    def reset(self, z: np.ndarray) -> None:
        self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=float)
        self.P = np.diag([25.0, 25.0, 400.0, 400.0])
        self.initialized = True

    def predict(self, dt: float) -> np.ndarray:
        dt = max(float(dt), 1e-3)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        # 恒速模型离散过程噪声
        q = self.q
        Q = np.array([
            [dt ** 3 / 3.0, 0, dt ** 2 / 2.0, 0],
            [0, dt ** 3 / 3.0, 0, dt ** 2 / 2.0],
            [dt ** 2 / 2.0, 0, dt, 0],
            [0, dt ** 2 / 2.0, 0, dt],
        ], dtype=float) * q
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[:2].copy()

    def update(self, z: np.ndarray, dt: float, gate: float = 36.0):
        """返回 (滤波位置, 是否接受量测)。门拒绝时只输出预测。"""
        self.predict(dt)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        R = np.eye(2, dtype=float) * self.r
        y = np.asarray(z, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(y.T @ S_inv @ y)
        except np.linalg.LinAlgError:
            S_inv = None
            d2 = 1e9
        self.last_rejected = d2 > gate
        if not self.last_rejected:
            K = self.P @ H.T @ S_inv
            self.x = self.x + K @ y
            I = np.eye(4)
            self.P = (I - K @ H) @ self.P
        return self.x[:2].copy(), not self.last_rejected


class DirectImageController:
    """纯图像闭环：KF + ImageTrackReference + 视觉估计目标高度。

    * bbox 中心 -> KF -> 插值参考 -> yaw/pitch 瞄准；
    * bbox 宽度 + 针孔模型估计距离 range_est；
    * 由相机上仰角和 bbox 竖直位置估计目标仰角 -> 目标高度；
    * 高度环直接控油门到估计目标高度，pitch 只负责瞄准 + 少量前推。
    """

    def __init__(self, width: int, height: int, hover_throttle: float = 1545.0,
                 fov_deg: float = 68.75, camera_tilt_deg: float = 44.0,
                 target_width_m: float = 0.8,
                 target_alt_m: Optional[float] = None,
                 simple_attack: bool = False) -> None:
        self.w = int(width)
        self.h = int(height)
        self.fx = (self.w * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
        self.fy = self.fx
        self.camera_tilt_deg = float(camera_tilt_deg)
        self.target_width_m = float(target_width_m)
        self.fixed_target_alt = None if target_alt_m is None else float(target_alt_m)
        self.simple_attack = bool(simple_attack)
        self.ref = ImageTrackReference(self.w, self.h)
        self.ref_started = False
        self.kf = KalmanFilterUV()
        self.hover_rc = float(hover_throttle)
        self.alt_kp = 80.0   # RC per m
        self.alt_kd = 25.0   # RC per (m/s)
        self.bw_f = None
        self.target_alt_f = None
        self.yaw_filtered = 0.0
        self.pitch_filtered = 0.0
        self.throttle_filtered = (float(hover_throttle) - 1000.0) / 1000.0
        self.last_bbox = None
        self.last_size = None
        self.last_t = 0.0

    def reset_track(self) -> None:
        """目标丢失后清空图像跟踪状态，保留最后一次目标高度估计用于高度保持。"""
        self.kf = KalmanFilterUV()
        self.ref_started = False
        self.bw_f = None
        self.yaw_filtered = 0.0
        self.pitch_filtered = 0.0

    def predict_bbox(self, timestamp: float, max_age_s: float = 0.5):
        """短时丢失时用 KF 外推一个虚拟 bbox，维持闭环；超过 max_age 返回 None。"""
        if self.last_bbox is None or self.last_size is None or not self.kf.initialized:
            return None
        dt = float(timestamp) - float(self.last_t)
        if dt <= 0.0 or dt > max_age_s:
            return None
        u = float(self.kf.x[0] + self.kf.x[2] * dt)
        v = float(self.kf.x[1] + self.kf.x[3] * dt)
        if not (0.0 <= u <= float(self.w) and 0.0 <= v <= float(self.h)):
            return None
        w, h = self.last_size
        return (u - w * 0.5, v - h * 0.5, u + w * 0.5, v + h * 0.5)

    def update(self, dt: float, bbox, timestamp: float,
               own_alt_m: float = 0.0, own_vz_mps: float = 0.0,
               own_pitch_deg: float = 0.0, max_alt_m: float = 8.0,
               predicted: bool = False) -> StrikeCommand:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        if not predicted:
            self.last_bbox = (x1, y1, x2, y2)
            self.last_size = (max(1.0, x2 - x1), max(1.0, y2 - y1))
            self.last_t = float(timestamp)
        z = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=float)

        if not self.kf.initialized:
            self.kf.reset(z)
            u, v = z
        elif predicted:
            # 虚拟 bbox 只用于短时保持闭环，不能反过来当量测污染 KF。
            u, v = z
        else:
            (u, v), _ = self.kf.update(z, dt)

        if not self.ref_started:
            self.ref.reset(float(u), float(v))
            self.ref_started = True
        self.ref.set_goal(self.w * 0.5, self.h * 0.5)
        state = self.ref.update(max(float(dt), 1e-3), float(u), float(v))
        u_ref, v_ref = float(state.position[0]), float(state.position[1])
        u_err = float(u) - u_ref
        v_err = float(v) - v_ref

        # ---- 离地阶段：先保持水平、给足总距，避免贴地大俯仰推不动 ----
        if float(own_alt_m) < 1.2:
            self.yaw_filtered *= 0.8
            self.pitch_filtered *= 0.8
            takeoff_thr = 0.65
            self.throttle_filtered = 0.7 * self.throttle_filtered + 0.3 * takeoff_thr
            return StrikeCommand(
                phase=StrikePhase.APPROACH,
                valid=True,
                u_ref_px=u_ref,
                v_ref_px=v_ref,
                u_meas_px=float(u),
                v_meas_px=float(v),
                pitch_cmd_deg=float(self.pitch_filtered),
                roll_cmd_deg=0.0,
                yaw_rate_dps=float(self.yaw_filtered),
                throttle_cmd=float(self.throttle_filtered),
                target_range_m=float("nan"),
                closing_speed_mps=float("nan"),
                t_go_s=float("nan"),
            )

        # ---- 水平：画面误差 -> yaw rate ----
        # 先只用画面水平误差；KF 前馈在实飞中容易激励 yaw 振荡，暂时不用。
        u_cmd = 0.0 if abs(u_err) < 20.0 else u_err
        yaw_rate = 0.4 * u_cmd
        yaw_rate = max(-15.0, min(15.0, yaw_rate))
        self.yaw_filtered = 0.85 * self.yaw_filtered + 0.15 * yaw_rate

        # ---- 视觉测距/测高 ----
        bw = max(1.0, x2 - x1)
        self.bw_f = bw if self.bw_f is None else 0.8 * self.bw_f + 0.2 * bw
        range_est = max(1.0, min(60.0, self.fx * self.target_width_m / max(1.0, self.bw_f)))
        # 相机光轴上仰 44°；目标在画面下方 -> 仰角低于光轴。
        el_deg = (
            float(own_pitch_deg)
            + self.camera_tilt_deg
            + math.degrees(math.atan2(self.h * 0.5 - v, self.fy))
        )
        target_alt_raw = float(own_alt_m) + range_est * math.sin(math.radians(el_deg))
        # bbox 跑出图像/距离离谱时，冻结上一次目标高度估计，而不是跟着自己高度走。
        if not (0.0 <= v <= float(self.h)) or not (0.5 <= range_est <= 30.0):
            target_alt_raw = (
                float(self.target_alt_f)
                if self.target_alt_f is not None
                else float(own_alt_m)
            )
        if self.fixed_target_alt is not None:
            target_alt_raw = float(self.fixed_target_alt)
        target_alt_est = max(0.0, min(float(max_alt_m) + 1.0, target_alt_raw))
        if self.target_alt_f is None:
            self.target_alt_f = target_alt_est
        else:
            self.target_alt_f = 0.8 * self.target_alt_f + 0.2 * target_alt_est

        # ---- 高度环：直接控油门到估计目标高度 ----
        throttle_rc = (
            self.hover_rc
            + self.alt_kp * (float(self.target_alt_f) - float(own_alt_m))
            - self.alt_kd * float(own_vz_mps)
        )
        # 大俯仰下推力垂直分量下降：按 1/cos(pitch) 补偿总距。
        cos_pitch = max(0.5, math.cos(math.radians(float(self.pitch_filtered))))
        throttle_rc += (1.0 / cos_pitch - 1.0) * self.hover_rc
        # 限制爬升速度，避免视觉高度估计误差造成大油门猛冲。
        if float(own_vz_mps) > 2.0:
            throttle_rc -= 180.0 * (float(own_vz_mps) - 2.0)
        if float(own_alt_m) > float(max_alt_m):
            throttle_rc -= 250.0 * (float(own_alt_m) - float(max_alt_m))
        if float(own_alt_m) > float(max_alt_m) + 1.0:
            throttle_rc = 1000.0
        throttle_rc = max(1000.0, min(1850.0, throttle_rc))
        throttle_cmd = (throttle_rc - 1000.0) / 1000.0
        self.throttle_filtered = 0.7 * self.throttle_filtered + 0.3 * throttle_cmd

        # ---- 前推偏置：距离越近越小，避免终端过冲 ----
        if self.simple_attack:
            # 近距离直接碰撞：固定前推小俯仰，不再做终端抬头刹车。
            fwd_bias = 8.0
            pitch = -fwd_bias - 0.05 * v_err
        else:
            fwd_bias = 7.0 * max(-1.5, min(1.0, (range_est - 1.5) / 12.0))
            pitch = -fwd_bias - 0.10 * v_err
            if range_est < 1.5:
                pitch += 8.0  # 终端主动抬头刹车，避免冲过目标
        pitch = max(-20.0, min(20.0, pitch))
        self.pitch_filtered = 0.7 * self.pitch_filtered + 0.3 * pitch

        if range_est < 2.0:
            phase = StrikePhase.TERMINAL
        elif range_est < 8.0:
            phase = StrikePhase.TRACK
        else:
            phase = StrikePhase.APPROACH

        return StrikeCommand(
            phase=phase,
            valid=True,
            u_ref_px=u_ref,
            v_ref_px=v_ref,
            u_meas_px=float(u),
            v_meas_px=float(v),
            pitch_cmd_deg=float(self.pitch_filtered),
            roll_cmd_deg=0.0,
            yaw_rate_dps=float(self.yaw_filtered),
            throttle_cmd=float(self.throttle_filtered),
            target_range_m=float(range_est),
            closing_speed_mps=float("nan"),
            t_go_s=float("nan"),
        )


def annotate(frame: np.ndarray, bbox, phase: str, range_m: float, conf: float,
             truth_range: float, hit: bool) -> np.ndarray:
    out = frame.copy()
    if bbox is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{conf:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 52), (0, 0, 0), -1)
    cv2.putText(out, f"{phase}  est={range_m:5.2f}m  truth={truth_range:5.2f}m  hit={hit}",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"yolo bbox={'none' if bbox is None else 'yes'}",
                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="真实 Gazebo+YOLO 视觉打击")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--interceptor", default="iris_with_Betaflight")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--tilt", type=float, default=44.0)
    parser.add_argument("--target-width", type=float, default=0.8,
                        help="YOLO 估计距离用的机体宽度 m（x500 渲染等效约 0.8）")
    parser.add_argument("--target-alt", type=float, default=None,
                        help="已知攻击高度 m；给定时 direct 高度环直接保持该高度")
    parser.add_argument("--simple-attack", action="store_true",
                        help="近距离直接碰撞模式：固定小俯仰前推，不做终端抬头刹车")
    parser.add_argument("--release-alt", type=float, default=10.0)
    parser.add_argument("--max-alt", type=float, default=8.0, help="direct 模式高度软上限 m")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--search-timeout", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--max-angle", type=float, default=60.0)
    parser.add_argument("--yolo-host", default=os.environ.get("YOLO_HOST", "127.0.0.1"),
                        help="YOLO 服务地址；默认取 $YOLO_HOST，否则本机 127.0.0.1")
    parser.add_argument("--yolo-port", type=int, default=9300)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yaw-sign", type=float, default=-1.0,
                        help="yaw 通道符号；sitl_gazebo_visual_strike 里 yaw_motors_reversed=ON 时为 -1")
    parser.add_argument("--speed-gain", type=float, default=0.8, help="guidance 接近速度增益 m/s per m")
    parser.add_argument("--vel-gain", type=float, default=1.5, help="guidance 速度误差增益")
    parser.add_argument("--detect-crop", type=float, default=1.0,
                        help="只用画面上方该比例做 YOLO；1.0=整帧")
    parser.add_argument("--controller", choices=("direct", "guidance"), default="direct",
                        help="direct=纯图像插值控制；guidance=原 VisualStrikeGuidance")
    parser.add_argument("--direct-attack", action="store_true",
                        help="guidance 模式也不先爬升/不开 ALT_HOLD，直接带视觉推力攻击")
    parser.add_argument("--log", default="/tmp/bf_anti_yolo.csv")
    parser.add_argument("--video", default="/tmp/bf_anti_yolo.mp4")
    parser.add_argument("--plot", default="/tmp/bf_anti_yolo.png")
    args = parser.parse_args()

    if not sitl_reachable():
        print("ERROR: SITL MSP 127.0.0.1:5761 not reachable", flush=True)
        return 2

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = RcState()
    running = [True]
    threading.Thread(target=rc_sender, args=(udp, rc, running), daemon=True).start()

    from run_intercept_mission import offboard
    link = offboard.MspLink()
    source = GazeboPoseSource(args.world, args.interceptor, args.target)
    camera = GazeboCameraSource(args.world, args.interceptor)
    yolo = RemoteYolo(args.yolo_host, args.yolo_port)

    frame0 = None
    deadline = time.time() + 10.0
    while time.time() < deadline and frame0 is None:
        frame0 = camera.frame()
        time.sleep(0.05)
    if frame0 is None:
        print("ERROR: no Gazebo camera frame; is the GUI/rendering path running?", flush=True)
        return 2
    h, w = frame0.shape[:2]
    crop_h = max(32, min(h, int(round(h * max(0.2, min(1.0, args.detect_crop))))))

    intrinsics = CameraIntrinsics.from_fov(w, h, hfov_deg=68.75)
    extrinsics = CameraExtrinsics(tilt_deg=args.tilt)
    guidance = VisualStrikeGuidance(
        camera_tilt_deg=args.tilt,
        image_size=(w, h),
        hfov_deg=68.75,
        hit_radius_m=0.8,
        speed_gain=args.speed_gain,
        velocity_gain=args.vel_gain,
    )
    estimator = TargetEstimator(intrinsics, extrinsics, target_width_m=args.target_width)
    direct = DirectImageController(w, h, hover_throttle=1545.0, fov_deg=68.75,
                                   target_width_m=args.target_width,
                                   target_alt_m=args.target_alt,
                                   simple_attack=args.simple_attack)

    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (w, h))
    if not writer.isOpened():
        print(f"WARNING: cannot open video {args.video}", flush=True)

    rows: list[list] = []
    hit = False
    min_truth_range = float("inf")
    last_est = None
    last_est_time = 0.0
    prev_channels = None
    last_target = None

    try:
        print("arming...", flush=True)
        if not wait_until_armed(link, rc):
            print("ERROR: failed to arm", flush=True)
            return 2
        if args.controller == "direct" or args.direct_attack:
            # 用户要求：44° 相机一旦能检测到就直接打；先垂直离地，不先爬到指定高度。
            print("spooling up; direct attack (takeoff then vision)", flush=True)
            spool_up(rc, target=1700, duration=1.5)
        else:
            print("spooling up / climbing for guidance mode...", flush=True)
            spool_up(rc, target=1545, duration=2.5)
            climb_to(rc, source, link, args.release_alt, timeout=30.0)
            rc.set_aux3(2000)  # ALT_HOLD
        print("searching for YOLO target...", flush=True)

        # ---- search with real YOLO ----
        t_search = time.monotonic()
        acquired = None
        while time.monotonic() - t_search < args.search_timeout:
            own_res = read_own_pose(link, source, last_target)
            if own_res is None:
                time.sleep(0.05)
                continue
            pose, last_target = own_res
            frame = camera.frame()
            if frame is None:
                time.sleep(0.02)
                continue
            bbox = choose_bbox(yolo.detect(frame[:crop_h, :]), min_conf=args.yolo_conf)
            elapsed = time.monotonic() - t_search
            if bbox is not None:
                acquired = (pose, frame, bbox)
                break
            yaw_off = 30.0 * math.sin(2.0 * math.pi * elapsed / 4.0)
            if args.controller == "direct":
                rc.set_rpyt(RC_MID, 1460, 1520,
                            int(RC_MID + yaw_off * 500.0 / 60.0))
            else:
                search_thr = 1700 if args.direct_attack else 1545
                rc.set_rpyt(RC_MID, int(RC_MID + 45.0 * 500.0 / 60.0), search_thr,
                            int(RC_MID + yaw_off * 500.0 / 60.0))
            time.sleep(0.02)
        if acquired is None:
            print("ERROR: YOLO target not acquired", flush=True)
            return 3
        pose, frame, bbox = acquired
        # 从搜索状态平滑接管，避免第一帧 RC 跳变把目标甩出画面。
        prev_channels = rc.snapshot()
        print(f"YOLO target acquired: bbox={bbox}", flush=True)

        start = time.monotonic()
        period = 1.0 / max(1.0, args.rate)
        while time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            own_res = read_own_pose(link, source, last_target)
            if own_res is None:
                # 状态丢失时不要保持上一个爬升/俯冲指令：回中、收油到悬停附近。
                rc.set_rpyt(RC_MID, RC_MID, 1460, RC_MID)
                time.sleep(period)
                continue
            pose, last_target = own_res
            frame = camera.frame()
            if frame is None:
                rc.set_rpyt(RC_MID, RC_MID, 1460, RC_MID)
                time.sleep(0.01)
                continue
            now = time.monotonic()
            dets = yolo.detect(frame[:crop_h, :])
            bbox = choose_bbox(dets, min_conf=args.yolo_conf)
            truth_range = float(np.linalg.norm(pose.target_position_enu - pose.own_position_enu))
            min_truth_range = min(min_truth_range, truth_range)
            hit_truth = truth_range <= 0.9

            est = None
            is_pred = False
            if bbox is None:
                # direct：短时丢失用 KF 外推虚拟 bbox 保持闭环。
                if args.controller == "direct":
                    pred_bbox = direct.predict_bbox(now, max_age_s=0.5)
                    if pred_bbox is not None:
                        bbox = pred_bbox
                        is_pred = True
                # guidance：短时丢失用估计器外推；否则扫描。
                if bbox is None and args.controller == "guidance" and last_est is not None and 0.0 < now - last_est_time < 0.6:
                    est = estimator.predict(now)
                elif bbox is None:
                    if args.controller == "direct":
                        direct.reset_track()
                    yaw_off = 30.0 * math.sin(2.0 * math.pi * (now - start) / 4.0)
                    if args.controller == "direct":
                        rc.set_rpyt(RC_MID, 1460, 1520,
                                    int(RC_MID + yaw_off * 500.0 / 60.0))
                    else:
                        search_thr = 1700 if args.direct_attack else 1545
                        rc.set_rpyt(RC_MID, int(RC_MID + 45.0 * 500.0 / 60.0), search_thr,
                                    int(RC_MID + yaw_off * 500.0 / 60.0))
                    if writer.isOpened():
                        writer.write(annotate(frame, None, "SEARCH", float("nan"), 0.0,
                                              truth_range, hit_truth))
                    if hit_truth:
                        hit = True
                        break
                    time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
                    continue

            if args.controller == "direct":
                cmd = direct.update(period, bbox, now,
                                    own_alt_m=float(pose.own_position_enu[2]),
                                    own_vz_mps=float(pose.own_velocity_enu[2]),
                                    own_pitch_deg=float(pose.own_rpy_deg[1]),
                                    max_alt_m=args.max_alt,
                                    predicted=is_pred)
            else:
                if est is None:
                    est = estimator.update(
                        bbox, pose.own_position_enu, now,
                        yaw_deg=pose.own_rpy_deg[2],
                        pitch_deg=pose.own_rpy_deg[1],
                        roll_deg=pose.own_rpy_deg[0],
                    )
                    last_est = est
                    last_est_time = now
                cmd = guidance.update(
                    est.position_enu,
                    est.velocity_enu,
                    pose.own_position_enu,
                    pose.own_velocity_enu,
                    pose.own_rpy_deg,
                    timestamp=now,
                    dt=period,
                    bbox_xyxy=bbox,
                    valid=True,
                )
            channels = strike_to_rc(cmd, args.max_angle, pose.own_rpy_deg[2],
                                    prev=prev_channels,
                                    althold=(args.controller == "guidance" and not args.direct_attack),
                                    rate_limit=30,
                                    yaw_sign=args.yaw_sign)
            prev_channels = channels
            rc.set_channels(channels)

            det_conf = max(
                [float(d.get("conf", 0.0)) for d in dets if d.get("name") in TARGET_NAMES],
                default=0.0,
            )
            if writer.isOpened():
                writer.write(annotate(frame, bbox, cmd.phase.value, cmd.target_range_m,
                                      det_conf, truth_range, hit_truth))

            est_pos = est.position_enu if est is not None else np.full(3, np.nan)
            est_vel = est.velocity_enu if est is not None else np.full(3, np.nan)
            rows.append([
                now - start, cmd.phase.value, cmd.target_range_m,
                *pose.own_position_enu, *pose.own_velocity_enu,
                *est_pos, *est_vel,
                *pose.target_position_enu, *pose.target_velocity_enu,
                *(bbox if bbox is not None else (float("nan"),) * 4),
                cmd.u_ref_px, cmd.v_ref_px, cmd.u_meas_px, cmd.v_meas_px,
                cmd.az_ref_deg, cmd.el_ref_deg, cmd.pitch_cmd_deg, cmd.roll_cmd_deg,
                cmd.yaw_cmd_deg, cmd.pitch_rate_dps, cmd.yaw_rate_dps,
                cmd.throttle_cmd, int(hit_truth or cmd.hit), truth_range,
            ])
            if int((now - start) * 5) % 5 == 0:
                print(f"t={now - start:5.2f} {cmd.phase.value:<8} est={cmd.target_range_m:6.2f} "
                      f"truth={truth_range:6.2f} u_meas={cmd.u_meas_px:6.1f} "
                      f"pitch={cmd.pitch_cmd_deg:6.1f} hit={hit_truth}", flush=True)
            if hit_truth or cmd.hit:
                hit = True
                break
            time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
    finally:
        if writer.isOpened():
            writer.release()
        rc.idle()
        time.sleep(1.0)
        running[0] = False
        udp.close()
        yolo.close()
        if rows:
            with open(args.log, "w", newline="", encoding="utf-8") as handle:
                writer_csv = csv.writer(handle)
                writer_csv.writerow([
                    "t", "phase", "range",
                    "own_e", "own_n", "own_u", "own_ve", "own_vn", "own_vu",
                    "est_e", "est_n", "est_u", "est_ve", "est_vn", "est_vu",
                    "truth_e", "truth_n", "truth_u", "truth_ve", "truth_vn", "truth_vu",
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "u_ref", "v_ref", "u_meas", "v_meas",
                    "az_ref", "el_ref", "pitch_cmd", "roll_cmd", "yaw_cmd",
                    "pitch_rate_cmd", "yaw_rate_cmd", "throttle_cmd", "hit", "truth_range",
                ])
                writer_csv.writerows(rows)
            print(f"wrote {args.log} ({len(rows)} samples)", flush=True)
        print(f"wrote {args.video}", flush=True)

    print("=== real Gazebo+YOLO visual strike summary ===")
    print(f"hit={hit} min_truth_range={min_truth_range:.3f} m")
    if last_est is not None:
        print(f"last estimated range={last_est.range_m:.3f} m")
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
