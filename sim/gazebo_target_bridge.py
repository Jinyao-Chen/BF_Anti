#!/usr/bin/env python3
"""Gazebo 固定靶机 -> 44° 上仰相机 -> 制导 -> RC 的拦截桥。

数据流（每个控制周期）::

    Gazebo pose (真值)  ->  机体姿态 / 相对位置
        ->  44° 相机投影 (可选：bbox 模拟检测)  ->  TargetEstimator
        ->  InterceptGuidance (APPROACH/TRACK/TERMINAL)
        ->  guidance_to_track_cmd  ->  RC 通道帧  ->  (可选) UDP 给 SITL

设计要点：

* 相机光轴固连机体、相对机头抬高 ``--tilt`` 度（默认 44），因此光轴瞄准
  目标所需的机体俯仰角是 ``elevation - tilt``（同高正前方目标 => -44°）。
* 桥本身是可离线测试的纯逻辑（``InterceptBridge``）；只有 ``GazeboPoseSource``
  依赖 gz-transport，实机/仿真时才会用到。
* ``--perception camera`` 会把真值目标投影成像素 bbox 再走一遍
  ``TargetEstimator``，从而把相机几何、测距、滤波都串进闭环；
  ``--perception truth`` 则直接使用真值（用于对照标定）。

用法::

    # 只打印制导/RC 指令，不发给飞控
    python3 sim/gazebo_target_bridge.py --duration 20

    # 真值直接给制导（跳过视觉），并发 RC 给 SITL
    python3 sim/gazebo_target_bridge.py --perception truth --send-rc --duration 20
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guidance.intercept_guidance import GuidanceCommand, InterceptGuidance  # noqa: E402
from guidance.track_cmd_adapter import guidance_to_track_cmd  # noqa: E402
from perception.camera_geometry import (  # noqa: E402
    CameraExtrinsics,
    CameraIntrinsics,
    rotation_earth_from_body,
)
from perception.target_estimator import TargetEstimator, TargetState  # noqa: E402

# Gazebo Harmonic nests iris_with_standoffs inside the moving iris_with_Betaflight
# model. pose/info reports the nested model pose relative to its parent, so the
# inner name reads a stale pose; the outer iris_with_Betaflight carries the world
# motion (verified against the firmware altitude/GPS).
DEFAULT_INTERCEPTOR = "iris_with_Betaflight"
DEFAULT_TARGET = "target_drone"
RC_MIN, RC_MAX, RC_MID = 1000, 2000, 1500


# --------------------------------------------------------------------------- #
# 坐标系与姿态
# --------------------------------------------------------------------------- #
def gazebo_xyz_to_enu(x: float, y: float, z: float) -> np.ndarray:
    """Gazebo world (x,y,z) -> ENU (E,N,U)。

    **2026-09-28 标定修正**：原实现取 东=y_g、北=x_g、上=z_g，该矩阵 det=-1（反射）。
    用真实相机画面（顶层 cam_link 渲染）与解析投影对照 14 组样本实测：
        旧映射      Δu=+260 px  Δv=-284 px
        本映射      Δu= +2.3 px Δv= +1.5 px（σ≈2 px）
    即 Gazebo 的 +y 是**西**：E=-y_g、N=x_g、U=z_g。
    历史上所有场景靶机都放在 y_g=0 的正前方（方位≈0），反射在方位≈0 处无差别，
    所以这个缺陷一直没暴露。
    """
    return np.array([-float(y), float(x), float(z)], dtype=float)


def quat_to_rotation_enu_body(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Gazebo 姿态四元数 -> ENU 系下 机体 FLU 的旋转矩阵 (body->ENU)。"""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n <= 0.0:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    r_gb = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    # 世界系：ENU = M @ Gazebo，M = [[0,-1,0],[1,0,0],[0,0,1]]（det=+1，真旋转）。
    # 机体系：Gazebo 模型机体系本身是 FLU(x前-y左-z上)，**不再**需要 FRD→FLU 翻转
    # （旧实现右乘 diag(1,-1,-1) 后再乘反射阵，得到 det=-1 的"反射矩阵"，
    #  俯仰正确但**方位被镜像**——实测 Δv≈0、Δu 符号相反地漂移即由此而来）。
    p = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return p @ r_gb


def rpy_from_rotation_enu_body(r_eb: np.ndarray) -> tuple[float, float, float]:
    """从 body->ENU 旋转矩阵提取 (roll, pitch, yaw)，约定同
    ``rotation_earth_from_body``：yaw 为罗盘航向，pitch 抬头为正，roll 右滚为正。"""
    x_b = r_eb[:, 0]
    y_b = r_eb[:, 1]
    yaw = math.degrees(math.atan2(x_b[0], x_b[1]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, x_b[2]))))
    psi = math.radians(yaw)
    theta = math.radians(pitch)
    north = np.array([0.0, 1.0, 0.0])
    heading = np.array([math.sin(psi), math.cos(psi), 0.0])
    left_h = np.array([-math.cos(psi), math.sin(psi), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    up_b = math.cos(theta) * up - math.sin(theta) * heading
    roll = math.degrees(math.atan2(float(np.dot(y_b, up_b)), float(np.dot(y_b, left_h))))
    _ = north
    return roll, pitch, yaw


@dataclass
class PoseState:
    """一次采样到的机体/目标状态。"""

    own_position_enu: np.ndarray
    own_velocity_enu: np.ndarray
    own_rpy_deg: tuple[float, float, float]
    target_position_enu: np.ndarray
    timestamp: float
    target_velocity_enu: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )


def project_target_to_bbox(
    pose: PoseState,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    target_width_m: float = 1.0,
) -> Optional[tuple[float, float, float, float]]:
    """把真值目标投影成 bbox（模拟检测器输出），相机看不到则返回 None。"""
    r_eb = rotation_earth_from_body(*pose.own_rpy_deg)
    rel_earth = pose.target_position_enu - pose.own_position_enu
    rng = float(np.linalg.norm(rel_earth))
    if rng < 1e-6:
        return None
    rel_body = r_eb.T @ rel_earth
    r_bc = extrinsics.rotation_body_from_camera()
    rel_cam = r_bc.T @ rel_body
    if rel_cam[2] <= 1e-6:
        return None
    u = intrinsics.fx * rel_cam[0] / rel_cam[2] + intrinsics.cx
    v = intrinsics.fy * rel_cam[1] / rel_cam[2] + intrinsics.cy
    w = intrinsics.fx * float(target_width_m) / rng
    if w < 2.0:
        w = 2.0
    h = w
    if not (-intrinsics.cx * 2.0 <= u <= intrinsics.width + intrinsics.cx * 2.0):
        return None
    if not (-intrinsics.cy * 2.0 <= v <= intrinsics.height + intrinsics.cy * 2.0):
        return None
    return (u - w * 0.5, v - h * 0.5, u + w * 0.5, v + h * 0.5)


# --------------------------------------------------------------------------- #
# 桥
# --------------------------------------------------------------------------- #
class InterceptBridge:
    """真值/视觉 -> 制导 -> RC 帧 的在线桥（无 IO，便于单元测试）。"""

    def __init__(
        self,
        camera_tilt_deg: float = 44.0,
        image_size: tuple[int, int] = (1280, 720),
        hfov_deg: float = 70.0,
        target_width_m: float = 1.0,
        max_angle_deg: float = 45.0,
        use_camera: bool = True,
    ) -> None:
        self.intrinsics = CameraIntrinsics.from_fov(*image_size, hfov_deg=hfov_deg)
        self.extrinsics = CameraExtrinsics(tilt_deg=camera_tilt_deg)
        self.target_width_m = float(target_width_m)
        self.max_angle_deg = float(max_angle_deg)
        self.use_camera = bool(use_camera)
        self.estimator = TargetEstimator(
            self.intrinsics, self.extrinsics, target_width_m=self.target_width_m
        )
        self.guidance = InterceptGuidance(camera_tilt_deg=camera_tilt_deg)
        self.last_command: Optional[GuidanceCommand] = None
        self.last_bbox: Optional[tuple[float, float, float, float]] = None

    def target_state(self, pose: PoseState) -> Optional[TargetState]:
        if not self.use_camera:
            rel = pose.target_position_enu - pose.own_position_enu
            rng = float(np.linalg.norm(rel))
            return TargetState(
                position_enu=pose.target_position_enu.copy(),
                velocity_enu=pose.target_velocity_enu.copy(),
                range_m=rng,
                az_deg=float(np.rad2deg(np.arctan2(rel[0], rel[1])) % 360.0),
                el_deg=float(np.rad2deg(np.arctan2(rel[2], float(np.hypot(rel[0], rel[1]))))),
                timestamp=pose.timestamp,
            )
        bbox = project_target_to_bbox(
            pose, self.intrinsics, self.extrinsics, self.target_width_m
        )
        self.last_bbox = bbox
        if bbox is None:
            return None
        return self.estimator.update(
            bbox,
            pose.own_position_enu,
            pose.timestamp,
            yaw_deg=pose.own_rpy_deg[2],
            pitch_deg=pose.own_rpy_deg[1],
            roll_deg=pose.own_rpy_deg[0],
        )

    def step(self, pose: PoseState) -> tuple[GuidanceCommand, dict]:
        target = self.target_state(pose)
        cmd = self.guidance.update(
            target,
            pose.own_position_enu,
            pose.own_velocity_enu,
            pose.timestamp,
        )
        track = guidance_to_track_cmd(
            cmd, max_angle_deg=self.max_angle_deg, timestamp=pose.timestamp
        )
        frame = {
            "roll": track.roll,
            "pitch": track.pitch,
            "throttle": track.throttle,
            "yaw": track.yaw,
        }
        self.last_command = cmd
        return cmd, frame


# --------------------------------------------------------------------------- #
# Gazebo pose 真值源
# --------------------------------------------------------------------------- #
class GazeboPoseSource:
    """订阅 /world/<world>/pose/info，缓存拦截机与靶机的真值位姿。"""

    def __init__(
        self,
        world: str = "betaflight_world",
        interceptor: str = DEFAULT_INTERCEPTOR,
        target: str = DEFAULT_TARGET,
    ) -> None:
        import gz.transport13 as gztransport  # 延迟导入，离线测试不需要
        from gz.msgs10.pose_v_pb2 import Pose_V

        self.Pose_V = Pose_V
        self.interceptor = interceptor
        self.target = target
        self._msg = None
        self._t = 0.0
        self._last_own_enu: Optional[np.ndarray] = None
        self._last_target_enu: Optional[np.ndarray] = None
        self._last_pose_time: Optional[float] = None
        self.node = gztransport.Node()
        topic = f"/world/{world}/pose/info"
        if not self.node.subscribe(Pose_V, topic, self._on_pose):
            raise RuntimeError(f"failed to subscribe {topic}")

    def _on_pose(self, msg) -> None:
        self._msg = msg
        self._t = time.monotonic()

    def pose(self) -> Optional[PoseState]:
        msg = self._msg
        if msg is None:
            return None
        own = tgt = None
        for p in msg.pose:
            if p.name == self.interceptor and own is None:
                own = p
            elif p.name == self.target and tgt is None:
                tgt = p
        if own is None or tgt is None:
            return None
        r_eb = quat_to_rotation_enu_body(
            own.orientation.x, own.orientation.y, own.orientation.z, own.orientation.w
        )
        rpy = rpy_from_rotation_enu_body(r_eb)
        own_enu = gazebo_xyz_to_enu(own.position.x, own.position.y, own.position.z)
        target_enu = gazebo_xyz_to_enu(tgt.position.x, tgt.position.y, tgt.position.z)

        # 用相邻真值位置差分估计速度；20 Hz 控制桥下比“恒为零”更接近真实，
        # 让 PN 的接近速度和速度外环前馈都能工作。
        now = self._t
        dt = (now - self._last_pose_time) if self._last_pose_time is not None else 0.0
        own_vel = np.zeros(3, dtype=float)
        target_vel = np.zeros(3, dtype=float)
        if 0.005 < dt < 0.5:
            if self._last_own_enu is not None:
                own_vel = (own_enu - self._last_own_enu) / dt
            if self._last_target_enu is not None:
                target_vel = (target_enu - self._last_target_enu) / dt
        self._last_own_enu = own_enu.copy()
        self._last_target_enu = target_enu.copy()
        self._last_pose_time = now

        return PoseState(
            own_position_enu=own_enu,
            own_velocity_enu=own_vel,
            own_rpy_deg=rpy,
            target_position_enu=target_enu,
            target_velocity_enu=target_vel,
            timestamp=now,
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _rc_frame_to_udp_frame(frame: dict, aux1: int = 2000, aux2: int = 2000) -> list[int]:
    """TrackCmd 风格 RC 帧 -> Betaflight UDP RC 通道数组。

    ``aux1`` 对应 AUX1/ARM，``aux2`` 对应 AUX2/ANGLE。BF_Anti 直接 RC
    跟踪模式要求 AUX2 高位常开，因此默认 2000。
    """
    channels = [RC_MID] * 16
    channels[0] = int(max(RC_MIN, min(RC_MAX, frame["roll"])))
    channels[1] = int(max(RC_MIN, min(RC_MAX, frame["pitch"])))
    channels[2] = int(max(RC_MIN, min(RC_MAX, frame["throttle"])))
    channels[3] = int(max(RC_MIN, min(RC_MAX, frame["yaw"])))
    channels[4] = int(aux1)
    channels[5] = int(aux2)
    return channels


def main() -> int:
    ap = argparse.ArgumentParser(description="Gazebo 固定靶机拦截桥")
    ap.add_argument("--world", default="betaflight_world")
    ap.add_argument("--interceptor", default=DEFAULT_INTERCEPTOR)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--tilt", type=float, default=44.0, help="相机相对机头上仰角(度)")
    ap.add_argument("--rate", type=float, default=20.0, help="桥控制频率 Hz")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--perception", choices=("camera", "truth"), default="camera")
    ap.add_argument("--target-width", type=float, default=1.0, help="靶机等效宽度 m")
    ap.add_argument("--max-angle", type=float, default=45.0, help="RC 姿态角满量程(度)")
    ap.add_argument("--send-rc", action="store_true", help="把 RC 帧发给 SITL (UDP 9004)")
    ap.add_argument("--log", default=None, help="可选 CSV 输出路径")
    args = ap.parse_args()

    bridge = InterceptBridge(
        camera_tilt_deg=args.tilt,
        target_width_m=args.target_width,
        max_angle_deg=args.max_angle,
        use_camera=(args.perception == "camera"),
    )
    source = GazeboPoseSource(args.world, args.interceptor, args.target)
    sock = None
    if args.send_rc:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    rows = []
    period = 1.0 / max(1.0, args.rate)
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            pose = source.pose()
            if pose is None:
                time.sleep(period)
                continue
            cmd, frame = bridge.step(pose)
            if sock is not None:
                import struct

                channels = _rc_frame_to_udp_frame(frame)
                sock.sendto(
                    struct.pack("<d" + "H" * 16, time.time(), *channels),
                    ("127.0.0.1", 9004),
                )
            print(
                f"t={pose.timestamp - start:6.2f} {cmd.phase.value:<8} "
                f"range={cmd.target_range_m:6.2f}m az={cmd.az_deg:6.1f} el={cmd.el_deg:6.1f} "
                f"pitch_cmd={cmd.pitch_cmd_deg:6.1f} thr={cmd.throttle_cmd:4.2f} "
                f"| RC r={frame['roll']} p={frame['pitch']} t={frame['throttle']} y={frame['yaw']}",
                flush=True,
            )
            rows.append((pose.timestamp - start, cmd, frame))
            time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
    except KeyboardInterrupt:
        pass
    finally:
        if sock is not None:
            sock.close()
    if args.log and rows:
        with open(args.log, "w") as f:
            f.write("t,phase,range_m,az_deg,el_deg,pitch_cmd_deg,throttle,rc_roll,rc_pitch,rc_throttle,rc_yaw\n")
            for t, cmd, frame in rows:
                f.write(
                    f"{t:.3f},{cmd.phase.value},{cmd.target_range_m:.3f},{cmd.az_deg:.2f},"
                    f"{cmd.el_deg:.2f},{cmd.pitch_cmd_deg:.2f},{cmd.throttle_cmd:.3f},"
                    f"{frame['roll']},{frame['pitch']},{frame['throttle']},{frame['yaw']}\n"
                )
        print(f"wrote {args.log} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
