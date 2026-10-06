#!/usr/bin/env python3
"""把 Gazebo 里的顶层模型 chase_cam 用 set_pose 钉在拦截机位姿上。

为什么需要它：本机 gz-sim 的相机渲染不跟随**嵌套模型**的 link（实测机体俯仰 40°、
平移 25 m，画面帧间差仍为 0.00），所以挂在内层模型上的相机永远输出那一帧。
把相机改挂在顶层模型上、并用 set_pose 驱动，渲染即可跟随（本文件即该驱动）。
"""

from __future__ import annotations

import time



class ChaseCameraPublisher:
    """发布 /world/<world>/set_pose，让 chase_cam 跟随给定位姿。"""

    def __init__(self, world: str = "betaflight_world", model: str = "chase_cam",
                 rate_hz: float = 30.0) -> None:
        import gz.transport13 as gztransport
        from gz.msgs10.pose_pb2 import Pose

        self.Pose = Pose
        self.model = model
        self.period = 1.0 / max(1.0, rate_hz)
        self._last = 0.0
        self._sent = 0
        self.node = gztransport.Node()
        self.pub = self.node.advertise(f"/world/{world}/set_pose", Pose)
        if not self.pub:
            raise RuntimeError("advertise set_pose failed")

    def update(self, position_enu, rpy_deg, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < self.period:
            return
        self._last = now
        # 姿态：ENU(FLU) → Gazebo(x=N, y=E, z=U) 需要交换 x/y，
        # 这里直接使用 Gazebo 世界的四元数来源更稳，故由调用方给 Gazebo 原生四元数。
        msg = self._make(position_enu, rpy_deg)
        self.pub.publish(msg)
        self._sent += 1

    def _make(self, position_enu, rpy_deg):
        import math

        msg = self.Pose()
        msg.name = self.model
        # ENU(E,N,U) → Gazebo(x,y,z) = (N, E, U)
        msg.position.x = float(position_enu[1])
        msg.position.y = float(position_enu[0])
        msg.position.z = float(position_enu[2])
        roll, pitch, yaw = (math.radians(float(v)) for v in rpy_deg)
        # Gazebo 中模型姿态：绕 z 的 yaw 用罗盘角，roll/pitch 直接对应
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        msg.orientation.w = cr * cp * cy + sr * sp * sy
        msg.orientation.x = sr * cp * cy - cr * sp * sy
        msg.orientation.y = cr * sp * cy + sr * cp * sy
        msg.orientation.z = cr * cp * sy - sr * sp * cy
        return msg

    @property
    def sent(self) -> int:
        return self._sent
