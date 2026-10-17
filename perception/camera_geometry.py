"""44° 上仰相机的几何工具。

坐标系约定
----------
Earth (ENU)
    x = East, y = North, z = Up。
Body
    x = 机头前方, y = 右方, z = 上方。
Camera
    x = 图像右方, y = 图像下方, z = 光轴前方。
    这对应 Gazebo / 常规针孔相机坐标系。

相机安装
--------
相机光轴相对机体 +X 向上抬高 ``tilt_deg``，绕机体 +Y 轴旋转。
因此相机光轴在机体系为::

    z_c_b = (cos(tilt), 0, sin(tilt))

相机 x 轴与机体 y 轴重合，y 轴为 z_c_b × x_c_b。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """针孔相机内参。"""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov_deg: float,
        vfov_deg: float | None = None,
    ) -> "CameraIntrinsics":
        """由视场角构造内参。

        如果只给 hfov，则根据图像宽高比推算 vfov。
        """
        hfov = np.deg2rad(hfov_deg)
        fx = (width * 0.5) / np.tan(hfov * 0.5)
        if vfov_deg is None:
            # 假设像素为正方形
            fy = fx
        else:
            vfov = np.deg2rad(vfov_deg)
            fy = (height * 0.5) / np.tan(vfov * 0.5)
        return cls(
            width=width,
            height=height,
            fx=float(fx),
            fy=float(fy),
            cx=width * 0.5,
            cy=height * 0.5,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    """相机相对机体的安装角。

    yaw_offset_deg 是相机光轴相对机体 +X 的水平偏角，正值 = 从机尾看
    光轴向右偏（与罗盘/航向角同向）。tilt_deg 是上仰角。
    """

    tilt_deg: float = 44.0
    yaw_offset_deg: float = 0.0

    def rotation_body_from_camera(self) -> np.ndarray:
        """返回 3x3 矩阵 R_bc，使 v_body = R_bc @ v_camera。

        机体使用 FLU：x 前、y 左、z 上（右手系）。
        相机使用 x 右、y 下、z 前（右手系）。
        """
        t = np.deg2rad(self.tilt_deg)
        c, s = np.cos(t), np.sin(t)
        # columns: x_c_b, y_c_b, z_c_b
        x_c_b = np.array([0.0, -1.0, 0.0])
        z_c_b = np.array([c, 0.0, s])
        y_c_b = np.cross(z_c_b, x_c_b)
        base = np.column_stack([x_c_b, y_c_b, z_c_b])
        if self.yaw_offset_deg == 0.0:
            return base
        # 光轴右偏 = 绕机体 +Z（上）顺时针 = 转 -yaw_offset。
        yaw = np.deg2rad(-float(self.yaw_offset_deg))
        cy, sy = np.cos(yaw), np.sin(yaw)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        return rz @ base


def rotation_earth_from_body(
    yaw_deg: float, pitch_deg: float, roll_deg: float
) -> np.ndarray:
    """由 yaw/pitch/roll 构造机体 FLU -> ENU 旋转矩阵。

    yaw   : 罗盘航向，0 = 北，90 = 东。
    pitch : 抬头为正。
    roll  : 右滚为正（右翼下压）。
    """
    psi = np.deg2rad(yaw_deg)
    theta = np.deg2rad(pitch_deg)
    phi = np.deg2rad(roll_deg)

    heading = np.array([np.sin(psi), np.cos(psi), 0.0])
    left_h = np.array([-np.cos(psi), np.sin(psi), 0.0])
    up = np.array([0.0, 0.0, 1.0])

    # 机体轴（FLU）：
    #   x_b = cosθ·heading + sinθ·up
    #   抬头后机体的"上"轴 up_b = cosθ·up - sinθ·heading
    #   绕机体 x 轴滚转 φ 后 y_b = cosφ·left_h + sinφ·up_b
    # 早先的实现把 y_b 写成 cosφ·left_h + sinφ·up，漏掉了 up_b 里的
    # heading 分量，x_b·y_b ≠ 0（矩阵非正交），非零滚转时视线换算会偏。
    up_b = np.cos(theta) * up - np.sin(theta) * heading
    x_b = np.cos(theta) * heading + np.sin(theta) * up
    y_b = np.cos(phi) * left_h + np.sin(phi) * up_b
    z_b = np.cross(x_b, y_b)
    z_b = z_b / np.linalg.norm(z_b)
    return np.column_stack([x_b, y_b, z_b])


def rotation_enu_body_to_rpy(r_eb: np.ndarray) -> tuple[float, float, float]:
    """``R_eb``（机体 FLU -> ENU）-> ``(roll, pitch, yaw)``，单位度。

    与 :func:`rotation_earth_from_body` 互逆，供制导层把期望推力方向
    ``z_b`` 转成机体姿态角。
    """
    r = np.asarray(r_eb, dtype=float)
    x_b = r[:, 0]
    y_b = r[:, 1]
    yaw = float(np.rad2deg(np.arctan2(x_b[0], x_b[1])))
    pitch = float(np.rad2deg(np.arcsin(np.clip(x_b[2], -1.0, 1.0))))
    psi = np.deg2rad(yaw)
    theta = np.deg2rad(pitch)
    heading = np.array([np.sin(psi), np.cos(psi), 0.0])
    left_h = np.array([-np.cos(psi), np.sin(psi), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    up_b = np.cos(theta) * up - np.sin(theta) * heading
    roll = float(np.rad2deg(np.arctan2(float(np.dot(y_b, up_b)), float(np.dot(y_b, left_h)))))
    return roll, pitch, yaw


def pixel_to_camera_ray(pixel_uv: tuple[float, float], intr: CameraIntrinsics) -> np.ndarray:
    """像素坐标 -> 相机系单位视线。"""
    u, v = pixel_uv
    x = (u - intr.cx) / intr.fx
    y = (v - intr.cy) / intr.fy
    z = 1.0
    ray = np.array([x, y, z], dtype=float)
    return ray / np.linalg.norm(ray)


def camera_ray_to_body(ray_cam: np.ndarray, extr: CameraExtrinsics) -> np.ndarray:
    """相机系单位视线 -> 机体系单位视线。"""
    return extr.rotation_body_from_camera() @ ray_cam


def body_ray_to_earth(
    ray_body: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> np.ndarray:
    """机体系单位视线 -> ENU 单位视线。"""
    return rotation_earth_from_body(yaw_deg, pitch_deg, roll_deg) @ ray_body


def pixel_to_earth_los(
    pixel_uv: tuple[float, float],
    intr: CameraIntrinsics,
    extr: CameraExtrinsics,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """像素 -> 相机 -> 机体 -> ENU 的单位视线。"""
    ray_cam = pixel_to_camera_ray(pixel_uv, intr)
    ray_body = camera_ray_to_body(ray_cam, extr)
    ray_earth = body_ray_to_earth(ray_body, yaw_deg, pitch_deg, roll_deg)
    return ray_earth / np.linalg.norm(ray_earth)


def earth_los_to_az_el(los_earth: np.ndarray) -> tuple[float, float]:
    """ENU 单位视线 -> (方位角, 俯仰角)，单位度。

    方位角为罗盘角：0 = 北，90 = 东，范围 [0, 360)。
    俯仰角抬头为正，范围 [-90, 90]。
    """
    los = los_earth / np.linalg.norm(los_earth)
    east, north, up = float(los[0]), float(los[1]), float(los[2])
    az = np.rad2deg(np.arctan2(east, north)) % 360.0
    el = np.rad2deg(np.arcsin(np.clip(up, -1.0, 1.0)))
    return float(az), float(el)


def desired_body_pitch_for_elevation(
    elevation_deg: float, tilt_deg: float = 44.0
) -> float:
    """让相机光轴对准给定俯仰角所需机体俯仰角。

    相机比机头高 ``tilt_deg``，所以::

        body_pitch = elevation - tilt
    """
    return float(elevation_deg - tilt_deg)


def target_bearing_from_bbox_center(
    bbox_xyxy: tuple[float, float, float, float],
    intr: CameraIntrinsics,
    extr: CameraExtrinsics,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> tuple[float, float]:
    """由目标框中心像素得到目标在地球系的 (方位角, 俯仰角)。"""
    x1, y1, x2, y2 = bbox_xyxy
    u = (x1 + x2) * 0.5
    v = (y1 + y2) * 0.5
    los_e = pixel_to_earth_los((u, v), intr, extr, yaw_deg, pitch_deg, roll_deg)
    return earth_los_to_az_el(los_e)


def range_from_bbox_width(
    bbox_xyxy: tuple[float, float, float, float],
    fx: float,
    target_width_m: float,
) -> float:
    """由目标框像素宽度估计距离。

    ``range = fx * target_width_m / bbox_width_px``
    """
    x1, _, x2, _ = bbox_xyxy
    w_px = max(float(x2) - float(x1), 1e-6)
    return float(fx * target_width_m / w_px)
