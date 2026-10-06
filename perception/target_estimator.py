"""目标状态估计：由检测框和本机位姿估计目标 ENU 位置/速度。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .camera_geometry import (
    CameraExtrinsics,
    CameraIntrinsics,
    earth_los_to_az_el,
    pixel_to_earth_los,
    range_from_bbox_width,
)


@dataclass
class TargetState:
    """目标在 ENU 下的状态。"""

    position_enu: np.ndarray
    velocity_enu: np.ndarray
    range_m: float
    az_deg: float
    el_deg: float
    timestamp: float
    valid: bool = True

    def relative_position(self, own_position_enu: np.ndarray) -> np.ndarray:
        return self.position_enu - own_position_enu


class AlphaBetaFilter:
    """常速度 alpha-beta 滤波器。"""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.1,
        initial_position: Optional[np.ndarray] = None,
        initial_velocity: Optional[np.ndarray] = None,
    ) -> None:
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.position = (
            np.zeros(3, dtype=float)
            if initial_position is None
            else np.array(initial_position, dtype=float)
        )
        self.velocity = (
            np.zeros(3, dtype=float)
            if initial_velocity is None
            else np.array(initial_velocity, dtype=float)
        )
        self.initialized = initial_position is not None

    def reset(self, position: np.ndarray, velocity: Optional[np.ndarray] = None) -> None:
        self.position = np.array(position, dtype=float)
        self.velocity = (
            np.zeros(3, dtype=float)
            if velocity is None
            else np.array(velocity, dtype=float)
        )
        self.initialized = True

    def predict(self, dt: float) -> np.ndarray:
        dt = max(float(dt), 0.0)
        self.position = self.position + self.velocity * dt
        return self.position.copy()

    def update(self, measurement: np.ndarray, dt: float) -> np.ndarray:
        measurement = np.array(measurement, dtype=float)
        if not self.initialized:
            self.reset(measurement)
            return self.position.copy()

        dt = max(float(dt), 1e-6)
        predicted = self.position + self.velocity * dt
        residual = measurement - predicted
        self.position = predicted + self.alpha * residual
        self.velocity = self.velocity + (self.beta / dt) * residual
        return self.position.copy()


class TargetEstimator:
    """由 2D 检测框和本机位姿估计 3D 目标状态。"""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        target_width_m: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.1,
    ) -> None:
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.target_width_m = float(target_width_m)
        self.filter = AlphaBetaFilter(alpha=alpha, beta=beta)
        self.last_timestamp: Optional[float] = None
        self.last_state: Optional[TargetState] = None

    def measurement_to_position(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        own_position_enu: np.ndarray,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> tuple[np.ndarray, float, float, float]:
        """返回 (目标 ENU 位置, range, az, el)。"""
        range_m = range_from_bbox_width(
            bbox_xyxy, self.intrinsics.fx, self.target_width_m
        )
        x1, y1, x2, y2 = bbox_xyxy
        u = (x1 + x2) * 0.5
        v = (y1 + y2) * 0.5
        los_e = pixel_to_earth_los(
            (u, v),
            self.intrinsics,
            self.extrinsics,
            yaw_deg,
            pitch_deg,
            roll_deg,
        )
        az, el = earth_los_to_az_el(los_e)
        position = np.array(own_position_enu, dtype=float) + los_e * range_m
        return position, range_m, az, el

    def update(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        own_position_enu: np.ndarray,
        timestamp: float,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> TargetState:
        position, range_m, az, el = self.measurement_to_position(
            bbox_xyxy, own_position_enu, yaw_deg, pitch_deg, roll_deg
        )
        if self.last_timestamp is None:
            self.filter.reset(position)
            dt = 0.0
        else:
            dt = max(float(timestamp) - self.last_timestamp, 1e-6)
            self.filter.update(position, dt)

        self.last_timestamp = float(timestamp)
        self.last_state = TargetState(
            position_enu=self.filter.position.copy(),
            velocity_enu=self.filter.velocity.copy(),
            range_m=float(range_m),
            az_deg=float(az),
            el_deg=float(el),
            timestamp=float(timestamp),
        )
        return self.last_state

    def predict(self, timestamp: float) -> Optional[TargetState]:
        if self.last_state is None or self.last_timestamp is None:
            return None
        dt = max(float(timestamp) - self.last_timestamp, 0.0)
        pos = self.filter.predict(dt)
        self.last_timestamp = float(timestamp)
        state = TargetState(
            position_enu=pos,
            velocity_enu=self.filter.velocity.copy(),
            range_m=self.last_state.range_m,
            az_deg=self.last_state.az_deg,
            el_deg=self.last_state.el_deg,
            timestamp=float(timestamp),
        )
        self.last_state = state
        return state

    @property
    def state(self) -> Optional[TargetState]:
        return self.last_state
