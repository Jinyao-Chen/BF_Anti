"""三维目标卡尔曼滤波（恒加速度模型），给 APN 提供位置/速度/加速度。

状态 ``x = [p(3), v(3), a(3)]``（ENU，m / m/s / m/s²），测量为位置
（由 bbox 已知尺寸测距 + 视线方向得到）。过程噪声按 jerk 白噪声建模，
用于吸收目标机动；Mahalanobis 门控抑制 YOLO 跳框。
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class TargetKF:
    def __init__(
        self,
        process_jerk: float = 6.0,
        meas_pos_var: float = 1.0,
        init_pos_var: float = 4.0,
        init_vel_var: float = 9.0,
        init_acc_var: float = 25.0,
        gate_chi2: float = 16.27,  # 3 自由度, 99.9%
        max_dt: float = 0.5,
    ) -> None:
        self.q = float(process_jerk)
        self.meas_pos_var = float(meas_pos_var)
        self.gate_chi2 = float(gate_chi2)
        self.max_dt = float(max_dt)
        self._init_pos_var = float(init_pos_var)
        self._init_vel_var = float(init_vel_var)
        self._init_acc_var = float(init_acc_var)

        self.x = np.zeros(9, dtype=float)
        self.P = np.diag(
            [self._init_pos_var] * 3 + [self._init_vel_var] * 3 + [self._init_acc_var] * 3
        ).astype(float)
        self.initialized = False
        self.last_t: Optional[float] = None
        self.rejected = 0
        self._consec_rejected = 0

    # ------------------------------------------------------------------ #
    def reset(self, position: np.ndarray) -> None:
        self.x = np.zeros(9, dtype=float)
        self.x[0:3] = np.asarray(position, dtype=float)
        self.P = np.diag(
            [self._init_pos_var] * 3 + [self._init_vel_var] * 3 + [self._init_acc_var] * 3
        ).astype(float)
        self.initialized = True
        self.last_t = None
        self.rejected = 0
        self._consec_rejected = 0

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def acceleration(self) -> np.ndarray:
        return self.x[6:9].copy()

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * (0.5 * dt * dt)
        F[3:6, 6:9] = np.eye(3) * dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        # 白 jerk 驱动的恒加速度模型（连续的离散近似）。
        q = self.q
        d2, d3, d4 = dt * dt, dt ** 3, dt ** 4
        d5 = dt ** 5
        Q = np.zeros((9, 9))
        Q[0:3, 0:3] = np.eye(3) * (q * d5 / 20.0)
        Q[0:3, 3:6] = np.eye(3) * (q * d4 / 8.0)
        Q[0:3, 6:9] = np.eye(3) * (q * d3 / 6.0)
        Q[3:6, 0:3] = Q[0:3, 3:6]
        Q[3:6, 3:6] = np.eye(3) * (q * d3 / 3.0)
        Q[3:6, 6:9] = np.eye(3) * (q * d2 / 2.0)
        Q[6:9, 0:3] = Q[0:3, 6:9]
        Q[6:9, 3:6] = Q[3:6, 6:9]
        Q[6:9, 6:9] = np.eye(3) * (q * dt)
        return Q

    def predict(self, timestamp: float) -> None:
        if not self.initialized:
            return
        if self.last_t is None:
            self.last_t = float(timestamp)
            return
        dt = min(max(float(timestamp) - self.last_t, 1e-3), self.max_dt)
        self.last_t = float(timestamp)
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def update_position(self, z: np.ndarray, variance: Optional[float] = None) -> bool:
        """位置测量（ENU, m）；返回 True 表示通过门控并完成更新。"""
        z = np.asarray(z, dtype=float)[:3]
        if not self.initialized:
            self.reset(z)
            return True
        r = float(self.meas_pos_var if variance is None else variance)
        H = np.zeros((3, 9))
        H[0:3, 0:3] = np.eye(3)
        y = z - H @ self.x
        S = H @ self.P @ H.T + np.eye(3) * r
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False
        d2 = float(y.T @ S_inv @ y)
        if d2 > self.gate_chi2:
            self.rejected += 1
            self._consec_rejected += 1
            if self._consec_rejected >= 5:
                self.reset(z)
                return True
            return False
        self._consec_rejected = 0
        R = np.eye(3) * r
        K = self.P @ H.T @ S_inv
        I_KH = np.eye(9) - K @ H
        self.x = self.x + K @ y
        # Joseph form preserves symmetry and positive-definiteness
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return True


__all__ = ["TargetKF"]
