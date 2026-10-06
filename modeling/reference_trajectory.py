"""PX4 风格的多旋翼参考轨迹生成（jerk-limited）。

这个模块是 PX4 `PositionSmoothing` / `VelocitySmoothing` 的纯 Python
等价实现，用于：

* 在建模/仿真中先验证“参考先逐步插值，再计算误差”的闭环；
* 作为 `~/betaflight` 固件侧 `position_reference.c` 的黄金参考；
* 为视觉打击控制律生成图像平面/ENU 的逐步跟踪参考。

核心思想
--------
直接把阶跃目标位置送到位置误差环，会让参考速度/加速度出现瞬时跳变，
速度环和姿态环必须靠高增益追赶，容易振铃、掉高、参数难调。PX4 的做法是
先由位置环生成一个“不平滑速度设定值”，再用 `VelocitySmoothing` 做
jerk 限制的积分，得到平滑的 `position/velocity/acceleration/jerk` 参考；
位置环误差是对当前插值参考的误差，而不是对最终目标的误差。

算法来源
--------
PX4-Autopilot `src/lib/motion_planning/VelocitySmoothing.cpp` 与
`PositionSmoothing.cpp`（BSD-3-Clause）。这里保留其 T1/T2/T3 三段式
jerk 积分结构，并做少量工程化处理（除零保护、3D 轴同步）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

_EPS = 1e-9


def _sign(value: float) -> int:
    if value > _EPS:
        return 1
    if value < -_EPS:
        return -1
    return 0


@dataclass
class ReferenceState:
    """一次参考轨迹更新的完整输出。"""

    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray
    unsmoothed_velocity: np.ndarray


class VelocitySmoothing1D:
    """单轴 jerk-limited 速度平滑器（PX4 VelocitySmoothing 的 Python 版）。"""

    def __init__(
        self,
        max_jerk: float,
        max_accel: float,
        max_vel: float,
        initial_accel: float = 0.0,
        initial_vel: float = 0.0,
        initial_pos: float = 0.0,
    ) -> None:
        self.max_jerk = max(float(max_jerk), 0.0)
        self.max_accel = max(float(max_accel), 0.0)
        self.max_vel = max(float(max_vel), 0.0)
        self.reset(initial_accel, initial_vel, initial_pos)

    # -- 状态管理 -------------------------------------------------------- #
    def reset(self, accel: float = 0.0, vel: float = 0.0, pos: float = 0.0) -> None:
        self._jerk = 0.0
        self._accel = float(accel)
        self._vel = float(vel)
        self._pos = float(pos)
        self._state_init = (self._accel, self._vel, self._pos)
        self._vel_sp = 0.0
        self._direction = 0
        self._T1 = self._T2 = self._T3 = 0.0
        self._local_time = 0.0

    # -- 只读状态 -------------------------------------------------------- #
    @property
    def position(self) -> float:
        return self._pos

    @property
    def velocity(self) -> float:
        return self._vel

    @property
    def acceleration(self) -> float:
        return self._accel

    @property
    def jerk(self) -> float:
        return self._jerk

    @property
    def total_time(self) -> float:
        return self._T1 + self._T2 + self._T3

    @property
    def durations(self) -> tuple[float, float, float]:
        return (self._T1, self._T2, self._T3)

    # -- 速度设定值 -> T1/T2/T3 ------------------------------------------ #
    def set_velocity_setpoint(self, vel_setpoint: float) -> None:
        if self.max_vel <= _EPS:
            self._vel_sp = 0.0
        else:
            self._vel_sp = float(np.clip(vel_setpoint, -self.max_vel, self.max_vel))
        self._local_time = 0.0
        self._state_init = (self._accel, self._vel, self._pos)

        if self.max_jerk <= _EPS or self.max_accel <= _EPS:
            # 退化为直接速度跟踪，避免除零。
            self._direction = 0
            self._T1 = self._T2 = self._T3 = 0.0
            return

        self._direction = self._compute_direction()
        self._update_durations_minimize_total_time()

    def _compute_direction(self) -> int:
        vel_zero_acc = self._velocity_at_zero_acc()
        direction = _sign(self._vel_sp - vel_zero_acc)
        if direction == 0:
            direction = _sign(self._accel)
        return direction

    def _velocity_at_zero_acc(self) -> float:
        if abs(self._accel) <= _EPS:
            return self._vel
        # 用最大反向 jerk 先把当前加速度削到 0。
        j_zero_acc = -_sign(self._accel) * self.max_jerk
        if abs(j_zero_acc) <= _EPS:
            return self._vel
        t_zero_acc = -self._accel / j_zero_acc
        return (
            self._vel
            + self._accel * t_zero_acc
            + 0.5 * j_zero_acc * t_zero_acc * t_zero_acc
        )

    @staticmethod
    def _saturate_t1_for_accel(
        a0: float, j_max: float, t1: float, a_max: float
    ) -> float:
        accel_t1 = a0 + j_max * t1
        if accel_t1 > a_max:
            return (a_max - a0) / j_max
        if accel_t1 < -a_max:
            return (-a_max - a0) / j_max
        return t1

    def _compute_t1(self, a0: float, v3: float, j_max: float, a_max: float) -> float:
        delta = 2.0 * a0 * a0 + 4.0 * j_max * v3
        if delta < 0.0:
            return 0.0

        sqrt_delta = float(np.sqrt(delta))
        t1_plus = (-a0 + 0.5 * sqrt_delta) / j_max
        t1_minus = (-a0 - 0.5 * sqrt_delta) / j_max
        t3_plus = a0 / j_max + t1_plus
        t3_minus = a0 / j_max + t1_minus

        t1 = 0.0
        if t1_plus >= 0.0 and t3_plus >= 0.0:
            t1 = t1_plus
        elif t1_minus >= 0.0 and t3_minus >= 0.0:
            t1 = t1_minus

        t1 = self._saturate_t1_for_accel(a0, j_max, t1, a_max)
        return max(t1, 0.0)

    def _compute_t2(
        self, t1: float, t3: float, a0: float, v3: float, j_max: float
    ) -> float:
        den = a0 + j_max * t1
        t2 = 0.0
        if abs(den) > _EPS:
            t2 = (
                -0.5 * t1 * t1 * j_max
                - t1 * t3 * j_max
                - t1 * a0
                + 0.5 * t3 * t3 * j_max
                - t3 * a0
                + v3
            ) / den
        return max(t2, 0.0)

    @staticmethod
    def _compute_t3(t1: float, a0: float, j_max: float) -> float:
        return max(a0 / j_max + t1, 0.0)

    def _update_durations_minimize_total_time(self) -> None:
        jerk_max_t1 = self._direction * self.max_jerk
        delta_v = self._vel_sp - self._vel
        if abs(jerk_max_t1) <= _EPS:
            self._T1 = self._T2 = self._T3 = 0.0
            return

        self._T1 = self._compute_t1(self._accel, delta_v, jerk_max_t1, self.max_accel)
        self._T3 = self._compute_t3(self._T1, self._accel, jerk_max_t1)
        self._T2 = self._compute_t2(
            self._T1, self._T3, self._accel, delta_v, jerk_max_t1
        )

    def _evaluate_poly(
        self,
        jerk: float,
        a0: float,
        v0: float,
        x0: float,
        t: float,
        direction: int,
    ) -> tuple[float, float, float, float]:
        jt = direction * jerk
        t2 = t * t
        t3 = t2 * t
        jerk_now = jt
        accel = a0 + jt * t
        vel = v0 + a0 * t + 0.5 * jt * t2
        pos = x0 + v0 * t + 0.5 * a0 * t2 + (1.0 / 6.0) * jt * t3
        return jerk_now, accel, vel, pos

    def update(self, dt: float, time_stretch: float = 1.0) -> None:
        """按 dt 推进参考轨迹状态。"""
        if dt <= 0.0:
            return
        if self.max_jerk <= _EPS or self.max_accel <= _EPS:
            self._jerk = 0.0
            self._accel = 0.0
            self._vel = self._vel_sp
            self._pos += self._vel * dt
            return

        self._local_time += dt * max(0.0, float(time_stretch))
        t_remain = self._local_time

        a_init, v_init, x_init = self._state_init
        if t_remain > 0.0:
            t1 = min(t_remain, self._T1)
            j, a, v, x = self._evaluate_poly(
                self.max_jerk, a_init, v_init, x_init, t1, self._direction
            )
            self._jerk, self._accel, self._vel, self._pos = j, a, v, x
            t_remain -= t1

        if t_remain > 0.0:
            t2 = min(t_remain, self._T2)
            j, a, v, x = self._evaluate_poly(
                0.0, self._accel, self._vel, self._pos, t2, 0
            )
            self._jerk, self._accel, self._vel, self._pos = j, a, v, x
            t_remain -= t2

        if t_remain > 0.0:
            t3 = min(t_remain, self._T3)
            j, a, v, x = self._evaluate_poly(
                self.max_jerk, self._accel, self._vel, self._pos, t3, -self._direction
            )
            self._jerk, self._accel, self._vel, self._pos = j, a, v, x
            t_remain -= t3

        if t_remain > 0.0:
            j, a, v, x = self._evaluate_poly(
                0.0, 0.0, self._vel, self._pos, t_remain, 0
            )
            self._jerk, self._accel, self._vel, self._pos = j, a, v, x

    # -- 多轴同步 -------------------------------------------------------- #
    @staticmethod
    def synchronize(axes: Sequence["VelocitySmoothing1D"]) -> None:
        """让多轴轨迹同时到达，避免斜线路径被单轴拖弯。

        PX4 的 `VelocitySmoothing::timeSynchronization`：以最长轴的总时间为
        目标，重新计算其他轴的 T1/T2/T3（不减慢最长轴）。
        """
        if not axes:
            return
        desired_time = max(axis.total_time for axis in axes)
        longest = max(range(len(axes)), key=lambda i: axes[i].total_time)
        if desired_time <= _EPS:
            return

        for i, axis in enumerate(axes):
            if i == longest or axis.total_time >= desired_time:
                continue
            axis._update_durations_given_total_time(desired_time)  # noqa: SLF001

    def _update_durations_given_total_time(self, total_time: float) -> None:
        if self.max_jerk <= _EPS or self.max_accel <= _EPS:
            self._T1 = self._T2 = self._T3 = 0.0
            return

        jerk_max_t1 = self._direction * self.max_jerk
        delta_v = self._vel_sp - self._vel
        if abs(jerk_max_t1) <= _EPS:
            self._T1 = self._T2 = self._T3 = 0.0
            return

        self._T1 = self._compute_t1_given_total_time(
            total_time, self._accel, delta_v, jerk_max_t1, self.max_accel
        )
        self._T3 = self._compute_t3(self._T1, self._accel, jerk_max_t1)
        self._T2 = max(total_time - self._T1 - self._T3, 0.0)

    def _compute_t1_given_total_time(
        self,
        total_time: float,
        a0: float,
        v3: float,
        j_max: float,
        a_max: float,
    ) -> float:
        a_coef = -j_max
        b_coef = j_max * total_time - a0
        delta = (
            total_time * total_time * j_max * j_max
            + 2.0 * total_time * a0 * j_max
            - a0 * a0
            - 4.0 * j_max * v3
        )
        if delta < 0.0:
            return 0.0

        sqrt_delta = float(np.sqrt(delta))
        denominator_inv = 1.0 / (2.0 * a_coef)
        t1_plus = max((-b_coef + sqrt_delta) * denominator_inv, 0.0)
        t1_minus = max((-b_coef - sqrt_delta) * denominator_inv, 0.0)

        t3_plus = a0 / j_max + t1_plus
        t3_minus = a0 / j_max + t1_minus

        t1 = 0.0
        if (t1_plus >= 0.0 and t3_plus >= 0.0) and (t1_plus + t3_plus <= total_time):
            t1 = t1_plus
        elif (t1_minus >= 0.0 and t3_minus >= 0.0) and (
            t1_minus + t3_minus <= total_time
        ):
            t1 = t1_minus

        t1 = self._saturate_t1_for_accel(a0, j_max, t1, a_max)
        return max(t1, 0.0)


class PositionReference3D:
    """3D 位置参考生成器：目标位置 + 前馈速度 -> 平滑参考。

    参数
    ----
    max_vel, max_accel, max_jerk
        可以传标量（三轴相同）或长度为 3 的序列（ENU）。
    safety_factor
        由剩余距离计算安全接近速度时的保守系数。0.7 表示比理论 S 曲线
        制动速度再保守 30%，留出 jerk 限制和估计滞后的余量。
    """

    def __init__(
        self,
        max_vel: float | Sequence[float] = 5.0,
        max_accel: float | Sequence[float] = 2.0,
        max_jerk: float | Sequence[float] = 4.0,
        safety_factor: float = 0.7,
    ) -> None:
        self.safety_factor = float(np.clip(safety_factor, 0.1, 1.0))
        self._limits = self._as_limits(max_vel, max_accel, max_jerk)
        self.axes: list[VelocitySmoothing1D] = [
            VelocitySmoothing1D(*self._limits[i]) for i in range(3)
        ]
        self.target_position = np.zeros(3, dtype=float)
        self.target_velocity = np.zeros(3, dtype=float)
        self._initialized = False
        self._last_state = ReferenceState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            jerk=np.zeros(3),
            unsmoothed_velocity=np.zeros(3),
        )

    @staticmethod
    def _as_limits(
        max_vel: float | Sequence[float],
        max_accel: float | Sequence[float],
        max_jerk: float | Sequence[float],
    ) -> list[tuple[float, float, float]]:
        def expand(value: float | Sequence[float]) -> list[float]:
            if np.isscalar(value):
                return [float(value)] * 3
            values = [float(v) for v in value]
            if len(values) == 1:
                return values * 3
            if len(values) != 3:
                raise ValueError("limits must be scalar or length-3")
            return values

        # 注意顺序与 VelocitySmoothing1D 构造器一致：(max_jerk, max_accel, max_vel)。
        return [
            (max(0.0, expand(max_jerk)[i]), max(0.0, expand(max_accel)[i]),
             max(0.0, expand(max_vel)[i]))
            for i in range(3)
        ]

    # -- 初始化/目标 ---------------------------------------------------- #
    def reset(
        self,
        position: Sequence[float],
        velocity: Optional[Sequence[float]] = None,
        acceleration: Optional[Sequence[float]] = None,
    ) -> None:
        pos = np.asarray(position, dtype=float).reshape(3)
        vel = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).reshape(3)
        acc = (
            np.zeros(3)
            if acceleration is None
            else np.asarray(acceleration, dtype=float).reshape(3)
        )
        for i, axis in enumerate(self.axes):
            axis.reset(float(acc[i]), float(vel[i]), float(pos[i]))
        self.target_position = pos.copy()
        self.target_velocity = vel.copy()
        self._initialized = True

    def set_target(
        self,
        position: Sequence[float],
        velocity: Optional[Sequence[float]] = None,
    ) -> None:
        self.target_position = np.asarray(position, dtype=float).reshape(3)
        self.target_velocity = (
            np.zeros(3)
            if velocity is None
            else np.asarray(velocity, dtype=float).reshape(3)
        )

    @property
    def initialized(self) -> bool:
        return self._initialized

    # -- 推进 ------------------------------------------------------------ #
    @staticmethod
    def _max_speed_for_distance(
        distance: float,
        max_vel: float,
        max_accel: float,
        max_jerk: float,
        safety_factor: float,
    ) -> float:
        """由剩余距离反推理论上不会冲过目标的最大速度。

        使用精确的 S 曲线制动距离：
          * 高速段（v >= A²/J）: d = v²/(2A) + v·A/(2J)
          * 低速三角段（v < A²/J）: d = (5/4)·v^(3/2)/√J
        再乘 ``safety_factor`` 留出估计/执行滞后余量。
        """
        if distance <= _EPS or max_vel <= _EPS:
            return 0.0
        if max_accel <= _EPS or max_jerk <= _EPS:
            # 无加速度/ jerk 限制：退化为 deadbeat 速度，由调用方按 dt 处理。
            return max_vel

        a = max_accel
        j = max_jerk
        v_transition = (a * a) / j
        d_transition = (v_transition * v_transition) / (2.0 * a) + (
            v_transition * a
        ) / (2.0 * j)

        if distance >= d_transition:
            # v² + (A²/J) v - 2 A d = 0
            b = (a * a) / j
            v = (-b + np.sqrt(b * b + 8.0 * a * distance)) / 2.0
        else:
            v = (4.0 * distance * np.sqrt(j) / 5.0) ** (2.0 / 3.0)

        return float(min(max_vel, safety_factor * max(0.0, v)))

    def update(self, dt: float, current_position: Optional[Sequence[float]] = None) -> ReferenceState:
        if dt <= 0.0:
            return self._last_state
        if not self._initialized:
            pos = (
                np.zeros(3)
                if current_position is None
                else np.asarray(current_position, dtype=float).reshape(3)
            )
            self.reset(pos)

        unsmoothed = np.zeros(3, dtype=float)
        # 3D 路径统一速度：用“剩余路径距离”生成一个路径速度，再沿误差方向分解。
        # 如果逐轴独立限速，斜线路径会被短轴拖弯；PX4 PositionSmoothing 也是
        # 先生成统一的 XY/Z 速度再交给各轴 smoothing。
        error_vec = self.target_position - self.position
        distance = float(np.linalg.norm(error_vec))
        max_vel_path = min(self._limits[i][2] for i in range(3))
        max_accel_path = min(self._limits[i][1] for i in range(3))
        max_jerk_path = min(self._limits[i][0] for i in range(3))
        if max_accel_path <= _EPS or max_jerk_path <= _EPS:
            # 退化配置：deadbeat 速度在一步内走到目标。
            v_path = 0.0 if dt <= _EPS else float(
                np.clip(distance / dt, 0.0, max_vel_path)
            )
        else:
            v_path = self._max_speed_for_distance(
                distance,
                max_vel_path,
                max_accel_path,
                max_jerk_path,
                self.safety_factor,
            )

        if distance > _EPS:
            direction = error_vec / distance
        else:
            direction = np.zeros(3, dtype=float)

        for i, axis in enumerate(self.axes):
            max_vel = self._limits[i][2]
            v_sp = float(
                np.clip(direction[i] * v_path + self.target_velocity[i], -max_vel, max_vel)
            )
            unsmoothed[i] = v_sp
            axis.set_velocity_setpoint(v_sp)

        VelocitySmoothing1D.synchronize(self.axes)

        # 可选的“时间拉伸”：当载具落后参考且参考正远离它时，放慢参考推进，
        # 避免参考跑掉后位置误差长期饱和。PX4 在每个轴上独立做这件事，
        # 这里用统一缩放因子保持 3D 路径形状。
        time_stretch = 1.0
        if current_position is not None:
            current = np.asarray(current_position, dtype=float).reshape(3)
            ref_pos = self.position
            ref_vel = self.velocity
            lag = ref_pos - current
            if np.linalg.norm(lag) > _EPS and float(np.dot(lag, ref_vel)) > 0.0:
                max_lag = max(self._limits[i][2] for i in range(3)) * 0.5
                if max_lag > _EPS:
                    time_stretch = float(
                        np.clip(1.0 - np.linalg.norm(lag) / (4.0 * max_lag), 0.25, 1.0)
                    )

        for axis in self.axes:
            axis.update(dt, time_stretch)

        self._last_state = ReferenceState(
            position=self.position,
            velocity=self.velocity,
            acceleration=self.acceleration,
            jerk=self.jerk,
            unsmoothed_velocity=unsmoothed,
        )
        return self._last_state

    # -- 输出 ------------------------------------------------------------ #
    @property
    def position(self) -> np.ndarray:
        return np.array([axis.position for axis in self.axes], dtype=float)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([axis.velocity for axis in self.axes], dtype=float)

    @property
    def acceleration(self) -> np.ndarray:
        return np.array([axis.acceleration for axis in self.axes], dtype=float)

    @property
    def jerk(self) -> np.ndarray:
        return np.array([axis.jerk for axis in self.axes], dtype=float)

    @property
    def state(self) -> ReferenceState:
        return self._last_state


__all__ = [
    "ReferenceState",
    "VelocitySmoothing1D",
    "PositionReference3D",
]
