"""四旋翼非线性 6-DOF 模型与悬停线性化。

坐标系
------
Earth/ENU
    x = East, y = North, z = Up。
Body/FLU
    x = 机头前方, y = 机身左方, z = 机身上方（与 :mod:`perception.camera_geometry`
    和 BF_Anti 制导层保持一致）。

模型状态
--------
``[p_e, p_n, p_u, v_e, v_n, v_u, qw, qx, qy, qz, wx, wy, wz, m0, m1, m2, m3]``

* ``p`` / ``v``：ENU 位置与速度，m / m/s
* ``q``：机体 FLU -> ENU 的四元数，顺序 ``(w, x, y, z)``
* ``w``：机体角速度，rad/s
* ``m_i``：四个电机的转速，rad/s；输入为归一化电机指令 ``u_i in [0, 1]``

参数默认值来自 Gazebo 的 Iris + MulticopterMotorModel 以及参考
Simulink 工程的四旋翼参数；不同平台可在 :class:`QuadrotorParams` 中覆盖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# 四元数工具
# --------------------------------------------------------------------------- #
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton 乘积 ``a ⊗ b``，均为 ``(w, x, y, z)``。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=float,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """机体 FLU -> ENU 的 3x3 旋转矩阵。"""
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_to_quat(r: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> 四元数 ``(w, x, y, z)``。"""
    r = np.asarray(r, dtype=float)
    tr = float(np.trace(r))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s]
        )
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        q = np.array(
            [(r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s]
        )
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        q = np.array(
            [(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        q = np.array(
            [(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s]
        )
    return quat_normalize(q)


# --------------------------------------------------------------------------- #
# 模型参数
# --------------------------------------------------------------------------- #
@dataclass
class QuadrotorParams:
    """Iris 级四旋翼参数。

    ``k_thrust`` / ``k_torque`` 的单位分别是 ``N/(rad/s)^2`` 和
    ``N*m per N``（即阻力矩与推力之比，等价于 ``cM/cT``）。
    """

    mass: float = 1.4
    gravity: float = 9.80665
    ixx: float = 0.0211
    iyy: float = 0.0219
    izz: float = 0.0366
    arm_length: float = 0.225
    k_thrust: float = 8.54858e-06
    k_torque: float = 0.0161
    motor_tau_up: float = 0.0125
    motor_tau_down: float = 0.025
    max_rpm: float = 1100.0
    min_rpm: float = 0.0
    drag_linear: float = 0.12
    drag_quadratic: float = 0.02
    # X 型布局：前右、前左、后左、后右
    motor_positions: Tuple[Tuple[float, float], ...] = (
        (0.1591, -0.1591),
        (0.1591, 0.1591),
        (-0.1591, 0.1591),
        (-0.1591, -0.1591),
    )
    motor_spin: Tuple[float, ...] = (1.0, -1.0, 1.0, -1.0)

    @property
    def inertia(self) -> np.ndarray:
        return np.diag([self.ixx, self.iyy, self.izz])

    @property
    def hover_motor_omega(self) -> float:
        force = self.mass * self.gravity / 4.0
        return float(np.sqrt(force / self.k_thrust))

    @property
    def hover_motor_cmd(self) -> float:
        return float(np.clip(self.hover_motor_omega / self.max_rpm, 0.0, 1.0))

    @property
    def max_motor_force(self) -> float:
        return float(self.k_thrust * self.max_rpm * self.max_rpm)

    def mixer_matrix(self) -> np.ndarray:
        """``[T, tau_x, tau_y, tau_z]^T = M @ [f0..f3]``。

        ``tau_x`` 为绕机体 x 轴（右滚为正），``tau_y`` 为绕 y 轴（抬头为正），
        ``tau_z`` 为绕 z 轴（偏航 CCW 为正）。各电机反扭矩由
        ``k_torque * f`` 给出，``motor_spin`` 给出方向。
        """
        m = np.zeros((4, 4), dtype=float)
        for i, (x, y) in enumerate(self.motor_positions):
            m[0, i] = 1.0
            m[1, i] = y
            m[2, i] = -x
            m[3, i] = self.k_torque * self.motor_spin[i]
        return m

    def mix(self, total_thrust: float, torque: np.ndarray) -> np.ndarray:
        """期望总推力/力矩 -> 四个电机推力（已限幅到物理范围）。"""
        desired = np.array([total_thrust, torque[0], torque[1], torque[2]], dtype=float)
        forces = np.linalg.solve(self.mixer_matrix(), desired)
        return np.clip(forces, 0.0, self.max_motor_force)

    def motor_force_to_cmd(self, forces: np.ndarray) -> np.ndarray:
        omega = np.sqrt(np.clip(forces, 0.0, self.max_motor_force) / self.k_thrust)
        return np.clip(omega / self.max_rpm, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 非线性模型
# --------------------------------------------------------------------------- #
class QuadrotorModel:
    """固定步长 RK4 非线性四旋翼模型。"""

    P = slice(0, 3)
    V = slice(3, 6)
    Q = slice(6, 10)
    W = slice(10, 13)
    M = slice(13, 17)
    STATE_DIM = 17

    def __init__(
        self,
        params: Optional[QuadrotorParams] = None,
        state: Optional[Sequence[float]] = None,
    ) -> None:
        self.params = params if params is not None else QuadrotorParams()
        if state is None:
            self.state = self.hover_state()
        else:
            self.state = np.asarray(state, dtype=float).copy()
            self.state[self.Q] = quat_normalize(self.state[self.Q])

    # ------------------------------------------------------------------ #
    # 初始状态与指令
    # ------------------------------------------------------------------ #
    def hover_state(self) -> np.ndarray:
        x = np.zeros(self.STATE_DIM, dtype=float)
        x[self.Q] = np.array([1.0, 0.0, 0.0, 0.0])
        x[self.M] = self.params.hover_motor_omega
        return x

    @property
    def position(self) -> np.ndarray:
        return self.state[self.P].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[self.V].copy()

    @property
    def quaternion(self) -> np.ndarray:
        return self.state[self.Q].copy()

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_rotation(self.state[self.Q])

    # ------------------------------------------------------------------ #
    # 动力学
    # ------------------------------------------------------------------ #
    def _motor_forces(self, motor_omega: np.ndarray) -> np.ndarray:
        return self.params.k_thrust * motor_omega * motor_omega

    def _body_torque(self, forces: np.ndarray) -> np.ndarray:
        torque = np.zeros(3, dtype=float)
        for i, (x, y) in enumerate(self.params.motor_positions):
            f = float(forces[i])
            torque[0] += y * f
            torque[1] += -x * f
            torque[2] += self.params.k_torque * self.params.motor_spin[i] * f
        return torque

    def derivative(self, state: np.ndarray, motor_cmd: np.ndarray) -> np.ndarray:
        v = state[self.V]
        q = quat_normalize(state[self.Q])
        omega = state[self.W]
        motor_omega = state[self.M]

        cmd_omega = np.clip(motor_cmd, 0.0, 1.0) * self.params.max_rpm
        tau_m = np.where(cmd_omega > motor_omega, self.params.motor_tau_up, self.params.motor_tau_down)
        motor_omega_dot = (cmd_omega - motor_omega) / tau_m

        forces = self._motor_forces(motor_omega)
        total_thrust = float(np.sum(forces))
        torque = self._body_torque(forces)

        r = quat_to_rotation(q)
        thrust_world = r @ np.array([0.0, 0.0, total_thrust])
        speed = float(np.linalg.norm(v))
        drag = self.params.drag_linear * v + self.params.drag_quadratic * speed * v
        accel = thrust_world / self.params.mass + np.array([0.0, 0.0, -self.params.gravity]) - drag / self.params.mass

        inertia = self.params.inertia
        omega_dot = np.linalg.solve(inertia, torque - np.cross(omega, inertia @ omega))
        q_dot = 0.5 * quat_multiply(q, np.array([0.0, *omega]))

        dx = np.zeros_like(state, dtype=float)
        dx[self.P] = v
        dx[self.V] = accel
        dx[self.Q] = q_dot
        dx[self.W] = omega_dot
        dx[self.M] = motor_omega_dot
        return dx

    def step(self, motor_cmd: Sequence[float], dt: float) -> np.ndarray:
        """RK4 推进一步，``motor_cmd`` 为 4 个归一化电机指令。"""
        u = np.clip(np.asarray(motor_cmd, dtype=float), 0.0, 1.0)
        x = self.state
        k1 = self.derivative(x, u)
        k2 = self.derivative(x + 0.5 * dt * k1, u)
        k3 = self.derivative(x + 0.5 * dt * k2, u)
        k4 = self.derivative(x + dt * k3, u)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x[self.Q] = quat_normalize(x[self.Q])
        self.state = x
        return x

    # ------------------------------------------------------------------ #
    # 线性化
    # ------------------------------------------------------------------ #
    def linearize_hover(self, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """完整 17 状态模型在悬停点的数值线性化。

        返回 ``(A, B)``，状态导数 ``dx = A x + B u``，输入为 4 个归一化电机
        指令的增量。
        """
        x0 = self.hover_state()
        u0 = np.full(4, self.params.hover_motor_cmd)

        a = np.zeros((self.STATE_DIM, self.STATE_DIM), dtype=float)
        b = np.zeros((self.STATE_DIM, 4), dtype=float)
        for j in range(self.STATE_DIM):
            dx = np.zeros(self.STATE_DIM)
            dx[j] = eps
            a[:, j] = (self.derivative(x0 + dx, u0) - self.derivative(x0 - dx, u0)) / (2.0 * eps)
        for j in range(4):
            du = np.zeros(4)
            du[j] = eps
            b[:, j] = (self.derivative(x0, u0 + du) - self.derivative(x0, u0 - du)) / (2.0 * eps)
        return a, b

    def reduced_hover_linear(self) -> Tuple[np.ndarray, np.ndarray]:
        """面向级联 PID 设计的 12 状态小角度悬停模型。

        状态：``[p_e,p_n,p_u, v_e,v_n,v_u, roll,pitch,yaw, wx,wy,wz]``
        输入：``[dT, tau_x, tau_y, tau_z]``，其中 ``dT`` 为总推力增量（N）。

        方程来自小角度牛顿-欧拉模型：

            v_e_dot = g*roll,   v_n_dot = -g*pitch,   v_u_dot = dT/m
            phi_dot = wx,       theta_dot = wy,       psi_dot = wz
            wx_dot = tau_x/Ixx, wy_dot = tau_y/Iyy,   wz_dot = tau_z/Izz
        """
        a = np.zeros((12, 12), dtype=float)
        b = np.zeros((12, 4), dtype=float)
        g = self.params.gravity
        m = self.params.mass

        # p_dot = v
        a[0:3, 3:6] = np.eye(3)
        # v_dot
        a[3, 6] = g          # East velocity from roll
        a[4, 7] = -g         # North velocity from pitch
        b[5, 0] = 1.0 / m    # Up velocity from thrust increment
        # attitude_dot = omega
        a[6:9, 9:12] = np.eye(3)
        # omega_dot = torque / inertia
        b[9, 1] = 1.0 / self.params.ixx
        b[10, 2] = 1.0 / self.params.iyy
        b[11, 3] = 1.0 / self.params.izz
        return a, b


def simulate_open_loop(
    params: Optional[QuadrotorParams] = None,
    motor_cmd: Optional[Sequence[float]] = None,
    duration: float = 2.0,
    dt: float = 1.0 / 250.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """简单开环仿真，主要用于检查模型平衡点是否正确。"""
    model = QuadrotorModel(params)
    u = np.full(4, model.params.hover_motor_cmd) if motor_cmd is None else np.asarray(motor_cmd, dtype=float)
    t = np.arange(0.0, duration + 0.5 * dt, dt)
    states = np.zeros((len(t), model.STATE_DIM), dtype=float)
    states[0] = model.state
    for i in range(1, len(t)):
        model.step(u, dt)
        states[i] = model.state
    return t, states
