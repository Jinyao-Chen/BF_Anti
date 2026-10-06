"""理论模型与 PID 整定工具链的单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from modeling.pid import PID, PIDGains
from modeling.quadrotor import (
    QuadrotorModel,
    QuadrotorParams,
    quat_to_rotation,
    simulate_open_loop,
)
from modeling.tune import design_cascade


def test_hover_equilibrium_has_zero_acceleration():
    model = QuadrotorModel(QuadrotorParams())
    u = np.full(4, model.params.hover_motor_cmd)
    dx = model.derivative(model.state, u)
    assert np.linalg.norm(dx[:6]) < 1e-9
    assert np.linalg.norm(dx[10:13]) < 1e-9


def test_open_loop_hover_stays_near_origin():
    _, states = simulate_open_loop(duration=1.0, dt=1.0 / 250.0)
    assert np.linalg.norm(states[-1, :3]) < 1e-3
    assert np.linalg.norm(states[-1, 3:6]) < 1e-3


def test_mixer_round_trip_for_positive_forces():
    p = QuadrotorParams()
    forces = np.array([3.0, 3.2, 2.8, 3.1])
    m = p.mixer_matrix()
    wrench = m @ forces
    recovered = np.linalg.solve(m, wrench)
    assert recovered == pytest.approx(forces, abs=1e-9)


def test_reduced_linear_model_thrust_axis():
    model = QuadrotorModel(QuadrotorParams())
    _, b = model.reduced_hover_linear()
    # 竖直速度由 dT 驱动：dv_u/d(dT) = 1/m
    assert b[5, 0] == pytest.approx(1.0 / model.params.mass)
    # 滚转力矩驱动 wx
    assert b[9, 1] == pytest.approx(1.0 / model.params.ixx)
    # 抬头力矩驱动 wy
    assert b[10, 2] == pytest.approx(1.0 / model.params.iyy)
    # 偏航力矩驱动 wz
    assert b[11, 3] == pytest.approx(1.0 / model.params.izz)


def test_quat_to_rotation_is_orthonormal():
    model = QuadrotorModel(QuadrotorParams())
    r = quat_to_rotation(model.state[model.Q])
    assert r @ r.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)


def test_pid_tracks_step_without_steady_error():
    pid = PID(PIDGains(kp=2.0, ki=5.0, kd=0.1), dt=0.01)
    y = 0.0
    for _ in range(500):
        u = pid.update(1.0, y)
        # 简单一阶对象 y' = -y + u
        y += (-y + u) * 0.01
    assert y == pytest.approx(1.0, abs=0.02)


def test_pid_output_respects_limits():
    pid = PID(PIDGains(kp=100.0, ki=0.0, kd=0.0), dt=0.01, output_limits=(-1.0, 1.0))
    assert pid.update(10.0, 0.0) == pytest.approx(1.0)
    assert pid.update(-10.0, 0.0) == pytest.approx(-1.0)


def test_design_cascade_returns_stable_metrics():
    gains = design_cascade(QuadrotorParams(), rate_bandwidth=40.0, attitude_bandwidth=8.0)
    for name in ("rate", "attitude", "velocity", "position", "altitude"):
        assert gains.metrics[name]["stable"] == 1.0
    assert gains.betaflight_hint["altitude_cascade_normalized"]["kp_per_s"] > 0.0
    assert gains.to_json()
