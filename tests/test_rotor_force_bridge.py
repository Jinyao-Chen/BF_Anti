"""RotorForce bridge 的力/力矩分配回归测试。"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("gz.transport13")

from sim.rotor_force_bridge import (  # noqa: E402
    GRAVITY,
    MAX_OMEGA,
    MOTOR_CONSTANT,
    motor_wrench,
)


def test_equal_hover_commands_balance_weight():
    mass = 1.6
    hover_cmd = math.sqrt(
        mass * GRAVITY / (4.0 * MOTOR_CONSTANT * MAX_OMEGA * MAX_OMEGA)
    )
    force, torque = motor_wrench([hover_cmd] * 4)
    assert force[0] == pytest.approx(0.0, abs=1e-9)
    assert force[1] == pytest.approx(0.0, abs=1e-9)
    assert force[2] == pytest.approx(mass * GRAVITY, rel=1e-9)
    assert torque == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)


def test_differential_pitch_produces_roll_torque():
    # Channels 0/1 are rear/front right vs left; make one side stronger.
    force, torque = motor_wrench([0.7, 0.7, 0.5, 0.5])
    assert force[2] > 0.0
    assert abs(torque[0]) > 0.1
    # Equal left/right sums should leave yaw torque small.
    assert abs(torque[2]) < 1e-6
