"""LiftDrag k_v 辨识与速度²前馈/推力耦合的回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

from modeling.coupled_altitude import case_params, simulate
from modeling.identify_lift_loss import (
    estimate_lift_loss,
    estimate_lift_loss_robust,
)


def _synthetic_log(k_true: float, speed_mps: float = 8.0, duration: float = 10.0) -> dict:
    dt = 0.005
    t = np.arange(0.0, duration, dt)
    # 匀速平飞：truth_e = v*t，真值高度恒定。
    truth_e = speed_mps * t
    truth_n = np.zeros_like(t)
    truth_u = np.full_like(t, 10.0)
    # 让高度环指令正好配平 k_true * v^2 的等效向下加速度。
    hover_norm = (1530.0 - 1000.0) / 1000.0
    g = 9.80665
    a_pred = k_true * speed_mps * speed_mps
    throttle_norm = hover_norm * (1.0 + a_pred / g)
    dbg0 = 1000.0 + throttle_norm * 1000.0
    return {
        "t": t,
        "truth_e": truth_e,
        "truth_n": truth_n,
        "truth_u": truth_u,
        "roll": np.zeros_like(t),
        "pitch": np.zeros_like(t),
        "dbg0": np.full_like(t, dbg0),
    }


def test_identify_lift_loss_recovers_synthetic_k():
    k_true = 0.005
    result = estimate_lift_loss(_synthetic_log(k_true), min_speed_mps=4.0)
    assert result.k_v == pytest.approx(k_true, rel=1e-3, abs=1e-5)
    assert result.n > 100


def test_identify_lift_loss_scales_with_speed_squared():
    low = estimate_lift_loss(_synthetic_log(0.004, speed_mps=5.0), min_speed_mps=4.0)
    high = estimate_lift_loss(_synthetic_log(0.004, speed_mps=10.0), min_speed_mps=4.0)
    assert low.k_v == pytest.approx(high.k_v, rel=1e-3)


def test_robust_multi_log_identification_recovers_k():
    k_true = 0.005
    logs = [
        _synthetic_log(k_true, speed_mps=5.0),
        _synthetic_log(k_true, speed_mps=8.0),
        _synthetic_log(k_true, speed_mps=11.0),
    ]
    result = estimate_lift_loss_robust(logs, min_speed_mps=4.0)
    assert result.k_v_huber == pytest.approx(k_true, rel=0.02, abs=1e-4)
    assert result.k_v_intercept_huber == pytest.approx(k_true, rel=0.05, abs=1e-4)
    assert result.n_logs == 3
    assert result.n > 100


def test_feedforward_recovers_altitude_with_plant_loss():
    plant = simulate(case_params("current_liftdrag"), duration=20.0)
    ff = simulate(case_params("current_ff"), duration=20.0)
    # 前馈应显著减小 LiftDrag plant 损失造成的掉高。
    assert ff.up.min() > plant.up.min() + 0.5
    # 前馈打开后必须真的有总距前馈和 plant 损失进入模型。
    assert ff.metrics()["max_lift_ff_accel_mps2"] > 0.05
    assert ff.metrics()["max_plant_lift_loss_accel_mps2"] > 0.05


def test_feedforward_case_couples_horizontal_tilt():
    p = case_params("current_ff")
    assert p.altitude_lift_ff_gain > 0.0
    assert p.lift_ff_couple_horizontal is True
