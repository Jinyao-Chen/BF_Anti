"""固件参数映射、固件等效高度 PID 与诊断量纲的回归测试。"""

from __future__ import annotations

import numpy as np
import pytest

from modeling.altitude_loop import (
    AltitudeGains,
    AltitudePlantParams,
    evaluate_altitude,
    simulate_altitude,
    sitl_params,
)
from modeling.diagnose_altitude import identify
from modeling.firmware_params import (
    SITL_FINAL_GAINS,
    altitude_kd_from_cli,
    altitude_ki_from_cli,
    altitude_kp_from_cli,
    cli_from_altitude_kd,
    cli_from_altitude_ki,
    cli_from_altitude_kp,
)


def test_firmware_altitude_scales_match_source_constants():
    assert altitude_kp_from_cli(50.0) == pytest.approx(0.25)
    assert altitude_ki_from_cli(50.0) == pytest.approx(0.10)
    assert altitude_kd_from_cli(20.0) == pytest.approx(0.20)
    assert cli_from_altitude_kp(0.25) == pytest.approx(50.0)
    assert cli_from_altitude_ki(0.10) == pytest.approx(50.0)
    assert cli_from_altitude_kd(0.20) == pytest.approx(20.0)


def test_sitl_final_gains_round_trip():
    assert SITL_FINAL_GAINS.kp_pwm_per_cm == pytest.approx(0.25)
    assert SITL_FINAL_GAINS.ki_pwm_per_cm_s == pytest.approx(0.10)
    assert SITL_FINAL_GAINS.kd_pwm_per_cm_s == pytest.approx(0.20)
    assert SITL_FINAL_GAINS.kf_pwm_per_cm_s == pytest.approx(0.0)
    assert SITL_FINAL_GAINS.to_cli() == pytest.approx(
        {
            "ap_altitude_p": 50.0,
            "ap_altitude_i": 50.0,
            "ap_altitude_d": 20.0,
            "ap_altitude_f": 0.0,
        }
    )


def test_altitude_gains_to_cli_uses_real_scales():
    gains = AltitudeGains(kp=0.25, ki=0.10, kd=0.20, kf=0.0)
    assert gains.to_betaflight() == pytest.approx(
        {
            "ap_altitude_p": 50.0,
            "ap_altitude_i": 50.0,
            "ap_altitude_d": 20.0,
            "ap_altitude_f": 0.0,
        }
    )


def test_sitl_plant_pwm_slope_is_physical():
    p = sitl_params()
    # 1 PWM 在 1.5 kg / 0.53 悬停油门附近约等于 0.0123 m/s²
    assert p.pwm_to_accel_mps2 == pytest.approx(9.80665 / (0.53 * 1000.0), rel=1e-12)
    assert p.max_accel_pwm == pytest.approx(3.0 / p.pwm_to_accel_mps2, rel=1e-12)


def test_firmware_pid_tracks_two_metre_step():
    """固件等效 PID 结构在通用 plant 上能完成 2 m 阶跃。"""
    p = AltitudePlantParams(mass=1.5, thrust_tau_s=0.08, measurement_tau_s=0.05)
    gains = AltitudeGains(kp=0.5, ki=0.0, kd=1.0)
    metrics = evaluate_altitude(gains, p, z_des=2.0, duration=10.0)
    assert metrics["steady_state_error_m"] < 0.05
    assert metrics["overshoot_m"] < 0.1
    _, z, _ = simulate_altitude(gains, p, z_des=2.0, duration=10.0)
    assert np.isfinite(z).all()


def test_sitl_final_gains_are_finite_in_model():
    """回归：最终 CLI 增益能按真实缩放进入模型，不产生 NaN。"""
    p = sitl_params()
    gains = AltitudeGains(
        kp=SITL_FINAL_GAINS.kp_pwm_per_cm,
        ki=SITL_FINAL_GAINS.ki_pwm_per_cm_s,
        kd=SITL_FINAL_GAINS.kd_pwm_per_cm_s,
        kf=SITL_FINAL_GAINS.kf_pwm_per_cm_s,
    )
    _, z, thrust = simulate_altitude(gains, p, z_des=2.0, duration=10.0)
    assert np.isfinite(z).all()
    assert np.isfinite(thrust).all()


def test_identify_computes_acceleration_not_velocity():
    """回归：高度一阶导数是垂速，二阶导数才是垂向加速度。"""
    dt = 0.01
    t = np.arange(0.0, 3.0, dt)
    # z = 10 - 0.5 * 1.0 * t^2 -> v = -t, a = -1 m/s²
    truth_u = 10.0 - 0.5 * t * t
    log = {
        "t": t,
        "up": np.full_like(t, 10.0),
        "vu": -t,
        "pitch": np.zeros_like(t),
        "truth_u": truth_u,
        "m0": np.full_like(t, 1500.0),
        "m1": np.full_like(t, 1500.0),
        "m2": np.full_like(t, 1500.0),
        "m3": np.full_like(t, 1500.0),
    }
    identified = identify(log)
    assert identified["min_truth_vel_mps"] < -2.0
    assert identified["min_truth_accel_mps2"] == pytest.approx(-1.0, abs=0.05)
