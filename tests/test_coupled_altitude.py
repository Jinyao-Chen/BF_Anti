"""测试水平机动-姿态-混控-高度耦合模型与诊断工具。"""

from __future__ import annotations

import math

import pytest

from modeling.coupled_altitude import (
    CASES,
    case_params,
    simulate,
    tilt_gravity_deficit,
)
from modeling.diagnose_altitude import _odd_window


def test_tilt_gravity_deficit_matches_small_angle():
    assert tilt_gravity_deficit(0.0) == pytest.approx(0.0, abs=1e-12)
    # 15° 的理论损失约 0.334 m/s²
    assert tilt_gravity_deficit(15.0) == pytest.approx(0.334, abs=0.01)
    # 小角度近似：g*(1-cosθ) ≈ g θ²/2
    theta = math.radians(10.0)
    assert tilt_gravity_deficit(10.0) == pytest.approx(
        9.80665 * theta * theta / 2.0, rel=5e-3
    )


def test_ideal_case_holds_altitude():
    result = simulate(case_params("ideal"), duration=8.0)
    # 当前固件等效 PID 是有限带宽控制器，理想姿态/估计下允许厘米级瞬态下沉。
    assert 9.7 < result.up.min() < 10.05
    assert result.up.max() < 10.15


def test_gravity_only_dip_is_small():
    result = simulate(case_params("gravity_only"), duration=8.0)
    # 仅关闭 1/cos 补偿时，掉高应明显小于 ring_mixer/full，且不应出现数米掉高。
    assert 8.8 < result.up.min() < 10.0


def test_ring_and_mixer_dominate_gravity():
    gravity = simulate(case_params("gravity_only"), duration=12.0)
    ring_mixer = simulate(case_params("ring_mixer"), duration=12.0)
    full = simulate(case_params("full"), duration=12.0)
    # 姿态振铃 + 混控饱和造成的掉高应显著大于纯重力补偿缺失。
    assert ring_mixer.up.min() < gravity.up.min() - 2.0
    assert full.up.min() <= ring_mixer.up.min() + 0.6


def test_current_firmware_model_beats_legacy_full():
    current = simulate(case_params("current"), duration=20.0)
    full = simulate(case_params("full"), duration=20.0)
    # 当前总距优先 + 精确倾角补偿模型应把控制链路导致的掉高压到 1 m 量级，
    # 而历史 full 场景会掉到 2 m 以下。
    assert current.up.min() > 9.0
    assert current.up.min() > full.up.min() + 6.0


def test_current_case_uses_exact_tilt_and_thrust_priority():
    p = case_params("current")
    assert p.mixer_mode == "thrust_priority"
    assert p.tilt_comp == "exact_cos"
    assert p.horizontal_map == "atan"
    assert p.altitude_structure == "firmware_pid"
    assert p.altitude_kp == pytest.approx(0.25)
    assert p.altitude_ki == pytest.approx(0.10)
    assert p.altitude_kd == pytest.approx(0.20)


def test_all_cases_have_metrics():
    for case in CASES:
        metrics = simulate(case_params(case), duration=5.0).metrics()
        assert set(metrics) >= {"min_up_m", "max_pitch_deg", "min_mixer_factor"}
        assert math.isfinite(metrics["min_up_m"])


def test_odd_window_is_safe_for_small_logs():
    assert _odd_window(1, 0.2) == 3
    assert _odd_window(5, 0.2) % 2 == 1
    assert _odd_window(10, 0.2) % 2 == 1
    assert _odd_window(10, 0.2) <= 9
