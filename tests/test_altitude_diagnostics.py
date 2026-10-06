"""Offline altitude diagnostic input validation and attitude projection."""

import numpy as np
import pytest

from modeling.diagnose_altitude import identify


def make_log(n=100):
    t = np.arange(n, dtype=float) * 0.05
    return {
        "t": t,
        "up": np.full(n, 10.0),
        "truth_u": np.full(n, 10.0),
        "pitch": np.zeros(n),
        "roll": np.zeros(n),
        **{f"m{i}": np.full(n, 1500.0) for i in range(4)},
    }


def test_combined_roll_pitch_projection():
    log = make_log()
    log["roll"][:] = 30.0
    log["pitch"][:] = 30.0
    result = identify(log)
    assert result["max_tilt_deg"] == pytest.approx(np.degrees(np.arccos(0.75)))
    assert result["gravity_deficit_max_mps2"] == pytest.approx(9.80665 * 0.25)
    assert result["ring_hz"] == 0.0


@pytest.mark.parametrize("n", [3, 4, 5])
def test_short_finite_logs_are_supported(n):
    result = identify(make_log(n))
    assert result["samples_used"] == n
    assert result["sample_rate_hz"] == pytest.approx(20.0)
    assert result["min_truth_accel_mps2"] == pytest.approx(0.0, abs=1e-9)


def test_nonfinite_samples_are_reported_and_removed():
    log = make_log()
    log["truth_u"][4] = np.nan
    result = identify(log)
    assert result["samples_used"] == 99
    assert result["samples_dropped_nonfinite"] == 1
    assert result["sample_period_max_s"] == pytest.approx(0.1)


@pytest.mark.parametrize("bad_time", [0.0, -1.0])
def test_invalid_timestamps_rejected(bad_time):
    log = make_log()
    log["t"][1] = bad_time
    with pytest.raises(ValueError, match="strictly increasing"):
        identify(log)


def test_bad_shapes_and_insufficient_data_rejected():
    log = make_log()
    log["pitch"] = np.zeros(2)
    with pytest.raises(ValueError, match="equally sized"):
        identify(log)
    with pytest.raises(ValueError, match="at least 3"):
        identify(make_log(2))


def test_drop_and_estimator_error_are_separate_metrics():
    log = make_log()
    log["truth_u"] = 10.0 - 0.1 * log["t"]
    result = identify(log)
    assert result["truth_altitude_drop_m"] == pytest.approx(0.495)
    assert result["altitude_estimation_rmse_m"] > 0.25
    assert result["min_truth_accel_mps2"] == pytest.approx(0.0, abs=1e-8)


def test_legacy_pitch_only_logs_remain_compatible():
    log = make_log()
    del log["roll"]
    result = identify(log)
    assert not result["roll_available"]
    assert result["max_tilt_deg"] == 0.0
