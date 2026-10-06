from sim.compute_hover_throttle import compute_hover_throttle


def test_hover_throttle_matches_force_world_formula():
    omega, normalized, pwm = compute_hover_throttle(
        1.75, 8.54858e-06, 1100.0
    )
    assert abs(omega - 708.44) < 0.1
    assert abs(normalized - 0.6440) < 0.001
    assert abs(pwm - 1644.0) < 0.5
