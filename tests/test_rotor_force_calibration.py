import numpy as np

from sim.calibrate_rotor_force import fit_accel


def test_fit_accel_recovers_synthetic_quadratic():
    t = np.linspace(0.0, 1.0, 100)
    a_true = -3.2
    v0 = 5.0
    z0 = 100.0
    z = z0 + v0 * t + 0.5 * a_true * t * t
    assert abs(fit_accel(t, z) - a_true) < 1e-6
