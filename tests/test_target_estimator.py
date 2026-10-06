import numpy as np

from perception.camera_geometry import CameraExtrinsics, CameraIntrinsics, pixel_to_earth_los
from perception.target_estimator import AlphaBetaFilter, TargetEstimator


def make_intrinsics():
    return CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)


def test_alpha_beta_converges_to_constant_velocity():
    f = AlphaBetaFilter(alpha=0.7, beta=0.3)
    pos = np.array([0.0, 0.0, 0.0])
    vel = np.array([1.0, 0.0, 0.0])
    f.reset(pos)
    for _ in range(20):
        pos = pos + vel * 0.1
        f.update(pos, 0.1)
    assert np.linalg.norm(f.velocity - vel) < 0.05


def test_estimator_center_pixel_range_and_bearing():
    intr = make_intrinsics()
    extr = CameraExtrinsics(tilt_deg=44.0)
    est = TargetEstimator(intr, extr, target_width_m=1.0)

    # Center pixel -> LOS 44 deg elevation.  bbox width 25 px -> range 20 m.
    bbox = (320.0 - 12.5, 240.0 - 12.5, 320.0 + 12.5, 240.0 + 12.5)
    pos, rng, az, el = est.measurement_to_position(bbox, np.zeros(3))
    assert abs(rng - 20.0) < 1e-6
    assert abs(el - 44.0) < 1e-6
    expected = pixel_to_earth_los((320.0, 240.0), intr, extr) * 20.0
    assert np.linalg.norm(pos - expected) < 1e-6

    state = est.update(bbox, np.zeros(3), timestamp=1.0)
    assert state.valid
    assert abs(state.range_m - 20.0) < 1e-6
    assert abs(state.el_deg - 44.0) < 1e-6
