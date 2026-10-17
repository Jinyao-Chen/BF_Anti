"""Camera yaw mounting offset: positive means the bore points right."""
import numpy as np
import pytest

from perception.camera_geometry import CameraExtrinsics, camera_ray_to_body


def test_zero_yaw_offset_keeps_old_behaviour():
    extr = CameraExtrinsics(tilt_deg=0.0, yaw_offset_deg=0.0)
    v = camera_ray_to_body(np.array([0.0, 0.0, 1.0]), extr)
    assert v == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


def test_positive_yaw_offset_points_right():
    extr = CameraExtrinsics(tilt_deg=0.0, yaw_offset_deg=90.0)
    v = camera_ray_to_body(np.array([0.0, 0.0, 1.0]), extr)
    # FLU: +y is left, so a right-pointing bore has negative y.
    assert v == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)
