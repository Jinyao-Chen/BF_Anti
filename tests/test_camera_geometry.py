
from perception.camera_geometry import (
    CameraExtrinsics,
    CameraIntrinsics,
    desired_body_pitch_for_elevation,
    earth_los_to_az_el,
    pixel_to_earth_los,
    range_from_bbox_width,
    target_bearing_from_bbox_center,
)


def make_intrinsics():
    return CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=320.0, cy=240.0)


def test_center_pixel_boresight_is_up_44deg():
    intr = make_intrinsics()
    extr = CameraExtrinsics(tilt_deg=44.0)
    los = pixel_to_earth_los((320.0, 240.0), intr, extr, yaw_deg=0.0)
    az, el = earth_los_to_az_el(los)
    assert abs(az) < 1e-6
    assert abs(el - 44.0) < 1e-6


def test_pixel_right_is_positive_azimuth():
    intr = make_intrinsics()
    extr = CameraExtrinsics(tilt_deg=44.0)
    az, _ = target_bearing_from_bbox_center((639.0, 239.0, 641.0, 241.0), intr, extr)
    assert az > 0.0


def test_pixel_below_center_is_lower_elevation():
    intr = make_intrinsics()
    extr = CameraExtrinsics(tilt_deg=44.0)
    _, el_center = earth_los_to_az_el(
        pixel_to_earth_los((320.0, 240.0), intr, extr)
    )
    _, el_down = earth_los_to_az_el(
        pixel_to_earth_los((320.0, 400.0), intr, extr)
    )
    assert el_down < el_center


def test_desired_body_pitch():
    assert abs(desired_body_pitch_for_elevation(0.0, 44.0) + 44.0) < 1e-9
    assert abs(desired_body_pitch_for_elevation(10.0, 44.0) + 34.0) < 1e-9


def test_range_from_bbox_width():
    assert abs(range_from_bbox_width((0.0, 0.0, 100.0, 100.0), 500.0, 1.0) - 5.0) < 1e-9
