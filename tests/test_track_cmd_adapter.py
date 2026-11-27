from guidance.intercept_guidance import GuidanceCommand, GuidancePhase
from guidance.track_cmd_adapter import guidance_to_track_cmd


def test_guidance_to_track_cmd_mapping():
    cmd = GuidanceCommand(
        phase=GuidancePhase.APPROACH,
        yaw_cmd_deg=45.0,
        pitch_cmd_deg=-44.0,
        throttle_cmd=0.6,
        target_range_m=20.0,
        valid=True,
    )
    track = guidance_to_track_cmd(cmd, max_angle_deg=45.0, timestamp=12.0)
    assert track.t == 12.0
    # Heading error +45 deg -> 90 dps yaw-rate command -> +225 RC units.
    assert track.yaw == 1725
    # Betaflight pitch high = nose down, so -44 deg body pitch -> high stick.
    assert track.pitch == 1989
    assert track.throttle == 1600
    assert track.roll == 1500


def test_guidance_to_track_cmd_converts_absolute_heading_to_rate():
    cmd = GuidanceCommand(
        phase=GuidancePhase.TRACK,
        yaw_cmd_deg=90.0,
        pitch_cmd_deg=0.0,
        throttle_cmd=0.5,
        valid=True,
    )
    # Already pointing east: no yaw-rate command, even though the absolute
    # heading is 90 deg (the old adapter commanded a large open-loop yaw).
    track = guidance_to_track_cmd(cmd, own_yaw_deg=90.0, timestamp=1.0)
    assert track.yaw == 1500
    # A 90 deg error is bounded by max_yaw_rate_dps, not by angle scale.
    track = guidance_to_track_cmd(
        cmd, own_yaw_deg=0.0, timestamp=1.0, max_yaw_rate_dps=120.0,
    )
    assert track.yaw == 2000


def test_guidance_to_track_cmd_clamps_and_rejects_nan():
    cmd = GuidanceCommand(
        phase=GuidancePhase.TERMINAL,
        yaw_cmd_deg=999.0,
        pitch_cmd_deg=-999.0,
        throttle_cmd=999.0,
        valid=True,
    )
    track = guidance_to_track_cmd(cmd, max_angle_deg=45.0, timestamp=0.0)
    # 999 deg wraps to -81 deg -> -162 dps -> below mid-stick.
    assert track.yaw == 1095
    assert track.pitch == 2000
    assert track.throttle == 2000

    cmd.yaw_cmd_deg = float("nan")
    cmd.pitch_cmd_deg = float("nan")
    cmd.throttle_cmd = float("nan")
    track = guidance_to_track_cmd(cmd, timestamp=0.0)
    assert 1000 <= track.yaw <= 2000
    assert 1000 <= track.pitch <= 2000
    assert 1000 <= track.throttle <= 2000
