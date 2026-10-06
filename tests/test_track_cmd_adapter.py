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
    assert track.yaw == 2000
    assert track.pitch == 1011
    assert track.throttle == 1600
    assert track.roll == 1500


def test_guidance_to_track_cmd_clamps():
    cmd = GuidanceCommand(
        phase=GuidancePhase.TERMINAL,
        yaw_cmd_deg=999.0,
        pitch_cmd_deg=-999.0,
        throttle_cmd=999.0,
        valid=True,
    )
    track = guidance_to_track_cmd(cmd, max_angle_deg=45.0, timestamp=0.0)
    assert track.yaw == 2000
    assert track.pitch == 1000
    assert track.throttle == 2000
