"""Onboard config: example loads, profile overlays, invalid values rejected."""
import json
import os

import pytest

from onboard.config import AppConfig, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "config", "onboard.example.json")


def test_example_config_loads_and_profile_overlays():
    cfg = load_config(EXAMPLE)
    assert cfg.profile == "real_pn8_angle"
    assert cfg.fc.aux_engage == 7
    # Values that exist only in config/profiles/real_pn8_angle.json
    assert cfg.guidance.params["kp_roll"] == 45.0
    assert cfg.guidance.params["pn_gain"] == 8.0
    assert cfg.guidance.params["max_pitch_down_deg"] == 50.0
    assert cfg.safety.min_altitude_m == 2.0
    assert cfg.control.att_mode == "angle"


def test_build_guidance_uses_measured_camera_geometry():
    cfg = load_config(EXAMPLE)
    g = cfg.build_guidance()
    assert g.kp_roll == 45.0
    assert g.camera_tilt_deg == cfg.camera.tilt_deg
    assert g.camera_yaw_offset_deg == cfg.camera.yaw_offset_deg
    assert (g.intrinsics.width, g.intrinsics.height) == (cfg.camera.width, cfg.camera.height)


def test_rotation_swaps_guidance_image_size(tmp_path):
    raw = json.load(open(EXAMPLE, encoding="utf-8"))
    raw["camera"]["rotate_deg"] = 90
    p = tmp_path / "rotated.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    cfg = load_config(str(p))
    g = cfg.build_guidance()
    assert (g.intrinsics.width, g.intrinsics.height) == (
        cfg.camera.height,
        cfg.camera.width,
    )


def test_duplicate_aux_channels_rejected(tmp_path):
    raw = json.load(open(EXAMPLE, encoding="utf-8"))
    # Legacy mode: separate OVERRIDE and MODE switches must be distinct.
    raw["fc"]["override_from_mode"] = False
    raw["fc"]["aux_engage"] = raw["fc"]["aux_override"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct"):
        load_config(str(p))


def test_mode_and_arm_channels_must_still_differ(tmp_path):
    raw = json.load(open(EXAMPLE, encoding="utf-8"))
    raw["fc"]["aux_arm"] = raw["fc"]["aux_engage"]
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="distinct"):
        load_config(str(p))


def test_bad_throttle_range_rejected(tmp_path):
    raw = json.load(open(EXAMPLE, encoding="utf-8"))
    raw["control"]["hover_throttle"] = 999
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="hover_throttle"):
        load_config(str(p))


def test_unknown_guidance_parameter_fails_at_build_time():
    base = load_config(EXAMPLE)
    from dataclasses import replace
    from onboard.config import GuidanceConfig

    bad = replace(base, guidance=GuidanceConfig(params={"not_a_real_gain": 1.0}))
    with pytest.raises(TypeError):
        bad.build_guidance()


def test_config_roundtrip_is_json_serialisable():
    cfg = load_config(EXAMPLE)
    again = AppConfig.from_dict(cfg.to_dict())
    assert again.to_dict() == cfg.to_dict()


def test_example_config_enables_the_blackbox():
    cfg = load_config(EXAMPLE)
    assert cfg.telemetry.enabled
    assert cfg.telemetry.hz == 10.0
    assert cfg.telemetry.max_runs == 20
    assert cfg.log_dir == "/home/pi5/bf_anti_logs"
    # log_dir must survive the profile overlay (it is not a profile key).
    assert cfg.to_dict()["telemetry"]["hz"] == 10.0


def test_bad_telemetry_hz_rejected(tmp_path):
    raw = json.load(open(EXAMPLE, encoding="utf-8"))
    raw["telemetry"]["hz"] = 500
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry.hz"):
        load_config(str(p))


def test_bad_telemetry_retention_rejected(tmp_path):
    raw = json.load(open(EXAMPLE, encoding="utf-8"))
    raw["telemetry"]["max_runs"] = 0
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="max_runs"):
        load_config(str(p))
