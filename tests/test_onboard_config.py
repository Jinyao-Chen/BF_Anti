"""Onboard config: example loads, profile overlays, invalid values rejected."""
import json
import os

import pytest

from onboard.config import AppConfig, load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "config", "onboard.example.json")


def test_example_config_loads_and_profile_overlays():
    cfg = load_config(EXAMPLE)
    assert cfg.profile == "real_conservative"
    assert cfg.fc.aux_engage == 7
    # Values that exist only in config/profiles/real_conservative.json
    assert cfg.guidance.params["kp_roll"] == 15.0
    assert cfg.guidance.params["pn_gain"] == 4.0
    assert cfg.safety.min_altitude_m == 3.0


def test_build_guidance_uses_measured_camera_geometry():
    cfg = load_config(EXAMPLE)
    g = cfg.build_guidance()
    assert g.kp_roll == 15.0
    assert g.camera_tilt_deg == cfg.camera.tilt_deg
    assert g.camera_yaw_offset_deg == cfg.camera.yaw_offset_deg
    assert (g.intrinsics.width, g.intrinsics.height) == (cfg.camera.width, cfg.camera.height)


def test_duplicate_aux_channels_rejected(tmp_path):
    raw = json.load(open(EXAMPLE, encoding="utf-8"))
    raw["fc"]["aux_engage"] = raw["fc"]["aux_override"]
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
