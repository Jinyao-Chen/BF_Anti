"""Onboard configuration: JSON in, validated dataclasses out.

The example config is config/onboard.example.json. The real aircraft config is
intentionally not committed; copy the example, fill in the measured camera
geometry / FC port / battery threshold, and keep it on the Pi.

Profiles (config/profiles/*.json) overlay the base config. They are for tuning
guidance and safety only; hardware paths live in the base file.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _pick(cls, data: dict):
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in known}


@dataclass(frozen=True)
class FcConfig:
    """MSP link and AUX channel mapping (0-based channel indices)."""

    port: str = "/dev/ttyACM0"
    baud: int = 115200
    tcp_host: str = ""
    tcp_port: int = 5761
    telemetry_hz: float = 25.0
    rc_hz: float = 50.0
    reply_timeout_s: float = 0.15
    link_timeout_s: float = 0.6
    # AETR layout: 0 roll, 1 pitch, 2 throttle, 3 yaw, 4 AUX1 ... 15 AUX12.
    aux_arm: int = 4
    aux_angle: int = 5
    aux_override: int = 6
    # Three-position pilot mode switch (AUX4 by default):
    #   low  (<1300) = FORCE MANUAL: companion never engages
    #   mid  (1300..1700) = AUTO: engage as soon as a target is confirmed
    #   high (>1700) = STRIKE: weapons-hot, same auto engagement
    aux_engage: int = 7
    require_override: bool = True
    require_engage: bool = True
    # When true, the same three-position MODE channel also drives the FC's
    # BOXMSPOVERRIDE range: low = override off (physical CRSF bypass),
    # mid/high = override on (companion may command RPYT).  This makes the
    # low position a hardware force-manual even if the Pi is dead.
    override_from_mode: bool = False
    # Stick takeover is an extra abort in the legacy two-position path.  In
    # the three-position path the mode switch is the deliberate abort; leave
    # this false so AUTO can take over while the pilot is still manoeuvring.
    stick_takeover: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "FcConfig":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class CameraConfig:
    device: str = "/dev/video0"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    fourcc: str = "MJPG"
    rotate_deg: int = 0
    hfov_deg: float = 70.0
    vfov_deg: Optional[float] = None
    tilt_deg: float = 44.0
    yaw_offset_deg: float = 0.0
    warmup_s: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> "CameraConfig":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class DetectorConfig:
    backend: str = "hailo"          # hailo | replay | null
    hef: str = "model/yolov8n_drone_h8l.hef"
    input_width: int = 640
    input_height: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 10
    class_names: tuple[str, ...] = ("drone",)
    async_infer: bool = True
    confirm_frames: int = 2
    max_coast_frames: int = 4
    gate_px: float = 90.0
    gate_gain: float = 3.0
    replay_path: str = ""
    replay_loop: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "DetectorConfig":
        raw = _pick(cls, data)
        if "class_names" in raw and isinstance(raw["class_names"], list):
            raw["class_names"] = tuple(str(x) for x in raw["class_names"])
        return cls(**raw)


@dataclass(frozen=True)
class GuidanceConfig:
    """Opaque parameter block passed to PVisualGuidance.

    Keep guidance gains in the profile, not in code, so a flight card can be
    reproduced exactly. Unknown keys are rejected by PVisualGuidance at build
    time (TypeError), which is deliberate.
    """

    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "GuidanceConfig":
        params = data.get("params", data) if isinstance(data, dict) else {}
        if not isinstance(params, dict):
            raise ValueError("guidance.params must be an object")
        return cls(params=dict(params))


@dataclass(frozen=True)
class SafetyConfig:
    max_rc_age_s: float = 0.30
    max_att_age_s: float = 0.25
    max_alt_age_s: float = 0.35
    max_status_age_s: float = 0.60
    max_loop_dt_s: float = 0.10
    max_det_age_s: float = 0.40
    min_altitude_m: float = 2.0
    max_altitude_m: float = 60.0
    max_range_m: float = 80.0
    min_battery_v: float = 0.0       # 0 disables; set per pack (e.g. 13.6 for 4S)
    target_confirm_s: float = 0.10
    target_lost_s: float = 0.40
    lost_reset_s: float = 1.5
    fault_latch: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "SafetyConfig":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class VideoConfig:
    enabled: bool = True
    fb: str = "/dev/fb0"
    width: int = 720
    height: int = 576
    fps: float = 25.0
    osd: bool = True
    letterbox: bool = True
    font_path: str = ""
    draw_box: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> "VideoConfig":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class ControlConfig:
    rate_hz: float = 50.0
    # angle = firmware ANGLE inner loop (RC stick is a desired angle)
    # rate  = companion attitude P + firmware ACRO rate loop
    att_mode: str = "angle"
    att_rate_gain: float = 6.0      # 1/s: attitude error (deg) -> rate (dps)
    rate_max_dps: float = 250.0
    rate_full_dps: float = 600.0    # must match Betaflight ACTUAL full-stick rate
    angle_limit_deg: float = 60.0
    yaw_full_rate_dps: float = 300.0
    rc_rate_limit: int = 60
    roll_sign: float = 1.0
    pitch_sign: float = 1.0
    yaw_sign: float = 1.0
    throttle_min: int = 1000
    throttle_max: int = 2000
    hover_throttle: int = 1670

    @classmethod
    def from_dict(cls, data: dict) -> "ControlConfig":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class TelemetryConfig:
    """Flight-data blackbox (see onboard/telemetry_log.py).

    hz is the CSV row rate, not the control rate: the 50 Hz loop is decimated
    so a 32 MB segment holds hours of flight.  Set hz to 0 to log every tick.
    """

    enabled: bool = True
    hz: float = 10.0
    max_file_mb: float = 32.0
    max_runs: int = 20
    flush_interval_s: float = 1.0
    fsync_interval_s: float = 5.0
    level: str = "INFO"

    @classmethod
    def from_dict(cls, data: dict) -> "TelemetryConfig":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class AppConfig:
    profile: str = ""
    log_dir: str = "logs"
    fc: FcConfig = field(default_factory=FcConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        return cls(
            profile=str(data.get("profile", "")),
            log_dir=str(data.get("log_dir", "logs")),
            fc=FcConfig.from_dict(data.get("fc", {})),
            camera=CameraConfig.from_dict(data.get("camera", {})),
            detector=DetectorConfig.from_dict(data.get("detector", {})),
            guidance=GuidanceConfig.from_dict(data.get("guidance", {})),
            safety=SafetyConfig.from_dict(data.get("safety", {})),
            video=VideoConfig.from_dict(data.get("video", {})),
            control=ControlConfig.from_dict(data.get("control", {})),
            telemetry=TelemetryConfig.from_dict(data.get("telemetry", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if not (10.0 <= self.control.rate_hz <= 200.0):
            raise ValueError("control.rate_hz must be in [10, 200]")
        if self.camera.width <= 0 or self.camera.height <= 0:
            raise ValueError("camera width/height must be positive")
        if not (0.0 < self.camera.fps <= 240.0):
            raise ValueError("camera.fps must be in (0, 240]")
        if int(self.camera.rotate_deg) % 360 not in (0, 90, 180, 270):
            raise ValueError("camera.rotate_deg must be 0/90/180/270")
        if self.detector.backend not in ("hailo", "replay", "null"):
            raise ValueError("detector.backend must be hailo, replay or null")
        if self.detector.input_width <= 0 or self.detector.input_height <= 0:
            raise ValueError("detector input size must be positive")
        if str(self.control.att_mode).lower() not in ("angle", "rate"):
            raise ValueError("control.att_mode must be angle or rate")
        if self.control.att_rate_gain <= 0.0 or self.control.rate_max_dps <= 0.0:
            raise ValueError("control.att_rate_gain/rate_max_dps must be positive")
        if self.control.rate_full_dps <= 0.0:
            raise ValueError("control.rate_full_dps must be positive")
        if self.control.throttle_min >= self.control.throttle_max:
            raise ValueError("throttle_min must be < throttle_max")
        if not (self.control.throttle_min <= self.control.hover_throttle <= self.control.throttle_max):
            raise ValueError("hover_throttle must be within throttle limits")
        aux = [self.fc.aux_arm, self.fc.aux_angle, self.fc.aux_override, self.fc.aux_engage]
        if any(not (0 <= a <= 15) for a in aux):
            raise ValueError("AUX channel indices must be in [0, 15]")
        if self.fc.override_from_mode:
            # override and mode are the same physical switch by design.
            if self.fc.aux_arm == self.fc.aux_engage:
                raise ValueError("aux_arm and aux_engage must be distinct")
            if self.fc.aux_angle == self.fc.aux_engage:
                raise ValueError("aux_angle and aux_engage must be distinct")
        elif len(set(aux)) != len(aux):
            raise ValueError("aux_arm/angle/override/engage must be distinct")
        for name in (
            "max_rc_age_s", "max_att_age_s", "max_alt_age_s", "max_status_age_s",
            "max_loop_dt_s", "max_det_age_s", "target_confirm_s", "target_lost_s",
        ):
            if float(getattr(self.safety, name)) <= 0.0:
                raise ValueError("safety.%s must be positive" % name)
        if self.safety.min_altitude_m >= self.safety.max_altitude_m:
            raise ValueError("safety.min_altitude_m must be < max_altitude_m")
        if not (0.0 <= self.telemetry.hz <= 100.0):
            raise ValueError("telemetry.hz must be in [0, 100]")
        if self.telemetry.max_file_mb <= 0.0:
            raise ValueError("telemetry.max_file_mb must be positive")
        if self.telemetry.max_runs < 1:
            raise ValueError("telemetry.max_runs must be >= 1")
        if self.telemetry.flush_interval_s <= 0.0:
            raise ValueError("telemetry.flush_interval_s must be positive")
        if self.telemetry.fsync_interval_s < 0.0:
            raise ValueError("telemetry.fsync_interval_s must be >= 0")
        if str(self.telemetry.level).upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError("telemetry.level must be DEBUG/INFO/WARNING/ERROR/CRITICAL")

    def build_guidance(self):
        """Instantiate PVisualGuidance, injecting measured camera geometry."""
        from guidance.p_visual_guidance import PVisualGuidance

        params = dict(self.guidance.params)
        params["camera_tilt_deg"] = float(self.camera.tilt_deg)
        width, height = int(self.camera.width), int(self.camera.height)
        hfov = float(self.camera.hfov_deg)
        rotate = int(self.camera.rotate_deg) % 360
        if rotate in (90, 270):
            width, height = height, width
            if self.camera.vfov_deg is not None:
                hfov = float(self.camera.vfov_deg)
            else:
                import math
                hfov = math.degrees(2.0 * math.atan(
                    math.tan(math.radians(hfov) * 0.5)
                    * float(self.camera.height) / float(self.camera.width)
                ))
        params["image_size"] = (width, height)
        params["hfov_deg"] = hfov
        params.setdefault("camera_yaw_offset_deg", float(self.camera.yaw_offset_deg))
        return PVisualGuidance(**params)


def load_config(path: str, profile_path: Optional[str] = None) -> AppConfig:
    """Load a base JSON config, optionally overlaid by a profile JSON."""
    with open(os.path.expanduser(path), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError("config root must be an object")
    raw = _expand(raw)

    profile = str(raw.get("profile", "")).strip()
    if profile and profile_path is None:
        base_dir = os.path.dirname(os.path.abspath(path))
        candidate = os.path.join(base_dir, "profiles", profile + ".json")
        if os.path.isfile(candidate):
            profile_path = candidate
    if profile_path:
        with open(os.path.expanduser(profile_path), "r", encoding="utf-8") as fh:
            overlay = json.load(fh)
        if not isinstance(overlay, dict):
            raise ValueError("profile root must be an object")
        raw = _deep_merge(raw, _expand(overlay))

    cfg = AppConfig.from_dict(raw)
    cfg.validate()
    return cfg
