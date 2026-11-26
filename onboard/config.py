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
    unknown = sorted(k for k in data if k not in known)
    if unknown:
        raise ValueError(
            "%s: unknown key(s) %s (typo or stale config?)"
            % (cls.__name__, ", ".join(unknown))
        )
    return {k: v for k, v in data.items() if k in known}


@dataclass(frozen=True)
class FcConfig:
    """MSP link and AUX channel mapping (0-based channel indices)."""

    port: str = "/dev/ttyACM0"
    baud: int = 115200
    tcp_host: str = ""
    tcp_port: int = 5761
    telemetry_hz: float = 50.0
    rc_hz: float = 50.0
    reply_timeout_s: float = 0.15
    link_timeout_s: float = 0.6
    # One extra metadata query per telemetry tick after (re)connect, so AUX
    # mode mapping and RX_MAP are verified without a blocking discovery pass.
    enable_aux_discovery: bool = True
    aux_meta_timeout_s: float = 0.08
    # MSP2_POSITION_STATE is a simulator-only extension on the flight FC.
    # The standard 4.5.3 firmware ignores it, costing 3 x reply_timeout at
    # boot; only enable it for SITL/bench firmware that implements 0x3015.
    enable_msp2_position: bool = False
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
    # Brown-Conrady k1,k2,p1,p2,k3 from a chessboard calibration.  None keeps
    # the legacy pinhole model.
    dist_coeffs: Optional[tuple[float, ...]] = None
    # Extra capture->timestamp delay not represented by time.monotonic()
    # (MJPEG decode / rotation / driver queue).  Calibrate, do not guess.
    latency_s: float = 0.0
    warmup_s: float = 1.0
    # Refuse to run on a driver-selected resolution that does not match the
    # static intrinsics.  Set false only for a consciously uncalibrated bench.
    strict_size: bool = True
    # Optional raw OpenCV/V4L2 controls (vendor units).  None = leave alone.
    autofocus: Optional[int] = None
    auto_exposure: Optional[int] = None
    exposure: Optional[float] = None
    gain: Optional[float] = None
    auto_wb: Optional[int] = None
    wb_temp: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict) -> "CameraConfig":
        raw = _pick(cls, data)
        if raw.get("dist_coeffs") is not None:
            raw["dist_coeffs"] = tuple(float(v) for v in raw["dist_coeffs"])
        return cls(**raw)


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
    # HailoRT job.wait timeout.  10 s used to be longer than stop()'s join,
    # so one wedged job could block shutdown; 1 s is enough for one inference.
    infer_timeout_ms: int = 1000
    confirm_frames: int = 2
    max_coast_frames: int = 6
    # Time-based coast bound removes the control-rate coupling of the old
    # frame counter (6 frames was 120 ms at 50 Hz but 300 ms at 20 Hz).
    max_coast_s: float = 0.25
    gate_px: float = 90.0
    # Minimum gate as a fraction of image width (0 disables).
    gate_frac: float = 0.15
    gate_gain: float = 3.0
    coast_vel_tau_s: float = 0.24
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
    # A single sagging battery sample must not latch FAULT.  0 = old
    # single-sample behaviour; e.g. 1.0 requires 1 s below min_battery_v.
    battery_debounce_s: float = 0.0
    # Battery voltage older than this is treated as absent; 0 V is never
    # interpreted as a valid pack voltage.
    max_analog_age_s: float = 2.0
    # Last-resort terminal coast window (guidance lost_timeout_s is preferred).
    terminal_coast_s: float = 0.9

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
    # Real-aircraft default: ACRO/rate.  ANGLE must be a measured, explicit
    # profile choice because the bench FC has no ANGLE mode range.
    att_mode: str = "rate"
    att_rate_gain: float = 6.0      # 1/s: attitude error (deg) -> rate (dps)
    rate_max_dps: float = 250.0
    rate_full_dps: float = 600.0    # must match Betaflight ACTUAL full-stick rate
    angle_limit_deg: float = 60.0
    yaw_full_rate_dps: float = 300.0
    rc_rate_limit: int = 60
    # 0 keeps the legacy per-tick limit; >0 is the physical slew limit in
    # RC-units/second, independent of control rate (e.g. 1200 = 500 units in
    # 0.42 s).  Profiles for the real aircraft should set this.
    rc_slew_rate_per_s: float = 0.0
    roll_sign: float = 1.0
    pitch_sign: float = 1.0
    yaw_sign: float = 1.0
    throttle_min: int = 1000
    throttle_max: int = 2000
    hover_throttle: int = 1430

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
    max_segments: int = 20
    flush_interval_s: float = 1.0
    fsync_interval_s: float = 5.0
    level: str = "INFO"

    @classmethod
    def from_dict(cls, data: dict) -> "TelemetryConfig":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class PowerConfig:
    """RC-controlled graceful shutdown (Pi 5 has no power-loss protection).

    aux_channel is a 0-based AETR index: 6 = CH7/AUX3.  -1 disables the
    feature.  The guard only fires after a low->high edge, with the FC
    disarmed, throttle at idle, fresh RC/status data, and the channel held
    high for hold_s.  It never cuts power itself; it asks systemd to halt so
    the blackbox and SD card are flushed before the pack is unplugged.
    """

    aux_channel: int = -1
    hold_s: float = 2.0
    require_disarmed: bool = True
    max_throttle: int = 1100
    max_rc_age_s: float = 0.30
    max_status_age_s: float = 0.60
    high_threshold: int = 1700
    # Require the switch to be genuinely LOW before accepting a new high
    # edge; a mid/1500 value or a boot-time 0 is not a deliberate cycle.
    low_threshold: int = 1300
    # Betaflight's RX_MIN_USEC; values below this are a missing/failsafe
    # channel, not a pilot switch position.
    min_low_threshold: int = 885

    @classmethod
    def from_dict(cls, data: dict) -> "PowerConfig":
        return cls(**_pick(cls, data))


@dataclass(frozen=True)
class AppConfig:
    profile: str = ""
    log_dir: str = "logs"
    fc: FcConfig = field(default_factory=FcConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    guidance: GuidanceConfig = field(default_factory=GuidanceConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(k for k in data if k not in known)
        if unknown:
            raise ValueError("AppConfig: unknown key(s) " + ", ".join(unknown))
        return cls(
            profile=str(data.get("profile", "")),
            log_dir=str(data.get("log_dir", "logs")),
            fc=FcConfig.from_dict(data.get("fc", {})),
            power=PowerConfig.from_dict(data.get("power", {})),
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
        if not (0.0 < float(self.camera.hfov_deg) < 180.0):
            raise ValueError("camera.hfov_deg must be in (0, 180)")
        if self.camera.vfov_deg is not None and not (0.0 < float(self.camera.vfov_deg) < 180.0):
            raise ValueError("camera.vfov_deg must be in (0, 180)")
        if not (0.0 < self.camera.fps <= 240.0):
            raise ValueError("camera.fps must be in (0, 240]")
        if not (0.0 <= self.camera.latency_s <= 0.5):
            raise ValueError("camera.latency_s must be in [0, 0.5]")
        if self.camera.dist_coeffs is not None:
            if not (4 <= len(self.camera.dist_coeffs) <= 8):
                raise ValueError("camera.dist_coeffs must have 4..8 values")
        if int(self.camera.rotate_deg) % 360 not in (0, 90, 180, 270):
            raise ValueError("camera.rotate_deg must be 0/90/180/270")
        if self.detector.backend not in ("hailo", "replay", "null"):
            raise ValueError("detector.backend must be hailo, replay or null")
        if self.detector.input_width <= 0 or self.detector.input_height <= 0:
            raise ValueError("detector input size must be positive")
        if self.detector.max_coast_s < 0.0 or self.detector.coast_vel_tau_s < 0.0:
            raise ValueError("detector coast time constants must be >= 0")
        if self.detector.gate_px < 0.0 or self.detector.gate_frac < 0.0:
            raise ValueError("detector gate_px/gate_frac must be >= 0")
        if not (50 <= int(self.detector.infer_timeout_ms) <= 10000):
            raise ValueError("detector.infer_timeout_ms must be in [50, 10000]")
        if str(self.control.att_mode).lower() not in ("angle", "rate"):
            raise ValueError("control.att_mode must be angle or rate")
        if self.control.att_rate_gain <= 0.0 or self.control.rate_max_dps <= 0.0:
            raise ValueError("control.att_rate_gain/rate_max_dps must be positive")
        if self.control.rate_full_dps <= 0.0:
            raise ValueError("control.rate_full_dps must be positive")
        if float(self.control.rc_slew_rate_per_s) < 0.0:
            raise ValueError("control.rc_slew_rate_per_s must be >= 0")
        if self.control.throttle_min >= self.control.throttle_max:
            raise ValueError("throttle_min must be < throttle_max")
        if not (self.control.throttle_min <= self.control.hover_throttle <= self.control.throttle_max):
            raise ValueError("hover_throttle must be within throttle limits")
        aux = [self.fc.aux_arm, self.fc.aux_angle, self.fc.aux_override, self.fc.aux_engage]
        if any(not (0 <= a <= 15) for a in aux):
            raise ValueError("AUX channel indices must be in [0, 15]")
        if self.fc.aux_arm == self.fc.aux_angle:
            raise ValueError("aux_arm and aux_angle must be distinct")
        if self.fc.override_from_mode:
            # override and mode are the same physical switch by design.
            if self.fc.aux_arm == self.fc.aux_engage:
                raise ValueError("aux_arm and aux_engage must be distinct")
            if self.fc.aux_angle == self.fc.aux_engage:
                raise ValueError("aux_angle and aux_engage must be distinct")
        elif len(set(aux)) != len(aux):
            raise ValueError("aux_arm/angle/override/engage must be distinct")
        if not (-1 <= int(self.power.aux_channel) <= 15):
            raise ValueError("power.aux_channel must be -1 or in [0, 15]")
        if int(self.power.aux_channel) >= 0 and int(self.power.aux_channel) in set(aux):
            raise ValueError("power.aux_channel must differ from the flight AUX channels")
        if self.power.hold_s <= 0.0:
            raise ValueError("power.hold_s must be positive")
        if not (1000 <= int(self.power.max_throttle) <= 2000):
            raise ValueError("power.max_throttle must be in [1000, 2000]")
        if not (1000 <= int(self.power.high_threshold) <= 2000):
            raise ValueError("power.high_threshold must be in [1000, 2000]")
        if not (1000 <= int(self.power.low_threshold) <= 2000):
            raise ValueError("power.low_threshold must be in [1000, 2000]")
        if int(self.power.low_threshold) >= int(self.power.high_threshold):
            raise ValueError("power.low_threshold must be < high_threshold")
        if not (800 <= int(self.power.min_low_threshold)
                <= int(self.power.low_threshold)):
            raise ValueError("power.min_low_threshold must be in [800, low_threshold]")
        if self.power.max_rc_age_s <= 0.0 or self.power.max_status_age_s <= 0.0:
            raise ValueError("power.max_*_age_s must be positive")
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
        if self.telemetry.max_segments < 1:
            raise ValueError("telemetry.max_segments must be >= 1")
        if self.safety.battery_debounce_s < 0.0:
            raise ValueError("safety.battery_debounce_s must be >= 0")
        if self.safety.max_analog_age_s <= 0.0:
            raise ValueError("safety.max_analog_age_s must be positive")
        if self.safety.terminal_coast_s <= 0.0:
            raise ValueError("safety.terminal_coast_s must be positive")
        if self.telemetry.flush_interval_s <= 0.0:
            raise ValueError("telemetry.flush_interval_s must be positive")
        # <=0 disables periodic fsync; the final close still fsyncs.
        if str(self.telemetry.level).upper() not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError("telemetry.level must be DEBUG/INFO/WARNING/ERROR/CRITICAL")

    def build_guidance(self):
        """Instantiate PVisualGuidance, injecting measured camera geometry."""
        from guidance.p_visual_guidance import PVisualGuidance

        params = dict(self.guidance.params)
        params["camera_tilt_deg"] = float(self.camera.tilt_deg)
        import math

        width, height = int(self.camera.width), int(self.camera.height)
        raw_hfov = float(self.camera.hfov_deg)
        raw_vfov = (float(self.camera.vfov_deg)
                    if self.camera.vfov_deg is not None else None)
        rotate = int(self.camera.rotate_deg) % 360
        if rotate in (90, 270):
            # The decoded frame is transposed.  Its horizontal FOV is the
            # sensor's vertical FOV and vice versa; passing only hfov here
            # would silently distort fy.
            if raw_vfov is None:
                raw_vfov = math.degrees(2.0 * math.atan(
                    math.tan(math.radians(raw_hfov) * 0.5)
                    * float(self.camera.height) / float(self.camera.width)
                ))
            width, height = height, width
            out_hfov, out_vfov = raw_vfov, raw_hfov
        else:
            out_hfov, out_vfov = raw_hfov, raw_vfov
        params["image_size"] = (width, height)
        params["hfov_deg"] = out_hfov
        params["vfov_deg"] = out_vfov
        # Keep the guidance's RC scaling/anchor consistent with the controller
        # unless the tuning card explicitly overrides it.
        params.setdefault("hover_pwm", float(self.control.hover_throttle))
        params.setdefault("angle_limit_deg", float(self.control.angle_limit_deg))
        params.setdefault("yaw_full_rate_dps", float(self.control.yaw_full_rate_dps))
        params.setdefault("camera_yaw_offset_deg", float(self.camera.yaw_offset_deg))
        params.setdefault("camera_dist_coeffs", self.camera.dist_coeffs)
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
        else:
            # A named profile that silently disappears would fall back to old
            # SITL gains.  Refuse to start instead of flying the wrong card.
            raise FileNotFoundError(
                "profile %r not found at %s (config references it)"
                % (profile, candidate)
            )
    if profile_path:
        with open(os.path.expanduser(profile_path), "r", encoding="utf-8") as fh:
            overlay = json.load(fh)
        if not isinstance(overlay, dict):
            raise ValueError("profile root must be an object")
        raw = _deep_merge(raw, _expand(overlay))

    cfg = AppConfig.from_dict(raw)
    cfg.validate()
    return cfg
