'''Regression tests for the 2026-11-26 full-stack audit.

Each test corresponds to a source-level finding that was verified against
Betaflight 4.5.3 / the actual runtime, not merely taken from the report.
'''
from __future__ import annotations

import json
import struct
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from onboard.app import OnboardApp
from guidance.p_visual_guidance import (
    PStrikeCommand,
    PStrikePhase,
    PVisualGuidance,
    _clamp,
    _rc,
    p_cmd_to_rc,
)
from msp import MSP_CMD, FrameParser
from onboard.camera import CameraSource
from onboard.config import (
    ControlConfig,
    FcConfig,
    PowerConfig,
    SafetyConfig,
    TelemetryConfig,
    load_config,
)
from onboard.detector.postprocess import MAX_NMS_CANDIDATES, Letterbox, _finish, nms
from onboard.fc.mode import AuxDiscovery
from onboard.fc.state import FcStateEstimator
from onboard.power import ShutdownGuard
from onboard.safety import SafetyInputs, SafetySupervisor
from onboard.telemetry_log import TelemetryLog
from onboard.types import FcState
from onboard.video.pipeline import DisplayPipeline
from perception.bbox_tracker import BboxTracker
from perception.camera_geometry import CameraIntrinsics, pixel_to_camera_ray
from rc_mux import RCMux, TrackCmd


# ---------------------------------------------------------------------------
# MSP framing / mode bits
# ---------------------------------------------------------------------------
def test_msp_v2_oversize_payload_resyncs_immediately():
    parser = FrameParser()
    # $X> cmd=0x1234 size=4096, but no payload follows.
    # $X + direction + flags + cmd(u16) + size(u16) + payload + crc
    frame = (b"$X>" + b"\x00" + struct.pack("<H", 0x1234)
             + struct.pack("<H", 4096) + b"\x00")
    assert parser.feed(frame) == []
    assert parser.errors >= 1
    assert len(parser.pending) < 9


def test_boxids_index_is_status_bit_not_permanent_id():
    disc = AuxDiscovery(
        boxes=[(0, "ARM"), (1, "ANGLE"), (50, "MSP OVERRIDE")],
        bit_index_by_pid={0: 0, 1: 1, 50: 2},
    )
    assert disc.bit_index_for(1) == 1
    assert disc.is_mode_active(1 << 0, 0) is True
    assert disc.is_mode_active(1 << 0, 1) is False
    assert disc.is_mode_active(1 << 1, 1) is True


def test_state_metadata_discovers_aux_layout_without_blocking_discover_aux():
    class Link:
        generation = 1
        errors = 0
        last_rx_t = 0.0
        last_telem_rx_t = 0.0

        def __init__(self):
            self.responses = {
                (MSP_CMD.BOXIDS, b"\x00"): bytes([0, 1, 50]),
                (MSP_CMD.MODE_RANGES, b""): (
                    bytes([0, 0, 32, 48]) + bytes([50, 3, 16, 48])
                ),
                (MSP_CMD.RX_MAP, b""): bytes([0, 1, 3, 2, 4, 5, 6, 7]),
            }

        def query(self, cmd, is_v2=False, timeout=None, payload=b""):
            return self.responses.get((cmd, payload))

    est = FcStateEstimator(Link(), FcConfig(telemetry_hz=50,
                                            enable_msp2_position=False,
                                            enable_aux_discovery=True))
    for _ in range(8):
        est._poll_once()
    disc = est.aux_discovery
    assert disc is not None
    assert disc.bit_index_for(1) == 1
    assert disc.arm_channel == 4
    assert disc.arm_range == (1700, 2100)
    assert disc.override_channel == 7
    assert disc.rx_map_is_aetr() is True


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
def _safe_input(**kw):
    base = dict(
        now=10.0, armed=True, override_on=True, engage_on=True,
        rc_age_s=0.01, att_age_s=0.01, alt_age_s=0.01, status_age_s=0.01,
        link_age_s=0.01, altitude_m=10.0, range_m=12.0, loop_dt_s=0.02,
        target_confirmed=True, target_age_s=0.02, battery_v=16.0,
        analog_age_s=0.02, fc_layout_ok=True,
    )
    base.update(kw)
    return SafetyInputs(**base)


def test_safety_fails_closed_on_unknown_layout_and_mode():
    sup = SafetySupervisor(SafetyConfig(), require_layout=True)
    dec = sup.evaluate(_safe_input(fc_layout_ok=False))
    assert dec.reason == "fc_layout_unknown"
    dec = sup.evaluate(_safe_input(angle_mode_active=None,
                                   expected_att_mode="rate"))
    assert dec.reason == "att_mode_unknown"


def test_disarmed_stale_does_not_latch_but_armed_stale_does():
    sup = SafetySupervisor(SafetyConfig())
    assert sup.evaluate(_safe_input(armed=False, rc_age_s=10.0)).critical is False
    assert sup.evaluate(_safe_input()).allow_auto is True
    sup = SafetySupervisor(SafetyConfig())
    assert sup.evaluate(_safe_input(rc_age_s=10.0)).critical is True


def test_battery_zero_and_stale_are_not_silently_accepted():
    sup = SafetySupervisor(SafetyConfig(min_battery_v=13.6), require_layout=True)
    assert sup.evaluate(_safe_input(battery_v=0.0)).reason == "battery_telemetry_stale"
    assert sup.evaluate(_safe_input(analog_age_s=5.0)).reason == "battery_telemetry_stale"
    assert sup.evaluate(_safe_input(battery_v=13.0)).critical is True


def test_coasting_window_is_explicit_and_bounded():
    sup = SafetySupervisor(SafetyConfig(target_lost_s=0.4), require_layout=True)
    dec = sup.evaluate(_safe_input(target_confirmed=True, target_age_s=99,
                                   coasting=True, track_lost_age_s=0.5,
                                   coast_max_s=0.9))
    assert dec.allow_auto is True
    dec = sup.evaluate(_safe_input(target_confirmed=True, target_age_s=99,
                                   coasting=True, track_lost_age_s=1.0,
                                   coast_max_s=0.9))
    assert dec.reason == "target_stale"


# ---------------------------------------------------------------------------
# Guidance / tracking
# ---------------------------------------------------------------------------
def _guidance_with_target(bbox=(520.0, 320.0, 580.0, 380.0)):
    g = PVisualGuidance(image_size=(640, 480), hfov_deg=68.75,
                        camera_tilt_deg=40.0, lift_ff=True, hover_pwm=1430.0)
    g.update(bbox, (0.0, 0.0, 10.0), np.zeros(3), (0.0, -10.0, 0.0),
             10.0, 0.05, fresh=True)
    return g


def test_coast_keeps_lift_ff_and_levels_attitude():
    g = _guidance_with_target()
    g._last_cmd = replace(g._last_cmd, roll_cmd_deg=28.0, pitch_cmd_deg=-35.0,
                          yaw_rate_dps=250.0)
    g.update(None, (0.0, 0.0, 10.0), np.zeros(3),
             (20.0, -30.0, 0.0), 10.02, 0.02, fresh=False)
    cmd = g.update(None, (0.0, 0.0, 10.0), np.zeros(3),
                   (20.0, -30.0, 0.0), 10.12, 0.02, fresh=False)
    assert cmd.phase == PStrikePhase.COAST
    assert cmd.yaw_rate_dps == 0.0
    assert abs(cmd.roll_cmd_deg) < 28.0
    assert cmd.throttle_pwm > 1430.0   # measured tilt lift compensation
    assert g._lost_since == 10.02


def test_reacquisition_clears_lost_since():
    g = _guidance_with_target()
    g.update(None, (0.0, 0.0, 10.0), np.zeros(3), (0.0, -10.0, 0.0),
             10.05, 0.05, fresh=False)
    assert g._lost_since is not None
    g.update((520.0, 320.0, 580.0, 380.0), (0.0, 0.0, 10.0), np.zeros(3),
             (0.0, -10.0, 0.0), 10.10, 0.05, fresh=True)
    assert g._lost_since is None


def test_clipped_bbox_freezes_range_and_does_not_hit():
    g = PVisualGuidance(image_size=(640, 480), hfov_deg=68.75,
                        camera_tilt_deg=40.0, lift_ff=True)
    first = g.update((300.0, 320.0, 340.0, 360.0), (0.0, 0.0, 10.0),
                     np.zeros(3), (0.0, -10.0, 0.0), 0.0, 0.05)
    clipped = g.update((0.0, 320.0, 20.0, 360.0), (0.0, 0.0, 10.0),
                       np.zeros(3), (0.0, -10.0, 0.0), 0.05, 0.05)
    assert clipped.bbox_clipped is True
    assert clipped.range_est_m == pytest.approx(first.range_est_m, rel=1e-6)
    assert clipped.hit is False


def test_nan_does_not_become_full_throttle_or_crash_rc_mapping():
    assert _clamp(float("nan"), 1000.0, 2000.0) == 1000.0
    assert _rc(float("nan")) == 1500
    ch = p_cmd_to_rc(PStrikeCommand(throttle_pwm=float("nan")))
    assert all(1000 <= int(v) <= 2000 for v in ch[:4])


def test_tracker_time_based_coast_is_independent_of_control_rate():
    t = BboxTracker(confirm_frames=1, max_coast_s=0.25,
                    coast_vel_tau_s=0.24, gate_px=10.0)
    t.update((100.0, 100.0, 110.0, 120.0), dt=0.05, timestamp=0.0)
    for i in range(4):
        out = t.update(None, dt=0.05, timestamp=0.05 * (i + 1))
        assert out.bbox is not None
    assert t.update(None, dt=0.05, timestamp=0.40).bbox is not None
    lost = t.update(None, dt=0.05, timestamp=0.45)
    assert lost.bbox is None and lost.reason == "lost"


def test_camera_strict_size_rejects_silent_resolution_fallback():
    class BadCapture:
        def __init__(self):
            self._w = 640
            self._h = 480

        def isOpened(self):
            return True

        def set(self, prop, value):
            return True

        def get(self, prop):
            if prop == 3:
                return self._w
            if prop == 4:
                return self._h
            if prop == 5:
                return 30.0
            if prop == 6:
                return 0.0
            return 0.0

        def release(self):
            pass

    cfg = SimpleNamespace(device="/dev/video-test", width=1280, height=720,
                          fps=30.0, fourcc="MJPG", warmup_s=0.0,
                          strict_size=True)
    src = CameraSource(cfg, capture_factory=lambda _dev: BadCapture())
    assert src._open() is None
    assert "geometry mismatch" in src.stats().last_error
    cfg2 = SimpleNamespace(**{**vars(cfg), "strict_size": False})
    src2 = CameraSource(cfg2, capture_factory=lambda _dev: BadCapture())
    assert src2._open() is not None


def test_eight_parameter_distortion_is_finite_at_extreme_pixels():
    intr = CameraIntrinsics(1280, 720, 500.0, 500.0, 640.0, 360.0,
                            dist_coeffs=(-0.25, 0.05, 0.001, -0.002, 0.0,
                                         0.01, -0.005, 0.001))
    for uv in ((-10_000.0, -10_000.0), (10_000.0, 10_000.0)):
        ray = pixel_to_camera_ray(uv, intr)
        assert np.all(np.isfinite(ray))
        assert np.linalg.norm(ray) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# NMS / video
# ---------------------------------------------------------------------------
def test_nms_topk_and_nan_filter():
    count = 1000
    boxes = np.column_stack([np.arange(count), np.arange(count),
                             np.arange(count) + 1.0, np.arange(count) + 1.0])
    scores = np.linspace(1.0, 0.1, count)
    kept = nms(boxes, scores, 0.45)
    assert len(kept) <= MAX_NMS_CANDIDATES
    params = Letterbox(640, 480, 640, 640, 1.0, 1.0, 0.0, 0.0)
    assert _finish(np.array([[np.nan, 0, 10, 10]]),
                   np.array([0.9]), np.array([0]), params, 10, 0.0) == []


def test_pipeline_renders_hud_without_a_camera_frame():
    rendered = []

    class Renderer:
        def render(self, frame, *args, **kwargs):
            rendered.append(frame)
            return np.zeros((2, 2, 4), dtype=np.uint8)

        def set_output_size(self, *args):
            pass

    class Sink:
        closed = False

        def open(self):
            return True

        def write(self, canvas):
            return True

        def close(self):
            self.closed = True

    pipe = DisplayPipeline(SimpleNamespace(enabled=True, fps=25),
                           sink=Sink(), renderer=Renderer(),
                           clock=lambda: 10.0)
    pipe._process((None, None, None, None, None, 10.0, 0.0, 10.0), 0.3)
    assert rendered == [None]


def test_pipeline_closes_sink_after_write_failure():
    class Renderer:
        def render(self, *args, **kwargs):
            return np.zeros((2, 2, 4), dtype=np.uint8)

    class Sink:
        closed = False

        def open(self):
            return True

        def write(self, canvas):
            return False

        def close(self):
            self.closed = True

    sink = Sink()
    pipe = DisplayPipeline(SimpleNamespace(enabled=True, fps=25),
                           sink=sink, renderer=Renderer(),
                           clock=lambda: 10.0)
    item = (np.zeros((2, 2, 3), dtype=np.uint8), None, None, None, None,
            10.0, 10.0, 10.0)
    pipe._process(item, 0.3)
    assert sink.closed is True


# ---------------------------------------------------------------------------
# Config / telemetry / power / mux
# ---------------------------------------------------------------------------
def test_config_rejects_missing_profile_and_unknown_key(tmp_path):
    raw = json.load(open("config/onboard.example.json", encoding="utf-8"))
    raw["profile"] = "missing_card"
    p = tmp_path / "missing.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="missing_card"):
        load_config(str(p))

    raw = json.load(open("config/onboard.example.json", encoding="utf-8"))
    raw["profile"] = ""
    raw["safety"]["min_batery_v"] = 13.6  # deliberate typo
    p = tmp_path / "typo.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="min_batery_v"):
        load_config(str(p))


def test_real_defaults_are_not_the_old_sitl_card():
    assert ControlConfig().att_mode == "rate"
    assert ControlConfig().hover_throttle == 1430
    assert FcConfig().telemetry_hz == 50.0
    assert TelemetryConfig().max_segments == 20


def test_telemetry_run_sequence_survives_wall_clock_rollback(tmp_path):
    seqs = []
    for _ in range(2):
        tlog = TelemetryLog(str(tmp_path), hz=1, max_runs=5,
                            fsync_interval_s=0)
        assert tlog.start()
        tlog.close("seq")
        manifest = json.load(open(
            __import__("os").path.join(tlog.run_dir, "startup.json"),
            encoding="utf-8",
        ))
        seqs.append(manifest["run_seq"])
    assert seqs[1] > seqs[0] > 0


def test_power_switch_needs_true_low_and_throttle_reset():
    g = ShutdownGuard(PowerConfig(aux_channel=6, hold_s=1.0,
                                  low_threshold=1300, max_throttle=1100))
    rc = [1500] * 16
    rc[2] = 1000
    rc[6] = 1500  # mid is not LOW
    st = FcState(rc=tuple(rc), rc_t=10.0, status_t=10.0, armed=False)
    assert g.update(10.0, st) is False
    rc[6] = 1200
    st = FcState(rc=tuple(rc), rc_t=10.1, status_t=10.1, armed=False)
    assert g.update(10.1, st) is False
    rc[6] = 2000
    st = FcState(rc=tuple(rc), rc_t=10.2, status_t=10.2, armed=False)
    assert g.update(10.2, st) is False
    st = FcState(rc=tuple(rc), rc_t=11.3, status_t=11.3, armed=False)
    assert g.update(11.3, st) is True


def test_rc_mux_uses_level_callback_and_discovers_arm_range():
    class Link:
        def __init__(self):
            self.sent = []

        def send_rc_16(self, ch):
            self.sent.append(tuple(int(v) for v in ch))

    ch = [1500] * 16
    ch[2] = 1000
    ch[4] = 1500  # inside a discovered 1300-1700 ARM range
    ch[7] = 1500
    link = Link()
    mux = RCMux(link, mode_channel=7, arm_channel=4,
                base_throttle=1430, channel_source=lambda: (tuple(ch), 0.01))
    mux.arm_range = (1300, 1700)
    mux.level_rpyt_fn = lambda thr: (1600, 1400, thr, 1500)
    mux._mode_seen_low = True
    mux._tick(10.0)
    assert link.sent[-1][:4] == (1600, 1400, 1430, 1500)
    ch[4] = 1000
    mux._tick(10.02)
    assert link.sent[-1][:4] == (1500, 1500, 1000, 1500)


# ---------------------------------------------------------------------------
# RC consent / arm range
# ---------------------------------------------------------------------------
def test_arming_does_not_bypass_mode_seen_low():
    class Link:
        def send_rc_16(self, ch):
            pass

    ch = [1500] * 16
    ch[2] = 1000
    ch[4] = 1000
    ch[7] = 1500
    mux = RCMux(Link(), mode_channel=7, arm_channel=4,
                base_throttle=1200, channel_source=lambda: (tuple(ch), 0.0),
                stick_takeover=False)
    mux.request_auto(True)
    mux.publish_cmd(TrackCmd(pitch=1600, throttle=1400, yaw=1500, t=10.0))
    mux._tick(10.0)          # arm low, mode mid: C7 has not seen LOW
    ch[4] = 2000
    mux._tick(10.02)         # arm edge is consent, but C7 still unmet
    assert mux._auto is False
    ch[7] = 1000
    mux._tick(10.04)         # now the mode is genuinely LOW
    ch[7] = 1500
    mux._tick(10.06)
    assert mux._auto is True


def test_discovered_arm_range_1300_1700_is_honoured():
    class Link:
        def send_rc_16(self, ch):
            pass

    ch = [1500] * 16
    ch[2] = 1000
    ch[4] = 1500          # would fail the old hard-coded >1700 test
    ch[7] = 1000
    mux = RCMux(Link(), mode_channel=7, arm_channel=4,
                base_throttle=1200, channel_source=lambda: (tuple(ch), 0.0),
                stick_takeover=False)
    mux.arm_range = (1300, 1700)
    mux.request_auto(True)
    mux.publish_cmd(TrackCmd(pitch=1600, throttle=1400, yaw=1500, t=10.0))
    mux._tick(10.0)
    ch[7] = 1500
    mux._tick(10.02)
    assert mux._auto is True


# ---------------------------------------------------------------------------
# Shutdown ordering / app safety inputs
# ---------------------------------------------------------------------------
def test_rc_shutdown_flushes_blackbox_before_poweroff():
    order = []

    class Telemetry:
        def event(self, *args, **kwargs):
            return True

        def close(self, reason=""):
            order.append("telemetry_close")

    class Mux:
        def publish_cmd(self, cmd):
            pass

        def stop(self):
            order.append("mux_stop")

    app = OnboardApp(load_config("config/onboard.example.json"),
                     telemetry=Telemetry(), mux=Mux(),
                     poweroff=lambda: (order.append("poweroff"), (True, "queued"))[1])
    app._poweroff_on_exit = True
    app.stop("rc_poweroff")
    assert order.index("telemetry_close") < order.index("poweroff")


def test_app_only_relaxes_target_age_for_near_terminal_coast():
    now = 10.0
    app, _, _, _ = _make_app()
    st = app.state.snapshot(now)
    near = SimpleNamespace(confirmed=True, reason="coast",
                           channels=(1500, 1500, 1430, 1500),
                           command=SimpleNamespace(range_est_m=2.0))
    far = SimpleNamespace(confirmed=True, reason="coast",
                          channels=(1500, 1500, 1430, 1500),
                          command=SimpleNamespace(range_est_m=20.0))
    inp_near = app._safety_inputs(st, None, near, now)
    inp_far = app._safety_inputs(st, None, far, now)
    assert inp_near.coasting is True
    assert inp_far.coasting is False


def _make_app():
    # Reuse the app-test fakes without importing pytest fixtures.
    from tests.test_onboard_app import build_app, fresh_state
    import time as _time
    now = _time.monotonic()
    app, mux, video, now = build_app(state=fresh_state(now))
    return app, mux, video, now


# ---------------------------------------------------------------------------
# Hailo cleanup ordering
# ---------------------------------------------------------------------------
def test_hailo_configure_failure_clears_cpp_objects_before_release(monkeypatch, tmp_path):
    import sys
    import types

    state = {"created": 0, "destroyed": 0, "release_saw_live": False}

    class Base:
        def __init__(self):
            state["created"] += 1

        def __del__(self):
            state["destroyed"] += 1

    class Stream(Base):
        name = "out0"
        shape = (8, 8, 3)
        is_nms = True
        format = SimpleNamespace(type="UINT8")

    class InferModel(Base):
        def set_batch_size(self, n):
            pass

        def input(self):
            return Stream()

        @property
        def outputs(self):
            return [Stream()]

        def configure(self):
            return Configured()

    class Configured(Base):
        def activate(self):
            pass

        def create_bindings(self):
            raise RuntimeError("bindings failed")

    class VDevice:
        def create_infer_model(self, hef):
            return InferModel()

        def release(self):
            state["release_saw_live"] = state["destroyed"] < state["created"]

    fake = types.ModuleType("hailo_platform")
    fake.FormatType = SimpleNamespace(UINT8="UINT8", FLOAT32="FLOAT32")
    fake.VDevice = VDevice
    monkeypatch.setitem(sys.modules, "hailo_platform", fake)

    hef = tmp_path / "model.hef"
    hef.write_bytes(b"fake")
    from onboard.config import DetectorConfig
    from onboard.detector.hailo import HailoDetector
    det = HailoDetector(DetectorConfig(hef=str(hef)))
    with pytest.raises(RuntimeError):
        det._configure()
    assert state["created"] == 4
    assert state["destroyed"] == state["created"]
    assert state["release_saw_live"] is False


def test_tracker_associates_multiple_candidates_near_prediction():
    from onboard.types import Detection as D

    t = BboxTracker(confirm_frames=1, max_coast_s=0.25, gate_px=200.0)
    t.update((100.0, 100.0, 140.0, 140.0), dt=0.02, timestamp=0.0)
    t.update((102.0, 100.0, 142.0, 140.0), dt=0.02, timestamp=0.02)
    false_pos = D((500.0, 300.0, 560.0, 360.0), 0.99, t=0.04)
    true_target = D((104.0, 100.0, 144.0, 140.0), 0.60, t=0.04)
    out = t.update_candidates([false_pos, true_target], dt=0.02, timestamp=0.04)
    assert out.bbox is not None
    assert abs((out.bbox[0] + out.bbox[2]) / 2 - 124.0) < 15.0


def test_config_rejects_arm_and_angle_on_same_channel(tmp_path):
    raw = json.load(open("config/onboard.example.json", encoding="utf-8"))
    raw["profile"] = ""
    raw["fc"]["aux_angle"] = raw["fc"]["aux_arm"]
    p = tmp_path / "same.json"
    p.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="aux_arm and aux_angle"):
        load_config(str(p))


def test_build_guidance_uses_control_limits_for_rc_scaling():
    cfg = load_config("config/onboard.example.json")
    g = cfg.build_guidance()
    assert g.angle_limit_deg == cfg.control.angle_limit_deg
    assert g.yaw_full_rate_dps == cfg.control.yaw_full_rate_dps
    assert g.hover_pwm == cfg.control.hover_throttle


def test_stale_auto_command_still_uses_level_callback():
    class Link:
        def __init__(self):
            self.sent = []

        def send_rc_16(self, ch):
            self.sent.append(tuple(ch))

    ch = [1500] * 16
    ch[2] = 1500
    ch[4] = 2000
    ch[7] = 1500
    link = Link()
    mux = RCMux(link, mode_channel=7, arm_channel=4, base_throttle=1430,
                channel_source=lambda: (tuple(ch), 0.0), stick_takeover=False)
    mux.level_rpyt_fn = lambda thr: (1600, 1400, thr, 1500)
    mux._auto = True
    mux._mode_pos = 1
    mux._mode_seen_low = True
    mux.publish_cmd(TrackCmd(pitch=1600, throttle=1400, yaw=1500, t=0.0))
    mux._tick(10.0)   # cmd is far older than cmd_timeout
    assert link.sent[-1][:4] == (1600, 1400, 1430, 1500)


def test_safe_hover_throttle_is_clamped_after_a_climb():
    class Link:
        def __init__(self):
            self.sent = []

        def send_rc_16(self, ch):
            self.sent.append(tuple(ch))

    ch = [1500] * 16
    ch[2] = 1850
    ch[4] = 2000
    ch[7] = 1000
    link = Link()
    mux = RCMux(link, mode_channel=7, arm_channel=4, base_throttle=1430,
                channel_source=lambda: (tuple(ch), 0.0), stick_takeover=False)
    mux._mode_seen_low = True
    mux._tick(10.0)       # sample pilot 1850 while override is off
    assert mux._safe_hover_thr <= 1580
    ch[7] = 1500
    mux._tick(10.02)
    assert link.sent[-1][2] <= 1580


def test_app_denies_auto_when_rx_map_is_not_aetr():
    app, _, _, now = _make_app()
    app.state.aux_discovery.rx_map = (0, 1, 2, 3, 4, 5, 6, 7)  # TAER-like
    app._engaged = True
    _, decision = app.step_once(now)
    assert decision.allow_auto is False
    assert decision.reason == "fc_layout_unknown"


def test_level_callback_signs_and_scale():
    app, _, _, now = _make_app()
    app.state._state.rpy_deg = (10.0, 20.0, 0.0)
    app.state._state.armed = True
    roll, pitch, throttle, yaw = app._level_rpyt(1430)
    # k=6: p=-60 dps, q=-120*cos(10 deg) dps; 500/600 RC-per-dps.
    assert roll == 1450
    assert 1597 <= pitch <= 1600
    assert throttle == 1430 and yaw == 1500


def test_msp_link_drops_after_repeated_uart_timeouts():
    from onboard.fc.link import MspLink

    class Transport:
        def __init__(self):
            self.open_calls = 0
            self.closed = False

        def open(self):
            self.open_calls += 1

        def read(self, timeout_s):
            return b""

        def write(self, data):
            return None

        def close(self):
            self.closed = True

    tr = Transport()
    link = MspLink(tr, reply_timeout_s=0.001, reconnect_interval_s=0.0)
    assert link.connect() is True
    for _ in range(5):
        assert link.query(MSP_CMD.GET_ATTITUDE, timeout=0.001) is None
    assert link.connected is False
    assert tr.closed is True


def test_ack_does_not_refresh_telemetry_stamp():
    from msp import build_frame
    from onboard.fc.link import MspLink

    class Transport:
        def __init__(self):
            self.frames = [build_frame(MSP_CMD.SET_RAW_RC, b"\x00" * 32),
                           build_frame(MSP_CMD.GET_ATTITUDE, b"\x00" * 6)]

        def open(self):
            pass

        def read(self, timeout_s):
            return self.frames.pop(0) if self.frames else b""

        def write(self, data):
            pass

        def close(self):
            pass

    link = MspLink(Transport(), reply_timeout_s=0.05)
    assert link.connect()
    link.query(MSP_CMD.SET_RAW_RC, timeout=0.05)
    assert link.last_rx_t > 0.0
    assert link.last_telem_rx_t == 0.0
    link.query(MSP_CMD.GET_ATTITUDE, timeout=0.05)
    assert link.last_telem_rx_t > 0.0


# ---------------------------------------------------------------------------
# Second-pass review regression tests (P0 & P1 fixes)
# ---------------------------------------------------------------------------

def test_level_rpyt_returns_neutral_when_attitude_stale():
    import time
    from onboard.app import OnboardApp
    from onboard.config import AppConfig
    from onboard.fc.state import FcState

    cfg = AppConfig()
    app = OnboardApp(cfg)
    now = time.monotonic()
    stale_snap = FcState(
        rpy_deg=(30.0, -40.0, 0.0),
        armed=True,
        status_t=now,
        att_t=now - 2.0,  # stale beyond max_att_age_s
    )
    app.state = SimpleNamespace(snapshot=lambda: stale_snap)
    assert app._level_rpyt(1450) == (1500, 1500, 1450, 1500)


def test_level_rpyt_does_not_cut_throttle_before_first_status():
    import time
    from onboard.app import OnboardApp
    from onboard.config import AppConfig
    from onboard.fc.state import FcState

    cfg = AppConfig()
    app = OnboardApp(cfg)
    now = time.monotonic()
    snap = FcState(rpy_deg=(0.0, -20.0, 0.0), armed=False, status_t=0.0, att_t=now)
    app.state = SimpleNamespace(snapshot=lambda: snap)
    r, p, thr, y = app._level_rpyt(1450)
    assert thr == 1450
    assert p < 1500  # nose-up recovery rate


def test_link_stamp_does_not_fallback_to_ack_last_rx_t():
    from onboard.config import FcConfig
    from onboard.fc.state import FcStateEstimator

    link = SimpleNamespace(last_telem_rx_t=0.0, last_rx_t=123.45)
    est = FcStateEstimator(link, FcConfig())
    assert est._link_stamp(999.0) == 0.0


def test_decode_motor_channels_pads_missing_aux_with_1000():
    from msp import decode_motor_channels

    ch = decode_motor_channels([1500, 1500, 1400, 1500, 1800, 1000, 1000, 1000])
    assert len(ch) == 16
    assert ch[8:] == (1000,) * 8


def test_safety_low_battery_debounce_resets_after_recovery():
    from onboard.config import SafetyConfig
    from onboard.safety import SafetyInputs, SafetySupervisor

    sup = SafetySupervisor(SafetyConfig(min_battery_v=13.6), battery_debounce_s=1.0)

    def mk(t, v):
        return SafetyInputs(
            now=t, armed=True, override_on=True, engage_on=True,
            rc_age_s=0.01, att_age_s=0.01, alt_age_s=0.01, status_age_s=0.01,
            link_age_s=0.01, altitude_m=5.0, range_m=5.0, loop_dt_s=0.02,
            target_confirmed=True, target_age_s=0.05, battery_v=v,
            angle_mode_active=False, expected_att_mode="rate",
            fc_layout_ok=True, analog_age_s=0.01,
        )

    assert sup.evaluate(mk(10.0, 13.5)).reason == "battery_low_pending"
    assert sup.evaluate(mk(10.2, 15.2)).allow_auto is True
    # A second brief sag 5s later must restart debounce, not latch immediately.
    d = sup.evaluate(mk(15.2, 13.5))
    assert d.reason == "battery_low_pending"
    assert sup._latched is None


def test_bbox_tracker_update_candidates_reinitialises_after_coast_window():
    from perception.bbox_tracker import BboxTracker

    tr = BboxTracker(confirm_frames=2, max_coast_s=0.2, gate_px=60.0, gate_frac=0.05)
    tr.update((100.0, 100.0, 140.0, 140.0), dt=0.05, timestamp=0.0)
    out = tr.update((100.0, 100.0, 140.0, 140.0), dt=0.05, timestamp=0.05)
    assert out.confirmed
    # Far candidate exceeds gate during coast window -> must report "gated".
    far = SimpleNamespace(bbox=(500.0, 400.0, 540.0, 440.0), conf=0.9)
    out1 = tr.update_candidates([far], dt=0.1, timestamp=0.15)
    assert out1.reason == "gated"
    # Once coast window expires, the same far candidate immediately re-initialises.
    out2 = tr.update_candidates([far], dt=0.15, timestamp=0.30)
    assert out2.reason == 'det'
    assert out2.bbox is not None and abs(out2.bbox[0] - 500.0) < 1e-6


def test_p_visual_guidance_low_alt_throttle_floor_and_hit_dwell():
    from guidance.p_visual_guidance import PStrikePhase, PVisualGuidance

    g = PVisualGuidance(hover_pwm=1430.0, thr_min_pwm=1180.0, hit_dwell=2, hit_range_m=2.0)
    # Low altitude (2.0 m) at range ~3.5 m: guard_strength ~0.6 -> floor > thr_min_pwm.
    cmd = g.update(
        bbox_xyxy=(590.0, 310.0, 690.0, 410.0),
        own_position_enu=(0.0, 0.0, 2.0),
        own_velocity_enu=(0.0, 0.0, 0.0),
        own_rpy_deg=(0.0, 0.0, 0.0),
        timestamp=1.0, dt=0.02, fresh=True,
    )
    assert cmd.throttle_pwm >= 1180.0

    # 50 Hz extrapolated frames between fresh detections must not reset hit_count.
    g2 = PVisualGuidance(image_size=(1280, 720), hover_pwm=1430.0, hit_dwell=2, hit_range_m=5.0)
    g2.update((540.0, 260.0, 740.0, 460.0), (0.0, 0.0, 5.0), (0.0, 0.0, 0.0),
              (0.0, 0.0, 0.0), timestamp=1.00, dt=0.02, fresh=True)
    c1 = g2.update((520.0, 240.0, 760.0, 480.0), (0.0, 0.0, 5.0), (0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0), timestamp=1.05, dt=0.02, fresh=True)
    assert not c1.hit
    # Intervening extrapolated frame
    g2.update((515.0, 235.0, 765.0, 485.0), (0.0, 0.0, 5.0), (0.0, 0.0, 0.0),
              (0.0, 0.0, 0.0), timestamp=1.07, dt=0.02, fresh=False)
    c2 = g2.update((500.0, 220.0, 780.0, 500.0), (0.0, 0.0, 5.0), (0.0, 0.0, 0.0),
                   (0.0, 0.0, 0.0), timestamp=1.10, dt=0.02, fresh=True)
    assert c2.hit and c2.phase == PStrikePhase.HIT


def test_visual_strike_guidance_closing_speed_sign():
    import numpy as np
    from guidance.visual_strike_guidance import InterceptTrajectory

    traj = InterceptTrajectory()
    st = traj.update(
        target_position_enu=np.array([0.0, 12.0, 5.0]),
        target_velocity_enu=np.array([0.0, 0.0, 0.0]),
        own_position_enu=np.array([0.0, 0.0, 5.0]),
        own_velocity_enu=np.array([0.0, 6.0, 0.0]),
    )
    assert st.closing_speed_mps > 5.5


def test_fc_identify_print_human_does_not_raise(capsys):
    from scripts.fc_identify import print_human

    print_human({
        "port": "/dev/ttyACM0",
        "baud": 115200,
        "firmware": {
            "api_version": {"protocol": 0, "major": 1, "minor": 46},
            "variant": "BTFL",
            "fc_version": {"year": 4, "month": 5, "patch": 3, "version_string": "4.5.3"},
            "board": {"board_identifier": "S7X2", "board_name": "TEST", "hardware_revision": 0},
        },
        "aux": {
            "boxids": [0, 1],
            "ranges": [],
            "arm_channel": 4,
            "angle_channel": 5,
            "override_channel": 7,
            "warnings": [],
        },
        "config_snippet": {},
    })
    assert "4.5.3" in capsys.readouterr().out


def test_level_rpyt_does_not_cut_throttle_on_stale_disarmed_status():
    import time
    from onboard.app import OnboardApp
    from onboard.config import AppConfig
    from onboard.fc.state import FcState

    app = OnboardApp(AppConfig())
    now = time.monotonic()
    snap = FcState(
        rpy_deg=(0.0, -20.0, 0.0),
        armed=False,
        status_t=now - 5.0,   # frozen disarmed packet
        att_t=now,             # attitude is still live
    )
    app.state = SimpleNamespace(snapshot=lambda: snap)
    _r, p, thr, _y = app._level_rpyt(1450)
    assert thr == 1450
    assert p < 1500


def test_coast_clears_altitude_integrator():
    from guidance.p_visual_guidance import PStrikePhase, PVisualGuidance

    g = PVisualGuidance()
    g._z_int = 5.0
    cmd = g.update(
        None, (0.0, 0.0, 5.0), (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0), timestamp=10.0, dt=0.02, fresh=False,
    )
    assert cmd.phase == PStrikePhase.COAST
    assert g._z_int == 0.0


def test_hailo_configure_negotiates_auto_input_format(monkeypatch, tmp_path):
    import sys
    import types

    state = {"negotiated": None, "released": False}

    class Input:
        name = "input"
        shape = (8, 8, 3)
        is_nms = False

        def __init__(self):
            self.format = SimpleNamespace(type="AUTO")

        def set_format_type(self, fmt):
            self.format = SimpleNamespace(type=fmt)
            state["negotiated"] = fmt

    class Output:
        name = "out0"
        shape = (1, 1, 1)
        is_nms = True
        format = SimpleNamespace(type="FLOAT32")

    class Bindings:
        def input(self):
            return SimpleNamespace(set_buffer=lambda buf: None)

        def output(self, name):
            return SimpleNamespace(set_buffer=lambda buf: None)

    class Configured:
        def activate(self):
            pass

        def create_bindings(self):
            return Bindings()

    class InferModel:
        def __init__(self):
            self._input = Input()
            self._output = Output()

        def set_batch_size(self, n):
            pass

        def input(self):
            return self._input

        @property
        def outputs(self):
            return [self._output]

        def configure(self):
            return Configured()

    class VDevice:
        def create_infer_model(self, hef):
            return InferModel()

        def release(self):
            state["released"] = True

    fake = types.ModuleType("hailo_platform")
    fake.FormatType = SimpleNamespace(
        UINT8="UINT8", FLOAT32="FLOAT32", AUTO="AUTO",
    )
    fake.VDevice = VDevice
    monkeypatch.setitem(sys.modules, "hailo_platform", fake)

    hef = tmp_path / "model.hef"
    hef.write_bytes(b"fake")
    from onboard.config import DetectorConfig
    from onboard.detector.hailo import HailoDetector

    det = HailoDetector(DetectorConfig(hef=str(hef)))
    det._configure()
    assert state["negotiated"] == "UINT8"
    det._release()
    assert state["released"] is True


def test_run_returns_nonzero_if_worker_crashed_before_loop():
    from onboard.app import OnboardApp
    from onboard.config import AppConfig

    app = OnboardApp(AppConfig())
    app._start_with_retry = lambda: None
    app.request_stop("thread_exception")
    assert app.run() == 1


def test_rc_mux_separate_override_switch_anchors_throttle_to_safe_hover():
    from rc_mux import RCMux

    class Msp:
        def send_rc_16(self, channels):
            self.last = tuple(channels)
            return True

    msp = Msp()
    mux = RCMux(
        msp,
        mode_channel=7,
        arm_channel=4,
        base_throttle=1430,
        override_channel=6,
        override_range=(1300, 2100),
        auto_engage=True,
    )
    mux._safe_hover_thr = 1430
    # Mode=low first (C7), then mid.  ARM and OVERRIDE are high; ch[2] is the
    # MSP echo (1500) and must not become the engaged throttle.
    low = [1500, 1500, 1500, 1500, 1800, 1000, 1900, 1000]
    mid = [1500, 1500, 1500, 1500, 1800, 1000, 1900, 1500]
    mux._update_mode(low)
    mux.request_auto(True)
    mux._update_mode(mid)
    assert mux._auto is True
    assert mux.engaged_throttle == 1430


def test_target_kf_rejects_nonfinite_measurement_without_poisoning_state():
    from perception.target_kf import TargetKF

    kf = TargetKF()
    assert kf.update_position(np.array([0.0, 0.0, 5.0])) is True
    before = kf.x.copy()
    assert kf.update_position(np.array([np.nan, 1.0, 6.0])) is False
    assert np.all(np.isfinite(kf.x))
    assert np.allclose(kf.x, before)


def test_visual_strike_pn_accel_points_toward_crossing_target():
    from guidance.visual_strike_guidance import InterceptTrajectory

    # Target is due north, crossing east.  A positive-x acceleration is the
    # collision-course correction; the old relative-velocity sign inverted it.
    st = InterceptTrajectory().update(
        target_position_enu=np.array([0.0, 20.0, 5.0]),
        target_velocity_enu=np.array([3.0, 0.0, 0.0]),
        own_position_enu=np.array([0.0, 0.0, 5.0]),
        own_velocity_enu=np.array([0.0, 6.0, 0.0]),
    )
    assert st.closing_speed_mps > 5.0
    assert st.accel_pn_enu[0] > 0.0


def test_msp_missing_throttle_defaults_to_idle():
    from msp import decode_motor_channels

    assert decode_motor_channels([])[:4] == (1500, 1500, 1000, 1500)
    assert decode_motor_channels([1500, 1500])[:4] == (1500, 1500, 1000, 1500)
