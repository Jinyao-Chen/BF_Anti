"""Onboard application: wires FC, camera, detector, controller, mux, video.

Every control tick is also recorded by the blackbox (onboard/telemetry_log.py):
flight-controller state, pilot RC, detections, safety gate and control command.
Startup is retried instead of exiting, so a FC/fb0 that is not enumerated yet
cannot leave a boot with no log at all.

This module is deliberately import-light. Hardware backends are imported in
_ensure_built() so config/controller/safety can be unit-tested on a laptop with
no camera, no FC, and no Hailo runtime.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

from onboard.config import AppConfig, load_config
from onboard.controller import Controller
from onboard.logging_setup import (
    attach_file_handler,
    install_excepthooks,
    install_faulthandler,
    set_thread_error_hook,
    setup_console_logging,
    shutdown_faulthandler,
)
from onboard.power import ShutdownGuard, request_poweroff
from onboard.safety import SafetyDecision, SafetyInputs, SafetySupervisor
from onboard.telemetry_log import TelemetryLog, collect_run_meta, safe_float
from onboard.types import ControlOutput, Detection, FcState

LOG = logging.getLogger("bf_anti.onboard")


class OnboardApp:
    # Seconds after the first complete FC sample during which stale fields only
    # deny (they do not latch a critical fault the pilot must acknowledge).
    STARTUP_GRACE_S = 2.0

    def __init__(
        self,
        cfg: AppConfig,
        link=None,
        state=None,
        detector=None,
        camera=None,
        mux=None,
        video=None,
        telemetry: Optional[TelemetryLog] = None,
        startup_retry_s: float = 2.0,
        poweroff=None,
    ) -> None:
        self.cfg = cfg
        self.link = link
        self.state = state
        self.detector = detector
        self.camera = camera
        self.mux = mux
        self.video = video
        self.telemetry = telemetry
        # RC-controlled graceful shutdown (CH7 by default).  The guard is pure;
        # poweroff() is injected so tests never halt the host.
        self.shutdown_guard = ShutdownGuard(cfg.power)
        self._poweroff = poweroff or request_poweroff
        self._poweroff_requested = False
        self._poweroff_value = 0
        # Retry cadence for components that fail at boot (fb0/FC not ready).
        self.startup_retry_s = max(0.05, float(startup_retry_s))
        self._start_failures: dict[str, float] = {}
        self._init_failures: dict[str, float] = {}
        self._retry_lock = threading.Lock()
        self._retry_thread: Optional[threading.Thread] = None
        self._poweroff_on_exit = False
        self._last_reason_event_t = 0.0
        self._last_tick_error_event_t = 0.0
        self._fc_seen_at: Optional[float] = None
        self._events_seen: dict[str, object] = {
            "armed": None, "allow": None, "reason": None,
            "det": None, "mux": None, "engaged": None,
        }

        self.controller = Controller(cfg)
        self.safety = SafetySupervisor(
            cfg.safety,
            require_override=cfg.fc.require_override,
            require_engage=cfg.fc.require_engage,
            battery_debounce_s=cfg.safety.battery_debounce_s,
            require_layout=True,
        )
        self._engaged = False
        self._reset_lock = threading.Lock()
        self._pending_reset: Optional[str] = None
        self._pending_hover_pwm: Optional[int] = None
        self._stop = threading.Event()
        self._stop_reason = "stop"
        self._stop_logged = False
        self._cleanup_started = False
        self._last_step: Optional[float] = None
        self._last_dt: float = 0.0
        self._built = False
        self._t_start = time.monotonic()
        self.last_output: ControlOutput = ControlOutput()
        self.last_decision: Optional[SafetyDecision] = None

    # -- dependency construction -----------------------------------------
    def _ensure_built(self) -> None:
        if self._built:
            return
        if self.link is None:
            from onboard.fc.link import make_link
            self.link = make_link(self.cfg.fc)
        if self.state is None:
            from onboard.fc.state import FcStateEstimator
            self.state = FcStateEstimator(self.link, self.cfg.fc)
        if self.camera is None and self.cfg.detector.backend != "replay":
            try:
                from onboard.camera import CameraSource
                self.camera = CameraSource(self.cfg.camera)
            except Exception as exc:
                # The camera thread retries opens internally; a constructor
                # failure means bad config, not a missing device.  Keep the
                # runtime alive so the FC path still logs.
                LOG.exception("camera init failed; scheduling background retry")
                self._event("component_init_failed", component="camera", error=repr(exc))
                self.camera = None
                self._init_failures["camera"] = time.monotonic()
        if self.detector is None:
            try:
                from onboard.detector import build_detector
                self.detector = build_detector(self.cfg.detector, source=self.camera)
            except Exception as exc:
                # Never let a missing HEF/HailoRT hide all FC/RC telemetry.
                from onboard.detector import NullDetector
                LOG.exception("detector init failed; falling back to null detector and scheduling retry")
                self._event("component_init_failed", component="detector", error=repr(exc))
                self.detector = NullDetector(self.cfg.detector)
                self._init_failures["detector"] = time.monotonic()
        elif self.camera is not None and hasattr(self.detector, "attach_source"):
            self.detector.attach_source(self.camera)
        if self.video is None and self.cfg.video.enabled:
            try:
                from onboard.video.pipeline import DisplayPipeline
                self.video = DisplayPipeline(self.cfg.video)
            except Exception as exc:
                LOG.exception("video init failed; scheduling background retry")
                self._event("component_init_failed", component="video", error=repr(exc))
                self.video = None
                self._init_failures["video"] = time.monotonic()
        if self.mux is None:
            from rc_mux import RCMux
            self.mux = RCMux(
                self.link,
                mode_channel=self.cfg.fc.aux_engage,
                arm_channel=self.cfg.fc.aux_arm,
                stick_takeover=self.cfg.fc.stick_takeover,
                base_throttle=self.cfg.control.hover_throttle,
                crsf_timeout=self.cfg.fc.link_timeout_s,
                cmd_timeout=self.cfg.safety.target_lost_s,
                rate_hz=self.cfg.fc.rc_hz,
                channel_source=self.state.rc_snapshot,
                rc_stale_timeout=self.cfg.fc.link_timeout_s,
            )
            self.mux.on_engage = self._on_engage
            self.mux.on_disengage = self._on_disengage
            self.mux.level_rpyt_fn = self._level_rpyt
        else:
            if hasattr(self.mux, "on_engage"):
                self.mux.on_engage = self._on_engage
            if hasattr(self.mux, "on_disengage"):
                self.mux.on_disengage = self._on_disengage
            if hasattr(self.mux, "level_rpyt_fn"):
                self.mux.level_rpyt_fn = self._level_rpyt
        self._built = True

    def _level_rpyt(self, safe_throttle: int) -> tuple[int, int, int, int]:
        """Level-attitude RPYT for ACRO when override is still active.

        Called from the mux thread.  In rate mode, (1500,1500) means zero
        angular rate and holds a dive; here we close a P loop on live roll/
        pitch to bring the body level while retaining safe hover throttle.
        """
        c = self.cfg.control
        if str(getattr(c, "att_mode", "rate")).lower() != "rate":
            # ANGLE self-levels on mid sticks.
            return (1500, 1500, int(safe_throttle), 1500)
        try:
            snap = self.state.snapshot() if self.state is not None else None
            rpy = getattr(snap, "rpy_deg", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
            if not bool(getattr(snap, "armed", False)):
                return (1500, 1500, 1000, 1500)
            k_att = float(c.att_rate_gain)
            rate_max = float(c.rate_max_dps)
            scale = 500.0 / max(1e-6, float(c.rate_full_dps))
            p_dps = max(-rate_max, min(rate_max, -k_att * float(rpy[0])))
            q_dps = max(-rate_max, min(rate_max, -k_att * float(rpy[1])))
            roll = int(round(1500.0 + float(c.roll_sign) * p_dps * scale))
            pitch = int(round(1500.0 - float(c.pitch_sign) * q_dps * scale))
            return (
                max(1000, min(2000, roll)),
                max(1000, min(2000, pitch)),
                int(safe_throttle),
                1500,
            )
        except Exception:
            return (1500, 1500, int(safe_throttle), 1500)

    def _on_engage(self, engaged_throttle: int) -> None:
        # Called from the mux thread.  Do not touch the guidance/tracker state
        # here: defer the reset to the control tick so the 50 Hz loop never
        # sees a half-reset guidance object.
        with self._reset_lock:
            self._engaged = True
            self._pending_reset = "engage"
            self._pending_hover_pwm = int(engaged_throttle)
        LOG.info("auto engaged, pilot throttle=%s", engaged_throttle)
        self._event("engaged", engaged=True, throttle=int(engaged_throttle))

    def _on_disengage(self) -> None:
        if self._engaged:
            LOG.info("auto disengaged -> pilot")
        with self._reset_lock:
            self._engaged = False
            self._pending_reset = "disengage"
            self._pending_hover_pwm = None
        self._event("engaged", engaged=False)

    # -- telemetry ---------------------------------------------------------
    def _event(self, name: str, **fields) -> None:
        """Best-effort blackbox event; logging must never break the loop."""
        tlog = self.telemetry
        if tlog is None:
            return
        try:
            tlog.event(name, **fields)
        except Exception:
            LOG.exception("telemetry event failed")

    @staticmethod
    def _age(stamp: float, now: float):
        try:
            stamp = float(stamp)
        except (TypeError, ValueError):
            return ""
        if stamp <= 0.0:
            return ""
        return round(max(0.0, float(now) - stamp), 3)

    def _log_telemetry(self, now, state, detection, out, decision) -> None:
        tlog = self.telemetry
        if tlog is None or not tlog.enabled:
            return
        try:
            self._emit_transitions(now, state, detection, decision)
            if tlog.due(now):
                tlog.sample(now, self._telemetry_fields(now, state, detection, out, decision))
        except Exception:
            LOG.exception("telemetry tick failed")

    def _emit_transitions(self, now, state, detection, decision) -> None:
        seen = self._events_seen
        status_ok = float(state.status_t) > 0.0
        if status_ok:
            armed = bool(state.armed)
            if seen["armed"] != armed:
                seen["armed"] = armed
                self._event("armed", armed=armed)
                LOG.info("FC armed=%s", armed)
        allow = bool(decision.allow_auto)
        if seen["allow"] != allow:
            seen["allow"] = allow
            seen["reason"] = decision.reason
            self._event(
                "safety_gate",
                allow=allow,
                reason=decision.reason,
                critical=bool(decision.critical),
            )
            LOG.info("safety gate %s: %s", "OPEN" if allow else "CLOSED", decision.reason)
        elif seen["reason"] != decision.reason:
            seen["reason"] = decision.reason
            # A 25 fps detector can flap between target_stale/unconfirmed;
            # cap the events file at 2 Hz per reason change.
            if now - self._last_reason_event_t >= 0.5:
                self._last_reason_event_t = now
                self._event(
                    "safety_reason",
                    reason=decision.reason,
                    critical=bool(decision.critical),
                )
        detected = detection is not None
        if seen["det"] != detected:
            seen["det"] = detected
            if detected:
                bbox = detection.bbox
                self._event(
                    "target_acquired",
                    conf=safe_float(detection.conf, 3),
                    class_id=int(detection.class_id),
                    age_s=self._age(detection.t, now),
                    x1=safe_float(bbox[0], 1),
                    y1=safe_float(bbox[1], 1),
                    x2=safe_float(bbox[2], 1),
                    y2=safe_float(bbox[3], 1),
                )
                LOG.info("target acquired conf=%.2f", float(detection.conf))
            else:
                self._event("target_lost")
                LOG.info("target lost")
        mux_mode = str(getattr(self.mux, "mode", "") or "")
        if seen["mux"] != mux_mode:
            seen["mux"] = mux_mode
            self._event("mux_mode", mode=mux_mode)

    def _telemetry_fields(self, now, state, detection, out, decision) -> dict:
        """Map one control tick to the CSV schema in onboard.telemetry_log."""
        row: dict = {}
        row["uptime_s"] = round(max(0.0, time.monotonic() - self._t_start), 3)
        status_ok = float(state.status_t) > 0.0
        row["armed"] = int(bool(state.armed)) if status_ok else ""
        row["fc_mode"] = int(state.flight_mode_flags) if status_ok else ""
        rpy = tuple(state.rpy_deg or (0.0, 0.0, 0.0))
        if float(state.att_t) > 0.0:
            row["roll_deg"] = safe_float(rpy[0])
            row["pitch_deg"] = safe_float(rpy[1])
            row["yaw_deg"] = safe_float(rpy[2])
        else:
            row["roll_deg"] = row["pitch_deg"] = row["yaw_deg"] = ""
        if float(state.alt_t) > 0.0:
            row["alt_m"] = safe_float(state.altitude_m)
            row["vario_mps"] = safe_float(state.vario_mps)
        else:
            row["alt_m"] = row["vario_mps"] = ""
        row["att_age_s"] = self._age(state.att_t, now)
        row["alt_age_s"] = self._age(state.alt_t, now)
        row["status_age_s"] = self._age(state.status_t, now)
        row["link_age_s"] = self._age(state.link_t, now)
        if float(state.analog_t) > 0.0:
            row["voltage_v"] = safe_float(state.voltage_v)
            row["current_a"] = safe_float(state.current_a, 1)
            row["mah"] = int(state.mah)
            row["rssi"] = int(state.rssi)
        else:
            row["voltage_v"] = row["current_a"] = row["mah"] = row["rssi"] = ""
        rc = tuple(state.rc or ())
        rc_ok = float(state.rc_t) > 0.0 and bool(rc)
        for index in range(16):
            row[f"rc{index}"] = int(rc[index]) if rc_ok and index < len(rc) else ""
        row["rc_age_s"] = self._age(state.rc_t, now)
        if detection is not None:
            bbox = detection.bbox
            row["det_present"] = 1
            row["det_conf"] = safe_float(detection.conf, 3)
            row["det_class"] = int(detection.class_id)
            row["det_age_s"] = self._age(detection.t, now)
            row["det_x1"] = safe_float(bbox[0], 1)
            row["det_y1"] = safe_float(bbox[1], 1)
            row["det_x2"] = safe_float(bbox[2], 1)
            row["det_y2"] = safe_float(bbox[3], 1)
        else:
            row["det_present"] = 0
        stats = None
        if hasattr(self.detector, "stats"):
            try:
                stats = self.detector.stats()
            except Exception:
                stats = None
        if stats is not None:
            row["det_fps"] = safe_float(getattr(stats, "fps", 0.0), 1)
            row["det_infer_ms"] = safe_float(getattr(stats, "infer_ms", 0.0), 2)
            row["det_errors"] = int(getattr(stats, "errors", 0) or 0)
        tracker = getattr(self.controller, "tracker", None)
        if tracker is not None:
            row["track_locked"] = int(bool(getattr(tracker, "locked", False)))
            row["track_streak"] = int(getattr(tracker, "streak", 0) or 0)
            row["track_miss"] = int(getattr(tracker, "misses", 0) or 0)
        row["cmd_valid"] = int(out.channels is not None)
        row["cmd_mode"] = str(getattr(out.mode, "value", out.mode))
        row["cmd_reason"] = str(out.reason)
        if out.channels is not None:
            roll, pitch, throttle, yaw = out.channels
            row["cmd_roll"] = int(roll)
            row["cmd_pitch"] = int(pitch)
            row["cmd_throttle"] = int(throttle)
            row["cmd_yaw"] = int(yaw)
        command = out.command
        row["cmd_range_m"] = (
            safe_float(getattr(command, "range_est_m", None)) if command is not None else ""
        )
        row["cmd_confirmed"] = int(bool(out.confirmed))
        row["allow_auto"] = int(bool(decision.allow_auto))
        row["safety_reason"] = str(decision.reason)
        row["safety_critical"] = int(bool(decision.critical))
        row["mux_mode"] = str(getattr(self.mux, "mode", "") or "")
        if callable(getattr(self.mux, "mode_position", None)):
            try:
                row["mux_pos"] = int(self.mux.mode_position())
            except Exception:
                row["mux_pos"] = ""
        row["engaged"] = int(bool(self._engaged))
        row["loop_dt_s"] = safe_float(self._last_dt, 4)
        row["link_errors"] = int(getattr(state, "errors", 0) or 0)
        return row

    # -- one control tick -------------------------------------------------
    def _safety_inputs(
        self,
        state: FcState,
        detection: Optional[Detection],
        out: ControlOutput,
        now: float,
    ) -> SafetyInputs:
        rc = tuple(state.rc) if state.rc else (1500,) * 16
        def aux(idx: int) -> int:
            return int(rc[idx]) if 0 <= idx < len(rc) else 1500
        target_age = (now - detection.t) if detection is not None and detection.t > 0.0 else float("inf")
        rng = float("inf")
        if out.command is not None:
            try:
                rng = float(out.command.range_est_m)
            except Exception:
                rng = float("inf")
        detector_ok = True
        if self.detector is not None and hasattr(self.detector, "healthy"):
            try:
                detector_ok = bool(self.detector.healthy(now))
            except Exception:
                detector_ok = False
        # FC data has been received at least once for every signal the safety
        # gate needs; before that, deny without latching (boot warm-up).
        fc_seen = all(
            float(stamp) > 0.0
            for stamp in (state.link_t, state.att_t, state.alt_t, state.rc_t, state.status_t)
        )
        if fc_seen and self._fc_seen_at is None:
            self._fc_seen_at = now
        warmup_grace = (
            not self._engaged
            and self._fc_seen_at is not None
            and (now - self._fc_seen_at) < self.STARTUP_GRACE_S
        )
        mode_pwm = aux(self.cfg.fc.aux_engage)
        if self.cfg.fc.override_from_mode:
            # The same 3-pos switch drives BOXMSPOVERRIDE on the FC: low is a
            # physical CRSF bypass, mid/high let the companion write RPYT.
            override_on = mode_pwm >= 1300
        else:
            override_on = aux(self.cfg.fc.aux_override) > 1700
        disc = None
        try:
            disc = getattr(self.state, "aux_discovery", None)
        except Exception:
            disc = None
        if disc is not None and getattr(disc, "arm_range", None) is not None:
            self.mux.arm_range = tuple(disc.arm_range)
        layout_ok, angle_active = self._fc_layout_state(disc, state)
        coast_max_s = min(
            float(getattr(self.controller.guidance, "lost_timeout_s", 0.0) or 0.0),
            float(self.cfg.safety.terminal_coast_s),
        )
        last_range = float("inf")
        if out.command is not None:
            try:
                last_range = float(out.command.range_est_m)
            except Exception:
                last_range = float("inf")
        terminal_range = float(getattr(self.controller.guidance, "terminal_range_m", 3.0))
        is_terminal_coast = bool(
            out.confirmed and str(getattr(out, "reason", "")) == "coast"
            and out.channels is not None
            and last_range == last_range
            and last_range <= max(2.0 * terminal_range, terminal_range + 1.0)
        )
        return SafetyInputs(
            now=now,
            armed=bool(state.armed),
            override_on=override_on,
            # Three-position mode: mid (AUTO) and high (STRIKE) both authorize
            # the companion; low (FORCE MANUAL) never does.
            engage_on=mode_pwm >= 1300,
            rc_age_s=state.rc_age_s(now),
            att_age_s=state.att_age_s(now),
            alt_age_s=state.alt_age_s(now),
            status_age_s=state.status_age_s(now),
            link_age_s=state.link_age_s(now),
            altitude_m=float(state.altitude_m),
            range_m=rng,
            loop_dt_s=float(self._last_dt),
            target_confirmed=bool(out.confirmed),
            target_age_s=target_age,
            battery_v=float(state.voltage_v),
            detector_ok=detector_ok,
            estop=self._stop.is_set(),
            fc_seen=fc_seen,
            warmup_grace=warmup_grace,
            # NEVER use bit 0: in Betaflight MSP_STATUS, bit 0 is BOXARM.
            angle_mode_active=angle_active,
            expected_att_mode=str(self.cfg.control.att_mode).lower(),
            fc_layout_ok=layout_ok,
            analog_age_s=(
                (now - float(state.analog_t))
                if float(getattr(state, "analog_t", 0.0) or 0.0) > 0.0
                else float("inf")
            ),
            coasting=is_terminal_coast,
            track_lost_age_s=float(
                self.controller.last_seen_age_s(now)
                if hasattr(self.controller, "last_seen_age_s")
                else target_age
            ),
            coast_max_s=coast_max_s,
        )

    def _fc_layout_state(self, disc, state) -> tuple[bool, Optional[bool]]:
        """Return (layout_verified, ANGLE_active) without ever guessing bit 0."""
        from onboard.fc.mode import BOX_PERMANENT_ID_ANGLE
        expected = str(self.cfg.control.att_mode).lower()
        if disc is None:
            return False, None
        flags = int(getattr(state, "flight_mode_flags", 0) or 0)
        ext = bytes(getattr(state, "extended_flight_mode_flags", b"") or b"")
        angle_active = disc.is_mode_active(flags, BOX_PERMANENT_ID_ANGLE, ext)
        if angle_active is None and expected == "rate" and disc.bit_index_for(BOX_PERMANENT_ID_ANGLE) is None:
            # No accelerometer/ANGLE box at all: the FC cannot be in ANGLE.
            angle_active = False

        rx_ok = disc.rx_map_is_aetr() is True
        arm_ok = (
            disc.arm_channel is not None
            and int(disc.arm_channel) == int(self.cfg.fc.aux_arm)
            and disc.arm_range is not None
        )
        if self.cfg.fc.override_from_mode:
            override_ok = (
                disc.override_channel is not None
                and int(disc.override_channel) == int(self.cfg.fc.aux_engage)
                and disc.override_range is not None
                and int(disc.override_range[0]) <= 1300
                and int(disc.override_range[1]) >= 1700
            )
        else:
            override_ok = (
                disc.override_channel is not None
                and int(disc.override_channel) == int(self.cfg.fc.aux_override)
            )
        if expected == "angle":
            angle_ok = (
                disc.angle_channel is not None
                and int(disc.angle_channel) == int(self.cfg.fc.aux_angle)
            )
        else:
            angle_ok = True
        return bool(rx_ok and arm_ok and override_ok and angle_ok), angle_active

    def step_once(self, now: Optional[float] = None) -> tuple[ControlOutput, SafetyDecision]:
        self._ensure_built()
        now = time.monotonic() if now is None else float(now)
        with self._reset_lock:
            pending_reset = self._pending_reset
            pending_hover = self._pending_hover_pwm
            self._pending_reset = None
            self._pending_hover_pwm = None
        if pending_reset is not None:
            if pending_reset == "engage":
                # Keep the tracker lock, clear guidance integrators/history,
                # and adapt hover_pwm to the pilot's real hover throttle.
                self.controller.guidance.reset()
                if pending_hover is not None and 1250 <= int(pending_hover) <= 1700:
                    self.controller.guidance.hover_pwm = float(pending_hover)
                self.safety.reset()
            else:
                self.controller.reset()
                self.safety.reset()
        period = 1.0 / max(1.0, float(self.cfg.control.rate_hz))
        self._last_dt = period if self._last_step is None else (now - self._last_step)
        self._last_step = now

        state = self.state.snapshot(now)

        # RC shutdown is independent of the flight-control gate.  It is allowed
        # only when the FC is disarmed and the pilot has deliberately held the
        # channel high; see onboard/power.py.
        if self.shutdown_guard.update(now, state):
            self._poweroff_requested = True
            channels = tuple(state.rc or ())
            idx = int(self.cfg.power.aux_channel)
            self._poweroff_value = int(channels[idx]) if 0 <= idx < len(channels) else 0
            LOG.warning(
                "RC shutdown channel rc%d=%d held for %.1f s; requesting poweroff",
                idx, self._poweroff_value, float(self.cfg.power.hold_s),
            )
            self._event(
                "poweroff_requested",
                channel=idx,
                value=self._poweroff_value,
                hold_s=float(self.cfg.power.hold_s),
            )

        detection = self.detector.latest(self.cfg.safety.max_det_age_s, now)
        candidates = None
        if detection is not None and hasattr(self.detector, "latest_detections"):
            try:
                candidates = self.detector.latest_detections(
                    self.cfg.safety.max_det_age_s, now
                )
            except Exception:
                candidates = None
        out = self.controller.step(
            detection, state, now, self._last_dt,
            state_estimator=self.state, candidates=candidates,
        )
        decision = self.safety.evaluate(self._safety_inputs(state, detection, out, now))

        # The mux owns engagement; the app only publishes a fresh command
        # when the safety gate is green.  This removes the old circular
        # dependency where the mux waited for a command that the app only
        # produced after the mux had already engaged.
        if hasattr(self.mux, "request_auto"):
            self.mux.request_auto(bool(decision.allow_auto))
        if decision.allow_auto and out.channels is not None:
            from rc_mux import TrackCmd
            roll, pitch, throttle, yaw = out.channels
            self.mux.publish_cmd(TrackCmd(
                roll=int(roll), pitch=int(pitch), throttle=int(throttle),
                yaw=int(yaw), t=now,
            ))
        else:
            # None makes the mux fall back to the pilot immediately.
            self.mux.publish_cmd(None)

        # Record the tick before touching the framebuffer: a slow fb0 write
        # must not delay the blackbox, and this row is the evidence that the
        # control loop is alive.
        self._log_telemetry(now, state, detection, out, decision)

        if self.video is not None:
            frame = None
            frame_t = 0.0
            video_det = detection
            # Prefer the detector's frame+box pair for the OSD without an
            # unnecessary camera.latest() lock when a paired frame is ready.
            used_pair = False
            if hasattr(self.detector, "latest_pair"):
                pair_frame, pair_det = self.detector.latest_pair(0.35, now)
                if pair_frame is not None:
                    frame = pair_frame
                    # The pair is the frame that was just inferred, even when
                    # it contains no detection.  Use its own capture stamp;
                    # falling back to "now" would hide pipeline latency.
                    pair_stamp = float(
                        getattr(self.detector, "last_pair_t", 0.0) or 0.0
                    )
                    frame_t = pair_stamp if pair_stamp > 0.0 else now
                    video_det = pair_det
                    used_pair = True
                else:
                    video_det = None
            if not used_pair and self.camera is not None:
                frame, frame_t = self.camera.latest(self.cfg.safety.max_det_age_s, now)
            try:
                self.video.submit(
                    frame, video_det, out, state, decision, now, frame_t
                )
            except Exception:
                LOG.exception("video submit failed")
        self.last_output = out
        self.last_decision = decision
        return out, decision

    # -- lifecycle --------------------------------------------------------
    def _start_component(self, name: str) -> None:
        obj = getattr(self, name, None)
        if obj is None:
            return
        try:
            obj.start()
        except Exception as exc:
            LOG.exception("start %s failed", name)
            self._start_failures[name] = time.monotonic()
            self._event("component_start_failed", component=name, error=repr(exc))
            if name in ("state", "mux"):
                # Without the FC reader or the RC writer there is no runtime.
                raise
        else:
            self._start_failures.pop(name, None)

    def start(self) -> None:
        """Build and start every component; raises only for critical ones."""
        self._ensure_built()
        # Start the RC path before camera/Hailo/video: the FC must see pilot
        # passthrough frames as early as possible, especially when the pilot
        # boots with CH8 already out of FORCE MANUAL.
        for name in ("state", "mux", "camera", "detector", "video"):
            self._start_component(name)

    def _start_with_retry(self) -> None:
        """Keep the service alive while the FC/fb0/camera enumerate at boot.

        systemd's StartLimitBurst used to stop the unit after five fast
        failures, which is exactly the "no logs after boot" symptom.  Retrying
        here keeps the process - and the blackbox - running, and records the
        failure reason in events.jsonl.
        """
        attempt = 0
        delay = self.startup_retry_s
        while not self._stop.is_set():
            attempt += 1
            try:
                self.start()
                return
            except Exception as exc:
                LOG.exception(
                    "onboard startup failed (attempt %d); retrying in %.1f s",
                    attempt, delay,
                )
                self._event(
                    "startup_retry",
                    attempt=attempt,
                    error=repr(exc),
                    retry_in_s=round(delay, 2),
                )
                if self._stop.wait(delay):
                    return
                delay = min(10.0, delay * 1.5)

    def _retry_components_worker(self) -> None:
        """Retry failed components off the 50 Hz loop.

        CameraSource.start() waits up to open_wait_s (1 s), Hailo configure can
        take seconds, and fb0 open can block.  Doing any of that in the control
        tick trips loop_overrun and latches FAULT.
        """
        while not self._stop.wait(1.0):
            now = time.monotonic()
            with self._retry_lock:
                init_items = list(self._init_failures.items())
                start_items = list(self._start_failures.items())
            for name, since in init_items:
                if now - since < 5.0:
                    continue
                try:
                    if name == "camera":
                        from onboard.camera import CameraSource
                        obj = CameraSource(self.cfg.camera)
                        obj.start()
                        self.camera = obj
                        if self.detector is not None and hasattr(self.detector, "attach_source"):
                            self.detector.attach_source(obj)
                    elif name == "detector":
                        from onboard.detector import build_detector
                        obj = build_detector(self.cfg.detector, source=self.camera)
                        obj.start()
                        self.detector = obj
                    elif name == "video":
                        from onboard.video.pipeline import DisplayPipeline
                        obj = DisplayPipeline(self.cfg.video)
                        obj.start()
                        self.video = obj
                    else:
                        with self._retry_lock:
                            self._init_failures.pop(name, None)
                        continue
                except Exception:
                    with self._retry_lock:
                        self._init_failures[name] = time.monotonic()
                    LOG.exception("retry init/start %s failed", name)
                else:
                    if self._stop.is_set():
                        try:
                            obj.stop()
                        except Exception:
                            pass
                        continue
                    with self._retry_lock:
                        self._init_failures.pop(name, None)
                    LOG.info("%s initialized after retry", name)
                    self._event("component_started", component=name)
            for name, since in start_items:
                if now - since < 5.0 or name in ("state", "mux"):
                    continue
                obj = getattr(self, name, None)
                if obj is None:
                    with self._retry_lock:
                        self._start_failures.pop(name, None)
                    continue
                try:
                    obj.start()
                except Exception:
                    with self._retry_lock:
                        self._start_failures[name] = time.monotonic()
                    LOG.exception("retry start %s failed", name)
                else:
                    if self._stop.is_set():
                        try:
                            obj.stop()
                        except Exception:
                            pass
                        continue
                    with self._retry_lock:
                        self._start_failures.pop(name, None)
                    LOG.info("%s started after retry", name)
                    self._event("component_started", component=name)

    def _retry_components(self, now: float) -> None:
        """Non-blocking hook: ensure the background retry worker is running."""
        with self._retry_lock:
            pending = bool(self._start_failures or self._init_failures)
            alive = self._retry_thread is not None and self._retry_thread.is_alive()
            if pending and not alive:
                self._retry_thread = threading.Thread(
                    target=self._retry_components_worker,
                    name="bf-anti-retry",
                    daemon=True,
                )
                self._retry_thread.start()

    def run(self) -> int:
        period = 1.0 / max(1.0, float(self.cfg.control.rate_hz))
        stop_reason = "stop"
        try:
            self._start_with_retry()
            if self._stop.is_set():
                return 0
            LOG.info(
                "onboard loop started at %.1f Hz (blackbox=%s)",
                self.cfg.control.rate_hz,
                "on" if (self.telemetry is not None and self.telemetry.enabled) else "off",
            )
            self._event(
                "control_loop_started",
                rate_hz=float(self.cfg.control.rate_hz),
                components=sorted(
                    name for name in ("state", "mux", "camera", "detector", "video")
                    if getattr(self, name, None) is not None
                ),
                start_failures=sorted(self._start_failures),
            )
            next_t = time.monotonic()
            while not self._stop.is_set():
                now = time.monotonic()
                self._retry_components(now)
                try:
                    out, decision = self.step_once(now)
                    if not decision.allow_auto and self._engaged:
                        LOG.debug("auto gate closed: %s", decision.reason)
                except Exception as exc:
                    LOG.exception("control tick failed")
                    if self.mux is not None:
                        try:
                            if hasattr(self.mux, "request_auto"):
                                self.mux.request_auto(False)
                            self.mux.publish_cmd(None)
                        except Exception:
                            pass
                    # Telemetry rows are written after the tick; record why they
                    # stopped instead of leaving a silently short CSV.
                    if now - self._last_tick_error_event_t >= 1.0:
                        self._last_tick_error_event_t = now
                        self._event("tick_error", error=repr(exc))

                if self._poweroff_requested:
                    self._poweroff_requested = False
                    # Do NOT queue systemctl poweroff here.  systemd starts
                    # shutdown immediately and SIGTERM can truncate the blackbox
                    # before telemetry.close()/fsync.  Stop the loop, clean up in
                    # stop(), then request poweroff as the last action.
                    self._poweroff_on_exit = True
                    stop_reason = "rc_poweroff"
                    self._stop_reason = "rc_poweroff"
                    self._event("poweroff_requested_after_flush")
                    LOG.warning("RC shutdown: stopping components and flushing blackbox first")
                    self._stop.set()
                    break

                next_t += period
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.monotonic()
        finally:
            final_reason = self._stop_reason if self._stop_reason != "stop" else stop_reason
            self.stop(final_reason)
        return 0

    def request_stop(self, reason: str = "stop") -> None:
        """Async-signal-safe stop trigger; cleanup runs in run()'s finally block."""
        self._stop_reason = str(reason)
        self._stop.set()

    def stop(self, reason: str = "stop") -> None:
        self._stop.set()
        if self._cleanup_started:
            return
        self._cleanup_started = True
        if not self._stop_logged:
            self._stop_logged = True
            self._event("run_stop_requested", reason=str(reason))
        # Betaflight 4.5.x holds the last MSP override values if the sender
        # disappears.  Give the mux a few frames of pilot passthrough before
        # it stops, and make the required pilot action unmistakable in the log.
        if self.mux is not None:
            try:
                self.mux.publish_cmd(None)
                time.sleep(0.12)
            except Exception:
                pass
            LOG.warning(
                "onboard stopping: if still airborne, flip CH8 to FORCE MANUAL "
                "(MSP OVERRIDE off) immediately; BF 4.5.x holds the last MSP frame"
            )
        for name in ("mux", "video", "detector", "camera", "state", "link"):
            obj = getattr(self, name, None)
            if obj is None:
                continue
            try:
                if name == "link" and hasattr(obj, "close"):
                    obj.close()
                elif hasattr(obj, "stop"):
                    obj.stop()
            except Exception:
                LOG.exception("stop %s failed", name)
        if self.telemetry is not None:
            try:
                self.telemetry.close(str(reason))
            except Exception:
                LOG.exception("closing blackbox failed")
        with self._retry_lock:
            retry_thread, self._retry_thread = self._retry_thread, None
        if retry_thread is not None and retry_thread is not threading.current_thread():
            retry_thread.join(timeout=1.5)
        if self._poweroff_on_exit:
            self._poweroff_on_exit = False
            try:
                os.sync()
            except Exception:
                pass
            last_detail = ""
            for attempt in range(3):
                try:
                    ok, detail = self._poweroff()
                except Exception as exc:
                    ok, detail = False, repr(exc)
                last_detail = str(detail)
                if ok:
                    self._event("poweroff_queued", detail=last_detail)
                    LOG.warning("RC shutdown: system poweroff queued (%s)", last_detail)
                    return
                LOG.error("RC shutdown attempt %d failed: %s", attempt + 1, last_detail)
                time.sleep(1.0)
            self._event("poweroff_failed", error=last_detail)
            LOG.error("RC shutdown: all poweroff attempts failed: %s", last_detail)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BF_Anti onboard runtime")
    parser.add_argument("--config", default="config/onboard.json", help="base JSON config")
    parser.add_argument("--profile", default=None, help="optional profile JSON overlay")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument(
        "--log-dir", default=None,
        help="blackbox root (default: config log_dir, then ./logs)",
    )
    parser.add_argument(
        "--no-telemetry", action="store_true",
        help="disable the CSV/JSONL blackbox; journald output only",
    )
    args = parser.parse_args(argv)

    # Console first so config/telemetry setup errors are visible in journald.
    setup_console_logging(args.log_level or "INFO")
    install_excepthooks()
    try:
        cfg = load_config(args.config, args.profile)
    except Exception as exc:
        LOG.exception("failed to load config %s", args.config)
        # Leave evidence on disk even when the config itself is broken; the
        # service is about to hit systemd's restart limit.
        try:
            fallback_dir = os.path.abspath(os.path.expanduser(args.log_dir or "logs"))
            fallback = TelemetryLog(fallback_dir, hz=1.0, max_runs=5)
            if fallback.start(collect_run_meta(argv=list(argv) if argv is not None else list(sys.argv[1:]))):
                fallback.event(
                    "config_error",
                    config=os.path.abspath(args.config),
                    error=repr(exc),
                )
                fallback.close("config_error")
        except Exception:
            LOG.exception("could not write the config-error run directory")
        return 2
    level = args.log_level or cfg.telemetry.level
    setup_console_logging(level)
    if float(cfg.safety.min_battery_v) <= 0.0:
        LOG.warning(
            "battery low-voltage protection is DISABLED (safety.min_battery_v=0); "
            "set a measured per-pack threshold before flight"
        )
    if not bool(getattr(cfg.camera, "strict_size", True)):
        LOG.warning(
            "camera.strict_size=false: a driver-selected resolution may not match "
            "the static intrinsics"
        )
    if not bool(getattr(cfg.fc, "enable_aux_discovery", True)):
        LOG.warning(
            "FC AUX discovery is disabled; the safety gate will keep AUTO closed "
            "because BOXIDS/RX_MAP cannot be verified"
        )

    log_dir = os.path.abspath(os.path.expanduser(args.log_dir or cfg.log_dir))
    tlog: Optional[TelemetryLog] = None
    if cfg.telemetry.enabled and not args.no_telemetry:
        tlog = TelemetryLog(
            log_dir,
            hz=cfg.telemetry.hz,
            max_file_bytes=int(cfg.telemetry.max_file_mb * 1024 * 1024),
            max_runs=cfg.telemetry.max_runs,
            max_segments=cfg.telemetry.max_segments,
            flush_interval_s=cfg.telemetry.flush_interval_s,
            fsync_interval_s=cfg.telemetry.fsync_interval_s,
        )
        meta = collect_run_meta(
            config=cfg.to_dict(),
            argv=list(argv) if argv is not None else list(sys.argv[1:]),
        )
        if tlog.start(meta):
            attach_file_handler(os.path.join(tlog.run_dir, "runtime.log"), level)
            install_faulthandler(os.path.join(tlog.run_dir, "faulthandler.log"))
            LOG.info("blackbox: %s", tlog.run_dir)
        else:
            LOG.error(
                "blackbox DISABLED at %s: %s (journald only)",
                log_dir, tlog.startup_error,
            )
            tlog = None
    else:
        LOG.warning("blackbox disabled by config or --no-telemetry; journald only")

    fc_target = (
        f"tcp://{cfg.fc.tcp_host}:{cfg.fc.tcp_port}"
        if str(cfg.fc.tcp_host or "").strip() else cfg.fc.port
    )
    LOG.info(
        "onboard config: %s profile=%s fc=%s detector=%s log_dir=%s",
        os.path.abspath(args.config), cfg.profile or "-", fc_target,
        cfg.detector.backend, log_dir,
    )
    app = OnboardApp(cfg, telemetry=tlog)

    def _on_thread_crash(thread_name, exc) -> None:
        try:
            app._event("thread_crash", thread=str(thread_name), error=repr(exc))
        except Exception:
            pass
        app.request_stop("thread_exception")

    set_thread_error_hook(_on_thread_crash)

    def _on_signal(reason: str) -> None:
        # A signal arriving during RC shutdown cleanup must not overwrite the
        # blackbox stop reason.
        if app._stop_reason != "rc_poweroff":
            app.request_stop(reason)

    signal.signal(signal.SIGINT, lambda *_: _on_signal("sigint"))
    signal.signal(signal.SIGTERM, lambda *_: _on_signal("sigterm"))
    try:
        return app.run()
    except Exception:
        LOG.exception("onboard runtime crashed")
        return 1
    finally:
        shutdown_faulthandler()
        if tlog is not None:
            tlog.close("process_exit")
        # runtime.log must be flushed before systemd's journal/file handles go
        # away; otherwise the last line of a crash/flight is lost.
        try:
            logging.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
