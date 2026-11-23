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
        )
        self._engaged = False
        self._pending_reset: Optional[str] = None
        self._stop = threading.Event()
        self._stop_logged = False
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
                LOG.exception("camera init failed; continuing without camera")
                self._event("component_init_failed", component="camera", error=repr(exc))
                self.camera = None
        if self.detector is None:
            try:
                from onboard.detector import build_detector
                self.detector = build_detector(self.cfg.detector, source=self.camera)
            except Exception as exc:
                # Never let a missing HEF/HailoRT hide all FC/RC telemetry.
                from onboard.detector import NullDetector
                LOG.exception("detector init failed; falling back to null detector")
                self._event("component_init_failed", component="detector", error=repr(exc))
                self.detector = NullDetector(self.cfg.detector)
        elif self.camera is not None and hasattr(self.detector, "attach_source"):
            self.detector.attach_source(self.camera)
        if self.video is None and self.cfg.video.enabled:
            try:
                from onboard.video.pipeline import DisplayPipeline
                self.video = DisplayPipeline(self.cfg.video)
            except Exception as exc:
                LOG.exception("video init failed; continuing without OSD")
                self._event("component_init_failed", component="video", error=repr(exc))
                self.video = None
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
        else:
            if hasattr(self.mux, "on_engage"):
                self.mux.on_engage = self._on_engage
            if hasattr(self.mux, "on_disengage"):
                self.mux.on_disengage = self._on_disengage
        self._built = True

    def _on_engage(self, engaged_throttle: int) -> None:
        # Called from the mux thread.  Do not touch the guidance/tracker state
        # here: defer the reset to the control tick so the 50 Hz loop never
        # sees a half-reset guidance object.
        self._engaged = True
        self._pending_reset = "engage"
        LOG.info("auto engaged, pilot throttle=%s", engaged_throttle)
        self._event("engaged", engaged=True, throttle=int(engaged_throttle))

    def _on_disengage(self) -> None:
        if self._engaged:
            LOG.info("auto disengaged -> pilot")
        self._engaged = False
        self._pending_reset = "disengage"
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
        )

    def step_once(self, now: Optional[float] = None) -> tuple[ControlOutput, SafetyDecision]:
        self._ensure_built()
        now = time.monotonic() if now is None else float(now)
        if self._pending_reset is not None:
            try:
                if self._pending_reset == "engage":
                    # Keep the tracker lock, but clear guidance integrators and
                    # any safety latch from before the pilot engaged.
                    self.controller.guidance.reset()
                    self.safety.reset()
                else:
                    self.controller.reset()
                    self.safety.reset()
            finally:
                self._pending_reset = None
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
        out = self.controller.step(detection, state, now, self._last_dt)
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
            if self.camera is not None:
                frame, frame_t = self.camera.latest(self.cfg.safety.max_det_age_s, now)
            # Prefer the detector's frame+box pair for the OSD.  Drawing a
            # 40-80 ms old box on a newer camera frame is exactly the "box
            # lags the aircraft" artefact seen in the GUI before the async fix.
            if hasattr(self.detector, "latest_pair"):
                pair_frame, pair_det = self.detector.latest_pair(0.35, now)
                if pair_frame is not None:
                    frame = pair_frame
                    frame_t = float(getattr(pair_det, "t", 0.0) or now)
                    video_det = pair_det
                else:
                    video_det = None
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

    def _retry_components(self, now: float) -> None:
        if not self._start_failures:
            return
        for name, since in list(self._start_failures.items()):
            if now - since < 5.0:
                continue
            obj = getattr(self, name, None)
            if obj is None:
                self._start_failures.pop(name, None)
                continue
            try:
                obj.start()
            except Exception:
                self._start_failures[name] = now
                LOG.exception("retry start %s failed", name)
            else:
                self._start_failures.pop(name, None)
                LOG.info("%s started after retry", name)
                self._event("component_started", component=name)

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
                    # Telemetry rows are written after the tick; record why they
                    # stopped instead of leaving a silently short CSV.
                    if now - self._last_tick_error_event_t >= 1.0:
                        self._last_tick_error_event_t = now
                        self._event("tick_error", error=repr(exc))

                if self._poweroff_requested:
                    self._poweroff_requested = False
                    try:
                        ok, detail = self._poweroff()
                    except Exception as exc:
                        ok, detail = False, repr(exc)
                    if ok:
                        self._event("poweroff_queued", detail=str(detail))
                        LOG.warning("RC shutdown: system poweroff queued (%s)", detail)
                        stop_reason = "rc_poweroff"
                        self._stop.set()
                        break
                    LOG.error("RC shutdown: system poweroff failed: %s", detail)
                    self._event("poweroff_failed", error=str(detail))
                    # Keep the service alive; retry while the switch is held.
                    self.shutdown_guard.retry_after(now)

                next_t += period
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.monotonic()
        finally:
            self.stop(stop_reason)
        return 0

    def stop(self, reason: str = "stop") -> None:
        already_stopped = self._stop.is_set()
        self._stop.set()
        if not self._stop_logged:
            self._stop_logged = True
            self._event("run_stop_requested", reason=str(reason))
        # Betaflight 4.5.x holds the last MSP override values if the sender
        # disappears.  Give the mux a few frames of pilot passthrough before
        # it stops, and make the required pilot action unmistakable in the log.
        if self.mux is not None and not already_stopped:
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

    log_dir = os.path.abspath(os.path.expanduser(args.log_dir or cfg.log_dir))
    tlog: Optional[TelemetryLog] = None
    if cfg.telemetry.enabled and not args.no_telemetry:
        tlog = TelemetryLog(
            log_dir,
            hz=cfg.telemetry.hz,
            max_file_bytes=int(cfg.telemetry.max_file_mb * 1024 * 1024),
            max_runs=cfg.telemetry.max_runs,
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
    signal.signal(signal.SIGINT, lambda *_: app.stop("sigint"))
    signal.signal(signal.SIGTERM, lambda *_: app.stop("sigterm"))
    try:
        return app.run()
    except Exception:
        LOG.exception("onboard runtime crashed")
        return 1
    finally:
        shutdown_faulthandler()
        if tlog is not None:
            tlog.close("process_exit")


if __name__ == "__main__":
    raise SystemExit(main())
