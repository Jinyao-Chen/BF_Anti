"""Onboard application: wires FC, camera, detector, controller, mux, video.

This module is deliberately import-light. Hardware backends are imported in
_ensure_built() so config/controller/safety can be unit-tested on a laptop with
no camera, no FC, and no Hailo runtime.
"""
from __future__ import annotations

import argparse
import logging
import math
import signal
import threading
import time
from typing import Optional

from onboard.config import AppConfig, load_config
from onboard.controller import Controller
from onboard.safety import SafetyDecision, SafetyInputs, SafetySupervisor
from onboard.types import ControlOutput, Detection, FcState, Mode

LOG = logging.getLogger("bf_anti.onboard")


class OnboardApp:
    def __init__(
        self,
        cfg: AppConfig,
        link=None,
        state=None,
        detector=None,
        camera=None,
        mux=None,
        video=None,
    ) -> None:
        self.cfg = cfg
        self.link = link
        self.state = state
        self.detector = detector
        self.camera = camera
        self.mux = mux
        self.video = video

        self.controller = Controller(cfg)
        self.safety = SafetySupervisor(
            cfg.safety,
            require_override=cfg.fc.require_override,
            require_engage=cfg.fc.require_engage,
        )
        self._engaged = False
        self._pending_reset: Optional[str] = None
        self._stop = threading.Event()
        self._last_step: Optional[float] = None
        self._last_dt: float = 0.0
        self._built = False
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
            from onboard.camera import CameraSource
            self.camera = CameraSource(self.cfg.camera)
        if self.detector is None:
            from onboard.detector import build_detector
            self.detector = build_detector(self.cfg.detector, source=self.camera)
        elif self.camera is not None and hasattr(self.detector, "attach_source"):
            self.detector.attach_source(self.camera)
        if self.video is None and self.cfg.video.enabled:
            from onboard.video.pipeline import DisplayPipeline
            self.video = DisplayPipeline(self.cfg.video)
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

    def _on_disengage(self) -> None:
        if self._engaged:
            LOG.info("auto disengaged -> pilot")
        self._engaged = False
        self._pending_reset = "disengage"

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
    def start(self) -> None:
        self._ensure_built()
        if self.state is not None:
            self.state.start()
        # Start the RC path before camera/Hailo/video: the FC must see pilot
        # passthrough frames as early as possible, especially when the pilot
        # boots with CH8 already out of FORCE MANUAL.
        if self.mux is not None:
            self.mux.start()
        if self.camera is not None:
            self.camera.start()
        if self.detector is not None:
            self.detector.start()
        if self.video is not None:
            self.video.start()

    def run(self) -> int:
        self.start()
        period = 1.0 / max(1.0, float(self.cfg.control.rate_hz))
        next_t = time.monotonic()
        LOG.info("onboard loop started at %.1f Hz", self.cfg.control.rate_hz)
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                try:
                    out, decision = self.step_once(now)
                    if not decision.allow_auto and self._engaged:
                        LOG.debug("auto gate closed: %s", decision.reason)
                except Exception:
                    LOG.exception("control tick failed")
                next_t += period
                sleep_s = next_t - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.monotonic()
        finally:
            self.stop()
        return 0

    def stop(self) -> None:
        self._stop.set()
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="BF_Anti onboard runtime")
    parser.add_argument("--config", default="config/onboard.json", help="base JSON config")
    parser.add_argument("--profile", default=None, help="optional profile JSON overlay")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config, args.profile)
    app = OnboardApp(cfg)
    signal.signal(signal.SIGINT, lambda *_: app.stop())
    signal.signal(signal.SIGTERM, lambda *_: app.stop())
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
