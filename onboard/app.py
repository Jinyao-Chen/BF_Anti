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
                switch_channel=self.cfg.fc.aux_engage,
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
        self._engaged = True
        self.safety.reset()
        # Keep the tracker lock, but clear guidance integrators / stale geometry.
        try:
            self.controller.guidance.reset()
        except Exception:
            pass
        LOG.info("auto engaged, pilot throttle=%s", engaged_throttle)

    def _on_disengage(self) -> None:
        if self._engaged:
            LOG.info("auto disengaged -> pilot")
        self._engaged = False
        self.controller.reset()
        self.safety.reset()

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
        return SafetyInputs(
            now=now,
            armed=bool(state.armed),
            override_on=aux(self.cfg.fc.aux_override) > 1700,
            engage_on=aux(self.cfg.fc.aux_engage) > 1700,
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
        period = 1.0 / max(1.0, float(self.cfg.control.rate_hz))
        self._last_dt = period if self._last_step is None else (now - self._last_step)
        self._last_step = now

        state = self.state.snapshot(now)
        detection = self.detector.latest(self.cfg.safety.max_det_age_s, now)
        out = self.controller.step(detection, state, now, self._last_dt)
        decision = self.safety.evaluate(self._safety_inputs(state, detection, out, now))

        if decision.allow_auto and self._engaged and out.channels is not None:
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
            if self.camera is not None:
                frame, frame_t = self.camera.latest(self.cfg.safety.max_det_age_s, now)
            try:
                self.video.submit(frame, detection, out, state, decision, now)
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
        if self.camera is not None:
            self.camera.start()
        if self.detector is not None:
            self.detector.start()
        if self.video is not None:
            self.video.start()
        if self.mux is not None:
            self.mux.start()

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
