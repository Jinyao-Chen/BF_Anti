"""Guidance controller: detection -> tracker -> PVisualGuidance -> 4 RC channels.

The controller never touches ARM, ANGLE, BOXMSPOVERRIDE, or any AUX channel.
It returns a 4-tuple (roll, pitch, throttle, yaw) or None. The RC mux is the
only writer and decides whether those values are allowed on the wire.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from guidance.p_visual_guidance import PStrikeCommand, PStrikePhase, PVisualGuidance
from onboard.config import AppConfig
from onboard.types import (
    RC_MAX,
    RC_MID,
    RC_MIN,
    ControlOutput,
    Detection,
    FcState,
    Mode,
)
from perception.bbox_tracker import BboxTracker


def _clamp_rc(value: float) -> int:
    return int(max(RC_MIN, min(RC_MAX, round(float(value)))))


def _slew(prev: int, target: int, limit: int) -> int:
    delta = int(target) - int(prev)
    if limit > 0 and abs(delta) > limit:
        return int(prev) + (limit if delta > 0 else -limit)
    return int(target)


class Controller:
    def __init__(
        self,
        cfg: AppConfig,
        guidance: Optional[PVisualGuidance] = None,
        tracker: Optional[BboxTracker] = None,
    ) -> None:
        self.cfg = cfg
        self.guidance = guidance if guidance is not None else cfg.build_guidance()
        self.tracker = tracker if tracker is not None else BboxTracker(
            confirm_frames=getattr(cfg.detector, "confirm_frames", 2),
            max_coast_frames=getattr(cfg.detector, "max_coast_frames", 4),
            gate_px=getattr(cfg.detector, "gate_px", 90.0),
            gate_gain=getattr(cfg.detector, "gate_gain", 3.0),
        )
        self._prev_rpyt: tuple[int, int, int, int] = (
            RC_MID, RC_MID, int(cfg.control.hover_throttle), RC_MID,
        )
        self._last_seen: float = 0.0
        self._had_lock: bool = False
        self._last_out = ControlOutput()
        self.last_command: Optional[PStrikeCommand] = None

    def reset(self) -> None:
        self.guidance.reset()
        self.tracker.reset()
        self._prev_rpyt = (RC_MID, RC_MID, int(self.cfg.control.hover_throttle), RC_MID)
        self._last_seen = 0.0
        self._had_lock = False
        self.last_command = None
        self._last_out = ControlOutput()

    def _own_inputs(self, state: FcState):
        alt = float(state.altitude_m)
        if state.position_enu is not None:
            pos = tuple(float(v) for v in state.position_enu[:3])
        else:
            # No GPS/optical flow: horizontal position is deliberately zero.
            # The guidance law only uses altitude and vertical speed in this
            # profile (speed_source=closing, kp_velalign=0).
            pos = (0.0, 0.0, alt)
        if state.velocity_enu is not None:
            vel = tuple(float(v) for v in state.velocity_enu[:3])
        else:
            vel = (0.0, 0.0, float(state.vario_mps))
        return pos, vel

    def _rpyt(self, cmd: PStrikeCommand) -> tuple[int, int, int, int]:
        c = self.cfg.control
        angle_scale = 500.0 / max(1e-6, float(c.angle_limit_deg))
        yaw_scale = 500.0 / max(1e-6, float(c.yaw_full_rate_dps))

        roll = RC_MID + float(c.roll_sign) * float(cmd.roll_cmd_deg) * angle_scale
        # Betaflight ANGLE: positive pitch stick = nose down. The guidance
        # convention is positive pitch = nose up, hence the minus sign.
        pitch = RC_MID - float(c.pitch_sign) * float(cmd.pitch_cmd_deg) * angle_scale
        yaw = RC_MID + float(c.yaw_sign) * float(cmd.yaw_rate_dps) * yaw_scale
        thr = float(cmd.throttle_pwm)

        raw = (
            _clamp_rc(roll),
            _clamp_rc(pitch),
            _clamp_rc(max(float(c.throttle_min), min(float(c.throttle_max), thr))),
            _clamp_rc(yaw),
        )
        lim = int(c.rc_rate_limit)
        return (
            _slew(self._prev_rpyt[0], raw[0], lim),
            _slew(self._prev_rpyt[1], raw[1], lim),
            _slew(self._prev_rpyt[2], raw[2], lim),
            _slew(self._prev_rpyt[3], raw[3], lim),
        )

    def step(
        self,
        detection: Optional[Detection],
        state: FcState,
        now: float,
        dt: float,
    ) -> ControlOutput:
        dt = max(1e-3, min(float(dt), 0.5))
        max_age = float(self.cfg.safety.max_det_age_s)
        if detection is not None and detection.t > 0.0 and (now - detection.t) > max_age:
            detection = None
        bbox = detection.bbox if detection is not None else None

        track = self.tracker.update(bbox, dt=dt, timestamp=now)
        if track.fresh:
            self._last_seen = now
        bbox = track.bbox

        if bbox is None or not track.confirmed:
            if self._had_lock and (now - self._last_seen) > float(self.cfg.safety.lost_reset_s):
                self.guidance.reset()
                self._had_lock = False
            self._last_out = ControlOutput(
                mode=Mode.SEARCH, channels=None, command=None,
                confirmed=False, reason=track.reason,
            )
            return self._last_out

        self._had_lock = True
        pos, vel = self._own_inputs(state)
        u_rate = None
        try:
            u_rate = self.tracker.center_velocity_px_s
        except AttributeError:
            u_rate = None
        cmd = self.guidance.update(
            bbox,
            pos,
            vel,
            state.rpy_deg,
            float(now),
            dt,
            fresh=track.fresh,
            u_rate_px_s=(u_rate[0] if u_rate is not None else None),
        )
        self.last_command = cmd

        if cmd.hit:
            self._last_out = ControlOutput(
                mode=Mode.HIT, channels=None, command=cmd,
                confirmed=True, reason="hit",
            )
            return self._last_out
        if not cmd.valid:
            self._last_out = ControlOutput(
                mode=Mode.SEARCH, channels=None, command=cmd,
                confirmed=True, reason="guidance_invalid",
            )
            return self._last_out

        mode = Mode.TRACK
        if cmd.phase == PStrikePhase.TERMINAL:
            mode = Mode.TERMINAL
        elif cmd.phase == PStrikePhase.HIT:
            mode = Mode.HIT
        channels = self._rpyt(cmd)
        self._prev_rpyt = channels
        self._last_out = ControlOutput(
            mode=mode, channels=channels, command=cmd,
            confirmed=True, reason=track.reason,
        )
        return self._last_out

    @property
    def last_output(self) -> ControlOutput:
        return self._last_out
