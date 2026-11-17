"""Guidance controller: detection -> tracker -> PVisualGuidance -> 4 RC channels.

The controller never touches ARM, ANGLE, BOXMSPOVERRIDE, or any AUX channel.
It returns a 4-tuple (roll, pitch, throttle, yaw) or None. The RC mux is the
only writer and decides whether those values are allowed on the wire.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from guidance.p_visual_guidance import (
    PStrikeCommand,
    PStrikePhase,
    PVisualGuidance,
    p_cmd_to_rc,
)
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

    def _predict_bbox(self, bbox, age_s: float):
        """Shift a measured box forward by the detection transport age.

        The camera timestamp is taken at capture, but the box arrives after
        JPEG/inference/transport.  Without this the controller always flies to
        where the target was 40-100 ms ago.  Same 120 ms / 90 px clamp as the
        Gazebo loop; width/height are left to the tracker.
        """
        if bbox is None or age_s <= 0.0:
            return bbox
        vel = None
        try:
            vel = self.tracker.center_velocity_px_s
        except Exception:
            vel = None
        if vel is None:
            return bbox
        age = min(0.12, max(0.0, float(age_s)))
        dx = max(-90.0, min(90.0, float(vel[0]) * age))
        dy = max(-90.0, min(90.0, float(vel[1]) * age))
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return bbox
        return (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)

    def _rpyt(self, cmd: PStrikeCommand, state: FcState) -> tuple[int, int, int, int]:
        c = self.cfg.control
        angle_limit = float(c.angle_limit_deg)
        yaw_full = float(c.yaw_full_rate_dps)
        att_mode = str(getattr(c, "att_mode", "angle")).lower()

        if att_mode == "rate":
            # Companion-side attitude P -> body-rate command, then ACRO.
            # This is the sim's --att-mode rate path: the firmware ANGLE loop
            # only realised ~50-70% of a commanded pitch step, while the rate
            # cascade reached the commanded attitude.
            k_att = float(getattr(c, "att_rate_gain", 6.0))
            rate_max = float(getattr(c, "rate_max_dps", 250.0))
            rpy = getattr(state, "rpy_deg", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
            roll_rate = max(-rate_max, min(rate_max, k_att * (float(cmd.roll_cmd_deg) - float(rpy[0]))))
            pitch_rate = max(-rate_max, min(rate_max, k_att * (float(cmd.pitch_cmd_deg) - float(rpy[1]))))
            channels = p_cmd_to_rc(
                cmd,
                angle_limit_deg=angle_limit,
                yaw_full_rate_dps=yaw_full,
                prev=self._prev_rpyt,
                rate_limit=int(c.rc_rate_limit),
                roll_sign=float(c.roll_sign),
                pitch_sign=float(c.pitch_sign),
                yaw_sign=float(c.yaw_sign),
                roll_rate_dps=roll_rate,
                pitch_rate_dps=pitch_rate,
                rate_full_dps=float(getattr(c, "rate_full_dps", 600.0)),
                angle_on=False,
            )
        else:
            channels = p_cmd_to_rc(
                cmd,
                angle_limit_deg=angle_limit,
                yaw_full_rate_dps=yaw_full,
                prev=self._prev_rpyt,
                rate_limit=int(c.rc_rate_limit),
                roll_sign=float(c.roll_sign),
                pitch_sign=float(c.pitch_sign),
                yaw_sign=float(c.yaw_sign),
                angle_on=True,
            )
        # AUX channels from the helper are intentionally discarded: the
        # companion writes RPYT only and never ARM/ANGLE/OVERRIDE.
        return tuple(int(v) for v in channels[:4])

    def step(
        self,
        detection: Optional[Detection],
        state: FcState,
        now: float,
        dt: float,
    ) -> ControlOutput:
        dt = max(1e-3, min(float(dt), 0.5))
        max_age = float(self.cfg.safety.max_det_age_s)
        det_age = 0.0
        if detection is not None and detection.t > 0.0:
            det_age = max(0.0, float(now) - float(detection.t))
            if det_age > max_age:
                detection = None
        bbox = detection.bbox if detection is not None else None
        bbox = self._predict_bbox(bbox, det_age)

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
        channels = self._rpyt(cmd, state)
        self._prev_rpyt = channels
        self._last_out = ControlOutput(
            mode=mode, channels=channels, command=cmd,
            confirmed=True, reason=track.reason,
        )
        return self._last_out

    @property
    def last_output(self) -> ControlOutput:
        return self._last_out
