"""Guidance controller: detection -> tracker -> PVisualGuidance -> 4 RC channels.

The controller never touches ARM, ANGLE, BOXMSPOVERRIDE, or any AUX channel.
It returns a 4-tuple (roll, pitch, throttle, yaw) or None. The RC mux is the
only writer and decides whether those values are allowed on the wire.
"""
from __future__ import annotations

import math
from typing import Optional

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
            max_coast_frames=getattr(cfg.detector, "max_coast_frames", 6),
            max_coast_s=getattr(cfg.detector, "max_coast_s", 0.25),
            gate_px=getattr(cfg.detector, "gate_px", 90.0),
            gate_frac=getattr(cfg.detector, "gate_frac", 0.15),
            image_width=int(getattr(cfg.camera, "width", 1280)),
            gate_gain=getattr(cfg.detector, "gate_gain", 3.0),
            coast_vel_tau_s=getattr(cfg.detector, "coast_vel_tau_s", 0.24),
        )
        self._prev_rpyt: tuple[int, int, int, int] = (
            RC_MID, RC_MID, int(cfg.control.hover_throttle), RC_MID,
        )
        self._last_seen: float = 0.0
        self._last_det_t: float = -math.inf
        self._had_lock: bool = False
        self._last_out = ControlOutput()
        self.last_command: Optional[PStrikeCommand] = None

    def reset(self) -> None:
        self.guidance.reset()
        self.tracker.reset()
        self._prev_rpyt = (RC_MID, RC_MID, int(self.cfg.control.hover_throttle), RC_MID)
        self._last_seen = 0.0
        self._last_det_t = -math.inf
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
        w_lim = float(self.guidance.intrinsics.width)
        h_lim = float(self.guidance.intrinsics.height)
        x1 = max(0.0, min(w_lim - 1.0, float(bbox[0]) + dx))
        y1 = max(0.0, min(h_lim - 1.0, float(bbox[1]) + dy))
        x2 = max(x1 + 1.0, min(w_lim, float(bbox[2]) + dx))
        y2 = max(y1 + 1.0, min(h_lim, float(bbox[3]) + dy))
        return (x1, y1, x2, y2)

    def last_seen_age_s(self, now: float) -> float:
        if self._last_seen <= 0.0:
            return float("inf")
        return max(0.0, float(now) - float(self._last_seen))

    def _sync_prev_from_state(self, state: FcState) -> None:
        """Keep the RC slew base near the FC's current inputs when idle.

        While BOXMSPOVERRIDE is active, MSP_RC echoes the last command, which is
        exactly the value the next AUTO engagement should slew away from.  In
        physical manual mode it is the pilot stick.  Either way this avoids a
        first-tick jump from a stale extreme command.
        """
        # Only after a real lock/output: a first acquisition should retain the
        # hover-anchored slew base, not the pilot's arbitrary stick position.
        if not (self._had_lock or self._last_out.channels is not None):
            return
        rc = getattr(state, "rc", None)
        if rc is not None and len(rc) >= 4:
            try:
                self._prev_rpyt = tuple(int(rc[i]) for i in range(4))
            except (TypeError, ValueError):
                pass

    def _rpyt(self, cmd: PStrikeCommand, state: FcState,
              dt: float = 0.02) -> tuple[int, int, int, int]:
        c = self.cfg.control
        angle_limit = float(c.angle_limit_deg)
        yaw_full = float(c.yaw_full_rate_dps)
        att_mode = str(getattr(c, "att_mode", "angle")).lower()
        slew_per_s = float(getattr(c, "rc_slew_rate_per_s", 0.0) or 0.0)
        if slew_per_s > 0.0:
            rate_limit = max(1, int(round(slew_per_s * max(1e-3, float(dt)))))
        else:
            rate_limit = int(c.rc_rate_limit)

        if att_mode == "rate":
            # Companion-side attitude P -> body-rate command, then ACRO.
            # This is the sim's --att-mode rate path: the firmware ANGLE loop
            # only realised ~50-70% of a commanded pitch step, while the rate
            # cascade reached the commanded attitude.
            k_att = float(getattr(c, "att_rate_gain", 6.0))
            rate_max = float(getattr(c, "rate_max_dps", 250.0))
            rpy = getattr(state, "rpy_deg", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
            phi_rad = math.radians(float(rpy[0]))
            theta_rad = math.radians(float(rpy[1]))
            dphi = k_att * (float(cmd.roll_cmd_deg) - float(rpy[0]))
            dtheta = k_att * (float(cmd.pitch_cmd_deg) - float(rpy[1]))
            dpsi = float(cmd.yaw_rate_dps)
            # Euler-rate to body-rate transform prevents large-pitch yaw/roll coupling
            p_dps = dphi - dpsi * math.sin(theta_rad)
            q_dps = dtheta * math.cos(phi_rad) + dpsi * math.sin(phi_rad) * math.cos(theta_rad)
            r_dps = -dtheta * math.sin(phi_rad) + dpsi * math.cos(phi_rad) * math.cos(theta_rad)
            roll_rate = max(-rate_max, min(rate_max, p_dps))
            pitch_rate = max(-rate_max, min(rate_max, q_dps))
            yaw_rate_body = max(-yaw_full, min(yaw_full, r_dps))
            cmd_body = PStrikeCommand(
                **{**cmd.__dict__, "yaw_rate_dps": yaw_rate_body}
            )
            channels = p_cmd_to_rc(
                cmd_body,
                angle_limit_deg=angle_limit,
                yaw_full_rate_dps=yaw_full,
                prev=self._prev_rpyt,
                rate_limit=rate_limit,
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
                rate_limit=rate_limit,
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
        state_estimator=None,
        candidates=None,
    ) -> ControlOutput:
        dt = max(1e-3, min(float(dt), 0.5))
        max_age = float(self.cfg.safety.max_det_age_s)
        det_age = 0.0
        det_t = 0.0
        if detection is not None and detection.t > 0.0:
            det_t = float(detection.t)
            det_age = max(0.0, float(now) - det_t)
            if det_age > max_age:
                detection = None

        # Deduplicate repeated historical detections from the 50 Hz loop so
        # BboxTracker.fresh stays False on missed/repeated frames.
        is_new_det = detection is not None and (
            det_t <= 0.0 or det_t > self._last_det_t + 1e-9
        )
        raw_bbox = None
        if is_new_det:
            if det_t > 0.0:
                self._last_det_t = det_t
            raw_bbox = detection.bbox
            if candidates:
                track = self.tracker.update_candidates(
                    candidates, dt=dt, timestamp=(det_t if det_t > 0.0 else now),
                )
            else:
                track = self.tracker.update(
                    raw_bbox, dt=dt, timestamp=(det_t if det_t > 0.0 else now),
                )
        else:
            track = self.tracker.update(None, dt=dt, timestamp=now)

        if track.fresh:
            self._last_seen = now
            # When multiple candidates were associated, the tracker's selected
            # box (not necessarily the detector's highest-confidence box) is
            # the target used for the inertial LOS.
            if track.bbox is not None:
                raw_bbox = track.bbox
        bbox = track.bbox

        if bbox is None or not track.confirmed:
            if self._had_lock:
                lost_for = now - self._last_seen
                if lost_for > float(self.cfg.safety.lost_reset_s):
                    self.guidance.reset()
                    self._had_lock = False
                elif lost_for <= float(getattr(self.guidance, "lost_timeout_s", 0.0)):
                    pos, vel = self._own_inputs(state)
                    cmd = self.guidance.update(
                        None, pos, vel, state.rpy_deg, float(now), dt, fresh=False,
                    )
                    self.last_command = cmd
                    if cmd.phase == PStrikePhase.COAST:
                        channels = self._rpyt(cmd, state, dt=dt)
                        self._prev_rpyt = channels
                        self._last_out = ControlOutput(
                            mode=Mode.TRACK, channels=channels, command=cmd,
                            confirmed=True, reason="coast",
                        )
                        return self._last_out
            self._sync_prev_from_state(state)
            self._last_out = ControlOutput(
                mode=Mode.SEARCH, channels=None, command=None,
                confirmed=False, reason=track.reason,
            )
            return self._last_out

        self._had_lock = True
        # Forward-predict once for guidance without feeding extrapolated pixels
        # back into BboxTracker velocity estimation.
        pred_age = (now - self._last_det_t) if self._last_det_t > 0.0 else det_age
        bbox_for_guid = self._predict_bbox(bbox, pred_age) if track.fresh else bbox

        # Use historical attitude interpolated at image capture time when available.
        rpy_for_los = state.rpy_deg
        if (
            track.fresh
            and self._last_det_t > 0.0
            and state_estimator is not None
            and hasattr(state_estimator, "attitude_at")
        ):
            cam_lat = float(getattr(self.cfg.camera, "latency_s", 0.0) or 0.0)
            rpy_for_los = state_estimator.attitude_at(self._last_det_t - cam_lat)

        pos, vel = self._own_inputs(state)
        u_rate = None
        try:
            u_rate = self.tracker.center_velocity_px_s
        except AttributeError:
            u_rate = None
        cmd = self.guidance.update(
            bbox_for_guid,
            pos,
            vel,
            rpy_for_los,
            float(now),
            dt,
            fresh=track.fresh,
            u_rate_px_s=(u_rate[0] if u_rate is not None else None),
            live_rpy_deg=state.rpy_deg,
            los_bbox_xyxy=(raw_bbox if track.fresh else None),
        )
        self.last_command = cmd

        if cmd.hit:
            self._sync_prev_from_state(state)
            self._last_out = ControlOutput(
                mode=Mode.HIT, channels=None, command=cmd,
                confirmed=True, reason="hit",
            )
            return self._last_out
        if not cmd.valid:
            self._sync_prev_from_state(state)
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
        channels = self._rpyt(cmd, state, dt=dt)
        self._prev_rpyt = channels
        self._last_out = ControlOutput(
            mode=mode, channels=channels, command=cmd,
            confirmed=True, reason=track.reason,
        )
        return self._last_out

    @property
    def last_output(self) -> ControlOutput:
        return self._last_out
