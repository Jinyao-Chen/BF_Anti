"""Safety supervisor: the single gate between guidance and the FC.

Design rules:
* Any stale link, stale attitude/altitude/status, or loop overrun immediately
  denies auto authority and latches a FAULT until the pilot cycles the engage
  or override switch. This is deliberately stricter than "try to keep flying".
* Target loss, low altitude, geofence, and low battery deny auto but do not
  latch; the pilot can re-engage after the condition clears.
* The supervisor never writes RC. It only says yes/no and why.
"""
from __future__ import annotations

from dataclasses import dataclass

from onboard.config import SafetyConfig


@dataclass
class SafetyInputs:
    now: float
    armed: bool
    override_on: bool
    engage_on: bool
    rc_age_s: float
    att_age_s: float
    alt_age_s: float
    status_age_s: float
    link_age_s: float
    altitude_m: float
    range_m: float
    loop_dt_s: float
    target_confirmed: bool
    target_age_s: float
    battery_v: float = 0.0
    detector_ok: bool = True
    estop: bool = False


@dataclass
class SafetyDecision:
    allow_auto: bool
    reason: str
    critical: bool = False


class SafetySupervisor:
    def __init__(
        self,
        cfg: SafetyConfig,
        require_override: bool = True,
        require_engage: bool = True,
    ) -> None:
        self.cfg = cfg
        self.require_override = bool(require_override)
        self.require_engage = bool(require_engage)
        self._latched: str | None = None
        self.last: SafetyDecision | None = None

    def reset(self) -> None:
        self._latched = None
        self.last = None

    def _critical(self, reason: str) -> SafetyDecision:
        if self.cfg.fault_latch and self._latched is None:
            self._latched = reason
        dec = SafetyDecision(False, "CRITICAL:" + reason, True)
        self.last = dec
        return dec

    def _deny(self, reason: str) -> SafetyDecision:
        dec = SafetyDecision(False, reason, False)
        self.last = dec
        return dec

    def evaluate(self, inp: SafetyInputs) -> SafetyDecision:
        c = self.cfg
        if inp.estop:
            return self._critical("estop")
        if inp.link_age_s > c.max_status_age_s:
            return self._critical("fc_link_stale")
        if inp.rc_age_s > c.max_rc_age_s:
            return self._critical("rc_stale")
        if inp.att_age_s > c.max_att_age_s:
            return self._critical("att_stale")
        if inp.alt_age_s > c.max_alt_age_s:
            return self._critical("alt_stale")
        if inp.status_age_s > c.max_status_age_s:
            return self._critical("status_stale")
        if inp.loop_dt_s > c.max_loop_dt_s:
            return self._critical("loop_overrun")
        if c.min_battery_v > 0.0 and 0.0 < inp.battery_v < c.min_battery_v:
            return self._critical("battery_low")

        if self._latched is not None:
            return self._deny("latched:" + self._latched)

        if not inp.armed:
            return self._deny("disarmed")
        if self.require_override and not inp.override_on:
            return self._deny("override_off")
        if self.require_engage and not inp.engage_on:
            return self._deny("engage_off")
        if not inp.detector_ok:
            return self._deny("detector_unhealthy")
        if not inp.target_confirmed:
            return self._deny("target_unconfirmed")
        if inp.target_age_s > c.target_lost_s:
            return self._deny("target_stale")
        if inp.altitude_m < c.min_altitude_m:
            return self._deny("altitude_low")
        if inp.altitude_m > c.max_altitude_m:
            return self._deny("altitude_high")
        if inp.range_m == inp.range_m and inp.range_m > c.max_range_m:
            return self._deny("range_high")

        dec = SafetyDecision(True, "ok", False)
        self.last = dec
        return dec
