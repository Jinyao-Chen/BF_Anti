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
    # True once every required FC signal (link/att/alt/rc/status) arrived at
    # least once.  Before that the gate denies but must NOT latch: a FC that
    # has not enumerated yet at boot is not a flight fault, and latching it
    # would keep the gate closed for the whole sortie.
    fc_seen: bool = True
    # After the first complete FC sample, tolerate a short startup window in
    # which a stale field denies auto but does not latch.  A slow first MSP
    # exchange is a boot artefact, not a fault the pilot must acknowledge.
    warmup_grace: bool = False
    # Optional FC actual flight-mode check to prevent ANGLE vs ACRO mismatch.
    angle_mode_active: bool | None = None
    expected_att_mode: str | None = None
    # Verification of the FC's physical switch layout (BOXIDS/RX_MAP).  The
    # app sets require_layout on the supervisor; standalone callers/tests keep
    # the old behaviour by leaving this True.
    fc_layout_ok: bool = True
    # Battery voltage freshness.  0.0 is "never received", not "0 V".
    analog_age_s: float = float("inf")
    # Terminal COAST: out.confirmed stays true while guidance coasts.  Allow a
    # short, explicitly bounded loss window independent of max_det_age_s.
    coasting: bool = False
    track_lost_age_s: float = float("inf")
    coast_max_s: float = 0.0


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
        battery_debounce_s: float = 0.0,
        require_layout: bool = False,
    ) -> None:
        self.cfg = cfg
        self.require_override = bool(require_override)
        self.require_engage = bool(require_engage)
        self.require_layout = bool(require_layout)
        self.battery_debounce_s = max(0.0, float(battery_debounce_s))
        self._latched: str | None = None
        self._low_batt_since: float | None = None
        self.last: SafetyDecision | None = None

    def reset(self) -> None:
        self._latched = None
        self._low_batt_since = None
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
        if not inp.fc_seen:
            return self._deny("fc_warming_up")

        def stale(reason: str) -> SafetyDecision:
            # A bench/boot transient while disarmed is not a flight fault and
            # must never latch a FAULT that survives the next arming.
            if inp.warmup_grace or not inp.armed:
                return self._deny(reason)
            return self._critical(reason)

        if inp.link_age_s > c.max_status_age_s:
            return stale("fc_link_stale")
        if inp.rc_age_s > c.max_rc_age_s:
            return stale("rc_stale")
        if inp.att_age_s > c.max_att_age_s:
            return stale("att_stale")
        if inp.alt_age_s > c.max_alt_age_s:
            return stale("alt_stale")
        if inp.status_age_s > c.max_status_age_s:
            return stale("status_stale")
        if inp.loop_dt_s > c.max_loop_dt_s:
            return stale("loop_overrun")
        if c.min_battery_v > 0.0:
            max_age = float(getattr(c, "max_analog_age_s", 2.0))
            if not (inp.battery_v > 0.0) or not (inp.analog_age_s <= max_age):
                self._low_batt_since = None
                return self._deny("battery_telemetry_stale")
            if inp.battery_v < c.min_battery_v:
                # Low pack on the bench is not a fault to latch; airborne it
                # is.  Debounce is never permission to fly.
                if not inp.armed:
                    self._low_batt_since = None
                    return self._deny("battery_low")
                if self.battery_debounce_s <= 0.0:
                    return self._critical("battery_low")
                if self._low_batt_since is None:
                    self._low_batt_since = float(inp.now)
                if (float(inp.now) - self._low_batt_since) >= self.battery_debounce_s:
                    return self._critical("battery_low")
                return self._deny("battery_low_pending")
            else:
                self._low_batt_since = None
        else:
            self._low_batt_since = None

        if self._latched is not None:
            return self._deny("latched:" + self._latched)

        if not inp.armed:
            return self._deny("disarmed")
        if self.require_layout and inp.fc_layout_ok is not True:
            return self._deny("fc_layout_unknown")
        if inp.expected_att_mode is not None:
            exp = str(inp.expected_att_mode).lower()
            if inp.angle_mode_active is None:
                return self._deny("att_mode_unknown")
            if (exp == "angle" and not inp.angle_mode_active) or (
                exp == "rate" and inp.angle_mode_active
            ):
                return self._deny("att_mode_mismatch")
        if self.require_override and not inp.override_on:
            return self._deny("override_off")
        if self.require_engage and not inp.engage_on:
            return self._deny("engage_off")
        if not inp.detector_ok:
            return self._deny("detector_unhealthy")
        if not inp.target_confirmed:
            return self._deny("target_unconfirmed")
        if inp.coasting:
            coast_limit = float(inp.coast_max_s) if inp.coast_max_s > 0.0 else float(c.target_lost_s)
            if inp.track_lost_age_s > coast_limit:
                return self._deny("target_stale")
        elif inp.target_age_s > c.target_lost_s:
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
