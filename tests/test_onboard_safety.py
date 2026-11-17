"""Safety supervisor: fail closed, latch critical faults, allow only fresh data."""
from dataclasses import replace

from onboard.config import SafetyConfig
from onboard.safety import SafetyInputs, SafetySupervisor


def fresh(**kw):
    base = dict(
        now=10.0,
        armed=True,
        override_on=True,
        engage_on=True,
        rc_age_s=0.02,
        att_age_s=0.02,
        alt_age_s=0.02,
        status_age_s=0.02,
        link_age_s=0.02,
        altitude_m=10.0,
        range_m=12.0,
        loop_dt_s=0.02,
        target_confirmed=True,
        target_age_s=0.02,
        battery_v=16.0,
        detector_ok=True,
        estop=False,
    )
    base.update(kw)
    return SafetyInputs(**base)


def test_all_fresh_allows_auto():
    sup = SafetySupervisor(SafetyConfig(), True, True)
    dec = sup.evaluate(fresh())
    assert dec.allow_auto and dec.reason == "ok"


def test_stale_rc_is_critical_and_latches_until_reset():
    sup = SafetySupervisor(SafetyConfig(), True, True)
    dec = sup.evaluate(fresh(rc_age_s=1.0))
    assert not dec.allow_auto
    assert dec.critical and dec.reason == "CRITICAL:rc_stale"
    # Condition clears, but the latch remains.
    dec2 = sup.evaluate(fresh())
    assert not dec2.allow_auto and dec2.reason.startswith("latched:")
    sup.reset()
    assert sup.evaluate(fresh()).allow_auto


def test_disarmed_is_not_a_latched_fault():
    sup = SafetySupervisor(SafetyConfig(), True, True)
    dec = sup.evaluate(fresh(armed=False))
    assert not dec.allow_auto and not dec.critical
    assert dec.reason == "disarmed"
    assert sup.evaluate(fresh()).allow_auto


def test_override_and_engage_are_required():
    sup = SafetySupervisor(SafetyConfig(), True, True)
    assert sup.evaluate(fresh(override_on=False)).reason == "override_off"
    assert sup.evaluate(fresh(engage_on=False)).reason == "engage_off"


def test_target_and_altitude_gates_do_not_latch():
    sup = SafetySupervisor(SafetyConfig(), True, True)
    assert sup.evaluate(fresh(target_confirmed=False)).reason == "target_unconfirmed"
    assert sup.evaluate(fresh(altitude_m=1.0)).reason == "altitude_low"
    assert sup.evaluate(fresh(range_m=1000.0)).reason == "range_high"
    assert sup.evaluate(fresh()).allow_auto


def test_loop_overrun_is_critical():
    sup = SafetySupervisor(SafetyConfig(), True, True)
    dec = sup.evaluate(fresh(loop_dt_s=0.5))
    assert dec.critical and dec.reason == "CRITICAL:loop_overrun"


def test_battery_threshold_only_applies_when_configured():
    sup = SafetySupervisor(SafetyConfig(min_battery_v=0.0), True, True)
    assert sup.evaluate(fresh(battery_v=3.0)).allow_auto
    sup = SafetySupervisor(SafetyConfig(min_battery_v=13.6), True, True)
    dec = sup.evaluate(fresh(battery_v=13.0))
    assert dec.critical and dec.reason == "CRITICAL:battery_low"
