"""RC shutdown guard: edge, hold, disarmed and throttle interlocks."""
from onboard.config import PowerConfig
from onboard.power import ShutdownGuard
from onboard.types import FcState


def make_state(now, *, ch6=1500, armed=False, throttle=1000, rc_age=0.02,
               status_age=0.02):
    rc = [1500] * 16
    rc[2] = throttle
    rc[6] = ch6
    return FcState(
        rc=tuple(rc),
        armed=armed,
        rc_t=now - rc_age,
        status_t=now - status_age,
    )


def guard(**kw):
    base = dict(aux_channel=6, hold_s=1.0, max_rc_age_s=0.3, max_status_age_s=0.6)
    base.update(kw)
    return ShutdownGuard(PowerConfig(**base))


def test_disabled_by_default_never_fires():
    g = ShutdownGuard(PowerConfig())
    now = 10.0
    assert not g.enabled
    assert not g.update(now, make_state(now, ch6=2000))
    assert not g.update(now + 10.0, make_state(now + 10.0, ch6=2000))


def test_requires_a_low_edge_before_high_hold():
    g = guard()
    now = 10.0
    # Switch is already high at boot: do not shut down until it is seen low.
    assert not g.update(now, make_state(now, ch6=2000))
    assert not g.update(now + 2.0, make_state(now + 2.0, ch6=2000))
    assert not g.update(now + 2.1, make_state(now + 2.1, ch6=1000))
    assert not g.update(now + 2.2, make_state(now + 2.2, ch6=2000))
    assert not g.update(now + 2.7, make_state(now + 2.7, ch6=2000))
    assert g.update(now + 3.2, make_state(now + 3.2, ch6=2000))
    assert g.fired
    assert not g.update(now + 4.0, make_state(now + 4.0, ch6=2000))


def test_release_resets_the_hold_timer():
    g = guard()
    now = 10.0
    assert not g.update(now, make_state(now, ch6=1000))
    assert not g.update(now + 0.1, make_state(now + 0.1, ch6=2000))
    assert not g.update(now + 0.6, make_state(now + 0.6, ch6=1000))
    assert not g.update(now + 0.7, make_state(now + 0.7, ch6=2000))
    assert not g.update(now + 1.0, make_state(now + 1.0, ch6=2000))
    assert g.update(now + 1.7, make_state(now + 1.7, ch6=2000))


def test_armed_blocks_and_requires_a_fresh_low_edge():
    g = guard()
    now = 10.0
    assert not g.update(now, make_state(now, ch6=1000, armed=True))
    assert not g.update(now + 0.1, make_state(now + 0.1, ch6=2000, armed=True))
    assert not g.update(now + 2.0, make_state(now + 2.0, ch6=2000, armed=False))
    assert not g.update(now + 2.1, make_state(now + 2.1, ch6=1000, armed=False))
    assert not g.update(now + 2.2, make_state(now + 2.2, ch6=2000, armed=False))
    assert g.update(now + 3.2, make_state(now + 3.2, ch6=2000, armed=False))


def test_high_throttle_blocks_the_hold():
    g = guard()
    now = 10.0
    assert not g.update(now, make_state(now, ch6=1000, throttle=1000))
    assert not g.update(now + 0.1, make_state(now + 0.1, ch6=2000, throttle=1500))
    assert not g.update(now + 1.0, make_state(now + 1.0, ch6=2000, throttle=1500))
    assert not g.update(now + 1.1, make_state(now + 1.1, ch6=2000, throttle=1000))
    assert g.update(now + 2.1, make_state(now + 2.1, ch6=2000, throttle=1000))


def test_stale_rc_or_status_blocks():
    g = guard()
    now = 10.0
    assert not g.update(now, make_state(now, ch6=1000))
    assert not g.update(now + 0.1, make_state(now + 0.1, ch6=2000, rc_age=1.0))
    assert not g.update(now + 2.0, make_state(now + 2.0, ch6=2000, status_age=1.0))
    # Both were dropped: a fresh low edge is required again.
    assert not g.update(now + 2.1, make_state(now + 2.1, ch6=2000))
    assert not g.update(now + 2.2, make_state(now + 2.2, ch6=1000))
    assert not g.update(now + 2.3, make_state(now + 2.3, ch6=2000))
    assert g.update(now + 3.3, make_state(now + 3.3, ch6=2000))


def test_retry_after_failed_command_allows_another_hold():
    g = guard()
    now = 10.0
    assert not g.update(now, make_state(now, ch6=1000))
    assert not g.update(now + 0.1, make_state(now + 0.1, ch6=2000))
    assert g.update(now + 1.1, make_state(now + 1.1, ch6=2000))
    g.retry_after(now + 1.1, delay_s=2.0)
    assert not g.update(now + 1.2, make_state(now + 1.2, ch6=2000))
    assert not g.update(now + 3.2, make_state(now + 3.2, ch6=2000))
    assert g.update(now + 4.2, make_state(now + 4.2, ch6=2000))


def test_missing_channel_does_not_raise():
    g = guard()
    now = 10.0
    st = FcState(rc=(1500, 1500, 1000), rc_t=now, status_t=now)
    assert not g.update(now, st)
