# tests/test_onboard_state.py
"""FcStateEstimator: units, AETR mapping, staleness and counters (no hardware)."""
import struct
import time

import pytest

from msp import MSP_CMD
from onboard.config import FcConfig
from onboard.fc.state import FcStateEstimator
from onboard.types import FcState


class FakeLink:
    def __init__(self, responses=None, generation=1, errors=0):
        self.responses = dict(responses or {})
        self.generation = generation
        self.errors = errors
        self.last_rx_t = 0.0
        self.queries = []

    def query(self, cmd, is_v2=False, timeout=None):
        self.queries.append((cmd, is_v2))
        payload = self.responses.get((cmd, is_v2))
        if payload is None:
            return None
        self.last_rx_t = time.monotonic()
        return payload


def _status_ex_payload():
    flags = 0x00000001
    head = (struct.pack("<HHHI", 100, 0, 0x3F, flags)
            + bytes([0]) + struct.pack("<H", 12) + bytes([2, 4]))
    return (head + bytes([0]) + bytes([29]) + struct.pack("<I", 1 << 3)
            + bytes([0]) + struct.pack("<H", 45))


def _responses():
    return {
        (MSP_CMD.GET_ATTITUDE, False): struct.pack("<hhH", 100, -50, 270),
        (MSP_CMD.GET_ALTITUDE, False): struct.pack("<Ih", 2500, 125),
        (MSP_CMD.GET_RC, False): struct.pack(
            "<16H", 1100, 1200, 1300, 1400, *range(1500, 1512)),
        (MSP_CMD.GET_STATUS_EX, False): _status_ex_payload(),
        (MSP_CMD.GET_ANALOG, False): struct.pack("<BHHhH", 126, 1234, 88, -350, 1680),
        (MSP_CMD.MSP2_POSITION_STATE, True):
            bytes([1, 0, 1]) + struct.pack("<fff", 1.5, -2.5, 3.5)
            + struct.pack("<fff", 0.1, -0.2, 0.3),
    }


def _cfg():
    return FcConfig(telemetry_hz=100.0, reply_timeout_s=0.01)


def test_single_poll_decodes_all_fields_and_uses_degrees():
    link = FakeLink(_responses(), generation=5, errors=2)
    estimator = FcStateEstimator(link, _cfg())
    estimator._poll_once()
    state = estimator.snapshot()

    assert isinstance(state, FcState)
    assert state.rpy_deg == (10.0, -5.0, 270.0)   # yaw is degrees
    assert state.altitude_m == 25.0
    assert state.vario_mps == 1.25
    assert state.rc[:4] == (1100, 1200, 1400, 1300)  # Y/T swapped to AETR
    assert state.rc[4:] == tuple(range(1500, 1512))
    assert state.armed is True
    assert state.flight_mode_flags == 0x00000001
    assert state.voltage_v == 16.8
    assert state.current_a == -3.5
    assert state.mah == 1234
    assert state.position_enu == pytest.approx((1.5, -2.5, 3.5))
    assert state.velocity_enu == pytest.approx((0.1, -0.2, 0.3))
    assert state.att_t > 0.0 and state.alt_t > 0.0 and state.rc_t > 0.0
    assert state.status_t > 0.0 and state.analog_t > 0.0 and state.pos_t > 0.0
    assert state.link_t > 0.0
    assert state.generation == 5
    assert state.errors == 2

    channels, age = estimator.rc_snapshot()
    assert channels[:4] == (1100, 1200, 1400, 1300)
    assert 0.0 <= age < 0.5

    assert (MSP_CMD.MSP2_POSITION_STATE, True) in link.queries
    assert (MSP_CMD.MSP2_POSITION_STATE, False) not in link.queries


def test_negative_yaw_is_kept_signed():
    responses = _responses()
    responses[(MSP_CMD.GET_ATTITUDE, False)] = struct.pack("<hhh", 0, 0, -14)
    estimator = FcStateEstimator(FakeLink(responses), _cfg())
    estimator._poll_once()
    assert estimator.snapshot().rpy_deg == (0.0, 0.0, -14.0)


def test_empty_msp_rc_reply_leaves_rc_unreceived():
    responses = _responses()
    responses[(MSP_CMD.GET_RC, False)] = b""
    link = FakeLink(responses)
    estimator = FcStateEstimator(link, _cfg())
    estimator._poll_once()
    state = estimator.snapshot()
    assert state.rc == (1500,) * 16     # FcState default, not a received value
    assert state.rc_t == 0.0
    assert estimator.rc_snapshot() == (None, float("inf"))
    assert estimator._errors == 0       # 0-channel FC is not a decode error


def test_never_received_fields_are_not_invented():
    link = FakeLink(generation=0, errors=0)
    estimator = FcStateEstimator(link, _cfg())
    estimator._poll_once()
    state = estimator.snapshot()

    assert state.rpy_deg == (0.0, 0.0, 0.0)
    assert state.att_t == 0.0
    assert state.altitude_m == 0.0
    assert state.alt_t == 0.0
    assert state.rc == (1500,) * 16
    assert state.rc_t == 0.0
    assert state.armed is False
    assert state.status_t == 0.0
    assert state.position_enu is None
    assert state.velocity_enu is None
    assert state.link_t == 0.0
    assert state.generation == 0
    assert estimator.rc_snapshot() == (None, float("inf"))


def test_stale_replies_hold_last_value_with_old_timestamp():
    responses = _responses()
    link = FakeLink(responses)
    estimator = FcStateEstimator(link, _cfg())
    estimator._poll_once()
    first = estimator.snapshot()
    att_t = first.att_t

    link.responses = {}
    for _ in range(5):
        estimator._poll_once()
    later = estimator.snapshot()
    assert later.rpy_deg == first.rpy_deg
    assert later.att_t == att_t
    assert later.att_age_s(time.monotonic()) > 0.0
    channels, age = estimator.rc_snapshot()
    assert channels == first.rc
    assert age >= 0.0


def test_start_stop_background_poller():
    link = FakeLink(_responses(), generation=2)
    estimator = FcStateEstimator(link, _cfg())
    estimator.start()
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if estimator.snapshot().att_t > 0.0:
                break
            time.sleep(0.01)
        assert estimator.running is True
        assert estimator.snapshot().att_t > 0.0
    finally:
        estimator.stop()
        estimator.stop()  # idempotent
    assert estimator.running is False


def test_optional_position_state_disabled_after_misses():
    responses = _responses()
    responses.pop((MSP_CMD.MSP2_POSITION_STATE, True))
    link = FakeLink(responses)
    estimator = FcStateEstimator(link, _cfg())
    for _ in range(16):
        estimator._poll_once()
    state = estimator.snapshot()
    assert state.position_enu is None
    tries = [entry for entry in link.queries if entry[0] == MSP_CMD.MSP2_POSITION_STATE]
    assert len(tries) == 3  # POSITION_MAX_MISSES, then disabled
