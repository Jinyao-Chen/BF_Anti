# tests/test_msp_flags.py
from msp import ARMING_FLAG_NAMES, ARM_SWITCH_BIT


def test_bf45_layout():
    # Verified against betaflight 4.5-maintenance fc/runtime_config.h
    assert ARMING_FLAG_NAMES[3] == "NOT_DISARMED"
    assert ARMING_FLAG_NAMES[23] == "ACC_CALIBRATION"
    assert ARMING_FLAG_NAMES[24] == "MOTOR_PROTOCOL"
    assert ARMING_FLAG_NAMES[25] == "ARM_SWITCH"
    assert ARM_SWITCH_BIT == 25
    assert 26 not in ARMING_FLAG_NAMES  # ALTHOLD/POSHOLD are 4.6+
