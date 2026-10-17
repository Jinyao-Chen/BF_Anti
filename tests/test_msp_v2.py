# tests/test_msp_v2.py
"""MSP v1/v2 framing, CRC and payload decoder contract tests."""
import struct

import pytest

from msp import (
    ARMING_FLAG_NAMES_SIM2REAL,
    MSP_CMD,
    FrameParser,
    MspFrame,
    build_frame,
    build_frame_v1,
    build_frame_v2,
    crc8_dvb_s2,
    decode_altitude,
    decode_analog,
    decode_arming_flags,
    decode_attitude,
    decode_board_info,
    decode_boxids,
    decode_boxnames,
    decode_mode_ranges,
    decode_msp2_position_state,
    decode_motor,
    decode_rc,
    decode_status,
    decode_status_ex,
)
from onboard.fc.link import MspLink


class FakeTransport:
    def __init__(self, response=b"", open_fails=0):
        self.response = response
        self.written = []
        self.opens = 0
        self.open_fails = open_fails
        self.closed = 0
        self._delivered = False

    def open(self):
        self.opens += 1
        if self.open_fails > 0:
            self.open_fails -= 1
            raise OSError("no link")

    def read(self, timeout_s):
        if self.response and not self._delivered:
            self._delivered = True
            data, self.response = self.response, b""
            return data
        return b""

    def write(self, data):
        self.written.append(bytes(data))

    def close(self):
        self.closed += 1


def test_crc8_dvb_s2_known_answer_and_incremental():
    assert crc8_dvb_s2(b"123456789") == 0xBC
    assert crc8_dvb_s2(b"") == 0
    assert crc8_dvb_s2(b"\x00") == 0
    assert crc8_dvb_s2(b"789", crc8_dvb_s2(b"123456")) == 0xBC


def test_build_frame_v1_layout_and_checksum():
    frame = build_frame_v1(108, b"\x01\x02")
    assert frame[0:3] == b"$M<"
    assert frame[3] == 2
    assert frame[4] == 108
    assert frame[5:7] == b"\x01\x02"
    assert frame[7] == (2 ^ 108 ^ 0x01 ^ 0x02)


def test_build_frame_v2_layout_and_crc_coverage():
    frame = build_frame_v2(0x3015, b"\xaa\xbb")
    assert frame[0:3] == b"$X<"
    assert frame[3] == 0  # flags
    assert struct.unpack_from("<HH", frame, 4) == (0x3015, 2)
    assert frame[8:10] == b"\xaa\xbb"
    assert frame[-1] == crc8_dvb_s2(frame[3:-1])


def test_parser_roundtrip_chunked_with_garbage():
    for is_v2, cmd in ((False, 108), (True, 0x3015)):
        payload = bytes(range(1, 20))
        frame = build_frame(cmd, payload, is_v2=is_v2, direction=">")
        parser = FrameParser()
        frames = []
        for byte in b"\x00\xff$Z" + frame:
            frames.extend(parser.feed(bytes([byte])))
        assert len(frames) == 1
        got = frames[0]
        assert isinstance(got, MspFrame)
        assert (got.cmd, got.payload, got.is_v2, got.direction) == (
            cmd, payload, is_v2, ">")


def test_parser_rejects_corrupt_crc():
    for is_v2 in (False, True):
        frame = bytearray(build_frame(108, b"\x01\x02\x03", is_v2=is_v2,
                                      direction=">"))
        frame[-1] ^= 0xFF
        parser = FrameParser()
        assert parser.feed(bytes(frame)) == []
        assert parser.errors >= 1


def test_v1_jumbo_response_decodes_with_logical_cmd():
    payload = bytes(range(256)) * 2  # 512 bytes, like BOXNAMES page 1
    header = b"$M>" + bytes([255, MSP_CMD.BOXNAMES]) + struct.pack("<H", len(payload))
    checksum = 0
    for byte in header[3:] + payload:
        checksum ^= byte
    frame = header + payload + bytes([checksum & 0xFF])
    frames = FrameParser().feed(frame)
    assert len(frames) == 1
    assert frames[0].cmd == MSP_CMD.BOXNAMES
    assert frames[0].payload == payload
    assert frames[0].is_v2 is False


def test_builder_emits_v1_jumbo_reply_for_large_payloads():
    payload = bytes(range(256)) + b"tail"
    frame = build_frame_v1(MSP_CMD.BOXNAMES, payload, direction=">")
    assert frame[0:3] == b"$M>"
    assert frame[3] == 255
    assert struct.unpack_from("<H", frame, 5)[0] == len(payload)
    frames = FrameParser().feed(frame)
    assert len(frames) == 1
    assert frames[0].cmd == MSP_CMD.BOXNAMES
    assert frames[0].payload == payload


def test_error_frame_is_decoded_but_flagged():
    frames = FrameParser().feed(build_frame_v2(1, b"", error=True,
                                               direction=">"))
    assert len(frames) == 1
    assert frames[0].error is True
    assert frames[0].direction == "!"


def test_link_query_accepts_v1_and_v2_replies():
    for is_v2 in (False, True):
        frame = build_frame(108, struct.pack("<hhH", 100, -50, 270),
                            is_v2=is_v2, direction=">")
        link = MspLink(FakeTransport(frame), reply_timeout_s=0.2,
                       reconnect_interval_s=0.0)
        payload = link.query(108, is_v2=is_v2, timeout=0.2)
        assert payload == struct.pack("<hhH", 100, -50, 270)
        assert link.connected is True
        assert link.generation == 1
        assert link.rx_frames == 1
        assert link.last_rx_t > 0.0
        link.close()


def test_link_reconnects_and_bumps_generation():
    transport = FakeTransport(open_fails=1)
    link = MspLink(transport, reply_timeout_s=0.05, reconnect_interval_s=0.0)
    assert link.connect() is False
    assert link.connect() is True
    assert link.generation == 1
    link._drop(OSError("boom"))
    assert link.connected is False
    assert link.connect() is True
    assert link.generation == 2


def test_link_send_rc_16_wire_payload_and_set_rc_compat():
    transport = FakeTransport()
    link = MspLink(transport, reply_timeout_s=0.1, reconnect_interval_s=0.0)
    channels = [1100, 1200, 1300, 1400] + list(range(1450, 1450 + 12))
    assert link.send_rc_16(channels) is True
    frames = FrameParser().feed(transport.written[-1])
    assert len(frames) == 1
    assert frames[0].cmd == MSP_CMD.SET_RAW_RC
    assert len(frames[0].payload) == 32
    assert list(struct.unpack("<16H", frames[0].payload)) == channels

    # 8-arg compatibility path pads channels 9..16 to neutral.
    assert link.set_rc(1500, 1400, 1200, 1300, 1000, 1100, 1100, 1000) is True
    frames = FrameParser().feed(transport.written[-1])
    values = list(struct.unpack("<16H", frames[0].payload))
    assert values[:8] == [1500, 1400, 1200, 1300, 1000, 1100, 1100, 1000]
    assert values[8:] == [1500] * 8
    link.close()


def test_decode_attitude_yaw_is_degrees_not_decidegrees():
    payload = struct.pack("<hhh", 123, -45, 270)
    assert decode_attitude(payload) == (12.3, -4.5, 270.0)


def test_decode_attitude_negative_yaw_stays_signed():
    # SITL live payload: roll=0 pitch=0 yaw=-14 -> 0x00 00 00 00 f2 ff.
    payload = bytes.fromhex("00000000f2ff")
    assert decode_attitude(payload) == (0.0, 0.0, -14.0)


def test_decode_altitude_u32_cm_and_i16_vario():
    payload = struct.pack("<Ih", 12345, -250)
    altitude_m, vario_mps = decode_altitude(payload)
    assert altitude_m == pytest.approx(123.45)
    assert vario_mps == pytest.approx(-2.5)


def test_decode_analog_9_byte_current_firmware():
    payload = struct.pack("<BHHhH", 126, 1234, 88, -350, 1680)
    decoded = decode_analog(payload)
    assert decoded["layout"] == "v2-9b"
    assert decoded["voltage_v"] == pytest.approx(16.8)
    assert decoded["current_a"] == pytest.approx(-3.5)
    assert decoded["mah"] == 1234
    assert decoded["rssi"] == 88


def test_decode_analog_legacy_7_byte():
    payload = struct.pack("<BhHH", 126, -350, 1234, 88)
    decoded = decode_analog(payload)
    assert decoded["layout"] == "legacy-7b"
    assert decoded["voltage_v"] == pytest.approx(12.6)
    assert decoded["current_a"] == pytest.approx(-3.5)
    assert decoded["mah"] == 1234
    assert decoded["rssi"] == 88


def test_decode_status_armed_bit_and_offset6():
    flags = 0x00000001 | 0x00000010
    payload = struct.pack("<HHHI", 100, 0, 0x3F, flags) + bytes([2]) + b"\x00\x00"
    decoded = decode_status(payload)
    assert decoded["flight_mode_flags"] == flags
    assert decoded["armed"] is True
    assert decoded["profile"] == 2

    payload = struct.pack("<HHHI", 100, 0, 0x3F, 0x10) + bytes([0]) + b"\x00\x00"
    assert decode_status(payload)["armed"] is False


def _status_ex_payload(byte_count, extended, arming_flags, flag_count=29,
                       config_state=0, cpu_temp=45):
    head = (struct.pack("<HHHI", 100, 0, 0x3F, 0x00000001)
            + bytes([0]) + struct.pack("<H", 12) + bytes([2, 4]))  # 15 bytes
    body = bytes([byte_count]) + extended + bytes([flag_count])
    body += struct.pack("<I", arming_flags)
    body += bytes([config_state]) + struct.pack("<H", cpu_temp)
    return head + body


def test_decode_status_ex_walks_byte_count_to_arming_flags():
    arming = (1 << 3) | (1 << 28)
    payload = _status_ex_payload(2, b"\xaa\xbb", arming)
    decoded = decode_status_ex(payload)
    assert decoded["armed"] is True
    assert decoded["extended_flight_mode_flags"] == b"\xaa\xbb"
    assert decoded["arming_flag_count"] == 29
    assert decoded["arming_flags"] == arming
    assert decoded["config_state"] == 0
    assert decoded["cpu_temp_c"] == 45.0
    assert "ARM_SWITCH" in decode_arming_flags(arming, "sim2real")
    assert "NOT_DISARMED" in decode_arming_flags(arming, "bf4.5")
    assert ARMING_FLAG_NAMES_SIM2REAL[28] == "ARM_SWITCH"


def test_decode_rc_swaps_yaw_and_throttle_to_aetr():
    firmware_order = [1100, 1200, 1300, 1400] + list(range(1500, 1512))
    payload = struct.pack("<16H", *firmware_order)
    decoded = decode_rc(payload)
    assert len(decoded) == 16
    assert decoded[0:2] == (1100, 1200)
    assert decoded[2] == 1400  # throttle
    assert decoded[3] == 1300  # yaw
    assert decoded[4:] == tuple(range(1500, 1512))


def test_decode_board_info_pstrings():
    payload = b"BTFL" + struct.pack("<H", 7) + bytes([2, 0x43])
    payload += bytes([9]) + b"SITL_GAZE"
    payload += bytes([5]) + b"BOARD"
    payload += bytes([4]) + b"BFAT"
    decoded = decode_board_info(payload)
    assert decoded["board_identifier"] == "BTFL"
    assert decoded["hardware_revision"] == 7
    assert decoded["board_type"] == 2
    assert decoded["target_name"] == "SITL_GAZE"
    assert decoded["board_name"] == "BOARD"
    assert decoded["manufacturer_id"] == "BFAT"


def test_decode_boxnames_boxids_mode_ranges():
    names = decode_boxnames(b"ARM;ANGLE;MSP OVERRIDE;")
    assert names == ["ARM", "ANGLE", "MSP OVERRIDE"]
    assert decode_boxids(bytes([0, 1, 50])) == [0, 1, 50]

    payload = bytes([0, 0, 32, 48, 1, 1, 32, 48, 50, 6, 24, 40, 0, 0, 0, 0])
    ranges = decode_mode_ranges(payload)
    assert len(ranges) == 4
    assert ranges[0]["start_pwm"] == 1700
    assert ranges[0]["end_pwm"] == 2100
    assert ranges[0]["channel_index"] == 4
    assert ranges[2]["channel_index"] == 10
    assert ranges[3]["valid"] is False
    assert ranges[2]["valid"] is True


def test_decode_motor_and_msp2_position_state_offsets():
    assert decode_motor(struct.pack("<8H", *range(8))) == tuple(range(8))

    payload = bytes([1, 0, 1])
    payload += struct.pack("<fff", 1.5, -2.5, 3.5)
    payload += struct.pack("<fff", 0.1, -0.2, 0.3)
    payload += struct.pack("<fff", 4.0, -5.0, 6.0)
    decoded = decode_msp2_position_state(payload)
    assert decoded["offboard_active"] is True
    assert decoded["reached"] is False
    assert decoded["position_enu"] == pytest.approx((1.5, -2.5, 3.5))
    assert decoded["velocity_enu"] == pytest.approx((0.1, -0.2, 0.3))
    assert decoded["target_velocity_enu"] == pytest.approx((4.0, -5.0, 6.0))
    assert decode_msp2_position_state(payload[:20]) is None
