# tests/test_crsf.py
from crsf import crc8_dvb_s2, crsf_to_rc, CRSFParser, CRSF_TYPE_RC


def build_rc_frame(raw_channels):
    """构造合法 CRSF RC 帧: [0xC8, len, 0x16, 22B payload, crc8]"""
    bits = 0
    for i, v in enumerate(raw_channels):
        bits |= (v & 0x7FF) << (11 * i)
    payload = bits.to_bytes(22, 'little')
    body = bytes([CRSF_TYPE_RC]) + payload
    return bytes([0xC8, len(body) + 1]) + body + bytes([crc8_dvb_s2(body)])


def test_crc8_known_vector():
    assert crc8_dvb_s2(bytes([0x01])) == 0xD5
    assert crc8_dvb_s2(b"") == 0


def test_crsf_to_rc_endpoints():
    assert crsf_to_rc(172) == 1000
    assert crsf_to_rc(992) == 1500
    assert crsf_to_rc(1811) == 2000


def test_parser_decodes_valid_frame():
    p = CRSFParser()
    got = p.feed(build_rc_frame([992] * 16))
    assert got is True
    assert p.channels is not None
    assert all(c == 1500 for c in p.channels)


def test_parser_rejects_corrupted_frame():
    p = CRSFParser()
    frame = bytearray(build_rc_frame([992] * 16))
    frame[5] ^= 0xFF  # 破坏 payload, CRC 不再匹配
    got = p.feed(bytes(frame))
    assert got is False
    assert p.channels is None
    assert p.frames_bad >= 1
    # 后续合法帧仍能解出 (resync)
    assert p.feed(build_rc_frame([172] * 16)) is True
    assert p.channels[0] == 1000


def test_parser_resyncs_over_garbage_prefix():
    p = CRSFParser()
    data = b"\xc8\x05junk" + build_rc_frame([1811] * 16)
    assert p.feed(data) is True
    assert p.channels[0] == 2000


def test_parser_ignores_non_rc_type():
    p = CRSFParser()
    body = bytes([0x14]) + bytes(10)  # link statistics frame
    frame = bytes([0xC8, len(body) + 1]) + body + bytes([crc8_dvb_s2(body)])
    assert p.feed(frame) is False
    assert p.channels is None
    assert p.frames_bad == 0
