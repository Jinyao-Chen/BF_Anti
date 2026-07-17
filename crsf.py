"""CRSF (ELRS/Crossfire) RC 通道读取, 带 CRC8 校验."""
import time
import threading

import serial

CRSF_SYNC = 0xC8
CRSF_TYPE_RC = 0x16
RC_PAYLOAD_LEN = 22  # 16ch x 11bit


def crc8_dvb_s2(data):
    """CRSF 标准 CRC8, poly 0xD5, 覆盖 type+payload."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0xD5) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def crsf_to_rc(raw):
    """CRSF raw (172-1811) -> RC (1000-2000)."""
    return int(round(1000 + (raw - 172) * 1000 / (1811 - 172)))


def decode_rc_payload(payload):
    bits = int.from_bytes(payload[:RC_PAYLOAD_LEN], 'little')
    return [crsf_to_rc((bits >> (11 * i)) & 0x7FF) for i in range(16)]


class CRSFParser:
    """纯字节流解析器 (无 IO): feed() 返回本次是否解出 >=1 个有效 RC 帧."""

    def __init__(self):
        self._buf = bytearray()
        self.channels = None
        self.frames_ok = 0
        self.frames_bad = 0

    def feed(self, data):
        self._buf.extend(data)
        got = False
        while True:
            pos = self._buf.find(bytes([CRSF_SYNC]))
            if pos < 0:
                self._buf.clear()
                break
            if pos > 0:
                del self._buf[:pos]
            if len(self._buf) < 2:
                break
            length = self._buf[1]
            if not (2 <= length <= 62):
                del self._buf[:1]
                continue
            end = 2 + length
            if len(self._buf) < end:
                break
            body = bytes(self._buf[2:end - 1])
            if crc8_dvb_s2(body) == self._buf[end - 1]:
                if body[0] == CRSF_TYPE_RC and len(body) - 1 >= RC_PAYLOAD_LEN:
                    self.channels = decode_rc_payload(body[1:1 + RC_PAYLOAD_LEN])
                    self.frames_ok += 1
                    got = True
                del self._buf[:end]
            else:
                self.frames_bad += 1
                del self._buf[:1]
        return got
