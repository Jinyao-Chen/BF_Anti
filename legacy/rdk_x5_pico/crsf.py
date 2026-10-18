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


class CRSFReader:
    """串口读线程. channels() 返回最新 16 通道或 None; age() 返回最后有效帧距今秒数."""

    def __init__(self, port, baud=420000):
        self._ser = serial.Serial(port, baud, timeout=0.02)
        self._parser = CRSFParser()
        self._lock = threading.Lock()
        self._channels = None
        self._last_valid = None
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def channels(self):
        with self._lock:
            return list(self._channels) if self._channels else None

    def age(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._last_valid is None:
                return float('inf')
            return now - self._last_valid

    def close(self):
        self._running = False
        self._thread.join(timeout=0.5)
        try:
            self._ser.close()
        except Exception:
            pass

    def _run(self):
        while self._running:
            try:
                data = self._ser.read(64)
            except Exception:
                if not self._running:
                    break
                time.sleep(0.05)
                continue
            if data and self._parser.feed(data):
                with self._lock:
                    self._channels = self._parser.channels
                    self._last_valid = time.monotonic()
