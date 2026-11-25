"""MSP link over serial (real FC) or TCP (Betaflight SITL).

The link is deliberately dumb: it moves validated MSP v1/v2 frames and never
decides which RC values are safe.  rc_mux.py is the single RC writer.

Threading contract:
* query() serialises request/response exchanges (one outstanding query).
* set_rc()/send_rc_16() only take the write lock, so the ~50 Hz RC stream
  never waits for a telemetry reply (a stalled reply must not trip the FC
  failsafe).
* Reconnect is lazy and rate-limited; generation increments on every successful
  (re)connect so FcState consumers can detect a firmware restart.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from typing import Optional, Sequence

from msp import (
    MSP_CMD,
    FrameParser,
    build_frame,
    decode_motor_channels,
)

__all__ = [
    "LinkError",
    "SerialTransport",
    "TcpTransport",
    "MspLink",
    "make_link",
]


class LinkError(RuntimeError):
    """Transport-level failure (open/read/write/closed by peer)."""


class SerialTransport:
    """pyserial-backed transport. Imported lazily so TCP/SITL works without it."""

    def __init__(self, port: str, baud: int = 115200, read_timeout_s: float = 0.05,
                 write_timeout_s: float = 0.05):
        self.port = str(port)
        self.baud = int(baud)
        self.read_timeout_s = float(read_timeout_s)
        self.write_timeout_s = float(write_timeout_s)
        self._ser = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and bool(getattr(self._ser, "is_open", False))

    def open(self) -> None:
        if self.is_open:
            return
        try:
            import serial  # pyserial
        except ImportError as exc:  # pragma: no cover - depends on host
            raise LinkError("pyserial is required for a serial MSP link") from exc
        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.read_timeout_s,
                write_timeout=self.write_timeout_s,
            )
        except (serial.SerialException, OSError, FileNotFoundError) as exc:
            self._ser = None
            raise LinkError("cannot open serial port %s: %s" % (self.port, exc)) from exc

    def read(self, timeout_s: float) -> bytes:
        ser = self._ser
        if ser is None or not bool(getattr(ser, "is_open", False)):
            raise LinkError("serial port is not open")
        ser.timeout = max(0.001, float(timeout_s))
        try:
            waiting = int(getattr(ser, "in_waiting", 0) or 0)
            return ser.read(waiting if waiting > 0 else 1)
        except Exception as exc:
            raise LinkError("serial read failed: %s" % exc) from exc

    def write(self, data: bytes) -> None:
        ser = self._ser
        if ser is None or not bool(getattr(ser, "is_open", False)):
            raise LinkError("serial port is not open")
        try:
            ser.write(data)
        except Exception as exc:
            raise LinkError("serial write failed: %s" % exc) from exc

    def close(self) -> None:
        ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass


class TcpTransport:
    """TCP transport for Betaflight SITL (MSP on 127.0.0.1:5761)."""

    def __init__(self, host: str, port: int = 5761, connect_timeout_s: float = 1.0):
        self.host = str(host)
        self.port = int(port)
        self.connect_timeout_s = float(connect_timeout_s)
        self._sock = None

    @property
    def is_open(self) -> bool:
        return self._sock is not None

    def open(self) -> None:
        if self._sock is not None:
            return
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout_s
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(self.connect_timeout_s)
            self._sock = sock
        except OSError as exc:
            self._sock = None
            raise LinkError(
                "cannot connect MSP TCP %s:%s: %s" % (self.host, self.port, exc)
            ) from exc

    def read(self, timeout_s: float) -> bytes:
        sock = self._sock
        if sock is None:
            raise LinkError("TCP link is not open")
        sock.settimeout(max(0.001, float(timeout_s)))
        try:
            data = sock.recv(4096)
        except socket.timeout:
            return b""
        except OSError as exc:
            raise LinkError("TCP read failed: %s" % exc) from exc
        if not data:
            raise LinkError("MSP TCP connection closed by peer")
        return data

    def write(self, data: bytes) -> None:
        sock = self._sock
        if sock is None:
            raise LinkError("TCP link is not open")
        try:
            sock.sendall(data)
        except OSError as exc:
            raise LinkError("TCP write failed: %s" % exc) from exc

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


class MspLink:
    """Thread-safe MSP link with lazy reconnect and a generation counter."""

    def __init__(
        self,
        transport,
        reply_timeout_s: float = 0.15,
        reconnect_interval_s: float = 0.5,
    ):
        self._transport = transport
        self.reply_timeout_s = max(0.001, float(reply_timeout_s))
        self.reconnect_interval_s = max(0.0, float(reconnect_interval_s))
        self._parser = FrameParser()
        self._write_lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._connect_lock = threading.Lock()
        self._closed = False
        self._connected = False
        self._generation = 0
        self._last_connect_attempt = float("-inf")
        self.errors = 0
        self.tx_frames = 0
        self.rx_frames = 0
        self.last_rx_t = 0.0
        self.last_error = ""
        self.last_rc = None

    # -- state -------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return bool(self._connected)

    @property
    def generation(self) -> int:
        return int(self._generation)

    @property
    def transport(self):
        return self._transport

    def connect(self) -> bool:
        """Force one connection attempt now (used by conformance/bench tools)."""
        self._last_connect_attempt = float("-inf")
        return self._ensure_connected()

    def _ensure_connected(self) -> bool:
        if self._connected:
            return True
        if self._closed:
            return False
        now = time.monotonic()
        if now - self._last_connect_attempt < self.reconnect_interval_s:
            return False
        with self._connect_lock:
            if self._connected:
                return True
            if self._closed:
                return False
            self._last_connect_attempt = time.monotonic()
            try:
                self._transport.open()
            except Exception as exc:
                self.errors += 1
                self.last_error = str(exc)
                return False
            self._connected = True
            self._generation += 1
            self._parser.reset()
            return True

    def _drop(self, exc, expected_gen: Optional[int] = None) -> None:
        with self._connect_lock:
            if expected_gen is not None and self._generation != expected_gen:
                return
            self._connected = False
            self.errors += 1
            self.last_error = str(exc)
            try:
                self._transport.close()
            except Exception:
                pass

    # -- low-level IO ------------------------------------------------------
    def _write_frame(self, frame: bytes) -> bool:
        with self._write_lock:
            if not self._ensure_connected():
                return False
            gen = self._generation
            try:
                self._transport.write(frame)
            except Exception as exc:
                self._drop(exc, expected_gen=gen)
                return False
            self.tx_frames += 1
            return True

    def send(self, frame: bytes) -> bool:
        """Write a pre-built MSP frame (bytes). Returns False on link failure."""
        return self._write_frame(bytes(frame))

    def query(self, cmd: int, is_v2: bool = False, timeout: Optional[float] = None,
              payload: bytes = b"") -> Optional[bytes]:
        """Send one command and return its payload, or None on timeout/error."""
        frame = build_frame(int(cmd), payload, is_v2=bool(is_v2))
        timeout_s = self.reply_timeout_s if timeout is None else max(0.0, float(timeout))
        with self._query_lock:
            if not self._write_frame(frame):
                return None
            gen = self._generation
            if timeout_s <= 0.0:
                return None
            deadline = time.monotonic() + timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                try:
                    chunk = self._transport.read(min(0.05, remaining))
                except Exception as exc:
                    self._drop(exc, expected_gen=gen)
                    return None
                if not chunk:
                    continue
                for reply in self._parser.feed(chunk):
                    self.rx_frames += 1
                    self.last_rx_t = time.monotonic()
                    if reply.cmd != int(cmd):
                        continue
                    if reply.error:
                        self.errors += 1
                        self.last_error = "MSP command %d returned error" % int(cmd)
                        return None
                    return reply.payload
            self.errors += 1
            self.last_error = "timeout waiting for MSP command %d" % int(cmd)
            return None

    # -- RC ----------------------------------------------------------------
    def send_rc_16(self, channels: Sequence[int]) -> bool:
        """Send MSP_SET_RAW_RC with 16 channels (AETR + AUX1..AUX12)."""
        values = decode_motor_channels(channels)
        payload = struct.pack("<16H", *values)
        frame = build_frame(MSP_CMD.SET_RAW_RC, payload, is_v2=False)
        self.last_rc = values
        return self._write_frame(frame)

    def set_rc(self, roll=1500, pitch=1500, throttle=1000, yaw=1500,
               aux1=1500, aux2=1500, aux3=1500, aux4=1500) -> bool:
        """Compatibility writer used by the RCMux bench path.

        Sends all 16 channels so the FC rxMspFrameReceive() does not zero
        channels 9..16.  Callers that own live pilot AUX data should call
        send_rc_16() and pass it through instead.
        """
        return self.send_rc_16([roll, pitch, throttle, yaw,
                                aux1, aux2, aux3, aux4])

    def close(self) -> None:
        with self._connect_lock:
            self._closed = True
            self._connected = False
            try:
                self._transport.close()
            except Exception:
                pass


def make_link(cfg) -> MspLink:
    """Build a link from onboard.config.FcConfig (TCP if tcp_host is set)."""
    host = str(getattr(cfg, "tcp_host", "") or "").strip()
    reply_timeout = float(getattr(cfg, "reply_timeout_s", 0.15) or 0.15)
    if host:
        transport = TcpTransport(host, int(getattr(cfg, "tcp_port", 5761)))
    else:
        transport = SerialTransport(
            str(getattr(cfg, "port", "/dev/ttyACM0")),
            int(getattr(cfg, "baud", 115200)),
        )
    return MspLink(transport, reply_timeout_s=reply_timeout)
