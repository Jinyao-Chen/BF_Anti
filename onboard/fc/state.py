"""Flight-controller telemetry poller.

FcStateEstimator owns the only background reader of the MSP link.  It polls at
cfg.telemetry_hz, decodes each payload with the pure helpers in msp.py, and
publishes an immutable-ish copy via snapshot()/rc_snapshot().

Rules:
* Never invent a value.  A field whose reply has not arrived keeps the FcState
  default and a zero timestamp; consumers check ages.
* yaw is whole degrees (MSP_ATTITUDE); roll/pitch are 0.1 deg.
* rpy_deg is aviation convention: +roll = right wing down, +pitch = nose up.
  MSP_ATTITUDE pitch is negated on ingest (Betaflight +pitch = nose down).
* Every field is timestamped when *its own reply* arrives, not when the poll
  started: a slow boot-time exchange must not publish a sample that is already
  older than the safety gate's max age.
* MSP_RC is converted to AETR (the FC sends R,P,Y,T).
* MSP2 0x3015 position state is optional; after a few misses it is disabled so
  an older FC does not pay a timeout on every poll.
"""
from __future__ import annotations

import collections
import dataclasses
import threading
import time

from msp import (
    MSP_CMD,
    decode_altitude,
    decode_analog,
    decode_attitude,
    decode_msp2_position_state,
    decode_rc,
    decode_status,
    decode_status_ex,
)
from onboard.types import FcState


class FcStateEstimator:
    """Background MSP telemetry poller for the onboard runtime."""

    # Poll one field every N ticks (attitude is every tick).
    ALTITUDE_EVERY = 2
    RC_EVERY = 2
    STATUS_EVERY = 4
    ANALOG_EVERY = 10
    POSITION_EVERY = 5
    POSITION_MAX_MISSES = 3

    def __init__(self, link, cfg):
        self.link = link
        self.cfg = cfg
        rate_hz = float(getattr(cfg, "telemetry_hz", 25.0) or 25.0)
        self.rate_hz = max(1.0, rate_hz)
        self._period = 1.0 / self.rate_hz
        self._timeout = float(getattr(cfg, "reply_timeout_s", 0.15) or 0.15)
        self._lock = threading.Lock()
        self._state = FcState()
        self._att_hist: collections.deque = collections.deque(maxlen=64)
        self._rc = None
        self._rc_t = 0.0
        self._errors = 0
        self._tick = 0
        self._position_misses = 0
        self._position_enabled = True
        self._status_ex_misses = 0
        self._legacy_status = False
        self._running = False
        self._thread = None

    # -- lifecycle ---------------------------------------------------------
    @property
    def running(self) -> bool:
        return bool(self._running)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name="fc-state", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)

    def _run(self) -> None:
        next_t = time.monotonic()
        while self._running:
            try:
                self._poll_once()
            except Exception:
                self._errors += 1
            next_t += self._period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()

    # -- polling -----------------------------------------------------------
    def _query(self, cmd, is_v2=False):
        try:
            return self.link.query(cmd, is_v2=is_v2, timeout=self._timeout)
        except TypeError:
            # Test doubles may implement the minimal query(cmd, is_v2, timeout).
            try:
                return self.link.query(cmd, is_v2=is_v2)
            except Exception:
                self._errors += 1
                return None
        except Exception:
            self._errors += 1
            return None

    def _link_stamp(self, fallback):
        stamp = float(getattr(self.link, "last_rx_t", 0.0) or 0.0)
        return stamp if stamp > 0.0 else float(fallback)

    def _poll_once(self) -> None:
        now = time.monotonic()
        tick = self._tick
        self._tick = tick + 1
        updates = {}

        # Attitude: every tick, published immediately so downstream timeouts
        # on secondary telemetry never delay attitude visibility.
        payload = self._query(MSP_CMD.GET_ATTITUDE)
        if payload is not None:
            attitude = decode_attitude(payload)
            if attitude is not None:
                # MSP_ATTITUDE follows Betaflight: +pitch is nose down.
                # FcState/guidance/OSD use aviation convention: +pitch is
                # nose up.  Roll (+right wing down) and yaw (+right turn)
                # already match.
                roll, pitch, yaw = attitude
                rpy = (roll, -pitch, yaw)
                att_stamp = time.monotonic()
                updates["rpy_deg"] = rpy
                updates["att_t"] = att_stamp
                with self._lock:
                    self._state.rpy_deg = rpy
                    self._state.att_t = att_stamp
                    self._att_hist.append((att_stamp, rpy))
            else:
                self._errors += 1

        # Stagger secondary queries across ticks so the old 6-query pileup
        # (tick % 20 == 0) cannot stall one control period.
        if tick % self.ALTITUDE_EVERY == 0:
            payload = self._query(MSP_CMD.GET_ALTITUDE)
            if payload is not None:
                altitude = decode_altitude(payload)
                if altitude is not None:
                    updates["altitude_m"] = altitude[0]
                    updates["vario_mps"] = altitude[1]
                    updates["alt_t"] = time.monotonic()
                else:
                    self._errors += 1

        if tick % self.RC_EVERY == 1:
            payload = self._query(MSP_CMD.GET_RC)
            if payload is not None:
                channels = decode_rc(payload)
                if channels is not None:
                    rc_stamp = time.monotonic()
                    updates["rc"] = channels
                    updates["rc_t"] = rc_stamp
                    self._rc = tuple(channels)
                    self._rc_t = rc_stamp
                elif len(payload) > 0:
                    # Short/corrupt reply: a decode error, not a 0-channel FC.
                    self._errors += 1
                # len(payload) == 0: FC receiver exposes no channels (e.g.
                # SITL RX_UDP).  Leave rc_t == 0 so consumers see "never
                # received" instead of inventing 16 neutral channels.

        if tick % self.STATUS_EVERY == 2:
            status = None
            if self._legacy_status:
                status_payload = self._query(MSP_CMD.GET_STATUS)
                status = decode_status(status_payload) if status_payload is not None else None
                if status is None and status_payload is not None:
                    self._errors += 1
            else:
                ex_payload = self._query(MSP_CMD.GET_STATUS_EX)
                if ex_payload is not None:
                    status = decode_status_ex(ex_payload)
                    if status is not None:
                        self._status_ex_misses = 0
                    else:
                        # A corrupt/unsupported STATUS_EX: try the legacy
                        # command once, but never on a timeout (that would
                        # double a 150 ms stall in the same tick).
                        status_payload = self._query(MSP_CMD.GET_STATUS)
                        status = decode_status(status_payload) if status_payload is not None else None
                        if status is None:
                            self._errors += 1
                else:
                    self._status_ex_misses += 1
                    if self._status_ex_misses >= 3:
                        self._legacy_status = True
            if status is not None:
                updates["armed"] = bool(status["armed"])
                updates["flight_mode_flags"] = int(status["flight_mode_flags"])
                updates["status_t"] = time.monotonic()

        if tick % self.ANALOG_EVERY == 3:
            payload = self._query(MSP_CMD.GET_ANALOG)
            if payload is not None:
                analog = decode_analog(payload)
                if analog is not None:
                    updates["voltage_v"] = float(analog["voltage_v"])
                    updates["current_a"] = float(analog["current_a"])
                    updates["mah"] = int(analog["mah"])
                    updates["rssi"] = int(analog["rssi"])
                    updates["analog_t"] = time.monotonic()
                else:
                    self._errors += 1

        if self._position_enabled and tick % self.POSITION_EVERY == 4:
            payload = self._query(MSP_CMD.MSP2_POSITION_STATE, is_v2=True)
            if payload is not None:
                position = decode_msp2_position_state(payload)
                if position is not None:
                    updates["position_enu"] = tuple(float(v) for v in position["position_enu"])
                    updates["velocity_enu"] = tuple(float(v) for v in position["velocity_enu"])
                    updates["pos_t"] = time.monotonic()
                    self._position_misses = 0
                else:
                    self._errors += 1
            else:
                self._position_misses += 1
                if self._position_misses >= self.POSITION_MAX_MISSES:
                    self._position_enabled = False

        link_stamp = self._link_stamp(now)
        with self._lock:
            state = self._state
            for key, value in updates.items():
                setattr(state, key, value)
            if updates:
                state.link_t = max(float(state.link_t), link_stamp)
            state.generation = int(getattr(self.link, "generation", 0) or 0)
            state.errors = int(getattr(self.link, "errors", 0) or 0) + self._errors

    # -- consumers ---------------------------------------------------------
    def snapshot(self, now=None) -> FcState:
        """Return a copy; safe to hand to the control loop at any time."""
        with self._lock:
            snap = dataclasses.replace(self._state)
        snap.generation = int(getattr(self.link, "generation", snap.generation) or 0)
        snap.errors = int(getattr(self.link, "errors", 0) or 0) + self._errors
        link_stamp = float(getattr(self.link, "last_rx_t", 0.0) or 0.0)
        if link_stamp > 0.0 and link_stamp > snap.link_t:
            snap.link_t = link_stamp
        return snap

    def attitude_at(self, query_t: float) -> tuple[float, float, float]:
        """Return linearly interpolated (roll, pitch, yaw) at query_t."""
        with self._lock:
            hist = list(self._att_hist)
            fallback = self._state.rpy_deg
        if not hist:
            return fallback
        if query_t <= hist[0][0]:
            return hist[0][1]
        if query_t >= hist[-1][0]:
            return hist[-1][1]
        for (t0, rpy0), (t1, rpy1) in zip(hist, hist[1:]):
            if t0 <= query_t <= t1:
                alpha = (query_t - t0) / max(1e-6, t1 - t0)
                r = rpy0[0] + alpha * (rpy1[0] - rpy0[0])
                p = rpy0[1] + alpha * (rpy1[1] - rpy0[1])
                dy = (rpy1[2] - rpy0[2] + 180.0) % 360.0 - 180.0
                y = (rpy0[2] + alpha * dy) % 360.0
                return (float(r), float(p), float(y))
        return hist[-1][1]

    def rc_snapshot(self, now=None):
        """Return (channels_or_None, age_s). age is inf when never received."""
        now = time.monotonic() if now is None else float(now)
        with self._lock:
            channels = self._rc
            rc_t = self._rc_t
        if channels is None or rc_t <= 0.0:
            return None, float("inf")
        return channels, max(0.0, now - rc_t)
