"""
Visual anti-drone tracking controller — PBVS + MSP Override.

CRSF protocol reading → stick deviation detection
TargetSelector (priority 3→1→2) → PID controllers → MSP RC override

Architecture:
  ELRS RX TX ── FC UART (SERIAL RX) + RDK X5 UART (stick read only)
  RDK X5 MSP → FC USB → BOXMSPOVERRIDE → motor control
"""

import time
import math
import struct
import threading
from collections import deque

import serial
import numpy as np


# ─── CRSF Protocol Reader ───────────────────────────────────────────────

class CRSFReader:
    """Read ELRS/Crossfire RC channels from a UART.

    CRSF frame (RC channels, type 0x16):
      sync(0xC8) len type(0x16) payload(22B, 16ch×11bit packed) crc8
    """

    SYNC = 0xC8
    TYPE_RC = 0x16
    PAYLOAD_LEN = 22   # 16 channels × 11 bits = 176 bits = 22 bytes

    def __init__(self, port, baud=420000):
        self._ser = serial.Serial(port, baud, timeout=0.05)
        self._buf = bytearray()
        self._channels = [1500] * 16
        self._lock = threading.Lock()
        self._running = True
        self._last_update = 0.0        # 最后一次收到有效数据的时间
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def channels(self):
        """Return latest 16-channel list [1000-2000]."""
        with self._lock:
            return list(self._channels)

    def alive(self):
        """True if CRSF data received within last 1 second."""
        return (time.monotonic() - self._last_update) < 1.0

    def close(self):
        self._running = False
        self._ser.close()

    def _reader(self):
        while self._running:
            try:
                if self._ser.in_waiting:
                    self._buf.extend(self._ser.read(self._ser.in_waiting))
                # find sync
                while True:
                    pos = self._buf.find(bytes([self.SYNC]))
                    if pos < 0 or len(self._buf) < pos + 4:
                        break
                    length = self._buf[pos + 1]
                    frame_end = pos + 2 + length
                    if len(self._buf) < frame_end:
                        break
                    # verify header
                    if length >= 2 and self._buf[pos + 2] == self.TYPE_RC:
                        self._decode_rc(self._buf[pos + 3 : pos + 3 + min(length - 2, self.PAYLOAD_LEN)])
                    self._buf = self._buf[frame_end:]
            except Exception:
                self._buf.clear()
            # small sleep to avoid busy-loop if serial errors
            if not self._ser.in_waiting:
                time.sleep(0.001)

    @staticmethod
    def _crsf_to_rc(raw):
        """Convert CRSF raw (172-1811) to Betaflight RC (1000-2000)."""
        return int(1000 + (raw - 172) * 1000 / (1811 - 172))

    def _decode_rc(self, payload):
        """Extract 11-bit channels from 22-byte payload."""
        bits = int.from_bytes(payload[:self.PAYLOAD_LEN], 'little')
        ch = [1500] * 16
        for i in range(min(16, self.PAYLOAD_LEN * 8 // 11)):
            raw = (bits >> (i * 11)) & 0x7FF
            ch[i] = self._crsf_to_rc(raw)
        with self._lock:
            self._channels = ch
            self._last_update = time.monotonic()


# ─── PID Controller ─────────────────────────────────────────────────────

class PID:
    """Simple PID with integral windup protection and output clamping."""

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, limit=300):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True

    def update(self, error, dt):
        if dt <= 0:
            dt = 0.033  # default ~30fps
        if self._first:
            self._prev_error = error
            self._first = False

        self._integral += error * dt
        # anti-windup: clamp integral
        self._integral = max(-self.limit, min(self.limit, self._integral))

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.limit, min(self.limit, output))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._first = True


# ─── Target Selector ────────────────────────────────────────────────────

class TargetSelector:
    """Pick best tracking target using priority 3→1→2.

    Priority 3: most stable confirmed track (largest age)
    Priority 1: highest confidence raw detection (score > threshold)
    Priority 2: nearest raw detection (largest box area)
    """

    def __init__(self, conf_thresh=0.6, min_track_age=5):
        self.conf_thresh = conf_thresh
        self.min_track_age = min_track_age

    def select(self, tracker, raw_boxes, raw_scores, raw_cls_ids):
        """Returns (box_xyxy, confidence, source) or None.

        Args:
            tracker: Tracker instance (has .tracks attribute)
            raw_boxes:  (N,4) array
            raw_scores: (N,) array
            raw_cls_ids:(N,) array
        """
        # Priority 3: stable confirmed track
        if tracker is not None:
            confirmed = [t for t in tracker.tracks
                         if t.is_confirmed and t.age >= self.min_track_age]
            if confirmed:
                best = max(confirmed, key=lambda t: (t.age, t.score))
                return (best.box.tolist(), best.score, "track")

        # Priority 1: highest confidence
        if len(raw_scores):
            best_idx = int(np.argmax(raw_scores))
            if raw_scores[best_idx] >= self.conf_thresh:
                return (raw_boxes[best_idx].tolist(),
                        float(raw_scores[best_idx]), "conf")

        # Priority 2: nearest (largest box)
        if len(raw_boxes):
            areas = (raw_boxes[:, 2] - raw_boxes[:, 0]) * \
                    (raw_boxes[:, 3] - raw_boxes[:, 1])
            best_idx = int(np.argmax(areas))
            if raw_scores[best_idx] >= self.conf_thresh:
                return (raw_boxes[best_idx].tolist(),
                        float(raw_scores[best_idx]), "nearest")

        return None


# ─── Tracking Controller ────────────────────────────────────────────────

class TrackingController:
    """PBVS tracking loop with MSP Override orchestration.

    State machine:
      MANUAL ──(engage)──▶ ARMED ──(target acquired)──▶ TRACKING
      TRACKING ──(target lost >2s)──▶ LOST
      TRACKING ──(stick move/AUX3 off)──▶ MANUAL
      LOST ──(target re-acquired)──▶ TRACKING
      LOST ──(lost >5s)──▶ MANUAL (hover)
    """

    # safety limits
    RC_LIMITS = {
        'roll':  (1200, 1800),
        'pitch': (1200, 1800),
        'yaw':   (1100, 1900),
        'thr':   (1050, 1600),   # 上限 1600 (从 1850 降低)
    }
    STICK_DEVIATION = 0.30   # 30% from center → pilot takeover
    CONF_THRESHOLD = 0.6     # minimum confidence to engage
    MIN_TRACK_AGE = 5        # frames before track is "stable"
    LOST_TIMEOUT = 2.0       # seconds before entering search mode
    MAX_LOST_TIME = 5.0      # seconds in search before giving up
    ENGAGE_RAMP = 0.5        # seconds to fade in tracking control
    RATE_LIMIT = 50          # max RC change per frame (anti-jerk)
    TARGET_AREA_RATIO = 0.05  # target box should be ~5% of frame

    def __init__(self, msp, crsf_port, base_throttle=1100,
                 pid_yaw=(2.0, 0.1, 0.5, 300),
                 pid_pitch=(2.5, 0.1, 0.5, 300),
                 pid_thr=(1.5, 0.05, 0.3, 200),
                 rate_limit=50, target_area_ratio=0.05,
                 lost_timeout=2.0, max_lost_time=5.0):
        """
        Args:
            msp: MSP instance (already connected to FC)
            crsf_port: e.g. '/dev/ttyS3' for ELRS CRSF stream
            base_throttle: idle/hover throttle (1100=bench safe, 1550=flight hover)
            pid_yaw: (kp, ki, kd, limit) for yaw PID
            pid_pitch: (kp, ki, kd, limit) for pitch PID
            pid_thr: (kp, ki, kd, limit) for throttle PID
            rate_limit: max RC change per frame (anti-jerk)
            target_area_ratio: target box should be ~5% of frame
            lost_timeout: seconds before entering search mode
            max_lost_time: seconds in search before giving up
        """
        self.msp = msp
        self.crsf = CRSFReader(crsf_port) if crsf_port else None
        self.selector = TargetSelector(
            conf_thresh=self.CONF_THRESHOLD,
            min_track_age=self.MIN_TRACK_AGE,
        )

        # PIDs — fully configurable
        self.pid_yaw   = PID(kp=pid_yaw[0], ki=pid_yaw[1], kd=pid_yaw[2], limit=int(pid_yaw[3]))
        self.pid_pitch = PID(kp=pid_pitch[0], ki=pid_pitch[1], kd=pid_pitch[2], limit=int(pid_pitch[3]))
        self.pid_thr   = PID(kp=pid_thr[0], ki=pid_thr[1], kd=pid_thr[2], limit=int(pid_thr[3]))

        # Tuning overrides
        self.RATE_LIMIT = rate_limit
        self.TARGET_AREA_RATIO = target_area_ratio
        self.LOST_TIMEOUT = lost_timeout
        self.MAX_LOST_TIME = max_lost_time

        # state
        self.state = "MANUAL"          # MANUAL | ARMED | TRACKING | LOST
        self._engage_time = 0.0
        self._lost_since = 0.0
        self._last_step_time = 0.0     # for real dt computation
        self._search_dir = 1           # +1 = yaw right, -1 = yaw left
        self._prev_rc = {'roll': 1500, 'pitch': 1500, 'yaw': 1500, 'throttle': 1050, 'aux1': 2000}
        self._base_throttle = base_throttle

    def step(self, frame, raw_boxes, raw_scores, raw_cls_ids, tracker):
        """Process one detection frame. Returns dict of RC values or None.

        Call this in the YOLO main loop after tracker.step().
        When self.state == 'MANUAL', returns None (no override).
        When tracking, returns dict with 'roll','pitch','yaw','thr','aux3'.
        """
        now = time.monotonic()
        fh, fw = frame.shape[:2]

        # Select target
        result = self.selector.select(tracker, raw_boxes, raw_scores, raw_cls_ids)

        # Safety checks
        stick_override = self._stick_deviation()
        aux3_ok = True  # always ok — we set AUX3 ourselves

        # ── State machine ──────────────────────────────────────────
        if self.state == "MANUAL":
            return None

        elif self.state == "ARMED":
            if result is not None:
                self.state = "TRACKING"
                self._engage_time = now
                self._lost_since = 0.0
            else:
                return None  # keep ARMED, no target yet

        elif self.state == "TRACKING":
            if stick_override:
                self._disengage()
                return None
            if result is None:
                self._lost_since = now if self._lost_since == 0 else self._lost_since
                if now - self._lost_since > self.LOST_TIMEOUT:
                    self.state = "LOST"
                    self._lost_since = now
                    self._search_dir = 1
                    self.pid_yaw.reset()
                    self.pid_pitch.reset()
                    self.pid_thr.reset()
                # coast with last known command
                return self._prev_rc
            else:
                self._lost_since = 0.0

        elif self.state == "LOST":
            if stick_override:
                self._disengage()
                return None
            if result is not None:
                self.state = "TRACKING"
                self._engage_time = now
                self._lost_since = 0.0
                self.pid_yaw.reset()
            if now - self._lost_since > self.MAX_LOST_TIME:
                self._disengage()
                return None
            # search: slow yaw sweep
            return self._search_mode(now)

        # ── Compute PBVS commands ──────────────────────────────────
        box, confidence, source = result
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw, bh = x2 - x1, y2 - y1
        box_area = bw * bh

        # errors
        ex = cx - fw / 2.0        # + = target is right
        ey = cy - fh / 2.0        # + = target is below
        area_target = fw * fh * self.TARGET_AREA_RATIO
        area_error = area_target - box_area  # + = box too small

        # normalize for PID
        ex_norm = ex / (fw / 2.0)   # [-1, 1]
        ey_norm = ey / (fh / 2.0)   # [-1, 1]

        # real frame interval (with fallback for first frame)
        if self._last_step_time > 0:
            dt = now - self._last_step_time
        else:
            dt = 1.0 / 30.0  # first frame: assume 30fps
        self._last_step_time = now

        # PID outputs
        yaw_out   = self.pid_yaw.update(ex_norm * 500, dt)
        pitch_out = self.pid_pitch.update(-ey_norm * 500, dt)  # inverted
        thr_out   = self.pid_thr.update(area_error / 10000.0, dt)

        # base RC + PID
        yaw_cmd   = 1500 + yaw_out
        pitch_cmd = 1500 + pitch_out

        # throttle with pitch compensation
        try:
            att = self.msp.att()
            pitch_rad = math.radians(att[1]) if att else 0.0
        except Exception:
            pitch_rad = 0.0
        cos_pitch = max(0.3, math.cos(pitch_rad))  # floor at 0.3 (safety)
        thr_cmd = (self._base_throttle + thr_out) / cos_pitch

        # clamp to safe ranges
        yaw_cmd   = max(self.RC_LIMITS['yaw'][0],   min(self.RC_LIMITS['yaw'][1],   yaw_cmd))
        pitch_cmd = max(self.RC_LIMITS['pitch'][0], min(self.RC_LIMITS['pitch'][1], pitch_cmd))
        thr_cmd   = max(self.RC_LIMITS['thr'][0],   min(self.RC_LIMITS['thr'][1],   thr_cmd))

        # ramp engagement
        blend = min(1.0, (now - self._engage_time) / self.ENGAGE_RAMP)
        yaw_cmd   = 1500 + (yaw_cmd - 1500) * blend
        pitch_cmd = 1500 + (pitch_cmd - 1500) * blend
        thr_cmd   = self._base_throttle + (thr_cmd - self._base_throttle) * blend

        # rate limit (anti-jerk)
        yaw_cmd   = self._rate_limit(yaw_cmd,   self._prev_rc['yaw'])
        pitch_cmd = self._rate_limit(pitch_cmd, self._prev_rc['pitch'])
        thr_cmd   = self._rate_limit(thr_cmd,   self._prev_rc['throttle'])

        rc = {
            'roll': 1500,
            'pitch': int(pitch_cmd),
            'yaw': int(yaw_cmd),
            'throttle': int(thr_cmd),
            'aux1': 2000,   # ARM
            'aux3': 2000,   # BOXMSPOVERRIDE active
        }
        self._prev_rc = rc
        return rc

    def engage(self):
        """Arm tracking mode. Pilot must have sticks centered."""
        if self.state != "MANUAL":
            return False
        if self._stick_deviation():
            return False   # pilot is flying, won't engage
        self.state = "ARMED"
        self._engage_time = time.monotonic()
        self._last_step_time = 0.0   # reset dt tracking
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.pid_thr.reset()
        self._prev_rc = {'roll': 1500, 'pitch': 1500, 'yaw': 1500, 'throttle': 1050, 'aux1': 2000}
        return True

    def _disengage(self):
        self.state = "MANUAL"
        self._prev_rc = {'roll': 1500, 'pitch': 1500, 'yaw': 1500, 'throttle': 1050, 'aux1': 2000}
        # 关闭 ARM + BOXMSPOVERRIDE, 油门最低
        self.msp.set_rc(1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)
        # aux1=1500 → DISARM, aux3=1500 → 取消 override

    def _stick_deviation(self):
        """True if any control channel deviates >30% from center.
        Also returns True if CRSF has disconnected (safety: force disengage)."""
        if self.crsf is None:
            return False
        if not self.crsf.alive():
            return True   # CRSF 断线 → 强制退出追踪
        ch = self.crsf.channels()
        if not ch or len(ch) < 4:
            return False
        # CRSF channel order: typically Roll=0, Pitch=1, Thr=2, Yaw=3
        for idx in range(4):
            dev = abs(ch[idx] - 1500) / 500.0  # 500 = half RC range
            if dev > self.STICK_DEVIATION:
                return True
        return False

    def _search_mode(self, now):
        """Slow yaw sweep to find lost target."""
        search_rate = 30  # RC units per second
        yaw = 1500 + int(self._search_dir * search_rate * min(
            3.0, now - self._lost_since))
        yaw = max(1200, min(1800, yaw))
        thr_safe = max(self.RC_LIMITS['thr'][0],
                       min(self.RC_LIMITS['thr'][1], self._base_throttle))
        rc = {
            'roll': 1500, 'pitch': 1500, 'yaw': max(1200, min(1800, yaw)),
            'throttle': thr_safe, 'aux1': 2000, 'aux3': 2000,
        }
        self._prev_rc = rc
        return rc

    def _rate_limit(self, new, prev):
        return max(prev - self.RATE_LIMIT, min(prev + self.RATE_LIMIT, new))

    def close(self):
        if self.crsf:
            self.crsf.close()
