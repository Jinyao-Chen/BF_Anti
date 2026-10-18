"""
Visual anti-drone tracking controller — 纯 PBVS 计算, 无 IO.

输入: 每视觉帧的检测/追踪结果
输出: TrackCmd(RPYT + 时间戳) -> rc_mux.RCMux
engage()/disengage() 由 RCMux 回调 (开关/接管/断线在 mux 层处理)。
控制器没有任何 DISARM 能力 — ARM 通道永远由飞手掌握。
"""

import time
import math

import numpy as np

from rc_mux import TrackCmd


# ─── PID Controller ─────────────────────────────────────────────────────

class PID:
    """PID with dt clamping, integral anti-windup, D low-pass, output clamp."""

    DT_MIN = 0.005
    DT_MAX = 0.2

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, limit=300, d_lpf_alpha=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = limit
        self.d_lpf_alpha = d_lpf_alpha
        self._integral = 0.0
        self._prev_error = 0.0
        self._d_filt = 0.0
        self._first = True

    def update(self, error, dt):
        dt = max(self.DT_MIN, min(self.DT_MAX, dt))
        if self._first:
            self._prev_error = error
            self._first = False

        self._integral += error * dt
        self._integral = max(-self.limit, min(self.limit, self._integral))

        d_raw = (error - self._prev_error) / dt
        self._d_filt += self.d_lpf_alpha * (d_raw - self._d_filt)
        self._prev_error = error

        output = (self.kp * error + self.ki * self._integral
                  + self.kd * self._d_filt)
        return max(-self.limit, min(self.limit, output))

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._d_filt = 0.0
        self._first = True


# ─── Target Selector ────────────────────────────────────────────────────

class TargetSelector:
    """Pick best tracking target using priority 3->1->2.

    Priority 3: most stable confirmed track (largest age)
    Priority 1: highest confidence raw detection (score > threshold)
    Priority 2: nearest raw detection (largest box area)
    """

    def __init__(self, conf_thresh=0.6, min_track_age=5, area_conf_thresh=0.4):
        self.conf_thresh = conf_thresh
        self.min_track_age = min_track_age
        self.area_conf_thresh = area_conf_thresh

    def select(self, tracker, raw_boxes, raw_scores, raw_cls_ids):
        """Returns (box_xyxy, confidence, source) or None."""
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
            if raw_scores[best_idx] >= self.area_conf_thresh:
                return (raw_boxes[best_idx].tolist(),
                        float(raw_scores[best_idx]), "nearest")

        return None


# ─── Tracking Controller ────────────────────────────────────────────────

class TrackingController:
    """PBVS 追踪状态机: MANUAL -> ARMED -> TRACKING <-> LOST.

    LOST 超时 -> MANUAL 并停止输出 (mux 因指令过期自动回透传, 不 disarm)。
    """

    RC_LIMITS = {
        'roll':  (1200, 1800),
        'pitch': (1200, 1800),
        'yaw':   (1100, 1900),
        'thr':   (1050, 1800),
    }
    CONF_THRESHOLD = 0.6
    AREA_CONF_THRESHOLD = 0.4
    MIN_TRACK_AGE = 5
    SEARCH_FLIP_PERIOD = 3.0
    SEARCH_YAW_OFFSET = 60

    def __init__(self, pid_yaw=(2.0, 0.1, 0.5, 300),
                 pid_pitch=(2.5, 0.1, 0.5, 300),
                 pid_thr=(1.5, 0.05, 0.3, 200),
                 rate_limit=50, target_area_ratio=0.05,
                 lost_timeout=2.0, max_lost_time=5.0,
                 attitude_fn=None):
        self.selector = TargetSelector(
            conf_thresh=self.CONF_THRESHOLD,
            min_track_age=self.MIN_TRACK_AGE,
            area_conf_thresh=self.AREA_CONF_THRESHOLD,
        )
        self.pid_yaw = PID(*pid_yaw[:3], limit=int(pid_yaw[3]))
        self.pid_pitch = PID(*pid_pitch[:3], limit=int(pid_pitch[3]))
        self.pid_thr = PID(*pid_thr[:3], limit=int(pid_thr[3]))

        self.RATE_LIMIT = rate_limit
        self.TARGET_AREA_RATIO = target_area_ratio
        self.LOST_TIMEOUT = lost_timeout
        self.MAX_LOST_TIME = max_lost_time
        self.attitude_fn = attitude_fn

        self.state = "MANUAL"
        self._base_throttle = 1100
        self._lost_since = 0.0
        self._last_step_time = 0.0
        self._search_dir = 1
        self._last_ex = 0.0
        self._prev_cmd = None

    def engage(self, base_throttle, now=None):
        now = time.monotonic() if now is None else now
        self._base_throttle = int(base_throttle)
        self._last_step_time = 0.0
        self._lost_since = 0.0
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.pid_thr.reset()
        self._prev_cmd = TrackCmd(pitch=1500, yaw=1500,
                                  throttle=self._base_throttle, t=now)
        self.state = "ARMED"

    def disengage(self):
        self.state = "MANUAL"

    def step(self, frame, raw_boxes, raw_scores, raw_cls_ids, tracker, now=None):
        now = time.monotonic() if now is None else now
        if self.state == "MANUAL":
            return None

        fh, fw = frame.shape[:2]
        result = self.selector.select(tracker, raw_boxes, raw_scores, raw_cls_ids)

        if self.state == "ARMED":
            if result is None:
                return None
            self._to_tracking(now)

        elif self.state == "TRACKING":
            if result is None:
                if self._lost_since == 0.0:
                    self._lost_since = now
                if now - self._lost_since > self.LOST_TIMEOUT:
                    self._enter_lost(now)
                    return self._search_cmd(now)
                return self._coast_cmd(now)
            self._lost_since = 0.0

        elif self.state == "LOST":
            if result is not None:
                self._to_tracking(now)
            elif now - self._lost_since > self.MAX_LOST_TIME:
                self.state = "MANUAL"
                return None
            else:
                return self._search_cmd(now)

        return self._pbvs(now, result, fw, fh)

    def _to_tracking(self, now):
        self.state = "TRACKING"
        self._lost_since = 0.0
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.pid_thr.reset()
        self._last_step_time = now

    def _enter_lost(self, now):
        self.state = "LOST"
        self._lost_since = now
        self._search_dir = 1 if self._last_ex >= 0 else -1

    def _pbvs(self, now, result, fw, fh):
        box, confidence, source = result
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        box_area = (x2 - x1) * (y2 - y1)

        ex = cx - fw / 2.0
        ey = cy - fh / 2.0
        self._last_ex = ex
        area_target = fw * fh * self.TARGET_AREA_RATIO
        area_error = area_target - box_area

        ex_norm = ex / (fw / 2.0)
        ey_norm = ey / (fh / 2.0)

        dt = (now - self._last_step_time) if self._last_step_time > 0 else 1.0 / 30.0
        self._last_step_time = now

        yaw_out = self.pid_yaw.update(ex_norm * 500, dt)
        pitch_out = self.pid_pitch.update(-ey_norm * 500, dt)
        thr_out = self.pid_thr.update(area_error / 10000.0, dt)

        yaw_cmd = 1500 + yaw_out
        pitch_cmd = 1500 + pitch_out

        pitch_rad = 0.0
        if self.attitude_fn:
            att = self.attitude_fn()
            if att is not None and (now - att[3]) < 0.5:
                pitch_rad = math.radians(att[1])
        cos_pitch = max(0.3, math.cos(pitch_rad))
        thr_cmd = (self._base_throttle + thr_out) / cos_pitch

        yaw_cmd = self._clamp_axis('yaw', yaw_cmd)
        pitch_cmd = self._clamp_axis('pitch', pitch_cmd)
        thr_lo = self.RC_LIMITS['thr'][0]
        thr_hi = min(self.RC_LIMITS['thr'][1], self._base_throttle + 250)
        thr_cmd = max(thr_lo, min(thr_hi, thr_cmd))

        cmd = TrackCmd(
            pitch=self._rate_limited(pitch_cmd, self._prev_cmd.pitch),
            yaw=self._rate_limited(yaw_cmd, self._prev_cmd.yaw),
            throttle=self._rate_limited(thr_cmd, self._prev_cmd.throttle),
            t=now,
        )
        self._prev_cmd = cmd
        return cmd

    def _search_cmd(self, now):
        elapsed = now - self._lost_since
        phase = int(elapsed / self.SEARCH_FLIP_PERIOD)
        direction = self._search_dir * (1 if phase % 2 == 0 else -1)
        yaw = self._clamp_axis('yaw', 1500 + direction * self.SEARCH_YAW_OFFSET)
        thr_lo, thr_hi = self.RC_LIMITS['thr']
        thr = max(thr_lo, min(thr_hi, self._base_throttle))
        cmd = TrackCmd(pitch=1500,
                       yaw=self._rate_limited(yaw, self._prev_cmd.yaw),
                       throttle=self._rate_limited(thr, self._prev_cmd.throttle),
                       t=now)
        self._prev_cmd = cmd
        return cmd

    def _coast_cmd(self, now):
        cmd = TrackCmd(pitch=self._prev_cmd.pitch, yaw=self._prev_cmd.yaw,
                       throttle=self._prev_cmd.throttle, t=now)
        self._prev_cmd = cmd
        return cmd

    def _clamp_axis(self, axis, v):
        lo, hi = self.RC_LIMITS[axis]
        return max(lo, min(hi, v))

    def _rate_limited(self, new, prev):
        return int(max(prev - self.RATE_LIMIT, min(prev + self.RATE_LIMIT, new)))
