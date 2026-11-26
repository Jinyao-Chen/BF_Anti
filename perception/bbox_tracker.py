"""真检测框的跨帧调理：确认、门限关联、短时预测外推。

**只做测量调理，不改变制导律结构**（制导仍是纯比例）。解决的实测问题：

真 YOLO 逐次方差大（同一配置 0.53 m ~ 5.30 m），失败的几次都走同一条链：

    YOLO 连续漏检 -> 本模块用"冻住的外推框"维持 -> 制导律拿到的是**陈旧像素**
    配**实时姿态** -> 俯仰几何偏置变成正反馈 -> 0.4 s 顶到限幅、目标出画 -> 失锁

因此本模块除了"维持闭环"之外，还**明确标注每一帧是新鲜检测还是外推**：
TrackOut.fresh 为 False 时，制导律会冻结绝对几何量（见
PVisualGuidance.update(fresh=...)），避免陈旧像素驱动姿态。

三个机制：
1. **确认**：连续 N 帧通过才认为锁定（防单帧假框）；
2. **门限关联**：与预测位置距离超过 gate 的框视为离群，用预测继续，
   避免跳到画面里别的物体；门限随"最近像素速度"自适应（近距离快转时
   一帧就能移动上百像素，固定门限会把真框当假框剔掉）；
3. **短时外推**：漏检时按最近的框心速度与尺寸增长率外推 <= max_coast 帧，
   终端（框快撑满画面、YOLO 容易漏）尤其重要。速度按**真实时间**折算
   （检测帧率 13 Hz != 控制帧率 20 Hz，按控制周期外推会快 1.5 倍）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

BBox = Tuple[float, float, float, float]


@dataclass
class TrackOut:
    bbox: Optional[BBox]          # 调理后的框（可能由外推得到）
    confirmed: bool               # 是否已达到确认帧数
    coasting: bool                # 本帧是外推（无检测）
    streak: int                   # 连续有效帧数
    reason: str                   # det / coast / gated / lost

    @property
    def fresh(self) -> bool:
        """本帧是否由**真实检测**驱动（外推/门控帧为 False）。"""
        return self.reason == "det"


class BboxTracker:
    def __init__(self, confirm_frames: int = 2, max_coast_frames: int = 6,
                 gate_px: float = 90.0, gate_gain: float = 3.0,
                 max_coast_s: float = 0.0, image_width: int = 0,
                 gate_frac: float = 0.0,
                 coast_vel_tau_s: float = 0.0) -> None:
        self.confirm_frames = max(1, int(confirm_frames))
        self.max_coast_frames = max(0, int(max_coast_frames))
        self.max_coast_s = max(0.0, float(max_coast_s))
        self.gate_px = float(gate_px)
        self.gate_frac = max(0.0, float(gate_frac))
        self.image_width = max(0, int(image_width))
        self.gate_gain = float(gate_gain)
        # 0 keeps the historical per-frame 0.92 decay for callers that have no
        # wall-clock dt (unit tests / old sim scripts).
        self.coast_vel_tau_s = max(0.0, float(coast_vel_tau_s))
        self._miss_s = 0.0
        self.reset()

    def reset(self) -> None:
        self._c: Optional[Tuple[float, float, float, float]] = None   # cx,cy,w,h（工作估计）
        self._det: Optional[Tuple[float, float, float, float]] = None  # 最近一次**真实检测**
        self._det_t: Optional[float] = None
        self._v: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)   # px/s
        self._dt = 0.05
        self._miss = 0
        self._miss_s = 0.0
        self._streak = 0

    # ------------------------------------------------------------------ #
    def _bbox(self) -> Optional[BBox]:
        if self._c is None:
            return None
        cx, cy, w, h = self._c
        w = max(2.0, w)
        h = max(2.0, h)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def _advance(self) -> None:
        if self._c is None:
            return
        cx, cy, w, h = self._c
        vx, vy, vw, vh = self._v
        dt = self._dt
        self._c = (cx + vx * dt, cy + vy * dt, max(2.0, w + vw * dt), max(2.0, h + vh * dt))
        # Mild coasting velocity decay prevents runaway extrapolation on long
        # misses.  Use physical time when configured; keep the legacy 0.92 per
        # call for callers without wall-clock dt.
        if self.coast_vel_tau_s > 0.0 and dt > 0.0:
            decay = math.exp(-dt / self.coast_vel_tau_s)
        else:
            decay = 0.92
        self._v = (vx * decay, vy * decay, vw * decay, vh * decay)

    def _predicted_center(self) -> Tuple[float, float]:
        assert self._c is not None
        return (self._c[0] + self._v[0] * self._dt, self._c[1] + self._v[1] * self._dt)

    @property
    def center_velocity_px_s(self) -> Optional[Tuple[float, float]]:
        """Last measured bbox-center velocity in px/s, or None before lock."""
        if self._c is None or self._streak <= 0:
            return None
        return (float(self._v[0]), float(self._v[1]))

    @property
    def locked(self) -> bool:
        """True once the track reached confirm_frames (used by the blackbox)."""
        return self._c is not None and self._streak >= self.confirm_frames

    @property
    def streak(self) -> int:
        return int(self._streak)

    @property
    def misses(self) -> int:
        return int(self._miss)

    # ------------------------------------------------------------------ #
    def _gate_now(self) -> float:
        if self.max_coast_s > 0.0:
            growth = 1.0 + min(1.0, self._miss_s / max(0.05, self.max_coast_s))
        else:
            growth = 1.0 + 0.15 * self._miss
        base = max(self.gate_px, self.gate_frac * float(self.image_width))
        return max(
            base * growth,
            self.gate_gain * math.hypot(self._v[0], self._v[1]) * self._dt * max(1, self._miss + 1),
        )

    def update_candidates(self, candidates, dt: float = 1.0,
                          timestamp: Optional[float] = None) -> TrackOut:
        """Associate *all* detector boxes with the predicted target.

        The detector's highest-confidence box is not necessarily the tracked
        target (a bird/glint can outscore it for one frame).  When locked,
        choose the candidate nearest the predicted center inside the gate; if
        every candidate is outside, coast rather than jumping to a false box.
        """
        self._dt = max(1e-3, float(dt))
        boxes = []
        for det in candidates or ():
            try:
                box = tuple(float(v) for v in det.bbox[:4])
            except (AttributeError, TypeError, ValueError, IndexError):
                continue
            if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append((box, float(getattr(det, "conf", 0.0) or 0.0)))
        if not boxes:
            return self.update(None, dt=dt, timestamp=timestamp)
        if self._c is None or self._streak < self.confirm_frames:
            return self.update(max(boxes, key=lambda item: item[1])[0],
                               dt=dt, timestamp=timestamp)
        px, py = self._predicted_center()
        gate = self._gate_now()
        def _distance(item):
            box = item[0]
            return math.hypot((box[0] + box[2]) * 0.5 - px,
                              (box[1] + box[3]) * 0.5 - py)
        nearest = min(boxes, key=lambda item: (_distance(item), -item[1]))
        if _distance(nearest) > gate:
            return self.update(None, dt=dt, timestamp=timestamp)
        return self.update(nearest[0], dt=dt, timestamp=timestamp)

    def update(self, det: Optional[BBox], dt: float = 1.0,
               timestamp: Optional[float] = None) -> TrackOut:
        """一帧调理。

        dt        : 本次调用到上次调用的间隔（s），用于外推步长（默认 1.0 = 每帧）。
        timestamp : 单调时间戳；给出时用它算**真实检测间隔**（检测帧率 != 控制帧率）。
        """
        self._dt = max(1e-3, float(dt))
        t = None if timestamp is None else float(timestamp)
        if det is not None:
            cx = (float(det[0]) + float(det[2])) * 0.5
            cy = (float(det[1]) + float(det[3])) * 0.5
            w = float(det[2]) - float(det[0])
            h = float(det[3]) - float(det[1])
            if w <= 0.0 or h <= 0.0 or not all(map(math.isfinite, (cx, cy, w, h))):
                return self._miss_step("lost")

            if self._c is not None and self._streak >= self.confirm_frames:
                px, py = self._predicted_center()
                if self.max_coast_s > 0.0:
                    gate_growth = 1.0 + min(1.0, self._miss_s / max(0.05, self.max_coast_s))
                else:
                    gate_growth = 1.0 + 0.15 * self._miss
                base_gate = max(self.gate_px, self.gate_frac * float(self.image_width))
                miss_scale = max(1, self._miss + 1)
                gate = max(
                    base_gate * gate_growth,
                    self.gate_gain * math.hypot(self._v[0], self._v[1]) * self._dt * miss_scale,
                )
                if math.hypot(cx - px, cy - py) > gate:
                    # 跳框：不采信，用外推继续。  After the coast window, keep
                    # the current candidate as a reacquisition instead of
                    # discarding it and waiting for another frame.
                    self._miss += 1
                    self._miss_s += self._dt
                    if self._within_coast_window():
                        self._advance()
                        return TrackOut(self._bbox(), True, True, self._streak, "gated")
                    # Fall through: reinitialise on the current detection.
                    self._streak = 0
                    self._c = None
                    self._det = None
                    self._det_t = None
                    self._miss = 0
                    self._miss_s = 0.0

            # 速度按**真实检测间隔**折算成 px/s
            if self._det is not None:
                if t is not None and self._det_t is not None:
                    dtd = min(0.5, max(0.02, t - self._det_t))
                else:
                    dtd = self._dt
                raw_v = ((cx - self._det[0]) / dtd, (cy - self._det[1]) / dtd,
                         (w - self._det[2]) / dtd, (h - self._det[3]) / dtd)
                # Single-frame width-rate is extremely noisy and grows
                # unbounded when a box clips an edge.  Clamp and smooth it;
                # center velocity stays responsive for gating.
                max_w_rate = max(50.0, 0.5 * max(w, self._det[2]) / dtd)
                raw_v = (raw_v[0], raw_v[1],
                         max(-max_w_rate, min(max_w_rate, raw_v[2])),
                         max(-max_w_rate, min(max_w_rate, raw_v[3])))
                if self._det is not None and self._v != (0.0, 0.0, 0.0, 0.0):
                    a = 0.35
                    self._v = tuple((1.0 - a) * old + a * new
                                    for old, new in zip(self._v, raw_v))
                else:
                    self._v = raw_v
            else:
                self._v = (0.0, 0.0, 0.0, 0.0)
            self._det = (cx, cy, w, h)
            self._det_t = t if t is not None else self._det_t
            self._c = (cx, cy, w, h)
            self._miss = 0
            self._miss_s = 0.0
            self._streak += 1
            return TrackOut(self._bbox(), self._streak >= self.confirm_frames,
                            False, self._streak, "det")

        return self._miss_step("coast")

    def _within_coast_window(self) -> bool:
        if self.max_coast_s > 0.0:
            return self._miss_s <= self.max_coast_s
        return self._miss <= self.max_coast_frames

    def _miss_step(self, reason: str) -> TrackOut:
        self._miss += 1
        self._miss_s += self._dt
        if (
            reason != "lost"
            and self._c is not None
            and self._within_coast_window()
            and self._streak > 0
        ):
            self._advance()
            return TrackOut(self._bbox(), self._streak >= self.confirm_frames,
                            True, self._streak, "coast")
        self._streak = 0
        self._c = None
        self._det = None
        self._det_t = None
        self._v = (0.0, 0.0, 0.0, 0.0)
        self._miss = 0
        self._miss_s = 0.0
        return TrackOut(None, False, False, 0, "lost")


__all__ = ["BboxTracker", "TrackOut", "BBox"]
