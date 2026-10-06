"""BboxTracker 单测：确认、门限、外推与丢锁。"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.bbox_tracker import BboxTracker  # noqa: E402


def box(cx, cy, w, h=20.0):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def test_requires_confirmation():
    t = BboxTracker(confirm_frames=2)
    assert t.update(box(100, 100, 10)).confirmed is False
    assert t.update(box(102, 100, 11)).confirmed is True


def test_coasts_with_prediction_then_loses():
    t = BboxTracker(confirm_frames=1, max_coast_frames=3)
    for k in range(4):
        t.update(box(100 + 5 * k, 100, 10 + 2 * k))
    out = t.update(None)                      # 第 1 帧漏检
    assert out.coasting and out.reason == "coast"
    assert out.bbox[0] > box(115, 100, 16)[0]  # 已按速度外推
    for _ in range(2):
        assert t.update(None).bbox is not None
    assert t.update(None).bbox is None         # 超过最大外推 → 丢锁


def test_gate_rejects_jump_boxes():
    t = BboxTracker(confirm_frames=1, gate_px=50.0, max_coast_frames=2)
    t.update(box(300, 200, 40))
    out = t.update(box(80, 400, 12))           # 远处的假框
    assert out.reason == "gated" and out.bbox is not None
    assert abs((out.bbox[0] + out.bbox[2]) / 2 - 300) < 30


def test_size_growth_is_extrapolated():
    t = BboxTracker(confirm_frames=1, max_coast_frames=4)
    for k in range(5):
        t.update(box(320, 240, 20 + 10 * k))
    w0 = t.update(None).bbox
    w1 = t.update(None).bbox
    assert (w1[2] - w1[0]) > (w0[2] - w0[0])   # 尺寸继续按增长率外推


def test_reset_clears_state():
    t = BboxTracker(confirm_frames=1)
    t.update(box(100, 100, 10))
    t.reset()
    assert t.update(None).bbox is None
