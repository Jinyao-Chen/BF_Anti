"""Async detector timing: capture age must not be reset by a failed frame."""
import threading
import time

import numpy as np

from sim.run_p_vision_strike import AsyncDetector


def _detector():
    d = AsyncDetector.__new__(AsyncDetector)
    d._lock = threading.Lock()
    d._bbox = (10.0, 10.0, 20.0, 20.0)
    d._bbox_t = time.monotonic() - 0.05
    d._frame_t = time.monotonic() - 0.20
    d._pair = (np.zeros((4, 4, 3), np.uint8), d._bbox)
    d._pair_t = time.monotonic() - 0.01
    return d


def test_age_uses_frame_capture_time():
    d = _detector()
    assert d.age() >= 0.15
    assert d.latest(max_age=0.5) is not None


def test_failed_detection_does_not_refresh_bbox_freshness():
    d = _detector()
    d._pair = (np.zeros((4, 4, 3), np.uint8), None)
    d._pair_t = time.monotonic()
    assert d.latest_pair()[1] is None
    # The last successful box is still old by its own timestamp.
    assert d.age() >= 0.15
    assert d.latest(max_age=0.01) is None
