"""Hailo detector timing regressions.

The real Pi path runs inference asynchronously.  These tests exercise the
publish step directly (no Hailo device required): a blank frame must not erase
the last good detection, detection age must be measured from capture time, and
the OSD must get the frame that produced the box.
"""
import numpy as np

from onboard.config import DetectorConfig
from onboard.detector.hailo import HailoDetector
from onboard.types import Detection


def make_det(t):
    return Detection(bbox=(100.0, 120.0, 180.0, 200.0), conf=0.8, class_id=0, t=t)


def make_detector():
    det = HailoDetector(DetectorConfig(backend="hailo", hef="/nonexistent.hef"))
    return det


def test_blank_frame_keeps_last_detection_until_max_age():
    det = make_detector()
    frame = np.zeros((8, 8, 3), np.uint8)
    first = make_det(10.0)
    det._publish(frame, [first], first, 10.0, 10.02, 5.0)

    assert det.latest(0.4, now=10.05) is first
    assert det.latest_pair(0.4, now=10.05) == (frame, first)

    # A later inference found nothing.  The box must still be available for
    # tracker coasting, but the per-frame list is empty for the HUD.
    det._publish(np.ones_like(frame), [], None, 10.06, 10.08, 5.0)
    assert det.latest(0.4, now=10.10) is first
    assert det.latest_detections(0.4, now=10.10) == []
    assert det.latest_pair(0.4, now=10.10) == (frame, first)

    # Past max_age the old box is gone.
    assert det.latest(0.4, now=10.60) is None
    assert det.latest_pair(0.4, now=10.60) == (None, None)
    det.stop()


def test_detection_age_uses_capture_stamp_not_publish_time():
    det = make_detector()
    frame = np.zeros((8, 8, 3), np.uint8)
    d = make_det(20.0)
    # Inference finished 80 ms after capture.
    det._publish(frame, [d], d, 20.0, 20.08, 80.0)
    assert abs(det.detection_age(now=20.10) - 0.10) < 1e-6
    det.stop()


def test_successful_frame_replaces_pair_and_stats_time():
    det = make_detector()
    f1 = np.zeros((8, 8, 3), np.uint8)
    f2 = np.ones((8, 8, 3), np.uint8)
    d1 = make_det(30.0)
    d2 = Detection(bbox=(10.0, 10.0, 30.0, 30.0), conf=0.9, class_id=0, t=30.1)
    det._publish(f1, [d1], d1, 30.0, 30.02, 4.0)
    det._publish(f2, [d2], d2, 30.1, 30.12, 4.0)
    assert det.latest(1.0, now=30.2) is d2
    assert det.latest_pair(1.0, now=30.2) == (f2, d2)
    assert det.stats().last_det_t == 30.1
    det.stop()


def test_next_frame_deduplicates_same_camera_stamp():
    class Source:
        def __init__(self):
            self.items = [(np.zeros((4, 4, 3), np.uint8), 10.0),
                          (np.ones((4, 4, 3), np.uint8), 10.0),
                          (np.ones((4, 4, 3), np.uint8), 10.04)]

        def latest(self, max_age_s, now=None):
            return self.items.pop(0) if self.items else (None, 0.0)

    det = make_detector()
    det.attach_source(Source())
    frame, stamp = det._next_frame()
    assert stamp == 10.0
    # Same capture stamp must not be inferred twice.
    assert det._next_frame() == (None, 0.0)
    frame, stamp = det._next_frame()
    assert stamp == 10.04
    det.stop()
