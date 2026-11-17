"""Detector factory tests: null/replay selection and interface conformance."""
import json
import sys

import pytest

from onboard.config import DetectorConfig
from onboard.detector import (
    Detector,
    HailoDetector,
    NullDetector,
    ReplayDetector,
    build_detector,
)


def _replay_cfg(tmp_path, loop=True):
    path = tmp_path / "replay.jsonl"
    path.write_text(
        json.dumps(
            {
                "t": 0.0,
                "detections": [{"bbox": [10, 10, 50, 50], "conf": 0.8, "class_id": 0}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return DetectorConfig(backend="replay", replay_path=str(path), replay_loop=loop)


def test_null_detector_contract():
    det = build_detector(DetectorConfig(backend="null"))
    assert isinstance(det, NullDetector)
    assert isinstance(det, Detector)
    assert det.latest(0.4) is None
    assert det.latest(0.4, now=123.0) is None
    assert det.healthy(0.0) is True
    det.start()
    assert det.healthy(0.0) is True
    assert det.stats().backend == "null"
    det.stop()
    assert det.latest(0.4) is None


def test_replay_backend_builds_and_runs(tmp_path):
    det = build_detector(_replay_cfg(tmp_path))
    assert isinstance(det, ReplayDetector)
    assert isinstance(det, Detector)
    assert det.healthy(0.0) is False
    det.start()
    detection = det.latest(0.4, now=10.0)
    assert detection is not None
    assert detection.conf == pytest.approx(0.8)
    assert det.healthy(10.0) is True
    det.stop()
    assert det.healthy(10.0) is False


def test_replay_backend_requires_path():
    with pytest.raises(ValueError):
        build_detector(DetectorConfig(backend="replay", replay_path=""))


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        build_detector(DetectorConfig(backend="bogus"))


def test_hailo_backend_is_lazy_and_importable():
    assert "hailo_platform" not in sys.modules
    det = build_detector(DetectorConfig(backend="hailo", hef="/nonexistent/model.hef"))
    assert isinstance(det, HailoDetector)
    assert isinstance(det, Detector)
    assert det.stats().backend == "hailo"
    assert det.latest(0.4) is None
    assert det.healthy(0.0) is False
    det.stop()
    assert "hailo_platform" not in sys.modules


class _FakeSource:
    def __init__(self):
        self.calls = 0

    def latest(self, max_age_s, now=None):
        self.calls += 1
        return None, 0.0


def test_hailo_accepts_source_at_build_and_attach():
    source = _FakeSource()
    det = build_detector(DetectorConfig(backend="hailo"), source=source)
    assert det.source is source
    other = _FakeSource()
    det.attach_source(other)
    assert det.source is other
    det.submit_frame(object())
    det.stop()


def test_null_and_replay_accept_attach_source(tmp_path):
    null = build_detector(DetectorConfig(backend="null"))
    null.attach_source(_FakeSource())
    replay = build_detector(_replay_cfg(tmp_path))
    replay.attach_source(_FakeSource())
    replay.start()
    assert replay.latest(1.0, now=0.0) is not None
    replay.stop()
