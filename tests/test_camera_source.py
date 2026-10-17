"""CameraSource tests: background capture, latest-only semantics, no hardware.

The real V4L2 device is not connected on this machine or on the Pi right now,
so every test injects a fake capture object (or a fake cv2 module). Nothing
here opens /dev/video* or requires OpenCV.
"""
from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import onboard.camera as camera_mod
from onboard.camera import (
    CAP_PROP_BUFFERSIZE,
    CAP_PROP_FPS,
    CAP_PROP_FOURCC,
    CAP_PROP_FRAME_HEIGHT,
    CAP_PROP_FRAME_WIDTH,
    CameraSource,
    fourcc_code,
)

FRAME = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)


def make_cfg(**overrides):
    values = dict(
        device="/dev/video-test",
        width=640,
        height=480,
        fps=30.0,
        fourcc="MJPG",
        warmup_s=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def wait_until(predicate, timeout=3.0, interval=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class FakeCapture:
    def __init__(self, opened=True, frame=FRAME, fail_reads=0):
        self.opened = opened
        self.frame = frame
        self.fail_reads = fail_reads
        self.props = {}
        self.set_calls = []
        self.read_count = 0
        self.released = False
        self.release_count = 0
        self._lock = threading.Lock()
        self._width = 640
        self._height = 480
        self._fps = 30.0

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        with self._lock:
            self.set_calls.append((prop, value))
            self.props[prop] = value
            if prop == CAP_PROP_FRAME_WIDTH:
                self._width = int(value)
            elif prop == CAP_PROP_FRAME_HEIGHT:
                self._height = int(value)
            elif prop == CAP_PROP_FPS:
                self._fps = float(value)
        return True

    def get(self, prop):
        if prop == CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == CAP_PROP_FPS:
            return float(self._fps)
        if prop == CAP_PROP_FOURCC:
            return float(fourcc_code("MJPG"))
        return 0.0

    def read(self):
        with self._lock:
            self.read_count += 1
            if self.fail_reads > 0:
                self.fail_reads -= 1
                return False, None
        return True, self.frame

    def release(self):
        self.released = True
        self.release_count += 1


class CaptureFactory:
    def __init__(self, captures):
        self.captures = list(captures)
        self.calls = 0
        self.devices = []

    def __call__(self, device):
        self.calls += 1
        self.devices.append(device)
        if self.captures:
            return self.captures.pop(0)
        return FakeCapture(frame=FRAME)


class FakeClock:
    def __init__(self, step=0.002, start=100.0):
        self.value = start
        self.step = step
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            self.value += self.step
            return self.value


def test_fourcc_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    assert fourcc_code("MJPG") == cv2.VideoWriter_fourcc(*"MJPG")
    assert fourcc_code("YUYV") == cv2.VideoWriter_fourcc(*"YUYV")


def test_config_validation_and_fields():
    src = CameraSource(make_cfg(), capture_factory=CaptureFactory([]))
    assert src.device == "/dev/video-test"
    assert (src.width, src.height) == (640, 480)
    assert src.fps == 30.0
    assert src.fourcc == "MJPG"
    with pytest.raises(ValueError):
        CameraSource(make_cfg(fps=0.0), capture_factory=CaptureFactory([]))


def test_start_is_idempotent_and_captures_latest():
    cap = FakeCapture()
    factory = CaptureFactory([cap])
    src = CameraSource(make_cfg(), capture_factory=factory)
    assert src.start() is True
    assert src.start() is True  # second call must not double-open
    assert wait_until(lambda: src.stats().frames >= 3)
    assert factory.calls == 1
    frame, stamp = src.latest(None)
    assert frame is cap.frame
    assert stamp > 0.0
    stats = src.stats()
    assert stats.opens == 1
    assert stats.frames >= 3
    assert stats.running is True
    src.stop()
    assert cap.released is True
    assert src.running is False


def test_latest_drops_stale_frame():
    cap = FakeCapture()
    src = CameraSource(make_cfg(), capture_factory=CaptureFactory([cap]))
    src.start()
    assert wait_until(lambda: src.stats().frames >= 1)
    src.stop()
    frame, stamp = src.latest(None)
    assert frame is not None
    stale, stale_stamp = src.latest(0.05, now=stamp + 1.0)
    assert stale is None
    assert stale_stamp == stamp
    fresh, fresh_stamp = src.latest(0.05, now=stamp + 0.01)
    assert fresh is cap.frame
    assert fresh_stamp == stamp


def test_missing_device_is_not_fatal():
    # Every retry returns a closed capture, as a missing UVC device would.
    factory = lambda device: FakeCapture(opened=False)  # noqa: E731 - test stub
    src = CameraSource(make_cfg(), capture_factory=factory)
    assert src.start() is True
    assert wait_until(lambda: src.stats().errors >= 1)
    frame, stamp = src.latest(None)
    assert frame is None and stamp == 0.0
    assert src.healthy() is False
    assert "cannot open" in src.stats().last_error
    src.stop()
    assert src.running is False


def test_open_exception_is_recorded():
    def boom(device):
        raise OSError("no such device")

    src = CameraSource(make_cfg(), capture_factory=boom)
    src.start()
    assert wait_until(lambda: "open failed" in src.stats().last_error)
    assert src.latest(None) == (None, 0.0)
    src.stop()


def test_capture_properties_are_configured():
    cap = FakeCapture()
    src = CameraSource(make_cfg(), capture_factory=CaptureFactory([cap]))
    src.start()
    assert wait_until(lambda: src.stats().opens == 1)
    assert cap.props[CAP_PROP_FOURCC] == fourcc_code("MJPG")
    assert cap.props[CAP_PROP_FRAME_WIDTH] == 640
    assert cap.props[CAP_PROP_FRAME_HEIGHT] == 480
    assert cap.props[CAP_PROP_FPS] == 30.0
    assert cap.props[CAP_PROP_BUFFERSIZE] == 1
    src.stop()


def test_start_waits_for_first_attempt_and_records_open_error():
    def slow_closed(device):
        time.sleep(0.05)
        return FakeCapture(opened=False)

    src = CameraSource(make_cfg(), capture_factory=slow_closed)
    started = time.monotonic()
    assert src.start() is True
    elapsed = time.monotonic() - started
    stats = src.stats()
    assert elapsed >= 0.03
    assert stats.open_attempts >= 1
    assert stats.opens == 0
    assert "cannot open" in stats.last_error
    assert "cannot open" in stats.last_open_error
    src.stop()
    assert src.running is False


def test_open_error_is_kept_after_recovery():
    factory = CaptureFactory([FakeCapture(opened=False), FakeCapture()])
    src = CameraSource(make_cfg(), capture_factory=factory)
    src.start()
    assert wait_until(lambda: src.stats().frames >= 1, timeout=3.0)
    stats = src.stats()
    assert stats.open_attempts >= 2
    assert "cannot open" in stats.last_open_error
    assert stats.last_error == ""
    src.stop()


def test_consecutive_read_failures_reopen_device(monkeypatch):
    monkeypatch.setattr(camera_mod, "_DEAD_AFTER_FAILS", 2)
    bad = FakeCapture(fail_reads=2)
    good = FakeCapture()
    factory = CaptureFactory([bad, good])
    src = CameraSource(make_cfg(), capture_factory=factory)
    src.start()
    assert wait_until(lambda: src.stats().reopens >= 1, timeout=4.0)
    assert wait_until(lambda: src.stats().frames >= 1, timeout=4.0)
    assert bad.released is True
    assert factory.calls >= 2
    src.stop()


def test_stop_releases_and_restart_opens_again():
    first = FakeCapture()
    second = FakeCapture()
    factory = CaptureFactory([first, second])
    src = CameraSource(make_cfg(), capture_factory=factory)
    src.start()
    assert wait_until(lambda: src.stats().opens == 1)
    src.stop()
    assert first.released is True
    src.start()
    assert wait_until(lambda: src.stats().opens == 2, timeout=4.0)
    src.stop()
    assert second.released is True


def test_warmup_frames_are_not_published():
    clock = FakeClock(step=0.002)
    cap = FakeCapture()
    src = CameraSource(
        make_cfg(warmup_s=0.02), capture_factory=CaptureFactory([cap]), clock=clock
    )
    src.start()
    assert wait_until(lambda: src.stats().warmup_frames >= 1)
    assert wait_until(lambda: src.stats().frames >= 1)
    assert src.stats().warmup_frames > 0
    src.stop()


def test_measured_fps_window():
    clock = FakeClock(step=0.05)
    src = CameraSource(make_cfg(), capture_factory=CaptureFactory([FakeCapture()]), clock=clock)
    src.start()
    assert wait_until(lambda: src.stats().frames >= 25)
    assert 10.0 <= src.stats().measured_fps <= 30.0
    src.stop()


def test_cv2_backend_is_pinned_to_v4l2(monkeypatch):
    class FakeCV2:
        CAP_V4L2 = 200

        def __init__(self):
            self.calls = []
            self.cap = FakeCapture()

        def VideoCapture(self, device, api=None):
            self.calls.append((device, api))
            return self.cap

    fake = FakeCV2()
    monkeypatch.setitem(sys.modules, "cv2", fake)
    src = CameraSource(make_cfg(), capture_factory=None)
    src.start()
    assert wait_until(lambda: src.stats().opens == 1)
    assert fake.calls == [("/dev/video-test", 200)]
    src.stop()
    assert fake.cap.released is True


def test_factory_path_does_not_import_cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    cap = FakeCapture()
    src = CameraSource(make_cfg(), capture_factory=CaptureFactory([cap]))
    src.start()
    assert wait_until(lambda: src.stats().frames >= 1)
    src.stop()


def test_healthy_requires_fresh_frames():
    src = CameraSource(make_cfg(), capture_factory=CaptureFactory([FakeCapture()]))
    src.start()
    assert wait_until(lambda: src.stats().frames >= 1)
    stamp = src.stats().last_frame_t
    assert src.healthy(now=stamp + 0.1) is True
    assert src.healthy(now=stamp + 100.0) is False
    src.stop()
    assert src.healthy(now=stamp + 0.1) is False
