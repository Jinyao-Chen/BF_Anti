"""DisplayPipeline tests: latest-only queue, non-blocking submit, retries.

The framebuffer is faked, so these tests run on a laptop with no /dev/fb0 and
never touch the Pi hardware.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from onboard.video.fbdev import FbGeometry
from onboard.video.pipeline import DisplayPipeline


def make_cfg(**overrides):
    values = dict(
        enabled=True,
        fb="/dev/fb0",
        width=64,
        height=48,
        fps=100.0,
        osd=False,
        letterbox=True,
        font_path="",
        draw_box=False,
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


class FakeSink:
    def __init__(self, delay=0.0, geometry=None):
        self.delay = delay
        self.geometry = geometry
        self.frames = []
        self.open_calls = 0
        self.close_calls = 0
        self.is_open = False
        self.fail_open_times = 0
        self.write_exceptions = 0

    def open(self):
        self.open_calls += 1
        if self.fail_open_times > 0:
            self.fail_open_times -= 1
            return False
        self.is_open = True
        return True

    def write(self, frame):
        if self.delay:
            time.sleep(self.delay)
        if self.write_exceptions > 0:
            self.write_exceptions -= 1
            raise OSError("framebuffer write failed")
        self.frames.append(np.array(frame, copy=True))
        return True

    def close(self):
        self.close_calls += 1
        self.is_open = False


class FakeRenderer:
    def __init__(self, width=64, height=48, fail=False):
        self.width = width
        self.height = height
        self.fail = fail
        self.calls = 0
        self.sizes = []

    def render(self, frame, detection=None, out=None, state=None,
               decision=None, now=0.0, status=None):
        if self.fail:
            raise RuntimeError("render boom")
        self.calls += 1
        fill = 0
        if frame is not None:
            arr = np.asarray(frame)
            if arr.size:
                fill = int(arr.reshape(-1)[0])
        canvas = np.full((self.height, self.width, 4), fill, dtype=np.uint8)
        canvas[:, :, 3] = 255
        return canvas

    def blank(self):
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def set_output_size(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.sizes.append((int(width), int(height)))


def test_disabled_pipeline_is_a_noop():
    sink = FakeSink()
    pipeline = DisplayPipeline(make_cfg(enabled=False), sink=sink, renderer=FakeRenderer())
    assert pipeline.start() is False
    pipeline.submit(np.zeros((48, 64, 3), np.uint8), None, None, None, None)
    assert pipeline.stats().submitted == 0
    assert sink.open_calls == 0
    pipeline.stop()
    assert sink.open_calls == 0


def test_start_is_idempotent_and_stop_closes_sink():
    sink = FakeSink()
    pipeline = DisplayPipeline(make_cfg(), sink=sink, renderer=FakeRenderer())
    assert pipeline.start() is True
    assert pipeline.start() is True
    assert sink.open_calls == 1
    assert pipeline.running is True
    pipeline.stop()
    assert pipeline.running is False
    assert sink.is_open is False
    assert sink.close_calls >= 1


def test_submit_renders_latest_frame():
    sink = FakeSink()
    pipeline = DisplayPipeline(make_cfg(), sink=sink, renderer=FakeRenderer())
    pipeline.start()
    pipeline.submit(np.full((48, 64, 3), 3, np.uint8), None, None, None, None)
    assert wait_until(lambda: pipeline.stats().displayed == 1)
    assert len(sink.frames) == 1
    assert sink.frames[0].shape == (48, 64, 4)
    assert sink.frames[0][0, 0, 3] == 255
    assert pipeline.stats().submitted == 1
    pipeline.stop()


def test_latest_only_queue_drops_older_frames():
    sink = FakeSink(delay=0.08)
    renderer = FakeRenderer()
    pipeline = DisplayPipeline(
        make_cfg(fps=200.0), sink=sink, renderer=renderer
    )
    pipeline.start()
    pipeline.submit(np.full((48, 64, 3), 1, np.uint8), None, None, None, None)
    time.sleep(0.02)  # worker is now blocked inside write()
    for value in (2, 3, 4, 5):
        pipeline.submit(np.full((48, 64, 3), value, np.uint8), None, None, None, None)
    assert wait_until(lambda: pipeline.stats().displayed >= 2, timeout=3.0)
    stats = pipeline.stats()
    assert stats.dropped >= 3
    assert int(sink.frames[-1][0, 0, 0]) == 5
    pipeline.stop()


def test_submit_never_blocks_on_slow_write():
    sink = FakeSink(delay=0.2)
    pipeline = DisplayPipeline(make_cfg(fps=200.0), sink=sink, renderer=FakeRenderer())
    pipeline.start()
    pipeline.submit(np.zeros((48, 64, 3), np.uint8), None, None, None, None)
    time.sleep(0.05)  # worker is inside the 0.2 s write
    started = time.perf_counter()
    for _ in range(20):
        pipeline.submit(np.zeros((48, 64, 3), np.uint8), None, None, None, None)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.05
    pipeline.stop()


def test_stale_pending_frame_is_dropped():
    sink = FakeSink()
    pipeline = DisplayPipeline(make_cfg(), sink=sink, renderer=FakeRenderer())
    pipeline.start()
    pipeline.submit(
        np.zeros((48, 64, 3), np.uint8), None, None, None, None,
        now=time.monotonic() - 10.0,
    )
    assert wait_until(lambda: pipeline.stats().stale >= 1)
    assert pipeline.stats().displayed == 0
    pipeline.stop()


def test_framebuffer_open_is_retried():
    sink = FakeSink()
    sink.fail_open_times = 2
    pipeline = DisplayPipeline(
        make_cfg(), sink=sink, renderer=FakeRenderer(), retry_s=0.02
    )
    pipeline.start()
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and pipeline.stats().displayed < 1:
        pipeline.submit(np.zeros((48, 64, 3), np.uint8), None, None, None, None)
        time.sleep(0.02)
    assert pipeline.stats().displayed >= 1
    assert sink.open_calls >= 3
    pipeline.stop()


def test_render_failure_falls_back_to_blank_canvas():
    sink = FakeSink()
    renderer = FakeRenderer(fail=True)
    pipeline = DisplayPipeline(make_cfg(), sink=sink, renderer=renderer)
    pipeline.start()
    pipeline.submit(np.zeros((48, 64, 3), np.uint8), None, None, None, None)
    assert wait_until(lambda: pipeline.stats().displayed >= 1)
    assert pipeline.stats().errors >= 1
    assert int(sink.frames[0].max()) == 0
    pipeline.stop()


def test_write_failure_marks_sink_closed_and_recovers():
    sink = FakeSink()
    sink.write_exceptions = 1
    pipeline = DisplayPipeline(
        make_cfg(fps=200.0), sink=sink, renderer=FakeRenderer(), retry_s=0.02
    )
    pipeline.start()
    pipeline.submit(np.zeros((48, 64, 3), np.uint8), None, None, None, None)
    assert wait_until(lambda: pipeline.stats().errors >= 1)
    assert pipeline.stats().sink_open is False
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and pipeline.stats().displayed < 1:
        pipeline.submit(np.ones((48, 64, 3), np.uint8), None, None, None, None)
        time.sleep(0.02)
    assert pipeline.stats().displayed >= 1
    assert sink.open_calls >= 2
    pipeline.stop()


def test_sink_geometry_overrides_configured_size():
    geometry = FbGeometry(width=80, height=60, bpp=32, stride=320, name="fake")
    sink = FakeSink(geometry=geometry)
    renderer = FakeRenderer()
    pipeline = DisplayPipeline(
        make_cfg(width=64, height=48), sink=sink, renderer=renderer
    )
    pipeline.start()
    assert renderer.sizes == [(80, 60)]
    pipeline.stop()


def test_loop_rate_is_measured_on_submit():
    pipeline = DisplayPipeline(make_cfg(), sink=FakeSink(), renderer=FakeRenderer())
    base = 1000.0
    for index in range(4):
        pipeline.submit(
            np.zeros((48, 64, 3), np.uint8), None, None, None, None,
            now=base + 0.02 * index,
        )
    assert 40.0 <= pipeline.stats().loop_hz <= 60.0


def test_healthy_state_transitions():
    disabled = DisplayPipeline(make_cfg(enabled=False), sink=FakeSink(), renderer=FakeRenderer())
    assert disabled.healthy() is True
    sink = FakeSink()
    pipeline = DisplayPipeline(make_cfg(), sink=sink, renderer=FakeRenderer())
    assert pipeline.healthy() is False
    pipeline.start()
    assert pipeline.healthy() is True
    pipeline.stop()
    assert pipeline.healthy() is False


def test_pipeline_with_real_osd_renderer():
    sink = FakeSink()
    pipeline = DisplayPipeline(
        make_cfg(osd=True, width=64, height=48), sink=sink
    )
    pipeline.start()
    pipeline.submit(np.full((48, 64, 3), 40, np.uint8), None, None, None, None)
    assert wait_until(lambda: pipeline.stats().displayed >= 1)
    assert sink.frames[0].shape == (48, 64, 4)
    assert np.all(sink.frames[0][:, :, 3] == 255)
    pipeline.stop()


def test_submit_after_stop_does_not_raise():
    sink = FakeSink()
    pipeline = DisplayPipeline(make_cfg(), sink=sink, renderer=FakeRenderer())
    pipeline.start()
    pipeline.stop()
    pipeline.submit(np.zeros((48, 64, 3), np.uint8), None, None, None, None)
    assert pipeline.stats().submitted == 1
