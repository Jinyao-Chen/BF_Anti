"""Independent display pipeline: latest-only queue -> OSD -> /dev/fb0.

Contract with the control loop (onboard/app.py):

    pipeline.submit(frame, detection, out, state, decision, now)

submit() never blocks on rendering, framebuffer I/O, or lock contention: it
stores one pending snapshot, replaces (and counts) any older one, and returns.
A dedicated display thread renders at video.fps, drops frames that are already
stale, retries a missing framebuffer, and writes BGRA straight to fbdev.

When video.enabled is false the pipeline is a complete no-op: no thread, no
framebuffer open, no OSD work, and submit() returns immediately.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from .fbdev import FBDevSink
from .osd import OsdRenderer, OsdStatus

if TYPE_CHECKING:  # pragma: no cover - typing only
    from onboard.config import VideoConfig

LOG = logging.getLogger("bf_anti.video.pipeline")


@dataclass(frozen=True)
class DisplayStats:
    """Snapshot of display-thread counters and timings."""

    submitted: int = 0
    displayed: int = 0
    dropped: int = 0
    stale: int = 0
    errors: int = 0
    sink_misses: int = 0
    loop_hz: float = 0.0
    display_fps: float = 0.0
    last_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    last_display_t: float = 0.0
    sink_open: bool = False
    running: bool = False


class DisplayPipeline:
    """Latest-only OSD + framebuffer display thread."""

    def __init__(
        self,
        cfg: "VideoConfig",
        sink: Optional[Any] = None,
        renderer: Optional[Any] = None,
        clock: Any = time.monotonic,
        logger: Optional[logging.Logger] = None,
        retry_s: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.enabled = bool(getattr(cfg, "enabled", True))
        self.fps = max(1.0, float(getattr(cfg, "fps", 25.0) or 25.0))
        self._clock = clock
        self._log = logger or LOG
        # How often to retry a missing framebuffer (boot ordering, hotplug).
        self._retry_s = (
            max(0.02, float(retry_s)) if retry_s is not None
            else max(1.0, 2.0 / self.fps)
        )

        self._sink = sink
        self._renderer = renderer
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # RLock: start() holds the lock while opening the sink, and the
        # geometry helper re-enters it. It is never held across a framebuffer
        # write, so submit() stays non-blocking in steady state.
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._pending: Optional[tuple] = None

        self._sink_open = False
        self._last_open_attempt = 0.0
        self._open_attempts = 0

        self._submitted = 0
        self._displayed = 0
        self._dropped = 0
        self._stale = 0
        self._errors = 0
        self._sink_misses = 0
        self._last_submit_t = 0.0
        self._last_display_t = 0.0
        self._loop_hz = 0.0
        self._display_fps = 0.0
        self._last_latency_ms = 0.0
        self._avg_latency_ms = 0.0

    # -- lifecycle -------------------------------------------------------
    def start(self) -> bool:
        """Open the framebuffer and start the display thread."""
        if not self.enabled:
            self._log.info("video disabled by config; display pipeline is a no-op")
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            if self._sink is None:
                self._sink = FBDevSink(
                    device=str(getattr(self.cfg, "fb", "/dev/fb0")),
                    width=int(getattr(self.cfg, "width", 720)),
                    height=int(getattr(self.cfg, "height", 576)),
                )
            if self._renderer is None:
                self._renderer = OsdRenderer(
                    width=int(getattr(self.cfg, "width", 720)),
                    height=int(getattr(self.cfg, "height", 576)),
                    osd=bool(getattr(self.cfg, "osd", True)),
                    letterbox=bool(getattr(self.cfg, "letterbox", True)),
                    font_path=str(getattr(self.cfg, "font_path", "") or ""),
                    draw_box=bool(getattr(self.cfg, "draw_box", True)),
                )
            self._open_sink_locked()
            self._thread = threading.Thread(
                target=self._run, name="bf-anti-display", daemon=True
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        """Stop the thread and unmap the framebuffer. Idempotent."""
        if not self.enabled:
            return
        with self._cond:
            self._stop.set()
            self._pending = None
            self._cond.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, 2.0 + 2.0 / self.fps))
            if thread.is_alive():
                self._log.warning("display thread did not stop within timeout")
        with self._lock:
            sink = self._sink
            self._thread = None
            self._sink_open = False
        if sink is not None:
            try:
                sink.close()
            except Exception:  # noqa: BLE001 - close is best effort
                self._log.exception("framebuffer close failed")

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    # -- data path -------------------------------------------------------
    def submit(
        self,
        frame: Any,
        detection: Any,
        out: Any,
        state: Any,
        decision: Any,
        now: Optional[float] = None,
    ) -> None:
        """Queue the newest snapshot. Never blocks on rendering or I/O."""
        if not self.enabled:
            return
        stamp = self._clock() if now is None else float(now)
        with self._cond:
            if self._pending is not None:
                self._dropped += 1
            self._submitted += 1
            if self._last_submit_t > 0.0:
                dt = stamp - self._last_submit_t
                if 0.0 < dt < 1.0:
                    inst = 1.0 / dt
                    self._loop_hz = (
                        inst if self._loop_hz <= 0.0
                        else (0.9 * self._loop_hz + 0.1 * inst)
                    )
            self._last_submit_t = stamp
            self._pending = (
                frame, detection, out, state, decision, stamp, self._clock(),
            )
            self._cond.notify()

    def stats(self) -> DisplayStats:
        with self._lock:
            return DisplayStats(
                submitted=self._submitted,
                displayed=self._displayed,
                dropped=self._dropped,
                stale=self._stale,
                errors=self._errors,
                sink_misses=self._sink_misses,
                loop_hz=self._loop_hz,
                display_fps=self._display_fps,
                last_latency_ms=self._last_latency_ms,
                avg_latency_ms=self._avg_latency_ms,
                last_display_t=self._last_display_t,
                sink_open=self._sink_open,
                running=self.running,
            )

    def healthy(self, now: Optional[float] = None) -> bool:
        if not self.enabled:
            return True
        now = self._clock() if now is None else float(now)
        with self._lock:
            last = self._last_display_t
            displayed = self._displayed
        if not self.running:
            return False
        if displayed <= 0:
            return True
        return (now - last) <= max(1.0, 5.0 / self.fps)

    # -- display thread --------------------------------------------------
    def _run(self) -> None:
        period = 1.0 / self.fps
        stale_s = max(0.30, 3.0 * period)
        next_t = self._clock()
        while not self._stop.is_set():
            item: Optional[tuple] = None
            with self._cond:
                while self._pending is None and not self._stop.is_set():
                    self._cond.wait(timeout=period)
                if self._stop.is_set():
                    break
                item = self._pending
                self._pending = None
            if item is None:
                continue
            self._process(item, stale_s)
            next_t += period
            delay = next_t - self._clock()
            if delay > 0.0:
                self._stop.wait(delay)
            else:
                next_t = self._clock()

    def _process(self, item: tuple, stale_s: float) -> None:
        frame, detection, out, state, decision, stamp, _enqueued = item
        now = self._clock()
        # Age is measured from the caller's timestamp (control-loop monotonic
        # clock) so a frame queued while the display thread was busy is
        # correctly recognised as stale instead of being rendered late.
        if (now - float(stamp)) > stale_s:
            with self._lock:
                self._stale += 1
            return
        if not self._ensure_sink():
            with self._lock:
                self._sink_misses += 1
            return

        renderer = self._renderer
        sink = self._sink
        if renderer is None or sink is None:
            return
        with self._lock:
            status = OsdStatus(
                loop_hz=self._loop_hz,
                latency_ms=self._last_latency_ms,
                display_fps=self._display_fps,
                dropped=self._dropped + self._stale,
            )
        started = self._clock()
        try:
            canvas = renderer.render(
                frame, detection, out, state, decision, float(stamp), status
            )
        except Exception:  # noqa: BLE001 - OSD must never kill the display thread
            with self._lock:
                self._errors += 1
            self._log.exception("OSD render failed, writing blank canvas")
            try:
                canvas = renderer.blank()
            except Exception:  # noqa: BLE001
                return
        try:
            ok = sink.write(canvas)
        except ValueError as exc:
            with self._lock:
                self._errors += 1
            self._log.error("display frame rejected: %s", exc)
            return
        except Exception:  # noqa: BLE001 - device errors are recovered by reopen
            with self._lock:
                self._errors += 1
                self._sink_open = False
            self._log.exception("framebuffer write failed")
            return
        finished = self._clock()
        if ok is False:
            with self._lock:
                self._errors += 1
                self._sink_open = False
            return
        latency_ms = max(0.0, (finished - started) * 1000.0)
        with self._lock:
            self._displayed += 1
            if self._last_display_t > 0.0:
                dt = finished - self._last_display_t
                if 0.0 < dt < 1.0:
                    inst = 1.0 / dt
                    self._display_fps = (
                        inst if self._display_fps <= 0.0
                        else (0.9 * self._display_fps + 0.1 * inst)
                    )
            self._last_display_t = finished
            self._last_latency_ms = latency_ms
            self._avg_latency_ms = (
                latency_ms if self._avg_latency_ms <= 0.0
                else (0.9 * self._avg_latency_ms + 0.1 * latency_ms)
            )

    def _ensure_sink(self) -> bool:
        with self._lock:
            if self._sink_open:
                return True
            sink = self._sink
            renderer = self._renderer
        if sink is None:
            return False
        now = self._clock()
        with self._lock:
            if (now - self._last_open_attempt) < self._retry_s:
                return False
            self._last_open_attempt = now
            self._open_attempts += 1
            attempt = self._open_attempts
        try:
            opened = bool(sink.open())
        except Exception:  # noqa: BLE001
            opened = False
            self._log.exception("framebuffer open failed")
        if not opened:
            if attempt in (1, 10) or attempt % 60 == 0:
                self._log.warning(
                    "framebuffer unavailable (attempt %d), retrying", attempt
                )
            return False
        self._open_sink_state(renderer, sink)
        self._log.info("framebuffer ready after %d attempt(s)", attempt)
        return True

    def _open_sink_locked(self) -> None:
        sink = self._sink
        renderer = self._renderer
        if sink is None:
            return
        try:
            opened = bool(sink.open())
        except Exception:  # noqa: BLE001
            opened = False
            self._log.exception("framebuffer open failed")
        self._last_open_attempt = self._clock()
        if opened:
            self._open_sink_state(renderer, sink)
        else:
            self._log.warning(
                "framebuffer unavailable at start; display thread will retry"
            )

    def _open_sink_state(self, renderer: Optional[Any], sink: Any) -> None:
        geometry = getattr(sink, "geometry", None)
        if geometry is not None and renderer is not None:
            try:
                renderer.set_output_size(int(geometry.width), int(geometry.height))
            except Exception:  # noqa: BLE001
                self._log.exception("could not apply framebuffer geometry to OSD")
        with self._lock:
            self._sink_open = True
