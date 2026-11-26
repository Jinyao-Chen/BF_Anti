"""V4L2 camera source: one background capture thread, latest-frame only.

Design constraints (see docs/sim2real/pi5_video.md):

* The control loop calls latest() at 50 Hz and must never block on the
  camera, so capture runs on its own daemon thread and readers only take a
  reference to the newest decoded frame.
* The device is opened exactly once per capture session, by the capture thread.
  start() is idempotent and never double-opens /dev/video*.
* The camera is hot-pluggable: if the UVC device is absent or fails, the
  thread backs off and retries instead of taking the whole onboard process
  down. healthy()/stats expose the reason.
* No X11 / no GUI backend: capture is pinned to cv2.CAP_V4L2 and we only
  decode MJPG in-process. There is no GStreamer/X11 path.
* cv2 is imported lazily so importing this module (and the unit tests) does
  not require OpenCV or a camera.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass

import numpy as np
from typing import Any, Callable, Optional, Tuple

LOG = logging.getLogger("bf_anti.camera")

# OpenCV videoio property ids (stable public ABI, kept here so the capture
# configuration path can be exercised by tests without importing cv2).
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5
CAP_PROP_FOURCC = 6
CAP_PROP_BUFFERSIZE = 38
CAP_V4L2 = 200

# Consecutive read failures before the device is considered dead and reopened.
_DEAD_AFTER_FAILS = 5
# Upper bound on device-open retry delay.
_MAX_BACKOFF_S = 5.0


def fourcc_code(text: str) -> int:
    """Return the little-endian FOURCC integer OpenCV expects.

    cv2.VideoWriter_fourcc(*name) produces the same value; we compute it
    ourselves so this module has no import-time OpenCV dependency.
    """
    raw = (str(text).upper() + "    ")[:4]
    return sum(ord(ch) << (8 * i) for i, ch in enumerate(raw))


@dataclass(frozen=True)
class CameraStats:
    """Snapshot of capture health. All times are monotonic seconds."""

    frames: int = 0
    errors: int = 0
    read_fails: int = 0
    opens: int = 0
    open_attempts: int = 0
    reopens: int = 0
    warmup_frames: int = 0
    measured_fps: float = 0.0
    last_frame_t: float = 0.0
    last_open_t: float = 0.0
    last_open_error: str = ""
    width: int = 0
    height: int = 0
    capture_width: int = 0
    capture_height: int = 0
    requested_width: int = 0
    requested_height: int = 0
    fps: float = 0.0
    fourcc: str = ""
    last_error: str = ""
    running: bool = False


class CameraSource:
    """Latest-frame V4L2 source.

    Parameters
    ----------
    cfg:
        onboard.config.CameraConfig-shaped object (device, width, height,
        fps, fourcc, warmup_s). Only attribute access is used.
    capture_factory:
        Optional factory(device) -> capture used by tests. The returned
        object must provide isOpened(), set(prop, value), get(prop),
        read() -> (ok, frame) and release().
    clock:
        Monotonic clock injection for tests.
    """

    def __init__(
        self,
        cfg: Any,
        capture_factory: Optional[Callable[[str], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
        logger: Optional[logging.Logger] = None,
        open_wait_s: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.device = str(getattr(cfg, "device", "/dev/video0"))
        # Requested sensor geometry is immutable.  The old code overwrote it
        # with the driver's actual value, so after one 640x480 fallback every
        # reconnect requested 640x480 and the calibration never recovered.
        self._req_width = int(getattr(cfg, "width", 1280))
        self._req_height = int(getattr(cfg, "height", 720))
        self._req_fps = float(getattr(cfg, "fps", 30.0))
        self._req_fourcc = str(getattr(cfg, "fourcc", "MJPG") or "MJPG")
        self.strict_size = bool(getattr(cfg, "strict_size", True))
        self.width = self._req_width       # output (post-rotation)
        self.height = self._req_height
        self.fps = self._req_fps
        self.fourcc = self._req_fourcc
        self._capture_width = self._req_width
        self._capture_height = self._req_height
        self._resolved_device = self.device
        self.warmup_s = max(0.0, float(getattr(cfg, "warmup_s", 1.0) or 0.0))
        self.rotate_deg = int(getattr(cfg, "rotate_deg", 0)) % 360
        if self.rotate_deg not in (0, 90, 180, 270):
            raise ValueError("camera rotate_deg must be 0/90/180/270")
        if self.fps <= 0.0:
            raise ValueError("camera fps must be positive")
        self._capture_factory = capture_factory
        self._clock = clock
        self._log = logger or LOG
        # start() waits this long for the first open attempt so stats() is
        # meaningful the moment start() returns (camera absent is common).
        self._open_wait_s = max(0.0, float(open_wait_s))

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._first_attempt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[Any] = None
        self._frame_t: float = 0.0

        self._frames = 0
        self._errors = 0
        self._read_fails = 0
        self._opens = 0
        self._open_attempts = 0
        self._reopens = 0
        self._warmup_frames = 0
        self._last_error = ""
        self._last_open_error = ""
        self._last_open_t = 0.0
        self._last_report_t = 0.0
        self._last_report_frames = 0
        self._measured_fps = 0.0

    # -- lifecycle -------------------------------------------------------
    def start(self) -> bool:
        """Start capture. Returns True if the thread is running afterwards.

        Idempotent: calling it twice does not open the device twice. start()
        waits up to open_wait_s for the first device-open attempt to finish,
        so stats().last_error / last_open_error are populated immediately even
        when the camera is absent. It never raises for a missing camera.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            self._first_attempt.clear()
            self._thread = threading.Thread(
                target=self._run, name="bf-anti-camera", daemon=True
            )
            self._thread.start()
        if self._open_wait_s > 0.0:
            self._first_attempt.wait(timeout=self._open_wait_s)
        return True

    def stop(self) -> None:
        """Stop capture and release the device. Safe to call repeatedly."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, 4.0 / max(1.0, self.fps)))
            if thread.is_alive():
                self._log.warning("camera thread did not stop within timeout")
        with self._lock:
            self._thread = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    # -- data path -------------------------------------------------------
    def latest(
        self, max_age_s: Optional[float], now: Optional[float] = None
    ) -> Tuple[Optional[Any], float]:
        """Return (frame, t_mono) for the newest frame.

        (None, t_mono) is returned when no frame has ever arrived, or when
        the newest one is older than max_age_s (None disables the age check).
        The returned frame must be treated as read-only: it is the same object
        handed to every consumer until the next capture replaces it.
        """
        now = self._clock() if now is None else float(now)
        with self._lock:
            frame = self._frame
            stamp = self._frame_t
        if frame is None:
            return None, 0.0
        if max_age_s is not None and float(max_age_s) > 0.0:
            if (now - stamp) > float(max_age_s):
                return None, stamp
        return frame, stamp

    def healthy(self, now: Optional[float] = None) -> bool:
        """True when a frame arrived recently and the thread is alive."""
        now = self._clock() if now is None else float(now)
        with self._lock:
            stamp = self._frame_t
            frames = self._frames
        if frames <= 0 or stamp <= 0.0 or not self.running:
            return False
        return (now - stamp) <= max(0.5, 5.0 / max(1.0, self.fps))

    @property
    def measured_fps(self) -> float:
        with self._lock:
            return self._measured_fps

    def stats(self) -> CameraStats:
        with self._lock:
            return CameraStats(
                frames=self._frames,
                errors=self._errors,
                read_fails=self._read_fails,
                opens=self._opens,
                open_attempts=self._open_attempts,
                reopens=self._reopens,
                warmup_frames=self._warmup_frames,
                measured_fps=self._measured_fps,
                last_frame_t=self._frame_t,
                last_open_t=self._last_open_t,
                last_open_error=self._last_open_error,
                width=self.width,
                height=self.height,
                capture_width=self._capture_width,
                capture_height=self._capture_height,
                requested_width=self._req_width,
                requested_height=self._req_height,
                fps=self.fps,
                fourcc=self.fourcc,
                last_error=self._last_error,
                running=self.running,
            )

    # -- capture thread --------------------------------------------------
    def _run(self) -> None:
        cap: Optional[Any] = None
        attempt = 0
        fails = 0
        try:
            while not self._stop.is_set():
                if cap is None:
                    cap = self._open()
                    self._first_attempt.set()
                    if cap is None:
                        attempt += 1
                        delay = min(_MAX_BACKOFF_S, 0.25 * attempt)
                        if attempt in (1, 10) or attempt % 50 == 0:
                            self._log.warning(
                                "camera %s unavailable (attempt %d): %s",
                                self.device, attempt, self._last_error,
                            )
                        self._stop.wait(delay)
                        continue
                    attempt = 0
                    fails = 0
                frame_t = 0.0
                try:
                    if hasattr(cap, "grab") and hasattr(cap, "retrieve"):
                        if cap.grab():
                            # Capture timestamp belongs to the moment the
                            # driver handed us the frame, before MJPEG decode
                            # and rotation add 20-60 ms of latency.
                            frame_t = self._clock()
                            ok, frame = cap.retrieve()
                        else:
                            ok, frame = False, None
                    else:
                        ok, frame = cap.read()
                        frame_t = self._clock()
                except Exception as exc:  # noqa: BLE001 - driver exceptions vary
                    ok, frame = False, None
                    self._record_error("read raised: %s" % (exc,))
                if not ok or frame is None:
                    fails += 1
                    with self._lock:
                        self._errors += 1
                        self._read_fails += 1
                    if fails >= _DEAD_AFTER_FAILS:
                        self._log.warning(
                            "camera %s: %d consecutive read failures, reopening",
                            self.device, fails,
                        )
                        self._release(cap)
                        cap = None
                        with self._lock:
                            self._reopens += 1
                        self._stop.wait(0.5)
                    continue
                fails = 0
                self._on_frame(self._rotate_frame(frame), stamp=frame_t)
        finally:
            self._first_attempt.set()
            if cap is not None:
                self._release(cap)

    def _rotate_frame(self, frame: Any) -> Any:
        """Apply the configured clockwise rotation to a captured frame."""
        if self.rotate_deg == 0:
            return frame
        try:
            import cv2
            code = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }[self.rotate_deg]
            return cv2.rotate(frame, code)
        except Exception:
            k = {90: -1, 180: 2, 270: 1}[self.rotate_deg]
            return np.ascontiguousarray(np.rot90(frame, k=k))

    def _resolve_device(self) -> str:
        """Return a capture node, tolerating a USB re-enumeration.

        Prefer the configured path.  Only when it is gone and exactly one
        /dev/v4l/by-id/*-video-index0 exists do we substitute it; picking the
        first of several /dev/video* nodes could silently select the wrong
        camera (or a metadata node).
        """
        if self._capture_factory is not None:
            return self.device
        if os.path.exists(self.device):
            return self.device
        # Only substitute a stable by-id node for a real /dev/videoN path.
        # Arbitrary paths (tests, future device aliases) must not silently
        # open whichever camera happens to be present.
        if not re.fullmatch(r"video\d+", os.path.basename(self.device)):
            return self.device
        try:
            import glob
            candidates = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
        except Exception:
            candidates = []
        if len(candidates) == 1:
            self._log.warning(
                "camera %s disappeared; using stable node %s",
                self.device, candidates[0],
            )
            return candidates[0]
        return self.device

    def _apply_controls(self, cap: Any) -> None:
        """Apply optional raw V4L2 controls; unsupported properties are ignored."""
        try:
            import cv2
        except Exception:
            return
        mappings = (
            ("autofocus", "CAP_PROP_AUTOFOCUS", int),
            ("auto_exposure", "CAP_PROP_AUTO_EXPOSURE", int),
            ("exposure", "CAP_PROP_EXPOSURE", float),
            ("gain", "CAP_PROP_GAIN", float),
            ("auto_wb", "CAP_PROP_AUTO_WB", int),
            ("wb_temp", "CAP_PROP_WB_TEMPERATURE", float),
        )
        for cfg_name, cv_name, caster in mappings:
            value = getattr(self.cfg, cfg_name, None)
            prop = getattr(cv2, cv_name, None)
            if value is None or prop is None:
                continue
            try:
                self._try_set(cap, int(prop), caster(value))
            except (TypeError, ValueError):
                pass

    def _open(self) -> Optional[Any]:
        """Open and configure the device; return the capture or None."""
        cap = None
        with self._lock:
            self._open_attempts += 1
        device = self._resolve_device()
        try:
            if self._capture_factory is not None:
                cap = self._capture_factory(device)
            else:
                import cv2  # local import: no OpenCV needed until capture

                cap = cv2.VideoCapture(device, CAP_V4L2)
            if cap is None or not cap.isOpened():
                self._record_open_error("cannot open %s" % device)
                if cap is not None:
                    cap.release()
                return None

            # MJPG first: the Pi 5 USB bandwidth cannot carry raw YUYV at
            # 1280x720@30 reliably, and MJPG is what the UVC camera exposes.
            self._try_set(cap, CAP_PROP_FOURCC, fourcc_code(self._req_fourcc))
            self._try_set(cap, CAP_PROP_FRAME_WIDTH, int(self._req_width))
            self._try_set(cap, CAP_PROP_FRAME_HEIGHT, int(self._req_height))
            self._try_set(cap, CAP_PROP_FPS, float(self._req_fps))
            # Keep only the newest driver buffer so userspace never replays a
            # backlog of stale frames after a scheduling hiccup.
            self._try_set(cap, CAP_PROP_BUFFERSIZE, 1)
            self._apply_controls(cap)

            actual_w = int(self._safe_get(cap, CAP_PROP_FRAME_WIDTH, self._req_width))
            actual_h = int(self._safe_get(cap, CAP_PROP_FRAME_HEIGHT, self._req_height))
            actual_fps = float(self._safe_get(cap, CAP_PROP_FPS, self._req_fps))
            raw_fcc = int(self._safe_get(cap, CAP_PROP_FOURCC, 0.0))
            actual_fourcc = (
                "".join(chr((raw_fcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
                if raw_fcc > 0 else self._req_fourcc
            )
            if actual_w <= 0 or actual_h <= 0:
                self._record_open_error("driver reported invalid geometry")
                cap.release()
                return None
            if actual_w != self._req_width or actual_h != self._req_height:
                message = (
                    "camera geometry mismatch: got %dx%d, requested %dx%d "
                    "(static intrinsics would be wrong)"
                    % (actual_w, actual_h, self._req_width, self._req_height)
                )
                self._log.error("%s", message)
                if self.strict_size:
                    self._record_open_error(message)
                    cap.release()
                    return None
            out_w, out_h = ((actual_h, actual_w)
                            if self.rotate_deg in (90, 270) else (actual_w, actual_h))
            with self._lock:
                self._opens += 1
                self._last_open_t = self._clock()
                self._last_error = ""
                self._capture_width = actual_w
                self._capture_height = actual_h
                self.width = out_w
                self.height = out_h
                self.fps = actual_fps if actual_fps > 0.0 else self._req_fps
                self.fourcc = actual_fourcc or self._req_fourcc
            self._log.info(
                "camera %s open: %dx%d @ %.1f fps %s -> %dx%d (requested %dx%d @ %.1f)",
                device, actual_w, actual_h, actual_fps, self.fourcc, out_w, out_h,
                self._req_width, self._req_height, self._req_fps,
            )
            return cap
        except Exception as exc:  # noqa: BLE001 - cv2 raises generic exceptions
            self._record_open_error("open failed: %s" % (exc,))
            if cap is not None:
                try:
                    cap.release()
                except Exception:  # noqa: BLE001
                    pass
            return None

    @staticmethod
    def _try_set(cap: Any, prop: int, value: Any) -> bool:
        try:
            return bool(cap.set(prop, value))
        except Exception:  # noqa: BLE001 - unsupported property is not fatal
            return False

    @staticmethod
    def _safe_get(cap: Any, prop: int, default: float) -> float:
        try:
            value = float(cap.get(prop))
        except Exception:  # noqa: BLE001
            return default
        return value if value == value else default  # NaN guard

    def _on_frame(self, frame: Any, stamp: float = 0.0) -> None:
        now = float(stamp) if float(stamp or 0.0) > 0.0 else self._clock()
        with self._lock:
            opened_t = self._last_open_t
            warming = self.warmup_s > 0.0 and (now - opened_t) < self.warmup_s
            if warming:
                self._warmup_frames += 1
            else:
                self._frame = frame
                self._frame_t = now
                self._frames += 1
            self._update_fps_locked(now)
        if warming:
            self._log.debug("camera warmup frame %d", self._warmup_frames)

    def _update_fps_locked(self, now: float) -> None:
        # Report measured capture FPS over a ~1 s window.
        if self._last_report_t <= 0.0:
            self._last_report_t = now
            self._last_report_frames = self._frames
            return
        elapsed = now - self._last_report_t
        if elapsed >= 1.0:
            delta = self._frames - self._last_report_frames
            self._measured_fps = delta / elapsed
            self._last_report_t = now
            self._last_report_frames = self._frames

    def _record_error(self, message: str) -> None:
        with self._lock:
            self._errors += 1
            self._last_error = message

    def _record_open_error(self, message: str) -> None:
        """Record a device-open failure (also kept as last_open_error)."""
        with self._lock:
            self._errors += 1
            self._last_error = message
            self._last_open_error = message

    @staticmethod
    def _release(cap: Any) -> None:
        try:
            cap.release()
        except Exception:  # noqa: BLE001 - release is best effort
            pass
