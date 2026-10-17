"""Hailo-8L detector backend (HailoRT 4.x Python API).

Design
------
* HailoRT is imported lazily inside the worker thread, so importing this
  module (and the whole test suite) works on a laptop with no Hailo package.
* A single daemon thread owns the device: it pulls the newest frame from an
  attached FrameSource (normally onboard.camera.CameraSource), or consumes
  frames pushed through submit_frame(), runs one inference at a time, and
  publishes the newest Detection. The 50 Hz control loop only ever copies a
  reference in latest(); it never blocks on capture or inference.
* Only the newest frame is processed. If the camera runs faster than the
  network, older frames are dropped rather than queued, which bounds latency
  instead of building a backlog.
* Configuration/device errors do not crash the app. The worker retries with a
  backoff and healthy() stays False, so the safety supervisor keeps auto
  authority closed until a real detection is flowing.

Frame source contract
---------------------
Attach any object with

    latest(max_age_s, now=None) -> (frame_bgr_uint8, t_monotonic)

exactly like onboard.camera.CameraSource. The recommended wiring is:

    camera = CameraSource(cfg.camera)
    detector = build_detector(cfg.detector, source=camera)
    app = OnboardApp(cfg, camera=camera, detector=detector)

or, if the app already built the camera first:

    detector.attach_source(camera)

Without a source the detector can still be fed with submit_frame(frame, t) for
benchmarks and tests, but in the real app an unattached Hailo detector reports
healthy() == False on purpose.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

import numpy as np

from onboard.detector.base import DetectorStats, FrameSource
from onboard.detector.postprocess import (
    Letterbox,
    letterbox_image,
    parse_hailo_nms,
    parse_yolo_v8_raw,
)
from onboard.types import Detection

LOG = logging.getLogger("bf_anti.detector.hailo")

__all__ = ["HailoDetector"]


class HailoDetector:
    """Hailo-8L detector with an async, latest-frame-only worker."""

    backend = "hailo"

    def __init__(
        self,
        cfg: Any,
        source: Optional[FrameSource] = None,
        timeout_ms: int = 10000,
        stale_s: float = 0.5,
        retry_s: float = 2.0,
        bgr_to_rgb: bool = True,
        clock=time.monotonic,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.hef = os.path.expanduser(
            os.path.expandvars(str(getattr(cfg, "hef", "")))
        )
        self.input_width = int(getattr(cfg, "input_width", 640))
        self.input_height = int(getattr(cfg, "input_height", 640))
        self.conf_threshold = float(getattr(cfg, "conf_threshold", 0.25))
        self.iou_threshold = float(getattr(cfg, "iou_threshold", 0.45))
        self.max_detections = int(getattr(cfg, "max_detections", 10))
        self.class_names = tuple(getattr(cfg, "class_names", ()))
        self.async_infer = bool(getattr(cfg, "async_infer", True))
        self.timeout_ms = int(timeout_ms)
        self.stale_s = max(0.05, float(stale_s))
        self.retry_s = max(0.1, float(retry_s))
        self.bgr_to_rgb = bool(bgr_to_rgb)
        self._clock = clock
        self._log = logger or LOG

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._source: Optional[FrameSource] = source
        self._pending: Optional[tuple[Any, float]] = None
        self._pending_generation = 0

        self._last_det: Optional[Detection] = None
        self._last_detections: list[Detection] = []
        self._last_frame_t = 0.0
        self._frames = 0
        self._errors = 0
        self._drops = 0
        self._infer_ms = 0.0
        self._fps = 0.0
        self._last_infer_t = 0.0
        self._last_error = ""
        self._ready = False
        self._fatal = False

        # Filled by _configure() in the worker thread.
        self._vdevice: Any = None
        self._infer_model: Any = None
        self._configured: Any = None
        self._bindings: Any = None
        self._input_name = ""
        self._input_buffer: Optional[np.ndarray] = None
        self._outputs: list[dict[str, Any]] = []
        self._letterbox: Optional[Letterbox] = None
        self._model_info: dict[str, Any] = {}

    # -- public API ------------------------------------------------------
    def attach_source(self, source: Optional[FrameSource]) -> None:
        """Attach (or replace) the frame source. Safe to call at runtime."""
        with self._lock:
            self._source = source

    def submit_frame(self, frame: Any, t: Optional[float] = None) -> None:
        """Push one frame for benchmark/test use when no source is attached.

        The frame is read on the worker thread. Do not mutate its buffer after
        submitting; the Pi camera allocates a new frame per read, so the real
        path never aliases.
        """
        stamp = self._clock() if t is None else float(t)
        with self._lock:
            if self._pending is not None:
                self._drops += 1
            self._pending = (frame, stamp)
            self._pending_generation += 1

    @property
    def source(self) -> Optional[FrameSource]:
        return self._source

    @property
    def model_info(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._model_info)

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            self._ready_event.clear()
            self._ready = False
            self._running = True
            self._thread = threading.Thread(
                target=self._run, name="bf-anti-hailo", daemon=True
            )
            self._thread.start()
        return True

    def stop(self) -> None:
        # Close the safety gate immediately: healthy() reads _running.
        with self._lock:
            self._running = False
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
            if thread.is_alive():
                # The worker still owns the device; it releases in its own
                # finally block. Never release from the caller (double free).
                self._log.warning("Hailo worker did not stop within 5 s")
        with self._lock:
            self._thread = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def wait_ready(self, timeout_s: float = 10.0) -> bool:
        """Block until the HEF is configured and the device is ready.

        Frames are not required for this: the control loop still sees
        healthy() == False until a real frame has been processed, because the
        safety gate must stay closed without detections.
        """
        return bool(self._ready_event.wait(max(0.0, float(timeout_s))))

    def latest(
        self, max_age_s: Optional[float], now: Optional[float] = None
    ) -> Optional[Detection]:
        now = self._clock() if now is None else float(now)
        with self._lock:
            det = self._last_det
        if det is None:
            return None
        if max_age_s is not None and float(max_age_s) > 0.0:
            if (now - det.t) > float(max_age_s):
                return None
        return det

    def latest_detections(
        self, max_age_s: Optional[float], now: Optional[float] = None
    ) -> list[Detection]:
        """All detections from the newest processed frame (OSD helper)."""
        now = self._clock() if now is None else float(now)
        with self._lock:
            dets = list(self._last_detections)
        if max_age_s is not None and float(max_age_s) > 0.0:
            dets = [det for det in dets if (now - det.t) <= float(max_age_s)]
        return dets

    def healthy(self, now: Optional[float] = None) -> bool:
        now = self._clock() if now is None else float(now)
        with self._lock:
            if not self._running or not self._ready or self._fatal:
                return False
            stamp = self._last_frame_t
            return stamp > 0.0 and (now - stamp) <= self.stale_s

    def stats(self) -> DetectorStats:
        with self._lock:
            return DetectorStats(
                backend=self.backend,
                frames=self._frames,
                errors=self._errors,
                drops=self._drops,
                infer_ms=self._infer_ms,
                fps=self._fps,
                last_frame_t=self._last_frame_t,
                last_det_t=self._last_det.t if self._last_det is not None else 0.0,
                last_error=self._last_error,
                running=self._running,
                ready=self._ready,
            )

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    # -- worker ----------------------------------------------------------
    def _run(self) -> None:
        self._stop.clear()
        try:
            while not self._stop.is_set():
                if self._configured is None:
                    if self._fatal:
                        if self._stop.wait(0.5):
                            break
                        continue
                    if not self._try_configure():
                        if self._stop.wait(self.retry_s):
                            break
                        continue
                frame, stamp = self._next_frame()
                if frame is None:
                    if self._stop.wait(0.002):
                        break
                    continue
                try:
                    self._infer(frame, stamp)
                except Exception as exc:  # noqa: BLE001 - keep the loop alive
                    self._note_error("inference failed: %s" % (exc,))
                    self._log.warning("Hailo inference failed: %s", exc)
                    if self._stop.wait(0.05):
                        break
        finally:
            self._release()

    def _try_configure(self) -> bool:
        try:
            self._configure()
            return True
        except Exception as exc:  # noqa: BLE001 - hardware/driver errors vary
            message = "%s: %s" % (type(exc).__name__, exc)
            self._note_error("configure failed: " + message)
            if isinstance(exc, (ValueError, FileNotFoundError)):
                # Model/config mismatch: no amount of retrying will fix it.
                self._fatal = True
                self._log.error("Hailo configure refused: %s", message)
            else:
                self._log.error(
                    "Hailo configure failed (%s); retrying in %.1f s",
                    message, self.retry_s,
                )
            self._release()
            return False

    def _note_error(self, message: str) -> None:
        with self._lock:
            self._errors += 1
            self._last_error = str(message)

    def _next_frame(self) -> tuple[Optional[Any], float]:
        with self._lock:
            pending = self._pending
            if pending is not None:
                self._pending = None
        if pending is not None:
            return pending
        source = self._source
        if source is None:
            return None, 0.0
        try:
            frame, stamp = source.latest(self.stale_s, None)
        except Exception as exc:  # noqa: BLE001 - a camera error must not kill us
            self._note_error("source.latest failed: %s" % (exc,))
            return None, 0.0
        if frame is None:
            return None, 0.0
        return frame, float(stamp)

    def _configure(self) -> None:
        if not self.hef:
            self._fatal = True
            raise FileNotFoundError("detector.hef is empty")
        if not os.path.isfile(self.hef):
            self._fatal = True
            raise FileNotFoundError("HEF not found: %s" % self.hef)

        from hailo_platform import FormatType, VDevice  # lazy: laptop tests

        vdevice = VDevice()
        try:
            infer_model = vdevice.create_infer_model(self.hef)
            infer_model.set_batch_size(1)

            input_stream = infer_model.input()
            input_shape = tuple(int(v) for v in input_stream.shape)
            if len(input_shape) != 3:
                raise ValueError(
                    "unsupported HEF input shape %r (expected HWC)" % (input_shape,)
                )
            if input_shape[2] != 3:
                raise ValueError(
                    "unsupported HEF input shape %r (expected 3 channels)" % (input_shape,)
                )
            if (input_shape[1], input_shape[0]) != (self.input_width, self.input_height):
                self._log.warning(
                    "detector input config is %dx%d but HEF wants %dx%d; using HEF",
                    self.input_width, self.input_height, input_shape[1], input_shape[0],
                )
                self.input_width, self.input_height = int(input_shape[1]), int(input_shape[0])

            outputs = []
            for stream in infer_model.outputs:
                if not stream.is_nms:
                    try:
                        stream.set_format_type(FormatType.FLOAT32)
                    except Exception:  # noqa: BLE001 - native dtype is fine
                        pass
                outputs.append(stream)

            configured = infer_model.configure()
            try:
                configured.activate()
            except Exception as exc:  # noqa: BLE001 - scheduler mode auto-activates
                self._log.debug("configured.activate() skipped: %s", exc)

            bindings = configured.create_bindings()
            input_buffer = np.zeros(
                (int(input_shape[0]), int(input_shape[1]), 3), dtype=np.uint8
            )
            bindings.input().set_buffer(input_buffer)

            output_infos: list[dict[str, Any]] = []
            for stream in outputs:
                if not stream.is_nms:
                    try:
                        native_dtype = stream.format.type
                    except Exception:  # noqa: BLE001
                        native_dtype = None
                    if native_dtype is not None and native_dtype != FormatType.FLOAT32:
                        raise ValueError(
                            "raw output %s is %s, not FLOAT32; compile the HEF with "
                            "a float output format or use an NMS HEF"
                            % (stream.name, native_dtype)
                        )
                dtype = np.float32
                buffer = np.empty(stream.shape, dtype=dtype)
                bindings.output(stream.name).set_buffer(buffer)
                output_infos.append(
                    {
                        "name": stream.name,
                        "shape": tuple(int(v) for v in stream.shape),
                        "is_nms": bool(stream.is_nms),
                        "dtype": np.dtype(dtype).name,
                    }
                )
        except Exception:
            try:
                vdevice.release()
            except Exception:  # noqa: BLE001
                pass
            raise

        with self._lock:
            self._vdevice = vdevice
            self._infer_model = infer_model
            self._configured = configured
            self._bindings = bindings
            self._input_name = input_stream.name
            self._input_buffer = input_buffer
            self._outputs = output_infos
            self._model_info = {
                "hef": self.hef,
                "network": os.path.basename(self.hef),
                "input_name": input_stream.name,
                "input_shape": input_shape,
                "input_type": str(input_stream.format.type),
                "outputs": output_infos,
            }
            self._last_error = ""
            self._ready = True
        self._ready_event.set()
        self._log.info(
            "Hailo model ready: %s input=%s outputs=%s",
            os.path.basename(self.hef),
            input_shape,
            [info["name"] for info in output_infos],
        )

    def _infer(self, frame: Any, stamp: float) -> None:
        started = time.perf_counter()
        rgb = frame
        if self.bgr_to_rgb:
            try:
                import cv2

                if getattr(frame, "ndim", 0) == 3 and frame.shape[2] == 3:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception:  # noqa: BLE001 - OpenCV is required only here
                rgb = frame
        image, params = letterbox_image(
            rgb, self.input_width, self.input_height, pad_value=114
        )
        with self._lock:
            input_buffer = self._input_buffer
            bindings = self._bindings
            configured = self._configured
            outputs = list(self._outputs)
        if input_buffer is None or bindings is None or configured is None:
            return
        np.copyto(input_buffer, image)
        if self.async_infer and hasattr(configured, "run_async"):
            job = configured.run_async([bindings])
            job.wait(self.timeout_ms)
        else:
            configured.run([bindings], self.timeout_ms)

        now = self._clock()
        dets: list[Detection] = []
        failures: list[str] = []
        for info in outputs:
            name = info["name"]
            try:
                if info["is_nms"]:
                    raw = bindings.output(name).get_buffer(tf_format=True)
                    dets.extend(
                        parse_hailo_nms(
                            raw,
                            params,
                            conf_threshold=self.conf_threshold,
                            iou_threshold=self.iou_threshold,
                            max_detections=self.max_detections,
                            t=stamp if stamp > 0.0 else now,
                        )
                    )
                else:
                    raw = bindings.output(name).get_buffer()
                    dets.extend(
                        parse_yolo_v8_raw(
                            raw,
                            params,
                            conf_threshold=self.conf_threshold,
                            iou_threshold=self.iou_threshold,
                            max_detections=self.max_detections,
                            num_classes=len(self.class_names) or None,
                            class_names=self.class_names or None,
                            t=stamp if stamp > 0.0 else now,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - a bad output must not wedge the loop
                failures.append("%s: %s" % (name, exc))

        if failures:
            self._note_error("postprocess failed: " + "; ".join(failures))
        dets.sort(key=lambda det: (-det.conf, det.class_id, det.bbox[0], det.bbox[1]))
        if self.max_detections > 0:
            dets = dets[: self.max_detections]
        best = dets[0] if dets else None

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        frame_stamp = float(stamp) if stamp > 0.0 else now
        with self._lock:
            self._frames += 1
            self._last_detections = dets
            self._last_det = best
            self._last_frame_t = frame_stamp
            alpha = 0.1
            self._infer_ms = (
                elapsed_ms if self._infer_ms <= 0.0 else (1.0 - alpha) * self._infer_ms + alpha * elapsed_ms
            )
            dt = now - self._last_infer_t if self._last_infer_t > 0.0 else 0.0
            self._last_infer_t = now
            if dt > 0.0:
                inst = 1.0 / dt
                self._fps = inst if self._fps <= 0.0 else 0.9 * self._fps + 0.1 * inst
            self._ready = True
        self._ready_event.set()

    def _release(self) -> None:
        """Drop Python-side HailoRT objects, then release the VDevice.

        Ordering matters. HailoRT 4.23 segfaults if
        ConfiguredInferModel.deactivate() is called after buffers were bound
        (observed on the Pi5), and destroying bindings / the configured model
        after VDevice.release() also faults. So:
          1. clear our references,
          2. drop bindings + configured + infer model while the device lives,
          3. release the VDevice last.
        This runs only on the worker thread; stop() never releases.
        """
        with self._lock:
            bindings = self._bindings
            configured = self._configured
            infer_model = self._infer_model
            vdevice = self._vdevice
            self._bindings = None
            self._configured = None
            self._infer_model = None
            self._vdevice = None
            self._input_buffer = None
            self._outputs = []
            self._ready = False
            self._ready_event.clear()
        # CPython drops these immediately at reassignment; do it before the
        # device goes away so their C++ destructors run against a live device.
        bindings = None
        configured = None
        infer_model = None
        if vdevice is not None:
            try:
                vdevice.release()
            except Exception:  # noqa: BLE001
                pass
