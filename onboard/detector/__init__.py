"""Detector backends and factory.

build_detector(cfg, source=None) is the only entry point the app needs. The
Hailo backend needs frames; pass the camera source at construction (or call
detector.attach_source(camera) after the app builds the camera) so the camera
is opened exactly once.

Interfaces used by onboard.app:

    detector.start()
    detector.stop()
    detector.latest(max_age_s, now=None) -> Detection | None
    detector.healthy(now=None) -> bool

Optional extensions implemented by the hardware/replay backends:

    detector.attach_source(source)      # source.latest(max_age_s, now) -> (frame, t)
    detector.submit_frame(frame, t=None)
    detector.stats() -> DetectorStats
    detector.latest_detections(max_age_s, now=None) -> list[Detection]

All backends are importable without HailoRT/OpenCV; HailoRT is imported only
when the Hailo worker thread starts.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from onboard.detector.base import Detector, DetectorStats, FrameSource
from onboard.detector.hailo import HailoDetector
from onboard.detector.replay import ReplayDetector
from onboard.types import Detection

__all__ = [
    "Detector",
    "DetectorStats",
    "FrameSource",
    "HailoDetector",
    "NullDetector",
    "ReplayDetector",
    "build_detector",
]


class NullDetector:
    """No-op backend: healthy, never returns a detection."""

    backend = "null"

    def __init__(self, cfg: Any = None, clock=time.monotonic) -> None:
        self.cfg = cfg
        self._clock = clock
        self._running = False
        self._started_t = 0.0

    def start(self) -> None:
        self._running = True
        self._started_t = self._clock()

    def stop(self) -> None:
        self._running = False

    def latest(
        self, max_age_s: Optional[float], now: Optional[float] = None
    ) -> Optional[Detection]:
        return None

    def latest_detections(
        self, max_age_s: Optional[float], now: Optional[float] = None
    ) -> list[Detection]:
        return []

    def healthy(self, now: Optional[float] = None) -> bool:
        return True

    def attach_source(self, source: Optional[FrameSource]) -> None:
        """Accepted for uniform wiring; the null backend ignores frames."""
        return None

    def submit_frame(self, frame: Any, t: Optional[float] = None) -> None:
        return None

    @property
    def running(self) -> bool:
        return self._running

    def stats(self) -> DetectorStats:
        return DetectorStats(
            backend=self.backend,
            running=self._running,
            ready=True,
        )


def build_detector(
    cfg: Any,
    source: Optional[FrameSource] = None,
) -> Detector:
    """Instantiate the detector selected by cfg.backend.

    Parameters
    ----------
    cfg:
        onboard.config.DetectorConfig-shaped object.
    source:
        Optional frame source for the Hailo backend, usually the camera:
        source.latest(max_age_s, now) -> (frame_bgr, t_monotonic).

    Raises
    ------
    ValueError:
        Unknown backend, or backend=replay without a replay_path.
    """
    backend = str(getattr(cfg, "backend", "null") or "null").strip().lower()
    if backend in ("null", "none", "disabled"):
        return NullDetector(cfg)
    if backend in ("hailo", "hailo8", "hailo8l", "hailo_8l"):
        return HailoDetector(cfg, source=source)
    if backend in ("replay", "replay_file"):
        return ReplayDetector(cfg)
    raise ValueError(
        "unknown detector backend %r (expected hailo, replay or null)" % (backend,)
    )
