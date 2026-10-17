"""Detector interface shared by the Hailo, replay, and null backends.

The onboard control loop only needs two calls:

* latest(max_age_s, now) -> newest onboard.types.Detection or None
* healthy(now) -> False when the safety gate must stay closed

Frames do not have to be owned by the detector. A detector can either pull
from a FrameSource (the camera source exposes exactly the same
latest(max_age_s, now) -> (frame, t) contract) or accept frames pushed with
submit_frame(). This keeps the camera single-owner: the app opens the camera
once and attaches it to the detector instead of letting the detector open
/dev/video* a second time.

This module is stdlib-only on purpose: importing it must work on a laptop with
no HailoRT, no OpenCV, and no camera.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from onboard.types import Detection

__all__ = ["Detector", "DetectorStats", "FrameSource"]


@dataclass(frozen=True)
class DetectorStats:
    """Detector health snapshot. Times are time.monotonic() seconds."""

    backend: str = ""
    frames: int = 0
    errors: int = 0
    drops: int = 0
    infer_ms: float = 0.0
    fps: float = 0.0
    last_frame_t: float = 0.0
    last_det_t: float = 0.0
    last_error: str = ""
    running: bool = False
    ready: bool = False


@runtime_checkable
class FrameSource(Protocol):
    """Anything the detector can pull frames from.

    onboard.camera.CameraSource implements this signature. max_age_s may be
    None to disable the freshness check; the returned timestamp is
    time.monotonic() or 0.0 when no frame is available.
    """

    def latest(
        self, max_age_s: Optional[float], now: Optional[float] = None
    ) -> Tuple[Optional[Any], float]:
        ...


@runtime_checkable
class Detector(Protocol):
    """Minimal detector contract used by onboard.app.OnboardApp."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def latest(
        self, max_age_s: Optional[float], now: Optional[float] = None
    ) -> Optional[Detection]:
        ...

    def healthy(self, now: Optional[float] = None) -> bool:
        ...
