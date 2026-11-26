"""Shared value types for the onboard runtime.

Keep this module dependency-free (stdlib only) so every subsystem can import it
without pulling in OpenCV, HailoRT, or the simulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

BBox = tuple[float, float, float, float]

RC_MIN = 1000
RC_MAX = 2000
RC_MID = 1500


class Mode(str, Enum):
    """Onboard mode. Mirrors the user-visible state machine, not FC flight mode."""

    BOOT = "BOOT"
    PASSTHROUGH = "PASSTHROUGH"
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    TERMINAL = "TERMINAL"
    HIT = "HIT"
    FAULT = "FAULT"


@dataclass(frozen=True)
class Detection:
    """One detector result, in the source frame's pixel coordinates."""

    bbox: BBox
    conf: float
    class_id: int = 0
    t: float = 0.0

    @property
    def width_px(self) -> float:
        return max(0.0, float(self.bbox[2]) - float(self.bbox[0]))

    @property
    def height_px(self) -> float:
        return max(0.0, float(self.bbox[3]) - float(self.bbox[1]))

    @property
    def center(self) -> tuple[float, float]:
        return (
            (float(self.bbox[0]) + float(self.bbox[2])) * 0.5,
            (float(self.bbox[1]) + float(self.bbox[3])) * 0.5,
        )


@dataclass
class FcState:
    """Latest flight-controller snapshot.

    All timestamps are time.monotonic() values. A zero timestamp means
    "never received". Consumers must check age before trusting a field; the
    safety supervisor is the single place that decides what is fresh enough.
    """

    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    altitude_m: float = 0.0
    vario_mps: float = 0.0
    rc: tuple[int, ...] = (RC_MID,) * 16
    armed: bool = False
    flight_mode_flags: int = 0
    extended_flight_mode_flags: bytes = b""
    voltage_v: float = 0.0
    current_a: float = 0.0
    mah: int = 0
    rssi: int = 0
    position_enu: Optional[tuple[float, float, float]] = None
    velocity_enu: Optional[tuple[float, float, float]] = None

    att_t: float = 0.0
    alt_t: float = 0.0
    rc_t: float = 0.0
    status_t: float = 0.0
    analog_t: float = 0.0
    pos_t: float = 0.0
    link_t: float = 0.0

    errors: int = 0
    generation: int = 0

    def age_s(self, stamp: float, now: float) -> float:
        if stamp <= 0.0:
            return float("inf")
        return max(0.0, float(now) - float(stamp))

    def att_age_s(self, now: float) -> float:
        return self.age_s(self.att_t, now)

    def alt_age_s(self, now: float) -> float:
        return self.age_s(self.alt_t, now)

    def rc_age_s(self, now: float) -> float:
        return self.age_s(self.rc_t, now)

    def status_age_s(self, now: float) -> float:
        return self.age_s(self.status_t, now)

    def link_age_s(self, now: float) -> float:
        return self.age_s(self.link_t, now)

    def has_position(self) -> bool:
        return self.position_enu is not None and self.velocity_enu is not None


@dataclass
class ControlOutput:
    """Controller result for one tick.

    channels is a 4-tuple (roll, pitch, throttle, yaw) or None. None means the
    onboard layer must not command the FC; the RC mux falls back to the pilot.
    Aux/ARM channels are never produced here.
    """

    mode: Mode = Mode.PASSTHROUGH
    channels: Optional[tuple[int, int, int, int]] = None
    command: object = None
    confirmed: bool = False
    reason: str = ""
