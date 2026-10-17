"""OSD compositor: source frame -> 720x576 BGRA canvas for /dev/fb0.

The compositor is the only place that knows about display geometry. It takes
the camera frame, scales it into the framebuffer with a 16:9 -> 4:3 letterbox
by default, and draws the operator HUD:

* detection box (source pixel coordinates are mapped through the same fit),
* centre crosshair,
* lock / mode / safety gate,
* altitude, vertical speed, battery voltage, range,
* loop rate, display latency and dropped-frame count.

Everything is cv2 + numpy, no fonts outside OpenCV, no GUI window, no X11.
A custom TTF can be supplied via video.font_path; when cv2.freetype is not
available the renderer falls back to the built-in Hershey font instead of
failing.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# BGRA palette (OpenCV order: B, G, R, A).
COLOR_BG = (0, 0, 0, 255)
COLOR_TEXT = (235, 235, 235, 255)
COLOR_DIM = (170, 170, 170, 255)
COLOR_SHADOW = (10, 10, 10, 255)
COLOR_BOX = (0, 255, 0, 255)
COLOR_BOX_ACQ = (0, 200, 255, 255)
COLOR_CROSSHAIR = (240, 240, 240, 255)
COLOR_OK = (0, 255, 0, 255)
COLOR_WARN = (0, 0, 255, 255)
COLOR_LOCK = (0, 255, 0, 255)
COLOR_NO_LOCK = (0, 165, 255, 255)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_THICKNESS = 1
_LINE = cv2.LINE_AA
_MARGIN = 8


@dataclass(frozen=True)
class Fit:
    """Mapping from source frame pixels to destination canvas pixels."""

    scale_x: float
    scale_y: float
    offset_x: int
    offset_y: int
    content_w: int
    content_h: int

    def map_point(self, x: float, y: float) -> Tuple[float, float]:
        return (
            float(x) * self.scale_x + self.offset_x,
            float(y) * self.scale_y + self.offset_y,
        )


@dataclass(frozen=True)
class OsdStatus:
    """Pipeline health numbers rendered on the HUD right-hand side."""

    loop_hz: float = 0.0
    latency_ms: float = 0.0
    display_fps: float = 0.0
    dropped: int = 0


def fit_letterbox(
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    letterbox: bool = True,
) -> Fit:
    """Compute a centred letterbox fit (or a full-frame stretch)."""
    src_w = int(src_w)
    src_h = int(src_h)
    dst_w = int(dst_w)
    dst_h = int(dst_h)
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise ValueError("fit_letterbox requires positive dimensions")
    if not letterbox:
        return Fit(
            scale_x=dst_w / float(src_w),
            scale_y=dst_h / float(src_h),
            offset_x=0,
            offset_y=0,
            content_w=dst_w,
            content_h=dst_h,
        )
    scale = min(dst_w / float(src_w), dst_h / float(src_h))
    content_w = max(1, min(dst_w, int(round(src_w * scale))))
    content_h = max(1, min(dst_h, int(round(src_h * scale))))
    return Fit(
        scale_x=content_w / float(src_w),
        scale_y=content_h / float(src_h),
        offset_x=(dst_w - content_w) // 2,
        offset_y=(dst_h - content_h) // 2,
        content_w=content_w,
        content_h=content_h,
    )


def to_bgr8(frame: Any) -> np.ndarray:
    """Normalise a camera frame to contiguous uint8 BGR."""
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim != 3:
        raise ValueError("unsupported frame shape: %r" % (arr.shape,))
    channels = arr.shape[2]
    if channels == 3:
        return np.ascontiguousarray(arr)
    if channels == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    raise ValueError("unsupported channel count: %d" % channels)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _mode_text(out: Any) -> str:
    if out is None:
        return "BOOT"
    mode = getattr(out, "mode", None)
    if mode is None:
        return "BOOT"
    value = getattr(mode, "value", mode)
    return str(value).upper()


def _lock_text(out: Any, detection: Any) -> str:
    if out is not None and bool(getattr(out, "confirmed", False)):
        return "LOCK"
    if detection is not None:
        return "ACQ"
    return "NO LOCK"


def _safety_text(decision: Any) -> str:
    if decision is None:
        return "N/A"
    if bool(getattr(decision, "allow_auto", False)):
        return "OK"
    reason = str(getattr(decision, "reason", "") or "deny").upper()
    return reason[:28]


def format_readout(
    out: Any = None,
    state: Any = None,
    decision: Any = None,
    status: Optional[OsdStatus] = None,
    detection: Any = None,
) -> Dict[str, str]:
    """Pure formatter used by the HUD and by unit tests."""
    status = status or OsdStatus()
    mode = _mode_text(out)
    lock = _lock_text(out, detection)
    altitude = _as_float(getattr(state, "altitude_m", 0.0))
    vario = _as_float(getattr(state, "vario_mps", 0.0))
    voltage = _as_float(getattr(state, "voltage_v", 0.0))
    rng = float("nan")
    command = getattr(out, "command", None) if out is not None else None
    if command is not None:
        try:
            rng = float(getattr(command, "range_est_m", float("nan")))
        except (TypeError, ValueError):
            rng = float("nan")
    return {
        "mode": "MODE %s" % mode,
        "lock": lock,
        "safety": "SAFE %s" % _safety_text(decision),
        "alt": "ALT %6.1fm  VS %+5.1fm/s" % (altitude, vario),
        "battery": ("BAT %5.2fV" % voltage) if voltage > 0.0 else "BAT  --.--V",
        "range": ("RNG %5.1fm" % rng) if math.isfinite(rng) else "RNG   ---",
        "loop": "LOOP %5.1fHz" % _as_float(status.loop_hz),
        "latency": "LAT %5.1fms" % _as_float(status.latency_ms),
        "display": "DSP %4.1ffps  DRP %d"
        % (_as_float(status.display_fps), int(status.dropped)),
    }


class _TextEngine:
    """putText wrapper with an optional FreeType font, never fatal."""

    def __init__(self, font_path: str = "") -> None:
        self._face = None
        if font_path:
            self._face = self._load(font_path)

    @staticmethod
    def _load(font_path: str) -> Any:
        if not os.path.isfile(font_path):
            return None
        freetype = getattr(cv2, "freetype", None)
        if freetype is None:
            return None
        try:
            face = freetype.createFreeType2()
            face.loadFontData(font_path, 0)
            return face
        except Exception:  # noqa: BLE001 - fall back to Hershey
            return None

    @property
    def uses_truetype(self) -> bool:
        return self._face is not None

    @staticmethod
    def _height(scale: float) -> int:
        return max(8, int(round(float(scale) * 26.0)))

    def text_size(self, text: str, scale: float) -> Tuple[int, int, int]:
        if self._face is not None:
            try:
                size, baseline = self._face.getTextSize(text, self._height(scale), -1)
                return int(size[0]), int(size[1]), int(baseline)
            except Exception:  # noqa: BLE001
                pass
        width, height = cv2.getTextSize(text, _FONT, scale, _THICKNESS)[0]
        return int(width), int(height), 0

    def put(self, img: np.ndarray, text: str, org: Tuple[int, int], scale: float,
            color: Tuple[int, int, int, int]) -> None:
        if not text:
            return
        if self._face is not None:
            try:
                self._face.putText(
                    img, text, org, self._height(scale), color,
                    -1, _LINE, True,
                )
                return
            except Exception:  # noqa: BLE001
                pass
        cv2.putText(img, text, org, _FONT, scale, color, _THICKNESS, _LINE)


class OsdRenderer:
    """Renders the display canvas at a fixed output size."""

    def __init__(
        self,
        width: int = 720,
        height: int = 576,
        osd: bool = True,
        letterbox: bool = True,
        font_path: str = "",
        draw_box: bool = True,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("OSD output size must be positive")
        self.osd = bool(osd)
        self.letterbox = bool(letterbox)
        self.draw_box = bool(draw_box)
        self._text = _TextEngine(str(font_path) if font_path else "")

    # -- geometry --------------------------------------------------------
    def set_output_size(self, width: int, height: int) -> None:
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("OSD output size must be positive")
        self.width = width
        self.height = height

    def blank(self) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255
        return canvas

    # -- main entry point ------------------------------------------------
    def render(
        self,
        frame: Any,
        detection: Any = None,
        out: Any = None,
        state: Any = None,
        decision: Any = None,
        now: float = 0.0,
        status: Optional[OsdStatus] = None,
    ) -> np.ndarray:
        """Return a self.height x self.width BGRA canvas."""
        canvas = self.blank()
        fit: Optional[Fit] = None
        if frame is not None:
            bgr = to_bgr8(frame)
            src_h, src_w = bgr.shape[:2]
            fit = fit_letterbox(src_w, src_h, self.width, self.height, self.letterbox)
            interp = cv2.INTER_AREA if fit.scale_x < 1.0 else cv2.INTER_LINEAR
            resized = cv2.resize(
                bgr, (fit.content_w, fit.content_h), interpolation=interp
            )
            x0 = max(0, fit.offset_x)
            y0 = max(0, fit.offset_y)
            x1 = min(self.width, x0 + resized.shape[1])
            y1 = min(self.height, y0 + resized.shape[0])
            if x1 > x0 and y1 > y0:
                canvas[y0:y1, x0:x1, :3] = resized[: y1 - y0, : x1 - x0]
        if self.osd:
            if fit is not None and self.draw_box and detection is not None:
                self._draw_detection(canvas, detection, fit, out)
            self._draw_hud(canvas, detection, out, state, decision, now, status)
            self._draw_crosshair(canvas)
        return canvas

    # -- overlay pieces --------------------------------------------------
    def _draw_detection(
        self, canvas: np.ndarray, detection: Any, fit: Fit, out: Any
    ) -> None:
        try:
            x0, y0, x1, y1 = (float(v) for v in detection.bbox)
        except (AttributeError, TypeError, ValueError):
            return
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
            return
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        mx0, my0 = fit.map_point(x0, y0)
        mx1, my1 = fit.map_point(x1, y1)
        px0 = int(max(0, min(self.width - 1, round(mx0))))
        py0 = int(max(0, min(self.height - 1, round(my0))))
        px1 = int(max(0, min(self.width - 1, round(mx1))))
        py1 = int(max(0, min(self.height - 1, round(my1))))
        if px1 - px0 < 2 or py1 - py0 < 2:
            return
        confirmed = bool(getattr(out, "confirmed", False)) if out is not None else False
        color = COLOR_BOX if confirmed else COLOR_BOX_ACQ
        cv2.rectangle(canvas, (px0, py0), (px1, py1), color, 2, _LINE)
        try:
            conf = float(getattr(detection, "conf", float("nan")))
        except (TypeError, ValueError):
            conf = float("nan")
        label = ("%.2f" % conf) if math.isfinite(conf) else ""
        if label:
            tx = max(0, min(self.width - 40, px0))
            ty = max(12, py0 - 4)
            self._text.put(canvas, label, (tx + 1, ty + 1), 0.42, COLOR_SHADOW)
            self._text.put(canvas, label, (tx, ty), 0.42, color)

    def _draw_hud(
        self,
        canvas: np.ndarray,
        detection: Any,
        out: Any,
        state: Any,
        decision: Any,
        now: float,
        status: Optional[OsdStatus],
    ) -> None:
        status = status or OsdStatus()
        readout = format_readout(out, state, decision, status, detection)
        mode = readout["mode"]
        approved = bool(getattr(decision, "allow_auto", False)) if decision is not None else False
        mode_color = COLOR_TEXT
        mode_name = _mode_text(out)
        if mode_name in ("HIT", "TERMINAL"):
            mode_color = COLOR_WARN
        elif mode_name == "TRACK":
            mode_color = COLOR_OK
        self._put(canvas, mode, (_MARGIN, 20), 0.55, mode_color)
        self._put(
            canvas, readout["safety"], (_MARGIN, 40), 0.45,
            COLOR_OK if approved else COLOR_WARN,
        )
        if detection is not None:
            try:
                stamp = float(getattr(detection, "t", 0.0))
            except (TypeError, ValueError):
                stamp = 0.0
            if stamp > 0.0 and now > 0.0:
                age_ms = max(0.0, (float(now) - stamp) * 1000.0)
                self._put(canvas, "TGT %4.0fms" % age_ms, (_MARGIN, 60), 0.42, COLOR_DIM)
        lock = readout["lock"]
        lock_color = COLOR_LOCK if lock == "LOCK" else COLOR_NO_LOCK
        lock_scale = 0.75 if lock == "LOCK" else 0.55
        lock_w = self._text.text_size(lock, lock_scale)[0]
        self._put(
            canvas, lock, (max(_MARGIN, (self.width - lock_w) // 2), 24),
            lock_scale, lock_color,
        )
        right_x = self.width - _MARGIN
        self._put_right(canvas, readout["loop"], right_x, 20, 0.45, COLOR_TEXT)
        self._put_right(canvas, readout["latency"], right_x, 40, 0.45, COLOR_TEXT)
        self._put_right(canvas, readout["display"], right_x, 60, 0.42, COLOR_DIM)
        self._put(canvas, readout["alt"], (_MARGIN, self.height - 28), 0.45, COLOR_TEXT)
        self._put(canvas, readout["battery"], (_MARGIN, self.height - 8), 0.45, COLOR_DIM)
        self._put_right(
            canvas, readout["range"], right_x, self.height - 28, 0.45, COLOR_TEXT
        )
        self._put_right(
            canvas, "BOX %s" % ("ON" if self.draw_box else "OFF"),
            right_x, self.height - 8, 0.42, COLOR_DIM,
        )

    def _draw_crosshair(self, canvas: np.ndarray) -> None:
        cx = self.width // 2
        cy = self.height // 2
        line = 14
        gap = 4
        cv2.line(canvas, (cx - line, cy), (cx - gap, cy), COLOR_CROSSHAIR, 1, _LINE)
        cv2.line(canvas, (cx + gap, cy), (cx + line, cy), COLOR_CROSSHAIR, 1, _LINE)
        cv2.line(canvas, (cx, cy - line), (cx, cy - gap), COLOR_CROSSHAIR, 1, _LINE)
        cv2.line(canvas, (cx, cy + gap), (cx, cy + line), COLOR_CROSSHAIR, 1, _LINE)
        # Exact centre pixel: cheap, deterministic, and used by the tests.
        canvas[cy, cx] = COLOR_CROSSHAIR

    def _put(
        self,
        canvas: np.ndarray,
        text: str,
        org: Tuple[int, int],
        scale: float,
        color: Tuple[int, int, int, int],
    ) -> None:
        self._text.put(canvas, text, (org[0] + 1, org[1] + 1), scale, COLOR_SHADOW)
        self._text.put(canvas, text, org, scale, color)

    def _put_right(
        self,
        canvas: np.ndarray,
        text: str,
        right_x: int,
        y: int,
        scale: float,
        color: Tuple[int, int, int, int],
    ) -> None:
        width = self._text.text_size(text, scale)[0]
        x = max(0, int(right_x) - int(width))
        self._put(canvas, text, (x, y), scale, color)
