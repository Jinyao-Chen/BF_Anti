"""Linux framebuffer sink for the J7 composite output.

The Pi 5 exposes the RP1 VEC CVBS output as /dev/fb0 (drm-rp1-vecdrmf,
720x576x32 on the verified setup). This module writes one BGRA frame directly
into the mapped framebuffer; there is no X11, no SDL, and no HDMI converter in
the path.

Geometry is read from sysfs, never hardcoded:

    /sys/class/graphics/fb0/virtual_size   -> "720,576"
    /sys/class/graphics/fb0/bits_per_pixel -> 32
    /sys/class/graphics/fb0/stride         -> 2880 (optional)

When sysfs is unavailable (e.g. a unit test with a regular file) the sink
falls back to the configured width/height at 32 bpp with a tight stride.

The framebuffer console can blank after its idle timeout (sysfs blank=4),
which would kill the CVBS output even though the control loop is fine. The
sink therefore sends FBIOBLANK/unblank on open and again every few hundred
frames; failure to issue the ioctl (regular file, no permission) is ignored.

Failure policy: a missing/unusable framebuffer is not fatal to the control
loop. open() returns False and records last_error; write() returns False and
the display thread keeps retrying later. Nothing here raises on absent
hardware.
"""
from __future__ import annotations

import fcntl
import logging
import mmap
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

LOG = logging.getLogger("bf_anti.video.fbdev")

DEFAULT_SYSFS_ROOT = "/sys/class/graphics"
_SUPPORTED_BPP = (16, 24, 32)

# linux/fb.h
FBIOBLANK = 0x4611
FB_BLANK_UNBLANK = 0
# Re-assert unblank every ~10 s at 25 fps. Console blanking is timer driven
# and is not reset by mmap writes, so a periodic ioctl is required.
UNBLANK_EVERY_FRAMES = 250


@dataclass(frozen=True)
class FbGeometry:
    """Framebuffer geometry as reported by sysfs."""

    width: int
    height: int
    bpp: int
    stride: int
    name: str = ""

    @property
    def bytes_per_pixel(self) -> int:
        return int(self.bpp) // 8

    @property
    def row_bytes(self) -> int:
        return int(self.width) * self.bytes_per_pixel

    @property
    def frame_bytes(self) -> int:
        return int(self.stride) * int(self.height)


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="ascii", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def read_geometry(
    device: str = "/dev/fb0", sysfs_root: str = DEFAULT_SYSFS_ROOT
) -> Optional[FbGeometry]:
    """Read fb geometry from sysfs; return None when it cannot be parsed."""
    node = os.path.basename(str(device).rstrip("/"))
    if not node:
        return None
    base = os.path.join(sysfs_root, node)
    virtual = _read_text(os.path.join(base, "virtual_size"))
    bpp_text = _read_text(os.path.join(base, "bits_per_pixel"))
    if not virtual or not bpp_text:
        return None
    try:
        width_s, height_s = virtual.split(",")[:2]
        width = int(width_s)
        height = int(height_s)
        bpp = int(bpp_text)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or bpp <= 0:
        return None
    stride_text = _read_text(os.path.join(base, "stride"))
    try:
        stride = int(stride_text)
    except (TypeError, ValueError):
        stride = 0
    min_stride = width * (bpp // 8 if bpp >= 8 else 1)
    if stride < min_stride:
        stride = min_stride
    name = _read_text(os.path.join(base, "name"))
    return FbGeometry(width=width, height=height, bpp=bpp, stride=stride, name=name)


class FBDevSink:
    """Memory-mapped BGRA writer for a Linux framebuffer device.

    Parameters
    ----------
    device:
        Framebuffer node, usually /dev/fb0.
    width, height:
        Requested geometry. Only used when sysfs cannot report the real one.
    sysfs_root:
        Root of the graphics class (tests point this at a temp tree).
    """

    def __init__(
        self,
        device: str = "/dev/fb0",
        width: int = 720,
        height: int = 576,
        sysfs_root: str = DEFAULT_SYSFS_ROOT,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.device = str(device)
        self.requested_width = int(width)
        self.requested_height = int(height)
        self.sysfs_root = str(sysfs_root)
        self._log = logger or LOG

        self._fd: int = -1
        self._mm: Optional[mmap.mmap] = None
        self._geometry: Optional[FbGeometry] = None
        self._frames_written = 0
        self._bytes_written = 0
        self._last_error = ""
        self._unblank_ok: Optional[bool] = None

    # -- introspection ---------------------------------------------------
    @property
    def geometry(self) -> Optional[FbGeometry]:
        return self._geometry

    @property
    def is_open(self) -> bool:
        return self._mm is not None and self._fd >= 0

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    @property
    def last_error(self) -> str:
        return self._last_error

    def probe(self) -> FbGeometry:
        """Return sysfs geometry, or a configured fallback when absent."""
        geo = read_geometry(self.device, self.sysfs_root)
        if geo is not None:
            return geo
        width = max(1, self.requested_width)
        height = max(1, self.requested_height)
        return FbGeometry(
            width=width, height=height, bpp=32, stride=width * 4, name="fallback"
        )

    # -- lifecycle -------------------------------------------------------
    def open(self) -> bool:
        """Map the framebuffer. Returns False (without raising) on failure."""
        if self.is_open:
            return True
        self.close()
        geo = self.probe()
        if geo.bpp not in _SUPPORTED_BPP:
            self._last_error = "unsupported framebuffer bpp: %d" % geo.bpp
            self._log.warning("%s (%s)", self._last_error, self.device)
            return False
        try:
            fd = os.open(self.device, os.O_RDWR)
        except OSError as exc:
            self._last_error = "open %s: %s" % (self.device, exc)
            self._log.warning("%s", self._last_error)
            return False
        try:
            mm = mmap.mmap(
                fd,
                geo.frame_bytes,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
            )
        except (OSError, ValueError, BufferError) as exc:
            os.close(fd)
            self._last_error = "mmap %s (%d B): %s" % (
                self.device, geo.frame_bytes, exc,
            )
            self._log.warning("%s", self._last_error)
            return False
        self._fd = fd
        self._mm = mm
        self._geometry = geo
        self._last_error = ""
        self._log.info(
            "fbdev %s mapped: %dx%d %d bpp stride=%d (%s)",
            self.device, geo.width, geo.height, geo.bpp, geo.stride, geo.name,
        )
        self.unblank()
        return True

    def close(self) -> None:
        """Unmap and close. Safe to call repeatedly."""
        mm, self._mm = self._mm, None
        fd, self._fd = self._fd, -1
        if mm is not None:
            try:
                mm.close()
            except (BufferError, OSError):
                pass
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        self._geometry = None

    def __enter__(self) -> "FBDevSink":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # -- data path -------------------------------------------------------
    def write(self, frame: Any) -> bool:
        """Write one HxWx4 uint8 BGRA frame. Returns True on success.

        Raises ValueError for a caller bug (wrong shape/dtype/layout), returns
        False for device-level failures. The caller (display thread) owns
        serialisation; this class is not thread-safe by itself.
        """
        if not self.is_open or self._mm is None or self._geometry is None:
            self._last_error = "framebuffer not open"
            return False
        arr = np.asarray(frame)
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError("expected HxWx4 BGRA frame, got shape %r" % (arr.shape,))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        geo = self._geometry
        if arr.shape[0] != geo.height or arr.shape[1] != geo.width:
            raise ValueError(
                "frame %dx%d does not match framebuffer %dx%d"
                % (arr.shape[1], arr.shape[0], geo.width, geo.height)
            )
        try:
            data = self._pack(arr, geo)
            view = np.frombuffer(self._mm, dtype=np.uint8).reshape(
                geo.height, geo.stride
            )
            row_bytes = geo.row_bytes
            view[:, :row_bytes] = data
        except (OSError, ValueError, BufferError) as exc:
            self._last_error = "fb write failed: %s" % (exc,)
            self._log.error("%s", self._last_error)
            return False
        self._frames_written += 1
        self._bytes_written += geo.frame_bytes
        self._last_error = ""
        if self._frames_written % UNBLANK_EVERY_FRAMES == 0:
            self.unblank()
        return True

    write_bgra = write

    def unblank(self) -> bool:
        """Ask the fb driver to unblank; returns False when unsupported.

        The Pi console blanks on a timer regardless of mmap traffic, so the
        display thread keeps the CVBS CRTC awake with this cheap ioctl.
        """
        if self._fd < 0:
            return False
        try:
            fcntl.ioctl(self._fd, FBIOBLANK, FB_BLANK_UNBLANK)
        except (OSError, ValueError):
            if self._unblank_ok is None:
                self._log.debug("FBIOBLANK unsupported on %s", self.device)
            self._unblank_ok = False
            return False
        if self._unblank_ok is None:
            self._log.info("fbdev %s unblanked", self.device)
        self._unblank_ok = True
        return True

    @staticmethod
    def _pack(arr: np.ndarray, geo: FbGeometry) -> np.ndarray:
        height = geo.height
        width = geo.width
        if geo.bpp == 32:
            return np.ascontiguousarray(arr).reshape(height, width * 4)
        if geo.bpp == 24:
            return np.ascontiguousarray(arr[:, :, :3]).reshape(height, width * 3)
        if geo.bpp == 16:
            b = arr[:, :, 0].astype(np.uint16)
            g = arr[:, :, 1].astype(np.uint16)
            r = arr[:, :, 2].astype(np.uint16)
            value = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
            return np.ascontiguousarray(value).view(np.uint8).reshape(height, width * 2)
        raise ValueError("unsupported bpp: %d" % geo.bpp)
