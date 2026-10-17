"""Display path: camera frame -> OSD canvas -> Linux framebuffer.

Import light: the heavy OpenCV work lives in .osd, and .pipeline only builds
the OSD renderer and framebuffer sink when start() is called.
"""
from .fbdev import FBDevSink, FbGeometry, read_geometry
from .osd import (
    COLOR_BOX,
    COLOR_CROSSHAIR,
    COLOR_TEXT,
    Fit,
    OsdRenderer,
    OsdStatus,
    fit_letterbox,
    format_readout,
    to_bgr8,
)
from .pipeline import DisplayPipeline, DisplayStats

__all__ = [
    "COLOR_BOX",
    "COLOR_CROSSHAIR",
    "COLOR_TEXT",
    "DisplayPipeline",
    "DisplayStats",
    "FBDevSink",
    "FbGeometry",
    "Fit",
    "OsdRenderer",
    "OsdStatus",
    "fit_letterbox",
    "format_readout",
    "read_geometry",
    "to_bgr8",
]
