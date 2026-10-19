#!/usr/bin/env python3
"""Stream the Pi5 OSD framebuffer as MJPEG for a local browser.

This is a read-only viewer for /dev/fb0: the onboard app still owns the
framebuffer and OSD.  It is intentionally separate from the flight runtime so
starting/stopping the viewer can never affect the control loop.

Usage:
    python3 scripts/pi5_osd_stream.py --fb /dev/fb0 --port 8088
    # browse http://<pi-ip>:8088/  (or /stream.mjpg, /snapshot.jpg)
"""
from __future__ import annotations

import argparse
import mmap
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - Pi has cv2
    raise SystemExit("OpenCV is required for the OSD stream: %s" % exc)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from onboard.video.fbdev import read_geometry


class FbReader:
    def __init__(
        self,
        device: str = "/dev/fb0",
        width: int = 720,
        height: int = 576,
        stride: int = 0,
        quality: int = 80,
    ) -> None:
        geo = read_geometry(device)
        self.width = int(geo.width if geo else width)
        self.height = int(geo.height if geo else height)
        self.stride = int(geo.stride if geo else (stride or self.width * 4))
        self.bpp = int(geo.bpp if geo else 32)
        if self.bpp != 32:
            raise SystemExit("only 32-bpp BGRA framebuffers are supported (got %d)" % self.bpp)
        self.device = device
        self.quality = max(1, min(100, int(quality)))
        self.frame_bytes = self.stride * self.height
        self.fd = os.open(device, os.O_RDONLY)
        self.mm = mmap.mmap(
            self.fd,
            self.frame_bytes,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ,
        )
        self._lock = threading.Lock()

    def jpeg(self) -> bytes:
        with self._lock:
            raw = self.mm[: self.frame_bytes]
        arr = np.frombuffer(raw, dtype=np.uint8)
        arr = arr.reshape(self.height, self.stride // 4, 4)
        arr = arr[:, : self.width, :3]  # BGRA -> BGR
        ok, enc = cv2.imencode(
            ".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        )
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return enc.tobytes()

    def close(self) -> None:
        try:
            self.mm.close()
        finally:
            os.close(self.fd)


HTML = b"""<!doctype html><html><head><meta charset=\"utf-8\">
<title>BF_Anti Pi5 OSD</title>
<style>body{background:#111;color:#ddd;font-family:monospace;text-align:center}
img{max-width:100%;image-rendering:auto;border:1px solid #444}</style></head>
<body><h3>BF_Anti Pi5 OSD (fb0)</h3><img src=\"/stream.mjpg\" alt=\"stream\">
<p><a href=\"/snapshot.jpg\">snapshot.jpg</a></p></body></html>"""


class Handler(BaseHTTPRequestHandler):
    reader: FbReader
    fps: float

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _jpeg(self) -> bytes:
        return self.reader.jpeg()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(HTML)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(HTML)
            return
        if path == "/snapshot.jpg":
            try:
                data = self._jpeg()
            except Exception as exc:  # noqa: BLE001
                self.send_error(503, "framebuffer error: %s" % exc)
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return
        if path != "/stream.mjpg":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            while True:
                jpg = self._jpeg()
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                time.sleep(1.0 / max(1.0, self.fps))
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fb", default="/dev/fb0")
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--stride", type=int, default=0)
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args(argv)

    reader = FbReader(args.fb, args.width, args.height, args.stride, args.quality)
    Handler.reader = reader
    Handler.fps = max(1.0, float(args.fps))
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(
        "OSD stream: http://%s:%d/  (%dx%d, %d bpp, %.1f fps)"
        % (args.host, args.port, reader.width, reader.height, reader.bpp, Handler.fps),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
