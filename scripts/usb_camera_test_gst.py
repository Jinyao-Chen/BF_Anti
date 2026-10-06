"""Pure GStreamer-based USB camera FPS tester.

Bypasses OpenCV's V4L2 capture (which on this board does not include
GStreamer support) by pulling frames directly from a Gst.appsink.
OpenCV is still used for imshow/putText if a display is available.
"""
import os
import time
import argparse
import threading
from collections import deque

import numpy as np
import cv2

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst


def build_pipeline(device, fmt, w, h, fps, io_mode):
    if fmt == 'MJPG':
        src_caps = f"image/jpeg,width={w},height={h},framerate={fps}/1"
        decode = " ! jpegdec"
    else:
        src_caps = f"video/x-raw,format={fmt},width={w},height={h},framerate={fps}/1"
        decode = ""
    return (
        f"v4l2src device={device} io-mode={io_mode} ! {src_caps}{decode} "
        f"! videoconvert ! video/x-raw,format=BGR "
        f"! appsink name=sink emit-signals=true sync=false max-buffers=2 drop=true"
    )


class GstCapture:
    """Minimal Gst pipeline -> numpy BGR frames."""

    def __init__(self, pipeline_str):
        Gst.init(None)
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name('sink')
        self.sink.set_property('emit-signals', True)
        self.sink.connect('new-sample', self._on_new_sample)

        self._lock = threading.Lock()
        self._latest = None        # (frame, t_arrival)
        self._frame_seq = 0

    def _on_new_sample(self, sink):
        sample = sink.emit('pull-sample')
        if sample is None:
            return Gst.FlowReturn.ERROR
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w = caps.get_value('width')
        h = caps.get_value('height')
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.ERROR
        try:
            frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape(h, w, 3).copy()
        finally:
            buf.unmap(mapinfo)
        with self._lock:
            self._latest = (frame, time.perf_counter())
            self._frame_seq += 1
        return Gst.FlowReturn.OK

    def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)

    def read_blocking(self, last_seq, timeout=1.0):
        """Wait until a frame newer than last_seq arrives. Returns (frame, seq, t)."""
        deadline = time.perf_counter() + timeout
        while True:
            with self._lock:
                if self._frame_seq > last_seq and self._latest is not None:
                    frame, t = self._latest
                    return frame, self._frame_seq, t
            if time.perf_counter() > deadline:
                return None, last_seq, None
            time.sleep(0.001)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default='/dev/video0')
    p.add_argument('--width',  type=int, default=640)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--fps',    type=int, default=60)
    p.add_argument('--format', choices=['MJPG', 'YUYV', 'NV12'], default='MJPG')
    p.add_argument('--io-mode', type=int, default=2,
                   help='v4l2src io-mode: 2=mmap (most compatible), 4=dmabuf-import.')
    p.add_argument('--no-display', action='store_true')
    p.add_argument('--fullscreen', action='store_true')
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--window-frames', type=int, default=30)
    p.add_argument('--print-interval', type=float, default=1.0)
    opt = p.parse_args()

    headless = opt.no_display or not os.environ.get('DISPLAY')
    if headless and not opt.no_display:
        print("[warn] DISPLAY not set, falling back to headless.")

    pipeline_str = build_pipeline(opt.device, opt.format, opt.width, opt.height,
                                  opt.fps, opt.io_mode)
    print(f"Pipeline: {pipeline_str}")

    cap = GstCapture(pipeline_str)
    cap.start()
    print("Pipeline started, waiting for first frame...")

    win_name = "USB Camera (GStreamer)"
    if not headless:
        flag = cv2.WINDOW_NORMAL
        cv2.namedWindow(win_name, flag)
        if opt.fullscreen:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    timestamps = deque(maxlen=max(2, opt.window_frames))
    last_seq = 0
    measured = 0
    raw = 0
    t_start = None
    last_print = time.perf_counter()

    try:
        while True:
            frame, last_seq, t = cap.read_blocking(last_seq, timeout=2.0)
            if frame is None:
                print("No frame within timeout.")
                break

            raw += 1
            if raw <= opt.warmup:
                if raw == opt.warmup:
                    print(f"[warmup] discarded first {opt.warmup} frames")
                continue
            if t_start is None:
                t_start = t

            timestamps.append(t)
            measured += 1

            inst = ((len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                    if len(timestamps) >= 2 else 0.0)
            avg = measured / (t - t_start) if (t - t_start) > 0 else 0.0
            h, w = frame.shape[:2]

            if headless:
                if t - last_print >= opt.print_interval:
                    print(f"frame={measured:6d}  FPS={inst:5.1f}  avg={avg:5.1f}  size={w}x{h}")
                    last_print = t
            else:
                text = f"FPS: {inst:5.1f}  (avg {avg:5.1f})  {w}x{h}"
                cv2.putText(frame, text, (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, text, (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow(win_name, frame)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break

            if opt.max_frames and measured >= opt.max_frames:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.stop()
        if not headless:
            cv2.destroyAllWindows()

    if t_start is not None:
        elapsed = time.perf_counter() - t_start
        if elapsed > 0:
            print(f"\nMeasured {measured} frames in {elapsed:.2f}s, avg FPS: {measured/elapsed:.2f}")


if __name__ == '__main__':
    main()
