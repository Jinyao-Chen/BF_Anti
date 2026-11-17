#!/usr/bin/env python3
"""Measure the real camera -> Hailo -> control age on the Pi.

This is the bench test for the async-detection bug: it reports capture age
(not inference completion), inference latency, detection rate and the paired
frame age.  No RC frames are sent and no video is written to fb0.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onboard.camera import CameraSource
from onboard.config import load_config
from onboard.detector import build_detector


def pct(values, q):
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return values[idx]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="camera/Hailo async latency check")
    ap.add_argument("--config", default="config/onboard.json")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--print-every", type=float, default=0.5)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    cam = CameraSource(cfg.camera)
    det = build_detector(cfg.detector, source=cam)
    cam.start()
    det.start()
    print("camera: %s  detector: %s  (no RC, no fb0)" % (cfg.camera.device, cfg.detector.hef))
    if not det.wait_ready(15.0):
        print("detector not ready after 15 s:", det.last_error)
        return 2
    print("detector ready; measuring %.1f s ..." % args.seconds)

    ages, infer, rates = [], [], []
    hits = 0
    frames = 0
    next_print = time.monotonic() + args.print_every
    end = time.monotonic() + args.seconds
    last_stats = det.stats()
    while time.monotonic() < end:
        now = time.monotonic()
        d = det.latest(1.0, now)
        age = det.detection_age(now)
        st = det.stats()
        if d is not None:
            hits += 1
            ages.append(age)
        if st.frames != last_stats.frames:
            frames += 1
            infer.append(st.infer_ms)
        if st.fps > 0:
            rates.append(st.fps)
        if now >= next_print:
            next_print += args.print_every
            bbox = "none" if d is None else "%.0fx%.0f" % (d.bbox[2]-d.bbox[0], d.bbox[3]-d.bbox[1])
            print("cam %.1ffps det %.1ffps infer %.1fms age %.0fms bbox %s conf %s"
                  % (cam.measured_fps, st.fps, st.infer_ms,
                     (age * 1000.0) if d is not None else -1.0,
                     bbox, ("%.2f" % d.conf) if d is not None else "-"))
        last_stats = st
        time.sleep(0.01)

    print("\nsummary over %.1f s:" % args.seconds)
    print("  detector frames: %d  detections seen: %d" % (last_stats.frames, hits))
    print("  infer ms p50/p95: %.2f / %.2f" % (pct(infer, 0.5), pct(infer, 0.95)))
    print("  capture age ms p50/p95: %.1f / %.1f" % (pct(ages, 0.5) * 1000, pct(ages, 0.95) * 1000))
    print("  det fps p50: %.1f" % pct(rates, 0.5))
    cam.stop()
    det.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
