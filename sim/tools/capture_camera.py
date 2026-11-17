#!/usr/bin/env python3
"""Capture one or more frames from a Gazebo camera sensor topic."""
from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_camera_source import GazeboCameraSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="inspect_world")
    ap.add_argument("--model", required=True)
    ap.add_argument("--inner-model", default="")
    ap.add_argument("--link", default="link")
    ap.add_argument("--sensor", default="cam")
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=15.0)
    ap.add_argument("--settle", type=float, default=0.3)
    args = ap.parse_args()

    cam = GazeboCameraSource(args.world, args.model,
                              inner_model=args.inner_model,
                              link=args.link, sensor=args.sensor)
    deadline = time.monotonic() + args.timeout
    frame = None
    while time.monotonic() < deadline:
        frame = cam.frame()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        print("ERROR: no frame from", cam.topic, file=sys.stderr)
        return 2
    time.sleep(max(0.0, args.settle))
    latest = cam.frame()
    if latest is not None:
        frame = latest
    import cv2
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, frame)
    print("wrote", args.out, frame.shape, "topic", cam.topic)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
