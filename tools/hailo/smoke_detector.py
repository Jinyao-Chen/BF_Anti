#!/usr/bin/env python3
"""Hardware smoke test for the HailoDetector end-to-end path.

Runs on the Pi5 with the HailoRT Python API and a HEF. Frames are synthetic
gradient patterns because no camera is attached, so detections (if any) are
not an accuracy result. The point is to prove: lazy HailoRT import, device
configure/activate, input buffers, NMS/raw output parsing, latest()/healthy()
and clean stop.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents, Path.cwd()]:
        if (candidate / "onboard").is_dir() and (candidate / "tools" / "hailo").is_dir():
            return candidate
    return here.parent


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from onboard.config import DetectorConfig  # noqa: E402
from onboard.detector import build_detector  # noqa: E402


def synthetic_frame(index: int, width: int = 640, height: int = 640):
    import numpy as np

    yy, xx = np.mgrid[0:height, 0:width]
    base = ((xx * 2 + yy + index * 13) % 256).astype(np.int16)
    return np.stack(
        [((base + 17 * channel) % 256).astype(np.uint8) for channel in range(3)],
        axis=2,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke the onboard HailoDetector with synthetic frames."
    )
    parser.add_argument(
        "--hef", default="/usr/share/hailo-models/yolov8s_h8l.hef",
        help="HEF path (the stock COCO model is acceptable for plumbing only)",
    )
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    cfg = DetectorConfig(
        backend="hailo",
        hef=args.hef,
        input_width=640,
        input_height=640,
        conf_threshold=0.25,
        class_names=("drone",),
    )
    detector = build_detector(cfg)
    start = time.monotonic()
    detector.start()
    detector.submit_frame(synthetic_frame(0))
    ready = detector.wait_ready(args.timeout)
    if not ready:
        print("SMOKE FAIL: detector not ready in %.1f s" % args.timeout)
        print("last_error:", detector.last_error)
        print("stats:", detector.stats())
        detector.stop()
        return 3

    processed = 1
    for index in range(1, max(1, args.frames)):
        before = detector.stats().frames
        detector.submit_frame(synthetic_frame(index))
        deadline = time.monotonic() + args.timeout
        while detector.stats().frames == before and time.monotonic() < deadline:
            time.sleep(0.005)
        if detector.stats().frames == before:
            print("SMOKE FAIL: frame %d was not processed in %.1f s" % (index, args.timeout))
            detector.stop()
            return 4
        processed += 1

    stats = detector.stats()
    detections = detector.latest_detections(1.0)
    healthy = detector.healthy()
    elapsed = time.monotonic() - start
    print("model:", detector.model_info)
    print("stats:", stats)
    print("healthy:", healthy)
    print("detections this frame:", len(detections))
    for detection in detections[:5]:
        print("  ", detection)
    detector.stop()
    processed = int(stats.frames)
    print("processed %d frames in %.2f s (%.1f fps)" % (processed, elapsed, processed / max(1e-9, elapsed)))
    ok = processed > 0 and stats.errors == 0 and detector.model_info.get("network")
    print("SMOKE PASS" if ok else "SMOKE FAIL")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
