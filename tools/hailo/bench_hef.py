#!/usr/bin/env python3
"""Benchmark a HEF on the Hailo-8L: p50/p95/FPS, chip temperature, outputs.

Run this on the Pi5. It is a performance tool, not an accuracy test: the
default frames are synthetic patterns, and the COCO precompiled HEF fallback
is only for pipeline bring-up. Record real Pi camera frames with --images
before quoting end-to-end numbers.

Example:
  tools/hailo/bench_hef.py --hef /usr/share/hailo-models/yolov8s_h8l.hef \
      --frames 300 --json build/bench/yolov8s_h8l.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

def find_repo_root() -> Path:
    """Locate the checkout even when this file is copied to /tmp for a probe."""
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents, Path.cwd()]
    for candidate in candidates:
        if (candidate / "onboard").is_dir() and (candidate / "tools" / "hailo").is_dir():
            return candidate
    return here.parent


REPO_ROOT = find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NOTES = [
    "latency is HailoRT run() plus host postprocess, measured on this host",
    "synthetic/default frames are not an accuracy evaluation",
    "the COCO yolov8s_h8l.hef fallback must never be quoted as drone accuracy",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_hef(raw: str) -> Path:
    if raw:
        return Path(raw).expanduser()
    for candidate in (
        REPO_ROOT / "model" / "yolov8n_drone_h8l.hef",
        Path("/usr/share/hailo-models/yolov8s_h8l.hef"),
    ):
        if candidate.is_file():
            return candidate
    return Path("/usr/share/hailo-models/yolov8s_h8l.hef")


def firmware_info() -> dict[str, Any]:
    info = {"firmware": None, "board": None, "architecture": None}
    try:
        result = subprocess.run(
            ["hailortcli", "fw-control", "identify"],
            capture_output=True, text=True, timeout=15.0, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return info
    if result.returncode != 0:
        return info
    for line in result.stdout.splitlines():
        text = line.strip()
        for prefix, key in (
            ("Firmware Version:", "firmware"),
            ("Board Name:", "board"),
            ("Device Architecture:", "architecture"),
        ):
            if text.startswith(prefix):
                info[key] = text[len(prefix):].strip()
    return info


def read_chip_temperature(vdevice: Any) -> Optional[dict[str, float]]:
    try:
        device = vdevice.get_physical_devices()[0]
        info = device.control.get_chip_temperature()
    except Exception:
        return None
    out: dict[str, float] = {}
    for attr in ("ts0_temperature", "ts1_temperature"):
        value = getattr(info, attr, None)
        if value is not None:
            out[attr] = float(value)
    return out or None


def read_soc_temperature() -> Optional[float]:
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return float(raw) / 1000.0
    except (OSError, ValueError):
        return None


def synthetic_frames(count: int, width: int, height: int) -> list[Any]:
    import numpy as np

    yy, xx = np.mgrid[0:height, 0:width]
    frames = []
    for index in range(max(1, int(count))):
        # numpy 2.x rejects uint8 + python int overflow, so compute in int16.
        base = ((xx * 3 + yy * 2 + index * 17) % 256).astype(np.int16)
        frame = np.stack(
            [((base + 11 * channel) % 256).astype(np.uint8) for channel in range(3)],
            axis=2,
        )
        frames.append(frame)
    return frames


def load_images(directory: Path) -> list[Any]:
    import cv2

    frames = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(image)
    if not frames:
        raise FileNotFoundError("no readable images in %s" % directory)
    return frames


def percentile_stats(values: list[float]) -> dict[str, float]:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a Hailo-8L HEF (latency/FPS/temperature)."
    )
    parser.add_argument("--hef", default="", help="HEF path (default: best local fallback)")
    parser.add_argument("--frames", type=int, default=200, help="timed frames")
    parser.add_argument("--warmup", type=int, default=20, help="warmup frames")
    parser.add_argument("--timeout-ms", type=int, default=10000)
    parser.add_argument("--images", default="", help="directory of real frames to preprocess")
    parser.add_argument("--classes", type=int, default=1,
                        help="class count for a raw (non-NMS) output")
    parser.add_argument("--no-bgr-to-rgb", action="store_true")
    parser.add_argument("--json", default="", help="write JSON here ('-' for stdout)")
    parser.add_argument("--label", default="", help="free-form label in the report")
    args = parser.parse_args(argv)

    hef_path = find_hef(args.hef)
    if not hef_path.is_file():
        print("error: HEF not found: %s" % hef_path, file=sys.stderr)
        return 2
    if args.frames <= 0:
        print("error: --frames must be positive", file=sys.stderr)
        return 2

    try:
        import numpy as np
        from hailo_platform import VDevice
    except ImportError as exc:
        print("BLOCKED: HailoRT Python API not importable: %s" % exc, file=sys.stderr)
        return 3

    from onboard.detector.postprocess import (
        letterbox_image,
        parse_hailo_nms,
        parse_yolo_v8_raw,
    )

    try:
        import cv2  # noqa: F401
    except ImportError:
        if args.images:
            print("error: --images needs OpenCV", file=sys.stderr)
            return 2

    source = "synthetic"
    if args.images:
        raw_frames = load_images(Path(args.images).expanduser())
        source = "images:%s" % Path(args.images).expanduser()
    else:
        raw_frames = None  # generated after the input shape is known

    started_wall = time.time()
    vdevice = VDevice()
    try:
        model = vdevice.create_infer_model(str(hef_path))
        model.set_batch_size(1)
        input_stream = model.input()
        input_shape = tuple(int(v) for v in input_stream.shape)
        if len(input_shape) != 3 or input_shape[2] != 3:
            print("error: unsupported input shape %r" % (input_shape,), file=sys.stderr)
            return 2
        height, width = int(input_shape[0]), int(input_shape[1])
        try:
            from hailo_platform import FormatType

            for stream in model.outputs:
                if not stream.is_nms:
                    stream.set_format_type(FormatType.FLOAT32)
        except Exception as exc:
            print("WARNING: set_format_type(FLOAT32) failed: %s" % exc, file=sys.stderr)

        configured = model.configure()
        try:
            configured.activate()
        except Exception as exc:
            print("WARNING: activate() skipped: %s" % exc, file=sys.stderr)
        bindings = configured.create_bindings()
        input_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        bindings.input().set_buffer(input_buffer)

        outputs = []
        for stream in model.outputs:
            dtype = np.float32
            if not stream.is_nms:
                try:
                    from hailo_platform import FormatType

                    if stream.format.type != FormatType.FLOAT32:
                        raise RuntimeError(
                            "raw output %s is %s, not FLOAT32; compile with a "
                            "float output format" % (stream.name, stream.format.type)
                        )
                except ImportError:
                    pass
            buffer = np.empty(stream.shape, dtype=dtype)
            bindings.output(stream.name).set_buffer(buffer)
            outputs.append(
                {
                    "name": stream.name,
                    "shape": [int(v) for v in stream.shape],
                    "is_nms": bool(stream.is_nms),
                    "dtype": np.dtype(dtype).name,
                }
            )

        if raw_frames is None:
            raw_frames = synthetic_frames(8, width, height)
        prepared: list[tuple[Any, Any]] = []
        preprocess_ms: list[float] = []
        for image in raw_frames:
            tick = time.perf_counter()
            rgb = image
            if not args.no_bgr_to_rgb and getattr(image, "ndim", 0) == 3:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            letterboxed, params = letterbox_image(rgb, width, height)
            preprocess_ms.append((time.perf_counter() - tick) * 1000.0)
            prepared.append((letterboxed, params))
        if not prepared:
            print("error: no frames to benchmark", file=sys.stderr)
            return 2

        def run_once(index: int) -> tuple[float, float, int]:
            letterboxed, params = prepared[index % len(prepared)]
            np.copyto(input_buffer, letterboxed)
            tick = time.perf_counter()
            configured.run([bindings], args.timeout_ms)
            infer_ms = (time.perf_counter() - tick) * 1000.0
            tick = time.perf_counter()
            detections = 0
            for info in outputs:
                if info["is_nms"]:
                    raw = bindings.output(info["name"]).get_buffer(tf_format=True)
                    detections += len(
                        parse_hailo_nms(
                            raw, params,
                            conf_threshold=0.25, iou_threshold=0.45,
                            max_detections=100,
                        )
                    )
                else:
                    raw = bindings.output(info["name"]).get_buffer()
                    detections += len(
                        parse_yolo_v8_raw(
                            raw, params,
                            conf_threshold=0.25, iou_threshold=0.45,
                            max_detections=100, num_classes=args.classes,
                        )
                    )
            post_ms = (time.perf_counter() - tick) * 1000.0
            return infer_ms, post_ms, detections

        temperature_before = read_chip_temperature(vdevice)
        soc_before = read_soc_temperature()
        for index in range(max(0, args.warmup)):
            run_once(index)

        infer_samples: list[float] = []
        post_samples: list[float] = []
        detection_counts: list[int] = []
        loop_started = time.perf_counter()
        for index in range(args.frames):
            infer_ms, post_ms, detections = run_once(index)
            infer_samples.append(infer_ms)
            post_samples.append(post_ms)
            detection_counts.append(detections)
        loop_elapsed = time.perf_counter() - loop_started
        temperature_after = read_chip_temperature(vdevice)
        soc_after = read_soc_temperature()

        fw = firmware_info()
        report = {
            "tool": "tools/hailo/bench_hef.py",
            "label": args.label,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": {
                "hostname": platform.node(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version.split()[0],
            },
            "hailo_platform": getattr(__import__("hailo_platform"), "__version__", None),
            "firmware": fw,
            "hef": {
                "path": str(hef_path),
                "size": hef_path.stat().st_size,
                "sha256": sha256_file(hef_path),
            },
            "input": {"shape": list(input_shape), "dtype": "uint8"},
            "outputs": outputs,
            "source": source,
            "frames": args.frames,
            "warmup": args.warmup,
            "latency_ms": percentile_stats(infer_samples),
            "postprocess_ms": percentile_stats(post_samples),
            "preprocess_ms_mean": float(np.mean(preprocess_ms)),
            "throughput_fps_infer": 1000.0 / max(1e-9, float(np.mean(infer_samples))),
            "throughput_fps_loop": args.frames / max(1e-9, loop_elapsed),
            "detections_per_frame_mean": float(np.mean(detection_counts)),
            "temperature_c": {"before": temperature_before, "after": temperature_after},
            "soc_thermal_zone0_c": {"before": soc_before, "after": soc_after},
            "wall_s": time.time() - started_wall,
            "notes": NOTES,
        }
        human = (
            "hef={hef} frames={frames} source={source}\n"
            "infer_ms: mean={mean:.2f} p50={p50:.2f} p95={p95:.2f} p99={p99:.2f}\n"
            "post_ms : mean={post:.3f}  fps(infer)={fps:.1f}  fps(loop)={loop:.1f}\n"
            "chip temp: {tb} -> {ta} C (Pi SoC {sb} -> {sa} C)"
        ).format(
            hef=hef_path.name,
            frames=args.frames,
            source=source,
            mean=report["latency_ms"]["mean"],
            p50=report["latency_ms"]["p50"],
            p95=report["latency_ms"]["p95"],
            p99=report["latency_ms"]["p99"],
            post=report["postprocess_ms"]["mean"],
            fps=report["throughput_fps_infer"],
            loop=report["throughput_fps_loop"],
            tb=temperature_before,
            ta=temperature_after,
            sb=round(soc_before, 2) if soc_before is not None else None,
            sa=round(soc_after, 2) if soc_after is not None else None,
        )
        if args.json:
            text = json.dumps(report, indent=2)
            if args.json == "-":
                print(text)
            else:
                destination = Path(args.json).expanduser()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(text, encoding="utf-8")
                print(human)
                print("json: %s" % destination)
        else:
            print(human)
        return 0
    except Exception as exc:
        print("error: HEF benchmark failed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 4
    finally:
        # HailoRT 4.23 faults in ConfiguredInferModel.deactivate() once buffers
        # are bound, and child objects must be gone before the VDevice is
        # released. Drop bindings/configured/model first, release last; this
        # mirrors onboard/detector/hailo.py:_release.
        try:
            bindings = None
            configured = None
            model = None
        except Exception:
            pass
        try:
            vdevice.release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
