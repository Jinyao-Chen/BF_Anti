#!/usr/bin/env python3
"""Enumerate, capture and benchmark a UVC camera on the Pi 5.

The script is safe to run with no camera attached: it enumerates V4L2 nodes
from sysfs, prints the platform video nodes (pispbe / hevc decoder) so they
are not mistaken for a camera, and exits 2 when no USB UVC device is present.

It exists to answer, on the real board:

* which FOURCC / size / fps combinations does the camera advertise,
* which combination is actually negotiated by the V4L2 backend,
* what is the measured capture FPS and first-frame latency,
* can we save a few sample JPGs for the flight log.

No X11, no GStreamer, no display: capture is pinned to cv2.CAP_V4L2. MJPG is
preferred because the Pi 5 USB bandwidth cannot sustain raw YUYV 720p30.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    import cv2  # type: ignore
except Exception:  # noqa: BLE001 - list-only mode works without OpenCV
    cv2 = None

sysfs_video_root = "/sys/class/video4linux"
CAP_PROP_FRAME_WIDTH = 3
CAP_PROP_FRAME_HEIGHT = 4
CAP_PROP_FPS = 5
CAP_PROP_FOURCC = 6
CAP_PROP_BUFFERSIZE = 38
CAP_V4L2 = 200

COMMON_PRESETS = [
    ("MJPG", 1280, 720, 30.0),
    ("MJPG", 1920, 1080, 30.0),
    ("YUYV", 640, 480, 30.0),
]


@dataclass(frozen=True)
class Mode:
    fourcc: str
    width: int
    height: int
    fps: float


def wait_hint() -> str:
    return "connect a UVC camera to a USB 3 port and rerun"


def fourcc_code(text: str) -> int:
    raw = (str(text).upper() + "    ")[:4]
    return sum(ord(ch) << (8 * i) for i, ch in enumerate(raw))


def fourcc_str(value: float) -> str:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return "????"
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def sysfs_video_nodes() -> List[Dict[str, Any]]:
    """List video nodes with their kernel name and whether they are on USB."""
    nodes: List[Dict[str, Any]] = []
    if not os.path.isdir(sysfs_video_root):
        return nodes
    for name in sorted(os.listdir(sysfs_video_root)):
        if not name.startswith("video"):
            continue
        base = os.path.join(sysfs_video_root, name)
        try:
            with open(os.path.join(base, "name"), "r", encoding="utf-8",
                      errors="replace") as handle:
                kernel_name = handle.read().strip()
        except OSError:
            kernel_name = ""
        device_path = os.path.realpath(os.path.join(base, "device"))
        nodes.append({
            "device": "/dev/" + name,
            "name": kernel_name,
            "usb": "usb" in device_path,
            "sysfs": base,
        })
    return nodes


def pick_camera(nodes: List[Dict[str, Any]], requested: Optional[str]) -> Optional[str]:
    if requested:
        return requested
    for node in nodes:
        if not node["usb"]:
            continue
        name = str(node["name"]).lower()
        if "pispbe" in name or "hevc" in name:
            continue
        return str(node["device"])
    return None


def list_formats(device: str) -> str:
    try:
        proc = subprocess.run(
            ["v4l2-ctl", "--list-formats-ext", "-d", device],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if proc.returncode == 0:
            return proc.stdout
        return proc.stdout + proc.stderr
    except FileNotFoundError:
        return ""
    except subprocess.TimeoutExpired:
        return ""


def parse_formats(text: str) -> List[Dict[str, Any]]:
    formats: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    size: Optional[tuple] = None
    for line in text.splitlines():
        match = re.match(r"^\s*\[\d+\]:\s*'([^']+)'", line)
        if match:
            current = {"fourcc": match.group(1), "sizes": {}}
            formats.append(current)
            size = None
            continue
        match = re.match(r"^\s*Size:\s+\w+\s+(\d+)x(\d+)", line)
        if match and current is not None:
            size = (int(match.group(1)), int(match.group(2)))
            current["sizes"].setdefault(size, [])
            continue
        match = re.search(r"\(([\d.]+)\s*fps\)", line)
        if match and current is not None and size is not None:
            current["sizes"][size].append(float(match.group(1)))
    return formats


def build_modes(
    formats: List[Dict[str, Any]], requested: Mode, limit: int
) -> List[Mode]:
    modes = [requested]
    by_code = {item["fourcc"]: item for item in formats}
    ordered = [by_code[code] for code in ("MJPG", "YUYV") if code in by_code]
    ordered.extend(item for item in formats if item["fourcc"] not in ("MJPG", "YUYV"))
    for item in ordered:
        for (width, height), fps_list in sorted(
            item["sizes"].items(), key=lambda kv: -(kv[0][0] * kv[0][1])
        ):
            if width * height < 320 * 240 or width * height > 1920 * 1080:
                continue
            fps = max(fps_list) if fps_list else requested.fps
            mode = Mode(item["fourcc"], width, height, max(1.0, min(fps, 120.0)))
            if mode not in modes:
                modes.append(mode)
    for fourcc, width, height, fps in COMMON_PRESETS:
        mode = Mode(fourcc, width, height, fps)
        if mode not in modes:
            modes.append(mode)
    return modes[: max(1, limit)]


def capture_mode(
    device: str,
    mode: Mode,
    warmup: int,
    frames: int,
    timeout_s: float,
) -> Dict[str, Any]:
    assert cv2 is not None
    result: Dict[str, Any] = {
        "device": device,
        "requested": asdict(mode),
        "opened": False,
        "actual": {},
        "measured_fps": 0.0,
        "first_frame_ms": None,
        "interval_mean_ms": None,
        "interval_p95_ms": None,
        "frames": 0,
        "error": "",
    }
    try:
        cap = cv2.VideoCapture(device, CAP_V4L2)
    except Exception as exc:  # noqa: BLE001
        result["error"] = "open raised: %s" % (exc,)
        return result
    if not cap.isOpened():
        result["error"] = "VideoCapture could not open the device"
        cap.release()
        return result
    result["opened"] = True
    cap.set(CAP_PROP_FOURCC, fourcc_code(mode.fourcc))
    cap.set(CAP_PROP_FRAME_WIDTH, int(mode.width))
    cap.set(CAP_PROP_FRAME_HEIGHT, int(mode.height))
    cap.set(CAP_PROP_FPS, float(mode.fps))
    cap.set(CAP_PROP_BUFFERSIZE, 1)
    result["actual"] = {
        "fourcc": fourcc_str(cap.get(CAP_PROP_FOURCC)),
        "width": int(cap.get(CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(CAP_PROP_FRAME_HEIGHT)),
        "fps_reported": float(cap.get(CAP_PROP_FPS)),
    }
    frame = None
    started = time.perf_counter()
    try:
        for _ in range(max(0, warmup)):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
        result["first_frame_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        stamps: List[float] = []
        deadline = time.perf_counter() + max(0.2, timeout_s)
        while len(stamps) < max(1, frames) and time.perf_counter() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            stamps.append(time.perf_counter())
        result["frames"] = len(stamps)
        if len(stamps) > 1:
            elapsed = stamps[-1] - stamps[0]
            result["measured_fps"] = round((len(stamps) - 1) / elapsed, 2)
            intervals = [
                (later - earlier) * 1000.0
                for earlier, later in zip(stamps, stamps[1:])
            ]
            result["interval_mean_ms"] = round(sum(intervals) / len(intervals), 2)
            ordered = sorted(intervals)
            result["interval_p95_ms"] = round(
                ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))], 2
            )
    except Exception as exc:  # noqa: BLE001
        result["error"] = "capture raised: %s" % (exc,)
    finally:
        cap.release()
    return result


def save_sample(frame: Any, save_dir: str, mode: Mode, index: int) -> str:
    assert cv2 is not None
    os.makedirs(save_dir, exist_ok=True)
    name = "cam%02d_%s_%dx%d_%03.0ffps.jpg" % (
        index, mode.fourcc, mode.width, mode.height, mode.fps,
    )
    path = os.path.join(save_dir, name)
    if frame is not None:
        cv2.imwrite(path, frame)
        return path
    return ""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default=None, help="V4L2 node (auto: first USB UVC)")
    parser.add_argument("--fourcc", default="MJPG", help="requested FOURCC (default MJPG)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--modes", type=int, default=6, help="max modes to benchmark")
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=4.0, help="capture window per mode (s)")
    parser.add_argument("--save-dir", default="", help="save one JPG per mode here")
    parser.add_argument("--json-out", default="", help="write the full report as JSON")
    parser.add_argument("--list-only", action="store_true", help="enumerate, do not capture")
    args = parser.parse_args(argv)

    nodes = sysfs_video_nodes()
    print("== V4L2 nodes ==")
    for node in nodes:
        kind = "USB" if node["usb"] else "platform"
        print("  %-14s %-28s [%s]" % (node["device"], node["name"], kind))
    if not nodes:
        print("  (none)")

    camera = pick_camera(nodes, args.device)
    if camera is None:
        print("")
        print("[FAIL] no USB UVC camera found. %s" % wait_hint())
        print("       platform nodes above belong to the ISP / HEVC decoder, not a camera.")
        return 2
    print("")
    print("selected camera: %s" % camera)

    formats_text = list_formats(camera)
    formats = parse_formats(formats_text)
    if formats_text:
        print("")
        print("== v4l2-ctl --list-formats-ext ==")
        for fmt in formats:
            print("  %s" % fmt["fourcc"])
            for (width, height), fps_list in fmt["sizes"].items():
                print("    %dx%d @ %s" % (
                    width, height,
                    ", ".join("%.2f" % fps for fps in fps_list) if fps_list else "?",
                ))
    else:
        print("[WARN] v4l2-ctl missing or returned nothing; falling back to preset modes")

    if args.list_only:
        return 0
    if cv2 is None:
        print("[FAIL] OpenCV (cv2) is not importable; cannot capture")
        return 5

    requested = Mode(args.fourcc.upper(), args.width, args.height, args.fps)
    modes = build_modes(formats, requested, args.modes)
    print("")
    print("== capture benchmark (warmup=%d frames, up to %d frames per mode) ==" % (
        args.warmup, args.frames,
    ))
    header = "%-6s %-11s %-11s %8s %9s %9s %8s %7s" % (
        "fourcc", "requested", "negotiated", "fps", "first_ms", "mean_ms", "p95_ms", "frames",
    )
    print(header)
    print("-" * len(header))

    reports: List[Dict[str, Any]] = []
    saved: List[str] = []
    best_frames = 0
    for index, mode in enumerate(modes):
        report = capture_mode(
            camera, mode, args.warmup, args.frames, args.timeout
        )
        reports.append(report)
        actual = report.get("actual") or {}
        negotiated = "%dx%d" % (
            int(actual.get("width", 0)), int(actual.get("height", 0))
        )
        print("%-6s %-11s %-11s %8.2f %9s %9s %8s %7d" % (
            mode.fourcc,
            "%dx%d@%.0f" % (mode.width, mode.height, mode.fps),
            negotiated,
            float(report.get("measured_fps") or 0.0),
            "%.1f" % report["first_frame_ms"] if report.get("first_frame_ms") is not None else "-",
            "%.1f" % report["interval_mean_ms"] if report.get("interval_mean_ms") is not None else "-",
            "%.1f" % report["interval_p95_ms"] if report.get("interval_p95_ms") is not None else "-",
            int(report.get("frames") or 0),
        ))
        best_frames = max(best_frames, int(report.get("frames") or 0))
        if args.save_dir and report.get("frames"):
            # Re-open briefly for a sample is expensive; capture_mode does not
            # return the frame, so save from a fresh single read instead.
            try:
                cap = cv2.VideoCapture(camera, CAP_V4L2)
                cap.set(CAP_PROP_FOURCC, fourcc_code(mode.fourcc))
                cap.set(CAP_PROP_FRAME_WIDTH, int(mode.width))
                cap.set(CAP_PROP_FRAME_HEIGHT, int(mode.height))
                cap.set(CAP_PROP_FPS, float(mode.fps))
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    path = save_sample(frame, args.save_dir, mode, index)
                    if path:
                        saved.append(path)
                print("       saved %s" % saved[-1])
            except Exception as exc:  # noqa: BLE001
                print("       save failed: %s" % (exc,))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(
                {"camera": camera, "nodes": nodes, "modes": reports, "saved": saved},
                handle, indent=2, sort_keys=True,
            )
        print("json written: %s" % args.json_out)

    if best_frames <= 0:
        print("")
        print("[FAIL] device opened but no frames were captured")
        return 3
    print("")
    print("[ OK ] captured frames from at least one mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
