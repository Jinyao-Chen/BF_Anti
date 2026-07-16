"""Probe USB camera capabilities by trying every (FOURCC, resolution, FPS)
combo the driver advertises (via `v4l2-ctl --list-formats-ext`) and measuring
the actually achieved FPS over a short capture window.

If v4l2-ctl is unavailable, falls back to a hardcoded common-modes preset.
"""
import os
import re
import sys
import time
import argparse
import subprocess

import cv2


COMMON_RESOLUTIONS = [
    (320, 240), (640, 360), (640, 480), (800, 600), (1024, 768),
    (1280, 720), (1280, 960), (1600, 1200),
    (1920, 1080), (2560, 1440), (3840, 2160),
]
COMMON_FOURCCS = ["MJPG", "YUYV"]
REQUESTED_FPS  = [120, 60, 30]


def is_usb_camera(device: str) -> bool:
    try:
        cap = cv2.VideoCapture(device)
        ok = cap.isOpened()
        cap.release()
        return ok
    except Exception:
        return False


def find_first_usb_camera():
    devs = sorted(os.path.join('/dev', d) for d in os.listdir('/dev') if d.startswith('video'))
    for d in devs:
        if is_usb_camera(d):
            return d
    return None


def fourcc_to_str(code: int) -> str:
    code = int(code)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def measure(device, fourcc, w, h, req_fps, warmup=5, frames=60, timeout_s=4.0):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FPS, req_fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    actual_fourcc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    # warmup — first frames are usually slow
    for _ in range(warmup):
        if not cap.read()[0]:
            cap.release()
            return None

    n = 0
    t0 = time.perf_counter()
    deadline = t0 + timeout_s
    while n < frames and time.perf_counter() < deadline:
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    elapsed = time.perf_counter() - t0
    cap.release()

    measured = n / elapsed if elapsed > 0 else 0.0
    return {
        "req":      (fourcc, w, h, req_fps),
        "actual":   (actual_fourcc, actual_w, actual_h, reported_fps),
        "measured": measured,
        "frames":   n,
        "elapsed":  elapsed,
    }


def run_v4l2_ctl(device):
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-formats-ext", "-d", device],
            text=True, stderr=subprocess.STDOUT, timeout=5,
        )
        print("=" * 72)
        print(f"v4l2-ctl --list-formats-ext -d {device}")
        print("=" * 72)
        print(out)
        return out
    except FileNotFoundError:
        print("[info] v4l2-ctl not installed. `sudo apt install v4l-utils` for "
              "authoritative capability listing.")
    except subprocess.CalledProcessError as e:
        print(f"[warn] v4l2-ctl failed: {e.output}")
    except subprocess.TimeoutExpired:
        print("[warn] v4l2-ctl timed out.")
    return None


_FMT_RE  = re.compile(r"^\s*\[\d+\]:\s*'([A-Z0-9]{4})'")
_SIZE_RE = re.compile(r"^\s*Size:\s*\S+\s+(\d+)x(\d+)")
_FPS_RE  = re.compile(r"\(([\d.]+)\s*fps\)")


def parse_v4l2_modes(text):
    """Parse `v4l2-ctl --list-formats-ext` output.

    Returns list of (fourcc, width, height, [fps_list]) tuples.
    """
    modes = []
    current_fcc = None
    current_size = None
    current_fps = []

    def flush():
        if current_fcc and current_size and current_fps:
            w, h = current_size
            modes.append((current_fcc, w, h, sorted(set(current_fps), reverse=True)))

    for line in text.splitlines():
        m = _FMT_RE.match(line)
        if m:
            flush()
            current_fcc = m.group(1)
            current_size = None
            current_fps = []
            continue
        m = _SIZE_RE.match(line)
        if m:
            flush()
            current_size = (int(m.group(1)), int(m.group(2)))
            current_fps = []
            continue
        m = _FPS_RE.search(line)
        if m and current_size:
            current_fps.append(float(m.group(1)))
    flush()
    return modes


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', default=None)
    p.add_argument('--frames', type=int, default=60,
                   help='Frames to capture per probe (after warmup).')
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--timeout', type=float, default=4.0,
                   help='Per-probe timeout in seconds.')
    p.add_argument('--no-v4l2', action='store_true',
                   help='Skip v4l2-ctl: use the hardcoded common-modes preset.')
    p.add_argument('--fourccs', nargs='+', default=None,
                   help='Override FOURCCs to probe (e.g. MJPG YUYV).')
    p.add_argument('--resolutions', nargs='+', default=None,
                   help='Override resolutions (e.g. 1280x720 1920x1080).')
    p.add_argument('--fps-list', nargs='+', type=int, default=None,
                   help='Override FPS values to request.')
    opt = p.parse_args()

    device = opt.device or find_first_usb_camera()
    if device is None:
        print("No USB camera found.")
        sys.exit(-1)
    print(f"Device: {device}\n")

    v4l2_text = None if opt.no_v4l2 else run_v4l2_ctl(device)
    v4l2_modes = parse_v4l2_modes(v4l2_text) if v4l2_text else []

    user_fourccs = set(opt.fourccs) if opt.fourccs else None
    user_resos = (set(tuple(map(int, r.lower().split('x'))) for r in opt.resolutions)
                  if opt.resolutions else None)
    user_fps = set(opt.fps_list) if opt.fps_list else None

    probes = []  # list of (fcc, w, h, req_fps)
    if v4l2_modes:
        print(f"[info] Parsed {len(v4l2_modes)} modes from v4l2-ctl.\n")
        for fcc, w, h, fps_list in v4l2_modes:
            if user_fourccs and fcc not in user_fourccs:
                continue
            if user_resos and (w, h) not in user_resos:
                continue
            fps_pool = user_fps if user_fps else set(int(round(f)) for f in fps_list)
            for f in sorted(fps_pool, reverse=True):
                probes.append((fcc, w, h, int(f)))
    else:
        print("[info] Using hardcoded common-modes preset.\n")
        fourccs = opt.fourccs or COMMON_FOURCCS
        resolutions = ([tuple(map(int, r.lower().split('x'))) for r in opt.resolutions]
                       if opt.resolutions else COMMON_RESOLUTIONS)
        fps_list = opt.fps_list or REQUESTED_FPS
        for fcc in fourccs:
            for (w, h) in resolutions:
                for req_fps in fps_list:
                    probes.append((fcc, w, h, req_fps))

    if not probes:
        print("Nothing to probe (filters too strict?).")
        sys.exit(0)

    print("=" * 72)
    print(f"Live probe ({len(probes)} combinations): request -> actual / measured FPS")
    print("=" * 72)
    header = f"{'req':>22}  {'actual':>22}  {'measured':>10}  {'frames':>7}"
    print(header)
    print("-" * len(header))

    for fcc, w, h, req_fps in probes:
        r = measure(device, fcc, w, h, req_fps,
                    warmup=opt.warmup, frames=opt.frames,
                    timeout_s=opt.timeout)
        if r is None:
            print(f"{fcc:>5} {w:>5}x{h:<5}@{req_fps:<3}  ---  open/read failed")
            continue
        a_fcc, a_w, a_h, a_fps = r["actual"]
        req_str = f"{fcc:>4} {w:>4}x{h:<4}@{req_fps:<3}"
        act_str = f"{a_fcc:>4} {a_w:>4}x{a_h:<4}@{a_fps:<5.1f}"
        print(f"{req_str:>22}  {act_str:>22}  {r['measured']:>9.2f}  "
              f"{r['frames']:>7d}")


if __name__ == '__main__':
    main()
