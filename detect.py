"""Real-time YOLO detection on RDK X5 USB camera + object tracking.

Combines:
  - ThreadedV4L2Capture (background reader thread, no V4L2 backpressure)
  - UltralyticsYOLODetect (BPU inference)
  - IoU-based tracker (persistent IDs, motion prediction, box smoothing)

Tracking defaults tuned for high-speed small drone targets:
  --track              Enable object tracking (default: enabled)
  --no-track           Disable tracking
  --track-dist-thresh  Center-distance score threshold (default: 0.15, score=1-dist/100px)
  --track-alpha        EMA smoothing 0..1 (default: 0.90, high = minimal lag)
  --track-max-lost     Keep predicting N frames after lost (default: 45)
"""
import os
import sys
import time
import argparse
import threading
from collections import deque

import cv2
import numpy as np

# rdk_model_zoo paths
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_ZOO    = os.path.join(SCRIPT_DIR, "rdk_model_zoo")
RUNTIME_DIR  = os.path.join(MODEL_ZOO, "samples/vision/ultralytics_yolo/runtime/python")
sys.path.append(MODEL_ZOO)
sys.path.append(RUNTIME_DIR)

import utils.py_utils.file_io as file_io     # noqa: E402
import utils.py_utils.visualize as visualize  # noqa: E402
from ultralytics_yolo_det import (             # noqa: E402
    UltralyticsYOLODetect, UltralyticsYOLODetectConfig,
)

# optional: IoU tracker
try:
    from tracker import Tracker, draw_tracks
    _HAVE_TRACKER = True
except ImportError:
    _HAVE_TRACKER = False


# --------------------------------------------------------------------------- #
# USB camera (V4L2 + reader thread)
# --------------------------------------------------------------------------- #

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


class VideoFileCapture:
    """OpenCV video file reader with the same interface as ThreadedV4L2Capture."""

    def __init__(self, path):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"failed to open {path}")
        self.read_count = 0
        self.actual_size = (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.actual_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30

    def read_blocking(self, _last_seq=0, timeout=2.0):
        ret, frame = self._cap.read()
        if not ret:
            return (None, 0, 0)
        self.read_count += 1
        return (frame, self.read_count, time.perf_counter())

    def start(self):
        pass  # video file doesn't need a background thread

    def stop(self):
        self._cap.release()


class ThreadedV4L2Capture:
    """V4L2 capture + background reader. Main thread always reads the latest frame."""

    def __init__(self, device, fourcc='MJPG', width=640, height=480, fps=60,
                 auto_exposure_mode=3):
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"failed to open {device}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure_mode)

        self._cap = cap
        self._lock = threading.Lock()
        self._latest = None        # (frame, t)
        self._seq = 0
        self._read_count = 0
        self._stop = threading.Event()
        self._alive = True
        self._thread = threading.Thread(target=self._reader, daemon=True)

    @property
    def actual_size(self):
        return (int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    @property
    def actual_fps(self):
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def read_count(self):
        return self._read_count

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self._alive = False
                break
            t = time.perf_counter()
            with self._lock:
                self._latest = (frame, t)
                self._seq += 1
                self._read_count += 1

    def read_blocking(self, last_seq, timeout=2.0):
        deadline = time.perf_counter() + timeout
        while True:
            with self._lock:
                if self._seq > last_seq and self._latest is not None:
                    frame, t = self._latest
                    return frame, self._seq, t
                if not self._alive:
                    return None, last_seq, None
            if time.perf_counter() > deadline:
                return None, last_seq, None
            time.sleep(0.0005)


# --------------------------------------------------------------------------- #
# Status bar drawing helpers
# --------------------------------------------------------------------------- #

def draw_status_bar(img, capture_fps, infer_fps, display_fps,
                    n_tracks, n_raw_dets, ui_scale=1.0, tracking_state=None):
    """Overlay FPS and track/detection counts on the image."""
    fs = ui_scale              # font scale multiplier
    lh = max(1, int(2 * fs))   # line height / thickness

    # left side: FPS
    left = (f"capture {capture_fps:5.1f} | infer {infer_fps:5.1f} | "
            f"display {display_fps:5.1f}")
    y = max(30, int(30 * fs))
    cv2.putText(img, left, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * fs, (0, 0, 0), lh + 2, cv2.LINE_AA)
    cv2.putText(img, left, (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * fs, (0, 255, 0), lh, cv2.LINE_AA)

    # right side: track / det counts
    right = f"tracked={n_tracks}  raw={n_raw_dets}"
    (rw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs, lh)
    x_right = img.shape[1] - rw - 12
    cv2.putText(img, right, (x_right, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs, (0, 0, 0), lh + 2, cv2.LINE_AA)
    cv2.putText(img, right, (x_right, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6 * fs, (200, 255, 200), lh, cv2.LINE_AA)

    # tracking state (on screen centre-top)
    if tracking_state and not tracking_state.endswith("MANUAL"):
        label = f"TRACK: {tracking_state}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs, max(2, int(2 * fs)))
        cx = (img.shape[1] - tw) // 2
        color = (0, 255, 0) if tracking_state == "TRACKING" else (0, 200, 255)
        cv2.putText(img, label, (cx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs, (0, 0, 0), max(3, int(3 * fs)), cv2.LINE_AA)
        cv2.putText(img, label, (cx, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2 * fs, color, max(1, int(2 * fs)), cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(
        description="Real-time YOLO detection on USB camera (with optional tracking).")

    # model
    p.add_argument("--model-path", required=True, help="Path to the X5 .bin model.")
    p.add_argument("--label-file", default=None,
                   help="Optional class names file (one name per line).")
    p.add_argument("--classes-num", type=int, default=0,
                   help="Number of classes. 0 = auto-detect from model output shape.")
    p.add_argument("--score-thres",  type=float, default=0.4)
    p.add_argument("--nms-thres",    type=float, default=0.70)
    p.add_argument("--reg",          type=int,   default=16)
    p.add_argument("--resize-type",  type=int,   default=1, choices=[0, 1])
    p.add_argument("--priority",  type=int, default=0)
    p.add_argument("--bpu-cores", type=int, nargs="+", default=[0, 1])

    # camera / video file
    p.add_argument("--device",  default=None, help="USB device path; auto-detect if empty.")
    p.add_argument("--video",   default=None, help="Video file path (mp4/avi); overrides camera.")
    p.add_argument("--cam-width",  type=int, default=1280)
    p.add_argument("--cam-height", type=int, default=720)
    p.add_argument("--cam-fps",    type=int, default=60)
    p.add_argument("--cam-fourcc", default="MJPG")

    # display
    p.add_argument("--no-display",  action="store_true",
                   help="Headless mode: print FPS, do not open a window.")
    p.add_argument("--fullscreen",  action="store_true")
    p.add_argument("--save",        default=None, help="Save output video to path (e.g. output.mp4).")
    p.add_argument("--max-frames",  type=int, default=0)
    p.add_argument("--print-interval", type=float, default=1.0)
    p.add_argument("--window-frames",  type=int, default=30)
    p.add_argument("--debug-dets", action="store_true",
                   help="Print first 5 detection boxes per frame for debugging.")

    # tracking
    p.add_argument("--track",     dest="track", action="store_true", default=True,
                   help="Enable object tracking (default: enabled).")
    p.add_argument("--no-track",  dest="track", action="store_false",
                   help="Disable object tracking.")
    p.add_argument("--track-dist-thresh", type=float, default=0.15,
                   help="Distance-score threshold (0..1). Lower = further matching range. Score=1-dist/100px.")
    p.add_argument("--track-alpha",   type=float, default=0.90,
                   help="EMA smoothing (0..1). Higher = faster response for agile drones.")
    p.add_argument("--track-max-lost", type=int, default=45,
                   help="Frames to keep predicting after detection loss. Higher = handles occlusion.")

    # goggles mode (for analog VTX → FPV goggles)
    p.add_argument("--goggles", action="store_true",
                   help="Goggles mode: set HDMI to PAL, fullscreen, large UI for CVBS.")
    p.add_argument("--goggles-res", default="720x576",
                   help="HDMI resolution in goggles mode (default: 720x576 PAL).")
    # SPI PAL video output (GPIO direct analog video)
    p.add_argument("--spi-video", action="store_true",
                   help="Enable SPI-based PAL composite video output via GPIO pin 19 (SPI1_MOSI).")
    # auto-tracking (PBVS + MSP Override)
    p.add_argument("--tracking", action="store_true",
                   help="Enable visual servo auto-tracking (requires MSP Override on FC).")
    p.add_argument("--crsf-port", default=None,
                   help="ELRS CRSF UART port for stick detection (e.g. /dev/ttyS3).")
    p.add_argument("--crsf-baud", type=int, default=420000,
                   help="CRSF UART baud rate (default: 420000).")
    p.add_argument("--tracking-switch", type=int, default=None,
                   help="CRSF channel index (0-15) for radio switch to engage tracking. "
                        ">1700 = engage, <1300 = disengage. "
                        "Example: --tracking-switch 7 (= AUX4/CH8).")
    p.add_argument("--base-throttle", type=int, default=1100,
                   help="Base throttle for tracking (default: 1100 bench, ~1550 hover). "
                        "Lower = safer for bench testing without props.")
    # tracking PID / tuning (all optional — defaults are for 5-inch drone)
    p.add_argument("--pid-yaw", type=float, nargs=4, default=[2.0, 0.1, 0.5, 300],
                   metavar=("KP", "KI", "KD", "LIMIT"),
                   help="Yaw PID: kp ki kd limit (default: 2.0 0.1 0.5 300)")
    p.add_argument("--pid-pitch", type=float, nargs=4, default=[2.5, 0.1, 0.5, 300],
                   metavar=("KP", "KI", "KD", "LIMIT"),
                   help="Pitch PID: kp ki kd limit (default: 2.5 0.1 0.5 300)")
    p.add_argument("--pid-thr", type=float, nargs=4, default=[1.5, 0.05, 0.3, 200],
                   metavar=("KP", "KI", "KD", "LIMIT"),
                   help="Throttle PID: kp ki kd limit (default: 1.5 0.05 0.3 200)")
    p.add_argument("--tracking-rate-limit", type=int, default=50,
                   help="Max RC change per frame (anti-jerk, default: 50)")
    p.add_argument("--tracking-target-area", type=float, default=0.05,
                   help="Target box area ratio (default: 0.05 = 5%% of frame)")
    p.add_argument("--tracking-lost-timeout", type=float, default=2.0,
                   help="Seconds before entering search mode (default: 2.0)")
    p.add_argument("--tracking-max-lost", type=float, default=5.0,
                   help="Seconds in search before disengaging (default: 5.0)")
    opt = p.parse_args()

    # --track / --no-track
    enable_track = opt.track and _HAVE_TRACKER
    if opt.track and not _HAVE_TRACKER:
        print("[warn] tracker.py not found; running without tracking.")
        print("       (copy tracker.py to the same directory as this script)")

    # --goggles mode: set HDMI to PAL, force fullscreen, larger UI
    if opt.goggles:
        opt.fullscreen = True
        goggles_res = opt.goggles_res
        try:
            import subprocess
            subprocess.run(
                ["xrandr", "--output", "HDMI-1", "--mode", goggles_res,
                 "--rate", "50"],
                capture_output=True, text=True, timeout=5)
            print(f"HDMI set to {goggles_res}@50Hz for goggles/CVBS output.")
        except Exception as e:
            print(f"[warn] Could not set HDMI resolution: {e}")

    headless = opt.no_display or not os.environ.get("DISPLAY")
    if headless and not opt.no_display:
        print("[warn] DISPLAY not set, running headless.")

    # ----- Camera / Video --------------------------------------------------- #
    if opt.video:
        print(f"Opening video: {opt.video}")
        cap = VideoFileCapture(opt.video)
        w, h = cap.actual_size
        print(f"Video: {w}x{h} @ {cap.actual_fps:.1f} FPS")
    else:
        device = opt.device or find_first_usb_camera()
        if device is None:
            print("No USB camera found.")
            sys.exit(-1)
        print(f"Opening camera: {device}")
        cap = ThreadedV4L2Capture(device, fourcc=opt.cam_fourcc,
                                  width=opt.cam_width, height=opt.cam_height,
                                  fps=opt.cam_fps)
        w, h = cap.actual_size
        print(f"Camera: {w}x{h} @ {cap.actual_fps:.1f} FPS (driver-reported)")

    # ----- Model ------------------------------------------------------------ #
    print(f"Loading model: {opt.model_path}")
    cfg = UltralyticsYOLODetectConfig(
        model_path   = opt.model_path,
        classes_num  = opt.classes_num or 1,
        score_thres  = opt.score_thres,
        nms_thres    = opt.nms_thres,
        reg          = opt.reg,
        resize_type  = opt.resize_type,
    )
    model = UltralyticsYOLODetect(cfg)
    model.set_scheduling_params(priority=opt.priority, bpu_cores=opt.bpu_cores)
    print(f"Model input: {model.input_w}x{model.input_h}")

    # Auto-infer classes_num
    box_last = 4 * opt.reg
    cls_dims = {
        model.model.output_shapes[model.model_name][n][-1]
        for n in model.output_names
        if model.model.output_shapes[model.model_name][n][-1] != box_last
    }
    if len(cls_dims) != 1:
        print(f"[error] cannot auto-detect classes_num, candidates={cls_dims}")
        sys.exit(-1)
    detected = cls_dims.pop()
    if opt.classes_num and opt.classes_num != detected:
        print(f"[warn] --classes-num={opt.classes_num} overridden by model's {detected}")
    cfg.classes_num = detected
    print(f"classes_num = {detected} (auto-detected from model)")

    if opt.label_file:
        class_names = file_io.load_class_names(opt.label_file)
        print(f"Loaded {len(class_names)} class names from {opt.label_file}")
    else:
        class_names = [f"cls_{i}" for i in range(detected)]

    # ----- Tracker ---------------------------------------------------------- #
    tracker = None
    if enable_track:
        tracker = Tracker(
            dist_thresh      = opt.track_dist_thresh,
            low_dist_thresh  = opt.track_dist_thresh * 0.33,  # relaxed for low-conf detections
            ema_alpha        = opt.track_alpha,
            max_lost         = opt.track_max_lost,
        )
        print(f"Tracker: dist_thresh={opt.track_dist_thresh}  alpha={opt.track_alpha}  "
              f"max_lost={opt.track_max_lost}")

    # ----- Auto-tracking: MSP + CRSF + RCMux + Controller --------------------- #
    tracking = opt.tracking
    tc = None
    msp = None
    crsf = None
    mux = None
    channel_source = None
    if tracking:
        from msp import MSP
        from crsf import CRSFReader
        from rc_mux import RCMux
        from tracking_controller import TrackingController
        msp = MSP()
        msp.telemetry_start()
        if opt.crsf_port:
            crsf = CRSFReader(opt.crsf_port, opt.crsf_baud)
        tc = TrackingController(pid_yaw=opt.pid_yaw,
                                pid_pitch=opt.pid_pitch,
                                pid_thr=opt.pid_thr,
                                rate_limit=opt.tracking_rate_limit,
                                target_area_ratio=opt.tracking_target_area,
                                lost_timeout=opt.tracking_lost_timeout,
                                max_lost_time=opt.tracking_max_lost,
                                attitude_fn=msp.attitude)

        # 通道源: CRSF 直连 > MSP RC 缓存 (FC 架构)
        if crsf is not None:
            channel_source = None  # RCMux 内部用 crsf.channels()
        else:
            def _msp_channel_source():
                ch, t = msp.rc_cached()
                if ch is None:
                    return None, float('inf')
                return ch, time.monotonic() - t
            channel_source = _msp_channel_source

        mux = RCMux(msp, crsf=crsf,
                    switch_channel=opt.tracking_switch,
                    base_throttle=opt.base_throttle,
                    channel_source=channel_source)
        mux.on_engage = tc.engage
        mux.on_disengage = tc.disengage
        mux.start()
        print("Auto-tracking: ENABLED (full MSP RX fusion)")
        print(f"  PID yaw={opt.pid_yaw}  pitch={opt.pid_pitch}  thr={opt.pid_thr}")
        if crsf is not None:
            print(f"  CRSF passthrough: {opt.crsf_port} — ARM 由飞手 AUX1 掌握")
            if opt.tracking_switch is not None:
                print(f"  切入开关: CRSF CH{opt.tracking_switch} (须先见低位)")
        elif channel_source is not None:
            print("  FC 架构: 接收机在飞控 (CRSF Serial RX), 通道读 MSP_GET_RC")
            print("  BOXMSPOVERRIDE 由飞手 AUX3 物理开关控制")
            if opt.tracking_switch is not None:
                print(f"  切入开关: CH{opt.tracking_switch} (MSP RC 缓存)")
        else:
            print(f"  BENCH mode (无 CRSF / 无 FC): 键盘 't' 切换, base_throttle={opt.base_throttle}")

    # ----- SPI PAL video output ---------------------------------------------- #
    spi_video = opt.spi_video
    pal = None
    if spi_video:
        from video import PALVideo
        pal = PALVideo()
        pal.start()
        print("SPI PAL video output: ENABLED (Pin 19 → FC VI)")

    # ----- Goggles / large UI mode ----------------------------------------- #
    goggles = opt.goggles
    if goggles:
        ui_scale = 1.8           # multiplier for all font sizes / thickness
    else:
        ui_scale = 1.0

    # ----- Window ----------------------------------------------------------- #
    win_name = "YOLO Detection (USB)"
    if not headless:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if opt.fullscreen:
            cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap.start()

    # Output video writer
    video_writer = None
    if opt.save:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(opt.save, fourcc, cap.actual_fps, (w, h))
        print(f"Saving output to: {opt.save}")

    # ----- Loop ------------------------------------------------------------- #
    def _fps(ts):
        return ((len(ts) - 1) / (ts[-1] - ts[0])) if len(ts) >= 2 else 0.0

    infer_ts   = deque(maxlen=opt.window_frames)
    display_ts = deque(maxlen=opt.window_frames)
    last_seq = 0
    n_infer   = 0
    n_display = 0
    t_loop_start = None
    last_print_t   = time.perf_counter()
    last_read_count = 0

    try:
        while True:
            frame, last_seq, t = cap.read_blocking(last_seq, timeout=2.0)
            if frame is None:
                print("Reader stopped or timed out.")
                break

            # ---- inference ---- #
            raw_boxes, raw_scores, raw_cls_ids = model.predict(frame)
            t_after_infer = time.perf_counter()
            infer_ts.append(t_after_infer)
            n_infer += 1

            if opt.debug_dets and len(raw_boxes) > 0:
                fh, fw = frame.shape[:2]
                print(f"[debug] frame={fw}x{fh}  raw_dets={len(raw_boxes)}")
                for b, s, c in zip(raw_boxes[:5], raw_scores[:5], raw_cls_ids[:5]):
                    name = (class_names[int(c)]
                            if int(c) < len(class_names) else str(int(c)))
                    print(f"        box=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}) "
                          f"score={s:.3f} cls={int(c)}({name})")

            # ---- tracking ---- #
            if tracker is not None:
                t_boxes, t_scores, t_cls_ids, t_ids = tracker.step(
                    raw_boxes, raw_scores, raw_cls_ids)
            else:
                t_boxes   = raw_boxes
                t_scores  = raw_scores
                t_cls_ids = raw_cls_ids
                t_ids     = None

            # ---- auto-tracking: 每帧计算并发布指令 (mux 决定是否使用) ---- #
            if tracking and tc is not None:
                cmd = tc.step(frame, raw_boxes, raw_scores, raw_cls_ids, tracker)
                if cmd is not None:
                    mux.publish_cmd(cmd)

            # ---- FPS counters ---- #
            if t_loop_start is None:
                t_loop_start = t_after_infer
                last_read_count = cap.read_count

            infer_fps   = _fps(infer_ts)
            display_fps = _fps(display_ts)
            window = max(t_after_infer - last_print_t, 1e-6)
            capture_fps = (cap.read_count - last_read_count) / window

            # ---- display / save ---- #
            do_vis = (not headless) or (video_writer is not None) or (spi_video and pal is not None)
            if do_vis:
                vis = frame.copy()

                if tracker is not None and len(t_ids):
                    draw_tracks(vis, tracker.tracks, max_trail=20,
                                thickness=max(2, int(2 * ui_scale)),
                                label_scale=min(1.0, 0.45 * ui_scale),
                                label_thickness=max(1, int(1 * ui_scale)),
                                trail_width=max(1, int(1 * ui_scale)))
                elif len(t_boxes):
                    visualize.draw_boxes(vis, np.array(t_boxes),
                                         np.array(t_cls_ids),
                                         np.array(t_scores),
                                         class_names, visualize.rdk_colors)

                draw_status_bar(vis, capture_fps, infer_fps, display_fps,
                                n_tracks=len(t_boxes), n_raw_dets=len(raw_boxes),
                                ui_scale=ui_scale,
                                tracking_state=(f"{mux.mode}:{tc.state}"
                                                if tc is not None else None))

                if spi_video and pal is not None:
                    pal.update_frame_bgr(vis)

                if video_writer is not None:
                    video_writer.write(vis)

                if not headless:
                    cv2.imshow(win_name, vis)
                    display_ts.append(time.perf_counter())
                    n_display += 1
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('t') and tracking and mux is not None:
                        if crsf is not None or channel_source is not None:
                            print("[TRACK] 摇杆/物理开关控制中, 键盘切换已禁用")
                        else:
                            engaged = mux.bench_toggle()
                            print(f"[TRACK] bench {'ENGAGED' if engaged else 'DISENGAGED'}")

            # ---- logging ---- #
            now = time.perf_counter()
            if now - last_print_t >= opt.print_interval:
                if tracker is not None:
                    print(f"infer={n_infer:6d}  capture={capture_fps:5.1f}  "
                          f"infer={infer_fps:5.1f}  display={display_fps:5.1f}  "
                          f"tracks={len(t_boxes)}  raw={len(raw_boxes)}")
                else:
                    print(f"infer={n_infer:6d}  capture={capture_fps:5.1f}  "
                          f"infer={infer_fps:5.1f}  display={display_fps:5.1f}  "
                          f"det={len(raw_boxes)}")
                last_print_t = now
                last_read_count = cap.read_count

            if opt.max_frames and n_infer >= opt.max_frames:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        # 先关追踪 (发送 DISARM + throttle=1000), 再关相机 (可能阻塞 2s)
        if mux is not None:
            mux.stop(park=(crsf is None))
            print(f"[TRACK] mux stopped (mode={mux.mode})")
        if crsf is not None:
            crsf.close()
        if msp is not None:
            msp.close()
        cap.stop()
        if pal is not None:
            fcnt, drop = pal.stats
            print(f"SPI video: {fcnt} frames sent, {drop} dropped")
            pal.stop()
        if video_writer is not None:
            video_writer.release()
            print(f"Saved: {opt.save}")
        if not headless:
            cv2.destroyAllWindows()

    if t_loop_start is not None:
        elapsed = time.perf_counter() - t_loop_start
        if elapsed > 0:
            print(f"\nElapsed: {elapsed:.2f}s")
            print(f"  capture: {cap.read_count} frames  ({cap.read_count/elapsed:.2f} FPS)")
            print(f"  infer:   {n_infer} frames  ({n_infer/elapsed:.2f} FPS)")
            if n_display:
                print(f"  display: {n_display} frames  ({n_display/elapsed:.2f} FPS)")


if __name__ == '__main__':
    main()
