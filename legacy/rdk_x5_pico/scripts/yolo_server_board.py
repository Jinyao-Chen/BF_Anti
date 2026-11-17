#!/usr/bin/env python3
"""把 RDK X5 (地瓜派) 当成 YOLO 推理服务 —— 协议与 GPU 上的 yolo_server.py **完全一致**。

为什么能直接换：本机仿真侧的客户端 sim/run_visual_strike_yolo.py 里的 RemoteYolo 只认
一件事 —— 4 字节大端长度前缀 + JPEG 请求，回 4 字节长度 + JSON {"dets":[{cls,name,conf,xyxy}]}。
把同样的协议在板上实现，客户端一行都不用改（只换 --yolo-host）。

板上用法::

    # 依赖: rdk_model_zoo 的 Ultralytics YOLO 运行时 + opencv-python + numpy
    git clone https://github.com/D-Robotics/rdk_model_zoo.git ~/rdk_model_zoo
    ln -s ~/rdk_model_zoo ~/BOARD_HOME/BF_Anti/rdk_model_zoo
    python3 legacy/rdk_x5_pico/scripts/yolo_server_board.py \
        --model-path legacy/rdk_x5_pico/model/v1_yolov8_mAP50_0.879_bayese_640x640_nv12.bin \
        --label-file legacy/rdk_x5_pico/model/labels.txt --host 0.0.0.0 --port 9300

本机（仿真）侧::

    # 直连
    YOLO_DIRECT=1 YOLO_HOST=<板子IP> YOLO=1 GUI=1 ./sim/run_p_strike_live.sh
    # 或沿用 SSH 隧道（板上开 sshd）
    YOLO_SSH=<板子用户>@<板子IP> YOLO_REMOTE_CMD='cd ~/BF_Anti && python3 legacy/rdk_x5_pico/scripts/yolo_server_board.py ...' \
        YOLO=1 GUI=1 ./sim/run_p_strike_live.sh

类别名注意：本机客户端的 TARGET_NAMES 只认 {"copter_drone","motor_drone"}。
本 .bin 只有 1 类，所以默认就把这一类命名为 **copter_drone**（--label-file 可覆盖），
否则客户端会把所有检测都过滤掉、表现为"全程 SEARCH"。
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import time

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))
MODEL_ZOO = os.path.join(REPO_ROOT, "rdk_model_zoo")
RUNTIME_DIR = os.path.join(MODEL_ZOO, "samples/vision/ultralytics_yolo/runtime/python")
sys.path.append(MODEL_ZOO)
sys.path.append(RUNTIME_DIR)

DEFAULT_MODEL = os.path.join(ROOT, "model",
                             "v1_yolov8_mAP50_0.879_bayese_640x640_nv12.bin")
DEFAULT_NAMES = ["copter_drone"]


def load_class_names(path, n_classes):
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            names = [ln.strip() for ln in fh if ln.strip()]
        if names:
            return names
    if n_classes == 1:
        # 单类：直接命名成客户端认得的名字，省掉一次改代码
        return list(DEFAULT_NAMES)
    return ["cls_%d" % i for i in range(n_classes)]


def build_model(args):
    try:
        from ultralytics_yolo_det import (UltralyticsYOLODetect,
                                          UltralyticsYOLODetectConfig)
    except ImportError as exc:                     # noqa: PERF203
        print("[board-server] 找不到 rdk_model_zoo 运行时: %s" % exc, flush=True)
        print("[board-server] 先执行: git clone https://github.com/D-Robotics/"
              "rdk_model_zoo.git ~/rdk_model_zoo && ln -s ~/rdk_model_zoo "
              "%s" % MODEL_ZOO, flush=True)
        raise
    cfg = UltralyticsYOLODetectConfig(
        model_path=args.model_path,
        classes_num=args.classes_num or 1,
        score_thres=args.score_thres,
        nms_thres=args.nms_thres,
        reg=args.reg,
        resize_type=args.resize_type,
    )
    model = UltralyticsYOLODetect(cfg)
    model.set_scheduling_params(priority=args.priority, bpu_cores=args.bpu_cores)
    print("[board-server] model input %dx%d" % (model.input_w, model.input_h), flush=True)

    # 与 legacy/rdk_x5_pico/detect.py 相同的类数自检：YOLOv8 头里 box 维 = 4*reg，剩下那一维就是类数
    n_classes = args.classes_num or 1
    try:
        box_last = 4 * args.reg
        dims = {model.model.output_shapes[model.model_name][n][-1]
                for n in model.output_names
                if model.model.output_shapes[model.model_name][n][-1] != box_last}
        if len(dims) == 1:
            detected = int(dims.pop())
            if args.classes_num and args.classes_num != detected:
                print("[board-server] WARN --classes-num=%d 与模型 %d 不一致，用 %d"
                      % (args.classes_num, detected, detected), flush=True)
            n_classes = detected
        print("[board-server] classes_num = %d (auto)" % n_classes, flush=True)
    except Exception as exc:                      # noqa: BLE001
        print("[board-server] WARN 无法自动判定类数(%r)，沿用 %d" % (exc, n_classes),
              flush=True)
    return model, n_classes


def recv_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def main():
    ap = argparse.ArgumentParser(description="RDK X5 YOLO 推理服务（与 GPU 版同协议）")
    ap.add_argument("--model-path", default=DEFAULT_MODEL, help="X5 .bin 模型路径")
    ap.add_argument("--label-file", default=None, help="类别名文件（每行一个）")
    ap.add_argument("--classes-num", type=int, default=0, help="0=从模型输出形状自动判定")
    ap.add_argument("--score-thres", type=float, default=0.25)
    ap.add_argument("--nms-thres", type=float, default=0.70)
    ap.add_argument("--reg", type=int, default=16)
    ap.add_argument("--resize-type", type=int, default=1, choices=[0, 1])
    ap.add_argument("--priority", type=int, default=0)
    ap.add_argument("--bpu-cores", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = 允许局域网直连")
    ap.add_argument("--port", type=int, default=9300)
    ap.add_argument("--max-det", type=int, default=20)
    ap.add_argument("--stats-every", type=int, default=100)
    args = ap.parse_args()

    if not os.path.isfile(args.model_path):
        print("[board-server] ERROR 模型不存在: %s" % args.model_path, flush=True)
        return 2

    model, n_classes = build_model(args)
    class_names = load_class_names(args.label_file, n_classes)
    print("[board-server] classes: %s" % class_names, flush=True)

    dummy = np.zeros((model.input_h, model.input_w, 3), dtype=np.uint8)
    t0 = time.time()
    model.predict(dummy)
    print("[board-server] warmup %.3fs" % (time.time() - t0), flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(2)
    print("[board-server] listening %s:%d  score_thres=%.2f bpu=%s"
          % (args.host, args.port, args.score_thres, args.bpu_cores), flush=True)

    while True:
        conn, addr = srv.accept()
        print("[board-server] client %s" % (addr,), flush=True)
        conn.settimeout(30.0)
        frames = 0
        t_start = time.time()
        try:
            while True:
                hdr = recv_exact(conn, 4)
                if hdr is None:
                    break
                n = struct.unpack("!I", hdr)[0]
                if n <= 0 or n > 20_000_000:
                    break
                data = recv_exact(conn, n)
                if data is None:
                    break
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                dets = []
                if img is not None:
                    boxes, scores, cls_ids = model.predict(img)
                    order = np.argsort(-np.asarray(scores, dtype=float))[: args.max_det]
                    for i in order:
                        c = int(cls_ids[i])
                        dets.append({
                            "cls": c,
                            "name": class_names[c] if 0 <= c < len(class_names) else str(c),
                            "conf": float(scores[i]),
                            "xyxy": [float(v) for v in boxes[i]],
                        })
                payload = json.dumps({"dets": dets}).encode()
                conn.sendall(struct.pack("!I", len(payload)) + payload)
                frames += 1
                if args.stats_every > 0 and frames % args.stats_every == 0:
                    print("[board-server] %d frames, %.1f FPS"
                          % (frames, frames / max(1e-6, time.time() - t_start)), flush=True)
        except (ConnectionResetError, BrokenPipeError, socket.timeout) as exc:
            print("[board-server] connection ended: %r" % (exc,), flush=True)
        finally:
            conn.close()
            if frames:
                print("[board-server] client %s done, %d frames" % (addr, frames), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
