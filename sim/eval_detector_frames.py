#!/usr/bin/env python3
"""离线检测器 A/B：在 --dump-dir 落的**原始仿真帧**上比较两个模型。

用途：换检测后端（.pt vs .onnx vs 板端 .bin）时，先离线看三个真正决定命中率的量：
  1) 召回随距离的分布   —— 远距离漏检 => 全程 SEARCH
  2) 置信度分布         —— 阈值怎么定
  3) 框宽/真值距离的隐含 W_eff = s_px*r/fx  —— 这个直接决定 --target-width，
                          错标会把 z_t = r*sin(el) 整体带偏（竖向通道）。

用法::
  python3 sim/eval_detector_frames.py --frames-dir /tmp/frames_pt \\
      --pt Anti-UAV/best.pt --onnx model/v1_yolov8_mAP50_0.879.onnx --out sim/results/...
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import numpy as np

DEFAULT_FX = 467.8   # 仿真相机 fx=fy（640x480, hfov 1.2 rad）


def letterbox(img, new=640, color=114):
    import cv2
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    top, left = (new - nh) // 2, (new - nw) // 2
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((new, new, 3), color, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = rs
    return out, r, float(left), float(top)


def nms(boxes, scores, iou_thres):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(1e-9, areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=int)


class OnnxDet:
    def __init__(self, path, providers=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(path, so, providers=providers or ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.out = self.sess.get_outputs()[0].name
        self.hw = int(self.inp.shape[2])

    def detect(self, img, conf_thres=0.25, iou_thres=0.7, max_det=20):
        x, r, dx, dy = letterbox(img, self.hw)
        blob = np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)
        y = np.asarray(self.sess.run([self.out], {self.inp.name: blob})[0])[0]
        boxes = y[:4].T.astype(np.float64)
        scores = y[4:].T.astype(np.float64)
        cls = scores.argmax(1)
        conf = scores[np.arange(scores.shape[0]), cls]
        m = conf >= conf_thres
        boxes, conf, cls = boxes[m], conf[m], cls[m]
        if boxes.shape[0] == 0:
            return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
        xyxy = np.stack([boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2,
                         boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2], 1)
        keep = nms(xyxy, conf, iou_thres)[:max_det]
        xyxy = xyxy[keep]
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - dx) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - dy) / r
        return xyxy, conf[keep], cls[keep]


def pick(boxes, confs):
    """选面积最大的框（仿真里只有一架目标，避免误检挑走）。"""
    if len(boxes) == 0:
        return None
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    i = int(np.argmax(area))
    b = boxes[i]
    return {"xyxy": [float(v) for v in b], "conf": float(confs[i]),
            "w": float(b[2] - b[0]), "h": float(b[3] - b[1]),
            "cx": float((b[0] + b[2]) / 2), "cy": float((b[1] + b[3]) / 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--pt", default=None)
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--conf-low", type=float, default=0.10)
    ap.add_argument("--fx", type=float, default=DEFAULT_FX)
    ap.add_argument("--cpu", action="store_true", help="强制 .pt 走 CPU（默认也走 CPU）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import cv2
    dump_csv = os.path.join(args.frames_dir, "dump.csv")
    entries = []
    if os.path.exists(dump_csv):
        with open(dump_csv) as fh:
            for row in csv.DictReader(fh):
                entries.append((float(row["t"]), float(row["truth_range"]),
                                row.get("file") or ""))
    else:
        for name in sorted(os.listdir(args.frames_dir)):
            if name.endswith((".jpg", ".png")):
                entries.append((0.0, float("nan"), name))
    print("frames in dir:", len(entries), flush=True)

    onnx_det = OnnxDet(args.onnx) if args.onnx else None
    pt = None
    if args.pt:
        from ultralytics import YOLO
        pt = YOLO(args.pt)

    rows = []
    seen = set()
    for t, trange, name in entries:
        if not name:
            name = "t%010.3f.jpg" % t
        path = os.path.join(args.frames_dir, name)
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        h = hashlib.md5(img.tobytes()).hexdigest()
        duplicate = h in seen          # 主循环 20 Hz 复用相机 13 Hz 的帧 -> 去重
        seen.add(h)
        row = {"t": t, "truth_range": trange, "file": name, "dup": int(duplicate)}
        if onnx_det is not None:
            b, c, _ = onnx_det.detect(img, conf_thres=args.conf_low)
            p = pick(b, c)
            row["onnx_conf"] = p["conf"] if p else 0.0
            row["onnx_w"] = p["w"] if p else 0.0
            row["onnx_h"] = p["h"] if p else 0.0
            row["onnx_ok"] = int(bool(p) and p["conf"] >= args.conf)
        if pt is not None:
            r = pt.predict(img, imgsz=640, conf=args.conf_low, device="cpu", verbose=False)[0]
            bx = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
            cf = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
            p = pick(bx, cf)
            row["pt_conf"] = p["conf"] if p else 0.0
            row["pt_w"] = p["w"] if p else 0.0
            row["pt_h"] = p["h"] if p else 0.0
            row["pt_ok"] = int(bool(p) and p["conf"] >= args.conf)
        rows.append(row)

    fresh = [r for r in rows if not r["dup"]]
    print("unique frames:", len(fresh), "of", len(rows), flush=True)

    bins = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)]
    summary = {"n_unique": len(fresh), "conf": args.conf, "bins": []}
    for lo, hi in bins:
        sel = [r for r in fresh if lo <= r["truth_range"] < hi]
        if not sel:
            continue
        e = {"range": "%g-%g" % (lo, hi), "n": len(sel)}
        for tag in ("onnx", "pt"):
            k = tag + "_ok"
            if k in sel[0]:
                e[tag + "_recall"] = round(sum(r[k] for r in sel) / len(sel), 3)
                ck = tag + "_conf"
                e[tag + "_conf_med"] = round(float(np.median([r[ck] for r in sel])), 3)
        summary["bins"].append(e)

    for tag in ("onnx", "pt"):
        if tag + "_ok" not in rows[0]:
            continue
        sel = [r for r in fresh if r[tag + "_ok"]]
        summary[tag + "_recall"] = round(len(sel) / max(1, len(fresh)), 3)
        band = [r for r in sel if 4.0 <= r["truth_range"] <= 35.0]
        if band:
            w = np.array([r[tag + "_w"] for r in band])
            rr = np.array([r["truth_range"] for r in band])
            w_eff = w * rr / args.fx
            summary[tag + "_w_eff_med"] = round(float(np.median(w_eff)), 3)
            summary[tag + "_w_eff_iqr"] = [round(float(np.percentile(w_eff, 25)), 3),
                                           round(float(np.percentile(w_eff, 75)), 3)]
            summary[tag + "_n_calib"] = len(band)

    print(json.dumps(summary, indent=2), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        print("wrote", args.out, flush=True)
        with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
            json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
