#!/usr/bin/env python3
"""打击日志诊断 v2：起飞段 / 检测连续性 / 终端几何 / 重捕。

用法: python3 sim/diag_run.py /tmp/trial_v1a.csv [更多...]
只读日志。
"""
from __future__ import annotations

import csv
import math
import sys

import numpy as np


def wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def load(path):
    rows = list(csv.DictReader(open(path)))
    d = {}
    for k in rows[0].keys():
        try:
            d[k] = np.array([float(r[k]) for r in rows], dtype=float)
        except ValueError:
            d[k] = np.array([r[k] for r in rows])
    return d


def ppt(y, m=None):
    v = y if m is None else y[m]
    v = v[np.isfinite(v)]
    return 0.0 if len(v) < 2 else float(np.nanmax(v) - np.nanmin(v))


def gaps(mask):
    """最长 False 连续段（单位：帧）。"""
    best = cur = 0
    for m in mask:
        cur = cur + 1 if not m else 0
        best = max(best, cur)
    return best


def report(path):
    d = load(path)
    t = d["t"]
    n = len(t)
    dt = np.median(np.diff(t))
    print("=" * 84)
    print("%s  n=%d  %.1fs  %.1f Hz" % (path, n, t[-1] - t[0], 1.0 / max(1e-9, dt)))
    tr = d["truth_range"]
    imin = int(np.nanargmin(tr))
    de = d["truth_e"][imin] - d["own_truth_e"][imin]
    dn = d["truth_n"][imin] - d["own_truth_n"][imin]
    du = d["truth_u"][imin] - d["own_truth_u"][imin]
    print("min truth range = %.3f m at t=%.2f (phase=%s)  miss: horiz=%.2f vert=%+.2f bearing-off-axis=%+.1f"
          % (tr[imin], t[imin], d["phase"][imin], math.hypot(de, dn), du,
             wrap(math.degrees(math.atan2(de, dn)) - d["truth_yaw"][imin])))
    det = np.isfinite(d["s_px"]) & (d["s_px"] > 0)
    print("tracked frames %d/%d (%.0f%%)   max detection gap = %.2f s" % (
        det.sum(), n, 100.0 * det.sum() / n, gaps(det) * dt))
    if "track_reason" in d:
        rs = d["track_reason"]
        for r in ("det", "coast", "gated", "lost"):
            m = rs == r
            if m.any():
                print("   reason %-6s %4d (%.0f%%)" % (r, m.sum(), 100.0 * m.sum() / n))
        raw = np.isfinite(d["raw_u"])
        print("   raw YOLO detections %d/%d (%.0f%%)  max raw gap %.2f s   median det_age %.3f s"
              % (raw.sum(), n, 100.0 * raw.sum() / n, gaps(raw) * dt,
                 np.nanmedian(d["det_age"])))
    for name, m in (("takeoff 0-3s", (t < 3.0)),
                    ("3-8s", (t >= 3) & (t < 8)),
                    ("terminal last 2s", (t >= t[imin] - 2) & (t <= t[imin]))):
        if m.sum() < 4:
            continue
        dr = np.r_[0.0, np.diff(d["truth_roll"]) / np.maximum(1e-3, np.diff(t))]
        dy = np.r_[0.0, np.diff(d["truth_yaw"]) / np.maximum(1e-3, np.diff(t))]
        zc = int((np.diff(np.sign(d["throttle_pwm"][m] - np.median(d["throttle_pwm"][m]))) != 0).sum())
        print("  [%-16s] roll pp=%5.1f yawr pp=%6.1f thr pp=%4.0f thr零穿=%3d sat=%2.0f%% | "
              "truth rollrate pp=%6.1f yawrate pp=%6.1f | det=%.0f%%"
              % (name, ppt(d["roll_cmd"], m), ppt(d["yaw_rate_cmd"], m), ppt(d["throttle_pwm"], m), zc,
                 100 * np.mean(d["sat"][m]), ppt(dr, m), ppt(dy, m), 100 * np.mean(det[m])))
    if "cam_age" in d:
        print("   cam_age: median %.3f  p90 %.3f  max %.3f s" % (
            np.nanmedian(d["cam_age"]), np.nanpercentile(d["cam_age"], 90), np.nanmax(d["cam_age"])))
    return d


if __name__ == "__main__":
    for p in sys.argv[1:]:
        try:
            report(p)
        except Exception as exc:
            print("FAILED %s: %r" % (p, exc))
