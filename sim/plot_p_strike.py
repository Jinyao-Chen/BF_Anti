#!/usr/bin/env python3
"""P-only 打击日志绘图：把 `sim/run_p_vision_strike.py` 的 CSV 画成判读图。

面板：3D 轨迹 / 俯视脱靶 / 距离与视觉测距 / 图像误差与 bbox / 姿态与油门 / 速度。

用法::
    python3 sim/plot_p_strike.py --log sim/results/p_strike/hit_b_0p45m.csv \
        --out sim/results/p_strike/hit_b_0p45m_fig.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def num(row, key):
    try:
        v = float(row.get(key, "nan"))
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description="P-only 打击日志绘图")
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="P-only visual strike (Gazebo)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(args.log, encoding="utf-8")))
    if not rows:
        print("empty log", file=sys.stderr)
        return 1
    t = np.array([num(r, "t") for r in rows])
    own = np.array([[num(r, "own_truth_e"), num(r, "own_truth_n"), num(r, "own_truth_u")] for r in rows])
    tgt = np.array([[num(r, "truth_e"), num(r, "truth_n"), num(r, "truth_u")] for r in rows])
    rng = np.array([num(r, "truth_range") for r in rows])
    r_est = np.array([num(r, "range_est") for r in rows])
    e_u = np.array([num(r, "e_u") for r in rows])
    e_v = np.array([num(r, "e_v") for r in rows])
    s_px = np.array([num(r, "s_px") for r in rows])
    thr = np.array([num(r, "throttle_pwm") for r in rows])
    pitch = np.array([num(r, "truth_pitch") for r in rows])
    roll = np.array([num(r, "truth_roll") for r in rows])
    spd = np.hypot([num(r, "own_ve") for r in rows], [num(r, "own_vn") for r in rows])
    i_min = int(np.nanargmin(rng))
    miss = rng[i_min]

    fig = plt.figure(figsize=(15, 9))
    # 标题用 ASCII：matplotlib 默认字体不含 CJK，中文会显示成方块
    fig.suptitle(f"{args.title}   |   min range {miss:.2f} m @ t={t[i_min]:.2f}s "
                 f"|   impact speed {spd[i_min]:.1f} m/s", fontsize=13)

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot(own[:, 0], own[:, 1], own[:, 2], "-", color="#1f77b4", lw=2, label="interceptor")
    ax.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], "--", color="#d62728", lw=1.5, label="target")
    ax.scatter(own[0, 0], own[0, 1], own[0, 2], c="g", marker="o", label="start")
    ax.scatter(own[i_min, 0], own[i_min, 1], own[i_min, 2], c="k", marker="x", label="closest")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("Up [m]")
    ax.set_title("3D trajectory"); ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(own[:, 0], own[:, 1], "-", color="#1f77b4", label="interceptor")
    ax.plot(tgt[:, 0], tgt[:, 1], "--", color="#d62728", label="target")
    ax.scatter(own[i_min, 0], own[i_min, 1], c="k", marker="x", s=60, label="closest")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Top view (E-N)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 3)
    ax.plot(t, rng, "-", color="#d62728", label="truth range")
    ax.plot(t, r_est, ":", color="#1f77b4", label="visual range est")
    ax.axhline(0.8, color="k", ls="--", lw=1, label="kill radius 0.8 m")
    ax.set_xlabel("t [s]"); ax.set_ylabel("range [m]"); ax.set_yscale("log")
    ax.set_title("Range"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 4)
    ax.plot(t, e_u, "-", label="e_u (horizontal)")
    ax.plot(t, e_v, "-", label="e_v (vertical)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t [s]"); ax.set_ylabel("normalized image error")
    ax.set_title("Image error (0 = centered)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t, s_px, color="#39a", label="bbox width [px]")
    ax.axhline(467.8 * 0.55 / 0.8, color="k", ls="--", lw=1, label="bbox hit size")
    ax.set_xlabel("t [s]"); ax.set_ylabel("px"); ax.set_title("Detection size (visual ranging)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = fig.add_subplot(2, 3, 6)
    ax.plot(t, pitch, label="pitch [deg]")
    ax.plot(t, roll, label="roll [deg]")
    ax.set_xlabel("t [s]"); ax.set_ylabel("attitude [deg]")
    ax2 = ax.twinx()
    ax2.plot(t, thr, color="0.5", ls=":", label="throttle PWM")
    ax2.set_ylabel("PWM")
    ax.set_title("Attitude & throttle"); ax.legend(fontsize=8, loc="lower left"); ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=110)
    print(f"wrote {args.out}  (min range {miss:.3f} m at t={t[i_min]:.2f}s, "
          f"impact speed {spd[i_min]:.1f} m/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
