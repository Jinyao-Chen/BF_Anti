"""从试验 CSV 画弹道轨迹图（ASCII 标签，避免 CJK 字体缺失变成方块）。

用法: python3 sim/plot_trial_trajectory.py out.png name1=标签1 name2=标签2 ...
"""
from __future__ import annotations

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    rows = list(csv.DictReader(open(path)))
    cols = {}
    for k in ("t", "own_e", "own_n", "own_u", "truth_e", "truth_n", "truth_u",
              "truth_range", "own_vu", "range_est", "e_u", "e_v",
              "rc_pitch", "rc_thr", "pitch", "phase"):
        try:
            cols[k] = [float(r[k]) if k != "phase" and r[k] not in ("", "nan") else
                       (r[k] if k == "phase" else float("nan")) for r in rows]
        except Exception:
            cols[k] = []
    return cols


def main():
    out = sys.argv[1]
    specs = [s.split("=", 1) for s in sys.argv[2:]]
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    for i, (name, label) in enumerate(specs):
        c = load("/tmp/trial_%s.csv" % name)
        col = colors[i % len(colors)]
        rng = c["truth_range"]
        j = rng.index(min(rng))
        miss = min(rng)
        tag = "%s  (miss %.2f m)" % (label, miss)

        # (a) 侧视弹道：北向距离 vs 高度，目标标出
        dn = [n - c["own_n"][0] for n in c["own_n"]]   # own_* 现为 own_truth_*（真值）
        ax[0, 0].plot(dn, c["own_u"], color=col, lw=1.6, label=tag)
        ax[0, 0].plot(dn[j], c["own_u"][j], "o", color=col, ms=7)
        if i == 0:
            tgt_n = c["truth_n"][0] - c["own_n"][0]
            ax[0, 0].plot([tgt_n], [c["truth_u"][0]], "k*", ms=16, label="target")
            ax[0, 0].axhline(c["truth_u"][0], color="k", ls=":", lw=1)

        # (b) 高度 vs 时间（含目标高度）
        ax[0, 1].plot(c["t"], c["own_u"], color=col, lw=1.6, label=tag)

        # (c) 距离 vs 时间（对数）
        ax[1, 0].plot(c["t"], rng, color=col, lw=1.6, label=tag)

        # (d) 上升速度 vs 时间
        ax[1, 1].plot(c["t"], c["own_vu"], color=col, lw=1.4, label=tag)

    ax[0, 0].set_title("(a) side view: downrange vs altitude")
    ax[0, 0].set_xlabel("north / downrange (m)")
    ax[0, 0].set_ylabel("altitude (m)")
    ax[0, 0].grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].axhline(specs and 12.0 or 12.0, color="k", ls=":", lw=1)
    ax[0, 1].set_title("(b) altitude vs time  (dotted = target 12 m)")
    ax[0, 1].set_xlabel("t (s)")
    ax[0, 1].set_ylabel("altitude (m)")
    ax[0, 1].grid(alpha=0.3)
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].axhline(0.8, color="k", ls="--", lw=1)
    ax[1, 0].set_title("(c) true range vs time  (dashed = 0.8 m kill radius)")
    ax[1, 0].set_xlabel("t (s)")
    ax[1, 0].set_ylabel("range (m)")
    ax[1, 0].set_ylim(0, 50)
    ax[1, 0].grid(alpha=0.3)
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].axhline(0, color="k", ls=":", lw=1)
    ax[1, 1].set_title("(d) vertical speed vs time  (+ = rising -> attacks from below)")
    ax[1, 1].set_xlabel("t (s)")
    ax[1, 1].set_ylabel("v_up (m/s)")
    ax[1, 1].grid(alpha=0.3)
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("P-only visual strike trajectories (real camera + real YOLO)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=115)
    print("wrote", out)


if __name__ == "__main__":
    main()