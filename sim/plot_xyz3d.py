"""四联图：三维轨迹 + xyz 三视图。

**标尺统一**：四个面板一律 50 m x 50 m、等比例(1:1)，因此 0.2 m 的横向偏移
在图上就是 0.2 m（不会被放大成满屏）。

用法: python3 sim/plot_xyz3d.py out.png <trial名> [标签]
"""
from __future__ import annotations

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

HALF = 25.0
SPAN = 50.0


def main():
    out = sys.argv[1]
    name = sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else name
    rows = list(csv.DictReader(open("/tmp/trial_%s.csv" % name)))
    g = lambda k: [float(r[k]) for r in rows if r[k] not in ("", "nan")]  # noqa: E731
    x, y, z = g("own_truth_e"), g("own_truth_n"), g("own_truth_u")
    tx, ty, tz = g("truth_e")[0], g("truth_n")[0], g("truth_u")[0]
    rng = g("truth_range")
    j = rng.index(min(rng))
    miss = min(rng)

    fig = plt.figure(figsize=(15, 10.5))

    ax = fig.add_subplot(2, 2, 1, projection="3d")
    # 必须用正交投影：默认的透视投影即使 box_aspect=(1,1,1) 也会把东向轴压短，
    # 导致三个方向的比例尺不一致（用户指出的问题）。正交投影下 1 m = 1 m。
    ax.set_proj_type("ortho")
    ax.plot(x, y, z, color="#1f77b4", lw=2.0)
    ax.scatter([tx], [ty], [tz], marker="*", s=340, color="k", label="target")
    ax.scatter([x[j]], [y[j]], [z[j]], s=110, facecolor="none", edgecolor="red",
               lw=2.2, label="closest %.2f m" % miss)
    ax.set_xlim(-HALF, HALF)
    ax.set_ylim(0, SPAN)
    ax.set_zlim(0, SPAN)
    ax.set_box_aspect((1, 1, 1))          # 三轴同跨度(50 m) + 正方体盒 = 等比例
    ax.set_xticks(range(-20, 21, 10))
    ax.set_yticks(range(0, 51, 10))
    ax.set_zticks(range(0, 51, 10))
    ax.set_xlabel("East x (m)")
    ax.set_ylabel("North y (m)")
    ax.set_zlabel("Up z (m)")
    ax.set_title("3D trajectory - %s" % label)
    ax.legend(loc="upper left", fontsize=9)
    ax.view_init(elev=18, azim=-70)

    def panel(pos, a1, a2, n1, n2, title, col, lim1, lim2):
        p = fig.add_subplot(2, 2, pos)
        p.plot(a1, a2, color=col, lw=2.0)
        p.scatter([a1[0]], [a2[0]], s=55, color="0.35", zorder=3, label="start")
        tgt1 = {"East x (m)": tx, "North y (m)": ty, "Up z (m)": tz}[n1]
        tgt2 = {"East x (m)": tx, "North y (m)": ty, "Up z (m)": tz}[n2]
        p.scatter([tgt1], [tgt2], marker="*", s=300, color="k", zorder=4, label="target")
        p.plot(a1[j], a2[j], "o", mfc="none", mec="red", ms=12, mew=2.4,
               label="closest %.2f m" % miss)
        p.set_xlim(*lim1)
        p.set_ylim(*lim2)
        p.set_aspect("equal", adjustable="box")
        p.set_xlabel(n1)
        p.set_ylabel(n2)
        p.set_title(title)
        p.grid(alpha=0.35)

    panel(2, x, y, "East x (m)", "North y (m)", "x-y  (both axes 50 m)",
          "#1f77b4", (-HALF, HALF), (0, SPAN))
    panel(3, x, z, "East x (m)", "Up z (m)", "x-z  (both axes 50 m)",
          "#d62728", (-HALF, HALF), (0, SPAN))
    panel(4, y, z, "North y (m)", "Up z (m)", "y-z  (both axes 50 m)",
          "#2ca02c", (0, SPAN), (0, SPAN))

    fig.suptitle("P-only visual strike - %s   (miss %.2f m;  all panels same 50 m scale)"
                 % (label, miss), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=115)
    print("wrote", out, "| miss %.3f m" % miss)


if __name__ == "__main__":
    main()