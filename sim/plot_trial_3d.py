"""最好一发弹道的三维图：3D 轨迹 + XY / XZ / YZ 三个投影（x,y,z 都要）。

用法: python3 sim/plot_trial_3d.py out.png best=trial名 [other=trial名 ...]
"""
from __future__ import annotations

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401


def load(name):
    rows = list(csv.DictReader(open("/tmp/trial_%s.csv" % name)))
    g = lambda k: [float(r[k]) for r in rows if r[k] not in ("", "nan")]  # noqa: E731
    # own_* 是固件估计位置（终端可差约 10 m）；own_truth_* 才是真值。
    d = {"t": g("t"), "e": g("own_truth_e"), "n": g("own_truth_n"), "u": g("own_truth_u"),
         "ee": g("own_e"), "en": g("own_n"), "eu": g("own_u"),
         "te": g("truth_e"), "tn": g("truth_n"), "tu": g("truth_u"),
         "r": g("truth_range"), "vu": g("own_vu"), "ve": g("own_ve"),
         "vn": g("own_vn"), "phase": [r["phase"] for r in rows]}
    j = d["r"].index(min(d["r"]))
    d["j"] = j
    d["miss"] = min(d["r"])
    return d


def main():
    out = sys.argv[1]
    spec = [s.split("=", 1) for s in sys.argv[2:]]
    best_name, best_label = spec[0]
    b = load(best_name)
    te, tn, tu = b["te"][0], b["tn"][0], b["tu"][0]
    j = b["j"]

    fig = plt.figure(figsize=(15, 10))

    # ---------------- 3D ----------------
    ax = fig.add_subplot(2, 3, (1, 2), projection="3d")
    p = ax.scatter(b["e"], b["n"], b["u"], c=b["t"], cmap="viridis", s=12)
    ax.plot(b["e"], b["n"], b["u"], color="0.6", lw=1.0, zorder=0)
    ax.scatter([te], [tn], [tu], marker="*", s=350, color="k", label="target")
    ax.scatter([b["e"][j]], [b["n"][j]], [b["u"][j]], marker="o", s=90,
               facecolor="none", edgecolor="red", lw=2, label="closest %.2f m" % b["miss"])
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North / downrange (m)")
    ax.set_zlabel("Up (m)")
    ax.set_title("3D intercept trajectory - %s\n(miss %.2f m, colour = time)"
                 % (best_label, b["miss"]), fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    if b.get("ee"):
        ax.plot(b["ee"], b["en"], b["eu"], color="crimson", lw=1.0, ls="--",
                label="firmware (EKF) pos estimate - lags ~0.9 s, alt ok")
    ax.view_init(elev=22, azim=-62)
    fig.colorbar(p, ax=ax, shrink=0.6, label="t (s)")

    # ------------- 三个投影 -------------
    # XY 俯视
    a1 = fig.add_subplot(2, 3, 4)
    a1.plot(b["e"], b["n"], "o-", ms=2.5, lw=1.2, color="#1f77b4")
    a1.plot(te, tn, "k*", ms=18)
    a1.plot(b["e"][j], b["n"][j], "o", mfc="none", mec="red", ms=11, mew=2)
    a1.set_xlabel("East x (m)")
    a1.set_ylabel("North y (m)")
    a1.set_title("top view (x-y)")
    a1.grid(alpha=0.3)
    a1.set_aspect("equal", adjustable="datalim")

    # XZ 前视
    a2 = fig.add_subplot(2, 3, 5)
    a2.plot(b["e"], b["u"], "o-", ms=2.5, lw=1.2, color="#d62728")
    a2.plot(te, tu, "k*", ms=18)
    a2.plot(b["e"][j], b["u"][j], "o", mfc="none", mec="red", ms=11, mew=2)
    a2.set_xlabel("East x (m)")
    a2.set_ylabel("Up z (m)")
    a2.set_title("front view (x-z)")
    a2.grid(alpha=0.3)
    a2.set_aspect("equal", adjustable="datalim")

    # YZ 侧视（关键：从下往上还是从上往下）
    a3 = fig.add_subplot(2, 3, 6)
    a3.plot(b["n"], b["u"], "o-", ms=2.5, lw=1.2, color="#2ca02c")
    a3.plot(tn, tu, "k*", ms=18, label="target")
    a3.plot(b["n"][j], b["u"][j], "o", mfc="none", mec="red", ms=11, mew=2)
    a3.annotate("closest %.2f m\n v_up %+.2f m/s" % (b["miss"], b["vu"][j]),
                (b["n"][j], b["u"][j]), textcoords="offset points",
                xytext=(-95, 18), fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1))
    a3.set_xlabel("North y (m)")
    a3.set_ylabel("Up z (m)")
    a3.set_title("side view (y-z)  <-- from-below vs from-above")
    a3.grid(alpha=0.3)
    a3.legend(fontsize=9)
    a3.set_aspect("equal", adjustable="datalim")

    # 对照曲线（可选）
    for name, label in spec[1:]:
        o = load(name)
        a3.plot(o["n"], o["u"], lw=1.0, alpha=0.55, label=label)
    a3.legend(fontsize=8)

    fig.suptitle("best run: %s   miss %.2f m   (real camera + real YOLO)"
                 % (best_label, b["miss"]), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=112)
    print("wrote", out)
    print("best=%s miss=%.3f m  closest t=%.1fs  alt=%.2f m (target %.1f)  "
          "east=%.2f  v_up=%+.2f m/s"
          % (best_name, b["miss"], b["t"][j], b["u"][j], tu, b["e"][j], b["vu"][j]))


if __name__ == "__main__":
    main()