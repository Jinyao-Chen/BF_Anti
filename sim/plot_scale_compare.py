"""对照图：同一发数据的 x-y 平面，左=被放大的 auto 比例，右=统一 50 m 量程。

用途：确认"横向 0.37 m 不应该被画成半屏"这一约定。
"""
from __future__ import annotations

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

name = sys.argv[2] if len(sys.argv) > 2 else "directki"
out = sys.argv[1]
rows = list(csv.DictReader(open("/tmp/trial_%s.csv" % name)))
g = lambda k: [float(r[k]) for r in rows if r[k] not in ("", "nan")]  # noqa: E731
x, y = g("own_truth_e"), g("own_truth_n")
tx, ty = g("truth_e")[0], g("truth_n")[0]
rng = g("truth_range")
j = rng.index(min(rng))

fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))

# 左：auto 比例（错误示范）
ax[0].plot(x, y, color="#d62728", lw=2.0)
ax[0].scatter([tx], [ty], marker="*", s=300, color="k")
ax[0].plot(x[j], y[j], "o", mfc="none", mec="red", ms=12, mew=2.4)
ax[0].set_xlabel("East x (m)")
ax[0].set_ylabel("North y (m)")
ax[0].set_title("WRONG: auto aspect\nx span 0.5 m  ->  0.1 m = ~120 px (magnified)")
ax[0].grid(alpha=0.35)

# 右：统一 50 m（1:1，与其它面板同标尺）
ax[1].plot(x, y, color="#1f77b4", lw=2.0)
ax[1].scatter([tx], [ty], marker="*", s=300, color="k", label="target")
ax[1].plot(x[j], y[j], "o", mfc="none", mec="red", ms=12, mew=2.4,
           label="closest %.2f m" % min(rng))
ax[1].set_xlim(-25, 25)
ax[1].set_ylim(0, 50)
ax[1].set_aspect("equal", adjustable="box")
ax[1].set_xlabel("East x (m)")
ax[1].set_ylabel("North y (m)")
ax[1].set_title("CORRECT: same 50 m scale on x and y\nx span 50 m  ->  0.1 m = ~1 px")
ax[1].grid(alpha=0.35)
ax[1].legend(fontsize=9)

fig.suptitle("x-y plane, true East span = %.3f m (run %s)" % (max(x) - min(x), name),
             fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=115)
print("wrote", out)
