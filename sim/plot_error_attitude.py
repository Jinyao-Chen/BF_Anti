#!/usr/bin/env python3
"""Plot tracking error and desired-vs-actual attitude from a trial CSV."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--t-min", type=float, default=0.0)
    ap.add_argument("--t-max", type=float, default=0.0)
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    def f(k):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(k, "nan")))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out
    t = f("t")
    if args.t_max > 0:
        keep = [i for i, tt in enumerate(t) if args.t_min <= tt <= args.t_max]
        rows = [rows[i] for i in keep]
        t = [t[i] for i in keep]
    eu, ev = f("e_u"), f("e_v")
    los = f("truth_los_err")
    pc, pa = f("pitch_cmd"), f("truth_pitch")
    rc, ra = f("roll_cmd"), f("truth_roll")

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2), sharex=True)
    ax = axes[0, 0]
    ax.plot(t, eu, label="e_u")
    ax.plot(t, ev, label="e_v")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("normalized image error")
    ax.set_title("image error")
    ax.grid(alpha=.25); ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.plot(t, los, color="tab:red", label="truth LOS error")
    ax.set_ylabel("deg")
    ax.set_title("true LOS error")
    ax.grid(alpha=.25); ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.plot(t, pc, "--", label="pitch cmd")
    ax.plot(t, pa, label="truth pitch")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t (s)"); ax.set_ylabel("deg")
    ax.set_title("pitch: desired vs actual")
    ax.grid(alpha=.25); ax.legend(fontsize=7)

    ax = axes[1, 1]
    ax.plot(t, rc, "--", label="roll cmd")
    ax.plot(t, ra, label="truth roll")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("t (s)"); ax.set_ylabel("deg")
    ax.set_title("roll: desired vs actual")
    ax.grid(alpha=.25); ax.legend(fontsize=7)

    fig.suptitle(args.title or Path(args.csv).stem)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
