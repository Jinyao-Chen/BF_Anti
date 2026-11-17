#!/usr/bin/env python3
"""Aggregate paired controller trials and print a compact markdown report."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def _num(v):
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _median(vals):
    vals = sorted(float(v) for v in vals if _num(v))
    if not vals:
        return float("nan")
    n = len(vals)
    return vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="sim/results/controller_bench_2026-11-05")
    args = ap.parse_args()
    root = Path(args.root)
    recs = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    groups = defaultdict(list)
    for r in recs:
        groups[r.get("mode", "?")].append(r)

    lines = ["# BF_Anti controller bench", ""]
    lines.append("| mode | n | hits | hit rate | CPA median (m) | track | "
                 "LOS p90 median (deg) | terminal pitch-rate p90 (deg/s) | roll range (deg) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for mode, rows in sorted(groups.items()):
        hits = sum(1 for r in rows if r.get("hit"))
        lines.append("| {} | {} | {} | {:.0%} | {:.3f} | {:.2f} | {:.2f} | {:.1f} | {:.1f} |".format(
            mode, len(rows), hits, hits / max(1, len(rows)),
            _median(r.get("min_truth_range") for r in rows),
            _median(r.get("track_fraction") for r in rows),
            _median(r.get("los_p90_track") for r in rows),
            _median(r.get("terminal_pitch_rate_p90") for r in rows),
            _median(r.get("terminal_roll_range") for r in rows)))
    lines += ["", "## Paired cases", "",
              "| case | mode | CPA (m) | hit | track | LOS p90 | terminal pitch | pitch-rate p90 | roll range |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(recs, key=lambda x: (x.get("case", ""), x.get("mode", ""))):
        lines.append("| {} | {} | {:.3f} | {} | {:.2f} | {:.2f} | {:.1f} | {:.1f} | {:.1f} |".format(
            r.get("case"), r.get("mode"),
            float(r.get("min_truth_range", float("nan"))),
            "Y" if r.get("hit") else "N",
            float(r.get("track_fraction", float("nan"))),
            float(r.get("los_p90_track", float("nan"))),
            float(r.get("terminal_pitch", float("nan"))),
            float(r.get("terminal_pitch_rate_p90", float("nan"))),
            float(r.get("terminal_roll_range", float("nan")))))
    text = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
