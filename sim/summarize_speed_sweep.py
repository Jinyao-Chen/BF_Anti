#!/usr/bin/env python3
"""Summarize normal-world multi-speed reference-tracking CSVs.

Usage:
    python3 sim/summarize_speed_sweep.py /tmp/bf_anti_speed_sweep/speed_*.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Any

import numpy as np
from scipy.signal import savgol_filter


def _column(rows, name):
    return np.array([float(r[name]) for r in rows], dtype=float)


def summarize(path: str) -> dict[str, Any]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty CSV: {path}")

    t = _column(rows, "t")
    truth_e = _column(rows, "truth_e")
    truth_n = _column(rows, "truth_n")
    truth_u = _column(rows, "truth_u")
    ref_e = _column(rows, "ref_e")
    ref_n = _column(rows, "ref_n")
    ve = _column(rows, "ve")
    vn = _column(rows, "vn")
    pitch = _column(rows, "pitch")
    motors = np.vstack([_column(rows, f"m{i}") for i in range(4)]).T

    horiz_err = np.hypot(truth_e - ref_e, truth_n - ref_n)
    speed = np.hypot(ve, vn)
    vertical_acc = np.gradient(
        savgol_filter(truth_u, 31, 3, mode="interp"), t
    )
    min_accel = float(np.nanmin(vertical_acc))
    max_pitch = float(np.nanmax(np.abs(pitch)))
    sat = float(np.mean(np.any(motors >= 1990.0, axis=1)))

    return {
        "file": path,
        "samples": len(rows),
        "duration_s": float(t[-1]),
        "initial_alt_m": float(truth_u[0]),
        "min_alt_m": float(np.nanmin(truth_u)),
        "drop_m": float(truth_u[0] - np.nanmin(truth_u)),
        "max_alt_m": float(np.nanmax(truth_u)),
        "final_alt_m": float(truth_u[-1]),
        "max_horiz_err_m": float(np.nanmax(horiz_err)),
        "rms_horiz_err_m": float(np.sqrt(np.nanmean(horiz_err**2))),
        "final_horiz_err_m": float(horiz_err[-1]),
        "max_speed_mps": float(np.nanmax(speed)),
        "min_vertical_accel_mps2": min_accel,
        "max_abs_pitch_deg": max_pitch,
        "motor_saturation_fraction": sat,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="多速度跟踪 CSV 汇总")
    parser.add_argument("csvs", nargs="+", help="speed_*.csv 路径（支持 glob）")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    paths = []
    for item in args.csvs:
        paths.extend(glob.glob(item))
    paths = sorted(set(paths))

    results = [summarize(path) for path in paths]
    print("file,samples,drop_m,min_alt_m,max_horiz_err_m,rms_horiz_err_m,"
          "max_speed_mps,min_accel_u,max_pitch_deg,sat_frac")
    for r in results:
        print(
            f"{os.path.basename(r['file'])},{r['samples']},"
            f"{r['drop_m']:.3f},{r['min_alt_m']:.3f},"
            f"{r['max_horiz_err_m']:.3f},{r['rms_horiz_err_m']:.3f},"
            f"{r['max_speed_mps']:.3f},{r['min_vertical_accel_mps2']:.3f},"
            f"{r['max_abs_pitch_deg']:.2f},{r['motor_saturation_fraction']:.3f}"
        )
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
