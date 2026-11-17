#!/usr/bin/env python3
"""Paired headless Gazebo + real-YOLO trial matrix.

Runs P and/or PN through sim/trial_p_strike.sh with a fixed target pose for
each case. All scored trials use the remote GPU1 detector at :9300; synthetic
bbox is never used here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CASES = {
    "static":       {"x": 45.0, "y": 0.0, "z": 12.0},
    "offset_left":  {"x": 45.0, "y": -8.0, "z": 12.0},   # Gazebo y=-E; N=+8
    "offset_right": {"x": 45.0, "y": 8.0, "z": 12.0},
    "high":         {"x": 40.0, "y": 0.0, "z": 16.0},
    # true lateral crossing: Gazebo y=-E, so vel_e moves the target across
    # the interceptor's northbound line of sight
    "crossing":     {"x": 45.0, "y": 0.0, "z": 12.0, "vel_e": 3.5,
                     "start_delay_s": 4.0},
    "recede":       {"x": 45.0, "y": 0.0, "z": 12.0, "vel_n": 3.5,
                     "start_delay_s": 4.0},
}


def yolo_ok(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def row_metrics(path: Path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return {"error": "empty csv"}
    def f(key):
        out = []
        for r in rows:
            try:
                out.append(float(r.get(key, "nan")))
            except (TypeError, ValueError):
                out.append(float("nan"))
        return out
    t = f("t")
    tr = f("truth_range")
    pitch = f("truth_pitch")
    roll = f("truth_roll")
    spd = [math.hypot(a, b) for a, b in zip(f("own_ve"), f("own_vn"))]
    good_tr = [(i, v) for i, v in enumerate(tr) if math.isfinite(v)]
    imin = min(good_tr, key=lambda kv: kv[1])[0]
    imin_t = t[imin]
    # terminal window: 0.5 s before CPA (or last available)
    term = [i for i, tt in enumerate(t) if imin_t - 0.5 <= tt <= imin_t + 0.05]
    term = [i for i in term if math.isfinite(pitch[i])]
    rates = []
    for a, b in zip(term, term[1:]):
        dt = t[b] - t[a]
        if dt > 1e-6:
            rates.append((pitch[b] - pitch[a]) / dt)
    track = sum(1 for r in rows if r.get("phase") in ("TRACK", "TERMINAL", "COAST"))
    los = [v for v in f("truth_los_err") if math.isfinite(v)]
    los_track = [v for r, v in zip(rows, f("truth_los_err"))
                 if r.get("phase") in ("TRACK", "TERMINAL") and math.isfinite(v)]
    return {
        "samples": len(rows),
        "duration": max(t) if t else 0.0,
        "min_truth_range": min(good_tr, key=lambda kv: kv[1])[1],
        "hit": bool(min(good_tr, key=lambda kv: kv[1])[1] <= 0.8),
        "lock_time": next((tt for r, tt in zip(rows, t) if r.get("phase") == "TRACK"), None),
        "track_fraction": track / len(rows),
        "max_speed": max(spd) if spd else float("nan"),
        "terminal_pitch": pitch[imin] if term else float("nan"),
        "terminal_roll_range": (max(roll[i] for i in term) - min(roll[i] for i in term))
                               if term else float("nan"),
        "terminal_pitch_rate_p90": (sorted(abs(v) for v in rates)[int(0.9 * (len(rates) - 1))]
                                    if rates else float("nan")),
        "los_rms_track": (math.sqrt(sum(v * v for v in los_track) / len(los_track))
                          if los_track else float("nan")),
        "los_p90_track": (sorted(los_track)[int(0.9 * (len(los_track) - 1))]
                          if los_track else float("nan")),
        "los_all_max": max(los) if los else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "sim/results/controller_bench_2026-11-05"))
    ap.add_argument("--modes", default="p,pn")
    ap.add_argument("--cases", default="static,offset_left,offset_right,high,crossing")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9300)
    ap.add_argument("--extra", default="", help="extra runner args, shell-quoted")
    ap.add_argument("--timeout", type=float, default=260.0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    extra = shlex.split(args.extra)
    if not yolo_ok(args.host, args.port):
        print(f"ABORT: remote YOLO {args.host}:{args.port} unreachable", flush=True)
        return 2
    records = []
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "modes": modes,
            "cases": cases, "extra": extra, "port": args.port,
            "gpu_expected": "GPU-40c96051-47f2-ac23-7d8d-ef868990967c"}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for mode in modes:
        for case in cases:
            for rep in range(args.repeats):
                name = f"{mode}_{case}_r{rep}"
                log_csv = out / f"{name}.csv"
                console = out / f"{name}.log"
                env = os.environ.copy()
                env.update({
                    "CONTROLLER": mode,
                    "ATT_MODE": "rate",
                    "YOLO_BACKEND": "pt",
                    "LOG": str(log_csv),
                    "TARGET_X": str(CASES[case]["x"]),
                    "TARGET_Y": str(CASES[case]["y"]),
                    "TARGET_Z": str(CASES[case]["z"]),
                })
                env.pop("TARGET_VEL_E", None)
                env.pop("TARGET_VEL_N", None)
                if "vel_e" in CASES[case]:
                    env["TARGET_VEL_E"] = str(CASES[case]["vel_e"])
                if "vel_n" in CASES[case]:
                    env["TARGET_VEL_N"] = str(CASES[case]["vel_n"])
                if "start_delay_s" in CASES[case]:
                    env["TARGET_START_DELAY_S"] = str(CASES[case]["start_delay_s"])
                cmd = ["bash", "sim/trial_p_strike.sh", name] + extra
                print(f"[matrix] {name} ...", flush=True)
                t0 = time.time()
                rec = {"name": name, "mode": mode, "case": case, "rep": rep,
                       "elapsed_s": None, "returncode": None}
                try:
                    proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                                          stdout=open(console, "w", encoding="utf-8"),
                                          stderr=subprocess.STDOUT, timeout=args.timeout)
                    rec["returncode"] = proc.returncode
                except subprocess.TimeoutExpired as exc:
                    rec["returncode"] = "timeout"
                    rec["error"] = str(exc)
                    subprocess.run(["bash", "sim/sim_restart.sh", "stop"], cwd=str(ROOT),
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                rec["elapsed_s"] = round(time.time() - t0, 1)
                if log_csv.exists():
                    try:
                        rec.update(row_metrics(log_csv))
                    except Exception as exc:  # noqa: BLE001
                        rec["error"] = f"metrics: {exc}"
                else:
                    rec["error"] = "missing csv"
                print(f"[matrix] {name} rc={rec['returncode']} "
                      f"min_r={rec.get('min_truth_range', float('nan')):.3f} "
                      f"hit={rec.get('hit')} track={rec.get('track_fraction', 0):.2f}",
                      flush=True)
                records.append(rec)
                (out / "summary.json").write_text(
                    json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
