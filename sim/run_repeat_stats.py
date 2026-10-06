#!/usr/bin/env python3
"""Automated repeated SITL altitude-drop statistics.

For each (cruise speed, ap_altitude_lift_ff) combination:
  * start a fresh headless SITL with the requested feedforward config;
  * run sim/run_altitude_diag.py with a settle phase;
  * kill SITL;
  * parse the truth height trace and record initial altitude, minimum altitude,
    drop, minimum vertical acceleration and mixer saturation.

The final JSON/CSV contains mean/std over the repeats so before/after
feedforward comparisons are not based on a single noisy run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover
    savgol_filter = None

ROOT = Path(__file__).resolve().parent.parent
LAUNCH = ROOT / "sim" / "launch_sitl.sh"
DIAG = ROOT / "sim" / "run_altitude_diag.py"
BF_CONFIG = (
    Path.home()
    / "betaflight"
    / "src"
    / "platform"
    / "SIMULATOR"
    / "gazebo"
    / "config"
    / "sitl_gazebo_config.txt"
)
DEFAULT_LOGDIR = Path("/tmp/bf_anti_repeat")


def _replace_setting(text: str, name: str, value: str) -> str:
    pattern = re.compile(rf"^set {re.escape(name)} = .*$", re.MULTILINE)
    replacement = f"set {name} = {value}"
    if pattern.search(text):
        return pattern.sub(replacement, text)
    return text.rstrip() + f"\n{replacement}\n"


def make_config(ff: int, path: Path) -> Path:
    text = BF_CONFIG.read_text(encoding="utf-8")
    text = _replace_setting(text, "ap_altitude_lift_ff", str(int(ff)))
    text = _replace_setting(text, "debug_mode", "AUTOPILOT_ALTITUDE")
    path.write_text(text, encoding="utf-8")
    return path


def _port_open(host: str = "127.0.0.1", port: int = 5761) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def start_sitl(config: Path, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["BF_ANTI_CONFIG"] = str(config)
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(LAUNCH), "--headless"],
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Attach the file object so it stays alive for the process lifetime.
    proc._bf_log = log  # type: ignore[attr-defined]
    return proc


def stop_sitl(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    log = getattr(proc, "_bf_log", None)
    if log is not None:
        try:
            log.close()
        except Exception:
            pass
    # Safety net for orphaned simulator processes.
    subprocess.run(["pkill", "-f", "betaflight_SITL_SITL_GAZEBO.elf"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "gz sim.*bf_anti_intercept"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait_for_sitl(proc: subprocess.Popen, timeout_s: float = 35.0) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if proc.poll() is not None:
            return False
        if _port_open():
            # Give Gazebo/BetaflightPlugin a moment to establish the FDM stream.
            time.sleep(3.0)
            return True
        time.sleep(0.5)
    return False


def run_diag(cruise: float, duration: float, settle: float, beyond: float,
             log_path: Path) -> int:
    cmd = [
        sys.executable, str(DIAG),
        "--cruise", str(cruise),
        "--duration", str(duration),
        "--settle", str(settle),
        "--beyond", str(beyond),
        "--log", str(log_path),
    ]
    timeout = duration + settle + 150.0
    try:
        res = subprocess.run(cmd, cwd=str(ROOT), timeout=timeout,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        return res.returncode
    except subprocess.TimeoutExpired:
        return 124


def analyse_log(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return None
    if len(rows) < 5:
        return None

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) for r in rows], dtype=float)

    t = col("t")
    truth_u = col("truth_u")
    if np.all(~np.isfinite(truth_u)):
        return None
    # Initial altitude: median of the first five samples.
    n0 = min(5, truth_u.size)
    initial = float(np.median(truth_u[:n0]))
    min_alt = float(np.nanmin(truth_u))
    drop = initial - min_alt

    # Uniform resample, then two smooth derivatives for truth vertical
    # acceleration.  A 0.6 s Savitzky-Golay window suppresses telemetry
    # quantisation spikes that otherwise dominate the second derivative.
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        dt = 0.08
    grid = np.arange(float(t[0]), float(t[-1]), dt)
    try:
        tu = np.interp(grid, t, truth_u)
    except Exception:
        return None
    win = max(5, int(round(0.6 / dt)))
    if win % 2 == 0:
        win += 1
    if savgol_filter is not None and tu.size >= win:
        tu = savgol_filter(tu, win, 2)
    elif tu.size >= 5:
        kernel = np.ones(5) / 5.0
        tu = np.convolve(tu, kernel, mode="same")
    vu = np.gradient(tu, dt)
    au = np.gradient(vu, dt)
    min_accel = float(np.nanmin(au))

    # Mixer differential saturation = max-min motor > 800.
    motor_cols = [f"m{i}" for i in range(4)]
    try:
        motors = np.column_stack([col(name) for name in motor_cols])
        mdif = motors.max(axis=1) - motors.min(axis=1)
        sat_frac = float(np.mean(mdif > 800.0))
    except Exception:
        sat_frac = float("nan")

    return {
        "initial_alt_m": initial,
        "min_alt_m": min_alt,
        "drop_m": drop,
        "min_accel_mps2": min_accel,
        "mixer_saturation_fraction": sat_frac,
        "samples": int(truth_u.size),
    }


def summarize(values: List[float]) -> Dict[str, float]:
    vals = [v for v in values if np.isfinite(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(statistics.mean(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
        "n": len(vals),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Repeated SITL altitude-drop statistics")
    parser.add_argument("--cruise", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument("--ff", type=int, nargs="+", default=[0, 5],
                        help="ap_altitude_lift_ff values")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--settle", type=float, default=5.0)
    parser.add_argument("--beyond", type=float, default=30.0)
    parser.add_argument("--logdir", default=str(DEFAULT_LOGDIR))
    args = parser.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {"runs": [], "summary": {}}

    for ff in args.ff:
        config = make_config(ff, logdir / f"sitl_ff{ff}.txt")
        for cruise in args.cruise:
            key = f"cruise{cruise:g}_ff{ff}"
            runs: List[Dict[str, float]] = []
            for rep in range(1, args.repeat + 1):
                tag = f"{key}_r{rep}"
                sitl_log = logdir / f"{tag}_sitl.log"
                diag_log = logdir / f"{tag}.csv"
                print(f"[repeat] {tag}: starting SITL", flush=True)
                proc = start_sitl(config, sitl_log)
                try:
                    if not wait_for_sitl(proc):
                        print(f"[repeat] {tag}: SITL failed to start", flush=True)
                        continue
                    rc = run_diag(cruise, args.duration, args.settle, args.beyond, diag_log)
                    metrics = analyse_log(diag_log)
                    if metrics is None:
                        print(f"[repeat] {tag}: diagnostic rc={rc}, no valid CSV", flush=True)
                    else:
                        metrics["rc"] = float(rc)
                        metrics["repeat"] = float(rep)
                        metrics["cruise"] = float(cruise)
                        metrics["ff"] = float(ff)
                        runs.append(metrics)
                        print(
                            f"[repeat] {tag}: drop={metrics['drop_m']:.3f} m, "
                            f"min_accel={metrics['min_accel_mps2']:.3f} m/s², "
                            f"sat={metrics['mixer_saturation_fraction']:.3f}",
                            flush=True,
                        )
                finally:
                    stop_sitl(proc)
                    time.sleep(1.0)

            summary["runs"].extend(runs)  # type: ignore[attr-defined]
            summary["summary"][key] = {  # type: ignore[index]
                "drop_m": summarize([r["drop_m"] for r in runs]),
                "min_accel_mps2": summarize([r["min_accel_mps2"] for r in runs]),
                "mixer_saturation_fraction": summarize(
                    [r["mixer_saturation_fraction"] for r in runs]
                ),
            }

    out_json = logdir / "statistics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    out_csv = logdir / "runs.csv"
    fields = [
        "cruise", "ff", "repeat", "initial_alt_m", "min_alt_m", "drop_m",
        "min_accel_mps2", "mixer_saturation_fraction", "samples", "rc",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary["runs"]:  # type: ignore[union-attr]
            writer.writerow({k: row.get(k, "") for k in fields})

    print(json.dumps(summary["summary"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {out_json} and {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
