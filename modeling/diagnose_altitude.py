#!/usr/bin/env python3
"""从 SITL 诊断日志识别耦合模型参数，并给出掉高归因。

输入：``sim/run_offboard_intercept.py`` 风格的 CSV，但需要额外记录
``up``/``vu``（位置估计）、``truth_u``（Gazebo 真值高度）、``pitch``、``m0..m3``。
仓库里的诊断脚本 ``/tmp/bfanti_motor_diag2.py`` 会输出这些列。

输出：
* 固件高度估计的等效一阶滞后 ``tau_s``；
* 姿态内环振铃主频 ``ring_hz``；
* 混控差动饱和占比；
* 倾角重力损失量级与实测垂向加速度的对比；
* 四个耦合模型场景的指标。
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Dict, Optional

import numpy as np

try:
    from scipy.signal import savgol_filter, welch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 scipy: pip install scipy") from exc

from .coupled_altitude import CASES, case_params, simulate
from .identify_lift_loss import estimate_lift_loss


# --------------------------------------------------------------------------- #
# 读取与识别
# --------------------------------------------------------------------------- #
def _odd_window(n: int, fraction: float) -> int:
    """返回小于 n 的奇数窗口，避免不同长度日志触发 savgol 报错。"""
    if n < 3:
        return 3
    w = max(3, int(n * fraction))
    if w % 2 == 0:
        w += 1
    return min(w, n if n % 2 == 1 else n - 1)


def load_log(path: str) -> Dict[str, np.ndarray]:
    """读取诊断 CSV，返回列名到数组的映射。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty log: {path}")

    cols = rows[0].keys()
    required = {"t", "up", "vu", "pitch", "truth_u", "m0", "m1", "m2", "m3"}
    missing = required - set(cols)
    if missing:
        raise ValueError(f"log missing columns: {sorted(missing)}")

    out: Dict[str, np.ndarray] = {}
    for name in cols:
        try:
            out[name] = np.array([float(r[name]) for r in rows], dtype=float)
        except (TypeError, ValueError):
            pass
    return out


def identify(log: Dict[str, np.ndarray]) -> Dict[str, float]:
    """从日志识别关键参数。"""
    required = ("t", "up", "truth_u", "pitch", "m0", "m1", "m2", "m3")
    missing = set(required) - set(log)
    if missing:
        raise ValueError(f"log missing columns: {sorted(missing)}")
    names = required + (("roll",) if "roll" in log else ())
    arrays = {name: np.asarray(log[name], dtype=float) for name in names}
    n = arrays["t"].size
    if any(a.ndim != 1 or a.size != n for a in arrays.values()):
        raise ValueError("diagnostic columns must be equally sized 1D arrays")
    valid = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    dropped = int(n - np.count_nonzero(valid))
    clean = {name: a[valid] for name, a in arrays.items()}
    if clean["t"].size < 3:
        raise ValueError("need at least 3 finite diagnostic samples")
    if np.any(np.diff(clean["t"]) <= 0.0):
        raise ValueError("diagnostic timestamps must be strictly increasing")
    log = clean
    t = log["t"]
    up = log["up"]
    truth = log["truth_u"]
    pitch = log["pitch"]
    mot = np.column_stack([log["m0"], log["m1"], log["m2"], log["m3"]])
    mdif = mot.max(axis=1) - mot.min(axis=1)
    mavg = mot.mean(axis=1)

    # 均匀重采样，便于频域和差分
    dt_nom = float(np.median(np.diff(t)))
    sample_count = max(3, int(round((t[-1] - t[0]) / dt_nom)) + 1)
    tu = np.linspace(float(t[0]), float(t[-1]), sample_count)
    dt_grid = float(tu[1] - tu[0])

    def rs(x: np.ndarray) -> np.ndarray:
        return np.interp(tu, t, x)

    up_u = rs(up)
    truth_u = rs(truth)
    pitch_u = rs(pitch)
    roll_u = rs(log.get("roll", np.zeros_like(pitch)))

    # --- 高度估计一阶滞后：up_dot = (truth - up)/tau ---
    w_up = _odd_window(len(up_u), 0.2)
    up_dot = np.gradient(savgol_filter(up_u, w_up, 2), tu) if len(up_u) >= 3 else np.zeros_like(up_u)
    err = truth_u - up_u
    mask = np.abs(up_dot) > 0.05
    tau_candidates = err[mask] / up_dot[mask]
    tau_candidates = tau_candidates[np.isfinite(tau_candidates)]
    tau_est = float(np.median(np.clip(tau_candidates, 0.0, 3.0))) if tau_candidates.size else 0.0

    # --- 姿态内环振铃主频 ---
    pitch_det = pitch_u - savgol_filter(pitch_u, _odd_window(len(pitch_u), 0.2), 2)
    fs = 1.0 / dt_grid
    f, pxx = welch(pitch_det, fs=fs, nperseg=min(512, len(pitch_det)))
    band = (f > 0.8) & (f < 3.0)
    ring_hz = float(f[band][np.argmax(pxx[band])]) if np.any(band) and np.max(pxx[band]) > 1e-12 else 0.0

    # --- 倾角重力损失 vs 实测垂向运动 ---
    g = 9.80665
    cos_tilt = np.cos(np.radians(roll_u)) * np.cos(np.radians(pitch_u))
    theta_deg = np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))
    max_tilt = float(np.max(theta_deg)) if theta_deg.size else 0.0
    gravity_deficit_max = float(g * (1.0 - np.cos(np.radians(max_tilt))))

    # 注意：高度的一阶导数是垂速，不是垂向加速度。原实现只做了一次
    # np.gradient，导致 min_truth_accel_mps2 实际是“最小垂速”，并把
    # 重力损失占比算错。这里对平滑后的高度先求垂速，再求垂向加速度。
    w_truth = _odd_window(len(truth_u), 0.2)
    truth_vel = np.gradient(savgol_filter(truth_u, w_truth, 2), tu)
    truth_acc = np.gradient(savgol_filter(truth_vel, w_truth, 2), tu)
    min_vel = float(np.min(truth_vel)) if truth_vel.size else 0.0
    min_acc = float(np.min(truth_acc)) if truth_acc.size else 0.0

    # --- 混控差动饱和统计 ---
    sat_frac = float(np.mean(mdif > 800.0))
    normal = mdif < 200.0
    saturated = mdif > 800.0
    mavg_normal = float(np.mean(mavg[normal])) if np.any(normal) else 0.0
    mavg_saturated = float(np.mean(mavg[saturated])) if np.any(saturated) else 0.0

    return {
        "samples_used": int(t.size),
        "samples_dropped_nonfinite": dropped,
        "sample_period_median_s": dt_nom,
        "sample_period_max_s": float(np.max(np.diff(t))),
        "sample_rate_hz": 1.0 / dt_nom,
        "truth_altitude_drop_m": float(max(0.0, truth[0] - np.min(truth))),
        "altitude_estimation_rmse_m": float(np.sqrt(np.mean((up - truth) ** 2))),
        "roll_available": bool("roll" in log),
        "tau_s": tau_est,
        "ring_hz": ring_hz,
        "max_tilt_deg": max_tilt,
        "gravity_deficit_max_mps2": gravity_deficit_max,
        "min_truth_vel_mps": min_vel,
        "min_truth_accel_mps2": min_acc,
        # 这个比例是“最大倾角纯重力损失 / 实测最大向下加速度”，
        # 只能说明重力项相对大小，不是严格的推力缺失占比。
        "gravity_fraction_of_peak_downward_accel": (
            gravity_deficit_max / abs(min_acc) if min_acc < -1e-6 else 0.0
        ),
        # 保留旧键名，避免已有外部脚本立刻失效；含义同上。
        "gravity_fraction_of_missing_thrust": (
            gravity_deficit_max / abs(min_acc) if min_acc < -1e-6 else 0.0
        ),
        "mixer_saturation_fraction": sat_frac,
        "mean_motor_normal": mavg_normal,
        "mean_motor_saturated": mavg_saturated,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(log_path: str, duration: float = 20.0) -> Dict[str, object]:
    log = load_log(log_path)
    identified = identify(log)
    cases = {
        case: simulate(case_params(case), duration=duration).metrics()
        for case in CASES
    }

    # 当前控制模型预测的最恶劣向下加速度与实测之差，用来量化
    # 未建模的 plant 气动损失（LiftDrag）或其它外部扰动。
    current_min_accel = float(cases.get("current", {}).get("min_accel_u_mps2", 0.0))
    measured_min_accel = float(identified.get("min_truth_accel_mps2", 0.0))
    identified["unmodeled_downward_accel_mps2"] = measured_min_accel - current_min_accel

    try:
        lift_loss = estimate_lift_loss(log).to_dict()
    except Exception as exc:  # pragma: no cover - 短日志/缺列时给出诊断
        lift_loss = {"error": str(exc)}

    return {
        "log": log_path,
        "identified": identified,
        "lift_loss_identification": lift_loss,
        "cases": cases,
        "note": (
            "min_truth_accel_mps2 由真值高度两次平滑微分得到；"
            "unmodeled_downward_accel_mps2 = 实测最恶劣垂向加速度 - current "
            "控制模型预测值。lift_loss_identification.k_v 是稳态水平段"
            "按 a_pred = k_v * v^2 辨识出的 LiftDrag 升力损失系数。"
        ),
    }


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="从 SITL 日志诊断水平掉高")
    parser.add_argument("--log", required=True, help="诊断 CSV 路径")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output", default="", help="可选 JSON 输出")
    args = parser.parse_args(argv)

    result = run(args.log, duration=args.duration)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
