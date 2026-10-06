"""从 SITL 日志辨识 LiftDrag 速度相关升力损失系数 ``k_v``。

模型
----
把前飞升力损失等效成额外的向下加速度：

.. math::

    a_{u,measured} = a_{u,thrust} - k_v v^2

其中 ``a_u,thrust`` 由高度环实际输出的总距和姿态投影得到：

.. math::

    a_{u,thrust} = g \\frac{u}{u_{hover}} \\cos\\phi\\cos\\theta - g

在稳态水平段，高度 PID 已经把 ``k_v v²`` 配平，所以指令总距对应的
``a_pred = a_u,thrust`` 本身约等于 ``k_v v²``。本模块优先使用
``MSP_DEBUG=AUTOPILOT_ALTITUDE`` 的 ``dbg0``（实际总距 PWM）和估计姿态
``roll/pitch`` 做基线，避免平均电机差动、真值 roll 环绕等污染。

输出：
* ``k_v``：通过原点的最小二乘估计；
* ``k_v_intercept`` / ``bias``：带截距的线性回归，用于判断模型偏差；
* ``r2``、样本数、速度范围。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

GRAVITY_MSS = 9.80665
HOVER_THROTTLE_PWM = 1530.0
PWM_RANGE_MIN = 1000.0
PWM_RANGE_MAX = 2000.0


@dataclass
class LiftLossIdentification:
    k_v: float
    k_v_intercept: float
    bias: float
    r2: float
    n: int
    speed_min_mps: float
    speed_max_mps: float
    source: str

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class LiftLossRobustIdentification:
    k_v_huber: float
    k_v_intercept_huber: float
    bias_huber: float
    r2_huber: float
    n: int
    n_logs: int
    speed_min_mps: float
    speed_max_mps: float
    per_log: list[dict]
    method: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class LiftLossResidualIdentification:
    k_v_huber: float
    k_v_intercept_huber: float
    bias_huber: float
    r2_huber: float
    k_v_se: float
    k_v_ci95_low: float
    k_v_ci95_high: float
    n: int
    n_logs: int
    speed_min_mps: float
    speed_max_mps: float
    per_log: list[dict]
    method: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _resample(log: Dict[str, np.ndarray], names: tuple[str, ...]) -> tuple[np.ndarray, ...]:
    t = np.asarray(log["t"], dtype=float)
    if t.size < 3:
        raise ValueError("log too short")
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("invalid sample time")
    grid = np.arange(float(t[0]), float(t[-1]), dt)
    out = []
    for name in names:
        arr = np.asarray(log[name], dtype=float)
        out.append(np.interp(grid, t, arr))
    return tuple(out)


def _smooth(x: np.ndarray, dt_raw: float, tau_s: float = 0.5) -> np.ndarray:
    win = max(3, int(round(tau_s / dt_raw)))
    if win % 2 == 0:
        win += 1
    if x.size < win or win < 3:
        return x
    return savgol_filter(x, win, 2)


def _signals(
    log: Dict[str, np.ndarray],
    hover_throttle_pwm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """提取平滑后的 ``(speed, a_pred, a_measured_u, source)``。"""
    t_raw = np.asarray(log["t"], dtype=float)
    dt_raw = float(np.median(np.diff(t_raw)))
    if not np.isfinite(dt_raw) or dt_raw <= 0.0:
        raise ValueError("invalid sample time")

    # 水平速度优先用真值位置差分；没有真值位置时退回估计速度。
    if {"truth_e", "truth_n"} <= set(log):
        te, tn, tu = _resample(log, ("truth_e", "truth_n", "truth_u"))
        ve, vn = np.gradient(te, dt_raw), np.gradient(tn, dt_raw)
        vu = np.gradient(tu, dt_raw)
        au = np.gradient(vu, dt_raw)
        source_pos = "truth"
    elif {"ve", "vn", "up"} <= set(log):
        # 估计速度单位为 cm/s，统一换算到 m/s。
        ve, vn, up = _resample(log, ("ve", "vn", "up"))
        ve, vn = ve * 0.01, vn * 0.01
        au = np.gradient(np.gradient(up, dt_raw), dt_raw)
        source_pos = "estimate"
    else:
        raise ValueError("log needs truth_e/truth_n/truth_u or ve/vn/up")

    speed = _smooth(np.sqrt(ve * ve + vn * vn), dt_raw)
    au = _smooth(au, dt_raw)

    # 姿态优先用估计 roll/pitch（MSP_ATTITUDE，数值干净且不环绕）；
    # 没有时才用真值。
    if {"roll", "pitch"} <= set(log):
        roll, pitch = _resample(log, ("roll", "pitch"))
        source_att = "msp_attitude"
    elif {"truth_roll", "truth_pitch"} <= set(log):
        roll, pitch = _resample(log, ("truth_roll", "truth_pitch"))
        source_att = "truth_rpy_may_wrap"
    else:
        raise ValueError("log needs roll/pitch or truth_roll/truth_pitch")

    tilt_cos = np.cos(np.radians(roll)) * np.cos(np.radians(pitch))

    # 基线优先用高度环实际输出的总距 dbg0（PWM）；否则用平均电机。
    if "dbg0" in log:
        dbg0 = _resample(log, ("dbg0",))[0]
        valid = dbg0 > PWM_RANGE_MIN
        if np.count_nonzero(valid) > 10:
            throttle_pwm = dbg0
            source_throttle = "debug_altitude_throttle"
        else:
            throttle_pwm = None
    else:
        throttle_pwm = None

    if throttle_pwm is None:
        motor_names = tuple(f"m{i}" for i in range(4))
        if not set(motor_names) <= set(log):
            raise ValueError("log needs dbg0 or m0..m3")
        motors = np.column_stack([_resample(log, (name,))[0] for name in motor_names])
        throttle_pwm = motors.mean(axis=1)
        source_throttle = "mean_motor_pwm"

    throttle_norm = (throttle_pwm - PWM_RANGE_MIN) / (PWM_RANGE_MAX - PWM_RANGE_MIN)
    hover_norm = (hover_throttle_pwm - PWM_RANGE_MIN) / (PWM_RANGE_MAX - PWM_RANGE_MIN)
    a_pred = GRAVITY_MSS * throttle_norm / hover_norm * tilt_cos - GRAVITY_MSS
    a_pred = _smooth(a_pred, dt_raw)
    return speed, a_pred, au, f"{source_pos}+{source_att}+{source_throttle}"


def _steady_arrays(
    log: Dict[str, np.ndarray],
    min_speed_mps: float,
    accel_tol_mps2: float,
    speed_accel_tol_mps2: float,
    hover_throttle_pwm: float,
) -> tuple[np.ndarray, np.ndarray, int, float, float, str]:
    """提取日志中的稳态速度平台，返回 ``(v2, a_pred, n, vmin, vmax, source)``。"""
    speed_s, a_pred_s, au_s, source = _signals(log, hover_throttle_pwm)
    dt_raw = float(np.median(np.diff(np.asarray(log["t"], dtype=float))))
    speed_accel = np.gradient(speed_s, dt_raw)

    win = max(3, int(round(0.5 / dt_raw)))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    speed_pad = np.pad(speed_s, pad, mode="edge")
    speed_std = np.array(
        [np.std(speed_pad[i : i + win]) for i in range(speed_s.size)],
        dtype=float,
    )

    mask = (
        (speed_s >= min_speed_mps)
        & np.isfinite(a_pred_s)
        & np.isfinite(au_s)
        & (np.abs(au_s) < accel_tol_mps2)
        & (np.abs(speed_accel) < speed_accel_tol_mps2)
        & (speed_std < 0.5)
    )
    if np.count_nonzero(mask) < 3:
        raise ValueError(
            "not enough steady speed plateau samples "
            f"(n={int(np.count_nonzero(mask))}, "
            f"min_speed={min_speed_mps}, accel_tol={accel_tol_mps2}, "
            f"speed_accel_tol={speed_accel_tol_mps2})"
        )

    v2 = speed_s[mask] ** 2
    y = a_pred_s[mask]
    return (
        v2,
        y,
        int(np.count_nonzero(mask)),
        float(speed_s[mask].min()),
        float(speed_s[mask].max()),
        source,
    )


def _residual_arrays(
    log: Dict[str, np.ndarray],
    min_speed_mps: float,
    hover_throttle_pwm: float,
) -> tuple[np.ndarray, np.ndarray, int, float, float, str]:
    """完整模型残差 ``a_pred - a_measured = k_v v²``，返回 ``(v2, y, ...)``。

    与 :func:`_steady_arrays` 不同，这里不要求高度加速度为零；高度振荡、
    执行器滞后等作为噪声由 Huber 抑制。
    """
    speed_s, a_pred_s, au_s, source = _signals(log, hover_throttle_pwm)
    y = a_pred_s - au_s
    mask = (speed_s >= min_speed_mps) & np.isfinite(y) & np.isfinite(speed_s)
    if np.count_nonzero(mask) < 5:
        raise ValueError(
            f"not enough residual samples above {min_speed_mps} m/s "
            f"(n={int(np.count_nonzero(mask))})"
        )
    v2 = speed_s[mask] ** 2
    return (
        v2,
        y[mask],
        int(np.count_nonzero(mask)),
        float(speed_s[mask].min()),
        float(speed_s[mask].max()),
        source,
    )


def estimate_lift_loss(
    log: Dict[str, np.ndarray],
    min_speed_mps: float = 4.0,
    accel_tol_mps2: float = 0.5,
    speed_accel_tol_mps2: float = 1.0,
    hover_throttle_pwm: float = HOVER_THROTTLE_PWM,
) -> LiftLossIdentification:
    """从单条日志估计 ``k_v``（普通最小二乘）。"""
    v2, y, n, vmin, vmax, source = _steady_arrays(
        log,
        min_speed_mps=min_speed_mps,
        accel_tol_mps2=accel_tol_mps2,
        speed_accel_tol_mps2=speed_accel_tol_mps2,
        hover_throttle_pwm=hover_throttle_pwm,
    )
    # 通过原点：k = (v2 · y) / (v2 · v2)
    denom = float(v2 @ v2)
    k_v = float(v2 @ y / denom) if denom > 1e-12 else 0.0
    # 带截距，用于衡量模型偏差。
    x = np.column_stack([np.ones_like(v2), v2])
    coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
    bias, k_intercept = float(coef[0]), float(coef[1])
    pred = x @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 1e-12 else 0.0

    return LiftLossIdentification(
        k_v=max(0.0, k_v),
        k_v_intercept=max(0.0, k_intercept),
        bias=bias,
        r2=r2,
        n=n,
        speed_min_mps=vmin,
        speed_max_mps=vmax,
        source=source,
    )


def estimate_lift_loss_robust(
    logs: list[Dict[str, np.ndarray]],
    names: Optional[list[str]] = None,
    min_speed_mps: float = 4.0,
    accel_tol_mps2: float = 0.5,
    speed_accel_tol_mps2: float = 1.0,
    hover_throttle_pwm: float = HOVER_THROTTLE_PWM,
    f_scale: float = 0.2,
) -> LiftLossRobustIdentification:
    """多日志、多稳态段 Huber 鲁棒辨识。

    每条日志先提取自己的稳态水平段，然后把所有 ``(v², a_pred)`` 样本合并，
    用 Huber 损失抑制高度振荡/瞬态段造成的离群点。返回值同时给出通过原点
    Huber 解和带截距 Huber 解，后者用于判断是否存在模型偏差。
    """
    all_v2: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    per_log: list[dict] = []
    for idx, log in enumerate(logs):
        name = names[idx] if names and idx < len(names) else f"log_{idx}"
        try:
            v2, y, n, vmin, vmax, source = _steady_arrays(
                log,
                min_speed_mps=min_speed_mps,
                accel_tol_mps2=accel_tol_mps2,
                speed_accel_tol_mps2=speed_accel_tol_mps2,
                hover_throttle_pwm=hover_throttle_pwm,
            )
        except ValueError as exc:
            per_log.append({"name": name, "error": str(exc)})
            continue
        all_v2.append(v2)
        all_y.append(y)
        per_log.append(
            {
                "name": name,
                "n": n,
                "speed_min_mps": vmin,
                "speed_max_mps": vmax,
                "source": source,
            }
        )

    if not all_v2:
        raise ValueError("no usable steady segments in any log")

    v2 = np.concatenate(all_v2)
    y = np.concatenate(all_y)
    # 通过原点 Huber 解。
    k0 = float(v2 @ y / (v2 @ v2)) if float(v2 @ v2) > 1e-12 else 0.0
    res_origin = least_squares(
        lambda k: k * v2 - y,
        x0=[k0],
        loss="huber",
        f_scale=f_scale,
    )
    k_huber = max(0.0, float(res_origin.x[0]))
    # 带截距 Huber 解。
    res_intercept = least_squares(
        lambda p: p[0] + p[1] * v2 - y,
        x0=[0.0, k_huber],
        loss="huber",
        f_scale=f_scale,
    )
    bias_huber = float(res_intercept.x[0])
    k_intercept_huber = max(0.0, float(res_intercept.x[1]))
    pred = bias_huber + k_intercept_huber * v2
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 1e-12 else 0.0

    return LiftLossRobustIdentification(
        k_v_huber=k_huber,
        k_v_intercept_huber=k_intercept_huber,
        bias_huber=bias_huber,
        r2_huber=r2,
        n=int(v2.size),
        n_logs=len(all_v2),
        speed_min_mps=float(np.sqrt(v2.min())),
        speed_max_mps=float(np.sqrt(v2.max())),
        per_log=per_log,
        method=f"huber_origin+huber_intercept(f_scale={f_scale})",
    )


def estimate_lift_loss_residual(
    logs: list[Dict[str, np.ndarray]],
    names: Optional[list[str]] = None,
    min_speed_mps: float = 3.0,
    hover_throttle_pwm: float = HOVER_THROTTLE_PWM,
    f_scale: float = 0.3,
    bootstrap: int = 200,
    seed: int = 0,
) -> LiftLossResidualIdentification:
    """完整模型残差 Huber 辨识。

    模型：

    .. math::

        a_{pred} - a_{measured} = k_v v^2

    其中 ``a_pred`` 由高度环总距和姿态投影得到，``a_measured`` 是真值垂向
    加速度。该式不要求高度加速度为零，因此适合 offboard 速度剖面本身有
    振荡的数据；Huber 损失把振荡/执行器滞后当作离群噪声。``bootstrap``
    用于估计 ``k_v`` 的标准误和 95% 置信区间。
    """
    all_v2: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    per_log: list[dict] = []
    for idx, log in enumerate(logs):
        name = names[idx] if names and idx < len(names) else f"log_{idx}"
        try:
            v2, y, n, vmin, vmax, source = _residual_arrays(
                log,
                min_speed_mps=min_speed_mps,
                hover_throttle_pwm=hover_throttle_pwm,
            )
        except ValueError as exc:
            per_log.append({"name": name, "error": str(exc)})
            continue
        all_v2.append(v2)
        all_y.append(y)
        per_log.append(
            {
                "name": name,
                "n": n,
                "speed_min_mps": vmin,
                "speed_max_mps": vmax,
                "source": source,
            }
        )

    if not all_v2:
        raise ValueError("no usable residual samples in any log")

    v2 = np.concatenate(all_v2)
    y = np.concatenate(all_y)
    k0 = float(v2 @ y / (v2 @ v2)) if float(v2 @ v2) > 1e-12 else 0.0
    res_origin = least_squares(
        lambda k: k * v2 - y,
        x0=[k0],
        loss="huber",
        f_scale=f_scale,
    )
    k_huber = max(0.0, float(res_origin.x[0]))
    res_intercept = least_squares(
        lambda p: p[0] + p[1] * v2 - y,
        x0=[0.0, k_huber],
        loss="huber",
        f_scale=f_scale,
    )
    bias_huber = float(res_intercept.x[0])
    k_intercept_huber = max(0.0, float(res_intercept.x[1]))
    pred = bias_huber + k_intercept_huber * v2
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot if ss_tot > 1e-12 else 0.0

    # Bootstrap standard error / 95% CI for the origin-constrained Huber fit.
    k_samples: list[float] = []
    rng = np.random.default_rng(seed)
    n = v2.size
    for _ in range(max(0, bootstrap)):
        idx = rng.integers(0, n, n)
        v2b, yb = v2[idx], y[idx]
        denom = float(v2b @ v2b)
        if denom <= 1e-12:
            continue
        kb0 = float(v2b @ yb / denom)
        try:
            rb = least_squares(
                lambda k: k * v2b - yb,
                x0=[kb0],
                loss="huber",
                f_scale=f_scale,
            )
        except Exception:
            continue
        k_samples.append(max(0.0, float(rb.x[0])))
    if k_samples:
        arr = np.asarray(k_samples)
        k_se = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        lo, hi = np.percentile(arr, [2.5, 97.5])
        k_ci_low, k_ci_high = float(lo), float(hi)
    else:
        k_se = k_ci_low = k_ci_high = float("nan")

    return LiftLossResidualIdentification(
        k_v_huber=k_huber,
        k_v_intercept_huber=k_intercept_huber,
        bias_huber=bias_huber,
        r2_huber=r2,
        k_v_se=k_se,
        k_v_ci95_low=k_ci_low,
        k_v_ci95_high=k_ci_high,
        n=int(v2.size),
        n_logs=len(all_v2),
        speed_min_mps=float(np.sqrt(v2.min())),
        speed_max_mps=float(np.sqrt(v2.max())),
        per_log=per_log,
        method=(
            f"residual_huber(f_scale={f_scale}) + bootstrap({bootstrap})"
        ),
    )


def load_log(path: str) -> Dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty log: {path}")
    out: Dict[str, np.ndarray] = {}
    for name in rows[0].keys():
        try:
            out[name] = np.array([float(r[name]) for r in rows], dtype=float)
        except (TypeError, ValueError):
            pass
    return out


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="辨识 LiftDrag 速度相关升力损失 k_v")
    parser.add_argument("--log", required=True, nargs="+", help="一个或多个诊断 CSV")
    parser.add_argument("--method", choices=("residual", "steady", "single"),
                        default="residual",
                        help="residual=完整模型残差 Huber；steady=稳态段；single=单日志 OLS")
    parser.add_argument("--min-speed", type=float, default=4.0)
    parser.add_argument("--accel-tol", type=float, default=0.5)
    parser.add_argument("--speed-accel-tol", type=float, default=1.0)
    parser.add_argument("--huber-scale", type=float, default=0.3)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    logs = [load_log(path) for path in args.log]
    if args.method == "single":
        result: Dict[str, object] = estimate_lift_loss(
            logs[0],
            min_speed_mps=args.min_speed,
            accel_tol_mps2=args.accel_tol,
            speed_accel_tol_mps2=args.speed_accel_tol,
        ).to_dict()
    elif args.method == "steady":
        result = estimate_lift_loss_robust(
            logs,
            names=list(args.log),
            min_speed_mps=args.min_speed,
            accel_tol_mps2=args.accel_tol,
            speed_accel_tol_mps2=args.speed_accel_tol,
            f_scale=args.huber_scale,
        ).to_dict()
    else:
        result = estimate_lift_loss_residual(
            logs,
            names=list(args.log),
            min_speed_mps=args.min_speed,
            f_scale=args.huber_scale,
            bootstrap=args.bootstrap,
        ).to_dict()

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
