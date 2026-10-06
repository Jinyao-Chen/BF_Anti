"""级联 PID 理论整定与线性评估（Simulink 替代工具链的核心）。

设计思路
--------
不直接对像素/RC 层盲调，而是先把被控对象抽象成悬停点线性模型，按带宽分离
原则独立设计每一级：

* 角速率环：``1/(I*s)``，PI
* 姿态角环：``1/s``，PD/PID（内环近似 1）
* 速度环：``1/s``，PI
* 位置环：``1/s``，P
* 高度环：``1/s²``，PID

所有回路都采用标准二阶/三阶极点配置，保证阻尼比、带宽和相位裕度可解释。
输出为 JSON，既可用于 Gazebo/SITL 调参，也可作为专利中的“模型自动整定”
技术证据。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from scipy import signal

try:  # 允许 ``python modeling/tune.py`` 直接运行
    from modeling.pid import PIDGains
    from modeling.quadrotor import QuadrotorParams
except ImportError:  # pragma: no cover
    from pid import PIDGains  # type: ignore
    from quadrotor import QuadrotorParams  # type: ignore


def _pid_tf(gains: PIDGains) -> signal.TransferFunction:
    """并联 PID 的传递函数 ``C(s) = kp + ki/s + kd*s``。"""
    return signal.TransferFunction([gains.kd, gains.kp, gains.ki], [1.0, 0.0])


def _pd_tf(gains: PIDGains) -> signal.TransferFunction:
    return signal.TransferFunction([gains.kd, gains.kp], [1.0])


def _poly_add(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = max(a.size, b.size)
    out = np.zeros(n, dtype=float)
    out[-a.size :] += a
    out[-b.size :] += b
    return out


def _closed_loop(plant: signal.TransferFunction, controller: signal.TransferFunction) -> signal.TransferFunction:
    """``T = C*P / (1 + C*P)``，只使用多项式运算，兼容 scipy 1.8。"""
    num = np.convolve(controller.num, plant.num)
    den = np.convolve(controller.den, plant.den)
    # 1 + C*P
    close_den = _poly_add(den, num)
    return signal.TransferFunction(num, close_den)


def _metrics(system: signal.TransferFunction, duration: float = 8.0) -> Dict[str, float]:
    """阶跃响应指标：超调、上升时间、调节时间、ITAE。"""
    poles = np.roots(system.den) if system.den.size > 1 else np.array([])
    if poles.size and np.max(np.real(poles)) > 1e-6:
        return {
            "stable": 0.0,
            "overshoot_pct": 1e6,
            "rise_time_s": 1e6,
            "settling_time_s": 1e6,
            "itae": 1e6,
            "steady_state": 1e6,
        }
    try:
        t, y = signal.step(system, T=np.linspace(0.0, duration, max(200, int(duration * 200))))
    except Exception:
        return {
            "stable": 0.0,
            "overshoot_pct": 1e6,
            "rise_time_s": 1e6,
            "settling_time_s": 1e6,
            "itae": 1e6,
            "steady_state": 1e6,
        }
    y = np.asarray(y, dtype=float).reshape(-1)
    final = float(y[-1]) if y.size else 0.0
    peak = float(np.max(y)) if y.size else 0.0
    overshoot = max(0.0, (peak - final) / max(abs(final), 1e-9) * 100.0) if final != 0.0 else 0.0
    target = final
    band = 0.02 * max(abs(target), 1e-9)
    idx = np.where(np.abs(y - target) <= band)[0]
    settling = float(t[idx[0]]) if idx.size else float(duration)
    # 上升时间：10% -> 90%
    lo = np.where(y >= 0.1 * target)[0]
    hi = np.where(y >= 0.9 * target)[0]
    rise = float(t[hi[0]] - t[lo[0]]) if lo.size and hi.size else float(duration)
    err = 1.0 - y
    itae = float(np.trapz(t * np.abs(err), t))
    return {
        "stable": 1.0,
        "overshoot_pct": float(overshoot),
        "rise_time_s": float(rise),
        "settling_time_s": float(settling),
        "itae": itae,
        "steady_state": float(y[-1] if y.size else 0.0),
    }


def _margins(plant: signal.TransferFunction, controller: signal.TransferFunction) -> Dict[str, float]:
    """开环 C*P 的幅值/相位裕度。"""
    loop = signal.TransferFunction(
        np.convolve(controller.num, plant.num),
        np.convolve(controller.den, plant.den),
    )
    try:
        gm, pm, wcg, wcp = signal.margin(loop)
    except Exception:
        return {
            "gain_margin_db": 0.0,
            "phase_margin_deg": 0.0,
            "gain_crossover_rad_s": 0.0,
            "margins_valid": 0.0,
        }
    valid = 0.0
    gm_db = 0.0
    pm_deg = 0.0
    wcp_rad = 0.0
    if gm is not None and not np.isnan(gm) and gm > 0.0:
        gm_db = float("inf") if np.isinf(gm) else float(20.0 * np.log10(gm))
        valid = 1.0
    if pm is not None and not np.isnan(pm):
        pm_deg = float(pm)
        valid = 1.0
    if wcp is not None and not np.isnan(wcp):
        wcp_rad = float(wcp)
    return {
        "gain_margin_db": gm_db,
        "phase_margin_deg": pm_deg,
        "gain_crossover_rad_s": wcp_rad,
        "margins_valid": valid,
    }


@dataclass
class CascadeGains:
    """一套理论整定得到的级联 PID 参数。"""

    rate: PIDGains
    attitude: PIDGains
    velocity: PIDGains
    position: PIDGains
    altitude: PIDGains
    bandwidths: Dict[str, float] = field(default_factory=dict)
    damping: float = 0.9
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    betaflight_hint: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "rate": asdict(self.rate),
            "attitude": asdict(self.attitude),
            "velocity": asdict(self.velocity),
            "position": asdict(self.position),
            "altitude": asdict(self.altitude),
            "bandwidths": self.bandwidths,
            "damping": self.damping,
            "metrics": self.metrics,
            "betaflight_hint": self.betaflight_hint,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _design_pi(plant_gain: float, bandwidth: float, damping: float) -> PIDGains:
    """对象 ``plant_gain / s``，PI 闭环特征方程 ``s² + 2ζωn s + ωn²``。

    开环 ``C*P = (kp + ki/s) * K/s``，特征方程 ``s² + K*kp*s + K*ki = 0``，
    因此 ``kp = 2ζωn/K``，``ki = ωn²/K``。
    """
    wn = float(bandwidth)
    gain = max(abs(float(plant_gain)), 1e-12)
    return PIDGains(kp=2.0 * damping * wn / gain, ki=wn * wn / gain)


def _design_pd(plant_gain: float, bandwidth: float, damping: float) -> PIDGains:
    """对象 ``plant_gain / s²``，PD 闭环 ``s² + 2ζωn s + ωn²``。"""
    wn = float(bandwidth)
    return PIDGains(kp=wn * wn / plant_gain, ki=0.0, kd=2.0 * damping * wn / plant_gain)


def _design_altitude_cascade(
    velocity_bandwidth: float,
    position_bandwidth: float,
    damping: float,
) -> PIDGains:
    """高度串级设计（归一化增益，不直接等于固件内部增益）。

    这里按“位置 P -> 期望垂速 -> 速度 PI”的串级对象设计，得到的是
    ``kp [1/s]``、``ki [1/s²]``、``kd [1/s]`` 的归一化量。它与
    Betaflight ``altitudeControl()`` 的 **P/I 直接作用于高度误差** 结构
    不同，因此不能再用 ``kp*100`` 之类的比例直接写 ``ap_altitude_*``。

    固件直写值请使用 :mod:`modeling.altitude_loop`：
    ``python3 -m modeling.altitude_loop --sitl``。

    内环对象为 ``1/s``（加速度积分成速度），取 ``kd = 2ζω_v``、``ki = ω_v²``；
    外环取 ``kp = ω_p``，且 ``ω_p <= ω_v/3`` 保证带宽分离。
    """
    wv = float(velocity_bandwidth)
    wp = min(float(position_bandwidth), wv / 3.0)
    return PIDGains(kp=wp, ki=wv * wv, kd=2.0 * damping * wv)


def design_cascade(
    params: Optional[QuadrotorParams] = None,
    rate_bandwidth: float = 45.0,
    attitude_bandwidth: float = 8.0,
    velocity_bandwidth: float = 3.0,
    position_bandwidth: float = 1.0,
    altitude_bandwidth: float = 1.2,
    damping: float = 0.9,
) -> CascadeGains:
    """按带宽分离原则给出一套可解释的级联 PID 初始值。

    带宽单位 rad/s。选取原则是内环至少比外环快 4~6 倍。
    """
    p = params if params is not None else QuadrotorParams()

    # 角速率环：对象 1/(I s)，PI。
    rate = _design_pi(1.0 / p.ixx, rate_bandwidth, damping)

    # 姿态角环：内环近似 1，对象 1/s，PD（积分交给配平，避免慢积分振荡）。
    attitude = _design_pd(1.0, attitude_bandwidth, damping)

    # 水平速度环：加速度 -> 速度，对象 1/s，PI。
    velocity = _design_pi(1.0, velocity_bandwidth, damping)

    # 位置环：速度 -> 位置，对象 1/s，P（速度环已提供阻尼）。
    position = PIDGains(kp=position_bandwidth, ki=0.0, kd=0.0)

    # 高度环：与固件一致的串级结构（速度内环 + 位置外环）。
    altitude = _design_altitude_cascade(
        velocity_bandwidth=altitude_bandwidth,
        position_bandwidth=altitude_bandwidth / 3.0,
        damping=damping,
    )

    gains = CascadeGains(
        rate=rate,
        attitude=attitude,
        velocity=velocity,
        position=position,
        altitude=altitude,
        bandwidths={
            "rate_rad_s": rate_bandwidth,
            "attitude_rad_s": attitude_bandwidth,
            "velocity_rad_s": velocity_bandwidth,
            "position_rad_s": position_bandwidth,
            "altitude_rad_s": altitude_bandwidth,
        },
        damping=damping,
    )
    gains.metrics = evaluate_linear(p, gains)
    gains.betaflight_hint = betaflight_hint(p, gains, gains.metrics)
    return gains


def evaluate_linear(params: QuadrotorParams, gains: CascadeGains) -> Dict[str, Dict[str, float]]:
    """对每个回路做独立的线性闭环评估。"""
    out: Dict[str, Dict[str, float]] = {}

    plant_rate = signal.TransferFunction([1.0], [params.ixx, 0.0])
    c_rate = _pid_tf(gains.rate)
    t_rate = _closed_loop(plant_rate, c_rate)
    out["rate"] = {**_metrics(t_rate), **_margins(plant_rate, c_rate)}

    # 姿态角：内环近似 1，对象 1/s。
    plant_att = signal.TransferFunction([1.0], [1.0, 0.0])
    c_att = _pd_tf(gains.attitude)
    t_att = _closed_loop(plant_att, c_att)
    out["attitude"] = {**_metrics(t_att), **_margins(plant_att, c_att)}

    # 速度环：对象 1/s。
    plant_vel = signal.TransferFunction([1.0], [1.0, 0.0])
    c_vel = _pid_tf(gains.velocity)
    t_vel = _closed_loop(plant_vel, c_vel)
    out["velocity"] = {**_metrics(t_vel), **_margins(plant_vel, c_vel)}

    # 位置环：对象 1/s，P。
    plant_pos = signal.TransferFunction([1.0], [1.0, 0.0])
    c_pos = signal.TransferFunction([gains.position.kp], [1.0])
    t_pos = _closed_loop(plant_pos, c_pos)
    out["position"] = {**_metrics(t_pos), **_margins(plant_pos, c_pos)}

    # 高度环：与固件一致的速度内环 + 位置外环。
    # 内环对象：加速度 -> 速度，1/s，PI 增益为 (altitude.kd, altitude.ki)。
    plant_vz = signal.TransferFunction([1.0], [1.0, 0.0])
    c_vz = _pid_tf(PIDGains(kp=gains.altitude.kd, ki=gains.altitude.ki))
    t_vz = _closed_loop(plant_vz, c_vz)
    # 外环对象：速度闭环 -> 高度，t_vz/s；控制器为位置 P。
    plant_z = signal.TransferFunction(t_vz.num, np.concatenate([t_vz.den, [0.0]]))
    c_z = signal.TransferFunction([gains.altitude.kp], [1.0])
    t_z = _closed_loop(plant_z, c_z)
    out["altitude"] = {**_metrics(t_z), **_margins(plant_z, c_z)}
    out["altitude_velocity_inner"] = {**_metrics(t_vz), **_margins(plant_vz, c_vz)}
    return out


def betaflight_hint(
    params: QuadrotorParams,
    gains: CascadeGains,
    metrics: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """把理论增益换算成可直接尝试的保守初值。

    高度环只输出 **归一化串级增益**，因为 ``tune.py`` 的结构与固件
    ``altitudeControl()`` 的 P/I 直接作用结构不同。要写
    ``ap_altitude_p/i/d/f``，请使用 :mod:`modeling.altitude_loop`。
    """
    # 姿态角环：angle_p_gain 是 RC 角度环增益，量纲近似 deg/s per deg。
    # 理论姿态环 kp 是 rad/s per rad，数值近似直接乘 100 作为保守起点。
    angle_p = max(10.0, min(250.0, gains.attitude.kp * 100.0))

    # 角速率环：Betaflight p_pitch/p_roll 是内部缩放，实际等价关系取决于
    # gyro 量纲。这里只输出一个与理论 kp 成比例的保守参考值。
    p_rate = max(20.0, min(150.0, gains.rate.kp * 1000.0))

    return {
        "altitude_cascade_normalized": {
            "kp_per_s": round(gains.altitude.kp, 6),
            "ki_per_s2": round(gains.altitude.ki, 6),
            "kd_per_s": round(gains.altitude.kd, 6),
        },
        "angle_p_gain": round(angle_p, 1),
        "p_roll_pitch_hint": round(p_rate, 1),
        "note": (
            "altitude_cascade_normalized 是归一化串级增益；固件 ap_altitude_* "
            "请用 modeling.altitude_loop --sitl 生成。"
        ),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _write_json(path: str, gains: CascadeGains) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(gains.to_json())


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="四旋翼级联 PID 理论整定（Simulink 替代）")
    parser.add_argument("--rate-bw", type=float, default=45.0, help="角速率环带宽 rad/s")
    parser.add_argument("--attitude-bw", type=float, default=8.0, help="姿态角环带宽 rad/s")
    parser.add_argument("--velocity-bw", type=float, default=3.0, help="速度环带宽 rad/s")
    parser.add_argument("--position-bw", type=float, default=1.0, help="位置环带宽 rad/s")
    parser.add_argument("--altitude-bw", type=float, default=1.2, help="高度环带宽 rad/s")
    parser.add_argument("--damping", type=float, default=0.9)
    parser.add_argument("--mass", type=float, default=1.4)
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "theory_pid.json"),
    )
    parser.add_argument("--print", action="store_true", help="同时打印 JSON")
    args = parser.parse_args(argv)

    params = QuadrotorParams(mass=args.mass)
    gains = design_cascade(
        params,
        rate_bandwidth=args.rate_bw,
        attitude_bandwidth=args.attitude_bw,
        velocity_bandwidth=args.velocity_bw,
        position_bandwidth=args.position_bw,
        altitude_bandwidth=args.altitude_bw,
        damping=args.damping,
    )
    _write_json(args.output, gains)
    print(f"wrote {args.output}")
    if args.print:
        print(gains.to_json())
    else:
        print(json.dumps(gains.betaflight_hint, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
