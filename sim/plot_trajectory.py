#!/usr/bin/env python3
"""飞行日志 -> 3D 轨迹图与时间序列图。

用法::

    python3 sim/plot_trajectory.py --log /tmp/altitude_diag.csv \
        --out /tmp/altitude_diag.png

默认优先使用 Gazebo 真值列（``truth_e/truth_n/truth_u``），没有时退回
固件估计列（``east/north/up``）。靶机位置支持
``target_e/target_n/target_u``、``tx/ty/tz`` 或
``truth_target_e/...`` 命名。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read_csv(path: str) -> dict[str, np.ndarray]:
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise SystemExit(f"empty CSV: {path}")
    columns: dict[str, list[float]] = {}
    for name in rows[0].keys():
        values: list[float] = []
        for row in rows:
            raw = row.get(name, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(float("nan"))
        columns[name] = np.asarray(values, dtype=float)
    return columns


def _pick(columns: dict[str, np.ndarray], *names: str) -> Optional[np.ndarray]:
    for name in names:
        if name in columns:
            return columns[name]
    return None


def _finite_or_none(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if array is None:
        return None
    if np.all(np.isnan(array)):
        return None
    return array


def plot_csv(log_path: str, out_path: str, title: str = "") -> str:
    columns = _read_csv(log_path)
    n = len(next(iter(columns.values())))

    time = _pick(columns, "t", "time", "timestamp")
    if time is None:
        time = np.arange(n, dtype=float) * 0.05
    time = time - np.nanmin(time)

    truth = np.vstack([
        _finite_or_none(_pick(columns, "truth_e", "truth_east")),
        _finite_or_none(_pick(columns, "truth_n", "truth_north")),
        _finite_or_none(_pick(columns, "truth_u", "truth_up")),
    ]) if all(
        _finite_or_none(_pick(columns, name)) is not None
        for name in ("truth_e", "truth_n", "truth_u")
    ) else None
    if truth is None:
        east = _finite_or_none(_pick(columns, "east", "x"))
        north = _finite_or_none(_pick(columns, "north", "y"))
        up = _finite_or_none(_pick(columns, "up", "z"))
        if east is not None and north is not None and up is not None:
            truth = np.vstack([east, north, up])
    if truth is None:
        raise SystemExit("no position columns found (truth_* or east/north/up)")

    target = None
    tx = _pick(columns, "target_e", "tx", "truth_target_e", "target_east")
    ty = _pick(columns, "target_n", "ty", "truth_target_n", "target_north")
    tz = _pick(columns, "target_u", "tz", "truth_target_u", "target_up")
    if tx is not None and ty is not None and tz is not None:
        target = np.vstack([tx, ty, tz])

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(title or Path(log_path).name, fontsize=13)

    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax3d.plot(truth[0], truth[1], truth[2], color="tab:blue", label="interceptor")
    if target is not None:
        ax3d.plot(target[0], target[1], target[2], color="tab:red", label="target")
    ax3d.scatter(truth[0, 0], truth[1, 0], truth[2, 0], color="green", marker="o", label="start")
    ax3d.scatter(truth[0, -1], truth[1, -1], truth[2, -1], color="black", marker="x", label="end")
    ax3d.set_xlabel("East [m]")
    ax3d.set_ylabel("North [m]")
    ax3d.set_zlabel("Up [m]")
    ax3d.set_title("3D trajectory")
    ax3d.legend(loc="upper left")

    axxy = fig.add_subplot(2, 3, 2)
    axxy.plot(truth[0], truth[1], color="tab:blue", label="interceptor")
    if target is not None:
        axxy.plot(target[0], target[1], color="tab:red", label="target")
    axxy.set_xlabel("East [m]")
    axxy.set_ylabel("North [m]")
    axxy.axis("equal")
    axxy.grid(True, alpha=0.3)
    axxy.set_title("Top view")
    axxy.legend(loc="best")

    axalt = fig.add_subplot(2, 3, 3)
    axalt.plot(time, truth[2], color="tab:blue", label="interceptor up")
    if target is not None:
        axalt.plot(time, target[2], color="tab:red", label="target up")
    axalt.set_xlabel("time [s]")
    axalt.set_ylabel("Up [m]")
    axalt.grid(True, alpha=0.3)
    axalt.set_title("Altitude")
    axalt.legend(loc="best")

    axvz = fig.add_subplot(2, 3, 4)
    vu = _pick(columns, "vu", "truth_vu", "vz")
    vz_cmd = _pick(columns, "msp_vario")
    if vu is not None:
        axvz.plot(time, vu, label="vertical velocity")
    if vz_cmd is not None:
        axvz.plot(time, vz_cmd, label="MSP vario")
    axvz.set_xlabel("time [s]")
    axvz.set_ylabel("vertical velocity [m/s]")
    axvz.grid(True, alpha=0.3)
    axvz.set_title("Vertical velocity")
    axvz.legend(loc="best")

    axpitch = fig.add_subplot(2, 3, 5)
    pitch = _pick(columns, "pitch", "truth_pitch")
    if pitch is not None:
        axpitch.plot(time, pitch, color="tab:orange")
    axpitch.set_xlabel("time [s]")
    axpitch.set_ylabel("pitch [deg]")
    axpitch.grid(True, alpha=0.3)
    axpitch.set_title("Pitch")

    axmotor = fig.add_subplot(2, 3, 6)
    motor_names = [name for name in ("m0", "m1", "m2", "m3") if name in columns]
    for name in motor_names:
        axmotor.plot(time, columns[name], label=name)
    if motor_names:
        axmotor.set_xlabel("time [s]")
        axmotor.set_ylabel("motor PWM")
        axmotor.grid(True, alpha=0.3)
        axmotor.set_title("Motors")
        axmotor.legend(loc="best")
    else:
        axmotor.axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="绘制飞行日志 3D 轨迹与时间序列")
    parser.add_argument("--log", required=True, help="CSV 日志路径")
    parser.add_argument("--out", default=None, help="输出 PNG（默认与日志同目录同名 .png）")
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = str(Path(args.log).with_suffix(".png"))
    result = plot_csv(args.log, out, title=args.title)
    print(f"wrote {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
