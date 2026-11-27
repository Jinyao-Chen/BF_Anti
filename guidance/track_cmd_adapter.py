"""GuidanceCommand -> rc_mux.TrackCmd 适配层。

GuidanceCommand 输出的是机体姿态角/归一化油门；TrackCmd 是 rc_mux 的
RC 单位 (1000-2000)。这里做一阶线性映射，便于仿真和后续真实链路调参。
"""

from __future__ import annotations

import math
import time

from rc_mux import TrackCmd

from .intercept_guidance import GuidanceCommand


def _clamp_rc(value: float) -> int:
    v = float(value)
    if not math.isfinite(v):
        return 1500
    return int(max(1000.0, min(2000.0, round(v))))


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _angle_to_rc(angle_deg: float, max_angle_deg: float = 45.0) -> int:
    """姿态角 -> RC 单位。

    1500 表示 0 度，+max_angle_deg 映射到 2000，-max_angle_deg 映射到 1000。
    """
    scale = 500.0 / max(1e-6, float(max_angle_deg))
    return _clamp_rc(1500.0 + float(angle_deg) * scale)


def _throttle_to_rc(throttle_cmd: float) -> int:
    """归一化油门 [0,1] -> RC 1000-2000。"""
    t = max(0.0, min(1.0, float(throttle_cmd)))
    return _clamp_rc(1000.0 + t * 1000.0)


def guidance_to_track_cmd(
    cmd: GuidanceCommand,
    max_angle_deg: float = 45.0,
    timestamp: float | None = None,
    roll_deg: float | None = None,
    own_yaw_deg: float = 0.0,
    max_yaw_rate_dps: float = 200.0,
    yaw_rate_gain: float = 2.0,
) -> TrackCmd:
    """把 GuidanceCommand 转成 rc_mux 使用的 TrackCmd。

    roll_deg 显式给出时覆盖 roll 指令；否则直接使用制导层的 roll 指令。

    Betaflight 约定：pitch 杆高 = 低头，而 GuidanceCommand 的 pitch 正方向
    是抬头，故取反；yaw 杆是角速率而不是绝对方位，故用 own_yaw_deg 把
    yaw_cmd_deg 的航向误差转成有界角速率。
    """
    roll = cmd.roll_cmd_deg if roll_deg is None else float(roll_deg)
    rate_max = max(1.0, float(max_yaw_rate_dps))
    yaw_err = _wrap180(float(cmd.yaw_cmd_deg) - float(own_yaw_deg))
    yaw_rate = max(-rate_max, min(rate_max, float(yaw_rate_gain) * yaw_err))
    yaw_rc = _clamp_rc(1500.0 + 500.0 * yaw_rate / rate_max)
    # Betaflight pitch stick high = nose down; GuidanceCommand uses +up.
    # Betaflight yaw is an angular rate, not an absolute heading.
    return TrackCmd(
        pitch=_angle_to_rc(-float(cmd.pitch_cmd_deg), max_angle_deg),
        yaw=yaw_rc,
        throttle=_throttle_to_rc(cmd.throttle_cmd),
        roll=_angle_to_rc(roll, max_angle_deg),
        t=time.monotonic() if timestamp is None else float(timestamp),
    )
