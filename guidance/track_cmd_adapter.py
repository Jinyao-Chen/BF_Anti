"""GuidanceCommand -> rc_mux.TrackCmd 适配层。

GuidanceCommand 输出的是机体姿态角/归一化油门；TrackCmd 是 rc_mux 的
RC 单位 (1000-2000)。这里做一阶线性映射，便于仿真和后续真实链路调参。
"""

from __future__ import annotations

import time

from rc_mux import TrackCmd

from .intercept_guidance import GuidanceCommand


def _clamp_rc(value: float) -> int:
    return int(max(1000.0, min(2000.0, round(value))))


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
) -> TrackCmd:
    """把 GuidanceCommand 转成 rc_mux 使用的 TrackCmd。

    ``roll_deg`` 显式给出时覆盖 ``cmd.roll_cmd_deg``；否则直接使用制导层
    的 roll 指令，让侧向机动也能被 RC 融合层执行。
    """
    roll = cmd.roll_cmd_deg if roll_deg is None else float(roll_deg)
    return TrackCmd(
        pitch=_angle_to_rc(cmd.pitch_cmd_deg, max_angle_deg),
        yaw=_angle_to_rc(cmd.yaw_cmd_deg, max_angle_deg),
        throttle=_throttle_to_rc(cmd.throttle_cmd),
        roll=_angle_to_rc(roll, max_angle_deg),
        t=time.monotonic() if timestamp is None else float(timestamp),
    )
