#!/usr/bin/env python3
"""订阅 Gazebo ``/betaflight/telemetry`` 并打印控制内部量。

数据布局（由 BetaflightPlugin 发布）::

    data[0..3] : 4 个归一化电机指令
    data[4]    : imu_link 世界高度 z (m)
    data[5..7] : roll / pitch / yaw (rad)

用法::

    # 终端 1
    ./sim/launch_sitl.sh --headless

    # 终端 2
    python3 sim/telemetry_monitor.py --duration 10
"""

from __future__ import annotations

import argparse
import time

import gz.transport13 as gztransport
from gz.msgs10.float_v_pb2 import Float_V


class Monitor:
    def __init__(self) -> None:
        self.last = None
        self.count = 0
        self.node = gztransport.Node()
        if not self.node.subscribe(Float_V, "/betaflight/telemetry", self._on_msg):
            raise RuntimeError("failed to subscribe /betaflight/telemetry")

    def _on_msg(self, msg: Float_V) -> None:
        self.last = msg
        self.count += 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Gazebo Betaflight 遥测监视器")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--rate", type=float, default=2.0, help="打印频率 Hz")
    args = parser.parse_args()

    monitor = Monitor()
    start = time.monotonic()
    period = 1.0 / max(1.0, args.rate)
    while time.monotonic() - start < args.duration:
        msg = monitor.last
        if msg is not None and len(msg.data) >= 8:
            motors = list(msg.data[0:4])
            alt = msg.data[4]
            roll, pitch, yaw = msg.data[5:8]
            print(
                f"t={time.monotonic() - start:5.1f}s motors="
                f"({motors[0]:5.2f},{motors[1]:5.2f},{motors[2]:5.2f},{motors[3]:5.2f}) "
                f"alt={alt:6.2f}m rpy=({roll:6.3f},{pitch:6.3f},{yaw:6.3f}) rad",
                flush=True,
            )
        time.sleep(period)
    print(f"received {monitor.count} telemetry messages", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
