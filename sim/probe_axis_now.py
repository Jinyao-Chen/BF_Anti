#!/usr/bin/env python3
"""轴上开环标定（当前 SITL 构建 + light_quad 世界）：姿态指令 -> 真值角速率。

用途：确认 yaw 通道的**实际增益与极性**（角速率命令 90 dps 到底转多少），
以及 roll/pitch 的 ANGLE 增益。基线：悬停油门 1430、angle_limit 40 时的映射
（rc = 1500 + cmd_deg * 500/40，yaw: rc = 1500 + dps * 500/300）。

用法: python3 sim/probe_axis_now.py
"""
from __future__ import annotations

import os
import sys
import threading
import time
import socket

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import GazeboPoseSource  # noqa: E402
from run_intercept_mission import RcState, rc_sender, spool_up, wait_until_armed  # noqa: E402
from run_visual_strike import sitl_reachable  # noqa: E402

_BF = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF):
    sys.path.insert(0, _BF)
import offboard_position_example as offboard  # noqa: E402

HOVER = 1430.0
ANGLE_LIMIT = 40.0
YAW_FULL = 300.0


def rc_att(roll_deg=0.0, pitch_deg=0.0, yaw_dps=0.0, thr=HOVER):
    ch = [1500] * 16
    ch[0] = int(round(1500 + roll_deg * 500.0 / ANGLE_LIMIT))
    ch[1] = int(round(1500 - pitch_deg * 500.0 / ANGLE_LIMIT))
    ch[2] = int(round(thr))
    ch[3] = int(round(1500 + yaw_dps * 500.0 / YAW_FULL))
    ch[4] = 2000
    ch[5] = 2000
    return [max(1000, min(2000, v)) for v in ch]


def main():
    if not sitl_reachable():
        print("ERROR: SITL MSP 不可达")
        return 2
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = RcState()
    running = [True]
    threading.Thread(target=rc_sender, args=(udp, rc, running), daemon=True).start()
    link = offboard.MspLink()
    src = GazeboPoseSource("betaflight_world", "iris_with_Betaflight", "target_drone")

    print("arming...", flush=True)
    if not wait_until_armed(link, rc, timeout=40.0):
        print("ERROR: 解锁失败")
        return 2
    spool_up(rc, target=int(HOVER), duration=2.0)
    print("hover settle 4 s ...", flush=True)
    rc.set_channels(rc_att())
    time.sleep(4.0)

    segs = [
        ("neutral", rc_att()),
        ("yaw +90dps", rc_att(yaw_dps=+90.0)),
        ("yaw -90dps", rc_att(yaw_dps=-90.0)),
        ("roll +15deg", rc_att(roll_deg=+15.0)),
        ("roll -15deg", rc_att(roll_deg=-15.0)),
        ("pitch +10deg", rc_att(pitch_deg=+10.0)),
        ("pitch -10deg", rc_att(pitch_deg=-10.0)),
        ("neutral2", rc_att()),
    ]
    print("%-14s | %-28s | %s" % ("segment", "truth rate (deg/s) r/p/y", "msp rpy (deg)"))
    for name, ch in segs:
        rc.set_channels(ch)
        time.sleep(1.0)                     # 让姿态/速率进入稳态
        t0 = time.monotonic()
        y0 = None
        rs, ps, ys = [], [], []
        while time.monotonic() - t0 < 1.6:
            pose = src.pose()
            att = link.query(0x6A, is_v2=False)
            if pose is not None:
                if y0 is None:
                    y0 = (np.array(pose.own_rpy_deg, dtype=float), time.monotonic())
                else:
                    dt = time.monotonic() - y0[1]
                    if dt > 0.2:
                        d = np.array(pose.own_rpy_deg) - y0[0]
                        d = (d + 180.0) % 360.0 - 180.0
                        rs.append(d[0] / dt)
                        ps.append(d[1] / dt)
                        ys.append(d[2] / dt)
            time.sleep(0.02)
        msp = "-"
        if att and len(att) >= 6:
            import struct as _s
            r, p, yw = _s.unpack_from("<3h", att, 0)
            msp = "%.1f %.1f %.1f" % (r / 10.0, p / 10.0, yw / 10.0)
        print("%-14s | %+7.1f %+7.1f %+7.1f | %s" % (
            name, np.median(rs) if rs else float("nan"),
            np.median(ps) if ps else float("nan"),
            np.median(ys) if ys else float("nan"), msp), flush=True)
    rc.set_channels([1500] * 16)
    rc.set_channels(rc_att(thr=1000.0))
    rc.idle()
    time.sleep(0.5)
    running[0] = False
    udp.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
