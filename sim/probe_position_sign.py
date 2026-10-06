"""东向符号验证探针：做一次大范围平移，比对真值 vs 固件估计的东向分量。

判据：corr(est_e, truth_e) 与拟合斜率。修前应为负相关（镜像），修后应接近 +1。
"""
from __future__ import annotations

import csv
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from gazebo_target_bridge import GazeboPoseSource  # noqa: E402
from run_intercept_mission import (RcState, climb_to, rc_sender, spool_up,  # noqa: E402
                                   wait_until_armed)
from run_p_vision_strike import read_own  # noqa: E402

_BF = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF):
    sys.path.insert(0, _BF)
import offboard_position_example as offboard  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
OUT = "/tmp/possign_%s.csv" % TAG


def main():
    src = GazeboPoseSource("betaflight_world", "iris_with_Betaflight", "target_drone")
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = RcState()
    run = [True]
    threading.Thread(target=rc_sender, args=(udp, rc, run), daemon=True).start()
    link = offboard.MspLink()
    print("arming...", flush=True)
    if not wait_until_armed(link, rc, timeout=40):
        print("ARM FAILED")
        return
    spool_up(rc, target=1430, duration=2.0)
    climb_to(rc, src, link, 10.0, timeout=25.0)
    time.sleep(1.0)

    rows = []
    t0 = time.monotonic()

    def fly(rl, pt, yw, thr, dur):
        end = time.monotonic() + dur
        while time.monotonic() < end:
            rc.set_rpyt(rl, pt, thr, yw)
            own = read_own(link)
            p = src.pose()
            if own is not None and p is not None:
                rows.append([time.monotonic() - t0,
                             *p.own_position_enu, *own["pos"]])
            time.sleep(0.02)

    print("yaw 90deg...", flush=True)
    fly(1500, 1500, 1700, 1470, 3.0)      # 偏航转向
    print("forward...", flush=True)
    fly(1500, 1650, 1500, 1470, 4.0)      # 前飞（沿新航向产生水平平移）
    print("stop...", flush=True)
    fly(1500, 1500, 1500, 1430, 2.0)

    rc.set_throttle(1000)
    rc.set_aux1(1000)
    time.sleep(0.3)
    run[0] = False
    udp.close()

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "te", "tn", "tu", "pe", "pn", "pu"])
        w.writerows(rows)

    te, tn = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    pe, pn = np.array([r[4] for r in rows]), np.array([r[5] for r in rows])

    print("=== %s ===" % TAG)
    print("真值东向范围 %.2f .. %.2f m (跨度 %.2f)   真值北向跨度 %.2f"
          % (te.min(), te.max(), te.max() - te.min(), tn.max() - tn.min()))
    for lbl, a, b in (("东向 E", pe, te), ("北向 N", pn, tn)):
        if float(np.std(b)) > 0.3:
            c = float(np.corrcoef(a, b)[0, 1])
            k = float(np.polyfit(b, a, 1)[0])
            print("%s: 相关系数 = %+0.3f   拟合斜率 = %+0.3f   (期望 +1；-1 表示镜像)"
                  % (lbl, c, k))
        else:
            print("%s: 真值变化太小(%.2f m)，跳过" % (lbl, float(np.std(b))))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
