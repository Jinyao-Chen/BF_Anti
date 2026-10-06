"""估计器滞后探针：固定杆量阶跃加速，直接量测固件估计 vs 真值的滞后。

不依赖制导闭环，避免跑飞样本污染统计。
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
OUT = "/tmp/estlag_%s.csv" % TAG


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

    def step(rl, pt, dur, label):
        end = time.monotonic() + dur
        while time.monotonic() < end:
            rc.set_rpyt(rl, pt, 1430, 1500)
            own = read_own(link)
            p = src.pose()
            if own is not None and p is not None:
                rows.append([time.monotonic() - t0,
                             *p.own_position_enu, *own["pos"], *own["vel"]])
            time.sleep(0.02)

    print("hover...", flush=True)
    step(1500, 1500, 2.0, "hover")
    print("STEP: pitch 1700 (forward accel) 4s", flush=True)
    step(1500, 1700, 4.0, "accel")
    print("release...", flush=True)
    step(1500, 1500, 4.0, "coast")

    rc.set_throttle(1000)
    rc.set_aux1(1000)
    time.sleep(0.3)
    run[0] = False
    udp.close()

    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "tn", "te", "tu", "pn", "pe", "pu", "vn", "ve", "vu"])
        w.writerows(rows)

    t = np.array([r[0] for r in rows])
    tn = np.array([r[1] for r in rows])
    pn = np.array([r[4] for r in rows])
    vn = np.array([r[7] for r in rows])
    vn_true = np.gradient(tn, t)

    def lagfit(sig, ref, rng):
        best = None
        for k in range(0, rng):
            lag = k * 0.02
            idx = np.searchsorted(t, t - lag)
            idx = np.clip(idx, 0, len(t) - 1)
            m = float(np.mean(np.abs(sig - ref[idx])))
            if best is None or m < best[0]:
                best = (m, lag)
        return best

    pl = lagfit(pn, tn, 120)
    vl = lagfit(vn, vn_true, 120)
    print("=== %s ===" % TAG)
    print("位置滞后 = %.2f s  (残差 %.3f m)" % (pl[1], pl[0]))
    print("速度滞后 = %.2f s  (残差 %.3f m/s)" % (vl[1], vl[0]))
    print("samples=%d  speed max truth=%.2f m/s  msp=%.2f m/s"
          % (len(rows), float(np.max(np.abs(vn_true))), float(np.max(np.abs(vn)))))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
