#!/usr/bin/env python3
"""诊断：Gazebo 相机画面是否跟随机体姿态（一次运行给出结论）。"""
import os, socket, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from gazebo_camera_source import GazeboCameraSource
from gazebo_target_bridge import GazeboPoseSource
from run_intercept_mission import RcState, climb_to, rc_sender, spool_up, wait_until_armed
_BF = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF):
    sys.path.insert(0, _BF)
import offboard_position_example as offboard


def main():
    inner = sys.argv[1] if len(sys.argv) > 1 else "light_quad"
    cam = GazeboCameraSource("betaflight_world", "iris_with_Betaflight",
                             inner_model=inner) if inner else GazeboCameraSource(
        "betaflight_world", "iris_with_Betaflight")
    src = GazeboPoseSource("betaflight_world", "iris_with_Betaflight", "target_drone")
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = RcState(); run = [True]
    threading.Thread(target=rc_sender, args=(udp, rc, run), daemon=True).start()
    link = offboard.MspLink()
    assert wait_until_armed(link, rc), "arm failed"
    spool_up(rc, target=1430, duration=2.0)
    climb_to(rc, src, link, 8.0, timeout=25.0)
    time.sleep(1.0)

    import cv2
    prev = None
    print("%6s %-24s %-24s %8s %8s" % ("t", "truth pos(E,N,U)", "truth rpy(r,p,y)", "img_diff", "obj_px"))
    for phase, stick in (("level", 1500), ("pitch+400", 1900), ("hold", 1900), ("release", 1500)):
        t0 = time.monotonic()
        while time.monotonic() - t0 < 2.5:
            ch = rc.snapshot(); ch[1] = stick; rc.set_channels(ch)
            f = cam.frame()
            if f is not None:
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                d = 0.0 if prev is None else float(np.abs(g.astype(int) - prev).mean())
                dark = g < 90
                dark[:110, :] = False; dark[440:, :] = False
                n = int(dark.sum())
                prev = g.astype(int)
            else:
                d, n = float("nan"), 0
            p = src.pose()
            if p is not None and int((time.monotonic() - t0) * 10) % 10 == 0:
                print("%6.2f (%6.1f,%6.1f,%6.1f)  (%6.1f,%6.1f,%6.1f)  %8.3f %8d" % (
                    time.monotonic() - t0, *p.own_position_enu, *p.own_rpy_deg, d, n), flush=True)
            time.sleep(0.1)
    rc.set_throttle(1000); rc.set_aux1(1000); time.sleep(0.3); run[0] = False; udp.close()


if __name__ == "__main__":
    main()
