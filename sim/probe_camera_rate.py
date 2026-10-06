#!/usr/bin/env python3
"""测量 Gazebo 相机帧的实际投递速率 / 重复帧比例（决定"框冻结"根因）。

用法: python3 sim/probe_camera_rate.py [--seconds 12]
输出: 回调速率、主线程采样到的"帧变化"速率、重复帧比例、帧内容差分。
"""
import argparse,  os,  sys,  time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "sim"))
from gazebo_camera_source import GazeboCameraSource

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--world", default="betaflight_world")
    ap.add_argument("--interceptor", default="iris_with_Betaflight")
    ap.add_argument("--cam-link", default="cam_link")
    a = ap.parse_args()

    cam = GazeboCameraSource(a.world, a.interceptor, inner_model="", link=a.cam_link)
    # 1) 主线程 20 Hz 采样：每次看 frame() 是否与上一次不同
    samples = []
    t0 = time.monotonic(); prev = None
    while time.monotonic() - t0 < a.seconds:
        f = cam.frame()
        if f is not None:
            if prev is None:
                samples.append(("new", 0.0))
            else:
                d = float(np.abs(f.astype(np.int16) - prev.astype(np.int16)).mean())
                samples.append(("new" if d > 1e-9 else "same", d))
            prev = f.copy()
        time.sleep(0.05)
    n = len(samples)
    news = sum(1 for s, _ in samples if s == "new")
    diffs = np.array([d for s, d in samples if s == "new"])
    print("main-thread 20Hz samples      : %d over %.1fs" % (n, a.seconds))
    print("distinct frames seen          : %d (=> image update rate ~%.1f Hz)" % (news, news / a.seconds))
    if len(diffs):
        print("frame-to-frame mean|diff|     : median %.2f  p10 %.2f  max %.2f" % (
            np.median(diffs), np.percentile(diffs, 10), diffs.max()))
    print("camera.age_s at end           : %.3f s (last callback age)" % cam.age_s)
    # 2) 统计回调次数（直接读私有计数需要挂钩子，这里用 _t 变化次数近似）
    times = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 3.0:
        t = cam._t; times.append(t); time.sleep(0.02)
    print("callback-timestamp changes    : %d in 3 s" % (len(set(round(x, 6) for x in times)) - 1))

if __name__ == "__main__":
    raise SystemExit(main())
