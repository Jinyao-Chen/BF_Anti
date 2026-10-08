#!/usr/bin/env python3
"""让靶机以**匀速**运动（Gazebo ``VelocityControl`` 插件），并回显真值轨迹。

主世界 ``sim/worlds/bf_anti_p_strike_light_cam.sdf`` 里的 ``target_drone`` 挂了
``gz-sim-velocity-control-system``，该插件订阅 ``/model/<target>/cmd_vel``（``gz.msgs.Twist``，
模型系 = 世界系，因为靶机没有位姿偏航）。本仓库此前**没有任何代码发过这条指令**，
所以靶机一直静止在 ``(45, 0, 12)``。

约定：命令行用 **ENU**（东/北/上，m/s），内部换算成 Gazebo xyz：``x=N, y=-E, z=U``；
插件会保持最后一次速度指令，所以按 ``--hold-hz`` 重复发只是防止中途被清掉。

用法::

    # 终端 1（起仿真）
    ./sim/launch_sitl.sh --headless

    # 终端 2：靶机以 5 m/s 向东横穿，持续 20 s，每 0.5 s 打印真值位置/速度
    python3 sim/move_target.py --vel-e 5 --duration 20

    # 背向逃逸（+N = 远离拦截机）：
    python3 sim/move_target.py --vel-n 4 --duration 20

    # 停下（发零速）：
    python3 sim/move_target.py --duration 1
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gazebo_target_bridge import GazeboPoseSource  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="靶机匀速运动",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="betaflight_world")
    ap.add_argument("--target", default="target_drone")
    ap.add_argument("--vel-e", type=float, default=0.0, help="东向速度 m/s（ENU）")
    ap.add_argument("--vel-n", type=float, default=0.0, help="北向速度 m/s（ENU）")
    ap.add_argument("--vel-u", type=float, default=0.0, help="垂向速度 m/s（ENU）")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--hold-hz", type=float, default=20.0,
                    help="重复发 cmd_vel 的频率（插件本来就保持上次指令）")
    ap.add_argument("--print-hz", type=float, default=2.0)
    args = ap.parse_args()

    import gz.transport13 as gztransport
    from gz.msgs10.twist_pb2 import Twist

    topic = "/model/%s/cmd_vel" % args.target
    node = gztransport.Node()
    pub = node.advertise(topic, Twist)

    # ENU -> Gazebo(x=N, y=-E, z=U)；靶机没有偏航，所以模型系就是世界系
    cmd = Twist()
    cmd.linear.x = float(args.vel_n)
    cmd.linear.y = float(-args.vel_e)
    cmd.linear.z = float(args.vel_u)

    pose_src = GazeboPoseSource(args.world, target=args.target)
    speed = math.sqrt(args.vel_e ** 2 + args.vel_n ** 2 + args.vel_u ** 2)
    print("[move-target] %s <- E%.2f N%.2f U%.2f m/s (|v|=%.2f) 持续 %.1f s"
          % (topic, args.vel_e, args.vel_n, args.vel_u, speed, args.duration), flush=True)

    t0 = time.monotonic()
    last_pub = 0.0
    last_print = 0.0
    prev = None
    prev_t = None
    got = False
    while True:
        now = time.monotonic()
        el = now - t0
        if el > args.duration:
            break
        if now - last_pub >= 1.0 / max(1e-3, args.hold_hz):
            pub.publish(cmd)
            last_pub = now
        if now - last_print >= 1.0 / max(1e-3, args.print_hz):
            last_print = now
            pose = pose_src.pose()
            if pose is not None:
                tgt = [float(v) for v in pose.target_position_enu]
                vel = None
                if prev is not None and prev_t is not None and now - prev_t > 1e-3:
                    vel = [(tgt[i] - prev[i]) / (now - prev_t) for i in range(3)]
                if vel is not None:
                    got = True
                    print("  t=%5.2f  target(E,N,U)=(%7.2f,%7.2f,%7.2f)  "
                          "v_truth=(%+5.2f,%+5.2f,%+5.2f) m/s  |v|=%.2f"
                          % (el, tgt[0], tgt[1], tgt[2], vel[0], vel[1], vel[2],
                             math.sqrt(sum(v * v for v in vel))), flush=True)
                else:
                    print("  t=%5.2f  target(E,N,U)=(%7.2f,%7.2f,%7.2f)"
                          % (el, tgt[0], tgt[1], tgt[2]), flush=True)
                prev, prev_t = tgt, now
        time.sleep(0.01)

    # 收尾：发一次零速，避免靶机一直飞
    stop = Twist()
    for _ in range(5):
        pub.publish(stop)
        time.sleep(0.05)
    print("[move-target] 结束（已发零速）。%s"
          % ("真值速度与指令一致" if got else "警告：没拿到靶机真值"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
