#!/usr/bin/env python3
"""查清真值姿态转换与 Gazebo 渲染是否一致（一次运行给结论）。

打印：机体/靶机在 pose/info 里的原始四元数、两种算法得到的 yaw、
以及"靶机相对本机的真实方位"——用于判定合成投影的方位是否可信。
"""
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import (  # noqa: E402
    gazebo_xyz_to_enu, quat_to_rotation_enu_body, rpy_from_rotation_enu_body,
)


def yaw_from_quat_direct(q):
    """直接由四元数算 Gazebo 系下的 yaw（绕 z），再转成罗盘角。"""
    x, y, z, w = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def main():
    import gz.transport13 as gztransport
    from gz.msgs10.pose_v_pb2 import Pose_V

    node = gztransport.Node()
    box = {}

    def cb(msg):
        box["msg"] = msg
    node.subscribe(Pose_V, "/world/betaflight_world/pose/info", cb)
    for _ in range(100):
        if "msg" in box:
            break
        time.sleep(0.1)
    msg = box.get("msg")
    if msg is None:
        print("no pose msg")
        return

    print("%-24s %-30s %8s %8s %8s" % ("entity", "pos ENU(E,N,U)", "qz_yaw", "conv_yaw", "compass"))
    ents = {}
    for p in msg.pose:
        q = (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w)
        enu = gazebo_xyz_to_enu(p.position.x, p.position.y, p.position.z)
        if p.name not in ents:
            ents[p.name] = (enu, q)
        conv = rpy_from_rotation_enu_body(quat_to_rotation_enu_body(*q))
        print("%-24s (%7.2f,%7.2f,%7.2f) %8.1f %8.1f %8.1f" % (
            p.name, enu[0], enu[1], enu[2], yaw_from_quat_direct(q), conv[2],
            math.degrees(math.atan2(0, 1))))

    if "iris_with_Betaflight" in ents and "target_drone" in ents:
        o_enu, _ = ents["iris_with_Betaflight"]
        t_enu, _ = ents["target_drone"]
        rel = t_enu - o_enu
        print("\n靶机相对本机: 方位=%.1f°  仰角=%.1f°  距离=%.2f m" % (
            math.degrees(math.atan2(rel[0], rel[1])) % 360.0,
            math.degrees(math.atan2(rel[2], math.hypot(rel[0], rel[1]))),
            float(np.linalg.norm(rel))))
        rpy = rpy_from_rotation_enu_body(quat_to_rotation_enu_body(*ents["iris_with_Betaflight"][1]))
        print("转换得到机体姿态 rpy = (%.1f, %.1f, %.1f)" % rpy)
        print("→ 若相机 tilt=20，靶机应成像于 u = %.0f, v = %.0f" % (
            320 + 467.8 * math.tan(math.radians(
                (math.degrees(math.atan2(rel[0], rel[1])) - rpy[2] + 180) % 360 - 180)),
            240 + 467.8 * math.tan(math.radians(
                math.degrees(math.atan2(rel[2], math.hypot(rel[0], rel[1]))) - (rpy[1] + 20.0)))))


if __name__ == "__main__":
    main()
