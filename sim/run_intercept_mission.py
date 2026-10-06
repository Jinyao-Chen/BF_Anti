#!/usr/bin/env python3
"""BF_Anti 固定靶机拦截任务执行器。

前置条件
--------
Gazebo + Betaflight SITL 已由 ``sim/launch_sitl.sh`` 启动，并使用
``sim/config/sitl_gazebo_tracking.txt``（ANGLE 模式、手动油门、无 ALT/POS HOLD）。

任务流程
--------
1. 50 Hz RC 线程持续发送 UDP RC，保证飞控不触发 RXLOSS；
2. 按官方 arming 流程拉起 AUX1；
3. 闭环爬升到释放高度；
4. 固定俯冲搜索角，等待 44° 上仰相机捕获目标；
5. 捕获后切入制导 bridge，发送 RPYT；
6. 命中判定或超时后自动收油门、拉低 AUX1。
"""

from __future__ import annotations

import argparse
import csv
import os
import socket
import struct
import sys
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import (  # noqa: E402
    GazeboPoseSource,
    InterceptBridge,
    PoseState,
    _rc_frame_to_udp_frame,
)

# 复用 release260904 的 MSP arming 工具，避免重复实现 Betaflight 的 latch 逻辑。
_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402

UDP_IP = "127.0.0.1"
UDP_PORT = 9004
RC_MIN, RC_MAX, RC_MID = 1000, 2000, 1500
THROTTLE_MIN = 1000

# MSP2_BETAFLIGHT_POSITION_STATE (see release260904 README section 9).
MSP2_POSITION_STATE = 0x3015
# Classic MSP_ATTITUDE: roll/pitch/yaw in 0.1 deg.
MSP_ATTITUDE = 108


class RcState:
    """线程安全的 RC 通道状态。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._channels = [RC_MID] * 16
        self._channels[2] = THROTTLE_MIN
        self._channels[4] = 1000  # AUX1 / ARM low
        self._channels[5] = 2000  # AUX2 / ANGLE high

    def snapshot(self) -> List[int]:
        with self._lock:
            return list(self._channels)

    def set_channels(self, channels: List[int]) -> None:
        with self._lock:
            self._channels = [int(max(RC_MIN, min(RC_MAX, c))) for c in channels[:16]]
            while len(self._channels) < 16:
                self._channels.append(RC_MID)

    def set_aux1(self, value: int) -> None:
        with self._lock:
            self._channels[4] = int(value)

    def set_aux2(self, value: int) -> None:
        with self._lock:
            self._channels[5] = int(value)

    def set_aux3(self, value: int) -> None:
        with self._lock:
            self._channels[6] = int(value)

    def set_throttle(self, value: int) -> None:
        with self._lock:
            self._channels[2] = int(value)

    def set_rpyt(self, roll: int, pitch: int, throttle: int, yaw: int) -> None:
        with self._lock:
            self._channels[0] = int(roll)
            self._channels[1] = int(pitch)
            self._channels[2] = int(throttle)
            self._channels[3] = int(yaw)

    def idle(self) -> None:
        """收尾：回中并收油/上锁，供所有 RC 脚本在 finally 中安全调用。"""
        with self._lock:
            self._channels = [RC_MID] * 16
            self._channels[2] = THROTTLE_MIN
            self._channels[4] = 1000  # AUX1 / ARM low
            self._channels[5] = 2000  # AUX2 / ANGLE high
            self._channels[6] = 1000  # AUX3 / ALT_HOLD low


def rc_sender(udp: socket.socket, state: RcState, running: List[bool]) -> None:
    """50 Hz 持续发送 RC，避免 SITL 的 RXLOSS。"""
    while running[0]:
        channels = state.snapshot()
        udp.sendto(
            struct.pack("<d" + "H" * 16, time.time(), *channels),
            (UDP_IP, UDP_PORT),
        )
        time.sleep(0.02)


def read_truth_pose(link, source: GazeboPoseSource) -> Optional[PoseState]:
    """从 MSP 真值状态读拦截机 ENU 位姿，从 Gazebo 读固定靶机位置。

    为什么不用 Gazebo 的 ``iris_with_Betaflight`` 位姿：该模型包含嵌套
    ``iris_with_standoffs``，``pose/info`` 里嵌套模型是相对父模型的坐标，
    直接当世界高度用会得到恒定 0.2 m 的假值。固件侧既然已经注入了
    fdm 真值，就直接通过 MSP2 读同一份状态，保证控制与评估一致。
    """
    gz_pose = source.pose()
    if gz_pose is None:
        return None

    pos_payload = link.query(MSP2_POSITION_STATE, is_v2=True)
    if pos_payload and len(pos_payload) >= 27:
        # u8 active, u8 reached, u8 offboard_velocity_active, then 3 floats pos.
        pos = np.array(struct.unpack_from("<3f", pos_payload, 3), dtype=float)
        vel = np.array(struct.unpack_from("<3f", pos_payload, 15), dtype=float)
    else:
        # 早期固件没有 0x3015 时退回 Gazebo 顶层模型位姿（仅用于定位问题）。
        pos = gz_pose.own_position_enu
        vel = gz_pose.own_velocity_enu

    att_payload = link.query(MSP_ATTITUDE, is_v2=False)
    if att_payload and len(att_payload) >= 6:
        raw = struct.unpack_from("<3h", att_payload, 0)
        # Betaflight/MSP 的 pitch 正方向是低头（见 imu.c: negative is backwards）；
        # 相机几何与所有 guidance 统一用航空航天约定：抬头为正。这里做一次轴翻转，
        # 避免把“低头”误当“抬头”，否则 44° 上仰相机的投影会随姿态发散。
        rpy = (float(raw[0]) / 10.0, -float(raw[1]) / 10.0, float(raw[2]) / 10.0)
    else:
        rpy = gz_pose.own_rpy_deg

    return PoseState(
        own_position_enu=pos,
        own_velocity_enu=vel,
        own_rpy_deg=rpy,
        target_position_enu=gz_pose.target_position_enu,
        target_velocity_enu=gz_pose.target_velocity_enu,
        timestamp=time.monotonic(),
    )


def wait_until_armed(link, rc: RcState, timeout: float = 40.0) -> bool:
    """按 Betaflight latch 规则解锁：AUX1 低 -> 等待无其他 arming block -> 拉高。"""
    rc.set_aux1(1000)
    rc.set_throttle(THROTTLE_MIN)
    deadline = time.time() + timeout
    ready = False
    while time.time() < deadline:
        status = offboard.msp_status(link)
        if status is not None:
            armed, flags = status
            if armed:
                return True
            if flags is not None and (flags & ~offboard.ARMING_DISABLED_ARM_SWITCH) == 0:
                channels = offboard.read_rc_channels(link)
                if channels is not None and channels[4] <= 1050:
                    ready = True
                    break
        time.sleep(0.2)
    if not ready:
        return False

    for _ in range(5):
        rc.set_aux1(2000)
        t0 = time.time()
        while time.time() - t0 < 4.0:
            status = offboard.msp_status(link)
            if status is not None:
                armed, flags = status
                if armed:
                    return True
                if flags is not None and (flags & offboard.ARMING_DISABLED_ARM_SWITCH):
                    break
            time.sleep(0.2)
        rc.set_aux1(1000)
        time.sleep(0.6)
    return False


def debug_monitor(link, running: List[bool], interval: float = 1.0) -> None:
    """后台打印 MSP 状态与电机输出，用于确认 RC 指令是否真的进入混控。"""
    while running[0]:
        status = offboard.msp_status(link)
        motor_payload = link.query(104, is_v2=False)  # MSP_MOTOR
        msg = "DBG "
        if status is None:
            msg += "status=none "
        else:
            armed, flags = status
            msg += f"armed={armed} flags={flags} "
        if motor_payload:
            count = len(motor_payload) // 2
            motors = struct.unpack("<" + "H" * count, motor_payload[: count * 2])
            msg += f"motors={motors[:4]}"
        else:
            msg += "motors=none"
        print(msg, flush=True)
        time.sleep(interval)


def spool_up(rc: RcState, target: int = 1545, duration: float = 2.5) -> None:
    """解锁后缓慢把油门推到悬停，避免起飞瞬间翻覆。"""
    t0 = time.time()
    while True:
        k = min(1.0, (time.time() - t0) / duration)
        rc.set_throttle(int(THROTTLE_MIN + (target - THROTTLE_MIN) * k))
        if k >= 1.0:
            break
        time.sleep(0.02)


def climb_to(
    rc: RcState,
    source: GazeboPoseSource,
    link,
    target_alt: float,
    timeout: float = 25.0,
) -> bool:
    """闭环爬升：只调总距，姿态保持水平。"""
    start = time.time()
    while time.time() - start < timeout:
        pose = read_truth_pose(link, source)
        if pose is None:
            time.sleep(0.05)
            continue
        alt = float(pose.own_position_enu[2])
        if alt >= target_alt:
            return True
        # 简单的真值高度 P 控制器：约 120 RC / m，限幅 1545~1750。
        throttle = int(max(1545, min(1750, 1545 + 120.0 * (target_alt - alt))))
        rc.set_rpyt(RC_MID, RC_MID, throttle, RC_MID)
        time.sleep(0.02)
    return False


def search_for_target(
    rc: RcState,
    bridge: InterceptBridge,
    source: GazeboPoseSource,
    link,
    pitch_deg: float,
    max_angle_deg: float,
    throttle: int,
    timeout: float,
) -> Tuple[Optional[PoseState], Optional[object]]:
    """固定搜索俯仰角，等待 44° 相机把目标投射到画面内。"""
    # pitch_deg 为航空气动约定（抬头正）；Betaflight ANGLE 通道正方向是低头，
    # 所以 RC 增量取反。
    pitch_rc = int(max(RC_MIN, min(RC_MAX, 1500.0 - pitch_deg * 500.0 / max_angle_deg)))
    start = time.time()
    while time.time() - start < timeout:
        pose = read_truth_pose(link, source)
        if pose is None:
            time.sleep(0.05)
            continue
        rc.set_rpyt(RC_MID, pitch_rc, throttle, RC_MID)
        target = bridge.target_state(pose)
        if target is not None and target.valid:
            return pose, target
        time.sleep(0.02)
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="BF_Anti 固定靶机拦截任务执行器")
    parser.add_argument("--world", default="betaflight_world")
    # 顶层 iris_with_Betaflight 的 pose 是世界坐标；嵌套 iris_with_standoffs
    # 在 pose/info 中是相对父模型的坐标，不能直接当世界高度用。
    parser.add_argument("--interceptor", default="iris_with_Betaflight")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--tilt", type=float, default=44.0)
    parser.add_argument("--target-width", type=float, default=1.0)
    parser.add_argument("--release-alt", type=float, default=20.0)
    parser.add_argument("--search-pitch", type=float, default=-60.0)
    parser.add_argument("--search-timeout", type=float, default=12.0)
    parser.add_argument("--max-angle", type=float, default=60.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--perception", choices=("camera", "truth"), default="camera")
    parser.add_argument("--debug", action="store_true", help="每秒打印 MSP 状态和电机输出")
    parser.add_argument("--log", default="")
    args = parser.parse_args()

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = RcState()
    running = [True]
    sender = threading.Thread(target=rc_sender, args=(udp, rc, running), daemon=True)
    sender.start()

    link = offboard.MspLink()
    bridge = InterceptBridge(
        camera_tilt_deg=args.tilt,
        target_width_m=args.target_width,
        max_angle_deg=args.max_angle,
        use_camera=(args.perception == "camera"),
    )
    bridge.guidance.min_safe_altitude_m = 0.0  # 固定靶机任务允许终端贴地
    source = GazeboPoseSource(args.world, args.interceptor, args.target)

    log_rows = []
    hit = False
    min_range = float("inf")
    phase_history = []
    try:
        print("arming...", flush=True)
        if not wait_until_armed(link, rc):
            print("ERROR: failed to arm", flush=True)
            return 2
        print("armed; spooling up", flush=True)
        if args.debug:
            threading.Thread(
                target=debug_monitor, args=(link, running, 1.0), daemon=True
            ).start()
        spool_up(rc, target=1545, duration=2.5)

        print(f"climbing to {args.release_alt:.1f} m ...", flush=True)
        if not climb_to(rc, source, link, args.release_alt, timeout=25.0):
            print("WARNING: climb timeout; continuing to search", flush=True)
        print("searching with fixed camera dive angle...", flush=True)
        pose, target = search_for_target(
            rc,
            bridge,
            source,
            link,
            pitch_deg=args.search_pitch,
            max_angle_deg=args.max_angle,
            throttle=1650,
            timeout=args.search_timeout,
        )
        if target is None:
            print("ERROR: target not acquired before timeout", flush=True)
            return 3
        print(f"target acquired: range={target.range_m:.2f} m", flush=True)

        start = time.monotonic()
        period = 1.0 / max(1.0, args.rate)
        while time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            pose = read_truth_pose(link, source)
            if pose is None:
                time.sleep(period)
                continue
            cmd, frame = bridge.step(pose)
            udp_frame = _rc_frame_to_udp_frame(frame, aux1=2000, aux2=2000)
            rc.set_channels(udp_frame)
            min_range = min(min_range, float(cmd.target_range_m))
            if not phase_history or phase_history[-1] != cmd.phase.value:
                phase_history.append(cmd.phase.value)
            log_rows.append(
                (
                    time.monotonic() - start,
                    cmd.phase.value,
                    cmd.target_range_m,
                    pose.own_position_enu[2],
                    cmd.az_deg,
                    cmd.el_deg,
                    cmd.pitch_cmd_deg,
                    cmd.throttle_cmd,
                    cmd.hit,
                )
            )
            print(
                f"t={time.monotonic() - start:5.2f} {cmd.phase.value:<8} "
                f"range={cmd.target_range_m:6.2f}m alt={pose.own_position_enu[2]:5.2f}m "
                f"az={cmd.az_deg:6.1f} el={cmd.el_deg:6.1f} pitch={cmd.pitch_cmd_deg:6.1f} "
                f"thr={cmd.throttle_cmd:4.2f} hit={cmd.hit}",
                flush=True,
            )
            if cmd.hit:
                hit = True
                break
            time.sleep(max(0.0, period - (time.monotonic() - loop_start)))
    finally:
        rc.set_throttle(THROTTLE_MIN)
        rc.set_aux1(1000)
        time.sleep(1.5)
        running[0] = False
        sender.join(timeout=1.0)
        udp.close()
        if args.log and log_rows:
            with open(args.log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["t", "phase", "range_m", "alt_m", "az_deg", "el_deg", "pitch_cmd_deg", "throttle_cmd", "hit"]
                )
                writer.writerows(log_rows)
            print(f"wrote {args.log}", flush=True)

    print("=== intercept mission summary ===")
    print(f"hit        : {hit}")
    print(f"min range  : {min_range:.3f} m")
    print(f"phases     : {' -> '.join(phase_history)}")
    return 0 if hit else 1


if __name__ == "__main__":
    raise SystemExit(main())
