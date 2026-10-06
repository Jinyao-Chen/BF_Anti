#!/usr/bin/env python3
"""Direct-RC axis response test for normal vs force-only worlds.

Arms, climbs to a safe altitude with a simple RC altitude P loop, then applies
a fixed roll/pitch/yaw stick for a short time while logging MSP_ATTITUDE.
Used to compare the sign of the roll/pitch/yaw response between the known-good
LiftDrag world and the RotorForcePlugin force world.

Usage:
    python3 sim/rc_axis_response_test.py --axis roll --value 1600 \
        --climb-alt 2 --throttle 1700 --duration 2 \
        --log /tmp/rc_roll_force.csv
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
from typing import Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import GazeboPoseSource  # noqa: E402
from run_intercept_mission import (
    MSP_ATTITUDE,
    RcState,
    climb_to,
    rc_sender,
    spool_up,
    wait_until_armed,
)  # noqa: E402

_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402

RC_MIN, RC_MAX, RC_MID = 1000, 2000, 1500


def read_attitude(link) -> Tuple[float, float, float]:
    payload = link.query(MSP_ATTITUDE, is_v2=False)
    if payload and len(payload) >= 6:
        raw = struct.unpack_from("<3h", payload, 0)
        return tuple(float(v) / 10.0 for v in raw)  # type: ignore[return-value]
    return (float("nan"),) * 3  # type: ignore[return-value]


def run(args: argparse.Namespace) -> int:
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = RcState()
    running = [True]
    threading.Thread(target=rc_sender, args=(udp, rc, running), daemon=True).start()
    link = offboard.MspLink()
    source = GazeboPoseSource(args.world, args.interceptor, args.target)
    rows: list[list] = []
    try:
        print("arming...", flush=True)
        if not wait_until_armed(link, rc):
            print("ERROR: failed to arm", flush=True)
            return 2
        print("spooling up...", flush=True)
        spool_up(rc, target=args.throttle, duration=2.0)
        print(f"climbing to {args.climb_alt:.1f} m...", flush=True)
        if not climb_to(rc, source, link, args.climb_alt, timeout=20.0):
            print("WARNING: climb timeout; continuing", flush=True)
        # hover at fixed throttle
        rc.set_rpyt(RC_MID, RC_MID, args.throttle, RC_MID)
        time.sleep(1.0)
        before = read_attitude(link)
        print(f"before attitude roll/pitch/yaw = {before}", flush=True)

        start = time.monotonic()
        while time.monotonic() - start < args.duration:
            roll = args.value if args.axis == "roll" else RC_MID
            pitch = args.value if args.axis == "pitch" else RC_MID
            yaw = args.value if args.axis == "yaw" else RC_MID
            rc.set_rpyt(roll, pitch, args.throttle, yaw)
            att = read_attitude(link)
            rows.append([time.monotonic() - start, *att])
            time.sleep(0.05)
        after = read_attitude(link)
        print(f"after  attitude roll/pitch/yaw = {after}", flush=True)
        delta = tuple(after[i] - before[i] for i in range(3))
        print(f"delta roll/pitch/yaw = {delta}", flush=True)
        return 0
    finally:
        rc.set_throttle(1000)
        rc.set_aux1(1000)
        time.sleep(0.5)
        running[0] = False
        udp.close()
        if rows:
            with open(args.log, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["t", "roll", "pitch", "yaw"])
                writer.writerows(rows)
            print(f"wrote {args.log}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="RC axis response test")
    parser.add_argument("--world", default="betaflight_world")
    parser.add_argument("--interceptor", default="iris_with_Betaflight")
    parser.add_argument("--target", default="target_drone")
    parser.add_argument("--axis", choices=("roll", "pitch", "yaw"), default="roll")
    parser.add_argument("--value", type=int, default=1600)
    parser.add_argument("--climb-alt", type=float, default=2.0)
    parser.add_argument("--throttle", type=int, default=1700)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--log", default="/tmp/rc_axis_response.csv")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
