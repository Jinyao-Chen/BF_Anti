#!/usr/bin/env python3
"""Pitch step-response test for the shell airframe.

Arms, climbs, then applies angle targets through either the firmware ANGLE
loop or the companion P + ACRO rate loop. Logs truth/MSP attitude so the
steady-state gain, rise time and overshoot can be compared.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim"))

from gazebo_target_bridge import GazeboPoseSource  # noqa: E402
from guidance.p_visual_guidance import PStrikeCommand, p_cmd_to_rc  # noqa: E402
from run_intercept_mission import (  # noqa: E402
    MSP_ATTITUDE, RcState, climb_to, rc_sender, spool_up, wait_until_armed,
)

_BF_SCRIPTS = os.path.expanduser("~/betaflight/src/platform/SIMULATOR/gazebo/scripts")
if os.path.isdir(_BF_SCRIPTS):
    sys.path.insert(0, _BF_SCRIPTS)
import offboard_position_example as offboard  # noqa: E402

RC_MID = 1500
HOVER = 1430.0


def read_pitch_deg(link) -> float:
    payload = link.query(MSP_ATTITUDE, is_v2=False)
    if payload and len(payload) >= 6:
        raw = struct.unpack_from("<3h", payload, 0)
        return -float(raw[1]) / 10.0
    return float("nan")


def throttle_for(z: float, vz: float, target_z: float) -> int:
    # Altitude P used only to keep the step test airborne; not the controller
    # under test.
    value = HOVER + 110.0 * (target_z - z) - 35.0 * vz
    return int(max(1200.0, min(1900.0, value)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="betaflight_world")
    ap.add_argument("--interceptor", default="iris_with_Betaflight")
    ap.add_argument("--target", default="target_drone")
    ap.add_argument("--climb-alt", type=float, default=8.0)
    ap.add_argument("--targets", default="-10,-20,-30,-40")
    ap.add_argument("--step-s", type=float, default=2.0)
    ap.add_argument("--settle-s", type=float, default=1.2)
    ap.add_argument("--k-att", type=float, default=8.0)
    ap.add_argument("--rate-max", type=float, default=220.0)
    ap.add_argument("--rate-full", type=float, default=600.0)
    ap.add_argument("--mode", choices=("angle", "rate", "both"), default="both")
    ap.add_argument("--log", default="/tmp/pitch_step.csv")
    args = ap.parse_args()

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rc = RcState()
    running = [True]
    threading.Thread(target=rc_sender, args=(udp, rc, running), daemon=True).start()
    link = offboard.MspLink()
    source = GazeboPoseSource(args.world, args.interceptor, args.target)
    rows: list[list] = []
    targets = [float(x) for x in args.targets.split(",") if x.strip()]
    modes = ("angle", "rate") if args.mode == "both" else (args.mode,)
    prev = [1500] * 16
    try:
        print("arming...", flush=True)
        if not wait_until_armed(link, rc):
            print("ERROR: failed to arm", flush=True)
            return 2
        spool_up(rc, target=HOVER, duration=2.0)
        print(f"climb to {args.climb_alt:.1f} m ...", flush=True)
        if not climb_to(rc, source, link, args.climb_alt, timeout=30.0):
            print("WARNING: climb timeout", flush=True)
        rc.set_aux2(2000)
        rc.set_aux3(1000)
        for mode in modes:
            rc.set_aux2(2000 if mode == "angle" else 1000)
            print(f"=== mode {mode} ===", flush=True)
            # settle at zero
            t_end = time.monotonic() + 1.5
            while time.monotonic() < t_end:
                pose = source.pose()
                z = pose.own_position_enu[2] if pose else args.climb_alt
                vz = pose.own_velocity_enu[2] if pose else 0.0
                cmd = PStrikeCommand(pitch_cmd_deg=0.0)
                ch = p_cmd_to_rc(cmd, angle_limit_deg=60.0, prev=prev,
                                 rate_limit=1000, angle_on=(mode == "angle"))
                ch[2] = throttle_for(z, vz, args.climb_alt)
                ch[4] = 2000
                prev = ch
                rc.set_channels(ch)
                time.sleep(0.05)
            for target in targets:
                t0 = time.monotonic()
                print(f"  step to {target:+.1f} deg", flush=True)
                while time.monotonic() - t0 < args.step_s:
                    pose = source.pose()
                    z = pose.own_position_enu[2] if pose else args.climb_alt
                    vz = pose.own_velocity_enu[2] if pose else 0.0
                    pitch_truth = pose.own_rpy_deg[1] if pose else float("nan")
                    pitch_msp = read_pitch_deg(link)
                    if mode == "angle":
                        cmd = PStrikeCommand(pitch_cmd_deg=target)
                        ch = p_cmd_to_rc(cmd, angle_limit_deg=60.0, prev=prev,
                                         rate_limit=1000, angle_on=True)
                    else:
                        rate_cmd = float(np.clip(args.k_att * (target - pitch_truth),
                                                  -args.rate_max, args.rate_max))
                        cmd = PStrikeCommand(pitch_cmd_deg=target)
                        ch = p_cmd_to_rc(cmd, angle_limit_deg=60.0, prev=prev,
                                         rate_limit=1000, pitch_rate_dps=rate_cmd,
                                         rate_full_dps=args.rate_full, angle_on=False)
                    ch[2] = throttle_for(z, vz, args.climb_alt)
                    ch[4] = 2000
                    prev = ch
                    rc.set_channels(ch)
                    rows.append([mode, time.monotonic() - t0, target, pitch_truth,
                                 pitch_msp, z, vz, rate_cmd if mode == "rate" else float("nan")])
                    time.sleep(0.05)
                # settle back to zero
                t_end = time.monotonic() + args.settle_s
                while time.monotonic() < t_end:
                    pose = source.pose()
                    z = pose.own_position_enu[2] if pose else args.climb_alt
                    vz = pose.own_velocity_enu[2] if pose else 0.0
                    ch = p_cmd_to_rc(PStrikeCommand(pitch_cmd_deg=0.0),
                                     angle_limit_deg=60.0, prev=prev, rate_limit=1000,
                                     angle_on=(mode == "angle"))
                    ch[2] = throttle_for(z, vz, args.climb_alt)
                    ch[4] = 2000
                    prev = ch
                    rc.set_channels(ch)
                    time.sleep(0.05)
    finally:
        rc.set_aux1(1000)
        rc.set_throttle(1000)
        time.sleep(0.5)
        running[0] = False
        udp.close()
        if rows:
            with open(args.log, "w", newline="", encoding="utf-8") as handle:
                w = csv.writer(handle)
                w.writerow(["mode", "t", "target", "pitch_truth", "pitch_msp",
                            "z", "vz", "rate_cmd"])
                w.writerows(rows)
            print(f"wrote {args.log} ({len(rows)} rows)", flush=True)

    # summary: steady-state mean over the final 0.5 s of each step
    for mode in modes:
        for target in targets:
            seg = [r for r in rows if r[0] == mode and r[1] >= args.step_s - 0.5
                   and abs(r[2] - target) < 1e-6]
            if not seg:
                continue
            vals = np.array([r[3] for r in seg if np.isfinite(r[3])])
            if vals.size:
                print(f"  {mode:5s} target={target:+6.1f}  truth={vals.mean():+6.1f}  "
                      f"gain={vals.mean() / target:5.2f}  spread={vals.std():4.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
