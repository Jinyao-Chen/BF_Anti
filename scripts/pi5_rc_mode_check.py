#!/usr/bin/env python3
"""Read-only RC/mode check for the three-position pilot switch.

Run on the Pi with the FC connected and the transmitter on.  It never sends an
RC frame and never arms anything: it only reads MSP_RC and prints the channel
values the companion would see.

    .venv/bin/python scripts/pi5_rc_mode_check.py --config config/onboard.json

Expected mapping (0-based AETR):
    ch4 AUX1 = ARM        low=disarmed, high=armed
    ch5 AUX2 = ANGLE      high=ANGLE (ANGLE profile) / low=ACRO (RATE profile)
    ch7 AUX4 = MODE       low=FORCE MANUAL + MSP OVERRIDE off (hardware
                          CRSF bypass), mid=AUTO, high=STRIKE
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onboard.config import load_config
from onboard.fc.link import make_link
from onboard.fc.state import FcStateEstimator


def mode_of(pwm: float) -> str:
    if pwm < 1300:
        return "FORCE_MANUAL"
    if pwm > 1700:
        return "STRIKE"
    return "AUTO"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="read-only RC/mode monitor")
    ap.add_argument("--config", default="config/onboard.json")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = until Ctrl-C")
    ap.add_argument("--interval", type=float, default=0.2)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    link = make_link(cfg.fc)
    state = FcStateEstimator(link, cfg.fc)
    state.start()
    print("read-only: no RC frames are sent, nothing is armed")
    print("ch0-3 R/P/T/Y | ch4 ARM | ch5 ANGLE | ch7 MODE/OVERRIDE")
    end = time.monotonic() + args.seconds if args.seconds > 0 else None
    last = 0.0
    try:
        while end is None or time.monotonic() < end:
            now = time.monotonic()
            if now - last < args.interval:
                time.sleep(0.01)
                continue
            last = now
            st = state.snapshot(now)
            rc = st.rc
            if not rc:
                print("waiting for MSP_RC ...", end="\r", flush=True)
                continue
            ch = list(rc) + [1500] * (16 - len(rc))
            mode = mode_of(ch[cfg.fc.aux_engage])
            armed = "ARMED" if ch[cfg.fc.aux_arm] > 1700 else "disarmed"
            if cfg.fc.override_from_mode:
                ovr = "OVR" if ch[cfg.fc.aux_engage] >= 1300 else "no-ovr"
            else:
                ovr = "OVR" if ch[cfg.fc.aux_override] > 1700 else "no-ovr"
            print(
                "R%4d P%4d T%4d Y%4d | %-8s %-6s | A1%4d A2%4d A3%4d A4%4d | %s"
                % (ch[0], ch[1], ch[2], ch[3], armed, ovr,
                   ch[4], ch[5], ch[6], ch[7], mode),
                end="\r", flush=True,
            )
    except KeyboardInterrupt:
        pass
    finally:
        print()
        state.stop()
        try:
            link.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
