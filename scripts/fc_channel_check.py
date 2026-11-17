#!/usr/bin/env python3
"""Read-only Betaflight channel/mode checker for a locally connected FC.

This is the laptop-side companion to scripts/pi5_rc_mode_check.py.  It talks
MSP over USB VCP, never sends MSP_SET_RAW_RC, and never arms anything.

Examples:
    python3 scripts/fc_channel_check.py --list
    python3 scripts/fc_channel_check.py --port /dev/ttyACM0
    python3 scripts/fc_channel_check.py --watch
    python3 scripts/fc_channel_check.py --json

Expected BF_Anti mapping (0-based AETR):
    CH5 / AUX1 = ARM
    CH6 / AUX2 = ANGLE (unused in RATE profile; should be absent)
    CH8 / AUX4 = MSP OVERRIDE / three-position mode
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import serial  # noqa: F401  (imported for the clear error below)
    from serial.tools import list_ports
except ImportError as exc:  # pragma: no cover - host dependency
    raise SystemExit(
        "pyserial is required: python3 -m pip install pyserial\n"
        "(%s)" % exc
    )

from msp import (  # noqa: E402
    ARMING_FLAG_NAMES,
    MSP,
    MSP_CMD,
    decode_board_info,
    decode_boxids,
    decode_boxnames,
    decode_mode_ranges,
    decode_rc,
    decode_status,
    decode_status_ex,
)

CHANNEL_NAMES = ["Roll", "Pitch", "Throttle", "Yaw"] + [
    "AUX%d" % i for i in range(1, 17)
]


def _port_text(p) -> str:
    return " ".join(
        str(x) for x in (p.device, p.description, p.hwid, p.manufacturer) if x
    )


def find_ports() -> list:
    all_ports = list(list_ports.comports())
    # Drop Linux legacy ttyS* noise; keep real USB/VCP devices.
    ports = [
        p for p in all_ports
        if not str(p.device).startswith("/dev/ttyS")
        and str(getattr(p, "hwid", "")).lower() not in ("n/a", "none", "")
    ]
    if not ports:
        ports = all_ports
    # Prefer an obvious STM32/Betaflight VCP over other serial devices.
    def score(p) -> tuple:
        text = _port_text(p).lower()
        return (
            0 if "0483:5740" in text else 1,
            0 if "betaflight" in text else 1,
            0 if "stm32" in text else 1,
            0 if "acm" in text else 1,
            p.device,
        )

    return sorted(ports, key=score)


def autodetect_port() -> str | None:
    ports = find_ports()
    if not ports:
        return None
    p = ports[0]
    text = _port_text(p).lower()
    if any(k in text for k in ("0483:5740", "betaflight", "stm32", "acm")):
        return p.device
    # Last resort: exactly one serial port.
    if len(ports) == 1:
        return ports[0].device
    return None


def open_msp(port: str, baud: int, timeout: float) -> MSP:
    """Open the port with an actionable error before MSP's retry loop."""
    try:
        probe = serial.Serial(port=port, baudrate=baud, timeout=0.1)
        probe.close()
    except serial.SerialException as exc:
        low = str(exc).lower()
        err = getattr(exc, "errno", None)
        if err == errno.EBUSY or "busy" in low:
            print(
                "ERROR: %s is busy. Disconnect/close Betaflight Configurator "
                "first, then rerun this script." % port,
                file=sys.stderr,
            )
        elif err in (errno.EACCES, errno.EPERM) or "permission" in low:
            print(
                "ERROR: permission denied for %s. Add your user to dialout "
                "and re-login: sudo usermod -aG dialout $USER" % port,
                file=sys.stderr,
            )
        else:
            print("ERROR: cannot open %s: %s" % (port, exc), file=sys.stderr)
        raise SystemExit(2)
    return MSP(port=port, baud=baud, retry_timeout=timeout)


def _norm(channel: int, pwm: int) -> float:
    if channel == 2:
        return (pwm - 1000.0) / 1000.0
    return (pwm - 1500.0) / 500.0


def _aux_state(pwm: int) -> str:
    if pwm < 1300:
        return "LOW"
    if pwm > 1700:
        return "HIGH"
    return "MID"


def _fmt_flags(raw) -> str:
    if raw is None:
        return "unknown"
    names = [ARMING_FLAG_NAMES.get(bit, "BIT%d" % bit) for bit in range(32) if raw & (1 << bit)]
    return ", ".join(names) if names else "none"


def read_snapshot(msp: MSP) -> dict:
    fc_variant = msp._send_cmd(MSP_CMD.FC_VARIANT)
    fc_version = msp._send_cmd(MSP_CMD.FC_VERSION)
    board_payload = msp._send_cmd(MSP_CMD.BOARD_INFO)
    status_payload = msp._send_cmd(MSP_CMD.GET_STATUS)
    status_ex_payload = msp._send_cmd(MSP_CMD.GET_STATUS_EX)
    rc_payload = msp._send_cmd(MSP_CMD.GET_RC)
    box_payload = msp._send_cmd(MSP_CMD.BOXNAMES)
    boxids_payload = msp._send_cmd(MSP_CMD.BOXIDS)
    ranges_payload = msp._send_cmd(MSP_CMD.MODE_RANGES)

    status = decode_status(status_payload) or {}
    status_ex = decode_status_ex(status_ex_payload) or {}
    board = decode_board_info(board_payload) or {}
    channels = list(decode_rc(rc_payload, max_channels=16) or ())
    box_names = decode_boxnames(box_payload)
    box_ids = decode_boxids(boxids_payload)
    ranges = decode_mode_ranges(ranges_payload)

    if box_names and len(box_ids) == len(box_names):
        box_by_id = dict(zip(box_ids, box_names))
    else:
        box_by_id = {i: name for i, name in enumerate(box_names)}

    version = "?"
    if fc_variant:
        try:
            version = bytes(fc_variant).decode("ascii", "replace").strip()
        except Exception:
            version = "?"
    if fc_version and len(fc_version) >= 3:
        version += " %d.%d.%d" % (fc_version[0], fc_version[1], fc_version[2])

    arming_flags = status_ex.get("arming_flags")
    flag_names = [ARMING_FLAG_NAMES.get(b, "BIT%d" % b) for b in range(32) if arming_flags is not None and arming_flags & (1 << b)]

    active_ranges = []
    mode_ranges = []
    for r in ranges:
        if not r.get("valid"):
            continue
        idx = int(r["channel_index"])
        pwm = channels[idx] if idx < len(channels) else 1500
        r = dict(r)
        r["name"] = box_by_id.get(r["box_id"], "BOX_%d" % r["box_id"])
        r["pwm"] = pwm
        r["active"] = bool(r["start_pwm"] <= pwm <= r["end_pwm"])
        mode_ranges.append(r)
        if r["active"]:
            active_ranges.append(r)

    active_names = [r["name"] for r in active_ranges]
    return {
        "port": msp.port,
        "firmware": version,
        "board": board,
        "armed": bool(status.get("armed", False)),
        "arming_flags_raw": arming_flags,
        "arming_flags": flag_names,
        "rc": channels,
        "box_names": box_names,
        "mode_ranges": mode_ranges,
        "active_ranges": active_ranges,
        "active_modes": active_names,
        "rxl": "RX_FAILSAFE" in flag_names,
        "angle_active": "ANGLE" in active_names,
        "msp_override_active": any("MSP" in n.upper() and "OVERRIDE" in n.upper() for n in active_names),
        "arm_range_active": "ARM" in active_names,
        "status_ex": status_ex,
    }


def render(snap: dict) -> str:
    lines = []
    board = snap["board"]
    board_name = board.get("board_name") or board.get("target_name") or "?"
    board_name = "".join(ch for ch in str(board_name) if ch.isprintable()).strip() or "?"
    lines.append("FC: %s | %s | %s" % (snap["firmware"], board_name, snap["port"]))
    lines.append(
        "State: %s | RXLOSS=%s | ANGLE=%s | MSP_OVERRIDE=%s"
        % (
            "ARMED" if snap["armed"] else "disarmed",
            "YES" if snap["rxl"] else "no",
            "ACTIVE" if snap["angle_active"] else "off",
            "ACTIVE" if snap["msp_override_active"] else "off",
        )
    )
    lines.append("Arming flags: %s" % (", ".join(snap["arming_flags"]) or "none"))
    lines.append("")
    lines.append("CH   PWM   NORM    NAME       AUX")
    lines.append("--  ----  ------  ---------  ---")
    for i in range(16):
        if i < len(snap["rc"]):
            pwm = int(snap["rc"][i])
            norm = "%+.2f" % _norm(i, pwm)
            pwm_s = "%4d" % pwm
        else:
            pwm_s = "  --"
            norm = "  --  "
        name = CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES) else "CH%d" % (i + 1)
        aux = _aux_state(pwm) if i >= 4 and i < len(snap["rc"]) else ""
        lines.append("%2d  %s  %s  %-9s  %s" % (i + 1, pwm_s, norm, name, aux))

    lines.append("")
    lines.append("Mode ranges:")
    if not snap["mode_ranges"]:
        lines.append("  (none / MSP_MODE_RANGES empty)")
    else:
        for r in snap["mode_ranges"]:
            if not r.get("valid"):
                continue
            name = r.get("name", "?")
            mark = " <-- ACTIVE" if r.get("active") else ""
            lines.append(
                "  %-14s AUX%d (%s)  %4d-%4d  CH%d=%s%s"
                % (
                    name,
                    r["aux_index"] + 1,
                    "CH%d" % (r["channel_index"] + 1),
                    r["start_pwm"],
                    r["end_pwm"],
                    r["channel_index"] + 1,
                    r.get("pwm", "?"),
                    mark,
                )
            )
    lines.append("")
    lines.append("Active modes: %s" % (", ".join(snap["active_modes"]) or "none"))
    if snap["rxl"]:
        lines.append("WARNING: RX_FAILSAFE is set. Check receiver power/bind/UART3 CRSF.")
    if not snap["rc"]:
        lines.append("WARNING: no MSP_RC data; is the FC in MSP mode (not CLI)?")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="read-only Betaflight channel/mode checker")
    ap.add_argument("--port", default=None, help="serial port, e.g. /dev/ttyACM0 or COM3")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--watch", action="store_true", help="refresh until Ctrl-C")
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--seconds", type=float, default=0.0, help="with --watch, stop after N s")
    ap.add_argument("--json", action="store_true", help="one JSON object per snapshot")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--timeout", type=float, default=5.0, help="port open retry timeout")
    args = ap.parse_args(argv)

    if args.list:
        ports = find_ports()
        if not ports:
            print("no serial ports found")
            return 1
        for p in ports:
            print("%-18s %s" % (p.device, _port_text(p)))
        return 0

    port = args.port or autodetect_port()
    if not port:
        print("no FC serial port found; use --list then --port", file=sys.stderr)
        return 2

    msp = None
    try:
        msp = open_msp(port=port, baud=args.baud, timeout=args.timeout)
        print("read-only: no RC frames are sent, nothing is armed", file=sys.stderr)
        end = time.monotonic() + args.seconds if args.seconds > 0 else None
        first = True
        while True:
            snap = read_snapshot(msp)
            snap["timestamp"] = time.time()
            if args.json:
                print(json.dumps(snap, ensure_ascii=False, default=str))
            else:
                if args.watch and first and sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                if args.watch and not first and sys.stdout.isatty():
                    print("\033[H\033[J", end="")
                print(render(snap))
                print()
            first = False
            if not args.watch:
                break
            if end is not None and time.monotonic() >= end:
                break
            time.sleep(max(0.05, args.interval))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 1
    finally:
        if msp is not None:
            try:
                msp.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
