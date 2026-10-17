#!/usr/bin/env python3
"""Identify a Betaflight FC over MSP and report its real AUX mode mapping.

Read-only: this script sends only getter commands (no RC writes, no arming).

Usage:
  python3 scripts/fc_identify.py --tcp 127.0.0.1:5761      # SITL
  python3 scripts/fc_identify.py --port /dev/ttyACM0       # real FC USB
  python3 scripts/fc_identify.py --tcp 127.0.0.1:5761 --json --out fc.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from onboard.config import FcConfig  # noqa: E402
from onboard.fc.link import make_link  # noqa: E402
from onboard.fc.mode import discover_aux  # noqa: E402
from msp import (  # noqa: E402
    MSP_CMD,
    decode_arming_flags,
    decode_board_info,
    decode_rc,
    decode_status,
    decode_status_ex,
)


def _parse_tcp(value):
    if not value:
        return None, None
    if ":" in value:
        host, port = value.rsplit(":", 1)
        return host, int(port)
    return value, 5761


def _api_version(payload):
    if not payload or len(payload) < 3:
        return None
    return {"protocol": payload[0], "major": payload[1], "minor": payload[2]}


def _fc_version(payload):
    if not payload or len(payload) < 3:
        return None
    info = {
        "year": 2000 + payload[0],
        "month": payload[1],
        "patch": payload[2],
        "version_string": None,
    }
    if len(payload) >= 4:
        length = payload[3]
        info["version_string"] = bytes(payload[4:4 + length]).decode("ascii", "replace")
    return info


def _query(link, cmd, is_v2=False):
    try:
        return link.query(cmd, is_v2=is_v2, timeout=0.5)
    except TypeError:
        return link.query(cmd, is_v2=is_v2)


def build_report(link):
    report = {"link": {}, "firmware": {}, "status": {}, "mode_discovery": {}}
    api = _query(link, MSP_CMD.API_VERSION)
    report["firmware"]["api_version"] = _api_version(api)
    variant = _query(link, MSP_CMD.FC_VARIANT)
    report["firmware"]["variant"] = (
        bytes(variant[:4]).decode("ascii", "replace") if variant else None
    )
    report["firmware"]["fc_version"] = _fc_version(_query(link, MSP_CMD.FC_VERSION))
    board = _query(link, MSP_CMD.BOARD_INFO)
    report["firmware"]["board"] = decode_board_info(board) if board else None

    status_ex = _query(link, MSP_CMD.GET_STATUS_EX)
    status = decode_status_ex(status_ex) if status_ex else None
    if status is None:
        raw = _query(link, MSP_CMD.GET_STATUS)
        status = decode_status(raw) if raw else None
    if status is not None:
        report["status"] = {
            "armed": status["armed"],
            "flight_mode_flags": status["flight_mode_flags"],
            "arming_flags": status.get("arming_flags"),
            "arming_flags_bf45": decode_arming_flags(status.get("arming_flags") or 0, "bf4.5"),
            "arming_flags_sim2real": decode_arming_flags(status.get("arming_flags") or 0, "sim2real"),
        }

    rc_payload = _query(link, MSP_CMD.GET_RC)
    rc = decode_rc(rc_payload) if rc_payload else None
    report["rc_channels"] = list(rc) if rc else None

    discovery = discover_aux(link, cfg=FcConfig())
    report["mode_discovery"] = discovery.to_dict()
    report["config_snippet"] = discovery.config_snippet()
    return report


def print_human(report):
    fw = report["firmware"]
    print("=== FC identify ===")
    api = fw.get("api_version") or {}
    print("API version : protocol=%s major=%s minor=%s"
          % (api.get("protocol"), api.get("major"), api.get("minor")))
    print("FC variant  : %s" % fw.get("variant"))
    version = fw.get("fc_version") or {}
    print("FC version  : %s-%s.%s.%s (%s)"
          % (version.get("year"), version.get("month"), version.get("patch"),
             version.get("version_string")))
    board = fw.get("board") or {}
    print("Board       : %s (%s) hw=%s"
          % (board.get("board_identifier"), board.get("board_name"),
             board.get("hardware_revision")))
    status = report.get("status") or {}
    print("Armed       : %s  flightModeFlags=0x%X"
          % (status.get("armed"), status.get("flight_mode_flags", 0)))
    print("Arm flags   : 0x%X  bf4.5=%s  sim2real=%s"
          % (status.get("arming_flags") or 0,
             status.get("arming_flags_bf45"), status.get("arming_flags_sim2real")))
    rc = report.get("rc_channels")
    print("RC (AETR)   : %s" % (rc,))

    md = report.get("mode_discovery") or {}
    print("\nMode ranges (box permanent id / AUX channel / PWM range):")
    for item in md.get("mode_ranges", []):
        print("  %-22s perm=%-3s AUX%-2s ch=%-2s %4d..%4d"
              % (item.get("name"), item.get("permanent_id"),
                 int(item.get("aux_index", 0)) + 1, item.get("channel_index"),
                 item.get("start_pwm"), item.get("end_pwm")))
    channels = md.get("channels") or {}
    print("\nResolved channels (0-based full AETR array): %s" % channels)
    if rc:
        for label in ("arm", "angle", "override", "engage"):
            channel = channels.get(label)
            if channel is not None and 0 <= channel < len(rc):
                print("  %-8s ch=%-2d current=%4d" % (label, channel, rc[channel]))
    print("\nConfig snippet:")
    print(json.dumps(report.get("config_snippet"), indent=2))
    warnings = md.get("warnings") or []
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print("  - %s" % warning)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only MSP FC identification and AUX discovery")
    parser.add_argument("--tcp", default="", help="TCP MSP endpoint host:port (SITL, default port 5761)")
    parser.add_argument("--host", default="", help="TCP host (alternative to --tcp)")
    parser.add_argument("--tcp-port", type=int, default=5761)
    parser.add_argument("--port", default="/dev/ttyACM0", help="serial port for a real FC")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--json", action="store_true", help="print the report as JSON only")
    parser.add_argument("--out", default="", help="write the JSON report to this path")
    args = parser.parse_args(argv)

    host, port = _parse_tcp(args.tcp)
    if args.host:
        host, port = args.host, args.tcp_port
    cfg = FcConfig(port=args.port, baud=args.baud,
                   tcp_host=host or "", tcp_port=port or 5761)
    link = make_link(cfg)
    try:
        if not link.connect():
            print("ERROR: FC link unavailable (%s)" % (link.last_error or "connect failed"),
                  file=sys.stderr)
            return 2
        report = build_report(link)
        report["transport"] = (
            "tcp %s:%s" % (host, port) if host else "serial %s@%s" % (args.port, args.baud)
        )
        if args.out:
            out_dir = os.path.dirname(os.path.abspath(args.out))
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
        if args.json:
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print_human(report)
            if args.out:
                print("\nJSON report written to %s" % args.out)
        return 0
    finally:
        link.close()


if __name__ == "__main__":
    raise SystemExit(main())
