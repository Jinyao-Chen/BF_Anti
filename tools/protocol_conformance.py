#!/usr/bin/env python3
"""MSP protocol conformance checker for BF_Anti sim2real.

Runs against Betaflight SITL (TCP, default 127.0.0.1:5761) or a real FC
(serial).  Read-only except for the opt-in RC round-trip (--allow-rc), which
refuses to run if the FC reports ARMED.

Every run writes a JSON report (default:
docs/sim2real/protocol_reports/<utc>-<transport>.json).  The report is evidence
about MSP framing/telemetry only: it is never a flight-clearance or hardware
qualification.  Status values: pass / fail / skip / blocked.

Usage:
  python3 tools/protocol_conformance.py --help
  python3 tools/protocol_conformance.py --offline
  python3 tools/protocol_conformance.py --host 127.0.0.1 --tcp-port 5761
  python3 tools/protocol_conformance.py --port /dev/ttyACM0 --baud 115200
  python3 tools/protocol_conformance.py --host 127.0.0.1 --allow-rc   # props OFF
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import struct
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from msp import (  # noqa: E402
    MSP_CMD,
    FrameParser,
    build_frame,
    build_frame_v1,
    build_frame_v2,
    crc8_dvb_s2,
    decode_altitude,
    decode_analog,
    decode_arming_flags,
    decode_attitude,
    decode_board_info,
    decode_msp2_position_state,
    decode_rc,
    decode_status,
    decode_status_ex,
)
from onboard.config import FcConfig  # noqa: E402
from onboard.fc.link import make_link  # noqa: E402
from onboard.fc.mode import discover_aux  # noqa: E402

TOOL_VERSION = "1.0"
PASS, FAIL, SKIP, BLOCKED = "pass", "fail", "skip", "blocked"


def _check(checks, check_id, status, detail="", data=None, required=True):
    checks.append({
        "id": check_id,
        "status": status,
        "required": bool(required),
        "detail": detail,
        "data": data,
    })


# ---- offline checks ---------------------------------------------------------

def run_framing_checks(checks):
    ok = True
    detail = []
    try:
        vector = crc8_dvb_s2(b"123456789")
        if vector != 0xBC:
            ok = False
            detail.append("CRC known-answer failed: got 0x%02X expected 0xBC" % vector)
        else:
            detail.append("CRC-8/DVB-S2 check vector OK (0xBC)")

        for is_v2 in (False, True):
            payload = bytes(range(1, 19))
            # Build as a reply frame (">") so the round-trip also checks the
            # parser's direction byte, which is what link.query() matches on.
            frame = build_frame(0x3015 if is_v2 else 108, payload, is_v2=is_v2,
                                direction=">")
            parser = FrameParser()
            decoded = []
            noise = b"\x00\xff$Z" + frame
            for byte in noise:
                decoded.extend(parser.feed(bytes([byte])))
            if len(decoded) != 1:
                ok = False
                detail.append("v2=%s chunked parse returned %d frames" % (is_v2, len(decoded)))
                continue
            got = decoded[0]
            if (got.cmd, got.payload, got.is_v2, got.direction) != (
                    0x3015 if is_v2 else 108, payload, is_v2, ">"):
                ok = False
                detail.append("v2=%s decoded mismatch" % is_v2)
            else:
                detail.append("v2=%s frame round-trip OK (%d B)" % (is_v2, len(frame)))

            corrupt = bytearray(frame)
            corrupt[-1] ^= 0xFF
            parser = FrameParser()
            if parser.feed(bytes(corrupt)):
                ok = False
                detail.append("v2=%s corrupt CRC was accepted" % is_v2)

        # v1 jumbo response (BOXNAMES page > 255 B) must decode with the
        # logical command preserved.
        big_payload = bytes(range(256)) * 2
        jumbo = b"$M>" + bytes([255, MSP_CMD.BOXNAMES]) + struct.pack("<H", len(big_payload)) + big_payload
        crc = 0
        for byte in jumbo[3:]:
            crc ^= byte
        jumbo += bytes([crc & 0xFF])
        parsed = FrameParser().feed(jumbo)
        if len(parsed) != 1 or parsed[0].payload != big_payload:
            ok = False
            detail.append("v1 jumbo frame decode failed")
        else:
            detail.append("v1 jumbo frame decode OK (%d B)" % len(big_payload))

        # Error frame parsing.
        err = build_frame_v2(MSP_CMD.API_VERSION, b"", error=True)
        parsed = FrameParser().feed(err)
        if len(parsed) != 1 or not parsed[0].error:
            ok = False
            detail.append("error frame decode failed")
        else:
            detail.append("v2 error frame decode OK")
    except Exception as exc:  # pragma: no cover - defensive
        ok = False
        detail.append("exception: %r" % (exc,))
    _check(checks, "framing_crc", PASS if ok else FAIL, "; ".join(detail),
           {"crc_known_answer": 0xBC})


# ---- device checks ----------------------------------------------------------

def _q(link, cmd, is_v2=False, timeout=0.5):
    return link.query(cmd, is_v2=is_v2, timeout=timeout)


def check_api(checks, link):
    payload = _q(link, MSP_CMD.API_VERSION)
    if not payload or len(payload) < 3:
        _check(checks, "api_version", FAIL, "no/short MSP_API_VERSION reply")
        return None
    info = {"protocol": payload[0], "major": payload[1], "minor": payload[2]}
    status = PASS if info["major"] >= 1 else FAIL
    _check(checks, "api_version", status,
           "protocol=%s api=%s.%s" % (info["protocol"], info["major"], info["minor"]), info)
    return info


def check_variant_version(checks, link):
    variant = _q(link, MSP_CMD.FC_VARIANT)
    text = bytes(variant[:4]).decode("ascii", "replace") if variant else None
    _check(checks, "fc_variant", PASS if text and text.isprintable() else FAIL,
           "variant=%r" % text, {"variant": text})

    payload = _q(link, MSP_CMD.FC_VERSION)
    if not payload or len(payload) < 4:
        _check(checks, "fc_version", FAIL, "no/short MSP_FC_VERSION reply")
        return text, None
    year = 2000 + payload[0]
    version_string = None
    if len(payload) >= 5:
        length = payload[3]
        version_string = bytes(payload[4:4 + length]).decode("ascii", "replace")
    info = {"year": year, "month": payload[1], "patch": payload[2],
            "version_string": version_string}
    plausible = 2015 <= year <= 2100 and 1 <= payload[1] <= 12
    _check(checks, "fc_version", PASS if plausible else FAIL,
           "%s.%s.%s %s" % (year, payload[1], payload[2], version_string), info)
    return text, info


def check_board(checks, link):
    payload = _q(link, MSP_CMD.BOARD_INFO)
    info = decode_board_info(payload) if payload else None
    if info is None:
        _check(checks, "board_info", FAIL, "no/short MSP_BOARD_INFO reply")
        return None
    printable = all(32 <= ord(ch) < 127 for ch in info["board_identifier"])
    _check(checks, "board_info", PASS if printable else FAIL,
           "board=%s name=%s target=%s hw=%s"
           % (info["board_identifier"], info.get("board_name"),
              info.get("target_name"), info["hardware_revision"]), info)
    return info


def _status_payload(link):
    payload = _q(link, MSP_CMD.GET_STATUS_EX)
    decoded = decode_status_ex(payload) if payload else None
    if decoded is None:
        payload = _q(link, MSP_CMD.GET_STATUS)
        decoded = decode_status(payload) if payload else None
    return decoded


def check_status(checks, link, allow_rc):
    status = _status_payload(link)
    if status is None:
        _check(checks, "status_armed", FAIL, "no MSP_STATUS/STATUS_EX reply")
        return None
    arming = status.get("arming_flags")
    data = {
        "armed": status["armed"],
        "flight_mode_flags": status["flight_mode_flags"],
        "sensors": status["sensors"],
        "arming_flags": arming,
        "arming_flag_count": status.get("arming_flag_count"),
        "arming_flags_bf45": decode_arming_flags(arming or 0, "bf4.5"),
        "arming_flags_sim2real": decode_arming_flags(arming or 0, "sim2real"),
        "config_state": status.get("config_state"),
    }
    _check(checks, "status_armed", PASS,
           "armed=%s flightModeFlags=0x%X armingFlags=%s"
           % (status["armed"], status["flight_mode_flags"],
              "0x%X" % arming if arming is not None else "unparsed"), data)
    if not allow_rc:
        _check(checks, "rc_roundtrip", SKIP,
               "requires --allow-rc (bench only, props removed)", None, required=False)
    return status


def check_attitude(checks, link, samples=5):
    rows = []
    ok = True
    for _ in range(max(1, samples)):
        payload = _q(link, MSP_CMD.GET_ATTITUDE)
        attitude = decode_attitude(payload) if payload else None
        if attitude is None:
            ok = False
            rows.append(None)
            continue
        rows.append([round(attitude[0], 2), round(attitude[1], 2), round(attitude[2], 2)])
        # Yaw is a signed int16 on the wire (e.g. -14 => heading 346).
        if not (-180.0 <= attitude[0] <= 180.0 and -90.0 <= attitude[1] <= 90.0
                and -180.0 <= attitude[2] <= 360.0):
            ok = False
        time.sleep(0.02)
    _check(checks, "attitude_units",
           PASS if ok and all(row is not None for row in rows) else FAIL,
           "roll/pitch=0.1deg, yaw=degrees; samples=%s" % rows,
           {"samples": rows,
            "yaw_unit_expected": "degrees",
            "roll_pitch_unit_expected": "0.1deg"})
    _check(checks, "attitude_sign", BLOCKED,
           "sign/axis verification requires physically tilting the FC; not performed",
           None, required=False)


def check_altitude(checks, link):
    payload = _q(link, MSP_CMD.GET_ALTITUDE)
    decoded = decode_altitude(payload) if payload else None
    if decoded is None:
        _check(checks, "altitude", FAIL, "no/short MSP_ALTITUDE reply")
        return None
    altitude_m, vario_mps = decoded
    plausible = -500.0 <= altitude_m <= 10000.0 and -100.0 <= vario_mps <= 100.0
    _check(checks, "altitude", PASS if plausible else FAIL,
           "alt=%.2f m vario=%.2f m/s (u32 cm + i16 cm/s)" % (altitude_m, vario_mps),
           {"altitude_m": altitude_m, "vario_mps": vario_mps})
    return decoded


def check_analog(checks, link):
    payload = _q(link, MSP_CMD.GET_ANALOG)
    decoded = decode_analog(payload) if payload else None
    if decoded is None:
        _check(checks, "analog", FAIL, "no/short MSP_ANALOG reply")
        return None
    plausible = (0.0 <= decoded["voltage_v"] <= 60.0
                 and -320.0 <= decoded["current_a"] <= 320.0
                 and 0 <= decoded["mah"] <= 65535)
    detail = "len=%d layout=%s V=%.2f I=%.2f mAh=%d rssi=%d" % (
        len(payload), decoded["layout"], decoded["voltage_v"],
        decoded["current_a"], decoded["mah"], decoded["rssi"])
    _check(checks, "analog", PASS if plausible else FAIL, detail,
           dict(decoded, payload_len=len(payload)))
    return decoded


def check_rc_roundtrip(checks, link, allow_rc, status):
    if not allow_rc:
        return
    if status is None or status.get("armed"):
        _check(checks, "rc_roundtrip", BLOCKED,
               "FC reports ARMED or status unknown; refusing to inject RC", None,
               required=False)
        return
    # Safe bench pattern: throttle low, AUX1 low (disarm), small offsets on
    # channels that are normally mid so no mode threshold (1300/1700) trips.
    sent = [1480, 1520, 1000, 1540,
            1000, 1500, 1500, 1500,
            1520, 1480, 1500, 1500,
            1510, 1490, 1505, 1495]
    link.send_rc_16(sent)
    for _ in range(10):
        time.sleep(0.02)
        link.send_rc_16(sent)
    payload = _q(link, MSP_CMD.GET_RC)
    # Leave the FC disarmed on the way out.
    link.send_rc_16([1500, 1500, 1000, 1500] + [1000, 1500, 1500, 1500] + [1500] * 8)
    if payload is None:
        _check(checks, "rc_roundtrip", FAIL, "no MSP_RC reply after SET_RAW_RC")
        return
    if len(payload) == 0:
        _check(checks, "rc_roundtrip", BLOCKED,
               "MSP_RC returned an empty payload: the FC receiver exposes 0 "
               "channels (e.g. SITL RX_UDP), so the 16ch round-trip cannot be "
               "verified on this config", None, required=False)
        return
    received = decode_rc(payload)
    if received is None:
        _check(checks, "rc_roundtrip", FAIL,
               "MSP_RC reply too short for channels (%d B)" % len(payload))
        return
    diffs = [int(abs(a - b)) for a, b in zip(sent, received)]
    max_diff = max(diffs) if diffs else 0
    data = {"sent_aetr": sent, "received_aetr": list(received), "max_diff": max_diff,
            "channel_count": len(received)}
    if max_diff <= 25:
        _check(checks, "rc_roundtrip", PASS,
               "16-channel SET_RAW_RC round-trip OK (max diff %d)" % max_diff, data)
    else:
        _check(checks, "rc_roundtrip", BLOCKED,
               "MSP_RC != sent (max diff %d); FC receiver is probably not RX_MSP "
               "or map is not AETR1234 - round-trip cannot be verified" % max_diff,
               data, required=False)


def check_boxes(checks, link):
    # Paged reads go through discover_aux for consistency with the runtime.
    cfg = FcConfig()
    discovery = discover_aux(link, cfg=cfg)
    pairs = discovery.boxes
    data = {
        "box_count": len(pairs),
        "boxes": [{"permanent_id": pid, "name": name} for pid, name in pairs],
        "override_present": any(pid == 50 for pid, _ in pairs),
        "warnings": discovery.warnings,
    }
    if not pairs:
        _check(checks, "boxnames_boxids", FAIL, "no BOXNAMES/BOXIDS data", data)
        return
    ids_ok = all(0 <= pid <= 255 for pid, _ in pairs)
    same_len = len(pairs) == len({pid for pid, _ in pairs})
    detail = "boxes=%d unique_ids=%s override=%s" % (
        len(pairs), same_len, data["override_present"])
    _check(checks, "boxnames_boxids", PASS if ids_ok and same_len else FAIL, detail, data)

    ranges = [r for r in discovery.mode_ranges]
    _check(checks, "mode_ranges", PASS if ranges else FAIL,
           "configured ranges=%d; override AUX=%s; channels=%s"
           % (len(ranges), discovery.override_channel, discovery.to_dict()["channels"]),
           {"ranges": ranges, "channels": discovery.to_dict()["channels"],
            "warnings": discovery.warnings})


def check_position_state(checks, link):
    payload = _q(link, MSP_CMD.MSP2_POSITION_STATE, is_v2=True)
    if not payload:
        _check(checks, "msp2_position_state", SKIP,
               "MSP2 0x3015 not supported by this firmware", None, required=False)
        return
    state = decode_msp2_position_state(payload)
    if state is None:
        _check(checks, "msp2_position_state", FAIL,
               "0x3015 reply too short (%d B, need >=27)" % len(payload))
        return
    finite = all(math.isfinite(v) for v in state["position_enu"] + state["velocity_enu"])
    _check(checks, "msp2_position_state", PASS if finite else FAIL,
           "pos=%s vel=%s" % (state["position_enu"], state["velocity_enu"]),
           {"position_enu": state["position_enu"], "velocity_enu": state["velocity_enu"],
            "offboard_active": state["offboard_active"], "reached": state["reached"]})


# ---- driver -----------------------------------------------------------------

def default_report_path(transport):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return os.path.join(REPO_ROOT, "docs", "sim2real", "protocol_reports",
                        "%s-%s.json" % (stamp, transport))


def summarise(checks):
    counts = {PASS: 0, FAIL: 0, SKIP: 0, BLOCKED: 0}
    for item in checks:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return counts


def _json_default(value):
    """Make device payload bytes (and other odd types) JSON-safe."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return repr(value)


def write_report(report, path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=_json_default)
    return path


def run_offline(args):
    checks = []
    run_framing_checks(checks)
    report = {
        "tool": "protocol_conformance",
        "version": TOOL_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "transport": "offline",
        "evidence_level": "offline",
        "flight_clearance": False,
        "checks": checks,
        "summary": summarise(checks),
        "notes": ["Offline framing/CRC self-test only; no FC was contacted."],
    }
    path = write_report(report, args.out or default_report_path("offline"))
    print("[offline] framing_crc=%s; report=%s"
          % (checks[0]["status"], path))
    return 0 if checks[0]["status"] == PASS else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MSP protocol conformance checker (SITL TCP or FC serial)")
    parser.add_argument("--host", default="", help="SITL TCP host, e.g. 127.0.0.1")
    parser.add_argument("--tcp-port", type=int, default=5761)
    parser.add_argument("--port", default="/dev/ttyACM0", help="real FC serial port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=0.5, help="per-query reply timeout")
    parser.add_argument("--allow-rc", action="store_true",
                        help="bench-only 16ch RC round-trip; PROPS MUST BE REMOVED")
    parser.add_argument("--samples", type=int, default=5, help="attitude samples")
    parser.add_argument("--out", default="", help="JSON report path")
    parser.add_argument("--offline", action="store_true",
                        help="run framing/CRC checks only (no transport)")
    parser.add_argument("--evidence", default="auto",
                        choices=["auto", "sitl", "fake-tcp", "fc-serial-link"],
                        help="evidence label for the report (default: auto)")
    args = parser.parse_args(argv)

    if args.offline:
        return run_offline(args)

    transport_name = "tcp" if args.host else "serial"
    cfg = FcConfig(port=args.port, baud=args.baud,
                   tcp_host=args.host, tcp_port=args.tcp_port,
                   reply_timeout_s=args.timeout)
    link = make_link(cfg)
    checks = []
    run_framing_checks(checks)
    report = {
        "tool": "protocol_conformance",
        "version": TOOL_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "transport": transport_name,
        "endpoint": ("%s:%s" % (args.host, args.tcp_port) if args.host
                     else "%s@%s" % (args.port, args.baud)),
        "allow_rc": bool(args.allow_rc),
        "evidence_level": (args.evidence if args.evidence != "auto" else
                           ("sitl" if args.host in ("127.0.0.1", "localhost")
                            else "fc-serial-link")),
        "flight_clearance": False,
        "checks": checks,
        "notes": [
            "MSP framing/telemetry conformance only; not a flight-safety test.",
            "Never send RC to a powered aircraft with props installed.",
        ],
    }
    exit_code = 0
    try:
        connected = link.connect()
        if not connected:
            _check(checks, "link", BLOCKED,
                   "cannot open %s: %s" % (report["endpoint"], link.last_error))
            _check(checks, "api_version", BLOCKED, "link unavailable")
            exit_code = 2
        else:
            report["link"] = {"generation": link.generation}
            report["firmware"] = {}
            report["firmware"]["api_version"] = check_api(checks, link)
            variant, fc_version = check_variant_version(checks, link)
            report["firmware"]["variant"] = variant
            report["firmware"]["fc_version"] = fc_version
            report["firmware"]["board"] = check_board(checks, link)
            status = check_status(checks, link, args.allow_rc)
            if status is not None:
                report["status"] = {
                    key: value for key, value in status.items()
                    if key != "extended_flight_mode_flags"
                }
            check_attitude(checks, link, args.samples)
            check_altitude(checks, link)
            check_analog(checks, link)
            check_rc_roundtrip(checks, link, args.allow_rc, status)
            check_boxes(checks, link)
            check_position_state(checks, link)
    finally:
        link.close()

    report["summary"] = summarise(checks)
    report["firmware"] = report.get("firmware") or {}
    if any(item["status"] == FAIL for item in checks):
        exit_code = 1
    path = write_report(report, args.out or default_report_path(transport_name))
    print("[%s] %s  %s" % (report["transport"], report["endpoint"], report["summary"]))
    for item in checks:
        print("  %-8s %-20s %s" % (item["status"], item["id"], item["detail"]))
    print("report: %s" % path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
