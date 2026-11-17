#!/usr/bin/env python3
"""Props-off motor A/B harness for the same FC, clean release vs sim2real.

record : replay a scripted RC sequence through MSP and sample MSP_MOTOR
compare: align two recordings and fail (non-zero exit) when any motor differs
         by more than the tolerance.

JSON recording schema (version 1):
  kind        "fw_motor_ab_recording"
  sequence    {"steps": [...], "sha256": "..."}
  firmware    {"api_version": {...}, "variant": "...", "fc_version": {...}}
  samples     [{"t": seconds, "step": 0, "motors": [m0..m7]}]
  props_off_acknowledged  true

Usage:
  python3 tools/fw_motor_ab.py record --sequence seq.json --out clean.json \
      --host 127.0.0.1 --props-off --allow-arm
  python3 tools/fw_motor_ab.py compare clean.json sim2real.json --tolerance 25 \
      --out ab_report.json

The sequence is user supplied; this tool does not invent an arming sequence.
It refuses to run without --props-off, and refuses AUX-high steps unless
--allow-arm is given explicitly.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from msp import (  # noqa: E402
    MSP_CMD,
    decode_motor,
    decode_motor_channels,
)
from onboard.config import FcConfig  # noqa: E402
from onboard.fc.link import make_link  # noqa: E402

RECORD_KIND = "fw_motor_ab_recording"
COMPARE_KIND = "fw_motor_ab_compare"
DEFAULT_RESAMPLE_POINTS = 50


# ---- sequence helpers -------------------------------------------------------

def normalise_sequence(raw):
    if not isinstance(raw, dict) or "steps" not in raw:
        raise ValueError("sequence must be an object with a 'steps' list")
    steps = raw["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("sequence.steps must be a non-empty list")
    out = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError("step %d must be an object" % index)
        hold_s = float(step.get("hold_s", step.get("hold", 1.0)))
        if hold_s <= 0.0:
            raise ValueError("step %d hold_s must be positive" % index)
        channels = step.get("channels")
        if isinstance(channels, dict):
            order = ["roll", "pitch", "throttle", "yaw",
                     "aux1", "aux2", "aux3", "aux4"]
            values = [int(channels.get(key, 1500)) for key in order]
        elif isinstance(channels, (list, tuple)):
            values = [int(value) for value in channels]
        else:
            raise ValueError("step %d needs channels list/object" % index)
        out.append({"hold_s": hold_s, "channels": list(decode_motor_channels(values))})
    return {"steps": out}


def sequence_hash(sequence):
    canonical = json.dumps(sequence, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sequence_has_arm_high(sequence, aux_start=4):
    for step in sequence["steps"]:
        for value in step["channels"][aux_start:]:
            if int(value) > 1700:
                return True
    return False


# ---- record -----------------------------------------------------------------

def _firmware_info(link):
    info = {}
    payload = link.query(MSP_CMD.API_VERSION, timeout=0.5)
    if payload and len(payload) >= 3:
        info["api_version"] = {"protocol": payload[0], "major": payload[1],
                               "minor": payload[2]}
    payload = link.query(MSP_CMD.FC_VARIANT, timeout=0.5)
    if payload:
        info["variant"] = bytes(payload[:4]).decode("ascii", "replace")
    payload = link.query(MSP_CMD.FC_VERSION, timeout=0.5)
    if payload and len(payload) >= 4:
        info["fc_version"] = {"year": 2000 + payload[0], "month": payload[1],
                              "patch": payload[2]}
    return info


def record_sequence(link, sequence, sample_hz=20.0, rc_hz=50.0, sample_timeout=0.05):
    samples = []
    start = time.monotonic()
    sample_period = 1.0 / max(1.0, float(sample_hz))
    rc_period = 1.0 / max(1.0, float(rc_hz))
    for step_index, step in enumerate(sequence["steps"]):
        channels = list(step["channels"])
        step_end = time.monotonic() + float(step["hold_s"])
        next_rc = time.monotonic()
        next_sample = time.monotonic()
        while True:
            now = time.monotonic()
            if now >= step_end:
                break
            if now >= next_rc:
                link.send_rc_16(channels)
                next_rc = now + rc_period
            if now >= next_sample:
                payload = link.query(MSP_CMD.MSP_MOTOR, timeout=sample_timeout)
                motors = decode_motor(payload) if payload else None
                samples.append({
                    "t": round(now - start, 4),
                    "step": step_index,
                    "motors": list(motors) if motors else [],
                })
                next_sample = now + sample_period
            sleep_for = min(0.005, max(0.0, step_end - time.monotonic()))
            if sleep_for > 0:
                time.sleep(sleep_for)
    return samples


def write_csv(path, samples):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_s", "step", "m0", "m1", "m2", "m3",
                         "m4", "m5", "m6", "m7"])
        for sample in samples:
            motors = list(sample["motors"]) + [""] * 8
            writer.writerow([sample["t"], sample["step"]] + motors[:8])


def cmd_record(args):
    if not args.props_off:
        print("ERROR: --props-off is mandatory; refusing to spin motors", file=sys.stderr)
        return 2
    try:
        with open(args.sequence, "r", encoding="utf-8") as handle:
            sequence = normalise_sequence(json.load(handle))
    except (OSError, ValueError) as exc:
        print("ERROR: bad sequence: %s" % exc, file=sys.stderr)
        return 2
    if sequence_has_arm_high(sequence) and not args.allow_arm:
        print("ERROR: sequence drives an AUX channel high (>1700); pass --allow-arm "
              "explicitly if that is an intentional props-off arm/mode test",
              file=sys.stderr)
        return 2
    cfg = FcConfig(port=args.port, baud=args.baud, tcp_host=args.host,
                   tcp_port=args.tcp_port, reply_timeout_s=0.1)
    link = make_link(cfg)
    try:
        if not link.connect():
            print("ERROR: FC link unavailable: %s" % (link.last_error or "connect failed"),
                  file=sys.stderr)
            return 2
        if not args.allow_arm:
            status = link.query(MSP_CMD.GET_STATUS_EX, timeout=0.5)
            armed = None
            if status and len(status) >= 10:
                armed = bool((status[6] | status[7] << 8 | status[8] << 16 | status[9] << 24) & 1)
            if armed:
                print("ERROR: FC reports ARMED; refusing to record", file=sys.stderr)
                return 2
        firmware = _firmware_info(link)
        samples = record_sequence(link, sequence, sample_hz=args.sample_hz,
                                  rc_hz=args.rc_hz)
    finally:
        # Best effort: leave the FC disarmed on the way out.
        try:
            link.send_rc_16([1500, 1500, 1000, 1500] + [1000, 1500, 1500, 1500] + [1500] * 8)
        except Exception:
            pass
        link.close()
    if not samples:
        print("ERROR: no MSP_MOTOR samples collected", file=sys.stderr)
        return 1
    recording = {
        "kind": RECORD_KIND,
        "version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "sequence": dict(sequence, sha256=sequence_hash(sequence)),
        "sample_hz": float(args.sample_hz),
        "rc_hz": float(args.rc_hz),
        "firmware": firmware,
        "samples": samples,
        "props_off_acknowledged": True,
        "allow_arm": bool(args.allow_arm),
    }
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(recording, handle, indent=2, sort_keys=True)
    if args.csv:
        write_csv(args.csv, samples)
    print("recorded %d samples over %d steps -> %s"
          % (len(samples), len(sequence["steps"]), args.out))
    return 0


# ---- compare ----------------------------------------------------------------

def load_recording(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("kind") != RECORD_KIND:
        raise ValueError("%s is not a %s file" % (path, RECORD_KIND))
    if not isinstance(data.get("samples"), list) or not data["samples"]:
        raise ValueError("%s has no samples" % path)
    return data


def _resample(points, count=DEFAULT_RESAMPLE_POINTS):
    if not points:
        return []
    if len(points) == 1:
        return [float(points[0][1])] * count
    t0 = float(points[0][0])
    t1 = float(points[-1][0])
    if t1 <= t0:
        return [float(points[0][1])] * count
    out = []
    cursor = 0
    for index in range(count):
        t = t0 + (t1 - t0) * index / max(1, count - 1)
        while cursor < len(points) - 2 and points[cursor + 1][0] < t:
            cursor += 1
        ta, va = points[cursor]
        tb, vb = points[cursor + 1]
        if tb <= ta:
            out.append(float(va))
            continue
        fraction = max(0.0, min(1.0, (t - ta) / (tb - ta)))
        out.append(float(va) + (float(vb) - float(va)) * fraction)
    return out


def _step_series(recording):
    by_step = {}
    for sample in recording["samples"]:
        motors = sample.get("motors") or []
        if not motors:
            continue
        by_step.setdefault(int(sample["step"]), {})
        for motor_index, value in enumerate(motors):
            by_step[int(sample["step"])].setdefault(motor_index, []).append(
                (float(sample["t"]), float(value)))
    return by_step


def compare_recordings(recording_a, recording_b, tolerance=25.0,
                       ignore_sequence_hash=False):
    tolerance = abs(float(tolerance))
    result = {
        "kind": COMPARE_KIND,
        "version": 1,
        "tolerance": tolerance,
        "passed": True,
        "reasons": [],
        "per_step_motor_max_diff": [],
        "max_diff": 0.0,
        "worst": None,
    }
    hash_a = (recording_a.get("sequence") or {}).get("sha256")
    hash_b = (recording_b.get("sequence") or {}).get("sha256")
    if hash_a and hash_b and hash_a != hash_b and not ignore_sequence_hash:
        result["passed"] = False
        result["reasons"].append("sequence sha256 differs between recordings")

    series_a = _step_series(recording_a)
    series_b = _step_series(recording_b)
    if set(series_a) != set(series_b):
        result["passed"] = False
        result["reasons"].append(
            "step sets differ: %s vs %s" % (sorted(series_a), sorted(series_b)))
    for step in sorted(set(series_a) & set(series_b)):
        motors_a = series_a[step]
        motors_b = series_b[step]
        for motor in sorted(set(motors_a) & set(motors_b)):
            curve_a = _resample(motors_a[motor])
            curve_b = _resample(motors_b[motor])
            worst = 0.0
            worst_index = 0
            for index, (value_a, value_b) in enumerate(zip(curve_a, curve_b)):
                diff = abs(value_a - value_b)
                if diff > worst:
                    worst = diff
                    worst_index = index
            entry = {
                "step": step,
                "motor": motor,
                "max_diff": round(worst, 3),
                "normalized_sample": worst_index,
            }
            result["per_step_motor_max_diff"].append(entry)
            if worst > result["max_diff"]:
                result["max_diff"] = round(worst, 3)
                result["worst"] = entry
            if worst > tolerance:
                result["passed"] = False
                result["reasons"].append(
                    "step %d motor %d differs by %.1f > tolerance %.1f"
                    % (step, motor, worst, tolerance))
        missing_a = sorted(set(motors_b) - set(motors_a))
        missing_b = sorted(set(motors_a) - set(motors_b))
        if missing_a or missing_b:
            result["passed"] = False
            result["reasons"].append(
                "step %d motor count mismatch (missing in A=%s, in B=%s)"
                % (step, missing_a, missing_b))
    return result


def cmd_compare(args):
    try:
        recording_a = load_recording(args.a)
        recording_b = load_recording(args.b)
    except (OSError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    result = compare_recordings(
        recording_a, recording_b, tolerance=args.tolerance,
        ignore_sequence_hash=args.ignore_sequence_hash)
    result["created_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result["file_a"] = os.path.abspath(args.a)
    result["file_b"] = os.path.abspath(args.b)
    if args.out:
        out_dir = os.path.dirname(os.path.abspath(args.out))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("tolerance=%.1f max_diff=%.3f passed=%s"
              % (result["tolerance"], result["max_diff"], result["passed"]))
        for reason in result["reasons"]:
            print("  reason: %s" % reason)
        for entry in result["per_step_motor_max_diff"]:
            print("  step=%d motor=%d max_diff=%.2f"
                  % (entry["step"], entry["motor"], entry["max_diff"]))
    return 0 if result["passed"] else 1


# ---- CLI --------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Props-off MSP_MOTOR A/B recorder/comparator for one FC")
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="replay an RC sequence and record MSP_MOTOR")
    record.add_argument("--sequence", required=True, help="sequence JSON")
    record.add_argument("--out", required=True, help="recording JSON output")
    record.add_argument("--csv", default="", help="optional CSV output")
    record.add_argument("--host", default="", help="SITL TCP host (empty = serial)")
    record.add_argument("--tcp-port", type=int, default=5761)
    record.add_argument("--port", default="/dev/ttyACM0")
    record.add_argument("--baud", type=int, default=115200)
    record.add_argument("--sample-hz", type=float, default=20.0)
    record.add_argument("--rc-hz", type=float, default=50.0)
    record.add_argument("--props-off", action="store_true",
                        help="required acknowledgement that props are removed")
    record.add_argument("--allow-arm", action="store_true",
                        help="allow AUX-high steps / an armed FC (props off only)")
    record.set_defaults(func=cmd_record)

    compare = sub.add_parser("compare", help="compare two recordings")
    compare.add_argument("a", help="clean-release recording JSON")
    compare.add_argument("b", help="sim2real recording JSON")
    compare.add_argument("--tolerance", type=float, default=25.0,
                         help="max allowed motor unit difference (default 25)")
    compare.add_argument("--ignore-sequence-hash", action="store_true")
    compare.add_argument("--json", action="store_true")
    compare.add_argument("--out", default="", help="comparison JSON output")
    compare.set_defaults(func=cmd_compare)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
