#!/usr/bin/env python3
"""Summarise BF_Anti blackbox runs: what did the service actually record?

Use this after a boot/flight when "there are no logs" or when a log exists but
you need to know which signal was silent:

    python3 tools/log_summary.py --dir /home/pi5/bf_anti_logs
    python3 tools/log_summary.py --dir logs --runs 3 --json

For every run it reports row counts plus coverage of the four required
signals (FC telemetry, pilot RC, detector, control command) and explains what
a missing signal usually means.  Exit code 0 when a run captured all four,
1 when something is missing, 2 on bad arguments.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

REQUIRED_SIGNALS = (
    ("fc", "飞控", lambda row: row.get("att_age_s", "") != "" or row.get("armed", "") != ""),
    ("rc", "遥控器", lambda row: row.get("rc_age_s", "") != ""),
    ("det", "检测", lambda row: row.get("det_present", "") == "1"),
    ("cmd", "控制指令", lambda row: row.get("cmd_valid", "") == "1"),
)


def find_runs(base: Path) -> list[Path]:
    if not base.is_dir():
        return []
    runs = [
        entry
        for entry in base.iterdir()
        if entry.is_dir()
        and entry.name.startswith("run_")
        and (entry / "startup.json").is_file()
    ]
    def _mtime(entry: Path) -> float:
        try:
            return entry.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(runs, key=_mtime, reverse=True)


def read_manifest(run: Path) -> dict:
    try:
        with open(run / "startup.json", "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def summarise_run(run: Path) -> dict:
    manifest = read_manifest(run)
    coverage = {key: 0 for key, _, _ in REQUIRED_SIGNALS}
    reasons: Counter = Counter()
    modes: Counter = Counter()
    rows = 0
    first_wall = last_wall = ""
    csv_path = run / manifest.get("telemetry_file", "telemetry.csv")
    if csv_path.is_file():
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    rows += 1
                    if not first_wall:
                        first_wall = row.get("t_wall", "")
                    last_wall = row.get("t_wall", "")
                    for key, _, predicate in REQUIRED_SIGNALS:
                        try:
                            if predicate(row):
                                coverage[key] += 1
                        except Exception:
                            pass
                    if row.get("safety_reason"):
                        reasons[row["safety_reason"]] += 1
                    if row.get("cmd_mode"):
                        modes[row["cmd_mode"]] += 1
        except OSError:
            pass
    event_counts: Counter = Counter()
    events_path = run / manifest.get("events_file", "events.jsonl")
    if events_path.is_file():
        try:
            with open(events_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event_counts[json.loads(line).get("event", "?")] += 1
                    except ValueError:
                        pass
        except OSError:
            pass
    return {
        "run": run.name,
        "path": str(run),
        "started_wall": manifest.get("started_wall", ""),
        "ended_wall": manifest.get("ended_wall", ""),
        "git": manifest.get("git", ""),
        "boot_id": manifest.get("boot_id", ""),
        "rows": rows,
        "rows_dropped": manifest.get("rows_dropped", 0),
        "io_error": manifest.get("io_error", ""),
        "coverage": coverage,
        "first_wall": first_wall,
        "last_wall": last_wall,
        "reasons": dict(reasons.most_common(6)),
        "modes": dict(modes.most_common(4)),
        "events": dict(event_counts.most_common(12)),
    }


def verdict(summary: dict) -> list[str]:
    notes: list[str] = []
    if summary["rows"] == 0:
        notes.append(
            "遥测 0 行：进程在写出第一行前退出或被信号杀死；"
            "先看 events.jsonl 的 startup_retry / run_start，再看 journalctl。"
        )
    for key, label, _ in REQUIRED_SIGNALS:
        if summary["coverage"].get(key, 0) == 0 and summary["rows"] > 0:
            if key == "fc":
                notes.append("飞控数据缺失：检查 /dev/ttyACM0、MSP 应答与 fc.port。")
            elif key == "rc":
                notes.append(
                    "遥控器数据缺失：FC 未返回 MSP_RC（RX_MSP 且无 CRSF 接收机时是预期现象）。"
                )
            elif key == "det":
                notes.append("检测数据缺失：检查相机 /dev/video0、Hailo 设备与 HEF 路径。")
            elif key == "cmd":
                notes.append(
                    "控制指令缺失：safety 门全程关闭；看 summary 里的 reasons 与 "
                    "det/rc 覆盖情况。"
                )
    if summary["io_error"]:
        notes.append(f"日志 I/O 中止：{summary['io_error']}（SD 卡满/只读？）")
    return notes


def print_human(summary: dict, notes: list[str]) -> None:
    print(f"== {summary['run']} ==")
    print(f"  路径      : {summary['path']}")
    print(f"  启动      : {summary['started_wall'] or '-'}")
    print(f"  结束      : {summary['ended_wall'] or '-'}")
    print(f"  遥测行数  : {summary['rows']}  (dropped={summary['rows_dropped']})")
    if summary["git"]:
        print(f"  git       : {summary['git'][:12]}")
    signals = "  ".join(
        f"{label}={'ok' if summary['coverage'].get(key, 0) else 'MISSING'}"
        f"({summary['coverage'].get(key, 0)})"
        for key, label, _ in REQUIRED_SIGNALS
    )
    print(f"  四路信号  : {signals}")
    if summary["reasons"]:
        print(f"  safety    : {summary['reasons']}")
    if summary["modes"]:
        print(f"  cmd mode  : {summary['modes']}")
    if summary["events"]:
        print(f"  事件      : {summary['events']}")
    for note in notes:
        print(f"  ! {note}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir", default=os.environ.get("BF_ANTI_LOG_DIR", "logs"),
        help="blackbox root directory (default: $BF_ANTI_LOG_DIR or ./logs)",
    )
    parser.add_argument("--runs", type=int, default=1, help="how many recent runs to show")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    base = Path(os.path.expanduser(args.dir)).resolve()
    runs = find_runs(base)
    if not runs:
        print(f"no run_* directories with startup.json under {base}", file=sys.stderr)
        return 2
    selected = runs[: max(1, args.runs)]
    summaries = [summarise_run(run) for run in selected]

    if args.json:
        json.dump(summaries, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        print(f"黑匣子根目录: {base}\n共 {len(runs)} 次运行，显示最近 {len(summaries)} 次\n")
        for summary in summaries:
            print_human(summary, verdict(summary))

    complete = all(
        all(summary["coverage"].get(key, 0) > 0 for key, _, _ in REQUIRED_SIGNALS)
        for summary in summaries
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
