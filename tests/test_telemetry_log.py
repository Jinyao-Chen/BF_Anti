"""Blackbox tests: run directory, CSV schema, rate limiting, durability."""
import csv
import json
import os
import time

from onboard.telemetry_log import CSV_COLUMNS, TelemetryLog, collect_run_meta


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_start_creates_manifest_header_and_latest_link(tmp_path):
    tlog = TelemetryLog(str(tmp_path), hz=10)
    try:
        assert tlog.start(meta={"git": "deadbeef", "hostname": "pi5"})
        assert tlog.run_dir and os.path.isdir(tlog.run_dir)
        assert os.path.islink(os.path.join(str(tmp_path), "latest"))
        manifest = json.load(open(os.path.join(tlog.run_dir, "startup.json"), encoding="utf-8"))
        assert manifest["git"] == "deadbeef"
        assert manifest["columns"] == list(CSV_COLUMNS)
        with open(tlog.telemetry_path, "r", encoding="utf-8") as fh:
            header = fh.readline().strip().split(",")
        assert header == list(CSV_COLUMNS)
    finally:
        tlog.close("test_start")


def test_sample_is_rate_limited_and_close_flushes(tmp_path):
    tlog = TelemetryLog(str(tmp_path), hz=10)
    assert tlog.start()
    base = 100.0
    assert tlog.sample(base, {"armed": 1, "rc0": 1500, "unknown": "ignored"})
    assert not tlog.sample(base + 0.01, {"armed": 1})
    assert tlog.sample(base + 0.11, {"armed": 1, "rc0": 1600})
    tlog.close("done")
    rows = read_rows(tlog.telemetry_path)
    assert len(rows) == 2
    assert rows[0]["armed"] == "1"
    assert rows[0]["rc0"] == "1500"
    assert rows[0]["t_wall"]
    assert rows[1]["rc0"] == "1600"
    manifest = json.load(open(os.path.join(tlog.run_dir, "startup.json"), encoding="utf-8"))
    assert manifest["rows"] == 2
    assert manifest["ended_wall"]


def test_events_are_jsonl_and_ordered(tmp_path):
    tlog = TelemetryLog(str(tmp_path), hz=10)
    assert tlog.start()
    tlog.event("safety_gate", allow=False, reason="disarmed", nested={"a": 1})
    tlog.close("test_events")
    with open(tlog.events_path, "r", encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if line.strip()]
    names = [event["event"] for event in events]
    assert names[0] == "run_start"
    assert "safety_gate" in names
    assert names[-1] == "run_stop"
    gate = next(event for event in events if event["event"] == "safety_gate")
    assert gate["allow"] is False
    assert gate["reason"] == "disarmed"


def test_telemetry_rotates_by_size(tmp_path):
    tlog = TelemetryLog(str(tmp_path), hz=0, max_file_bytes=64 * 1024)
    assert tlog.start()
    for index in range(600):
        tlog.sample(float(index), {"armed": 1, "rc0": 1500, "det_present": 0})
    tlog.close("rotate")
    rotated = os.path.join(tlog.run_dir, "telemetry.001.csv")
    assert os.path.exists(rotated)
    assert len(read_rows(rotated)) > 0
    assert len(read_rows(tlog.telemetry_path)) > 0
    for path in (rotated, tlog.telemetry_path):
        with open(path, "r", encoding="utf-8") as fh:
            assert fh.readline().strip().split(",") == list(CSV_COLUMNS)


def test_unwritable_directory_degrades_without_raising(tmp_path):
    blocker = tmp_path / "logs"
    blocker.write_text("not a directory", encoding="utf-8")
    tlog = TelemetryLog(str(blocker), hz=10)
    assert tlog.start() is False
    assert not tlog.enabled
    assert tlog.startup_error
    # Producers and close must stay safe when the blackbox is disabled.
    assert tlog.sample(time.monotonic(), {"armed": 1}) is False
    assert tlog.event("nope") is False
    tlog.close("disabled")


def test_old_runs_are_pruned(tmp_path):
    for stamp in ("20260101_000000", "20260102_000000", "20260103_000000"):
        run_dir = tmp_path / ("run_" + stamp)
        run_dir.mkdir()
        (run_dir / "startup.json").write_text("{}", encoding="utf-8")
    tlog = TelemetryLog(str(tmp_path), hz=10, max_runs=2)
    assert tlog.start()
    tlog.close("prune")
    runs = sorted(name for name in os.listdir(tmp_path) if name.startswith("run_"))
    assert len(runs) == 2
    assert "run_20260103_000000" in runs


def test_collect_run_meta_captures_argv_and_config():
    meta = collect_run_meta({"telemetry": {"hz": 10}}, ["prog", "--config", "x.json"])
    assert meta["argv"] == ["prog", "--config", "x.json"]
    assert meta["config"] == {"telemetry": {"hz": 10}}
    assert meta["schema"] == 1
    assert meta["hostname"]


def test_close_is_idempotent(tmp_path):
    tlog = TelemetryLog(str(tmp_path), hz=10)
    assert tlog.start()
    tlog.close("first")
    tlog.close("second")
    assert not tlog.enabled
