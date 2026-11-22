"""log_summary.py: four-signal coverage report over a real run directory."""
import json

from onboard.telemetry_log import TelemetryLog
from tools import log_summary


def make_run(base, complete=True):
    tlog = TelemetryLog(str(base), hz=0)
    assert tlog.start(meta={"git": "abc123"})
    for index in range(3):
        fields = {
            "armed": 1,
            "att_age_s": 0.01,
            "rc_age_s": 0.02,
            "safety_reason": "ok",
            "allow_auto": 1,
        }
        if complete:
            fields.update({"det_present": 1, "cmd_valid": 1, "cmd_mode": "TRACK"})
        else:
            fields.update({"det_present": 0, "cmd_valid": 0, "cmd_mode": "SEARCH"})
        tlog.sample(float(index), fields)
    tlog.close("test")
    return tlog.run_dir


def test_summary_reports_complete_run(tmp_path, capsys):
    make_run(tmp_path, complete=True)
    assert log_summary.main(["--dir", str(tmp_path), "--runs", "1"]) == 0
    out = capsys.readouterr().out
    assert "四路信号" in out
    assert "飞控=ok" in out and "遥控器=ok" in out
    assert "检测=ok" in out and "控制指令=ok" in out


def test_summary_flags_missing_detection_and_command(tmp_path, capsys):
    make_run(tmp_path, complete=False)
    assert log_summary.main(["--dir", str(tmp_path), "--runs", "1"]) == 1
    out = capsys.readouterr().out
    assert "检测=MISSING" in out
    assert "控制指令=MISSING" in out
    assert "检测数据缺失" in out


def test_summary_json_output(tmp_path, capsys):
    make_run(tmp_path, complete=True)
    assert log_summary.main(["--dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["rows"] == 3
    assert payload[0]["coverage"]["fc"] == 3


def test_summary_without_runs_returns_two(tmp_path):
    assert log_summary.main(["--dir", str(tmp_path / "missing")]) == 2
