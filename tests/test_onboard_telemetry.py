"""Onboard app -> blackbox wiring: one row per tick, events, failure isolation."""
import csv
import json
import time

from onboard.telemetry_log import TelemetryLog
from onboard.types import ControlOutput
from onboard.types import Detection
from tests.test_onboard_app import build_app, fresh_state


def read_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_events(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_step_once_records_fc_rc_detection_and_command(tmp_path):
    now = time.monotonic()
    det = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now)
    app, mux, _, _ = build_app(det, state=fresh_state(now))
    tlog = TelemetryLog(str(tmp_path), hz=0)
    assert tlog.start()
    app.telemetry = tlog
    app._engaged = True
    try:
        app.step_once(now)
        app.detector.detection = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now + 0.02)
        app.step_once(now + 0.02)
    finally:
        tlog.close("test_rows")
    rows = read_rows(tlog.telemetry_path)
    assert rows, "no telemetry rows written"
    last = rows[-1]
    # Flight controller
    assert last["armed"] == "1"
    assert last["alt_m"] == "10.0"
    assert last["att_age_s"] != ""
    # Pilot RC, all 16 channels
    assert last["rc0"] == "1500"
    assert last["rc7"] == "2000"
    assert last["rc15"] == "1500"
    # Detector
    assert last["det_present"] == "1"
    assert last["det_conf"] == "0.9"
    assert last["det_x1"] == "600.0"
    # Safety + control command
    assert last["allow_auto"] == "1"
    assert last["cmd_valid"] == "1"
    assert last["cmd_throttle"] != ""
    assert last["cmd_mode"] in ("TRACK", "TERMINAL")
    assert last["cmd_reason"]
    assert last["mux_mode"] == ""
    # Loop health
    assert float(last["loop_dt_s"]) > 0.0


def test_engage_and_target_events_are_recorded(tmp_path):
    now = time.monotonic()
    det = Detection((600.0, 320.0, 680.0, 400.0), 0.9, t=now)
    app, _, _, _ = build_app(det, state=fresh_state(now))
    tlog = TelemetryLog(str(tmp_path), hz=0)
    assert tlog.start()
    app.telemetry = tlog
    try:
        app._on_engage(1500)
        app.step_once(now)
    finally:
        tlog.close("test_events")
    events = read_events(tlog.events_path)
    names = [event["event"] for event in events]
    assert "engaged" in names
    assert "safety_gate" in names
    assert "target_acquired" in names
    engaged = [event for event in events if event["event"] == "engaged"][-1]
    assert engaged["engaged"] is True
    assert engaged["throttle"] == 1500
    acquired = next(event for event in events if event["event"] == "target_acquired")
    assert acquired["conf"] == 0.9


def test_missing_fc_data_is_empty_not_zero(tmp_path):
    now = time.monotonic()
    state = fresh_state(now)
    state.att_t = 0.0
    state.alt_t = 0.0
    state.rc_t = 0.0
    state.status_t = 0.0
    state.analog_t = 0.0
    state.link_t = 0.0
    tlog = TelemetryLog(str(tmp_path), hz=0)
    assert tlog.start()
    app, _, _, _ = build_app(None, state=state)
    app.telemetry = tlog
    try:
        app.step_once(now)
    finally:
        tlog.close("test_empty")
    row = read_rows(tlog.telemetry_path)[-1]
    assert row["armed"] == ""
    assert row["alt_m"] == ""
    assert row["rc0"] == ""
    assert row["att_age_s"] == ""
    assert row["det_present"] == "0"
    # Boot warm-up must deny without latching a critical fault.
    assert row["safety_reason"] == "fc_warming_up"
    assert row["safety_critical"] == "0"


def test_broken_blackbox_never_breaks_control_tick(tmp_path):
    class Exploding(TelemetryLog):
        @property
        def enabled(self):
            return True

        def due(self, now):
            raise RuntimeError("boom")

    now = time.monotonic()
    app, mux, _, _ = build_app(None, state=fresh_state(now))
    broken = Exploding(str(tmp_path), hz=0)
    app.telemetry = broken
    out, decision = app.step_once(now)
    assert decision is not None
    assert mux.cmds and mux.cmds[-1] is None


def test_run_retries_startup_instead_of_exiting():
    app, _, _, _ = build_app()
    calls = []

    def flaky_start():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("/dev/ttyACM0 missing")
        app._stop.set()

    app.start = flaky_start  # type: ignore[assignment]
    app.startup_retry_s = 0.01
    assert app.run() == 0
    assert len(calls) == 3


def test_run_logs_stop_event(tmp_path):
    now = time.monotonic()
    app, _, _, _ = build_app(None, state=fresh_state(now))
    tlog = TelemetryLog(str(tmp_path), hz=0)
    assert tlog.start()
    app.telemetry = tlog
    app._stop.set()
    app.run()
    events = read_events(tlog.events_path)
    names = [event["event"] for event in events]
    assert "run_stop_requested" in names
    assert names[-1] == "run_stop"


def test_main_bootstraps_blackbox_and_file_log(tmp_path, monkeypatch):
    import json
    import logging

    from onboard import app as app_module

    root = logging.getLogger()
    before = list(root.handlers)
    config = tmp_path / "onboard.json"
    config.write_text(
        json.dumps(
            {
                "log_dir": str(tmp_path / "logs"),
                "telemetry": {"hz": 0},
                "detector": {"backend": "null"},
                "video": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_module.OnboardApp, "run", lambda self: 0)
    try:
        assert app_module.main(["--config", str(config)]) == 0
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()
    runs = [p for p in (tmp_path / "logs").iterdir() if p.name.startswith("run_")]
    assert len(runs) == 1
    run = runs[0]
    assert (run / "telemetry.csv").is_file()
    assert (run / "startup.json").is_file()
    assert (run / "runtime.log").is_file()
    manifest = json.loads((run / "startup.json").read_text(encoding="utf-8"))
    assert manifest["config"]["log_dir"] == str(tmp_path / "logs")
    assert manifest["telemetry_file"] == "telemetry.csv"
    assert (run / "faulthandler.log").is_file()


def test_main_records_config_error_on_disk(tmp_path, monkeypatch):
    from onboard import app as app_module

    logs = tmp_path / "logs"
    logs.mkdir()
    # A real flight run that must survive a config typo / restart loop.
    old_run = logs / "run_20260101-000000"
    old_run.mkdir()
    (old_run / "telemetry.csv").write_text("sentinel\n", encoding="utf-8")

    bad = tmp_path / "bad.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    rc = app_module.main(["--config", str(bad), "--log-dir", str(logs)])
    assert rc == 2
    # The config-error blackbox is isolated in its own subdirectory; it must
    # not run TelemetryLog._prune_old_runs() over the real flight logs.
    assert old_run.is_dir() and (old_run / "telemetry.csv").is_file()
    runs = [p for p in (logs / "config_errors").iterdir() if p.name.startswith("run_")]
    assert len(runs) == 1
    events = read_events(runs[0] / "events.jsonl")
    assert any(event["event"] == "config_error" for event in events)


def test_run_records_tick_errors_in_events(tmp_path):
    now = time.monotonic()
    app, _, _, _ = build_app(None, state=fresh_state(now))
    tlog = TelemetryLog(str(tmp_path), hz=0)
    assert tlog.start()
    app.telemetry = tlog

    def exploding_tick(now=None):
        app._stop.set()
        raise RuntimeError("detector exploded")

    app.step_once = exploding_tick  # type: ignore[assignment]
    app.start = lambda: None  # type: ignore[assignment]
    app.run()
    names = [event["event"] for event in read_events(tlog.events_path)]
    assert "tick_error" in names


def test_startup_grace_expires_after_first_complete_fc_sample():
    now = time.monotonic()
    app, _, _, _ = build_app(None, state=fresh_state(now))
    state = app.state.snapshot(now)
    first = app._safety_inputs(state, None, ControlOutput(), now)
    assert first.fc_seen and first.warmup_grace
    later = app._safety_inputs(state, None, ControlOutput(), now + 3.0)
    assert later.fc_seen and not later.warmup_grace
