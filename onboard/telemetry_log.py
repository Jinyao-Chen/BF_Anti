"""Flight-data blackbox: per-run CSV telemetry plus JSONL events.

Why this exists
---------------
The onboard service used to log only to stderr, which systemd forwards to
journald.  On a Pi that is not enough:

* journald is volatile unless ``/var/log/journal`` exists (Ubuntu/RPi OS ship
  ``Storage=auto``), so a reboot can erase every line the service wrote;
* the process could die before its first log line (FC not enumerated, fb0 not
  ready) and systemd's ``StartLimitBurst`` then stopped retrying, leaving
  nothing on disk;
* native crashes in HailoRT/OpenCV write nothing at all.

This module writes one directory per service start under ``log_dir``::

    logs/
      run_20261122_164721/
        startup.json      # code version, config, boot id, pid, start/stop
        telemetry.csv     # one row per control tick at telemetry_hz
        events.jsonl      # state transitions, one JSON object per line
        runtime.log       # the human-readable stdlib log (see logging_setup)
        faulthandler.log  # only written if the process takes a fatal signal
      latest -> run_20261122_164721

Design rules
------------
* The control loop must never block on disk I/O: ``sample()``/``event()``
  only enqueue, a writer thread does the formatting, flush and fsync.
* A full disk, a read-only log dir, or a dead SD card must never take the
  aircraft down: failures disable the blackbox and are reported once.
* Rows are line buffered and fsynced on an interval, so a power cut loses at
  most a few seconds of data instead of the whole flight.
* Missing data is written as an empty field, never as an invented zero.  A
  zero timestamp means "never received" (see onboard.types.FcState).

This module is stdlib-only on purpose so it can be unit-tested on a laptop.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import queue
import shutil
import socket
import sys
import threading
import time
from datetime import datetime
from typing import Any, Optional

# Column order of telemetry.csv.  Keep this stable: the summary tool and any
# post-flight notebook address columns by name, but a rename still breaks the
# "same columns across runs" assumption.
CSV_COLUMNS: tuple[str, ...] = (
    # time base
    "t_wall", "t_mono", "uptime_s",
    # flight controller
    "armed", "fc_mode", "roll_deg", "pitch_deg", "yaw_deg",
    "alt_m", "vario_mps",
    "att_age_s", "alt_age_s", "status_age_s", "link_age_s",
    "voltage_v", "current_a", "mah", "rssi",
    # pilot RC (AETR + AUX1..AUX12); empty when the FC never returned MSP_RC
    "rc_age_s", *(f"rc{i}" for i in range(16)),
    # detector
    "det_present", "det_conf", "det_class", "det_age_s",
    "det_x1", "det_y1", "det_x2", "det_y2",
    "det_fps", "det_infer_ms", "det_errors",
    # tracker inside the controller
    "track_locked", "track_streak", "track_miss",
    # control output (what the companion wants to send)
    "cmd_valid", "cmd_mode", "cmd_reason",
    "cmd_roll", "cmd_pitch", "cmd_throttle", "cmd_yaw",
    "cmd_range_m", "cmd_confirmed",
    # safety gate and mux (who actually owns the sticks)
    "allow_auto", "safety_reason", "safety_critical",
    "mux_mode", "mux_pos", "engaged",
    # loop health
    "loop_dt_s", "link_errors",
)

MANIFEST_NAME = "startup.json"
TELEMETRY_NAME = "telemetry.csv"
EVENTS_NAME = "events.jsonl"
RUN_PREFIX = "run_"
LATEST_LINK = "latest"

_STOP = object()


def _iso_now(clock=time.time) -> str:
    """Local ISO-8601 with milliseconds; empty when the clock is nonsense."""
    try:
        return datetime.fromtimestamp(float(clock())).astimezone().isoformat(
            timespec="milliseconds"
        )
    except (OverflowError, OSError, ValueError):
        return ""


def _read_boot_id() -> str:
    try:
        with open("/proc/sys/kernel/random/boot_id", "r", encoding="ascii") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _find_git_sha(start: Optional[str] = None) -> str:
    """Best-effort HEAD sha without spawning git (the flight image may not have it)."""
    path = os.path.abspath(start or os.getcwd())
    for _ in range(6):
        git_dir = os.path.join(path, ".git")
        head_path = os.path.join(git_dir, "HEAD")
        if os.path.isfile(head_path):
            try:
                with open(head_path, "r", encoding="utf-8") as fh:
                    head = fh.read().strip()
                if head.startswith("ref:"):
                    ref = head.split(":", 1)[1].strip()
                    with open(os.path.join(git_dir, ref), "r", encoding="utf-8") as fh:
                        return fh.read().strip()[:40]
                return head[:40]
            except OSError:
                return ""
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return ""


def collect_run_meta(
    config: Optional[dict] = None,
    argv: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Metadata written to startup.json before the first telemetry row."""
    now = time.time()
    return {
        "schema": 1,
        "created_wall": _iso_now(),
        "clock_synced": now >= 1_600_000_000.0,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "boot_id": _read_boot_id(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "argv": list(argv) if argv is not None else list(sys.argv),
        "git": _find_git_sha(),
        "config": config or {},
    }


class TelemetryLog:
    """Per-run flight-data recorder.

    Parameters
    ----------
    base_dir:
        Directory that holds one ``run_*`` directory per service start.
    hz:
        Row rate.  The control loop runs faster (50 Hz); rows are decimated.
        ``0`` records every call.
    max_file_bytes:
        Rotate telemetry.csv when it grows past this size.
    max_runs:
        Keep at most this many run directories (oldest removed at start).
    flush_interval_s / fsync_interval_s:
        Writer-thread flush and fsync cadence.  fsync is what bounds data loss
        on a power cut; it runs off the control thread so it can be slow.
    queue_size:
        Rows buffered while the SD card is slow.  When full, rows are dropped
        (and counted) rather than blocking the controller.
    """

    def __init__(
        self,
        base_dir: str,
        *,
        hz: float = 10.0,
        max_file_bytes: int = 32 * 1024 * 1024,
        max_runs: int = 20,
        flush_interval_s: float = 1.0,
        fsync_interval_s: float = 5.0,
        queue_size: int = 4096,
        segment_index: int = 0,
        clock=time.time,
        monotonic=time.monotonic,
    ) -> None:
        self.base_dir = os.path.abspath(os.path.expanduser(str(base_dir)))
        self.hz = max(0.0, float(hz))
        self.max_file_bytes = max(64 * 1024, int(max_file_bytes))
        self.max_runs = max(0, int(max_runs))
        self.flush_interval_s = max(0.05, float(flush_interval_s))
        self.fsync_interval_s = max(0.0, float(fsync_interval_s))
        self.queue_size = max(64, int(queue_size))
        self._clock = clock
        self._mono = monotonic

        self.run_dir: Optional[str] = None
        self.telemetry_path: Optional[str] = None
        self.events_path: Optional[str] = None
        self.segment = int(segment_index)

        self._meta: dict[str, Any] = {}
        self._close_reason = ""
        self._stop_requested = False
        self._run_stop_written = False
        self._enabled = False
        self._closed = False
        self._broken = False
        self._io_error = ""
        self._started_mono = 0.0
        self._next_sample_t = 0.0

        self._csv_file = None
        self._csv = None
        self._events_file = None
        self._thread: Optional[threading.Thread] = None
        self._queue: Optional[queue.Queue] = None

        self.rows_written = 0
        self.rows_dropped = 0
        self.events_written = 0
        self.startup_error = ""

    # -- public state ------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self._enabled and not self._closed)

    @property
    def broken(self) -> bool:
        return bool(self._broken)

    @property
    def io_error(self) -> str:
        return self._io_error

    # -- lifecycle ---------------------------------------------------------
    def start(self, meta: Optional[dict] = None) -> bool:
        """Create the run directory and start the writer thread.

        Returns False (and sets ``startup_error``) when the directory cannot be
        created or written.  Never raises: a blackbox failure must not ground
        the aircraft.
        """
        if self._enabled:
            return True
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            self._prune_old_runs()
            self.run_dir = self._new_run_dir()
            os.makedirs(self.run_dir, exist_ok=True)
            self._meta = dict(meta or {})
            self._write_manifest()
            if self._broken:
                # E.g. the log filesystem went read-only between mkdir and the
                # first write: report it instead of starting a dead blackbox.
                self.startup_error = self._io_error or "manifest write failed"
                self._close_files_quietly()
                self.run_dir = None
                self._enabled = False
                return False
            self.telemetry_path = os.path.join(self.run_dir, TELEMETRY_NAME)
            self.events_path = os.path.join(self.run_dir, EVENTS_NAME)
            # Line buffering: every row reaches the kernel immediately, so a
            # crash loses at most the row being formatted.
            self._csv_file = open(
                self.telemetry_path, "w", newline="", encoding="utf-8", buffering=1
            )
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(list(CSV_COLUMNS))
            self._events_file = open(
                self.events_path, "a", encoding="utf-8", buffering=1
            )
            self._link_latest()
        except OSError as exc:
            self.startup_error = f"{exc.__class__.__name__}: {exc}"
            self._close_files_quietly()
            self.run_dir = None
            self._enabled = False
            return False

        self._started_mono = self._mono()
        self._next_sample_t = 0.0
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._thread = threading.Thread(
            target=self._writer_loop, name="bf-anti-blackbox", daemon=True
        )
        self._thread.start()
        self._enabled = True
        self._closed = False
        self.event(
            "run_start",
            pid=os.getpid(),
            hostname=self._meta.get("hostname", ""),
            git=self._meta.get("git", ""),
            boot_id=self._meta.get("boot_id", ""),
        )
        return True

    def close(self, reason: str = "", timeout_s: float = 5.0) -> None:
        """Flush, fsync and stop the writer.  Safe to call more than once."""
        if not self._enabled or self._closed:
            return
        self._closed = True
        self._close_reason = str(reason)
        try:
            if self._queue is not None:
                # The writer emits run_stop after draining, so its row/event
                # counters are the final ones.
                self._queue.put(_STOP, timeout=2.0)
        except Exception:
            pass
        finally:
            # Even if the queue was full (slow card), the writer exits once it
            # has drained; the flag is its fallback stop condition.
            self._stop_requested = True
            thread, self._thread = self._thread, None
            if thread is not None and thread.is_alive():
                thread.join(timeout=timeout_s)
            self._enabled = False
            self._update_manifest()
            self._close_files_quietly()

    def __enter__(self) -> "TelemetryLog":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close("context_exit")

    # -- producer side (control loop) -------------------------------------
    def due(self, now: float) -> bool:
        """True when a row is due at monotonic time ``now`` (does not consume)."""
        if not self.enabled:
            return False
        if self.hz <= 0.0:
            return True
        return float(now) >= self._next_sample_t

    def sample(self, now: float, fields: dict[str, Any]) -> bool:
        """Enqueue one telemetry row if the sample period elapsed.

        ``fields`` is a mapping of CSV column -> value.  Unknown keys are
        ignored and missing columns are written empty.
        """
        if not self.due(now):
            return False
        period = (1.0 / self.hz) if self.hz > 0.0 else 0.0
        self._next_sample_t = float(now) + period
        row = {key: fields.get(key, "") for key in CSV_COLUMNS if key not in ("t_wall", "t_mono")}
        row["t_wall"] = _iso_now(self._clock)
        row["t_mono"] = f"{float(now):.3f}"
        return self._put(("row", row), block=False)

    def event(self, name: str, **fields: Any) -> bool:
        """Append one JSON event line (state transition, failure, ...)."""
        if not self.enabled:
            return False
        return self._put(("event", self._event_line(name, **fields)), block=False)

    def _event_line(self, name: str, **fields: Any) -> str:
        record: dict[str, Any] = {
            "t_wall": _iso_now(self._clock),
            "t_mono": round(self._mono(), 3),
            "uptime_s": round(max(0.0, self._mono() - self._started_mono), 3),
            "event": str(name),
        }
        for key, value in fields.items():
            if isinstance(value, float) and not math.isfinite(value):
                value = ""
            elif isinstance(value, (str, int, float, bool)) or value is None:
                pass
            else:
                value = str(value)
            record[key] = value
        try:
            return json.dumps(record, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return json.dumps({"event": str(name), "encode_error": True})

    # -- queueing ----------------------------------------------------------
    def _put(self, item, block: bool) -> bool:
        if self._queue is None or self._broken:
            self.rows_dropped += 1
            return False
        try:
            if block:
                self._queue.put(item, timeout=1.0)
            else:
                self._queue.put_nowait(item)
            return True
        except queue.Full:
            self.rows_dropped += 1
            return False
        except Exception:
            self.rows_dropped += 1
            return False

    # -- writer thread -----------------------------------------------------
    def _writer_loop(self) -> None:
        last_flush = self._mono()
        last_fsync = last_flush
        while True:
            try:
                item = self._queue.get(timeout=0.2)  # type: ignore[union-attr]
            except queue.Empty:
                # Only advance the timers when the action actually ran;
                # otherwise an idle queue would keep pushing fsync out forever.
                now = self._mono()
                if now - last_flush >= self.flush_interval_s:
                    self._flush()
                    last_flush = now
                if self.fsync_interval_s >= 0.0 and now - last_fsync >= self.fsync_interval_s:
                    self._fsync()
                    last_fsync = now
                if self._stop_requested and (self._queue is None or self._queue.empty()):
                    self._finish_run()
                    break
                continue
            if item is _STOP:
                self._finish_run()
                break
            kind, payload = item
            if self._broken:
                continue
            try:
                if kind == "row":
                    self._csv.writerow([payload.get(col, "") for col in CSV_COLUMNS])
                    self.rows_written += 1
                    self._rotate_if_needed()
                else:
                    self._events_file.write(payload + "\n")
                    self.events_written += 1
            except OSError as exc:
                self._fail_io(exc)
            except (ValueError, csv.Error) as exc:
                self._fail_io(exc)
            now = self._mono()
            if now - last_flush >= self.flush_interval_s:
                self._flush()
                last_flush = now
            if self.fsync_interval_s >= 0.0 and now - last_fsync >= self.fsync_interval_s:
                self._fsync()
                last_fsync = now
        self._flush()
        self._fsync()

    def _finish_run(self) -> None:
        """Write the final run_stop event exactly once (writer thread)."""
        if self._run_stop_written:
            return
        self._run_stop_written = True
        if self._broken or self._events_file is None:
            return
        try:
            self._events_file.write(
                self._event_line(
                    "run_stop",
                    reason=self._close_reason,
                    rows=self.rows_written,
                    dropped=self.rows_dropped,
                    io_error=self._io_error,
                )
                + "\n"
            )
            self.events_written += 1
        except OSError as exc:
            self._fail_io(exc)

    def _flush(self) -> None:
        for handle in (self._csv_file, self._events_file):
            if handle is None:
                continue
            try:
                handle.flush()
            except OSError as exc:
                self._fail_io(exc)
                return

    def _fsync(self) -> None:
        if self.fsync_interval_s < 0.0:
            return
        for handle in (self._csv_file, self._events_file):
            if handle is None:
                continue
            try:
                os.fsync(handle.fileno())
            except (OSError, ValueError) as exc:
                # ENOSPC/EIO: keep flying, stop trusting the card for fsync.
                self._fail_io(exc)
                return

    def _fail_io(self, exc: BaseException) -> None:
        if self._broken:
            return
        self._broken = True
        self._io_error = f"{exc.__class__.__name__}: {exc}"
        try:
            msg = (
                f"[bf_anti] telemetry log disabled after I/O error: {self._io_error} "
                f"(run dir {self.run_dir})\n"
            )
            sys.stderr.write(msg)
            sys.stderr.flush()
        except Exception:
            pass

    # -- rotation ----------------------------------------------------------
    def _rotate_if_needed(self) -> None:
        """Rotate when the current segment exceeds max_file_bytes.

        Called from the writer thread only, so no locking is needed.
        """
        if self._csv_file is None or self.telemetry_path is None:
            return
        try:
            size = self._csv_file.tell()
        except (OSError, ValueError):
            return
        if size < self.max_file_bytes:
            return
        try:
            self._csv_file.flush()
            os.fsync(self._csv_file.fileno())
            self._csv_file.close()
            self.segment += 1
            rotated = os.path.join(self.run_dir or "", f"telemetry.{self.segment:03d}.csv")
            os.replace(self.telemetry_path, rotated)
            self._csv_file = open(
                self.telemetry_path, "w", newline="", encoding="utf-8", buffering=1
            )
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow(list(CSV_COLUMNS))
            self.event("telemetry_rotated", segment=self.segment)
        except OSError as exc:
            self._fail_io(exc)

    # -- run directory helpers --------------------------------------------
    def _new_run_dir(self) -> str:
        now = float(self._clock())
        stamp = datetime.fromtimestamp(now).strftime("%Y%m%d_%H%M%S")
        if now < 1_600_000_000.0:
            # No RTC/NTP yet: use boot id + uptime so runs stay distinguishable.
            stamp = "unsynced_" + (_read_boot_id()[:8] or "noboot")
        base = os.path.join(self.base_dir, RUN_PREFIX + stamp)
        candidate = base
        suffix = 1
        while os.path.exists(candidate):
            suffix += 1
            candidate = f"{base}_{suffix}"
        return candidate

    def _link_latest(self) -> None:
        if not self.base_dir or not self.run_dir:
            return
        link = os.path.join(self.base_dir, LATEST_LINK)
        tmp = os.path.join(self.base_dir, f".{LATEST_LINK}.tmp")
        try:
            if os.path.lexists(tmp):
                os.unlink(tmp)
            os.symlink(os.path.basename(self.run_dir), tmp)
            os.replace(tmp, link)
        except OSError:
            # FAT/exFAT log dirs cannot hold symlinks; latest is a convenience.
            try:
                if os.path.lexists(tmp):
                    os.unlink(tmp)
            except OSError:
                pass

    def _prune_old_runs(self) -> None:
        if self.max_runs <= 0:
            return
        try:
            entries = [
                entry
                for entry in os.listdir(self.base_dir)
                if entry.startswith(RUN_PREFIX)
                and os.path.isdir(os.path.join(self.base_dir, entry))
                and os.path.isfile(os.path.join(self.base_dir, entry, MANIFEST_NAME))
            ]
        except OSError:
            return
        def _mtime(name: str) -> float:
            try:
                return os.path.getmtime(os.path.join(self.base_dir, name))
            except OSError:
                return 0.0

        # Sort by creation time, not name: an unsynced clock produces
        # run_unsynced_* names that would otherwise sort after dated runs.
        # mtime is the primary key; the run name is a deterministic tie-breaker
        # when several directories are created inside the same filesystem tick.
        entries.sort(key=lambda name: (_mtime(name), name))
        # Keep max_runs-1 so the new run does not push the count over.
        while len(entries) >= self.max_runs:
            victim = entries.pop(0)
            try:
                shutil.rmtree(os.path.join(self.base_dir, victim))
            except OSError:
                pass

    def _write_manifest(self) -> None:
        if not self.run_dir:
            return
        payload = dict(self._meta)
        payload.update(
            {
                "run": os.path.basename(self.run_dir),
                "run_dir": self.run_dir,
                "started_wall": _iso_now(self._clock),
                "started_mono": round(self._mono(), 3),
                "telemetry_file": TELEMETRY_NAME,
                "events_file": EVENTS_NAME,
                "columns": list(CSV_COLUMNS),
            }
        )
        path = os.path.join(self.run_dir, MANIFEST_NAME)
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
                fh.write("\n")
            os.replace(tmp, path)
        except OSError as exc:
            self._fail_io(exc)

    def _update_manifest(self) -> None:
        if not self.run_dir:
            return
        try:
            path = os.path.join(self.run_dir, MANIFEST_NAME)
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError):
            payload = dict(self._meta)
        payload.update(
            {
                "ended_wall": _iso_now(self._clock),
                "duration_s": round(max(0.0, self._mono() - self._started_mono), 3),
                "rows": self.rows_written,
                "rows_dropped": self.rows_dropped,
                "events_written": self.events_written,
                "io_error": self._io_error,
            }
        )
        tmp = os.path.join(self.run_dir, MANIFEST_NAME + ".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
                fh.write("\n")
            os.replace(tmp, os.path.join(self.run_dir, MANIFEST_NAME))
        except OSError:
            pass

    def _close_files_quietly(self) -> None:
        for handle in (self._csv_file, self._events_file):
            if handle is None:
                continue
            try:
                handle.close()
            except OSError:
                pass
        self._csv_file = None
        self._events_file = None
        self._csv = None


def safe_float(value: Any, digits: int = 2) -> Any:
    """Round a finite float for CSV; return "" for None/inf/nan/garbage."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return round(number, digits)
