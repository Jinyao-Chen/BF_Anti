"""Logging bootstrap: journald + per-run file log, crash dumps.

The unit runs with ``StandardError=journal``, so the console handler is what
``journalctl -u bf-anti-onboard`` shows.  The file handler is what survives a
reboot when journald is volatile, and the faulthandler dump is what survives a
native crash in HailoRT/OpenCV (which otherwise produces no Python traceback).

Everything here is best-effort: a read-only log directory or a full disk must
degrade to journald logging, never abort the onboard runtime.
"""
from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import os
import signal
import sys
import threading
from typing import IO, Optional

LOGGER_NAME = "bf_anti"
_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s "
    "[%(threadName)s]: %(message)s"
)
_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Keep the crash file object alive for the whole process; faulthandler writes
# through the descriptor, not through Python's logging machinery.
_FAULT_FILE: Optional[IO[str]] = None
_THREAD_ERROR_HOOK = None


def set_thread_error_hook(callback) -> None:
    """Register a best-effort callback(thread_name, exc) for worker crashes."""
    global _THREAD_ERROR_HOOK
    _THREAD_ERROR_HOOK = callback


def level_from_name(name: str, default: int = logging.INFO) -> int:
    if isinstance(name, int):
        return int(name)
    return getattr(logging, str(name or "").upper(), default)


def setup_console_logging(level: str | int = "INFO") -> None:
    """Configure stderr logging (journald under systemd).  Idempotent."""
    numeric = level_from_name(level)
    root = logging.getLogger()
    if any(getattr(h, "_bf_anti_console", False) for h in root.handlers):
        root.setLevel(numeric)
        return
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handler._bf_anti_console = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(numeric)
    logging.raiseExceptions = False


def attach_file_handler(
    path: str,
    level: str | int = "INFO",
    *,
    max_bytes: int = 8 * 1024 * 1024,
    backup_count: int = 5,
) -> Optional[logging.Handler]:
    """Add a rotating file handler; return None when the path is unwritable.

    The PID is part of the caller-provided path (per-run directory), so several
    rotations only happen during a very long flight.
    """
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            path,
            maxBytes=max(64 * 1024, int(max_bytes)),
            backupCount=max(0, int(backup_count)),
            encoding="utf-8",
            delay=False,
        )
    except OSError as exc:
        logging.getLogger(LOGGER_NAME).warning(
            "file logging disabled (%s): %s", path, exc
        )
        return None
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handler.setLevel(level_from_name(level))
    logging.getLogger().addHandler(handler)
    return handler


def install_excepthooks() -> None:
    """Route uncaught main-thread and worker-thread exceptions into the log."""
    crash = logging.getLogger(LOGGER_NAME + ".crash")

    previous = sys.excepthook

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, tb)
            return
        crash.critical("uncaught exception", exc_info=(exc_type, exc, tb))
        previous(exc_type, exc, tb)

    sys.excepthook = _excepthook

    def _thread_excepthook(args):
        if args.exc_type is SystemExit:
            return
        thread_name = getattr(args.thread, "name", "?")
        crash.critical(
            "uncaught exception in thread %s", thread_name,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
        hook = _THREAD_ERROR_HOOK
        if hook is not None:
            try:
                hook(thread_name, args.exc_value)
            except Exception:
                pass

    threading.excepthook = _thread_excepthook


def install_faulthandler(path: str) -> Optional[IO[str]]:
    """Dump Python stacks on SIGSEGV/SIGABRT and on SIGUSR1 into ``path``.

    The SIGUSR1 hook lets a hung service be inspected without restarting it::

        systemctl kill -s USR1 bf-anti-onboard
        tail -n 40 <run dir>/faulthandler.log
    """
    global _FAULT_FILE
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        handle = open(path, "a", encoding="utf-8", buffering=1)
    except OSError:
        return None
    try:
        faulthandler.enable(file=handle, all_threads=True)
    except (RuntimeError, ValueError):
        handle.close()
        return None
    if hasattr(signal, "SIGUSR1") and threading.current_thread() is threading.main_thread():
        try:
            faulthandler.register(signal.SIGUSR1, file=handle, all_threads=True)
        except (RuntimeError, ValueError, AttributeError):
            pass
    _FAULT_FILE = handle
    return handle


def shutdown_faulthandler() -> None:
    global _FAULT_FILE
    try:
        faulthandler.disable()
    except Exception:
        pass
    handle, _FAULT_FILE = _FAULT_FILE, None
    if handle is not None:
        try:
            handle.close()
        except OSError:
            pass
