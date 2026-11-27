"""RC-controlled graceful shutdown for the Pi 5.

The Pi 5 has no battery-backed power path: pulling the flight pack while Linux
is still writing the SD card can corrupt the filesystem.  This module is the
software half of the shutdown procedure.  When power.aux_channel is configured
(e.g. CH7/AUX3 in the example profile), that pilot switch is read from MSP_RC;
when it is held high with the FC disarmed and throttle at idle, the runtime
asks systemd to halt.  The pilot waits for the Pi activity LED to go out, then
unplugs the pack.  The code default is -1 (shutdown disabled) so a minimal
config cannot halt the Pi by accident.

Safety rules:
* a low->high edge is required, so a switch left high at boot cannot shut the
  Pi down before the pilot sees it;
* the channel must be fresh and the FC must report disarmed;
* throttle must be at idle;
* the switch must stay high for hold_s seconds;
* the command is systemctl poweroff --no-block (a polkit rule lets the onboard
  user do this without an interactive prompt); the service exits only after
  the job is queued.

This module is stdlib-only and does not import OpenCV/Hailo, so the guard can
be unit-tested on a laptop.
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Optional

from onboard.config import PowerConfig
from onboard.types import FcState

LOG = logging.getLogger("bf_anti.power")

# systemd's --no-block returns as soon as the job is queued; the service can
# then stop its threads and flush the blackbox while the shutdown proceeds.
_POWEROFF_CMD = ("/usr/bin/systemctl", "poweroff", "--no-block")


class ShutdownGuard:
    """Pure decision logic for one RC channel; never executes the command."""

    def __init__(self, cfg: PowerConfig):
        self.cfg = cfg
        self._seen_low = False
        self._high_since: Optional[float] = None
        self._fired = False
        self._cooldown_until = 0.0

    @property
    def enabled(self) -> bool:
        return int(self.cfg.aux_channel) >= 0

    @property
    def fired(self) -> bool:
        return bool(self._fired)

    def reset(self) -> None:
        self._seen_low = False
        self._high_since = None
        self._fired = False
        self._cooldown_until = 0.0

    def retry_after(self, now: float, delay_s: float = 10.0) -> None:
        """The system command failed; allow another attempt while held."""
        self._fired = False
        self._high_since = None
        self._cooldown_until = float(now) + max(0.0, float(delay_s))

    def _drop_high(self) -> None:
        self._high_since = None

    def _require_low(self) -> None:
        self._seen_low = False
        self._high_since = None

    def update(self, now: float, state: Optional[FcState]) -> bool:
        """Return True exactly once when a valid shutdown hold completes."""
        cfg = self.cfg
        if not self.enabled or self._fired or float(now) < self._cooldown_until:
            return False
        if state is None:
            self._require_low()
            return False

        # Never act on stale RC or a stale armed flag.  A reconnect must be
        # followed by a deliberate low->high switch edge.
        if (state.rc_age_s(now) > cfg.max_rc_age_s
                or state.status_age_s(now) > cfg.max_status_age_s):
            self._require_low()
            return False

        channels = tuple(state.rc or ())
        idx = int(cfg.aux_channel)
        if idx >= len(channels):
            self._require_low()
            return False

        raw = int(channels[idx])
        if raw < int(cfg.high_threshold):
            # Hysteresis: only the genuinely low segment arms the edge
            # detector.  1300-1700 is a dead band; 1500 or a boot-time 0 must
            # not count as a deliberate low.
            low_min = int(getattr(cfg, "min_low_threshold", 885))
            low_max = int(getattr(cfg, "low_threshold", 1300))
            if low_min <= raw <= low_max:
                self._seen_low = True
            elif raw < low_min:
                # 0/unplugged/failsafe is not a pilot low edge.
                self._seen_low = False
            self._drop_high()
            return False
        if not self._seen_low:
            # High at boot (or after stale data) is ignored until the pilot
            # deliberately moves the switch low once.
            return False

        if cfg.require_disarmed and bool(state.armed):
            self._require_low()
            return False

        throttle = int(channels[2]) if len(channels) > 2 else 1500
        if throttle > int(cfg.max_throttle):
            # Pulling the throttle down after this must not complete a hold
            # that began while the throttle was high: require a fresh cycle.
            self._require_low()
            return False

        if self._high_since is None:
            self._high_since = float(now)
        if float(now) - self._high_since >= float(cfg.hold_s):
            self._fired = True
            return True
        return False


def request_poweroff(timeout_s: float = 5.0) -> tuple[bool, str]:
    """Ask systemd to halt; return (queued, human-readable detail)."""
    cmd = list(_POWEROFF_CMD)
    if not os.path.exists(cmd[0]):
        cmd[0] = "systemctl"
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=max(1.0, float(timeout_s)),
            check=False,
        )
    except Exception as exc:  # missing binary, timeout, permission denied
        return False, repr(exc)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    detail = "; ".join(part for part in (stdout, stderr) if part)
    if proc.returncode == 0:
        return True, detail or "poweroff queued"
    return False, "exit %s: %s" % (proc.returncode, detail or "no output")
