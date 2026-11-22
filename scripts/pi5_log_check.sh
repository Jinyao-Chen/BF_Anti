#!/usr/bin/env bash
# pi5_log_check.sh - "why are there no logs after boot?" field check.
#
# Run this on the Pi after a boot (no root required for most checks):
#   bash scripts/pi5_log_check.sh
#   bash scripts/pi5_log_check.sh --root /home/pi5/BF_Anti --config /home/pi5/BF_Anti/config/onboard.json
#
# It answers, in order:
#   1. did the service start this boot, and is it still running?
#   2. is journald persistent, or did the reboot erase it?
#   3. did the blackbox write a run_* directory since boot, with rows?
#   4. what do the newest rows/events say, and is the SD card full?
#
# Exit status: 0 = a run exists since this boot and has telemetry rows,
# 1 = something is missing (hints printed), 2 = bad arguments.

set -u

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PREFIX=${BF_ANTI_HOME:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
CONFIG=$PREFIX/config/onboard.json
UNIT=bf-anti-onboard.service
LOG_DIR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --root) PREFIX="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

ok()   { printf '[ OK ] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
bad()  { printf '[FAIL] %s\n' "$*"; }
info() { printf '[INFO] %s\n' "$*"; }
step() { printf '\n== %s ==\n' "$*"; }

ISSUES=0

resolve_log_dir() {
  if [ -n "$LOG_DIR" ]; then
    printf '%s\n' "$LOG_DIR"
    return
  fi
  local parsed=""
  if [ -f "$CONFIG" ] && command -v python3 >/dev/null 2>&1; then
    parsed=$(python3 - "$CONFIG" "$PREFIX" <<'PY' 2>/dev/null || true
import json, os, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        raw = json.load(fh)
except Exception:
    sys.exit(1)
value = os.path.expandvars(os.path.expanduser(str(raw.get("log_dir", "logs"))))
if not os.path.isabs(value):
    value = os.path.join(sys.argv[2], value)
print(os.path.abspath(value))
PY
)
  fi
  if [ -n "$parsed" ]; then
    printf '%s\n' "$parsed"
  else
    printf '%s/logs\n' "$PREFIX"
  fi
}

step "1. time base and boot"
date -Is
if uptime -s >/dev/null 2>&1; then
  info "booted at: $(uptime -s)"
  BOOT_EPOCH=$(date -d "$(uptime -s)" +%s 2>/dev/null || echo 0)
else
  BOOT_EPOCH=0
fi
if command -v timedatectl >/dev/null 2>&1; then
  NTP=$(timedatectl show -p NTPSynchronized --value 2>/dev/null || true)
  [ -n "$NTP" ] && info "NTP synchronized: $NTP"
fi

step "2. systemd unit"
if ! command -v systemctl >/dev/null 2>&1; then
  warn "systemctl not present; not running on the Pi?"
  exit 1
fi
if systemctl is-enabled --quiet "$UNIT" 2>/dev/null; then
  ok "$UNIT is enabled (starts at boot)"
else
  bad "$UNIT is NOT enabled -> sudo systemctl enable $UNIT"
  ISSUES=$((ISSUES + 1))
fi
ACTIVE=$(systemctl is-active "$UNIT" 2>/dev/null || true)
if [ "$ACTIVE" = "active" ]; then
  ok "$UNIT is active"
else
  bad "$UNIT is $ACTIVE -> journalctl -u $UNIT -b --no-pager"
  ISSUES=$((ISSUES + 1))
fi
systemctl show "$UNIT" \
  -p NRestarts -p ExecMainStartTimestamp -p ActiveEnterTimestamp 2>/dev/null | sed 's/^/       /'

step "3. journald persistence"
if [ -d /var/log/journal ]; then
  ok "/var/log/journal exists (logs survive reboot)"
  journalctl --disk-usage 2>/dev/null | sed 's/^/       /' || true
else
  warn "no /var/log/journal: journald is volatile, previous boots are gone"
  info "fix: sudo bash $PREFIX/scripts/pi5_enable_persistent_journal.sh"
fi
journalctl --list-boots 2>/dev/null | tail -n 3 | sed 's/^/       /' || true
if journalctl -u "$UNIT" -b --no-pager -n 8 >/dev/null 2>&1; then
  info "last $UNIT lines this boot:"
  journalctl -u "$UNIT" -b --no-pager -n 8 | sed 's/^/       /'
fi

step "4. blackbox directory"
LOG_DIR=$(resolve_log_dir)
info "log_dir = $LOG_DIR"
if [ ! -d "$LOG_DIR" ]; then
  bad "log_dir does not exist -> the service never created it"
  info "check config log_dir and permissions; run: systemctl status $UNIT"
  exit 1
fi
if [ -w "$LOG_DIR" ]; then
  ok "log_dir is writable by $(id -un)"
else
  warn "log_dir is not writable by $(id -un) (service runs as its own user)"
fi
df -h "$LOG_DIR" 2>/dev/null | tail -n 1 | sed 's/^/       /'

LATEST=$(find "$LOG_DIR" -maxdepth 1 -type d -name 'run_*' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -n 1 | cut -d' ' -f2-)
if [ -z "$LATEST" ]; then
  bad "no run_* directory under $LOG_DIR"
  info "the blackbox never started; check the service journal above"
  ISSUES=$((ISSUES + 1))
  exit 1
fi
ok "latest run: $LATEST"
find "$LOG_DIR" -maxdepth 1 -type d -name 'run_*' -printf '%TY-%Tm-%Td %TH:%TM  %p\n' 2>/dev/null \
  | sort -r | head -n 5 | sed 's/^/       /'

RUN_EPOCH=$(stat -c %Y "$LATEST" 2>/dev/null || echo 0)
if [ "$BOOT_EPOCH" -gt 0 ] && [ "$RUN_EPOCH" -gt 0 ] && [ "$RUN_EPOCH" -lt "$BOOT_EPOCH" ]; then
  bad "latest run is OLDER than this boot -> service has not run since boot"
  ISSUES=$((ISSUES + 1))
else
  ok "latest run was created during this boot"
fi

step "5. telemetry rows"
CSV="$LATEST/telemetry.csv"
if [ -f "$CSV" ]; then
  LINES=$(wc -l < "$CSV")
  if [ "$LINES" -gt 1 ]; then
    ok "$((LINES - 1)) telemetry rows in $CSV"
  else
    bad "$CSV has a header but no data rows"
    ISSUES=$((ISSUES + 1))
  fi
  if [ "$ACTIVE" = "active" ]; then
    LAST_WRITE=$(stat -c %Y "$CSV" 2>/dev/null || echo 0)
    NOW_EPOCH=$(date +%s)
    if [ "$LAST_WRITE" -gt 0 ] && [ $((NOW_EPOCH - LAST_WRITE)) -gt 10 ]; then
      warn "telemetry.csv has not been updated for $((NOW_EPOCH - LAST_WRITE)) s"
      info "service is running but idle -> check events.jsonl for startup_retry/tick_error"
    else
      ok "telemetry.csv is being updated"
    fi
  fi
  info "newest rows (truncated):"
  tail -n 3 "$CSV" | cut -c1-220 | sed 's/^/       /'
else
  bad "missing $CSV"
  ISSUES=$((ISSUES + 1))
fi
if [ -f "$LATEST/events.jsonl" ]; then
  info "latest events:"
  tail -n 8 "$LATEST/events.jsonl" | cut -c1-220 | sed 's/^/       /'
fi
if [ -f "$LATEST/startup.json" ]; then
  info "manifest:"
  grep -E '"(started_wall|git|hostname|boot_id|ended_wall|rows|rows_dropped|io_error)"' \
    "$LATEST/startup.json" 2>/dev/null | sed 's/^/       /' || true
fi

step "6. quick verdict"
if command -v python3 >/dev/null 2>&1 && [ -f "$PREFIX/tools/log_summary.py" ]; then
  python3 "$PREFIX/tools/log_summary.py" --dir "$LOG_DIR" --runs 1 2>/dev/null \
    | sed 's/^/       /' || true
fi

printf '\n'
if [ "$ISSUES" -eq 0 ] && [ -s "$CSV" ] && [ "$(wc -l < "$CSV")" -gt 1 ]; then
  ok "logs are being recorded; copy the run dir before rebooting"
  exit 0
fi
warn "$ISSUES blocking issue(s); see hints above"
exit 1
