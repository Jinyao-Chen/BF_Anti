#!/usr/bin/env bash
# pi5_install.sh - idempotent Pi 5 install: apt deps, rsync layout, venv, unit.
#
# Copies this checkout to the board prefix, installs the OS packages the
# onboard runtime needs, creates a venv with system-site-packages (so the
# apt-built OpenCV and the HailoRT Python binding are visible), copies the
# example config into place if no real config exists, and installs the
# systemd unit with the install path rendered into it.
#
# It never touches WiFi credentials, never edits boot config, never reboots,
# and does not flash firmware. HailoRT itself is intentionally NOT installed
# here (it is a large vendor package): the script checks for it and warns.
#
# Usage:
#   sudo bash scripts/pi5_install.sh [options]
#   bash scripts/pi5_install.sh --dry-run [options]
#
# Options:
#   --prefix DIR        install root (default /home/pi5/BF_Anti)
#   --user USER         owning user (default pi5)
#   --source DIR        source checkout (default: parent of this script)
#   --no-apt            skip apt-get
#   --no-rsync          skip file copy
#   --no-venv           skip venv
#   --no-service        skip systemd unit
#   --enable-service    systemctl enable the unit (never starts it)
#   --persistent-journal  keep journald logs across reboots (root only)
#   --dry-run           print actions without changing anything
#   -h, --help          this text
#
# Exit codes: 0 = success, 1 = a required step failed, 2 = bad arguments.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
SOURCE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PREFIX=/home/pi5/BF_Anti
BOARD_USER=pi5
DO_APT=1
DO_RSYNC=1
DO_VENV=1
DO_SERVICE=1
ENABLE_SERVICE=0
DO_PERSISTENT_JOURNAL=0
DRY_RUN=0

usage() {
  sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --user) BOARD_USER="$2"; shift 2 ;;
    --source) SOURCE="$2"; shift 2 ;;
    --no-apt) DO_APT=0; shift ;;
    --no-rsync) DO_RSYNC=0; shift ;;
    --no-venv) DO_VENV=0; shift ;;
    --no-service) DO_SERVICE=0; shift ;;
    --enable-service) ENABLE_SERVICE=1; shift ;;
    --persistent-journal) DO_PERSISTENT_JOURNAL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

ok()   { printf '[ OK ] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
info() { printf '[INFO] %s\n' "$*"; }
step() { printf '\n== %s ==\n' "$*"; }

run() {
  printf '  +'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

quiet() {
  desc="$1"
  shift
  if [ "$DRY_RUN" -eq 1 ]; then
    info "would run: $desc"
    return 0
  fi
  if "$@" >/tmp/bf_anti_install_step.log 2>&1; then
    ok "$desc"
  else
    warn "$desc failed (see /tmp/bf_anti_install_step.log)"
  fi
}

if [ ! -d "$SOURCE" ]; then
  printf 'error: source checkout not found: %s\n' "$SOURCE" >&2
  exit 2
fi
if [ "$DO_SERVICE" -eq 1 ] && [ ! -f "$SOURCE/systemd/bf-anti-onboard.service" ]; then
  printf 'error: missing systemd/bf-anti-onboard.service under %s\n' "$SOURCE" >&2
  exit 2
fi

IS_ROOT=0
if [ "$(id -u)" -eq 0 ]; then
  IS_ROOT=1
fi

step "1. plan"
info "source:  $SOURCE"
info "prefix:  $PREFIX"
info "user:    $BOARD_USER"
info "dry-run: $DRY_RUN"

if [ "$IS_ROOT" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
  if [ "$DO_APT" -eq 1 ]; then
    warn "not root: skipping apt-get (run with sudo to install packages)"
    DO_APT=0
  fi
  if [ "$DO_SERVICE" -eq 1 ]; then
    warn "not root: skipping systemd unit install"
    DO_SERVICE=0
  fi
fi

if [ "$IS_ROOT" -eq 1 ] && ! id -u "$BOARD_USER" >/dev/null 2>&1; then
  printf 'error: board user does not exist: %s\n' "$BOARD_USER" >&2
  exit 1
fi

step "2. OS packages"
if [ "$DO_APT" -eq 1 ]; then
  run apt-get update
  run env DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-venv python3-pip python3-opencv python3-serial v4l-utils libdrm-tests rsync
else
  info "apt step skipped"
fi
if command -v hailortcli >/dev/null 2>&1; then
  ok "HailoRT present: $(hailortcli -v 2>/dev/null | head -n 1)"
else
  warn "HailoRT not found; install hailort + python3-hailort 4.23.x before flight"
fi
if command -v v4l2-ctl >/dev/null 2>&1; then
  ok "v4l2-ctl present"
else
  warn "v4l2-ctl missing (v4l-utils)"
fi

step "3. checkout layout"
if [ "$DO_RSYNC" -eq 1 ]; then
  if [ "$DRY_RUN" -eq 0 ]; then
    if [ "$IS_ROOT" -eq 1 ]; then
      install -d -o "$BOARD_USER" -g "$BOARD_USER" "$PREFIX"
    elif [ ! -d "$PREFIX" ]; then
      printf 'error: %s does not exist (run with sudo first)\n' "$PREFIX" >&2
      exit 1
    fi
  fi
  run rsync -a \
    --exclude .git --exclude __pycache__ --exclude .pytest_cache \
    --exclude .venv --exclude logs --exclude '*.log' --exclude '*.mp4' \
    "$SOURCE"/ "$PREFIX"/
  if [ "$IS_ROOT" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
    run chown -R "$BOARD_USER:$BOARD_USER" "$PREFIX"
  fi
else
  info "rsync step skipped"
fi
if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$PREFIX/logs"
  if [ "$IS_ROOT" -eq 1 ]; then
    chown "$BOARD_USER:$BOARD_USER" "$PREFIX/logs"
  fi
fi

step "4. config"
CONFIG_EXAMPLE="$PREFIX/config/onboard.example.json"
CONFIG_REAL="$PREFIX/config/onboard.json"
if [ -f "$CONFIG_REAL" ]; then
  ok "real config present: $CONFIG_REAL"
elif [ -f "$CONFIG_EXAMPLE" ] || [ "$DRY_RUN" -eq 1 ]; then
  run cp "$CONFIG_EXAMPLE" "$CONFIG_REAL"
  if [ "$IS_ROOT" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
    chown "$BOARD_USER:$BOARD_USER" "$CONFIG_REAL"
  fi
  warn "config/onboard.json created from the example; set camera/FC/safety values"
else
  warn "no config example at $CONFIG_EXAMPLE"
fi

step "4b. log directory"
LOG_DIR="$PREFIX/logs"
CONFIG_FOR_LOG_DIR="$CONFIG_REAL"
if [ ! -f "$CONFIG_FOR_LOG_DIR" ] && [ -f "$CONFIG_EXAMPLE" ]; then
  CONFIG_FOR_LOG_DIR="$CONFIG_EXAMPLE"
fi
if [ ! -f "$CONFIG_FOR_LOG_DIR" ] && [ -f "$SOURCE/config/onboard.example.json" ]; then
  CONFIG_FOR_LOG_DIR="$SOURCE/config/onboard.example.json"
fi
if [ -f "$CONFIG_FOR_LOG_DIR" ] && command -v python3 >/dev/null 2>&1; then
  PARSED_LOG_DIR=$(python3 - "$CONFIG_FOR_LOG_DIR" "$PREFIX" <<'PY' 2>/dev/null || true
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
  [ -n "$PARSED_LOG_DIR" ] && LOG_DIR="$PARSED_LOG_DIR"
fi
if [ "$DRY_RUN" -eq 0 ]; then
  mkdir -p "$LOG_DIR" || warn "could not create log directory: $LOG_DIR"
  if [ "$IS_ROOT" -eq 1 ] && [ -d "$LOG_DIR" ]; then
    chown "$BOARD_USER:$BOARD_USER" "$LOG_DIR" || true
  fi
  ok "blackbox log directory: $LOG_DIR"
else
  info "would create blackbox log directory: $LOG_DIR"
fi

step "5. Python venv"
VENV="$PREFIX/.venv"
if [ "$DO_VENV" -eq 1 ]; then
  if [ -x "$VENV/bin/python" ]; then
    ok "venv already present: $VENV"
  else
    run python3 -m venv --system-site-packages "$VENV"
    if [ "$IS_ROOT" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
      run chown -R "$BOARD_USER:$BOARD_USER" "$VENV"
    fi
  fi
  if [ "$DRY_RUN" -eq 0 ] && [ -x "$VENV/bin/pip" ]; then
    quiet "pip upgrade" "$VENV/bin/pip" install --upgrade pip
    quiet "install pyserial" "$VENV/bin/pip" install pyserial
    if "$VENV/bin/python" -c 'import numpy, cv2, serial' >/dev/null 2>&1; then
      ok "venv imports numpy/cv2/serial"
    else
      warn "venv import check failed (numpy/cv2/serial)"
    fi
    if "$VENV/bin/python" -c 'import hailo_platform' >/dev/null 2>&1; then
      ok "venv sees system hailo_platform"
    else
      warn "venv cannot import hailo_platform (install python3-hailort system-wide)"
    fi
  fi
else
  info "venv step skipped"
fi

step "6. systemd unit"
if [ "$DO_SERVICE" -eq 1 ]; then
  UNIT_DST=/etc/systemd/system/bf-anti-onboard.service
  if [ "$DRY_RUN" -eq 1 ]; then
    info "would render systemd/bf-anti-onboard.service -> $UNIT_DST"
  else
    install -d -m 0755 /etc/bf-anti
    if [ ! -f /etc/bf-anti/onboard.env ]; then
      printf '%s\n' \
        '# Extra environment for bf-anti-onboard.service (optional).' \
        '# BF_ANTI_CONFIG=/home/pi5/BF_Anti/config/onboard.json' \
        '# BF_ANTI_VENV=/home/pi5/BF_Anti/.venv' \
        > /etc/bf-anti/onboard.env
      chmod 0644 /etc/bf-anti/onboard.env
      chown root:root /etc/bf-anti/onboard.env
      ok "created /etc/bf-anti/onboard.env"
    fi
    sed -e "s|__INSTALL_DIR__|$PREFIX|g" \
        -e "s|__BOARD_USER__|$BOARD_USER|g" \
        "$SOURCE/systemd/bf-anti-onboard.service" > "$UNIT_DST.tmp"
    chmod 0644 "$UNIT_DST.tmp"
    chown root:root "$UNIT_DST.tmp"
    mv "$UNIT_DST.tmp" "$UNIT_DST"
    systemctl daemon-reload
    ok "installed $UNIT_DST"
    if [ "$ENABLE_SERVICE" -eq 1 ]; then
      systemctl enable bf-anti-onboard.service
      ok "enabled bf-anti-onboard.service (not started)"
    else
      warn "unit NOT enabled: it will not start after boot. Use --enable-service"
      info "or run: sudo systemctl enable --now bf-anti-onboard"
    fi
  fi

  step "6b. polkit poweroff rule"
  POLKIT_RULE_SRC="$SOURCE/systemd/50-bf-anti-poweroff.rules"
  POLKIT_RULE_DST=/etc/polkit-1/rules.d/50-bf-anti-poweroff.rules
  if [ ! -f "$POLKIT_RULE_SRC" ]; then
    warn "missing $POLKIT_RULE_SRC; RC soft-shutdown will fail authentication"
  elif [ "$DRY_RUN" -eq 1 ]; then
    info "would install $POLKIT_RULE_DST"
  else
    install -d -m 0755 /etc/polkit-1/rules.d
    TMP_RULE=$(mktemp)
    sed "s|__BOARD_USER__|$BOARD_USER|g" "$POLKIT_RULE_SRC" > "$TMP_RULE"
    install -m 0644 -o root -g root "$TMP_RULE" "$POLKIT_RULE_DST"
    rm -f "$TMP_RULE"
    # Membership takes effect on next login; the unit's SupplementaryGroups
    # covers the current service, but the install user needs the groups too.
    for group in video dialout render; do
      if getent group "$group" >/dev/null 2>&1; then
        usermod -aG "$group" "$BOARD_USER" || true
      fi
    done
    ok "installed $POLKIT_RULE_DST for $BOARD_USER (polkit reloads automatically)"
  fi

  if [ "$DO_PERSISTENT_JOURNAL" -eq 1 ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
      info "would enable persistent journald"
    elif [ "$IS_ROOT" -eq 1 ]; then
      bash "$SOURCE/scripts/pi5_enable_persistent_journal.sh" || warn "persistent journal setup failed"
    else
      warn "not root: skipping persistent journald"
    fi
  else
    info "journald left as-is; add --persistent-journal to keep logs across reboots"
  fi
else
  info "systemd step skipped"
fi

step "7. next steps"
printf '%s\n' \
  "1) bash $PREFIX/scripts/pi5_bringup.sh --root $PREFIX" \
  "2) bash $PREFIX/scripts/pi5_hailo_check.sh --root $PREFIX" \
  "3) python3 $PREFIX/scripts/pi5_camera_probe.py --list-only" \
  "4) copy a real config to $PREFIX/config/onboard.json and run:" \
  "   $PREFIX/.venv/bin/python -m onboard.app --config $PREFIX/config/onboard.json" \
  "5) only after the checks are green: sudo systemctl enable --now bf-anti-onboard" \
  "6) after the next boot verify the blackbox: bash $PREFIX/scripts/pi5_log_check.sh"

if [ "$DRY_RUN" -eq 1 ]; then
  info "dry-run complete; nothing was changed"
fi
exit 0
