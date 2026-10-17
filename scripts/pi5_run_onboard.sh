#!/usr/bin/env bash
# pi5_run_onboard.sh - activate the venv and exec the onboard runtime.
#
# The systemd unit calls this script. It resolves the checkout from its own
# location (or BF_ANTI_HOME), prefers the venv interpreter, verifies the
# config exists, clears DISPLAY (there is no X11 on the aircraft), and then
# execs python so systemd signals reach the runtime directly.
#
# Usage:
#   scripts/pi5_run_onboard.sh [extra onboard.app args]
#
# Environment overrides:
#   BF_ANTI_HOME    checkout root   (default: parent of this script)
#   BF_ANTI_VENV    venv root       (default: BF_ANTI_HOME/.venv)
#   BF_ANTI_CONFIG  config JSON     (default: BF_ANTI_HOME/config/onboard.json)

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
DEFAULT_PREFIX=$(cd -- "$SCRIPT_DIR/.." && pwd)
PREFIX=${BF_ANTI_HOME:-$DEFAULT_PREFIX}
VENV=${BF_ANTI_VENV:-$PREFIX/.venv}
CONFIG=${BF_ANTI_CONFIG:-$PREFIX/config/onboard.json}

unset DISPLAY

if [ ! -r "$CONFIG" ]; then
  printf 'error: config not found: %s\n' "$CONFIG" >&2
  printf 'hint: cp %s/config/onboard.example.json %s and edit it\n' "$PREFIX" "$CONFIG" >&2
  exit 2
fi

if [ -x "$VENV/bin/python" ]; then
  PYTHON="$VENV/bin/python"
else
  PYTHON=$(command -v python3 || true)
  if [ -z "$PYTHON" ]; then
    printf 'error: no venv python at %s and no python3 on PATH\n' "$VENV/bin/python" >&2
    exit 2
  fi
  printf '[WARN] venv missing at %s; falling back to %s\n' "$VENV" "$PYTHON" >&2
fi

cd -- "$PREFIX"
exec "$PYTHON" -m onboard.app --config "$CONFIG" "$@"
