#!/usr/bin/env bash
# deploy_pi5.sh - sync this checkout to the Pi and install the boot service.
#
# Run from the laptop (the Pi must be reachable over SSH):
#   bash scripts/deploy_pi5.sh
#   PI_HOST=user@host PI_PASS='...' bash scripts/deploy_pi5.sh
#
# It never edits WiFi, never flashes firmware, and never starts the motors.
# The service is installed+enabled; pass --no-start to leave it stopped.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
SOURCE=$(cd -- "$SCRIPT_DIR/.." && pwd)
PI_HOST=${PI_HOST:?set PI_HOST=user@host (for example pi@raspberrypi.local)}
PREFIX=${PI_PREFIX:-/home/pi5/BF_Anti}
PI_PASS=${PI_PASS:-}
START=1
SSH_OPTS=${PI_SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new}

usage() { sed -n '2,16p' "$0" | sed 's/^# {0,1}//'; }
while [ $# -gt 0 ]; do
  case "$1" in
    --no-start) START=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

say() { printf '\n== %s ==\n' "$*"; }
run_ssh() { ssh $SSH_OPTS "$PI_HOST" "$@"; }
sudo_remote() {
  if [ -n "$PI_PASS" ]; then
    printf '%s\n' "$PI_PASS" | ssh $SSH_OPTS "$PI_HOST" "sudo -S -p '' $*"
  else
    run_ssh "sudo -n $*"
  fi
}

say "1. connectivity"
run_ssh 'hostname; uptime; ip -brief addr show eth0 2>/dev/null || true'

say "2. sync checkout"
ssh $SSH_OPTS "$PI_HOST" "mkdir -p '$PREFIX'"
rsync -a --delete-delay \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '.pytest_cache' --exclude 'build' --exclude 'logs' \
  --exclude 'sim/results' --exclude 'sim/models' --exclude '*.mp4' --exclude '*.log' \
  --exclude '*.whl' --exclude '*.SLDPRT' --exclude 'Anti-UAV' --exclude 'legacy' \
  --exclude '*.onnx' --exclude 'docs' \
  --exclude 'config/onboard.json' --exclude 'config/onboard.json.bak.*' \
  --exclude 'config/onboard.pi5.json' \
  -e "ssh $SSH_OPTS" "$SOURCE/" "$PI_HOST:$PREFIX/"
run_ssh "chmod +x '$PREFIX'/scripts/*.sh '$PREFIX'/scripts/*.py 2>/dev/null || true"

say "3. config"
if [ -f "$SOURCE/config/onboard.pi5.json" ]; then
  # Keep the previous aircraft config for post-flight comparison.
  run_ssh "if [ -f '$PREFIX/config/onboard.json' ]; then cp '$PREFIX/config/onboard.json' '$PREFIX/config/onboard.json.bak.\$(date +%Y%m%d-%H%M%S)'; fi"
  rsync -a -e "ssh $SSH_OPTS" "$SOURCE/config/onboard.pi5.json" "$PI_HOST:$PREFIX/config/onboard.json"
  echo "installed config/onboard.pi5.json -> config/onboard.json"
elif ! run_ssh "test -f '$PREFIX/config/onboard.json'"; then
  run_ssh "cp '$PREFIX/config/onboard.example.json' '$PREFIX/config/onboard.json'"
  echo "WARN: no config/onboard.pi5.json; copied the example. Measure camera tilt/HFOV and set the HEF before flight."
fi

say "4. install service (no apt, no rsync repeat)"
# A deployed board already has its venv.  Skipping pip keeps an offline
# flight-line update from hanging on a package index it cannot reach.
if run_ssh "test -x '$PREFIX/.venv/bin/python'"; then
  VENV_FLAG=--no-venv
else
  VENV_FLAG=
fi
sudo_remote "bash '$PREFIX/scripts/pi5_install.sh' --no-apt --no-rsync --enable-service $VENV_FLAG"

say "5. service state"
if [ "$START" -eq 1 ]; then
  sudo_remote "systemctl restart bf-anti-onboard.service"
  sleep 3
fi
sudo_remote "systemctl --no-pager --full status bf-anti-onboard.service || true"
echo
echo "logs: ssh $PI_HOST 'journalctl -u bf-anti-onboard -f'"
