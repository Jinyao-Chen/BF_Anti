#!/usr/bin/env bash
# Run one normal LiftDrag real-target trial: clean old sim, start SITL/world
# with the selected config, run run_altitude_diag.py, then stop everything.
#
# Usage:
#   ./sim/run_normal_trial.sh [run_altitude_diag.py args...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORLD="${NORMAL_WORLD:-$ROOT/sim/worlds/bf_anti_intercept_real.sdf}"
CONFIG="${BF_ANTI_CONFIG:-$HOME/betaflight/src/platform/SIMULATOR/gazebo/config/sitl_gazebo_config.txt}"
LAUNCH_LOG="${NORMAL_TRIAL_LAUNCH_LOG:-/tmp/normal_trial_launch.log}"
RUN_LOG="${NORMAL_TRIAL_RUN_LOG:-/tmp/normal_trial_run.log}"

cleanup() {
    ps -eo pid,comm,args | awk '$2=="python3" && /run_altitude_diag/ {print $1}' | xargs -r kill 2>/dev/null || true
    ps -eo pid,comm,args | awk '$2=="betaflight_SITL" {print $1}' | xargs -r kill 2>/dev/null || true
    ps -eo pid,comm,args | awk '$2=="ruby" && /gz sim/ {print $1}' | xargs -r kill 2>/dev/null || true
    if [[ -n "${LAUNCH_PID:-}" ]]; then
        kill "$LAUNCH_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

cleanup
sleep 1

cd "$ROOT"
BF_ANTI_WORLD="$WORLD" BF_ANTI_CONFIG="$CONFIG" \
    ./sim/launch_sitl.sh --headless > "$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!

echo "waiting for MSP 5761..."
for _ in $(seq 1 100); do
    if (exec 3<>/dev/tcp/127.0.0.1/5761) 2>/dev/null; then
        exec 3>&- 3<&-
        echo "SITL up"
        break
    fi
    sleep 0.5
done
sleep 2

set +e
python3 sim/run_altitude_diag.py "$@" > "$RUN_LOG" 2>&1
status=$?
set -e
echo "run status=$status; log=$RUN_LOG"
cat "$RUN_LOG"
exit "$status"
