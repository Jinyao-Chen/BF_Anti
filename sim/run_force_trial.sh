#!/usr/bin/env bash
# Run one force-only-world trial: clean old sim, start SITL with a given
# config, run run_force_offboard_test.py, then stop everything.
#
# Usage:
#   ./sim/run_force_trial.sh <sitl_config.txt> [run_force_offboard_test.py args...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:?usage: run_force_trial.sh <sitl_config> [test args...]}"
shift
WORLD="${FORCE_WORLD:-$ROOT/sim/worlds/bf_anti_intercept_force_flat.sdf}"
LAUNCH_LOG="${FORCE_TRIAL_LAUNCH_LOG:-/tmp/force_trial_launch.log}"
RUN_LOG="${FORCE_TRIAL_RUN_LOG:-/tmp/force_trial_run.log}"

cleanup() {
    ps -eo pid,comm,args | awk '$2=="python3" && /run_force_offboard_test/ {print $1}' | xargs -r kill 2>/dev/null || true
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
for _ in $(seq 1 80); do
    if (exec 3<>/dev/tcp/127.0.0.1/5761) 2>/dev/null; then
        exec 3>&- 3<&-
        echo "SITL up"
        break
    fi
    sleep 0.5
done

set +e
python3 sim/run_force_offboard_test.py "$@" > "$RUN_LOG" 2>&1
status=$?
set -e
echo "run status=$status; log=$RUN_LOG"
cat "$RUN_LOG"
exit "$status"
