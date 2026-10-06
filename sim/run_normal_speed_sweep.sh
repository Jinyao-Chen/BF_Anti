#!/usr/bin/env bash
# Multi-speed normal-world reference tracking sweep.
# Usage: ./sim/run_normal_speed_sweep.sh [speeds...]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [ "$#" -gt 0 ]; then
    SPEEDS=("$@")
else
    SPEEDS=(5 8 10 12)
fi
OUTDIR="${NORMAL_SWEEP_OUTDIR:-/tmp/bf_anti_speed_sweep}"
mkdir -p "$OUTDIR"

for v in "${SPEEDS[@]}"; do
    echo "======== speed $v m/s ========"
    NORMAL_TRIAL_LAUNCH_LOG="$OUTDIR/launch_${v}.log" \
    NORMAL_TRIAL_RUN_LOG="$OUTDIR/run_${v}.log" \
    ./sim/run_normal_trial.sh \
        --world betaflight_world \
        --cruise "$v" \
        --duration 20 \
        --settle 4 \
        --period 0.05 \
        --log "$OUTDIR/speed_${v}.csv" \
        --plot "$OUTDIR/speed_${v}.png" || true
done

echo "sweep done; outputs in $OUTDIR"
