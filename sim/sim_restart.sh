#!/bin/bash
# 安全重启 headless SITL+Gazebo（放在脚本里，避免 pkill 模式匹配到调用者自身命令行）
#   ./sim/sim_restart.sh            # 重启并等待 MSP 就绪
#   ./sim/sim_restart.sh stop       # 只停止
set -u
cd "$(dirname "$0")/.."
WORLD="${BF_ANTI_WORLD:-$PWD/sim/worlds/bf_anti_p_strike_light_cam.sdf}"
pkill -9 -f "gz sim" 2>/dev/null
pkill -9 -f "betaflight_SITL" 2>/dev/null
pkill -f "launch_sitl" 2>/dev/null
sleep 3
if [ "${1:-start}" = "stop" ]; then echo "[sim] stopped"; exit 0; fi
LOG="${LOG:-/tmp/sim_restart.log}"
BF_ROOT="${BF_ROOT:-$HOME/betaflight}" BF_ANTI_WORLD="$WORLD" \
    nohup ./sim/launch_sitl.sh --headless > "$LOG" 2>&1 &
echo "[sim] launching (log $LOG)"
for i in $(seq 1 70); do
    if (exec 3<>/dev/tcp/127.0.0.1/5761) 2>/dev/null; then echo "[sim] MSP up after ${i}s"; sleep 8; echo "[sim] ready"; exit 0; fi
    sleep 1
done
echo "[sim] ERROR: MSP 未就绪"; exit 1
