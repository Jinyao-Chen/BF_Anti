#!/bin/bash
# 单次真 YOLO 打击试验：重启仿真 -> 跑一发 -> 打印最近真值距离
#   ./sim/trial_p_strike.sh <名称> [额外参数...]
set -u
cd "$(dirname "$0")/.."
NAME="$1"; shift
LOG="/tmp/trial_${NAME}.csv"
# 检测后端可切换（A/B 用）：默认仍是 H100 隧道 127.0.0.1:9300
YOLO_HOST="${YOLO_HOST_OVERRIDE:-127.0.0.1}"
YOLO_PORT="${YOLO_PORT:-9300}"
# 防呆：YOLO 后端不通时，真检测永远不返回 -> 全程 SEARCH -> "脱靶"数字毫无意义
# （已实测踩坑：隧道静默断开导致 5 发数据作废）
if ! python3 -c "import socket;socket.create_connection(('${YOLO_HOST}',${YOLO_PORT}),timeout=4)" 2>/dev/null; then
    echo "[${NAME}] ABORT: YOLO 后端 ${YOLO_HOST}:${YOLO_PORT} 不可用（隧道: ssh -N -L ${YOLO_PORT}:127.0.0.1:9300 GPU_USER@GPU_HOST）"
    exit 2
fi
DUMP_ARGS=()
if [ -n "${DUMP_DIR:-}" ]; then
    mkdir -p "${DUMP_DIR}/${NAME}"
    DUMP_ARGS=(--dump-dir "${DUMP_DIR}/${NAME}" --dump-hz "${DUMP_HZ:-10}")
fi
pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f betaflight_SITL 2>/dev/null; sleep 3
BF_ROOT="$HOME/betaflight" BF_ANTI_WORLD="$PWD/sim/worlds/bf_anti_p_strike_light_cam.sdf" \
    ./sim/launch_sitl.sh --headless > "/tmp/trial_${NAME}_sim.log" 2>&1 &
SIM_PID=$!
sleep 52
timeout 90 python3 sim/run_p_vision_strike.py --yolo-host "${YOLO_HOST}" --yolo-port "${YOLO_PORT}" \
    --min-conf 0.10 --inner-model "" --cam-link cam_link --view camera \
    --hover-pwm 1430 --tilt 20 --climb-alt 5 --stage-pitch 8 --duration 30 \
    --target-width 2.0 --hit-radius 0.8 \
    --max-pitch-down 25 --angle-limit 40 --kp-pitch 13 --kp-yaw 220 --max-yaw-rate 300 \
    --kp-roll 45 --kp-z 1.0 --max-roll 30 --kp-speed 4 --v-max 5 --kp-velalign 1.5 \
    --vz-time-to-go 1 --ki-z 0.05 \
    --align-gate 0.05 --align-pitch 5 --lift-ff --lost-timeout 0.9 --lost-frames 8 \
    --err-soft 0.35 --tau-max 3.0 --velalign-min-speed 2.5 --velalign-soft 10 \
    --range-lpf 0.25 --closing-max 25 --vmax-close-range 12 --vmax-close-floor 3 \
    --reacquire-angle 60 --reacquire-rate 200 \
    "${DUMP_ARGS[@]}" --log "$LOG" "$@" > "/tmp/trial_${NAME}.out" 2>&1
MIN=$(python3 -c "
import csv
try:
    rows=list(csv.DictReader(open('$LOG')))
    print('%.3f'%min(float(r['truth_range']) for r in rows) if rows else 'NaN')
except Exception:
    print('NaN')")
LOCK=$(grep -o "locked *: *[A-Za-z]*" "/tmp/trial_${NAME}.out" | head -1)
TRACKPCT=$(python3 -c "
import csv
try:
    rs=list(csv.DictReader(open('$LOG')))
    n=sum(1 for r in rs if r['phase'] in ('TRACK','TERMINAL','COAST'))
    print('%.0f'%(100.0*n/len(rs)) if rs else '0')
except Exception:
    print('0')")
if [ "${TRACKPCT}" -lt 20 ]; then
    echo "[${NAME}] WARNING: 有效跟踪帧仅 ${TRACKPCT}% -> 本次数据多半是丢检测，不是脱靶，请勿采信"
fi
echo "[${NAME}] min_truth_range=${MIN} m   ${LOCK}"
kill $SIM_PID 2>/dev/null; pkill -9 -f "gz sim" 2>/dev/null; pkill -9 -f betaflight_SITL 2>/dev/null