#!/bin/bash
# BF_Anti 纯 P 视觉制导打击 —— 一键实时演示
#
#   ./sim/run_p_strike_live.sh              # 实时窗口（相机画面 + 检测框 + HUD）
#   GUI=1 ./sim/run_p_strike_live.sh        # 同时打开 Gazebo 3D 界面
#   DURATION=40 ./sim/run_p_strike_live.sh  # 加长任务时间
#   VIDEO=/tmp/run.mp4 ./sim/run_p_strike_live.sh
#
# 窗口内容：机体相机实拍画面、绿色检测框（合成 bbox；加 --yolo-host 则用真 YOLO）、
# 画面中心十字（制导瞄准点）、顶部 HUD（阶段/真值距离/视觉测距/图像误差/姿态/RC）。
# 按 q 或 ESC 提前退出；退出时自动关闭 SITL 与 Gazebo。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BF_ROOT="${BF_ROOT:-$HOME/betaflight}"
WORLD="${BF_ANTI_WORLD:-$SCRIPT_DIR/worlds/bf_anti_p_strike_light_cam.sdf}"
DURATION="${DURATION:-32}"
VIDEO="${VIDEO:-/tmp/p_strike_live.mp4}"
GUI="${GUI:-0}"
LOG="${LOG:-/tmp/p_strike_live.csv}"

SIM_PID=""
TUNNEL_PID=""
YOLO_HOST="${YOLO_HOST:-127.0.0.1}"
# SSH 隧道目标：默认「当前用户 @ $YOLO_HOST」；连远程 GPU 机时显式给 YOLO_SSH 或 YOLO_USER+YOLO_HOST
YOLO_USER="${YOLO_USER:-${USER}}"
YOLO_SSH="${YOLO_SSH:-${YOLO_USER}@${YOLO_HOST}}"
# 检测后端：pt = 我们的 YOLOv8-L(anti_uav_best.pt, :9300)
#           onnx = 第三方的 YOLOv8n(v1_yolov8_mAP50_0.879.onnx, :9301)
# 两个后端都在 GPU 机的 **/dev/nvidia0** (= GPU-40c96051 = nvidia-smi index 1) 上，
# 由 start_yolo.sh 按 UUID 绑卡；绝不能碰 nvidia-smi index 0（其他用户的卡）。
YOLO_BACKEND="${YOLO_BACKEND:-pt}"
case "${YOLO_BACKEND}" in
    onnx) YOLO_PORT_DEFAULT=9301 ;;
    *)    YOLO_PORT_DEFAULT=9300 ;;
esac
YOLO_PORT="${YOLO_PORT:-$YOLO_PORT_DEFAULT}"
# 换模型的框松紧不同 -> W_eff 不同 -> 距离尺度整体偏。实测 ONNX 的框比 .pt 宽 18.8%，
# 所以 2.0 -> 2.38；不改这张表会让 z_t = r*sin(el) 系统性偏低。
if [ "${YOLO_BACKEND}" = "onnx" ]; then TARGET_WIDTH_DEFAULT=2.38; else TARGET_WIDTH_DEFAULT=2.0; fi
TARGET_WIDTH="${TARGET_WIDTH:-$TARGET_WIDTH_DEFAULT}"
# MASK_HORIZ=none|pos|vel|both：把飞控给**制导律**的水平位置/速度清零，
# 用来预演"无 GPS / 无光流"的真机（none=正常仿真）。见 run_p_vision_strike.py --help。
MASK_HORIZ="${MASK_HORIZ:-none}"
# 靶机匀速运动（可选）：TARGET_VEL_E / TARGET_VEL_N（m/s，ENU）。不设则靶机静止。
TARGET_VEL_E="${TARGET_VEL_E:-}"
TARGET_VEL_N="${TARGET_VEL_N:-}"
# YOLO_DIRECT=1: 不走 SSH 隧道，直接把检测服务放在 YOLO_HOST 上（例如 RDK X5 直连）。
YOLO_DIRECT="${YOLO_DIRECT:-0}"
# 远程工作目录 / 远程拉起服务的命令（换板子、换机器时设 YOLO_REMOTE_DIR 或 YOLO_REMOTE_CMD）。
YOLO_REMOTE_DIR="${YOLO_REMOTE_DIR:-.}"
YOLO_REMOTE_CMD="${YOLO_REMOTE_CMD:-cd ${YOLO_REMOTE_DIR} && (setsid bash start_yolo.sh ${YOLO_BACKEND} >> logs/live_remote_start.log 2>&1 &)}"

probe_yolo() {
    local P_HOST="${1:-127.0.0.1}"
    python3 - "$P_HOST" "$YOLO_PORT" <<'PY' >/dev/null 2>&1
import socket, struct, sys, json
import numpy as np, cv2
s = socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=6)
ok, jpg = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
s.sendall(struct.pack("!I", len(jpg.tobytes())) + jpg.tobytes())
h = b""
while len(h) < 4: h += s.recv(4 - len(h))
n = struct.unpack("!I", h)[0]; buf = b""
while len(buf) < n: buf += s.recv(n - len(buf))
json.loads(buf); s.close()
PY
}

ensure_yolo() {
    if [ "${YOLO_DIRECT}" = "1" ]; then
        # 直连模式：检测服务已经在 YOLO_HOST 上跑（X5 旧服务见 legacy/rdk_x5_pico/scripts/yolo_server_board.py）
        if probe_yolo "${YOLO_HOST}"; then
            echo "[live] YOLO 直连可用 (${YOLO_HOST}:${YOLO_PORT})"; return 0
        fi
        echo "[live] WARNING: 直连 ${YOLO_HOST}:${YOLO_PORT} 不可用。"
        echo "        先在板上起服务: python3 legacy/rdk_x5_pico/scripts/yolo_server_board.py --host 0.0.0.0 --port ${YOLO_PORT}"
        return 1
    fi
    if probe_yolo; then echo "[live] YOLO 服务可用 (127.0.0.1:${YOLO_PORT})"; return 0; fi
    if ! (exec 3<>/dev/tcp/127.0.0.1/${YOLO_PORT}) 2>/dev/null; then
        echo "[live] 建立 SSH 隧道 → ${YOLO_SSH}:${YOLO_PORT}"
        ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 \
            -L ${YOLO_PORT}:127.0.0.1:${YOLO_PORT} "${YOLO_SSH}" &
        TUNNEL_PID=$!
        for _ in $(seq 1 15); do
            (exec 3<>/dev/tcp/127.0.0.1/${YOLO_PORT}) 2>/dev/null && break
            sleep 1
        done
    fi
    if probe_yolo; then echo "[live] YOLO 服务可用"; return 0; fi
    echo "[live] 远程拉起 YOLO 服务 ..."
    ssh "${YOLO_SSH}" "${YOLO_REMOTE_CMD}" || true
    for _ in $(seq 1 30); do probe_yolo && { echo "[live] YOLO 服务就绪"; return 0; }; sleep 2; done
    echo "[live] WARNING: YOLO 服务不可用，改用合成 bbox"
    return 1
}

cleanup() {
    echo ""
    echo "[live] 清理 SITL / Gazebo ..."
    if [ -n "${TGT_PID}" ]; then kill "${TGT_PID}" 2>/dev/null || true; fi
    if [ -n "${TUNNEL_PID}" ]; then kill "${TUNNEL_PID}" 2>/dev/null || true; fi
    if [ -n "${SIM_PID}" ]; then kill "${SIM_PID}" 2>/dev/null || true; fi
    pkill -f "betaflight_SITL_SITL_GAZEBO.elf" 2>/dev/null || true
    pkill -f "gz sim.*bf_anti" 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[live] 完成。日志: ${LOG}   录像: ${VIDEO}"
}
trap cleanup EXIT INT TERM

if [ -n "${DISPLAY:-}" ]; then
    echo "[live] DISPLAY=${DISPLAY} → 实时窗口可用"
else
    echo "[live] 警告: 没有 DISPLAY，--show 将自动降级为录像 ${VIDEO}"
fi

if [ "${YOLO:-0}" = "1" ]; then
    if ! ensure_yolo; then YOLO=0; fi
fi

echo "[live] 启动 SITL + Gazebo ($([ "${GUI}" = "1" ] && echo GUI || echo headless)) ..."
if [ "${GUI}" = "1" ]; then
    BF_ROOT="${BF_ROOT}" BF_ANTI_WORLD="${WORLD}" "${SCRIPT_DIR}/launch_sitl.sh" &
else
    BF_ROOT="${BF_ROOT}" BF_ANTI_WORLD="${WORLD}" "${SCRIPT_DIR}/launch_sitl.sh" --headless &
fi
SIM_PID=$!

# 等 MSP 端口（SITL 就绪）——固件在 Gazebo 起来前不接受 FDM，故再等几秒
for _ in $(seq 1 60); do
    if (exec 3<>/dev/tcp/127.0.0.1/5761) 2>/dev/null; then break; fi
    if ! kill -0 "${SIM_PID}" 2>/dev/null; then
        echo "[live] ERROR: 仿真启动失败" >&2
        exit 1
    fi
    sleep 1
done
sleep 8

# 靶机匀速运动（可选）：在打击前起一个后台进程持续发 cmd_vel
TGT_PID=""
if [ -n "${TARGET_VEL_E}" ] || [ -n "${TARGET_VEL_N}" ]; then
    echo "[live] 靶机匀速运动：E=${TARGET_VEL_E:-0} N=${TARGET_VEL_N:-0} m/s"
    python3 "${SCRIPT_DIR}/move_target.py" --vel-e "${TARGET_VEL_E:-0}" \
        --vel-n "${TARGET_VEL_N:-0}" --duration "$((DURATION + 20))" --print-hz 0.5 \
        > /tmp/live_target.log 2>&1 &
    TGT_PID=$!
    sleep 1
fi

echo "[live] 开始打击（窗口标题: BF_Anti P-only visual strike）..."
# YOLO=1 时用远程真检测（默认走 SSH 隧道到 127.0.0.1:9300），否则用合成 bbox
if [ "${YOLO:-0}" = "1" ]; then
    # 真 YOLO 的框含桨臂，46 m 处 20.4 px ⟹ 等效宽度 ≈2.0 m（实测标定）；
    # 用 0.55 m 反算会把距离算小 3.6 倍，高度环会被带偏。
    DETECTOR="--yolo-host ${YOLO_HOST} --yolo-port ${YOLO_PORT} --target-width ${TARGET_WIDTH}"
else
    DETECTOR="--synthetic"
fi

python3 "${SCRIPT_DIR}/run_p_vision_strike.py" \
    ${DETECTOR} --mask-own-horiz "${MASK_HORIZ}" \
    --show --video "${VIDEO}" --duration "${DURATION}" \
    --inner-model "" --cam-link cam_link --view camera \
    --hover-pwm 1430 --tilt 20 --climb-alt ${CLIMB_ALT:-5} --stage-pitch 8 \
    --max-pitch-down 25 --angle-limit 40 --kp-pitch 13 --kp-yaw 220 --max-yaw-rate 300 \
    --kp-roll ${KP_ROLL:-45} --kp-z ${KP_Z:-1.0} --max-roll 30 --kp-speed 4 --v-max 5 --kp-velalign ${KP_VELALIGN:-1.5} \
    --align-gate 0.05 --align-pitch 5 --lift-ff --lost-timeout 0.9 --lost-frames 8 \
    --vz-time-to-go ${T2G:-1} --ki-z ${KI_Z:-0.05} \
    --err-soft ${ERR_SOFT:-0.35} --tau-max ${TAU_MAX:-3.0} \
    --velalign-min-speed ${VA_MIN:-2.5} --velalign-soft ${VA_SOFT:-10} \
    --range-lpf ${RANGE_LPF:-0.25} --closing-max ${CLOSE_MAX:-25} \
    --vmax-close-range ${VMAX_CLOSE_R:-12} --vmax-close-floor ${VMAX_CLOSE_F:-3} \
    --reacquire-angle ${REACQ_ANGLE:-60} --reacquire-rate ${REACQ_RATE:-200} \
    --log "${LOG}" --plot "${LOG%.csv}.png" "$@"