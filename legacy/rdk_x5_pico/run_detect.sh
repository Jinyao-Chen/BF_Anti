#!/bin/bash
# BF_Anti 检测 + 追踪启动脚本
# 架构: 接收机→FC(CRSF) + X5→FC(MSP) — BOXMSPOVERRIDE 融合
#
# 用法:
#   ./run_detect.sh              # 地面测试 (只检测, 不追踪)
#   ./run_detect.sh track        # 地面追踪 (base_throttle=1100, CH8 开关切入)
#   ./run_detect.sh fly          # 飞行追踪 (base_throttle=1550, CH8 开关切入)

set -e

MODEL="${BF_MODEL:-$HOME/BF_Anti/model/v1_yolov8_mAP50_0.879_bayese_640x640_nv12.bin}"
CAM_W="${CAM_W:-1280}"
CAM_H="${CAM_H:-720}"
CAM_FPS="${CAM_FPS:-30}"
SCORE_THRES="${SCORE_THRES:-0.65}"
TRACKING_SWITCH="${TRACKING_SWITCH:-6}"   # CH7 / AUX3 (与 BOXMSPOVERRIDE 同一个开关)

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

case "${1:-bench}" in
  fly|flight)
    echo "=== 飞行追踪: 油门=1550, CH${TRACKING_SWITCH} 开关切入 ==="
    echo "    飞手操作: AUX1 ARM → AUX3 OVERRIDE → CH${TRACKING_SWITCH} 切入追踪"
    exec python3 "$BASE_DIR/detect.py" \
      --model-path "$MODEL" \
      --cam-width "$CAM_W" --cam-height "$CAM_H" --cam-fps "$CAM_FPS" \
      --track --score-thres "$SCORE_THRES" \
      --tracking --base-throttle 1550 \
      --tracking-switch "$TRACKING_SWITCH"
    ;;
  track|ground)
    echo "=== 地面追踪: 油门=1100, CH${TRACKING_SWITCH} 开关切入 ==="
    echo "    ⚠️ 拆桨! 飞手操作: AUX1 ARM → AUX3 OVERRIDE → CH${TRACKING_SWITCH} 切入"
    exec python3 "$BASE_DIR/detect.py" \
      --model-path "$MODEL" \
      --cam-width "$CAM_W" --cam-height "$CAM_H" --cam-fps "$CAM_FPS" \
      --track --score-thres "$SCORE_THRES" \
      --tracking --base-throttle 1100 \
      --tracking-switch "$TRACKING_SWITCH"
    ;;
  bench|detect|*)
    echo "=== 基准检测 (无追踪) ==="
    exec python3 "$BASE_DIR/detect.py" \
      --model-path "$MODEL" \
      --cam-width "$CAM_W" --cam-height "$CAM_H" --cam-fps "$CAM_FPS" \
      --track --score-thres "$SCORE_THRES"
    ;;
esac
