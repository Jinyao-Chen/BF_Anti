#!/bin/bash
# BF_Anti 检测 + 追踪启动脚本
# 用法:
#   ./run_detect.sh              # 地面测试 (base_throttle=1100, 键盘 t 切入)
#   ./run_detect.sh fly          # 飞行模式 (base_throttle=1550, CH8 开关切入)
#   ./run_detect.sh bench        # 基准检测 (无追踪)

set -e

MODEL="${BF_MODEL:-$HOME/BF_Anti/model/v1_yolov8_mAP50_0.879_bayese_640x640_nv12.bin}"
CAM_W="${CAM_W:-1280}"
CAM_H="${CAM_H:-720}"
CAM_FPS="${CAM_FPS:-30}"
SCORE_THRES="${SCORE_THRES:-0.65}"
TRACKING_SWITCH="${TRACKING_SWITCH:-7}"   # CH8 / AUX4

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

case "${1:-track}" in
  fly|flight)
    echo "=== 飞行模式: 追踪 + CRSF 摇杆检测 ==="
    exec python3 "$BASE_DIR/detect.py" \
      --model-path "$MODEL" \
      --cam-width "$CAM_W" --cam-height "$CAM_H" --cam-fps "$CAM_FPS" \
      --track --score-thres "$SCORE_THRES" \
      --tracking --base-throttle 1550 \
      --tracking-switch "$TRACKING_SWITCH"
    ;;
  bench|detect)
    echo "=== 基准检测 (无追踪) ==="
    exec python3 "$BASE_DIR/detect.py" \
      --model-path "$MODEL" \
      --cam-width "$CAM_W" --cam-height "$CAM_H" --cam-fps "$CAM_FPS" \
      --track --score-thres "$SCORE_THRES"
    ;;
  track|ground|*)
    echo "=== 地面测试: 追踪 + 键盘 t 切入 ==="
    exec python3 "$BASE_DIR/detect.py" \
      --model-path "$MODEL" \
      --cam-width "$CAM_W" --cam-height "$CAM_H" --cam-fps "$CAM_FPS" \
      --track --score-thres "$SCORE_THRES" \
      --tracking --base-throttle 1100
    ;;
esac
