#!/usr/bin/env bash
# Build the 1-class drone HEF for the Hailo-8L on the Pi5.
#
# Two pipelines:
#   dfc (default)  DFC 3.33 translate/optimize/compile of the single-output
#                  YOLOv8 ONNX (model/v1_yolov8_mAP50_0.879.onnx). NMS runs in
#                  onboard/detector/postprocess.py on the Pi CPU.
#   hmz            Hailo Model Zoo v2.17 compile with on-device NMS. Requires
#                  an HMZ-compatible (raw-head) checkpoint plus a matching
#                  model script, or a custom YAML that resolves one.
#
# DFC 3.33.0 is installed in the local venv for this workspace. The official
# wheel is still required on a clean build host; without it this script exits
# early with BLOCKED and the exact command it would have run.
set -euo pipefail

# Use the DFC virtualenv explicitly, e.g.
#   PYTHON=$HOME/hailo_local/dfc_venv/bin/python tools/hailo/build_hef.sh ...
PYTHON="${PYTHON:-python3}"
# DFC's HSim op library needs libcudart symbols on a CPU-only host. Set this
# to the local libcudart path if ClientRunner reports cudaPeekAtLastError:
#   DFC_LD_PRELOAD=/path/to/libcudart.so.12
if [ -n "${DFC_LD_PRELOAD:-}" ]; then
  export LD_PRELOAD="$DFC_LD_PRELOAD"
fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PIPELINE=dfc
ONNX="$REPO_ROOT/model/v1_yolov8_mAP50_0.879.onnx"
CALIB="$REPO_ROOT/build/calib/calib.npy"
CALIB_DIR="$REPO_ROOT/build/calib/images"
OUT="$REPO_ROOT/model/yolov8n_drone_h8l.hef"
NETWORK=yolov8n_drone
HW_ARCH=hailo8l
CLASSES=1
HMZ_DIR="${HAILO_MODEL_ZOO:-}"
YAML=""
MODEL_SCRIPT=""
ALLS="$REPO_ROOT/tools/hailo/yolov8n_drone_nms.alls"
END_NODES=(
  /model.22/cv2.0/cv2.0.2/Conv
  /model.22/cv3.0/cv3.0.2/Conv
  /model.22/cv2.1/cv2.1.2/Conv
  /model.22/cv3.1/cv3.1.2/Conv
  /model.22/cv2.2/cv2.2.2/Conv
  /model.22/cv3.2/cv3.2.2/Conv
)
DRY_RUN=0
SKIP_CHECK=0

usage() {
  cat <<'USAGE'
Build the BF_Anti Hailo-8L drone HEF.

Usage: tools/hailo/build_hef.sh [options]

  --pipeline dfc|hmz     dfc = raw-output ONNX + CPU NMS (default)
                         hmz = HMZ v2.17 compile with on-device NMS
  --onnx PATH            input ONNX (default model/v1_yolov8_mAP50_0.879.onnx)
  --calib PATH           calib.npy for the dfc pipeline
  --calib-dir DIR        letterboxed jpg dir for the hmz pipeline
  --out PATH             output HEF (default model/yolov8n_drone_h8l.hef)
  --network NAME         network name (default yolov8n_drone)
  --hw-arch ARCH         hailo8l (default) or hailo8
  --classes N            class count for HMZ NMS (default 1)
  --hmz-dir DIR          Hailo Model Zoo v2.17 checkout (or HAILO_MODEL_ZOO)
  --yaml PATH            custom HMZ network YAML
  --model-script PATH    .alls model script for the hmz pipeline
  --alls PATH            .alls script for the dfc pipeline
  --skip-check           do not refuse when check_env --role build fails
  --dry-run              print commands, do not compile
  -h, --help             this text

Blocked gate: DFC 3.33.0 is a Hailo Developer Zone download, not on PyPI.
Prepare calibration first:
  tools/hailo/prepare_calib.py --pi-frames DIR --drone-images DIR ...
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pipeline) PIPELINE="$2"; shift 2 ;;
    --onnx) ONNX="$2"; shift 2 ;;
    --calib) CALIB="$2"; shift 2 ;;
    --calib-dir) CALIB_DIR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --network) NETWORK="$2"; shift 2 ;;
    --hw-arch) HW_ARCH="$2"; shift 2 ;;
    --classes) CLASSES="$2"; shift 2 ;;
    --hmz-dir) HMZ_DIR="$2"; shift 2 ;;
    --yaml) YAML="$2"; shift 2 ;;
    --model-script) MODEL_SCRIPT="$2"; shift 2 ;;
    --alls) ALLS="$2"; shift 2 ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$PIPELINE" in
  dfc|hmz) ;;
  *) echo "error: --pipeline must be dfc or hmz" >&2; exit 2 ;;
esac

if [ ! -f "$ONNX" ]; then
  echo "error: ONNX not found: $ONNX" >&2
  exit 2
fi

echo "BF_Anti HEF build"
echo "  pipeline : $PIPELINE"
echo "  onnx     : $ONNX"
echo "  hw_arch  : $HW_ARCH"
echo "  out      : $OUT"

if [ "$SKIP_CHECK" -eq 0 ]; then
  if ! "$PYTHON" "$REPO_ROOT/tools/hailo/check_env.py" --role build; then
    echo "" >&2
    echo "BLOCKED: build environment check failed (see BLOCKED lines above)." >&2
    echo "Install Hailo DFC 3.33.0 and Hailo Model Zoo v2.17, or pass --skip-check." >&2
    exit 3
  fi
fi

if [ "$PIPELINE" = "dfc" ]; then
  if [ ! -f "$CALIB" ]; then
    echo "error: calibration npy not found: $CALIB" >&2
    echo "run tools/hailo/prepare_calib.py first" >&2
    exit 2
  fi
  echo "DFC command:"
  echo "  $PYTHON tools/hailo/compile_onnx.py --onnx $ONNX --calib $CALIB --out $OUT --name $NETWORK --hw-arch $HW_ARCH --alls $ALLS --end-nodes ${END_NODES[*]}"
  if [ "$DRY_RUN" -eq 1 ]; then
    "$PYTHON" "$REPO_ROOT/tools/hailo/compile_onnx.py" --onnx "$ONNX" --calib "$CALIB" --out "$OUT" --name "$NETWORK" --hw-arch "$HW_ARCH" --alls "$ALLS" --end-nodes "${END_NODES[@]}" --dry-run || true
    exit 0
  fi
  "$PYTHON" "$REPO_ROOT/tools/hailo/compile_onnx.py" --onnx "$ONNX" --calib "$CALIB" --out "$OUT" --name "$NETWORK" --hw-arch "$HW_ARCH" --alls "$ALLS" --end-nodes "${END_NODES[@]}"
else
  if [ -z "$HMZ_DIR" ]; then
    echo "error: --hmz-dir (or HAILO_MODEL_ZOO) is required for --pipeline hmz" >&2
    exit 2
  fi
  if [ ! -f "$HMZ_DIR/hailo_model_zoo/cfg/networks/yolov8n.yaml" ] && [ -z "$YAML" ]; then
    echo "error: no yolov8n.yaml in $HMZ_DIR and no --yaml given" >&2
    exit 2
  fi
  if [ -z "$YAML" ]; then
    YAML="$HMZ_DIR/hailo_model_zoo/cfg/networks/yolov8n.yaml"
  fi
  if [ -z "$MODEL_SCRIPT" ]; then
    echo "BLOCKED: --model-script is required for the hmz pipeline." >&2
    echo "The official HMZ yolov8n.alls references the raw-head checkpoint layer" >&2
    echo "names (conv41..conv63); model/v1_yolov8_mAP50_0.879.onnx is a single" >&2
    echo "output0 graph, so it needs a matching .alls generated from the parsed HAR." >&2
    echo "Use --pipeline dfc (CPU NMS) or supply --model-script." >&2
    exit 3
  fi
  if [ ! -d "$CALIB_DIR" ]; then
    echo "error: --calib-dir not found: $CALIB_DIR" >&2
    exit 2
  fi
  if ! command -v hailomz >/dev/null 2>&1; then
    echo "error: hailomz not in PATH (install Hailo Model Zoo v2.17)" >&2
    exit 3
  fi
  BUILD_DIR="$REPO_ROOT/build/hmz"
  mkdir -p "$BUILD_DIR"
  echo "HMZ command (in $BUILD_DIR):"
  echo "  hailomz compile $NETWORK --yaml $YAML --ckpt $ONNX --calib-path $CALIB_DIR --model-script $MODEL_SCRIPT --classes $CLASSES --hw-arch $HW_ARCH"
  if [ "$DRY_RUN" -eq 1 ]; then
    exit 0
  fi
  (
    cd "$BUILD_DIR"
    hailomz compile "$NETWORK" --yaml "$YAML" --ckpt "$ONNX" --calib-path "$CALIB_DIR" --model-script "$MODEL_SCRIPT" --classes "$CLASSES" --hw-arch "$HW_ARCH"
  )
  HEF_FOUND="$(find "$BUILD_DIR" -maxdepth 1 -name '*.hef' -printf '%T@ %p' | sort -n | tail -1 | cut -d' ' -f2-)"
  if [ -z "$HEF_FOUND" ]; then
    echo "error: hailomz finished but no HEF was produced in $BUILD_DIR" >&2
    exit 4
  fi
  mkdir -p "$(dirname "$OUT")"
  cp "$HEF_FOUND" "$OUT"
fi

if [ ! -f "$OUT" ]; then
  echo "error: expected HEF was not produced: $OUT" >&2
  exit 4
fi
echo "wrote $OUT ($(stat -c%s "$OUT") bytes)"
sha256sum "$OUT" || true
if command -v hailortcli >/dev/null 2>&1; then
  echo "HEF summary:"
  hailortcli parse-hef "$OUT" | head -25 || true
fi
echo "next: tools/hailo/bench_hef.py --hef $OUT --frames 300 --json build/bench/hef.json"
