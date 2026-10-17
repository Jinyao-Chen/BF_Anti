#!/usr/bin/env bash
# Stage the precompiled COCO yolov8s HEF for pipeline bring-up ONLY.
#
# The COCO model has 80 classes and was not trained for this airframe. It may
# be used to prove the HailoRT/HEF path works (device, buffers, NMS output,
# postprocess), and must NEVER be quoted as drone detection accuracy.
#
# Source order:
#   1. $HEF_SRC
#   2. /usr/share/hailo-models/<model>.hef (Pi5 hailo-models package)
#   3. model/ or build/ copies already in the repo
#   4. --download: Hailo Model Zoo v2.17.0 S3 artifact
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$REPO_ROOT/model/precompiled"
MODEL=yolov8s_h8l
FORCE=0
DOWNLOAD=0

usage() {
  cat <<'USAGE'
Stage a precompiled HEF for Hailo pipeline bring-up only.

Usage: tools/hailo/fetch_precompiled.sh [options]

  --dest DIR       destination directory (default model/precompiled)
  --model NAME     yolov8s_h8l (default) or yolov6n_h8l
  --download       fetch the HMZ v2.17.0 S3 artifact when no local copy exists
  --force          overwrite an existing destination file
  -h, --help       this text

Output file is renamed with an explicit _coco80 suffix and a NOTICE.txt
records that it is not the 1-class drone model.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --download) DOWNLOAD=1; shift ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

TARGET="$DEST/${MODEL}_coco80.hef"
if [ -f "$TARGET" ] && [ "$FORCE" -eq 0 ]; then
  echo "already staged: $TARGET (use --force to replace)"
  exit 0
fi
mkdir -p "$DEST"

SOURCE=""
for candidate in "${HEF_SRC:-}" "/usr/share/hailo-models/${MODEL}.hef" "$REPO_ROOT/model/${MODEL}.hef" "$REPO_ROOT/model/precompiled/${MODEL}.hef"; do
  if [ -n "$candidate" ] && [ -f "$candidate" ]; then
    SOURCE="$candidate"
    break
  fi
done

if [ -z "$SOURCE" ] && [ "$DOWNLOAD" -eq 1 ]; then
  URL="https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.17.0/hailo8l/${MODEL}.hef"
  echo "downloading $URL"
  mkdir -p "$DEST"
  curl -fL --retry 3 -o "$TARGET.tmp" "$URL"
  mv "$TARGET.tmp" "$TARGET"
  SOURCE="$TARGET"
fi

if [ -z "$SOURCE" ]; then
  echo "error: no local ${MODEL}.hef found and --download not set." >&2
  echo "On the Pi5 this normally lives at /usr/share/hailo-models/${MODEL}.hef." >&2
  echo "On an x86 build host use: --download" >&2
  exit 3
fi

if [ "$SOURCE" != "$TARGET" ]; then
  cp "$SOURCE" "$TARGET"
fi
cat > "$DEST/NOTICE.txt" <<EOF
PIPELINE BRING-UP ONLY.
${MODEL}.hef is a precompiled COCO 80-class model. It is not the BF_Anti
1-class drone detector and must never be used for accuracy claims, tuning, or
flight decisions. Use it only to verify HailoRT device/buffer/NMS plumbing.
The final HEF is built from model/v1_yolov8_mAP50_0.879.onnx with
tools/hailo/build_hef.sh once Hailo DFC 3.33.0 is available.
EOF
echo "staged $TARGET"
sha256sum "$TARGET" || true
if command -v hailortcli >/dev/null 2>&1; then
  echo "HEF summary:"
  hailortcli parse-hef "$TARGET" | head -20 || true
fi
