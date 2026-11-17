# Hailo DFC / HEF build notes

## Version matrix

- DFC 3.33.0
- HailoRT 4.23.0 on the Pi5
- Hailo Model Zoo v2.17 (only needed for the hmz pipeline/check_env)
- Python 3.10, x86_64 Linux build host

DFC is a Hailo Developer Zone wheel, not a PyPI package. Keep it in a local
virtualenv; do not install it on the Pi.

## Local build (this workspace)

The working venv is at ~/hailo_local/dfc_venv (outside the repo). It has DFC
3.33.0, TensorFlow CPU 2.18 and the NMS dependencies. The DFC HSim op library
loads libcudart; on a CPU-only host set:

    DFC_LD_PRELOAD=~/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12

Use the venv Python explicitly:

    PYTHON=~/hailo_local/dfc_venv/bin/python

## Why the raw ONNX needs help

model/v1_yolov8_mAP50_0.879.onnx has the decoded single output
output0 [1,5,8400]. DFC 3.33 rejects the DFL Reshape in that graph with
UnsupportedShuffleLayerError.

The working recipe ends at the six raw conv heads and fuses NMS:

- end nodes: /model.22/cv2.0/cv2.0.2/Conv, /model.22/cv3.0/cv3.0.2/Conv,
  /model.22/cv2.1/cv2.1.2/Conv, /model.22/cv3.1/cv3.1.2/Conv,
  /model.22/cv2.2/cv2.2.2/Conv, /model.22/cv3.2/cv3.2.2/Conv
- model script: yolov8n_drone_nms.alls
- NMS config: yolov8n_drone_nms_config.json (1 class, 640x640,
  max 100 proposals/class)

build_hef.sh already uses these defaults. From the repo root:

    HAILO_MODEL_ZOO=~/hailo_local/hailo_model_zoo \
    DFC_LD_PRELOAD=~/.local/lib/python3.10/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 \
    PYTHON=~/hailo_local/dfc_venv/bin/python \
    bash tools/hailo/build_hef.sh \
      --calib build/calib_smoke/calib.npy \
      --out build/hef/yolov8n_drone_nms_smoke.hef

A real build must use at least ~1024 real target-domain calibration frames
(prepare_calib.py enforces real-frame provenance). The 3-frame
build/calib_smoke set is only for pipeline validation and must not be used
for accuracy claims.

## Verify on the Pi

Copy the HEF and run:

    hailortcli parse-hef build/hef/yolov8n_drone_nms_smoke.hef
    hailortcli run build/hef/yolov8n_drone_nms_smoke.hef

The HEF contains an NMS postprocess; onboard/detector/hailo.py consumes
HailoRT NMS streams via parse_hailo_nms().

