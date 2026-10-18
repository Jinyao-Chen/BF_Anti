# Legacy: RDK X5 companion + Pico CVBS video

Archived: 2026-10-18
Status: not maintained, not part of the Pi5 flight path.

## Why archived

The active airframe path is now:

    Pi5 + AI HAT+ Hailo-8L
      -> onboard/app.py
      -> USB UVC camera + Hailo YOLOv8
      -> RCMux -> Betaflight (FC-side CRSF receiver, BOXMSPOVERRIDE)
      -> OSD -> /dev/fb0 -> J7 composite -> 5.8 GHz VTX

The RDK X5 used a different model format (BPU .bin), a local CRSF receiver, and
a Pico co-processor to generate PAL over SPI. None of that is used by the Pi5
mainline. Keeping it at the repository root made the active tree ambiguous.

## Contents

| Path | What it is |
|---|---|
| detect.py | RDK X5 YOLO/BPU main loop + tracking + MSP override |
| run_detect.sh | legacy launcher for detect.py |
| tracker.py | IoU/EMA tracker used by detect.py |
| tracking_controller.py | legacy PBVS tracking state machine |
| video.py | PAL frame encoding -> SPI -> Pico |
| pico_fw/ | RP2040 PIO/DMA CVBS firmware |
| motors.py | legacy props-off motor step test |
| crsf.py | local CRSF parser used by the X5 companion |
| scripts/ | X5 pinmux, GPIO/SPI probes, USB camera probes, old systemd unit |
| model/*.bin | X5 BPU quantized model (not the Hailo HEF) |

The Hailo ONNX source model remains active at model/v1_yolov8_mAP50_0.879.onnx.

## Runtime caveats

- These files may import X5-only SDKs (rdk_model_zoo, spidev, ultralytics_yolo_det)
  that are not installed on the Pi5.
- detect.py adds the repository root to sys.path so shared active modules
  (msp.py, rc_mux.py) remain importable, but the X5 SDK path is still external.
- The root test suite still covers the legacy tracking controller and CRSF parser
  through tests/test_tracking_controller.py and tests/test_crsf.py; they import
  legacy.rdk_x5_pico.* explicitly.
- Do not add new features here. If a legacy idea is still needed, port it into
  onboard/ or tools/ and test it against the Pi5 path.

## Old docs

Historical RDK X5/Pico notes were moved to docs/archive/2026-10-x5-pico/.
The docs/ tree is local-only by repository policy; the index is docs/README.md.
