#!/usr/bin/env python3
"""Compile a single-output YOLOv8 ONNX to a Hailo-8L HEF with DFC 3.33.

This is the raw-head path: the ONNX output stays (1, 4+C, N) and NMS runs in
onboard/detector/postprocess.py on the Pi CPU. It is the path that matches
model/v1_yolov8_mAP50_0.879.onnx (one class, 8400 anchors). On-device NMS
requires an HMZ-compatible checkpoint and a model script; use
tools/hailo/build_hef.sh --pipeline hmz for that.

BLOCKED GATE: Hailo DFC is not installed in this workspace. The official
3.33.0 wheel must be provided by the user (Hailo Developer Zone account); it
is not on PyPI. This script therefore cannot be executed end-to-end here and
its DFC calls are documented, version-pinned templates, not a validated pass.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

def find_repo_root() -> Path:
    """Locate the checkout even when this file is copied to /tmp for a probe."""
    here = Path(__file__).resolve()
    candidates = [here.parent, *here.parents, Path.cwd()]
    for candidate in candidates:
        if (candidate / "onboard").is_dir() and (candidate / "tools" / "hailo").is_dir():
            return candidate
    return here.parent


REPO_ROOT = find_repo_root()
EXPECTED_DFC = "3.33.0"
DEFAULT_ALLS = REPO_ROOT / "tools" / "hailo" / "yolov8n_drone_raw.alls"


def dfc_version() -> Optional[str]:
    try:
        module = importlib.import_module("hailo_sdk_client")
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as pkg_version

            version = pkg_version("hailo_dataflow_compiler")
        except Exception:
            version = None
    return None if version is None else str(version)


def load_calibration(path: Path) -> Any:
    import numpy as np

    data = np.load(str(path), allow_pickle=False)
    if data.ndim != 4:
        raise ValueError("calibration must be (N, H, W, 3), got %r" % (data.shape,))
    if data.shape[-1] != 3:
        if data.shape[1] == 3:
            data = np.transpose(data, (0, 2, 3, 1))
        else:
            raise ValueError("calibration must be NHWC or NCHW with 3 channels")
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    if data.shape[0] == 0:
        raise ValueError("calibration set is empty")
    return data


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="DFC 3.33 translate -> optimize -> compile a YOLOv8 ONNX for Hailo-8L."
    )
    parser.add_argument("--onnx", required=True, help="input ONNX path")
    parser.add_argument("--calib", required=True, help="calib.npy from prepare_calib.py")
    parser.add_argument("--out", required=True, help="output .hef path")
    parser.add_argument("--name", default="yolov8n_drone", help="network name")
    parser.add_argument("--hw-arch", default="hailo8l", choices=("hailo8l", "hailo8"))
    parser.add_argument("--alls", default=str(DEFAULT_ALLS),
                        help="model script (.alls); default normalises uint8 RGB")
    parser.add_argument("--input-name", default="", help="ONNX input node (optional)")
    parser.add_argument("--end-nodes", nargs="*", default=None,
                        help="ONNX end node names (optional; default graph outputs)")
    parser.add_argument("--dry-run", action="store_true", help="validate inputs only")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args = parser.parse_args(argv)

    onnx_path = Path(args.onnx).expanduser()
    calib_path = Path(args.calib).expanduser()
    out_path = Path(args.out).expanduser()
    if not onnx_path.is_file():
        print("error: ONNX not found: %s" % onnx_path, file=sys.stderr)
        return 2
    if not calib_path.is_file():
        print("error: calibration npy not found: %s" % calib_path, file=sys.stderr)
        return 2
    alls_path = Path(args.alls).expanduser() if args.alls else None
    if alls_path is not None and not alls_path.is_file():
        print("error: model script not found: %s" % alls_path, file=sys.stderr)
        return 2

    calib = load_calibration(calib_path)
    height, width = int(calib.shape[1]), int(calib.shape[2])
    print("calibration: %r uint8 (%d frames, %dx%d)" % (calib.shape, calib.shape[0], width, height))

    version = dfc_version()
    if version is None:
        print("BLOCKED: hailo_sdk_client (DFC %s) is not importable." % EXPECTED_DFC, file=sys.stderr)
        print("Install the official Hailo DFC %s wheel, then rerun this script." % EXPECTED_DFC, file=sys.stderr)
        return 3
    print("DFC: %s" % version)
    if not str(version).startswith("3.33") and not args.allow_version_mismatch:
        print("BLOCKED: DFC %s found, this branch pins %s (use --allow-version-mismatch to override)."
              % (version, EXPECTED_DFC), file=sys.stderr)
        return 3
    if args.dry_run:
        print("dry-run: ONNX=%s alls=%s out=%s hw=%s" % (onnx_path, alls_path, out_path, args.hw_arch))
        return 0

    from hailo_sdk_client import ClientRunner

    out_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    runner = ClientRunner(hw_arch=args.hw_arch)
    translate_kwargs: dict[str, Any] = {}
    if args.end_nodes:
        translate_kwargs["end_node_names"] = list(args.end_nodes)
    if args.input_name:
        translate_kwargs["start_node_names"] = [args.input_name]
        translate_kwargs["net_input_shapes"] = {
            args.input_name: [1, 3, height, width]
        }
    print("translate ...")
    runner.translate_onnx_model(str(onnx_path), args.name, **translate_kwargs)
    if alls_path is not None:
        print("load model script: %s" % alls_path)
        runner.load_model_script(alls_path.read_text(encoding="utf-8"))
    print("optimize (%d calibration frames) ..." % calib.shape[0])
    runner.optimize(calib)
    print("compile ...")
    hef = runner.compile()
    if isinstance(hef, (bytes, bytearray)):
        out_path.write_bytes(hef)
    else:
        save_hef = getattr(runner, "save_hef", None)
        if save_hef is None:
            raise RuntimeError("DFC compile() did not return bytes and has no save_hef()")
        save_hef(str(out_path))
    elapsed = time.time() - started
    size = out_path.stat().st_size
    print("wrote %s (%.1f MB) in %.1f s" % (out_path, size / 1e6, elapsed))
    print("next: tools/hailo/bench_hef.py --hef %s" % out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
