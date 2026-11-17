#!/usr/bin/env python3
"""Report the Hailo toolchain available on this host.

Roles:
  pi5    runtime host: HailoRT python + firmware + /dev/hailo0
  build  HEF build host: DFC 3.33.x + Hailo Model Zoo v2.17 (GPU is NOT needed)
  any    informational only (default)

Exit status is 0 when the selected role's requirements are met, 1 otherwise.
With --json the machine-readable object goes to stdout and the human summary
goes to stderr, so scripts can pipe it into jq.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
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
EXPECTED_HMZ = "2.17"
ALLOWED_GPU_UUID = "GPU-40c96051-47f2-ac23-7d8d-ef868990967c"
HMZ_CANDIDATES = (
    os.environ.get("HAILO_MODEL_ZOO", ""),
    "/opt/hailo_model_zoo",
    str(Path.home() / "hailo_model_zoo"),
    "/datadisk/hailo_model_zoo",
    "/tmp/hmz",
)


def run(cmd: list[str], timeout: float = 10.0) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None


def module_version(name: str) -> Optional[str]:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as pkg_version

            version = pkg_version(name)
        except Exception:
            version = None
    return None if version is None else str(version)


def probe_hmz() -> dict[str, Any]:
    for raw in HMZ_CANDIDATES:
        if not raw:
            continue
        path = Path(os.path.expanduser(raw))
        cfg_file = path / "hailo_model_zoo" / "cfg" / "networks" / "yolov8n.yaml"
        if not cfg_file.is_file():
            continue
        version = None
        git = run(["git", "-C", str(path), "describe", "--tags", "--always"])
        if git is not None and git.returncode == 0:
            version = git.stdout.strip()
        if version is None:
            version = module_version("hailo_model_zoo")
        return {"path": str(path), "version": version, "has_yolov8n_yaml": True}
    return {"path": None, "version": None, "has_yolov8n_yaml": False}


def probe_device() -> dict[str, Any]:
    info: dict[str, Any] = {
        "device_node": os.path.exists("/dev/hailo0"),
        "hailortcli": shutil.which("hailortcli"),
        "board": None,
        "firmware": None,
        "architecture": None,
        "error": "",
    }
    if not info["hailortcli"]:
        info["error"] = "hailortcli not found"
        return info
    result = run(["hailortcli", "fw-control", "identify"], timeout=15.0)
    if result is None or result.returncode != 0:
        info["error"] = "fw-control identify failed"
        return info
    for line in result.stdout.splitlines():
        text = line.strip()
        for prefix, key in (
            ("Firmware Version:", "firmware"),
            ("Board Name:", "board"),
            ("Device Architecture:", "architecture"),
        ):
            if text.startswith(prefix):
                info[key] = text[len(prefix):].strip()
    return info


def probe_hefs() -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    patterns = [REPO_ROOT / "model", REPO_ROOT / "build"]
    for directory in patterns:
        if directory.is_dir():
            for path in sorted(directory.rglob("*.hef")):
                found.append({"path": str(path), "size": path.stat().st_size})
    stock = Path("/usr/share/hailo-models")
    if stock.is_dir():
        for path in sorted(stock.glob("*.hef")):
            found.append({"path": str(path), "size": path.stat().st_size})
    return found


def probe_thermal() -> Optional[float]:
    path = Path("/sys/class/thermal/thermal_zone0/temp")
    try:
        return float(path.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def probe_gpus() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader",
        ],
        timeout=10.0,
    )
    if result is None or result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": parts[0], "uuid": parts[1], "name": parts[2]})
    return gpus


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check HailoRT / DFC / HMZ / device availability for BF_Anti."
    )
    parser.add_argument("--role", choices=("pi5", "build", "any"), default="any")
    parser.add_argument("--json", action="store_true", help="emit JSON on stdout")
    args = parser.parse_args(argv)

    hailo_platform = module_version("hailo_platform")
    dfc = module_version("hailo_sdk_client")
    hmz = probe_hmz()
    device = probe_device()
    report: dict[str, Any] = {
        "host": {
            "hostname": platform.node(),
            "os": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "repo": str(REPO_ROOT),
        },
        "packages": {
            "hailo_platform": hailo_platform,
            "hailort_cli": shutil.which("hailortcli"),
            "hailo_sdk_client_dfc": dfc,
            "expected_dfc": EXPECTED_DFC,
            "hailo_model_zoo": hmz,
            "expected_hmz": EXPECTED_HMZ,
        },
        "device": device,
        "thermal_zone0_c": probe_thermal(),
        "gpus": probe_gpus(),
        "allowed_gpu_uuid": ALLOWED_GPU_UUID,
        "hefs": probe_hefs(),
        "blocked": [],
        "warnings": [],
    }

    blocked: list[str] = report["blocked"]
    warnings: list[str] = report["warnings"]
    if args.role == "pi5":
        if not hailo_platform:
            blocked.append("hailo_platform (HailoRT python) is not importable")
        if not device["device_node"]:
            blocked.append("/dev/hailo0 is missing")
        if not device["firmware"]:
            blocked.append("Hailo firmware not readable via hailortcli")
        elif not str(device["firmware"]).startswith("4.23"):
            warnings.append(
                "HailoRT firmware %s; this branch targets 4.23.0" % device["firmware"]
            )
    elif args.role == "build":
        if not dfc:
            blocked.append(
                "Hailo DFC (hailo_sdk_client) %s is not installed; the official "
                "wheel must be provided by the user" % EXPECTED_DFC
            )
        elif not str(dfc).startswith("3.33"):
            blocked.append(
                "Hailo DFC %s found, expected %s" % (dfc, EXPECTED_DFC)
            )
        if not hmz.get("path"):
            blocked.append(
                "Hailo Model Zoo v%s checkout not found (set HAILO_MODEL_ZOO)"
                % EXPECTED_HMZ
            )
        elif hmz.get("version") and EXPECTED_HMZ not in str(hmz["version"]):
            warnings.append(
                "HMZ version %s; this branch pins v%s"
                % (hmz["version"], EXPECTED_HMZ)
            )
    if report["gpus"]:
        warnings.append(
            "GPU present: build/bench must never use nvidia-smi index 0; only %s"
            % ALLOWED_GPU_UUID
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        out = sys.stderr
    else:
        out = sys.stdout
        print("host      : %s (%s, %s)" % (
            report["host"]["hostname"], report["host"]["machine"], report["host"]["python"]
        ), file=out)
        print("HailoRT   : %s (cli %s)" % (hailo_platform, report["packages"]["hailort_cli"]), file=out)
        print("DFC       : %s (expected %s)" % (dfc, EXPECTED_DFC), file=out)
        print("HMZ       : %s (expected v%s)" % (hmz.get("version") or hmz.get("path"), EXPECTED_HMZ), file=out)
        print("device    : %s fw=%s arch=%s node=%s" % (
            device["board"], device["firmware"], device["architecture"], device["device_node"]
        ), file=out)
        print("thermal   : %s C" % report["thermal_zone0_c"], file=out)
        print("HEF files : %d" % len(report["hefs"]), file=out)
        for hef in report["hefs"][:10]:
            print("  %s (%.1f MB)" % (hef["path"], hef["size"] / 1e6), file=out)
        for gpu in report["gpus"]:
            print("gpu%s     : %s %s" % (gpu["index"], gpu["uuid"], gpu["name"]), file=out)
        for message in warnings:
            print("WARNING: " + message, file=out)
        for message in blocked:
            print("BLOCKED: " + message, file=out)
        print("RESULT    : %s (role=%s)" % ("OK" if not blocked else "BLOCKED", args.role), file=out)
    return 0 if not blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
