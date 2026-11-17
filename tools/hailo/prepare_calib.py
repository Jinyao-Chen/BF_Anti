#!/usr/bin/env python3
"""Build a deterministic Hailo calibration set with provenance.

Priority order (highest first):
  1. real Pi camera frames      --pi-frames DIR
  2. real drone images          --drone-images DIR
  3. Gazebo / simulation frames --gazebo-frames DIR
  4. real-world negatives       --negatives DIR  (e.g. VisDrone)

The tool refuses to silently produce a calibration set that is not anchored
in real target-domain imagery. It exits with status 3 when too few real frames
are selected unless --allow-sim-only is given explicitly; the flag and the
provenance counts are recorded in calib_manifest.json.

Outputs in --out:
  calib.npy              (N, size, size, 3) uint8 RGB, letterboxed with pad 114
  calib_manifest.json    selection, hashes, policy decision, warnings
  images/NNNNNN.jpg      the same frames as JPEG (for hailomz --calib-path);
                         disable with --no-emit-images

The npy is consumed by tools/hailo/compile_onnx.py (DFC optimize); the image
directory is consumed by hailomz compile --calib-path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
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
for _candidate in (REPO_ROOT, Path.cwd()):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
PAD_VALUE = 114

# (manifest key, label, nominal share, argument attribute)
SOURCES = (
    ("pi_frames", "real_pi_camera", 0.50, "pi_frames"),
    ("drone_images", "real_drone_photos", 0.25, "drone_images"),
    ("gazebo_frames", "sim_gazebo", 0.15, "gazebo_frames"),
    ("negatives", "real_world_negative", 0.10, "negatives"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_images(dirs: list[str], extensions: tuple[str, ...]) -> list[Path]:
    found: list[Path] = []
    for raw in dirs or []:
        directory = Path(raw).expanduser()
        if not directory.is_dir():
            raise FileNotFoundError("not a directory: %s" % directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                found.append(path)
    # Stable, de-duplicated candidate order.
    return sorted(set(found))


def allocate(
    count: int,
    available: dict[str, list[Path]],
    shares: dict[str, float],
    seed: int,
) -> dict[str, list[Path]]:
    """Deterministic allocation in priority order, then fill from leftovers."""
    rng = random.Random(int(seed))
    chosen: dict[str, list[Path]] = {}
    used: set[Path] = set()
    for key, _label, _share, _attr in SOURCES:
        files = available.get(key, [])
        target = int(round(count * shares[key]))
        take = min(max(target, 0), len(files))
        picked = rng.sample(files, take) if take else []
        picked = sorted(picked)
        chosen[key] = picked
        used.update(picked)
    remaining = int(count) - sum(len(v) for v in chosen.values())
    for key, _label, _share, _attr in SOURCES:
        if remaining <= 0:
            break
        leftovers = [path for path in available.get(key, []) if path not in used]
        take = min(remaining, len(leftovers))
        if take <= 0:
            continue
        picked = sorted(rng.sample(leftovers, take))
        chosen[key].extend(picked)
        used.update(picked)
        remaining -= take
    return chosen


def policy_check(
    counts: dict[str, int],
    requested: int,
    min_real: int,
    min_real_fraction: float,
    allow_sim_only: bool,
) -> list[str]:
    """Return policy errors. Empty list means the set may be written."""
    errors: list[str] = []
    real = int(counts.get("real_pi_camera", 0)) + int(counts.get("real_drone_photos", 0))
    total = sum(int(v) for v in counts.values())
    fraction = (real / total) if total else 0.0
    if allow_sim_only:
        return errors
    if real == 0:
        errors.append(
            "no real target-domain frames selected (Pi camera or drone images); "
            "this would be an all-simulation/background calibration set"
        )
    elif real < int(min_real):
        errors.append(
            "only %d real target-domain frame(s) selected; --min-real is %d"
            % (real, int(min_real))
        )
    elif total and fraction < float(min_real_fraction):
        errors.append(
            "real target-domain fraction %.1f%% is below --min-real-fraction %.1f%%"
            % (100.0 * fraction, 100.0 * float(min_real_fraction))
        )
    return errors


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, provenance-tracked Hailo calibration set."
    )
    parser.add_argument("--pi-frames", action="append", default=[], metavar="DIR",
                        help="real Raspberry Pi camera frames (highest priority)")
    parser.add_argument("--drone-images", action="append", default=[], metavar="DIR",
                        help="real drone images (labels are not required)")
    parser.add_argument("--gazebo-frames", action="append", default=[], metavar="DIR",
                        help="Gazebo / simulation frames")
    parser.add_argument("--negatives", action="append", default=[], metavar="DIR",
                        help="real-world negative frames, e.g. VisDrone")
    parser.add_argument("--out", default="build/calib", help="output directory")
    parser.add_argument("--count", type=int, default=256, help="number of frames")
    parser.add_argument("--size", type=int, default=640, help="square model input size")
    parser.add_argument("--seed", type=int, default=0, help="deterministic sampling seed")
    parser.add_argument("--min-real", type=int, default=8,
                        help="minimum real target-domain frames (default 8)")
    parser.add_argument("--min-real-fraction", type=float, default=0.25,
                        help="minimum real target-domain fraction (default 0.25)")
    parser.add_argument("--allow-sim-only", action="store_true",
                        help="allow a set that fails the real-frame policy (recorded)")
    parser.add_argument("--no-emit-images", action="store_true",
                        help="do not write the hailomz image directory")
    parser.add_argument("--dry-run", action="store_true", help="print the plan only")
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS),
                        help="comma-separated image extensions")
    args = parser.parse_args(argv)

    if args.count <= 0:
        print("error: --count must be positive", file=sys.stderr)
        return 2
    if args.size < 32 or args.size % 32 != 0:
        print("error: --size must be a multiple of 32 and >= 32", file=sys.stderr)
        return 2
    extensions = tuple(
        ext if ext.startswith(".") else "." + ext
        for ext in (part.strip().lower() for part in args.extensions.split(","))
        if ext
    )

    shares = {key: share for key, _label, share, _attr in SOURCES}
    available: dict[str, list[Path]] = {}
    try:
        for key, _label, _share, attr in SOURCES:
            available[key] = list_images(getattr(args, attr), extensions)
    except FileNotFoundError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    total_available = sum(len(v) for v in available.values())
    if total_available == 0:
        print("error: no images found in any --*-frames/--*-images/--negatives dir",
              file=sys.stderr)
        return 2

    chosen = allocate(args.count, available, shares, args.seed)
    counts = {label: len(chosen[key]) for key, label, _share, _attr in SOURCES}
    warnings: list[str] = []
    if counts["real_pi_camera"] == 0:
        warnings.append(
            "no real Pi camera frames selected (camera gate is blocked); "
            "calibration is anchored in other real/sim sources"
        )
    if sum(len(v) for v in chosen.values()) < args.count:
        warnings.append(
            "only %d of the requested %d frames are available"
            % (sum(len(v) for v in chosen.values()), args.count)
        )

    errors = policy_check(
        counts, args.count, args.min_real, args.min_real_fraction, args.allow_sim_only
    )
    if args.dry_run:
        plan = {
            "requested": args.count,
            "selected": counts,
            "sources": {
                key: {
                    "available": len(available[key]),
                    "selected": len(chosen[key]),
                    "share": shares[key],
                }
                for key, _label, _share, _attr in SOURCES
            },
            "policy_errors": errors,
            "warnings": warnings,
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0 if not errors else 3
    if errors:
        for message in errors:
            print("BLOCKED: " + message, file=sys.stderr)
        print(
            "Refusing to write a calibration set that fails the real-frame policy. "
            "Collect Pi camera frames or pass --allow-sim-only deliberately.",
            file=sys.stderr,
        )
        return 3

    import cv2  # lazy: listing/dry-run does not need OpenCV

    from onboard.detector.postprocess import letterbox_image

    out_dir = Path(args.out).expanduser()
    images_dir = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_emit_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    frames: list[Any] = []
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    written_counts = {label: 0 for _key, label, _share, _attr in SOURCES}
    for key, label, _share, _attr in SOURCES:
        for path in chosen[key]:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                skipped.append({"path": str(path), "reason": "unreadable"})
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            letterboxed, _params = letterbox_image(
                rgb, args.size, args.size, pad_value=PAD_VALUE
            )
            index = len(frames)
            frames.append(letterboxed)
            record = {
                "index": index,
                "source": label,
                "path": str(path),
                "sha256": sha256_file(path),
            }
            records.append(record)
            written_counts[label] += 1
            if not args.no_emit_images:
                bgr = cv2.cvtColor(letterboxed, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(images_dir / ("%06d.jpg" % index)), bgr)

    if not frames:
        print("error: no readable images; nothing written", file=sys.stderr)
        return 2

    # Re-check the policy on what was actually decoded.
    written_errors = policy_check(
        written_counts, len(frames), args.min_real, args.min_real_fraction,
        args.allow_sim_only,
    )
    if written_errors:
        for message in written_errors:
            print("BLOCKED: " + message, file=sys.stderr)
        return 3

    import numpy as np

    array = np.stack(frames).astype(np.uint8, copy=False)
    tmp_npy = out_dir / "calib.npy.tmp"
    # numpy.save appends .npy to a filename without that suffix, so write via
    # a file handle and then atomically rename.
    with open(tmp_npy, "wb") as fh:
        np.save(fh, array)
    tmp_npy.replace(out_dir / "calib.npy")

    real = written_counts["real_pi_camera"] + written_counts["real_drone_photos"]
    manifest = {
        "tool": "tools/hailo/prepare_calib.py",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "requested": args.count,
        "written": int(array.shape[0]),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": args.size,
        "pad_value": PAD_VALUE,
        "color_order": "RGB",
        "sources": {
            key: {
                "dirs": list(getattr(args, _attr)),
                "available": len(available[key]),
                "selected": len(chosen[key]),
                "written": written_counts[label],
                "share": shares[key],
            }
            for key, label, _share, _attr in SOURCES
        },
        "counts": written_counts,
        "real_target_frames": real,
        "simulation_or_negative_frames": int(array.shape[0]) - real,
        "real_fraction": (real / float(array.shape[0])) if array.shape[0] else 0.0,
        "policy": {
            "min_real": args.min_real,
            "min_real_fraction": args.min_real_fraction,
            "allow_sim_only": bool(args.allow_sim_only),
            "passed": True,
        },
        "warnings": warnings,
        "skipped": skipped,
        "images": records,
    }
    manifest_path = out_dir / "calib_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("wrote %s %s" % (out_dir / "calib.npy", tuple(array.shape)))
    if not args.no_emit_images:
        print("wrote %s (%d jpg)" % (images_dir, len(records)))
    print("wrote %s" % manifest_path)
    print(
        "sources: pi=%d drone=%d gazebo=%d negatives=%d real=%d (%.0f%%)"
        % (
            written_counts["real_pi_camera"],
            written_counts["real_drone_photos"],
            written_counts["sim_gazebo"],
            written_counts["real_world_negative"],
            real,
            100.0 * manifest["real_fraction"],
        )
    )
    for message in warnings:
        print("WARNING: " + message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
