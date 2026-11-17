#!/usr/bin/env python3
"""Build the visible nose canopy for bf_anti_shell.

The CAD-derived head_shell.stl is almost exactly coincident with the body mesh,
so adding it to the SDF changes no silhouette. This tool takes that surface,
inflates it away from the body, cuts a cylindrical aperture along the camera
optical axis, and adds a raised lens ring. The result is visual-only.
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

# camera sensor origin in the light_quad/body frame (see model.sdf)
CAMERA_POSE = np.array([-0.022, -0.008, 0.206], dtype=float)
# head-shell surface centroid / inflation pivot; kept close to the CAD cap
PIVOT = np.array([-0.033, 0.0, 0.216], dtype=float)


def read_stl(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        header = handle.read(84)
        if len(header) < 84:
            raise ValueError(f"not a binary STL: {path}")
        count = struct.unpack("<I", header[80:84])[0]
        data = np.frombuffer(handle.read(count * 50), dtype=np.uint8)
    if data.size != count * 50:
        raise ValueError(f"truncated STL: {path}")
    tri = data.reshape(count, 50)[:, 12:48].copy().view("<f4").reshape(count, 3, 3)
    return np.asarray(tri, dtype=np.float64)


def write_stl(path: Path, tri: np.ndarray) -> None:
    tri = np.asarray(tri, dtype=np.float32)
    header = b"BF_Anti head canopy (visual only)"[:80].ljust(80, b" ")
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(struct.pack("<I", len(tri)))
        for t in tri:
            n = np.cross(t[1] - t[0], t[2] - t[0])
            n /= max(float(np.linalg.norm(n)), 1e-12)
            handle.write(struct.pack("<3f", *n))
            handle.write(struct.pack("<9f", *t.reshape(-1)))
            handle.write(b"\x00\x00")


def optical_axis(tilt_deg: float) -> np.ndarray:
    t = np.deg2rad(float(tilt_deg))
    return np.array([np.cos(t), 0.0, np.sin(t)], dtype=float)


def remove_aperture(tri: np.ndarray, center: np.ndarray, axis: np.ndarray,
                    radius: float) -> np.ndarray:
    """Drop triangles that intersect the camera aperture cylinder."""
    v = tri.reshape(-1, 3)
    rel = v - center
    axial = rel @ axis
    radial = np.linalg.norm(rel - axial[:, None] * axis, axis=1)
    inside = ((radial < radius) & (axial > -0.020) & (axial < 0.080)).reshape(-1, 3)
    all_in = inside.all(axis=1)
    any_in = inside.any(axis=1)
    c = tri.mean(axis=1)
    crel = c - center
    cax = crel @ axis
    crad = np.linalg.norm(crel - np.outer(cax, axis), axis=1)
    centroid_in = (crad < radius * 0.98) & (cax > -0.015) & (cax < 0.080)
    keep = ~(all_in | centroid_in | any_in)
    return tri[keep]


def torus(center: np.ndarray, axis: np.ndarray, major_r: float, tube_r: float,
          n_major: int = 72, n_minor: int = 14) -> np.ndarray:
    """Generate a torus whose symmetry axis is axis."""
    axis = axis / np.linalg.norm(axis)
    ref = np.array([0.0, 1.0, 0.0])
    if abs(float(ref @ axis)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    u = ref - axis * float(ref @ axis)
    u /= np.linalg.norm(u)
    w = np.cross(axis, u)
    verts = np.empty((n_major, n_minor, 3), dtype=float)
    for i, th in enumerate(np.linspace(0.0, 2.0 * np.pi, n_major, endpoint=False)):
        radial = np.cos(th) * u + np.sin(th) * w
        c = center + major_r * radial
        for j, ph in enumerate(np.linspace(0.0, 2.0 * np.pi, n_minor, endpoint=False)):
            verts[i, j] = c + tube_r * (np.cos(ph) * radial + np.sin(ph) * axis)
    tris = []
    for i in range(n_major):
        ni = (i + 1) % n_major
        for j in range(n_minor):
            nj = (j + 1) % n_minor
            a, b, c, d = verts[i, j], verts[ni, j], verts[ni, nj], verts[i, nj]
            tris.append([a, b, c])
            tris.append([a, c, d])
    return np.asarray(tris, dtype=float)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="sim/models/bf_anti_shell/meshes/bf_anti_shell.stl")
    ap.add_argument("--source", default="sim/models/bf_anti_shell/meshes/head_shell.stl")
    ap.add_argument("--out", default="sim/models/bf_anti_shell/meshes/head_canopy.stl")
    ap.add_argument("--tilt-deg", type=float, default=40.0)
    ap.add_argument("--scale", type=float, nargs=3, default=(1.38, 1.38, 1.12))
    ap.add_argument("--aperture-radius-mm", type=float, default=11.0)
    ap.add_argument("--ring-major-mm", type=float, default=12.6)
    ap.add_argument("--ring-tube-mm", type=float, default=1.5)
    args = ap.parse_args()

    body = read_stl(Path(args.body))
    src = read_stl(Path(args.source))
    if len(src) > len(body) * 0.2:
        raise SystemExit("source mesh looks like the full body; rejected")
    scale = np.asarray(args.scale, dtype=float)
    tri = (src - PIVOT) * scale + PIVOT
    axis = optical_axis(args.tilt_deg)
    lens = CAMERA_POSE + axis * 0.012
    tri = remove_aperture(tri, lens, axis, args.aperture_radius_mm * 1e-3)
    ring = torus(lens + axis * 0.002, axis,
                 args.ring_major_mm * 1e-3, args.ring_tube_mm * 1e-3)
    out = np.concatenate([tri, ring], axis=0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_stl(Path(args.out), out)
    v = out.reshape(-1, 3)
    print(f"body tris={len(body)} source tris={len(src)} canopy tris={len(out)}")
    print(f"bbox min={v.min(0)} max={v.max(0)}")
    print(f"lens={lens} axis={axis} aperture={args.aperture_radius_mm}mm")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
