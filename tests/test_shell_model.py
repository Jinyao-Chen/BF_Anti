"""Regression tests for the visible nose canopy and 40-degree camera."""
from __future__ import annotations

import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "sim/models/bf_anti_shell/model.sdf"
CANOPY = ROOT / "sim/models/bf_anti_shell/meshes/head_canopy.stl"


def _stl_tris(path: Path) -> np.ndarray:
    with open(path, "rb") as handle:
        handle.read(80)
        count = struct.unpack("<I", handle.read(4))[0]
        data = np.frombuffer(handle.read(count * 50), dtype=np.uint8)
    return data.reshape(count, 50)[:, 12:48].copy().view("<f4").reshape(count, 3, 3)


def test_camera_is_forty_degrees_up_and_uses_canopy():
    root = ET.parse(MODEL).getroot()
    cam = None
    canopy = None
    for sensor in root.iter("sensor"):
        if sensor.get("name") == "camera_sensor":
            cam = sensor
    for visual in root.iter("visual"):
        if visual.get("name") == "head_canopy_visual":
            canopy = visual
    assert cam is not None and canopy is not None
    pose = [float(v) for v in cam.findtext("pose").split()]
    assert pose[:3] == [-0.022, -0.008, 0.206]
    assert pose[4] == math.radians(-40.0) or abs(pose[4] + math.radians(40.0)) < 1e-5
    uri = canopy.find("geometry/mesh/uri").text
    assert uri.endswith("head_canopy.stl")


def test_canopy_is_proud_of_the_body_and_aperture_is_clear():
    tris = _stl_tris(CANOPY)
    assert len(tris) > 3000
    verts = tris.reshape(-1, 3)
    # 1.38x inflation gives a noticeably wider fairing than the CAD head cap
    assert verts[:, 1].max() - verts[:, 1].min() > 0.06
    assert verts[:, 2].max() > 0.240
    # camera optical axis 40 deg; no canopy surface inside 8 mm of the lens axis
    axis = np.array([math.cos(math.radians(40.0)), 0.0,
                     math.sin(math.radians(40.0))])
    lens = np.array([-0.022, -0.008, 0.206]) + axis * 0.012
    rel = verts - lens
    axial = rel @ axis
    radial = np.linalg.norm(rel - axial[:, None] * axis, axis=1)
    front = (axial > -0.01) & (axial < 0.04)
    assert radial[front].min() > 0.008
