"""Calibration builder tests: priority, determinism, real-frame guardrail."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")


def load_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "hailo" / "prepare_calib.py"
    spec = importlib.util.spec_from_file_location("prepare_calib_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_images(directory: Path, count: int, tint: int = 0) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        image = np.full((48, 64, 3), (10 + tint) % 256, dtype=np.uint8)
        image[0:8, 0:8] = (index * 20 + tint) % 256
        image[8:16, 8:16] = (255 - index * 10) % 256
        assert cv2.imwrite(str(directory / ("frame_%03d.jpg" % index)), image)
    return directory


def run_main(module, *args):
    return module.main(list(args))


def test_priority_and_manifest(tmp_path):
    module = load_module()
    pi = make_images(tmp_path / "pi", 2, 0)
    drone = make_images(tmp_path / "drone", 2, 20)
    gazebo = make_images(tmp_path / "gazebo", 2, 40)
    neg = make_images(tmp_path / "neg", 2, 60)
    out = tmp_path / "calib"
    code = run_main(
        module,
        "--pi-frames", str(pi),
        "--drone-images", str(drone),
        "--gazebo-frames", str(gazebo),
        "--negatives", str(neg),
        "--out", str(out),
        "--count", "4", "--size", "64", "--seed", "7",
        "--min-real", "1", "--min-real-fraction", "0.0",
    )
    assert code == 0
    array = np.load(out / "calib.npy")
    assert array.shape == (4, 64, 64, 3)
    assert array.dtype == np.uint8
    assert array[0, 0, 0].tolist() == [114, 114, 114]
    assert len(list((out / "images").glob("*.jpg"))) == 4
    manifest = json.loads((out / "calib_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["real_pi_camera"] == 2
    assert manifest["counts"]["real_drone_photos"] == 1
    assert manifest["counts"]["sim_gazebo"] == 1
    assert manifest["real_target_frames"] == 3
    assert manifest["policy"]["passed"] is True
    assert manifest["color_order"] == "RGB"
    assert len(manifest["images"]) == 4
    assert manifest["images"][0]["sha256"]


def test_sim_only_is_blocked_without_flag(tmp_path):
    module = load_module()
    gazebo = make_images(tmp_path / "gazebo", 2)
    out = tmp_path / "calib"
    code = run_main(
        module,
        "--gazebo-frames", str(gazebo),
        "--out", str(out), "--count", "2", "--size", "64",
        "--min-real", "1", "--min-real-fraction", "0.0",
    )
    assert code == 3
    assert not (out / "calib.npy").exists()
    assert not (out / "calib_manifest.json").exists()


def test_allow_sim_only_records_the_flag(tmp_path):
    module = load_module()
    gazebo = make_images(tmp_path / "gazebo", 2)
    out = tmp_path / "calib"
    code = run_main(
        module,
        "--gazebo-frames", str(gazebo),
        "--out", str(out), "--count", "2", "--size", "64",
        "--min-real", "1", "--min-real-fraction", "0.0",
        "--allow-sim-only",
    )
    assert code == 0
    manifest = json.loads((out / "calib_manifest.json").read_text(encoding="utf-8"))
    assert manifest["policy"]["allow_sim_only"] is True
    assert manifest["real_target_frames"] == 0
    assert any("Pi camera" in message for message in manifest["warnings"])


def test_low_real_fraction_is_blocked(tmp_path):
    module = load_module()
    pi = make_images(tmp_path / "pi", 1)
    gazebo = make_images(tmp_path / "gazebo", 7)
    out = tmp_path / "calib"
    code = run_main(
        module,
        "--pi-frames", str(pi),
        "--gazebo-frames", str(gazebo),
        "--out", str(out), "--count", "4", "--size", "64",
        "--min-real", "1", "--min-real-fraction", "0.5",
    )
    assert code == 3


def test_allow_overrides_low_real_fraction(tmp_path):
    module = load_module()
    pi = make_images(tmp_path / "pi", 1)
    gazebo = make_images(tmp_path / "gazebo", 7)
    out = tmp_path / "calib"
    code = run_main(
        module,
        "--pi-frames", str(pi),
        "--gazebo-frames", str(gazebo),
        "--out", str(out), "--count", "4", "--size", "64",
        "--min-real", "1", "--min-real-fraction", "0.5",
        "--allow-sim-only",
    )
    assert code == 0


def test_deterministic_selection(tmp_path):
    module = load_module()
    pi = make_images(tmp_path / "pi", 2)
    drone = make_images(tmp_path / "drone", 2)
    gazebo = make_images(tmp_path / "gazebo", 2)
    neg = make_images(tmp_path / "neg", 2)
    common = [
        "--pi-frames", str(pi),
        "--drone-images", str(drone),
        "--gazebo-frames", str(gazebo),
        "--negatives", str(neg),
        "--count", "4", "--size", "64", "--seed", "11",
        "--min-real", "1", "--min-real-fraction", "0.0",
    ]
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    assert run_main(module, *common, "--out", str(out_a)) == 0
    assert run_main(module, *common, "--out", str(out_b)) == 0
    assert np.array_equal(np.load(out_a / "calib.npy"), np.load(out_b / "calib.npy"))
    manifest_a = json.loads((out_a / "calib_manifest.json").read_text(encoding="utf-8"))
    manifest_b = json.loads((out_b / "calib_manifest.json").read_text(encoding="utf-8"))
    assert manifest_a["images"] == manifest_b["images"]


def test_missing_inputs_and_dir(tmp_path):
    module = load_module()
    assert run_main(module, "--out", str(tmp_path / "x")) == 2
    assert run_main(
        module,
        "--pi-frames", str(tmp_path / "does-not-exist"),
        "--out", str(tmp_path / "x"),
    ) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_main(
        module, "--pi-frames", str(empty), "--out", str(tmp_path / "x")
    ) == 2


def test_dry_run_does_not_write(tmp_path):
    module = load_module()
    pi = make_images(tmp_path / "pi", 2)
    out = tmp_path / "calib"
    code = run_main(
        module,
        "--pi-frames", str(pi),
        "--out", str(out), "--count", "2", "--size", "64",
        "--min-real", "1", "--min-real-fraction", "0.0", "--dry-run",
    )
    assert code == 0
    assert not out.exists()
