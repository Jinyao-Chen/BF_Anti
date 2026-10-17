"""Replay detector tests: formats, playback, loop, health, errors."""
import json

import numpy as np
import pytest

from onboard.config import DetectorConfig
from onboard.detector.replay import ReplayDetector, load_replay


def cfg(path, loop=True):
    return DetectorConfig(backend="replay", replay_path=str(path), replay_loop=loop)


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    return str(path)


FRAMES = [
    {
        "t": 0.0,
        "detections": [
            {"bbox": [10.0, 10.0, 50.0, 50.0], "conf": 0.5, "class_id": 0},
            {"bbox": [100.0, 100.0, 140.0, 140.0], "conf": 0.9, "class_id": 0},
        ],
    },
    {
        "t": 0.033,
        "detections": [
            {"bbox": [12.0, 10.0, 52.0, 50.0], "conf": 0.6, "class_id": 0}
        ],
    },
    {"t": 0.066, "detections": []},
]


def test_jsonl_playback_and_best_detection(tmp_path):
    path = write_jsonl(tmp_path / "flight.jsonl", FRAMES)
    det = ReplayDetector(cfg(path), realtime=False)
    assert det.frame_count == 3
    det.start()
    first = det.latest(1.0, now=100.0)
    assert first is not None and first.conf == pytest.approx(0.9)
    assert first.bbox == (100.0, 100.0, 140.0, 140.0)
    second = det.latest(1.0, now=100.01)
    assert second is not None and second.conf == pytest.approx(0.6)
    third = det.latest(1.0, now=100.02)
    assert third is None
    fourth = det.latest(1.0, now=100.03)
    assert fourth is not None and fourth.conf == pytest.approx(0.9)


def test_jsonl_latest_detections_returns_all(tmp_path):
    path = write_jsonl(tmp_path / "two.jsonl", [FRAMES[0]])
    det = ReplayDetector(cfg(path), realtime=False)
    det.start()
    dets = det.latest_detections(1.0, now=5.0)
    assert len(dets) == 2
    assert all(d.t == pytest.approx(5.0) for d in dets)


def test_replay_loop_disabled_holds_last_frame(tmp_path):
    path = write_jsonl(tmp_path / "flight.jsonl", FRAMES)
    det = ReplayDetector(cfg(path, loop=False), realtime=False)
    det.start()
    assert det.latest(1.0, now=1.0) is not None
    assert det.latest(1.0, now=1.0) is not None
    assert det.latest(1.0, now=1.0) is None
    assert det.finished is True
    assert det.latest(1.0, now=1.0) is None


def test_file_time_can_go_stale(tmp_path):
    path = write_jsonl(tmp_path / "flight.jsonl", FRAMES)
    det = ReplayDetector(cfg(path), realtime=False, use_file_time=True, clock=lambda: 1000.0)
    det.start()
    assert det.latest(0.001, now=1000.2) is None


def test_healthy_lifecycle(tmp_path):
    path = write_jsonl(tmp_path / "flight.jsonl", FRAMES)
    det = ReplayDetector(cfg(path), realtime=False)
    assert det.healthy(0.0) is False
    det.start()
    assert det.healthy(0.0) is True
    det.stop()
    assert det.healthy(0.0) is False
    assert det.stats().backend == "replay"


def test_bare_detection_line_and_time_sort(tmp_path):
    path = (tmp_path / "bare.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": 1.0, "bbox": [1, 2, 3, 4], "conf": 0.4}) + "\n")
        fh.write(json.dumps({"t": 0.0, "bbox": [5, 6, 7, 8], "conf": 0.7}) + "\n")
    frames = load_replay(str(path))
    assert [frame.t for frame in frames] == [0.0, 1.0]
    det = ReplayDetector(cfg(path), realtime=False)
    det.start()
    assert det.latest(1.0, now=0.0).bbox == (5.0, 6.0, 7.0, 8.0)


def test_json_array_file(tmp_path):
    path = tmp_path / "frames.json"
    path.write_text(json.dumps({"frames": FRAMES[:2]}), encoding="utf-8")
    det = ReplayDetector(cfg(path), realtime=False)
    assert det.frame_count == 2


def test_csv_with_header_groups_rows(tmp_path):
    path = tmp_path / "frames.csv"
    path.write_text(
        "t,x1,y1,x2,y2,conf,class_id\n"
        "0.0,10,10,50,50,0.5,0\n"
        "0.0,100,100,140,140,0.9,1\n"
        "0.1,12,10,52,50,0.6,0\n",
        encoding="utf-8",
    )
    frames = load_replay(str(path))
    assert len(frames) == 2
    assert len(frames[0].detections) == 2
    assert frames[0].detections[1].class_id == 1
    det = ReplayDetector(cfg(path), realtime=False)
    det.start()
    assert det.latest(1.0, now=0.0).conf == pytest.approx(0.9)


def test_csv_headerless_seven_columns(tmp_path):
    path = tmp_path / "frames.csv"
    path.write_text("0.0,10,10,50,50,0.5,0\n0.1,12,10,52,50,0.6,0\n", encoding="utf-8")
    frames = load_replay(str(path))
    assert len(frames) == 2
    assert frames[0].detections[0].bbox == (10.0, 10.0, 50.0, 50.0)


def test_npz_frames(tmp_path):
    path = tmp_path / "frames.npz"
    np.savez(
        path,
        t=np.array([0.0, 0.1, 0.1]),
        boxes=np.array(
            [[10, 10, 50, 50], [100, 100, 140, 140], [12, 10, 52, 50]], dtype=np.float32
        ),
        conf=np.array([0.5, 0.9, 0.6], dtype=np.float32),
        class_id=np.array([0, 0, 1]),
        frame=np.array([0, 1, 1]),
    )
    frames = load_replay(str(path))
    assert len(frames) == 2
    assert len(frames[1].detections) == 2
    det = ReplayDetector(cfg(path), realtime=False)
    det.start()
    assert det.latest(1.0, now=0.0).conf == pytest.approx(0.5)


@pytest.mark.parametrize("suffix", [".txt", ".yaml"])
def test_unsupported_suffix(tmp_path, suffix):
    path = tmp_path / ("x" + suffix)
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_replay(str(path))


def test_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_replay(str(tmp_path / "missing.jsonl"))
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_replay(str(empty))
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"detections": [{"bbox": [1, 2]}]}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_replay(str(bad))
    with pytest.raises(ValueError):
        ReplayDetector(DetectorConfig(backend="replay", replay_path=""))
