"""Deterministic replay detector for bench runs and unit tests.

ReplayDetector reads detections (not pixels) from a recorded JSONL/JSON/CSV/NPZ
file and replays them with the same latest()/healthy() contract as the Hailo
backend. It never imports HailoRT, OpenCV, or a camera, so the whole app can be
exercised on a laptop and on the Pi without a detector.

File formats
------------
JSONL / NDJSON: one frame record per line::

    {"t": 0.0, "detections": [{"bbox": [x1,y1,x2,y2], "conf": 0.9,
                               "class_id": 0}]}

A line containing a bare detection (with bbox/conf, without detections) is
accepted as a one-detection frame. JSON files may contain the same frame list
directly or under a "frames" key.

CSV: a header row with x1,y1,x2,y2 and optional t, conf, class_id, frame.
Rows sharing a frame id (or, without ids, consecutive rows with the same t)
form one frame. Without a header, columns are interpreted as
t,x1,y1,x2,y2,conf,class_id (7) or x1,y1,x2,y2,conf,class_id (6); headerless
rows are one frame per row.

NPZ: arrays t (N,), boxes (N,4) or bbox (N,4), conf or scores (N,), optional
class_id (N,), optional frame (N,). Rows are grouped by frame id when present,
otherwise by equal t values.

Boxes are already in source-frame pixels: replay deliberately skips the
letterbox inverse mapping performed by the Hailo backend.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from onboard.detector.base import DetectorStats
from onboard.types import Detection

LOG = logging.getLogger("bf_anti.detector.replay")

__all__ = ["ReplayDetector", "ReplayFrame", "load_replay"]


@dataclass(frozen=True)
class ReplayFrame:
    """One recorded frame: relative time in seconds plus source-pixel boxes."""

    t: float
    detections: tuple[Detection, ...]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if out == out else float(default)


def _make_detection(
    bbox: Any, conf: Any = 1.0, class_id: Any = 0, where: str = ""
) -> Detection:
    try:
        values = [float(v) for v in bbox]
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid bbox at %s: %r" % (where, bbox)) from exc
    if len(values) != 4:
        raise ValueError("bbox must have 4 values at %s: %r" % (where, bbox))
    x1, y1, x2, y2 = values
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return Detection(
        bbox=(x1, y1, x2, y2),
        conf=_as_float(conf, 0.0),
        class_id=int(_as_float(class_id, 0.0)),
    )


def _frame_from_record(record: dict, index: int, where: str) -> ReplayFrame:
    if not isinstance(record, dict):
        raise ValueError("frame record must be an object at %s" % where)
    t = _as_float(record.get("t", record.get("timestamp", 0.0)))
    raw_dets = record.get("detections")
    if raw_dets is None:
        raw_dets = record.get("dets")
    if raw_dets is None and ("bbox" in record or "box" in record):
        raw_dets = [record]
    if raw_dets is None:
        raw_dets = []
    if not isinstance(raw_dets, (list, tuple)):
        raise ValueError("detections must be a list at %s" % where)
    detections = []
    for item in raw_dets:
        if not isinstance(item, dict):
            raise ValueError("detection must be an object at %s" % where)
        bbox = item.get("bbox", item.get("box"))
        detections.append(
            _make_detection(
                bbox,
                item.get("conf", item.get("score", 1.0)),
                item.get("class_id", item.get("cls", 0)),
                where,
            )
        )
    return ReplayFrame(t=t, detections=tuple(detections))


def _load_jsonl(path: str) -> list[ReplayFrame]:
    frames = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d: %s" % (path, line_no, exc)) from exc
            frames.append(_frame_from_record(record, len(frames), "%s:%d" % (path, line_no)))
    return frames


def _load_json(path: str) -> list[ReplayFrame]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("frames", data.get("records", []))
    if not isinstance(data, list):
        raise ValueError("%s: expected a frame list or {frames: [...]}" % path)
    return [
        _frame_from_record(record, idx, "%s[%d]" % (path, idx))
        for idx, record in enumerate(data)
    ]


def _split_csv_frames(rows: list[dict]) -> list[ReplayFrame]:
    grouped: list[list[dict]] = []
    last_key: Any = object()
    for row in rows:
        key = row.get("frame", row.get("frame_id"))
        if key is None:
            key = row.get("t", row.get("timestamp"))
        if grouped and key is not None and key == last_key:
            grouped[-1].append(row)
        else:
            grouped.append([row])
        last_key = key
    frames = []
    for index, group in enumerate(grouped):
        detections = tuple(
            _make_detection(
                [row.get("x1"), row.get("y1"), row.get("x2"), row.get("y2")],
                row.get("conf", row.get("score", 1.0)),
                row.get("class_id", row.get("cls", 0)),
                "csv row",
            )
            for row in group
        )
        t = _as_float(group[0].get("t", group[0].get("timestamp", index)))
        frames.append(ReplayFrame(t=t, detections=detections))
    return frames


def _load_csv(path: str) -> list[ReplayFrame]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        has_header = any(
            token in sample.lower()
            for token in ("x1", "bbox", "conf", "class_id", "frame")
        )
        if has_header:
            rows = [dict(row) for row in csv.DictReader(fh)]
        else:
            rows = []
            for raw in csv.reader(fh):
                values = [cell.strip() for cell in raw if cell.strip() != ""]
                if not values:
                    continue
                if len(values) == 7:
                    keys = ("t", "x1", "y1", "x2", "y2", "conf", "class_id")
                elif len(values) == 6:
                    keys = ("x1", "y1", "x2", "y2", "conf", "class_id")
                else:
                    raise ValueError(
                        "%s: headerless CSV needs 6 or 7 columns, got %d"
                        % (path, len(values))
                    )
                rows.append(dict(zip(keys, values)))
    return _split_csv_frames(rows)


def _load_npz(path: str) -> list[ReplayFrame]:
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        boxes_key = "boxes" if "boxes" in keys else ("bbox" if "bbox" in keys else None)
        if boxes_key is None or "t" not in keys and "timestamps" not in keys:
            if boxes_key is None:
                raise ValueError("%s: missing boxes/bbox array" % path)
        boxes = np.asarray(data[boxes_key], dtype=np.float64)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("%s: boxes must be (N, 4), got %r" % (path, boxes.shape))
        count = int(boxes.shape[0])
        if "t" in keys:
            times = np.asarray(data["t"], dtype=np.float64).reshape(-1)
        elif "timestamps" in keys:
            times = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
        else:
            times = np.arange(count, dtype=np.float64) / 30.0
        if times.size != count:
            raise ValueError("%s: t has %d values, boxes has %d" % (path, times.size, count))
        if "conf" in keys:
            conf = np.asarray(data["conf"], dtype=np.float64).reshape(-1)
        elif "scores" in keys:
            conf = np.asarray(data["scores"], dtype=np.float64).reshape(-1)
        else:
            conf = np.ones(count, dtype=np.float64)
        if conf.size != count:
            raise ValueError("%s: conf has %d values, boxes has %d" % (path, conf.size, count))
        if "class_id" in keys:
            class_ids = np.asarray(data["class_id"]).reshape(-1)
        elif "classes" in keys:
            class_ids = np.asarray(data["classes"]).reshape(-1)
        else:
            class_ids = np.zeros(count, dtype=np.int64)
        if class_ids.size != count:
            raise ValueError(
                "%s: class_id has %d values, boxes has %d" % (path, class_ids.size, count)
            )
        if "frame" in keys:
            frame_ids = np.asarray(data["frame"]).reshape(-1)
        elif "frame_index" in keys:
            frame_ids = np.asarray(data["frame_index"]).reshape(-1)
        else:
            frame_ids = None

    frames: list[ReplayFrame] = []
    if frame_ids is not None:
        if frame_ids.size != count:
            raise ValueError("%s: frame index has %d values, boxes has %d" % (path, frame_ids.size, count))
        for frame_id in np.unique(frame_ids):
            mask = frame_ids == frame_id
            indices = np.flatnonzero(mask)
            detections = tuple(
                _make_detection(boxes[i], conf[i], class_ids[i], path) for i in indices
            )
            frames.append(ReplayFrame(t=float(times[indices[0]]), detections=detections))
    else:
        last_time: Optional[float] = None
        for i in range(count):
            current = float(times[i])
            det = _make_detection(boxes[i], conf[i], class_ids[i], path)
            if frames and last_time is not None and current == last_time:
                frames[-1] = ReplayFrame(t=frames[-1].t, detections=frames[-1].detections + (det,))
            else:
                frames.append(ReplayFrame(t=current, detections=(det,)))
            last_time = current
    return frames


def load_replay(path: str) -> list[ReplayFrame]:
    """Load and validate a replay file. Raises ValueError with the file name."""
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    if not expanded:
        raise ValueError("replay path is empty")
    if not os.path.isfile(expanded):
        raise FileNotFoundError("replay file not found: %s" % expanded)
    suffix = os.path.splitext(expanded)[1].lower()
    if suffix in (".jsonl", ".ndjson"):
        frames = _load_jsonl(expanded)
    elif suffix == ".json":
        frames = _load_json(expanded)
    elif suffix in (".csv", ".tsv"):
        frames = _load_csv(expanded)
    elif suffix == ".npz":
        frames = _load_npz(expanded)
    else:
        raise ValueError(
            "unsupported replay format %r (use .jsonl/.json/.csv/.npz)" % suffix
        )
    if not frames:
        raise ValueError("replay file has no frames: %s" % expanded)
    frames.sort(key=lambda frame: frame.t)
    return frames


class ReplayDetector:
    """Play recorded detections with the Detector interface.

    Parameters
    ----------
    cfg:
        object with replay_path / replay_loop attributes (DetectorConfig).
    path:
        override for cfg.replay_path.
    realtime:
        True (default) advances frames from the wall clock at fps. False is
        step mode: every latest() call advances exactly one frame, which is
        what tests and deterministic benches want.
    loop:
        override for cfg.replay_loop.
    """

    def __init__(
        self,
        cfg: Any,
        path: Optional[str] = None,
        realtime: bool = True,
        loop: Optional[bool] = None,
        fps: float = 30.0,
        use_file_time: bool = False,
        clock=time.monotonic,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.cfg = cfg
        self.path = os.path.expanduser(
            os.path.expandvars(str(path if path is not None else getattr(cfg, "replay_path", "")))
        )
        if not self.path:
            raise ValueError(
                "detector.backend=replay requires detector.replay_path (JSONL/CSV/NPZ)"
            )
        self.frames: list[ReplayFrame] = load_replay(self.path)
        self.realtime = bool(realtime)
        self.loop = bool(getattr(cfg, "replay_loop", True) if loop is None else loop)
        self.fps = float(fps)
        if self.fps <= 0.0:
            raise ValueError("replay fps must be positive")
        self.use_file_time = bool(use_file_time)
        self._clock = clock
        self._log = logger or LOG

        self._lock = threading.Lock()
        self._running = False
        self._index = 0
        self._t0 = 0.0
        self._played = 0
        self._last_det_t = 0.0
        self._last_detection: Optional[Detection] = None
        self._last_frame_detections: Optional[list[Detection]] = None
        self._last_latest_now: Optional[float] = None
        self._finished = False

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            self._running = True
            self._index = 0
            self._played = 0
            self._finished = False
            self._t0 = self._clock()
            self._last_frame_detections = None
            self._last_latest_now = None

    def stop(self) -> None:
        with self._lock:
            self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    # -- playback --------------------------------------------------------
    def _current_index(self, now: float) -> int:
        count = len(self.frames)
        if self.realtime:
            elapsed = max(0.0, float(now) - self._t0)
            index = int(elapsed * self.fps)
            if self.loop:
                index %= count
            elif index >= count:
                index = count - 1
                self._finished = True
            self._index = index
            return index
        index = self._index
        self._index += 1
        if self._index >= count:
            if self.loop:
                self._index = 0
            else:
                self._index = count - 1
                self._finished = True
        return index

    def _stamp(self, index: int, now: float) -> float:
        if self.use_file_time:
            return self._t0 + float(self.frames[index].t)
        if self.realtime:
            return self._t0 + float(index) / self.fps
        return float(now)

    def _stamped(self, index: int, now: float) -> list[Detection]:
        stamp = self._stamp(index, now)
        frame = self.frames[index]
        return [
            Detection(bbox=det.bbox, conf=det.conf, class_id=det.class_id, t=stamp)
            for det in frame.detections
        ]

    def latest(
        self, max_age_s: Optional[float], now: Optional[float] = None
    ) -> Optional[Detection]:
        if not self._running:
            self.start()
        now = self._clock() if now is None else float(now)
        with self._lock:
            index = self._current_index(now)
            detections = self._stamped(index, now)
            self._played += 1
            self._last_frame_detections = list(detections)
            self._last_latest_now = now
            best: Optional[Detection] = None
            if detections:
                best = max(detections, key=lambda det: det.conf)
                self._last_det_t = best.t
                self._last_detection = best
            elif self._last_detection is not None:
                # No box in this frame: report nothing rather than a stale box.
                best = None
        if best is None:
            return None
        if max_age_s is not None and float(max_age_s) > 0.0:
            if (now - best.t) > float(max_age_s):
                return None
        return best

    def latest_detections(
        self, max_age_s: Optional[float] = None, now: Optional[float] = None
    ) -> list[Detection]:
        """All detections for the current frame (useful for OSD and tests)."""
        if not self._running:
            self.start()
        now = self._clock() if now is None else float(now)
        with self._lock:
            if (
                self._last_frame_detections is not None
                and self._last_latest_now is not None
                and abs(now - self._last_latest_now) <= 1e-9
            ):
                detections = list(self._last_frame_detections)
            else:
                index = self._current_index(now)
                detections = self._stamped(index, now)
                self._played += 1
                self._last_frame_detections = list(detections)
                self._last_latest_now = now
        if max_age_s is not None and float(max_age_s) > 0.0:
            detections = [d for d in detections if (now - d.t) <= float(max_age_s)]
        return detections

    def healthy(self, now: Optional[float] = None) -> bool:
        return self._running and bool(self.frames)

    def attach_source(self, source: Any) -> None:
        """Accepted for uniform app wiring; replay ignores live frames."""
        return None

    def submit_frame(self, frame: Any, t: Optional[float] = None) -> None:
        return None

    def stats(self) -> DetectorStats:
        with self._lock:
            return DetectorStats(
                backend="replay",
                frames=self._played,
                errors=0,
                drops=0,
                infer_ms=0.0,
                fps=0.0,
                last_frame_t=self._last_det_t,
                last_det_t=self._last_det_t,
                running=self._running,
                ready=True,
            )
