"""Letterbox geometry and numpy-only detection postprocessing.

The Hailo HEF is fed a letterboxed RGB frame. Every detection therefore has to
be mapped back to the original camera frame before the tracker sees it. This
module owns that mapping and the CPU-side NMS/filter, and depends on numpy
only so it can be unit-tested without HailoRT or OpenCV.

Coordinate conventions
----------------------
* Model input is input_height x input_width (normally 640x640) and is produced
  by letterbox_image() / letterbox_params().
* YOLOv8 raw head output is (cx, cy, w, h) in letterboxed input pixels and the
  class axis is already sigmoid-activated (standard Ultralytics ONNX export),
  so there is no objectness term.
* Hailo on-device NMS output (TF layout) is (y_min, x_min, y_max, x_max)
  normalised to the letterboxed input, with classes on the first axis.
* Detection.bbox is always (x1, y1, x2, y2) in source-frame pixels.

All outputs are sorted deterministically (score descending, then class id and
top-left x) so a replay/bench run is bit-for-bit repeatable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from onboard.types import Detection

__all__ = [
    "DEFAULT_INPUT_SIZE",
    "Letterbox",
    "letterbox_image",
    "letterbox_params",
    "nms",
    "parse_detections",
    "parse_hailo_nms",
    "parse_yolo_v8_raw",
]

DEFAULT_INPUT_SIZE = 640
DEFAULT_PAD_VALUE = 114
_EPS = 1e-9


@dataclass(frozen=True)
class Letterbox:
    """Affine mapping between source pixels and letterboxed model pixels."""

    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale_x: float
    scale_y: float
    pad_x: float
    pad_y: float
    pad_value: int = DEFAULT_PAD_VALUE

    @property
    def scale(self) -> float:
        """Nominal scale (x and y differ only by integer-pad rounding)."""
        return float(self.scale_x)

    def to_model_point(self, x: float, y: float) -> tuple[float, float]:
        """Source pixel -> letterboxed model pixel."""
        return (
            float(x) * self.scale_x + self.pad_x,
            float(y) * self.scale_y + self.pad_y,
        )

    def to_source_point(self, x: float, y: float) -> tuple[float, float]:
        """Letterboxed model pixel -> source pixel."""
        return (
            (float(x) - self.pad_x) / self.scale_x,
            (float(y) - self.pad_y) / self.scale_y,
        )

    def to_source_box(
        self, box: Sequence[float], clip: bool = True
    ) -> tuple[float, float, float, float]:
        """Map an (x1, y1, x2, y2) box from model to source pixels."""
        x1, y1, x2, y2 = (float(v) for v in box)
        sx1, sy1 = self.to_source_point(x1, y1)
        sx2, sy2 = self.to_source_point(x2, y2)
        if sx2 < sx1:
            sx1, sx2 = sx2, sx1
        if sy2 < sy1:
            sy1, sy2 = sy2, sy1
        if clip:
            sx1 = min(max(sx1, 0.0), float(self.src_w))
            sy1 = min(max(sy1, 0.0), float(self.src_h))
            sx2 = min(max(sx2, 0.0), float(self.src_w))
            sy2 = min(max(sy2, 0.0), float(self.src_h))
        return sx1, sy1, sx2, sy2


def letterbox_params(
    src_w: int,
    src_h: int,
    dst_w: int = DEFAULT_INPUT_SIZE,
    dst_h: Optional[int] = None,
    scale_up: bool = True,
    pad_value: int = DEFAULT_PAD_VALUE,
) -> Letterbox:
    """Compute the letterbox transform without touching pixels."""
    src_w, src_h = int(src_w), int(src_h)
    dst_w = int(dst_w)
    dst_h = int(dst_w if dst_h is None else dst_h)
    if src_w <= 0 or src_h <= 0:
        raise ValueError("source width/height must be positive")
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError("destination width/height must be positive")
    scale = min(float(dst_w) / src_w, float(dst_h) / src_h)
    if not scale_up:
        scale = min(scale, 1.0)
    if scale <= 0.0:
        raise ValueError("letterbox scale must be positive")
    pad_x = (float(dst_w) - src_w * scale) * 0.5
    pad_y = (float(dst_h) - src_h * scale) * 0.5
    return Letterbox(
        src_w=src_w,
        src_h=src_h,
        dst_w=dst_w,
        dst_h=dst_h,
        scale_x=scale,
        scale_y=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        pad_value=int(pad_value),
    )


def letterbox_image(
    image: Any,
    dst_w: int = DEFAULT_INPUT_SIZE,
    dst_h: Optional[int] = None,
    pad_value: int = DEFAULT_PAD_VALUE,
    scale_up: bool = True,
) -> tuple[np.ndarray, Letterbox]:
    """Resize and pad an image to a letterboxed uint8 RGB frame.

    OpenCV is imported lazily: the postprocess tests and the laptop test suite
    never load it, while the Pi image path uses the same code as the camera.
    """
    import cv2  # local import: no OpenCV requirement for the mapping helpers

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pass
    else:
        raise ValueError("expected HxW, HxWx3 or HxWx4 image, got %r" % (arr.shape,))
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    src_h, src_w = int(arr.shape[0]), int(arr.shape[1])
    dst_h = int(dst_w if dst_h is None else dst_h)
    params = letterbox_params(
        src_w, src_h, dst_w, dst_h, scale_up=scale_up, pad_value=pad_value
    )

    new_w = max(1, int(round(src_w * params.scale_x)))
    new_h = max(1, int(round(src_h * params.scale_y)))
    new_w = min(new_w, dst_w)
    new_h = min(new_h, dst_h)
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    left = int(round(params.pad_x))
    top = int(round(params.pad_y))
    left = min(max(left, 0), dst_w - new_w)
    top = min(max(top, 0), dst_h - new_h)
    right = dst_w - new_w - left
    bottom = dst_h - new_h - top
    if left or top or right or bottom:
        out = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(int(pad_value),) * 3,
        )
    else:
        out = resized

    # Recompute scales from the integer crop actually applied so inverse
    # mapping stays exact even when rounding differs by one pixel.
    actual = Letterbox(
        src_w=src_w,
        src_h=src_h,
        dst_w=dst_w,
        dst_h=dst_h,
        scale_x=float(new_w) / float(src_w),
        scale_y=float(new_h) / float(src_h),
        pad_x=float(left),
        pad_y=float(top),
        pad_value=int(pad_value),
    )
    return np.ascontiguousarray(out), actual


def _box_iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one (4,) box against an (N, 4) array."""
    if boxes.size == 0:
        return np.empty(0, dtype=np.float64)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_a + area_b - inter
    return np.where(union > _EPS, inter / np.maximum(union, _EPS), 0.0)


def nms(boxes: Any, scores: Any, iou_threshold: float = 0.45) -> np.ndarray:
    """Class-agnostic greedy NMS. Returns kept indices, score-descending."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError("boxes and scores must have the same length")
    count = int(scores.shape[0])
    if count == 0:
        return np.empty(0, dtype=np.int64)
    threshold = float(iou_threshold)
    order = np.lexsort((np.arange(count, dtype=np.int64), -scores))
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _box_iou_one(boxes[current], boxes[rest])
        order = rest[ious <= threshold]
    return np.asarray(keep, dtype=np.int64)


def _validate_max_detections(max_detections: int) -> int:
    value = int(max_detections)
    if value < 0:
        raise ValueError("max_detections must be >= 0 (0 disables the cap)")
    return value


def _finish(
    boxes_model: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    params: Letterbox,
    max_detections: int,
    t: float,
) -> list[Detection]:
    if boxes_model.size == 0:
        return []
    out: list[Detection] = []
    for box, score, class_id in zip(boxes_model, scores, class_ids):
        x1, y1, x2, y2 = params.to_source_box(box)
        if x2 - x1 <= 0.5 or y2 - y1 <= 0.5:
            continue
        out.append(
            Detection(
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                conf=float(score),
                class_id=int(class_id),
                t=float(t),
            )
        )
    out.sort(key=lambda d: (-d.conf, d.class_id, d.bbox[0], d.bbox[1]))
    if max_detections > 0:
        out = out[:max_detections]
    return out


def _as_2d(output: Any) -> np.ndarray:
    arr = np.asarray(output, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("detection tensor cannot be a scalar")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError("expected a 2-D detection tensor, got %r" % (arr.shape,))
    return arr


def _normalise_yolo_layout(
    output: Any, num_classes: int, with_objectness: bool
) -> np.ndarray:
    arr = _as_2d(output)
    expected = 4 + int(num_classes) + (1 if with_objectness else 0)
    if arr.shape[1] == expected:
        return arr
    if arr.shape[0] == expected:
        return arr.T
    raise ValueError(
        "YOLO output shape %r does not match 4+classes%s=%d"
        % (arr.shape, "+objectness" if with_objectness else "", expected)
    )


def parse_yolo_v8_raw(
    output: Any,
    params: Letterbox,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 10,
    num_classes: Optional[int] = None,
    with_objectness: bool = False,
    class_names: Optional[Sequence[str]] = None,
    t: float = 0.0,
) -> list[Detection]:
    """Decode a flat YOLO head tensor to source-frame detections.

    Accepts both (1, 4+C, N) and (1, N, 4+C) layouts (also without the batch
    axis). with_objectness=True decodes YOLOv5-style (obj, cls...) columns;
    the default is YOLOv8 (cls... only). Coordinates are interpreted as
    letterboxed input pixels.
    """
    if num_classes is None:
        if class_names is not None and len(class_names) > 0:
            num_classes = len(class_names)
        else:
            arr = _as_2d(output)
            offset = 5 if with_objectness else 4
            candidates = [int(dim) - offset for dim in arr.shape]
            positive = [count for count in candidates if count > 0]
            if not positive:
                raise ValueError(
                    "cannot infer class count from YOLO output shape %r" % (arr.shape,)
                )
            # The class/channel axis is the small one; the anchor axis is 8400.
            num_classes = min(positive)
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    arr = _normalise_yolo_layout(output, num_classes, with_objectness)
    max_det = _validate_max_detections(max_detections)
    conf_th = float(conf_threshold)
    iou_th = float(iou_threshold)

    boxes = np.stack(
        [
            arr[:, 0] - arr[:, 2] * 0.5,
            arr[:, 1] - arr[:, 3] * 0.5,
            arr[:, 0] + arr[:, 2] * 0.5,
            arr[:, 1] + arr[:, 3] * 0.5,
        ],
        axis=1,
    )
    cls_start = 5 if with_objectness else 4
    cls_scores = arr[:, cls_start : cls_start + num_classes]
    class_ids = np.argmax(cls_scores, axis=1).astype(np.int64)
    best = np.max(cls_scores, axis=1)
    scores = arr[:, 4] * best if with_objectness else best

    finite = np.isfinite(scores) & np.isfinite(boxes).all(axis=1)
    if conf_th > 0.0:
        keep = finite & (scores >= conf_th)
    else:
        keep = finite & (scores > 0.0)
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    if boxes.shape[0] == 0:
        return []
    kept = nms(boxes, scores, iou_th)
    return _finish(boxes[kept], scores[kept], class_ids[kept], params, max_det, t)


def _nms_rows(
    rows: np.ndarray,
    params: Letterbox,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    t: float,
    normalized: bool,
    class_id: int = 0,
) -> list[Detection]:
    """Filter (M, 5) [y1, x1, y2, x2, score] rows, then map and NMS."""
    if rows.ndim != 2 or rows.shape[1] < 5:
        raise ValueError("NMS rows must have shape (M, 5), got %r" % (rows.shape,))
    rows = rows[:, :5]
    scores = rows[:, 4]
    finite = np.isfinite(rows).all(axis=1)
    if conf_threshold > 0.0:
        mask = finite & (scores >= float(conf_threshold))
    else:
        mask = finite & (scores > 0.0)
    rows = rows[mask]
    scores = scores[mask]
    if rows.shape[0] == 0:
        return []
    if normalized:
        x1 = rows[:, 1] * params.dst_w
        y1 = rows[:, 0] * params.dst_h
        x2 = rows[:, 3] * params.dst_w
        y2 = rows[:, 2] * params.dst_h
    else:
        y1, x1, y2, x2 = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    class_ids = np.full(rows.shape[0], int(class_id), dtype=np.int64)
    kept = nms(boxes, scores, float(iou_threshold))
    return _finish(
        boxes[kept],
        scores[kept],
        class_ids[kept],
        params,
        _validate_max_detections(max_detections),
        t,
    )


def _is_hailo_detection(obj: Any) -> bool:
    return all(
        hasattr(obj, name) for name in ("get_bbox", "get_score", "get_class_id")
    )


def _parse_hailo_detection_objects(
    objects: Sequence[Any], params: Letterbox
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = []
    scores = []
    class_ids = []
    for det in objects:
        bbox = np.asarray(det.get_bbox(), dtype=np.float64).reshape(-1)
        if bbox.size != 4:
            continue
        y1, x1, y2, x2 = bbox
        boxes.append(
            [
                x1 * params.dst_w,
                y1 * params.dst_h,
                x2 * params.dst_w,
                y2 * params.dst_h,
            ]
        )
        scores.append(float(det.get_score()))
        class_ids.append(int(det.get_class_id()))
    if not boxes:
        return (
            np.empty((0, 4), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
    return (
        np.asarray(boxes, dtype=np.float64),
        np.asarray(scores, dtype=np.float64),
        np.asarray(class_ids, dtype=np.int64),
    )


def _finish_from_nms_tensor(
    arr: np.ndarray,
    params: Letterbox,
    conf_th: float,
    iou_th: float,
    max_det: int,
    t: float,
    normalized: bool,
) -> list[Detection]:
    classes = arr.shape[0]
    flat = arr.transpose(0, 2, 1).reshape(-1, 5)  # (C*M, 5)
    class_ids = np.repeat(np.arange(classes, dtype=np.int64), arr.shape[2])
    scores = flat[:, 4]
    finite = np.isfinite(flat).all(axis=1)
    if conf_th > 0.0:
        mask = finite & (scores >= conf_th)
    else:
        mask = finite & (scores > 0.0)
    flat, class_ids = flat[mask], class_ids[mask]
    if flat.shape[0] == 0:
        return []
    x1 = flat[:, 1] * (params.dst_w if normalized else 1.0)
    y1 = flat[:, 0] * (params.dst_h if normalized else 1.0)
    x2 = flat[:, 3] * (params.dst_w if normalized else 1.0)
    y2 = flat[:, 2] * (params.dst_h if normalized else 1.0)
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    kept = nms(boxes, flat[:, 4], iou_th)
    return _finish(boxes[kept], flat[kept, 4], class_ids[kept], params, max_det, t)


def parse_hailo_nms(
    output: Any,
    params: Letterbox,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 10,
    normalized: bool = True,
    t: float = 0.0,
) -> list[Detection]:
    """Decode Hailo on-device NMS output.

    Accepted shapes/objects:

    * (C, 5, M) or (1, C, 5, M) float array, [y1, x1, y2, x2, score]
      (the get_buffer(tf_format=True) layout).
    * (5, M) single-class array.
    * (M, 5) rows for one class.
    * list/tuple of per-class (M, 5) arrays (HAILO_NMS_BY_CLASS hailo format),
      or HailoRT Detection objects exposing get_bbox/get_score/get_class_id.
    * Empty slots with score 0 are dropped by the confidence filter.
    """
    conf_th = float(conf_threshold)
    iou_th = float(iou_threshold)
    max_det = _validate_max_detections(max_detections)

    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            return []
        if _is_hailo_detection(output[0]):
            boxes, scores, class_ids = _parse_hailo_detection_objects(output, params)
            if boxes.shape[0] == 0:
                return []
            keep_conf = scores >= conf_th if conf_th > 0.0 else scores > 0.0
            boxes, scores, class_ids = (
                boxes[keep_conf],
                scores[keep_conf],
                class_ids[keep_conf],
            )
            if boxes.shape[0] == 0:
                return []
            kept = nms(boxes, scores, iou_th)
            return _finish(
                boxes[kept], scores[kept], class_ids[kept], params, max_det, t
            )
        candidates: list[Detection] = []
        for class_id, per_class in enumerate(output):
            arr = np.asarray(per_class, dtype=np.float64)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2 or arr.shape[1] < 5:
                raise ValueError(
                    "per-class NMS array must be (M, 5), got %r" % (arr.shape,)
                )
            classes = np.full(arr.shape[0], int(class_id), dtype=np.int64)
            mask = arr[:, 4] >= conf_th if conf_th > 0.0 else arr[:, 4] > 0.0
            arr = arr[mask]
            if arr.shape[0] == 0:
                continue
            x1 = arr[:, 1] * params.dst_w
            y1 = arr[:, 0] * params.dst_h
            x2 = arr[:, 3] * params.dst_w
            y2 = arr[:, 2] * params.dst_h
            boxes = np.stack([x1, y1, x2, y2], axis=1)
            kept = nms(boxes, arr[:, 4], iou_th)
            candidates.extend(
                _finish(boxes[kept], arr[kept, 4], classes[kept], params, 0, t)
            )
        candidates.sort(key=lambda d: (-d.conf, d.class_id, d.bbox[0], d.bbox[1]))
        return candidates[:max_det] if max_det > 0 else candidates

    arr = np.asarray(output, dtype=np.float64)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        raise ValueError(
            "flat Hailo NMS buffer %r needs its NMS metadata; pass "
            "get_buffer(tf_format=True) instead" % (arr.shape,)
        )
    if arr.ndim == 2:
        if arr.shape[1] == 5 and arr.shape[0] != 5:
            return _nms_rows(arr, params, conf_th, iou_th, max_det, t, normalized)
        if arr.shape[0] == 5:
            return _nms_rows(
                arr.T, params, conf_th, iou_th, max_det, t, normalized
            )
        raise ValueError("unsupported Hailo NMS shape %r" % (arr.shape,))
    if arr.ndim != 3 or arr.shape[1] != 5:
        raise ValueError(
            "expected (C, 5, M) Hailo NMS output, got %r" % (arr.shape,)
        )
    return _finish_from_nms_tensor(arr, params, conf_th, iou_th, max_det, t, normalized)


def _looks_like_hailo_nms(arr: np.ndarray) -> bool:
    if arr.ndim == 4:
        # (1, C, 5, M) tf layout, or (C, 5, M) with a leading unit axis.
        return arr.shape[2] == 5 or arr.shape[1] == 5
    if arr.ndim == 3:
        return arr.shape[1] == 5
    if arr.ndim == 2:
        return arr.shape[0] == 5 or arr.shape[1] == 5
    return False


def parse_detections(
    output: Any,
    params: Letterbox,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 10,
    num_classes: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    layout: str = "auto",
    t: float = 0.0,
) -> list[Detection]:
    """Dispatch to the YOLO or Hailo-NMS decoder.

    layout is "yolo", "hailo_nms" or "auto". Auto-detection is intentionally
    conservative; pass the explicit layout whenever the caller knows which HEF
    stream it holds (the Hailo backend does).
    """
    layout = str(layout).lower()
    if layout in ("yolo", "yolo_v8", "raw"):
        return parse_yolo_v8_raw(
            output,
            params,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            num_classes=num_classes,
            class_names=class_names,
            t=t,
        )
    if layout in ("hailo_nms", "nms", "hailo"):
        return parse_hailo_nms(
            output,
            params,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            t=t,
        )
    if layout != "auto":
        raise ValueError("unknown layout %r" % (layout,))

    if isinstance(output, (list, tuple)):
        return parse_hailo_nms(
            output,
            params,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections,
            t=t,
        )
    arr = np.asarray(output)
    if _looks_like_hailo_nms(arr):
        # A flat YOLO head (1, 5, 8400) also has a length-5 axis; use the size
        # of the candidate axis to tell 100 NMS slots from thousands of anchors.
        if arr.ndim == 3 and arr.shape[1] == 5 and arr.shape[2] > 1000:
            pass
        else:
            return parse_hailo_nms(
                output,
                params,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                t=t,
            )
    return parse_yolo_v8_raw(
        output,
        params,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        num_classes=num_classes,
        class_names=class_names,
        t=t,
    )
