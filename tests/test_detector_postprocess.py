"""Postprocess tests: letterbox geometry, NMS, YOLOv8 and Hailo NMS decode."""
import math

import numpy as np
import pytest

from onboard.detector.postprocess import (
    Letterbox,
    letterbox_image,
    letterbox_params,
    nms,
    parse_detections,
    parse_hailo_nms,
    parse_yolo_v8_raw,
)


def params_1280x720():
    return letterbox_params(1280, 720, 640, 640)


def test_letterbox_params_16x9():
    p = params_1280x720()
    assert p.scale == pytest.approx(0.5)
    assert p.pad_x == pytest.approx(0.0)
    assert p.pad_y == pytest.approx(140.0)
    assert p.to_model_point(640, 360) == pytest.approx((320.0, 320.0))
    x, y = p.to_source_point(320, 320)
    assert (x, y) == pytest.approx((640.0, 360.0))
    box = p.to_source_box((270.0, 295.0, 370.0, 345.0))
    assert box == pytest.approx((540.0, 310.0, 740.0, 410.0))


def test_letterbox_clips_to_source():
    p = params_1280x720()
    box = p.to_source_box((-50.0, -50.0, 5000.0, 5000.0))
    assert box == (0.0, 0.0, 1280.0, 720.0)


def test_letterbox_non_square_and_no_upscale():
    p = letterbox_params(320, 240, 640, 480, scale_up=False)
    assert p.scale == pytest.approx(1.0)
    assert (p.pad_x, p.pad_y) == pytest.approx((160.0, 120.0))
    p2 = letterbox_params(1280, 720, 640, 480)
    assert p2.scale == pytest.approx(0.5)
    assert (p2.pad_x, p2.pad_y) == pytest.approx((0.0, 60.0))


def test_letterbox_rejects_bad_geometry():
    with pytest.raises(ValueError):
        letterbox_params(0, 720)
    with pytest.raises(ValueError):
        letterbox_params(1280, 720, 0, 640)


def test_letterbox_image_shape_and_padding():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:, :] = 200
    out, p = letterbox_image(frame, 640, 640)
    assert out.shape == (640, 640, 3)
    assert out.dtype == np.uint8
    # Top/bottom padding must be the letterbox value, content keeps 200.
    assert out[0, 0].tolist() == [114, 114, 114]
    assert out[-1, -1].tolist() == [114, 114, 114]
    assert out[320, 320].tolist() == [200, 200, 200]
    assert out[140, 0].tolist() == [200, 200, 200]
    assert p.pad_y == pytest.approx(140.0)
    # The returned mapping must invert consistently.
    x, y = p.to_source_point(320.0, 320.0)
    assert (x, y) == pytest.approx((640.0, 360.0), abs=2.0)


def test_nms_basic_and_tie_stability():
    boxes = np.array(
        [
            [0.0, 0.0, 10.0, 10.0],
            [1.0, 1.0, 11.0, 11.0],
            [50.0, 50.0, 60.0, 60.0],
        ]
    )
    scores = np.array([0.9, 0.8, 0.7])
    kept = nms(boxes, scores, 0.45)
    assert kept.tolist() == [0, 2]
    tie = nms(boxes, np.array([0.5, 0.5, 0.5]), 0.45)
    assert tie.tolist() == [0, 2]
    assert nms(np.empty((0, 4)), np.empty((0,))) .tolist() == []
    with pytest.raises(ValueError):
        nms(boxes, scores[:2])


def _yolo_output(box, conf, layout, num_classes=1, extra=None):
    cols = 4 + num_classes
    rows = np.zeros((1, cols, 8400), dtype=np.float32)
    rows[0, 0, 0], rows[0, 1, 0], rows[0, 2, 0], rows[0, 3, 0] = box
    rows[0, 4, 0] = conf
    if extra is not None:
        rows[0, 4 + extra[0], 0] = extra[1]
    if layout == "cn":
        return rows
    return np.transpose(rows, (0, 2, 1))


@pytest.mark.parametrize("layout", ["cn", "nc"])
def test_parse_yolo_v8_raw_layouts(layout):
    p = params_1280x720()
    out = _yolo_output((320.0, 320.0, 100.0, 50.0), 0.9, layout)
    dets = parse_yolo_v8_raw(out, p, conf_threshold=0.25)
    assert len(dets) == 1
    assert dets[0].conf == pytest.approx(0.9)
    assert dets[0].class_id == 0
    assert dets[0].bbox == pytest.approx((540.0, 310.0, 740.0, 410.0), abs=0.6)


def test_parse_yolo_filters_low_conf_and_limits():
    p = params_1280x720()
    out = _yolo_output((320.0, 320.0, 10.0, 10.0), 0.1, "cn")
    out[0, :, 1] = [320.0, 320.0, 10.0, 10.0, 0.8]
    assert [d.conf for d in parse_yolo_v8_raw(out, p, conf_threshold=0.25)] == pytest.approx([0.8])
    assert len(parse_yolo_v8_raw(out, p, conf_threshold=0.05, max_detections=1)) == 1


def test_parse_yolo_multiclass():
    p = letterbox_params(640, 640, 640, 640)
    out = np.zeros((1, 6, 10), dtype=np.float32)
    out[0, :, 0] = [320.0, 320.0, 40.0, 40.0, 0.2, 0.9]
    dets = parse_yolo_v8_raw(out, p, conf_threshold=0.25, num_classes=2)
    assert len(dets) == 1
    assert dets[0].class_id == 1
    assert dets[0].conf == pytest.approx(0.9)
    assert dets[0].bbox == pytest.approx((300.0, 300.0, 340.0, 340.0))


def test_parse_yolo_objectness_layout():
    p = letterbox_params(640, 640, 640, 640)
    out = np.zeros((1, 7, 1), dtype=np.float32)
    out[0, :, 0] = [320.0, 320.0, 20.0, 20.0, 0.5, 0.0, 0.8]
    dets = parse_yolo_v8_raw(
        out, p, conf_threshold=0.25, num_classes=2, with_objectness=True
    )
    assert len(dets) == 1
    assert dets[0].conf == pytest.approx(0.4)
    assert dets[0].class_id == 1


def test_parse_yolo_bad_shape():
    p = letterbox_params(640, 640, 640, 640)
    with pytest.raises(ValueError):
        parse_yolo_v8_raw(np.zeros((1, 7, 10)), p, num_classes=1)


def test_parse_hailo_nms_tf_layout():
    p = params_1280x720()
    nms_out = np.zeros((1, 5, 100), dtype=np.float32)
    nms_out[0, :, 0] = [0.4, 0.3, 0.6, 0.7, 0.8]
    dets = parse_hailo_nms(nms_out, p, conf_threshold=0.25)
    assert len(dets) == 1
    assert dets[0].class_id == 0
    assert dets[0].bbox == pytest.approx((384.0, 232.0, 896.0, 488.0), abs=0.5)


def test_parse_hailo_nms_batch_axis_and_empty_slots():
    p = params_1280x720()
    nms_out = np.zeros((1, 2, 5, 100), dtype=np.float32)
    nms_out[0, 1, :, 0] = [0.25, 0.25, 0.5, 0.5, 0.7]
    dets = parse_hailo_nms(nms_out, p, conf_threshold=0.25)
    assert len(dets) == 1
    assert dets[0].class_id == 1
    assert dets[0].conf == pytest.approx(0.7)


def test_parse_hailo_nms_per_class_list():
    p = letterbox_params(640, 640, 640, 640)
    cls0 = np.array([[0.1, 0.1, 0.2, 0.2, 0.9]], dtype=np.float32)
    cls1 = np.array([[0.3, 0.3, 0.4, 0.4, 0.8]], dtype=np.float32)
    dets = parse_hailo_nms([cls0, cls1], p, conf_threshold=0.25)
    assert [d.class_id for d in dets] == [0, 1]
    assert [d.conf for d in dets] == pytest.approx([0.9, 0.8])
    assert dets[0].bbox == pytest.approx((64.0, 64.0, 128.0, 128.0), abs=0.5)


class _FakeHailoDetection:
    def __init__(self, bbox, score, class_id):
        self._bbox = bbox
        self._score = score
        self._class_id = class_id

    def get_bbox(self):
        return self._bbox

    def get_score(self):
        return self._score

    def get_class_id(self):
        return self._class_id


def test_parse_hailo_nms_detection_objects():
    p = letterbox_params(640, 640, 640, 640)
    objects = [
        _FakeHailoDetection((0.1, 0.1, 0.2, 0.2), 0.9, 0),
        _FakeHailoDetection((0.5, 0.5, 0.6, 0.6), 0.1, 0),
    ]
    dets = parse_hailo_nms(objects, p, conf_threshold=0.25)
    assert len(dets) == 1
    assert dets[0].conf == pytest.approx(0.9)


def test_parse_hailo_nms_2d_layouts():
    p = letterbox_params(640, 640, 640, 640)
    single = np.array([[0.1, 0.1, 0.2, 0.2, 0.9]], dtype=np.float32)
    assert len(parse_hailo_nms(single, p)) == 1
    transposed = single.T
    assert len(parse_hailo_nms(transposed, p)) == 1


def test_parse_hailo_nms_rejects_flat_buffer():
    p = letterbox_params(640, 640, 640, 640)
    with pytest.raises(ValueError):
        parse_hailo_nms(np.zeros(40080, dtype=np.float32), p)


def test_parse_detections_auto_layout():
    p = params_1280x720()
    yolo = _yolo_output((320.0, 320.0, 100.0, 50.0), 0.9, "cn")
    dets = parse_detections(yolo, p, conf_threshold=0.25, layout="auto")
    assert len(dets) == 1
    nms_out = np.zeros((1, 80, 5, 100), dtype=np.float32)
    nms_out[0, 0, :, 0] = [0.4, 0.3, 0.6, 0.7, 0.9]
    dets2 = parse_detections(nms_out, p, conf_threshold=0.25, layout="auto")
    assert len(dets2) == 1
    with pytest.raises(ValueError):
        parse_detections(nms_out, p, layout="bogus")


def test_postprocess_is_deterministic():
    p = params_1280x720()
    out = np.zeros((1, 5, 20), dtype=np.float32)
    out[0, :, 0] = [100.0, 100.0, 40.0, 40.0, 0.5]
    out[0, :, 1] = [120.0, 120.0, 40.0, 40.0, 0.5]
    out[0, :, 2] = [400.0, 400.0, 40.0, 40.0, 0.5]
    first = parse_yolo_v8_raw(out, p)
    second = parse_yolo_v8_raw(out.copy(), p)
    assert first == second
    assert [d.bbox[0] for d in first] == sorted(d.bbox[0] for d in first)
