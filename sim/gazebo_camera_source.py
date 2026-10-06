#!/usr/bin/env python3
"""Gazebo Harmonic camera image source (gz-transport) -> BGR numpy frame."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


class GazeboCameraSource:
    """订阅 Gazebo 相机 sensor 的 image topic，缓存最后一帧 BGR 图像。

    主题按 Harmonic 默认命名拼接，例如：
    /world/<world>/model/<outer>/model/iris_with_standoffs/link/base_link/sensor/camera_sensor/image
    """

    def __init__(
        self,
        world: str = "betaflight_world",
        outer_model: str = "iris_with_Betaflight",
        inner_model: str = "iris_with_standoffs",
        link: str = "base_link",
        sensor: str = "camera_sensor",
    ) -> None:
        import gz.transport13 as gztransport
        from gz.msgs10.image_pb2 import Image, PixelFormatType

        self.Image = Image
        self.PixelFormatType = PixelFormatType
        self._msg = None
        self._t = 0.0
        self.node = gztransport.Node()
        # inner_model 为空 = 模型已扁平化（<include merge="true">），
        # 此时相机挂在顶层模型的 link 上，话题里没有嵌套 model 段。
        if inner_model:
            topic = (
                f"/world/{world}/model/{outer_model}/model/{inner_model}"
                f"/link/{link}/sensor/{sensor}/image"
            )
        else:
            topic = (
                f"/world/{world}/model/{outer_model}"
                f"/link/{link}/sensor/{sensor}/image"
            )
        self.topic = topic
        if not self.node.subscribe(Image, topic, self._on_image):
            raise RuntimeError(f"failed to subscribe camera topic {topic}")

    def _on_image(self, msg) -> None:
        self._msg = msg
        self._t = time.monotonic()

    def frame(self) -> Optional[np.ndarray]:
        """返回 BGR uint8 (H,W,3)；还没有帧时返回 None。"""
        msg = self._msg
        if msg is None or msg.width <= 0 or msg.height <= 0:
            return None
        w, h = int(msg.width), int(msg.height)
        data = np.frombuffer(msg.data, dtype=np.uint8)
        fmt = int(msg.pixel_format_type)
        if fmt == self.PixelFormatType.BGR_INT8:
            channels = 3
            frame = data[: h * w * channels].reshape(h, w, channels)
            return frame.copy()
        if fmt == self.PixelFormatType.RGB_INT8:
            channels = 3
            frame = data[: h * w * channels].reshape(h, w, channels)
            return frame[:, :, ::-1].copy()
        if fmt == self.PixelFormatType.RGBA_INT8:
            channels = 4
            frame = data[: h * w * channels].reshape(h, w, channels)
            return frame[:, :, [2, 1, 0]].copy()
        if fmt == self.PixelFormatType.BGRA_INT8:
            channels = 4
            frame = data[: h * w * channels].reshape(h, w, channels)
            return frame[:, :, :3].copy()
        if fmt == self.PixelFormatType.L_INT8:
            gray = data[: h * w].reshape(h, w)
            return np.repeat(gray[:, :, None], 3, axis=2)
        raise ValueError(f"unsupported Gazebo pixel format {fmt}")

    @property
    def age_s(self) -> float:
        return float("inf") if self._t <= 0.0 else time.monotonic() - self._t
