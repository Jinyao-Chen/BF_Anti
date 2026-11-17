"""FBDevSink tests against temp files and a fake sysfs tree (no /dev/fb0)."""
from __future__ import annotations

import numpy as np
import pytest

from onboard.video.fbdev import FBDevSink, FbGeometry, read_geometry


def make_sysfs(tmp_path, width=4, height=3, bpp=32, stride=None, node="fb0"):
    root = tmp_path / "sys"
    path = root / node
    path.mkdir(parents=True, exist_ok=True)
    (path / "virtual_size").write_text("%d,%d" % (width, height), encoding="ascii")
    (path / "bits_per_pixel").write_text(str(bpp), encoding="ascii")
    if stride is not None:
        (path / "stride").write_text(str(stride), encoding="ascii")
    (path / "name").write_text("testfb", encoding="ascii")
    return root


def make_fb(tmp_path, size, node="fb0"):
    dev = tmp_path / node
    dev.write_bytes(b"\x00" * size)
    return dev


def solid_frame(width=4, height=3, bgra=(10, 20, 30, 255)):
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    frame[:, :, 0] = bgra[0]
    frame[:, :, 1] = bgra[1]
    frame[:, :, 2] = bgra[2]
    frame[:, :, 3] = bgra[3]
    return frame


def test_read_geometry_32bpp(tmp_path):
    root = make_sysfs(tmp_path, width=720, height=576, bpp=32, stride=2880)
    geo = read_geometry("/dev/fb0", str(root))
    assert geo == FbGeometry(
        width=720, height=576, bpp=32, stride=2880, name="testfb"
    )
    assert geo.bytes_per_pixel == 4
    assert geo.row_bytes == 2880
    assert geo.frame_bytes == 2880 * 576


def test_read_geometry_missing_sysfs(tmp_path):
    assert read_geometry("/dev/fb0", str(tmp_path / "nope")) is None


def test_read_geometry_malformed_virtual_size(tmp_path):
    root = make_sysfs(tmp_path)
    (root / "fb0" / "virtual_size").write_text("not-a-size", encoding="ascii")
    assert read_geometry("/dev/fb0", str(root)) is None


def test_read_geometry_stride_fallback_and_oversized_stride(tmp_path):
    root = make_sysfs(tmp_path, width=4, height=3, bpp=32)
    assert read_geometry("/dev/fb0", str(root)).stride == 16
    root2 = make_sysfs(tmp_path / "second", width=4, height=3, bpp=32, stride=20)
    assert read_geometry("/dev/fb0", str(root2)).stride == 20


def test_open_and_write_bgra_32bpp(tmp_path):
    root = make_sysfs(tmp_path)
    dev = make_fb(tmp_path, 4 * 3 * 4)
    sink = FBDevSink(str(dev), 4, 3, sysfs_root=str(root))
    assert sink.open() is True
    assert sink.is_open
    frame = solid_frame()
    assert sink.write(frame) is True
    assert dev.read_bytes() == frame.reshape(3, 16).tobytes()
    assert sink.frames_written == 1
    assert sink.bytes_written == 48
    sink.close()
    assert not sink.is_open


def test_stride_padding_is_not_overwritten(tmp_path):
    root = make_sysfs(tmp_path, width=4, height=3, bpp=32, stride=20)
    dev = make_fb(tmp_path, 20 * 3)
    sink = FBDevSink(str(dev), 4, 3, sysfs_root=str(root))
    assert sink.open()
    frame = solid_frame()
    assert sink.write(frame)
    raw = dev.read_bytes()
    for row in range(3):
        base = row * 20
        assert raw[base:base + 16] == frame[row].tobytes()
        assert raw[base + 16:base + 20] == b"\x00" * 4


def test_write_bgr_24bpp(tmp_path):
    root = make_sysfs(tmp_path, width=2, height=1, bpp=24, stride=8)
    dev = make_fb(tmp_path, 8)
    sink = FBDevSink(str(dev), 2, 1, sysfs_root=str(root))
    assert sink.open()
    frame = np.array(
        [[[1, 2, 3, 255], [4, 5, 6, 255]]], dtype=np.uint8
    )
    assert sink.write(frame)
    # BGRA input -> BGR bytes, row padding preserved.
    assert dev.read_bytes() == bytes([1, 2, 3, 4, 5, 6, 0, 0])


def test_write_rgb565_16bpp(tmp_path):
    root = make_sysfs(tmp_path, width=2, height=1, bpp=16, stride=4)
    dev = make_fb(tmp_path, 4)
    sink = FBDevSink(str(dev), 2, 1, sysfs_root=str(root))
    assert sink.open()
    frame = np.array(
        [[[255, 0, 0, 255], [255, 255, 255, 255]]], dtype=np.uint8
    )
    assert sink.write(frame)
    assert dev.read_bytes() == bytes([0x1F, 0x00, 0xFF, 0xFF])


def test_unsupported_bpp_fails_gracefully(tmp_path):
    root = make_sysfs(tmp_path, width=2, height=2, bpp=8)
    dev = make_fb(tmp_path, 4)
    sink = FBDevSink(str(dev), 2, 2, sysfs_root=str(root))
    assert sink.open() is False
    assert "unsupported" in sink.last_error
    assert sink.write(solid_frame(2, 2)) is False


def test_missing_device_fails_gracefully(tmp_path):
    root = make_sysfs(tmp_path)
    sink = FBDevSink(str(tmp_path / "no-fb"), 4, 3, sysfs_root=str(root))
    assert sink.open() is False
    assert sink.is_open is False
    assert "open" in sink.last_error
    assert sink.write(solid_frame()) is False


def test_truncated_device_fails_gracefully(tmp_path):
    root = make_sysfs(tmp_path, width=100, height=100, bpp=32)
    dev = make_fb(tmp_path, 16)
    sink = FBDevSink(str(dev), 100, 100, sysfs_root=str(root))
    assert sink.open() is False
    assert "mmap" in sink.last_error


def test_write_before_open_returns_false(tmp_path):
    sink = FBDevSink(str(tmp_path / "fb0"), 4, 3, sysfs_root=str(tmp_path / "sys"))
    assert sink.write(solid_frame()) is False


def test_shape_mismatch_raises_value_error(tmp_path):
    root = make_sysfs(tmp_path, width=4, height=3, bpp=32)
    dev = make_fb(tmp_path, 4 * 3 * 4)
    sink = FBDevSink(str(dev), 4, 3, sysfs_root=str(root))
    assert sink.open()
    with pytest.raises(ValueError):
        sink.write(np.zeros((2, 2, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        sink.write(np.zeros((3, 4, 3), dtype=np.uint8))


def test_fallback_geometry_without_sysfs(tmp_path):
    dev = make_fb(tmp_path, 2 * 2 * 4)
    sink = FBDevSink(str(dev), 2, 2, sysfs_root=str(tmp_path / "missing-sys"))
    assert sink.open()
    assert sink.geometry.width == 2
    assert sink.geometry.height == 2
    assert sink.geometry.bpp == 32
    assert sink.geometry.stride == 8
    assert sink.write(solid_frame(2, 2)) is True


def test_non_contiguous_frame_is_copied(tmp_path):
    root = make_sysfs(tmp_path, width=4, height=3, bpp=32)
    dev = make_fb(tmp_path, 4 * 3 * 4)
    sink = FBDevSink(str(dev), 4, 3, sysfs_root=str(root))
    assert sink.open()
    frame = solid_frame()[:, ::-1]
    assert sink.write(frame) is True
    assert dev.read_bytes() == np.ascontiguousarray(frame).reshape(3, 16).tobytes()


def test_unblank_ioctl_is_issued_on_open_and_periodically(tmp_path, monkeypatch):
    import fcntl

    from onboard.video import fbdev as fbdev_mod

    calls = []

    def fake_ioctl(fd, request, arg):
        calls.append((request, arg))
        return 0

    monkeypatch.setattr(fcntl, "ioctl", fake_ioctl)
    monkeypatch.setattr(fbdev_mod, "UNBLANK_EVERY_FRAMES", 2)
    root = make_sysfs(tmp_path)
    dev = make_fb(tmp_path, 4 * 3 * 4)
    sink = FBDevSink(str(dev), 4, 3, sysfs_root=str(root))
    assert sink.open()
    assert calls == [(fbdev_mod.FBIOBLANK, fbdev_mod.FB_BLANK_UNBLANK)]
    assert sink.write(solid_frame())
    assert len(calls) == 1
    assert sink.write(solid_frame())
    assert len(calls) == 2
    sink.close()


def test_unblank_failure_is_graceful(tmp_path):
    root = make_sysfs(tmp_path)
    dev = make_fb(tmp_path, 4 * 3 * 4)
    sink = FBDevSink(str(dev), 4, 3, sysfs_root=str(root))
    assert sink.open()
    # FBIOBLANK is not supported on a regular file; this must not raise.
    assert sink.unblank() is False
    assert sink.write(solid_frame()) is True
    sink.close()
    assert sink.unblank() is False


def test_context_manager_and_idempotent_close(tmp_path):
    root = make_sysfs(tmp_path)
    dev = make_fb(tmp_path, 4 * 3 * 4)
    with FBDevSink(str(dev), 4, 3, sysfs_root=str(root)) as sink:
        assert sink.is_open
        assert sink.write(solid_frame())
    assert not sink.is_open
    sink.close()
    sink.close()
    assert not sink.is_open
