"""OSD compositor tests with synthetic frames (no camera, no framebuffer)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onboard.safety import SafetyDecision
from onboard.types import ControlOutput, Detection, FcState, Mode
from onboard.video.osd import (
    COLOR_BG,
    COLOR_BOX,
    COLOR_BOX_ACQ,
    COLOR_CROSSHAIR,
    OsdRenderer,
    OsdStatus,
    fit_letterbox,
    format_readout,
)

GRAY = 100


def gray_frame(width=160, height=90):
    return np.full((height, width, 3), GRAY, dtype=np.uint8)


def color_mask(img, color):
    return np.all(img == np.array(color, dtype=np.uint8), axis=2)


def make_out(confirmed=True, mode=Mode.TRACK, range_m=42.0):
    command = SimpleNamespace(range_est_m=range_m) if range_m is not None else None
    return ControlOutput(mode=mode, confirmed=confirmed, command=command)


def make_state():
    return FcState(altitude_m=12.3, vario_mps=-0.5, voltage_v=15.8)


def test_fit_letterbox_16_9_to_pal():
    fit = fit_letterbox(1280, 720, 720, 576)
    assert fit.content_w == 720
    assert fit.content_h == 405
    assert fit.offset_x == 0
    assert fit.offset_y == 85
    assert fit.scale_x == pytest.approx(720 / 1280)
    assert fit.scale_y == pytest.approx(405 / 720)


def test_fit_letterbox_4_3_source():
    fit = fit_letterbox(640, 480, 720, 576)
    assert (fit.content_w, fit.content_h) == (720, 540)
    assert fit.offset_y == 18
    assert fit.offset_x == 0


def test_fit_stretch_disables_bars():
    fit = fit_letterbox(1280, 720, 720, 576, letterbox=False)
    assert (fit.content_w, fit.content_h) == (720, 576)
    assert (fit.offset_x, fit.offset_y) == (0, 0)
    assert fit.scale_x == pytest.approx(720 / 1280)
    assert fit.scale_y == pytest.approx(576 / 720)


def test_fit_rejects_bad_dimensions():
    with pytest.raises(ValueError):
        fit_letterbox(0, 720, 720, 576)
    with pytest.raises(ValueError):
        fit_letterbox(1280, 720, 720, 0)


def test_map_point_round_trip():
    fit = fit_letterbox(1280, 720, 720, 576)
    x, y = fit.map_point(640, 360)
    assert x == pytest.approx(360.0, abs=0.6)
    assert y == pytest.approx(360.0 * 0.5625 + 85, abs=0.6)


def test_render_shape_dtype_and_alpha():
    renderer = OsdRenderer(320, 240, osd=False)
    img = renderer.render(gray_frame())
    assert img.shape == (240, 320, 4)
    assert img.dtype == np.uint8
    assert np.all(img[:, :, 3] == 255)


def test_letterbox_bars_are_black_and_content_is_scaled():
    renderer = OsdRenderer(320, 240, osd=False)
    img = renderer.render(gray_frame())
    # 160x90 -> 320x180, centred: 30 px bars top and bottom.
    assert np.all(img[:30] == np.array(COLOR_BG, dtype=np.uint8))
    assert np.all(img[210:] == np.array(COLOR_BG, dtype=np.uint8))
    assert img[120, 160].tolist() == [GRAY, GRAY, GRAY, 255]


def test_stretch_fills_the_whole_canvas():
    renderer = OsdRenderer(160, 90, osd=False, letterbox=False)
    img = renderer.render(gray_frame())
    assert img.shape == (90, 160, 4)
    assert img[0, 0].tolist() == [GRAY, GRAY, GRAY, 255]
    assert img[-1, -1].tolist() == [GRAY, GRAY, GRAY, 255]


def test_osd_off_has_no_crosshair_or_box():
    renderer = OsdRenderer(160, 90, osd=False)
    det = Detection((20.0, 10.0, 60.0, 40.0), 0.9)
    img = renderer.render(gray_frame(), det, make_out(), make_state(), None, 1.0)
    assert img[45, 80].tolist() == [GRAY, GRAY, GRAY, 255]
    assert not color_mask(img, COLOR_CROSSHAIR).any()
    assert not color_mask(img, COLOR_BOX).any()
    assert not color_mask(img, COLOR_BOX_ACQ).any()


def test_crosshair_marks_exact_centre():
    renderer = OsdRenderer(160, 90)
    img = renderer.render(gray_frame())
    assert img[45, 80].tolist() == list(COLOR_CROSSHAIR)


def test_box_is_drawn_through_the_letterbox_mapping():
    renderer = OsdRenderer(320, 240)
    det = Detection((20.0, 10.0, 60.0, 40.0), 0.9)
    img = renderer.render(
        gray_frame(), det, make_out(confirmed=True), make_state(),
        SafetyDecision(True, "ok"), 1.0,
    )
    mask = color_mask(img, COLOR_BOX)
    assert mask.any()
    ys, xs = np.nonzero(mask)
    # 2x scale + 30 px top bar: source box 20..60 x 10..40 -> 40..120 x 50..110.
    assert abs(int(xs.max()) - 120) <= 3
    assert abs(int(ys.max()) - 110) <= 3
    assert int(xs.min()) <= 44


def test_unconfirmed_detection_uses_acquisition_colour():
    renderer = OsdRenderer(320, 240)
    det = Detection((20.0, 10.0, 60.0, 40.0), 0.5)
    img = renderer.render(gray_frame(), det, make_out(confirmed=False))
    assert color_mask(img, COLOR_BOX_ACQ).any()
    assert not color_mask(img, COLOR_BOX).any()


def test_draw_box_option_disables_only_the_box():
    renderer = OsdRenderer(160, 90, draw_box=False)
    det = Detection((20.0, 10.0, 60.0, 40.0), 0.9)
    img = renderer.render(gray_frame(), det, make_out(confirmed=True))
    assert not color_mask(img, COLOR_BOX).any()
    assert color_mask(img, COLOR_CROSSHAIR).any()


def test_box_is_clipped_to_canvas():
    renderer = OsdRenderer(160, 90)
    det = Detection((-100.0, -100.0, 500.0, 500.0), 0.9)
    img = renderer.render(gray_frame(), det, make_out(confirmed=True))
    assert img.shape == (90, 160, 4)
    assert color_mask(img, COLOR_BOX).any()


def test_nan_detection_is_ignored():
    renderer = OsdRenderer(160, 90)
    det = Detection((float("nan"), 0.0, 10.0, 10.0), 0.9)
    img = renderer.render(gray_frame(), det, make_out(confirmed=True))
    assert img.shape == (90, 160, 4)
    assert not color_mask(img, COLOR_BOX).any()


def test_frame_none_renders_black_with_overlay():
    renderer = OsdRenderer(160, 90)
    img = renderer.render(None)
    assert img[0, 0].tolist() == list(COLOR_BG)
    assert img[45, 80].tolist() == list(COLOR_CROSSHAIR)


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_supported_input_channels(channels):
    renderer = OsdRenderer(160, 90, osd=False)
    if channels == 1:
        frame = np.full((90, 160), GRAY, dtype=np.uint8)
    else:
        frame = np.full((90, 160, channels), GRAY, dtype=np.uint8)
    img = renderer.render(frame)
    assert img[45, 80].tolist() == [GRAY, GRAY, GRAY, 255]


def test_float_frame_is_cast_to_uint8():
    renderer = OsdRenderer(160, 90, osd=False)
    frame = np.full((90, 160, 3), GRAY, dtype=np.float32)
    img = renderer.render(frame)
    assert img[45, 80].tolist() == [GRAY, GRAY, GRAY, 255]


def test_format_readout_full():
    status = OsdStatus(loop_hz=50.0, latency_ms=3.2, display_fps=25.0, dropped=4)
    readout = format_readout(
        make_out(confirmed=True), make_state(), SafetyDecision(True, "ok"),
        status, Detection((0.0, 0.0, 1.0, 1.0), 0.9),
    )
    assert readout["mode"] == "MODE TRACK"
    assert readout["lock"] == "LOCK"
    assert readout["safety"] == "SAFE OK"
    assert readout["alt"] == "ALT   12.3m  VS  -0.5m/s"
    assert readout["battery"] == "BAT 15.80V"
    assert readout["range"] == "RNG  42.0m"
    assert readout["loop"] == "LOOP  50.0Hz"
    assert readout["latency"] == "LAT   3.2ms"
    assert readout["display"] == "DSP 25.0fps  DRP 4"


def test_format_readout_denied_and_missing_values():
    readout = format_readout(
        make_out(confirmed=True, mode=Mode.SEARCH, range_m=None),
        FcState(), SafetyDecision(False, "override_off"),
    )
    assert readout["mode"] == "MODE SEARCH"
    assert readout["safety"] == "SAFE OVERRIDE_OFF"
    assert readout["battery"] == "BAT  --.--V"
    assert readout["range"] == "RNG   ---"


def test_format_readout_defaults_to_boot():
    readout = format_readout(None, None, None)
    assert readout["mode"] == "MODE BOOT"
    assert readout["lock"] == "NO LOCK"
    assert readout["safety"] == "SAFE N/A"


def test_format_readout_acquisition_when_detection_unconfirmed():
    readout = format_readout(
        make_out(confirmed=False), None, SafetyDecision(False, "disarmed"),
        detection=Detection((1.0, 2.0, 3.0, 4.0), 0.5),
    )
    assert readout["lock"] == "ACQ"


def test_mode_change_alters_hud_pixels():
    renderer = OsdRenderer(320, 240)
    track = renderer.render(gray_frame(), None, make_out(mode=Mode.TRACK))
    hit = renderer.render(gray_frame(), None, make_out(mode=Mode.HIT))
    assert np.any(track != hit)


def test_custom_output_size_and_font_fallback():
    renderer = OsdRenderer(320, 240, font_path="/nonexistent/font.ttf")
    renderer.set_output_size(160, 90)
    img = renderer.render(gray_frame())
    assert img.shape == (90, 160, 4)
    with pytest.raises(ValueError):
        renderer.set_output_size(0, 90)


def test_target_age_is_rendered_without_crashing():
    renderer = OsdRenderer(320, 240)
    det = Detection((20.0, 10.0, 60.0, 40.0), 0.9, t=10.0)
    img = renderer.render(gray_frame(), det, make_out(), None, None, 10.25)
    assert img.shape == (240, 320, 4)
