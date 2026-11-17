# %%
# Tracker test — Integration with TrackingController after refactor

import numpy as np
from legacy.rdk_x5_pico.tracking_controller import PID, TargetSelector, TrackingController
from rc_mux import TrackCmd

FRAME = np.zeros((720, 1280, 3), dtype=np.uint8)
DET = (np.array([[600, 330, 680, 390]], dtype=np.float32),
       np.array([0.9], dtype=np.float32),
       np.array([0], dtype=np.float32))
EMPTY = (np.empty((0, 4), dtype=np.float32),
         np.empty((0,), dtype=np.float32),
         np.empty((0,), dtype=np.float32))

# -- PID ------------------------------------------------------------------


def test_pid_dt_clamp_prevents_derivative_spike():
    pid = PID(kp=0.0, ki=0.0, kd=1.0, limit=300)
    pid.update(0.0, 0.02)
    out = pid.update(100.0, 1e-9)   # dt clamp -> 0.005
    assert abs(out) <= 300


def test_pid_dt_clamp_upper():
    pid = PID(kp=0.0, ki=1.0, kd=0.0, limit=300)
    pid.update(0.0, 0.02)
    out = pid.update(10.0, 100.0)   # dt clamp -> 0.2, integral adds 2.0
    assert abs(out) <= 2.0 + 1e-6


def test_pid_d_lowpass_smooths_kick():
    raw = PID(kp=0.0, ki=0.0, kd=1.0, limit=10000, d_lpf_alpha=1.0)
    filt = PID(kp=0.0, ki=0.0, kd=1.0, limit=10000, d_lpf_alpha=0.3)
    raw.update(0.0, 0.02)
    filt.update(0.0, 0.02)
    r = raw.update(50.0, 0.02)
    f = filt.update(50.0, 0.02)
    assert abs(f) < abs(r)


# -- TargetSelector --------------------------------------------------------


def test_selector_area_fallback_uses_lower_threshold():
    sel = TargetSelector(conf_thresh=0.6, min_track_age=5,
                         area_conf_thresh=0.4)
    boxes = np.array([[100, 100, 200, 200], [300, 300, 320, 320]],
                     dtype=np.float32)
    scores = np.array([0.45, 0.50], dtype=np.float32)
    cls = np.array([0, 0], dtype=np.float32)
    r = sel.select(None, boxes, scores, cls)
    assert r is not None
    assert r[2] == "nearest"
    assert r[0] == [100.0, 100.0, 200.0, 200.0]


def test_selector_high_conf_priority():
    sel = TargetSelector(conf_thresh=0.6, min_track_age=5)
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9], dtype=np.float32)
    cls = np.array([0], dtype=np.float32)
    r = sel.select(None, boxes, scores, cls)
    assert r is not None and r[2] == "conf"


# -- TrackingController ----------------------------------------------------


def make_tc():
    return TrackingController(attitude_fn=lambda: None)


def test_engage_returns_none_in_manual():
    tc = make_tc()
    cmd = tc.step(FRAME, *DET, tracker=None, now=100.0)
    assert cmd is None
    assert tc.state == "MANUAL"


def test_engage_then_detect_enters_tracking():
    tc = make_tc()
    tc.engage(1500, now=100.0)
    assert tc.state == "ARMED"
    cmd = tc.step(FRAME, *DET, tracker=None, now=100.03)
    assert tc.state == "TRACKING"
    assert isinstance(cmd, TrackCmd)
    assert cmd.t == 100.03


def test_lost_reacquire_continues_tracking():
    tc = make_tc()
    tc.engage(1500, now=100.0)
    tc.step(FRAME, *DET, tracker=None, now=100.03)
    tc.step(FRAME, *EMPTY, tracker=None, now=100.1)
    cmd = tc.step(FRAME, *EMPTY, tracker=None, now=102.8)   # >2s -> LOST
    assert tc.state == "LOST"
    assert cmd is not None
    cmd = tc.step(FRAME, *DET, tracker=None, now=103.0)       # reacquire
    assert tc.state == "TRACKING"
    assert cmd is not None
    assert cmd.t == 103.0


def test_lost_timeout_returns_to_manual():
    tc = make_tc()
    tc.engage(1500, now=100.0)
    tc.step(FRAME, *DET, tracker=None, now=100.03)
    tc.step(FRAME, *EMPTY, tracker=None, now=100.1)
    tc.step(FRAME, *EMPTY, tracker=None, now=102.8)          # -> LOST
    cmd = tc.step(FRAME, *EMPTY, tracker=None, now=108.0)     # >5s
    assert tc.state == "MANUAL"
    assert cmd is None


def test_search_sweep_flips_direction():
    tc = make_tc()
    tc.engage(1500, now=100.0)
    tc.step(FRAME, *DET, tracker=None, now=100.03)
    tc.step(FRAME, *EMPTY, tracker=None, now=100.1)
    tc.step(FRAME, *EMPTY, tracker=None, now=102.8)           # -> LOST
    c1 = tc.step(FRAME, *EMPTY, tracker=None, now=103.8)      # phase 0
    for t in (106.0, 106.3, 106.6):                            # intermediate
        tc.step(FRAME, *EMPTY, tracker=None, now=t)
    c2 = tc.step(FRAME, *EMPTY, tracker=None, now=106.9)      # phase 1
    assert (c1.yaw - 1500) * (c2.yaw - 1500) < 0               # 方向翻转


def test_coast_refreshes_timestamp():
    tc = make_tc()
    tc.engage(1500, now=100.0)
    tc.step(FRAME, *DET, tracker=None, now=100.03)
    tc.step(FRAME, *EMPTY, tracker=None, now=101.0)            # 丢失<2s: coast
    assert tc.state == "TRACKING"
    cmd = tc.step(FRAME, *EMPTY, tracker=None, now=101.03)
    assert cmd is not None and cmd.t == 101.03
