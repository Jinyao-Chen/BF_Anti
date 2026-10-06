"""纯 P 视觉制导律的单元测试：契约（无 I/D）、符号、几何、限幅、命中、RC 映射。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from guidance.p_visual_guidance import (
    PStrikePhase,
    PVisualGuidance,
    _FORBIDDEN_STATE_HINTS,
    p_cmd_to_rc,
)
from perception.camera_geometry import rotation_earth_from_body

IMAGE = (640, 480)
HFOV = 68.75
TILT = 44.0


def make_guidance(**kwargs) -> PVisualGuidance:
    params = dict(image_size=IMAGE, hfov_deg=HFOV, camera_tilt_deg=TILT)
    params.update(kwargs)
    return PVisualGuidance(**params)


def project(point_enu, own_pos, own_rpy, intr, extr):
    """ENU 目标点 → 像素；相机看不到时返回 None。"""
    rel = np.asarray(point_enu, dtype=float) - np.asarray(own_pos, dtype=float)
    r_eb = rotation_earth_from_body(*own_rpy)
    rel_cam = extr.rotation_body_from_camera().T @ (r_eb.T @ rel)
    if rel_cam[2] <= 1e-6:
        return None
    return (float(intr.fx * rel_cam[0] / rel_cam[2] + intr.cx),
            float(intr.fy * rel_cam[1] / rel_cam[2] + intr.cy))


def bbox_around(u, v, width_px):
    return (u - width_px * 0.5, v - width_px * 0.5, u + width_px * 0.5, v + width_px * 0.5)


def bbox_for_target(guidance, own_pos, own_rpy, point_enu, width_px=40.0):
    uv = project(point_enu, own_pos, own_rpy, guidance.intrinsics, guidance.extrinsics)
    if uv is None:
        return None
    return bbox_around(uv[0], uv[1], width_px)


OWN_POS = np.array([0.0, 0.0, 10.0])


# --------------------------------------------------------------------- #
# 契约：纯 P
# --------------------------------------------------------------------- #
def test_same_input_gives_identical_output():
    """无积分/微分状态：同样输入连续两帧输出必须逐位相同。"""
    g = make_guidance()
    own_rpy = (0.0, -44.0, 0.0)
    bbox = bbox_around(360.0, 260.0, 40.0)
    a = g.update(bbox, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05)
    b = g.update(bbox, OWN_POS, np.zeros(3), own_rpy, 0.05, 0.05)
    assert a.roll_cmd_deg == b.roll_cmd_deg
    assert a.pitch_cmd_deg == b.pitch_cmd_deg
    assert a.yaw_rate_dps == b.yaw_rate_dps
    assert a.throttle_pwm == b.throttle_pwm


def test_no_integrator_or_derivative_state_attributes():
    """P-only 契约（2026-09-29 按用户要求放宽一处）。

    用户明确要求在竖向加"一小点 Ki"，因此允许存在 **默认惰性** 的竖向积分状态；
    其余 I/D 状态仍禁止，且 ki_z=0（默认）时积分不得参与任何输出。
    """
    g = make_guidance()
    assert g.ki_z == 0.0, "默认必须是纯 P：竖向积分增益为 0"
    names = list(vars(g).keys()) + [n for n in dir(g) if not n.startswith("__")]
    allowed = {"ki_z", "_z_int", "ki_z_limit_m"}
    for name in names:
        low = name.lower()
        if name in allowed:
            continue
        assert not any(h in low for h in _FORBIDDEN_STATE_HINTS), f"疑似 I/D 状态: {name}"


def test_vertical_integral_is_inert_by_default():
    """ki_z=0 时积分状态不得影响输出（逐帧与关闭积分完全一致）。"""
    a = make_guidance()
    b = make_guidance()
    assert b.ki_z == 0.0
    for k in range(30):
        u = 320.0 + 40.0 * (k % 5)
        bb = bbox_around(u, 300.0, 30.0)
        ca = a.update(bb, OWN_POS, np.zeros(3), (0.0, -30.0, 0.0), k * 0.05, 0.05)
        cb = b.update(bb, OWN_POS, np.zeros(3), (0.0, -30.0, 0.0), k * 0.05, 0.05)
        assert ca.throttle_pwm == pytest.approx(cb.throttle_pwm, abs=1e-12)


def test_roll_and_yaw_are_proportional_to_error():
    """仍是 P：无记忆、奇对称、单调，小误差处与线性一致，大误差处软饱和。

    2026-09-29：为抑制"捕获段指令撞 ±max_roll 限幅导致 bang-bang 抖动"，
    加了误差软饱和 e_cmd = w*tanh(e/w)（w=err_soft）。它不引入 I/D、不引入状态，
    但在大误差处不再严格等于 kp*e，故此处按软饱和形式断言。
    """
    g = make_guidance()
    w = g.err_soft
    own_rpy = (0.0, -44.0, 0.0)
    for u in (240.0, 300.0, 320.0, 400.0, 460.0):
        bbox = bbox_around(u, 240.0, 30.0)
        cmd = g.update(bbox, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05)
        e_u = (u - 320.0) / 320.0
        e_cmd = w * math.tanh(e_u / w) if w > 0 else e_u
        assert cmd.e_u == pytest.approx(e_u, abs=1e-12)          # 上报的仍是原始误差
        assert cmd.roll_cmd_deg == pytest.approx(g.kp_roll * e_cmd, abs=1e-9)
        assert cmd.yaw_rate_dps == pytest.approx(g.kp_yaw * e_cmd, abs=1e-9)
        # 软饱和性质：奇对称、单调、有界
        assert abs(e_cmd) < w * 1.0001
        if abs(e_u) < 0.1:
            assert e_cmd == pytest.approx(e_u, rel=0.05)          # 小误差≈线性(1%以内)


def test_pitch_is_monotone_in_vertical_error():
    """目标越靠画面下方 ⟹ 越低头（pitch 更负）。"""
    g = make_guidance()
    own_rpy = (0.0, -30.0, 0.0)
    pitches = []
    for v in (180.0, 220.0, 260.0, 300.0):
        bbox = bbox_around(320.0, v, 30.0)
        cmd = g.update(bbox, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05)
        pitches.append(cmd.pitch_cmd_deg)
    assert pitches == sorted(pitches, reverse=True), pitches
    assert pitches[0] > pitches[-1]


# --------------------------------------------------------------------- #
# 几何
# --------------------------------------------------------------------- #
def test_centered_target_sets_geometric_bias_to_tilt():
    """机身低头 44°、目标在画面中心 ⟹ 视线仰角≈0 ⟹ 俯仰指令≈-44°（保持）。"""
    g = make_guidance()
    bbox = bbox_around(g.intrinsics.cx, g.intrinsics.cy, 40.0)
    cmd = g.update(bbox, OWN_POS, np.zeros(3), (0.0, -TILT, 0.0), 0.0, 0.05)
    assert cmd.el_los_deg == pytest.approx(0.0, abs=0.5)
    assert cmd.pitch_cmd_deg == pytest.approx(-TILT, abs=1.0)
    assert cmd.pitch_bias_deg == pytest.approx(-TILT, abs=0.5)


def test_level_body_centered_target_needs_no_pitch():
    """机身水平、目标在画面中心 ⟹ 视线仰角≈44° ⟹ 俯仰指令≈0（几何自洽）。"""
    g = make_guidance()
    bbox = bbox_around(g.intrinsics.cx, g.intrinsics.cy, 40.0)
    cmd = g.update(bbox, OWN_POS, np.zeros(3), (0.0, 0.0, 0.0), 0.0, 0.05)
    assert cmd.el_los_deg == pytest.approx(TILT, abs=0.5)
    assert cmd.pitch_cmd_deg == pytest.approx(0.0, abs=1.0)


def test_visual_range_and_height_from_bbox_width():
    g = make_guidance()
    r = 20.0
    s_px = g.intrinsics.fx * g.target_width_m / r
    bbox = bbox_around(g.intrinsics.cx, g.intrinsics.cy, s_px)
    cmd = g.update(bbox, OWN_POS, np.zeros(3), (0.0, 0.0, 0.0), 0.0, 0.05)
    assert cmd.range_est_m == pytest.approx(r, rel=1e-6)
    # 目标在画面中心 + 机身为水平 ⟹ 视线仰角 = 44° ⟹ 目标高于本机
    assert cmd.z_target_est_m > OWN_POS[2]


def test_throttle_rises_when_target_above():
    # 小增益让两个工况都落在未饱和区（饱和由 test_clamps_extreme_errors 覆盖）
    g = make_guidance(kp_z=0.2, kp_vz=20.0)
    low = g.update(bbox_around(320.0, 260.0, 30.0), OWN_POS, np.zeros(3),
                   (0.0, 0.0, 0.0), 0.0, 0.05)
    high = g.update(bbox_around(320.0, 100.0, 30.0), OWN_POS, np.zeros(3),
                    (0.0, 0.0, 0.0), 0.0, 0.05)
    assert high.z_target_est_m > low.z_target_est_m
    assert high.throttle_pwm > low.throttle_pwm
    # 高度环是两个 P 串联：z_err → vz_cmd → pwm（无 I/D）
    def expect(z_err):
        vz = max(-g.vz_max_mps, min(g.vz_max_mps, g.kp_z * z_err))
        return max(g.thr_min_pwm, min(g.thr_max_pwm, g.hover_pwm + g.kp_vz * vz))
    assert high.throttle_pwm == pytest.approx(expect(high.z_target_est_m - OWN_POS[2]), rel=1e-9)
    assert low.throttle_pwm == pytest.approx(expect(low.z_target_est_m - OWN_POS[2]), rel=1e-9)


# --------------------------------------------------------------------- #
# 限幅 / 安全
# --------------------------------------------------------------------- #
def test_clamps_extreme_errors():
    g = make_guidance()
    far_right = g.update(bbox_around(639.0, 15.0, 30.0), OWN_POS, np.zeros(3),
                         (0.0, 0.0, 0.0), 0.0, 0.05)
    assert far_right.roll_cmd_deg <= g.max_roll_deg + 1e-9
    assert far_right.yaw_rate_dps <= g.max_yaw_rate_dps + 1e-9
    assert far_right.pitch_cmd_deg >= -g.max_pitch_down_deg - 1e-9
    assert far_right.pitch_cmd_deg <= g.max_pitch_up_deg + 1e-9
    assert g.thr_min_pwm <= far_right.throttle_pwm <= g.thr_max_pwm
    assert far_right.saturated is True


def test_low_altitude_guard_limits_nose_down():
    g = make_guidance()
    low_own = np.array([0.0, 0.0, 1.0])
    bbox = bbox_around(320.0, 470.0, 20.0)  # 目标在画面下方 → 想大低头
    cmd = g.update(bbox, low_own, np.zeros(3), (0.0, 0.0, 0.0), 0.0, 0.05)
    assert cmd.pitch_cmd_deg >= -15.0 - 1e-9
    assert cmd.throttle_pwm >= g.hover_pwm


# --------------------------------------------------------------------- #
# 无目标 / 命中
# --------------------------------------------------------------------- #
def test_no_bbox_searches_and_holds_altitude():
    g = make_guidance()
    g.z_hold_m = 10.0
    cmd = g.update(None, np.array([0.0, 0.0, 9.9]), np.zeros(3), (0, 0, 0), 0.0, 0.05)
    assert cmd.phase == PStrikePhase.COAST
    assert cmd.valid is False
    assert cmd.roll_cmd_deg == 0.0 and cmd.yaw_rate_dps == 0.0
    # 串联 P：vz_cmd = kp_z*(z_hold-z) 未饱和 → pwm = hover + kp_vz*vz_cmd
    assert cmd.throttle_pwm == pytest.approx(
        g.hover_pwm + g.kp_vz * (g.kp_z * 0.1), rel=1e-6)
    # 高度误差很大时被限幅（不会无限加油门）
    sat = g.update(None, np.array([0.0, 0.0, 1.0]), np.zeros(3), (0, 0, 0), 0.02, 0.05)
    assert sat.throttle_pwm == pytest.approx(g.thr_max_pwm, rel=1e-9)
    # 超过丢失超时后转 SEARCH
    cmd2 = g.update(None, np.array([0.0, 0.0, 9.0]), np.zeros(3), (0, 0, 0), 1.0, 0.05)
    assert cmd2.phase == PStrikePhase.SEARCH


def test_degenerate_bbox_is_rejected():
    g = make_guidance()
    assert g.update((10.0, 10.0, 11.0, 20.0), OWN_POS, np.zeros(3), (0, 0, 0), 0.0).phase == PStrikePhase.COAST
    assert g.update((0.0, float("nan"), 30.0, 30.0), OWN_POS, np.zeros(3), (0, 0, 0), 1.0).phase == PStrikePhase.SEARCH


def test_hit_when_bbox_reaches_hit_size_and_growing():
    g = make_guidance()
    s_hit = g.hit_s_px
    assert s_hit == pytest.approx(g.intrinsics.fx * 0.55 / 0.8, rel=1e-9)
    first = g.update(bbox_around(320.0, 240.0, s_hit * 0.90), OWN_POS, np.zeros(3),
                     (0.0, -44.0, 0.0), 0.0, 0.05)
    assert first.hit is False
    second = g.update(bbox_around(320.0, 240.0, s_hit * 1.05), OWN_POS, np.zeros(3),
                      (0.0, -44.0, 0.0), 0.05, 0.05)
    assert second.range_est_m <= g.hit_range_m
    assert second.closing_ratio > 0.0
    assert second.hit is True
    assert second.phase == PStrikePhase.HIT


def test_hit_requires_closing():
    """尺寸够大但在远离（bbox 缩小）时不判命中。"""
    g = make_guidance()
    s_hit = g.hit_s_px
    g.update(bbox_around(320.0, 240.0, s_hit * 1.2), OWN_POS, np.zeros(3), (0, -44, 0), 0.0)
    cmd = g.update(bbox_around(320.0, 240.0, s_hit * 1.05), OWN_POS, np.zeros(3), (0, -44, 0), 0.05)
    assert cmd.closing_ratio < 0.0
    assert cmd.hit is False


def test_hit_dwell_requires_consecutive_frames():
    g = make_guidance(hit_dwell=2)
    s_hit = g.hit_s_px
    g.update(bbox_around(320.0, 240.0, s_hit * 0.9), OWN_POS, np.zeros(3), (0, -44, 0), 0.0)
    a = g.update(bbox_around(320.0, 240.0, s_hit * 1.1), OWN_POS, np.zeros(3), (0, -44, 0), 0.05)
    b = g.update(bbox_around(320.0, 240.0, s_hit * 1.3), OWN_POS, np.zeros(3), (0, -44, 0), 0.10)
    assert a.hit is False and b.hit is True


# --------------------------------------------------------------------- #
# RC 映射
# --------------------------------------------------------------------- #
def test_rc_mapping_signs_and_aux():
    """符号标定（2026-09-28 第二次）：坐标映射修正后（E=-y_g, det=+1），
    水平面解释翻转，roll/yaw 不再取反；pitch 仍在垂直轴，低头为正。"""
    g = make_guidance()
    cmd = g.update(bbox_around(420.0, 300.0, 40.0), OWN_POS, np.zeros(3),
                   (0.0, -40.0, 0.0), 0.0, 0.05)
    assert cmd.roll_cmd_deg > 0 and cmd.pitch_cmd_deg < 0 and cmd.yaw_rate_dps > 0
    ch = p_cmd_to_rc(cmd, angle_limit_deg=g.angle_limit_deg)
    assert ch[4] == 2000 and ch[5] == 2000          # ARM + ANGLE
    assert ch[0] > 1500                              # 右滚 → roll 通道高于中位（修正后不取反）
    assert ch[1] > 1500                              # 低头 → pitch 通道高于中位
    assert ch[3] > 1500                              # 右偏航率（修正后不取反）
    assert 1000 <= ch[2] <= 2000
    # 抬头指令必须让 pitch 通道低于中位
    g2 = make_guidance()
    up_cmd = g2.update(bbox_around(320.0, 60.0, 40.0), OWN_POS, np.zeros(3),
                       (0.0, 0.0, 0.0), 0.0, 0.05)
    assert up_cmd.pitch_cmd_deg > 0
    assert p_cmd_to_rc(up_cmd, angle_limit_deg=g2.angle_limit_deg)[1] < 1500
    # 显式符号参数仍然是可覆盖的（换机型/固件要重标）
    flipped = p_cmd_to_rc(cmd, angle_limit_deg=g.angle_limit_deg, roll_sign=+1.0)
    assert flipped[0] > 1500
    flipped_yaw = p_cmd_to_rc(cmd, angle_limit_deg=g.angle_limit_deg, yaw_sign=+1.0)
    assert flipped_yaw[3] > 1500


def test_rc_rate_limit():
    g = make_guidance()
    cmd = g.update(bbox_around(639.0, 15.0, 30.0), OWN_POS, np.zeros(3),
                   (0.0, 0.0, 0.0), 0.0, 0.05)
    prev = [1500] * 16
    ch = p_cmd_to_rc(cmd, angle_limit_deg=g.angle_limit_deg, prev=prev, rate_limit=60)
    for axis in (0, 1, 3, 2):
        assert abs(ch[axis] - 1500) <= 60


# --------------------------------------------------------------------- #
# 可选的比例导引横向项（pn_gain>0）与视觉速度代理（speed_source）
# --------------------------------------------------------------------- #
def test_pn_term_disabled_by_default():
    """默认 pn_gain=0：喂不喂像素速率，输出必须完全一致。"""
    bb = bbox_around(340.0, 300.0, 40.0)
    own_rpy = (0.0, -30.0, 0.0)
    a = make_guidance()
    b = make_guidance()
    ca = a.update(bb, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05, True, 120.0)
    cb = b.update(bb, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05, True, 0.0)
    assert ca.roll_cmd_deg == pytest.approx(cb.roll_cmd_deg, abs=1e-12)


def test_pn_term_sign_and_magnitude():
    """pn_gain=N：滚转增量 = cap*tanh(raw/cap)，raw = atan(N*Vc*lambda_dot/g)。

    lambda_dot = u_dot/fx；第一帧低通状态为空，直接取原始值，便于手算验算。
    """
    bb = bbox_around(320.0, 300.0, 40.0)          # u 恰在 cx -> e_u = 0，基线干净
    own_rpy = (0.0, -30.0, 0.0)

    def run(rate):
        g = make_guidance(pn_gain=4.0, speed_source="closing")
        g._closing_mps = 10.0                     # 白盒：固定接近速度便于验算（同时过低速门）
        return g.update(bb, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05, True, rate)

    fx = make_guidance().intrinsics.fx
    cmd0, cmd1, cmd2 = run(0.0), run(46.78), run(-46.78)
    raw = math.degrees(math.atan(
        4.0 * 10.0 * math.radians(math.degrees(46.78 / fx)) / 9.80665))
    cap = 8.0
    expect = cap * math.tanh(raw / cap)
    assert cmd1.roll_cmd_deg - cmd0.roll_cmd_deg == pytest.approx(expect, abs=1e-9)
    assert cmd2.roll_cmd_deg - cmd0.roll_cmd_deg == pytest.approx(-expect, abs=1e-9)
    assert cmd0.roll_cmd_deg == pytest.approx(0.0, abs=1e-9)
    assert abs(cmd1.roll_cmd_deg) <= cap + 1e-9


def test_pn_term_is_soft_capped():
    """像素速率再大，比例导引项也不能超过 pn_roll_cap_deg（防滚转饱和 -> 绕飞）。"""
    bb = bbox_around(320.0, 300.0, 40.0)
    own_rpy = (0.0, -30.0, 0.0)
    cap = 8.0
    g = make_guidance(pn_gain=4.0, pn_roll_cap_deg=cap, speed_source="closing")
    g._closing_mps = 10.0
    cmd = g.update(bb, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05, True, 5000.0)
    assert abs(cmd.roll_cmd_deg) <= cap + 1e-9
    assert abs(cmd.roll_cmd_deg) > cap * 0.9          # 确实饱和到接近 cap


def test_pn_term_gated_inside_terminal_range():
    """终端段（bbox 撑满 -> range_est <= terminal_range）不用比例导引项。"""
    own_rpy = (0.0, -30.0, 0.0)
    bb = bbox_around(320.0, 300.0, 200.0)             # 框很大 -> 距离近
    g = make_guidance(pn_gain=4.0, speed_source="closing")
    g._closing_mps = 10.0
    cmd = g.update(bb, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05, True, 300.0)
    assert cmd.range_est_m <= g.terminal_range_m
    assert cmd.pn_roll_deg == pytest.approx(0.0, abs=1e-12)


def test_speed_source_closing_replaces_fc_velocity():
    """speed_source="closing"：飞控速度为 0 时，仍用视觉接近率触发超速治理。"""
    bb = bbox_around(340.0, 400.0, 40.0)
    own_rpy = (0.0, -30.0, 0.0)
    kw = dict(kp_speed=4.0, v_max_mps=5.0, max_pitch_down_deg=25.0,
              speed_source="closing")

    def run(closing):
        g = make_guidance(**kw)
        g._closing_mps = closing
        return g.update(bb, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05)

    fast, slow = run(16.0), run(1.0)
    assert fast.pitch_cmd_deg != slow.pitch_cmd_deg
    assert fast.pitch_cmd_deg > slow.pitch_cmd_deg     # 超速 -> 抬高低头下限

    fc = make_guidance(kp_speed=4.0, v_max_mps=5.0, max_pitch_down_deg=25.0)
    fc._closing_mps = 16.0
    ref = fc.update(bb, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05)
    assert ref.pitch_cmd_deg == pytest.approx(slow.pitch_cmd_deg, abs=1e-12)



def test_pn_term_gated_at_low_speed():
    """刚起飞时速度≈0：比例导引项必须为 0（否则会压出坡度 -> 侧向漂移）。

    现场：MASK_HORIZ=both 起飞段航向对准时 ψ̇≈-20~-60 dps，比例导引项被顶到 -8 度限幅，
    8 s 向左漂 2.65 m。低速门 + "缺帧不叠加偏航速率" 一起修掉这个现象。
    """
    bb = bbox_around(320.0, 300.0, 40.0)
    own_rpy = (0.0, -30.0, 0.0)
    g = make_guidance(pn_gain=4.0, pn_min_speed_mps=2.5)
    g._closing_mps = 0.5                          # 慢速：fc 源下 own_vel=0 -> 代理=0
    cmd = g.update(bb, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05, True, 800.0)
    assert cmd.pn_roll_deg == pytest.approx(0.0, abs=1e-12)
    # 一旦有速度（closing 源 = 10 m/s）就恢复
    g2 = make_guidance(pn_gain=4.0, pn_min_speed_mps=2.5, speed_source="closing")
    g2._closing_mps = 10.0
    cmd2 = g2.update(bb, OWN_POS, np.zeros(3), own_rpy, 0.0, 0.05, True, 800.0)
    assert abs(cmd2.pn_roll_deg) > 1.0

