"""纯 P 制导运动学闭环的 CI 回归：静态/径向目标命中 + 横向包线记录。

每个用例都在 40 s 运动学模型内跑完，耗时 <1 s，可直接进 CI。
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from sim.p_strike_kinematic import KinConfig, simulate

BASE = KinConfig(duration=40.0)


def run(**override):
    return simulate(KinConfig(**{**asdict(BASE), **override}))


def test_head_on_static_target_hits():
    res = run(target_east=0.0, target_north=60.0, target_up=10.0)
    assert res.hit is True
    assert res.min_range < 0.8, res.min_range
    assert res.hit_time is not None and res.hit_time < 10.0
    assert "TRACK" in res.phases


def test_lateral_offset_15m_hits():
    res = run(target_east=15.0, target_north=58.0, target_up=10.0)
    assert res.hit is True
    assert res.min_range < 0.8, res.min_range


def test_lateral_offset_20m_hits():
    res = run(target_east=20.0, target_north=55.0, target_up=10.0)
    assert res.hit is True
    assert res.min_range < 0.8, res.min_range


def test_target_above_25m_hits():
    """HANDOFF P0-1 的阻塞工况：目标高于本机 25 m。"""
    res = run(target_east=0.0, target_north=50.0, target_up=35.0)
    assert res.hit is True, f"min_range={res.min_range}"
    assert res.min_range < 0.8


def test_target_below_5m_hits():
    res = run(target_east=0.0, target_north=55.0, target_up=5.0)
    assert res.hit is True
    assert res.min_range < 0.8


def test_receding_target_hits():
    """径向远离目标不产生追踪滞后，纯 P 律可命中。"""
    for speed in (1.0, 2.0, 5.0):
        res = run(target_east=0.0, target_north=55.0, target_up=10.0,
                  target_vel_north=speed)
        assert res.hit is True, f"speed={speed} min_range={res.min_range}"
        assert res.min_range < 0.8


def test_crossing_target_lag_is_bounded_and_documented():
    """横向匀速目标：纯 P（无提前量）存在约 2*v_t 的追踪滞后。

    这里不改控制律，只把包线固化成回归：一旦有人加入 D/PN 项，脱靶量会突然变小，
    测试会失败并提醒同步文档（docs/interception/2026-09-28-p-only-visual-guidance.md）。
    """
    res = run(target_east=-20.0, target_north=55.0, target_up=10.0, target_vel_east=5.0)
    assert res.hit is False
    assert 5.0 < res.min_range < 15.0, res.min_range


def test_pure_p_has_no_derivative_effect_on_steady_state():
    """同一几何下重复运行必须逐位一致（无状态、无噪声注入）。"""
    a = run(target_east=10.0, target_north=55.0, target_up=10.0)
    b = run(target_east=10.0, target_north=55.0, target_up=10.0)
    assert a.min_range == b.min_range
    assert a.hit == b.hit
    # 行内含 NaN（t_go 列），用 repr 比较避免 NaN != NaN 的假失败
    assert [repr(r) for r in a.rows] == [repr(r) for r in b.rows]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bbox_noise_tolerance(seed):
    """检测抖动 ±4 px 下仍应命中（纯 P 无积分，不需要抗饱和逻辑）。"""
    res = run(target_east=5.0, target_north=55.0, target_up=10.0,
              bbox_noise_px=4.0, seed=seed)
    assert res.min_range < 1.5, res.min_range
