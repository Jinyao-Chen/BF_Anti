import numpy as np

from modeling.reference_trajectory import PositionReference3D, VelocitySmoothing1D


def _run_1d(axis, vel_sp, duration=6.0, dt=0.01):
    axis.set_velocity_setpoint(vel_sp)
    max_a = 0.0
    max_j = 0.0
    max_v = 0.0
    history = []
    for _ in range(int(duration / dt)):
        axis.update(dt)
        max_a = max(max_a, abs(axis.acceleration))
        max_j = max(max_j, abs(axis.jerk))
        max_v = max(max_v, abs(axis.velocity))
        history.append((axis.position, axis.velocity, axis.acceleration, axis.jerk))
    return max_a, max_j, max_v, history


def test_velocity_smoothing_limits_and_converges():
    axis = VelocitySmoothing1D(max_jerk=4.0, max_accel=2.0, max_vel=5.0)
    axis.reset(0.0, 0.0, 0.0)

    max_accel, max_jerk, max_vel, _ = _run_1d(axis, 3.0, duration=4.0)

    assert max_accel <= 2.0001
    assert max_jerk <= 4.0001
    assert max_vel <= 3.0001
    assert abs(axis.velocity - 3.0) < 1e-3
    assert abs(axis.acceleration) < 1e-3
    assert axis.position > 0.0


def test_velocity_step_is_not_a_velocity_jump():
    axis = VelocitySmoothing1D(max_jerk=4.0, max_accel=2.0, max_vel=5.0)
    axis.reset(0.0, 0.0, 0.0)
    axis.set_velocity_setpoint(5.0)

    # 第一个 10 ms 内速度只能增加 0.5*a*dt^2 级别，绝不能瞬间到 5 m/s。
    axis.update(0.01)
    assert axis.velocity < 0.01
    assert abs(axis.acceleration) <= 0.041
    assert abs(axis.jerk) <= 4.0001


def test_position_reference_limits_and_convergence():
    ref = PositionReference3D(max_vel=5.0, max_accel=2.0, max_jerk=4.0)
    ref.reset([0.0, 0.0, 0.0])
    ref.set_target([10.0, 0.0, 0.0])

    max_v = max_a = max_j = 0.0
    for _ in range(800):
        state = ref.update(0.01, current_position=None)
        max_v = max(max_v, float(np.max(np.abs(state.velocity))))
        max_a = max(max_a, float(np.max(np.abs(state.acceleration))))
        max_j = max(max_j, float(np.max(np.abs(state.jerk))))

    assert max_v <= 5.0001
    assert max_a <= 2.0001
    assert max_j <= 4.0001
    assert abs(ref.position[0] - 10.0) < 0.05
    assert abs(ref.velocity[0]) < 0.02
    assert abs(ref.position[1]) < 1e-9


def test_position_reference_does_not_overshoot_step():
    ref = PositionReference3D(max_vel=5.0, max_accel=2.0, max_jerk=4.0)
    ref.reset([0.0, 0.0, 0.0])
    ref.set_target([4.0, 0.0, 0.0])
    max_pos = 0.0
    for _ in range(600):
        ref.update(0.01, current_position=None)
        max_pos = max(max_pos, ref.position[0])
    assert max_pos <= 4.02


def test_3d_axes_arrive_together_for_diagonal_path():
    ref = PositionReference3D(max_vel=5.0, max_accel=2.0, max_jerk=4.0)
    ref.reset([0.0, 0.0, 0.0])
    ref.set_target([6.0, 3.0, 1.5])

    arrival = [None, None, None]
    for step in range(2000):
        ref.update(0.01, current_position=None)
        for axis in range(3):
            if arrival[axis] is None and abs(ref.position[axis] - ref.target_position[axis]) < 0.05:
                arrival[axis] = step
        if all(a is not None for a in arrival):
            break

    assert all(a is not None for a in arrival)
    # 离散积分和逐周期重新规划会让各轴到达时间有几拍的差异；这里验证
    # “大致同时到达且路径保持直线”，而不是要求逐采样点完全同步。
    assert max(arrival) - min(arrival) <= 100  # 1.0 s
    # 路径应保持近似直线：各轴进度比例接近目标比例。
    scale = ref.position / np.array([6.0, 3.0, 1.5])
    assert np.max(np.abs(scale - scale[0])) < 0.08


def test_moving_target_feedforward_tracks_velocity():
    ref = PositionReference3D(max_vel=5.0, max_accel=2.0, max_jerk=4.0)
    ref.reset([0.0, 0.0, 0.0])
    target_vel = np.array([1.0, 0.0, 0.0])

    t = 0.0
    for _ in range(600):
        ref.set_target(target_vel * t, target_vel)
        ref.update(0.01, current_position=ref.position)
        t += 0.01

    assert abs(ref.velocity[0] - target_vel[0]) < 0.05
    assert ref.position[0] <= t * target_vel[0]
    assert t * target_vel[0] - ref.position[0] < 0.3


def test_zero_jerk_fallback_does_not_divide_by_zero():
    axis = VelocitySmoothing1D(max_jerk=0.0, max_accel=2.0, max_vel=5.0)
    axis.reset(0.0, 0.0, 0.0)
    axis.set_velocity_setpoint(2.0)
    axis.update(0.1)
    assert abs(axis.velocity - 2.0) < 1e-9
    assert abs(axis.position - 0.2) < 1e-9

    ref = PositionReference3D(max_vel=5.0, max_accel=0.0, max_jerk=0.0)
    ref.reset([0.0, 0.0, 0.0])
    ref.set_target([1.0, 0.0, 0.0])
    for _ in range(100):
        ref.update(0.01, current_position=ref.position)
    assert abs(ref.position[0] - 1.0) < 1e-6


def _simulate_reference_tracking(smoothed: bool):
    """一阶执行器 + PX4 风格 P/D + 加速度前馈的简化位置环。"""
    ref = None
    if smoothed:
        ref = PositionReference3D(max_vel=1.0, max_accel=1.0, max_jerk=2.0)
        ref.reset([0.0, 0.0, 0.0])
        ref.set_target([1.0, 0.0, 0.0])

    position = 0.0
    velocity = 0.0
    accel = 0.0
    prev_cmd = 0.0
    max_cmd_jerk = 0.0
    max_position = 0.0
    dt = 0.01
    tau = 0.2
    kp = 1.2
    kd = 2.0
    for _ in range(600):
        if smoothed:
            state = ref.update(dt, current_position=[position, 0.0, 0.0])
            p_ref = state.position[0]
            v_ref = state.velocity[0]
            a_ref = state.acceleration[0]
        else:
            p_ref, v_ref, a_ref = 1.0, 0.0, 0.0

        a_cmd = kp * (p_ref - position) + kd * (v_ref - velocity) + a_ref
        max_cmd_jerk = max(max_cmd_jerk, abs(a_cmd - prev_cmd) / dt)
        prev_cmd = a_cmd

        accel += (a_cmd - accel) * dt / tau
        velocity += accel * dt
        position += velocity * dt
        max_position = max(max_position, position)
    return max_cmd_jerk, max_position


def test_reference_smoothing_reduces_command_jerk():
    step_jerk, _ = _simulate_reference_tracking(smoothed=False)
    smooth_jerk, _ = _simulate_reference_tracking(smoothed=True)
    # 目标阶跃会让 a_cmd 在第一个周期跳变；jerk-limited 参考则从 0 连续上升。
    assert smooth_jerk < 0.25 * step_jerk
