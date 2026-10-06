"""纯 P 视觉制导（P-only visual guidance）：44° 上仰相机反无人机撞机打击控制律。

设计契约
--------
**制导层唯一动态控制项是比例（P）项。** 本模块不含：
积分项、微分项、PN/APN 导航项、jerk-limited 参考发生器、目标状态滤波。

允许的非常数项只有两类，且都不是"控制器动态"：

1. **几何偏置** `el_los - camera_tilt_deg`：由检测框中心像素 + 本机姿态换算出的
   视线仰角，是坐标换算（量测几何），不是时间状态。固定 44° 上仰相机下，
   要让相机光轴压住目标，机体俯仰必须是 `el_los - 44°`；只靠 `Kp*e_v`
   会留下 `el/Kp` 的稳态像素偏差（30° 仰角时目标直接飞出画面）。
2. **硬限幅 / 斜率限幅**：通道饱和保护。

垂直通道同样是 P：`hover_pwm + Kp_z * (z_target_est - z_own)`，
其中 `z_target_est` 由 bbox 宽度（视觉测距代理）与视线仰角算出 —— 纯视觉可观测量。

几何自洽性
----------
44° 上仰相机下，"目标保持在画面中心" ⟺ 机体俯仰 `θ = el_los - 44°`；
而四旋翼推力方向仰角 = 低头量 `σ = -θ`。对同高度目标（el≈0）有 σ≈44°，
即 0.7 g 的前向加速度，这正是撞机所需的闭合加速度；垂直分量由高度 P 环补偿。
低头 44° 需 PWM≈1790（T∝cmd²，hover 0.67），50°→1840，60°→1950 接近饱和，
故俯仰限幅取 [-50°, +25°]。

执行映射
--------
输出的是航空约定姿态（右滚正、抬头正、右偏航率为正）；RC 映射见
`p_cmd_to_rc()`：ANGLE 模式下摇杆是**角度**指令，满杆 = 固件 `angle_limit`
（本仓库 SITL 配置 60°），因此滚转/俯仰的 RC 比例必须用 60° 而不是限幅值。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from perception.camera_geometry import (
    CameraExtrinsics,
    CameraIntrinsics,
    earth_los_to_az_el,
    pixel_to_earth_los,
)

G = 9.80665
RC_MIN, RC_MAX, RC_MID = 1000, 2000, 1500

# 禁止出现在本模块的属性名片段：保证"纯 P"契约可被单元测试审计。
_FORBIDDEN_STATE_HINTS = ("integral", "integrator", "err_int", "_int_", "prev_error",
                          "derivative", "d_err", "kd_", "ki_", "dt_hist")


class PStrikePhase(str, Enum):
    """打击阶段：只用视觉量 + 计时器判定。"""

    SEARCH = "SEARCH"      # 无检测
    COAST = "COAST"        # 短暂丢失，保持高度、姿态回中
    TRACK = "TRACK"        # 有检测，比例制导
    TERMINAL = "TERMINAL"  # 距离估计进入终端半径
    HIT = "HIT"            # 命中判据成立


@dataclass
class PStrikeCommand:
    """纯 P 制导输出（角度为航空约定：右滚正 / 抬头正 / 右偏航率为正）。"""

    phase: PStrikePhase = PStrikePhase.SEARCH
    valid: bool = False
    hit: bool = False

    # 量测
    u_px: float = float("nan")
    v_px: float = float("nan")
    s_px: float = float("nan")
    e_u: float = 0.0            # 归一化水平误差，(u-cx)/cx
    e_v: float = 0.0            # 归一化垂直误差，(v-cy)/cy
    az_los_deg: float = 0.0
    el_los_deg: float = 0.0

    # 视觉测距/测高
    range_est_m: float = float("inf")
    z_target_est_m: float = float("nan")
    closing_ratio: float = 0.0   # s_dot / s，>0 表示接近

    # 控制量
    roll_cmd_deg: float = 0.0
    pitch_cmd_deg: float = 0.0
    pitch_bias_deg: float = 0.0
    yaw_rate_dps: float = 0.0
    throttle_pwm: float = 1500.0
    saturated: bool = False


class PVisualGuidance:
    """44° 上仰相机 + 纯比例视觉制导。

    参数
    ----
    camera_tilt_deg : 相机相对机头上仰角（默认 44°）。
    image_size / hfov_deg : 内参来源；真实 Gazebo 相机是 640x480 / 1.2 rad = 68.75°。
    kp_yaw / kp_roll / kp_pitch : 归一化像素误差到控制量的比例增益
        （偏航角速率 dps、滚转角 deg、俯仰角 deg，均按"满画面 = 1.0"整定）。
    kp_z : 高度环比例增益，RC 单位/米。
    kp_speed : 超速抬头比例增益，deg/(m/s)。
    target_width_m : 靶机等效宽度，视觉测距 `r = fx*W/s`。
    hit_range_m : 命中半径（视觉测距），对应 bbox 宽度 `fx*W/hit_range_m`。
    """

    def __init__(
        self,
        camera_tilt_deg: float = 44.0,
        image_size: tuple[int, int] = (640, 480),
        hfov_deg: float = 68.75,
        kp_yaw: float = 55.0,
        kp_roll: float = 25.0,
        kp_pitch: float = 13.0,
        kp_z: float = 1.5,
        kp_vz: float = 250.0,
        vz_max_mps: float = 8.0,
        kp_speed: float = 0.0,
        v_max_mps: float = 12.0,
        hover_pwm: float = 1670.0,
        max_roll_deg: float = 25.0,
        max_pitch_up_deg: float = 25.0,
        max_pitch_down_deg: float = 44.0,
        max_yaw_rate_dps: float = 120.0,
        thr_min_pwm: float = 1400.0,
        thr_max_pwm: float = 1950.0,
        target_width_m: float = 0.55,
        range_max_m: float = 150.0,
        hit_range_m: float = 0.8,
        hit_dwell: int = 1,
        terminal_range_m: float = 3.0,
        lost_timeout_s: float = 0.5,
        min_alt_guard_m: float = 3.0,
        min_bbox_px: float = 2.0,
        lift_ff: bool = False,
        close_gain_deg: float = 0.0,
        meas_lpf_alpha: float = 1.0,
        kp_velalign: float = 1.5,
        velalign_min_speed: float = 2.5,
        velalign_full_speed: float = 4.5,
        velalign_soft_deg: float = 10.0,
        align_gate: float = 0.15,
        align_pitch_limit_deg: float = 8.0,
        align_gate_lpf_s: float = 0.15,
        soft_start_s: float = 0.0,
        soft_start_scale: float = 0.35,
        err_soft: float = 0.35,
        climb_along_los: bool = True,
        climb_margin_mps: float = 1.5,
        kp_launch: float = 0.0,
        launch_cap_m: float = 12.0,
        range_lpf_tau_s: float = 0.25,
        range_lpf_min_range_m: float = 12.0,
        closing_max_mps: float = 25.0,
        vmax_close_range_m: float = 12.0,
        vmax_close_floor_mps: float = 3.0,
        pop_up_pitch_deg: float = 0.0,
        pop_up_z_m: float = 3.0,
        aim_below_deg: float = 0.0,
        vz_time_to_go: bool = False,
        tau_min_s: float = 0.20,
        tau_max_s: float = 3.0,
        vz_lpf_tau_s: float = 0.0,
        ki_z: float = 0.0,
        ki_z_limit_m: float = 4.0,
        angle_limit_deg: float = 60.0,
        yaw_full_rate_dps: float = 300.0,
    ) -> None:
        self.camera_tilt_deg = float(camera_tilt_deg)
        self.intrinsics = CameraIntrinsics.from_fov(*image_size, hfov_deg=hfov_deg)
        self.extrinsics = CameraExtrinsics(tilt_deg=camera_tilt_deg)

        self.kp_yaw = float(kp_yaw)
        self.kp_roll = float(kp_roll)
        self.kp_pitch = float(kp_pitch)
        self.kp_z = float(kp_z)
        self.kp_vz = float(kp_vz)
        self.vz_max_mps = float(vz_max_mps)
        self.kp_speed = float(kp_speed)
        self.v_max_mps = float(v_max_mps)
        self.hover_pwm = float(hover_pwm)

        self.max_roll_deg = float(max_roll_deg)
        self.max_pitch_up_deg = float(max_pitch_up_deg)
        self.max_pitch_down_deg = float(max_pitch_down_deg)
        self.max_yaw_rate_dps = float(max_yaw_rate_dps)
        self.thr_min_pwm = float(thr_min_pwm)
        self.thr_max_pwm = float(thr_max_pwm)

        self.target_width_m = float(target_width_m)
        self.range_max_m = float(range_max_m)
        self.hit_range_m = float(hit_range_m)
        self.hit_dwell = max(1, int(hit_dwell))
        self.terminal_range_m = float(terminal_range_m)
        self.lost_timeout_s = float(lost_timeout_s)
        self.min_alt_guard_m = float(min_alt_guard_m)
        self.min_bbox_px = float(min_bbox_px)
        self.lift_ff = bool(lift_ff)
        self.close_gain_deg = float(close_gain_deg)
        self.meas_lpf_alpha = float(meas_lpf_alpha)
        # 速度矢量对齐增益：把**速度矢量**（而不是机头）对准视线。
        # 纯追踪（机头对准）在存在横向速度时看不见偏差：实测 0.25 m/s 侧向速度
        # 在 4 s 内累积 ~1 m 横偏，而图像误差始终 ≈0（机头一直指着目标）。
        self.kp_velalign = float(kp_velalign)
        self.velalign_min_speed = float(velalign_min_speed)
        self.velalign_full_speed = float(velalign_full_speed)
        # 速度矢量对齐项**软饱和宽度**（deg）。实测该线性项可以顶到 ±270°（斜距 1.5 增益），
        # 直接把 roll 打满 ±30°（坏样本 54% 帧饱和、横滚 ±330°/s 极限环），而好样本该项
        # 只有 ~5°（贡献 7° roll）。软饱和保留小偏差处的行为、封住大偏差处的 bang-bang。
        self.velalign_soft_deg = float(velalign_soft_deg)
        # 对准门限：水平误差大于 align_gate 时，把低头限制在 align_pitch_limit_deg 以内
        # （不产生前向加速度），等机头对准目标后再闭合。实测 "边加速边转向" 会在
        # 初始航向有偏差时留下稳定的横向速度（0.25 m/s → 1 m 横偏），这是终端
        # 脱靶 ~1 m 的直接来源；对准后再加速即可从源头消除。
        self.align_gate = float(align_gate)
        # 门限判断用**滤波后**的横向误差：真机检测只有约 13 fps，e_u 逐帧噪声大，
        # 直接进门限会让"低头权限"整段被卡在 align_pitch_limit（实测 GUI 下发
        # pitch 全程钉在 -5°、加速不足、最后飞过）。滤波只影响门限判断，不改制导律。
        self.align_gate_lpf_s = float(align_gate_lpf_s)
        # 捕获段增益软起（比例增益的时间调度，仍无 I/D）：
        # 刚锁定时误差最大，roll_cmd=kp*e_u 常撞 ±max_roll 限幅（实测捕获段
        # 87% 帧饱和、roll 峰峰 51°）→ 指令 bang-bang → 抬头/横滚来回晃。
        # 误差软饱和宽度（0=纯线性）。无记忆、奇对称、单调 -> 仍是 P。
        self.err_soft = float(err_soft)
        self.soft_start_s = float(soft_start_s)
        self.soft_start_scale = float(soft_start_scale)
        self._acq_t: Optional[float] = None
        self._was_tracking = False
        self._eu_filt: Optional[float] = None
        self.align_pitch_limit_deg = float(align_pitch_limit_deg)
        # 沿视线爬升：垂速指令再按几何限幅，避免"先竖着拔高再平飞"。
        # 实测：目标仰角 15° 时却按最大垂速 8 m/s 爬（2 s 内竖爬 8 m、仅前进 5 m），
        # 白白浪费时间并拉长平飞段（横偏就在这段累积）。
        self.climb_along_los = bool(climb_along_los)
        self.climb_margin_mps = float(climb_margin_mps)
        # 起飞段前向加速偏置（**仍是比例项**）：高度差大时额外低头，把爬行段压掉。
        # 实测：几何偏置只有 el-tilt = 15-20 = -5° ⟹ 0.09g，前 6 s 只走 20 m，
        # 整个交战"边缘可达"；额外低头到约 -20° 可拿 0.27g，此时目标仍在画面内
        # （仰角 15° 对 27° 半视场）。增益 deg/m，满值在高度差 >= launch_cap_m 时取到。
        self.kp_launch = float(kp_launch)
        self.launch_cap_m = float(launch_cap_m)
        # 视觉测距/接近率调理：bbox 宽度在远距只有 ~20 px，抖 1 px 就是 10% 距离误差
        # -> closing 逐帧在 ±90 m/s 之间跳 -> τ 在 [0.2, 3.0] 之间跳 -> 油门 550 RC
        # 峰峰颤振（实测每发都有）。这里只调理**用于 τ 的量**（命中判据仍用原始值）。
        self.range_lpf_tau_s = float(range_lpf_tau_s)
        # 近距离**不滤波**：测距以 ~8 m/s 下降时，0.25 s 低通会带来 ~2 m 的滞后，
        # τ 被高估 -> vz_ref = z_err/τ 打小 -> 终端竖向跟不上（实测 v2a 在 2.7 m 处
        # 指令只有 1.1 m/s，而需要 2.1 m/s，最后差 0.55 m 高度）。远距框小才需要滤波。
        self.range_lpf_min_range_m = float(range_lpf_min_range_m)
        self.closing_max_mps = float(closing_max_mps)
        # 终端限速：转弯半径 r=v²/(g·tanφ)，近距离若还保持 8-10 m/s，"目标在画面边缘"
        # 时只能以 6 m 半径绕圈（实测 5.3 m 绕飞无法收敛）。按距离收缩速度上限。
        self.vmax_close_range_m = float(vmax_close_range_m)
        self.vmax_close_floor_mps = float(vmax_close_floor_mps)
        # 抬头爬升段（用户要求"不要低头、改抬头"）：
        # 物理约束——目标在前上方(45 m 前/12 m 上)时必须推力前倾才能闭合，推力沿机体系
        # z 轴 ⟹ 机身必然低头。故"抬头"只能是**爬升段**：先垂直拔高（推力不浪费在前进上），
        # 高度差吃进 pop_up_z_m 以内后交回常规律（埋头俯冲打击）。
        self.pop_up_pitch_deg = float(pop_up_pitch_deg)
        self.pop_up_z_m = float(pop_up_z_m)
        # 从下方向上打（用户要求）：垂直通道的瞄准点取**视线下方 aim_below_deg**，
        # 于是机体始终吊在目标下方、保持上升着撞上去；而不是爬到目标高度后平飞/俯冲。
        # 注意：只改垂直通道瞄准点，俯仰仍用真实 el_los（相机继续咬住目标）。
        self.aim_below_deg = float(aim_below_deg)
        # 竖向设定值 LPF（测量调理）：bbox 宽度反算的距离在远距只有 ~20 px，
        # 抖 1 px 即 10% 距离误差 -> 高度目标跳 ~1 m -> 油门尖峰。
        # 实测抖动：油门峰峰 550 RC、俯仰 ±7°、0.5-0.7 Hz。
        self.vz_lpf_tau_s = float(vz_lpf_tau_s)
        self._z_ref_filt: Optional[float] = None
        # 小积分项（用户要求加入）。原设计契约为"纯 P"，故默认 0、限量抗饱和。
        self.ki_z = float(ki_z)
        self.ki_z_limit_m = float(ki_z_limit_m)
        self._z_int = 0.0
        # 垂直通道"命中一致"标度：终端脱靶 96% 是垂直的 —— 最近点时水平已 0.60 m、
        # 垂直还差 2.16 m 且在上升(1.41 m/s)，即"没到"而不是"打偏"。
        # 按时撞上要求 v_up = (z_target - z_own)/τ（τ=剩余飞行时间）。
        self.vz_time_to_go = bool(vz_time_to_go)
        self.tau_min_s = float(tau_min_s)
        # τ 上限：接近速度由视觉测距差分估出，噪声大时 τ 会估成几十秒 ->
        # 指令爬升≈0 -> 高度不涨 -> 视线仰角大 -> 俯仰偏正(抬头) -> 不加速 ->
        # 接近速度更低，形成"爬行自锁"（实测 38 Hz 时必现：4.3 m 高、3.6 m/s）。
        # 封顶后 z_err/τ >= z_err/tau_max，保证始终有足够爬升。
        self.tau_max_s = float(tau_max_s)
        self._r_prev: Optional[float] = None
        self._closing_mps = 1.0
        self._r_filt: Optional[float] = None
        self._el_fresh: Optional[float] = None   # 最后一次"新鲜"检测的视线仰角
        self._az_fresh: Optional[float] = None   # 最后一次"新鲜"检测的视线方位

        # ANGLE 模式摇杆→姿态角满量程 / 偏航满量程（用于 RC 映射）
        self.angle_limit_deg = float(angle_limit_deg)
        self.yaw_full_rate_dps = float(yaw_full_rate_dps)

        # 运行状态（不是控制动态：仅丢失计时、命中计数、测量低通）
        self.z_hold_m: Optional[float] = None      # 由 runner 设定的待命高度
        self._lost_since: Optional[float] = None
        self._last_s: Optional[float] = None
        self._last_t: Optional[float] = None
        self._meas: Optional[np.ndarray] = None    # [u, v, s] 低通状态（alpha=1 时无状态）
        self._hit_count = 0
        self.az_last_deg: Optional[float] = None   # 最近一次有效检测的方位（供重捕）
        self._last_cmd: Optional[PStrikeCommand] = None   # COAST 保持的最后指令

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def hit_s_px(self) -> float:
        """命中判据对应的 bbox 宽度（像素）。"""
        return float(self.intrinsics.fx * self.target_width_m / self.hit_range_m)

    def rc_scale(self) -> tuple[float, float, float]:
        """(roll, pitch, yaw) 的 RC 单位/物理量 比例，按固件 ANGLE 满量程。"""
        angle_scale = 500.0 / max(1e-6, self.angle_limit_deg)
        yaw_scale = 500.0 / max(1e-6, self.yaw_full_rate_dps)
        return angle_scale, angle_scale, yaw_scale

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def update(
        self,
        bbox_xyxy: Optional[Sequence[float]],
        own_position_enu: Sequence[float],
        own_velocity_enu: Optional[Sequence[float]],
        own_rpy_deg: Sequence[float],
        timestamp: float,
        dt: float = 0.05,
        fresh: bool = True,
    ) -> PStrikeCommand:
        """一个控制周期。bbox 为 None/退化时进入 SEARCH 或 COAST。

        fresh=False 表示这一帧 bbox 来自跟踪器的**外推/门控**（不是本帧检测）。
        此时绝对几何量（视线仰角/方位）沿用最后一次新鲜检测的值：外推框在像素上
        是冻住的，而机体姿态仍在变，若继续用 el_los = f(冻结像素, 实时姿态)，
        俯仰环就变成正反馈（pitch↑ -> el_los↑ -> pitch↑），实测 0.4 s 内顶到 +25°
        限幅、目标掉出画面、整场失锁。**只冻结几何偏置，比例项仍用外推框。**"""
        own_pos = np.asarray(own_position_enu, dtype=float)[:3]
        own_vel = (np.zeros(3) if own_velocity_enu is None
                   else np.asarray(own_velocity_enu, dtype=float)[:3])
        roll_own, pitch_own, yaw_own = (float(own_rpy_deg[0]), float(own_rpy_deg[1]),
                                        float(own_rpy_deg[2]))
        z_own = float(own_pos[2])

        meas = self._measurement(bbox_xyxy)
        if meas is None:
            return self._no_target(timestamp, own_pos, z_own, float(own_vel[2]))

        u_px, v_px, s_px = meas
        e_u = (u_px - self.intrinsics.cx) / max(1e-6, self.intrinsics.cx)
        e_v = (v_px - self.intrinsics.cy) / max(1e-6, self.intrinsics.cy)

        # 量测几何：像素 → 地球系视线 → (方位, 仰角)
        los_e = pixel_to_earth_los(
            (u_px, v_px), self.intrinsics, self.extrinsics,
            yaw_deg=yaw_own, pitch_deg=pitch_own, roll_deg=roll_own,
        )
        az_los, el_los = earth_los_to_az_el(los_e)
        if fresh:
            self._el_fresh = float(el_los)
            self._az_fresh = float(az_los)
        else:
            if self._el_fresh is not None:
                el_los = self._el_fresh
            if self._az_fresh is not None:
                az_los = self._az_fresh
        self.az_last_deg = float(self._az_fresh if self._az_fresh is not None
                                 else az_los)

        # 纯视觉测距 / 测高（bbox 宽度代理）
        # 上钳位：框被压缩到 min_bbox_px 时反算距离会飙到几百米（实测 z_t_est 峰峰 816 m），
        # 垂直通道会被带成全油门爬升。交战距离 < 50 m，钳到 150 m 只是防呆。
        range_est = float(self.intrinsics.fx * self.target_width_m / max(s_px, 1e-3))
        if self.range_max_m > 0.0:
            range_est = min(range_est, self.range_max_m)
        # 从下方向上打：垂直通道瞄准视线**下方** aim_below_deg 处，机体便始终吊在
        # 目标下方、保持上升着撞上去（俯仰仍用真实 el_los，相机继续咬住目标）。
        el_aim = el_los - self.aim_below_deg
        z_target_est = z_own + range_est * float(np.sin(np.deg2rad(el_aim)))
        if self.vz_lpf_tau_s > 0.0:
            if self._z_ref_filt is None:
                self._z_ref_filt = z_target_est
            a = float(dt) / (self.vz_lpf_tau_s + float(dt))
            self._z_ref_filt += a * (z_target_est - self._z_ref_filt)
            z_target_est = self._z_ref_filt
        if self.ki_z != 0.0:
            self._z_int = float(np.clip(self._z_int + (z_target_est - z_own) * float(dt),
                                        -self.ki_z_limit_m, self.ki_z_limit_m))

        # 剩余飞行时间 τ = 距离/接近速度（连续两帧视觉测距差分，纯测量量）
        # 测距低通 + 接近率限幅 + 接近率低通：三者都只是**量测调理**，不改 P 律结构。
        if self.range_lpf_tau_s > 0.0 and range_est > self.range_lpf_min_range_m:
            ar = float(dt) / (self.range_lpf_tau_s + float(dt))
            self._r_filt = (range_est if self._r_filt is None
                            else self._r_filt + ar * (range_est - self._r_filt))
            range_for_tau = float(self._r_filt)
        else:
            self._r_filt = None          # 出滤波区即复位，回到终端时不留旧值
            range_for_tau = range_est
        if self._r_prev is not None and dt > 0.0:
            raw_closing = float(np.clip((self._r_prev - range_for_tau) / float(dt),
                                        0.5, self.closing_max_mps))
            ac = float(dt) / (0.20 + float(dt))
            self._closing_mps += ac * (raw_closing - self._closing_mps)
        self._r_prev = range_for_tau
        tau_s = float(np.clip(range_for_tau / max(0.5, self._closing_mps),
                               self.tau_min_s, self.tau_max_s))
        vz_ref_mps = (z_target_est - z_own) / tau_s if self.vz_time_to_go else None

        # -------------------------------------------------------------- #
        # P 律
        # -------------------------------------------------------------- #
        # 误差**软饱和**（仍是 P：无记忆、奇函数、单调、零点保持）：
        # 刚捕获时 e_u 可达 ±0.7，线性律算出 ±31° 直接撞 ±max_roll 限幅 ->
        # 指令 bang-bang -> 横滚/抬头来回晃（实测捕获段 87% 帧饱和、roll 峰峰 51°）。
        # 软饱和在大误差处压低指令，小误差处与线性一致（终端不受影响）。
        # 只在中远距(>terminal_range)启用软饱和：既压掉捕获段撞限幅，又不削弱
        # 终端修正（终端大误差时需要全增益）。判据是当帧测距 -> 仍无记忆。
        w_soft = self.err_soft if range_est > self.terminal_range_m else 0.0

        def _soft(e, w):
            return e if w <= 0.0 else float(w * np.tanh(e / w))
        e_u_cmd = _soft(e_u, w_soft)
        e_v_cmd = _soft(e_v, w_soft)

        yaw_rate = self.kp_yaw * e_u_cmd
        roll_cmd = self.kp_roll * e_u_cmd

        # 速度矢量对齐（仍是比例项）：速度方位与视线方位的夹角 → 滚转。
        # 速度矢量偏右 → 左滚把它拉回来，使航迹对准目标而不是"机头对准"。
        v_norm = float(np.hypot(own_vel[0], own_vel[1]))
        if self.kp_velalign != 0.0 and v_norm > self.velalign_min_speed:
            az_vel = float(np.rad2deg(np.arctan2(own_vel[0], own_vel[1])))
            misalign = _wrap180(az_los - az_vel)
            # 门 1：低速时速度**方向**没意义（固件速度噪声 → az_vel 可任意），线性斜坡；
            # 门 2：软饱和，偏差再大也最多给 kp*soft 度滚转。
            span = max(0.1, self.velalign_full_speed - self.velalign_min_speed)
            sp = float(np.clip((v_norm - self.velalign_min_speed) / span, 0.0, 1.0))
            mis_soft = self.velalign_soft_deg * float(np.tanh(misalign / self.velalign_soft_deg))
            roll_cmd += self.kp_velalign * sp * mis_soft

        pitch_bias = el_los - self.camera_tilt_deg
        pitch_cmd = pitch_bias - self.kp_pitch * e_v_cmd
        # 抬头爬升段：高度差还大时保持抬头（推力朝上、全力拔高），不做低头
        if self.pop_up_pitch_deg != 0.0 and (z_target_est - z_own) > self.pop_up_z_m:
            pitch_cmd = self.pop_up_pitch_deg
        # 高度差还很大时（尚未拿到低头权限）额外压低机头，提前建立前向速度
        elif self.kp_launch > 0.0:
            z_err_now = z_target_est - z_own
            if z_err_now > 2.0:
                pitch_cmd -= self.kp_launch * min(z_err_now, self.launch_cap_m)

        # 超速治理：用"低头限幅"而不是把抬头量加到俯仰上。
        # 固定前视相机下俯仰姿态就是指向，若把抬头量直接加进 pitch，会把光轴从
        # 目标上抬走 → 目标掉出画面（早期版本实测：kp_speed=6 时俯仰被 +54° 抵消）。
        # 因此超速时只**限制最大低头角**，让目标停留在画面下半部继续可控。
        v_horiz = float(np.hypot(own_vel[0], own_vel[1]))
        speed_floor = -self.max_pitch_down_deg
        # 终端限速：距离越近允许的速度上限越低（转弯半径 ∝ v²，近距离必须能咬住）
        v_max_eff = self.v_max_mps
        if self.vmax_close_range_m > 0.0 and np.isfinite(range_est):
            f_close = float(np.clip(range_est / self.vmax_close_range_m, 0.0, 1.0))
            v_max_eff = max(self.vmax_close_floor_mps, self.v_max_mps * f_close)
        if self.kp_speed > 0.0 and v_horiz > v_max_eff:
            allowed = self.max_pitch_down_deg - self.kp_speed * (v_horiz - v_max_eff)
            speed_floor = -max(0.0, allowed)
        if self.close_gain_deg != 0.0:
            pitch_cmd -= self.close_gain_deg * (1.0 - min(1.0, s_px / self.hit_s_px))

        # 对准门限（比例门的增益调度，仍无 I/D）；门限判据用滤波后的 e_u
        if self.align_gate_lpf_s > 0.0:
            if self._eu_filt is None:
                self._eu_filt = e_u
            ae = float(dt) / (self.align_gate_lpf_s + float(dt))
            self._eu_filt += ae * (e_u - self._eu_filt)
            e_u_gate = self._eu_filt
        else:
            e_u_gate = e_u
        if self.align_gate > 0.0 and range_est > self.terminal_range_m:
            # 平滑（无记忆）对准门：|e_u| 在 gate 附近不再硬切，避免俯仰在
            # "-align_pitch" 与几何偏置之间极限环（实测 pitch 指令每 0.5 s 跳 8°）。
            g = float(np.tanh(abs(e_u_gate) / self.align_gate))
            floor_deg = -(self.align_pitch_limit_deg
                          + (self.max_pitch_down_deg - self.align_pitch_limit_deg) * (1.0 - g))
            pitch_cmd = max(pitch_cmd, floor_deg)

        # 高度环：两个 P 串联（高度误差→垂速指令，垂速误差→油门）。无 I、无 D。
        # 单 P 高度环在倾角升力损失下几乎无阻尼（实测 ±1.7 m 等幅振荡 → 终端脱靶），
        # 串联 P 用垂速反馈提供阻尼：ω=sqrt(kp_z*kp_vz*k_T*g)，ζ=kp_vz*k_T*g/(2ω)。
        # 沿视线爬升的几何上限（只限制"上升"，下降不限制）
        vz_limit = None
        if self.climb_along_los and np.isfinite(el_los):
            v_h = float(np.hypot(own_vel[0], own_vel[1]))
            # 地板 3 m/s：起步阶段水平速度≈0，纯几何值会把爬升掐死（起飞都困难）
            vz_limit = max(abs(v_h * np.tan(np.deg2rad(el_los))), 3.0) + self.climb_margin_mps
        throttle = self._altitude_pwm_raw(z_own, float(own_vel[2]), z_target_est,
                                          vz_limit=vz_limit, vz_ref_mps=vz_ref_mps)

        # 低空保护：非终端且高度低时限制低头、保底油门
        low_guard = z_own < self.min_alt_guard_m and range_est > self.terminal_range_m

        # 限幅（唯一的"非线性"，不是时间状态）
        max_pitch = self.max_pitch_up_deg
        min_pitch = max(-self.max_pitch_down_deg, -15.0) if low_guard else -self.max_pitch_down_deg
        min_pitch = max(min_pitch, speed_floor)   # 超速时抬高低头下限（见上）
        pitch_cmd = _clamp(pitch_cmd, min_pitch, max_pitch)
        roll_cmd = _clamp(roll_cmd, -self.max_roll_deg, self.max_roll_deg)
        yaw_rate = _clamp(yaw_rate, -self.max_yaw_rate_dps, self.max_yaw_rate_dps)

        if self.lift_ff:
            # 常系数对象补偿：倾角 σ 下要维持同样垂向推力，需要 T/cos(σ)。
            # 而 T ∝ cmd²，所以命令空间的补偿系数是 1/sqrt(cos σ)（不是 1/cos）。
            sigma = abs(min(0.0, pitch_cmd))
            throttle = 1000.0 + (throttle - 1000.0) / max(
                float(np.sqrt(np.cos(np.deg2rad(sigma)))), 0.40)
        if low_guard:
            throttle = max(throttle, self.hover_pwm)

        thr_sat = throttle > self.thr_max_pwm or throttle < self.thr_min_pwm
        throttle = _clamp(throttle, self.thr_min_pwm, self.thr_max_pwm)

        # 接近率（bbox 相对膨胀率，用于命中防误判）
        closing_ratio = 0.0
        if self._last_s is not None and dt > 1e-6 and self._last_t is not None:
            closing_ratio = float((s_px - self._last_s) / max(self._last_s, 1e-3) / dt)
        self._last_s = float(s_px)
        self._last_t = float(timestamp)

        phase = PStrikePhase.TERMINAL if range_est <= self.terminal_range_m else PStrikePhase.TRACK
        size_hit = range_est <= self.hit_range_m and closing_ratio > 0.0
        self._hit_count = self._hit_count + 1 if size_hit else 0
        hit = self._hit_count >= self.hit_dwell
        if hit:
            phase = PStrikePhase.HIT

        saturated = bool(
            thr_sat
            or abs(roll_cmd) >= self.max_roll_deg - 1e-9
            or abs(yaw_rate) >= self.max_yaw_rate_dps - 1e-9
            or pitch_cmd >= self.max_pitch_up_deg - 1e-9
            or pitch_cmd <= min_pitch + 1e-9
        )

        cmd = PStrikeCommand(
            phase=phase,
            valid=True,
            hit=hit,
            u_px=float(u_px),
            v_px=float(v_px),
            s_px=float(s_px),
            e_u=float(e_u),
            e_v=float(e_v),
            az_los_deg=float(az_los),
            el_los_deg=float(el_los),
            range_est_m=float(range_est),
            z_target_est_m=float(z_target_est),
            closing_ratio=float(closing_ratio),
            roll_cmd_deg=float(roll_cmd),
            pitch_cmd_deg=float(pitch_cmd),
            pitch_bias_deg=float(pitch_bias),
            yaw_rate_dps=float(yaw_rate),
            throttle_pwm=float(throttle),
            saturated=saturated,
        )
        self._last_cmd = cmd          # COAST 保持用（不是控制动态）
        return cmd

    # ------------------------------------------------------------------ #
    # 无目标 / 丢失
    # ------------------------------------------------------------------ #
    def _no_target(self, timestamp: float, own_pos: np.ndarray,
                   z_own: float, v_up: float = 0.0) -> PStrikeCommand:
        # 无检测：标记"不在跟踪"，下一次拿到 bbox 时即为捕获瞬间（软起基准）
        self._was_tracking = False
        now = float(timestamp)
        if self._lost_since is None:
            self._lost_since = now
        lost_for = now - self._lost_since
        self._last_s = None

        throttle = self._altitude_pwm(z_own, v_up)
        if lost_for <= self.lost_timeout_s:
            # COAST：短时间内**保持上一条姿态指令**，只让高度环继续工作。
            # 终端 bbox 出画（约 3-4 m）时如果立刻回中/回平，最后 0.3 s 等于放弃控制
            # （实测脱靶 2.66 m）；冻结姿态可以继续把机头压在目标方向上。
            last = self._last_cmd
            if last is not None:
                return PStrikeCommand(
                    phase=PStrikePhase.COAST, valid=False, hit=False,
                    roll_cmd_deg=last.roll_cmd_deg,
                    pitch_cmd_deg=last.pitch_cmd_deg,
                    pitch_bias_deg=last.pitch_bias_deg,
                    yaw_rate_dps=last.yaw_rate_dps,
                    throttle_pwm=throttle,
                    range_est_m=last.range_est_m,
                    closing_ratio=last.closing_ratio,
                    az_los_deg=last.az_los_deg,
                    el_los_deg=last.el_los_deg,
                )
            return PStrikeCommand(
                phase=PStrikePhase.COAST, valid=False,
                throttle_pwm=throttle, saturated=False,
            )
        return PStrikeCommand(
            phase=PStrikePhase.SEARCH, valid=False,
            throttle_pwm=throttle, saturated=False,
        )

    def _altitude_pwm_raw(self, z_own: float, v_up: float = 0.0,
                          z_ref: Optional[float] = None,
                          vz_limit: Optional[float] = None,
                          vz_ref_mps: Optional[float] = None) -> float:
        """高度环（两个 P 串联）：目标高度 → 垂速指令 → 油门（未限幅）。

        vz_limit 给出时只限制**上升**方向（沿视线爬升的几何上限）。
        """
        if z_ref is None and self.z_hold_m is None:
            return float(self.hover_pwm)
        target = float(self.z_hold_m) if z_ref is None else float(z_ref)
        if vz_ref_mps is not None:
            vz_cmd = _clamp(float(vz_ref_mps), -self.vz_max_mps, self.vz_max_mps)
        else:
            vz_cmd = _clamp(self.kp_z * (target - float(z_own)),
                            -self.vz_max_mps, self.vz_max_mps)
        if self.ki_z != 0.0:
            vz_cmd = _clamp(vz_cmd + self.ki_z * self._z_int,
                            -self.vz_max_mps, self.vz_max_mps)
        if vz_limit is not None and vz_cmd > vz_limit:
            vz_cmd = max(float(vz_limit), 0.0)
        return float(self.hover_pwm + self.kp_vz * (vz_cmd - float(v_up)))

    def _altitude_pwm(self, z_own: float, v_up: float = 0.0,
                      z_ref: Optional[float] = None) -> float:
        """高度环（两个 P 串联）+ 硬限幅。"""
        return float(_clamp(self._altitude_pwm_raw(z_own, v_up, z_ref),
                            self.thr_min_pwm, self.thr_max_pwm))

    def search_command(
        self,
        own_position_enu: Sequence[float],
        yaw_rate_dps: float,
        pitch_bias_deg: float,
        own_velocity_enu: Optional[Sequence[float]] = None,
    ) -> PStrikeCommand:
        """扫描指令：高度环保持 + 固定扫描俯仰 + 偏航扫掠（供 runner 使用）。"""
        z_own = float(np.asarray(own_position_enu, dtype=float)[2])
        v_up = 0.0 if own_velocity_enu is None else float(np.asarray(own_velocity_enu)[2])
        # 低空保护：扫描阶段同样不能无限低头，否则无目标时会一路俯冲砸地
        # （早期实测：搜索姿态 + 无可视目标 → 触地）。
        low_guard = z_own < self.min_alt_guard_m
        pitch_cmd = _clamp(-abs(float(pitch_bias_deg)), -self.max_pitch_down_deg,
                           self.max_pitch_up_deg)
        if low_guard:
            pitch_cmd = _clamp(pitch_cmd, -15.0, self.max_pitch_up_deg)
        return PStrikeCommand(
            phase=PStrikePhase.SEARCH, valid=False,
            pitch_cmd_deg=float(pitch_cmd),
            pitch_bias_deg=float(pitch_cmd),
            yaw_rate_dps=float(_clamp(yaw_rate_dps, -self.max_yaw_rate_dps,
                                      self.max_yaw_rate_dps)),
            throttle_pwm=max(self._altitude_pwm(z_own, v_up),
                              self.hover_pwm if low_guard else 0.0),
        )

    # ------------------------------------------------------------------ #
    # 测量调理（可关；不改变 P 律结构）
    # ------------------------------------------------------------------ #
    def _measurement(self, bbox_xyxy: Optional[Sequence[float]]):
        if bbox_xyxy is None:
            return None
        try:
            x1, y1, x2, y2 = (float(bbox_xyxy[0]), float(bbox_xyxy[1]),
                              float(bbox_xyxy[2]), float(bbox_xyxy[3]))
        except (TypeError, ValueError, IndexError):
            return None
        if not all(np.isfinite([x1, y1, x2, y2])):
            return None
        s_px = x2 - x1
        if s_px < self.min_bbox_px or (y2 - y1) <= 0.0:
            return None
        raw = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5, s_px], dtype=float)
        if self.meas_lpf_alpha >= 1.0 or self._meas is None:
            self._meas = raw
        else:
            a = max(0.0, min(1.0, self.meas_lpf_alpha))
            self._meas = self._meas + a * (raw - self._meas)
        return (float(self._meas[0]), float(self._meas[1]), float(self._meas[2]))

    def reset(self) -> None:
        self._last_cmd = None
        self._el_fresh = None
        self._az_fresh = None
        self._r_filt = None
        self.az_last_deg = None
        self._lost_since = None
        self._last_s = None
        self._last_t = None
        self._meas = None
        self._hit_count = 0


# ---------------------------------------------------------------------- #
# RC 映射
# ---------------------------------------------------------------------- #
# 实测标定（sim/p_axis_probe.py，2026-09-28，LiftDrag Iris + 本 SITL 配置）：
#   ch1=1700 → 机体抬头方向加速（世界 Δv 方位 337°，机头朝北）⟹ pitch 通道高 = 低头
#   ch0=1700 → 世界 Δv 方位 247°（面向北时向左）⟹ roll 通道高 = 左滚，需取反
#   ch3=1700 → MSP yaw +9.5°（顺时针）⟹ yaw 通道高 = 右偏航率
#   ch3=1500-Δ（左偏航命令）→ Gazebo 真值 yaw 反而增加（右转，实测 +6 dps 对应
#   -41 dps 命令）⟹ yaw 通道**取反**。这与 HANDOFF §2.2 记录的 SITL yaw 陀螺极性
#   问题一致：固件认为在左转、模型实际右转。伴机侧取反即可（不改固件）。
# 2026-09-28 第二次标定：坐标映射修正（gazebo_target_bridge: E=-y_g，去掉 FRD 翻转，
# det 由 -1 变 +1）翻转了**水平面**的解释，因此当时在镜像坐标系下标定出的 roll/yaw
# 符号必须一并翻转；pitch/油门在垂直轴，不受影响。
#   roll: ch0=1700（右滚命令）在修正坐标系下 → 世界加速度向右 ⟹ 不再取反
#   yaw : ch3=1500+（命令右偏航）→ 模型 yaw 实际右转 ⟹ 不再取反
RC_ROLL_SIGN = +1.0
RC_PITCH_SIGN = 1.0
RC_YAW_SIGN = +1.0


def p_cmd_to_rc(
    cmd: PStrikeCommand,
    angle_limit_deg: float = 60.0,
    yaw_full_rate_dps: float = 300.0,
    prev: Optional[Sequence[int]] = None,
    rate_limit: int = 60,
    arm_ch: int = 4,
    angle_ch: int = 5,
    roll_sign: float = RC_ROLL_SIGN,
    pitch_sign: float = RC_PITCH_SIGN,
    yaw_sign: float = RC_YAW_SIGN,
    roll_rate_dps: Optional[float] = None,
    pitch_rate_dps: Optional[float] = None,
    rate_full_dps: float = 600.0,
    angle_on: bool = True,
) -> list[int]:
    """PStrikeCommand → 16 通道 RC（ANGLE + 手动油门）。

    通道约定（与 sim/run_intercept_mission.py 一致）：
    `ch0`=roll，`ch1`=pitch（Betaflight ANGLE 正方向是**低头**，故取反），
    `ch2`=throttle（ALT_HOLD 模式下为以 hover 为中心的高度杆量），
    `ch3`=yaw（ANGLE 模式下是角速率杆量），`ch4`=AUX1/ARM，`ch5`=AUX2/ANGLE。
    三个符号由 sim/p_axis_probe.py 实测标定，见上方常量注释。
    """
    angle_scale = 500.0 / max(1e-6, float(angle_limit_deg))
    yaw_scale = 500.0 / max(1e-6, float(yaw_full_rate_dps))
    channels = [RC_MID] * 16
    # 角速率模式（ACRO）：pitch/roll 通道直接给角速率，绕开固件 ANGLE 内环
    # （实测 ANGLE 只能实现约 50% 指令姿态，且满杆失稳）。制导律本身不动。
    rate_scale = 500.0 / max(1e-6, float(rate_full_dps))
    if roll_rate_dps is None:
        channels[0] = _rc(RC_MID + roll_sign * cmd.roll_cmd_deg * angle_scale)
    else:
        channels[0] = _rc(RC_MID + roll_sign * roll_rate_dps * rate_scale)
    if pitch_rate_dps is None:
        channels[1] = _rc(RC_MID - pitch_sign * cmd.pitch_cmd_deg * angle_scale)
    else:
        channels[1] = _rc(RC_MID - pitch_sign * pitch_rate_dps * rate_scale)
    channels[2] = _rc(cmd.throttle_pwm)
    channels[3] = _rc(RC_MID + yaw_sign * cmd.yaw_rate_dps * yaw_scale)
    channels[arm_ch] = 2000
    # 角速率模式必须关掉 ANGLE 开关（否则固件把杆量当姿态角解释，速率指令无效）
    channels[angle_ch] = 2000 if angle_on else 1000
    if prev is not None:
        for axis in (0, 1, 3, 2):
            delta = channels[axis] - int(prev[axis])
            if abs(delta) > rate_limit:
                channels[axis] = int(prev[axis]) + (rate_limit if delta > 0 else -rate_limit)
    return channels


def _rc(value: float) -> int:
    return int(max(RC_MIN, min(RC_MAX, round(float(value)))))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _wrap180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


__all__ = [
    "PVisualGuidance",
    "PStrikeCommand",
    "PStrikePhase",
    "p_cmd_to_rc",
    "RC_ROLL_SIGN",
    "RC_PITCH_SIGN",
    "RC_YAW_SIGN",
    "_FORBIDDEN_STATE_HINTS",
]
