"""Betaflight release260904 高度/offboard 回路的权威参数比例。

这些常量必须与 ``~/betaflight/src/main/flight/autopilot_multirotor.c`` 中的
``ALTITUDE_*_SCALE``、``POSHOLD_TASK_RATE_HZ`` 以及
``src/main/pg/autopilot.h`` 中的 ``offboard*`` 注释一致。所有理论模型、日志
诊断和 README 都从这里取值，避免出现“模型一套、固件一套”的隐性偏差。

注意：``config/theory_pid.json`` 里的 ``altitude.kp/ki/kd`` 是归一化串级
设计量，不是 PWM 工程量；需要直接映射到 ``ap_altitude_*`` 时，请使用
:mod:`modeling.altitude_loop`（固件等效 PID 结构）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

# --------------------------------------------------------------------------- #
# autopilot_multirotor.c：高度环内部缩放
# --------------------------------------------------------------------------- #
ALTITUDE_P_SCALE = 0.005          # altitudeKp = ap_altitude_p * 0.005 [PWM/cm]
ALTITUDE_I_SCALE = 0.002          # altitudeKi = ap_altitude_i * 0.002 [PWM/(cm*s)]
ALTITUDE_D_SCALE = 0.01           # altitudeKd = ap_altitude_d * 0.01  [PWM/(cm/s)]
ALTITUDE_F_KF_REF = 30.0
ALTITUDE_F_SCALE = 0.1 / ALTITUDE_F_KF_REF
ALTITUDE_I_LIMIT = 150.0          # [PWM]
ALTITUDE_VEL_CMD_MAX_DEFAULT_CM_S = 1500.0
POSHOLD_TASK_RATE_HZ = 100.0
ALTHOLD_TASK_RATE_HZ = 100.0
PWM_RANGE_MIN = 1000.0
PWM_RANGE_MAX = 2000.0

# --------------------------------------------------------------------------- #
# pg/autopilot.h：offboard 速度环参数缩放
# --------------------------------------------------------------------------- #
OFFBOARD_VEL_P_SCALE = 0.0001     # 200 -> 0.02 deg/(cm/s) = 2 deg per m/s
OFFBOARD_VEL_D_SCALE = 0.0001     # 300 -> 0.03 deg/(cm/s) = 3 deg per m/s
OFFBOARD_ACCEL_FF_SCALE = 0.0001  # 100 -> 0.01 deg/(cm/s^2) = 1 deg per m/s^2

# position_offboard.c：release260904 最终物理加减速上限
OFFBOARD_MAX_ACCEL_MPS2 = 2.0
OFFBOARD_MAX_DECEL_MPS2 = 2.0


def altitude_kp_from_cli(ap_altitude_p: float) -> float:
    """``ap_altitude_p`` -> 固件内部 ``altitudeKp`` [PWM/cm]。"""
    return float(ap_altitude_p) * ALTITUDE_P_SCALE


def altitude_ki_from_cli(ap_altitude_i: float) -> float:
    """``ap_altitude_i`` -> 固件内部 ``altitudeKi`` [PWM/(cm*s)]。"""
    return float(ap_altitude_i) * ALTITUDE_I_SCALE


def altitude_kd_from_cli(ap_altitude_d: float) -> float:
    """``ap_altitude_d`` -> 固件内部 ``altitudeKd`` [PWM/(cm/s)]。"""
    return float(ap_altitude_d) * ALTITUDE_D_SCALE


def altitude_kf_from_cli(ap_altitude_f: float) -> float:
    """``ap_altitude_f`` -> 固件内部 ``altitudeKf`` [PWM/(cm/s)]。"""
    return float(ap_altitude_f) * ALTITUDE_F_SCALE


def cli_from_altitude_kp(kp: float) -> float:
    return float(kp) / ALTITUDE_P_SCALE


def cli_from_altitude_ki(ki: float) -> float:
    return float(ki) / ALTITUDE_I_SCALE


def cli_from_altitude_kd(kd: float) -> float:
    return float(kd) / ALTITUDE_D_SCALE


def cli_from_altitude_kf(kf: float) -> float:
    return float(kf) / ALTITUDE_F_SCALE


@dataclass(frozen=True)
class FirmwareAltitudeGains:
    """固件内部高度环增益（PWM 工程量）。"""

    kp_pwm_per_cm: float
    ki_pwm_per_cm_s: float
    kd_pwm_per_cm_s: float
    kf_pwm_per_cm_s: float = 0.0

    @classmethod
    def from_cli(
        cls,
        ap_altitude_p: float,
        ap_altitude_i: float,
        ap_altitude_d: float,
        ap_altitude_f: float = 0.0,
    ) -> "FirmwareAltitudeGains":
        return cls(
            kp_pwm_per_cm=altitude_kp_from_cli(ap_altitude_p),
            ki_pwm_per_cm_s=altitude_ki_from_cli(ap_altitude_i),
            kd_pwm_per_cm_s=altitude_kd_from_cli(ap_altitude_d),
            kf_pwm_per_cm_s=altitude_kf_from_cli(ap_altitude_f),
        )

    def to_cli(self) -> Dict[str, float]:
        return {
            "ap_altitude_p": round(cli_from_altitude_kp(self.kp_pwm_per_cm), 4),
            "ap_altitude_i": round(cli_from_altitude_ki(self.ki_pwm_per_cm_s), 4),
            "ap_altitude_d": round(cli_from_altitude_kd(self.kd_pwm_per_cm_s), 4),
            "ap_altitude_f": round(cli_from_altitude_kf(self.kf_pwm_per_cm_s), 4),
        }


# --------------------------------------------------------------------------- #
# sitl_gazebo_config.txt：当前 release260904 最终配置
# --------------------------------------------------------------------------- #
SITL_FINAL_ALTITUDE_CLI = {
    "ap_hover_throttle": 1530.0,
    "ap_altitude_p": 50.0,
    "ap_altitude_i": 50.0,
    "ap_altitude_d": 20.0,
    "ap_altitude_f": 0.0,
    "ap_offboard_vel_p": 200.0,
    "ap_offboard_vel_d": 300.0,
    "ap_offboard_accel_ff": 100.0,
    "ap_max_angle": 15.0,
}

SITL_FINAL_GAINS = FirmwareAltitudeGains.from_cli(
    ap_altitude_p=SITL_FINAL_ALTITUDE_CLI["ap_altitude_p"],
    ap_altitude_i=SITL_FINAL_ALTITUDE_CLI["ap_altitude_i"],
    ap_altitude_d=SITL_FINAL_ALTITUDE_CLI["ap_altitude_d"],
    ap_altitude_f=SITL_FINAL_ALTITUDE_CLI["ap_altitude_f"],
)

# 已验证的垂直估计/执行器等效滞后（第 15/16 节，单位 s）。
SITL_ESTIMATOR_TAU_S = 0.6
# 垂速 D 反馈走 position.c 的 altitude_d_lpf（默认 300 -> 3 Hz），
# 等效一阶滞后约 1/(2*pi*3) ~= 0.053 s，不能和高度估计的 0.6 s 混为一谈。
SITL_VZ_ESTIMATOR_TAU_S = 0.05
SITL_ACTUATOR_TAU_S = 0.28
# 当前 release260904 在 LiftDrag world、5 m/s、15° 机动下的实测主振铃频率。
SITL_RING_HZ = 0.93

assert abs(altitude_kp_from_cli(50.0) - 0.25) < 1e-12
assert abs(altitude_ki_from_cli(50.0) - 0.10) < 1e-12
assert abs(altitude_kd_from_cli(20.0) - 0.20) < 1e-12
assert abs(altitude_kf_from_cli(30.0) - 0.10) < 1e-12
