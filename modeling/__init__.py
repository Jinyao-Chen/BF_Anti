"""理论建模与 PID 整定工具链（替代 Simulink 的最小闭环实现）。

模块分层：

* :mod:`modeling.quadrotor`  —— 非线性 6-DOF 模型 + 悬停线性化
* :mod:`modeling.pid`        —— 带抗积分饱和/微分滤波的离散 PID
* :mod:`modeling.tune`       —— 级联回路设计、频域裕度、步响应评估与 CLI

该包只依赖 NumPy/SciPy，不依赖 MATLAB/Simulink，也不修改 Gazebo/SITL
进程。调参结果写入 JSON，供 SITL 配置和 BF_Anti 制导层读取。
"""

from .quadrotor import QuadrotorModel, QuadrotorParams
from .pid import PID, PIDGains
from .tune import CascadeGains, design_cascade, evaluate_linear

__all__ = [
    "QuadrotorModel",
    "QuadrotorParams",
    "PID",
    "PIDGains",
    "CascadeGains",
    "design_cascade",
    "evaluate_linear",
]
