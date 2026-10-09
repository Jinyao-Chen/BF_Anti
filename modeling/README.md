# 理论建模、PID 整定与掉高诊断

这个目录用纯 Python 复现四旋翼悬停线性化、固件等效高度 PID、水平机动-姿态-
混控-高度耦合模型，并给出可写回 Betaflight 的 CLI 初值。它不依赖
MATLAB/Simulink，也不在运行时修改 Gazebo。

## 目录分层

| 文件 | 作用 |
|---|---|
| `firmware_params.py` | **唯一可信来源**：固件 `ALTITUDE_*_SCALE`、offboard 缩放、最终 SITL 配置 |
| `quadrotor.py` | 17 状态非线性 6-DOF + 12 状态悬停线性化 |
| `pid.py` | 离散 PID（抗饱和、微分滤波） |
| `tune.py` | 归一化级联带宽设计；**不直接输出 `ap_altitude_*`** |
| `altitude_loop.py` | **固件等效**高度 PID 一维模型 + 自动整定 + CLI 直写值 |
| `coupled_altitude.py` | 水平位置环→倾角→姿态→混控→高度环耦合模型 |
| `diagnose_altitude.py` | 从 SITL 日志识别滞后/振铃/混控饱和，并给出模型残差 |
| `identify_lift_loss.py` | 从稳态水平段辨识 LiftDrag 速度²升力损失系数 `k_v` |

## 快速使用

```bash
cd ~/BF_Anti

# 1) 固件等效高度 PID：使用 SITL 已辨识的 0.28 s 执行器 / 0.6 s 高度估计滞后
python3 -m modeling.altitude_loop --sitl

# 2) 当前固件与历史掉高模型对比
python3 -m modeling.coupled_altitude --all --duration 20

# 3) 从真实 SITL 日志诊断；--settle 让水平腿从稳定悬停开始
python3 sim/run_altitude_diag.py --cruise 5 --settle 5 --log /tmp/altitude_diag.csv
python3 -m modeling.diagnose_altitude --log /tmp/altitude_diag.csv

# 4) 辨识 LiftDrag 速度²升力损失系数 k_v
python3 -m modeling.identify_lift_loss --log /tmp/altitude_diag.csv --min-speed 4

# 5) 全部回归
python3 -m pytest -q
```

## 固件高度环的真实结构

`src/main/flight/autopilot_multirotor.c` 中不是“位置 P → 期望垂速 → 速度
PI”的串级，而是：

```c
altitudeP = altitudeErrorCm * altitudeKp;
altitudeI += altitudeErrorCm * altitudeKi * itermRelax * dt;
velocityError = targetVerticalVelocity - verticalVelocity;
altitudeD = velocityError * altitudeKd * dBoost;
altitudeF = targetVerticalVelocity * altitudeKf;
throttleOffset = altitudeP + altitudeI + altitudeD + altitudeF + tiltHover;
```

因此：

- P/I 直接作用于**高度误差**；
- D 作用于**垂速误差**；
- F 作用于**目标垂速**；
- 倾角补偿只作用于 `tiltHover`，不是乘整个 PID 输出。

## CLI 缩放（以 `autopilot_multirotor.c` 为准）

| 固件内部 | 来源常量 | CLI |
|---|---:|---:|
| `altitudeKp = ap_altitude_p * 0.005` | `ALTITUDE_P_SCALE` | `ap_altitude_p = altitudeKp / 0.005` |
| `altitudeKi = ap_altitude_i * 0.002` | `ALTITUDE_I_SCALE` | `ap_altitude_i = altitudeKi / 0.002` |
| `altitudeKd = ap_altitude_d * 0.01` | `ALTITUDE_D_SCALE` | `ap_altitude_d = altitudeKd / 0.01` |
| `altitudeKf = ap_altitude_f * (0.1/30)` | `ALTITUDE_F_SCALE` | `ap_altitude_f = altitudeKf / (0.1/30)` |

当前 release260904 已验证的最终值为：

```text
set ap_hover_throttle = 1530
set ap_altitude_p = 50     # altitudeKp = 0.25 PWM/cm
set ap_altitude_i = 50     # altitudeKi = 0.10 PWM/(cm*s)
set ap_altitude_d = 20     # altitudeKd = 0.20 PWM/(cm/s)
set ap_altitude_f = 0
set ap_offboard_vel_p = 200
set ap_offboard_vel_d = 300
set ap_offboard_accel_ff = 100
set ap_max_angle = 15
```

`modeling.tune` 的 `altitude.kp/ki/kd` 是归一化串级设计量（1/s、1/s²、1/s），
**不能**再乘 100/10 直接写 CLI。要写固件请用：

```bash
python3 -m modeling.altitude_loop --sitl
```

## 水平机动掉高模型

`coupled_altitude.py` 用 `mixer_mode` 区分两代固件：

- `legacy`：默认 Betaflight 混控在差动饱和时压低总距；旧固件倾角补偿是
  `1/cos²`。`ring_mixer` / `full` 保留历史拟合参数，用于复现第 13 节日志。
- `thrust_priority`：当前 `mixer.c` 在 `ALT_HOLD/POS_HOLD/GPS_RESCUE` 下
  优先保总距，只按需缩差动；`altitudeControl()` 使用精确的
  `1/(cos(roll)cos(pitch))`，offboard 水平映射使用 `atan(a/g)`，导航加减速
  上限 2.0 m/s²。`current` 场景对应当前固件。

典型结果（`--duration 20`，目标 10 m）：

| 场景 | 最低高度 | 说明 |
|---|---:|---|
| `ideal` | 9.83 m | 理想姿态/估计；有限带宽 PID 的小瞬态 |
| `gravity_only` | 9.13 m | 关闭倾角补偿；仍远大于历史 full 的掉高 |
| `ring_mixer` | 4.85 m | 历史：姿态振铃 + 旧混控压总距 |
| `full` | 1.53 m | 历史：再加 1.13 s 高度估计滞后 |
| `current` | 9.48 m | 当前：总距优先 + 精确倾角补偿 + 分离的高度/垂速滞后 |
| `current_liftdrag` | 8.04 m | 当前控制 + 已辨识 LiftDrag 损失（无前馈） |
| `current_ff` | 9.87 m | 当前控制 + LiftDrag 损失 + 速度²前馈 + 水平推力耦合 |

结论：

1. 15° 倾角纯重力损失只有约 0.33 m/s²，不是米级掉高的主因。
2. 历史大幅掉高的主因是**姿态振铃 + 旧混控差动饱和压低总距**，高度估计
   滞后进一步放大。
3. 当前总距优先混控 + 精确倾角补偿后，控制链路本身只解释约 0.5 m 的
   瞬态下沉；SITL 中残余的 1~3 m 掉高主要来自 Gazebo `LiftDrag` 旋翼
   模型的前飞升力损失，属于 plant 气动特性，不应再用 `1/cos²` 去掩盖。
4. `current_liftdrag` 与 `current_ff` 的差值就是速度²前馈 + 水平推力耦合
   可以恢复的高度量级。

LiftDrag 损失用 `plant_lift_loss_accel_gain`（额外向下加速度，单位
m/s² per (m/s)²）表示，由 `identify_lift_loss` 从 SITL 日志辨识；控制器侧
对应固件参数 `ap_altitude_lift_ff`（缩放 1000，5 = 0.005 m/s² per
(m/s)²，默认 0）。真实硬件默认关闭，SITL 才写入辨识值。

## 日志诊断

`diagnose_altitude.py` 识别：

- 高度估计等效一阶滞后 `tau_s`；
- 姿态内环振铃主频 `ring_hz`；
- 最大合成倾角（`cosθ=cos(roll)*cos(pitch)`）与纯重力损失 `g(1-cosθ)`；
- 真值最小垂速与**最小垂向加速度**；
- 电机差动饱和占比；
- 当前控制模型预测与实测的 `unmodeled_downward_accel_mps2`。
- `truth_altitude_drop_m`（首样本到最低点）、`altitude_estimation_rmse_m`、
  实际采样间隔/频率、有效样本数与丢弃的非有限样本数。

采样周期必须从 CSV 的 `t` 列核实；采集脚本的 `--period` 只是循环等待
下限，顺序 MSP 查询等开销可能令实际频率低于请求值。用于频域分析的
`ring_hz` 仅是 0.8–3 Hz 波段内的最大谱峰，不能单凭该数值断言存在振铃。

对仓库已有 `sim/results/merge_upstream_2026-09/alt_*.csv` 四份日志重新分析，
中位采样频率均约 **8.1 Hz**（非 20 Hz）；三份修复后日志的
`truth_altitude_drop_m` 为 **0.086–0.093 m**，旧 `alt_merge4_2ms.csv`
为 **3.944 m**。这些只是原工况首样本到最低点的统计，不能外推到更高速度、
任意初始高度或视觉闭环运行。日志中的 `gravity_fraction_of_peak_downward_accel`
是两个不同时刻的峰值比，**不是**掉高成因的严格比例。

> 修正记录：旧版脚本用真值高度的**一阶导数**当垂向加速度，实际那是垂速。
> 现在先平滑求垂速，再平滑求加速度，因此第 13 节的“-3.1 m/s² 重力占比
> 9.5%”结论需要用新日志重新确认；旧数值不再作为证据引用。

本轮 5 m/s / 15° SITL 冒烟实测：

先用 `identify_lift_loss` 在 `/tmp/altitude_diag_fixed.csv` 上辨识：

```text
k_v = 0.00532 m/s² per (m/s)²
n = 9, R² = 0.57
```

因此 SITL 配置写入 `set ap_altitude_lift_ff = 5`（0.005）。

随后用 `--settle 5` 做前馈前后对比（水平腿从稳定悬停高度开始，
所以比较的是“从腿起始高度的掉高”）：

| 测试 | 腿起始高度 | 真值最低高度 | 掉高 | 真值最小垂向加速度 |
|---|---:|---:|---:|---:|
| `ap_altitude_lift_ff = 0` | 11.44 m | 9.63 m | 1.81 m | -5.04 m/s² |
| `ap_altitude_lift_ff = 5` | 10.83 m | 9.78 m | 1.05 m | -1.83 m/s² |

结论：

- 前馈 + 水平推力耦合把单次 SITL 冒烟中的掉高从 1.81 m 压到 1.05 m，
  最恶劣向下加速度从 -5.04 m/s² 压到 -1.83 m/s²；
- 差动饱和两组均为 0，说明总距优先混控稳定；
- SITL 单次运行仍有较大离散性，需后续做 5/10 m/s 多次重复和更长稳态段
  再定最终 `k_v`；
- 真实硬件默认 `ap_altitude_lift_ff = 0`，该前馈只针对当前 LiftDrag world。

## 与参考 Simulink 工程的关系

原文模型整体正确，但存在小角度线性化和符号书写问题。详细推导见：

- `docs/interception/2026-09-18-theory-model-and-progress.md`（本地，未纳入版本控制）
- `docs/interception/2026-09-19-simulink-model-review-corrected.md`（同上）

核心修正：

- 高度通道：\(T_d = m(g-\dot v_{zd})/(\cos\phi\cos\theta)\)；
- 固件 `getCosTiltAngle()` 返回 `rMat.m[2][2]`，已经是 \(\cos\phi\cos\theta\)，
  所以用 `1.0f / getCosTiltAngle()`；
- 水平通道：速度环输出解释为 \(a/g\)，用 `atan(a/g)` 映射到大倾角；
- LiftDrag 前飞升力损失是 plant 问题，应通过 `MulticopterMotorModel` /
  力模型 / 数据辨识前馈解决，而不是控制器硬补 `1/cos²`。

## 多次 SITL 统计与鲁棒辨识（2026-09-19）

自动化脚本：

```bash
python3 sim/run_repeat_stats.py \
  --cruise 5 10 --ff 0 5 --repeat 3 \
  --duration 20 --settle 5 --beyond 30 --logdir /tmp/bf_anti_stats
```

12 次独立 SITL 运行，真值高度用 0.6 s Savgol 重新处理：

| 组合 | 掉高均值 ± std | 真值最小垂向加速度均值 ± std |
|---|---:|---:|
| 5 m/s, FF=0 | 1.495 ± 0.034 m | -5.63 ± 1.79 m/s² |
| 5 m/s, FF=5 | 0.975 ± 0.309 m | -3.51 ± 1.59 m/s² |
| 10 m/s, FF=0 | 0.255 ± 0.132 m | -2.33 ± 0.68 m/s² |
| 10 m/s, FF=5 | 0.397 ± 0.306 m | -2.35 ± 0.39 m/s² |

所有 12 次的差动饱和均为 0%。

残差 Huber 辨识（6 条 FF=0 日志）：

```text
k_v = 0.00360 m/s² per (m/s)²
95% CI = [0.00140, 0.00537]
n = 380, n_logs = 6
R² ≈ 0.001
```

结论：

- 5 m/s 下前馈 + 水平耦合显著减小掉高和最恶劣垂向加速度；
- 10 m/s 下差异小于一个标准差，暂未证明收益；
- R² 很低说明 offboard 速度振荡仍然污染数据，需要定速平台实验；
- 当前 SITL 保留 `ap_altitude_lift_ff=5`（在 95% CI 内）。
