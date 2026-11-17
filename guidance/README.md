# guidance/ —— 制导与控制律

机载链路与仿真链路**共用**这一层。全部是纯计算模块（不碰串口 / 不碰 Gazebo），因此可直接单测。

| 文件 | 状态 | 职责 |
|---|---|---|
| `p_visual_guidance.py` | **当前主线** | 纯 P + 比例导引的视觉制导律：图像误差 -> 滚转/俯仰/偏航/油门（RC ANGLE），含量测调理、软饱和、终端门控、超速治理、垂向串级 P |
| `track_cmd_adapter.py` | 在用 | 把制导输出适配成 `rc_mux.TrackCmd`（机载链路复用 RC 融合层） |
| `xy_velocity_controller.py` | 在用 | 伴机 XY 速度外环（固件 `ap_offboard_xy_enable` 配合）；设计与标定见本地文档 `docs/archive/2026-09-10-sim-design/2026-09-26-companion-xy-velocity-loop.md`（未纳入版本控制） |
| `intercept_guidance.py` | 历史 | 早期 D 阶段拦截制导 |
| `visual_strike_guidance.py` | 历史 | P 律之前的纯视觉打击控制律 |

## 入口与文档

- 控制律公式与推导：本地 8 页讲稿 `docs/reports/beamer/guidance_algo.pdf`（源码同目录 `.tex`）—— **未纳入版本控制**
- 控制律现状与实测：本地 `docs/simulation/2026-10-09-sim-stage-closeout.md` —— **未纳入版本控制**
- 仿真侧调用：`sim/run_p_vision_strike.py`；试验台：`sim/trial_p_strike.sh`、`sim/run_p_strike_live.sh`

## 设计约束（有测试把关）

`tests/test_p_visual_guidance.py` 里的 `_FORBIDDEN_STATE_HINTS` 会**审计属性名**，禁止引入
积分/微分环节与跨帧控制状态：所有非线性记忆less（软饱和 `w*tanh(e/w)`），滤波只允许出现在**量测**侧。
改控制律前先跑该测试文件。
