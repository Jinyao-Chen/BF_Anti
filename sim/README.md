# sim/ — Gazebo SITL 仿真工作区

> 纯 P 视觉制导反无人机打击的仿真链路。真实机载链路见顶层 [README.md](../README.md)，
> 设计与进展文档在本地 `../docs/`（**未纳入版本控制**）。

---

## 0. 一分钟上手（主线：纯 P 视觉制导打击）

```bash
# 前置：SITL 已编译、检测后端可达（见 §6）

# a) 无头单发试验（自动重启仿真 -> 跑一发 -> 打印最近真值距离）
YOLO_PORT=9300 ./sim/trial_p_strike.sh a_pt            # 我们的 YOLOv8-L
YOLO_PORT=9301 ./sim/trial_p_strike.sh b_onnx --target-width 2.38   # 第三方的 YOLOv8n

# b) 实时窗口演示（相机画面 + 检测框 + HUD；GUI=1 同时开 Gazebo 3D）
YOLO=1 YOLO_BACKEND=pt GUI=1 ./sim/run_p_strike_live.sh

# c) 只验证制导-控制链路（不接检测，真值投影合成 bbox）
python3 sim/run_p_vision_strike.py --synthetic --duration 30 --log /tmp/kin.csv
```

**判据**：`--hit-radius 0.8`，最新一轮 15/15 命中，最近真值距离 0.163~0.441 m。

---

## 1. 入口脚本（日常就用这些）

| 文件 | 作用 | 备注 |
|---|---|---|
| `run_p_vision_strike.py` | **主线**：纯 P 视觉制导打击闭环（相机 → 检测 → 跟踪 → P 律 → RC/MSP） | 参数最多，`--help` 有全量说明；`--synthetic` 走真值投影 |
| `run_p_strike_live.sh` | 一键实时演示：起仿真 + 探活检测后端 + 开窗口/录像 | `YOLO=1` 用真检测；`YOLO_BACKEND=pt\|onnx` 选后端；`GUI=1` 开 3D |
| `trial_p_strike.sh` | 无头单发试验 + 统计（`sim/trial_<名>.out`/`.csv`） | `YOLO_PORT`/`DUMP_DIR`/`DUMP_HZ` 可注入；后端不通会 ABORT 而不是产生假脱靶 |
| `launch_sitl.sh` | 启动 Betaflight SITL + Gazebo | `BF_ANTI_WORLD` 选世界，`--headless` 无界面 |
| `sim_restart.sh` | 安全重启仿真（避免 `pkill` 命中调用者自身） | 每次试验前必跑 |
| `yolo_server_board.py` | RDK X5 板端检测服务（协议与 GPU 版一致） | 部署见本地文档 `docs/interception/2026-10-01-x5-as-inference-server.md`（未纳入版本控制） |
| `eval_detector_frames.py` | **换检测模型必跑**：在落盘的原始帧上对比两个模型的召回/置信度/框宽 | 输出隐含 `W_eff` -> 决定 `--target-width` |

早期阶段的运行器仍在（`run_visual_strike*.py`、`run_intercept_mission.py`、`run_xy_velocity_test.py`、
`run_force_offboard_test.py` 等），**P 版会 import 其中一部分工具**，不要删。

---

## 2. 库模块（被入口 import）

| 文件 | 职责 |
|---|---|
| `bbox_tracker.py` | 检测框跟踪器：确认帧数、跳框门限、漏检外推（coast），对外暴露 `TrackOut.fresh` 区分"本帧检测"与"外推" |
| `gazebo_camera_source.py` | 从 Gazebo 相机 topic 取帧（lock_step 下与物理同步） |
| `gazebo_target_bridge.py` | 真值位姿源 + 真值投影成 bbox（合成模式与可视化用） |
| `guidance_kinematic.py` / `p_strike_kinematic.py` / `visual_strike_kinematic.py` | 无 Gazebo 的运动学闭环仿真（验证控制律包线） |
| `chase_camera.py` | 跟拍相机（录 3D 演示用） |
| `telemetry_monitor.py` | 订阅 Gazebo 遥测并打印/落盘 |
| `rotor_force_bridge.py` | 旋翼力模型标定桥接 |

制导律本体在 `../guidance/`：**`p_visual_guidance.py` 是当前主线**（纯 P 契约见文档）。

---

## 3. 分析 / 标定工具（一次性，随时可改）

| 文件 | 用途 |
|---|---|
| `probe_camera_rate.py` | 相机投递率/时延（曾确认相机 25~30 Hz、不是陈旧帧） |
| `probe_axis_now.py` / `p_axis_probe.py` / `probe_position_sign.py` | RC 通道 → 机体系/世界系 极性与增益实测 |
| `rc_axis_response_test.py` | RC → 世界加速度标定 |
| `probe_estimator_lag.py` | 状态估计滞后 |
| `diag_run.py` | 一次打击的完整诊断（真值最近距离、丢检间隙、跟踪原因直方图、分段震荡） |
| `diag_camera_motion.py` / `diag_truth_attitude.py` | 相机运动/真值姿态对照 |
| `calibrate_rotor_force.py` / `compute_hover_throttle.py` | 悬停油门与旋翼力标定 |
| `plot_*.py` | 轨迹/3D/对比图（`plot_trajectory.py` 被 `run_p_vision_strike.py --plot` 直接 import） |
| `summarize_speed_sweep.py` / `run_repeat_stats.py` | 批量试验汇总统计 |
| `tools/legacy_xy_probes/` | **归档**：XY 速度外环阶段的一次性探针（9 个），仅作历史参考，未做风格清理（见 `tools/README.md`） |

---

## 4. 世界文件 `worlds/`

| 世界 | 用途 |
|---|---|
| `bf_anti_p_strike_light_cam.sdf` | **当前主线**：轻量机体 + 机载相机 + 靶机 |
| `bf_anti_p_strike_light{,_close,_nolock,_high}.sdf` | 轻量机体的距离/锁定变体，用于扫包线 |
| `bf_anti_p_strike{,_flat,_moving}.sdf` | 早期 P 打击世界（重机体/平地/移动靶） |
| `bf_anti_vision_strike.sdf`、`bf_anti_intercept*.sdf`、`bf_anti_p_strike_chase*.sdf` | 历史阶段世界（纯视觉打击、D 阶段拦截、跟拍） |
| `rotor_force_*.sdf` | 旋翼力标定专用世界 |

机体/靶机模型在 `models/`：`light_quad`（主线，相机 `cam_link` 固定在 `base_link`）、`x500_target`。

---

## 5. 结果与数据 `results/`

```
sim/results/
  p_strike/           # 主线：打击试验 CSV/PNG、detector_ab_*.{csv,json}、两份分析报告
  xy_velocity/        # XY 外环标定数据
  merge_upstream_2026-09/   # 固件合并期的高度环数据
```

约定：**报告（`.md`）跟数据放一起**（`sim/results/**/*.md`，与 `docs/` 一样**本地保留、不提交**）。
内部/交流材料（交接、周报、审计、汇报 PPT）**不在本仓库**，见 `../BF_Anti_materials/internal/`。
踩坑提示：`trial_p_strike.sh` 的日志在 `/tmp/trial_<名>.{out,csv}`，**同名会覆盖上一发**，批量跑要换名。

---

## 6. 依赖与外部服务

| 依赖 | 说明 |
|---|---|
| SITL 固件 | `~/betaflight`（或 `/tmp/bf_merge`），产物 `betaflight_SITL_SITL_GAZEBO.elf`；`BF_ROOT` 指过去 |
| Gazebo | Harmonic 8.9.0 + `BetaflightPlugin`（固件仓库内） |
| MSP | `127.0.0.1:5761`，试验脚本会先探活 |
| 检测服务 | TCP `4字节长度 + JPEG` → `{"dets":[{cls,name,conf,xyxy}]}`。GPU 版在 `REMOTE_WS/yolo_server.py`，用 `start_yolo.sh {pt\|onnx}` 起，**只能绑 `/dev/nvidia0`**（见下） |
| SSH 隧道 | `ssh -N -L 9300:127.0.0.1:9300 -L 9301:127.0.0.1:9301 GPU_USER@GPU_HOST &`。长连接会掉，跑之前先探 |
| 客户端类别白名单 | `run_visual_strike_yolo.choose_bbox` 只认 `{"copter_drone","motor_drone"}`；别的模型必须在服务端改名 |

---

## 7. 踩坑清单（都是实测踩过的）

1. **`/dev/nvidia0` ≠ nvidia-smi index 0**。前者是 `0000:41:00.0`（index 1 = `GPU-40c96051-...`），index 0 是其他用户的卡。按 UUID 绑。
2. **ultralytics 会覆写 `CUDA_VISIBLE_DEVICES`**：`select_device()` 里 `os.environ[...] = device`，所以 `device=0` 会把 UUID 覆盖成 `"0"`。要传 nvidia-smi index。
3. **换检测模型必须重标 `--target-width`**：`r = fx·W/s_px`。ONNX 的框比 .pt 宽 18% → 2.0 要改 2.38，否则高度环被带偏。
4. **类别名不同步 = 静默失效**：检测全被丢掉，表现是"全程 SEARCH"，看起来像制导坏了。
5. **别用 `pkill -f "gz sim"` 直接杀**：会命中调用者自己的 cmdline 把自己杀掉。用 `sim_restart.sh` 或括号写法 `[g]z sim`。
6. **检测间隙**：>40 m 远处小模型会漏（ONNX 该段召回 0.82~0.95），靠 `--lost-frames/--coast-frames` 撑；间隙超过 8 个控制帧才会真丢锁。

