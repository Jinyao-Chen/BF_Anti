# BF_Anti — 反无人机视觉制导 + Betaflight MSP 飞控

[![Pi 5](https://img.shields.io/badge/Raspberry%20Pi-5%204GB-C51A4A)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Hailo](https://img.shields.io/badge/AI%20HAT%2B-Hailo--8L%2013TOPS-00A6FF)](https://hailo.ai/)
[![Betaflight](https://img.shields.io/badge/Betaflight-release260904%20%2B%20SITL-00A6FF)](https://github.com/betaflight/betaflight)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-FFB300)](https://github.com/ultralytics/ultralytics)

> **当前主线：Raspberry Pi 5 4GB + AI HAT+ Hailo-8L (13 TOPS)**。
> USB 相机 → Hailo YOLOv8 → 跟踪/纯 P 制导 → MSP 覆盖 RPYT → Betaflight；
> J7 复合视频 (720x576i PAL) 直接进 VTX/眼镜，不走 HDMI 转换器。
> 旧的 RDK X5 + Pico 链路保留在仓库中作参考，见 §旧版 RDK X5 链路。

仓库里有三条链路，别混淆：

| 链路 | 跑在哪 | 入口 | 说明 |
|------|--------|------|------|
| **Pi5 机载（主线）** | Pi 5 + Hailo-8L + Betaflight FC | [`onboard/app.py`](onboard/app.py) | 相机/检测/制导/OSD/图传；相机已接、FC 未接；DFC 已本地安装，见 §Pi5 机载（sim2real） |
| **SITL 仿真** | PC + Gazebo Harmonic + Betaflight SITL | [`sim/run_p_vision_strike.py`](sim/run_p_vision_strike.py) | 纯 P 视觉制导打击，见 §纯 P 视觉制导打击 与 [`sim/README.md`](sim/README.md) |
| **旧版 RDK X5** | RDK X5 + Pico | [`detect.py`](legacy/rdk_x5_pico/detect.py) / [`run_detect.sh`](legacy/rdk_x5_pico/run_detect.sh) | BPU .bin 推理 + SPI PAL；已归档到 legacy/rdk_x5_pico/ |

---

## Pi5 机载（sim2real）

### 目标硬件

| 组件 | 当前选择 | 状态 |
|---|---|---|
| 机载电脑 | Raspberry Pi 5 Model B 4GB | 已实测 |
| NPU | AI HAT+ Hailo-8L 13 TOPS (HAILO8L) | 已实测，HailoRT 4.23.0 |
| 相机 | USB UVC 1280x720 MJPG（05a3:9230） | 已接；实际约 25 fps；图像需 rotate 90°；FOV/tilt 待标定 |
| 飞控 | Betaflight, 板卡待识别 | **未接**；target 不能猜 |
| 视频 | J7 复合视频 720x576i PAL -> 5.8G VTX -> 眼镜 | J7/fb0 已实测；目视未验 |
| 模型 | YOLOv8n 1-class drone -> Hailo HEF | **128 帧真实相机校准 HEF 已在 Pi5 parse/run**；bench p50 16.21 ms，61.7 fps |

### 数据流

~~~text
USB UVC 1280x720 MJPG
  -> CameraSource (latest-frame, no X11)
  -> HailoDetector (YOLOv8, 640x640, async)
  -> BboxTracker (confirm/gate/coast)
  -> SafetySupervisor (ARM/AUX freshness/altitude/loop)
  -> PVisualGuidance (P + PN, closing speed source)
  -> RCMux (single MSP RC writer, 50 Hz)
  -> Betaflight (ARM + ANGLE + BOXMSPOVERRIDE)
  -> MSP telemetry (ATTITUDE/ALTITUDE/STATUS/RC/ANALOG)
  -> OSD -> /dev/fb0 (720x576 BGRA) -> J7 -> VTX -> goggles
~~~

### 关键约束

- 飞手拥有 ARM / ANGLE / BOXMSPOVERRIDE / 切入开关；伴机只写 RPYT，**绝不命令 AUX/ARM**。
- 任何 RC/姿态/高度/链路过期、回路超时、目标丢失、低电量/高度/距离越界 → 立即回 PASSTHROUGH；
  关键故障锁存，需飞手把切入开关回低再重新进入。
- 第一版真机只用 ANGLE + BOXMSPOVERRIDE + 伴机 RPYT + 直接油门；
  不用 ALT_HOLD / POS_HOLD / offboard / GPS rescue。
- 无 GPS/光流：制导用 speed_source=closing，水平位置/速度不参与；高度用 MSP_ALTITUDE。
- 相机 tilt/yaw、FC 板卡、AUX 映射、电池阈值必须实测后写入 config/onboard.json；不要猜。

### 快速命令

~~~bash
# 本机单测（当前 348 项）
python3 -m pytest -q

# 协议自检（离线 / SITL TCP / 真机串口）
python3 tools/protocol_conformance.py --offline
python3 tools/protocol_conformance.py --host 127.0.0.1 --tcp-port 5761

# Pi5 只读体检 + 部署（在 Pi 上运行 install）
bash scripts/pi5_bringup.sh --root /home/pi5/BF_Anti
sudo bash scripts/pi5_install.sh
python3 -m onboard.app --config config/onboard.json

# Hailo 工具
python3 tools/hailo/check_env.py --role pi5
python3 tools/hailo/bench_hef.py --hef /usr/share/hailo-models/yolov8s_h8l.hef --frames 300
~~~

### 硬件与证据边界

- 已实测：Pi5 J7 PAL 720x576i、fb0 BGRA 写入、HailoRT 4.23 + HAILO8L 识别、
  预编译 yolov8s HEF 100 帧、OSD+DisplayPipeline 25 FPS/6.5 ms 渲染、SITL MSP 协议 12 项通过。
- 相机已接：/dev/video0（05a3:9230）MJPG 1280x720，CSV 实测约 25 fps；CameraSource rotate_deg=90
  后输出 720x1280，HailoDetector 约 51 fps、单帧推理约 19.4 ms，0 errors。
- 自编 HEF 已在 Pi5 验证：hailortcli parse-hef 显示 HAILO8L + 1-class NMS；hailortcli run
  100 帧 HW Latency 14.89 ms，芯片 31.3 C。
- 已用真实 Pi 相机帧生成 128 帧校准集（100% real_pi_camera），自编 HEF 在 Pi5 parse/run：
  parse-hef HAILO8L + 1-class NMS；bench 200 帧 p50 16.21 ms、p95 16.26 ms、61.7 infer fps。
  OSD 实测截图见 `build/camera_local/fb0_pi128.png`；当前场景墙面出现 0.31 的框，属未验证/疑似
  误报，不作为精度结论。
- 未实测（阻塞）：真实 FC 串口/接收机、BOXMSPOVERRIDE 实配、目标检测精度、1024 帧校准、
  拆桨电机 A/B、J7 目视、飞行。不要把这些写成已通过。

---

## 旧版 RDK X5 链路

> 以下 §架构/§硬件/§配置/§使用方法 描述的是已归档的 RDK X5 + Pico 链路，
> 代码在 [legacy/rdk_x5_pico/](legacy/rdk_x5_pico/README.md)，图上文件名仅作历史示意。
> 新工作请从上面的 Pi5 主线开始。

---

## 架构

```
接收机 ──(CRSF)──▶ FC UART          ← 飞手全权, 物理开关控制 ARM/ANGLE/OVERRIDE
 X5   ──(MSP USB)──▶ FC            ← 读 RC 通道 + 追踪时覆盖 RPYT

AUX3 低位 → FC 用 CRSF (飞手飞)
AUX3 高位 → FC 用 MSP  (X5 PBVS 追踪)
```

```
USB 相机 ──▶ RDK X5 ── BPU 推理 ── YOLO 检测框
                │
        ┌───────┴────────┐
        │   tracker.py    │
        │   中心距离+EMA   │
        └───────┬────────┘
                │
        ┌───────┴───────────────┐
        │  tracking_controller  │
        │  PBVS 视觉伺服 + PID   │
        └───────┬───────────────┘
                │
        ┌───────┴────────┐
        │   rc_mux.py     │
        │   50Hz RC 融合   │
        └───────┬────────┘
                │
        ┌───────┴────────┐
        │   msp.py        │
        │   MSP USB VCP   │
        └───────┬────────┘
                │
        ┌───────┴────────┐
        │ Betaflight FC   │
        │ BOXMSPOVERRIDE  │
        └────────────────┘
```

---

## 硬件

| 组件 | 型号 | 用途 |
|------|------|------|
| 机载电脑 | RDK X5 (Sunrise 5, 10 TOPS) | YOLO 推理 + MSP 控制 |
| 飞控 | Betaflight 4.5.3 | Serial RX + CRSF, BOXMSPOVERRIDE |
| 摄像头 | USB 摄像头 | YOLO 检测输入 |
| 接收机 | ELRS / Crossfire | 接飞控 UART, CRSF 协议 |
| 图传 | 5.8G VTX + HDMI 转换器 | 视频发射 |
| 眼镜 | FPV Goggles | 接收显示 |

## 文件结构

```
BF_Anti/
├── onboard/                  Pi5 机载主线: app/camera/controller/safety/fc/detector/video
├── config/                   onboard.example.json + profiles/*.json
├── tools/                    protocol_conformance / fw_motor_ab / hailo/*
├── systemd/                  bf-anti-onboard.service
├── legacy/rdk_x5_pico/       旧版 X5 + Pico 链路（已归档，见 legacy/README.md）
│   ├── detect.py / tracker.py / tracking_controller.py
│   ├── video.py / pico_fw/    PAL -> SPI -> Pico（旧图传）
│   ├── crsf.py / motors.py / run_detect.sh
│   ├── scripts/               X5 pinmux / GPIO / USB camera / 旧 service
│   └── model/*.bin            X5 BPU 模型
├── rc_mux.py                 RC 融合层 (50Hz, 透传/替换)
├── msp.py                    Betaflight MSP 协议驱动 (USB VCP)
├── model/                    检测模型（ONNX 仿真/Hailo 源；X5 .bin 已归档）
├── guidance/                 制导律（主线 p_visual_guidance.py）+ XY 速度外环
├── perception/               相机几何 / 目标估计 / 3D KF
├── modeling/                 理论模型与整定（四旋翼、高度环、PID）
├── sim/                      Gazebo SITL 仿真工作区 -> 见 sim/README.md
├── docs/                     文档（本地保留、未纳入版本控制；见 .gitignore）
├── tests/                    单元测试（348 项）
└── scripts/
    ├── pi5_bringup.sh / pi5_install.sh / pi5_run_onboard.sh
    ├── pi5_composite_check.sh / pi5_camera_probe.py
    ├── fc_identify.py / rc_monitor.py / test_msp_override.py
    └── legacy 脚本已移到 legacy/rdk_x5_pico/scripts/
```

---

## 安装依赖

```bash
# rdk_model_zoo (BPU 推理运行时)
git clone https://github.com/D-Robotics/rdk_model_zoo.git ~/rdk_model_zoo
ln -s ~/rdk_model_zoo ~/BF_Anti/rdk_model_zoo

# Python 依赖
pip3 install pyserial numpy opencv-python

# 开机自启
cd ~/BF_Anti && ./legacy/rdk_x5_pico/scripts/service_ctl.sh install
```

---

## 配置

### 1. 飞控 (Betaflight 4.5.3)

**Ports 标签页:**
- USB VCP → MSP ✅ (115200)
- 接收机所在 UART → 串行接收机 ✅

**Configuration 标签页:**
- Receiver Mode → **串行接收机**
- Serial Receiver Provider → **CRSF**

**Modes 标签页:**
- ARM → AUX1, 1900–2100
- ANGLE → AUX2, 900–2100 (全量程常开)
- BOXMSPOVERRIDE → AUX3, 1900–2100

**CLI:**
```
set msp_override_channels_mask = 15
set msp_override_failsafe = ON
save
```

| CLI 参数 | 值 | 说明 |
|----------|---|------|
| `msp_override_channels_mask` | 15 | bit0-3 = RPYT 四个通道走 MSP, AUX 走 CRSF |
| `msp_override_failsafe` | ON | OVERRIDE 期间 MSP 断线触发 failsafe |

### 2. 接线

```
接收机 VCC → 飞控 5V
接收机 GND → 飞控 GND
接收机 TX  → 飞控 RX (串行接收机 UART)
X5 USB     → 飞控 USB
```

---

## 使用方法

### 基准检测 (无追踪)

```bash
./legacy/rdk_x5_pico/run_detect.sh bench
```

### 追踪测试

```bash
# 地面追踪 (base_throttle=1100)
./legacy/rdk_x5_pico/run_detect.sh track

# 飞行追踪 (base_throttle=1550)
./legacy/rdk_x5_pico/run_detect.sh fly
```

### MSP 控制命令

```bash
python3 msp.py              # 快速通信测试
python3 msp.py check        # 完整诊断 (RC → FAILSAFE → ARM)
python3 msp.py monitor      # 持续读姿态角
```

### 电机 / Override 测试

```bash
# RC 通道实时监控
python3 scripts/rc_monitor.py

# BOXMSPOVERRIDE 全流程测试
python3 scripts/test_msp_override.py
```

---

## 参数参考

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model-path` | (必填) | X5 .bin 模型路径 |
| `--cam-width` | 1280 | 摄像头宽度 |
| `--cam-height` | 720 | 摄像头高度 |
| `--cam-fps` | 60 | 摄像头帧率 |
| `--score-thres` | 0.4 | YOLO 检测置信度阈值 |
| `--track` | on | 启用目标追踪 |
| `--tracking` | off | 启用 PBVS 视觉伺服追踪 |
| `--tracking-switch` | 6 | AUX3 (CH7), 追踪切入开关 |
| `--base-throttle` | 1100 | 追踪基础油门 (飞行设 1550) |
| `--pid-yaw` | 2.0 0.1 0.5 300 | Yaw PID 参数 |
| `--pid-pitch` | 2.5 0.1 0.5 300 | Pitch PID 参数 |
| `--pid-thr` | 1.5 0.05 0.3 200 | Thr  PID 参数 |
| `--tracking-lost-timeout` | 2.0 | 目标丢失几秒进搜索模式 |
| `--tracking-max-lost` | 5.0 | 搜索几秒放弃, 退回 MANUAL |
| `--no-display` | off | 无头模式 |
| `--save` | — | 保存检测结果视频 |

---

## 追踪状态机

```
MANUAL ──(AUX3 高位)──▶ ARMED ──(检测到目标)──▶ TRACKING
                                                     │
                                       ┌─ 目标丢失 >2s ─▶ LOST ──(>5s)──▶ MANUAL
                                       │                    │
                                       └─ 摇杆偏差 >30% ────┴──▶ MANUAL
                                       └─ AUX3 拉低 ──────────▶ MANUAL
```

## 安全机制

| 机制 | 行为 |
|------|------|
| AUX3 拉低 | 立即退出追踪, FC 切回 CRSF 直通 |
| 摇杆偏差 >30% | 飞手接管, 退出追踪 |
| 追踪油门限幅 | 1050–1800 RC 单位 |
| 异常退出 | `finally` 块先 DISARM 再关相机 |
| ARM 通道 | AUX1 永远走 CRSF, X5 绝不控制 |

---

## 服务管理

```bash
sudo systemctl start bf_anti     # 启动追踪
sudo systemctl stop bf_anti      # 停止追踪
systemctl status bf_anti         # 查看状态
journalctl -u bf_anti -f         # 实时日志
```

---

## 纯 P 视觉制导打击（仿真）

Gazebo/Betaflight SITL 下的**纯比例（P-only）视觉制导**打击闭环：相机 → 检测 → 跟踪 →
纯 P 律 → RC(ANGLE) → SITL，真值只用于评分。

| 阶段 | 结果 |
|---|---|
| 2026-09-28 首版 | 5 次运行 3 次进入 0.8 m 命中半径（最近 0.45 m） |
| **2026-10-01 稳定性修复后** | **15/15 全部命中，最近真值距离 0.163 ~ 0.441 m** |

文档（均在本地 `docs/` 与 `sim/results/`，**未纳入版本控制**）：纯 P 制导设计与标定 ·
稳定性根因与修复 · 检测后端实测对比。

```bash
# 单元测试（纯 P 契约 + 运动学回归 + Pi5 机载，共 348 项）
python3 -m pytest -q

# 运动学包线：静态/径向目标命中，横向匀速目标记录滞后
python3 sim/p_strike_kinematic.py --benchmark

# Gazebo 闭环（先合成 bbox 验证几何，再换真检测）
BF_ROOT=~/betaflight BF_ANTI_WORLD=$PWD/sim/worlds/bf_anti_p_strike_light_cam.sdf \
    ./sim/launch_sitl.sh --headless &
python3 sim/run_p_vision_strike.py --synthetic --duration 45 \
    --inner-model light_quad --hover-pwm 1410 --tilt 20 --climb-alt 12 \
    --log /tmp/p.csv --plot /tmp/p.png

# 接真检测：一键演示 / 无头单发试验（换后端只改 YOLO_BACKEND）
YOLO=1 YOLO_BACKEND=pt   GUI=1 ./sim/run_p_strike_live.sh     # 我们的 YOLOv8-L  :9300
YOLO=1 YOLO_BACKEND=onnx GUI=1 ./sim/run_p_strike_live.sh     # 第三方的 YOLOv8n  :9301
YOLO_PORT=9301 ./sim/trial_p_strike.sh b_onnx --target-width 2.38   # 无头单发
```

核心文件：[`guidance/p_visual_guidance.py`](guidance/p_visual_guidance.py)（制导律）、
[`sim/run_p_vision_strike.py`](sim/run_p_vision_strike.py)（闭环 runner）、
[`perception/bbox_tracker.py`](perception/bbox_tracker.py)（检测框跟踪；sim/bbox_tracker.py 是兼容 shim）、
[`sim/p_axis_probe.py`](sim/p_axis_probe.py)（执行层符号标定，换机体/固件必做）。
仿真工作区全貌与踩坑清单：[`sim/README.md`](sim/README.md)。

### 检测后端（两个都已跑通闭环）

| 后端 | 模型 | 服务端口 | 单帧 | 同帧召回 | 打击最近距离 | `--target-width` |
|---|---|---|---|---|---|---|
| `pt` | `Anti-UAV/best.pt`（YOLOv8-L, 2 类） | :9300 | 6.13 ms | 0.97 | 0.260 / 0.279 m | 2.0 |
| `onnx` | `model/v1_yolov8_mAP50_0.879.onnx`（YOLOv8n, 1 类 `drone`） | :9301 | 4.12 ms | 0.96 | 0.257 / 0.337 m | **2.38** |

换模型必做三件事（实测踩过）：① `--target-width` 重标（框松紧变了，ONNX 宽 18%）；
② 类别名映射（客户端只认 `copter_drone`/`motor_drone`，服务端用 `YOLO_LABEL_MAP`）；
③ 服务端 `YOLO_CONF` 与客户端 `--min-conf` 取 max，客户端调低无效。
标定方法：`sim/eval_detector_frames.py` 在落盘原始帧上对比召回/框宽。

GPU 绑定红线：检测服务**只能绑 `/dev/nvidia0`**（= nvidia-smi index **1** = `GPU-40c96051-...`），
绝不上 nvidia-smi index 0（其他用户的卡）。细节与两个坑见本地文档
`docs/archive/2026-09-10-sim-design/2026-10-05-onnx-nvidia0-gpu-test.md`（未纳入版本控制）。

---

## 模型指标

| 模型 | 规模 | 输入 | 类别 | 速度（实测） | 用途 |
|------|------|------|------|------|------|
| `model/*_bayese_640x640_nv12.bin` | YOLOv8n, 3.74 MB int8 | 640×640 NV12 | 1 (drone) | 37–43 FPS（X5 BPU 双核） | 机载上板 |
| `model/v1_yolov8_mAP50_0.879.onnx` | YOLOv8n, 3.0 M 参数 | 640×640 | 1 (drone) | 4.12 ms（H100 /dev/nvidia0） | 仿真 / 转换源头 |
| `Anti-UAV/best.pt` | YOLOv8-L, 43.6 M 参数 | 640×640 | 2 | 6.13 ms（H100 /dev/nvidia0） | 仿真 / 地面高精度 |

> 三者 mAP50：`.bin`/`.onnx` 同源（0.879，1 类）；`best.pt` 验证集 mAP50 0.9805 / mAP50-95 0.8205。
> 仿真闭环里 `.onnx` 与 `.pt` 的**命中精度无差别**（见上表），差异只在 >40 m 段召回。
> 追踪：中心距离匹配 + EMA 平滑（旧版机载 legacy/rdk_x5_pico/tracker.py）/ 门限关联 + 外推（perception/bbox_tracker.py，仿真与 Pi5 共用）。

---

## 文档与工作区

| 入口 | 内容 |
|------|------|
| [`sim/README.md`](sim/README.md) | 仿真工作区地图：入口脚本 / 库模块 / 分析工具 / 世界 / 踩坑清单 |
| [`modeling/README.md`](modeling/README.md) | 理论模型与整定工具链 |
| `guidance/` `perception/` `tests/` `sim/tools/` 下的 README | 各层职责与入口 |
| `docs/`（本地，未纳入版本控制） | 设计与进展文档、架构总览、索引、讲稿 PDF/tex 与三场景录像 |
| `sim/results/` | 验证数据（`.csv`/`.png`）；复盘报告 `*.md` 为本地文档 |

代码规范：Python 以 `python3 -m flake8 --select=E9,F63,F7,F82,F401,F841,E722,W605` 零告警为准，
`python3 -m pytest -q` 全绿（当前 348 项）；每个模块首行写模块 docstring，脚本首段写清用途与复现命令。
