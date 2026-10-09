# BF_Anti — 反无人机 YOLO 检测 + Betaflight MSP 飞控

[![Betaflight](https://img.shields.io/badge/Betaflight-4.5.3-00A6FF)](https://github.com/betaflight/betaflight)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-FFB300)](https://github.com/ultralytics/ultralytics)
[![RDK X5](https://img.shields.io/badge/RDK-X5%20Sunrise%205-10TOPS-00C853)](https://d-robotics.github.io/rdk_doc/)
[![RP2040](https://img.shields.io/badge/Pico-RP2040-FF69B4)](https://www.raspberrypi.com/products/raspberry-pi-pico/)

> 基于地平线 RDK X5 (Sunrise 5, 10 TOPS) 的实时反无人机检测系统。
> USB 相机 → YOLO BPU 推理 → 目标追踪 → PBVS 视觉伺服 → MSP 飞控控制。

仓库里有**两条链路**，别混淆：

| 链路 | 跑在哪 | 入口 | 说明 |
|------|--------|------|------|
| **机载链路** | RDK X5 + Betaflight FC | [`detect.py`](detect.py) / [`run_detect.sh`](run_detect.sh) | 本文档 §架构 起介绍；BPU `.bin` 推理 + MSP 覆盖 RC |
| **仿真链路** | PC + Gazebo SITL | [`sim/run_p_vision_strike.py`](sim/run_p_vision_strike.py) | 纯 P 视觉制导打击，见 §纯 P 视觉制导打击 与 [`sim/README.md`](sim/README.md) |

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
├── detect.py                 主程序: YOLO 检测 + 追踪 + MSP 控制
├── tracker.py                中心距离目标追踪 (EMA 平滑 + 轨迹)
├── tracking_controller.py    PBVS 视觉伺服 + 状态机 + PID 控制 (纯计算)
├── rc_mux.py                 RC 融合层 (50Hz, 透传/替换)
├── crsf.py                   CRSF 协议解码 + CRC8 校验
├── msp.py                    Betaflight MSP 协议驱动 (USB VCP)
├── video.py                  PAL 视频帧编码 → SPI → Pico
├── motors.py                 安全电机测试 (油门 1050→1100)
├── run_detect.sh             一键启动脚本 (bench/track/fly)
├── model/                    检测模型（.onnx 仿真 / .bin 上板）
├── guidance/                 制导律（主线 p_visual_guidance.py）+ XY 速度外环
├── perception/               相机几何 / 目标估计 / 3D KF
├── modeling/                 理论模型与整定（四旋翼、高度环、PID）
├── sim/                      Gazebo SITL 仿真工作区 → 见 sim/README.md
├── docs/                     文档（本地保留、未纳入版本控制；见 .gitignore）
├── pico_fw/                  Pico 协处理器固件 (C + PIO 汇编)
├── tests/                    单元测试（153 项）
└── scripts/
    ├── test_msp_override.py  BOXMSPOVERRIDE 地面测试
    ├── rc_monitor.py         RC 通道实时监控
    ├── crsf_monitor.py       CRSF 串口数据监控
    ├── bf_anti.service       systemd 自启服务
    ├── service_ctl.sh        服务管理脚本
    ├── hw_test.py            SPI 硬件链路诊断
    ├── gpio_probe.py         GPIO 探测工具
    ├── usb_camera_probe.py   USB 摄像头能力探测
    └── usb_camera_test_gst.py GStreamer 摄像头测试
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
cd ~/BF_Anti && ./scripts/service_ctl.sh install
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
./run_detect.sh bench
```

### 追踪测试

```bash
# 地面追踪 (base_throttle=1100)
./run_detect.sh track

# 飞行追踪 (base_throttle=1550)
./run_detect.sh fly
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
# 单元测试（纯 P 契约 + 运动学回归，共 153 项）
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
[`sim/bbox_tracker.py`](sim/bbox_tracker.py)（检测框跟踪）、
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
`docs/interception/2026-10-05-onnx-nvidia0-gpu-test.md`（未纳入版本控制）。

---

## 模型指标

| 模型 | 规模 | 输入 | 类别 | 速度（实测） | 用途 |
|------|------|------|------|------|------|
| `model/*_bayese_640x640_nv12.bin` | YOLOv8n, 3.74 MB int8 | 640×640 NV12 | 1 (drone) | 37–43 FPS（X5 BPU 双核） | 机载上板 |
| `model/v1_yolov8_mAP50_0.879.onnx` | YOLOv8n, 3.0 M 参数 | 640×640 | 1 (drone) | 4.12 ms（H100 /dev/nvidia0） | 仿真 / 转换源头 |
| `Anti-UAV/best.pt` | YOLOv8-L, 43.6 M 参数 | 640×640 | 2 | 6.13 ms（H100 /dev/nvidia0） | 仿真 / 地面高精度 |

> 三者 mAP50：`.bin`/`.onnx` 同源（0.879，1 类）；`best.pt` 验证集 mAP50 0.9805 / mAP50-95 0.8205。
> 仿真闭环里 `.onnx` 与 `.pt` 的**命中精度无差别**（见上表），差异只在 >40 m 段召回。
> 追踪：中心距离匹配 + EMA 平滑（机载 `tracker.py`）/ 门限关联 + 外推（仿真 `sim/bbox_tracker.py`）。

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
`python3 -m pytest -q` 全绿（当前 153 项）；每个模块首行写模块 docstring，脚本首段写清用途与复现命令。
