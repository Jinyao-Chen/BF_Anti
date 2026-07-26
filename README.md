# BF_Anti — 反无人机 YOLO 检测 + Betaflight MSP 飞控

[![Betaflight](https://img.shields.io/badge/Betaflight-4.5.3-00A6FF)](https://github.com/betaflight/betaflight)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-FFB300)](https://github.com/ultralytics/ultralytics)
[![RDK X5](https://img.shields.io/badge/RDK-X5%20Sunrise%205-10TOPS-00C853)](https://d-robotics.github.io/rdk_doc/)
[![RP2040](https://img.shields.io/badge/Pico-RP2040-FF69B4)](https://www.raspberrypi.com/products/raspberry-pi-pico/)

> 基于地平线 RDK X5 (Sunrise 5, 10 TOPS) 的实时反无人机检测系统。
> USB 相机 → YOLO BPU 推理 → 目标追踪 → PBVS 视觉伺服 → MSP 飞控控制。

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
├── model/                    YOLO 模型文件 (.bin)
├── pico_fw/                  Pico 协处理器固件 (C + PIO 汇编)
├── docs/                     设计文档
├── tests/                    单元测试
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

## 模型指标

| 指标 | 数值 |
|------|------|
| 模型 | YOLOv8, mAP50=0.879 |
| 输入 | 640×640 NV12 |
| 输出 | 1 类 (drone) |
| 推理速度 | 37–43 FPS (BPU 双核) |
| 追踪 | 中心距离匹配 + EMA 平滑 |
