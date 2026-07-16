# BF_Anti — 反无人机 YOLO 检测 + Betaflight MSP 飞控

[![Betaflight](https://img.shields.io/badge/Betaflight-4.5.3-00A6FF)](https://github.com/betaflight/betaflight)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-FFB300)](https://github.com/ultralytics/ultralytics)
[![RDK X5](https://img.shields.io/badge/RDK-X5%20Sunrise%205-10TOPS-00C853)](https://d-robotics.github.io/rdk_doc/)
[![RP2040](https://img.shields.io/badge/Pico-RP2040-FF69B4)](https://www.raspberrypi.com/products/raspberry-pi-pico/)

> 基于地平线 RDK X5 (Sunrise 5, 10 TOPS) 的实时反无人机检测系统。
> USB 相机 → YOLO BPU 推理 → 目标追踪 → PBVS 视觉伺服 → MSP 飞控控制。

---

## 系统架构

```
USB 相机 ──▶ RDK X5 ── BPU 推理 ── YOLO 检测框
                │                        │
                │              ┌─────────┴──────────┐
                │              │   tracker.py        │
                │              │   中心距离 + EMA 追踪│
                │              └────────────────────┘
                │                        │
                │        ┌───────────────┴───────────────┐
                │        │  tracking_controller.py       │
                │        │  PBVS 视觉伺服 + PID 控制      │
                │        │  TargetSelector (3→1→2 优先级)│
                │        └───────────────┬───────────────┘
                │                        │
        ┌───────┴────────┐     ┌────────┴────────┐
        │  MSP 飞控控制    │     │  CRSF 摇杆检测   │
        │  /dev/ttyACM0   │     │  /dev/ttyS3     │
        │  RC 通道 + ARM   │     │  摇杆偏差 >30%   │
        │  姿态读取        │     │  → 退出追踪      │
        └────────────────┘     └─────────────────┘
```

## 硬件

| 组件 | 型号 | 用途 |
|------|------|------|
| 机载电脑 | RDK X5 (Sunrise 5, 10 TOPS) | YOLO 推理 + MSP 控制 |
| 飞控 | Betaflight 4.5.3 | MSP RX 接收机模式 |
| 摄像头 | USB 摄像头 | YOLO 检测输入 |
| 接收机 | ELRS / Crossfire | CRSF 协议，摇杆检测 |
| 图传 | 5.8G VTX + HDMI 转换器 | 视频发射 |
| 眼镜 | FPV Goggles | 接收显示 |

## 文件结构

```
BF_Anti/
├── detect.py                 主程序: YOLO 检测 + 追踪 + MSP 控制
├── tracker.py                中心距离目标追踪 (EMA 平滑 + 轨迹)
├── tracking_controller.py    PBVS 视觉伺服 + 状态机 + PID 控制
├── msp.py                    Betaflight MSP 协议驱动 (USB VCP)
├── video.py                  PAL 视频帧编码 → SPI → Pico
├── motors.py                 安全电机测试 (油门 1050→1100)
├── model/                    YOLO 模型文件
├── pico_fw/                  Pico 协处理器固件 (C + PIO 汇编)
│   ├── main.c                SPI 从机 + SCLK 间隔同步 + DMA + 看门狗
│   ├── pal_video.pio         PIO 状态机 (PAL 像素输出)
│   └── CMakeLists.txt
└── scripts/
    ├── hw_test.py            SPI 硬件链路诊断
    ├── gpio_probe.py         GPIO 探测工具
    ├── usb_camera_probe.py   USB 摄像头能力探测
    └── usb_camera_test_gst.py GStreamer 摄像头测试
```

---

## 配置

### 1. 飞控 (Betaflight 4.5.3)

**Ports 标签页:**
- USB VCP → MSP ✅ 打开 (波特率 115200)

**Configuration 标签页:**
- Receiver Mode → **MSP RX (MSP)**
- ESC Protocol → **DSHOT600**

**Modes 标签页:**
- ARM → AUX1, 范围 1900–2100
- 其他模式全部删除

**CLI:**
```
set small_angle = 180
set osd_displayport_device = AUTO
set vcd_video_system = PAL
feature -MOTOR_STOP
save
```

### 2. RDK X5

```bash
# HDMI PAL 输出 (goggles 模式)
xrandr --output HDMI-1 --mode 720x576 --rate 50
```

### 3. Pico 固件 (可选, 仅 SPI 图传方案需要)

```bash
# 安装 Pico SDK (一次性)
git clone https://github.com/raspberrypi/pico-sdk.git ~/pico-sdk
cd ~/pico-sdk && git submodule update --init
echo 'export PICO_SDK_PATH=$HOME/pico-sdk' >> ~/.bashrc

# 编译
cd ~/BF_Anti/pico_fw
mkdir -p build && cd build
cmake .. -DPICO_BOARD=pico
make -j4

# 烧录: 按住 Pico BOOTSEL → 插 USB → 拖 pico_pal.uf2 到 RPI-RP2
```

> **已知问题**: RDK X5 的 SPI 硬件 CS (Pin 24/26) 不工作 (DesignWare SSI + syscon 寄存器)。固件已改用 **SCLK 空闲间隔检测** (>5ms) 做帧同步，无需 CS 线。接线时 **不要连接 X5 Pin 24 到 Pico**。

### 4. 接线

```
RDK X5                     Pico GPIO
Pin 19 (MOSI) ──────────── GP4  (SPI RX)
Pin 23 (SCLK) ──────────── GP6  (SPI SCK)
GND           ──────────── GND
5V            ──────────── VBUS
⚠️ Pin 24 (CS) 不接!       GP5 内部下拉

Pico GPIO0 ──── 270Ω ──── FC VI (视频入)
```

---

## 使用方法

### 基准检测 (无追踪)

```bash
python3 ~/BF_Anti/detect.py \
  --model-path ~/BF_Anti/model/v1_yolov8_mAP50_0.879_bayese_640x640_nv12.bin \
  --cam-width 1280 --cam-height 720 --cam-fps 30 \
  --track --score-thres 0.65
```

### 追踪控制 (地面测试)

```bash
python3 ~/BF_Anti/detect.py \
  --model-path ~/BF_Anti/model/v1_yolov8_mAP50_0.879_bayese_640x640_nv12.bin \
  --cam-width 1280 --cam-height 720 --cam-fps 30 \
  --track --score-thres 0.65 \
  --tracking --base-throttle 1100
```

### 追踪控制 (飞行, 带 CRSF 摇杆)

```bash
python3 ~/BF_Anti/detect.py \
  --model-path ~/BF_Anti/model/v1_yolov8_mAP50_0.879_bayese_640x640_nv12.bin \
  --cam-width 1280 --cam-height 720 --cam-fps 30 \
  --track --score-thres 0.65 \
  --tracking --crsf-port /dev/ttyS3 --tracking-switch 7 \
  --base-throttle 1550
```

### MSP 控制命令

```bash
python3 msp.py              # 快速通信测试 (3 次读姿态)
python3 msp.py check        # 完整诊断 (RC → FAILSAFE → ARM)
python3 msp.py arm          # 解锁 + 保活
python3 msp.py throttle     # 手动推油门
python3 msp.py monitor      # 持续读姿态角
```

### 电机测试

**⚠️ 务必拆桨！**

```bash
python3 ~/BF_Anti/motors.py
```

电机解锁后从 1050 缓慢爬升到 1100, 最后上锁。

---

## 参数参考

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model-path` | (必填) | X5 .bin 模型路径 |
| `--cam-width` | 1280 | 摄像头宽度 |
| `--cam-height` | 720 | 摄像头高度 |
| `--cam-fps` | 60 | 摄像头帧率 |
| `--score-thres` | 0.4 | YOLO 检测置信度阈值 (建议 0.5–0.7) |
| `--track` | on | 启用目标追踪 |
| `--track-dist-thresh` | 0.15 | 中心距离匹配阈值 (0–1) |
| `--track-alpha` | 0.90 | EMA 平滑系数 (越高越灵敏) |
| `--track-max-lost` | 45 | 目标丢失后的预测帧数 |
| `--tracking` | off | 启用 PBVS 视觉伺服追踪 |
| `--tracking-switch` | — | CRSF 通道号, 摇杆开关切入追踪 |
| `--crsf-port` | — | ELRS CRSF UART 端口 (如 /dev/ttyS3) |
| `--base-throttle` | 1100 | 追踪基础油门 (飞行设 1550) |
| `--no-display` | off | 无头模式, 不弹 OpenCV 窗口 |
| `--save` | — | 保存检测结果视频 (e.g. output.mp4) |
| `--video` | — | 视频文件路径 (替代摄像头) |
| `--goggles` | off | HDMI PAL 模式 (720x576@50Hz) |

---

## 追踪状态机

```
MANUAL ──(t 键 / 摇杆开关)──▶ ARMED ──(检测到目标)──▶ TRACKING
                                                         │
                                           ┌─ 目标丢失 >2s ─▶ LOST ──(>5s)──▶ MANUAL
                                           │                    │
                                           └─ 摇杆偏差 >30% ────┴──▶ MANUAL
                                           └─ CRSF 断线 >1s ───────▶ MANUAL
```

## 安全机制

| 机制 | 行为 |
|------|------|
| 摇杆偏差 >30% | 立即退出追踪, DISARM + 油门归零 |
| CRSF 断线 (>1s) | 强制退出追踪 |
| 追踪油门限幅 | 1050–1600 RC 单位 |
| MSP 保活重连 | 串口异常不退出, 自动重试 |
| Pico 看门狗 | X5 停止发帧 >2s → Pico 自动复位 |
| 异常退出 | `finally` 块先 DISARM 再关相机 |

---

## 模型指标

| 指标 | 数值 |
|------|------|
| 模型 | YOLOv8, mAP50=0.879 |
| 输入 | 640×640 NV12 |
| 输出 | 1 类 (drone) |
| 推理速度 | 37–43 FPS (BPU 双核) |
| 追踪 | 中心距离匹配 + EMA 平滑 |

---

## 已知限制

| 项目 | 说明 |
|------|------|
| 追踪 PID | 参数硬编码, 需按机型手动调整 |
| dt 估算 | 固定 1/30, 帧率波动时 PID 行为漂移 |
| MSP 线程 | `_rc_data` 无锁 (CPython GIL 下安全) |
| SPI 图传 | 仅 1-bit 黑白 408×576 @ 25fps |
| X5 SPI CS | 硬件 CS 不工作, Pico 用 SCLK 间隔同步 |
