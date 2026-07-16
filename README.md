# BF_Anti — 反无人机 YOLO 检测 + Betaflight MSP 飞控

[![Betaflight](https://img.shields.io/badge/Betaflight-4.5.3-00A6FF?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==)](https://github.com/betaflight/betaflight)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-FFB300)](https://github.com/ultralytics/ultralytics)
[![RDK X5](https://img.shields.io/badge/RDK-X5%20Sunrise%205-10TOPS-00C853)](https://d-robotics.github.io/rdk_doc/)
[![RP2040](https://img.shields.io/badge/Pico-RP2040-FF69B4)](https://www.raspberrypi.com/products/raspberry-pi-pico/)

> 基于地平线 RDK X5 (Sunrise 5, 10 TOPS) 的实时反无人机检测系统。USB 相机 → YOLO 推理 → 目标追踪 → MSP 飞控控制 + 模拟图传。

---

## 系统架构

```
USB 相机 ──▶ RDK X5 ── BPU 推理 ── YOLO 检测框
                │                        │
                │              ┌─────────┴──────────┐
                │              │   tracker.py        │
                │              │   IoU + EMA 追踪    │
                │              └────────────────────┘
                │
        ┌───────┴────────┐
        │  MSP 飞控控制    │        视频图传
        │  /dev/ttyACM0   │    ┌──────────────┐
        │  RC 通道 + ARM   │    │ video.py     │
        │  姿态读取        │    │ PAL 帧编码    │
        └────────────────┘    │     ↓ SPI      │
                              │ Raspberry Pi   │
                              │ Pico (PIO+DMA) │
                              │     ↓ GPIO     │
                              │ 270Ω → FC VI   │
                              │ OSD → VTX → 眼镜│
                              └──────────────┘
```

## 硬件

| 组件 | 型号 | 用途 |
|------|------|------|
| 机载电脑 | RDK X5 (Sunrise 5, 10 TOPS) | YOLO 推理 + MSP 控制 |
| 飞控 | OMNIBUSF4SD / S405 | BF 4.5.3, MSP RX |
| 图传协处理器 | Raspberry Pi Pico (RP2040) | PAL 视频生成 |
| 摄像头 | USB 摄像头 | YOLO 检测输入 |
| 图传 | 5.8G VTX | 模拟视频发射 |
| 眼镜 | FPV Goggles | 接收显示 |

## 文件结构

```
BF_Anti/
├── detect.py         主程序: YOLO 检测 + 追踪 + 显示
├── tracker.py        IoU 目标追踪 (EMA 平滑 + 轨迹)
├── msp.py            MSP 通信库 (RC控制 + 姿态 + 解锁)
├── video.py          PAL 视频帧编码 → SPI 发给 Pico
├── motors.py         一键解锁 + 油门测试
├── model/            YOLO 模型文件
├── rdk_model_zoo/    → ~/Desktop/rdk_model_zoo 符号链接
├── betaflight/       → ~/betaflight BF 源码 符号链接
├── pico_fw/          Pico 固件 (C + PIO 汇编)
│   ├── main.c        SPI 从机 + DMA + PIO 驱动
│   ├── pal_video.pio PIO 状态机 (PAL 像素输出)
│   └── CMakeLists.txt
└── scripts/
    └── hw_test.py     SPI 硬件链路诊断
```

## 配置

### 1. 飞控 (Betaflight 4.5.3)

**Ports 标签页:**
- USB VCP → MSP ✅ 打开

**Configuration 标签页:**
- Receiver Mode → **MSP RX (MSP)**
- ESC Protocol → **DSHOT600**

**Modes 标签页:**
- ARM → AUX1, 范围 1900–2100
- 其他模式全部删除或设为 AUX 空

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

# GPIO 引脚 (Pico 接线)
# Pin 19 (SPI1_MOSI) → Pico GP4
# Pin 23 (SPI1_SCLK) → Pico GP6
# Pin 24 (SPI1_CS)   → Pico GP5
# Pin 20 (GND)       → Pico GND
# Pin 2  (5V)        → Pico VBUS
```

### 3. Pico 固件

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

### 4. 接线总图

```
RDK X5                     Pico GPIO
Pin 19 (MOSI) ──────────── GP4  (SPI RX)
Pin 23 (SCLK) ──────────── GP6  (SPI SCK)
Pin 24 (CS)   ──────────── GP5  (SPI CS)
GND           ──────────── GND
5V            ──────────── VBUS

Pico GPIO0 ──── 270Ω ──── FC VI (视频入)

FC VO  ──── VTX Video
FC GND ──── VTX GND
电池+  ──── FC VBAT + VTX VIN (12V)
```

---

## 模型演示

```bash
# 对 MP4 视频运行检测 (弹窗显示)
python3 ~/BF_Anti/detect.py \
  --model-path ~/BF_Anti/model/v1_yolov8_mAP50_0.879_bayese_640x640_nv12.bin \
  --video BOARD_HOME/counter-uav-measures/uav.mp4 \
  --track --score-thres 0.5

# 保存检测结果视频
python3 ~/BF_Anti/detect.py \
  --model-path ~/BF_Anti/model/v1_yolov8_mAP50_0.879_bayese_640x640_nv12.bin \
  --video BOARD_HOME/counter-uav-measures/uav.mp4 \
  --track --score-thres 0.5 \
  --no-display --save ~/BF_Anti/demo_output.mp4
```

| 指标 | 数值 |
|------|------|
| 模型 | YOLOv8, mAP50=0.879 |
| 输入 | 640×640 NV12 |
| 输出 | 1 类 (drone) |
| 推理速度 | 37–43 FPS (BPU) |
| 追踪 | IoU 匹配 + EMA 平滑 |

---

## 测试流程

### 第一步: 通信诊断

```bash
# 飞控 USB 连接后
python3 ~/BF_Anti/msp.py check
```

确认: 通道映射 OK, FAILSAFE 消除, 无解锁阻挡位

### 第二步: 电机测试

**⚠️ 务必拆桨！接电池！**

```bash
python3 ~/BF_Anti/motors.py
```

电机应自动解锁并怠速旋转，然后油门从 1050 爬升到 1400。

### 第三步: 视频图传

**Pico 固件烧录后:**

```bash
# 发送棋盘格测试图案
python3 ~/BF_Anti/video.py
```

眼镜应看到棋盘格 + 移动竖条。拧电位器调最佳亮度。

### 第四步: YOLO 实时检测

```bash
python3 ~/BF_Anti/detect.py \
  --model-path model/v1_yolov8_mAP50_0.879_bayese_640x640_nv12.bin \
  --cam-width 1280 --cam-height 720 --cam-fps 30 \
  --track --score-thres 0.5 \
  --spi-video
```

| 参数 | 说明 |
|------|------|
| `--track` | 启用目标追踪 |
| `--no-track` | 关闭追踪 |
| `--spi-video` | 开启 PAL 视频输出 (Pico 模式) |
| `--goggles` | HDMI PAL 输出模式 (HDMI→AV 转换器) |
| `--score-thres` | 检测置信度阈值 (默认 0.5) |
| `--video` | 视频文件路径 (mp4/avi), 替代摄像头 |
| `--save` | 保存检测结果视频 (e.g. output.mp4) |
| `--cam-width/height` | 摄像头分辨率 |

### MSP 控制命令汇总

```bash
python3 msp.py              # 快速通信测试 (3 次读姿态)
python3 msp.py check        # 完整诊断 (RC → FAILSAFE → ARM)
python3 msp.py arm          # 解锁 + 保活
python3 msp.py throttle     # 手动推油门
python3 msp.py monitor      # 持续读姿态角
python3 msp.py hover        # 慢推油门测试
```

---

## 已知限制

| 项目 | 说明 |
|------|------|
| 视频 | Pico 方案输出 1-bit 黑白 408×576 @ 25fps |
| 解锁 | 需先 DISARM 预热 2s (BF ARM_SWITCH 安全检测) |
| 固件 | OMNIBUSF4SD 最高支持 BF 4.1.1, 4.2+ 已移除该目标 |
| SPI | 无 DMA 时内核 4KB 拆包间隙不可见 (块对齐到行边界) |
