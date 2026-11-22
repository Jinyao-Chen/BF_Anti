# BF_Anti Betaflight MSP 飞控

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
| 相机 | USB UVC 1280x720 MJPG（05a3:9230） | 已接；实际约 25 fps；相机已放平，rotate_deg=0；FOV/tilt 待标定 |
| 飞控 | Betaflight 4.5.3, S405 / OMNIBUSF4SD | **已接** USB `/dev/ttyACM0`；MSP override 已配置；当前 `RX_MSP`（无 CRSF 串行接收机）、`RX_FAILSAFE`、电池未接 |
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
  -> Blackbox (logs/run_*/telemetry.csv + events.jsonl, 掉电安全)
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
# 本机单测（当前 388 项）
python3 -m pytest -q

# 协议自检（离线 / SITL TCP / 真机串口）
python3 tools/protocol_conformance.py --offline
python3 tools/protocol_conformance.py --host 127.0.0.1 --tcp-port 5761

# Pi5 只读体检 + 部署（在 Pi 上运行 install）
# --enable-service: 开机自启；--persistent-journal: journald 日志跨重启保留
bash scripts/pi5_bringup.sh --root /home/pi5/BF_Anti
sudo bash scripts/pi5_install.sh --enable-service --persistent-journal
python3 -m onboard.app --config config/onboard.json

# 实时 OSD 网页（Pi 上另开进程；本机浏览器打开 http://<Pi-IP>:8088/）
.venv/bin/python scripts/pi5_osd_stream.py --fb /dev/fb0 --port 8088

# FC 姿态符号检查（只读；Pi 上跑，本机用 SSH 隧道看）
.venv/bin/python scripts/fc_attitude_viewer.py --port /dev/ttyACM0 --http-port 8090
# 本机：ssh -N -L 8090:127.0.0.1:8090 pi5@10.42.0.228
# 浏览器：http://127.0.0.1:8090/

# Hailo 工具
python3 tools/hailo/check_env.py --role pi5
python3 tools/hailo/bench_hef.py --hef /usr/share/hailo-models/yolov8s_h8l.hef --frames 300
~~~

### 运行日志（blackbox）

开机自启后，每次服务启动都会在 `log_dir`（`config/onboard.json`，示例为
`/home/pi5/bf_anti_logs`）下生成一个运行目录；**不依赖 journald 是否持久化**：

~~~text
/home/pi5/bf_anti_logs/
  run_20261122_164721/
    startup.json      # git sha、完整配置快照、boot_id、pid、起止时间、行数、io_error
    telemetry.csv     # 飞控 / 遥控器 / 检测 / 控制指令，默认 10 Hz
    events.jsonl      # 状态变化：armed、safety 门、目标出现/丢失、mux 模式、启动失败
    runtime.log       # 人读日志（与 journald 同内容）
    faulthandler.log  # 原生崩溃栈；kill -USR1 可主动导出
  latest -> run_20261122_164721
~~~

`telemetry.csv` 的列（节选，完整顺序见 [`onboard/telemetry_log.py`](onboard/telemetry_log.py)）：

| 组 | 列 | 说明 |
|---|---|---|
| 飞控 | `armed` `fc_mode` `roll/pitch/yaw_deg` `alt_m` `vario_mps` `voltage_v` `current_a` `mah` `rssi` | 来自 MSP 遥测；`*_age_s` 给出各字段新鲜度 |
| 遥控器 | `rc_age_s` `rc0..rc15` | AETR + AUX1..AUX12；FC 未返回 MSP_RC 时**留空**，不补 1500 |
| 检测 | `det_present` `det_conf` `det_class` `det_age_s` `det_x1..y2` `det_fps` `det_infer_ms` `det_errors` | 检测框、置信度、检测线程健康 |
| 跟踪 | `track_locked` `track_streak` `track_miss` | 跟踪器是否确认/外推 |
| 控制指令 | `cmd_valid` `cmd_mode` `cmd_reason` `cmd_roll/pitch/throttle/yaw` `cmd_range_m` | 伴机想写给 FC 的 RPYT 与制导状态 |
| 安全/权限 | `allow_auto` `safety_reason` `safety_critical` `mux_mode` `mux_pos` `engaged` | 为什么没有放行、当前谁在控 |
| 回路健康 | `loop_dt_s` `link_errors` | 50 Hz 回路是否超时、MSP 错误计数 |

> 开机最初几行出现 `safety_reason=fc_warming_up`（且 `safety_critical=0`）是正常的：
> 此时飞控数据尚未齐全，安全门只拒绝、**不锁存**；约 2 s 内会转为 `ok` 或
> `target_unconfirmed`。只有 `latched:*` 才是需要飞手把模式开关回低再重新进入的故障。

写入规则（都是为掉电/坏卡设计的）：

- 控制环只入队，独立 writer 线程落盘；SD 卡慢或 fsync 阻塞**不会**拖垮 50 Hz 回路。
- 行级 flush（约 1 s 一批）、每 5 s fsync；断电最多丢几秒，而不是整场。
- `telemetry.csv` 超过 `max_file_mb`（默认 32 MB）自动轮转；只保留最近 `max_runs`（默认 20）次运行。
- 空字段表示"从未收到"，不是 0；磁盘满/只读时自动降级为 journald，并在 `startup.json` 的 `io_error` 里留证据。

~~~json
"log_dir": "/home/pi5/bf_anti_logs",
"telemetry": {
  "enabled": true, "hz": 10, "max_file_mb": 32, "max_runs": 20,
  "flush_interval_s": 1, "fsync_interval_s": 5, "level": "INFO"
}
~~~

`hz` 是落盘频率，不是控制频率（默认 10 Hz 抽稀）；要逐 tick 记录 50 Hz 的完整控制指令，
把 `hz` 设为 `0`。临时排查可加 `--no-telemetry`（只留 journald）或 `--log-dir /tmp/logs`。

#### 开机后确认"到底记没记"

~~~bash
systemctl status bf-anti-onboard
journalctl -u bf-anti-onboard -b --no-pager | tail -40   # 本次开机
bash scripts/pi5_log_check.sh                            # 一键体检：unit/日志目录/行数/启动竞态
python3 tools/log_summary.py --dir /home/pi5/bf_anti_logs --runs 1
tail -f /home/pi5/bf_anti_logs/latest/telemetry.csv
systemctl kill -s USR1 bf-anti-onboard                   # 卡住时导出 Python 栈
~~~

`log_summary.py` 会直接给出四路信号是否齐全：
`飞控=ok/N 遥控器=ok/N 检测=ok/N 控制指令=ok/N`，缺哪一路就打印对应的排查方向。

#### 为什么以前"开机后没有日志"

按可能性排序，前三条是这次修复的重点：

1. **只有 stderr，没有落盘文件。** 旧代码 `logging.basicConfig()` 只写 stderr，systemd 转进
   journald；而 Ubuntu/RPi OS 的 journald 默认 `Storage=auto`，`/var/log/journal` 不存在时是
   **易失**的，重启后 `journalctl -b -1` 为空。配置里的 `log_dir` 从来没有任何代码读取。
   → 现在固定写 `run_*/telemetry.csv`；另可 `sudo bash scripts/pi5_enable_persistent_journal.sh`
   让 journald 也持久化。
2. **开机启动竞态把进程打死。** 开机瞬间 `/dev/fb0` 往往还没就绪（`DisplayPipeline.start()`
   直接抛异常），HEF/HailoRT 初始化失败也会；进程退出后 systemd `StartLimitBurst=5` 在 60 s
   内耗尽，**不再重启**，于是整场没有日志。FC 虽不会让进程退出，但旧逻辑会在缺乏数据时把
   safety 锁存成 `rc_stale`，整场 `cmd_valid=0`（看起来就是"没有控制指令数据"）。
   → 现在 `_start_with_retry()` 常驻重试（1→10 s 退避），每次失败写 `events.jsonl` 的
   `startup_retry`；开机未收到完整 FC 数据时安全门只拒绝、不锁存。
3. **原生库崩溃没有 Python traceback。** HailoRT/OpenCV 段错误会让进程直接消失。
   → `faulthandler.log` 记录崩溃时的 Python 栈；`SIGUSR1` 可主动 dump。
4. **服务没 enable / 用户权限。** `systemctl is-enabled bf-anti-onboard`；服务用户（安装时
   `--user`，默认 `pi5`）需要 `video`/`dialout` 组；`ProtectSystem=full` 下 `log_dir` 必须在
   `/home` 或 `/var`（unit 已 `ReadWritePaths=安装目录`）。
5. **磁盘满 / SD 卡只读。** `df -h`；`startup.json` 的 `io_error` 会写明原因。

> 状态：本机新增测试（[`tests/test_telemetry_log.py`](tests/test_telemetry_log.py)、
> [`tests/test_onboard_telemetry.py`](tests/test_onboard_telemetry.py)、
> [`tests/test_log_summary.py`](tests/test_log_summary.py)）全部通过；
> **Pi5 真机开机落盘尚未复测**。首次上机请先跑 `scripts/pi5_log_check.sh`，并保留
> `run_*/startup.json` 作为“代码版本 + 配置 + boot_id”的证据。

### 硬件与证据边界

- 已实测：Pi5 J7 PAL 720x576i、fb0 BGRA 写入、HailoRT 4.23 + HAILO8L 识别、
  预编译 yolov8s HEF 100 帧、OSD+DisplayPipeline 25 FPS/6.5 ms 渲染、SITL MSP 协议 12 项通过。
- 相机已接：/dev/video0（05a3:9230）MJPG 1280x720，CSV 实测约 25 fps；CameraSource rotate_deg=0
  后输出 1280x720，HailoDetector 约 51 fps、单帧推理约 19.4 ms，0 errors。
- 自编 HEF 已在 Pi5 验证：hailortcli parse-hef 显示 HAILO8L + 1-class NMS；hailortcli run
  100 帧 HW Latency 14.89 ms，芯片 31.3 C。
- 已用真实 Pi 相机帧生成 128 帧校准集（100% real_pi_camera），自编 HEF 在 Pi5 parse/run：
  parse-hef HAILO8L + 1-class NMS；bench 200 帧 p50 16.21 ms、p95 16.26 ms、61.7 infer fps。
  OSD 实测截图见 `build/camera_local/fb0_pi128.png`；当前场景墙面出现 0.31 的框，属未验证/疑似
  误报，不作为精度结论。
- FC 已接入并完成只读+配置检查：S405/OMNIBUSF4SD，Betaflight 4.5.3，`/dev/ttyACM0`；
  `msp_override_channels_mask=15`、`msp_override_failsafe=ON`，AUX1=ARM、AUX2=ANGLE、AUX3=MSP OVERRIDE；
  16 通道 RC round-trip max diff 0，协议 12 pass / 0 fail（attitude_sign 仍需实机倾斜，MSP2 position 不支持）。
- 但 FC 当前 `feature` 是 **RX_MSP**（`RX_SERIAL` 关闭），三路 UART 都没有 serial RX；`serialrx_provider=CRSF`
  只是未生效的配置。RC round-trip 验证的是 MSP 输入，不是飞手 CRSF 接收机。
- 手动姿态测试：ROLL + = 右翼下压、YAW 向左减小；PITCH 的 Betaflight 原始符号是 + = 低头，
  已在 `onboard/fc/state.py` 转为 + = 抬头，OSD/guidance 使用航空约定。
- 当前 FC 仍报 `RX_FAILSAFE`，且 `0S battery - NOT PRESENT`；接收机/电池/拆桨电机 A/B/飞行未验。
- 未实测（阻塞）：CRSF 接收机接入与 UART 配置、飞手 ARM/ANGLE/OVERRIDE 独立开关、拆桨电机 A/B、
  目标检测精度、1024 帧校准、J7 目视、飞行。不要把这些写成已通过。

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
├── tests/                    单元测试（350 项）
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
# 单元测试（纯 P 契约 + 运动学回归 + Pi5 机载，共 388 项）
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
`python3 -m pytest -q` 全绿（当前 350 项）；每个模块首行写模块 docstring，脚本首段写清用途与复现命令。
