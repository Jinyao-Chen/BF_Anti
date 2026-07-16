#!/usr/bin/env python3
"""
Betaflight MSP 通信 & 控制脚本
RDK X5 → F405 (USB /dev/ttyACM0)

基于 Betaflight 源码分析 (rx/msp.c, rx/rx.c, flight/failsafe.c):

  数据路径:
    MSP_SET_RAW_RC → rxMspFrameReceive → rxMspFrameDone=true
    RX任务 → rxMspFrameStatus=COMPLETE → rxSignalReceived=true
    → rxFlightChannelsValid → failsafeOnValidDataReceived
    → rxDataRecoveryPeriod(默认1秒) → ARMING_DISABLED_RX_FAILSAFE 清除

  关键: FAILSAFE 不是即消的, 需要 ≥1 秒持续有效数据。
  源码: flight/failsafe.c:188-194

用法:
  python3 msp_test.py              # 快速通信测试
  python3 msp_test.py arm          # 解锁 (保活 + ARM AUX1)
  python3 msp_test.py check        # 完整诊断
  python3 msp_test.py monitor      # 持续读姿态
"""

import serial
import struct
import time
import sys
import threading

# ---- MSP 协议常量 -----------------------------------------------------------

class MSP_CMD:
    """MSP 命令码 (来自 msp_protocol.h)"""
    API_VERSION       = 1
    FC_VARIANT        = 2
    FC_VERSION        = 3
    GET_ATTITUDE      = 108
    GET_ALTITUDE      = 109
    GET_ANALOG        = 110
    GET_RC            = 105
    GET_STATUS_EX     = 150
    SET_RAW_RC        = 200


# Arming disable flags (fc/runtime_config.h)
ARMING_FLAG_NAMES = {
    0:  "NO_GYRO",
    1:  "FAILSAFE",           # 飞控处于 FAILSAFE 状态
    2:  "RX_FAILSAFE",        # 无有效接收机信号 ← 我们遇到的标志
    3:  "NOT_DISARMED",
    4:  "BOXFAILSAFE",
    5:  "RUNAWAY_TAKEOFF",
    6:  "CRASH_DETECTED",
    7:  "THROTTLE",
    8:  "ANGLE",
    9:  "BOOT_GRACE_TIME",
    10: "NOPREARM",
    11: "LOAD",
    12: "CALIBRATING",
    13: "CLI",
    14: "CMS_MENU",
    15: "BST",
    16: "MSP",
    17: "PARALYZE",
    18: "GPS",
    19: "RESC",
    20: "DSHOT_TELEM",
    21: "REBOOT_REQUIRED",
    22: "DSHOT_BITBANG",
    23: "ACC_CALIBRATION",
    24: "MOTOR_PROTOCOL",
    25: "CRASHFLIP",
    26: "ALTHOLD",
    27: "POSHOLD",
    28: "ARM_SWITCH",         # 没配 ARM 开关
}


# ---- 串口通信层 --------------------------------------------------------------

class MSP:
    """MSP 协议通信 (USB VCP)

    源码验证:
      - MSP_SET_RAW_RC handler (msp.c:2887-2902) 调用 rxMspFrameReceive
      - rxMspFrameReceive (rx/msp.c:61-77) 设置 rxMspFrameDone=true
      - rxMspFrameStatus (rx/msp.c:90-98) 返回 RX_FRAME_COMPLETE
      - rxFrameCheck (rx.c:566+) 设置 rxSignalReceived=true
      - failsafeOnValidDataReceived (flight/failsafe.c:171-196)
        首次有效数据后设置 RX_FAILSAFE 标志, 1 秒持续数据后清除
    """

    MSP_HEADER_V1 = b"$M<"
    MSP_HEADER_V1_R = b"$M>"

    def __init__(self, port="/dev/ttyACM0", baud=115200):
        self.port = port
        self.ser = serial.Serial(
            port=port, baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05)
        self.buf = bytearray()
        time.sleep(1.5)  # FC 启动时间

        # AETR: [Roll, Pitch, Thr, Yaw, AUX1..4]
        self._rc_data = [1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500]
        self._stream_alive = False
        self._stream_thread = None
        self._serial_lock = threading.Lock()

    # -- MSP 协议 ---------------------------------------------------------------

    def _send_cmd(self, cmd, payload=b""):
        """发送 MSPv1 命令, 返回 payload 或 None"""
        size = len(payload)
        chk = size ^ cmd
        for b in payload:
            chk ^= b
        frame = self.MSP_HEADER_V1 + bytes([size, cmd]) + payload + bytes([chk])
        with self._serial_lock:
            self.ser.write(frame)
            self.ser.flush()
        if cmd == MSP_CMD.SET_RAW_RC:
            return None
        return self._read_reply(cmd)

    def _read_reply(self, expected_cmd, timeout=0.5):
        """等待指定命令的响应"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.ser.in_waiting:
                self.buf.extend(self.ser.read(self.ser.in_waiting))
            while True:
                pos = self.buf.find(self.MSP_HEADER_V1_R)
                if pos < 0 or len(self.buf) < pos + 5:
                    break
                size = self.buf[pos + 3]
                pkt_end = pos + 5 + size + 1  # header(4) + size(1) + payload + chk(1)
                if len(self.buf) < pkt_end:
                    break
                cmd_rcv = self.buf[pos + 4]
                payload = bytes(self.buf[pos + 5 : pos + 5 + size])
                # checksum verify
                chk = size ^ cmd_rcv
                for b in payload:
                    chk ^= b
                if chk == self.buf[pos + 5 + size] and cmd_rcv == expected_cmd:
                    self.buf = self.buf[pkt_end:]
                    return payload
                self.buf = self.buf[pos + 1:]
        return None

    # -- RC 保活线程 (后台持续发, 频率≥10Hz 防 FAILSAFE) ------------------------

    def _stream_task(self):
        """后台 ~55Hz 循环, 持续发 MSP_SET_RAW_RC

        源码依据:
          rxMspFrameReceive (rx/msp.c:61) → 设 rxMspFrameDone=true → rxSignalReceived
          needRxSignalMaxDelayUs = RXLOSS_TRIGGER_INTERVAL (默认 150ms)
          所以频率 > 1/150ms = ~7Hz 即可, 我们 55Hz 绰绰有余
        """
        nch = len(self._rc_data)
        while self._stream_alive:
            loop_start = time.monotonic()
            try:
                payload = struct.pack('<' + 'H' * nch, *self._rc_data)
                self._send_cmd(MSP_CMD.SET_RAW_RC, payload)
            except Exception:
                time.sleep(0.05)
                # 继续尝试发送, 不静默退出
            # 18ms 周期 → ~55Hz
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.002, 0.018 - elapsed))

    def stream_start(self):
        """开保活"""
        if not self._stream_alive:
            self._stream_alive = True
            self._stream_thread = threading.Thread(target=self._stream_task, daemon=True)
            self._stream_thread.start()

    def stream_stop(self):
        """关保活"""
        self._stream_alive = False
        if self._stream_thread:
            self._stream_thread.join(timeout=0.5)

    # -- RC 通道 ----------------------------------------------------------------

    def set_rc(self, roll=1500, pitch=1500, throttle=1000, yaw=1500,
               aux1=1500, aux2=1500, aux3=1500, aux4=1500):
        """设 8 通道 AETR (map AETR1234)
        [Roll, Pitch, Throttle, Yaw, AUX1..4]
        """
        self._rc_data = [roll, pitch, throttle, yaw, aux1, aux2, aux3, aux4]
        payload = struct.pack('<' + 'H' * 8, *self._rc_data)
        self._send_cmd(MSP_CMD.SET_RAW_RC, payload)

    def set_rc_dict(self, rc: dict):
        """便捷方法: set_rc(**{roll, pitch, throttle, yaw, aux1..4})"""
        self.set_rc(
            roll=rc.get('roll', 1500),
            pitch=rc.get('pitch', 1500),
            throttle=rc.get('throttle', 1000),
            yaw=rc.get('yaw', 1500),
            aux1=rc.get('aux1', 1500),
            aux2=rc.get('aux2', 1500),
            aux3=rc.get('aux3', 1500),
            aux4=rc.get('aux4', 1500),
        )

    # -- 遥测读取 ----------------------------------------------------------------

    def att(self):
        """读姿态 (roll°, pitch°, yaw° 0-359)"""
        p = self._send_cmd(MSP_CMD.GET_ATTITUDE)
        if p and len(p) >= 6:
            r, pi, y = struct.unpack_from('<hhh', p, 0)
            return (r / 10.0, pi / 10.0, float(y))
        return None

    def rc_channels(self):
        """读飞控端 rcData 值 (8 通道)"""
        p = self._send_cmd(MSP_CMD.GET_RC)
        if p and len(p) >= 16:
            return list(struct.unpack_from('<' + 'H' * min(len(p)//2, 8), p, 0))
        return None

    def arming_flags(self):
        """读解锁阻止标志 (armingDisableFlags_e 位掩码)

        MSP_STATUS_EX 响应格式 (BF 4.3+):
          [固定头部 15 字节] [1B ext_count] [ext_count B 扩展标志]
          [1B flag_count] [4B arming_flags] [1B config_state]

        按协议逐步解析, 不靠固定偏移.
        """
        p = self._send_cmd(MSP_CMD.GET_STATUS_EX)
        if not p or len(p) < 20:
            return None

        # 跳过固定头部 (15 bytes for STATUS_EX)
        off = 15
        # 跳过扩展飞行模式标志
        byte_count = p[off]; off += 1 + byte_count
        # 跳过 arming flag count byte
        off += 1
        # 读取 4 字节 arming flags
        if off + 4 > len(p):
            return None
        return struct.unpack_from('<I', p, off)[0]

    def flags_decode(self, val):
        """位掩码 → 名称列表"""
        names = []
        for bit, name in ARMING_FLAG_NAMES.items():
            if val & (1 << bit):
                names.append(name)
        return names

    def version(self):
        """固件版本"""
        v = self._send_cmd(MSP_CMD.FC_VARIANT)
        var = v.decode('ascii', 'replace').strip() if v else "?"
        pv = self._send_cmd(MSP_CMD.FC_VERSION)
        return (var, pv[0], pv[1], pv[2]) if pv and len(pv) >= 3 else None

    def close(self):
        self.stream_stop()
        self.ser.close()


# ---- 诊断 / 控制函数 ---------------------------------------------------------

def cmd_check():
    """三步诊断"""
    m = MSP()
    print("=" * 56)

    # 1. RC 到达确认
    print("A. RC 通道测试")
    # AETR send: Roll=1100, Pitch=1200, Thr=1300, Yaw=1400
    # rcData order: [Roll, Pitch, Yaw, Thr] → expect [1100, 1200, 1400, 1300]
    m.set_rc(1100, 1200, 1300, 1400, 1500, 1500, 1500, 1500)
    time.sleep(0.15)
    rc = m.rc_channels()
    if rc:
        for i, (n, v) in enumerate(zip(
            ["Roll","Pitch","Yaw","Thr","AUX1","AUX2","AUX3","AUX4"], rc)):
            bar = "█" * max(0, (v - 1000) // 50) if v > 1000 else ""
            print(f"  Ch{i+1} {n:>6}: {v:>5} {bar}")
        ok = all(abs(rc[i] - [1100, 1200, 1400, 1300][i]) < 50 for i in range(4))
        print(f"  通道映射: {'OK' if ok else '不对 (run: map AETR1234)'}")
    else:
        print("  ❌ 无响应, 检查连接")
        m.close(); return

    # 2. 保活 + 等 FAILSAFE 消除
    print("\nB. 保活 + 等 RX_FAILSAFE 消除")
    print("   (failsafe.c:188 需要 rxDataRecoveryPeriod ≈ 1s)")
    m.set_rc(1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)
    m.stream_start()

    cleared = False
    for i in range(30):  # 3 秒
        flags = m.arming_flags()
        if flags is not None:
            rx = bool(flags & 0x04)
            elapsed = (i + 1) * 0.1
            names = m.flags_decode(flags)
            other = [n for n in names if n not in ("NO_GYRO", "RX_FAILSAFE")]
            print(f"  {elapsed:.1f}s  RX_FAILSAFE={'⛔' if rx else '✅'}  flags=0x{flags:02X}{' ' + ' '.join(other) if other else ''}")
            if not rx:
                print(f"  ✅ FAILSAFE 已消除!")
                cleared = True
                break
        time.sleep(0.1)

    if not cleared:
        print("\n  ⛔ 3 秒后 FAILSAFE 仍在。请确认:")
        print("    1. Receiver Mode = MSP RX")
        print("    2. Betaflight 版本 ≥ 4.5")
        print("    3. Ports → USB VCP → MSP 勾选")

    # 3. ARM 测试
    print("\nC. ARM 测试")
    m.set_rc(1500, 1500, 1000, 1500, 2000, 1500, 1500, 1500)  # AUX1=2000
    time.sleep(0.3)
    flags = m.arming_flags()
    if flags is not None:
        names = m.flags_decode(flags)
        for n in names:
            e = "✅" if n in ("NO_GYRO",) else "⛔"
            print(f"  {e} {n}")
        if flags == 0:
            print("  ✅ 飞控已解锁!")
        elif (flags & 0x04) and cleared:
            print("  ⚠️  FAILSAFE 已消但 ARM 开关未检测, 确认 Modes→ARM→AUX1")
        elif flags & 0x04:
            print("  ⚠️  FAILSAFE 未消 — 物理接收机是唯一解决方案")

    m.close()


def cmd_arm():
    """解锁 (DISARM 预热 2s → 等 ARM_SWITCH 消除 → ARM)"""
    m = MSP()

    # Step 1: DISARM 状态启动保活, 让 FC 看到 ARM 开关在 OFF
    print("→ DISARM 预热 (2 秒)...")
    m.set_rc(1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)
    m.stream_start()

    for i in range(20):
        time.sleep(0.1)
        flags = m.arming_flags()
        if flags is not None and not (flags & 0x04):
            break

    # Step 2: 等 ARM_SWITCH 消除 (BF 检测到 DISARM)
    print("→ 等 ARM_SWITCH 消除...")
    for i in range(30):
        flags = m.arming_flags()
        if flags is not None:
            arm_switch = bool(flags & (1 << 23))
            if not arm_switch:
                print(f"  ✅ ARM_SWITCH 消除 (flags=0x{flags:08X})")
                break
        time.sleep(0.1)
    else:
        print("  ⚠️  ARM_SWITCH 未消, 但仍尝试 ARM")

    # Step 3: 解锁
    print("→ ARM (AUX1=2000)...")
    m.set_rc(1500, 1500, 1000, 1500, 2000, 1500, 1500, 1500)
    time.sleep(0.5)

    flags = m.arming_flags()
    if flags == 0:
        print("✅ 飞控已解锁! 电机应怠速旋转")
    else:
        print(f"flags=0x{flags:08X}  {m.flags_decode(flags)}")

    print("\n持续 ARM 中 (Ctrl+C 退出, 可手动推油门)")
    print("  在另一终端跑: python3 ~/BF_Anti/msp.py throttle")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n→ DISARM...")
        m.set_rc(1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)
        time.sleep(0.3)
        m.close()
        print("✅ 已上锁")


def cmd_throttle():
    """手动推油门 (需先 ARM)"""
    import select
    m = MSP()
    m.set_rc(1500, 1500, 1000, 1500, 2000, 1500, 1500, 1500)
    m.stream_start()

    print("油门控制: 输入 1000-2000, 回车确认")
    print("(1000=最低, 1200=慢转, 1500=悬停/中速)")
    print("输入 'q' 退出")

    thr = 1100
    try:
        while True:
            m.set_rc(1500, 1500, thr, 1500, 2000, 1500, 1500, 1500)
            print(f"油门={thr}  [输入新值或 q]")
            # 简单读输入
            line = ""
            try:
                line = input()
            except EOFError:
                break
            if line == 'q':
                break
            try:
                thr = int(line)
                thr = max(1000, min(2000, thr))
            except ValueError:
                pass
    except KeyboardInterrupt:
        pass
    m.set_rc(1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)
    m.close()
    print("✅ 已停止")


def cmd_monitor():
    """持续读姿态"""
    m = MSP()
    fmt = "{:>8.1f}°  {:>8.1f}°  {:>7.0f}°"
    print(f"{'Roll(横滚)':>10}   {'Pitch(俯仰)':>10}   {'Yaw(航向)':>10}")
    print("-" * 40)
    try:
        while True:
            a = m.att()
            if a:
                print(fmt.format(a[0], a[1], a[2]), end="\r")
            else:
                print("无响应", end="\r")
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n✅ 已停止")
    m.close()


def cmd_test():
    """快速通信测试 (3 次读姿态)"""
    m = MSP()
    ok = 0
    for i in range(3):
        a = m.att()
        if a:
            print(f"  [{i+1}] Roll={a[0]:6.1f}  Pitch={a[1]:6.1f}  Yaw={a[2]:6.1f}")
            ok += 1
        else:
            print(f"  [{i+1}] ❌")
        time.sleep(0.1)
    print(f"✅ {ok}/3" if ok == 3 else f"⚠️ {ok}/3")
    m.close()


def cmd_hover():
    """慢推油门测试 (需电池 + 螺旋桨已拆)"""
    print("⚠️  确保已拆桨! 确保电池已接!")
    input("Enter 继续...")
    m = MSP()
    m.stream_start()
    time.sleep(1.5)  # 等 FAILSAFE 消
    print("→ ARM")
    m.set_rc(1500, 1500, 1000, 1500, 2000, 1500, 1500, 1500)
    time.sleep(0.5)
    print("→ 缓慢推油门 1000→1300")
    for thr in range(1000, 1301, 10):
        m.set_rc(1500, 1500, thr, 1500, 2000, 1500, 1500, 1500)
        a = m.att()
        if a:
            print(f"  油门={thr}  Roll={a[0]:.1f}°  Pitch={a[1]:.1f}°  Yaw={a[2]:.1f}°")
        time.sleep(0.1)
    m.set_rc(1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)
    m.close()
    print("✅ 完成")


# ---- 入口 -------------------------------------------------------------------

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        cmd_test()
    else:
        cmd = args[0]
        if cmd == "arm":
            cmd_arm()
        elif cmd == "check":
            cmd_check()
        elif cmd == "monitor":
            cmd_monitor()
        elif cmd == "throttle":
            cmd_throttle()
        elif cmd == "test":
            cmd_test()
        else:
            print(f"未知: {cmd}")
            print("用法: msp_test.py [arm|check|monitor|hover|test]")
