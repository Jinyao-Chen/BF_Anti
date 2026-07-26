#!/usr/bin/env python3
"""CRSF 通道实时监控 — 无需飞控, 直读接收机 UART.

用法:
  python3 scripts/crsf_monitor.py /dev/ttyS3
  python3 scripts/crsf_monitor.py /dev/ttyS3 --baud 420000
"""

import sys
import time
import argparse
import threading

sys.path.insert(0, "BOARD_HOME/BF_Anti")
from crsf import CRSFParser
import serial

CH_NAMES = ["R", "P", "T", "Y", "A1", "A2", "A3", "A4",
            "CH9", "CH10", "CH11", "CH12", "CH13", "CH14", "CH15", "CH16"]

def bar(v, lo=1000, hi=2000, width=40):
    """视化条形: 1000-2000 → ─█────"""
    if v is None:
        return " " * width
    pos = int((v - lo) / (hi - lo) * (width - 1))
    pos = max(0, min(width - 1, pos))
    return "─" * pos + "█" + "─" * (width - pos - 1)

def main():
    p = argparse.ArgumentParser(description="CRSF 通道实时监控")
    p.add_argument("port", nargs="?", default="/dev/ttyS3")
    p.add_argument("--baud", type=int, default=420000)
    args = p.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.02)
    except Exception as e:
        print(f"无法打开 {args.port}: {e}")
        sys.exit(1)

    parser = CRSFParser()
    channels = None
    lock = threading.Lock()
    last_ok = 0
    last_bad = 0

    def reader():
        nonlocal channels
        while True:
            try:
                data = ser.read(64)
            except Exception:
                break
            if data and parser.feed(data):
                with lock:
                    channels = parser.channels

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    print(f"\n{'CRSF 通道监控':—^62}")
    print(f"端口: {args.port} @ {args.baud}")
    print(f"对频后动动摇杆, Ctrl+C 退出\n")

    try:
        while True:
            with lock:
                ch = list(channels) if channels else None
                f_ok = parser.frames_ok
                f_bad = parser.frames_bad

            now = time.monotonic()
            if f_ok > last_ok:
                last_ok = f_ok
                status = "✅"
            elif f_bad > last_bad:
                status = "❌ CRC"
            else:
                status = ""

            last_ok, last_bad = f_ok, f_bad

            if ch:
                # 主 8 通道 + AUX 标注
                print(f"\033[2J\033[H{status} 帧OK={f_ok}  CRC坏={f_bad}", flush=True)
                print()
                for i in range(8):
                    nm = CH_NAMES[i]
                    v = ch[i]
                    tag = ""
                    if   i == 2: tag = " ← 油门"
                    elif i == 3: tag = " ← 偏航"
                    elif i == 4: tag = " ← ARM(飞手)"
                    elif i == 5: tag = " ← ANGLE常开"
                    print(f"  {nm:>4}  {v:>5} {bar(v)}{tag}")
                print()
            else:
                print(f"\033[2J\033[H⏳ 等待数据... 帧OK={f_ok}  CRC坏={f_bad}", flush=True)
                print("\n  确认: 接收机已対频? TX/RX/GND 接线正确?")

            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print(f"\n退出. 共收到 {parser.frames_ok} 帧, CRC 坏 {parser.frames_bad} 帧")

if __name__ == "__main__":
    main()
