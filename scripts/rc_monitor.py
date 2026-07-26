#!/usr/bin/env python3
"""RC 通道实时监控 — 看哪个通道在动."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msp import MSP

m = MSP()
m.telemetry_start()
time.sleep(2)

print("拨动摇杆看数值变化 (Ctrl+C 退出)")
print(f"{'Roll':>7} {'Pitch':>7} {'Thr':>7} {'Yaw':>7}  {'A1':>5} {'A2':>5} {'A3':>5} {'A4':>5}")
print("-" * 64)

try:
    while True:
        ch, _ = m.rc_cached()
        if ch:
            print(f"{ch[0]:7} {ch[1]:7} {ch[2]:7} {ch[3]:7}  {ch[4]:5} {ch[5]:5} {ch[6]:5} {ch[7]:5}", end="\r")
        time.sleep(0.08)
except KeyboardInterrupt:
    print("\n停止")
finally:
    m.close()
