#!/usr/bin/env python3
"""⚠️ 拆桨后运行: python3 ~/BF_Anti/motors.py"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from msp import MSP

BF411 = {0:"NO_GYRO",1:"FAILSAFE",2:"RX_FAILSAFE",3:"BAD_RX_RECOVERY",
    4:"BOXFAILSAFE",5:"RUNAWAY",6:"CRASH",7:"THROTTLE",8:"ANGLE",
    9:"BOOT_GRACE",10:"NOPREARM",11:"LOAD",12:"CALIBRATING",13:"CLI",
    14:"CMS_MENU",15:"BST",16:"MSP",17:"PARALYZE",18:"GPS",19:"RESC",
    20:"RPMFILTER",21:"REBOOT",22:"DSHOT_BITBANG",23:"ARM_SWITCH"}

# 油门台阶 (RC 值, 1000=最低, 1100=慢转)
THROTTLE_STEPS = [1050, 1060, 1070, 1080, 1090, 1100]

m = MSP()
print("⚠️  确认螺旋桨已拆除!")
input("按 Enter 继续...")

print("\n1/4 DISARM 预热...")
m.set_rc(1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)
m.stream_start()
time.sleep(2)

print("2/4 ARM...")
m.set_rc(1500, 1500, 1000, 1500, 2000, 1500, 1500, 1500)
time.sleep(0.5)

f = m.arming_flags()
flags = (f & 0x00FFFFFF) if f else None

if flags is None:
    print("⚠️ 无法连接 — 插 USB + 接电池")
elif flags == 0:
    print("3/4 ✅ 已解锁！慢推油门...")
    for thr in THROTTLE_STEPS:
        m.set_rc(1500, 1500, thr, 1500, 2000, 1500, 1500, 1500)
        pct = (thr - 1000) / 10
        print(f"   油门={thr} ({pct:.0f}%)")
        time.sleep(1.0)
    print("4/4 ✅ 完成")
else:
    blocks = [BF411.get(i, f"B{i}") for i in range(24) if flags & (1<<i)]
    print(f"⛔ 阻挡: {blocks}")

print("\n→ DISARM...")
m.set_rc(1500, 1500, 1000, 1500, 1500, 1500, 1500, 1500)
time.sleep(0.3)
m.close()
print("✅ 已上锁")
