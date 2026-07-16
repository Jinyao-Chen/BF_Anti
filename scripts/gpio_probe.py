#!/usr/bin/env python3
"""需要 sudo 运行: 查看 GPIO 映射并找出可用的 CS 引脚"""
import os, time, sys

print("=" * 55)
print("GPIO 排针映射探测")
print("=" * 55)

# 1. 显示所有 GPIO 状态
print("\n--- 完整 GPIO 状态 ---")
ret = os.system("cat /sys/kernel/debug/gpio 2>/dev/null | head -50")
if ret != 0:
    print("(无法读取 debug gpio)")

# 2. 尝试导出 40-pin 排针上可能用到的 GPIO
print("\n--- 尝试导出新 GPIO ---")
exported = []
for gpio in range(411, 450):
    path = f"/sys/class/gpio/gpio{gpio}"
    if os.path.exists(path):
        continue
    try:
        with open("/sys/class/gpio/export", "w") as f:
            f.write(str(gpio))
        time.sleep(0.05)
        exported.append(gpio)
    except Exception as e:
        pass

print(f"成功导出 {len(exported)} 个新 GPIO:")
for g in exported:
    path = f"/sys/class/gpio/gpio{g}"
    try:
        label = open(f"{path}/label").read().strip()
        print(f"  GPIO {g} ({label})")
    except:
        print(f"  GPIO {g}")

# 3. 列出所有已导出的 GPIO
print("\n--- 所有已导出的 GPIO ---")
for entry in sorted(os.listdir("/sys/class/gpio")):
    if entry.startswith("gpio") and entry != "gpiochip" and "chip" not in entry:
        path = f"/sys/class/gpio/{entry}"
        try:
            direction = open(f"{path}/direction").read().strip()
            value = open(f"{path}/value").read().strip()
            label = open(f"{path}/label").read().strip()
            print(f"  {entry}: dir={direction} val={value} label={label}")
        except:
            pass

print(f"\n下一步: 逐个翻转这些 GPIO，用万用表找到 Pin 26 对应的编号")
