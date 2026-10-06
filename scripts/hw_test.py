#!/usr/bin/env python3
"""硬件链路诊断: 发送 1kHz 方波, 眼镜应看到明暗交替竖条"""
import spidev

spi = spidev.SpiDev()
spi.open(1, 0)
spi.max_speed_hz = 1_000_000
spi.mode = 0

low = bytes([0x00] * 500)
high = bytes([0xFF] * 500)
wave = low + high

print("Pin19 → 电位器 → FC VI, 方波发送中...")
print("边跑边拧电位器, 眼镜应该看到明暗竖条")
print("Ctrl+C 停止")

try:
    while True:
        spi.writebytes2(wave * 20)
except KeyboardInterrupt:
    pass

spi.close()
print("Done")
