#!/usr/bin/env python3
"""
X5 LSIO pinmux register dumper v2 — byte-level decoding.

The LSIO iomuxc (0x34180000) packs per-pin config into bytes.
Each byte = 1 pin's mux + pull + drive config.

Target: find which byte controls UART1_RX (40-pin header Pin 10),
         then disable its pull-up.

Usage: sudo python3 scripts/x5_pinmux_tool_v2.py
"""

import sys
import os
import struct
import mmap
import signal

LSIO_BASE = 0x34180000
MAP_SIZE  = 0x1000  # 4KB (smaller to avoid bus error)


def handler(signum, frame):
    print("\n[SIGBUS] Unmapped address reached — end of register block", file=sys.stderr)
    sys.exit(0)


def decode_byte(b, pin_idx):
    """Decode one pin's config byte.  Layout (educated guess):
        bits 2-0: function (0=GPIO, 3=UART, ...)
        bits 4-3: pull (0=NONE, 1=UP, 2=DOWN)
        bit  5:   drive strength LSB
        bit  6:   schmitt trigger
        bit  7:   slew rate
    """
    func = b & 0x7
    pull = (b >> 3) & 0x3
    drive = (b >> 5) & 0x1
    schmitt = (b >> 6) & 0x1
    slew = (b >> 7) & 0x1
    func_names = {0: "GPIO", 3: "UART", 1: "I2C?", 2: "SPI?", 4: "I2S?",
                  5: "PWM?", 6: "SD?", 7: "?"}
    pull_names = {0: "NONE", 1: "UP", 2: "DOWN", 3: "BUS"}
    return (f"pin={pin_idx:3d} func={func_names.get(func, str(func)):5s} "
            f"pull={pull_names.get(pull, '???'):4s} "
            f"drv={drive} sch={schmitt} slew={slew}  [0x{b:02x}]")


def main():
    signal.signal(signal.SIGBUS, handler)

    try:
        fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    except PermissionError:
        print("ERROR: need root. Run with sudo.", file=sys.stderr)
        sys.exit(1)

    mem = mmap.mmap(fd, MAP_SIZE, mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=LSIO_BASE)
    os.close(fd)

    print("=== X5 LSIO Pinmux Register Dump (byte-level) ===")
    print(f"Base: 0x{LSIO_BASE:08x}")
    print()

    # Read entire 4KB, decode non-zero registers
    for offset in range(0, MAP_SIZE, 4):
        try:
            val = struct.unpack_from("<I", mem, offset)[0]
        except Exception:
            break
        if val == 0:
            continue
        b0 = val & 0xFF
        b1 = (val >> 8) & 0xFF
        b2 = (val >> 16) & 0xFF
        b3 = (val >> 24) & 0xFF
        print(f"0x{offset:04x}: 0x{val:08x}")
        for i, b in enumerate([b0, b1, b2, b3]):
            pin = (offset // 4) * 4 + i
            if b != 0:
                print(f"  [{i}] {decode_byte(b, pin)}")

    mem.close()


if __name__ == "__main__":
    main()
