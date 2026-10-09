#!/usr/bin/env python3
"""
X5 LSIO pinmux register dumper & pull-up disabler.

（已被同目录 x5_pinmux_tool_v2.py 的字节级解码版取代，此处保留供对照。）

The LSIO iomuxc at 0x34180000 controls the 40-pin header GPIOs.
Each pin has a 4-byte config register. This tool:
  1. Dumps the first 64 registers so we can find the UART1 RX pin
  2. Optionally writes a modified value to disable pull-up

Usage:
  sudo python3 scripts/x5_pinmux_tool.py                # dump registers
  sudo python3 scripts/x5_pinmux_tool.py --fix <reg> <value>  # write fix

Based on D-Robotics X5 pinmux: each register = [func(3):pull(2):drive(3):...]
"""

import sys
import os
import struct
import argparse
import mmap

LSIO_BASE = 0x34180000
MAP_SIZE  = 0x10000   # 64KB window
REG_COUNT = 64        # first 64 pin config registers


def read_pinmux(mem, offset):
    """Read 32-bit register at given offset from LSIO base."""
    return struct.unpack_from("<I", mem, offset)[0]


def write_pinmux(mem, offset, value):
    """Write 32-bit register."""
    struct.pack_into("<I", mem, offset, value)


def fmt_reg(val):
    """Pretty-print a pinmux config register."""
    func = val & 0x7
    pull = (val >> 3) & 0x3
    drive = (val >> 5) & 0x7
    schmitt = (val >> 8) & 0x1
    slew = (val >> 9) & 0x1
    pull_name = {0: "NONE", 1: "UP  ", 2: "DOWN", 3: "BUS "}.get(pull, "??? ")
    func_name = {0: "GPIO   ", 1: "I2C    ", 2: "SPI    ", 3: "UART   ",
                 4: "I2S    ", 5: "PWM    ", 6: "SD/MMC ", 7: "CAN    "}.get(func, f"FN{func}")
    return f"func={func_name} pull={pull_name} drive={drive} schmitt={schmitt} slew={slew}"


def main():
    p = argparse.ArgumentParser(description="X5 LSIO pinmux register tool")
    p.add_argument("--fix", type=int, nargs=2, metavar=("REG_OFFSET", "NEW_VALUE"),
                   help="Write hex value to register offset (e.g. --fix 0x10 0x81)")
    p.add_argument("--no-pullup", type=int, metavar="REG_OFFSET",
                   help="Find and disable pull-up on a specific register")
    opt = p.parse_args()

    try:
        fd = os.open("/dev/mem", os.O_RDWR | os.O_SYNC)
    except PermissionError:
        print("ERROR: need root. Run with sudo.", file=sys.stderr)
        sys.exit(1)

    mem = mmap.mmap(fd, MAP_SIZE, mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=LSIO_BASE)
    os.close(fd)

    if opt.fix:
        off, val = opt.fix
        old = read_pinmux(mem, off)
        write_pinmux(mem, off, val)
        print(f"Wrote register 0x{off:04x}: 0x{old:08x} → 0x{val:08x}")
        print(f"  Old: {fmt_reg(old)}")
        print(f"  New: {fmt_reg(val)}")
        mem.close()
        return

    if opt.no_pullup is not None:
        off = opt.no_pullup
        old = read_pinmux(mem, off)
        new = old & ~(0x3 << 3)  # clear pull bits
        write_pinmux(mem, off, new)
        print(f"Register 0x{off:04x}: pull-up DISABLED")
        print(f"  0x{old:08x} → 0x{new:08x}")
        print(f"  {fmt_reg(old)} → {fmt_reg(new)}")
        mem.close()
        return

    # Default: dump first 64 registers
    print(f"{'Offset':>8}  {'Raw':>10}  {'Func':8}  Pull   Drive  Schmitt  Slew")
    print("-" * 72)
    for i in range(REG_COUNT):
        off = i * 4
        val = read_pinmux(mem, off)
        if val == 0:
            continue  # skip unused pins
        print(f"0x{off:06x}  0x{val:08x}  {fmt_reg(val)}")

    mem.close()


if __name__ == "__main__":
    main()
