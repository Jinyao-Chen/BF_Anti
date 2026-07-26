#!/usr/bin/env python3
"""全流程测试: 遥控器 ARM → X5 接管油门 → 遥控器收回.

测试流程:
  1. 遥控器 AUX1 推高位 (解锁)
  2. X5 检测到 ARM 后, 遥控器 AUX3 推高位 (开启 BOXMSPOVERRIDE)
  3. X5 接管 RPYT, 油门从 1050 缓慢爬升到 1150
  4. 遥控器 AUX3 拉回低位 → 遥控器收回控制权
  5. 遥控器 AUX1 拉回低位 → 上锁

⚠️ 拆桨! 接电池!
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from msp import MSP

THR_STEPS = list(range(1050, 1151, 5))  # 1050→1150, 慢爬
CH_NAMES = ["Roll", "Pitch", "Thr", "Yaw", "AUX1", "AUX2", "AUX3", "AUX4"]


def bar(val, lo=1000, hi=2000):
    w = int((val - lo) / (hi - lo) * 20)
    return "█" * w + "░" * (20 - w)


def show(ch, label=""):
    line = "  ".join(f"{CH_NAMES[i]}={ch[i]:>4} {bar(ch[i])}" for i in range(8))
    print(f"\r[{label}] {line}", end="")


def main():
    print("=" * 56)
    print("全流程测试: 遥控器 ARM → X5 接管油门")
    print("=" * 56)
    print()
    input("⚠️  确认螺旋桨已拆除! 按 Enter 继续...")
    print()

    m = MSP()
    m.telemetry_start()
    time.sleep(2)

    print("等待遥控器解锁 (AUX1 推高位)...")
    armed = False
    while not armed:
        ch, _ = m.rc_cached()
        if ch and ch[4] > 1700:
            armed = True
            print(f"\n✅ 已解锁! AUX1={ch[4]}")
        else:
            if ch:
                print(f"\rAUX1={ch[4]} (需要 >1700 解锁)", end="")
            time.sleep(0.15)

    # 发中立值, AUX 全透传
    for _ in range(10):
        ch, _ = m.rc_cached()
        if ch:
            m.set_rc(1500, 1500, 1000, 1500, ch[4], ch[5], ch[6], ch[7])
        time.sleep(0.02)

    print("\n等待 OVERRIDE (AUX3 推高位)...")
    override = False
    while not override:
        ch, _ = m.rc_cached()
        if ch and ch[6] > 1700:
            override = True
            print(f"\n✅ OVERRIDE 激活! AUX3={ch[6]}")
        else:
            if ch:
                show(ch, "等待 AUX3↑")
            time.sleep(0.1)

    # ── 接管开始 ────────────────────────────────────
    print("\nX5 接管 RPYT — 油门慢爬...")
    print("随时拨 AUX3 回低位即可收回控制权!")
    print()

    takeover = True
    thr_idx = 0
    last_show = 0
    t0 = time.monotonic()

    while takeover:
        now = time.monotonic()
        ch, _ = m.rc_cached()
        if ch is None:
            time.sleep(0.05)
            continue

        # 飞手收回 OVERRIDE?
        if ch[6] < 1300:
            takeover = False
            print(f"\n⚠️  飞手收回 OVERRIDE! AUX3={ch[6]}")
            break

        # 飞手收回 ARM?
        if ch[4] < 1300:
            takeover = False
            print(f"\n⚠️  飞手 DISARM! AUX1={ch[4]}")
            break

        # 油门爬升
        thr = THR_STEPS[min(thr_idx, len(THR_STEPS) - 1)]
        if now - t0 > (thr_idx + 1) * 0.5:  # 每 0.5s 升一档
            thr_idx += 1

        # 发送: RPYT 用接管值, AUX 透传飞手
        m.set_rc(1500, 1500, thr, 1500, ch[4], ch[5], ch[6], ch[7])

        if now - last_show > 0.3:
            pct = (thr - 1000) / 10
            print(f"  油门={thr} ({pct:.0f}%)  [AUX3={ch[6]} OVERRIDE]  "
                  f"按 Ctrl+C 或拨 AUX3 低位退出")
            last_show = now

        time.sleep(0.05)

    # ── 恢复 ────────────────────────────────────────
    print("\n恢复中立值...")
    for _ in range(10):
        ch, _ = m.rc_cached()
        if ch:
            m.set_rc(1500, 1500, 1000, 1500, ch[4], ch[5], 1500, ch[7])
        time.sleep(0.02)

    print("等待遥控器上锁 (AUX1 拉回低位)...")
    while True:
        ch, _ = m.rc_cached()
        if ch and ch[4] < 1300:
            print(f"✅ 已上锁! AUX1={ch[4]}")
            break
        time.sleep(0.15)

    m.close()
    print("测试完成")


if __name__ == "__main__":
    main()
