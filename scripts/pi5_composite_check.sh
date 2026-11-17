#!/usr/bin/env bash
# pi5_composite_check.sh — 检查树莓派 5 的复合视频（J7 / RP1 VEC）输出是否就绪
#
# 用法：bash scripts/pi5_composite_check.sh
# 退出码：0 = 就绪；1 = 有关键项未通过
#
# 只读检查，不修改系统文件。
# 实测依据（Pi 5 Model B Rev 1.1 / kernel 6.18.50+rpt-rpi-2712）：
#   - 驱动是 drm-rp1-vec（不是 vc4），制式要用 cmdline 的 video=Composite-1:PAL 指定
#   - vc4.tv_norm=PAL 在 Pi 5 上无效（vc4 模块不管复合输出）

set -u

PASS=0
FAIL=0
CONFIG=/boot/firmware/config.txt
CMDLINE=/boot/firmware/cmdline.txt

say()  { printf '%s\n' "$*"; }
ok()   { printf '[ OK ] %s\n' "$*"; PASS=$((PASS + 1)); }
bad()  { printf '[FAIL] %s\n' "$*"; FAIL=$((FAIL + 1)); }
warn() { printf '[WARN] %s\n' "$*"; }
info() { printf '[INFO] %s\n' "$*"; }

[ -f "$CONFIG" ] || CONFIG=/boot/config.txt
[ -f "$CMDLINE" ] || CMDLINE=/boot/cmdline.txt

say "== 1. config.txt =="
if [ -f "$CONFIG" ]; then
  if grep -qE '^[[:space:]]*dtoverlay=vc4-kms-v3d[^#]*composite' "$CONFIG"; then
    ok "vc4-kms-v3d,composite 已打开"
  else
    bad "缺少 dtoverlay=vc4-kms-v3d,composite"
  fi
  if grep -qE '^[[:space:]]*enable_tvout=1' "$CONFIG"; then
    ok "enable_tvout=1 已设置"
  else
    bad "缺少 enable_tvout=1（旗舰机型默认 0；它同时会关闭 HDMI 输出，机载正好）"
  fi
  if grep -qE '^[[:space:]]*dtoverlay=vec-gpio-pi5' "$CONFIG"; then
    info "检测到 vec-gpio-pi5：那是无 J7 机型（CM5 / Pi 500）的 R-2R 备用路线，Pi 5 Model B 不需要"
  fi
else
  bad "找不到 $CONFIG"
fi

say ""
say "== 2. cmdline.txt 制式 =="
if [ -f "$CMDLINE" ]; then
  VID=$(grep -oE 'video=Composite-[0-9]+:[A-Za-z0-9-]+' "$CMDLINE" | head -n 1)
  if [ -n "$VID" ]; then
    ok "$VID"
    case "$VID" in
      *:PAL)   info "PAL：720x576i@50，576 行，推荐" ;;
      *:NTSC)  info "NTSC：720x480i@59.94" ;;
      *)       info "非常规制式，确认眼镜能认" ;;
    esac
  else
    warn "未写 video=Composite-1:PAL，将按默认 NTSC(704x432i@59.98) 输出"
    FAIL=$((FAIL + 1))
  fi
  if grep -q 'vc4.tv_norm' "$CMDLINE"; then
    info "cmdline 里的 vc4.tv_norm 在 Pi 5 上无效，可删；改用 video=Composite-1:PAL"
  fi
else
  bad "找不到 $CMDLINE"
fi

say ""
say "== 3. 内核驱动 drm-rp1-vec =="
# dmesg 是环形缓冲，Hailo/PCIe 等日志会把它冲掉；不要再拿 dmesg 当判据。
if [ -d /sys/module/drm_rp1_vec ] || [ -d /sys/module/drm-rp1-vec ]; then
  ok "drm-rp1-vec 驱动已加载"
  if command -v modetest >/dev/null 2>&1; then
    if modetest -M drm-rp1-vec -p >/dev/null 2>&1; then
      ok "modetest 能打开 drm-rp1-vec 并读到当前时序"
    else
      warn "驱动在，但 modetest 打不开 drm-rp1-vec"
    fi
  else
    info "没有 modetest（apt install libdrm-tests）；跳过时序探测"
  fi
  info "dmesg 环缓冲会回卷，本脚本不再用 dmesg 作为判据"
else
  bad "没有 drm_rp1_vec 模块：composite 开关没生效或还没重启"
fi

say ""
say "== 4. DRM 连接器 =="
COMPOSITE=$(ls -d /sys/class/drm/card*-Composite-* 2>/dev/null | head -n 1)
if [ -n "$COMPOSITE" ]; then
  STATUS=$(cat "$COMPOSITE/status" 2>/dev/null)
  ENABLED=$(cat "$COMPOSITE/enabled" 2>/dev/null)
  ok "$COMPOSITE  status=$STATUS enabled=$ENABLED"
  info "可用模式：$(cat "$COMPOSITE/modes" 2>/dev/null | tr '\n' ' ')"
else
  bad "没有 card*-Composite-* 连接器：composite 没启用（看 config.txt 后重启）"
fi

say ""
say "== 5. 帧缓冲 /dev/fb0 =="
if [ -e /sys/class/graphics/fb0/virtual_size ]; then
  VS=$(cat /sys/class/graphics/fb0/virtual_size)
  BPP=$(cat /sys/class/graphics/fb0/bits_per_pixel)
  ok "fb0 $VS @ $BPP bpp"
  if [ -w /dev/fb0 ]; then
    info "/dev/fb0 对当前用户可写"
  else
    info "/dev/fb0 需要 root 或 video 组"
  fi
else
  bad "没有 /dev/fb0"
fi

say ""
say "== 6. 当前实际时序 =="
if command -v modetest >/dev/null 2>&1; then
  modetest -M drm-rp1-vec -p 2>/dev/null | sed -n '1,6p' | sed 's/^/       /'
  modetest -M drm-rp1-vec -c 2>/dev/null | grep -A1 'TV mode' | tail -1 | sed 's/^/       /'
else
  warn "没有 modetest（apt install libdrm-tests）"
fi

say ""
say "== 7. 下一步 =="
if [ "$FAIL" -eq 0 ]; then
  say "就绪。出测试彩条："
  say "  python3 scripts/pi5_composite_testcard.py"
  say "接线：J7 焊盘（HDMI 插座旁）信号孔 → 图传 Video In；另一孔 → 图传 GND；共地。"
  say "眼镜制式设成 PAL。"
else
  say "有 $FAIL 项未通过："
  say "  config.txt 末尾要有  enable_tvout=1  与  dtoverlay=vc4-kms-v3d,composite"
  say "  cmdline.txt 单行末尾要有  video=Composite-1:PAL"
  say "  然后 sudo reboot，再跑本脚本。"
fi

say ""
say "结果：通过 $PASS 项，未通过 $FAIL 项。"
if [ "$FAIL" -eq 0 ]; then exit 0; else exit 1; fi
