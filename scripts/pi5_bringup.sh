#!/usr/bin/env bash
# pi5_bringup.sh - read-only inventory for the Pi 5 onboard rig.
#
# Checks board/OS, undervoltage+throttle, CVBS boot config, DRM VEC connector,
# /dev/fb0 geometry, USB UVC camera, Hailo-8L, storage, network, Python deps
# and the installed repo layout. It never writes to the system and never
# reboots: every step is a read, a query, or an idempotent --help/--list call.
#
# Usage:
#   bash scripts/pi5_bringup.sh [--root DIR]
#
# Exit codes: 0 = no FAIL, 1 = at least one FAIL, 2 = bad arguments.
# The camera and goggles are external: a missing camera is reported as FAIL
# because the video chain cannot be verified without it, not because the Pi
# is broken.

set -u

PASS=0
FAIL=0
WARN=0
ROOT=""

usage() {
  sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --root)
      if [ $# -lt 2 ]; then
        printf 'error: --root needs a directory\n' >&2
        exit 2
      fi
      ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

if [ -z "$ROOT" ]; then
  ROOT=$(cd -- "$(dirname -- "$0")/.." && pwd)
fi

ok()      { printf '[ OK ] %s\n' "$*"; PASS=$((PASS + 1)); }
bad()     { printf '[FAIL] %s\n' "$*"; FAIL=$((FAIL + 1)); }
warn()    { printf '[WARN] %s\n' "$*"; WARN=$((WARN + 1)); }
info()    { printf '[INFO] %s\n' "$*"; }
section() { printf '\n== %s ==\n' "$*"; }
have()    { command -v "$1" >/dev/null 2>&1; }
slash()   { tr '\n' ';'; }

section "1. board and OS"
if [ -r /proc/device-tree/model ]; then
  MODEL=$(tr -d '\0' < /proc/device-tree/model)
  ok "model: $MODEL"
  case "$MODEL" in
    *"Raspberry Pi 5"*) : ;;
    *) warn "expected a Raspberry Pi 5 board" ;;
  esac
else
  warn "no /proc/device-tree/model (not a Raspberry Pi?)"
fi

if [ -r /etc/os-release ]; then
  PRETTY=$(grep -m1 '^PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '"')
  ok "os: $PRETTY"
else
  warn "no /etc/os-release"
fi
info "kernel: $(uname -r) $(uname -m)"
info "uptime: $(uptime -p 2>/dev/null || uptime)"

section "2. power / thermal / throttle"
if have vcgencmd; then
  ok "vcgencmd present: $(vcgencmd version 2>/dev/null | tail -n 1)"
  info "soc temp: $(vcgencmd measure_temp 2>/dev/null)"
  info "core volts: $(vcgencmd measure_volts core 2>/dev/null)"
  info "arm clock: $(vcgencmd measure_clock arm 2>/dev/null)"
  info "arm freq config: $(vcgencmd get_config arm_freq 2>/dev/null)"
  THROTTLE=$(vcgencmd get_throttled 2>/dev/null | cut -d= -f2)
  if [ "$THROTTLE" = "0x0" ]; then
    ok "no undervoltage / throttle flags"
  else
    warn "throttled=$THROTTLE (0x1 under-voltage now, 0x2 freq capped now, 0x4 throttled now; 0x10000+ = occurred since boot)"
  fi
else
  warn "vcgencmd missing (raspi-utils / libraspberrypi-bin)"
fi

section "3. CVBS boot configuration (read-only)"
CONFIG=/boot/firmware/config.txt
if [ ! -r "$CONFIG" ]; then
  CONFIG=/boot/config.txt
fi
if [ -r "$CONFIG" ]; then
  info "config: $CONFIG"
  if grep -qE '^[[:space:]]*dtoverlay=vc4-kms-v3d[^#]*composite' "$CONFIG"; then
    ok "dtoverlay=vc4-kms-v3d,composite configured"
  else
    bad "missing dtoverlay=vc4-kms-v3d,composite in $CONFIG"
  fi
  if grep -qE '^[[:space:]]*enable_tvout=1' "$CONFIG"; then
    ok "enable_tvout=1 configured"
  else
    bad "missing enable_tvout=1 in $CONFIG"
  fi
else
  bad "no $CONFIG"
fi

CMDLINE=/boot/firmware/cmdline.txt
if [ ! -r "$CMDLINE" ]; then
  CMDLINE=/boot/cmdline.txt
fi
if [ -r "$CMDLINE" ]; then
  VIDEO=$(grep -oE 'video=Composite-[0-9]+:[A-Za-z0-9-]+' "$CMDLINE" | head -n 1)
  if [ -n "$VIDEO" ]; then
    ok "cmdline: $VIDEO"
    case "$VIDEO" in
      *:PAL) info "PAL = 720x576i@50" ;;
      *:NTSC) info "NTSC = 720x480i@59.94, goggles must accept it" ;;
      *) info "non-standard TV norm; confirm the goggles lock" ;;
    esac
  else
    bad "missing video=Composite-1:PAL in $CMDLINE (defaults to NTSC)"
  fi
else
  bad "no $CMDLINE"
fi
info "this script does not modify boot files; see docs/sim2real/pi5_video.md"

section "4. DRM / RP1 VEC composite output"
COMPOSITE=""
for candidate in /sys/class/drm/card*-Composite-*; do
  if [ -d "$candidate" ]; then
    COMPOSITE="$candidate"
    break
  fi
done
if [ -n "$COMPOSITE" ]; then
  STATUS=$(cat "$COMPOSITE/status" 2>/dev/null)
  ENABLED=$(cat "$COMPOSITE/enabled" 2>/dev/null)
  MODES=$(tr '\n' ' ' < "$COMPOSITE/modes" 2>/dev/null)
  if [ "$STATUS" = "connected" ] && [ "$ENABLED" = "enabled" ]; then
    ok "$COMPOSITE connected/enabled"
  else
    bad "$COMPOSITE status=$STATUS enabled=$ENABLED"
  fi
  info "modes: $MODES"
else
  bad "no card*-Composite-* connector (composite overlay not active)"
fi

if have modetest; then
  if timeout 5 modetest -M drm-rp1-vec -p 2>/dev/null | grep -q '720x576'; then
    ok "modetest reports a 720x576 CRTC"
  else
    warn "modetest did not report 720x576 (try: modetest -M drm-rp1-vec -p)"
  fi
else
  warn "modetest missing (apt install libdrm-tests)"
fi

section "5. framebuffer /dev/fb0"
if [ -e /dev/fb0 ]; then
  FB_VS=$(cat /sys/class/graphics/fb0/virtual_size 2>/dev/null)
  FB_BPP=$(cat /sys/class/graphics/fb0/bits_per_pixel 2>/dev/null)
  FB_STRIDE=$(cat /sys/class/graphics/fb0/stride 2>/dev/null)
  FB_NAME=$(cat /sys/class/graphics/fb0/name 2>/dev/null)
  ok "fb0 geometry: $FB_VS, bpp=$FB_BPP, stride=$FB_STRIDE ($FB_NAME)"
  case "$FB_VS" in
    720,576) ok "PAL geometry 720x576" ;;
    *) warn "framebuffer is $FB_VS, not PAL 720x576" ;;
  esac
  if [ "$FB_BPP" = "32" ]; then
    ok "32 bpp BGRA path (as used by onboard/video/fbdev.py)"
  else
    warn "bpp=$FB_BPP; fbdev sink supports 16/24/32 but the OSD path is tuned for 32"
  fi
  if [ -w /dev/fb0 ]; then
    info "/dev/fb0 writable by $(id -un)"
  else
    warn "/dev/fb0 not writable by $(id -un); add the service user to the video group"
  fi
  FB_BLANK=$(cat /sys/class/graphics/fb0/blank 2>/dev/null)
  if [ "$FB_BLANK" = "0" ]; then
    ok "fb0 is unblanked"
  else
    warn "fb0 blank=$FB_BLANK (console blanking); FBDevSink re-asserts unblank on open and every ~10 s"
  fi
else
  bad "/dev/fb0 missing"
fi

section "6. USB UVC camera / V4L2"
UVC_COUNT=0
PLATFORM_COUNT=0
for devlink in /sys/class/video4linux/video*; do
  if [ ! -e "$devlink" ]; then
    continue
  fi
  NODE=$(basename "$devlink")
  NAME=$(cat "$devlink/name" 2>/dev/null)
  DEVPATH=$(readlink -f "$devlink/device" 2>/dev/null)
  case "$DEVPATH" in
    *usb*)
      UVC_COUNT=$((UVC_COUNT + 1))
      info "$NODE: $NAME [USB]"
      ;;
    *)
      PLATFORM_COUNT=$((PLATFORM_COUNT + 1))
      ;;
  esac
done
if [ "$PLATFORM_COUNT" -gt 0 ]; then
  info "$PLATFORM_COUNT platform video node(s): ISP/HEVC decode, not cameras"
fi
if [ "$UVC_COUNT" -gt 0 ]; then
  ok "$UVC_COUNT USB UVC video node(s) present"
else
  bad "no USB UVC camera; video chain cannot be verified (camera not connected)"
fi
if have v4l2-ctl; then
  info "v4l2-ctl devices: $(v4l2-ctl --list-devices 2>/dev/null | slash)"
else
  warn "v4l2-ctl missing (apt install v4l-utils)"
fi
if have lsusb; then
  info "lsusb: $(lsusb 2>/dev/null | slash)"
fi

section "7. Hailo-8L / HailoRT"
if have hailortcli; then
  ok "hailortcli: $(hailortcli -v 2>/dev/null | head -n 1)"
  HAILO_ID=$(timeout 15 hailortcli fw-control identify 2>/dev/null | tr -d '\000')
  if printf '%s' "$HAILO_ID" | grep -q 'Firmware Version'; then
    ok "device identified: $(printf '%s' "$HAILO_ID" | grep -E 'Firmware Version|Board Name|Device Architecture' | slash)"
  else
    bad "hailortcli fw-control identify failed"
  fi
  info "scan: $(hailortcli scan 2>/dev/null | tr -d '\000' | slash)"
else
  bad "hailortcli missing (install HailoRT)"
fi
if [ -e /dev/hailo0 ]; then
  ok "/dev/hailo0 present ($(stat -c '%a %U:%G' /dev/hailo0 2>/dev/null))"
else
  warn "/dev/hailo0 missing"
fi
if have lspci; then
  info "PCIe: $(lspci -nn 2>/dev/null | grep -i -E 'hailo|1e60' | slash)"
fi
if python3 -c 'import hailo_platform' >/dev/null 2>&1; then
  ok "python3 hailo_platform import OK"
else
  bad "python3 hailo_platform import failed"
fi

section "8. storage and memory"
info "disk: $(df -h / 2>/dev/null | tail -n 1)"
AVAIL_KB=$(df -Pk / 2>/dev/null | awk 'NR==2 {print $4}')
if [ -n "$AVAIL_KB" ] && [ "$AVAIL_KB" -lt 1048576 ]; then
  warn "less than 1 GiB free on / (logs and HEF files need headroom)"
else
  ok "root filesystem has room"
fi
info "memory: $(free -h 2>/dev/null | grep -E 'Mem|Swap' | slash)"
if [ -r /sys/class/thermal/thermal_zone0/temp ]; then
  info "thermal_zone0: $(( $(cat /sys/class/thermal/thermal_zone0/temp) / 1000 )) C"
fi

section "9. network"
info "addresses: $(ip -brief addr 2>/dev/null | slash)"
ROUTE=$(ip route 2>/dev/null | grep -m1 '^default' || true)
if [ -n "$ROUTE" ]; then
  ok "default route: $ROUTE"
else
  bad "no IPv4 default route"
fi
if [ -r /etc/resolv.conf ]; then
  DNS=$(grep -E '^nameserver' /etc/resolv.conf 2>/dev/null | slash)
  if [ -n "$DNS" ]; then
    info "dns: $DNS"
  else
    warn "no nameserver in /etc/resolv.conf"
  fi
else
  warn "no /etc/resolv.conf"
fi

section "10. Python dependencies"
PYTHON=python3
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON="$ROOT/.venv/bin/python"
  ok "using venv: $ROOT/.venv"
else
  info "no venv at $ROOT/.venv (scripts/pi5_install.sh creates it); using system python"
fi
info "python: $("$PYTHON" -V 2>&1)"
for module in numpy cv2 serial hailo_platform; do
  if "$PYTHON" -c "import $module" >/dev/null 2>&1; then
    ok "python import $module"
  else
    case "$module" in
      serial) bad "python import serial failed (install python3-serial / pyserial)" ;;
      hailo_platform) bad "python import hailo_platform failed (install python3-hailort)" ;;
      *) bad "python import $module failed" ;;
    esac
  fi
done
info "versions: $("$PYTHON" -c 'import numpy, cv2; print("numpy", numpy.__version__, "cv2", cv2.__version__)' 2>/dev/null)"

section "11. repository and service layout"
for rel in onboard/app.py onboard/camera.py onboard/video/pipeline.py config/onboard.example.json scripts/pi5_run_onboard.sh; do
  if [ -e "$ROOT/$rel" ]; then
    ok "repo: $rel"
  else
    warn "repo: missing $rel under $ROOT"
  fi
done
if [ -f /etc/systemd/system/bf-anti-onboard.service ]; then
  ok "systemd unit installed"
  info "unit: $(systemctl is-enabled bf-anti-onboard.service 2>/dev/null) / $(systemctl is-active bf-anti-onboard.service 2>/dev/null)"
else
  info "systemd unit not installed (scripts/pi5_install.sh --service)"
fi

section "summary"
printf 'PASS=%d WARN=%d FAIL=%d\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf 'result: BLOCKED (%d critical check(s) failed)\n' "$FAIL"
  exit 1
fi
printf 'result: READY for hardware bring-up\n'
exit 0
