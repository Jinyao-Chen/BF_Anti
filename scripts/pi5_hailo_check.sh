#!/usr/bin/env bash
# pi5_hailo_check.sh - Hailo-8L / HailoRT health check, plus a short HEF smoke.
#
# Read-only for the system: it queries the PCIe device, driver, packages and
# Python binding, then optionally runs a precompiled HEF for a few frames to
# prove the runtime can execute a network. It does not install, flash, or
# modify anything.
#
# Usage:
#   bash scripts/pi5_hailo_check.sh [--hef PATH] [--no-run] [--root DIR]
#
# Exit codes: 0 = no FAIL, 1 = at least one FAIL, 2 = bad arguments.

set -u

PASS=0
FAIL=0
WARN=0
HEF=""
RUN_SMOKE=1
ROOT=""

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --hef)
      if [ $# -lt 2 ]; then
        printf 'error: --hef needs a path\n' >&2
        exit 2
      fi
      HEF="$2"
      shift 2
      ;;
    --no-run)
      RUN_SMOKE=0
      shift
      ;;
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

section "1. HailoRT CLI"
if have hailortcli; then
  ok "hailortcli: $(hailortcli -v 2>/dev/null | head -n 1)"
  CLI_VERSION=$(hailortcli -v 2>/dev/null | head -n 1 | awk '{print $3}')
  info "CLI version: $CLI_VERSION"
else
  bad "hailortcli missing"
  CLI_VERSION=""
fi

section "2. device discovery"
if have hailortcli; then
  SCAN=$(hailortcli scan 2>/dev/null | tr -d '\000')
  if printf '%s' "$SCAN" | grep -q 'Device:'; then
    ok "device: $(printf '%s' "$SCAN" | grep 'Device:' | slash)"
  else
    bad "no Hailo device in hailortcli scan"
  fi
  IDENT=$(timeout 15 hailortcli fw-control identify 2>/dev/null | tr -d '\000')
  if printf '%s' "$IDENT" | grep -q 'Firmware Version'; then
    ok "fw-control identify succeeded"
    printf '%s\n' "$IDENT" | sed -n '1,12p' | sed 's/^/       /'
  else
    bad "fw-control identify failed"
  fi
else
  SCAN=""
  IDENT=""
fi

FW_VERSION=$(printf '%s' "$IDENT" | awk -F': ' '/Firmware Version/ {print $2; exit}' | awk '{print $1}')
BOARD=$(printf '%s' "$IDENT" | awk -F': ' '/Board Name/ {print $2; exit}' | sed 's/[[:space:]]*$//')
ARCH=$(printf '%s' "$IDENT" | awk -F': ' '/Device Architecture/ {print $2; exit}' | sed 's/[[:space:]]*$//')
info "firmware=$FW_VERSION board=$BOARD arch=$ARCH"

section "3. version compatibility"
if [ -n "$CLI_VERSION" ] && [ -n "$FW_VERSION" ]; then
  if [ "$CLI_VERSION" = "$FW_VERSION" ]; then
    ok "HailoRT CLI/driver and firmware both $CLI_VERSION"
  else
    bad "HailoRT CLI $CLI_VERSION vs firmware $FW_VERSION (must match)"
  fi
else
  warn "cannot compare HailoRT versions (missing CLI or firmware)"
fi
info "packages: $(dpkg -l 2>/dev/null | awk '/^ii +hailort|^ii +hailo-tappas|^ii +python3-hailort/ {print $2, $3}' | slash)"

section "4. PCIe link and kernel driver"
if have lspci; then
  HAILO_PCI=$(lspci -Dnn 2>/dev/null | grep -i -E 'hailo|1e60' | head -n 1)
  if [ -n "$HAILO_PCI" ]; then
    ok "PCIe: $HAILO_PCI"
    BDF=$(printf '%s' "$HAILO_PCI" | awk '{print $1}')
    LSPCI_VV=$(lspci -vv -s "$BDF" 2>/dev/null)
    if printf '%s' "$LSPCI_VV" | grep -q 'Kernel driver in use: hailo'; then
      ok "kernel driver: hailo"
    else
      bad "kernel driver not bound (expected 'Kernel driver in use: hailo')"
    fi
    printf '%s' "$LSPCI_VV" | grep -E 'LnkCap:|LnkSta:' | sed 's/^/       /'
  else
    bad "no Hailo PCIe function in lspci"
  fi
else
  warn "lspci missing (apt install pciutils)"
fi
if have lsmod; then
  MOD=$(lsmod 2>/dev/null | grep -E '^hailo' | head -n 1)
  if [ -n "$MOD" ]; then
    ok "module: $MOD"
  else
    warn "hailo kernel module not listed in lsmod"
  fi
fi
if [ -e /dev/hailo0 ]; then
  ok "/dev/hailo0: $(stat -c '%a %U:%G' /dev/hailo0 2>/dev/null)"
  if [ -r /dev/hailo0 ] && [ -w /dev/hailo0 ]; then
    ok "/dev/hailo0 readable+writable by $(id -un)"
  else
    warn "/dev/hailo0 not writable by $(id -un)"
  fi
else
  bad "/dev/hailo0 missing"
fi

section "5. Python binding"
if python3 -c 'import hailo_platform' >/dev/null 2>&1; then
  PY_VER=$(python3 -c 'import hailo_platform; print(getattr(hailo_platform, "__version__", "unknown"))' 2>/dev/null)
  ok "hailo_platform import OK (version $PY_VER)"
else
  bad "hailo_platform import failed (install python3-hailort)"
fi

section "6. HEF smoke test"
if [ -z "$HEF" ]; then
  for candidate in "$ROOT"/model/*.hef /usr/share/hailo-models/yolov8s_h8l.hef; do
    if [ -f "$candidate" ]; then
      HEF="$candidate"
      break
    fi
  done
fi
if [ -z "$HEF" ]; then
  warn "no HEF found under $ROOT/model or /usr/share/hailo-models"
elif [ ! -f "$HEF" ]; then
  bad "HEF not found: $HEF"
else
  info "HEF: $HEF"
  HEF_INFO=$(timeout 30 hailortcli parse-hef "$HEF" 2>/dev/null)
  if [ -n "$HEF_INFO" ]; then
    ok "parse-hef: $(printf '%s' "$HEF_INFO" | head -n 1)"
    HEF_ARCH=$(printf '%s' "$HEF_INFO" | awk -F': ' '/Architecture HEF/ {print $2; exit}')
    info "HEF architecture: $HEF_ARCH"
    if [ -n "$ARCH" ] && [ -n "$HEF_ARCH" ] && [ "$ARCH" != "$HEF_ARCH" ]; then
      bad "HEF architecture $HEF_ARCH does not match device $ARCH"
    fi
  else
    bad "hailortcli parse-hef failed"
  fi
  if [ "$RUN_SMOKE" -eq 1 ]; then
    info "running 100 frames (this exercises the PCIe link and firmware)..."
    if SMOKE=$(timeout 120 hailortcli run "$HEF" --frames-count 100 \
        --dont-show-progress --measure-latency 2>&1); then
      ok "HEF run smoke passed"
      printf '%s\n' "$SMOKE" | grep -iE 'Latency|Inference result|Network group' | tail -n 6 | sed 's/^/       /'
    else
      bad "HEF run smoke failed"
      printf '%s\n' "$SMOKE" | tail -n 12 | sed 's/^/       /'
    fi
  else
    info "smoke run skipped (--no-run)"
  fi
fi

section "summary"
printf 'PASS=%d WARN=%d FAIL=%d\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  printf 'result: BLOCKED\n'
  exit 1
fi
printf 'result: Hailo runtime healthy\n'
exit 0
