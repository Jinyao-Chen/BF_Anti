#!/usr/bin/env bash
# pi5_network_check.sh - read-only network / link diagnostics for the aircraft.
#
# Prints interfaces, addresses, default route, DNS, SSH listener, time sync and
# (optionally) gateway reachability. It never changes NetworkManager state,
# never joins a network, and contains no SSID or credential.
#
# Usage:
#   bash scripts/pi5_network_check.sh [--iface NAME] [--timeout SEC]
#
# Exit codes: 0 = default route and DNS present, 1 = a critical check failed,
# 2 = bad arguments. Gateway ICMP failure is a WARN (ICMP is often filtered).

set -u

IFACE=""
TIMEOUT=2
FAIL=0
WARN=0
PASS=0

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
  case "$1" in
    --iface)
      if [ $# -lt 2 ]; then
        printf 'error: --iface needs a name\n' >&2
        exit 2
      fi
      IFACE="$2"
      shift 2
      ;;
    --timeout)
      if [ $# -lt 2 ]; then
        printf 'error: --timeout needs seconds\n' >&2
        exit 2
      fi
      TIMEOUT="$2"
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

ok()   { printf '[ OK ] %s\n' "$*"; PASS=$((PASS + 1)); }
bad()  { printf '[FAIL] %s\n' "$*"; FAIL=$((FAIL + 1)); }
warn() { printf '[WARN] %s\n' "$*"; WARN=$((WARN + 1)); }
info() { printf '[INFO] %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
slash() { tr '\n' ';'; }

printf '== interfaces ==\n'
if have ip; then
  ip -brief addr 2>/dev/null | sed 's/^/  /'
else
  bad "iproute2 missing"
fi
if [ -n "$IFACE" ]; then
  if ! ip link show "$IFACE" >/dev/null 2>&1; then
    bad "requested interface not found: $IFACE"
  else
    ok "requested interface present: $IFACE"
  fi
fi

printf '\n== routes ==\n'
ROUTE=$(ip route 2>/dev/null | grep -m1 '^default' || true)
if [ -n "$ROUTE" ]; then
  ok "default route: $ROUTE"
  GATEWAY=$(printf '%s' "$ROUTE" | awk '{print $3}')
else
  bad "no IPv4 default route"
  GATEWAY=""
fi
ip route 2>/dev/null | sed 's/^/  /'

printf '\n== DNS and identity ==\n'
if [ -r /etc/resolv.conf ]; then
  NS=$(grep -E '^nameserver' /etc/resolv.conf 2>/dev/null | slash)
  if [ -n "$NS" ]; then
    ok "nameservers: $NS"
  else
    warn "no nameserver in /etc/resolv.conf"
  fi
  grep -E '^(nameserver|search|options)' /etc/resolv.conf 2>/dev/null | sed 's/^/  /'
else
  warn "no /etc/resolv.conf"
fi
info "hostname: $(hostname 2>/dev/null)"
info "kernel: $(uname -r)"
if have hostname; then
  LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  if [ -n "$LOCAL_IP" ]; then
    ok "local address: $LOCAL_IP"
  else
    warn "no IPv4 address on any interface"
  fi
fi
if have timedatectl; then
  SYNC=$(timedatectl show -p NTPSynchronized --value 2>/dev/null)
  info "time: $(date -Is) ntp_synchronized=$SYNC"
  if [ "$SYNC" = "yes" ]; then
    ok "clock synchronised"
  else
    warn "clock not NTP-synchronised (matters for logs, not for flight)"
  fi
fi

printf '\n== listeners ==\n'
if have ss; then
  ss -ltn 2>/dev/null | sed 's/^/  /'
  if ss -ltn 2>/dev/null | grep -q ':22 '; then
    ok "SSH listener on :22"
  else
    warn "no SSH listener on :22"
  fi
else
  warn "ss missing (iproute2)"
fi

if [ -n "$GATEWAY" ] && have ping; then
  printf '\n== gateway reachability ==\n'
  if ping -c 2 -W "$TIMEOUT" "$GATEWAY" >/dev/null 2>&1; then
    ok "gateway $GATEWAY responds to ICMP"
  else
    warn "gateway $GATEWAY does not answer ICMP (may be filtered)"
  fi
fi

if have nmcli; then
  printf '\n== NetworkManager (read-only) ==\n'
  nmcli -t -f DEVICE,STATE,CONNECTION device 2>/dev/null | sed 's/^/  /' || true
fi

printf '\n== summary ==\n'
printf 'PASS=%d WARN=%d FAIL=%d\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
exit 0
