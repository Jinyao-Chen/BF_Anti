#!/usr/bin/env bash
# pi5_enable_persistent_journal.sh - keep journald logs across reboots.
#
# Ubuntu / Raspberry Pi OS ship journald with Storage=auto, so logs live only
# in /run/log/journal when /var/log/journal is absent.  That is the usual
# reason "there were no logs after a reboot".  This script creates the
# persistent journal directory and pins the setting in a drop-in.
#
# Usage:
#   sudo bash scripts/pi5_enable_persistent_journal.sh [--max-use 512M]
#
# Safe to re-run.  Does not touch the onboard config or the service.

set -euo pipefail

MAX_USE=${BF_ANTI_JOURNAL_MAX:-512M}
while [ $# -gt 0 ]; do
  case "$1" in
    --max-use) MAX_USE="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'error: unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [ "$(id -u)" -ne 0 ]; then
  printf 'error: must run as root (sudo)\n' >&2
  exit 1
fi

DEST=/etc/systemd/journald.conf.d/99-bf-anti-persistent.conf
install -d -m 0755 /etc/systemd/journald.conf.d
cat > "$DEST" <<EOF
# Installed by BF_Anti scripts/pi5_enable_persistent_journal.sh
[Journal]
Storage=persistent
SystemMaxUse=$MAX_USE
EOF
chmod 0644 "$DEST"

# Creates /var/log/journal with the ownership and mode journald expects.
systemd-tmpfiles --create --prefix /var/log/journal
systemctl restart systemd-journald
journalctl --flush >/dev/null 2>&1 || true

if [ -d /var/log/journal ]; then
  printf '[ OK ] persistent journal enabled (%s, max use %s)\n' "$DEST" "$MAX_USE"
  printf '       verify: journalctl --list-boots | tail -3\n'
else
  printf '[FAIL] /var/log/journal still missing after tmpfiles --create\n' >&2
  exit 1
fi
