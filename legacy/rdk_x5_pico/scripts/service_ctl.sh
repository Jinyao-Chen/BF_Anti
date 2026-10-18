#!/bin/bash
# BF_Anti 服务管理脚本
#
#   ./scripts/service_ctl.sh print    # 只把渲染后的 systemd unit 打到 stdout（人工核对，不用 sudo）
#   ./scripts/service_ctl.sh install  # 安装并启用 (开机自启)
#   ./scripts/service_ctl.sh start    # 立即启动
#   ./scripts/service_ctl.sh stop     # 停止
#   ./scripts/service_ctl.sh status   # 查看状态
#   ./scripts/service_ctl.sh logs     # 查看日志
#   ./scripts/service_ctl.sh disable  # 禁用开机自启
#
# 占位符：__BASE_DIR__ -> 本仓库根目录；__BOARD_USER__ -> $BOARD_USER（缺省当前用户）。
# 修掉了旧版的坑：unit 的 ExecStart 用占位符，而安装脚本只替换了写死的板端家目录，
# 结果装出来的 ExecStart 是字面量 —— 服务永远起不来。

set -e
BASE="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE="bf_anti"
BOARD_USER="${BOARD_USER:-$(id -un)}"
UNIT_SRC="$BASE/scripts/bf_anti.service"
UNIT_TMP="/tmp/bf_anti.service"

_render() {
    sed -e "s|__BASE_DIR__|${BASE}|g" -e "s|__BOARD_USER__|${BOARD_USER}|g" "$UNIT_SRC"
}

_install() {
    echo "→ 安装 systemd 服务 (BOARD_USER=${BOARD_USER}, BASE=${BASE})..."
    _render > "$UNIT_TMP"
    if grep -qE "__BASE_DIR__|__BOARD_USER__" "$UNIT_TMP"; then
        echo "❌ 渲染失败：unit 里仍有未替换的占位符，已中止" >&2
        exit 1
    fi
    sudo cp "$UNIT_TMP" /etc/systemd/system/bf_anti.service
    sudo systemctl daemon-reload
    sudo systemctl enable bf_anti
    echo "✅ 已安装并启用开机自启"
    echo "   sudo systemctl start bf_anti   # 立即启动"
}

case "${1:-status}" in
  print)   _render ;;
  install) _install ;;
  start)   sudo systemctl start $SERVICE  ;;
  stop)    sudo systemctl stop $SERVICE   ;;
  restart) sudo systemctl restart $SERVICE ;;
  status)  systemctl status $SERVICE --no-pager ;;
  logs)    journalctl -u $SERVICE -f ;;
  disable) sudo systemctl disable $SERVICE ;;
  *)
    echo "用法: $0 {print|install|start|stop|restart|status|logs|disable}"
    ;;
esac
