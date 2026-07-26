#!/bin/bash
# BF_Anti 服务管理脚本
#
#   ./scripts/service_ctl.sh install    # 安装并启用 (开机自启)
#   ./scripts/service_ctl.sh start      # 立即启动
#   ./scripts/service_ctl.sh stop       # 停止
#   ./scripts/service_ctl.sh status     # 查看状态
#   ./scripts/service_ctl.sh logs       # 查看日志
#   ./scripts/service_ctl.sh disable    # 禁用开机自启

set -e
BASE="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE="bf_anti"

_install() {
    echo "→ 安装 systemd 服务..."
    cp "$BASE/scripts/bf_anti.service" /tmp/bf_anti.service
    sed -i "s|BOARD_HOME/BF_Anti|$BASE|g" /tmp/bf_anti.service
    sudo cp /tmp/bf_anti.service /etc/systemd/system/bf_anti.service
    sudo systemctl daemon-reload
    sudo systemctl enable bf_anti
    echo "✅ 已安装并启用开机自启"
    echo "   sudo systemctl start bf_anti   # 立即启动"
}

case "${1:-status}" in
  install) _install ;;
  start)   sudo systemctl start $SERVICE  ;;
  stop)    sudo systemctl stop $SERVICE   ;;
  restart) sudo systemctl restart $SERVICE ;;
  status)  systemctl status $SERVICE --no-pager ;;
  logs)    journalctl -u $SERVICE -f ;;
  disable) sudo systemctl disable $SERVICE ;;
  *)
    echo "用法: $0 {install|start|stop|restart|status|logs|disable}"
    ;;
esac
