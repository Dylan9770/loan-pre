#!/usr/bin/env bash
# ============================================================
#  贷款智能决策系统 — 停止脚本
#  使用: bash stop.sh
# ============================================================
G='\033[0;32m'; Y='\033[1;33m'; N='\033[0m'

PIDS=$(pgrep -f "service.flask.app" || true)
if [ -z "$PIDS" ]; then
    echo -e "${Y}!${N} 没有运行中的 Flask 进程"
    exit 0
fi

echo -e "${Y}→${N} 停止 Flask 进程: $PIDS"
echo "$PIDS" | xargs -r kill 2>/dev/null || true
sleep 2

# 还在的话强制杀
LEFT=$(pgrep -f "service.flask.app" || true)
if [ -n "$LEFT" ]; then
    echo -e "${Y}!${N} 强制终止: $LEFT"
    echo "$LEFT" | xargs -r kill -9 2>/dev/null || true
fi

rm -f /tmp/flask.pid
echo -e "${G}✓${N} Flask 已停止"
