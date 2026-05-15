#!/usr/bin/env bash
# ============================================================
#  贷款智能决策系统 — 一键启动脚本
#  使用: bash start.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="/tmp/flask.log"
PORT=5000

# ---- 颜色 ----
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; B='\033[0;34m'; N='\033[0m'

ok()    { echo -e "${G}✓${N} $1"; }
warn()  { echo -e "${Y}!${N} $1"; }
err()   { echo -e "${R}✗${N} $1"; }
info()  { echo -e "${B}→${N} $1"; }

echo "============================================================"
echo "  贷款智能决策系统启动中..."
echo "============================================================"

# ---- 1. 检查 MySQL ----
info "检查 MySQL 服务..."
if systemctl is-active --quiet mysql; then
    ok "MySQL 已在运行"
else
    warn "MySQL 未启动，尝试启动..."
    sudo systemctl start mysql
    sleep 2
    if systemctl is-active --quiet mysql; then
        ok "MySQL 启动成功"
    else
        err "MySQL 启动失败，请手动检查: sudo systemctl status mysql"
        exit 1
    fi
fi

# ---- 2. 检查模型制品 ----
info "检查模型文件..."
MISSING=0
for f in artifacts/default_model.joblib artifacts/fraud_model.joblib; do
    if [ ! -f "$f" ]; then
        err "缺少模型文件: $f"
        MISSING=1
    fi
done
if [ $MISSING -eq 1 ]; then
    err "模型文件缺失，请先训练: python3 run_decision_suite.py"
    exit 1
fi
ok "模型文件齐全 (default + fraud)"

# ---- 3. 杀掉已有的 Flask 进程 ----
info "清理已有 Flask 进程..."
fuser -k $PORT/tcp 2>/dev/null || true
sleep 2
ok "已清理旧进程"

# ---- 4. 启动 Flask ----
PYTHON="${SCRIPT_DIR}/venv/bin/python"
[ -x "$PYTHON" ] || PYTHON="python3"
info "启动 Flask 服务 ($PYTHON)..."
nohup "$PYTHON" -m service.flask.app > "$LOG_FILE" 2>&1 &
FLASK_PID=$!
echo "$FLASK_PID" > /tmp/flask.pid

# ---- 5. 等待启动 ----
info "等待服务就绪（最长 30 秒）..."
for i in $(seq 1 30); do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null | grep -q 200; then
        ok "Flask 已就绪 (耗时 ${i}s)"
        break
    fi
    if ! kill -0 $FLASK_PID 2>/dev/null; then
        err "Flask 启动失败，查看日志: tail -30 $LOG_FILE"
        tail -20 "$LOG_FILE"
        exit 1
    fi
    sleep 1
done

# ---- 6. 健康检查 ----
HEALTH=$(curl -s "http://localhost:$PORT/health" 2>/dev/null || echo "FAIL")
if echo "$HEALTH" | grep -q "ok"; then
    ok "健康检查通过: $HEALTH"
else
    err "健康检查失败: $HEALTH"
    tail -20 "$LOG_FILE"
    exit 1
fi

# ---- 7. 输出访问信息 ----
LAN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo ""
echo "============================================================"
ok "启动完成！"
echo "============================================================"
echo "  访问地址："
echo "    本机:   http://localhost:$PORT"
[ -n "$LAN_IP" ] && echo "    局域网: http://$LAN_IP:$PORT"
echo ""
echo "  进程信息："
echo "    Flask PID: $FLASK_PID  (已写入 /tmp/flask.pid)"
echo "    日志:      $LOG_FILE"
echo ""
echo "  常用命令："
echo "    查看日志:  tail -f $LOG_FILE"
echo "    停止服务:  bash stop.sh   (或 kill \$(cat /tmp/flask.pid))"
echo "============================================================"
