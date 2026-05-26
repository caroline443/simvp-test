#!/bin/bash
# run_full.sh — 全量训练一键脚本
# 用法：
#   chmod +x run_full.sh
#   tmux new-session -d -s train 'bash run_full.sh'   # 后台启动，终端关闭不中断
#   tmux attach -t train                               # 随时查看进度
#
# 训练顺序：baseline → opsd → opsd_rw
# 日志：logs/<name>_<timestamp>.log
# 完成后自动关机

set -e  # 任意步骤失败立即退出（防止 OOM 后继续扣费）

CONFIG="configs/default.yaml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

run_step() {
    local name=$1
    shift
    local step_log="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    log "========== START: $name =========="
    log "log -> $step_log"
    # tee 同时写文件和终端（tmux 里可见）
    "$@" 2>&1 | tee "$step_log"
    local code=${PIPESTATUS[0]}
    if [ $code -ne 0 ]; then
        log "========== FAILED: $name (exit $code) =========="
        log "训练失败，跳过关机，请检查日志：$step_log"
        exit $code
    fi
    log "========== DONE: $name =========="
}

log "训练开始，config=$CONFIG"
log "主日志：$MAIN_LOG"

run_step "baseline"    python train_baseline.py --config "$CONFIG"
run_step "opsd"        python train_opsd.py     --config "$CONFIG"
run_step "opsd_rw"     python train_opsd.py     --config "$CONFIG" --reward_weight

log "所有训练完成，准备关机..."
/usr/bin/shutdown
