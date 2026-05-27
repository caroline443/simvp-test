#!/bin/bash
# run_mamba.sh — Mamba 全量训练脚本
# 训练顺序：mamba_baseline → mamba_opsd → mamba_opsd_rw
# 用法：
#   tmux new-session -d -s mamba 'bash run_mamba.sh'

set -e

CONFIG="configs/default_mamba.yaml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/mamba_run_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

run_step() {
    local name=$1
    shift
    local step_log="$LOG_DIR/${name}_$(date +%Y%m%d_%H%M%S).log"
    log "========== START: $name =========="
    "$@" 2>&1 | tee "$step_log"
    local code=${PIPESTATUS[0]}
    if [ $code -ne 0 ]; then
        log "========== FAILED: $name (exit $code) =========="
        exit $code
    fi
    log "========== DONE: $name =========="
}

log "Mamba 训练开始，config=$CONFIG"

run_step "mamba_baseline"  python -u train_baseline.py --config "$CONFIG"
run_step "mamba_opsd"      python -u train_opsd.py     --config "$CONFIG"
run_step "mamba_opsd_rw"   python -u train_opsd.py     --config "$CONFIG" --reward_weight

log "Mamba 训练完成，准备关机..."
/usr/bin/shutdown
