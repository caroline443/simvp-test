#!/bin/bash
# run_cbam.sh — Inception + CBAM 全量训练脚本
# 训练顺序：cbam_baseline → cbam_opsd → cbam_opsd_rw → 关机

set -e

CONFIG="configs/default_cbam.yaml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/cbam_run_$(date +%Y%m%d_%H%M%S).log"

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

log "Inception+CBAM 训练开始，config=$CONFIG"

run_step "cbam_baseline"  python -u train_baseline.py --config "$CONFIG"
run_step "cbam_opsd"      python -u train_opsd.py     --config "$CONFIG"
run_step "cbam_opsd_rw"   python -u train_opsd.py     --config "$CONFIG" --reward_weight

log "Inception+CBAM 训练完成，准备关机..."
/usr/bin/shutdown
