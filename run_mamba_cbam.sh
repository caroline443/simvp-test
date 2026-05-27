#!/bin/bash
# run_mamba_cbam.sh — Mamba + CBAM 全量训练脚本
# 训练顺序：mamba_cbam_baseline → mamba_cbam_opsd → mamba_cbam_opsd_rw → 关机

set -e

CONFIG="configs/default_mamba_cbam.yaml"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/mamba_cbam_run_$(date +%Y%m%d_%H%M%S).log"

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

log "Mamba+CBAM 训练开始，config=$CONFIG"

run_step "mamba_cbam_baseline"  python -u train_baseline.py --config "$CONFIG"
run_step "mamba_cbam_opsd"      python -u train_opsd.py     --config "$CONFIG"
run_step "mamba_cbam_opsd_rw"   python -u train_opsd.py     --config "$CONFIG" --reward_weight

log "Mamba+CBAM 训练完成，准备关机..."
/usr/bin/shutdown
