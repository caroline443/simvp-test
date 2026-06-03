#!/bin/bash
# run_fullres.sh — 384×384 全分辨率对比实验
# 训练顺序：ConvLSTM baseline → Mamba baseline → Mamba OPSD-RW → 关机
# 用于与 EarthFormer 等 SOTA 做公平对比

set -e

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/fullres_run_$(date +%Y%m%d_%H%M%S).log"

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

log "384×384 全分辨率训练开始"

# Step 1: ConvLSTM baseline（用于校验实现，对齐 EarthFormer 论文数字）
run_step "fullres_convlstm" \
    python -u train_sota.py --model convlstm --config configs/fullres_convlstm.yaml

# Step 2: Mamba baseline
run_step "fullres_mamba_baseline" \
    python -u train_baseline.py --config configs/fullres_mamba_opsd_rw.yaml

# Step 3: Mamba OPSD-RW（从 baseline 热启动）
run_step "fullres_mamba_opsd_rw" \
    python -u train_opsd.py --config configs/fullres_mamba_opsd_rw.yaml --reward_weight

log "384×384 训练完成，准备关机..."
/usr/bin/shutdown
