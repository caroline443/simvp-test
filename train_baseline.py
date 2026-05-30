"""
Baseline 训练脚本
=================
第一阶段：使用标准交叉熵损失对 SimVP 进行全监督训练。

训练策略：
  - 自回归展开（On-Policy），学生每步用自己的预测值更新滑动窗口
  - 交叉熵损失对齐预测 Logit 与真实 bin 标签
  - AdamW 优化器 + CosineAnnealingLR 学习率调度

用法：
  python train_baseline.py --config configs/default.yaml
  python train_baseline.py --config configs/default.yaml --resume checkpoints/baseline/checkpoint.pth
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler  # noqa: F401 (AMP disabled)

from utils import (
    load_config, set_seed, save_checkpoint, load_checkpoint,
    AverageMeter, logits_to_vil, compute_all_metrics, visualize_prediction
)
from data.sevir_dataset import build_dataloaders
from models.simvp import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="SimVP Baseline Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--resume", type=str, default=None,
                        help="从指定 checkpoint 恢复训练")
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备，如 cuda:0 或 cpu（默认自动检测）")
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch):
    """执行一个 epoch 的训练，返回平均 loss。"""
    model.train()
    loss_meter = AverageMeter("CE_Loss")
    # reduction='none'：逐像素计算 CE，以便后续手动做加权平均
    criterion_none = nn.CrossEntropyLoss(reduction="none")

    log_interval = cfg["training"]["log_interval"]
    num_bins = cfg["model"]["num_bins"]
    foreground_weight = cfg["training"].get("foreground_weight", 5.0)

    for step, (input_frames, target_bins, _future_frames) in enumerate(loader):
        # input_frames:  [B, in_seq_len, 1, H, W]  float32
        # target_bins:   [B, out_seq_len, H, W]     int64
        # _future_frames: 本阶段不使用（OPSD 专用）
        input_frames = input_frames.to(device, non_blocking=True)
        target_bins = target_bins.to(device, non_blocking=True)

        optimizer.zero_grad()

        all_logits = model(input_frames, privileged_future=None)

        B, T_out, C_bins, H, W = all_logits.shape
        logits_flat = all_logits.view(B * T_out, C_bins, H, W)
        targets_flat = target_bins.view(B * T_out, H, W)

        has_echo = (targets_flat > 0).float()
        pixel_weights = 1.0 + has_echo * (foreground_weight - 1.0)

        ce_per_pixel = criterion_none(logits_flat, targets_flat)
        loss = (ce_per_pixel * pixel_weights).sum() / (pixel_weights.sum() + 1e-8)

        if not torch.isfinite(loss):
            print(f"  [WARNING] Step {step+1}: loss={loss.item():.4f}，跳过该 batch")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        loss_meter.update(loss.item(), B)

        if (step + 1) % log_interval == 0:
            print(
                f"  [Epoch {epoch}] Step {step+1}/{len(loader)} | "
                f"Loss: {loss_meter.val:.4f} (avg: {loss_meter.avg:.4f})"
            )

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device, cfg):
    """在验证集上评估，返回平均 loss 和核心气象指标。"""
    model.eval()
    loss_meter = AverageMeter("Val_CE_Loss")
    criterion = nn.CrossEntropyLoss()

    num_bins = cfg["model"]["num_bins"]
    vil_max = cfg["data"]["vil_max"]
    thresholds = cfg["eval"]["thresholds"]

    all_pred_vil = []
    all_true_vil = []

    for input_frames, target_bins, future_frames in loader:
        input_frames = input_frames.to(device, non_blocking=True)
        target_bins = target_bins.to(device, non_blocking=True)

        all_logits = model(input_frames, privileged_future=None)

        B, T_out, C_bins, H, W = all_logits.shape
        logits_flat = all_logits.view(B * T_out, C_bins, H, W)
        targets_flat = target_bins.view(B * T_out, H, W)
        loss = criterion(logits_flat, targets_flat)
        if torch.isfinite(loss):
            loss_meter.update(loss.item(), B)

        # 转换为 VIL 像素值用于气象指标计算
        pred_vil = logits_to_vil(all_logits, num_bins, vil_max)  # [B, T_out, H, W]
        # target_bins -> VIL 像素值
        bin_width = vil_max / num_bins
        true_vil = (target_bins.cpu().numpy().astype(float) + 0.5) * bin_width

        all_pred_vil.append(pred_vil)
        all_true_vil.append(true_vil)

    all_pred_vil = __import__("numpy").concatenate(all_pred_vil, axis=0)
    all_true_vil = __import__("numpy").concatenate(all_true_vil, axis=0)

    metrics = compute_all_metrics(all_pred_vil, all_true_vil, thresholds)

    return loss_meter.avg, metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    # 设备选择
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] 使用设备：{device}")

    # 数据加载
    print("[Data] 正在加载 SEVIR VIL 数据集...")
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # 模型构建
    print("[Model] 正在构建 SimVP 模型...")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 可训练参数量：{total_params / 1e6:.2f}M")

    # 优化器与调度器
    train_cfg = cfg["training"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["baseline_lr"],
        weight_decay=train_cfg["baseline_weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["baseline_epochs"],
        eta_min=train_cfg["baseline_lr"] * 0.01,
    )
    scaler = None  # AMP disabled: Inception FP16 overflow on large temporal_ch

    # 恢复训练
    start_epoch = 1
    best_val_loss = float("inf")
    best_val_csi  = 0.0           # val_loss 易受 AMP 精度影响变 NaN，改用 CSI@74 做 checkpoint 判据
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device=str(device))
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_csi  = ckpt.get("best_val_csi", 0.0)

    ckpt_dir = train_cfg["baseline_ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n[Train] 开始 Baseline 训练，共 {train_cfg['baseline_epochs']} 个 Epoch")
    print("=" * 60)

    for epoch in range(start_epoch, train_cfg["baseline_epochs"] + 1):
        t0 = time.time()

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)

        # 验证
        val_loss, val_metrics = validate(model, val_loader, device, cfg)

        scheduler.step()
        elapsed = time.time() - t0

        # 打印 epoch 摘要
        print(
            f"[Epoch {epoch:03d}/{train_cfg['baseline_epochs']}] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
        )

        # 打印关键气象指标（以 74 阈值为代表）
        thr_key = 74
        if thr_key in val_metrics:
            m = val_metrics[thr_key]
            print(
                f"  [Metrics @{thr_key}] "
                f"CSI={m['CSI']:.4f} | POD={m['POD']:.4f} | "
                f"FAR={m['FAR']:.4f} | HSS={m['HSS']:.4f}"
            )

        # 以 CSI@74 为判据保存最佳模型（比 val_loss 更稳定，不受 AMP NaN 影响）
        val_csi = val_metrics.get(74, {}).get("CSI", 0.0)
        if not torch.isfinite(torch.tensor(val_loss)):
            print(f"  [WARNING] Val Loss 为非有限值（{val_loss}），但仍按 CSI 判据保存")
        if val_csi > best_val_csi:
            best_val_csi  = val_csi
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_val_csi": best_val_csi,
                    "val_metrics": val_metrics,
                    "cfg": cfg,
                },
                ckpt_dir,
                filename="best.pth",
            )
            print(f"  [Checkpoint] 保存最佳模型，CSI@74: {best_val_csi:.4f}")

        # 定期保存 checkpoint
        if epoch % train_cfg["save_interval"] == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "cfg": cfg,
                },
                ckpt_dir,
                filename=f"checkpoint_epoch{epoch:03d}.pth",
            )

        print("-" * 60)

    if best_val_csi > 0.0:
        print(f"\n[Done] Baseline 训练完成！最佳 CSI@74: {best_val_csi:.4f}")
        print(f"       最佳模型已保存至：{os.path.join(ckpt_dir, 'best.pth')}")
    else:
        print(f"\n[Done] Baseline 训练完成！CSI 全程为 0，best.pth 未保存，请检查数据。")


if __name__ == "__main__":
    main()
