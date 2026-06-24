"""
SOTA Baseline 训练脚本（ConvLSTM / PredRNN / EarthFormer）
=============================================================
MSE 损失，一次性预测，与 train_vanilla.py 结构一致。

用法：
  python train_sota.py --model convlstm    --config configs/wadepre_align_convlstm.yaml
  python train_sota.py --model predrnn     --config configs/default_predrnn.yaml
  python train_sota.py --model earthformer --config configs/wadepre_align_earthformer.yaml
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import load_config, set_seed, save_checkpoint, load_checkpoint, AverageMeter, compute_all_metrics
from data.sevir_dataset import build_dataloaders
from models.convlstm import build_convlstm
from models.predrnn import build_predrnn
from models.earthformer_wrapper import build_earthformer


MODEL_BUILDERS = {
    "convlstm":    build_convlstm,
    "predrnn":     build_predrnn,
    "earthformer": build_earthformer,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, required=True, choices=["convlstm", "predrnn", "earthformer"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def weighted_mse_mae_loss(pred, target, vil_max, foreground_weight=5.0):
    """
    前景加权 MSE+MAE。
    VIL>=16（有回波）的像素权重 foreground_weight，其余为 1.0。
    """
    fg_thresh = 16.0 / vil_max
    weights = torch.ones_like(target)
    weights[target >= fg_thresh] = foreground_weight
    mse = (weights * (pred - target) ** 2).mean()
    mae = (weights * (pred - target).abs()).mean()
    return mse + mae


def train_one_epoch(model, loader, optimizer, device, cfg, epoch, model_name):
    model.train()
    loss_meter = AverageMeter("WMSEMAELoss")
    log_interval = cfg["training"]["log_interval"]
    vil_max = cfg["data"]["vil_max"]
    fg_weight = cfg["training"].get("foreground_weight", 5.0)

    for step, (input_frames, _target_bins, future_frames) in enumerate(loader):
        input_frames  = input_frames.to(device, non_blocking=True)
        future_frames = future_frames.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(input_frames)                  # [B, T_out, 1, H, W]
        loss = weighted_mse_mae_loss(pred, future_frames, vil_max, fg_weight)  # noqa

        if not torch.isfinite(loss):
            print(f"  [WARNING] Step {step+1}: loss={loss.item():.6f}，跳过该 batch")
            optimizer.zero_grad()
            continue

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"  [WARNING] Step {step+1}: grad_norm inf，跳过该 batch")
            optimizer.zero_grad()
            continue
        optimizer.step()

        loss_meter.update(loss.item(), input_frames.size(0))
        if (step + 1) % log_interval == 0:
            print(f"  [Epoch {epoch}] Step {step+1}/{len(loader)} | Loss: {loss_meter.val:.6f} (avg: {loss_meter.avg:.6f})")

    return loss_meter.avg


@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    loss_meter = AverageMeter("Val_WMSEMAELoss")
    vil_max = cfg["data"]["vil_max"]
    thresholds = cfg["eval"]["thresholds"]
    fg_weight = cfg["training"].get("foreground_weight", 5.0)

    all_pred_vil = []
    all_true_vil = []

    for input_frames, _target_bins, future_frames in loader:
        input_frames  = input_frames.to(device, non_blocking=True)
        future_frames = future_frames.to(device, non_blocking=True)

        pred = model(input_frames)
        loss = weighted_mse_mae_loss(pred, future_frames, vil_max, fg_weight)
        if torch.isfinite(loss):
            loss_meter.update(loss.item(), input_frames.size(0))  # noqa

        pred_vil = (pred.squeeze(2).cpu().numpy() * vil_max)
        true_vil = (future_frames.squeeze(2).cpu().numpy() * vil_max)
        all_pred_vil.append(pred_vil)
        all_true_vil.append(true_vil)

    all_pred_vil = np.concatenate(all_pred_vil, axis=0)
    all_true_vil = np.concatenate(all_true_vil, axis=0)
    metrics = compute_all_metrics(all_pred_vil, all_true_vil, thresholds)

    return loss_meter.avg, metrics


def main():
    args = parse_args()

    # 如果没有指定 config，使用默认路径
    if args.config is None:
        args.config = f"configs/default_{args.model}.yaml"

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Device] {device}")
    print(f"[Model] {args.model.upper()}")

    print("[Data] 加载 SEVIR VIL 数据集...")
    train_loader, val_loader, _ = build_dataloaders(cfg)

    print(f"[Model] 构建 {args.model.upper()} 模型...")
    model = MODEL_BUILDERS[args.model](cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 可训练参数量：{total_params / 1e6:.2f}M")

    train_cfg = cfg["training"]
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg["baseline_lr"], weight_decay=train_cfg["baseline_weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=train_cfg["baseline_epochs"], eta_min=train_cfg["baseline_lr"] * 0.01)

    start_epoch = 1
    best_val_csi = 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device=str(device))
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_csi = ckpt.get("best_val_csi", 0.0)

    ckpt_dir = train_cfg["baseline_ckpt_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n[Train] 开始训练，共 {train_cfg['baseline_epochs']} 个 Epoch")
    print("=" * 60)

    for epoch in range(start_epoch, train_cfg["baseline_epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch, args.model)
        val_loss, val_metrics = validate(model, val_loader, device, cfg)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"[Epoch {epoch:03d}/{train_cfg['baseline_epochs']}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s")

        thr_key = 74
        if thr_key in val_metrics:
            m = val_metrics[thr_key]
            print(f"  [Metrics @{thr_key}] CSI={m['CSI']:.4f} | POD={m['POD']:.4f} | FAR={m['FAR']:.4f} | HSS={m['HSS']:.4f}")

        val_csi = val_metrics.get(74, {}).get("CSI", 0.0)
        if val_csi > best_val_csi:
            best_val_csi = val_csi
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val_csi": best_val_csi, "val_metrics": val_metrics, "cfg": cfg},
                ckpt_dir, filename="best.pth",
            )
            print(f"  [Checkpoint] 保存最佳模型，CSI@74: {best_val_csi:.4f}")

        if epoch % train_cfg["save_interval"] == 0:
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_val_csi": best_val_csi, "cfg": cfg},
                ckpt_dir, filename=f"checkpoint_epoch{epoch:03d}.pth",
            )

        print("-" * 60)

    print(f"\n[Done] 训练完成！最佳 CSI@74: {best_val_csi:.4f}")


if __name__ == "__main__":
    main()
