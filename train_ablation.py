"""
OPSD 消融实验训练脚本
====================
每次只改 OPSD 的一个组件，其余保持默认，用于填写论文消融表。

消融变体（--ablation）：
  standard        λ_KL=1.0  λ_CE=0.5  T=2.0  fg=5.0  （完整 OPSD，作为对照）
  no_fg_weight    λ_KL=1.0  λ_CE=0.5  T=2.0  fg=1.0  去掉前景加权
  no_temperature  λ_KL=1.0  λ_CE=0.5  T=1.0  fg=5.0  去掉温度缩放（硬标签）
  kl_only         λ_KL=1.0  λ_CE=0.0  T=2.0  fg=5.0  仅 KL 蒸馏，去掉 CE 辅助
  ce_only         λ_KL=0.0  λ_CE=1.0  T=2.0  fg=5.0  仅在策略 CE，去掉教师蒸馏

Checkpoint 保存路径：{ablation_ckpt_base}/{ablation_name}/best.pth
不影响已有的 checkpoints_fast / checkpoints_10step / checkpoints 目录。

用法（以 5-5 配置为例）：
  python train_ablation.py --config configs/ablation_5_5.yaml --ablation standard
  python train_ablation.py --config configs/ablation_5_5.yaml --ablation no_fg_weight
  python train_ablation.py --config configs/ablation_5_5.yaml --ablation no_temperature
  python train_ablation.py --config configs/ablation_5_5.yaml --ablation kl_only
  python train_ablation.py --config configs/ablation_5_5.yaml --ablation ce_only
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler

from utils import (
    load_config, set_seed, save_checkpoint, load_checkpoint,
    AverageMeter, logits_to_vil, compute_all_metrics
)
from data.sevir_dataset import build_dataloaders
from models.simvp import build_model

# ---------------------------------------------------------------------------
# 消融变体定义
# ---------------------------------------------------------------------------
ABLATION_VARIANTS = {
    "standard":       dict(kl_weight=1.0, ce_weight=0.5, temperature=2.0, fg_weight=5.0),
    "no_fg_weight":   dict(kl_weight=1.0, ce_weight=0.5, temperature=2.0, fg_weight=1.0),
    "no_temperature": dict(kl_weight=1.0, ce_weight=0.5, temperature=1.0, fg_weight=5.0),
    "kl_only":        dict(kl_weight=1.0, ce_weight=0.0, temperature=2.0, fg_weight=5.0),
    "ce_only":        dict(kl_weight=0.0, ce_weight=1.0, temperature=2.0, fg_weight=5.0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="SimVP OPSD 消融实验")
    parser.add_argument("--config",  type=str, required=True,  help="消融配置文件路径")
    parser.add_argument("--ablation", type=str, required=True,
                        choices=list(ABLATION_VARIANTS.keys()),
                        help="消融变体名称")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="覆盖 Baseline 热启动路径")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 损失函数（与 train_opsd.py 完全一致）
# ---------------------------------------------------------------------------

def kl_divergence_loss(student_logits, teacher_logits, temperature=2.0,
                       pixel_weights=None):
    student_logits = student_logits.float()
    teacher_logits = teacher_logits.float()
    log_q = F.log_softmax(student_logits / temperature, dim=1)
    p     = F.softmax(teacher_logits  / temperature, dim=1)
    kl_px = F.kl_div(log_q, p, reduction="none", log_target=False).sum(dim=1)
    if pixel_weights is not None:
        loss = (kl_px * pixel_weights).sum() / (pixel_weights.sum() + 1e-8)
    else:
        loss = kl_px.mean()
    return loss * (temperature ** 2)


def build_pixel_weights(targets_flat, fg_weight):
    has_echo = (targets_flat > 0).float()
    return 1.0 + has_echo * (fg_weight - 1.0)


# ---------------------------------------------------------------------------
# 训练一个 epoch
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device, cfg,
                    epoch, ablation_cfg):
    model.train()
    loss_meter = AverageMeter("Total_Loss")
    kl_meter   = AverageMeter("KL_Loss")
    ce_meter   = AverageMeter("CE_Loss")

    criterion_none = nn.CrossEntropyLoss(reduction="none")
    log_interval   = cfg["training"]["log_interval"]
    num_bins       = cfg["model"]["num_bins"]

    kl_weight   = ablation_cfg["kl_weight"]
    ce_weight   = ablation_cfg["ce_weight"]
    temperature = ablation_cfg["temperature"]
    fg_weight   = ablation_cfg["fg_weight"]

    for step, (input_frames, target_bins, future_frames) in enumerate(loader):
        input_frames  = input_frames.to(device, non_blocking=True)
        target_bins   = target_bins.to(device, non_blocking=True)
        future_frames = future_frames.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type):
            student_logits = model(input_frames, privileged_future=None)
            with torch.no_grad():
                # ce_only 变体不需要教师前向，跳过节省时间
                if kl_weight > 0:
                    teacher_logits = model(input_frames, privileged_future=future_frames)
                else:
                    teacher_logits = None

            B, T_out, C_bins, H, W = student_logits.shape
            student_flat = student_logits.float().view(B * T_out, C_bins, H, W)
            targets_flat = target_bins.view(B * T_out, H, W)
            pw           = build_pixel_weights(targets_flat, fg_weight)

            # KL 蒸馏损失
            if kl_weight > 0:
                teacher_flat = teacher_logits.float().view(B * T_out, C_bins, H, W)
                loss_kl = kl_divergence_loss(student_flat, teacher_flat,
                                             temperature, pw)
            else:
                loss_kl = torch.zeros((), device=device)

            # CE 辅助损失
            if ce_weight > 0:
                ce_px   = criterion_none(student_flat, targets_flat)
                loss_ce = (ce_px * pw).sum() / (pw.sum() + 1e-8)
            else:
                loss_ce = torch.zeros((), device=device)

            loss = kl_weight * loss_kl + ce_weight * loss_ce

        if not torch.isfinite(loss):
            print(f"  [WARNING] Step {step+1}: loss={loss.item():.4f}，跳过该 batch")
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), B)
        kl_meter.update(loss_kl.item(), B)
        ce_meter.update(loss_ce.item(), B)

        if (step + 1) % log_interval == 0:
            print(
                f"  [Epoch {epoch}] Step {step+1}/{len(loader)} | "
                f"Total: {loss_meter.val:.4f} | "
                f"KL: {kl_meter.val:.4f} | CE: {ce_meter.val:.4f}"
            )

    return loss_meter.avg, kl_meter.avg, ce_meter.avg


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, device, cfg):
    model.eval()
    loss_meter = AverageMeter("Val_CE_Loss")
    criterion  = nn.CrossEntropyLoss()
    num_bins   = cfg["model"]["num_bins"]
    vil_max    = cfg["data"]["vil_max"]
    thresholds = cfg["eval"]["thresholds"]

    all_pred_vil, all_true_vil = [], []

    for input_frames, target_bins, _ in loader:
        input_frames = input_frames.to(device, non_blocking=True)
        target_bins  = target_bins.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            all_logits = model(input_frames, privileged_future=None)

        B, T_out, C_bins, H, W = all_logits.shape
        logits_flat  = all_logits.float().view(B * T_out, C_bins, H, W)
        targets_flat = target_bins.view(B * T_out, H, W)
        loss = criterion(logits_flat, targets_flat)
        if torch.isfinite(loss):
            loss_meter.update(loss.item(), B)

        pred_vil = logits_to_vil(all_logits, num_bins, vil_max)
        bin_width = vil_max / num_bins
        true_vil  = (target_bins.cpu().numpy().astype(float) + 0.5) * bin_width
        all_pred_vil.append(pred_vil)
        all_true_vil.append(true_vil)

    import numpy as np
    all_pred_vil = np.concatenate(all_pred_vil, axis=0)
    all_true_vil = np.concatenate(all_true_vil, axis=0)
    metrics = compute_all_metrics(all_pred_vil, all_true_vil, thresholds)
    return loss_meter.avg, metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args      = parse_args()
    cfg       = load_config(args.config)
    abl_cfg   = ABLATION_VARIANTS[args.ablation]
    set_seed(cfg["training"]["seed"])

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[Device] {device}  |  [Ablation] {args.ablation}")
    print(f"  kl={abl_cfg['kl_weight']}  ce={abl_cfg['ce_weight']}  "
          f"T={abl_cfg['temperature']}  fg={abl_cfg['fg_weight']}")

    train_cfg = cfg["training"]
    ckpt_dir  = os.path.join(train_cfg["ablation_ckpt_base"], args.ablation)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("[Data] 正在加载 SEVIR VIL 数据集...")
    train_loader, val_loader, _ = build_dataloaders(
        cfg, batch_size=train_cfg["ablation_batch_size"]
    )

    print("[Model] 正在构建 SimVP 模型...")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 可训练参数量：{total_params / 1e6:.2f}M")

    optimizer = optim.AdamW(model.parameters(),
                            lr=train_cfg["ablation_lr"],
                            weight_decay=train_cfg["ablation_weight_decay"])
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=train_cfg["ablation_epochs"],
                                  eta_min=train_cfg["ablation_lr"] * 0.01)
    scaler = GradScaler()

    # 热启动
    pretrained_path = args.pretrained or train_cfg.get("ablation_pretrained")
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[Pretrained] 从 Baseline 权重热启动：{pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=str(device))
        model.load_state_dict(ckpt["model_state_dict"])
    elif pretrained_path:
        print(f"[WARNING] 未找到预训练权重：{pretrained_path}，从随机初始化开始")

    best_val_csi  = 0.0
    best_val_loss = float("inf")

    print(f"\n[Train] 消融 [{args.ablation}]，共 {train_cfg['ablation_epochs']} Epoch"
          f"  →  {ckpt_dir}")
    print("=" * 60)

    for epoch in range(1, train_cfg["ablation_epochs"] + 1):
        t0 = time.time()

        train_loss, kl_loss, ce_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, cfg, epoch, abl_cfg
        )
        val_loss, val_metrics = validate(model, val_loader, device, cfg)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"[Epoch {epoch:03d}/{train_cfg['ablation_epochs']}] "
            f"Train: {train_loss:.4f} (KL:{kl_loss:.4f} CE:{ce_loss:.4f}) | "
            f"Val: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
        )

        for thr_key in [74, 133]:
            if thr_key in val_metrics:
                m = val_metrics[thr_key]
                print(f"  [Metrics @{thr_key}] "
                      f"CSI={m['CSI']:.4f} | POD={m['POD']:.4f} | "
                      f"FAR={m['FAR']:.4f} | HSS={m['HSS']:.4f}")

        val_csi = val_metrics.get(74, {}).get("CSI", 0.0)
        if val_csi > best_val_csi:
            best_val_csi  = val_csi
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_csi": best_val_csi,
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "cfg": cfg,
                    "ablation": args.ablation,
                    "ablation_cfg": abl_cfg,
                },
                ckpt_dir, filename="best.pth",
            )
            print(f"  [Checkpoint] 保存最佳模型，CSI@74: {best_val_csi:.4f}")

        if epoch % train_cfg["save_interval"] == 0:
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "cfg": cfg, "ablation": args.ablation},
                ckpt_dir, filename=f"checkpoint_epoch{epoch:03d}.pth",
            )

        print("-" * 60)

    print(f"\n[Done] 消融 [{args.ablation}] 完成！最佳 CSI@74: {best_val_csi:.4f}")
    print(f"       已保存至：{os.path.join(ckpt_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
