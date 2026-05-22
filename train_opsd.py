"""
OPSD 训练脚本
=============
第二阶段：On-Policy Self-Distillation（在策略自蒸馏）训练。

核心机制（方案一：时序遮掩特征拼接）：
  - 学生分支：自回归展开，每步用 argmax(logit) 的预测值更新滑动窗口
  - 教师分支：每步用真实的未来帧更新滑动窗口（特权信息）
  - 教师分支只做前向传播（torch.no_grad()），不计算梯度，显存几乎不增加
  - 损失函数：KL 散度（教师 -> 学生）+ 交叉熵辅助损失

损失函数：
  L_total = w_kl * KL(P_teacher || P_student) + w_ce * CE(P_student, target)

  KL 散度使用 temperature softmax 软化教师分布，防止教师过于自信导致蒸馏退化。

用法：
  # 标准 OPSD（KL 各步等权重）
  python train_opsd.py --config configs/default.yaml

  # Reward-Weighted OPSD（KL 按每步 CSI 反向加权，难帧得到更大梯度）
  python train_opsd.py --config configs/default.yaml --reward_weight

  # 对比评估两个变体
  python evaluate.py --config configs/default.yaml \
      --ckpt checkpoints/opsd/best.pth checkpoints/opsd_rw/best.pth \
      --tag opsd opsd_rw
"""

import os
import sys
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


def parse_args():
    parser = argparse.ArgumentParser(description="SimVP OPSD Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="配置文件路径")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="从 Baseline 权重热启动（覆盖配置文件中的设置）")
    parser.add_argument("--resume", type=str, default=None,
                        help="从 OPSD checkpoint 恢复训练")
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备，如 cuda:0 或 cpu")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="KL 蒸馏温度（默认 2.0，越大教师分布越软）")
    parser.add_argument("--reward_weight", action="store_true",
                        help="启用 Reward-Weighted OPSD：KL 损失按每步 CSI 的反向值加权，"
                             "预测质量差的帧获得更大的蒸馏梯度。"
                             "checkpoint 保存在 opsd_rw_ckpt_dir（默认 checkpoints/opsd_rw）")
    return parser.parse_args()


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    pixel_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    计算教师到学生的掩码加权 KL 散度蒸馏损失。

    背景：VIL 雷达图中 80%+ 的像素是晴空（值为 0）。
    如果对全图像素做 batchmean，大量零值区域产生的微小 KL 损失会在数量上
    彻底淹没强对流核心区域的梯度，导致模型退化为"全图薄雾"预测。
    解决方案：对每个像素位置的 KL 值乘以空间权重，再做加权平均。

    L_KL = T^2 * weighted_mean( KL_per_pixel(P_teacher || P_student) )

    使用 F.kl_div(reduction='none') 而非手写 log，原因：
    - PyTorch 内部对 log_softmax + kl_div 做了联合数值优化，避免 log(p+eps) 在
      p→0 时产生的 -20.7 * p 浮点震荡，从根本上消除 NaN 风险。
    - F.kl_div 的输入约定：input=log(Q)，target=P，计算 P*(log(P)-log(Q))。

    Args:
        student_logits: [N, num_bins, H, W]
        teacher_logits: [N, num_bins, H, W]
        temperature:    蒸馏温度 T（越大分布越软，默认 2.0）
        pixel_weights:  [N, H, W] 每个像素位置的权重（None 时退化为均匀加权）

    Returns:
        scalar loss
    """
    # log Q（学生）和 P（教师），均在 bin 维度做 softmax
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)  # [N, C, H, W]
    p_teacher     = F.softmax(teacher_logits / temperature, dim=1)       # [N, C, H, W]

    # F.kl_div(input=log Q, target=P, reduction='none') -> [N, C, H, W]
    # 每个元素 = P * (log P - log Q)，在 bin 维度求和得到逐像素 KL
    kl_per_pixel = F.kl_div(
        log_p_student, p_teacher, reduction="none", log_target=False
    ).sum(dim=1)  # [N, H, W]

    if pixel_weights is not None:
        # 加权平均：强对流区域权重高，晴空区域权重低
        loss = (kl_per_pixel * pixel_weights).sum() / (pixel_weights.sum() + 1e-8)
    else:
        loss = kl_per_pixel.mean()

    # 乘以 T^2 补偿温度缩放对梯度幅度的影响（Hinton et al., 2015）
    return loss * (temperature ** 2)


def build_pixel_weights(
    targets_flat: torch.Tensor,
    num_bins: int,
    foreground_weight: float = 5.0,
) -> torch.Tensor:
    """
    根据真实 bin 标签构建逐像素空间权重。

    策略：bin > 0（有回波）的像素权重为 foreground_weight，
          bin == 0（晴空无回波）的像素权重为 1.0。
    这样强对流区域的梯度贡献被放大 foreground_weight 倍，
    防止大面积晴空区域的背景噪声淹没真正有意义的气象信号。

    Args:
        targets_flat:      [N, H, W]，int64，bin 索引
        num_bins:          bin 总数（未使用，保留接口一致性）
        foreground_weight: 有回波区域的权重倍数（默认 5.0）

    Returns:
        weights: [N, H, W]，float32
    """
    # bin > 0 表示有雷达回波（非晴空）
    has_echo = (targets_flat > 0).float()
    weights = 1.0 + has_echo * (foreground_weight - 1.0)
    return weights


@torch.no_grad()
def compute_step_csi(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    num_bins: int,
    vil_max: float,
    threshold: float,
) -> float:
    """
    计算单步预测的批均 CSI，用作奖励信号（不参与梯度计算）。

    Args:
        logits:      [B, num_bins, H, W]，学生当前步的 logit
        target_bins: [B, H, W]，int64，当前步的真实 bin 索引
        num_bins:    bin 总数
        vil_max:     VIL 像素最大值（用于还原物理量）
        threshold:   VIL 像素阈值，用于二值化（如 74 对应中等对流）

    Returns:
        mean_csi: float，批内各样本 CSI 的均值，值域 [0, 1]
    """
    bin_width = vil_max / num_bins
    # argmax -> VIL 像素值（取 bin 中心）
    pred_vil = (logits.argmax(dim=1).float() + 0.5) * bin_width   # [B, H, W]
    true_vil = (target_bins.float() + 0.5) * bin_width             # [B, H, W]

    pred_pos = pred_vil >= threshold   # [B, H, W] bool
    true_pos = true_vil >= threshold

    # 逐样本统计（dim=(-2,-1) 对 H×W 求和，保留 B 维）
    hits        = (pred_pos & true_pos).float().sum(dim=(-2, -1))   # [B]
    misses      = (~pred_pos & true_pos).float().sum(dim=(-2, -1))
    false_alarms = (pred_pos & ~true_pos).float().sum(dim=(-2, -1))

    denom = hits + misses + false_alarms
    csi = torch.where(denom > 0, hits / denom.clamp(min=1e-8), torch.zeros_like(hits))
    return csi.mean().item()   # Python float，脱离计算图


def train_one_epoch_opsd(
    model, loader, optimizer, scaler, device, cfg, epoch, temperature,
    use_reward_weight: bool = False,
):
    """
    执行一个 epoch 的 OPSD 训练。

    OPSD 核心逻辑：
    1. 学生分支：model(input_frames, privileged_future=None)
       -> 自回归，每步用预测值填充滑动窗口
    2. 教师分支：model(input_frames, privileged_future=future_frames)
       -> 每步用真实未来帧填充滑动窗口（特权信息）
       -> 包裹在 torch.no_grad() 中，不计算梯度
    3. 逐步计算 KL 散度 + 交叉熵，反向传播更新学生参数
    """
    model.train()

    loss_meter   = AverageMeter("Total_Loss")
    kl_meter     = AverageMeter("KL_Loss")
    ce_meter     = AverageMeter("CE_Loss")
    reward_meter = AverageMeter("CSI_Reward")  # 仅 reward_weight 模式下有意义

    criterion_ce_none = nn.CrossEntropyLoss(reduction="none")
    train_cfg        = cfg["training"]
    kl_weight        = train_cfg["opsd_kl_weight"]
    ce_weight        = train_cfg["opsd_ce_weight"]
    foreground_weight = train_cfg.get("foreground_weight", 5.0)
    log_interval     = train_cfg["log_interval"]
    num_bins         = cfg["model"]["num_bins"]
    vil_max          = cfg["data"]["vil_max"]
    reward_threshold = float(train_cfg.get("reward_threshold", 74))

    for step, (input_frames, target_bins, future_frames) in enumerate(loader):
        input_frames  = input_frames.to(device, non_blocking=True)
        target_bins   = target_bins.to(device, non_blocking=True)
        future_frames = future_frames.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type):
            student_logits = model(input_frames, privileged_future=None)
            with torch.no_grad():
                teacher_logits = model(input_frames, privileged_future=future_frames)

            B, T_out, C_bins, H, W = student_logits.shape
            student_flat  = student_logits.view(B * T_out, C_bins, H, W)
            teacher_flat  = teacher_logits.view(B * T_out, C_bins, H, W)
            targets_flat  = target_bins.view(B * T_out, H, W)
            pixel_weights = build_pixel_weights(targets_flat, num_bins, foreground_weight)

            if use_reward_weight:
                # ---- Reward-Weighted KL ----
                # 逐步计算 KL，以 (1 - CSI_t) 作为步权重：
                # 预测越差（CSI 低）→ 权重越大 → 该步蒸馏梯度越强。
                # compute_step_csi 带 @torch.no_grad，CSI 值仅作常数权重，不参与梯度。
                kl_accum  = torch.zeros((), device=device)
                step_csis = []
                for t in range(T_out):
                    s_t   = student_logits[:, t]   # [B, num_bins, H, W]，梯度正常流
                    tea_t = teacher_logits[:, t]
                    tgt_t = target_bins[:, t]
                    pw_t  = build_pixel_weights(tgt_t, num_bins, foreground_weight)

                    csi_t  = compute_step_csi(s_t, tgt_t, num_bins, vil_max, reward_threshold)
                    step_w = 1.0 - csi_t           # Python float：难帧权重大，易帧权重小
                    kl_accum = kl_accum + step_w * kl_divergence_loss(
                        s_t, tea_t, temperature, pw_t
                    )
                    step_csis.append(csi_t)

                loss_kl = kl_accum / T_out
                reward_meter.update(sum(step_csis) / len(step_csis), B)
            else:
                # ---- 标准 OPSD：各步 KL 等权重 ----
                loss_kl = kl_divergence_loss(
                    student_flat, teacher_flat, temperature, pixel_weights
                )

            # CE 辅助损失（两种模式共用）
            ce_per_pixel = criterion_ce_none(student_flat, targets_flat)
            loss_ce = (ce_per_pixel * pixel_weights).sum() / (pixel_weights.sum() + 1e-8)

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
            reward_str = (
                f" | CSI@{reward_threshold:.0f}: {reward_meter.avg:.4f}"
                if use_reward_weight else ""
            )
            print(
                f"  [Epoch {epoch}] Step {step+1}/{len(loader)} | "
                f"Total: {loss_meter.val:.4f} | "
                f"KL: {kl_meter.val:.4f} | "
                f"CE: {ce_meter.val:.4f}{reward_str}"
            )

    return loss_meter.avg, kl_meter.avg, ce_meter.avg, reward_meter.avg


@torch.no_grad()
def validate(model, loader, device, cfg):
    """在验证集上评估，返回平均 loss 和气象指标。"""
    model.eval()
    loss_meter = AverageMeter("Val_CE_Loss")
    criterion = nn.CrossEntropyLoss()

    num_bins = cfg["model"]["num_bins"]
    vil_max = cfg["data"]["vil_max"]
    thresholds = cfg["eval"]["thresholds"]

    all_pred_vil = []
    all_true_vil = []

    for input_frames, target_bins, _future_frames in loader:
        input_frames = input_frames.to(device, non_blocking=True)
        target_bins = target_bins.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            # 验证时走学生模式（无特权信息），模拟真实推理
            all_logits = model(input_frames, privileged_future=None)

        B, T_out, C_bins, H, W = all_logits.shape
        logits_flat = all_logits.view(B * T_out, C_bins, H, W)
        targets_flat = target_bins.view(B * T_out, H, W)
        loss = criterion(logits_flat, targets_flat)
        loss_meter.update(loss.item(), B)

        pred_vil = logits_to_vil(all_logits, num_bins, vil_max)
        bin_width = vil_max / num_bins
        true_vil = (target_bins.cpu().numpy().astype(float) + 0.5) * bin_width

        all_pred_vil.append(pred_vil)
        all_true_vil.append(true_vil)

    import numpy as np
    all_pred_vil = np.concatenate(all_pred_vil, axis=0)
    all_true_vil = np.concatenate(all_true_vil, axis=0)

    metrics = compute_all_metrics(all_pred_vil, all_true_vil, thresholds)
    return loss_meter.avg, metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_cfg = cfg["training"]
    mode_tag = "OPSD+RW" if args.reward_weight else "OPSD"
    print(f"[Device] {device}  |  [Mode] {mode_tag}  |  [T] {args.temperature}")
    if args.reward_weight:
        print(f"[RW] CSI 奖励阈值: {train_cfg.get('reward_threshold', 74)}")

    print("[Data] 正在加载 SEVIR VIL 数据集...")
    train_loader, val_loader, _ = build_dataloaders(
        cfg, batch_size=train_cfg["opsd_batch_size"]
    )

    print("[Model] 正在构建 SimVP 模型...")
    model = build_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] 可训练参数量：{total_params / 1e6:.2f}M")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg["opsd_lr"],
        weight_decay=train_cfg["opsd_weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["opsd_epochs"],
        eta_min=train_cfg["opsd_lr"] * 0.01,
    )
    scaler = GradScaler()

    start_epoch   = 1
    best_val_loss = float("inf")

    # 热启动：--pretrained > 配置文件中的 opsd_pretrained
    pretrained_path = args.pretrained or train_cfg.get("opsd_pretrained")
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"[Pretrained] 从 Baseline 权重热启动：{pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=str(device))
        model.load_state_dict(ckpt["model_state_dict"])
        best = ckpt.get("best_val_loss", float("nan"))
        print(f"  Baseline 最佳 Val Loss: {best:.4f}")
    elif pretrained_path:
        print(f"[WARNING] 未找到预训练权重：{pretrained_path}，从随机初始化开始")

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device=str(device))
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # reward_weight 模式存到独立目录，方便直接对比 checkpoint
    ckpt_dir = (
        train_cfg.get("opsd_rw_ckpt_dir", "checkpoints/opsd_rw")
        if args.reward_weight
        else train_cfg["opsd_ckpt_dir"]
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n[Train] {mode_tag} 训练，共 {train_cfg['opsd_epochs']} Epoch  →  {ckpt_dir}")
    print(f"        KL 权重: {train_cfg['opsd_kl_weight']} | CE 权重: {train_cfg['opsd_ce_weight']}")
    print("=" * 60)

    for epoch in range(start_epoch, train_cfg["opsd_epochs"] + 1):
        t0 = time.time()

        train_loss, kl_loss, ce_loss, mean_reward = train_one_epoch_opsd(
            model, train_loader, optimizer, scaler, device, cfg, epoch,
            args.temperature, use_reward_weight=args.reward_weight,
        )

        val_loss, val_metrics = validate(model, val_loader, device, cfg)
        scheduler.step()
        elapsed = time.time() - t0

        reward_str = f" | Reward: {mean_reward:.4f}" if args.reward_weight else ""
        print(
            f"[Epoch {epoch:03d}/{train_cfg['opsd_epochs']}] "
            f"Train: {train_loss:.4f} (KL:{kl_loss:.4f} CE:{ce_loss:.4f}){reward_str} | "
            f"Val: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.1f}s"
        )

        for thr_key in [74, 133]:
            if thr_key in val_metrics:
                m = val_metrics[thr_key]
                print(
                    f"  [Metrics @{thr_key}] "
                    f"CSI={m['CSI']:.4f} | POD={m['POD']:.4f} | "
                    f"FAR={m['FAR']:.4f} | HSS={m['HSS']:.4f}"
                )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "val_metrics": val_metrics,
                    "cfg": cfg,
                    "temperature": args.temperature,
                    "reward_weight": args.reward_weight,
                },
                ckpt_dir,
                filename="best.pth",
            )
            print(f"  [Checkpoint] 保存最佳 {mode_tag} 模型，Val Loss: {best_val_loss:.4f}")

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

    print(f"\n[Done] {mode_tag} 训练完成！最佳 Val Loss: {best_val_loss:.4f}")
    print(f"       最佳模型已保存至：{os.path.join(ckpt_dir, 'best.pth')}")


if __name__ == "__main__":
    main()
