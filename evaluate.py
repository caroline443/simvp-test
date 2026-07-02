"""
评估脚本
========
对训练好的模型（Baseline 或 OPSD）在测试集上进行全面评估。

输出内容：
  1. 各预测步骤（t+5min ~ t+60min）在多个阈值下的 CSI / POD / FAR / HSS（POOL1/4/16）
  2. 整体 CSI-M / CSI-181 / CSI-219 / SSIM / HSS（SimCast Table II 格式）
  3. 整体 MSE / MAE
  4. 可视化对比图（预测 vs 真实）
  5. 结果保存为 CSV 文件

用法：
  # 评估 Baseline 模型
  python evaluate.py --config configs/default.yaml --ckpt checkpoints/baseline/best.pth --tag baseline

  # 评估 OPSD 模型
  python evaluate.py --config configs/default.yaml --ckpt checkpoints/opsd/best.pth --tag opsd

  # 对比两个模型（会在同一张图上绘制 CSI 曲线）
  python evaluate.py --config configs/default.yaml \
      --ckpt checkpoints/baseline/best.pth checkpoints/opsd/best.pth \
      --tag baseline opsd
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

from utils import (
    load_config, set_seed, load_checkpoint,
    logits_to_vil, compute_metrics_at_threshold, compute_mse, compute_mae
)
from data.sevir_dataset import build_dataloaders
from models.simvp import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="SimVP Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, nargs="+", required=True,
                        help="checkpoint 路径，可传入多个用于对比")
    parser.add_argument("--tag", type=str, nargs="+", default=None,
                        help="每个 checkpoint 对应的标签名（用于图例）")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results",
                        help="评估结果保存目录")
    parser.add_argument("--n_vis", type=int, default=4,
                        help="可视化样本数量")
    return parser.parse_args()


def max_pool_numpy(arr, pool_size):
    """Max pool [N, H, W] numpy array with non-overlapping pool_size×pool_size windows."""
    if pool_size == 1:
        return arr
    N, H, W = arr.shape
    H2, W2 = H // pool_size, W // pool_size
    arr = arr[:, :H2 * pool_size, :W2 * pool_size]
    return arr.reshape(N, H2, pool_size, W2, pool_size).max(axis=(2, 4))


def compute_ssim_batch(pred_vil, true_vil, vil_max):
    """Mean SSIM over [N, H, W] arrays; requires scikit-image."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return float("nan")
    pred_n = np.clip(pred_vil / vil_max, 0.0, 1.0)
    true_n = np.clip(true_vil / vil_max, 0.0, 1.0)
    return float(np.mean([
        ssim(true_n[i], pred_n[i], data_range=1.0)
        for i in range(len(pred_vil))
    ]))


@torch.no_grad()
def evaluate_model(model, loader, device, cfg, n_vis=4, vis_dir=None, tag="model"):
    """
    在测试集上完整评估模型。

    Returns:
        metrics_by_pool: {pool_size: {step: {threshold: {metric: value}}}}
        overall_mse:     float
        overall_mae:     float
        overall_ssim:    float (mean SSIM over all steps)
    """
    model.eval()
    num_bins = cfg["model"]["num_bins"]
    vil_max = cfg["data"]["vil_max"]
    out_seq_len = cfg["data"]["out_seq_len"]
    thresholds = cfg["eval"]["thresholds"]
    pool_sizes = [1, 4, 16]

    per_step_preds = [[] for _ in range(out_seq_len)]
    per_step_trues = [[] for _ in range(out_seq_len)]

    vis_count = 0
    bin_width = vil_max / num_bins

    for batch_idx, (input_frames, target_bins, _future_frames) in enumerate(loader):
        input_frames = input_frames.to(device, non_blocking=True)
        target_bins_np = target_bins.numpy()  # [B, T_out, H, W]

        with torch.amp.autocast(device_type=device.type):
            all_logits = model(input_frames, privileged_future=None)

        pred_vil = logits_to_vil(all_logits, num_bins, vil_max)  # [B, T_out, H, W]
        true_vil = (target_bins_np.astype(float) + 0.5) * bin_width

        for t in range(out_seq_len):
            per_step_preds[t].append(pred_vil[:, t])
            per_step_trues[t].append(true_vil[:, t])

        if vis_dir and vis_count < n_vis:
            for b in range(min(input_frames.shape[0], n_vis - vis_count)):
                _save_vis_sample(
                    input_frames[b].cpu().numpy(),
                    pred_vil[b],
                    true_vil[b],
                    vis_dir, tag, vis_count, vil_max,
                )
                vis_count += 1

    per_step_preds = [np.concatenate(p, axis=0) for p in per_step_preds]
    per_step_trues = [np.concatenate(t, axis=0) for t in per_step_trues]

    # Compute metrics for each pool size
    metrics_by_pool = {}
    for ps in pool_sizes:
        metrics_by_pool[ps] = {}
        for t in range(out_seq_len):
            pred_t = max_pool_numpy(per_step_preds[t], ps)
            true_t = max_pool_numpy(per_step_trues[t], ps)
            metrics_by_pool[ps][t] = {}
            for thr in thresholds:
                # pool thresholds scale: max pool of pool_size means any pixel
                # in window exceeds thr, so threshold stays the same value
                metrics_by_pool[ps][t][thr] = compute_metrics_at_threshold(
                    pred_t, true_t, thr
                )

    # SSIM over all steps (POOL1 only, full resolution)
    ssim_per_step = [
        compute_ssim_batch(per_step_preds[t], per_step_trues[t], vil_max)
        for t in range(out_seq_len)
    ]
    ssim_vals = [v for v in ssim_per_step if not np.isnan(v)]
    overall_ssim = float(np.mean(ssim_vals)) if ssim_vals else float("nan")

    all_pred = np.stack(per_step_preds, axis=1)
    all_true = np.stack(per_step_trues, axis=1)
    overall_mse = compute_mse(all_pred, all_true)
    overall_mae = compute_mae(all_pred, all_true)

    return metrics_by_pool, overall_mse, overall_mae, overall_ssim


def _save_vis_sample(input_frames, pred_vil, true_vil, vis_dir, tag, idx, vil_max):
    """保存单个样本的可视化对比图。"""
    os.makedirs(vis_dir, exist_ok=True)
    T_out = pred_vil.shape[0]
    sample_steps = [0, T_out // 2, T_out - 1]

    fig, axes = plt.subplots(2, len(sample_steps), figsize=(4 * len(sample_steps), 8))
    fig.suptitle(f"{tag} - Sample {idx}", fontsize=12)

    for col, step in enumerate(sample_steps):
        axes[0, col].imshow(pred_vil[step], cmap="jet", vmin=0, vmax=vil_max)
        axes[0, col].set_title(f"Pred t+{(step+1)*5}min")
        axes[0, col].axis("off")

        axes[1, col].imshow(true_vil[step], cmap="jet", vmin=0, vmax=vil_max)
        axes[1, col].set_title(f"True t+{(step+1)*5}min")
        axes[1, col].axis("off")

    plt.tight_layout()
    save_path = os.path.join(vis_dir, f"{tag}_sample{idx:03d}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_metrics_csv(metrics_by_pool, out_seq_len, thresholds, output_path, tag):
    """将逐步指标（含 pool_size）保存为 CSV 文件。"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "pool_size", "step", "lead_time_min", "threshold",
                         "CSI", "POD", "FAR", "HSS"])
        for ps, per_step_metrics in metrics_by_pool.items():
            for t in range(out_seq_len):
                lead_min = (t + 1) * 5
                for thr in thresholds:
                    m = per_step_metrics[t][thr]
                    writer.writerow([
                        tag, ps, t + 1, lead_min, thr,
                        f"{m['CSI']:.4f}", f"{m['POD']:.4f}",
                        f"{m['FAR']:.4f}", f"{m['HSS']:.4f}",
                    ])
    print(f"  [CSV] 指标已保存：{output_path}")


def _mean_csi(per_step_metrics, out_seq_len, thresholds):
    """Mean CSI across all steps and thresholds."""
    vals = [
        per_step_metrics[t][thr]["CSI"]
        for t in range(out_seq_len)
        for thr in thresholds
    ]
    return float(np.mean(vals))


def _mean_csi_thr(per_step_metrics, out_seq_len, thr):
    """Mean CSI across all steps for a single threshold."""
    vals = [per_step_metrics[t][thr]["CSI"] for t in range(out_seq_len)]
    return float(np.mean(vals))


def _mean_hss(per_step_metrics, out_seq_len, thresholds):
    vals = [
        per_step_metrics[t][thr]["HSS"]
        for t in range(out_seq_len)
        for thr in thresholds
    ]
    return float(np.mean(vals))


def print_simcast_summary(tag, metrics_by_pool, out_seq_len, thresholds, overall_ssim):
    """打印 SimCast Table II 风格的汇总指标。"""
    print(f"\n  ── {tag} SimCast-style Summary ──")
    header = f"  {'Metric':<14}" + "".join(f"  {'POOL'+str(ps):>8}" for ps in [1, 4, 16])
    print(header)
    print("  " + "-" * (14 + 3 * 10))

    for label, fn in [
        ("CSI-M",   lambda m: _mean_csi(m, out_seq_len, thresholds)),
        ("CSI-181", lambda m: _mean_csi_thr(m, out_seq_len, 181)),
        ("CSI-219", lambda m: _mean_csi_thr(m, out_seq_len, 219)),
        ("HSS",     lambda m: _mean_hss(m, out_seq_len, thresholds)),
    ]:
        row = f"  {label:<14}"
        for ps in [1, 4, 16]:
            v = fn(metrics_by_pool[ps])
            row += f"  {v:>8.4f}"
        print(row)

    ssim_str = f"{overall_ssim:.4f}" if not np.isnan(overall_ssim) else "  N/A (install scikit-image)"
    print(f"  {'SSIM':<14}  {ssim_str:>8}")


def plot_csi_curves(all_results, thresholds, out_seq_len, output_dir):
    """
    绘制多个模型在不同阈值下的 CSI 随预测步骤变化曲线（POOL1）。

    Args:
        all_results: list of (tag, metrics_by_pool)
    """
    os.makedirs(output_dir, exist_ok=True)
    lead_times = [(t + 1) * 5 for t in range(out_seq_len)]

    for thr in thresholds:
        fig, ax = plt.subplots(figsize=(10, 5))
        for tag, metrics_by_pool in all_results:
            per_step = metrics_by_pool[1]  # POOL1
            csi_values = [per_step[t][thr]["CSI"] for t in range(out_seq_len)]
            ax.plot(lead_times, csi_values, marker="o", label=tag, linewidth=2)

        ax.set_xlabel("Lead Time (min)")
        ax.set_ylabel("CSI")
        ax.set_title(f"CSI vs Lead Time (Threshold = {thr}, POOL1)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        save_path = os.path.join(output_dir, f"csi_thr{thr}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Plot] CSI 曲线已保存：{save_path}")

    for thr in thresholds:
        fig, ax = plt.subplots(figsize=(10, 5))
        for tag, metrics_by_pool in all_results:
            per_step = metrics_by_pool[1]
            pod_values = [per_step[t][thr]["POD"] for t in range(out_seq_len)]
            ax.plot(lead_times, pod_values, marker="s", label=tag, linewidth=2)

        ax.set_xlabel("Lead Time (min)")
        ax.set_ylabel("POD")
        ax.set_title(f"POD vs Lead Time (Threshold = {thr}, POOL1)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        save_path = os.path.join(output_dir, f"pod_thr{thr}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] 使用设备：{device}")

    tags = args.tag if args.tag else [f"model_{i}" for i in range(len(args.ckpt))]
    if len(tags) != len(args.ckpt):
        raise ValueError("--tag 数量必须与 --ckpt 数量一致")

    print("[Data] 正在加载测试集...")
    _, _, test_loader = build_dataloaders(cfg)

    out_seq_len = cfg["data"]["out_seq_len"]
    thresholds = cfg["eval"]["thresholds"]
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = []

    for ckpt_path, tag in zip(args.ckpt, tags):
        print(f"\n[Eval] 正在评估：{tag} ({ckpt_path})")

        raw = torch.load(ckpt_path, map_location="cpu")
        ckpt_model_cfg = raw.get("cfg", {}).get("model", {})
        if ckpt_model_cfg:
            merged_cfg = {**cfg, "model": {**cfg["model"], **ckpt_model_cfg}}
            translator_type = ckpt_model_cfg.get("translator_type", "inception")
            print(f"  [Config] 从 checkpoint 读取模型配置，translator_type={translator_type}")
        else:
            merged_cfg = cfg
        model = build_model(merged_cfg).to(device)
        load_checkpoint(ckpt_path, model, device=str(device))

        vis_dir = os.path.join(args.output_dir, "visualizations")
        metrics_by_pool, mse, mae, overall_ssim = evaluate_model(
            model, test_loader, device, cfg,
            n_vis=args.n_vis,
            vis_dir=vis_dir,
            tag=tag,
        )

        print(f"  [Overall] MSE: {mse:.4f} | MAE: {mae:.4f}")

        print_simcast_summary(tag, metrics_by_pool, out_seq_len, thresholds, overall_ssim)

        # 保存 CSV
        csv_path = os.path.join(args.output_dir, f"{tag}_metrics.csv")
        save_metrics_csv(metrics_by_pool, out_seq_len, thresholds, csv_path, tag)

        all_results.append((tag, metrics_by_pool))

    print("\n[Plot] 正在绘制 CSI / POD 对比曲线...")
    plot_csi_curves(all_results, thresholds, out_seq_len, args.output_dir)

    print(f"\n[Done] 评估完成！所有结果已保存至：{args.output_dir}/")


if __name__ == "__main__":
    main()
