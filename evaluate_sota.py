"""
SOTA 模型评估脚本（ConvLSTM / PredRNN / EarthFormer）
=======================================================
这些模型输出直接是 VIL 归一化值 [0,1]，不是 logits，
与 evaluate.py 中的 SimVP 接口不同，需要单独处理。

用法：
  python evaluate_sota.py --model convlstm --config configs/wadepre_align_convlstm.yaml \
      --ckpt checkpoints_wadepre_align/convlstm_baseline/best.pth --tag wa_convlstm \
      --output_dir eval_results/wadepre_align
"""

import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

from utils import load_config, set_seed, load_checkpoint, compute_metrics_at_threshold, compute_mse, compute_mae
from data.sevir_dataset import build_dataloaders
from models.convlstm import build_convlstm
from models.predrnn import build_predrnn

MODEL_BUILDERS = {
    "convlstm":    build_convlstm,
    "predrnn":     build_predrnn,
}

try:
    from models.earthformer_wrapper import build_earthformer
    MODEL_BUILDERS["earthformer"] = build_earthformer
except ImportError:
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      type=str, required=True, choices=list(MODEL_BUILDERS.keys()))
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--ckpt",       type=str, required=True)
    parser.add_argument("--tag",        type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--n_vis",      type=int, default=4)
    return parser.parse_args()


def max_pool_numpy(arr, pool_size):
    if pool_size == 1:
        return arr
    N, H, W = arr.shape
    H2 = H // pool_size * pool_size
    W2 = W // pool_size * pool_size
    arr = arr[:, :H2, :W2]
    arr = arr.reshape(N, H2 // pool_size, pool_size, W2 // pool_size, pool_size)
    return arr.max(axis=(2, 4))


@torch.no_grad()
def evaluate_model(model, loader, device, cfg, n_vis=4, vis_dir=None, tag="model"):
    model.eval()
    vil_max    = cfg["data"]["vil_max"]
    out_seq_len = cfg["data"]["out_seq_len"]
    thresholds  = cfg["eval"]["thresholds"]
    pool_sizes  = [1, 4, 16]

    per_step_preds = [[] for _ in range(out_seq_len)]
    per_step_trues = [[] for _ in range(out_seq_len)]
    vis_count = 0

    for input_frames, _target_bins, future_frames in loader:
        input_frames  = input_frames.to(device, non_blocking=True)
        future_frames_np = future_frames.squeeze(2).numpy() * vil_max  # [B, T, H, W]

        pred = model(input_frames)                        # [B, T_out, 1, H, W]
        pred_np = pred.squeeze(2).cpu().numpy() * vil_max # [B, T, H, W]

        for t in range(out_seq_len):
            per_step_preds[t].append(pred_np[:, t])
            per_step_trues[t].append(future_frames_np[:, t])

        if vis_dir and vis_count < n_vis:
            for b in range(min(input_frames.shape[0], n_vis - vis_count)):
                _save_vis(
                    input_frames[b].cpu().numpy(),
                    pred_np[b], future_frames_np[b],
                    vis_dir, tag, vis_count, vil_max,
                )
                vis_count += 1

    per_step_preds = [np.concatenate(p, axis=0) for p in per_step_preds]
    per_step_trues = [np.concatenate(t, axis=0) for t in per_step_trues]

    metrics_by_pool = {}
    for ps in pool_sizes:
        metrics_by_pool[ps] = {}
        for t in range(out_seq_len):
            pred_t = max_pool_numpy(per_step_preds[t], ps)
            true_t = max_pool_numpy(per_step_trues[t], ps)
            metrics_by_pool[ps][t] = {}
            for thr in thresholds:
                metrics_by_pool[ps][t][thr] = compute_metrics_at_threshold(pred_t, true_t, thr)

    all_pred = np.stack(per_step_preds, axis=1)
    all_true = np.stack(per_step_trues, axis=1)
    mse = compute_mse(all_pred, all_true)
    mae = compute_mae(all_pred, all_true)

    return metrics_by_pool, mse, mae


def _save_vis(input_frames, pred_vil, true_vil, vis_dir, tag, idx, vil_max):
    os.makedirs(vis_dir, exist_ok=True)
    T_out = pred_vil.shape[0]
    steps = [0, T_out // 2, T_out - 1]
    fig, axes = plt.subplots(2, len(steps), figsize=(4 * len(steps), 8))
    fig.suptitle(f"{tag} - Sample {idx}")
    for col, s in enumerate(steps):
        axes[0, col].imshow(pred_vil[s], cmap="jet", vmin=0, vmax=vil_max)
        axes[0, col].set_title(f"Pred t+{(s+1)*10}min")
        axes[0, col].axis("off")
        axes[1, col].imshow(true_vil[s], cmap="jet", vmin=0, vmax=vil_max)
        axes[1, col].set_title(f"True t+{(s+1)*10}min")
        axes[1, col].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f"{tag}_sample{idx:03d}.png"), dpi=100)
    plt.close()


def _mean_csi(metrics, out_seq_len, thresholds):
    return float(np.mean([metrics[t][thr]["CSI"] for t in range(out_seq_len) for thr in thresholds]))

def _mean_csi_thr(metrics, out_seq_len, thr):
    return float(np.mean([metrics[t][thr]["CSI"] for t in range(out_seq_len)]))

def _mean_hss(metrics, out_seq_len, thresholds):
    return float(np.mean([metrics[t][thr]["HSS"] for t in range(out_seq_len) for thr in thresholds]))


def save_csv(metrics_by_pool, out_seq_len, thresholds, path, tag):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "pool_size", "step", "lead_time_min", "threshold", "CSI", "POD", "FAR", "HSS"])
        for ps, step_metrics in metrics_by_pool.items():
            for t, thr_metrics in step_metrics.items():
                lead = (t + 1) * 10  # 10分钟间隔
                for thr, m in thr_metrics.items():
                    w.writerow([tag, ps, t+1, lead, thr,
                                f"{m['CSI']:.4f}", f"{m['POD']:.4f}",
                                f"{m['FAR']:.4f}", f"{m['HSS']:.4f}"])
    print(f"  [CSV] 指标已保存：{path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    tag = args.tag or f"{args.model}_{os.path.basename(args.ckpt)}"

    print(f"[Device] {device}")
    print("[Data] 加载测试集...")
    _, _, test_loader = build_dataloaders(cfg)

    print(f"[Model] 构建 {args.model.upper()}...")
    model = MODEL_BUILDERS[args.model](cfg).to(device)
    load_checkpoint(args.ckpt, model, device=str(device))
    print(f"[Model] 参数量：{sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")

    print(f"\n[Eval] 评估：{tag}")
    metrics_by_pool, mse, mae = evaluate_model(
        model, test_loader, device, cfg,
        n_vis=args.n_vis, vis_dir=vis_dir, tag=tag,
    )

    thresholds  = cfg["eval"]["thresholds"]
    out_seq_len = cfg["data"]["out_seq_len"]

    print(f"  MSE: {mse:.4f} | MAE: {mae:.4f}")
    m1 = metrics_by_pool[1]
    print(f"  CSI-M:   {_mean_csi(m1, out_seq_len, thresholds):.4f}")
    print(f"  CSI-219: {_mean_csi_thr(m1, out_seq_len, 219):.4f}")
    print(f"  CSI-181: {_mean_csi_thr(m1, out_seq_len, 181):.4f}")
    print(f"  CSI-74:  {_mean_csi_thr(m1, out_seq_len, 74):.4f}")
    print(f"  HSS:     {_mean_hss(m1, out_seq_len, thresholds):.4f}")

    csv_path = os.path.join(args.output_dir, f"{tag}_metrics.csv")
    save_csv(metrics_by_pool, out_seq_len, thresholds, csv_path, tag)
    print(f"\n[Done] 结果已保存至 {args.output_dir}/")


if __name__ == "__main__":
    main()
