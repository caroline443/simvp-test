"""
工具函数
========
包含：
  - 配置文件加载
  - 随机种子固定
  - 气象评估指标（CSI / POD / FAR / HSS）
  - 可视化工具
  - Checkpoint 保存与加载
"""

import os
import random
import numpy as np
import yaml
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")  # 无头模式，适合服务器环境
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件，返回字典。"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# 随机种子
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """固定所有随机种子，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 气象评估指标
# ---------------------------------------------------------------------------

def compute_metrics_at_threshold(
    pred_vil: np.ndarray,
    true_vil: np.ndarray,
    threshold: float,
) -> dict:
    """
    在给定 VIL 像素阈值下计算二值化气象评估指标。

    Args:
        pred_vil:  预测 VIL 值，任意形状的 float32 数组
        true_vil:  真实 VIL 值，与 pred_vil 形状相同
        threshold: VIL 像素阈值（如 74 对应中等对流）

    Returns:
        dict 包含 hits, misses, false_alarms, correct_negatives,
             CSI, POD, FAR, HSS
    """
    pred_bin = (pred_vil >= threshold).astype(np.float32)
    true_bin = (true_vil >= threshold).astype(np.float32)

    hits = float(np.sum((pred_bin == 1) & (true_bin == 1)))
    misses = float(np.sum((pred_bin == 0) & (true_bin == 1)))
    false_alarms = float(np.sum((pred_bin == 1) & (true_bin == 0)))
    correct_neg = float(np.sum((pred_bin == 0) & (true_bin == 0)))

    # CSI (Critical Success Index) = hits / (hits + misses + false_alarms)
    denom_csi = hits + misses + false_alarms
    csi = hits / denom_csi if denom_csi > 0 else 0.0

    # POD (Probability of Detection) = hits / (hits + misses)
    denom_pod = hits + misses
    pod = hits / denom_pod if denom_pod > 0 else 0.0

    # FAR (False Alarm Ratio) = false_alarms / (hits + false_alarms)
    denom_far = hits + false_alarms
    far = false_alarms / denom_far if denom_far > 0 else 0.0

    # HSS (Heidke Skill Score)
    n_total = hits + misses + false_alarms + correct_neg
    expected = ((hits + misses) * (hits + false_alarms) +
                (correct_neg + misses) * (correct_neg + false_alarms)) / (n_total + 1e-9)
    denom_hss = n_total - expected
    hss = (hits + correct_neg - expected) / denom_hss if denom_hss > 0 else 0.0

    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_neg,
        "CSI": csi,
        "POD": pod,
        "FAR": far,
        "HSS": hss,
    }


def compute_all_metrics(
    pred_vil: np.ndarray,
    true_vil: np.ndarray,
    thresholds: list,
) -> dict:
    """
    在多个阈值下批量计算评估指标。

    Args:
        pred_vil:   [N, T, H, W] 或 [T, H, W]，预测 VIL 值
        true_vil:   与 pred_vil 形状相同，真实 VIL 值
        thresholds: 阈值列表

    Returns:
        dict: {threshold: {metric_name: value}}
    """
    results = {}
    for thr in thresholds:
        results[thr] = compute_metrics_at_threshold(pred_vil, true_vil, thr)
    return results


def compute_mse(pred: np.ndarray, true: np.ndarray) -> float:
    """计算均方误差（MSE）。"""
    return float(np.mean((pred - true) ** 2))


def compute_mae(pred: np.ndarray, true: np.ndarray) -> float:
    """计算平均绝对误差（MAE）。"""
    return float(np.mean(np.abs(pred - true)))


# ---------------------------------------------------------------------------
# Logit -> VIL 像素值转换（用于评估）
# ---------------------------------------------------------------------------

def logits_to_vil(logits: torch.Tensor, num_bins: int, vil_max: float = 255.0) -> np.ndarray:
    """
    将模型输出的 Logit 转换为 VIL 像素值（取 argmax bin 的中心值）。

    Args:
        logits:   [B, T, num_bins, H, W] 或 [B, num_bins, H, W]
        num_bins: bin 数量
        vil_max:  VIL 最大值

    Returns:
        vil_pred: numpy 数组，与输入形状对应（去掉 num_bins 维度），值域 [0, vil_max]
    """
    pred_bins = torch.argmax(logits, dim=-3)  # 在 num_bins 维度取 argmax
    bin_width = vil_max / num_bins
    vil_pred = (pred_bins.float() + 0.5) * bin_width
    return vil_pred.cpu().numpy()


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

# SEVIR VIL 的标准配色方案（参考官方文档）
VIL_CMAP_COLORS = [
    "#000000",  # 0: 无回波
    "#00BFFF",  # 轻微
    "#00FF00",
    "#32CD32",
    "#ADFF2F",
    "#FFFF00",
    "#FFD700",
    "#FFA500",
    "#FF6347",
    "#FF4500",
    "#FF0000",
    "#DC143C",
    "#8B0000",
    "#800080",
    "#4B0082",
    "#FFFFFF",  # 15: 极端强对流
]


def get_vil_cmap():
    """返回 VIL 专用 colormap。"""
    return mcolors.ListedColormap(VIL_CMAP_COLORS)


def visualize_prediction(
    input_frames: np.ndarray,
    pred_vil: np.ndarray,
    true_vil: np.ndarray,
    save_path: str,
    sample_steps: list = None,
):
    """
    可视化预测结果与真实值的对比图。

    Args:
        input_frames: [in_seq_len, H, W]，历史帧（归一化到 [0,1]）
        pred_vil:     [out_seq_len, H, W]，预测 VIL 值 [0, 255]
        true_vil:     [out_seq_len, H, W]，真实 VIL 值 [0, 255]
        save_path:    图片保存路径
        sample_steps: 要可视化的预测步骤列表，默认 [0, 4, 9]
    """
    if sample_steps is None:
        sample_steps = [0, 4, min(9, pred_vil.shape[0] - 1)]

    cmap = get_vil_cmap()
    n_steps = len(sample_steps)
    fig, axes = plt.subplots(2, n_steps, figsize=(4 * n_steps, 8))

    for col, step in enumerate(sample_steps):
        # 预测
        axes[0, col].imshow(pred_vil[step], cmap=cmap, vmin=0, vmax=255)
        axes[0, col].set_title(f"Pred t+{(step+1)*5}min")
        axes[0, col].axis("off")

        # 真实
        axes[1, col].imshow(true_vil[step], cmap=cmap, vmin=0, vmax=255)
        axes[1, col].set_title(f"True t+{(step+1)*5}min")
        axes[1, col].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Checkpoint 管理
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, ckpt_dir: str, filename: str = "checkpoint.pth"):
    """保存训练状态到文件。"""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    return path


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer = None,
                    device: str = "cpu") -> dict:
    """
    加载 checkpoint，返回训练状态字典。

    Args:
        path:      checkpoint 文件路径
        model:     模型实例（会被 in-place 更新）
        optimizer: 优化器实例（可选，会被 in-place 更新）
        device:    加载到的设备

    Returns:
        state dict（包含 epoch、best_val_loss 等信息）
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[Checkpoint] 已加载：{path}，epoch={ckpt.get('epoch', '?')}")
    return ckpt


# ---------------------------------------------------------------------------
# 训练日志打印
# ---------------------------------------------------------------------------

class AverageMeter:
    """跟踪并计算指标的滑动平均值。"""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return f"{self.name}: {self.avg:.4f}"
