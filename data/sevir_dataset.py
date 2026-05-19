"""
SEVIR VIL Dataset Loader
========================
支持从标准 SEVIR 目录结构加载 VIL（Vertical Integrated Liquid）雷达回波数据。

SEVIR 目录结构预期：
    <data_root>/
        SEVIR_CATALOG.csv
        data/
            vil/
                *.h5

每个 .h5 文件包含多个事件（event），每个事件为 49 帧，空间分辨率 384x384。
本模块从每个事件中随机采样连续 (in_seq_len + out_seq_len) 帧，
并将像素值离散化为 num_bins 个类别。
"""

import os
import glob
import random
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# VIL 通道在 SEVIR 中的 HDF5 key
VIL_KEY = "vil"

# SEVIR VIL 像素值范围 [0, 255]，对应 0 ~ 75 kg/m^2
VIL_MAX = 255.0


def build_bins(num_bins: int, vil_max: float = VIL_MAX):
    """
    将 [0, vil_max] 均匀划分为 num_bins 个区间，返回每个像素对应的 bin 索引。
    bin 0 对应无回波，bin num_bins-1 对应最强回波。
    """
    edges = np.linspace(0, vil_max, num_bins + 1)
    return edges


def vil_to_bins(vil_array: np.ndarray, num_bins: int, vil_max: float = VIL_MAX) -> np.ndarray:
    """
    将连续 VIL 像素值 [0, 255] 转换为离散 bin 索引 [0, num_bins-1]。

    Args:
        vil_array: float32 数组，值域 [0, 255]
        num_bins:  离散化 bin 数量
        vil_max:   VIL 最大值

    Returns:
        int64 数组，值域 [0, num_bins-1]
    """
    edges = build_bins(num_bins, vil_max)
    # np.digitize 返回 [1, num_bins]，减 1 变为 [0, num_bins-1]
    indices = np.digitize(vil_array, edges[1:])  # edges[1:] 作为右边界
    indices = np.clip(indices, 0, num_bins - 1)
    return indices.astype(np.int64)


def bins_to_vil(bin_indices: np.ndarray, num_bins: int, vil_max: float = VIL_MAX) -> np.ndarray:
    """
    将 bin 索引还原为 VIL 像素值（取每个 bin 的中心值）。
    用于可视化和评估。
    """
    edges = build_bins(num_bins, vil_max)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers[bin_indices].astype(np.float32)


class SEVIRVILDataset(Dataset):
    """
    SEVIR VIL 数据集。

    每个样本包含：
        input_frames:  [in_seq_len, 1, H, W]  float32，归一化到 [0, 1]
        target_bins:   [out_seq_len, H, W]     int64，bin 索引 [0, num_bins-1]
        future_frames: [out_seq_len, 1, H, W]  float32，归一化到 [0, 1]
                       （供 OPSD 教师模型使用的特权信息）

    Args:
        data_root:    SEVIR 数据根目录
        in_seq_len:   输入历史帧数
        out_seq_len:  预测未来帧数
        num_bins:     离散化 bin 数量
        vil_max:      VIL 最大像素值
        split:        'train' | 'val' | 'test'
        train_ratio:  训练集比例
        val_ratio:    验证集比例
        seed:         随机种子
    """

    def __init__(
        self,
        data_root: str,
        in_seq_len: int = 10,
        out_seq_len: int = 10,
        num_bins: int = 16,
        vil_max: float = VIL_MAX,
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.data_root = data_root
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.total_seq_len = in_seq_len + out_seq_len
        self.num_bins = num_bins
        self.vil_max = vil_max
        self.split = split

        # 扫描所有 VIL h5 文件
        vil_dir = os.path.join(data_root, "data", "vil")
        h5_files = sorted(glob.glob(os.path.join(vil_dir, "*.h5")))
        if len(h5_files) == 0:
            raise FileNotFoundError(
                f"在 {vil_dir} 下未找到任何 .h5 文件，"
                f"请确认 SEVIR 数据已正确放置。"
            )

        # 构建 (文件路径, 事件索引) 的样本列表
        self.samples = []
        for fpath in h5_files:
            try:
                with h5py.File(fpath, "r") as f:
                    if VIL_KEY not in f:
                        continue
                    n_events = f[VIL_KEY].shape[0]
                    n_frames = f[VIL_KEY].shape[1]
                    # 每个事件必须有足够的帧数
                    if n_frames < self.total_seq_len:
                        continue
                    for event_idx in range(n_events):
                        # 记录可用的起始帧偏移（随机采样窗口）
                        max_start = n_frames - self.total_seq_len
                        self.samples.append((fpath, event_idx, max_start))
            except Exception as e:
                print(f"[WARNING] 跳过文件 {fpath}，原因：{e}")

        if len(self.samples) == 0:
            raise RuntimeError("未找到任何有效的 SEVIR VIL 样本，请检查数据目录。")

        # 按比例划分 train / val / test
        rng = random.Random(seed)
        all_indices = list(range(len(self.samples)))
        rng.shuffle(all_indices)

        n_total = len(all_indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if split == "train":
            self.indices = all_indices[:n_train]
        elif split == "val":
            self.indices = all_indices[n_train: n_train + n_val]
        elif split == "test":
            self.indices = all_indices[n_train + n_val:]
        else:
            raise ValueError(f"split 必须是 'train'/'val'/'test'，收到：{split}")

        print(
            f"[SEVIRVILDataset] split={split}, "
            f"样本数={len(self.indices)}, "
            f"总事件数={len(self.samples)}"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fpath, event_idx, max_start = self.samples[self.indices[idx]]

        # 随机选取起始帧（训练时增加多样性）
        if self.split == "train" and max_start > 0:
            start = random.randint(0, max_start)
        else:
            # val/test 固定从中间开始，保证可复现
            start = max_start // 2

        end = start + self.total_seq_len

        with h5py.File(fpath, "r") as f:
            # shape: [total_seq_len, H, W]
            frames = f[VIL_KEY][event_idx, start:end].astype(np.float32)

        # 归一化到 [0, 1]
        frames_norm = frames / self.vil_max  # [T, H, W]

        # 拆分历史帧和未来帧
        input_raw = frames_norm[: self.in_seq_len]    # [in_seq_len, H, W]
        future_raw = frames_norm[self.in_seq_len:]    # [out_seq_len, H, W]

        # 增加通道维度 -> [T, 1, H, W]
        input_frames = input_raw[:, np.newaxis, :, :]
        future_frames = future_raw[:, np.newaxis, :, :]

        # 将未来帧的原始像素值离散化为 bin 索引
        future_raw_uint8 = (future_raw * self.vil_max)  # 还原到 [0, 255]
        target_bins = vil_to_bins(future_raw_uint8, self.num_bins, self.vil_max)
        # target_bins: [out_seq_len, H, W], int64

        return (
            torch.from_numpy(input_frames),   # [in_seq_len, 1, H, W]  float32
            torch.from_numpy(target_bins),    # [out_seq_len, H, W]    int64
            torch.from_numpy(future_frames),  # [out_seq_len, 1, H, W] float32（教师特权信息）
        )


def build_dataloaders(cfg: dict):
    """
    根据配置字典构建 train / val / test DataLoader。

    Args:
        cfg: 完整配置字典（对应 default.yaml 的 data 节）

    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    common_kwargs = dict(
        data_root=data_cfg["data_root"],
        in_seq_len=data_cfg["in_seq_len"],
        out_seq_len=data_cfg["out_seq_len"],
        num_bins=data_cfg["num_bins"],
        vil_max=data_cfg["vil_max"],
        train_ratio=data_cfg["train_ratio"],
        val_ratio=data_cfg["val_ratio"],
        seed=train_cfg["seed"],
    )

    train_ds = SEVIRVILDataset(split="train", **common_kwargs)
    val_ds = SEVIRVILDataset(split="val", **common_kwargs)
    test_ds = SEVIRVILDataset(split="test", **common_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["baseline_batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["baseline_batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
