"""
SEVIR VIL Dataset Loader
========================
支持从 SEVIR 数据目录加载 VIL（Vertical Integrated Liquid）雷达回波数据。

SEVIR 目录结构预期：
    <data_root>/
        SEVIR_VIL_RANDOMEVENTS_2017_0501_0831.h5
        SEVIR_VIL_RANDOMEVENTS_2018_0501_0831.h5
        SEVIR_VIL_STORMEVENTS_2017_0701_1231.h5
        ...（共 14 个文件，RANDOM + STORM，2017~2019）

数据集划分策略（按年份，与 SEVIR 论文一致，避免数据泄露）：
    train：文件名中年份 in train_years（默认 2017、2018）的所有事件，
           再从中随机留出 val_ratio 比例作为 val
    test： 文件名中年份 in test_years（默认 2019）的所有事件

h5 文件中 vil 数据集的维度顺序为 [N, H, W, T]，读取后自动转置为 [N, T, H, W]。
本模块从每个事件中随机采样连续 (in_seq_len + out_seq_len) 帧，
并将像素值离散化为 num_bins 个类别。
"""

import os
import re
import glob
import random
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


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


def _extract_year(fpath: str) -> int:
    """
    从文件名中提取年份，例如：
        SEVIR_VIL_RANDOMEVENTS_2018_0501_0831.h5  ->  2018
    找不到则返回 -1。
    """
    basename = os.path.basename(fpath)
    m = re.search(r"_(\d{4})_", basename)
    if m:
        return int(m.group(1))
    return -1


class SEVIRVILDataset(Dataset):
    """
    SEVIR VIL 数据集。

    每个样本包含：
        input_frames:  [in_seq_len, 1, H, W]  float32，归一化到 [0, 1]
        target_bins:   [out_seq_len, H, W]     int64，bin 索引 [0, num_bins-1]
        future_frames: [out_seq_len, 1, H, W]  float32，归一化到 [0, 1]
                       （供 OPSD 教师模型使用的特权信息）

    Args:
        data_root:    存放 SEVIR VIL .h5 文件的目录
        in_seq_len:   输入历史帧数
        out_seq_len:  预测未来帧数
        num_bins:     离散化 bin 数量
        vil_max:      VIL 最大像素值
        split:        'train' | 'val' | 'test'
        train_years:  用于训练/验证的年份列表，默认 [2017, 2018]
        test_years:   用于测试的年份列表，默认 [2019]
        val_ratio:    从 train_years 事件中留出多少比例作为 val
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
        train_years: list = None,
        test_years: list = None,
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

        if train_years is None:
            train_years = [2017, 2018]
        if test_years is None:
            test_years = [2019]

        # 扫描 data_root 目录下所有 h5 文件，按年份分组
        all_h5 = sorted(glob.glob(os.path.join(data_root, "*.h5")))
        if len(all_h5) == 0:
            raise FileNotFoundError(
                f"在 {data_root} 下未找到任何 .h5 文件，"
                f"请确认 data_root 指向包含 SEVIR VIL .h5 文件的目录。"
            )

        train_files = [f for f in all_h5 if _extract_year(f) in train_years]
        test_files  = [f for f in all_h5 if _extract_year(f) in test_years]

        print(f"[SEVIRVILDataset] 发现 {len(all_h5)} 个 h5 文件")
        print(f"  train_years={train_years} -> {len(train_files)} 个文件")
        print(f"  test_years={test_years}  -> {len(test_files)} 个文件")

        if split in ("train", "val"):
            source_files = train_files
        else:
            source_files = test_files

        if len(source_files) == 0:
            raise RuntimeError(
                f"split='{split}' 对应的文件列表为空，"
                f"请检查 train_years/test_years 配置与实际文件名是否匹配。"
            )

        # 构建 (文件路径, 事件索引, max_start) 的样本列表
        all_samples = []
        for fpath in source_files:
            try:
                with h5py.File(fpath, "r") as f:
                    if VIL_KEY not in f:
                        print(f"[WARNING] {os.path.basename(fpath)} 中无 '{VIL_KEY}' 数据集，跳过")
                        continue
                    # h5 中 vil 维度为 [N, H, W, T]，T 在最后一维
                    shape = f[VIL_KEY].shape
                    n_events = shape[0]
                    n_frames = shape[3]   # T 轴
                    if n_frames < self.total_seq_len:
                        print(f"[WARNING] {os.path.basename(fpath)} 帧数 {n_frames} < {self.total_seq_len}，跳过")
                        continue
                    max_start = n_frames - self.total_seq_len
                    for event_idx in range(n_events):
                        all_samples.append((fpath, event_idx, max_start))
            except Exception as e:
                print(f"[WARNING] 跳过文件 {fpath}，原因：{e}")

        if len(all_samples) == 0:
            raise RuntimeError(f"split='{split}' 未找到任何有效样本，请检查数据文件。")

        # train/val 从同一批文件中按事件索引随机划分（test 直接全用）
        if split in ("train", "val"):
            rng = random.Random(seed)
            indices = list(range(len(all_samples)))
            rng.shuffle(indices)
            n_val = int(len(indices) * val_ratio)
            if split == "val":
                selected = indices[:n_val]
            else:
                selected = indices[n_val:]
        else:
            selected = list(range(len(all_samples)))

        self.samples = all_samples
        self.indices = selected

        print(
            f"[SEVIRVILDataset] split={split}, "
            f"使用样本数={len(self.indices)}, "
            f"来源文件事件总数={len(all_samples)}"
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
            # h5 原始 shape: [N, H, W, T]，按时间轴切片后为 [H, W, total_seq_len]
            raw = f[VIL_KEY][event_idx, :, :, start:end].astype(np.float32)
            # raw shape: [H, W, total_seq_len] -> 转置为 [total_seq_len, H, W]
            frames = np.transpose(raw, (2, 0, 1))

        # 归一化到 [0, 1]
        frames_norm = frames / self.vil_max  # [T, H, W]

        # 拆分历史帧和未来帧
        input_raw = frames_norm[: self.in_seq_len]    # [in_seq_len, H, W]
        future_raw = frames_norm[self.in_seq_len:]    # [out_seq_len, H, W]

        # 增加通道维度 -> [T, 1, H, W]
        input_frames = input_raw[:, np.newaxis, :, :]
        future_frames = future_raw[:, np.newaxis, :, :]

        # 将未来帧的原始像素值离散化为 bin 索引
        future_raw_uint8 = future_raw * self.vil_max  # 还原到 [0, 255]
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
        cfg: 完整配置字典（对应 default.yaml）

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
        train_years=data_cfg.get("train_years", [2017, 2018]),
        test_years=data_cfg.get("test_years", [2019]),
        val_ratio=data_cfg.get("val_ratio", 0.1),
        seed=train_cfg["seed"],
    )

    train_ds = SEVIRVILDataset(split="train", **common_kwargs)
    val_ds   = SEVIRVILDataset(split="val",   **common_kwargs)
    test_ds  = SEVIRVILDataset(split="test",  **common_kwargs)

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
