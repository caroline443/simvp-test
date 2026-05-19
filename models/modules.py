"""
SimVP 子模块
============
包含 Spatial Encoder、Temporal Translator（基于 Inception 模块）、Spatial Decoder。
所有模块均为纯 2D/3D CNN，无 Attention，显存友好。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ---------------------------------------------------------------------------
# 基础卷积块
# ---------------------------------------------------------------------------

class ConvBnAct(nn.Module):
    """Conv2d + GroupNorm + LeakyReLU"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, groups: int = 1):
        super().__init__()
        num_groups = min(groups, out_ch)
        # GroupNorm 要求 out_ch 能被 num_groups 整除
        while out_ch % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride,
            padding=padding, bias=False
        )
        self.norm = nn.GroupNorm(num_groups, out_ch)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# Spatial Encoder
# ---------------------------------------------------------------------------

class SpatialEncoder(nn.Module):
    """
    将单帧图像 [B, in_ch, H, W] 编码为特征图 [B, hidden_ch, H/16, W/16]。
    使用 4 层步长为 2 的卷积进行下采样（每层 stride=2，共 2^4=16 倍下采样）。
    384x384 -> 24x24

    use_checkpoint: 启用梯度检查点（Gradient Checkpointing）。
        原理：前向传播时不保存中间激活值，反向传播时重新计算。
        代价：约增加 20% 计算时间。
        收益：激活值显存占用降低约 60%，对 384x384 高分辨率场景至关重要。
    """

    def __init__(self, in_ch: int = 1, hidden_ch: int = 64, n_layers: int = 4,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        layers = []
        ch = in_ch
        for i in range(n_layers):
            out_ch = hidden_ch * (2 ** min(i, 2))  # 64, 128, 256, 256
            out_ch = min(out_ch, hidden_ch * 4)    # 上限 256
            layers.append(ConvBnAct(ch, out_ch, kernel_size=3, stride=2, padding=1))
            ch = out_ch
        # 拆成 ModuleList 以便逐层应用 checkpoint
        self.layers = nn.ModuleList(layers)
        self.out_channels = ch

    def forward(self, x):
        # x: [B, in_ch, H, W]
        for layer in self.layers:
            if self.use_checkpoint and x.requires_grad:
                # 梯度检查点：不保存该层的中间激活值，反向时重算
                x = grad_checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x  # [B, out_channels, H/16, W/16]


# ---------------------------------------------------------------------------
# Inception 模块（用于 Temporal Translator）
# ---------------------------------------------------------------------------

class InceptionBlock(nn.Module):
    """
    轻量化 Inception 模块，在特征图的空间维度上捕捉多尺度局部模式。
    输入输出通道数相同，便于堆叠。
    """

    def __init__(self, ch: int):
        super().__init__()
        branch_ch = ch // 4

        self.branch1 = nn.Sequential(
            nn.Conv2d(ch, branch_ch, 1, bias=False),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(ch, branch_ch, 1, bias=False),
            nn.Conv2d(branch_ch, branch_ch, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(ch, branch_ch, 1, bias=False),
            nn.Conv2d(branch_ch, branch_ch, 5, padding=2, bias=False),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(ch, branch_ch, 1, bias=False),
            nn.GroupNorm(max(1, branch_ch // 4), branch_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(branch_ch * 4, ch, 1, bias=False),
            nn.GroupNorm(max(1, ch // 4), ch),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(x)
        out = torch.cat([b1, b3, b5, bp], dim=1)
        out = self.proj(out)
        return self.act(out + x)  # 残差连接


# ---------------------------------------------------------------------------
# Temporal Translator
# ---------------------------------------------------------------------------

class TemporalTranslator(nn.Module):
    """
    在时间维度上传播特征，将 T 帧的编码特征映射为 T 帧的预测特征。

    输入：[B*T, C, h, w]（将 Batch 和 Time 合并后的特征图）
    输出：[B*T, C, h, w]

    实现方式：
    1. 将 [B*T, C, h, w] reshape 为 [B, T*C, h, w]，在通道维度上融合时序信息。
    2. 通过 n_layers 个 Inception 模块处理。
    3. reshape 回 [B*T, C, h, w]。
    """

    def __init__(self, hidden_ch: int, seq_len: int, n_layers: int = 4,
                 use_checkpoint: bool = False):
        super().__init__()
        self.hidden_ch = hidden_ch
        self.seq_len = seq_len
        self.use_checkpoint = use_checkpoint
        temporal_ch = hidden_ch * seq_len

        # 时序融合：将 T 帧特征在通道维度拼接后用 Inception 处理
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(temporal_ch, temporal_ch, 1, bias=False),
            nn.GroupNorm(max(1, seq_len), temporal_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.inception_blocks = nn.ModuleList(
            [InceptionBlock(temporal_ch) for _ in range(n_layers)]
        )
        self.out_proj = nn.Conv2d(temporal_ch, temporal_ch, 1, bias=False)

    def forward(self, x):
        # x: [B*T, C, h, w]
        B_T, C, h, w = x.shape
        T = self.seq_len
        B = B_T // T

        # [B, T*C, h, w]
        x = x.view(B, T * C, h, w)
        x = self.temporal_conv(x)
        for block in self.inception_blocks:
            if self.use_checkpoint and x.requires_grad:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.out_proj(x)
        # [B*T, C, h, w]
        x = x.view(B * T, C, h, w)
        return x


# ---------------------------------------------------------------------------
# Spatial Decoder
# ---------------------------------------------------------------------------

class SpatialDecoder(nn.Module):
    """
    将特征图 [B, enc_ch, H/16, W/16] 解码回 [B, num_bins, H, W]。
    使用 4 层转置卷积进行上采样（每层 stride=2，共 2^4=16 倍上采样）。
    24x24 -> 384x384
    """

    def __init__(self, enc_ch: int, hidden_ch: int = 64,
                 n_layers: int = 4, num_bins: int = 16):
        super().__init__()
        layers = []
        ch = enc_ch
        for i in range(n_layers - 1):
            # 逐步减少通道数
            out_ch = max(hidden_ch, ch // 2)
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(max(1, out_ch // 4), out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch = out_ch

        # 最后一层输出 num_bins 个通道（Logit）
        layers.append(
            nn.ConvTranspose2d(ch, num_bins, kernel_size=4, stride=2, padding=1, bias=True)
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, enc_ch, H/16, W/16]
        return self.decoder(x)  # [B, num_bins, H, W]
