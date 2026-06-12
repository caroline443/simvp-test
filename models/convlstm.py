"""
ConvLSTM — Shi et al., NeurIPS 2015
=====================================
在空间维度上做卷积的 LSTM，直接建模时空序列。

输入：[B, T_in, C, H, W]
输出：[B, T_out, C_out, H, W]（C_out=1，归一化 VIL 值）
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """单个 ConvLSTM 时间步。"""

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_ch = hidden_ch
        # 四个门（i, f, o, g）合并成一个卷积，效率更高
        self.conv = nn.Conv2d(
            in_ch + hidden_ch, 4 * hidden_ch,
            kernel_size, padding=pad, bias=True
        )

    def forward(self, x, h, c):
        """
        x: [B, in_ch, H, W]
        h, c: [B, hidden_ch, H, W]
        """
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, H, W, device):
        return (
            torch.zeros(B, self.hidden_ch, H, W, device=device),
            torch.zeros(B, self.hidden_ch, H, W, device=device),
        )


class ConvLSTM(nn.Module):
    """
    多层 ConvLSTM，一次性预测 T_out 帧（非自回归）。

    编码阶段：用 T_in 帧历史更新 LSTM 状态
    解码阶段：用零输入滚动 T_out 步，输出每步预测
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 4,
        in_seq_len: int = 12,
        out_seq_len: int = 12,
    ):
        super().__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.num_layers = num_layers
        self.in_channels = in_channels

        # 构建多层 ConvLSTM
        cells = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            cells.append(ConvLSTMCell(in_ch, hidden_channels))
        self.cells = nn.ModuleList(cells)

        # 最终输出头：hidden → 1通道 VIL，sigmoid 限制到 [0,1]
        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, input_frames: torch.Tensor) -> torch.Tensor:
        """
        input_frames: [B, T_in, 1, H, W]，归一化 [0,1]
        returns: [B, T_out, 1, H, W]，归一化 [0,1]
        """
        B, T_in, C, H, W = input_frames.shape
        device = input_frames.device

        # 初始化所有层的隐状态
        states = [cell.init_hidden(B, H, W, device) for cell in self.cells]

        # 编码阶段：输入历史帧，更新状态
        for t in range(T_in):
            x = input_frames[:, t]      # [B, 1, H, W]
            for i, cell in enumerate(self.cells):
                h, c = states[i]
                h, c = cell(x, h, c)
                states[i] = (h, c)
                x = h                   # 下一层的输入

        # 解码阶段：以零输入滚动 T_out 步
        # 第一层 in_ch=in_channels，解码时用零帧作为输入
        # 从第二步开始用上一步的预测帧作为输入（自回归）
        preds = []
        prev_pred = torch.zeros(B, self.in_channels, H, W, device=device)
        for t in range(self.out_seq_len):
            x = prev_pred
            for i, cell in enumerate(self.cells):
                h, c = states[i]
                h, c = cell(x, h, c)
                states[i] = (h, c)
                x = h
            pred = torch.sigmoid(self.output_conv(x))  # [B, 1, H, W]
            preds.append(pred)
            prev_pred = pred  # 下一步用当前预测作为输入

        return torch.stack(preds, dim=1)    # [B, T_out, 1, H, W]


def build_convlstm(cfg: dict) -> ConvLSTM:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return ConvLSTM(
        in_channels=model_cfg["in_channels"],
        hidden_channels=model_cfg["hidden_channels"],
        num_layers=model_cfg.get("num_layers", 4),
        in_seq_len=data_cfg["in_seq_len"],
        out_seq_len=data_cfg["out_seq_len"],
    )
