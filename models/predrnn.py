"""
PredRNN — Wang et al., NeurIPS 2017
======================================
在 ConvLSTM 基础上增加"空间-时序记忆流"（M），
跨层传递记忆，捕获更复杂的时空依赖。

输入：[B, T_in, C, H, W]
输出：[B, T_out, 1, H, W]（归一化 VIL 值）
"""

import torch
import torch.nn as nn


class PredRNNCell(nn.Module):
    """
    PredRNN 的核心单元，包含两个记忆：
      - C：层内时序记忆（和 ConvLSTM 一样）
      - M：跨层空间-时序记忆（PredRNN 的创新点）
    """

    def __init__(self, in_ch: int, hidden_ch: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_ch = hidden_ch
        pad = kernel_size // 2

        # 处理 x + h 的标准门
        self.conv_xh = nn.Conv2d(
            in_ch + hidden_ch, 4 * hidden_ch,
            kernel_size, padding=pad, bias=True
        )
        # 处理 M（空间-时序记忆）的门
        self.conv_m = nn.Conv2d(
            hidden_ch, 3 * hidden_ch,
            kernel_size, padding=pad, bias=True
        )
        # 输出门融合 h 和 M
        self.conv_o = nn.Conv2d(
            2 * hidden_ch, hidden_ch,
            1, bias=True
        )

    def forward(self, x, h, c, m):
        """
        x:    [B, in_ch, H, W]
        h, c: [B, hidden_ch, H, W]  层内记忆
        m:    [B, hidden_ch, H, W]  跨层空间-时序记忆
        """
        xh = torch.cat([x, h], dim=1)
        gates_xh = self.conv_xh(xh)
        i, f, g, o_xh = gates_xh.chunk(4, dim=1)

        gates_m = self.conv_m(m)
        i_m, f_m, g_m = gates_m.chunk(3, dim=1)

        i   = torch.sigmoid(i)
        f   = torch.sigmoid(f)
        g   = torch.tanh(g)
        i_m = torch.sigmoid(i_m)
        f_m = torch.sigmoid(f_m)
        g_m = torch.tanh(g_m)

        c_next = f * c + i * g
        m_next = f_m * m + i_m * g_m

        o = torch.sigmoid(o_xh + self.conv_o(torch.cat([c_next, m_next], dim=1)))
        h_next = o * torch.tanh(c_next)

        return h_next, c_next, m_next

    def init_hidden(self, B, H, W, device):
        zeros = lambda: torch.zeros(B, self.hidden_ch, H, W, device=device)
        return zeros(), zeros(), zeros()   # h, c, m


class PredRNN(nn.Module):
    """
    多层 PredRNN。
    M 记忆按"之字形"在层间和时间步间传递。
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

        cells = []
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            cells.append(PredRNNCell(in_ch, hidden_channels))
        self.cells = nn.ModuleList(cells)

        self.output_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, input_frames: torch.Tensor) -> torch.Tensor:
        """
        input_frames: [B, T_in, 1, H, W]
        returns:      [B, T_out, 1, H, W]
        """
        B, T_in, C, H, W = input_frames.shape
        device = input_frames.device

        # 初始化各层 h, c；共享一个全局 m
        h_list = []
        c_list = []
        for cell in self.cells:
            h, c, _ = cell.init_hidden(B, H, W, device)
            h_list.append(h)
            c_list.append(c)
        m = torch.zeros(B, self.cells[0].hidden_ch, H, W, device=device)

        def _step(x):
            nonlocal m
            for i, cell in enumerate(self.cells):
                h_list[i], c_list[i], m = cell(x, h_list[i], c_list[i], m)
                x = h_list[i]
            return x

        # 编码阶段
        for t in range(T_in):
            _step(input_frames[:, t])

        # 解码阶段
        preds = []
        x = torch.zeros(B, 1, H, W, device=device)
        for t in range(self.out_seq_len):
            out = _step(x)
            pred = torch.sigmoid(self.output_conv(out))
            preds.append(pred)

        return torch.stack(preds, dim=1)   # [B, T_out, 1, H, W]


def build_predrnn(cfg: dict) -> PredRNN:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return PredRNN(
        in_channels=model_cfg["in_channels"],
        hidden_channels=model_cfg["hidden_channels"],
        num_layers=model_cfg.get("num_layers", 4),
        in_seq_len=data_cfg["in_seq_len"],
        out_seq_len=data_cfg["out_seq_len"],
    )
