"""
Mamba Temporal Translator
=========================
用 Mamba SSM（选择性状态空间模型）替换原始 TemporalTranslator 中的 Inception 块。

核心思想：
  原始 TemporalTranslator 将 T 帧特征在通道维度拼接（T×C），
  用 2D Inception 卷积隐式建模时序关系——这种方式无法显式捕获长程时序依赖。

  MambaTemporalTranslator 的做法：
    1. 对每个空间位置 (h, w)，将 T 帧特征视为长度 T 的序列
    2. 用 Mamba SSM 显式建模每个空间位置的时序动态
    3. 用一层轻量空间卷积恢复跨位置的空间信息交互

  接口与 TemporalTranslator 完全相同，SimVP 的 Encoder/Decoder 无需修改。

  纯 PyTorch 实现，无需安装 mamba-ssm 包。
  序列长度 T=5~13（帧数），短序列下顺序扫描足够快。

参考：
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023.
  https://arxiv.org/abs/2312.00752
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .modules import SafeGroupNorm, ConvBnAct


# ---------------------------------------------------------------------------
# Selective Scan（S6 核心）
# ---------------------------------------------------------------------------

def selective_scan(
    x:  torch.Tensor,   # [B, L, d]    float32
    dt: torch.Tensor,   # [B, L, d]    float32
    A:  torch.Tensor,   # [d, N]       float32，负值
    B:  torch.Tensor,   # [B, L, N]    float32
    C:  torch.Tensor,   # [B, L, N]    float32
    D:  torch.Tensor,   # [d]          float32
) -> torch.Tensor:
    """
    Mamba S6 顺序选择性扫描。

    对每个时间步 i：
      A_bar[i] = exp(dt[i] * A)          离散化转移矩阵
      B_bar[i] = dt[i] * B[i]            离散化输入矩阵
      h[i]     = A_bar[i] * h[i-1] + B_bar[i] * x[i]   状态更新
      y[i]     = sum(C[i] * h[i]) + D * x[i]            输出

    L（帧数）通常为 5~13，顺序循环开销可忽略。
    """
    B_sz, L, d = x.shape
    N = A.shape[1]

    h = x.new_zeros(B_sz, d, N)
    ys = []

    for i in range(L):
        dt_i  = dt[:, i]                                       # [B, d]
        A_bar = torch.exp(dt_i[:, :, None] * A[None, :, :])   # [B, d, N]
        B_bar = dt_i[:, :, None] * B[:, i, None, :]           # [B, d, N]
        x_i   = x[:, i]                                        # [B, d]
        h     = A_bar * h + B_bar * x_i[:, :, None]           # [B, d, N]
        y_i   = (h * C[:, i, None, :]).sum(-1) + D * x_i      # [B, d]
        ys.append(y_i)

    return torch.stack(ys, dim=1)   # [B, L, d]


# ---------------------------------------------------------------------------
# MambaBlock
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    单个 Mamba 块，含残差连接。输入输出均为 [B, L, d_model]。

    结构（参照原论文图 3）：
      LayerNorm -> 输入投影 (2×expand) -> 因果 1D 卷积 + SiLU
      -> S6 选择性扫描 -> 门控 (SiLU) -> 输出投影 -> 残差

    全程在 float32 下计算，避免 AMP float16 在 exp/log 中溢出产生 NaN。
    输出在残差加法前转回原始 dtype，保持与 AMP 的兼容性。

    Args:
        d_model:  特征维度（=encoder 输出通道数）
        d_state:  SSM 隐状态维度 N（默认 16，与原论文一致）
        d_conv:   因果 1D 卷积核宽度（默认 4）
        expand:   内部扩展比例（默认 2，d_inner = expand * d_model）
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv:  int = 4,
        expand:  int = 2,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = int(expand * d_model)
        d_inner       = self.d_inner
        dt_rank       = max(1, math.ceil(d_model / 16))
        self.dt_rank  = dt_rank

        self.norm     = nn.LayerNorm(d_model)

        # 输入投影：x + gating branch
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)

        # 因果深度可分离 1D 卷积（对序列维度 L 做局部上下文）
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # SSM 参数
        self.x_proj   = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, d_inner, bias=True)

        # A：固定结构参数（对数形式保持负值约束）
        A = torch.arange(1, d_state + 1, dtype=torch.float32) \
               .unsqueeze(0).expand(d_inner, -1).clone()
        self.A_log    = nn.Parameter(torch.log(A))

        # D：跳跃连接系数
        self.D        = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        residual = x
        orig_dtype = x.dtype

        x   = self.norm(x)
        B_sz, L, _ = x.shape

        # 扩展并拆分为 x_ 和门控 z
        xz  = self.in_proj(x.float())               # [B, L, 2*d_inner]  float32
        x_, z = xz.split(self.d_inner, dim=-1)      # each [B, L, d_inner]

        # 因果 1D 卷积（在 L 维度上做）
        x_  = x_.transpose(1, 2)                    # [B, d_inner, L]
        x_  = self.conv1d(x_)[..., :L]              # 截断因果填充
        x_  = x_.transpose(1, 2)                    # [B, L, d_inner]
        x_  = F.silu(x_)

        # SSM 参数（输入依赖）
        A   = -torch.exp(self.A_log.float())         # [d_inner, d_state]，负值
        ssm = self.x_proj(x_)                        # [B, L, dt_rank + 2*N]
        dt_raw, B_ssm, C_ssm = ssm.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt  = F.softplus(self.dt_proj(dt_raw))       # [B, L, d_inner]，正值

        # S6 选择性扫描
        y   = selective_scan(x_, dt, A, B_ssm, C_ssm, self.D.float())  # [B, L, d_inner]

        # 门控
        y   = y * F.silu(z)

        # 输出投影 + 残差
        out = self.out_proj(y).to(orig_dtype)        # 转回原始 dtype
        return out + residual


# ---------------------------------------------------------------------------
# MambaTemporalTranslator
# ---------------------------------------------------------------------------

class MambaTemporalTranslator(nn.Module):
    """
    基于 Mamba SSM 的时序翻译器，与 TemporalTranslator 接口完全相同。

    设计：
      对每个空间位置独立建模 T 帧的时序演化，再用轻量空间卷积恢复位置间交互。

      [B*T, C, h, w]
        ↓  展开空间维度，聚合时序
      [B*h*w, T, C]       ← 每个空间位置是一条长度 T 的序列
        ↓  n_layers 个 MambaBlock（显式时序 SSM）
      [B*h*w, T, C]
        ↓  恢复形状
      [B*T, C, h, w]
        ↓  轻量 2D 空间卷积（跨位置信息交互）
      [B*T, C, h, w]

    与原始 TemporalTranslator 的对比：
      原始：时序信息通过 T×C 通道拼接 + 2D Inception 卷积隐式建模（无显式时序轴）
      本模块：每个空间位置有独立的 Mamba 时序状态，显式建模时序动态

    Args:
        hidden_ch:     encoder 输出通道数（即 d_model）
        seq_len:       输入序列长度（=in_seq_len，自回归每步固定）
        n_layers:      Mamba 块数量（默认 4，与 Inception 版本对齐）
        d_state:       SSM 隐状态维度（默认 16）
        use_checkpoint: 启用梯度检查点节省显存（384×384 时建议开启）
    """

    def __init__(
        self,
        hidden_ch:      int,
        seq_len:        int,
        n_layers:       int  = 4,
        d_state:        int  = 16,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.hidden_ch      = hidden_ch
        self.seq_len        = seq_len
        self.use_checkpoint = use_checkpoint

        self.mamba_blocks = nn.ModuleList([
            MambaBlock(hidden_ch, d_state=d_state)
            for _ in range(n_layers)
        ])

        # 空间交互：轻量 3×3 卷积，恢复跨位置的局部上下文
        self.spatial_mix = ConvBnAct(hidden_ch, hidden_ch)

    def _run_mamba(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*h*w, T, C]，顺序过所有 Mamba 块。"""
        for block in self.mamba_blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, h, w]
        B_T, C, h, w = x.shape
        T = self.seq_len
        B = B_T // T

        # [B*T, C, h, w] -> [B, T, C, h, w] -> [B*h*w, T, C]
        x = x.view(B, T, C, h, w)
        x = x.permute(0, 3, 4, 1, 2).contiguous()   # [B, h, w, T, C]
        x = x.view(B * h * w, T, C)

        # Mamba 时序建模
        if self.use_checkpoint and x.requires_grad:
            x = grad_checkpoint(self._run_mamba, x, use_reentrant=False)
        else:
            x = self._run_mamba(x)

        # [B*h*w, T, C] -> [B, h, w, T, C] -> [B, T, C, h, w] -> [B*T, C, h, w]
        x = x.view(B, h, w, T, C)
        x = x.permute(0, 3, 4, 1, 2).contiguous()   # [B, T, C, h, w]
        x = x.view(B * T, C, h, w)

        # 空间交互（轻量卷积，帮助相邻位置信息融合）
        x = self.spatial_mix(x)

        return x
