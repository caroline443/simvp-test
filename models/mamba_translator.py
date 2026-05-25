"""
Mamba Temporal Translator
=========================
用 Mamba SSM（选择性状态空间模型）替换原始 TemporalTranslator 中的 Inception 块。

核心思想：
  对每个空间位置 (h, w)，将 T 帧特征视为长度 T 的序列，
  用 Mamba SSM 显式建模每个空间位置的时序动态，
  再用一层轻量空间卷积恢复跨位置的信息交互。

接口与 TemporalTranslator 完全相同，SimVP 的 Encoder/Decoder 无需修改。
纯 PyTorch 实现，无需安装 mamba-ssm 包。

性能设计：
  - A_bar / B_bar 在循环外一次性计算（减少 GPU kernel 启动次数）
  - 默认 expand=1（d_inner=C，而非 2C）、d_state=8，减少状态张量尺寸
  - MambaBlock 全程 float32，避免 AMP float16 在 LayerNorm / exp 中溢出

参考：
  Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .modules import ConvBnAct


# ---------------------------------------------------------------------------
# Selective Scan（S6 核心，向量化版本）
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
    Mamba S6 选择性扫描（向量化优化版）。

    优化关键：A_bar / B_bar 在循环外一次性计算为 [B, L, d, N] 张量，
    将 L 次独立 torch.exp kernel 合并为 1 次大操作，
    显著减少 GPU kernel 启动次数，提升 GPU 利用率。

    对每个时间步 i：
      h[i] = A_bar[i] * h[i-1] + B_bar[i] * x[i]
      y[i] = sum(C[i] * h[i]) + D * x[i]

    L 通常为 5~13（帧数），顺序循环开销可忽略。
    """
    B_sz, L, d = x.shape
    N = A.shape[1]

    # 一次性计算所有步的 A_bar 和 B_bar，避免循环内重复 kernel 启动
    # dt[:, :, :, None]: [B, L, d, 1]  A[None, None]: [1, 1, d, N]
    A_bar = torch.exp(dt[:, :, :, None] * A[None, None, :, :])  # [B, L, d, N]
    B_bar = dt[:, :, :, None] * B[:, :, None, :]                # [B, L, d, N]

    h  = x.new_zeros(B_sz, d, N)
    ys = []
    for i in range(L):
        # 循环内只做简单逐元素运算，无 exp / 广播，kernel 开销极小
        h  = A_bar[:, i] * h + B_bar[:, i] * x[:, i, :, None]  # [B, d, N]
        yi = (h * C[:, i, None, :]).sum(-1) + D * x[:, i]       # [B, d]
        ys.append(yi)

    return torch.stack(ys, dim=1)   # [B, L, d]


# ---------------------------------------------------------------------------
# MambaBlock
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """
    单个 Mamba 块，含残差连接。输入输出均为 [B, L, d_model]。

    结构（参照原论文图 3）：
      LayerNorm -> 输入投影 (expand 倍) -> 因果 1D 卷积 + SiLU
      -> S6 选择性扫描 -> 门控 (SiLU) -> 输出投影 -> 残差

    数值稳定性设计：
      - 入口处统一 .float()，LayerNorm / exp / log 全程在 float32 下计算
      - dt 限幅到 [1e-4, 1.0]，防止状态爆炸或梯度消失
      - 出口转回 orig_dtype，兼容 AMP 混合精度训练

    Args:
        d_model:  特征维度（= encoder 输出通道数）
        d_state:  SSM 隐状态维度 N（默认 8，权衡精度与速度）
        d_conv:   因果 1D 卷积核宽度（默认 4）
        expand:   内部扩展比例（默认 1，d_inner = d_model；
                  设为 2 可增加容量但会使状态张量翻倍）
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 8,
        d_conv:  int = 4,
        expand:  int = 1,
    ):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = int(expand * d_model)
        d_inner       = self.d_inner
        dt_rank       = max(1, math.ceil(d_model / 16))
        self.dt_rank  = dt_rank

        self.norm     = nn.LayerNorm(d_model)

        # 输入投影：x branch + gating branch
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)

        # 因果深度可分离 1D 卷积
        self.conv1d   = nn.Conv1d(
            d_inner, d_inner, d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )

        # SSM 参数
        self.x_proj   = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj  = nn.Linear(dt_rank, d_inner, bias=True)

        # A：对数形式保持负值约束
        A = torch.arange(1, d_state + 1, dtype=torch.float32) \
               .unsqueeze(0).expand(d_inner, -1).clone()
        self.A_log    = nn.Parameter(torch.log(A))

        # D：跳跃连接系数
        self.D        = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        residual   = x
        orig_dtype = x.dtype

        # 全程 float32：LayerNorm / exp / log 在 float16 下易溢出产生 NaN
        x = x.float()

        x   = self.norm(x)
        B_sz, L, _ = x.shape

        # 扩展并拆分为 x_ 和门控 z
        xz  = self.in_proj(x)                    # [B, L, 2*d_inner]
        x_, z = xz.split(self.d_inner, dim=-1)   # each [B, L, d_inner]

        # 因果 1D 卷积（序列维度 L）
        x_  = x_.transpose(1, 2)                 # [B, d_inner, L]
        x_  = self.conv1d(x_)[..., :L]           # 截断因果填充
        x_  = x_.transpose(1, 2)                 # [B, L, d_inner]
        x_  = F.silu(x_)

        # SSM
        A   = -torch.exp(self.A_log.float())      # [d_inner, d_state]，负值
        ssm = self.x_proj(x_)                     # [B, L, dt_rank + 2*N]
        dt_raw, B_ssm, C_ssm = ssm.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # dt 限幅：防止离散化步长过大导致状态爆炸（A_bar → 1）
        dt  = F.softplus(self.dt_proj(dt_raw)).clamp(min=1e-4, max=1.0)

        y   = selective_scan(x_, dt, A, B_ssm, C_ssm, self.D.float())
        y   = y * F.silu(z)

        # 输出投影 + 残差：在 float32 下相加再转回，防止训练后期
        # out_proj 输出幅值增大超过 float16 上限（65504）→ inf → NaN
        out = self.out_proj(y)                          # float32
        return (out + residual.float()).to(orig_dtype)  # float32 加法再转回


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

    Args:
        hidden_ch:     encoder 输出通道数（即 d_model）
        seq_len:       输入序列长度（= in_seq_len）
        n_layers:      Mamba 块数量（默认 4）
        d_state:       SSM 隐状态维度（默认 8）
        expand:        MambaBlock 内部扩展比例（默认 1）
        use_checkpoint: 启用梯度检查点节省显存
    """

    def __init__(
        self,
        hidden_ch:      int,
        seq_len:        int,
        n_layers:       int  = 4,
        d_state:        int  = 8,
        expand:         int  = 1,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.hidden_ch      = hidden_ch
        self.seq_len        = seq_len
        self.use_checkpoint = use_checkpoint

        self.mamba_blocks = nn.ModuleList([
            MambaBlock(hidden_ch, d_state=d_state, expand=expand)
            for _ in range(n_layers)
        ])

        # 空间交互：轻量 3×3 卷积，恢复跨位置的局部上下文
        self.spatial_mix = ConvBnAct(hidden_ch, hidden_ch)

    def _run_mamba(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B*h*w, T, C]"""
        for block in self.mamba_blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B*T, C, h, w]
        B_T, C, h, w = x.shape
        T = self.seq_len
        B = B_T // T

        # [B*T, C, h, w] -> [B*h*w, T, C]
        x = x.view(B, T, C, h, w)
        x = x.permute(0, 3, 4, 1, 2).contiguous()   # [B, h, w, T, C]
        x = x.view(B * h * w, T, C)

        # Mamba 时序建模
        if self.use_checkpoint and x.requires_grad:
            x = grad_checkpoint(self._run_mamba, x, use_reentrant=False)
        else:
            x = self._run_mamba(x)

        # [B*h*w, T, C] -> [B*T, C, h, w]
        x = x.view(B, h, w, T, C)
        x = x.permute(0, 3, 4, 1, 2).contiguous()   # [B, T, C, h, w]
        x = x.view(B * T, C, h, w)

        # 空间交互
        x = self.spatial_mix(x)

        return x
