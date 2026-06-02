"""
SimVP Vanilla — 原汁原味版本
=============================
与原论文对齐：
  - 一次性预测所有输出帧（非自回归）
  - MSE 损失，输出直接是 VIL 像素值（归一化到 [0,1]）
  - Encoder 共享权重处理输入帧，Translator 时序传播，Decoder 输出 out_seq_len 帧

网络结构：
  SpatialEncoder (in_seq_len 帧) -> TemporalTranslator -> SpatialDecoder (out_seq_len 帧)
"""

import torch
import torch.nn as nn
from .modules import SpatialEncoder, TemporalTranslator, SafeGroupNorm
from .mamba_translator import MambaTemporalTranslator


class VanillaSpatialDecoder(nn.Module):
    """
    将特征图 [B*T_out, enc_ch, h, w] 解码回 [B*T_out, 1, H, W]。
    输出单通道（VIL 归一化值），用 Sigmoid 限制到 [0,1]。
    """

    def __init__(self, enc_ch: int, hidden_ch: int = 64, n_layers: int = 4):
        super().__init__()
        layers = []
        ch = enc_ch
        for _ in range(n_layers - 1):
            out_ch = max(hidden_ch, ch // 2)
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
                SafeGroupNorm(max(1, out_ch // 4), out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            ch = out_ch
        layers.append(
            nn.ConvTranspose2d(ch, 1, kernel_size=4, stride=2, padding=1, bias=True)
        )
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.decoder(x))  # [B*T, 1, H, W], 值域 [0,1]


class SimVPVanilla(nn.Module):
    """
    原版 SimVP：一次性预测，MSE 损失。

    输入：[B, in_seq_len, 1, H, W]  （归一化到 [0,1] 的 VIL）
    输出：[B, out_seq_len, 1, H, W] （归一化到 [0,1] 的预测 VIL）
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        encoder_layers: int = 4,
        translator_layers: int = 4,
        decoder_layers: int = 4,
        in_seq_len: int = 12,
        out_seq_len: int = 12,
        use_checkpoint: bool = False,
        translator_type: str = "inception",
    ):
        super().__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len

        self.encoder = SpatialEncoder(
            in_ch=in_channels,
            hidden_ch=hidden_channels,
            n_layers=encoder_layers,
            use_checkpoint=use_checkpoint,
        )
        enc_ch = self.encoder.out_channels

        if translator_type == "mamba":
            self.translator = MambaTemporalTranslator(
                hidden_ch=enc_ch,
                seq_len=in_seq_len,
                n_layers=translator_layers,
                use_checkpoint=use_checkpoint,
            )
        else:
            self.translator = TemporalTranslator(
                hidden_ch=enc_ch,
                seq_len=in_seq_len,
                n_layers=translator_layers,
                use_checkpoint=use_checkpoint,
            )

        # 将 translator 输出的最后一帧特征展开为 out_seq_len 帧
        self.frame_expand = nn.Conv2d(enc_ch, enc_ch * out_seq_len, 1, bias=False)

        self.decoder = VanillaSpatialDecoder(
            enc_ch=enc_ch,
            hidden_ch=hidden_channels,
            n_layers=decoder_layers,
        )

    def forward(self, input_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_frames: [B, in_seq_len, 1, H, W]

        Returns:
            pred: [B, out_seq_len, 1, H, W]
        """
        B, T_in, C, H, W = input_frames.shape

        # 编码所有输入帧
        frames_flat = input_frames.view(B * T_in, C, H, W)
        features = self.encoder(frames_flat)          # [B*T_in, enc_ch, h, w]
        _, enc_ch, h, w = features.shape

        # 时序传播
        features = self.translator(features)           # [B*T_in, enc_ch, h, w]

        # 取最后一帧特征，展开为 out_seq_len 帧
        features = features.view(B, T_in, enc_ch, h, w)
        last_feat = features[:, -1]                    # [B, enc_ch, h, w]
        expanded = self.frame_expand(last_feat)        # [B, enc_ch*T_out, h, w]
        expanded = expanded.view(B * self.out_seq_len, enc_ch, h, w)

        # 解码
        pred = self.decoder(expanded)                  # [B*T_out, 1, H, W]
        pred = pred.view(B, self.out_seq_len, 1, H, W)
        return pred


def build_vanilla_model(cfg: dict) -> SimVPVanilla:
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return SimVPVanilla(
        in_channels=model_cfg["in_channels"],
        hidden_channels=model_cfg["hidden_channels"],
        encoder_layers=model_cfg["encoder_layers"],
        translator_layers=model_cfg["translator_layers"],
        decoder_layers=model_cfg["decoder_layers"],
        in_seq_len=data_cfg["in_seq_len"],
        out_seq_len=data_cfg["out_seq_len"],
        use_checkpoint=model_cfg.get("use_checkpoint", False),
        translator_type=model_cfg.get("translator_type", "inception"),
    )
