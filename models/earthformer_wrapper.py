"""
EarthFormer Wrapper
====================
封装 earthformer 包中的 CuboidTransformerModel，
适配本项目的输入输出格式。

输入：[B, T_in, 1, H, W]，归一化 [0,1]
输出：[B, T_out, 1, H, W]，归一化 [0,1]

EarthFormer 原始接口：
  输入：[B, T, H, W, C]（时间在前，通道在后）
  输出：[B, T, H, W, C]
"""

import torch
import torch.nn as nn


class EarthFormerWrapper(nn.Module):

    def __init__(
        self,
        in_seq_len: int = 6,
        out_seq_len: int = 6,
        img_size: int = 128,
        base_units: int = 64,
    ):
        super().__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len

        from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel

        self.model = CuboidTransformerModel(
            input_shape=[in_seq_len, img_size, img_size, 1],
            target_shape=[out_seq_len, img_size, img_size, 1],
            base_units=base_units,
            # Encoder：2 层，每层深度 4
            enc_depth=[4, 4],
            enc_attn_patterns=None,
            enc_cuboid_size=[(4, 4, 4), (4, 4, 4)],
            enc_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
            enc_shift_size=[(0, 0, 0), (0, 0, 0)],
            enc_use_inter_ffn=True,
            # Decoder：2 层，每层深度 2
            dec_depth=[2, 2],
            dec_cross_start=0,
            dec_self_attn_patterns=None,
            dec_self_cuboid_size=[(4, 4, 4), (4, 4, 4)],
            dec_self_cuboid_strategy=[('l', 'l', 'l'), ('d', 'd', 'd')],
            dec_self_shift_size=[(1, 1, 1), (0, 0, 0)],
            dec_cross_attn_patterns=None,
            dec_cross_cuboid_hw=[(4, 4), (4, 4)],
            dec_cross_cuboid_strategy=[('l', 'l', 'l'), ('d', 'l', 'l')],
            dec_cross_shift_hw=[(0, 0), (0, 0)],
            dec_cross_n_temporal=[1, 2],
            dec_use_inter_ffn=True,
            dec_hierarchical_pos_embed=False,
            # Global vectors
            num_global_vectors=4,
            use_dec_self_global=True,
            dec_self_update_global=True,
            use_dec_cross_global=True,
            use_global_vector_ffn=True,
            use_global_self_attn=False,
            # 其他
            num_heads=4,
            attn_drop=0.0,
            proj_drop=0.0,
            ffn_drop=0.0,
            ffn_activation='leaky',
            norm_layer='layer_norm',
            padding_type='ignore',
            pos_embed_type='t+hw',
            use_relative_pos=True,
            downsample=2,
            downsample_type='patch_merge',
            upsample_type='upsample',
            upsample_kernel_size=3,
            initial_downsample_type='conv',
            initial_downsample_scale=1,
            initial_downsample_conv_layers=2,
            final_upsample_conv_layers=2,
            z_init_method='nearest_interp',
            checkpoint_level=True,
        )

    def forward(self, input_frames: torch.Tensor) -> torch.Tensor:
        """
        input_frames: [B, T_in, 1, H, W]
        returns:      [B, T_out, 1, H, W]
        """
        B, T, C, H, W = input_frames.shape
        # [B, T, 1, H, W] -> [B, T, H, W, 1]
        x = input_frames.permute(0, 1, 3, 4, 2)
        out = self.model(x)                    # [B, T_out, H, W, 1]
        out = torch.sigmoid(out)               # 限制到 [0,1]
        # [B, T_out, H, W, 1] -> [B, T_out, 1, H, W]
        out = out.permute(0, 1, 4, 2, 3)
        return out


def build_earthformer(cfg: dict) -> EarthFormerWrapper:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    return EarthFormerWrapper(
        in_seq_len=data_cfg["in_seq_len"],
        out_seq_len=data_cfg["out_seq_len"],
        img_size=data_cfg.get("crop_size", 128),
        base_units=model_cfg.get("base_units", 64),
    )
