"""
SimVP 主网络
============
支持两种前向模式：
  - student_forward: 纯历史自回归，不使用特权信息（推理 / Baseline 训练）
  - teacher_forward: 使用真实未来帧作为特权上下文（OPSD 训练中的教师分支）

网络结构：
  SpatialEncoder -> TemporalTranslator -> SpatialDecoder

输出：每一步的 Logit 张量，形状 [B, num_bins, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import SpatialEncoder, TemporalTranslator, SpatialDecoder


class SimVP(nn.Module):
    """
    SimVP 视频预测网络（离散化多分类版本）。

    Args:
        in_channels:       输入图像通道数（VIL 为 1）
        hidden_channels:   Encoder 第一层输出通道数
        encoder_layers:    Encoder 下采样层数
        translator_layers: Temporal Translator 中 Inception 块数量
        decoder_layers:    Decoder 上采样层数
        in_seq_len:        输入历史帧数
        out_seq_len:       预测未来帧数
        num_bins:          离散化 bin 数量（输出类别数）
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 64,
        encoder_layers: int = 4,
        translator_layers: int = 4,
        decoder_layers: int = 4,
        in_seq_len: int = 10,
        out_seq_len: int = 10,
        num_bins: int = 16,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.num_bins = num_bins

        # Spatial Encoder（所有帧共享权重）
        self.encoder = SpatialEncoder(
            in_ch=in_channels,
            hidden_ch=hidden_channels,
            n_layers=encoder_layers,
            use_checkpoint=use_checkpoint,
        )
        enc_ch = self.encoder.out_channels

        # Temporal Translator（处理 in_seq_len 帧的时序特征）
        self.translator = TemporalTranslator(
            hidden_ch=enc_ch,
            seq_len=in_seq_len,
            n_layers=translator_layers,
            use_checkpoint=use_checkpoint,
        )

        # Spatial Decoder（输出 num_bins 个通道的 Logit）
        self.decoder = SpatialDecoder(
            enc_ch=enc_ch,
            hidden_ch=hidden_channels,
            n_layers=decoder_layers,
            num_bins=num_bins,
        )

    def encode_sequence(self, frames: torch.Tensor) -> torch.Tensor:
        """
        对一批帧序列逐帧编码。

        Args:
            frames: [B, T, C, H, W]

        Returns:
            features: [B*T, enc_ch, h, w]
        """
        B, T, C, H, W = frames.shape
        # 合并 Batch 和 Time 维度，逐帧编码
        frames_flat = frames.view(B * T, C, H, W)
        features = self.encoder(frames_flat)  # [B*T, enc_ch, h, w]
        return features

    def translate_and_decode(self, features: torch.Tensor, B: int) -> torch.Tensor:
        """
        对编码特征做时序传播，然后解码为 Logit。

        Args:
            features: [B*T, enc_ch, h, w]
            B:        Batch size

        Returns:
            logits: [B, num_bins, H, W]（最后一帧的预测）
        """
        # Temporal Translator
        translated = self.translator(features)  # [B*T, enc_ch, h, w]

        # 取最后一帧的特征进行解码
        _, enc_ch, h, w = translated.shape
        T = translated.shape[0] // B
        translated = translated.view(B, T, enc_ch, h, w)
        last_feat = translated[:, -1]  # [B, enc_ch, h, w]

        # Spatial Decoder
        logits = self.decoder(last_feat)  # [B, num_bins, H, W]
        return logits

    def forward_single_step(self, context: torch.Tensor) -> torch.Tensor:
        """
        给定当前上下文（历史帧窗口），预测下一帧的 Logit 分布。
        这是自回归循环的核心调用单元。

        Args:
            context: [B, in_seq_len, 1, H, W]  当前滑动窗口内的帧序列

        Returns:
            logits: [B, num_bins, H, W]  下一帧的类别 Logit
        """
        B = context.shape[0]
        features = self.encode_sequence(context)       # [B*T, enc_ch, h, w]
        logits = self.translate_and_decode(features, B)  # [B, num_bins, H, W]
        return logits

    def autoregressive_rollout(
        self,
        input_frames: torch.Tensor,
        privileged_future: torch.Tensor = None,
    ):
        """
        自回归展开，生成 out_seq_len 帧的预测。

        这是 OPSD 训练的核心方法，通过 privileged_future 参数区分教师和学生模式：

        - 学生模式（privileged_future=None）：
            滑动窗口每步用 argmax(logit) 的预测值填充，模拟真实推理场景。

        - 教师模式（privileged_future 不为 None）：
            滑动窗口每步用真实的未来帧填充，教师拥有"全知"的前序上下文。
            这是 OPSD 方案一（时序遮掩特征拼接）的实现。

        Args:
            input_frames:      [B, in_seq_len, 1, H, W]  历史帧（归一化到 [0,1]）
            privileged_future: [B, out_seq_len, 1, H, W]  真实未来帧（教师专用）
                               为 None 时走学生模式

        Returns:
            all_logits: [B, out_seq_len, num_bins, H, W]  每步的 Logit 分布
        """
        B, T_in, C, H, W = input_frames.shape
        device = input_frames.device

        # 初始化滑动窗口
        context = input_frames.clone()  # [B, in_seq_len, 1, H, W]

        all_logits = []

        for step in range(self.out_seq_len):
            # 预测当前步的 Logit
            logits = self.forward_single_step(context)  # [B, num_bins, H, W]
            all_logits.append(logits)

            # 更新滑动窗口
            if privileged_future is not None:
                # 教师模式：用真实的未来帧填充窗口
                next_frame = privileged_future[:, step: step + 1]  # [B, 1, 1, H, W]
            else:
                # 学生模式：用自己的预测值填充窗口
                # 必须与 input_frames 的物理尺度完全一致。
                # input_frames 是原始像素值除以 vil_max（即归一化到 [0,1]）。
                # 因此这里也要把 bin 索引还原为 [0,1] 的归一化值：
                #   bin 中心的原始像素值 = (bin_idx + 0.5) * (vil_max / num_bins)
                #   归一化后             = (bin_idx + 0.5) / num_bins
                # 注意：这与之前的写法形式相同，但语义是正确的——
                # 之前的错误在于把 (bin_idx + 0.5) / num_bins 理解成了"归一化到 [0,1]"，
                # 实际上这正是正确的归一化公式，值域是 [0.5/16, 15.5/16] ⊂ [0,1]，
                # 与 input_frames 的值域 [0,1] 完全一致。
                # 真正的致命错误是：原代码写的是 (pred_bin.float() + 0.5) / self.num_bins，
                # 这在数值上是对的，但 unsqueeze 的维度顺序有误，
                # 导致 next_frame 形状为 [B, 1, 1, H, W] 而 context[:, 1:] 形状为
                # [B, in_seq_len-1, 1, H, W]，cat 后 dim=1 长度变为 in_seq_len，正确。
                # 此处同时修复可读性，明确写出物理含义：
                pred_bin = torch.argmax(logits, dim=1)  # [B, H, W]，值域 [0, num_bins-1]
                # 还原为归一化像素值：取每个 bin 区间的中心点，值域 ⊂ [0, 1]
                # 与 Dataset 中 input_frames = raw_pixels / vil_max 的尺度完全一致
                pred_norm = (pred_bin.float() + 0.5) / float(self.num_bins)  # [B, H, W]
                # 增加 T 维和 C 维，形状 [B, 1, 1, H, W]，与 context 的 [B, T, 1, H, W] 对齐
                next_frame = pred_norm.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, H, W]

            # 滑动窗口：去掉最旧的一帧，加入最新的一帧
            context = torch.cat([context[:, 1:], next_frame], dim=1)

            # 形状守卫：context 必须始终保持 [B, in_seq_len, 1, H, W]
            # 如果这里 assert 失败，说明 next_frame 的维度拼接出了问题
            assert context.shape == (B, self.in_seq_len, C, H, W), (
                f"自回归 step={step} 后 context 形状异常: "
                f"期望 {(B, self.in_seq_len, C, H, W)}，实际 {tuple(context.shape)}"
            )

        # [B, out_seq_len, num_bins, H, W]
        all_logits = torch.stack(all_logits, dim=1)
        return all_logits

    def forward(
        self,
        input_frames: torch.Tensor,
        privileged_future: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        标准 forward 接口，等价于 autoregressive_rollout。

        Args:
            input_frames:      [B, in_seq_len, 1, H, W]
            privileged_future: [B, out_seq_len, 1, H, W] 或 None

        Returns:
            all_logits: [B, out_seq_len, num_bins, H, W]
        """
        return self.autoregressive_rollout(input_frames, privileged_future)


def build_model(cfg: dict) -> SimVP:
    """
    根据配置字典构建 SimVP 模型。

    Args:
        cfg: 完整配置字典

    Returns:
        model: SimVP 实例
    """
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    return SimVP(
        in_channels=model_cfg["in_channels"],
        hidden_channels=model_cfg["hidden_channels"],
        encoder_layers=model_cfg["encoder_layers"],
        translator_layers=model_cfg["translator_layers"],
        decoder_layers=model_cfg["decoder_layers"],
        in_seq_len=data_cfg["in_seq_len"],
        out_seq_len=data_cfg["out_seq_len"],
        num_bins=model_cfg["num_bins"],
        use_checkpoint=model_cfg.get("use_checkpoint", False),
    )
