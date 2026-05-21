"""
快速冒烟测试（Smoke Test）
==========================
用随机张量验证整个训练管线，无需真实 SEVIR 数据即可运行。

测试内容：
  1. 模型前向传播（学生模式 / OPSD 教师模式）
  2. 损失计算与反向传播（梯度非 NaN）
  3. OPSD KL 散度损失
  4. 气象评估指标（CSI / POD / FAR / HSS）
  5. logits_to_vil 转换
  6. Dataset 接口（有数据时顺带测试，无数据时跳过）
  7. AMP autocast 兼容性（有 CUDA 时测试）

用法：
  python quick_test.py                 # 全部测试
  python quick_test.py --data_root F:/zyx/dataset/sevir_data  # 额外测试真实数据加载
"""

import argparse
import sys
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def ok(msg: str):
    print(f"  [PASS] {msg}")

def fail(msg: str):
    print(f"  [FAIL] {msg}", file=sys.stderr)

def section(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print('='*55)


# ---------------------------------------------------------------------------
# 测试 1：模型前向传播
# ---------------------------------------------------------------------------

def test_model_forward():
    section("Test 1: 模型前向传播（小参数，CPU）")
    from models.simvp import SimVP

    B, T_in, T_out = 2, 4, 3
    H, W = 32, 32
    num_bins = 4
    hidden_ch = 16

    model = SimVP(
        in_channels=1,
        hidden_channels=hidden_ch,
        encoder_layers=2,
        translator_layers=1,
        decoder_layers=2,
        in_seq_len=T_in,
        out_seq_len=T_out,
        num_bins=num_bins,
        use_checkpoint=False,
    )
    model.eval()

    x = torch.rand(B, T_in, 1, H, W)

    # 学生模式
    with torch.no_grad():
        logits = model(x, privileged_future=None)

    expected = (B, T_out, num_bins, H, W)
    assert logits.shape == expected, f"学生模式输出形状 {logits.shape} != {expected}"
    assert torch.isfinite(logits).all(), "学生模式输出含 NaN/Inf"
    ok(f"学生模式输出形状正确: {tuple(logits.shape)}")

    # 教师模式（OPSD）
    future = torch.rand(B, T_out, 1, H, W)
    with torch.no_grad():
        logits_t = model(x, privileged_future=future)

    assert logits_t.shape == expected, f"教师模式输出形状 {logits_t.shape} != {expected}"
    assert torch.isfinite(logits_t).all(), "教师模式输出含 NaN/Inf"
    ok(f"教师模式输出形状正确: {tuple(logits_t.shape)}")

    # 两种模式输出应有差异（不完全相同）
    diff = (logits - logits_t).abs().mean().item()
    assert diff > 1e-6, f"学生/教师 logit 完全相同，疑似模式未生效（diff={diff:.2e}）"
    ok(f"学生/教师 logit 差异正常: mean_diff={diff:.4f}")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ok(f"模型参数量: {param_count:,} ({param_count/1e6:.2f}M)")


# ---------------------------------------------------------------------------
# 测试 2：损失计算与反向传播
# ---------------------------------------------------------------------------

def test_loss_and_backward():
    section("Test 2: CE 损失计算与反向传播")
    from models.simvp import SimVP

    B, T_in, T_out = 2, 4, 3
    H, W = 32, 32
    num_bins = 4

    model = SimVP(
        in_channels=1,
        hidden_channels=16,
        encoder_layers=2,
        translator_layers=1,
        decoder_layers=2,
        in_seq_len=T_in,
        out_seq_len=T_out,
        num_bins=num_bins,
        use_checkpoint=False,
    ).train()

    x = torch.rand(B, T_in, 1, H, W)
    target = torch.randint(0, num_bins, (B, T_out, H, W))

    logits = model(x, privileged_future=None)  # [B, T_out, num_bins, H, W]

    # 掩码加权 CE（模仿 train_baseline.py 中的写法）
    foreground_weight = 5.0
    criterion_none = nn.CrossEntropyLoss(reduction="none")
    B_, T_, C_, H_, W_ = logits.shape
    logits_flat = logits.view(B_ * T_, C_, H_, W_)
    targets_flat = target.view(B_ * T_, H_, W_)
    has_echo = (targets_flat > 0).float()
    pixel_weights = 1.0 + has_echo * (foreground_weight - 1.0)
    ce_per_pixel = criterion_none(logits_flat, targets_flat)
    loss = (ce_per_pixel * pixel_weights).sum() / (pixel_weights.sum() + 1e-8)

    assert torch.isfinite(loss), f"CE 损失为 NaN/Inf: {loss.item()}"
    ok(f"CE 损失值正常: {loss.item():.4f}")

    loss.backward()

    grad_norms = []
    nan_params = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            gn = p.grad.norm().item()
            grad_norms.append(gn)
            if not torch.isfinite(p.grad).all():
                nan_params.append(name)

    assert len(nan_params) == 0, f"以下参数梯度含 NaN: {nan_params}"
    ok(f"反向传播成功，{len(grad_norms)} 个参数有梯度，最大梯度范数: {max(grad_norms):.4f}")


# ---------------------------------------------------------------------------
# 测试 3：OPSD KL 散度损失
# ---------------------------------------------------------------------------

def test_kl_loss():
    section("Test 3: OPSD KL 散度损失")
    from train_opsd import kl_divergence_loss, build_pixel_weights

    B, num_bins, H, W = 4, 8, 16, 16
    student_logits = torch.randn(B, num_bins, H, W)
    teacher_logits = torch.randn(B, num_bins, H, W)
    targets = torch.randint(0, num_bins, (B, H, W))

    # 无权重
    loss_unweighted = kl_divergence_loss(student_logits, teacher_logits, temperature=2.0)
    assert torch.isfinite(loss_unweighted), f"KL 损失（无权重）为 NaN: {loss_unweighted}"
    ok(f"KL 损失（无权重）: {loss_unweighted.item():.4f}")

    # 有像素权重
    weights = build_pixel_weights(targets, num_bins, foreground_weight=5.0)
    loss_weighted = kl_divergence_loss(student_logits, teacher_logits, temperature=2.0,
                                       pixel_weights=weights)
    assert torch.isfinite(loss_weighted), f"KL 损失（有权重）为 NaN: {loss_weighted}"
    ok(f"KL 损失（有权重）: {loss_weighted.item():.4f}")

    # 温度变化时损失应有差异
    loss_t1 = kl_divergence_loss(student_logits, teacher_logits, temperature=1.0)
    loss_t4 = kl_divergence_loss(student_logits, teacher_logits, temperature=4.0)
    assert abs(loss_t1.item() - loss_t4.item()) > 1e-6, "不同温度下 KL 损失完全相同，可能有 bug"
    ok(f"温度敏感性正常: T=1.0 -> {loss_t1.item():.4f}, T=4.0 -> {loss_t4.item():.4f}")


# ---------------------------------------------------------------------------
# 测试 4：评估指标
# ---------------------------------------------------------------------------

def test_metrics():
    section("Test 4: 气象评估指标")
    from utils import compute_metrics_at_threshold, compute_all_metrics, compute_mse, compute_mae

    N, H, W = 10, 64, 64
    rng = np.random.default_rng(42)

    # 完美预测
    true = rng.uniform(0, 255, (N, H, W)).astype(np.float32)
    pred_perfect = true.copy()
    m = compute_metrics_at_threshold(pred_perfect, true, threshold=74)
    assert abs(m["CSI"] - 1.0) < 1e-6, f"完美预测 CSI 应为 1.0，实际 {m['CSI']:.4f}"
    assert abs(m["POD"] - 1.0) < 1e-6, f"完美预测 POD 应为 1.0，实际 {m['POD']:.4f}"
    assert abs(m["FAR"] - 0.0) < 1e-6, f"完美预测 FAR 应为 0.0，实际 {m['FAR']:.4f}"
    ok("完美预测时 CSI=1.0, POD=1.0, FAR=0.0")

    # 全零预测（全部漏报）
    pred_zeros = np.zeros_like(true)
    m_zero = compute_metrics_at_threshold(pred_zeros, true, threshold=74)
    assert m_zero["CSI"] == 0.0, "全零预测 CSI 应为 0"
    assert m_zero["POD"] == 0.0, "全零预测 POD 应为 0"
    ok("全零预测时 CSI=0, POD=0")

    # 多阈值批量计算
    thresholds = [16, 74, 133]
    all_m = compute_all_metrics(true, true, thresholds)
    assert set(all_m.keys()) == set(thresholds), "compute_all_metrics 阈值键不匹配"
    ok(f"多阈值批量计算正常: {thresholds}")

    # MSE / MAE
    mse = compute_mse(true, pred_perfect)
    mae = compute_mae(true, pred_perfect)
    assert mse == 0.0 and mae == 0.0, "完美预测 MSE/MAE 应为 0"
    ok("完美预测 MSE=0, MAE=0")

    # 随机预测的合理性检查
    pred_rand = rng.uniform(0, 255, (N, H, W)).astype(np.float32)
    mse_rand = compute_mse(pred_rand, true)
    assert mse_rand > 0, "随机预测 MSE 应大于 0"
    ok(f"随机预测 MSE={mse_rand:.2f} (>0)")


# ---------------------------------------------------------------------------
# 测试 5：logits_to_vil 转换
# ---------------------------------------------------------------------------

def test_logits_to_vil():
    section("Test 5: logits_to_vil 转换")
    from utils import logits_to_vil

    B, T, num_bins, H, W = 2, 3, 16, 32, 32
    vil_max = 255.0

    # 构造一个确定性 logit：第 k 个 bin 对应位置的 logit 最大
    logits = torch.zeros(B, T, num_bins, H, W)
    logits[:, :, 7, :, :] = 100.0  # 强制 argmax = 7

    vil = logits_to_vil(logits, num_bins, vil_max)
    expected_val = (7 + 0.5) * (vil_max / num_bins)  # bin 7 中心值
    assert vil.shape == (B, T, H, W), f"输出形状错误: {vil.shape}"
    assert abs(vil.mean() - expected_val) < 1e-3, (
        f"VIL 值错误: 期望 {expected_val:.2f}, 实际 {vil.mean():.2f}"
    )
    ok(f"logits_to_vil 正确: bin=7 -> VIL={vil.mean():.2f} (期望 {expected_val:.2f})")

    # 值域检查
    assert vil.min() >= 0.0 and vil.max() <= vil_max, (
        f"VIL 值域超出 [0, {vil_max}]: [{vil.min():.2f}, {vil.max():.2f}]"
    )
    ok(f"VIL 值域 [{vil.min():.2f}, {vil.max():.2f}] 在 [0, {vil_max}] 内")


# ---------------------------------------------------------------------------
# 测试 6：Dataset 离散化工具函数
# ---------------------------------------------------------------------------

def test_discretization():
    section("Test 6: VIL 离散化与反离散化")
    from data.sevir_dataset import vil_to_bins, bins_to_vil

    num_bins = 16
    vil_max = 255.0

    # 边界值测试
    vil = np.array([0.0, 127.5, 255.0], dtype=np.float32)
    bins = vil_to_bins(vil, num_bins, vil_max)
    assert bins.min() >= 0 and bins.max() < num_bins, (
        f"bin 索引超出 [0, {num_bins-1}]: [{bins.min()}, {bins.max()}]"
    )
    ok(f"边界值离散化: {vil} -> bins {bins}")

    # 反离散化后应近似还原
    vil_back = bins_to_vil(bins, num_bins, vil_max)
    max_err = np.abs(vil_back - vil).max()
    # 最大误差不超过一个 bin 宽度的一半
    assert max_err <= vil_max / num_bins, f"反离散化误差过大: {max_err:.2f}"
    ok(f"反离散化误差 <= {vil_max/num_bins:.1f}: max_err={max_err:.2f}")

    # 批量随机测试
    vil_rand = np.random.default_rng(0).uniform(0, vil_max, (100, 64, 64)).astype(np.float32)
    bins_rand = vil_to_bins(vil_rand, num_bins, vil_max)
    assert bins_rand.dtype == np.int64, "bin 索引 dtype 应为 int64"
    assert bins_rand.min() >= 0 and bins_rand.max() < num_bins, "批量离散化越界"
    ok(f"批量离散化 shape={bins_rand.shape}, 值域 [{bins_rand.min()}, {bins_rand.max()}]")


# ---------------------------------------------------------------------------
# 测试 7：AMP autocast 兼容性（有 CUDA 时）
# ---------------------------------------------------------------------------

def test_amp_cuda():
    section("Test 7: AMP autocast 兼容性")

    if not torch.cuda.is_available():
        print("  [SKIP] 当前环境无 CUDA，跳过 AMP 测试")
        return

    from models.simvp import SimVP
    from torch.cuda.amp import GradScaler, autocast

    device = torch.device("cuda:0")
    B, T_in, T_out = 2, 4, 3
    H, W = 64, 64
    num_bins = 8

    model = SimVP(
        in_channels=1,
        hidden_channels=32,
        encoder_layers=2,
        translator_layers=1,
        decoder_layers=2,
        in_seq_len=T_in,
        out_seq_len=T_out,
        num_bins=num_bins,
        use_checkpoint=True,  # 测试 gradient checkpointing
    ).to(device).train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    x = torch.rand(B, T_in, 1, H, W, device=device)
    target = torch.randint(0, num_bins, (B, T_out, H, W), device=device)

    optimizer.zero_grad()
    with autocast():
        logits = model(x)
        logits_flat = logits.view(B * T_out, num_bins, H, W)
        targets_flat = target.view(B * T_out, H, W)
        loss = nn.CrossEntropyLoss()(logits_flat, targets_flat)

    assert torch.isfinite(loss), f"AMP 下 CE 损失为 NaN: {loss.item()}"
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optimizer)
    scaler.update()

    ok(f"AMP + GradScaler + gradient_checkpoint 训练步骤正常, loss={loss.item():.4f}")
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# 测试 8：真实数据加载（可选，有 data_root 时才运行）
# ---------------------------------------------------------------------------

def test_real_data(data_root: str):
    section(f"Test 8: 真实 SEVIR 数据加载 ({data_root})")
    import os
    if not os.path.isdir(data_root):
        print(f"  [SKIP] data_root 不存在: {data_root}")
        return

    from data.sevir_dataset import SEVIRVILDataset

    ds = SEVIRVILDataset(
        data_root=data_root,
        in_seq_len=5,
        out_seq_len=5,
        num_bins=16,
        split="train",
        val_ratio=0.1,
        seed=42,
    )
    ok(f"数据集加载成功，样本数: {len(ds)}")

    # 取第一个样本
    inp, tgt, fut = ds[0]
    assert inp.shape == (5, 1, 384, 384), f"input_frames 形状错误: {inp.shape}"
    assert tgt.shape == (5, 384, 384), f"target_bins 形状错误: {tgt.shape}"
    assert fut.shape == (5, 1, 384, 384), f"future_frames 形状错误: {fut.shape}"
    ok(f"样本形状正确: input={tuple(inp.shape)}, target={tuple(tgt.shape)}")

    # 值域检查
    assert inp.min() >= 0.0 and inp.max() <= 1.0, f"input 值域超出 [0,1]: [{inp.min():.4f},{inp.max():.4f}]"
    assert tgt.min() >= 0 and tgt.max() < 16, f"target_bins 越界: [{tgt.min()},{tgt.max()}]"
    ok(f"值域正常: input 在 [0,1], target_bins 在 [0,15]")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SimVP 快速冒烟测试")
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="SEVIR 数据目录路径（可选，如 F:/zyx/dataset/sevir_data）"
    )
    args = parser.parse_args()

    print("\n" + "="*55)
    print("  SimVP SEVIR VIL 冒烟测试")
    print("="*55)
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    tests = [
        ("模型前向传播",   test_model_forward),
        ("损失与反向传播", test_loss_and_backward),
        ("KL 散度损失",   test_kl_loss),
        ("评估指标",       test_metrics),
        ("logits_to_vil", test_logits_to_vil),
        ("离散化工具",     test_discretization),
        ("AMP 兼容性",    test_amp_cuda),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            fail(f"测试 [{name}] 异常: {e}")
            traceback.print_exc()
            failed += 1

    # 可选的真实数据测试
    if args.data_root:
        try:
            test_real_data(args.data_root)
            passed += 1
        except Exception as e:
            fail(f"测试 [真实数据] 异常: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*55}")
    print(f"  结果: {passed} 通过, {failed} 失败")
    print('='*55 + "\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
