# SimVP + OPSD 强对流天气临近预报

基于 SEVIR VIL 数据集，将 **OPSD（On-Policy Self-Distillation，在策略自蒸馏）** 引入雷达回波外推任务的实验代码。

## 核心思想

传统自回归外推模型在预测后期（60min+）会因误差累积导致画面严重模糊。OPSD 通过引入"全知教师"解决这一问题：

- **学生**：只看历史帧，自回归生成未来帧（模拟真实推理）
- **教师**：同一个模型，但滑动窗口每步用真实未来帧填充（特权信息）
- **蒸馏**：用教师在每步的 Logit 分布，通过 KL 散度纠正学生的预测分布

教师分支只做前向传播（`torch.no_grad()`），显存几乎不增加，16G 单卡可跑。

## 项目结构

```
simvp-test/
├── configs/
│   └── default.yaml          # 所有超参数
├── data/
│   └── sevir_dataset.py      # SEVIR VIL 数据加载 + 离散化
├── models/
│   ├── modules.py            # Encoder / Translator / Decoder 子模块
│   └── simvp.py              # SimVP 主网络（含 OPSD 接口）
├── train_baseline.py         # 第一阶段：交叉熵监督训练
├── train_opsd.py             # 第二阶段：OPSD 自蒸馏训练
├── evaluate.py               # 评估脚本（CSI/POD/FAR/HSS + 可视化）
├── utils.py                  # 工具函数
└── requirements.txt
```

## 环境配置

```bash
pip install -r requirements.txt
```

## 数据准备

将 SEVIR 数据放置到以下目录结构：

```
C:/data/sevir/          # 或任意路径，在 configs/default.yaml 中修改 data.data_root
    SEVIR_CATALOG.csv
    data/
        vil/
            *.h5
```

然后修改 `configs/default.yaml` 中的 `data.data_root` 为实际路径。

## 快速开始

### 第一步：训练 Baseline

```bash
python train_baseline.py --config configs/default.yaml
```

训练完成后，最佳模型保存在 `checkpoints/baseline/best.pth`。

### 第二步：训练 OPSD

```bash
python train_opsd.py --config configs/default.yaml
```

脚本会自动从 `checkpoints/baseline/best.pth` 热启动（在配置文件中指定）。

### 第三步：评估与对比

```bash
# 评估单个模型
python evaluate.py --config configs/default.yaml \
    --ckpt checkpoints/baseline/best.pth --tag baseline

# 对比 Baseline 和 OPSD
python evaluate.py --config configs/default.yaml \
    --ckpt checkpoints/baseline/best.pth checkpoints/opsd/best.pth \
    --tag baseline opsd
```

评估结果（CSV + 对比曲线图）保存在 `eval_results/` 目录。

## 关键配置说明

编辑 `configs/default.yaml` 调整实验参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `data.data_root` | `C:/data/sevir` | SEVIR 数据根目录 |
| `data.in_seq_len` | 10 | 输入历史帧数（10帧 = 50min） |
| `data.out_seq_len` | 10 | 预测未来帧数（10帧 = 50min） |
| `data.num_bins` | 16 | VIL 离散化 bin 数量 |
| `model.hidden_channels` | 64 | 网络隐藏层通道数 |
| `training.baseline_batch_size` | 4 | Baseline 训练 batch size |
| `training.opsd_kl_weight` | 1.0 | KL 散度损失权重 |
| `training.opsd_ce_weight` | 0.5 | 交叉熵辅助损失权重 |

如果显存不足，可以将 `model.hidden_channels` 从 64 降到 32，或将 batch size 降到 2。

## 评估指标

- **CSI**（Critical Success Index）：综合衡量命中率和虚警率，越高越好
- **POD**（Probability of Detection）：命中率，越高越好
- **FAR**（False Alarm Ratio）：虚警率，越低越好
- **HSS**（Heidke Skill Score）：相对于随机预报的技巧分，越高越好

评估阈值对应的物理含义（VIL 像素值）：

| 阈值 | 物理含义 |
|------|----------|
| 16 | 轻微回波 |
| 74 | 中等对流 |
| 133 | 强对流 |
| 160 | 极强对流 |
| 181 | 冰雹风险 |
| 219 | 极端强对流 |

## 预期实验结果

- **Baseline**：前 30min 预测质量较好，60min 后画面开始模糊，CSI 明显下降
- **OPSD**：后期（40~50min）CSI 下降趋势明显减缓，强对流单体边缘更锐利

## 参考

- SEVIR 数据集：[MIT Lincoln Laboratory](https://github.com/MIT-AI-Accelerator/neurips-2020-sevir)
- SimVP：[SimVP: Simpler yet Better Video Prediction](https://arxiv.org/abs/2206.05099)
- OPSD：On-Policy Self-Distillation for LLM reasoning（迁移自 LLM 领域）
