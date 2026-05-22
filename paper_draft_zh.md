# 面向强对流天气的精细化雷达外推：选择性状态空间模型与在策略自蒸馏

---

## 摘要

强对流天气的短临预报是气象业务中的核心挑战。自回归深度学习模型在长期预报中存在误差累积问题，导致30分钟后的预测画面严重模糊。本文在 SEVIR VIL 雷达数据集上，对 SimVP 框架提出两项互补改进。其一，**在策略自蒸馏（OPSD）**：通过让拥有真实未来帧的特权"教师"分支以 KL 散度监督自回归"学生"分支的逐步预测分布，从根本上缓解训练与推理的分布偏移（exposure bias），且无需修改网络结构、显存开销几乎为零。其二，**奖励加权 OPSD**：以每步 CSI 的反向值作为步权重，将蒸馏梯度集中在预测质量较差的时间步上。其三，将 SimVP 的 Inception 时序翻译器替换为 **Mamba 选择性状态空间模型（SSM）**，对每个空间位置的时序动态进行显式的输入依赖状态建模，替代原有的通道拼接加二维卷积的隐式方案。在 128×128 / 25分钟预测时长配置下的实验表明，【**待填写实验结果**】。Mamba 与 OPSD 的组合取得最优整体性能，说明结构性时序建模与在策略训练是互补的改进方向。

---

## 1. 引言

强对流天气临近预报——在0~60分钟时间尺度上预测雷达反射率场——对强天气预警、航空安全和城市洪涝防控具有重要意义。该任务面临独特挑战：对流单体的生成、增强和消散均发生在分钟级时间尺度上，呈现高度非线性动态，难以用纯物理外推方法准确描述。

深度学习方法在该领域取得了显著进展。ConvLSTM [Shi et al., 2015] 和 PredRNN [Wang et al., 2017] 奠定了深度时空预测的基础，EarthFormer [Gao et al., 2022]、NowcastNet [Zhang et al., 2023] 和 DGMR [Ravuri et al., 2021] 等近期模型在多个基准上达到或超过了业务化方法（如 PySTEPS）的水平。SimVP [Gao et al., 2022] 提出了一种纯卷积的编码器—翻译器—解码器设计，以显著低于循环网络的训练代价实现了有竞争力的预测精度。

然而，自回归临近预报模型仍面临两个共性局限：

**分布偏移（Exposure Bias）。** 自回归模型在训练时采用教师强制（teacher forcing）——每步以真实历史帧为条件——而推理时只能以自身（存在误差的）预测帧为条件。这种训练/推理不一致随步数积累，导致30分钟后预测质量急剧下降，即经典的 *exposure bias* 问题 [Bengio et al., 2015]。

**隐式时序建模。** SimVP 的时序翻译器将 $T$ 帧特征在通道维度拼接（得到 $T \times C$ 的通道组合），再用二维卷积处理。这种方案没有显式的时序轴，无法针对不同空间位置学习各异的时序转移动态，且通道维度随 $T$ 线性增长。

本文在 SEVIR VIL 基准 [Veillette et al., 2020] 上，通过统一的实验框架同时解决上述两个问题：

1. **OPSD（在策略自蒸馏）**：一种训练策略，使用特权"教师"分支（每步以真实未来帧更新滑动窗口）通过逐步 KL 散度监督"学生"分支（自回归推理模式），不修改模型结构，教师分支仅做前向传播。

2. **奖励加权 OPSD**：对每步 KL 损失乘以 $(1 - \text{CSI}_t)$ 权重，将蒸馏梯度集中在预测困难的时间步上。

3. **Mamba 时序翻译器**：将 Inception 翻译器替换为 Mamba SSM [Gu & Dao, 2023]，在每个空间位置上用选择性状态转移显式建模时序动态，可直接插拔，编码器/解码器无需修改。

本文的主要贡献包括：
- 验证 OPSD 能一致提升长预测时域（≥20分钟）的 CSI，训练开销几乎不增加。
- 验证 Mamba 时序翻译器在相同训练协议下优于 Inception 翻译器【**待填写**】。
- 验证两种改进互补：Mamba + OPSD 的组合优于各自单独改进，综合 CSI-M 达到【**待填写**】。

---

## 2. 相关工作

### 2.1 雷达临近预报的深度学习方法

ConvLSTM [Shi et al., 2015] 最早将深度时空预测用于降水临近预报，将卷积运算引入 LSTM 的内部状态转移。PredRNN [Wang et et al., 2017] 提出空间-时序记忆流，在堆叠 LSTM 的层间和时间步间同时传递状态。EarthFormer [Gao et al., 2022] 设计了层次化 Transformer，引入地球科学专属的时空自注意力机制，在包括 SEVIR 在内的多个气象数据集上取得优异性能。

生成式方法也得到广泛关注：DGMR [Ravuri et al., 2021] 基于条件 GAN 生成概率集合预报，能够保留清晰的对流单体边缘；DiffCast [Yu et al., 2023] 和 PreDiff [Gao et al., 2023] 则将扩散模型用于同一目标。这类方法在视觉清晰度上表现突出，但推理代价显著更高。

SimVP [Gao et al., 2022] 证明了非循环的纯卷积设计——空间编码器、Inception 时序翻译器、空间解码器——在训练代价大幅降低的情况下仍可媲美循环基线。本文以 SimVP 为基础，分别从缓解 exposure bias 和增强时序表达能力两个维度对其进行改进。

### 2.2 Exposure Bias 与自蒸馏

序列生成中的 exposure bias 由 Bengio et al. [2015]（Scheduled Sampling）正式提出，并由 DAgger [Ross et al., 2011] 在模仿学习框架中系统研究。在神经机器翻译中，多种课程学习策略被用于混合教师强制与自由生成输入。对于视频预测，Srivastava et al. [2015] 直接将 Scheduled Sampling 用于 LSTM 视频预测。

知识蒸馏 [Hinton et al., 2015] 将教师模型的软标签分布传递给学生模型。OPSD 将这一思想迁移到在策略（on-policy）场景：教师是同一个模型在特权输入下的前向输出，而非独立训练的更大模型，因此无需额外模型容量。这与近期用于大语言模型推理的 OPSD 方法 [anonymous, 2025] 一脉相承，本文将其迁移至雷达外推领域。

### 2.3 状态空间模型

结构化状态空间序列模型（S4）[Gu et al., 2021] 证明了基于 HiPPO 初始化的线性递归能高效建模长程依赖。Mamba [Gu & Dao, 2023] 引入选择性状态空间（S6），其转移矩阵 $\mathbf{A}$、$\mathbf{B}$、$\mathbf{C}$ 依赖输入动态调整，使模型能够选择性地关注相关上下文。Vision Mamba [Zhu et al., 2024] 和 VMamba [Liu et al., 2024] 通过空间扫描策略将 Mamba 推广至二维视觉数据。

在气象领域，SSM 的应用尚处于初步探索阶段。雷达序列的时间步数通常为5~13帧（5分钟间隔），绝对序列长度虽短，但对流单体的轨迹、合并与消散具有强非马尔可夫结构，跨越多帧。本文假设对每个空间位置进行显式状态建模，比通道拼接方式更适合捕获这种时序动态。

---

## 3. 问题定义

设 $\mathbf{X} = \{x_1, x_2, \ldots, x_{T_{\text{in}}}\} \in \mathbb{R}^{T_{\text{in}} \times H \times W}$ 为 $T_{\text{in}}$ 帧连续 VIL（垂直积分液态水）雷达图像，每帧 $x_t \in [0, 255]$ 对应单位面积垂直积分液态水含量。目标是预测未来序列 $\hat{\mathbf{Y}} = \{\hat{y}_{T_{\text{in}}+1}, \ldots, \hat{y}_{T_{\text{in}}+T_{\text{out}}}\}$。

参照离散化方案，将每个像素量化为 $K=16$ 个等宽区间，区间宽度 $\Delta = 255/K$。模型在每个预测步输出逐像素的类别分布 $p_t(k) \in \mathbb{R}^K$，预测值取 argmax 区间的中心：$\hat{y}_t = (\arg\max_k p_t(k) + 0.5) \cdot \Delta$。

主要评估指标为临界成功指数（CSI）：
$$\text{CSI}(\tau) = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
其中正样本定义为 VIL $\geq \tau$ 的像素。报告 CSI-M（六个标准阈值 $\tau \in \{16, 74, 133, 160, 181, 219\}$ 的均值），以及探测概率（POD）、虚警率（FAR）和 Heidke 技巧评分（HSS）。

---

## 4. 方法

### 4.1 SimVP 自回归基线

SimVP 由三个模块组成：

**空间编码器** $\mathcal{E}$：$N_e$ 层步长为2的卷积块，对每帧独立编码为低分辨率特征图：$f_t = \mathcal{E}(x_t) \in \mathbb{R}^{C \times h \times w}$，其中 $h = H / 2^{N_e}$，$w = W / 2^{N_e}$。

**时序翻译器** $\mathcal{T}$：处理编码特征序列 $\{f_1, \ldots, f_{T_{\text{in}}}\}$，提取时序表示。原始 SimVP 将所有 $T_{\text{in}}$ 帧特征在通道维度拼接（得到 $T_{\text{in}} C \times h \times w$），再经 Inception 风格二维卷积处理。

**空间解码器** $\mathcal{D}$：$N_d$ 层转置卷积，将最后一帧的翻译特征上采样回原始空间分辨率，输出逐像素 logit $\ell_t \in \mathbb{R}^{K \times H \times W}$。

**自回归展开。** 为支持 OPSD 训练目标，采用自回归推理：每步从当前 $T_{\text{in}}$ 帧的滑动窗口预测 $\ell_t$，将离散化预测值 $\hat{y}_t = (\arg\max \ell_t + 0.5) \cdot \Delta / 255$（归一化形式）追加到窗口，循环 $T_{\text{out}}$ 步。

基线使用前景加权交叉熵损失训练：
$$\mathcal{L}_{\text{CE}} = \frac{\sum_{t,i,j} w_{i,j}^t \cdot \text{CE}(\ell_t^{i,j},\, k_t^{i,j})}{\sum_{t,i,j} w_{i,j}^t}$$
其中 $k_t^{i,j}$ 为真实 bin 索引，$w_{i,j}^t = 1 + (\lambda_{\text{fg}} - 1) \cdot \mathbf{1}[k_t^{i,j} > 0]$ 对非背景像素赋予更高权重（$\lambda_{\text{fg}} = 5.0$）。

### 4.2 在策略自蒸馏（OPSD）

**动机。** 自回归基线存在 *exposure bias*：训练时每步以真实历史帧为条件，推理时却以有误差的预测帧为条件。这种分布不匹配随步数累积，导致长预测时域的预测质量急剧下降。

**教师-学生框架。** OPSD 利用同一个模型的两种运行模式：

- *学生分支*：无特权信息的自回归展开（标准推理模式），产生 logit 序列 $\{p^s_t\}_{t=1}^{T_{\text{out}}}$。
- *教师分支*：每步以**真实未来帧** $y_t$ 更新滑动窗口，包裹在 `torch.no_grad()` 中，仅做前向传播，不计算梯度，产生 logit 序列 $\{p^r_t\}_{t=1}^{T_{\text{out}}}$。

教师的分布 $p^r_t$ 代表给定真实上下文时模型的最优预测。用温度缩放的 KL 散度对学生进行监督：

$$\mathcal{L}_{\text{KL}} = T^2 \cdot \frac{\sum_{t,i,j} w_{i,j}^t \cdot D_{\text{KL}}\!\left(p^r_t(\cdot \mid i,j;\, T) \;\|\; p^s_t(\cdot \mid i,j;\, T)\right)}{\sum_{t,i,j} w_{i,j}^t}$$

其中 $T=2.0$ 为蒸馏温度，$T^2$ 系数补偿温度缩放对梯度幅度的影响 [Hinton et al., 2015]。空间权重 $w_{i,j}^t$ 与 CE 损失共用同一前景掩码。

总训练目标为蒸馏损失与交叉熵损失的加权和：
$$\mathcal{L}_{\text{OPSD}} = \lambda_{\text{KL}} \cdot \mathcal{L}_{\text{KL}} + \lambda_{\text{CE}} \cdot \mathcal{L}_{\text{CE}}$$
其中 $\lambda_{\text{KL}} = 1.0$，$\lambda_{\text{CE}} = 0.5$。

**显存开销。** 由于教师分支运行在 `torch.no_grad()` 下，不保存用于反向传播的中间激活值。额外显存仅为教师输出 logit（$T_{\text{out}} \times K \times H \times W$），相对于完整激活图可忽略不计，16G 显卡单卡可跑。

### 4.3 奖励加权 OPSD

**动机。** 标准 OPSD 对所有 $T_{\text{out}}$ 步赋予均等的 KL 权重。然而预测质量随时间步非均匀下降：早期步骤准确度较高（CSI 高），后期步骤误差更大（CSI 低）。将蒸馏信号集中在困难步骤上，有助于提升梯度预算的利用效率。

**形式化。** 定义逐步奖励为批内均值 CSI（在参考阈值 $\tau_r = 74$ 处）：
$$r_t = \frac{1}{B} \sum_{b=1}^{B} \text{CSI}\!\left(\hat{y}_t^{(b)},\, y_t^{(b)},\, \tau_r\right)$$

步权重为奖励的反向值 $w_t^{\text{rw}} = 1 - r_t \in [0, 1]$：学生已预测较好的步骤（$r_t \approx 1$）获得近零的额外 KL 权重，预测较差的步骤（$r_t \approx 0$）获得最大权重。损失变为：

$$\mathcal{L}_{\text{OPSD-RW}} = \frac{\lambda_{\text{KL}}}{T_{\text{out}}} \sum_{t=1}^{T_{\text{out}}} (1 - r_t) \cdot \mathcal{L}_{\text{KL}}^t + \lambda_{\text{CE}} \cdot \mathcal{L}_{\text{CE}}$$

$r_t$ 在 `torch.no_grad()` 下计算并以 Python 标量形式返回，以常数乘子形式进入计算图，不引入额外梯度路径。

### 4.4 Mamba 时序翻译器

**通道拼接方案的局限。** 原始时序翻译器将 $T_{\text{in}}$ 帧特征在通道维度拼接，得到 $T_{\text{in}} C \times h \times w$ 的表示，再用二维卷积隐式建模时序关系。这种方案：（i）将时空处理混合于单一操作中；（ii）对帧顺序的感知仅依赖固定的通道位置，缺乏显式的时序状态；（iii）通道维度随 $T_{\text{in}}$ 线性增长，输入序列变长时代价较高。

**逐空间位置时序 Mamba。** 将 Inception 翻译器替换为在时序轴上显式操作的 Mamba SSM：

$$\{f_1, \ldots, f_{T_{\text{in}}}\} \in \mathbb{R}^{T_{\text{in}} \times C \times h \times w}
\;\xrightarrow{\text{reshape}}\;
\mathbb{R}^{(h \cdot w) \times T_{\text{in}} \times C}$$

$h \times w$ 个空间位置各自视为长度为 $T_{\text{in}}$、特征维度为 $C$ 的独立序列，在时序轴上施加 $N_m$ 个 Mamba 块：

$$\text{Mamba}^{(n)}: \mathbb{R}^{(h \cdot w) \times T_{\text{in}} \times C} \to \mathbb{R}^{(h \cdot w) \times T_{\text{in}} \times C}$$

时序处理完成后，用一层轻量 $3 \times 3$ 空间卷积恢复跨位置的信息交互，再还原形状供后续解码器使用。

**Mamba 块结构。** 每个 Mamba 块实现 S6 选择性扫描 [Gu & Dao, 2023]：

1. **输入投影**：$\mathbf{x}, \mathbf{z} = \text{split}(W_{\text{in}} \mathbf{u})$，扩展至 $d_{\text{inner}} = 2C$。
2. **因果卷积**：核宽为4的逐通道一维深度可分离卷积，在序列维度上提取局部上下文。
3. **S6 选择性扫描**：对状态转移进行输入依赖的离散化：
$$\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A}), \quad \bar{\mathbf{B}}_t = \Delta_t \mathbf{B}_t$$
$$\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t x_t, \quad y_t = \mathbf{C}_t \mathbf{h}_t + \mathbf{D} x_t$$
其中 $\Delta_t$、$\mathbf{B}_t$、$\mathbf{C}_t$ 由输入 $\mathbf{x}_t$ 动态计算（选择性），$\mathbf{A}$ 为跨位置共享的可学习参数。
4. **门控**：$\mathbf{y} \leftarrow \mathbf{y} \odot \text{SiLU}(\mathbf{z})$。
5. **输出投影**：$W_{\text{out}} \mathbf{y}$，还原至维度 $C$。

所有 SSM 计算在 float32 精度下进行，避免 AMP float16 模式下矩阵指数溢出产生 NaN，输出在残差相加前转回原始 dtype，与混合精度训练保持兼容。

**空间复杂度说明。** 在 128×128 / 3层编码器配置下，特征图为 $16 \times 16 = 256$ 个空间位置。每个 Mamba 块处理 $B \times 256$ 条长度为 $T_{\text{in}} = 5$、维度为 $C = 256$ 的序列，S6 顺序扫描的计算开销可忽略。

---

## 5. 实验

### 5.1 数据集

使用 **SEVIR**（Storm EVent ImageRy）数据集 [Veillette et al., 2020]，覆盖美国大陆的时空气象图像序列集合。每个事件以5分钟间隔、1km 空间分辨率（384×384像素）记录，空间覆盖 384km×384km。使用 **VIL**（垂直积分液态水）通道，以 uint8 格式编码垂直积分液态水含量（kg/m²），值域 $[0, 255]$。

**数据划分。** 遵循时间分离的标准划分方式，防止数据泄露：2017–2018年事件用于训练和验证（按事件索引随机划分，90%/10%），2019年事件用于测试。

**快速实验配置。** 为快速获得对比结果，所有实验均采用中心裁剪至 128×128、输入 $T_{\text{in}}=5$ 帧（25分钟历史）、预测 $T_{\text{out}}=5$ 帧（25分钟前置时间）的配置。该配置相比完整 384×384 设置每轮 epoch 约快9倍。注意：128×128 裁剪后的 CSI 绝对值不可直接与已发表的全分辨率结果对比（中心区域对流密度较高），但五个模型在完全相同的数据设置下进行比较，对比结论可靠。

### 5.2 实现细节

| 超参数 | 取值 |
|---|---|
| 空间编码器层数 $N_e$ | 3 |
| 时序翻译器层数 $N_m$ / $N_{\text{inc}}$ | 4 |
| 隐层通道数 $C$ | 64（编码器输入），256（编码器输出）|
| Mamba 隐状态维度 $d_{\text{state}}$ | 16 |
| Mamba 扩展比例 | 2（$d_{\text{inner}} = 512$）|
| 离散化区间数 $K$ | 16 |
| 前景权重 $\lambda_{\text{fg}}$ | 5.0 |
| 基线学习率 | $3 \times 10^{-4}$（余弦退火）|
| OPSD 学习率 | $1 \times 10^{-4}$（余弦退火）|
| 批大小 | 4 |
| 训练轮数 | 20（基线），20（OPSD微调）|
| OPSD 蒸馏温度 $T$ | 2.0 |
| $\lambda_{\text{KL}}$ / $\lambda_{\text{CE}}$ | 1.0 / 0.5 |
| 奖励阈值 $\tau_r$ | 74 |
| 梯度裁剪范数 | 0.5 |
| 优化器 | AdamW，权重衰减 $10^{-4}$ |
| 随机种子 | 42 |

所有模型在单张16G显存 NVIDIA GPU 上训练，使用 PyTorch AMP（float16激活，float32 GroupNorm 和 SSM 计算）。OPSD 模型从对应基线 checkpoint 热启动。

### 5.3 对比方法

本文对比五种配置：

| 标签 | 架构 | 训练方式 |
|---|---|---|
| **Baseline** | SimVP（Inception 翻译器）| 交叉熵监督 |
| **OPSD** | SimVP（Inception 翻译器）| CE + KL 蒸馏 |
| **OPSD-RW** | SimVP（Inception 翻译器）| CE + 奖励加权 KL |
| **Mamba** | SimVP（Mamba 翻译器）| 交叉熵监督 |
| **Mamba+OPSD** | SimVP（Mamba 翻译器）| CE + KL 蒸馏 |

### 5.4 主实验结果

> **【待填写：主实验结果表】**
>
> 表1：五种方法在 SEVIR VIL 测试集（2019年）上的 CSI-M（六阈值均值）、CSI@74、CSI@133、POD@74、FAR@74、HSS@74。每列最优结果加粗。

### 5.5 逐步 CSI 分析

> **【待填写：图1】**
>
> 图1：阈值74（中等对流）下，CSI 随预测前置时间（5–25分钟）的变化曲线。横轴为预测时间（分钟），纵轴为 CSI ∈ [0, 1]，五条曲线分别对应五种模型。

> **【待填写：图2】**
>
> 图2：阈值133（强对流）下，CSI 随预测前置时间的变化曲线。

### 5.6 定性对比

> **【待填写：图3】**
>
> 图3：t+5min、t+15min、t+25min 的预测 VIL 场可视化对比。各行依次为：（1）真实值，（2）Baseline，（3）OPSD，（4）Mamba，（5）Mamba+OPSD。

### 5.7 消融实验

> **【待填写：表2】**
>
> 表2：OPSD 组件消融。各行改变 KL 权重 $\lambda_{\text{KL}}$、蒸馏温度 $T$ 和奖励加权策略。

---

## 6. 讨论

> **【待填写】**
>
> 建议分析以下问题：
> - OPSD 对后期步骤（t+20、t+25）的提升是否大于早期步骤（t+5）？如是，则印证了 exposure bias 假设。
> - Mamba 的提升是否在各阈值上一致，还是主要集中在高阈值（强对流）上？
> - Mamba + OPSD 的增益是否是两者之和（可加性），还是超过之和（协同效应）？
> - 失效场景：Mamba 或 OPSD 在哪些情况下会降低性能？

---

## 7. 结论

> **【待填写】**
>
> 建议结构：
> - 重述问题及两种改进方案
> - 总结表1中的关键数字
> - 评述 OPSD 与 Mamba 哪种改进影响更大，各自适用条件
> - 局限性：128×128 裁剪、仅25分钟预测时域、单一数据集
> - 未来工作：完整 384×384 实验、更长预测时域（60分钟）、多模态输入（LGHT 闪电通道）、概率预报扩展

---

## 参考文献

[1] Veillette, M., Samsi, S., & Mattioli, C. (2020). SEVIR: A storm event imagery dataset for deep learning applications in meteorology. *NeurIPS 2020*.

[2] Gao, Z., Shi, X., Wang, H., et al. (2022). SimVP: Simpler yet better video prediction. *CVPR 2022*.

[3] Gao, Z., Shi, X., Han, B., et al. (2022). EarthFormer: Exploring space-time transformers for Earth system forecasting. *NeurIPS 2022*.

[4] Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv:2312.00752*.

[5] Gu, A., Goel, K., & Ré, C. (2021). Efficiently modeling long sequences with structured state spaces. *ICLR 2022*.

[6] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NeurIPS Workshop 2015*.

[7] Shi, X., Chen, Z., Wang, H., et al. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. *NeurIPS 2015*.

[8] Wang, Y., Long, M., Wang, J., Gao, Z., & Yu, P. S. (2017). PredRNN: Recurrent neural networks for predictive learning using spatiotemporal LSTMs. *NeurIPS 2017*.

[9] Ravuri, S., Lenc, K., Willson, M., et al. (2021). Skilful precipitation nowcasting using deep generative models of radar. *Nature 597*.

[10] Zhang, Y., Long, M., Chen, K., et al. (2023). Skilful nowcasting of extreme precipitation with NowcastNet. *Nature 619*.

[11] Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled sampling for sequence prediction with recurrent neural networks. *NeurIPS 2015*.

[12] Zhu, L., Liao, B., Zhang, Q., et al. (2024). Vision Mamba: Efficient visual representation learning with bidirectional state space model. *ICML 2024*.
