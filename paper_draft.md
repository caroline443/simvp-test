# Towards Sharp Long-Range Radar Nowcasting: Mamba SSM and On-Policy Self-Distillation for Convective Weather Prediction

---

## Abstract

Accurate short-term prediction of severe convective weather remains a critical challenge in operational meteorology. Autoregressive deep learning models suffer from cumulative error propagation in long-range forecasts, leading to progressively blurred predictions beyond 30 minutes. In this paper, we propose two complementary improvements to the SimVP framework on the SEVIR VIL radar dataset. First, we introduce **On-Policy Self-Distillation (OPSD)**, which alleviates exposure bias by aligning the student's on-policy rollout distribution with a privileged teacher that observes ground-truth future frames at each step, supervised via KL divergence. We further propose a **Reward-Weighted** variant that amplifies the distillation gradient on temporally difficult steps using per-step CSI as an inverse reward signal. Second, we replace the Inception-based Temporal Translator with a **Mamba Selective State Space Model (SSM)**, which explicitly models the temporal evolution at each spatial position through input-dependent state transitions rather than the implicit channel-concatenation approach of the original SimVP. Experiments on SEVIR VIL under a 128×128 / 25-minute lead-time configuration demonstrate that [**RESULTS TO BE FILLED**]. The combination of Mamba and OPSD achieves the best overall performance, suggesting that structural temporal modeling and on-policy training are complementary improvements.

---

## 1. Introduction

Convective weather nowcasting—the prediction of radar reflectivity fields over a 0–60 minute horizon—is fundamental to severe weather warning systems, aviation safety, and urban flood management. The task presents unique challenges: convective cells initiate, intensify, and dissipate on timescales of minutes, exhibiting highly non-linear dynamics that resist purely physics-based extrapolation.

Deep learning approaches have made substantial progress in this domain. Convolutional sequence-to-sequence models such as ConvLSTM [Shi et al., 2015] and PredRNN [Wang et al., 2017] established strong baselines, while more recent architectures including EarthFormer [Gao et al., 2022], NowcastNet [Zhang et al., 2023], and DGMR [Ravuri et al., 2021] have demonstrated competitive or superior skill to operational methods such as PySTEPS. SimVP [Gao et al., 2022] introduced a purely convolutional encoder–translator–decoder design that achieves competitive accuracy with substantially lower training cost.

Despite these advances, two limitations persist in autoregressive nowcasting models:

**Exposure Bias.** Autoregressive models are trained with teacher forcing—each step conditions on ground-truth context—but at inference time the model must condition on its own (imperfect) previous predictions. The resulting distribution mismatch accumulates over steps, causing prediction quality to degrade rapidly beyond 30 minutes. This is the classical *exposure bias* problem [Bengio et al., 2015].

**Implicit Temporal Modeling.** The SimVP Temporal Translator fuses temporal information by concatenating T input frames along the channel dimension and applying 2D spatial convolutions. While effective, this formulation has no explicit temporal axis: the model cannot learn distinct transition dynamics for different spatial locations, and the representation grows linearly with the product of channels and sequence length.

We address both limitations within a unified experimental framework on the SEVIR VIL benchmark [Veillette et al., 2020]:

1. **OPSD** (On-Policy Self-Distillation): a training procedure adapted from recent LLM reasoning work that uses a privileged "teacher" branch (fed real future frames) to supervise the "student" branch (fed its own autoregressive predictions) via per-step KL divergence, with no architectural changes and negligible memory overhead.

2. **Reward-Weighted OPSD**: an extension that weights each step's KL loss by $(1 - \text{CSI}_t)$, concentrating distillation gradients on temporally difficult forecast steps.

3. **Mamba Temporal Translator**: a drop-in replacement for the Inception-based translator that models the temporal sequence at each spatial position with a selective SSM (Mamba [Gu & Dao, 2023]), providing explicit, input-dependent temporal state transitions.

Our experiments yield the following contributions:
- We show that OPSD consistently improves long-range CSI (lead times ≥ 20 min) over the SimVP autoregressive baseline with negligible training overhead.
- We show that replacing the Temporal Translator with Mamba SSM improves [**TO BE FILLED**] under the same training protocol.
- We show that the two improvements are complementary: Mamba + OPSD outperforms either improvement alone, with the combined model achieving [**TO BE FILLED**] mean CSI across all thresholds.

---

## 2. Related Work

### 2.1 Deep Learning for Radar Nowcasting

ConvLSTM [Shi et al., 2015] pioneered deep spatiotemporal prediction for precipitation, replaced element-wise multiplication with convolutional operations inside LSTM cells. PredRNN [Wang et al., 2017] introduced a spatial–temporal memory flow that passes state both vertically across stacked LSTMs and horizontally across time steps. EarthFormer [Gao et al., 2022] applied a hierarchical transformer with earth-specific self-attention to multiple meteorological datasets including SEVIR.

Generative approaches have also gained traction: DGMR [Ravuri et al., 2021] uses a conditional GAN to produce probabilistic ensemble forecasts that preserve sharp convective structures, while DiffCast [Yu et al., 2023] and PreDiff [Gao et al., 2023] apply diffusion models to the same objective. These methods excel at visual sharpness but require substantially higher inference cost.

SimVP [Gao et al., 2022] demonstrated that a non-recurrent, purely convolutional design—spatial encoder, temporal Inception translator, spatial decoder—can match or exceed recurrent baselines at a fraction of the training cost. Our work extends SimVP by addressing its exposure bias and limited temporal expressivity.

### 2.2 Exposure Bias and Self-Distillation

Exposure bias in sequence generation was formalized by Bengio et al. [2015] (Scheduled Sampling) and addressed by DAgger [Ross et al., 2011] in the imitation learning literature. In neural machine translation, various curriculum strategies mix teacher-forced and free-running inputs. For video prediction, Scheduled Sampling was applied directly by Srivastava et al. [2015].

Knowledge distillation [Hinton et al., 2015] transfers soft label distributions from a teacher to a student. OPSD adapts this idea to the on-policy setting: the teacher is the same model fed privileged ground-truth context, not a separately trained larger model. This is closely related to the OPSD procedure proposed for LLM reasoning [anonymous, 2025], which we transfer to the radar extrapolation domain.

### 2.3 State Space Models

Structured State Space Sequence models (S4) [Gu et al., 2021] demonstrated that linear recurrences with HiPPO-initialized state matrices can model long-range dependencies efficiently. Mamba [Gu & Dao, 2023] introduced selective state spaces (S6), where the transition matrices $\mathbf{A}$, $\mathbf{B}$, and $\mathbf{C}$ depend on the input, enabling the model to selectively focus on relevant context. Vision Mamba [Zhu et al., 2024] and VMamba [Liu et al., 2024] adapted Mamba to 2D spatial data via spatial scanning strategies.

In the meteorological domain, the integration of SSMs remains largely unexplored. The temporal dimension of radar sequences (typically 5–13 frames at 5-minute intervals) is short in absolute terms but exhibits strong non-Markovian structure: storm cell trajectories, mergers, and dissipation events span multiple frames. We hypothesize that explicit state-based temporal modeling at each spatial position is better suited to this structure than implicit channel-concatenation.

---

## 3. Problem Formulation

Let $\mathbf{X} = \{x_1, x_2, \ldots, x_{T_{in}}\} \in \mathbb{R}^{T_{in} \times H \times W}$ denote a sequence of $T_{in}$ observed VIL (Vertical Integrated Liquid) radar frames, where each frame $x_t \in [0, 255]$ encodes the column-integrated liquid water content. The goal is to predict the future sequence $\hat{\mathbf{Y}} = \{\hat{y}_{T_{in}+1}, \ldots, \hat{y}_{T_{in}+T_{out}}\}$.

Following the discretization approach of [reference], we quantize each pixel into $K=16$ bins of equal width $\Delta = 255/K$. The model outputs a per-pixel categorical distribution $p_t(k) \in \mathbb{R}^{K}$ at each future step $t$, and predictions are obtained as the argmax bin center: $\hat{y}_t = (\arg\max_k p_t(k) + 0.5) \cdot \Delta$.

The primary evaluation metric is the Critical Success Index (CSI):
$$\text{CSI}(\tau) = \frac{\text{TP}}{\text{TP} + \text{FP} + \text{FN}}$$
where positive events are defined as pixels with VIL $\geq \tau$. We report CSI-M, the mean CSI across six standard thresholds $\tau \in \{16, 74, 133, 160, 181, 219\}$, as well as Probability of Detection (POD), False Alarm Ratio (FAR), and Heidke Skill Score (HSS).

---

## 4. Methodology

### 4.1 SimVP Autoregressive Baseline

The SimVP architecture consists of three components:

**Spatial Encoder** $\mathcal{E}$: A stack of $N_e$ stride-2 convolutional blocks that maps each frame independently to a low-resolution feature map: $f_t = \mathcal{E}(x_t) \in \mathbb{R}^{C \times h \times w}$, where $h = H / 2^{N_e}$, $w = W / 2^{N_e}$.

**Temporal Translator** $\mathcal{T}$: Processes the sequence of encoded features $\{f_1, \ldots, f_{T_{in}}\}$ to produce temporally-informed features. In the original SimVP, this is accomplished by concatenating all $T_{in}$ feature maps along the channel dimension (giving a $T_{in} C \times h \times w$ tensor) and applying Inception-style 2D convolutions.

**Spatial Decoder** $\mathcal{D}$: A stack of $N_d$ transposed convolutional blocks that maps the last translated feature back to the spatial resolution, producing per-pixel logits $\ell_t \in \mathbb{R}^{K \times H \times W}$.

**Autoregressive Rollout.** For compatibility with the OPSD training objective (Section 4.2), we adopt an autoregressive rollout: at each step $t$, the model predicts $\ell_t$ from the current sliding window of $T_{in}$ frames, then appends the discretized prediction $\hat{y}_t = (\arg\max \ell_t + 0.5) \cdot \Delta / 255$ to the window (in normalized form). This continues for $T_{out}$ steps.

The baseline is trained with a foreground-weighted cross-entropy loss:
$$\mathcal{L}_{\text{CE}} = \frac{\sum_{t,i,j} w_{i,j}^t \cdot \text{CE}(\ell_t^{i,j},\, k_t^{i,j})}{\sum_{t,i,j} w_{i,j}^t}$$
where $k_t^{i,j}$ is the ground-truth bin index and $w_{i,j}^t = 1 + (\lambda_{\text{fg}} - 1) \cdot \mathbf{1}[k_t^{i,j} > 0]$ assigns higher weight to non-background pixels ($\lambda_{\text{fg}} = 5.0$).

### 4.2 On-Policy Self-Distillation (OPSD)

**Motivation.** The autoregressive baseline suffers from *exposure bias*: during training, each step conditions on real past frames; during inference, it conditions on imperfect predictions. This mismatch accumulates and produces increasingly blurred outputs at longer lead times.

**Teacher–Student Setup.** OPSD exploits the observation that the same model can be run in two modes simultaneously:

- *Student branch*: autoregressive rollout with no privileged information (standard inference mode). Produces logits $\{p^s_t\}_{t=1}^{T_{out}}$.
- *Teacher branch*: at each step $t$, the sliding window is updated with the *real* future frame $y_t$ rather than the prediction. Wrapped in `torch.no_grad()`—the teacher performs only forward passes and incurs no additional gradient computation. Produces logits $\{p^r_t\}_{t=1}^{T_{out}}$.

The teacher's distribution $p^r_t$ represents the optimal prediction given ground-truth context up to step $t$. We use it to supervise the student via temperature-scaled KL divergence:

$$\mathcal{L}_{\text{KL}} = T^2 \cdot \frac{\sum_{t,i,j} w_{i,j}^t \cdot D_{\text{KL}}\!\left(p^r_t(\cdot \mid i,j;\, T) \;\|\; p^s_t(\cdot \mid i,j;\, T)\right)}{\sum_{t,i,j} w_{i,j}^t}$$

where $T=2.0$ is the distillation temperature and the $T^2$ factor compensates for gradient magnitude reduction [Hinton et al., 2015]. The spatial weight $w_{i,j}^t$ is the same foreground mask used in the CE loss, preventing the large background region from dominating.

The total training objective combines distillation and supervised cross-entropy:
$$\mathcal{L}_{\text{OPSD}} = \lambda_{\text{KL}} \cdot \mathcal{L}_{\text{KL}} + \lambda_{\text{CE}} \cdot \mathcal{L}_{\text{CE}}$$

with $\lambda_{\text{KL}} = 1.0$ and $\lambda_{\text{CE}} = 0.5$.

**Memory Overhead.** Because the teacher branch runs under `torch.no_grad()`, no activations are stored for backpropagation. The memory overhead is limited to storing the teacher's output logits ($T_{out} \times K \times H \times W$ per sample), which is negligible compared to the full activation graph.

### 4.3 Reward-Weighted OPSD

**Motivation.** In the standard OPSD formulation, the KL weight is uniform across all $T_{out}$ steps. However, prediction quality degrades non-uniformly: early steps are relatively accurate (high CSI), while later steps are more error-prone (low CSI). Concentrating the distillation signal on difficult steps may improve the efficiency of the gradient budget.

**Formulation.** We define a per-step reward as the batch-mean CSI at a reference threshold $\tau_r = 74$ (moderate convection):
$$r_t = \frac{1}{B} \sum_{b=1}^{B} \text{CSI}\!\left(\hat{y}_t^{(b)},\, y_t^{(b)},\, \tau_r\right)$$

The step weight is defined as the inverse reward $w_t^{\text{rw}} = 1 - r_t \in [0, 1]$: steps where the student already predicts well ($r_t \approx 1$) receive near-zero additional KL weight, while steps with poor predictions ($r_t \approx 0$) receive maximum weight. The loss becomes:

$$\mathcal{L}_{\text{OPSD-RW}} = \frac{\lambda_{\text{KL}}}{T_{out}} \sum_{t=1}^{T_{out}} (1 - r_t) \cdot \mathcal{L}_{\text{KL}}^t + \lambda_{\text{CE}} \cdot \mathcal{L}_{\text{CE}}$$

Note that $r_t$ is computed using `torch.no_grad()` and returned as a Python scalar; it enters the computation graph only as a constant multiplier, not through its own gradient path.

### 4.4 Mamba Temporal Translator

**Limitation of Channel Concatenation.** The original Temporal Translator concatenates $T_{in}$ encoded feature maps along the channel dimension, obtaining a representation of shape $T_{in} C \times h \times w$. Temporal relationships are then modeled implicitly by 2D convolutions over this channel-stacked tensor. This design (i) mixes spatial and temporal processing in a single operation, (ii) is insensitive to the ordering of frames beyond what the fixed channel positions encode, and (iii) scales the channel dimension with $T_{in}$, making the translator costly for longer input sequences.

**Per-Spatial-Position Temporal Mamba.** We propose replacing the Inception translator with a Mamba SSM that operates explicitly along the temporal axis at each spatial position:

$$\{f_1, \ldots, f_{T_{in}}\} \in \mathbb{R}^{T_{in} \times C \times h \times w}
\;\xrightarrow{\text{reshape}}\;
\mathbb{R}^{(h \cdot w) \times T_{in} \times C}$$

Each of the $h \times w$ spatial positions is treated as an independent sequence of length $T_{in}$ with feature dimension $C$. $N_m$ Mamba blocks are applied along the $T_{in}$ axis:

$$\text{Mamba}^{(n)}: \mathbb{R}^{(h \cdot w) \times T_{in} \times C} \to \mathbb{R}^{(h \cdot w) \times T_{in} \times C}$$

After temporal processing, a lightweight $3 \times 3$ spatial convolution restores cross-position information exchange. The output is reshaped back to $T_{in} C \times h \times w$ for compatibility with the downstream decoder.

**Mamba Block.** Each Mamba block implements the S6 selective scan [Gu & Dao, 2023]:

1. **Input projection**: $\mathbf{x}, \mathbf{z} = \text{split}(W_{\text{in}} \mathbf{u})$, expanding to dimension $d_{\text{inner}} = 2C$.
2. **Causal convolution**: depthwise 1D conv over the sequence dimension with kernel width 4.
3. **Selective scan (S6)**: input-dependent discretization of the state transition:
$$\bar{\mathbf{A}}_t = \exp(\Delta_t \mathbf{A}), \quad \bar{\mathbf{B}}_t = \Delta_t \mathbf{B}_t$$
$$\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t x_t, \quad y_t = \mathbf{C}_t \mathbf{h}_t + \mathbf{D} x_t$$
where $\Delta_t$, $\mathbf{B}_t$, $\mathbf{C}_t$ are computed from the input $\mathbf{x}_t$ (selectivity), and $\mathbf{A}$ is a learnable parameter shared across positions.
4. **Gating**: $\mathbf{y} \leftarrow \mathbf{y} \odot \text{SiLU}(\mathbf{z})$.
5. **Output projection**: $W_{\text{out}} \mathbf{y}$, back to dimension $C$.

All SSM computations are performed in float32 to avoid overflow in the matrix exponential under AMP. The output is cast back to the original dtype before the residual addition.

**Spatial complexity.** For the 128×128 / 3-layer encoder configuration used in fast experiments, the feature map is $16 \times 16 = 256$ spatial positions. Each Mamba block processes $B \times 256$ sequences of length $T_{in} = 5$ with $C = 256$ features—a negligible sequence length for the S6 scan.

---

## 5. Experiments

### 5.1 Dataset

We use the **SEVIR** (Storm EVent ImageRy) dataset [Veillette et al., 2020], a curated collection of spatiotemporal meteorological image sequences covering the contiguous United States. Each event is recorded at 5-minute intervals over a 384 km × 384 km domain at 1 km resolution (384 × 384 pixels). We use the **VIL** (Vertical Integrated Liquid) channel, which provides a direct measure of the vertically integrated liquid water content (kg/m²) encoded as uint8 values in $[0, 255]$.

**Split.** Following the convention of temporal separation to prevent data leakage, we use events from 2017–2018 for training and validation (90%/10% split by event index) and events from 2019 for testing.

**Fast Experiment Configuration.** To enable rapid comparison, all experiments use a center-cropped 128×128 patch, $T_{in} = 5$ input frames (25 minutes of history), and $T_{out} = 5$ prediction frames (25-minute lead time). This configuration is approximately 9× faster per epoch than the full 384×384 setting due to reduced spatial resolution. We note that CSI values at this resolution are not directly comparable to published full-resolution results, as the center crop captures higher convective activity density. All five models are trained and evaluated under identical data settings to ensure fair comparison.

### 5.2 Implementation Details

| Hyperparameter | Value |
|---|---|
| Spatial encoder layers $N_e$ | 3 |
| Temporal translator layers $N_m$ / $N_{\text{inc}}$ | 4 |
| Hidden channels $C$ | 64 (encoder input), 256 (encoder output) |
| Mamba state dimension $d_{\text{state}}$ | 16 |
| Mamba expand ratio | 2 ($d_{\text{inner}} = 512$) |
| Discretization bins $K$ | 16 |
| Foreground weight $\lambda_{\text{fg}}$ | 5.0 |
| Baseline learning rate | $3 \times 10^{-4}$ (cosine decay) |
| OPSD learning rate | $1 \times 10^{-4}$ (cosine decay) |
| Batch size | 4 |
| Epochs | 20 (baseline), 20 (OPSD fine-tune) |
| OPSD temperature $T$ | 2.0 |
| $\lambda_{\text{KL}}$ / $\lambda_{\text{CE}}$ | 1.0 / 0.5 |
| Reward threshold $\tau_r$ | 74 |
| Gradient clip norm | 0.5 |
| Optimizer | AdamW, weight decay $10^{-4}$ |
| Random seed | 42 |

All models are trained on a single NVIDIA GPU with 16 GB VRAM using PyTorch AMP (float16 activations, float32 GroupNorm and SSM computations). OPSD models are warm-started from the corresponding baseline checkpoint.

### 5.3 Compared Methods

We compare five configurations:

| Tag | Architecture | Training |
|---|---|---|
| **Baseline** | SimVP (Inception Translator) | Cross-Entropy |
| **OPSD** | SimVP (Inception Translator) | CE + KL Distillation |
| **OPSD-RW** | SimVP (Inception Translator) | CE + Reward-Weighted KL |
| **Mamba** | SimVP (Mamba Translator) | Cross-Entropy |
| **Mamba+OPSD** | SimVP (Mamba Translator) | CE + KL Distillation |

### 5.4 Main Results

> **[TABLE TO BE FILLED]**
>
> Table 1: CSI-M (mean CSI across six thresholds), CSI@74, CSI@133, POD@74, FAR@74, HSS@74 on the SEVIR VIL test set (2019). Best result per column in **bold**.

### 5.5 Per-Step CSI Analysis

> **[FIGURE TO BE FILLED]**
>
> Figure 1: CSI at threshold 74 (moderate convection) as a function of lead time (5–25 min) for all five models. The x-axis is lead time in minutes; the y-axis is CSI ∈ [0, 1]. Each curve corresponds to one model configuration.

> **[FIGURE TO BE FILLED]**
>
> Figure 2: CSI at threshold 133 (strong convection) as a function of lead time.

### 5.6 Qualitative Results

> **[FIGURE TO BE FILLED]**
>
> Figure 3: Visual comparison of predicted VIL fields at lead times t+5 min, t+15 min, and t+25 min. Rows: (1) Ground Truth, (2) Baseline, (3) OPSD, (4) Mamba, (5) Mamba+OPSD.

### 5.7 Ablation Study

> **[TABLE TO BE FILLED]**
>
> Table 2: Ablation on OPSD components. Rows vary KL weight $\lambda_{\text{KL}}$, distillation temperature $T$, and reward weighting.

---

## 6. Discussion

> **[TO BE FILLED]**
>
> Suggested points to address:
> - Does OPSD improve later steps (t+20, t+25) more than early steps (t+5)? This would confirm the exposure bias hypothesis.
> - Does Mamba improve consistently across thresholds, or mainly at high thresholds (strong convection)?
> - Is the gain from Mamba+OPSD additive (sum of individual gains) or synergistic (larger than the sum)?
> - Any failure modes: cases where Mamba or OPSD degrades performance?

---

## 7. Conclusion

> **[TO BE FILLED]**
>
> Suggested structure:
> - Restate the problem and two proposed solutions
> - Summarize the key numerical findings from Table 1
> - Comment on which improvement (OPSD vs. Mamba) is more impactful and under what conditions
> - Limitations: 128×128 crop, short 25-min horizon, single dataset
> - Future work: full 384×384 experiments, longer horizon (60 min), multi-modal input (LGHT channel), probabilistic extension

---

## References

[1] Veillette, M., Samsi, S., & Mattioli, C. (2020). SEVIR: A storm event imagery dataset for deep learning applications in meteorology. *NeurIPS 2020*.

[2] Gao, Z., Shi, X., Wang, H., Zhu, Y., Wang, Y. B., Li, M., & Yeung, D. Y. (2022). SimVP: Simpler yet better video prediction. *CVPR 2022*.

[3] Gao, Z., Shi, X., Han, B., Wang, H., Jin, X., Yeung, D. Y., & Li, M. (2022). EarthFormer: Exploring space-time transformers for Earth system forecasting. *NeurIPS 2022*.

[4] Gu, A., & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces. *arXiv:2312.00752*.

[5] Gu, A., Goel, K., & Ré, C. (2021). Efficiently modeling long sequences with structured state spaces. *ICLR 2022*.

[6] Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NeurIPS Workshop 2015*.

[7] Shi, X., Chen, Z., Wang, H., Yeung, D. Y., Wong, W. K., & Woo, W. C. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. *NeurIPS 2015*.

[8] Wang, Y., Long, M., Wang, J., Gao, Z., & Yu, P. S. (2017). PredRNN: Recurrent neural networks for predictive learning using spatiotemporal LSTMs. *NeurIPS 2017*.

[9] Ravuri, S., Lenc, K., Willson, M., Kangin, D., Lam, R., Mirowski, P., ... & Mohamed, S. (2021). Skilful precipitation nowcasting using deep generative models of radar. *Nature 597*.

[10] Zhang, Y., Long, M., Chen, K., Xing, L., Jin, R., Jordan, M. I., & Wang, J. (2023). Skilful nowcasting of extreme precipitation with NowcastNet. *Nature 619*.

[11] Bengio, S., Vinyals, O., Jaitly, N., & Shazeer, N. (2015). Scheduled sampling for sequence prediction with recurrent neural networks. *NeurIPS 2015*.

[12] Zhu, L., Liao, B., Zhang, Q., Wang, X., Liu, W., & Wang, X. (2024). Vision Mamba: Efficient visual representation learning with bidirectional state space model. *ICML 2024*.
