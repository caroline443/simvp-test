"""
论文框架图绘制脚本
生成三张图：
  1. fig1_overall_arch.pdf  — 整体架构图
  2. fig2_mamba_translator.pdf — Mamba 翻译器细节
  3. fig3_opsd.pdf — OPSD 教师-学生框架
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 颜色方案 ────────────────────────────────────────────────────────────────
C_BLUE      = "#0070C0"
C_DARKBLUE  = "#003366"
C_ORANGE    = "#FF6600"
C_GREEN     = "#00A651"
C_PURPLE    = "#7030A0"
C_GRAY      = "#F2F2F2"
C_DARKGRAY  = "#595959"
C_WHITE     = "#FFFFFF"
C_RED       = "#C00000"
C_LIGHTBLUE = "#DDEEFF"
C_YELLOW    = "#FFD966"


def add_box(ax, x, y, w, h, label, sublabel=None, color=C_LIGHTBLUE,
            fontsize=10, subfontsize=8, text_color="black", radius=0.04,
            bold=False):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor=C_DARKBLUE, linewidth=1.2, zorder=3)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    if sublabel:
        ax.text(x, y + h*0.12, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=text_color, zorder=4)
        ax.text(x, y - h*0.2, sublabel, ha="center", va="center",
                fontsize=subfontsize, color=C_DARKGRAY, zorder=4, style="italic")
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=text_color, zorder=4)


def add_arrow(ax, x1, y1, x2, y2, color=C_DARKBLUE, lw=1.5, label=None, fontsize=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw), zorder=5)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.02, my+0.02, label, fontsize=fontsize, color=color, zorder=6)


# ═══════════════════════════════════════════════════════════════════════════════
# 图 1：整体架构图
# ═══════════════════════════════════════════════════════════════════════════════
def draw_overall_arch():
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_facecolor(C_WHITE)
    fig.patch.set_facecolor(C_WHITE)

    ax.text(7, 4.65, "Overall Architecture of Proposed Framework",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_DARKBLUE)

    # ── 输入序列 ──
    for i, t in enumerate(range(1, 7)):
        xi = 0.5 + i * 0.38
        box = FancyBboxPatch((xi-0.16, 1.8), 0.32, 1.4,
                             boxstyle="round,pad=0.02",
                             facecolor=C_LIGHTBLUE, edgecolor=C_BLUE, linewidth=1, zorder=3)
        ax.add_patch(box)
        ax.text(xi, 2.5, f"$x_{t}$", ha="center", va="center", fontsize=9, color=C_DARKBLUE, zorder=4)
    ax.text(1.4, 1.55, "Input Frames\n$[B, T_{in}, 1, H, W]$",
            ha="center", va="top", fontsize=8, color=C_DARKGRAY)

    # ── 箭头 输入→编码器 ──
    add_arrow(ax, 2.85, 2.5, 3.3, 2.5, lw=2)

    # ── Spatial Encoder ──
    add_box(ax, 3.85, 2.5, 1.0, 1.5, "Spatial\nEncoder", "$\\mathcal{E}$\n4× stride-2 conv",
            color=C_BLUE, text_color=C_WHITE, fontsize=10, subfontsize=7.5, bold=True)
    ax.text(3.85, 1.55, "$[B{\\cdot}T, C, h, w]$",
            ha="center", va="top", fontsize=7.5, color=C_DARKGRAY)

    # ── 箭头 →翻译器 ──
    add_arrow(ax, 4.38, 2.5, 4.85, 2.5, lw=2)

    # ── Mamba Temporal Translator ──
    add_box(ax, 5.85, 2.5, 1.9, 1.7,
            "Mamba\nTemporal Translator",
            "S6 Selective Scan\n$[B{\\cdot}hw, T, C]$",
            color=C_PURPLE, text_color=C_WHITE, fontsize=10, subfontsize=7.5, bold=True)
    ax.text(5.85, 1.55, "$[B{\\cdot}T, C, h, w]$",
            ha="center", va="top", fontsize=7.5, color=C_DARKGRAY)

    # ── 箭头 →解码器 ──
    add_arrow(ax, 6.82, 2.5, 7.3, 2.5, lw=2)

    # ── Spatial Decoder ──
    add_box(ax, 7.85, 2.5, 1.0, 1.5, "Spatial\nDecoder", "$\\mathcal{D}$\n4× transposed conv",
            color=C_BLUE, text_color=C_WHITE, fontsize=10, subfontsize=7.5, bold=True)
    ax.text(7.85, 1.55, "$[B, K, H, W]$",
            ha="center", va="top", fontsize=7.5, color=C_DARKGRAY)

    # ── 箭头 →输出 ──
    add_arrow(ax, 8.38, 2.5, 8.85, 2.5, lw=2)

    # ── 自回归展开框 ──
    ar_box = FancyBboxPatch((8.85, 1.65), 2.6, 1.7,
                             boxstyle="round,pad=0.06",
                             facecolor=C_GRAY, edgecolor=C_ORANGE, linewidth=1.5,
                             linestyle="--", zorder=3)
    ax.add_patch(ar_box)
    ax.text(10.15, 3.1, "Autoregressive Rollout", ha="center", va="center",
            fontsize=9, fontweight="bold", color=C_ORANGE, zorder=4)

    for i, t in enumerate(range(1, 7)):
        xi = 9.1 + i * 0.37
        box2 = FancyBboxPatch((xi-0.15, 1.8), 0.3, 0.9,
                              boxstyle="round,pad=0.02",
                              facecolor=C_YELLOW, edgecolor=C_ORANGE, linewidth=1, zorder=4)
        ax.add_patch(box2)
        ax.text(xi, 2.25, f"$\\hat{{y}}_{t}$", ha="center", va="center",
                fontsize=9, color=C_DARKBLUE, zorder=5)

    # 自回归反馈箭头
    ax.annotate("", xy=(9.5, 1.65), xytext=(10.15, 1.65),
                arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.2,
                                connectionstyle="arc3,rad=0.4"), zorder=5)
    ax.text(9.82, 1.2, "append to context", ha="center", fontsize=7.5,
            color=C_ORANGE, style="italic")

    # ── OPSD 训练目标框（右侧） ──
    opsd_box = FancyBboxPatch((11.7, 1.5), 2.1, 2.0,
                               boxstyle="round,pad=0.06",
                               facecolor="#FFF0E0", edgecolor=C_RED, linewidth=1.5, zorder=3)
    ax.add_patch(opsd_box)
    ax.text(12.75, 3.28, "Training Objective", ha="center", fontsize=9,
            fontweight="bold", color=C_RED, zorder=4)
    ax.text(12.75, 2.85, "$\\mathcal{L}_{\\mathrm{OPSD}}=$", ha="center",
            fontsize=9, color=C_DARKBLUE, zorder=4)
    ax.text(12.75, 2.45, "$\\lambda_{KL}\\cdot\\mathcal{L}_{KL}$", ha="center",
            fontsize=9, color=C_PURPLE, zorder=4)
    ax.text(12.75, 2.1, "$+\\,\\lambda_{CE}\\cdot\\mathcal{L}_{CE}$", ha="center",
            fontsize=9, color=C_BLUE, zorder=4)
    ax.text(12.75, 1.72, "OPSD-RW: weight $(1-r_t)$", ha="center",
            fontsize=7.5, color=C_ORANGE, style="italic", zorder=4)

    add_arrow(ax, 11.45, 2.5, 11.68, 2.5, color=C_RED, lw=1.5)

    plt.tight_layout(pad=0.2)
    path = os.path.join(OUT_DIR, "fig1_overall_arch.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 图 2：Mamba 翻译器细节
# ═══════════════════════════════════════════════════════════════════════════════
def draw_mamba_translator():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [1, 1]})
    fig.patch.set_facecolor(C_WHITE)
    fig.suptitle("Mamba Temporal Translator", fontsize=13,
                 fontweight="bold", color=C_DARKBLUE, y=0.97)

    # ── 左图：reshape流程 ──
    ax = axes[0]
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor(C_WHITE)
    ax.set_title("Spatial-to-Temporal Reshape", fontsize=10, color=C_DARKBLUE, pad=6)

    # 输入特征图（多帧叠加示意）
    for i in range(4):
        rect = FancyBboxPatch((0.2+i*0.12, 4.5-i*0.12), 1.2, 0.9,
                              boxstyle="round,pad=0.02",
                              facecolor=C_LIGHTBLUE, edgecolor=C_BLUE,
                              linewidth=1, alpha=0.85, zorder=3+i)
        ax.add_patch(rect)
    ax.text(1.0, 5.05, "$[B{\\cdot}T, C, h, w]$", ha="center", va="center",
            fontsize=9, color=C_DARKBLUE, zorder=8)
    ax.text(1.0, 4.35, "T=6 frames\n$h\\times w$ spatial locs", ha="center",
            fontsize=8, color=C_DARKGRAY)

    add_arrow(ax, 1.55, 4.5, 1.55, 3.8, lw=2, color=C_DARKBLUE,
              label="reshape\n+ permute")

    # 序列示意（多条时序序列）
    for i in range(5):
        yy = 3.4 - i*0.32
        rect2 = FancyBboxPatch((0.3, yy), 2.2, 0.22,
                               boxstyle="round,pad=0.02",
                               facecolor=C_PURPLE if i==2 else "#E8D5F5",
                               edgecolor=C_PURPLE, linewidth=0.8, zorder=3)
        ax.add_patch(rect2)
        if i == 2:
            ax.text(1.4, yy+0.11, "position $(i,j)$: $[T, C]$ sequence",
                    ha="center", va="center", fontsize=7.5, color=C_WHITE, zorder=4)
        elif i == 0:
            ax.text(1.4, yy+0.11, "position $(0,0)$", ha="center", va="center",
                    fontsize=7.5, color=C_PURPLE, zorder=4)
    ax.text(0.3, 1.7, "$[B{\\cdot}h{\\cdot}w,\\ T,\\ C]$\n$h\\times w$ independent sequences",
            ha="left", fontsize=8.5, color=C_PURPLE)

    add_arrow(ax, 1.55, 1.6, 1.55, 0.9, lw=2, color=C_PURPLE,
              label="Mamba ×N")

    # 输出
    for i in range(4):
        rect3 = FancyBboxPatch((0.2+i*0.12, 0.35-i*0.12), 1.2, 0.5,
                               boxstyle="round,pad=0.02",
                               facecolor="#E8D5F5", edgecolor=C_PURPLE,
                               linewidth=1, alpha=0.85, zorder=3+i)
        ax.add_patch(rect3)
    ax.text(1.0, 0.55, "$[B{\\cdot}T, C, h, w]$", ha="center", va="center",
            fontsize=9, color=C_PURPLE, zorder=8)

    # 3x3 spatial conv
    add_arrow(ax, 2.7, 0.6, 3.2, 0.6, lw=2, color=C_GREEN)
    add_box(ax, 3.7, 0.6, 0.9, 0.45, "$3{\\times}3$ Conv\n(spatial)", color="#D5F0E0",
            fontsize=8, bold=False)
    ax.text(3.7, 0.18, "restore cross-location\ninteraction", ha="center",
            fontsize=7.5, color=C_DARKGRAY)

    # ── 右图：Mamba Block ──
    ax2 = axes[1]
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 6.5)
    ax2.axis("off")
    ax2.set_facecolor(C_WHITE)
    ax2.set_title("Mamba Block (S6 Selective Scan)", fontsize=10,
                  color=C_DARKBLUE, pad=6)

    # 组件列表（从下到上）
    comps = [
        (2.5, 0.5,  1.8, 0.45, "Input $\\mathbf{u}_t$", C_LIGHTBLUE, "black"),
        (1.5, 1.3,  1.4, 0.45, "Linear Proj ×2\n(split $\\mathbf{x}, \\mathbf{z}$)", C_LIGHTBLUE, "black"),
        (3.5, 1.3,  0.9, 0.45, "Gate branch\n$\\mathbf{z}$", C_GRAY, C_DARKGRAY),
        (1.5, 2.2,  1.4, 0.45, "Depthwise Conv\n(causal, k=4)", "#D5F0E0", "black"),
        (1.5, 3.1,  1.4, 0.45, "SiLU Activation", "#D5F0E0", "black"),
        (1.5, 4.0,  1.8, 0.55, "S6 Selective Scan\n$\\Delta_t,B_t,C_t=f(\\mathbf{x}_t)$", "#E8D5F5", C_PURPLE),
        (2.5, 5.0,  1.4, 0.45, "Gate: $\\mathbf{y}{\\odot}\\mathrm{SiLU}(\\mathbf{z})$", C_YELLOW, "black"),
        (2.5, 5.9,  1.4, 0.45, "Output Proj + Residual", C_LIGHTBLUE, "black"),
    ]

    for (cx, cy, bw, bh, label, fc, tc) in comps:
        box = FancyBboxPatch((cx-bw/2, cy-bh/2), bw, bh,
                             boxstyle="round,pad=0.04",
                             facecolor=fc, edgecolor=C_DARKBLUE, linewidth=1, zorder=3)
        ax2.add_patch(box)
        ax2.text(cx, cy, label, ha="center", va="center",
                 fontsize=8, color=tc, zorder=4)

    # 垂直箭头（主路径）
    main_path = [(2.5,0.73),(1.5,1.08),(1.5,1.98),(1.5,2.88),(1.5,3.78),(2.5,4.73),(2.5,5.23),(2.5,5.68)]
    for i in range(len(main_path)-1):
        add_arrow(ax2, main_path[i][0], main_path[i][1],
                  main_path[i+1][0], main_path[i+1][1], lw=1.5)

    # 门控支路
    add_arrow(ax2, 2.5, 0.73, 3.5, 1.08, color=C_ORANGE, lw=1.2)
    ax2.annotate("", xy=(3.5, 4.75), xytext=(3.5, 1.53),
                arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.2), zorder=5)
    ax2.text(3.88, 3.1, "$\\mathbf{z}$", ha="center", fontsize=9, color=C_ORANGE)

    # 残差连接
    ax2.annotate("", xy=(2.5, 5.68), xytext=(4.4, 5.68),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.2,
                                connectionstyle="arc3,rad=0"), zorder=5)
    ax2.annotate("", xy=(4.4, 5.68), xytext=(4.4, 0.5),
                arrowprops=dict(arrowstyle="-", color=C_GREEN, lw=1.2), zorder=5)
    ax2.annotate("", xy=(4.4, 0.5), xytext=(3.4, 0.5),
                arrowprops=dict(arrowstyle="-", color=C_GREEN, lw=1.2), zorder=5)
    ax2.text(4.55, 3.1, "Residual", ha="left", fontsize=8, color=C_GREEN,
             rotation=90)

    # SSM 公式注释
    ax2.text(0.1, 4.0,
             "$\\bar{A}_t = e^{\\Delta_t A}$\n"
             "$h_t = \\bar{A}_t h_{t-1} + \\bar{B}_t x_t$\n"
             "$y_t = C_t h_t + D x_t$",
             ha="left", va="center", fontsize=7.5, color=C_PURPLE,
             bbox=dict(facecolor="#F5EEFF", edgecolor=C_PURPLE, boxstyle="round,pad=0.3"))

    plt.tight_layout(pad=0.5)
    path = os.path.join(OUT_DIR, "fig2_mamba_translator.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"saved {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 图 3：OPSD 教师-学生框架
# ═══════════════════════════════════════════════════════════════════════════════
def draw_opsd():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_facecolor(C_WHITE)
    fig.patch.set_facecolor(C_WHITE)
    ax.set_title("On-Policy Self-Distillation with Reward Weighting (OPSD-RW)",
                 fontsize=13, fontweight="bold", color=C_DARKBLUE, pad=10)

    # ── 共享模型（中心） ──
    model_box = FancyBboxPatch((5.5, 2.4), 3.0, 1.2,
                               boxstyle="round,pad=0.08",
                               facecolor=C_DARKBLUE, edgecolor=C_DARKBLUE,
                               linewidth=2, zorder=3)
    ax.add_patch(model_box)
    ax.text(7.0, 3.0, "Shared Model $\\mathcal{M}$\n(SimVP + Mamba Translator)",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=C_WHITE, zorder=4)

    # ── 历史帧输入 ──
    hist_box = FancyBboxPatch((0.3, 2.5), 2.0, 1.0,
                              boxstyle="round,pad=0.06",
                              facecolor=C_LIGHTBLUE, edgecolor=C_BLUE,
                              linewidth=1.5, zorder=3)
    ax.add_patch(hist_box)
    ax.text(1.3, 3.0, "Historical Frames\n$x_1, \\ldots, x_{T_{in}}$",
            ha="center", va="center", fontsize=9, color=C_DARKBLUE, zorder=4)

    add_arrow(ax, 2.32, 3.0, 5.48, 3.0, lw=2, color=C_DARKBLUE)

    # ── 真实未来帧 ──
    truth_box = FancyBboxPatch((0.3, 0.8), 2.0, 1.0,
                               boxstyle="round,pad=0.06",
                               facecolor="#D5F0E0", edgecolor=C_GREEN,
                               linewidth=1.5, zorder=3)
    ax.add_patch(truth_box)
    ax.text(1.3, 1.3, "Ground Truth\n$y_{T_{in}+1}, \\ldots, y_{T_{in}+T_{out}}$",
            ha="center", va="center", fontsize=9, color=C_GREEN, zorder=4)

    # ── 教师分支（上） ──
    teacher_box = FancyBboxPatch((5.5, 4.2), 3.0, 1.1,
                                 boxstyle="round,pad=0.06",
                                 facecolor="#D5F0E0", edgecolor=C_GREEN,
                                 linewidth=2, linestyle="--", zorder=3)
    ax.add_patch(teacher_box)
    ax.text(7.0, 4.75, "Teacher Branch\n(privileged: uses true future frames)\ntorch.no_grad() — zero memory overhead",
            ha="center", va="center", fontsize=8.5, color=C_GREEN, zorder=4)

    # 历史帧 → 教师
    ax.annotate("", xy=(5.5, 4.58), xytext=(2.32, 3.2),
                arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.5,
                                connectionstyle="arc3,rad=-0.25"), zorder=5)
    # 真实帧 → 教师
    ax.annotate("", xy=(5.5, 4.42), xytext=(2.32, 1.5),
                arrowprops=dict(arrowstyle="-|>", color=C_GREEN, lw=1.5,
                                connectionstyle="arc3,rad=0.3"), zorder=5)
    ax.text(3.5, 4.3, "oracle context", fontsize=8, color=C_GREEN, style="italic")

    # ── 学生分支（下） ──
    student_box = FancyBboxPatch((5.5, 0.7), 3.0, 1.1,
                                 boxstyle="round,pad=0.06",
                                 facecolor="#DDEEFF", edgecolor=C_BLUE,
                                 linewidth=2, zorder=3)
    ax.add_patch(student_box)
    ax.text(7.0, 1.25, "Student Branch\n(on-policy: uses own predictions)\nautoregressive rollout",
            ha="center", va="center", fontsize=8.5, color=C_BLUE, zorder=4)

    # 历史帧 → 学生
    ax.annotate("", xy=(5.5, 1.05), xytext=(2.32, 2.8),
                arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.5,
                                connectionstyle="arc3,rad=0.25"), zorder=5)

    # 自回归反馈
    ax.annotate("", xy=(7.0, 0.7), xytext=(8.5, 0.7),
                arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.2,
                                connectionstyle="arc3,rad=-0.4"), zorder=5)
    ax.text(7.75, 0.2, "append $\\hat{y}_t$ to context", fontsize=8,
            color=C_ORANGE, ha="center", style="italic")

    # ── 教师/学生输出 ──
    t_out = FancyBboxPatch((9.0, 4.3), 1.8, 0.9,
                           boxstyle="round,pad=0.05",
                           facecolor="#D5F0E0", edgecolor=C_GREEN, linewidth=1.5, zorder=3)
    ax.add_patch(t_out)
    ax.text(9.9, 4.75, "$p^r_t$ (teacher\nlogits)", ha="center", va="center",
            fontsize=8.5, color=C_GREEN, zorder=4)

    s_out = FancyBboxPatch((9.0, 0.8), 1.8, 0.9,
                           boxstyle="round,pad=0.05",
                           facecolor="#DDEEFF", edgecolor=C_BLUE, linewidth=1.5, zorder=3)
    ax.add_patch(s_out)
    ax.text(9.9, 1.25, "$p^s_t$ (student\nlogits)", ha="center", va="center",
            fontsize=8.5, color=C_BLUE, zorder=4)

    add_arrow(ax, 8.52, 4.75, 8.98, 4.75, color=C_GREEN, lw=1.5)
    add_arrow(ax, 8.52, 1.25, 8.98, 1.25, color=C_BLUE, lw=1.5)

    # ── KL 损失框 ──
    kl_box = FancyBboxPatch((11.0, 2.2), 2.7, 1.6,
                            boxstyle="round,pad=0.08",
                            facecolor="#FFF0E0", edgecolor=C_RED,
                            linewidth=2, zorder=3)
    ax.add_patch(kl_box)
    ax.text(12.35, 3.35, "Loss", ha="center", fontsize=10,
            fontweight="bold", color=C_RED, zorder=4)
    ax.text(12.35, 2.95,
            "$\\mathcal{L}_{KL} = T^2 D_{KL}(p^r_t \\| p^s_t)$",
            ha="center", fontsize=8.5, color=C_PURPLE, zorder=4)
    ax.text(12.35, 2.60,
            "$\\mathcal{L}_{CE}$ (cross-entropy)",
            ha="center", fontsize=8.5, color=C_BLUE, zorder=4)
    ax.text(12.35, 2.30,
            "weight: $(1-r_t)$, $r_t = \\mathrm{CSI}_t$",
            ha="center", fontsize=8, color=C_ORANGE, style="italic", zorder=4)

    add_arrow(ax, 10.82, 4.75, 11.0, 3.6, color=C_GREEN, lw=1.5)
    add_arrow(ax, 10.82, 1.25, 11.0, 2.4, color=C_BLUE, lw=1.5)

    # ── CSI reward 标注 ──
    rw_box = FancyBboxPatch((9.2, 2.5), 1.6, 1.0,
                            boxstyle="round,pad=0.05",
                            facecolor=C_YELLOW, edgecolor=C_ORANGE,
                            linewidth=1.5, zorder=3)
    ax.add_patch(rw_box)
    ax.text(10.0, 3.0, "Per-step reward\n$r_t = \\mathrm{CSI}_t(\\hat{y}_t, y_t)$",
            ha="center", va="center", fontsize=8, color=C_ORANGE, zorder=4)

    add_arrow(ax, 9.9, 1.7, 9.9, 2.48, color=C_ORANGE, lw=1.2)
    add_arrow(ax, 10.82, 3.0, 11.0, 3.0, color=C_ORANGE, lw=1.2)

    # ── 图例 ──
    legend_items = [
        mpatches.Patch(facecolor=C_GREEN, edgecolor=C_GREEN, label="Teacher (oracle context)"),
        mpatches.Patch(facecolor=C_BLUE,  edgecolor=C_BLUE,  label="Student (on-policy)"),
        mpatches.Patch(facecolor=C_ORANGE,edgecolor=C_ORANGE,label="Reward weight $(1-r_t)$"),
        mpatches.Patch(facecolor=C_RED,   edgecolor=C_RED,   label="Training loss"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=8.5,
              framealpha=0.9, ncol=2)

    plt.tight_layout(pad=0.3)
    path = os.path.join(OUT_DIR, "fig3_opsd.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    plt.close()
    print(f"saved {path}")


if __name__ == "__main__":
    draw_overall_arch()
    draw_mamba_translator()
    draw_opsd()
    print("All figures saved.")
