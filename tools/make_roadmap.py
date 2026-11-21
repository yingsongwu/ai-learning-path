# -*- coding: utf-8 -*-
"""生成 00-roadmap/figures/roadmap.png：学习单元依赖图。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "00-roadmap", "figures")
os.makedirs(OUT, exist_ok=True)

for p in ["/system/fonts/HarmonyOS_Sans_SC.ttf", "C:/Windows/Fonts/msyh.ttc",
          "/System/Library/Fonts/PingFang.ttc"]:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
        matplotlib.rcParams["font.sans-serif"] = [fm.FontProperties(fname=p).get_name()]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

INK, BLUE, GREEN, ORANGE, GRAY, RED = "#22303f", "#2471a3", "#1e8449", "#b9770e", "#8b98a5", "#c0392b"

def box(ax, x, y, w, h, text, color, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc="white", ec=color, lw=1.8, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4, linespacing=1.4)

def arrow(ax, p1, p2, color=GRAY):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                 lw=1.4, color=color, zorder=2))

fig, ax = plt.subplots(figsize=(11, 6.4))
ax.set_xlim(0, 11); ax.set_ylim(0, 6.4); ax.axis("off")

# 第一列：地基
box(ax, 0.3, 4.9, 2.0, 0.95, "01 线性代数\n矩阵·形状·相似度", BLUE)
box(ax, 0.3, 3.4, 2.0, 0.95, "02 微积分\n梯度·链式法则", BLUE)
box(ax, 0.3, 1.9, 2.0, 0.95, "03 概率统计\n交叉熵·softmax", BLUE)
box(ax, 0.3, 0.45, 2.0, 0.95, "04 Python/numpy\n向量化·axis", GREEN)

# 中列
box(ax, 3.1, 3.4, 2.1, 1.0, "05 神经网络\n前向·反向·训练", ORANGE, fs=10.5)
box(ax, 3.1, 1.9, 2.1, 0.95, "06 训练与调优\n正则·学习率·评估", ORANGE)
box(ax, 3.1, 0.45, 2.1, 0.95, "参考工程\nai-from-scratch（完整版）", GRAY, fs=9)

# 右列：方向
box(ax, 6.0, 4.5, 2.2, 0.95, "07 CNN\n卷积·感受野·残差", "#7d3c98")
box(ax, 6.0, 3.15, 2.2, 0.95, "08 Transformer\n注意力·位置编码", "#7d3c98")
box(ax, 6.0, 1.8, 2.2, 0.95, "09 大模型应用\nRAG·Agent·LoRA", "#7d3c98")
box(ax, 6.0, 0.45, 2.2, 0.95, "10 部署\nONNX·量化·服务", RED)

# 汇合
box(ax, 8.75, 2.6, 2.0, 1.0, "能训练\n能做产品\n能上线", INK, fs=10)

# 箭头
for y1, y2 in [(5.35, 4.35), (3.85, 3.85)]:
    pass
arrow(ax, (2.35, 5.2), (3.05, 4.1))          # 01 → 05
arrow(ax, (2.35, 3.75), (3.05, 3.95))        # 02 → 05
arrow(ax, (2.35, 2.35), (3.05, 3.55))        # 03 → 05
arrow(ax, (2.35, 0.95), (3.05, 3.3), GREEN)  # 04 → 05
arrow(ax, (4.15, 3.35), (4.15, 2.9))         # 05 → 06
arrow(ax, (5.25, 3.9), (5.95, 4.9))          # 05 → 07
arrow(ax, (5.25, 3.75), (5.95, 3.6))         # 05 → 08
arrow(ax, (5.25, 2.3), (5.95, 2.25))         # 06 → 09
arrow(ax, (8.25, 4.6), (8.75, 3.4))          # 07 → 汇合
arrow(ax, (8.25, 3.3), (8.75, 3.2))          # 08 → 汇合
arrow(ax, (8.25, 2.2), (8.75, 3.0))          # 09 → 汇合
arrow(ax, (8.25, 0.95), (8.75, 2.7), RED)    # 10 → 汇合
ax.add_patch(FancyArrowPatch((4.15, 3.3), (4.15, 1.45), arrowstyle="-|>", mutation_scale=12, lw=1.2, color=GRAY, linestyle=":", zorder=2))

ax.text(5.5, 6.1, "AI 学习路线：左边是地基，中间是模型，右边是方向与落地",
        ha="center", fontsize=12.5, color=INK)
legend = [("地基 1~4", BLUE), ("模型 5~6", ORANGE), ("方向 7~9", "#7d3c98"), ("落地 10", RED)]
x = 0.3
for text, c in legend:
    ax.add_patch(plt.Rectangle((x, 0.12), 0.18, 0.18, fc=c, ec="none"))
    ax.text(x + 0.26, 0.22, text, fontsize=8.5, color=GRAY, va="center")
    x += 1.55
fig.savefig(os.path.join(OUT, "roadmap.png"), dpi=150, bbox_inches="tight", facecolor="white")
print("saved 00-roadmap/figures/roadmap.png")
