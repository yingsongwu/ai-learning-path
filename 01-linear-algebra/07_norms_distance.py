# -*- coding: utf-8 -*-
"""
第 7 课（进阶）：范数与距离 —— "有多远"有三种量法

三个词：
    范数：一个向量的"长度"（衡量单个向量的大小）
    距离：两个向量的"差异"（|a-b| 的范数）
    相似度：越像分越高（余弦最常用）

三种最常用的范数：
    L2（欧氏）：sqrt(Σx²)          → 单位球是圆的，最常用
    L1（曼哈顿）：Σ|x|              → 单位球是菱形，会让参数变稀疏
    L∞（切比雪夫）：max|x|          → 单位球是正方形，只看最大分量

为什么 AI 里要区分它们？
    · L2 正则 → 参数整体变小、平滑
    · L1 正则 → 很多参数恰好变成 0（特征选择、剪枝）
    · 余弦相似度 → 只看方向不看长度（embedding 检索的标准做法）

运行：python 07_norms_distance.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GREEN, GRAY, INK

np.set_printoptions(precision=3, suppress=True)
a = np.array([3.0, 4.0])
b = np.array([1.0, 1.0])

print("=" * 58)
print("a =", a, " b =", b)
print(f"\n1) a 的三种长度：L2={np.linalg.norm(a,2):.3f}  L1={np.linalg.norm(a,1):.3f}  L∞={np.linalg.norm(a,np.inf):.3f}")
d = a - b
print(f"\n2) a 与 b 的距离（d=a-b={d}）：L2={np.linalg.norm(d,2):.3f}  L1={np.linalg.norm(d,1):.3f}  L∞={np.linalg.norm(d,np.inf):.3f}")
print("   同两个点，换一种范数，「距离」就变了 —— 选哪种取决于你在意什么")

print("\n3) 方向 vs 长度：余弦相似度只看夹角")
u = np.array([2.0, 0.5])
v = np.array([4.0, 1.0])          # v = 2u，方向完全一样
cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
print(f"   u={u}  v={v}（v 是 u 的两倍长）")
print(f"   L2 距离 = {np.linalg.norm(u-v):.3f}（差很多）   余弦 = {cos:.3f}（完全相同）")
print("   → 检索 embedding 时用余弦，就不会被'长度'干扰")

print("\n4) 在 AI 里的位置")
for line in [
    "L2 正则：loss += λ·||W||²  → 权重整体收缩，输出更平滑",
    "L1 正则：loss += λ·||W||₁  → 一堆权重变 0，天然做特征选择",
    "梯度裁剪：把梯度向量的 L2 范数限制在阈值内，防爆炸",
    "余弦相似度：向量检索、对比学习、推荐召回",
]:
    print("   ·", line)

# ---------------- 图 ----------------
g = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(g, g)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))

levels = [1.0]
data = [("L2：圆", np.abs(X)**2 + np.abs(Y)**2, BLUE),
        ("L1：菱形", np.abs(X) + np.abs(Y), ORANGE),
        ("L∞：正方形", np.maximum(np.abs(X), np.abs(Y)), GREEN)]
for ax, (title, Z, c) in zip(axes[:3], data):
    ax.contour(X, Y, Z, levels=levels, colors=[c], linewidths=2)
    ax.axhline(0, color="#eef3f8"); ax.axvline(0, color="#eef3f8")
    ax.set_aspect("equal"); ax.set_title(f"{title}（单位球）", fontsize=10.5, color=INK)
    ax.tick_params(labelsize=8, colors=GRAY)

ax = axes[3]
ax.annotate("", xy=a, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
ax.annotate("", xy=b, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
ax.plot([a[0], b[0]], [a[1], b[1]], "o--", color=GRAY, lw=1.2, ms=4)
ax.text(0.15, 3.35, f"L2 距离 {np.linalg.norm(a-b):.2f}\n余弦 {float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))):.2f}",
        fontsize=9, color=INK, linespacing=1.5)
ax.set_xlim(-0.5, 3.8); ax.set_ylim(-0.5, 4.5); ax.set_aspect("equal")
ax.set_title("距离 vs 夹角", fontsize=10.5, color=INK)
ax.tick_params(labelsize=8, colors=GRAY)

fig.suptitle("同样「长度为 1」的集合，在三种范数下形状不同：这就是 L1/L2 正则效果不同的根源",
             fontsize=11, color=INK)
save(fig, "07_norms.png")

print("\n" + "=" * 58)
print("记住：L2 看整体大小，L1 偏爱稀疏，余弦只看方向。")
