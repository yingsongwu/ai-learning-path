# -*- coding: utf-8 -*-
"""
第 4 课：线性组合、张成、秩 —— 为什么"降维"是有损的

三个词一次说清：
    线性组合：用几个向量各乘一个系数再加起来，c1·v1 + c2·v2 + …
    张成（span）：所有可能的线性组合能覆盖的空间
    秩（rank）：真正"独立"的方向有几个

为什么重要：
    · 秩不足 = 信息冗余 = 可以用更少维度表示（这就是 PCA 和压缩的根基）
    · 神经网络的瓶颈层（把 256 压到 32）本质是"主动降秩"
    · 注意力矩阵 低秩 时说明它在"平均化"，失去了分辨能力

运行：python 04_rank_span.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, GREEN, GRAY, INK

np.set_printoptions(precision=2, suppress=True)

v1 = np.array([1.0, 0.5])
v2 = np.array([2.0, 1.0])       # v2 = 2 * v1，同方向 → 冗余
v3 = np.array([0.0, 1.0])       # 新方向

print("=" * 58)
print("v1 =", v1, "  v2 =", v2, "  v3 =", v3)
print("\n1) 判断是否独立：能不能用别人线性组合出来？")
print("   v2 = 2 * v1 ?", np.allclose(v2, 2 * v1), " ← 能，所以 v2 不带来新信息")

M12 = np.stack([v1, v2])
M13 = np.stack([v1, v3])
print("\n2) 秩 = 独立方向数")
print("   矩阵 [v1; v2] 的秩 =", np.linalg.matrix_rank(M12), "（两个向量挤在一条线上）")
print("   矩阵 [v1; v3] 的秩 =", np.linalg.matrix_rank(M13), "（真的撑开一个平面）")

print("\n3) 行列式为 0 ⟺ 线性相关（空间被压扁）")
print("   det[v1;v2] =", round(float(np.linalg.det(M12)), 6), "  det[v1;v3] =",
      round(float(np.linalg.det(M13)), 6))

print("\n4) 张成：v1、v2 只能覆盖一条直线；加上 v3 才能覆盖整个平面")
print("   任意方向都能表示吗？试 (5, -2)：")
A = np.stack([v1, v3]).T
coef = np.linalg.solve(A, np.array([5.0, -2.0]))
print("   解 c1, c2 =", coef, " 验证：", coef[0] * v1 + coef[1] * v3)

print("\n5) 秩亏的后果（换个说法）：")
X = np.random.randn(100, 2) @ M12          # 100 个点，全在一条线上
print("   100 个二维点，实际只用了", np.linalg.matrix_rank(X), "个方向")
print("   → 压缩成 1 维也不丢信息；反之，秩为 2 的数据压到 1 维必然有损")

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
for v, name, c in [(v1, "v1", BLUE), (v2, "v2 (=2·v1)", RED), (v3, "v3", GREEN)]:
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax.text(v[0] * 1.06 + 0.05, v[1] * 1.06, name, color=c, fontsize=10)
xx = np.linspace(-0.6, 2.6, 2)
ax.plot(xx, xx * 0.5, "--", color=GRAY, lw=1)
ax.text(1.7, 0.55, "v1、v2 张成的只有这条线（秩=1）", fontsize=9, color=GRAY)
ax.set_xlim(-0.6, 2.8); ax.set_ylim(-0.4, 1.8); ax.set_aspect("equal")
ax.axhline(0, color="#dfe7ef", lw=1); ax.axvline(0, color="#dfe7ef", lw=1)
ax.set_title("秩 = 1：两条箭头重叠", fontsize=11, color=INK)

ax = axes[1]
for v, name, c in [(v1, "v1", BLUE), (v3, "v3", GREEN)]:
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
    ax.text(v[0] * 1.06 + 0.05, v[1] * 1.06, name, color=c, fontsize=10)
ax.fill([0, v1[0], v1[0] + v3[0], v3[0]], [0, v1[1], v1[1] + v3[1], v3[1]],
        color="#eef3f8", zorder=0)
ax.text(0.75, 0.95, "张成整个平面\n（秩 = 2）", fontsize=10, color=GRAY)
ax.set_xlim(-0.6, 2.4); ax.set_ylim(-0.4, 1.9); ax.set_aspect("equal")
ax.axhline(0, color="#dfe7ef", lw=1); ax.axvline(0, color="#dfe7ef", lw=1)
ax.set_title("秩 = 2：真正撑开平面", fontsize=11, color=INK)

save(fig, "04_rank_span.png")

print("\n" + "=" * 58)
print("一句话记住：秩 = 有效维度数。降维就是「故意让秩变小」，前提是信息本来冗余。")
