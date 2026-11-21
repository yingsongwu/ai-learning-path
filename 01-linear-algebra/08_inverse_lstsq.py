# -*- coding: utf-8 -*-
"""
第 8 课（进阶）：逆矩阵与最小二乘 —— 线性回归的几何真相

两个问题：
    1. 方程 Ax = b 怎么解？    → 方阵且可逆时，x = A⁻¹b
    2. 如果方程比未知数多（超定）、根本无解？ → 找"最接近"的解：最小二乘

最小二乘在干什么：
    把 b 投影到 A 的列空间上（投影 = 第 1 课学的东西），
    让残差 |Ax - b| 的平方和最小。这就是"线性回归"的全部数学。

一句话：拟合直线 = 解一个无解的方程组，取它最接近的解。

运行：python 08_inverse_lstsq.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GRAY, INK

np.set_printoptions(precision=3, suppress=True)

print("=" * 60)
print("1) 逆矩阵：A @ A⁻¹ = I（单位矩阵）")
A = np.array([[4.0, 2.0], [1.0, 3.0]])
Ainv = np.linalg.inv(A)
print("   A =\n", A, "\n   A⁻¹ =\n", Ainv)
print("   A@A⁻¹ =\n", A @ Ainv, "  ← 近似单位矩阵")
print("   解的方程 A x = b，b=(1,2)：x =", np.linalg.solve(A, np.array([1.0, 2.0])))

print("\n2) 不可逆的情况：行列式为 0（秩亏）")
S = np.array([[1.0, 2.0], [2.0, 4.0]])
print("   S =\n", S, "  det =", round(float(np.linalg.det(S)), 10), " 秩 =", np.linalg.matrix_rank(S))
print("   两行成比例 → 空间被压成一条线 → 没有唯一的逆（信息丢了，回不去）")

print("\n3) 超定方程：5 个点拟合一条直线，方程(5)多于未知数(2)")
xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([1.0, 3.0, 2.0, 5.0, 4.0])          # 带噪声
M = np.stack([xs, np.ones_like(xs)], axis=1)       # 设计矩阵 (5,2)：y = a*x + b
print("   设计矩阵 M (5,2) =\n", M)
print("   M 是 5×2，不能求逆 —— 但可以求最小二乘解")
coef, residuals, rank, sv = np.linalg.lstsq(M, ys, rcond=None)
a, b = coef
print(f"   解：a={a:.3f}, b={b:.3f}  →  y = {a:.2f}x + {b:.2f}")
pred = M @ coef
print(f"   残差平方和 = {float(((ys-pred)**2).sum()):.4f}（最小二乘就是让这个数最小）")

print("\n4) 和正规方程的关系")
print("   最小二乘解 = (MᵀM)⁻¹ Mᵀ y  ——  就是 numpy 内部干的事")
normal = np.linalg.inv(M.T @ M) @ M.T @ ys
print("   手算 (MᵀM)⁻¹Mᵀy =", normal, "  与 lstsq 一致:", np.allclose(normal, coef))
print("\n   注意：这正是线性回归；加个 sigmoid 就是逻辑回归；")
print("        再堆几层线性层 + 激活，就是你在 05 单元要写的神经网络。")

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

ax = axes[0]
ax.scatter(xs, ys, s=55, color=BLUE, zorder=3, label="观测点（带噪声）")
xx = np.linspace(-0.3, 4.3, 50)
ax.plot(xx, a * xx + b, color=RED, lw=2, label=f"最小二乘直线 y={a:.2f}x+{b:.2f}")
for xi, yi, pi in zip(xs, ys, pred):
    ax.plot([xi, xi], [yi, pi], "--", color=GRAY, lw=1)
ax.set_xlabel("x", fontsize=10, color=INK); ax.set_ylabel("y", fontsize=10, color=INK)
ax.set_title("最小二乘：让所有虚线的平方和最小", fontsize=10.5, color=INK)
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 4.4); ax.axis("off")
ax.text(5, 4.0, "三种方程状态，三种解法", ha="center", fontsize=11, color=INK)
rows = [("方阵 + 可逆", "唯一解 x = A^-1 b", BLUE),
        ("方阵 + 不可逆", "无解或无穷多解（秩亏）", ORANGE),
        ("行多于列（超定）", "最小二乘：投影到列空间", RED)]
for i, (cond, sol, c) in enumerate(rows):
    y = 3.0 - i * 1.0
    ax.add_patch(plt.Rectangle((0.3, y - 0.3), 9.4, 0.75, fill=False, ec=c, lw=1.4))
    ax.text(0.5, y, cond, fontsize=9.5, color=c, va="center")
    ax.text(3.6, y, "→  " + sol, fontsize=9.5, color=INK, va="center")
ax.text(5, 0.4, "线性回归 = 第三种情况；神经网络 = 第三种情况的非线性版本",
        ha="center", fontsize=9, color=GRAY)
save(fig, "08_lstsq.png")

print("\n" + "=" * 60)
print("结论：所谓「拟合」，就是在无解里找最接近的答案。")
