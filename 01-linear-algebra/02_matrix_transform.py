# -*- coding: utf-8 -*-
"""
第 2 课：矩阵 = 对空间的一次线性变换

一句话：矩阵不是"数字表格"，它是"把空间拉伸/旋转/剪切"的动作。

关键直觉（务必记住）：
    矩阵的每一列 = 原来那根坐标轴被搬到了哪里
        单位矩阵 [[1,0],[0,1]] 的列是 (1,0) 和 (0,1)：空间不动
        旋转矩阵的列是转过去的两根轴
    所以 A @ x 的意思是："按 A 指定的方式，把整个空间搬一下，然后看 x 搬到哪"

神经网络里的一层 y = x @ W，本质就是：
    用一次线性变换，把 256 维输入搬到 64 维空间里（顺便重新组合信息）

运行：python 02_matrix_transform.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, INK

np.set_printoptions(precision=3, suppress=True)

# 单位网格
g = np.linspace(-1.5, 1.5, 9)
X, Y = np.meshgrid(g, g)
pts = np.stack([X.ravel(), Y.ravel()], axis=1)          # (81, 2)

mats = {
    "旋转 30°": np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                          [np.sin(np.pi/6),  np.cos(np.pi/6)]]),
    "缩放 (2, 0.4)": np.array([[2.0, 0.0], [0.0, 0.4]]),
    "剪切 (k=1)": np.array([[1.0, 1.0], [0.0, 1.0]]),
}

print("=" * 58)
print("1) 看矩阵的列：它们就是坐标轴被搬去的位置")
for name, A in mats.items():
    print(f"\n  {name}:")
    print("    A =\n", np.array2string(A, prefix="       "))
    print("    第 1 列（x 轴去向）:", A[:, 0], "  第 2 列（y 轴去向）:", A[:, 1])

print("\n2) 矩阵乘向量 = 变换一个点")
A = mats["剪切 (k=1)"]
p0 = np.array([1.0, 1.0])
print(f"   原坐标 {p0} → 剪切后 {A @ p0}   （x 增加了 1*y）")

print("\n3) 复合变换 = 矩阵相乘（顺序不能反！）")
R, S = mats["旋转 30°"], mats["缩放 (2, 0.4)"]
print("   先旋转再缩放，等价于矩阵", np.array2string(R @ S, prefix="   "))
print("   先缩放再旋转，结果不同 →", not np.allclose(R @ S, S @ R))

print("\n4) 面积缩放倍数 = |det(A)|")
for name, A in mats.items():
    print(f"   {name:14s} det={np.linalg.det(A):+.3f}  面积倍数={abs(np.linalg.det(A)):.3f}")

# ---------------- 图：网格变形 ----------------
fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
titles = ["原空间（单位格子）"] + list(mats.keys())
for ax, (title, M) in zip(axes, [(titles[0], np.eye(2))] + [(k, v) for k, v in mats.items()]):
    moved = pts @ M.T
    for i in range(9):
        row = moved[i * 9:(i + 1) * 9]
        ax.plot(row[:, 0], row[:, 1], color="#c9d6e2", lw=0.8)
        col = moved[i::9]
        ax.plot(col[:, 0], col[:, 1], color="#c9d6e2", lw=0.8)
    # 两根基向量
    ax.annotate("", xy=M[:, 0], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
    ax.annotate("", xy=M[:, 1], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.axhline(0, color="#eef3f8", lw=0.8); ax.axvline(0, color="#eef3f8", lw=0.8)
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal")
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.tick_params(labelsize=7, colors="#8b98a5")

fig.suptitle("同一个格子，被三种矩阵「搬」成三种样子（蓝=x轴去向，红=y轴去向）", fontsize=11.5, color=INK)
save(fig, "02_matrix_transform.png")

print("\n" + "=" * 58)
print("记住这张图：所谓「学习」，就是让网络自己找到一组好用的矩阵。")
