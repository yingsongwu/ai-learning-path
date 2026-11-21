# -*- coding: utf-8 -*-
"""
第 3 课：矩阵乘法与形状 —— 90% 的深度学习报错都出在这

一句话：(n, k) @ (k, m) → (n, m)，中间的 k 必须相等，并且会被"消掉"。

记法：把形状写成两边，中间对齐才能乘：
        (n, k) · (k, m) → (n, m)
              ↑ 这两个必须一样
读法："n 个样本，每个 k 维；用 (k,m) 的矩阵换成 m 维；变成 n 个 m 维。"

运行：python 03_shapes.py
"""
import numpy as np
from _plot import plt, save, BLUE, GREEN, RED, ORANGE, GRAY, INK

np.set_printoptions(precision=2, suppress=True)

print("=" * 60)
print("1) 手算一个 2×3 乘 3×2，看清「行乘列」")
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[7, 8], [9, 10], [11, 12]])
print("  A (2,3) =\n", A)
print("  B (3,2) =\n", B)
print("  A@B (2,2) =\n", A @ B)
print("  第一个元素 = A 第 1 行 · B 第 1 列 = 1*7+2*9+3*11 =", 1*7 + 2*9 + 3*11)

print("\n" + "=" * 60)
print("2) 形状规则：(n,k) @ (k,m) → (n,m)")
print("  (2,3) @ (3,2) = (2,2)   ✅")
try:
    np.ones((2, 3)) @ np.ones((2, 3))
except ValueError as e:
    print("  (2,3) @ (2,3) ✅? 不 →", str(e)[:60])

print("\n" + "=" * 60)
print("3) 神经网络里的形状链（本路线 05 单元会用到）")
table = [
    ("输入 x",            "(batch, 256)"),
    ("× W1 (256,64) + b1", "(batch, 64)"),
    ("ReLU",              "(batch, 64)  形状不变"),
    ("× W2 (64,10) + b2",  "(batch, 10)"),
    ("softmax",           "(batch, 10)  每行和为 1"),
]
for step, shape in table:
    print(f"   {step:22s} → {shape}")

batch = 8
x = np.random.rand(batch, 256)
W1 = np.random.randn(256, 64) * np.sqrt(2/256); b1 = np.zeros(64)
W2 = np.random.randn(64, 10) * 0.2;            b2 = np.zeros(10)
h = np.maximum(x @ W1 + b1, 0)
z = h @ W2 + b2
print(f"\n   实测：x{x.shape} → h{h.shape} → z{z.shape}   （batch=8）")

print("\n" + "=" * 60)
print("4) 三个最常犯的形状错误")
print("   ① 忘了 batch 维：用 (256,) 而不是 (1,256) → 结果变成一维，和权重相乘报错")
print("   ② 转置弄反：(n,k)@(n,m) 不合法，应为 (n,k)@(k,m)")
print("   ③ 广播悄悄生效：bias 加法要注意是不是加在了错误的位置")

try:
    np.zeros(256) @ np.zeros((256, 10))
    print("\n   （一维 @ 二维 在 numpy 里其实合法，结果是一维 —— 这正是坑）")
except ValueError as e:
    print("  ", e)

# ---------------- 图：形状流 ----------------
fig, ax = plt.subplots(figsize=(11, 2.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 2.6); ax.axis("off")
boxes = [(0.2, "(8, 256)\nx", BLUE), (2.4, "×W1(256,64)\n+b1", ORANGE),
         (4.8, "(8, 64)\nh=ReLU", GREEN), (7.0, "×W2(64,10)\n+b2", ORANGE),
         (9.2, "(8, 10)\n概率", BLUE)]
for x0, t, c in boxes:
    ax.add_patch(plt.Rectangle((x0, 0.85), 1.5, 0.85, fill=False, ec=c, lw=1.6, zorder=3))
    ax.text(x0 + 0.75, 1.28, t, ha="center", va="center", fontsize=9, color=INK, zorder=4)
for i in range(len(boxes) - 1):
    x1, x2 = boxes[i][0] + 1.5, boxes[i + 1][0]
    ax.annotate("", xy=(x2 - 0.05, 1.28), xytext=(x1 + 0.05, 1.28),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))
ax.text(5.6, 0.35, "中间的 256 和 64 都被「消掉」，只留下 batch 和输出维度",
        ha="center", fontsize=9.5, color=GRAY)
save(fig, "03_shapes.png")

print("\n" + "=" * 60)
print("过关标准：看到任何一层，能立刻说出输出的形状。")
