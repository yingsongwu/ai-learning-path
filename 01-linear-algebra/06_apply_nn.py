# -*- coding: utf-8 -*-
"""
第 6 课：把线性代数用回神经网络（衔接 05 单元）

三件事一次串起来：
    1. 一个神经元 = 一次点积 + 一个偏置        → 向量层面
    2. 一层网络   = 一次矩阵乘法 + 偏置        → 矩阵层面
    3. "学习"     = 找一组让相似度算得准的 W

下面用真实数字演示：W 的每一列是一个"模板"，输入和谁最像，谁就被激活。

运行：python 06_apply_nn.py
"""
import numpy as np
from _plot import plt, save, BLUE, GREEN, RED, ORANGE, GRAY, INK

np.set_printoptions(precision=2, suppress=True)
rng = np.random.default_rng(0)

n_in, n_hidden = 8, 5
W = rng.standard_normal((n_in, n_hidden))
b = np.zeros(n_hidden)

x = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0])

print("=" * 60)
print("1) 一个神经元 = 点积 + 偏置")
w0 = W[:, 0]
print("   第 1 个神经元的权重 w =", w0)
print("   输入 x               =", x)
print("   w·x =", round(float(w0 @ x), 4), "  这就是它没加激活函数的输出")

print("\n2) 一层 = 一次矩阵乘法（5 个神经元同时打分）")
z = W.T @ x + b          # 等价于 x @ W
print("   所有神经元的输出 z =", z)

print("\n3) ReLU 之后：负的被砍掉，正的是「这个模板匹配上了」")
h = np.maximum(z, 0)
print("   h =", h)
print("   被激活的神经元编号:", np.where(h > 0)[0].tolist())

print("\n4) 为什么说「权重就是模板」？看相似度矩阵")
# 造 4 个输入，每个都"像"某个模板
X = np.stack([w0 + 0.6 * rng.standard_normal(n_in),
              W[:, 2] + 0.6 * rng.standard_normal(n_in),
              rng.standard_normal(n_in),
              w0 * -1 + 0.6 * rng.standard_normal(n_in)])
sim = X @ W                      # (4, 5) 的相似度分数
print("   X（4 个输入） × W（5 个模板） → 相似度矩阵：")
print(sim)
print("   每一行里最大的那个位置，就是「这个输入最像哪个神经元」：")
print("   ", sim.argmax(axis=1).tolist())

print("\n5) 和 05 单元的关系：那里的 W1(256,64) 就是 64 个这样的模板；")
print("   所谓训练，就是把这 64 列从随机噪声调成「有意义的图案」。")

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})

ax = axes[0]
im = ax.imshow(W, cmap="RdBu_r", aspect="auto", vmin=-abs(W).max(), vmax=abs(W).max())
ax.set_xlabel("神经元编号（5 个模板）", fontsize=9, color=GRAY)
ax.set_ylabel("输入维度", fontsize=9, color=GRAY)
ax.set_title("W 的每一列 = 一个模板", fontsize=10.5, color=INK)
fig.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1]
im = ax.imshow(sim, cmap="Blues", aspect="auto")
for i in range(sim.shape[0]):
    for j in range(sim.shape[1]):
        ax.text(j, i, f"{sim[i, j]:.1f}", ha="center", va="center", fontsize=8,
                color="white" if sim[i, j] > sim.max() * 0.6 else INK)
ax.set_xlabel("神经元编号", fontsize=9, color=GRAY)
ax.set_ylabel("输入样本", fontsize=9, color=GRAY)
ax.set_title("相似度矩阵：越亮说明越「像」这个模板", fontsize=10.5, color=INK)
fig.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("线性代数在神经网络里的全部用途，就这一张图：模板矩阵 + 相似度打分", fontsize=11, color=INK)
save(fig, "06_apply_nn.png")
