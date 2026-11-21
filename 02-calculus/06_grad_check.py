# -*- coding: utf-8 -*-
"""
第 6 课：数值梯度校验 —— 判断「导数算得对不对」的万能方法

思路：不用管公式推得对不对，直接用定义近似：
    f'(x) ≈ [f(x+h) − f(x−h)] / (2h)      ← 中心差分，比单侧更准

如果解析导数和数值导数不一致（相对误差 > 1e-4），公式就写错了。

h 怎么选：
    h 太大 → 截断误差（用直线代替曲线）
    h 太小 → 浮点误差（两个相近的数相减）
    通常 1e-5 ~ 1e-6 最好，本课会用图把这条 U 形曲线画出来。

运行：python 06_grad_check.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GREEN, GRAY, INK


def f(x):
    return x ** 3 + 2 * x


def analytic(x):
    return 3 * x ** 2 + 2


def numeric(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


x0 = 1.5
true = analytic(x0)
print("=" * 64)
print(f"f(x) = x³ + 2x，在 x = {x0} 处")
print(f"解析导数 f'(x) = 3x² + 2 = {true:.6f}\n")
print("  h          数值导数        相对误差")
print("  " + "-" * 48)
for h in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    n = numeric(x0, h)
    print(f"  {h:<10.0e} {n:>12.8f}   {abs(n-true)/abs(true):>10.2e}")
print("\n  → 1e-4 ~ 1e-6 误差最小；再小反而变差（浮点精度不够）")

print("\n" + "=" * 64)
print("用在神经网络上：逐元素比较 dW")
rng = np.random.default_rng(0)
W = rng.standard_normal((2, 3))
X = rng.standard_normal((4, 2))


def loss_of(W):
    H = np.maximum(X @ W, 0)
    return float((H ** 2).sum())


def dW_analytic(W):
    H = np.maximum(X @ W, 0)
    dH = 2 * H
    dH[(X @ W) <= 0] = 0
    return X.T @ dH


dA = dW_analytic(W)
dN = np.zeros_like(W)
eps = 1e-6
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        old = W[i, j]
        W[i, j] = old + eps
        lp = loss_of(W)
        W[i, j] = old - eps
        lm = loss_of(W)
        W[i, j] = old
        dN[i, j] = (lp - lm) / (2 * eps)

diff = np.abs(dA - dN).max()
scale = np.abs(dN).max()
print(f"  解析梯度范围：{np.round(dA, 3).tolist()}")
print(f"  最大绝对偏差：{diff:.3e}     梯度最大幅值：{scale:.3f}")
print(f"  相对偏差：{diff/scale:.2e}  →  {'通过 ✅' if diff/scale < 1e-4 else '有问题 ❌'}")
print("\n  这条规则记住：写任何新的层/损失，先跑一次数值校验。")

# ---------------- 图 ----------------
hs = np.logspace(-10, 0, 45)
errs = [abs(numeric(x0, h) - true) / abs(true) for h in hs]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.plot(hs, errs, "o-", color=BLUE, ms=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvspan(1e-6, 1e-4, color="#eef6ff", zorder=0)
ax.text(3e-5, max(errs) * 0.3, "最佳区间\n1e-6 ~ 1e-4", fontsize=9, color=GRAY, ha="center")
ax.set_xlabel("h（对数轴）", fontsize=10, color=INK)
ax.set_ylabel("相对误差", fontsize=10, color=INK)
ax.set_title("h 太大或太小都不好（U 形）", fontsize=10.5, color=INK)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 5)
ax.text(5, 4.5, "校验三行代码", ha="center", fontsize=11, color=INK)
code = ("w[i] += eps → loss1\nw[i] -= 2eps → loss2\n"
        "num = (loss1 - loss2) / (2eps)\n"
        "assert |num - analytic| / |analytic| < 1e-4")
ax.text(0.4, 2.2, code, fontsize=10, color=INK, family="monospace", va="center", linespacing=1.9)
ax.text(5, 0.7, "写完新层先跑一遍，比 debug 一小时快", ha="center", fontsize=9.5, color=GRAY)
save(fig, "06_gradcheck.png")
