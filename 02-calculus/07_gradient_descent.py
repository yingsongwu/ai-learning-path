# -*- coding: utf-8 -*-
"""
第 7 课：梯度下降 —— 用导数把损失压下去

更新规则只有一行：
    w ← w − 学习率 × dL/dw

为什么是减：梯度指向上坡，减掉才能下坡。

学习率决定命运：
    太小 → 走不动（loss 掉得极慢）
    合适 → 平稳收敛
    太大 → 震荡甚至发散（loss 越走越大）

运行：python 07_gradient_descent.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GREEN, GRAY, INK


def f(w):
    return w ** 2 + 2 * w + 3


def df(w):
    return 2 * w + 2


def run(lr, start=4.0, steps=25):
    w = start
    hist = [(w, f(w))]
    for _ in range(steps):
        w = w - lr * df(w)
        hist.append((w, f(w)))
    return hist


print("=" * 62)
print("f(w) = w² + 2w + 3，最小值在 w = −1（f=2）")
print("f'(w) = 2w + 2\n")

print("学习率对比（起点 w=4，走 25 步）：")
print("  lr        终点 w      终点 loss      行为")
print("  " + "-" * 52)
for lr in [0.01, 0.1, 0.5, 0.9, 1.1]:
    h = run(lr)
    w_end, l_end = h[-1]
    if not np.isfinite(l_end) or abs(w_end) > 50:
        behavior = "发散 💥"
    elif lr <= 0.01:
        behavior = "太慢（还没到）"
    elif abs(l_end - 2) < 0.05:
        behavior = "收敛 ✅"
    else:
        behavior = "震荡"
    print(f"  {lr:<8} {w_end:>10.3f} {l_end:>14.3f}      {behavior}")

print("\n为什么 lr=1.1 会发散：")
print("  更新式 w ← w − lr·(2w+2)；当 lr > 1，误差会被放大而不是缩小")
print("  数学上：误差每步乘 |1 − 2·lr|，lr=1.1 时 |1−2.2| = 1.2 > 1 → 越走越远")

print("\n取 lr=0.1，逐步看：")
print("  step     w        f(w)     f'(w)")
w = 4.0
for i in range(8):
    print(f"  {i:<5} {w:>7.3f} {f(w):>9.3f} {df(w):>8.3f}")
    w -= 0.1 * df(w)

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
ws = np.linspace(-3, 5, 200)
ax.plot(ws, f(ws), color=INK, lw=2, label="f(w)=w²+2w+3")
for lr, c, lab in [(0.01, GRAY, "lr=0.01 太慢"), (0.1, GREEN, "lr=0.1 收敛"), (1.1, RED, "lr=1.1 发散")]:
    h = run(lr, steps=12)
    pts = np.array([[w, v] for w, v in h])
    pts = pts[np.abs(pts[:, 0]) < 12]      # 裁掉发散到很远的点
    ax.plot(pts[:, 0], pts[:, 1], "o-", ms=3, lw=1, color=c, alpha=.9, label=lab)
ax.set_ylim(0, 30); ax.set_xlim(-3, 5)
ax.set_xlabel("w", fontsize=10, color=INK); ax.set_ylabel("loss", fontsize=10, color=INK)
ax.set_title("在抛物线上往下走", fontsize=10.5, color=INK)
ax.legend(fontsize=8.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
for lr, c in [(0.01, GRAY), (0.1, GREEN), (0.5, ORANGE), (1.1, RED)]:
    h = run(lr, steps=25)
    ys = [v for _, v in h]
    ax.plot(ys, lw=1.8, color=c, label=f"lr={lr}")
ax.set_yscale("log")
ax.set_xlabel("步数", fontsize=10, color=INK); ax.set_ylabel("loss（对数轴）", fontsize=10, color=INK)
ax.set_title("学习率决定收敛速度 / 是否发散", fontsize=10.5, color=INK)
ax.legend(fontsize=8.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "07_gd.png")

print("\n" + "=" * 62)
print("记住：w ← w − lr·f'(w)；lr 太小走不动，太大直接炸。")
