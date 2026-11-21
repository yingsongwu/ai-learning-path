# -*- coding: utf-8 -*-
"""
第 1 课：导数是什么 —— 从「割线」到「切线」

一句话：导数是「这一点上的瞬时变化率」，也就是切线斜率。
怎么算出来：先算两点间的平均变化率（割线斜率），再让两点无限靠近。

    f'(x) = lim_{h→0} [f(x+h) − f(x)] / h

这一课用 f(x) = x² 在 x = 3 处把 h 从 1 缩到 0.0001，
亲眼看差商怎么逼近真实导数 6。

运行：python 01_derivative.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GRAY, INK


def f(x):
    return x ** 2


def diff_quotient(x, h):
    """差商 = 平均变化率 = 割线斜率"""
    return (f(x + h) - f(x)) / h


x0 = 3.0
print("=" * 62)
print("f(x) = x²，求 x = 3 处的导数\n")
print("  h          差商 [f(3+h)−f(3)]/h     与 6 的差距")
print("  " + "-" * 52)
for h in [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]:
    q = diff_quotient(x0, h)
    print(f"  {h:<8} {q:>16.6f}      {abs(q - 6):>10.6f}")
print("\n  h 越小，差商越接近 6 → 极限就是导数 f'(3) = 2x|₃ = 6")
print("  几何上：割线（两点连线）在 h→0 时变成切线")

print("\n为什么导数这么重要：")
print("  · 它告诉我们「参数往哪边调、调多少」—— 这就是反向传播要的东西")
print("  · 损失函数对参数的导数 = 参数增加一点点，损失会变化多少")
print("  · 有了方向，就能沿着反方向走（第 7 课的梯度下降）")

# ---------------- 图 ----------------
xs = np.linspace(0.2, 6, 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
ax.plot(xs, f(xs), color=INK, lw=2, label="f(x) = x²")
for h, c in [(2.0, GRAY), (1.0, ORANGE), (0.4, RED)]:
    x1, y1 = x0, f(x0)
    x2, y2 = x0 + h, f(x0 + h)
    ax.plot([x1, x2], [y1, y2], "--o", color=c, lw=1.4, ms=4,
            label=f"割线 h={h}（斜率 {diff_quotient(x0, h):.2f}）")
# 切线
ax.plot(xs, f(x0) + 6 * (xs - x0), color=BLUE, lw=2.4, label="切线（斜率 6）")
ax.set_xlim(1.5, 5.2); ax.set_ylim(0, 30)
ax.set_xlabel("x", fontsize=10, color=INK); ax.set_ylabel("f(x)", fontsize=10, color=INK)
ax.set_title("割线越短，越贴近切线", fontsize=11, color=INK)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
hs = np.array([1, 0.5, 0.1, 0.01, 0.001, 0.0001])
qs = np.array([diff_quotient(x0, h) for h in hs])
ax.plot(hs, qs, "o-", color=BLUE, ms=5)
ax.axhline(6, color=RED, ls="--", lw=1.2, label="真实导数 6")
ax.set_xscale("log")
ax.set_xlabel("h（对数轴）", fontsize=10, color=INK)
ax.set_ylabel("差商", fontsize=10, color=INK)
ax.set_title("h 越小，差商越接近 6", fontsize=11, color=INK)
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "01_derivative.png")

print("\n" + "=" * 62)
print("记住：导数 = 切线斜率 = 瞬时变化率 = 差商在 h→0 时的极限。")
