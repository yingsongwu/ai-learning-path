# -*- coding: utf-8 -*-
"""
第 3 课：偏导与梯度 —— 多变量时「往哪走最陡」

偏导：只对一个变量求导，其他当常数。
     f(x,y) = x² + 3y²
     ∂f/∂x = 2x        （把 y 当常数）
     ∂f/∂y = 6y

梯度：把所有偏导排成向量 ∇f = (∂f/∂x, ∂f/∂y)
     它指向「上升最快的方向」，它的长度是「有多陡」

神经网络里参数有上万个，梯度就是上万维的「最陡方向」。

运行：python 03_partial_gradient.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GRAY, INK


def f(x, y):
    return x ** 2 + 3 * y ** 2


x0, y0 = 1.0, 2.0
print("=" * 62)
print("f(x, y) = x² + 3y²，在点 (1, 2) 处\n")
print("1) 偏导（把另一个变量当常数）")
print("   ∂f/∂x = 2x        → 在 (1,2) 处 = 2×1 = 2")
print("   ∂f/∂y = 6y        → 在 (1,2) 处 = 6×2 = 12")
print(f"\n2) 梯度 ∇f = (2, 12)，当前函数值 f = {f(x0, y0):.1f}")

print("\n3) 用数值验证偏导（沿 x 走一小步）")
for h in [0.1, 0.01, 0.001]:
    approx = (f(x0 + h, y0) - f(x0 - h, y0)) / (2 * h)
    print(f"   h={h:<6} 中心差分 = {approx:.6f}   （解析值 2）")

print("\n4) 梯度告诉我们什么")
print("   · 方向：(2,12) 指向「上升最快」的方向 —— 所以下降要走反方向 (-2,-12)")
print("   · 大小：‖∇f‖ = √(4+144) ≈ 12.2，说明 y 方向更陡（因为系数 3 更大）")

print("\n5) 一个关键直觉")
print("   如果你只想动 x，最优步长比是 2 : 12 —— 陡的方向应该少走一点。")
print("   这就是为什么优化器要给每个参数单独的学习率（Adam 干的事）。")

# ---------------- 图 ----------------
gx = np.linspace(-3, 3, 300)
gy = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(gx, gy)
Z = f(X, Y)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
cs = ax.contour(X, Y, Z, levels=10, cmap="Blues", linewidths=1)
ax.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
for px, py in [(1, 2), (-1, 1.5), (1.5, -1)]:
    g = np.array([2 * px, 6 * py])
    g = g / np.linalg.norm(g) * 0.7
    ax.annotate("", xy=(px + g[0], py + g[1]), xytext=(px, py),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
    ax.plot(px, py, "o", color=INK, ms=5)
ax.set_xlabel("x", fontsize=10, color=INK); ax.set_ylabel("y", fontsize=10, color=INK)
ax.set_title("梯度箭头：指向上升最陡的方向（垂直于等高线）", fontsize=10, color=INK)
ax.set_aspect("equal")

ax = axes[1]
ax.axis("off")
ax.set_xlim(0, 10); ax.set_ylim(0, 5)
ax.text(5, 4.4, "两个方向的陡峭程度不一样", ha="center", fontsize=11, color=INK)
ax.annotate("", xy=(7.2, 2.4), xytext=(3.0, 2.4), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.4))
ax.text(5.1, 2.7, "x 方向（斜率 2）", ha="center", fontsize=9.5, color=BLUE)
ax.annotate("", xy=(3.6, 3.6), xytext=(3.6, 1.6), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.4))
ax.text(2.4, 2.6, "y 方向\n（斜率 12）", ha="center", fontsize=9.5, color=ORANGE, linespacing=1.5)
ax.text(5, 0.8, "同样走一步，y 方向变化大 6 倍 → 应该走更小的步", ha="center",
        fontsize=9.5, color=GRAY)
save(fig, "03_gradient.png")

print("\n" + "=" * 62)
print("记住：偏导 = 只动一个变量；梯度 = 所有偏导排成向量 = 最陡上升方向。")
