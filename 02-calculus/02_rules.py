# -*- coding: utf-8 -*-
"""
第 2 课：求导规则 —— 不用每次都用极限

三个够用的规则：
    幂法则   (xⁿ)' = n·xⁿ⁻¹
    常数倍   (c·f)' = c·f'
    链式法则 (f(g(x)))' = f'(g(x)) · g'(x)      ← 最重要，神经网络全靠它

链式法则的直觉：像「换挡」—— 外层先对外层变量求导，再乘内层对内层变量的导数。

运行：python 02_rules.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GREEN, GRAY, INK

print("=" * 64)
print("1) 幂法则：(xⁿ)' = n·xⁿ⁻¹")
for n in [2, 3, 5]:
    x = 2.0
    print(f"   f(x)=x^{n}  →  f'(x)={n}·x^{n-1}  →  f'(2)={n}·2^{n-1}={n * x ** (n - 1):.0f}")

print("\n2) 常数倍与加法：(3x²)' = 3·2x = 6x")
print("   x=2 时 = 12")

print("\n3) 链式法则：复合函数要「一层层乘」")
print("   y = (3x + 1)²")
print("   设 u = 3x + 1  →  y = u²")
print("   dy/dx = dy/du × du/dx = 2u × 3 = 6u = 6(3x+1)")
x = 1.0
u = 3 * x + 1
print(f"\n   在 x=1：u = {u}，dy/du = 2u = {2*u}，du/dx = 3")
print(f"   所以 dy/dx = {2*u} × 3 = {2*u*3}")
print("   验证：直接用展开式 y = 9x²+6x+1 → y' = 18x+6 → x=1 时 24 ✓")

print("\n4) 为什么链式法则最关键")
print("   神经网络的 loss 是复合函数：loss = L(y),  y = u(W·x),  W 在最里面")
print("   想求 dL/dW 只能一层层乘回去 —— 这就是第 5 课的反向传播")

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ax = axes[0]
xs = np.linspace(-0.6, 1.6, 300)
ax.plot(xs, (3 * xs + 1) ** 2, color=INK, lw=2, label="y = (3x+1)²")
ax.plot(xs, 9 * xs ** 2 + 6 * xs + 1, "--", color=ORANGE, lw=1.4, label="展开后 9x²+6x+1（同一条线）")
# 切线
tangent = (3 * x + 1) ** 2 + 24 * (xs - x)
ax.plot(xs, tangent, color=BLUE, lw=1.8, label="x=1 处切线（斜率 24）")
ax.plot([x], [(3 * x + 1) ** 2], "o", color=RED, ms=7)
ax.set_xlim(-0.4, 1.4); ax.set_ylim(-2, 20)
ax.set_title("复合函数的切线斜率 = 两层导数相乘", fontsize=10.5, color=INK)
ax.legend(fontsize=8.5, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

ax = axes[1]
ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off")
boxes = [(0.6, 3.6, "x"), (3.4, 3.6, "u = 3x + 1"), (6.8, 3.6, "y = u²")]
for x0, y0, t in boxes:
    ax.add_patch(plt.Rectangle((x0, y0 - 0.35), 2.0, 0.7, fc="white", ec=BLUE, lw=1.5))
    ax.text(x0 + 1.0, y0, t, ha="center", va="center", fontsize=10, color=INK)
for i in range(2):
    x1 = boxes[i][0] + 2.0; x2 = boxes[i + 1][0]
    ax.annotate("", xy=(x2 - 0.05, 3.6), xytext=(x1 + 0.05, 3.6),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))
ax.text(3.0, 3.95, "×3", fontsize=9.5, color=ORANGE, ha="center")
ax.text(6.3, 3.95, "×2u", fontsize=9.5, color=ORANGE, ha="center")
ax.text(5.0, 2.4, "dy/dx = (dy/du) × (du/dx)\n= 2u × 3 = 6u", ha="center",
        fontsize=10.5, color=INK, linespacing=1.8)
ax.text(5.0, 1.1, "链式法则 = 把每一小段的导数乘起来", ha="center", fontsize=9.5, color=GRAY)
ax.set_title("把复合函数拆成链条", fontsize=10.5, color=INK)
save(fig, "02_chain_rule.png")
