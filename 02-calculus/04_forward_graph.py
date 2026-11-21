# -*- coding: utf-8 -*-
"""
第 4 课：计算图 —— 把复杂函数拆成一串小步骤

为什么要拆：
    直接对 loss = (w·x + b − y)² 求导很乱；
    拆成节点之后，每一步都是「加减乘」这种简单运算，导数一眼就能写。

本例：x=2, w=3, b=1, 目标 y=5
    u = w · x      = 6
    p = u + b      = 7
    L = (p − y)²   = 4

每个节点都记下两件事：**它的值** 和 **它的局部导数**（它对自己输入的变化率）。

运行：python 04_forward_graph.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GREEN, GRAY, INK

x, w, b, y = 2.0, 3.0, 1.0, 5.0

print("=" * 62)
print(f"输入：x={x}, w={w}, b={b}, 目标 y={y}\n")

u = w * x
print("① u = w · x")
print(f"   值：u = {w} × {x} = {u}")
print(f"   局部导数：∂u/∂w = x = {x}    ∂u/∂x = w = {w}")

p = u + b
print("\n② p = u + b")
print(f"   值：p = {u} + {b} = {p}")
print("   局部导数：∂p/∂u = 1，∂p/∂b = 1（加法把梯度原样传下去）")

L = (p - y) ** 2
print("\n③ L = (p − y)²")
print(f"   值：L = ({p} − {y})² = {L}")
print(f"   局部导数：∂L/∂p = 2(p − y) = {2*(p-y)}")

print("\n" + "=" * 62)
print("计算图长这样（前向）：")
print(f"   x={x} ─┐")
print(f"          ×(w={w}) → u={u} ─(+b={b})→ p={p} ─(p−y)²→ L={L}")
print(f"   w={w} ─┘")
print("\n为什么要画成图：")
print("   · 每个节点的导数都能手算（乘法、加法、平方）")
print("   · loss 对任意输入的导数 = 沿路径把局部导数乘起来（第 5 课）")
print("   · 框架（PyTorch）内部就是维护这样一张图，叫 autograd")

# ---------------- 图 ----------------
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.set_xlim(0, 11); ax.set_ylim(0, 4.4); ax.axis("off")

nodes = [(1.0, 3.0, f"x = {x}"), (1.0, 1.2, f"w = {w}"), (4.2, 2.1, f"u = w·x\n= {u}"),
         (7.0, 2.1, f"p = u+b\n= {p}"), (9.6, 2.1, f"L = (p−y)²\n= {L}")]
for x0, y0, t in nodes:
    ax.add_patch(plt.Rectangle((x0 - 0.75, y0 - 0.42), 1.5, 0.84, fc="white", ec=BLUE, lw=1.5))
    ax.text(x0, y0, t, ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.5)

for (x1, y1), (x2, y2) in [((1.0, 3.0), (4.2, 2.1)), ((1.0, 1.2), (4.2, 2.1)),
                           ((4.2, 2.1), (7.0, 2.1)), ((7.0, 2.1), (9.6, 2.1))]:
    ax.annotate("", xy=(x2 - 0.78, y2 + (0.05 if y2 > y1 else -0.05)), xytext=(x1 + 0.78, y1 - (0.05 if y2 > y1 else -0.05)),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.3))

ax.text(2.6, 2.75, "∂u/∂w = x = 2", fontsize=8.5, color=ORANGE, ha="center")
ax.text(2.6, 1.55, "∂u/∂x = w = 3", fontsize=8.5, color=ORANGE, ha="center")
ax.text(5.6, 2.35, "∂p/∂u = 1", fontsize=8.5, color=ORANGE, ha="center")
ax.text(8.3, 2.35, "∂L/∂p = 2(p−y) = 4", fontsize=8.5, color=ORANGE, ha="center")
ax.text(5.5, 0.35, "前向：从输入一路算到 loss；每个箭头旁边写的是「局部导数」",
        ha="center", fontsize=9.5, color=GRAY)
save(fig, "04_graph.png")
