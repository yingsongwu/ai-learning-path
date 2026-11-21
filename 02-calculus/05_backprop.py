# -*- coding: utf-8 -*-
"""
第 5 课：反向传播 —— 沿计算图把导数「乘回去」

规则只有一条（链式法则）：
    dL/d(某个输入) = 上游传来的梯度 × 这个节点的局部导数

本例沿用第 4 课：x=2, w=3, b=1, y=5，loss = (wx+b−y)²
从右往左算：
    dL/dp = 2(p−y)        = 4
    dL/du = dL/dp · 1     = 4
    dL/dw = dL/du · x     = 8
    dL/db = dL/dp · 1     = 4

然后走一步梯度下降，loss 应该变小。

运行：python 05_backprop.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GREEN, GRAY, INK

x, w, b, y = 2.0, 3.0, 1.0, 5.0
lr = 0.05


def forward(w, b):
    u = w * x
    p = u + b
    L = (p - y) ** 2
    return u, p, L


print("=" * 64)
print(f"初始：w={w}, b={b}，目标 y={y}")
u, p, L = forward(w, b)
print(f"前向：u={u}, p={p}, L={L}\n")

print("反向（从 loss 往输入走）：")
dL_dp = 2 * (p - y)
print(f"  dL/dp = 2(p−y) = 2×({p}−{y}) = {dL_dp}")
dL_du = dL_dp * 1
print(f"  dL/du = dL/dp × ∂p/∂u = {dL_dp} × 1 = {dL_du}")
dL_db = dL_dp * 1
print(f"  dL/db = dL/dp × ∂p/∂b = {dL_dp} × 1 = {dL_db}")
dL_dw = dL_du * x
print(f"  dL/dw = dL/du × ∂u/∂w = {dL_du} × {x} = {dL_dw}")

print(f"\n梯度的方向含义：")
print(f"  dL/dw = {dL_dw} > 0 → w 增大 loss 会增大 → 应该减小 w")
print(f"  dL/db = {dL_db} > 0 → b 同理")

print(f"\n走一步梯度下降（学习率 {lr}）：")
w2 = w - lr * dL_dw
b2 = b - lr * dL_db
u2, p2, L2 = forward(w2, b2)
print(f"  w: {w} → {w2:.3f}      b: {b} → {b2:.3f}")
print(f"  loss: {L} → {L2:.4f}     （变小了 {L-L2:.4f}）")

print("\n再走几步：")
ww, bb = w, b
for i in range(1, 8):
    uu, pp, LL = forward(ww, bb)
    dp = 2 * (pp - y)
    ww -= lr * dp * x
    bb -= lr * dp
    print(f"  第 {i} 步：w={ww:6.3f}  b={bb:6.3f}  loss={forward(ww,bb)[2]:.5f}")

print("\n关键观察：")
print("  · 梯度里乘了 x（=2），所以 w 的梯度是 b 的两倍 —— 「输入大的权重改得多」")
print("  · 这解释了为什么输入要归一化：否则某些参数的梯度和别的不在一个量级")

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
ax = axes[0]
ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
nodes = [(1.2, 3.2, f"x=2"), (1.2, 1.4, f"w={w}"), (4.2, 2.3, f"u={u}"), (6.8, 2.3, f"p={p}"), (9.3, 2.3, f"L={L}")]
for x0, y0, t in nodes:
    ax.add_patch(plt.Rectangle((x0 - 0.62, y0 - 0.34), 1.24, 0.68, fc="white", ec=BLUE, lw=1.4))
    ax.text(x0, y0, t, ha="center", va="center", fontsize=9, color=INK)
for (x1, y1), (x2, y2) in [((1.2, 3.2), (4.2, 2.3)), ((1.2, 1.4), (4.2, 2.3)),
                           ((4.2, 2.3), (6.8, 2.3)), ((6.8, 2.3), (9.3, 2.3))]:
    ax.annotate("", xy=(x2 - 0.65, y2 + 0.03), xytext=(x1 + 0.65, y1 - 0.03),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2, alpha=.55))
# 反向箭头
for (x1, y1), (x2, y2) in [((9.3, 1.9), (6.8, 1.9)), ((6.8, 1.9), (4.2, 1.9)),
                           ((4.2, 1.9), (1.2, 1.0))]:
    ax.annotate("", xy=(x2 + 0.68, y2), xytext=(x1 - 0.68, y1),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8))
ax.text(8.1, 1.55, "×1", fontsize=9, color=RED, ha="center")
ax.text(5.5, 1.55, "×1", fontsize=9, color=RED, ha="center")
ax.text(2.7, 1.35, "×x=2", fontsize=9, color=RED, ha="center")
ax.text(5.5, 0.5, "反向：每一步只乘一个局部导数 —— 这就是链式法则的全部",
        ha="center", fontsize=9.5, color=GRAY)

ax = axes[1]
losses = []
ww, bb = 3.0, 1.0
for _ in range(12):
    losses.append(forward(ww, bb)[2])
    dp = 2 * (forward(ww, bb)[1] - y)
    ww -= lr * dp * x
    bb -= lr * dp
ax.plot(losses, "o-", color=GREEN, ms=4)
ax.set_xlabel("更新步数", fontsize=10, color=INK); ax.set_ylabel("loss", fontsize=10, color=INK)
ax.set_title("每走一步，loss 都在下降", fontsize=10.5, color=INK)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "05_backprop.png")
