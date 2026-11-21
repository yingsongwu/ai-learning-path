# -*- coding: utf-8 -*-
"""
第 1 课：向量、点积、夹角 —— "相似度"的唯一来源

一句话：向量是一组数字，也是一支箭头；点积 = 两支箭头"同方向的程度"。

为什么 AI 里天天用点积？
    · 神经元：w·x 就是"权重和输入有多像"
    · 本工程第 1 步：拿模板给图片打分，用的就是点积
    · 注意力机制：Q·K 算"查询和键有多像"
它无处不在，因为它是"相似度"最便宜、最好算的定义。

运行：python 01_vectors.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, GRAY, INK, ORANGE

np.set_printoptions(precision=3, suppress=True)

a = np.array([3.0, 1.0])
b = np.array([1.0, 2.0])
c = np.array([-1.0, 3.0])          # 与 a 垂直


def angle(u, v):
    """两向量夹角（度）。cosθ = u·v / (|u||v|)"""
    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


print("=" * 56)
print("1) 向量就是一组数字，也可以看成从原点出发的箭头")
print("   a =", a, " b =", b, " c =", c)

print("\n2) 点积：逐个数相乘再相加")
print("   a·b = 3*1 + 1*2 =", float(a @ b))
print("   a·c = 3*(-1) + 1*3 =", float(a @ c), "  ← 为 0 说明垂直（毫无相似）")

print("\n3) 点积 ÷ 长度 = 余弦相似度（-1 ~ 1），这才是可比的标准")
for name, v in [("b", b), ("c", c)]:
    print(f"   cos(a,{name}) = {float(a @ v / (np.linalg.norm(a)*np.linalg.norm(v))):+.3f}"
          f"   夹角 {angle(a, v):5.1f}°")

print("\n4) 长度（范数）：L2 = 各分量平方和开根号")
print("   |a| =", round(float(np.linalg.norm(a)), 4),
      "  L1 =", float(np.abs(a).sum()), "（曼哈顿距离）")

print("\n5) 投影：a 在 b 方向上的分量长度 = a·b / |b|")
print("   a 在 b 上的投影长度 =", round(float((a @ b) / np.linalg.norm(b)), 4))

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

ax = axes[0]
for v, name, color in [(a, "a", BLUE), (b, "b", RED), (c, "c", GRAY)]:
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2))
    ax.text(v[0] * 1.08, v[1] * 1.08, name, color=color, fontsize=13)
ax.axhline(0, color="#dfe7ef", lw=1); ax.axvline(0, color="#dfe7ef", lw=1)
ax.set_xlim(-2, 4); ax.set_ylim(-0.5, 3.6)
ax.set_aspect("equal")
ax.set_title(f"向量与夹角：a∠b={angle(a,b):.0f}°，a⊥c", fontsize=11, color=INK)

ax = axes[1]
ax.annotate("", xy=a, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.2))
ax.annotate("", xy=b, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2.2))
# 投影
t = (a @ b) / (b @ b)
proj = t * b
ax.plot([a[0], proj[0]], [a[1], proj[1]], "--", color=GRAY, lw=1.2)
ax.annotate("", xy=proj, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2.6))
ax.text(0.15, 1.4, "a 在 b 上的投影\n（点积决定它有多长）", fontsize=9.5, color=GRAY)
ax.axhline(0, color="#dfe7ef", lw=1); ax.axvline(0, color="#dfe7ef", lw=1)
ax.set_xlim(-0.3, 3.4); ax.set_ylim(-0.3, 2.4)
ax.set_aspect("equal")
ax.set_title("投影：把 a 投到 b 的方向上", fontsize=11, color=INK)

save(fig, "01_vectors.png")

print("\n" + "=" * 56)
print("小结：点积 = 相似度。角度越接近 0，点积越大；垂直时点积为 0。")
print("下一步：把很多「向量」排成矩阵，看矩阵对空间做了什么。")
