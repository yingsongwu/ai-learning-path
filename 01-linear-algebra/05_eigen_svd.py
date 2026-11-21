# -*- coding: utf-8 -*-
"""
第 5 课：特征值与奇异值分解 —— 矩阵的"骨架"

特征向量：被矩阵变换后方向不变、只被拉长/缩短的那些方向。
        A @ v = λ · v          λ 就是"拉伸倍数"（特征值），沿 v 方向不变
        意义：它是这个变换最"本质"的方向 —— 说清一个矩阵在干什么

奇异值分解（SVD）：任意矩阵 A = U · S · Vᵀ
        意义：任何矩阵 = 旋转 · 拉伸 · 旋转；取前 k 个最大的奇异值，
             就能得到"最像原矩阵"的 k 秩近似 → 图像压缩、PCA 的底座

运行：python 05_eigen_svd.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GRAY, INK

np.set_printoptions(precision=3, suppress=True)

# ---------------- 1) 特征值 ----------------
A = np.array([[2.0, 1.0],
              [1.0, 2.0]])
vals, vecs = np.linalg.eigh(A)
print("=" * 58)
print("A =\n", A)
print("\n特征值 λ =", vals)
print("特征向量（每一列）=\n", vecs)

print("\n验证 A·v = λ·v：")
for i in range(2):
    v = vecs[:, i]
    print(f"  第 {i+1} 个方向 v={v} → A·v={A @ v}，λ·v={vals[i] * v}"
          f"  {'✅' if np.allclose(A @ v, vals[i] * v) else '❌'}")

print("\n几何含义：")
print("  λ>1 的方向被拉长，λ<1 被压缩，λ<0 相当于翻转 + 缩放")
print("  这两个方向就是椭圆的「长轴/短轴」方向")

# ---------------- 2) SVD 压缩 ----------------
H, W = 200, 200
yy, xx = np.mgrid[-1:1:complex(H), -1:1:complex(W)]
# 造一张"有结构也有细节"的图：同心圆环 + 方块 + 斜条纹 + 一点噪声
img = (np.sin(7 * np.sqrt(xx**2 + yy**2)) + 1) / 2 * 0.8
img[(np.abs(xx - 0.45) < 0.22) & (np.abs(yy + 0.45) < 0.18)] = 0.95      # 亮方块
img[(np.abs(xx + 0.5) < 0.12) & (np.abs(yy - 0.35) < 0.3)] = 0.05       # 暗竖条
img += 0.12 * np.sin(12 * (xx + yy))                                     # 斜条纹（高频）
img += 0.04 * np.random.default_rng(0).standard_normal(img.shape)        # 噪声
img = np.clip(img, 0, 1)

U, S, Vt = np.linalg.svd(img, full_matrices=False)
print("\n" + "=" * 58)
print(f"图像 {img.shape}，奇异值共 {len(S)} 个；最大的几个：{np.round(S[:5], 2)}")
print("前 k 个奇异值占的比例：")
for k in [5, 10, 20, 50]:
    ratio = S[:k].sum() / S.sum()
    print(f"   k={k:3d} → 保留 {ratio*100:5.1f}% 的能量，"
          f"存储 {k*(H+W+1)} 个数 vs 原件 {H*W} 个数")

def rebuild(k):
    return (U[:, :k] * S[:k]) @ Vt[:k, :]

# ---------------- 图 ----------------
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(2, 3, 1)
ax.imshow(img, cmap="gray")
ax.set_title("原图 200×200", fontsize=10, color=INK); ax.axis("off")

for j, k in enumerate([5, 20]):
    ax = fig.add_subplot(2, 3, 2 + j)
    ax.imshow(rebuild(k), cmap="gray")
    ax.set_title(f"只保留 {k} 个奇异值", fontsize=10, color=INK); ax.axis("off")

ax = fig.add_subplot(2, 3, 4)
ax.imshow(img, cmap="gray", alpha=0.15)
for i in range(2):
    v = vecs[:, i] * np.sqrt(vals[i]) * 1.3
    c = [BLUE, RED][i]
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.4))
    ax.annotate("", xy=-v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, ls=":"))
    ax.text(v[0] * 1.1, v[1] * 1.1, f"λ={vals[i]:.1f}", color=c, fontsize=9)
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
ax.set_title("特征方向：变换后方向不变", fontsize=10, color=INK)

ax = fig.add_subplot(2, 3, 5)
ax.plot(S[:60], "o-", ms=3, color=ORANGE)
ax.set_title("奇异值衰减（掉得越快越能压缩）", fontsize=10, color=INK)
ax.set_xlabel("序号", fontsize=9, color=GRAY); ax.set_ylabel("大小", fontsize=9, color=GRAY)
ax.spines[["top", "right"]].set_visible(False)

ax = fig.add_subplot(2, 3, 6)
ks = [1, 2, 5, 10, 20, 50, 100]
errs = [np.linalg.norm(img - rebuild(k)) / np.linalg.norm(img) for k in ks]
ax.plot(ks, [e * 100 for e in errs], "o-", color=BLUE, ms=4)
ax.set_xscale("log")
ax.set_xlabel("k（保留的奇异值个数，对数轴）", fontsize=9, color=GRAY)
ax.set_ylabel("相对误差 %", fontsize=9, color=GRAY)
ax.set_title("k 越大越准，但存储也越大", fontsize=10, color=INK)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("SVD：任何矩阵 = 旋转·拉伸·旋转，取前 k 个奇异值就是最好的 k 秩近似", fontsize=11.5, color=INK)
fig.tight_layout()
save(fig, "05_eigen_svd.png")

print("\n" + "=" * 58)
print("和 AI 的关系：PCA 降维 = 对数据矩阵做 SVD；")
print("LoRA 微调 = 假设权重更新是低秩的（两个小矩阵相乘），本质同一件事。")
