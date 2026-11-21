# -*- coding: utf-8 -*-
"""
第 10 课（深入）：PCA 实操 —— 用 SVD 把二维压成一维

PCA 要解决的问题：
    一堆点，怎么用最少的维度、尽量不丢信息地描述它们？

答案（就是第 5 课的 SVD）：
    1. 先减去均值（中心化）—— 让数据以原点为中心
    2. 求主方向 = 协方差矩阵的最大特征向量（等价于对数据做 SVD）
    3. 把点投影到这个方向上 —— 从二维变成一维
    4. 投影后能还原多少？看"解释方差比例"

在 AI 里的位置：
    · 降维可视化（把 256 维降到 2 维看聚类）
    · 去掉冗余特征、加速训练
    · LoRA 假设权重更新是低秩的，本质同一件事

运行：python 10_pca.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GRAY, INK

rng = np.random.default_rng(7)

# ---------- 1) 造一批"明显有方向性"的二维数据 ----------
n = 200
base = rng.standard_normal((n, 2))
base[:, 1] *= 0.25                                  # 让一个方向更窄
theta = np.deg2rad(35)
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
X = base @ R.T + np.array([2.0, 1.5])               # 旋转 + 平移

print("=" * 60)
print(f"数据：{n} 个二维点，肉眼能看出是斜着拉长的一团")

# ---------- 2) 中心化 ----------
mean = X.mean(axis=0)
Xc = X - mean
print(f"\n1) 中心化：均值 {np.round(mean, 3)} 减掉 → 新均值 {np.round(Xc.mean(axis=0), 6)}")

# ---------- 3) 协方差 + 特征分解 ----------
cov = Xc.T @ Xc / (n - 1)
vals, vecs = np.linalg.eigh(cov)
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]
print("\n2) 协方差矩阵 =\n", np.round(cov, 4))
print("   特征值（方差大小） =", np.round(vals, 4))
print("   主方向（第一主成分）=", np.round(vecs[:, 0], 4))

ratio = vals[0] / vals.sum()
print(f"\n3) 第一主成分解释了 {ratio*100:.1f}% 的方差 → 压成一维只丢 {(1-ratio)*100:.1f}% 的信息")

# ---------- 4) 投影与重建 ----------
v1 = vecs[:, 0]
proj = Xc @ v1                       # 一维坐标（投影到主方向）
recon = np.outer(proj, v1) + mean    # 用一维还原二维
err = np.linalg.norm(X - recon, axis=1).mean()
print(f"\n4) 投影后每个点的平均重建误差 = {err:.3f}（数据本身尺度约 {np.abs(Xc).mean():.2f}）")

# 对比：用随机方向投影会怎样
rand_dir = np.array([1.0, 0.0])
proj2 = Xc @ rand_dir
recon2 = np.outer(proj2, rand_dir) + mean
err2 = np.linalg.norm(X - recon2, axis=1).mean()
print(f"   如果随便用 x 轴方向投影，误差 = {err2:.3f}（明显更大）")
print("   → PCA 就是「挑一个让误差最小的方向」")

# ---------- 5) 与 SVD 完全等价 ----------
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
print("\n5) 用 SVD 验证：Vt 的第一行 =", np.round(Vt[0], 4),
      " 与主方向一致:", np.allclose(np.abs(Vt[0]), np.abs(v1)))
print("   （SVD 的奇异值平方 / (n-1) 就是特征值：", np.round(S**2/(n-1), 4), "）")

# ---------------- 图 ----------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], s=14, color=BLUE, alpha=0.55, label="原始数据")
ax.set_title("原始二维数据：斜着拉长", fontsize=10.5, color=INK)
ax.legend(fontsize=9, frameon=False)

ax = axes[1]
ax.scatter(Xc[:, 0], Xc[:, 1], s=14, color=BLUE, alpha=0.45)
for i, (val, vec, c) in enumerate(zip(vals, vecs.T, [RED, ORANGE])):
    e = vec * np.sqrt(val) * 2.2
    ax.annotate("", xy=e, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
    ax.annotate("", xy=-e, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.2))
    ax.text(e[0] * 1.12, e[1] * 1.12, f"PC{i+1}", color=c, fontsize=10)
ax.axhline(0, color="#eef3f8"); ax.axvline(0, color="#eef3f8")
ax.set_title("两个主方向（长度 ∝ 方差）", fontsize=10.5, color=INK)
ax.set_aspect("equal")

ax = axes[2]
ax.scatter(X[:, 0], X[:, 1], s=12, color="#c9d6e2", label="原数据")
ax.scatter(recon[:, 0], recon[:, 1], s=14, color=RED, alpha=0.8, label="一维投影还原")
for i in range(0, n, 8):
    ax.plot([X[i, 0], recon[i, 0]], [X[i, 1], recon[i, 1]], "-", color=GRAY, lw=0.5)
ax.set_title(f"只保留 1 维（解释 {ratio*100:.0f}% 方差）", fontsize=10.5, color=INK)
ax.legend(fontsize=9, frameon=False)

for ax in axes:
    ax.tick_params(labelsize=8, colors=GRAY)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("PCA = 找方差最大的方向，把数据投上去；和 SVD 是同一件事", fontsize=11, color=INK)
fig.tight_layout()
save(fig, "10_pca.png")

print("\n" + "=" * 60)
print("一句话：PCA 不是「随便降维」，是「沿着最不丢信息的方向降维」。")
