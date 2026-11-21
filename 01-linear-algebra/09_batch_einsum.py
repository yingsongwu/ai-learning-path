# -*- coding: utf-8 -*-
"""
第 9 课（深入）：批量矩阵乘、广播、einsum、三维张量

为什么单独讲这一课？
    因为真实模型里几乎不出现"一个向量乘一个矩阵"，全是：
        (batch, 特征) × (特征, 输出)        ← 一次算 64 个样本
        (batch, 序列, 维度) × (维度, 输出)   ← 一次算一批句子
    形状多一维，脑子里就要多一根弦。

四个必须会的东西：
    1. 批量矩阵乘：@ 对最右两维做矩阵乘，前面所有维都是"批量"
    2. 广播：bias (64,) 自动加到 (batch,64) 的每一行
    3. einsum：用字母直白写出"谁和谁相乘、求和掉哪个维度"
    4. 注意力里的 QKᵀ：本质就是一次批量矩阵乘

运行：python 09_batch_einsum.py
"""
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GREEN, GRAY, INK

rng = np.random.default_rng(0)


def show(name, arr):
    print(f"   {name:26s} shape={str(arr.shape):18s} 前两个值={np.round(arr.ravel()[:2], 3)}")


print("=" * 62)
print("1) 批量矩阵乘：前面的维度都是批量")
X = rng.random((8, 256)).astype(np.float32)     # 8 个样本，每个 256 维
W = rng.standard_normal((256, 64)).astype(np.float32)
H = X @ W
show("X", X); show("W", W); show("X @ W", H)
print("   规则：(8,256)@(256,64) → (8,64)，8 是批量维，原样保留")

print("\n2) 广播：bias 怎么加到每个样本上")
b = np.zeros(64)
print(f"   H(8,64) + b(64,) → {(H + b).shape}：b 自动复制成 8 行")
try:
    H + np.zeros(8)
except ValueError as e:
    print("   H + 长度 8 的向量 会怎样？", str(e).split("\n")[0])
print("   记忆：从最右边对齐，长度 1 或相等才能广播")

print("\n3) 三维张量：(batch, 序列长度, 维度)")
T = rng.random((4, 10, 32)).astype(np.float32)   # 4 句、每句 10 词、每词 32 维
Wo = rng.random((32, 16)).astype(np.float32)
out = T @ Wo
show("T (batch,seq,dim)", T); show("Wo", Wo); show("T @ Wo", out)
print("   @ 只管最右两维：(10,32)@(32,16) → (10,16)，batch=4 原样保留")

print("\n4) einsum：把形状写成字母，最不容易错")
A = rng.random((2, 3))
Bm = rng.random((3, 4))
print("   普通矩阵乘   np.einsum('ik,kj->ij', A, Bm) =", np.allclose(np.einsum('ik,kj->ij', A, Bm), A @ Bm))
print("   批量矩阵乘   np.einsum('bik,kj->bij', T, Wo).shape =", np.einsum('bik,kj->bij', T, Wo).shape)
q = rng.random((4, 10, 32)); k = rng.random((4, 12, 32))
scores = np.einsum('bqd,bkd->bqk', q, k)
print("   注意力打分   'bqd,bkd->bqk' →", scores.shape, "（每句 10×12 的相似度矩阵）")
print("   ↑ 这就是 QKᵀ：q 和 k 逐词算点积，得到'谁该关注谁'")

print("\n5) 为什么必须用 einsum 思维写代码")
print("   普通 for 循环：慢、易错、看不到形状")
print("   矩阵写法   ：一次算完、形状即文档、GPU 友好")

# ---------------- 图 ----------------
fig = plt.figure(figsize=(12.5, 4.2))
ax = fig.add_subplot(1, 3, 1)
for i in range(4):
    for j in range(6):
        ax.add_patch(plt.Rectangle((i * 1.3 + j * 0.12, j * 0.12 + i * 0.1), 0.9, 0.9,
                                   fc="#eef3f8", ec=BLUE, lw=0.8, alpha=0.9))
ax.set_xlim(-0.3, 6.2); ax.set_ylim(-0.3, 3.2); ax.axis("off")
ax.set_title("(batch, seq, dim)\n三个维度 = 三层抽屉", fontsize=10, color=INK)

ax = fig.add_subplot(1, 3, 2)
ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
ax.text(5, 3.5, "(4,10,32) @ (32,16) → (4,10,16)", ha="center", fontsize=10, color=INK)
ax.annotate("", xy=(7.5, 2.2), xytext=(2.5, 2.2),
            arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.6))
ax.text(2.2, 2.45, "只看最右两维", fontsize=9, color=GRAY)
ax.text(7.5, 2.45, "batch 原样保留", fontsize=9, color=GRAY)
ax.text(5, 1.2, "内维 32 被消掉\n序列维 10 不动", ha="center", fontsize=9.5, color=INK, linespacing=1.6)

ax = fig.add_subplot(1, 3, 3)
sc = np.random.default_rng(1).random((10, 12))
im = ax.imshow(sc, cmap="Blues")
ax.set_xlabel("key 位置（12）", fontsize=9, color=GRAY)
ax.set_ylabel("query 位置（10）", fontsize=8.5, color=GRAY, labelpad=6)
ax.tick_params(labelsize=8)
ax.set_title("注意力打分矩阵 'bqd,bkd->bqk'", fontsize=10, color=INK)
fig.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("多一维不可怕：记住「最右两维做矩阵乘，左边全是批量」", fontsize=11, color=INK)
fig.tight_layout()
save(fig, "09_einsum.png")
