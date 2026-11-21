# -*- coding: utf-8 -*-
"""
公式运算实验室 —— 把每个公式拆成"能拿笔算出来"的小步骤

为什么要有这个文件？
    看懂符号 ≠ 会算。很多人卡在"知道 softmax 是 e^z / Σe^z，但给我三个数我算不出来"。
    这个脚本的每个函数都只做一件事：
        用极小的数字 → 打印每一步的中间结果 → 告诉你这一步在干什么

8 个公式，全部可以在纸上重算：
    F1 点积与余弦      F5 反向传播（dW = xᵀ·dh）
    F2 矩阵乘法        F6 特征值方程 Av = λv
    F3 softmax         F7 SVD 的 k 秩近似
    F4 交叉熵与梯度    F8 最小二乘正规方程
    （另附 F9 PCA 五步、F10 einsum 字母规则）

运行：
    python formula_lab.py            # 全部跑一遍
    python formula_lab.py 3          # 只看 F3
"""
import sys
import numpy as np
from _plot import plt, save, BLUE, RED, ORANGE, GREEN, GRAY, INK

np.set_printoptions(precision=4, suppress=True)


def title(s):
    print("\n" + "=" * 66)
    print(s)
    print("=" * 66)


def step(k, expr, value, note=""):
    """打印一行计算步骤：第几步 / 式子 / 数值 / 说明"""
    kk = f"{k:>2}." if not isinstance(k, str) else f"{k:>4}"
    print(f"  {kk} {expr:<34} = {value!s:<20} {note}")


# ============================================================ F1 点积与余弦
def f1():
    title("F1 点积与余弦相似度：两个向量的「像不像」")
    a = np.array([3.0, 4.0])
    b = np.array([1.0, 2.0])

    print("  已知：a = (3, 4)   b = (1, 2)")
    print("  公式：a·b = a₁b₁ + a₂b₂      cos = a·b / (|a|·|b|)\n")

    step(1, "a₁×b₁ = 3×1", 3 * 1)
    step(2, "a₂×b₂ = 4×2", 4 * 2)
    step(3, "相加 a·b = 3 + 8", 3 + 8, "← 这就是点积")
    step(4, "|a| = √(3²+4²)", round(float(np.linalg.norm(a)), 4), "3-4-5 三角形")
    step(5, "|b| = √(1²+2²)", round(float(np.linalg.norm(b)), 4))
    step(6, "cos = 11 / (5×2.236)", round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))), 4))
    step(7, "夹角 = arccos(cos)",
         f"{np.degrees(np.arccos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))):.1f}°")

    print("\n  为什么这样定义？")
    print("    · a·b 大 = 两个箭头同方向 = 相似度高")
    print("    · 除以两个长度，是为了「排除长度影响」，只比方向")
    print("    · 垂直时 a·b = 0 → cos = 0（完全不相关）")
    print("\n  记忆：先相乘、再相加、最后除长度。")

    # 图：相乘 → 相加
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    ax.bar([0, 1], [3 * 1, 4 * 2], color=[BLUE, ORANGE], width=0.55)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["a₁b₁ = 3", "a₂b₂ = 8"])
    ax.set_title("第 1 步：逐项相乘", fontsize=10.5, color=INK)
    for i, v in enumerate([3, 8]):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=11, color=INK)
    ax.set_ylim(0, 9.5)

    ax = axes[1]
    ax.bar([0], [11], color=GREEN, width=0.4)
    ax.bar([0], [3], color=BLUE, width=0.4)
    ax.text(0, 11.3, "合计 11", ha="center", fontsize=11, color=INK)
    ax.set_xticks([0]); ax.set_xticklabels(["a·b = 3 + 8"])
    ax.set_title("第 2 步：相加得到点积", fontsize=10.5, color=INK)
    ax.set_ylim(0, 13)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=GRAY, labelsize=9)
    save(fig, "F1_dot.png")


# ============================================================ F2 矩阵乘法
def f2():
    title("F2 矩阵乘法：为什么是「行乘列」")
    A = np.array([[1, 2, 3],
                  [4, 5, 6]])
    B = np.array([[7, 8],
                  [9, 10],
                  [11, 12]])

    print("  A 是 2×3，B 是 3×2，结果 C 是 2×2")
    print("  规则：C[i][j] = A 的第 i 行 · B 的第 j 列\n")

    for i in range(2):
        for j in range(2):
            terms = [f"{A[i][k]}×{B[k][j]}" for k in range(3)]
            step(f"C[{i+1}][{j+1}]", " + ".join(terms), int(A[i] @ B[:, j]),
                 f"← A 第{i+1}行·B 第{j+1}列")
    print("\n  验证 numpy：A@B =\n", A @ B)

    print("\n  为什么要「行乘列」？换个角度算：")
    print("    C = A@B 等价于：把 B 的每一行按 A 的系数加权后相加")
    for i in range(2):
        parts = " + ".join(f"{A[i][k]}×B第{k+1}行" for k in range(3))
        print(f"      C 第{i+1}行 = {parts} = {A[i] @ B}")
    print("    也就是说：结果的每一行，是 B 各行的「加权平均」")

    # 图：高亮一格的计算
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def grid(x0, y0, M, name, hl_row=None, hl_col=None):
        rows, cols = M.shape
        for i in range(rows):
            for j in range(cols):
                fc = "#eef3f8"
                if hl_row is not None and i == hl_row: fc = "#fde9e7"
                if hl_col is not None and j == hl_col: fc = "#e8f6ee"
                ax.add_patch(plt.Rectangle((x0 + j * 0.9, y0 - i * 0.9), 0.85, 0.85,
                                           fc=fc, ec=GRAY, lw=1))
                ax.text(x0 + j * 0.9 + 0.42, y0 - i * 0.9 + 0.42, str(M[i, j]),
                        ha="center", va="center", fontsize=11, color=INK)
        ax.text(x0 + cols * 0.45, y0 + 1.0, name, ha="center", fontsize=11, color=INK)

    grid(0.6, 4.6, A, "A (2×3)", hl_row=0)
    grid(4.0, 4.6, B, "B (3×2)", hl_col=0)
    ax.text(3.55, 4.1, "@", fontsize=18, color=INK)
    ax.text(7.6, 4.1, "=", fontsize=18, color=INK)
    C = A @ B
    ax.add_patch(plt.Rectangle((7.9, 3.75), 0.85, 0.85, fc="#ffe9a8", ec=ORANGE, lw=2))
    ax.text(8.32, 4.17, str(C[0, 0]), ha="center", va="center", fontsize=12, color=INK)
    ax.text(8.4, 3.4, "C[1][1]\n=1×7+2×9+3×11\n=58", ha="center", va="top",
            fontsize=9, color=INK, linespacing=1.5)
    ax.text(5, 0.4, "红：A 的第 1 行　绿：B 的第 1 列　两者逐项相乘再相加 = C[1][1]",
            ha="center", fontsize=9, color=GRAY)
    save(fig, "F2_matmul.png")


# ============================================================ F3 softmax
def f3():
    title("F3 softmax：三个分数怎么变成三个概率")
    z = np.array([2.0, 1.0, 0.0])
    e = np.exp(z)
    s = e.sum()
    p = e / s

    print("  已知 logits：z = (2, 1, 0)")
    print("  公式：pᵢ = e^{zᵢ} / Σⱼ e^{zⱼ}\n")
    for i in range(3):
        step(f"e^{z[i]}", f"e^{z[i]:.0f}", round(float(e[i]), 4), "先取指数（都变正数）")
    step("分母", "e² + e¹ + e⁰", round(float(s), 4), "三者相加")
    for i in range(3):
        step(f"p{i+1}", f"{e[i]:.4f} / {s:.4f}", round(float(p[i]), 4), "最后归一化")
    print(f"\n  检查：p 加起来 = {p.sum():.4f}（必须是 1）")
    print("  注意：z 只差 2，概率却差 7 倍多 —— 指数会放大差距")

    print("\n  为什么要减最大值（数值稳定）：")
    zz = np.array([1000.0, 999.0, 998.0])
    print(f"    直接算 e^1000 = {np.exp(1000.0) if False else 'inf（溢出）'}")
    zs = zz - zz.max()
    print(f"    减最大值后 z' = {zs}，e^z' = {np.round(np.exp(zs),4)}，概率完全一样")

    # 图：三阶段
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    stages = [(z, "① logits（可正可负）", GRAY),
              (e, "② 取指数 e^z（都为正）", ORANGE),
              (p, "③ 除以总和（加起来=1）", GREEN)]
    for ax, (vals, t, c) in zip(axes, stages):
        bars = ax.bar(range(3), vals, color=c, width=0.6, alpha=0.85)
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + max(vals) * 0.03,
                    f"{v:.2f}", ha="center", fontsize=9, color=INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(["类别1", "类别2", "类别3"])
        ax.set_title(t, fontsize=10, color=INK)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=GRAY, labelsize=8)
    save(fig, "F3_softmax.png")


# ============================================================ F4 交叉熵与梯度
def f4():
    title("F4 交叉熵与它的梯度：为什么导数恰好是 p − y")
    p = np.array([0.1, 0.7, 0.2])
    y_idx = 1          # 正确答案是第 2 类
    y = np.zeros(3); y[y_idx] = 1

    print("  模型输出概率 p = (0.1, 0.7, 0.2)，正确答案是第 2 类\n")
    step(1, "取出正确答案的概率", f"p[{y_idx+1}] = {p[y_idx]:.2f}")
    step(2, "取负对数 loss = −ln(0.7)", round(float(-np.log(p[y_idx])), 4))
    step(3, "梯度 dz = p − y", np.round(p - y, 3), "← 关键：直接减 one-hot")
    print("\n  为什么梯度这么简单？推导两步：")
    print("    loss = −ln(p_c)，而 p_c = e^{z_c}/Σe^{z_j}")
    print("    对 z_c 求导 → (p_c − 1)；对其他 z_j 求导 → p_j")
    print("    合起来就是 p − y（y 是 one-hot，只有正确类是 1）")
    print("\n  梯度告诉我们什么：")
    print(f"    · 正确类 z2 的梯度 = {p[1]-1:+.2f} → 负的，所以要「增大」它")
    print(f"    · 错误类 z1 的梯度 = {p[0]:+.2f} → 正的，所以要「减小」它")
    print("    · 幅度还和概率有关：错得越离谱（p_c 越小），梯度越大，改得越狠")

    # 图：loss 曲线 + 梯度
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    x = np.linspace(0.02, 1, 300)
    axes[0].plot(x, -np.log(x), color=RED, lw=2.2)
    axes[0].plot([0.7], [-np.log(0.7)], "o", color=INK, ms=8)
    axes[0].annotate("正确类概率 0.7\nloss = 0.357", xy=(0.7, 0.357), xytext=(0.42, 1.5),
                     fontsize=9, color=INK, linespacing=1.5,
                     arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))
    axes[0].set_xlabel("p（正确类概率）", fontsize=10, color=INK)
    axes[0].set_ylabel("loss = −ln(p)", fontsize=10, color=INK)
    axes[0].set_title("第 2 步：loss 只由正确类决定", fontsize=10.5, color=INK)

    axes[1].bar(range(3), p - y, color=[BLUE, RED, BLUE], width=0.55)
    axes[1].axhline(0, color=GRAY, lw=1)
    axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["类别1", "类别2(正确)", "类别3"])
    axes[1].set_title("第 3 步：梯度 dz = p − y", fontsize=10.5, color=INK)
    for i, v in enumerate(p - y):
        axes[1].text(i, v + (0.03 if v > 0 else -0.06), f"{v:+.2f}", ha="center",
                     fontsize=9.5, color=INK)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=GRAY, labelsize=8)
    save(fig, "F4_ce_grad.png")


# ============================================================ F5 反向传播
def f5():
    title("F5 反向传播：把数字一路乘回去（2→2→1 的小网络）")
    x = np.array([[1.0, 2.0]])          # 1 个样本，2 个输入
    W1 = np.array([[0.5, -1.0],
                   [1.0, 0.5]])         # 2×2
    b1 = np.zeros(2)
    W2 = np.array([[2.0], [-1.0]])      # 2×1
    b2 = np.array([0.1])
    y = np.array([[1.0]])               # 目标

    print("  输入 x = (1, 2)，目标 y = 1\n")
    print("  —— 前向 ——")
    z1 = x @ W1 + b1
    expr1 = f"1×({W1[0,0]},{W1[0,1]}) + 2×({W1[1,0]},{W1[1,1]})"
    step("z1 = x·W1 + b1", expr1, np.round(z1.ravel(), 3))
    h = np.maximum(z1, 0)
    step("h = ReLU(z1)", f"负数变 0（本行 {z1.ravel()} → {h.ravel()}）", np.round(h.ravel(), 3))
    z2 = h @ W2 + b2
    expr2 = f"{h[0,0]:.1f}×{W2[0,0]} + {h[0,1]:.1f}×{W2[1,0]} + {b2[0]}"
    step("z2 = h·W2 + b2", expr2, np.round(z2.ravel(), 3))
    p = z2.copy()                       # 回归任务用 MSE，省掉 softmax
    step("预测 p = z2", "", np.round(p.ravel(), 3))

    diff = p - y
    loss = float((diff ** 2).mean())
    print("\n  —— 损失 ——")
    step("diff = p − y", f"{p[0,0]:.1f} − {y[0,0]}", np.round(diff.ravel(), 3))
    step("loss = diff²", "", round(loss, 4))

    print("\n  —— 反向（每一步都是「局部导数 × 后面传回来的梯度」）——")
    d_p = 2 * diff / len(diff)             # dL/dp
    step("dL/dp = 2(p−y)/n", f"2×{diff[0,0]:.1f}/{len(diff)}", np.round(d_p.ravel(), 3))
    dW2 = h.T @ d_p
    step("dW2 = h.T·dp", f"{h.ravel()}×{d_p[0,0]:.2f}", np.round(dW2.ravel(), 3), "← 形状 (2,1)，和 W2 一致")
    db2 = d_p.sum(axis=0)
    step("db2 = Σdp", "", np.round(db2, 3))
    dh = d_p @ W2.T
    step("dh = dp·W2.T", f"{d_p[0,0]:.2f}×{W2.ravel()}", np.round(dh.ravel(), 3))
    dh[z1 <= 0] = 0
    step("dh[z1≤0] = 0", "ReLU 反向", np.round(dh.ravel(), 3), "被砍掉的神经元不传梯度")
    dW1 = x.T @ dh
    step("dW1 = x.T·dh", f"[{x.ravel()}]外积{dh.ravel()}", np.round(dW1.ravel(), 3), "← (2,1)@(1,2)=(2,2)")
    db1 = dh.sum(axis=0)
    step("db1 = Σdh", "", np.round(db1, 3))

    print("\n  三个必须理解的细节：")
    print("    1) 每个 dW 的形状都等于对应 W 的形状 —— 形状对上，公式基本就对了")
    print("    2) 转置出现的原因：把「(样本,输出)」的梯度变成「(输入,输出)」的权重梯度")
    print("    3) ReLU 的 mask 不能漏：漏了梯度会错，但程序不报错，只是学得慢")

    # 图：前向/反向数字（数值全部取自上面算出来的结果）
    fig, ax = plt.subplots(figsize=(10.4, 4.4))
    ax.set_xlim(0, 10.4); ax.set_ylim(0, 4.6); ax.axis("off")

    pos = [(1.2, 3.3), (4.6, 3.3), (8.2, 3.3)]
    texts = [f"x\n({x[0,0]:.0f}, {x[0,1]:.0f})",
             f"h = ReLU(z1)\n({h[0,0]:.1f}, {h[0,1]:.1f})",
             f"p = z2\n{p[0,0]:.1f}"]
    for (x0, y0), t in zip(pos, texts):
        ax.add_patch(plt.Rectangle((x0 - 0.85, y0 - 0.55), 1.7, 1.1, fc="white", ec=BLUE, lw=1.6))
        ax.text(x0, y0, t, ha="center", va="center", fontsize=9, color=INK, linespacing=1.6)

    for i in range(2):
        ax.annotate("", xy=(pos[i + 1][0] - 0.9, 3.3), xytext=(pos[i][0] + 0.9, 3.3),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6))
    ax.text(2.9, 3.62, "×W1 + b1", fontsize=8.5, color=BLUE, ha="center")
    ax.text(6.4, 3.62, "×W2 + b2", fontsize=8.5, color=BLUE, ha="center")

    # 反向：箭头在下方，文字在箭头再下方，避免重叠
    for i in range(2):
        ax.annotate("", xy=(pos[i][0] + 0.05, 2.15), xytext=(pos[i + 1][0] - 0.05, 2.15),
                    arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6))
    ax.text(2.9, 1.55, "dW1 = x^T·dh   (2,2)\ndb1 = Σdh", fontsize=8.5, color=RED,
            ha="center", va="top", linespacing=1.6)
    ax.text(6.4, 1.55, "dW2 = h^T·dp   (2,1)\ndb2 = Σdp", fontsize=8.5, color=RED,
            ha="center", va="top", linespacing=1.6)
    ax.text(5.2, 0.55, "前向用矩阵乘；反向也用矩阵乘，只是转置位置变了 —— 这就是「反向传播」的全部",
            ha="center", fontsize=9, color=GRAY)
    save(fig, "F5_backprop.png")


# ============================================================ F6 特征值
def f6():
    title("F6 特征值方程 Av = λv：为什么 (A−λI)v=0 能解出 λ")
    A = np.array([[2.0, 1.0],
                  [1.0, 2.0]])
    print("  A = [[2,1],[1,2]]\n")
    print("  方程：A v = λ v  →  (A − λI) v = 0")
    print("  关键推理：v 不能是零向量，所以 (A − λI) 必须「不可逆」")
    print("            不可逆 ⟺ det(A − λI) = 0   ← 特征多项式\n")

    lam = 1.0
    M = A - lam * np.eye(2)
    step(1, "取 λ=1，A − 1·I", np.array2string(M, prefix=""))
    step(2, "det(A−λI)", "(2−1)(2−1) − 1×1", round(float(np.linalg.det(M)), 6))
    print("    det = 0 ✓ → λ=1 是特征值")

    vals, vecs = np.linalg.eigh(A)
    for i in range(2):
        step(3 + i, f"代入 λ={vals[i]:.0f} 解 (A−λI)v=0", "取 v 的一个解",
             np.round(vecs[:, i], 3))
    print("\n  验证 A v = λ v：")
    for i in range(2):
        v = vecs[:, i]
        print(f"    A·v = {np.round(A @ v, 3)}   λ·v = {np.round(vals[i] * v, 3)}  "
              f"{'✅' if np.allclose(A @ v, vals[i] * v) else '❌'}")
    print("\n  直觉：λ 是「沿这个方向拉伸几倍」，v 是「哪个方向不被拧歪」")
    print("  对称矩阵的特征向量互相垂直 —— 这正是 PCA 能用来找正交主方向的原因")


# ============================================================ F7 SVD
def f7():
    title("F7 SVD 的 k 秩近似：为什么丢掉小奇异值几乎不损失信息")
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((40, 30)) @ np.diag(np.linspace(10, 0.1, 30))).round(2)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    print(f"  A 是 40×30 的矩阵，做完 SVD：A = U Σ Vᵀ")
    print(f"  前 5 个奇异值：{np.round(S[:5], 2)}")
    print(f"  最后 5 个奇异值：{np.round(S[-5:], 3)}\n")

    print("  逐步看 k 秩近似的误差：")
    print("  " + "-" * 56)
    print(f"  {'k':>3}   {'保留能量':>10}   {'相对误差':>10}   {'存储量':>10}")
    for k in [1, 2, 5, 10, 20, 30]:
        Ak = (U[:, :k] * S[:k]) @ Vt[:k, :]
        energy = S[:k].sum() / S.sum()
        err = np.linalg.norm(A - Ak) / np.linalg.norm(A)
        print(f"  {k:>3}   {energy*100:>9.1f}%   {err*100:>9.1f}%   {k*(40+30+1):>10}")
    print(f"  （原矩阵需要存 {40*30} 个数）")

    print("\n  原理：A = Σᵢ σᵢ·uᵢvᵢᵀ —— 每个奇异值贡献一层「图案」")
    print("        σ 大的层是主要结构，σ 小的层是细节/噪声")
    print("        丢掉 σ 小的层 = 丢掉细节，主结构还在")
    print("\n  ⚠️ 注意：存储量不是越小越好 —— k=10 时存 710 个数，而原矩阵 1200 个，才省 40%")
    print("     只有当矩阵很大且奇异值掉得很快时，压缩才划算")


# ============================================================ F8 最小二乘
def f8():
    title("F8 最小二乘正规方程：为什么 x = (AᵀA)⁻¹Aᵀb")
    A = np.array([[0.0, 1.0],
                  [1.0, 1.0],
                  [2.0, 1.0]])
    b = np.array([1.0, 3.0, 2.0])

    print("  3 个点要在 y = a·x + c 上拟合：")
    print(f"    A（设计矩阵）= \n{np.array2string(A, prefix='    ')}")
    print(f"    b = {b}\n")

    print("  目标：找 x 让 ||A x − b||² 最小")
    print("  几何：把 b 投影到 A 的列空间上，投影点就是 Ax，残差垂直于列空间\n")

    AtA = A.T @ A
    Atb = A.T @ b
    step(1, "AᵀA", np.array2string(AtA, prefix=""))
    step(2, "Aᵀb", np.round(Atb, 3))
    step(3, "(AᵀA)⁻¹", np.array2string(np.linalg.inv(AtA), prefix=""))
    x = np.linalg.inv(AtA) @ Atb
    step(4, "x = (AᵀA)⁻¹Aᵀb", np.round(x, 4), "← a=?, c=?")
    print(f"    → 拟合直线：y = {x[0]:.2f}x + {x[1]:.2f}")

    pred = A @ x
    res = b - pred
    step(5, "预测 Ax", np.round(pred, 3))
    step(6, "残差 b − Ax", np.round(res, 3), "注意：残差和 A 的每一列都垂直")
    step(7, "残差平方和", round(float((res ** 2).sum()), 4))
    print(f"\n  验证「残差 ⊥ 列空间」：Aᵀ·res = {np.round(A.T @ res, 10)}（应为 0）")
    print("  这就是 Aᵀ(b − Ax) = 0 → AᵀAx = Aᵀb → 正规方程的由来")

    # 图：投影示意
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.2, 4.2)
    ax.scatter(A[:, 0], b, s=70, color=BLUE, zorder=3, label="观测点")
    xs = np.linspace(-0.3, 2.4, 40)
    ax.plot(xs, x[0] * xs + x[1], color=RED, lw=2, label=f"拟合 y={x[0]:.2f}x+{x[1]:.2f}")
    for xi, yi, pi in zip(A[:, 0], b, pred):
        ax.plot([xi, xi], [yi, pi], "--", color=GRAY, lw=1.2)
    ax.set_xlabel("x", fontsize=10, color=INK); ax.set_ylabel("y", fontsize=10, color=INK)
    ax.set_title("最小二乘 = 让所有虚线（残差）的平方和最小", fontsize=11, color=INK)
    ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "F8_lstsq.png")


# ============================================================ F9 PCA
def f9():
    title("F9 PCA 五步：每一步的数字都看得见")
    rng = np.random.default_rng(3)
    X = rng.standard_normal((5, 2)) * np.array([3.0, 0.8]) + np.array([5.0, 2.0])
    print(f"  5 个二维点：\n{np.round(X, 2)}")

    step(1, "均值 μ", np.round(X.mean(axis=0), 3))
    Xc = X - X.mean(axis=0)
    step(2, "中心化 Xc = X − μ", "每列减均值", np.round(Xc.mean(axis=0), 6))
    C = Xc.T @ Xc / (len(X) - 1)
    step(3, "协方差 C = XcᵀXc/(n−1)", np.array2string(C, prefix=""))
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    step(4, "特征值 λ", "方差大小", np.round(vals, 4))
    step(5, "第一主方向 v1", np.round(vecs[:, 0], 4))
    ratio = vals[0] / vals.sum()
    step(6, "解释方差比 λ1/Σλ", f"{ratio*100:.2f}%")
    proj = Xc @ vecs[:, 0]
    step(7, "投影 z = Xc·v1（5 个数）", np.round(proj, 3), "← 二维变一维")
    recon = np.outer(proj, vecs[:, 0]) + X.mean(axis=0)
    step(8, "还原 = z·v1ᵀ + μ", "见下方误差")
    err = np.linalg.norm(X - recon, axis=1).mean()
    step(9, "平均重建误差", round(float(err), 4))
    print("\n  每一步都在做同一件事：把「斜着的方差」转到坐标轴上，再丢掉小的那一维")


# ============================================================ F10 einsum
def f10():
    title("F10 einsum 字母规则：三句话看懂所有批量运算")
    print("  规则 1：字母表示维度，同名 = 同一维")
    print("  规则 2：出现在输出 → 保留；不出现在输出 → 沿它求和")
    print("  规则 3：顺序无所谓，但输出顺序决定结果摆放顺序\n")

    A = np.arange(6).reshape(2, 3)
    Bm = np.arange(12).reshape(3, 4)
    print("  例 1：'ik,kj->ij'  = 普通矩阵乘")
    print(f"    A{A.shape} @ B{Bm.shape} → {np.einsum('ik,kj->ij', A, Bm).shape}"
          f"（k 不在输出里，被求和掉）")

    T = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    W = np.arange(4 * 5).reshape(4, 5)
    print("\n  例 2：'bik,kj->bij'  = 批量矩阵乘")
    print(f"    T{T.shape} @ W{W.shape} → {np.einsum('bik,kj->bij', T, W).shape}"
          "（b 保留，k 求和）")

    q = np.random.default_rng(0).random((2, 3, 4))
    k = np.random.default_rng(1).random((2, 5, 4))
    print("\n  例 3：'bqd,bkd->bqk'  = 注意力打分（QKᵀ）")
    print(f"    q{q.shape}, k{k.shape} → {np.einsum('bqd,bkd->bqk', q, k).shape}"
          "（d 求和 = 点积；输出留下 query×key）")

    print("\n  口诀：把 einsum 当「乘法口诀表」读 ——")
    print("    两边都有的字母 = 要相乘并求和的维度（内积）")
    print("    只在输出里的字母 = 新的结果维度")


ALL = {"1": f1, "2": f2, "3": f3, "4": f4, "5": f5,
       "6": f6, "7": f7, "8": f8, "9": f9, "10": f10}

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target and target in ALL:
        ALL[target]()
    else:
        for k in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]:
            ALL[k]()
        print("\n" + "=" * 66)
        print("全部公式都跑完了。建议：对着输出，用纸笔把 F1/F2/F3/F5 各重算一遍。")
        print("=" * 66)
