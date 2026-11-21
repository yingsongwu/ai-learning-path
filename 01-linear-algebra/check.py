# -*- coding: utf-8 -*-
"""
01 单元自测：14 道题（入门 10 + 深入 4），≥12 道算过

用法：
    python check.py            # 交互答题
    python check.py --answers  # 直接看答案（先自己做完再看）

题目都是随机生成的，每次不一样，可以反复练。
"""
import sys
import numpy as np

rng = np.random.default_rng()
SHOW = "--answers" in sys.argv


def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def q_shape():
    a, b, c = rng.integers(2, 9, 3)
    return (f"矩阵 (n,{a}) 乘 ({a},{c}) 的结果形状是？写成 a,b 的形式",
            f"n,{c}&{a},{c}")


def q_dot():
    u = rng.integers(-3, 4, 3)
    v = rng.integers(-3, 4, 3)
    return (f"求点积：u={u.tolist()}，v={v.tolist()}",
            str(int(u @ v)) + "&点积=逐项相乘再相加")


def q_transpose():
    a, b = rng.integers(2, 7, 2)
    return (f"矩阵 ({a},{b}) 转置后形状？写成 a,b",
            f"{b},{a}&行变列、列变行")


def q_rank():
    k = rng.integers(2, 5)
    return (f"向量 v1=(1,2)，v2={k}*(1,2)，两个向量组成的矩阵的秩是？", "1&同方向，秩=1")


def q_cos():
    return ("(1,0) 和 (0,1) 的余弦相似度是多少？", "0&垂直，点积=0，余弦=0")


def q_det():
    s = rng.integers(2, 6)
    return (f"矩阵 [[{s},0],[0,{s}]] 的行列式是？", f"{s*s}&对角矩阵，det=对角元乘积")


def q_norm():
    v = rng.integers(-4, 5, 2)
    return (f"向量 {v.tolist()} 的 L2 范数（保留 3 位小数）",
            f"{np.linalg.norm(v):.3f}&sqrt(x^2+y^2)")


def q_eig():
    a = rng.integers(2, 6)
    return (f"对角矩阵 [[{a},0],[0,1]] 较大的特征值是？", f"{a}&对角元就是特征值")


def q_mult_elem():
    a, b = rng.integers(-3, 4, 2)
    return (f"A=[[1,2],[3,4]]，B=[[{a}],[{b}]]，A@B 的第一个元素是？",
            f"{(1*a + 2*b)}&1*{a}+2*{b}")


def q_nn():
    return ("batch=16，x 是 (16,256)，W1 是 (256,64)，ReLU 后 h 的形状？（写成 b,d）",
            "16,64&内维 256 被消掉，batch 保留")


def q_l1():
    v = rng.integers(1, 6, 3)
    return (f"向量 {v.tolist()} 的 L1 范数是？", f"{int(np.abs(v).sum())}&各分量绝对值之和")


def q_inv():
    s = rng.integers(2, 5)
    return (f"矩阵 [[{s},1],[{2*s},2]] 可逆吗？（可逆答 1，不可逆答 0）",
            "0&两行成比例，det=0，秩1")


def q_einsum():
    return ("einsum('bqd,bkd->bqk')，若 q 是 (2,7,16)、k 是 (2,9,16)，输出形状是？（写成 b,q,k）",
            "2,7,9&d 被求和掉，batch 保留")


def q_pca():
    return ("PCA 后第一主成分解释了 90% 的方差，压成 1 维会丢掉多少比例的方差？（填 0.1 这种小数）",
            "0.1&1-0.9=0.1")


QUESTIONS = [q_shape, q_dot, q_transpose, q_rank, q_cos,
             q_det, q_norm, q_eig, q_mult_elem, q_nn,
             q_l1, q_inv, q_einsum, q_pca]

print("=" * 60)
print("01 线性代数 · 自测（14 题：入门 10 + 深入 4）")
print("=" * 60)

score = 0
for i, gen in enumerate(QUESTIONS, 1):
    text, pack = gen()
    ans, why = pack.split("&", 1)
    print(f"\n第 {i} 题：{text}")
    if SHOW:
        print(f"   ✅ 答案：{ans}   （{why}）")
        score += 1
        continue
    user = ask("  你的答案：")
    ok = False
    if "," in ans:      # 形状题，宽松比较
        norm = lambda s: s.replace(" ", "").replace("(", "").replace(")", "")
        ok = norm(user) == norm(ans)
    else:
        try:
            ok = abs(float(user) - float(ans)) < 1e-2
        except ValueError:
            ok = False
    if ok:
        score += 1
        print("  ✅ 正确")
    else:
        print(f"  ❌ 正确答案是 {ans}（{why}）")

print("\n" + "=" * 60)
total = len(QUESTIONS)
print(f"得分：{score}/{total} —— " + ("过关 🎉 可以进 02 单元了"
                                  if score >= int(total * 0.8) else "再回看一遍对应课程"))
print("=" * 60)
