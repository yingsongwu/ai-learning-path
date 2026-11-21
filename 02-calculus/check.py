# -*- coding: utf-8 -*-
"""
02 微积分 · 自测（12 题，≥10 题过关）

用法：python check.py            # 交互答题
      python check.py --answers  # 看答案
"""
import sys
import numpy as np

rng = np.random.default_rng()
SHOW = "--answers" in sys.argv


def ask(p):
    try:
        return input(p).strip()
    except EOFError:
        return ""


def q1():
    a = int(rng.integers(2, 6))
    return (f"f(x)=x²，f'({a}) = ?", f"{2*a}&幂法则 2x")


def q2():
    n, a = int(rng.integers(3, 5)), int(rng.integers(2, 4))
    return (f"f(x)=x^{n}，f'({a}) = ?", f"{n*a**(n-1)}&n·x^(n-1)")


def q3():
    a, b, x = int(rng.integers(2, 4)), int(rng.integers(1, 4)), 1
    val = 2 * (a * x + b) * a
    return (f"y=({a}x+{b})²，x={x} 处的导数 = ?", f"{val}&2({a}x+{b})·{a}")


def q4():
    c, y = int(rng.integers(2, 5)), int(rng.integers(1, 4))
    return (f"f=x²+{c}y²，∂f/∂y 在 y={y} 处 = ?", f"{2*c*y}&2·{c}·y")


def q5():
    x, y = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    return (f"f=x²+y²，在 ({x},{y}) 处梯度的长度 = ?（保留 2 位小数）",
            f"{np.hypot(2*x, 2*y):.2f}&√((2x)²+(2y)²)")


def q6():
    return ("加法节点 p=u+b 的 ∂p/∂u = ?", "1&加法把梯度原样传下去")


def q7():
    dp, x = int(rng.integers(2, 6)), int(rng.integers(2, 5))
    return (f"dL/du={dp}，x={x}，u=w·x，dL/dw = ?", f"{dp*x}&dL/du × ∂u/∂w = dL/du × x")


def q8():
    x, h = 2.0, 1e-3
    val = ((x + h) ** 2 - (x - h) ** 2) / (2 * h)
    return (f"f(x)=x²，用中心差分算 f'({x:g})（h=1e-3），结果保留 3 位 = ?",
            f"{val:.3f}&应接近 2x=4")


def q9():
    w = int(rng.integers(2, 8)); lr = round(float(rng.choice([0.1, 0.2, 0.5])), 1)
    g = int(rng.integers(1, 6))
    return (f"w={w}, lr={lr}, dL/dw={g}，更新后 w = ?（保留 1 位小数）",
            f"{w - lr*g:.1f}&w − lr×梯度")


def q10():
    return ("梯度方向指向 loss 上升还是下降？答 1=上升，0=下降", "1&所以要减梯度")


def q11():
    return ("数值梯度校验里，h 的最佳量级大约是？（答 1e-1 / 1e-5 / 1e-12 之一）",
            "1e-5&太小浮点误差，太大截断误差")


def q12():
    return ("反向传播得到的 dW 形状应该等于？答 1=和 W 相同，0=和 x 相同", "1&形状必须对齐")


QS = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12]

print("=" * 58)
print("02 微积分 · 自测（12 题）")
print("=" * 58)
score = 0
for i, g in enumerate(QS, 1):
    text, pack = g()
    ans, why = pack.split("&", 1)
    print(f"\n第 {i} 题：{text}")
    if SHOW:
        print(f"   ✅ 答案：{ans}   （{why}）")
        score += 1
        continue
    u = ask("  你的答案：")
    try:
        ok = abs(float(u) - float(ans)) < 0.02
    except ValueError:
        ok = u.lower() == ans.lower()
    if ok:
        score += 1; print("  ✅ 正确")
    else:
        print(f"  ❌ 正确答案 {ans}（{why}）")

total = len(QS)
print("\n" + "=" * 58)
print(f"得分：{score}/{total} —— " + ("过关 🎉" if score >= int(total * 0.8) else "再看一遍对应课程"))
print("=" * 58)
