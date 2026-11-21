# -*- coding: utf-8 -*-
"""公共绘图设置：中文字体 + 统一的配色和保存路径。"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

INK, BLUE, GREEN, RED, ORANGE, GRAY = "#22303f", "#2471a3", "#1e8449", "#c0392b", "#b9770e", "#8b98a5"

_CANDIDATES = [
    "/system/fonts/HarmonyOS_Sans_SC.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
for path in _CANDIDATES:
    if os.path.exists(path):
        try:
            fm.fontManager.addfont(path)
            matplotlib.rcParams["font.sans-serif"] = [fm.FontProperties(fname=path).get_name()]
        except Exception:
            pass
        break
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 130


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  图已保存: figures/{name}")
