"""Plot the ten-world core paired estimates and the recorded 95% intervals."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

OUT = Path(__file__).resolve().parent
CORE = OUT.parent / "run/results/core_main"
font = FontProperties(fname=str(OUT.parents[2] / "report/assets/NotoSansSC-Regular.ttf"))
points = pd.read_csv(CORE / "table1.csv").set_index(["b", "policy", "a"])
intervals = pd.read_csv(CORE / "paired_ci.csv").set_index(["b", "policy", "a", "metric"])
fig, panels = plt.subplots(1, 3, figsize=(11.2, 3.25), sharey=True)
for ax, budget, name in zip(panels, "TML", ["紧档 · 3 秒", "中档 · 4.5 秒", "松档 · 9 秒"]):
    ax.axvline(0, color="#576474", linewidth=.9)
    ax.set_xlim(-375, 270)
    ax.set_ylim(-.65, 3.65)
    for y, policy in enumerate(["both_tuned", "confirmation_common", "dev_5pct", "native"]):
        if (budget, policy, "G8P") not in points.index:
            ax.text(-345, y, "四方案无共同风险点", fontproperties=font, fontsize=9, va="center", color="#586372")
            continue
        delta = points.loc[budget, policy, "G8P"].mbps - points.loc[budget, policy, "Cnative"].mbps
        row = intervals.loc[budget, policy, "G8P", "mbps"]
        color = "#126b62" if delta > 0 else "#a24e2a"
        ax.hlines(y, row.lo95, row.hi95, colors=color, linewidth=2.4)
        ax.plot(delta, y, "o", color=color, markersize=6)
    ax.set_title(name, fontproperties=font, fontsize=13)
    ax.set_xticks([-300, -150, 0, 150])
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e5e7eb")
    ax.spines[["top", "right", "left"]].set_visible(False)
panels[0].set_yticks(range(4), ["双方均调管", "参照原管", "开发冻结", "原倍率"], fontproperties=font)
fig.supxlabel("G8P 减 Cnative 的全站净吞吐（Mbps）", fontproperties=font, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "throughput-delta.png", dpi=185, bbox_inches="tight", facecolor="white")
