#!/usr/bin/env python3
"""Terraform 工作流 — 垂直步骤流 (ZH)."""
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patheffects import withSimplePatchShadow

# --- tokens ---
BG = "#fdfcf9"
RED, AMBER, PURPLE, BLUE, GREEN = "#e85d4a", "#f5a834", "#8b5cf6", "#3b82f6", "#10b981"
ARROW_C = "#9ca3af"
SHADOW = dict(offset=(1.2, -1.2), shadow_rgbFace="#000000", alpha=0.18)
BOX_KW = dict(boxstyle="round,pad=0.5,rounding_size=1.5", ec="none")

fig, ax = plt.subplots(figsize=(11, 8))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 11)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Terraform 工作流", fontsize=14, fontweight="bold", pad=12, color="#1e293b")

steps = [
    (6.8, PURPLE, "HCL 文件",            "基础设施即代码 (.tf 定义)"),
    (5.4, BLUE,   "terraform init",      "下载 Provider 和模块"),
    (4.0, AMBER,  "terraform plan",      "对比期望状态与实际状态的差异"),
    (2.6, RED,    "terraform apply",     "调用云 API 收敛到期望状态"),
    (1.2, GREEN,  "terraform.tfstate",   "Terraform 记录的资源现状"),
]

bx, bw, bh = 2.2, 3.6, 0.7
for y, color, label, desc in steps:
    box = FancyBboxPatch((bx, y - bh / 2), bw, bh, facecolor=color, **BOX_KW)
    box.set_path_effects([withSimplePatchShadow(**SHADOW)])
    ax.add_patch(box)
    ax.text(bx + bw / 2, y, label, ha="center", va="center",
            fontsize=12, fontweight="bold", color="white")
    ax.text(bx + bw + 0.5, y, desc, ha="left", va="center",
            fontsize=10.5, color="#475569")

for i in range(len(steps) - 1):
    y_start = steps[i][0] - bh / 2 - 0.05
    y_end = steps[i + 1][0] + bh / 2 + 0.05
    ax.annotate("", xy=(bx + bw / 2, y_end), xytext=(bx + bw / 2, y_start),
                arrowprops=dict(arrowstyle="-|>", color=ARROW_C, lw=2,
                                mutation_scale=18))

fig.savefig("fig_terraform_pipeline_zh.png", dpi=160, bbox_inches="tight",
            facecolor=BG, pad_inches=0.1)
plt.close()
print("saved fig_terraform_pipeline_zh.png")
