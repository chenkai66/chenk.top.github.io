"""
Figures for Recommendation Systems Part 4: CTR Prediction.

Generates 6 production-quality figures and saves them to BOTH:
  - source/_posts/en/recommendation-systems/04-ctr-prediction/
  - source/_posts/zh/recommendation-systems/04-CTR预估与点击率建模/

Run:
    python 04-ctr-prediction.py
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# ----------------------------- Style ----------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GRAY = "#6b7280"
LIGHT = "#e5e7eb"
DARK = "#111827"

PALETTE = [BLUE, PURPLE, GREEN, ORANGE]

# ----------------------------- Output paths ---------------------------------

ROOT = Path(__file__).resolve().parents[2].parent  # chenk-site/
EN_DIR = ROOT / "source/_posts/en/recommendation-systems/04-ctr-prediction"
ZH_DIR = ROOT / "source/_posts/zh/recommendation-systems/04-CTR预估与点击率建模"
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    """Save figure to both EN and ZH asset directories."""
    for d in (EN_DIR, ZH_DIR):
        path = d / name
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"  saved -> {path}")
    plt.close(fig)


# ----------------------------- Figure 1 -------------------------------------
# LR limitation: linear boundary fails on non-linear (XOR-like) CTR pattern.

def fig1_lr_limitation() -> None:
    rng = np.random.default_rng(42)

    # Create XOR-like 2D pattern: clicks happen when (young & action) or (old & comedy)
    n = 220
    pts = rng.uniform(-1, 1, size=(n, 2))
    # XOR rule
    labels = ((pts[:, 0] > 0) ^ (pts[:, 1] > 0)).astype(int)
    # Add a touch of noise to look realistic
    flip = rng.uniform(0, 1, size=n) < 0.05
    labels = np.where(flip, 1 - labels, labels)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # ---- Left: data with linear LR boundary ----
    ax = axes[0]
    ax.scatter(pts[labels == 1, 0], pts[labels == 1, 1],
               c=BLUE, s=42, edgecolor="white", linewidth=0.7,
               label="Click (y=1)", zorder=3)
    ax.scatter(pts[labels == 0, 0], pts[labels == 0, 1],
               c=ORANGE, s=42, edgecolor="white", linewidth=0.7,
               marker="X", label="No click (y=0)", zorder=3)

    # An LR boundary fit (just a line) -- show it cannot separate XOR
    xs = np.linspace(-1.05, 1.05, 100)
    ax.plot(xs, 0.18 * xs + 0.05, color=DARK, linewidth=2.2,
            linestyle="--", label="LR decision boundary", zorder=4)

    ax.set_title("Logistic Regression: a single hyperplane")
    ax.set_xlabel("Feature 1 (e.g. user age, normalised)")
    ax.set_ylabel("Feature 2 (e.g. item category embedding)")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.text(-1.05, -1.0, "AUC ≈ 0.52  (~ random)",
            fontsize=10, color=DARK,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fef3c7",
                      ec=ORANGE, lw=1.0))

    # ---- Right: same data, non-linear boundary works ----
    ax = axes[1]
    ax.scatter(pts[labels == 1, 0], pts[labels == 1, 1],
               c=BLUE, s=42, edgecolor="white", linewidth=0.7,
               label="Click (y=1)", zorder=3)
    ax.scatter(pts[labels == 0, 0], pts[labels == 0, 1],
               c=ORANGE, s=42, edgecolor="white", linewidth=0.7,
               marker="X", label="No click (y=0)", zorder=3)

    # Non-linear boundary: x*y = 0
    xx, yy = np.meshgrid(np.linspace(-1.05, 1.05, 200),
                         np.linspace(-1.05, 1.05, 200))
    zz = xx * yy
    ax.contour(xx, yy, zz, levels=[0], colors=[GREEN],
               linewidths=2.2, linestyles="-")

    # Shade regions
    ax.contourf(xx, yy, zz, levels=[-2, 0], colors=[ORANGE], alpha=0.08)
    ax.contourf(xx, yy, zz, levels=[0, 2], colors=[BLUE], alpha=0.08)

    ax.set_title("With feature interaction (x₁·x₂)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.plot([], [], color=GREEN, linewidth=2.2,
            label="FM / DeepFM boundary")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.text(-1.05, -1.0, "AUC ≈ 0.93  (captures XOR)",
            fontsize=10, color=DARK,
            bbox=dict(boxstyle="round,pad=0.35", fc="#dcfce7",
                      ec=GREEN, lw=1.0))

    fig.suptitle("Why LR is not enough: feature interactions matter",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_lr_limitation.png")


# ----------------------------- Figure 2 -------------------------------------
# Feature interaction methods comparison.

def fig2_interaction_methods() -> None:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.6))

    methods = [
        ("FM",       "⟨vᵢ, vⱼ⟩",          BLUE,
         "shared\nembedding"),
        ("FFM",      "⟨vᵢ,fⱼ , vⱼ,fᵢ⟩",   PURPLE,
         "field-aware\nembedding"),
        ("DeepFM",   "FM ⊕ MLP",          GREEN,
         "explicit pairwise\n+ implicit deep"),
        ("DCN",      "x₀·(wᵀxₗ) + xₗ",    ORANGE,
         "bounded-degree\ncross"),
        ("AutoInt",  "softmax(QKᵀ/√d)·V", "#ef4444",
         "multi-head\nself-attention"),
    ]

    for ax, (name, formula, color, sub) in zip(axes, methods):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Title chip
        chip = FancyBboxPatch((0.1, 0.82), 0.8, 0.13,
                              boxstyle="round,pad=0.02,rounding_size=0.04",
                              fc=color, ec=color, alpha=0.95)
        ax.add_patch(chip)
        ax.text(0.5, 0.885, name, ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")

        # Formula box
        ax.add_patch(FancyBboxPatch(
            (0.05, 0.42), 0.9, 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            fc="white", ec=color, lw=1.6))
        ax.text(0.5, 0.58, formula, ha="center", va="center",
                fontsize=11, fontweight="bold", color=DARK,
                family="monospace")

        # Sub description
        ax.text(0.5, 0.25, sub, ha="center", va="center",
                fontsize=9.5, color=GRAY)

        # Order/expressiveness bar
        order = {"FM": 2, "FFM": 2, "DeepFM": 4, "DCN": 4, "AutoInt": 5}[name]
        ax.text(0.5, 0.08,
                f"interactions: {'★' * order}{'☆' * (5 - order)}",
                ha="center", va="center", fontsize=9, color=color,
                fontweight="bold")

    fig.suptitle("Feature interaction mechanisms across CTR models",
                 fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "fig2_interaction_methods.png")


# ----------------------------- Figure 3 -------------------------------------
# DeepFM architecture: parallel FM + Deep with shared embeddings.

def fig3_deepfm_arch() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ---- Sparse input row (bottom) ----
    n_fields = 5
    field_names = ["user_id", "age", "item_id", "category", "device"]
    field_x = np.linspace(1.0, 11.0, n_fields)
    for x, name in zip(field_x, field_names):
        ax.add_patch(FancyBboxPatch((x - 0.55, 0.2), 1.1, 0.6,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=LIGHT, ec=GRAY, lw=1.0))
        ax.text(x, 0.5, name, ha="center", va="center",
                fontsize=9, color=DARK)

    ax.text(6.0, -0.05, "Sparse one-hot input  x",
            ha="center", fontsize=10, color=GRAY, style="italic")

    # ---- Shared embedding layer ----
    emb_y = 1.7
    for x in field_x:
        ax.add_patch(FancyBboxPatch((x - 0.45, emb_y), 0.9, 0.55,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=BLUE, ec=BLUE, alpha=0.85))
        ax.text(x, emb_y + 0.275, "vᵢ", ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        # arrow from input to embedding
        ax.annotate("", xy=(x, emb_y), xytext=(x, 0.85),
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.9))

    ax.text(0.4, emb_y + 0.275, "Shared\nEmbedding", ha="center",
            va="center", fontsize=9.5, color=BLUE, fontweight="bold")

    # ---- Two parallel branches ----
    # FM branch (left)
    fm_box = FancyBboxPatch((0.8, 3.4), 4.6, 1.6,
                            boxstyle="round,pad=0.04,rounding_size=0.1",
                            fc="white", ec=PURPLE, lw=2.0)
    ax.add_patch(fm_box)
    ax.text(3.1, 4.65, "FM Component",
            ha="center", fontsize=12, fontweight="bold", color=PURPLE)
    ax.text(3.1, 4.10,
            "linear  +  ½[(Σvᵢxᵢ)² − Σ(vᵢxᵢ)²]",
            ha="center", fontsize=10, family="monospace", color=DARK)
    ax.text(3.1, 3.65, "explicit pairwise interactions",
            ha="center", fontsize=9, style="italic", color=GRAY)

    # Deep branch (right)
    deep_box = FancyBboxPatch((6.6, 3.4), 4.6, 1.6,
                              boxstyle="round,pad=0.04,rounding_size=0.1",
                              fc="white", ec=GREEN, lw=2.0)
    ax.add_patch(deep_box)
    ax.text(8.9, 4.65, "Deep Component",
            ha="center", fontsize=12, fontweight="bold", color=GREEN)
    ax.text(8.9, 4.10, "MLP:  ReLU → ReLU → ReLU",
            ha="center", fontsize=10, family="monospace", color=DARK)
    ax.text(8.9, 3.65, "implicit high-order interactions",
            ha="center", fontsize=9, style="italic", color=GRAY)

    # arrows from embedding layer up to each branch
    ax.annotate("", xy=(3.1, 3.4), xytext=(3.5, emb_y + 0.55),
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5))
    ax.annotate("", xy=(8.9, 3.4), xytext=(8.5, emb_y + 0.55),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))

    # ---- Sum + sigmoid ----
    sum_box = FancyBboxPatch((4.7, 5.6), 2.6, 1.0,
                             boxstyle="round,pad=0.04,rounding_size=0.1",
                             fc=ORANGE, ec=ORANGE, alpha=0.92)
    ax.add_patch(sum_box)
    ax.text(6.0, 6.1, "σ( y_FM + y_Deep )",
            ha="center", va="center", color="white",
            fontsize=12, fontweight="bold", family="monospace")

    ax.annotate("", xy=(5.6, 5.6), xytext=(3.1, 5.0),
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5))
    ax.annotate("", xy=(6.4, 5.6), xytext=(8.9, 5.0),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))

    # ---- Output ----
    out_box = FancyBboxPatch((5.0, 7.05), 2.0, 0.7,
                             boxstyle="round,pad=0.04,rounding_size=0.1",
                             fc=DARK, ec=DARK)
    ax.add_patch(out_box)
    ax.text(6.0, 7.4, "p(click)  ∈ [0, 1]",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
    ax.annotate("", xy=(6.0, 7.05), xytext=(6.0, 6.6),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5))

    ax.set_title("DeepFM: parallel FM and Deep branches with shared embeddings",
                 fontsize=14, pad=16)
    fig.tight_layout()
    save(fig, "fig3_deepfm_arch.png")


# ----------------------------- Figure 4 -------------------------------------
# DCN cross network: how a cross layer builds higher-order interactions.

def fig4_dcn_cross() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- Left: layer-by-layer schematic ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    layers = [
        (1.0, "x₀", "raw embedding\n(degree 1)", BLUE),
        (3.5, "x₁", "x₀·(wᵀx₀) + x₀\n(degree 2)", PURPLE),
        (6.0, "x₂", "x₀·(wᵀx₁) + x₁\n(degree 3)", GREEN),
        (8.5, "x₃", "x₀·(wᵀx₂) + x₂\n(degree 4)", ORANGE),
    ]
    for x, name, desc, color in layers:
        ax.add_patch(FancyBboxPatch((x - 0.7, 3.4), 1.4, 1.6,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc="white", ec=color, lw=2.0))
        ax.text(x, 4.65, name, ha="center", va="center",
                fontsize=14, fontweight="bold", color=color)
        ax.text(x, 3.85, desc, ha="center", va="center",
                fontsize=8.5, color=DARK)

    # arrows between layers
    for i in range(3):
        x_from, x_to = layers[i][0] + 0.7, layers[i + 1][0] - 0.7
        ax.annotate("", xy=(x_to, 4.2), xytext=(x_from, 4.2),
                    arrowprops=dict(arrowstyle="->", color=GRAY,
                                    lw=1.6, connectionstyle="arc3,rad=0"))

    # x0 broadcast arrows (curved, going under)
    for i in range(1, 4):
        ax.annotate("", xy=(layers[i][0], 3.4), xytext=(layers[0][0], 2.5),
                    arrowprops=dict(arrowstyle="->", color=BLUE,
                                    lw=1.0, alpha=0.55,
                                    connectionstyle="arc3,rad=-0.25"))

    ax.text(layers[0][0], 2.2, "x₀ injected into every cross layer",
            ha="center", fontsize=9, color=BLUE, style="italic")

    ax.text(5.0, 6.3, "Cross Network: each layer adds one degree",
            ha="center", fontsize=12.5, fontweight="bold", color=DARK)
    ax.text(5.0, 5.7,
            "After L cross layers → polynomial interactions up to degree L+1",
            ha="center", fontsize=10, color=GRAY, style="italic")

    # ---- Right: parameter cost vs polynomial order (log scale) ----
    ax = axes[1]
    d = 100  # input dim
    orders = np.arange(1, 7)
    naive = d ** orders          # explicit polynomial expansion
    cross = d * orders           # DCN: linear in d per layer

    ax.semilogy(orders, naive, "o-", color=ORANGE, linewidth=2.4,
                markersize=8, label="Explicit polynomial: O(dᵏ)")
    ax.semilogy(orders, cross, "s-", color=BLUE, linewidth=2.4,
                markersize=8, label="DCN cross net: O(d·k)")

    ax.set_xlabel("Interaction order (k)")
    ax.set_ylabel("Parameter count (log scale,  d = 100)")
    ax.set_title("Parameter efficiency of the cross network")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_xticks(orders)

    # Annotate gap
    ax.annotate(f"{naive[-1] / cross[-1]:,.0f}× fewer parameters\nat order 6",
                xy=(orders[-1], cross[-1]),
                xytext=(orders[-1] - 1.6, cross[-1] * 0.04),
                fontsize=10, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))

    fig.suptitle("DCN: efficient bounded-degree feature crossing",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_dcn_cross.png")


# ----------------------------- Figure 5 -------------------------------------
# AUC / Logloss comparison across CTR models.

def fig5_auc_logloss() -> None:
    # Numbers reflect typical relative ordering reported in DeepFM, xDeepFM,
    # AutoInt, FiBiNet papers on Criteo / Avazu. Values are illustrative.
    models = ["LR", "FM", "FFM", "DeepFM", "xDeepFM", "DCN", "AutoInt", "FiBiNet"]
    auc    = [0.7820, 0.7926, 0.7980, 0.8028, 0.8052, 0.8042, 0.8061, 0.8068]
    logloss= [0.4695, 0.4592, 0.4555, 0.4514, 0.4501, 0.4509, 0.4495, 0.4490]

    colors = [GRAY, BLUE, BLUE, PURPLE, PURPLE, GREEN, ORANGE, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # ---- AUC ----
    ax = axes[0]
    bars = ax.bar(models, auc, color=colors, edgecolor="white",
                  linewidth=1.4, alpha=0.92)
    ax.set_ylim(0.775, 0.812)
    ax.set_ylabel("AUC  (higher is better)")
    ax.set_title("AUC on Criteo-style benchmarks (illustrative)")
    ax.axhline(auc[0], color=GRAY, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(7.5, auc[0] + 0.0005, "LR baseline",
            fontsize=8.5, color=GRAY, ha="right")

    for bar, val in zip(bars, auc):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0004,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=8.5, color=DARK)

    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    # ---- Logloss ----
    ax = axes[1]
    bars = ax.bar(models, logloss, color=colors, edgecolor="white",
                  linewidth=1.4, alpha=0.92)
    ax.set_ylim(0.444, 0.473)
    ax.set_ylabel("Logloss  (lower is better)")
    ax.set_title("Logloss on Criteo-style benchmarks (illustrative)")
    ax.axhline(logloss[0], color=GRAY, linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(7.5, logloss[0] - 0.0009, "LR baseline",
            fontsize=8.5, color=GRAY, ha="right")

    for bar, val in zip(bars, logloss):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.0003,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=8.5, color=DARK)

    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.suptitle("CTR model performance: each generation closes a real gap",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_auc_logloss.png")


# ----------------------------- Figure 6 -------------------------------------
# CTR pipeline overview: data -> features -> embedding -> model -> serving.

def fig6_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    stages = [
        ("Raw Logs",       "user clicks\nimpressions\ncontext",         BLUE),
        ("Feature Eng.",   "categorical\nnumerical\ncross features",    PURPLE),
        ("Embedding",      "shared lookup\nlow-dim dense\nvectors",     GREEN),
        ("CTR Model",      "FM / DeepFM\nDCN / AutoInt\nFiBiNet",       ORANGE),
        ("Ranking & A/B",  "top-K sort\n< 10 ms p99\nonline test",      "#ef4444"),
    ]

    box_w, box_h = 2.2, 2.4
    centers = np.linspace(1.6, 12.4, len(stages))

    for cx, (name, body, color) in zip(centers, stages):
        ax.add_patch(FancyBboxPatch(
            (cx - box_w / 2, 2.0), box_w, box_h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            fc="white", ec=color, lw=2.2))
        # Title chip
        ax.add_patch(FancyBboxPatch(
            (cx - box_w / 2 + 0.05, 2.0 + box_h - 0.55),
            box_w - 0.1, 0.5,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            fc=color, ec=color))
        ax.text(cx, 2.0 + box_h - 0.3, name,
                ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(cx, 2.0 + (box_h - 0.55) / 2 + 0.05, body,
                ha="center", va="center",
                fontsize=9.5, color=DARK)

    # Arrows between stages
    for i in range(len(stages) - 1):
        x_from = centers[i] + box_w / 2
        x_to = centers[i + 1] - box_w / 2
        ax.annotate("", xy=(x_to, 3.2), xytext=(x_from, 3.2),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=2.0))

    # Feedback loop arrow (bottom)
    ax.annotate("", xy=(centers[1], 1.7), xytext=(centers[-1], 1.0),
                arrowprops=dict(arrowstyle="->", color=GRAY,
                                lw=1.4, alpha=0.7,
                                connectionstyle="arc3,rad=0.18"))
    ax.text((centers[1] + centers[-1]) / 2, 0.55,
            "online clicks → new logs → retrain (daily / hourly)",
            ha="center", fontsize=10, style="italic", color=GRAY)

    # Top label
    ax.text(7.0, 5.5, "End-to-end CTR prediction pipeline",
            ha="center", fontsize=14, fontweight="bold", color=DARK)
    ax.text(7.0, 5.05,
            "from raw user events to ranked recommendations -- with continuous feedback",
            ha="center", fontsize=10.5, color=GRAY, style="italic")

    fig.tight_layout()
    save(fig, "fig6_pipeline.png")


# ----------------------------- Main -----------------------------------------

def main() -> None:
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")
    print()
    print("Generating fig1 (LR limitation)...")
    fig1_lr_limitation()
    print("Generating fig2 (interaction methods)...")
    fig2_interaction_methods()
    print("Generating fig3 (DeepFM architecture)...")
    fig3_deepfm_arch()
    print("Generating fig4 (DCN cross network)...")
    fig4_dcn_cross()
    print("Generating fig5 (AUC / Logloss bars)...")
    fig5_auc_logloss()
    print("Generating fig6 (CTR pipeline)...")
    fig6_pipeline()
    print("\nDone.")


if __name__ == "__main__":
    main()
