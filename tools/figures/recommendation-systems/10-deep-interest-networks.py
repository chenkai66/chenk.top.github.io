"""Figures for Recommendation Systems Part 10 — Deep Interest Networks.

Generates 7 publication-quality figures explaining DIN, DIEN, DSIN, BST
and the production tricks (Dice activation, attention behavior).

Output is written to BOTH the EN and ZH asset folders so the script is
the single source of truth. Run from any working directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Style — applied once for the whole script
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

# Brand palette (must match site)
BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GRAY = "#64748b"
LIGHT = "#e5e7eb"

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/recommendation-systems/10-deep-interest-networks"
ZH_DIR = REPO / "source/_posts/zh/recommendation-systems/10-深度兴趣网络与注意力机制"


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — DIN attention weights over user history
# ---------------------------------------------------------------------------
def fig1_attention_weights() -> None:
    behaviors = [
        "Running\nshoes",
        "Phone\ncharger",
        "Hiking\nbackpack",
        "Sneakers",
        "Headphones",
        "Yoga\nmat",
        "Trail\nrunners",
        "T-shirt",
        "GPS\nwatch",
        "Water\nbottle",
    ]
    # Two candidate items produce different attention distributions
    weights_shoes = np.array([0.22, 0.02, 0.08, 0.20, 0.03, 0.04, 0.25, 0.05, 0.07, 0.04])
    weights_audio = np.array([0.04, 0.18, 0.05, 0.05, 0.40, 0.04, 0.05, 0.06, 0.08, 0.05])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    x = np.arange(len(behaviors))

    for ax, weights, title, color in zip(
        axes,
        [weights_shoes, weights_audio],
        ["Candidate: Trail running shoes", "Candidate: Wireless earbuds"],
        [BLUE, PURPLE],
    ):
        bars = ax.bar(x, weights, color=color, alpha=0.85, edgecolor="white", linewidth=1.2)
        # highlight top-3
        top3 = np.argsort(weights)[-3:]
        for i in top3:
            bars[i].set_edgecolor(ORANGE)
            bars[i].set_linewidth(2.4)
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, fontsize=8.5, rotation=0)
        ax.set_ylim(0, 0.5)
        ax.set_title(title, color=color)
        ax.set_ylabel("Attention weight" if ax is axes[0] else "")
        ax.grid(axis="y", alpha=0.4)
        for i, v in enumerate(weights):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5, color=GRAY)

    fig.suptitle(
        "Same user, same history — DIN reads it differently for each candidate",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.text(
        0.5, -0.03,
        "Orange outline = top-3 most relevant past behaviors. The model lights up shoes for shoes, audio for audio.",
        ha="center", fontsize=9, style="italic", color=GRAY,
    )
    fig.tight_layout()
    save(fig, "fig1_attention_weights.png")


# ---------------------------------------------------------------------------
# Figure 2 — DIEN architecture (GRU + AUGRU)
# ---------------------------------------------------------------------------
def fig2_dien_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    T = 5
    x_positions = np.linspace(1.2, 9.0, T)

    # Layer labels (left)
    labels = [
        ("Behavior\nembedding", 0.95),
        ("Interest\nextractor (GRU)", 2.55),
        ("Attention\nvs candidate", 4.1),
        ("Interest\nevolution (AUGRU)", 5.55),
    ]
    for text, y in labels:
        ax.text(0.05, y, text, fontsize=9, ha="left", va="center",
                fontweight="bold", color=GRAY)

    # Behavior embeddings
    for i, x in enumerate(x_positions):
        box = FancyBboxPatch((x - 0.45, 0.55), 0.9, 0.7,
                              boxstyle="round,pad=0.04", linewidth=1.4,
                              facecolor=LIGHT, edgecolor=GRAY)
        ax.add_patch(box)
        ax.text(x, 0.9, f"$e_{{b_{i+1}}}$", ha="center", va="center", fontsize=11)

    # GRU cells
    for i, x in enumerate(x_positions):
        box = FancyBboxPatch((x - 0.45, 2.2), 0.9, 0.7,
                              boxstyle="round,pad=0.04", linewidth=1.6,
                              facecolor=BLUE, edgecolor=BLUE, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, 2.55, f"$h_{i+1}$", ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        # vertical arrow from emb to GRU
        ax.annotate("", xy=(x, 2.18), xytext=(x, 1.28),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))

    # GRU horizontal arrows
    for i in range(T - 1):
        ax.annotate("", xy=(x_positions[i + 1] - 0.5, 2.55),
                    xytext=(x_positions[i] + 0.5, 2.55),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.6))

    # Candidate item (right)
    cand_x, cand_y = 11.2, 4.1
    box = FancyBboxPatch((cand_x - 0.7, cand_y - 0.45), 1.4, 0.9,
                          boxstyle="round,pad=0.05", linewidth=1.8,
                          facecolor=ORANGE, edgecolor=ORANGE, alpha=0.9)
    ax.add_patch(box)
    ax.text(cand_x, cand_y, "Candidate\n$e_i$", ha="center", va="center",
            color="white", fontsize=10.5, fontweight="bold")

    # Attention weights
    a_values = [0.10, 0.20, 0.35, 0.20, 0.15]
    for i, (x, a) in enumerate(zip(x_positions, a_values)):
        # attention node
        circ = mpatches.Circle((x, 4.1), 0.28, facecolor="white",
                                edgecolor=PURPLE, linewidth=1.6)
        ax.add_patch(circ)
        ax.text(x, 4.1, f"$a_{i+1}$", ha="center", va="center",
                fontsize=9, color=PURPLE, fontweight="bold")
        # arrow from h to attention
        ax.annotate("", xy=(x, 3.83), xytext=(x, 2.92),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))
        # arrow from candidate to attention
        ax.annotate("", xy=(x + 0.28, 4.1), xytext=(cand_x - 0.7, cand_y),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7,
                                    alpha=0.45))
        # value text below
        ax.text(x, 3.7, f"{a:.2f}", ha="center", fontsize=7.5, color=GRAY)

    # AUGRU cells
    for i, x in enumerate(x_positions):
        box = FancyBboxPatch((x - 0.45, 5.2), 0.9, 0.75,
                              boxstyle="round,pad=0.04", linewidth=1.6,
                              facecolor=GREEN, edgecolor=GREEN, alpha=0.85)
        ax.add_patch(box)
        ax.text(x, 5.575, f"$h'_{i+1}$", ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        # arrow from h to AUGRU
        ax.annotate("", xy=(x, 5.18), xytext=(x, 4.4),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.1))

    # AUGRU horizontal arrows (varying width by attention)
    for i in range(T - 1):
        ax.annotate("", xy=(x_positions[i + 1] - 0.5, 5.575),
                    xytext=(x_positions[i] + 0.5, 5.575),
                    arrowprops=dict(arrowstyle="->", color=GREEN,
                                    lw=1.0 + 4.0 * a_values[i]))

    # Final interest output
    ax.annotate("", xy=(10.5, 5.575), xytext=(x_positions[-1] + 0.5, 5.575),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=2.0))
    ax.text(11.2, 5.575, "final\ninterest", ha="center", va="center",
            fontsize=10, fontweight="bold", color=GREEN)

    # Auxiliary loss caption
    ax.text(5.0, 6.55,
            r"Auxiliary loss: $h_t$ predicts $b_{t+1}$ (vs negative) — keeps GRU honest",
            ha="center", fontsize=10, style="italic", color=PURPLE,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3e8ff",
                      edgecolor=PURPLE, linewidth=1))

    ax.set_title("DIEN — interest extractor (GRU) + interest evolution (AUGRU) toward the candidate",
                 fontsize=13, pad=14)
    fig.tight_layout()
    save(fig, "fig2_dien_architecture.png")


# ---------------------------------------------------------------------------
# Figure 3 — DSIN session split + intra/inter session attention
# ---------------------------------------------------------------------------
def fig3_dsin_sessions() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Timeline of behaviors with timestamps; gaps > 30 min split sessions
    events = [
        # (time_label, item_short, session_id)
        ("12:01", "laptop", 0),
        ("12:04", "laptop\ncase", 0),
        ("12:08", "USB-C\nhub", 0),
        ("19:30", "head-\nphones", 1),
        ("19:34", "earbuds", 1),
        ("19:39", "DAC", 1),
        ("08:15", "running\nshoes", 2),
        ("08:18", "shorts", 2),
        ("08:22", "GPS\nwatch", 2),
    ]
    session_colors = [BLUE, PURPLE, GREEN]

    n = len(events)
    xs = np.linspace(0.7, 9.0, n)
    for i, (t, label, sid) in enumerate(events):
        col = session_colors[sid]
        box = FancyBboxPatch((xs[i] - 0.42, 1.4), 0.84, 0.95,
                              boxstyle="round,pad=0.04", linewidth=1.4,
                              facecolor=col, edgecolor=col, alpha=0.85)
        ax.add_patch(box)
        ax.text(xs[i], 1.88, label, ha="center", va="center", color="white",
                fontsize=8, fontweight="bold")
        ax.text(xs[i], 1.05, t, ha="center", va="center", color=GRAY, fontsize=7.5)

    # Session brackets above
    sessions = [(0, 2), (3, 5), (6, 8)]
    session_titles = ["Session 1 — laptop accessories",
                      "Session 2 — audio gear",
                      "Session 3 — running"]
    for (a, b), title, col in zip(sessions, session_titles, session_colors):
        x_left, x_right = xs[a] - 0.5, xs[b] + 0.5
        ax.plot([x_left, x_left, x_right, x_right],
                [2.5, 2.8, 2.8, 2.5], color=col, lw=1.8)
        ax.text((x_left + x_right) / 2, 3.0, title, ha="center",
                fontsize=9, color=col, fontweight="bold")

    # Gap markers
    for (a, b) in [(2, 3), (5, 6)]:
        gx = (xs[a] + xs[b]) / 2
        ax.text(gx, 1.85, ">30 min\ngap", ha="center", va="center",
                fontsize=7.5, color=ORANGE, style="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fffbeb",
                          edgecolor=ORANGE, linewidth=0.8))
        ax.plot([gx, gx], [1.4, 2.5], color=ORANGE, lw=1.2, linestyle=":")

    # Intra-session self-attention layer
    for i, (a, b) in enumerate(sessions):
        x_mid = (xs[a] + xs[b]) / 2
        box = FancyBboxPatch((x_mid - 1.0, 3.5), 2.0, 0.7,
                              boxstyle="round,pad=0.05",
                              facecolor="white", edgecolor=session_colors[i], linewidth=2)
        ax.add_patch(box)
        ax.text(x_mid, 3.85, "Self-Attention", ha="center", va="center",
                fontsize=8.5, color=session_colors[i], fontweight="bold")
        # Down arrow
        ax.annotate("", xy=(x_mid, 3.48), xytext=(x_mid, 3.05),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

    # Bi-LSTM across sessions
    bilstm = FancyBboxPatch((1.0, 4.5), 8.5, 0.8,
                             boxstyle="round,pad=0.05", linewidth=1.6,
                             facecolor=ORANGE, edgecolor=ORANGE, alpha=0.85)
    ax.add_patch(bilstm)
    ax.text(5.25, 4.9, "Bi-LSTM across sessions   ⇄   captures inter-session interest drift",
            ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    for i, (a, b) in enumerate(sessions):
        x_mid = (xs[a] + xs[b]) / 2
        ax.annotate("", xy=(x_mid, 4.48), xytext=(x_mid, 4.22),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

    # Target attention to candidate
    cand_x = 11.5
    box = FancyBboxPatch((cand_x - 0.8, 4.55), 1.6, 0.7,
                          boxstyle="round,pad=0.05", linewidth=1.8,
                          facecolor=PURPLE, edgecolor=PURPLE, alpha=0.9)
    ax.add_patch(box)
    ax.text(cand_x, 4.9, "Candidate\n+ target attn",
            ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(cand_x - 0.8, 4.9), xytext=(9.5, 4.9),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4))

    ax.text(7.0, 5.75,
            "Behaviors → sessions → intra-session self-attention → inter-session Bi-LSTM → target attention",
            ha="center", fontsize=10, style="italic", color=GRAY)

    ax.set_title("DSIN — making session structure explicit in user history", pad=10)
    fig.tight_layout()
    save(fig, "fig3_dsin_sessions.png")


# ---------------------------------------------------------------------------
# Figure 4 — BST: Transformer over behaviors + candidate
# ---------------------------------------------------------------------------
def fig4_bst_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    T = 6
    x_positions = np.linspace(1.0, 9.5, T + 1)  # behaviors + candidate

    # Token row
    for i, x in enumerate(x_positions):
        is_cand = (i == T)
        col = ORANGE if is_cand else BLUE
        label = "Candidate" if is_cand else f"$b_{i+1}$"
        box = FancyBboxPatch((x - 0.42, 0.6), 0.84, 0.7,
                              boxstyle="round,pad=0.04", linewidth=1.6,
                              facecolor=col, edgecolor=col, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, 0.95, label, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
        # position embedding
        ax.text(x, 0.25, f"+ pos$_{i+1}$", ha="center", va="center",
                fontsize=8, color=GRAY)

    # Multi-head self-attention block
    box = FancyBboxPatch((0.5, 2.0), 9.5, 1.2,
                          boxstyle="round,pad=0.05", linewidth=1.8,
                          facecolor=PURPLE, edgecolor=PURPLE, alpha=0.85)
    ax.add_patch(box)
    ax.text(5.25, 2.6, "Multi-Head Self-Attention   (every token attends to every other)",
            ha="center", va="center", color="white", fontsize=11, fontweight="bold")

    # Up arrows from tokens to attention
    for x in x_positions:
        ax.annotate("", xy=(x, 1.98), xytext=(x, 1.32),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

    # FFN block
    box = FancyBboxPatch((0.5, 3.55), 9.5, 0.9,
                          boxstyle="round,pad=0.05", linewidth=1.6,
                          facecolor=GREEN, edgecolor=GREEN, alpha=0.85)
    ax.add_patch(box)
    ax.text(5.25, 4.0, "Feed-Forward + Add & Norm   (×N layers)",
            ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
    for x in x_positions:
        ax.annotate("", xy=(x, 3.53), xytext=(x, 3.22),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))

    # Output tokens (Z)
    for i, x in enumerate(x_positions):
        circ = mpatches.Circle((x, 4.95), 0.27, facecolor="white",
                                edgecolor=PURPLE, linewidth=1.6)
        ax.add_patch(circ)
        ax.text(x, 4.95, f"$z_{i+1}$", ha="center", va="center",
                fontsize=9, color=PURPLE)
        ax.annotate("", xy=(x, 4.68), xytext=(x, 4.47),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.8))

    # MLP head on the right
    box = FancyBboxPatch((10.6, 2.0), 1.9, 2.95,
                          boxstyle="round,pad=0.05", linewidth=1.6,
                          facecolor=BLUE, edgecolor=BLUE, alpha=0.9)
    ax.add_patch(box)
    ax.text(11.55, 4.5, "MLP", ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")
    ax.text(11.55, 3.7, "concat\n+ user/ctx\nfeatures", ha="center", va="center",
            color="white", fontsize=8.5)
    ax.text(11.55, 2.55, "CTR\nprob", ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    ax.annotate("", xy=(10.55, 4.95), xytext=(9.85, 4.95),
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.5))

    ax.text(5.25, 5.85,
            "BST: candidate is just another token. Self-attention finds its connections to history in one shot.",
            ha="center", fontsize=10, style="italic", color=GRAY)

    ax.set_title("BST — Behavior Sequence Transformer for CTR", pad=12)
    fig.tight_layout()
    save(fig, "fig4_bst_architecture.png")


# ---------------------------------------------------------------------------
# Figure 5 — Interest evolution over time
# ---------------------------------------------------------------------------
def fig5_interest_evolution() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.2))

    weeks = np.arange(1, 13)
    # Interest intensity over 12 weeks for three categories
    laptops    = np.array([0.20, 0.50, 0.85, 0.92, 0.70, 0.45, 0.25, 0.15, 0.10, 0.08, 0.06, 0.05])
    accessories= np.array([0.05, 0.10, 0.20, 0.45, 0.78, 0.92, 0.85, 0.60, 0.35, 0.20, 0.12, 0.08])
    chairs     = np.array([0.04, 0.05, 0.05, 0.06, 0.10, 0.18, 0.35, 0.62, 0.85, 0.94, 0.88, 0.72])

    ax.fill_between(weeks, laptops, alpha=0.20, color=BLUE)
    ax.fill_between(weeks, accessories, alpha=0.20, color=PURPLE)
    ax.fill_between(weeks, chairs, alpha=0.20, color=GREEN)
    ax.plot(weeks, laptops, color=BLUE, lw=2.4, marker="o", label="Laptops")
    ax.plot(weeks, accessories, color=PURPLE, lw=2.4, marker="s", label="Laptop accessories")
    ax.plot(weeks, chairs, color=GREEN, lw=2.4, marker="^", label="Ergonomic chairs")

    # Annotate where interest peaks shift
    ax.annotate("Research phase", xy=(4, 0.92), xytext=(4, 1.05),
                ha="center", fontsize=9, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1))
    ax.annotate("Buy accessories", xy=(6, 0.92), xytext=(6.6, 1.05),
                ha="center", fontsize=9, color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1))
    ax.annotate("Office setup", xy=(10, 0.94), xytext=(10, 1.07),
                ha="center", fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))

    ax.set_xlabel("Week")
    ax.set_ylabel("Interest intensity")
    ax.set_xticks(weeks)
    ax.set_ylim(0, 1.18)
    ax.legend(loc="upper right", fontsize=9, ncol=3)
    ax.set_title("User interest is a moving target — DIN treats it as static, DIEN models the trajectory",
                 fontsize=12, pad=10)
    ax.grid(alpha=0.4)

    fig.tight_layout()
    save(fig, "fig5_interest_evolution.png")


# ---------------------------------------------------------------------------
# Figure 6 — Static avg vs DIN attention on Amazon Books CTR (AUC progression)
# ---------------------------------------------------------------------------
def fig6_performance_comparison() -> None:
    models = ["Wide&Deep", "Sum/Avg\npooling", "DIN", "DIEN", "DSIN", "BST"]
    # Approximate AUCs reported on Amazon Books / Taobao-style benchmarks
    auc = [0.7405, 0.7472, 0.7634, 0.7716, 0.7782, 0.7810]
    # Relative gain over baseline
    base = auc[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Left: AUC bar chart
    ax = axes[0]
    colors = [GRAY, GRAY, BLUE, PURPLE, GREEN, ORANGE]
    bars = ax.bar(models, auc, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax.set_ylim(0.73, 0.79)
    ax.set_ylabel("AUC")
    ax.set_title("AUC on Amazon Books CTR", color=GRAY)
    for bar, v in zip(bars, auc):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.0008, f"{v:.4f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(base, color=GRAY, lw=1, linestyle=":", alpha=0.6)
    ax.text(0.05, base + 0.0005, "baseline", fontsize=8, color=GRAY)
    ax.grid(axis="y", alpha=0.4)

    # Right: relative AUC lift over baseline
    ax = axes[1]
    lifts = [(a - base) * 100 for a in auc]
    bars = ax.barh(models, lifts, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
    ax.set_xlabel("AUC lift over Wide&Deep baseline (×100)")
    ax.set_title("Where the gains actually come from", color=GRAY)
    for bar, v in zip(bars, lifts):
        if v == 0:
            ax.text(0.02, bar.get_y() + bar.get_height() / 2, "—", va="center", fontsize=9)
        else:
            ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"+{v:.2f}", va="center", fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.4)

    fig.suptitle("Attention is the single biggest jump — everything after is incremental",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig6_performance_comparison.png")


# ---------------------------------------------------------------------------
# Figure 7 — Dice activation vs PReLU
# ---------------------------------------------------------------------------
def fig7_activation_functions() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    x = np.linspace(-4, 4, 400)
    alpha = 0.25

    def prelu(x, a):
        return np.where(x > 0, x, a * x)

    def dice(x, mu, sigma, a):
        # data-adaptive: smooth transition centered on batch mean
        p = 1.0 / (1.0 + np.exp(-(x - mu) / sigma))
        return p * x + (1 - p) * a * x

    # Left: PReLU vs Dice for the same batch distribution (mu=0)
    ax = axes[0]
    ax.plot(x, prelu(x, alpha), color=GRAY, lw=2.4, label="PReLU (hard switch at 0)")
    ax.plot(x, dice(x, mu=0.0, sigma=1.0, a=alpha), color=BLUE, lw=2.4,
            label="Dice (smooth, μ=0)")
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5, linestyle=":")
    ax.set_title("Dice smooths the inflection point", color=GRAY)
    ax.set_xlabel("input  $x$")
    ax.set_ylabel("activation  $f(x)$")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.4)

    # Right: Dice with three different batch means (data-adaptive)
    ax = axes[1]
    for mu, color, lbl in [(-1.5, BLUE, "batch mean = -1.5"),
                            (0.0, PURPLE, "batch mean = 0.0"),
                            (1.5, GREEN, "batch mean = 1.5")]:
        ax.plot(x, dice(x, mu=mu, sigma=0.8, a=alpha), color=color, lw=2.4, label=lbl)
        ax.axvline(mu, color=color, lw=1, linestyle=":", alpha=0.6)
    ax.plot(x, prelu(x, alpha), color=GRAY, lw=1.6, linestyle="--", label="PReLU (fixed)")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Dice's switch follows the data", color=GRAY)
    ax.set_xlabel("input  $x$")
    ax.set_ylabel("activation  $f(x)$")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.4)

    fig.suptitle("Dice — a data-adaptive PReLU that recenters with the batch",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_activation_functions.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 10 figures (DIN/DIEN/DSIN/BST)...")
    fig1_attention_weights()
    print("  fig1 ok")
    fig2_dien_architecture()
    print("  fig2 ok")
    fig3_dsin_sessions()
    print("  fig3 ok")
    fig4_bst_architecture()
    print("  fig4 ok")
    fig5_interest_evolution()
    print("  fig5 ok")
    fig6_performance_comparison()
    print("  fig6 ok")
    fig7_activation_functions()
    print("  fig7 ok")
    print(f"All saved to:\n  EN: {EN_DIR}\n  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
