"""
Figures for Recommendation Systems Part 06: Sequential Recommendation.

Generates six pedagogical figures used by the EN and ZH posts:
  1. fig1_sequence_timeline.png         - User interaction timeline
  2. fig2_gru4rec_architecture.png      - GRU4Rec architecture
  3. fig3_attention_heatmap.png         - SASRec self-attention heatmap
  4. fig4_bert4rec_masking.png          - BERT4Rec masked item prediction
  5. fig5_position_plus_item.png        - Position + item embedding combined
  6. fig6_performance_vs_length.png     - Performance vs sequence length

All figures saved to BOTH the EN and ZH asset folders.
Style: seaborn-v0_8-whitegrid, dpi=150, palette = {#2563eb, #7c3aed, #10b981, #f59e0b}.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GREY = "#64748b"
LIGHT = "#e5e7eb"
DARK = "#1e293b"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source/_posts/en/recommendation-systems/06-sequential-recommendation"
ZH_DIR = REPO_ROOT / "source/_posts/zh/recommendation-systems/06-序列推荐与会话建模"


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for folder in (EN_DIR, ZH_DIR):
        out = folder / name
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 -- User interaction sequence timeline
# ---------------------------------------------------------------------------
def fig1_sequence_timeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.4))

    items = [
        ("Phone",   BLUE,   "Browse"),
        ("Case",    BLUE,   "Browse"),
        ("Cable",   BLUE,   "Browse"),
        ("Earbuds", PURPLE, "Click"),
        ("Earbuds", GREEN,  "Add to Cart"),
        ("Charger", PURPLE, "Click"),
        ("?",       ORANGE, "Predict"),
    ]
    n = len(items)
    xs = np.linspace(1.0, 12.0, n)
    y_axis = 1.0

    # baseline timeline
    ax.plot([0.4, 12.6], [y_axis, y_axis], color=GREY, lw=1.6, zorder=1)
    ax.annotate("", xy=(12.7, y_axis), xytext=(12.45, y_axis),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.6))

    # tick marks at each step
    for i, x in enumerate(xs):
        ax.plot([x, x], [y_axis - 0.06, y_axis + 0.06], color=GREY, lw=1.2)
        ax.text(x, y_axis - 0.32, f"t = {i+1}", ha="center", va="top",
                fontsize=10, color=GREY)

    # nodes
    for x, (label, color, action) in zip(xs, items):
        is_pred = label == "?"
        face = "white" if is_pred else color
        edge = color
        circle = Circle((x, y_axis + 0.55), 0.34, facecolor=face,
                        edgecolor=edge, lw=2.2, zorder=3)
        if is_pred:
            circle.set_linestyle((0, (4, 3)))
        ax.add_patch(circle)
        ax.text(x, y_axis + 0.55, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                color=color if is_pred else "white")
        # action label above
        ax.text(x, y_axis + 1.12, action, ha="center", va="bottom",
                fontsize=9.5, color=DARK)

    # connecting arrows between consecutive items (showing dependency)
    for i in range(n - 1):
        ax.annotate("",
                    xy=(xs[i + 1] - 0.34, y_axis + 0.55),
                    xytext=(xs[i] + 0.34, y_axis + 0.55),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0,
                                    alpha=0.55, connectionstyle="arc3,rad=0.0"))

    # title and annotations
    ax.text(6.5, 2.85, "User Interaction Sequence",
            ha="center", fontsize=14, fontweight="bold", color=DARK)
    ax.text(6.5, 2.55, r"$P(i_{t+1}\,|\,i_1, i_2, \ldots, i_t)$  -- predict next action from ordered history",
            ha="center", fontsize=10.5, color=GREY, style="italic")

    # legend for action types
    legend_elems = [
        mpatches.Patch(color=BLUE,   label="Browse"),
        mpatches.Patch(color=PURPLE, label="Click"),
        mpatches.Patch(color=GREEN,  label="Add to Cart"),
        mpatches.Patch(facecolor="white", edgecolor=ORANGE,
                       linestyle="--", label="Next-item prediction"),
    ]
    ax.legend(handles=legend_elems, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=10)

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.2)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig1_sequence_timeline.png")


# ---------------------------------------------------------------------------
# Figure 2 -- GRU4Rec architecture
# ---------------------------------------------------------------------------
def fig2_gru4rec_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6))

    seq_items = [r"$i_1$", r"$i_2$", r"$i_3$", r"$i_4$"]
    n = len(seq_items)
    xs = np.linspace(1.6, 10.4, n)

    layers_y = {
        "input":  0.6,
        "embed":  2.0,
        "gru":    3.6,
        "output": 5.2,
    }

    # ---- Input layer (item IDs) ----
    for x, label in zip(xs, seq_items):
        rect = FancyBboxPatch((x - 0.45, layers_y["input"] - 0.28), 0.9, 0.56,
                              boxstyle="round,pad=0.04",
                              facecolor=LIGHT, edgecolor=GREY, lw=1.4)
        ax.add_patch(rect)
        ax.text(x, layers_y["input"], label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=DARK)
    ax.text(0.4, layers_y["input"], "Item IDs", ha="right", va="center",
            fontsize=10.5, color=GREY)

    # ---- Embedding layer ----
    for x in xs:
        rect = FancyBboxPatch((x - 0.45, layers_y["embed"] - 0.28), 0.9, 0.56,
                              boxstyle="round,pad=0.04",
                              facecolor=BLUE, edgecolor=BLUE, lw=1.4, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, layers_y["embed"], "emb", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        # vertical arrow input -> embed
        ax.annotate("", xy=(x, layers_y["embed"] - 0.32),
                    xytext=(x, layers_y["input"] + 0.32),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
    ax.text(0.4, layers_y["embed"], "Embedding", ha="right", va="center",
            fontsize=10.5, color=GREY)

    # ---- GRU cells ----
    h_labels = [r"$h_1$", r"$h_2$", r"$h_3$", r"$h_4$"]
    for i, (x, h) in enumerate(zip(xs, h_labels)):
        rect = FancyBboxPatch((x - 0.55, layers_y["gru"] - 0.34), 1.1, 0.68,
                              boxstyle="round,pad=0.04",
                              facecolor=PURPLE, edgecolor=PURPLE, lw=1.4, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x, layers_y["gru"] + 0.06, "GRU", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white")
        ax.text(x, layers_y["gru"] - 0.18, h, ha="center", va="center",
                fontsize=9.5, color="white")
        # vertical arrow embed -> gru
        ax.annotate("", xy=(x, layers_y["gru"] - 0.38),
                    xytext=(x, layers_y["embed"] + 0.32),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.2))
        # horizontal arrow between GRU cells (hidden state passing)
        if i > 0:
            ax.annotate("", xy=(x - 0.56, layers_y["gru"]),
                        xytext=(xs[i - 1] + 0.56, layers_y["gru"]),
                        arrowprops=dict(arrowstyle="->", color=ORANGE,
                                        lw=2.0, alpha=0.95))
    ax.text(0.4, layers_y["gru"], "GRU Layer", ha="right", va="center",
            fontsize=10.5, color=GREY)
    # hidden state legend
    ax.text(xs[1] + (xs[2] - xs[1]) / 2, layers_y["gru"] + 0.55,
            "hidden state passes forward",
            ha="center", fontsize=9, color=ORANGE, style="italic")

    # ---- Output / softmax over items ----
    out_x = xs[-1]
    rect = FancyBboxPatch((out_x - 0.95, layers_y["output"] - 0.34), 1.9, 0.68,
                          boxstyle="round,pad=0.04",
                          facecolor=GREEN, edgecolor=GREEN, lw=1.4, alpha=0.9)
    ax.add_patch(rect)
    ax.text(out_x, layers_y["output"], "Softmax over items",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color="white")
    ax.annotate("", xy=(out_x, layers_y["output"] - 0.38),
                xytext=(out_x, layers_y["gru"] + 0.36),
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.text(out_x + 1.25, layers_y["output"], r"$P(i_{t+1}\,|\,h_t)$",
            ha="left", va="center", fontsize=11, color=DARK)

    # title
    ax.text(6.0, 6.15, "GRU4Rec: Sequence -> Hidden State -> Next-Item Distribution",
            ha="center", fontsize=13, fontweight="bold", color=DARK)

    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig2_gru4rec_architecture.png")


# ---------------------------------------------------------------------------
# Figure 3 -- SASRec self-attention heatmap (causal)
# ---------------------------------------------------------------------------
def fig3_attention_heatmap() -> None:
    items = ["Phone", "Case", "Cable", "Earbuds", "Charger", "Stand", "Speaker", "[next]"]
    n = len(items)

    # Build a plausible causal attention matrix.
    # Row q can only look at keys with index <= q. Recent items get more weight,
    # plus a thematic bump (earbuds <-> speakers and accessories cluster).
    rng = np.random.default_rng(7)
    A = np.full((n, n), np.nan)
    for q in range(n):
        raw = np.zeros(q + 1)
        for k in range(q + 1):
            recency = np.exp(-(q - k) / 2.5)
            theme = 1.0
            if {items[q], items[k]}.issubset({"Earbuds", "Speaker", "Charger", "Cable"}):
                theme += 0.6
            if items[q] == items[k]:
                theme += 0.3
            raw[k] = recency * theme + rng.uniform(0, 0.05)
        weights = raw / raw.sum()
        A[q, : q + 1] = weights

    fig, ax = plt.subplots(figsize=(8.2, 6.6))

    cmap = plt.get_cmap("Blues")
    cmap.set_bad(color="#f1f5f9")

    im = ax.imshow(A, cmap=cmap, vmin=0, vmax=np.nanmax(A), aspect="equal")

    # annotate cells
    for q in range(n):
        for k in range(n):
            if not np.isnan(A[q, k]):
                val = A[q, k]
                color = "white" if val > 0.45 * np.nanmax(A) else DARK
                ax.text(k, q, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=color)
            elif k > q:
                # masked cell -- mark with diagonal shade
                ax.add_patch(Rectangle((k - 0.5, q - 0.5), 1, 1,
                                       facecolor="#f1f5f9",
                                       edgecolor="white", lw=0.5,
                                       hatch="///", alpha=0.55))

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(items, rotation=35, ha="right", fontsize=10)
    ax.set_yticklabels(items, fontsize=10)
    ax.set_xlabel("Key (attended-to item)", fontsize=11, color=DARK)
    ax.set_ylabel("Query (current position)", fontsize=11, color=DARK)
    ax.set_title("SASRec Self-Attention Weights\n(causal mask: hatched cells are blocked)",
                 fontsize=12.5, fontweight="bold", color=DARK, pad=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("attention weight", fontsize=10, color=GREY)
    cbar.ax.tick_params(labelsize=9)
    ax.grid(False)

    fig.tight_layout()
    save(fig, "fig3_attention_heatmap.png")


# ---------------------------------------------------------------------------
# Figure 4 -- BERT4Rec masked item prediction
# ---------------------------------------------------------------------------
def fig4_bert4rec_masking() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))

    original = ["Phone", "Case", "Cable", "Earbuds", "Charger", "Stand", "Speaker"]
    mask_idx = {2, 5}  # Cable, Stand are masked
    n = len(original)
    xs = np.linspace(1.2, 11.8, n)

    y_in, y_out = 1.1, 4.0

    # title
    ax.text(6.5, 5.05, "BERT4Rec: Cloze-style Masked Item Prediction",
            ha="center", fontsize=13.5, fontweight="bold", color=DARK)
    ax.text(6.5, 4.72,
            "Bidirectional attention reads both left and right context to fill [MASK] positions",
            ha="center", fontsize=10.5, color=GREY, style="italic")

    # input row (with masks)
    for i, (x, label) in enumerate(zip(xs, original)):
        if i in mask_idx:
            face = "white"
            edge = ORANGE
            txt = "[MASK]"
            txt_color = ORANGE
            lw = 2.2
            ls = (0, (4, 3))
        else:
            face = BLUE
            edge = BLUE
            txt = label
            txt_color = "white"
            lw = 1.4
            ls = "-"
        rect = FancyBboxPatch((x - 0.6, y_in - 0.34), 1.2, 0.68,
                              boxstyle="round,pad=0.04",
                              facecolor=face, edgecolor=edge, lw=lw,
                              linestyle=ls)
        ax.add_patch(rect)
        ax.text(x, y_in, txt, ha="center", va="center",
                fontsize=10, fontweight="bold", color=txt_color)
        ax.text(x, y_in - 0.6, f"pos {i+1}", ha="center", va="top",
                fontsize=9, color=GREY)

    ax.text(0.2, y_in, "Input\n(masked)", ha="right", va="center",
            fontsize=10.5, color=GREY)

    # transformer block in middle
    block = FancyBboxPatch((1.0, 2.25), 11.6, 1.0,
                           boxstyle="round,pad=0.05",
                           facecolor=PURPLE, edgecolor=PURPLE, lw=1.4, alpha=0.9)
    ax.add_patch(block)
    ax.text(6.8, 2.75, "Bidirectional Transformer Encoder (no causal mask)",
            ha="center", va="center", fontsize=11.5, fontweight="bold", color="white")

    # arrows input -> encoder
    for x in xs:
        ax.annotate("", xy=(x, 2.23), xytext=(x, y_in + 0.36),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=1.0))

    # output row (predictions only at masked positions)
    for i, x in enumerate(xs):
        if i in mask_idx:
            ax.annotate("", xy=(x, y_out - 0.36),
                        xytext=(x, 3.28),
                        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6))
            rect = FancyBboxPatch((x - 0.6, y_out - 0.34), 1.2, 0.68,
                                  boxstyle="round,pad=0.04",
                                  facecolor=GREEN, edgecolor=GREEN, lw=1.4, alpha=0.95)
            ax.add_patch(rect)
            ax.text(x, y_out, original[i], ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white")
            ax.text(x, y_out + 0.42, "predict", ha="center", va="bottom",
                    fontsize=9, color=GREEN, style="italic")
        else:
            # faded marker showing the encoder still emits a vector
            ax.plot(x, y_out, marker="s", markersize=6,
                    markerfacecolor=LIGHT, markeredgecolor=GREY)

    ax.text(0.2, y_out, "Output", ha="right", va="center",
            fontsize=10.5, color=GREY)

    # legend
    legend_elems = [
        mpatches.Patch(color=BLUE,   label="observed item"),
        mpatches.Patch(facecolor="white", edgecolor=ORANGE, linestyle="--",
                       label="[MASK] token"),
        mpatches.Patch(color=GREEN,  label="predicted item"),
    ]
    ax.legend(handles=legend_elems, loc="lower center",
              bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=10)

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.4)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "fig4_bert4rec_masking.png")


# ---------------------------------------------------------------------------
# Figure 5 -- Position + item embedding combined
# ---------------------------------------------------------------------------
def fig5_position_plus_item() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))

    n_pos = 6
    d = 12  # embedding dim shown as colored cells
    rng = np.random.default_rng(11)

    # item embeddings -- structured (each item has a distinct profile)
    item_emb = rng.normal(0, 1, size=(n_pos, d))
    # positional encoding -- sinusoidal pattern (Transformer style)
    pos = np.arange(n_pos)[:, None]
    dim = np.arange(d)[None, :]
    angle = pos / np.power(10000, 2 * (dim // 2) / d)
    pos_enc = np.zeros_like(angle)
    pos_enc[:, 0::2] = np.sin(angle[:, 0::2])
    pos_enc[:, 1::2] = np.cos(angle[:, 1::2])

    combined = item_emb + pos_enc

    titles = ["Item embedding\n$E_{item}$",
              "Positional encoding\n$E_{pos}$",
              "Combined input\n$E_{item} + E_{pos}$"]
    matrices = [item_emb, pos_enc, combined]
    cmaps = ["Blues", "Purples", "Greens"]

    panel_xs = [0.04, 0.36, 0.68]   # left edge of each panel in axes coords
    panel_w = 0.27
    panel_h = 0.62
    panel_y = 0.22

    for i, (title, M, cmap) in enumerate(zip(titles, matrices, cmaps)):
        sub_ax = fig.add_axes([panel_xs[i], panel_y, panel_w, panel_h])
        vmax = np.max(np.abs(M))
        im = sub_ax.imshow(M, cmap=cmap, aspect="auto",
                           vmin=-vmax, vmax=vmax)
        sub_ax.set_xticks([])
        sub_ax.set_yticks(range(n_pos))
        sub_ax.set_yticklabels([f"pos {p+1}" for p in range(n_pos)], fontsize=9)
        sub_ax.set_xlabel("embedding dim", fontsize=9.5, color=GREY)
        sub_ax.set_title(title, fontsize=11, color=DARK, fontweight="bold", pad=8)
        sub_ax.grid(False)

    # arithmetic operators between panels
    fig.text(0.34, panel_y + panel_h / 2, "+", fontsize=28,
             ha="center", va="center", color=DARK, fontweight="bold")
    fig.text(0.66, panel_y + panel_h / 2, "=", fontsize=28,
             ha="center", va="center", color=DARK, fontweight="bold")

    # main title
    fig.suptitle("Position + Item Embeddings: How Transformers Inject Order Information",
                 fontsize=13.5, fontweight="bold", color=DARK, y=0.97)
    fig.text(0.5, 0.07,
             "Each row is one timestep. Without positional encoding the Transformer would treat the sequence as a bag of items.",
             ha="center", fontsize=10, color=GREY, style="italic")

    # the figure-level ax we created originally is unused; hide it
    ax.axis("off")
    save(fig, "fig5_position_plus_item.png")


# ---------------------------------------------------------------------------
# Figure 6 -- Performance vs sequence length
# ---------------------------------------------------------------------------
def fig6_performance_vs_length() -> None:
    lengths = np.array([5, 10, 20, 30, 50, 75, 100, 150, 200])

    # Plausible HR@10 trajectories (illustrative, not actual benchmark numbers)
    def saturating(low, high, k):
        return low + (high - low) * (1 - np.exp(-lengths / k))

    markov   = np.clip(saturating(0.18, 0.27, 8) - 0.02 * np.log1p(lengths / 30),
                       0.10, 0.35)
    gru4rec  = saturating(0.24, 0.41, 18) - 0.015 * np.maximum(lengths - 60, 0) / 80
    sasrec   = saturating(0.26, 0.49, 28)
    bert4rec = saturating(0.27, 0.515, 32)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- left: HR@10 vs sequence length ----
    ax1.plot(lengths, markov,   color=GREY,   marker="o", lw=2.0, label="Markov chain")
    ax1.plot(lengths, gru4rec,  color=BLUE,   marker="s", lw=2.2, label="GRU4Rec (RNN)")
    ax1.plot(lengths, sasrec,   color=PURPLE, marker="^", lw=2.4, label="SASRec (Transformer)")
    ax1.plot(lengths, bert4rec, color=GREEN,  marker="D", lw=2.4, label="BERT4Rec (Bi-Transformer)")

    ax1.set_xlabel("Maximum sequence length", fontsize=11, color=DARK)
    ax1.set_ylabel("HR@10 (higher is better)", fontsize=11, color=DARK)
    ax1.set_title("Recommendation Quality vs Sequence Length",
                  fontsize=12.5, fontweight="bold", color=DARK)
    ax1.set_xscale("log")
    ax1.set_xticks(lengths)
    ax1.set_xticklabels([str(l) for l in lengths], fontsize=9)
    ax1.tick_params(axis="y", labelsize=9)
    ax1.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95)
    ax1.set_ylim(0.10, 0.55)

    # annotate where transformers pull ahead
    ax1.axvspan(40, 200, alpha=0.08, color=PURPLE)
    ax1.text(85, 0.135, "Transformers dominate\nfor long sequences",
             ha="center", fontsize=9.5, color=PURPLE, style="italic")

    # ---- right: training time vs length (relative) ----
    rel_time_rnn   = (lengths / 50) ** 1.0  # linear in length, can't parallelize
    rel_time_trans = 0.35 + 0.025 * (lengths / 50) ** 1.4  # GPU-parallel, near-flat

    ax2.plot(lengths, rel_time_rnn,   color=BLUE,   marker="s", lw=2.2,
             label="RNN (sequential)")
    ax2.plot(lengths, rel_time_trans, color=PURPLE, marker="^", lw=2.4,
             label="Transformer (parallel)")
    ax2.fill_between(lengths, rel_time_trans, rel_time_rnn,
                     where=rel_time_rnn > rel_time_trans,
                     color=ORANGE, alpha=0.15)
    ax2.set_xlabel("Maximum sequence length", fontsize=11, color=DARK)
    ax2.set_ylabel("Relative training time per step", fontsize=11, color=DARK)
    ax2.set_title("Training Cost: RNN vs Transformer",
                  fontsize=12.5, fontweight="bold", color=DARK)
    ax2.set_xscale("log")
    ax2.set_xticks(lengths)
    ax2.set_xticklabels([str(l) for l in lengths], fontsize=9)
    ax2.tick_params(axis="y", labelsize=9)
    ax2.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.95)
    ax2.text(110, 1.6, "Transformer\nspeed advantage",
             ha="center", fontsize=9.5, color=ORANGE, style="italic")

    fig.tight_layout()
    save(fig, "fig6_performance_vs_length.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")
    print()
    fig1_sequence_timeline()
    fig2_gru4rec_architecture()
    fig3_attention_heatmap()
    fig4_bert4rec_masking()
    fig5_position_plus_item()
    fig6_performance_vs_length()
    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()
