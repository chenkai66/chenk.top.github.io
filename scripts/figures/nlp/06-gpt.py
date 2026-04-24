"""
Figure generation script for NLP Part 06: GPT and Generative Language Models.

Generates 7 figures used in both EN and ZH versions of the article. Each figure
teaches a single specific concept cleanly and pedagogically.

Figures:
    fig1_decoder_only_arch       Decoder-only Transformer block diagram with a
                                 zoom-in on masked self-attention.
    fig2_autoregressive_step     Step-by-step autoregressive generation: prompt
                                 -> next token distribution -> append -> repeat.
    fig3_scaling_laws            Compute / parameter scaling vs. test loss
                                 (Kaplan-style power-law on log-log axes).
    fig4_gpt_evolution           GPT-1 -> GPT-2 -> GPT-3 -> GPT-4 timeline with
                                 parameter counts and capability annotations.
    fig5_sampling_strategies     Side-by-side visualisation of greedy, top-k,
                                 top-p and temperature on the same logits.
    fig6_emergent_capabilities   Capability-vs-scale curves: smooth vs.
                                 emergent (sharp phase transition) abilities.
    fig7_in_context_learning     Few-shot prompt anatomy: task / examples /
                                 query / completion with attention "look-back".

Usage:
    python3 scripts/figures/nlp/06-gpt.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so that the
    markdown references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary
C_PURPLE = "#7c3aed"   # secondary
C_GREEN = "#10b981"    # accent / good
C_AMBER = "#f59e0b"    # warning / highlight
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "nlp" / "gpt-generative-models"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "nlp" / "06-GPT与生成式语言模型"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, x, y, w, h, label, fc, ec=None, fs=10, fw="bold", tc="white"):
    """Draw a rounded rectangle with a centred label."""
    if ec is None:
        ec = fc
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center", fontsize=fs, fontweight=fw, color=tc)


def _arrow(ax, p1, p2, color=C_DARK, lw=1.5, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color,
    ))


# ---------------------------------------------------------------------------
# Figure 1: Decoder-only architecture with masked self-attention zoom
# ---------------------------------------------------------------------------
def fig1_decoder_only_arch() -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 7),
                                     gridspec_kw={"width_ratios": [1, 1.05]})

    # ---- Left: stacked decoder blocks ----
    ax_a.set_xlim(0, 10)
    ax_a.set_ylim(0, 11)
    ax_a.axis("off")
    ax_a.set_title("Decoder-Only Transformer (GPT)",
                   fontsize=13, fontweight="bold", pad=10)

    # Input
    _box(ax_a, 1.5, 0.3, 7, 0.8, r"Input tokens  $x_1, x_2, \ldots, x_t$",
         "#1e293b", fs=11)
    _box(ax_a, 1.5, 1.4, 7, 0.8,
         "Token embed  +  Positional embed", C_GRAY, fs=10)

    # N x decoder block
    block_y = 2.6
    block_h = 5.6
    outer = FancyBboxPatch((1.0, block_y), 8.0, block_h,
                           boxstyle="round,pad=0.03,rounding_size=0.15",
                           linewidth=1.2, edgecolor=C_BLUE,
                           facecolor="#eff6ff")
    ax_a.add_patch(outer)
    ax_a.text(9.6, block_y + block_h - 0.3, "L×",
              ha="right", va="top", fontsize=14, fontweight="bold",
              color=C_BLUE)

    _box(ax_a, 1.5, block_y + 4.4, 7, 0.7,
         "Masked Multi-Head Self-Attention", C_BLUE, fs=10)
    _box(ax_a, 1.5, block_y + 3.5, 7, 0.55,
         "Add & LayerNorm", C_GRAY, fs=9)
    _box(ax_a, 1.5, block_y + 2.4, 7, 0.7,
         "Feed-Forward (MLP)", C_PURPLE, fs=10)
    _box(ax_a, 1.5, block_y + 1.5, 7, 0.55,
         "Add & LayerNorm", C_GRAY, fs=9)
    _box(ax_a, 1.5, block_y + 0.4, 7, 0.7,
         "Residual stream", "#cbd5e1", fs=9, tc=C_DARK)

    # Output head
    _box(ax_a, 1.5, 8.5, 7, 0.7,
         "Linear  →  Softmax over vocab", C_GREEN, fs=10)
    _box(ax_a, 1.5, 9.6, 7, 0.8,
         r"$P(x_{t+1}\,|\,x_1,\ldots,x_t)$",
         C_AMBER, fs=12)

    # connectors
    for y1, y2 in [(1.1, 1.4), (2.2, 2.6), (8.2, 8.5), (9.2, 9.6)]:
        _arrow(ax_a, (5.0, y1), (5.0, y2), color=C_GRAY, lw=1.2)

    # ---- Right: causal mask zoom ----
    ax_b.set_title("Causal Mask: position i sees only tokens ≤ i",
                   fontsize=13, fontweight="bold", pad=10)
    n = 7
    M = np.tril(np.ones((n, n)))
    # Display: visible cells = blue, masked cells = light grey
    display = np.where(M == 1, 1.0, 0.0)
    cmap = plt.matplotlib.colors.ListedColormap(["#f1f5f9", C_BLUE])
    ax_b.imshow(display, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    tokens = ["The", "cat", "sat", "on", "the", "mat", "."]
    ax_b.set_xticks(range(n))
    ax_b.set_yticks(range(n))
    ax_b.set_xticklabels(tokens, fontsize=10)
    ax_b.set_yticklabels(tokens, fontsize=10)
    ax_b.set_xlabel("Key (attended-to position j)", fontsize=11)
    ax_b.set_ylabel("Query (current position i)", fontsize=11)

    for i in range(n):
        for j in range(n):
            if j <= i:
                ax_b.text(j, i, "1", ha="center", va="center",
                          color="white", fontsize=11, fontweight="bold")
            else:
                ax_b.text(j, i, "−∞", ha="center", va="center",
                          color=C_DARK, fontsize=10)

    ax_b.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax_b.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax_b.grid(which="minor", color="white", linewidth=1.5)
    ax_b.tick_params(which="minor", length=0)
    ax_b.tick_params(axis="x", labelrotation=30)

    # caption strip
    fig.text(0.5, 0.01,
             "Lower triangle = visible past   |   Upper triangle masked to −∞ before softmax",
             ha="center", fontsize=10, color=C_DARK,
             bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT, boxstyle="round,pad=0.5"))

    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "fig1_decoder_only_arch")


# ---------------------------------------------------------------------------
# Figure 2: Autoregressive generation step-by-step
# ---------------------------------------------------------------------------
def fig2_autoregressive_step() -> None:
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.30,
                          height_ratios=[1, 1.1])
    fig.suptitle("Autoregressive Generation: one token at a time",
                 fontsize=14, fontweight="bold", y=0.995)

    # Top row: 3 generation steps (prompt grows)
    steps = [
        ("Step t=1", ["The", "cat", "sat", "on", "the"], "mat"),
        ("Step t=2", ["The", "cat", "sat", "on", "the", "mat"], "."),
        ("Step t=3", ["The", "cat", "sat", "on", "the", "mat", "."], "<eos>"),
    ]
    for k, (title, ctx, pred) in enumerate(steps):
        ax = fig.add_subplot(gs[0, k])
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color=C_BLUE, loc="left")

        # context tokens
        n_ctx = len(ctx)
        w = 8.5 / max(n_ctx + 1, 6)
        for i, tok in enumerate(ctx):
            x = 0.4 + i * w
            _box(ax, x, 2.0, w * 0.85, 0.7, tok, C_GRAY, fs=9)
        # predicted token
        x = 0.4 + n_ctx * w
        _box(ax, x, 2.0, w * 0.85, 0.7, pred, C_AMBER, fs=9)

        # arrow showing "predict next"
        _arrow(ax, (x + w * 0.4, 1.6), (x + w * 0.4, 1.95),
               color=C_AMBER, lw=2)
        ax.text(x + w * 0.4, 1.4, "argmax / sample",
                ha="center", fontsize=8, color=C_AMBER, fontweight="bold")

        # context label
        ax.text(0.4 + (n_ctx * w) / 2, 3.0, "context (visible to model)",
                ha="center", fontsize=8.5, color=C_DARK, style="italic")

    # Bottom row: probability distribution at step t=1
    ax2 = fig.add_subplot(gs[1, :])
    vocab = ["mat", "floor", "couch", "sofa", "bed", "rug", "chair", "table"]
    probs = np.array([0.42, 0.18, 0.11, 0.09, 0.07, 0.06, 0.04, 0.03])
    colors = [C_AMBER if v == "mat" else C_BLUE for v in vocab]
    bars = ax2.bar(vocab, probs, color=colors, edgecolor="white", linewidth=1.5)
    for bar, p in zip(bars, probs):
        ax2.text(bar.get_x() + bar.get_width() / 2, p + 0.012,
                 f"{p:.2f}", ha="center", fontsize=10, fontweight="bold",
                 color=C_DARK)

    ax2.set_title(
        "Step t=1  →  P( next token | \"The cat sat on the\" )",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax2.set_ylabel("Probability", fontsize=11)
    ax2.set_ylim(0, 0.55)
    ax2.tick_params(axis="x", labelsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.text(0.0, 0.50,
             "  selected (greedy: argmax)",
             color=C_AMBER, fontsize=10, fontweight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig2_autoregressive_step")


# ---------------------------------------------------------------------------
# Figure 3: Scaling laws (Kaplan-style)
# ---------------------------------------------------------------------------
def fig3_scaling_laws() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Power-law: L = (Nc / N)^alpha  on log-log
    N = np.logspace(6, 12, 200)  # 1M to 1T params
    Nc = 8.8e13
    alpha = 0.076
    L = (Nc / N) ** alpha
    L_irr = 1.69  # irreducible loss floor

    ax1.loglog(N, L, color=C_BLUE, lw=2.5, label=r"$L(N) \propto N^{-\alpha}$")
    ax1.axhline(L_irr, color=C_GRAY, linestyle="--", lw=1.2,
                label="irreducible loss")

    # Markers for known model sizes
    models = [("GPT-1\n117M", 1.17e8),
              ("GPT-2\n1.5B", 1.5e9),
              ("GPT-3\n175B", 1.75e11)]
    for name, n in models:
        l = (Nc / n) ** alpha
        ax1.scatter([n], [l], s=100, color=C_AMBER, zorder=5,
                    edgecolor="white", linewidth=1.5)
        ax1.annotate(name, (n, l), xytext=(8, 8),
                     textcoords="offset points", fontsize=9,
                     fontweight="bold", color=C_DARK)

    ax1.set_xlabel("Model parameters  N", fontsize=11)
    ax1.set_ylabel("Test cross-entropy loss", fontsize=11)
    ax1.set_title("Loss vs. parameters (log-log, power-law)",
                  fontsize=12, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, which="both", alpha=0.3)

    # Right: compute scaling
    C = np.logspace(-3, 4, 200)  # PetaFLOP-days
    Cc = 2.3e8
    beta = 0.050
    Lc = (Cc / C) ** beta
    ax2.loglog(C, Lc, color=C_PURPLE, lw=2.5,
               label=r"$L(C) \propto C^{-\beta}$")
    ax2.axhline(L_irr, color=C_GRAY, linestyle="--", lw=1.2,
                label="irreducible loss")

    ax2.set_xlabel("Training compute  C  (PetaFLOP-days)", fontsize=11)
    ax2.set_ylabel("Test cross-entropy loss", fontsize=11)
    ax2.set_title("Loss vs. compute (log-log, power-law)",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "Scaling Laws (Kaplan et al. 2020):  more parameters / data / compute "
        "→  predictably lower loss",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig3_scaling_laws")


# ---------------------------------------------------------------------------
# Figure 4: GPT evolution timeline
# ---------------------------------------------------------------------------
def fig4_gpt_evolution() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 6.5)
    ax.axis("off")
    ax.set_title("GPT evolution: 5 years, ~10,000× parameter growth",
                 fontsize=14, fontweight="bold", pad=10)

    # Timeline axis
    ax.plot([0.5, 9.5], [1.2, 1.2], color=C_GRAY, lw=2)

    models = [
        ("GPT-1", "Jun 2018", "117 M",
         "Decoder pretrain\n+ fine-tune", C_BLUE,    1.5),
        ("GPT-2", "Feb 2019", "1.5 B",
         "Zero-shot\nemerges",           C_PURPLE,  3.7),
        ("GPT-3", "May 2020", "175 B",
         "Few-shot\nin-context learning", C_GREEN,   5.9),
        ("GPT-4", "Mar 2023", "~1.7 T*",
         "Multimodal\n+ RLHF reasoning", C_AMBER,   8.1),
    ]

    # bar heights encode log10(params)
    log_params = [np.log10(p) for p in [1.17e8, 1.5e9, 1.75e11, 1.7e12]]
    max_h = max(log_params)

    for (name, date, p, capa, color, x), lp in zip(models, log_params):
        h = 0.4 + (lp / max_h) * 3.6
        # bar
        ax.add_patch(Rectangle((x - 0.55, 1.25), 1.1, h,
                               facecolor=color, edgecolor="white",
                               linewidth=2, alpha=0.92))
        # name + params on bar
        ax.text(x, 1.25 + h + 0.18, name, ha="center",
                fontsize=13, fontweight="bold", color=color)
        ax.text(x, 1.25 + h - 0.25, p, ha="center",
                fontsize=11, fontweight="bold", color="white")
        # date below axis
        ax.text(x, 0.85, date, ha="center", fontsize=10,
                color=C_DARK, fontweight="bold")
        ax.text(x, 0.45, capa, ha="center", fontsize=9,
                color=C_DARK, style="italic")
        # marker on axis
        ax.scatter([x], [1.2], s=80, color=color, zorder=5,
                   edgecolor="white", linewidth=2)

    # Connecting arrows between bars
    for (m1, m2) in zip(models[:-1], models[1:]):
        x1, x2 = m1[5], m2[5]
        h1 = 0.4 + (np.log10({"117 M": 1.17e8, "1.5 B": 1.5e9,
                              "175 B": 1.75e11, "~1.7 T*": 1.7e12}[m1[2]])
                    / max_h) * 3.6
        _arrow(ax, (x1 + 0.6, 1.25 + h1 / 2),
               (x2 - 0.6, 1.25 + h1 / 2), color=C_GRAY, lw=1.5)

    # Footnote
    ax.text(9.5, 5.9, "* GPT-4 size unofficial / rumoured",
            ha="right", fontsize=8.5, color=C_GRAY, style="italic")
    ax.text(0.5, -0.3, r"Bar height = $\log_{10}$(parameters)",
            fontsize=9, color=C_GRAY, style="italic")

    _save(fig, "fig4_gpt_evolution")


# ---------------------------------------------------------------------------
# Figure 5: Sampling strategies on the same logits
# ---------------------------------------------------------------------------
def fig5_sampling_strategies() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "Decoding Strategies on the same next-token distribution",
        fontsize=14, fontweight="bold", y=0.995,
    )

    vocab = ["mat", "floor", "couch", "sofa", "bed",
             "rug", "chair", "table", "desk", "ground"]
    logits = np.array([3.2, 2.5, 1.8, 1.5, 1.2, 1.0, 0.7, 0.4, 0.1, -0.2])

    def softmax(x, T=1.0):
        z = x / T
        e = np.exp(z - z.max())
        return e / e.sum()

    def plot_dist(ax, probs, title, kept_mask, sampled_idx=None, note=""):
        colors = [C_AMBER if kept_mask[i] else C_GRAY for i in range(len(probs))]
        if sampled_idx is not None:
            colors[sampled_idx] = C_GREEN
        ax.bar(vocab, probs, color=colors,
               edgecolor="white", linewidth=1.2)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylim(0, max(probs) * 1.25)
        ax.tick_params(axis="x", labelrotation=35, labelsize=9)
        ax.set_ylabel("Probability", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if note:
            ax.text(0.99, 0.95, note, transform=ax.transAxes,
                    ha="right", va="top", fontsize=9.5, color=C_DARK,
                    bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                              boxstyle="round,pad=0.35"))

    # (a) Greedy
    p1 = softmax(logits, T=1.0)
    mask = [i == 0 for i in range(len(p1))]
    plot_dist(axes[0, 0], p1,
              "(a) Greedy:  argmax",
              mask, sampled_idx=0,
              note="picks 'mat' every time\n→ deterministic, repetitive")

    # (b) Top-k = 4
    p2 = softmax(logits, T=1.0)
    k = 4
    topk = np.argsort(-p2)[:k]
    mask = [i in topk for i in range(len(p2))]
    plot_dist(axes[0, 1], p2,
              "(b) Top-k  (k=4)",
              mask, sampled_idx=2,
              note="sample uniformly* from top-4\n*after re-normalising")

    # (c) Top-p = 0.85 (nucleus)
    p3 = softmax(logits, T=1.0)
    order = np.argsort(-p3)
    cum = np.cumsum(p3[order])
    cutoff = np.searchsorted(cum, 0.85) + 1
    nucleus = set(order[:cutoff].tolist())
    mask = [i in nucleus for i in range(len(p3))]
    plot_dist(axes[1, 0], p3,
              "(c) Top-p / Nucleus  (p=0.85)",
              mask, sampled_idx=1,
              note=f"smallest set whose mass ≥ 0.85\nhere |nucleus| = {cutoff}")

    # (d) Temperature comparison: T=0.5 vs T=1.5
    ax = axes[1, 1]
    p_low = softmax(logits, T=0.5)
    p_high = softmax(logits, T=1.5)
    x = np.arange(len(vocab))
    w = 0.4
    ax.bar(x - w / 2, p_low, w, color=C_BLUE, label="T = 0.5  (sharp)",
           edgecolor="white", linewidth=1)
    ax.bar(x + w / 2, p_high, w, color=C_PURPLE, label="T = 1.5  (flat)",
           edgecolor="white", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(vocab, rotation=35, fontsize=9)
    ax.set_title("(d) Temperature  T",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Probability", fontsize=10)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.99, 0.78,
            "lower T → more conservative\nhigher T → more diverse",
            transform=ax.transAxes,
            ha="right", va="top", fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.35"))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig5_sampling_strategies")


# ---------------------------------------------------------------------------
# Figure 6: Emergent capabilities curves
# ---------------------------------------------------------------------------
def fig6_emergent_capabilities() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))

    N = np.logspace(7, 12, 200)  # 10M to 1T params

    def smooth(N, mid=1e9, k=1.0):
        x = np.log10(N) - np.log10(mid)
        return 100 / (1 + np.exp(-k * x))

    def emergent(N, threshold=2e10, k=4.5):
        x = np.log10(N) - np.log10(threshold)
        return 100 / (1 + np.exp(-k * x))

    # Smooth scaling: e.g. perplexity-like / classification
    y_smooth = smooth(N, mid=3e9, k=1.6) * 0.85

    # Emergent: arithmetic, ICL, chain-of-thought
    y_arith = emergent(N, threshold=2e10, k=5.0) * 0.80
    y_icl = emergent(N, threshold=8e9, k=4.5) * 0.88
    y_cot = emergent(N, threshold=6e10, k=6.0) * 0.78

    ax.semilogx(N, y_smooth, color=C_BLUE, lw=2.6,
                label="Sentiment classification  (smooth)")
    ax.semilogx(N, y_icl, color=C_PURPLE, lw=2.6,
                label="Few-shot in-context learning  (emergent)")
    ax.semilogx(N, y_arith, color=C_AMBER, lw=2.6,
                label="3-digit arithmetic  (emergent)")
    ax.semilogx(N, y_cot, color=C_GREEN, lw=2.6,
                label="Chain-of-thought reasoning  (emergent)")

    # Random baseline
    ax.axhline(10, color=C_GRAY, linestyle=":", lw=1.2)
    ax.text(1.1e7, 12, "random baseline",
            fontsize=9, color=C_GRAY, style="italic")

    # Vertical band for "emergence zone"
    ax.axvspan(8e9, 1e11, alpha=0.10, color=C_AMBER)
    ax.text(2.8e10, 95, "emergence zone",
            ha="center", fontsize=10, color=C_AMBER,
            fontweight="bold", style="italic")

    # Model markers on x axis
    for name, n in [("GPT-2", 1.5e9), ("GPT-3", 1.75e11)]:
        ax.axvline(n, color=C_GRAY, linestyle="--", lw=0.8, alpha=0.7)
        ax.text(n, -8, name, ha="center", fontsize=9,
                color=C_DARK, fontweight="bold")

    ax.set_xlabel("Model parameters", fontsize=11)
    ax.set_ylabel("Task accuracy (%)", fontsize=11)
    ax.set_title(
        "Emergent capabilities: some abilities appear suddenly with scale",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_ylim(-5, 105)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    _save(fig, "fig6_emergent_capabilities")


# ---------------------------------------------------------------------------
# Figure 7: In-context learning prompt anatomy
# ---------------------------------------------------------------------------
def fig7_in_context_learning() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("Few-Shot In-Context Learning  (no gradient updates)",
                 fontsize=14, fontweight="bold", pad=12)

    # Big rounded panel = prompt
    panel = FancyBboxPatch((0.4, 1.3), 5.6, 7.2,
                           boxstyle="round,pad=0.05,rounding_size=0.18",
                           linewidth=1.5, facecolor="#f1f5f9",
                           edgecolor=C_GRAY)
    ax.add_patch(panel)
    ax.text(3.2, 8.25, "Prompt fed to GPT", ha="center",
            fontsize=11, fontweight="bold", color=C_DARK)

    # Task header
    _box(ax, 0.7, 7.4, 5.0, 0.55,
         "Task: classify sentiment as positive / negative",
         C_BLUE, fs=10)

    # Examples
    ex = [
        ("Review: Loved every minute of it.",        "positive"),
        ("Review: Total waste of two hours.",        "negative"),
        ("Review: Acting was superb and moving.",    "positive"),
    ]
    y = 6.6
    for inp, out in ex:
        _box(ax, 0.7, y, 3.6, 0.5, inp, C_PURPLE, fs=9)
        _box(ax, 4.5, y, 1.2, 0.5, out, C_GREEN, fs=9)
        _arrow(ax, (4.32, y + 0.25), (4.5, y + 0.25),
               color=C_DARK, lw=1.2)
        y -= 0.75

    # Query
    _box(ax, 0.7, 3.8, 3.6, 0.5,
         "Review: I want my money back.", C_AMBER, fs=9)
    _box(ax, 4.5, 3.8, 1.2, 0.5, " ?", "#fff7ed",
         ec=C_AMBER, fs=11, tc=C_AMBER)
    _arrow(ax, (4.32, 4.05), (4.5, 4.05), color=C_AMBER, lw=1.5)

    # Labels
    ax.text(0.7, 7.05, "examples\n(in-context)", fontsize=8.5,
            color=C_PURPLE, fontweight="bold", style="italic", va="top")
    ax.text(0.7, 4.45, "query", fontsize=8.5,
            color=C_AMBER, fontweight="bold", style="italic")

    # Arrow from prompt panel to model
    _arrow(ax, (6.05, 4.9), (7.05, 4.9), color=C_DARK, lw=2)

    # Model box
    _box(ax, 7.1, 4.0, 2.3, 1.8, "GPT\n(frozen)", C_BLUE, fs=14)

    # Output arrow
    _arrow(ax, (8.25, 3.9), (8.25, 3.0), color=C_DARK, lw=2)

    # Output prediction
    _box(ax, 7.1, 2.1, 2.3, 0.9, "negative",
         C_GREEN, fs=13)
    ax.text(8.25, 1.7, "model completion",
            ha="center", fontsize=9, color=C_GREEN,
            fontweight="bold", style="italic")

    # Bottom caption
    cap = ("The model pattern-matches input → output pairs in the prompt\n"
           "and applies the same mapping to the query. Weights never change.")
    fig.text(0.5, 0.02, cap, ha="center", fontsize=10.5, color=C_DARK,
             bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                       boxstyle="round,pad=0.5"))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig7_in_context_learning")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for NLP Part 06: GPT and Generative Models")
    print(f"  EN dir: {EN_DIR}")
    print(f"  ZH dir: {ZH_DIR}")
    print()

    fig1_decoder_only_arch()
    fig2_autoregressive_step()
    fig3_scaling_laws()
    fig4_gpt_evolution()
    fig5_sampling_strategies()
    fig6_emergent_capabilities()
    fig7_in_context_learning()

    print("\nDone.")


if __name__ == "__main__":
    main()
