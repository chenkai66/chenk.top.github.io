"""Figures for NLP Part 9 — LLM Architecture Deep Dive.

Generates 7 publication-quality figures for the article on modern LLM
architectures. Each figure is saved into BOTH the English and Chinese asset
folders so the same image renders on both sites.

Figures:
    fig1_modern_block       — LLaMA-style decoder block: pre-norm, RMSNorm,
                              SwiGLU, RoPE, GQA
    fig2_kv_cache           — KV cache: redundant compute without cache vs
                              O(1) per-step compute with cache
    fig3_position_encoding  — Sinusoidal vs RoPE vs ALiBi: bias / decay
                              behavior across relative distance
    fig4_attention_variants — MHA vs MQA vs GQA: head sharing layout and
                              KV-cache memory cost
    fig5_flash_attention    — GPU memory hierarchy and tiled attention
                              (HBM vs SRAM, online softmax)
    fig6_moe                — Sparse MoE: router top-k selection across N
                              experts (Mixtral-style)
    fig7_quantization       — Numeric range / memory cost: FP16, INT8, INT4

Style: seaborn-v0_8-whitegrid, dpi=150, palette
    #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (orange).

References (verified):
    - Touvron et al., LLaMA / LLaMA 2 (Meta, 2023)
    - Su et al., RoFormer: Enhanced Transformer with Rotary Position
      Embedding (Neurocomputing 2024)
    - Press et al., Train Short, Test Long: Attention with Linear Biases
      Enables Input Length Extrapolation (ICLR 2022)
    - Shazeer, Fast Transformer Decoding: One Write-Head is All You Need
      (arXiv 2019)
    - Ainslie et al., GQA: Training Generalized Multi-Query Transformer
      Models from Multi-Head Checkpoints (EMNLP 2023)
    - Dao et al., FlashAttention / FlashAttention-2 (NeurIPS 2022, 2023)
    - Jiang et al., Mixtral of Experts (Mistral AI, 2024)
    - Frantar et al., GPTQ: Accurate Post-Training Quantization (ICLR 2023)
    - Lin et al., AWQ: Activation-aware Weight Quantization (MLSys 2024)
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

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GRAY = "#64748b"
LIGHT = "#e5e7eb"
DARK = "#1f2937"
RED = "#ef4444"

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/nlp/llm-architecture-deep-dive"
ZH_DIR = REPO / "source/_posts/zh/nlp/09-大语言模型架构深度解析"


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders, then close it."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def _rounded(ax, x, y, w, h, text, edge, fc, *, fs=10, weight="normal", tc=None):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=edge, facecolor=fc, alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, weight=weight,
        color=tc if tc else DARK,
    )


def _arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.6, style="->", mut=14):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=mut, linewidth=lw, color=color,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1 — Modern LLM block (LLaMA-style): pre-norm, RMSNorm, SwiGLU,
#            RoPE, GQA — contrasted with the original post-norm block.
# ---------------------------------------------------------------------------
def fig1_modern_block() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 6.6))

    def block(ax, title, items, accent):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis("off")
        ax.text(5, 11.6, title, ha="center", va="center",
                fontsize=13, weight="bold", color=DARK)
        # Vertical residual rail on the left
        ax.plot([1.0, 1.0], [0.6, 10.6], color=GRAY, lw=1.2, ls="--")
        ax.text(0.4, 5.6, "residual", rotation=90, color=GRAY,
                ha="center", va="center", fontsize=9)

        y = 1.0
        h = 0.95
        for label, sub, color in items:
            _rounded(ax, 2.3, y, 6.4, h, "", color, color + "22", fs=10)
            ax.text(5.5, y + h / 2 + 0.12, label, ha="center", va="center",
                    fontsize=10.5, weight="bold", color=DARK)
            ax.text(5.5, y + h / 2 - 0.22, sub, ha="center", va="center",
                    fontsize=8.6, color=GRAY)
            # add link from rail
            _arrow(ax, 1.0, y + h / 2, 2.3, y + h / 2, color=GRAY, lw=1.0)
            y += h + 0.35
        # output arrow
        _arrow(ax, 5.5, y, 5.5, y + 0.5, color=accent, lw=2)
        ax.text(5.5, y + 0.7, "to next layer", ha="center",
                fontsize=9, color=accent)

    # --- Left: original Transformer (post-norm, MHA, ReLU/GELU FFN, abs pos)
    original = [
        ("Multi-Head Self-Attention", "QKV with H heads, abs/sin pos",  BLUE),
        ("Add & LayerNorm",           "post-norm: norm AFTER residual", GRAY),
        ("Feed-Forward (GELU)",       "two linears, dense activation",  PURPLE),
        ("Add & LayerNorm",           "post-norm: norm AFTER residual", GRAY),
    ]
    block(axL, "Original Transformer block (2017)", original, GRAY)

    # --- Right: LLaMA-style modern block
    modern = [
        ("RMSNorm (pre)",         "norm BEFORE attention, no bias",   ORANGE),
        ("Grouped-Query Attn + RoPE", "GQA (e.g. 32 Q / 8 KV heads), rotary pos", BLUE),
        ("RMSNorm (pre)",         "norm BEFORE FFN",                  ORANGE),
        ("SwiGLU FFN",            "Swish(W1·x) ⊙ (W3·x) → W2",        GREEN),
    ]
    block(axR, "LLaMA-style decoder block (2023)", modern, GREEN)

    # Footer comparison strip
    fig.text(
        0.5, 0.02,
        "Pre-norm + RMSNorm = stable training without warmup ·  RoPE = relative "
        "position via rotation ·  SwiGLU ≈ +1–2% perplexity vs GELU ·  GQA shrinks "
        "the KV cache without losing quality.",
        ha="center", fontsize=9.2, color=GRAY,
    )

    plt.suptitle(
        "Figure 1 · Modern LLM block: what changed since the 2017 Transformer",
        y=0.995, fontsize=14, weight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    save(fig, "fig1_modern_block.png")


# ---------------------------------------------------------------------------
# Figure 2 — KV cache: redundant compute without cache vs O(1) per-step with
#            cache; right panel: cumulative FLOPs comparison.
# ---------------------------------------------------------------------------
def fig2_kv_cache() -> None:
    fig = plt.figure(figsize=(13, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # ---- Left: token-by-token recompute vs cache reuse
    axL.set_xlim(-0.4, 8.6)
    axL.set_ylim(-0.5, 6.4)
    axL.axis("off")
    axL.set_title("Generating 4 new tokens after a 4-token prompt",
                  loc="left", fontsize=12)

    tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "purred"]
    # Two rows: top = without cache (recompute), bottom = with cache (reuse)
    for row, (label, color, recompute) in enumerate(
        [("Without KV cache (recompute every step)", RED, True),
         ("With KV cache (reuse + append)",         GREEN, False)]
    ):
        y0 = 4.1 if row == 0 else 0.6
        axL.text(-0.3, y0 + 1.3, label, fontsize=10.5,
                 weight="bold", color=color)
        for step in range(4):  # 4 generation steps
            yy = y0 + 0.95 - step * 0.32
            for i, tok in enumerate(tokens[: 4 + step + 1]):
                if i < 4 + step:  # already-seen tokens
                    if recompute:
                        fc, ec = RED + "33", RED
                    else:
                        fc, ec = GREEN + "22", GREEN
                else:  # newly produced token
                    fc, ec = ORANGE + "55", ORANGE
                axL.add_patch(Rectangle(
                    (i * 0.95, yy), 0.85, 0.28,
                    facecolor=fc, edgecolor=ec, linewidth=1.0,
                ))
                if step == 0:
                    axL.text(i * 0.95 + 0.42, yy + 0.55, tok,
                             ha="center", fontsize=8, color=DARK)
        # legend per row
        if recompute:
            axL.text(8.0, y0 + 0.6,
                     "every cell\n recomputed",
                     fontsize=8.4, color=RED, ha="left")
        else:
            axL.text(8.0, y0 + 0.6,
                     "cached K,V\n reused",
                     fontsize=8.4, color=GREEN, ha="left")

    # ---- Right: cumulative attention FLOPs
    n = np.arange(1, 257)
    flops_no = np.cumsum(n)               # ~n^2/2
    flops_yes = np.cumsum(np.ones_like(n))  # ~n
    axR.plot(n, flops_no, color=RED, lw=2.4,
             label="Without cache  (Σ t = O(n²))")
    axR.plot(n, flops_yes, color=GREEN, lw=2.4,
             label="With cache      (Σ 1 = O(n))")
    axR.fill_between(n, flops_yes, flops_no, color=RED, alpha=0.10,
                     label="Compute saved by cache")
    axR.set_xlabel("Generated token index  n")
    axR.set_ylabel("Cumulative attention work (relative)")
    axR.set_title("Cumulative attention FLOPs", loc="left", fontsize=12)
    axR.legend(loc="upper left", fontsize=9)
    # Annotate saving at n=128
    n_mark = 128
    axR.annotate(
        f"At n={n_mark}: cache does ≈{n_mark}\nops vs ≈{n_mark*(n_mark+1)//2} without",
        xy=(n_mark, flops_no[n_mark - 1]),
        xytext=(40, flops_no[n_mark - 1] * 0.45),
        fontsize=9, color=DARK,
        arrowprops=dict(arrowstyle="->", color=DARK, lw=1.0),
    )

    plt.suptitle(
        "Figure 2 · KV cache turns O(n²) prefix recompute into O(n) per step",
        y=1.02, fontsize=14, weight="bold",
    )
    plt.tight_layout()
    save(fig, "fig2_kv_cache.png")


# ---------------------------------------------------------------------------
# Figure 3 — Position encoding behavior: sinusoidal embedding vs RoPE
#            similarity vs ALiBi penalty as a function of relative distance.
# ---------------------------------------------------------------------------
def fig3_position_encoding() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    # ----- (a) Sinusoidal: heatmap of PE_{pos, dim}
    ax = axes[0]
    pos = np.arange(64)
    dim = 64
    i = np.arange(dim)
    angle = pos[:, None] / np.power(10000, (2 * (i // 2)) / dim)
    pe = np.zeros((len(pos), dim))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    im = ax.imshow(pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("(a) Sinusoidal absolute PE\nadded at the embedding layer",
                 loc="left", fontsize=11)
    ax.set_xlabel("dimension")
    ax.set_ylabel("position")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)

    # ----- (b) RoPE: cosine similarity between rotated q and k as a
    #         function of relative distance, for several frequency pairs.
    ax = axes[1]
    rel = np.arange(0, 256)
    ax.set_title("(b) RoPE\ndecays cleanly with relative distance",
                 loc="left", fontsize=11)
    for theta_idx, color, label in [
        (1, BLUE, "low-freq pair (θ small)"),
        (16, PURPLE, "mid-freq pair"),
        (64, ORANGE, "high-freq pair (θ large)"),
    ]:
        theta = 10000 ** (-2 * theta_idx / 128)
        sim = np.cos(rel * theta)
        # apply mild long-range decay envelope for visual realism
        env = 1.0 / (1 + 0.002 * rel)
        ax.plot(rel, sim * env, color=color, lw=1.8, label=label)
    ax.axhline(0, color=GRAY, lw=0.8, ls=":")
    ax.set_xlabel("relative distance  m − n")
    ax.set_ylabel("cos(q·R_{m−n}·k) (norm.)")
    ax.legend(fontsize=8.4, loc="upper right")
    ax.set_ylim(-1.05, 1.15)

    # ----- (c) ALiBi: bias = -m * |i-j| for several heads
    ax = axes[2]
    n_heads = 8
    rel = np.arange(0, 256)
    ax.set_title("(c) ALiBi\nlinear bias added to attention scores",
                 loc="left", fontsize=11)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n_heads))
    for h in range(n_heads):
        slope = 2 ** (-8 / n_heads * (h + 1))
        ax.plot(rel, -slope * rel, color=cmap[h], lw=1.5,
                label=f"head {h}: m={slope:.4f}")
    ax.set_xlabel("relative distance  |i − j|")
    ax.set_ylabel("attention bias")
    ax.set_ylim(-25, 1)
    ax.legend(fontsize=7.2, ncol=2, loc="lower left")

    plt.suptitle(
        "Figure 3 · Three ways to inject position into attention",
        y=1.04, fontsize=14, weight="bold",
    )
    plt.tight_layout()
    save(fig, "fig3_position_encoding.png")


# ---------------------------------------------------------------------------
# Figure 4 — MHA vs MQA vs GQA: head layout + per-token KV cache memory.
# ---------------------------------------------------------------------------
def fig4_attention_variants() -> None:
    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    ax = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Head-sharing layout for the K/V projections (8 query heads shown)",
                 loc="left", fontsize=12)

    def draw_heads(x0, y0, label, q_groups, color_label):
        # 8 query heads on top
        for i in range(8):
            _rounded(ax, x0 + i * 0.42, y0 + 1.4, 0.36, 0.45, f"Q{i}",
                     BLUE, BLUE + "22", fs=8)
        # KV heads below, grouped according to q_groups
        unique = sorted(set(q_groups))
        for kv_id in unique:
            qs = [i for i, g in enumerate(q_groups) if g == kv_id]
            xleft = x0 + qs[0] * 0.42
            xright = x0 + qs[-1] * 0.42 + 0.36
            _rounded(ax, xleft, y0 + 0.55, xright - xleft, 0.45,
                     f"KV{kv_id}", PURPLE, PURPLE + "22", fs=8)
            # link arrows
            for i in qs:
                _arrow(ax,
                       x0 + i * 0.42 + 0.18, y0 + 1.4,
                       x0 + i * 0.42 + 0.18, y0 + 1.0,
                       color=GRAY, lw=0.9, mut=8)
        ax.text(x0 + 1.7, y0 + 2.15, label, fontsize=11,
                weight="bold", color=DARK, ha="center")
        ax.text(x0 + 1.7, y0 + 0.15, color_label, fontsize=8.6,
                color=GRAY, ha="center")

    # MHA: 8 KV heads (one per Q)
    draw_heads(0.4, 2.3, "MHA  (8 KV heads)",
               list(range(8)),
               "8 Q  ·  8 KV   →   1× cache size")
    # GQA-2: 2 KV heads (4 Q share each)
    draw_heads(4.6, 2.3, "GQA-2  (2 KV heads)",
               [0, 0, 0, 0, 1, 1, 1, 1],
               "8 Q  ·  2 KV   →   1/4× cache")
    # MQA: 1 KV head shared by all 8 Q heads
    draw_heads(8.8, 2.3, "MQA  (1 KV head)",
               [0] * 8,
               "8 Q  ·  1 KV   →   1/8× cache")

    # Top row: per-token KV cache memory for LLaMA-2 70B style
    # (n_layers=80, head_dim=128, fp16 = 2 bytes)
    layers = 80
    head_dim = 128
    bytes_per = 2
    n_q = 64  # query heads
    schemes = ["MHA (64 KV)", "GQA-8 (8 KV)", "MQA (1 KV)"]
    n_kv = [64, 8, 1]
    kb_per_token = [2 * layers * h * head_dim * bytes_per / 1024 for h in n_kv]
    colors = [BLUE, GREEN, PURPLE]

    bars = axR.bar(schemes, kb_per_token, color=colors, alpha=0.9,
                   edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, kb_per_token):
        axR.text(b.get_x() + b.get_width() / 2, v * 1.02,
                 f"{v/1024:.2f} MB / token",
                 ha="center", fontsize=9, color=DARK)
    axR.set_ylabel("KV cache per token (KB)")
    axR.set_title("Per-token KV memory (LLaMA-70B-shape: 80L · d_h=128 · fp16)",
                  loc="left", fontsize=11)
    axR.set_yscale("log")

    plt.suptitle(
        "Figure 4 · MHA → GQA → MQA: trading head diversity for cache size",
        y=1.02, fontsize=14, weight="bold",
    )
    plt.tight_layout()
    save(fig, "fig4_attention_variants.png")


# ---------------------------------------------------------------------------
# Figure 5 — FlashAttention: GPU memory hierarchy + tiled attention pattern.
# ---------------------------------------------------------------------------
def fig5_flash_attention() -> None:
    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15])
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # ---- Left: memory hierarchy pyramid with bandwidth annotations
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 8)
    axL.axis("off")
    axL.set_title("GPU memory hierarchy (A100 / H100 class)",
                  loc="left", fontsize=12)

    layers = [
        # (y, h, w, label, sub, color)
        (6.2, 1.2, 9.2, "DRAM / CPU host memory",
         "≈ 50 GB · ~50 GB/s PCIe link",                GRAY),
        (4.6, 1.2, 8.0, "GPU HBM (high-bandwidth memory)",
         "40–80 GB · ~1.5–3 TB/s",                      BLUE),
        (3.0, 1.2, 6.0, "L2 cache",
         "~40 MB · ~5 TB/s",                            PURPLE),
        (1.4, 1.2, 4.0, "SRAM / shared memory per SM",
         "~192 KB · ~19 TB/s",                          GREEN),
    ]
    for y, h, w, name, sub, color in layers:
        x = (10 - w) / 2
        _rounded(axL, x, y, w, h, "", color, color + "22")
        axL.text(5, y + h / 2 + 0.18, name, ha="center", weight="bold",
                 color=DARK, fontsize=10.5)
        axL.text(5, y + h / 2 - 0.22, sub, ha="center", color=GRAY, fontsize=9)

    axL.annotate(
        "FlashAttention loads Q,K,V tiles\nfrom HBM into SRAM, computes\n"
        "softmax + matmul there, and writes\n only the output back.",
        xy=(8.0, 2.0), xytext=(0.3, 0.3),
        fontsize=9, color=DARK,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2),
    )

    # ---- Right: standard vs tiled attention matrix
    axR.set_xlim(-0.5, 12.5)
    axR.set_ylim(-0.5, 7)
    axR.axis("off")
    axR.set_title("Materializing  S = QKᵀ  (n × n)  —  tiled vs full",
                  loc="left", fontsize=12)

    # full n x n matrix on left
    n = 8
    for i in range(n):
        for j in range(n):
            axR.add_patch(Rectangle(
                (j * 0.55, 5.0 - i * 0.55), 0.5, 0.5,
                facecolor=RED + "55", edgecolor=RED, linewidth=0.5,
            ))
    axR.text(2.2, 5.95, "Standard attention",
             ha="center", fontsize=10.5, weight="bold", color=RED)
    axR.text(2.2, -0.2,
             "Writes the full n×n matrix to HBM\n→ O(n²) memory and traffic",
             ha="center", fontsize=9, color=GRAY)

    # tiled version on right: only highlighted blocks resident at a time
    bx0 = 7.0
    for i in range(n):
        for j in range(n):
            tile_i, tile_j = i // 2, j // 2
            in_tile = (tile_i == 1 and tile_j == 2)
            fc = GREEN if in_tile else LIGHT
            ec = GREEN if in_tile else GRAY
            axR.add_patch(Rectangle(
                (bx0 + j * 0.55, 5.0 - i * 0.55), 0.5, 0.5,
                facecolor=fc, edgecolor=ec, linewidth=0.5,
            ))
    # block grid emphasis
    for k in range(0, n + 1, 2):
        axR.plot([bx0, bx0 + n * 0.55], [5.0 + 0.5 - k * 0.55] * 2,
                 color=DARK, lw=1.0)
        axR.plot([bx0 + k * 0.55] * 2,
                 [5.0 + 0.5, 5.0 + 0.5 - n * 0.55],
                 color=DARK, lw=1.0)
    axR.text(bx0 + 2.2, 5.95, "FlashAttention (tiled)",
             ha="center", fontsize=10.5, weight="bold", color=GREEN)
    axR.text(bx0 + 2.2, -0.2,
             "Only one block resident in SRAM\n→ O(n) memory, online softmax",
             ha="center", fontsize=9, color=GRAY)

    plt.suptitle(
        "Figure 5 · FlashAttention — same math, IO-aware schedule, 2–4× faster",
        y=1.02, fontsize=14, weight="bold",
    )
    plt.tight_layout()
    save(fig, "fig5_flash_attention.png")


# ---------------------------------------------------------------------------
# Figure 6 — MoE: router top-k selection over N experts (Mixtral-style),
#            with parameter / active-parameter counts on the right.
# ---------------------------------------------------------------------------
def fig6_moe() -> None:
    fig = plt.figure(figsize=(13.5, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Sparse MoE layer  (Mixtral-style 8 experts, top-2 routing)",
                 loc="left", fontsize=12)

    # Token input
    _rounded(ax, 0.3, 3.0, 1.6, 1.0, "token  x", BLUE, BLUE + "22",
             fs=10.5, weight="bold")

    # Router
    _rounded(ax, 2.4, 3.0, 1.8, 1.0, "Router\nW_g · x", ORANGE,
             ORANGE + "22", fs=10, weight="bold")
    _arrow(ax, 1.9, 3.5, 2.4, 3.5, color=DARK)

    # 8 experts in a column, with router scores
    np.random.seed(7)
    raw = np.array([0.31, 0.84, 0.12, 0.55, 0.91, 0.07, 0.22, 0.43])
    softmax = np.exp(raw) / np.exp(raw).sum()
    top2 = np.argsort(softmax)[-2:]

    for i in range(8):
        y = 6.0 - i * 0.74
        chosen = i in top2
        ec = GREEN if chosen else GRAY
        fc = GREEN + "33" if chosen else LIGHT
        _rounded(ax, 5.4, y, 2.2, 0.55,
                 f"Expert {i}  (FFN)", ec, fc,
                 fs=9.4, weight="bold" if chosen else "normal")
        # arrow from router with weight label
        if chosen:
            _arrow(ax, 4.2, 3.5, 5.4, y + 0.27, color=GREEN, lw=1.6)
            ax.text(4.85, y + 0.45,
                    f"g={softmax[i]:.2f}", fontsize=8.6,
                    color=GREEN, weight="bold")
        else:
            _arrow(ax, 4.2, 3.5, 5.4, y + 0.27, color=GRAY, lw=0.6,
                   style="-", mut=4)

    # Combiner
    _rounded(ax, 8.4, 3.0, 1.8, 1.0,
             "Σ gᵢ · Eᵢ(x)", PURPLE, PURPLE + "22",
             fs=10, weight="bold")
    for i in top2:
        y = 6.0 - i * 0.74 + 0.27
        _arrow(ax, 7.6, y, 8.4, 3.5, color=GREEN, lw=1.4)

    _rounded(ax, 10.4, 3.0, 1.4, 1.0, "y", BLUE, BLUE + "22",
             fs=11, weight="bold")
    _arrow(ax, 10.2, 3.5, 10.4, 3.5, color=DARK)

    ax.text(6.5, 0.1,
            "Each token activates only K=2 of N=8 experts.  Total params grow with N, "
            "but per-token compute stays at K-expert cost.",
            ha="center", fontsize=9, color=GRAY)

    # ---- Right: total vs active parameters for representative MoE models
    models = ["Mixtral\n8x7B", "DeepSeek-MoE\n16B", "Qwen1.5-MoE\nA2.7B", "Switch-C\n1.6T"]
    total = [46.7, 16.4, 14.3, 1571]
    active = [12.9, 2.8, 2.7, 7.0]

    x = np.arange(len(models))
    w = 0.35
    bars1 = axR.bar(x - w / 2, total, w, color=PURPLE, alpha=0.85,
                    label="Total params (B)")
    bars2 = axR.bar(x + w / 2, active, w, color=GREEN, alpha=0.9,
                    label="Active per token (B)")
    axR.set_yscale("log")
    axR.set_xticks(x)
    axR.set_xticklabels(models, fontsize=9)
    axR.set_ylabel("Parameters (billions, log)")
    axR.set_title("Total vs active parameters across MoE LLMs",
                  loc="left", fontsize=11)
    axR.legend(loc="upper left", fontsize=9)
    for b, v in zip(bars1, total):
        axR.text(b.get_x() + b.get_width() / 2, v * 1.1,
                 f"{v:g}", ha="center", fontsize=8, color=PURPLE)
    for b, v in zip(bars2, active):
        axR.text(b.get_x() + b.get_width() / 2, v * 1.1,
                 f"{v:g}", ha="center", fontsize=8, color=GREEN)

    plt.suptitle(
        "Figure 6 · MoE — sparse activation grows capacity without compute",
        y=1.02, fontsize=14, weight="bold",
    )
    plt.tight_layout()
    save(fig, "fig6_moe.png")


# ---------------------------------------------------------------------------
# Figure 7 — Quantization: numeric grid + memory footprint at FP16/INT8/INT4.
# ---------------------------------------------------------------------------
def fig7_quantization() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    # ---- (a) Quantization grid: original FP weights + INT8 / INT4 buckets
    ax = axes[0]
    rng = np.random.default_rng(42)
    w = rng.normal(0, 0.5, 800)
    w = np.clip(w, -1.6, 1.6)
    ax.hist(w, bins=80, color=GRAY, alpha=0.55, label="FP16 weight dist.")

    # INT8 bucket centers (256 levels) — too dense to draw; show INT4 (16)
    int4_levels = np.linspace(-1.6, 1.6, 16)
    for lv in int4_levels:
        ax.axvline(lv, color=ORANGE, lw=0.8, alpha=0.85)
    ax.axvline(int4_levels[0], color=ORANGE, lw=0.8, label="INT4 grid (16 levels)")
    ax.set_xlabel("weight value")
    ax.set_ylabel("count")
    ax.set_title("(a) FP16 weights mapped onto an INT4 grid", loc="left")
    ax.legend(fontsize=8.4)

    # ---- (b) Bytes per parameter at each precision
    ax = axes[1]
    precisions = ["FP32", "FP16 / BF16", "INT8", "INT4", "INT3*", "INT2*"]
    bytes_per = [4, 2, 1, 0.5, 0.375, 0.25]
    colors = [GRAY, BLUE, PURPLE, GREEN, ORANGE, RED]
    bars = ax.bar(precisions, bytes_per, color=colors, alpha=0.9,
                  edgecolor="white", linewidth=1.4)
    for b, v in zip(bars, bytes_per):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05,
                f"{v} B", ha="center", fontsize=9.2, color=DARK)
    ax.set_ylabel("bytes per parameter")
    ax.set_title("(b) Per-parameter cost", loc="left")
    ax.set_ylim(0, 4.6)
    ax.tick_params(axis="x", rotation=20)

    # ---- (c) Total weight memory for 7B / 13B / 70B
    ax = axes[2]
    sizes = ["7B", "13B", "70B"]
    params = np.array([7.0, 13.0, 70.0])  # billions
    fp16_gb = params * 2
    int8_gb = params * 1
    int4_gb = params * 0.5

    x = np.arange(len(sizes))
    w_ = 0.25
    ax.bar(x - w_, fp16_gb, w_, color=BLUE, alpha=0.9, label="FP16")
    ax.bar(x,        int8_gb, w_, color=PURPLE, alpha=0.9, label="INT8")
    ax.bar(x + w_,  int4_gb, w_, color=GREEN, alpha=0.9, label="INT4")
    for i, p in enumerate(params):
        for off, gb, c in [(-w_, p * 2, BLUE),
                           (0,    p * 1, PURPLE),
                           (w_,   p * 0.5, GREEN)]:
            ax.text(i + off, gb + 1, f"{gb:.0f} GB",
                    ha="center", fontsize=8.6, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("weight memory (GB)")
    ax.set_title("(c) Model weights memory", loc="left")
    ax.legend(fontsize=9)

    fig.text(
        0.5, -0.02,
        "* INT3 / INT2 require advanced calibration (GPTQ, AWQ, SqueezeLLM); "
        "INT4 is the current sweet spot for >7B LLMs (typically <2% perplexity loss).",
        ha="center", fontsize=8.8, color=GRAY,
    )
    plt.suptitle(
        "Figure 7 · Quantization shrinks the memory footprint of LLM weights",
        y=1.04, fontsize=14, weight="bold",
    )
    plt.tight_layout()
    save(fig, "fig7_quantization.png")


# ---------------------------------------------------------------------------
def main() -> None:
    fig1_modern_block()
    fig2_kv_cache()
    fig3_position_encoding()
    fig4_attention_variants()
    fig5_flash_attention()
    fig6_moe()
    fig7_quantization()
    print("[ok] generated 7 figures into:")
    print(f"      {EN_DIR}")
    print(f"      {ZH_DIR}")


if __name__ == "__main__":
    main()
