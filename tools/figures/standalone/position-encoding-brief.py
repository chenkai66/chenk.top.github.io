"""
Figure generation script for the Position Encoding Brief article.

Generates 5 figures used in both EN and ZH versions of the article. Each
figure teaches one specific idea cleanly, with no visual clutter.

Figures:
    fig1_sinusoidal       Sinusoidal position encoding heatmap (positions x
                          dimensions) plus one row showing how a single
                          position is the unique fingerprint of multi-scale
                          sines and cosines.
    fig2_learned          Learned position encoding: a randomly-initialized
                          matrix gradually shaping into a smooth, position
                          aware representation; contrasts with sinusoidal
                          determinism and shows the extrapolation cliff at
                          the trained context length.
    fig3_rope             RoPE rotation visualization: query/key vectors in
                          a 2D subspace rotated by an angle proportional to
                          their absolute position; shows that the dot
                          product depends only on the relative offset.
    fig4_alibi            ALiBi (Attention with Linear Biases): the linear
                          distance penalty added to the QK^T attention
                          scores, plotted both as a 2D bias matrix and as
                          per-head slopes.
    fig5_compare          Comparison: long-context perplexity (or accuracy)
                          for sinusoidal vs. RoPE vs. ALiBi as the test
                          length exceeds the training length, showing
                          extrapolation behavior.

Usage:
    python3 scripts/figures/standalone/position-encoding-brief.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"
C_BG = "#f8fafc"
C_RED = "#ef4444"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "position-encoding-brief"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "standalone" / "浅谈位置编码"


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Sinusoidal position encoding
# ---------------------------------------------------------------------------
def fig1_sinusoidal() -> None:
    """Sinusoidal PE heatmap + per-position 'fingerprint' interpretation."""
    seq_len = 100
    d_model = 128

    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    div_term = np.exp(-(np.log(10000.0) * (2 * (i // 2)) / d_model))
    angles = pos * div_term
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.28)

    # ---- Left: heatmap ----
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(
        pe,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        origin="lower",
    )
    ax.set_xlabel("dimension index $i$", fontsize=11, color=C_DARK)
    ax.set_ylabel("position $p$", fontsize=11, color=C_DARK)
    ax.set_title(
        "Sinusoidal PE: low dims rotate fast, high dims rotate slowly",
        fontsize=12, color=C_DARK, pad=8,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("PE value", fontsize=10, color=C_DARK)

    # mark a single position to point at the right panel
    p_show = 30
    ax.axhline(p_show, color=C_AMBER, lw=1.6, alpha=0.9)
    ax.text(
        d_model + 2, p_show, f"p = {p_show}",
        va="center", ha="left", color=C_AMBER, fontsize=10, weight="bold",
        clip_on=False,
    )

    # ---- Right: a single row, two scales ----
    ax2 = fig.add_subplot(gs[0, 1])
    row = pe[p_show]
    ax2.plot(row[:32], color=C_BLUE, lw=1.4, label="dims 0-31 (high freq)")
    ax2.plot(
        np.arange(32, 64), row[32:64],
        color=C_PURPLE, lw=1.4, label="dims 32-63 (mid)",
    )
    ax2.plot(
        np.arange(64, 128), row[64:],
        color=C_GREEN, lw=1.4, label="dims 64-127 (low freq)",
    )
    ax2.axhline(0, color=C_GRAY, lw=0.8, alpha=0.6)
    ax2.set_xlabel("dimension index $i$", fontsize=11, color=C_DARK)
    ax2.set_ylabel("PE value", fontsize=11, color=C_DARK)
    ax2.set_title(
        f"Position p = {p_show} as a multi-scale fingerprint",
        fontsize=12, color=C_DARK, pad=8,
    )
    ax2.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
    ax2.set_ylim(-1.15, 1.15)

    fig.suptitle(
        "Sinusoidal positional encoding (Vaswani et al., 2017)",
        fontsize=13.5, color=C_DARK, y=1.02,
    )
    _save(fig, "fig1_sinusoidal")


# ---------------------------------------------------------------------------
# Figure 2: Learned position encoding
# ---------------------------------------------------------------------------
def fig2_learned() -> None:
    """Learned PE: random init -> trained pattern + extrapolation cliff."""
    rng = np.random.default_rng(7)
    seq_len = 64
    d_model = 64

    init = rng.normal(0.0, 0.3, size=(seq_len, d_model))

    # Synthesize a "trained" PE: low-frequency smooth columns plus noise.
    p = np.arange(seq_len)[:, None] / seq_len
    k = np.arange(d_model)[None, :]
    freq = 0.5 + 1.5 * (k / d_model)
    phase = rng.uniform(0, 2 * np.pi, size=(1, d_model))
    trained = np.sin(2 * np.pi * freq * p + phase) * (0.5 + 0.5 * (1 - k / d_model))
    trained += rng.normal(0.0, 0.08, size=trained.shape)

    fig = plt.figure(figsize=(13.0, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.32)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(init, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax0.set_title("Initialization (random)", fontsize=11, color=C_DARK, pad=6)
    ax0.set_xlabel("dim", fontsize=10)
    ax0.set_ylabel("position", fontsize=10)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(trained, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, origin="lower")
    ax1.set_title("After training (smooth, structured)", fontsize=11, color=C_DARK, pad=6)
    ax1.set_xlabel("dim", fontsize=10)
    ax1.set_yticks([])

    # ---- Extrapolation cliff ----
    ax2 = fig.add_subplot(gs[0, 2])
    train_len = 64
    test_len = 128
    xs = np.arange(test_len)
    # Performance proxy (perplexity): flat in-domain, sharp degradation past
    # the trained max length.
    over = np.maximum(xs - train_len, 0).astype(float)
    perf = np.where(
        xs < train_len,
        18.0 - 0.01 * xs,
        18.0 + 4.5 * (over / 8.0) ** 1.4,
    )
    perf = np.clip(perf, 16.5, 90)
    ax2.plot(xs[:train_len], perf[:train_len], color=C_BLUE, lw=2.0,
             label="in-domain (trained)")
    ax2.plot(xs[train_len:], perf[train_len:], color=C_RED, lw=2.0,
             label="beyond training length")
    ax2.axvline(train_len, color=C_GRAY, lw=1.2, ls="--")
    ax2.text(
        train_len + 1, 80, "training\nmax length",
        color=C_GRAY, fontsize=9, va="top",
    )
    ax2.set_xlabel("test sequence length", fontsize=10)
    ax2.set_ylabel("perplexity (lower is better)", fontsize=10)
    ax2.set_title("Learned PE has no extrapolation", fontsize=11, color=C_DARK, pad=6)
    ax2.set_ylim(15, 95)
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "Learned positional encoding: flexible inside training range, brittle outside",
        fontsize=13.5, color=C_DARK, y=1.02,
    )
    _save(fig, "fig2_learned")


# ---------------------------------------------------------------------------
# Figure 3: RoPE rotation
# ---------------------------------------------------------------------------
def fig3_rope() -> None:
    """RoPE: rotate (q, k) by position; dot product depends only on offset."""
    fig = plt.figure(figsize=(13.0, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.1], wspace=0.32)

    # ---- Left: rotating q at three positions ----
    ax = fig.add_subplot(gs[0, 0])
    ax.set_aspect("equal")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axhline(0, color=C_GRAY, lw=0.8, alpha=0.6)
    ax.axvline(0, color=C_GRAY, lw=0.8, alpha=0.6)

    # base unrotated query
    base = np.array([1.0, 0.25])
    base /= np.linalg.norm(base)

    theta = 0.6  # rotation per position step
    positions = [0, 2, 4]
    colors = [C_BLUE, C_PURPLE, C_GREEN]
    for p, c in zip(positions, colors):
        a = p * theta
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        v = R @ base
        ax.add_patch(FancyArrowPatch(
            (0, 0), (v[0], v[1]),
            arrowstyle="-|>", mutation_scale=14, color=c, lw=2.0,
        ))
        ax.text(
            v[0] * 1.12, v[1] * 1.12, f"q at p={p}",
            color=c, fontsize=10, weight="bold",
            ha="center", va="center",
        )

    # arc showing rotation between p=0 and p=4
    arc_pts = np.linspace(0, 4 * theta, 60)
    rr = 0.55
    ax.plot(rr * np.cos(arc_pts), rr * np.sin(arc_pts),
            color=C_AMBER, lw=1.4)
    ax.text(0.05, 0.78, r"angle $\propto$ position",
            color=C_AMBER, fontsize=10, weight="bold")

    ax.set_title(
        "RoPE rotates each (q, k) pair by an angle proportional to its position",
        fontsize=11, color=C_DARK, pad=8,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    # ---- Right: dot product depends only on relative offset ----
    ax2 = fig.add_subplot(gs[0, 1])
    offsets = np.arange(-30, 31)
    # Relative inner-product structure under RoPE: cos of angle * decay
    base_inner = np.cos(theta * offsets)
    decay = np.exp(-np.abs(offsets) / 18.0)
    inner = base_inner * decay
    ax2.plot(offsets, inner, color=C_BLUE, lw=2.0)
    ax2.fill_between(offsets, 0, inner, color=C_BLUE, alpha=0.12)
    ax2.axhline(0, color=C_GRAY, lw=0.8)
    ax2.axvline(0, color=C_GRAY, lw=0.8, ls="--", alpha=0.6)
    ax2.set_xlabel(r"relative offset  $m - n$", fontsize=11, color=C_DARK)
    ax2.set_ylabel(r"$\langle R_m\,q,\; R_n\,k\rangle$", fontsize=11, color=C_DARK)
    ax2.set_title(
        r"Inner product is a function of $(m - n)$ only",
        fontsize=11, color=C_DARK, pad=8,
    )

    fig.suptitle(
        "Rotary Position Embedding (RoPE): absolute rotation, relative inner product",
        fontsize=13.5, color=C_DARK, y=1.02,
    )
    _save(fig, "fig3_rope")


# ---------------------------------------------------------------------------
# Figure 4: ALiBi
# ---------------------------------------------------------------------------
def fig4_alibi() -> None:
    """ALiBi: linear distance bias added to attention scores; per-head slopes."""
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.3)

    # ---- Left: ALiBi bias matrix for one head (causal) ----
    n = 16
    slope = 0.5
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    bias = -slope * np.abs(i - j).astype(float)
    # mask future (causal)
    mask = j > i
    bias_display = bias.copy()
    bias_display[mask] = np.nan

    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color=C_LIGHT)
    im = ax.imshow(bias_display, cmap=cmap, origin="upper", aspect="equal")
    ax.set_xlabel("key position $j$", fontsize=11, color=C_DARK)
    ax.set_ylabel("query position $i$", fontsize=11, color=C_DARK)
    ax.set_title(
        f"Per-head distance bias  $-m\\,|i-j|$  (slope $m={slope}$)",
        fontsize=11, color=C_DARK, pad=8,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("bias added to QK^T", fontsize=10, color=C_DARK)

    # ---- Right: per-head slopes ----
    ax2 = fig.add_subplot(gs[0, 1])
    H = 8
    slopes = 1.0 / (2.0 ** (np.arange(1, H + 1) * (8.0 / H)))
    dist = np.arange(0, 64)
    palette = plt.cm.viridis(np.linspace(0.05, 0.9, H))
    for h, m in enumerate(slopes):
        ax2.plot(
            dist, -m * dist, color=palette[h], lw=1.8,
            label=f"head {h + 1}  ($m$={m:.3f})",
        )
    ax2.set_xlabel(r"distance $|i - j|$", fontsize=11, color=C_DARK)
    ax2.set_ylabel("attention bias", fontsize=11, color=C_DARK)
    ax2.set_title(
        "Different heads decay at different rates",
        fontsize=11, color=C_DARK, pad=8,
    )
    ax2.legend(loc="lower left", fontsize=8, ncol=2, framealpha=0.95)

    fig.suptitle(
        "ALiBi: a fixed, parameter-free distance penalty added to attention scores",
        fontsize=13.5, color=C_DARK, y=1.02,
    )
    _save(fig, "fig4_alibi")


# ---------------------------------------------------------------------------
# Figure 5: Comparison
# ---------------------------------------------------------------------------
def fig5_compare() -> None:
    """Long-context extrapolation: sinusoidal vs RoPE vs ALiBi."""
    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    train_len = 1024
    test_lens = np.array([
        128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
    ])

    # Stylized perplexity curves: in-distribution all flat, out-of-distribution
    # diverge with rates roughly matching reported behavior.
    base = 12.0

    def degrade(L, k_in, k_out, hard=False):
        over = np.maximum((L - train_len) / train_len, 0.0)
        out = np.where(
            L <= train_len,
            base + k_in * (1024 - L) / 1024,
            base + k_out * over ** (2.0 if hard else 1.1),
        )
        return out

    sin_pe = degrade(test_lens, 0.0, 22.0, hard=True)         # explodes
    rope = degrade(test_lens, 0.0, 6.5, hard=False)            # degrades slowly
    alibi = degrade(test_lens, 0.0, 1.6, hard=False)           # very stable

    # clip for visualization
    sin_pe = np.minimum(sin_pe, 90)

    ax.plot(test_lens, sin_pe, color=C_AMBER, lw=2.2, marker="o",
            markersize=6, label="Sinusoidal (absolute)")
    ax.plot(test_lens, rope, color=C_PURPLE, lw=2.2, marker="s",
            markersize=6, label="RoPE")
    ax.plot(test_lens, alibi, color=C_GREEN, lw=2.2, marker="^",
            markersize=7, label="ALiBi")

    ax.axvline(train_len, color=C_GRAY, lw=1.2, ls="--")
    ax.text(
        train_len * 1.03, 80,
        "training context\nlength = 1024",
        color=C_GRAY, fontsize=10, va="top",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(test_lens)
    ax.set_xticklabels([str(int(x)) for x in test_lens], rotation=0, fontsize=9)
    ax.set_xlabel("evaluation context length (tokens, log scale)",
                  fontsize=11, color=C_DARK)
    ax.set_ylabel("perplexity (lower is better)", fontsize=11, color=C_DARK)
    ax.set_ylim(10, 95)
    ax.set_title(
        "Length extrapolation: how each scheme behaves past its training length",
        fontsize=12.5, color=C_DARK, pad=10,
    )
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.96)

    # annotation arrows
    ax.annotate(
        "absolute PE\nblows up",
        xy=(test_lens[5], sin_pe[5]), xytext=(test_lens[2], 70),
        fontsize=10, color=C_AMBER, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.2),
    )
    ax.annotate(
        "ALiBi extrapolates\nalmost for free",
        xy=(test_lens[-2], alibi[-2]), xytext=(test_lens[3], 40),
        fontsize=10, color=C_GREEN, ha="center",
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2),
    )

    # Indicative numbers (illustrative, not from a single benchmark).
    fig.text(
        0.5, -0.03,
        "Curves are illustrative: actual numbers depend on the model, dataset, "
        "and decoding setup, but the qualitative ordering matches reports in "
        "Press et al. (2022) and the RoFormer paper.",
        ha="center", fontsize=8.5, color=C_GRAY,
    )

    _save(fig, "fig5_compare")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Position Encoding Brief figures...")
    fig1_sinusoidal()
    fig2_learned()
    fig3_rope()
    fig4_alibi()
    fig5_compare()
    print("Done.")


if __name__ == "__main__":
    main()
