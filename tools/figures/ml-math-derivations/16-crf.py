"""
Figure generation script for ML Math Derivations Part 16:
Conditional Random Fields.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates a single specific idea so the math becomes visible.

Figures:
    fig1_chain_structure        Linear-chain CRF graph: undirected cliques
                                between adjacent labels conditioned on the
                                full observation sequence (vs HMM directed
                                generative graph for contrast).
    fig2_feature_templates      Feature templates for NER (BIO scheme) on a
                                concrete sentence. Shows how transition and
                                state features fire at each position.
    fig3_forward_backward       Forward and backward trellis on a small label
                                set. Highlights one (alpha_t * beta_t) cell
                                and the marginal it produces.
    fig4_viterbi_decoding       Viterbi trellis: per-cell delta scores and
                                the back-pointer trace recovering the best
                                label sequence.
    fig5_hmm_memm_crf           HMM vs MEMM vs CRF: directed generative,
                                directed locally-normalised, and undirected
                                globally-normalised graphs side-by-side.
    fig6_generative_vs_disc     Generative vs discriminative: how the two
                                families partition probability and what each
                                spends capacity on (joint surface vs
                                conditional decision boundary).
    fig7_ner_tagging            End-to-end NER application: an input
                                sentence, the per-token marginals
                                (confidence) and the Viterbi-decoded BIO
                                tags with span boxes.

Usage:
    python3 scripts/figures/ml-math-derivations/16-crf.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS, annotate_callout  # noqa: E402, F401
setup_style()

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# style applied via _style.setup_style()

C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG = COLORS["bg"]

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "16-Conditional-Random-Fields"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "16-条件随机场"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tiny graph helpers (we draw nodes manually to keep full control)
# ---------------------------------------------------------------------------
def _node(ax, x, y, label, color=C_BLUE, r=0.32, text_color="white",
          fontsize=11, fontweight="bold"):
    circ = Circle((x, y), r, facecolor=color, edgecolor=C_DARK,
                  linewidth=1.4, zorder=3)
    ax.add_patch(circ)
    ax.text(x, y, label, ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight=fontweight, zorder=4)


def _box(ax, x, y, w, h, label, color=C_LIGHT, edge=C_DARK,
         text_color=C_DARK, fontsize=10, fontweight="normal"):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor=edge, linewidth=1.2, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight=fontweight, zorder=4)


def _arrow(ax, p1, p2, color=C_DARK, lw=1.4, style="-|>", mutation=14,
           ls="-", alpha=1.0):
    a = FancyArrowPatch(
        p1, p2, arrowstyle=style, mutation_scale=mutation,
        color=color, lw=lw, linestyle=ls, alpha=alpha, zorder=2,
    )
    ax.add_patch(a)


def _undirected(ax, p1, p2, color=C_DARK, lw=1.6, alpha=1.0):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw,
            alpha=alpha, zorder=2, solid_capstyle="round")


# ---------------------------------------------------------------------------
# Figure 1: Linear-chain CRF structure (with HMM comparison)
# ---------------------------------------------------------------------------
def fig1_chain_structure() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.4))

    # --- Top: HMM directed generative ---
    ax = axes[0]
    T = 5
    xs = np.linspace(1.0, 9.0, T)
    y_lab, y_obs = 1.55, 0.25
    for t, x in enumerate(xs, start=1):
        _node(ax, x, y_lab, f"$y_{t}$", color=C_PURPLE)
        _node(ax, x, y_obs, f"$x_{t}$", color=C_AMBER)
        _arrow(ax, (x, y_lab - 0.32), (x, y_obs + 0.32),
               color=C_DARK, lw=1.4)
    for i in range(T - 1):
        _arrow(ax, (xs[i] + 0.32, y_lab), (xs[i + 1] - 0.32, y_lab),
               color=C_DARK, lw=1.4)
    ax.text(0.25, y_lab, "HMM", fontsize=12, fontweight="bold",
            color=C_DARK, va="center")
    ax.text(0.25, y_lab - 0.45,
            r"directed, generative$P(\mathbf{X},\mathbf{Y})$",
            fontsize=9.5, color=C_GRAY, va="center")
    ax.text(5.0, 2.25,
            r"emissions $P(x_t\mid y_t)$ assume observation independence",
            ha="center", fontsize=10, color=C_DARK, style="italic")
    ax.set_xlim(-0.2, 10)
    ax.set_ylim(-0.3, 2.55)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Bottom: linear-chain CRF (undirected, conditioned on X) ---
    ax = axes[1]
    y_lab, y_obs = 1.55, 0.25
    # Observation block (whole sequence X is shared, drawn as a strip)
    strip_w = (xs[-1] - xs[0]) + 1.4
    strip_x = (xs[0] + xs[-1]) / 2 - strip_w / 2
    rect = FancyBboxPatch(
        (strip_x, y_obs - 0.32), strip_w, 0.64,
        boxstyle="round,pad=0.02,rounding_size=0.18",
        facecolor=C_AMBER, edgecolor=C_DARK, linewidth=1.4,
        alpha=0.85, zorder=3,
    )
    ax.add_patch(rect)
    ax.text(
        (xs[0] + xs[-1]) / 2, y_obs,
        r"observation sequence $\mathbf{X} = (x_1, x_2, \dots, x_T)$",
        ha="center", va="center", color="white", fontsize=11,
        fontweight="bold", zorder=4,
    )
    for t, x in enumerate(xs, start=1):
        _node(ax, x, y_lab, f"$y_{t}$", color=C_BLUE)
        # Each y_t conditions on entire X (light dashed line down to strip)
        ax.plot([x, x], [y_lab - 0.32, y_obs + 0.32],
                color=C_GRAY, lw=1.0, ls=":", zorder=1)
    for i in range(T - 1):
        _undirected(ax, (xs[i] + 0.32, y_lab), (xs[i + 1] - 0.32, y_lab),
                    color=C_DARK, lw=2.0)
    ax.text(0.25, y_lab, "CRF", fontsize=12, fontweight="bold",
            color=C_DARK, va="center")
    ax.text(0.25, y_lab - 0.45,
            r"undirected, $P(\mathbf{Y}\mid\mathbf{X})$",
            fontsize=9.5, color=C_GRAY, va="center")
    ax.text(5.0, 2.25,
            r"each clique $(y_{t-1}, y_t, \mathbf{X})$ may use any feature of"
            r" the whole $\mathbf{X}$",
            ha="center", fontsize=10, color=C_DARK, style="italic")
    ax.set_xlim(-0.2, 10)
    ax.set_ylim(-0.3, 2.55)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(
        "Linear-chain CRF vs HMM: undirected, conditional, no observation"
        " independence",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.tight_layout()
    _save(fig, "fig1_chain_structure")


# ---------------------------------------------------------------------------
# Figure 2: Feature templates on an NER example
# ---------------------------------------------------------------------------
def fig2_feature_templates() -> None:
    tokens = ["Barack", "Obama", "visited", "New", "York"]
    tags = ["B-PER", "I-PER", "O", "B-LOC", "I-LOC"]
    tag_colors = {
        "B-PER": C_BLUE, "I-PER": C_BLUE,
        "B-LOC": C_GREEN, "I-LOC": C_GREEN,
        "O": C_GRAY,
    }
    T = len(tokens)
    xs = np.linspace(1.0, 9.0, T)

    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    y_tok = 4.6
    y_tag = 3.4
    # Tokens
    for x, w in zip(xs, tokens):
        _box(ax, x, y_tok, 1.4, 0.7, w, color=C_BG, edge=C_DARK,
             fontsize=11, fontweight="bold")
    # Tags
    for x, t in zip(xs, tags):
        _box(ax, x, y_tag, 1.4, 0.7, t, color=tag_colors[t], edge=C_DARK,
             text_color="white", fontsize=10.5, fontweight="bold")
        ax.plot([x, x], [y_tok - 0.35, y_tag + 0.35],
                color=C_GRAY, lw=1.0, ls=":")
    # Transition arcs between tags
    for i in range(T - 1):
        _undirected(ax, (xs[i] + 0.7, y_tag),
                    (xs[i + 1] - 0.7, y_tag),
                    color=C_DARK, lw=1.8)

    # Highlight one position (t=4: "New" -> B-LOC) with feature box
    t_focus = 3
    x_focus = xs[t_focus]
    # State features panel
    sf_y = 1.55
    _box(ax, x_focus - 1.85, sf_y, 3.2, 1.6,
         "", color=C_BG, edge=C_BLUE, fontsize=10)
    ax.text(x_focus - 1.85, sf_y + 0.55,
            r"state features $s_l(y_t,\mathbf{X},t)$",
            ha="center", fontsize=10, color=C_BLUE, fontweight="bold")
    feats_state = [
        r"$\mathbb{1}[y_t=\mathrm{B\text{-}LOC},\ x_t\ \mathrm{capitalized}]$",
        r"$\mathbb{1}[y_t=\mathrm{B\text{-}LOC},\ x_{t+1}=\mathrm{York}]$",
        r"$\mathbb{1}[y_t=\mathrm{B\text{-}LOC},\ \mathrm{suffix}{=}\text{ew}]$",
    ]
    for k, fs in enumerate(feats_state):
        ax.text(x_focus - 1.85, sf_y + 0.18 - 0.32 * k, fs,
                ha="center", fontsize=9.5, color=C_DARK)

    # Transition features panel
    tf_y = 1.55
    _box(ax, x_focus + 1.85, tf_y, 3.2, 1.6,
         "", color=C_BG, edge=C_AMBER, fontsize=10)
    ax.text(x_focus + 1.85, tf_y + 0.55,
            r"transition features $t_k(y_{t-1},y_t,\mathbf{X},t)$",
            ha="center", fontsize=10, color=C_AMBER, fontweight="bold")
    feats_trans = [
        r"$\mathbb{1}[y_{t-1}=\mathrm{O},\ y_t=\mathrm{B\text{-}LOC}]$",
        r"$\mathbb{1}[y_{t-1}=\mathrm{B\text{-}LOC},\ y_t=\mathrm{I\text{-}LOC}]$",
        r"$\mathbb{1}[y_{t-1}=\mathrm{I\text{-}PER},\ y_t=\mathrm{B\text{-}LOC}]$",
    ]
    for k, fs in enumerate(feats_trans):
        ax.text(x_focus + 1.85, tf_y + 0.18 - 0.32 * k, fs,
                ha="center", fontsize=9.5, color=C_DARK)

    # Pointer from focus position
    _arrow(ax, (x_focus, y_tag - 0.4), (x_focus - 1.85, sf_y + 0.85),
           color=C_BLUE, lw=1.4, style="-|>", mutation=14)
    _arrow(ax, (x_focus, y_tag - 0.4), (x_focus + 1.85, sf_y + 0.85),
           color=C_AMBER, lw=1.4, style="-|>", mutation=14)

    ax.text(5.0, 5.6,
            "Feature templates fire on (token, tag) and (prev tag, tag)"
            " pairs -- features may overlap",
            ha="center", fontsize=11, color=C_DARK, fontweight="bold")
    ax.text(5.0, 0.25,
            r"score at $t$:  $\mathbf{w}^\top \mathbf{f}(y_{t-1},y_t,\mathbf{X},t)"
            r" = \sum_k \lambda_k t_k + \sum_l \mu_l s_l$",
            ha="center", fontsize=11, color=C_DARK)

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.1, 6.1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    _save(fig, "fig2_feature_templates")


# ---------------------------------------------------------------------------
# Figure 3: Forward-backward trellis
# ---------------------------------------------------------------------------
def fig3_forward_backward() -> None:
    T = 5
    L = 3
    labels = ["B", "I", "O"]
    xs = np.arange(T) * 1.7 + 1.2
    ys = np.arange(L) * 1.3 + 1.0  # bottom -> top: O, I, B (we reverse)
    # We want B on top, O on bottom -> reverse for plotting
    ys = ys[::-1]

    rng = np.random.default_rng(11)
    alpha = rng.uniform(0.25, 1.0, size=(T, L))
    beta = rng.uniform(0.25, 1.0, size=(T, L))
    Z = (alpha * beta).sum(axis=1, keepdims=True) + 1e-9
    marginal = alpha * beta / Z  # not used directly, just for shading

    fig, ax = plt.subplots(figsize=(11.5, 5.4))

    # Forward arrows (left -> right) faint
    for t in range(T - 1):
        for i in range(L):
            for j in range(L):
                _arrow(ax, (xs[t] + 0.32, ys[i]),
                       (xs[t + 1] - 0.32, ys[j]),
                       color=C_BLUE, lw=0.6, style="-|>", mutation=8,
                       alpha=0.18)
    # Backward arrows (right -> left) faint
    for t in range(T - 1, 0, -1):
        for i in range(L):
            for j in range(L):
                _arrow(ax, (xs[t] - 0.32, ys[i]),
                       (xs[t - 1] + 0.32, ys[j]),
                       color=C_PURPLE, lw=0.6, style="-|>", mutation=8,
                       alpha=0.12)

    # Highlight cell (t=2, j=1) -- middle position, label I
    t_h, j_h = 2, 1
    # Highlighted forward path coming in
    for i in range(L):
        _arrow(ax, (xs[t_h - 1] + 0.32, ys[i]),
               (xs[t_h] - 0.32, ys[j_h]),
               color=C_BLUE, lw=2.0, style="-|>", mutation=14, alpha=0.95)
    # Highlighted backward path going out
    for j in range(L):
        _arrow(ax, (xs[t_h] + 0.32, ys[j_h]),
               (xs[t_h + 1] - 0.32, ys[j]),
               color=C_PURPLE, lw=2.0, style="-|>", mutation=14, alpha=0.95)

    # Nodes
    for t in range(T):
        for k, lab in enumerate(labels):
            color = C_LIGHT
            txtcol = C_DARK
            if t == t_h and k == j_h:
                color = C_GREEN
                txtcol = "white"
            _node(ax, xs[t], ys[k], lab, color=color, text_color=txtcol,
                  fontsize=11)

    # Time axis labels
    for t in range(T):
        ax.text(xs[t], -0.05, f"$t={t+1}$", ha="center", fontsize=10,
                color=C_DARK)
    # Label axis on the left
    for k, lab in enumerate(labels):
        ax.text(0.45, ys[k], lab, ha="right", va="center", fontsize=11,
                color=C_DARK, fontweight="bold")
    ax.text(0.35, (ys[0] + ys[-1]) / 2 + 1.0, "label",
            ha="right", fontsize=10, color=C_GRAY, rotation=90)

    # Legend / formula
    ax.text(xs[t_h], ys[j_h] + 0.95,
            r"$P(y_t=j\mid\mathbf{X}) = \dfrac{\alpha_t(j)\,\beta_t(j)}{Z(\mathbf{X})}$",
            ha="center", fontsize=12, color=C_GREEN, fontweight="bold")
    # forward / backward labels
    ax.text(1.1, ys[0] + 0.85, r"$\alpha_t(j)$ -- forward sums",
            color=C_BLUE, fontsize=10.5, fontweight="bold")
    ax.text(xs[-1] - 0.4, ys[0] + 0.85,
            r"$\beta_t(j)$ -- backward sums",
            color=C_PURPLE, fontsize=10.5, fontweight="bold", ha="right")

    ax.set_xlim(0.0, xs[-1] + 0.8)
    ax.set_ylim(-0.45, ys[0] + 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Forward-backward on the CRF trellis: "
        r"marginal $\propto \alpha_t(j)\,\beta_t(j)$, complexity $O(TL^2)$",
        fontsize=12.5, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    _save(fig, "fig3_forward_backward")


# ---------------------------------------------------------------------------
# Figure 4: Viterbi decoding
# ---------------------------------------------------------------------------
def fig4_viterbi_decoding() -> None:
    T = 6
    L = 3
    labels = ["B-LOC", "I-LOC", "O"]
    xs = np.arange(T) * 1.85 + 1.3
    ys = (np.arange(L) * 1.35 + 1.0)[::-1]

    # Hand-designed delta values that produce a sensible best path:
    # best path: O -> O -> B-LOC -> I-LOC -> O -> O
    rng = np.random.default_rng(7)
    delta = rng.uniform(-3.5, -1.0, size=(T, L))
    best_path = [2, 2, 0, 1, 2, 2]
    for t, j in enumerate(best_path):
        delta[t, j] = -0.4 - 0.15 * t
    # back pointers (precomputed to match best path)
    psi = np.full((T, L), -1, dtype=int)
    psi[1, :] = 2
    psi[2, :] = 2
    psi[3, :] = 0
    psi[4, :] = 1
    psi[5, :] = 2

    fig, ax = plt.subplots(figsize=(11.8, 5.6))

    # Draw faint all transitions
    for t in range(T - 1):
        for i in range(L):
            for j in range(L):
                _arrow(ax, (xs[t] + 0.36, ys[i]),
                       (xs[t + 1] - 0.36, ys[j]),
                       color=C_GRAY, lw=0.5, style="-|>", mutation=7,
                       alpha=0.25)

    # Highlight best path arrows
    for t in range(T - 1):
        i = best_path[t]
        j = best_path[t + 1]
        _arrow(ax, (xs[t] + 0.36, ys[i]),
               (xs[t + 1] - 0.36, ys[j]),
               color=C_AMBER, lw=2.6, style="-|>", mutation=16, alpha=1.0)

    # Nodes coloured by delta with best path highlighted
    dmin, dmax = delta.min(), delta.max()
    for t in range(T):
        for k in range(L):
            score = delta[t, k]
            on_path = (k == best_path[t])
            color = C_AMBER if on_path else C_LIGHT
            txtcol = "white" if on_path else C_DARK
            _node(ax, xs[t], ys[k], labels[k], color=color,
                  text_color=txtcol, fontsize=9.5, r=0.4)
            ax.text(xs[t], ys[k] - 0.62, f"{score:+.2f}",
                    ha="center", fontsize=8.5,
                    color=C_AMBER if on_path else C_GRAY,
                    fontweight="bold" if on_path else "normal")

    # Time axis with example tokens
    tokens = ["He", "left", "New", "York", "yesterday", "."]
    for t in range(T):
        ax.text(xs[t], -0.55, f"$t={t+1}$", ha="center",
                fontsize=10, color=C_DARK)
        ax.text(xs[t], -1.05, tokens[t], ha="center",
                fontsize=9.5, color=C_GRAY, style="italic")

    ax.text(xs[0] - 0.6, ys[0] + 0.85,
            r"$\delta_t(j) = \max_i\,[\,\delta_{t-1}(i) + \mathbf{w}^\top"
            r"\mathbf{f}(i,j,\mathbf{X},t)\,]$",
            fontsize=11.5, color=C_DARK, fontweight="bold")
    ax.text(xs[-1] + 0.4, ys[0] + 0.4,
            "back-pointers\n trace best path",
            fontsize=9.5, color=C_AMBER, ha="left",
            fontweight="bold")
    _arrow(ax, (xs[-1] + 0.4, ys[0] + 0.1),
           (xs[-1] + 0.05, ys[2] - 0.05),
           color=C_AMBER, lw=1.4, style="-|>", mutation=12)

    ax.set_xlim(0.2, xs[-1] + 1.6)
    ax.set_ylim(-1.4, ys[0] + 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Viterbi decoding: same trellis as forward, sum replaced by max"
        r" -- $O(TL^2)$",
        fontsize=12.5, fontweight="bold", pad=10,
    )
    fig.tight_layout()
    _save(fig, "fig4_viterbi_decoding")


# ---------------------------------------------------------------------------
# Figure 5: HMM vs MEMM vs CRF graphical models
# ---------------------------------------------------------------------------
def fig5_hmm_memm_crf() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    T = 4
    xs = np.linspace(0.8, 4.4, T)

    def base(ax, title, subtitle, color):
        ax.set_xlim(0.0, 5.2)
        ax.set_ylim(-0.4, 3.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=12.5, fontweight="bold",
                     color=color, pad=4)
        ax.text(2.6, 3.05, subtitle, ha="center", fontsize=9.5,
                color=C_GRAY, style="italic")

    # --- HMM ---
    ax = axes[0]
    base(ax, "HMM",
         r"generative, local: $P(\mathbf{X},\mathbf{Y})$", C_PURPLE)
    y_lab, y_obs = 2.1, 0.6
    for t, x in enumerate(xs, start=1):
        _node(ax, x, y_lab, f"$y_{t}$", color=C_PURPLE, r=0.28,
              fontsize=10)
        _node(ax, x, y_obs, f"$x_{t}$", color=C_AMBER, r=0.28,
              fontsize=10)
        _arrow(ax, (x, y_lab - 0.28), (x, y_obs + 0.28),
               color=C_DARK, lw=1.2)
    for i in range(T - 1):
        _arrow(ax, (xs[i] + 0.28, y_lab), (xs[i + 1] - 0.28, y_lab),
               color=C_DARK, lw=1.2)
    ax.text(2.6, -0.25, "observation independence",
            ha="center", fontsize=9.5, color=C_DARK)

    # --- MEMM ---
    ax = axes[1]
    base(ax, "MEMM",
         r"discriminative, local: $\prod_t P(y_t\mid y_{t-1},\mathbf{X})$",
         C_AMBER)
    for t, x in enumerate(xs, start=1):
        _node(ax, x, y_lab, f"$y_{t}$", color=C_AMBER, r=0.28,
              fontsize=10)
        _node(ax, x, y_obs, f"$x_{t}$", color=C_BLUE, r=0.28,
              fontsize=10)
        # X conditions y (arrow up) -- discriminative direction
        _arrow(ax, (x, y_obs + 0.28), (x, y_lab - 0.28),
               color=C_DARK, lw=1.2)
    for i in range(T - 1):
        _arrow(ax, (xs[i] + 0.28, y_lab), (xs[i + 1] - 0.28, y_lab),
               color=C_DARK, lw=1.2)
    ax.text(2.6, -0.25, "label-bias problem",
            ha="center", fontsize=9.5, color="#b91c1c", fontweight="bold")

    # --- CRF ---
    ax = axes[2]
    base(ax, "CRF",
         r"discriminative, global: $P(\mathbf{Y}\mid\mathbf{X}) /"
         r" Z(\mathbf{X})$", C_BLUE)
    # observation strip
    strip_w = (xs[-1] - xs[0]) + 0.9
    strip_x = (xs[0] + xs[-1]) / 2 - strip_w / 2
    rect = FancyBboxPatch(
        (strip_x, y_obs - 0.28), strip_w, 0.56,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=C_AMBER, edgecolor=C_DARK, linewidth=1.2,
        alpha=0.85, zorder=3,
    )
    ax.add_patch(rect)
    ax.text((xs[0] + xs[-1]) / 2, y_obs, r"$\mathbf{X}$",
            ha="center", va="center", color="white", fontsize=11,
            fontweight="bold", zorder=4)
    for t, x in enumerate(xs, start=1):
        _node(ax, x, y_lab, f"$y_{t}$", color=C_BLUE, r=0.28,
              fontsize=10)
        ax.plot([x, x], [y_lab - 0.28, y_obs + 0.28],
                color=C_GRAY, lw=0.9, ls=":", zorder=1)
    for i in range(T - 1):
        _undirected(ax, (xs[i] + 0.28, y_lab),
                    (xs[i + 1] - 0.28, y_lab),
                    color=C_DARK, lw=1.7)
    ax.text(2.6, -0.25, "globally normalised",
            ha="center", fontsize=9.5, color=C_GREEN, fontweight="bold")

    fig.suptitle("HMM vs MEMM vs CRF",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_hmm_memm_crf")


# ---------------------------------------------------------------------------
# Figure 6: Generative vs discriminative
# ---------------------------------------------------------------------------
def fig6_generative_vs_disc() -> None:
    rng = np.random.default_rng(2)
    n = 90
    # Two classes
    mu1, mu2 = np.array([-1.4, 0.0]), np.array([1.4, 0.4])
    cov = np.array([[0.8, 0.15], [0.15, 0.5]])
    L = np.linalg.cholesky(cov)
    X1 = rng.standard_normal((n, 2)) @ L.T + mu1
    X2 = rng.standard_normal((n, 2)) @ L.T + mu2

    xs = np.linspace(-4.5, 4.5, 220)
    ys = np.linspace(-3.0, 3.0, 180)
    XX, YY = np.meshgrid(xs, ys)
    XYflat = np.stack([XX.ravel(), YY.ravel()], axis=1)

    def gauss(x, mu, sigma):
        d = x - mu
        inv = np.linalg.inv(sigma)
        det = np.linalg.det(sigma)
        ex = -0.5 * np.einsum("ij,jk,ik->i", d, inv, d)
        return np.exp(ex) / (2 * np.pi * np.sqrt(det))

    p1 = gauss(XYflat, mu1, cov).reshape(XX.shape)
    p2 = gauss(XYflat, mu2, cov).reshape(XX.shape)
    joint = 0.5 * p1 + 0.5 * p2  # marginal P(X)
    posterior = (0.5 * p2) / (joint + 1e-12)  # P(class=2 | x)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # --- Left: generative -- model joint density ---
    ax = axes[0]
    ax.contourf(XX, YY, joint, levels=18, cmap="Purples", alpha=0.85)
    ax.contour(XX, YY, p1, levels=5, colors=[C_BLUE], linewidths=1.0,
               alpha=0.8)
    ax.contour(XX, YY, p2, levels=5, colors=[C_AMBER], linewidths=1.0,
               alpha=0.8)
    ax.scatter(X1[:, 0], X1[:, 1], s=22, color=C_BLUE,
               edgecolor="white", linewidth=0.8, label="class 1",
               zorder=3)
    ax.scatter(X2[:, 0], X2[:, 1], s=22, color=C_AMBER,
               edgecolor="white", linewidth=0.8, label="class 2",
               zorder=3)
    ax.set_title(r"Generative: model $P(\mathbf{X},\mathbf{Y})$"
                 " (full joint density)",
                 fontsize=12, fontweight="bold", pad=8, color=C_PURPLE)
    ax.text(0.02, 0.98,
            "spends capacity on modelling$\\mathbf{X}$\n"
            "(HMM, Naive Bayes, GMM)",
            transform=ax.transAxes, va="top", fontsize=9.5,
            color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_GRAY, alpha=0.9))
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel(r"$x_1$", fontsize=10.5)
    ax.set_ylabel(r"$x_2$", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)

    # --- Right: discriminative -- model conditional decision boundary ---
    ax = axes[1]
    ax.contourf(XX, YY, posterior, levels=20, cmap="RdYlBu_r", alpha=0.7)
    ax.contour(XX, YY, posterior, levels=[0.5], colors=[C_DARK],
               linewidths=2.2)
    ax.scatter(X1[:, 0], X1[:, 1], s=22, color=C_BLUE,
               edgecolor="white", linewidth=0.8, label="class 1",
               zorder=3)
    ax.scatter(X2[:, 0], X2[:, 1], s=22, color=C_AMBER,
               edgecolor="white", linewidth=0.8, label="class 2",
               zorder=3)
    ax.set_title(r"Discriminative: model $P(\mathbf{Y}\mid\mathbf{X})$"
                 " (just the boundary)",
                 fontsize=12, fontweight="bold", pad=8, color=C_BLUE)
    ax.text(0.02, 0.98,
            "all capacity goes to the boundary\n"
            "(CRF, Logistic Regression, MEMM)",
            transform=ax.transAxes, va="top", fontsize=9.5,
            color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_GRAY, alpha=0.9))
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-3.0, 3.0)
    ax.set_xlabel(r"$x_1$", fontsize=10.5)
    ax.set_ylabel(r"$x_2$", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=9.5, frameon=True)

    fig.suptitle(
        "Generative vs discriminative: where the modelling budget goes",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig6_generative_vs_disc")


# ---------------------------------------------------------------------------
# Figure 7: NER tagging with confidence
# ---------------------------------------------------------------------------
def fig7_ner_tagging() -> None:
    tokens = ["Apple", "CEO", "Tim", "Cook", "visited", "Beijing",
              "last", "week"]
    tags = ["B-ORG", "O", "B-PER", "I-PER", "O", "B-LOC", "O", "O"]
    # Per-token marginal confidences from the (hypothetical) trained CRF
    confidences = np.array([0.93, 0.97, 0.95, 0.98, 0.99, 0.96, 0.99, 0.99])

    tag_colors = {
        "B-ORG": C_PURPLE,
        "B-PER": C_BLUE, "I-PER": C_BLUE,
        "B-LOC": C_GREEN,
        "O": C_GRAY,
    }
    T = len(tokens)
    xs = np.arange(T) * 1.45 + 1.0

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 5.6),
                             gridspec_kw={"height_ratios": [1.4, 1.0]})

    # --- Top: tokens, tags, span boxes ---
    ax = axes[0]
    y_tok = 2.3
    y_tag = 1.05
    for x, w in zip(xs, tokens):
        _box(ax, x, y_tok, 1.3, 0.7, w, color=C_BG, edge=C_DARK,
             fontsize=11.5, fontweight="bold")
    for x, t in zip(xs, tags):
        _box(ax, x, y_tag, 1.3, 0.7, t, color=tag_colors[t],
             edge=C_DARK, text_color="white", fontsize=10.5,
             fontweight="bold")
        ax.plot([x, x], [y_tok - 0.35, y_tag + 0.35],
                color=C_GRAY, lw=0.9, ls=":")

    # Span brackets
    spans = [(0, 0, "ORG", C_PURPLE),
             (2, 3, "PER", C_BLUE),
             (5, 5, "LOC", C_GREEN)]
    for s, e, name, color in spans:
        x1 = xs[s] - 0.66
        x2 = xs[e] + 0.66
        rect = Rectangle(
            (x1, y_tok - 0.55), x2 - x1, 1.6,
            facecolor="none", edgecolor=color, linewidth=2.0,
            linestyle="--", zorder=1,
        )
        ax.add_patch(rect)
        ax.text((x1 + x2) / 2, y_tok + 1.2, name,
                ha="center", fontsize=10.5, color=color,
                fontweight="bold")

    ax.set_xlim(0.0, xs[-1] + 1.0)
    ax.set_ylim(0.4, 3.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "CRF NER output: BIO tags decoded by Viterbi, spans from contiguous"
        " B/I groups",
        fontsize=12.5, fontweight="bold", pad=8,
    )

    # --- Bottom: per-token confidence bars (marginals) ---
    ax = axes[1]
    bar_colors = [tag_colors[t] for t in tags]
    bars = ax.bar(xs, confidences, width=1.05, color=bar_colors,
                  edgecolor=C_DARK, linewidth=0.9, alpha=0.9)
    for bar, c in zip(bars, confidences):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{c:.2f}", ha="center", fontsize=9.5,
                color=C_DARK, fontweight="bold")
    ax.axhline(0.5, color=C_GRAY, lw=0.9, ls="--", alpha=0.8)
    ax.text(xs[-1] + 0.4, 0.5, "0.5", color=C_GRAY, fontsize=8.5,
            va="center")
    ax.set_xlim(0.0, xs[-1] + 1.0)
    ax.set_ylim(0.0, 1.18)
    ax.set_xticks(xs)
    ax.set_xticklabels(tokens, fontsize=10)
    ax.set_ylabel(r"$P(y_t\mid\mathbf{X})$", fontsize=10.5)
    ax.set_title(
        "Per-token marginal confidence from forward-backward",
        fontsize=11, fontweight="bold", pad=6,
    )

    fig.tight_layout()
    _save(fig, "fig7_ner_tagging")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"EN -> {EN_DIR}")
    print(f"ZH -> {ZH_DIR}")
    print()
    for fn in (
        fig1_chain_structure,
        fig2_feature_templates,
        fig3_forward_backward,
        fig4_viterbi_decoding,
        fig5_hmm_memm_crf,
        fig6_generative_vs_disc,
        fig7_ner_tagging,
    ):
        print(f"[render] {fn.__name__}")
        fn()
    print("\nDone.")


if __name__ == "__main__":
    main()
