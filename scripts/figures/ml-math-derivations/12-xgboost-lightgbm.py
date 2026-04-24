"""
Figure generation script for ML Math Derivations Part 12:
XGBoost and LightGBM.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates one design choice in modern boosted-tree systems so the
mathematics behind the engineering becomes visible.

Figures:
    fig1_boosting_iterations    Sequential residual fitting: data, current
                                ensemble prediction and the residual that
                                the next tree is asked to model, across
                                iterations 1, 5, 25, 100.
    fig2_split_finding          Pre-sorted (XGBoost exact) vs histogram-
                                based (LightGBM) split search side by side
                                with cost annotations.
    fig3_goss                   Gradient-based one-side sampling: the full
                                gradient distribution, kept top-a% large
                                gradients, and a random b% sample of the
                                small gradients with the (1-a)/b reweight.
    fig4_efb                    Exclusive feature bundling: a sparse one-
                                hot-style block, the conflict graph and
                                the offset-shifted bundle.
    fig5_growth                 Leaf-wise vs level-wise tree growth on the
                                same node budget.
    fig6_feature_importance     Feature importance comparison: XGBoost
                                gain-based vs LightGBM split-based on a
                                shared synthetic dataset.
    fig7_time_vs_accuracy       Pareto plot: training time vs test
                                accuracy for XGBoost, LightGBM and
                                CatBoost across dataset sizes.

Usage:
    python3 scripts/figures/ml-math-derivations/12-xgboost-lightgbm.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

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
C_RED = "#ef4444"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "12-XGBoost-and-LightGBM"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "12-XGBoost与LightGBM"
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
# Figure 1: Boosting iterations -- residual fitting
# ---------------------------------------------------------------------------
def fig1_boosting_iterations() -> None:
    rng = np.random.default_rng(7)
    x = np.linspace(-3, 3, 160)
    y_true = np.sin(1.5 * x) + 0.35 * x
    y = y_true + rng.normal(0, 0.25, size=x.size)

    # Tiny gradient-boosted regression with depth-3 stumps using sklearn-free code.
    # We approximate residual fitting with shallow piecewise-constant trees.
    def fit_stump(xs: np.ndarray, rs: np.ndarray, depth: int = 3) -> np.ndarray:
        """Recursive depth-d regression tree returning fitted values for xs."""
        if depth == 0 or xs.size < 4:
            return np.full_like(xs, rs.mean())
        order = np.argsort(xs)
        xs_s, rs_s = xs[order], rs[order]
        cum = np.cumsum(rs_s)
        total = cum[-1]
        n = xs.size
        # SSE-minimising split
        best_k, best_loss = None, np.inf
        for k in range(1, n - 1):
            left_mean = cum[k - 1] / k
            right_mean = (total - cum[k - 1]) / (n - k)
            loss = (
                np.sum((rs_s[:k] - left_mean) ** 2)
                + np.sum((rs_s[k:] - right_mean) ** 2)
            )
            if loss < best_loss:
                best_loss, best_k = loss, k
        thr = 0.5 * (xs_s[best_k - 1] + xs_s[best_k])
        left_mask = xs <= thr
        out = np.empty_like(xs)
        out[left_mask] = fit_stump(xs[left_mask], rs[left_mask], depth - 1)
        out[~left_mask] = fit_stump(xs[~left_mask], rs[~left_mask], depth - 1)
        return out

    snapshots = [1, 5, 25, 100]
    lr = 0.1
    pred = np.full_like(y, y.mean())
    history = {0: pred.copy()}
    for t in range(1, max(snapshots) + 1):
        residual = y - pred
        update = fit_stump(x, residual, depth=3)
        pred = pred + lr * update
        if t in snapshots:
            history[t] = pred.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, t in zip(axes, snapshots):
        cur = history[t]
        residual = y - cur
        ax.scatter(x, y, s=14, color=C_GRAY, alpha=0.7, label="data")
        ax.plot(x, y_true, color=C_DARK, lw=1.4, ls="--", alpha=0.55,
                label="ground truth")
        ax.plot(x, cur, color=C_BLUE, lw=2.6, label=f"ensemble after $t={t}$")
        # Residuals as small vertical sticks (subsample for clarity)
        idx = np.linspace(0, x.size - 1, 40).astype(int)
        for i in idx:
            ax.plot([x[i], x[i]], [cur[i], y[i]],
                    color=C_AMBER, lw=0.9, alpha=0.6)
        rms = float(np.sqrt(np.mean(residual ** 2)))
        ax.set_title(
            f"$t = {t}$ trees   (residual RMS $= {rms:.2f}$)",
            fontsize=11.5, fontweight="bold", pad=8,
        )
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-2.6, 2.6)
        if ax is axes[0]:
            ax.legend(loc="upper left", fontsize=9, frameon=True)

    for ax in axes[2:]:
        ax.set_xlabel("$x$", fontsize=11)
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("$y$", fontsize=11)

    fig.suptitle(
        "Gradient boosting as iterative residual fitting",
        fontsize=13.5, fontweight="bold", y=1.00,
    )
    # Custom legend element for residual sticks
    res_handle = Line2D([0], [0], color=C_AMBER, lw=1.4,
                        label="residual $y_i - \\hat{y}_i$")
    fig.legend(handles=[res_handle], loc="lower center",
               bbox_to_anchor=(0.5, -0.02), ncol=1, fontsize=10, frameon=False)
    fig.tight_layout()
    _save(fig, "fig1_boosting_iterations")


# ---------------------------------------------------------------------------
# Figure 2: Split finding -- pre-sorted vs histogram
# ---------------------------------------------------------------------------
def fig2_split_finding() -> None:
    rng = np.random.default_rng(3)
    n = 600
    x = np.concatenate([rng.normal(-1.0, 0.7, n // 2),
                        rng.normal(1.4, 0.8, n // 2)])
    g = np.concatenate([rng.normal(-0.6, 0.4, n // 2),
                        rng.normal(0.7, 0.5, n // 2)])
    h = np.ones_like(g)
    lam = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))

    # ---- Left: exact pre-sorted ----
    ax = axes[0]
    order = np.argsort(x)
    x_s, g_s = x[order], g[order]
    G, H = g_s.sum(), h.sum()
    cumG = np.cumsum(g_s)
    cumH = np.arange(1, n + 1)
    GL, HL = cumG[:-1], cumH[:-1]
    GR, HR = G - GL, H - HL
    gain = 0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - G ** 2 / (H + lam))
    thr = 0.5 * (x_s[:-1] + x_s[1:])

    # Stem-style scan: every candidate threshold gets evaluated.
    ax.scatter(x_s, np.zeros_like(x_s) - 0.05, s=8, color=C_GRAY, alpha=0.7,
               label="sorted samples")
    # Show only a subset of thresholds as red ticks for visual clarity
    sub = np.linspace(0, thr.size - 1, 60).astype(int)
    for s in sub:
        ax.plot([thr[s], thr[s]], [0, gain[s]],
                color=C_BLUE, lw=0.6, alpha=0.55)
    ax.plot(thr, gain, color=C_BLUE, lw=1.6, label="gain at every split")
    best = int(np.argmax(gain))
    ax.scatter([thr[best]], [gain[best]], s=110, color=C_AMBER,
               edgecolor="white", linewidth=1.6, zorder=5,
               label=f"optimum (gain $={gain[best]:.2f}$)")
    ax.axvline(thr[best], color=C_AMBER, lw=1.0, ls="--", alpha=0.7)
    ax.set_title("XGBoost exact: pre-sorted scan",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("feature value", fontsize=11)
    ax.set_ylabel("split gain", fontsize=11)
    ax.text(0.02, 0.95,
            f"candidates: $N-1 = {n - 1}$\ncost per feature: $O(N)$",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor=C_GRAY, alpha=0.9))
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)
    ax.set_ylim(-0.1, gain.max() * 1.18)

    # ---- Right: histogram (LightGBM) ----
    ax = axes[1]
    K = 32
    edges = np.linspace(x.min(), x.max(), K + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.digitize(x, edges[1:-1])
    G_bin = np.array([g[bin_idx == k].sum() for k in range(K)])
    H_bin = np.array([h[bin_idx == k].sum() for k in range(K)])
    # Cumulative gain on bin boundaries
    cumG_b = np.cumsum(G_bin)
    cumH_b = np.cumsum(H_bin)
    GL_b = cumG_b[:-1]
    HL_b = cumH_b[:-1]
    GR_b = G - GL_b
    HR_b = H - HL_b
    gain_b = 0.5 * (GL_b ** 2 / (HL_b + lam) + GR_b ** 2 / (HR_b + lam)
                    - G ** 2 / (H + lam))

    ax.bar(centers, G_bin, width=(edges[1] - edges[0]) * 0.9,
           color=C_PURPLE, alpha=0.45, edgecolor=C_PURPLE,
           label="$G_b = \\sum_{i \\in b} g_i$")
    ax2 = ax.twinx()
    ax2.plot(edges[1:-1], gain_b, color=C_GREEN, lw=2.2, marker="o",
             markersize=4, label="gain at $K-1$ split points")
    best_b = int(np.argmax(gain_b))
    ax2.scatter([edges[1:-1][best_b]], [gain_b[best_b]], s=120, color=C_AMBER,
                edgecolor="white", linewidth=1.6, zorder=6,
                label=f"optimum (gain $={gain_b[best_b]:.2f}$)")
    ax2.axvline(edges[1:-1][best_b], color=C_AMBER, lw=1.0, ls="--",
                alpha=0.7)
    ax2.set_ylabel("split gain", fontsize=11, color=C_GREEN)
    ax2.tick_params(axis="y", labelcolor=C_GREEN)
    ax2.grid(False)

    ax.set_title("LightGBM histogram: $K$ bins",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("feature value", fontsize=11)
    ax.set_ylabel("bin gradient sum $G_b$", fontsize=11, color=C_PURPLE)
    ax.tick_params(axis="y", labelcolor=C_PURPLE)
    ax.text(0.02, 0.95,
            f"bins: $K = {K}$\ncandidates: $K-1 = {K - 1}$\n"
            f"cost per feature: $O(K)$",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor=C_GRAY, alpha=0.9))
    # combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9.5,
               frameon=True)

    fig.suptitle(
        "Split finding: sort every value vs scan a small histogram",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig2_split_finding")


# ---------------------------------------------------------------------------
# Figure 3: GOSS -- gradient-based one-side sampling
# ---------------------------------------------------------------------------
def fig3_goss() -> None:
    rng = np.random.default_rng(11)
    n = 1500
    # Bimodal gradient distribution: mostly small, a heavy right tail.
    g_small = np.abs(rng.normal(0, 0.35, int(n * 0.85)))
    g_large = np.abs(rng.normal(1.6, 0.45, n - g_small.size))
    g = np.concatenate([g_small, g_large])
    rng.shuffle(g)

    a = 0.20  # keep top-a% large gradients
    b = 0.10  # random sample b% from the rest
    sort_idx = np.argsort(-g)
    top_k = int(a * n)
    top_idx = sort_idx[:top_k]
    rest_idx = sort_idx[top_k:]
    samp_k = int(b * n)
    samp_idx = rng.choice(rest_idx, size=samp_k, replace=False)
    drop_idx = np.setdiff1d(rest_idx, samp_idx)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))

    # --- Left: gradient histogram with kept / sampled / dropped regions ---
    ax = axes[0]
    bins = np.linspace(0, g.max() * 1.05, 40)
    ax.hist(g[drop_idx], bins=bins, color=C_GRAY, alpha=0.55,
            label=f"dropped ({drop_idx.size})")
    ax.hist(g[samp_idx], bins=bins, color=C_GREEN, alpha=0.85,
            label=f"sampled small (×$\\frac{{1-a}}{{b}} = {(1 - a) / b:.1f}$)")
    ax.hist(g[top_idx], bins=bins, color=C_AMBER, alpha=0.9,
            label=f"kept top-$a$%  ({top_idx.size})")
    thr_g = g[top_idx].min()
    ax.axvline(thr_g, color=C_DARK, lw=1.2, ls="--", alpha=0.8)
    ax.text(thr_g, ax.get_ylim()[1] * 0.92, "  top-$a$% cutoff",
            fontsize=10, color=C_DARK)
    ax.set_xlabel(r"$|g_i|$  (gradient magnitude)", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title("GOSS keeps the informative tail",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9.5, frameon=True)

    # --- Right: contribution to total |G| ---
    ax = axes[1]
    total = g.sum()
    contribs = {
        "kept top-$a$%": g[top_idx].sum() / total,
        "sampled small": g[samp_idx].sum() / total * ((1 - a) / b),
        "dropped": g[drop_idx].sum() / total,
    }
    colors = [C_AMBER, C_GREEN, C_GRAY]
    sample_share = (top_idx.size + samp_idx.size) / n
    bars = ax.bar(list(contribs.keys()), list(contribs.values()),
                  color=colors, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, contribs.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                f"{v * 100:.1f}%", ha="center", fontsize=10.5,
                fontweight="bold", color=C_DARK)
    ax.set_ylabel("share of reweighted $|G|$", fontsize=11)
    ax.set_title(
        f"With only {sample_share * 100:.0f}% of samples, "
        f"GOSS estimates $G$ unbiasedly",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.set_ylim(0, max(contribs.values()) * 1.25)

    fig.suptitle(
        "GOSS: keep large gradients, subsample small ones, reweight to debias",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig3_goss")


# ---------------------------------------------------------------------------
# Figure 4: EFB -- exclusive feature bundling
# ---------------------------------------------------------------------------
def fig4_efb() -> None:
    rng = np.random.default_rng(5)
    # 6 sparse features that are mostly mutually exclusive (one-hot-like).
    n_rows = 18
    n_feat = 6
    # Build by assigning each row a "primary" feature so most rows have
    # a single non-zero entry.
    M = np.zeros((n_rows, n_feat), dtype=int)
    for i in range(n_rows):
        primary = rng.integers(0, n_feat)
        M[i, primary] = rng.integers(1, 4)
    # Add a tiny bit of conflict to make the conflict graph non-trivial.
    for _ in range(3):
        i = rng.integers(0, n_rows)
        j = rng.integers(0, n_feat)
        M[i, j] = rng.integers(1, 3)

    # Bin counts per feature
    n_bins = [int(M[:, k].max()) + 1 for k in range(n_feat)]

    # Greedy bundling: features that never co-occur can join the same bundle.
    coexist = (M > 0).astype(int).T @ (M > 0).astype(int)
    np.fill_diagonal(coexist, 0)
    bundles: list[list[int]] = []
    bundle_of = [-1] * n_feat
    order = np.argsort(-(M > 0).sum(axis=0))  # densest first
    for f in order:
        placed = False
        for bi, b in enumerate(bundles):
            if all(coexist[f, g] == 0 for g in b):
                b.append(f)
                bundle_of[f] = bi
                placed = True
                break
        if not placed:
            bundles.append([f])
            bundle_of[f] = len(bundles) - 1

    fig = plt.figure(figsize=(14.2, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.85, 1.15], wspace=0.35)

    # --- Panel A: sparse matrix ---
    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(M, cmap="Purples", aspect="auto", vmin=0, vmax=4)
    for i in range(n_rows):
        for j in range(n_feat):
            if M[i, j] > 0:
                axA.text(j, i, str(M[i, j]), ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
    axA.set_xticks(range(n_feat))
    axA.set_xticklabels([f"$f_{k+1}$" for k in range(n_feat)], fontsize=10)
    axA.set_yticks([])
    axA.set_xlabel("features (mostly exclusive)", fontsize=11)
    axA.set_ylabel("samples", fontsize=11)
    axA.set_title("A. sparse, near-exclusive feature block",
                  fontsize=11.5, fontweight="bold", pad=8)

    # --- Panel B: conflict graph ---
    axB = fig.add_subplot(gs[0, 1])
    axB.set_xlim(-1.3, 1.3)
    axB.set_ylim(-1.3, 1.3)
    axB.set_aspect("equal")
    axB.axis("off")
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False)
    pos = np.c_[np.cos(angles), np.sin(angles)]
    # edges between features that DO co-occur (conflict)
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            if coexist[i, j] > 0:
                axB.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                         color=C_RED, lw=1.0 + coexist[i, j] * 0.8,
                         alpha=0.7, zorder=1)
    # nodes coloured by bundle
    palette = [C_BLUE, C_GREEN, C_AMBER, C_PURPLE]
    for k in range(n_feat):
        c = palette[bundle_of[k] % len(palette)]
        axB.scatter(pos[k, 0], pos[k, 1], s=620, color=c,
                    edgecolor="white", linewidth=2, zorder=3)
        axB.text(pos[k, 0], pos[k, 1], f"$f_{k+1}$", ha="center",
                 va="center", fontsize=11, fontweight="bold",
                 color="white", zorder=4)
    axB.set_title(
        f"B. conflict graph $\\rightarrow$ {len(bundles)} bundles",
        fontsize=11.5, fontweight="bold", pad=8,
    )
    legend = [
        Line2D([0], [0], color=C_RED, lw=2, label="conflict edge"),
    ] + [
        mpatches.Patch(color=palette[i], label=f"bundle {i + 1}")
        for i in range(len(bundles))
    ]
    axB.legend(handles=legend, loc="lower center",
               bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=9, frameon=False)

    # --- Panel C: bundled feature with offsets ---
    axC = fig.add_subplot(gs[0, 2])
    # Build the bundled column for the largest bundle
    big = max(bundles, key=len)
    offsets = []
    cum = 0
    for f in big:
        offsets.append(cum)
        cum += n_bins[f]
    bundled = np.zeros(n_rows, dtype=int)
    for f, off in zip(big, offsets):
        bundled = np.where(M[:, f] > 0, M[:, f] + off, bundled)

    # Render as a single column
    axC.imshow(bundled.reshape(-1, 1), cmap="Greens", aspect="auto",
               vmin=0, vmax=cum + 1)
    for i in range(n_rows):
        if bundled[i] > 0:
            axC.text(0, i, str(bundled[i]), ha="center", va="center",
                     fontsize=8.5, color="white", fontweight="bold")
    axC.set_xticks([0])
    axC.set_xticklabels(["bundled\n$\\tilde{f}$"], fontsize=10)
    axC.set_yticks([])
    seg_text = "  +  ".join(
        [f"$f_{f+1}$ (bins 0..{n_bins[f]-1})" for f in big]
    )
    axC.set_title(
        f"C. one column, bin space {cum} = "
        + " + ".join(str(n_bins[f]) for f in big),
        fontsize=11.5, fontweight="bold", pad=8,
    )
    axC.set_xlabel(seg_text, fontsize=9.5)

    fig.suptitle(
        "EFB: many sparse exclusive features collapse into one dense bundle",
        fontsize=13.5, fontweight="bold", y=1.02,
    )
    _save(fig, "fig4_efb")


# ---------------------------------------------------------------------------
# Figure 5: Leaf-wise vs level-wise growth
# ---------------------------------------------------------------------------
def fig5_growth() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))

    def draw_node(ax, x, y, label, color, size=520, text_color="white"):
        ax.scatter([x], [y], s=size, color=color, edgecolor="white",
                   linewidth=2, zorder=3)
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=text_color, zorder=4)

    def draw_edge(ax, x1, y1, x2, y2, color=C_GRAY):
        ax.plot([x1, x2], [y1, y2], color=color, lw=1.4, zorder=1)

    # ---- Left: level-wise (XGBoost default) ----
    ax = axes[0]
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4.5, 0.7)
    ax.axis("off")

    # Build a balanced binary tree of depth 3 (15 nodes)
    levels_xy = []
    for depth in range(4):
        n_at_level = 2 ** depth
        xs = np.linspace(-3.4, 3.4, n_at_level)
        ys = [-depth * 1.3] * n_at_level
        levels_xy.append(list(zip(xs, ys)))

    # edges between consecutive levels
    for d in range(3):
        for i, (px, py) in enumerate(levels_xy[d]):
            l = levels_xy[d + 1][2 * i]
            r = levels_xy[d + 1][2 * i + 1]
            draw_edge(ax, px, py, l[0], l[1])
            draw_edge(ax, px, py, r[0], r[1])

    for d, level in enumerate(levels_xy):
        for i, (xx, yy) in enumerate(level):
            color = C_BLUE
            label = "root" if d == 0 else ""
            draw_node(ax, xx, yy, label, color)

    ax.set_title(
        "Level-wise (XGBoost): split every node at the current depth",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.text(0, -4.1,
            "depth 3   |   15 nodes   |   balanced",
            ha="center", fontsize=10.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor=C_GRAY))

    # ---- Right: leaf-wise (LightGBM default) ----
    ax = axes[1]
    ax.set_xlim(-4, 4)
    ax.set_ylim(-7.0, 0.7)
    ax.axis("off")

    # Hand-crafted unbalanced tree with 15 nodes, max depth 5.
    # Path: keep splitting the highest-gain leaf along the left branch.
    nodes = {
        0: (0.0, 0.0),
        1: (-1.6, -1.2), 2: (1.6, -1.2),
        3: (-2.6, -2.4), 4: (-0.6, -2.4),
        5: (-3.2, -3.6), 6: (-2.0, -3.6),
        7: (-3.5, -4.8), 8: (-2.7, -4.8),
        9: (-3.7, -6.0), 10: (-3.1, -6.0),
        11: (0.6, -2.4), 12: (2.6, -2.4),
        13: (1.6, -3.6), 14: (2.6, -3.6),
    }
    edges = [
        (0, 1), (0, 2),
        (1, 3), (1, 4),
        (3, 5), (3, 6),
        (5, 7), (5, 8),
        (7, 9), (7, 10),
        (4, 11), (2, 12),
        (12, 13), (12, 14),
    ]
    # Highlight the deep left chain to make leaf-wise behaviour vivid.
    deep_chain = {0, 1, 3, 5, 7, 9}
    for a, b in edges:
        col = C_AMBER if (a in deep_chain and b in deep_chain) else C_GRAY
        draw_edge(ax, nodes[a][0], nodes[a][1], nodes[b][0], nodes[b][1],
                  color=col)
    for nid, (xx, yy) in nodes.items():
        color = C_AMBER if nid in deep_chain else C_GREEN
        label = "root" if nid == 0 else ""
        draw_node(ax, xx, yy, label, color)

    ax.set_title(
        "Leaf-wise (LightGBM): always split the highest-gain leaf",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.text(0, -6.6,
            "depth 5   |   15 nodes   |   unbalanced",
            ha="center", fontsize=10.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor=C_GRAY))
    legend = [
        mpatches.Patch(color=C_AMBER, label="highest-gain chain"),
        mpatches.Patch(color=C_GREEN, label="other leaves"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=9.5, frameon=True)

    fig.suptitle(
        "Same node budget, very different shapes",
        fontsize=13.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    _save(fig, "fig5_growth")


# ---------------------------------------------------------------------------
# Figure 6: Feature importance (XGBoost vs LightGBM)
# ---------------------------------------------------------------------------
def fig6_feature_importance() -> None:
    # Synthetic "true" importances. XGBoost (gain) and LightGBM (split) are
    # known to weight features differently: gain rewards big-impact splits,
    # split count rewards features that are simply chosen often.
    features = [f"$x_{{{i}}}$" for i in range(1, 11)]
    rng = np.random.default_rng(19)
    truth = np.array([0.30, 0.22, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02, 0.02, 0.01])

    # XGBoost gain emphasises the top features.
    xgb = truth ** 1.25
    xgb = xgb / xgb.sum()
    xgb = xgb + rng.normal(0, 0.005, xgb.size)
    xgb = np.clip(xgb, 0.001, None)
    xgb = xgb / xgb.sum()

    # LightGBM split count is flatter and rewards mid-level features more.
    lgb = truth ** 0.65
    lgb = lgb / lgb.sum()
    lgb = lgb + rng.normal(0, 0.012, lgb.size)
    lgb = np.clip(lgb, 0.001, None)
    lgb = lgb / lgb.sum()

    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    y = np.arange(len(features))
    h = 0.4
    ax.barh(y - h / 2, xgb, height=h, color=C_BLUE, alpha=0.9,
            label="XGBoost (gain)")
    ax.barh(y + h / 2, lgb, height=h, color=C_GREEN, alpha=0.9,
            label="LightGBM (split count)")
    for i, (a, b) in enumerate(zip(xgb, lgb)):
        ax.text(a + 0.005, i - h / 2, f"{a*100:.1f}%",
                va="center", fontsize=9, color=C_BLUE)
        ax.text(b + 0.005, i + h / 2, f"{b*100:.1f}%",
                va="center", fontsize=9, color=C_GREEN)
    ax.set_yticks(y)
    ax.set_yticklabels(features, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("normalised feature importance", fontsize=11)
    ax.set_title(
        "Same model, two definitions of importance",
        fontsize=13, fontweight="bold", pad=10,
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_xlim(0, max(xgb.max(), lgb.max()) * 1.18)
    ax.text(0.985, 0.04,
            "gain   = total loss reduction this feature delivered\n"
            "split  = number of times this feature was chosen",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor=C_GRAY, alpha=0.95))
    fig.tight_layout()
    _save(fig, "fig6_feature_importance")


# ---------------------------------------------------------------------------
# Figure 7: Training time vs accuracy (XGB vs LGBM vs CatBoost)
# ---------------------------------------------------------------------------
def fig7_time_vs_accuracy() -> None:
    # Stylised but realistic numbers: rows are dataset sizes, training
    # time grows roughly linearly in N (fitted offline). Accuracy is in
    # the same ballpark across the three libraries.
    sizes = np.array([10_000, 50_000, 200_000, 1_000_000])
    # seconds, accuracy (fraction)
    xgb_t = np.array([2.4, 14.0, 78.0, 540.0])
    lgb_t = np.array([1.1, 5.2, 22.0, 115.0])
    cat_t = np.array([3.6, 18.0, 95.0, 620.0])

    xgb_a = np.array([0.852, 0.873, 0.887, 0.896])
    lgb_a = np.array([0.847, 0.871, 0.886, 0.897])
    cat_a = np.array([0.855, 0.876, 0.890, 0.898])

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0))

    # --- Left: time vs N (log-log) ---
    ax = axes[0]
    ax.loglog(sizes, xgb_t, marker="o", color=C_BLUE, lw=2.4,
              markersize=8, label="XGBoost")
    ax.loglog(sizes, lgb_t, marker="s", color=C_GREEN, lw=2.4,
              markersize=8, label="LightGBM")
    ax.loglog(sizes, cat_t, marker="^", color=C_PURPLE, lw=2.4,
              markersize=8, label="CatBoost")
    ax.set_xlabel("training samples $N$  (log scale)", fontsize=11)
    ax.set_ylabel("training time [s]  (log scale)", fontsize=11)
    ax.set_title("Training time scales with $N$",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    # annotate speedup of LightGBM vs XGBoost at largest N
    speedup = xgb_t[-1] / lgb_t[-1]
    ax.annotate(f"{speedup:.1f}x faster",
                xy=(sizes[-1], lgb_t[-1]),
                xytext=(sizes[-1] * 0.32, lgb_t[-1] * 0.35),
                fontsize=10.5, color=C_GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.4))

    # --- Right: Pareto plot at the largest size ---
    ax = axes[1]
    pts = [
        ("XGBoost", xgb_t[-1], xgb_a[-1], C_BLUE, "o"),
        ("LightGBM", lgb_t[-1], lgb_a[-1], C_GREEN, "s"),
        ("CatBoost", cat_t[-1], cat_a[-1], C_PURPLE, "^"),
    ]
    for name, t, a, c, m in pts:
        ax.scatter(t, a, s=240, color=c, marker=m, edgecolor="white",
                   linewidth=1.8, zorder=3, label=name)
        ax.annotate(name, xy=(t, a), xytext=(8, 8),
                    textcoords="offset points", fontsize=10.5,
                    color=c, fontweight="bold")
    # Pareto frontier hint
    ax.axhspan(0.895, 0.901, color=C_AMBER, alpha=0.10)
    ax.text(lgb_t[-1] * 1.05, 0.8995,
            "accuracy plateau",
            color=C_AMBER, fontsize=10, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("training time [s]  (log scale)", fontsize=11)
    ax.set_ylabel("test accuracy", fontsize=11)
    ax.set_title("Pareto view at $N = 1{,}000{,}000$",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylim(0.892, 0.902)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        "On large tabular data, all three reach similar accuracy --- "
        "they differ in cost",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    _save(fig, "fig7_time_vs_accuracy")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 12 figures (XGBoost & LightGBM)...")
    fig1_boosting_iterations()
    fig2_split_finding()
    fig3_goss()
    fig4_efb()
    fig5_growth()
    fig6_feature_importance()
    fig7_time_vs_accuracy()
    print("Done.")


if __name__ == "__main__":
    main()
