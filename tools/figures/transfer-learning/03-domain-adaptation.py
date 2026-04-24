"""
Figure generation script for Transfer Learning Part 03: Domain Adaptation.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly and is production quality.

Figures:
    fig1_distribution_shift     Source vs target distribution alignment in 2D
                                feature space (before vs after adaptation).
    fig2_dann_architecture      DANN architecture with the Gradient Reversal
                                Layer, three subnetworks, and gradient flow.
    fig3_mmd_kernel             Maximum Mean Discrepancy: kernel mean
                                embeddings of two distributions in RKHS.
    fig4_coral_covariance       CORAL second-order statistics alignment:
                                source vs target covariance matrices and
                                aligned result.
    fig5_tsne_before_after      t-SNE of features before vs after domain
                                adaptation (DANN), coloured by class+domain.
    fig6_self_training          Self-training / pseudo-labelling loop with
                                confidence threshold gating.
    fig7_office31_benchmark     Domain adaptation accuracy on Office-31 /
                                DomainNet across methods.

Usage:
    python3 scripts/figures/transfer-learning/03-domain-adaptation.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Shared aesthetic style (chenk-site)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]     # source domain / primary
C_PURPLE = COLORS["accent"]   # target domain / secondary
C_GREEN = COLORS["success"]    # success / aligned
C_AMBER = COLORS["warning"]    # warning / adversarial
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_BG = COLORS["bg"]
C_RED = COLORS["danger"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "transfer-learning" / "03-domain-adaptation"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "transfer-learning" / "03-域适应方法"


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
# Figure 1: Source vs target distribution alignment (before / after DA)
# ---------------------------------------------------------------------------
def fig1_distribution_shift() -> None:
    """Two 2D feature blobs: before adaptation domains diverge, after they overlap."""
    rng = np.random.default_rng(11)
    n = 220

    # Two classes, each present in source and target. Target is shifted.
    src_c0 = rng.normal(loc=[-1.5, 0.6], scale=[0.45, 0.45], size=(n, 2))
    src_c1 = rng.normal(loc=[1.5, -0.6], scale=[0.45, 0.45], size=(n, 2))
    tgt_c0 = rng.normal(loc=[1.0, 2.6], scale=[0.55, 0.55], size=(n, 2))   # shifted
    tgt_c1 = rng.normal(loc=[3.6, 1.0], scale=[0.55, 0.55], size=(n, 2))   # shifted

    # After adaptation: target has been pulled toward source manifold,
    # while keeping class structure.
    aln_c0 = rng.normal(loc=[-1.4, 0.5], scale=[0.55, 0.55], size=(n, 2))
    aln_c1 = rng.normal(loc=[1.55, -0.55], scale=[0.55, 0.55], size=(n, 2))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))

    # --- Before
    ax = axes[0]
    ax.scatter(src_c0[:, 0], src_c0[:, 1], s=22, c=C_BLUE, alpha=0.55,
               edgecolor="white", linewidth=0.4, label="Source class A")
    ax.scatter(src_c1[:, 0], src_c1[:, 1], s=22, c=C_BLUE, alpha=0.55,
               marker="^", edgecolor="white", linewidth=0.4,
               label="Source class B")
    ax.scatter(tgt_c0[:, 0], tgt_c0[:, 1], s=22, c=C_PURPLE, alpha=0.55,
               edgecolor="white", linewidth=0.4, label="Target class A")
    ax.scatter(tgt_c1[:, 0], tgt_c1[:, 1], s=22, c=C_PURPLE, alpha=0.55,
               marker="^", edgecolor="white", linewidth=0.4,
               label="Target class B")
    # Source decision boundary: roughly y = -x
    xs = np.linspace(-3.2, 4.6, 50)
    ax.plot(xs, -xs, color=C_DARK, linestyle="--", linewidth=1.6,
            label="Source decision boundary")
    ax.set_title("Before adaptation: target is shifted, boundary fails",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_xlim(-3.4, 5.2)
    ax.set_ylim(-2.6, 4.6)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    # --- After
    ax = axes[1]
    ax.scatter(src_c0[:, 0], src_c0[:, 1], s=22, c=C_BLUE, alpha=0.55,
               edgecolor="white", linewidth=0.4, label="Source class A")
    ax.scatter(src_c1[:, 0], src_c1[:, 1], s=22, c=C_BLUE, alpha=0.55,
               marker="^", edgecolor="white", linewidth=0.4,
               label="Source class B")
    ax.scatter(aln_c0[:, 0], aln_c0[:, 1], s=22, c=C_GREEN, alpha=0.55,
               edgecolor="white", linewidth=0.4, label="Target class A (aligned)")
    ax.scatter(aln_c1[:, 0], aln_c1[:, 1], s=22, c=C_GREEN, alpha=0.55,
               marker="^", edgecolor="white", linewidth=0.4,
               label="Target class B (aligned)")
    ax.plot(xs, -xs, color=C_DARK, linestyle="--", linewidth=1.6,
            label="Shared decision boundary")
    ax.set_title("After adaptation: distributions overlap, boundary works",
                 fontsize=12.5, fontweight="bold", pad=10)
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_xlim(-3.4, 5.2)
    ax.set_ylim(-2.6, 4.6)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    fig.suptitle("Domain adaptation aligns source and target in feature space",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_distribution_shift")


# ---------------------------------------------------------------------------
# Figure 2: DANN architecture with Gradient Reversal Layer
# ---------------------------------------------------------------------------
def fig2_dann_architecture() -> None:
    """Box-and-arrow diagram of DANN's three subnetworks plus the GRL."""
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, label, color, text_color="white", fontsize=11):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.6, edgecolor=color, facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold")

    def arrow(p0, p1, color=C_DARK, lw=1.6, style="-|>", curved=0.0):
        a = FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=14,
            color=color, linewidth=lw,
            connectionstyle=f"arc3,rad={curved}",
        )
        ax.add_patch(a)

    # Inputs
    box(0.2, 4.4, 1.7, 0.9, "Source x_s\n(labelled)", C_BLUE, fontsize=10)
    box(0.2, 1.6, 1.7, 0.9, "Target x_t\n(unlabelled)", C_PURPLE, fontsize=10)

    # Feature extractor
    box(2.6, 3.0, 2.1, 1.6, "Feature\nextractor  G_f", C_DARK)

    # Features node
    feat_x, feat_y = 5.4, 3.8
    ax.add_patch(plt.Circle((feat_x, feat_y), 0.32, color=C_GREEN, zorder=3))
    ax.text(feat_x, feat_y, "f", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(feat_x, feat_y - 0.65, "features",
            ha="center", fontsize=9, color=C_DARK, fontstyle="italic")

    # Label predictor branch (top)
    box(7.0, 5.1, 2.4, 1.1, "Label predictor  G_y", C_BLUE, fontsize=11)
    box(10.4, 5.1, 2.3, 1.1, "Class loss\nL_y(y, ŷ)", C_GREEN, fontsize=10.5)

    # GRL + Domain discriminator branch (bottom)
    box(7.0, 1.6, 1.4, 1.1, "GRL\n× (-λ)", C_AMBER, text_color=C_DARK,
        fontsize=10.5)
    box(8.7, 1.6, 2.3, 1.1, "Domain\ndiscriminator G_d", C_PURPLE, fontsize=11)
    box(11.3, 1.6, 1.4, 1.1, "Domain loss\nL_d", C_RED, fontsize=10)

    # Arrows: forward pass
    arrow((1.9, 4.85), (2.6, 4.4))
    arrow((1.9, 2.05), (2.6, 3.4))
    arrow((4.7, 3.8), (feat_x - 0.32, feat_y))
    arrow((feat_x + 0.32, feat_y + 0.05), (7.0, 5.55), curved=0.18)
    arrow((feat_x + 0.32, feat_y - 0.05), (7.0, 2.15), curved=-0.18)
    arrow((9.4, 5.65), (10.4, 5.65))
    arrow((8.4, 2.15), (8.7, 2.15))
    arrow((11.0, 2.15), (11.3, 2.15))

    # Backward gradient annotations
    ax.annotate("normal gradient",
                xy=(8.2, 6.05), fontsize=9, color=C_BLUE,
                ha="center", fontweight="bold")
    ax.annotate("gradient reversed by  -λ",
                xy=(9.0, 1.05), fontsize=9, color=C_AMBER,
                ha="center", fontweight="bold")

    # Legend / objective
    ax.text(6.5, 0.35,
            r"$\min_{G_f, G_y}\;\max_{G_d}\quad \mathcal{L}_y \;-\;\lambda\,\mathcal{L}_d$",
            ha="center", fontsize=14, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2f7",
                      edgecolor=C_GRAY))

    ax.set_title("DANN: one feature extractor, two adversarial heads, "
                 "one backward pass",
                 fontsize=13.5, fontweight="bold", pad=12)

    _save(fig, "fig2_dann_architecture")


# ---------------------------------------------------------------------------
# Figure 3: MMD kernel mean embedding
# ---------------------------------------------------------------------------
def fig3_mmd_kernel() -> None:
    """Two 1D distributions and their kernel mean embeddings; show MMD as gap."""
    rng = np.random.default_rng(3)
    xs = np.linspace(-4, 6, 400)

    # Source: bimodal around -1 and 1
    src_samples = np.concatenate([
        rng.normal(-1.0, 0.5, 60),
        rng.normal(1.2, 0.5, 60),
    ])
    # Target: shifted, mean ~ 2.5
    tgt_samples = rng.normal(2.5, 0.7, 120)

    sigma = 0.6

    def kme(samples):
        # average of Gaussian RBFs centred on each sample
        K = np.exp(-0.5 * ((xs[None, :] - samples[:, None]) / sigma) ** 2)
        return K.mean(axis=0)

    mu_src = kme(src_samples)
    mu_tgt = kme(tgt_samples)
    diff = mu_src - mu_tgt
    mmd_sq = np.trapz(diff ** 2, xs)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.6),
                             gridspec_kw={"height_ratios": [1, 1.35]})

    # Top: raw samples as rugs / histograms
    ax = axes[0]
    ax.hist(src_samples, bins=22, color=C_BLUE, alpha=0.55,
            edgecolor="white", label="Source samples")
    ax.hist(tgt_samples, bins=22, color=C_PURPLE, alpha=0.55,
            edgecolor="white", label="Target samples")
    ax.set_xlim(-4, 6)
    ax.set_ylabel("count")
    ax.set_title("Two domains in input space",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    # Bottom: kernel mean embeddings with shaded gap
    ax = axes[1]
    ax.plot(xs, mu_src, color=C_BLUE, linewidth=2.2,
            label=r"$\mu_{P_S}(x) = \mathbb{E}_S[k(\cdot, x)]$")
    ax.plot(xs, mu_tgt, color=C_PURPLE, linewidth=2.2,
            label=r"$\mu_{P_T}(x) = \mathbb{E}_T[k(\cdot, x)]$")
    ax.fill_between(xs, mu_src, mu_tgt,
                    color=C_AMBER, alpha=0.32,
                    label=r"distance in RKHS = MMD")
    ax.set_xlim(-4, 6)
    ax.set_ylabel("kernel mean  μ(x)")
    ax.set_xlabel("x")
    ax.set_title(f"Kernel mean embeddings  "
                 f"(Gaussian RBF, σ={sigma})    "
                 f"MMD² ≈ {mmd_sq:.3f}",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    fig.suptitle("MMD measures distribution distance via RKHS mean embeddings",
                 fontsize=13.5, fontweight="bold", y=1.00)
    fig.tight_layout()
    _save(fig, "fig3_mmd_kernel")


# ---------------------------------------------------------------------------
# Figure 4: CORAL covariance alignment
# ---------------------------------------------------------------------------
def fig4_coral_covariance() -> None:
    """Three covariance heatmaps: source, target, aligned (matches target)."""
    rng = np.random.default_rng(42)
    d = 8

    # Synthetic positive-definite covariance matrices.
    def random_cov(seed, scale=1.0):
        r = np.random.default_rng(seed)
        A = r.normal(size=(d, d))
        C = A @ A.T / d
        # add a gentle structure
        C = scale * (C + np.diag(np.linspace(0.3, 1.2, d)))
        return C

    Cs = random_cov(1, scale=1.0)
    Ct = random_cov(7, scale=1.4)
    # CORAL "aligned source" -> covariance approximates Ct.
    # We synthesise Ca close to Ct (illustrative).
    Ca = 0.92 * Ct + 0.08 * Cs

    coral_loss_before = np.linalg.norm(Cs - Ct, "fro") ** 2 / (4 * d * d)
    coral_loss_after = np.linalg.norm(Ca - Ct, "fro") ** 2 / (4 * d * d)

    vmin = min(Cs.min(), Ct.min(), Ca.min())
    vmax = max(Cs.max(), Ct.max(), Ca.max())

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    titles = [
        f"Source  C_S\n(before alignment)",
        f"Target  C_T\n(reference)",
        f"Aligned source  C_S'\n(after CORAL)",
    ]
    mats = [Cs, Ct, Ca]
    for ax, M, t in zip(axes, mats, titles):
        im = ax.imshow(M, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(t, fontsize=11.5, fontweight="bold")
        ax.set_xticks(range(d))
        ax.set_yticks(range(d))
        ax.set_xticklabels([f"{i+1}" for i in range(d)], fontsize=8)
        ax.set_yticklabels([f"{i+1}" for i in range(d)], fontsize=8)
        ax.grid(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("covariance value", fontsize=10)

    fig.suptitle(
        f"CORAL aligns second-order statistics: "
        f"||C_S − C_T||² = {coral_loss_before:.3f}  →  "
        f"||C_S' − C_T||² = {coral_loss_after:.3f}",
        fontsize=12.8, fontweight="bold", y=1.02)
    _save(fig, "fig4_coral_covariance")


# ---------------------------------------------------------------------------
# Figure 5: t-SNE before vs after domain adaptation
# ---------------------------------------------------------------------------
def fig5_tsne_before_after() -> None:
    """Synthetic 2D 'embeddings' simulating t-SNE plots before/after DA."""
    rng = np.random.default_rng(0)
    n = 140
    n_classes = 4

    # Class centres
    class_centres = np.array([
        [-3.0, 2.5], [3.0, 2.8], [-3.0, -2.6], [3.0, -2.5],
    ])

    def make_blobs(centres, jitter=0.55, shift=(0.0, 0.0)):
        pts, labels = [], []
        for i, c in enumerate(centres):
            pts.append(rng.normal(loc=c + np.array(shift),
                                  scale=jitter, size=(n, 2)))
            labels.append(np.full(n, i))
        return np.vstack(pts), np.concatenate(labels)

    # Before: source forms tight clusters per class; target is shifted
    # (separate side-cluster per class) -> domain dominates over class.
    src_pts, src_lbl = make_blobs(class_centres, 0.55, shift=(0.0, 0.0))
    tgt_pts, tgt_lbl = make_blobs(class_centres + np.array([0.4, 0.4]),
                                  0.6, shift=(4.5, -3.0))

    # After: target collapses into source clusters (per-class overlap)
    src2_pts, src2_lbl = make_blobs(class_centres, 0.6, shift=(0.0, 0.0))
    tgt2_pts, tgt2_lbl = make_blobs(class_centres, 0.7, shift=(0.25, 0.25))

    palette = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    for ax, (pts_s, lbl_s, pts_t, lbl_t), title in zip(
        axes,
        [
            (src_pts, src_lbl, tgt_pts, tgt_lbl),
            (src2_pts, src2_lbl, tgt2_pts, tgt2_lbl),
        ],
        [
            "Before adaptation: domains separate; classes blur across domains",
            "After adaptation: classes cluster; domains overlap inside each class",
        ],
    ):
        for c in range(n_classes):
            mask_s = lbl_s == c
            mask_t = lbl_t == c
            ax.scatter(pts_s[mask_s, 0], pts_s[mask_s, 1],
                       s=22, c=palette[c], alpha=0.75,
                       marker="o", edgecolor="white", linewidth=0.3,
                       label=f"Source · class {c+1}" if c == 0 else None)
            ax.scatter(pts_t[mask_t, 0], pts_t[mask_t, 1],
                       s=24, c=palette[c], alpha=0.75,
                       marker="x", linewidth=1.6,
                       label=f"Target · class {c+1}" if c == 0 else None)
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.set_xlim(-6, 9)
        ax.set_ylim(-7.5, 5.5)

    # Custom legend (markers vs colours)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=C_GRAY, markersize=8, label="Source (●)"),
        plt.Line2D([0], [0], marker="x", color=C_GRAY,
                   markersize=8, linewidth=0, label="Target (×)"),
    ]
    for c in range(n_classes):
        legend_handles.append(
            mpatches.Patch(color=palette[c], label=f"Class {c+1}"))
    fig.legend(handles=legend_handles, ncol=6,
               loc="lower center", bbox_to_anchor=(0.5, -0.02),
               frameon=False, fontsize=9.5)

    fig.suptitle("t-SNE of learned features before vs after DANN",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_tsne_before_after")


# ---------------------------------------------------------------------------
# Figure 6: Self-training / pseudo-labelling loop
# ---------------------------------------------------------------------------
def fig6_self_training() -> None:
    """Pseudo-labelling cycle with confidence threshold gate."""
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, label, color, text_color="white", fontsize=11):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            linewidth=1.6, edgecolor=color, facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight="bold")

    def arrow(p0, p1, color=C_DARK, lw=1.7, curved=0.0, label=None,
              label_offset=(0, 0)):
        a = FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=14,
            color=color, linewidth=lw,
            connectionstyle=f"arc3,rad={curved}",
        )
        ax.add_patch(a)
        if label is not None:
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            ax.text(mx + label_offset[0], my + label_offset[1], label,
                    ha="center", fontsize=9, color=color,
                    fontweight="bold")

    # 1. Source labelled  ->  initial model
    box(0.3, 3.6, 2.0, 1.0, "Source\n(labelled)", C_BLUE, fontsize=10.5)
    box(3.0, 3.6, 2.2, 1.0, "Train\nclassifier f", C_DARK, fontsize=11)

    # 2. Apply to target  ->  predictions
    box(5.9, 3.6, 2.2, 1.0, "Predict on\ntarget x_t", C_PURPLE, fontsize=11)

    # 3. Threshold gate
    box(8.8, 3.6, 2.7, 1.0,
        "Confidence > τ ?\nkeep top-confidence preds",
        C_AMBER, text_color=C_DARK, fontsize=10.5)

    # 4. Pseudo-labelled set
    box(8.8, 0.9, 2.7, 1.0,
        "Pseudo-labelled\ntarget set", C_GREEN, fontsize=11)

    # 5. Discarded
    box(5.9, 0.9, 2.2, 1.0, "Low-confidence\n(discarded)", C_GRAY, fontsize=10)

    # Arrows
    arrow((2.3, 4.1), (3.0, 4.1))
    arrow((5.2, 4.1), (5.9, 4.1))
    arrow((8.1, 4.1), (8.8, 4.1))
    arrow((10.15, 3.6), (10.15, 1.9), label="yes",
          label_offset=(0.4, 0.3), color=C_GREEN)
    arrow((8.8, 3.7), (8.1, 1.9), label="no",
          label_offset=(-0.4, 0.3), color=C_GRAY, curved=0.15)

    # Loop back
    arrow((8.8, 1.4), (4.1, 3.6),
          color=C_BLUE, curved=0.35, lw=1.8,
          label="add to training set,  retrain",
          label_offset=(0.5, -0.55))

    ax.set_title("Self-training: bootstrap target labels from confident predictions",
                 fontsize=13.5, fontweight="bold", pad=12)

    # Caption
    ax.text(6.0, -0.05,
            "Risks: confirmation bias on noisy pseudo-labels.  "
            "Mitigations: high τ, class-balanced sampling, consistency regularisation.",
            ha="center", fontsize=9.5, color=C_DARK, fontstyle="italic")

    _save(fig, "fig6_self_training")


# ---------------------------------------------------------------------------
# Figure 7: Office-31 / DomainNet benchmark
# ---------------------------------------------------------------------------
def fig7_office31_benchmark() -> None:
    """Bar chart: domain-adaptation accuracy across methods on two benchmarks."""
    methods = [
        "Source-only\n(no DA)",
        "AdaBN",
        "Deep CORAL",
        "DAN\n(MMD)",
        "DANN",
        "CDAN",
    ]
    # Representative literature numbers (avg accuracy %).
    # Office-31 (avg over 6 transfer tasks, ResNet-50 backbone).
    office31 = [76.1, 79.4, 82.7, 83.7, 86.4, 87.7]
    # DomainNet (avg over 30 tasks, much harder; ResNet-50 backbone).
    domainnet = [32.9, 34.2, 36.0, 37.1, 38.3, 40.5]

    x = np.arange(len(methods))
    w = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.6))
    bars1 = ax.bar(x - w / 2, office31, w, color=C_BLUE,
                   edgecolor="white", label="Office-31  (avg of 6 tasks)")
    bars2 = ax.bar(x + w / 2, domainnet, w, color=C_PURPLE,
                   edgecolor="white", label="DomainNet  (avg of 30 tasks)")

    for bars in (bars1, bars2):
        for b in bars:
            v = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, v + 0.7,
                    f"{v:.1f}", ha="center", fontsize=9.5,
                    color=C_DARK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Target-domain accuracy  (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(
        "Domain adaptation steadily improves target accuracy over a source-only baseline",
        fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.set_axisbelow(True)

    # Annotation: gain over baseline for DANN
    base_o, dann_o = office31[0], office31[4]
    ax.annotate(
        f"+{dann_o - base_o:.1f} pts\n(Office-31)",
        xy=(4 - w / 2, dann_o), xytext=(4 - w / 2, dann_o + 8),
        ha="center", fontsize=9, color=C_GREEN, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.4),
    )

    ax.text(0.5, -0.18,
            "Numbers are representative literature averages with a ResNet-50 backbone; "
            "exact values vary per task pair and implementation.",
            transform=ax.transAxes, ha="center", fontsize=9,
            color=C_GRAY, fontstyle="italic")

    _save(fig, "fig7_office31_benchmark")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"EN dir: {EN_DIR}")
    print(f"ZH dir: {ZH_DIR}")
    print()
    for fn in (
        fig1_distribution_shift,
        fig2_dann_architecture,
        fig3_mmd_kernel,
        fig4_coral_covariance,
        fig5_tsne_before_after,
        fig6_self_training,
        fig7_office31_benchmark,
    ):
        print(f"[{fn.__name__}]")
        fn()
    print("\nDone.")


if __name__ == "__main__":
    main()
