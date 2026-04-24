"""
Figure generation script for the standalone article "Kernel Methods".

Generates 6 figures used in both EN and ZH versions of the article. Every
figure is intentionally simple and self-contained -- one geometric or
algorithmic idea per figure.

Figures:
    fig1_kernel_trick           Concentric ring data lifted by phi(x) =
                                (x1, x2, x1^2 + x2^2): the 2D problem is
                                not linearly separable, the 3D problem
                                is separable by a flat plane.
    fig2_common_kernels         2x2 panel of decision-boundary slices
                                produced by linear / polynomial / RBF /
                                sigmoid kernels on the same dataset.
    fig3_rbf_gamma              RBF gamma sweep: underfit -> sweet spot
                                -> overfit, plus a bias-variance curve.
    fig4_mercer                 Mercer decomposition of the RBF kernel
                                visualised as eigenvalue spectrum and
                                top eigenfunctions on [-1, 1].
    fig5_gram_matrix            Gram-matrix structure for three RBF
                                bandwidths on a 3-cluster dataset, plus
                                the eigenvalue spectra.
    fig6_decision_tree          A clean kernel-selection decision tree
                                rendered with matplotlib.

Usage:
    python3 scripts/figures/standalone/kernel-methods.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D proj)

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
C_RED = "#dc2626"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "kernel-methods"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "standalone" / "kernel-methods"


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
# Figure 1: The kernel trick -- 2D ring becomes 3D linearly separable
# ---------------------------------------------------------------------------
def fig1_kernel_trick() -> None:
    rng = np.random.default_rng(2)
    n_in, n_out = 110, 130
    r_in = rng.normal(0.55, 0.18, n_in)
    th_in = rng.uniform(0, 2 * np.pi, n_in)
    inner = np.column_stack([r_in * np.cos(th_in), r_in * np.sin(th_in)])

    r_out = rng.normal(2.1, 0.18, n_out)
    th_out = rng.uniform(0, 2 * np.pi, n_out)
    outer = np.column_stack([r_out * np.cos(th_out), r_out * np.sin(th_out)])

    fig = plt.figure(figsize=(13.2, 5.6))

    # ---- Left: 2D, not linearly separable -----------------------------
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(inner[:, 0], inner[:, 1], c=C_BLUE, s=42,
                edgecolor="white", linewidth=0.8, label="Class A")
    ax1.scatter(outer[:, 0], outer[:, 1], c=C_AMBER, s=42,
                edgecolor="white", linewidth=0.8, label="Class B")

    # Draw a few "failed" linear separators
    xs = np.linspace(-2.7, 2.7, 100)
    for k, (m, c) in enumerate([(0.6, 0.0), (-0.8, 0.4), (0.1, -0.2)]):
        ax1.plot(xs, m * xs + c, color=C_RED, lw=1.2, ls="--", alpha=0.55,
                 label="Failed linear cut" if k == 0 else None)

    ax1.set_xlim(-2.9, 2.9)
    ax1.set_ylim(-2.9, 2.9)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x_1$")
    ax1.set_ylabel(r"$x_2$")
    ax1.set_title("Input space $\\mathbb{R}^2$: not linearly separable",
                  fontsize=12, fontweight="bold", color=C_DARK)
    ax1.legend(loc="upper right", framealpha=0.95, fontsize=9)

    # ---- Right: 3D after phi(x) = (x1, x2, x1^2 + x2^2) ----------------
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    z_in = inner[:, 0] ** 2 + inner[:, 1] ** 2
    z_out = outer[:, 0] ** 2 + outer[:, 1] ** 2
    ax2.scatter(inner[:, 0], inner[:, 1], z_in, c=C_BLUE, s=42,
                edgecolor="white", linewidth=0.6, depthshade=True)
    ax2.scatter(outer[:, 0], outer[:, 1], z_out, c=C_AMBER, s=42,
                edgecolor="white", linewidth=0.6, depthshade=True)

    # Separating plane z = 1.8
    xx, yy = np.meshgrid(np.linspace(-2.6, 2.6, 12),
                         np.linspace(-2.6, 2.6, 12))
    zz = np.full_like(xx, 1.8)
    ax2.plot_surface(xx, yy, zz, color=C_GREEN, alpha=0.18,
                     edgecolor=C_GREEN, linewidth=0.4)

    ax2.set_xlabel(r"$x_1$")
    ax2.set_ylabel(r"$x_2$")
    ax2.set_zlabel(r"$x_1^2 + x_2^2$")
    ax2.set_title(r"Feature space $\phi(x)$: separable by a plane",
                  fontsize=12, fontweight="bold", color=C_DARK)
    ax2.view_init(elev=22, azim=-58)

    fig.suptitle("The Kernel Trick: lift, then separate",
                 fontsize=14, fontweight="bold", y=1.01, color=C_DARK)
    fig.tight_layout()
    _save(fig, "fig1_kernel_trick")


# ---------------------------------------------------------------------------
# Figure 2: Four common kernels on the same data
# ---------------------------------------------------------------------------
def fig2_common_kernels() -> None:
    from sklearn.datasets import make_moons
    from sklearn.svm import SVC

    X, y = make_moons(n_samples=260, noise=0.22, random_state=4)

    kernels = [
        ("Linear", dict(kernel="linear", C=1.0), C_BLUE),
        ("Polynomial (d=3)", dict(kernel="poly", degree=3, C=1.0,
                                  gamma="scale", coef0=1.0), C_PURPLE),
        ("RBF ($\\gamma=1.0$)", dict(kernel="rbf", C=1.0, gamma=1.0), C_GREEN),
        ("Sigmoid", dict(kernel="sigmoid", C=1.0, gamma=0.5,
                         coef0=-1.0), C_AMBER),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 9.4))

    # Mesh for decision-function contour
    xx, yy = np.meshgrid(np.linspace(-1.8, 2.8, 240),
                         np.linspace(-1.4, 1.8, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for ax, (name, params, color) in zip(axes.ravel(), kernels):
        clf = SVC(**params).fit(X, y)
        Z = clf.decision_function(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9],
                    colors=[C_LIGHT, "#fef3c7"], alpha=0.55)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[color, C_DARK, color],
                   linestyles=["--", "-", "--"], linewidths=[1.0, 1.8, 1.0])

        ax.scatter(X[y == 0, 0], X[y == 0, 1], c=C_BLUE, s=28,
                   edgecolor="white", linewidth=0.6)
        ax.scatter(X[y == 1, 0], X[y == 1, 1], c=C_AMBER, s=28,
                   edgecolor="white", linewidth=0.6)

        train_acc = clf.score(X, y)
        ax.set_title(f"{name}  -  train acc = {train_acc:.2f}",
                     fontsize=11, fontweight="bold", color=C_DARK)
        ax.set_xlim(-1.8, 2.8)
        ax.set_ylim(-1.4, 1.8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Common kernels on the same moons dataset",
                 fontsize=14, fontweight="bold", y=0.995, color=C_DARK)
    fig.tight_layout()
    _save(fig, "fig2_common_kernels")


# ---------------------------------------------------------------------------
# Figure 3: RBF gamma effect -- underfit, sweet spot, overfit + curve
# ---------------------------------------------------------------------------
def fig3_rbf_gamma() -> None:
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    X, y = make_moons(n_samples=320, noise=0.28, random_state=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4,
                                              random_state=1)

    fig = plt.figure(figsize=(13.4, 8.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.95], hspace=0.32,
                          wspace=0.22)

    settings = [
        (0.05, "Too small  $\\gamma=0.05$  (underfit)"),
        (1.0, "Sweet spot  $\\gamma=1.0$"),
        (50.0, "Too large  $\\gamma=50$  (overfit)"),
    ]

    xx, yy = np.meshgrid(np.linspace(-1.8, 2.8, 220),
                         np.linspace(-1.4, 1.8, 180))
    grid = np.c_[xx.ravel(), yy.ravel()]

    for col, (g, title) in enumerate(settings):
        ax = fig.add_subplot(gs[0, col])
        clf = SVC(kernel="rbf", C=1.0, gamma=g).fit(X_tr, y_tr)
        Z = clf.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-1e9, 0, 1e9],
                    colors=[C_LIGHT, "#fef3c7"], alpha=0.55)
        ax.contour(xx, yy, Z, levels=[0], colors=[C_DARK], linewidths=1.8)

        ax.scatter(X_tr[y_tr == 0, 0], X_tr[y_tr == 0, 1], c=C_BLUE, s=22,
                   edgecolor="white", linewidth=0.5)
        ax.scatter(X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1], c=C_AMBER, s=22,
                   edgecolor="white", linewidth=0.5)

        train_acc = clf.score(X_tr, y_tr)
        test_acc = clf.score(X_te, y_te)
        ax.set_title(title, fontsize=11, fontweight="bold", color=C_DARK)
        ax.text(0.02, 0.97,
                f"train = {train_acc:.2f}\n test = {test_acc:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C_GRAY, alpha=0.9))
        ax.set_xlim(-1.8, 2.8)
        ax.set_ylim(-1.4, 1.8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bias-variance style curve across gammas
    ax_b = fig.add_subplot(gs[1, :])
    gammas = np.logspace(-2, 2, 28)
    tr_scores, te_scores = [], []
    for g in gammas:
        clf = SVC(kernel="rbf", C=1.0, gamma=g).fit(X_tr, y_tr)
        tr_scores.append(clf.score(X_tr, y_tr))
        te_scores.append(clf.score(X_te, y_te))

    ax_b.plot(gammas, tr_scores, color=C_BLUE, lw=2.2, marker="o",
              markersize=4, label="Train accuracy")
    ax_b.plot(gammas, te_scores, color=C_AMBER, lw=2.2, marker="s",
              markersize=4, label="Test accuracy")

    best_g = gammas[int(np.argmax(te_scores))]
    ax_b.axvline(best_g, color=C_GREEN, ls="--", lw=1.4,
                 label=f"Best $\\gamma\\approx{best_g:.2f}$")
    ax_b.axvspan(20, 1e2, alpha=0.10, color=C_RED)
    ax_b.axvspan(1e-2, 0.1, alpha=0.10, color=C_RED)
    ax_b.text(0.025, 0.55, "underfit\nzone", color=C_RED, ha="center",
              fontsize=10, fontweight="bold")
    ax_b.text(45, 0.55, "overfit\nzone", color=C_RED, ha="center",
              fontsize=10, fontweight="bold")

    ax_b.set_xscale("log")
    ax_b.set_xlabel(r"RBF bandwidth $\gamma$ (log scale)")
    ax_b.set_ylabel("Accuracy")
    ax_b.set_title("RBF $\\gamma$ controls the bias-variance trade-off",
                   fontsize=12, fontweight="bold", color=C_DARK)
    ax_b.set_ylim(0.4, 1.02)
    ax_b.legend(loc="lower center", ncol=3, framealpha=0.95)

    fig.suptitle("RBF kernel: bandwidth $\\gamma$ decides under- vs. over-fit",
                 fontsize=14, fontweight="bold", y=1.00, color=C_DARK)
    _save(fig, "fig3_rbf_gamma")


# ---------------------------------------------------------------------------
# Figure 4: Mercer's theorem -- eigenvalue spectrum + top eigenfunctions
# ---------------------------------------------------------------------------
def fig4_mercer() -> None:
    # Discretise [-1, 1] and form the RBF Gram matrix; eigendecompose.
    n = 220
    xs = np.linspace(-1.0, 1.0, n)
    sigma = 0.35
    diff = xs[:, None] - xs[None, :]
    K = np.exp(-(diff ** 2) / (2 * sigma ** 2))

    # eigh returns ascending; reverse for descending
    w, V = np.linalg.eigh(K)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]

    # Normalise eigenvectors so max(|phi|) = 1 for plotting
    eigfuncs = V[:, :4] / np.max(np.abs(V[:, :4]), axis=0, keepdims=True)
    # Fix sign so the first non-zero value is positive (cosmetic).
    for j in range(eigfuncs.shape[1]):
        idx = np.argmax(np.abs(eigfuncs[:, j]))
        if eigfuncs[idx, j] < 0:
            eigfuncs[:, j] *= -1

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.0),
                             gridspec_kw={"width_ratios": [1.0, 1.2]})

    # ---- Left: eigenvalue spectrum (log) -------------------------------
    k = 30
    axes[0].bar(range(1, k + 1), w[:k] / w[0], color=C_BLUE,
                edgecolor=C_DARK, linewidth=0.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Index $k$")
    axes[0].set_ylabel(r"Normalised eigenvalue $\lambda_k / \lambda_1$")
    axes[0].set_title("RBF eigenvalue spectrum decays fast",
                      fontsize=12, fontweight="bold", color=C_DARK)
    axes[0].text(0.98, 0.95,
                 r"$K(x,y)=\sum_k \lambda_k\,\phi_k(x)\phi_k(y)$",
                 transform=axes[0].transAxes, ha="right", va="top",
                 fontsize=11, bbox=dict(boxstyle="round,pad=0.35",
                 facecolor="white", edgecolor=C_GRAY, alpha=0.95))

    # ---- Right: top 4 eigenfunctions -----------------------------------
    palette = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
    for j in range(4):
        axes[1].plot(xs, eigfuncs[:, j], color=palette[j], lw=2.0,
                     label=f"$\\phi_{{{j + 1}}}(x)$")
    axes[1].axhline(0, color=C_GRAY, lw=0.6)
    axes[1].set_xlabel(r"$x$")
    axes[1].set_ylabel(r"$\phi_k(x)$")
    axes[1].set_title("Top eigenfunctions (the implicit feature map)",
                      fontsize=12, fontweight="bold", color=C_DARK)
    axes[1].legend(loc="upper right", ncol=2, framealpha=0.95)

    fig.suptitle("Mercer's theorem: kernel = sum over eigenfunctions",
                 fontsize=14, fontweight="bold", y=1.02, color=C_DARK)
    fig.tight_layout()
    _save(fig, "fig4_mercer")


# ---------------------------------------------------------------------------
# Figure 5: Gram matrix structure for three RBF bandwidths
# ---------------------------------------------------------------------------
def fig5_gram_matrix() -> None:
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=120, centers=3, cluster_std=0.6,
                      random_state=2)
    # Sort rows by cluster so the block structure is visible
    order = np.argsort(y)
    X = X[order]

    fig, axes = plt.subplots(2, 3, figsize=(13.4, 8.6),
                             gridspec_kw={"height_ratios": [1.0, 0.85]})

    gammas = [0.05, 0.5, 5.0]
    titles = [r"$\gamma=0.05$ (too global)",
              r"$\gamma=0.5$ (block structure visible)",
              r"$\gamma=5.0$ (too local, near identity)"]

    for col, (g, t) in enumerate(zip(gammas, titles)):
        diff = X[:, None, :] - X[None, :, :]
        d2 = np.sum(diff ** 2, axis=-1)
        K = np.exp(-g * d2)

        im = axes[0, col].imshow(K, cmap="viridis", vmin=0, vmax=1)
        axes[0, col].set_title(t, fontsize=11, fontweight="bold",
                               color=C_DARK)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        fig.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

        # Eigenvalue spectrum on a log scale
        eigvals = np.linalg.eigvalsh(K)[::-1]
        eigvals = np.clip(eigvals, 1e-10, None)
        axes[1, col].plot(range(1, len(eigvals) + 1), eigvals,
                          color=C_BLUE, lw=1.8, marker="o", markersize=3)
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel("Index $k$")
        axes[1, col].set_ylabel(r"$\lambda_k$")
        axes[1, col].set_title("Eigenvalue spectrum",
                               fontsize=10, color=C_DARK)
        axes[1, col].set_ylim(1e-8, 1e2)

    fig.suptitle("Gram matrix $K_{ij}=k(x_i,x_j)$ structure under different $\\gamma$",
                 fontsize=14, fontweight="bold", y=1.00, color=C_DARK)
    fig.tight_layout()
    _save(fig, "fig5_gram_matrix")


# ---------------------------------------------------------------------------
# Figure 6: Kernel selection decision tree
# ---------------------------------------------------------------------------
def fig6_decision_tree() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 8.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, fontcolor=C_DARK,
            fontweight="bold", fontsize=10):
        patch = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                               boxstyle="round,pad=0.04,rounding_size=0.18",
                               facecolor=face, edgecolor=edge, linewidth=1.6)
        ax.add_patch(patch)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=fontcolor)

    def arrow(x1, y1, x2, y2, label=None, label_offset=(0.0, 0.0),
              color=C_GRAY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="-|>", mutation_scale=14,
                     color=color, linewidth=1.4))
        if label:
            ax.text((x1 + x2) / 2 + label_offset[0],
                    (y1 + y2) / 2 + label_offset[1], label,
                    fontsize=9, color=C_DARK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.18",
                              facecolor="white", edgecolor="none",
                              alpha=0.9))

    # Decision nodes (light) and leaf nodes (coloured)
    box(6.0, 9.4, 5.0, 0.7, "Start: choose a kernel", C_DARK, C_DARK,
        fontcolor="white", fontsize=12)

    box(6.0, 8.2, 5.6, 0.7, "Is the data already linearly separable?",
        C_LIGHT, C_GRAY)
    box(2.2, 7.0, 3.0, 0.7, "LINEAR kernel", C_BLUE, C_DARK,
        fontcolor="white")

    box(8.2, 7.0, 5.4, 0.7, "Time series with seasonality?",
        C_LIGHT, C_GRAY)
    box(11.0, 5.8, 2.0, 0.7, "PERIODIC", C_PURPLE, C_DARK, fontcolor="white")

    box(7.2, 5.8, 4.6, 0.7, "Need smoothness control (GP)?",
        C_LIGHT, C_GRAY)
    box(10.6, 4.6, 2.4, 0.7, "MATERN", C_GREEN, C_DARK, fontcolor="white")

    box(6.4, 4.6, 4.4, 0.7, "Sparse high-dim (text, genes)?",
        C_LIGHT, C_GRAY)
    box(9.4, 3.4, 3.0, 0.7, "LINEAR / POLY (d=2)", C_BLUE, C_DARK,
        fontcolor="white")

    box(5.6, 3.4, 4.0, 0.7, "Default choice", C_AMBER, C_DARK,
        fontcolor="white")
    box(5.6, 2.0, 4.0, 0.9, "RBF kernel\n(tune $\\gamma$ and $C$)",
        C_AMBER, C_DARK, fontcolor="white", fontsize=11)

    # Arrows
    arrow(6.0, 9.05, 6.0, 8.55)
    arrow(5.0, 7.85, 2.6, 7.35, "Yes", label_offset=(-0.2, 0.05))
    arrow(7.0, 7.85, 8.2, 7.35, "No", label_offset=(0.2, 0.05))

    arrow(9.4, 6.65, 10.9, 6.15, "Yes", label_offset=(0.05, 0.05))
    arrow(7.4, 6.65, 7.2, 6.15, "No", label_offset=(-0.25, 0.05))

    arrow(8.4, 5.45, 10.4, 4.95, "Yes", label_offset=(0.05, 0.05))
    arrow(6.5, 5.45, 6.4, 4.95, "No", label_offset=(-0.25, 0.05))

    arrow(7.4, 4.25, 9.0, 3.75, "Yes", label_offset=(0.05, 0.05))
    arrow(5.7, 4.25, 5.6, 3.75, "No", label_offset=(-0.25, 0.05))

    arrow(5.6, 2.95, 5.6, 2.45)

    # Legend
    legend_elems = [
        Line2D([0], [0], marker="s", color="w", label="Decision node",
               markerfacecolor=C_LIGHT, markeredgecolor=C_GRAY,
               markersize=12),
        Line2D([0], [0], marker="s", color="w", label="Default leaf",
               markerfacecolor=C_AMBER, markersize=12),
        Line2D([0], [0], marker="s", color="w", label="Specialised leaf",
               markerfacecolor=C_GREEN, markersize=12),
        Line2D([0], [0], marker="s", color="w", label="Simple leaf",
               markerfacecolor=C_BLUE, markersize=12),
    ]
    ax.legend(handles=legend_elems, loc="lower left",
              bbox_to_anchor=(0.0, 0.0), framealpha=0.95, ncol=2,
              fontsize=9)

    ax.set_title("Kernel selection decision tree",
                 fontsize=14, fontweight="bold", color=C_DARK, pad=10)
    _save(fig, "fig6_decision_tree")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    figures = [
        ("fig1_kernel_trick", fig1_kernel_trick),
        ("fig2_common_kernels", fig2_common_kernels),
        ("fig3_rbf_gamma", fig3_rbf_gamma),
        ("fig4_mercer", fig4_mercer),
        ("fig5_gram_matrix", fig5_gram_matrix),
        ("fig6_decision_tree", fig6_decision_tree),
    ]
    for name, func in figures:
        print(f"[kernel-methods] {name}")
        func()
    print("Done.")


if __name__ == "__main__":
    main()
