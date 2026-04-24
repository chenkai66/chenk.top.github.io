"""
Figure generation script for Linear Algebra Chapter 12:
Sparse Matrices and Compressed Sensing.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly.

Figures:
    fig1_sparse_formats        Same sparse matrix shown as a spy-plot together
                               with its COO / CSR / CSC storage layouts.
    fig2_memory_savings        Memory used by dense vs sparse storage as the
                               density of nonzeros increases - shows where
                               sparse formats start to pay off.
    fig3_l1_vs_l2_geometry     The classic diamond-vs-circle picture: an
                               affine constraint line touches the L1 ball at
                               a corner (sparse), the L2 ball on its surface
                               (dense).
    fig4_compressed_sensing    Recovery of a 256-dim 10-sparse signal from
                               only 64 random Gaussian measurements via ISTA;
                               compares true vs recovered stems.
    fig5_lasso_path            Coefficient paths of LASSO on a synthetic
                               problem as lambda decreases - features enter
                               one at a time, path is piecewise linear.
    fig6_iht_visualization     Iterative Hard Thresholding: shows the signal
                               at iterations 0, 2, 5 and the residual norm
                               curve dropping to zero.
    fig7_rip_intuition         Restricted Isometry Property intuition: a
                               random Gaussian Phi nearly preserves the norm
                               of sparse vectors but distorts dense ones.

Usage:
    python3 scripts/figures/linear-algebra/12-sparse-matrices-and-compressed-sensing.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from scipy import sparse

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

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "12-sparse-matrices-and-compressed-sensing"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "12-稀疏矩阵与压缩感知"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH EN and ZH asset folders as <name>.png."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        path = d / f"{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def hard_threshold(x: np.ndarray, k: int) -> np.ndarray:
    """Keep top-k entries by magnitude, zero the rest."""
    out = np.zeros_like(x)
    if k <= 0:
        return out
    idx = np.argpartition(np.abs(x), -k)[-k:]
    out[idx] = x[idx]
    return out


def ista(Phi: np.ndarray, y: np.ndarray, lam: float,
         max_iter: int = 2000, tol: float = 1e-7) -> np.ndarray:
    n = Phi.shape[1]
    x = np.zeros(n)
    L = float(np.linalg.norm(Phi.T @ Phi, 2))
    for _ in range(max_iter):
        x_old = x.copy()
        x = soft_threshold(x - (Phi.T @ (Phi @ x - y)) / L, lam / L)
        if np.linalg.norm(x - x_old) < tol:
            break
    return x


# ---------------------------------------------------------------------------
# Figure 1: Sparse formats - spy plot + COO/CSR/CSC layouts
# ---------------------------------------------------------------------------
def fig1_sparse_formats() -> None:
    rng = np.random.default_rng(7)
    n = 8
    M = np.zeros((n, n))
    # Hand-picked nonzeros so the storage layout is short enough to read.
    triples = [
        (0, 0, 5), (0, 3, 8),
        (1, 1, 3),
        (2, 0, 2), (2, 4, 7),
        (3, 3, 1), (3, 6, 4),
        (4, 2, 9),
        (5, 5, 6), (5, 7, 2),
        (6, 1, 4),
        (7, 4, 3), (7, 7, 8),
    ]
    for r, c, v in triples:
        M[r, c] = v

    sp = sparse.csr_matrix(M)
    coo = sp.tocoo()

    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6], wspace=0.25)

    # --- left: spy plot ---
    ax = fig.add_subplot(gs[0, 0])
    ax.spy(M, markersize=18, color=C_BLUE)
    ax.set_title("Sparse matrix (spy plot)\n13 nonzeros / 64 entries  (20% density)",
                 fontsize=11, color=C_DARK, pad=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.tick_params(labelsize=9)

    # --- right: storage layouts ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    def row_text(y: float, label: str, values, color=C_DARK):
        ax2.text(0.02, y, label, fontsize=11, fontweight="bold",
                 color=color, va="center")
        ax2.text(0.22, y, " ".join(f"{int(v):>2}" for v in values),
                 fontsize=10, family="monospace", va="center", color=C_DARK)

    ax2.text(0.02, 0.96, "Storage formats (same matrix)",
             fontsize=12, fontweight="bold", color=C_DARK)

    # COO
    ax2.text(0.02, 0.86, "COO  (coordinate)", fontsize=11,
             fontweight="bold", color=C_BLUE)
    row_text(0.78, "rows ", coo.row, color=C_GRAY)
    row_text(0.72, "cols ", coo.col, color=C_GRAY)
    row_text(0.66, "vals ", coo.data, color=C_GRAY)

    # CSR
    ax2.text(0.02, 0.55, "CSR  (compressed row)", fontsize=11,
             fontweight="bold", color=C_PURPLE)
    row_text(0.47, "indptr", sp.indptr, color=C_GRAY)
    row_text(0.41, "cols  ", sp.indices, color=C_GRAY)
    row_text(0.35, "vals  ", sp.data, color=C_GRAY)

    # CSC
    csc = sp.tocsc()
    ax2.text(0.02, 0.24, "CSC  (compressed column)", fontsize=11,
             fontweight="bold", color=C_GREEN)
    row_text(0.16, "indptr", csc.indptr, color=C_GRAY)
    row_text(0.10, "rows  ", csc.indices, color=C_GRAY)
    row_text(0.04, "vals  ", csc.data, color=C_GRAY)

    save(fig, "fig1_sparse_formats")


# ---------------------------------------------------------------------------
# Figure 2: Memory savings vs density
# ---------------------------------------------------------------------------
def fig2_memory_savings() -> None:
    n = 1000  # n x n matrix
    densities = np.linspace(0.001, 0.8, 200)
    bytes_per_value = 8       # float64
    bytes_per_index = 4       # int32

    dense_mb = (n * n * bytes_per_value) / (1024 ** 2)

    nnz = densities * n * n
    # CSR: data (8) + col idx (4) per nnz, plus (n+1)*4 row pointers
    csr_mb = (nnz * (bytes_per_value + bytes_per_index) +
              (n + 1) * bytes_per_index) / (1024 ** 2)
    # COO: 3 * nnz - row(4) + col(4) + val(8)
    coo_mb = (nnz * (2 * bytes_per_index + bytes_per_value)) / (1024 ** 2)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axhline(dense_mb, color=C_AMBER, linewidth=2.5,
               label=f"Dense (always {dense_mb:.1f} MB)")
    ax.plot(densities * 100, csr_mb, color=C_BLUE, linewidth=2.5,
            label="CSR  (4 + 8 bytes per nonzero)")
    ax.plot(densities * 100, coo_mb, color=C_PURPLE, linewidth=2.5,
            linestyle="--", label="COO  (4 + 4 + 8 bytes per nonzero)")

    # Crossover with CSR (where CSR memory == dense memory).
    crossover_csr = bytes_per_value / (bytes_per_value + bytes_per_index)
    ax.axvline(crossover_csr * 100, color=C_BLUE, linestyle=":",
               linewidth=1.5, alpha=0.7)
    ax.text(crossover_csr * 100 - 1, 1.0,
            f"CSR breaks\neven ~{crossover_csr*100:.0f}%",
            fontsize=9, color=C_BLUE, ha="right")

    # Highlight the region where sparse really wins (<10% density).
    ax.axvspan(0, 10, color=C_GREEN, alpha=0.08)
    ax.text(5, dense_mb * 0.92, "sparse wins big\n(< 10% density)",
            fontsize=10, color=C_GREEN, ha="center", fontweight="bold")

    ax.set_xlabel("Density of nonzeros (%)", fontsize=11)
    ax.set_ylabel("Memory (MB)", fontsize=11)
    ax.set_title(f"Memory cost vs density   (n × n matrix, n = {n})",
                 fontsize=12, color=C_DARK, pad=10)
    ax.legend(loc="upper left", framealpha=0.95, bbox_to_anchor=(0.0, 0.95))
    ax.set_xlim(0, 80)
    ax.set_ylim(0, dense_mb * 1.55)

    save(fig, "fig2_memory_savings")


# ---------------------------------------------------------------------------
# Figure 3: L1 vs L2 geometry
# ---------------------------------------------------------------------------
def fig3_l1_vs_l2_geometry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))

    # The constraint line: a x + b y = c so that the closest L1 contact is a corner.
    a, b, c = 1.0, 0.6, 1.0          # line a*x + b*y = c
    xs = np.linspace(-2.2, 2.2, 200)
    ys = (c - a * xs) / b

    # ---- L1 ----
    ax = axes[0]
    diamond = Polygon([(1, 0), (0, 1), (-1, 0), (0, -1)],
                      closed=True, facecolor=C_BLUE, alpha=0.18,
                      edgecolor=C_BLUE, linewidth=2)
    ax.add_patch(diamond)
    ax.plot(xs, ys, color=C_DARK, linewidth=2,
            label=r"constraint  $a^\top x = c$")

    # The L1-minimizing scaled diamond just touches the line.
    # Touch point on diamond = solve min ||x||_1 s.t. a*x1 + b*x2 = c.
    # With a > b > 0 and c > 0 the optimum is at (c/a, 0).
    tx, ty = c / a, 0.0
    ax.plot(tx, ty, "o", markersize=14, color=C_AMBER,
            markeredgecolor=C_DARK, markeredgewidth=1.5, zorder=5)
    ax.annotate("sparse contact\n(corner on axis)",
                xy=(tx, ty), xytext=(1.35, -0.7), fontsize=10,
                color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.2))

    ax.set_xlim(-2.0, 2.2)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axhline(0, color=C_GRAY, linewidth=0.8, alpha=0.6)
    ax.axvline(0, color=C_GRAY, linewidth=0.8, alpha=0.6)
    ax.set_title(r"$L_1$ ball  $|x_1|+|x_2| \leq r$"
                 "\nfirst contact happens at a corner",
                 fontsize=11, color=C_DARK, pad=8)
    ax.legend(loc="upper right", fontsize=10)

    # ---- L2 ----
    ax = axes[1]
    # L2 contact point: x* = c * a_vec / ||a_vec||^2
    a_vec = np.array([a, b])
    x_star = c * a_vec / (a_vec @ a_vec)
    r = np.linalg.norm(x_star)
    circle = Circle((0, 0), r, facecolor=C_PURPLE, alpha=0.18,
                    edgecolor=C_PURPLE, linewidth=2)
    ax.add_patch(circle)
    ax.plot(xs, ys, color=C_DARK, linewidth=2,
            label=r"constraint  $a^\top x = c$")
    ax.plot(x_star[0], x_star[1], "o", markersize=14, color=C_AMBER,
            markeredgecolor=C_DARK, markeredgewidth=1.5, zorder=5)
    ax.annotate("dense contact\n(both coords nonzero)",
                xy=(x_star[0], x_star[1]), xytext=(1.0, 1.2),
                fontsize=10, color=C_DARK,
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.2))

    ax.set_xlim(-2.0, 2.2)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect("equal")
    ax.axhline(0, color=C_GRAY, linewidth=0.8, alpha=0.6)
    ax.axvline(0, color=C_GRAY, linewidth=0.8, alpha=0.6)
    ax.set_title(r"$L_2$ ball  $x_1^2 + x_2^2 \leq r^2$"
                 "\nfirst contact is generic - no zero coords",
                 fontsize=11, color=C_DARK, pad=8)
    ax.legend(loc="upper right", fontsize=10)

    fig.suptitle("Why $L_1$ promotes sparsity",
                 fontsize=13, color=C_DARK, y=1.02)
    save(fig, "fig3_l1_vs_l2_geometry")


# ---------------------------------------------------------------------------
# Figure 4: Compressed sensing recovery
# ---------------------------------------------------------------------------
def fig4_compressed_sensing() -> None:
    rng = np.random.default_rng(0)
    n, k, m = 256, 10, 64

    x_true = np.zeros(n)
    support = rng.choice(n, size=k, replace=False)
    x_true[support] = rng.standard_normal(k) * 1.5

    Phi = rng.standard_normal((m, n)) / np.sqrt(m)
    y = Phi @ x_true
    x_hat = ista(Phi, y, lam=0.01, max_iter=4000)

    err = np.linalg.norm(x_true - x_hat) / np.linalg.norm(x_true)

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.0), sharex=True)

    ax = axes[0]
    markerline, stemlines, baseline = ax.stem(
        np.arange(n), x_true,
        linefmt="-", markerfmt="o", basefmt=" ")
    plt.setp(stemlines, color=C_BLUE, linewidth=1.5)
    plt.setp(markerline, color=C_BLUE, markersize=5)
    ax.set_title(f"True signal:  n = {n},  k = {k} nonzeros",
                 fontsize=11, color=C_DARK, pad=8)
    ax.set_ylabel("amplitude")
    ax.set_ylim(-3, 3)

    ax = axes[1]
    markerline, stemlines, baseline = ax.stem(
        np.arange(n), x_hat,
        linefmt="-", markerfmt="o", basefmt=" ")
    plt.setp(stemlines, color=C_GREEN, linewidth=1.5)
    plt.setp(markerline, color=C_GREEN, markersize=5)
    ax.set_title(f"Recovered from m = {m} measurements   "
                 f"(relative error = {err:.2e})",
                 fontsize=11, color=C_DARK, pad=8)
    ax.set_xlabel("index")
    ax.set_ylabel("amplitude")
    ax.set_ylim(-3, 3)

    fig.suptitle(r"Compressed sensing:  $y = \Phi x$,  recover via $L_1$",
                 fontsize=13, color=C_DARK, y=1.00)
    fig.tight_layout()
    save(fig, "fig4_compressed_sensing")


# ---------------------------------------------------------------------------
# Figure 5: LASSO solution path
# ---------------------------------------------------------------------------
def fig5_lasso_path() -> None:
    from sklearn.linear_model import lars_path


    rng = np.random.default_rng(3)
    n_samples, n_features, k = 100, 12, 5
    X = rng.standard_normal((n_samples, n_features))
    X /= np.linalg.norm(X, axis=0)
    beta = np.zeros(n_features)
    support = rng.choice(n_features, k, replace=False)
    beta[support] = rng.standard_normal(k) * 2
    y = X @ beta + 0.1 * rng.standard_normal(n_samples)

    alphas, _, coefs = lars_path(X, y, method="lasso", verbose=0)
    # log axis - guard against alpha == 0
    log_alpha = -np.log10(np.maximum(alphas, 1e-6))

    fig, ax = plt.subplots(figsize=(11, 5.6))
    palette = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER,
               "#ef4444", "#06b6d4", "#a855f7", "#22c55e",
               "#f97316", "#14b8a6", "#6366f1", "#84cc16"]

    for j in range(n_features):
        is_true = beta[j] != 0
        ax.plot(log_alpha, coefs[j],
                color=palette[j % len(palette)],
                linewidth=2.5 if is_true else 1.2,
                alpha=1.0 if is_true else 0.45,
                label=f"feature {j}{' *' if is_true else ''}")

    ax.axhline(0, color=C_GRAY, linewidth=0.8)
    ax.set_xlabel(r"$-\log_{10}\lambda$  (regularization weakens →)")
    ax.set_ylabel("coefficient value")
    ax.set_title("LASSO solution path   "
                 "(features marked * are truly nonzero)",
                 fontsize=12, color=C_DARK, pad=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=9, framealpha=0.95)
    fig.tight_layout()
    save(fig, "fig5_lasso_path")


# ---------------------------------------------------------------------------
# Figure 6: Iterative Hard Thresholding
# ---------------------------------------------------------------------------
def fig6_iht_visualization() -> None:
    rng = np.random.default_rng(11)
    n, k, m = 200, 8, 80
    x_true = np.zeros(n)
    support = rng.choice(n, k, replace=False)
    x_true[support] = rng.standard_normal(k) * 1.5

    Phi = rng.standard_normal((m, n)) / np.sqrt(m)
    y = Phi @ x_true

    # Run IHT and snapshot a few iterates.
    # Step size 1/L where L is the spectral norm squared of Phi
    # (Lipschitz constant of the squared-error gradient).
    L = float(np.linalg.norm(Phi, 2)) ** 2
    snapshots = {0: np.zeros(n)}
    x = np.zeros(n)
    residuals = []
    n_iter = 60
    for t in range(1, n_iter + 1):
        grad = Phi.T @ (Phi @ x - y)
        x = hard_threshold(x - grad / L, k)
        residuals.append(np.linalg.norm(Phi @ x - y))
        if t in (2, 5, 30):
            snapshots[t] = x.copy()

    fig = plt.figure(figsize=(13, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.55, wspace=0.3)

    iters_to_plot = [0, 2, 5, 30]
    colors = [C_GRAY, C_PURPLE, C_BLUE, C_GREEN]
    titles = ["iter 0  (start)", "iter 2", "iter 5", "iter 30  (converged)"]

    # The first three on the top row, last on the bottom-left
    positions = [(0, 0), (0, 1), (0, 2), (1, 0)]
    for (r, c), it, color, title in zip(positions, iters_to_plot, colors, titles):
        ax = fig.add_subplot(gs[r, c])
        x_show = snapshots.get(it, x)
        markerline, stemlines, baseline = ax.stem(
            np.arange(n), x_true, linefmt="-", markerfmt=" ", basefmt=" ")
        plt.setp(stemlines, color=C_LIGHT, linewidth=1.0, alpha=0.9)
        markerline2, stemlines2, baseline2 = ax.stem(
            np.arange(n), x_show, linefmt="-", markerfmt="o", basefmt=" ")
        plt.setp(stemlines2, color=color, linewidth=1.5)
        plt.setp(markerline2, color=color, markersize=4)
        ax.set_title(title, fontsize=10, color=C_DARK)
        ax.set_ylim(-3, 3)
        ax.set_xticks([])
        ax.tick_params(labelsize=8)

    # Residual curve on the bottom-right (spans 2 cols)
    ax = fig.add_subplot(gs[1, 1:])
    ax.semilogy(np.arange(1, n_iter + 1), residuals,
                color=C_BLUE, linewidth=2.2)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\|\Phi x_t - y\|_2$  (log)")
    ax.set_title("Residual norm during IHT", fontsize=10, color=C_DARK)

    fig.suptitle("Iterative Hard Thresholding  "
                 r"($x_{t+1} = H_k(x_t - \Phi^\top(\Phi x_t - y))$)",
                 fontsize=13, color=C_DARK, y=1.01)
    save(fig, "fig6_iht_visualization")


# ---------------------------------------------------------------------------
# Figure 7: RIP intuition
# ---------------------------------------------------------------------------
def fig7_rip_intuition() -> None:
    rng = np.random.default_rng(5)
    n, m = 400, 80
    Phi = rng.standard_normal((m, n)) / np.sqrt(m)

    n_trials = 800
    sparsities = [4, 16, 64, 200]
    labels = [f"k = {k}" for k in sparsities]
    colors = [C_GREEN, C_BLUE, C_PURPLE, C_AMBER]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # Left: histograms of ||Phi x|| / ||x|| for varying sparsity.
    ax = axes[0]
    for k, color, label in zip(sparsities, colors, labels):
        ratios = np.empty(n_trials)
        for t in range(n_trials):
            x = np.zeros(n)
            supp = rng.choice(n, size=k, replace=False)
            x[supp] = rng.standard_normal(k)
            ratios[t] = np.linalg.norm(Phi @ x) / np.linalg.norm(x)
        ax.hist(ratios, bins=40, color=color, alpha=0.45,
                edgecolor=color, label=label, density=True)

    ax.axvline(1.0, color=C_DARK, linestyle="--", linewidth=1.5,
               label=r"isometry  $\|\Phi x\| = \|x\|$")
    ax.set_xlabel(r"$\|\Phi x\|_2 / \|x\|_2$")
    ax.set_ylabel("density")
    ax.set_title(f"Random Gaussian $\\Phi$ ({m} × {n})\n"
                 "tightly preserves the norm of sparse vectors",
                 fontsize=11, color=C_DARK, pad=8)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0.0, 2.0)

    # Right: empirical RIP constant delta_k vs k.
    ax = axes[1]
    ks = np.arange(2, 80, 4)
    deltas = []
    for k in ks:
        worst = 0.0
        for _ in range(150):
            x = np.zeros(n)
            supp = rng.choice(n, size=k, replace=False)
            x[supp] = rng.standard_normal(k)
            r = (np.linalg.norm(Phi @ x) ** 2) / (np.linalg.norm(x) ** 2)
            worst = max(worst, abs(r - 1.0))
        deltas.append(worst)

    ax.plot(ks, deltas, "o-", color=C_BLUE, linewidth=2, markersize=6)
    ax.axhline(np.sqrt(2) - 1, color=C_AMBER, linestyle="--",
               linewidth=1.8,
               label=r"Cand$\grave{\mathrm{e}}$s-Tao bound  $\sqrt{2}-1 \approx 0.414$")
    ax.set_xlabel("sparsity $k$")
    ax.set_ylabel(r"empirical $\delta_k$")
    ax.set_title("RIP constant grows with sparsity\n"
                 "below the bound → exact recovery is guaranteed",
                 fontsize=11, color=C_DARK, pad=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(1.0, max(deltas) * 1.1))

    fig.suptitle("Restricted Isometry Property (RIP) intuition",
                 fontsize=13, color=C_DARK, y=1.03)
    fig.tight_layout()
    save(fig, "fig7_rip_intuition")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_sparse_formats()
    fig2_memory_savings()
    fig3_l1_vs_l2_geometry()
    fig4_compressed_sensing()
    fig5_lasso_path()
    fig6_iht_visualization()
    fig7_rip_intuition()
    print("All 7 figures written to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
