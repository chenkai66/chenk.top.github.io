"""
Figure generation script for the standalone post:
"Low-Rank Matrix Approximation & Pseudoinverse" (EN + ZH).

Generates 5 figures used in both the EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in 3Blue1Brown style.

Figures:
    fig1_lowrank_compression   Image-compression demonstration: a procedurally
                               generated grayscale "scene" rebuilt at rank
                               k = 2, 8, 32 plus the singular-value decay
                               curve and the cumulative-energy curve.  Shows
                               directly that a few singular values carry most
                               of the information.
    fig2_pseudoinverse_svd     Pseudoinverse via SVD on an overdetermined
                               least-squares problem y ~ a x + b.  Left panel
                               shows the data and the regression line that
                               A^+ b produces; right panel shows the
                               geometric meaning -- A^+ b is the projection
                               of b onto the column space of A.
    fig3_eckart_young          Eckart-Young theorem: best rank-k approximation
                               error vs k, compared with random rank-k
                               approximations.  The truncated-SVD curve sits
                               exactly on sqrt(sum_{i>k} sigma_i^2), and the
                               random baseline is far above it.
    fig4_recommender_mf        Recommender-system matrix factorization: a
                               sparse user-item rating matrix R is
                               approximated by U V^T with rank r.  Shows the
                               observed entries, the latent-factor heatmap,
                               and the reconstructed dense matrix.
    fig5_lora_connection       LoRA as low-rank adaptation of a frozen weight:
                               W' = W + B A with rank r << min(d, k).  Shows
                               (a) full fine-tuning parameter cost, (b) LoRA
                               B A = Delta W parameter cost, and (c) the
                               singular-value spectrum of an empirically low-
                               rank Delta W to motivate the method.

Usage:
    python3 scripts/figures/standalone/low-rank-pseudoinverse.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary
C_PURPLE = "#7c3aed"   # secondary
C_GREEN = "#10b981"    # accent (signal / kept)
C_AMBER = "#f59e0b"    # warning (truncated / discarded / random baseline)
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/standalone/"
    "low-rank-approximation-pseudoinverse"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/standalone/"
    "矩阵低秩近似-伪逆"
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
def truncated_svd(A: np.ndarray, k: int) -> np.ndarray:
    """Best rank-k approximation of A in Frobenius / spectral norm."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def make_scene(n: int = 96, seed: int = 0) -> np.ndarray:
    """Procedural grayscale 'scene' that compresses well: low-rank smooth
    gradient + a few sharp ring/stripe features."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:n, 0:n] / n
    g = 0.55 * (0.6 + 0.4 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y))
    rings = 0
    for cx, cy, r, w, amp in [
        (0.30, 0.32, 0.18, 0.025, 0.45),
        (0.70, 0.55, 0.12, 0.018, 0.35),
        (0.55, 0.78, 0.08, 0.012, 0.30),
    ]:
        d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        rings = rings + amp * np.exp(-((d - r) ** 2) / (2 * w * w))
    stripes = 0.18 * np.sin(14 * np.pi * x + 1.0)
    img = g + 0.6 * rings + 0.25 * stripes
    img = img + 0.01 * rng.standard_normal(img.shape)
    img = (img - img.min()) / (img.max() - img.min())
    return img


# ===========================================================================
# Figure 1: Low-rank approximation = image compression
# ===========================================================================
def fig1_lowrank_compression():
    img = make_scene(n=96)
    n, m = img.shape
    full_params = n * m
    U, s, Vt = np.linalg.svd(img, full_matrices=False)

    ks = [2, 8, 32]
    recons = [truncated_svd(img, k) for k in ks]
    fro_total = np.linalg.norm(img, "fro")
    errs = [np.linalg.norm(img - r, "fro") / fro_total for r in recons]
    energy = np.cumsum(s ** 2) / np.sum(s ** 2)

    fig = plt.figure(figsize=(15, 10.5), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 4, height_ratios=[1.4, 0.85], width_ratios=[1, 1, 1, 1.05],
    )

    # Top row: original + 3 reconstructions
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax0.set_title(f"Original\n{n}x{m} = {full_params:,} numbers",
                  fontsize=10, color=C_DARK, pad=6)
    ax0.set_xticks([]); ax0.set_yticks([])

    panel_axes = [fig.add_subplot(gs[0, i + 1]) for i in range(3)]
    for ax, k, r, e in zip(panel_axes, ks, recons, errs):
        ax.imshow(r, cmap="gray", vmin=0, vmax=1)
        kept = k * (n + m + 1)
        ratio = kept / full_params
        ax.set_title(
            f"Rank $k={k}$\n{kept:,} numbers ({ratio*100:.0f}% of full)\n"
            f"relative error = {e*100:.1f}%",
            fontsize=10, color=C_DARK, pad=6,
        )
        ax.set_xticks([]); ax.set_yticks([])

    # Bottom-left: singular values (log)
    ax_s = fig.add_subplot(gs[1, :2])
    ax_s.semilogy(np.arange(1, len(s) + 1), s, color=C_BLUE, lw=2)
    for k, c in zip(ks, [C_GREEN, C_PURPLE, C_AMBER]):
        ax_s.axvline(k, color=c, lw=1.4, ls="--", alpha=0.8)
        ax_s.text(k, s.max() * 0.7, f"$k={k}$", color=c,
                  fontsize=10, fontweight="bold", ha="left", va="center")
    ax_s.set_xlabel("index $i$", fontsize=10)
    ax_s.set_ylabel(r"$\sigma_i$ (log)", fontsize=10)
    ax_s.set_title("Singular-value decay: a few values dominate",
                   fontsize=11, color=C_DARK, pad=6)
    ax_s.tick_params(labelsize=9)

    # Bottom-right: cumulative energy
    ax_e = fig.add_subplot(gs[1, 2:])
    ax_e.plot(np.arange(1, len(energy) + 1), energy * 100,
              color=C_PURPLE, lw=2)
    ax_e.axhline(100, color=C_GRAY, lw=0.8, ls=":")
    for k, c in zip(ks, [C_GREEN, C_PURPLE, C_AMBER]):
        ax_e.axvline(k, color=c, lw=1.4, ls="--", alpha=0.8)
        ax_e.scatter([k], [energy[k - 1] * 100], color=c, zorder=5)
        ax_e.text(k + 1.5, energy[k - 1] * 100 - 4,
                  f"$k={k}$: {energy[k - 1]*100:.1f}%",
                  color=c, fontsize=9.5, fontweight="bold")
    ax_e.set_xlabel("rank $k$", fontsize=10)
    ax_e.set_ylabel(r"cumulative energy  $\sum_{i \leq k}\sigma_i^2 \,/\, \sum_i \sigma_i^2$  (%)",
                    fontsize=10)
    ax_e.set_title("Cumulative energy: rank 32 captures > 99%",
                   fontsize=11, color=C_DARK, pad=6)
    ax_e.set_ylim(0, 105)
    ax_e.tick_params(labelsize=9)

    fig.suptitle(
        "Low-rank approximation as image compression: "
        r"$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^{\!\top}$",
        fontsize=13, color=C_DARK,
    )
    save(fig, "fig1_lowrank_compression")


# ===========================================================================
# Figure 2: Pseudoinverse via SVD = least squares = projection
# ===========================================================================
def fig2_pseudoinverse_svd():
    rng = np.random.default_rng(2)
    n = 14
    x = np.linspace(0.0, 1.0, n)
    a_true, b_true = 1.7, 0.4
    y = a_true * x + b_true + 0.18 * rng.standard_normal(n)

    A = np.column_stack([x, np.ones_like(x)])         # n x 2
    A_pinv = np.linalg.pinv(A)
    coef = A_pinv @ y                                 # least-squares solution
    a_hat, b_hat = coef
    y_hat = A @ coef                                  # fitted values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Left panel: data + LS line
    ax = axes[0]
    ax.scatter(x, y, color=C_BLUE, s=46, zorder=4,
               edgecolor="white", linewidth=1.0, label="data $b_i$")
    xs = np.linspace(-0.05, 1.05, 100)
    ax.plot(xs, a_hat * xs + b_hat, color=C_GREEN, lw=2.4,
            label=f"LS fit  $\\hat a={a_hat:.2f}$, $\\hat b={b_hat:.2f}$")
    for xi, yi, yhi in zip(x, y, y_hat):
        ax.plot([xi, xi], [yi, yhi], color=C_AMBER, lw=1.2, alpha=0.8)
    # one residual labelled
    ax.text(x[3] + 0.02, 0.5 * (y[3] + y_hat[3]),
            "residual", color=C_AMBER, fontsize=9.5, fontweight="bold")
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$b$", fontsize=11)
    ax.set_title(r"Pseudoinverse solves least squares: $\hat\beta = A^{+}\,b$",
                 fontsize=11.5, color=C_DARK)
    ax.legend(fontsize=9.5, frameon=True, loc="upper left")
    ax.tick_params(labelsize=9)

    # --- Right panel: column-space projection picture (3D-ish 2D schematic)
    ax = axes[1]
    # Draw a shaded parallelogram = col(A) plane in R^n (2D schematic).
    plane = np.array([[-0.8, -0.5], [1.2, -0.7], [1.4, 0.7], [-0.6, 0.9]])
    ax.add_patch(Polygon(plane, closed=True, facecolor=C_BLUE,
                         alpha=0.10, edgecolor=C_BLUE, lw=1.4))
    ax.text(1.05, -0.85, r"col$(A)$", color=C_BLUE, fontsize=11,
            fontweight="bold")

    b_pt = np.array([0.4, 1.55])
    Pb_pt = np.array([0.4, 0.30])     # projection (visual)

    # Vectors
    arr_b = FancyArrowPatch((0, 0), b_pt, arrowstyle="-|>",
                            mutation_scale=14, color=C_PURPLE, lw=2.2)
    arr_p = FancyArrowPatch((0, 0), Pb_pt, arrowstyle="-|>",
                            mutation_scale=14, color=C_GREEN, lw=2.2)
    arr_r = FancyArrowPatch(Pb_pt, b_pt, arrowstyle="-|>",
                            mutation_scale=14, color=C_AMBER, lw=2.0)
    for a in (arr_b, arr_p, arr_r):
        ax.add_patch(a)

    # Right-angle marker at Pb
    sq = 0.07
    ax.add_patch(Polygon(
        [Pb_pt, Pb_pt + np.array([sq, 0]),
         Pb_pt + np.array([sq, sq]), Pb_pt + np.array([0, sq])],
        closed=True, facecolor="none", edgecolor=C_DARK, lw=1.0))

    ax.text(b_pt[0] + 0.05, b_pt[1] + 0.02, r"$b$",
            color=C_PURPLE, fontsize=13, fontweight="bold")
    ax.text(Pb_pt[0] + 0.06, Pb_pt[1] - 0.06,
            r"$A\hat\beta = AA^{+}b = \mathrm{Proj}_{\mathrm{col}(A)}(b)$",
            color=C_GREEN, fontsize=10.5, fontweight="bold")
    ax.text(Pb_pt[0] + 0.08, 0.5 * (Pb_pt[1] + b_pt[1]),
            r"$b - A\hat\beta$", color=C_AMBER, fontsize=11,
            fontweight="bold")

    # SVD recipe box
    ax.text(-1.55, 1.55,
            r"$A = U\Sigma V^{\!\top}$"
            "\n"
            r"$A^{+} = V\,\Sigma^{+}\,U^{\!\top}$"
            "\n"
            r"$\Sigma^{+}_{ii} = 1/\sigma_i$  if $\sigma_i>0$, else $0$",
            color=C_DARK, fontsize=10, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.45", fc="white",
                      ec=C_GRAY, lw=1))

    ax.set_xlim(-1.6, 2.0)
    ax.set_ylim(-1.1, 2.0)
    ax.set_aspect("equal")
    ax.axhline(0, color=C_GRAY, lw=0.5)
    ax.axvline(0, color=C_GRAY, lw=0.5)
    ax.tick_params(labelsize=8, colors=C_GRAY)
    ax.set_title(r"Geometry: $A^{+}b$ projects $b$ onto $\mathrm{col}(A)$",
                 fontsize=11.5, color=C_DARK)

    fig.suptitle("Pseudoinverse via SVD: least squares = orthogonal projection",
                 fontsize=13, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_pseudoinverse_svd")


# ===========================================================================
# Figure 3: Eckart-Young -- truncated SVD is THE best rank-k approximation
# ===========================================================================
def fig3_eckart_young():
    rng = np.random.default_rng(7)
    n, m = 60, 50
    # Construct A with controlled spectrum
    U_, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V_, _ = np.linalg.qr(rng.standard_normal((m, m)))
    s_true = 5.0 * np.exp(-np.arange(min(n, m)) / 8.0) + 0.05
    Sigma = np.zeros((n, m))
    np.fill_diagonal(Sigma, s_true)
    A = U_ @ Sigma @ V_.T
    A_fro = np.linalg.norm(A, "fro")

    ks = np.arange(1, min(n, m) + 1)
    # Truncated-SVD error
    err_svd = np.array([
        np.sqrt(np.sum(s_true[k:] ** 2)) for k in ks
    ]) / A_fro

    # Random rank-k baseline: pick random orthonormal U_k, V_k, project A
    err_rand_runs = []
    for trial in range(8):
        errs = []
        for k in ks:
            Q1, _ = np.linalg.qr(rng.standard_normal((n, k)))
            Q2, _ = np.linalg.qr(rng.standard_normal((m, k)))
            # Best B with given column/row spaces is Q1 (Q1^T A Q2) Q2^T
            B = Q1 @ (Q1.T @ A @ Q2) @ Q2.T
            errs.append(np.linalg.norm(A - B, "fro") / A_fro)
        err_rand_runs.append(errs)
    err_rand = np.array(err_rand_runs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

    # Left: spectrum + truncation marker
    ax = axes[0]
    ax.semilogy(np.arange(1, len(s_true) + 1), s_true,
                color=C_BLUE, lw=2, marker="o", markersize=4)
    k_show = 8
    ax.axvline(k_show, color=C_GREEN, ls="--", lw=1.4)
    ax.fill_between(np.arange(k_show + 1, len(s_true) + 1),
                    1e-3, s_true[k_show:],
                    color=C_AMBER, alpha=0.25,
                    label=r"truncated tail $\sigma_{k+1}, \sigma_{k+2}, \ldots$")
    ax.text(k_show + 0.3, s_true.max() * 0.6,
            f"keep top $k={k_show}$", color=C_GREEN,
            fontsize=10.5, fontweight="bold")
    ax.set_xlabel("singular-value index $i$", fontsize=10.5)
    ax.set_ylabel(r"$\sigma_i$ (log)", fontsize=10.5)
    ax.set_title("Eckart-Young: discard the tail, keep the head",
                 fontsize=11.5, color=C_DARK)
    ax.legend(fontsize=9.5, frameon=True)
    ax.tick_params(labelsize=9)

    # Right: error vs k -- SVD vs random baseline
    ax = axes[1]
    # Plot random trials individually then mean
    for r in err_rand:
        ax.plot(ks, r, color=C_AMBER, lw=0.8, alpha=0.35)
    ax.plot(ks, err_rand.mean(axis=0), color=C_AMBER, lw=2.2,
            label="random rank-$k$ projection (mean)")
    ax.plot(ks, err_svd, color=C_GREEN, lw=2.6,
            label=r"truncated SVD = $\sqrt{\sum_{i>k}\sigma_i^2}\,/\,\|A\|_F$")
    ax.set_xlabel("rank $k$", fontsize=10.5)
    ax.set_ylabel(r"relative error  $\|A - A_k\|_F\,/\,\|A\|_F$",
                  fontsize=10.5)
    ax.set_title("Truncated SVD is provably optimal (Eckart-Young, 1936)",
                 fontsize=11.5, color=C_DARK)
    ax.set_ylim(0, max(err_rand.max(), 1.05))
    ax.legend(fontsize=9.5, frameon=True)
    ax.tick_params(labelsize=9)

    fig.suptitle(
        r"Best rank-$k$ approximation: $A_k = \arg\min_{\mathrm{rank}(B)\leq k}\|A-B\|_F$",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig3_eckart_young")


# ===========================================================================
# Figure 4: Recommender systems -- matrix factorization
# ===========================================================================
def fig4_recommender_mf():
    rng = np.random.default_rng(11)
    n_users, n_items, r_true = 18, 22, 3

    # True latent factors (positive-ish so the heatmap reads as 'taste')
    U_true = rng.uniform(0.0, 1.2, size=(n_users, r_true))
    V_true = rng.uniform(0.0, 1.2, size=(n_items, r_true))
    R_full = U_true @ V_true.T
    # Rescale to a 1..5 rating range
    R_full = 1 + 4 * (R_full - R_full.min()) / (R_full.max() - R_full.min())

    # Mask: only ~35% of entries observed
    mask = rng.random(R_full.shape) < 0.35
    R_obs = np.where(mask, R_full, np.nan)

    # Fit U, V by alternating least squares to observed entries.
    # Initialize with mean-imputed SVD for a much better starting point.
    r_fit = 3
    R_init = np.where(mask, R_full, np.nanmean(np.where(mask, R_full, np.nan)))
    Us, ss, Vts = np.linalg.svd(R_init, full_matrices=False)
    Uf = Us[:, :r_fit] * np.sqrt(ss[:r_fit])
    Vf = (Vts[:r_fit, :].T) * np.sqrt(ss[:r_fit])
    lam = 0.02
    for _ in range(200):
        # update Uf
        for i in range(n_users):
            obs = mask[i]
            if obs.any():
                Vi = Vf[obs]
                ri = R_full[i, obs]
                A = Vi.T @ Vi + lam * np.eye(r_fit)
                Uf[i] = np.linalg.solve(A, Vi.T @ ri)
        # update Vf
        for j in range(n_items):
            obs = mask[:, j]
            if obs.any():
                Uj = Uf[obs]
                rj = R_full[obs, j]
                A = Uj.T @ Uj + lam * np.eye(r_fit)
                Vf[j] = np.linalg.solve(A, Uj.T @ rj)

    R_hat = Uf @ Vf.T
    R_hat = np.clip(R_hat, 1, 5)

    fig = plt.figure(figsize=(14.5, 5.4))
    gs = fig.add_gridspec(1, 5, width_ratios=[1.4, 0.5, 0.6, 0.5, 1.4],
                          wspace=0.35)

    # Left: observed matrix R (sparse)
    ax = fig.add_subplot(gs[0])
    R_show = np.where(mask, R_full, np.nan)
    im = ax.imshow(R_show, cmap="viridis", vmin=1, vmax=5, aspect="auto")
    ax.set_title(f"Observed ratings $R$\n"
                 f"{n_users} users x {n_items} items, "
                 f"{int(mask.sum())} / {n_users*n_items} known "
                 f"({mask.mean()*100:.0f}%)",
                 fontsize=10.5, color=C_DARK, pad=6)
    ax.set_xlabel("items"); ax.set_ylabel("users")
    ax.tick_params(labelsize=8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("rating", fontsize=9)

    # Middle-left: U factor
    ax = fig.add_subplot(gs[1])
    ax.imshow(Uf, cmap="coolwarm", aspect="auto",
              vmin=-np.abs(Uf).max(), vmax=np.abs(Uf).max())
    ax.set_title(f"$U$\n{n_users} x {r_fit}", fontsize=10, color=C_DARK)
    ax.set_xticks(range(r_fit))
    ax.set_xticklabels([f"f{i+1}" for i in range(r_fit)], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("latent factors", fontsize=9)

    # Middle: x sign
    ax = fig.add_subplot(gs[2]); ax.axis("off")
    ax.text(0.5, 0.5, r"$\approx$" "\n" r"$U V^{\!\top}$",
            ha="center", va="center", fontsize=22, color=C_DARK,
            fontweight="bold")

    # Middle-right: V^T factor
    ax = fig.add_subplot(gs[3])
    ax.imshow(Vf.T, cmap="coolwarm", aspect="auto",
              vmin=-np.abs(Vf).max(), vmax=np.abs(Vf).max())
    ax.set_title(f"$V^{{\\top}}$\n{r_fit} x {n_items}",
                 fontsize=10, color=C_DARK)
    ax.set_yticks(range(r_fit))
    ax.set_yticklabels([f"f{i+1}" for i in range(r_fit)], fontsize=8)
    ax.set_xticks([])
    ax.set_ylabel("factors", fontsize=9)

    # Right: reconstructed full matrix (predictions)
    ax = fig.add_subplot(gs[4])
    im = ax.imshow(R_hat, cmap="viridis", vmin=1, vmax=5, aspect="auto")
    rmse_obs = np.sqrt(np.mean((R_hat[mask] - R_full[mask]) ** 2))
    rmse_hid = np.sqrt(np.mean((R_hat[~mask] - R_full[~mask]) ** 2))
    ax.set_title(
        f"Predicted $\\hat R = U V^{{\\top}}$\n"
        f"RMSE: observed={rmse_obs:.2f},  held-out={rmse_hid:.2f}",
        fontsize=10.5, color=C_DARK, pad=6,
    )
    ax.set_xlabel("items"); ax.set_ylabel("users")
    ax.tick_params(labelsize=8)
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03).set_label(
        "rating", fontsize=9)

    fig.suptitle(
        "Recommender systems: a sparse rating matrix is well approximated by a "
        f"rank-{r_fit} factorisation",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig4_recommender_mf")


# ===========================================================================
# Figure 5: LoRA -- low-rank adaptation of a frozen weight
# ===========================================================================
def fig5_lora_connection():
    fig = plt.figure(figsize=(14.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.0], wspace=0.32)

    # ---- Panel 1: schematic of W' = W + B A
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(ax, x, y, w, h, color, label, sublabel=None,
            face_alpha=0.18, fontweight="bold"):
        ax.add_patch(Polygon(
            [(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
            closed=True, facecolor=color, alpha=face_alpha,
            edgecolor=color, lw=1.6,
        ))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=color, fontsize=12, fontweight=fontweight)
        if sublabel:
            ax.text(x + w / 2, y - 0.32, sublabel, ha="center", va="top",
                    color=C_DARK, fontsize=9.5)

    # Frozen W
    box(ax, 0.6, 1.6, 2.6, 2.8, C_BLUE,
        "$W$\n(frozen)", sublabel="$d \\times k$  -- not updated")
    # Plus sign
    ax.text(3.7, 3.0, "+", fontsize=26, color=C_DARK,
            fontweight="bold", ha="center", va="center")
    # B
    box(ax, 4.2, 1.6, 0.9, 2.8, C_GREEN,
        "$B$", sublabel="$d \\times r$")
    # A
    box(ax, 5.4, 2.4, 2.4, 1.2, C_GREEN,
        "$A$", sublabel="$r \\times k$")
    # Equals delta W
    ax.text(8.05, 3.0, "=", fontsize=22, color=C_DARK,
            fontweight="bold", ha="center", va="center")
    box(ax, 8.4, 1.6, 1.4, 2.8, C_PURPLE,
        r"$\Delta W$", sublabel=r"low-rank ($\leq r$)")

    ax.set_title(
        r"LoRA:  $W' = W + \Delta W = W + B A$,   $r \ll \min(d, k)$",
        fontsize=12, color=C_DARK, pad=8,
    )

    # ---- Panel 2: parameter count comparison (bar chart)
    ax = fig.add_subplot(gs[1])
    d, k = 4096, 4096           # typical LLM linear layer
    ranks = [1, 2, 4, 8, 16, 32, 64]
    full = d * k
    lora = [d * r + r * k for r in ranks]
    bar_x = np.arange(len(ranks))
    bars = ax.bar(bar_x, [v / 1e6 for v in lora], color=C_GREEN, alpha=0.85,
                  edgecolor=C_DARK, lw=0.6)
    ax.axhline(full / 1e6, color=C_BLUE, lw=2, ls="--",
               label=f"full fine-tune: {full/1e6:.1f} M params")
    ax.set_xticks(bar_x)
    ax.set_xticklabels([f"r={r}" for r in ranks], fontsize=9)
    ax.set_ylabel("trainable parameters (millions)", fontsize=10)
    ax.set_title(f"Parameter cost: one $W \\in \\mathbb{{R}}^{{{d}\\times{k}}}$ layer",
                 fontsize=11.5, color=C_DARK)
    ax.legend(fontsize=9.5, frameon=True, loc="upper left")
    # annotate the savings on top of one bar
    r_show = ranks.index(8)
    saving = full / lora[r_show]
    ax.annotate(
        f"r=8:\n{lora[r_show]/1e6:.2f} M\n"
        f"({saving:.0f}x fewer)",
        xy=(bar_x[r_show], lora[r_show] / 1e6),
        xytext=(bar_x[r_show] + 0.6, full / 1e6 * 0.45),
        arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.0),
        fontsize=9.5, color=C_DARK,
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec=C_GRAY, lw=0.8),
    )
    ax.tick_params(labelsize=9)
    # log y so small bars are still visible
    ax.set_yscale("log")
    ax.set_ylim(min(lora) / 1e6 * 0.5, full / 1e6 * 2.5)

    # ---- Panel 3: empirical low-rank spectrum of Delta W
    ax = fig.add_subplot(gs[2])
    rng = np.random.default_rng(3)
    d2 = 200
    # Synthesise a Delta W that is approximately low-rank (rank ~ 8)
    r_eff = 8
    B = rng.standard_normal((d2, r_eff)) * 1.0
    A = rng.standard_normal((r_eff, d2)) * 1.0
    DW = B @ A + 0.04 * rng.standard_normal((d2, d2))
    s = np.linalg.svd(DW, compute_uv=False)
    ax.semilogy(np.arange(1, len(s) + 1), s, color=C_PURPLE, lw=2)
    ax.axvline(r_eff, color=C_GREEN, ls="--", lw=1.4)
    ax.fill_between(np.arange(r_eff + 1, len(s) + 1),
                    1e-3, s[r_eff:],
                    color=C_AMBER, alpha=0.25,
                    label="negligible tail")
    ax.text(r_eff + 3, s.max() * 0.5,
            f"effective rank $\\approx {r_eff}$",
            color=C_GREEN, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("singular-value index", fontsize=10)
    ax.set_ylabel(r"$\sigma_i(\Delta W)$ (log)", fontsize=10)
    ax.set_title(r"Why LoRA works: $\Delta W$ has low intrinsic rank",
                 fontsize=11.5, color=C_DARK)
    ax.legend(fontsize=9.5, frameon=True)
    ax.tick_params(labelsize=9)

    fig.suptitle(
        "From Eckart-Young to LoRA: low-rank approximation enables "
        "parameter-efficient fine-tuning",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig5_lora_connection")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Generating figures for low-rank approximation & pseudoinverse...")
    fig1_lowrank_compression()
    print("  fig1_lowrank_compression  done")
    fig2_pseudoinverse_svd()
    print("  fig2_pseudoinverse_svd    done")
    fig3_eckart_young()
    print("  fig3_eckart_young         done")
    fig4_recommender_mf()
    print("  fig4_recommender_mf       done")
    fig5_lora_connection()
    print("  fig5_lora_connection      done")
    print(f"\nSaved to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
