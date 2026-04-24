"""Figure generation for the standalone Proximal Operator article.

Five figures, each teaching a single concept and saved into both the EN and
ZH article asset folders so markdown references stay in sync.

Figures:
    fig1_prox_definition    The prox map of a non-smooth function visualised
                            as a balance between f(x) and the quadratic
                            anchor (1/2 lambda) ||x - v||^2. Includes the
                            Moreau envelope as a smoothed version of f.
    fig2_soft_threshold     Soft-thresholding (the prox of the L1 norm)
                            against the identity, with the dead zone
                            highlighted, plus a 1-D denoising example
                            showing how it shrinks small coefficients to
                            exactly zero.
    fig3_ista_iterations    ISTA iterates on a 2-D LASSO problem: contour
                            of the smooth term, level set of the L1 ball
                            implicit in the regulariser, and the iterate
                            trajectory snapping onto an axis (sparsity).
    fig4_fista_acceleration ISTA vs FISTA convergence on a LASSO problem,
                            with the O(1/k) and O(1/k^2) reference curves.
    fig5_lasso_path         LASSO solution path (coefficients vs lambda)
                            on a synthetic problem with planted sparse
                            ground truth, showing variable selection.

Usage:
    python3 scripts/figures/standalone/proximal-operator.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # noqa: F401  (registers style)
from matplotlib.patches import FancyArrowPatch

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

COLOR_BLUE = COLORS["primary"]     # primary signal / data
COLOR_PURPLE = COLORS["accent"]   # secondary / model output
COLOR_GREEN = COLORS["success"]    # accelerated / good
COLOR_AMBER = COLORS["warning"]    # accent / threshold / warning
COLOR_GREY = COLORS["muted"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "proximal-operator"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "standalone" / "proximal-operator"


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in (EN_DIR, ZH_DIR):
        fig.savefig(out_dir / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 - prox definition + Moreau envelope
# ---------------------------------------------------------------------------

def fig1_prox_definition() -> None:
    """Show the prox of f(x)=|x| at v with varying lambda, and the Moreau envelope."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Left: balance interpretation -------------------------------------
    ax = axes[0]
    x = np.linspace(-2.5, 2.5, 400)
    f = np.abs(x)
    v = 1.4
    lam = 0.6
    quad = (1.0 / (2.0 * lam)) * (x - v) ** 2
    obj = f + quad

    ax.plot(x, f, color=COLOR_BLUE, lw=2.2, label=r"$f(x)=|x|$")
    ax.plot(x, quad, color=COLOR_GREY, lw=1.6, ls="--",
            label=r"$\frac{1}{2\lambda}\|x-v\|^{2}$")
    ax.plot(x, obj, color=COLOR_PURPLE, lw=2.2,
            label=r"$f(x)+\frac{1}{2\lambda}\|x-v\|^{2}$")

    # prox of L1: soft-threshold of v at lambda
    prox = np.sign(v) * max(abs(v) - lam, 0.0)
    ax.scatter([v], [0], color=COLOR_BLUE, zorder=5, s=60)
    ax.annotate("v (anchor)", xy=(v, 0), xytext=(v + 0.05, 0.55),
                fontsize=10, color=COLOR_BLUE)
    ax.scatter([prox], [np.abs(prox) + (1.0 / (2.0 * lam)) * (prox - v) ** 2],
               color=COLOR_AMBER, zorder=5, s=80, edgecolor="white", linewidth=1.4)
    ax.annotate(r"$\mathrm{prox}_{\lambda f}(v)$",
                xy=(prox, np.abs(prox) + (1.0 / (2.0 * lam)) * (prox - v) ** 2),
                xytext=(prox - 1.7, 1.6),
                fontsize=11, color=COLOR_AMBER,
                arrowprops=dict(arrowstyle="->", color=COLOR_AMBER, lw=1.2))

    ax.set_ylim(-0.15, 3.2)
    ax.set_xlim(-2.5, 2.5)
    ax.set_title("Proximal Operator as a Balance", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # --- Right: Moreau envelope vs f --------------------------------------
    ax = axes[1]
    x = np.linspace(-3.0, 3.0, 400)
    f = np.abs(x)
    # Moreau envelope of |.| at parameter lam: Huber function
    lam = 0.5

    def moreau_l1(z, lam):
        out = np.where(np.abs(z) <= lam,
                       (z ** 2) / (2.0 * lam),
                       np.abs(z) - lam / 2.0)
        return out

    env_05 = moreau_l1(x, 0.5)
    env_15 = moreau_l1(x, 1.5)

    ax.plot(x, f, color=COLOR_BLUE, lw=2.2, label=r"$f(x)=|x|$  (non-smooth)")
    ax.plot(x, env_05, color=COLOR_PURPLE, lw=2.0,
            label=r"Moreau env. $\widehat{f}_{0.5}$")
    ax.plot(x, env_15, color=COLOR_GREEN, lw=2.0,
            label=r"Moreau env. $\widehat{f}_{1.5}$  (smoother)")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.2, 3.2)
    ax.set_title("Moreau Envelope: Smoothing the Non-Smooth", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.legend(loc="upper center", fontsize=9, framealpha=0.95)

    fig.suptitle("Figure 1  Proximal Operator and Moreau Envelope",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_prox_definition.png")


# ---------------------------------------------------------------------------
# Figure 2 - soft thresholding
# ---------------------------------------------------------------------------

def fig2_soft_threshold() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Left: shape of soft threshold ------------------------------------
    ax = axes[0]
    v = np.linspace(-3, 3, 400)
    lam = 1.0
    soft = np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)

    # dead zone shading
    ax.axvspan(-lam, lam, color=COLOR_AMBER, alpha=0.12, label="dead zone")
    ax.plot(v, v, color=COLOR_GREY, lw=1.5, ls="--", label="identity")
    ax.plot(v, soft, color=COLOR_BLUE, lw=2.4,
            label=r"$\mathrm{prox}_{\lambda \|\cdot\|_{1}}(v)$")
    ax.axvline(lam, color=COLOR_AMBER, lw=1.0, ls=":")
    ax.axvline(-lam, color=COLOR_AMBER, lw=1.0, ls=":")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title(r"Soft-Threshold ($\lambda=1$)", fontsize=12)
    ax.set_xlabel("v")
    ax.set_ylabel("output")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    # --- Right: 1-D denoising example -------------------------------------
    ax = axes[1]
    rng = np.random.default_rng(0)
    n = 80
    truth = np.zeros(n)
    spikes = [10, 25, 40, 55, 70]
    truth[spikes] = [2.5, -1.8, 3.0, -2.2, 1.5]
    noisy = truth + 0.45 * rng.standard_normal(n)
    denoised = np.sign(noisy) * np.maximum(np.abs(noisy) - 0.8, 0.0)

    idx = np.arange(n)
    markerline, stemlines, baseline = ax.stem(
        idx, noisy, linefmt="-", markerfmt="o",
        basefmt=" ", label="noisy",
    )
    plt.setp(stemlines, color=COLOR_GREY, lw=1.0, alpha=0.8)
    plt.setp(markerline, color=COLOR_GREY, markersize=4)
    ax.scatter(idx, denoised, color=COLOR_BLUE, s=22, zorder=4,
               label="soft-thresholded")
    ax.scatter(spikes, truth[spikes], color=COLOR_AMBER, marker="x",
               s=70, zorder=5, label="true spikes")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_title("1-D Sparse Denoising via Soft-Threshold", fontsize=12)
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.suptitle("Figure 2  Soft-Thresholding: the Prox of the L1 Norm",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_soft_threshold.png")


# ---------------------------------------------------------------------------
# Figure 3 - ISTA iterations
# ---------------------------------------------------------------------------

def _lasso_data(n: int = 80, d: int = 2, seed: int = 1):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, d))
    x_true = np.array([1.6, 0.0]) if d == 2 else rng.standard_normal(d)
    y = A @ x_true + 0.25 * rng.standard_normal(n)
    return A, y, x_true


def _ista(A, y, lam, lr, n_iter, x0=None):
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    x = x0.copy()
    history = [x.copy()]
    for _ in range(n_iter):
        grad = A.T @ (A @ x - y)
        z = x - lr * grad
        x = np.sign(z) * np.maximum(np.abs(z) - lr * lam, 0.0)
        history.append(x.copy())
    return np.array(history)


def fig3_ista_iterations() -> None:
    A, y, x_true = _lasso_data(n=120, d=2, seed=2)
    lam = 6.0
    L = np.linalg.norm(A, 2) ** 2
    lr = 1.0 / L

    history = _ista(A, y, lam, lr, n_iter=40, x0=np.array([2.5, 2.2]))

    fig, ax = plt.subplots(figsize=(7.8, 6.2))

    # contours of the smooth part 0.5 ||Ax - y||^2 + lam * |x|_1
    grid = np.linspace(-1.0, 3.0, 220)
    g1, g2 = np.meshgrid(grid, grid)
    obj = np.zeros_like(g1)
    for i in range(grid.size):
        for j in range(grid.size):
            x = np.array([g1[i, j], g2[i, j]])
            obj[i, j] = 0.5 * np.sum((A @ x - y) ** 2) + lam * np.sum(np.abs(x))

    cs = ax.contour(g1, g2, obj, levels=18, colors=COLOR_GREY,
                    linewidths=0.7, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")

    # axes (sparsity targets)
    ax.axhline(0, color=COLOR_AMBER, lw=1.4, alpha=0.7,
               label="sparse axes (x_i = 0)")
    ax.axvline(0, color=COLOR_AMBER, lw=1.4, alpha=0.7)

    # iterate trajectory
    ax.plot(history[:, 0], history[:, 1], "-o", color=COLOR_BLUE,
            ms=4.5, lw=1.6, label="ISTA iterates")
    ax.scatter([history[0, 0]], [history[0, 1]], color=COLOR_PURPLE,
               s=120, zorder=5, marker="*", label="start")
    ax.scatter([history[-1, 0]], [history[-1, 1]], color=COLOR_GREEN,
               s=120, zorder=5, marker="X", label="solution (sparse)")

    ax.set_xlim(-0.4, 2.9)
    ax.set_ylim(-0.4, 2.5)
    ax.set_xlabel(r"$x_{1}$")
    ax.set_ylabel(r"$x_{2}$")
    ax.set_title("Figure 3  ISTA on a 2-D LASSO: Iterates Snap to Sparse Axis",
                 fontsize=12)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    _save(fig, "fig3_ista_iterations.png")


# ---------------------------------------------------------------------------
# Figure 4 - FISTA acceleration
# ---------------------------------------------------------------------------

def _fista(A, y, lam, lr, n_iter, x0=None):
    if x0 is None:
        x0 = np.zeros(A.shape[1])
    x = x0.copy()
    z = x0.copy()
    t = 1.0
    history = [x.copy()]
    for _ in range(n_iter):
        grad = A.T @ (A @ z - y)
        u = z - lr * grad
        x_new = np.sign(u) * np.maximum(np.abs(u) - lr * lam, 0.0)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        z = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new
        history.append(x.copy())
    return np.array(history)


def fig4_fista_acceleration() -> None:
    rng = np.random.default_rng(7)
    n, d = 200, 60
    A = rng.standard_normal((n, d))
    x_true = np.zeros(d)
    x_true[:8] = rng.standard_normal(8) * 1.5
    y = A @ x_true + 0.1 * rng.standard_normal(n)
    lam = 2.0
    L = np.linalg.norm(A, 2) ** 2
    lr = 1.0 / L

    n_iter = 200

    def loss(xk):
        return 0.5 * np.sum((A @ xk - y) ** 2) + lam * np.sum(np.abs(xk))

    hist_ista = _ista(A, y, lam, lr, n_iter=n_iter)
    hist_fista = _fista(A, y, lam, lr, n_iter=n_iter)

    f_star = min(loss(hist_ista[-1]), loss(hist_fista[-1])) - 1e-9
    err_ista = np.array([loss(x) - f_star for x in hist_ista])
    err_fista = np.array([loss(x) - f_star for x in hist_fista])

    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    k = np.arange(1, n_iter + 1)
    ax.loglog(k, err_ista[1:], color=COLOR_BLUE, lw=2.2, label="ISTA")
    ax.loglog(k, err_fista[1:], color=COLOR_GREEN, lw=2.2, label="FISTA")

    # reference rates
    c1 = err_ista[1] * 1.0
    ax.loglog(k, c1 / k, color=COLOR_BLUE, ls="--", lw=1.0,
              alpha=0.7, label=r"$\mathcal{O}(1/k)$")
    c2 = err_fista[1] * 1.0
    ax.loglog(k, c2 / (k ** 2), color=COLOR_GREEN, ls="--", lw=1.0,
              alpha=0.7, label=r"$\mathcal{O}(1/k^{2})$")

    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$F(x_{k}) - F^{\star}$")
    ax.set_title("Figure 4  FISTA Acceleration vs ISTA on LASSO",
                 fontsize=12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.4)

    fig.tight_layout()
    _save(fig, "fig4_fista_acceleration.png")


# ---------------------------------------------------------------------------
# Figure 5 - LASSO solution path
# ---------------------------------------------------------------------------

def fig5_lasso_path() -> None:
    rng = np.random.default_rng(11)
    n, d = 150, 12
    A = rng.standard_normal((n, d))
    x_true = np.zeros(d)
    active = [0, 2, 5, 8]
    x_true[active] = [2.0, -1.6, 1.2, -0.9]
    y = A @ x_true + 0.5 * rng.standard_normal(n)

    lams = np.logspace(-2, 1.4, 40)
    L = np.linalg.norm(A, 2) ** 2
    lr = 1.0 / L

    coefs = []
    for lam in lams:
        x = np.zeros(d)
        for _ in range(800):
            grad = A.T @ (A @ x - y)
            z = x - lr * grad
            x = np.sign(z) * np.maximum(np.abs(z) - lr * lam, 0.0)
        coefs.append(x)
    coefs = np.array(coefs)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    cmap = plt.get_cmap("tab20")
    for j in range(d):
        is_active = j in active
        ax.plot(lams, coefs[:, j],
                color=COLOR_PURPLE if is_active else COLOR_GREY,
                lw=2.0 if is_active else 1.0,
                alpha=0.95 if is_active else 0.7,
                label=f"true non-zero (idx {j})" if is_active else None)

    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel(r"regularisation $\lambda$  (log scale)")
    ax.set_ylabel("coefficient value")
    ax.set_title("Figure 5  LASSO Solution Path: Variable Selection by Increasing $\\lambda$",
                 fontsize=12)

    # custom legend (compact)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=COLOR_PURPLE, lw=2.0,
               label="true non-zero coefficients"),
        Line2D([0], [0], color=COLOR_GREY, lw=1.0,
               label="true zero coefficients"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              fontsize=10, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.35)

    fig.tight_layout()
    _save(fig, "fig5_lasso_path.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_prox_definition()
    fig2_soft_threshold()
    fig3_ista_iterations()
    fig4_fista_acceleration()
    fig5_lasso_path()
    print(f"Saved 5 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
