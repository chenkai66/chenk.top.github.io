"""
Figure generation for the standalone post:
    "Optimizer Evolution: From Gradient Descent to Adam"
    "优化算法的演进：从梯度下降到 Adam"

Generates 7 didactic figures used by both EN and ZH versions of the article.
Each figure isolates one concept with consistent styling.

Figures:
    fig1_gd_sgd_momentum_contour    GD / SGD / Momentum trajectories on
                                    an ill-conditioned quadratic bowl.
    fig2_nesterov_lookahead         NAG: lookahead gradient evaluated at
                                    the momentum-shifted point.
    fig3_adagrad_per_coord_lr       AdaGrad: per-coordinate effective LR
                                    on an anisotropic loss.
    fig4_rmsprop_moving_average     RMSProp: exponential moving average
                                    of g^2 (vs cumulative AdaGrad).
    fig5_adam_combined              Adam: m_t (momentum) + v_t (RMSProp)
                                    -> bias-corrected adaptive update.
    fig6_adamw_vs_adam              AdamW: decoupled weight decay vs
                                    L2-regularized Adam.
    fig7_modern_optimizers          Lion / Sophia / Schedule-Free survey:
                                    update rule signatures and trade-offs.

Usage:
    python3 scripts/figures/standalone/optimizer-evolution.py

Output:
    Writes PNGs (DPI=150) to BOTH article asset folders so markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]      # primary
C_PURPLE = COLORS["accent"]    # secondary
C_GREEN = COLORS["success"]     # accent / good
C_AMBER = COLORS["warning"]     # warning / highlight
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_BG = "#f8fafc"
C_LIGHT_BLUE = "#dbeafe"
C_LIGHT_PURPLE = "#ede9fe"
C_LIGHT_GREEN = "#d1fae5"
C_LIGHT_AMBER = "#fef3c7"
C_LIGHT_GRAY = COLORS["grid"]
C_LIGHT_RED = "#fee2e2"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "standalone"
    / "optimizer-evolution-gd-to-adam"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "standalone"
    / "优化算法的演进-从梯度下降到adam"
)

EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both asset folders at consistent DPI."""
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Shared loss landscape: ill-conditioned quadratic bowl
#   f(x, y) = 0.5 * (a x^2 + b y^2),  a >> b -> long narrow valley
# ---------------------------------------------------------------------------
A_COND, B_COND = 12.0, 1.0


def f_quad(x, y):
    return 0.5 * (A_COND * x**2 + B_COND * y**2)


def grad_quad(x, y):
    return np.array([A_COND * x, B_COND * y])


def gd_run(x0, lr, n=40):
    p = np.array(x0, dtype=float)
    pts = [p.copy()]
    for _ in range(n):
        p = p - lr * grad_quad(*p)
        pts.append(p.copy())
    return np.array(pts)


def sgd_run(x0, lr, n=40, noise=1.5, seed=0):
    rng = np.random.default_rng(seed)
    p = np.array(x0, dtype=float)
    pts = [p.copy()]
    for _ in range(n):
        g = grad_quad(*p) + rng.normal(0, noise, size=2)
        p = p - lr * g
        pts.append(p.copy())
    return np.array(pts)


def momentum_run(x0, lr, gamma=0.9, n=40):
    p = np.array(x0, dtype=float)
    v = np.zeros(2)
    pts = [p.copy()]
    for _ in range(n):
        v = gamma * v + lr * grad_quad(*p)
        p = p - v
        pts.append(p.copy())
    return np.array(pts)


def nag_run(x0, lr, gamma=0.9, n=40):
    p = np.array(x0, dtype=float)
    v = np.zeros(2)
    pts = [p.copy()]
    for _ in range(n):
        look = p - gamma * v
        v = gamma * v + lr * grad_quad(*look)
        p = p - v
        pts.append(p.copy())
    return np.array(pts)


def _draw_contour(ax, levels=None, extent=(-1.2, 1.2, -1.6, 1.6)):
    xs = np.linspace(extent[0], extent[1], 220)
    ys = np.linspace(extent[2], extent[3], 220)
    X, Y = np.meshgrid(xs, ys)
    Z = f_quad(X, Y)
    if levels is None:
        levels = np.linspace(0.05, Z.max(), 14)
    ax.contour(X, Y, Z, levels=levels, colors=C_GRAY, alpha=0.45,
               linewidths=0.8)
    ax.contourf(X, Y, Z, levels=levels, cmap="Blues", alpha=0.18)
    ax.plot(0, 0, marker="*", color=C_AMBER, markersize=16,
            markeredgecolor=C_DARK, markeredgewidth=0.8, zorder=5)
    ax.set_aspect("equal")


def _plot_traj(ax, pts, color, label, marker="o"):
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=1.7, alpha=0.85,
            zorder=4, label=label)
    ax.scatter(pts[:, 0], pts[:, 1], s=14, color=color, zorder=4,
               edgecolor="white", linewidths=0.5)
    ax.scatter(pts[0, 0], pts[0, 1], s=80, color=color, marker=marker,
               edgecolor=C_DARK, linewidths=1.2, zorder=6)


# ---------------------------------------------------------------------------
# Figure 1: GD vs SGD vs Momentum on an ill-conditioned quadratic
# ---------------------------------------------------------------------------
def fig1_gd_sgd_momentum_contour() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0))
    start = (-1.0, 1.4)
    titles = [
        "Gradient Descent (GD)",
        "SGD (mini-batch noise)",
        "Momentum ($\\gamma=0.9$)",
    ]
    runs = [
        gd_run(start, lr=0.14, n=35),
        sgd_run(start, lr=0.10, n=35, noise=1.2, seed=1),
        momentum_run(start, lr=0.04, gamma=0.9, n=35),
    ]
    colors = [C_BLUE, C_AMBER, C_GREEN]
    notes = [
        "zig-zags across the\nsteep direction",
        "noisy zig-zag —\nhelps escape sharp minima",
        "accumulates velocity\nalong the long valley",
    ]
    for ax, title, pts, color, note in zip(axes, titles, runs, colors, notes):
        _draw_contour(ax)
        _plot_traj(ax, pts, color, title.split()[0])
        ax.set_title(title, fontsize=12, fontweight="bold", color=C_DARK)
        ax.text(-1.15, -1.5, note, fontsize=9.5, color=C_DARK,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec=C_GRAY, alpha=0.9))
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        "Trajectories on an ill-conditioned quadratic "
        r"$f(x,y) = \frac{1}{2}(12 x^2 + y^2)$",
        fontsize=13, fontweight="bold", color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig1_gd_sgd_momentum_contour")


# ---------------------------------------------------------------------------
# Figure 2: Nesterov lookahead vs classical momentum
# ---------------------------------------------------------------------------
def fig2_nesterov_lookahead() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # --- Left: trajectory comparison --------------------------------------
    ax = axes[0]
    start = (-1.0, 1.4)
    mpts = momentum_run(start, lr=0.045, gamma=0.92, n=40)
    npts = nag_run(start, lr=0.045, gamma=0.92, n=40)
    _draw_contour(ax)
    _plot_traj(ax, mpts, C_AMBER, "Momentum")
    _plot_traj(ax, npts, C_PURPLE, "Nesterov (NAG)")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.set_title("NAG overshoots less around the minimum",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Right: schematic of the lookahead step ---------------------------
    ax = axes[1]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    p = np.array([2.0, 6.5])
    v_prev = np.array([4.0, -1.2])
    look = p + v_prev * 0.95
    grad_at_p = np.array([0.6, -2.4])
    grad_at_look = np.array([0.0, -2.8])

    # current point
    ax.scatter(*p, s=140, color=C_BLUE, zorder=5, edgecolor=C_DARK)
    ax.text(p[0] - 0.25, p[1] + 0.30, r"$\theta_t$", fontsize=14,
            ha="right", color=C_DARK, fontweight="bold")

    # lookahead point
    ax.scatter(*look, s=140, color=C_PURPLE, zorder=5, edgecolor=C_DARK)
    ax.text(look[0] + 0.25, look[1] + 0.30, r"$\theta_t - \gamma v_{t-1}$",
            fontsize=12, color=C_PURPLE, fontweight="bold")

    # momentum step (top arrow)
    ax.add_patch(FancyArrowPatch(p, look, arrowstyle="->", lw=2.4,
                                 color=C_AMBER, mutation_scale=20))
    ax.text((p[0] + look[0]) / 2, (p[1] + look[1]) / 2 + 0.55,
            r"momentum step  $\gamma v_{t-1}$",
            fontsize=11.5, color=C_AMBER, fontweight="bold", ha="center")

    # gradient at current (classical momentum) — angled away
    end_g_p = p + grad_at_p
    ax.add_patch(FancyArrowPatch(p, end_g_p, arrowstyle="->", lw=1.8,
                                 color=C_GRAY, ls="--", mutation_scale=16))
    ax.text(end_g_p[0] - 0.15, end_g_p[1] - 0.15,
            r"$\nabla J(\theta_t)$" "\n" "(classical)",
            fontsize=10.5, color=C_GRAY, ha="right", va="top")

    # gradient at lookahead (NAG)
    end_g_l = look + grad_at_look
    ax.add_patch(FancyArrowPatch(look, end_g_l, arrowstyle="->", lw=2.4,
                                 color=C_PURPLE, mutation_scale=20))
    ax.text(end_g_l[0] + 0.20, end_g_l[1] + 0.10,
            r"$\nabla J(\theta_t - \gamma v_{t-1})$" "\n" "(NAG)",
            fontsize=10.5, color=C_PURPLE, fontweight="bold",
            ha="left", va="center")

    # caption box
    box = FancyBboxPatch(
        (0.4, 0.3), 11.2, 1.6,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        linewidth=1.0, edgecolor=C_GRAY, facecolor=C_BG,
    )
    ax.add_patch(box)
    ax.text(6.0, 1.45,
            "Classical Momentum: gradient at the CURRENT point.",
            fontsize=11, ha="center", color=C_DARK)
    ax.text(6.0, 0.75,
            "Nesterov: peek where momentum is taking you, "
            "THEN evaluate the gradient.",
            fontsize=11, ha="center", color=C_PURPLE, fontweight="bold")

    ax.set_title("Lookahead gradient evaluation",
                 fontsize=12, fontweight="bold", color=C_DARK)

    fig.tight_layout()
    save(fig, "fig2_nesterov_lookahead")


# ---------------------------------------------------------------------------
# Figure 3: AdaGrad - per-coordinate adaptive LR
# ---------------------------------------------------------------------------
def fig3_adagrad_per_coord_lr() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Anisotropic loss: x is "frequent" (large gradient), y is "rare"
    def f_aniso(x, y):
        return 0.5 * (10.0 * x**2 + 0.5 * y**2)

    def g_aniso(x, y):
        return np.array([10.0 * x, 0.5 * y])

    # --- Left: GD vs AdaGrad on the anisotropic loss ----------------------
    ax = axes[0]
    xs = np.linspace(-1.2, 1.2, 220)
    ys = np.linspace(-2.0, 2.0, 220)
    X, Y = np.meshgrid(xs, ys)
    Z = f_aniso(X, Y)
    levels = np.linspace(0.03, Z.max(), 14)
    ax.contour(X, Y, Z, levels=levels, colors=C_GRAY, alpha=0.5, linewidths=0.8)
    ax.contourf(X, Y, Z, levels=levels, cmap="Blues", alpha=0.18)
    ax.plot(0, 0, marker="*", color=C_AMBER, markersize=16,
            markeredgecolor=C_DARK, markeredgewidth=0.8, zorder=5)

    # GD
    p = np.array([-1.0, 1.7])
    pts_gd, lr = [p.copy()], 0.12
    for _ in range(50):
        p = p - lr * g_aniso(*p)
        pts_gd.append(p.copy())
    pts_gd = np.array(pts_gd)

    # AdaGrad
    p = np.array([-1.0, 1.7])
    G = np.zeros(2)
    pts_ada, lr, eps = [p.copy()], 0.6, 1e-8
    for _ in range(50):
        g = g_aniso(*p)
        G = G + g**2
        p = p - lr * g / (np.sqrt(G) + eps)
        pts_ada.append(p.copy())
    pts_ada = np.array(pts_ada)

    _plot_traj(ax, pts_gd, C_BLUE, "GD")
    _plot_traj(ax, pts_ada, C_GREEN, "AdaGrad")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.set_title("AdaGrad shrinks the steep direction automatically",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Right: per-coordinate effective LR over time ---------------------
    ax = axes[1]
    p = np.array([-1.0, 1.7])
    G = np.zeros(2)
    eff_lr_x, eff_lr_y, lr, eps = [], [], 0.6, 1e-8
    for _ in range(80):
        g = g_aniso(*p)
        G = G + g**2
        eff = lr / (np.sqrt(G) + eps)
        eff_lr_x.append(eff[0])
        eff_lr_y.append(eff[1])
        p = p - eff * g

    steps = np.arange(len(eff_lr_x))
    ax.plot(steps, eff_lr_x, color=C_AMBER, lw=2.2,
            label="frequent (x): big gradients")
    ax.plot(steps, eff_lr_y, color=C_PURPLE, lw=2.2,
            label="rare (y): small gradients")
    ax.set_yscale("log")
    ax.set_xlabel("step", fontsize=11)
    ax.set_ylabel("effective LR  $\\eta / (\\sqrt{G_t}+\\epsilon)$",
                  fontsize=11)
    ax.set_title("Per-coordinate effective learning rate",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(loc="upper right", fontsize=10)
    ax.text(0.55, 0.10,
            "Both decay monotonically -> AdaGrad's flaw\n"
            "(eventually stops learning)",
            transform=ax.transAxes, fontsize=10, color=C_DARK,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc=C_LIGHT_AMBER,
                      ec=C_AMBER, alpha=0.8))

    fig.tight_layout()
    save(fig, "fig3_adagrad_per_coord_lr")


# ---------------------------------------------------------------------------
# Figure 4: RMSProp - exponential moving average vs AdaGrad accumulation
# ---------------------------------------------------------------------------
def fig4_rmsprop_moving_average() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    rng = np.random.default_rng(7)
    T = 120
    # Synthetic squared-gradient stream: nonstationary (drops mid-training)
    g2 = np.where(np.arange(T) < 60,
                  rng.gamma(2.0, 1.0, T),
                  rng.gamma(2.0, 0.25, T))

    # AdaGrad: cumulative sum
    G_ada = np.cumsum(g2)
    # RMSProp: EMA
    rho = 0.9
    G_rms = np.zeros(T)
    G_rms[0] = g2[0]
    for t in range(1, T):
        G_rms[t] = rho * G_rms[t - 1] + (1 - rho) * g2[t]

    eta = 1e-2
    eff_ada = eta / (np.sqrt(G_ada) + 1e-8)
    eff_rms = eta / (np.sqrt(G_rms) + 1e-8)

    # --- Left: g^2 stream + accumulators ----------------------------------
    ax = axes[0]
    ax.plot(g2, color=C_GRAY, lw=1.0, alpha=0.7, label=r"$g_t^2$ (raw)")
    ax.plot(G_ada / G_ada.max() * g2.max(), color=C_BLUE, lw=2.2,
            label=r"AdaGrad  $\sum g_t^2$ (rescaled)")
    ax.plot(G_rms, color=C_PURPLE, lw=2.4,
            label=r"RMSProp  $E[g^2]_t$")
    ax.axvline(60, color=C_AMBER, ls="--", lw=1.4, alpha=0.7)
    ax.text(58, ax.get_ylim()[1] * 0.55,
            "regime change\n(gradient magnitude drops)",
            fontsize=9.5, color=C_AMBER, fontweight="bold",
            ha="right")
    ax.set_xlabel("step", fontsize=11)
    ax.set_ylabel("magnitude", fontsize=11)
    ax.set_title("Cumulative sum vs exponential moving average",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(loc="lower right", fontsize=9.5)

    # --- Right: effective learning rate over time -------------------------
    ax = axes[1]
    ax.plot(eff_ada, color=C_BLUE, lw=2.2, label="AdaGrad effective LR")
    ax.plot(eff_rms, color=C_PURPLE, lw=2.2, label="RMSProp effective LR")
    ax.axvline(60, color=C_AMBER, ls="--", lw=1.4, alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("step", fontsize=11)
    ax.set_ylabel("effective LR", fontsize=11)
    ax.set_title("RMSProp adapts; AdaGrad keeps shrinking",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(loc="lower left", fontsize=10)
    ax.text(0.40, 0.85,
            "After step 60, RMSProp re-opens the LR\n"
            "to match the new (smaller) gradient regime",
            transform=ax.transAxes, fontsize=9.5, color=C_DARK,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc=C_LIGHT_PURPLE,
                      ec=C_PURPLE, alpha=0.85))

    fig.tight_layout()
    save(fig, "fig4_rmsprop_moving_average")


# ---------------------------------------------------------------------------
# Figure 5: Adam - momentum + RMSProp (block diagram)
# ---------------------------------------------------------------------------
def fig5_adam_combined() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(xy, w, h, label, fc, ec=C_DARK, fs=11, fw="bold"):
        x, y = xy
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.10",
            linewidth=1.4, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=C_DARK)

    def arrow(p1, p2, color=C_DARK, lw=1.6, ls="-"):
        a = FancyArrowPatch(p1, p2, arrowstyle="->",
                            mutation_scale=16, linewidth=lw,
                            color=color, linestyle=ls,
                            shrinkA=4, shrinkB=4)
        ax.add_patch(a)

    # Input gradient
    box((0.3, 3.6), 1.7, 1.0, r"$g_t$", C_LIGHT_BLUE, fs=14)

    # Top branch: momentum (1st moment)
    box((3.2, 5.6), 3.6, 1.2,
        r"$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$",
        C_LIGHT_AMBER, ec=C_AMBER, fs=11)
    box((7.6, 5.6), 2.6, 1.2,
        r"$\hat m_t = m_t / (1-\beta_1^t)$",
        C_LIGHT_AMBER, ec=C_AMBER, fs=11)

    # Bottom branch: variance (2nd moment)
    box((3.2, 1.4), 3.6, 1.2,
        r"$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$",
        C_LIGHT_PURPLE, ec=C_PURPLE, fs=11)
    box((7.6, 1.4), 2.6, 1.2,
        r"$\hat v_t = v_t / (1-\beta_2^t)$",
        C_LIGHT_PURPLE, ec=C_PURPLE, fs=11)

    # Combine -> update
    box((11.0, 3.5), 2.8, 1.2,
        r"$\theta_{t+1} = \theta_t - \dfrac{\eta\,\hat m_t}{\sqrt{\hat v_t}+\epsilon}$",
        C_LIGHT_GREEN, ec=C_GREEN, fs=11.5)

    # Arrows
    arrow((2.0, 4.4), (3.2, 6.2), color=C_AMBER, lw=2.0)
    arrow((2.0, 3.8), (3.2, 2.0), color=C_PURPLE, lw=2.0)
    arrow((6.8, 6.2), (7.6, 6.2), color=C_AMBER, lw=2.0)
    arrow((6.8, 2.0), (7.6, 2.0), color=C_PURPLE, lw=2.0)
    arrow((10.2, 6.2), (11.4, 4.7), color=C_AMBER, lw=2.0)
    arrow((10.2, 2.0), (11.4, 4.0), color=C_PURPLE, lw=2.0)

    # Branch labels
    ax.text(2.7, 6.95, "1st moment  (momentum)",
            fontsize=11.5, color=C_AMBER, fontweight="bold")
    ax.text(2.7, 0.85, "2nd moment  (RMSProp variance)",
            fontsize=11.5, color=C_PURPLE, fontweight="bold")

    # Footer takeaway
    foot = FancyBboxPatch(
        (0.3, 7.2), 13.4, 0.8,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor=C_GRAY, facecolor=C_BG,
    )
    ax.add_patch(foot)
    ax.text(7.0, 7.6,
            "Adam = Momentum (direction) + RMSProp (per-coordinate scale) "
            "+ bias correction (warm-start fix)",
            ha="center", va="center", fontsize=12,
            color=C_DARK, fontweight="bold")

    save(fig, "fig5_adam_combined")


# ---------------------------------------------------------------------------
# Figure 6: AdamW vs Adam (decoupled weight decay)
# ---------------------------------------------------------------------------
def fig6_adamw_vs_adam() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))

    # --- Left: schematic of where weight decay enters ---------------------
    ax = axes[0]
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(xy, w, h, label, fc, ec=C_DARK, fs=10.5, fw="bold"):
        x, y = xy
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.2, edgecolor=ec, facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight=fw, color=C_DARK)

    def arr(p1, p2, color=C_DARK, lw=1.5, ls="-"):
        a = FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=14,
                            linewidth=lw, color=color, linestyle=ls,
                            shrinkA=4, shrinkB=4)
        ax.add_patch(a)

    # Adam (L2): wd folded into the gradient
    ax.text(3.0, 6.5, "Adam + L2  (coupled)", fontsize=12,
            fontweight="bold", color=C_AMBER, ha="center")
    box((0.5, 5.0), 5.0, 0.9,
        r"$g_t \leftarrow g_t + \lambda \theta_t$",
        C_LIGHT_AMBER, ec=C_AMBER)
    box((0.5, 3.8), 5.0, 0.9,
        r"adaptive scale  $\hat v_t$  applied to $g_t$",
        C_LIGHT_AMBER, ec=C_AMBER)
    box((0.5, 2.6), 5.0, 0.9,
        r"$\theta_{t+1} = \theta_t - \dfrac{\eta\,\hat m_t}{\sqrt{\hat v_t}+\epsilon}$",
        C_LIGHT_AMBER, ec=C_AMBER, fs=10)
    arr((3.0, 5.0), (3.0, 4.7), color=C_AMBER)
    arr((3.0, 3.8), (3.0, 3.5), color=C_AMBER)
    ax.text(3.0, 1.6,
            "wd is rescaled by  $1/\\sqrt{\\hat v_t}$\n"
            "->  effective decay shrinks for params with\nlarge gradient history",
            ha="center", fontsize=10, color=C_AMBER, style="italic")

    # AdamW: decoupled
    ax.text(9.0, 6.5, "AdamW  (decoupled)", fontsize=12,
            fontweight="bold", color=C_GREEN, ha="center")
    box((6.5, 5.0), 5.0, 0.9, r"$g_t$  (no wd added here)",
        C_LIGHT_GREEN, ec=C_GREEN)
    box((6.5, 3.8), 5.0, 0.9,
        r"adaptive scale  $\hat v_t$  applied to $g_t$",
        C_LIGHT_GREEN, ec=C_GREEN)
    box((6.5, 2.6), 5.0, 0.9,
        r"$\theta_{t+1} = \theta_t - \eta\,\dfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} - \eta\lambda\,\theta_t$",
        C_LIGHT_GREEN, ec=C_GREEN, fs=9.5)
    arr((9.0, 5.0), (9.0, 4.7), color=C_GREEN)
    arr((9.0, 3.8), (9.0, 3.5), color=C_GREEN)
    ax.text(9.0, 1.6,
            "wd applied DIRECTLY to weights\n"
            "->  uniform shrink, independent of $\\hat v_t$",
            ha="center", fontsize=10, color=C_GREEN, style="italic")

    # --- Right: synthetic generalization curves ---------------------------
    ax = axes[1]
    rng = np.random.default_rng(2)
    epochs = np.arange(0, 120)
    train_adam = 1.6 * np.exp(-epochs / 35) + 0.10 + rng.normal(0, 0.012, len(epochs))
    val_adam = 1.6 * np.exp(-epochs / 30) + 0.32 + 0.0014 * np.maximum(0, epochs - 60) \
        + rng.normal(0, 0.015, len(epochs))
    train_adamw = 1.6 * np.exp(-epochs / 32) + 0.11 + rng.normal(0, 0.012, len(epochs))
    val_adamw = 1.6 * np.exp(-epochs / 30) + 0.22 + rng.normal(0, 0.012, len(epochs))

    ax.plot(epochs, train_adam, color=C_AMBER, lw=1.4, alpha=0.55,
            ls="--", label="Adam train")
    ax.plot(epochs, val_adam, color=C_AMBER, lw=2.4, label="Adam val")
    ax.plot(epochs, train_adamw, color=C_GREEN, lw=1.4, alpha=0.55,
            ls="--", label="AdamW train")
    ax.plot(epochs, val_adamw, color=C_GREEN, lw=2.4, label="AdamW val")
    ax.set_xlabel("epoch", fontsize=11)
    ax.set_ylabel("loss", fontsize=11)
    ax.set_title("AdamW typically generalizes better at the same wd",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(loc="upper right", fontsize=9.5, ncol=2)
    ax.set_ylim(0, 2.0)
    ax.text(0.04, 0.10,
            "Same $\\lambda$, same LR -> AdamW shows a smaller\n"
            "train/val gap because wd isn't gradient-rescaled",
            transform=ax.transAxes, fontsize=10, color=C_DARK, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRAY,
                      alpha=0.9))

    fig.tight_layout()
    save(fig, "fig6_adamw_vs_adam")


# ---------------------------------------------------------------------------
# Figure 7: Modern optimizers landscape (Lion, Sophia, Schedule-Free)
# ---------------------------------------------------------------------------
def fig7_modern_optimizers() -> None:
    fig, ax = plt.subplots(figsize=(14.0, 7.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    title_box = FancyBboxPatch(
        (0.3, 7.0), 13.4, 0.8,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.0, edgecolor=C_GRAY, facecolor=C_BG,
    )
    ax.add_patch(title_box)
    ax.text(7.0, 7.4,
            "Beyond Adam (2023-2025): three directions explored at scale",
            ha="center", va="center", fontsize=13.5,
            color=C_DARK, fontweight="bold")

    cards = [
        {
            "x": 0.3, "y": 0.4, "w": 4.4, "h": 6.3,
            "title": "Lion  (Google, 2023)",
            "color": C_BLUE, "light": C_LIGHT_BLUE,
            "rule": (
                r"$m_t = \beta_2 m_{t-1} + (1-\beta_2) g_t$" "\n"
                r"$\theta_{t+1} = \theta_t - \eta \cdot$"
                "\n"
                r"$\quad \mathrm{sign}(\beta_1 m_{t-1} + (1-\beta_1) g_t)$"
            ),
            "bullets": [
                "sign update -> half memory (no $v_t$)",
                "found by program search (AutoML)",
                "needs ~10x smaller LR than AdamW",
                "wd ~10x larger; fast for ViT/LLM",
            ],
        },
        {
            "x": 4.85, "y": 0.4, "w": 4.4, "h": 6.3,
            "title": "Sophia  (Stanford, 2023)",
            "color": C_PURPLE, "light": C_LIGHT_PURPLE,
            "rule": (
                r"$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$" "\n"
                r"$h_t \approx \mathrm{diag}(H_t)$  "
                "(Hutchinson est.)\n"
                r"$\theta_{t+1} = \theta_t - \eta\,\mathrm{clip}\!\left(\dfrac{m_t}{\max(\gamma h_t,\varepsilon)}\right)$"
            ),
            "bullets": [
                "uses diagonal Hessian (2nd-order)",
                "clipping handles non-convexity",
                "~2x wall-clock vs AdamW (LLM pretrain)",
                "extra cost: occasional Hessian probe",
            ],
        },
        {
            "x": 9.4, "y": 0.4, "w": 4.4, "h": 6.3,
            "title": "Schedule-Free  (Meta, 2024)",
            "color": C_GREEN, "light": C_LIGHT_GREEN,
            "rule": (
                r"$y_t = (1-\beta) z_t + \beta x_t$"
                "\n"
                r"$z_{t+1} = z_t - \eta\,\nabla J(y_t)$"
                "\n"
                r"$x_{t+1} = (1-c_t) x_t + c_t z_{t+1}$"
            ),
            "bullets": [
                "no LR schedule, no total-step prior",
                "iterate averaging baked in",
                "matches cosine without retuning",
                "drop-in for AdamW / SGD-momentum",
            ],
        },
    ]

    for c in cards:
        box = FancyBboxPatch(
            (c["x"], c["y"]), c["w"], c["h"],
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.6, edgecolor=c["color"], facecolor=c["light"],
        )
        ax.add_patch(box)
        ax.text(c["x"] + c["w"] / 2, c["y"] + c["h"] - 0.45,
                c["title"], ha="center", va="center",
                fontsize=13, fontweight="bold", color=c["color"])
        # rule block
        rule_box = FancyBboxPatch(
            (c["x"] + 0.3, c["y"] + c["h"] - 2.7), c["w"] - 0.6, 1.9,
            boxstyle="round,pad=0.04,rounding_size=0.06",
            linewidth=0.8, edgecolor=c["color"], facecolor="white",
        )
        ax.add_patch(rule_box)
        ax.text(c["x"] + c["w"] / 2, c["y"] + c["h"] - 1.75,
                c["rule"], ha="center", va="center",
                fontsize=10, color=C_DARK)
        # bullets
        bullet_y = c["y"] + c["h"] - 3.2
        for b in c["bullets"]:
            ax.text(c["x"] + 0.35, bullet_y, "- " + b,
                    fontsize=10.5, color=C_DARK, va="top")
            bullet_y -= 0.55

    fig.tight_layout()
    save(fig, "fig7_modern_optimizers")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating optimizer-evolution figures...")
    fig1_gd_sgd_momentum_contour()
    fig2_nesterov_lookahead()
    fig3_adagrad_per_coord_lr()
    fig4_rmsprop_moving_average()
    fig5_adam_combined()
    fig6_adamw_vs_adam()
    fig7_modern_optimizers()
    print("Done.")


if __name__ == "__main__":
    main()
