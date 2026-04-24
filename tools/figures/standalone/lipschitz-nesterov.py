"""
Figure generation for the standalone post:
    "Lipschitz Continuity, Strong Convexity & Accelerated Gradient Descent"
    "Lipschitz 连续性、强凸性与加速梯度下降"

Generates 5 didactic figures used by both EN and ZH versions of the article.

Figures:
    fig1_lipschitz_geometry      Lipschitz cone: |f(y)-f(x)| <= L|y-x|
                                 enforces a slope envelope around every point.
    fig2_l_smooth_quadratic      L-smooth function: bounded above by a
                                 quadratic upper model with curvature L.
    fig3_strong_convexity        mu-strongly convex function: bounded below
                                 by a quadratic lower model with curvature mu.
    fig4_convergence_rates       GD vs Heavy Ball vs Nesterov on a strongly
                                 convex quadratic; suboptimality vs iteration.
    fig5_condition_number        Effect of condition number kappa = L / mu
                                 on the iterations needed to reach tolerance.

Usage:
    python3 scripts/figures/standalone/lipschitz-nesterov.py

Output:
    Writes PNGs (DPI=150) to BOTH article asset folders so markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

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
C_GREEN = COLORS["success"]     # accent
C_AMBER = COLORS["warning"]     # highlight / warning
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT_BLUE = "#dbeafe"
C_LIGHT_PURPLE = "#ede9fe"
C_LIGHT_GREEN = "#d1fae5"
C_LIGHT_AMBER = "#fef3c7"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "standalone"
    / "lipschitz-continuity-strong-convexity-nesterov"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "standalone"
    / "深入解析非线性优化中的lipschitz连续性-强凸性与加速梯度下降算法"
)

EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both asset folders at a consistent DPI."""
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  wrote {out.relative_to(REPO_ROOT)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Lipschitz continuity geometric intuition
#   For every point x0, the graph of f must stay inside the double cone
#   { (x, y) : |y - f(x0)| <= L * |x - x0| }.
# ---------------------------------------------------------------------------
def fig1_lipschitz_geometry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    x = np.linspace(-3, 3, 400)

    # Left: a 1-Lipschitz function (|sin| with slope at most 1) sitting
    # comfortably inside its Lipschitz cones at sampled anchors.
    L = 1.0
    f_ok = np.sin(x)
    ax = axes[0]
    ax.plot(x, f_ok, color=C_BLUE, lw=2.4, label=r"$f(x) = \sin x$  (1-Lipschitz)")

    for x0 in (-1.8, 0.4, 2.2):
        y0 = np.sin(x0)
        xs = np.array([x0 - 1.6, x0, x0 + 1.6])
        ys_up = y0 + L * np.abs(xs - x0)
        ys_dn = y0 - L * np.abs(xs - x0)
        cone = Polygon(
            np.column_stack([np.r_[xs, xs[::-1]], np.r_[ys_up, ys_dn[::-1]]]),
            closed=True,
            facecolor=C_LIGHT_AMBER,
            edgecolor=C_AMBER,
            alpha=0.55,
            lw=1.2,
        )
        ax.add_patch(cone)
        ax.plot(x0, y0, "o", color=C_AMBER, ms=7, zorder=5)

    ax.set_title("Lipschitz cone contains the graph", fontsize=13, color=C_DARK)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend(loc="upper left", fontsize=10)
    ax.text(
        0.02, 0.02,
        r"$|f(y)-f(x)| \leq L\,|y-x|$" + "\n" + f"$L = {L}$",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor=C_GRAY, alpha=0.9),
        verticalalignment="bottom",
    )

    # Right: a non-Lipschitz function (sqrt(|x|)) whose slope blows up at 0,
    # so no finite L can contain it near the origin.
    ax = axes[1]
    f_bad = np.sign(x) * np.sqrt(np.abs(x))
    ax.plot(x, f_bad, color=C_PURPLE, lw=2.4, label=r"$f(x) = \mathrm{sign}(x)\sqrt{|x|}$")

    # show that the slope diverges at 0: candidate cone with L=2 still fails
    L_try = 2.0
    x0 = 0.0
    y0 = 0.0
    xs = np.array([x0 - 1.4, x0, x0 + 1.4])
    ys_up = y0 + L_try * np.abs(xs - x0)
    ys_dn = y0 - L_try * np.abs(xs - x0)
    cone = Polygon(
        np.column_stack([np.r_[xs, xs[::-1]], np.r_[ys_up, ys_dn[::-1]]]),
        closed=True,
        facecolor=C_LIGHT_AMBER,
        edgecolor=C_AMBER,
        alpha=0.45,
        lw=1.2,
        label=f"candidate cone L={L_try}",
    )
    ax.add_patch(cone)
    # mark the violating regions where |f(x)| > L|x|
    x_fine = np.linspace(-1.4, 1.4, 400)
    f_fine = np.sign(x_fine) * np.sqrt(np.abs(x_fine))
    mask = np.abs(f_fine) > L_try * np.abs(x_fine) + 1e-9
    ax.fill_between(
        x_fine, f_fine, L_try * np.sign(x_fine) * np.abs(x_fine),
        where=mask, color=C_RED, alpha=0.25, label="violation",
    )
    ax.plot(0, 0, "o", color=C_RED, ms=7, zorder=5)

    ax.set_title("No finite L works near a vertical-tangent point", fontsize=13, color=C_DARK)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.legend(loc="upper left", fontsize=10)
    ax.text(
        0.98, 0.02,
        "slope " + r"$\to \infty$ at $x=0$" + "\nnot Lipschitz",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor=C_GRAY, alpha=0.9),
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    fig.suptitle(
        "Figure 1  Lipschitz continuity: bounded slope as a geometric envelope",
        fontsize=14, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig1_lipschitz_geometry")


# ---------------------------------------------------------------------------
# Figure 2: L-smooth function (gradient is L-Lipschitz)
#   Descent lemma: f(y) <= f(x) + <grad f(x), y-x> + (L/2)|y-x|^2.
# ---------------------------------------------------------------------------
def fig2_l_smooth_quadratic() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))

    # f(x) = 0.5 sin(2x) + 0.5 x^2  -> smooth, max curvature ~ 1 + 2 = 3
    def f(x):
        return 0.5 * np.sin(2 * x) + 0.5 * x ** 2

    def gf(x):
        return np.cos(2 * x) + x

    L = 3.0  # valid upper bound on |f''(x)| for this f

    x = np.linspace(-2.2, 2.2, 400)
    ax.plot(x, f(x), color=C_BLUE, lw=2.6, label=r"$f(x)$")

    # quadratic upper model at three anchors
    for x0, color in zip((-1.4, 0.2, 1.5), (C_PURPLE, C_AMBER, C_GREEN)):
        y0 = f(x0)
        g0 = gf(x0)
        upper = y0 + g0 * (x - x0) + 0.5 * L * (x - x0) ** 2
        lower_lin = y0 + g0 * (x - x0)
        ax.plot(
            x, upper, "--", color=color, lw=1.6,
            label=fr"upper model at $x_0={x0:+.1f}$",
        )
        ax.plot(x, lower_lin, ":", color=color, lw=1.0, alpha=0.7)
        ax.plot(x0, y0, "o", color=color, ms=8, zorder=5)
        # shade the gap between f and the upper model, restricted to a window
        win = (x > x0 - 1.0) & (x < x0 + 1.0)
        ax.fill_between(
            x[win], f(x[win]), upper[win],
            color=color, alpha=0.10,
        )

    ax.set_title(
        r"$L$-smooth descent lemma: $f(y) \leq f(x_0) + \langle \nabla f(x_0), y-x_0 \rangle + \frac{L}{2}\|y-x_0\|^2$",
        fontsize=12.5, color=C_DARK,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.0, 4.5)
    ax.legend(loc="upper center", fontsize=9.5, ncol=2, framealpha=0.95)
    ax.text(
        0.02, 0.97,
        f"L = {L}  (here a valid global bound on $|f''|$)\n"
        "dotted = tangent line   dashed = upper quadratic",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor=C_GRAY, alpha=0.9),
        verticalalignment="top",
    )

    fig.suptitle(
        "Figure 2  L-smoothness gives a global quadratic upper bound at every point",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig2_l_smooth_quadratic")


# ---------------------------------------------------------------------------
# Figure 3: Strong convexity
#   f is mu-strongly convex iff
#       f(y) >= f(x) + <grad f(x), y-x> + (mu/2) |y-x|^2.
#   The graph sits ABOVE a quadratic lower model.
# ---------------------------------------------------------------------------
def fig3_strong_convexity() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: strongly convex example
    mu = 0.8
    L_curve = 4.0  # actual curvature
    x = np.linspace(-2.2, 2.2, 400)

    def f_sc(x):
        return 0.5 * L_curve * x ** 2 + 0.3 * np.cos(3 * x)

    def gf_sc(x):
        return L_curve * x - 0.9 * np.sin(3 * x)

    ax = axes[0]
    ax.plot(x, f_sc(x), color=C_BLUE, lw=2.6, label=r"$f(x)$")
    for x0, color in zip((-1.3, 0.0, 1.3), (C_PURPLE, C_AMBER, C_GREEN)):
        y0 = f_sc(x0)
        g0 = gf_sc(x0)
        lower = y0 + g0 * (x - x0) + 0.5 * mu * (x - x0) ** 2
        ax.plot(x, lower, "--", color=color, lw=1.6, label=fr"lower model at $x_0={x0:+.1f}$")
        ax.plot(x0, y0, "o", color=color, ms=8, zorder=5)

    ax.set_title(fr"$\mu$-strongly convex  ($\mu = {mu}$)", fontsize=13, color=C_DARK)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1, 11)
    ax.legend(loc="upper center", fontsize=9.5, ncol=2, framealpha=0.95)
    ax.text(
        0.02, 0.97,
        r"$f(y) \geq f(x_0) + \langle \nabla f(x_0), y-x_0 \rangle + \frac{\mu}{2}\|y-x_0\|^2$",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor=C_GRAY, alpha=0.9),
        verticalalignment="top",
    )

    # Right: merely convex (mu = 0) for contrast: linear in flat region
    ax = axes[1]
    def f_flat(x):
        # convex, but flat near the minimum (mu = 0): a quartic
        return 0.05 * x ** 4

    def gf_flat(x):
        return 0.2 * x ** 3

    ax.plot(x, f_flat(x), color=C_BLUE, lw=2.6, label=r"$f(x) = 0.05\, x^4$ (convex, $\mu=0$)")
    for x0, color in zip((-1.6, 0.0, 1.6), (C_PURPLE, C_AMBER, C_GREEN)):
        y0 = f_flat(x0)
        g0 = gf_flat(x0)
        tan = y0 + g0 * (x - x0)
        ax.plot(x, tan, "--", color=color, lw=1.4, label=fr"tangent at $x_0={x0:+.1f}$")
        ax.plot(x0, y0, "o", color=color, ms=8, zorder=5)

    ax.set_title(r"Convex but not strongly convex (any $\mu>0$ fails near 0)", fontsize=12.5, color=C_DARK)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.2, 1.1)
    ax.legend(loc="upper center", fontsize=9.5, ncol=2, framealpha=0.95)
    ax.text(
        0.5, 0.04,
        "no $\\mu$-quadratic fits below near $x=0$\n$\\Rightarrow$ no linear convergence guarantee",
        transform=ax.transAxes,
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor=C_GRAY, alpha=0.9),
        ha="center",
    )

    fig.suptitle(
        "Figure 3  Strong convexity = a quadratic LOWER bound at every point",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig3_strong_convexity")


# ---------------------------------------------------------------------------
# Figure 4: Convergence rate comparison
#   Quadratic f(x) = 0.5 x^T A x with kappa = L/mu = 100,
#   compare GD, Heavy Ball (Polyak), Nesterov AGD.
# ---------------------------------------------------------------------------
def _quadratic(d=50, kappa=100.0, seed=0):
    """Build a diagonal quadratic with eigenvalues spanning [mu, L]."""
    rng = np.random.default_rng(seed)
    L_val = 1.0
    mu_val = L_val / kappa
    eig = np.exp(np.linspace(np.log(mu_val), np.log(L_val), d))
    rng.shuffle(eig)
    x_star = np.zeros(d)
    x0 = rng.normal(size=d)
    return eig, x0, x_star, L_val, mu_val


def _f_and_grad(eig, x):
    return 0.5 * np.sum(eig * x ** 2), eig * x


def _gd(eig, x0, L_val, n_iter=400):
    eta = 1.0 / L_val
    x = x0.copy()
    hist = []
    for _ in range(n_iter):
        f, g = _f_and_grad(eig, x)
        hist.append(f)
        x = x - eta * g
    return np.array(hist)


def _heavy_ball(eig, x0, L_val, mu_val, n_iter=400):
    # Polyak optimal: alpha = 4 / (sqrt(L) + sqrt(mu))^2,
    #                 beta  = ((sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu)))^2
    sL, smu = np.sqrt(L_val), np.sqrt(mu_val)
    alpha = 4.0 / (sL + smu) ** 2
    beta = ((sL - smu) / (sL + smu)) ** 2
    x_prev = x0.copy()
    x = x0.copy()
    hist = []
    for _ in range(n_iter):
        f, g = _f_and_grad(eig, x)
        hist.append(f)
        x_next = x - alpha * g + beta * (x - x_prev)
        x_prev = x
        x = x_next
    return np.array(hist)


def _nesterov(eig, x0, L_val, mu_val, n_iter=400):
    # Constant-momentum form for strongly convex case.
    sk = np.sqrt(mu_val / L_val)
    beta = (1 - sk) / (1 + sk)
    eta = 1.0 / L_val
    x_prev = x0.copy()
    x = x0.copy()
    hist = []
    for _ in range(n_iter):
        y = x + beta * (x - x_prev)
        f, g = _f_and_grad(eig, x)
        hist.append(f)
        _, gy = _f_and_grad(eig, y)
        x_next = y - eta * gy
        x_prev = x
        x = x_next
    return np.array(hist)


def fig4_convergence_rates() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel A: kappa = 100
    eig, x0, _, L_val, mu_val = _quadratic(d=50, kappa=100.0, seed=1)
    n_iter = 400
    hist_gd = _gd(eig, x0, L_val, n_iter)
    hist_hb = _heavy_ball(eig, x0, L_val, mu_val, n_iter)
    hist_ne = _nesterov(eig, x0, L_val, mu_val, n_iter)

    ax = axes[0]
    ax.semilogy(hist_gd, color=C_BLUE, lw=2.2, label="Gradient descent")
    ax.semilogy(hist_hb, color=C_AMBER, lw=2.2, label="Heavy ball (Polyak)")
    ax.semilogy(hist_ne, color=C_PURPLE, lw=2.2, label="Nesterov AGD")

    # Theoretical envelopes: GD ~ ((kappa-1)/(kappa+1))^(2t), Nesterov ~ (1-sqrt(1/kappa))^t
    t = np.arange(n_iter)
    rho_gd = ((1 - 1.0 / 100) ** (2 * t))   # not exact; illustrative
    rho_ne = (1 - np.sqrt(1.0 / 100)) ** t
    ax.semilogy(t, hist_gd[0] * rho_gd, "--", color=C_BLUE, lw=1.0, alpha=0.6,
                label=r"GD bound $\propto (1-1/\kappa)^{2t}$")
    ax.semilogy(t, hist_ne[0] * rho_ne, "--", color=C_PURPLE, lw=1.0, alpha=0.6,
                label=r"AGD bound $\propto (1-1/\sqrt{\kappa})^t$")

    ax.set_title(r"Suboptimality vs iteration  ($\kappa = L/\mu = 100$)", fontsize=12.5, color=C_DARK)
    ax.set_xlabel("iteration t")
    ax.set_ylabel(r"$f(x_t) - f^\star$")
    ax.set_ylim(1e-12, 1e3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    # Panel B: zoom early iterations to reveal momentum overshoot
    ax = axes[1]
    n_zoom = 80
    ax.plot(hist_gd[:n_zoom], color=C_BLUE, lw=2.2, label="GD (monotone)")
    ax.plot(hist_hb[:n_zoom], color=C_AMBER, lw=2.2, label="Heavy ball (oscillates)")
    ax.plot(hist_ne[:n_zoom], color=C_PURPLE, lw=2.2, label="Nesterov AGD")
    ax.set_yscale("log")
    ax.set_title("Early iterations: AGD oscillates but tracks the lower envelope",
                 fontsize=12.5, color=C_DARK)
    ax.set_xlabel("iteration t")
    ax.set_ylabel(r"$f(x_t) - f^\star$")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    fig.suptitle(
        "Figure 4  Acceleration replaces $\\kappa$ with $\\sqrt{\\kappa}$ in the rate",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig4_convergence_rates")


# ---------------------------------------------------------------------------
# Figure 5: Effect of condition number kappa = L / mu
#   Iterations to reach tolerance epsilon for GD vs Nesterov as kappa grows.
# ---------------------------------------------------------------------------
def fig5_condition_number() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    kappas = np.array([2, 5, 10, 30, 100, 300, 1000, 3000, 10000])
    eps = 1e-6
    iters_gd = []
    iters_ne = []
    for k in kappas:
        eig, x0, _, L_val, mu_val = _quadratic(d=30, kappa=float(k), seed=2)
        # Run long enough for the worse method
        n_iter = int(min(60000, max(2000, 12 * k)))
        h_gd = _gd(eig, x0, L_val, n_iter)
        h_ne = _nesterov(eig, x0, L_val, mu_val, n_iter)
        f0 = h_gd[0]
        target = eps * f0
        # First t where h_t <= target; if never, use n_iter (will be visible as plateau).
        t_gd = int(np.argmax(h_gd <= target)) if (h_gd <= target).any() else n_iter
        t_ne = int(np.argmax(h_ne <= target)) if (h_ne <= target).any() else n_iter
        iters_gd.append(max(t_gd, 1))
        iters_ne.append(max(t_ne, 1))

    iters_gd = np.array(iters_gd)
    iters_ne = np.array(iters_ne)

    # Left: empirical iterations vs kappa
    ax = axes[0]
    ax.loglog(kappas, iters_gd, "o-", color=C_BLUE, lw=2.2, ms=7, label="GD (empirical)")
    ax.loglog(kappas, iters_ne, "s-", color=C_PURPLE, lw=2.2, ms=7, label="Nesterov AGD (empirical)")
    # theory reference lines
    ref_gd = kappas * np.log(1 / eps)
    ref_ne = np.sqrt(kappas) * np.log(1 / eps)
    ax.loglog(kappas, ref_gd, "--", color=C_BLUE, lw=1.0, alpha=0.5,
              label=r"$\kappa\,\log(1/\epsilon)$")
    ax.loglog(kappas, ref_ne, "--", color=C_PURPLE, lw=1.0, alpha=0.5,
              label=r"$\sqrt{\kappa}\,\log(1/\epsilon)$")
    ax.set_xlabel(r"condition number $\kappa = L / \mu$")
    ax.set_ylabel(r"iterations to reach $f - f^\star \leq 10^{-6} f_0$")
    ax.set_title("Iterations grow linearly in $\\kappa$ for GD, $\\sqrt{\\kappa}$ for AGD",
                 fontsize=12.5, color=C_DARK)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)

    # Right: speedup ratio
    ax = axes[1]
    speedup = iters_gd / iters_ne
    ax.semilogx(kappas, speedup, "o-", color=C_GREEN, lw=2.4, ms=8, label="GD iters / AGD iters")
    ax.semilogx(kappas, np.sqrt(kappas), "--", color=C_GRAY, lw=1.2,
                label=r"theoretical $\sqrt{\kappa}$")
    ax.set_xlabel(r"condition number $\kappa$")
    ax.set_ylabel("speedup factor")
    ax.set_title("Acceleration speedup vs $\\kappa$", fontsize=12.5, color=C_DARK)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    for k, s in zip(kappas, speedup):
        if k in (10, 100, 1000, 10000):
            ax.annotate(
                f"{s:.1f}x",
                xy=(k, s), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9, color=C_DARK,
            )

    fig.suptitle(
        "Figure 5  Condition number $\\kappa$ controls the cost; acceleration breaks the linear law",
        fontsize=13, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig5_condition_number")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Lipschitz / strong convexity / Nesterov figures ...")
    fig1_lipschitz_geometry()
    fig2_l_smooth_quadratic()
    fig3_strong_convexity()
    fig4_convergence_rates()
    fig5_condition_number()
    print("Done.")


if __name__ == "__main__":
    main()
