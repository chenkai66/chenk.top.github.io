"""
Figure generation script for Linear Algebra Chapter 10:
Matrix Norms and Condition Numbers.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in 3Blue1Brown style.

Figures:
    fig1_norm_unit_balls       Vector norm unit balls in 2D: L1 (diamond),
                               L2 (circle), L_inf (square), plus an L_p
                               family interpolation showing how p morphs
                               the shape from sharp to flat.
    fig2_operator_norm         Operator (induced) norm as the maximum
                               stretch of the unit circle: trace x -> Ax
                               for x on the unit circle, mark the extremal
                               directions sigma_max and sigma_min.
    fig3_frob_vs_spectral      Frobenius vs spectral norm: same matrix
                               drawn as ellipse with both norms annotated;
                               compares "total energy" vs "max stretch".
    fig4_condition_number      Well-conditioned (kappa near 1) vs
                               ill-conditioned (kappa large) ellipses
                               side by side.
    fig5_perturbation          Sensitivity to perturbation: tiny rotation
                               of b explodes the solution x for an
                               ill-conditioned A.
    fig6_singular_values       Singular value spectra and condition
                               number: well- vs ill-conditioned matrices,
                               plus Hilbert matrix kappa(n) growth.
    fig7_numerical_stability   Solving Ax = b with three methods (normal
                               equations, QR, SVD) as kappa(A) increases:
                               relative error vs condition number.

Usage:
    python3 scripts/figures/linear-algebra/10-matrix-norms-and-condition-numbers.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the
    markdown references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from scipy.linalg import hilbert

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary - L2 / well-conditioned / spectral
C_PURPLE = "#7c3aed"   # secondary - L1 / Frobenius
C_GREEN = "#10b981"    # accent - safe / stable algorithm
C_AMBER = "#f59e0b"    # warning - L_inf / ill-conditioned / perturbed
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"
C_RED = "#dc2626"

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "10-matrix-norms-and-condition-numbers"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "10-矩阵范数与条件数"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH EN and ZH asset folders as <name>.png."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        path = d / f"{name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style_axes(ax, lim=1.6, title=None, equal=True):
    if equal:
        ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color=C_GRAY, lw=0.6, alpha=0.6)
    ax.axvline(0, color=C_GRAY, lw=0.6, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(C_LIGHT)
    if title:
        ax.set_title(title, fontsize=11, color=C_DARK, pad=8)


# ---------------------------------------------------------------------------
# Fig 1: Norm unit balls (L1, L2, L_inf) plus L_p family
# ---------------------------------------------------------------------------
def fig1_norm_unit_balls() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))

    # Parametric unit circle samples used as direction vectors.
    theta = np.linspace(0, 2 * np.pi, 720)
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (N,2)

    def lp_ball(p):
        if np.isinf(p):
            r = 1.0 / np.max(np.abs(dirs), axis=1)
        else:
            r = 1.0 / np.power(np.sum(np.abs(dirs) ** p, axis=1), 1.0 / p)
        return dirs * r[:, None]

    # --- L1 ---
    ax = axes[0]
    pts = lp_ball(1)
    ax.fill(pts[:, 0], pts[:, 1], color=C_PURPLE, alpha=0.18)
    ax.plot(pts[:, 0], pts[:, 1], color=C_PURPLE, lw=2.2)
    _style_axes(ax, title=r"$L^1$ ball  (Manhattan)")
    ax.text(0, -1.35, "diamond — sharp corners on axes",
            ha="center", fontsize=9, color=C_DARK)

    # --- L2 ---
    ax = axes[1]
    pts = lp_ball(2)
    ax.fill(pts[:, 0], pts[:, 1], color=C_BLUE, alpha=0.18)
    ax.plot(pts[:, 0], pts[:, 1], color=C_BLUE, lw=2.2)
    _style_axes(ax, title=r"$L^2$ ball  (Euclidean)")
    ax.text(0, -1.35, "circle — rotation-invariant",
            ha="center", fontsize=9, color=C_DARK)

    # --- L_inf ---
    ax = axes[2]
    pts = lp_ball(np.inf)
    ax.fill(pts[:, 0], pts[:, 1], color=C_AMBER, alpha=0.20)
    ax.plot(pts[:, 0], pts[:, 1], color=C_AMBER, lw=2.2)
    _style_axes(ax, title=r"$L^\infty$ ball  (Chebyshev)")
    ax.text(0, -1.35, "square — flat faces along axes",
            ha="center", fontsize=9, color=C_DARK)

    # --- L_p family ---
    ax = axes[3]
    for p, c, alpha, lw in [
        (1, C_PURPLE, 0.9, 1.6),
        (1.5, "#a855f7", 0.7, 1.3),
        (2, C_BLUE, 0.9, 1.6),
        (4, "#0ea5e9", 0.7, 1.3),
        (np.inf, C_AMBER, 0.9, 1.6),
    ]:
        pts = lp_ball(p)
        label = r"$p=\infty$" if np.isinf(p) else fr"$p={p}$"
        ax.plot(pts[:, 0], pts[:, 1], color=c, lw=lw, alpha=alpha, label=label)
    _style_axes(ax, title=r"$L^p$ balls as $p$ varies")
    ax.text(0, -1.35, "diamond → circle → square",
            ha="center", fontsize=9, color=C_DARK)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.suptitle("Vector norms: the shape of the unit ball depends on $p$",
                 fontsize=12.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig1_norm_unit_balls")


# ---------------------------------------------------------------------------
# Fig 2: Operator (induced) norm as max stretch on unit circle
# ---------------------------------------------------------------------------
def fig2_operator_norm() -> None:
    A = np.array([[2.0, 0.6], [0.4, 1.1]])
    U, s, Vt = np.linalg.svd(A)
    sigma_max, sigma_min = s

    theta = np.linspace(0, 2 * np.pi, 360)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    image = circle @ A.T

    # Extremal pre-image directions = right singular vectors (rows of Vt).
    v_max = Vt[0]
    v_min = Vt[1]
    Av_max = A @ v_max
    Av_min = A @ v_min

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    # Left: input unit circle with extremal directions.
    ax = axes[0]
    ax.fill(circle[:, 0], circle[:, 1], color=C_BLUE, alpha=0.10)
    ax.plot(circle[:, 0], circle[:, 1], color=C_BLUE, lw=1.8,
            label="unit circle  $\\|x\\|=1$")
    for v, c, name in [(v_max, C_AMBER, r"$v_{\max}$"),
                       (v_min, C_GREEN, r"$v_{\min}$")]:
        ax.add_patch(FancyArrowPatch((0, 0), (v[0], v[1]),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=c, lw=2))
        ax.text(v[0] * 1.18, v[1] * 1.18, name, color=c,
                fontsize=11, ha="center", va="center", weight="bold")
    _style_axes(ax, lim=2.6, title="Input  ($\\|x\\|_2 = 1$)")

    # Right: image ellipse with sigma_max and sigma_min axes.
    ax = axes[1]
    ax.fill(image[:, 0], image[:, 1], color=C_PURPLE, alpha=0.10)
    ax.plot(image[:, 0], image[:, 1], color=C_PURPLE, lw=1.8,
            label="image  $\\{Ax : \\|x\\|=1\\}$")
    ax.add_patch(FancyArrowPatch((0, 0), (Av_max[0], Av_max[1]),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=C_AMBER, lw=2.6))
    ax.add_patch(FancyArrowPatch((0, 0), (Av_min[0], Av_min[1]),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=C_GREEN, lw=2.6))
    ax.text(0, -2.85,
            fr"$\|A\|_2 = \sigma_{{\max}} = {sigma_max:.2f}$",
            color=C_AMBER, fontsize=11, weight="bold", ha="center")
    ax.text(0, 2.55,
            fr"$\sigma_{{\min}} = {sigma_min:.2f}$",
            color=C_GREEN, fontsize=11, weight="bold", ha="center")
    _style_axes(ax, lim=2.6, title="Output  $A \\cdot$ unit circle  →  ellipse")

    fig.suptitle(r"Operator norm $\|A\|_2$ = longest semi-axis of the image ellipse",
                 fontsize=12.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig2_operator_norm")


# ---------------------------------------------------------------------------
# Fig 3: Frobenius vs Spectral norm — same matrix, two viewpoints
# ---------------------------------------------------------------------------
def fig3_frob_vs_spectral() -> None:
    A = np.array([[2.4, 0.8], [0.7, 1.3]])
    U, s, Vt = np.linalg.svd(A)
    spectral = s[0]
    frob = np.sqrt(np.sum(s ** 2))

    theta = np.linspace(0, 2 * np.pi, 360)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    image = circle @ A.T

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # Left: spectral norm = longest axis only.
    ax = axes[0]
    ax.fill(image[:, 0], image[:, 1], color=C_BLUE, alpha=0.10)
    ax.plot(image[:, 0], image[:, 1], color=C_BLUE, lw=1.6)
    # Major axis (longest semi-axis = sigma_max).
    major = U[:, 0] * spectral
    ax.add_patch(FancyArrowPatch((0, 0), (major[0], major[1]),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=C_AMBER, lw=2.6))
    ax.text(0, -2.7,
            fr"$\|A\|_2 = \sigma_1 = {spectral:.2f}$",
            color=C_AMBER, fontsize=11, weight="bold", ha="center")
    _style_axes(ax, lim=3.2, title="Spectral norm — only $\\sigma_1$ counts")

    # Right: Frobenius = sqrt(sum of all squared singular values).
    ax = axes[1]
    ax.fill(image[:, 0], image[:, 1], color=C_PURPLE, alpha=0.10)
    ax.plot(image[:, 0], image[:, 1], color=C_PURPLE, lw=1.6)
    for k, sk in enumerate(s):
        axis = U[:, k] * sk
        ax.add_patch(FancyArrowPatch((0, 0), (axis[0], axis[1]),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=C_PURPLE, lw=2.2))
        ax.text(axis[0] * 0.55 + 0.05, axis[1] * 0.55 + 0.05,
                fr"$\sigma_{{{k+1}}}={sk:.2f}$",
                color=C_PURPLE, fontsize=10, weight="bold")
    ax.text(0, -2.7,
            fr"$\|A\|_F = \sqrt{{\sigma_1^2 + \sigma_2^2}} = {frob:.2f}$",
            color=C_PURPLE, fontsize=11, weight="bold", ha="center")
    _style_axes(ax, lim=3.2, title="Frobenius norm — every $\\sigma_i$ counts")

    fig.suptitle("Two ways to size up a matrix: peak stretch vs total energy",
                 fontsize=12.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig3_frob_vs_spectral")


# ---------------------------------------------------------------------------
# Fig 4: Condition number — well- vs ill-conditioned
# ---------------------------------------------------------------------------
def fig4_condition_number() -> None:
    # Well-conditioned: nearly orthogonal scaling.
    A_well = np.array([[1.2, 0.0], [0.0, 1.0]])
    # Ill-conditioned: one direction strongly squashed.
    A_ill = np.array([[1.0, 0.0], [0.0, 0.06]])

    theta = np.linspace(0, 2 * np.pi, 360)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, A, color, label in [
        (axes[0], A_well, C_GREEN, "Well-conditioned"),
        (axes[1], A_ill, C_AMBER, "Ill-conditioned"),
    ]:
        s = np.linalg.svd(A, compute_uv=False)
        kappa = s[0] / s[1]

        ax.fill(circle[:, 0], circle[:, 1], color=C_GRAY, alpha=0.08)
        ax.plot(circle[:, 0], circle[:, 1], color=C_GRAY, lw=1.0,
                ls="--", label="unit circle (input)")

        image = circle @ A.T
        ax.fill(image[:, 0], image[:, 1], color=color, alpha=0.18)
        ax.plot(image[:, 0], image[:, 1], color=color, lw=2.0,
                label="$A \\cdot$ unit circle")

        # Annotate axes: show sigma_max and sigma_min.
        U, _, _ = np.linalg.svd(A)
        for k, sk in enumerate(s):
            axis = U[:, k] * sk
            ax.add_patch(FancyArrowPatch((0, 0), (axis[0], axis[1]),
                                         arrowstyle="-|>", mutation_scale=12,
                                         color=C_DARK, lw=1.6, alpha=0.85))
        title = (fr"{label}: $\kappa = \sigma_{{\max}}/\sigma_{{\min}}"
                 fr" = {kappa:.2f}$")
        _style_axes(ax, lim=1.5, title=title)
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        "Condition number = how flat the output ellipse is",
        fontsize=12.5, color=C_DARK, y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig4_condition_number")


# ---------------------------------------------------------------------------
# Fig 5: Sensitivity to perturbation — small db, huge dx for ill-conditioned A
# ---------------------------------------------------------------------------
def fig5_perturbation() -> None:
    # Build two 2x2 systems with the same true x but very different kappa.
    rng = np.random.default_rng(7)

    def make_system(kappa, x_true):
        # A = U diag(s) V^T with sigma_max=1, sigma_min=1/kappa.
        Q1, _ = np.linalg.qr(rng.standard_normal((2, 2)))
        Q2, _ = np.linalg.qr(rng.standard_normal((2, 2)))
        s = np.array([1.0, 1.0 / kappa])
        A = Q1 @ np.diag(s) @ Q2
        b = A @ x_true
        return A, b

    x_true = np.array([1.0, 1.0])

    # Apply same relative perturbation to b in many directions.
    n_dir = 60
    angles = np.linspace(0, 2 * np.pi, n_dir, endpoint=False)
    rel_db = 0.02   # 2% perturbation magnitude

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

    for ax, kappa, color, label in [
        (axes[0], 2.0, C_GREEN, fr"Well-conditioned  $\kappa\approx{2:.0f}$"),
        (axes[1], 200.0, C_AMBER, fr"Ill-conditioned  $\kappa\approx{200:.0f}$"),
    ]:
        A, b = make_system(kappa, x_true)
        b_norm = np.linalg.norm(b)
        x_solutions = []
        for a in angles:
            db = rel_db * b_norm * np.array([np.cos(a), np.sin(a)])
            x_pert = np.linalg.solve(A, b + db)
            x_solutions.append(x_pert)
        x_solutions = np.array(x_solutions)

        # Light reference circle of size = max perturbation in x for context.
        spread = np.max(np.linalg.norm(x_solutions - x_true, axis=1))
        ref_theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(x_true[0] + spread * np.cos(ref_theta),
                x_true[1] + spread * np.sin(ref_theta),
                color=C_GRAY, ls=":", lw=1.0, alpha=0.7,
                label=fr"radius $={spread:.2g}$")

        # Cloud of perturbed solutions.
        ax.scatter(x_solutions[:, 0], x_solutions[:, 1],
                   color=color, alpha=0.7, s=26,
                   label=fr"$x+\delta x$ for $\|\delta b\|/\|b\|={rel_db*100:.0f}\%$")
        ax.scatter([x_true[0]], [x_true[1]], color=C_DARK, s=70,
                   zorder=5, label="true $x$")

        # Compute and show observed amplification.
        rel_dx = spread / np.linalg.norm(x_true)
        amp = rel_dx / rel_db
        sub = (fr"$\|\delta x\|/\|x\| \leq {rel_dx:.2f}$  "
               fr"(amplification $\approx {amp:.0f}\times$)")

        # Tight per-axis limits centered on x_true.
        pad = max(spread * 1.4, 0.08)
        ax.set_xlim(x_true[0] - pad, x_true[0] + pad)
        ax.set_ylim(x_true[1] - pad, x_true[1] + pad)
        ax.set_aspect("equal")
        ax.axhline(x_true[1], color=C_GRAY, lw=0.4, alpha=0.5)
        ax.axvline(x_true[0], color=C_GRAY, lw=0.4, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(C_LIGHT)
        ax.set_title(label, fontsize=11, color=C_DARK, pad=8)
        ax.text(0.5, -0.1, sub, transform=ax.transAxes,
                ha="center", fontsize=10, color=C_DARK)
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        r"Same 2% wobble in $b$ → tiny vs catastrophic shift in $x$",
        fontsize=12.5, color=C_DARK, y=1.04,
    )
    fig.tight_layout()
    save(fig, "fig5_perturbation")


# ---------------------------------------------------------------------------
# Fig 6: Singular values & condition number; Hilbert kappa(n)
# ---------------------------------------------------------------------------
def fig6_singular_values() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: singular value spectra of two matrices.
    ax = axes[0]
    n = 20
    rng = np.random.default_rng(0)

    def random_matrix_with_kappa(kappa):
        Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
        Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = np.logspace(0, -np.log10(kappa), n)
        return Q1 @ np.diag(s) @ Q2, s

    _, s_well = random_matrix_with_kappa(5.0)
    _, s_ill = random_matrix_with_kappa(1e6)

    idx = np.arange(1, n + 1)
    ax.semilogy(idx, s_well, "o-", color=C_GREEN, lw=2,
                markersize=6, label=fr"well-conditioned  $\kappa={s_well[0]/s_well[-1]:.0f}$")
    ax.semilogy(idx, s_ill, "o-", color=C_AMBER, lw=2,
                markersize=6, label=fr"ill-conditioned  $\kappa={s_ill[0]/s_ill[-1]:.0e}$")
    ax.axhline(1.0, color=C_GRAY, lw=0.6, ls="--")
    ax.set_xlabel("singular value index $i$", fontsize=10)
    ax.set_ylabel(r"$\sigma_i$  (log scale)", fontsize=10)
    ax.set_title("Singular value decay sets the condition number",
                 fontsize=11, color=C_DARK)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.35)

    # Right: Hilbert matrix kappa(n).
    ax = axes[1]
    sizes = np.arange(2, 16)
    conds = [np.linalg.cond(hilbert(int(k))) for k in sizes]
    ax.semilogy(sizes, conds, "o-", color=C_PURPLE, lw=2.2, markersize=7,
                label=r"$\kappa_2(H_n)$")
    # Reference: double precision = 1e16.
    ax.axhline(1e16, color=C_RED, lw=1.0, ls="--",
               label=r"double-precision wall  ($10^{16}$)")
    ax.set_xlabel("matrix size $n$", fontsize=10)
    ax.set_ylabel(r"$\kappa_2(H_n)$  (log scale)", fontsize=10)
    ax.set_title("Hilbert matrix: condition number explodes with $n$",
                 fontsize=11, color=C_DARK)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    ax.grid(True, which="both", alpha=0.35)

    fig.suptitle("Two views of conditioning: spectrum and growth",
                 fontsize=12.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_singular_values")


# ---------------------------------------------------------------------------
# Fig 7: Numerical stability — solve Ax=b, three algorithms vs kappa(A)
# ---------------------------------------------------------------------------
def fig7_numerical_stability() -> None:
    rng = np.random.default_rng(2024)
    n = 40

    kappas = np.logspace(2, 14, 25)
    err_normal = []
    err_qr = []
    err_svd = []

    for k in kappas:
        Q1, _ = np.linalg.qr(rng.standard_normal((n, n)))
        Q2, _ = np.linalg.qr(rng.standard_normal((n, n)))
        s = np.logspace(0, -np.log10(k), n)
        A = Q1 @ np.diag(s) @ Q2
        x_true = rng.standard_normal(n)
        b = A @ x_true

        # Normal equations: solve A^T A x = A^T b (squares the condition number)
        try:
            x_n = np.linalg.solve(A.T @ A, A.T @ b)
            err_normal.append(np.linalg.norm(x_n - x_true)
                              / np.linalg.norm(x_true))
        except np.linalg.LinAlgError:
            err_normal.append(np.nan)

        # QR
        Q, R = np.linalg.qr(A)
        x_q = np.linalg.solve(R, Q.T @ b)
        err_qr.append(np.linalg.norm(x_q - x_true) / np.linalg.norm(x_true))

        # SVD (lstsq with default rcond)
        x_s, *_ = np.linalg.lstsq(A, b, rcond=None)
        err_svd.append(np.linalg.norm(x_s - x_true) / np.linalg.norm(x_true))

    fig, ax = plt.subplots(figsize=(9.5, 5))

    ax.loglog(kappas, err_normal, "o-", color=C_AMBER, lw=2, markersize=6,
              label=r"Normal equations  ($A^{\!\top}\!A\,x = A^{\!\top}b$)")
    ax.loglog(kappas, err_qr, "s-", color=C_BLUE, lw=2, markersize=6,
              label=r"QR  ($Rx = Q^{\!\top}b$)")
    ax.loglog(kappas, err_svd, "^-", color=C_GREEN, lw=2, markersize=6,
              label=r"SVD  (pseudoinverse)")

    # Reference lines: epsilon * kappa and epsilon * kappa^2.
    eps = 2.2e-16
    ax.loglog(kappas, eps * kappas, "--", color=C_BLUE, alpha=0.4, lw=1.2,
              label=r"$\varepsilon_\mathrm{mach}\cdot\kappa$  (QR/SVD limit)")
    ax.loglog(kappas, eps * kappas ** 2, "--", color=C_AMBER, alpha=0.4, lw=1.2,
              label=r"$\varepsilon_\mathrm{mach}\cdot\kappa^2$  (normal eq. limit)")

    ax.axhline(1.0, color=C_RED, lw=0.8, ls=":", alpha=0.7)
    ax.text(kappas[1], 1.4, "useless (>100% error)", color=C_RED,
            fontsize=9, va="bottom")

    ax.set_xlabel(r"condition number  $\kappa(A)$", fontsize=11)
    ax.set_ylabel(r"relative error  $\|\hat x - x\|/\|x\|$", fontsize=11)
    ax.set_title("Solving $Ax=b$: normal equations lose digits twice as fast",
                 fontsize=12, color=C_DARK)
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
    ax.grid(True, which="both", alpha=0.35)
    ax.set_ylim(1e-17, 1e3)

    fig.tight_layout()
    save(fig, "fig7_numerical_stability")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig1_norm_unit_balls()
    fig2_operator_norm()
    fig3_frob_vs_spectral()
    fig4_condition_number()
    fig5_perturbation()
    fig6_singular_values()
    fig7_numerical_stability()
    print("All 7 figures generated.")
    print(f"  EN -> {EN_DIR}")
    print(f"  ZH -> {ZH_DIR}")
