"""Figure generation script for PDE+ML Part 02: Neural Operator Theory.

Generates 7 figures used in both EN and ZH versions of the article:
    fig1_operator_concept.png      - Operator learning: function -> function map
    fig2_deeponet_architecture.png - DeepONet branch + trunk decomposition
    fig3_fno_spectral_layer.png    - Fourier Neural Operator spectral convolution
    fig4_resolution_invariance.png - Train at coarse grid, test at fine grid
    fig5_method_comparison.png     - PINN vs FNO vs DeepONet comparison
    fig6_geometry_application.png  - Operator solves PDE family across geometries
    fig7_universal_approx.png      - Chen-Chen universal approximation theorem

Run from anywhere; outputs are duplicated to both EN and ZH asset folders.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = PROJECT_ROOT / "source/_posts/en/pde-ml/02-Neural-Operator-Theory"
ZH_DIR = PROJECT_ROOT / "source/_posts/zh/pde-ml/02-神经算子理论"

# Brand palette
BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
RED = "#ef4444"
AMBER = "#f59e0b"
INK = "#1f2937"
MUTED = "#6b7280"
LIGHT = "#f3f4f6"
SOFT = "#eef2ff"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "grid.color": "#e5e7eb",
        "grid.linewidth": 0.6,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name)
    plt.close(fig)


def _box(ax, xy, w, h, text, fc=SOFT, ec=BLUE, fontsize=10, weight="bold"):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.6, edgecolor=ec, facecolor=fc,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, color=INK)


def _arrow(ax, p0, p1, color=INK, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color,
    ))


# ---------------------------------------------------------------------------
# Fig 1 - Operator learning: a map between function spaces
# ---------------------------------------------------------------------------
def fig1_operator_concept() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    x = np.linspace(0, 2 * np.pi, 256)
    rng = np.random.default_rng(0)

    # Left: input function space (several sample initial conditions)
    ax = axes[0]
    inputs = []
    for k in range(4):
        a = rng.uniform(0.5, 1.2)
        f = rng.integers(1, 4)
        ph = rng.uniform(0, 2 * np.pi)
        u0 = a * np.sin(f * x + ph) + 0.4 * rng.standard_normal()
        inputs.append(u0)
        ax.plot(x, u0, color=BLUE, alpha=0.55, lw=1.7)
    ax.set_title(r"Input function space $\mathcal{A}$", color=BLUE)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u_0(x)$")
    ax.text(0.5, -0.27,
            "Initial conditions, parameters, source terms",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=10)

    # Middle: the operator G itself
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, (0.18, 0.40), 0.64, 0.22,
         r"Neural Operator  $\mathcal{G}_\theta$",
         fc=SOFT, ec=PURPLE, fontsize=14)
    ax.text(0.5, 0.78,
            r"$\mathcal{G}_\theta : \mathcal{A} \to \mathcal{U}$",
            ha="center", fontsize=15, color=PURPLE, weight="bold")
    ax.text(0.5, 0.27,
            "Trains once on (input, output)\nfunction pairs",
            ha="center", fontsize=10, color=INK)
    ax.text(0.5, 0.10,
            "Discretization-invariant",
            ha="center", fontsize=10, color=GREEN, style="italic", weight="bold")

    # Right: output function space (corresponding solutions)
    ax = axes[2]
    for u0 in inputs:
        # cheap "diffusion" surrogate to mimic an output
        uT = np.real(np.fft.ifft(np.fft.fft(u0) * np.exp(-0.05 * np.arange(len(u0)) ** 1.2)))
        ax.plot(x, uT, color=GREEN, alpha=0.7, lw=1.7)
    ax.set_title(r"Output function space $\mathcal{U}$", color=GREEN)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x, T)$")
    ax.text(0.5, -0.27,
            "Solution snapshots, predicted fields",
            transform=ax.transAxes, ha="center", color=MUTED, fontsize=10)

    # Connect with arrows in figure coordinates
    for x0, x1 in [(0.30, 0.385), (0.625, 0.71)]:
        fig.add_artist(FancyArrowPatch(
            (x0, 0.5), (x1, 0.5),
            transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=20, lw=2.0, color=PURPLE,
        ))

    fig.suptitle("Operator learning: a map between infinite-dimensional function spaces",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig1_operator_concept.png")


# ---------------------------------------------------------------------------
# Fig 2 - DeepONet architecture
# ---------------------------------------------------------------------------
def fig2_deeponet_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")

    # Branch input (function values at sensor points)
    _box(ax, (0.3, 5.2), 1.9, 1.0,
         "Input function\n" + r"$u(x_1),\ldots,u(x_m)$",
         fc=SOFT, ec=BLUE, fontsize=10)
    # Trunk input (query coordinate)
    _box(ax, (0.3, 1.0), 1.9, 1.0,
         "Query point\n" + r"$y \in \Omega$",
         fc="#fef3c7", ec=AMBER, fontsize=10)

    # Branch network stack
    for i, name in enumerate(["FC", "FC", "FC"]):
        _box(ax, (3.0 + i * 1.2, 5.3), 0.95, 0.8, name, fc=LIGHT, ec=BLUE, fontsize=10)
    _arrow(ax, (2.2, 5.7), (3.0, 5.7), color=BLUE)
    _arrow(ax, (3.95, 5.7), (4.2, 5.7), color=BLUE)
    _arrow(ax, (5.15, 5.7), (5.4, 5.7), color=BLUE)

    # Trunk network stack
    for i, name in enumerate(["FC", "FC", "FC"]):
        _box(ax, (3.0 + i * 1.2, 1.1), 0.95, 0.8, name, fc=LIGHT, ec=AMBER, fontsize=10)
    _arrow(ax, (2.2, 1.5), (3.0, 1.5), color=AMBER)
    _arrow(ax, (3.95, 1.5), (4.2, 1.5), color=AMBER)
    _arrow(ax, (5.15, 1.5), (5.4, 1.5), color=AMBER)

    # Branch output vector b_k
    _box(ax, (6.6, 5.3), 1.4, 0.8, r"$b \in \mathbb{R}^{p}$",
         fc=SOFT, ec=BLUE, fontsize=11)
    _box(ax, (6.6, 1.1), 1.4, 0.8, r"$t \in \mathbb{R}^{p}$",
         fc="#fef3c7", ec=AMBER, fontsize=11)
    _arrow(ax, (6.35, 5.7), (6.6, 5.7), color=BLUE)
    _arrow(ax, (6.35, 1.5), (6.6, 1.5), color=AMBER)

    # Inner-product node
    _box(ax, (8.6, 3.2), 1.5, 1.4,
         r"$\langle b,\, t \rangle$" + "\nInner product",
         fc=SOFT, ec=PURPLE, fontsize=11)
    _arrow(ax, (8.0, 5.7), (8.9, 4.55), color=PURPLE, lw=1.8)
    _arrow(ax, (8.0, 1.5), (8.9, 3.25), color=PURPLE, lw=1.8)

    # Output
    _box(ax, (10.6, 3.4), 1.3, 1.0,
         r"$\mathcal{G}(u)(y)$",
         fc=SOFT, ec=GREEN, fontsize=12)
    _arrow(ax, (10.1, 3.9), (10.6, 3.9), color=PURPLE, lw=1.8)

    # Branch / Trunk labels
    ax.text(4.4, 6.45, "Branch network", ha="center", color=BLUE, fontsize=12, weight="bold")
    ax.text(4.4, 0.6, "Trunk network", ha="center", color=AMBER, fontsize=12, weight="bold")

    # Equation footer
    ax.text(6.0, 0.05,
            r"$\mathcal{G}(u)(y) \approx \sum_{k=1}^{p} b_k(u)\, t_k(y)$"
            r"   —   Chen-Chen (1995) operator universal approximation",
            ha="center", color=INK, fontsize=11, style="italic")

    ax.set_title("DeepONet: branch encodes the input function, trunk encodes the query location",
                 fontsize=13)
    fig.tight_layout()
    _save(fig, "fig2_deeponet_architecture.png")


# ---------------------------------------------------------------------------
# Fig 3 - FNO spectral convolution layer
# ---------------------------------------------------------------------------
def fig3_fno_spectral_layer() -> None:
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1.0], hspace=0.55, wspace=0.45)

    # ---- Top row: schematic flow ----
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")

    _box(ax, (0.2, 1.0), 1.5, 1.0, r"$v(x)$", fc=SOFT, ec=BLUE, fontsize=12)
    _box(ax, (2.2, 1.0), 1.5, 1.0, r"$\mathcal{F}$  FFT", fc=LIGHT, ec=PURPLE, fontsize=11)
    _box(ax, (4.2, 1.0), 2.0, 1.0,
         r"$R_\theta \cdot$  (truncate to $k_{\max}$)",
         fc=SOFT, ec=PURPLE, fontsize=11)
    _box(ax, (6.7, 1.0), 1.6, 1.0, r"$\mathcal{F}^{-1}$  iFFT", fc=LIGHT, ec=PURPLE, fontsize=11)
    _box(ax, (8.8, 1.0), 1.5, 1.0, r"$+\, W v$", fc="#fef3c7", ec=AMBER, fontsize=12)
    _box(ax, (10.7, 1.0), 1.1, 1.0, r"$\sigma$", fc=SOFT, ec=GREEN, fontsize=12)

    for x0, x1 in [(1.7, 2.2), (3.7, 4.2), (6.2, 6.7), (8.3, 8.8), (10.3, 10.7)]:
        _arrow(ax, (x0, 1.5), (x1, 1.5), color=INK)

    ax.text(5.2, 2.35, "Spectral path: global linear operator", ha="center",
            color=PURPLE, fontsize=10, weight="bold")
    ax.text(9.0, 2.35, "Local pointwise residual", ha="center",
            color=AMBER, fontsize=10, weight="bold")

    # ---- Bottom-left: an input signal ----
    ax = fig.add_subplot(gs[1, 0])
    x = np.linspace(0, 2 * np.pi, 256)
    sig = (np.sin(x) + 0.5 * np.sin(3 * x + 0.4) +
           0.3 * np.sin(7 * x) + 0.15 * np.sin(15 * x))
    ax.plot(x, sig, color=BLUE, lw=1.4)
    ax.set_title(r"$v(x)$", color=BLUE)
    ax.set_xlabel(r"$x$"); ax.set_ylabel("amp")
    ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])

    # ---- Bottom-mid-left: full spectrum ----
    ax = fig.add_subplot(gs[1, 1])
    spec = np.abs(np.fft.rfft(sig))
    k = np.arange(len(spec))
    ax.bar(k[:40], spec[:40], color=PURPLE, alpha=0.8)
    ax.set_title(r"$\mathcal{F}(v)$  spectrum", color=PURPLE)
    ax.set_xlabel("frequency $k$"); ax.set_ylabel(r"$|\hat v_k|$")

    # ---- Bottom-mid-right: truncate + learnable filter ----
    ax = fig.add_subplot(gs[1, 2])
    kmax = 12
    R = np.zeros_like(spec)
    R[:kmax] = np.linspace(1.0, 0.4, kmax) * (1 + 0.3 * np.sin(np.linspace(0, 3, kmax)))
    truncated = spec * R
    ax.bar(k[:40], truncated[:40], color=GREEN, alpha=0.85,
           label=f"keep $k < {kmax}$")
    ax.axvline(kmax, color=RED, ls="--", lw=1.4, label=r"$k_{\max}$ cut-off")
    ax.set_title(r"$R_\theta \cdot \hat v$", color=GREEN)
    ax.set_xlabel("frequency $k$")
    ax.legend(fontsize=8, loc="upper right")

    # ---- Bottom-right: reconstructed output ----
    ax = fig.add_subplot(gs[1, 3])
    full = np.zeros_like(spec, dtype=complex)
    fft_v = np.fft.rfft(sig)
    full[:kmax] = fft_v[:kmax] * R[:kmax]
    out = np.real(np.fft.irfft(full, n=len(sig)))
    ax.plot(x, sig, color=BLUE, lw=1.0, alpha=0.4, label="input")
    ax.plot(x, out, color=GREEN, lw=1.6, label="output")
    ax.set_title(r"$\mathcal{F}^{-1}(R_\theta \hat v)$", color=GREEN)
    ax.set_xlabel(r"$x$")
    ax.set_xticks([0, np.pi, 2 * np.pi]); ax.set_xticklabels(["0", r"$\pi$", r"$2\pi$"])
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Fourier Neural Operator: spectral convolution = global linear filter learned in $k$-space",
                 fontsize=13, weight="bold")
    _save(fig, "fig3_fno_spectral_layer.png")


# ---------------------------------------------------------------------------
# Fig 4 - Resolution invariance
# ---------------------------------------------------------------------------
def fig4_resolution_invariance() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # 2D Gaussian-bump field as a stand-in solution
    def field(n: int) -> np.ndarray:
        x = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, x)
        z = (np.exp(-((X - 0.2) ** 2 + (Y + 0.1) ** 2) / 0.12)
             - 0.6 * np.exp(-((X + 0.4) ** 2 + (Y - 0.3) ** 2) / 0.08)
             + 0.4 * np.sin(3 * X) * np.cos(3 * Y) * np.exp(-(X ** 2 + Y ** 2)))
        return z

    grids = [(32, "Train @ 32x32"), (64, "Test @ 64x64"), (256, "Test @ 256x256")]
    for ax, (n, title) in zip(axes, grids):
        z = field(n)
        im = ax.imshow(z, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                       extent=[-1, 1, -1, 1])
        ax.set_title(title, color=BLUE if n == 32 else GREEN)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{n*n} dof", color=MUTED)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("solution value", color=INK)

    fig.suptitle("Resolution invariance: a single set of weights evaluates at any grid resolution",
                 fontsize=13, weight="bold")
    _save(fig, "fig4_resolution_invariance.png")


# ---------------------------------------------------------------------------
# Fig 5 - PINN vs FNO vs DeepONet
# ---------------------------------------------------------------------------
def fig5_method_comparison() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Left: bar chart comparing four properties on a 0..5 scale
    ax = axes[0]
    methods = ["PINN", "FNO", "DeepONet"]
    colors = [RED, BLUE, PURPLE]
    properties = ["Multi-task\nreuse", "Resolution\ninvariance",
                  "Geometry\nflexibility", "Inference\nspeed"]
    scores = np.array([
        [1, 1, 4, 1],   # PINN
        [5, 5, 2, 5],   # FNO
        [5, 4, 5, 5],   # DeepONet
    ])
    x = np.arange(len(properties))
    w = 0.25
    for i, (m, c) in enumerate(zip(methods, colors)):
        ax.bar(x + (i - 1) * w, scores[i], width=w, color=c, label=m,
               edgecolor=INK, linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(properties, fontsize=10)
    ax.set_ylabel("Capability score (0-5)")
    ax.set_ylim(0, 5.5)
    ax.set_title("Capability profile by method", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    for i in range(len(methods)):
        for j, s in enumerate(scores[i]):
            ax.text(x[j] + (i - 1) * w, s + 0.12, str(s),
                    ha="center", fontsize=9, color=INK)

    # Right: cost vs reuse scatter / arrows
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 10)
    ax.set_xlabel("Inference cost per new instance  -->  (lower is better)")
    ax.set_ylabel("Reusability across instances  -->  (higher is better)")
    pts = {
        "PINN":      (8.5, 1.2, RED),
        "FNO":       (1.5, 8.5, BLUE),
        "DeepONet":  (1.8, 8.0, PURPLE),
        "Classical solver": (6.5, 5.0, MUTED),
    }
    for name, (xp, yp, c) in pts.items():
        ax.scatter(xp, yp, s=320, color=c, edgecolor=INK, linewidth=1.0, zorder=3)
        ax.text(xp + 0.25, yp + 0.25, name, fontsize=10, weight="bold", color=c)

    # quadrant tinting
    ax.add_patch(plt.Rectangle((0, 5), 5, 5, color=GREEN, alpha=0.06))
    ax.text(0.4, 9.5, "Train once, reuse forever", color=GREEN, fontsize=10, weight="bold")
    ax.add_patch(plt.Rectangle((5, 0), 5, 5, color=RED, alpha=0.06))
    ax.text(9.6, 0.5, "Per-instance retraining", color=RED, fontsize=10, weight="bold",
            ha="right")

    ax.set_title("Cost vs reusability landscape", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("PINN solves one PDE; neural operators solve a family",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig5_method_comparison.png")


# ---------------------------------------------------------------------------
# Fig 6 - Application: solving PDEs across geometries / parameters
# ---------------------------------------------------------------------------
def fig6_geometry_application() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.0))

    rng = np.random.default_rng(42)
    x = np.linspace(-1, 1, 96)
    X, Y = np.meshgrid(x, x)

    # Top row: input "permeability" / coefficient field
    # Bottom row: corresponding "pressure" output
    for col in range(4):
        # Random smooth Gaussian-mix coefficient
        n_blobs = rng.integers(2, 5)
        a = np.zeros_like(X)
        for _ in range(n_blobs):
            cx, cy = rng.uniform(-0.7, 0.7, size=2)
            sig = rng.uniform(0.08, 0.25)
            amp = rng.uniform(0.5, 1.2)
            a += amp * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / sig)
        a = a / a.max()

        # cheap pressure surrogate: smoothed inverse of a
        p = np.real(np.fft.ifft2(np.fft.fft2(1 - a) *
                                 np.exp(-0.0008 * (np.fft.fftfreq(len(x)) ** 2 +
                                                   np.fft.fftfreq(len(x))[:, None] ** 2))))
        p = (p - p.min()) / (p.max() - p.min())

        ax = axes[0, col]
        im0 = ax.imshow(a, cmap="viridis", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(r"Input $a(x)$" + "\n(coefficient)", color=BLUE, fontsize=10)
        ax.set_title(f"sample {col + 1}", fontsize=10, color=MUTED)

        ax = axes[1, col]
        im1 = ax.imshow(p, cmap="RdBu_r", origin="lower")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(r"Output $u(x)$" + "\n(solution)", color=GREEN, fontsize=10)

    fig.suptitle(
        "One trained operator solves a PDE family: 4 different coefficient fields, 4 different solutions",
        fontsize=13, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig6_geometry_application.png")


# ---------------------------------------------------------------------------
# Fig 7 - Universal approximation theorem for operators (Chen-Chen 1995)
# ---------------------------------------------------------------------------
def fig7_universal_approx() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Left: schematic of the theorem statement
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title("Chen-Chen (1995): operator universal approximation", fontsize=12)

    _box(ax, (0.4, 4.0), 2.4, 1.2,
         r"Continuous operator" + "\n" + r"$\mathcal{G}: C(K_1) \to C(K_2)$",
         fc=SOFT, ec=BLUE, fontsize=10)
    _box(ax, (3.6, 4.0), 2.6, 1.2,
         r"Sampled function" + "\n" + r"$u(x_1),\ldots,u(x_m)$",
         fc=LIGHT, ec=AMBER, fontsize=10)
    _box(ax, (7.0, 4.0), 2.6, 1.2,
         "Two-layer NN\n" + r"$\sigma\!\left( \sum c_{ij}\,\sigma(\cdot) + \theta \right)$",
         fc=SOFT, ec=PURPLE, fontsize=10)

    _arrow(ax, (2.8, 4.6), (3.6, 4.6))
    _arrow(ax, (6.2, 4.6), (7.0, 4.6))

    ax.text(5.0, 2.5,
            r"$\sup_{u,\, y} \, \left| \mathcal{G}(u)(y) - "
            r"\sum_{k=1}^{p} b_k(u)\, t_k(y) \right| < \varepsilon$",
            ha="center", fontsize=12, color=INK)
    ax.text(5.0, 1.3,
            r"For ANY continuous operator $\mathcal{G}$ and $\varepsilon > 0$,"
            "\nthere exist branch outputs $b_k$ and trunk outputs $t_k$"
            " realising the bound.",
            ha="center", fontsize=10, color=MUTED, style="italic")
    ax.text(5.0, 0.2,
            r"$\Longrightarrow$  DeepONet (and FNO) can in principle approximate any continuous PDE solution map.",
            ha="center", fontsize=10.5, color=GREEN, weight="bold")

    # Right: error decay vs network width (illustrative)
    ax = axes[1]
    p = np.array([4, 8, 16, 32, 64, 128, 256])
    # approximation: ~1/sqrt(p), statistical: ~1/sqrt(N), noise floor
    approx = 1.5 / np.sqrt(p)
    stat = 0.4 / np.sqrt(p)
    noise = np.full_like(p, 0.02, dtype=float)
    total = np.sqrt(approx ** 2 + stat ** 2 + noise ** 2)

    ax.plot(p, approx, "o-", color=BLUE, lw=2, label=r"approximation  $\mathcal{O}(p^{-1/2})$")
    ax.plot(p, stat, "s-", color=PURPLE, lw=2, label=r"statistical  $\mathcal{O}(N^{-1/2})$")
    ax.plot(p, noise, "--", color=MUTED, lw=1.5, label="noise floor")
    ax.plot(p, total, "^-", color=RED, lw=2.2, label="total error")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Network width / basis size  $p$")
    ax.set_ylabel("Error (illustrative)")
    ax.set_title("Error decomposition: approximation + statistical + noise", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.4)

    fig.tight_layout()
    _save(fig, "fig7_universal_approx.png")


# ---------------------------------------------------------------------------
def main() -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)

    fig1_operator_concept()
    fig2_deeponet_architecture()
    fig3_fno_spectral_layer()
    fig4_resolution_invariance()
    fig5_method_comparison()
    fig6_geometry_application()
    fig7_universal_approx()

    print("Saved 7 figures to:")
    print(f"  {EN_DIR}")
    print(f"  {ZH_DIR}")


if __name__ == "__main__":
    main()
