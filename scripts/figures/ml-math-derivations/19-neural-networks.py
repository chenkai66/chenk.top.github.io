"""
Figure generation script for ML Math Derivations Part 19:
Neural Networks and Backpropagation.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure isolates one piece of the story so the math becomes visible at a
glance.

Figures:
    fig1_mlp_architecture       Multilayer perceptron architecture: input,
                                two hidden layers, output, with annotated
                                weight matrices and dimensions.
    fig2_forward_propagation    Forward pass diagram showing how a single
                                layer maps h^{(l-1)} -> z^{(l)} -> h^{(l)}
                                via W^{(l)}, b^{(l)}, and an activation.
    fig3_backprop_chain         Backpropagation gradient flow: arrows from
                                loss back through layers, illustrating the
                                recursive delta = (W^T delta_next) o sigma'.
    fig4_activations            Sigmoid / Tanh / ReLU / GELU / Swish curves
                                with their derivatives in a second panel.
    fig5_universal_approx       Universal approximation: a 1-hidden-layer
                                ReLU network fitting three target functions
                                (sin, |x|, step) trained with SGD.
    fig6_vanishing_exploding    Gradient norm vs depth for sigmoid (vanish),
                                tanh, ReLU (stable), and an unscaled-init
                                ReLU (explode), on a log scale.
    fig7_loss_landscape         A 3D loss landscape with a gradient-descent
                                trajectory over a non-convex surface.

Usage:
    python3 scripts/figures/ml-math-derivations/19-neural-networks.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the
    markdown image references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_RED = "#ef4444"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "en"
    / "ml-math-derivations"
    / "19-Neural-Networks-and-Backpropagation"
)
ZH_DIR = (
    REPO_ROOT
    / "source"
    / "_posts"
    / "zh"
    / "ml-math-derivations"
    / "19-神经网络与反向传播"
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
# Helpers
# ---------------------------------------------------------------------------
def _node(ax, xy, label, color, *, radius=0.22, text_color="white", fs=10):
    c = Circle(xy, radius, facecolor=color, edgecolor=C_DARK, lw=1.2, zorder=3)
    ax.add_patch(c)
    ax.text(
        xy[0], xy[1], label,
        ha="center", va="center",
        color=text_color, fontsize=fs, fontweight="bold", zorder=4,
    )


def _arrow(ax, p0, p1, color=C_GRAY, lw=0.8, alpha=0.6, style="-|>", mutation=10):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, color=color,
        lw=lw, alpha=alpha, mutation_scale=mutation, zorder=2,
    )
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Figure 1: MLP architecture
# ---------------------------------------------------------------------------
def fig1_mlp_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))

    layers = [4, 6, 6, 3]                 # neurons per layer
    layer_names = ["Input\n$\\mathbf{x}$", "Hidden 1\n$\\mathbf{h}^{(1)}$",
                   "Hidden 2\n$\\mathbf{h}^{(2)}$", "Output\n$\\hat{\\mathbf{y}}$"]
    colors = [C_GRAY, C_BLUE, C_PURPLE, C_GREEN]

    xs = np.linspace(0.8, 9.2, len(layers))

    # nodes
    positions: list[list[tuple[float, float]]] = []
    for li, n in enumerate(layers):
        ys = np.linspace(4.6, 0.4, n)
        col_pos = []
        for ni, y in enumerate(ys):
            _node(ax, (xs[li], y), str(ni + 1), colors[li], radius=0.22, fs=9)
            col_pos.append((xs[li], y))
        positions.append(col_pos)
        ax.text(xs[li], 5.3, layer_names[li],
                ha="center", va="bottom", fontsize=11, color=C_DARK, fontweight="bold")
        ax.text(xs[li], -0.2, f"$d_{li}={n}$",
                ha="center", va="top", fontsize=10, color=C_GRAY)

    # edges
    for li in range(len(layers) - 1):
        for p0 in positions[li]:
            for p1 in positions[li + 1]:
                _arrow(ax,
                       (p0[0] + 0.22, p0[1]),
                       (p1[0] - 0.22, p1[1]),
                       color=C_GRAY, lw=0.5, alpha=0.35, style="-", mutation=1)

    # weight labels between layers
    for li in range(len(layers) - 1):
        mid_x = (xs[li] + xs[li + 1]) / 2
        ax.text(mid_x, 5.05,
                f"$\\mathbf{{W}}^{{({li + 1})}}$"
                f" $\\in\\mathbb{{R}}^{{{layers[li + 1]}\\times{layers[li]}}}$",
                ha="center", va="center", fontsize=10,
                color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=C_LIGHT, lw=0.8))

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.8, 5.8)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Multilayer Perceptron Architecture",
                 fontsize=13, fontweight="bold", pad=10)

    _save(fig, "fig1_mlp_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Forward propagation flow
# ---------------------------------------------------------------------------
def fig2_forward_propagation() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 4.6))

    # Boxes for h^{(l-1)}, z^{(l)}, h^{(l)}
    boxes = [
        (1.2, "$\\mathbf{h}^{(l-1)}$", "Activations from\nprevious layer", C_GRAY),
        (4.4, "$\\mathbf{z}^{(l)}$",
         "$\\mathbf{z}^{(l)} = \\mathbf{W}^{(l)}\\mathbf{h}^{(l-1)} + \\mathbf{b}^{(l)}$",
         C_BLUE),
        (7.8, "$\\mathbf{h}^{(l)}$",
         "$\\mathbf{h}^{(l)} = \\sigma(\\mathbf{z}^{(l)})$", C_GREEN),
        (10.7, "$\\to$ next layer", "", C_PURPLE),
    ]

    for x, label, sub, color in boxes:
        box = FancyBboxPatch(
            (x - 0.95, 1.4), 1.9, 1.5,
            boxstyle="round,pad=0.05,rounding_size=0.18",
            facecolor=color, edgecolor=C_DARK, lw=1.2, alpha=0.92, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, 2.5, label, ha="center", va="center",
                color="white", fontsize=15, fontweight="bold", zorder=3)
        if sub:
            ax.text(x, 1.85, sub, ha="center", va="center",
                    color="white", fontsize=8.5, zorder=3)

    # arrows between
    centers = [b[0] for b in boxes]
    for c0, c1 in zip(centers[:-1], centers[1:]):
        _arrow(ax, (c0 + 0.95, 2.15), (c1 - 0.95, 2.15),
               color=C_DARK, lw=1.6, alpha=0.95, style="-|>", mutation=18)

    # operator labels above arrows
    op_labels = ["linear", "activation", ""]
    for i, lbl in enumerate(op_labels):
        if not lbl:
            continue
        mid = (centers[i] + centers[i + 1]) / 2
        ax.text(mid, 2.55, lbl, ha="center", va="bottom",
                fontsize=10, color=C_DARK, style="italic")

    # Header
    ax.text(6.0, 4.0,
            "Forward Propagation: one layer of computation",
            ha="center", va="center", fontsize=13, fontweight="bold",
            color=C_DARK)

    # Footer with stored quantities
    ax.text(6.0, 0.55,
            "Cache $\\mathbf{h}^{(l-1)}$ and $\\mathbf{z}^{(l)}$ "
            "during the forward pass — they are reused in backprop.",
            ha="center", va="center", fontsize=10,
            color=C_GRAY, style="italic")

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")

    _save(fig, "fig2_forward_propagation")


# ---------------------------------------------------------------------------
# Figure 3: Backprop chain rule
# ---------------------------------------------------------------------------
def fig3_backprop_chain() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.4))

    # Three "layers" of nodes plus a loss node, drawn from left to right
    xs = [1.2, 4.0, 6.8, 9.6, 11.6]
    labels = ["$\\mathbf{h}^{(l-1)}$", "$\\mathbf{z}^{(l)}$",
              "$\\mathbf{h}^{(l)}$", "$\\mathbf{z}^{(l+1)}$", "$\\mathcal{L}$"]
    colors = [C_GRAY, C_BLUE, C_GREEN, C_BLUE, C_AMBER]

    y = 2.5
    for x, lab, c in zip(xs, labels, colors):
        box = FancyBboxPatch(
            (x - 0.55, y - 0.45), 1.1, 0.9,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=c, edgecolor=C_DARK, lw=1.2, alpha=0.92, zorder=3,
        )
        ax.add_patch(box)
        ax.text(x, y, lab, ha="center", va="center",
                color="white", fontsize=13, fontweight="bold", zorder=4)

    # Forward arrows on top
    for x0, x1 in zip(xs[:-1], xs[1:]):
        _arrow(ax, (x0 + 0.55, y + 0.35), (x1 - 0.55, y + 0.35),
               color=C_GRAY, lw=1.0, alpha=0.6, style="-|>", mutation=12)
    ax.text((xs[0] + xs[-1]) / 2, y + 0.85, "forward pass",
            ha="center", va="center", fontsize=10, color=C_GRAY, style="italic")

    # Backward arrows below (red/amber)
    grad_labels = [
        "$\\mathbf{W}^{(l)\\,T}\\boldsymbol{\\delta}^{(l)}$",
        "$\\odot\\,\\sigma'(\\mathbf{z}^{(l)})$",
        "$\\mathbf{W}^{(l+1)\\,T}\\boldsymbol{\\delta}^{(l+1)}$",
        "$\\partial\\mathcal{L}/\\partial\\mathbf{z}^{(l+1)}$",
    ]
    for i, (x1, x0) in enumerate(zip(xs[:-1], xs[1:])):
        _arrow(ax, (x0 - 0.55, y - 0.35), (x1 + 0.55, y - 0.35),
               color=C_RED, lw=1.4, alpha=0.85, style="-|>", mutation=14)
        mid = (x0 + x1) / 2
        ax.text(mid, y - 0.95, grad_labels[i],
                ha="center", va="center", fontsize=9.5,
                color=C_RED,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=C_LIGHT, lw=0.6))
    ax.text((xs[0] + xs[-1]) / 2, y - 1.55,
            "backward pass — chain rule applied right to left",
            ha="center", va="center", fontsize=10, color=C_RED, style="italic")

    # Title and key formula
    ax.text(6.4, 4.5,
            "Backpropagation: gradients flow backward through the same graph",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_DARK)
    ax.text(6.4, 0.25,
            "Recurrence:  $\\boldsymbol{\\delta}^{(l)} = "
            "(\\mathbf{W}^{(l+1)\\,T}\\boldsymbol{\\delta}^{(l+1)})"
            "\\odot\\sigma'(\\mathbf{z}^{(l)})$"
            ",   "
            "$\\partial\\mathcal{L}/\\partial\\mathbf{W}^{(l)} = "
            "\\boldsymbol{\\delta}^{(l)}\\,\\mathbf{h}^{(l-1)\\,T}$",
            ha="center", va="center", fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor=C_GRAY, lw=0.8))

    ax.set_xlim(0, 13)
    ax.set_ylim(-0.2, 5.1)
    ax.set_aspect("equal")
    ax.axis("off")

    _save(fig, "fig3_backprop_chain")


# ---------------------------------------------------------------------------
# Figure 4: Activation functions
# ---------------------------------------------------------------------------
def fig4_activations() -> None:
    z = np.linspace(-5, 5, 401)

    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
    def dsigmoid(x):
        s = sigmoid(x); return s * (1 - s)

    def tanh(x): return np.tanh(x)
    def dtanh(x): return 1 - np.tanh(x) ** 2

    def relu(x): return np.maximum(0.0, x)
    def drelu(x): return (x > 0).astype(float)

    # GELU (tanh approximation)
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    def dgelu(x):
        # numerical derivative — accurate enough for plotting
        h = 1e-3
        return (gelu(x + h) - gelu(x - h)) / (2 * h)

    def swish(x): return x * sigmoid(x)
    def dswish(x):
        s = sigmoid(x); return s + x * s * (1 - s)

    funcs = [
        ("Sigmoid", sigmoid, dsigmoid, C_BLUE),
        ("Tanh", tanh, dtanh, C_PURPLE),
        ("ReLU", relu, drelu, C_GREEN),
        ("GELU", gelu, dgelu, C_AMBER),
        ("Swish", swish, dswish, C_RED),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    for name, f, _, c in funcs:
        ax.plot(z, f(z), color=c, lw=2.0, label=name)
    ax.axhline(0, color=C_GRAY, lw=0.6, ls="--")
    ax.axvline(0, color=C_GRAY, lw=0.6, ls="--")
    ax.set_xlabel("$z$"); ax.set_ylabel("$\\sigma(z)$")
    ax.set_title("Activation functions", fontsize=12, fontweight="bold")
    ax.set_ylim(-1.2, 3.5)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    ax = axes[1]
    for name, _, df, c in funcs:
        ax.plot(z, df(z), color=c, lw=2.0, label=name)
    ax.axhline(0, color=C_GRAY, lw=0.6, ls="--")
    ax.axvline(0, color=C_GRAY, lw=0.6, ls="--")
    ax.set_xlabel("$z$"); ax.set_ylabel("$\\sigma'(z)$")
    ax.set_title("Derivatives — gradient signal at each unit",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(-0.2, 1.2)
    ax.legend(loc="upper left", frameon=True, fontsize=9)

    fig.suptitle("Common activation functions and their derivatives",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    _save(fig, "fig4_activations")


# ---------------------------------------------------------------------------
# Figure 5: Universal approximation (3 targets)
# ---------------------------------------------------------------------------
def _train_relu_net(x, y, hidden=64, lr=0.02, epochs=4000, seed=0):
    """Tiny NumPy MLP: x -> hidden (ReLU) -> 1, trained with full-batch SGD."""
    rng = np.random.default_rng(seed)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    n_in, n_h = 1, hidden
    W1 = rng.standard_normal((n_h, n_in)) * np.sqrt(2.0 / n_in)
    b1 = np.zeros((n_h, 1))
    W2 = rng.standard_normal((1, n_h)) * np.sqrt(2.0 / n_h)
    b2 = np.zeros((1, 1))

    N = x.shape[0]
    for _ in range(epochs):
        z1 = x @ W1.T + b1.T          # (N, n_h)
        h1 = np.maximum(0, z1)
        yhat = h1 @ W2.T + b2.T        # (N, 1)
        err = (yhat - y) / N           # MSE gradient (mean)
        gW2 = err.T @ h1
        gb2 = err.sum(axis=0, keepdims=True).T
        dh1 = err @ W2
        dz1 = dh1 * (z1 > 0)
        gW1 = dz1.T @ x
        gb1 = dz1.sum(axis=0, keepdims=True).T
        W1 -= lr * gW1; b1 -= lr * gb1
        W2 -= lr * gW2; b2 -= lr * gb2

    def predict(xt):
        xt = xt.reshape(-1, 1)
        z1 = xt @ W1.T + b1.T
        h1 = np.maximum(0, z1)
        return (h1 @ W2.T + b2.T).ravel()
    return predict


def fig5_universal_approx() -> None:
    rng = np.random.default_rng(42)
    x_train = np.linspace(-3, 3, 120)
    x_plot = np.linspace(-3, 3, 400)

    targets = [
        ("$f(x) = \\sin(2x)$", np.sin(2 * x_train),  np.sin(2 * x_plot),  C_BLUE),
        ("$f(x) = |x|$",       np.abs(x_train),      np.abs(x_plot),      C_PURPLE),
        ("step function",      (x_train > 0).astype(float),
                               (x_plot > 0).astype(float),                C_GREEN),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (name, ytr, ypl, c) in zip(axes, targets):
        ytr_noisy = ytr + rng.normal(0, 0.05, size=ytr.shape)
        predict = _train_relu_net(x_train, ytr_noisy, hidden=64, lr=0.02,
                                  epochs=4000, seed=0)
        ax.scatter(x_train, ytr_noisy, color=C_GRAY, s=10, alpha=0.5,
                   label="training data")
        ax.plot(x_plot, ypl, color=c, lw=2.0, ls="--", label="target")
        ax.plot(x_plot, predict(x_plot), color=C_AMBER, lw=2.0, label="MLP fit")
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
        ax.legend(loc="best", fontsize=8.5)

    fig.suptitle("Universal approximation: a 1-hidden-layer ReLU MLP "
                 "(64 units) fits diverse targets",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    _save(fig, "fig5_universal_approx")


# ---------------------------------------------------------------------------
# Figure 6: Vanishing vs exploding gradients
# ---------------------------------------------------------------------------
def fig6_vanishing_exploding() -> None:
    """Track the L2 norm of the gradient at layer 1 as depth grows."""
    rng = np.random.default_rng(7)
    depths = np.arange(2, 41)
    width = 64
    n_trials = 5

    def grad_norm_at_input(depth, activation, init_scale, seed):
        """Forward random unit input through `depth` layers, then back-propagate
        a unit error signal to layer 1 and report ||grad||."""
        local_rng = np.random.default_rng(seed)
        h = local_rng.standard_normal(width)
        zs, hs = [], [h]
        Ws = []
        for _ in range(depth):
            W = local_rng.standard_normal((width, width)) * init_scale
            Ws.append(W)
            z = W @ hs[-1]
            zs.append(z)
            hs.append(activation(z))

        # Back-propagate a unit error signal at the output
        delta = np.ones(width)
        # multiply by activation derivative at output layer
        for li in range(depth - 1, -1, -1):
            z = zs[li]
            if activation is sigmoid_np:
                d = sigmoid_np(z) * (1 - sigmoid_np(z))
            elif activation is tanh_np:
                d = 1 - np.tanh(z) ** 2
            else:  # relu
                d = (z > 0).astype(float)
            delta = delta * d
            if li > 0:
                delta = Ws[li].T @ delta
        return np.linalg.norm(delta)

    def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-x))
    def tanh_np(x): return np.tanh(x)
    def relu_np(x): return np.maximum(0.0, x)

    # Re-bind for use inside grad_norm_at_input
    globals()["sigmoid_np"] = sigmoid_np
    globals()["tanh_np"] = tanh_np
    globals()["relu_np"] = relu_np

    settings = [
        ("Sigmoid (Xavier init)", sigmoid_np,
         np.sqrt(1.0 / width), C_BLUE, "vanishes"),
        ("Tanh (Xavier init)", tanh_np,
         np.sqrt(1.0 / width), C_PURPLE, "vanishes more slowly"),
        ("ReLU (He init)", relu_np,
         np.sqrt(2.0 / width), C_GREEN, "stable"),
        ("ReLU (no scaling)", relu_np,
         1.0, C_RED, "explodes"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for name, act, scale, color, _ in settings:
        means = []
        for d in depths:
            ts = [grad_norm_at_input(d, act, scale, seed=int(rng.integers(0, 1 << 31)))
                  for _ in range(n_trials)]
            means.append(np.mean(ts))
        ax.plot(depths, means, color=color, lw=2.0, marker="o",
                ms=3.5, label=name)

    ax.set_yscale("log")
    ax.set_xlabel("Network depth $L$")
    ax.set_ylabel("Gradient norm at layer 1 (log scale)")
    ax.set_title("Vanishing vs. exploding gradients across depth",
                 fontsize=12.5, fontweight="bold")
    ax.legend(loc="best", fontsize=10, frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(1.0, color=C_GRAY, lw=0.8, ls=":", alpha=0.7)
    ax.text(depths[-1], 1.0, " healthy regime", va="center", ha="left",
            fontsize=9, color=C_GRAY)

    _save(fig, "fig6_vanishing_exploding")


# ---------------------------------------------------------------------------
# Figure 7: 3D loss landscape
# ---------------------------------------------------------------------------
def fig7_loss_landscape() -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

    # Non-convex landscape: two basins + a saddle
    def loss(w1, w2):
        return (
            0.6 * (w1 ** 2 + w2 ** 2) * 0.05
            + np.exp(-((w1 - 1.5) ** 2 + (w2 - 1.0) ** 2) / 0.6) * (-0.9)
            + np.exp(-((w1 + 1.7) ** 2 + (w2 + 1.2) ** 2) / 0.7) * (-0.7)
            + np.exp(-((w1 - 0.0) ** 2 + (w2 + 1.4) ** 2) / 0.4) * 0.4
            + np.sin(1.6 * w1) * np.cos(1.6 * w2) * 0.07
        )

    grid = np.linspace(-3, 3, 120)
    W1, W2 = np.meshgrid(grid, grid)
    Z = loss(W1, W2)

    # Numerical gradient descent trajectory from a poor init
    def numgrad(w1, w2, h=1e-3):
        gw1 = (loss(w1 + h, w2) - loss(w1 - h, w2)) / (2 * h)
        gw2 = (loss(w1, w2 + h) - loss(w1, w2 - h)) / (2 * h)
        return gw1, gw2

    traj = [(-2.6, 2.2)]
    lr = 0.45
    for _ in range(60):
        w1, w2 = traj[-1]
        gw1, gw2 = numgrad(w1, w2)
        traj.append((w1 - lr * gw1, w2 - lr * gw2))
    traj = np.array(traj)
    traj_z = np.array([loss(p[0], p[1]) for p in traj]) + 0.02

    fig = plt.figure(figsize=(11, 5.6))

    # Left: 3D surface with trajectory
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot_surface(W1, W2, Z, cmap="viridis", alpha=0.85,
                     linewidth=0, antialiased=True, rstride=2, cstride=2)
    ax3.plot(traj[:, 0], traj[:, 1], traj_z,
             color=C_RED, lw=2.2, marker="o", ms=3, label="SGD path")
    ax3.scatter(traj[0, 0], traj[0, 1], traj_z[0],
                color=C_AMBER, s=60, label="start", zorder=10)
    ax3.scatter(traj[-1, 0], traj[-1, 1], traj_z[-1],
                color=C_GREEN, s=60, label="end", zorder=10)
    ax3.set_xlabel("$w_1$"); ax3.set_ylabel("$w_2$")
    ax3.set_zlabel("$\\mathcal{L}(w_1, w_2)$")
    ax3.set_title("Loss surface (3D)", fontsize=11, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3.view_init(elev=35, azim=-58)

    # Right: top-down contour with trajectory
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(W1, W2, Z, levels=22, cmap="viridis", alpha=0.9)
    ax2.contour(W1, W2, Z, levels=10, colors="white", linewidths=0.5, alpha=0.55)
    ax2.plot(traj[:, 0], traj[:, 1], color=C_RED, lw=2.0, marker="o", ms=3)
    ax2.scatter(traj[0, 0], traj[0, 1], color=C_AMBER, s=80, zorder=10,
                edgecolor="white", lw=1.2, label="start")
    ax2.scatter(traj[-1, 0], traj[-1, 1], color=C_GREEN, s=80, zorder=10,
                edgecolor="white", lw=1.2, label="end (local min)")
    ax2.set_xlabel("$w_1$"); ax2.set_ylabel("$w_2$")
    ax2.set_title("Contour view + SGD trajectory", fontsize=11, fontweight="bold")
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right", fontsize=9)
    fig.colorbar(cs, ax=ax2, fraction=0.045, pad=0.04, label="$\\mathcal{L}$")

    fig.suptitle("Non-convex loss landscape of a small neural network",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    _save(fig, "fig7_loss_landscape")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 19 (Neural Networks) figures ...")
    fig1_mlp_architecture()
    fig2_forward_propagation()
    fig3_backprop_chain()
    fig4_activations()
    fig5_universal_approx()
    fig6_vanishing_exploding()
    fig7_loss_landscape()
    print("Done.")


if __name__ == "__main__":
    main()
