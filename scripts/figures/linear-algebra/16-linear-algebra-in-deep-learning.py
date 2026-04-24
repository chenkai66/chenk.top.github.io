"""
Figure generation script for Linear Algebra Chapter 16: Linear Algebra in
Deep Learning.

Generates 7 figures used in both EN and ZH versions of the article. Each
figure illustrates one core idea cleanly, in 3Blue1Brown-flavoured style.

Figures:
    fig1_network_as_matmul       Forward pass of a 3-layer MLP visualised as a
                                 chain of matrix multiplications with shape
                                 annotations - the network IS the matmul chain.
    fig2_weight_heatmap          Heatmap of a trained weight matrix
                                 (digit-classifier first layer) showing
                                 learned receptive fields plus the row-norm
                                 distribution.
    fig3_im2col                  How a 2D convolution becomes a single GEMM:
                                 patches unfolded into columns, kernel
                                 flattened to rows, output reshaped.
    fig4_attention               Scaled dot-product attention, four panels:
                                 Q.K^T scores, /sqrt(d_k), softmax weights,
                                 weights . V output.
    fig5_backprop_flow           Gradient flow through a deep MLP - forward
                                 activations and backward gradients side by
                                 side, showing how W^T transports the signal.
    fig6_init_eigenvalues        Singular value spectra of products of weight
                                 matrices for naive vs Xavier vs He init,
                                 demonstrating gradient explosion / vanishing.
    fig7_lora_decomposition      Low-rank adaptation: full Delta W versus the
                                 BA factorisation, with parameter-count and
                                 reconstruction-error bars across ranks.

Usage:
    python3 scripts/figures/linear-algebra/16-linear-algebra-in-deep-learning.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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

DPI = 150
RNG = np.random.default_rng(20240428)

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/linear-algebra/"
    "16-linear-algebra-in-deep-learning"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/linear-algebra/"
    "16-深度学习中的线性代数"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to BOTH EN and ZH asset folders as <name>.png."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Custom colormaps for matrices
# ---------------------------------------------------------------------------
CMAP_BLUE = LinearSegmentedColormap.from_list(
    "blueheat", ["#ffffff", "#dbeafe", "#93c5fd", "#3b82f6", "#1e3a8a"]
)
CMAP_DIVERGE = LinearSegmentedColormap.from_list(
    "diverge", ["#1e3a8a", "#93c5fd", "#ffffff", "#fcd34d", "#b45309"]
)


# ---------------------------------------------------------------------------
# Figure 1: Network as a chain of matrix multiplications
# ---------------------------------------------------------------------------
def fig1_network_as_matmul() -> None:
    """3-layer MLP as a chain of matmuls with annotated shapes."""
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # Layer specs: (label, height, width, x-center, y-center, color, kind)
    boxes = [
        ("x", 3.4, 0.55, 0.7, 2.6, C_GRAY, "vec"),
        ("W1", 2.4, 1.6, 2.0, 2.6, C_BLUE, "mat"),
        ("h1", 2.4, 0.55, 3.4, 2.6, C_PURPLE, "vec"),
        ("W2", 1.6, 2.4, 4.7, 2.6, C_BLUE, "mat"),
        ("h2", 1.6, 0.55, 6.2, 2.6, C_PURPLE, "vec"),
        ("W3", 0.8, 1.6, 7.45, 2.6, C_BLUE, "mat"),
        ("y", 0.8, 0.55, 8.7, 2.6, C_GREEN, "vec"),
    ]

    shapes = ["784", "256x784", "256", "128x256", "128", "10x128", "10"]

    for (label, h, w, cx, cy, color, kind), shape in zip(boxes, shapes):
        x0 = cx - w / 2
        y0 = cy - h / 2
        rect = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.4, edgecolor=color,
            facecolor=color, alpha=0.18 if kind == "mat" else 0.32,
        )
        ax.add_patch(rect)
        ax.text(cx, cy + 0.15, label, ha="center", va="center",
                fontsize=15, fontweight="bold", color=C_DARK,
                fontstyle="italic")
        ax.text(cx, y0 - 0.25, shape, ha="center", va="top",
                fontsize=10, color=C_DARK, family="monospace")

    # Operation labels between blocks
    op_positions = [
        (1.35, 4.2, r"$\sigma(W_1\,\mathbf{x}+b_1)$"),
        (4.05, 4.2, r"$\sigma(W_2\,\mathbf{h}_1+b_2)$"),
        (6.85, 4.2, r"$\sigma(W_3\,\mathbf{h}_2+b_3)$"),
    ]
    for x, y, s in op_positions:
        ax.text(x, y, s, ha="center", va="center", fontsize=11,
                color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor=C_LIGHT, edgecolor=C_GRAY, linewidth=0.8))

    # Arrows between blocks
    arrow_pairs = [(0.97, 1.2), (2.8, 3.12), (3.67, 3.5),
                   (5.9, 5.94), (6.47, 6.65), (8.25, 8.42)]
    for x1, x2 in arrow_pairs:
        ax.add_patch(FancyArrowPatch(
            (x1, 2.6), (x2, 2.6), arrowstyle="->", mutation_scale=14,
            color=C_DARK, linewidth=1.3,
        ))

    # Bottom annotation
    ax.text(4.7, 0.55,
            "Each layer is one matrix-vector product followed by an "
            "elementwise nonlinearity.",
            ha="center", fontsize=10.5, color=C_DARK, style="italic")

    # Right side: batched form
    ax.text(10.4, 4.2, "Batch form", ha="center", fontsize=11,
            fontweight="bold", color=C_DARK)
    ax.add_patch(FancyBboxPatch(
        (9.4, 1.6), 2.0, 2.2,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        linewidth=1.0, edgecolor=C_GRAY, facecolor="white"))
    ax.text(10.4, 3.3, r"$X \in \mathbb{R}^{B\times d}$",
            ha="center", fontsize=11, color=C_DARK)
    ax.text(10.4, 2.7,
            r"$H = \sigma(X W^{\!\top}\!+\!\mathbf{1}b^{\!\top})$",
            ha="center", fontsize=11, color=C_BLUE)
    ax.text(10.4, 2.05,
            "GPU sees one giant\nGEMM per layer",
            ha="center", fontsize=9.5, color=C_DARK)

    fig.suptitle("A neural network IS a chain of matrix multiplications",
                 fontsize=14, color=C_DARK, y=0.97)
    save(fig, "fig1_network_as_matmul")


# ---------------------------------------------------------------------------
# Figure 2: Weight matrix heatmap (toy trained weights)
# ---------------------------------------------------------------------------
def fig2_weight_heatmap() -> None:
    """Heatmap of a 'trained' weight matrix - synthesised receptive fields."""
    n_neurons = 16
    img_size = 14  # 14x14 receptive fields, easier to read than 28x28

    # Synthesise a weight matrix whose ROWS look like trained digit features:
    # mixture of oriented Gabors, blob detectors and pen-stroke templates.
    rows = []
    rng = np.random.default_rng(20240428)
    coords = np.linspace(-1, 1, img_size)
    xx, yy = np.meshgrid(coords, coords)

    for i in range(n_neurons):
        if i % 3 == 0:
            theta = rng.uniform(0, np.pi)
            freq = rng.uniform(2.0, 4.5)
            sigma = rng.uniform(0.35, 0.55)
            x_rot = xx * np.cos(theta) + yy * np.sin(theta)
            y_rot = -xx * np.sin(theta) + yy * np.cos(theta)
            patch = np.exp(-(x_rot ** 2 + y_rot ** 2) / (2 * sigma ** 2)) \
                    * np.cos(2 * np.pi * freq * x_rot)
        elif i % 3 == 1:
            cx, cy = rng.uniform(-0.4, 0.4, 2)
            sigma = rng.uniform(0.3, 0.5)
            patch = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2)
                           / (2 * sigma ** 2))
            if rng.random() < 0.5:
                patch = -patch
        else:
            angle = rng.uniform(0, 2 * np.pi)
            stroke = np.cos(angle) * xx + np.sin(angle) * yy
            patch = np.tanh(3 * stroke) * np.exp(-(xx ** 2 + yy ** 2) / 0.6)
        patch += 0.15 * rng.standard_normal(patch.shape)
        patch /= np.max(np.abs(patch)) + 1e-9
        rows.append(patch.flatten())

    W = np.stack(rows)  # (16, 196)

    fig = plt.figure(figsize=(13, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1.0, 0.95],
                          wspace=0.32)

    # Panel A: full weight matrix as a long heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(W, cmap=CMAP_DIVERGE, vmin=-1, vmax=1, aspect="auto")
    ax1.set_xlabel("Input pixel index (flattened 14x14)")
    ax1.set_ylabel("Hidden neuron index")
    ax1.set_title("Weight matrix W (16 x 196)", color=C_DARK, fontsize=12)
    ax1.set_xticks([0, 49, 98, 147, 195])
    ax1.set_yticks(range(0, 16, 2))
    cbar = fig.colorbar(im, ax=ax1, fraction=0.05, pad=0.02)
    cbar.set_label("Weight value", fontsize=9)

    # Panel B: each row reshaped as a receptive field grid
    ax2 = fig.add_subplot(gs[0, 1])
    grid = np.zeros((4 * img_size, 4 * img_size))
    for k in range(n_neurons):
        r, c = divmod(k, 4)
        grid[r * img_size:(r + 1) * img_size,
             c * img_size:(c + 1) * img_size] = W[k].reshape(img_size, img_size)
    ax2.imshow(grid, cmap=CMAP_DIVERGE, vmin=-1, vmax=1)
    for k in range(1, 4):
        ax2.axhline(k * img_size - 0.5, color="white", linewidth=1.2)
        ax2.axvline(k * img_size - 0.5, color="white", linewidth=1.2)
    ax2.set_title("Each row reshaped: learned receptive fields",
                  color=C_DARK, fontsize=12)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Panel C: row-norm distribution and weight value histogram
    ax3 = fig.add_subplot(gs[0, 2])
    row_norms = np.linalg.norm(W, axis=1)
    ax3.bar(range(n_neurons), row_norms, color=C_BLUE, alpha=0.85,
            edgecolor="white", linewidth=0.6)
    ax3.set_xlabel("Neuron index")
    ax3.set_ylabel(r"$\|W_{i,:}\|_2$")
    ax3.set_title("Row norms: which neurons are 'loud'",
                  color=C_DARK, fontsize=12)
    ax3.set_xticks(range(0, 16, 2))

    fig.suptitle("Reading a weight matrix: rows are the neurons' filters",
                 fontsize=14, color=C_DARK, y=1.02)
    save(fig, "fig2_weight_heatmap")


# ---------------------------------------------------------------------------
# Figure 3: im2col - convolution as GEMM
# ---------------------------------------------------------------------------
def fig3_im2col() -> None:
    """Visualise im2col: input -> column matrix; kernel -> row; matmul."""
    # Tiny example: 1 channel 5x5 input, 3x3 kernel, stride 1, no pad -> 3x3 out
    H = W_in = 5
    K = 3
    out_h = out_w = H - K + 1
    img = np.array([
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 1],
        [2, 3, 4, 5, 6],
        [7, 8, 9, 1, 2],
        [3, 4, 5, 6, 7],
    ], dtype=float)

    # Build im2col matrix (K*K, out_h*out_w)
    cols = np.zeros((K * K, out_h * out_w))
    for i in range(out_h):
        for j in range(out_w):
            cols[:, i * out_w + j] = img[i:i + K, j:j + K].flatten()
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
    krow = kernel.flatten()[None, :]  # (1, 9)
    out_flat = krow @ cols  # (1, 9)
    out_map = out_flat.reshape(out_h, out_w)

    fig = plt.figure(figsize=(13, 6.0))
    gs = fig.add_gridspec(2, 4,
                          width_ratios=[1.0, 0.9, 1.5, 1.0],
                          height_ratios=[1, 0.55],
                          hspace=0.35, wspace=0.45)

    # A: input image with one patch highlighted
    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(img, cmap=CMAP_BLUE, vmin=0, vmax=9)
    for i in range(H + 1):
        axA.axhline(i - 0.5, color="white", linewidth=0.8)
        axA.axvline(i - 0.5, color="white", linewidth=0.8)
    # highlight top-left 3x3 patch
    axA.add_patch(plt.Rectangle((-0.5, -0.5), K, K,
                                fill=False, edgecolor=C_AMBER, linewidth=2.6))
    for i in range(H):
        for j in range(W_in):
            axA.text(j, i, int(img[i, j]), ha="center", va="center",
                     color=C_DARK, fontsize=9)
    axA.set_title("Input (5x5)\npatch shown in amber",
                  color=C_DARK, fontsize=11)
    axA.set_xticks([]); axA.set_yticks([])

    # B: kernel flattened to row
    axB = fig.add_subplot(gs[0, 1])
    axB.imshow(kernel, cmap=CMAP_DIVERGE, vmin=-1, vmax=1)
    for i in range(K):
        for j in range(K):
            axB.text(j, i, int(kernel[i, j]), ha="center", va="center",
                     color=C_DARK, fontsize=11, fontweight="bold")
    axB.set_title("Kernel (3x3)", color=C_DARK, fontsize=11)
    axB.set_xticks([]); axB.set_yticks([])

    # C: im2col matrix
    axC = fig.add_subplot(gs[0, 2])
    im = axC.imshow(cols, cmap=CMAP_BLUE, vmin=0, vmax=9, aspect="auto")
    axC.set_title("im2col matrix (9 x 9): each column = one patch",
                  color=C_DARK, fontsize=11)
    axC.set_xlabel("Output position (row-major)")
    axC.set_ylabel("Patch entries")
    # highlight first column = first patch
    axC.add_patch(plt.Rectangle((-0.5, -0.5), 1, K * K,
                                fill=False, edgecolor=C_AMBER, linewidth=2.4))
    for i in range(K * K):
        for j in range(out_h * out_w):
            axC.text(j, i, int(cols[i, j]), ha="center", va="center",
                     color=C_DARK, fontsize=7)
    axC.set_xticks(range(out_h * out_w))
    axC.set_yticks(range(K * K))

    # D: output feature map
    axD = fig.add_subplot(gs[0, 3])
    axD.imshow(out_map, cmap=CMAP_DIVERGE,
               vmin=-np.abs(out_map).max(), vmax=np.abs(out_map).max())
    for i in range(out_h):
        for j in range(out_w):
            axD.text(j, i, f"{out_map[i, j]:.0f}",
                     ha="center", va="center", color=C_DARK, fontsize=10)
    axD.set_title("Output (3x3) = reshape of\n"
                  "kernel_row x im2col matrix",
                  color=C_DARK, fontsize=11)
    axD.set_xticks([]); axD.set_yticks([])

    # Bottom: equation strip
    axEq = fig.add_subplot(gs[1, :])
    axEq.axis("off")
    axEq.text(0.5, 0.65,
              r"$\mathbf{Y}_{\mathrm{flat}}\;=\;"
              r"\mathbf{w}_{\mathrm{row}}\,(1\times K^2)"
              r"\;\cdot\;\mathbf{X}_{\mathrm{col}}\,(K^2\times HW)"
              r"\;\Rightarrow\;\mathrm{reshape}\;\to\;\mathbf{Y}$",
              ha="center", va="center", fontsize=14, color=C_DARK)
    axEq.text(0.5, 0.18,
              "Convolution becomes a single GEMM - the workhorse "
              "every BLAS library is tuned for.",
              ha="center", va="center", fontsize=10.5, color=C_DARK,
              style="italic")

    fig.suptitle("im2col: turning convolution into one matrix multiplication",
                 fontsize=14, color=C_DARK, y=0.99)
    save(fig, "fig3_im2col")


# ---------------------------------------------------------------------------
# Figure 4: Attention as Q.K^T -> softmax -> .V
# ---------------------------------------------------------------------------
def fig4_attention() -> None:
    """Step-by-step scaled dot-product attention on a toy sequence."""
    n = 6  # sequence length
    d_k = 4
    rng = np.random.default_rng(7)

    # Seed Q and K so a clear pattern emerges
    base = rng.standard_normal((n, d_k))
    Q = base + 0.3 * rng.standard_normal((n, d_k))
    K = base + 0.3 * rng.standard_normal((n, d_k))
    # Boost the diagonal-ish similarity
    K = 0.6 * K + 0.4 * Q
    V = rng.standard_normal((n, d_k))

    scores = Q @ K.T
    scaled = scores / np.sqrt(d_k)
    # stable softmax
    e = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    weights = e / e.sum(axis=1, keepdims=True)
    out = weights @ V

    tokens = ["the", "cat", "sat", "on", "the", "mat"]

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4))

    panels = [
        (scores, r"$QK^{\!\top}$ (raw scores)", CMAP_DIVERGE,
         "Token (key)", "Token (query)", True),
        (scaled, r"$QK^{\!\top}/\sqrt{d_k}$ (scaled)", CMAP_DIVERGE,
         "Token (key)", "", True),
        (weights, "softmax(.) - attention weights", CMAP_BLUE,
         "Token (key)", "", False),
        (out, "weights . V (output)", CMAP_DIVERGE,
         "Feature", "", True),
    ]

    for ax, (M, title, cmap, xlab, ylab, sym) in zip(axes, panels):
        if sym:
            v = np.abs(M).max()
            im = ax.imshow(M, cmap=cmap, vmin=-v, vmax=v, aspect="auto")
        else:
            im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if (sym and abs(val) > 0.55 * v) or
                                         (not sym and val > 0.55)
                        else C_DARK)
        ax.set_title(title, color=C_DARK, fontsize=11)
        ax.set_xlabel(xlab, fontsize=9.5)
        if ylab:
            ax.set_ylabel(ylab, fontsize=9.5)
        if M.shape[1] == n:
            ax.set_xticks(range(n))
            ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8.5)
        else:
            ax.set_xticks(range(M.shape[1]))
        ax.set_yticks(range(n))
        ax.set_yticklabels(tokens, fontsize=8.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    fig.suptitle(r"Scaled dot-product attention: "
                 r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}"
                 r"(QK^{\!\top}/\sqrt{d_k})\,V$",
                 fontsize=13.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_attention")


# ---------------------------------------------------------------------------
# Figure 5: Backpropagation gradient flow
# ---------------------------------------------------------------------------
def fig5_backprop_flow() -> None:
    """Forward activations and backward gradients side-by-side for a deep MLP."""
    # Build a deep network and run one forward + backward pass
    L = 6  # number of layers
    widths = [64, 64, 64, 64, 64, 64, 64]  # constant width
    rng = np.random.default_rng(2024)

    # Three init schemes
    schemes = {
        "He init (good)": lambda fin, fout: rng.standard_normal((fout, fin))
                                            * np.sqrt(2.0 / fin),
        "Naive N(0,1) (explodes)": lambda fin, fout:
            rng.standard_normal((fout, fin)),
        "Tiny N(0,0.01) (vanishes)": lambda fin, fout:
            rng.standard_normal((fout, fin)) * 0.01,
    }
    colors = [C_GREEN, C_AMBER, C_PURPLE]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    for (name, init), col in zip(schemes.items(), colors):
        Ws = [init(widths[i], widths[i + 1]) for i in range(L)]
        x = rng.standard_normal(widths[0])
        # forward (linear, no activation, to isolate the matrix effect)
        acts = [x]
        for W in Ws:
            acts.append(W @ acts[-1])
        act_norms = [np.linalg.norm(a) for a in acts]

        # backward: assume grad at output = unit vector
        grad = rng.standard_normal(widths[-1])
        grad /= np.linalg.norm(grad)
        grad_norms = [np.linalg.norm(grad)]
        for W in reversed(Ws):
            grad = W.T @ grad
            grad_norms.append(np.linalg.norm(grad))
        grad_norms = list(reversed(grad_norms))

        axes[0].plot(range(L + 1), act_norms, marker="o", linewidth=2.0,
                     color=col, label=name)
        axes[1].plot(range(L + 1), grad_norms, marker="o", linewidth=2.0,
                     color=col, label=name)

    for ax, title in zip(axes,
                         ["Forward pass: activation norm by layer",
                          "Backward pass: gradient norm by layer"]):
        ax.set_yscale("log")
        ax.set_xlabel("Layer index")
        ax.set_ylabel("Norm (log scale)")
        ax.set_title(title, color=C_DARK, fontsize=12)
        ax.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1.0,
                   alpha=0.7)
        ax.legend(loc="best", fontsize=9, framealpha=0.92)

    fig.suptitle("Gradient flow through a 6-layer linear network: "
                 r"$\nabla_{\mathbf{x}} = W_1^{\!\top} W_2^{\!\top}"
                 r"\cdots W_L^{\!\top}\, \nabla_{\mathbf{y}}$",
                 fontsize=13.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_backprop_flow")


# ---------------------------------------------------------------------------
# Figure 6: Init schemes - eigenvalue / singular value distributions
# ---------------------------------------------------------------------------
def fig6_init_eigenvalues() -> None:
    """Singular value spectra of W and W^L for naive vs Xavier vs He."""
    n = 256       # matrix size
    L = 5         # depth for product
    rng = np.random.default_rng(11)

    schemes = [
        ("Naive N(0, 1)",
         lambda: rng.standard_normal((n, n)),
         C_AMBER),
        ("Xavier (tanh)",
         lambda: rng.standard_normal((n, n)) * np.sqrt(1.0 / n),
         C_BLUE),
        ("He (ReLU)",
         lambda: rng.standard_normal((n, n)) * np.sqrt(2.0 / n),
         C_GREEN),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Left: singular values of a single W
    axL = axes[0]
    for label, sampler, col in schemes:
        W = sampler()
        sv = np.linalg.svd(W, compute_uv=False)
        axL.hist(sv, bins=50, density=True, color=col, alpha=0.45,
                 edgecolor="white", linewidth=0.4, label=label)
    axL.set_xlabel(r"Singular value $\sigma$ of $W$")
    axL.set_ylabel("Density")
    axL.set_title(f"Single-layer spectrum (n = {n})",
                  color=C_DARK, fontsize=12)
    axL.legend(loc="upper right", fontsize=9, framealpha=0.92)
    axL.set_xlim(0, None)

    # Right: largest singular value of W_L ... W_1 vs depth
    axR = axes[1]
    depths = list(range(1, L + 1))
    for label, sampler, col in schemes:
        max_sv_by_depth = []
        prod = np.eye(n)
        for _ in depths:
            prod = sampler() @ prod
            max_sv_by_depth.append(np.linalg.svd(prod, compute_uv=False)[0])
        axR.plot(depths, max_sv_by_depth, marker="o", linewidth=2.0,
                 color=col, label=label)
    axR.set_yscale("log")
    axR.set_xlabel("Depth (number of W factors)")
    axR.set_ylabel(r"$\sigma_{\max}(W_L\cdots W_1)$  (log)")
    axR.set_title("Top singular value of the product",
                  color=C_DARK, fontsize=12)
    axR.axhline(1.0, color=C_GRAY, linestyle="--", linewidth=1.0, alpha=0.7)
    axR.legend(loc="best", fontsize=9, framealpha=0.92)

    fig.suptitle("Why initialization matters: singular value distributions",
                 fontsize=13.5, color=C_DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig6_init_eigenvalues")


# ---------------------------------------------------------------------------
# Figure 7: LoRA - low-rank decomposition
# ---------------------------------------------------------------------------
def fig7_lora_decomposition() -> None:
    """Visualise W = W0 + B A and the rank/parameter trade-off."""
    d_in = d_out = 64
    # Build a 'true' low-rank update so the picture is not noise
    true_rank = 8
    rng = np.random.default_rng(42)
    B_true = rng.standard_normal((d_out, true_rank)) * 0.5
    A_true = rng.standard_normal((true_rank, d_in)) * 0.5
    DeltaW = B_true @ A_true

    # LoRA factorisation at rank r=8 (matches true rank for clean visual)
    r_show = 8
    U, S, Vt = np.linalg.svd(DeltaW, full_matrices=False)
    B_lora = U[:, :r_show] * S[:r_show]
    A_lora = Vt[:r_show, :]

    fig = plt.figure(figsize=(13.5, 5.6))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.4, 0.25, 0.5, 0.5, 1.4],
                          height_ratios=[1, 0.55],
                          hspace=0.35, wspace=0.25)

    vmax = np.abs(DeltaW).max()

    # Left: full Delta W
    axA = fig.add_subplot(gs[0, 0])
    axA.imshow(DeltaW, cmap=CMAP_DIVERGE, vmin=-vmax, vmax=vmax)
    axA.set_title(r"$\Delta W \in \mathbb{R}^{64\times 64}$"
                  "\n(4096 params)",
                  color=C_DARK, fontsize=11)
    axA.set_xticks([0, 31, 63]); axA.set_yticks([0, 31, 63])

    # equals sign
    axEq = fig.add_subplot(gs[0, 1]); axEq.axis("off")
    axEq.text(0.5, 0.5, "=", ha="center", va="center",
              fontsize=28, color=C_DARK)

    # B (tall thin)
    axB = fig.add_subplot(gs[0, 2])
    axB.imshow(B_lora, cmap=CMAP_DIVERGE, vmin=-np.abs(B_lora).max(),
               vmax=np.abs(B_lora).max(), aspect="auto")
    axB.set_title(r"$B\in\mathbb{R}^{64\times 8}$"
                  "\n(512 params)",
                  color=C_DARK, fontsize=11)
    axB.set_xticks([]); axB.set_yticks([])

    # multiplication dot
    axMul = fig.add_subplot(gs[0, 3]); axMul.axis("off")
    axMul.text(0.5, 0.5, r"$\times$", ha="center", va="center",
               fontsize=22, color=C_DARK)

    # A (short wide)
    axAw = fig.add_subplot(gs[0, 4])
    axAw.imshow(A_lora, cmap=CMAP_DIVERGE, vmin=-np.abs(A_lora).max(),
                vmax=np.abs(A_lora).max(), aspect="auto")
    axAw.set_title(r"$A\in\mathbb{R}^{8\times 64}$"
                   "\n(512 params)",
                   color=C_DARK, fontsize=11)
    axAw.set_xticks([]); axAw.set_yticks([])

    # Bottom: parameter count and reconstruction error vs rank
    axBar = fig.add_subplot(gs[1, :2])
    ranks = [1, 2, 4, 8, 16, 32, 64]
    full_params = d_in * d_out
    lora_params = [r * (d_in + d_out) for r in ranks]
    bars = axBar.bar([str(r) for r in ranks], lora_params,
                     color=C_BLUE, alpha=0.85, edgecolor="white")
    axBar.axhline(full_params, color=C_AMBER, linestyle="--",
                  linewidth=1.4,
                  label=f"Full: {full_params}")
    axBar.set_xlabel("LoRA rank r")
    axBar.set_ylabel("Trainable params")
    axBar.set_title(r"Param count: $r(d_{\rm in}+d_{\rm out})$ vs "
                    r"$d_{\rm in}d_{\rm out}$",
                    color=C_DARK, fontsize=11)
    axBar.legend(loc="upper left", fontsize=9)

    axErr = fig.add_subplot(gs[1, 2:])
    errors = []
    for r in ranks:
        Wr = U[:, :r] * S[:r] @ Vt[:r, :]
        errors.append(np.linalg.norm(DeltaW - Wr) /
                      np.linalg.norm(DeltaW))
    axErr.plot(ranks, errors, marker="o", color=C_GREEN, linewidth=2.0)
    axErr.axvline(true_rank, color=C_AMBER, linestyle="--",
                  linewidth=1.2,
                  label=f"True rank = {true_rank}")
    axErr.set_xlabel("LoRA rank r")
    axErr.set_ylabel(r"$\|\Delta W - BA\|_F / \|\Delta W\|_F$")
    axErr.set_title("Reconstruction error vs rank",
                    color=C_DARK, fontsize=11)
    axErr.set_xscale("log", base=2)
    axErr.set_xticks(ranks)
    axErr.set_xticklabels([str(r) for r in ranks])
    axErr.legend(loc="upper right", fontsize=9)

    fig.suptitle(r"LoRA: $W' = W_0 + BA$ - a low-rank update with "
                 "orders of magnitude fewer parameters",
                 fontsize=13.5, color=C_DARK, y=1.02)
    save(fig, "fig7_lora_decomposition")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_network_as_matmul()
    fig2_weight_heatmap()
    fig3_im2col()
    fig4_attention()
    fig5_backprop_flow()
    fig6_init_eigenvalues()
    fig7_lora_decomposition()
    print("All figures generated for chapter 16.")


if __name__ == "__main__":
    main()
