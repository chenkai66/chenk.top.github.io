"""
Figure generation for Transfer Learning Part 04: Few-Shot Learning.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure conveys a single core idea cleanly.

Figures:
    fig1_nway_kshot           5-way 1-shot episode: support set + query set
                              layout, the standard FSL evaluation protocol.
    fig2_prototypical         Prototypical Network embedding space: per-class
                              samples cluster around their mean prototype, with
                              nearest-prototype decision regions shaded.
    fig3_matching             Matching Network attention: cosine similarities
                              between a query and 5 support samples become
                              softmax weights for the prediction.
    fig4_maml                 MAML inner/outer loop: a meta-initialization
                              theta from which a few gradient steps reach the
                              optimum of any sampled task.
    fig5_relation             Relation Network architecture: shared embedder
                              + learned relation module producing scalar
                              similarity scores.
    fig6_mini_imagenet        miniImageNet 5-way benchmark accuracies for
                              representative methods (1-shot vs 5-shot).
    fig7_episodic             Episodic training timeline: each iteration
                              samples a new N-way K-shot task to mimic
                              test-time conditions.

Usage:
    python3 scripts/figures/transfer-learning/04-few-shot.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary
C_PURPLE = "#7c3aed"   # secondary
C_GREEN = "#10b981"    # accent / good
C_AMBER = "#f59e0b"    # warning / highlight
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, "#ef4444"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "transfer-learning" / "04-few-shot-learning"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "transfer-learning" / "04-Few-Shot-Learning"


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
# Figure 1: N-way K-shot episode layout
# ---------------------------------------------------------------------------
def fig1_nway_kshot() -> None:
    """5-way 1-shot support set on the left, query set on the right."""
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    classes = ["Pangolin", "Quokka", "Axolotl", "Tapir", "Fossa"]
    colors = PALETTE

    # Title
    ax.text(6, 6.6, "5-way 1-shot Episode", ha="center", fontsize=15,
            fontweight="bold", color=C_DARK)

    # ---- Support set (left) ----
    sup_box = FancyBboxPatch((0.3, 0.7), 5.2, 5.2,
                             boxstyle="round,pad=0.05,rounding_size=0.15",
                             linewidth=1.6, edgecolor=C_BLUE,
                             facecolor=C_BG)
    ax.add_patch(sup_box)
    ax.text(2.9, 5.55, "Support Set  S", ha="center", fontsize=12,
            fontweight="bold", color=C_BLUE)
    ax.text(2.9, 5.18, "N = 5 classes,  K = 1 example each",
            ha="center", fontsize=9.5, color=C_DARK, style="italic")

    for i, (name, c) in enumerate(zip(classes, colors)):
        y = 4.5 - i * 0.78
        # Image tile
        img = Rectangle((0.7, y - 0.28), 0.6, 0.56,
                        facecolor=c, edgecolor=C_DARK, linewidth=1.0)
        ax.add_patch(img)
        # Label
        ax.text(1.55, y, name, ha="left", va="center",
                fontsize=10.5, color=C_DARK)
        # Class id chip
        ax.text(4.9, y, f"y = {i}", ha="right", va="center",
                fontsize=9.5, color=C_GRAY, family="monospace")

    # ---- Query set (right) ----
    q_box = FancyBboxPatch((6.5, 0.7), 5.2, 5.2,
                           boxstyle="round,pad=0.05,rounding_size=0.15",
                           linewidth=1.6, edgecolor=C_PURPLE,
                           facecolor=C_BG)
    ax.add_patch(q_box)
    ax.text(9.1, 5.55, "Query Set  Q", ha="center", fontsize=12,
            fontweight="bold", color=C_PURPLE)
    ax.text(9.1, 5.18, "Unlabeled examples to classify",
            ha="center", fontsize=9.5, color=C_DARK, style="italic")

    rng = np.random.default_rng(3)
    q_classes = rng.integers(0, 5, size=10)
    for k, cls in enumerate(q_classes):
        col = k % 5
        row = k // 5
        x = 6.95 + col * 0.85
        y = 4.3 - row * 1.5
        img = Rectangle((x, y - 0.32), 0.66, 0.64,
                        facecolor=colors[cls], edgecolor=C_DARK, linewidth=1.0)
        ax.add_patch(img)
        ax.text(x + 0.33, y - 0.65, "?", ha="center", fontsize=11,
                color=C_DARK, fontweight="bold")

    # Arrow between
    arrow = FancyArrowPatch((5.5, 3.3), (6.5, 3.3),
                            arrowstyle="-|>", mutation_scale=18,
                            color=C_DARK, linewidth=1.5)
    ax.add_patch(arrow)
    ax.text(6.0, 3.55, "classify", ha="center", fontsize=9, color=C_DARK)

    # Footnote
    ax.text(6, 0.25,
            "An episode = (support, query) drawn from a held-out class set; "
            "accuracy is averaged over thousands of episodes.",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    _save(fig, "fig1_nway_kshot")


# ---------------------------------------------------------------------------
# Figure 2: Prototypical Network embedding space
# ---------------------------------------------------------------------------
def fig2_prototypical() -> None:
    """Embedding-space view: clusters, prototypes, decision regions, query."""
    fig, ax = plt.subplots(figsize=(8.2, 6.8))
    rng = np.random.default_rng(11)

    centers = np.array([
        [-2.4,  1.6],
        [ 2.0,  2.0],
        [ 2.4, -1.8],
        [-2.0, -1.6],
    ])
    colors = PALETTE[:4]
    names = ["Class A", "Class B", "Class C", "Class D"]

    # ---- Soft Voronoi background using nearest-prototype rule ----
    xs = np.linspace(-5.5, 5.5, 400)
    ys = np.linspace(-5.0, 5.0, 360)
    XX, YY = np.meshgrid(xs, ys)
    flat = np.stack([XX.ravel(), YY.ravel()], axis=1)
    d2 = np.sum((flat[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    region = np.argmin(d2, axis=1).reshape(XX.shape)

    from matplotlib.colors import ListedColormap
    bg_colors = ["#dbeafe", "#ede9fe", "#d1fae5", "#fef3c7"]
    ax.imshow(region, extent=(-5.5, 5.5, -5.0, 5.0),
              origin="lower", cmap=ListedColormap(bg_colors),
              alpha=0.55, aspect="auto", zorder=0)

    # ---- Support samples + prototypes ----
    K = 5
    for c, ctr, name in zip(colors, centers, names):
        pts = ctr + rng.normal(scale=0.55, size=(K, 2))
        ax.scatter(pts[:, 0], pts[:, 1], s=70, color=c,
                   edgecolor="white", linewidth=1.2, zorder=3,
                   label=f"{name}  support")
        proto = pts.mean(axis=0)
        ax.scatter(proto[0], proto[1], s=320, marker="*", color=c,
                   edgecolor=C_DARK, linewidth=1.4, zorder=5)
        ax.annotate(f"prototype  c_{name[-1]}",
                    xy=(proto[0], proto[1]),
                    xytext=(proto[0] + 0.35, proto[1] + 0.45),
                    fontsize=9, color=C_DARK, fontweight="bold")

    # ---- A query point and its nearest prototype line ----
    query = np.array([0.4, 0.6])
    ax.scatter(query[0], query[1], s=180, marker="X", color=C_DARK,
               edgecolor="white", linewidth=1.4, zorder=6)
    ax.annotate("query  x_q", xy=query,
                xytext=(query[0] + 0.4, query[1] - 0.65),
                fontsize=10, color=C_DARK, fontweight="bold")
    # Nearest prototype to query
    protos_actual = []
    for c, ctr in zip(colors, centers):
        pts = ctr + rng.normal(scale=0.55, size=(K, 2))  # same seed reuse
        protos_actual.append(pts.mean(axis=0))
    # Easier: just use class centers for the line
    dists = np.linalg.norm(centers - query, axis=1)
    nearest = int(np.argmin(dists))
    ax.plot([query[0], centers[nearest, 0]],
            [query[1], centers[nearest, 1]],
            color=C_DARK, linestyle="--", linewidth=1.2, zorder=4)

    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-5.0, 5.0)
    ax.set_xlabel("embedding dim 1", fontsize=10)
    ax.set_ylabel("embedding dim 2", fontsize=10)
    ax.set_title("Prototypical Networks: classify by nearest class mean",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)

    # Compact legend (one entry per class)
    handles = [
        mpatches.Patch(color=colors[i], label=names[i]) for i in range(4)
    ]
    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                              markerfacecolor=C_GRAY, markeredgecolor=C_DARK,
                              markersize=14, label="prototype (mean)"))
    handles.append(plt.Line2D([0], [0], marker="X", color="w",
                              markerfacecolor=C_DARK, markersize=11,
                              label="query"))
    ax.legend(handles=handles, loc="upper left", fontsize=8.5,
              framealpha=0.92, ncol=2)

    _save(fig, "fig2_prototypical")


# ---------------------------------------------------------------------------
# Figure 3: Matching Network attention weights
# ---------------------------------------------------------------------------
def fig3_matching() -> None:
    """Attention from query to support samples, then weighted vote."""
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6, 6.55, "Matching Networks: attention-weighted prediction",
            ha="center", fontsize=13.5, fontweight="bold", color=C_DARK)

    # Support samples
    classes = ["A", "A", "B", "B", "C"]
    sims = np.array([0.82, 0.41, 0.18, 0.10, 0.05])
    weights = np.exp(sims) / np.exp(sims).sum()  # softmax

    cls_color = {"A": C_BLUE, "B": C_PURPLE, "C": C_GREEN}

    # Query node (left)
    qx, qy = 1.4, 3.5
    q_circ = plt.Circle((qx, qy), 0.55, facecolor=C_DARK,
                        edgecolor="white", linewidth=1.6, zorder=4)
    ax.add_patch(q_circ)
    ax.text(qx, qy, "x_q", color="white", ha="center", va="center",
            fontsize=11, fontweight="bold")
    ax.text(qx, qy - 1.0, "query", ha="center", fontsize=10, color=C_DARK)

    # Support samples (column)
    sx = 6.0
    ys = np.linspace(5.6, 1.4, 5)
    for i, (cls, s, w, y) in enumerate(zip(classes, sims, weights, ys)):
        col = cls_color[cls]
        c = plt.Circle((sx, y), 0.4, facecolor=col, edgecolor="white",
                       linewidth=1.4, zorder=4)
        ax.add_patch(c)
        ax.text(sx, y, f"x_{i+1}", color="white", ha="center", va="center",
                fontsize=9.5, fontweight="bold")
        ax.text(sx + 0.65, y + 0.05, f"class {cls}",
                fontsize=9, color=C_DARK, va="center")

        # Edge from query -> support, width ~ attention weight
        lw = 0.4 + w * 9.0
        ax.plot([qx + 0.55, sx - 0.4], [qy, y],
                color=col, linewidth=lw, alpha=0.55, zorder=2)
        # Weight label near support
        ax.text(sx + 1.95, y, f"a = {w:.2f}",
                fontsize=9.5, color=col, fontweight="bold", va="center",
                family="monospace")

    # Aggregate label sum
    px, py = 10.4, 3.5
    pred_box = FancyBboxPatch((px - 1.0, py - 0.85), 2.0, 1.7,
                              boxstyle="round,pad=0.05,rounding_size=0.12",
                              linewidth=1.5, edgecolor=C_AMBER,
                              facecolor="#fff7ed")
    ax.add_patch(pred_box)
    # Aggregated probabilities
    p_a = sum(w for cls, w in zip(classes, weights) if cls == "A")
    p_b = sum(w for cls, w in zip(classes, weights) if cls == "B")
    p_c = sum(w for cls, w in zip(classes, weights) if cls == "C")
    ax.text(px, py + 0.55, "prediction", ha="center",
            fontsize=10, fontweight="bold", color=C_AMBER)
    ax.text(px, py + 0.15, f"P(A) = {p_a:.2f}", ha="center", fontsize=9.5,
            color=C_BLUE, family="monospace")
    ax.text(px, py - 0.20, f"P(B) = {p_b:.2f}", ha="center", fontsize=9.5,
            color=C_PURPLE, family="monospace")
    ax.text(px, py - 0.55, f"P(C) = {p_c:.2f}", ha="center", fontsize=9.5,
            color=C_GREEN, family="monospace")

    # Arrow from sims column to prediction
    arrow = FancyArrowPatch((sx + 3.0, 3.5), (px - 1.05, 3.5),
                            arrowstyle="-|>", mutation_scale=16,
                            color=C_DARK, linewidth=1.3)
    ax.add_patch(arrow)
    ax.text((sx + 3.0 + px - 1.05) / 2, 3.85,
            "softmax(cos sim)", ha="center", fontsize=8.5,
            color=C_DARK, style="italic")

    ax.text(6, 0.5,
            "Edge thickness  proportional to softmax attention weight a(x_q, x_i)",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    _save(fig, "fig3_matching")


# ---------------------------------------------------------------------------
# Figure 4: MAML inner/outer loop
# ---------------------------------------------------------------------------
def fig4_maml() -> None:
    """Meta-initialization theta with task-specific adapted thetas."""
    fig, ax = plt.subplots(figsize=(8.6, 6.6))

    # Build a synthetic 2D loss landscape that's relatively flat near origin
    xs = np.linspace(-4.5, 4.5, 320)
    ys = np.linspace(-4.5, 4.5, 320)
    XX, YY = np.meshgrid(xs, ys)
    # A few task-specific minima
    task_optima = np.array([
        [ 2.6,  2.4],   # task 1
        [-2.8,  1.8],   # task 2
        [ 0.6, -3.0],   # task 3
    ])
    Z = np.zeros_like(XX)
    for opt in task_optima:
        Z += np.exp(-0.18 * ((XX - opt[0]) ** 2 + (YY - opt[1]) ** 2))
    # Convert "high reward" to loss (lower is better)
    Z = -Z

    cs = ax.contourf(XX, YY, Z, levels=18, cmap="Blues_r", alpha=0.55)
    ax.contour(XX, YY, Z, levels=10, colors="white", linewidths=0.5, alpha=0.6)

    # Meta initialisation theta at origin-ish
    theta = np.array([0.0, 0.0])
    ax.scatter(*theta, s=260, marker="*", color=C_AMBER,
               edgecolor=C_DARK, linewidth=1.4, zorder=6)
    ax.annotate(r"$\theta$  (meta-init)", xy=theta,
                xytext=(theta[0] + 0.35, theta[1] + 0.45),
                fontsize=11, color=C_DARK, fontweight="bold")

    # For each task, draw a few inner-loop steps from theta toward optimum
    task_colors = [C_BLUE, C_PURPLE, C_GREEN]
    task_names = ["Task 1", "Task 2", "Task 3"]
    for opt, col, name in zip(task_optima, task_colors, task_names):
        # Three steps along straight line toward optimum
        steps = np.linspace(0, 0.92, 4)
        path = theta[None, :] * (1 - steps[:, None]) + opt[None, :] * steps[:, None]
        ax.plot(path[:, 0], path[:, 1], color=col, linewidth=2.2,
                marker="o", markersize=6, markeredgecolor="white",
                markeredgewidth=1.0, zorder=5, label=f"{name}: inner loop")
        # End point = adapted theta_i'
        ax.scatter(path[-1, 0], path[-1, 1], s=130, color=col,
                   edgecolor=C_DARK, linewidth=1.3, zorder=6)
        ax.annotate(rf"$\theta'_{{{name[-1]}}}$",
                    xy=(path[-1, 0], path[-1, 1]),
                    xytext=(path[-1, 0] + 0.25, path[-1, 1] + 0.25),
                    fontsize=10.5, color=col, fontweight="bold")
        # Mark task optimum
        ax.scatter(opt[0], opt[1], s=120, marker="P", color=col,
                   edgecolor="white", linewidth=1.2, zorder=6)

    # Outer-loop arrow: where the meta gradient nudges theta over time
    outer_target = np.array([0.05, -0.4])
    arrow = FancyArrowPatch(theta, outer_target,
                            arrowstyle="-|>", mutation_scale=18,
                            color=C_AMBER, linewidth=2.0)
    # Skip drawing - too short to be useful; instead annotate
    ax.text(0.0, -4.3,
            "Outer loop updates  theta  to minimize the post-adaptation loss "
            "averaged over tasks",
            ha="center", fontsize=9.5, color=C_DARK, style="italic")

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel("parameter dim 1", fontsize=10)
    ax.set_ylabel("parameter dim 2", fontsize=10)
    ax.set_title("MAML: one initialization, fast adaptation to many tasks",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)

    _save(fig, "fig4_maml")


# ---------------------------------------------------------------------------
# Figure 5: Relation Network architecture
# ---------------------------------------------------------------------------
def fig5_relation() -> None:
    """Embedding module + relation module producing a learned similarity."""
    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(6, 5.6, "Relation Network: learn the similarity, do not assume it",
            ha="center", fontsize=13, fontweight="bold", color=C_DARK)

    def block(x, y, w, h, label, color, sub=None):
        b = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.05,rounding_size=0.12",
                           linewidth=1.4, edgecolor=color,
                           facecolor=C_BG)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0),
                label, ha="center", va="center",
                fontsize=10.5, color=color, fontweight="bold")
        if sub:
            ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center",
                    va="center", fontsize=8.5, color=C_DARK, style="italic")

    # Inputs: support image + query image
    block(0.3, 3.6, 1.6, 1.0, "support  x_i",
          C_BLUE, "(class c)")
    block(0.3, 1.0, 1.6, 1.0, "query  x_q",
          C_PURPLE)

    # Shared embedding module
    block(2.6, 3.6, 1.8, 1.0, "f_theta", C_DARK, "embedding CNN")
    block(2.6, 1.0, 1.8, 1.0, "f_theta", C_DARK, "(shared weights)")

    # Concatenation
    block(5.1, 2.3, 1.7, 1.0, "concat", C_GRAY,
          "[ f(x_i) | f(x_q) ]")

    # Relation module
    block(7.6, 2.3, 2.2, 1.0, "g_phi", C_AMBER,
          "relation MLP")

    # Output relation score
    block(10.4, 2.3, 1.4, 1.0, "r in [0,1]", C_GREEN,
          "similarity")

    # Arrows
    def arr(x1, y1, x2, y2, color=C_DARK):
        a = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle="-|>", mutation_scale=14,
                            color=color, linewidth=1.2)
        ax.add_patch(a)

    arr(1.9, 4.1, 2.6, 4.1, C_BLUE)
    arr(1.9, 1.5, 2.6, 1.5, C_PURPLE)
    arr(4.4, 4.1, 5.1, 3.1, C_DARK)
    arr(4.4, 1.5, 5.1, 2.5, C_DARK)
    arr(6.8, 2.8, 7.6, 2.8, C_DARK)
    arr(9.8, 2.8, 10.4, 2.8, C_DARK)

    ax.text(6, 0.35,
            "Loss: MSE between r and 1{i and q same class}.  "
            "Both modules are trained end-to-end.",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    _save(fig, "fig5_relation")


# ---------------------------------------------------------------------------
# Figure 6: miniImageNet 5-way benchmark
# ---------------------------------------------------------------------------
def fig6_mini_imagenet() -> None:
    """Reported 5-way 1-shot vs 5-shot accuracy on miniImageNet."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    methods = ["Matching\nNets", "MAML", "Prototypical\nNets",
               "Relation\nNets", "Reptile", "Baseline++\n(transfer)"]
    # Numbers from the original papers / Chen et al. 2019 survey.
    one_shot = [43.6, 48.7, 49.4, 50.4, 49.97, 48.2]
    five_shot = [55.3, 63.1, 68.2, 65.3, 65.99, 66.4]

    x = np.arange(len(methods))
    w = 0.36

    bars1 = ax.bar(x - w / 2, one_shot, w, color=C_BLUE,
                   edgecolor="white", linewidth=1.0,
                   label="5-way 1-shot")
    bars2 = ax.bar(x + w / 2, five_shot, w, color=C_PURPLE,
                   edgecolor="white", linewidth=1.0,
                   label="5-way 5-shot")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.6,
                    f"{h:.1f}", ha="center", fontsize=8.7, color=C_DARK)

    # Random-guess baseline at 20% (5-way)
    ax.axhline(20.0, color=C_GRAY, linestyle="--", linewidth=1.0)
    ax.text(len(methods) - 0.4, 21.5, "random  (20%)",
            ha="right", fontsize=9, color=C_GRAY, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9.5)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_ylim(0, 80)
    ax.set_title("miniImageNet 5-way benchmark: reported accuracies",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    fig.text(0.5, -0.02,
             "Numbers from the original papers and Chen et al. (ICLR 2019); "
             "newer methods now exceed 80%.",
             ha="center", fontsize=8.5, color=C_GRAY, style="italic")

    _save(fig, "fig6_mini_imagenet")


# ---------------------------------------------------------------------------
# Figure 7: Episodic training timeline
# ---------------------------------------------------------------------------
def fig7_episodic() -> None:
    """Each training step samples a new few-shot task: train like you test."""
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    ax.text(6, 4.5, "Episodic training: every step is a fresh N-way K-shot task",
            ha="center", fontsize=13, fontweight="bold", color=C_DARK)

    # Big base-class pool at left
    pool = FancyBboxPatch((0.3, 1.0), 2.2, 2.6,
                          boxstyle="round,pad=0.05,rounding_size=0.12",
                          linewidth=1.4, edgecolor=C_DARK,
                          facecolor=C_BG)
    ax.add_patch(pool)
    ax.text(1.4, 3.3, "Base classes", ha="center", fontsize=10.5,
            fontweight="bold", color=C_DARK)
    ax.text(1.4, 2.95, "(many examples)", ha="center", fontsize=8.5,
            color=C_DARK, style="italic")

    # Tile a small grid of class chips inside pool
    rng = np.random.default_rng(2)
    chip_colors = rng.choice(PALETTE + [C_GRAY], size=24)
    for k in range(24):
        cx = 0.55 + (k % 6) * 0.30
        cy = 1.20 + (k // 6) * 0.30
        ax.add_patch(Rectangle((cx, cy), 0.22, 0.22,
                               facecolor=chip_colors[k], edgecolor="white",
                               linewidth=0.6))

    # Three sequential episodes to the right
    ep_x = [3.5, 6.4, 9.3]
    ep_classes = [
        [C_BLUE, C_PURPLE, C_GREEN, C_AMBER, "#ef4444"],
        [C_PURPLE, C_AMBER, C_GREEN, C_BLUE, "#ef4444"],
        [C_GREEN, C_BLUE, "#ef4444", C_PURPLE, C_AMBER],
    ]
    titles = ["Episode t", "Episode t+1", "Episode t+2"]

    for x0, title, classes in zip(ep_x, titles, ep_classes):
        ep = FancyBboxPatch((x0, 1.0), 2.2, 2.6,
                            boxstyle="round,pad=0.05,rounding_size=0.12",
                            linewidth=1.3, edgecolor=C_BLUE,
                            facecolor="white")
        ax.add_patch(ep)
        ax.text(x0 + 1.1, 3.3, title, ha="center", fontsize=10.5,
                fontweight="bold", color=C_BLUE)
        ax.text(x0 + 1.1, 2.95, "5-way 1-shot", ha="center",
                fontsize=8.5, color=C_DARK, style="italic")

        # Support row (5 chips)
        ax.text(x0 + 0.13, 2.55, "S:", fontsize=8.5, color=C_DARK)
        for i, c in enumerate(classes):
            ax.add_patch(Rectangle((x0 + 0.40 + i * 0.30, 2.42), 0.24, 0.24,
                                   facecolor=c, edgecolor=C_DARK,
                                   linewidth=0.7))
        # Query row
        ax.text(x0 + 0.13, 1.95, "Q:", fontsize=8.5, color=C_DARK)
        for i in range(5):
            qcol = classes[(i + 2) % 5]
            ax.add_patch(Rectangle((x0 + 0.40 + i * 0.30, 1.82), 0.24, 0.24,
                                   facecolor=qcol, edgecolor=C_DARK,
                                   linewidth=0.7))
        # Loss arrow
        ax.text(x0 + 1.1, 1.30, "loss = CE(query, prediction)",
                ha="center", fontsize=8, color=C_GRAY, style="italic")

        # Arrow from pool to first episode, and between episodes
    arrow_y = 2.3
    a1 = FancyArrowPatch((2.5, arrow_y), (3.5, arrow_y),
                         arrowstyle="-|>", mutation_scale=14,
                         color=C_DARK, linewidth=1.1)
    ax.add_patch(a1)
    a2 = FancyArrowPatch((5.7, arrow_y), (6.4, arrow_y),
                         arrowstyle="-|>", mutation_scale=14,
                         color=C_DARK, linewidth=1.1)
    ax.add_patch(a2)
    a3 = FancyArrowPatch((8.6, arrow_y), (9.3, arrow_y),
                         arrowstyle="-|>", mutation_scale=14,
                         color=C_DARK, linewidth=1.1)
    ax.add_patch(a3)

    ax.text(6, 0.45,
            "Training difficulty matches test difficulty: the model never sees "
            "the full base set at once.",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    _save(fig, "fig7_episodic")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Few-Shot Learning figures...")
    fig1_nway_kshot()
    fig2_prototypical()
    fig3_matching()
    fig4_maml()
    fig5_relation()
    fig6_mini_imagenet()
    fig7_episodic()
    print("Done.")


if __name__ == "__main__":
    main()
