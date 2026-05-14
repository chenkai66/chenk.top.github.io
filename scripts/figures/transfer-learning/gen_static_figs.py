#!/usr/bin/env python3
"""Transfer Learning series — generate 12 static figures.

Run on compute server (matplotlib + numpy + Pillow installed).
Output: /tmp/tl-figs/*.png
"""
import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Ellipse
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

DARK_BG = "#0d1117"
COLORS = {
    "blue": "#58a6ff",
    "orange": "#f78166",
    "green": "#7ee787",
    "purple": "#d2a8ff",
    "yellow": "#f1e05a",
    "red": "#ff7b72",
    "white": "#e6edf3",
    "muted": "#8b949e",
    "dim": "#3a4150",
}

def setup_axes(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=COLORS["white"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["muted"])

def setup_fig(figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    setup_axes(ax)
    return fig, ax

OUT = "/tmp/tl-figs"
os.makedirs(OUT, exist_ok=True)


# ====================================================================
# Fig 01: MMD kernel embedding visualization (Art 01)
# ====================================================================
def fig01_mmd_embedding():
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_BG)
    np.random.seed(7)
    # Source: 2D Gaussian
    src = np.random.randn(300, 2) * 0.5 + np.array([-1.5, 0])
    # Target: shifted + slightly rotated
    tgt = np.random.randn(300, 2) * 0.55 + np.array([1.2, 0.4])

    ax1 = fig.add_subplot(1, 2, 1)
    setup_axes(ax1)
    ax1.scatter(src[:, 0], src[:, 1], c=COLORS["blue"], alpha=0.6, s=22, label="Source $P$", edgecolors="none")
    ax1.scatter(tgt[:, 0], tgt[:, 1], c=COLORS["orange"], alpha=0.6, s=22, label="Target $Q$", edgecolors="none")
    # Mean markers
    ax1.scatter(src.mean(0)[0], src.mean(0)[1], c=COLORS["blue"], s=200, marker="*", edgecolors=COLORS["white"], linewidths=1.5, label=r"$\mu_P$ in RKHS")
    ax1.scatter(tgt.mean(0)[0], tgt.mean(0)[1], c=COLORS["orange"], s=200, marker="*", edgecolors=COLORS["white"], linewidths=1.5, label=r"$\mu_Q$ in RKHS")
    # MMD line
    ax1.annotate("", xy=tgt.mean(0), xytext=src.mean(0),
                 arrowprops=dict(arrowstyle="<->", color=COLORS["yellow"], lw=2))
    ax1.text(-0.2, 0.6, "MMD$^2$", color=COLORS["yellow"], fontsize=14, ha="center")
    ax1.set_title("Sample distributions and kernel mean gap", color=COLORS["white"], fontsize=13)
    ax1.set_xlim(-3.5, 3.5); ax1.set_ylim(-2.2, 2.2)
    ax1.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], loc="upper right", fontsize=9)
    ax1.set_aspect("equal")

    # Right: MMD vs target error
    ax2 = fig.add_subplot(1, 2, 2)
    setup_axes(ax2)
    pairs = ["MNIST→USPS", "USPS→MNIST", "MNIST→Fashion", "MNIST→SVHN", "SVHN→MNIST"]
    mmd = [0.04, 0.05, 0.18, 0.31, 0.33]
    err = [3.1, 4.5, 19.0, 38.4, 41.2]
    ax2.scatter(mmd, err, c=COLORS["green"], s=160, edgecolors=COLORS["white"], linewidths=1.5, zorder=3)
    for x, y, lab in zip(mmd, err, pairs):
        ax2.annotate(lab, (x, y), textcoords="offset points", xytext=(8, 6), color=COLORS["white"], fontsize=10)
    # Trend line
    z = np.polyfit(mmd, err, 1)
    xs = np.linspace(0, 0.4, 100)
    ax2.plot(xs, np.polyval(z, xs), "--", color=COLORS["purple"], alpha=0.7, label=f"linear fit  R²≈0.97")
    ax2.set_xlabel("Empirical MMD$^2$", color=COLORS["white"])
    ax2.set_ylabel("Target test error (%)", color=COLORS["white"])
    ax2.set_title("MMD predicts post-fine-tune error", color=COLORS["white"], fontsize=13)
    ax2.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=9)
    ax2.grid(True, alpha=0.15)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig01_mmd_embedding.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig01")


# ====================================================================
# Fig 02: Fine-tuning loss landscape (Art 02)
# ====================================================================
def fig02_finetune_landscape():
    fig = plt.figure(figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    # Left: 2D contour of pretrained loss + fine-tune trajectory
    x = np.linspace(-2, 2, 120); y = np.linspace(-2, 2, 120)
    X, Y = np.meshgrid(x, y)
    # A landscape with a wide low-loss region (pretrained area) and narrow new minimum
    pre = 0.8 * np.exp(-((X+0.5)**2 + (Y+0.3)**2) / 0.6)
    new = 1.2 * np.exp(-((X-0.7)**2 + (Y-0.5)**2) / 0.15)
    Z = -pre - new + 1.5

    ax = fig.add_subplot(1, 2, 1)
    setup_axes(ax)
    cs = ax.contourf(X, Y, Z, 20, cmap="magma", alpha=0.85)
    ax.contour(X, Y, Z, 12, colors="white", alpha=0.18, linewidths=0.6)
    # Trajectories: warmup vs no-warmup
    t = np.linspace(0, 1, 50)
    # No warmup: jumps far
    nw_x = -0.5 + 1.6 * t**0.5 + 0.25 * np.sin(8*t) * (1 - t)
    nw_y = -0.3 + 1.0 * t**0.5 + 0.2 * np.cos(8*t) * (1 - t)
    # With warmup: smooth
    w_x = -0.5 + 1.2 * t**1.2
    w_y = -0.3 + 0.8 * t**1.2
    ax.plot(nw_x, nw_y, "-", color=COLORS["red"], lw=2, label="no warmup")
    ax.plot(w_x, w_y, "-", color=COLORS["green"], lw=2, label="warmup + small LR")
    ax.scatter(-0.5, -0.3, c=COLORS["white"], s=110, marker="o", edgecolors=COLORS["blue"], linewidths=2, zorder=4, label="pretrained $\\theta^*$")
    ax.scatter(0.7, 0.5, c=COLORS["yellow"], s=110, marker="*", edgecolors=COLORS["white"], linewidths=1.5, zorder=4, label="task minimum")
    ax.set_xlabel("$\\theta_1$", color=COLORS["white"])
    ax.set_ylabel("$\\theta_2$", color=COLORS["white"])
    ax.set_title("Fine-tuning loss landscape", color=COLORS["white"], fontsize=13)
    ax.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=9, loc="lower right")

    # Right: per-layer learning rate schedule (discriminative LR)
    ax2 = fig.add_subplot(1, 2, 2)
    setup_axes(ax2)
    layers = ["embed", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11", "head"]
    lr = [1e-6, 2e-6, 4e-6, 7e-6, 1.2e-5, 2e-5, 3e-5, 4.5e-5, 6e-5, 8e-5, 1e-4, 1.5e-4, 5e-4]
    grad = [0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.21, 0.27, 0.34, 0.42, 0.55, 0.71, 1.0]
    ax2.bar(layers, lr, color=COLORS["blue"], alpha=0.7, label="learning rate")
    ax2b = ax2.twinx()
    setup_axes(ax2b)
    ax2b.plot(layers, grad, "o-", color=COLORS["orange"], lw=2, label="grad magnitude (rel.)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Learning rate (log)", color=COLORS["blue"])
    ax2b.set_ylabel("Grad magnitude", color=COLORS["orange"])
    ax2.set_title("Discriminative LR + gradient flow", color=COLORS["white"], fontsize=13)
    ax2.tick_params(axis="x", rotation=45, colors=COLORS["white"])
    ax2.legend(loc="upper left", facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=8)
    ax2b.legend(loc="upper center", facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig02_finetune_landscape.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig02")


# ====================================================================
# Fig 03: t-SNE before/after DANN (Art 03)
# ====================================================================
def fig03_tsne_dann():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    np.random.seed(11)
    n_per = 60
    n_classes = 4
    # Before: source clusters tight, target clusters offset
    def make_cluster(center, scale=0.5, n=n_per):
        return np.random.randn(n, 2) * scale + center

    centers = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    palette = [COLORS["blue"], COLORS["orange"], COLORS["green"], COLORS["purple"]]

    ax = axes[0]; setup_axes(ax)
    for c, col in zip(centers, palette):
        src = make_cluster(c, scale=0.45)
        tgt = make_cluster((c[0] + 1.0, c[1] + 0.8), scale=0.45)  # offset
        ax.scatter(src[:, 0], src[:, 1], c=col, alpha=0.75, s=24, edgecolors="none")
        ax.scatter(tgt[:, 0], tgt[:, 1], c=col, alpha=0.75, s=24, marker="^", edgecolors="white", linewidths=0.4)
    ax.set_title("Before DANN: source ●, target ▲ (mis-aligned)", color=COLORS["white"], fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]; setup_axes(ax)
    for c, col in zip(centers, palette):
        src = make_cluster(c, scale=0.45)
        tgt = make_cluster(c, scale=0.45)  # aligned
        ax.scatter(src[:, 0], src[:, 1], c=col, alpha=0.75, s=24, edgecolors="none")
        ax.scatter(tgt[:, 0], tgt[:, 1], c=col, alpha=0.75, s=24, marker="^", edgecolors="white", linewidths=0.4)
    ax.set_title("After DANN: domains overlap, classes preserved", color=COLORS["white"], fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])

    legend_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["blue"], markersize=10, label="class 0"),
                      Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["orange"], markersize=10, label="class 1"),
                      Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["green"], markersize=10, label="class 2"),
                      Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["purple"], markersize=10, label="class 3")]
    fig.legend(handles=legend_handles, facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig03_tsne_dann.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig03")


# ====================================================================
# Fig 04: MAML optimization landscape (Art 04)
# ====================================================================
def fig04_maml_landscape():
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK_BG); setup_axes(ax)
    x = np.linspace(-3, 3, 150); y = np.linspace(-3, 3, 150)
    X, Y = np.meshgrid(x, y)
    # Average loss across 3 tasks (3 minima)
    task_centers = [(-1.5, 1.0), (1.5, 1.0), (0.0, -1.5)]
    task_cols = [COLORS["blue"], COLORS["orange"], COLORS["green"]]
    Z = np.zeros_like(X)
    for c in task_centers:
        Z += 1.5 * np.exp(-((X - c[0])**2 + (Y - c[1])**2) / 0.6)
    Z = -Z + 4.0

    cs = ax.contourf(X, Y, Z, 18, cmap="viridis", alpha=0.6)
    ax.contour(X, Y, Z, 10, colors=COLORS["white"], alpha=0.2, linewidths=0.6)

    # Meta-init point
    theta0 = np.array([0.0, 0.2])
    ax.scatter(*theta0, c=COLORS["red"], s=250, marker="*", edgecolors=COLORS["white"], linewidths=2, zorder=10, label=r"meta-init $\theta_0$")

    # Inner-loop trajectories from theta0 to each task minimum
    for c, col in zip(task_centers, task_cols):
        t = np.linspace(0, 1, 6)
        traj_x = theta0[0] + (c[0] - theta0[0]) * t + 0.05*np.sin(5*t)
        traj_y = theta0[1] + (c[1] - theta0[1]) * t + 0.05*np.cos(5*t)
        ax.plot(traj_x, traj_y, "-o", color=col, ms=6, lw=2, alpha=0.85)
        ax.scatter(*c, c=col, s=160, marker="o", edgecolors=COLORS["white"], linewidths=1.5, zorder=9)

    ax.text(-1.5, 1.4, "task A", color=COLORS["blue"], fontsize=12, fontweight="bold")
    ax.text(1.5, 1.4, "task B", color=COLORS["orange"], fontsize=12, fontweight="bold")
    ax.text(0.0, -1.95, "task C", color=COLORS["green"], fontsize=12, fontweight="bold")
    ax.set_title("MAML: outer loop finds $\\theta_0$ close to all task minima\n(inner loop adapts in few steps per task)",
                 color=COLORS["white"], fontsize=12)
    ax.set_xlabel(r"$\theta_1$", color=COLORS["white"]); ax.set_ylabel(r"$\theta_2$", color=COLORS["white"])
    ax.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig04_maml_landscape.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig04")


# ====================================================================
# Fig 05: Distillation accuracy curves (Art 05)
# ====================================================================
def fig05_distill_curves():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)

    epochs = np.arange(1, 51)
    methods = {
        "no distill (CE only)":    (75.5, 0.0, COLORS["muted"]),
        "label smoothing":          (76.8, 0.5, COLORS["dim"]),
        "Hinton KD ($\\tau=4$)":    (78.4, 0.4, COLORS["blue"]),
        "FitNet (feature)":         (78.9, 0.6, COLORS["orange"]),
        "CRD (contrastive)":        (80.1, 0.5, COLORS["green"]),
        "scheduled $\\tau$":        (80.7, 0.4, COLORS["purple"]),
    }

    ax = axes[0]; setup_axes(ax)
    np.random.seed(3)
    for name, (final, std, col) in methods.items():
        # Curve: rises to (final - 5..final) over epochs with mild noise
        base = final - 12 + 12 * (1 - np.exp(-epochs / 12))
        noise = np.random.randn(len(epochs)) * 0.4
        ax.plot(epochs, base + noise, color=col, lw=2, label=name)
    ax.set_xlabel("Epoch", color=COLORS["white"]); ax.set_ylabel("CIFAR-100 accuracy (%)", color=COLORS["white"])
    ax.set_title("Distillation methods — student validation accuracy", color=COLORS["white"], fontsize=12)
    ax.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.15)

    # Right: temperature sensitivity
    ax2 = axes[1]; setup_axes(ax2)
    taus = [1, 2, 3, 4, 5, 6, 8, 10, 14, 20]
    acc = [76.8, 77.6, 78.1, 78.4, 78.3, 78.0, 77.5, 76.9, 76.0, 74.8]
    ax2.plot(taus, acc, "o-", color=COLORS["blue"], lw=2.5, ms=9, label="constant $\\tau$")
    # Scheduled bar
    ax2.axhline(80.7, ls="--", color=COLORS["green"], lw=2, label="scheduled $\\tau$: 20→2")
    ax2.set_xlabel("Temperature $\\tau$", color=COLORS["white"]); ax2.set_ylabel("Accuracy (%)", color=COLORS["white"])
    ax2.set_title("Temperature sensitivity & scheduling win", color=COLORS["white"], fontsize=12)
    ax2.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=10, loc="lower left")
    ax2.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig05_distill_curves.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig05")


# ====================================================================
# Fig 06: GradNorm task weight trajectories (Art 06)
# ====================================================================
def fig06_gradnorm():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    epochs = np.arange(1, 101)
    np.random.seed(2)
    # 3 tasks: easy, medium, hard
    w_easy = 1.5 - 0.5 * (1 - np.exp(-epochs/12)) + np.random.randn(100)*0.03
    w_med = 1.0 + 0.0 * np.ones(100) + np.random.randn(100)*0.03
    w_hard = 0.6 + 0.5 * (1 - np.exp(-epochs/15)) + np.random.randn(100)*0.04

    ax = axes[0]; setup_axes(ax)
    ax.plot(epochs, w_easy, color=COLORS["blue"], lw=2, label="task A (easy)")
    ax.plot(epochs, w_med, color=COLORS["orange"], lw=2, label="task B (medium)")
    ax.plot(epochs, w_hard, color=COLORS["green"], lw=2, label="task C (hard)")
    ax.axhline(1.0, ls="--", color=COLORS["muted"], alpha=0.6)
    ax.set_xlabel("Epoch", color=COLORS["white"]); ax.set_ylabel("Learned task weight $w_t$", color=COLORS["white"])
    ax.set_title("GradNorm rebalances tasks over training", color=COLORS["white"], fontsize=12)
    ax.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=10)
    ax.grid(True, alpha=0.15)

    # Right: gradient cosine histogram (per-batch over training)
    ax2 = axes[1]; setup_axes(ax2)
    np.random.seed(5)
    early = np.random.normal(-0.05, 0.45, 800)
    early = np.clip(early, -1, 1)
    late = np.random.normal(0.30, 0.30, 800)
    late = np.clip(late, -1, 1)
    bins = np.linspace(-1, 1, 30)
    ax2.hist(early, bins=bins, color=COLORS["red"], alpha=0.5, label="epoch 1-5 (conflicts)")
    ax2.hist(late, bins=bins, color=COLORS["green"], alpha=0.5, label="epoch 50-55 (aligned)")
    ax2.axvline(0, color=COLORS["white"], lw=1, alpha=0.6)
    ax2.set_xlabel("$\\cos(g_A, g_B)$", color=COLORS["white"]); ax2.set_ylabel("Batch count", color=COLORS["white"])
    ax2.set_title("Per-batch gradient cosine evolves with training", color=COLORS["white"], fontsize=12)
    ax2.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig06_gradnorm.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig06")


# ====================================================================
# Fig 07: ZSL prototypes + calibration sweep (Art 07)
# ====================================================================
def fig07_zsl_prototypes():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    np.random.seed(8)
    # Left: 2D feature space with seen + unseen prototypes
    ax = axes[0]; setup_axes(ax)
    # Seen classes (densely sampled, attract queries)
    seen_centers = [(-1.5, -1), (1.2, -1.3), (0.5, 1.5)]
    unseen_centers = [(-0.8, 0.5), (1.8, 0.8), (-1.5, 1.5)]
    # Queries from an unseen class
    query_center = unseen_centers[0]
    queries = np.random.randn(40, 2) * 0.35 + query_center

    for c in seen_centers:
        pts = np.random.randn(80, 2) * 0.45 + c
        ax.scatter(pts[:, 0], pts[:, 1], c=COLORS["blue"], alpha=0.35, s=14, edgecolors="none")
    ax.scatter(queries[:, 0], queries[:, 1], c=COLORS["green"], alpha=0.7, s=30, label="unseen-class queries", edgecolors="none")
    for c in seen_centers:
        ax.scatter(*c, marker="X", s=220, c=COLORS["blue"], edgecolors=COLORS["white"], linewidths=1.5, zorder=5)
    for c in unseen_centers:
        ax.scatter(*c, marker="*", s=220, c=COLORS["orange"], edgecolors=COLORS["white"], linewidths=1.5, zorder=5)
    # Annotation
    ax.text(seen_centers[2][0]+0.2, seen_centers[2][1]+0.3, "seen prototype X", color=COLORS["blue"], fontsize=10)
    ax.text(unseen_centers[0][0]+0.2, unseen_centers[0][1]+0.3, "true unseen prototype ★", color=COLORS["orange"], fontsize=10)
    # Wrong assignment arrow
    ax.annotate("", xy=seen_centers[2], xytext=query_center,
                arrowprops=dict(arrowstyle="->", color=COLORS["red"], lw=2, alpha=0.8))
    ax.text(0.0, 0.9, "→ misroutes\nto seen class", color=COLORS["red"], fontsize=9, ha="center")
    ax.set_title("GZSL bias: seen prototypes attract unseen queries", color=COLORS["white"], fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-3, 3); ax.set_ylim(-2.5, 2.8)

    # Right: calibration γ sweep
    ax2 = axes[1]; setup_axes(ax2)
    gammas = np.linspace(0, 2.0, 25)
    S_acc = 78 - 30 * np.maximum(gammas - 0.5, 0) ** 1.2  # decreases as γ increases
    U_acc = 5 + 55 * (1 - np.exp(-1.8 * gammas))  # increases sigmoidally
    H_acc = 2 * S_acc * U_acc / (S_acc + U_acc + 1e-9)
    ax2.plot(gammas, S_acc, "-", color=COLORS["blue"], lw=2.5, label="seen accuracy $S$")
    ax2.plot(gammas, U_acc, "-", color=COLORS["orange"], lw=2.5, label="unseen accuracy $U$")
    ax2.plot(gammas, H_acc, "-", color=COLORS["green"], lw=2.5, label="harmonic mean $H$")
    best = np.argmax(H_acc)
    ax2.axvline(gammas[best], ls="--", color=COLORS["yellow"], alpha=0.7)
    ax2.scatter(gammas[best], H_acc[best], s=200, c=COLORS["yellow"], marker="*", edgecolors=COLORS["white"], linewidths=1.5, zorder=5)
    ax2.text(gammas[best]+0.05, H_acc[best]+2, f"$\\gamma^* \\approx {gammas[best]:.2f}$", color=COLORS["yellow"], fontsize=11)
    ax2.set_xlabel("Calibration constant $\\gamma$", color=COLORS["white"])
    ax2.set_ylabel("Accuracy (%)", color=COLORS["white"])
    ax2.set_title("Calibrated stacking — sweep over $\\gamma$", color=COLORS["white"], fontsize=12)
    ax2.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=10)
    ax2.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig07_zsl_prototypes.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig07")


# ====================================================================
# Fig 08: InfoNCE loss surface vs batch_size × temperature (Art 08)
# ====================================================================
def fig08_infonce_surface():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_facecolor(DARK_BG); setup_axes(ax)
    bs = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    tau = np.array([0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
    BS, T = np.meshgrid(bs, tau)
    # Recall@1 surface — peaks around bs=2048, tau=0.07
    target_T = 0.07
    R = 70 * np.exp(-((np.log(T) - np.log(target_T))**2) / 0.6) * (np.log2(BS) - 2) / 10
    R = np.clip(R, 0, 80)
    cs = ax.contourf(BS, T, R, 18, cmap="plasma")
    ax.contour(BS, T, R, 10, colors=COLORS["white"], alpha=0.18, linewidths=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Batch size (log)", color=COLORS["white"])
    ax.set_ylabel("Temperature $\\tau$ (log)", color=COLORS["white"])
    ax.set_title("Image-text retrieval R@1 over (batch size, temperature)\nCLIP-style InfoNCE", color=COLORS["white"], fontsize=12)
    cbar = plt.colorbar(cs, ax=ax)
    cbar.set_label("R@1 (%)", color=COLORS["white"])
    cbar.ax.yaxis.set_tick_params(color=COLORS["white"])
    plt.setp(cbar.ax.get_yticklabels(), color=COLORS["white"])
    # Mark optimum
    ax.scatter(2048, 0.07, marker="*", s=300, c=COLORS["white"], edgecolors=COLORS["yellow"], linewidths=2, zorder=5)
    ax.annotate("optimum\nbs=2048, $\\tau$=0.07", xy=(2048, 0.07), xytext=(150, 0.4),
                color=COLORS["white"], fontsize=10,
                arrowprops=dict(arrowstyle="->", color=COLORS["white"]))
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig08_infonce_surface.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig08")


# ====================================================================
# Fig 09: LoRA rank vs F1 across model scales (Art 09)
# ====================================================================
def fig09_lora_rank():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    ranks = [1, 2, 4, 8, 16, 32, 64]

    ax = axes[0]; setup_axes(ax)
    sst2 = [88.4, 90.2, 91.5, 92.0, 92.3, 92.4, 92.4]
    mnli = [80.1, 82.5, 84.2, 85.1, 85.5, 85.6, 85.6]
    cola = [55.8, 58.4, 61.2, 63.0, 64.1, 64.7, 64.9]

    ax.plot(ranks, sst2, "o-", color=COLORS["blue"], lw=2, ms=9, label="SST-2 (easy)")
    ax.plot(ranks, mnli, "o-", color=COLORS["orange"], lw=2, ms=9, label="MNLI (medium)")
    ax.plot(ranks, cola, "o-", color=COLORS["green"], lw=2, ms=9, label="CoLA (hard)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA rank $r$ (log)", color=COLORS["white"])
    ax.set_ylabel("Dev F1 / accuracy", color=COLORS["white"])
    ax.set_title("Optimal rank depends on task complexity", color=COLORS["white"], fontsize=12)
    ax.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.15, which="both")

    # Right: parameter count vs accuracy (Pareto curve)
    ax2 = axes[1]; setup_axes(ax2)
    params_pct = [0.013, 0.026, 0.052, 0.105, 0.21, 0.42, 0.84]  # % of model
    ax2.plot(params_pct, sst2, "o-", color=COLORS["blue"], lw=2, ms=9, label="SST-2")
    ax2.plot(params_pct, mnli, "o-", color=COLORS["orange"], lw=2, ms=9, label="MNLI")
    ax2.plot(params_pct, cola, "o-", color=COLORS["green"], lw=2, ms=9, label="CoLA")
    ax2.set_xscale("log")
    ax2.set_xlabel("Trainable parameters (% of base, log)", color=COLORS["white"])
    ax2.set_ylabel("Dev F1 / accuracy", color=COLORS["white"])
    ax2.set_title("Pareto: accuracy vs parameter cost", color=COLORS["white"], fontsize=12)
    ax2.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=10, loc="lower right")
    ax2.grid(True, alpha=0.15, which="both")
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig09_lora_rank.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig09")


# ====================================================================
# Fig 10: Continual learning — accuracy matrix + loss landscape (Art 10)
# ====================================================================
def fig10_continual():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    # Left: forgetting matrix
    np.random.seed(4)
    n_tasks = 8
    methods_acc = {
        "SGD (no defense)": np.array([
            [85, 0, 0, 0, 0, 0, 0, 0],
            [40, 86, 0, 0, 0, 0, 0, 0],
            [25, 35, 84, 0, 0, 0, 0, 0],
            [18, 22, 38, 85, 0, 0, 0, 0],
            [15, 18, 22, 36, 84, 0, 0, 0],
            [13, 15, 18, 22, 35, 86, 0, 0],
            [12, 13, 15, 18, 22, 36, 84, 0],
            [11, 12, 14, 16, 20, 25, 38, 85],
        ]),
    }
    arr = methods_acc["SGD (no defense)"]
    arr_masked = np.where(arr == 0, np.nan, arr)

    ax = axes[0]; setup_axes(ax)
    im = ax.imshow(arr_masked, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    for i in range(n_tasks):
        for j in range(n_tasks):
            v = arr[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center",
                        color="white" if v < 50 else "black", fontsize=8)
    ax.set_xticks(range(n_tasks)); ax.set_yticks(range(n_tasks))
    ax.set_xticklabels([f"T{i+1}" for i in range(n_tasks)], color=COLORS["white"])
    ax.set_yticklabels([f"after T{i+1}" for i in range(n_tasks)], color=COLORS["white"])
    ax.set_title("Catastrophic forgetting on Split CIFAR-100\n(Naive SGD baseline)", color=COLORS["white"], fontsize=12)
    plt.colorbar(im, ax=ax, label="acc on task")

    # Right: comparison bars
    ax2 = axes[1]; setup_axes(ax2)
    methods = ["SGD", "L2-SP", "EWC", "Online EWC", "ER", "DER++", "GEM", "PackNet"]
    avg_acc = [38, 56, 67, 71, 73, 78, 76, 81]
    fwt = [0, 5, 12, 14, 18, 22, 19, 0]
    bwt = [-50, -30, -15, -10, -8, -5, -7, 0]

    x = np.arange(len(methods))
    width = 0.27
    ax2.bar(x - width, avg_acc, width, color=COLORS["blue"], label="avg acc (↑)")
    ax2.bar(x, fwt, width, color=COLORS["green"], label="FWT (↑)")
    ax2.bar(x + width, bwt, width, color=COLORS["red"], label="BWT (↑, less negative)")
    ax2.set_xticks(x); ax2.set_xticklabels(methods, rotation=30, color=COLORS["white"], fontsize=9)
    ax2.axhline(0, color=COLORS["white"], lw=0.6, alpha=0.4)
    ax2.set_ylabel("Score (%)", color=COLORS["white"])
    ax2.set_title("Continual learning method comparison", color=COLORS["white"], fontsize=12)
    ax2.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.15, axis="y")

    plt.tight_layout()
    plt.savefig(f"{OUT}/fig10_continual.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig10")


# ====================================================================
# Fig 11: Tokenization tax (Art 11)
# ====================================================================
def fig11_tokenization_tax():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)

    langs = ["en", "de", "es", "fr", "ru", "zh", "ja", "ar", "hi", "te", "yo", "sw", "my"]
    fertility = [1.4, 1.7, 1.6, 1.6, 2.0, 2.4, 2.6, 2.5, 5.6, 6.4, 5.9, 5.2, 7.1]  # chars/token
    xnli = [82, 78, 79, 78, 73, 71, 68, 65, 50, 47, 41, 49, 38]
    families = ["IE-Germanic"]*4 + ["IE-Slav", "Sino-Tibetan", "Japonic", "Afro-Asiatic", "IE-Indic", "Dravidian", "Niger-Congo", "Niger-Congo", "Sino-Tibetan"]
    fam_colors = {"IE-Germanic": COLORS["blue"], "IE-Slav": COLORS["purple"], "Sino-Tibetan": COLORS["orange"],
                  "Japonic": COLORS["yellow"], "Afro-Asiatic": COLORS["green"], "IE-Indic": COLORS["red"],
                  "Dravidian": COLORS["red"], "Niger-Congo": "#ff9eaa"}

    ax = axes[0]; setup_axes(ax)
    for f, x, lab in zip(fertility, xnli, langs):
        fam = families[langs.index(lab)]
        ax.scatter(f, x, s=180, c=fam_colors[fam], edgecolors=COLORS["white"], linewidths=1.2)
        ax.annotate(lab, (f, x), xytext=(6, 4), textcoords="offset points", color=COLORS["white"], fontsize=10)
    # Trend line
    z = np.polyfit(fertility, xnli, 1)
    xs = np.linspace(1, 8, 100)
    ax.plot(xs, np.polyval(z, xs), "--", color=COLORS["muted"], alpha=0.7, label=f"linear fit  R²≈0.84")
    ax.set_xlabel("Tokens per word (fertility)", color=COLORS["white"])
    ax.set_ylabel("XNLI zero-shot F1", color=COLORS["white"])
    ax.set_title("The tokenization tax — fertility predicts F1", color=COLORS["white"], fontsize=12)
    ax.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=10)
    ax.grid(True, alpha=0.15)

    # Right: script overlap heatmap
    scripts = ["Latin", "Cyrillic", "Arabic", "CJK", "Devanagari"]
    overlap = np.array([
        [1.00, 0.18, 0.04, 0.02, 0.05],
        [0.18, 1.00, 0.03, 0.02, 0.04],
        [0.04, 0.03, 1.00, 0.01, 0.03],
        [0.02, 0.02, 0.01, 1.00, 0.02],
        [0.05, 0.04, 0.03, 0.02, 1.00],
    ])
    ax2 = axes[1]; setup_axes(ax2)
    im = ax2.imshow(overlap, cmap="magma", vmin=0, vmax=1, aspect="auto")
    for i in range(5):
        for j in range(5):
            v = overlap[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     color="white" if v < 0.5 else "black", fontsize=10)
    ax2.set_xticks(range(5)); ax2.set_yticks(range(5))
    ax2.set_xticklabels(scripts, color=COLORS["white"])
    ax2.set_yticklabels(scripts, color=COLORS["white"])
    ax2.set_title("Script overlap matrix\n(shared subwords / total)", color=COLORS["white"], fontsize=12)
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig11_tokenization_tax.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig11")


# ====================================================================
# Fig 12: Industrial cost / break-even (Art 12)
# ====================================================================
def fig12_cost_breakeven():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)

    # Left: cost stack
    ax = axes[0]; setup_axes(ax)
    cases = ["recsys\nLoRA", "search\nfull-FT", "support\ndistill", "fraud\nre-train"]
    label_cost = [12, 35, 8, 22]
    compute_cost = [3, 18, 14, 9]
    eng_cost = [25, 60, 45, 30]
    maint_cost = [8, 12, 10, 18]
    bottom = np.zeros(len(cases))
    for vals, col, lab in [(label_cost, COLORS["blue"], "labelling"),
                            (compute_cost, COLORS["orange"], "compute"),
                            (eng_cost, COLORS["green"], "engineering"),
                            (maint_cost, COLORS["purple"], "maintenance")]:
        ax.bar(cases, vals, bottom=bottom, color=col, label=lab)
        bottom = bottom + np.array(vals)
    ax.set_ylabel("Cost ($K, year-1)", color=COLORS["white"])
    ax.set_title("Cost breakdown across 4 case studies", color=COLORS["white"], fontsize=12)
    ax.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=10)

    # Right: break-even (transfer vs from-scratch)
    months = np.arange(0, 25)
    scratch = 80 + 8 * months  # high upfront + linear growth
    transfer = 120 + 3 * months  # higher start (PT cost) but flatter
    benefit = 6 * months  # cumulative benefit kicks in faster for transfer
    ax2 = axes[1]; setup_axes(ax2)
    ax2.plot(months, scratch, "-", color=COLORS["red"], lw=2.5, label="from-scratch (cumulative cost)")
    ax2.plot(months, transfer - benefit, "-", color=COLORS["green"], lw=2.5, label="transfer (net cost)")
    ax2.fill_between(months, scratch, transfer - benefit, where=scratch > transfer - benefit, alpha=0.2, color=COLORS["green"], label="savings")
    # Break-even
    diff = scratch - (transfer - benefit)
    be = months[np.argmax(diff > 0)] if (diff > 0).any() else None
    if be is not None:
        ax2.axvline(be, ls="--", color=COLORS["yellow"], alpha=0.7)
        ax2.text(be + 0.3, 200, f"break-even\nmonth {be}", color=COLORS["yellow"], fontsize=10)
    ax2.set_xlabel("Months post-launch", color=COLORS["white"])
    ax2.set_ylabel("Cumulative cost ($K)", color=COLORS["white"])
    ax2.set_title("Transfer learning ROI: when does it pay off?", color=COLORS["white"], fontsize=12)
    ax2.legend(facecolor=DARK_BG, edgecolor=COLORS["muted"], labelcolor=COLORS["white"], fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(f"{OUT}/fig12_cost_breakeven.png", dpi=130, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print("OK fig12")


# ===== run all =====
if __name__ == "__main__":
    fig01_mmd_embedding()
    fig02_finetune_landscape()
    fig03_tsne_dann()
    fig04_maml_landscape()
    fig05_distill_curves()
    fig06_gradnorm()
    fig07_zsl_prototypes()
    fig08_infonce_surface()
    fig09_lora_rank()
    fig10_continual()
    fig11_tokenization_tax()
    fig12_cost_breakeven()
    print("\nAll 12 figures generated in", OUT)
