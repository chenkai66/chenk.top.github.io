#!/usr/bin/env python3
"""Transfer Learning series — generate 5 GIF animations.

Run on ai4m. Output: /tmp/tl-figs/anim_*.gif
Uses matplotlib FuncAnimation + PillowWriter (no imageio dep).
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

DARK_BG = "#0d1117"
COL = {"blue":"#58a6ff","orange":"#f78166","green":"#7ee787","purple":"#d2a8ff",
       "yellow":"#f1e05a","red":"#ff7b72","white":"#e6edf3","muted":"#8b949e"}
OUT = "/tmp/tl-figs"
os.makedirs(OUT, exist_ok=True)


def setup(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=COL["white"])
    for s in ax.spines.values(): s.set_color(COL["muted"])


# ============================================================
# Anim 1: Adversarial domain alignment (Art 03)
# Source/target features merge under DANN training
# ============================================================
def anim_dann_alignment():
    np.random.seed(0)
    n_classes = 3
    n_per = 80
    centers_s = np.array([(-1.5,-1.0),(1.5,-1.0),(0.0,1.5)])
    centers_t = centers_s + np.array([1.0, 0.7])  # initial offset

    fig, ax = plt.subplots(figsize=(7.5, 6))
    fig.patch.set_facecolor(DARK_BG); setup(ax)
    palette = [COL["blue"], COL["orange"], COL["green"]]

    src_pts = np.concatenate([np.random.randn(n_per,2)*0.4 + c for c in centers_s])
    tgt0 = np.concatenate([np.random.randn(n_per,2)*0.4 + c for c in centers_t])
    src_lbl = np.repeat(np.arange(n_classes), n_per)
    tgt_lbl = src_lbl.copy()

    src_sc = ax.scatter(src_pts[:,0], src_pts[:,1], c=[palette[l] for l in src_lbl],
                        s=22, alpha=0.7, edgecolors="none", label="source")
    tgt_sc = ax.scatter(tgt0[:,0], tgt0[:,1], c=[palette[l] for l in tgt_lbl],
                        s=22, alpha=0.7, marker="^", edgecolors="white", linewidths=0.4)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3, 3.5)
    ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title("DANN training, epoch 0", color=COL["white"], fontsize=13)
    txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, color=COL["white"], fontsize=10, va="top")

    n_frames = 60
    def update(i):
        # Target features migrate from offset toward source centers
        progress = i / (n_frames - 1)
        # Current target centers
        cur_centers = centers_t * (1 - progress) + centers_s * progress
        cur = np.concatenate([np.random.randn(n_per,2)*0.4 + c for c in cur_centers])
        np.random.seed(i)  # deterministic noise per frame
        tgt_sc.set_offsets(cur)
        title.set_text(f"DANN training, epoch {i+1}/{n_frames}")
        # Fake metrics
        loss_d = 0.69 * np.exp(-0.04 * i) + 0.05
        acc_t = 0.42 + 0.45 * (1 - np.exp(-0.06 * i))
        txt.set_text(f"discriminator loss: {loss_d:.3f}\ntarget accuracy:    {acc_t*100:.1f}%")
        return tgt_sc, title, txt

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
    anim.save(f"{OUT}/anim_dann_alignment.gif", writer=PillowWriter(fps=12))
    plt.close()
    print("OK anim_dann_alignment")


# ============================================================
# Anim 2: MAML training (Art 04)
# Inner loop curves descending; outer loop converging on theta_0
# ============================================================
def anim_maml_training():
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(DARK_BG)
    for ax in axes: setup(ax)

    # Left: 2D theta landscape with theta_0 walking toward center
    x = np.linspace(-3, 3, 100); y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    task_centers = np.array([(-1.5,1.0),(1.5,1.0),(0.0,-1.5)])
    Z = sum(1.5*np.exp(-((X-c[0])**2 + (Y-c[1])**2)/0.6) for c in task_centers)
    Z = -Z + 4.0
    axes[0].contourf(X, Y, Z, 18, cmap="viridis", alpha=0.5)
    axes[0].set_title("Outer-loop trajectory of $\\theta_0$", color=COL["white"], fontsize=12)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for c in task_centers:
        axes[0].scatter(*c, c=COL["green"], s=120, edgecolors=COL["white"], linewidths=1.5, zorder=4)
    theta_path_x, theta_path_y = [], []
    theta_pt = axes[0].scatter([], [], c=COL["red"], s=300, marker="*", edgecolors=COL["white"], linewidths=2, zorder=10)
    theta_line, = axes[0].plot([], [], "-", color=COL["yellow"], lw=1.5, alpha=0.7)

    # Right: meta-loss curve
    axes[1].set_xlabel("Outer iteration", color=COL["white"])
    axes[1].set_ylabel("Meta-loss (avg over tasks)", color=COL["white"])
    axes[1].set_title("Meta-loss decreasing", color=COL["white"], fontsize=12)
    axes[1].set_xlim(0, 60); axes[1].set_ylim(0, 1.2)
    axes[1].grid(True, alpha=0.15)
    loss_line, = axes[1].plot([], [], "-", color=COL["blue"], lw=2.5)
    losses = []

    n_frames = 60
    # theta_0 starts at corner, walks to (0, 0.2) (centroid)
    start = np.array([-2.5, 2.5]); end = np.array([0, 0.2])

    def update(i):
        t = i / (n_frames - 1)
        theta = start * (1 - t) + end * t + 0.04*np.random.randn(2)
        theta_path_x.append(theta[0]); theta_path_y.append(theta[1])
        theta_pt.set_offsets([theta])
        theta_line.set_data(theta_path_x, theta_path_y)
        # loss decay
        loss = 1.0 * np.exp(-0.06 * i) + 0.05
        losses.append(loss)
        loss_line.set_data(np.arange(len(losses)), losses)
        return theta_pt, theta_line, loss_line

    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
    anim.save(f"{OUT}/anim_maml_training.gif", writer=PillowWriter(fps=12))
    plt.close()
    print("OK anim_maml_training")


# ============================================================
# Anim 3: LoRA rank sweep (Art 09)
# F1 evolution as rank increases, multiple tasks
# ============================================================
def anim_lora_rank_sweep():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG); setup(ax)
    ranks = [1, 2, 4, 8, 16, 32, 64]
    sst2 = [88.4, 90.2, 91.5, 92.0, 92.3, 92.4, 92.4]
    mnli = [80.1, 82.5, 84.2, 85.1, 85.5, 85.6, 85.6]
    cola = [55.8, 58.4, 61.2, 63.0, 64.1, 64.7, 64.9]

    ax.set_xscale("log", base=2)
    ax.set_xlabel("LoRA rank $r$ (log scale)", color=COL["white"])
    ax.set_ylabel("Dev accuracy / F1 (%)", color=COL["white"])
    ax.set_title("LoRA — rank sensitivity sweep", color=COL["white"], fontsize=13)
    ax.set_xlim(0.8, 80); ax.set_ylim(50, 95)
    ax.grid(True, alpha=0.15, which="both")

    line1, = ax.plot([], [], "o-", color=COL["blue"],   lw=2.5, ms=11, label="SST-2 (easy)")
    line2, = ax.plot([], [], "o-", color=COL["orange"], lw=2.5, ms=11, label="MNLI (medium)")
    line3, = ax.plot([], [], "o-", color=COL["green"],  lw=2.5, ms=11, label="CoLA (hard)")
    txt = ax.text(0.96, 0.05, "", transform=ax.transAxes, ha="right", color=COL["yellow"], fontsize=14)
    ax.legend(facecolor=DARK_BG, edgecolor=COL["muted"], labelcolor=COL["white"], fontsize=11, loc="lower right")

    def update(i):
        idx = (i % len(ranks)) + 1
        line1.set_data(ranks[:idx], sst2[:idx])
        line2.set_data(ranks[:idx], mnli[:idx])
        line3.set_data(ranks[:idx], cola[:idx])
        txt.set_text(f"r = {ranks[idx-1]}")
        return line1, line2, line3, txt

    anim = FuncAnimation(fig, update, frames=len(ranks)*4, interval=400, blit=False)
    anim.save(f"{OUT}/anim_lora_rank_sweep.gif", writer=PillowWriter(fps=4))
    plt.close()
    print("OK anim_lora_rank_sweep")


# ============================================================
# Anim 4: Catastrophic forgetting (Art 10)
# Task 1 accuracy dropping while task 2 trains
# ============================================================
def anim_forgetting():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(DARK_BG); setup(ax)
    epochs = np.arange(0, 100)
    # Task 1 phase 0-49, Task 2 phase 50-99
    t1_acc, t2_acc, ewc_t1 = [], [], []

    line_t1, = ax.plot([], [], "-", color=COL["blue"], lw=2.5, label="Task 1 acc (naive SGD)")
    line_t2, = ax.plot([], [], "-", color=COL["orange"], lw=2.5, label="Task 2 acc (training)")
    line_ewc, = ax.plot([], [], "--", color=COL["green"], lw=2.5, label="Task 1 acc (EWC defended)")
    ax.axvline(50, ls="--", color=COL["muted"], alpha=0.5)
    ax.text(50.5, 92, "Switch to Task 2", color=COL["muted"], fontsize=10)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xlabel("Epoch", color=COL["white"]); ax.set_ylabel("Accuracy (%)", color=COL["white"])
    ax.set_title("Catastrophic forgetting — naive vs EWC", color=COL["white"], fontsize=13)
    ax.legend(facecolor=DARK_BG, edgecolor=COL["muted"], labelcolor=COL["white"], fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.15)

    def update(i):
        # Task 1 trains epochs 0-49, then naive forgets, EWC retains
        if i <= 50:
            t1_acc.append(20 + 70*(1 - np.exp(-0.08*i)))
            ewc_t1.append(20 + 70*(1 - np.exp(-0.08*i)))
            t2_acc.append(0)
        else:
            di = i - 50
            t1_acc.append(t1_acc[-1] - 1.6 + 0.2*np.random.randn())  # rapid forgetting
            t1_acc[-1] = max(15, t1_acc[-1])
            ewc_t1.append(ewc_t1[-1] - 0.2 + 0.1*np.random.randn())  # slow drift
            ewc_t1[-1] = max(78, ewc_t1[-1])
            t2_acc.append(20 + 70*(1 - np.exp(-0.1*di)))
        line_t1.set_data(np.arange(len(t1_acc)), t1_acc)
        line_t2.set_data(np.arange(len(t2_acc)), t2_acc)
        line_ewc.set_data(np.arange(len(ewc_t1)), ewc_t1)
        return line_t1, line_t2, line_ewc

    anim = FuncAnimation(fig, update, frames=100, interval=80, blit=False)
    anim.save(f"{OUT}/anim_forgetting.gif", writer=PillowWriter(fps=15))
    plt.close()
    print("OK anim_forgetting")


# ============================================================
# Anim 5: CLIP embedding alignment (Art 08)
# Image+text embeddings starting random, converging through contrastive training
# ============================================================
def anim_clip_alignment():
    np.random.seed(2)
    n_pairs = 12
    fig, ax = plt.subplots(figsize=(7.5, 7))
    fig.patch.set_facecolor(DARK_BG); setup(ax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("CLIP — image (●) and text (★) embeddings aligning", color=COL["white"], fontsize=12)

    # Final positions on a circle
    angles = np.linspace(0, 2*np.pi, n_pairs, endpoint=False)
    final_pts = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 1.5

    img_init = np.random.randn(n_pairs, 2) * 1.8
    txt_init = np.random.randn(n_pairs, 2) * 1.8

    palette = plt.cm.tab20(np.linspace(0, 1, n_pairs))
    img_sc = ax.scatter(img_init[:,0], img_init[:,1], c=palette, s=120, edgecolors=COL["white"], linewidths=1.5)
    txt_sc = ax.scatter(txt_init[:,0], txt_init[:,1], c=palette, s=200, marker="*", edgecolors=COL["white"], linewidths=1.5)
    lines = [ax.plot([], [], "-", color=palette[i], alpha=0.4, lw=1)[0] for i in range(n_pairs)]
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5)
    title2 = ax.text(0.5, -0.08, "", transform=ax.transAxes, ha="center", color=COL["yellow"], fontsize=12)

    n_frames = 50
    def update(i):
        t = i / (n_frames - 1)
        # Both image+text move toward final shared position with mild noise
        cur_img = img_init * (1-t) + final_pts * t + 0.04*np.random.randn(n_pairs, 2)
        cur_txt = txt_init * (1-t) + final_pts * t + 0.04*np.random.randn(n_pairs, 2)
        img_sc.set_offsets(cur_img)
        txt_sc.set_offsets(cur_txt)
        for k, ln in enumerate(lines):
            ln.set_data([cur_img[k,0], cur_txt[k,0]], [cur_img[k,1], cur_txt[k,1]])
        # InfoNCE loss
        loss = 2.5 * np.exp(-2.5 * t) + 0.05
        title2.set_text(f"InfoNCE loss: {loss:.3f}   |   step {i+1}/{n_frames}")
        return [img_sc, txt_sc, title2] + lines

    anim = FuncAnimation(fig, update, frames=n_frames, interval=110, blit=False)
    anim.save(f"{OUT}/anim_clip_alignment.gif", writer=PillowWriter(fps=10))
    plt.close()
    print("OK anim_clip_alignment")


if __name__ == "__main__":
    anim_dann_alignment()
    anim_maml_training()
    anim_lora_rank_sweep()
    anim_forgetting()
    anim_clip_alignment()
    print("\nAll 5 animations saved to", OUT)
