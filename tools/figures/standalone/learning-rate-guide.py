"""
Generate figures for the Learning Rate guide post.

Outputs 7 PNGs into both EN and ZH asset directories:
  fig1_lr_regimes.png       Too-small / just-right / too-large LR (3 panels)
  fig2_lr_schedules.png     Cosine, step, linear-warmup-decay, WSD on one chart
  fig3_lr_range_test.png    Classic LR range test (loss vs log-LR)
  fig4_adam_vs_sgd.png      Adam vs SGD with the same cosine schedule
  fig5_layerwise_lr.png     ULMFiT-style discriminative / layer-wise LR
  fig6_schedule_free.png    D-Adaptation / schedule-free vs cosine
  fig7_llm_schedule.png     Real-world LLM schedule (GPT-3 / LLaMA style)

Style: seaborn-v0_8-whitegrid, dpi=150, palette {#2563eb, #7c3aed, #10b981, #f59e0b}.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
AMBER = COLORS["warning"]
GRAY = COLORS["text2"]

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/standalone/learning-rate-guide"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/standalone/"
    "学习率-从入门到大模型训练的终极指南-2026"
)
EN_DIR.mkdir(parents=True, exist_ok=True)
ZH_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure into both EN and ZH asset directories."""
    fig.tight_layout()
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {name}")


# ---------------------------------------------------------------------------
# Figure 1: Three regimes -- too small / just right / too large
# ---------------------------------------------------------------------------
def fig1_regimes() -> None:
    a = 4.0  # curvature of L(theta) = 0.5 * a * theta^2
    theta0 = 2.0
    steps = 40
    settings = [
        ("Too small (eta=0.05)", 0.05, BLUE),
        ("Just right (eta=0.35)", 0.35, GREEN),
        ("Too large (eta=0.55)", 0.55, AMBER),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    theta_grid = np.linspace(-2.5, 2.5, 200)
    loss_grid = 0.5 * a * theta_grid ** 2

    for ax, (title, eta, color) in zip(axes, settings):
        theta = theta0
        traj_t = [theta]
        traj_l = [0.5 * a * theta ** 2]
        for _ in range(steps):
            g = a * theta
            theta = theta - eta * g
            traj_t.append(theta)
            traj_l.append(0.5 * a * theta ** 2)
        traj_t = np.array(traj_t)
        traj_l = np.array(traj_l)

        ax.plot(theta_grid, loss_grid, color=GRAY, lw=1.5, alpha=0.6, label="L(θ)")
        ax.plot(traj_t, traj_l, "o-", color=color, lw=1.5, ms=4.5, alpha=0.9)
        ax.scatter([traj_t[0]], [traj_l[0]], color="black", zorder=5, s=45,
                   label="start")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("parameter θ")
        ax.set_ylabel("loss L(θ)")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-0.5, 13)
        ax.legend(frameon=False, fontsize=9, loc="upper center")

    fig.suptitle(
        "Three learning-rate regimes on a 1-D quadratic L(θ)=½·a·θ²  (a=4)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    _save(fig, "fig1_lr_regimes.png")


# ---------------------------------------------------------------------------
# Figure 2: Common LR schedules on one axis
# ---------------------------------------------------------------------------
def fig2_schedules() -> None:
    total = 1000
    warmup = 80
    lr_max = 1.0
    lr_min = 0.1
    steps = np.arange(total)

    # cosine with warmup
    cosine = np.zeros_like(steps, dtype=float)
    for i, s in enumerate(steps):
        if s < warmup:
            cosine[i] = lr_max * (s + 1) / warmup
        else:
            t = (s - warmup) / (total - warmup)
            cosine[i] = lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * t))

    # step decay (drop by 0.5 every 250 steps after warmup)
    step_decay = np.zeros_like(steps, dtype=float)
    for i, s in enumerate(steps):
        if s < warmup:
            step_decay[i] = lr_max * (s + 1) / warmup
        else:
            drops = (s - warmup) // 250
            step_decay[i] = lr_max * (0.5 ** drops)

    # linear warmup then linear decay
    linear = np.zeros_like(steps, dtype=float)
    for i, s in enumerate(steps):
        if s < warmup:
            linear[i] = lr_max * (s + 1) / warmup
        else:
            t = (s - warmup) / (total - warmup)
            linear[i] = lr_max * (1 - t) + lr_min * t

    # WSD: warmup 8% -- stable -- cooldown last 20%
    cooldown = int(0.2 * total)
    stable_end = total - cooldown
    wsd = np.zeros_like(steps, dtype=float)
    for i, s in enumerate(steps):
        if s < warmup:
            wsd[i] = lr_max * (s + 1) / warmup
        elif s < stable_end:
            wsd[i] = lr_max
        else:
            t = (s - stable_end) / cooldown
            wsd[i] = lr_max + (lr_min - lr_max) * t

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(steps, cosine, color=BLUE, lw=2.2, label="Warmup + Cosine")
    ax.plot(steps, step_decay, color=AMBER, lw=2.2, label="Warmup + Step decay")
    ax.plot(steps, linear, color=PURPLE, lw=2.2, label="Linear warmup-decay")
    ax.plot(steps, wsd, color=GREEN, lw=2.4, label="WSD (Warmup-Stable-Decay)")

    ax.axvspan(0, warmup, color=BLUE, alpha=0.06)
    ax.axvspan(stable_end, total, color=GREEN, alpha=0.06)
    ax.text(warmup / 2, 0.02, "warmup", ha="center", color=GRAY, fontsize=9)
    ax.text((stable_end + total) / 2, 0.02, "cooldown\n(WSD)", ha="center",
            color=GRAY, fontsize=9)

    ax.set_xlabel("training step")
    ax.set_ylabel("learning rate (fraction of peak)")
    ax.set_title("Common learning-rate schedules", fontsize=13, fontweight="bold")
    ax.set_ylim(-0.02, 1.1)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    _save(fig, "fig2_lr_schedules.png")


# ---------------------------------------------------------------------------
# Figure 3: LR range test
# ---------------------------------------------------------------------------
def fig3_range_test() -> None:
    rng = np.random.default_rng(7)
    lrs = np.logspace(-7, 1, 220)

    # synthetic but realistic loss-vs-log-lr curve:
    # high constant loss, then descent, then rapid blow-up past stability
    log_lr = np.log10(lrs)
    base = 2.4 + 0.0 * log_lr
    descent = 1.8 / (1.0 + np.exp(-(log_lr + 3.5) * 1.6))   # decreasing component
    explosion = np.where(log_lr > -1.5,
                         3.5 * (1.0 / (1.0 + np.exp(-(log_lr + 0.2) * 4.0))),
                         0.0)
    loss = base - descent + explosion + rng.normal(0, 0.04, size=lrs.shape)

    smooth = np.convolve(loss, np.ones(7) / 7, mode="same")

    # interpretation points
    safe_idx = int(np.argmin(smooth))
    edge_idx = safe_idx + np.argmax(np.diff(smooth[safe_idx:]) > 0.02) if \
        np.any(np.diff(smooth[safe_idx:]) > 0.02) else safe_idx + 5
    peak_idx = int(np.argmin(np.abs(lrs - lrs[edge_idx] * 0.3)))

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(lrs, loss, color=GRAY, alpha=0.35, lw=1.0, label="raw loss")
    ax.plot(lrs, smooth, color=BLUE, lw=2.2, label="smoothed loss")

    ax.axvline(lrs[edge_idx], color=AMBER, lw=1.6, ls="--",
               label=f"stability edge ≈ {lrs[edge_idx]:.1e}")
    ax.axvline(lrs[peak_idx], color=GREEN, lw=1.6, ls="--",
               label=f"recommended peak LR ≈ {lrs[peak_idx]:.1e}")

    ax.set_xscale("log")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("training loss after 1 mini-batch")
    ax.set_title("LR range test: pick 0.3-1× the stability edge",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _save(fig, "fig3_lr_range_test.png")


# ---------------------------------------------------------------------------
# Figure 4: Adam vs SGD with the same cosine schedule
# ---------------------------------------------------------------------------
def fig4_adam_vs_sgd() -> None:
    rng = np.random.default_rng(3)
    steps = np.arange(0, 1000)
    warmup = 50
    lr_max = 1.0
    cosine = np.where(
        steps < warmup,
        lr_max * (steps + 1) / warmup,
        0.5 * (1 + np.cos(np.pi * (steps - warmup) / (1000 - warmup))),
    )

    # Loss curves -- Adam usually descends faster initially, SGD often
    # catches up or matches with proper schedule.
    base = 1.0 / (1.0 + 0.02 * steps)
    sgd_loss = 2.4 * base + 0.55 * np.exp(-steps / 250) + rng.normal(0, 0.015, steps.shape)
    adam_loss = 2.4 * base * 0.78 + 0.20 * np.exp(-steps / 200) + \
        rng.normal(0, 0.012, steps.shape)

    sgd_loss = np.maximum(sgd_loss, 0.18)
    adam_loss = np.maximum(adam_loss, 0.16)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6),
                             gridspec_kw={"width_ratios": [1.05, 1.45]})

    axes[0].plot(steps, cosine, color=PURPLE, lw=2.2)
    axes[0].set_title("Shared schedule: warmup + cosine", fontweight="bold")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("LR (fraction of peak)")
    axes[0].set_ylim(-0.02, 1.1)

    axes[1].plot(steps, sgd_loss, color=AMBER, lw=2.0, label="SGD + momentum")
    axes[1].plot(steps, adam_loss, color=BLUE, lw=2.0, label="AdamW")
    axes[1].set_title("Training loss under the same schedule", fontweight="bold")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("training loss")
    axes[1].legend(frameon=False, fontsize=10)

    fig.suptitle("Optimizer choice changes how aggressively the schedule can be used",
                 fontsize=13, fontweight="bold", y=1.03)
    _save(fig, "fig4_adam_vs_sgd.png")


# ---------------------------------------------------------------------------
# Figure 5: Layer-wise LR for fine-tuning (ULMFiT-style)
# ---------------------------------------------------------------------------
def fig5_layerwise() -> None:
    n_layers = 12
    layers = np.arange(n_layers)

    base_lr = 3e-4
    decay = 0.8  # each layer below the head uses lr * decay^k
    head_idx = n_layers - 1

    discriminative = np.array([base_lr * (decay ** (head_idx - i)) for i in layers])
    flat = np.full(n_layers, base_lr)

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    width = 0.38
    x = layers
    ax.bar(x - width / 2, flat, width, color=GRAY, alpha=0.55,
           label=f"Single LR (3e-4)")
    ax.bar(x + width / 2, discriminative, width, color=BLUE,
           label=f"Discriminative LR (factor={decay})")

    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i+1}\n(emb)" if i == 0 else
                        ("head" if i == head_idx else f"L{i+1}")
                        for i in layers], fontsize=9)
    ax.set_xlabel("layer (embeddings → task head)")
    ax.set_ylabel("learning rate (log scale)")
    ax.set_title("Layer-wise (discriminative) LR for fine-tuning — ULMFiT style",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    ax.annotate("preserve pretrained\nlow-level features",
                xy=(0.5, discriminative[0]), xytext=(2.2, 1.2e-5),
                fontsize=9, color=PURPLE,
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1))
    ax.annotate("aggressively learn\ntask-specific head",
                xy=(head_idx + 0.2, discriminative[-1]),
                xytext=(head_idx - 4.5, 6e-4),
                fontsize=9, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    _save(fig, "fig5_layerwise_lr.png")


# ---------------------------------------------------------------------------
# Figure 6: Schedule-free / D-Adaptation vs cosine
# ---------------------------------------------------------------------------
def fig6_schedule_free() -> None:
    rng = np.random.default_rng(11)
    steps = np.arange(0, 1000)

    cosine = 0.5 * (1 + np.cos(np.pi * steps / 1000))  # no warmup for clarity
    # schedule-free behaves close to the running average of a constant LR;
    # we emulate it by holding near 1.0 the whole time.
    sched_free = np.full_like(cosine, 1.0)

    base = 1.0 / (1.0 + 0.015 * steps)
    cosine_loss = 2.0 * base + 0.30 * np.exp(-steps / 220) + rng.normal(0, 0.012, steps.shape)
    sf_loss = 2.0 * base + 0.34 * np.exp(-steps / 200) + rng.normal(0, 0.014, steps.shape)
    cosine_loss = np.maximum(cosine_loss, 0.18)
    sf_loss = np.maximum(sf_loss, 0.20)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))

    axes[0].plot(steps, cosine, color=BLUE, lw=2.2, label="Cosine (needs total T)")
    axes[0].plot(steps, sched_free, color=GREEN, lw=2.2,
                 label="Schedule-free (no decay)")
    axes[0].set_title("Effective LR over training", fontweight="bold")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("LR (fraction of peak)")
    axes[0].legend(frameon=False, fontsize=10)
    axes[0].set_ylim(-0.05, 1.15)

    axes[1].plot(steps, cosine_loss, color=BLUE, lw=2.0, label="Cosine")
    axes[1].plot(steps, sf_loss, color=GREEN, lw=2.0,
                 label="Schedule-free / D-Adaptation")
    axes[1].axvline(700, color=GRAY, ls=":", lw=1)
    axes[1].text(710, 1.6, "extend training\nwithout retuning →",
                 fontsize=9, color=GRAY)
    axes[1].set_title("Training loss (synthetic, illustrative)", fontweight="bold")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("training loss")
    axes[1].legend(frameon=False, fontsize=10)

    fig.suptitle("Schedule-free optimizers: competitive without knowing total steps",
                 fontsize=13, fontweight="bold", y=1.03)
    _save(fig, "fig6_schedule_free.png")


# ---------------------------------------------------------------------------
# Figure 7: Real-world LLM schedule (GPT-3 / LLaMA inspired)
# ---------------------------------------------------------------------------
def fig7_llm_schedule() -> None:
    total = 300_000        # tokens dimension simplified to "steps"
    warmup = 2_000         # GPT-3-ish: 375M tokens warmup ~ small fraction
    lr_peak = 6e-4         # GPT-3 175B used 0.6e-4; LLaMA-7B used 3e-4
    lr_min = 0.1 * lr_peak # 10% of peak, common in LLM recipes

    steps = np.arange(total)
    lr = np.zeros_like(steps, dtype=float)
    for i, s in enumerate(steps):
        if s < warmup:
            lr[i] = lr_peak * (s + 1) / warmup
        else:
            t = (s - warmup) / (total - warmup)
            lr[i] = lr_min + (lr_peak - lr_min) * 0.5 * (1 + math.cos(math.pi * t))

    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.plot(steps, lr, color=BLUE, lw=2.2)
    ax.axvspan(0, warmup * 4, color=AMBER, alpha=0.12)
    ax.text(warmup * 2, lr_peak * 1.04, "linear warmup\n(~0.1-1% of steps)",
            ha="center", color=AMBER, fontsize=9)
    ax.axhline(lr_peak, color=GRAY, ls=":", lw=1)
    ax.axhline(lr_min, color=GRAY, ls=":", lw=1)
    ax.text(total * 0.98, lr_peak * 1.02, f"peak LR = {lr_peak:.0e}",
            ha="right", color=GRAY, fontsize=9)
    ax.text(total * 0.98, lr_min * 1.05, f"min LR = 10% of peak",
            ha="right", color=GRAY, fontsize=9)

    ax.set_xlabel("training step (~ tokens consumed)")
    ax.set_ylabel("learning rate")
    ax.set_title(
        "Typical LLM pretraining schedule (GPT-3 / LLaMA style):\n"
        "linear warmup → cosine decay to 10% of peak",
        fontsize=12.5, fontweight="bold",
    )
    ax.set_ylim(0, lr_peak * 1.18)
    _save(fig, "fig7_llm_schedule.png")


def main() -> None:
    fig1_regimes()
    fig2_schedules()
    fig3_range_test()
    fig4_adam_vs_sgd()
    fig5_layerwise()
    fig6_schedule_free()
    fig7_llm_schedule()
    print("\nAll figures generated.")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
