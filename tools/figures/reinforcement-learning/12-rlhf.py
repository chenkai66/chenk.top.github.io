"""Figures for Reinforcement Learning Part 12 — RLHF and LLM Applications.

Generates 7 publication-quality figures covering the alignment stack that
turned base language models into ChatGPT/Claude-class assistants.

Figures:
    fig1_rlhf_three_stage_pipeline   — SFT -> Reward Model -> PPO pipeline
    fig2_reward_model_training       — Bradley-Terry preference learning
    fig3_ppo_kl_constraint           — PPO with KL anchor to reference policy
    fig4_dpo_derivation              — From RLHF closed-form to the DPO loss
    fig5_bradley_terry               — Bradley-Terry sigmoid + preference data
    fig6_chatgpt_architecture        — End-to-end ChatGPT/Claude training stack
    fig7_reward_hacking              — Goodhart's law: proxy vs true reward

Output is written to BOTH the EN and ZH asset folders.

References (verified):
    - Christiano et al., Deep RL from Human Preferences, NeurIPS 2017
      [1706.03741]
    - Stiennon et al., Learning to summarize with human feedback, NeurIPS 2020
      [2009.01325]
    - Ouyang et al., Training language models to follow instructions with
      human feedback (InstructGPT), NeurIPS 2022 [2203.02155]
    - Bai et al., Training a Helpful and Harmless Assistant with RLHF,
      Anthropic 2022 [2204.05862]
    - Bai et al., Constitutional AI: Harmlessness from AI Feedback,
      Anthropic 2022 [2212.08073]
    - Rafailov et al., Direct Preference Optimization, NeurIPS 2023
      [2305.18290]
    - Lee et al., RLAIF: Scaling RL from Human Feedback with AI Feedback,
      2023 [2309.00267]
    - Bradley & Terry, Rank Analysis of Incomplete Block Designs, 1952
    - Gao et al., Scaling Laws for Reward Model Overoptimization,
      ICML 2023 [2210.10760]
    - Skalse et al., Defining and Characterizing Reward Hacking,
      NeurIPS 2022 [2209.13085]
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Shared aesthetic style (chenk-site)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
ORANGE = COLORS["warning"]
GRAY = COLORS["text2"]
LIGHT = COLORS["grid"]
DARK = COLORS["text"]
RED = COLORS["danger"]

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/reinforcement-learning/12-rlhf-and-llm-applications"
ZH_DIR = REPO / "source/_posts/zh/reinforcement-learning/12-RLHF与大语言模型应用"


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def _box(ax, xy, w, h, text, color, fontsize=10, text_color="white",
         alpha=1.0, weight="bold"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                facecolor=color, edgecolor=color,
                                alpha=alpha, linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, weight=weight)


def _arrow(ax, p0, p1, color=DARK, lw=1.6, style="-|>", mut=14):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style,
                                 color=color, lw=lw,
                                 mutation_scale=mut))


# ---------------------------------------------------------------------------
# Fig 1: Three-stage RLHF pipeline (SFT -> RM -> PPO)
# ---------------------------------------------------------------------------
def fig1_rlhf_three_stage_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6.4))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title("RLHF Three-Stage Pipeline (InstructGPT / ChatGPT / Claude)",
                 fontsize=14, weight="bold", pad=10)

    # Pretrained base
    _box(ax, (0.2, 2.7), 1.9, 1.0,
         "Pretrained\nBase LLM\n(GPT-3, Llama)", GRAY, fontsize=9)

    # Stage 1: SFT
    _box(ax, (2.6, 4.6), 3.4, 1.4,
         "Stage 1 — SFT\nFine-tune on\n~13K demonstrations", BLUE, fontsize=10)
    _box(ax, (2.6, 2.7), 3.4, 1.0,
         r"$\mathcal{L}_{\rm SFT} = -\sum_t \log \pi_\theta(y_t | x, y_{<t})$",
         "white", text_color=DARK, fontsize=10, weight="normal", alpha=1.0)
    ax.add_patch(Rectangle((2.6, 2.7), 3.4, 1.0, facecolor="white",
                            edgecolor=BLUE, lw=1.2))
    ax.text(2.6 + 1.7, 2.7 + 0.5,
            r"$\mathcal{L}_{\rm SFT}=-\sum_t\log\pi_\theta(y_t|x,y_{<t})$",
            ha="center", va="center", color=DARK, fontsize=10)
    _box(ax, (2.6, 1.0), 3.4, 1.4,
         "Output: π_SFT\n(can follow\ninstructions)", BLUE, fontsize=9,
         alpha=0.65)

    # Stage 2: Reward Model
    _box(ax, (6.4, 4.6), 3.4, 1.4,
         "Stage 2 — Reward Model\n~33K preference pairs\n(y_w ≻ y_l)", PURPLE,
         fontsize=10)
    ax.add_patch(Rectangle((6.4, 2.7), 3.4, 1.0, facecolor="white",
                            edgecolor=PURPLE, lw=1.2))
    ax.text(6.4 + 1.7, 2.7 + 0.5,
            r"$\mathcal{L}_{\rm RM}=-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))$",
            ha="center", va="center", color=DARK, fontsize=9.5)
    _box(ax, (6.4, 1.0), 3.4, 1.4,
         "Output: r_φ(x, y)\nscalar score\nproxy for humans", PURPLE,
         fontsize=9, alpha=0.65)

    # Stage 3: PPO
    _box(ax, (10.2, 4.6), 4.0, 1.4,
         "Stage 3 — PPO Fine-tuning\nMaximize reward,\nstay near π_ref", GREEN,
         fontsize=10)
    ax.add_patch(Rectangle((10.2, 2.7), 4.0, 1.0, facecolor="white",
                            edgecolor=GREEN, lw=1.2))
    ax.text(10.2 + 2.0, 2.7 + 0.5,
            r"$\max_\pi\;\mathbb{E}[r_\phi(x,y)]-\beta\,D_{\rm KL}[\pi\|\pi_{\rm ref}]$",
            ha="center", va="center", color=DARK, fontsize=9.5)
    _box(ax, (10.2, 1.0), 4.0, 1.4,
         "Output: aligned π*\n(helpful, honest,\nharmless)", GREEN, fontsize=9,
         alpha=0.65)

    # Arrows between stages
    _arrow(ax, (2.1, 3.2), (2.6, 3.2), color=DARK)
    _arrow(ax, (6.0, 5.3), (6.4, 5.3), color=DARK)
    _arrow(ax, (9.8, 5.3), (10.2, 5.3), color=DARK)

    # Loop arrow: SFT -> PPO as π_ref
    ax.add_patch(FancyArrowPatch((4.3, 1.0), (12.2, 0.4),
                                 connectionstyle="arc3,rad=-0.18",
                                 arrowstyle="-|>", color=ORANGE, lw=1.4,
                                 mutation_scale=14))
    ax.text(8.3, 0.18, "π_SFT used as reference π_ref (KL anchor)",
            ha="center", color=ORANGE, fontsize=9, style="italic")

    # Footer key insight
    ax.text(7.25, 6.15,
            "Three artifacts, three losses, one model: each stage cheaper than"
            " the next but harder to debug.",
            ha="center", color=GRAY, fontsize=9.5, style="italic")

    save(fig, "fig1_rlhf_three_stage_pipeline.png")


# ---------------------------------------------------------------------------
# Fig 2: Reward model training (preference pairs + Bradley-Terry)
# ---------------------------------------------------------------------------
def fig2_reward_model_training() -> None:
    fig = plt.figure(figsize=(14.0, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.25)

    # ---- Left: preference data flow ----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Preference data → Bradley-Terry loss",
                 fontsize=12, weight="bold", pad=8)

    # Prompt
    _box(ax, (0.4, 4.6), 2.4, 1.0,
         "Prompt x\n\"Explain RLHF\"", BLUE, fontsize=9)

    # Two completions
    _box(ax, (3.4, 5.0), 2.6, 0.9,
         "Completion y_A\n(thorough)", GREEN, fontsize=9, alpha=0.85)
    _box(ax, (3.4, 3.6), 2.6, 0.9,
         "Completion y_B\n(generic)", ORANGE, fontsize=9, alpha=0.85)
    _arrow(ax, (2.8, 5.1), (3.4, 5.4))
    _arrow(ax, (2.8, 4.8), (3.4, 4.05))

    # Annotator
    _box(ax, (6.5, 4.2), 1.6, 1.4,
         "Human\nannotator\nranks", DARK, fontsize=9)
    _arrow(ax, (6.0, 5.4), (6.5, 5.0))
    _arrow(ax, (6.0, 4.05), (6.5, 4.6))

    ax.text(8.45, 5.1, "y_w = y_A", color=GREEN, fontsize=10, weight="bold")
    ax.text(8.45, 4.55, "y_l = y_B", color=ORANGE, fontsize=10, weight="bold")
    ax.text(8.45, 4.0, "(y_w ≻ y_l)", color=DARK, fontsize=9, style="italic")

    # Reward model
    _box(ax, (3.4, 1.5), 2.6, 1.0,
         "Reward Model r_φ\n(LLM + scalar head)", PURPLE, fontsize=9)
    _arrow(ax, (4.7, 3.5), (4.7, 2.55))

    ax.text(5.0, 0.6,
            r"Loss: $-\log\sigma(r_\phi(x,y_w)-r_\phi(x,y_l))$",
            ha="center", color=DARK, fontsize=10)
    _arrow(ax, (4.7, 1.45), (4.7, 0.95))

    # ---- Right: Bradley-Terry probability surface ----
    ax2 = fig.add_subplot(gs[1])
    delta = np.linspace(-5, 5, 400)
    p_pref = 1.0 / (1.0 + np.exp(-delta))
    loss = -np.log(p_pref + 1e-12)
    ax2.plot(delta, p_pref, color=PURPLE, lw=2.4,
             label=r"$P(y_w \succ y_l)=\sigma(\Delta r)$")
    ax2.plot(delta, loss / 5.0, color=ORANGE, lw=2.0, ls="--",
             label=r"$\mathcal{L}_{\rm RM}/5$ (per-sample)")
    ax2.axvline(0, color=GRAY, lw=0.8, ls=":")
    ax2.fill_between(delta, 0, p_pref, where=delta < 0,
                     color=ORANGE, alpha=0.10)
    ax2.fill_between(delta, 0, p_pref, where=delta >= 0,
                     color=GREEN, alpha=0.12)
    ax2.set_xlabel(r"$\Delta r = r_\phi(x,y_w) - r_\phi(x,y_l)$")
    ax2.set_ylabel("Probability / scaled loss")
    ax2.set_title("Bradley-Terry: preferences as a sigmoid",
                  fontsize=12, weight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="center right", fontsize=9)
    ax2.text(2.6, 0.18, "model already correct\n(small gradient)",
             ha="center", color=GREEN, fontsize=8.5)
    ax2.text(-2.6, 0.78, "model wrong\n(large gradient)",
             ha="center", color=RED, fontsize=8.5)

    save(fig, "fig2_reward_model_training.png")


# ---------------------------------------------------------------------------
# Fig 3: PPO with KL constraint to reference model
# ---------------------------------------------------------------------------
def fig3_ppo_kl_constraint() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))

    # ---- Left: trajectory of policy in 2D parameter space ----
    ax = axes[0]
    theta_x = np.linspace(-3, 3, 200)
    theta_y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(theta_x, theta_y)
    # Reward landscape (proxy reward, has spurious peak)
    reward = 0.7 * np.exp(-((X - 1.4) ** 2 + (Y - 1.0) ** 2) / 1.2) + \
             1.1 * np.exp(-((X - 2.5) ** 2 + (Y + 1.6) ** 2) / 0.5)  # hack
    cs = ax.contourf(X, Y, reward, levels=14, cmap="YlOrRd", alpha=0.85)
    plt.colorbar(cs, ax=ax, fraction=0.045, pad=0.03,
                 label="proxy reward r_φ(x,y)")

    # KL ball around reference
    ref = (-1.4, -0.5)
    for r, a in [(2.6, 0.06), (2.0, 0.10), (1.4, 0.16)]:
        ax.add_patch(plt.Circle(ref, r, color=BLUE, alpha=a, lw=0))
    ax.plot(*ref, "o", color=BLUE, ms=12, zorder=5)
    ax.annotate("π_ref\n(SFT)", ref, xytext=(-2.7, -1.4),
                fontsize=10, color=BLUE, weight="bold")

    # PPO trajectory (stays near KL ball, moderate-reward peak)
    traj = np.array([
        [-1.4, -0.5], [-0.7, -0.2], [0.0, 0.2], [0.5, 0.5],
        [1.0, 0.8], [1.3, 0.95], [1.4, 1.0],
    ])
    ax.plot(traj[:, 0], traj[:, 1], "-o", color=GREEN, lw=2.0, ms=5,
            label="PPO + KL (β·D_KL)")
    ax.plot(traj[-1, 0], traj[-1, 1], "*", color=GREEN, ms=18,
            markeredgecolor=DARK)

    # Naive RL trajectory: chases hack
    naive = np.array([
        [-1.4, -0.5], [-0.5, -0.8], [0.5, -1.0], [1.5, -1.3],
        [2.2, -1.5], [2.5, -1.6],
    ])
    ax.plot(naive[:, 0], naive[:, 1], "--o", color=RED, lw=1.6, ms=4,
            label="No KL (reward hack)")
    ax.plot(naive[-1, 0], naive[-1, 1], "X", color=RED, ms=14)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title("KL anchor keeps π near π_ref",
                 fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=9, facecolor="white",
              framealpha=0.85, frameon=True)

    # ---- Right: KL coefficient effect on reward & coherence ----
    ax2 = axes[1]
    beta = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3])
    reward_curve = np.array([5.8, 5.2, 4.7, 4.2, 3.6, 3.0, 2.0])  # proxy
    true_quality = np.array([1.2, 3.4, 4.4, 4.6, 4.5, 4.2, 3.5])  # human
    ax2.plot(beta, reward_curve, "-o", color=ORANGE, lw=2.2,
             label="proxy reward r_φ")
    ax2.plot(beta, true_quality, "-s", color=GREEN, lw=2.2,
             label="true human quality")
    ax2.axvspan(0.008, 0.03, color=BLUE, alpha=0.10)
    ax2.text(0.018, 5.6, "sweet spot\n(β ≈ 0.01-0.03)", ha="center",
             color=BLUE, fontsize=9)
    ax2.set_xscale("symlog", linthresh=0.005)
    ax2.set_xlabel(r"KL coefficient $\beta$")
    ax2.set_ylabel("score (a.u.)")
    ax2.set_title("Choosing β: proxy ↑ ≠ humans happy",
                  fontsize=12, weight="bold")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.set_ylim(0, 6.5)

    plt.tight_layout()
    save(fig, "fig3_ppo_kl_constraint.png")


# ---------------------------------------------------------------------------
# Fig 4: DPO derivation (closed form, no RL)
# ---------------------------------------------------------------------------
def fig4_dpo_derivation() -> None:
    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("From RLHF to DPO: eliminating the reward model",
                 fontsize=13, weight="bold", pad=10)

    # Step 1
    _box(ax, (0.3, 4.7), 4.2, 1.0, "Step 1 — KL-regularised RL objective",
         BLUE, fontsize=10)
    ax.text(2.4, 4.05,
            r"$\max_\pi\; \mathbb{E}_{x,y}[r(x,y)]"
            r" - \beta\,D_{\rm KL}[\pi\|\pi_{\rm ref}]$",
            ha="center", fontsize=11, color=DARK)

    # Step 2
    _box(ax, (4.9, 4.7), 4.2, 1.0, "Step 2 — Closed-form optimum",
         PURPLE, fontsize=10)
    ax.text(7.0, 4.0,
            r"$\pi^*(y|x)=\frac{1}{Z(x)}\pi_{\rm ref}(y|x)"
            r"\exp\!\left(\frac{r(x,y)}{\beta}\right)$",
            ha="center", fontsize=11, color=DARK)

    # Step 3
    _box(ax, (9.5, 4.7), 4.2, 1.0, "Step 3 — Invert for r(x,y)",
         GREEN, fontsize=10)
    ax.text(11.6, 4.0,
            r"$r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\rm ref}(y|x)}+\beta\log Z(x)$",
            ha="center", fontsize=10, color=DARK)

    _arrow(ax, (4.5, 5.2), (4.9, 5.2))
    _arrow(ax, (9.1, 5.2), (9.5, 5.2))

    # Down arrow
    _arrow(ax, (11.6, 3.7), (11.6, 3.1))

    # Step 4: substitute into BT
    _box(ax, (8.0, 1.8), 5.7, 1.2,
         "Step 4 — Substitute into Bradley-Terry preference\n"
         "(Z(x) cancels, reward model disappears)", ORANGE, fontsize=10)

    # Final loss box
    ax.add_patch(FancyBboxPatch((0.6, 1.0), 7.0, 2.1,
                                 boxstyle="round,pad=0.04",
                                 facecolor="white", edgecolor=DARK, lw=2))
    ax.text(4.1, 2.7, "DPO loss (single supervised pass)",
            ha="center", fontsize=11, weight="bold", color=DARK)
    ax.text(4.1, 1.85,
            r"$\mathcal{L}_{\rm DPO}=-\,\mathbb{E}\!\left[\log\sigma\!\left("
            r"\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\rm ref}(y_w|x)}"
            r"-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\rm ref}(y_l|x)}\right)\right]$",
            ha="center", fontsize=11, color=DARK)
    ax.text(4.1, 1.25,
            "no reward model · no rollouts · no PPO clipping",
            ha="center", fontsize=9.5, color=GRAY, style="italic")

    _arrow(ax, (8.0, 2.4), (7.6, 2.1))

    # Footer comparison strip
    ax.text(7.0, 0.35,
            "RLHF: 3 models in memory + sampling loop  vs.  "
            "DPO: 2 models, plain log-likelihood backprop",
            ha="center", color=GRAY, fontsize=10, style="italic")

    save(fig, "fig4_dpo_derivation.png")


# ---------------------------------------------------------------------------
# Fig 5: Bradley-Terry preference distribution
# ---------------------------------------------------------------------------
def fig5_bradley_terry() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # ---- Left: BT sigmoid family ----
    ax = axes[0]
    delta = np.linspace(-6, 6, 400)
    for beta_val, color in [(0.5, BLUE), (1.0, PURPLE), (2.0, GREEN)]:
        ax.plot(delta, 1.0 / (1.0 + np.exp(-beta_val * delta)),
                color=color, lw=2.2,
                label=fr"$\beta={beta_val}$ (temperature)")
    ax.axvline(0, color=GRAY, lw=0.8, ls=":")
    ax.axhline(0.5, color=GRAY, lw=0.8, ls=":")
    ax.set_xlabel(r"$r(x,y_w)-r(x,y_l)$")
    ax.set_ylabel(r"$P(y_w \succ y_l)$")
    ax.set_title("Bradley-Terry model (1952)\nP(A beats B) = σ(s_A − s_B)",
                 fontsize=12, weight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(-0.02, 1.02)

    # ---- Right: empirical preference distribution + annotator noise ----
    ax2 = axes[1]
    rng = np.random.default_rng(7)
    # Generate latent reward gaps and observed preferences
    gaps = rng.normal(loc=0.0, scale=2.0, size=2000)
    p = 1.0 / (1.0 + np.exp(-gaps))
    obs = rng.binomial(1, p)
    # Bin by gap, plot empirical fraction
    bins = np.linspace(-6, 6, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means, counts = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (gaps >= lo) & (gaps < hi)
        if mask.sum() > 0:
            means.append(obs[mask].mean())
            counts.append(mask.sum())
        else:
            means.append(np.nan)
            counts.append(0)
    means = np.array(means)
    counts = np.array(counts)
    sizes = 25 + 6 * counts
    ax2.scatter(centers, means, s=sizes, color=PURPLE, alpha=0.55,
                edgecolor=DARK, linewidth=0.6, label="empirical (size = N)")
    ax2.plot(np.linspace(-6, 6, 200),
             1.0 / (1.0 + np.exp(-np.linspace(-6, 6, 200))),
             color=GREEN, lw=2.2, label="Bradley-Terry fit")
    # Inter-annotator ceiling
    ax2.axhline(0.78, color=ORANGE, ls="--", lw=1.4)
    ax2.text(-5.6, 0.81, "human–human agreement ≈ 78%\n(InstructGPT)",
             color=ORANGE, fontsize=8.5)

    ax2.set_xlabel(r"latent reward gap")
    ax2.set_ylabel(r"P(annotator picks y_w)")
    ax2.set_title("Why preferences (not scores)?\nMonotone & noise-robust",
                  fontsize=12, weight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    save(fig, "fig5_bradley_terry.png")


# ---------------------------------------------------------------------------
# Fig 6: ChatGPT / Claude end-to-end training architecture
# ---------------------------------------------------------------------------
def fig6_chatgpt_architecture() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.2))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("From web text to aligned assistant: the full training stack",
                 fontsize=14, weight="bold", pad=10)

    # Layer 1: Pretraining
    _box(ax, (0.5, 5.6), 13.5, 1.1,
         "Pretraining — trillions of tokens (Common Crawl, books, code)\n"
         "self-supervised next-token prediction · 90-99% of compute",
         GRAY, fontsize=10)

    # Layer 2: SFT / instruction tuning
    _box(ax, (0.5, 4.1), 4.3, 1.2,
         "SFT / Instruction Tuning\n10K-100K curated\nprompt→response pairs",
         BLUE, fontsize=10)

    # Layer 3: Preference modelling — fork RLHF vs RLAIF vs CAI
    _box(ax, (5.2, 4.1), 4.3, 1.2,
         "Preference data\nHuman labels (RLHF)\nor AI labels (RLAIF / CAI)",
         PURPLE, fontsize=10)

    _box(ax, (9.9, 4.1), 4.1, 1.2,
         "Reward Model r_φ\nor implicit reward\n(DPO, IPO, KTO)",
         PURPLE, fontsize=10, alpha=0.85)

    # Layer 4: Optimization
    _box(ax, (0.5, 2.4), 4.3, 1.3,
         "Policy Optimization\nPPO + KL anchor\n(OpenAI, Anthropic v1)",
         GREEN, fontsize=10)
    _box(ax, (5.2, 2.4), 4.3, 1.3,
         "Direct Optimization\nDPO / IPO / KTO\n(Llama-3, Qwen, Mistral)",
         GREEN, fontsize=10, alpha=0.85)
    _box(ax, (9.9, 2.4), 4.1, 1.3,
         "Constitutional AI\nself-critique loop\n(Anthropic Claude)",
         ORANGE, fontsize=10)

    # Layer 5: Deployment
    _box(ax, (0.5, 0.7), 13.5, 1.2,
         "Deployment-time guardrails\n"
         "system prompts · safety classifiers · tool-use scaffolding · "
         "online red-teaming · evals (MT-Bench, Arena)",
         DARK, fontsize=10)

    # Connectors
    for x in (2.65, 7.35, 11.95):
        _arrow(ax, (x, 5.55), (x, 5.35), color=GRAY, lw=1.0)
    _arrow(ax, (4.8, 4.7), (5.2, 4.7), color=GRAY, lw=1.0)
    _arrow(ax, (9.5, 4.7), (9.9, 4.7), color=GRAY, lw=1.0)
    for x in (2.65, 7.35, 11.95):
        _arrow(ax, (x, 4.05), (x, 3.75), color=GRAY, lw=1.0)
        _arrow(ax, (x, 2.35), (x, 1.95), color=GRAY, lw=1.0)

    # Side caption: which lab uses what
    ax.text(7.25, 0.25,
            "Same scaffold, different choices: OpenAI ⇒ PPO+RLHF · "
            "Anthropic ⇒ PPO+CAI · Meta/Mistral/Qwen ⇒ DPO families",
            ha="center", color=GRAY, fontsize=9.5, style="italic")

    save(fig, "fig6_chatgpt_architecture.png")


# ---------------------------------------------------------------------------
# Fig 7: Reward hacking / Goodhart's law
# ---------------------------------------------------------------------------
def fig7_reward_hacking() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.6))

    # ---- Left: proxy vs true reward over training ----
    ax = axes[0]
    steps = np.linspace(0, 1.0, 200)
    proxy = 1.0 - np.exp(-3.5 * steps)  # monotone up
    proxy = proxy / proxy.max() * 0.95
    # True reward rises then falls — overoptimization (Gao et al. 2023)
    true_r = 0.85 * np.exp(-((steps - 0.32) ** 2) / 0.045) + 0.15
    true_r = np.clip(true_r, 0, None)

    ax.plot(steps, proxy, color=ORANGE, lw=2.4,
            label="proxy reward r_φ (RM score)")
    ax.plot(steps, true_r, color=GREEN, lw=2.4,
            label="true reward (gold humans)")
    # Mark divergence
    diverge_idx = int(0.42 * len(steps))
    ax.axvline(steps[diverge_idx], color=RED, ls="--", lw=1.2)
    ax.text(steps[diverge_idx] + 0.02, 0.05,
            "overoptimisation\nbegins", color=RED, fontsize=9)
    ax.fill_between(steps, true_r, proxy, where=proxy > true_r,
                    color=RED, alpha=0.10, label="Goodhart gap")

    ax.set_xlabel("training KL from π_ref  (effective dose)")
    ax.set_ylabel("reward / quality (normalised)")
    ax.set_title("Goodhart's law in RLHF\n(Gao et al., ICML 2023)",
                 fontsize=12, weight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)

    # ---- Right: gallery of failure modes ----
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.axis("off")
    ax2.set_title("Common reward-hacking failure modes",
                  fontsize=12, weight="bold")

    failures = [
        ("Length hacking", "answers grow to 2-3× length\nfor marginal score gain",
         BLUE),
        ("Sycophancy", "model agrees with the user even\nwhen the user is wrong",
         PURPLE),
        ("Format hacking", "bullet-list everything; reward model\nlikes structure",
         ORANGE),
        ("Confident BS", "fluent, formatted, wrong\n(reward model can't fact-check)",
         RED),
        ("Refusal creep", "over-refuses benign queries to\nhedge harmlessness reward",
         GREEN),
    ]
    for i, (title, body, col) in enumerate(failures):
        y = 5.0 - i * 1.05
        ax2.add_patch(FancyBboxPatch((0.2, y - 0.45), 0.5, 0.9,
                                      boxstyle="round,pad=0.02",
                                      facecolor=col, edgecolor=col))
        ax2.text(0.45, y, str(i + 1), ha="center", va="center",
                 color="white", fontsize=11, weight="bold")
        ax2.text(0.95, y + 0.18, title, fontsize=10, weight="bold", color=DARK)
        ax2.text(0.95, y - 0.22, body, fontsize=9, color=GRAY)

    plt.tight_layout()
    save(fig, "fig7_reward_hacking.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_rlhf_three_stage_pipeline()
    fig2_reward_model_training()
    fig3_ppo_kl_constraint()
    fig4_dpo_derivation()
    fig5_bradley_terry()
    fig6_chatgpt_architecture()
    fig7_reward_hacking()
    print("Wrote 7 figures to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
