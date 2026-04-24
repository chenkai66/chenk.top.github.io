"""
Figure generation for Reinforcement Learning Part 03:
Policy Gradient and Actor-Critic Methods.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single, specific idea cleanly and can stand alone.

Figures:
    fig1_policy_gradient_theorem
        Visualises the score-function update on a 1D action distribution:
        the gradient pushes probability mass toward high-reward actions,
        away from low-reward ones.

    fig2_variance_reduction
        REINFORCE (raw return) vs Actor-Critic (advantage) sample-by-sample
        gradient signal. Demonstrates how using A = Q - V collapses noise
        without changing the expected gradient.

    fig3_actor_critic_architecture
        Architecture diagram with two heads (actor + critic) sharing a
        backbone, including the data-flow arrows for one update step.

    fig4_advantage_decomposition
        Q(s, a), V(s) and A(s, a) for several actions in one state.
        Makes "how much better than average" geometrically obvious.

    fig5_gae_lambda_sweep
        Sweep of lambda in GAE: bias-variance trade-off curve, plus the
        underlying multi-step return weighting kernel.

    fig6_action_policies
        Side-by-side: discrete softmax categorical vs continuous Gaussian
        squashed-Gaussian policy. Same input, different output families.

    fig7_policy_optimization_landscape
        Stochastic gradient ascent in policy parameter space:
        contours of J(theta), trajectory of SGD updates, optima and
        local plateaus. Conveys "we're climbing a noisy hill in theta".

Usage:
    python3 scripts/figures/reinforcement-learning/03-policy-gradient.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the
    markdown references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
C_BLUE = COLORS["primary"]     # primary
C_PURPLE = COLORS["accent"]   # secondary
C_GREEN = COLORS["success"]    # accent / good
C_AMBER = COLORS["warning"]    # warning / highlight
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_BG = COLORS["bg"]

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (REPO_ROOT / "source" / "_posts" / "en" / "reinforcement-learning"
          / "03-policy-gradient-and-actor-critic")
ZH_DIR = (REPO_ROOT / "source" / "_posts" / "zh" / "reinforcement-learning"
          / "03-Policy-Gradient与Actor-Critic方法")


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1: Policy gradient theorem - score-function update on a distribution
# ---------------------------------------------------------------------------
def fig1_policy_gradient_theorem() -> None:
    """Show how the policy gradient reshapes pi(a|s) toward high-reward a."""
    a = np.linspace(-4, 4, 400)

    # Initial policy: Gaussian centered slightly off the optimum.
    p_before = np.exp(-0.5 * ((a - (-0.5)) / 1.1) ** 2)
    p_before /= p_before.sum() * (a[1] - a[0])

    # Reward landscape: a smooth bump favouring a ~ +1.5.
    reward = np.exp(-0.5 * ((a - 1.5) / 1.2) ** 2)

    # After one (large, illustrative) PG update.
    p_after = np.exp(-0.5 * ((a - 0.9) / 0.95) ** 2)
    p_after /= p_after.sum() * (a[1] - a[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1, 1]})

    # --- Left: distribution before / after -----------------------------------
    ax = axes[0]
    ax.fill_between(a, p_before, color=C_GRAY, alpha=0.35,
                    label=r"$\pi_\theta(a|s)$ before")
    ax.plot(a, p_before, color=C_GRAY, lw=2)
    ax.fill_between(a, p_after, color=C_BLUE, alpha=0.35,
                    label=r"$\pi_\theta(a|s)$ after update")
    ax.plot(a, p_after, color=C_BLUE, lw=2.4)

    # Reward as a dashed twin overlay.
    ax2 = ax.twinx()
    ax2.plot(a, reward, color=C_AMBER, lw=2.2, ls="--",
             label=r"reward $r(a)$")
    ax2.set_ylim(0, 1.25)
    ax2.set_ylabel("reward (a.u.)", color=C_AMBER, fontsize=10)
    ax2.tick_params(axis="y", colors=C_AMBER)
    ax2.grid(False)

    # Annotate the shift with an arrow.
    ax.annotate("", xy=(0.9, 0.34), xytext=(-0.5, 0.34),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=2.2))
    ax.text(0.2, 0.39, "policy mass\nmoves toward\nhigh reward",
            ha="center", color=C_GREEN, fontsize=10, fontweight="bold")

    ax.set_xlabel("action $a$", fontsize=11)
    ax.set_ylabel("probability density", fontsize=11)
    ax.set_title("Policy gradient reshapes $\\pi_\\theta(a|s)$",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_xlim(-4, 4)

    # --- Right: score function and update direction --------------------------
    ax = axes[1]
    # Score for the BEFORE Gaussian: d/dtheta log N(a; mu, sigma) wrt mu.
    sigma = 1.1
    mu = -0.5
    score = (a - mu) / sigma ** 2  # gradient wrt mean parameter

    # Weight by reward to get the per-action contribution to grad J.
    weighted = score * reward * p_before * 80  # rescale for visibility

    ax.axhline(0, color=C_DARK, lw=0.8)
    ax.plot(a, score, color=C_PURPLE, lw=2,
            label=r"score $\nabla_\theta \log\pi_\theta(a|s)$")
    ax.plot(a, weighted, color=C_GREEN, lw=2.4,
            label=r"weighted: score $\cdot\, r(a) \cdot \pi(a|s)$")
    ax.fill_between(a, weighted, where=(weighted > 0),
                    color=C_GREEN, alpha=0.2)
    ax.fill_between(a, weighted, where=(weighted < 0),
                    color=C_RED, alpha=0.2)

    ax.text(2.2, 1.5, "increase\n$P(a)$ here",
            color=C_GREEN, fontsize=10, fontweight="bold", ha="center")
    ax.text(-2.6, -1.5, "decrease\n$P(a)$ here",
            color=C_RED, fontsize=10, fontweight="bold", ha="center")

    ax.set_xlabel("action $a$", fontsize=11)
    ax.set_ylabel("gradient signal (a.u.)", fontsize=11)
    ax.set_title("Score function $\\times$ reward = update direction",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3.5, 3.5)

    fig.suptitle("Policy Gradient Theorem in one picture",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_policy_gradient_theorem")


# ---------------------------------------------------------------------------
# Figure 2: Variance reduction REINFORCE vs A2C (advantage)
# ---------------------------------------------------------------------------
def fig2_variance_reduction() -> None:
    """Empirical demo: replacing G_t by A_t shrinks gradient variance ~10x."""
    rng = np.random.default_rng(2025)
    steps = 400

    # Simulated true value function: rises then plateaus.
    t = np.arange(steps)
    true_V = 60 + 40 * (1 - np.exp(-t / 100))

    # REINFORCE: Monte-Carlo returns are noisy around the true V.
    returns = true_V + rng.normal(0, 28, size=steps)

    # Advantage: same noise minus a learned baseline (~ true V).
    baseline = true_V + rng.normal(0, 4, size=steps)
    advantage = returns - baseline  # zero-mean, much smaller magnitude

    # Per-step gradient signal magnitude.
    grad_reinforce = returns
    grad_a2c = advantage

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Left: trajectories of gradient signal -------------------------------
    ax = axes[0]
    ax.plot(t, grad_reinforce, color=C_AMBER, lw=1.0, alpha=0.85,
            label=f"REINFORCE  $G_t$  (std = {grad_reinforce.std():.1f})")
    ax.plot(t, grad_a2c, color=C_BLUE, lw=1.0, alpha=0.95,
            label=f"A2C  $A_t = G_t - V_\\phi(s_t)$  "
                  f"(std = {grad_a2c.std():.1f})")
    ax.axhline(0, color=C_DARK, lw=0.7)

    ax.set_xlabel("update step", fontsize=11)
    ax.set_ylabel("gradient weight $\\hat{A}_t$ or $G_t$", fontsize=11)
    ax.set_title("Same trajectories, two gradient estimators",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    # --- Right: histogram of magnitudes --------------------------------------
    ax = axes[1]
    bins = np.linspace(-100, 200, 40)
    ax.hist(grad_reinforce, bins=bins, color=C_AMBER, alpha=0.65,
            label="REINFORCE", edgecolor="white")
    ax.hist(grad_a2c, bins=bins, color=C_BLUE, alpha=0.75,
            label="A2C (advantage)", edgecolor="white")
    ax.axvline(0, color=C_DARK, lw=0.8)

    # Variance ratio annotation.
    ratio = grad_reinforce.var() / grad_a2c.var()
    ax.text(0.97, 0.97,
            f"variance ratio\n"
            f"Var$(G_t)\\,/\\,$Var$(A_t) \\approx {ratio:.0f}\\times$",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.5", fc="white",
                      ec=C_GRAY, lw=1))

    ax.set_xlabel("per-step gradient weight", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    ax.set_title("Distribution of the gradient weight",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle("Why Actor-Critic trains more smoothly than REINFORCE",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_variance_reduction")


# ---------------------------------------------------------------------------
# Figure 3: Actor-Critic architecture diagram
# ---------------------------------------------------------------------------
def fig3_actor_critic_architecture() -> None:
    """Two-headed network: shared trunk -> actor (policy) + critic (value)."""
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    def box(x, y, w, h, text, color, fc=None, fontsize=11, fontweight="bold"):
        if fc is None:
            fc = color
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0.04,rounding_size=0.18",
                               linewidth=2, edgecolor=color, facecolor=fc)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=fontweight, color=C_DARK)

    def arrow(x1, y1, x2, y2, color=C_GRAY, lw=1.8, style="->"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle=style, color=color,
                                     mutation_scale=14, lw=lw))

    # Environment (left)
    box(0.2, 2.6, 1.8, 1.2, "Environment", C_GRAY, "#eef2f7", fontsize=11)

    # State input
    box(2.4, 2.6, 1.4, 1.2, "state $s_t$", C_DARK, "white", fontsize=11)

    # Shared backbone
    box(4.2, 2.4, 2.0, 1.6, "Shared\nencoder\n$f_\\psi(s)$",
        C_PURPLE, "#f3e8ff", fontsize=11)

    # Actor head
    box(7.0, 4.2, 2.4, 1.5,
        "Actor  $\\pi_\\theta(a|s)$\npolicy head",
        C_BLUE, "#dbeafe", fontsize=11)

    # Critic head
    box(7.0, 0.7, 2.4, 1.5,
        "Critic  $V_\\phi(s)$\nvalue head",
        C_GREEN, "#d1fae5", fontsize=11)

    # Action sample
    box(10.0, 4.4, 1.7, 1.1, "action $a_t$\n$\\sim \\pi_\\theta$",
        C_BLUE, "white", fontsize=10)

    # TD target / advantage
    box(10.0, 0.85, 1.7, 1.1,
        "TD error\n$\\delta_t$",
        C_AMBER, "#fef3c7", fontsize=10)

    # Arrows: env -> s
    arrow(2.0, 3.2, 2.4, 3.2)
    # s -> backbone
    arrow(3.8, 3.2, 4.2, 3.2)
    # backbone -> actor
    arrow(6.2, 3.5, 7.0, 4.6, color=C_BLUE)
    # backbone -> critic
    arrow(6.2, 2.9, 7.0, 1.6, color=C_GREEN)
    # actor -> action
    arrow(9.4, 4.95, 10.0, 4.95, color=C_BLUE)
    # critic -> td error
    arrow(9.4, 1.45, 10.0, 1.4, color=C_GREEN)

    # Action back to env (loop down and left).
    arrow(10.85, 4.4, 10.85, 0.3, color=C_BLUE, style="-")
    arrow(10.85, 0.3, 1.1, 0.3, color=C_BLUE, style="-")
    arrow(1.1, 0.3, 1.1, 2.6, color=C_BLUE, style="->")

    ax.text(6.0, 0.05, "interaction loop",
            ha="center", color=C_BLUE, fontsize=10, style="italic")

    # Update arrows: TD error trains BOTH heads.
    arrow(10.85, 1.4, 10.85, 2.4, color=C_AMBER, style="-")
    arrow(10.85, 2.4, 8.2, 2.4, color=C_AMBER, style="-")
    arrow(8.2, 2.4, 8.2, 4.2, color=C_AMBER)  # to actor
    ax.text(8.05, 2.55, "$\\delta_t$ as advantage\n(actor weight)",
            color=C_AMBER, fontsize=9, ha="right")
    arrow(8.2, 2.4, 8.2, 2.2, color=C_AMBER)  # to critic
    ax.text(7.4, 2.18, "$\\delta_t$ as TD target\n(critic loss)",
            color=C_AMBER, fontsize=9, ha="right")

    ax.set_title("Actor-Critic: one network, two jobs",
                 fontsize=13.5, fontweight="bold", pad=8)

    fig.tight_layout()
    _save(fig, "fig3_actor_critic_architecture")


# ---------------------------------------------------------------------------
# Figure 4: Advantage decomposition A = Q - V across actions
# ---------------------------------------------------------------------------
def fig4_advantage_decomposition() -> None:
    """For one state, plot Q(s, a_i), V(s) and A(s, a_i) = Q - V side-by-side."""
    actions = ["left", "stay", "right", "jump", "crouch"]
    Q = np.array([3.0, 4.2, 5.7, 6.8, 4.0])
    V = Q.mean()                # baseline = state value
    A = Q - V

    x = np.arange(len(actions))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # --- Left: Q-values with V baseline --------------------------------------
    ax = axes[0]
    bars = ax.bar(x, Q, color=C_BLUE, alpha=0.85, edgecolor="white", width=0.6,
                  label="$Q^\\pi(s, a_i)$")
    ax.axhline(V, color=C_PURPLE, lw=2, ls="--",
               label=f"$V^\\pi(s) = {V:.2f}$")
    for xi, q in zip(x, Q):
        ax.text(xi, q + 0.12, f"{q:.1f}", ha="center", color=C_DARK,
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(actions, fontsize=10)
    ax.set_ylabel("value", fontsize=11)
    ax.set_ylim(0, 8)
    ax.set_title("Action values vs the average",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # --- Right: signed advantage ---------------------------------------------
    ax = axes[1]
    colors = [C_GREEN if a > 0 else C_RED for a in A]
    bars = ax.bar(x, A, color=colors, alpha=0.85, edgecolor="white", width=0.6)
    ax.axhline(0, color=C_DARK, lw=0.8)
    for xi, a in zip(x, A):
        offset = 0.12 if a >= 0 else -0.18
        va = "bottom" if a >= 0 else "top"
        ax.text(xi, a + offset, f"{a:+.1f}",
                ha="center", va=va, color=C_DARK,
                fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(actions, fontsize=10)
    ax.set_ylabel("$A^\\pi(s, a) = Q^\\pi(s, a) - V^\\pi(s)$",
                  fontsize=11)
    ax.set_ylim(-2.3, 2.3)
    ax.set_title("Advantage: how much better than the average",
                 fontsize=12, fontweight="bold", pad=10)

    # Legend via proxies.
    from matplotlib.patches import Patch


    proxies = [Patch(color=C_GREEN, alpha=0.85, label="reinforce  ($A > 0$)"),
               Patch(color=C_RED, alpha=0.85, label="suppress   ($A < 0$)")]
    ax.legend(handles=proxies, loc="upper left", fontsize=10,
              framealpha=0.95)

    fig.suptitle("Advantage centres the gradient around zero",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig4_advantage_decomposition")


# ---------------------------------------------------------------------------
# Figure 5: GAE - bias-variance trade-off and weighting kernel
# ---------------------------------------------------------------------------
def fig5_gae_lambda_sweep() -> None:
    """Bias / variance vs lambda + the GAE weighting of n-step returns."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Left: bias-variance vs lambda --------------------------------------
    ax = axes[0]
    lam = np.linspace(0, 1, 200)
    # Stylised curves: variance grows from low (TD) to high (MC);
    # bias falls in the opposite direction.
    variance = 0.15 + 1.4 * lam ** 1.7
    bias = 0.95 * (1 - lam) ** 1.3 + 0.05
    total = variance + bias

    ax.plot(lam, variance, color=C_AMBER, lw=2.4, label="variance")
    ax.plot(lam, bias, color=C_PURPLE, lw=2.4, label="bias")
    ax.plot(lam, total, color=C_DARK, lw=2.6, ls="--",
            label="bias + variance")

    # Optimum (rough).
    opt = lam[np.argmin(total)]
    ax.axvline(opt, color=C_GREEN, lw=1.6, ls=":")
    ax.text(opt + 0.02, 0.05,
            f"$\\lambda^\\star \\approx {opt:.2f}$",
            color=C_GREEN, fontsize=11, fontweight="bold")

    # End-point labels.
    ax.text(0.02, 0.05, "TD(0)", color=C_BLUE, fontsize=10,
            fontweight="bold")
    ax.text(0.92, 0.05, "Monte\nCarlo", color=C_BLUE, fontsize=10,
            fontweight="bold", ha="right")

    ax.set_xlabel("GAE parameter  $\\lambda$", fontsize=11)
    ax.set_ylabel("error contribution", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.0)
    ax.set_title("GAE interpolates TD and Monte Carlo",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    # --- Right: GAE n-step weighting -----------------------------------------
    ax = axes[1]
    n = np.arange(0, 20)
    gamma = 0.99
    for lam_val, c in [(0.0, C_BLUE), (0.5, C_PURPLE),
                       (0.9, C_GREEN), (0.99, C_AMBER)]:
        w = (1 - lam_val) * (gamma * lam_val) ** n if lam_val < 1 else \
            np.where(n == 19, 1.0, 0.0)  # MC: all weight on full return
        ax.plot(n, w, marker="o", lw=2, color=c,
                label=f"$\\lambda = {lam_val}$")

    ax.set_xlabel("step  $k$  in $k$-step return", fontsize=11)
    ax.set_ylabel("weight on $\\delta_{t+k}$", fontsize=11)
    ax.set_title("How GAE blends multi-step TD errors",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_xlim(-0.5, 19.5)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95,
              title="$\\gamma = 0.99$")

    fig.suptitle("GAE($\\lambda$): one knob for the bias-variance dial",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig5_gae_lambda_sweep")


# ---------------------------------------------------------------------------
# Figure 6: Discrete vs continuous action policies
# ---------------------------------------------------------------------------
def fig6_action_policies() -> None:
    """Categorical (discrete) vs squashed-Gaussian (continuous) policy heads."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Left: discrete softmax ---------------------------------------------
    ax = axes[0]
    actions = ["up", "down", "left", "right", "noop"]
    logits = np.array([0.3, 1.7, -0.4, 2.4, 0.1])
    probs = np.exp(logits) / np.exp(logits).sum()

    bars = ax.bar(actions, probs, color=C_BLUE, alpha=0.85,
                  edgecolor="white", width=0.6)
    for xi, p in enumerate(probs):
        ax.text(xi, p + 0.012, f"{p:.2f}", ha="center",
                fontsize=10, fontweight="bold", color=C_DARK)

    ax.set_ylim(0, max(probs) + 0.12)
    ax.set_ylabel("$\\pi_\\theta(a|s)$", fontsize=11)
    ax.set_title("Discrete actions: softmax categorical",
                 fontsize=12, fontweight="bold", pad=10)
    ax.text(0.02, 0.97,
            "head: linear $\\to$ softmax\n"
            "loss: $-\\log \\pi(a|s) \\cdot \\hat{A}$",
            transform=ax.transAxes, va="top", fontsize=10,
            color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=C_GRAY, lw=1))

    # --- Right: continuous Gaussian -----------------------------------------
    ax = axes[1]
    a = np.linspace(-1.05, 1.05, 400)
    # Pre-tanh Gaussian: N(0.4, 0.55). After tanh, density is reweighted.
    mu, sigma = 0.4, 0.55
    pre = a.copy()
    # density via tanh change of variables: u = atanh(a)
    u = np.arctanh(np.clip(a, -0.999, 0.999))
    base = np.exp(-0.5 * ((u - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    jacobian = 1.0 / (1.0 - a ** 2 + 1e-6)
    density = base * jacobian

    ax.fill_between(a, density, color=C_PURPLE, alpha=0.35)
    ax.plot(a, density, color=C_PURPLE, lw=2.4)

    # Draw 5 sampled actions.
    rng = np.random.default_rng(11)
    samples_u = rng.normal(mu, sigma, size=200)
    samples_a = np.tanh(samples_u)
    ax.plot(samples_a, np.full_like(samples_a, -0.05), "|",
            color=C_PURPLE, alpha=0.5, ms=10, mew=1)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.15, density.max() * 1.2)
    ax.set_xlabel("bounded action $a \\in (-1, 1)$", fontsize=11)
    ax.set_ylabel("$\\pi_\\theta(a|s)$", fontsize=11)
    ax.set_title("Continuous actions: tanh-squashed Gaussian",
                 fontsize=12, fontweight="bold", pad=10)
    ax.text(0.02, 0.97,
            "head: $(\\mu_\\theta, \\log\\sigma_\\theta)$\n"
            "sample $u\\sim\\mathcal{N}$, then $a = \\tanh(u)$",
            transform=ax.transAxes, va="top", fontsize=10,
            color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=C_GRAY, lw=1))

    fig.suptitle("One framework, two policy heads",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "fig6_action_policies")


# ---------------------------------------------------------------------------
# Figure 7: Policy optimisation landscape - SGD in policy space
# ---------------------------------------------------------------------------
def fig7_policy_optimization_landscape() -> None:
    """Contour of J(theta) with a noisy SGD trajectory and a plateau."""
    # Build a 2D landscape with one dominant optimum and a misleading plateau.
    grid = np.linspace(-3, 3, 220)
    X, Y = np.meshgrid(grid, grid)
    J = (np.exp(-((X - 1.4) ** 2 + (Y - 0.9) ** 2) / 1.4) * 1.0
         + np.exp(-((X + 1.2) ** 2 + (Y + 1.1) ** 2) / 0.7) * 0.55
         - 0.05 * (X ** 2 + Y ** 2))

    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    cf = ax.contourf(X, Y, J, levels=22, cmap="Blues", alpha=0.85)
    ax.contour(X, Y, J, levels=10, colors="white", linewidths=0.5,
               alpha=0.6)

    # Simulate stochastic gradient ascent from a poor init.
    rng = np.random.default_rng(7)
    theta = np.array([-2.4, -2.0])
    traj = [theta.copy()]
    eps = 1e-3
    lr = 0.32
    for _ in range(45):
        # Numerical gradient on the analytic surface.
        def f(x, y):
            return (np.exp(-((x - 1.4) ** 2 + (y - 0.9) ** 2) / 1.4)
                    + np.exp(-((x + 1.2) ** 2 + (y + 1.1) ** 2) / 0.7) * 0.55
                    - 0.05 * (x ** 2 + y ** 2))
        gx = (f(theta[0] + eps, theta[1]) - f(theta[0] - eps, theta[1])) / (2 * eps)
        gy = (f(theta[0], theta[1] + eps) - f(theta[0], theta[1] - eps)) / (2 * eps)
        grad = np.array([gx, gy])
        # Inject MC noise (REINFORCE-like).
        noise = rng.normal(0, 0.55, size=2)
        theta = theta + lr * (grad + noise)
        theta = np.clip(theta, -2.95, 2.95)
        traj.append(theta.copy())
    traj = np.array(traj)

    ax.plot(traj[:, 0], traj[:, 1], "-", color=C_AMBER, lw=2.0, alpha=0.95)
    ax.scatter(traj[:, 0], traj[:, 1], color=C_AMBER, s=22,
               edgecolor="white", lw=0.7, zorder=3)

    # Markers
    ax.scatter([traj[0, 0]], [traj[0, 1]], color=C_RED, s=130, zorder=4,
               edgecolor="white", lw=1.3, label="start  $\\theta_0$")
    ax.scatter([1.4], [0.9], marker="*", color=C_GREEN, s=320,
               edgecolor="white", lw=1.3, zorder=4,
               label="global optimum  $\\theta^\\star$")
    ax.scatter([-1.2], [-1.1], marker="X", color=C_PURPLE, s=170,
               edgecolor="white", lw=1.2, zorder=4,
               label="local plateau")

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_xlabel("policy parameter  $\\theta_1$", fontsize=11)
    ax.set_ylabel("policy parameter  $\\theta_2$", fontsize=11)
    ax.set_title("Stochastic gradient ascent on $J(\\theta)$",
                 fontsize=13.5, fontweight="bold", pad=10)

    cbar = fig.colorbar(cf, ax=ax, shrink=0.82, pad=0.02)
    cbar.set_label("expected return  $J(\\theta)$", fontsize=10)

    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)

    ax.text(0.98, 0.02,
            "noisy gradients $\\to$ jagged path,\n"
            "but the policy still climbs.",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", fc="white",
                      ec=C_GRAY, lw=1))

    fig.tight_layout()
    _save(fig, "fig7_policy_optimization_landscape")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for RL Part 03...")
    fig1_policy_gradient_theorem()
    fig2_variance_reduction()
    fig3_actor_critic_architecture()
    fig4_advantage_decomposition()
    fig5_gae_lambda_sweep()
    fig6_action_policies()
    fig7_policy_optimization_landscape()
    print("Done.")


if __name__ == "__main__":
    main()
