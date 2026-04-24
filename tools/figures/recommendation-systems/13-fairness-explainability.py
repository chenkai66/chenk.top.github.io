"""Figures for Recommendation Systems Part 13 — Fairness, Debiasing, Explainability.

Generates 7 publication-quality figures covering popularity bias, position
bias, selection bias, the accuracy-fairness Pareto frontier, IPS reweighting,
LIME local explanations, and counterfactual explanations.

Output is written to BOTH the EN and ZH asset folders so the script is the
single source of truth. Run from any working directory.
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
RED = COLORS["danger"]

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/recommendation-systems/13-fairness-explainability"
ZH_DIR = REPO / "source/_posts/zh/recommendation-systems/13-公平性-去偏与可解释性"


def save(fig: plt.Figure, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Popularity bias: long-tail distribution
# ---------------------------------------------------------------------------
def fig1_popularity_bias() -> None:
    rng = np.random.default_rng(7)
    n_items = 200
    ranks = np.arange(1, n_items + 1)
    # Zipf-like distribution
    interactions = 5000.0 / (ranks ** 0.95)
    interactions += rng.normal(0, interactions * 0.05)
    interactions = np.clip(interactions, 1, None)

    # Recommendations are even more concentrated than ground-truth interactions
    rec_share = (1.0 / (ranks ** 1.4))
    rec_share = rec_share / rec_share.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Left panel: long-tail with head/torso/tail segments ---
    ax = axes[0]
    head = ranks <= 20
    torso = (ranks > 20) & (ranks <= 80)
    tail = ranks > 80
    ax.bar(ranks[head], interactions[head], color=BLUE, width=1.0, label="Head (top 10%)")
    ax.bar(ranks[torso], interactions[torso], color=PURPLE, width=1.0, label="Torso (10-40%)")
    ax.bar(ranks[tail], interactions[tail], color=ORANGE, width=1.0, label="Long tail (60%)")
    ax.set_xlabel("Item rank (sorted by popularity)")
    ax.set_ylabel("Number of interactions")
    ax.set_title("Long-tail distribution of item interactions")
    ax.legend(loc="upper right")
    # Annotate concentration
    head_share = interactions[head].sum() / interactions.sum()
    ax.text(
        0.97,
        0.62,
        f"Top 10% items\ncapture {head_share:.0%}\nof interactions",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=BLUE, alpha=0.9),
    )

    # --- Right panel: recommendation share vs catalog share ---
    ax = axes[1]
    cum_rec = np.cumsum(rec_share)
    cum_catalog = np.cumsum(np.ones(n_items) / n_items)
    ax.plot(ranks, cum_rec, color=BLUE, lw=2.4, label="Recommendation share")
    ax.plot(ranks, cum_catalog, color=GRAY, lw=2.0, ls="--", label="Uniform (no bias)")
    ax.fill_between(ranks, cum_catalog, cum_rec, where=cum_rec > cum_catalog,
                    alpha=0.18, color=ORANGE, label="Bias gap")
    ax.set_xlabel("Item rank (sorted by popularity)")
    ax.set_ylabel("Cumulative exposure share")
    ax.set_title("Recommendations concentrate on the head")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.text(
        0.05,
        0.92,
        f"Top 20 items get {cum_rec[19]:.0%}\nof all recommendations",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=BLUE,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=BLUE, alpha=0.9),
    )

    fig.suptitle("Popularity bias: head items dominate exposure", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig1_popularity_bias.png")


# ---------------------------------------------------------------------------
# Figure 2 — Position bias: CTR vs ranking position
# ---------------------------------------------------------------------------
def fig2_position_bias() -> None:
    positions = np.arange(1, 21)
    # Empirical position bias decay (matches Joachims et al. 2017 IPS paper)
    observed_ctr = 0.32 / (positions ** 0.85)
    # If items were equally relevant across positions, CTR would be flat
    true_relevance = np.full_like(positions, 0.10, dtype=float)
    # Examination probability: probability the user looks at position k
    examination = 1.0 / (positions ** 0.65)
    examination = examination / examination.max()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    bars = ax.bar(positions, observed_ctr, color=BLUE, alpha=0.85, label="Observed CTR")
    ax.plot(positions, true_relevance, color=GREEN, lw=2.4, marker="o",
            ms=5, label="Underlying relevance (constant)")
    ax.set_xlabel("Ranking position")
    ax.set_ylabel("Click-through rate")
    ax.set_title("CTR drops sharply with position\neven when items are equally relevant")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.legend(loc="upper right")
    ax.text(
        0.55,
        0.55,
        f"Position 1: {observed_ctr[0]:.0%}\nPosition 10: {observed_ctr[9]:.0%}\n3.5x gap from rank alone",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRAY, alpha=0.9),
    )

    ax = axes[1]
    ax.plot(positions, examination, color=PURPLE, lw=2.4, marker="s", ms=5,
            label="Examination probability P(see pos k)")
    ax.fill_between(positions, 0, examination, color=PURPLE, alpha=0.15)
    ax.set_xlabel("Ranking position")
    ax.set_ylabel("Probability of being examined")
    ax.set_title("Position bias model: P(click) = P(examine) x P(relevant)")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.text(
        0.05,
        0.30,
        "IPS divides each click\nby P(examine) to recover\nthe true relevance signal",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color=PURPLE,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=PURPLE, alpha=0.9),
    )

    fig.suptitle("Position bias: top of the list collects clicks regardless of relevance",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_position_bias.png")


# ---------------------------------------------------------------------------
# Figure 3 — Selection bias: observed vs true preference distribution
# ---------------------------------------------------------------------------
def fig3_selection_bias() -> None:
    rng = np.random.default_rng(13)
    n_items = 600

    # True average rating — roughly normal centered at 3.4
    true_ratings = np.clip(rng.normal(3.4, 0.9, n_items), 1, 5)
    # Observation probability is correlated with rating (people rate items they like)
    obs_prob = 0.05 + 0.85 * (true_ratings - 1) / 4
    observed_mask = rng.random(n_items) < obs_prob
    observed_ratings = true_ratings[observed_mask]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    # --- Left: distributions ---
    ax = axes[0]
    bins = np.linspace(1, 5, 25)
    ax.hist(true_ratings, bins=bins, color=GRAY, alpha=0.55,
            label=f"True (all items, mean={true_ratings.mean():.2f})", density=True)
    ax.hist(observed_ratings, bins=bins, color=BLUE, alpha=0.75,
            label=f"Observed (rated items, mean={observed_ratings.mean():.2f})", density=True)
    ax.axvline(true_ratings.mean(), color=GRAY, ls="--", lw=1.5)
    ax.axvline(observed_ratings.mean(), color=BLUE, ls="--", lw=1.5)
    ax.set_xlabel("Rating")
    ax.set_ylabel("Density")
    ax.set_title("Observed ratings are systematically inflated")
    ax.legend(loc="upper left")

    # --- Right: missing-not-at-random (MNAR) pattern ---
    ax = axes[1]
    rating_bins = np.arange(1, 6)
    obs_rate_per_bin = []
    for r in rating_bins:
        mask = (true_ratings >= r - 0.5) & (true_ratings < r + 0.5)
        if mask.sum() > 0:
            obs_rate_per_bin.append(observed_mask[mask].mean())
        else:
            obs_rate_per_bin.append(0)
    bars = ax.bar(rating_bins, obs_rate_per_bin, color=ORANGE, alpha=0.85)
    for r, p in zip(rating_bins, obs_rate_per_bin):
        ax.text(r, p + 0.02, f"{p:.0%}", ha="center", fontsize=10, color="black")
    ax.set_xlabel("True rating")
    ax.set_ylabel("Probability of being rated")
    ax.set_title("Missing-not-at-random:\nliked items are far more likely to be rated")
    ax.set_xticks(rating_bins)
    ax.set_ylim(0, 1.0)

    fig.suptitle("Selection bias: the observed sample is not the population",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_selection_bias.png")


# ---------------------------------------------------------------------------
# Figure 4 — Pareto frontier: accuracy vs fairness
# ---------------------------------------------------------------------------
def fig4_pareto_frontier() -> None:
    rng = np.random.default_rng(21)

    # Pareto frontier: accuracy decreases as fairness improves
    fairness = np.linspace(0.20, 0.95, 60)
    # Logistic-shaped trade-off: small fairness gains are cheap, large are expensive
    accuracy = 0.92 - 0.08 * (fairness - 0.20) / 0.75 - 0.18 * np.maximum(0, fairness - 0.75) ** 2 / 0.04
    accuracy = np.clip(accuracy, 0.50, 1.0)

    # Off-Pareto (suboptimal) configurations
    off_fair = rng.uniform(0.20, 0.95, 80)
    off_acc = []
    for f in off_fair:
        # Best achievable accuracy at this fairness level
        idx = np.argmin(np.abs(fairness - f))
        best = accuracy[idx]
        off_acc.append(best - rng.uniform(0.04, 0.18))
    off_acc = np.array(off_acc)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    ax.scatter(off_fair, off_acc, color=GRAY, alpha=0.45, s=40,
               label="Suboptimal configurations")
    ax.plot(fairness, accuracy, color=BLUE, lw=2.8, label="Pareto frontier")
    ax.fill_between(fairness, accuracy, accuracy.min() - 0.05,
                    alpha=0.10, color=BLUE)

    # Highlight three operating points
    points = {
        "Accuracy-first\n(no fairness)": (0.25, 0.915, ORANGE),
        "Balanced\n(recommended)": (0.65, 0.855, GREEN),
        "Fairness-first\n(strict parity)": (0.92, 0.69, PURPLE),
    }
    for label, (f, a, color) in points.items():
        ax.scatter([f], [a], color=color, s=200, zorder=5, edgecolor="white", lw=2)
        offset_y = 18 if "Balanced" in label else -22
        ax.annotate(label, (f, a), xytext=(10, offset_y),
                    textcoords="offset points", fontsize=10, color=color, fontweight="bold")

    ax.set_xlabel("Fairness (1 - demographic disparity)")
    ax.set_ylabel("Accuracy (NDCG@10)")
    ax.set_title("Accuracy vs fairness: a Pareto frontier")
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(0.55, 0.95)
    ax.legend(loc="lower left")
    ax.text(
        0.97,
        0.95,
        "Points on the frontier are optimal:\nyou cannot improve fairness without\nlosing accuracy (and vice versa)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=BLUE, alpha=0.9),
    )

    fig.tight_layout()
    save(fig, "fig4_pareto_frontier.png")


# ---------------------------------------------------------------------------
# Figure 5 — IPS reweighting concept
# ---------------------------------------------------------------------------
def fig5_ips_reweighting() -> None:
    positions = np.arange(1, 11)
    # Examination propensity (P(observed) at each position)
    propensity = 1.0 / (positions ** 0.7)
    propensity = propensity / propensity.max()
    # Naive click counts — heavily biased toward top
    naive_clicks = (np.array([100, 65, 45, 30, 22, 17, 14, 11, 9, 7])).astype(float)
    # IPS-corrected counts: divide by propensity
    ips_clicks = naive_clicks / propensity

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    # --- Left: the conceptual diagram ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Boxes
    def box(ax, x, y, w, h, text, color, text_color="white"):
        rect = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            linewidth=2, edgecolor=color, facecolor=color, alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=11, color=text_color, fontweight="bold")

    box(ax, 0.5, 7.5, 3.0, 1.6, "Observed click\non item i\nat position k", BLUE)
    box(ax, 6.5, 7.5, 3.0, 1.6, "Propensity\nP(examine | k)", ORANGE)
    box(ax, 3.5, 3.5, 3.0, 1.6, "IPS weight\n= 1 / P", PURPLE)
    box(ax, 3.5, 0.4, 3.0, 1.6, "Unbiased\nrelevance signal", GREEN)

    # Arrows
    arrow_kw = dict(arrowstyle="-|>", lw=1.8, color=GRAY, mutation_scale=18)
    ax.add_patch(FancyArrowPatch((2.0, 7.5), (4.0, 5.1), **arrow_kw))
    ax.add_patch(FancyArrowPatch((8.0, 7.5), (6.0, 5.1), **arrow_kw))
    ax.add_patch(FancyArrowPatch((5.0, 3.5), (5.0, 2.0), **arrow_kw))

    ax.text(5.0, 6.2, "x", ha="center", fontsize=20, color=GRAY, fontweight="bold")
    ax.set_title("IPS pipeline: reweight clicks by inverse propensity")

    # --- Right: numerical effect ---
    ax = axes[1]
    width = 0.38
    naive_norm = naive_clicks / naive_clicks.sum()
    ips_norm = ips_clicks / ips_clicks.sum()
    ax.bar(positions - width / 2, naive_norm, width, color=BLUE, alpha=0.85,
           label="Naive count (biased)")
    ax.bar(positions + width / 2, ips_norm, width, color=GREEN, alpha=0.85,
           label="IPS-corrected (unbiased)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Estimated relevance share")
    ax.set_title("After IPS, lower-ranked items recover their true weight")
    ax.set_xticks(positions)
    ax.legend(loc="upper right")

    fig.suptitle("Inverse Propensity Scoring: divide each click by P(observed)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig5_ips_reweighting.png")


# ---------------------------------------------------------------------------
# Figure 6 — LIME-style local explanation
# ---------------------------------------------------------------------------
def fig6_lime_explanation() -> None:
    features = [
        "watched: Inception",
        "genre match: sci-fi",
        "director: Nolan",
        "release year: 2010-2020",
        "rating: 8+ on IMDb",
        "runtime > 120 min",
        "watched: The Notebook",
        "genre match: romance",
        "language: French",
    ]
    # Positive contributions push prediction up; negative push down
    contributions = np.array([0.31, 0.24, 0.19, 0.11, 0.08, 0.05, -0.07, -0.13, -0.05])
    # Sort by absolute contribution
    order = np.argsort(-np.abs(contributions))
    features = [features[i] for i in order]
    contributions = contributions[order]
    colors = [GREEN if c > 0 else RED for c in contributions]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                             gridspec_kw={"width_ratios": [1.6, 1.0]})

    # --- Left: feature contribution bar chart ---
    ax = axes[0]
    y_pos = np.arange(len(features))
    ax.barh(y_pos, contributions, color=colors, alpha=0.85, edgecolor="white", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Contribution to predicted rating")
    ax.set_title("LIME local explanation\nWhy the model recommended 'Interstellar' to user 42")
    for i, c in enumerate(contributions):
        ax.text(c + (0.01 if c > 0 else -0.01), i,
                f"{c:+.2f}", va="center",
                ha="left" if c > 0 else "right", fontsize=9)
    ax.set_xlim(-0.30, 0.45)

    # --- Right: how LIME builds the explanation ---
    ax = axes[1]
    rng = np.random.default_rng(3)
    # Original instance
    ax.scatter([0.5], [0.5], color=BLUE, s=320, zorder=5,
               edgecolor="white", lw=2, label="Instance to explain")
    # Perturbed neighbors
    pert_x = rng.normal(0.5, 0.18, 80)
    pert_y = rng.normal(0.5, 0.18, 80)
    # Black-box prediction (simulated)
    pred = 1 / (1 + np.exp(-3 * (pert_x + pert_y - 1.0)))
    sc = ax.scatter(pert_x, pert_y, c=pred, cmap="RdYlGn",
                    s=60, alpha=0.75, edgecolor="white", lw=0.5,
                    label="Perturbed samples")
    # Local linear approximation (decision boundary)
    xs = np.linspace(0, 1, 50)
    ys = 1.0 - xs
    ax.plot(xs, ys, color=PURPLE, lw=2.5, ls="--", label="Local linear model")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("LIME = local linear approximation\nof a black-box model")
    ax.legend(loc="upper right", fontsize=9)
    plt.colorbar(sc, ax=ax, label="Black-box prediction", shrink=0.8)

    fig.tight_layout()
    save(fig, "fig6_lime_explanation.png")


# ---------------------------------------------------------------------------
# Figure 7 — Counterfactual explanation
# ---------------------------------------------------------------------------
def fig7_counterfactual() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Left: factual vs counterfactual user profile ---
    ax = axes[0]
    features = ["Sci-fi\nwatched", "Drama\nwatched", "Action\nwatched",
                "Watch\nfrequency", "Avg\nrating"]
    factual = [12, 3, 2, 0.7, 4.2]
    counterfactual = [12, 3, 8, 0.7, 4.2]  # change: more action movies
    x = np.arange(len(features))
    width = 0.36
    ax.bar(x - width / 2, factual, width, color=BLUE, alpha=0.85, label="Actual profile")
    ax.bar(x + width / 2, counterfactual, width, color=ORANGE, alpha=0.85,
           label="Counterfactual ('what if')")
    ax.set_xticks(x)
    ax.set_xticklabels(features)
    ax.set_ylabel("Value")
    ax.set_title("Counterfactual: minimal changes that flip the recommendation")
    ax.legend(loc="upper right")
    # Highlight changed feature
    ax.annotate("Only this\nchanged", xy=(2 + width / 2, 8), xytext=(2.6, 11),
                fontsize=10, color=ORANGE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8))

    # --- Right: prediction change & decision boundary ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # Decision regions
    xx, yy = np.meshgrid(np.linspace(0, 10, 60), np.linspace(0, 10, 60))
    z = (xx + 0.7 * yy) > 8.5
    ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5],
                colors=[LIGHT, "#fde68a"], alpha=0.6)

    # Boundary
    xs = np.linspace(0, 10, 50)
    ys = (8.5 - xs) / 0.7
    ax.plot(xs, ys, color=GRAY, lw=2, ls="--", label="Recommendation boundary")

    # Factual point (NOT recommended Inception 2)
    ax.scatter([3], [4], color=BLUE, s=320, zorder=5, edgecolor="white", lw=2)
    ax.annotate("Actual user\n(not recommended)", (3, 4), xytext=(3.4, 4.4),
                fontsize=10, color=BLUE, fontweight="bold")

    # Counterfactual point (recommended)
    ax.scatter([7], [4], color=ORANGE, s=320, zorder=5, edgecolor="white", lw=2)
    ax.annotate("Counterfactual\n(recommended)", (7, 4), xytext=(7.2, 4.4),
                fontsize=10, color=ORANGE, fontweight="bold")

    # Arrow showing the minimal change
    ax.add_patch(FancyArrowPatch((3.3, 4), (6.7, 4), arrowstyle="-|>",
                                 lw=2.2, color=PURPLE, mutation_scale=18))
    ax.text(5, 3.4, "Smallest change\nthat flips the decision",
            ha="center", fontsize=10, color=PURPLE, fontweight="bold")

    ax.text(8.5, 9, "Recommend", ha="center", fontsize=11,
            color=ORANGE, fontweight="bold")
    ax.text(2, 9, "Don't recommend", ha="center", fontsize=11,
            color=GRAY, fontweight="bold")

    ax.set_xlabel("Action movies watched")
    ax.set_ylabel("Avg rating")
    ax.set_title("Counterfactual: the closest 'recommend' point\nto the actual user")
    ax.legend(loc="lower right")

    fig.suptitle("Counterfactual explanation: 'If you watched 6 more action films, we would have recommended this'",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_counterfactual.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_popularity_bias()
    fig2_position_bias()
    fig3_selection_bias()
    fig4_pareto_frontier()
    fig5_ips_reweighting()
    fig6_lime_explanation()
    fig7_counterfactual()
    print(f"Saved 7 figures to:\n  EN: {EN_DIR}\n  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
