"""
Figure generation script for Transfer Learning Part 12:
Industrial Applications and Best Practices (series finale).

Generates 7 figures used in both EN and ZH versions of the article.
Each figure isolates one teaching point and is built to be self-contained.

Figures:
    fig1_decision_flowchart     When to use transfer learning -- a decision
                                tree that walks from the four core questions
                                (data volume, label availability, latency
                                budget, domain shift) to the recommended
                                strategy: train from scratch, full fine-tune,
                                LoRA / adapter, or zero-shot prompting.
    fig2_production_pipeline    End-to-end production pipeline: foundation
                                model -> domain pretrain -> task fine-tune ->
                                compress -> serve -> monitor -> retrain.
                                Shows artefacts, owners, and feedback loop.
    fig3_compute_cost_savings   Compute and dollar cost: training a 7B-class
                                model from scratch vs LoRA fine-tune vs
                                prompt engineering, on a log scale, with
                                payback period annotated.
    fig4_industry_case_studies  Four landmark industrial deployments
                                (Google Translate GNMT, OpenAI ChatGPT,
                                Tesla Autopilot, GitHub Copilot) plotted
                                on adoption-vs-impact axes with model
                                lineage callouts.
    fig5_ab_testing             A/B testing two transfer learning models in
                                production: traffic split, daily conversion
                                rate with confidence bands, t-test result,
                                and the rollout decision.
    fig6_distribution_shift     Monitoring distribution shift: KL divergence
                                between training and production feature
                                distributions across 30 days, with the alert
                                threshold and the moment retraining is
                                triggered.
    fig7_roi_curve              ROI of transfer learning over 12 months:
                                cumulative cost (one-off pretrain + ongoing
                                serving) vs cumulative value, with the
                                breakeven month highlighted.

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb #7c3aed #10b981 #f59e0b.

Usage:
    python3 scripts/figures/transfer-learning/12-industrial.py

Output:
    Writes the same PNGs into BOTH the EN and ZH article asset folders so
    the markdown image references stay consistent across languages.
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
C_LIGHT = COLORS["grid"]
C_DARK = COLORS["text"]
C_BG = COLORS["bg"]

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "en" / "transfer-learning"
    / "12-industrial-applications-and-best-practices"
)
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "transfer-learning"
    / "12-工业应用与最佳实践"
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to BOTH EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, ec=None, fontsize=10, fontweight="normal",
         text_color="white", alpha=1.0, rounding=0.05):
    ec = ec or fc
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.2, facecolor=fc, edgecolor=ec, alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
            ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight=fontweight)


def _arrow(ax, xy_from, xy_to, color=C_DARK, lw=1.4, style="-|>",
           connection="arc3,rad=0"):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw, connectionstyle=connection,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: Decision flowchart -- when to use transfer learning
# ---------------------------------------------------------------------------
def fig1_decision_flowchart() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(7, 8.55, "When to Use Transfer Learning -- A Decision Tree",
            ha="center", fontsize=14, fontweight="bold", color=C_DARK)

    # Root
    _box(ax, (5.5, 7.3), 3, 0.7, "Start: new ML task", C_DARK,
         fontsize=10.5, fontweight="bold")

    # Q1: pretrained model exists?
    _box(ax, (5.0, 6.1), 4, 0.7, "Pretrained model in this modality?",
         C_BLUE, fontsize=10)
    _arrow(ax, (7, 7.3), (7, 6.8))

    # Yes branch -- right side
    # Q2: enough labels?
    _box(ax, (8.5, 4.9), 4.0, 0.7, "Have >= 100 labelled examples?",
         C_BLUE, fontsize=10)
    _arrow(ax, (8, 6.45), (10.5, 5.6),
           connection="arc3,rad=-0.15", color=C_GREEN, lw=1.6)
    ax.text(9.7, 5.95, "yes", color=C_GREEN, fontsize=10, fontweight="bold")

    # No branch -- left side
    _box(ax, (0.4, 4.9), 4.0, 0.7, "Train from scratch", C_RED,
         fontsize=10, fontweight="bold")
    _arrow(ax, (6, 6.45), (3.5, 5.6),
           connection="arc3,rad=0.15", color=C_RED, lw=1.6)
    ax.text(4.0, 5.95, "no", color=C_RED, fontsize=10, fontweight="bold")
    ax.text(2.4, 4.55,
            "needs >> 100k labels and weeks of GPU\n"
            "consider self-supervised pretraining first",
            ha="center", fontsize=8.5, color=C_GRAY, style="italic")

    # Q3: domain shift large?
    _box(ax, (8.5, 3.6), 4.0, 0.7, "Domain similar to pretraining corpus?",
         C_BLUE, fontsize=10)
    _arrow(ax, (10.5, 4.9), (10.5, 4.3), color=C_GREEN, lw=1.6)
    ax.text(10.85, 4.6, "yes", color=C_GREEN, fontsize=9.5, fontweight="bold")

    # No-labels branch (zero-shot)
    _box(ax, (8.5, 2.3), 4.0, 0.7, "Zero-shot / few-shot prompting",
         C_PURPLE, fontsize=10, fontweight="bold")
    _arrow(ax, (8.5, 5.25), (8.5, 3.0),
           connection="arc3,rad=0.4", color=C_AMBER, lw=1.6)
    ax.text(7.5, 3.7, "no labels", color=C_AMBER, fontsize=9.5,
            fontweight="bold")

    # Q4: latency budget
    _box(ax, (8.5, 2.3), 4.0, 0.7, "Latency budget < 50 ms ?",
         C_BLUE, fontsize=10, alpha=0.0, text_color=C_DARK)
    # actually replace with a different question -- compute budget
    _box(ax, (5.0, 2.3), 4.0, 0.7, "Multi-task or many tenants?",
         C_BLUE, fontsize=10)
    _arrow(ax, (10.5, 3.6), (7.0, 3.0),
           connection="arc3,rad=0.15", color=C_GREEN, lw=1.4)
    ax.text(8.5, 3.25, "yes -- domain match", color=C_GREEN,
            fontsize=9, fontweight="bold")

    # Domain mismatch branch
    _box(ax, (0.4, 3.6), 4.0, 0.7, "Domain-adaptive pretraining first",
         C_AMBER, fontsize=10, fontweight="bold")
    _arrow(ax, (8.5, 3.95), (4.4, 3.95), color=C_AMBER, lw=1.6)
    ax.text(6.5, 4.15, "no -- shift", color=C_AMBER, fontsize=9.5,
            fontweight="bold")

    # Final leaves
    _box(ax, (5.0, 1.0), 4.0, 0.7, "LoRA / adapter fine-tune",
         C_PURPLE, fontsize=10, fontweight="bold")
    _arrow(ax, (7.0, 2.3), (7.0, 1.7), color=C_PURPLE, lw=1.6)
    ax.text(7.4, 2.0, "yes", color=C_PURPLE, fontsize=9.5,
            fontweight="bold")

    _box(ax, (0.4, 1.0), 4.0, 0.7, "Full fine-tune", C_GREEN,
         fontsize=10, fontweight="bold")
    _arrow(ax, (5.0, 2.65), (2.4, 1.7),
           connection="arc3,rad=0.15", color=C_GREEN, lw=1.6)
    ax.text(2.9, 2.2, "no", color=C_GREEN, fontsize=9.5, fontweight="bold")

    # Footer legend
    legend = (
        "Heuristics: < 100 labels -> prompting | 100-10k -> LoRA | "
        ">10k & one task -> full fine-tune | mismatched domain -> "
        "domain pretraining first"
    )
    ax.text(7, 0.25, legend, ha="center", fontsize=9, color=C_DARK,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor=C_GRAY))

    _save(fig, "fig1_decision_flowchart")


# ---------------------------------------------------------------------------
# Figure 2: Production pipeline
# ---------------------------------------------------------------------------
def fig2_production_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(7, 7.05, "From Foundation Model to Production Service",
            ha="center", fontsize=14, fontweight="bold", color=C_DARK)
    ax.text(7, 6.65,
            "artefacts in boxes, owners in italics, feedback loop in amber",
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    stages = [
        ("Foundation\nmodel", "ML platform",
         "e.g. Llama-3, ViT-L\n300 GB checkpoint", C_BLUE),
        ("Domain\npretrain", "Research", "10-50 B tokens\nin-domain corpus",
         C_PURPLE),
        ("Task\nfine-tune", "Applied team",
         "10k-100k labels\nLoRA / full FT", C_BLUE),
        ("Compress &\nexport", "MLOps", "INT8 quant\nONNX / TensorRT",
         C_PURPLE),
        ("Serve", "Platform SRE",
         "Triton / vLLM\nbatch + cache", C_GREEN),
        ("Monitor", "On-call",
         "latency p99\ndrift, accuracy", C_AMBER),
    ]
    n = len(stages)
    box_w = 1.85
    gap = 0.25
    total = n * box_w + (n - 1) * gap
    x0 = (14 - total) / 2

    centers = []
    for i, (title, owner, detail, color) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        # main box
        _box(ax, (x, 3.2), box_w, 1.6, title, color,
             fontsize=11, fontweight="bold")
        centers.append(x + box_w / 2)
        # owner above
        ax.text(x + box_w / 2, 5.05, owner, ha="center",
                fontsize=8.5, color=C_DARK, style="italic")
        # artefact below
        ax.text(x + box_w / 2, 2.85, detail, ha="center",
                fontsize=8, color=C_GRAY, va="top")
        # arrow forward
        if i < n - 1:
            x_next = x0 + (i + 1) * (box_w + gap)
            _arrow(ax, (x + box_w + 0.04, 4.0),
                   (x_next - 0.04, 4.0),
                   color=C_DARK, lw=1.4)

    # feedback loop from monitor back to fine-tune
    monitor_x = centers[-1]
    finetune_x = centers[2]
    _arrow(ax, (monitor_x, 3.2),
           (finetune_x, 3.2),
           color=C_AMBER, lw=1.8,
           connection="arc3,rad=0.35")
    ax.text((monitor_x + finetune_x) / 2, 1.55,
            "drift detected  -->  retrain on fresh + buffered data",
            ha="center", fontsize=10, color=C_AMBER, fontweight="bold")

    # checkpoints emphasised
    ax.text(7, 0.7,
            "Reproducibility checkpoints saved at every stage "
            "(model + tokenizer + eval set + git SHA + data hash)",
            ha="center", fontsize=9.5, color=C_DARK,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_LIGHT,
                      edgecolor=C_GRAY))

    _save(fig, "fig2_production_pipeline")


# ---------------------------------------------------------------------------
# Figure 3: Compute / dollar savings
# ---------------------------------------------------------------------------
def fig3_compute_cost_savings() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # Left: compute (GPU-hours) -- log scale
    methods = ["Train\nfrom scratch", "Continued\npretraining",
               "Full\nfine-tune", "LoRA\nfine-tune", "Prompt\nengineering"]
    gpu_hours = [180_000, 12_000, 800, 40, 0.5]
    dollars = [h * 2.5 for h in gpu_hours]    # $2.5 / A100 hour, indicative
    colors = [C_RED, C_AMBER, C_BLUE, C_PURPLE, C_GREEN]

    ax = axes[0]
    bars = ax.bar(methods, gpu_hours, color=colors, edgecolor=C_DARK,
                  linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("GPU-hours (log scale, A100)", fontsize=11)
    ax.set_title("Compute cost: training a 7B-class model",
                 fontsize=12, fontweight="bold", color=C_DARK)
    for b, h in zip(bars, gpu_hours):
        ax.text(b.get_x() + b.get_width() / 2,
                h * 1.6,
                f"{h:,.1f}".rstrip("0").rstrip(".") + " h",
                ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    ax.set_ylim(0.2, 1e6)
    ax.tick_params(axis="x", labelsize=9)

    # Right: dollar cost + savings vs from-scratch
    ax = axes[1]
    baseline = dollars[0]
    savings = [(baseline - d) / baseline * 100 for d in dollars]
    bars = ax.barh(methods[::-1], dollars[::-1], color=colors[::-1],
                   edgecolor=C_DARK, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Cost (USD, log scale @ $2.5 / GPU-hour)", fontsize=11)
    ax.set_title("Dollar cost & savings vs from-scratch",
                 fontsize=12, fontweight="bold", color=C_DARK)
    for b, d, s in zip(bars, dollars[::-1], savings[::-1]):
        if d >= 1000:
            label = f"${d/1000:,.1f}k"
        else:
            label = f"${d:,.2f}"
        if s > 0.1:
            label += f"  (-{s:.2f}%)"
        ax.text(d * 1.4, b.get_y() + b.get_height() / 2, label,
                va="center", fontsize=9, color=C_DARK, fontweight="bold")
    ax.set_xlim(0.5, 1e7)

    fig.suptitle("Transfer learning collapses both compute and dollars by "
                 "3-5 orders of magnitude",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_compute_cost_savings")


# ---------------------------------------------------------------------------
# Figure 4: Industry case studies
# ---------------------------------------------------------------------------
def fig4_industry_case_studies() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.set_xlabel("User-facing impact (DAU x criticality, log scale)",
                  fontsize=11)
    ax.set_ylabel("Pretraining investment (B parameters x corpus, log scale)",
                  fontsize=11)
    ax.set_title("Landmark transfer-learning deployments",
                 fontsize=14, fontweight="bold", color=C_DARK, pad=15)

    cases = [
        # (label, x, y, bubble size, color, lineage)
        ("Google Translate\n(GNMT, 2016)", 7.8, 6.0, 1800, C_BLUE,
         "seq2seq LSTM\n-> Transformer\n-> multilingual\nshared encoder"),
        ("OpenAI ChatGPT\n(GPT-3.5/4, 2022-)", 9.0, 9.0, 3000, C_PURPLE,
         "GPT-3 base\n+ instruction tune\n+ RLHF"),
        ("Tesla Autopilot\n(HydraNet, 2020-)", 6.5, 7.5, 1500, C_GREEN,
         "shared CNN backbone\n+ task-specific\nheads (48 tasks)"),
        ("GitHub Copilot\n(Codex, 2021-)", 7.0, 8.2, 1700, C_AMBER,
         "GPT-3 -> code\ndomain pretraining\non public repos"),
    ]

    for label, x, y, s, color, lineage in cases:
        ax.scatter(x, y, s=s, color=color, alpha=0.55,
                   edgecolor=C_DARK, linewidth=1.2, zorder=3)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=C_DARK, zorder=4)

    # Lineage callouts
    callouts = [
        ("Google Translate", 1.3, 7.5,
         "GNMT shared encoder across 100+\n"
         "language pairs -- a textbook example\nof multi-task transfer."),
        ("ChatGPT", 1.3, 4.4,
         "Foundation model (GPT-3, 175B)\nfine-tuned with instructions\n"
         "and RLHF -- transfer at every layer."),
        ("Tesla Autopilot", 1.3, 1.8,
         "One backbone, 48 task heads\nshare features -- a production\n"
         "instance of multi-task transfer."),
        ("GitHub Copilot", 6.8, 1.8,
         "Continued pretraining of GPT\non code corpora -- domain\n"
         "adaptation at scale."),
    ]
    for title, x, y, body in callouts:
        ax.text(x, y, f"{title}\n{body}", ha="left", va="center",
                fontsize=9, color=C_DARK,
                bbox=dict(boxstyle="round,pad=0.45", facecolor=C_LIGHT,
                          edgecolor=C_GRAY, linewidth=0.8))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, alpha=0.25)

    _save(fig, "fig4_industry_case_studies")


# ---------------------------------------------------------------------------
# Figure 5: A/B testing transfer-learning models
# ---------------------------------------------------------------------------
def fig5_ab_testing() -> None:
    fig = plt.figure(figsize=(13, 6.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 2.0, 1.0], wspace=0.35)

    # --- Left: traffic split donut ---
    ax = fig.add_subplot(gs[0])
    ax.pie([50, 50],
           colors=[C_BLUE, C_PURPLE],
           labels=["Control\nBERT-base FT", "Treatment\nRoBERTa LoRA"],
           wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2),
           startangle=90, textprops=dict(fontsize=10, color=C_DARK,
                                         fontweight="bold"))
    ax.text(0, 0, "50 / 50\nsplit", ha="center", va="center",
            fontsize=12, fontweight="bold", color=C_DARK)
    ax.set_title("Traffic allocation", fontsize=12, fontweight="bold",
                 color=C_DARK)

    # --- Centre: daily conversion rate with CI ---
    ax = fig.add_subplot(gs[1])
    rng = np.random.default_rng(42)
    days = np.arange(1, 15)
    control = 0.082 + rng.normal(0, 0.0035, days.size)
    treat = 0.087 + rng.normal(0, 0.0040, days.size) + 0.0008 * days / 14
    ci = 0.005

    ax.plot(days, control, color=C_BLUE, lw=2.2, marker="o",
            label="Control: BERT-base full FT")
    ax.fill_between(days, control - ci, control + ci, color=C_BLUE,
                    alpha=0.18)
    ax.plot(days, treat, color=C_PURPLE, lw=2.2, marker="s",
            label="Treatment: RoBERTa-large LoRA")
    ax.fill_between(days, treat - ci, treat + ci, color=C_PURPLE,
                    alpha=0.18)

    ax.set_xlabel("Day of experiment", fontsize=11)
    ax.set_ylabel("Conversion rate", fontsize=11)
    ax.set_title("Daily conversion rate (mean +/- 95 % CI)",
                 fontsize=12, fontweight="bold", color=C_DARK)
    ax.legend(loc="lower right", fontsize=9.5)
    ax.set_xlim(0.5, 14.5)
    ax.set_ylim(0.07, 0.10)

    # annotate stat test on day 14
    ax.annotate("Welch t-test\nt = 3.42, p = 0.002\nlift = +6.5 %",
                xy=(14, treat[-1]), xytext=(10.2, 0.094),
                fontsize=9.5, color=C_GREEN, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C_GREEN),
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2))

    # --- Right: decision card ---
    ax = fig.add_subplot(gs[2])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _box(ax, (0.05, 0.7), 0.9, 0.22,
         "Decision\nFULL ROLLOUT", C_GREEN,
         fontsize=12, fontweight="bold")
    rows = [
        ("p-value", "0.002 (< 0.05)"),
        ("Effect size", "+6.5 % conversion"),
        ("Min sample", "reached on day 9"),
        ("Guardrail", "p99 latency  ok"),
        ("Cost delta", "-18 % infra"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.55 - i * 0.10
        ax.text(0.06, y, k, fontsize=9.5, color=C_GRAY)
        ax.text(0.94, y, v, fontsize=9.5, color=C_DARK,
                ha="right", fontweight="bold")
    ax.set_title("Verdict", fontsize=12, fontweight="bold", color=C_DARK)

    fig.suptitle("A/B testing two transfer-learning candidates in production",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    _save(fig, "fig5_ab_testing")


# ---------------------------------------------------------------------------
# Figure 6: Distribution shift monitoring
# ---------------------------------------------------------------------------
def fig6_distribution_shift() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.8),
                             gridspec_kw={"height_ratios": [2.0, 1.2]},
                             sharex=True)

    # Top: KL divergence over time
    rng = np.random.default_rng(7)
    days = np.arange(1, 31)
    kl = (0.04 + 0.005 * rng.standard_normal(days.size)
          + np.where(days > 18, 0.018 * (days - 18), 0)
          + np.where(days > 24, 0.025 * (days - 24), 0))
    threshold = 0.15

    ax = axes[0]
    ax.plot(days, kl, color=C_BLUE, lw=2.2, marker="o", markersize=5,
            label="KL(prod || train)")
    ax.axhline(threshold, color=C_RED, lw=1.5, linestyle="--",
               label=f"Alert threshold = {threshold}")
    ax.fill_between(days, 0, kl, where=kl >= threshold, color=C_RED,
                    alpha=0.18, label="Above threshold")

    # Annotate retrain trigger
    trigger_day = int(days[np.argmax(kl >= threshold)]) if any(kl >= threshold) else None
    if trigger_day is not None:
        ax.axvline(trigger_day, color=C_AMBER, lw=1.4, linestyle=":")
        ax.annotate(f"Day {trigger_day}: retrain triggered",
                    xy=(trigger_day, threshold),
                    xytext=(trigger_day - 9, 0.22),
                    fontsize=10, color=C_AMBER, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.2))

    ax.set_ylabel("KL divergence", fontsize=11)
    ax.set_title("Distribution shift monitor: production vs training "
                 "feature distribution",
                 fontsize=13, fontweight="bold", color=C_DARK)
    ax.legend(loc="upper left", fontsize=9.5)
    ax.set_ylim(0, 0.3)

    # Bottom: accuracy on holdout dropping in sync
    acc = (0.913 - 0.001 * rng.standard_normal(days.size)
           - np.where(days > 18, 0.0035 * (days - 18), 0)
           - np.where(days > 24, 0.005 * (days - 24), 0))
    ax2 = axes[1]
    ax2.plot(days, acc * 100, color=C_PURPLE, lw=2.2, marker="s",
             markersize=4, label="Holdout accuracy")
    ax2.axhline(85, color=C_RED, lw=1.4, linestyle="--",
                label="SLO = 85 %")
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_xlabel("Day", fontsize=11)
    ax2.legend(loc="lower left", fontsize=9.5)
    ax2.set_ylim(82, 93)

    fig.tight_layout()
    _save(fig, "fig6_distribution_shift")


# ---------------------------------------------------------------------------
# Figure 7: ROI of transfer learning
# ---------------------------------------------------------------------------
def fig7_roi_curve() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))

    months = np.arange(0, 13)
    # cumulative cost: large one-off setup + small ongoing
    setup = 80_000     # USD -- engineering + initial pretrain rental
    ongoing = 6_000    # USD / month serving + monitoring
    cost = setup + ongoing * months

    # cumulative value: ramps up then steady (replaced rules / vendor)
    monthly_value = np.array([0, 5_000, 12_000, 22_000, 32_000, 40_000,
                              45_000, 48_000, 50_000, 50_000, 50_000,
                              50_000, 50_000])
    value = np.cumsum(monthly_value)

    ax.fill_between(months, 0, cost, color=C_RED, alpha=0.18,
                    label="Cumulative cost")
    ax.fill_between(months, 0, value, color=C_GREEN, alpha=0.22,
                    label="Cumulative value")
    ax.plot(months, cost, color=C_RED, lw=2.4, marker="o")
    ax.plot(months, value, color=C_GREEN, lw=2.4, marker="s")

    # breakeven
    diff = value - cost
    breakeven_idx = int(np.argmax(diff > 0)) if any(diff > 0) else None
    if breakeven_idx:
        bx = months[breakeven_idx]
        by = value[breakeven_idx]
        ax.axvline(bx, color=C_AMBER, lw=1.6, linestyle="--")
        ax.annotate(f"Breakeven: month {bx}",
                    xy=(bx, by), xytext=(bx + 0.4, by + 30_000),
                    fontsize=11, color=C_AMBER, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.4))

    # final annotations
    final_roi = (value[-1] - cost[-1]) / cost[-1] * 100
    ax.text(12.2, value[-1], f"Year-1 ROI\n+{final_roi:.0f} %",
            color=C_GREEN, fontsize=11, fontweight="bold", va="center")
    ax.text(12.2, cost[-1], f"Total cost\n${cost[-1]/1000:.0f}k",
            color=C_RED, fontsize=10.5, fontweight="bold", va="center")

    ax.set_xlabel("Months since launch", fontsize=11)
    ax.set_ylabel("Cumulative USD", fontsize=11)
    ax.set_title("ROI of a transfer-learning project (12-month horizon)",
                 fontsize=14, fontweight="bold", color=C_DARK)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, max(value.max(), cost.max()) * 1.18)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    _save(fig, "fig7_roi_curve")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 12 figures (Industrial Applications)...")
    fig1_decision_flowchart()
    fig2_production_pipeline()
    fig3_compute_cost_savings()
    fig4_industry_case_studies()
    fig5_ab_testing()
    fig6_distribution_shift()
    fig7_roi_curve()
    print("Done.")


if __name__ == "__main__":
    main()
