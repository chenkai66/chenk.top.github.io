"""
Figure generation for the standalone post "Prompt Engineering Complete Guide".

Generates 7 figures shared between the EN and ZH versions. Each figure teaches
one specific idea cleanly, with no decorative noise.

Figures:
    fig1_prompt_anatomy        Anatomy of a structured prompt (role / context /
                               instruction / examples / format) shown as a
                               labelled stack with token-budget annotations.
    fig2_paradigm_compare      Zero-shot vs. Few-shot vs. Chain-of-Thought:
                               accuracy and token cost on a multi-step
                               arithmetic benchmark (GSM8K-class).
    fig3_cot_flow              Chain-of-thought reasoning flow on a multi-step
                               problem, contrasting a "direct answer" path
                               (often wrong) with the explicit reasoning path
                               (correct).
    fig4_prompt_sensitivity    Heatmap of accuracy across prompt formats x
                               example orderings, revealing the variance that
                               makes empirical evaluation necessary.
    fig5_self_consistency      Self-consistency: many sampled reasoning paths
                               converge via majority vote on the correct
                               answer; wrong paths shown faded.
    fig6_tree_of_thoughts      Tree of thoughts: state-space search with
                               branching, scoring, and backtracking on the
                               Game of 24.
    fig7_template_library      Reusable prompt template library (extraction,
                               classification, RAG, code-gen, summarization,
                               creative).

Usage:
    python3 scripts/figures/standalone/prompt-engineering.py

Output:
    Writes PNGs into BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
C_RED = "#ef4444"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = (REPO_ROOT / "source" / "_posts" / "en" / "standalone"
          / "prompt-engineering-complete-guide")
ZH_DIR = (REPO_ROOT / "source" / "_posts" / "zh" / "standalone"
          / "提示词工程完全指南-从零基础到高级优化")


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# fig1: Prompt anatomy
# ---------------------------------------------------------------------------
def fig1_prompt_anatomy() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 7.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    blocks = [
        ("Role / Persona", 8.6, 1.3, C_BLUE,
         "You are a senior support engineer.\n"
         "Be concise; cite ticket IDs."),
        ("Context", 7.0, 1.5, C_PURPLE,
         "Customer purchased plan P3 on 2025-11-02.\n"
         "Open ticket history attached below."),
        ("Instruction", 5.4, 1.3, C_AMBER,
         "Classify the next message as:\n"
         "  bug | billing | feature-request | other"),
        ("Few-Shot Examples", 3.4, 2.0, C_GREEN,
         "Msg: 'card was charged twice'  ->  billing\n"
         "Msg: 'app crashes on iPad'      ->  bug\n"
         "Msg: 'add dark mode pls'        ->  feature-request"),
        ("Output Format", 1.4, 1.4, C_GRAY,
         "Return JSON only:\n"
         '{"label": "...", "confidence": 0.0-1.0,\n'
         ' "evidence": "<=20 words"}'),
    ]

    x0, w = 0.3, 6.2
    for title, y, h, color, body in blocks:
        ax.add_patch(FancyBboxPatch(
            (x0, y - h / 2), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.6, edgecolor=color, facecolor=color + "18"))
        ax.text(x0 + 0.18, y + h / 2 - 0.30, title,
                fontsize=11, fontweight="bold", color=color, va="top")
        ax.text(x0 + 0.18, y + h / 2 - 0.72, body,
                fontsize=9.0, color=C_DARK, va="top", family="monospace")

    # Right column: role of each block
    ax.text(7.0, 9.5, "What each block does",
            fontsize=12, fontweight="bold", color=C_DARK)
    notes = [
        ("Role", "Picks tone, expertise, refusal rules.",
         "~25 tok", C_BLUE),
        ("Context", "Grounds answer in your data.",
         "varies", C_PURPLE),
        ("Instruction", "States the goal in one sentence.",
         "~20 tok", C_AMBER),
        ("Examples", "Primary in-context-learning signal.",
         "~120 tok", C_GREEN),
        ("Format", "Pins schema for downstream parsers.",
         "~25 tok", C_GRAY),
    ]
    y = 8.85
    for name, desc, cost, color in notes:
        ax.add_patch(plt.Circle((7.05, y + 0.05), 0.10,
                                color=color, transform=ax.transData))
        ax.text(7.28, y + 0.10, name, fontsize=10, fontweight="bold",
                color=C_DARK, va="center")
        ax.text(7.28, y - 0.30, desc, fontsize=8.8, color=C_DARK,
                va="center")
        ax.text(9.95, y + 0.10, cost, fontsize=9, color=color,
                va="center", ha="right", fontweight="bold")
        y -= 0.95

    # Vertical assembly arrow
    arr = FancyArrowPatch((6.6, 8.6), (6.6, 1.4),
                          arrowstyle="-|>", mutation_scale=14,
                          color=C_DARK, linewidth=1.2,
                          linestyle=(0, (3, 3)))
    ax.add_patch(arr)
    ax.text(6.74, 5.0, "concatenated\ntop -> bottom",
            fontsize=8.5, color=C_DARK, rotation=90, va="center")

    ax.set_title("Anatomy of a production prompt: five composable blocks",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=12)
    _save(fig, "fig1_prompt_anatomy")


# ---------------------------------------------------------------------------
# fig2: Zero-shot vs Few-shot vs CoT comparison
# ---------------------------------------------------------------------------
def fig2_paradigm_compare() -> None:
    fig = plt.figure(figsize=(12, 6.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.7], hspace=0.42,
                          wspace=0.28)

    paradigms = [
        ("Zero-Shot", C_BLUE,
         "Q: Roger has 5 tennis balls.\n"
         "He buys 2 cans, each with 3.\n"
         "How many balls now?\n"
         "A:",
         "Direct answer.\nNo examples.\nNo reasoning shown."),
        ("Few-Shot", C_PURPLE,
         "Q: 5 + 3*2 = ?\nA: 11\n"
         "Q: 7 + 4*3 = ?\nA: 19\n"
         "Q: Roger has 5 ... ?\nA:",
         "k examples teach\nthe input -> output\npattern."),
        ("Chain-of-Thought", C_GREEN,
         "Q: Roger has 5 ... ?\n"
         "Let's think step by step.\n"
         "2 cans * 3 = 6.\n"
         "5 + 6 = 11.\nA: 11",
         "Model emits its\nreasoning before\nthe final answer."),
    ]

    for i, (name, color, prompt, note) in enumerate(paradigms):
        ax = fig.add_subplot(gs[0, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.96,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            linewidth=1.8, edgecolor=color, facecolor=color + "12"))
        ax.text(0.5, 0.93, name, fontsize=13, fontweight="bold",
                color=color, ha="center")
        ax.text(0.06, 0.80, prompt, fontsize=9.2, color=C_DARK, va="top",
                family="monospace")
        ax.text(0.5, 0.13, note, fontsize=9.2, color=C_DARK, ha="center",
                style="italic")

    # Bar chart: accuracy + token cost on GSM8K-class arithmetic
    ax = fig.add_subplot(gs[1, :])
    labels = ["Zero-Shot", "Few-Shot (k=8)", "Chain-of-Thought"]
    acc = [17.7, 33.5, 78.7]   # representative GSM8K-class accuracy
    tok = [40, 320, 240]
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, acc, w, color=[C_BLUE, C_PURPLE, C_GREEN],
                label="Accuracy (%)", edgecolor="white", linewidth=1.2)
    ax2 = ax.twinx()
    b2 = ax2.bar(x + w / 2, tok, w, color=C_AMBER, alpha=0.85,
                 label="Tokens / call", edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10, color=C_DARK)
    ax2.set_ylabel("Tokens / call", fontsize=10, color=C_AMBER)
    ax.set_ylim(0, 95)
    ax2.set_ylim(0, 400)
    ax.grid(axis="y", alpha=0.3)
    ax2.grid(False)
    for rect, v in zip(b1, acc):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1.6, f"{v:.1f}",
                ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    for rect, v in zip(b2, tok):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + 8, f"{v}",
                 ha="center", fontsize=9, color=C_AMBER, fontweight="bold")
    ax.set_title("Representative accuracy / cost on GSM8K-class arithmetic "
                 "(Wei et al. 2022; Kojima et al. 2022)",
                 fontsize=10.5, color=C_DARK, pad=4)

    fig.suptitle("Three prompting paradigms: same model, different framing",
                 fontsize=13.5, fontweight="bold", color=C_DARK, y=1.00)
    _save(fig, "fig2_paradigm_compare")


# ---------------------------------------------------------------------------
# fig3: CoT reasoning flow
# ---------------------------------------------------------------------------
def fig3_cot_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Problem box
    problem = ("Problem: A book has 120 pages. On day 1, 30 pages were read.\n"
               "On day 2, twice as many as day 1 were read.\n"
               "On day 3, half of the remaining pages were read.\n"
               "How many pages were read on day 3?")
    ax.add_patch(FancyBboxPatch(
        (0.4, 5.1), 11.2, 1.5,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.6, edgecolor=C_DARK, facecolor=C_LIGHT))
    ax.text(0.6, 5.85, "INPUT", fontsize=9, fontweight="bold",
            color=C_DARK)
    ax.text(0.6, 5.5, problem, fontsize=9.5, color=C_DARK, va="center",
            family="monospace")

    # Direct path (top, wrong)
    ax.add_patch(FancyBboxPatch(
        (0.4, 3.4), 5.4, 1.2,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.6, edgecolor=C_RED, facecolor=C_RED + "12"))
    ax.text(0.6, 4.4, "Direct answer", fontsize=10.5, fontweight="bold",
            color=C_RED)
    ax.text(0.6, 4.05, "Prompt: \"... How many pages on day 3? Answer:\"",
            fontsize=8.8, color=C_DARK, family="monospace")
    ax.text(0.6, 3.65, "Model output:  \"60\"   (wrong, off by 4x)",
            fontsize=9, color=C_RED, family="monospace")

    # CoT path (top right, correct)
    ax.add_patch(FancyBboxPatch(
        (6.2, 3.4), 5.4, 1.2,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.6, edgecolor=C_GREEN, facecolor=C_GREEN + "12"))
    ax.text(6.4, 4.4, "Chain-of-thought", fontsize=10.5, fontweight="bold",
            color=C_GREEN)
    ax.text(6.4, 4.05, "Prompt: \"... Let's think step by step.\"",
            fontsize=8.8, color=C_DARK, family="monospace")
    ax.text(6.4, 3.65, "Model emits intermediate steps -> 15  (correct)",
            fontsize=9, color=C_GREEN, family="monospace")

    # Step-by-step boxes for the CoT chain
    steps = [
        "Step 1\nDay 1: 30",
        "Step 2\n2x30 = 60",
        "Step 3\nRead so far: 90",
        "Step 4\nLeft: 120-90 = 30",
        "Step 5\nDay 3: 30/2 = 15",
    ]
    sx = np.linspace(0.6, 9.6, 5)
    for i, s in enumerate(steps):
        ax.add_patch(FancyBboxPatch(
            (sx[i], 1.4), 1.7, 1.2,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.4, edgecolor=C_GREEN, facecolor="white"))
        ax.text(sx[i] + 0.85, 2.0, s, fontsize=8.6, ha="center",
                va="center", color=C_DARK)
        if i < 4:
            arr = FancyArrowPatch(
                (sx[i] + 1.7, 2.0), (sx[i + 1], 2.0),
                arrowstyle="-|>", mutation_scale=12,
                color=C_GREEN, linewidth=1.2)
            ax.add_patch(arr)

    # Final answer chip
    ax.add_patch(FancyBboxPatch(
        (10.4, 1.4), 1.4, 1.2,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.8, edgecolor=C_DARK, facecolor=C_AMBER + "30"))
    ax.text(11.1, 2.0, "Answer\n15", fontsize=11, fontweight="bold",
            ha="center", va="center", color=C_DARK)
    arr = FancyArrowPatch(
        (sx[4] + 1.7, 2.0), (10.4, 2.0),
        arrowstyle="-|>", mutation_scale=14,
        color=C_DARK, linewidth=1.4)
    ax.add_patch(arr)

    ax.text(6.0, 0.55,
            "CoT pushes intermediate state into the output, "
            "so each step conditions on the prior reasoning.",
            fontsize=9.5, color=C_DARK, ha="center", style="italic")

    ax.set_title("Chain-of-thought reasoning: explicit steps cut error rates",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    _save(fig, "fig3_cot_flow")


# ---------------------------------------------------------------------------
# fig4: Prompt sensitivity to format / order
# ---------------------------------------------------------------------------
def fig4_prompt_sensitivity() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8),
                                   gridspec_kw={"width_ratios": [1.4, 1.0]})

    rng = np.random.default_rng(7)

    formats = [
        "Q:/A:",
        "Input/Output",
        "Numbered (1) (2)",
        "JSON keys",
        "Markdown bullets",
        "Tagged <ex>",
    ]
    orders = [f"order #{i}" for i in range(1, 7)]

    base = rng.uniform(58, 84, size=(len(formats), len(orders)))
    base[3] += 4   # JSON keys slightly better
    base[2] -= 6   # Numbered worse
    base = np.clip(base, 45, 92)

    im = ax1.imshow(base, cmap="RdYlGn", vmin=45, vmax=92, aspect="auto")
    ax1.set_xticks(range(len(orders)))
    ax1.set_xticklabels(orders, rotation=20, fontsize=9.5)
    ax1.set_yticks(range(len(formats)))
    ax1.set_yticklabels(formats, fontsize=10)
    for i in range(len(formats)):
        for j in range(len(orders)):
            ax1.text(j, i, f"{base[i, j]:.0f}", ha="center", va="center",
                     fontsize=9, color=C_DARK)
    ax1.set_title("Same model, same examples — accuracy by format x order",
                  fontsize=11.5, fontweight="bold", color=C_DARK)
    cbar = fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02)
    cbar.set_label("Accuracy (%)", fontsize=9)

    spreads = base.tolist()
    bp = ax2.boxplot(spreads, vert=True, widths=0.55, patch_artist=True,
                     medianprops=dict(color=C_DARK, linewidth=1.6))
    palette = [C_BLUE, C_PURPLE, C_GRAY, C_GREEN, C_AMBER, C_DARK]
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c + "55")
        patch.set_edgecolor(c)
        patch.set_linewidth(1.4)
    ax2.set_xticks(range(1, len(formats) + 1))
    ax2.set_xticklabels(formats, rotation=25, fontsize=9, ha="right")
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.set_title("Spread caused by example ordering alone",
                  fontsize=11.5, fontweight="bold", color=C_DARK)
    ax2.set_ylim(40, 95)

    fig.suptitle("Prompt sensitivity: format and order can swing accuracy "
                 "by 20+ points",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig4_prompt_sensitivity")


# ---------------------------------------------------------------------------
# fig5: Self-consistency
# ---------------------------------------------------------------------------
def fig5_self_consistency() -> None:
    fig, ax = plt.subplots(figsize=(11.6, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Root: question
    ax.add_patch(FancyBboxPatch(
        (4.6, 5.7), 2.8, 1.0,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.8, edgecolor=C_DARK, facecolor=C_LIGHT))
    ax.text(6.0, 6.3, "Question + 'think step by step'",
            fontsize=11, fontweight="bold", color=C_DARK, ha="center")
    ax.text(6.0, 5.95, "sample k=5 paths at temperature 0.7",
            fontsize=8.6, color=C_DARK, ha="center", style="italic")

    # Five reasoning paths
    paths = [
        (1.2, "Path A", ["94", "117", "117"], "117", True),
        (3.4, "Path B", ["94", "118", "118"], "118", False),
        (5.6, "Path C", ["94", "117", "117"], "117", True),
        (7.8, "Path D", ["47", "70", "70"], "70", False),
        (10.0, "Path E", ["94", "117", "117"], "117", True),
    ]
    for x, name, steps, answer, correct in paths:
        color = C_GREEN if correct else C_GRAY
        alpha = 1.0 if correct else 0.55
        ax.add_patch(FancyArrowPatch(
            (6.0, 5.7), (x + 0.6, 5.0),
            arrowstyle="-|>", mutation_scale=10,
            color=color, linewidth=1.2, alpha=alpha))
        ax.text(x + 0.6, 4.8, name, fontsize=9.2, fontweight="bold",
                color=color, ha="center", alpha=alpha)
        ys = [4.3, 3.7, 3.1]
        for s, y in zip(steps, ys):
            ax.add_patch(FancyBboxPatch(
                (x, y - 0.22), 1.2, 0.44,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                linewidth=1.2, edgecolor=color,
                facecolor=color + ("25" if correct else "12"),
                alpha=alpha))
            ax.text(x + 0.6, y, s, fontsize=9, color=C_DARK,
                    ha="center", va="center", alpha=alpha)
        ax.add_patch(FancyBboxPatch(
            (x + 0.05, 2.25), 1.1, 0.5,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6, edgecolor=color,
            facecolor=color + ("40" if correct else "20"),
            alpha=alpha))
        ax.text(x + 0.6, 2.5, f"-> {answer}", fontsize=10,
                fontweight="bold", color=C_DARK, ha="center",
                va="center", alpha=alpha)
        ax.add_patch(FancyArrowPatch(
            (x + 0.6, 2.25), (6.0, 1.55),
            arrowstyle="-|>", mutation_scale=10,
            color=color, linewidth=1.0, alpha=alpha * 0.8))

    # Vote / majority box
    ax.add_patch(FancyBboxPatch(
        (4.4, 0.55), 3.2, 1.0,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=2.0, edgecolor=C_GREEN, facecolor=C_GREEN + "25"))
    ax.text(6.0, 1.25, "Majority vote", fontsize=11, fontweight="bold",
            color=C_GREEN, ha="center")
    ax.text(6.0, 0.85, "117  (3/5 paths agree, conf 0.6)",
            fontsize=10, color=C_DARK, ha="center", family="monospace")

    ax.set_title("Self-consistency: sample many reasoning paths, "
                 "vote on the answer",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    _save(fig, "fig5_self_consistency")


# ---------------------------------------------------------------------------
# fig6: Tree of Thoughts
# ---------------------------------------------------------------------------
def fig6_tree_of_thoughts() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def node(x, y, w, h, label, color, alpha=1.0, bold=False, score=None):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.03,rounding_size=0.10",
            linewidth=1.6 if not bold else 2.0,
            edgecolor=color, facecolor=color + ("30" if bold else "15"),
            alpha=alpha))
        ax.text(x, y + 0.04, label, fontsize=9, color=C_DARK,
                ha="center", va="center", alpha=alpha,
                fontweight="bold" if bold else "normal",
                family="monospace")
        if score is not None:
            ax.text(x + w / 2 - 0.05, y + h / 2 - 0.18, f"v={score}",
                    fontsize=7.4, color=color, ha="right", va="top",
                    fontweight="bold", alpha=alpha)

    def arrow(p1, p2, color, alpha=1.0, dashed=False):
        ax.add_patch(FancyArrowPatch(
            p1, p2, arrowstyle="-|>", mutation_scale=11,
            color=color, linewidth=1.3, alpha=alpha,
            linestyle=(0, (3, 3)) if dashed else "-"))

    # Root
    node(6.0, 6.3, 2.2, 0.7, "{4, 9, 10, 13}", C_DARK, bold=True)
    ax.text(6.0, 5.85, "target = 24", fontsize=8.6, color=C_DARK,
            ha="center", style="italic")

    # Layer 1: three candidate first ops
    layer1 = [
        (1.8, "13 - 9 = 4\n{4, 4, 10}", C_GREEN, 8, True),
        (6.0, "10 - 4 = 6\n{6, 9, 13}", C_AMBER, 5, False),
        (10.2, "9 + 10 = 19\n{4, 13, 19}", C_GRAY, 2, False),
    ]
    for x, lbl, color, sc, kept in layer1:
        node(x, 4.6, 2.4, 0.95, lbl, color,
             alpha=1.0 if kept else 0.55, score=sc)
        arrow((6.0, 5.95), (x, 5.1), color,
              alpha=1.0 if kept else 0.55)

    # Layer 2 under the best branch
    layer2 = [
        (0.6, "10 - 4 = 6\n{4, 6}", C_GREEN, 9, True),
        (2.4, "4 + 4 = 8\n{8, 10}", C_GRAY, 3, False),
        (4.2, "10 * 4 = 40\n{4, 40}", C_GRAY, 1, False),
    ]
    for x, lbl, color, sc, kept in layer2:
        node(x, 3.0, 1.7, 0.9, lbl, color,
             alpha=1.0 if kept else 0.55, score=sc)
        arrow((1.8, 4.1), (x, 3.5), color,
              alpha=1.0 if kept else 0.45,
              dashed=not kept)

    # Layer 3: solve from {4, 6}
    node(0.6, 1.4, 1.7, 0.9, "6 * 4 = 24\nDONE", C_GREEN, bold=True, score=10)
    arrow((0.6, 2.55), (0.6, 1.85), C_GREEN)

    # Solution chip
    ax.add_patch(FancyBboxPatch(
        (8.6, 1.0), 3.0, 1.6,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=2.0, edgecolor=C_GREEN, facecolor=C_GREEN + "25"))
    ax.text(10.1, 2.25, "Solution found", fontsize=11, fontweight="bold",
            color=C_GREEN, ha="center")
    ax.text(10.1, 1.75, "(13 - 9) -> 4\nthen (10 - 4) = 6\nthen 6 * 4 = 24",
            fontsize=9, color=C_DARK, ha="center", family="monospace")

    arrow((1.45, 1.4), (8.6, 1.6), C_GREEN)

    # Legend
    ax.text(6.0, 0.35,
            "v = self-evaluated value of the partial state.   "
            "Faded = pruned branch.   Bold = chosen path.",
            fontsize=9, color=C_DARK, ha="center", style="italic")

    ax.set_title("Tree of Thoughts: branch, score, and backtrack "
                 "(Game of 24, Yao et al. 2023)",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    _save(fig, "fig6_tree_of_thoughts")


# ---------------------------------------------------------------------------
# fig7: Prompt template library
# ---------------------------------------------------------------------------
def fig7_template_library() -> None:
    fig, ax = plt.subplots(figsize=(12.4, 7.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    cards = [
        # (col, row, title, color, body)
        (0, 1, "Extraction", C_BLUE,
         "Schema: {name, date, amount}\n"
         "Rules: dates as ISO 8601;\n"
         "skip fields not in source.\n"
         "Return JSON only."),
        (1, 1, "Classification", C_PURPLE,
         "Labels: {bug, billing,\n"
         "  feature, other}\n"
         "If unsure -> 'other' with\n"
         "confidence < 0.5."),
        (2, 1, "Retrieval-Augmented QA", C_GREEN,
         "Use ONLY the passages below.\n"
         "Cite as [#1], [#2].\n"
         "If insufficient: say\n"
         "'no evidence in context'."),
        (0, 0, "Code Generation", C_AMBER,
         "Language: Python 3.11.\n"
         "Constraints: no I/O,\n"
         "pure function, type hints,\n"
         "include 3 doctests."),
        (1, 0, "Summarization", C_GRAY,
         "Output: 5 bullets, <=15\n"
         "words each.\n"
         "Preserve numbers verbatim.\n"
         "No opinions."),
        (2, 0, "Creative Rewrite", C_DARK,
         "Voice: warm, second-person.\n"
         "Reading level: grade 8.\n"
         "Keep all factual claims;\n"
         "vary sentence length."),
    ]

    cw, ch = 3.7, 2.9
    x0, y0 = 0.4, 1.0
    gap_x, gap_y = 0.20, 0.50

    for col, row, title, color, body in cards:
        x = x0 + col * (cw + gap_x)
        y = y0 + row * (ch + gap_y)
        ax.add_patch(FancyBboxPatch(
            (x, y), cw, ch,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            linewidth=1.8, edgecolor=color, facecolor=color + "12"))
        ax.text(x + 0.18, y + ch - 0.30, title,
                fontsize=12, fontweight="bold", color=color, va="top")
        ax.text(x + 0.18, y + ch - 0.95, body,
                fontsize=9.0, color=C_DARK, va="top", family="monospace")
        # Tag chip with reusable structure
        ax.add_patch(FancyBboxPatch(
            (x + cw - 1.65, y + 0.18), 1.45, 0.42,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0, edgecolor=color, facecolor=color + "25"))
        ax.text(x + cw - 0.93, y + 0.39,
                "role + task + format",
                fontsize=7.6, color=color, ha="center",
                va="center", fontweight="bold")

    ax.text(6.0, 7.45,
            "Prompt template library: same five blocks, swapped per task",
            fontsize=13.5, fontweight="bold", color=C_DARK, ha="center")
    ax.text(6.0, 7.05,
            "Reusing the same skeleton makes evaluation, caching, "
            "and version control dramatically easier.",
            fontsize=10, color=C_DARK, ha="center", style="italic")

    _save(fig, "fig7_template_library")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_prompt_anatomy()
    fig2_paradigm_compare()
    fig3_cot_flow()
    fig4_prompt_sensitivity()
    fig5_self_consistency()
    fig6_tree_of_thoughts()
    fig7_template_library()
    print(f"Wrote 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
