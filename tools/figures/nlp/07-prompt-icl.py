"""
Figure generation script for NLP Part 07: Prompt Engineering & In-Context Learning.

Generates 7 figures used in both EN and ZH versions of the article. Each figure
teaches one specific idea cleanly, no decorative noise.

Figures:
    fig1_prompt_anatomy        Anatomy of a structured prompt (system / few-shot
                               examples / user query / format spec) shown as a
                               labelled stack with byte-budget annotations.
    fig2_prompting_paradigms   Side-by-side comparison: Zero-shot vs. Few-shot
                               vs. Chain-of-Thought, with token cost and
                               accuracy bars on a representative reasoning task.
    fig3_cot_flow              Chain-of-thought reasoning flow on a multi-step
                               arithmetic problem, contrasting a "direct answer"
                               path (often wrong) with the explicit reasoning
                               path (correct).
    fig4_shots_saturation      Accuracy vs. number of in-context examples on
                               three task families, showing the saturation
                               curve and the cost of each extra shot.
    fig5_prompt_sensitivity    Heatmap of accuracy across prompt formats x
                               example orderings, revealing the variance that
                               makes prompt engineering necessary.
    fig6_react_pattern         ReAct loop: Thought -> Action -> Observation,
                               drawn as a feedback cycle around an LLM core,
                               with an external tool environment.
    fig7_self_consistency      Self-consistency / tree-of-thought: many sampled
                               reasoning paths converging via majority vote on
                               the correct answer; wrong paths shown faded.

Usage:
    python3 scripts/figures/nlp/07-prompt-icl.py

Output:
    Writes PNGs into BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Shared style ----------------------------------------------------------------
import sys
from pathlib import Path as _StylePath
sys.path.insert(0, str(_StylePath(__file__).parent.parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
# Color palette (from shared _style)
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["gray"]
C_DARK = COLORS["ink"]
C_LIGHT = COLORS["light"]
C_RED = COLORS["danger"]


DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "nlp" / "prompt-engineering-icl"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "nlp" / "07-提示工程与In-Context-Learning"


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
    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    blocks = [
        ("System / Role", 8.0, 1.6, C_BLUE,
         "You are a senior data analyst. Be concise and cite numbers."),
        ("Task Instruction", 6.6, 1.2, C_PURPLE,
         "Classify each review as positive / negative / neutral."),
        ("Few-Shot Examples", 4.4, 2.0, C_GREEN,
         "Review: 'Battery dies in 3h.'  ->  negative\n"
         "Review: 'Sharp screen, fair price.'  ->  positive\n"
         "Review: 'It exists.'  ->  neutral"),
        ("User Query", 2.6, 1.6, C_AMBER,
         "Review: 'Camera is fine but the app is buggy.'\nLabel:"),
        ("Output Format Spec", 1.0, 1.4, C_GRAY,
         "JSON: {\"label\": ..., \"confidence\": 0.0-1.0}"),
    ]

    x0, w = 0.4, 6.0
    for title, y, h, color, body in blocks:
        ax.add_patch(FancyBboxPatch(
            (x0, y - h / 2), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.6, edgecolor=color, facecolor=color + "18"))
        ax.text(x0 + 0.18, y + h / 2 - 0.32, title,
                fontsize=11, fontweight="bold", color=color, va="top")
        ax.text(x0 + 0.18, y + h / 2 - 0.78, body,
                fontsize=9.2, color=C_DARK, va="top", family="monospace")

    # Right column: token / role annotations
    ax.text(7.0, 9.4, "What each block does",
            fontsize=12, fontweight="bold", color=C_DARK)
    notes = [
        ("System / Role", "Sets persona, tone, refusal rules.",
         "~30 tok", C_BLUE),
        ("Task", "States the goal in one sentence.",
         "~20 tok", C_PURPLE),
        ("Examples", "Demonstrates pattern; primary ICL signal.",
         "~120 tok", C_GREEN),
        ("Query", "The actual input to be processed.",
         "varies", C_AMBER),
        ("Format", "Pins output schema for parsers.",
         "~25 tok", C_GRAY),
    ]
    y = 8.7
    for name, desc, cost, color in notes:
        ax.add_patch(plt.Circle((7.05, y + 0.05), 0.10,
                                color=color, transform=ax.transData))
        ax.text(7.28, y + 0.08, name, fontsize=10, fontweight="bold",
                color=C_DARK, va="center")
        ax.text(7.28, y - 0.30, desc, fontsize=9, color=C_DARK, va="center")
        ax.text(9.85, y + 0.08, cost, fontsize=9, color=color,
                va="center", ha="right", fontweight="bold")
        y -= 0.85

    # Arrow showing assembly order
    arr = FancyArrowPatch((6.5, 8.0), (6.5, 1.0),
                          arrowstyle="-|>", mutation_scale=14,
                          color=C_DARK, linewidth=1.2,
                          linestyle=(0, (3, 3)))
    ax.add_patch(arr)
    ax.text(6.62, 4.5, "concatenated\ntop -> bottom",
            fontsize=8.5, color=C_DARK, rotation=90, va="center")

    ax.set_title("Anatomy of a structured prompt: five composable blocks",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=12)
    _save(fig, "fig1_prompt_anatomy")


# ---------------------------------------------------------------------------
# fig2: Zero-shot vs Few-shot vs CoT
# ---------------------------------------------------------------------------
def fig2_prompting_paradigms() -> None:
    fig = plt.figure(figsize=(12, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.6], hspace=0.35,
                          wspace=0.28)

    paradigms = [
        ("Zero-Shot", C_BLUE,
         "Q: 23 + 47 * 2 = ?\nA:",
         "Direct answer.\nNo examples.\nNo reasoning."),
        ("Few-Shot", C_PURPLE,
         "Q: 5 + 3 * 2 = ?\nA: 11\nQ: 7 + 4 * 3 = ?\nA: 19\nQ: 23 + 47 * 2 = ?\nA:",
         "k examples teach\nthe input -> output\npattern."),
        ("Chain-of-Thought", C_GREEN,
         "Q: 23 + 47 * 2 = ?\nLet's think step by step.\nFirst 47 * 2 = 94.\nThen 23 + 94 = 117.\nA: 117",
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
        ax.text(0.5, 0.92, name, fontsize=13, fontweight="bold",
                color=color, ha="center")
        ax.text(0.06, 0.78, prompt, fontsize=9.5, color=C_DARK, va="top",
                family="monospace")
        ax.text(0.5, 0.16, note, fontsize=9.2, color=C_DARK, ha="center",
                style="italic")

    # Bar chart: representative accuracy + token cost on GSM8K-style task
    ax = fig.add_subplot(gs[1, :])
    labels = ["Zero-Shot", "Few-Shot (k=8)", "Chain-of-Thought"]
    acc = [17.7, 33.5, 56.9]   # representative numbers, GSM8K-class
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
    ax.set_ylim(0, 75)
    ax2.set_ylim(0, 400)
    ax.grid(axis="y", alpha=0.3)
    ax2.grid(False)
    for rect, v in zip(b1, acc):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v:.1f}",
                ha="center", fontsize=9, fontweight="bold", color=C_DARK)
    for rect, v in zip(b2, tok):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + 8, f"{v}",
                 ha="center", fontsize=9, color=C_AMBER, fontweight="bold")
    ax.set_title("Representative accuracy / cost on multi-step arithmetic",
                 fontsize=10.5, color=C_DARK, pad=4)

    fig.suptitle("Three prompting paradigms: same model, different framing",
                 fontsize=13.5, fontweight="bold", color=C_DARK, y=1.00)
    _save(fig, "fig2_prompting_paradigms")


# ---------------------------------------------------------------------------
# fig3: CoT reasoning flow
# ---------------------------------------------------------------------------
def fig3_cot_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
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

    # CoT path (bottom, right)
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

    # Answer
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

    # Caption
    ax.text(6.0, 0.55,
            "CoT pushes intermediate state into the output, "
            "so each step conditions on prior reasoning.",
            fontsize=9.5, color=C_DARK, ha="center", style="italic")

    ax.set_title("Chain-of-thought reasoning: explicit steps cut error rates",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    _save(fig, "fig3_cot_flow")


# ---------------------------------------------------------------------------
# fig4: Saturation of accuracy vs # of examples
# ---------------------------------------------------------------------------
def fig4_shots_saturation() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))

    k = np.array([0, 1, 2, 4, 8, 16, 32])

    # Three task families. Numbers are illustrative but qualitatively faithful
    # to the published ICL literature (Brown et al. 2020 + follow-ups).
    cls_acc = np.array([55, 67, 73, 79, 83, 84.5, 85])     # sentiment-like
    qa_acc = np.array([41, 50, 55, 60, 64, 66, 66.5])      # extractive QA
    math_acc = np.array([18, 22, 27, 33, 38, 41, 42])      # arithmetic w/o CoT

    ax.plot(k, cls_acc, "o-", color=C_BLUE, linewidth=2.2,
            markersize=8, label="Classification (sentiment)")
    ax.plot(k, qa_acc, "s-", color=C_PURPLE, linewidth=2.2,
            markersize=8, label="Extractive QA")
    ax.plot(k, math_acc, "^-", color=C_GREEN, linewidth=2.2,
            markersize=8, label="Arithmetic (no CoT)")

    # Saturation annotation
    ax.axvline(8, color=C_AMBER, linestyle="--", linewidth=1.4, alpha=0.8)
    ax.text(8.3, 30, "diminishing returns\nbeyond k ~ 8",
            fontsize=10, color=C_AMBER, fontweight="bold")

    # Cost overlay (right axis)
    ax2 = ax.twinx()
    ax2.bar(k, k * 40, width=1.4, color=C_GRAY, alpha=0.18, zorder=0)
    ax2.set_ylabel("Tokens spent on examples (~40 tok / shot)",
                   color=C_GRAY, fontsize=10)
    ax2.set_ylim(0, 1500)
    ax2.grid(False)

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks(k)
    ax.set_xticklabels([str(v) for v in k])
    ax.set_xlabel("k = number of in-context examples", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(10, 95)
    ax.set_title("Accuracy saturates fast with k; cost grows linearly",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    ax.legend(loc="lower right", fontsize=10, frameon=True)
    ax.set_zorder(2)
    ax.patch.set_visible(False)

    _save(fig, "fig4_shots_saturation")


# ---------------------------------------------------------------------------
# fig5: Prompt sensitivity to format / order
# ---------------------------------------------------------------------------
def fig5_prompt_sensitivity() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6),
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
    # Make some formats systematically better
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

    # Right: per-format spread (boxplot-style)
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

    fig.suptitle("Prompt sensitivity: format and order can swing accuracy by 20+ points",
                 fontsize=13, fontweight="bold", color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig5_prompt_sensitivity")


# ---------------------------------------------------------------------------
# fig6: ReAct loop
# ---------------------------------------------------------------------------
def fig6_react_pattern() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Center: LLM core
    ax.add_patch(FancyBboxPatch(
        (5.0, 2.6), 2.0, 1.8,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        linewidth=2.0, edgecolor=C_BLUE, facecolor=C_BLUE + "18"))
    ax.text(6.0, 3.7, "LLM", fontsize=14, fontweight="bold",
            color=C_BLUE, ha="center")
    ax.text(6.0, 3.2, "decoder", fontsize=9, color=C_BLUE,
            ha="center")

    # Three nodes around it: Thought (top), Action (right), Observation (left)
    nodes = [
        (6.0, 5.7, "Thought",
         "Reason about the\ncurrent state.", C_PURPLE),
        (10.0, 3.5, "Action",
         "Call a tool:\nsearch / calc / API.", C_GREEN),
        (2.0, 3.5, "Observation",
         "Receive tool result,\nfeed back into context.", C_AMBER),
    ]
    for x, y, name, desc, color in nodes:
        ax.add_patch(FancyBboxPatch(
            (x - 1.3, y - 0.7), 2.6, 1.4,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            linewidth=1.8, edgecolor=color, facecolor=color + "15"))
        ax.text(x, y + 0.32, name, fontsize=11.5, fontweight="bold",
                color=color, ha="center")
        ax.text(x, y - 0.25, desc, fontsize=9, color=C_DARK,
                ha="center", va="center")

    # Arrows: LLM -> Thought -> Action -> Tool -> Observation -> LLM
    arrows = [
        ((6.0, 4.4), (6.0, 5.0), C_PURPLE),   # LLM -> Thought
        ((7.0, 5.4), (8.7, 3.95), C_GREEN),   # Thought -> Action
        ((9.0, 3.0), (3.3, 3.0), C_AMBER),    # Action(tool) -> Observation
        ((2.0, 4.2), (5.0, 3.7), C_BLUE),     # Observation -> LLM
    ]
    for (a, b, c) in arrows:
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=16,
            color=c, linewidth=2.0,
            connectionstyle="arc3,rad=0.15"))

    # External tool environment
    ax.add_patch(FancyBboxPatch(
        (8.4, 0.6), 3.4, 1.4,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.4, edgecolor=C_GRAY, facecolor=C_LIGHT))
    ax.text(10.1, 1.65, "External world", fontsize=10, fontweight="bold",
            color=C_DARK, ha="center")
    ax.text(10.1, 1.10, "search engines, calculators,\ncode runners, databases",
            fontsize=8.6, color=C_DARK, ha="center")
    ax.add_patch(FancyArrowPatch(
        (9.5, 2.8), (9.7, 2.0), arrowstyle="-|>", mutation_scale=12,
        color=C_GRAY, linewidth=1.4, linestyle=(0, (3, 3))))

    # Loop label
    ax.text(6.0, 0.5,
            "Repeat until the model emits a final Answer instead of an Action.",
            fontsize=10, color=C_DARK, ha="center", style="italic")

    ax.set_title("ReAct: interleave Thought, Action, Observation",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=10)
    _save(fig, "fig6_react_pattern")


# ---------------------------------------------------------------------------
# fig7: Self-consistency / tree-of-thought
# ---------------------------------------------------------------------------
def fig7_self_consistency() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Root: question
    ax.add_patch(FancyBboxPatch(
        (4.8, 5.6), 2.4, 1.0,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        linewidth=1.8, edgecolor=C_DARK, facecolor=C_LIGHT))
    ax.text(6.0, 6.1, "Question", fontsize=11, fontweight="bold",
            color=C_DARK, ha="center")
    ax.text(6.0, 5.8, "(sample k=5 reasoning paths)",
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
        # Connector from root
        ax.add_patch(FancyArrowPatch(
            (6.0, 5.6), (x + 0.6, 5.0),
            arrowstyle="-|>", mutation_scale=10,
            color=color, linewidth=1.2, alpha=alpha))
        # Path name
        ax.text(x + 0.6, 4.8, name, fontsize=9.2, fontweight="bold",
                color=color, ha="center", alpha=alpha)
        # Steps stacked vertically
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
        # Final answer chip
        ax.add_patch(FancyBboxPatch(
            (x + 0.05, 2.25), 1.1, 0.5,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.6, edgecolor=color,
            facecolor=color + ("40" if correct else "20"),
            alpha=alpha))
        ax.text(x + 0.6, 2.5, f"-> {answer}", fontsize=10,
                fontweight="bold", color=C_DARK, ha="center",
                va="center", alpha=alpha)
        # Arrow down to vote box
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
    _save(fig, "fig7_self_consistency")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_prompt_anatomy()
    fig2_prompting_paradigms()
    fig3_cot_flow()
    fig4_shots_saturation()
    fig5_prompt_sensitivity()
    fig6_react_pattern()
    fig7_self_consistency()
    print(f"Wrote 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
