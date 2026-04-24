"""
Figures for the standalone post: AI Agents Complete Guide.

Generates 7 figures used by both EN and ZH versions. Each figure is rendered
once and saved into both asset folders so the two language versions stay
in sync.

Run:
    python ai-agents-complete-guide.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
RED = "#ef4444"
GREY = "#94a3b8"
DARK = "#1f2937"
LIGHT = "#f1f5f9"
WHITE = "#ffffff"

DPI = 150

EN_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/en/"
    "standalone/ai-agents-complete-guide"
)
ZH_DIR = Path(
    "/Users/kchen/Desktop/Project/chenk-site/source/_posts/zh/"
    "standalone/ai-agent完全指南-从理论到工业实践"
)


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure into both EN and ZH asset folders."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)


def rounded_box(ax, x, y, w, h, text, fc=BLUE, ec=None, tc=WHITE,
                fontsize=11, fontweight="bold", alpha=1.0, pad=0.08):
    """Draw a rounded rectangle with centered text."""
    if ec is None:
        ec = fc
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad={pad}",
        linewidth=1.5, edgecolor=ec, facecolor=fc, alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", color=tc,
            fontsize=fontsize, fontweight=fontweight)


def arrow(ax, p1, p2, color=DARK, lw=1.8, style="->",
          rad=0.0, mutation=18):
    """Draw a curved arrow from p1 to p2."""
    a = FancyArrowPatch(
        p1, p2,
        arrowstyle=style,
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw,
        mutation_scale=mutation,
    )
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Figure 1: Agent core loop (Observe -> Think -> Act -> Memory)
# ---------------------------------------------------------------------------

def fig1_agent_loop() -> None:
    fig, ax = plt.subplots(figsize=(10, 7.2), dpi=DPI)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.set_title("AI Agent Cognitive Loop: Perceive, Reason, Act, Remember",
                 fontsize=14, fontweight="bold", pad=14, color=DARK)

    # Center: LLM Brain
    center = (5, 3.6)
    brain_r = 0.95
    brain = Circle(center, brain_r, facecolor=PURPLE,
                   edgecolor=DARK, linewidth=2, zorder=5)
    ax.add_patch(brain)
    ax.text(center[0], center[1] + 0.12, "LLM",
            ha="center", va="center", color=WHITE,
            fontsize=14, fontweight="bold", zorder=6)
    ax.text(center[0], center[1] - 0.25, "Brain",
            ha="center", va="center", color=WHITE,
            fontsize=10, zorder=6)

    # 4 modules around the brain
    modules = [
        ("Observe", (5, 6.2), BLUE,
         "Sensors / Inputs\nuser msg, tool output"),
        ("Think", (8.0, 3.6), ORANGE,
         "Plan + Reason\nCoT / ReAct / ToT"),
        ("Act", (5, 1.0), GREEN,
         "Tool Calls\nAPI / code / search"),
        ("Memory", (2.0, 3.6), RED,
         "State + History\nshort + long term"),
    ]
    positions = []
    for label, (x, y), color, sub in modules:
        rounded_box(ax, x - 1.1, y - 0.55, 2.2, 1.1, label,
                    fc=color, fontsize=13, pad=0.12)
        ax.text(x, y - 0.95, sub, ha="center", va="top",
                fontsize=8.5, color=DARK, style="italic")
        positions.append((x, y))

    # Connect brain to each module
    for x, y in positions:
        # vector from center to module
        dx, dy = x - center[0], y - center[1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        ux, uy = dx / dist, dy / dist
        # line from brain edge to module edge
        p1 = (center[0] + ux * brain_r, center[1] + uy * brain_r)
        # stop at module box
        p2 = (x - ux * 1.15, y - uy * 0.6)
        arrow(ax, p1, p2, color=DARK, lw=1.6)

    # Outer cycle arrows (Observe -> Think -> Act -> Memory -> Observe)
    cycle_color = GREY
    cycle_pts = positions
    for i in range(4):
        p1 = cycle_pts[i]
        p2 = cycle_pts[(i + 1) % 4]
        arrow(ax, p1, p2, color=cycle_color, lw=1.4, rad=0.28,
              style="->", mutation=15)

    # Side annotations
    ax.text(0.2, 6.9, "Environment",
            fontsize=10, color=DARK, fontweight="bold")
    ax.text(0.2, 6.55, "users, files, web,\ncode, databases",
            fontsize=8.5, color=DARK, style="italic")

    ax.text(9.8, 0.6, "Goal achieved\nor max steps",
            ha="right", fontsize=9, color=DARK, style="italic")

    save(fig, "fig1_agent_loop.png")


# ---------------------------------------------------------------------------
# Figure 2: LLM-based Agent vs Rule-based system
# ---------------------------------------------------------------------------

def fig2_agent_vs_rule() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), dpi=DPI)
    fig.suptitle("Rule-Based System vs LLM-Based Agent",
                 fontsize=15, fontweight="bold", y=0.99, color=DARK)

    # ---- Left: Rule-based ----
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Rule-Based System", fontsize=12,
                 fontweight="bold", color=GREY, pad=8)

    rounded_box(ax, 1, 6.8, 8, 0.9, "Input: user query",
                fc=LIGHT, ec=GREY, tc=DARK, fontsize=11)

    # Decision tree
    rounded_box(ax, 3.5, 5.3, 3, 0.9, "if intent == A?",
                fc=ORANGE, fontsize=10)
    arrow(ax, (5, 6.8), (5, 6.2), color=DARK)

    rounded_box(ax, 0.5, 3.6, 2.4, 0.9, "Rule R1",
                fc=BLUE, fontsize=10)
    rounded_box(ax, 3.8, 3.6, 2.4, 0.9, "Rule R2",
                fc=BLUE, fontsize=10)
    rounded_box(ax, 7.1, 3.6, 2.4, 0.9, "Rule R3",
                fc=BLUE, fontsize=10)
    arrow(ax, (4.3, 5.3), (1.7, 4.5), color=DARK, rad=0.1)
    arrow(ax, (5, 5.3), (5, 4.5), color=DARK)
    arrow(ax, (5.7, 5.3), (8.3, 4.5), color=DARK, rad=-0.1)

    rounded_box(ax, 0.5, 1.9, 2.4, 0.9, "Template T1",
                fc=GREEN, fontsize=10)
    rounded_box(ax, 3.8, 1.9, 2.4, 0.9, "Template T2",
                fc=GREEN, fontsize=10)
    rounded_box(ax, 7.1, 1.9, 2.4, 0.9, "Template T3",
                fc=GREEN, fontsize=10)
    arrow(ax, (1.7, 3.6), (1.7, 2.8), color=DARK)
    arrow(ax, (5, 3.6), (5, 2.8), color=DARK)
    arrow(ax, (8.3, 3.6), (8.3, 2.8), color=DARK)

    rounded_box(ax, 1, 0.4, 8, 0.9, "Fixed-template response",
                fc=LIGHT, ec=GREY, tc=DARK, fontsize=11)
    arrow(ax, (5, 1.9), (5, 1.3), color=DARK)

    cons = [
        "- brittle: fails on new phrasing",
        "- expensive to extend (manual rules)",
        "- no real reasoning or planning",
    ]
    for i, c in enumerate(cons):
        ax.text(0.3, -0.5 - i * 0.35, c, fontsize=8.5, color=RED)

    # ---- Right: Agent ----
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("LLM-Based Agent", fontsize=12,
                 fontweight="bold", color=PURPLE, pad=8)

    rounded_box(ax, 1, 6.8, 8, 0.9, "Input: open-ended goal",
                fc=LIGHT, ec=GREY, tc=DARK, fontsize=11)

    # LLM core in middle
    rounded_box(ax, 3.5, 4.8, 3, 1.3, "LLM\nReason + Plan",
                fc=PURPLE, fontsize=11, pad=0.15)
    arrow(ax, (5, 6.8), (5, 6.1), color=DARK)

    # Tool ring
    tools = [
        ("Search", 0.7, 3.0, BLUE),
        ("Code", 3.5, 2.6, BLUE),
        ("DB", 6.4, 2.6, BLUE),
        ("API", 8.5, 3.0, BLUE),
    ]
    for name, x, y, c in tools:
        rounded_box(ax, x, y, 1.4, 0.85, name,
                    fc=c, fontsize=10)
        arrow(ax, (5, 4.8), (x + 0.7, y + 0.85),
              color=GREY, lw=1.2, rad=-0.15, style="<->")

    # Memory
    rounded_box(ax, 3.5, 0.8, 3, 0.9, "Memory + Reflection",
                fc=ORANGE, fontsize=10)
    arrow(ax, (5, 4.8), (5, 1.7), color=GREY,
          lw=1.2, rad=0.45, style="<->")

    pros = [
        "+ generalises to unseen tasks",
        "+ uses tools to ground in reality",
        "+ self-corrects via reflection",
    ]
    for i, c in enumerate(pros):
        ax.text(0.3, -0.5 - i * 0.35, c, fontsize=8.5, color=GREEN)

    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    save(fig, "fig2_agent_vs_rule.png")


# ---------------------------------------------------------------------------
# Figure 3: Tool use / function calling pipeline
# ---------------------------------------------------------------------------

def fig3_tool_calling_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=DPI)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Function Calling Pipeline: Schema -> LLM -> Execution -> Result",
                 fontsize=14, fontweight="bold", pad=14, color=DARK)

    steps = [
        (0.3, BLUE, "1. User query",
         "\"Weather in\nTokyo and Paris?\""),
        (2.7, PURPLE, "2. LLM picks tool",
         "Reads tool schemas,\nreturns JSON call(s)"),
        (5.4, ORANGE, "3. Validate args",
         "Type-check, sanitize,\nguardrails"),
        (8.1, GREEN, "4. Execute tool",
         "get_weather(city=...)\nAPI / code / DB"),
        (10.8, BLUE, "5. LLM synthesises",
         "Tool output ->\nnatural language"),
    ]

    box_w, box_h = 2.2, 1.4
    y_box = 3.1
    centers = []
    for x, color, title, body in steps:
        rounded_box(ax, x, y_box, box_w, box_h, title,
                    fc=color, fontsize=10.5, pad=0.1)
        ax.text(x + box_w / 2, y_box - 0.55, body,
                ha="center", va="top", fontsize=8.5,
                color=DARK, style="italic")
        centers.append((x + box_w / 2, y_box + box_h / 2))

    # Arrows between steps
    for i in range(len(centers) - 1):
        x1 = steps[i][0] + box_w
        x2 = steps[i + 1][0]
        y = y_box + box_h / 2
        arrow(ax, (x1, y), (x2, y), color=DARK, lw=2)

    # JSON example below
    json_text = (
        '{\n'
        '  "name": "get_weather",\n'
        '  "arguments": {\n'
        '    "city": "Tokyo",\n'
        '    "unit": "celsius"\n'
        '  }\n'
        '}'
    )
    ax.text(0.3, 1.4, "Schema given to LLM:",
            fontsize=10, fontweight="bold", color=DARK)
    schema_text = (
        'tools = [{\n'
        '  "name": "get_weather",\n'
        '  "description": "Get weather",\n'
        '  "parameters": {\n'
        '    "city": {"type": "string"},\n'
        '    "unit": {"enum": ["c","f"]}\n'
        '  }\n'
        '}]'
    )
    ax.text(0.3, 1.1, schema_text, fontsize=8.5, color=DARK,
            family="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT,
                      edgecolor=GREY))

    ax.text(7.0, 1.4, "Model response:",
            fontsize=10, fontweight="bold", color=DARK)
    ax.text(7.0, 1.1, json_text, fontsize=8.5, color=DARK,
            family="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT,
                      edgecolor=PURPLE))

    # Loop arrow back from step 5 to step 2 (multi-turn)
    arrow(ax, (centers[4][0], y_box + box_h),
          (centers[1][0], y_box + box_h),
          color=GREY, lw=1.4, rad=-0.35, style="->")
    ax.text(7, 5.55, "loop until done",
            ha="center", fontsize=9, color=GREY, style="italic")

    save(fig, "fig3_tool_calling_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 4: Multi-agent collaboration patterns
# ---------------------------------------------------------------------------

def fig4_multi_agent_patterns() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), dpi=DPI)
    fig.suptitle("Multi-Agent Collaboration Patterns",
                 fontsize=15, fontweight="bold", y=1.0, color=DARK)

    # Pattern 1: Hierarchical
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Hierarchical (Manager + Workers)",
                 fontsize=11, fontweight="bold", color=DARK, pad=4)

    rounded_box(ax, 3.5, 4.5, 3, 0.9, "Manager", fc=PURPLE, fontsize=11)
    workers = [("Researcher", 0.5, BLUE),
               ("Analyst", 4, GREEN),
               ("Writer", 7.5, ORANGE)]
    for name, x, c in workers:
        rounded_box(ax, x, 1.8, 2, 0.9, name, fc=c, fontsize=10)
        arrow(ax, (5, 4.5), (x + 1, 2.7), color=DARK, lw=1.4)
    ax.text(5, 0.5, "Top-down task decomposition,\nresults bubble up",
            ha="center", fontsize=8.5, color=DARK, style="italic")

    # Pattern 2: Debate
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Debate (Adversarial Critique)",
                 fontsize=11, fontweight="bold", color=DARK, pad=4)

    rounded_box(ax, 0.5, 3.5, 2.5, 1, "Agent A\n(pro)",
                fc=BLUE, fontsize=10, pad=0.12)
    rounded_box(ax, 7, 3.5, 2.5, 1, "Agent B\n(con)",
                fc=RED, fontsize=10, pad=0.12)
    rounded_box(ax, 3.75, 0.7, 2.5, 1, "Judge",
                fc=PURPLE, fontsize=10, pad=0.12)
    arrow(ax, (3, 4), (7, 4), color=DARK, lw=1.4, rad=0.18, style="<->")
    arrow(ax, (1.75, 3.5), (4.5, 1.7), color=DARK, lw=1.4, rad=-0.1)
    arrow(ax, (8.25, 3.5), (5.5, 1.7), color=DARK, lw=1.4, rad=0.1)
    ax.text(5, 5.5, "rounds of argument & rebuttal",
            ha="center", fontsize=8.5, color=DARK, style="italic")

    # Pattern 3: Pipeline / Assembly line
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Pipeline (Specialist Chain)",
                 fontsize=11, fontweight="bold", color=DARK, pad=4)

    stages = [("PM", 0.3, BLUE),
              ("Arch", 2.7, PURPLE),
              ("Eng", 5.1, GREEN),
              ("QA", 7.5, ORANGE)]
    for name, x, c in stages:
        rounded_box(ax, x, 2.8, 2, 1, name, fc=c, fontsize=11, pad=0.12)
    for i in range(len(stages) - 1):
        x1 = stages[i][1] + 2
        x2 = stages[i + 1][1]
        arrow(ax, (x1, 3.3), (x2, 3.3), color=DARK, lw=1.8)
    # feedback arrow QA -> Eng
    arrow(ax, (8.5, 2.8), (6.1, 2.8), color=RED, lw=1.4,
          rad=0.4, style="->")
    ax.text(7.3, 1.7, "bug feedback", ha="center", fontsize=8.5,
            color=RED, style="italic")
    ax.text(5, 0.5, "MetaGPT-style software team",
            ha="center", fontsize=8.5, color=DARK, style="italic")

    plt.tight_layout(rect=[0, 0.0, 1, 0.94])
    save(fig, "fig4_multi_agent_patterns.png")


# ---------------------------------------------------------------------------
# Figure 5: ReAct flow (Thought / Action / Observation loop)
# ---------------------------------------------------------------------------

def fig5_react_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2), dpi=DPI)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("ReAct Trace: Interleaved Reasoning and Acting",
                 fontsize=14, fontweight="bold", pad=14, color=DARK)

    # Question at the top
    rounded_box(ax, 0.5, 6.0, 11, 0.85,
                "Q: \"Which 2024 Nobel-physics laureate has the highest h-index?\"",
                fc=DARK, fontsize=11, pad=0.1)

    # Trace rows
    rows = [
        ("Thought 1", "I need the laureate list first.",
         PURPLE, 5.0),
        ("Action 1", "search(\"2024 Nobel physics laureates\")",
         BLUE, 4.2),
        ("Observation 1", "Hopfield, Hinton",
         GREY, 3.4),
        ("Thought 2", "Now look up h-indices for each.",
         PURPLE, 2.6),
        ("Action 2", "scholar_lookup(\"Geoffrey Hinton\")",
         BLUE, 1.8),
        ("Observation 2", "h-index = 188",
         GREY, 1.0),
        ("Final Answer", "Hinton (h-index 188)",
         GREEN, 0.2),
    ]

    label_w = 1.8
    body_w = 8.5
    for label, body, color, y in rows:
        rounded_box(ax, 0.5, y, label_w, 0.65, label,
                    fc=color, fontsize=10, pad=0.06)
        rounded_box(ax, label_w + 0.7, y, body_w, 0.65, body,
                    fc=LIGHT, ec=color, tc=DARK,
                    fontsize=10, fontweight="normal", pad=0.06)

    # Vertical arrows linking rows
    for i in range(len(rows) - 1):
        y1 = rows[i][3]
        y2 = rows[i + 1][3] + 0.65
        arrow(ax, (1.4, y1), (1.4, y2), color=DARK, lw=1.4, mutation=12)

    # Side label: loop hint
    ax.annotate("Reason -> Act -> Observe\nrepeats until 'Final Answer'",
                xy=(11.7, 3.0), ha="right", fontsize=9.5,
                color=DARK, style="italic")

    save(fig, "fig5_react_flow.png")


# ---------------------------------------------------------------------------
# Figure 6: Industrial agent platform landscape
# ---------------------------------------------------------------------------

def fig6_platform_landscape() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.2), dpi=DPI)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_title("Agent Frameworks & Platforms: Positioning Map",
                 fontsize=14, fontweight="bold", pad=14, color=DARK)

    # Axes
    ax.annotate("", xy=(11.5, 0.6), xytext=(0.5, 0.6),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5))
    ax.annotate("", xy=(0.5, 6.6), xytext=(0.5, 0.6),
                arrowprops=dict(arrowstyle="->", color=DARK, lw=1.5))
    ax.text(11.6, 0.4, "Autonomy ->", fontsize=10, color=DARK,
            fontweight="bold")
    ax.text(0.4, 6.7, "Composability ^", fontsize=10, color=DARK,
            fontweight="bold")
    ax.text(0.6, 0.3, "scripted", fontsize=8, color=GREY, style="italic")
    ax.text(11.0, 0.3, "self-directed", fontsize=8, color=GREY,
            style="italic")

    # Quadrant lines (light)
    ax.plot([6, 6], [0.6, 6.6], color=GREY, lw=0.8, ls="--", alpha=0.4)
    ax.plot([0.5, 11.5], [3.6, 3.6], color=GREY, lw=0.8, ls="--", alpha=0.4)

    # Platform points (x, y, label, color, kind)
    items = [
        # composable / scripted
        (1.8, 5.4, "LangChain", BLUE, "Toolkit"),
        (2.2, 4.6, "LlamaIndex", BLUE, "RAG-first"),
        (3.5, 5.8, "LangGraph", PURPLE, "DAG runtime"),
        (3.5, 4.2, "Semantic\nKernel", BLUE, "MS SDK"),
        # composable / autonomous
        (8.2, 6.0, "CrewAI", GREEN, "Multi-agent"),
        (9.2, 5.2, "AutoGen", GREEN, "MS multi-agent"),
        (7.5, 4.4, "MetaGPT", GREEN, "SW team"),
        # scripted / low autonomy product
        (2.0, 2.5, "OpenAI\nAssistants", ORANGE, "Hosted"),
        (3.5, 1.8, "Bedrock\nAgents", ORANGE, "AWS"),
        (5.0, 2.5, "Dify /\nFlowise", ORANGE, "No-code"),
        # autonomous, low composability
        (8.5, 2.4, "AutoGPT", RED, "Goal-driven"),
        (10.0, 1.8, "BabyAGI", RED, "Task loop"),
        (9.0, 3.0, "Devin-like", RED, "Agentic IDE"),
    ]

    for x, y, label, color, kind in items:
        ax.scatter([x], [y], s=320, color=color, edgecolor=DARK,
                   linewidth=1.2, zorder=3)
        ax.text(x, y - 0.42, label, ha="center", va="top",
                fontsize=9, color=DARK, fontweight="bold")
        ax.text(x, y - 0.85, kind, ha="center", va="top",
                fontsize=7.5, color=GREY, style="italic")

    # Legend
    legend_items = [
        (BLUE, "Composable SDK"),
        (PURPLE, "Graph runtime"),
        (GREEN, "Multi-agent"),
        (ORANGE, "Hosted product"),
        (RED, "Autonomous loop"),
    ]
    for i, (c, t) in enumerate(legend_items):
        y = 6.5 - i * 0.35
        ax.scatter([10.5], [y], s=120, color=c, edgecolor=DARK, linewidth=1)
        ax.text(10.8, y, t, fontsize=8.5, color=DARK, va="center")

    save(fig, "fig6_platform_landscape.png")


# ---------------------------------------------------------------------------
# Figure 7: Agent evaluation framework
# ---------------------------------------------------------------------------

def fig7_evaluation_framework() -> None:
    fig = plt.figure(figsize=(14, 6.5), dpi=DPI)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1])

    # Left: dimensions of evaluation (radar)
    ax1 = fig.add_subplot(gs[0, 0], projection="polar")
    ax1.set_title("Evaluation Dimensions (illustrative scores)",
                  fontsize=12, fontweight="bold", color=DARK, pad=14)

    dims = ["Planning", "Tool Use", "Memory",
            "Reflection", "Robustness", "Cost\nEfficiency"]
    n = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    agents = [
        ("GPT-4 Agent", [0.82, 0.85, 0.70, 0.75, 0.70, 0.45], PURPLE),
        ("Claude Agent", [0.84, 0.88, 0.72, 0.78, 0.74, 0.55], BLUE),
        ("OSS 70B Agent", [0.55, 0.58, 0.50, 0.50, 0.45, 0.85], GREEN),
    ]
    for name, vals, color in agents:
        v = vals + vals[:1]
        ax1.plot(angles, v, color=color, linewidth=2, label=name)
        ax1.fill(angles, v, color=color, alpha=0.15)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(dims, fontsize=9.5, color=DARK)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax1.set_yticklabels(["0.2", "0.4", "0.6", "0.8"],
                        fontsize=8, color=GREY)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper right", bbox_to_anchor=(1.32, 1.1),
               fontsize=9, frameon=True)

    # Right: benchmark coverage matrix
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("What Each Benchmark Covers",
                  fontsize=12, fontweight="bold", color=DARK, pad=14)

    benchmarks = ["AgentBench", "GAIA", "AgentBoard",
                  "WebArena", "SWE-bench", "ToolBench"]
    capabilities = ["OS / Code", "Web nav", "Multi-hop QA",
                    "Tool API", "Long horizon", "Multi-modal"]

    # 1 = covered, 0.5 = partial, 0 = no
    matrix = np.array([
        # OS  Web  QA   Tool LH   MM
        [1.0, 1.0, 0.5, 1.0, 0.5, 0.0],   # AgentBench
        [0.5, 1.0, 1.0, 1.0, 1.0, 1.0],   # GAIA
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.5],   # AgentBoard
        [0.0, 1.0, 0.5, 0.5, 0.5, 0.0],   # WebArena
        [1.0, 0.0, 0.0, 0.5, 1.0, 0.0],   # SWE-bench
        [0.0, 0.0, 0.5, 1.0, 0.5, 0.0],   # ToolBench
    ])

    cmap = plt.cm.get_cmap("BuPu")
    im = ax2.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(capabilities)))
    ax2.set_xticklabels(capabilities, rotation=30, ha="right",
                        fontsize=9, color=DARK)
    ax2.set_yticks(range(len(benchmarks)))
    ax2.set_yticklabels(benchmarks, fontsize=10, color=DARK)
    ax2.set_xticks(np.arange(-0.5, len(capabilities), 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(benchmarks), 1), minor=True)
    ax2.grid(which="minor", color=WHITE, linewidth=2)
    ax2.tick_params(which="minor", length=0)

    for i in range(len(benchmarks)):
        for j in range(len(capabilities)):
            v = matrix[i, j]
            if v == 1.0:
                txt = "Yes"
            elif v == 0.5:
                txt = "Part"
            else:
                txt = "-"
            color = WHITE if v >= 0.5 else DARK
            ax2.text(j, i, txt, ha="center", va="center",
                     color=color, fontsize=8.5, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.03)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["no", "partial", "yes"])
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("Evaluation Framework: Capability Profile + Benchmark Coverage",
                 fontsize=14, fontweight="bold", y=1.02, color=DARK)

    plt.tight_layout()
    save(fig, "fig7_evaluation_framework.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating figure 1 (agent loop)...")
    fig1_agent_loop()

    print("Generating figure 2 (agent vs rule)...")
    fig2_agent_vs_rule()

    print("Generating figure 3 (tool calling pipeline)...")
    fig3_tool_calling_pipeline()

    print("Generating figure 4 (multi-agent patterns)...")
    fig4_multi_agent_patterns()

    print("Generating figure 5 (ReAct flow)...")
    fig5_react_flow()

    print("Generating figure 6 (platform landscape)...")
    fig6_platform_landscape()

    print("Generating figure 7 (evaluation framework)...")
    fig7_evaluation_framework()

    print(f"\nDone. Saved to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
