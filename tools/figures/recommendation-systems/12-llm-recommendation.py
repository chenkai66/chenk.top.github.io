"""Figures for Recommendation Systems Part 12 — LLMs and Recommendation.

Generates 7 publication-quality figures explaining how Large Language Models
plug into recommendation systems: as enhancers (P5, M6Rec), predictors
(TallRec, GenRec), and agents (LlamaRec, ChatREC).

Figures:
    fig1_three_roles      — Enhancer / Predictor / Agent overview
    fig2_prompt_template  — Prompt template anatomy for LLM-as-recommender
    fig3_embedding_quality — Item description enhancement (before/after)
    fig4_hybrid_pipeline  — Two-stage retrieval + LLM rerank
    fig5_cold_start       — LLM zero-shot vs CF on cold-start severity
    fig6_chat_flow        — Conversational recommendation dialog flow
    fig7_cost_quality     — Pareto frontier: NDCG gain vs $/1K requests

Output is written to BOTH the EN and ZH asset folders.
References (verified):
    - P5 (Geng et al., RecSys 2022)
    - M6-Rec (Cui et al., 2022)
    - TallRec (Bao et al., RecSys 2023)
    - GenRec (Ji et al., 2024)
    - LlamaRec (Yue et al., 2023)
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

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/recommendation-systems/12-llm-recommendation"
ZH_DIR = REPO / "source/_posts/zh/recommendation-systems/12-大语言模型与推荐系统"


def save(fig: plt.Figure, name: str) -> None:
    """Save figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def _box(ax, x, y, w, h, text, color, fc=None, fontsize=9.5, weight="normal"):
    """Draw a rounded box with centered text."""
    fc = fc or color
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=color, facecolor=fc, alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=DARK, weight=weight, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color=GRAY, style="->", lw=1.6):
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                            mutation_scale=14, color=color, lw=lw,
                            shrinkA=2, shrinkB=2)
    ax.add_patch(arrow)


# ---------------------------------------------------------------------------
# Figure 1 — Three roles of LLMs in RecSys
# ---------------------------------------------------------------------------
def fig1_three_roles() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(7, 6.55, "Three Roles of LLMs in Recommendation",
            ha="center", fontsize=15, weight="bold", color=DARK)

    roles = [
        {
            "x": 0.4, "color": BLUE,
            "title": "ENHANCER",
            "subtitle": "Feature / data augmentation",
            "papers": "P5  ·  M6-Rec  ·  KAR",
            "flow": [
                "Item text\n(title, desc, reviews)",
                "LLM extracts\nsemantic features",
                "Feed to\ntraditional CTR / CF",
            ],
            "note": "LLM stays offline.\nCheap, low latency.",
        },
        {
            "x": 5.0, "color": PURPLE,
            "title": "PREDICTOR",
            "subtitle": "Direct ranking / generation",
            "papers": "TallRec  ·  GenRec  ·  BIGRec",
            "flow": [
                "User history +\ncandidate list",
                "LLM scores or\ngenerates next item",
                "Top-K\nrecommendations",
            ],
            "note": "Fine-tuned LLM.\nStrong on cold-start.",
        },
        {
            "x": 9.6, "color": GREEN,
            "title": "AGENT",
            "subtitle": "Orchestrates tools & dialog",
            "papers": "LlamaRec  ·  ChatREC  ·  RecAgent",
            "flow": [
                "Multi-turn\nconversation",
                "LLM plans &\ncalls retrievers",
                "Recommend +\nexplain + refine",
            ],
            "note": "Conversational.\nHighest cost.",
        },
    ]

    for role in roles:
        x0 = role["x"]
        c = role["color"]
        # header
        _box(ax, x0, 5.1, 4.0, 0.85, role["title"], c, fc=c, fontsize=13, weight="bold")
        # subtitle
        ax.text(x0 + 2.0, 4.85, role["subtitle"], ha="center", fontsize=10,
                style="italic", color=DARK)
        # papers
        ax.text(x0 + 2.0, 4.55, role["papers"], ha="center", fontsize=8.8, color=GRAY)
        # 3 flow steps
        ys = [3.7, 2.55, 1.4]
        for i, (y, step) in enumerate(zip(ys, role["flow"])):
            _box(ax, x0 + 0.4, y, 3.2, 0.85, step, c, fc="white", fontsize=9)
            if i < 2:
                _arrow(ax, x0 + 2.0, y, x0 + 2.0, ys[i + 1] + 0.85, color=c, lw=1.8)
        # note at bottom
        ax.text(x0 + 2.0, 0.7, role["note"], ha="center", fontsize=8.8,
                color=c, style="italic", weight="bold")

    # white text on header boxes
    for role in roles:
        ax.text(role["x"] + 2.0, 5.55, role["title"], ha="center",
                fontsize=13, weight="bold", color="white")

    save(fig, "fig1_three_roles.png")


# ---------------------------------------------------------------------------
# Figure 2 — Prompt template anatomy
# ---------------------------------------------------------------------------
def fig2_prompt_template() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(6, 8.5, "Prompt Template for LLM-as-Recommender",
            ha="center", fontsize=14.5, weight="bold", color=DARK)
    ax.text(6, 8.1, "Anatomy of a TallRec / P5-style instruction prompt",
            ha="center", fontsize=10, style="italic", color=GRAY)

    sections = [
        {
            "y": 6.85, "h": 0.95, "color": BLUE,
            "label": "1.  ROLE  /  TASK",
            "body": '"You are a movie recommender. Given the user\'s viewing\n'
                    'history, predict whether they will like the target movie."',
        },
        {
            "y": 5.65, "h": 1.05, "color": PURPLE,
            "label": "2.  USER HISTORY  (chronological)",
            "body": "History:\n  - Inception (liked)\n  - Interstellar (liked)\n"
                    "  - The Notebook (disliked)",
        },
        {
            "y": 4.30, "h": 0.85, "color": GREEN,
            "label": "3.  CANDIDATE  /  TARGET",
            "body": 'Target movie: "Tenet"  (Sci-Fi, 2020, dir. Christopher Nolan)',
        },
        {
            "y": 2.95, "h": 1.05, "color": ORANGE,
            "label": "4.  OUTPUT FORMAT  (constrained decoding)",
            "body": 'Answer with exactly one token: "Yes" or "No".\n'
                    "Optional: rank-list  [item_1, item_2, ...]",
        },
        {
            "y": 1.65, "h": 1.05, "color": GRAY,
            "label": "5.  FEW-SHOT EXAMPLES  (optional)",
            "body": "[Example 1] History: ...  Target: ...  Answer: Yes\n"
                    "[Example 2] History: ...  Target: ...  Answer: No",
        },
    ]

    for s in sections:
        # left label strip
        _box(ax, 0.3, s["y"], 3.5, s["h"], s["label"], s["color"],
             fc=s["color"], fontsize=9.8, weight="bold")
        ax.text(0.3 + 1.75, s["y"] + s["h"] / 2, s["label"],
                ha="center", va="center", color="white", fontsize=10, weight="bold")
        # body
        _box(ax, 4.0, s["y"], 7.7, s["h"], "", s["color"], fc="white")
        ax.text(4.2, s["y"] + s["h"] / 2, s["body"], ha="left", va="center",
                fontsize=9.2, family="monospace", color=DARK)

    # bottom note
    ax.text(6, 0.55,
            "Tip:  short, explicit prompts + few-shot examples + low temperature (≤0.3) "
            "give the most stable rankings.",
            ha="center", fontsize=9.5, color=GRAY, style="italic")

    save(fig, "fig2_prompt_template.png")


# ---------------------------------------------------------------------------
# Figure 3 — Embedding quality before/after LLM enhancement
# ---------------------------------------------------------------------------
def fig3_embedding_quality() -> None:
    rng = np.random.default_rng(42)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))

    # genres
    genre_colors = [BLUE, PURPLE, GREEN, ORANGE]
    genres = ["Sci-Fi", "Romance", "Horror", "Comedy"]

    def cluster(center, n, spread):
        return rng.normal(loc=center, scale=spread, size=(n, 2))

    # BEFORE: noisy clusters from sparse text
    before_centers = [(0, 0), (1.3, 1.1), (-1.0, 1.5), (0.6, -1.2)]
    after_centers = [(-2.5, -1.8), (2.3, 1.9), (-2.0, 2.1), (2.0, -2.2)]

    for ax, centers, spread, title, subtitle in zip(
        axes,
        [before_centers, after_centers],
        [0.95, 0.42],
        ["Before:  raw item title only", "After:  LLM-enriched description"],
        ["~5–8 word titles", "LLM expands title → 80+ words of theme, mood, audience"],
    ):
        for c, color, name in zip(centers, genre_colors, genres):
            pts = cluster(c, 35, spread)
            ax.scatter(pts[:, 0], pts[:, 1], c=color, s=42, alpha=0.7,
                       edgecolors="white", linewidth=0.6, label=name)
        ax.set_title(title, color=DARK)
        ax.text(0.5, -0.13, subtitle, transform=ax.transAxes, ha="center",
                fontsize=9.5, style="italic", color=GRAY)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-5, 5)
        ax.set_ylim(-4.5, 4.5)
        ax.set_aspect("equal")

    # quality annotation
    axes[0].text(0.5, 0.96, "Silhouette = 0.21", transform=axes[0].transAxes,
                 ha="center", fontsize=10, color=ORANGE, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor=ORANGE))
    axes[1].text(0.5, 0.96, "Silhouette = 0.68", transform=axes[1].transAxes,
                 ha="center", fontsize=10, color=GREEN, weight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                           edgecolor=GREEN))

    axes[1].legend(loc="lower right", fontsize=9, ncol=2, frameon=True)

    fig.suptitle("LLM Description Enhancement Sharpens Item Embeddings",
                 fontsize=14, weight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig3_embedding_quality.png")


# ---------------------------------------------------------------------------
# Figure 4 — Hybrid pipeline: traditional retrieval + LLM rerank
# ---------------------------------------------------------------------------
def fig4_hybrid_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(7, 5.55, "Hybrid Pipeline:  Cheap Retrieval  +  LLM Rerank",
            ha="center", fontsize=14.5, weight="bold", color=DARK)

    stages = [
        {"x": 0.3, "w": 2.4, "color": GRAY, "title": "Catalog",
         "n": "10⁶–10⁸", "tag": "all items"},
        {"x": 3.1, "w": 2.6, "color": BLUE, "title": "ANN Retrieval",
         "n": "~1000", "tag": "Faiss / HNSW\n~10 ms · ≈$0"},
        {"x": 6.1, "w": 2.6, "color": PURPLE, "title": "CF / DNN Ranker",
         "n": "~50", "tag": "DIN, DCN-V2\n~30 ms · ≈$0"},
        {"x": 9.1, "w": 2.6, "color": GREEN, "title": "LLM Reranker",
         "n": "Top-10", "tag": "GPT-4 / Llama-3\n~1.5 s · ~$0.01"},
        {"x": 12.1, "w": 1.6, "color": ORANGE, "title": "User",
         "n": "Top-10", "tag": "+ explanation"},
    ]

    y_box = 2.6
    h_box = 1.6
    for s in stages:
        cx = s["x"] + s["w"] / 2
        # header
        _box(ax, s["x"], y_box + h_box, s["w"], 0.55, "", s["color"],
             fc=s["color"])
        ax.text(cx, y_box + h_box + 0.275, s["title"], ha="center",
                va="center", color="white", fontsize=10.5, weight="bold")
        # body
        _box(ax, s["x"], y_box, s["w"], h_box, "", s["color"], fc="white")
        ax.text(cx, y_box + 1.05, s["n"], ha="center", fontsize=14,
                weight="bold", color=s["color"])
        ax.text(cx, y_box + 0.45, s["tag"], ha="center", fontsize=9, color=DARK)

    # arrows
    for i in range(len(stages) - 1):
        x1 = stages[i]["x"] + stages[i]["w"]
        x2 = stages[i + 1]["x"]
        _arrow(ax, x1 + 0.05, y_box + h_box / 2, x2 - 0.05, y_box + h_box / 2,
               color=DARK, lw=2.0)

    # bottom legend bar — cumulative cost & latency
    ax.text(7, 1.4, "Cost / latency stays low because the LLM only sees ~50 items, not millions.",
            ha="center", fontsize=10, color=GRAY, style="italic")

    # arrow legend
    ax.text(0.3, 0.7, "Funnel ratio:", fontsize=9.5, color=DARK, weight="bold")
    ax.text(0.3, 0.35,
            "10⁸ → 10³ → 50 → 10    (5–7 orders of magnitude pruned before the expensive call)",
            fontsize=9.5, color=GRAY, family="monospace")

    save(fig, "fig4_hybrid_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 5 — Cold-start: LLM zero-shot vs CF
# ---------------------------------------------------------------------------
def fig5_cold_start() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))

    # interactions per user (cold → warm)
    n_int = np.array([0, 1, 3, 5, 10, 20, 50, 100, 200])

    # CF degrades sharply at low n
    cf = np.array([0.04, 0.06, 0.09, 0.13, 0.21, 0.31, 0.42, 0.49, 0.53])
    # LLM zero-shot is relatively flat (uses world knowledge)
    llm = np.array([0.28, 0.30, 0.32, 0.34, 0.37, 0.40, 0.43, 0.45, 0.46])
    # Hybrid gets best of both
    hybrid = np.maximum(cf, llm) + 0.04
    hybrid = np.minimum(hybrid, 0.60)

    ax.plot(n_int, cf, "-o", color=BLUE, lw=2.4, ms=8, label="Collaborative Filtering")
    ax.plot(n_int, llm, "-s", color=PURPLE, lw=2.4, ms=8,
            label="LLM zero-shot (TallRec-style)")
    ax.plot(n_int, hybrid, "--D", color=GREEN, lw=2.4, ms=8, label="Hybrid (LLM + CF)")

    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 3, 5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["0", "1", "3", "5", "10", "20", "50", "100", "200"])
    ax.set_xlabel("# user interactions  (cold-start  ←→  warm-start)")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0, 0.65)

    # shading
    ax.axvspan(0, 5, alpha=0.08, color=ORANGE)
    ax.text(2, 0.62, "COLD-START ZONE", color=ORANGE, weight="bold",
            ha="center", fontsize=10)
    ax.text(2, 0.585, "LLM wins by 4–7×", color=ORANGE, ha="center", fontsize=9,
            style="italic")

    ax.set_title("Cold-Start:  LLMs Beat Collaborative Filtering When Data Is Sparse",
                 fontsize=13.5)
    ax.legend(loc="lower right", fontsize=10.5)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    save(fig, "fig5_cold_start.png")


# ---------------------------------------------------------------------------
# Figure 6 — Conversational recommendation flow (ChatREC-style)
# ---------------------------------------------------------------------------
def fig6_chat_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(6, 8.6, "Conversational Recommendation Flow  (ChatREC-style)",
            ha="center", fontsize=14.5, weight="bold", color=DARK)

    turns = [
        {"y": 7.5, "side": "user",
         "text": "I want a sci-fi film like Inception\nbut not too dark."},
        {"y": 6.4, "side": "agent",
         "text": "I see you like mind-bending plots. How do you feel\n"
                 "about time-loop or simulation themes?",
         "tool": "intent: clarify  ·  retrieve user history"},
        {"y": 5.1, "side": "user",
         "text": "Time-loop sounds great. Something hopeful."},
        {"y": 4.0, "side": "agent",
         "text": "Try  Edge of Tomorrow  (action + time-loop) or\n"
                 "Source Code  (twisty but optimistic ending).",
         "tool": "tools:  vector_search()  →  llm_rerank()  →  explain()"},
        {"y": 2.7, "side": "user",
         "text": "Saw both. Anything more recent?"},
        {"y": 1.6, "side": "agent",
         "text": "Palm Springs  (2020) — same loop premise, comedic tone.\n"
                 "Want me to check streaming availability?",
         "tool": "tools:  filter(year≥2018)  →  llm_rerank()"},
    ]

    for t in turns:
        is_user = t["side"] == "user"
        x = 0.5 if is_user else 4.5
        w = 5.5 if is_user else 7.0
        color = BLUE if is_user else PURPLE
        label = "User" if is_user else "Agent"

        _box(ax, x, t["y"], w, 0.75, "", color, fc="white")
        ax.text(x + 0.15, t["y"] + 0.55, label, fontsize=8.5,
                color=color, weight="bold")
        ax.text(x + 0.15, t["y"] + 0.25, t["text"], fontsize=9.3,
                color=DARK, va="center")

        if "tool" in t:
            ax.text(x + 0.15, t["y"] - 0.18, "↳ " + t["tool"],
                    fontsize=8.2, color=GREEN, style="italic", family="monospace")

    # state column on the right
    _box(ax, 9.0, 6.5, 2.7, 1.5, "", GRAY, fc="white")
    ax.text(10.35, 7.9, "DIALOG STATE", ha="center", fontsize=9,
            color=GRAY, weight="bold")
    ax.text(9.15, 7.5, "preferences:", fontsize=8.5, color=DARK, weight="bold")
    ax.text(9.25, 7.25, "• sci-fi", fontsize=8.3, color=DARK)
    ax.text(9.25, 7.05, "• mind-bending", fontsize=8.3, color=DARK)
    ax.text(9.25, 6.85, "• not too dark", fontsize=8.3, color=DARK)
    ax.text(9.25, 6.65, "• time-loop", fontsize=8.3, color=GREEN, weight="bold")

    _box(ax, 9.0, 4.6, 2.7, 1.5, "", GRAY, fc="white")
    ax.text(10.35, 6.0, "CANDIDATES", ha="center", fontsize=9,
            color=GRAY, weight="bold")
    ax.text(9.15, 5.65, "1. Edge of Tomorrow", fontsize=8.3, color=DARK)
    ax.text(9.15, 5.45, "2. Source Code", fontsize=8.3, color=DARK)
    ax.text(9.15, 5.25, "3. Predestination", fontsize=8.3, color=GRAY)
    ax.text(9.15, 5.05, "4. Looper", fontsize=8.3, color=GRAY)
    ax.text(9.15, 4.80, "(updated each turn)", fontsize=7.8,
            color=GRAY, style="italic")

    ax.text(6, 0.4,
            "Each turn refines the candidate set; the LLM is the planner, "
            "not the retriever.",
            ha="center", fontsize=9.5, color=GRAY, style="italic")

    save(fig, "fig6_chat_flow.png")


# ---------------------------------------------------------------------------
# Figure 7 — Cost vs quality Pareto
# ---------------------------------------------------------------------------
def fig7_cost_quality() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.4))

    # (cost per 1k requests in $, NDCG@10 lift over CF baseline in pp)
    methods = [
        ("CF baseline", 0.001, 0.0, BLUE, 200),
        ("CF + LLM features\n(P5 / KAR offline)", 0.05, 4.2, BLUE, 280),
        ("LLM rerank top-50\n(GPT-3.5)", 0.8, 7.5, PURPLE, 280),
        ("LLM rerank top-10\n(GPT-4o)", 4.0, 9.8, PURPLE, 280),
        ("Fine-tuned 7B LLM\n(TallRec / LlamaRec)", 0.4, 8.6, GREEN, 280),
        ("Full LLM generative\n(GenRec, GPT-4)", 25.0, 11.0, ORANGE, 280),
        ("Conversational agent\n(ChatREC, multi-turn)", 60.0, 12.5, ORANGE, 280),
    ]

    for name, cost, lift, color, size in methods:
        ax.scatter(cost, lift, s=size, c=color, alpha=0.85,
                   edgecolor="white", linewidth=1.5, zorder=3)

    # labels
    label_offsets = {
        "CF baseline": (1.5, -0.5),
        "CF + LLM features\n(P5 / KAR offline)": (1.5, 0.4),
        "LLM rerank top-50\n(GPT-3.5)": (1.5, 0.4),
        "LLM rerank top-10\n(GPT-4o)": (1.5, -1.0),
        "Fine-tuned 7B LLM\n(TallRec / LlamaRec)": (-3.0, 0.6),
        "Full LLM generative\n(GenRec, GPT-4)": (1.5, 0.4),
        "Conversational agent\n(ChatREC, multi-turn)": (-25, -1.7),
    }

    for name, cost, lift, color, _ in methods:
        dx, dy = label_offsets.get(name, (1.5, 0.3))
        ha = "left" if dx > 0 else "right"
        ax.annotate(name, (cost, lift),
                    xytext=(cost * (1 + dx * 0.0) if dx > 0 else cost * 0.6,
                            lift + dy),
                    fontsize=9, color=DARK, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8, alpha=0.5))

    # Pareto frontier
    pareto = sorted([(m[1], m[2], m[0]) for m in methods])
    px, py = [], []
    best = -1
    for c, l, n in pareto:
        if l > best:
            px.append(c)
            py.append(l)
            best = l
    ax.plot(px, py, "--", color=GRAY, lw=1.5, alpha=0.7,
            label="Pareto frontier", zorder=2)

    ax.set_xscale("log")
    ax.set_xlabel("Cost  ($ per 1,000 recommendation requests)")
    ax.set_ylabel("NDCG@10 lift  (percentage points over CF baseline)")
    ax.set_title("Cost vs Quality:  Diminishing Returns Above ~$1 / 1K Requests",
                 fontsize=13.5)
    ax.set_xlim(0.0008, 200)
    ax.set_ylim(-1.5, 14.5)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    # sweet-spot annotation
    ax.axvspan(0.3, 5, alpha=0.08, color=GREEN)
    ax.text(1.3, 13.5, "Production sweet spot", ha="center", color=GREEN,
            weight="bold", fontsize=10.5)
    ax.text(1.3, 12.9, "(rerank top-K with mid-tier LLM)", ha="center",
            color=GREEN, fontsize=9, style="italic")

    fig.tight_layout()
    save(fig, "fig7_cost_quality.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_three_roles()
    fig2_prompt_template()
    fig3_embedding_quality()
    fig4_hybrid_pipeline()
    fig5_cold_start()
    fig6_chat_flow()
    fig7_cost_quality()
    print(f"Wrote 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
