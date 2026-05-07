#!/usr/bin/env python3
"""Regenerate matplotlib figures that have text-overlap bugs.

Fixes:
- fig3_cot_flow.png      (prompt-engineering article)
- fig5_react_flow.png    (ai-agents article)
- fig2_prompt_template.png (llm-recommendation article)
- fig3_tsne_clusters.png (word-embeddings article)

Also uploads each to its EN and ZH OSS path (paths preserved exactly).
"""
import os, sys, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

OSSUTIL = "/root/.aliyun/ossutil"
OSS_BUCKET = "blog-pic-ck"
OSS_AK = os.environ["OSS_AK"]
OSS_SK = os.environ["OSS_SK"]
OSS_ENDPOINT = "oss-cn-beijing.aliyuncs.com"

OUT = Path("/tmp/figaudit/fixed"); OUT.mkdir(parents=True, exist_ok=True)

def upload(local: Path, key: str):
    """key = path under bucket (no leading slash). e.g. posts/en/.../fig.png"""
    oss_url = f"oss://{OSS_BUCKET}/{key}"
    cmd = [OSSUTIL, "cp", "-f", "--meta", "Cache-Control:public, max-age=300, must-revalidate", "--meta", "Cache-Control:public, max-age=300, must-revalidate",
           "-i", OSS_AK, "-k", OSS_SK, "-e", OSS_ENDPOINT,
           "--region", "cn-beijing",
           str(local), oss_url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"  UPLOAD FAIL {key}: rc={r.returncode} stderr={r.stderr[-300:]}")
        return False
    print(f"  uploaded {key}")
    return True

# ====================== FIG 1: CoT flow ======================
def make_cot_flow():
    fig = plt.figure(figsize=(14, 8.5))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.set_title("Chain-of-thought reasoning: explicit steps cut error rates",
                 fontsize=15, pad=12)
    # Top input box
    input_text = ("Problem: A book has 120 pages. On day 1, 30 pages were read.\n"
                  "On day 2, twice as many as day 1 were read.\n"
                  "On day 3, half of the remaining pages were read.\n"
                  "How many pages were read on day 3?")
    ax.add_patch(FancyBboxPatch((4, 70), 92, 19, boxstyle="round,pad=0.4",
                                fc="#dde6f0", ec="#1f3a5f", lw=1.5))
    ax.text(5, 86.5, "INPUT", fontsize=10, fontweight="bold", color="#1f3a5f", va="top")
    ax.text(6, 82, input_text, fontsize=11, family="monospace", va="top")
    # Two side-by-side panels
    # Direct answer (left, red)
    ax.add_patch(FancyBboxPatch((4, 44), 44, 22, boxstyle="round,pad=0.4",
                                fc="#fff0ee", ec="#d64545", lw=2))
    ax.text(6, 63, "Direct answer", fontsize=12, fontweight="bold", color="#b03030")
    ax.text(6, 58, "Prompt:  \"... How many pages on day 3? Answer:\"",
            fontsize=10, family="monospace")
    ax.text(6, 53, "Model output:  \"60\"  (wrong, off by 4x)",
            fontsize=10, family="monospace", color="#a02020")
    # CoT (right, green)
    ax.add_patch(FancyBboxPatch((52, 44), 44, 22, boxstyle="round,pad=0.4",
                                fc="#e8f7ee", ec="#1f8a3a", lw=2))
    ax.text(54, 63, "Chain-of-thought", fontsize=12, fontweight="bold", color="#0e6a2a")
    ax.text(54, 58, "Prompt:  \"... Let’s think step by step.\"",
            fontsize=10, family="monospace")
    ax.text(54, 53, "Model emits intermediate steps → 15  (correct)",
            fontsize=10, family="monospace", color="#0e6a2a")
    # Step boxes row
    steps = [
        ("Step 1", "Day 1: 30"),
        ("Step 2", "2×30 = 60"),
        ("Step 3", "Read so far: 90"),
        ("Step 4", "Left: 120-90 = 30"),
        ("Step 5", "Day 3: 30/2 = 15"),
        ("Answer", "15"),
    ]
    n = len(steps)
    box_w = 13.6
    gap = 1.4
    total_w = n*box_w + (n-1)*gap
    x0 = (100 - total_w) / 2
    y0 = 18; y_top = 36
    for i, (head, body) in enumerate(steps):
        x = x0 + i*(box_w + gap)
        # Last box "Answer" highlighted
        if head == "Answer":
            fc = "#ffe9b3"; ec = "#d18a00"
        else:
            fc = "white"; ec = "#1f8a3a"
        ax.add_patch(FancyBboxPatch((x, y0), box_w, y_top - y0,
                                    boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=2))
        ax.text(x + box_w/2, y0 + (y_top-y0) - 4.5, head, ha="center", va="center",
                fontsize=11, fontweight="bold")
        ax.text(x + box_w/2, y0 + (y_top-y0)/2 - 2.5, body, ha="center", va="center",
                fontsize=9, family="monospace")
        # Arrow to next box
        if i < n-1:
            ax.annotate("", xy=(x + box_w + gap*0.95, y0 + (y_top-y0)/2),
                        xytext=(x + box_w + gap*0.05, y0 + (y_top-y0)/2),
                        arrowprops=dict(arrowstyle="->", color="#1f8a3a", lw=1.5))
    # Footer caption
    ax.text(50, 6, "CoT pushes intermediate state into the output, "
                   "so each step conditions on the prior reasoning.",
            ha="center", fontsize=11, style="italic", color="#404040")
    out = OUT / "fig3_cot_flow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

# ====================== FIG 2: ReAct flow ======================
def make_react_flow():
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.set_title("ReAct Trace: Interleaved Reasoning and Acting",
                 fontsize=15, pad=12)
    # Top question
    ax.add_patch(FancyBboxPatch((4, 86), 92, 9, boxstyle="round,pad=0.3",
                                fc="#22303c", ec="#22303c", lw=1.5))
    ax.text(50, 90.5, "Q: \"Which 2024 Nobel-physics laureate has the highest h-index?\"",
            ha="center", va="center", color="white", fontsize=12)
    # Rows: (label color, label text, content)
    rows = [
        ("#7f3fff", "Thought 1", "I need the laureate list first."),
        ("#3274d9", "Action 1",  "search(\"2024 Nobel physics laureates\")"),
        ("#9aa0a6", "Observation 1", "Hopfield, Hinton"),
        ("#7f3fff", "Thought 2", "Now look up h-indices for each."),
        ("#3274d9", "Action 2",  "scholar_lookup(\"Geoffrey Hinton\")"),
        ("#9aa0a6", "Observation 2", "h-index = 188"),
        ("#1f8a3a", "Final Answer", "Hinton (h-index 188)"),
    ]
    # Layout: leave right margin for the side note, no overlap
    label_x, label_w = 4, 16
    content_x, content_w = 22, 58  # narrowed to keep clear of side annotation
    note_x = 82
    n = len(rows)
    top = 80; bottom = 8
    row_h = (top - bottom) / n
    pad = 1.2
    for i, (col, lbl, content) in enumerate(rows):
        y = top - (i+1)*row_h + pad/2
        h = row_h - pad
        ax.add_patch(FancyBboxPatch((label_x, y), label_w, h,
                                    boxstyle="round,pad=0.2", fc=col, ec=col, lw=0))
        ax.text(label_x + label_w/2, y + h/2, lbl, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        ax.add_patch(FancyBboxPatch((content_x, y), content_w, h,
                                    boxstyle="round,pad=0.2", fc="#f5f7fb", ec=col, lw=1.5))
        ax.text(content_x + content_w/2, y + h/2, content, ha="center", va="center",
                fontsize=11)
        # arrow between consecutive (top -> bottom)
        if i < n-1:
            ax.annotate("", xy=(label_x + label_w/2, y - pad),
                        xytext=(label_x + label_w/2, y),
                        arrowprops=dict(arrowstyle="->", color="#404040", lw=1))
    # Right-side annotation (placed outside content boxes, no overlap)
    ax.text(note_x + 8, top - 2*row_h, "Reason → Act → Observe\nrepeats until ‘Final Answer’",
            ha="center", va="top", fontsize=10, color="#404040", style="italic")
    out = OUT / "fig5_react_flow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

# ====================== FIG 3: Prompt template ======================
def make_prompt_template():
    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    ax.set_title("Prompt Template for LLM-as-Recommender",
                 fontsize=15, pad=4)
    ax.text(50, 92, "Anatomy of a TallRec / P5-style instruction prompt",
            ha="center", fontsize=11, style="italic", color="#555555")
    rows = [
        ("#3274d9", "1. ROLE / TASK", "(role definition)",
         "\"You are a movie recommender. Given the user’s viewing\nhistory, predict whether they will like the target movie.\""),
        ("#7f3fff", "2. USER HISTORY", "(chronological)",
         "History:\n  - Inception (liked)\n  - Interstellar (liked)\n  - The Notebook (disliked)"),
        ("#1f8a3a", "3. CANDIDATE / TARGET", "",
         "Target movie: \"Tenet\"  (Sci-Fi, 2020, dir. Christopher Nolan)"),
        ("#e8a52f", "4. OUTPUT FORMAT", "(constrained decoding)",
         "Answer with exactly one token: \"Yes\" or \"No\".\nOptional: rank-list  [item_1, item_2, ...]"),
        ("#4a5870", "5. FEW-SHOT EXAMPLES", "(optional)",
         "[Example 1] History: ...  Target: ...  Answer: Yes\n[Example 2] History: ...  Target: ...  Answer: No"),
    ]
    n = len(rows)
    top = 86; bottom = 10
    row_h = (top - bottom) / n
    pad = 1.5
    label_x, label_w = 4, 22
    content_x, content_w = 28, 68
    for i, (col, head, sub, content) in enumerate(rows):
        y = top - (i+1)*row_h + pad/2
        h = row_h - pad
        ax.add_patch(FancyBboxPatch((label_x, y), label_w, h,
                                    boxstyle="round,pad=0.2", fc=col, ec=col, lw=0))
        ax.text(label_x + label_w/2, y + h*0.62, head, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        if sub:
            ax.text(label_x + label_w/2, y + h*0.30, sub, ha="center", va="center",
                    color="white", fontsize=9, style="italic")
        ax.add_patch(FancyBboxPatch((content_x, y), content_w, h,
                                    boxstyle="round,pad=0.2", fc="white", ec=col, lw=1.5))
        ax.text(content_x + 1.5, y + h - 1.5, content, ha="left", va="top",
                fontsize=10, family="monospace")
    ax.text(50, 5, "Tip:  short, explicit prompts + few-shot examples + low temperature (≤0.3) give the most stable rankings.",
            ha="center", fontsize=10, color="#555555", style="italic")
    out = OUT / "fig2_prompt_template.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

# ====================== FIG 4: t-SNE clusters ======================
def make_tsne_clusters():
    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_title("t-SNE projection of trained embeddings: semantic neighbours form tight clusters",
                 fontsize=13, pad=10)
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.set_xlim(-7, 7); ax.set_ylim(-5, 5)
    ax.grid(True, alpha=0.3)
    rng = np.random.default_rng(7)
    # Cluster definitions: (center, color, words). Words placed roughly around center but with controlled spacing.
    clusters = [
        ((-3.5, 2.2), "#3274d9", "Animals", ["cat","dog","horse","tiger","wolf","kitten","rabbit"]),
        ((3.2, 2.4), "#7f3fff", "Royalty", ["king","queen","prince","princess","throne"]),
        ((-3.4, -2.4), "#1f8a3a", "Countries", ["Brazil","Germany","France","Spain","Italy","Japan"]),
        ((3.6, -2.0), "#e8a52f", "Tech", ["computer","laptop","data","software","internet","network"]),
    ]
    # Carefully place each label so they do not collide.
    # Use a deterministic offset grid per cluster.
    for (cx, cy), color, name, words in clusters:
        # ellipse background
        ell = mpatches.Ellipse((cx, cy), 4.0, 2.6, fc=color, ec=color, alpha=0.13, lw=1.2, ls="--")
        ax.add_patch(ell)
        ax.text(cx, cy + 1.7, name, ha="center", color=color, fontsize=13, fontweight="bold")
        # Place words on a non-overlapping grid inside ellipse
        n = len(words)
        # rows of 2-3 columns
        cols = 2 if n <= 4 else 3
        rows = (n + cols - 1) // cols
        col_w = 2.8 / cols
        row_h = 1.4 / max(rows, 1)
        for k, w in enumerate(words):
            r = k // cols; c = k % cols
            # alternate row offset to avoid stacking
            x = cx - 1.4 + (c + 0.5) * col_w + (0.15 if r % 2 else -0.15)
            y = cy + 0.4 - (r + 0.5) * row_h
            ax.scatter([x], [y], c=color, s=55, edgecolor="white", linewidth=1.2, zorder=3)
            ax.text(x + 0.18, y + 0.05, w, fontsize=10, va="center", ha="left", zorder=4)
    plt.tight_layout()
    out = OUT / "fig3_tsne_clusters.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out

# ====================== Main ======================
TARGETS = {
    "fig3_cot_flow.png": [
        "posts/en/standalone/prompt-engineering-complete-guide/fig3_cot_flow.png",
        "posts/zh/standalone/提示词工程完全指南-从零基础到高级优化/fig3_cot_flow.png",
        "posts/zh/nlp/07-提示工程与In-Context-Learning/fig3_cot_flow.png",
    ],
    "fig5_react_flow.png": [
        "posts/en/standalone/ai-agents-complete-guide/fig5_react_flow.png",
        "posts/zh/standalone/ai-agent完全指南-从理论到工业实践/fig5_react_flow.png",
    ],
    "fig2_prompt_template.png": [
        "posts/en/recommendation-systems/12-llm-recommendation/fig2_prompt_template.png",
        "posts/zh/recommendation-systems/12-大语言模型与推荐系统/fig2_prompt_template.png",
    ],
    "fig3_tsne_clusters.png": [
        "posts/en/nlp/word-embeddings-lm/fig3_tsne_clusters.png",
        "posts/zh/nlp/02-词向量与语言模型/fig3_tsne_clusters.png",
    ],
}

GEN = {
    "fig3_cot_flow.png": make_cot_flow,
    "fig5_react_flow.png": make_react_flow,
    "fig2_prompt_template.png": make_prompt_template,
    "fig3_tsne_clusters.png": make_tsne_clusters,
}

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, gen in GEN.items():
        if only and name != only: continue
        print(f"== generate {name} ==")
        local = gen()
        print(f"  local: {local} ({local.stat().st_size} bytes)")
        for key in TARGETS[name]:
            upload(local, key)

if __name__ == "__main__":
    main()
