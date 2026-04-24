"""
Figure generation script for the LLMGR standalone paper review.

Generates 5 self-contained figures used in both the EN and ZH versions of
the article. Each figure isolates exactly one teaching point.

Figures:
    fig1_framework               LLMGR framework. Two-stream architecture
                                 fusing an LLM "semantic engine" with a GNN
                                 session-graph encoder via a hybrid encoding
                                 layer; a ranking head produces next-item
                                 scores.
    fig2_multitask_prompts       Multi-task prompt design. Side-by-side
                                 visualisation of (a) the auxiliary node-text
                                 alignment prompt and (b) the main behaviour
                                 prediction prompt that share the same model.
    fig3_hybrid_encoding         Hybrid encoding layer. ID embeddings (low
                                 dim) projected through a learnable W_p into
                                 the LLM hidden space (high dim) and
                                 concatenated with text token embeddings.
    fig4_two_stage_tuning        Two-stage prompt tuning timeline. Stage 1
                                 freezes GNN, learns semantic anchoring (1
                                 epoch); Stage 2 unfreezes GNN, learns
                                 behaviour patterns (3 epochs).
    fig5_coldstart_perf          Cold-start gain. Bar groups across
                                 Music/Beauty/Pantry comparing baseline (best
                                 GNN) vs LLMGR in warm-start vs cold-start
                                 buckets, highlighting the larger uplift on
                                 cold items.

Style:
    matplotlib seaborn-v0_8-whitegrid, dpi=150.
    Palette: #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (amber).

Usage:
    python3 scripts/figures/standalone/llmgr.py

Output:
    Writes PNGs into BOTH the EN and ZH asset folders so the markdown image
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

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_GRAY = "#94a3b8"
C_LIGHT = "#e2e8f0"
C_DARK = "#0f172a"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "llmgr"
ZH_DIR = (
    REPO_ROOT
    / "source" / "_posts" / "zh" / "standalone"
    / "integrating-large-language-models-with-graphical-session-bas"
)


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, xy, w, h, text, fc, ec=None, fontsize=10,
         fontweight="normal", text_color="white", alpha=1.0,
         rounding=0.05):
    ec = ec or fc
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=1.2, facecolor=fc, edgecolor=ec, alpha=alpha,
    )
    ax.add_patch(box)
    if text:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
                ha="center", va="center",
                color=text_color, fontsize=fontsize,
                fontweight=fontweight)


def _arrow(ax, xy_from, xy_to, color=C_DARK, lw=1.4,
           style="-|>", connection="arc3,rad=0"):
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw, connectionstyle=connection,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: LLMGR framework overview
# ---------------------------------------------------------------------------
def fig1_framework() -> None:
    """LLM (semantic engine) + GNN (graph encoder) fused for SBR ranking."""
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(6, 6.55,
            "LLMGR framework: LLM as semantic engine, GNN as ranker",
            ha="center", va="center", fontsize=14,
            fontweight="bold", color=C_DARK)

    # ---------- inputs ----------
    # Session click stream (left top)
    _box(ax, (0.3, 5.0), 3.4, 0.8,
         "Session  s = [v1, v3, v2, v3, v4]",
         C_DARK, fontsize=10.5, fontweight="bold", rounding=0.12)

    # Item text catalogue (left bottom)
    _box(ax, (0.3, 1.0), 3.4, 0.8,
         "Item text: title / description / attributes",
         C_DARK, fontsize=10.5, fontweight="bold", rounding=0.12)

    # ---------- top branch: GNN over session graph ----------
    _arrow(ax, (3.7, 5.4), (4.4, 5.4), color=C_BLUE, lw=1.6)
    _box(ax, (4.4, 4.7), 2.6, 1.4,
         "GNN encoder\n(message passing)",
         C_BLUE, fontsize=10.5, fontweight="bold", rounding=0.10)
    ax.text(5.7, 4.55, "node embeddings  $x_v \\in \\mathbb{R}^{d}$",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    # ---------- bottom branch: LLM tokeniser over text ----------
    _arrow(ax, (3.7, 1.4), (4.4, 1.4), color=C_PURPLE, lw=1.6)
    _box(ax, (4.4, 0.7), 2.6, 1.4,
         "LLM tokenizer\n+ word embedding",
         C_PURPLE, fontsize=10.5, fontweight="bold", rounding=0.10)
    ax.text(5.7, 0.55, "text embeddings  $e_t \\in \\mathbb{R}^{D}$",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    # ---------- hybrid encoding layer (centre) ----------
    _arrow(ax, (7.0, 5.4), (7.7, 4.0), color=C_BLUE, lw=1.6,
           connection="arc3,rad=-0.2")
    _arrow(ax, (7.0, 1.4), (7.7, 3.0), color=C_PURPLE, lw=1.6,
           connection="arc3,rad=0.2")

    _box(ax, (7.7, 2.6), 1.9, 1.8,
         "Hybrid\nencoding\nlayer\n$W_p$: d -> D",
         C_GREEN, fontsize=10.5, fontweight="bold", rounding=0.10)

    # ---------- LLM core ----------
    _arrow(ax, (9.6, 3.5), (10.2, 3.5), color=C_GREEN, lw=1.8)
    _box(ax, (10.2, 2.6), 1.6, 1.8,
         "LLaMA2-7B\n(LoRA)",
         C_DARK, fontsize=10.5, fontweight="bold", rounding=0.10)

    # ---------- ranking head ----------
    _arrow(ax, (11.0, 2.6), (11.0, 1.85), color=C_DARK, lw=1.6)
    _box(ax, (10.0, 1.05), 1.95, 0.8,
         "MLP ranking head",
         C_AMBER, fontsize=10, fontweight="bold", rounding=0.18)
    _arrow(ax, (11.0, 1.05), (11.0, 0.55), color=C_AMBER, lw=1.6)
    ax.text(11.0, 0.30, "p(v_{n+1} | s)",
            ha="center", fontsize=10, color=C_DARK,
            fontweight="bold")

    # ---------- caption strip ----------
    ax.text(6, -0.1,
            "LLM extracts text semantics; GNN learns transition structure; "
            "hybrid layer projects $x_v$ into the LLM space "
            "so both streams share one representation.",
            ha="center", fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    fig.suptitle("Figure 1. LLMGR end-to-end architecture",
                 fontsize=12.5, color=C_DARK, y=0.01)
    _save(fig, "fig1_framework")


# ---------------------------------------------------------------------------
# Figure 2: Multi-task prompt design
# ---------------------------------------------------------------------------
def fig2_multitask_prompts() -> None:
    """Auxiliary node-text alignment + main behaviour prediction prompts."""
    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(6, 5.85,
            "Multi-task prompts: same model, two supervision signals",
            ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=C_DARK)

    # ---------- left: auxiliary task ----------
    _box(ax, (0.3, 0.6), 5.4, 4.7, "",
         C_PURPLE, alpha=0.08, ec=C_PURPLE, rounding=0.08)
    ax.text(3.0, 4.95,
            "Auxiliary task  -  node <-> text alignment",
            ha="center", fontsize=11.5, color=C_PURPLE,
            fontweight="bold")
    ax.text(3.0, 4.55, "(semantic grounding, Stage 1)",
            ha="center", fontsize=9.5, color=C_PURPLE, style="italic")

    aux_lines = [
        "Prompt:",
        "  Candidate items: { v1, v2, v3, v4, v5 }",
        "  Description:",
        "    \"Seagull Pro-G Guitar Stand, Black,",
        "     foldable A-frame, fits acoustic guitars\"",
        "  Question:",
        "    Which item ID does this text describe?",
        "",
        "Target:  v3",
    ]
    for i, ln in enumerate(aux_lines):
        ax.text(0.55, 3.95 - i * 0.34, ln,
                fontsize=9.2, color=C_DARK,
                family="monospace")

    # forces "anchor text -> ID" mapping
    ax.text(3.0, 0.85,
            "Forces: text semantics anchored to item IDs",
            ha="center", fontsize=9.5, color=C_PURPLE,
            fontweight="bold")

    # ---------- right: main task ----------
    _box(ax, (6.3, 0.6), 5.4, 4.7, "",
         C_BLUE, alpha=0.08, ec=C_BLUE, rounding=0.08)
    ax.text(9.0, 4.95,
            "Main task  -  next-item prediction",
            ha="center", fontsize=11.5, color=C_BLUE,
            fontweight="bold")
    ax.text(9.0, 4.55, "(behaviour pattern, Stage 2)",
            ha="center", fontsize=9.5, color=C_BLUE, style="italic")

    main_lines = [
        "Prompt:",
        "  Session graph nodes: { v1, v3, v2, v4 }",
        "  Edges: v1->v3, v3->v2, v2->v3, v3->v4",
        "  Last clicked: v4",
        "  Candidate set: { v5, v6, ..., v_K }",
        "  Question:",
        "    Predict the next item the user clicks.",
        "",
        "Target:  v6",
    ]
    for i, ln in enumerate(main_lines):
        ax.text(6.55, 3.95 - i * 0.34, ln,
                fontsize=9.2, color=C_DARK,
                family="monospace")

    ax.text(9.0, 0.85,
            "Forces: structural transitions -> ranking signal",
            ha="center", fontsize=9.5, color=C_BLUE,
            fontweight="bold")

    # ---------- shared model arrow ----------
    _arrow(ax, (5.7, 2.9), (6.3, 2.9), color=C_GREEN, lw=1.5,
           style="<|-|>")
    ax.text(6.0, 3.15, "shared LLM\n+ hybrid layer",
            ha="center", fontsize=8.8, color=C_GREEN,
            fontweight="bold")

    fig.suptitle(
        "Figure 2. Two prompt families: auxiliary anchors text to IDs, "
        "main learns next-item ranking",
        fontsize=12, color=C_DARK, y=0.01)
    _save(fig, "fig2_multitask_prompts")


# ---------------------------------------------------------------------------
# Figure 3: Hybrid encoding layer
# ---------------------------------------------------------------------------
def fig3_hybrid_encoding() -> None:
    """Show projection W_p: R^d -> R^D and concat with text embeddings."""
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    ax.text(6, 5.8,
            "Hybrid encoding: project ID embeddings into LLM hidden space, "
            "then concat",
            ha="center", va="center", fontsize=13,
            fontweight="bold", color=C_DARK)

    # ---------- left: ID embeddings (small slabs) ----------
    ax.text(1.6, 5.05, "GNN ID embeddings  $x_v$",
            ha="center", fontsize=10.5, color=C_BLUE,
            fontweight="bold")
    ax.text(1.6, 4.7, "$d = 64$",
            ha="center", fontsize=9, color=C_DARK, style="italic")

    n_ids = 4
    for i in range(n_ids):
        y = 4.0 - i * 0.7
        # short slab => small dim
        _box(ax, (1.0, y - 0.18), 1.2, 0.36,
             f"$v_{{{i+1}}}$", C_BLUE, alpha=0.85,
             fontsize=10.5, fontweight="bold", rounding=0.08)

    # ---------- W_p projection block ----------
    _arrow(ax, (2.3, 2.5), (3.5, 2.5), color=C_BLUE, lw=1.6)
    _box(ax, (3.5, 1.8), 1.6, 1.4,
         "$W_p$\nlinear\n$d \\to D$",
         C_GREEN, fontsize=11, fontweight="bold", rounding=0.08)
    _arrow(ax, (5.1, 2.5), (6.3, 2.5), color=C_GREEN, lw=1.6)

    # ---------- projected ID embeddings (long slabs) ----------
    ax.text(7.1, 5.05, "projected: $W_p\\, x_v \\in \\mathbb{R}^{D}$",
            ha="center", fontsize=10.5, color=C_GREEN,
            fontweight="bold")
    for i in range(n_ids):
        y = 4.0 - i * 0.7
        _box(ax, (6.3, y - 0.18), 3.6, 0.36,
             f"$v_{{{i+1}}}$", C_GREEN, alpha=0.85,
             fontsize=10.5, fontweight="bold", rounding=0.04)

    # ---------- text embeddings (long slabs, concat below) ----------
    ax.text(7.1, 1.05, "LLM text token embeddings  $e_t$",
            ha="center", fontsize=10.5, color=C_PURPLE,
            fontweight="bold")
    for i in range(3):
        x0 = 6.3 + i * 1.25
        _box(ax, (x0, 0.45), 1.15, 0.36,
             ["The", "guitar", "stand"][i], C_PURPLE, alpha=0.85,
             fontsize=9.5, fontweight="bold", rounding=0.04)

    # ---------- concat note ----------
    ax.text(11.0, 2.5,
            "concat\n[ $W_p x_v$ ; $e_t$ ]\nfed to LLM",
            ha="center", va="center", fontsize=10,
            color=C_DARK, fontweight="bold",
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.45"))
    _arrow(ax, (9.95, 2.5), (10.4, 2.5), color=C_DARK, lw=1.4)

    # ---------- caption ----------
    ax.text(6, -0.1,
            "$W_p$ is the *only* parameter bridging the two spaces -- "
            "all GNN/LLM weights stay native; "
            "projected IDs share LLM hidden dim $D = 4096$ for LLaMA2-7B.",
            ha="center", fontsize=9.5, color=C_DARK,
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT,
                      boxstyle="round,pad=0.4"))

    fig.suptitle("Figure 3. Hybrid encoding layer aligns ID and text spaces",
                 fontsize=12, color=C_DARK, y=0.01)
    _save(fig, "fig3_hybrid_encoding")


# ---------------------------------------------------------------------------
# Figure 4: Two-stage prompt tuning timeline
# ---------------------------------------------------------------------------
def fig4_two_stage_tuning() -> None:
    """Stage 1 freezes GNN (1 epoch); Stage 2 unfreezes (3 epochs)."""
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.text(6, 6.65,
            "Two-stage prompt tuning: align first, then learn behaviour",
            ha="center", va="center", fontsize=13.5,
            fontweight="bold", color=C_DARK)

    # ---------- timeline backbone ----------
    ax.plot([0.6, 11.4], [3.0, 3.0], color=C_GRAY, lw=2, zorder=1)

    # ---------- Stage 1 ----------
    _box(ax, (0.6, 3.4), 4.9, 2.3, "",
         C_PURPLE, alpha=0.10, ec=C_PURPLE, rounding=0.06)
    ax.text(3.05, 5.45, "Stage 1  -  semantic grounding",
            ha="center", fontsize=11.5, color=C_PURPLE,
            fontweight="bold")
    ax.text(3.05, 5.10, "(auxiliary node-text alignment, 1 epoch)",
            ha="center", fontsize=9.2, color=C_PURPLE,
            style="italic")
    ax.text(3.05, 4.50,
            "- freeze GNN (no shortcut via edges)\n"
            "- train hybrid layer + LLM (LoRA)\n"
            "- loss: CE on \"which ID is this text?\"",
            ha="center", fontsize=9, color=C_DARK,
            linespacing=1.5)

    # ---------- Stage 2 ----------
    _box(ax, (6.5, 3.4), 4.9, 2.3, "",
         C_BLUE, alpha=0.10, ec=C_BLUE, rounding=0.06)
    ax.text(8.95, 5.45, "Stage 2  -  behaviour pattern learning",
            ha="center", fontsize=11.5, color=C_BLUE,
            fontweight="bold")
    ax.text(8.95, 5.10, "(main next-item prediction, ~3 epochs / dataset)",
            ha="center", fontsize=9.2, color=C_BLUE,
            style="italic")
    ax.text(8.95, 4.50,
            "- unfreeze GNN (let it adapt to ranking)\n"
            "- keep semantic anchor from Stage 1\n"
            "- loss: CE on next-item one-hot",
            ha="center", fontsize=9, color=C_DARK,
            linespacing=1.5)

    # ---------- timeline markers ----------
    for x, lab in [(0.8, "start"), (5.4, "switch"), (11.2, "deploy")]:
        ax.scatter(x, 3.0, s=60, color=C_DARK, zorder=2)
        ax.text(x, 2.7, lab, ha="center", fontsize=9,
                color=C_DARK, fontweight="bold")

    # ---------- failure mode arrow (what one-stage joint training does) ----------
    _box(ax, (1.4, 0.55), 9.2, 1.85, "",
         C_AMBER, alpha=0.10, ec=C_AMBER, rounding=0.06)
    ax.text(6, 2.10,
            "Why split?  Joint training collapses to ID + edges:",
            ha="center", fontsize=10.5, color=C_AMBER,
            fontweight="bold")
    ax.text(6, 1.65,
            "the model has not yet learned which text -> which ID, "
            "so behaviour noise dominates the gradient",
            ha="center", fontsize=9.5, color=C_DARK)
    ax.text(6, 1.20,
            "=>  no different from a vanilla GNN-SBR; cold-start gain "
            "disappears.",
            ha="center", fontsize=9.5, color=C_DARK)
    ax.text(6, 0.78,
            "Ablation: removing Stage 1 drops NDCG@20 by ~4.16% on Beauty.",
            ha="center", fontsize=9.5, color=C_DARK,
            fontweight="bold")

    fig.suptitle("Figure 4. Two-stage prompt tuning timeline and the trap "
                 "it avoids",
                 fontsize=12, color=C_DARK, y=0.01)
    _save(fig, "fig4_two_stage_tuning")


# ---------------------------------------------------------------------------
# Figure 5: Cold-start performance
# ---------------------------------------------------------------------------
def fig5_coldstart_perf() -> None:
    """Warm vs cold buckets across 3 datasets, baseline vs LLMGR."""
    fig, ax = plt.subplots(figsize=(11.5, 5.8))

    # Indicative numbers in the spirit of the LLMGR paper:
    # - HR@20 reported uplift ~8.68% overall over the strongest baseline
    # - cold-start uplift is consistently larger than warm-start uplift
    # Absolute numbers are illustrative; the gap pattern matches the paper.
    datasets = ["Music", "Beauty", "Pantry"]
    buckets = ["Warm-start (>=50 inter.)", "Cold-start (5-10 inter.)"]

    # rows: dataset; cols: (baseline_warm, llmgr_warm, baseline_cold, llmgr_cold)
    base_warm = np.array([0.225, 0.183, 0.158])
    llmgr_warm = base_warm * np.array([1.05, 1.06, 1.05])
    base_cold = np.array([0.118, 0.092, 0.074])
    llmgr_cold = base_cold * np.array([1.18, 1.21, 1.19])

    x = np.arange(len(datasets))
    w = 0.18
    offs = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    bars_a = ax.bar(x + offs[0], base_warm, width=w,
                    color=C_GRAY, edgecolor=C_DARK, lw=0.7,
                    label="Best GNN baseline  (warm)")
    bars_b = ax.bar(x + offs[1], llmgr_warm, width=w,
                    color=C_BLUE, edgecolor=C_DARK, lw=0.7,
                    label="LLMGR  (warm)")
    bars_c = ax.bar(x + offs[2], base_cold, width=w,
                    color=C_LIGHT, edgecolor=C_DARK, lw=0.7,
                    label="Best GNN baseline  (cold)")
    bars_d = ax.bar(x + offs[3], llmgr_cold, width=w,
                    color=C_AMBER, edgecolor=C_DARK, lw=0.7,
                    label="LLMGR  (cold)")

    # delta annotations
    for xi, bw, lw_v in zip(x, base_warm, llmgr_warm):
        d = (lw_v - bw) / bw * 100
        ax.text(xi + offs[1], lw_v + 0.005, f"+{d:.1f}%",
                ha="center", fontsize=8.5, color=C_BLUE,
                fontweight="bold")
    for xi, bc, lc in zip(x, base_cold, llmgr_cold):
        d = (lc - bc) / bc * 100
        ax.text(xi + offs[3], lc + 0.005, f"+{d:.1f}%",
                ha="center", fontsize=8.5, color=C_AMBER,
                fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("HR@20", fontsize=11)
    ax.set_ylim(0, max(llmgr_warm) * 1.22)
    ax.set_title("Figure 5. LLMGR uplift is largest on cold-start items "
                 "(text semantics rescue sparse buckets)",
                 fontsize=12, color=C_DARK, pad=12)
    ax.legend(loc="upper right", frameon=True, fontsize=9.5, ncol=2)
    ax.tick_params(axis="y", labelsize=9.5)

    ax.text(0.0, -0.18,
            "Indicative HR@20 on Amazon Music/Beauty/Pantry. "
            "Warm-start uplift ~5-6%; cold-start uplift ~18-21%. "
            "Pattern reproduces the paper's RQ4 finding that "
            "gains concentrate on sparse items.",
            transform=ax.transAxes, fontsize=8.8, color=C_GRAY)

    fig.tight_layout()
    _save(fig, "fig5_coldstart_perf")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating LLMGR figures into:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")
    fig1_framework()
    fig2_multitask_prompts()
    fig3_hybrid_encoding()
    fig4_two_stage_tuning()
    fig5_coldstart_perf()
    print("Done.")


if __name__ == "__main__":
    main()
