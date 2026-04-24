"""
Figure generation for LeetCode Patterns 01: Hash Tables.

Generates 5 figures used in both EN and ZH versions of the article.
Each figure teaches a single concrete idea in clean editorial print style.

Figures:
    fig1_hash_function          Visualises a hash function: keys flow through
                                hash(k) % N to bucket indices in an array.
    fig2_collision_resolution   Side-by-side: separate chaining (linked list
                                per bucket) vs open addressing (linear probe).
    fig3_two_sum_flow           Step-by-step trace of one-pass Two Sum on
                                nums = [2, 7, 11, 15], target = 9.
    fig4_complexity_compare     Operation cost (lookup) vs n for hash table
                                vs sorted-array binary search vs linear scan.
    fig5_set_vs_map_decision    Decision tree: when to use hash set vs hash
                                map vs plain array.

Usage:
    python3 scripts/figures/leetcode/01-hash-tables.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"
C_PURPLE = "#7c3aed"
C_GREEN = "#10b981"
C_AMBER = "#f59e0b"
C_RED = "#dc2626"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"
C_BG = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "leetcode" / "hash-tables"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "leetcode" / "01-哈希表"


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        fig.savefig(d / f"{name}.png", dpi=DPI, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)


def _rounded_box(ax, x, y, w, h, *, fc, ec=C_DARK, lw=1.2, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                         linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha)
    ax.add_patch(box)
    return box


# ---------------------------------------------------------------------------
# Fig 1 — Hash function: key -> bucket index
# ---------------------------------------------------------------------------
def fig1_hash_function() -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Title
    ax.text(5.5, 5.6, "Hash Function: key  →  hash(key) % N  →  bucket index",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DARK)

    # Keys (left)
    keys = [("\"alice\"", C_BLUE),
            ("\"bob\"",   C_PURPLE),
            ("42",        C_GREEN),
            ("\"carol\"", C_AMBER)]
    key_x, key_w, key_h = 0.3, 1.6, 0.7
    for i, (k, c) in enumerate(keys):
        y = 4.4 - i * 1.0
        _rounded_box(ax, key_x, y, key_w, key_h, fc=c, ec=c, alpha=0.18)
        ax.text(key_x + key_w / 2, y + key_h / 2, k,
                ha="center", va="center", fontsize=11,
                color=C_DARK, fontweight="bold")

    # Hash function box (center)
    hf_x, hf_w = 3.6, 2.2
    _rounded_box(ax, hf_x, 1.6, hf_w, 2.6, fc=C_BG, ec=C_DARK, lw=1.6)
    ax.text(hf_x + hf_w / 2, 3.4, "hash()", ha="center", va="center",
            fontsize=14, fontweight="bold", color=C_DARK)
    ax.text(hf_x + hf_w / 2, 2.8, "mix bits\nof key",
            ha="center", va="center", fontsize=9, color=C_GRAY,
            style="italic")
    ax.text(hf_x + hf_w / 2, 2.05, "% N",
            ha="center", va="center", fontsize=11,
            color=C_BLUE, fontweight="bold")

    # Buckets (right) — array of slots
    n_buckets = 7
    b_x0 = 7.3
    b_w = 1.2
    b_h = 0.55
    b_gap = 0.05
    bucket_y_top = 4.7
    bucket_centers = []
    for i in range(n_buckets):
        y = bucket_y_top - i * (b_h + b_gap)
        rect = Rectangle((b_x0, y), b_w, b_h, facecolor="white",
                         edgecolor=C_DARK, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(b_x0 - 0.25, y + b_h / 2, f"[{i}]",
                ha="right", va="center", fontsize=9, color=C_GRAY)
        bucket_centers.append((b_x0 + b_w / 2, y + b_h / 2))

    # Map keys to buckets (deterministic illustrative mapping)
    target_buckets = [2, 5, 0, 5]   # carol collides with bob → bucket 5
    fill_colors = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]
    labels_for_bucket = [[] for _ in range(n_buckets)]
    for (k, _), tb, fc in zip(keys, target_buckets, fill_colors):
        labels_for_bucket[tb].append((k, fc))

    # Draw arrows: key -> hash box, hash box -> bucket
    for i, (k, c) in enumerate(keys):
        y_src = 4.4 - i * 1.0 + 0.35
        # Key -> hash
        ax.add_patch(FancyArrowPatch((key_x + key_w + 0.05, y_src),
                                     (hf_x - 0.05, 2.9),
                                     arrowstyle="->", mutation_scale=12,
                                     color=C_GRAY, lw=1.0))
        # hash -> bucket
        bx, by = bucket_centers[target_buckets[i]]
        ax.add_patch(FancyArrowPatch((hf_x + hf_w + 0.05, 2.9),
                                     (bx - b_w / 2 - 0.02, by),
                                     arrowstyle="->", mutation_scale=12,
                                     color=c, lw=1.4))

    # Render values inside buckets
    for idx, items in enumerate(labels_for_bucket):
        if not items:
            continue
        bx, by = bucket_centers[idx]
        if len(items) == 1:
            k, fc = items[0]
            ax.text(bx, by, k, ha="center", va="center",
                    fontsize=9, color=fc, fontweight="bold")
        else:
            # Collision: stack labels horizontally
            txt = "  +  ".join(k for k, _ in items)
            ax.text(bx, by, txt, ha="center", va="center",
                    fontsize=8, color=C_RED, fontweight="bold")
            ax.text(b_x0 + b_w + 0.1, by, "← collision",
                    ha="left", va="center", fontsize=8.5,
                    color=C_RED, style="italic")

    # Legend / takeaway
    ax.text(5.5, 0.55,
            "Equal keys always land in the same bucket.  Different keys "
            "may collide — that is what the resolution strategy handles.",
            ha="center", va="center", fontsize=10, color=C_DARK,
            style="italic")

    fig.tight_layout()
    _save(fig, "fig1_hash_function")


# ---------------------------------------------------------------------------
# Fig 2 — Collision resolution: chaining vs open addressing
# ---------------------------------------------------------------------------
def fig2_collision_resolution() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # ---- LEFT: separate chaining --------------------------------------------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(5, 6.55, "Separate Chaining",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DARK)
    ax.text(5, 6.05, "(Java HashMap, C++ unordered_map)",
            ha="center", va="center", fontsize=9.5, color=C_GRAY,
            style="italic")

    n = 5
    bx0, bw, bh = 0.6, 1.4, 0.7
    chains = {
        0: [],
        1: [("\"cat\"", C_BLUE), ("\"act\"", C_PURPLE)],   # collision
        2: [("\"dog\"", C_GREEN)],
        3: [],
        4: [("\"owl\"", C_AMBER), ("\"low\"", C_RED), ("\"wol\"", C_BLUE)],
    }
    for i in range(n):
        y = 5.0 - i * 0.95
        rect = Rectangle((bx0, y), bw, bh, facecolor=C_BG,
                         edgecolor=C_DARK, linewidth=1.1)
        ax.add_patch(rect)
        ax.text(bx0 - 0.1, y + bh / 2, f"[{i}]",
                ha="right", va="center", fontsize=9, color=C_GRAY)

        items = chains[i]
        x_cur = bx0 + bw + 0.25
        for j, (lab, col) in enumerate(items):
            node_w = 1.05
            _rounded_box(ax, x_cur, y, node_w, bh, fc=col, ec=col, alpha=0.18)
            ax.text(x_cur + node_w / 2, y + bh / 2, lab,
                    ha="center", va="center", fontsize=9.5,
                    color=col, fontweight="bold")
            # arrow connecting bucket / previous node to this one
            src_x = bx0 + bw if j == 0 else (x_cur - 0.18)
            ax.add_patch(FancyArrowPatch((src_x, y + bh / 2),
                                         (x_cur - 0.02, y + bh / 2),
                                         arrowstyle="->", mutation_scale=10,
                                         color=C_DARK, lw=0.9))
            x_cur += node_w + 0.25

    ax.text(5, 0.4,
            "Each bucket owns a small linked list.\nCollisions append; "
            "lookup walks the chain.",
            ha="center", va="center", fontsize=10, color=C_DARK)

    # ---- RIGHT: open addressing (linear probing) ----------------------------
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(5, 6.55, "Open Addressing (linear probing)",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DARK)
    ax.text(5, 6.05, "(Python dict, Go map)",
            ha="center", va="center", fontsize=9.5, color=C_GRAY,
            style="italic")

    n = 8
    bx0, bw, bh = 1.6, 0.95, 0.7
    # Slot occupants after inserts: cat→1, act→1 (probe to 2), dog→3, owl→3 (probe to 4)
    slots = [None,
             ("\"cat\"", C_BLUE),
             ("\"act\"", C_PURPLE),
             ("\"dog\"", C_GREEN),
             ("\"owl\"", C_AMBER),
             None, None, None]
    for i in range(n):
        x = bx0 + i * (bw + 0.05)
        y = 3.6
        fc = C_BG if slots[i] is None else "white"
        rect = Rectangle((x, y), bw, bh, facecolor=fc,
                         edgecolor=C_DARK, linewidth=1.1)
        ax.add_patch(rect)
        ax.text(x + bw / 2, y - 0.25, f"[{i}]",
                ha="center", va="top", fontsize=9, color=C_GRAY)
        if slots[i] is not None:
            lab, col = slots[i]
            ax.text(x + bw / 2, y + bh / 2, lab,
                    ha="center", va="center", fontsize=8.5,
                    color=col, fontweight="bold")

    # Annotate "act" collision and probe
    cat_x = bx0 + 1 * (bw + 0.05) + bw / 2
    act_x = bx0 + 2 * (bw + 0.05) + bw / 2
    ax.text(cat_x, 5.4, "hash(\"act\") = 1\nslot 1 taken!",
            ha="center", va="center", fontsize=8.5, color=C_RED,
            style="italic")
    ax.add_patch(FancyArrowPatch((cat_x + 0.05, 5.0), (act_x, 4.4),
                                 arrowstyle="->", mutation_scale=12,
                                 color=C_RED, lw=1.2,
                                 connectionstyle="arc3,rad=-0.35"))
    ax.text(act_x + 0.55, 4.85, "probe →",
            ha="left", va="center", fontsize=9, color=C_RED,
            fontweight="bold")

    ax.text(5, 1.7,
            "On collision, scan the next slot until an empty one appears.\n"
            "Lookup re-runs the same probe sequence.",
            ha="center", va="center", fontsize=10, color=C_DARK)

    fig.suptitle("Two Common Collision-Resolution Strategies",
                 fontsize=15, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_collision_resolution")


# ---------------------------------------------------------------------------
# Fig 3 — Two Sum: one-pass hash table trace
# ---------------------------------------------------------------------------
def fig3_two_sum_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    nums = [2, 7, 11, 15]
    target = 9

    ax.text(6, 6.6, "Two Sum — one pass with a hash table",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DARK)
    ax.text(6, 6.15, f"nums = {nums},  target = {target}",
            ha="center", va="center", fontsize=11, color=C_GRAY,
            family="monospace")

    # Array row at the top
    cell_w = 1.0
    array_x0 = 4.0
    array_y = 5.0
    for i, v in enumerate(nums):
        x = array_x0 + i * cell_w
        rect = Rectangle((x, array_y), cell_w, 0.6,
                         facecolor=C_BG, edgecolor=C_DARK, linewidth=1.0)
        ax.add_patch(rect)
        ax.text(x + cell_w / 2, array_y + 0.3, str(v),
                ha="center", va="center", fontsize=11,
                color=C_DARK, fontweight="bold", family="monospace")
        ax.text(x + cell_w / 2, array_y - 0.18, f"i={i}",
                ha="center", va="top", fontsize=9, color=C_GRAY)

    # Step rows
    rows = [
        # (i, num, complement, seen_state, action_text, action_color)
        (0, 2, 7, "{}", "miss → store seen[2] = 0", C_GRAY),
        (1, 7, 2, "{2: 0}",
         "hit!  seen[2] = 0  →  return [0, 1]", C_GREEN),
    ]

    header_y = 4.0
    headers = ["i", "num", "complement\n= target − num",
               "seen (before)", "action"]
    col_x = [0.4, 1.5, 2.7, 5.4, 7.7]
    col_w = [1.0, 1.0, 2.5, 2.1, 4.2]
    for x, w, h in zip(col_x, col_w, headers):
        _rounded_box(ax, x, header_y, w, 0.55, fc=C_DARK, ec=C_DARK)
        ax.text(x + w / 2, header_y + 0.27, h,
                ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold")

    for ridx, (i, num, comp, seen, act, ac) in enumerate(rows):
        y = header_y - 0.85 - ridx * 0.85
        cells = [str(i), str(num), f"{target} − {num} = {comp}", seen, act]
        for cidx, (x, w, txt) in enumerate(zip(col_x, col_w, cells)):
            fc = "white"
            ec = C_LIGHT
            _rounded_box(ax, x, y, w, 0.7, fc=fc, ec=ec, lw=0.8)
            color = ac if cidx == 4 else C_DARK
            weight = "bold" if cidx == 4 else "normal"
            ax.text(x + w / 2, y + 0.35, txt, ha="center", va="center",
                    fontsize=9.5, color=color, fontweight=weight,
                    family="monospace")

        # Highlight current array cell
        cur_x = array_x0 + i * cell_w
        marker = Rectangle((cur_x, array_y), cell_w, 0.6, fill=False,
                           edgecolor=ac, linewidth=2.4)
        ax.add_patch(marker)

    ax.text(6, 0.6,
            "Insight: every element only asks one question — "
            "\"have I seen target − num before?\"\n"
            "Answering it in O(1) turns an O(n²) scan into a single O(n) pass.",
            ha="center", va="center", fontsize=10, color=C_DARK,
            style="italic")

    fig.tight_layout()
    _save(fig, "fig3_two_sum_flow")


# ---------------------------------------------------------------------------
# Fig 4 — Lookup cost: hash table vs sorted array vs linear scan
# ---------------------------------------------------------------------------
def fig4_complexity_compare() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8))

    n = np.linspace(1, 1000, 400)
    linear = n
    binsearch = np.log2(n + 1)
    hashed = np.ones_like(n)

    ax.plot(n, linear, color=C_RED, lw=2.4,
            label="Linear scan in unsorted array — O(n)")
    ax.plot(n, binsearch, color=C_AMBER, lw=2.4,
            label="Binary search in sorted array — O(log n)")
    ax.plot(n, hashed, color=C_BLUE, lw=2.6,
            label="Hash table lookup — O(1) average")

    ax.set_xlabel("Input size n (number of stored items)",
                  fontsize=11, color=C_DARK)
    ax.set_ylabel("Operations per lookup", fontsize=11, color=C_DARK)
    ax.set_title("Lookup cost grows very differently",
                 fontsize=14, fontweight="bold", color=C_DARK, pad=12)

    # Annotate at n = 1000
    ax.annotate("≈ 1000 ops", xy=(1000, 1000), xytext=(720, 880),
                fontsize=10, color=C_RED,
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.0))
    ax.annotate("≈ 10 ops", xy=(1000, np.log2(1001)), xytext=(720, 180),
                fontsize=10, color=C_AMBER,
                arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.0))
    ax.annotate("≈ 1 op", xy=(1000, 1), xytext=(720, 60),
                fontsize=10, color=C_BLUE,
                arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0))

    ax.legend(loc="upper left", fontsize=10, frameon=True,
              facecolor="white", edgecolor=C_LIGHT)
    ax.set_xlim(0, 1050)
    ax.set_ylim(0, 1080)
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    _save(fig, "fig4_complexity_compare")


# ---------------------------------------------------------------------------
# Fig 5 — Decision tree: array vs hash set vs hash map
# ---------------------------------------------------------------------------
def fig5_set_vs_map_decision() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(5.5, 6.6, "Pick the right structure: array  vs  hash set  vs  hash map",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DARK)

    def diamond(cx, cy, w, h, text, fc):
        pts = np.array([[cx, cy + h / 2],
                        [cx + w / 2, cy],
                        [cx, cy - h / 2],
                        [cx - w / 2, cy]])
        ax.fill(pts[:, 0], pts[:, 1], facecolor=fc, edgecolor=C_DARK,
                linewidth=1.2, alpha=0.18)
        ax.plot(np.append(pts[:, 0], pts[0, 0]),
                np.append(pts[:, 1], pts[0, 1]),
                color=C_DARK, lw=1.2)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=10, color=C_DARK, fontweight="bold")

    def leaf(cx, cy, w, h, title, body, color):
        _rounded_box(ax, cx - w / 2, cy - h / 2, w, h, fc=color, ec=color,
                     alpha=0.18, lw=1.5)
        ax.text(cx, cy + 0.18, title, ha="center", va="center",
                fontsize=11, color=color, fontweight="bold")
        ax.text(cx, cy - 0.22, body, ha="center", va="center",
                fontsize=9, color=C_DARK)

    def arrow(x1, y1, x2, y2, label="", color=C_GRAY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="->", mutation_scale=12,
                                     color=color, lw=1.0))
        if label:
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2,
                    label, fontsize=9, color=color, style="italic")

    # Root question
    diamond(5.5, 5.5, 4.4, 1.0,
            "Are keys integers in a known small range  [0..N)?",
            C_BLUE)

    # YES branch -> array
    arrow(4.0, 5.2, 2.0, 4.4, "yes")
    leaf(2.0, 3.9, 3.0, 1.0,
         "Plain array",
         "scores[i] — fastest, no hashing",
         C_GREEN)

    # NO branch -> next question
    arrow(7.0, 5.2, 8.5, 4.4, "no")
    diamond(8.5, 3.9, 3.6, 1.0,
            "Do you also need a value\nattached to each key?",
            C_PURPLE)

    # NO -> hash set
    arrow(7.0, 3.5, 5.0, 2.2, "no — just membership")
    leaf(5.0, 1.7, 3.6, 1.0,
         "Hash set",
         "set() — \"have I seen x?\"",
         C_AMBER)

    # YES -> hash map
    arrow(9.5, 3.3, 9.0, 2.2, "yes")
    leaf(9.0, 1.7, 3.4, 1.0,
         "Hash map",
         "dict — value, count, index, …",
         C_BLUE)

    ax.text(5.5, 0.4,
            "Rule of thumb:  array > set > map  in memory cost.  "
            "Use the lightest one your problem allows.",
            ha="center", va="center", fontsize=10, color=C_DARK,
            style="italic")

    fig.tight_layout()
    _save(fig, "fig5_set_vs_map_decision")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_hash_function()
    fig2_collision_resolution()
    fig3_two_sum_flow()
    fig4_complexity_compare()
    fig5_set_vs_map_decision()
    print(f"Wrote 5 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
