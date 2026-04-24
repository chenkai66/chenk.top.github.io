"""
Figure generation script for Linux Article 08: Pipelines and File Operations.

Generates 5 conceptual figures used in both EN and ZH versions of the article.
Each figure is rendered to BOTH article asset folders so the markdown image
references stay in sync across languages.

Figures:
    fig1_pipe_data_flow         The three standard streams (stdin/stdout/
                                stderr) flowing through a process, and how
                                the pipe operator only carries stdout from
                                one process to the next.
    fig2_redirection            Side-by-side panel showing the six common
                                redirection forms ('>', '>>', '<', '2>',
                                '2>&1', '&>') with what each does to fd 1
                                and fd 2.
    fig3_grep_awk_sed_pipeline  An end-to-end three-stage pipeline that
                                takes an Nginx-style log line through grep
                                (filter) -> awk (extract field) -> sort |
                                uniq -c | sort -nr (aggregate / rank).
    fig4_named_pipes            Named pipes (FIFOs) on disk: 'mkfifo'
                                creates a filesystem entry that two
                                independent processes use to talk, contrast
                                with anonymous '|' pipes.
    fig5_process_substitution   '<(cmd)' as a temporary file: bash exposes
                                the producer's stdout as '/dev/fd/N' so
                                tools that expect a filename argument can
                                consume command output without temp files.

Usage:
    python3 scripts/figures/linux/08-pipelines.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401  (registers the seaborn style we use)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

COLOR_BLUE = "#2563eb"
COLOR_PURPLE = "#7c3aed"
COLOR_GREEN = "#10b981"
COLOR_AMBER = "#f59e0b"
COLOR_GREY = "#475569"
COLOR_LIGHT = "#e2e8f0"
COLOR_RED = "#dc2626"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "linux" / "pipelines"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "linux" / "pipelines"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for out_dir in (EN_DIR, ZH_DIR):
        fig.savefig(out_dir / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _rounded_box(ax, x, y, w, h, *, facecolor, edgecolor=None, alpha=1.0, lw=1.2):
    """Helper to draw a rounded rectangle that we can label."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw,
        facecolor=facecolor,
        edgecolor=edgecolor or facecolor,
        alpha=alpha,
    )
    ax.add_patch(box)
    return box


# ---------------------------------------------------------------------------
# Figure 1: stdin / stdout / stderr and the pipe
# ---------------------------------------------------------------------------

def fig1_pipe_data_flow() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ---- Upper half: a single process and its three standard streams ----
    proc_x, proc_y, proc_w, proc_h = 4.8, 5.0, 2.4, 1.6
    _rounded_box(ax, proc_x, proc_y, proc_w, proc_h,
                 facecolor=COLOR_BLUE, edgecolor=COLOR_BLUE, alpha=0.9)
    ax.text(proc_x + proc_w / 2, proc_y + proc_h / 2 + 0.2, "process",
            ha="center", va="center", color="white",
            fontsize=14, fontweight="bold")
    ax.text(proc_x + proc_w / 2, proc_y + proc_h / 2 - 0.3,
            "e.g.  grep \"ERROR\"",
            ha="center", va="center", color="white",
            fontsize=10, family="monospace", style="italic")

    # stdin (left side, into process)
    arr_in = FancyArrowPatch((1.3, proc_y + proc_h / 2),
                             (proc_x, proc_y + proc_h / 2),
                             arrowstyle="->", mutation_scale=18,
                             color=COLOR_GREY, lw=1.8)
    ax.add_patch(arr_in)
    ax.text(0.6, proc_y + proc_h / 2 + 0.32, "stdin",
            ha="left", va="center", fontsize=12, color=COLOR_GREY,
            fontweight="bold")
    ax.text(0.6, proc_y + proc_h / 2 - 0.05, "fd 0",
            ha="left", va="center", fontsize=9, color=COLOR_GREY,
            family="monospace")
    ax.text(0.6, proc_y + proc_h / 2 - 0.4,
            "keyboard, file, or\nupstream pipe",
            ha="left", va="center", fontsize=8.5, color=COLOR_GREY,
            style="italic")

    # stdout (right side, normal data out)
    arr_out = FancyArrowPatch((proc_x + proc_w, proc_y + proc_h * 0.7),
                              (10.7, proc_y + proc_h * 0.7),
                              arrowstyle="->", mutation_scale=18,
                              color=COLOR_GREEN, lw=1.8)
    ax.add_patch(arr_out)
    ax.text(11.0, proc_y + proc_h * 0.7 + 0.15, "stdout",
            ha="left", va="center", fontsize=12, color=COLOR_GREEN,
            fontweight="bold")
    ax.text(11.0, proc_y + proc_h * 0.7 - 0.18, "fd 1",
            ha="left", va="center", fontsize=9, color=COLOR_GREEN,
            family="monospace")

    # stderr (right side, errors out -- separate channel)
    arr_err = FancyArrowPatch((proc_x + proc_w, proc_y + proc_h * 0.3),
                              (10.7, proc_y + proc_h * 0.3),
                              arrowstyle="->", mutation_scale=18,
                              color=COLOR_RED, lw=1.8)
    ax.add_patch(arr_err)
    ax.text(11.0, proc_y + proc_h * 0.3 + 0.15, "stderr",
            ha="left", va="center", fontsize=12, color=COLOR_RED,
            fontweight="bold")
    ax.text(11.0, proc_y + proc_h * 0.3 - 0.18, "fd 2",
            ha="left", va="center", fontsize=9, color=COLOR_RED,
            family="monospace")

    # Note on separation
    ax.text(6.0, 7.55,
            "Every process opens three standard streams; the kernel keeps "
            "stdout (data) and stderr (diagnostics) separate.",
            ha="center", va="center", fontsize=10, color=COLOR_GREY,
            style="italic")

    # ---- Lower half: two-process pipe -- only stdout flows through ----
    p1_x, p1_y, pw, ph = 1.2, 1.3, 2.6, 1.5
    p2_x = 5.2
    p3_x = 9.0

    for x, label, sub in [
        (p1_x, "cat access.log", "produces lines"),
        (p2_x, "grep \"ERROR\"", "filters lines"),
        (p3_x, "wc -l", "counts lines"),
    ]:
        _rounded_box(ax, x, p1_y, pw, ph,
                     facecolor=COLOR_PURPLE, edgecolor=COLOR_PURPLE, alpha=0.9)
        ax.text(x + pw / 2, p1_y + ph / 2 + 0.2, label,
                ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", family="monospace")
        ax.text(x + pw / 2, p1_y + ph / 2 - 0.28, sub,
                ha="center", va="center", color="white",
                fontsize=9, style="italic")

    # Pipe arrows between processes (green -- it's stdout flowing)
    for x_from, x_to in [(p1_x + pw, p2_x), (p2_x + pw, p3_x)]:
        arr = FancyArrowPatch((x_from, p1_y + ph / 2),
                              (x_to, p1_y + ph / 2),
                              arrowstyle="->", mutation_scale=18,
                              color=COLOR_GREEN, lw=2.0)
        ax.add_patch(arr)
        ax.text((x_from + x_to) / 2, p1_y + ph / 2 + 0.3, "|",
                ha="center", va="center", fontsize=18,
                color=COLOR_GREEN, fontweight="bold", family="monospace")
        ax.text((x_from + x_to) / 2, p1_y + ph / 2 - 0.32,
                "stdout -> stdin",
                ha="center", va="center", fontsize=8.5, color=COLOR_GREEN,
                family="monospace")

    # stderr drops out the side of each (to terminal) -- show on first one
    arr_e = FancyArrowPatch((p1_x + pw / 2, p1_y),
                            (p1_x + pw / 2, 0.3),
                            arrowstyle="->", mutation_scale=14,
                            color=COLOR_RED, lw=1.4, linestyle="--")
    ax.add_patch(arr_e)
    ax.text(p1_x + pw / 2 + 0.1, 0.45, "stderr -> terminal\n(not piped)",
            ha="left", va="center", fontsize=8.5, color=COLOR_RED,
            style="italic")

    # Section heading
    ax.text(6.0, 3.2,
            "The pipe operator |  carries only stdout.  "
            "stderr stays on the terminal unless you redirect it.",
            ha="center", va="center", fontsize=10.5, color=COLOR_GREY,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc",
                      edgecolor=COLOR_LIGHT, lw=1.0))

    ax.set_title("Pipe Data Flow: stdin / stdout / stderr",
                 fontsize=15, fontweight="bold", pad=12, color="#1e293b")

    _save(fig, "fig1_pipe_data_flow.png")


# ---------------------------------------------------------------------------
# Figure 2: Redirection forms
# ---------------------------------------------------------------------------

def fig2_redirection() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # 6 cards in a 2 x 3 grid
    cards = [
        {
            "op": "cmd > file",
            "name": "overwrite stdout",
            "desc": "fd 1 -> file, truncate first.\nfd 2 still goes to terminal.",
            "color": COLOR_BLUE,
        },
        {
            "op": "cmd >> file",
            "name": "append stdout",
            "desc": "fd 1 -> file, keep existing\ncontent and append at end.",
            "color": COLOR_BLUE,
        },
        {
            "op": "cmd < file",
            "name": "read stdin",
            "desc": "fd 0 reads from file instead\nof the terminal keyboard.",
            "color": COLOR_PURPLE,
        },
        {
            "op": "cmd 2> file",
            "name": "redirect stderr",
            "desc": "fd 2 -> file. fd 1 still on\nterminal -- errors go quiet.",
            "color": COLOR_AMBER,
        },
        {
            "op": "cmd > out 2>&1",
            "name": "merge stderr into stdout",
            "desc": "Order matters: redirect fd 1\nfirst, then duplicate fd 2 to it.",
            "color": COLOR_GREEN,
        },
        {
            "op": "cmd &> file",
            "name": "merge -- modern shorthand",
            "desc": "Both fd 1 and fd 2 -> file in\none token (bash / zsh).",
            "color": COLOR_GREEN,
        },
    ]

    cols = 3
    rows = 2
    margin_x = 0.5
    margin_y = 0.6
    gap_x = 0.4
    gap_y = 0.6
    title_band = 1.2  # space at top for big title

    avail_w = 12 - 2 * margin_x - (cols - 1) * gap_x
    avail_h = 9 - title_band - 2 * margin_y - (rows - 1) * gap_y
    card_w = avail_w / cols
    card_h = avail_h / rows

    for i, card in enumerate(cards):
        r = i // cols
        c = i % cols
        x = margin_x + c * (card_w + gap_x)
        y = 9 - title_band - margin_y - (r + 1) * card_h - r * gap_y

        # Outer card
        _rounded_box(ax, x, y, card_w, card_h,
                     facecolor="white", edgecolor=card["color"], lw=1.6)

        # Header band
        header_h = 0.8
        _rounded_box(ax, x, y + card_h - header_h, card_w, header_h,
                     facecolor=card["color"], edgecolor=card["color"], alpha=0.95)
        ax.text(x + card_w / 2, y + card_h - header_h / 2, card["op"],
                ha="center", va="center", color="white",
                fontsize=12.5, fontweight="bold", family="monospace")

        # Body
        ax.text(x + card_w / 2, y + card_h - header_h - 0.45, card["name"],
                ha="center", va="center", color=card["color"],
                fontsize=11, fontweight="bold")
        ax.text(x + card_w / 2, y + card_h / 2 - 0.55, card["desc"],
                ha="center", va="center", color=COLOR_GREY,
                fontsize=9.6)

        # Mini fd diagram at the bottom of the card
        diag_y = y + 0.45
        # fd 1 chip
        _rounded_box(ax, x + 0.3, diag_y, 0.7, 0.35,
                     facecolor=COLOR_GREEN, alpha=0.85)
        ax.text(x + 0.65, diag_y + 0.175, "fd 1",
                ha="center", va="center", color="white",
                fontsize=8, fontweight="bold")
        # fd 2 chip
        _rounded_box(ax, x + 1.05, diag_y, 0.7, 0.35,
                     facecolor=COLOR_RED, alpha=0.85)
        ax.text(x + 1.4, diag_y + 0.175, "fd 2",
                ha="center", va="center", color="white",
                fontsize=8, fontweight="bold")

        # Per-card targets
        if i == 0:  # >
            ax.text(x + card_w - 0.3, diag_y + 0.18, "-> file",
                    ha="right", va="center", color=COLOR_BLUE,
                    fontsize=9, family="monospace")
        elif i == 1:  # >>
            ax.text(x + card_w - 0.3, diag_y + 0.18, "->> file",
                    ha="right", va="center", color=COLOR_BLUE,
                    fontsize=9, family="monospace")
        elif i == 2:  # <
            # show fd 0 chip instead
            _rounded_box(ax, x + 0.3, diag_y, 0.7, 0.35,
                         facecolor=COLOR_PURPLE, alpha=0.85)
            ax.text(x + 0.65, diag_y + 0.175, "fd 0",
                    ha="center", va="center", color="white",
                    fontsize=8, fontweight="bold")
            # hide fd 2 chip behind a white rect for cleanliness
            ax.add_patch(Rectangle((x + 1.05, diag_y - 0.02), 0.72, 0.4,
                                   facecolor="white", edgecolor="white"))
            ax.text(x + card_w - 0.3, diag_y + 0.18, "<- file",
                    ha="right", va="center", color=COLOR_PURPLE,
                    fontsize=9, family="monospace")
        elif i == 3:  # 2>
            ax.text(x + card_w - 0.3, diag_y + 0.18, "fd 2 -> file",
                    ha="right", va="center", color=COLOR_AMBER,
                    fontsize=9, family="monospace")
        elif i == 4:  # 2>&1
            ax.text(x + card_w - 0.3, diag_y + 0.18, "both -> file",
                    ha="right", va="center", color=COLOR_GREEN,
                    fontsize=9, family="monospace")
        else:  # &>
            ax.text(x + card_w - 0.3, diag_y + 0.18, "both -> file",
                    ha="right", va="center", color=COLOR_GREEN,
                    fontsize=9, family="monospace")

    # Big title
    ax.text(6.0, 8.55, "Redirection: where each file descriptor goes",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color="#1e293b")
    ax.text(6.0, 8.10,
            "fd 0 = stdin   |   fd 1 = stdout   |   fd 2 = stderr",
            ha="center", va="center", fontsize=10, color=COLOR_GREY,
            family="monospace")

    _save(fig, "fig2_redirection.png")


# ---------------------------------------------------------------------------
# Figure 3: grep | awk | sort | uniq pipeline (Nginx top-IP example)
# ---------------------------------------------------------------------------

def fig3_grep_awk_sed_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Stage boxes along a horizontal flow
    stages = [
        {
            "title": "access.log",
            "sub": "raw input",
            "cmd": "cat",
            "color": COLOR_GREY,
            "sample": [
                "10.0.0.1 ... \"GET /api\" 200 ...",
                "10.0.0.2 ... \"GET /img\" 404 ...",
                "10.0.0.1 ... \"GET /api\" 500 ...",
                "10.0.0.3 ... \"GET /api\" 200 ...",
                "10.0.0.1 ... \"GET /api\" 500 ...",
            ],
        },
        {
            "title": "grep",
            "sub": "filter:  status 5xx",
            "cmd": "grep -E ' 5[0-9]{2} '",
            "color": COLOR_BLUE,
            "sample": [
                "10.0.0.1 ... \"GET /api\" 500 ...",
                "10.0.0.1 ... \"GET /api\" 500 ...",
            ],
        },
        {
            "title": "awk",
            "sub": "extract field 1 (IP)",
            "cmd": "awk '{print $1}'",
            "color": COLOR_PURPLE,
            "sample": [
                "10.0.0.1",
                "10.0.0.1",
            ],
        },
        {
            "title": "sort | uniq -c",
            "sub": "group + count",
            "cmd": "sort | uniq -c | sort -nr",
            "color": COLOR_GREEN,
            "sample": [
                "  2  10.0.0.1",
            ],
        },
    ]

    n = len(stages)
    box_w = 2.65
    box_h = 3.5
    margin = 0.4
    gap = (13 - 2 * margin - n * box_w) / (n - 1)
    y_box = 2.4

    centers_x = []
    for i, st in enumerate(stages):
        x = margin + i * (box_w + gap)
        centers_x.append(x + box_w / 2)

        # Card frame
        _rounded_box(ax, x, y_box, box_w, box_h,
                     facecolor="white", edgecolor=st["color"], lw=1.6)
        # Header
        header_h = 0.7
        _rounded_box(ax, x, y_box + box_h - header_h, box_w, header_h,
                     facecolor=st["color"], edgecolor=st["color"], alpha=0.95)
        ax.text(x + box_w / 2, y_box + box_h - header_h / 2, st["title"],
                ha="center", va="center", color="white",
                fontsize=13, fontweight="bold")

        # Sub-label
        ax.text(x + box_w / 2, y_box + box_h - header_h - 0.3, st["sub"],
                ha="center", va="center", color=st["color"],
                fontsize=9.5, fontweight="bold", style="italic")

        # Sample data block
        sample_top = y_box + box_h - header_h - 0.7
        line_h = 0.32
        for j, line in enumerate(st["sample"]):
            ax.text(x + 0.15, sample_top - j * line_h, line,
                    ha="left", va="center", fontsize=8.2,
                    family="monospace", color="#1e293b")

        # Command at the bottom of the card
        ax.text(x + box_w / 2, y_box + 0.3, st["cmd"],
                ha="center", va="center", color=COLOR_GREY,
                fontsize=8.8, family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#f1f5f9",
                          edgecolor=COLOR_LIGHT, lw=0.8))

    # Pipe arrows between stages
    for i in range(n - 1):
        x_from = margin + i * (box_w + gap) + box_w
        x_to = margin + (i + 1) * (box_w + gap)
        y_arrow = y_box + box_h / 2
        arr = FancyArrowPatch((x_from + 0.05, y_arrow),
                              (x_to - 0.05, y_arrow),
                              arrowstyle="->", mutation_scale=20,
                              color=COLOR_GREEN, lw=2.2)
        ax.add_patch(arr)
        ax.text((x_from + x_to) / 2, y_arrow + 0.4, "|",
                ha="center", va="center", fontsize=18, color=COLOR_GREEN,
                fontweight="bold", family="monospace")

    # Full one-liner across the top
    ax.text(6.5, 7.4,
            "grep -E ' 5[0-9]{2} ' access.log | awk '{print $1}' | "
            "sort | uniq -c | sort -nr | head",
            ha="center", va="center", fontsize=11.5, color="#1e293b",
            family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7",
                      edgecolor=COLOR_AMBER, lw=1.2))
    ax.text(6.5, 6.85,
            "Each tool does one thing.  Compose them with | to answer "
            "\"which IPs are causing 5xx errors?\"",
            ha="center", va="center", fontsize=10, color=COLOR_GREY,
            style="italic")

    # Bottom annotation: data shape narrows at each stage
    labels = ["all lines", "only 5xx", "just IPs", "ranked counts"]
    for cx, lbl in zip(centers_x, labels):
        ax.text(cx, 1.7, lbl,
                ha="center", va="center", fontsize=9, color=COLOR_GREY,
                style="italic")

    # Funnel arrow underneath
    arr2 = FancyArrowPatch((centers_x[0], 1.1), (centers_x[-1], 1.1),
                           arrowstyle="->", mutation_scale=14,
                           color=COLOR_GREY, lw=1.0, linestyle=":")
    ax.add_patch(arr2)
    ax.text((centers_x[0] + centers_x[-1]) / 2, 0.7,
            "data narrows and aggregates from left to right",
            ha="center", va="center", fontsize=9, color=COLOR_GREY,
            style="italic")

    ax.set_title("A grep / awk / sort / uniq pipeline",
                 fontsize=15, fontweight="bold", pad=12, color="#1e293b")

    _save(fig, "fig3_grep_awk_sed_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 4: Named pipes (FIFOs) vs anonymous |
# ---------------------------------------------------------------------------

def fig4_named_pipes() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    fig.subplots_adjust(wspace=0.25)

    # ---------- LEFT: anonymous pipe ----------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Anonymous pipe   producer | consumer",
                 fontsize=12.5, fontweight="bold", color=COLOR_PURPLE)

    # Two processes
    _rounded_box(ax, 0.6, 2.7, 2.4, 1.6,
                 facecolor=COLOR_PURPLE, edgecolor=COLOR_PURPLE, alpha=0.9)
    ax.text(1.8, 3.7, "producer", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(1.8, 3.2, "ls -l",
            ha="center", va="center", color="white",
            fontsize=10, family="monospace", style="italic")

    _rounded_box(ax, 7.0, 2.7, 2.4, 1.6,
                 facecolor=COLOR_PURPLE, edgecolor=COLOR_PURPLE, alpha=0.9)
    ax.text(8.2, 3.7, "consumer", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(8.2, 3.2, "wc -l",
            ha="center", va="center", color="white",
            fontsize=10, family="monospace", style="italic")

    # Pipe in the middle (kernel buffer)
    _rounded_box(ax, 3.4, 3.05, 3.2, 0.9,
                 facecolor=COLOR_GREEN, edgecolor=COLOR_GREEN, alpha=0.85)
    ax.text(5.0, 3.5, "kernel pipe buffer",
            ha="center", va="center", color="white",
            fontsize=10, fontweight="bold")
    ax.text(5.0, 3.15, "no name on disk",
            ha="center", va="center", color="white",
            fontsize=8.5, style="italic")

    arr1 = FancyArrowPatch((3.0, 3.5), (3.4, 3.5),
                           arrowstyle="->", mutation_scale=14,
                           color=COLOR_GREEN, lw=1.6)
    arr2 = FancyArrowPatch((6.6, 3.5), (7.0, 3.5),
                           arrowstyle="->", mutation_scale=14,
                           color=COLOR_GREEN, lw=1.6)
    ax.add_patch(arr1)
    ax.add_patch(arr2)

    # Notes
    ax.text(5.0, 5.7,
            "$ ls -l | wc -l",
            ha="center", va="center", fontsize=12, color="#1e293b",
            family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                      edgecolor=COLOR_AMBER, lw=1.0))
    ax.text(5.0, 5.05,
            "Both processes are spawned by the same shell;\n"
            "the pipe lives only as long as they do.",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY,
            style="italic")

    ax.text(5.0, 1.5,
            "Limitation: producer and consumer must be on\n"
            "the same command line (same parent shell).",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY)

    # ---------- RIGHT: named pipe (FIFO) ----------
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Named pipe (FIFO)   mkfifo /tmp/jobs",
                 fontsize=12.5, fontweight="bold", color=COLOR_BLUE)

    # Two unrelated processes (different terminals)
    _rounded_box(ax, 0.4, 2.7, 2.6, 1.6,
                 facecolor=COLOR_BLUE, edgecolor=COLOR_BLUE, alpha=0.9)
    ax.text(1.7, 3.75, "shell A", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(1.7, 3.25, "echo job-42 > /tmp/jobs",
            ha="center", va="center", color="white",
            fontsize=8.5, family="monospace", style="italic")

    _rounded_box(ax, 7.0, 2.7, 2.6, 1.6,
                 facecolor=COLOR_BLUE, edgecolor=COLOR_BLUE, alpha=0.9)
    ax.text(8.3, 3.75, "shell B", ha="center", va="center",
            color="white", fontsize=12, fontweight="bold")
    ax.text(8.3, 3.25, "while read j; do ...; done < /tmp/jobs",
            ha="center", va="center", color="white",
            fontsize=8.0, family="monospace", style="italic")

    # FIFO file on disk in the middle
    _rounded_box(ax, 3.6, 2.95, 2.8, 1.1,
                 facecolor=COLOR_AMBER, edgecolor=COLOR_AMBER, alpha=0.9)
    ax.text(5.0, 3.7, "/tmp/jobs",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", family="monospace")
    ax.text(5.0, 3.25, "p file (FIFO, named pipe)",
            ha="center", va="center", color="white",
            fontsize=8.5, style="italic")

    # Filesystem indicator under FIFO
    ax.text(5.0, 2.65, "ls -l shows  prw-r--r--",
            ha="center", va="center", fontsize=8, color=COLOR_AMBER,
            family="monospace")

    arr1 = FancyArrowPatch((3.0, 3.5), (3.6, 3.5),
                           arrowstyle="->", mutation_scale=14,
                           color=COLOR_AMBER, lw=1.6)
    arr2 = FancyArrowPatch((6.4, 3.5), (7.0, 3.5),
                           arrowstyle="->", mutation_scale=14,
                           color=COLOR_AMBER, lw=1.6)
    ax.add_patch(arr1)
    ax.add_patch(arr2)

    # Notes
    ax.text(5.0, 5.7,
            "$ mkfifo /tmp/jobs",
            ha="center", va="center", fontsize=12, color="#1e293b",
            family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                      edgecolor=COLOR_AMBER, lw=1.0))
    ax.text(5.0, 5.05,
            "The pipe has a name on the filesystem.\n"
            "Any two processes can rendezvous on it.",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY,
            style="italic")

    ax.text(5.0, 1.5,
            "Use case: simple job queue, one-shot signalling,\n"
            "decoupling producer and consumer in time.",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY)

    fig.suptitle("Pipes have two flavours: anonymous  |  and named (FIFO)",
                 fontsize=14, fontweight="bold", y=1.02, color="#1e293b")

    _save(fig, "fig4_named_pipes.png")


# ---------------------------------------------------------------------------
# Figure 5: Process substitution <(cmd)
# ---------------------------------------------------------------------------

def fig5_process_substitution() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Top: the literal command the user types
    ax.text(6, 7.3,
            "$ diff  <(sort file1)  <(sort file2)",
            ha="center", va="center", fontsize=14, color="#1e293b",
            family="monospace", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fef3c7",
                      edgecolor=COLOR_AMBER, lw=1.2))
    ax.text(6, 6.75,
            "diff expects two filename arguments -- and that is exactly "
            "what process substitution gives it.",
            ha="center", va="center", fontsize=10, color=COLOR_GREY,
            style="italic")

    # Two producer boxes
    _rounded_box(ax, 0.6, 3.6, 2.6, 1.5,
                 facecolor=COLOR_PURPLE, edgecolor=COLOR_PURPLE, alpha=0.9)
    ax.text(1.9, 4.55, "sort file1",
            ha="center", va="center", color="white",
            fontsize=11.5, fontweight="bold", family="monospace")
    ax.text(1.9, 4.05, "producer A",
            ha="center", va="center", color="white",
            fontsize=9, style="italic")

    _rounded_box(ax, 8.8, 3.6, 2.6, 1.5,
                 facecolor=COLOR_PURPLE, edgecolor=COLOR_PURPLE, alpha=0.9)
    ax.text(10.1, 4.55, "sort file2",
            ha="center", va="center", color="white",
            fontsize=11.5, fontweight="bold", family="monospace")
    ax.text(10.1, 4.05, "producer B",
            ha="center", va="center", color="white",
            fontsize=9, style="italic")

    # Bash exposes each producer's stdout as a /dev/fd/N entry
    _rounded_box(ax, 0.6, 1.6, 2.6, 1.0,
                 facecolor=COLOR_AMBER, edgecolor=COLOR_AMBER, alpha=0.9)
    ax.text(1.9, 2.1, "/dev/fd/63",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", family="monospace")

    _rounded_box(ax, 8.8, 1.6, 2.6, 1.0,
                 facecolor=COLOR_AMBER, edgecolor=COLOR_AMBER, alpha=0.9)
    ax.text(10.1, 2.1, "/dev/fd/64",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold", family="monospace")

    # Producer -> /dev/fd arrows
    arr_pa = FancyArrowPatch((1.9, 3.6), (1.9, 2.6),
                             arrowstyle="->", mutation_scale=16,
                             color=COLOR_AMBER, lw=1.6)
    arr_pb = FancyArrowPatch((10.1, 3.6), (10.1, 2.6),
                             arrowstyle="->", mutation_scale=16,
                             color=COLOR_AMBER, lw=1.6)
    ax.add_patch(arr_pa)
    ax.add_patch(arr_pb)
    ax.text(2.2, 3.1, "stdout exposed\nas a fd path",
            ha="left", va="center", fontsize=8.5, color=COLOR_AMBER,
            style="italic")
    ax.text(9.8, 3.1, "stdout exposed\nas a fd path",
            ha="right", va="center", fontsize=8.5, color=COLOR_AMBER,
            style="italic")

    # Center: diff consumes both as if they were files
    _rounded_box(ax, 4.4, 3.4, 3.2, 1.8,
                 facecolor=COLOR_BLUE, edgecolor=COLOR_BLUE, alpha=0.9)
    ax.text(6.0, 4.6, "diff",
            ha="center", va="center", color="white",
            fontsize=14, fontweight="bold", family="monospace")
    ax.text(6.0, 4.05, "consumer",
            ha="center", va="center", color="white",
            fontsize=9.5, style="italic")
    ax.text(6.0, 3.65, "opens both fd paths\nas regular files",
            ha="center", va="center", color="white",
            fontsize=8.5)

    # /dev/fd -> diff arrows
    arr_da = FancyArrowPatch((3.2, 2.1), (4.4, 3.7),
                             arrowstyle="->", mutation_scale=16,
                             color=COLOR_BLUE, lw=1.6)
    arr_db = FancyArrowPatch((8.8, 2.1), (7.6, 3.7),
                             arrowstyle="->", mutation_scale=16,
                             color=COLOR_BLUE, lw=1.6)
    ax.add_patch(arr_da)
    ax.add_patch(arr_db)

    # Footer comparison
    ax.text(6.0, 0.85,
            "Equivalent to creating two temp files and deleting them, "
            "but bash does the bookkeeping.",
            ha="center", va="center", fontsize=10, color=COLOR_GREY,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8fafc",
                      edgecolor=COLOR_LIGHT, lw=1.0))

    ax.set_title("Process substitution:  <(cmd)  exposes a command's stdout "
                 "as a filename",
                 fontsize=14, fontweight="bold", pad=12, color="#1e293b")

    _save(fig, "fig5_process_substitution.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_pipe_data_flow()
    fig2_redirection()
    fig3_grep_awk_sed_pipeline()
    fig4_named_pipes()
    fig5_process_substitution()
    print("Wrote 5 figures to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
