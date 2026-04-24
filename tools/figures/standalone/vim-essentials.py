"""
Figure generation script for the standalone article: Vim Essentials.

Generates 5 figures used in both the EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in editorial print style.

Figures:
    fig1_mode_state_diagram   The four primary modes (Normal / Insert /
                              Visual / Command-line) and the keystrokes
                              that transition between them. Replace mode
                              shown as a side state.
    fig2_motion_cheatsheet    Keyboard cheat-sheet for movement keys --
                              hjkl, w/b/e, 0/^/$, gg/G -- laid out on a
                              schematic keyboard with arrows showing what
                              each key does.
    fig3_text_objects         Visual decomposition of operator + text
                              object: ciw, dap, ci" rendered against
                              real example text with selection brackets.
    fig4_vim_file_structure   Anatomy of a .vimrc and the surrounding
                              filesystem (~/.vimrc, ~/.vim/, plugins,
                              swap, undo). Shows what lives where.
    fig5_workflows            Two common workflows -- search & replace
                              and multi-file editing with splits/buffers
                              -- shown as labelled step diagrams.

Usage:
    python3 scripts/figures/standalone/vim-essentials.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (

    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]     # primary
C_PURPLE = COLORS["accent"]   # secondary
C_GREEN = COLORS["success"]    # accent / good
C_AMBER = COLORS["warning"]    # warning / highlight
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG_SOFT = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "vim-essentials"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "linux" / "Vim-解析"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _no_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)


# ---------------------------------------------------------------------------
# Figure 1 -- Mode state diagram
# ---------------------------------------------------------------------------
def fig1_mode_state_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.5))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)

    # Modes as nodes (x, y, label, sublabel, color)
    nodes = {
        "normal":  (6.5, 5.4, "NORMAL",       "navigate / operate",   C_BLUE),
        "insert":  (2.4, 5.4, "INSERT",       "type text",            C_GREEN),
        "visual":  (10.6, 5.4, "VISUAL",      "select region",        C_PURPLE),
        "command": (6.5, 1.9, "COMMAND-LINE", ":w  :%s  :e ...",      C_AMBER),
        "replace": (10.6, 1.9, "REPLACE",     "overwrite chars",      C_RED),
    }

    def draw_node(key: str, w: float = 2.4, h: float = 1.05) -> None:
        x, y, label, sub, color = nodes[key]
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06", facecolor=color, edgecolor="none",
            alpha=0.95, zorder=2,
        ))
        ax.text(x, y + 0.2, label, ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")
        ax.text(x, y - 0.22, sub, ha="center", va="center",
                fontsize=10, color="white", style="italic")

    for k in nodes:
        draw_node(k)

    # Arrow helper with a key label on the midpoint
    def arrow(p0, p1, key: str, curve: float = 0.0, color: str = C_DARK,
              dx: float = 0.0, dy: float = 0.0, fontsize: float = 11) -> None:
        arr = FancyArrowPatch(
            p0, p1,
            arrowstyle="-|>", mutation_scale=18,
            lw=1.6, color=color,
            connectionstyle=f"arc3,rad={curve}",
            zorder=1,
        )
        ax.add_patch(arr)
        mx = (p0[0] + p1[0]) / 2 + dx
        my = (p0[1] + p1[1]) / 2 + dy
        ax.add_patch(FancyBboxPatch(
            (mx - 0.42, my - 0.18), 0.84, 0.36,
            boxstyle="round,pad=0.02", facecolor="white",
            edgecolor=color, lw=1.2, zorder=3,
        ))
        ax.text(mx, my, key, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color,
                family="monospace", zorder=4)

    nx, ny = nodes["normal"][0], nodes["normal"][1]
    ix, iy = nodes["insert"][0], nodes["insert"][1]
    vx, vy = nodes["visual"][0], nodes["visual"][1]
    cx, cy = nodes["command"][0], nodes["command"][1]
    rx, ry = nodes["replace"][0], nodes["replace"][1]

    # Normal <-> Insert
    arrow((nx - 1.25, ny + 0.25), (ix + 1.25, iy + 0.25), "i  a  o",
          curve=0.18, color=C_GREEN, dy=0.35)
    arrow((ix + 1.25, iy - 0.25), (nx - 1.25, ny - 0.25), "Esc",
          curve=0.18, color=C_DARK, dy=-0.35)

    # Normal <-> Visual
    arrow((nx + 1.25, ny + 0.25), (vx - 1.25, vy + 0.25), "v  V  ^V",
          curve=0.18, color=C_PURPLE, dy=0.35)
    arrow((vx - 1.25, vy - 0.25), (nx + 1.25, ny - 0.25), "Esc",
          curve=0.18, color=C_DARK, dy=-0.35)

    # Normal <-> Command-line
    arrow((nx - 0.25, ny - 0.55), (cx - 0.25, cy + 0.55), ":  /  ?",
          curve=0.0, color=C_AMBER, dx=-0.55)
    arrow((cx + 0.25, cy + 0.55), (nx + 0.25, ny - 0.55), "Enter / Esc",
          curve=0.0, color=C_DARK, dx=0.85, fontsize=10)

    # Normal <-> Replace
    arrow((nx + 1.0, ny - 0.5), (rx - 1.2, ry + 0.5), "R",
          curve=-0.2, color=C_RED, dx=0.4, dy=0.1)
    arrow((rx - 1.2, ry + 0.2), (nx + 1.0, ny - 0.8), "Esc",
          curve=0.2, color=C_DARK, dx=-0.6, dy=-0.2)

    # Title and tagline
    ax.text(6.5, 7.85, "The Four Modes of Vim",
            ha="center", va="center", fontsize=18, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.42, "Normal mode is home -- every other mode returns to it via Esc",
            ha="center", va="center", fontsize=11.5, style="italic", color=C_GRAY)

    # "Home base" badge under Normal
    ax.add_patch(FancyBboxPatch(
        (nx - 1.05, ny - 1.25), 2.1, 0.45,
        boxstyle="round,pad=0.04", facecolor=C_BG_SOFT,
        edgecolor=C_BLUE, lw=1.2,
    ))
    ax.text(nx, ny - 1.025, "home base -- spend 80% here",
            ha="center", va="center", fontsize=10, color=C_BLUE, fontweight="bold")

    # Bottom legend strip
    ax.text(0.4, 0.55, "Tip:", fontsize=10.5, fontweight="bold", color=C_DARK)
    ax.text(1.1, 0.55,
            "if you are ever lost, press Esc until the bottom status line is empty -- "
            "you are back in Normal mode.",
            fontsize=10.5, color=C_DARK)

    _save(fig, "fig1_mode_state_diagram")


# ---------------------------------------------------------------------------
# Figure 2 -- Motion cheat-sheet on a schematic keyboard
# ---------------------------------------------------------------------------
def fig2_motion_cheatsheet() -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8)

    # Title
    ax.text(6.5, 7.55, "Vim Motion Cheat Sheet",
            ha="center", va="center", fontsize=18, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.17, "the keys you use 1000 times a day",
            ha="center", va="center", fontsize=11.5, style="italic", color=C_GRAY)

    # Keyboard rows -- only the keys we care about, drawn as a real-ish layout.
    # Each row: list of (label, color or None)
    KEY_W, KEY_H = 0.78, 0.78
    ROW_GAP = 0.18

    rows = [
        # row label, x_off, list of (key, color, note)
        ("home row", 1.6, [
            ("h", C_BLUE,   "left"),
            ("j", C_BLUE,   "down"),
            ("k", C_BLUE,   "up"),
            ("l", C_BLUE,   "right"),
            ("",  None,     ""),
            ("0", C_PURPLE, "line start"),
            ("^", C_PURPLE, "first non-blank"),
            ("$", C_PURPLE, "line end"),
        ]),
        ("words", 1.6, [
            ("b", C_GREEN, "prev word start"),
            ("w", C_GREEN, "next word start"),
            ("e", C_GREEN, "next word end"),
            ("",  None,    ""),
            ("f", C_AMBER, "find char fwd"),
            ("F", C_AMBER, "find char back"),
            ("t", C_AMBER, "till char fwd"),
            ("T", C_AMBER, "till char back"),
        ]),
        ("file & screen", 1.6, [
            ("g g", C_BLUE,   "top of file"),
            ("G",   C_BLUE,   "end of file"),
            ("nG",  C_BLUE,   "go to line n"),
            ("",    None,     ""),
            ("^F",  C_PURPLE, "page down"),
            ("^B",  C_PURPLE, "page up"),
            ("^D",  C_PURPLE, "half down"),
            ("^U",  C_PURPLE, "half up"),
        ]),
        ("search", 1.6, [
            ("/",   C_RED,   "search fwd"),
            ("?",   C_RED,   "search back"),
            ("n",   C_RED,   "next match"),
            ("N",   C_RED,   "prev match"),
            ("",    None,    ""),
            ("*",   C_AMBER, "word under cursor fwd"),
            ("#",   C_AMBER, "word under cursor back"),
            ("%",   C_AMBER, "match bracket"),
        ]),
    ]

    y_top = 6.4
    for ri, (row_label, x_off, keys) in enumerate(rows):
        y = y_top - ri * (KEY_H + ROW_GAP + 0.55)

        # Row label
        ax.text(0.95, y, row_label, ha="right", va="center",
                fontsize=11, fontweight="bold", color=C_DARK)

        x = x_off
        for key, color, note in keys:
            if key == "":
                x += 0.4
                continue
            # Key cap
            kw = KEY_W if len(key) <= 1 else KEY_W + 0.3 * (len(key) - 1)
            ax.add_patch(FancyBboxPatch(
                (x, y - KEY_H / 2), kw, KEY_H,
                boxstyle="round,pad=0.04", facecolor="white",
                edgecolor=color, lw=1.8, zorder=2,
            ))
            ax.text(x + kw / 2, y, key, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color,
                    family="monospace", zorder=3)
            # Note below
            ax.text(x + kw / 2, y - KEY_H / 2 - 0.22, note,
                    ha="center", va="top", fontsize=8.7, color=C_DARK)
            x += kw + 0.16

    # Bottom hint: count + motion grammar
    box_y = 0.55
    ax.add_patch(FancyBboxPatch(
        (0.6, box_y - 0.05), 11.8, 0.85,
        boxstyle="round,pad=0.05", facecolor=C_BG_SOFT,
        edgecolor=C_DARK, lw=1.0,
    ))
    ax.text(0.95, box_y + 0.55, "Grammar:",
            fontsize=11.5, fontweight="bold", color=C_DARK)
    ax.text(2.15, box_y + 0.55,
            "[count] [operator] [motion]   e.g.   3 dw  =  delete next 3 words   |   "
            "5j  =  move 5 lines down   |   d2}  =  delete 2 paragraphs",
            fontsize=10.7, color=C_DARK, family="monospace")
    ax.text(0.95, box_y + 0.05,
            "Rule of thumb: any motion can follow an operator (d / c / y) to operate "
            "on exactly that range.",
            fontsize=10.3, color=C_GRAY, style="italic")

    _save(fig, "fig2_motion_cheatsheet")


# ---------------------------------------------------------------------------
# Figure 3 -- Text objects (operator + text object) decomposition
# ---------------------------------------------------------------------------
def fig3_text_objects() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.4))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.4)

    ax.text(6.5, 7.95, "Text Objects -- Edit by Meaning, Not by Position",
            ha="center", va="center", fontsize=17, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.55, "operator + (i)nner / (a)round + object  =  one fluent edit",
            ha="center", va="center", fontsize=11.2, style="italic", color=C_GRAY)

    # Three example rows. Each: (label, code, before_text, before_highlight (start,end), explanation)
    examples = [
        {
            "title": "ciw -- change inner word",
            "code":  "c i w",
            "color": C_BLUE,
            "before": "let userName = 'alice'",
            "hl_start": 4, "hl_end": 12,
            "note": "cursor anywhere on userName -> deletes the word and enters Insert mode",
        },
        {
            "title": "dap -- delete around paragraph",
            "code":  "d a p",
            "color": C_PURPLE,
            "before": "  This whole paragraph (and the trailing blank line) goes away.",
            "hl_start": 0, "hl_end": 63,
            "note": "removes the paragraph plus its surrounding blank line -- one keystroke trio",
        },
        {
            "title": 'ci" -- change inside quotes',
            "code":  "c i \"",
            "color": C_GREEN,
            "before": 'config.host = "old.server.com"',
            "hl_start": 15, "hl_end": 29,
            "note": "even if cursor is on \", jumps inside and replaces the contents",
        },
    ]

    row_y_top = 6.7
    row_h = 1.85

    for i, ex in enumerate(examples):
        y = row_y_top - i * row_h
        color = ex["color"]

        # Card background
        ax.add_patch(FancyBboxPatch(
            (0.6, y - 1.55), 11.8, 1.7,
            boxstyle="round,pad=0.06", facecolor=C_BG_SOFT,
            edgecolor=C_LIGHT, lw=1.0,
        ))

        # Title
        ax.text(0.95, y - 0.05, ex["title"],
                fontsize=12.5, fontweight="bold", color=color)

        # Decompose the keystroke into 3 cells
        keys = ex["code"].split()
        legend = ["operator", "inner / around", "text object"]
        cell_x = 9.0
        cell_w = 0.85
        for k_i, (k, lab) in enumerate(zip(keys, legend)):
            cx = cell_x + k_i * (cell_w + 0.1)
            ax.add_patch(FancyBboxPatch(
                (cx, y - 0.35), cell_w, 0.6,
                boxstyle="round,pad=0.03", facecolor="white",
                edgecolor=color, lw=1.6,
            ))
            ax.text(cx + cell_w / 2, y - 0.05, k, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color,
                    family="monospace")
            ax.text(cx + cell_w / 2, y - 0.55, lab, ha="center", va="top",
                    fontsize=8.3, color=C_GRAY, style="italic")

        # Code line with highlight underlay
        text_x = 0.95
        text_y = y - 0.85
        before = ex["before"]
        # Render a fixed-width block by drawing a faint base
        char_w = 0.13  # approx monospaced char width in axes units
        ax.text(text_x, text_y, before, ha="left", va="center",
                fontsize=12.2, family="monospace", color=C_DARK, zorder=3)
        # Highlight slice
        hs, he = ex["hl_start"], ex["hl_end"]
        ax.add_patch(Rectangle(
            (text_x + hs * char_w - 0.04, text_y - 0.22),
            (he - hs) * char_w + 0.08, 0.44,
            facecolor=color, alpha=0.22, edgecolor=color, lw=1.0, zorder=2,
        ))
        # Bracket markers under highlight
        ax.text(text_x + hs * char_w - 0.05, text_y - 0.45, "[",
                fontsize=12, color=color, fontweight="bold", family="monospace")
        ax.text(text_x + he * char_w - 0.05, text_y - 0.45, "]",
                fontsize=12, color=color, fontweight="bold", family="monospace")

        # Note
        ax.text(0.95, y - 1.32, ex["note"],
                fontsize=10.2, color=C_DARK, style="italic")

    # Bottom legend
    ax.add_patch(FancyBboxPatch(
        (0.6, 0.25), 11.8, 0.7,
        boxstyle="round,pad=0.05", facecolor="white",
        edgecolor=C_DARK, lw=1.0,
    ))
    ax.text(0.95, 0.78, "i  vs  a:",
            fontsize=11, fontweight="bold", color=C_DARK)
    ax.text(2.05, 0.78,
            "i = inner (excludes delimiters)   a = around (includes delimiters)",
            fontsize=10.5, color=C_DARK)
    ax.text(0.95, 0.40,
            "Objects: w word  s sentence  p paragraph  \"  '  `  ( ) [ ] { } < >  t HTML tag",
            fontsize=10.3, color=C_GRAY, family="monospace")

    _save(fig, "fig3_text_objects")


# ---------------------------------------------------------------------------
# Figure 4 -- Vim file structure (config + plugins + state)
# ---------------------------------------------------------------------------
def fig4_vim_file_structure() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.5))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.5)

    ax.text(6.5, 8.05, "Where Vim Keeps Things",
            ha="center", va="center", fontsize=17, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.65, "config, plugins, runtime state -- and what each is for",
            ha="center", va="center", fontsize=11.2, style="italic", color=C_GRAY)

    # Tree column on the left
    root_x, root_y = 1.4, 7.0
    ax.add_patch(FancyBboxPatch(
        (root_x - 0.35, root_y - 0.25), 0.9, 0.5,
        boxstyle="round,pad=0.03", facecolor=C_DARK, edgecolor="none",
    ))
    ax.text(root_x + 0.1, root_y, "~/", ha="center", va="center",
            fontsize=12, fontweight="bold", color="white", family="monospace")

    entries = [
        # path,                role,                                   color, y
        (".vimrc",            "your config file (settings + maps)",  C_BLUE,   6.30),
        (".vim/",             "Vim's per-user runtime directory",    C_PURPLE, 5.75),
        (".vim/plugin/",      "auto-loaded plain Vim scripts",       C_PURPLE, 5.20),
        (".vim/colors/",      "color scheme files (*.vim)",          C_PURPLE, 4.65),
        (".vim/pack/.../",    "native package plugins (Vim 8+)",     C_PURPLE, 4.10),
        (".vim/swap/",        "swap files (.swp) -- crash recovery", C_AMBER,  3.55),
        (".vim/undo/",        "persistent undo history",             C_AMBER,  3.00),
        (".config/nvim/",     "Neovim config dir (init.vim/.lua)",   C_GREEN,  2.45),
    ]

    branch_x = 2.05
    label_x = 2.45

    ax.plot([root_x + 0.1, root_x + 0.1],
            [root_y - 0.25, entries[-1][3]], color=C_GRAY, lw=1.4, zorder=0)

    for path, role, color, y in entries:
        ax.plot([root_x + 0.1, branch_x], [y, y], color=C_GRAY, lw=1.2, zorder=0)
        ax.add_patch(FancyBboxPatch(
            (label_x, y - 0.22), 2.05, 0.44,
            boxstyle="round,pad=0.03", facecolor=color, edgecolor="none", alpha=0.95,
        ))
        ax.text(label_x + 1.025, y, path, ha="center", va="center",
                fontsize=10.2, color="white", fontweight="bold", family="monospace")
        ax.text(label_x + 2.2, y, role, ha="left", va="center",
                fontsize=9.7, color=C_DARK)

    # Right side: minimal .vimrc as a "code card"
    card_x, card_y = 7.5, 6.85
    card_w, card_h = 5.0, 5.4
    ax.add_patch(FancyBboxPatch(
        (card_x, card_y - card_h), card_w, card_h,
        boxstyle="round,pad=0.05", facecolor=C_DARK, edgecolor="none",
    ))
    # "filename" tab on the card
    ax.add_patch(FancyBboxPatch(
        (card_x + 0.2, card_y - 0.15), 1.55, 0.42,
        boxstyle="round,pad=0.02", facecolor=C_BLUE, edgecolor="none",
    ))
    ax.text(card_x + 0.97, card_y + 0.06, "~/.vimrc",
            ha="center", va="center", fontsize=10.2, fontweight="bold",
            color="white", family="monospace")

    vimrc_lines = [
        ('" ---- display ----',                 C_GRAY,   False),
        ("set number",                          C_LIGHT,  False),
        ("set relativenumber",                  C_LIGHT,  False),
        ("set cursorline",                      C_LIGHT,  False),
        ("syntax on",                           C_GREEN,  True),
        ("",                                    C_LIGHT,  False),
        ('" ---- indentation ----',             C_GRAY,   False),
        ("set expandtab",                       C_LIGHT,  False),
        ("set tabstop=4",                       C_LIGHT,  False),
        ("set shiftwidth=4",                    C_LIGHT,  False),
        ("",                                    C_LIGHT,  False),
        ('" ---- on save: trim trailing ws ----', C_GRAY,  False),
        ("autocmd BufWritePre *",               C_AMBER,  False),
        (" \\ :%s/\\s\\+$//e",                  C_AMBER,  False),
    ]
    line_y = card_y - 0.55
    for line, color, _ in vimrc_lines:
        ax.text(card_x + 0.3, line_y, line, ha="left", va="center",
                fontsize=10.2, color=color, family="monospace")
        line_y -= 0.32

    # Legend strip
    leg_y = 0.55
    ax.add_patch(FancyBboxPatch(
        (0.6, leg_y - 0.05), 11.8, 0.85,
        boxstyle="round,pad=0.05", facecolor=C_BG_SOFT,
        edgecolor=C_DARK, lw=1.0,
    ))
    items = [
        ("config",  C_BLUE),
        ("plugins / runtime", C_PURPLE),
        ("state (swap / undo)", C_AMBER),
        ("Neovim", C_GREEN),
    ]
    x = 1.0
    ax.text(x, leg_y + 0.55, "Legend:",
            fontsize=11, fontweight="bold", color=C_DARK)
    x = 2.0
    for label, color in items:
        ax.add_patch(Rectangle((x, leg_y + 0.42), 0.3, 0.3,
                               facecolor=color, edgecolor="none"))
        ax.text(x + 0.4, leg_y + 0.57, label,
                fontsize=10.2, color=C_DARK, va="center")
        x += 2.4
    ax.text(1.0, leg_y + 0.05,
            "Edit ~/.vimrc -> save -> open Vim again. There is no daemon to restart -- "
            "config is read at startup.",
            fontsize=10.2, color=C_GRAY, style="italic")

    _save(fig, "fig4_vim_file_structure")


# ---------------------------------------------------------------------------
# Figure 5 -- Common workflows (search/replace + multi-file)
# ---------------------------------------------------------------------------
def fig5_workflows() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.6))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.6)

    ax.text(6.5, 8.20, "Two Workflows You Will Use Every Day",
            ha="center", va="center", fontsize=17, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.85, "left: project-wide search & replace   |   right: edit multiple files at once",
            ha="center", va="center", fontsize=11, style="italic", color=C_GRAY)

    # ---- Left workflow: search & replace ----
    left_x = 0.7
    panel_w = 5.8
    panel_top = 7.35
    panel_h = 6.65

    ax.add_patch(FancyBboxPatch(
        (left_x, panel_top - panel_h), panel_w, panel_h,
        boxstyle="round,pad=0.05", facecolor=C_BG_SOFT,
        edgecolor=C_BLUE, lw=1.4,
    ))
    ax.text(left_x + panel_w / 2, panel_top - 0.3,
            "Workflow A -- Search & Replace",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_BLUE)

    steps_a = [
        ("1", "/oldName",        "search to verify the match before changing anything"),
        ("2", "n   N",           "step through hits with n (next) and N (prev)"),
        ("3", ":%s//newName/gc", "empty pattern reuses last search; gc = global + confirm"),
        ("4", "y  n  a  q",      "y = yes, n = skip, a = all, q = quit replacement"),
        ("5", ":noh",            "clear the leftover highlight when you're done"),
    ]
    sy = panel_top - 1.0
    for num, cmd, note in steps_a:
        # Step circle
        ax.add_patch(Circle((left_x + 0.55, sy), 0.27,
                            facecolor=C_BLUE, edgecolor="none"))
        ax.text(left_x + 0.55, sy, num, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        # Command box
        ax.add_patch(FancyBboxPatch(
            (left_x + 1.05, sy - 0.27), 4.5, 0.54,
            boxstyle="round,pad=0.03", facecolor="white",
            edgecolor=C_BLUE, lw=1.2,
        ))
        ax.text(left_x + 1.2, sy, cmd, ha="left", va="center",
                fontsize=11, family="monospace", color=C_DARK, fontweight="bold")
        # Note
        ax.text(left_x + 0.55, sy - 0.55, note, ha="left", va="center",
                fontsize=9.5, color=C_DARK, style="italic")
        sy -= 1.05

    # Bottom tip box
    tip_y = panel_top - panel_h + 0.55
    ax.add_patch(FancyBboxPatch(
        (left_x + 0.3, tip_y - 0.35), panel_w - 0.6, 0.75,
        boxstyle="round,pad=0.03", facecolor=C_BLUE, edgecolor="none", alpha=0.12,
    ))
    ax.text(left_x + 0.5, tip_y + 0.13, "Tip:",
            fontsize=10.3, fontweight="bold", color=C_BLUE)
    ax.text(left_x + 1.15, tip_y + 0.13,
            "narrow first  ->  :10,40s/foo/bar/g",
            fontsize=10.2, color=C_DARK, family="monospace")
    ax.text(left_x + 0.5, tip_y - 0.18,
            "trust nothing -- always run with c (confirm) on the first pass",
            fontsize=9.7, color=C_DARK, style="italic")

    # ---- Right workflow: multi-file ----
    right_x = 6.7
    ax.add_patch(FancyBboxPatch(
        (right_x, panel_top - panel_h), panel_w, panel_h,
        boxstyle="round,pad=0.05", facecolor=C_BG_SOFT,
        edgecolor=C_PURPLE, lw=1.4,
    ))
    ax.text(right_x + panel_w / 2, panel_top - 0.3,
            "Workflow B -- Multi-file Editing",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_PURPLE)

    steps_b = [
        ("1", ":e  src/api.py",     "open file 1 in current window (becomes a buffer)"),
        ("2", ":vsp src/main.py",   "vertical split with file 2 -- side by side"),
        ("3", ":sp  README.md",     "horizontal split with file 3 above current"),
        ("4", "Ctrl-w  h j k l",    "jump between windows in the four directions"),
        ("5", ":ls   :b api",       "list buffers, switch to one by name fragment"),
    ]
    sy = panel_top - 1.0
    for num, cmd, note in steps_b:
        ax.add_patch(Circle((right_x + 0.55, sy), 0.27,
                            facecolor=C_PURPLE, edgecolor="none"))
        ax.text(right_x + 0.55, sy, num, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.add_patch(FancyBboxPatch(
            (right_x + 1.05, sy - 0.27), 4.5, 0.54,
            boxstyle="round,pad=0.03", facecolor="white",
            edgecolor=C_PURPLE, lw=1.2,
        ))
        ax.text(right_x + 1.2, sy, cmd, ha="left", va="center",
                fontsize=11, family="monospace", color=C_DARK, fontweight="bold")
        ax.text(right_x + 0.55, sy - 0.55, note, ha="left", va="center",
                fontsize=9.5, color=C_DARK, style="italic")
        sy -= 1.05

    # Bottom mental-model box
    tip_y = panel_top - panel_h + 0.55
    ax.add_patch(FancyBboxPatch(
        (right_x + 0.3, tip_y - 0.35), panel_w - 0.6, 0.75,
        boxstyle="round,pad=0.03", facecolor=C_PURPLE, edgecolor="none", alpha=0.12,
    ))
    ax.text(right_x + 0.5, tip_y + 0.13, "Mental model:",
            fontsize=10.3, fontweight="bold", color=C_PURPLE)
    ax.text(right_x + 2.05, tip_y + 0.13,
            "buffer = file in memory   |   window = view onto a buffer",
            fontsize=10.0, color=C_DARK)
    ax.text(right_x + 0.5, tip_y - 0.18,
            "tabs are layouts of windows -- not files like in a browser",
            fontsize=9.7, color=C_DARK, style="italic")

    _save(fig, "fig5_workflows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Vim Essentials figures...")
    fig1_mode_state_diagram()
    fig2_motion_cheatsheet()
    fig3_text_objects()
    fig4_vim_file_structure()
    fig5_workflows()
    print("Done.")


if __name__ == "__main__":
    main()
