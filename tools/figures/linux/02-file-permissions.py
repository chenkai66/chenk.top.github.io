"""
Figure generation script for Linux Part 02: File Permissions.

Generates 5 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in editorial print style.

Figures:
    fig1_rwx_bits           Anatomy of the 10-character mode string and how
                            the same r/w/x bits mean different things on a
                            regular file vs a directory.
    fig2_chmod_notations    Numeric (755) vs symbolic (u+x) notation: same
                            target permissions reached two different ways,
                            with the bit-to-digit arithmetic spelled out.
    fig3_ugo_matrix         Owner / Group / Others permission matrix on a
                            shared project directory; shows decisions for
                            three concrete users (alice, bob, eve).
    fig4_special_bits       Special permission bits SUID, SGID, Sticky --
                            mode strings, numeric prefix, and a one-line
                            real-world example for each.
    fig5_acl_extension      Traditional UGO permissions vs ACL extension:
                            shows where ACL adds per-user / per-group
                            entries that classic mode bits cannot express.

Usage:
    python3 scripts/figures/linux/02-file-permissions.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
C_BLUE = COLORS["primary"]     # owner / read
C_PURPLE = COLORS["accent"]   # group / write
C_GREEN = COLORS["success"]    # others / execute / good
C_AMBER = COLORS["warning"]    # special / highlight
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG_SOFT = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "linux" / "file-permissions"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "linux" / "文件权限"


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
# Figure 1 -- The rwx bits: file mode anatomy + file-vs-directory semantics
# ---------------------------------------------------------------------------
def fig1_rwx_bits() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.2)
    _no_axis(ax)

    ax.text(
        6.5, 6.85, "Anatomy of the Linux mode string",
        ha="center", va="center", fontsize=15, fontweight="bold", color=C_DARK,
    )
    ax.text(
        6.5, 6.45, "Same three bits, different meaning on files vs directories",
        ha="center", va="center", fontsize=10.5, color=C_GRAY, style="italic",
    )

    # The mode string broken into 10 cells
    chars = ["-", "r", "w", "x", "r", "-", "x", "r", "-", "x"]
    colors = [C_GRAY, C_BLUE, C_BLUE, C_BLUE,
              C_PURPLE, C_PURPLE, C_PURPLE,
              C_GREEN, C_GREEN, C_GREEN]
    labels = ["type", "", "owner (u)", "", "", "group (g)", "", "", "others (o)", ""]

    cell_w = 0.85
    start_x = 6.5 - (10 * cell_w) / 2
    y_band = 5.2
    for i, (ch, col) in enumerate(zip(chars, colors)):
        x = start_x + i * cell_w
        ax.add_patch(FancyBboxPatch(
            (x, y_band), cell_w * 0.92, 0.95,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=col, edgecolor="white", linewidth=1.5,
        ))
        ax.text(x + cell_w * 0.46, y_band + 0.48, ch,
                ha="center", va="center", fontsize=20, fontweight="bold",
                color="white", family="monospace")

    # Group brackets / labels
    bracket_y = y_band - 0.35
    for grp_chars, grp_label, grp_col in [
        ((1, 4), "owner (u)", C_BLUE),
        ((4, 7), "group (g)", C_PURPLE),
        ((7, 10), "others (o)", C_GREEN),
    ]:
        x0 = start_x + grp_chars[0] * cell_w
        x1 = start_x + grp_chars[1] * cell_w - (cell_w - cell_w * 0.92)
        ax.plot([x0, x1], [bracket_y, bracket_y], color=grp_col, lw=2)
        ax.text((x0 + x1) / 2, bracket_y - 0.32, grp_label,
                ha="center", va="center", fontsize=10.5, color=grp_col,
                fontweight="bold")

    # type cell label
    ax.annotate(
        "file type\n('-' file, 'd' dir, 'l' link)",
        xy=(start_x + cell_w * 0.46, y_band + 0.95),
        xytext=(start_x - 0.5, y_band + 1.85),
        ha="center", fontsize=9, color=C_DARK,
        arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=1),
    )

    # Numeric annotation
    ax.text(start_x + 2 * cell_w, bracket_y - 0.95, "rwx = 7",
            ha="center", fontsize=10, color=C_BLUE, fontweight="bold",
            family="monospace")
    ax.text(start_x + 5 * cell_w, bracket_y - 0.95, "r-x = 5",
            ha="center", fontsize=10, color=C_PURPLE, fontweight="bold",
            family="monospace")
    ax.text(start_x + 8 * cell_w, bracket_y - 0.95, "r-x = 5",
            ha="center", fontsize=10, color=C_GREEN, fontweight="bold",
            family="monospace")
    ax.text(6.5, bracket_y - 1.45, "-rwxr-xr-x  =  755",
            ha="center", fontsize=12, fontweight="bold", color=C_DARK,
            family="monospace")

    # Two side-by-side cards: file vs directory
    card_w, card_h = 5.5, 2.05
    card_y = 0.25
    file_x = 0.6
    dir_x = 6.9
    rows_file = [
        ("r", "read file contents", "cat file.txt"),
        ("w", "modify file contents", "echo > file.txt"),
        ("x", "execute as a program", "./script.sh"),
    ]
    rows_dir = [
        ("r", "list filenames in dir", "ls dir/"),
        ("w", "create / delete entries (needs x)", "touch dir/new"),
        ("x", "enter and traverse the dir", "cd dir/"),
    ]

    for x0, title, rows, accent in [
        (file_x, "On a regular file", rows_file, C_BLUE),
        (dir_x, "On a directory", rows_dir, C_PURPLE),
    ]:
        ax.add_patch(FancyBboxPatch(
            (x0, card_y), card_w, card_h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=C_BG_SOFT, edgecolor=accent, linewidth=1.6,
        ))
        ax.text(x0 + 0.2, card_y + card_h - 0.28, title,
                ha="left", va="center", fontsize=11.5, fontweight="bold",
                color=accent)
        for i, (bit, meaning, example) in enumerate(rows):
            y = card_y + card_h - 0.75 - i * 0.45
            # bit chip
            ax.add_patch(FancyBboxPatch(
                (x0 + 0.2, y - 0.16), 0.34, 0.32,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=accent, edgecolor="none",
            ))
            ax.text(x0 + 0.37, y, bit, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white",
                    family="monospace")
            ax.text(x0 + 0.7, y, meaning, ha="left", va="center",
                    fontsize=10, color=C_DARK)
            ax.text(x0 + card_w - 0.2, y, example, ha="right", va="center",
                    fontsize=9, color=C_GRAY, family="monospace")

    fig.tight_layout()
    _save(fig, "fig1_rwx_bits")


# ---------------------------------------------------------------------------
# Figure 2 -- chmod: numeric (755) vs symbolic (u+x) notation
# ---------------------------------------------------------------------------
def fig2_chmod_notations() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.6)
    _no_axis(ax)

    ax.text(6.5, 6.25, "Two ways to say the same thing: 'chmod 755' vs 'chmod u=rwx,go=rx'",
            ha="center", va="center", fontsize=14, fontweight="bold", color=C_DARK)
    ax.text(6.5, 5.85, "Both reach -rwxr-xr-x; they differ in intent",
            ha="center", va="center", fontsize=10.5, color=C_GRAY, style="italic")

    # ---- Numeric column ----
    nx = 0.6
    nw = 5.7
    ax.add_patch(FancyBboxPatch(
        (nx, 0.4), nw, 5.0,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=C_BG_SOFT, edgecolor=C_BLUE, linewidth=1.6,
    ))
    ax.text(nx + 0.25, 5.1, "Numeric notation", fontsize=12,
            fontweight="bold", color=C_BLUE)
    ax.text(nx + 0.25, 4.78, "Absolute: overwrite all 9 bits at once",
            fontsize=9.5, color=C_GRAY, style="italic")

    # bit table r=4 w=2 x=1
    bits = [("r", 4, C_BLUE), ("w", 2, C_PURPLE), ("x", 1, C_GREEN)]
    bx = nx + 0.4
    by = 4.05
    for i, (b, v, c) in enumerate(bits):
        ax.add_patch(FancyBboxPatch(
            (bx + i * 0.95, by), 0.8, 0.55,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=c, edgecolor="none",
        ))
        ax.text(bx + i * 0.95 + 0.4, by + 0.36, b,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white", family="monospace")
        ax.text(bx + i * 0.95 + 0.4, by + 0.13, f"= {v}",
                ha="center", va="center", fontsize=9, color="white",
                family="monospace")

    # arithmetic
    ax.text(nx + 0.4, 3.55, "owner  rwx  = 4 + 2 + 1 = 7",
            fontsize=10.5, color=C_DARK, family="monospace")
    ax.text(nx + 0.4, 3.20, "group  r-x  = 4 + 0 + 1 = 5",
            fontsize=10.5, color=C_DARK, family="monospace")
    ax.text(nx + 0.4, 2.85, "others r-x  = 4 + 0 + 1 = 5",
            fontsize=10.5, color=C_DARK, family="monospace")

    # the command
    ax.add_patch(FancyBboxPatch(
        (nx + 0.4, 1.95), nw - 0.8, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=C_DARK, edgecolor="none",
    ))
    ax.text(nx + nw / 2, 2.22, "$ chmod 755 script.sh",
            ha="center", va="center", fontsize=12, color="white",
            family="monospace", fontweight="bold")

    # use case
    ax.text(nx + 0.4, 1.40, "Use it when:", fontsize=10,
            fontweight="bold", color=C_DARK)
    ax.text(nx + 0.4, 1.05,
            "  - you want a known final state",
            fontsize=9.5, color=C_DARK)
    ax.text(nx + 0.4, 0.75,
            "  - scripting deploys, fresh files",
            fontsize=9.5, color=C_DARK)

    # ---- Symbolic column ----
    sx = 6.7
    sw = 5.7
    ax.add_patch(FancyBboxPatch(
        (sx, 0.4), sw, 5.0,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=C_BG_SOFT, edgecolor=C_PURPLE, linewidth=1.6,
    ))
    ax.text(sx + 0.25, 5.1, "Symbolic notation", fontsize=12,
            fontweight="bold", color=C_PURPLE)
    ax.text(sx + 0.25, 4.78, "Relative: tweak just the bits you mention",
            fontsize=9.5, color=C_GRAY, style="italic")

    # who chips
    chips = [("u", "owner", C_BLUE), ("g", "group", C_PURPLE),
             ("o", "others", C_GREEN), ("a", "all", C_AMBER)]
    cx = sx + 0.4
    for i, (k, name, c) in enumerate(chips):
        ax.add_patch(FancyBboxPatch(
            (cx + i * 1.25, 4.05), 1.05, 0.55,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=c, edgecolor="none",
        ))
        ax.text(cx + i * 1.25 + 0.52, 4.42, k,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white", family="monospace")
        ax.text(cx + i * 1.25 + 0.52, 4.18, name,
                ha="center", va="center", fontsize=8.5, color="white")

    # operators
    ax.text(sx + 0.4, 3.55, "operators:  + add    - remove    = set",
            fontsize=10, color=C_DARK, family="monospace")
    ax.text(sx + 0.4, 3.20, "u+x   add execute for owner only",
            fontsize=10, color=C_DARK, family="monospace")
    ax.text(sx + 0.4, 2.85, "go-w  remove write from group + others",
            fontsize=10, color=C_DARK, family="monospace")

    ax.add_patch(FancyBboxPatch(
        (sx + 0.4, 1.95), sw - 0.8, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=C_DARK, edgecolor="none",
    ))
    ax.text(sx + sw / 2, 2.22, "$ chmod u=rwx,go=rx script.sh",
            ha="center", va="center", fontsize=12, color="white",
            family="monospace", fontweight="bold")

    ax.text(sx + 0.4, 1.40, "Use it when:", fontsize=10,
            fontweight="bold", color=C_DARK)
    ax.text(sx + 0.4, 1.05,
            "  - tweaking one dimension only",
            fontsize=9.5, color=C_DARK)
    ax.text(sx + 0.4, 0.75,
            "  - 'chmod -R u+rwX,go+rX' is safe on trees",
            fontsize=9.5, color=C_DARK)

    fig.tight_layout()
    _save(fig, "fig2_chmod_notations")


# ---------------------------------------------------------------------------
# Figure 3 -- Owner / Group / Others permission matrix
# ---------------------------------------------------------------------------
def fig3_ugo_matrix() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.8)
    _no_axis(ax)

    ax.text(6.5, 6.45,
            "Who can do what: the owner / group / others decision table",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DARK)
    ax.text(6.5, 6.05,
            "Target: /srv/project owned by alice:developers, mode 2770 "
            "(rwxrws---)",
            ha="center", va="center", fontsize=10.5, color=C_GRAY,
            style="italic", family="monospace")

    # Three users
    users = [
        ("alice", "owner",      C_BLUE,   "uid=alice  gid=developers"),
        ("bob",   "group member", C_PURPLE, "uid=bob    gid=developers"),
        ("eve",   "outsider",   C_RED,    "uid=eve    gid=marketing"),
    ]

    # Matrix: rows = users, cols = actions
    actions = [
        ("ls /srv/project",     [True,  True,  False]),
        ("cd /srv/project",     [True,  True,  False]),
        ("touch newfile",       [True,  True,  False]),
        ("cat shared.conf",     [True,  True,  False]),
        ("rm bobs_file.txt",    [True,  False, False]),  # bob owns it
        ("chown :other dir",    [True,  False, False]),
    ]

    # Layout
    left = 0.6
    top = 5.55
    user_col_w = 2.7
    cell_w = 1.55
    cell_h = 0.55

    # Header row: actions
    ax.add_patch(Rectangle(
        (left, top - cell_h), user_col_w, cell_h,
        facecolor=C_DARK, edgecolor="white",
    ))
    ax.text(left + user_col_w / 2, top - cell_h / 2, "User  /  action",
            ha="center", va="center", fontsize=10, color="white",
            fontweight="bold")
    for j, (act, _) in enumerate(actions):
        x = left + user_col_w + j * cell_w
        ax.add_patch(Rectangle(
            (x, top - cell_h), cell_w, cell_h,
            facecolor=C_DARK, edgecolor="white",
        ))
        ax.text(x + cell_w / 2, top - cell_h / 2, act,
                ha="center", va="center", fontsize=8.5, color="white",
                family="monospace")

    # Body rows
    for i, (uname, role, ucol, uid_str) in enumerate(users):
        y = top - cell_h - (i + 1) * cell_h
        # user cell
        ax.add_patch(Rectangle(
            (left, y), user_col_w, cell_h,
            facecolor=ucol, edgecolor="white",
        ))
        ax.text(left + 0.15, y + cell_h * 0.66, uname,
                ha="left", va="center", fontsize=10.5,
                color="white", fontweight="bold")
        ax.text(left + 0.15, y + cell_h * 0.28, role,
                ha="left", va="center", fontsize=8.5, color="white",
                style="italic")

        for j, (_, allowed) in enumerate(actions):
            x = left + user_col_w + j * cell_w
            ok = allowed[i]
            face = "#dcfce7" if ok else "#fee2e2"
            edge = C_GREEN if ok else C_RED
            mark = "OK" if ok else "DENY"
            ax.add_patch(Rectangle(
                (x, y), cell_w, cell_h,
                facecolor=face, edgecolor="white",
            ))
            ax.add_patch(FancyBboxPatch(
                (x + cell_w / 2 - 0.34, y + cell_h * 0.18),
                0.68, cell_h * 0.62,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=edge, edgecolor="none",
            ))
            ax.text(x + cell_w / 2, y + cell_h / 2, mark,
                    ha="center", va="center", fontsize=9.5,
                    color="white", fontweight="bold", family="monospace")

    # Legend / key takeaways below
    bottom_y = 2.05
    ax.add_patch(FancyBboxPatch(
        (0.6, 0.3), 12.0, bottom_y - 0.3,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=C_BG_SOFT, edgecolor=C_LIGHT, linewidth=1.0,
    ))
    ax.text(0.85, bottom_y - 0.25, "How the kernel decides", fontsize=11,
            fontweight="bold", color=C_DARK)

    bullets = [
        ("1.  Is the caller the OWNER (uid match)?  -> use owner bits, stop.",
         C_BLUE),
        ("2.  Else, does the caller belong to the file's GROUP?  "
         "-> use group bits, stop.",
         C_PURPLE),
        ("3.  Else, fall through to OTHERS bits.",
         C_GREEN),
        ("    NOTE: root bypasses these checks (CAP_DAC_OVERRIDE).  "
         "Sticky bit on a directory is the exception that limits delete to "
         "the file's owner.",
         C_AMBER),
    ]
    for i, (txt, col) in enumerate(bullets):
        ax.text(0.95, bottom_y - 0.6 - i * 0.32, txt,
                fontsize=9.7, color=col,
                family="DejaVu Sans Mono" if txt.startswith("    ") else None)

    fig.tight_layout()
    _save(fig, "fig3_ugo_matrix")


# ---------------------------------------------------------------------------
# Figure 4 -- Special permission bits: SUID, SGID, Sticky
# ---------------------------------------------------------------------------
def fig4_special_bits() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.6)
    _no_axis(ax)

    ax.text(6.5, 6.25,
            "The fourth digit: SUID, SGID, Sticky bit",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DARK)
    ax.text(6.5, 5.85,
            "chmod takes a 4-digit number; the leading digit packs three "
            "special bits",
            ha="center", va="center", fontsize=10.5, color=C_GRAY,
            style="italic")

    cards = [
        {
            "title": "SUID  (4xxx)",
            "where": "on an executable file",
            "mode":  "-rwsr-xr-x",
            "color": C_BLUE,
            "what":  "process runs with the file OWNER's effective uid,\n"
                     "not the caller's",
            "why":   "lets unprivileged users perform a small, audited\n"
                     "set of root operations",
            "ex":    "/usr/bin/passwd  edits /etc/shadow as root",
            "set":   "chmod u+s prog        # or  chmod 4755 prog",
            "find":  "find / -perm -4000 -type f 2>/dev/null",
        },
        {
            "title": "SGID  (2xxx)",
            "where": "on a directory (or executable)",
            "mode":  "drwxrws---",
            "color": C_PURPLE,
            "what":  "files created in the dir inherit the dir's GROUP\n"
                     "instead of the creator's primary group",
            "why":   "shared team folders without manual chgrp every time",
            "ex":    "chmod 2770 /srv/project   # team-only project dir",
            "set":   "chmod g+s dir         # or  chmod 2770 dir",
            "find":  "find / -perm -2000 -type d 2>/dev/null",
        },
        {
            "title": "Sticky  (1xxx)",
            "where": "on a world-writable directory",
            "mode":  "drwxrwxrwt",
            "color": C_AMBER,
            "what":  "only the file's OWNER (or root) may unlink or rename\n"
                     "entries inside, even though everyone has 'w'",
            "why":   "/tmp must let anyone create files, but A must not\n"
                     "be able to nuke B's files",
            "ex":    "/tmp is mode 1777 by default",
            "set":   "chmod +t dir          # or  chmod 1777 dir",
            "find":  "find / -perm -1000 -type d 2>/dev/null",
        },
    ]

    card_w = 4.05
    card_h = 5.0
    gap = 0.15
    total_w = card_w * 3 + gap * 2
    start_x = (13 - total_w) / 2

    for i, card in enumerate(cards):
        x0 = start_x + i * (card_w + gap)
        y0 = 0.4
        col = card["color"]
        # Card background
        ax.add_patch(FancyBboxPatch(
            (x0, y0), card_w, card_h,
            boxstyle="round,pad=0.04,rounding_size=0.14",
            facecolor=C_BG_SOFT, edgecolor=col, linewidth=1.6,
        ))
        # Header band
        ax.add_patch(FancyBboxPatch(
            (x0, y0 + card_h - 0.85), card_w, 0.85,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=col, edgecolor="none",
        ))
        ax.text(x0 + 0.2, y0 + card_h - 0.32, card["title"],
                fontsize=12.5, color="white", fontweight="bold",
                family="monospace")
        ax.text(x0 + 0.2, y0 + card_h - 0.68, card["where"],
                fontsize=9.5, color="white", style="italic")

        # Mode chip
        ax.add_patch(FancyBboxPatch(
            (x0 + 0.2, y0 + card_h - 1.45), card_w - 0.4, 0.45,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=C_DARK, edgecolor="none",
        ))
        ax.text(x0 + card_w / 2, y0 + card_h - 1.22, card["mode"],
                ha="center", va="center", fontsize=12,
                color="white", fontweight="bold", family="monospace")

        # Sections
        sections = [
            ("WHAT",  card["what"]),
            ("WHY",   card["why"]),
            ("EXAMPLE", card["ex"]),
            ("SET",   card["set"]),
            ("AUDIT", card["find"]),
        ]
        y_cursor = y0 + card_h - 1.65
        for label, body in sections:
            y_cursor -= 0.18
            ax.text(x0 + 0.2, y_cursor, label,
                    fontsize=8.5, color=col, fontweight="bold")
            y_cursor -= 0.2
            mono = label in ("EXAMPLE", "SET", "AUDIT", "MODE")
            for line in body.split("\n"):
                ax.text(x0 + 0.2, y_cursor, line,
                        fontsize=8.7, color=C_DARK,
                        family="DejaVu Sans Mono" if mono else None)
                y_cursor -= 0.22
            y_cursor -= 0.04

    fig.tight_layout()
    _save(fig, "fig4_special_bits")


# ---------------------------------------------------------------------------
# Figure 5 -- ACL extension over classic UGO
# ---------------------------------------------------------------------------
def fig5_acl_extension() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.8)
    _no_axis(ax)

    ax.text(6.5, 6.45,
            "When 3 buckets aren't enough: ACL extends the classic UGO model",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=C_DARK)
    ax.text(6.5, 6.05,
            "Same file, two views: traditional mode bits vs full ACL listing",
            ha="center", va="center", fontsize=10.5, color=C_GRAY,
            style="italic")

    # ----- Left card: classic UGO -----
    lx = 0.6
    lw = 5.7
    ly = 0.45
    lh = 5.25
    ax.add_patch(FancyBboxPatch(
        (lx, ly), lw, lh,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=C_BG_SOFT, edgecolor=C_BLUE, linewidth=1.6,
    ))
    ax.text(lx + 0.25, ly + lh - 0.32,
            "Classic UGO   (ls -l)",
            fontsize=12, fontweight="bold", color=C_BLUE)
    ax.text(lx + 0.25, ly + lh - 0.62,
            "exactly 3 buckets, 9 bits",
            fontsize=9.5, color=C_GRAY, style="italic")

    ax.add_patch(FancyBboxPatch(
        (lx + 0.25, ly + lh - 1.4), lw - 0.5, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=C_DARK, edgecolor="none",
    ))
    ax.text(lx + lw / 2, ly + lh - 1.13,
            "-rw-r----- alice developers report.csv",
            ha="center", va="center", fontsize=10.5,
            color="white", family="monospace", fontweight="bold")

    rows = [
        ("alice (owner)",       "rw-",  C_BLUE, True),
        ("anyone in 'developers'", "r--", C_PURPLE, True),
        ("everyone else",        "---",  C_GRAY, False),
    ]
    y = ly + lh - 2.15
    for who, bits, col, ok in rows:
        ax.add_patch(FancyBboxPatch(
            (lx + 0.4, y - 0.18), lw - 0.8, 0.42,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="white", edgecolor=col, linewidth=1.0,
        ))
        ax.text(lx + 0.55, y, who, fontsize=10, color=C_DARK)
        ax.text(lx + lw - 0.55, y, bits, fontsize=11,
                color=col, fontweight="bold", family="monospace",
                ha="right", va="center")
        y -= 0.55

    # Pain point box
    ax.add_patch(FancyBboxPatch(
        (lx + 0.25, ly + 0.3), lw - 0.5, 1.7,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        facecolor="#fff7ed", edgecolor=C_AMBER, linewidth=1.0,
    ))
    ax.text(lx + 0.45, ly + 1.78,
            "But what if you also need...",
            fontsize=10, color=C_AMBER, fontweight="bold")
    ax.text(lx + 0.45, ly + 1.45,
            "  -  let auditor 'eve' read it (she's not a dev)",
            fontsize=9.5, color=C_DARK)
    ax.text(lx + 0.45, ly + 1.15,
            "  -  let group 'qa' read AND write",
            fontsize=9.5, color=C_DARK)
    ax.text(lx + 0.45, ly + 0.85,
            "  -  block one specific dev 'mallory'",
            fontsize=9.5, color=C_DARK)
    ax.text(lx + 0.45, ly + 0.5,
            "Three buckets cannot express any of these.",
            fontsize=9.5, color=C_RED, style="italic", fontweight="bold")

    # ----- Arrow to right card -----
    ax.add_patch(FancyArrowPatch(
        (lx + lw + 0.05, 3.4), (6.85, 3.4),
        arrowstyle="->,head_length=8,head_width=6",
        color=C_DARK, linewidth=2.0, mutation_scale=14,
    ))
    ax.text((lx + lw + 0.05 + 6.85) / 2, 3.65,
            "setfacl",
            ha="center", fontsize=10, color=C_DARK,
            fontweight="bold", family="monospace")

    # ----- Right card: ACL -----
    rx = 6.95
    rw = 5.45
    ry = 0.45
    rh = 5.25
    ax.add_patch(FancyBboxPatch(
        (rx, ry), rw, rh,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=C_BG_SOFT, edgecolor=C_PURPLE, linewidth=1.6,
    ))
    ax.text(rx + 0.25, ry + rh - 0.32,
            "ACL extension   (getfacl)",
            fontsize=12, fontweight="bold", color=C_PURPLE)
    ax.text(rx + 0.25, ry + rh - 0.62,
            "arbitrary number of per-user / per-group entries",
            fontsize=9.5, color=C_GRAY, style="italic")

    # the ls -l + symbol indicator
    ax.add_patch(FancyBboxPatch(
        (rx + 0.25, ry + rh - 1.4), rw - 0.5, 0.55,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        facecolor=C_DARK, edgecolor="none",
    ))
    ax.text(rx + rw / 2, ry + rh - 1.13,
            "-rw-r-----+ alice developers report.csv",
            ha="center", va="center", fontsize=10.5,
            color="white", family="monospace", fontweight="bold")
    ax.text(rx + 0.25, ry + rh - 1.65,
            "the trailing '+' means: ACL entries present",
            fontsize=8.8, color=C_GRAY, style="italic")

    acl_rows = [
        ("user::",          "rw-",  C_BLUE,   "owner alice"),
        ("user:eve:",       "r--",  C_GREEN,  "auditor (extra)"),
        ("user:mallory:",   "---",  C_RED,    "blocked (extra)"),
        ("group::",         "r--",  C_PURPLE, "developers"),
        ("group:qa:",       "rw-",  C_GREEN,  "QA team (extra)"),
        ("mask::",          "rw-",  C_GRAY,   "ceiling for extras"),
        ("other::",         "---",  C_GRAY,   "everyone else"),
    ]
    y = ry + rh - 2.15
    for who, bits, col, note in acl_rows:
        ax.text(rx + 0.4, y, who, fontsize=9.5, color=col,
                family="monospace", fontweight="bold")
        ax.text(rx + 1.95, y, bits, fontsize=10, color=col,
                family="monospace", fontweight="bold")
        ax.text(rx + 2.7, y, note, fontsize=9, color=C_DARK)
        y -= 0.34

    fig.tight_layout()
    _save(fig, "fig5_acl_extension")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for Linux Part 02: File Permissions ...")
    fig1_rwx_bits()
    fig2_chmod_notations()
    fig3_ugo_matrix()
    fig4_special_bits()
    fig5_acl_extension()
    print("Done.")


if __name__ == "__main__":
    main()
