"""
Figure generation script for Linux Part 01: Basics.

Generates 5 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in editorial print style.

Figures:
    fig1_directory_tree     FHS directory tree showing /, /etc, /var, /home,
                            /usr, /opt, /tmp, /dev, /proc, /sys, /boot etc.
                            with one-line role labels.
    fig2_command_anatomy    Anatomy of a shell command: command + options +
                            arguments, decomposed with colour-coded callouts
                            on a real example (`ls -lah /var/log`).
    fig3_command_cheatsheet Common commands organized by category
                            (navigation, viewing, editing, search, info,
                             permissions, processes, network) -- card grid.
    fig4_pipeline_flow      Pipeline data flow: stdin/stdout streaming
                            through `cat | grep | sort | head` with byte
                            arrows and intermediate buffers.
    fig5_distro_family_tree Distro lineage: Debian / Red Hat / Arch / SUSE /
                            independent branches, with year markers and
                            modern descendants (Ubuntu, Rocky, Manjaro...).

Usage:
    python3 scripts/figures/linux/01-basics.py

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

    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
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
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "linux" / "basics"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "linux" / "basics"


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
# Figure 1 -- Linux directory tree (FHS)
# ---------------------------------------------------------------------------
def fig1_directory_tree() -> None:
    fig, ax = plt.subplots(figsize=(13, 9))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)

    # Root node at top-left
    root_x, root_y = 1.6, 8.3
    ax.add_patch(FancyBboxPatch(
        (root_x - 0.6, root_y - 0.32), 1.2, 0.64,
        boxstyle="round,pad=0.04", facecolor=C_DARK, edgecolor="none",
    ))
    ax.text(root_x, root_y, "/", ha="center", va="center",
            fontsize=18, fontweight="bold", color="white", family="monospace")

    # Children: (path, role, color, y position)
    entries = [
        ("/etc",    "system-wide config files", C_BLUE,    7.55),
        ("/home",   "regular user home dirs",  C_BLUE,    7.05),
        ("/root",   "root user home dir",      C_BLUE,    6.55),
        ("/var",    "variable data, logs, mail", C_PURPLE, 6.05),
        ("/usr",    "user programs and libs",  C_PURPLE,  5.55),
        ("/opt",    "third-party large packages", C_PURPLE, 5.05),
        ("/tmp",    "temp files (wiped on boot)", C_AMBER, 4.55),
        ("/boot",   "kernel + bootloader",     C_AMBER,   4.05),
        ("/dev",    "device files (disks, tty)", C_GREEN, 3.55),
        ("/proc",   "process info (virtual fs)", C_GREEN, 3.05),
        ("/sys",    "kernel + hardware info",  C_GREEN,   2.55),
        ("/lib",    "shared libraries (.so)",  C_GRAY,    2.05),
        ("/mnt",    "manual mount points",     C_GRAY,    1.55),
        ("/media",  "auto mount (USB, CD)",    C_GRAY,    1.05),
    ]

    branch_x = 3.2
    label_x = 3.6

    # Trunk vertical line from root down to the last child level
    ax.plot([root_x, root_x], [root_y - 0.32, entries[-1][3]],
            color=C_GRAY, lw=1.4, zorder=0)

    for path, role, color, y in entries:
        # horizontal connector
        ax.plot([root_x, branch_x], [y, y], color=C_GRAY, lw=1.2, zorder=0)
        # path label box
        ax.add_patch(FancyBboxPatch(
            (label_x, y - 0.22), 1.6, 0.44,
            boxstyle="round,pad=0.03", facecolor=color, edgecolor="none", alpha=0.95,
        ))
        ax.text(label_x + 0.8, y, path, ha="center", va="center",
                fontsize=11, color="white", fontweight="bold", family="monospace")
        # role description
        ax.text(label_x + 1.85, y, role, ha="left", va="center",
                fontsize=10.5, color=C_DARK)

    # Subnotes for /var and /usr expansions on the right side
    sub_x = 9.5
    sub_entries = [
        ("/var/log",   "rotating service logs",      6.05, C_PURPLE),
        ("/var/www",   "web server document roots",  5.65, C_PURPLE),
        ("/usr/bin",   "most user-facing commands",  5.20, C_PURPLE),
        ("/usr/local", "manually installed software", 4.80, C_PURPLE),
    ]
    # bracket from /var down to /usr/local on the right
    ax.plot([sub_x - 0.25, sub_x - 0.25], [4.65, 6.20], color=C_GRAY, lw=1.0)
    for path, role, y, color in sub_entries:
        ax.plot([sub_x - 0.25, sub_x - 0.05], [y, y], color=C_GRAY, lw=1.0)
        ax.text(sub_x, y, path, ha="left", va="center",
                fontsize=10, color=color, family="monospace", fontweight="bold")
        ax.text(sub_x + 1.6, y, role, ha="left", va="center",
                fontsize=9.5, color=C_DARK)

    # Title and subtitle
    ax.text(6.5, 8.75, "Linux Filesystem Hierarchy (FHS)",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(6.5, 8.42, "Everything hangs off a single root '/' -- no drive letters",
            ha="center", va="center", fontsize=10.5, style="italic", color=C_GRAY)

    # Legend by colour
    legend_y = 0.45
    legend_items = [
        ("config",   C_BLUE),
        ("data",     C_PURPLE),
        ("runtime",  C_AMBER),
        ("kernel",   C_GREEN),
        ("misc",     C_GRAY),
    ]
    lx = 1.0
    for name, color in legend_items:
        ax.add_patch(Rectangle((lx, legend_y - 0.12), 0.3, 0.24,
                               facecolor=color, edgecolor="none"))
        ax.text(lx + 0.4, legend_y, name, ha="left", va="center",
                fontsize=10, color=C_DARK)
        lx += 1.8

    plt.tight_layout()
    _save(fig, "fig1_directory_tree")


# ---------------------------------------------------------------------------
# Figure 2 -- Anatomy of a shell command
# ---------------------------------------------------------------------------
def fig2_command_anatomy() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.5))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.5)

    # Title
    ax.text(6.5, 6.1, "Anatomy of a Shell Command",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(6.5, 5.75, "command + options + arguments -- separated by spaces, parsed left to right",
            ha="center", va="center", fontsize=10.5, style="italic", color=C_GRAY)

    # The command line: a dark "terminal" panel
    panel = FancyBboxPatch((0.6, 3.4), 11.8, 1.4,
                           boxstyle="round,pad=0.04",
                           facecolor=C_DARK, edgecolor="none")
    ax.add_patch(panel)
    # Prompt
    ax.text(1.0, 4.1, "$", ha="left", va="center", fontsize=18,
            color=C_GREEN, fontweight="bold", family="monospace")

    # Tokens with their bounds: (text, x_center, color)
    tokens = [
        ("ls",            2.05, C_BLUE),
        ("-l",            3.05, C_PURPLE),
        ("-a",            3.85, C_PURPLE),
        ("-h",            4.65, C_PURPLE),
        ("--color=auto",  6.20, C_AMBER),
        ("/var/log",      8.20, C_GREEN),
        ("/etc",          9.55, C_GREEN),
    ]
    for text, x, color in tokens:
        ax.text(x, 4.1, text, ha="center", va="center",
                fontsize=15, color=color, family="monospace", fontweight="bold")

    # Callouts
    callouts = [
        # (label, sub, x_token, y_box, color)
        ("command",        "the program to run",                 2.05, 1.6, C_BLUE),
        ("short options",  "single-letter flags, can stack",     3.85, 1.6, C_PURPLE),
        ("long option",    "verbose form: --name=value",         6.20, 1.6, C_AMBER),
        ("arguments",      "what the command operates on",       8.85, 1.6, C_GREEN),
    ]
    for label, sub, x, y_box, color in callouts:
        # vertical connector line
        ax.plot([x, x], [3.4, y_box + 0.6], color=color, lw=1.2,
                linestyle="--", alpha=0.7)
        # box
        box = FancyBboxPatch((x - 1.1, y_box - 0.3), 2.2, 0.9,
                             boxstyle="round,pad=0.04",
                             facecolor="white", edgecolor=color, linewidth=1.6)
        ax.add_patch(box)
        ax.text(x, y_box + 0.32, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)
        ax.text(x, y_box - 0.02, sub, ha="center", va="center",
                fontsize=8.8, color=C_DARK)

    # Footer notes
    ax.text(0.6, 0.55, "Tips:", ha="left", va="center",
            fontsize=10.5, fontweight="bold", color=C_DARK)
    ax.text(1.4, 0.55,
            "ls -lah  ==  ls -l -a -h    .    quote paths with spaces: \"/var/log my dir\"    .    "
            "use --help or man <cmd> to discover options",
            ha="left", va="center", fontsize=10, color=C_DARK)

    plt.tight_layout()
    _save(fig, "fig2_command_anatomy")


# ---------------------------------------------------------------------------
# Figure 3 -- Common commands by category (cheatsheet card grid)
# ---------------------------------------------------------------------------
def fig3_command_cheatsheet() -> None:
    fig, ax = plt.subplots(figsize=(14, 9))
    _no_axis(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)

    # Title
    ax.text(7.0, 8.55, "Essential Linux Commands by Category",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(7.0, 8.22, "Memorise these eight cards and you can navigate any Linux box",
            ha="center", va="center", fontsize=10.5, style="italic", color=C_GRAY)

    # Eight cards in a 4x2 grid
    cards = [
        # (title, color, commands)
        ("Navigate", C_BLUE, [
            "pwd            print working dir",
            "cd /path       change directory",
            "cd ~ / cd ..   home / up one",
            "cd -           previous dir",
        ]),
        ("List & inspect", C_BLUE, [
            "ls -lah        long, all, human size",
            "tree -L 2      visual tree (2 levels)",
            "stat file      timestamps, inode",
            "file foo       detect true file type",
        ]),
        ("View content", C_PURPLE, [
            "cat f          dump full file",
            "less f         pageable viewer (q quit)",
            "head -n 20 f   first 20 lines",
            "tail -f log    follow appended lines",
        ]),
        ("Create & edit", C_PURPLE, [
            "mkdir -p a/b   create nested dirs",
            "touch f        empty file or touch mtime",
            "echo x > f     overwrite ; >> appends",
            "vim / nano f   interactive editor",
        ]),
        ("Copy / move / delete", C_AMBER, [
            "cp -r src dst  recursive copy",
            "mv old new     rename or move",
            "rm -i f        prompt before delete",
            "rm -rf dir     force recursive  (danger)",
        ]),
        ("Search", C_AMBER, [
            "find . -name '*.log'    by name",
            "find . -mtime -1        modified <1d",
            "grep -rIn 'TODO' .      recursive grep",
            "which / type cmd        locate binary",
        ]),
        ("System info", C_GREEN, [
            "uname -a       kernel + arch",
            "df -h / du -sh disk free / size",
            "free -h        memory usage",
            "uptime / who   load + sessions",
        ]),
        ("Processes & net", C_GREEN, [
            "ps aux | grep  filter processes",
            "top / htop     live monitor",
            "kill -9 PID    force terminate",
            "ip a / ss -tlnp  iface / open ports",
        ]),
    ]

    cw, ch = 3.2, 1.85
    gap_x, gap_y = 0.18, 0.30
    x0, y0 = 0.5, 5.95
    for i, (title, color, lines) in enumerate(cards):
        col = i % 4
        row = i // 4
        x = x0 + col * (cw + gap_x)
        y = y0 - row * (ch + gap_y)
        # card body
        ax.add_patch(FancyBboxPatch((x, y), cw, ch,
                                    boxstyle="round,pad=0.04",
                                    facecolor="white", edgecolor=color, linewidth=1.6))
        # header strip
        ax.add_patch(FancyBboxPatch((x, y + ch - 0.42), cw, 0.42,
                                    boxstyle="round,pad=0.02",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + 0.15, y + ch - 0.21, title, ha="left", va="center",
                fontsize=11.5, color="white", fontweight="bold")
        # lines
        for j, ln in enumerate(lines):
            ax.text(x + 0.13, y + ch - 0.68 - j * 0.30, ln,
                    ha="left", va="center", fontsize=9.0,
                    color=C_DARK, family="monospace")

    # Bottom hint bar
    ax.add_patch(FancyBboxPatch((0.5, 0.45), 13.0, 0.55,
                                boxstyle="round,pad=0.04",
                                facecolor=C_BG_SOFT, edgecolor=C_LIGHT, linewidth=1.0))
    ax.text(7.0, 0.72,
            "Discover more:  man <cmd>  .  <cmd> --help  .  apropos <topic>  .  history | grep <cmd>",
            ha="center", va="center", fontsize=10.5, color=C_DARK, family="monospace")

    plt.tight_layout()
    _save(fig, "fig3_command_cheatsheet")


# ---------------------------------------------------------------------------
# Figure 4 -- Pipeline data flow
# ---------------------------------------------------------------------------
def fig4_pipeline_flow() -> None:
    fig, ax = plt.subplots(figsize=(14, 7.2))
    _no_axis(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)

    ax.text(7.0, 6.85, "Pipeline: Each Stage Streams Its stdout into the Next stdin",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(7.0, 6.50, "cat /var/log/syslog  |  grep ERROR  |  awk '{print $5}'  |  sort | uniq -c | sort -rn | head",
            ha="center", va="center", fontsize=11, color=C_GRAY,
            family="monospace", style="italic")

    # Stages
    stages = [
        ("cat",      "read whole file\nas a stream",       C_BLUE),
        ("grep",     "keep lines\nmatching pattern",       C_PURPLE),
        ("awk",      "extract\ncolumn $5",                 C_AMBER),
        ("sort | uniq -c", "group + count\nidentical lines", C_GREEN),
        ("head",     "keep top\nN entries",                C_BLUE),
    ]
    n = len(stages)
    box_w = 1.9
    box_h = 1.6
    gap = 0.55
    total_w = n * box_w + (n - 1) * gap
    x_start = (14 - total_w) / 2
    y_box = 3.6

    centers = []
    for i, (name, sub, color) in enumerate(stages):
        x = x_start + i * (box_w + gap)
        ax.add_patch(FancyBboxPatch((x, y_box), box_w, box_h,
                                    boxstyle="round,pad=0.04",
                                    facecolor=color, edgecolor="none", alpha=0.95))
        ax.text(x + box_w / 2, y_box + box_h - 0.42, name,
                ha="center", va="center", fontsize=13,
                fontweight="bold", color="white", family="monospace")
        ax.text(x + box_w / 2, y_box + 0.55, sub,
                ha="center", va="center", fontsize=9.5, color="white")
        centers.append((x + box_w / 2, x + box_w))

    # Pipe arrows + buffer labels between stages
    for i in range(n - 1):
        x_from = x_start + i * (box_w + gap) + box_w
        x_to = x_start + (i + 1) * (box_w + gap)
        y_mid = y_box + box_h / 2
        arrow = FancyArrowPatch((x_from + 0.02, y_mid), (x_to - 0.02, y_mid),
                                arrowstyle="-|>", mutation_scale=18,
                                color=C_DARK, lw=1.8)
        ax.add_patch(arrow)
        ax.text((x_from + x_to) / 2, y_mid + 0.30, "|",
                ha="center", va="center", fontsize=18,
                fontweight="bold", color=C_DARK, family="monospace")
        ax.text((x_from + x_to) / 2, y_mid - 0.32, "buffered",
                ha="center", va="center", fontsize=8.0,
                color=C_GRAY, style="italic")

    # stdin label on far left
    ax.annotate("", xy=(x_start - 0.05, y_box + box_h / 2),
                xytext=(x_start - 0.85, y_box + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.8))
    ax.text(x_start - 0.85, y_box + box_h / 2 + 0.32, "stdin",
            ha="center", va="center", fontsize=10, color=C_DARK, fontweight="bold")
    ax.text(x_start - 0.85, y_box + box_h / 2 - 0.32, "(file)",
            ha="center", va="center", fontsize=8.5, color=C_GRAY, style="italic")

    # stdout label on far right
    x_end = x_start + total_w
    ax.annotate("", xy=(x_end + 0.85, y_box + box_h / 2),
                xytext=(x_end + 0.05, y_box + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.8))
    ax.text(x_end + 0.85, y_box + box_h / 2 + 0.32, "stdout",
            ha="center", va="center", fontsize=10, color=C_DARK, fontweight="bold")
    ax.text(x_end + 0.85, y_box + box_h / 2 - 0.32, "(terminal)",
            ha="center", va="center", fontsize=8.5, color=C_GRAY, style="italic")

    # Sample data flowing -- show 3 sample rows with progressive shrinking
    sample_y = 1.95
    samples = [
        ("11k lines",  "230 lines",  "230 IPs",   "57 unique", "10 rows"),
    ]
    for row in samples:
        for i, txt in enumerate(row):
            x = x_start + i * (box_w + gap) + box_w / 2
            ax.add_patch(FancyBboxPatch((x - 0.7, sample_y - 0.22), 1.4, 0.44,
                                        boxstyle="round,pad=0.02",
                                        facecolor=C_BG_SOFT, edgecolor=C_LIGHT))
            ax.text(x, sample_y, txt, ha="center", va="center",
                    fontsize=10, color=C_DARK, family="monospace")
    ax.text(x_start - 0.85, sample_y, "size",
            ha="center", va="center", fontsize=9.5, color=C_GRAY,
            fontweight="bold", style="italic")

    # Why pipelines win
    ax.text(7.0, 0.95,
            "Why pipelines:  no temp files . streamed in memory . each tool does ONE thing well (Unix philosophy)",
            ha="center", va="center", fontsize=10.5, color=C_DARK, style="italic")
    ax.text(7.0, 0.55,
            "Use  >  to redirect final stdout to a file,   2>&1  to merge stderr,   tee  to fork the stream",
            ha="center", va="center", fontsize=10, color=C_GRAY, family="monospace")

    plt.tight_layout()
    _save(fig, "fig4_pipeline_flow")


# ---------------------------------------------------------------------------
# Figure 5 -- Distro family tree
# ---------------------------------------------------------------------------
def fig5_distro_family_tree() -> None:
    fig, ax = plt.subplots(figsize=(14, 8.5))
    _no_axis(ax)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.5)

    ax.text(7.0, 8.10, "Linux Distribution Family Tree",
            ha="center", va="center", fontsize=16, fontweight="bold", color=C_DARK)
    ax.text(7.0, 7.78, "Three big lineages: Debian (apt) . Red Hat (yum/dnf) . Arch (pacman)  -- plus SUSE and the independents",
            ha="center", va="center", fontsize=10.5, style="italic", color=C_GRAY)

    # Year axis at the bottom
    years = [1993, 1995, 2000, 2005, 2010, 2015, 2020, 2025]
    y_axis = 0.65
    ax.plot([0.7, 13.3], [y_axis, y_axis], color=C_GRAY, lw=1.2)
    for yr in years:
        x = 0.7 + (yr - 1993) * (12.6 / (2025 - 1993))
        ax.plot([x, x], [y_axis - 0.08, y_axis + 0.08], color=C_GRAY, lw=1.2)
        ax.text(x, y_axis - 0.35, str(yr), ha="center", va="center",
                fontsize=9, color=C_GRAY)

    def x_for_year(yr: float) -> float:
        return 0.7 + (yr - 1993) * (12.6 / (2025 - 1993))

    # Lineages: each one has a parent node and child distros
    lineages = [
        # (label, color, y_band, parent (name, year), children [(name, year)...])
        ("Debian family (apt)",    C_BLUE,
         6.55,
         ("Debian", 1993),
         [("Ubuntu",   2004),
          ("Ubuntu LTS", 2008),
          ("Linux Mint", 2006),
          ("Kali",     2013),
          ("Raspberry Pi OS", 2012),
          ("Pop!_OS",  2017)]),

        ("Red Hat family (yum/dnf)", C_PURPLE,
         5.10,
         ("Red Hat Linux", 1994),
         [("Fedora",   2003),
          ("RHEL",     2003),
          ("CentOS",   2004),
          ("Amazon Linux", 2010),
          ("CentOS Stream", 2019),
          ("Rocky / Alma", 2021)]),

        ("Arch family (pacman)",   C_AMBER,
         3.65,
         ("Arch Linux", 2002),
         [("Manjaro",  2011),
          ("EndeavourOS", 2019),
          ("SteamOS 3", 2022)]),

        ("SUSE family (zypper)",   C_GREEN,
         2.30,
         ("SUSE Linux", 1994),
         [("openSUSE", 2005),
          ("SLES",     2000),
          ("Tumbleweed", 2014)]),

        ("Independent",            C_GRAY,
         1.20,
         ("Slackware", 1993),
         [("Gentoo",   2000),
          ("Alpine",   2005),
          ("NixOS",    2003),
          ("Void",     2008)]),
    ]

    for label, color, y_band, parent, children in lineages:
        p_name, p_year = parent
        x_p = x_for_year(p_year)
        # row label on the left, with a coloured tag
        ax.add_patch(FancyBboxPatch((0.05, y_band - 0.22), 0.5, 0.44,
                                    boxstyle="round,pad=0.01",
                                    facecolor=color, edgecolor="none"))
        ax.text(0.05 + 0.6, y_band, label, ha="left", va="center",
                fontsize=10.5, color=color, fontweight="bold")

        # parent node
        ax.add_patch(FancyBboxPatch((x_p - 0.55, y_band - 0.62), 1.1, 0.42,
                                    boxstyle="round,pad=0.03",
                                    facecolor=color, edgecolor="none"))
        ax.text(x_p, y_band - 0.41, p_name, ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold")

        # connectors and children
        # spine line at this band height
        last_x = x_p
        for name, year in children:
            x_c = x_for_year(year)
            # curved-ish connector via a vertical drop and horizontal step
            ax.plot([x_p, x_c, x_c], [y_band - 0.20, y_band - 0.20, y_band - 0.05],
                    color=color, lw=1.2, alpha=0.55)
            # child node
            ax.add_patch(FancyBboxPatch((x_c - 0.55, y_band - 0.05), 1.1, 0.40,
                                        boxstyle="round,pad=0.03",
                                        facecolor="white",
                                        edgecolor=color, linewidth=1.2))
            ax.text(x_c, y_band + 0.15, name, ha="center", va="center",
                    fontsize=8.8, color=C_DARK)
            last_x = max(last_x, x_c)

    # Footer: pick-a-distro hint
    ax.text(7.0, 0.05,
            "Pick by use case:  servers -> Ubuntu LTS or RHEL/Rocky    .    desktop -> Ubuntu / Mint / Fedora    "
            ".    cloud images -> distro your provider blesses",
            ha="center", va="center", fontsize=10, color=C_DARK, style="italic")

    plt.tight_layout()
    _save(fig, "fig5_distro_family_tree")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Linux Basics figures...")
    print(f"  EN -> {EN_DIR}")
    print(f"  ZH -> {ZH_DIR}")
    fig1_directory_tree()
    fig2_command_anatomy()
    fig3_command_cheatsheet()
    fig4_pipeline_flow()
    fig5_distro_family_tree()
    print("Done.")


if __name__ == "__main__":
    main()
