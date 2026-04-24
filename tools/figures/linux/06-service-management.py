"""
Figure generation script for Linux Article 06: Service Management (systemd).

Generates 5 conceptual figures used in both EN and ZH versions of the
article. Each figure is rendered to BOTH article asset folders so the
markdown image references stay in sync across languages.

Figures:
    fig1_systemd_architecture   systemd as PID 1, the unit catalogue it
                                manages, and how units roll up into
                                targets (multi-user, graphical, ...).
    fig2_service_lifecycle      Service unit state machine: inactive ->
                                activating -> active -> deactivating
                                -> inactive, with the failed branch and
                                the auto-restart loop drawn explicitly.
    fig3_unit_file_anatomy      A real .service file annotated by section
                                ([Unit], [Service], [Install]), showing
                                which keys answer which question
                                (dependencies / how to run / when to start).
    fig4_journalctl_filters     journald as a single ring buffer fed by
                                every unit, and the most useful
                                journalctl filter recipes mapped onto it.
    fig5_boot_timeline          Boot timeline from firmware -> bootloader
                                -> kernel -> systemd, with the target
                                chain (sysinit -> basic -> multi-user)
                                drawn against time.

Usage:
    python3 scripts/figures/linux/06-service-management.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401  (registers the seaborn style we use)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# --- Shared style ----------------------------------------------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

COLOR_BLUE = COLORS["primary"]
COLOR_PURPLE = COLORS["accent"]
COLOR_GREEN = COLORS["success"]
COLOR_AMBER = COLORS["warning"]
COLOR_GREY = COLORS["text2"]
COLOR_LIGHT = COLORS["grid"]
COLOR_RED = COLORS["danger"]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "linux" / "service-management"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "linux" / "service-management"


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
# Figure 1: systemd architecture (PID 1 + units + targets)
# ---------------------------------------------------------------------------

def fig1_systemd_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    # Reserve a left gutter for layer labels
    gutter = 1.5

    # PID 1 box (top, the supervisor)
    pid_w = 3.6
    pid_x = gutter + ((13 - gutter) - pid_w) / 2
    _rounded_box(ax, pid_x, 6.85, pid_w, 1.25, facecolor=COLOR_BLUE,
                 edgecolor=COLOR_BLUE)
    ax.text(pid_x + pid_w / 2, 7.65, "systemd", ha="center", va="center",
            color="white", fontsize=15, fontweight="bold")
    ax.text(pid_x + pid_w / 2, 7.15, "PID 1  -  first user-space process",
            ha="center", va="center", color="white",
            fontsize=9.5, style="italic")

    # Unit catalogue layer (middle): the kinds of units systemd knows about
    unit_y = 4.0
    unit_h = 1.4
    units = [
        (".service", "long-running\ndaemons", COLOR_PURPLE),
        (".socket",  "on-demand\nactivation",  COLOR_AMBER),
        (".timer",   "scheduled\ntasks",       COLOR_GREEN),
        (".mount",   "filesystem\nmounts",     COLOR_GREY),
        (".path",    "watch a\nfile / dir",    COLOR_GREY),
        (".target",  "groups of\nunits",       COLOR_BLUE),
    ]
    n = len(units)
    avail_w = 13 - gutter - 0.3
    box_w = 1.55
    spacing = (avail_w - n * box_w) / (n + 1)
    pid_centre_x = pid_x + pid_w / 2
    for i, (name, desc, color) in enumerate(units):
        x = gutter + spacing + i * (box_w + spacing)
        _rounded_box(ax, x, unit_y, box_w, unit_h, facecolor="white",
                     edgecolor=color, lw=1.6)
        ax.text(x + box_w / 2, unit_y + unit_h - 0.32, name,
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=color)
        ax.text(x + box_w / 2, unit_y + 0.42, desc,
                ha="center", va="center", fontsize=8.3, color=COLOR_GREY)
        # Connector down from PID 1 to each unit type
        ax.plot([pid_centre_x, x + box_w / 2], [6.85, unit_y + unit_h],
                color=COLOR_LIGHT, lw=1.2, zorder=0)

    ax.text(0.3, unit_y + unit_h / 2, "Units",
            ha="left", va="center", fontsize=11,
            color=COLORS["text"], fontweight="bold")

    # Target chain (bottom): the runlevels of the systemd world
    tgt_y = 1.2
    tgt_h = 1.2
    targets = [
        ("sysinit.target",   "early init\n(mounts, swap)", COLOR_GREY),
        ("basic.target",     "sockets, timers,\npaths ready", COLOR_AMBER),
        ("multi-user.target","text login,\nall services up", COLOR_BLUE),
        ("graphical.target", "desktop /\ndisplay manager", COLOR_PURPLE),
    ]
    tw = 2.45
    tgap = (avail_w - len(targets) * tw) / (len(targets) + 1)
    prev_x = None
    for i, (name, desc, color) in enumerate(targets):
        x = gutter + tgap + i * (tw + tgap)
        _rounded_box(ax, x, tgt_y, tw, tgt_h, facecolor=color,
                     edgecolor=color, alpha=0.9)
        ax.text(x + tw / 2, tgt_y + tgt_h - 0.32, name,
                ha="center", va="center", color="white",
                fontsize=10.5, fontweight="bold")
        ax.text(x + tw / 2, tgt_y + 0.32, desc,
                ha="center", va="center", color="white", fontsize=8.5)
        if prev_x is not None:
            arr = FancyArrowPatch((prev_x + tw, tgt_y + tgt_h / 2),
                                  (x, tgt_y + tgt_h / 2),
                                  arrowstyle="->", mutation_scale=14,
                                  color=COLOR_GREY, lw=1.4)
            ax.add_patch(arr)
        prev_x = x

    ax.text(0.3, tgt_y + tgt_h / 2, "Targets\n(boot chain)",
            ha="left", va="center", fontsize=11,
            color=COLORS["text"], fontweight="bold")

    # Title
    ax.text(6.5, 8.4,
            "systemd architecture:  PID 1  ->  units  ->  targets",
            ha="center", fontsize=14, fontweight="bold", color=COLORS["text"])

    _save(fig, "fig1_systemd_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2: Service unit lifecycle (state machine)
# ---------------------------------------------------------------------------

def fig2_service_lifecycle() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    # State boxes
    states = {
        "inactive":    (1.0, 3.5, COLOR_GREY,   "stopped\n(not running)"),
        "activating":  (3.6, 5.0, COLOR_AMBER,  "ExecStart in\nprogress"),
        "active":      (6.4, 5.0, COLOR_GREEN,  "running\nMain PID alive"),
        "deactivating":(9.0, 5.0, COLOR_AMBER,  "ExecStop in\nprogress"),
        "failed":      (6.4, 1.2, COLOR_RED,    "exit != 0\nor crash"),
    }
    box_w, box_h = 2.1, 1.4
    for name, (x, y, color, sub) in states.items():
        _rounded_box(ax, x, y, box_w, box_h, facecolor=color,
                     edgecolor=color, alpha=0.9)
        ax.text(x + box_w / 2, y + box_h - 0.4, name,
                ha="center", va="center", color="white",
                fontsize=12, fontweight="bold")
        ax.text(x + box_w / 2, y + 0.4, sub,
                ha="center", va="center", color="white", fontsize=8.5)

    # Helper for arrows between named state centres
    def arrow(src, dst, label, *, color=COLOR_GREY, dy_label=0.25,
              curve=0.0, lw=1.4):
        sx, sy, _, _ = states[src]
        dx, dy, _, _ = states[dst]
        sx_c = sx + box_w / 2
        sy_c = sy + box_h / 2
        dx_c = dx + box_w / 2
        dy_c = dy + box_h / 2
        cs = f"arc3,rad={curve}"
        arr = FancyArrowPatch((sx_c, sy_c), (dx_c, dy_c),
                              arrowstyle="->", mutation_scale=16,
                              color=color, lw=lw,
                              connectionstyle=cs)
        ax.add_patch(arr)
        ax.text((sx_c + dx_c) / 2,
                (sy_c + dy_c) / 2 + dy_label,
                label, ha="center", va="center",
                fontsize=9, color=color, style="italic",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor=color, lw=0.8))

    # Happy path
    arrow("inactive", "activating", "systemctl start",
          color=COLOR_BLUE, dy_label=0.35, curve=-0.15)
    arrow("activating", "active", "ExecStart OK",
          color=COLOR_GREEN, dy_label=0.30)
    arrow("active", "deactivating", "systemctl stop",
          color=COLOR_BLUE, dy_label=0.30)
    arrow("deactivating", "inactive", "ExecStop OK",
          color=COLOR_GREY, dy_label=0.35, curve=-0.15)

    # Failure branches
    arrow("activating", "failed", "ExecStart fails",
          color=COLOR_RED, dy_label=0.0, curve=0.25)
    arrow("active", "failed", "process exits\nor segfaults",
          color=COLOR_RED, dy_label=0.0, curve=-0.2)

    # Auto-restart loop (back to activating)
    arr = FancyArrowPatch((states["failed"][0] + box_w / 2,
                           states["failed"][1] + box_h),
                          (states["activating"][0] + box_w,
                           states["activating"][1]),
                          arrowstyle="->", mutation_scale=16,
                          color=COLOR_PURPLE, lw=1.6,
                          connectionstyle="arc3,rad=0.35")
    ax.add_patch(arr)
    ax.text(3.7, 3.0, "Restart=on-failure\nafter RestartSec",
            ha="center", va="center", fontsize=9, color=COLOR_PURPLE,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=COLOR_PURPLE, lw=0.8))

    # Title and footnote
    ax.text(6.0, 7.2,
            "Service lifecycle:  the states reported by 'systemctl status'",
            ha="center", fontsize=14, fontweight="bold", color=COLORS["text"])
    ax.text(6.0, 0.35,
            "An auto-restart policy turns 'failed' from a terminal state "
            "back into a transient one  -  the daemon stays up.",
            ha="center", fontsize=9.5, color=COLOR_GREY, style="italic")

    _save(fig, "fig2_service_lifecycle.png")


# ---------------------------------------------------------------------------
# Figure 3: Unit file anatomy ([Unit] / [Service] / [Install])
# ---------------------------------------------------------------------------

def fig3_unit_file_anatomy() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 9.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Sections of the file (with section colours)
    sections = [
        ("[Unit]", COLOR_BLUE,
         "metadata + ordering",
         "What this unit is, what must run\n"
         "before it (After=), and what it\n"
         "depends on (Requires=, Wants=).",
         [
            "Description=My Custom Application",
            "Documentation=https://example.com/myapp",
            "After=network-online.target",
            "Wants=network-online.target",
            "Requires=postgresql.service",
         ]),
        ("[Service]", COLOR_PURPLE,
         "how to run the process",
         "Type, ExecStart, restart policy,\n"
         "user/group, working directory, env\n"
         "vars, and resource limits applied\n"
         "via cgroups.",
         [
            "Type=simple",
            "ExecStart=/usr/local/bin/myapp --port 8080",
            "ExecReload=/bin/kill -HUP $MAINPID",
            "Restart=on-failure",
            "RestartSec=5",
            "User=myapp",
            "Group=myapp",
            "WorkingDirectory=/var/lib/myapp",
            "Environment=LOG_LEVEL=info",
            "MemoryMax=512M",
            "CPUQuota=50%",
         ]),
        ("[Install]", COLOR_GREEN,
         "when to enable it",
         "Read by 'systemctl enable' only.",
         [
            "WantedBy=multi-user.target",
         ]),
    ]

    # Layout constants
    left_x = 0.4
    left_w = 7.4
    right_x = left_x + left_w + 0.5
    right_w = 4.7
    line_h = 0.32
    section_gap = 0.45
    header_extra = 0.1

    top_y = 10.5
    cursor_y = top_y

    for header, color, title, body, lines in sections:
        # Compute this section's geometry first
        n = len(lines)
        section_h = line_h * 1.2 + line_h * n + line_h * 0.4
        section_top = cursor_y
        section_bottom = section_top - section_h

        # File panel (dark) for this section
        ax.add_patch(Rectangle((left_x, section_bottom), left_w, section_h,
                               facecolor=COLORS["text"], edgecolor=color, lw=1.6))

        # Header line inside the panel
        header_y = section_top - line_h * 0.7
        ax.text(left_x + 0.3, header_y, header,
                ha="left", va="center", color=color,
                fontsize=12, fontweight="bold", family="monospace")

        # File lines
        line_y = header_y - line_h
        for line in lines:
            if "=" in line:
                key, val = line.split("=", 1)
                # Use a single text() with two coloured spans? matplotlib
                # has no rich text, so render twice: key in light grey, then
                # full text width using monospace assumption.
                # Render key + '=' in grey, value in amber, both in
                # monospace at the same font size so the offsets align.
                key_str = key + "="
                ax.text(left_x + 0.5, line_y, key_str,
                        ha="left", va="center", color=COLORS["border"],
                        fontsize=9.5, family="monospace")
                # Estimate width using monospace character width.
                # 9.5pt monospace at 150 DPI is ~0.094 data units per char
                # for our axis width; tuned by eye.
                char_w = 0.105
                offset = len(key_str) * char_w
                ax.text(left_x + 0.5 + offset, line_y, val,
                        ha="left", va="center", color=COLORS["warning"],
                        fontsize=9.5, family="monospace")
            else:
                ax.text(left_x + 0.5, line_y, line,
                        ha="left", va="center", color=COLORS["grid"],
                        fontsize=9.5, family="monospace")
            line_y -= line_h

        # Right-side annotation card aligned with this section's TOP.
        # Use a min height of 1.05 so even single-line panels get a card
        # with room for both title and body. The card may extend slightly
        # below the panel into the inter-section gap, which is fine.
        card_h = max(1.05, section_h)
        card_y = section_top - card_h
        _rounded_box(ax, right_x, card_y, right_w, card_h,
                     facecolor="white", edgecolor=color, lw=1.6)
        ax.text(right_x + 0.25, card_y + card_h - 0.4,
                f"{header}  -  {title}",
                ha="left", va="top", fontsize=11.5,
                fontweight="bold", color=color)
        if card_h >= 1.4:
            ax.text(right_x + 0.25, card_y + card_h - 0.95,
                    body, ha="left", va="top", fontsize=9.5,
                    color=COLORS["text"])
        else:
            single = body.replace("\n", " ")
            ax.text(right_x + 0.25, card_y + card_h - 0.85,
                    single, ha="left", va="top", fontsize=9.0,
                    color=COLORS["text"])

        cursor_y = section_bottom - section_gap

        cursor_y = section_bottom - section_gap

    # Title and file path label at the very top
    ax.text(left_x, top_y + 0.55,
            "/etc/systemd/system/myapp.service",
            ha="left", va="bottom", color=COLOR_GREY,
            fontsize=10.5, family="monospace", style="italic")
    ax.text(6.5, 11.4,
            "Anatomy of a .service unit file",
            ha="center", fontsize=14, fontweight="bold", color=COLORS["text"])

    _save(fig, "fig3_unit_file_anatomy.png")


# ---------------------------------------------------------------------------
# Figure 4: journalctl - one journal, many filters
# ---------------------------------------------------------------------------

def fig4_journalctl_filters() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 6.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Top: producers feeding the journal
    producers = [
        ("sshd",       COLOR_BLUE),
        ("nginx",      COLOR_PURPLE),
        ("kernel",     COLOR_AMBER),
        ("cron",       COLOR_GREEN),
        ("myapp",      COLOR_GREY),
    ]
    py = 6.4
    pw = 1.55
    pgap = (12 - len(producers) * pw) / (len(producers) + 1)
    for i, (name, color) in enumerate(producers):
        x = pgap + i * (pw + pgap)
        _rounded_box(ax, x, py, pw, 0.95, facecolor=color,
                     edgecolor=color, alpha=0.9)
        ax.text(x + pw / 2, py + 0.5, name,
                ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
        # Arrow down to the journal bar
        arr = FancyArrowPatch((x + pw / 2, py),
                              (x + pw / 2, 4.95),
                              arrowstyle="->", mutation_scale=12,
                              color=color, lw=1.2, alpha=0.8)
        ax.add_patch(arr)

    # Middle: the journal as a single ring buffer
    jb_x, jb_y, jb_w, jb_h = 0.6, 4.05, 10.8, 0.95
    _rounded_box(ax, jb_x, jb_y, jb_w, jb_h,
                 facecolor=COLORS["text"], edgecolor=COLORS["text"])
    ax.text(jb_x + 0.25, jb_y + jb_h / 2,
            "journald  ->  /var/log/journal/   (binary, indexed)",
            ha="left", va="center", color="white",
            fontsize=11, fontweight="bold", family="monospace")
    # tick marks suggesting a stream
    for i in range(20):
        tx = jb_x + 4.1 + i * 0.32
        ax.add_patch(Rectangle((tx, jb_y + 0.18), 0.16, jb_h - 0.36,
                               facecolor=COLORS["text2"], edgecolor=COLORS["text2"]))

    # Bottom: filter recipes drawn as cards reading from the journal
    filters = [
        ("-u nginx",          "logs of one unit",            COLOR_PURPLE),
        ("-f",                "follow new entries\n(like tail -f)",  COLOR_BLUE),
        ("-p err",            "priority err and worse\n(0=emerg .. 7=debug)", COLOR_RED),
        ("--since '1h ago'",  "time window filter",           COLOR_AMBER),
        ("-b -1",             "logs from the previous\nboot",  COLOR_GREEN),
    ]
    fy = 0.8
    fh = 2.4
    fw = 2.05
    fgap = (12 - len(filters) * fw) / (len(filters) + 1)
    for i, (cmd, desc, color) in enumerate(filters):
        x = fgap + i * (fw + fgap)
        _rounded_box(ax, x, fy, fw, fh, facecolor="white",
                     edgecolor=color, lw=1.6)
        ax.text(x + fw / 2, fy + fh - 0.4,
                f"journalctl", ha="center", va="center",
                fontsize=9, color=COLOR_GREY, family="monospace")
        ax.text(x + fw / 2, fy + fh - 0.85,
                cmd, ha="center", va="center",
                fontsize=11, color=color, family="monospace",
                fontweight="bold")
        ax.text(x + fw / 2, fy + 0.55, desc,
                ha="center", va="center", fontsize=9,
                color=COLORS["text"])
        # Arrow up from card to journal bar
        arr = FancyArrowPatch((x + fw / 2, fy + fh),
                              (x + fw / 2, jb_y),
                              arrowstyle="->", mutation_scale=12,
                              color=color, lw=1.1, alpha=0.85)
        ax.add_patch(arr)

    # Title
    ax.text(6.0, 7.7,
            "journald:  one unified log, queried with journalctl filters",
            ha="center", fontsize=14, fontweight="bold", color=COLORS["text"])

    _save(fig, "fig4_journalctl_filters.png")


# ---------------------------------------------------------------------------
# Figure 5: Boot timeline (firmware -> bootloader -> kernel -> systemd)
# ---------------------------------------------------------------------------

def fig5_boot_timeline() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.2)
    ax.axis("off")

    # Title at the very top
    ax.text(6.5, 8.95,
            "Boot timeline:  firmware  ->  bootloader  ->  kernel  ->  systemd  "
            "->  multi-user.target",
            ha="center", fontsize=13.5, fontweight="bold", color=COLORS["text"])

    # Useful diagnostic commands box (top, just below title)
    cmd_y = 7.35
    _rounded_box(ax, 0.6, cmd_y, 11.8, 1.2, facecolor="#f8fafc",
                 edgecolor=COLOR_LIGHT, lw=1.0)
    ax.text(6.5, cmd_y + 0.85,
            "Diagnose boot performance",
            ha="center", fontsize=11, fontweight="bold", color=COLORS["text"])
    ax.text(6.5, cmd_y + 0.32,
            "systemd-analyze       systemd-analyze blame       "
            "systemd-analyze critical-chain",
            ha="center", fontsize=10, color=COLOR_BLUE,
            family="monospace")

    # Target chain detail (middle band): parallel activation
    chain_y = 5.2
    chain_items = [
        ("local-fs.target", COLOR_GREY),
        ("swap.target",     COLOR_GREY),
        ("sysinit.target",  COLOR_GREY),
        ("sockets.target",  COLOR_AMBER),
        ("timers.target",   COLOR_AMBER),
        ("basic.target",    COLOR_AMBER),
        ("network.target",  COLOR_BLUE),
        ("sshd.service",    COLOR_BLUE),
        ("nginx.service",   COLOR_BLUE),
        ("multi-user.target", COLOR_GREEN),
    ]
    n_chain = len(chain_items)
    chain_left = 0.4
    chain_right = 12.6
    chain_gap = 0.08
    cw = (chain_right - chain_left - (n_chain - 1) * chain_gap) / n_chain
    for i, (name, color) in enumerate(chain_items):
        cx = chain_left + i * (cw + chain_gap)
        _rounded_box(ax, cx, chain_y, cw, 0.55, facecolor="white",
                     edgecolor=color, lw=1.3)
        ax.text(cx + cw / 2, chain_y + 0.275, name,
                ha="center", va="center", fontsize=6.6,
                color=color, fontweight="bold", family="monospace")

    ax.text(0.6, chain_y + 0.85,
            "Targets activated by systemd  (parallel where dependencies allow)",
            ha="left", fontsize=10, color=COLORS["text"], fontweight="bold")
    ax.text(0.6, chain_y - 0.35,
            "Ordering comes from After= / Before=; concurrency comes from "
            "Wants= without explicit ordering.",
            ha="left", fontsize=9, color=COLOR_GREY, style="italic")

    # Time axis (bottom band)
    axis_y = 2.0
    ax.add_patch(Rectangle((0.6, axis_y - 0.04), 11.8, 0.08,
                           facecolor=COLOR_GREY, edgecolor=COLOR_GREY))
    arr = FancyArrowPatch((12.4, axis_y), (12.7, axis_y),
                          arrowstyle="->", mutation_scale=18,
                          color=COLOR_GREY, lw=1.6)
    ax.add_patch(arr)
    ax.text(12.7, axis_y - 0.4, "time", ha="right", va="center",
            fontsize=9.5, color=COLOR_GREY, style="italic")

    # Phases on the timeline
    phases = [
        ("Firmware",     0.6,  1.6, COLOR_GREY,
         "BIOS / UEFI POST,\nfind boot device"),
        ("Bootloader",   2.2,  1.6, COLOR_AMBER,
         "GRUB picks kernel\n+ initrd, loads them"),
        ("Kernel",       3.8,  1.8, COLOR_PURPLE,
         "kernel + initramfs:\nmount root rw"),
        ("systemd",      5.6,  2.2, COLOR_BLUE,
         "PID 1 starts,\nreads unit files"),
        ("sysinit\n.target", 7.8, 1.6, COLOR_GREY,
         "early mounts,\nswap, udev"),
        ("basic\n.target", 9.4,  1.4, COLOR_AMBER,
         "sockets / timers /\npaths armed"),
        ("multi-user\n.target", 10.8, 1.6, COLOR_GREEN,
         "all enabled services\nactive, login ready"),
    ]
    bar_h = 0.55
    for name, x, w, color, desc in phases:
        ax.add_patch(Rectangle((x, axis_y - bar_h / 2), w, bar_h,
                               facecolor=color, edgecolor=color, alpha=0.9))
        ax.text(x + w / 2, axis_y, name,
                ha="center", va="center", color="white",
                fontsize=8.8, fontweight="bold")
        ax.text(x + w / 2, axis_y - 0.95, desc,
                ha="center", va="top", fontsize=8.3,
                color=COLOR_GREY)

    # Vertical "where systemd takes over" marker
    ax.plot([5.6, 5.6], [axis_y + 0.35, chain_y - 0.7],
            color=COLOR_BLUE, lw=1.0, ls="--", alpha=0.7)
    ax.text(5.6, chain_y - 0.65, "systemd takes over as PID 1",
            ha="center", va="bottom", fontsize=9.5,
            color=COLOR_BLUE, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#eff6ff",
                      edgecolor=COLOR_BLUE, lw=1.0))

    _save(fig, "fig5_boot_timeline.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_systemd_architecture()
    fig2_service_lifecycle()
    fig3_unit_file_anatomy()
    fig4_journalctl_filters()
    fig5_boot_timeline()
    print("Wrote 5 figures to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
