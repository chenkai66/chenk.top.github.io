"""
Figure generation script for Linux Article 07: Process and Resource Management.

Generates 5 conceptual figures used in both EN and ZH versions of the
article. Each figure is rendered to BOTH article asset folders so the
markdown image references stay in sync across languages.

Figures:
    fig1_process_states         The Linux process state machine: Running,
                                Sleeping (interruptible / uninterruptible),
                                Stopped, Zombie, Dead, with the events
                                that drive transitions between them.
    fig2_fork_exec_model        The classic fork() + exec() pattern: parent
                                duplicates itself into a child, then the
                                child overlays a new program image, while
                                the parent eventually wait()s for it.
    fig3_top_dissected          A realistic `top` screen with each region
                                (load average / Tasks / %Cpu / Mem / process
                                table) annotated so a beginner can map
                                numbers to meaning.
    fig4_cgroups_namespaces     The two kernel mechanisms that compose into
                                "containers": namespaces (what a process
                                can see) on one axis, cgroups (what a
                                process can use) on the other.
    fig5_signals_table          The signals you actually use day-to-day,
                                grouped by intent (graceful stop, hard
                                kill, reload, job control, debugging),
                                with default action and catchability.

Usage:
    python3 scripts/figures/linux/07-process-resource.py
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
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "linux" / "process-resource-management"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "linux" / "process-resource-management"


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
# Figure 1: process state machine
# ---------------------------------------------------------------------------

def fig1_process_states() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # State boxes: (key, x, y, w, h, color, label, code, sub)
    states = [
        ("running", 5.0, 5.6, 2.0, 1.2, COLOR_GREEN,
         "Running", "R", "on CPU or runnable"),
        ("sleep_s", 1.2, 5.6, 2.2, 1.2, COLOR_BLUE,
         "Sleeping", "S", "interruptible\n(waiting for event)"),
        ("sleep_d", 8.6, 5.6, 2.2, 1.2, COLOR_PURPLE,
         "Uninterruptible", "D", "sleeping in kernel\n(disk I/O)"),
        ("stopped", 1.2, 2.4, 2.2, 1.2, COLOR_AMBER,
         "Stopped", "T", "paused by SIGSTOP\nor Ctrl+Z"),
        ("zombie", 8.6, 2.4, 2.2, 1.2, COLOR_RED,
         "Zombie", "Z", "exited, parent\nhasn't wait()ed"),
        ("dead", 5.0, 0.5, 2.0, 1.0, COLOR_GREY,
         "Dead / reaped", "X", "removed from process table"),
    ]

    pos = {}
    for key, x, y, w, h, color, label, code, sub in states:
        pos[key] = (x, y, w, h)
        _rounded_box(ax, x, y, w, h, facecolor=color, edgecolor=color,
                     alpha=0.9)
        ax.text(x + w / 2, y + h - 0.32, label,
                ha="center", va="center", color="white",
                fontsize=12, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.62, f"({code})",
                ha="center", va="center", color="white",
                fontsize=10, family="monospace")
        ax.text(x + w / 2, y + 0.25, sub,
                ha="center", va="center", color="white", fontsize=8.2)

    def _arrow(src, dst, label, *, color=COLOR_GREY, dx_lbl=0.0, dy_lbl=0.0,
               curve=0.0, src_side="auto", dst_side="auto"):
        sx, sy, sw, sh = pos[src]
        dx, dy, dw, dh = pos[dst]
        scx, scy = sx + sw / 2, sy + sh / 2
        dcx, dcy = dx + dw / 2, dy + dh / 2

        def _anchor(box_x, box_y, box_w, box_h, side, toward):
            cx, cy = box_x + box_w / 2, box_y + box_h / 2
            if side == "auto":
                if abs(toward[0] - cx) > abs(toward[1] - cy):
                    side = "right" if toward[0] > cx else "left"
                else:
                    side = "top" if toward[1] > cy else "bottom"
            if side == "right":
                return (box_x + box_w, cy)
            if side == "left":
                return (box_x, cy)
            if side == "top":
                return (cx, box_y + box_h)
            return (cx, box_y)

        a = _anchor(sx, sy, sw, sh, src_side, (dcx, dcy))
        b = _anchor(dx, dy, dw, dh, dst_side, (scx, scy))
        connectionstyle = f"arc3,rad={curve}" if curve else "arc3"
        arr = FancyArrowPatch(a, b, arrowstyle="->", mutation_scale=14,
                              color=color, lw=1.4,
                              connectionstyle=connectionstyle)
        ax.add_patch(arr)
        mx = (a[0] + b[0]) / 2 + dx_lbl
        my = (a[1] + b[1]) / 2 + dy_lbl
        ax.text(mx, my, label, ha="center", va="center", fontsize=8.5,
                color=color, family="monospace",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="none", alpha=0.9))

    # Sleeping <-> Running (interruptible: waiting for event then woken)
    _arrow("sleep_s", "running", "wake up",
           color=COLOR_BLUE, dy_lbl=0.25, curve=-0.25)
    _arrow("running", "sleep_s", "wait()/select()",
           color=COLOR_BLUE, dy_lbl=-0.25, curve=-0.25)

    # Running <-> Uninterruptible (issuing/finishing a kernel I/O)
    _arrow("running", "sleep_d", "blocking syscall",
           color=COLOR_PURPLE, dy_lbl=0.25, curve=-0.25)
    _arrow("sleep_d", "running", "I/O completes",
           color=COLOR_PURPLE, dy_lbl=-0.25, curve=-0.25)

    # Running <-> Stopped (SIGSTOP / SIGCONT)
    _arrow("running", "stopped", "SIGSTOP",
           color=COLOR_AMBER, dx_lbl=-0.4, curve=0.2)
    _arrow("stopped", "running", "SIGCONT",
           color=COLOR_AMBER, dx_lbl=0.5, dy_lbl=0.2, curve=0.2)

    # Running -> Zombie (exit())
    _arrow("running", "zombie", "exit()",
           color=COLOR_RED, dx_lbl=0.4, curve=-0.2)

    # Zombie -> Dead (parent calls wait())
    _arrow("zombie", "dead", "parent wait()",
           color=COLOR_GREY, dx_lbl=0.6, curve=0.25)

    ax.text(6.0, 7.4,
            "Linux Process State Machine",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=COLORS["text"])
    ax.text(6.0, 7.0,
            "what `ps`/`top` print in the STAT/S column - "
            "and what each transition actually means",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY,
            style="italic")

    _save(fig, "fig1_process_states.png")


# ---------------------------------------------------------------------------
# Figure 2: fork() + exec() model
# ---------------------------------------------------------------------------

def fig2_fork_exec_model() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Three vertical lifelines: shell (parent before fork), parent, child.
    # We show 4 time-step columns: BEFORE fork - AFTER fork - AFTER exec - AFTER wait.
    cols_x = [1.5, 4.5, 7.5, 10.5]
    col_labels = [
        "1. before fork()",
        "2. after fork()",
        "3. child exec()s",
        "4. parent wait()s",
    ]

    # Header row
    for x, label in zip(cols_x, col_labels):
        ax.text(x, 7.4, label, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color=COLORS["text"])

    # Helper to draw a process box
    def proc(x, y, *, pid, prog, color, w=2.2, h=1.4, note=None):
        _rounded_box(ax, x - w / 2, y - h / 2, w, h, facecolor=color,
                     edgecolor=color, alpha=0.9)
        ax.text(x, y + 0.32, prog, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold",
                family="monospace")
        ax.text(x, y - 0.05, f"PID {pid}", ha="center", va="center",
                color="white", fontsize=9.5)
        if note:
            ax.text(x, y - 0.42, note, ha="center", va="center",
                    color="white", fontsize=8, style="italic")

    # Column 1: just bash
    proc(cols_x[0], 5.4, pid="1234", prog="bash", color=COLOR_BLUE,
         note="the shell")

    # Column 2: bash + duplicated bash (child of fork())
    proc(cols_x[1], 6.0, pid="1234", prog="bash", color=COLOR_BLUE,
         note="parent")
    proc(cols_x[1], 3.2, pid="5678", prog="bash (copy)", color=COLOR_BLUE,
         note="child - same code,\nsame open files")

    # Column 3: bash + ls (after exec replaces image)
    proc(cols_x[2], 6.0, pid="1234", prog="bash", color=COLOR_BLUE,
         note="parent waiting")
    proc(cols_x[2], 3.2, pid="5678", prog="ls", color=COLOR_GREEN,
         note="same PID,\nbrand-new program")

    # Column 4: bash again, child reaped
    proc(cols_x[3], 5.4, pid="1234", prog="bash", color=COLOR_BLUE,
         note="reaped child,\nexit code -> $?")
    # ghost outline of the gone child
    _rounded_box(ax, cols_x[3] - 1.1, 2.5, 2.2, 1.4, facecolor="white",
                 edgecolor=COLOR_GREY, lw=1.0, alpha=0.4)
    ax.text(cols_x[3], 3.2, "(child gone)", ha="center", va="center",
            color=COLOR_GREY, fontsize=9, style="italic")

    # Arrows between columns
    def hop(x0, y0, x1, y1, label, color):
        arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->",
                              mutation_scale=14, color=color, lw=1.4)
        ax.add_patch(arr)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.25, label,
                ha="center", va="center", fontsize=9, color=color,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none"))

    hop(cols_x[0] + 1.1, 5.7, cols_x[1] - 1.1, 6.0, "fork()",
        COLOR_PURPLE)
    hop(cols_x[0] + 1.1, 5.1, cols_x[1] - 1.1, 3.2, "creates child",
        COLOR_PURPLE)
    hop(cols_x[1] + 1.1, 3.2, cols_x[2] - 1.1, 3.2, "execve(\"/bin/ls\")",
        COLOR_GREEN)
    hop(cols_x[2] + 1.1, 3.2, cols_x[3] - 1.1, 2.9, "exit(0)",
        COLOR_RED)
    hop(cols_x[2] + 1.1, 6.0, cols_x[3] - 1.1, 5.4, "waitpid(5678)",
        COLOR_GREY)

    # Bottom explanatory band
    ax.text(6.0, 1.1,
            "Linux never starts a process from scratch.  Every process is a "
            "fork() of an existing one,",
            ha="center", va="center", fontsize=10.5, color=COLORS["text"])
    ax.text(6.0, 0.65,
            "then exec() optionally replaces the program image while keeping "
            "the same PID, file descriptors and PPID.",
            ha="center", va="center", fontsize=10.5, color=COLORS["text"])
    ax.text(6.0, 0.18,
            "PID 1 (init / systemd) is the only process the kernel itself "
            "creates - every other PID is some descendant of it.",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY,
            style="italic")

    fig.suptitle("The fork() + exec() Model",
                 fontsize=14, fontweight="bold", y=0.98, color=COLORS["text"])

    _save(fig, "fig2_fork_exec_model.png")


# ---------------------------------------------------------------------------
# Figure 3: top output dissected
# ---------------------------------------------------------------------------

def fig3_top_dissected() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # Terminal-looking panel
    panel_x, panel_y, panel_w, panel_h = 0.4, 1.3, 8.2, 7.0
    ax.add_patch(Rectangle((panel_x, panel_y), panel_w, panel_h,
                           facecolor=COLORS["text"], edgecolor=COLORS["text"]))

    # Lines of a representative top output
    lines = [
        ("top - 12:00:00 up 10 days,  3:45,  2 users,  load average: 1.23, 0.87, 0.45", "header"),
        ("Tasks: 150 total,   2 running, 148 sleeping,   0 stopped,   0 zombie", "tasks"),
        ("%Cpu(s):  5.2 us,  2.1 sy,  0.0 ni, 92.3 id,  0.3 wa,  0.0 hi,  0.1 si,  0.0 st", "cpu"),
        ("MiB Mem :  15872.0 total,   8234.5 free,   3456.2 used,   4181.3 buff/cache", "mem"),
        ("MiB Swap:   2048.0 total,   2048.0 free,      0.0 used.  11234.5 avail Mem", "swap"),
        ("", "blank"),
        ("  PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND", "thead"),
        (" 1234 root      20   0  123456  12345   1234 R  50.0   0.8   1:23.45 python3", "row"),
        (" 5678 www-data  20   0  234567  23456   2345 S  10.0   1.5   0:12.34 nginx",   "row"),
        (" 9012 mysql     20   0  456789  45678   4567 S   5.0   2.9   3:45.67 mysqld",  "row"),
        (" 3456 alice     20   0   12345   1234    123 D   0.0   0.1   0:00.12 cp",      "row"),
    ]

    line_y = panel_y + panel_h - 0.45
    line_h = 0.45
    for i, (text, kind) in enumerate(lines):
        y = line_y - i * line_h
        color = COLORS["grid"]
        if kind == "thead":
            color = COLORS["warning"]
        elif kind == "header":
            color = "#93c5fd"
        ax.text(panel_x + 0.2, y, text, ha="left", va="center",
                fontsize=8.7, family="monospace", color=color)

    # ---------- Annotations on the right ----------
    notes = [
        (8.9, line_y - 0 * line_h, "load average 1/5/15 min\ncompare against #CPU cores",
         COLOR_BLUE),
        (8.9, line_y - 1 * line_h, "process counts\nwatch for zombies (Z)",
         COLOR_PURPLE),
        (8.9, line_y - 2 * line_h, "%Cpu breakdown:\nus/sy busy, wa = I/O wait,\nst = stolen by hypervisor",
         COLOR_AMBER),
        (8.9, line_y - 3 * line_h - 0.05, "memory: trust 'avail Mem',\nnot 'free'",
         COLOR_GREEN),
        (8.9, line_y - 4 * line_h - 0.1, "swap > 0 and growing =\nreal memory pressure",
         COLOR_RED),
        (8.9, line_y - 6 * line_h - 0.1, "process table:\nP sorts by CPU, M by MEM,\nk sends a signal, 1 toggles per-CPU",
         COLOR_GREY),
        (8.9, line_y - 9 * line_h - 0.2, "STAT 'D' = uninterruptible:\nstuck in disk/network I/O,\ncan't even be killed",
         COLOR_PURPLE),
    ]
    for x, y, txt, color in notes:
        ax.annotate(
            txt, xy=(panel_x + panel_w - 0.05, y),
            xytext=(x, y), fontsize=8.8, color=color,
            ha="left", va="center", fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=1.0,
                            connectionstyle="arc3,rad=-0.05"),
            bbox=dict(boxstyle="round,pad=0.25",
                      facecolor="white", edgecolor=color, lw=1.0),
        )

    # Bottom legend strip
    ax.text(6.5, 0.7,
            "STAT codes:  R running   S interruptible sleep   D uninterruptible sleep   "
            "T stopped   Z zombie",
            ha="center", va="center", fontsize=9.5, color=COLORS["text"],
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                      edgecolor=COLOR_AMBER, lw=1.0))

    ax.set_title("Reading `top` - what every region tells you",
                 fontsize=14, fontweight="bold", pad=10, color=COLORS["text"])

    _save(fig, "fig3_top_dissected.png")


# ---------------------------------------------------------------------------
# Figure 4: cgroups + namespaces (container foundation)
# ---------------------------------------------------------------------------

def fig4_cgroups_namespaces() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title + subtitle
    ax.text(6.0, 7.55,
            "Containers = namespaces (what you see) + cgroups (what you can use)",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=COLORS["text"])
    ax.text(6.0, 7.15,
            "Two orthogonal kernel mechanisms.  Combine them and you have a container.",
            ha="center", va="center", fontsize=10, color=COLOR_GREY,
            style="italic")

    # ----- Left column: namespaces -----
    ns_x, ns_y, ns_w, ns_h = 0.4, 0.7, 5.6, 6.0
    _rounded_box(ax, ns_x, ns_y, ns_w, ns_h, facecolor="#eff6ff",
                 edgecolor=COLOR_BLUE, lw=1.6)
    ax.text(ns_x + ns_w / 2, ns_y + ns_h - 0.4,
            "namespaces", ha="center", va="center",
            fontsize=14, fontweight="bold", color=COLOR_BLUE)
    ax.text(ns_x + ns_w / 2, ns_y + ns_h - 0.85,
            "isolate the view of system resources",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY,
            style="italic")

    namespaces = [
        ("pid",  "own PID 1, can't see host PIDs"),
        ("net",  "own NICs, routing table, ports"),
        ("mnt",  "own mount table / root filesystem"),
        ("uts",  "own hostname and domain name"),
        ("ipc",  "own SysV IPC, POSIX message queues"),
        ("user", "own UID/GID range, root inside only"),
        ("cgroup", "own view of the cgroup hierarchy"),
    ]
    row_h = 0.55
    start_y = ns_y + ns_h - 1.5
    for i, (ns, desc) in enumerate(namespaces):
        y = start_y - i * row_h
        _rounded_box(ax, ns_x + 0.3, y - 0.2, 1.4, 0.42,
                     facecolor=COLOR_BLUE, edgecolor=COLOR_BLUE, alpha=0.9)
        ax.text(ns_x + 1.0, y, ns, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold",
                family="monospace")
        ax.text(ns_x + 1.85, y, desc, ha="left", va="center",
                fontsize=9.5, color=COLORS["text"])

    # ----- Right column: cgroups -----
    cg_x, cg_y, cg_w, cg_h = 6.2, 0.7, 5.4, 6.0
    _rounded_box(ax, cg_x, cg_y, cg_w, cg_h, facecolor="#f5f3ff",
                 edgecolor=COLOR_PURPLE, lw=1.6)
    ax.text(cg_x + cg_w / 2, cg_y + cg_h - 0.4,
            "cgroups (v2)", ha="center", va="center",
            fontsize=14, fontweight="bold", color=COLOR_PURPLE)
    ax.text(cg_x + cg_w / 2, cg_y + cg_h - 0.85,
            "limit / account / prioritise resource use",
            ha="center", va="center", fontsize=9.5, color=COLOR_GREY,
            style="italic")

    controllers = [
        ("cpu",     "cpu.max = 50000 100000 -> 0.5 core"),
        ("memory",  "memory.max = 512M, hard ceiling -> OOM"),
        ("io",      "io.max  per-device read/write IOPS + BPS"),
        ("pids",    "pids.max prevents fork bombs"),
        ("cpuset",  "pin to specific CPUs / NUMA nodes"),
        ("hugetlb", "limit huge-page consumption"),
        ("rdma",    "limit RDMA HCA handles per group"),
    ]
    for i, (ctrl, desc) in enumerate(controllers):
        y = start_y - i * row_h
        _rounded_box(ax, cg_x + 0.3, y - 0.2, 1.4, 0.42,
                     facecolor=COLOR_PURPLE, edgecolor=COLOR_PURPLE,
                     alpha=0.9)
        ax.text(cg_x + 1.0, y, ctrl, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold",
                family="monospace")
        ax.text(cg_x + 1.85, y, desc, ha="left", va="center",
                fontsize=9, color=COLORS["text"], family="monospace")

    # Bottom synthesis bar
    syn_x, syn_y, syn_w, syn_h = 0.4, 0.05, 11.2, 0.55
    _rounded_box(ax, syn_x, syn_y, syn_w, syn_h, facecolor=COLOR_GREEN,
                 edgecolor=COLOR_GREEN, alpha=0.9)
    ax.text(syn_x + syn_w / 2, syn_y + syn_h / 2,
            "Docker / Podman / Kubernetes pods  =  one cgroup  +  a bundle of namespaces  "
            "+  an image rootfs",
            ha="center", va="center", color="white",
            fontsize=11, fontweight="bold")

    _save(fig, "fig4_cgroups_namespaces.png")


# ---------------------------------------------------------------------------
# Figure 5: signals reference table
# ---------------------------------------------------------------------------

def fig5_signals_table() -> None:
    fig, ax = plt.subplots(figsize=(12, 7.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # Columns: Signal | Num | Default action | Catchable? | Typical use
    columns = ["Signal", "Num", "Default action", "Catchable?", "Typical use"]
    col_x = [0.4, 2.6, 3.4, 6.4, 7.7]
    col_w = [2.2, 0.8, 3.0, 1.3, 4.0]

    # Group rows: (group_label, group_color, [(signal, num, default, catchable, use)])
    groups = [
        ("Graceful stop", COLOR_GREEN, [
            ("SIGTERM", "15", "Terminate", "yes",
             "default for `kill PID` - let the process clean up"),
            ("SIGINT",  "2",  "Terminate", "yes",
             "what Ctrl+C sends - interactive interrupt"),
            ("SIGQUIT", "3",  "Terminate + core", "yes",
             "Ctrl+\\ - terminate and dump core for debugging"),
        ]),
        ("Hard kill", COLOR_RED, [
            ("SIGKILL", "9",  "Terminate",  "no",
             "uncatchable - last resort when SIGTERM is ignored"),
            ("SIGSEGV", "11", "Terminate + core", "yes",
             "invalid memory access - usually a crash you didn't send"),
        ]),
        ("Reload / reconfigure", COLOR_BLUE, [
            ("SIGHUP",  "1",  "Terminate",  "yes",
             "by convention: re-read config (nginx, sshd, syslog)"),
            ("SIGUSR1", "10", "Terminate",  "yes",
             "user-defined - many daemons use it to rotate logs"),
            ("SIGUSR2", "12", "Terminate",  "yes",
             "user-defined - app-specific event"),
        ]),
        ("Job control", COLOR_AMBER, [
            ("SIGSTOP", "19", "Stop",       "no",
             "uncatchable pause - process freezes until SIGCONT"),
            ("SIGTSTP", "20", "Stop",       "yes",
             "Ctrl+Z - request a pause (catchable variant)"),
            ("SIGCONT", "18", "Continue",   "yes",
             "resume a stopped process - paired with `bg`/`fg`"),
        ]),
        ("Child / pipe", COLOR_PURPLE, [
            ("SIGCHLD", "17", "Ignore",     "yes",
             "child changed state - parent should wait() to reap"),
            ("SIGPIPE", "13", "Terminate",  "yes",
             "wrote to a pipe with no reader (`head` closing early)"),
        ]),
    ]

    # Header
    header_y = 8.7
    header_h = 0.55
    ax.add_patch(Rectangle((col_x[0], header_y), sum(col_w) + 0.2, header_h,
                           facecolor=COLORS["text"], edgecolor=COLORS["text"]))
    for i, label in enumerate(columns):
        ax.text(col_x[i] + 0.1, header_y + header_h / 2, label,
                ha="left", va="center", color="white",
                fontsize=10.5, fontweight="bold")

    row_h = 0.48
    y = header_y - 0.1
    for group_label, group_color, rows in groups:
        # Group separator label
        y -= row_h * 0.7
        ax.add_patch(Rectangle((col_x[0], y - row_h * 0.4),
                               sum(col_w) + 0.2, row_h * 0.55,
                               facecolor=group_color, edgecolor=group_color,
                               alpha=0.85))
        ax.text(col_x[0] + 0.15, y - row_h * 0.12, group_label,
                ha="left", va="center", color="white",
                fontsize=10.5, fontweight="bold")
        y -= row_h * 0.6

        for j, (sig, num, default, catchable, use) in enumerate(rows):
            y -= row_h
            bg = "#f8fafc" if j % 2 == 0 else "white"
            ax.add_patch(Rectangle((col_x[0], y - row_h * 0.1),
                                   sum(col_w) + 0.2, row_h * 0.95,
                                   facecolor=bg, edgecolor=COLOR_LIGHT,
                                   lw=0.5))
            ax.text(col_x[0] + 0.1, y + row_h * 0.35, sig,
                    ha="left", va="center", color=group_color,
                    fontsize=10, fontweight="bold", family="monospace")
            ax.text(col_x[1] + 0.1, y + row_h * 0.35, num,
                    ha="left", va="center", color=COLORS["text"],
                    fontsize=9.5, family="monospace")
            ax.text(col_x[2] + 0.1, y + row_h * 0.35, default,
                    ha="left", va="center", color=COLORS["text"], fontsize=9.5)
            catch_color = COLOR_GREEN if catchable == "yes" else COLOR_RED
            ax.text(col_x[3] + 0.1, y + row_h * 0.35, catchable,
                    ha="left", va="center", color=catch_color,
                    fontsize=9.5, fontweight="bold")
            ax.text(col_x[4] + 0.1, y + row_h * 0.35, use,
                    ha="left", va="center", color=COLORS["text2"], fontsize=9)

    # Footnote
    ax.text(6.0, 0.25,
            "Send a signal with:  kill -<NAME|NUM> <PID>  -  e.g. `kill -HUP $(pidof nginx)` "
            "or `kill -9 12345`",
            ha="center", va="center", fontsize=9.5, color=COLORS["text"],
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef3c7",
                      edgecolor=COLOR_AMBER, lw=1.0))

    ax.set_title("Linux signals you actually send",
                 fontsize=14, fontweight="bold", pad=10, color=COLORS["text"])

    _save(fig, "fig5_signals_table.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    fig1_process_states()
    fig2_fork_exec_model()
    fig3_top_dissected()
    fig4_cgroups_namespaces()
    fig5_signals_table()
    print("Wrote 5 figures to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
