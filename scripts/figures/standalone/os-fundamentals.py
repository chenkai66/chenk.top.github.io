"""
Figure generation for the standalone article: Operating System Fundamentals.

Generates 7 figures used in both the EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in editorial print style.

Figures:
    fig1_kernel_architectures   Monolithic vs microkernel vs hybrid -- what
                                lives in kernel mode and what is pushed out
                                to user mode, and the cost of crossing.
    fig2_process_states         Five-state process state machine
                                (new / ready / running / blocked / terminated)
                                with the events that drive each transition.
    fig3_virtual_memory         Virtual memory + paging: per-process virtual
                                address space, page table, physical RAM, and
                                swap. Shows page fault path and TLB.
    fig4_file_system            Inode-based file system: directory entries
                                point to inodes; inodes point to data blocks
                                via direct + indirect pointers.
    fig5_io_subsystem           Layered I/O stack: app -> syscall -> VFS ->
                                generic block layer -> driver -> device,
                                with interrupt + DMA path back up.
    fig6_syscall_interface      System call boundary: user mode app traps
                                into kernel mode, kernel does the work,
                                returns. Shows the privilege transition.
    fig7_schedulers             Four schedulers compared on the same job
                                set: FCFS, SJF, Round-Robin, Linux CFS.
                                Gantt charts + average wait time.

Usage:
    python3 scripts/figures/standalone/os-fundamentals.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
    Polygon,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

C_BLUE = "#2563eb"     # primary
C_PURPLE = "#7c3aed"   # secondary
C_GREEN = "#10b981"    # accent / good
C_AMBER = "#f59e0b"    # warning / highlight
C_RED = "#dc2626"
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"
C_BG_SOFT = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "standalone" / "operating-system-fundamentals-deep-dive"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "standalone" / "操作系统基础深度解析"


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
# Figure 1 -- Kernel architectures
# ---------------------------------------------------------------------------
def fig1_kernel_architectures() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.6))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.6)

    ax.text(6.5, 8.20, "Monolithic vs Microkernel -- Where Does the Code Live?",
            ha="center", va="center", fontsize=17, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.85,
            "left: everything in kernel mode (Linux)   |   right: only the bare minimum (Mach, seL4)",
            ha="center", va="center", fontsize=11, style="italic", color=C_GRAY)

    # ---- Left panel: monolithic ----
    lx, ly, lw, lh = 0.6, 0.7, 5.9, 6.7
    ax.add_patch(FancyBboxPatch(
        (lx, ly), lw, lh,
        boxstyle="round,pad=0.05", facecolor=C_BG_SOFT,
        edgecolor=C_BLUE, lw=1.4,
    ))
    ax.text(lx + lw / 2, ly + lh - 0.3, "Monolithic Kernel  (Linux, BSD)",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_BLUE)

    # User mode strip
    ax.add_patch(FancyBboxPatch(
        (lx + 0.3, ly + lh - 1.5), lw - 0.6, 0.85,
        boxstyle="round,pad=0.03", facecolor="white",
        edgecolor=C_GRAY, lw=1.0,
    ))
    ax.text(lx + 0.5, ly + lh - 0.85, "user mode",
            fontsize=9.5, color=C_GRAY, style="italic")
    user_apps_l = ["bash", "vim", "nginx", "python"]
    ux = lx + 1.25
    for app in user_apps_l:
        ax.add_patch(FancyBboxPatch(
            (ux, ly + lh - 1.35), 0.95, 0.55,
            boxstyle="round,pad=0.02", facecolor=C_BLUE, edgecolor="none", alpha=0.85,
        ))
        ax.text(ux + 0.475, ly + lh - 1.075, app, ha="center", va="center",
                fontsize=9.5, color="white", fontweight="bold", family="monospace")
        ux += 1.05

    # Trap line
    ax.plot([lx + 0.3, lx + lw - 0.3], [ly + lh - 1.75, ly + lh - 1.75],
            color=C_RED, lw=1.4, ls="--")
    ax.text(lx + lw / 2, ly + lh - 1.95, "syscall trap (mode switch)",
            ha="center", va="center", fontsize=9.5, color=C_RED, style="italic")

    # Kernel mode big box
    ax.add_patch(FancyBboxPatch(
        (lx + 0.3, ly + 0.4), lw - 0.6, 4.4,
        boxstyle="round,pad=0.04", facecolor=C_DARK,
        edgecolor="none",
    ))
    ax.text(lx + 0.5, ly + 4.55, "kernel mode -- one big address space",
            fontsize=9.5, color="white", style="italic")

    kernel_subs = [
        ("scheduler",   C_BLUE),
        ("VM / paging", C_PURPLE),
        ("VFS",         C_GREEN),
        ("ext4 / xfs",  C_GREEN),
        ("TCP / IP",    C_AMBER),
        ("device drivers", C_RED),
        ("IPC",         C_BLUE),
        ("syscall API", C_PURPLE),
    ]
    cols, rows = 2, 4
    cell_w = (lw - 1.0) / cols
    cell_h = 0.78
    for i, (name, c) in enumerate(kernel_subs):
        col = i % cols
        row = i // cols
        x = lx + 0.5 + col * cell_w
        y = ly + 4.0 - row * (cell_h + 0.18)
        ax.add_patch(FancyBboxPatch(
            (x, y - cell_h), cell_w - 0.15, cell_h,
            boxstyle="round,pad=0.03", facecolor=c, edgecolor="none", alpha=0.95,
        ))
        ax.text(x + (cell_w - 0.15) / 2, y - cell_h / 2, name,
                ha="center", va="center", fontsize=10.5,
                color="white", fontweight="bold", family="monospace")

    # Property strip
    ax.add_patch(FancyBboxPatch(
        (lx + 0.3, ly + 0.05), lw - 0.6, 0.45,
        boxstyle="round,pad=0.02", facecolor=C_BLUE, edgecolor="none", alpha=0.18,
    ))
    ax.text(lx + lw / 2, ly + 0.275,
            "fast (no IPC inside) | one bug = whole kernel panic",
            ha="center", va="center", fontsize=9.7, color=C_DARK, style="italic")

    # ---- Right panel: microkernel ----
    rx, ry = 6.7, 0.7
    rw, rh = 5.9, 6.7
    ax.add_patch(FancyBboxPatch(
        (rx, ry), rw, rh,
        boxstyle="round,pad=0.05", facecolor=C_BG_SOFT,
        edgecolor=C_PURPLE, lw=1.4,
    ))
    ax.text(rx + rw / 2, ry + rh - 0.3, "Microkernel  (Mach, seL4, QNX)",
            ha="center", va="center", fontsize=13, fontweight="bold", color=C_PURPLE)

    # User mode strip with services
    ax.add_patch(FancyBboxPatch(
        (rx + 0.3, ry + 1.5), rw - 0.6, 4.0,
        boxstyle="round,pad=0.03", facecolor="white",
        edgecolor=C_GRAY, lw=1.0,
    ))
    ax.text(rx + 0.5, ry + 5.25, "user mode -- services run as ordinary processes",
            fontsize=9.5, color=C_GRAY, style="italic")

    services = [
        ("file server",   C_GREEN),
        ("network stack", C_AMBER),
        ("device driver", C_RED),
        ("display server", C_BLUE),
        ("vim",           C_BLUE),
        ("python",        C_BLUE),
    ]
    svw = (rw - 1.0) / 3
    svh = 0.7
    for i, (name, c) in enumerate(services):
        col = i % 3
        row = i // 3
        x = rx + 0.5 + col * svw
        y = ry + 4.7 - row * (svh + 0.2)
        ax.add_patch(FancyBboxPatch(
            (x, y - svh), svw - 0.15, svh,
            boxstyle="round,pad=0.03", facecolor=c, edgecolor="none", alpha=0.85,
        ))
        ax.text(x + (svw - 0.15) / 2, y - svh / 2, name,
                ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold", family="monospace")

    # IPC arrows showing message-passing
    ax.annotate("", xy=(rx + 1.6, ry + 3.3), xytext=(rx + 3.3, ry + 4.45),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color=C_PURPLE,
                                connectionstyle="arc3,rad=0.2"))
    ax.text(rx + 2.6, ry + 3.6, "IPC", ha="center",
            fontsize=9, color=C_PURPLE, fontweight="bold", family="monospace")

    # Trap line
    ax.plot([rx + 0.3, rx + rw - 0.3], [ry + 1.45, ry + 1.45],
            color=C_RED, lw=1.4, ls="--")

    # Kernel mode strip -- thin
    ax.add_patch(FancyBboxPatch(
        (rx + 0.3, ry + 0.55), rw - 0.6, 0.85,
        boxstyle="round,pad=0.04", facecolor=C_DARK,
        edgecolor="none",
    ))
    ax.text(rx + 0.5, ry + 1.25, "kernel mode -- minimal",
            fontsize=9.5, color="white", style="italic")
    micro_core = [("IPC", C_PURPLE), ("scheduler", C_BLUE), ("VM", C_GREEN)]
    cw = (rw - 1.2) / 3
    for i, (name, c) in enumerate(micro_core):
        x = rx + 0.5 + i * (cw + 0.1)
        ax.add_patch(FancyBboxPatch(
            (x, ry + 0.7), cw, 0.5,
            boxstyle="round,pad=0.02", facecolor=c, edgecolor="none", alpha=0.95,
        ))
        ax.text(x + cw / 2, ry + 0.95, name, ha="center", va="center",
                fontsize=10.5, color="white", fontweight="bold", family="monospace")

    # Property strip
    ax.add_patch(FancyBboxPatch(
        (rx + 0.3, ry + 0.05), rw - 0.6, 0.45,
        boxstyle="round,pad=0.02", facecolor=C_PURPLE, edgecolor="none", alpha=0.18,
    ))
    ax.text(rx + rw / 2, ry + 0.275,
            "isolated (driver crash != kernel panic) | every IPC is a context switch",
            ha="center", va="center", fontsize=9.7, color=C_DARK, style="italic")

    _save(fig, "fig1_kernel_architectures")


# ---------------------------------------------------------------------------
# Figure 2 -- Process state diagram
# ---------------------------------------------------------------------------
def fig2_process_states() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.2))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.2)

    ax.text(6.5, 7.75, "The Five-State Process Lifecycle",
            ha="center", va="center", fontsize=18, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.35,
            "every running process is in exactly one of these five states at any instant",
            ha="center", va="center", fontsize=11.2, style="italic", color=C_GRAY)

    # Nodes: (x, y, label, sub, color)
    nodes = {
        "new":        (1.6, 4.5, "NEW",        "being created",    C_GRAY),
        "ready":      (4.6, 4.5, "READY",      "waiting for CPU",  C_BLUE),
        "running":    (8.0, 4.5, "RUNNING",    "executing on CPU", C_GREEN),
        "blocked":    (8.0, 1.6, "BLOCKED",    "waiting for I/O",  C_AMBER),
        "terminated": (11.4, 4.5, "TERMINATED", "done / killed",   C_RED),
    }

    def draw_node(key, w=2.0, h=1.05):
        x, y, label, sub, color = nodes[key]
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06", facecolor=color, edgecolor="none",
            alpha=0.95, zorder=2,
        ))
        ax.text(x, y + 0.18, label, ha="center", va="center",
                fontsize=12.5, fontweight="bold", color="white")
        ax.text(x, y - 0.22, sub, ha="center", va="center",
                fontsize=9.5, color="white", style="italic")

    for k in nodes:
        draw_node(k)

    # Transitions: list of (from, to, label, curve, dy)
    def arrow(p0, p1, key, curve=0.0, color=C_DARK, dx=0.0, dy=0.0,
              fontsize=10.2, sub=None):
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
        # Label box
        w = max(0.08 * len(key) + 0.4, 0.9)
        ax.add_patch(FancyBboxPatch(
            (mx - w / 2, my - 0.18), w, 0.36,
            boxstyle="round,pad=0.02", facecolor="white",
            edgecolor=color, lw=1.2, zorder=3,
        ))
        ax.text(mx, my, key, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=color, zorder=4)
        if sub:
            ax.text(mx, my - 0.32, sub, ha="center", va="center",
                    fontsize=8.7, color=C_GRAY, style="italic")

    # NEW -> READY
    arrow((nodes["new"][0] + 1.0, nodes["new"][1]),
          (nodes["ready"][0] - 1.0, nodes["ready"][1]),
          "admit", color=C_BLUE, dy=0.32, sub="loaded into memory")
    # READY -> RUNNING (top arc)
    arrow((nodes["ready"][0] + 1.0, nodes["ready"][1] + 0.25),
          (nodes["running"][0] - 1.0, nodes["running"][1] + 0.25),
          "dispatch", curve=0.18, color=C_GREEN, dy=0.45, sub="scheduler picks me")
    # RUNNING -> READY (bottom arc): preempt
    arrow((nodes["running"][0] - 1.0, nodes["running"][1] - 0.25),
          (nodes["ready"][0] + 1.0, nodes["ready"][1] - 0.25),
          "preempt", curve=0.18, color=C_AMBER, dy=-0.45, sub="time slice / higher prio")
    # RUNNING -> BLOCKED
    arrow((nodes["running"][0] - 0.3, nodes["running"][1] - 0.55),
          (nodes["blocked"][0] - 0.3, nodes["blocked"][1] + 0.55),
          "wait for I/O", color=C_AMBER, dx=-0.95, fontsize=10,
          sub="read() / sleep() / lock")
    # BLOCKED -> READY (long arc)
    arrow((nodes["blocked"][0] - 1.0, nodes["blocked"][1] + 0.05),
          (nodes["ready"][0] + 0.5, nodes["ready"][1] - 0.55),
          "I/O done", curve=-0.25, color=C_BLUE, dx=-0.7, dy=-0.55,
          sub="device interrupt wakes me")
    # RUNNING -> TERMINATED
    arrow((nodes["running"][0] + 1.0, nodes["running"][1]),
          (nodes["terminated"][0] - 1.0, nodes["terminated"][1]),
          "exit", color=C_RED, dy=0.32, sub="exit() / signal")

    # Bottom legend strip explaining colors of transitions
    ax.add_patch(FancyBboxPatch(
        (0.5, 0.25), 12.0, 0.75,
        boxstyle="round,pad=0.05", facecolor=C_BG_SOFT,
        edgecolor=C_DARK, lw=1.0,
    ))
    ax.text(0.85, 0.78, "Three things move you off CPU:",
            fontsize=10.8, fontweight="bold", color=C_DARK)
    items = [
        ("voluntary -- syscall blocks (read, lock, sleep)", C_AMBER),
        ("involuntary -- preemption (time slice expired)", C_BLUE),
        ("permanent -- exit() or fatal signal",            C_RED),
    ]
    x = 0.85
    for label, color in items:
        ax.add_patch(Rectangle((x, 0.36), 0.2, 0.2, facecolor=color, edgecolor="none"))
        ax.text(x + 0.3, 0.45, label, fontsize=9.7, color=C_DARK, va="center")
        x += 4.05

    _save(fig, "fig2_process_states")


# ---------------------------------------------------------------------------
# Figure 3 -- Virtual memory + paging
# ---------------------------------------------------------------------------
def fig3_virtual_memory() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.6))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.6)

    ax.text(6.5, 8.20, "Virtual Memory -- Every Process Thinks It Owns the Machine",
            ha="center", va="center", fontsize=17, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.85,
            "the page table + MMU translate virtual addresses into physical frames; the OS handles the rest",
            ha="center", va="center", fontsize=11, style="italic", color=C_GRAY)

    # Three columns: virtual / page table / physical
    # ---- Virtual address space (process A) ----
    vx, vy, vw, vh = 0.6, 1.0, 2.6, 6.3
    ax.add_patch(FancyBboxPatch(
        (vx, vy), vw, vh,
        boxstyle="round,pad=0.04", facecolor="white",
        edgecolor=C_BLUE, lw=1.4,
    ))
    ax.text(vx + vw / 2, vy + vh + 0.18, "Virtual Address Space\n(process A, 0..2^48)",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color=C_BLUE)

    # Virtual segments
    segs = [
        ("stack -- grows down", C_PURPLE, 0.85),
        ("(unmapped)",           C_LIGHT,  1.10),
        ("heap -- grows up",     C_GREEN,  0.85),
        ("BSS / data",           C_AMBER,  0.55),
        ("text (code)",          C_BLUE,   0.85),
    ]
    sy = vy + vh - 0.15
    seg_rects = []
    for label, color, h in segs:
        ax.add_patch(Rectangle((vx + 0.12, sy - h), vw - 0.24, h,
                               facecolor=color, edgecolor="white", lw=1.2, alpha=0.85))
        ax.text(vx + vw / 2, sy - h / 2, label, ha="center", va="center",
                fontsize=9.5, color="white" if color != C_LIGHT else C_GRAY,
                fontweight="bold", style="italic" if color == C_LIGHT else "normal")
        seg_rects.append((sy - h / 2, color))
        sy -= h
    # Address tick marks
    ax.text(vx - 0.05, vy + vh - 0.05, "high", ha="right", fontsize=9, color=C_GRAY)
    ax.text(vx - 0.05, vy + 0.05, "0x0", ha="right", fontsize=9, color=C_GRAY)

    # ---- Page table ----
    px, py, pw, ph = 4.3, 1.0, 3.6, 6.3
    ax.add_patch(FancyBboxPatch(
        (px, py), pw, ph,
        boxstyle="round,pad=0.04", facecolor=C_BG_SOFT,
        edgecolor=C_DARK, lw=1.4,
    ))
    ax.text(px + pw / 2, py + ph + 0.18, "Page Table\n(per process, kept by OS)",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color=C_DARK)

    # Entries: VPN | PFN | bits
    headers = ["VPN", "PFN", "flags"]
    col_w = [0.85, 0.85, 1.5]
    hx = px + 0.2
    for i, h in enumerate(headers):
        ax.text(hx + col_w[i] / 2, py + ph - 0.3, h, ha="center", va="center",
                fontsize=10, fontweight="bold", color=C_DARK)
        hx += col_w[i] + 0.1

    rows = [
        ("0x07f", "0x21a", "R W X", C_PURPLE),
        ("0x07e", "  -- ",  "--   ",  C_GRAY),
        ("0x040", "0x118", "R W  ",  C_GREEN),
        ("0x03f", "swap",  "R W S",  C_AMBER),
        ("0x002", "0x019", "R     ", C_BLUE),
        ("0x001", "0x004", "R   X", C_BLUE),
        ("0x000", "0x000", "--   ",  C_GRAY),
    ]
    ry = py + ph - 0.7
    rh = 0.52
    pte_y = {}
    for vpn, pfn, flags, color in rows:
        ax.add_patch(Rectangle((px + 0.15, ry - rh), pw - 0.3, rh,
                               facecolor="white" if color != C_GRAY else C_LIGHT,
                               edgecolor=C_LIGHT, lw=0.8))
        hx = px + 0.2
        for i, val in enumerate([vpn, pfn, flags]):
            ax.text(hx + col_w[i] / 2, ry - rh / 2, val, ha="center", va="center",
                    fontsize=9.5, family="monospace",
                    color=color if color != C_GRAY else C_GRAY,
                    fontweight="bold")
            hx += col_w[i] + 0.1
        pte_y[vpn] = ry - rh / 2
        ry -= rh

    # MMU + TLB block on top of page table
    ax.add_patch(FancyBboxPatch(
        (px + 0.4, py - 0.05), pw - 0.8, 0.7,
        boxstyle="round,pad=0.03", facecolor=C_DARK, edgecolor="none",
    ))
    ax.text(px + pw / 2, py + 0.3, "MMU  (with TLB cache)",
            ha="center", va="center", fontsize=10.2, color="white",
            fontweight="bold", family="monospace")

    # ---- Physical RAM ----
    rx_, ry_, rw_, rh_ = 9.0, 1.0, 2.3, 6.3
    ax.add_patch(FancyBboxPatch(
        (rx_, ry_), rw_, rh_,
        boxstyle="round,pad=0.04", facecolor="white",
        edgecolor=C_GREEN, lw=1.4,
    ))
    ax.text(rx_ + rw_ / 2, ry_ + rh_ + 0.18, "Physical RAM\n(shared by all processes)",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color=C_GREEN)

    frames = [
        ("0x21a", C_PURPLE, "stack(A)"),
        ("0x118", C_GREEN,  "heap(A)"),
        ("0x019", C_BLUE,   "data(A)"),
        ("0x004", C_BLUE,   "code(A)"),
        ("0x300", C_RED,    "code(B)"),
        ("0x301", C_RED,    "heap(B)"),
        ("--",    C_LIGHT,  "free"),
        ("0x402", C_AMBER,  "kernel"),
    ]
    fy = ry_ + rh_ - 0.2
    fh = 0.73
    frame_y = {}
    for label, color, content in frames:
        ax.add_patch(Rectangle((rx_ + 0.12, fy - fh), rw_ - 0.24, fh,
                               facecolor=color, edgecolor="white", lw=1.2, alpha=0.85))
        ax.text(rx_ + 0.25, fy - fh / 2 + 0.13, label,
                fontsize=9, color="white" if color != C_LIGHT else C_GRAY,
                fontweight="bold", family="monospace")
        ax.text(rx_ + rw_ - 0.3, fy - fh / 2 - 0.1, content,
                ha="right", fontsize=8.7,
                color="white" if color != C_LIGHT else C_GRAY, style="italic")
        frame_y[label] = fy - fh / 2
        fy -= fh

    # ---- Swap (disk) ----
    sx, sy_ = 11.6, 1.0
    sw, sh = 1.2, 6.3
    ax.add_patch(FancyBboxPatch(
        (sx, sy_), sw, sh,
        boxstyle="round,pad=0.04", facecolor=C_AMBER,
        edgecolor="none", alpha=0.4,
    ))
    ax.text(sx + sw / 2, sy_ + sh + 0.18, "Swap\n(disk)",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color=C_AMBER)
    ax.text(sx + sw / 2, sy_ + sh / 2, "evicted\npages\nlive here",
            ha="center", va="center", fontsize=10, color=C_DARK,
            fontweight="bold", style="italic")

    # ---- Connection arrows ----
    # Virtual stack -> PTE 0x07f -> Frame 0x21a
    ax.annotate("", xy=(px + 0.05, pte_y["0x07f"]),
                xytext=(vx + vw + 0.05, seg_rects[0][0]),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=C_PURPLE))
    ax.annotate("", xy=(rx_ - 0.05, frame_y["0x21a"]),
                xytext=(px + pw + 0.05, pte_y["0x07f"]),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=C_PURPLE))
    # Virtual code -> PTE 0x001 -> Frame 0x004
    ax.annotate("", xy=(px + 0.05, pte_y["0x001"]),
                xytext=(vx + vw + 0.05, seg_rects[4][0]),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=C_BLUE))
    ax.annotate("", xy=(rx_ - 0.05, frame_y["0x004"]),
                xytext=(px + pw + 0.05, pte_y["0x001"]),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=C_BLUE))
    # PTE swap -> swap area  (page fault path, dashed red)
    ax.annotate("", xy=(sx + 0.05, sy_ + sh / 2),
                xytext=(px + pw + 0.05, pte_y["0x03f"]),
                arrowprops=dict(arrowstyle="->", lw=1.4, color=C_RED, ls="--"))
    ax.text((px + pw + sx) / 2, pte_y["0x03f"] + 0.25,
            "page fault -> swap in",
            ha="center", fontsize=9, color=C_RED, fontweight="bold", style="italic")

    # Bottom legend / formula
    ax.add_patch(FancyBboxPatch(
        (0.5, 0.05), 12.2, 0.7,
        boxstyle="round,pad=0.04", facecolor=C_BG_SOFT,
        edgecolor=C_DARK, lw=1.0,
    ))
    ax.text(0.85, 0.55,
            "translation:",
            fontsize=10.5, fontweight="bold", color=C_DARK)
    ax.text(2.0, 0.55,
            "virtual addr  =  [VPN | offset]    ->    physical addr  =  [PFN | offset]",
            fontsize=10.3, color=C_DARK, family="monospace")
    ax.text(0.85, 0.22,
            "TLB hit -> 1 cycle  |  TLB miss + PT walk -> ~100 cycles  |  page fault (swap) -> ~10 ms (100,000x slower)",
            fontsize=9.7, color=C_GRAY, style="italic")

    _save(fig, "fig3_virtual_memory")


# ---------------------------------------------------------------------------
# Figure 4 -- Inode-based file system
# ---------------------------------------------------------------------------
def fig4_file_system() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.6))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.6)

    ax.text(6.5, 8.20, "Inode File System -- Names, Metadata, and Blocks Are Three Different Things",
            ha="center", va="center", fontsize=16.5, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.85,
            "directory entry --> inode --> data blocks (direct + indirect pointers)",
            ha="center", va="center", fontsize=11, style="italic", color=C_GRAY)

    # ---- Directory entries (left) ----
    dx, dy, dw, dh = 0.5, 1.0, 3.1, 6.3
    ax.add_patch(FancyBboxPatch(
        (dx, dy), dw, dh,
        boxstyle="round,pad=0.04", facecolor="white",
        edgecolor=C_BLUE, lw=1.4,
    ))
    ax.text(dx + dw / 2, dy + dh + 0.18, "Directory  /home/alice/",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color=C_BLUE)

    dirents = [
        ("notes.txt", "42"),
        ("photo.jpg", "73"),
        ("draft.md",  "42"),  # hard link -- same inode!
        ("project/",  "85"),
        (".bashrc",   "11"),
    ]
    eh = 0.85
    ey = dy + dh - 0.4
    dirent_y = {}
    for name, ino in dirents:
        is_link = name == "draft.md"
        color = C_AMBER if is_link else C_BLUE
        ax.add_patch(FancyBboxPatch(
            (dx + 0.18, ey - eh), dw - 0.36, eh,
            boxstyle="round,pad=0.03", facecolor=color, edgecolor="none", alpha=0.85,
        ))
        ax.text(dx + 0.3, ey - 0.27, name, ha="left", va="center",
                fontsize=10.5, color="white", fontweight="bold", family="monospace")
        ax.text(dx + dw - 0.5, ey - 0.27, "ino " + ino, ha="right", va="center",
                fontsize=9.5, color="white", family="monospace")
        if is_link:
            ax.text(dx + 0.3, ey - 0.6, "(hard link to inode 42)",
                    fontsize=8.5, color="white", style="italic")
        dirent_y[ino if not is_link else "42b"] = ey - eh / 2
        ey -= eh + 0.12

    # ---- Inode table (middle) ----
    ix, iy, iw, ih = 4.6, 1.0, 4.4, 6.3
    ax.add_patch(FancyBboxPatch(
        (ix, iy), iw, ih,
        boxstyle="round,pad=0.04", facecolor=C_BG_SOFT,
        edgecolor=C_DARK, lw=1.4,
    ))
    ax.text(ix + iw / 2, iy + ih + 0.18, "Inode Table",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color=C_DARK)

    # Show ONE detailed inode (inode 42)
    inode_x = ix + 0.25
    inode_y = iy + ih - 0.5
    inode_w = iw - 0.5
    ax.add_patch(FancyBboxPatch(
        (inode_x, inode_y - 4.6), inode_w, 4.6,
        boxstyle="round,pad=0.04", facecolor="white",
        edgecolor=C_PURPLE, lw=1.6,
    ))
    ax.text(inode_x + 0.15, inode_y - 0.25, "inode  #42",
            fontsize=11, fontweight="bold", color=C_PURPLE, family="monospace")
    ax.text(inode_x + inode_w - 0.15, inode_y - 0.25, "(file)",
            ha="right", fontsize=9.5, color=C_GRAY, style="italic")

    # Metadata rows
    meta_lines = [
        ("type",    "regular file"),
        ("mode",    "0644 (rw-r--r--)"),
        ("uid:gid", "1000:1000"),
        ("size",    "13,824 bytes"),
        ("atime",   "2024-02-16 09:00"),
        ("mtime",   "2024-02-16 08:55"),
        ("nlink",   "2  <- two names point here"),
    ]
    my = inode_y - 0.55
    for k, v in meta_lines:
        ax.text(inode_x + 0.2, my, k, fontsize=9.6, color=C_DARK,
                fontweight="bold", family="monospace")
        ax.text(inode_x + 1.45, my, v, fontsize=9.6, color=C_DARK,
                family="monospace")
        my -= 0.32

    # Block pointer section
    bp_y = my - 0.05
    ax.plot([inode_x + 0.1, inode_x + inode_w - 0.1], [bp_y, bp_y],
            color=C_LIGHT, lw=1.0)
    ax.text(inode_x + 0.15, bp_y - 0.25, "block pointers:",
            fontsize=9.6, fontweight="bold", color=C_DARK, family="monospace")

    # Direct pointers
    dp_y = bp_y - 0.7
    direct = ["#101", "#102", "#103", "#104"]
    box_w = 0.55
    for i, blk in enumerate(direct):
        x = inode_x + 0.2 + i * (box_w + 0.05)
        ax.add_patch(FancyBboxPatch(
            (x, dp_y - 0.25), box_w, 0.45,
            boxstyle="round,pad=0.02", facecolor=C_GREEN, edgecolor="none", alpha=0.9,
        ))
        ax.text(x + box_w / 2, dp_y - 0.025, blk,
                ha="center", va="center", fontsize=8.8, color="white",
                fontweight="bold", family="monospace")
    ax.text(inode_x + 0.2 + 4 * (box_w + 0.05), dp_y - 0.025,
            "...12 direct (= 48 KiB)", fontsize=8.5, color=C_GRAY,
            va="center", style="italic")

    # Indirect pointer
    ind_y = dp_y - 0.55
    ax.add_patch(FancyBboxPatch(
        (inode_x + 0.2, ind_y - 0.25), 1.4, 0.45,
        boxstyle="round,pad=0.02", facecolor=C_AMBER, edgecolor="none", alpha=0.9,
    ))
    ax.text(inode_x + 0.9, ind_y - 0.025, "indirect -> #207",
            ha="center", va="center", fontsize=8.8, color="white",
            fontweight="bold", family="monospace")
    ax.text(inode_x + 1.7, ind_y - 0.025,
            "(holds 1024 more block ptrs)",
            fontsize=8.5, color=C_GRAY, va="center", style="italic")

    # ---- Data blocks (right) ----
    bx, by, bw, bh = 9.6, 1.0, 3.1, 6.3
    ax.add_patch(FancyBboxPatch(
        (bx, by), bw, bh,
        boxstyle="round,pad=0.04", facecolor="white",
        edgecolor=C_GREEN, lw=1.4,
    ))
    ax.text(bx + bw / 2, by + bh + 0.18, "Data Blocks  (4 KiB each)",
            ha="center", va="center", fontsize=10.8, fontweight="bold", color=C_GREEN)

    blocks = [
        ("#101", '"# notes\\nbuy mil"', C_GREEN),
        ("#102", '"k, eggs\\n# tod"',   C_GREEN),
        ("#103", '"o\\n- read paper"',  C_GREEN),
        ("#104", '"...rest of f"',      C_GREEN),
        ("#207", "[#205, #208, ...]",   C_AMBER),
        ("#205", '"...continues"',      C_GREEN),
    ]
    by_pos = by + bh - 0.4
    blk_y = {}
    for blk, content, color in blocks:
        ax.add_patch(FancyBboxPatch(
            (bx + 0.18, by_pos - 0.7), bw - 0.36, 0.7,
            boxstyle="round,pad=0.03", facecolor=color, edgecolor="none", alpha=0.85,
        ))
        ax.text(bx + 0.3, by_pos - 0.22, blk, fontsize=9.7,
                color="white", fontweight="bold", family="monospace")
        ax.text(bx + 0.3, by_pos - 0.5, content, fontsize=8.5,
                color="white", family="monospace")
        blk_y[blk] = by_pos - 0.35
        by_pos -= 0.85

    # ---- Connections ----
    # dirent ino 42 -> inode 42 header
    ax.annotate("", xy=(inode_x, inode_y - 0.25),
                xytext=(dx + dw - 0.05, dirent_y["42"]),
                arrowprops=dict(arrowstyle="->", lw=1.3, color=C_BLUE))
    # dirent draft.md -> same inode 42 (hard link)
    ax.annotate("", xy=(inode_x, inode_y - 0.25),
                xytext=(dx + dw - 0.05, dirent_y["42b"]),
                arrowprops=dict(arrowstyle="->", lw=1.3, color=C_AMBER,
                                connectionstyle="arc3,rad=-0.18"))
    # direct ptr -> block 101
    ax.annotate("", xy=(bx + 0.05, blk_y["#101"]),
                xytext=(inode_x + 0.475, dp_y - 0.05),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=C_GREEN))
    # indirect ptr -> #207 -> #205
    ax.annotate("", xy=(bx + 0.05, blk_y["#207"]),
                xytext=(inode_x + 0.9, ind_y - 0.25),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=C_AMBER))
    ax.annotate("", xy=(bx + bw / 2, blk_y["#205"] + 0.35),
                xytext=(bx + bw / 2, blk_y["#207"] - 0.35),
                arrowprops=dict(arrowstyle="->", lw=1.0, color=C_AMBER,
                                connectionstyle="arc3,rad=0.5"))

    # Bottom takeaway
    ax.add_patch(FancyBboxPatch(
        (0.5, 0.05), 12.2, 0.7,
        boxstyle="round,pad=0.04", facecolor=C_BG_SOFT,
        edgecolor=C_DARK, lw=1.0,
    ))
    ax.text(0.85, 0.55, "Three layers of indirection:",
            fontsize=10.5, fontweight="bold", color=C_DARK)
    ax.text(3.5, 0.55,
            "name -> inode  (rename / hard link)   |   "
            "inode -> block (where the bytes live)   |   "
            "block -> disk LBA (storage layer)",
            fontsize=9.8, color=C_DARK)
    ax.text(0.85, 0.22,
            "rm just removes a name + decrements nlink. The inode (and its blocks) survive until nlink == 0.",
            fontsize=9.7, color=C_GRAY, style="italic")

    _save(fig, "fig4_file_system")


# ---------------------------------------------------------------------------
# Figure 5 -- I/O subsystem layered stack
# ---------------------------------------------------------------------------
def fig5_io_subsystem() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.6))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.6)

    ax.text(6.5, 8.20, "The I/O Stack -- One read() Call, Seven Layers Down",
            ha="center", va="center", fontsize=17, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.85,
            "going down: synchronous request   |   coming back up: interrupt-driven completion",
            ha="center", va="center", fontsize=11, style="italic", color=C_GRAY)

    # Layers stacked vertically (down request, up completion via DMA + IRQ)
    layers = [
        ("application",          "read(fd, buf, 4096)",                 C_BLUE,   True),
        ("syscall interface",    "trap into kernel; arg validation",    C_BLUE,   True),
        ("VFS (virtual FS)",     "dispatch by fd's file_operations",    C_GREEN,  True),
        ("file system  (ext4)",  "block = ino->extent_tree->LBA",       C_GREEN,  True),
        ("page cache",           "if hit: copy_to_user, return",        C_PURPLE, True),
        ("block layer",          "build BIO, merge, schedule (mq)",     C_PURPLE, True),
        ("device driver  (NVMe)", "submit to hardware queue",           C_AMBER,  True),
        ("hardware  (SSD / disk / NIC)", "fetch via DMA, raise IRQ",    C_RED,    False),
    ]

    bx0, bx1 = 1.6, 8.0
    by_top = 7.2
    h = 0.65
    gap = 0.13
    layer_y = []
    for i, (name, sub, color, _) in enumerate(layers):
        y = by_top - i * (h + gap)
        layer_y.append(y)
        ax.add_patch(FancyBboxPatch(
            (bx0, y - h), bx1 - bx0, h,
            boxstyle="round,pad=0.03", facecolor=color, edgecolor="none", alpha=0.9,
        ))
        ax.text(bx0 + 0.18, y - h / 2 + 0.09, name,
                fontsize=10.7, color="white", fontweight="bold", family="monospace")
        ax.text(bx0 + 0.18, y - h / 2 - 0.13, sub,
                fontsize=9.2, color="white", style="italic")

    # Down arrow (request path) on the left
    ax.annotate("", xy=(bx0 - 0.5, layer_y[-1] - h + 0.05),
                xytext=(bx0 - 0.5, layer_y[0]),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=C_DARK))
    ax.text(bx0 - 0.7, (layer_y[0] + layer_y[-1] - h) / 2, "request",
            ha="center", va="center", fontsize=11, color=C_DARK,
            fontweight="bold", rotation=90)

    # Up arrow (completion) on the right -- bypasses page cache fill
    ax.annotate("", xy=(bx1 + 0.5, layer_y[0]),
                xytext=(bx1 + 0.5, layer_y[-1] - h + 0.05),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=C_RED))
    ax.text(bx1 + 0.75, (layer_y[0] + layer_y[-1] - h) / 2,
            "completion (IRQ + DMA)",
            ha="center", va="center", fontsize=11, color=C_RED,
            fontweight="bold", rotation=270)

    # Privilege boundary line between layers 1 and 2 (syscall) and below 7 (kernel/HW)
    # User/Kernel boundary
    yb = (layer_y[0] - h + layer_y[1]) / 2
    ax.plot([0.5, 12.5], [yb, yb], color=C_RED, ls="--", lw=1.4)
    ax.text(12.3, yb + 0.12, "user / kernel", ha="right", fontsize=9.5,
            color=C_RED, fontweight="bold", style="italic")
    # Kernel/HW boundary
    yh = (layer_y[-2] - h + layer_y[-1]) / 2
    ax.plot([0.5, 12.5], [yh, yh], color=C_RED, ls="--", lw=1.4)
    ax.text(12.3, yh + 0.12, "kernel / hardware", ha="right", fontsize=9.5,
            color=C_RED, fontweight="bold", style="italic")

    # Side annotations: page cache fast path
    ax.add_patch(FancyBboxPatch(
        (8.4, layer_y[4] - h - 0.05), 4.0, h + 0.1,
        boxstyle="round,pad=0.03", facecolor=C_BG_SOFT,
        edgecolor=C_PURPLE, lw=1.0,
    ))
    ax.text(8.55, layer_y[4] - h / 2 + 0.1, "fast path:",
            fontsize=9.5, fontweight="bold", color=C_PURPLE)
    ax.text(8.55, layer_y[4] - h / 2 - 0.13,
            "cache hit -> return in microseconds",
            fontsize=8.8, color=C_DARK, style="italic")

    ax.add_patch(FancyBboxPatch(
        (8.4, layer_y[5] - h - 0.05), 4.0, h + 0.1,
        boxstyle="round,pad=0.03", facecolor=C_BG_SOFT,
        edgecolor=C_AMBER, lw=1.0,
    ))
    ax.text(8.55, layer_y[5] - h / 2 + 0.1, "elevator / scheduler:",
            fontsize=9.5, fontweight="bold", color=C_AMBER)
    ax.text(8.55, layer_y[5] - h / 2 - 0.13,
            "merge adjacent BIOs, reduce seeks",
            fontsize=8.8, color=C_DARK, style="italic")

    # Bottom timing strip
    ax.add_patch(FancyBboxPatch(
        (0.5, 0.15), 12.0, 0.85,
        boxstyle="round,pad=0.04", facecolor=C_BG_SOFT,
        edgecolor=C_DARK, lw=1.0,
    ))
    ax.text(0.85, 0.78, "Latency budget for one 4 KiB read:",
            fontsize=10.5, fontweight="bold", color=C_DARK)
    ax.text(0.85, 0.45,
            "page cache hit  ~1 us   |   NVMe SSD  ~80 us   |   SATA SSD  ~150 us   |   "
            "7200 rpm HDD  ~10 ms   |   network round trip  ~500 us",
            fontsize=10, color=C_DARK, family="monospace")
    ax.text(0.85, 0.20,
            "the layers above the device exist to make the device-level number look smaller -- "
            "caching, batching, prefetching, async.",
            fontsize=9.5, color=C_GRAY, style="italic")

    _save(fig, "fig5_io_subsystem")


# ---------------------------------------------------------------------------
# Figure 6 -- System call interface
# ---------------------------------------------------------------------------
def fig6_syscall_interface() -> None:
    fig, ax = plt.subplots(figsize=(13, 8.4))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 8.4)

    ax.text(6.5, 8.05, "The System Call Boundary -- Crossing from User to Kernel",
            ha="center", va="center", fontsize=17, fontweight="bold", color=C_DARK)
    ax.text(6.5, 7.7,
            "the only legal way for a user-mode program to ask the kernel for anything",
            ha="center", va="center", fontsize=11, style="italic", color=C_GRAY)

    # Two horizontal bands: user mode (top) and kernel mode (bottom)
    user_y_top = 7.1
    user_y_bot = 4.4
    kern_y_top = 4.0
    kern_y_bot = 1.5

    # User band background
    ax.add_patch(Rectangle((0.5, user_y_bot), 12.0, user_y_top - user_y_bot,
                           facecolor=C_BG_SOFT, edgecolor=C_BLUE, lw=1.4))
    ax.text(0.7, user_y_top - 0.25, "USER MODE  (ring 3, restricted)",
            fontsize=10.8, fontweight="bold", color=C_BLUE)

    # Kernel band background
    ax.add_patch(Rectangle((0.5, kern_y_bot), 12.0, kern_y_top - kern_y_bot,
                           facecolor=C_DARK, edgecolor="none"))
    ax.text(0.7, kern_y_top - 0.25, "KERNEL MODE  (ring 0, full privilege)",
            fontsize=10.8, fontweight="bold", color="white")

    # Boundary -- the trap line
    yb = (user_y_bot + kern_y_top) / 2
    ax.plot([0.5, 12.5], [yb, yb], color=C_RED, lw=2.2, ls="--")
    ax.text(0.7, yb + 0.13, "syscall boundary  (trap instruction)",
            fontsize=10, color=C_RED, fontweight="bold", style="italic")

    # ---- Six numbered steps across the diagram ----
    steps = [
        # (x, label_top, code, side_note)
        (1.7,  "1.  C library wrapper",
         "n = read(fd, buf, 4096);",
         "ordinary function call into glibc"),
        (4.6,  "2.  set up registers",
         "rax=0  rdi=fd  rsi=buf\nrdx=4096",
         "syscall number + 6 args"),
        (7.5,  "3.  trap instruction",
         "syscall   ; or int 0x80",
         "CPU jumps to kernel handler"),
        (10.4, "4.  return to caller",
         "ret value in rax\nresume in user mode",
         "kernel restores user context"),
    ]
    for x, label, code, note in steps:
        # User-side label box
        ax.add_patch(FancyBboxPatch(
            (x - 1.2, user_y_top - 1.3), 2.4, 0.85,
            boxstyle="round,pad=0.03", facecolor="white",
            edgecolor=C_BLUE, lw=1.3,
        ))
        ax.text(x, user_y_top - 0.7, label, ha="center", va="center",
                fontsize=10.3, fontweight="bold", color=C_BLUE)
        ax.text(x, user_y_top - 1.0, code, ha="center", va="center",
                fontsize=8.7, color=C_DARK, family="monospace")
        ax.text(x, user_y_top - 1.55, note, ha="center", va="center",
                fontsize=8.5, color=C_GRAY, style="italic")

    # Down arrow at trap (step 3 column) crossing into kernel
    ax.annotate("", xy=(7.5, kern_y_top - 0.2), xytext=(7.5, user_y_bot + 0.1),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=C_RED))
    # Up arrow returning (step 4 column)
    ax.annotate("", xy=(10.4, user_y_bot + 0.1), xytext=(10.4, kern_y_top - 0.2),
                arrowprops=dict(arrowstyle="->", lw=2.2, color=C_GREEN))

    # ---- Kernel-side work boxes ----
    work = [
        (4.6,  "5.  dispatcher",
         "sys_call_table[rax]\n-> sys_read",
         C_PURPLE),
        (7.5,  "6.  kernel work",
         "VFS -> ext4 -> page cache\n-> copy_to_user(buf)",
         C_GREEN),
        (10.4, "7.  return path",
         "set rax = bytes_read\niretq / sysret",
         C_AMBER),
    ]
    for x, label, code, color in work:
        ax.add_patch(FancyBboxPatch(
            (x - 1.25, kern_y_bot + 0.55), 2.5, 1.5,
            boxstyle="round,pad=0.04", facecolor=color, edgecolor="none", alpha=0.95,
        ))
        ax.text(x, kern_y_bot + 1.7, label, ha="center", va="center",
                fontsize=10.3, fontweight="bold", color="white")
        ax.text(x, kern_y_bot + 1.15, code, ha="center", va="center",
                fontsize=8.8, color="white", family="monospace")

    # Connect dispatcher -> kernel work -> return
    ax.annotate("", xy=(work[1][0] - 1.3, kern_y_bot + 1.3),
                xytext=(work[0][0] + 1.3, kern_y_bot + 1.3),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="white"))
    ax.annotate("", xy=(work[2][0] - 1.3, kern_y_bot + 1.3),
                xytext=(work[1][0] + 1.3, kern_y_bot + 1.3),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="white"))

    # Trap arrow from dispatcher "down arrow" to dispatcher box (already exists)
    # Cost annotation
    ax.add_patch(FancyBboxPatch(
        (0.5, 0.2), 12.0, 1.0,
        boxstyle="round,pad=0.04", facecolor=C_BG_SOFT,
        edgecolor=C_DARK, lw=1.0,
    ))
    ax.text(0.85, 0.95, "Cost of one syscall (no I/O):",
            fontsize=10.7, fontweight="bold", color=C_DARK)
    ax.text(0.85, 0.65,
            "~100 ns (mode switch + register save) + cache pollution.   "
            "10x slower than a function call.",
            fontsize=10, color=C_DARK, family="monospace")
    ax.text(0.85, 0.32,
            "Hot loops batch syscalls (sendmmsg, io_uring) precisely to amortize this cost. "
            "Spectre / Meltdown mitigations made it worse (KPTI flushes the TLB).",
            fontsize=9.6, color=C_GRAY, style="italic")

    _save(fig, "fig6_syscall_interface")


# ---------------------------------------------------------------------------
# Figure 7 -- Scheduler comparison (FCFS / SJF / RR / CFS)
# ---------------------------------------------------------------------------
def fig7_schedulers() -> None:
    fig, ax = plt.subplots(figsize=(13, 9.0))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 9.0)

    ax.text(6.5, 8.65, "Four Schedulers, Same Workload -- The Same Three Jobs Look Very Different",
            ha="center", va="center", fontsize=16.5, fontweight="bold", color=C_DARK)
    ax.text(6.5, 8.30,
            "jobs: P1 burst=8 (arrives t=0)   |   P2 burst=4 (arrives t=1)   |   P3 burst=2 (arrives t=2)",
            ha="center", va="center", fontsize=10.7, style="italic", color=C_GRAY)

    job_color = {"P1": C_BLUE, "P2": C_PURPLE, "P3": C_GREEN, "idle": C_LIGHT}

    def draw_gantt(y_top: float, title: str, segments, avg_wait: float, note: str,
                   total_t: int = 14):
        # Title strip
        ax.text(0.5, y_top - 0.05, title, fontsize=12, fontweight="bold", color=C_DARK)
        ax.text(12.5, y_top - 0.05, f"avg wait = {avg_wait:.2f}",
                fontsize=10.5, color=C_DARK, ha="right",
                fontweight="bold", family="monospace")

        # Time axis
        chart_x0 = 0.8
        chart_x1 = 11.8
        bar_y = y_top - 0.85
        bar_h = 0.6
        unit = (chart_x1 - chart_x0) / total_t

        # Background grid
        for t in range(total_t + 1):
            x = chart_x0 + t * unit
            ax.plot([x, x], [bar_y, bar_y + bar_h], color=C_LIGHT, lw=0.6)
            if t % 2 == 0:
                ax.text(x, bar_y - 0.18, str(t), ha="center", va="top",
                        fontsize=8.5, color=C_GRAY, family="monospace")

        # Segments
        t = 0
        for seg in segments:
            job, dur = seg
            x = chart_x0 + t * unit
            w = dur * unit
            color = job_color[job]
            ax.add_patch(Rectangle((x, bar_y), w, bar_h,
                                   facecolor=color, edgecolor="white", lw=1.2,
                                   alpha=0.95))
            if job != "idle" and dur >= 1:
                ax.text(x + w / 2, bar_y + bar_h / 2, job,
                        ha="center", va="center", fontsize=10,
                        color="white", fontweight="bold", family="monospace")
            t += dur

        # Note
        ax.text(0.5, y_top - 1.55, note, fontsize=9.5,
                color=C_DARK, style="italic")

    # Layout: 4 panels stacked
    panel_h = 1.85
    panel_top = 7.7

    # ---- 1) FCFS: 0-8 P1, 8-12 P2, 12-14 P3 ----
    # Wait: P1=0, P2=8-1=7, P3=12-2=10. avg = 17/3 ~= 5.67
    draw_gantt(panel_top,
               "1.  FCFS  (First-Come-First-Served)",
               [("P1", 8), ("P2", 4), ("P3", 2)],
               avg_wait=17 / 3,
               note="P1 monopolizes the CPU. The short jobs (P2, P3) wait forever -- "
                    "this is the convoy effect.")

    # ---- 2) SJF (non-preemptive, but P1 already running): 0-8 P1, 8-10 P3, 10-14 P2 ----
    # at t=8 ready: P2 burst 4, P3 burst 2 -> pick P3 first
    # Wait: P1=0, P3=8-2=6, P2=10-1=9 -> avg = 15/3 = 5.00
    draw_gantt(panel_top - panel_h,
               "2.  SJF  (Shortest-Job-First, non-preemptive)",
               [("P1", 8), ("P3", 2), ("P2", 4)],
               avg_wait=15 / 3,
               note="Optimal average wait *if* burst lengths are known. They almost never are -- "
                    "and starvation of long jobs is real.")

    # ---- 3) Round Robin (q=2): tick by tick ----
    # Order: P1(2) P2(2) P3(2) P1(2) P2(2) P1(2) P1(2)  = 14
    # Compute waits roughly:
    # P1 finishes at 14, burst 8, arr 0 -> turnaround 14, wait = 14-8 = 6
    # P2 finishes at 10, burst 4, arr 1 -> turnaround 9, wait = 9-4 = 5
    # P3 finishes at 6,  burst 2, arr 2 -> turnaround 4, wait = 4-2 = 2
    # avg wait = (6+5+2)/3 = 13/3 ~= 4.33
    draw_gantt(panel_top - 2 * panel_h,
               "3.  Round Robin  (quantum = 2)",
               [("P1", 2), ("P2", 2), ("P3", 2),
                ("P1", 2), ("P2", 2), ("P1", 2), ("P1", 2)],
               avg_wait=13 / 3,
               note="No starvation, fair latency. The trade-off: more context switches. "
                    "Quantum too small -> overhead dominates; too large -> looks like FCFS.")

    # ---- 4) CFS-style weighted fair: roughly 1:1:1 vruntime ratio at start ----
    # vruntime accounting: pick task with smallest vruntime each tick.
    # Assume tick = 1 unit. arrivals: P1@0, P2@1, P3@2.
    # We'll just show fine-grained interleave that balances vruntime and finishes
    # short jobs faster:
    # t0:P1, t1:P2, t2:P3, t3:P3, t4:P2, t5:P1, t6:P2, t7:P1, t8:P2, t9:P1, t10..13:P1
    # Adjusted: P1 burst 8, P2 burst 4, P3 burst 2
    # P3 done by t4; P2 done somewhere mid; P1 finishes last at t14.
    # We'll simplify the picture but keep correct totals.
    cfs_segments = [
        ("P1", 1), ("P2", 1), ("P3", 1), ("P1", 1), ("P3", 1),
        ("P2", 1), ("P1", 1), ("P2", 1), ("P1", 1), ("P2", 1),
        ("P1", 1), ("P1", 1), ("P1", 1), ("P1", 1),
    ]
    # Wait calculation (approx for the picture):
    # P3 last unit at t=4, burst 2, arr 2 -> turnaround 3, wait 1
    # P2 last unit at t=9, burst 4, arr 1 -> turnaround 9, wait 5
    # P1 last unit at t=13, burst 8, arr 0 -> turnaround 14, wait 6
    # avg wait = (6+5+1)/3 = 4.0
    draw_gantt(panel_top - 3 * panel_h,
               "4.  Linux CFS  (Completely Fair Scheduler -- vruntime-balanced)",
               cfs_segments,
               avg_wait=12 / 3,
               note="Pick the task with the smallest accumulated vruntime; weights come from nice values. "
                    "Approaches ideal proportional sharing as quantum -> 0.")

    # Bottom legend
    ax.add_patch(FancyBboxPatch(
        (0.5, 0.05), 12.0, 0.55,
        boxstyle="round,pad=0.03", facecolor=C_BG_SOFT,
        edgecolor=C_DARK, lw=1.0,
    ))
    items = [("P1", C_BLUE), ("P2", C_PURPLE), ("P3", C_GREEN)]
    x = 0.85
    ax.text(x, 0.32, "jobs:", fontsize=10, fontweight="bold", color=C_DARK)
    x += 0.7
    for label, color in items:
        ax.add_patch(Rectangle((x, 0.22), 0.25, 0.25, facecolor=color, edgecolor="none"))
        ax.text(x + 0.32, 0.34, label, fontsize=9.5, color=C_DARK,
                va="center", family="monospace", fontweight="bold")
        x += 1.0
    ax.text(x + 0.2, 0.34,
            "smaller average wait = better throughput; flatter response time = better interactivity.",
            fontsize=9.5, color=C_GRAY, va="center", style="italic")

    _save(fig, "fig7_schedulers")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Operating System Fundamentals figures...")
    fig1_kernel_architectures()
    fig2_process_states()
    fig3_virtual_memory()
    fig4_file_system()
    fig5_io_subsystem()
    fig6_syscall_interface()
    fig7_schedulers()
    print("Done.")


if __name__ == "__main__":
    main()
