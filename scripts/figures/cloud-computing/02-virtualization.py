"""
Figure generation script for Cloud Computing Part 02: Virtualization.

Generates 7 production-quality figures used in BOTH the EN and ZH versions
of the article. Each figure teaches a single specific idea cleanly so it
can stand on its own in the surrounding prose.

Figures:
    fig1_hypervisor_types          Type 1 (bare-metal) vs Type 2 (hosted)
                                   hypervisor stack diagram.
    fig2_vm_vs_container           VM vs container resource isolation
                                   stack comparison.
    fig3_startup_and_memory        Startup time + memory overhead bars
                                   (container vs VM).
    fig4_hypervisor_matrix         KVM / Xen / VMware ESXi / Hyper-V
                                   capability radar comparison.
    fig5_live_migration            Live migration (pre-copy) phases
                                   timeline with dirty-page rate.
    fig6_nested_virtualization     Nested virtualization stack (L0/L1/L2)
                                   showing performance amplification.
    fig7_gpu_virtualization        GPU sharing modes: time-slice vs vGPU
                                   vs MIG vs passthrough.

Usage:
    python3 scripts/figures/cloud-computing/02-virtualization.py

Output:
    Writes PNGs into BOTH the EN and ZH article asset folders so the
    markdown image references stay consistent across languages.
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

C_BLUE = "#2563eb"      # primary
C_PURPLE = "#7c3aed"    # secondary
C_GREEN = "#10b981"     # accent / good
C_AMBER = "#f59e0b"     # warning / highlight
C_GRAY = "#94a3b8"
C_DARK = "#0f172a"
C_LIGHT = "#e2e8f0"
C_BG = "#f8fafc"

PALETTE = [C_BLUE, C_PURPLE, C_GREEN, C_AMBER]

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "cloud-computing" / "virtualization"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "cloud-computing" / "virtualization"


def _save(fig: plt.Figure, name: str) -> None:
    """Save the figure to BOTH the EN and ZH asset directories."""
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, x, y, w, h, label, color, fc=None, fontsize=10,
         fontweight="normal", text_color=C_DARK, alpha=1.0,
         radius=0.04):
    """Draw a rounded rectangle with centred text."""
    fc = fc if fc is not None else color
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=1.2, edgecolor=color, facecolor=fc, alpha=alpha,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label,
            ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color)


# ---------------------------------------------------------------------------
# Figure 1: Type 1 vs Type 2 hypervisor architectures
# ---------------------------------------------------------------------------
def fig1_hypervisor_types() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))

    def draw_stack(ax, layers, title):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color=C_DARK, pad=12)
        y = 0.6
        for label, color, fc, h, fs, fw in layers:
            _box(ax, 0.6, y, 8.8, h, label, color,
                 fc=fc, fontsize=fs, fontweight=fw)
            y += h + 0.18

    # Type 1: bare metal
    t1 = [
        ("Hardware (CPU / Memory / NIC / Disk)", C_DARK, C_LIGHT, 1.0, 10, "bold"),
        ("Hypervisor (ESXi / KVM / Xen / Hyper-V)", C_BLUE, "#dbeafe", 1.4, 11, "bold"),
        ("Guest OS 1", C_PURPLE, "#ede9fe", 0.9, 10, "normal"),
        ("App A1     App A2", C_PURPLE, "#f5f3ff", 0.9, 10, "normal"),
        ("Guest OS 2", C_GREEN, "#d1fae5", 0.9, 10, "normal"),
        ("App B1     App B2", C_GREEN, "#ecfdf5", 0.9, 10, "normal"),
    ]
    draw_stack(axes[0], t1, "Type 1 — Bare-Metal Hypervisor")
    axes[0].text(5.0, 9.7,
                 "Direct hardware access · lowest overhead · production clouds",
                 ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    # Type 2: hosted
    t2 = [
        ("Hardware (CPU / Memory / NIC / Disk)", C_DARK, C_LIGHT, 1.0, 10, "bold"),
        ("Host OS (Linux / macOS / Windows)", C_GRAY, "#f1f5f9", 1.0, 11, "bold"),
        ("Hypervisor (VirtualBox / VMware Workstation)", C_AMBER, "#fef3c7", 1.2, 11, "bold"),
        ("Guest OS", C_PURPLE, "#ede9fe", 0.9, 10, "normal"),
        ("App 1     App 2", C_PURPLE, "#f5f3ff", 0.9, 10, "normal"),
    ]
    draw_stack(axes[1], t2, "Type 2 — Hosted Hypervisor")
    axes[1].text(5.0, 9.7,
                 "Runs on a host OS · easy to install · dev / labs",
                 ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    fig.suptitle("Hypervisor Architectures: Where the VMM Sits",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig1_hypervisor_types")


# ---------------------------------------------------------------------------
# Figure 2: VM vs Container resource isolation
# ---------------------------------------------------------------------------
def fig2_vm_vs_container() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))

    def draw(ax, layers, title, sub):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold",
                     color=C_DARK, pad=10)
        ax.text(5, 9.4, sub, ha="center", fontsize=9.5,
                color=C_GRAY, style="italic")
        y = 0.4
        for label, color, fc, h, fs, fw in layers:
            _box(ax, 0.4, y, 9.2, h, label, color,
                 fc=fc, fontsize=fs, fontweight=fw)
            y += h + 0.15

    # Virtual machines: each VM has its own kernel
    vm = [
        ("Hardware", C_DARK, C_LIGHT, 0.85, 10, "bold"),
        ("Hypervisor", C_BLUE, "#dbeafe", 0.95, 11, "bold"),
        ("Guest Kernel  |  Guest Kernel  |  Guest Kernel", C_PURPLE, "#ede9fe", 0.85, 10, "bold"),
        ("Libs / Bins   |  Libs / Bins   |  Libs / Bins ", C_PURPLE, "#f5f3ff", 0.75, 9.5, "normal"),
        ("App A         |  App B         |  App C       ", C_AMBER, "#fef3c7", 0.85, 10, "bold"),
    ]
    draw(axes[0], vm, "Virtual Machines",
         "Strong isolation · GB-scale images · minutes to boot")
    axes[0].text(5, 5.3, "3 kernels", ha="center", fontsize=11,
                 color=C_BLUE, fontweight="bold")

    # Containers: shared host kernel
    ct = [
        ("Hardware", C_DARK, C_LIGHT, 0.85, 10, "bold"),
        ("Host OS Kernel  (shared by all containers)", C_BLUE, "#dbeafe", 1.05, 11, "bold"),
        ("Container Runtime  (containerd · runc)", C_GREEN, "#d1fae5", 0.85, 10, "bold"),
        ("Libs / Bins  |  Libs / Bins  |  Libs / Bins  |  Libs / Bins",
         C_GREEN, "#ecfdf5", 0.75, 9, "normal"),
        ("App A   |   App B   |   App C   |   App D",
         C_AMBER, "#fef3c7", 0.85, 10, "bold"),
    ]
    draw(axes[1], ct, "Containers",
         "Process-level isolation · MB-scale images · seconds to boot")
    axes[1].text(5, 4.4, "1 shared kernel", ha="center", fontsize=11,
                 color=C_GREEN, fontweight="bold")

    fig.suptitle("VM vs Container Isolation Boundary",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig2_vm_vs_container")


# ---------------------------------------------------------------------------
# Figure 3: Startup time and memory overhead
# ---------------------------------------------------------------------------
def fig3_startup_and_memory() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    workloads = ["Container\n(Alpine)", "Container\n(Ubuntu)",
                 "MicroVM\n(Firecracker)", "VM\n(KVM/QEMU)",
                 "VM\n(VMware/ESXi)"]
    # Realistic order-of-magnitude numbers (seconds)
    startup = [0.05, 0.5, 0.125, 25, 45]
    # Idle memory footprint (MB)
    memory = [5, 50, 30, 1024, 1500]
    colors = [C_GREEN, C_GREEN, C_AMBER, C_BLUE, C_BLUE]

    # --- Startup time (log scale) ---
    ax = axes[0]
    bars = ax.bar(workloads, startup, color=colors,
                  edgecolor="white", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Startup time  (seconds, log scale)",
                  fontsize=10.5, color=C_DARK)
    ax.set_title("Cold-Start Latency", fontsize=12.5,
                 fontweight="bold", color=C_DARK, pad=10)
    ax.set_ylim(0.02, 200)
    for b, v in zip(bars, startup):
        if v < 1:
            label = f"{int(v * 1000)} ms"
        else:
            label = f"{v:g} s"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.25, label,
                ha="center", va="bottom", fontsize=9.5,
                color=C_DARK, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.4)

    # --- Memory footprint (log scale) ---
    ax = axes[1]
    bars = ax.bar(workloads, memory, color=colors,
                  edgecolor="white", linewidth=1.2)
    ax.set_yscale("log")
    ax.set_ylabel("Idle memory footprint  (MB, log scale)",
                  fontsize=10.5, color=C_DARK)
    ax.set_title("Memory Overhead at Rest", fontsize=12.5,
                 fontweight="bold", color=C_DARK, pad=10)
    ax.set_ylim(2, 4000)
    for b, v in zip(bars, memory):
        if v >= 1024:
            label = f"{v/1024:.1f} GB"
        else:
            label = f"{v} MB"
        ax.text(b.get_x() + b.get_width() / 2, v * 1.25, label,
                ha="center", va="bottom", fontsize=9.5,
                color=C_DARK, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.4)

    fig.suptitle("Startup Latency and Memory Cost: Container · MicroVM · VM",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig3_startup_and_memory")


# ---------------------------------------------------------------------------
# Figure 4: Hypervisor capability comparison (radar)
# ---------------------------------------------------------------------------
def fig4_hypervisor_matrix() -> None:
    categories = ["Performance", "Ecosystem", "Mgmt UX",
                  "Cost\nefficiency", "Linux\nguest", "Windows\nguest"]
    # Normalised 1..5 subjective ratings based on common benchmarks/reviews
    data = {
        "KVM":         [5, 5, 3, 5, 5, 4],
        "Xen":         [4, 3, 3, 5, 5, 3],
        "VMware ESXi": [5, 5, 5, 2, 5, 5],
        "Hyper-V":     [4, 4, 4, 4, 4, 5],
    }
    colors = {
        "KVM": C_BLUE,
        "Xen": C_PURPLE,
        "VMware ESXi": C_GREEN,
        "Hyper-V": C_AMBER,
    }

    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.4, 7.8),
                           subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10.5, color=C_DARK)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8.5,
                       color=C_GRAY)
    ax.set_ylim(0, 5)
    ax.grid(color=C_GRAY, alpha=0.35)

    for name, vals in data.items():
        v = vals + vals[:1]
        ax.plot(angles, v, color=colors[name], linewidth=2.2,
                label=name, marker="o", markersize=5)
        ax.fill(angles, v, color=colors[name], alpha=0.12)

    ax.set_title("Hypervisor Capability Comparison  (1 = weak, 5 = strong)",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=22)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.13),
              ncol=4, frameon=False, fontsize=10)
    fig.tight_layout()
    _save(fig, "fig4_hypervisor_matrix")


# ---------------------------------------------------------------------------
# Figure 5: Live migration (pre-copy) timeline
# ---------------------------------------------------------------------------
def fig5_live_migration() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Pre-Copy Live Migration: How a VM Moves Without Going Down",
                 fontsize=13.5, fontweight="bold", color=C_DARK, pad=10)

    # Source/dest host bars
    _box(ax, 0.2, 4.6, 5.6, 0.7, "Source host  (origin)", C_BLUE,
         fc="#dbeafe", fontsize=11, fontweight="bold")
    _box(ax, 6.2, 4.6, 5.6, 0.7, "Destination host  (target)", C_GREEN,
         fc="#d1fae5", fontsize=11, fontweight="bold")

    # Phase lanes along a single timeline
    timeline_y = 2.6
    ax.annotate("", xy=(11.7, timeline_y), xytext=(0.3, timeline_y),
                arrowprops=dict(arrowstyle="->", lw=1.6, color=C_DARK))
    ax.text(0.3, timeline_y - 0.45, "t=0",
            fontsize=9, color=C_GRAY)
    ax.text(11.5, timeline_y - 0.45, "switchover",
            fontsize=9, color=C_GRAY, ha="right")

    phases = [
        (0.5, 4.0, "1. Initial copy",
         "Send full memory image\nover the network",
         C_BLUE, "#dbeafe"),
        (4.7, 5.0, "2. Iterative dirty-page rounds",
         "Re-send only pages dirtied\nsince last round",
         C_PURPLE, "#ede9fe"),
        (9.9, 1.6, "3. Stop & switch",
         "Pause < 100 ms ·\nflush + resume on dest",
         C_AMBER, "#fef3c7"),
    ]
    for x, w, title, body, color, fc in phases:
        _box(ax, x, 3.1, w, 0.95, title, color,
             fc=fc, fontsize=10.5, fontweight="bold")
        ax.text(x + w / 2, 1.85, body, ha="center", va="center",
                fontsize=9.2, color=C_DARK)

    # Convergence note
    ax.text(6, 0.6,
            "Pre-copy converges when the dirty-page rate drops below the network bandwidth.\n"
            "Post-copy is the alternative: switch first, then page-fault memory across the network.",
            ha="center", fontsize=9.5, color=C_GRAY, style="italic")

    # Arrow from source -> dest above timeline
    arrow = FancyArrowPatch((3.0, 4.55), (9.0, 4.55),
                            arrowstyle="->", lw=2, color=C_PURPLE,
                            connectionstyle="arc3,rad=-0.25")
    ax.add_patch(arrow)
    ax.text(6.0, 5.55, "Memory pages stream while VM keeps running",
            ha="center", fontsize=10, color=C_PURPLE,
            fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig5_live_migration")


# ---------------------------------------------------------------------------
# Figure 6: Nested virtualization (L0 / L1 / L2)
# ---------------------------------------------------------------------------
def fig6_nested_virtualization() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2),
                             gridspec_kw={"width_ratios": [1.05, 1]})

    # --- Stack diagram ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Nested Virtualization Stack",
                 fontsize=12.5, fontweight="bold", color=C_DARK, pad=8)

    layers = [
        ("Hardware  (Intel VT-x / AMD-V)", C_DARK, C_LIGHT, 1.0, "bold", 10),
        ("L0 Hypervisor  (e.g. KVM on bare metal)", C_BLUE, "#dbeafe", 1.0, "bold", 10.5),
        ("L1 Guest OS + Hypervisor  (e.g. KVM inside a VM)",
         C_PURPLE, "#ede9fe", 1.1, "bold", 10.5),
        ("L2 Guest VM  (workload)", C_GREEN, "#d1fae5", 1.0, "bold", 10.5),
        ("Application", C_AMBER, "#fef3c7", 0.9, "bold", 10.5),
    ]
    y = 1.4
    for label, color, fc, h, fw, fs in layers:
        _box(ax, 0.6, y, 8.8, h, label, color,
             fc=fc, fontsize=fs, fontweight=fw)
        y += h + 0.25

    ax.text(5, 0.55,
            "Each level adds a VM-exit hop · enable with kvm-intel nested=1",
            ha="center", fontsize=9, color=C_GRAY, style="italic")

    # --- Performance amplification bars ---
    ax = axes[1]
    levels = ["Bare metal", "L1 guest", "L2 guest"]
    perf = [100, 95, 70]   # rough relative throughput (CPU bound)
    overhead = [0, 5, 30]
    x = np.arange(len(levels))

    ax.bar(x, perf, color=[C_DARK, C_BLUE, C_PURPLE],
           edgecolor="white", linewidth=1.2)
    for i, (p, o) in enumerate(zip(perf, overhead)):
        ax.text(i, p + 1.5, f"{p}%", ha="center", fontsize=11,
                color=C_DARK, fontweight="bold")
        if o > 0:
            ax.text(i, 50, f"-{o}%\noverhead",
                    ha="center", va="center", fontsize=10,
                    color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(levels, fontsize=10.5)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Relative CPU throughput  (%)",
                  fontsize=10.5, color=C_DARK)
    ax.set_title("Cost of Each Nesting Level",
                 fontsize=12.5, fontweight="bold", color=C_DARK, pad=8)
    ax.grid(axis="y", alpha=0.35)

    fig.suptitle("Nested Virtualization: VM Inside a VM",
                 fontsize=14, fontweight="bold", color=C_DARK, y=1.02)
    fig.tight_layout()
    _save(fig, "fig6_nested_virtualization")


# ---------------------------------------------------------------------------
# Figure 7: GPU virtualization modes
# ---------------------------------------------------------------------------
def fig7_gpu_virtualization() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("GPU Virtualization: Four Ways to Share a GPU Across VMs",
                 fontsize=13.5, fontweight="bold", color=C_DARK, pad=10)

    modes = [
        ("Time-slicing",
         "One GPU, many VMs share by\ntime quantum (round-robin)",
         "Cheap · context-switch tax",
         C_GRAY, "#f1f5f9"),
        ("vGPU\n(NVIDIA GRID)",
         "Mediated passthrough · each VM\nsees a virtual GPU slice",
         "Hard QoS · NVIDIA license required",
         C_BLUE, "#dbeafe"),
        ("MIG\n(A100 / H100)",
         "Hardware-partitioned GPU into\nup to 7 isolated instances",
         "True isolation · fixed shapes",
         C_PURPLE, "#ede9fe"),
        ("Passthrough\n(SR-IOV / VFIO)",
         "Whole physical GPU dedicated\nto one VM",
         "Bare-metal speed · no sharing",
         C_GREEN, "#d1fae5"),
    ]
    box_w, box_h, gap = 2.7, 4.0, 0.3
    x0 = 0.4
    for i, (name, what, trade, color, fc) in enumerate(modes):
        x = x0 + i * (box_w + gap)
        # Title bar
        _box(ax, x, 6.4, box_w, 0.95, name, color,
             fc=color, fontsize=11.5, fontweight="bold",
             text_color="white")
        # Body
        _box(ax, x, 2.3, box_w, 4.0, "", color,
             fc=fc, fontsize=10)
        ax.text(x + box_w / 2, 4.95, what,
                ha="center", va="center", fontsize=10, color=C_DARK)
        ax.text(x + box_w / 2, 2.95, trade,
                ha="center", va="center", fontsize=9.2,
                color=color, fontweight="bold", style="italic")

    # Spectrum arrow
    ax.annotate("", xy=(11.6, 1.2), xytext=(0.4, 1.2),
                arrowprops=dict(arrowstyle="->", lw=2, color=C_DARK))
    ax.text(0.4, 0.6, "Higher density / lower isolation",
            ha="left", fontsize=9.5, color=C_GRAY)
    ax.text(11.6, 0.6, "Higher performance / lower density",
            ha="right", fontsize=9.5, color=C_GRAY)

    fig.tight_layout()
    _save(fig, "fig7_gpu_virtualization")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating figures for Cloud Computing Part 02 (Virtualization)…")
    fig1_hypervisor_types()
    fig2_vm_vs_container()
    fig3_startup_and_memory()
    fig4_hypervisor_matrix()
    fig5_live_migration()
    fig6_nested_virtualization()
    fig7_gpu_virtualization()
    print("Done.")


if __name__ == "__main__":
    main()
