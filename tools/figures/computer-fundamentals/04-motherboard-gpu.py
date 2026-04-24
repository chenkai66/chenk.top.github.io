"""
Figure generator for Computer Fundamentals Part 04:
Motherboard, PCIe & GPU.

Produces 7 production-quality figures:
  fig1_motherboard_layout.png   — annotated motherboard floor plan
  fig2_pcie_generations.png     — PCIe Gen 2/3/4/5 bandwidth comparison
  fig3_gpu_simt.png             — SIMT architecture: warps, SMs, CUDA cores
  fig4_cpu_vs_gpu.png           — latency-oriented CPU vs throughput GPU
  fig5_memory_bandwidth.png     — DDR vs GDDR vs HBM bandwidth & topology
  fig6_display_interfaces.png   — DP / HDMI / USB-C bandwidth & capabilities
  fig7_chipset_evolution.png    — northbridge/southbridge → modern PCH

Outputs PNGs (dpi=150) into the EN and ZH asset directories:
  source/_posts/en/computer-fundamentals/04-motherboard-gpu/
  source/_posts/zh/computer-fundamentals/04-motherboard-gpu/

Style: matplotlib seaborn-v0_8-whitegrid; brand palette
  blue   #2563eb  (primary / CPU / DP)
  purple #7c3aed  (PCIe / GPU / NVMe)
  green  #10b981  (memory / HBM / success)
  amber  #f59e0b  (chipset / legacy / warning)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
AMBER = "#f59e0b"
DARK = "#0f172a"
GRAY = "#64748b"
LIGHT = "#f1f5f9"

DPI = 150

REPO = Path(__file__).resolve().parents[3]
OUT_DIRS = [
    REPO / "source/_posts/en/computer-fundamentals/04-motherboard-gpu",
    REPO / "source/_posts/zh/computer-fundamentals/04-motherboard-gpu",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "axes.labelweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#cbd5e1",
    "axes.linewidth": 1.0,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
})


def _save(fig: plt.Figure, name: str) -> None:
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  wrote {name} -> {len(OUT_DIRS)} dirs")


# ---------------------------------------------------------------------------
# Figure 1: Motherboard layout (CPU socket, RAM, PCIe, chipset, VRM, M.2)
# ---------------------------------------------------------------------------

def fig1_motherboard_layout() -> None:
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.set_aspect("equal"); ax.axis("off")

    # PCB outline (ATX 305 x 244)
    ax.add_patch(FancyBboxPatch((0.4, 0.4), 13.2, 9.2,
                                boxstyle="round,pad=0.04,rounding_size=0.25",
                                facecolor="#0b3d2e", edgecolor=DARK, linewidth=2,
                                alpha=0.92))
    # silkscreen text
    ax.text(13.4, 0.65, "ATX motherboard (305 × 244 mm)",
            ha="right", color="#a7f3d0", fontsize=8.5, style="italic")

    # ----- I/O shield (top-left) -----
    ax.add_patch(Rectangle((0.6, 7.8), 2.6, 1.6,
                           facecolor="#1e293b", edgecolor=DARK, linewidth=1))
    ax.text(1.9, 8.6, "Rear I/O\nUSB · LAN · audio · DP",
            ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")

    # ----- CPU socket -----
    ax.add_patch(FancyBboxPatch((4.1, 7.0), 2.6, 2.4,
                                boxstyle="round,pad=0.03,rounding_size=0.1",
                                facecolor=BLUE, edgecolor=DARK, linewidth=1.5))
    ax.text(5.4, 8.5, "CPU socket", ha="center", color="white",
            fontsize=11, fontweight="bold")
    ax.text(5.4, 8.05, "LGA 1700 / AM5", ha="center", color="white", fontsize=9)
    ax.text(5.4, 7.3, "16–24 PCIe lanes\n+ DDR5 controller",
            ha="center", color="white", fontsize=8.5, style="italic")

    # ----- VRM (left of socket) -----
    for i, y in enumerate(np.linspace(7.1, 9.2, 8)):
        ax.add_patch(Rectangle((3.3, y), 0.55, 0.22,
                               facecolor=AMBER, edgecolor=DARK, linewidth=0.6))
    ax.text(3.55, 6.85, "VRM phases\n(MOSFET + choke)",
            ha="center", color=AMBER, fontsize=8.5, fontweight="bold")
    ax.add_patch(FancyBboxPatch((3.25, 9.45), 0.65, 0.25,
                                boxstyle="round,pad=0.01,rounding_size=0.05",
                                facecolor="#1e293b", edgecolor=DARK))
    ax.text(3.575, 9.575, "EPS 8-pin", ha="center", va="center",
            color="white", fontsize=7.5, fontweight="bold")

    # ----- RAM slots (right of socket) -----
    for i, x in enumerate([7.1, 7.5, 7.9, 8.3]):
        col = GREEN if i in (1, 3) else "#065f46"
        ax.add_patch(Rectangle((x, 6.6), 0.32, 2.8,
                               facecolor=col, edgecolor=DARK, linewidth=0.8))
    ax.text(7.7, 6.25, "DIMM × 4  (DDR5)",
            ha="center", color=GREEN, fontsize=9, fontweight="bold")
    ax.text(7.7, 5.95, "fill A2 + B2 first → dual-channel",
            ha="center", color="#a7f3d0", fontsize=8, style="italic")

    # ----- 24-pin ATX power -----
    ax.add_patch(FancyBboxPatch((9.3, 7.3), 0.6, 2.0,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor="#1e293b", edgecolor=DARK))
    ax.text(9.6, 9.5, "ATX 24-pin", ha="center", color="white",
            fontsize=8.5, fontweight="bold")

    # ----- Chipset (PCH) -----
    ax.add_patch(FancyBboxPatch((9.6, 4.0), 1.6, 1.4,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                facecolor=AMBER, edgecolor=DARK, linewidth=1.5))
    ax.text(10.4, 4.95, "Chipset", ha="center", color="white",
            fontsize=11, fontweight="bold")
    ax.text(10.4, 4.45, "Z790 / B650 (PCH)",
            ha="center", color="white", fontsize=8.5)
    ax.text(10.4, 4.15, "DMI 4.0 ×8 ↔ CPU",
            ha="center", color="white", fontsize=8, style="italic")

    # DMI link CPU -> chipset
    ax.annotate("", xy=(10.4, 5.45), xytext=(5.4, 7.0),
                arrowprops=dict(arrowstyle="<->", color="#fde68a",
                                lw=1.6, linestyle=(0, (4, 2))))
    ax.text(8.0, 6.45, "DMI link\n~16 GB/s",
            ha="center", color="#fde68a", fontsize=8, fontweight="bold",
            rotation=-15)

    # ----- PCIe slots (left side, stacked) -----
    pcie_specs = [
        (3.6, "PCIe 5.0 ×16  (CPU-direct, GPU)",   PURPLE, 6.4),
        (2.7, "PCIe 4.0 ×4   (chipset)",           AMBER,  3.6),
        (1.8, "PCIe 3.0 ×1   (chipset)",           AMBER,  1.8),
        (0.9, "PCIe 4.0 ×4   (chipset, 2nd ×16 phys.)", AMBER, 6.4),
    ]
    for y, label, color, w in pcie_specs:
        ax.add_patch(Rectangle((0.7, y), w, 0.45,
                               facecolor=color, edgecolor=DARK, linewidth=0.8))
        ax.text(0.7 + w + 0.15, y + 0.22, label,
                va="center", color=color, fontsize=8.5, fontweight="bold")

    # ----- M.2 slots -----
    ax.add_patch(Rectangle((4.2, 4.6), 3.0, 0.4,
                           facecolor=PURPLE, edgecolor=DARK, linewidth=0.8))
    ax.text(5.7, 4.45, "M.2_1   PCIe 5.0 ×4  (CPU-direct, NVMe)",
            ha="center", color=PURPLE, fontsize=8.3, fontweight="bold")
    ax.add_patch(Rectangle((4.2, 3.6), 3.0, 0.4,
                           facecolor=AMBER, edgecolor=DARK, linewidth=0.8))
    ax.text(5.7, 3.45, "M.2_2   PCIe 4.0 ×4  (chipset)",
            ha="center", color=AMBER, fontsize=8.3, fontweight="bold")

    # ----- SATA + USB headers (bottom-right) -----
    for i in range(4):
        ax.add_patch(Rectangle((11.7 + (i % 2) * 0.55, 2.0 + (i // 2) * 0.55),
                               0.45, 0.45,
                               facecolor="#dc2626", edgecolor=DARK, linewidth=0.6))
    ax.text(12.0, 1.55, "SATA × 4", color="#fecaca", fontsize=8.5, fontweight="bold")

    ax.add_patch(Rectangle((11.5, 3.5), 1.4, 0.4,
                           facecolor="#1e293b", edgecolor=DARK, linewidth=0.6))
    ax.text(12.2, 3.7, "USB 3.2 hdr", ha="center", va="center",
            color="white", fontsize=7.5, fontweight="bold")

    # Front-panel header
    ax.add_patch(Rectangle((11.7, 0.7), 1.2, 0.5,
                           facecolor="#475569", edgecolor=DARK, linewidth=0.6))
    ax.text(12.3, 0.95, "F_PANEL", ha="center", va="center",
            color="white", fontsize=7.5, fontweight="bold")

    # Title
    fig.suptitle("Motherboard Floor Plan — Where Every Bus Lives",
                 fontsize=15, fontweight="bold", color=DARK, y=0.97)

    # Legend strip
    handles = [
        mpatches.Patch(color=BLUE,   label="CPU socket / CPU-direct lanes"),
        mpatches.Patch(color=GREEN,  label="DDR5 DIMM slots"),
        mpatches.Patch(color=PURPLE, label="PCIe 5.0 (CPU-direct)"),
        mpatches.Patch(color=AMBER,  label="Chipset (PCH) — shared bandwidth"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    _save(fig, "fig1_motherboard_layout.png")


# ---------------------------------------------------------------------------
# Figure 2: PCIe generations bandwidth comparison
# ---------------------------------------------------------------------------

def fig2_pcie_generations() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6),
                                     gridspec_kw={"width_ratios": [1.2, 1]})

    gens = ["PCIe 2.0\n2007", "PCIe 3.0\n2010", "PCIe 4.0\n2017", "PCIe 5.0\n2022"]
    per_lane = [0.5, 1.0, 2.0, 4.0]      # GB/s per lane (one direction)
    x16 = [v * 16 for v in per_lane]
    x4 = [v * 4 for v in per_lane]
    x1 = [v * 1 for v in per_lane]
    colors = [GRAY, BLUE, PURPLE, GREEN]

    # ---- Left: grouped bars by lane width ----
    x = np.arange(len(gens))
    w = 0.25
    ax_l.bar(x - w, x1,  w, label="×1 (sound, NIC)",       color=AMBER,  edgecolor=DARK)
    ax_l.bar(x,     x4,  w, label="×4 (NVMe, 10 GbE)",     color=BLUE,   edgecolor=DARK)
    ax_l.bar(x + w, x16, w, label="×16 (GPU)",             color=PURPLE, edgecolor=DARK)

    for i, (a, b, c) in enumerate(zip(x1, x4, x16)):
        ax_l.text(i - w, a + 1.5, f"{a:g}",  ha="center", fontsize=8.5, fontweight="bold")
        ax_l.text(i,     b + 1.5, f"{b:g}",  ha="center", fontsize=8.5, fontweight="bold")
        ax_l.text(i + w, c + 1.5, f"{c:g}",  ha="center", fontsize=8.5, fontweight="bold")

    ax_l.set_xticks(x); ax_l.set_xticklabels(gens, fontsize=9.5)
    ax_l.set_ylabel("Bandwidth per direction (GB/s)")
    ax_l.set_title("Each generation doubles per-lane bandwidth", color=DARK)
    ax_l.set_ylim(0, 80)
    ax_l.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_l.grid(True, axis="y", alpha=0.4)

    # ---- Right: real-world device requirements (donut + table) ----
    ax_r.set_xlim(0, 10); ax_r.set_ylim(0, 10); ax_r.axis("off")
    ax_r.set_title("How much bandwidth do real devices need?",
                   color=DARK, fontsize=13, pad=8)

    rows = [
        ("RTX 4090",          "×16",  "≈22 GB/s peak",  PURPLE),
        ("NVMe Gen5 SSD",     "×4",   "12 GB/s",        GREEN),
        ("NVMe Gen4 SSD",     "×4",   "7 GB/s",         BLUE),
        ("10 GbE NIC",        "×4",   "1.25 GB/s",      BLUE),
        ("USB 4 / TB4 card",  "×4",   "5 GB/s",         AMBER),
        ("Sound / capture",   "×1",   "<0.5 GB/s",      AMBER),
    ]
    ax_r.text(0.4, 9.0, "Device",        fontsize=10, fontweight="bold", color=DARK)
    ax_r.text(4.0, 9.0, "Lanes",         fontsize=10, fontweight="bold", color=DARK)
    ax_r.text(5.8, 9.0, "Sustained need", fontsize=10, fontweight="bold", color=DARK)
    ax_r.plot([0.3, 9.7], [8.7, 8.7], color=GRAY, lw=0.8)

    for i, (dev, lanes, need, color) in enumerate(rows):
        y = 8.0 - i * 1.05
        ax_r.add_patch(Rectangle((0.25, y - 0.2), 0.25, 0.55,
                                 facecolor=color, edgecolor=DARK))
        ax_r.text(0.7, y + 0.06, dev,   fontsize=10, color=DARK, fontweight="bold")
        ax_r.text(4.0, y + 0.06, lanes, fontsize=10, color=color, fontweight="bold")
        ax_r.text(5.8, y + 0.06, need,  fontsize=10, color=DARK)

    ax_r.text(5.0, 1.0,
              "Rule of thumb: a PCIe 4.0 ×16 slot already exceeds\n"
              "the practical needs of every consumer GPU through 2026.\n"
              "Gen 5 first matters for NVMe SSDs, not GPUs.",
              ha="center", fontsize=9.5, color=DARK, style="italic",
              bbox=dict(boxstyle="round,pad=0.4", facecolor=LIGHT,
                        edgecolor=GRAY, linewidth=0.8))

    fig.suptitle("PCIe Generations & Lane Widths — Bandwidth in Context",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig2_pcie_generations.png")


# ---------------------------------------------------------------------------
# Figure 3: GPU SIMT architecture — warp / SM / CUDA cores
# ---------------------------------------------------------------------------

def fig3_gpu_simt() -> None:
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10); ax.axis("off")

    # ----- GPU chip outline -----
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 13.4, 9.4,
                                boxstyle="round,pad=0.04,rounding_size=0.2",
                                facecolor="#1e1b4b", edgecolor=DARK, linewidth=2,
                                alpha=0.95))
    ax.text(13.55, 0.55, "GPU die (e.g. AD104 / Navi 31)",
            ha="right", color="#c7d2fe", fontsize=9, style="italic")

    # ----- L2 cache strip across top -----
    ax.add_patch(FancyBboxPatch((0.6, 8.6), 12.8, 0.7,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=GREEN, edgecolor=DARK, linewidth=1))
    ax.text(7.0, 8.95, "Shared L2 cache  (32–96 MB)",
            ha="center", color="white", fontsize=11, fontweight="bold")

    # ----- HBM / GDDR controllers strip across bottom -----
    ax.add_patch(FancyBboxPatch((0.6, 0.7), 12.8, 0.6,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=BLUE, edgecolor=DARK, linewidth=1))
    ax.text(7.0, 1.0, "Memory controllers  →  GDDR6X / HBM3  (912 GB/s … 3 TB/s)",
            ha="center", color="white", fontsize=10.5, fontweight="bold")

    # ----- Streaming Multiprocessors (SMs) -----
    sm_w, sm_h = 2.9, 3.2
    starts = [(0.7, 4.7), (3.8, 4.7), (6.9, 4.7), (10.0, 4.7),
              (0.7, 1.45), (3.8, 1.45), (6.9, 1.45), (10.0, 1.45)]
    # We'll draw 4 detailed SMs on top row, 4 placeholders bottom; but bottom
    # collides with memory strip (1.3) so reduce: only top row of 4 + arrows.
    starts = [(0.7, 4.7), (3.8, 4.7), (6.9, 4.7), (10.0, 4.7)]

    for idx, (sx, sy) in enumerate(starts):
        # SM box
        ax.add_patch(FancyBboxPatch((sx, sy), sm_w, sm_h,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor="#312e81", edgecolor=PURPLE,
                                    linewidth=1.4))
        ax.text(sx + sm_w / 2, sy + sm_h - 0.32,
                f"SM #{idx + 1}",
                ha="center", color="white", fontsize=10.5, fontweight="bold")
        # warp scheduler bar
        ax.add_patch(Rectangle((sx + 0.15, sy + sm_h - 0.85), sm_w - 0.3, 0.32,
                               facecolor=AMBER, edgecolor=DARK, linewidth=0.6))
        ax.text(sx + sm_w / 2, sy + sm_h - 0.69,
                "warp scheduler · 4 warps × 32 threads",
                ha="center", color="white", fontsize=7.5, fontweight="bold")
        # CUDA cores grid (8x8 = 64 lanes / SM, per Ada partition)
        cw, ch = 0.27, 0.22
        for r in range(6):
            for c in range(8):
                cx = sx + 0.25 + c * (cw + 0.03)
                cy = sy + 0.55 + r * (ch + 0.05)
                ax.add_patch(Rectangle((cx, cy), cw, ch,
                                       facecolor=PURPLE, alpha=0.55 + 0.05 * (r % 3),
                                       edgecolor=DARK, linewidth=0.3))
        # Tensor + RT block at bottom of SM
        ax.add_patch(Rectangle((sx + 0.2, sy + 0.2), 1.1, 0.3,
                               facecolor=GREEN, edgecolor=DARK, linewidth=0.6))
        ax.text(sx + 0.75, sy + 0.35, "Tensor",
                ha="center", va="center", color="white",
                fontsize=7.5, fontweight="bold")
        ax.add_patch(Rectangle((sx + 1.5, sy + 0.2), 1.1, 0.3,
                               facecolor=AMBER, edgecolor=DARK, linewidth=0.6))
        ax.text(sx + 2.05, sy + 0.35, "RT core",
                ha="center", va="center", color="white",
                fontsize=7.5, fontweight="bold")

    ax.text(7.0, 4.35, "Streaming Multiprocessors (SM) — 46 to 144 per chip",
            ha="center", color="#c7d2fe", fontsize=10, fontweight="bold")

    # ----- annotation box explaining SIMT -----
    ax.add_patch(FancyBboxPatch((0.6, 1.7), 12.8, 2.4,
                                boxstyle="round,pad=0.04,rounding_size=0.1",
                                facecolor="#0f172a", edgecolor="#475569",
                                linewidth=0.8, alpha=0.85))
    ax.text(0.95, 3.85, "SIMT execution model",
            color="white", fontsize=11, fontweight="bold")
    bullets = [
        "1 warp = 32 threads execute the same instruction on 32 data lanes (lock-step).",
        "Each SM holds many warps in flight; the scheduler hides memory latency",
        "    by switching warps every cycle — the GPU's super-power.",
        "Tensor cores accelerate FP16/BF16/INT8 matrix-multiply for AI workloads.",
        "RT cores accelerate ray-triangle intersection tests for real-time ray tracing.",
    ]
    for i, line in enumerate(bullets):
        ax.text(0.95, 3.45 - i * 0.32, "• " + line,
                color="#e2e8f0", fontsize=9.5)

    fig.suptitle("GPU SIMT Architecture — Warps, SMs and Specialised Cores",
                 fontsize=15, fontweight="bold", color=DARK, y=0.99)
    fig.tight_layout()
    _save(fig, "fig3_gpu_simt.png")


# ---------------------------------------------------------------------------
# Figure 4: CPU vs GPU compute model — latency vs throughput
# ---------------------------------------------------------------------------

def fig4_cpu_vs_gpu() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6.4))

    # ----- Left: schematic core layouts -----
    ax_l.set_xlim(0, 10); ax_l.set_ylim(0, 10); ax_l.axis("off")
    ax_l.set_title("Same silicon area — different priorities",
                   color=DARK, fontsize=13, pad=10)

    # CPU half
    ax_l.add_patch(FancyBboxPatch((0.3, 0.3), 4.6, 9.4,
                                  boxstyle="round,pad=0.03,rounding_size=0.12",
                                  facecolor="#dbeafe", edgecolor=BLUE, linewidth=1.5))
    ax_l.text(2.6, 9.3, "CPU — latency-optimised", color=BLUE,
              ha="center", fontsize=11.5, fontweight="bold")
    # 4 big cores + huge cache
    for i, (cx, cy) in enumerate([(1.1, 6.7), (2.7, 6.7), (1.1, 5.0), (2.7, 5.0)]):
        ax_l.add_patch(FancyBboxPatch((cx, cy), 1.4, 1.4,
                                      boxstyle="round,pad=0.02,rounding_size=0.06",
                                      facecolor=BLUE, edgecolor=DARK, linewidth=0.8))
        ax_l.text(cx + 0.7, cy + 0.7, f"Core\n{i+1}", ha="center", va="center",
                  color="white", fontsize=9.5, fontweight="bold")
    # huge L3 cache
    ax_l.add_patch(Rectangle((0.7, 1.2), 3.8, 3.0,
                             facecolor="#bfdbfe", edgecolor=BLUE, linewidth=1.0))
    ax_l.text(2.6, 2.7, "Large L1/L2/L3 caches\n+ branch predictor\n+ out-of-order engine",
              ha="center", color=BLUE, fontsize=9.5, fontweight="bold")
    ax_l.text(2.6, 0.5, "≈ 8–24 fat cores", ha="center",
              color=BLUE, fontsize=9, style="italic")

    # GPU half
    ax_l.add_patch(FancyBboxPatch((5.1, 0.3), 4.6, 9.4,
                                  boxstyle="round,pad=0.03,rounding_size=0.12",
                                  facecolor="#ede9fe", edgecolor=PURPLE, linewidth=1.5))
    ax_l.text(7.4, 9.3, "GPU — throughput-optimised", color=PURPLE,
              ha="center", fontsize=11.5, fontweight="bold")
    # tiny cores grid
    grid_x0, grid_y0 = 5.3, 0.9
    cw, ch = 0.30, 0.30
    for r in range(20):
        for c in range(14):
            cx = grid_x0 + c * (cw + 0.02)
            cy = grid_y0 + r * (ch + 0.04)
            ax_l.add_patch(Rectangle((cx, cy), cw, ch,
                                     facecolor=PURPLE,
                                     alpha=0.45 + 0.04 * ((r + c) % 4),
                                     edgecolor=PURPLE, linewidth=0.2))
    ax_l.text(7.4, 8.7, "thousands of tiny ALUs\nsmall cache, no OoO",
              ha="center", color=PURPLE, fontsize=9.5, fontweight="bold")
    ax_l.text(7.4, 0.5, "≈ 5 000–18 000 lanes", ha="center",
              color=PURPLE, fontsize=9, style="italic")

    # ----- Right: latency vs throughput axis chart -----
    ax_r.set_title("Latency hidden by parallelism — work scaling",
                   color=DARK, fontsize=13, pad=10)
    workload = np.array([1, 4, 16, 64, 256, 1024, 4096, 16384, 65536])
    cpu_time = workload / 8.0 + 0.05            # 8 cores, low overhead
    gpu_time = workload / 5000.0 + 1.5          # 5 000 lanes, ~1.5 ms launch
    ax_r.plot(workload, cpu_time, "-o", color=BLUE, lw=2.2, ms=7,
              label="CPU (8 cores · low launch cost)")
    ax_r.plot(workload, gpu_time, "-s", color=PURPLE, lw=2.2, ms=7,
              label="GPU (5 000 lanes · ~1.5 ms launch)")
    ax_r.set_xscale("log"); ax_r.set_yscale("log")
    ax_r.set_xlabel("Independent work items (log)")
    ax_r.set_ylabel("Wall-clock time (ms, log)")
    ax_r.set_xlim(0.7, 1e5)
    ax_r.set_ylim(0.05, 5000)
    ax_r.legend(loc="upper left", fontsize=10, frameon=True)
    ax_r.grid(True, which="both", alpha=0.4)

    # crossover annotation
    cross_x = 250
    ax_r.axvline(cross_x, color=GRAY, ls="--", lw=1)
    ax_r.text(cross_x * 1.15, 0.12,
              "crossover\nGPU faster →",
              color=GRAY, fontsize=9, fontweight="bold")
    ax_r.text(2, 200, "← CPU wins on small,\n   serial / branchy work",
              color=BLUE, fontsize=9.5, fontweight="bold")
    ax_r.text(8000, 0.35, "GPU wins on huge,\nuniform work →",
              color=PURPLE, fontsize=9.5, fontweight="bold", ha="right")

    fig.suptitle("CPU vs GPU — Latency-First vs Throughput-First Compute",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig4_cpu_vs_gpu.png")


# ---------------------------------------------------------------------------
# Figure 5: Memory bandwidth — DDR vs GDDR vs HBM
# ---------------------------------------------------------------------------

def fig5_memory_bandwidth() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6.4),
                                     gridspec_kw={"width_ratios": [1, 1.1]})

    # ----- Left: bandwidth bars -----
    techs = ["DDR4-3200\ndual ch.", "DDR5-6000\ndual ch.",
             "GDDR6\n256-bit", "GDDR6X\n384-bit",
             "HBM2e\n1024-bit", "HBM3\n1024-bit"]
    bw    = [51,                 96,
             448,                 1008,
             460,                  3200]
    colors = [GRAY, BLUE, AMBER, PURPLE, "#0d9488", GREEN]

    bars = ax_l.barh(techs, bw, color=colors, edgecolor=DARK, linewidth=1.0)
    ax_l.set_xscale("log")
    ax_l.set_xlim(20, 6000)
    ax_l.set_xlabel("Peak bandwidth (GB/s, log)")
    ax_l.set_title("DDR vs GDDR vs HBM — bandwidth at a glance",
                   color=DARK, fontsize=13, pad=8)
    for b, v in zip(bars, bw):
        ax_l.text(v * 1.06, b.get_y() + b.get_height() / 2,
                  f"{v:,} GB/s", va="center", fontsize=9.5,
                  fontweight="bold", color=DARK)
    ax_l.grid(True, axis="x", which="both", alpha=0.4)
    ax_l.invert_yaxis()

    # ----- Right: physical topology comparison -----
    ax_r.set_xlim(0, 10); ax_r.set_ylim(0, 10); ax_r.axis("off")
    ax_r.set_title("Why HBM is so much faster — package topology",
                   color=DARK, fontsize=13, pad=8)

    # DDR5 (top): CPU + DIMMs on PCB, ~64-bit channels
    ax_r.add_patch(FancyBboxPatch((0.4, 7.3), 9.2, 2.4,
                                  boxstyle="round,pad=0.03,rounding_size=0.08",
                                  facecolor="#eff6ff", edgecolor=BLUE, linewidth=1.2))
    ax_r.text(0.6, 9.4, "DDR5 — DIMMs over the PCB",
              color=BLUE, fontsize=10.5, fontweight="bold")
    ax_r.add_patch(Rectangle((1.1, 7.7), 1.6, 1.2,
                             facecolor=BLUE, edgecolor=DARK))
    ax_r.text(1.9, 8.3, "CPU", ha="center", va="center",
              color="white", fontweight="bold", fontsize=10)
    for i, x in enumerate([3.4, 4.4, 5.4, 6.4]):
        ax_r.add_patch(Rectangle((x, 7.7), 0.4, 1.2,
                                 facecolor=GREEN, edgecolor=DARK))
        # PCB trace
        ax_r.plot([2.7, x + 0.2], [8.3, 8.3], color=GRAY, lw=1)
    ax_r.text(8.6, 8.3, "64-bit ×2 ch.\n≈ 96 GB/s",
              ha="center", va="center", color=BLUE, fontsize=9, fontweight="bold")

    # GDDR6X (mid): GPU + 12 chips around it on PCB
    ax_r.add_patch(FancyBboxPatch((0.4, 4.3), 9.2, 2.6,
                                  boxstyle="round,pad=0.03,rounding_size=0.08",
                                  facecolor="#fef3c7", edgecolor=AMBER, linewidth=1.2))
    ax_r.text(0.6, 6.6, "GDDR6X — chips ringing the GPU on the PCB",
              color=AMBER, fontsize=10.5, fontweight="bold")
    ax_r.add_patch(Rectangle((4.0, 4.7), 2.0, 1.5,
                             facecolor=PURPLE, edgecolor=DARK))
    ax_r.text(5.0, 5.45, "GPU die", ha="center", va="center",
              color="white", fontweight="bold", fontsize=10)
    for x, y in [(2.4, 4.6), (2.4, 5.6), (3.0, 4.4), (3.5, 4.4),
                 (6.5, 4.4), (7.0, 4.4), (7.6, 5.6), (7.6, 4.6)]:
        ax_r.add_patch(Rectangle((x, y), 0.4, 0.55,
                                 facecolor=AMBER, edgecolor=DARK))
        ax_r.plot([x + 0.2, 5.0], [y + 0.27, 5.45],
                  color=GRAY, lw=0.8, alpha=0.7)
    ax_r.text(8.6, 5.45, "384-bit\n≈ 1 TB/s",
              ha="center", va="center", color=AMBER, fontsize=9, fontweight="bold")

    # HBM3 (bottom): silicon interposer with 3D stacks beside die
    ax_r.add_patch(FancyBboxPatch((0.4, 1.0), 9.2, 2.7,
                                  boxstyle="round,pad=0.03,rounding_size=0.08",
                                  facecolor="#d1fae5", edgecolor=GREEN, linewidth=1.2))
    ax_r.text(0.6, 3.4, "HBM3 — 3-D stacks on a silicon interposer",
              color=GREEN, fontsize=10.5, fontweight="bold")
    # interposer
    ax_r.add_patch(Rectangle((1.0, 1.4), 7.0, 1.6,
                             facecolor="#a7f3d0", edgecolor=GREEN, linewidth=0.8))
    ax_r.text(1.2, 1.55, "silicon interposer (TSV)",
              color=GREEN, fontsize=8, style="italic")
    # GPU die centered
    ax_r.add_patch(Rectangle((3.7, 1.55), 1.6, 1.3,
                             facecolor=PURPLE, edgecolor=DARK))
    ax_r.text(4.5, 2.2, "GPU\ndie", ha="center", va="center",
              color="white", fontweight="bold", fontsize=9.5)
    # HBM stacks left & right
    for x in [2.0, 5.7]:
        for layer, dy in enumerate([0, 0.18, 0.36, 0.54]):
            ax_r.add_patch(Rectangle((x, 1.55 + dy), 1.4, 0.16,
                                     facecolor=GREEN, alpha=0.6 + 0.1 * layer,
                                     edgecolor=DARK, linewidth=0.4))
        ax_r.text(x + 0.7, 2.5, "HBM\nstack",
                  ha="center", va="center", color=DARK,
                  fontsize=8.5, fontweight="bold")
    ax_r.text(8.6, 2.2, "1024-bit ×2\n≈ 3 TB/s",
              ha="center", va="center", color=GREEN, fontsize=9, fontweight="bold")

    fig.suptitle("Memory Bandwidth — Why GPUs and Accelerators Need More Than DDR",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig5_memory_bandwidth.png")


# ---------------------------------------------------------------------------
# Figure 6: Display interfaces — DP / HDMI / USB-C
# ---------------------------------------------------------------------------

def fig6_display_interfaces() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 6.4),
                                     gridspec_kw={"width_ratios": [1.1, 1]})

    # ----- Left: bandwidth ladder -----
    interfaces = [
        ("VGA (analog)",        0.7,  GRAY,  "1080p60"),
        ("DVI-D dual",          7.9,  GRAY,  "2560×1600"),
        ("HDMI 1.4",            8.16, AMBER, "1080p120 / 4K30"),
        ("HDMI 2.0",            14.4, AMBER, "4K60"),
        ("DisplayPort 1.4",     25.92, BLUE, "4K120 / 8K60 (DSC)"),
        ("HDMI 2.1",            42.6, AMBER, "4K144 / 8K60"),
        ("USB-C / TB4 (DP-Alt)", 40.0, PURPLE,"4K144 + 100 W PD"),
        ("DisplayPort 2.1",     77.4, GREEN, "4K240 / 8K120"),
    ]
    names  = [i[0] for i in interfaces]
    rates  = [i[1] for i in interfaces]
    colors = [i[2] for i in interfaces]
    notes  = [i[3] for i in interfaces]

    bars = ax_l.barh(names, rates, color=colors, edgecolor=DARK, linewidth=1.0)
    ax_l.set_xlabel("Effective payload bandwidth (Gbit/s)")
    ax_l.set_title("Display interface bandwidth (effective, post-encoding)",
                   color=DARK, fontsize=13, pad=8)
    for b, v, note in zip(bars, rates, notes):
        ax_l.text(v + 1.5, b.get_y() + b.get_height() / 2,
                  f"{v:.1f} Gb/s   ·   {note}",
                  va="center", fontsize=9, color=DARK)
    ax_l.set_xlim(0, 100)
    ax_l.invert_yaxis()
    ax_l.grid(True, axis="x", alpha=0.4)

    # ----- Right: feature matrix -----
    ax_r.set_xlim(0, 10); ax_r.set_ylim(0, 10); ax_r.axis("off")
    ax_r.set_title("Choose the right cable for the right job",
                   color=DARK, fontsize=13, pad=8)

    rows = [
        ("Use case",          "Best port",          "Why"),
        ("4K HDR TV / console",   "HDMI 2.1",  "ARC, VRR, ALLM, eARC"),
        ("High-refresh PC monitor", "DP 1.4 / 2.1","Higher refresh, MST hub"),
        ("Single cable laptop dock","USB-C / TB4","Video + power + USB"),
        ("Daisy-chain 2 monitors", "DP MST",     "DP only — HDMI cannot"),
        ("Office / older display", "HDMI 2.0", "Universal; cheap cables OK"),
        ("Legacy projector",    "VGA / DVI",   "No HDR, no audio"),
    ]
    cell_h = 1.25
    col_x  = [0.25, 3.7, 6.4]
    col_w  = [3.4, 2.5, 3.5]
    for i, row in enumerate(rows):
        y = 9.0 - i * cell_h
        is_head = (i == 0)
        for j, txt in enumerate(row):
            face = DARK if is_head else (LIGHT if i % 2 else "white")
            ax_r.add_patch(Rectangle((col_x[j], y - cell_h * 0.45),
                                     col_w[j], cell_h * 0.9,
                                     facecolor=face, edgecolor="#cbd5e1",
                                     linewidth=0.6))
            ax_r.text(col_x[j] + 0.15, y,
                      txt,
                      va="center",
                      color="white" if is_head else DARK,
                      fontsize=9.8 if is_head else 9.3,
                      fontweight="bold" if is_head or j == 1 else "normal")

    fig.suptitle("Display Interfaces — DP, HDMI, USB-C in 2025",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig6_display_interfaces.png")


# ---------------------------------------------------------------------------
# Figure 7: Chipset evolution — northbridge/southbridge → modern PCH
# ---------------------------------------------------------------------------

def fig7_chipset_evolution() -> None:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 7.0))

    def box(ax, x, y, w, h, color, label, sublabel=None, text_color="white"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor=color, edgecolor=DARK, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                color=text_color, fontsize=10.5, fontweight="bold")
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25,
                    sublabel, ha="center", va="center",
                    color=text_color, fontsize=8.5, style="italic")

    def link(ax, x1, y1, x2, y2, label, color=GRAY, lw=1.6, style="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="<->", color=color,
                                    lw=lw, linestyle=style))
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.25,
                label, ha="center", color=color, fontsize=9, fontweight="bold")

    # ---------- LEFT: legacy (pre-2008) ----------
    ax = ax_l
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Legacy era — Northbridge + Southbridge (≈ 1995–2008)",
                 color=AMBER, fontsize=12.5, pad=10)

    box(ax, 4.0, 8.4, 2.0, 1.3, BLUE, "CPU",
        "FSB ↔ northbridge")
    box(ax, 0.4, 8.6, 2.5, 1.0, GREEN, "RAM",  "DDR / DDR2")
    box(ax, 7.1, 8.6, 2.5, 1.0, PURPLE, "AGP / PCIe ×16", "graphics")

    box(ax, 3.0, 5.2, 4.0, 1.6, AMBER, "Northbridge (MCH)",
        "memory + AGP + FSB")
    box(ax, 3.0, 2.3, 4.0, 1.6, "#b45309", "Southbridge (ICH)",
        "PCI · USB · SATA · audio · BIOS")

    # legacy peripherals
    for i, lbl in enumerate(["PCI", "USB 1.1/2.0", "ATA / SATA", "AC'97 audio"]):
        x = 0.3 + i * 2.5
        box(ax, x, 0.4, 2.2, 1.1, "#475569", lbl)
        ax.annotate("", xy=(x + 1.1, 1.5), xytext=(5.0, 2.3),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1))

    link(ax, 5.0, 8.4, 5.0, 6.8, "FSB", color=BLUE, lw=2)
    link(ax, 2.9, 9.1, 3.0, 6.5, "DDR bus", color=GREEN)
    link(ax, 7.1, 9.1, 7.0, 6.5, "AGP / PCIe", color=PURPLE)
    link(ax, 5.0, 5.2, 5.0, 3.9, "DMI / hub-link", color="#b45309", lw=2)

    ax.text(0.4, 0.05,
            "FSB chokepoint · memory and PCIe both tunnel through the northbridge.",
            color=AMBER, fontsize=9, style="italic")

    # ---------- RIGHT: modern (≈ 2010 → today) ----------
    ax = ax_r
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.set_title("Modern era — northbridge absorbed into the CPU (2010 →)",
                 color=GREEN, fontsize=12.5, pad=10)

    box(ax, 3.0, 7.2, 4.0, 2.4, BLUE,
        "CPU package",
        "cores · IMC · PCIe root complex · iGPU")

    box(ax, 0.3, 7.6, 2.3, 1.5, GREEN, "DDR5 DIMMs",
        "direct ↔ CPU IMC")
    box(ax, 7.4, 7.6, 2.3, 1.5, PURPLE, "PCIe ×16 GPU",
        "direct ↔ CPU root")
    box(ax, 7.4, 5.4, 2.3, 1.4, PURPLE, "M.2 NVMe",
        "PCIe ×4 from CPU")

    link(ax, 2.6, 8.4, 3.0, 8.4, "DDR5",      color=GREEN, lw=2)
    link(ax, 7.0, 8.4, 7.4, 8.4, "PCIe 5 ×16", color=PURPLE, lw=2)
    link(ax, 7.0, 7.4, 7.4, 6.1, "PCIe 5 ×4",  color=PURPLE, lw=1.5)

    # PCH (chipset) only handles legacy / low-speed
    box(ax, 3.0, 4.0, 4.0, 1.7, AMBER,
        "PCH / chipset",
        "Z790 · B650 — slow lanes only")

    link(ax, 5.0, 7.2, 5.0, 5.7, "DMI 4.0 ×8\n≈ 16 GB/s", color=AMBER, lw=2)

    # peripherals from PCH
    pcs = ["SATA", "USB 3.2", "extra PCIe ×4", "Wi-Fi / GbE", "audio"]
    for i, lbl in enumerate(pcs):
        x = 0.2 + i * 2.0
        box(ax, x, 1.3, 1.85, 1.0, "#64748b", lbl)
        ax.annotate("", xy=(x + 0.92, 2.3), xytext=(5.0, 4.0),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9))

    ax.text(0.2, 0.4,
            "GPU + RAM + primary NVMe sit on CPU-direct lanes — full bandwidth, low latency.\n"
            "Everything else shares the DMI link to the chipset (one bottleneck for many devices).",
            color=DARK, fontsize=9, style="italic")

    fig.suptitle("Chipset Architecture — From Two Chips to One Bus",
                 fontsize=15, fontweight="bold", color=DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig7_chipset_evolution.png")


# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating motherboard / GPU figures to:")
    for d in OUT_DIRS:
        print(f"  - {d}")
    fig1_motherboard_layout()
    fig2_pcie_generations()
    fig3_gpu_simt()
    fig4_cpu_vs_gpu()
    fig5_memory_bandwidth()
    fig6_display_interfaces()
    fig7_chipset_evolution()
    print("Done.")


if __name__ == "__main__":
    main()
