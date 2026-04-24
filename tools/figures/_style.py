"""
Shared aesthetic style for all chenk-site figures.

Every figure script should import:
    from _style import setup_style, COLORS, save_figure

Goals:
  - Modern, magazine-quality look (ample whitespace, refined typography)
  - Cohesive palette across the entire site
  - English-only labels (no CN unless intentional like multilingual examples)
"""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl


# === Palette =================================================================
COLORS = {
    "primary":  "#2563eb",   # blue
    "accent":   "#7c3aed",   # purple
    "success":  "#10b981",   # green
    "warning":  "#f59e0b",   # amber
    "danger":   "#ef4444",   # red
    "info":     "#06b6d4",   # cyan
    "pink":     "#ec4899",
    "indigo":   "#6366f1",
    "text":     "#1e293b",
    "text2":    "#475569",
    "muted":    "#94a3b8",
    "border":   "#cbd5e1",
    "grid":     "#e2e8f0",
    "bg":       "#fafafa",
    "card":     "#ffffff",
}

CYCLE = [
    COLORS["primary"], COLORS["accent"], COLORS["success"],
    COLORS["warning"], COLORS["info"], COLORS["pink"],
    COLORS["indigo"], COLORS["danger"],
]


def setup_style() -> None:
    """Apply the chenk-site matplotlib defaults. Call once at top of script."""
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update({
        # Figure
        "figure.dpi": 160,
        "figure.facecolor": "white",
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.25,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",

        # Fonts — refined sans-serif stack, English-friendly
        "font.family": ["Inter", "Helvetica Neue", "DejaVu Sans", "Arial"],
        "font.size": 11,
        "axes.titlesize": 13.5,
        "axes.titleweight": "bold",
        "axes.titlepad": 14,
        "axes.labelsize": 11,
        "axes.labelweight": "regular",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10.5,

        # Axes — minimal chrome
        "axes.facecolor": "white",
        "axes.edgecolor": COLORS["border"],
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlecolor": COLORS["text"],
        "axes.labelcolor": COLORS["text2"],
        "axes.prop_cycle": mpl.cycler(color=CYCLE),

        # Ticks — subtle
        "xtick.color": COLORS["text2"],
        "ytick.color": COLORS["text2"],
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,

        # Grid — soft, in background
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.55,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "axes.axisbelow": True,

        # Legend
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.edgecolor": COLORS["grid"],
        "legend.framealpha": 0.95,
        "legend.borderpad": 0.5,
        "legend.labelspacing": 0.4,

        # Lines / patches
        "lines.linewidth": 2.2,
        "lines.markersize": 6,
        "patch.linewidth": 0.6,

        # Text
        "text.color": COLORS["text"],
    })


def save_figure(fig, en_dir: Path, zh_dir: Path | None, name: str) -> None:
    """Save figure into one or both target dirs (EN + ZH share images)."""
    en_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(en_dir / name)
    if zh_dir is not None:
        zh_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(zh_dir / name)
    plt.close(fig)


def add_title_subtitle(ax, title: str, subtitle: str | None = None) -> None:
    """Apply a two-line title with refined hierarchy."""
    ax.set_title(title, loc="left", pad=18,
                 fontsize=14, fontweight="bold", color=COLORS["text"])
    if subtitle:
        ax.text(0, 1.04, subtitle, transform=ax.transAxes,
                fontsize=10.5, color=COLORS["text2"], ha="left", va="bottom")


def annotate_callout(ax, text: str, xy, xytext, color: str = None) -> None:
    """Drop a refined annotation callout."""
    if color is None:
        color = COLORS["primary"]
    ax.annotate(
        text, xy=xy, xytext=xytext,
        fontsize=10, color=color, fontweight="medium",
        bbox=dict(boxstyle="round,pad=0.4", fc="white",
                  ec=color, lw=1.2, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.4,
                        connectionstyle="arc3,rad=0.15"),
    )
