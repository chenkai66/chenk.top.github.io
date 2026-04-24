"""Shared matplotlib style for chenk.top blog figures.

Use:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from _style import setup_style, COLORS, save_figure
    setup_style()

Then reference colors via ``COLORS["primary"]`` etc, and optionally save
with ``save_figure(fig, [out_dir1, out_dir2], "name.png")``.

Design goals: a polished, modern aesthetic — bolder titles, generous
padding, clean axes (no top/right spines), consistent palette.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import matplotlib.pyplot as plt


# Modern, cohesive palette (inspired by Tailwind 600 weights).
COLORS = {
    "primary":   "#2563eb",   # blue
    "accent":    "#7c3aed",   # purple
    "success":   "#10b981",   # green
    "warning":   "#f59e0b",   # amber
    "danger":    "#ef4444",   # red
    "info":      "#06b6d4",   # cyan
    "pink":      "#ec4899",
    "slate":     "#475569",
    "gray":      "#6b7280",
    "muted":     "#94a3b8",
    "light":     "#f1f5f9",
    "bg":        "#ffffff",
    "ink":       "#1e293b",   # default text
    "grid":      "#e2e8f0",
}

# Backwards-compatible aliases for scripts that import single names.
COLORS["blue"]   = COLORS["primary"]
COLORS["purple"] = COLORS["accent"]
COLORS["green"]  = COLORS["success"]
COLORS["amber"]  = COLORS["warning"]
COLORS["orange"] = COLORS["warning"]
COLORS["red"]    = COLORS["danger"]


def setup_style() -> None:
    """Apply the global matplotlib style. Idempotent."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi":          150,
        "savefig.dpi":         150,
        "savefig.bbox":        "tight",
        "savefig.facecolor":   "white",
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",

        "font.family":         "DejaVu Sans",
        "font.size":           11,

        "axes.titlesize":      13.5,
        "axes.titleweight":    "bold",
        "axes.titlepad":       12,
        "axes.titlelocation":  "left",

        "axes.labelsize":      11,
        "axes.labelcolor":     COLORS["ink"],

        "axes.edgecolor":      COLORS["muted"],
        "axes.linewidth":      0.9,
        "axes.spines.top":     False,
        "axes.spines.right":   False,

        "grid.color":          COLORS["grid"],
        "grid.linewidth":      0.7,
        "grid.alpha":          0.8,

        "xtick.color":         COLORS["slate"],
        "ytick.color":         COLORS["slate"],
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,

        "legend.frameon":      True,
        "legend.framealpha":   0.92,
        "legend.edgecolor":    COLORS["grid"],
        "legend.fontsize":     10,

        "figure.titlesize":    15,
        "figure.titleweight":  "bold",
    })


PathLike = Union[str, Path]


def save_figure(
    fig: plt.Figure,
    out_dirs: Union[PathLike, Iterable[PathLike]],
    name: str,
    *,
    close: bool = True,
) -> None:
    """Save ``fig`` as ``name`` into one or more directories.

    Creates directories if missing. Always uses tight bbox, white bg.
    """
    if isinstance(out_dirs, (str, Path)):
        out_dirs = [out_dirs]
    for d in out_dirs:
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white", bbox_inches="tight")
    if close:
        plt.close(fig)
