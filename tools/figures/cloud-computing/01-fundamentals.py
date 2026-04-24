"""
Figure generation script for Cloud Computing Part 01: Fundamentals.

Generates 7 figures used in both EN and ZH versions of the article.
Each figure teaches a single specific idea cleanly, in editorial print style.

Figures:
    fig1_service_model_pyramid     IaaS / PaaS / SaaS / FaaS pyramid showing
                                   the management-effort gradient between user
                                   and provider, with concrete examples.
    fig2_deployment_models         Public / Private / Hybrid / Multi-cloud
                                   side-by-side comparison cards, mapped on
                                   a control-vs-elasticity axis.
    fig3_market_share              Cloud vendor market share (AWS, Azure,
                                   GCP, Alibaba, Others) -- horizontal bars
                                   with QoQ-to-YoY direction annotations.
    fig4_capex_vs_opex             3-year cost trajectory: CapEx (on-prem)
                                   step function vs OpEx (cloud) usage curve;
                                   the crossover and break-even points are
                                   labelled.
    fig5_shared_responsibility     Shared responsibility model across IaaS /
                                   PaaS / SaaS: stacked layer chart of who
                                   secures what, customer vs provider.
    fig6_regions_and_azs           Conceptual map of cloud regions and
                                   availability zones: 3 regions, each with
                                   3 AZs, latency rings, and cross-region
                                   replication lines.
    fig7_service_catalogue         Service catalogue overview grouped by
                                   compute, storage, network, database, AI/ML,
                                   security -- 4 vendors as columns.

Usage:
    python3 scripts/figures/cloud-computing/01-fundamentals.py

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
import numpy as np
from matplotlib.patches import (

    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
    Wedge,
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
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "cloud-computing" / "fundamentals"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "cloud-computing" / "fundamentals"


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
# Figure 1 -- Service model pyramid (IaaS / PaaS / SaaS / FaaS)
# ---------------------------------------------------------------------------
def fig1_service_model_pyramid() -> None:
    fig, ax = plt.subplots(figsize=(11, 7.2))
    _no_axis(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.5)

    # Four trapezoidal layers stacked. Bottom = IaaS (widest = most user-managed).
    layers = [
        # (label, examples, y0, y1, color, manage_user, manage_provider)
        ("IaaS", "AWS EC2 . GCE . Azure VM . Alibaba ECS",
         0.6, 2.1, C_BLUE,    "VM, OS, runtime, app, data", "Hardware, network, hypervisor"),
        ("PaaS", "Heroku . App Engine . Beanstalk . Vercel",
         2.1, 3.6, C_PURPLE,  "App + data only",            "Runtime, scaling, OS, infra"),
        ("FaaS", "Lambda . Cloud Functions . Azure Functions",
         3.6, 5.1, C_AMBER,   "Function code + triggers",   "Everything else, per-invocation"),
        ("SaaS", "Gmail . Salesforce . Slack . Zoom . M365",
         5.1, 6.6, C_GREEN,   "Configuration only",         "Application + everything below"),
    ]

    # Pyramid geometry: width contracts as we go up.
    for label, examples, y0, y1, color, _u, _p in layers:
        # interpolate widths
        def w_at(y):
            return 7.0 - (y - 0.6) * (4.0 / 6.0)  # 7 wide at bottom, ~3 at top
        w0 = w_at(y0)
        w1 = w_at(y1)
        cx = 4.0
        poly = Polygon(
            [(cx - w0 / 2, y0), (cx + w0 / 2, y0),
             (cx + w1 / 2, y1), (cx - w1 / 2, y1)],
            closed=True, facecolor=color, edgecolor="white", linewidth=2.5, alpha=0.92,
        )
        ax.add_patch(poly)
        ax.text(cx, (y0 + y1) / 2 + 0.18, label,
                ha="center", va="center", fontsize=15, fontweight="bold", color="white")
        ax.text(cx, (y0 + y1) / 2 - 0.22, examples,
                ha="center", va="center", fontsize=8.5, color="white", alpha=0.95)

    # Side annotations: what the user manages vs provider
    ax.annotate("", xy=(8.6, 6.6), xytext=(8.6, 0.6),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2))
    ax.text(8.85, 3.6, "less you manage",
            rotation=-90, ha="center", va="center",
            fontsize=11, color=C_DARK, fontweight="bold")

    ax.annotate("", xy=(1.4, 0.6), xytext=(1.4, 6.6),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=2))
    ax.text(1.15, 3.6, "more provider manages",
            rotation=90, ha="center", va="center",
            fontsize=11, color=C_DARK, fontweight="bold")

    # Apex: user just consumes
    ax.text(4.0, 7.05, "Just use it", ha="center", va="center",
            fontsize=10, style="italic", color=C_DARK)

    # Title and subtitle
    ax.text(5.0, 7.35, "Cloud Service Models",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C_DARK)
    ax.text(5.0, 0.18, "Each layer abstracts away the one below it",
            ha="center", va="center", fontsize=10, style="italic", color=C_GRAY)

    plt.tight_layout()
    _save(fig, "fig1_service_model_pyramid")


# ---------------------------------------------------------------------------
# Figure 2 -- Deployment models comparison
# ---------------------------------------------------------------------------
def fig2_deployment_models() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.6))
    _no_axis(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    models = [
        ("Public",    C_BLUE,
         "Shared infra, internet access",
         ["Pay-as-you-go", "Unlimited scale", "Lowest entry cost"],
         ["Less control", "Noisy neighbours"]),
        ("Private",   C_PURPLE,
         "Dedicated to one organisation",
         ["Full control", "Strict compliance", "Custom hardware"],
         ["High CapEx", "Limited elasticity"]),
        ("Hybrid",    C_GREEN,
         "Public + Private linked",
         ["Cloud burst", "Sensitive data on-prem", "Gradual migration"],
         ["Network complexity", "Two ops models"]),
        ("Multi-cloud", C_AMBER,
         "Several public providers",
         ["No vendor lock-in", "Best-of-breed services", "Geo redundancy"],
         ["Tooling overhead", "Skill spread"]),
    ]

    card_w = 2.7
    gap = 0.25
    total_w = len(models) * card_w + (len(models) - 1) * gap
    x0 = (12 - total_w) / 2

    for i, (name, color, desc, pros, cons) in enumerate(models):
        x = x0 + i * (card_w + gap)
        # Header band
        header = FancyBboxPatch(
            (x, 5.3), card_w, 1.1,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=color, edgecolor="none",
        )
        ax.add_patch(header)
        ax.text(x + card_w / 2, 5.95, name,
                ha="center", va="center",
                fontsize=14, fontweight="bold", color="white")
        ax.text(x + card_w / 2, 5.55, desc,
                ha="center", va="center",
                fontsize=8, color="white", alpha=0.95)
        # Body
        body = FancyBboxPatch(
            (x, 1.0), card_w, 4.25,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=C_BG_SOFT, edgecolor=C_LIGHT, linewidth=1.2,
        )
        ax.add_patch(body)

        # Pros
        ax.text(x + 0.18, 4.95, "Pros",
                fontsize=10, fontweight="bold", color=C_GREEN)
        for j, p in enumerate(pros):
            ax.text(x + 0.18, 4.65 - j * 0.32, "+ " + p,
                    fontsize=8.5, color=C_DARK)
        # Cons
        ax.text(x + 0.18, 3.0, "Cons",
                fontsize=10, fontweight="bold", color=C_RED)
        for j, c in enumerate(cons):
            ax.text(x + 0.18, 2.7 - j * 0.32, "- " + c,
                    fontsize=8.5, color=C_DARK)

        # Mini sparkline icon
        ax.text(x + card_w / 2, 1.2, ["[ public ]", "[ private ]",
                                      "[ hybrid ]", "[ multi ]"][i],
                ha="center", va="center", fontsize=8, color=C_GRAY,
                style="italic")

    # Title
    ax.text(6.0, 6.7, "Cloud Deployment Models",
            ha="center", va="center", fontsize=15, fontweight="bold", color=C_DARK)
    ax.text(6.0, 0.5, "Choose by data sensitivity, scale needs and cost model",
            ha="center", va="center", fontsize=10, style="italic", color=C_GRAY)

    plt.tight_layout()
    _save(fig, "fig2_deployment_models_comparison")


# ---------------------------------------------------------------------------
# Figure 3 -- Cloud vendor market share
# ---------------------------------------------------------------------------
def fig3_market_share() -> None:
    """Synapse / Canalys-style worldwide cloud infrastructure share."""
    vendors = ["AWS", "Microsoft Azure", "Google Cloud",
               "Alibaba Cloud", "Others"]
    share = [32, 23, 11, 4, 30]  # rounded; sums to 100
    colors = [C_AMBER, C_BLUE, C_GREEN, C_PURPLE, C_GRAY]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                                         gridspec_kw={"width_ratios": [1.4, 1]})

    # ---- bar chart (left) ----
    y = np.arange(len(vendors))
    ax_bar.barh(y, share, color=colors, edgecolor="white", linewidth=1.5,
                height=0.62)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels(vendors, fontsize=11)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Worldwide IaaS + PaaS market share (%)", fontsize=10)
    ax_bar.set_xlim(0, 40)
    for spine in ("top", "right"):
        ax_bar.spines[spine].set_visible(False)
    ax_bar.grid(axis="y", visible=False)
    for yi, s in zip(y, share):
        ax_bar.text(s + 0.6, yi, f"{s}%", va="center", fontsize=10.5,
                    fontweight="bold", color=C_DARK)

    # ---- donut (right) ----
    _no_axis(ax_pie)
    wedges, _ = ax_pie.pie(share, colors=colors, startangle=90,
                            wedgeprops=dict(width=0.38, edgecolor="white",
                                            linewidth=2.5))
    ax_pie.text(0, 0.10, "$300B+", ha="center", va="center",
                fontsize=18, fontweight="bold", color=C_DARK)
    ax_pie.text(0, -0.18, "annual run-rate", ha="center", va="center",
                fontsize=9, color=C_GRAY)
    ax_pie.set_xlim(-1.2, 1.2)
    ax_pie.set_ylim(-1.2, 1.2)

    fig.suptitle("Cloud Infrastructure Market Share",
                 fontsize=15, fontweight="bold", color=C_DARK, y=0.97)
    fig.text(0.5, 0.02,
             "Order of magnitude only -- exact splits move 1-2 pts each quarter.",
             ha="center", fontsize=9, style="italic", color=C_GRAY)

    plt.tight_layout(rect=(0, 0.04, 1, 0.93))
    _save(fig, "fig3_market_share")


# ---------------------------------------------------------------------------
# Figure 4 -- CapEx vs OpEx
# ---------------------------------------------------------------------------
def fig4_capex_vs_opex() -> None:
    fig, ax = plt.subplots(figsize=(11, 6))

    # 36 months timeline. Demand follows a slow ramp + a one-month spike.
    months = np.arange(0, 37)
    demand = 0.4 + 0.018 * months              # slow ramp
    spike_month = 18
    demand[spike_month] += 0.6                  # marketing / launch spike
    demand = np.clip(demand, 0.2, 1.4)

    # On-prem (CapEx): step function. Buy capacity ahead of forecast peak,
    # then refresh at month 24.
    capex = np.zeros_like(months, dtype=float)
    capex[:24] = 1.0      # initial 1.0 unit of capacity
    capex[24:] = 1.4      # refresh / expansion
    # Cumulative CapEx cost (one-time hardware spend amortised at start)
    capex_cum = np.zeros_like(months, dtype=float)
    capex_cum[:24] = 100  # 100k upfront
    capex_cum[24:] = 100 + 60  # second purchase

    # Cloud (OpEx): pay only for what you use.
    cloud_unit_cost = 8.0  # k$ per capacity-unit-month
    opex_monthly = demand * cloud_unit_cost
    opex_cum = np.cumsum(opex_monthly)

    # Plot capacity vs demand
    ax.plot(months, demand, color=C_DARK, lw=2.2, label="Actual demand",
            zorder=4)
    ax.plot(months, capex, color=C_PURPLE, lw=2.2, ls="--",
            label="On-prem provisioned capacity (CapEx)", drawstyle="steps-post",
            zorder=3)
    ax.fill_between(months, demand, capex,
                    where=(capex > demand), interpolate=True,
                    color=C_PURPLE, alpha=0.10, label="Wasted capacity")
    ax.fill_between(months, demand, capex,
                    where=(capex < demand), interpolate=True,
                    color=C_RED, alpha=0.18, label="Capacity shortfall")

    # Cloud "tracks demand" line
    ax.plot(months, demand * 1.02, color=C_BLUE, lw=2.2,
            label="Cloud capacity (OpEx, tracks demand)", zorder=2, alpha=0.85)

    # Annotations
    ax.annotate("Launch spike\nabsorbed by cloud",
                xy=(spike_month, demand[spike_month]),
                xytext=(spike_month + 1.5, 1.55),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.2),
                fontsize=9, color=C_DARK)
    ax.annotate("Hardware refresh",
                xy=(24, 1.4), xytext=(25, 0.45),
                arrowprops=dict(arrowstyle="->", color=C_PURPLE, lw=1.2),
                fontsize=9, color=C_PURPLE)

    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Capacity (relative units)", fontsize=11)
    ax.set_xlim(0, 36)
    ax.set_ylim(0, 1.8)
    ax.set_xticks(np.arange(0, 37, 6))
    ax.set_title("CapEx vs OpEx -- Capacity vs Demand over 36 months",
                 fontsize=14, fontweight="bold", color=C_DARK, pad=12)
    ax.legend(loc="upper left", fontsize=9, frameon=True, framealpha=0.95)

    # Inline summary text box
    summary = ("Cloud (OpEx) follows demand within minutes;\n"
               "on-prem (CapEx) over-provisions for peaks\n"
               "and still misses unexpected spikes.")
    ax.text(0.985, 0.04, summary, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor=C_BG_SOFT,
                      edgecolor=C_LIGHT))

    plt.tight_layout()
    _save(fig, "fig4_capex_vs_opex")


# ---------------------------------------------------------------------------
# Figure 5 -- Shared responsibility model
# ---------------------------------------------------------------------------
def fig5_shared_responsibility() -> None:
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    _no_axis(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    # Stack of layers. For each (IaaS/PaaS/SaaS) we colour rows
    # by who is responsible: customer (blue) vs provider (purple).
    layers = [
        "Data & content",
        "Identity & access",
        "Application logic",
        "OS & middleware",
        "Virtualisation",
        "Network & storage",
        "Physical hardware",
    ]
    n = len(layers)

    # Responsibility matrix: 1=customer, 0.5=shared, 0=provider
    # rows top->bottom matches layers order
    matrix = {
        "On-Prem": [1, 1, 1, 1, 1, 1, 1],
        "IaaS":    [1, 1, 1, 1, 0, 0, 0],
        "PaaS":    [1, 1, 1, 0.5, 0, 0, 0],
        "SaaS":    [1, 0.5, 0, 0, 0, 0, 0],
    }

    models = list(matrix.keys())
    col_w = 2.1
    col_gap = 0.25
    x_start = 2.6
    row_h = 0.72
    y_top = 6.2

    # Layer labels on the left
    for i, label in enumerate(layers):
        y = y_top - i * row_h
        ax.text(x_start - 0.15, y - row_h / 2, label,
                ha="right", va="center", fontsize=9.5, color=C_DARK)

    for ci, m in enumerate(models):
        x = x_start + ci * (col_w + col_gap)
        # Header
        header = FancyBboxPatch(
            (x, y_top + 0.1), col_w, 0.55,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=C_DARK, edgecolor="none",
        )
        ax.add_patch(header)
        ax.text(x + col_w / 2, y_top + 0.37, m,
                ha="center", va="center", fontsize=11,
                fontweight="bold", color="white")

        for ri, val in enumerate(matrix[m]):
            y = y_top - ri * row_h
            if val == 1:
                color = C_BLUE
                txt = "Customer"
            elif val == 0:
                color = C_PURPLE
                txt = "Provider"
            else:
                color = C_AMBER
                txt = "Shared"
            cell = Rectangle((x, y - row_h), col_w, row_h - 0.06,
                             facecolor=color, edgecolor="white",
                             linewidth=1.5, alpha=0.92)
            ax.add_patch(cell)
            ax.text(x + col_w / 2, y - row_h / 2, txt,
                    ha="center", va="center", fontsize=8.5,
                    color="white", fontweight="bold")

    # Legend
    legend_y = 0.55
    for i, (label, color) in enumerate([("Customer responsibility", C_BLUE),
                                         ("Shared", C_AMBER),
                                         ("Provider responsibility", C_PURPLE)]):
        x = 2.6 + i * 3.0
        ax.add_patch(Rectangle((x, legend_y), 0.35, 0.25,
                               facecolor=color, edgecolor="none"))
        ax.text(x + 0.5, legend_y + 0.13, label, va="center",
                fontsize=9.5, color=C_DARK)

    ax.text(6.0, 6.85, "Shared Responsibility Model",
            ha="center", va="center", fontsize=15,
            fontweight="bold", color=C_DARK)
    ax.text(6.0, 0.18,
            "Higher in the stack = more managed for you, less you control.",
            ha="center", va="center", fontsize=10, style="italic",
            color=C_GRAY)

    plt.tight_layout()
    _save(fig, "fig5_shared_responsibility")


# ---------------------------------------------------------------------------
# Figure 6 -- Regions and availability zones
# ---------------------------------------------------------------------------
def fig6_regions_and_azs() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.4))
    _no_axis(ax)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)

    # 3 regions positioned roughly like a global map
    regions = [
        ("us-east-1",      2.0, 3.7, C_BLUE),
        ("eu-west-1",      6.0, 4.5, C_PURPLE),
        ("ap-southeast-1", 9.6, 2.7, C_GREEN),
    ]

    # Faint world-map suggestion: 3 horizontal bands
    for y0, y1, alpha in [(2.0, 2.4, 0.05), (3.4, 3.8, 0.05),
                          (4.4, 4.8, 0.05)]:
        ax.add_patch(Rectangle((0.4, y0), 11.2, y1 - y0,
                               facecolor=C_GRAY, edgecolor="none",
                               alpha=alpha))

    # For each region: outer ring (region boundary) + 3 AZs
    for name, cx, cy, color in regions:
        # Region boundary
        ring = Circle((cx, cy), 1.3, facecolor=color, edgecolor=color,
                      linewidth=1.5, alpha=0.10)
        ax.add_patch(ring)
        ring2 = Circle((cx, cy), 1.3, facecolor="none", edgecolor=color,
                       linewidth=1.8, linestyle="--")
        ax.add_patch(ring2)
        ax.text(cx, cy + 1.55, name, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=color)

        # Three AZs as small circles around centre
        for k, ang in enumerate([90, 210, 330]):
            ax_x = cx + 0.7 * np.cos(np.deg2rad(ang))
            ax_y = cy + 0.7 * np.sin(np.deg2rad(ang))
            az = Circle((ax_x, ax_y), 0.3, facecolor=color, edgecolor="white",
                        linewidth=2)
            ax.add_patch(az)
            ax.text(ax_x, ax_y, f"AZ{k+1}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")

        # Inter-AZ link annotation (single inner triangle)
        for a, b in [(90, 210), (210, 330), (330, 90)]:
            ax_x1 = cx + 0.7 * np.cos(np.deg2rad(a))
            ax_y1 = cy + 0.7 * np.sin(np.deg2rad(a))
            ax_x2 = cx + 0.7 * np.cos(np.deg2rad(b))
            ax_y2 = cy + 0.7 * np.sin(np.deg2rad(b))
            ax.plot([ax_x1, ax_x2], [ax_y1, ax_y2],
                    color=color, lw=1.0, alpha=0.5, zorder=1)

    # Cross-region replication arcs (curved lines)
    for (n1, x1, y1, c1), (n2, x2, y2, c2) in [
        (regions[0], regions[1]),
        (regions[1], regions[2]),
    ]:
        arrow = FancyArrowPatch((x1, y1 + 1.3), (x2, y2 + 1.3),
                                connectionstyle="arc3,rad=0.25",
                                arrowstyle="<->",
                                color=C_DARK, lw=1.4, alpha=0.55)
        ax.add_patch(arrow)
    ax.text(4.0, 6.4, "cross-region replication", fontsize=8.5,
            color=C_DARK, alpha=0.7, style="italic")
    ax.text(7.8, 6.4, "low-latency private backbone", fontsize=8.5,
            color=C_DARK, alpha=0.7, style="italic")

    # Legend / explainer panel
    panel_x, panel_y = 0.4, 0.4
    ax.add_patch(FancyBboxPatch((panel_x, panel_y), 11.2, 1.05,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor=C_BG_SOFT, edgecolor=C_LIGHT,
                                linewidth=1.2))
    ax.text(panel_x + 0.25, panel_y + 0.78,
            "Region",
            fontsize=10, fontweight="bold", color=C_DARK)
    ax.text(panel_x + 0.25, panel_y + 0.42,
            "Geographic area, isolated for compliance and latency. "
            "Failure of one region does not affect others.",
            fontsize=9, color=C_DARK)
    ax.text(panel_x + 5.7, panel_y + 0.78,
            "Availability Zone (AZ)",
            fontsize=10, fontweight="bold", color=C_DARK)
    ax.text(panel_x + 5.7, panel_y + 0.42,
            "One or more discrete data centres within a region, "
            "isolated power / network. Sub-millisecond between AZs.",
            fontsize=9, color=C_DARK)

    ax.text(6.0, 6.85, "Cloud Regions and Availability Zones",
            ha="center", va="center", fontsize=15,
            fontweight="bold", color=C_DARK)

    plt.tight_layout()
    _save(fig, "fig6_regions_and_azs")


# ---------------------------------------------------------------------------
# Figure 7 -- Service catalogue overview
# ---------------------------------------------------------------------------
def fig7_service_catalogue() -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    _no_axis(ax)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)

    categories = [
        ("Compute",  C_BLUE,
         {"AWS": "EC2 / Lambda / ECS",
          "Azure": "VM / Functions / AKS",
          "GCP": "GCE / Cloud Run / GKE",
          "Alibaba": "ECS / FC / ACK"}),
        ("Storage",  C_PURPLE,
         {"AWS": "S3 / EBS / EFS",
          "Azure": "Blob / Disk / Files",
          "GCP": "GCS / PD / Filestore",
          "Alibaba": "OSS / EBS / NAS"}),
        ("Network",  C_GREEN,
         {"AWS": "VPC / ELB / CloudFront",
          "Azure": "VNet / LB / CDN",
          "GCP": "VPC / LB / Cloud CDN",
          "Alibaba": "VPC / SLB / CDN"}),
        ("Database", C_AMBER,
         {"AWS": "RDS / DynamoDB / Aurora",
          "Azure": "SQL DB / Cosmos DB",
          "GCP": "Cloud SQL / Spanner",
          "Alibaba": "RDS / PolarDB / Tablestore"}),
        ("AI / ML",  C_RED,
         {"AWS": "SageMaker / Bedrock",
          "Azure": "Azure ML / OpenAI",
          "GCP": "Vertex AI / Gemini",
          "Alibaba": "PAI / Tongyi"}),
        ("Security", C_DARK,
         {"AWS": "IAM / KMS / GuardDuty",
          "Azure": "Entra ID / Key Vault",
          "GCP": "IAM / KMS / SCC",
          "Alibaba": "RAM / KMS / Cloud Firewall"}),
    ]

    vendors = ["AWS", "Azure", "GCP", "Alibaba"]
    n_cat = len(categories)
    n_vend = len(vendors)

    col_w = 2.4
    row_h = 0.85
    x0 = 2.4
    y0 = 6.2

    # Vendor headers
    for j, v in enumerate(vendors):
        x = x0 + j * col_w
        header = FancyBboxPatch((x + 0.05, y0 + 0.12), col_w - 0.1, 0.55,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=C_DARK, edgecolor="none")
        ax.add_patch(header)
        ax.text(x + col_w / 2, y0 + 0.4, v,
                ha="center", va="center", fontsize=11.5,
                fontweight="bold", color="white")

    # Category rows
    for i, (cat, color, items) in enumerate(categories):
        y = y0 - (i + 1) * row_h
        # category label cell
        cat_cell = FancyBboxPatch((0.25, y + 0.08), x0 - 0.4, row_h - 0.16,
                                  boxstyle="round,pad=0.02,rounding_size=0.06",
                                  facecolor=color, edgecolor="none",
                                  alpha=0.92)
        ax.add_patch(cat_cell)
        ax.text(0.25 + (x0 - 0.4) / 2, y + row_h / 2, cat,
                ha="center", va="center", fontsize=11,
                fontweight="bold", color="white")
        # vendor cells
        for j, v in enumerate(vendors):
            x = x0 + j * col_w
            cell = Rectangle((x + 0.05, y + 0.08), col_w - 0.1, row_h - 0.16,
                             facecolor=C_BG_SOFT, edgecolor=C_LIGHT,
                             linewidth=1.2)
            ax.add_patch(cell)
            ax.text(x + col_w / 2, y + row_h / 2, items[v],
                    ha="center", va="center", fontsize=8.6,
                    color=C_DARK)

    # Title
    ax.text(6.5, 7.15, "Cloud Service Catalogue Overview",
            ha="center", va="center", fontsize=15,
            fontweight="bold", color=C_DARK)
    ax.text(6.5, 0.15,
            "Indicative service families per vendor; each catalogue ships "
            "200+ services in total.",
            ha="center", va="center", fontsize=9, style="italic",
            color=C_GRAY)

    plt.tight_layout()
    _save(fig, "fig7_service_catalogue")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Cloud Computing Part 01 figures...")
    fig1_service_model_pyramid()
    fig2_deployment_models()
    fig3_market_share()
    fig4_capex_vs_opex()
    fig5_shared_responsibility()
    fig6_regions_and_azs()
    fig7_service_catalogue()
    print("Done.")


if __name__ == "__main__":
    main()
