"""
Figure generation script for Cloud Computing Part 08: Multi-Cloud & Hybrid.

Generates 5 figures used in both EN and ZH versions of the article. Each
figure teaches one concrete idea cleanly and is sized for blog reading.

Figures:
    fig1_multi_cloud_architecture
        Workloads distributed across AWS / Azure / GCP behind a global
        traffic manager, with shared identity and observability layers.

    fig2_hybrid_connectivity
        Three connectivity options (Site-to-site VPN, Direct Connect /
        ExpressRoute, SD-WAN) compared on a bandwidth x latency plane,
        with on-prem -> cloud paths drawn for each.

    fig3_lockin_mitigation
        Five lock-in dimensions (data, API, architecture, skills, contract)
        as a stacked maturity radar showing baseline vs portable strategy.

    fig4_cross_cloud_data_sync
        Cross-region active-active replication topology with three sync
        patterns annotated (sync, async, CDC stream) and conflict
        resolution callouts.

    fig5_cost_optimization
        Cumulative monthly spend chart comparing on-demand baseline,
        reserved/committed-use, spot mix, and a fully-optimized
        (RI + spot + right-size + FinOps) curve.

Usage:
    python3 scripts/figures/cloud-computing/08-multi-cloud.py

Output:
    Writes PNGs to BOTH the EN and ZH article asset folders so the markdown
    references stay consistent across languages.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "cloud-computing" / "multi-cloud-hybrid"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "cloud-computing" / "multi-cloud-hybrid"


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(ax, x, y, w, h, text, *, fc=C_BLUE, ec=None, tc="white",
         fontsize=9, weight="normal", radius=0.04):
    ec = ec or fc
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=1.1, facecolor=fc, edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize,
            color=tc, weight=weight)


def _arrow(ax, x1, y1, x2, y2, *, color=C_DARK, lw=1.4, style="->",
           rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=12,
        color=color, lw=lw,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1: Multi-cloud architecture
# ---------------------------------------------------------------------------
def fig1_multi_cloud_architecture() -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 10)
    ax.axis("off")

    # Users globe
    _box(ax, 5.0, 9.0, 3.0, 0.9, "Global Users",
         fc=C_DARK, fontsize=10.5, weight="bold", radius=0.05)

    # Global traffic manager (DNS / GSLB / CDN)
    _box(ax, 4.2, 7.5, 4.6, 1.1,
         "Global Traffic Manager  (Anycast DNS  +  GSLB  +  CDN)",
         fc=C_AMBER, fontsize=10, weight="bold", radius=0.05)
    _arrow(ax, 6.5, 9.0, 6.5, 8.6, color=C_GRAY, lw=1.2)

    # Three clouds
    cloud_specs = [
        ("AWS",   "us-east-1",  0.4, C_AMBER, ["EC2 / EKS", "RDS Postgres", "S3 + CloudFront"]),
        ("GCP",   "europe-w1",  4.7, C_BLUE,  ["GKE", "BigQuery", "Vertex AI / ML"]),
        ("Azure", "japan-east", 9.0, C_PURPLE, ["AKS", "Cosmos DB", "AD / M365"]),
    ]
    for name, region, x, col, services in cloud_specs:
        _box(ax, x, 3.4, 3.6, 3.6, "", fc="white", ec=col,
             tc=C_DARK, radius=0.04)
        ax.text(x + 1.8, 6.65, name,
                ha="center", fontsize=12, weight="bold", color=col)
        ax.text(x + 1.8, 6.25, region,
                ha="center", fontsize=8.5, color=C_GRAY, style="italic")
        for i, s in enumerate(services):
            _box(ax, x + 0.3, 5.5 - i * 0.7, 3.0, 0.55, s,
                 fc=col, fontsize=9, weight="bold", radius=0.03)
        # GTM -> cloud arrows
        _arrow(ax, 6.5, 7.5, x + 1.8, 7.0,
               color=C_GRAY, lw=1.0, rad=0.05, style="->")

    # Inter-cloud peering
    _arrow(ax, 4.0, 5.2, 4.7, 5.2, color=C_GREEN, lw=1.6, style="<->")
    _arrow(ax, 8.3, 5.2, 9.0, 5.2, color=C_GREEN, lw=1.6, style="<->")
    ax.text(4.35, 5.45, "peer", fontsize=8, color=C_GREEN, weight="bold")
    ax.text(8.65, 5.45, "peer", fontsize=8, color=C_GREEN, weight="bold")

    # Shared horizontal layers (identity + observability)
    _box(ax, 0.4, 1.7, 12.2, 1.0,
         "Federated Identity  (Okta / Azure AD / Workload Identity Federation)",
         fc=C_BLUE, fontsize=10, weight="bold", radius=0.04)
    _box(ax, 0.4, 0.4, 12.2, 1.0,
         "Unified Observability  (OpenTelemetry  +  Prometheus  +  Grafana  +  central SIEM)",
         fc=C_GREEN, fontsize=10, weight="bold", radius=0.04)

    for x in [2.2, 6.5, 10.8]:
        _arrow(ax, x, 3.4, x, 2.7, color=C_GRAY, lw=1.0)

    ax.set_title("Multi-Cloud Architecture: Best-of-breed Services Behind One Front Door",
                 fontsize=14, weight="bold", color=C_DARK, pad=10, loc="left")

    fig.tight_layout()
    _save(fig, "fig1_multi_cloud_architecture")


# ---------------------------------------------------------------------------
# Figure 2: Hybrid connectivity (VPN / Direct Connect / SD-WAN)
# ---------------------------------------------------------------------------
def fig2_hybrid_connectivity() -> None:
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.25)

    # ----- Left: topology -----
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 12); ax.set_ylim(0, 10)
    ax.axis("off")

    # On-prem DC
    _box(ax, 0.3, 3.7, 2.8, 3.6, "", fc="#f1f5f9", ec=C_GRAY,
         tc=C_DARK, radius=0.05)
    ax.text(1.7, 7.0, "On-Prem DC", ha="center", fontsize=11,
            weight="bold", color=C_DARK)
    for i, s in enumerate(["Mainframe", "Legacy ERP", "File Server"]):
        _box(ax, 0.55, 5.5 - i * 0.95, 2.3, 0.7, s,
             fc=C_DARK, fontsize=9, weight="bold", radius=0.04)

    # Cloud
    _box(ax, 8.7, 3.7, 3.0, 3.6, "", fc="#eff6ff", ec=C_BLUE,
         tc=C_DARK, radius=0.05)
    ax.text(10.2, 7.0, "Public Cloud VPC", ha="center", fontsize=11,
            weight="bold", color=C_BLUE)
    for i, s in enumerate(["EKS / AKS / GKE", "Managed DB", "Object Storage"]):
        _box(ax, 8.95, 5.5 - i * 0.95, 2.5, 0.7, s,
             fc=C_BLUE, fontsize=9, weight="bold", radius=0.04)

    # Three connectivity paths
    paths = [
        (8.0, "Site-to-Site VPN",          C_GRAY,   "via internet  /  IPsec"),
        (5.5, "Direct Connect / ExpressRoute", C_BLUE, "private fiber  /  dedicated"),
        (3.0, "SD-WAN",                    C_PURPLE, "policy-driven overlay"),
    ]
    for y, label, col, sub in paths:
        # curve via control-point mid
        _arrow(ax, 3.1, y, 8.7, y, color=col, lw=2.0, style="<->", rad=0.0)
        _box(ax, 4.7, y - 0.45, 2.6, 0.9, label,
             fc=col, fontsize=9.5, weight="bold", radius=0.06)
        ax.text(6.0, y - 0.7, sub, ha="center", fontsize=8,
                color=col, style="italic")

    ax.set_title("Hybrid Connectivity: Three Paths from DC to Cloud",
                 fontsize=13, weight="bold", color=C_DARK, pad=8, loc="left")

    # ----- Right: bandwidth vs latency scatter -----
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("Bandwidth vs Latency Trade-off",
                  fontsize=12, weight="bold", color=C_DARK, pad=8)

    options = [
        ("Site-to-site VPN",       0.5,   25, 250, C_GRAY,  "low"),
        ("SD-WAN",                 1.0,   15, 700, C_PURPLE, "med"),
        ("Direct Connect 1 Gbps",  1.0,    5, 900, C_BLUE,  "high"),
        ("Direct Connect 10 Gbps", 10.0,   3, 1500, C_BLUE, "high"),
        ("Direct Connect 100 Gbps", 100.0, 2, 2400, C_BLUE, "premium"),
    ]
    for label, bw, lat, sz, col, cost in options:
        ax2.scatter([bw], [lat], s=sz, color=col, edgecolor=C_DARK,
                    linewidth=1.2, alpha=0.85, zorder=3)
        ax2.annotate(f"{label}\n[{cost}]",
                     xy=(bw, lat), xytext=(8, 8),
                     textcoords="offset points",
                     fontsize=8.5, color=C_DARK)

    ax2.set_xscale("log")
    ax2.set_xlabel("Bandwidth  (Gbps, log scale)", fontsize=10)
    ax2.set_ylabel("Round-trip latency  (ms)", fontsize=10)
    ax2.set_xlim(0.2, 300)
    ax2.set_ylim(0, 32)
    ax2.grid(True, which="both", alpha=0.4)
    ax2.text(0.95, 0.04, "marker size = monthly cost",
             transform=ax2.transAxes, fontsize=8.5, color=C_GRAY,
             ha="right", style="italic")

    fig.suptitle("Hybrid Cloud Networking: Pick the Right Pipe",
                 fontsize=14.5, weight="bold", color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig2_hybrid_connectivity")


# ---------------------------------------------------------------------------
# Figure 3: Vendor lock-in mitigation radar
# ---------------------------------------------------------------------------
def fig3_lockin_mitigation() -> None:
    dims = ["Data\nportability", "API\nabstraction", "Architecture\nportability",
            "Skill\ndiversity", "Contract\nflexibility"]
    # Scores (0-5): higher = less locked in.
    baseline = [1.5, 1.0, 1.0, 1.0, 1.5]      # provider-native lock-in
    portable = [4.5, 4.0, 4.5, 4.0, 3.5]      # mitigated

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    baseline_p = baseline + baseline[:1]
    portable_p = portable + portable[:1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8),
                             subplot_kw=dict(polar=True))

    for ax in axes:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["1", "2", "3", "4", "5"], fontsize=8, color=C_GRAY)
        ax.set_ylim(0, 5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dims, fontsize=10, color=C_DARK)
        ax.grid(alpha=0.4)

    axes[0].plot(angles, baseline_p, color=C_AMBER, lw=2.0)
    axes[0].fill(angles, baseline_p, color=C_AMBER, alpha=0.30)
    axes[0].set_title("Baseline\n(provider-native build)",
                      fontsize=12.5, weight="bold", color=C_AMBER, pad=18)

    axes[1].plot(angles, baseline_p, color=C_AMBER, lw=1.4,
                 linestyle="--", alpha=0.7, label="Baseline")
    axes[1].fill(angles, baseline_p, color=C_AMBER, alpha=0.10)
    axes[1].plot(angles, portable_p, color=C_GREEN, lw=2.2,
                 label="Portable")
    axes[1].fill(angles, portable_p, color=C_GREEN, alpha=0.30)
    axes[1].set_title("Mitigated\n(Kubernetes + Terraform + open standards)",
                      fontsize=12.5, weight="bold", color=C_GREEN, pad=18)
    axes[1].legend(loc="lower right", bbox_to_anchor=(1.18, -0.05),
                   fontsize=9, frameon=True)

    fig.suptitle("Vendor Lock-in Mitigation: Five Dimensions, Two Postures",
                 fontsize=14.5, weight="bold", color=C_DARK, y=1.02)

    # Footer with concrete tactics
    tactics = ("  Data: standard formats (Parquet, OpenAPI) + tested exports     "
               "API: abstraction layers (CrossPlane, Terraform)\n"
               "  Arch: Kubernetes + service mesh + open observability       "
               "Skills: cross-train teams        Contract: short terms + exit clauses")
    fig.text(0.5, -0.03, tactics, ha="center", fontsize=9,
             color=C_DARK, family="monospace")

    fig.tight_layout()
    _save(fig, "fig3_lockin_mitigation")


# ---------------------------------------------------------------------------
# Figure 4: Cross-cloud data sync
# ---------------------------------------------------------------------------
def fig4_cross_cloud_data_sync() -> None:
    fig, ax = plt.subplots(figsize=(13, 7.8))
    ax.set_xlim(0, 13); ax.set_ylim(0, 10)
    ax.axis("off")

    # Three regions on three clouds
    regions = [
        ("AWS  us-east-1",    1.5,  C_AMBER),
        ("GCP  europe-west1", 6.5,  C_BLUE),
        ("Azure  japan-east", 11.5, C_PURPLE),
    ]
    for name, x, col in regions:
        _box(ax, x - 1.5, 5.4, 3.0, 3.4, "", fc="white", ec=col,
             tc=C_DARK, radius=0.05)
        ax.text(x, 8.5, name, ha="center", fontsize=10.5,
                weight="bold", color=col)
        # Primary DB
        _box(ax, x - 1.3, 7.0, 2.6, 0.9, "Primary DB",
             fc=col, fontsize=9.5, weight="bold", radius=0.04)
        # Read replica + cache
        _box(ax, x - 1.3, 6.0, 2.6, 0.7, "Read replicas + cache",
             fc="white", ec=col, tc=col, fontsize=9, weight="bold",
             radius=0.04)
        # CDC stream out
        _box(ax, x - 1.3, 5.5, 2.6, 0.4, "CDC log (Debezium)",
             fc=C_GREEN, fontsize=8, weight="bold", radius=0.03)

    # Sync arrows (sync, async, CDC stream)
    _arrow(ax, 3.0, 7.4, 5.0, 7.4, color=C_BLUE, lw=2.0, style="<->",
           rad=-0.18)
    ax.text(4.0, 8.0, "synchronous\n(zero RPO, +latency)",
            ha="center", fontsize=8.5, color=C_BLUE, style="italic")

    _arrow(ax, 8.0, 7.4, 10.0, 7.4, color=C_AMBER, lw=2.0, style="->",
           rad=-0.18)
    ax.text(9.0, 8.0, "async replication\n(seconds RPO)",
            ha="center", fontsize=8.5, color=C_AMBER, style="italic")

    # CDC stream bus
    _box(ax, 1.0, 3.4, 11.0, 0.9,
         "Kafka  /  Pub/Sub  /  Event Hubs   <-- Change Data Capture stream -->",
         fc=C_GREEN, fontsize=10, weight="bold", radius=0.04)
    for x in [1.5, 6.5, 11.5]:
        _arrow(ax, x, 5.5, x, 4.3, color=C_GREEN, lw=1.4, style="<->")

    # Conflict resolution box
    _box(ax, 0.5, 1.4, 12.0, 1.4, "", fc="#fff7ed", ec=C_AMBER,
         tc=C_DARK, radius=0.04)
    ax.text(6.5, 2.55, "Conflict Resolution Strategies",
            ha="center", fontsize=10.5, weight="bold", color=C_AMBER)
    strategies = [
        "Last-write-wins (LWW)\nsimple, may lose data",
        "Vector clocks / CRDTs\ndetect causality",
        "App-level merge\nbusiness rules win",
    ]
    for i, s in enumerate(strategies):
        ax.text(2.2 + i * 4.2, 1.85, s, ha="center", va="center",
                fontsize=9, color=C_DARK)

    # Footer key insight
    _box(ax, 0.5, 0.2, 12.0, 0.85,
         "Pick per-table, not per-system: critical writes go sync;"
         "  analytics-friendly tables go CDC + async.",
         fc=C_DARK, fontsize=9.5, weight="bold", radius=0.04)

    ax.set_title("Cross-Cloud Data Synchronization Topology",
                 fontsize=14, weight="bold", color=C_DARK, pad=10, loc="left")

    fig.tight_layout()
    _save(fig, "fig4_cross_cloud_data_sync")


# ---------------------------------------------------------------------------
# Figure 5: Cost optimization across clouds
# ---------------------------------------------------------------------------
def fig5_cost_optimization() -> None:
    fig = plt.figure(figsize=(13, 7.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.28)

    # Left: cumulative monthly spend over 12 months
    ax = fig.add_subplot(gs[0])
    months = np.arange(1, 13)
    on_demand = np.full_like(months, 100, dtype=float)
    reserved = np.full_like(months, 60, dtype=float)
    spot_mix = np.full_like(months, 45, dtype=float)
    optimized = np.array([100, 92, 80, 70, 62, 55, 50, 45, 42, 39, 37, 35],
                         dtype=float)

    ax.plot(months, on_demand, color=C_AMBER, lw=2.2, marker="o",
            markersize=5, label="On-demand baseline")
    ax.plot(months, reserved, color=C_BLUE, lw=2.2, marker="s",
            markersize=5, label="+ Reserved / Committed-Use")
    ax.plot(months, spot_mix, color=C_PURPLE, lw=2.2, marker="^",
            markersize=5, label="+ Spot for fault-tolerant workloads")
    ax.plot(months, optimized, color=C_GREEN, lw=2.6, marker="D",
            markersize=6, label="+ Right-sizing + FinOps loop")

    ax.fill_between(months, optimized, on_demand, color=C_GREEN, alpha=0.10)
    ax.annotate("65% savings\nat steady state",
                xy=(12, 35), xytext=(8.5, 78),
                fontsize=10, color=C_GREEN, weight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.5))

    ax.set_xlabel("Month", fontsize=10.5)
    ax.set_ylabel("Monthly cost  (% of baseline)", fontsize=10.5)
    ax.set_title("Cumulative Cost Optimization Curve",
                 fontsize=12.5, weight="bold", color=C_DARK)
    ax.set_xticks(months)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(alpha=0.4)

    # Right: cost drivers stack (per-cloud)
    ax2 = fig.add_subplot(gs[1])
    clouds = ["AWS", "GCP", "Azure"]
    compute = np.array([45, 42, 48])
    storage = np.array([18, 22, 20])
    network = np.array([20, 25, 17])
    other = np.array([17, 11, 15])

    x = np.arange(len(clouds))
    ax2.bar(x, compute, color=C_BLUE,   label="Compute")
    ax2.bar(x, storage, bottom=compute, color=C_GREEN,  label="Storage")
    ax2.bar(x, network, bottom=compute + storage, color=C_AMBER, label="Egress / Network")
    ax2.bar(x, other,   bottom=compute + storage + network,
            color=C_PURPLE, label="Managed services")

    for i, c in enumerate(clouds):
        total = compute[i] + storage[i] + network[i] + other[i]
        ax2.text(i, total + 1.5, f"{total}k USD", ha="center",
                 fontsize=9.5, weight="bold", color=C_DARK)

    ax2.set_xticks(x); ax2.set_xticklabels(clouds, fontsize=10.5)
    ax2.set_ylabel("Monthly spend  (USD k)", fontsize=10.5)
    ax2.set_ylim(0, 130)
    ax2.set_title("Cost Drivers per Cloud",
                  fontsize=12.5, weight="bold", color=C_DARK)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10),
               ncol=2, fontsize=8.8, frameon=False)

    # Egress callout
    ax2.annotate("Egress is\nthe sneaky one",
                 xy=(2, 85), xytext=(0.4, 116),
                 fontsize=9, color=C_AMBER, weight="bold",
                 arrowprops=dict(arrowstyle="->", color=C_AMBER, lw=1.3))

    fig.suptitle("Cost Optimization Across Clouds: Compounding Levers",
                 fontsize=14.5, weight="bold", color=C_DARK, y=1.00)
    fig.tight_layout()
    _save(fig, "fig5_cost_optimization")


# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Cloud Computing Part 08 figures...")
    fig1_multi_cloud_architecture()
    fig2_hybrid_connectivity()
    fig3_lockin_mitigation()
    fig4_cross_cloud_data_sync()
    fig5_cost_optimization()
    print("Done.")


if __name__ == "__main__":
    main()
