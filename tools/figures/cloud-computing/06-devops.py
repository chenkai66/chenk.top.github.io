"""
Figures for Cloud Computing Chapter 06: Operations and DevOps.

Style: clean whitegrid, color palette
  Blue   #2563eb
  Purple #7c3aed
  Green  #10b981
  Amber  #f59e0b

Generates 5 figures and saves them to BOTH the EN and ZH asset folders so the
two language versions stay in sync.

Figures:
    fig1_cicd_pipeline           CI/CD pipeline stages from commit to
                                 production with quality gates and
                                 rollback paths.
    fig2_iac_terraform           Terraform workflow: HCL -> plan -> apply,
                                 state file, and provider abstraction over
                                 multiple clouds.
    fig3_monitoring_stack        Prometheus + Grafana + Alertmanager
                                 architecture: scraping, storage, query,
                                 dashboards, alerting.
    fig4_logging_architecture    EFK / ELK pipeline: app -> shipper ->
                                 buffer -> processor -> store -> dashboard.
    fig5_error_budget            SRE error budget over time, with burn
                                 rate, SLO threshold and remediation
                                 zones.

Usage:
    python3 scripts/figures/cloud-computing/06-devops.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

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
C_BLUE = COLORS["primary"]
C_PURPLE = COLORS["accent"]
C_GREEN = COLORS["success"]
C_AMBER = COLORS["warning"]
C_RED = COLORS["danger"]
C_GRAY = COLORS["muted"]
C_DARK = COLORS["text"]
C_LIGHT = COLORS["grid"]
C_BG = "#f8fafc"

DPI = 150

REPO_ROOT = Path(__file__).resolve().parents[3]
EN_DIR = REPO_ROOT / "source" / "_posts" / "en" / "cloud-computing" / "operations-devops"
ZH_DIR = REPO_ROOT / "source" / "_posts" / "zh" / "cloud-computing" / "operations-devops"


def _save(fig: plt.Figure, name: str) -> None:
    EN_DIR.mkdir(parents=True, exist_ok=True)
    ZH_DIR.mkdir(parents=True, exist_ok=True)
    for d in (EN_DIR, ZH_DIR):
        out = d / f"{name}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor="white")
        print(f"  -> {out}")
    plt.close(fig)


def _box(
    ax,
    x: float, y: float, w: float, h: float, text: str,
    *,
    fc: str = "#ffffff", ec: str = C_DARK,
    text_color: str | None = None,
    fontsize: float = 10.0, weight: str = "normal",
    radius: float = 0.04, lw: float = 1.4, zorder: int = 3,
) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, color=text_color or C_DARK, weight=weight,
        zorder=zorder + 1,
    )


def _arrow(
    ax, start, end,
    *,
    color: str = C_GRAY, lw: float = 1.6,
    style: str = "-|>", mutation_scale: float = 14,
    zorder: int = 2, connectionstyle: str = "arc3,rad=0",
) -> None:
    a = FancyArrowPatch(
        start, end,
        arrowstyle=style, mutation_scale=mutation_scale,
        color=color, linewidth=lw, zorder=zorder,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(a)


# ---------------------------------------------------------------------------
# Figure 1: CI/CD pipeline
# ---------------------------------------------------------------------------
def fig1_cicd_pipeline() -> None:
    """Linear pipeline with quality gates and a rollback path."""
    fig, ax = plt.subplots(figsize=(12, 5.6))

    stages = [
        ("Commit", C_BLUE, "git push"),
        ("Build", C_BLUE, "docker build"),
        ("Unit Tests", C_GREEN, "pytest"),
        ("Security Scan", C_AMBER, "trivy / SAST"),
        ("Deploy Staging", C_PURPLE, "k8s apply"),
        ("Smoke Tests", C_GREEN, "e2e suite"),
        ("Deploy Prod", C_PURPLE, "canary -> full"),
        ("Verify", C_GREEN, "SLO check"),
    ]

    box_w, box_h = 1.32, 0.95
    gap = 0.18
    x0, y_main = 0.3, 3.2

    centers = []
    for i, (name, color, sub) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        fc = {
            C_BLUE: "#dbeafe", C_GREEN: "#d1fae5",
            C_AMBER: "#fef3c7", C_PURPLE: "#ede9fe",
        }[color]
        _box(ax, x, y_main, box_w, box_h, name,
             fc=fc, ec=color, fontsize=10.5, weight="bold", text_color=color)
        ax.text(x + box_w / 2, y_main - 0.2, sub,
                ha="center", va="top", fontsize=8.5, color=C_DARK, style="italic")
        centers.append((x + box_w / 2, y_main + box_h / 2))

    # Arrows between stages
    for i in range(len(stages) - 1):
        x1 = x0 + i * (box_w + gap) + box_w
        x2 = x0 + (i + 1) * (box_w + gap)
        _arrow(ax, (x1, y_main + box_h / 2), (x2, y_main + box_h / 2),
               color=C_DARK, lw=1.6, mutation_scale=14)

    # Quality gate annotations on certain transitions
    gates = {2: "fail -> stop", 3: "vulns -> stop", 5: "fail -> rollback", 7: "regression -> rollback"}
    for idx, label in gates.items():
        x = x0 + idx * (box_w + gap) + box_w + (gap / 2)
        ax.text(x, y_main + box_h + 0.18, label,
                ha="center", va="bottom", fontsize=8.5, color=C_RED, style="italic")

    # Rollback path (curved arrow from Verify back to Deploy Staging)
    start = centers[-1]
    end = centers[4]
    _arrow(ax, (start[0], y_main), (end[0], y_main),
           color=C_RED, lw=1.6, mutation_scale=16,
           connectionstyle="arc3,rad=-0.35")
    ax.text((start[0] + end[0]) / 2, y_main - 1.05, "rollback path (auto on SLO breach)",
            ha="center", va="center", fontsize=9.5, color=C_RED, style="italic", weight="semibold")

    # Top section: source / artifact / deploy lanes
    _box(ax, 0.1, 4.7, (box_w + gap) * 2 - gap, 0.45,
         "SOURCE",
         fc=C_BG, ec=C_BLUE, text_color=C_BLUE, fontsize=10, weight="bold", lw=1.4)
    _box(ax, 0.1 + (box_w + gap) * 2, 4.7, (box_w + gap) * 2 - gap, 0.45,
         "BUILD & TEST",
         fc=C_BG, ec=C_GREEN, text_color=C_GREEN, fontsize=10, weight="bold", lw=1.4)
    _box(ax, 0.1 + (box_w + gap) * 4, 4.7, (box_w + gap) * 2 - gap, 0.45,
         "STAGING",
         fc=C_BG, ec=C_PURPLE, text_color=C_PURPLE, fontsize=10, weight="bold", lw=1.4)
    _box(ax, 0.1 + (box_w + gap) * 6, 4.7, (box_w + gap) * 2 - gap, 0.45,
         "PRODUCTION",
         fc=C_BG, ec=C_AMBER, text_color="#92400e", fontsize=10, weight="bold", lw=1.4)

    # Bottom feedback annotation
    ax.text(
        (x0 + len(stages) * (box_w + gap) / 2),
        0.7,
        "Each stage emits metrics, logs and traces. The pipeline is the system of record for every release.",
        ha="center", va="center",
        fontsize=10, color=C_DARK, style="italic",
        bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT, boxstyle="round,pad=0.4"),
    )

    ax.set_xlim(0, x0 + len(stages) * (box_w + gap) + 0.2)
    ax.set_ylim(0.2, 5.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("CI/CD Pipeline: From Commit to Production", fontsize=14, weight="bold", color=C_DARK, pad=14)
    ax.grid(False)

    _save(fig, "fig1_cicd_pipeline")


# ---------------------------------------------------------------------------
# Figure 2: Infrastructure as Code with Terraform
# ---------------------------------------------------------------------------
def fig2_iac_terraform() -> None:
    """HCL -> plan -> apply, with state and provider abstraction."""
    fig, ax = plt.subplots(figsize=(12, 6.4))

    # Top row: source HCL files
    src_files = ["main.tf", "variables.tf", "outputs.tf", "modules/"]
    src_x0, src_y, src_w, src_h, src_gap = 0.3, 5.0, 1.5, 0.7, 0.15
    for i, f in enumerate(src_files):
        x = src_x0 + i * (src_w + src_gap)
        _box(ax, x, src_y, src_w, src_h, f,
             fc="#dbeafe", ec=C_BLUE, fontsize=10.5, weight="semibold", text_color=C_BLUE)
    ax.text(src_x0 + (4 * (src_w + src_gap)) / 2 - src_gap / 2, src_y + src_h + 0.18,
            "HCL source (declarative desired state)",
            ha="center", va="bottom", fontsize=10, color=C_DARK, weight="semibold")

    # Middle: terraform CLI workflow
    cli_y = 3.3
    cli_w, cli_h = 1.5, 0.85
    cli_gap = 0.4
    cli_x0 = 0.5
    cli_steps = [
        ("init", "downloads\nproviders", C_GRAY),
        ("plan", "diff vs state\nshow changes", C_AMBER),
        ("apply", "execute against\ncloud APIs", C_GREEN),
        ("destroy", "tear down\nall resources", C_RED),
    ]
    cli_centers = []
    for i, (name, sub, c) in enumerate(cli_steps):
        x = cli_x0 + i * (cli_w + cli_gap)
        fc = {C_GRAY: "#f1f5f9", C_AMBER: "#fef3c7", C_GREEN: "#d1fae5", C_RED: "#fee2e2"}[c]
        _box(ax, x, cli_y, cli_w, cli_h, "",
             fc=fc, ec=c, lw=1.6)
        ax.text(x + cli_w / 2, cli_y + cli_h - 0.2, f"terraform {name}",
                ha="center", va="center", fontsize=10.5, weight="bold", color=c)
        ax.text(x + cli_w / 2, cli_y + 0.22, sub,
                ha="center", va="center", fontsize=8.5, color=C_DARK, style="italic")
        cli_centers.append((x + cli_w / 2, cli_y + cli_h / 2))
        if i < len(cli_steps) - 1:
            _arrow(ax, (x + cli_w, cli_y + cli_h / 2),
                   (x + cli_w + cli_gap, cli_y + cli_h / 2),
                   color=C_DARK, lw=1.4, mutation_scale=14)

    # Source files -> CLI
    for i in range(4):
        x = src_x0 + i * (src_w + src_gap) + src_w / 2
        _arrow(ax, (x, src_y), (cli_centers[1][0], cli_y + cli_h),
               color=C_GRAY, lw=1.0, mutation_scale=10,
               connectionstyle="arc3,rad=-0.05")

    # State file (right of CLI)
    state_x, state_y = cli_x0 + 4 * (cli_w + cli_gap) + 0.1, cli_y - 0.05
    state_w, state_h = 1.8, 1.0
    _box(ax, state_x, state_y, state_w, state_h, "",
         fc="#ede9fe", ec=C_PURPLE, lw=1.8)
    ax.text(state_x + state_w / 2, state_y + state_h - 0.2, "terraform.tfstate",
            ha="center", va="center", fontsize=10.5, weight="bold", color=C_PURPLE)
    ax.text(state_x + state_w / 2, state_y + state_h / 2 - 0.05, "remote backend\n(S3 + DynamoDB lock)",
            ha="center", va="center", fontsize=8.5, color=C_DARK, style="italic")
    # bidirectional arrow plan/apply <-> state
    _arrow(ax, (cli_centers[1][0] + cli_w / 2, cli_y + cli_h / 2),
           (state_x, state_y + state_h / 2),
           color=C_PURPLE, lw=1.4, style="<|-|>", mutation_scale=14,
           connectionstyle="arc3,rad=-0.15")

    # Providers row
    prov_y, prov_w, prov_h = 1.4, 1.7, 0.85
    prov_gap = 0.35
    prov_x0 = 0.5
    providers = [
        ("AWS", C_AMBER),
        ("GCP", C_BLUE),
        ("Azure", C_BLUE),
        ("Kubernetes", C_GREEN),
        ("Datadog", C_PURPLE),
    ]
    for i, (name, c) in enumerate(providers):
        x = prov_x0 + i * (prov_w + prov_gap)
        fc = {C_AMBER: "#fef3c7", C_BLUE: "#dbeafe",
              C_GREEN: "#d1fae5", C_PURPLE: "#ede9fe"}.get(c, "white")
        _box(ax, x, prov_y, prov_w, prov_h, name,
             fc=fc, ec=c, fontsize=11, weight="bold", text_color=c)
        # apply -> provider arrow
        _arrow(ax, (cli_centers[2][0], cli_y),
               (x + prov_w / 2, prov_y + prov_h),
               color=C_GREEN, lw=1.0, mutation_scale=10,
               connectionstyle="arc3,rad=0.12")
    ax.text(prov_x0 + (5 * (prov_w + prov_gap)) / 2 - prov_gap / 2, prov_y - 0.18,
            "Cloud APIs (provider plugins translate HCL into REST/SDK calls)",
            ha="center", va="top", fontsize=10, color=C_DARK, style="italic")

    # Bottom annotation
    ax.text(6.0, 0.45,
            "One declarative codebase reconciled against state -> reproducible infrastructure across environments and clouds.",
            ha="center", va="center",
            fontsize=10, color=C_DARK, style="italic",
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT, boxstyle="round,pad=0.4"))

    ax.set_xlim(0, 12)
    ax.set_ylim(0.0, 6.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Infrastructure as Code: Terraform Workflow", fontsize=14, weight="bold", color=C_DARK, pad=14)
    ax.grid(False)

    _save(fig, "fig2_iac_terraform")


# ---------------------------------------------------------------------------
# Figure 3: Prometheus + Grafana monitoring stack
# ---------------------------------------------------------------------------
def fig3_monitoring_stack() -> None:
    """Pull-based scraping, TSDB, query, dashboard, alerting."""
    fig, ax = plt.subplots(figsize=(12, 6.6))

    # Left: scrape targets
    targets = [
        ("App\n(/metrics)", C_BLUE, 5.0),
        ("Node Exporter", C_BLUE, 3.8),
        ("kube-state-metrics", C_BLUE, 2.6),
        ("cAdvisor", C_BLUE, 1.4),
    ]
    for name, c, y in targets:
        _box(ax, 0.3, y, 2.0, 0.85, name,
             fc="#dbeafe", ec=c, fontsize=10, weight="semibold")

    # Centre-left: Prometheus server
    prom_x, prom_y, prom_w, prom_h = 3.3, 2.2, 2.6, 3.3
    _box(ax, prom_x, prom_y, prom_w, prom_h, "",
         fc="#fef3c7", ec=C_AMBER, lw=2.2)
    ax.text(prom_x + prom_w / 2, prom_y + prom_h - 0.3,
            "Prometheus Server",
            ha="center", va="center", fontsize=12, weight="bold", color="#92400e")
    parts = [
        "* scraper (pull every 15s)",
        "* service discovery",
        "* TSDB (local + remote)",
        "* PromQL query engine",
        "* rule evaluation",
    ]
    for k, p in enumerate(parts):
        ax.text(prom_x + 0.18, prom_y + prom_h - 0.85 - k * 0.42, p,
                ha="left", va="center", fontsize=9.5, color=C_DARK)

    # Targets -> Prometheus (pull arrows)
    for _, _, y in targets:
        _arrow(ax, (prom_x, prom_y + prom_h / 2),
               (2.3, y + 0.42),
               color=C_AMBER, lw=1.3, style="<|-",
               connectionstyle="arc3,rad=0.0", mutation_scale=12)
    ax.text(2.8, 5.65, "scrape (pull)",
            ha="center", va="bottom", fontsize=9, color=C_AMBER, style="italic", weight="semibold")

    # Right top: Grafana
    graf_x, graf_y, graf_w, graf_h = 6.6, 4.4, 2.4, 1.3
    _box(ax, graf_x, graf_y, graf_w, graf_h, "",
         fc="#d1fae5", ec=C_GREEN, lw=1.8)
    ax.text(graf_x + graf_w / 2, graf_y + graf_h - 0.3, "Grafana",
            ha="center", va="center", fontsize=11.5, weight="bold", color=C_GREEN)
    ax.text(graf_x + graf_w / 2, graf_y + 0.3, "dashboards & ad-hoc queries",
            ha="center", va="center", fontsize=9, color=C_DARK, style="italic")
    _arrow(ax, (prom_x + prom_w, prom_y + prom_h * 0.75),
           (graf_x, graf_y + graf_h / 2),
           color=C_GREEN, lw=1.4, mutation_scale=14)
    ax.text((prom_x + prom_w + graf_x) / 2, (prom_y + prom_h * 0.75 + graf_y + graf_h / 2) / 2 + 0.18,
            "PromQL", ha="center", va="bottom", fontsize=9, color=C_GREEN, style="italic", weight="semibold")

    # Right middle: Alertmanager
    am_x, am_y, am_w, am_h = 6.6, 2.6, 2.4, 1.3
    _box(ax, am_x, am_y, am_w, am_h, "",
         fc="#fee2e2", ec=C_RED, lw=1.8)
    ax.text(am_x + am_w / 2, am_y + am_h - 0.3, "Alertmanager",
            ha="center", va="center", fontsize=11.5, weight="bold", color=C_RED)
    ax.text(am_x + am_w / 2, am_y + 0.3, "dedupe / group / route",
            ha="center", va="center", fontsize=9, color=C_DARK, style="italic")
    _arrow(ax, (prom_x + prom_w, prom_y + prom_h * 0.25),
           (am_x, am_y + am_h / 2),
           color=C_RED, lw=1.4, mutation_scale=14)
    ax.text((prom_x + prom_w + am_x) / 2, (prom_y + prom_h * 0.25 + am_y + am_h / 2) / 2 - 0.25,
            "fired alerts", ha="center", va="top", fontsize=9, color=C_RED, style="italic", weight="semibold")

    # Far right: humans / channels
    sinks = [
        ("Engineer", C_BLUE, 5.4),
        ("Slack / PagerDuty", C_RED, 3.85),
        ("Email", C_GRAY, 2.3),
    ]
    for name, c, y in sinks:
        fc = {C_BLUE: "#dbeafe", C_RED: "#fee2e2"}.get(c, "#f1f5f9")
        _box(ax, 9.7, y, 1.9, 0.7, name,
             fc=fc, ec=c, fontsize=10, weight="semibold")
    _arrow(ax, (graf_x + graf_w, graf_y + graf_h / 2),
           (9.7, 5.75),
           color=C_BLUE, lw=1.3, mutation_scale=12)
    _arrow(ax, (am_x + am_w, am_y + am_h * 0.7),
           (9.7, 4.2),
           color=C_RED, lw=1.3, mutation_scale=12)
    _arrow(ax, (am_x + am_w, am_y + am_h * 0.3),
           (9.7, 2.65),
           color=C_GRAY, lw=1.3, mutation_scale=12)

    # Bottom: long-term storage option
    ax.text(prom_x + prom_w / 2, 1.4, "remote_write -> Thanos / Cortex / Mimir (long-term, multi-cluster)",
            ha="center", va="center", fontsize=9.5, color=C_DARK, style="italic",
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT, boxstyle="round,pad=0.3"))

    # Layer label
    ax.text(0.4, 6.05, "Targets (export /metrics)",
            ha="left", va="bottom", fontsize=9.5, color=C_DARK, weight="semibold")

    ax.set_xlim(0, 12)
    ax.set_ylim(0.7, 6.4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Monitoring Stack: Prometheus + Grafana + Alertmanager", fontsize=14, weight="bold", color=C_DARK, pad=14)
    ax.grid(False)

    _save(fig, "fig3_monitoring_stack")


# ---------------------------------------------------------------------------
# Figure 4: Logging architecture (EFK / ELK)
# ---------------------------------------------------------------------------
def fig4_logging_architecture() -> None:
    """End-to-end log pipeline from app to dashboard."""
    fig, ax = plt.subplots(figsize=(12.5, 6.2))

    # Stages of pipeline
    stages = [
        ("Application\nstdout / files", C_BLUE,
         "structured JSON\nrequest_id, level"),
        ("Shipper\n(Filebeat / Fluent Bit)", C_BLUE,
         "tail + tag +\nbasic enrich"),
        ("Buffer\n(Kafka / Redis)", C_AMBER,
         "decouple, absorb\nbursts, retry"),
        ("Processor\n(Logstash / Fluentd)", C_PURPLE,
         "parse, filter,\nenrich, route"),
        ("Store\n(Elasticsearch / OpenSearch)", C_GREEN,
         "inverted index,\nhot-warm-cold"),
        ("Dashboard\n(Kibana / Grafana)", C_GREEN,
         "search, viz,\nsaved queries"),
    ]
    box_w, box_h = 1.7, 1.3
    gap = 0.35
    x0, y_main = 0.3, 3.0

    for i, (name, color, sub) in enumerate(stages):
        x = x0 + i * (box_w + gap)
        fc = {
            C_BLUE: "#dbeafe", C_GREEN: "#d1fae5",
            C_AMBER: "#fef3c7", C_PURPLE: "#ede9fe",
        }[color]
        _box(ax, x, y_main, box_w, box_h, "",
             fc=fc, ec=color, fontsize=10, lw=1.6)
        ax.text(x + box_w / 2, y_main + box_h - 0.32, name,
                ha="center", va="center", fontsize=10, weight="bold", color=color)
        ax.text(x + box_w / 2, y_main + 0.32, sub,
                ha="center", va="center", fontsize=8.5, color=C_DARK, style="italic")
        if i < len(stages) - 1:
            x_arrow_start = x + box_w
            x_arrow_end = x + box_w + gap
            _arrow(ax, (x_arrow_start, y_main + box_h / 2),
                   (x_arrow_end, y_main + box_h / 2),
                   color=C_DARK, lw=1.6, mutation_scale=14)

    # Sidecar pattern annotation (above shipper)
    ax.annotate("often a sidecar or DaemonSet",
                xy=(x0 + 1 * (box_w + gap) + box_w / 2, y_main + box_h),
                xytext=(x0 + 1 * (box_w + gap) + box_w / 2, y_main + box_h + 0.7),
                ha="center", va="bottom", fontsize=9, color=C_DARK, style="italic",
                arrowprops=dict(arrowstyle="-", color=C_GRAY, lw=1))

    # Below: retention tiers
    tier_y = 1.2
    tiers = [
        ("hot (7d)", C_RED, "SSD, sub-second search"),
        ("warm (30d)", C_AMBER, "HDD, slower search"),
        ("cold (90d+)", C_BLUE, "S3 / object store"),
        ("archive (1-7y)", C_GRAY, "Glacier / tape"),
    ]
    tier_w = 2.3
    tier_x0 = 0.5
    for i, (name, c, desc) in enumerate(tiers):
        x = tier_x0 + i * (tier_w + 0.25)
        fc = {C_RED: "#fee2e2", C_AMBER: "#fef3c7", C_BLUE: "#dbeafe"}.get(c, "#f1f5f9")
        _box(ax, x, tier_y, tier_w, 0.7, "",
             fc=fc, ec=c, lw=1.4)
        ax.text(x + tier_w / 2, tier_y + 0.48, name,
                ha="center", va="center", fontsize=10, weight="bold", color=c)
        ax.text(x + tier_w / 2, tier_y + 0.18, desc,
                ha="center", va="center", fontsize=8.5, color=C_DARK, style="italic")
    ax.text(tier_x0 + (4 * (tier_w + 0.25)) / 2 - 0.125, tier_y + 0.95,
            "Retention tiers (ILM): trade access latency for storage cost",
            ha="center", va="bottom", fontsize=10, color=C_DARK, weight="semibold")

    # Bottom hint
    ax.text(6.0, 0.4,
            "Decouple shippers from store with a buffer; structure logs as JSON; tier storage to control cost.",
            ha="center", va="center",
            fontsize=10, color=C_DARK, style="italic",
            bbox=dict(facecolor=C_BG, edgecolor=C_LIGHT, boxstyle="round,pad=0.4"))

    ax.set_xlim(0, 12.5)
    ax.set_ylim(0.0, 5.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Centralised Logging Architecture (EFK / ELK)", fontsize=14, weight="bold", color=C_DARK, pad=14)
    ax.grid(False)

    _save(fig, "fig4_logging_architecture")


# ---------------------------------------------------------------------------
# Figure 5: SRE error budget over time
# ---------------------------------------------------------------------------
def fig5_error_budget() -> None:
    """Error budget burn over a 30-day window with policy zones."""
    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    # x: days 0..30, y: error budget remaining (%)
    days = np.arange(0, 31)
    np.random.seed(42)

    # Construct a realistic burn trajectory:
    # - First 10 days: gentle burn (good)
    # - Days 10-15: incident, fast burn
    # - Days 15-22: stabilised, gentle burn
    # - Days 22-26: another small bump
    # - Days 26-30: feature freeze, almost flat
    burn = np.zeros_like(days, dtype=float)
    burn[0] = 100.0
    daily = np.array([
        2.0, 1.8, 2.2, 1.5, 2.0, 1.9, 1.7, 2.1, 1.8, 2.0,    # 1-10
        9.0, 12.0, 8.0, 6.0, 4.5,                              # 11-15 incident
        2.0, 1.8, 1.5, 2.2, 2.0, 1.7, 1.9,                    # 16-22
        4.5, 3.2, 2.8, 2.0,                                    # 23-26
        0.6, 0.5, 0.4, 0.3                                     # 27-30 freeze
    ])
    for i in range(1, 31):
        burn[i] = max(0, burn[i - 1] - daily[i - 1])

    # Background zones: green (healthy), amber (caution), red (critical)
    ax.axhspan(50, 100, facecolor="#d1fae5", alpha=0.5, zorder=0)
    ax.axhspan(20, 50, facecolor="#fef3c7", alpha=0.5, zorder=0)
    ax.axhspan(0, 20, facecolor="#fee2e2", alpha=0.5, zorder=0)

    # Zone labels
    ax.text(30.5, 75, "HEALTHY\nfeature work\nproceeds",
            ha="left", va="center", fontsize=9.5, color=C_GREEN, weight="bold")
    ax.text(30.5, 35, "CAUTION\nslow risky\nchanges",
            ha="left", va="center", fontsize=9.5, color="#92400e", weight="bold")
    ax.text(30.5, 10, "CRITICAL\nfreeze features\nfocus reliability",
            ha="left", va="center", fontsize=9.5, color=C_RED, weight="bold")

    # Burn line
    ax.plot(days, burn, color=C_BLUE, linewidth=2.6, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor=C_BLUE, zorder=4)

    # Mark the incident
    ax.axvspan(10, 15, facecolor=C_RED, alpha=0.08, zorder=1)
    ax.annotate("Incident:\nbad deploy\n12% burn in 24h",
                xy=(11, burn[11]),
                xytext=(7.5, 18),
                ha="center", va="center", fontsize=9.5, color=C_RED, weight="semibold",
                arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.4))

    # Mark feature freeze
    ax.axvspan(26, 30, facecolor=C_GREEN, alpha=0.08, zorder=1)
    ax.annotate("Feature freeze:\nbudget recovers",
                xy=(28, burn[28]),
                xytext=(24, 8),
                ha="center", va="center", fontsize=9.5, color=C_GREEN, weight="semibold",
                arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.4))

    # SLO budget line at 0
    ax.axhline(0, color=C_DARK, linewidth=1.2, linestyle="--", zorder=2)
    ax.text(0.2, 2, "SLO breach (0% remaining)",
            ha="left", va="bottom", fontsize=9, color=C_DARK, style="italic")

    # Title and axes
    ax.set_xlabel("Day of 30-day rolling window", fontsize=11, color=C_DARK)
    ax.set_ylabel("Error budget remaining (%)", fontsize=11, color=C_DARK)
    ax.set_xlim(-0.5, 30.5)
    ax.set_ylim(-2, 105)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.set_yticks([0, 20, 50, 80, 100])
    ax.tick_params(labelsize=10, colors=C_DARK)
    for spine in ax.spines.values():
        spine.set_color(C_LIGHT)
    ax.grid(True, color=C_LIGHT, linewidth=0.7, zorder=0)

    ax.set_title("SRE Error Budget: Burn Rate Drives Release Policy", fontsize=14, weight="bold", color=C_DARK, pad=14)

    # Top-right summary box
    ax.text(0.99, 0.99,
            "SLO   = 99.9%  (rolling 30 days)\nBudget = 0.1%  ~= 43 min/month",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=10, color=C_DARK, family="monospace",
            bbox=dict(facecolor="white", edgecolor=C_LIGHT, boxstyle="round,pad=0.5"))

    _save(fig, "fig5_error_budget")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("[Cloud Computing 06: DevOps] generating figures...")
    fig1_cicd_pipeline()
    fig2_iac_terraform()
    fig3_monitoring_stack()
    fig4_logging_architecture()
    fig5_error_budget()
    print("done.")
