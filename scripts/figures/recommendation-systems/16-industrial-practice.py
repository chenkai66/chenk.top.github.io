"""Figures for Recommendation Systems Part 16 — Industrial Architecture & Best Practices.

Generates 7 publication-quality figures explaining the production
recommendation stack: full pipeline, recall/rank/rerank funnel, feature
store, A/B testing, training pipeline, serving infrastructure, and
organizational responsibilities.

Output is written to BOTH the EN and ZH asset folders so the script is
the single source of truth. Run from any working directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

BLUE = "#2563eb"
PURPLE = "#7c3aed"
GREEN = "#10b981"
ORANGE = "#f59e0b"
GRAY = "#64748b"
LIGHT = "#e5e7eb"
DARK = "#1e293b"

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/recommendation-systems/16-industrial-practice"
ZH_DIR = REPO / "source/_posts/zh/recommendation-systems/16-工业级架构与最佳实践"


def save(fig: plt.Figure, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def rounded_box(ax, x, y, w, h, text, color, text_color="white", fontsize=10, weight="bold"):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=color, edgecolor="none",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, weight=weight)


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=1.6, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        color=color, linewidth=lw,
    ))


# ---------------------------------------------------------------------------
# Figure 1 — Full industrial pipeline
# ---------------------------------------------------------------------------
def fig1_industrial_pipeline():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(6.5, 7.15, "Industrial Recommendation Pipeline (YouTube / Taobao / TikTok-style)",
            ha="center", fontsize=14, weight="bold", color=DARK)

    # Three swim lanes: Data, Training, Serving
    lanes = [
        (0.3, 5.0, 12.4, 1.7, "Data Plane",     "#eef2ff"),
        (0.3, 2.8, 12.4, 1.7, "Training Plane", "#fef3c7"),
        (0.3, 0.4, 12.4, 1.9, "Serving Plane",  "#ecfdf5"),
    ]
    for x, y, w, h, label, color in lanes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=LIGHT, linewidth=1))
        ax.text(x + 0.08, y + h - 0.18, label, fontsize=9, weight="bold", color=GRAY)

    # Data plane
    data_boxes = [
        (0.6, 5.5, 1.7, 0.9, "User Logs\n(clicks, dwell)", BLUE),
        (2.6, 5.5, 1.7, 0.9, "Item Catalog\n+ Content", PURPLE),
        (4.6, 5.5, 1.7, 0.9, "Realtime Stream\n(Kafka / Flink)", ORANGE),
        (6.6, 5.5, 1.7, 0.9, "Feature Store\n(offline + online)", GREEN),
        (8.6, 5.5, 1.7, 0.9, "Sample\nGeneration", GRAY),
        (10.6, 5.5, 1.7, 0.9, "Data Lake\n(HDFS / S3)", DARK),
    ]
    for x, y, w, h, t, c in data_boxes:
        rounded_box(ax, x, y, w, h, t, c, fontsize=8.5)

    # Training plane
    train_boxes = [
        (0.6, 3.2, 2.0, 1.0, "Daily Batch\nTraining", BLUE),
        (3.0, 3.2, 2.0, 1.0, "Hourly\nIncremental", PURPLE),
        (5.4, 3.2, 2.0, 1.0, "Online Learning\n(streaming)", ORANGE),
        (7.8, 3.2, 2.0, 1.0, "Model Validation\n+ Offline AUC", GREEN),
        (10.2, 3.2, 2.1, 1.0, "Model Registry\n+ Versioning", DARK),
    ]
    for x, y, w, h, t, c in train_boxes:
        rounded_box(ax, x, y, w, h, t, c, fontsize=8.5)

    # Serving plane funnel
    funnel = [
        (0.6, 0.7, 1.9, 1.4, "Recall\n~10⁶ → 2K", BLUE, "<30 ms"),
        (2.9, 0.7, 1.9, 1.4, "Coarse Rank\n2K → 200", PURPLE, "<20 ms"),
        (5.2, 0.7, 1.9, 1.4, "Fine Rank\n200 → 50", ORANGE, "<50 ms"),
        (7.5, 0.7, 1.9, 1.4, "Re-rank\n50 → 20", GREEN, "<20 ms"),
        (9.8, 0.7, 2.5, 1.4, "Response\nto User", DARK, "p95 < 100 ms"),
    ]
    for x, y, w, h, t, c, lat in funnel:
        rounded_box(ax, x, y, w, h, t, c, fontsize=9)
        ax.text(x + w / 2, y - 0.18, lat, ha="center", fontsize=8, color=GRAY, style="italic")

    # Cross-plane arrows
    arrow(ax, 6.5, 5.45, 6.5, 4.25, color=GRAY, lw=1.4)
    arrow(ax, 11.2, 3.15, 11.2, 2.15, color=GRAY, lw=1.4)
    arrow(ax, 7.5, 6.3, 7.5, 5.5, color=GRAY, lw=1.2, style="-|>")
    # Funnel flow
    for i in range(4):
        arrow(ax, 0.6 + 1.9 + i * 2.3, 1.4, 0.6 + 1.9 + i * 2.3 + 0.4, 1.4, color=DARK, lw=1.6)

    save(fig, "fig1_industrial_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 2 — Recall + Ranking + Reranking funnel with item counts
# ---------------------------------------------------------------------------
def fig2_funnel():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    ax.text(5.5, 6.15, "Multi-Stage Funnel: Item Counts and Latency Budget",
            ha="center", fontsize=14, weight="bold", color=DARK)

    stages = [
        ("Item Pool",       1_000_000, 4.5, BLUE,    "Catalog of\nall items"),
        ("Recall",          2_000,     3.6, PURPLE,  "Multi-channel:\nCF + 2-tower\n+ graph + RT"),
        ("Coarse Rank",     200,       2.7, ORANGE,  "XGBoost or\n2-tower DNN"),
        ("Fine Rank",       50,        1.8, GREEN,   "Wide&Deep,\nDeepFM, DIN"),
        ("Re-rank",         20,        1.0, DARK,    "Diversity (MMR)\nbusiness rules"),
    ]

    max_w = 9.0
    log_max = np.log10(stages[0][1])
    for name, n, y, color, desc in stages:
        w = max_w * (np.log10(n) / log_max)
        x = (11 - w) / 2
        rounded_box(ax, x, y, w, 0.65, f"{name}: {n:,} items", color, fontsize=11)
        ax.text(11 - 0.3, y + 0.32, desc, ha="right", va="center",
                fontsize=8.5, color=GRAY, style="italic")

    # Latency budgets on left
    budgets = [("", 4.5), ("<30 ms", 3.6), ("<20 ms", 2.7), ("<50 ms", 1.8), ("<20 ms", 1.0)]
    for lat, y in budgets:
        ax.text(0.3, y + 0.32, lat, ha="left", va="center", fontsize=9,
                color=ORANGE, weight="bold")

    # Arrows
    for y_top, y_bot in [(4.5, 4.25), (3.6, 3.35), (2.7, 2.45), (1.8, 1.65)]:
        arrow(ax, 5.5, y_top, 5.5, y_bot, color=DARK, lw=1.8)

    ax.text(5.5, 0.4, "Total p95 latency budget: < 100 ms",
            ha="center", fontsize=10, weight="bold", color=ORANGE)

    save(fig, "fig2_funnel.png")


# ---------------------------------------------------------------------------
# Figure 3 — Feature store architecture (offline batch + online realtime)
# ---------------------------------------------------------------------------
def fig3_feature_store():
    fig, ax = plt.subplots(figsize=(12.5, 7))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(6.25, 6.65, "Feature Store: Unified Offline + Online Pipelines",
            ha="center", fontsize=14, weight="bold", color=DARK)

    # Offline (top)
    ax.add_patch(Rectangle((0.3, 3.6), 12.0, 2.6, facecolor="#eef2ff",
                           edgecolor=LIGHT, linewidth=1))
    ax.text(0.5, 6.0, "Offline Batch Path (T+1, daily / hourly)",
            fontsize=10, weight="bold", color=BLUE)

    offline = [
        (0.6, 4.4, 1.7, 1.1, "Data Lake\n(Hive / S3)", BLUE),
        (2.7, 4.4, 1.7, 1.1, "Spark / Flink\nBatch Jobs", BLUE),
        (4.8, 4.4, 1.7, 1.1, "Feature\nDefinitions\n(YAML / SQL)", PURPLE),
        (6.9, 4.4, 1.7, 1.1, "Backfill +\nQA Checks", PURPLE),
        (9.0, 4.4, 1.7, 1.1, "Offline Store\n(Parquet,\ntraining)", GREEN),
        (10.9, 4.0, 1.4, 1.7, "Training\nDataset", DARK),
    ]
    for x, y, w, h, t, c in offline:
        rounded_box(ax, x, y, w, h, t, c, fontsize=8.5)
    for i in range(4):
        arrow(ax, 0.6 + 1.7 + i * 2.1, 4.95, 0.6 + 1.7 + i * 2.1 + 0.4, 4.95, color=BLUE, lw=1.6)
    arrow(ax, 10.7, 4.95, 10.9, 4.95, color=BLUE, lw=1.6)

    # Online (bottom)
    ax.add_patch(Rectangle((0.3, 0.3), 12.0, 2.6, facecolor="#ecfdf5",
                           edgecolor=LIGHT, linewidth=1))
    ax.text(0.5, 2.65, "Online Realtime Path (event-driven, ms latency)",
            fontsize=10, weight="bold", color=GREEN)

    online = [
        (0.6, 1.1, 1.7, 1.1, "Kafka\nEvent Stream", ORANGE),
        (2.7, 1.1, 1.7, 1.1, "Flink Stream\nAggregations", ORANGE),
        (4.8, 1.1, 1.7, 1.1, "Same Feature\nDefinition\n(reuse)", PURPLE),
        (6.9, 1.1, 1.7, 1.1, "Feature\nValidator", PURPLE),
        (9.0, 1.1, 1.7, 1.1, "Online Store\n(Redis /\nKV)", GREEN),
        (10.9, 0.7, 1.4, 1.7, "Serving\n(< 5 ms)", DARK),
    ]
    for x, y, w, h, t, c in online:
        rounded_box(ax, x, y, w, h, t, c, fontsize=8.5)
    for i in range(4):
        arrow(ax, 0.6 + 1.7 + i * 2.1, 1.65, 0.6 + 1.7 + i * 2.1 + 0.4, 1.65, color=GREEN, lw=1.6)
    arrow(ax, 10.7, 1.65, 10.9, 1.65, color=GREEN, lw=1.6)

    # Shared definition arrow
    ax.annotate("", xy=(5.65, 2.2), xytext=(5.65, 4.4),
                arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=2,
                                connectionstyle="arc3,rad=0"))
    ax.text(5.95, 3.3, "Single feature\ndefinition\nprevents skew",
            fontsize=8, color=PURPLE, style="italic", weight="bold")

    save(fig, "fig3_feature_store.png")


# ---------------------------------------------------------------------------
# Figure 4 — A/B test results
# ---------------------------------------------------------------------------
def fig4_ab_test():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Left: CTR over 14 days
    np.random.seed(7)
    days = np.arange(1, 15)
    ctrl = 0.0420 + np.random.normal(0, 0.0008, 14)
    treat = 0.0438 + np.random.normal(0, 0.0009, 14) + np.linspace(0, 0.0012, 14)

    ax = axes[0]
    ax.plot(days, ctrl * 100, marker="o", color=GRAY, lw=2, label="Control",
            markersize=6, markeredgecolor="white")
    ax.plot(days, treat * 100, marker="o", color=BLUE, lw=2, label="Treatment (DIN)",
            markersize=6, markeredgecolor="white")

    # Confidence band
    ctrl_se = 0.0008
    ax.fill_between(days, (ctrl - ctrl_se) * 100, (ctrl + ctrl_se) * 100,
                    color=GRAY, alpha=0.15)
    ax.fill_between(days, (treat - 0.0009) * 100, (treat + 0.0009) * 100,
                    color=BLUE, alpha=0.15)

    # Significance marker
    ax.axvline(x=7, color=ORANGE, linestyle="--", alpha=0.7, lw=1.5)
    ax.text(7.2, 4.55, "p < 0.05\nreached", fontsize=9, color=ORANGE, weight="bold")

    ax.set_title("A/B Test: CTR over 14 days", color=DARK)
    ax.set_xlabel("Day")
    ax.set_ylabel("CTR (%)")
    ax.legend(loc="lower right")
    ax.set_ylim(4.0, 4.8)

    # Right: lift breakdown by metric
    ax = axes[1]
    metrics = ["CTR", "CVR", "Dwell\nTime", "Diversity\n(entropy)", "Revenue\nper user"]
    lifts = [4.3, 2.1, 6.8, -1.4, 3.7]
    colors = [GREEN if v >= 0 else ORANGE for v in lifts]
    bars = ax.barh(metrics, lifts, color=colors, edgecolor="white", linewidth=1.2)

    for bar, v in zip(bars, lifts):
        x = v + (0.2 if v >= 0 else -0.2)
        ha = "left" if v >= 0 else "right"
        ax.text(x, bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}%", va="center", ha=ha,
                fontsize=10, weight="bold", color=DARK)

    ax.axvline(x=0, color=DARK, lw=1)
    ax.set_title("Lift vs Control (statistically significant)", color=DARK)
    ax.set_xlabel("Relative Lift (%)")
    ax.set_xlim(-3, 9)

    fig.suptitle("A/B Test Results — Control vs DIN Treatment (mock data)",
                 fontsize=14, weight="bold", color=DARK, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_ab_test_results.png")


# ---------------------------------------------------------------------------
# Figure 5 — ML training pipeline with retrain triggers
# ---------------------------------------------------------------------------
def fig5_training_pipeline():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(6.5, 6.65, "Continuous Training Pipeline with Retrain Triggers",
            ha="center", fontsize=14, weight="bold", color=DARK)

    # Main pipeline (horizontal)
    main = [
        (0.4, 3.7, 1.9, 1.0, "Data\nIngestion", BLUE),
        (2.6, 3.7, 1.9, 1.0, "Feature\nEngineering", BLUE),
        (4.8, 3.7, 1.9, 1.0, "Sample\nGeneration", PURPLE),
        (7.0, 3.7, 1.9, 1.0, "Model\nTraining", PURPLE),
        (9.2, 3.7, 1.9, 1.0, "Offline Eval\n(AUC, NDCG)", GREEN),
        (11.4, 3.7, 1.4, 1.0, "Model\nRegistry", DARK),
    ]
    for x, y, w, h, t, c in main:
        rounded_box(ax, x, y, w, h, t, c, fontsize=9)
    for i in range(5):
        x_end = 0.4 + 1.9 + i * 2.2
        arrow(ax, x_end, 4.2, x_end + 0.3, 4.2, color=DARK, lw=1.8)

    # Triggers (top) feeding into pipeline
    ax.text(6.5, 6.0, "Retrain Triggers", ha="center", fontsize=11,
            weight="bold", color=ORANGE)
    triggers = [
        (0.7, 5.05, 2.4, 0.7, "Scheduled (daily / hourly)", ORANGE),
        (3.4, 5.05, 2.4, 0.7, "Drift detected (PSI > 0.2)", ORANGE),
        (6.1, 5.05, 2.4, 0.7, "AUC drop > 2%", ORANGE),
        (8.8, 5.05, 2.4, 0.7, "New feature / model code", ORANGE),
    ]
    for x, y, w, h, t, c in triggers:
        rounded_box(ax, x, y, w, h, t, c, fontsize=9)
    arrow(ax, 6.5, 5.05, 6.5, 4.7, color=ORANGE, lw=1.8)

    # Deployment (bottom) — canary, full rollout
    ax.text(6.5, 3.0, "Deployment", ha="center", fontsize=11, weight="bold", color=GREEN)
    deploy = [
        (0.7, 1.7, 2.0, 1.0, "Shadow\n(0% traffic)", GRAY),
        (3.0, 1.7, 2.0, 1.0, "Canary\n(1-10%)", ORANGE),
        (5.3, 1.7, 2.0, 1.0, "A/B Test\n(50%)", BLUE),
        (7.6, 1.7, 2.0, 1.0, "Full Rollout\n(100%)", GREEN),
        (9.9, 1.7, 2.6, 1.0, "Auto-Rollback\nif metrics drop", DARK),
    ]
    for x, y, w, h, t, c in deploy:
        rounded_box(ax, x, y, w, h, t, c, fontsize=9)
    for i in range(4):
        x_end = 0.7 + 2.0 + i * 2.3
        arrow(ax, x_end, 2.2, x_end + 0.3, 2.2, color=DARK, lw=1.6)

    arrow(ax, 12.1, 3.7, 12.1, 2.7, color=DARK, lw=1.6)
    ax.text(12.4, 3.2, "promote", fontsize=8, color=GRAY, style="italic")

    # Monitoring feedback
    ax.add_patch(FancyArrowPatch((11.2, 1.7), (1.5, 0.7),
                                  arrowstyle="-|>", mutation_scale=14,
                                  color=PURPLE, linewidth=1.5,
                                  connectionstyle="arc3,rad=0.15"))
    ax.text(6.5, 0.4, "Online metrics feed back to drift detector  →  triggers next retrain",
            ha="center", fontsize=9, color=PURPLE, style="italic", weight="bold")

    save(fig, "fig5_training_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 6 — Serving infrastructure
# ---------------------------------------------------------------------------
def fig6_serving_infra():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.text(6.5, 7.15, "Serving Infrastructure: Load Balancer + Model Servers + Cache",
            ha="center", fontsize=14, weight="bold", color=DARK)

    # Client
    rounded_box(ax, 0.3, 3.2, 1.5, 1.0, "Mobile /\nWeb Client", DARK, fontsize=9)
    arrow(ax, 1.8, 3.7, 2.4, 3.7, color=DARK, lw=1.8)

    # API Gateway / LB
    rounded_box(ax, 2.4, 3.2, 1.6, 1.0, "API Gateway\n+ LB (Nginx /\nEnvoy)", BLUE, fontsize=9)
    arrow(ax, 4.0, 3.7, 4.6, 3.7, color=DARK, lw=1.8)

    # Recommendation orchestrator
    rounded_box(ax, 4.6, 3.2, 2.0, 1.0, "Rec Orchestrator\n(stateless,\nautoscaled)", PURPLE, fontsize=9)

    # Three downstream services
    ax.add_patch(Rectangle((6.9, 0.4), 6.0, 6.5, facecolor="#f8fafc",
                           edgecolor=LIGHT, linewidth=1))
    ax.text(9.9, 6.6, "Backend Services", ha="center", fontsize=10,
            weight="bold", color=GRAY)

    # Recall service
    rounded_box(ax, 7.1, 5.4, 2.5, 0.9, "Recall Service\n(2-tower + ANN)", BLUE, fontsize=9)
    rounded_box(ax, 9.9, 5.4, 1.4, 0.9, "Faiss /\nHNSW", GRAY, fontsize=8.5)
    rounded_box(ax, 11.5, 5.4, 1.3, 0.9, "Item Emb\nIndex", GRAY, fontsize=8.5)

    # Ranking service
    rounded_box(ax, 7.1, 4.0, 2.5, 0.9, "Ranking Service\n(GPU model server)", PURPLE, fontsize=9)
    rounded_box(ax, 9.9, 4.0, 1.4, 0.9, "TF-Serving /\nTriton", GRAY, fontsize=8.5)
    rounded_box(ax, 11.5, 4.0, 1.3, 0.9, "Model\nReplicas", GRAY, fontsize=8.5)

    # Feature service
    rounded_box(ax, 7.1, 2.6, 2.5, 0.9, "Feature Service", GREEN, fontsize=9)
    rounded_box(ax, 9.9, 2.6, 1.4, 0.9, "Redis\nCluster", GRAY, fontsize=8.5)
    rounded_box(ax, 11.5, 2.6, 1.3, 0.9, "HBase\n(cold)", GRAY, fontsize=8.5)

    # Caches
    rounded_box(ax, 7.1, 1.2, 2.5, 0.9, "Prediction Cache", ORANGE, fontsize=9)
    rounded_box(ax, 9.9, 1.2, 2.9, 0.9, "Redis (TTL=300 s, ~30% hit)", GRAY, fontsize=8.5)

    # Connections from orchestrator
    for y in [5.85, 4.45, 3.05, 1.65]:
        arrow(ax, 6.6, 3.7, 7.1, y, color=PURPLE, lw=1.4)

    # Connections within services
    for y in [5.85, 4.45, 3.05]:
        arrow(ax, 9.6, y, 9.9, y, color=GRAY, lw=1.2)
        arrow(ax, 11.3, y, 11.5, y, color=GRAY, lw=1.2)
    arrow(ax, 9.6, 1.65, 9.9, 1.65, color=GRAY, lw=1.2)

    # Monitoring sidecar
    rounded_box(ax, 0.3, 0.5, 4.0, 1.0, "Observability: Prometheus + Grafana + alerts",
                DARK, fontsize=9)
    arrow(ax, 5.6, 3.2, 2.3, 1.5, color=GRAY, lw=1.0, style="-|>")

    save(fig, "fig6_serving_infra.png")


# ---------------------------------------------------------------------------
# Figure 7 — Org / role responsibilities
# ---------------------------------------------------------------------------
def fig7_org_roles():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(6.5, 6.65, "Who Owns What: Roles in a Production Recommendation Team",
            ha="center", fontsize=14, weight="bold", color=DARK)

    # Layers (vertical) showing the stack
    layers = [
        (0.3, 5.4, "Business / Product", "Define KPIs, growth strategy, content policy", DARK),
        (0.3, 4.4, "Algorithm Engineer", "Recall / ranking / re-rank models, feature design, A/B experiments", PURPLE),
        (0.3, 3.4, "Data Engineer",      "ETL pipelines, feature store, sample generation, data quality", BLUE),
        (0.3, 2.4, "MLOps / Platform",   "Training infra, model registry, CI/CD, serving runtime", ORANGE),
        (0.3, 1.4, "SRE / Infra",        "Capacity, latency SLOs, incident response, autoscaling", GREEN),
        (0.3, 0.4, "Analyst / Research", "Long-horizon evaluation, causal inference, ranking diagnostics", GRAY),
    ]
    for x, y, role, desc, color in layers:
        rounded_box(ax, x, y, 2.6, 0.85, role, color, fontsize=10)
        ax.text(3.1, y + 0.43, desc, va="center", fontsize=9.5, color=DARK)

    # Right column: ownership matrix
    ax.text(11.0, 6.0, "Primary Owners by Stage", ha="center", fontsize=11,
            weight="bold", color=DARK)

    matrix = [
        ("Recall",        ["Algorithm", "Data"]),
        ("Ranking",       ["Algorithm", "MLOps"]),
        ("Re-ranking",    ["Algorithm", "Product"]),
        ("Feature store", ["Data", "MLOps"]),
        ("Serving",       ["MLOps", "SRE"]),
        ("Monitoring",    ["SRE", "MLOps"]),
        ("A/B testing",   ["Algorithm", "Analyst"]),
    ]
    role_colors = {
        "Algorithm": PURPLE, "Data": BLUE, "MLOps": ORANGE,
        "SRE": GREEN, "Analyst": GRAY, "Product": DARK,
    }
    y0 = 5.4
    for i, (stage, owners) in enumerate(matrix):
        y = y0 - i * 0.7
        ax.text(9.4, y, stage, ha="right", va="center",
                fontsize=9.5, color=DARK, weight="bold")
        for j, owner in enumerate(owners):
            x = 9.7 + j * 1.55
            rounded_box(ax, x, y - 0.2, 1.45, 0.4, owner,
                        role_colors[owner], fontsize=8.5)

    save(fig, "fig7_org_roles.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    fig1_industrial_pipeline()
    fig2_funnel()
    fig3_feature_store()
    fig4_ab_test()
    fig5_training_pipeline()
    fig6_serving_infra()
    fig7_org_roles()
    print(f"Wrote 7 figures to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
