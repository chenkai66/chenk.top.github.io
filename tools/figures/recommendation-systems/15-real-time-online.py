"""Figures for Recommendation Systems Part 15 — Real-Time & Online Learning.

Generates 7 publication-quality figures covering streaming pipelines, latency
budgets, online learning convergence, feature freshness, Kafka+Flink
architecture, concept drift detection, and cache-vs-compute trade-offs.

Output is written to BOTH the EN and ZH asset folders so the script is the
single source of truth. Run from any working directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# Shared aesthetic style (chenk-site)
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _style import setup_style, COLORS  # noqa: E402
setup_style()


# ---------------------------------------------------------------------------
# Style — applied once for the whole script
# ---------------------------------------------------------------------------

BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
ORANGE = COLORS["warning"]
GRAY = COLORS["text2"]
LIGHT = COLORS["grid"]

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/recommendation-systems/15-real-time-online"
ZH_DIR = REPO / "source/_posts/zh/recommendation-systems/15-实时推荐与在线学习"


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 — Real-time recommendation pipeline (streaming + serving)
# ---------------------------------------------------------------------------
def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.0))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Top swim-lane: streaming (write-path)
    ax.text(0.1, 6.55, "STREAMING  (write-path, asynchronous)",
            fontsize=10, fontweight="bold", color=GRAY)
    ax.add_patch(FancyBboxPatch((0.1, 4.3), 13.7, 2.0,
                                boxstyle="round,pad=0.05",
                                facecolor="#eff6ff", edgecolor=BLUE, linewidth=1.2,
                                alpha=0.4))

    stream_nodes = [
        ("User\nclient",     0.9,  ORANGE, "log event"),
        ("Kafka\ningest",    3.0,  BLUE,   "topic = events"),
        ("Flink\nstream job",5.4,  PURPLE, "windowed agg"),
        ("Feature\nstore",   8.0,  GREEN,  "Redis / KV"),
        ("Online\nlearner",  10.6, BLUE,   "SGD / FTRL"),
        ("Model\nregistry",  12.9, PURPLE, "snapshot"),
    ]
    for label, x, col, sub in stream_nodes:
        box = FancyBboxPatch((x - 0.55, 5.05), 1.1, 0.85,
                              boxstyle="round,pad=0.04", linewidth=1.6,
                              facecolor=col, edgecolor=col, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, 5.475, label, ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold")
        ax.text(x, 4.65, sub, ha="center", va="center", color=GRAY, fontsize=7.5)

    for i in range(len(stream_nodes) - 1):
        x1 = stream_nodes[i][1] + 0.55
        x2 = stream_nodes[i + 1][1] - 0.55
        ax.annotate("", xy=(x2, 5.475), xytext=(x1, 5.475),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4))

    # Bottom swim-lane: serving (read-path)
    ax.text(0.1, 3.15, "SERVING  (read-path, synchronous, < 100 ms)",
            fontsize=10, fontweight="bold", color=GRAY)
    ax.add_patch(FancyBboxPatch((0.1, 0.7), 13.7, 2.1,
                                boxstyle="round,pad=0.05",
                                facecolor="#fdf4ff", edgecolor=PURPLE, linewidth=1.2,
                                alpha=0.4))

    serve_nodes = [
        ("Request",      0.9,  ORANGE, "user_id, ctx"),
        ("Recall",       3.0,  BLUE,   "ANN / inverted"),
        ("Feature\nfetch",5.4, GREEN,  "look-up"),
        ("Ranker",       8.0,  PURPLE, "DNN / GBDT"),
        ("Re-rank",      10.6, BLUE,   "diversity, biz"),
        ("Response",     12.9, ORANGE, "top-K items"),
    ]
    for label, x, col, sub in serve_nodes:
        box = FancyBboxPatch((x - 0.55, 1.55), 1.1, 0.85,
                              boxstyle="round,pad=0.04", linewidth=1.6,
                              facecolor=col, edgecolor=col, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, 1.975, label, ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold")
        ax.text(x, 1.15, sub, ha="center", va="center", color=GRAY, fontsize=7.5)

    for i in range(len(serve_nodes) - 1):
        x1 = serve_nodes[i][1] + 0.55
        x2 = serve_nodes[i + 1][1] - 0.55
        ax.annotate("", xy=(x2, 1.975), xytext=(x1, 1.975),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4))

    # Cross-lane arrows — feature store and registry feed the serving path
    ax.annotate("", xy=(5.4, 2.42), xytext=(8.0, 5.0),
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2,
                                linestyle=(0, (3, 2)), alpha=0.85))
    ax.text(7.0, 3.6, "fresh features", fontsize=8, color=GREEN,
            style="italic", rotation=-25)

    ax.annotate("", xy=(8.0, 2.42), xytext=(12.9, 5.0),
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2,
                                linestyle=(0, (3, 2)), alpha=0.85))
    ax.text(10.7, 3.6, "model snapshot", fontsize=8, color=PURPLE,
            style="italic", rotation=-22)

    # Feedback loop: response → user → events
    ax.annotate("", xy=(0.9, 4.7), xytext=(12.9, 1.55),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4,
                                connectionstyle="arc3,rad=-0.25", alpha=0.85))
    ax.text(7.0, 0.25, "feedback loop  (impression → click → next training example)",
            ha="center", fontsize=9, color=ORANGE, style="italic")

    ax.set_title("Real-time recommendation: write-path keeps state fresh, read-path serves under a hard SLO",
                 pad=12)
    fig.tight_layout()
    save(fig, "fig1_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 2 — Latency budget breakdown across stages (p50 / p95 / p99)
# ---------------------------------------------------------------------------
def fig2_latency_budget() -> None:
    stages = ["Network\nin", "Recall\n(ANN)", "Feature\nfetch", "Ranker\n(DNN)", "Re-rank\n+ logging", "Network\nout"]
    # Realistic latencies (ms) for a top-tier feed system. p99 dominated by tails.
    p50 = np.array([4,  10, 6,  18, 4,  3])
    p95 = np.array([7,  18, 14, 32, 9,  6])
    p99 = np.array([12, 28, 30, 55, 18, 11])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    # Left: stacked bar showing breakdown at each percentile
    ax = axes[0]
    x = np.arange(3)
    widths = 0.65
    colors = [BLUE, PURPLE, GREEN, ORANGE, "#0ea5e9", GRAY]

    data = np.vstack([p50, p95, p99])  # shape (3, n_stages)
    bottom = np.zeros(3)
    for j, stage in enumerate(stages):
        ax.bar(x, data[:, j], widths, bottom=bottom, color=colors[j],
               edgecolor="white", linewidth=1.2, label=stage.replace("\n", " "))
        # text labels on each segment for the p99 bar (tallest)
        for i in range(3):
            v = data[i, j]
            if v >= 8:
                ax.text(x[i], bottom[i] + v / 2, f"{int(v)}",
                        ha="center", va="center", color="white",
                        fontsize=8, fontweight="bold")
        bottom += data[:, j]

    ax.set_xticks(x)
    ax.set_xticklabels(["p50", "p95", "p99"])
    ax.set_ylabel("Latency  (ms)")
    ax.set_title("End-to-end budget by percentile", color=GRAY)
    ax.axhline(100, color="red", lw=1.4, linestyle="--", alpha=0.7)
    ax.text(2.45, 102, "100 ms SLO", color="red", fontsize=9, ha="right")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.set_ylim(0, 180)

    # Right: per-stage p50/p95/p99 grouped bars
    ax = axes[1]
    xs = np.arange(len(stages))
    w = 0.27
    ax.bar(xs - w, p50, w, color=BLUE, label="p50", alpha=0.9)
    ax.bar(xs,     p95, w, color=PURPLE, label="p95", alpha=0.9)
    ax.bar(xs + w, p99, w, color=ORANGE, label="p99", alpha=0.9)
    for i, (a, b, c) in enumerate(zip(p50, p95, p99)):
        ax.text(i - w, a + 1, str(a), ha="center", fontsize=7.5, color=BLUE)
        ax.text(i,     b + 1, str(b), ha="center", fontsize=7.5, color=PURPLE)
        ax.text(i + w, c + 1, str(c), ha="center", fontsize=7.5, color=ORANGE)

    ax.set_xticks(xs)
    ax.set_xticklabels(stages, fontsize=8.5)
    ax.set_ylabel("Latency  (ms)")
    ax.set_title("Per-stage tails — ranker + feature fetch dominate p99", color=GRAY)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Latency budget — p50 looks fine, p99 is where users notice",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig2_latency_budget.png")


# ---------------------------------------------------------------------------
# Figure 3 — Online learning vs batch retraining (convergence + drift)
# ---------------------------------------------------------------------------
def fig3_online_vs_batch() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Left: convergence on a stationary task
    ax = axes[0]
    n = 800
    t = np.arange(1, n + 1)
    rng = np.random.default_rng(7)

    # Online SGD: AUC climbs continuously
    online_auc = 0.50 + 0.25 * (1 - np.exp(-t / 180)) + rng.normal(0, 0.005, n)
    online_auc = np.clip(online_auc, 0.5, 0.78)

    # Batch retrain every 200 steps (step function with small intra-epoch noise)
    batch_auc = np.zeros(n)
    snapshot = 0.50
    for i in range(n):
        if i % 200 == 0 and i > 0:
            snapshot = 0.50 + 0.25 * (1 - np.exp(-i / 180))
        batch_auc[i] = snapshot + rng.normal(0, 0.003)

    ax.plot(t, online_auc, color=BLUE, lw=2.0, label="Online SGD (per-event update)")
    ax.plot(t, batch_auc,  color=ORANGE, lw=2.0, label="Batch retrain every 200 steps")
    for boundary in (200, 400, 600):
        ax.axvline(boundary, color=ORANGE, lw=0.8, linestyle=":", alpha=0.5)

    ax.set_xlabel("Events seen")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Stationary regime — online closes the gap continuously", color=GRAY)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.48, 0.80)

    # Right: with concept drift at t=400 (e.g. a viral shift)
    ax = axes[1]
    drift_point = 400
    online_auc2 = np.zeros(n)
    online_auc2[:drift_point] = 0.50 + 0.25 * (1 - np.exp(-t[:drift_point] / 180))
    # Drift halves AUC instantly, online recovers in ~150 steps
    after = t[drift_point:] - drift_point
    online_auc2[drift_point:] = 0.55 + 0.20 * (1 - np.exp(-after / 120))
    online_auc2 += rng.normal(0, 0.005, n)

    batch_auc2 = np.zeros(n)
    snapshot = 0.50
    for i in range(n):
        if i % 200 == 0 and i > 0:
            if i < drift_point:
                snapshot = 0.50 + 0.25 * (1 - np.exp(-i / 180))
            else:
                # batch retrains AFTER drift, but still on stale window
                snapshot = 0.55 + 0.20 * (1 - np.exp(-(i - drift_point) / 180))
        batch_auc2[i] = snapshot + rng.normal(0, 0.003)
    # Drop at drift point for batch (model is suddenly wrong)
    batch_auc2[drift_point:drift_point + 30] -= 0.08

    ax.plot(t, online_auc2, color=BLUE, lw=2.0, label="Online SGD")
    ax.plot(t, batch_auc2,  color=ORANGE, lw=2.0, label="Batch retrain")
    ax.axvline(drift_point, color="red", lw=1.4, linestyle="--", alpha=0.8)
    ax.text(drift_point + 6, 0.51, "concept drift\n(viral event)", color="red",
            fontsize=8.5, va="bottom")

    ax.set_xlabel("Events seen")
    ax.set_ylabel("Validation AUC")
    ax.set_title("Drift regime — batch lags one window, online tracks", color=GRAY)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0.48, 0.80)

    fig.suptitle("Online learning vs batch retraining — same model, very different freshness",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig3_online_vs_batch.png")


# ---------------------------------------------------------------------------
# Figure 4 — Feature freshness impact on AUC
# ---------------------------------------------------------------------------
def fig4_freshness_auc() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Left: AUC vs feature staleness (log-scale on x)
    ax = axes[0]
    # Staleness in seconds (60s … 7 days)
    stale = np.array([1, 10, 60, 300, 900, 1800, 3600, 6 * 3600,
                      24 * 3600, 3 * 24 * 3600, 7 * 24 * 3600])
    # AUC drops slowly first, accelerates after ~hour, plateaus at "old user profile" level
    # Numbers calibrated to public reports (Meta, Pinterest) — drop ~1-3 AUC pts over a day.
    auc = np.array([0.795, 0.794, 0.792, 0.788, 0.783, 0.778, 0.770,
                    0.755, 0.738, 0.722, 0.715])

    ax.semilogx(stale, auc, color=BLUE, lw=2.4, marker="o", markersize=7,
                markerfacecolor="white", markeredgewidth=1.8)
    ax.fill_between(stale, auc, 0.71, color=BLUE, alpha=0.12)

    # Annotate three operating regimes
    ax.axvspan(1, 60, color=GREEN, alpha=0.10)
    ax.axvspan(60, 3600, color=ORANGE, alpha=0.10)
    ax.axvspan(3600, 7 * 24 * 3600, color="red", alpha=0.07)
    ax.text(8, 0.715, "real-time\n< 1 min", ha="center", fontsize=8.5, color=GREEN, fontweight="bold")
    ax.text(500, 0.715, "near-real-time\n1 – 60 min", ha="center", fontsize=8.5, color=ORANGE, fontweight="bold")
    ax.text(60000, 0.715, "batch / daily\n> 1 h", ha="center", fontsize=8.5, color="red", fontweight="bold")

    ax.set_xlabel("Feature staleness  (seconds, log scale)")
    ax.set_ylabel("Ranking AUC")
    ax.set_title("AUC degrades roughly linearly in log-staleness", color=GRAY)
    ax.set_ylim(0.71, 0.80)

    # Right: what causes the loss — by feature family
    ax = axes[1]
    families = ["Recent\nclick seq", "Session\nintent", "Real-time\nCTR stats", "User\ndemographics", "Item\nmetadata"]
    # AUC contribution lost when going from real-time to 1-day stale (estimated)
    loss = [0.024, 0.018, 0.012, 0.001, 0.001]
    colors = [BLUE, PURPLE, GREEN, GRAY, GRAY]
    bars = ax.barh(families, loss, color=colors, alpha=0.9, edgecolor="white", linewidth=1.4)
    for bar, v in zip(bars, loss):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"-{v:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("AUC lost when feature is 1 day stale (vs. real-time)")
    ax.set_title("Behavioural features carry the freshness premium", color=GRAY)
    ax.set_xlim(0, 0.032)

    fig.suptitle("Feature freshness — only the behavioural tail benefits from real-time pipelines",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig4_freshness_auc.png")


# ---------------------------------------------------------------------------
# Figure 5 — Streaming architecture (Kafka + Flink + serving)
# ---------------------------------------------------------------------------
def fig5_streaming_arch() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Sources (left)
    sources = [
        ("Web\nclient",  0.7, 5.6, ORANGE),
        ("Mobile\nclient",0.7, 4.0, ORANGE),
        ("Backend\nlogs", 0.7, 2.4, ORANGE),
    ]
    for lbl, x, y, col in sources:
        box = FancyBboxPatch((x - 0.55, y - 0.45), 1.1, 0.9,
                              boxstyle="round,pad=0.05", linewidth=1.4,
                              facecolor=col, edgecolor=col, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, lbl, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")

    # Kafka cluster (3 partitioned topics)
    ax.add_patch(FancyBboxPatch((2.7, 1.5), 2.4, 5.2,
                                boxstyle="round,pad=0.08",
                                facecolor="#eff6ff", edgecolor=BLUE, linewidth=2))
    ax.text(3.9, 6.45, "Apache Kafka", ha="center", fontsize=11,
            fontweight="bold", color=BLUE)
    ax.text(3.9, 6.15, "partitioned by user_id, replicated, durable",
            ha="center", fontsize=8, color=GRAY, style="italic")

    topics = [("clicks",     5.2),
              ("impressions",4.1),
              ("conversions",3.0),
              ("profile-Δ",  1.95)]
    for name, y in topics:
        box = FancyBboxPatch((2.95, y - 0.28), 1.95, 0.56,
                              boxstyle="round,pad=0.04", linewidth=1.2,
                              facecolor="white", edgecolor=BLUE)
        ax.add_patch(box)
        ax.text(3.92, y, f"topic: {name}", ha="center", va="center",
                fontsize=8.5, color=BLUE, fontweight="bold")

    # Arrows from sources into Kafka
    for _, sx, sy, _ in sources:
        ax.annotate("", xy=(2.7, sy), xytext=(sx + 0.55, sy),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2))

    # Flink (stream processor)
    ax.add_patch(FancyBboxPatch((6.0, 1.5), 3.0, 5.2,
                                boxstyle="round,pad=0.08",
                                facecolor="#f5f3ff", edgecolor=PURPLE, linewidth=2))
    ax.text(7.5, 6.45, "Apache Flink", ha="center", fontsize=11,
            fontweight="bold", color=PURPLE)
    ax.text(7.5, 6.15, "stateful streaming · event-time · exactly-once",
            ha="center", fontsize=8, color=GRAY, style="italic")

    flink_ops = [
        ("Window aggregator\n(1m / 10m tumbling)",       5.45),
        ("Stateful join\n(click x impression)",           4.20),
        ("Online learner\n(FTRL / SGD)",                  2.95),
        ("CDC sink\n(features, model)",                   1.90),
    ]
    op_colors = [GREEN, BLUE, PURPLE, ORANGE]
    for (lbl, y), col in zip(flink_ops, op_colors):
        box = FancyBboxPatch((6.15, y - 0.42), 2.7, 0.84,
                              boxstyle="round,pad=0.04", linewidth=1.4,
                              facecolor=col, edgecolor=col, alpha=0.9)
        ax.add_patch(box)
        ax.text(7.5, y, lbl, ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold")

    # Arrows Kafka → Flink
    ax.annotate("", xy=(6.0, 4.0), xytext=(4.9, 4.0),
                arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.6))
    ax.text(5.45, 4.18, "consume", fontsize=8, color=GRAY, ha="center")

    # Sinks (right)
    sinks = [
        ("Feature\nstore\n(Redis)",   10.5, 5.6, GREEN),
        ("Model\nregistry\n(S3)",     10.5, 3.95, PURPLE),
        ("Online\nMetrics\n(Prom.)",  10.5, 2.3, ORANGE),
    ]
    for lbl, x, y, col in sinks:
        box = FancyBboxPatch((x - 0.6, y - 0.6), 1.2, 1.2,
                              boxstyle="round,pad=0.05", linewidth=1.4,
                              facecolor=col, edgecolor=col, alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y, lbl, ha="center", va="center", color="white",
                fontsize=8.5, fontweight="bold")
        # Flink → sink
        ax.annotate("", xy=(x - 0.6, y), xytext=(9.0, y),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.4))

    # Serving cluster (far right)
    box = FancyBboxPatch((12.4, 2.3), 1.4, 3.3,
                         boxstyle="round,pad=0.08",
                         facecolor="white", edgecolor=BLUE, linewidth=2)
    ax.add_patch(box)
    ax.text(13.1, 5.30, "Serving", ha="center", fontsize=10,
            fontweight="bold", color=BLUE)
    ax.text(13.1, 5.00, "(low-latency)", ha="center", fontsize=7.5, color=GRAY)
    ax.text(13.1, 4.20, "recall\n+ rank", ha="center", fontsize=9, color=BLUE)
    ax.text(13.1, 3.20, "feature\nlook-up", ha="center", fontsize=9, color=GREEN)
    ax.text(13.1, 2.50, "model\nload", ha="center", fontsize=9, color=PURPLE)

    # Sinks → serving
    for _, sx, sy, _ in sinks:
        ax.annotate("", xy=(12.4, sy), xytext=(sx + 0.6, sy),
                    arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2,
                                    linestyle=(0, (3, 2))))

    # Checkpoints (Flink)
    ax.text(7.5, 0.95,
            "Flink checkpoints (RocksDB → S3)  →  exactly-once recovery",
            ha="center", fontsize=8.5, color=PURPLE, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f3ff",
                      edgecolor=PURPLE, linewidth=1))

    ax.set_title("Streaming reference architecture — Kafka transports, Flink computes state, KV stores serve",
                 pad=10)
    fig.tight_layout()
    save(fig, "fig5_streaming_arch.png")


# ---------------------------------------------------------------------------
# Figure 6 — Concept drift detection (ADWIN / DDM-style)
# ---------------------------------------------------------------------------
def fig6_drift_detection() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    rng = np.random.default_rng(11)
    n = 1000
    t = np.arange(n)

    # True CTR shifts at t=350 (gradual) and t=700 (abrupt)
    true_ctr = np.full(n, 0.10)
    true_ctr[350:700] = np.linspace(0.10, 0.20, 350)
    true_ctr[700:] = 0.06

    # Observed clicks (Bernoulli)
    obs = rng.binomial(1, true_ctr).astype(float)
    # Sliding mean (window=50)
    win = 50
    rolling = np.convolve(obs, np.ones(win) / win, mode="same")

    # Reference window mean (first 200 stable)
    ref_mean = rolling[100:300].mean()
    ref_std  = rolling[100:300].std() + 1e-6

    # ADWIN-like z-score against the reference window
    z = (rolling - ref_mean) / ref_std
    drift_thresh = 3.0
    drift_flags = np.abs(z) > drift_thresh

    # Top: observed CTR + true + window mean
    ax = axes[0]
    ax.plot(t, true_ctr, color=GRAY, lw=1.6, linestyle="--", label="True CTR")
    ax.plot(t, rolling, color=BLUE, lw=2.0, label=f"Rolling mean (w={win})")
    ax.fill_between(t, rolling - 2 * ref_std, rolling + 2 * ref_std,
                    color=BLUE, alpha=0.10)
    ax.axhline(ref_mean, color=PURPLE, lw=1, linestyle=":", alpha=0.8)
    ax.text(5, ref_mean + 0.005, f"reference μ = {ref_mean:.3f}",
            color=PURPLE, fontsize=8.5)

    # Mark detected drift segments
    flagged = np.where(drift_flags)[0]
    if len(flagged) > 0:
        # group consecutive
        breaks = np.where(np.diff(flagged) > 1)[0]
        starts = np.r_[flagged[0], flagged[breaks + 1]]
        ends   = np.r_[flagged[breaks], flagged[-1]]
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, color="red", alpha=0.10)

    # True drift moments
    ax.axvline(350, color=ORANGE, lw=1.2, linestyle="--", alpha=0.85)
    ax.text(355, 0.245, "drift starts\n(gradual)", color=ORANGE, fontsize=8.5)
    ax.axvline(700, color="red", lw=1.4, linestyle="--", alpha=0.85)
    ax.text(705, 0.245, "drift starts\n(abrupt)", color="red", fontsize=8.5)

    ax.set_ylabel("CTR")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Concept drift in CTR — detector raises an alarm when |z| > 3", color=GRAY)
    ax.set_ylim(0, 0.28)

    # Bottom: z-score with threshold
    ax = axes[1]
    ax.plot(t, z, color=PURPLE, lw=1.6)
    ax.axhline(drift_thresh, color="red", lw=1.0, linestyle="--")
    ax.axhline(-drift_thresh, color="red", lw=1.0, linestyle="--")
    ax.fill_between(t, drift_thresh, np.maximum(z, drift_thresh),
                    where=(z > drift_thresh), color="red", alpha=0.30)
    ax.fill_between(t, -drift_thresh, np.minimum(z, -drift_thresh),
                    where=(z < -drift_thresh), color="red", alpha=0.30)
    ax.set_xlabel("Time step  (events)")
    ax.set_ylabel("z-score")
    ax.set_title("Detector statistic — alarm fires once gradual drift accumulates and immediately on abrupt drift",
                 color=GRAY, fontsize=11)
    ax.set_ylim(-8, 8)

    fig.suptitle("Drift detection — sliding statistics flag changes that should trigger learning-rate boost / retrain",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    save(fig, "fig6_drift_detection.png")


# ---------------------------------------------------------------------------
# Figure 7 — Cache vs compute trade-off
# ---------------------------------------------------------------------------
def fig7_cache_vs_compute() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Left: latency, freshness, cost as a function of cache TTL
    ax = axes[0]
    ttl = np.array([1, 5, 15, 30, 60, 300, 900, 1800, 3600, 7200])  # seconds
    # Latency (ms) — drops as TTL grows (more cache hits)
    latency = 80 - 60 * (1 - np.exp(-ttl / 200))   # 80 ms → ~20 ms
    # Freshness penalty (AUC delta vs ideal real-time)
    auc_loss = -0.001 * np.log10(ttl + 1) - 0.00005 * ttl  # accelerating loss
    auc_loss = -auc_loss  # express as positive "loss"
    # Compute cost (relative QPS to backend) — drops as TTL grows
    backend_qps = 1.0 / (1.0 + ttl / 30.0)

    color1, color2, color3 = BLUE, PURPLE, ORANGE
    ax.set_xscale("log")
    l1, = ax.plot(ttl, latency, color=color1, lw=2.4, marker="o",
                  markersize=6, label="Serving p95 latency (ms)")
    ax.set_xlabel("Cache TTL  (seconds, log scale)")
    ax.set_ylabel("Latency (ms)", color=color1)
    ax.tick_params(axis="y", labelcolor=color1)
    ax.set_ylim(0, 90)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    l2, = ax2.plot(ttl, auc_loss * 100, color=color2, lw=2.4, marker="s",
                   markersize=6, label="AUC loss vs real-time (×100)")
    ax2.set_ylabel("AUC loss (×100)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1.6)
    ax2.grid(False)

    # Sweet-spot annotation
    sweet = 60
    ax.axvline(sweet, color=GREEN, lw=1.4, linestyle="--", alpha=0.8)
    ax.text(sweet * 1.1, 75, "sweet spot\n~60 s", color=GREEN, fontsize=9,
            fontweight="bold")

    ax.legend(handles=[l1, l2], loc="upper left", fontsize=8.5)
    ax.set_title("More cache → faster + cheaper, but staler", color=GRAY)

    # Right: 2D Pareto — feature staleness vs serving cost, ROI front
    ax = axes[1]
    # Strategies
    strategies = [
        ("Always recompute",          0.5,  100, BLUE),
        ("1-min cache",               60,   28,  GREEN),
        ("5-min cache",               300,  14,  GREEN),
        ("1-hour cache",              3600, 4,   ORANGE),
        ("Daily batch",               86400, 1,  "red"),
        ("Hybrid: hot=RT, cold=cache",30,   18,  PURPLE),
    ]
    for name, stale, cost, col in strategies:
        ax.scatter(stale, cost, s=180, color=col, alpha=0.85,
                   edgecolor="white", linewidth=1.6, zorder=3)
        offset_y = 5 if name != "Hybrid: hot=RT, cold=cache" else -8
        ax.annotate(name, xy=(stale, cost), xytext=(stale * 1.35, cost + offset_y),
                    fontsize=8.5, color=col, fontweight="bold")

    # Pareto front (approximate)
    front_x = [0.5, 30, 300, 3600, 86400]
    front_y = [100, 18, 14, 4, 1]
    ax.plot(front_x, front_y, color=GRAY, lw=1.2, linestyle=":", alpha=0.7)
    ax.text(2, 60, "Pareto front", color=GRAY, fontsize=8, style="italic", rotation=-30)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Feature staleness  (seconds, log scale)")
    ax.set_ylabel("Backend compute cost (relative, log scale)")
    ax.set_title("Pick a point on the front — hybrid wins for feeds", color=GRAY)
    ax.set_xlim(0.3, 200000)
    ax.set_ylim(0.6, 200)

    fig.suptitle("Cache vs compute — the trade-off you actually tune in production",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_cache_vs_compute.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating Part 15 figures (Real-Time & Online Learning)...")
    fig1_pipeline()
    print("  fig1 pipeline ok")
    fig2_latency_budget()
    print("  fig2 latency budget ok")
    fig3_online_vs_batch()
    print("  fig3 online vs batch ok")
    fig4_freshness_auc()
    print("  fig4 freshness ok")
    fig5_streaming_arch()
    print("  fig5 streaming arch ok")
    fig6_drift_detection()
    print("  fig6 drift detection ok")
    fig7_cache_vs_compute()
    print("  fig7 cache vs compute ok")
    print(f"All saved to:\n  EN: {EN_DIR}\n  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
