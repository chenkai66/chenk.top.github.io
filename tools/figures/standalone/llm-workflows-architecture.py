"""Figures for "LLM Workflows and Application Architecture".

Generates 7 publication-quality figures that anchor the standalone
article on production-grade LLM applications. Each figure is saved into
both the English and Chinese asset folders.

Figures:
    fig1_application_stack    — Layered LLM application stack (UI/Agent/RAG/Model/Tools)
    fig2_workflow_patterns    — Orchestration patterns: chain, branch, loop, parallel
    fig3_rag_vs_finetuning    — Decision tree: RAG vs fine-tuning vs prompt engineering
    fig4_production_deploy    — Production deployment topology with cache + LLM gateway
    fig5_cost_optimization    — Cost levers: caching, routing, batching, quantization
    fig6_observability        — Logs / metrics / traces stack with golden signals
    fig7_enterprise_patterns  — Enterprise integration patterns (SSO, RBAC, audit, BYO-K)

Style: seaborn-v0_8-whitegrid, dpi=150, palette
    #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (orange).

References (verified):
    - Lewis et al., Retrieval-Augmented Generation for Knowledge-Intensive NLP (NeurIPS 2020)
    - Brown et al., Language Models are Few-Shot Learners (NeurIPS 2020)
    - Bai et al., Constitutional AI (Anthropic, 2022)
    - Schulhoff et al., The Prompt Report (2024)
    - Liu et al., Lost in the Middle: How Language Models Use Long Contexts (TACL 2024)
    - Anthropic, Building Effective Agents (engineering blog, 2024)
    - Karpukhin et al., Dense Passage Retrieval (EMNLP 2020)
    - Beirne et al., Site Reliability Engineering (Google, 2016) — golden signals
    - Sigelman et al., Dapper (Google Tech Report, 2010) — distributed tracing
    - Frantar et al., GPTQ: Accurate Post-Training Quantization (ICLR 2023)
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

BLUE = COLORS["primary"]
PURPLE = COLORS["accent"]
GREEN = COLORS["success"]
ORANGE = COLORS["warning"]
GRAY = COLORS["text2"]
LIGHT = COLORS["grid"]
DARK = "#1f2937"
RED = COLORS["danger"]

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/standalone/llm-workflows-architecture"
ZH_DIR = REPO / "source/_posts/zh/standalone/llm工作流与应用架构-企业级实战指南"


def save(fig: plt.Figure, name: str) -> None:
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def _rounded(ax, x, y, w, h, text, edge, fc, *, fs=10, weight="normal", tc=None):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.4, edgecolor=edge, facecolor=fc, alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, weight=weight,
        color=tc if tc else DARK,
    )


def _arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.6, style="->", mut=14):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=mut, linewidth=lw, color=color,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1 — Application stack
# ---------------------------------------------------------------------------
def fig1_application_stack() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.text(6.25, 6.7, "LLM Application Stack — five layers, one request path",
            ha="center", fontsize=14, weight="bold")

    layers = [
        ("Experience layer",       "Web · Mobile · IDE plugin · Slack/Teams bot",         BLUE,   "#dbeafe", 5.55),
        ("Agent / orchestration",  "Planner · tool-router · memory · guardrails",         PURPLE, "#ede9fe", 4.45),
        ("Retrieval & context",    "RAG · semantic cache · session store · prompt builder", GREEN,  "#d1fae5", 3.35),
        ("Model serving",          "LLM gateway · routing · fallback · streaming",        ORANGE, "#fef3c7", 2.25),
        ("Tools & data plane",     "SQL · search · code-exec · APIs · vector DB",         GRAY,   "#f1f5f9", 1.15),
    ]
    for title, sub, edge, fc, y in layers:
        _rounded(ax, 0.6, y, 11.3, 0.95, "", edge, fc)
        ax.text(0.95, y + 0.62, title, fontsize=11.5, weight="bold", color=DARK)
        ax.text(0.95, y + 0.28, sub, fontsize=9.5, color=DARK)

    # Cross-cutting concerns column on the right
    _rounded(ax, 9.7, 1.15, 2.2, 5.35, "", DARK, "white")
    ax.text(10.8, 6.30, "Cross-cutting", ha="center", fontsize=10, weight="bold", color=DARK)
    items = ["AuthN / AuthZ", "Audit & PII redact", "Rate limit / quota",
             "Observability", "Cost & budget", "Eval & A/B"]
    for i, it in enumerate(items):
        ax.text(10.8, 5.85 - i * 0.6, "• " + it, ha="center", fontsize=9, color=DARK)

    # Request flow arrow
    _arrow(ax, 0.3, 6.0, 0.3, 1.6, color=BLUE, lw=2.2)
    ax.text(0.15, 3.8, "request", rotation=90, ha="center", va="center",
            fontsize=9, color=BLUE, weight="bold")

    ax.text(6.25, 0.45,
            "A request descends the stack; only the model layer is non-deterministic — "
            "everything else is plain distributed systems.",
            ha="center", fontsize=9, style="italic", color=GRAY)

    save(fig, "fig1_application_stack.png")


# ---------------------------------------------------------------------------
# Figure 2 — Workflow orchestration patterns
# ---------------------------------------------------------------------------
def fig2_workflow_patterns() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0))
    fig.suptitle("Orchestration patterns — pick the smallest one that works",
                 fontsize=14, weight="bold", y=0.98)

    # --- Chain (sequential) ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Chain — sequential pipeline", color=BLUE)
    nodes = [("Input", LIGHT), ("Step A", "#dbeafe"), ("Step B", "#dbeafe"), ("Step C", "#dbeafe"), ("Output", LIGHT)]
    for i, (name, fc) in enumerate(nodes):
        _rounded(ax, 0.2 + i * 2.0, 1.6, 1.6, 0.9, name, BLUE, fc, fs=10)
        if i < len(nodes) - 1:
            _arrow(ax, 1.85 + i * 2.0, 2.05, 2.15 + i * 2.0, 2.05, color=BLUE)
    ax.text(5.0, 0.6,
            "Use when steps are deterministic and ordered\n(extract → transform → summarise).",
            ha="center", fontsize=9.2, color=GRAY)

    # --- Branch (router) ---
    ax = axes[0, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Branch — classify then route", color=PURPLE)
    _rounded(ax, 0.4, 1.6, 1.7, 0.9, "Input", PURPLE, "#ede9fe")
    _rounded(ax, 2.7, 1.6, 1.9, 0.9, "Router\nLLM", PURPLE, "#ede9fe", weight="bold")
    branches = [("FAQ → small LM", 3.1, GREEN),
                ("Code → tool-use", 2.05, ORANGE),
                ("Reasoning → big LM", 1.0, RED)]
    for label, y, c in branches:
        _rounded(ax, 5.6, y - 0.45, 3.8, 0.9, label, c, "white", fs=9.5)
        _arrow(ax, 4.65, 2.05, 5.55, y, color=c)
    _arrow(ax, 2.15, 2.05, 2.65, 2.05, color=PURPLE)
    ax.text(5.0, 0.3,
            "Use when input distribution is mixed —\nsave 60–80 % cost vs always-big-LM.",
            ha="center", fontsize=9.2, color=GRAY)

    # --- Loop (reflect) ---
    ax = axes[1, 0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Loop — generate → critique → revise", color=GREEN)
    _rounded(ax, 0.4, 1.6, 1.7, 0.9, "Input", GREEN, "#d1fae5")
    _rounded(ax, 2.7, 1.6, 1.9, 0.9, "Generate", GREEN, "#d1fae5")
    _rounded(ax, 5.2, 1.6, 1.9, 0.9, "Critique", GREEN, "#d1fae5")
    _rounded(ax, 7.7, 1.6, 1.9, 0.9, "Output", GREEN, "#d1fae5")
    _arrow(ax, 2.15, 2.05, 2.65, 2.05, color=GREEN)
    _arrow(ax, 4.65, 2.05, 5.15, 2.05, color=GREEN)
    _arrow(ax, 7.15, 2.05, 7.65, 2.05, color=GREEN)
    # back-edge
    arr = FancyArrowPatch((6.1, 2.55), (3.7, 2.55),
                          connectionstyle="arc3,rad=-0.45",
                          arrowstyle="->", mutation_scale=14,
                          linewidth=1.6, color=ORANGE)
    ax.add_patch(arr)
    ax.text(4.9, 3.35, "if score < τ: revise (≤ N)", ha="center", fontsize=9, color=ORANGE, weight="bold")
    ax.text(5.0, 0.55,
            "Bound the loop with a budget (N≤3) and a hard tokens cap.",
            ha="center", fontsize=9.2, color=GRAY)

    # --- Parallel + reduce ---
    ax = axes[1, 1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_title("Parallel — fan-out, reduce", color=ORANGE)
    _rounded(ax, 0.4, 1.6, 1.7, 0.9, "Input", ORANGE, "#fef3c7")
    for i, y in enumerate([3.0, 2.05, 1.1]):
        _rounded(ax, 3.4, y - 0.45, 2.4, 0.9, f"Worker {i+1}", ORANGE, "#fef3c7", fs=9.5)
        _arrow(ax, 2.15, 2.05, 3.35, y, color=ORANGE)
        _arrow(ax, 5.85, y, 6.95, 2.05, color=ORANGE)
    _rounded(ax, 7.0, 1.6, 2.6, 0.9, "Reduce / vote", ORANGE, "#fef3c7", weight="bold")
    ax.text(5.0, 0.55,
            "Use for chunked summarisation, self-consistency,\nor multi-source synthesis.",
            ha="center", fontsize=9.2, color=GRAY)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig2_workflow_patterns.png")


# ---------------------------------------------------------------------------
# Figure 3 — RAG vs fine-tuning vs prompt engineering decision tree
# ---------------------------------------------------------------------------
def fig3_rag_vs_finetuning() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.6))
    ax.set_xlim(0, 13.0); ax.set_ylim(0, 7.6); ax.axis("off")

    ax.text(6.5, 7.25, "Adapt an LLM to your task — pick the cheapest tool that closes the gap",
            ha="center", fontsize=14, weight="bold")

    # Root
    _rounded(ax, 5.0, 6.05, 3.0, 0.7, "Start: baseline LLM\nis insufficient", DARK, "white", weight="bold", fs=10)

    # Q1: knowledge gap?
    _rounded(ax, 5.0, 4.85, 3.0, 0.7, "Missing knowledge?", BLUE, "#dbeafe", weight="bold")
    _arrow(ax, 6.5, 6.05, 6.5, 5.55, color=BLUE)

    # Left branch: behavior gap (no knowledge missing)
    _rounded(ax, 0.6, 3.55, 3.0, 0.7, "Wrong style/format?", PURPLE, "#ede9fe", weight="bold")
    _arrow(ax, 5.0, 5.20, 3.6, 3.90, color=PURPLE)
    ax.text(4.0, 4.65, "no", fontsize=9, color=PURPLE, weight="bold")

    _rounded(ax, 0.2, 1.95, 3.6, 1.05,
             "Prompt engineering\n• few-shot · CoT · structured output\n• cheapest, instant, version-controlled",
             PURPLE, "#ede9fe", fs=9.2)
    _arrow(ax, 2.1, 3.55, 2.0, 3.05, color=PURPLE)

    _rounded(ax, 0.2, 0.30, 3.6, 1.30,
             "Fine-tuning (SFT / LoRA / DPO)\n• fixed style, tone, schemas\n• needs ≥ 10³–10⁴ labelled examples\n• only after prompts are exhausted",
             ORANGE, "#fef3c7", fs=9.2)
    _arrow(ax, 1.5, 1.95, 1.5, 1.65, color=ORANGE)
    ax.text(2.85, 2.45, "still wrong", fontsize=8.5, color=ORANGE)

    # Right branch: knowledge gap → RAG
    _rounded(ax, 9.4, 3.55, 3.2, 0.7, "Knowledge changes\nfrequently?", GREEN, "#d1fae5", weight="bold")
    _arrow(ax, 8.0, 5.20, 9.4, 3.90, color=GREEN)
    ax.text(9.0, 4.65, "yes", fontsize=9, color=GREEN, weight="bold")

    _rounded(ax, 9.4, 1.95, 3.2, 1.05,
             "RAG\n• cite sources, refresh in minutes\n• vector + BM25 hybrid recommended",
             GREEN, "#d1fae5", fs=9.2)
    _arrow(ax, 11.0, 3.55, 11.0, 3.05, color=GREEN)

    _rounded(ax, 9.4, 0.30, 3.2, 1.30,
             "RAG + fine-tune\n• stable corpus, narrow domain\n• fine-tune retriever or reader\n• highest quality, highest cost",
             BLUE, "#dbeafe", fs=9.2)
    _arrow(ax, 11.0, 1.95, 11.0, 1.65, color=BLUE)
    ax.text(12.6, 2.45, "no", fontsize=8.5, color=BLUE)

    # Bottom legend / heuristic table
    ax.text(6.5, 5.15,
            "Rule of thumb: prompts < retrieval < fine-tuning  in both effort and risk.",
            ha="center", fontsize=9.5, color=GRAY, style="italic")

    save(fig, "fig3_rag_vs_finetuning.png")


# ---------------------------------------------------------------------------
# Figure 4 — Production deployment topology
# ---------------------------------------------------------------------------
def fig4_production_deploy() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.5))
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 7.5); ax.axis("off")

    ax.text(6.75, 7.2, "Production deployment — gateway, cache, and model fleet",
            ha="center", fontsize=14, weight="bold")

    # Clients
    _rounded(ax, 0.2, 5.6, 1.8, 0.9, "Clients\n(web · API)", DARK, "white", fs=9.5)

    # Edge
    _rounded(ax, 2.4, 5.6, 1.9, 0.9, "CDN + WAF\n+ rate limit", BLUE, "#dbeafe", fs=9.5)
    _arrow(ax, 2.05, 6.05, 2.35, 6.05, color=BLUE)

    # API gateway / load balancer
    _rounded(ax, 4.7, 5.6, 2.2, 0.9, "API gateway\nauthN · quota · audit", BLUE, "#dbeafe", fs=9.5, weight="bold")
    _arrow(ax, 4.35, 6.05, 4.65, 6.05, color=BLUE)

    # App servers (k8s)
    _rounded(ax, 7.3, 5.6, 2.6, 0.9, "App servers (HPA)\nFastAPI / Node — N pods", PURPLE, "#ede9fe", fs=9.5, weight="bold")
    _arrow(ax, 6.95, 6.05, 7.25, 6.05, color=PURPLE)

    # Semantic cache
    _rounded(ax, 10.4, 5.6, 2.8, 0.9, "Semantic cache (Redis)\nexact + embedding hit", GREEN, "#d1fae5", fs=9.5, weight="bold")
    _arrow(ax, 9.95, 6.05, 10.35, 6.05, color=GREEN)

    # LLM gateway
    _rounded(ax, 4.7, 3.7, 4.4, 1.0,
             "LLM gateway — routing · fallback · streaming · token budget",
             ORANGE, "#fef3c7", fs=10, weight="bold")
    _arrow(ax, 8.6, 5.55, 7.6, 4.75, color=ORANGE)

    # Model fleet
    _rounded(ax, 0.4, 1.6, 2.3, 1.0, "Small LM\n(local · vLLM)\nGPU pool A", BLUE, "#dbeafe", fs=9)
    _rounded(ax, 3.0, 1.6, 2.3, 1.0, "Mid LM\n(open-weight)\nGPU pool B", BLUE, "#dbeafe", fs=9)
    _rounded(ax, 5.6, 1.6, 2.3, 1.0, "Frontier API\n(Claude / GPT)\nexternal", PURPLE, "#ede9fe", fs=9)
    _rounded(ax, 8.2, 1.6, 2.3, 1.0, "Code / tool\nsandbox\n(Firecracker)", GRAY, "#f1f5f9", fs=9)
    _rounded(ax, 10.8, 1.6, 2.3, 1.0, "Vector DB\n+ BM25\n(retrieval)", GREEN, "#d1fae5", fs=9)
    for x in (1.55, 4.15, 6.75, 9.35, 11.95):
        _arrow(ax, 6.9, 3.65, x, 2.65, color=ORANGE, lw=1.2)

    # Async track at bottom
    _rounded(ax, 0.4, 0.15, 12.7, 0.9,
             "Async track — message queue (Kafka/SQS) · workers · embeddings ingest · evals · cost rollups",
             DARK, "#f1f5f9", fs=9.5)

    # Observability strip
    _rounded(ax, 11.5, 3.7, 1.8, 1.0, "Telemetry\nlogs · metrics\ntraces", DARK, "white", fs=8.5)
    _arrow(ax, 9.15, 4.20, 11.45, 4.20, color=GRAY, style="-")

    save(fig, "fig4_production_deploy.png")


# ---------------------------------------------------------------------------
# Figure 5 — Cost optimization
# ---------------------------------------------------------------------------
def fig5_cost_optimization() -> None:
    fig = plt.figure(figsize=(13.0, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.0], wspace=0.25)

    # Left: stacked bar showing cumulative cost reduction
    ax = fig.add_subplot(gs[0, 0])
    stages = ["Baseline", "+ prompt\ncompression", "+ small-model\nrouting",
              "+ semantic\ncache", "+ batching", "+ INT8\nquantization"]
    # cost remaining as a fraction of baseline (illustrative, plausible upper-bounds)
    remaining = [1.00, 0.78, 0.42, 0.27, 0.21, 0.16]
    saved = [1 - r for r in remaining]

    x = np.arange(len(stages))
    ax.bar(x, remaining, color=BLUE, alpha=0.85, label="cost remaining")
    ax.bar(x, saved, bottom=remaining, color=GREEN, alpha=0.55, label="cost removed")
    for i, r in enumerate(remaining):
        ax.text(i, r + 0.02, f"{r*100:.0f}%", ha="center", fontsize=9.5, weight="bold", color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Spend vs baseline")
    ax.set_title("Cumulative cost levers (typical envelope)", color=DARK)
    ax.legend(loc="upper right", fontsize=9)
    ax.text(2.5, -0.18,
            "Numbers are an upper-bound envelope — measure on your traffic before promising savings.",
            ha="center", fontsize=8.5, color=GRAY, style="italic", transform=ax.transData)

    # Right: scatter — quality vs cost for model tiers
    ax = fig.add_subplot(gs[0, 1])
    tiers = [
        ("7B local",      0.18, 0.62, BLUE),
        ("8B INT8",       0.10, 0.58, GREEN),
        ("Mid open-wt",   0.55, 0.78, ORANGE),
        ("Frontier API",  1.00, 0.93, PURPLE),
        ("Frontier + RAG",1.05, 0.96, RED),
    ]
    for name, cost, qual, color in tiers:
        ax.scatter(cost, qual, s=320, color=color, alpha=0.85, edgecolor=DARK, linewidth=1.2)
        ax.text(cost, qual + 0.025, name, ha="center", fontsize=9, weight="bold")
    ax.set_xlabel("Relative$/1k requests")
    ax.set_ylabel("Task quality (eval score)")
    ax.set_xlim(-0.05, 1.20)
    ax.set_ylim(0.50, 1.02)
    ax.set_title("Quality / cost frontier — route per request", color=DARK)
    ax.axhline(0.85, color=GRAY, linestyle="--", linewidth=1)
    ax.text(1.15, 0.86, "SLA bar", ha="right", fontsize=8.5, color=GRAY)

    fig.suptitle("Cost optimisation — stack the levers, route on the frontier",
                 fontsize=14, weight="bold", y=1.02)
    save(fig, "fig5_cost_optimization.png")


# ---------------------------------------------------------------------------
# Figure 6 — Observability stack
# ---------------------------------------------------------------------------
def fig6_observability() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.2))
    ax.set_xlim(0, 13.0); ax.set_ylim(0, 7.2); ax.axis("off")

    ax.text(6.5, 6.9, "Observability — logs, metrics, traces, plus LLM-specific signals",
            ha="center", fontsize=14, weight="bold")

    # Three pillars
    pillars = [
        ("Logs",    "request · prompt (hashed)\nresponse · errors",   BLUE,   "#dbeafe", 0.4),
        ("Metrics", "QPS · p50/p95/p99\ntoken rate · cost / req",     GREEN,  "#d1fae5", 4.7),
        ("Traces",  "span tree\n(retrieve · rerank · LLM · tool)",   PURPLE, "#ede9fe", 9.0),
    ]
    for title, sub, edge, fc, x in pillars:
        _rounded(ax, x, 4.5, 3.6, 1.5, "", edge, fc)
        ax.text(x + 1.8, 5.55, title, ha="center", fontsize=12, weight="bold", color=DARK)
        ax.text(x + 1.8, 5.00, sub, ha="center", fontsize=9.5, color=DARK)

    # LLM-specific row
    _rounded(ax, 0.4, 2.7, 12.2, 1.4, "", ORANGE, "#fef3c7")
    ax.text(6.5, 3.85, "LLM-specific signals", ha="center", fontsize=11.5, weight="bold", color=DARK)
    items = [
        "hallucination rate\n(LLM-as-judge)",
        "groundedness\n(citation hit)",
        "guardrail trips\n(injection · PII)",
        "cache hit ratio\n(exact / semantic)",
        "fallback rate\n(per provider)",
    ]
    for i, it in enumerate(items):
        ax.text(1.6 + i * 2.4, 3.10, it, ha="center", fontsize=9, color=DARK)

    # Storage / sinks
    _rounded(ax, 0.4, 0.6, 3.6, 1.4, "Loki / OpenSearch\n(structured logs)", BLUE, "white", fs=9.5)
    _rounded(ax, 4.7, 0.6, 3.6, 1.4, "Prometheus / VictoriaMetrics\n(time series + alerts)", GREEN, "white", fs=9.5)
    _rounded(ax, 9.0, 0.6, 3.6, 1.4, "Tempo / Jaeger\n(OpenTelemetry traces)", PURPLE, "white", fs=9.5)

    for x in (2.2, 6.5, 10.8):
        _arrow(ax, x, 2.65, x, 2.05, color=GRAY)
    for x in (2.2, 6.5, 10.8):
        _arrow(ax, x, 4.45, x, 4.15, color=GRAY)

    ax.text(6.5, 0.20,
            "Golden signals: latency · traffic · errors · saturation. Add token-cost as the fifth.",
            ha="center", fontsize=9, style="italic", color=GRAY)

    save(fig, "fig6_observability.png")


# ---------------------------------------------------------------------------
# Figure 7 — Enterprise integration patterns
# ---------------------------------------------------------------------------
def fig7_enterprise_patterns() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.4))
    ax.set_xlim(0, 13.0); ax.set_ylim(0, 7.4); ax.axis("off")

    ax.text(6.5, 7.1, "Enterprise integration — what changes between MVP and contract-grade",
            ha="center", fontsize=14, weight="bold")

    # Center: the LLM platform
    _rounded(ax, 4.7, 3.0, 3.6, 1.4, "LLM platform\n(your service)", DARK, "white", weight="bold", fs=11)

    patterns = [
        # (label, detail, x, y, color, fc)
        ("Identity",       "SSO · SAML / OIDC\nSCIM provisioning",         0.4, 5.4, BLUE,   "#dbeafe"),
        ("Authorisation",  "RBAC + ABAC\ndoc-level filters",               4.7, 5.4, PURPLE, "#ede9fe"),
        ("Data residency", "regional routing\nEU / US / CN",               9.0, 5.4, GREEN,  "#d1fae5"),
        ("Bring-your-own", "BYO-key (KMS)\nBYO-model endpoint",            0.4, 0.4, ORANGE, "#fef3c7"),
        ("Audit & DLP",    "immutable log\nPII redact · DSAR",             4.7, 0.4, RED,    "#fee2e2"),
        ("Compliance",     "SOC2 · ISO 27001\nHIPAA · GDPR",               9.0, 0.4, BLUE,   "#dbeafe"),
    ]
    for label, detail, x, y, edge, fc in patterns:
        _rounded(ax, x, y, 3.6, 1.4, "", edge, fc)
        ax.text(x + 1.8, y + 0.95, label, ha="center", fontsize=11, weight="bold", color=DARK)
        ax.text(x + 1.8, y + 0.40, detail, ha="center", fontsize=9.2, color=DARK)

    # Connections to center
    centers_top = [(2.2, 5.4), (6.5, 5.4), (10.8, 5.4)]
    centers_bot = [(2.2, 1.8), (6.5, 1.8), (10.8, 1.8)]
    for x, y in centers_top:
        _arrow(ax, x, y, 6.5, 4.4, color=GRAY, lw=1.1, style="-")
    for x, y in centers_bot:
        _arrow(ax, x, y, 6.5, 3.0, color=GRAY, lw=1.1, style="-")

    ax.text(6.5, 2.55,
            "Most enterprise blockers are not model quality — they are these six boxes.",
            ha="center", fontsize=9.2, style="italic", color=GRAY)

    save(fig, "fig7_enterprise_patterns.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_application_stack()
    fig2_workflow_patterns()
    fig3_rag_vs_finetuning()
    fig4_production_deploy()
    fig5_cost_optimization()
    fig6_observability()
    fig7_enterprise_patterns()
    print("Wrote 7 figures to:")
    print(f"  EN: {EN_DIR}")
    print(f"  ZH: {ZH_DIR}")


if __name__ == "__main__":
    main()
