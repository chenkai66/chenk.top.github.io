"""Figures for NLP Part 12 — Frontiers and Practical Applications (Series Finale).

Generates 7 publication-quality figures for the final installment of the
12-part NLP series. Each figure is saved into BOTH the English and
Chinese asset folders.

Figures:
    fig1_agent_architecture     — ReAct agent loop: LLM <-> tools <-> memory
    fig2_tool_use_pipeline      — Function-calling pipeline (JSON Schema -> exec)
    fig3_code_generation        — Code generation pipeline & HumanEval pass@k
    fig4_long_context           — Sliding window, dilated, infinite attention
    fig5_reasoning_models       — O1/R1 chain-of-thought scaling (test-time compute)
    fig6_production_deploy      — FastAPI + Docker + GPU serving stack
    fig7_series_journey         — 12-chapter NLP series journey map

Style: seaborn-v0_8-whitegrid, dpi=150, palette
    #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (orange).

References (verified):
    - Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models
      (ICLR 2023, arXiv:2210.03629).
    - Schick et al., Toolformer: Language Models Can Teach Themselves to Use
      Tools (NeurIPS 2023).
    - Roziere et al., Code Llama: Open Foundation Models for Code
      (Meta AI, arXiv:2308.12950, 2023).
    - Chen et al., Evaluating Large Language Models Trained on Code
      (HumanEval / Codex, arXiv:2107.03374, 2021).
    - Beltagy et al., Longformer: The Long-Document Transformer
      (arXiv:2004.05150, 2020).
    - Munkhdalai et al., Leave No Context Behind: Efficient Infinite Context
      Transformers with Infini-attention (arXiv:2404.07143, 2024).
    - OpenAI, Learning to Reason with LLMs (o1 system card, 2024).
    - DeepSeek-AI, DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
      Reinforcement Learning (arXiv:2501.12948, 2025).
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
DARK = "#1f2937"
RED = "#ef4444"
TEAL = "#0d9488"

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/nlp/frontiers-applications"
ZH_DIR = REPO / "source/_posts/zh/nlp/12-前沿技术与实战应用"


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure into both EN and ZH asset folders, then close it."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def _rounded(ax, x, y, w, h, text, edge, fc, *, fs=10, weight="normal", tc=None):
    """Draw a rounded box with centered text."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=edge, facecolor=fc, alpha=0.95,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, weight=weight,
        color=tc if tc else DARK,
    )


def _arrow(ax, x1, y1, x2, y2, color=DARK, lw=1.6, style="->", mut=14, ls="-"):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style,
        mutation_scale=mut, linewidth=lw, color=color, linestyle=ls,
    )
    ax.add_patch(arr)


# ---------------------------------------------------------------------------
# Figure 1 — Agent architecture: ReAct loop (LLM + tools + memory)
# ---------------------------------------------------------------------------
def fig1_agent_architecture() -> None:
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")

    ax.text(6, 6.05, "ReAct Agent Architecture — Reasoning + Acting Loop",
            ha="center", fontsize=14, weight="bold")
    ax.text(6, 5.7, "Yao et al., ICLR 2023 (arXiv:2210.03629)",
            ha="center", fontsize=9, color=GRAY, style="italic")

    # Central LLM
    _rounded(ax, 4.6, 2.5, 2.8, 1.2, "LLM Policy\n(GPT-4 / Claude / Llama-3)",
             BLUE, "#dbeafe", fs=11, weight="bold", tc=BLUE)

    # User input (left)
    _rounded(ax, 0.2, 2.7, 1.7, 0.9, "User Goal", DARK, "#fef3c7",
             fs=10, weight="bold")
    _arrow(ax, 1.95, 3.15, 4.55, 3.1, color=DARK, lw=1.8)

    # Final answer (right)
    _rounded(ax, 10.1, 2.7, 1.7, 0.9, "Final Answer", GREEN, "#d1fae5",
             fs=10, weight="bold", tc=GREEN)
    _arrow(ax, 7.45, 3.1, 10.05, 3.15, color=GREEN, lw=1.8)

    # ReAct loop — Thought -> Action -> Observation cycling back
    # Thought (top)
    _rounded(ax, 5.1, 4.4, 1.8, 0.7, "Thought", PURPLE, "#ede9fe",
             fs=10, weight="bold", tc=PURPLE)
    # Action (right)
    _rounded(ax, 7.9, 1.4, 1.6, 0.7, "Action", ORANGE, "#fef3c7",
             fs=10, weight="bold", tc=ORANGE)
    # Observation (left)
    _rounded(ax, 2.5, 1.4, 1.8, 0.7, "Observation", TEAL, "#ccfbf1",
             fs=10, weight="bold", tc=TEAL)

    # Loop arrows (counter-clockwise: Thought -> Action -> Obs -> Thought)
    _arrow(ax, 6.0, 4.4, 8.3, 2.1, color=PURPLE, lw=1.8)
    _arrow(ax, 7.9, 1.5, 4.3, 1.5, color=ORANGE, lw=1.8)
    _arrow(ax, 3.4, 2.1, 5.7, 4.4, color=TEAL, lw=1.8)

    # Tools panel (bottom right)
    _rounded(ax, 7.6, 0.1, 4.2, 1.0,
             "Tools  |  Search  •  Calculator  •  Code-Interpreter  •  SQL  •  REST APIs",
             ORANGE, "#fffbeb", fs=9.5)
    _arrow(ax, 8.7, 1.4, 8.7, 1.1, color=ORANGE, lw=1.6, style="<->")

    # Memory panel (bottom left)
    _rounded(ax, 0.2, 0.1, 4.2, 1.0,
             "Memory  |  Scratchpad (short-term)  •  Vector store (long-term)",
             GRAY, "#f3f4f6", fs=9.5)
    _arrow(ax, 3.3, 1.4, 3.3, 1.1, color=GRAY, lw=1.6, style="<->")

    # Iteration counter annotation
    ax.text(6, 3.95, "loop  k = 1 .. K",
            ha="center", fontsize=9, color=GRAY, style="italic")

    save(fig, "fig1_agent_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2 — Function calling / tool use pipeline
# ---------------------------------------------------------------------------
def fig2_tool_use_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 5.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    ax.text(6.5, 5.1, "Function Calling Pipeline — from JSON Schema to Tool Result",
            ha="center", fontsize=14, weight="bold")

    # Stage boxes (5 stages)
    stages = [
        (0.3, "1. Schema\nDeclaration", "JSON Schema for\nname/args/return", BLUE, "#dbeafe"),
        (2.8, "2. Routing", "Model decides:\ndirect reply or call?", PURPLE, "#ede9fe"),
        (5.3, "3. Argument\nGeneration", "Constrained decoding\n-> valid JSON", ORANGE, "#fef3c7"),
        (7.8, "4. Execution", "Sandboxed runtime\nruns the function", GREEN, "#d1fae5"),
        (10.3, "5. Integration", "Result fed back as\ntool message", TEAL, "#ccfbf1"),
    ]
    for x, title, body, edge, fc in stages:
        _rounded(ax, x, 3.1, 2.3, 1.5, title, edge, fc,
                 fs=10.5, weight="bold", tc=edge)
        ax.text(x + 1.15, 2.55, body, ha="center", va="top",
                fontsize=8.8, color=DARK)

    # Connecting arrows
    for i in range(4):
        x = 0.3 + 2.5 * i + 2.3
        _arrow(ax, x, 3.85, x + 0.2, 3.85, color=DARK, lw=1.8)

    # Bottom: example JSON snippet
    code = (
        '{ "name": "get_weather",\n'
        '  "arguments": { "city": "Hangzhou", "unit": "celsius" } }'
    )
    _rounded(ax, 1.0, 0.2, 5.5, 1.6,
             "Example tool call (model output)\n\n" + code,
             BLUE, "#f8fafc", fs=9.2, tc=DARK)

    obs = (
        '{ "tool": "get_weather",\n'
        '  "result": "Sunny, 25 degC, humidity 60%" }'
    )
    _rounded(ax, 6.7, 0.2, 5.5, 1.6,
             "Tool message returned to model\n\n" + obs,
             GREEN, "#f0fdf4", fs=9.2, tc=DARK)

    _arrow(ax, 6.5, 1.0, 6.7, 1.0, color=DARK, lw=1.6)

    save(fig, "fig2_tool_use_pipeline.png")


# ---------------------------------------------------------------------------
# Figure 3 — Code generation: pipeline + HumanEval pass@k curves
# ---------------------------------------------------------------------------
def fig3_code_generation() -> None:
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ---- Left: code generation pipeline ----
    ax1.set_xlim(0, 11)
    ax1.set_ylim(0, 5.4)
    ax1.axis("off")
    ax1.set_title("Code Generation Pipeline (Codex / Code Llama / StarCoder)",
                  loc="left")

    pipeline = [
        (0.2, "Natural-Language\nIntent", BLUE, "#dbeafe"),
        (2.4, "Context\n(repo + docs)", PURPLE, "#ede9fe"),
        (4.6, "Code-LLM\nDecoder", ORANGE, "#fef3c7"),
        (6.8, "Execution\n+ Unit Tests", TEAL, "#ccfbf1"),
        (9.0, "Self-Repair\n+ Final Code", GREEN, "#d1fae5"),
    ]
    for x, lab, edge, fc in pipeline:
        _rounded(ax1, x, 3.0, 1.9, 1.4, lab, edge, fc,
                 fs=10, weight="bold", tc=edge)

    for i in range(4):
        x = 0.2 + 2.2 * i + 1.9
        _arrow(ax1, x, 3.7, x + 0.3, 3.7, color=DARK, lw=1.8)

    # Feedback loop arrow (test failure -> Code-LLM)
    _arrow(ax1, 7.75, 3.0, 5.55, 3.0, color=RED, lw=1.6, style="->", ls="--")
    ax1.text(6.6, 2.6, "test failure -> regenerate",
             ha="center", fontsize=9, color=RED, style="italic")

    # Training data callouts
    ax1.text(0.3, 1.8, "Training data",
             fontsize=9.5, weight="bold", color=DARK)
    ax1.text(0.3, 1.3, "• GitHub permissive code (~1T tokens)",
             fontsize=9, color=DARK)
    ax1.text(0.3, 0.95, "• Stack Overflow Q&A, docs",
             fontsize=9, color=DARK)
    ax1.text(0.3, 0.6, "• Synthetic instructions + unit tests",
             fontsize=9, color=DARK)
    ax1.text(0.3, 0.25, "• RLHF / RLAIF on human preferences",
             fontsize=9, color=DARK)

    # ---- Right: HumanEval pass@k for representative models ----
    models = [
        ("Codex-12B (2021)", 28.8, BLUE),
        ("Code Llama 34B-I", 53.7, PURPLE),
        ("Code Llama 70B-I", 67.8, ORANGE),
        ("DeepSeek-Coder 33B", 79.3, TEAL),
        ("GPT-4 (2023)", 67.0, GREEN),
        ("GPT-4o (2024)", 90.2, RED),
    ]
    names = [m[0] for m in models]
    scores = [m[1] for m in models]
    colors = [m[2] for m in models]
    y = np.arange(len(models))
    ax2.barh(y, scores, color=colors, alpha=0.88,
             edgecolor="white", linewidth=1.4)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=9.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 100)
    ax2.set_xlabel("HumanEval pass@1 (%)")
    ax2.set_title("Code generation benchmarks (HumanEval pass@1)", loc="left")
    for i, s in enumerate(scores):
        ax2.text(s + 1.5, i, f"{s:.1f}", va="center", fontsize=9.5, color=DARK)
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.grid(False, axis="y")

    save(fig, "fig3_code_generation.png")


# ---------------------------------------------------------------------------
# Figure 4 — Long context: full vs sliding window vs dilated vs infini-attn
# ---------------------------------------------------------------------------
def fig4_long_context() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.0))
    n = 14

    def draw_mask(ax, mask, title, edge):
        ax.imshow(mask, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_title(title, loc="left", color=edge, fontsize=11.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(edge)
            s.set_linewidth(1.4)
        ax.set_xlabel("key position")
        ax.set_ylabel("query position")

    # Full causal attention
    full = np.tril(np.ones((n, n)))
    draw_mask(axes[0], full, f"Full Causal\nO(n^2) — n={n}", BLUE)

    # Sliding window (Longformer local)
    w = 4
    sw = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - w + 1), i + 1):
            sw[i, j] = 1.0
    draw_mask(axes[1], sw, f"Sliding Window\nLongformer (w={w})", PURPLE)

    # Dilated / strided + global tokens
    dl = np.zeros((n, n))
    for i in range(n):
        for j in range(0, i + 1, 2):
            dl[i, j] = 0.85
    # add 2 global tokens (every query attends to first two)
    dl[:, 0] = 1.0
    dl[:, 1] = 1.0
    dl[0, :] = 0.0
    dl[1, :] = 0.0
    dl = np.tril(dl) + np.triu(np.zeros_like(dl))
    dl[:, 0] = np.tril(np.ones((n, n)))[:, 0]
    dl[:, 1] = np.tril(np.ones((n, n)))[:, 1]
    draw_mask(axes[2], dl, "Dilated + Global\nBigBird-style", ORANGE)

    # Infini-attention: local sliding + compressed memory
    ia = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - w + 1), i + 1):
            ia[i, j] = 1.0
    # compressed memory = soft attention to all earlier (low intensity)
    for i in range(n):
        for j in range(0, max(0, i - w + 1)):
            ia[i, j] = 0.35
    draw_mask(axes[3], ia, "Infini-Attention\nlocal + compressed mem", GREEN)

    fig.suptitle(
        "Attention masks for long-context modeling (white = masked, blue = attended)",
        fontsize=13, weight="bold", y=1.02,
    )
    fig.tight_layout()
    save(fig, "fig4_long_context.png")


# ---------------------------------------------------------------------------
# Figure 5 — Reasoning models (O1, R1): test-time compute scaling
# ---------------------------------------------------------------------------
def fig5_reasoning_models() -> None:
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # ---- Left: chain-of-thought structure (visual) ----
    ax1.set_xlim(0, 12)
    ax1.set_ylim(0, 5.4)
    ax1.axis("off")
    ax1.set_title("Reasoning models — internal chain-of-thought (CoT)",
                  loc="left")

    _rounded(ax1, 0.3, 4.0, 2.4, 1.0, "Question", DARK, "#fef3c7",
             fs=10.5, weight="bold")

    # CoT chain
    cot_steps = [
        ("3.2", 4.0, "Step 1\nidentify\nsub-goals", BLUE),
        ("5.5", 4.0, "Step 2\nmath /\nlookup", PURPLE),
        ("7.8", 4.0, "Step 3\nverify\n+ branch", ORANGE),
        ("10.1", 4.0, "Step k\nself-\nreflect", TEAL),
    ]
    for xs, y, lab, c in cot_steps:
        _rounded(ax1, float(xs), y, 1.7, 1.0, lab, c, "#ffffff",
                 fs=9.2, tc=c)

    # arrows along the chain
    xs = [2.7, 4.9, 7.2, 9.5]
    for x in xs:
        _arrow(ax1, x, 4.5, x + 0.5, 4.5, color=DARK, lw=1.6)

    _rounded(ax1, 4.6, 1.5, 2.6, 1.0, "Final Answer", GREEN, "#d1fae5",
             fs=11, weight="bold", tc=GREEN)
    _arrow(ax1, 5.9, 4.0, 5.9, 2.55, color=DARK, lw=1.6)

    # Self-correction loop
    _arrow(ax1, 9.7, 4.0, 7.5, 2.0, color=RED, lw=1.4, ls="--")
    ax1.text(8.3, 2.7, "self-correction\n(rollback / retry)",
             ha="center", fontsize=9, color=RED, style="italic")

    ax1.text(0.3, 0.6,
             "OpenAI o1 (2024) and DeepSeek-R1 (2025) trade test-time compute\n"
             "for accuracy: more reasoning tokens -> higher score on math/code.",
             fontsize=9, color=DARK)

    # ---- Right: test-time scaling curve ----
    tokens = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    # Stylised: log-linear gains for math reasoning (AIME-like benchmark)
    base = 13 + 8.5 * np.log2(tokens) + np.random.RandomState(0).normal(0, 0.4, len(tokens))
    base = np.clip(base, 0, 95)
    classic = 13 + 0.6 * np.log2(tokens)  # standard model — flat
    classic = np.clip(classic, 0, 95)

    ax2.plot(tokens, base, "-o", color=PURPLE, lw=2.4, ms=6,
             label="Reasoning model (o1 / R1-style)")
    ax2.plot(tokens, classic, "-s", color=GRAY, lw=2.0, ms=5,
             label="Standard LLM (GPT-4o-style)")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Reasoning tokens at test time (log scale)")
    ax2.set_ylabel("Accuracy on hard math benchmark (%)")
    ax2.set_title("Test-time compute scaling (stylised, AIME-like)",
                  loc="left")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower right", fontsize=9.5)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(50, color=LIGHT, lw=1, ls=":")
    ax2.text(1.1, 51, "human expert tier", fontsize=8, color=GRAY)

    save(fig, "fig5_reasoning_models.png")


# ---------------------------------------------------------------------------
# Figure 6 — Production deployment: FastAPI + Docker + GPU stack
# ---------------------------------------------------------------------------
def fig6_production_deploy() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    ax.text(6.75, 6.3, "Production Deployment Stack — FastAPI + Docker + GPU Inference",
            ha="center", fontsize=14, weight="bold")

    # Layered stack — bottom to top
    layers = [
        (0.3, "Hardware",      "NVIDIA A100 / H100  •  CUDA 12  •  NVMe SSD",                     GRAY,   "#f3f4f6"),
        (1.3, "Inference Engine", "vLLM  •  TensorRT-LLM  •  Triton  •  llama.cpp  (PagedAttention, continuous batching)", ORANGE, "#fef3c7"),
        (2.3, "Model Layer",   "Quantised weights (AWQ / GPTQ 4-bit)  •  LoRA adapters  •  KV-cache",                       PURPLE, "#ede9fe"),
        (3.3, "API Layer",     "FastAPI  •  pydantic validation  •  /chat  /embeddings  /health",                          BLUE,   "#dbeafe"),
        (4.3, "Container",     "Docker + nvidia-container-runtime  •  K8s Deployment + HPA",                                TEAL,   "#ccfbf1"),
        (5.3, "Edge",          "Nginx / Envoy gateway  •  rate-limit  •  auth (JWT)  •  TLS",                               GREEN,  "#d1fae5"),
    ]
    for y, name, body, edge, fc in layers:
        _rounded(ax, 0.4, y, 9.5, 0.85, "", edge, fc)
        ax.text(0.7, y + 0.42, name, fontsize=11, weight="bold",
                color=edge, va="center")
        ax.text(2.6, y + 0.42, body, fontsize=9.5, color=DARK, va="center")

    # Right column — observability & ops
    _rounded(ax, 10.2, 1.3, 3.0, 4.85, "", DARK, "#ffffff")
    ax.text(11.7, 5.85, "Observability & Ops",
            ha="center", fontsize=11.5, weight="bold", color=DARK)

    obs_items = [
        ("Prometheus", "QPS, latency p50/p95/p99,"),
        ("",            "GPU util, VRAM, queue depth"),
        ("Grafana",    "dashboards + SLO alerts"),
        ("OpenTelemetry", "request traces"),
        ("Loki / ELK", "structured JSON logs"),
        ("Sentry",     "error capture"),
        ("Canary +",   "rollback on regression"),
        ("blue-green", "deploys"),
    ]
    for i, (k, v) in enumerate(obs_items):
        ax.text(10.4, 5.45 - i * 0.42, k, fontsize=9.5,
                weight="bold", color=BLUE)
        ax.text(11.5, 5.45 - i * 0.42, v, fontsize=9, color=DARK)

    # Request flow callout
    ax.text(0.4, 0.55,
            "Request flow:  client  ->  Nginx  ->  FastAPI  ->  vLLM  ->  GPU  "
            "->  streamed tokens back",
            fontsize=10, color=DARK, weight="bold")
    ax.text(0.4, 0.2,
            "Targets (7B model, A100 80GB):  ~1500 tok/s/GPU  •  p95 first-token < 300 ms  •  ~30 concurrent streams",
            fontsize=9, color=GRAY)

    save(fig, "fig6_production_deploy.png")


# ---------------------------------------------------------------------------
# Figure 7 — 12-chapter NLP series journey map
# ---------------------------------------------------------------------------
def fig7_series_journey() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 6.6))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 6.8)
    ax.axis("off")

    ax.text(7.25, 6.5, "The 12-Chapter NLP Journey — from Tokens to Production Agents",
            ha="center", fontsize=14.5, weight="bold")

    chapters = [
        (1,  "Intro &\nPreprocessing",     BLUE),
        (2,  "Word Embeddings\n& LMs",     BLUE),
        (3,  "RNN / LSTM\nSeq Modeling",   BLUE),
        (4,  "Attention &\nTransformer",   PURPLE),
        (5,  "BERT &\nPretraining",        PURPLE),
        (6,  "GPT &\nGenerative LMs",      PURPLE),
        (7,  "Prompting &\nIn-Context",    ORANGE),
        (8,  "Fine-Tuning\n& PEFT",        ORANGE),
        (9,  "LLM Architecture\nDeep Dive", ORANGE),
        (10, "RAG & Knowledge\nEnhancement", GREEN),
        (11, "Multimodal\nNLP",            GREEN),
        (12, "Frontiers\n& Production",    RED),
    ]

    # Layout: 6 columns x 2 rows zig-zag, with arrows linking sequentially
    cols = 6
    cell_w = 14.5 / cols
    positions = []
    for i, (num, label, color) in enumerate(chapters):
        row = i // cols  # 0 = top, 1 = bottom
        col = i % cols if row == 0 else (cols - 1 - (i % cols))
        x = col * cell_w + 0.25
        # Top row higher, bottom row lower
        y = 4.4 if row == 0 else 1.7
        positions.append((x, y, num, label, color))

    # Draw chapter cards
    box_w = cell_w - 0.5
    box_h = 1.5
    for x, y, num, label, color in positions:
        face = "#ffffff" if num != 12 else "#fee2e2"
        _rounded(ax, x, y, box_w, box_h, "", color, face)
        ax.text(x + 0.25, y + box_h - 0.25, f"Ch {num}",
                fontsize=10.5, weight="bold", color=color)
        ax.text(x + box_w / 2, y + 0.55, label,
                ha="center", va="center", fontsize=10, color=DARK,
                weight="bold" if num == 12 else "normal")

    # Connect with arrows in series order
    for i in range(len(positions) - 1):
        x1, y1, *_ = positions[i]
        x2, y2, *_ = positions[i + 1]
        cx1 = x1 + box_w / 2
        cx2 = x2 + box_w / 2
        if y1 == y2:
            # horizontal neighbours
            if cx2 > cx1:
                _arrow(ax, x1 + box_w, y1 + box_h / 2,
                       x2, y2 + box_h / 2, color=GRAY, lw=1.5)
            else:
                _arrow(ax, x1, y1 + box_h / 2,
                       x2 + box_w, y2 + box_h / 2, color=GRAY, lw=1.5)
        else:
            # vertical drop between rows (same column)
            _arrow(ax, cx1, y1, cx2, y2 + box_h, color=GRAY, lw=1.5)

    # Theme bands
    bands = [
        ("Foundations (1-3)",   BLUE,   0.3),
        ("Transformers (4-6)",  PURPLE, 4.0),
        ("Adaptation (7-9)",    ORANGE, 7.6),
        ("Applications (10-12)", GREEN, 11.2),
    ]
    for name, color, x in bands:
        ax.text(x, 0.4, name, fontsize=10, weight="bold", color=color)

    ax.text(7.25, 0.05, "End of series — thank you for reading.",
            ha="center", fontsize=10.5, color=DARK, style="italic")

    save(fig, "fig7_series_journey.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    fig1_agent_architecture()
    fig2_tool_use_pipeline()
    fig3_code_generation()
    fig4_long_context()
    fig5_reasoning_models()
    fig6_production_deploy()
    fig7_series_journey()
    print("Saved 7 figures to:")
    print(f"  EN  {EN_DIR}")
    print(f"  ZH  {ZH_DIR}")


if __name__ == "__main__":
    main()
