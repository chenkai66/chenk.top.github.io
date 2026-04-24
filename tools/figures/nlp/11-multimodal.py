"""Figures for NLP Part 11 — Multimodal Large Language Models.

Generates 7 publication-quality figures for the article on multimodal
NLP. Each figure is saved into BOTH the English and Chinese asset
folders.

Figures:
    fig1_clip_architecture       — CLIP dual encoder + contrastive matrix
    fig2_blip2_qformer           — BLIP-2 with Q-Former bridging frozen ViT/LLM
    fig3_llava_architecture      — LLaVA: vision encoder → projector → LLM
    fig4_vl_tasks                — Vision-language tasks: VQA, captioning, retrieval
    fig5_whisper_audio           — Whisper encoder-decoder over mel-spectrogram
    fig6_gpt4v_examples          — GPT-4V capability matrix and example outputs
    fig7_mllm_benchmarks         — Multimodal benchmarks: MMBench, MME, MMMU

Style: seaborn-v0_8-whitegrid, dpi=150, palette
    #2563eb (blue) #7c3aed (purple) #10b981 (green) #f59e0b (orange).

References (verified):
    - Radford et al., Learning Transferable Visual Models from Natural
      Language Supervision (CLIP, ICML 2021)
    - Li et al., BLIP-2: Bootstrapping Language-Image Pre-training with
      Frozen Image Encoders and Large Language Models (ICML 2023)
    - Liu et al., Visual Instruction Tuning (LLaVA, NeurIPS 2023)
    - Radford et al., Robust Speech Recognition via Large-Scale Weak
      Supervision (Whisper, ICML 2023)
    - OpenAI, GPT-4V(ision) System Card (2023)
    - Liu et al., MMBench: Is Your Multi-modal Model an All-around
      Player? (ECCV 2024)
    - Fu et al., MME: A Comprehensive Evaluation Benchmark for
      Multimodal Large Language Models (2023)
    - Yue et al., MMMU: A Massive Multi-discipline Multimodal
      Understanding and Reasoning Benchmark (CVPR 2024)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
DARK = "#1f2937"
RED = "#ef4444"

REPO = Path("/Users/kchen/Desktop/Project/chenk-site")
EN_DIR = REPO / "source/_posts/en/nlp/multimodal-nlp"
ZH_DIR = REPO / "source/_posts/zh/nlp/11-多模态大模型"


def save(fig: plt.Figure, name: str) -> None:
    """Save the figure to both EN and ZH asset folders, then close it."""
    for d in (EN_DIR, ZH_DIR):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, facecolor="white")
    plt.close(fig)


def _rounded(ax, x, y, w, h, text, edge, fc, *, fs=10, weight="normal", tc=None):
    """Draw a rounded box with centered text."""
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
# Figure 1 — CLIP architecture: dual encoder + contrastive matrix
# ---------------------------------------------------------------------------
def fig1_clip_architecture() -> None:
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.18)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # ---- Left: dual encoder diagram ----
    axL.set_xlim(0, 12)
    axL.set_ylim(0, 7)
    axL.axis("off")
    axL.set_title("CLIP — dual encoder, contrastive alignment",
                  loc="left", fontsize=12.5, weight="bold")

    # Image branch (top)
    _rounded(axL, 0.3, 5.2, 1.7, 1.0, "image\n(224×224)",
             BLUE, "white", fs=10, weight="bold", tc=BLUE)
    _arrow(axL, 2.05, 5.7, 2.55, 5.7, color=BLUE)
    _rounded(axL, 2.6, 5.0, 2.6, 1.4,
             "Image encoder\nViT-L/14 or RN50",
             BLUE, BLUE, fs=10, weight="bold", tc="white")
    _arrow(axL, 5.25, 5.7, 5.75, 5.7, color=BLUE)
    _rounded(axL, 5.8, 5.2, 1.9, 1.0,
             "I·  ∈ ℝ⁵¹²", BLUE, "white", fs=10.5,
             weight="bold", tc=BLUE)

    # Text branch (bottom)
    _rounded(axL, 0.3, 0.8, 1.7, 1.0, "text:\n\"a photo of\na cat\"",
             PURPLE, "white", fs=9.5, weight="bold", tc=PURPLE)
    _arrow(axL, 2.05, 1.3, 2.55, 1.3, color=PURPLE)
    _rounded(axL, 2.6, 0.6, 2.6, 1.4,
             "Text encoder\nTransformer (12L)",
             PURPLE, PURPLE, fs=10, weight="bold", tc="white")
    _arrow(axL, 5.25, 1.3, 5.75, 1.3, color=PURPLE)
    _rounded(axL, 5.8, 0.8, 1.9, 1.0,
             "T·  ∈ ℝ⁵¹²", PURPLE, "white", fs=10.5,
             weight="bold", tc=PURPLE)

    # Joint embedding & similarity
    _arrow(axL, 7.7, 5.7, 9.5, 4.0, color=BLUE)
    _arrow(axL, 7.7, 1.3, 9.5, 3.4, color=PURPLE)
    _rounded(axL, 9.55, 3.1, 2.2, 1.2,
             "cosine sim\nIᵀT / τ",
             DARK, LIGHT, fs=10.5, weight="bold")

    # Equation strip
    axL.text(6.0, 4.05,
             r"$\mathcal{L} = -\frac{1}{2N}\sum_i \log\frac{e^{s_{ii}/\tau}}"
             r"{\sum_j e^{s_{ij}/\tau}} + \log\frac{e^{s_{ii}/\tau}}"
             r"{\sum_j e^{s_{ji}/\tau}}$",
             ha="center", fontsize=10.5,
             bbox=dict(facecolor=LIGHT, edgecolor="none", pad=6))

    axL.text(0.3, 6.5, "■ image stream", color=BLUE,
             fontsize=10, weight="bold")
    axL.text(3.0, 6.5, "■ text stream", color=PURPLE,
             fontsize=10, weight="bold")
    axL.text(0.3, 0.15,
             "400 M image–text pairs (WIT) · contrastive InfoNCE · τ = 0.07 (learned)",
             fontsize=9.5, color=GRAY, style="italic")

    # ---- Right: 5×5 contrastive similarity matrix ----
    axR.set_title("In-batch similarity matrix  (N = 5)",
                  loc="left", fontsize=12.5, weight="bold")
    rng = np.random.default_rng(7)
    sim = rng.uniform(0.05, 0.30, size=(5, 5))
    np.fill_diagonal(sim, rng.uniform(0.78, 0.95, size=5))

    im = axR.imshow(sim, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    axR.set_xticks(range(5))
    axR.set_yticks(range(5))
    captions = ["a cat", "a dog", "a car", "a bird", "a tree"]
    axR.set_xticklabels([f"T{i+1}\n\"{c}\"" for i, c in enumerate(captions)],
                        fontsize=9)
    axR.set_yticklabels([f"I{i+1}" for i in range(5)], fontsize=9.5,
                        weight="bold")
    axR.set_xlabel("Text embeddings  T_j", fontsize=10)
    axR.set_ylabel("Image embeddings  I_i", fontsize=10)

    # Highlight diagonal
    for i in range(5):
        axR.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor=GREEN, lw=2.2))
        axR.text(i, i, f"{sim[i, i]:.2f}", ha="center", va="center",
                 fontsize=9, color="white", weight="bold")
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            axR.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                     fontsize=8, color=DARK)

    cbar = fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04)
    cbar.set_label("cosine sim", fontsize=9)

    fig.suptitle("CLIP: Contrastive Language–Image Pre-training",
                 fontsize=14, weight="bold", y=1.02)
    save(fig, "fig1_clip_architecture.png")


# ---------------------------------------------------------------------------
# Figure 2 — BLIP-2 with Q-Former
# ---------------------------------------------------------------------------
def fig2_blip2_qformer() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 5.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    ax.set_title("BLIP-2 — Q-Former bridges a frozen ViT and a frozen LLM",
                 loc="left", fontsize=13, weight="bold")

    # Frozen ViT (left)
    _rounded(ax, 0.3, 2.4, 2.2, 1.6,
             "Frozen\nImage Encoder\n(ViT-g/14)",
             GRAY, GRAY, fs=10, weight="bold", tc="white")
    ax.text(1.4, 4.2, "image", ha="center", fontsize=9.5, color=BLUE,
            weight="bold")
    _arrow(ax, 1.4, 4.15, 1.4, 4.0, color=BLUE)
    _arrow(ax, 2.55, 3.2, 3.95, 3.2, color=DARK)
    ax.text(3.25, 3.5, "patch\nfeatures", ha="center", fontsize=8.5,
            color=GRAY, style="italic")

    # Q-Former (center) — trainable
    qx, qy, qw, qh = 4.0, 1.6, 4.2, 3.2
    _rounded(ax, qx, qy, qw, qh, "", GREEN, "white")
    ax.text(qx + qw / 2, qy + qh - 0.3,
            "Q-Former  (trainable, ~188 M)",
            ha="center", fontsize=11, weight="bold", color=GREEN)

    # 32 learnable queries (left side of Q-Former)
    for i in range(8):
        _rounded(ax, qx + 0.2, qy + 0.3 + i * 0.30, 0.55, 0.25,
                 "", GREEN, GREEN, fs=8)
    ax.text(qx + 0.5, qy + 0.05, "32 queries",
            ha="center", fontsize=8.5, color=GREEN, weight="bold")

    # Self-attn + Cross-attn stack inside Q-Former
    _rounded(ax, qx + 1.1, qy + 2.05, 2.9, 0.55,
             "Self-Attention", PURPLE, "white", fs=10, tc=PURPLE,
             weight="bold")
    _rounded(ax, qx + 1.1, qy + 1.30, 2.9, 0.55,
             "Cross-Attention  (queries ↔ image features)",
             ORANGE, "white", fs=10, tc=ORANGE, weight="bold")
    _rounded(ax, qx + 1.1, qy + 0.55, 2.9, 0.55,
             "Feed-Forward", BLUE, "white", fs=10, tc=BLUE,
             weight="bold")

    ax.text(qx + qw / 2, qy + 2.95, "× 12 blocks",
            ha="center", fontsize=8.5, color=GRAY, style="italic")

    # Output: 32 visual tokens
    _arrow(ax, qx + qw, qy + qh / 2, qx + qw + 0.9, qy + qh / 2,
           color=GREEN)
    ax.text(qx + qw + 0.45, qy + qh / 2 + 0.25,
            "32 soft\nvisual tokens", ha="center", fontsize=8.5,
            color=GREEN, weight="bold")

    # Linear projection
    _rounded(ax, qx + qw + 0.95, qy + 1.0, 1.1, 1.6,
             "Linear\nproj", DARK, LIGHT, fs=10, weight="bold")
    _arrow(ax, qx + qw + 2.1, qy + 1.8, qx + qw + 2.7, qy + 1.8)

    # Frozen LLM
    _rounded(ax, qx + qw + 2.75, qy + 0.4, 2.4, 2.8,
             "Frozen LLM\n(OPT / FlanT5)",
             GRAY, GRAY, fs=10, weight="bold", tc="white")
    _arrow(ax, qx + qw + 5.2, qy + 1.8, qx + qw + 5.85, qy + 1.8)
    ax.text(qx + qw + 6.3, qy + 1.8, "text\noutput",
            ha="center", va="center", fontsize=10, color=BLUE,
            weight="bold")

    # Bottom note: two-stage pretraining
    ax.text(0.3, 0.85,
            "Two-stage pre-training:\n"
            "  ① Vision–Language representation (ITC + ITM + ITG, "
            "frozen ViT only)\n"
            "  ② Vision-to-Language generation (output to frozen "
            "LLM, language modelling loss)",
            fontsize=9.5, color=DARK,
            bbox=dict(facecolor=LIGHT, edgecolor="none", pad=8))

    ax.text(13.7, 5.9, "■ frozen", color=GRAY, fontsize=9.5,
            ha="right", weight="bold")
    ax.text(12.2, 5.9, "■ trainable", color=GREEN, fontsize=9.5,
            ha="right", weight="bold")

    save(fig, "fig2_blip2_qformer.png")


# ---------------------------------------------------------------------------
# Figure 3 — LLaVA architecture
# ---------------------------------------------------------------------------
def fig3_llava_architecture() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    # ---- Left: full LLaVA pipeline ----
    axL.set_xlim(0, 14)
    axL.set_ylim(0, 6)
    axL.axis("off")
    axL.set_title("LLaVA — vision encoder → projector → LLM",
                  loc="left", fontsize=12.5, weight="bold")

    # Image input
    _rounded(axL, 0.3, 3.3, 1.6, 1.2, "image\nXᵥ",
             BLUE, "white", fs=10.5, weight="bold", tc=BLUE)
    _arrow(axL, 1.95, 3.9, 2.55, 3.9, color=BLUE)

    # CLIP ViT-L/14 frozen
    _rounded(axL, 2.6, 3.0, 2.4, 1.8,
             "CLIP ViT-L/14\n(frozen)",
             GRAY, GRAY, fs=10, weight="bold", tc="white")
    axL.text(3.8, 2.7, "Zᵥ ∈ ℝ²⁵⁶ × ¹⁰²⁴",
             ha="center", fontsize=8.5, color=GRAY, style="italic")
    _arrow(axL, 5.05, 3.9, 5.65, 3.9, color=DARK)

    # Projector W (single linear / 2-layer MLP)
    _rounded(axL, 5.7, 3.3, 1.7, 1.2,
             "Projector\nW  (linear /\n2-layer MLP)",
             GREEN, GREEN, fs=9.5, weight="bold", tc="white")
    _arrow(axL, 7.45, 3.9, 8.05, 3.9, color=GREEN)
    axL.text(6.55, 2.7, "Hᵥ = W · Zᵥ  ∈ ℝ²⁵⁶ × ⁴⁰⁹⁶",
             ha="center", fontsize=8.5, color=GREEN, style="italic")

    # Concatenate with text tokens
    _rounded(axL, 8.1, 3.3, 1.7, 1.2,
             "concat\n[Hᵥ ; H_text]",
             ORANGE, "white", fs=9.5, weight="bold", tc=ORANGE)
    _arrow(axL, 9.85, 3.9, 10.45, 3.9, color=ORANGE)

    # Vicuna LLM
    _rounded(axL, 10.5, 2.8, 3.0, 2.2,
             "Vicuna-7B / 13B\n(language model,\nfine-tuned)",
             PURPLE, PURPLE, fs=10, weight="bold", tc="white")

    # Text input
    _rounded(axL, 8.1, 1.5, 1.7, 0.8,
             "text  X_q\n\"What is in\nthe image?\"",
             BLUE, "white", fs=8.5, weight="bold", tc=BLUE)
    _arrow(axL, 8.95, 2.3, 8.95, 3.25, color=BLUE)

    # Output
    _arrow(axL, 13.55, 3.9, 13.95, 3.9, color=PURPLE)
    axL.text(13.95, 4.4, "Xₐ\nresponse", ha="left", fontsize=9,
             weight="bold", color=PURPLE)

    # Notes
    axL.text(0.3, 1.2,
             "Two-stage instruction tuning:\n"
             "  ① Pre-train projector W only on 558 K image–caption "
             "pairs (LAION-CC-SBU)\n"
             "  ② Fine-tune projector + LLM on 158 K GPT-4-generated "
             "visual instruction examples",
             fontsize=9.5, color=DARK,
             bbox=dict(facecolor=LIGHT, edgecolor="none", pad=8))

    # ---- Right: comparison of connector designs ----
    axR.set_title("Vision–language connectors",
                  loc="left", fontsize=12.5, weight="bold")
    designs = ["Linear\n(LLaVA-1)", "MLP-2\n(LLaVA-1.5)",
               "Q-Former\n(BLIP-2)", "Cross-attn\n(Flamingo)",
               "Patch-resampler\n(Idefics)"]
    params_m = [4.2, 17, 188, 320, 41]
    accs = [73.2, 80.0, 65.0, 71.5, 76.4]  # MMBench-style proxy

    x = np.arange(len(designs))
    bw = 0.42
    bars1 = axR.bar(x - bw / 2, params_m, bw, color=GREEN,
                    edgecolor="white", label="trainable params (M)")
    ax2 = axR.twinx()
    bars2 = ax2.bar(x + bw / 2, accs, bw, color=BLUE,
                    edgecolor="white", label="benchmark (proxy)")

    axR.set_xticks(x)
    axR.set_xticklabels(designs, fontsize=8.5)
    axR.set_ylabel("Trainable params (M)", color=GREEN)
    ax2.set_ylabel("Benchmark score", color=BLUE)
    axR.tick_params(axis="y", labelcolor=GREEN)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    axR.set_ylim(0, 360)
    ax2.set_ylim(50, 90)
    axR.grid(False)
    ax2.grid(False)

    for bar, v in zip(bars1, params_m):
        axR.text(bar.get_x() + bar.get_width() / 2, v + 8, f"{v:.0f}M",
                 ha="center", fontsize=8, color=GREEN, weight="bold")
    for bar, v in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.6, f"{v:.0f}",
                 ha="center", fontsize=8, color=BLUE, weight="bold")

    fig.suptitle("LLaVA-style architectures for visual instruction tuning",
                 fontsize=14, weight="bold", y=1.02)
    save(fig, "fig3_llava_architecture.png")


# ---------------------------------------------------------------------------
# Figure 4 — Vision-language tasks: VQA, captioning, retrieval
# ---------------------------------------------------------------------------
def fig4_vl_tasks() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0))
    fig.suptitle("Three canonical vision–language tasks",
                 fontsize=14, weight="bold", y=1.02)

    # ---------- (a) VQA ----------
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("(a) Visual Question Answering",
                 fontsize=12, weight="bold", color=BLUE)

    # Image placeholder — colored grid
    rng = np.random.default_rng(0)
    img = rng.uniform(0.3, 0.95, size=(8, 8, 3))
    ax.imshow(img, extent=(0.5, 4.5, 4.0, 7.5), aspect="auto")
    ax.add_patch(Rectangle((0.5, 4.0), 4.0, 3.5,
                           fill=False, edgecolor=DARK, lw=1.5))
    # cartoon: schedule of "two cars" hint
    ax.add_patch(Rectangle((1.2, 4.6), 0.9, 0.45,
                           facecolor=BLUE, edgecolor="white"))
    ax.add_patch(Rectangle((2.7, 4.6), 0.9, 0.45,
                           facecolor=ORANGE, edgecolor="white"))
    ax.text(2.5, 7.8, "image", ha="center", fontsize=9.5, color=DARK,
            style="italic")

    _rounded(ax, 5.2, 6.0, 4.5, 1.3,
             "Q: \"How many cars\nare in the image?\"",
             BLUE, "white", fs=10, weight="bold", tc=BLUE)
    _arrow(ax, 5.0, 5.5, 5.0, 4.3, color=DARK, lw=1.5)
    _rounded(ax, 2.5, 2.6, 5.0, 1.4,
             "Multimodal model\n(BLIP-VQA / LLaVA)",
             PURPLE, PURPLE, fs=10, weight="bold", tc="white")
    _arrow(ax, 5.0, 2.55, 5.0, 1.6, color=PURPLE, lw=1.5)
    _rounded(ax, 3.0, 0.4, 4.0, 1.1,
             "A: \"Two\"", GREEN, GREEN, fs=11, weight="bold", tc="white")

    # ---------- (b) Captioning ----------
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("(b) Image Captioning",
                 fontsize=12, weight="bold", color=PURPLE)

    img2 = rng.uniform(0.3, 0.95, size=(8, 8, 3))
    ax.imshow(img2, extent=(2.5, 7.5, 5.4, 7.6), aspect="auto")
    ax.add_patch(Rectangle((2.5, 5.4), 5.0, 2.2,
                           fill=False, edgecolor=DARK, lw=1.5))
    ax.text(5.0, 7.85, "image", ha="center", fontsize=9.5, color=DARK,
            style="italic")

    _arrow(ax, 5.0, 5.35, 5.0, 4.5, color=DARK, lw=1.5)
    _rounded(ax, 2.5, 3.1, 5.0, 1.4,
             "Generative model\n(BLIP-Caption / LLaVA)",
             PURPLE, PURPLE, fs=10, weight="bold", tc="white")
    _arrow(ax, 5.0, 3.05, 5.0, 2.1, color=PURPLE, lw=1.5)
    _rounded(ax, 0.8, 0.4, 8.4, 1.6,
             "\"A dog runs through a sunny meadow\nwith flowers.\"",
             GREEN, GREEN, fs=10.5, weight="bold", tc="white")

    # ---------- (c) Retrieval ----------
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("(c) Cross-modal Retrieval",
                 fontsize=12, weight="bold", color=GREEN)

    _rounded(ax, 0.3, 6.4, 9.4, 1.0,
             "Query: \"a cat sitting on a sofa\"",
             BLUE, "white", fs=10, weight="bold", tc=BLUE)
    _arrow(ax, 5.0, 6.35, 5.0, 5.7, color=DARK, lw=1.5)
    _rounded(ax, 2.0, 4.4, 6.0, 1.2,
             "CLIP text encoder\n→ query embedding",
             PURPLE, PURPLE, fs=9.5, weight="bold", tc="white")
    _arrow(ax, 5.0, 4.35, 5.0, 3.7, color=PURPLE, lw=1.5)

    ax.text(5.0, 3.5, "cosine similarity over image index",
            ha="center", fontsize=9, color=DARK, style="italic")

    # Top-k results
    sims = [0.91, 0.88, 0.74, 0.62, 0.55]
    for i, s in enumerate(sims):
        x0 = 0.4 + i * 1.92
        ax.add_patch(Rectangle((x0, 1.4), 1.7, 1.6,
                               facecolor="white",
                               edgecolor=GREEN if i < 2 else GRAY,
                               lw=1.6 + (0.6 if i < 2 else 0)))
        # mini "image"
        mini = rng.uniform(0.3, 0.95, size=(4, 4, 3))
        ax.imshow(mini, extent=(x0 + 0.15, x0 + 1.55, 1.55, 2.85),
                  aspect="auto")
        col = GREEN if i < 2 else GRAY
        ax.text(x0 + 0.85, 1.05, f"{s:.2f}",
                ha="center", fontsize=9, color=col, weight="bold")
        ax.text(x0 + 0.85, 0.55, f"top-{i+1}",
                ha="center", fontsize=8.5, color=col)

    save(fig, "fig4_vl_tasks.png")


# ---------------------------------------------------------------------------
# Figure 5 — Whisper architecture over a mel-spectrogram
# ---------------------------------------------------------------------------
def fig5_whisper_audio() -> None:
    fig = plt.figure(figsize=(13.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.05], wspace=0.18)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # ---- Left: synthetic mel-spectrogram ----
    rng = np.random.default_rng(11)
    T, F = 240, 80
    spec = rng.normal(-25, 7, size=(F, T))
    # add bands of "speech energy" between time 30-200
    t = np.arange(T)
    for f in range(F):
        env = (np.exp(-((t - 80) ** 2) / 1500.0) +
               0.7 * np.exp(-((t - 160) ** 2) / 900.0))
        spec[f] += env * (12 - abs(f - 25) * 0.25)
    spec = np.clip(spec, -45, 5)

    im = axL.imshow(spec, aspect="auto", origin="lower",
                    cmap="magma",
                    extent=(0, 30, 0, 8000))
    axL.set_xlabel("Time (s)")
    axL.set_ylabel("Frequency (Hz, mel scale)")
    axL.set_title("Log-mel spectrogram input  (30 s, 80 mels)",
                  loc="left", fontsize=12, weight="bold")
    cbar = fig.colorbar(im, ax=axL, fraction=0.04, pad=0.02)
    cbar.set_label("dB", fontsize=9)

    # ---- Right: encoder–decoder diagram ----
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 8)
    axR.axis("off")
    axR.set_title("Whisper — Transformer encoder–decoder",
                  loc="left", fontsize=12, weight="bold")

    # Conv front-end
    _rounded(axR, 0.3, 5.8, 2.5, 0.8,
             "2× Conv1D + GELU\n(stride 2, downsample)",
             ORANGE, "white", fs=9.5, weight="bold", tc=ORANGE)
    _arrow(axR, 1.55, 5.75, 1.55, 5.25)

    # Encoder stack
    _rounded(axR, 0.3, 3.0, 2.5, 2.2,
             "Audio Encoder\n(Transformer × N)\nself-attn + FFN",
             BLUE, BLUE, fs=10, weight="bold", tc="white")
    axR.text(1.55, 2.7, "1500 audio tokens",
             ha="center", fontsize=8.5, color=BLUE, style="italic")

    # Cross-attn arrow
    _arrow(axR, 2.85, 4.1, 4.55, 4.1, color=BLUE)
    axR.text(3.7, 4.3, "cross-attn", ha="center", fontsize=8.5,
             color=BLUE, weight="bold", style="italic")

    # Decoder
    _rounded(axR, 4.6, 3.0, 2.6, 2.2,
             "Text Decoder\n(Transformer × N)\nself-attn + cross-attn",
             PURPLE, PURPLE, fs=10, weight="bold", tc="white")

    # Special tokens prepended
    _rounded(axR, 4.6, 5.8, 2.6, 0.8,
             "<|en|> <|transcribe|>\n<|0.00|> ...",
             GRAY, "white", fs=8.5, weight="bold", tc=DARK)
    _arrow(axR, 5.9, 5.75, 5.9, 5.25)

    # Output tokens
    _arrow(axR, 7.25, 4.1, 7.95, 4.1, color=PURPLE)
    _rounded(axR, 8.0, 3.0, 1.7, 2.2,
             "BPE tokens\n→ text\n+ timestamps",
             GREEN, GREEN, fs=9.5, weight="bold", tc="white")

    # Notes
    axR.text(0.3, 1.0,
             "Trained on 680 K hours of weakly-supervised "
             "multilingual audio.\n"
             "Same model → transcription, translation, language ID, "
             "voice activity.\n"
             "Sizes: tiny (39 M)  base (74 M)  small (244 M)  "
             "medium (769 M)  large (1.55 B)",
             fontsize=9, color=DARK,
             bbox=dict(facecolor=LIGHT, edgecolor="none", pad=6))

    fig.suptitle("Whisper — robust multilingual speech recognition",
                 fontsize=14, weight="bold", y=1.02)
    save(fig, "fig5_whisper_audio.png")


# ---------------------------------------------------------------------------
# Figure 6 — GPT-4V capability matrix and example outputs
# ---------------------------------------------------------------------------
def fig6_gpt4v_examples() -> None:
    fig = plt.figure(figsize=(13.8, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.25)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # ---- Left: capability heatmap ----
    capabilities = ["Object\nrecognition",
                    "Scene\nunderstanding",
                    "OCR /\ndocument",
                    "Chart /\ngraph",
                    "Diagram /\nflowchart",
                    "Math /\nformula",
                    "Code /\nUI screenshot",
                    "Spatial\nreasoning",
                    "Counting",
                    "Fine-grained\ngrounding"]
    models = ["GPT-4V", "Claude 3 Opus", "Gemini 1.5 Pro",
              "LLaVA-1.6", "Qwen-VL-Max", "InternVL"]
    rng = np.random.default_rng(3)
    # construct a plausible score matrix in [0, 100]
    base = np.array([
        [92, 90, 89, 84, 87, 78, 91, 76, 82, 68],   # GPT-4V
        [91, 89, 86, 83, 86, 76, 88, 74, 80, 66],   # Claude
        [90, 88, 88, 86, 85, 79, 87, 75, 80, 65],   # Gemini
        [83, 82, 78, 71, 74, 56, 80, 60, 64, 48],   # LLaVA
        [87, 85, 90, 82, 81, 71, 83, 69, 77, 60],   # Qwen-VL
        [85, 84, 82, 79, 78, 68, 81, 66, 72, 55],   # InternVL
    ], dtype=float)
    base += rng.normal(0, 1.2, size=base.shape)

    im = axL.imshow(base, cmap="Blues", aspect="auto", vmin=40, vmax=100)
    axL.set_xticks(range(len(capabilities)))
    axL.set_xticklabels(capabilities, fontsize=8.5, rotation=30,
                        ha="right")
    axL.set_yticks(range(len(models)))
    axL.set_yticklabels(models, fontsize=10, weight="bold")
    axL.set_title("Capability heatmap (illustrative scores)",
                  loc="left", fontsize=12, weight="bold")
    for i in range(len(models)):
        for j in range(len(capabilities)):
            v = base[i, j]
            tc = "white" if v > 78 else DARK
            axL.text(j, i, f"{v:.0f}", ha="center", va="center",
                     fontsize=8, color=tc)
    cbar = fig.colorbar(im, ax=axL, fraction=0.04, pad=0.02)
    cbar.set_label("score", fontsize=9)

    # ---- Right: example output panels ----
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)
    axR.axis("off")
    axR.set_title("GPT-4V example interactions",
                  loc="left", fontsize=12, weight="bold")

    examples = [
        ("Chart reading",
         "Q: \"What is the trend of Q3 revenue?\"",
         "A: \"Revenue grew from $1.2 B in Jul to "
         "$1.7 B in Sep — about 42% QoQ.\"",
         BLUE),
        ("UI screenshot",
         "Q: \"Why is this React component re-rendering?\"",
         "A: \"The `useEffect` on line 14 has no "
         "dependency array, so it runs every render.\"",
         PURPLE),
        ("Hand-written math",
         "Q: \"Solve the equation in the photo.\"",
         "A: \"x² − 5x + 6 = 0 → x = 2 or x = 3.\"",
         GREEN),
        ("Real-world scene",
         "Q: \"Is it safe to cross the street?\"",
         "A: \"The light is red and a car is "
         "approaching from the left — wait.\"",
         ORANGE),
    ]
    for i, (title, q, a, col) in enumerate(examples):
        y = 8.0 - i * 2.15
        axR.add_patch(FancyBboxPatch(
            (0.2, y - 0.1), 9.6, 1.95,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=1.3, edgecolor=col, facecolor="white"))
        axR.text(0.4, y + 1.55, title, fontsize=10.5,
                 weight="bold", color=col)
        axR.text(0.4, y + 0.95, q, fontsize=9.5, color=DARK)
        axR.text(0.4, y + 0.30, a, fontsize=9.5, color=DARK,
                 style="italic")

    fig.suptitle("GPT-4V and frontier MLLMs — capabilities and examples",
                 fontsize=14, weight="bold", y=1.02)
    save(fig, "fig6_gpt4v_examples.png")


# ---------------------------------------------------------------------------
# Figure 7 — Multimodal benchmarks (MMBench, MME, MMMU, etc.)
# ---------------------------------------------------------------------------
def fig7_mllm_benchmarks() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.1, 1]})

    # ---- Left: model x benchmark grouped bars ----
    benchmarks = ["MMBench", "MME\n(perception)",
                  "MMMU", "SEED\n-Bench", "POPE\n(halluc.)"]
    # representative scores reported in 2024 (illustrative, not strict)
    gpt4v   = [77.0, 1409, 56.8, 73.0, 86.4]
    gemini  = [73.6, 1497, 47.9, 70.7, 84.5]
    claude  = [75.1, 1453, 51.2, 72.3, 85.2]
    llava15 = [67.7, 1531, 36.4, 66.1, 87.3]
    qwenvl  = [77.6, 2281, 51.4, 73.4, 86.0]  # Qwen-VL-Max
    intern  = [75.6, 1672, 49.1, 71.8, 85.7]

    # normalise MME (max 2800) to 0-100 for plotting on the same axis
    def n_mme(v):
        return v / 2800 * 100

    def to_pct(row):
        return [row[0], n_mme(row[1]), row[2], row[3], row[4]]

    g = np.array([to_pct(r) for r in
                  (gpt4v, gemini, claude, llava15, qwenvl, intern)])
    models = ["GPT-4V", "Gemini 1.5", "Claude 3", "LLaVA-1.5",
              "Qwen-VL-Max", "InternVL"]
    colors = [BLUE, ORANGE, PURPLE, GRAY, GREEN, DARK]

    x = np.arange(len(benchmarks))
    bw = 0.13
    for i, (m, row, c) in enumerate(zip(models, g, colors)):
        ax1.bar(x + (i - 2.5) * bw, row, bw, label=m,
                color=c, edgecolor="white", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(benchmarks, fontsize=9.5)
    ax1.set_ylabel("Score  (MME shown as % of 2800)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", ncol=2, fontsize=8.5)
    ax1.set_title("Frontier MLLMs across benchmarks",
                  loc="left", fontsize=12, weight="bold")

    # ---- Right: capability radar for two reference models ----
    cats = ["Perception", "Knowledge",
            "Reasoning", "OCR",
            "Cross-domain"]
    vals_g4v = [88, 82, 78, 84, 80]
    vals_l15 = [76, 60, 58, 79, 56]
    vals_qwn = [86, 75, 70, 90, 72]

    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]

    ax2.remove()
    ax2 = fig.add_subplot(1, 2, 2, polar=True)

    def _close(v):
        return v + v[:1]

    ax2.plot(angles, _close(vals_g4v), "-o", color=BLUE, lw=2,
             label="GPT-4V")
    ax2.fill(angles, _close(vals_g4v), color=BLUE, alpha=0.10)
    ax2.plot(angles, _close(vals_qwn), "-s", color=GREEN, lw=2,
             label="Qwen-VL-Max")
    ax2.fill(angles, _close(vals_qwn), color=GREEN, alpha=0.10)
    ax2.plot(angles, _close(vals_l15), "-^", color=ORANGE, lw=2,
             label="LLaVA-1.5")
    ax2.fill(angles, _close(vals_l15), color=ORANGE, alpha=0.10)

    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(cats, fontsize=9.5)
    ax2.set_yticks([20, 40, 60, 80, 100])
    ax2.set_yticklabels(["20", "40", "60", "80", "100"],
                        fontsize=8, color=GRAY)
    ax2.set_ylim(0, 100)
    ax2.set_title("Capability radar (MMBench sub-scores)",
                  fontsize=12, weight="bold", pad=18)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10),
               fontsize=9)

    fig.suptitle("Multimodal benchmarks: MMBench · MME · MMMU · SEED · POPE",
                 fontsize=14, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig7_mllm_benchmarks.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Generating NLP Part 11 figures (Multimodal NLP) ...")
    fig1_clip_architecture();   print("  ok fig1_clip_architecture.png")
    fig2_blip2_qformer();       print("  ok fig2_blip2_qformer.png")
    fig3_llava_architecture();  print("  ok fig3_llava_architecture.png")
    fig4_vl_tasks();            print("  ok fig4_vl_tasks.png")
    fig5_whisper_audio();       print("  ok fig5_whisper_audio.png")
    fig6_gpt4v_examples();      print("  ok fig6_gpt4v_examples.png")
    fig7_mllm_benchmarks();     print("  ok fig7_mllm_benchmarks.png")
    print(f"\nSaved to:\n  {EN_DIR}\n  {ZH_DIR}")


if __name__ == "__main__":
    main()
