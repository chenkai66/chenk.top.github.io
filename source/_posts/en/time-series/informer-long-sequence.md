---
title: "Time Series Forecasting (8): Informer -- Efficient Long-Sequence Forecasting"
date: 2024-12-20 09:00:00
tags:
  - Time Series
  - Deep Learning
  - Informer
  - Transformer
categories: Algorithm
series: Time Series Forecasting
lang: en
mathjax: true
description: "Informer reduces Transformer complexity from O(L^2) to O(L log L) via ProbSparse attention, distilling, and a one-shot generative decoder. Full math, PyTorch code, and ETT/weather benchmarks."
disableNunjucks: true
series_order: 8
---


The Transformer is wonderful at sequence modeling -- right up to the moment your sequence gets long. Vanilla self-attention costs $\mathcal{O}(L^2)$ in both compute and memory, so a one-week hourly window (168 steps) is fine, a one-month window (720 steps) is painful, and a three-month window (2160 steps) is essentially impossible on a single GPU. That is exactly the regime real-world long-horizon forecasting lives in: weather, energy, finance, IoT.

**Informer** (Zhou et al., AAAI 2021 best paper) is the architecture that finally made Transformers practical for these settings. It does three things, each of which would be a contribution on its own:

1. **ProbSparse self-attention** keeps only the $\mathcal{O}(\log L)$ most informative queries, dropping per-layer cost from $\mathcal{O}(L^2)$ to $\mathcal{O}(L \log L)$.
2. **Self-attention distilling** halves the sequence length between encoder layers, so memory shrinks geometrically with depth.
3. **A generative decoder** predicts the entire forecast horizon in one forward pass instead of running $H$ autoregressive steps.

Combined, the three changes deliver about a 6-10x speedup and 5-10% better MSE than a vanilla Transformer on long-horizon ETT/weather/electricity benchmarks. This chapter unpacks the math behind each one and walks through the implementation.

## What you will learn

- The exact $\mathcal{O}(L^2)$ pain points in vanilla self-attention for long sequences.
- ProbSparse's KL-divergence sparsity measure and its $\max - \mathrm{mean}$ approximation.
- How encoder distilling exchanges sequence length for depth without losing dominant patterns.
- Why the generative decoder is much faster *and* slightly more accurate than autoregressive decoding.
- A complete PyTorch implementation, plus how Informer scores on the ETT and weather benchmarks.

**Prerequisites**: Part 5 (Transformer architecture). Comfort with Big-O reasoning and basic information theory (entropy, KL divergence).

---

## Why long sequences kill the vanilla Transformer

Self-attention computes, for each query $q_i$:

$$
\mathrm{Attn}(q_i, K, V) = \sum_{j=1}^{L} \mathrm{softmax}\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right) v_j.
$$

To do this for every query you need the full $L \times L$ score matrix. Three costs scale as $\mathcal{O}(L^2)$:

- $L$ query-key dot products of dimension $d$: $L^2 d$ FLOPs.
- $L^2$ softmax operations.
- $L^2$ floats of attention-matrix storage during the backward pass.

Concrete numbers for a forecast horizon of $L = 720$ with $d = 64$ and 8 heads on a single sample:

- Attention scores: $720 \times 720 = 518\text{K}$ entries per head per layer.
- Memory: ~16 MB just for attention weights at batch 32 (float32, 8 heads, 3 layers). Activations during backprop push that an order of magnitude higher.
- FLOPs: ~33 M per head per layer, dominated by the $L^2 d$ matmul.

Push to $L = 2160$ and you are looking at almost 5 M attention entries per head, which is enough to OOM a 24 GB GPU at batch sizes anyone uses for training.

Several papers tried to attack this with structural sparsity (Longformer's local + global windows, BigBird's random + global) or low-rank approximation (Linformer, Performer). Informer's pitch is different: **let the data tell you which queries deserve full attention**.

---

## ProbSparse: the queries that matter, the queries that don't

### The intuition

If you plot the attention distribution of a typical query $q_i$ over the keys, you see two qualitatively different shapes:

- **Peaked.** A handful of keys get most of the probability mass. This query is "selective" -- it knows what it is looking for.
- **Uniform.** Probability is spread evenly across all keys. This query is "vague" -- it would benefit from looking at everything.

Peaked queries can be approximated very efficiently by computing attention only over their top few keys. Uniform queries cannot. The trick is to identify which is which **without** computing the full attention matrix first.

### The KL-based sparsity measure

A natural way to measure how peaked a distribution is to compare it to uniform via KL divergence. Let

$$
p(k_j \mid q_i) = \mathrm{softmax}_j\!\left(\frac{q_i^\top k_j}{\sqrt{d}}\right).
$$

Then

$$
\mathrm{KL}(q_i \,\|\, U) = \log L + \frac{1}{L}\sum_{j=1}^{L} \log p(k_j \mid q_i).
$$

After dropping constants and substituting the softmax, Zhou et al. show that

$$
\mathrm{KL}(q_i \,\|\, U) \;\propto\; \log\!\left(\sum_{j=1}^{L} e^{q_i^\top k_j / \sqrt{d}}\right) - \frac{1}{L}\sum_{j=1}^{L} \frac{q_i^\top k_j}{\sqrt{d}}.
$$

Call this quantity $M(q_i, K)$. High $M$ means peaked distribution -- selective query, deserves full attention. Low $M$ means uniform -- vague query, can be skipped.

But computing $M$ exactly still requires the $L$ inner products, defeating the point. Informer's second trick is to approximate $M$ from a **random sample** of $u = c \log L$ keys (with $c$ a constant, typically 5):

$$
\bar{M}(q_i, K) \;=\; \max_{j \in \mathcal{S}} \frac{q_i^\top k_j}{\sqrt{d}} \;-\; \frac{1}{|\mathcal{S}|} \sum_{j \in \mathcal{S}} \frac{q_i^\top k_j}{\sqrt{d}}.
$$

The $\max$ here replaces the LogSumExp because under the concentration of measure that holds for high-dimensional Gaussian-like vectors, LogSumExp is dominated by its largest term. Empirically, $\bar{M}$ ranks queries almost identically to the exact $M$ at a fraction of the cost.

### What ProbSparse actually computes

The procedure for a single attention head:

1. Sample $u = c \log L$ keys uniformly at random.
2. Compute $\bar{M}(q_i, K)$ for every query $q_i$ -- $\mathcal{O}(L \log L)$ work.
3. Pick the top $u$ queries by $\bar{M}$.
4. For those $u$ queries, compute attention over the **full** $L$ keys. For the remaining $L - u$ queries, fill in the mean of $V$ as the output.

Total cost: $\mathcal{O}(L \log L)$. Memory: also $\mathcal{O}(L \log L)$.

![ProbSparse attention vs full attention](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/informer-long-sequence/fig1_probsparse_vs_full.png)

In the figure, the right panel keeps only the rows corresponding to high-$M$ queries. The other rows are not zero in practice -- they are filled with the mean value of $V$, which is a reasonable approximation for a uniform attention distribution.

A clean implementation:

```python
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, factor: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.d_k = d_model // n_heads
        self.factor = factor
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, sample_size, n_top):
        # Q, K: (B, H, L, d_k)
        B, H, L_K, _ = K.shape
        L_Q = Q.size(2)

        # 1) Sample u keys uniformly from K
        idx = torch.randint(0, L_K, (sample_size,), device=K.device)
        K_sample = K[:, :, idx, :]                                  # (B, H, u, d_k)

        # 2) Sparsity measure M_bar(q, K_sample) = max - mean
        Q_K_s = torch.matmul(Q, K_sample.transpose(-2, -1))         # (B, H, L_Q, u)
        M_bar = Q_K_s.max(dim=-1).values - Q_K_s.mean(dim=-1)       # (B, H, L_Q)

        # 3) Pick top-n_top queries
        top_idx = M_bar.topk(n_top, dim=-1).indices                 # (B, H, n_top)

        # 4) Full attention only for those queries
        Q_top = torch.gather(
            Q, 2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        )                                                           # (B, H, n_top, d_k)
        scores = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        return scores, top_idx

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, _ = queries.shape
        L_K = keys.size(1)

        Q = self.W_q(queries).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(keys).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(values).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        u = max(1, min(L_K, self.factor * int(np.ceil(np.log(L_K)))))

        scores, top_idx = self._prob_QK(Q, K, sample_size=u, n_top=u)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask.unsqueeze(0).unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out_top = torch.matmul(attn, V)                              # (B, H, n_top, d_k)

        # Initialise non-selected query outputs with V's mean
        ctx = V.mean(dim=2, keepdim=True).expand(-1, -1, L_Q, -1).clone()
        ctx.scatter_(
            2,
            top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k),
            out_top,
        )

        ctx = ctx.transpose(1, 2).contiguous().view(B, L_Q, self.d_model)
        return self.W_o(ctx)
```

A few subtleties:

- The $u = c \ln L$ formula uses natural log; with `numpy` use `np.log` and round up.
- The "fill non-selected queries with mean of V" step is the mathematically correct treatment, not a hack. It corresponds to the unique distribution that maximises entropy subject to the constraint that the unselected queries have uniform attention.
- For decoder masked self-attention, you must mask out future keys *before* the softmax. The implementation above does the mask as `-1e9` which is the standard trick.

---

## Encoder distilling: pyramidal sequence compression

Even with ProbSparse, three encoder layers each operating on $L = 720$ are expensive. Informer adds a **distilling** operation between encoder layers that halves the sequence length:

$$
X_{\ell+1} = \mathrm{MaxPool}_{k=3, s=2}\!\Big(\mathrm{ELU}\big(\mathrm{Conv1d}_{k=3, s=2}(X_\ell)\big)\Big).
$$

The Conv1d with stride 2 acts as a learned downsampler; the MaxPool keeps the dominant value at each pair of adjacent positions; the ELU non-linearity in between gives the operator some expressivity beyond pure pooling.

![Encoder distilling pyramid](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/informer-long-sequence/fig3_encoder_distilling.png)

The effect compounds: a 3-layer encoder takes a 720-step input through layers of length $720 \to 360 \to 180 \to 90$. Memory is geometric in depth, not linear. And because lower layers see longer history, the receptive field at the top layer easily covers thousands of original time steps.

Two things to be careful about:

- **Last layer should not distil.** The decoder's cross-attention reads from the encoder output; if you distil the very last encoder layer, you halve the resolution again and lose information. The standard recipe is "distil after every layer except the last."
- **Stack two encoders for robustness.** The original paper runs two encoders in parallel, one over the full input and one over a halved version of the input, then concatenates the outputs. The redundancy guards against unlucky distilling decisions on a particular sequence.

```python
class DistillingLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                              stride=2, padding=1)
        self.act = nn.ELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        x = self.conv(x.transpose(1, 2))
        x = self.act(x)
        x = self.pool(x)
        return x.transpose(1, 2)
```

---

## Generative decoder: one shot for the whole horizon

A vanilla Transformer decoder generates the forecast autoregressively: predict $\hat{y}_1$, feed it back, predict $\hat{y}_2$, and so on. For a horizon of $H = 168$ that means 168 sequential decoder forward passes. Latency aside, this also makes errors compound: a mistake at step 5 is fed back as input for step 6.

Informer's generative decoder takes a different approach. The decoder input is constructed as

$$
X_\text{dec} = \big[\, X_\text{token} \;;\; X_0 \,\big],
$$

where $X_\text{token}$ is the last `label_len` time steps of the encoder input (acting as a "prompt") and $X_0$ is `out_len` placeholder tokens (typically zero vectors of the right dimension). The decoder runs **once** over this entire $\text{label\_len} + \text{out\_len}$ sequence, and the last `out_len` outputs are the forecast.

![Autoregressive vs generative decoder](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/informer-long-sequence/fig2_generative_decoder.png)

Three benefits:

- **Speed.** $H$ sequential passes $\to$ 1 pass. Inference latency drops by a factor of $H$.
- **No error compounding.** All horizon predictions are produced from the same encoder context; no prediction depends on a previous prediction.
- **Better long-horizon accuracy.** Counter-intuitively, the generative decoder tends to be *more* accurate than autoregressive decoding for long horizons. The intuition is that the autoregressive decoder is forced into a "myopic" optimisation: each step is trained to match the next token assuming the previous tokens were perfect. The generative decoder optimises the joint forecast directly.

The label tokens are essential: they give the decoder a few "real" data points at the start, which anchors the placeholder tokens. Empirically `label_len = out_len / 2` works well.

---

## Putting it together: the Informer model

The full model is encoder + decoder, with embeddings that combine value, position, and time-feature information.

```python
class TemporalEmbedding(nn.Module):
    """Embeds (hour, day_of_week, month) into d_model."""

    def __init__(self, d_model: int):
        super().__init__()
        self.hour = nn.Embedding(24, d_model)
        self.dow = nn.Embedding(7, d_model)
        self.month = nn.Embedding(12, d_model)

    def forward(self, time_feat):
        # time_feat: (B, L, 3) integer values
        return (self.hour(time_feat[..., 0])
                + self.dow(time_feat[..., 1])
                + self.month(time_feat[..., 2]))


class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, distil=True):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.distil = DistillingLayer(d_model) if distil else None

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.attn(x, x, x)))
        x = self.norm2(x + self.ffn(x))
        if self.distil is not None:
            x = self.distil(x)
        return x


class InformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = ProbSparseAttention(d_model, n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, self_mask=None):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, attn_mask=self_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, enc_out, enc_out)))
        x = self.norm3(x + self.ffn(x))
        return x


class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out,
                 seq_len, label_len, out_len,
                 d_model=512, n_heads=8, e_layers=3, d_layers=2,
                 d_ff=2048, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.out_len = out_len

        self.enc_value = nn.Linear(enc_in, d_model)
        self.dec_value = nn.Linear(dec_in, d_model)
        self.pos_enc = nn.Embedding(seq_len, d_model)
        self.pos_dec = nn.Embedding(label_len + out_len, d_model)
        self.t_emb = TemporalEmbedding(d_model)

        self.encoder = nn.ModuleList([
            InformerEncoderLayer(d_model, n_heads, d_ff, dropout,
                                 distil=(i < e_layers - 1))
            for i in range(e_layers)
        ])
        self.decoder = nn.ModuleList([
            InformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(d_layers)
        ])
        self.head = nn.Linear(d_model, c_out)

    def _embed(self, x_value, time_feat, embed_value, pos_emb):
        B, L, _ = x_value.shape
        positions = torch.arange(L, device=x_value.device).unsqueeze(0).expand(B, L)
        return embed_value(x_value) + pos_emb(positions) + self.t_emb(time_feat)

    def forward(self, x_enc, t_enc, x_dec, t_dec):
        enc = self._embed(x_enc, t_enc, self.enc_value, self.pos_enc)
        for layer in self.encoder:
            enc = layer(enc)

        dec = self._embed(x_dec, t_dec, self.dec_value, self.pos_dec)
        L_dec = dec.size(1)
        # causal mask for decoder self-attention
        causal = torch.tril(torch.ones(L_dec, L_dec, device=dec.device)).bool()
        for layer in self.decoder:
            dec = layer(dec, enc, self_mask=causal)

        return self.head(dec[:, -self.out_len:, :])  # (B, out_len, c_out)
```

For training data construction, the decoder input is built by concatenating the last `label_len` real values with `out_len` zero placeholders:

```python
def build_decoder_input(x_enc, label_len, out_len):
    # x_enc: (B, seq_len, F)
    start = x_enc[:, -label_len:, :]
    placeholder = torch.zeros(
        x_enc.size(0), out_len, x_enc.size(-1), device=x_enc.device
    )
    return torch.cat([start, placeholder], dim=1)
```

---

## Long-horizon performance

The headline visual: ground truth, vanilla Transformer, and Informer on a 480-step horizon.

![Long-horizon forecast: vanilla Transformer drifts, Informer hugs the truth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/informer-long-sequence/fig4_long_sequence_forecast.png)

The vanilla Transformer's autoregressive errors compound around step 100 and drift increasingly far from the truth. Informer's one-shot generative decoder produces a coherent forecast across the full window because all output tokens were optimised jointly.

The numbers from the paper on the canonical ETT (Electricity Transformer Temperature) benchmark:

![ETTh1 univariate MSE and resource cost at L = 720](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/informer-long-sequence/fig5_ett_benchmark.png)

Two takeaways:

- **Accuracy at long horizons.** At horizon 720, Informer's MSE is 0.235 vs the vanilla Transformer's 0.269. Modest in absolute terms, but the vanilla Transformer at horizon 720 is already at the edge of trainability.
- **Resource cost.** At $L = 720$, Informer uses 1.8 GB peak GPU memory and 9.5 s per epoch on a single V100; the vanilla Transformer needs 10.5 GB and 104 s per epoch. The difference is the entire reason Informer exists.

---

## Hyperparameter cheat sheet

| Hyperparameter | Default | Notes |
|---|---|---|
| `d_model` | 512 | Standard; 256 if memory-constrained. |
| `n_heads` | 8 | $d_k = d_\text{model} / n_\text{heads}$, so 64 here. |
| `e_layers` | 3 | More layers $\to$ more aggressive distilling; rarely worth it. |
| `d_layers` | 2 | Asymmetric: encoder does the heavy lifting. |
| `d_ff` | 2048 | 4x `d_model` is standard. |
| `factor` ($c$ in $u = c \log L$) | 5 | 3 for speed, 7 for accuracy; rarely change. |
| `seq_len` | 96 to 720 | Roughly equal to your weekly/monthly cycle length. |
| `label_len` | `out_len / 2` | Anchors decoder; do not zero it out. |
| `out_len` | task-driven | What you actually want to forecast. |
| `dropout` | 0.05-0.1 | More on small datasets. |
| Optimizer | Adam, lr 1e-4 | Half the learning rate of a "normal" Transformer. |
| LR schedule | StepLR, gamma 0.5 every 30 epochs | Or cosine annealing. |
| Loss | MSE | L1 (MAE) is more robust to outliers. |

---

## Common pitfalls

- **Forgetting to mask the decoder self-attention.** Without the causal mask, the decoder can peek at future placeholder tokens during training. Output looks miraculous at training time and garbage at test time.
- **Distilling on a too-short input.** If `seq_len` is short (e.g. 24), three layers of distilling collapse the encoder output to a length of 3, throwing away most of the context. Turn off distilling when `seq_len < 96`.
- **Skipping per-window normalisation.** Same issue as N-BEATS: standardise each window before feeding the model, inverse-transform at output.
- **Not encoding time features.** The temporal embedding (hour-of-day, day-of-week, month) is a major contributor to performance; without it the model has to infer time-of-day from raw data.
- **Tiny `label_len`.** Some implementations default to `label_len = 0`, which destroys decoder anchoring. The paper uses `label_len = out_len / 2`.

---

## When *not* to use Informer

- **Short sequences ($L < 96$).** ProbSparse and distilling have constant overhead; for short sequences a vanilla Transformer or an LSTM is simpler and just as fast.
- **Cross-feature interactions are the main signal** (multivariate with strong inter-feature dependencies). Informer attends along the time axis; for cross-feature attention, look at TFT or TimesNet.
- **You need exact attention patterns for interpretability.** ProbSparse loses the row-wise attention map for non-selected queries. If you must visualise full attention, use a vanilla Transformer.
- **Streaming inference at high frequency** (kHz, MHz). Informer is built for batch forecasting; streaming requires more specialised architectures.
- **Very small datasets (<1k samples).** Informer has tens of millions of parameters and will overfit. Use a smaller, less expressive model.

---

## Q&A

**Why $u = c \log L$ specifically?**
The bound comes from the expected probability that a "selective" query has its top key sampled. With $u = c \log L$ and $c = 5$, the probability that we miss the top-1 key for any given query is $\leq 1/L^4$. In practice $c = 3$ also works fine.

**Does ProbSparse actually identify the *right* queries?**
Empirically yes -- the $\max - \mathrm{mean}$ approximation correlates near-perfectly (>0.95 Spearman) with the exact KL divergence on attention distributions seen during training. The paper has the full ablation.

**Why use the *mean* of $V$ for non-selected queries instead of zero?**
Because a uniform attention distribution evaluates to exactly $\frac{1}{L}\sum_j v_j$. The mean is the analytically correct fill-in for queries we have classified as "uniform attention".

**How is Informer different from Reformer / Performer / Linformer?**
- **Reformer** uses LSH-bucketed attention. Cost $\mathcal{O}(L \log L)$ but the bucketing is data-independent.
- **Performer** uses random-feature kernel approximation. Cost $\mathcal{O}(L)$ but accuracy degrades on long sequences with sharp attention.
- **Linformer** projects keys/values to a fixed low-rank dimension. Cost $\mathcal{O}(L)$ but the projection is fixed at training time.
- **Informer** picks queries adaptively based on data. Cost $\mathcal{O}(L \log L)$, best accuracy retention on time-series benchmarks.

**Can I use Informer for multivariate input with variable encoder/decoder feature dimensions?**
Yes -- `enc_in` and `dec_in` are independent. A common pattern is to feed all variables into the encoder and only the target variable into the decoder.

**What about Autoformer / FEDformer?**
Both are direct successors. Autoformer (2021) replaces self-attention with autocorrelation along the series and adds an explicit decomposition layer. FEDformer (2022) adds frequency-domain attention. Both beat Informer on the same benchmarks but are more complex to implement; Informer is the right starting point.

**Should I always pre-train on a large multi-series dataset?**
Helpful but not required. Unlike NLP, the domain gap between time-series datasets is large, and naive pre-training often hurts more than it helps. Domain-specific fine-tuning from scratch is usually the better default.

---

## Summary

Informer is the architecture that made Transformers practical for long-horizon time-series forecasting. The three core ideas -- ProbSparse self-attention, encoder distilling, and a generative decoder -- compose into an end-to-end $\mathcal{O}(L \log L)$ system that beats the vanilla $\mathcal{O}(L^2)$ Transformer in both accuracy and wall-clock time on every long-horizon benchmark.

For a forecasting task with $L > 96$ on a single GPU, Informer is the obvious starting point. Newer architectures (Autoformer, FEDformer, PatchTST) refine the recipe further, but each builds on Informer's central observation that **not every query needs full attention** and that **autoregressive decoding is a self-imposed bottleneck**.

This concludes the time-series forecasting series. Across eight chapters we walked from classical ARIMA to LSTM to Transformer to TCN to N-BEATS to Informer; pick the architecture that matches your data, ensemble when it counts, and remember that simple baselines often beat fashionable models on small problems.

---

## References and further reading

- Zhou, H. et al. (2021). *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI Best Paper.
- Wu, H. et al. (2021). *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting.* NeurIPS.
- Zhou, T. et al. (2022). *FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting.* ICML.
- Nie, Y. et al. (2023). *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST).* ICLR.
- Wang, S. et al. (2020). *Linformer: Self-Attention with Linear Complexity.* arXiv:2006.04768.
- Choromanski, K. et al. (2021). *Rethinking Attention with Performers.* ICLR.

*This article concludes the Time Series Forecasting series. Use the navigation at the top to revisit earlier chapters.*
