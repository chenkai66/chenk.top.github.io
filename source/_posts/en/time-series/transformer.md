---
title: "Time Series Forecasting (5): Transformer Architecture for Time Series"
date: 2024-12-20 09:00:00
tags:
  - Time Series
  - Deep Learning
  - Transformer
categories: Algorithm
series: Time Series Forecasting
lang: en
mathjax: true
description: "Transformers for time series, end to end: encoder-decoder anatomy, temporal positional encoding, the O(n^2) attention bottleneck, decoder-only forecasting, and patching. With variants (Autoformer, FEDformer, Informer, PatchTST) and a real implementation."
disableNunjucks: true
---

## What You Will Learn

- The full encoder-decoder Transformer, redrawn for time series
- Why position must be injected, and how sinusoidal / learned / time-aware encodings differ
- What multi-head attention actually learns over a temporal sequence
- Where vanilla attention breaks down (O(n^2)) and the four families of fixes: sparse, linear, patched, decoder-only
- A clean PyTorch reference implementation, plus when to reach for Autoformer / FEDformer / Informer / PatchTST

## Prerequisites

- Self-attention and multi-head attention (Part 4)
- Encoder-decoder architectures and teacher forcing
- PyTorch fundamentals (`nn.Module`, training loops)

---

## 1. Why Transformers for Time Series

LSTM and GRU process a sequence step by step. Three things follow from
that:

1. **Path length is O(L)**. Information from step $t-L$ has to ride
   through $L$ recurrence steps before it can influence step $t$.
   That's where vanishing gradients come from.
2. **Training is sequential**. Step $t+1$ cannot start until step $t$
   has finished, so a GPU sits half-idle.
3. **The hidden state is a bottleneck**. The model has to compress
   everything it might need from the past into one fixed-size vector.

Self-attention removes all three constraints at once. Every position
sees every other position in **one matrix multiply**, the path length
between any two steps is $O(1)$, and the whole sequence is processed
in parallel. The cost is memory: storing $n \times n$ attention weights
is $O(n^2)$, which we will deal with in Sections 5 and 7.

![Encoder-decoder Transformer adapted for time series. The encoder reads the lookback window in parallel; the decoder generates the forecast horizon and attends to encoder memory through cross-attention.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/transformer/fig1_architecture.png)
*Figure 1. Encoder-decoder Transformer adapted for time series. The encoder reads the lookback window in parallel; the decoder generates the forecast horizon and attends to encoder memory through cross-attention.*

## 2. The Architecture, Block by Block

A time-series Transformer is the original 2017 architecture with three
small but consequential changes:

| Component        | Original NLP                         | Time series                                               |
|------------------|--------------------------------------|-----------------------------------------------------------|
| Input embedding  | Token embedding lookup               | **Linear projection** of continuous features              |
| Positional info  | Sinusoidal on token index            | **Time-aware** encoding (calendar features, irregular dt) |
| Output head      | Softmax over vocabulary              | **Linear** to a real-valued forecast vector               |

Everything else -- multi-head self-attention, feed-forward, residual
connections, LayerNorm, decoder cross-attention, causal masking -- is
unchanged. The four sub-layers per block are:

$$
\begin{aligned}
h_1 &= \text{LayerNorm}(x + \text{MHSA}(x)) \\
h_2 &= \text{LayerNorm}(h_1 + \text{FFN}(h_1))
\end{aligned}
$$

with the standard scaled dot-product attention from Part 4:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

### 2.1 Encoder

Reads the lookback window $x_{t-L+1:t}$ and produces context vectors
$M \in \mathbb{R}^{L \times d_{\text{model}}}$. No mask -- every
position attends to every other.

### 2.2 Decoder

Takes the **label window** (last $L_{\text{label}}$ steps of history)
concatenated with placeholder zeros for the forecast horizon, and emits
the prediction $\hat{y}_{t+1:t+H}$. It uses two attention sub-layers per
block:

- **Masked self-attention** with a causal mask, so step $t+k$ can only
  see steps up to $t+k-1$.
- **Cross-attention** where queries come from the decoder and keys /
  values come from the encoder memory $M$. This is the only place the
  decoder ever looks at the encoder output.

### 2.3 The label window: a small but useful trick

Pure encoder-decoder models often suffer at the boundary between
history and forecast. The fix used by Informer / Autoformer is to feed
the decoder $L_{\text{label}}$ steps of *known history* plus $H$
zero-filled placeholders, so the decoder always starts from a known
state and rolls forward into the unknown.

## 3. Positional Encoding for Time

Self-attention is permutation invariant -- shuffle the input, you get
the same output. For language that's a bug; for time series it's
catastrophic. We inject position with sinusoidal encodings:

$$
\text{PE}_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right), \qquad
\text{PE}_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right).
$$

Each position $p$ ends up with a **unique signature** built from a
geometric series of frequencies. Low-index dimensions oscillate fast
(they encode short-range position), high-index dimensions oscillate
slowly (they encode long-range position).

![Sinusoidal positional encoding. Left: the full encoding matrix where each row is a unique signature. Right: four representative dimensions oscillating at different frequencies.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/transformer/fig2_positional_encoding.png)
*Figure 2. Sinusoidal positional encoding. Left: the full encoding matrix where each row is a unique signature. Right: four representative dimensions oscillating at different frequencies.*

For time series we usually want **richer** positional information than
just "step index":

- **Calendar features**: hour-of-day, day-of-week, month, holiday
  flag. Each gets its own learned embedding and is added to the input.
- **Irregular sampling**: replace position $p$ with the actual
  timestamp $\tau_p$, normalised. Used by Time2Vec and Continuous-Time
  Transformer.
- **Relative position**: encode $\tau_q - \tau_k$ inside the attention
  score itself (T5 / TUPE style). Better for very long contexts.

```python
import torch
import torch.nn as nn
import math

class TemporalPositionalEncoding(nn.Module):
    """Sinusoidal PE + optional calendar feature embeddings."""

    def __init__(self, d_model: int, max_len: int = 5000,
                 calendar_sizes=(24, 7, 31, 12)):
        super().__init__()
        # ---- sinusoidal -----------------------------------------------------
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, d)

        # ---- calendar embeddings (hour, weekday, day, month) ---------------
        self.cal_embeds = nn.ModuleList(
            nn.Embedding(n, d_model) for n in calendar_sizes
        )

    def forward(self, x: torch.Tensor, cal: torch.Tensor | None = None):
        # x: (B, L, d). cal: (B, L, 4) integer calendar features.
        out = x + self.pe[:, : x.size(1)]
        if cal is not None:
            for i, emb in enumerate(self.cal_embeds):
                out = out + emb(cal[..., i])
        return out
```

## 4. What Multi-Head Attention Learns

A single attention head can only model one type of relationship. Multi-
head splits the model into $h$ parallel attention computations on
$d_k = d_{\text{model}} / h$-dimensional projections, then concatenates.
For time series, different heads tend to specialise:

![Four heads from a trained Transformer over a 48-step window, each attending to a different temporal pattern. Note the causal mask: nothing above the diagonal.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/transformer/fig3_multihead_patterns.png)
*Figure 3. Four heads from a trained Transformer over a 48-step window, each attending to a different temporal pattern. Note the causal mask: nothing above the diagonal.*

| Head pattern         | What the model is doing                                  |
|----------------------|----------------------------------------------------------|
| **Local** (diagonal) | Mostly an autoregressive moving average                  |
| **Periodic stripes** | Locking onto a known cycle (24-hour, weekly)             |
| **Long-range diffuse** | Pulling in slow trend information                      |
| **Anchored bands**   | Latching onto a specific past event (spike, regime shift) |

This is also where **interpretability** comes from: averaging the final
layer's attention over heads tells you which historical steps the
forecast actually depends on.

## 5. The O(n^2) Bottleneck

Vanilla attention stores an $n \times n$ score matrix per head per
layer. In fp16 with 8 heads, the per-layer attention memory is

$$
M_{\text{attn}} = h \cdot n^2 \cdot 2 \;\text{bytes}.
$$

This is fine at $n=512$ (4 MB) and uncomfortable at $n=4096$ (256 MB);
at $n=16384$ a single layer needs over 4 GB just for the attention
matrices. Compute scales the same way: each layer is
$O(n^2 d_{\text{model}})$ FLOPs.

![Attention memory and FLOPs as a function of sequence length. Vanilla O(n^2) becomes infeasible past a few thousand steps; sparse / linear / patched alternatives keep the cost manageable.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/transformer/fig5_quadratic_bottleneck.png)
*Figure 5. Attention memory and FLOPs as a function of sequence length. Vanilla O(n^2) becomes infeasible past a few thousand steps; sparse / linear / patched alternatives keep the cost manageable.*

There are four families of fixes, in increasing order of "how much they
change the model":

1. **Sparse attention** (Longformer, BigBird, Informer's ProbSparse):
   only compute scores for a sparse subset of $(q, k)$ pairs. Cost
   $O(n \cdot w)$ where $w$ is a window or number of selected keys.
2. **Linear attention** (Performer, Linformer, Nystromformer): replace
   the softmax with a kernel that factorises, so attention becomes
   $O(n \cdot d^2)$.
3. **Patching** (PatchTST, autoformer-style series decomposition):
   *shorten the sequence itself* by grouping consecutive steps into
   patches. We come back to this in Section 7.
4. **Decoder-only with KV cache** (Section 6): you still pay $O(n^2)$
   in training, but inference is incremental.

In practice, for forecasting horizons up to a few hundred steps with
lookback windows under 2k, vanilla attention is fine. Past that,
**patching is the most cost-effective change** -- it usually improves
accuracy *and* slashes compute.

## 6. Decoder-Only Autoregressive Forecasting

GPT-style decoder-only Transformers have largely won in NLP. The same
recipe works for forecasting: drop the encoder, train one stack with a
causal mask, and roll predictions forward one step at a time.

![Decoder-only forecasting with a causal mask. At each step we feed the model everything it has produced so far and ask for the next value. The mask on the right shows which positions are visible.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/transformer/fig6_decoder_only_forecast.png)
*Figure 6. Decoder-only forecasting with a causal mask. At each step we feed the model everything it has produced so far and ask for the next value. The mask on the right shows which positions are visible.*

```python
@torch.no_grad()
def autoregressive_forecast(model, history: torch.Tensor, horizon: int):
    """history: (B, L, d). Returns (B, horizon, d)."""
    seq = history
    out = []
    for _ in range(horizon):
        pred = model(seq)[:, -1:, :]   # last position is the next-step prediction
        out.append(pred)
        seq = torch.cat([seq, pred], dim=1)
    return torch.cat(out, dim=1)
```

**Trade-offs vs. encoder-decoder**:

| Property               | Encoder-decoder              | Decoder-only                 |
|------------------------|------------------------------|------------------------------|
| Training cost          | Two stacks                   | One stack                    |
| Inference latency      | One forward pass for all $H$ | $H$ forward passes (with KV cache, much cheaper) |
| Exposure bias          | Mitigated by teacher forcing | Present unless you do scheduled sampling |
| Pre-training transfer  | Awkward                      | Natural -- this is how foundation TS models (TimesFM, Lag-Llama, Chronos) are built |

For forecasting from a single foundation model on many tasks,
decoder-only is now the dominant choice.

## 7. Patching: The Single Best Speedup

PatchTST (Nie et al., ICLR 2023) made a quietly revolutionary
observation: **time steps are not the right tokens**. A length-512
hourly series has way more tokens than a typical NLP sentence, but
each "token" carries almost no information. Group them into patches of
size $P$ and you get $\lceil L / P \rceil$ tokens that each summarise a
short waveform.

![Patching strategy. Top: split a length-96 series into eight patches of size 12. Bottom: each patch becomes one token via a linear projection. Right: relative attention cost as a function of patch size -- O(n^2) shrinks fast.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/transformer/fig7_patching.png)
*Figure 7. Patching strategy. Top: split a length-96 series into eight patches of size 12. Bottom: each patch becomes one token via a linear projection. Right: relative attention cost as a function of patch size -- O(n^2) shrinks fast.*

Why patching helps so much:

- **Attention cost drops by $P^2$**. With $P=16$ on $L=512$,
  you go from 262k attention entries per head to ~1k.
- **Each token is meaningful**. A patch of 12 hourly values captures
  half a day -- a useful unit. A single hour does not.
- **Locality bias for free**. The local pattern inside a patch is
  handled by the linear projection; attention only needs to model
  cross-patch (longer-range) interactions.
- **Per-channel independence**. PatchTST treats each variable as its
  own sequence, share weights, and avoids spurious cross-channel
  attention in early training.

```python
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: int, d_model: int, in_channels: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_channels, d_model)

    def forward(self, x):                # x: (B, L, C)
        B, L, C = x.shape
        P = self.patch_size
        L_trim = (L // P) * P
        x = x[:, :L_trim, :].reshape(B, L_trim // P, P * C)
        return self.proj(x)              # (B, L/P, d_model)
```

## 8. A Reference Implementation

Putting it together with `nn.Transformer`:

```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, d_model=128, n_heads=8,
                 n_enc=3, n_dec=2, d_ff=512, dropout=0.1,
                 lookback=512, horizon=96, patch=16):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch, d_model, n_features)
        n_tokens = lookback // patch
        self.pos = TemporalPositionalEncoding(d_model, max_len=n_tokens + horizon)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, n_enc)
        self.decoder = nn.TransformerDecoder(dec_layer, n_dec)
        self.head = nn.Linear(d_model, n_features)
        self.horizon = horizon

    def forward(self, src, tgt):
        # src: (B, L, C). tgt: (B, H, C) -- use teacher forcing in training.
        memory = self.encoder(self.pos(self.patch_embed(src)))
        tgt_emb = self.pos(self.patch_embed(tgt))
        L_tgt = tgt_emb.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(L_tgt).to(src.device)
        out = self.decoder(tgt_emb, memory, tgt_mask=causal)
        return self.head(out)            # (B, L_tgt, C)
```

A few production notes:

- **`norm_first=True`** (pre-LN) is more stable for deep stacks; the
  original post-LN can require warm-up to converge.
- **GELU** rather than ReLU in the FFN -- standard since BERT and
  consistently better in our experience.
- **Always normalise per series** (z-score) before the model and
  invert at the output. Forgetting this is the most common reason a
  Transformer "won't train" on time series.

## 9. Variants and How to Pick One

| Variant      | Key idea                              | Best when                                  | Year |
|--------------|---------------------------------------|--------------------------------------------|------|
| **Vanilla**  | Encoder-decoder + sinusoidal PE       | Lookback < 1k, you want a baseline         | 2017 |
| **Informer** | ProbSparse attention + label window   | Very long lookback (5k-10k)                | 2021 |
| **Autoformer** | Series decomposition + auto-correlation in place of self-attention | Strong, clean seasonality | 2021 |
| **FEDformer**| Attention in the frequency domain     | Periodic data, long horizons               | 2022 |
| **PatchTST** | Patching + channel independence       | Most multivariate forecasting              | 2023 |
| **iTransformer** | Treat each variable as a token, attend across variables | Many correlated channels    | 2024 |

If you're starting fresh in 2024-2025, our default recommendation is
**PatchTST or iTransformer**, both of which beat the older variants on
the standard ETT / Electricity / Traffic benchmarks while being simpler
to implement and faster to train.

## 10. Performance and Engineering

### 10.1 Forecast quality

We forecast a 96-step horizon on a synthetic signal with daily and
weekly seasonality plus random spikes. The Transformer cleanly captures
both seasonalities; the LSTM tracks the dominant daily cycle but drifts
on the weekly component.

![Forecast quality on a daily + weekly seasonal signal. The Transformer locks both cycles; the LSTM captures the dominant daily one but drifts on the weekly. Right: MAE comparison across architectures.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/transformer/fig4_lstm_vs_transformer.png)
*Figure 4. Forecast quality on a daily + weekly seasonal signal. The Transformer locks both cycles; the LSTM captures the dominant daily one but drifts on the weekly. Right: MAE comparison across architectures.*

### 10.2 Training recipe (the boring stuff that matters)

- **Optimizer**: AdamW, $\beta = (0.9, 0.95)$ (the GPT-3 setting -- the
  default 0.999 is too sluggish for time series).
- **Schedule**: linear warm-up over the first 5-10% of steps, then
  cosine decay to zero. Without warm-up, deep Transformers diverge.
- **Learning rate**: start at $1\text{e-}4$ for $d_{\text{model}}=128$,
  scale down for larger models.
- **Gradient clipping**: $\|g\| \le 1.0$. Non-negotiable.
- **Batch size**: as large as fits. Transformers benefit dramatically
  from large batches for stability.
- **Mixed precision** (`torch.cuda.amp` or `bfloat16`): 2-3x speedup
  with no accuracy loss.
- **Patience**: forecasting Transformers usually need 100-300 epochs;
  language Transformers' 3-10 epochs do not transfer.

### 10.3 Production: serving cost and the RevIN trick

- Use **`torch.compile`** (PyTorch 2.x) -- 1.5-2x latency win for free.
- For decoder-only deployments, **cache K and V** across steps so each
  new prediction is $O(n)$ rather than $O(n^2)$.
- **Reversible Instance Normalization** (RevIN, ICLR 2022): normalise
  each input series at inference time, denormalise at the output. A
  one-line change that fixes the "model trained on history that drifts
  away from production" failure mode.

## 11. Common Pitfalls

| Symptom                                       | Likely cause                                                  | Fix                                            |
|-----------------------------------------------|---------------------------------------------------------------|------------------------------------------------|
| Loss flat at the variance of the data        | Forgot to normalise the input                                  | z-score per series, denormalise at output      |
| Loss diverges after a few hundred steps      | No warm-up, post-LN with high LR                               | Linear warm-up + `norm_first=True`             |
| Validation collapses to a constant           | Decoder leaks future via wrong mask                            | Verify `tgt_mask` is strictly upper-triangular |
| OOM at lookback > 1024                       | Vanilla attention                                              | Patching first, then sparse / linear if needed |
| Forecast tracks recent value, ignores trend  | Position not injected, or PE swamped by feature scale          | Scale PE to feature norm; add calendar features |
| "Transformer is worse than LSTM"             | Dataset under 10k samples, model under-regularised             | Smaller model, dropout 0.2-0.3, weight decay   |

## 12. Summary

The Transformer is not magic -- it's the simplest architecture that
gives every time step direct access to every other, in parallel. For
time series, three things matter:

1. **Position is the input** -- without good positional information, a
   Transformer cannot tell a Monday from a Friday. Use sinusoidal PE
   plus calendar features (or relative position for irregular data).
2. **Vanilla attention is O(n^2)** -- and that's only a problem past a
   few thousand steps. The cheapest fix is **patching**, which usually
   improves accuracy too.
3. **Pick the variant that matches the data** -- PatchTST or
   iTransformer for most multivariate problems, FEDformer / Autoformer
   for clean seasonality, decoder-only for foundation-model-style
   transfer.

The fundamental attention formula stays the same throughout:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

Everything in this article is just engineering on top of that.

## Further Reading

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017
- Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021
- Wu et al., *Autoformer: Decomposition Transformers with Auto-Correlation*, NeurIPS 2021
- Zhou et al., *FEDformer: Frequency Enhanced Decomposed Transformer*, ICML 2022
- Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (PatchTST)*, ICLR 2023
- Liu et al., *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, ICLR 2024
- Kim et al., *Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift*, ICLR 2022
