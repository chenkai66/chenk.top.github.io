---
title: "Time Series Forecasting (4): Attention Mechanisms -- Direct Long-Range Dependencies"
date: 2024-10-16 09:00:00
tags:
  - Time Series
  - Deep Learning
  - Attention
categories: Algorithm
series: Time Series Forecasting
lang: en
mathjax: true
description: "Self-attention, multi-head attention, and positional encoding for time series. Step-by-step math, PyTorch implementations, and visualization techniques for interpretable forecasting."
disableNunjucks: true
series_order: 4
---


## What you will learn

- Why recurrent models hit a wall on long-range dependencies, and how attention removes it.
- The Query / Key / Value mechanism, scaled dot-product attention, and the role of $1/\sqrt{d_k}$.
- Two classic scoring functions -- **Bahdanau** (additive) and **Luong** (multiplicative).
- How to wire **attention into an LSTM encoder/decoder** for time series.
- **Multi-head attention** specialised for time -- different heads for recency, period, anomaly.
- The $O(n^2)$ memory wall and how sparse / linear attention bypass it.
- A worked **stock-prediction case** with attention-weight overlays.

**Prerequisites**: RNN/LSTM/GRU intuition (Parts 2-3), basic linear algebra, PyTorch.

---

## 1. Why attention? The bottleneck of recurrence

In a length-$n$ recurrent model, the path between two time steps that are $k$ apart is **$O(k)$ steps long**. Every step squeezes information through a single hidden vector, and every step risks attenuating the gradient.

Real time series rarely cooperate with that geometry:

- An ECG anomaly minutes ago matters more than the last 200 samples of baseline.
- Today's electricity load looks most like *the same hour, last Wednesday*.
- A stock price reacts to an **earnings event** that happened weeks ago.

Attention proposes a radically different geometry: every step has a **direct, learned link to every other step**. The path length between any two positions becomes $O(1)$, and the link strength -- the *attention weight* -- is itself interpretable.

![Attention weights over a 24-step window. The bright diagonal is recency, the off-diagonal is a 12-step periodicity, and the vertical band is persistent memory of an anomaly at t=5.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/attention-mechanism/fig1_attention_heatmap.png)
*Figure 1. A causal attention map already encodes three useful priors -- recency, periodicity, and persistent memory of anomalies -- without any handcrafted features.*

---

## 2. Scaled dot-product attention from first principles

Stack the input sequence as a matrix $X \in \mathbb{R}^{n \times d}$, one row per time step. Three learned linear maps produce three "views" of the same data:

$$
Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V,
$$

with $W^Q, W^K \in \mathbb{R}^{d \times d_k}$ and $W^V \in \mathbb{R}^{d \times d_v}$.

- **Query** $Q$ -- "what is this step looking for?"
- **Key** $K$ -- "what does this step advertise?"
- **Value** $V$ -- "what does this step actually carry?"

The compatibility between query $i$ and key $j$ is a dot product. Stacking:

$$
\text{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V.
$$

### Why divide by $\sqrt{d_k}$?

If the entries of $Q$ and $K$ are i.i.d. with variance $1$, then each dot product $q_i^\top k_j$ has variance $d_k$. As $d_k$ grows, the softmax inputs become large in magnitude and the softmax saturates -- gradients collapse to near zero on all but one position. Dividing by $\sqrt{d_k}$ rescales the variance back to $1$ and keeps gradients healthy.

### A minimal implementation

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K, V: (batch, seq_len, d). mask: (batch, seq_len, seq_len) or broadcastable."""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)            # (B, n, n)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)                          # row-stochastic
    return weights @ V, weights                                  # (B, n, d_v), (B, n, n)
```

The whole mechanism is two matrix multiplications surrounding a softmax. The expressive power lives in the learned projections $W^Q, W^K, W^V$.

---

## 3. Bahdanau vs Luong: two classic scoring functions

Before the Transformer, Bahdanau et al. (2015) introduced **additive attention** for sequence-to-sequence translation, and Luong et al. (2015) followed with **multiplicative (dot-product)** variants. Both are still useful when you wire attention into an RNN.

![Bahdanau (additive) vs Luong (multiplicative) attention scoring functions.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/attention-mechanism/fig2_bahdanau_vs_luong.png)
*Figure 2. Two ways to score query-key compatibility. Additive uses a small MLP, multiplicative uses a dot product. Transformers use multiplicative with the $1/\sqrt{d_k}$ scaling factor.*

| Property | Bahdanau (additive) | Luong (multiplicative) |
|---|---|---|
| Score | $v^\top \tanh(W_1 h_i + W_2 s_{t-1})$ | $s_t^\top W h_i$ |
| Cost per pair | One MLP forward | One dot product |
| Parametrisation | $v, W_1, W_2$ | $W$ (often identity) |
| Best when | Query and key live in different spaces | $Q$ and $K$ share a space |
| Modern usage | Rare in pure Transformers | Standard (with $1/\sqrt{d_k}$) |

Both produce a vector of pre-softmax scores, both finish with softmax + weighted sum. The Transformer simply chose the *cheaper* one and added the scale factor.

---

## 4. Self-attention applied to a time series

In the seq2seq world, queries come from the decoder and keys/values come from the encoder -- two different sequences. **Self-attention** drops that distinction: the same sequence acts as $Q$, $K$, and $V$. Each step looks at every other step in the *same* window.

For time series this is exactly what we want. Suppose we want to forecast the next value given a 12-step window. The attention weights from "now" back into the past tell us which historical step the model is leaning on.

![Self-attention from the most recent step into a 12-step window. Arc thickness is proportional to the attention weight; the bottom bar chart shows the same weights.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/attention-mechanism/fig3_self_attention_ts.png)
*Figure 3. The forecast query at $t = 11$ does not just look at $t = 10$; it also strongly attends to $t = 5$, six steps back, because the underlying signal has period $\approx 6$. Attention has discovered the seasonality without being told.*

### Causal masking

For forecasting we must prevent step $i$ from looking at the future. The standard fix is a **causal mask** -- a lower-triangular matrix added to the scores, with $-\infty$ in the upper triangle so the softmax kills those entries:

```python
def causal_mask(n, device):
    return torch.tril(torch.ones(n, n, device=device)).bool()  # 1 below diag

scores = scores.masked_fill(~causal_mask(n, scores.device), float("-inf"))
```

This is the only line that distinguishes a forecasting Transformer from a sequence-classification Transformer.

---

## 5. Multi-head attention, specialised for time

A single attention map averages whatever patterns it finds. Multi-head attention runs $h$ independent attentions in parallel, each on a slice of the embedding, then concatenates and projects:

$$
\text{MultiHead}(X) = [\text{head}_1; \dots; \text{head}_h] \, W^O,
\qquad
\text{head}_j = \text{Attention}(X W^{Q}_j, X W^{K}_j, X W^{V}_j).
$$

Each head has its own $W^Q_j, W^K_j, W^V_j \in \mathbb{R}^{d \times (d/h)}$ and is free to specialise. In time series we typically observe four families of heads after training:

![Four attention heads, each specialising in a different temporal pattern: local recency, long-range trend, lag-7 periodicity, and persistent memory of an anomaly at t=4.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/attention-mechanism/fig6_multihead_for_time.png)
*Figure 4. The same 18-step window seen through four heads. Multi-head attention is essentially a learned ensemble of temporal kernels.*

| Head | What it learns | Why it matters |
|---|---|---|
| Local | Sharp diagonal | Short-term momentum |
| Long-range | Diffuse triangle | Slow drift, regime |
| Periodic | Off-diagonal stripes | Daily / weekly cycles |
| Anomaly | Vertical column | "Remember the spike at $t=k$" |

PyTorch implementation:

```python
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, n, _ = x.shape
        # Project, reshape to (B, h, n, d_k)
        q = self.W_q(x).view(B, n, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, n, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, n, self.h, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = self.dropout(F.softmax(scores, dim=-1))
        out = weights @ v                                # (B, h, n, d_k)
        out = out.transpose(1, 2).reshape(B, n, -1)      # (B, n, d_model)
        return self.W_o(out), weights
```

**How many heads?** For $d_\text{model} = 64\!-\!128$, four heads is a sensible starting point. If you visualise heads after training and several are near-identical, *reduce*. If a single head is trying to encode multiple distinct patterns, *increase*.

---

## 6. Positional encoding: putting time back in

Self-attention is **permutation-invariant** -- shuffling the input shuffles the output identically. For time series, that throws away the most important variable in the dataset. We must inject position explicitly.

### Sinusoidal encoding

The original Transformer adds fixed sinusoids of geometrically spaced frequencies:

$$
PE_{(p, 2i)} = \sin\!\left(\frac{p}{10000^{2i/d}}\right),
\qquad
PE_{(p, 2i+1)} = \cos\!\left(\frac{p}{10000^{2i/d}}\right).
$$

Why this exact form?

- **Boundedness**: every entry sits in $[-1, 1]$, regardless of $p$.
- **Linear shift-equivariance**: $PE_{p+\Delta}$ is a fixed linear function of $PE_p$, so the model can learn relative offsets like "look 7 steps back" with a single linear projection.
- **Multi-scale**: low-index dimensions move slowly (long-term position), high-index dimensions move quickly (fine-grained position).

### Time-aware encoding for irregular sampling

When samples are not equally spaced (sensor data, trades), feed the **actual timestamp difference** rather than the index. A common pattern:

```python
def time_features(timestamps, d_model):
    """timestamps: (B, n) in seconds. Returns (B, n, d_model)."""
    deltas = timestamps - timestamps[:, :1]              # seconds since window start
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2) / d_model))
    args = deltas.unsqueeze(-1) * freqs                  # (B, n, d_model/2)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
```

This generalises sinusoidal PE to arbitrary time intervals -- the same code handles 1 Hz IoT data, irregular trade ticks, and missing samples uniformly.

---

## 7. Attention + LSTM: the practical hybrid

Pure Transformers shine on long sequences but require a lot of data. For windows in the **50-500 step** range, a hybrid is often the strongest baseline: an LSTM extracts local temporal features cheaply, and attention then chooses which encoder state matters at each forecast step.

![LSTM encoder + attention + LSTM decoder. The encoder produces hidden states h1...h5; attention scores them against the decoder state and forms a context vector that conditions the prediction.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/attention-mechanism/fig5_attention_lstm_hybrid.png)
*Figure 5. The hybrid keeps the LSTM's strong inductive bias for local sequential structure and uses attention as a learned, content-based pointer back into history.*

```python
class LSTMAttention(nn.Module):
    def __init__(self, n_features, hidden, horizon):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(n_features + hidden, hidden, batch_first=True)
        # Luong-style multiplicative attention
        self.W_a = nn.Linear(hidden, hidden, bias=False)
        self.head = nn.Linear(hidden * 2, 1)
        self.horizon = horizon

    def forward(self, x, last_obs):
        H, (h, c) = self.encoder(x)                     # H: (B, n, hidden)
        outs = []
        y_prev = last_obs                                # (B, 1, n_features)
        for _ in range(self.horizon):
            s = h[-1]                                    # (B, hidden)
            scores = (self.W_a(s).unsqueeze(1) * H).sum(-1)         # (B, n)
            alpha = F.softmax(scores, dim=-1)
            ctx = (alpha.unsqueeze(-1) * H).sum(1)                  # (B, hidden)
            dec_in = torch.cat([y_prev, ctx.unsqueeze(1)], dim=-1)
            o, (h, c) = self.decoder(dec_in, (h, c))
            y = self.head(torch.cat([o.squeeze(1), ctx], dim=-1))
            outs.append(y)
            y_prev = y.unsqueeze(1).expand(-1, 1, x.size(-1))       # naive feedback
        return torch.cat(outs, dim=1), alpha
```

Empirically this architecture (or variants -- DA-RNN, dual-stage attention, etc.) wins many M-competition style benchmarks, especially when the horizon is short and data is limited.

---

## 8. The $O(n^2)$ wall and how to escape it

The attention matrix has $n^2$ entries. Every entry is computed and stored. For a 4096-step window with float32, that is 64 MB *per head*, *per layer*, *per example*. The wall is real.

![Compute and memory scaling: RNN is O(n) memory, full attention is O(n^2). Sparse attention is O(n log n) and linear attention is O(n).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/attention-mechanism/fig4_complexity_vs_length.png)
*Figure 6. RNNs win on memory; attention wins on parallelism. The crossover in compute lies near $n \approx d$. For $n \gg d$ you need a sub-quadratic variant.*

| Variant | Time | Memory | Idea |
|---|---|---|---|
| Full attention | $O(n^2 d)$ | $O(n^2)$ | Compute every pair |
| Sparse / strided | $O(n \log n \cdot d)$ | $O(n \log n)$ | Local windows + dilated jumps (Longformer, BigBird) |
| Linear attention | $O(n d^2)$ | $O(n d)$ | Replace softmax with a kernel feature map (Linformer, Performer) |
| Informer ProbSparse | $O(n \log n \cdot d)$ | $O(n \log n)$ | Score only the top-$\log n$ queries (covered in Part 8) |

For most time-series problems, $n$ is in the hundreds, $d$ is in the tens to a few hundred, and the crossover with RNNs sits in your favour. Reach for sub-quadratic variants only when the standard implementation runs out of memory.

---

## 9. Case study: forecasting a stock price

To make the whole pipeline concrete, here is a synthetic stock series with three regimes -- a slow trend, a 30-day cycle, and an earnings event at day 60 -- forecast 10 days ahead by an LSTM+attention model and a no-attention baseline.

![Stock price forecast with attention overlay. The LSTM+attention forecast (orange) tracks the post-earnings regime; the no-attention baseline (grey dashed) under-shoots. The bottom panel shows the attention weights from the forecast query into the past 30 days, with the earnings day spiking in red.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/attention-mechanism/fig7_stock_attention_app.png)
*Figure 7. The attention weights are not a black box. We can see the model leaning hard on the earnings event and on the most recent week.*

Three things to notice:

1. **The earnings-day weight is large** -- attention has discovered an *event* memory without being told what an earnings release is.
2. **The cycle peak is preserved** -- the orange forecast follows the 30-day oscillation, while the baseline collapses to a near-linear extrapolation.
3. **Interpretability comes for free** -- the same matrix that drives the prediction also explains it. With LSTMs you would need post-hoc tools (integrated gradients, SHAP); with attention the explanation is a softmax row.

A note of caution: attention weights are **correlated with importance, not identical to it**. For high-stakes deployments, validate explanations with perturbation tests (zero-out a key step, see the prediction shift) rather than reading the heatmap as ground truth.

---

## 10. Practical recipe for time-series attention

1. **Standardise inputs**. Attention scores are dot products; without normalisation, large-scale features dominate.
2. **Add positional encoding**. Sinusoidal for regular sampling, time-aware for irregular.
3. **Use causal masks** during training and inference for forecasting.
4. **Start with 4 heads, $d_\text{model} \in [64, 128]$**. Scale only if validation loss demands it.
5. **Layer-norm before attention**, dropout on attention weights and on the feed-forward block.
6. **Lower learning rate than RNNs** -- $10^{-4}$ to $5 \cdot 10^{-4}$ with a warm-up of a few hundred steps.
7. **Visualise heads early**. If they collapse to identical patterns, reduce $h$ or add diversity regularisation.
8. **Beware the $O(n^2)$ wall**. If you need $n > 1024$, go straight to a sub-quadratic variant or to Informer (Part 8).

---

## 11. Common pitfalls

- **Forgetting to scale by $\sqrt{d_k}$** -- training stalls within a few steps.
- **Wrong masking** -- subtle data leakage that inflates training metrics and crashes at deployment.
- **Attention to padding** -- forgetting the padding mask leaks the special token's signal into every position.
- **Treating weights as causal explanations** -- they are evidence, not proof.
- **Training on too-short windows** -- if all useful history fits in 10 steps, an LSTM will probably outrun a Transformer.

---

## 12. Summary

Attention replaces the **sequential, lossy information channel of an RNN** with a **direct, content-addressable lookup**. The math is two matrix multiplications and a softmax; the consequences are profound:

- $O(1)$ path length between any two time steps.
- Fully parallel training -- every position is computed at once.
- Built-in interpretability via the attention matrix.
- A clean abstraction for multi-scale temporal patterns through multi-head attention.

The price is $O(n^2)$ memory and the need to inject position explicitly. For most time-series problems those costs are well worth paying -- and Parts 5, 6, and 8 of this series will show how the Transformer, TCN, and Informer push the idea further.

> **Mnemonic** -- *Q asks, K answers, V carries; scale by $\sqrt{d_k}$, softmax to weights, multiply V to read; many heads, many views.*

---

## References

1. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
2. Bahdanau, Cho, Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015.
3. Luong, Pham, Manning, *Effective Approaches to Attention-based Neural Machine Translation*, EMNLP 2015.
4. Qin et al., *A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction*, IJCAI 2017.
5. Kitaev, Kaiser, Levskaya, *Reformer: The Efficient Transformer*, ICLR 2020.
6. Beltagy, Peters, Cohan, *Longformer: The Long-Document Transformer*, 2020.
7. Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. -- covered in Part 8.

---

**Series navigation**

