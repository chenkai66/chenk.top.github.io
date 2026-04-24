---
title: "Time Series Forecasting (2): LSTM -- Gate Mechanisms and Long-Term Dependencies"
date: 2024-11-02 09:00:00
tags:
  - Time Series
  - Deep Learning
  - LSTM
categories: Algorithm
series: Time Series Forecasting
lang: en
mathjax: true
description: "How LSTM's forget, input, and output gates solve the vanishing gradient problem. Complete PyTorch code for time series forecasting with practical tuning tips."
---
> **Series**: Time Series Forecasting -- Part 2 of 8
> [<-- Previous: Traditional Models](/en/time-series-1-traditional-models/) | [Next: GRU -->](/en/time-series-gru/)

## What You Will Learn

- Why vanilla RNNs fail on long sequences and how LSTM fixes the gradient problem
- The intuition behind each gate (forget, input, output) and the cell-state "highway"
- How to structure inputs/outputs for one-step and multi-step time series forecasting
- Practical recipes: regularization, sequence length, bidirectional vs stacked LSTM, when to choose LSTM vs GRU

## Prerequisites

- Basic understanding of neural networks (forward pass, backpropagation)
- Familiarity with PyTorch (`nn.Module`, tensors, optimizers)
- Part 1 of this series (helpful but not required)

---

## 1. The Problem LSTM Solves

A vanilla RNN updates its hidden state recursively:
$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b).$$

When you backpropagate the loss at step $T$ to a much earlier step $k$, the gradient picks up a long product of Jacobians:
$$\frac{\partial h_T}{\partial h_k} = \prod_{t=k+1}^{T} \mathrm{diag}\!\left(1 - h_t^2\right) W_h.$$

Two regimes appear:

- If the dominant singular value of $W_h$ is below 1, the product **vanishes** exponentially and the network cannot learn from anything more than ~10 steps in the past.
- If it is above 1, the product **explodes** and training diverges.

LSTM (Hochreiter & Schmidhuber, 1997) replaces the single recurrent state with **two** states and three learned gates that decide what to remember, what to overwrite, and what to expose. The result is a near-additive update over time, which lets gradients survive the long walk back.

## 2. Anatomy of an LSTM Cell

Inside one cell, four gating units share the same input $[h_{t-1}, x_t]$ and emit three sigmoid gates plus one $\tanh$ candidate:

$$
\begin{aligned}
f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{forget gate} \\
i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{input gate} \\
\tilde C_t &= \tanh(W_C [h_{t-1}, x_t] + b_C) && \text{candidate} \\
o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{output gate}
\end{aligned}
$$

These four signals combine into the cell-state update and the hidden output:

$$
C_t = f_t \odot C_{t-1} + i_t \odot \tilde C_t, \qquad
h_t = o_t \odot \tanh(C_t).
$$

The product $\odot$ is element-wise. **Read this in plain English**: erase the fraction $1 - f_t$ of old memory, write the fraction $i_t$ of fresh candidate, then look at the result through the lens $o_t$.

![LSTM cell architecture: three sigmoid gates plus a tanh candidate sit beneath a horizontal cell-state highway. The forget gate multiplies the incoming cell state, the input gate scales the candidate, and the output gate exposes a filtered view as the hidden state.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig1_lstm_cell.png)
*LSTM cell — three gates over a cell-state highway.*

### Why the cell state is special

The hidden state $h_t$ is what the rest of the network sees, but the **cell state** $C_t$ is where memory actually lives. It runs as an unbroken horizontal line across time and is touched only by element-wise multiplication ($f_t$) and addition ($i_t \odot \tilde C_t$) — never by a fresh matrix multiplication. That single design choice is the reason gradients survive across hundreds of steps.

![Two parallel state streams across time: the green cell-state highway carries long-term memory with near-identity updates, while the dashed purple hidden-state line is the gated, filtered view that downstream layers actually see.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig2_state_highway.png)
*Cell state vs hidden state — two parallel streams.*

### Gradient flow, made explicit

Differentiating the cell update with respect to a much earlier cell state gives
$$\frac{\partial C_t}{\partial C_{t-1}} = f_t,$$
so the long-range gradient is a **product of forget gates**, not a product of $\tanh$ derivatives times a recurrent matrix:
$$\frac{\partial C_T}{\partial C_k} = \prod_{t=k+1}^{T} f_t.$$

Whenever the model wants to remember, it can learn to push $f_t$ close to 1 for the relevant coordinates, and the corresponding gradient stays close to 1 too. That is the entire trick.

## 3. A Minimal PyTorch Implementation

For a univariate or multivariate forecaster, this is all you need:

```python
import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2,
                 output_size=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)              # (batch, seq_len, hidden_size)
        return self.head(out[:, -1, :])     # use last time step
```

A few non-obvious points:

- `batch_first=True` makes the input shape `(batch, seq_len, features)`, which is the convention everyone outside the original PyTorch examples expects.
- The built-in `dropout` argument is **inter-layer** only — it does *not* drop activations between time steps. For recurrent dropout, use `nn.LSTMCell` and apply a fixed mask yourself, or use the `weight_drop` trick from AWD-LSTM.
- Initialize the forget-gate bias to **+1** so the network starts in "remember" mode. PyTorch does not do this by default:

```python
for name, p in model.lstm.named_parameters():
    if "bias" in name:
        n = p.size(0)
        p.data[n // 4 : n // 2].fill_(1.0)   # forget-gate bias
```

## 4. From Cell to Forecaster

For time series, the loop is:

1. **Window the series** into overlapping sequences of length $L$ — the *lookback*.
2. **Standardize** each feature using the training set's mean and standard deviation.
3. Train the model to predict the next value (one-step) or the next $H$ values (multi-step).
4. Validate on a **chronologically held-out** tail — never shuffle.

A clean one-step forecast on a noisy seasonal signal looks like this:

![One-step LSTM forecast versus the actual series. The shaded blue band is a 95% interval derived from the residual standard deviation on the test window. The model tracks both the dominant 24-step and the slower 75-step seasonal components with a small, characteristic lag of one step.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig3_forecast.png)
*One-step LSTM forecast vs actual on a noisy seasonal signal.*

### Multi-step ahead: recursive vs direct

For horizon $H > 1$ there are two common strategies:

| Strategy | How | Trade-off |
| --- | --- | --- |
| **Recursive** | Train one one-step model; feed its prediction back as input. | Simple, but errors compound — variance grows like $\sqrt{H}$. |
| **Direct** | Train $H$ separate heads (or a single model with an $H$-dim output) to predict each future step directly. | More parameters, but no error feedback loop. |

![Multi-step ahead forecast: amber recursive predictions show a fan-shaped uncertainty band that widens with horizon, while the green direct forecaster keeps a roughly constant band. The dotted vertical line marks the forecast origin.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig4_multistep.png)
*Multi-step ahead — recursive forecasts accumulate error; direct forecasts pay in parameters but stay tighter.*

A useful hybrid is **seq2seq with teacher forcing**: an LSTM encoder reads the lookback window into a final $(h, C)$ pair, an LSTM decoder generates $H$ outputs one at a time, and during training the decoder receives the *true* previous value (not its own prediction) with some scheduled-sampling probability. This is what most production forecasters use today.

## 5. Architectural Variants

### Bidirectional LSTM

A BiLSTM runs one LSTM forward and a second one backward, then concatenates the two hidden states at each step:
$$y_t = [\,\overrightarrow{h}_t \,;\, \overleftarrow{h}_t\,].$$

![Bidirectional LSTM: forward purple chain reads left-to-right, backward amber chain reads right-to-left, and the output at each step is the concatenation of both hidden states.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig5_bilstm.png)
*Bidirectional LSTM — combines past and future context per step.*

**Use it for** sequence labeling, classification, missing-value imputation — any setting where the *whole* sequence is in hand at inference time. **Do not use it for** real-time forecasting: peeking at $x_{t+1}, x_{t+2}, \dots$ during training while predicting $x_{t+1}$ at inference is data leakage.

### Stacked (deep) LSTM

Stacking layers lets later layers operate on smoother, slower features. Layer 1 sees the raw input; layer 2 sees the hidden states of layer 1, and so on.

![Stacked LSTM with three layers: each layer recurses left-to-right and feeds its hidden state up to the next layer. Lower layers capture short, local structure; higher layers compose longer-range, more abstract patterns.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig6_stacked_lstm.png)
*Stacked LSTM — hierarchical feature extraction in time.*

In practice, **2–3 layers** is the sweet spot for forecasting. Going deeper rarely helps without residual connections, and it markedly increases the risk of vanishing gradients along the *depth* axis (between layers, not time).

## 6. Training Recipe That Actually Works

The following defaults work on most univariate or moderately multivariate forecasting problems with a few thousand to a few hundred thousand training windows:

| Knob | Default | When to deviate |
| --- | --- | --- |
| Lookback $L$ | 2–3 dominant periods of the series | Use autocorrelation to pick — see below |
| `hidden_size` | 64 | Up to 128–256 for $\geq$ 50 k windows |
| `num_layers` | 2 | 1 if data is small; 3 only with residual connections |
| `dropout` | 0.2 | Up to 0.5 if you see overfitting |
| Optimizer | Adam, lr = 1e-3 | Switch to AdamW + cosine schedule for long runs |
| Batch size | 32–64 | Larger only if you scale lr like $\sqrt{B/32}$ |
| Loss | MSE or Huber | Huber if the target has heavy tails / outliers |
| Gradient clipping | `clip_grad_norm_(..., 1.0)` | Always — cheap insurance against exploding updates |
| Early stopping | patience = 8–10 | On validation loss, with model-state restore |

### Picking the lookback

Plot the autocorrelation function and find the largest lag $k$ where $|\rho_k|$ is still above a small threshold (say 0.1). Round up to the next dominant seasonal period. For hourly data with daily and weekly cycles, 168 (one week) is a natural ceiling.

### Anatomy of a good training run

A healthy LSTM training curve has the validation loss following the training loss closely until it bottoms out, then drifting upward as the model starts to memorize. Early stopping triggers a fixed number of epochs after the best validation loss and restores the best weights:

![Training and validation loss across 60 epochs. The green dashed line marks the best validation epoch (~35); the purple dotted line marks the early-stop trigger after a patience window of 8 epochs.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/lstm/fig7_training_curves.png)
*Training and validation loss with early stopping — restore the weights at the green line, not the purple one.*

## 7. LSTM vs GRU — Which Should You Reach For?

| Aspect | LSTM | GRU |
| --- | --- | --- |
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Separate cell state $C_t$ | Yes | No |
| Parameters per cell | 4 weight matrices | 3 weight matrices |
| Speed | Baseline | ~25 % faster, comparable accuracy |
| Best for | Long sequences, very large datasets, when you need maximum capacity | Smaller datasets, real-time inference, mobile/edge |

Empirically, on most forecasting tasks the gap between LSTM and GRU is within noise. **Default to GRU** for fast iteration; switch to LSTM if you have plenty of data and very long-range dependencies (sequences of several hundred steps). For sequences past ~500 steps, both are usually beaten by a Temporal Convolutional Network (Part 6) or an Informer-style sparse Transformer (Part 8).

## 8. Common Pitfalls

- **Forgetting to standardize per feature.** LSTMs are scale-sensitive; raw stock prices and percent returns mixed in one input will train badly.
- **Shuffling time-series windows across the train/test boundary.** Use `TimeSeriesSplit` or a fixed chronological cut.
- **Reading the last hidden state as if it were the prediction.** It is just a feature vector — you still need a linear head, and you should standardize the target too.
- **Using a BiLSTM for forecasting.** It will look great in the notebook because the model is reading the future, and then collapse in production.
- **Tuning hidden size before the lookback.** Lookback decides what information the cell *can* see; making the cell wider doesn't help if the window is too short.
- **Trusting a single random seed.** RNN training is noisy. Report mean ± std over at least 3 seeds.

## Summary

LSTM solves the vanishing gradient problem by routing memory through an additive **cell-state highway** that the network controls with three multiplicative gates. The forget gate decides what to erase, the input gate decides what to write, and the output gate decides what to expose — and because the long-range gradient is a product of forget gates rather than a product of recurrent Jacobians, the model can learn dependencies hundreds of steps long.

For time series, that translates into a small set of practical recipes: window the series with a sensible lookback, stack 1–3 layers of moderate width, regularize with dropout and early stopping, choose direct multi-step over recursive when horizons are long, and keep BiLSTM for offline tasks only. The next part covers GRU — the slimmer cousin that achieves nearly the same with fewer parameters.

## References

- Hochreiter & Schmidhuber, *Long Short-Term Memory*, Neural Computation (1997)
- Gers, Schmidhuber & Cummins, *Learning to Forget: Continual Prediction with LSTM* (2000) — the paper that introduced the +1 forget-gate bias trick
- Olah, *Understanding LSTM Networks*, colah.github.io (2015) — the canonical illustrated explanation
- Greff et al., *LSTM: A Search Space Odyssey*, IEEE TNNLS (2017) — empirical study of LSTM variants
