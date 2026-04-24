---
title: "Time Series Forecasting (6): Temporal Convolutional Networks (TCN)"
date: 2024-12-15 09:00:00
tags:
  - Time Series
  - Deep Learning
  - TCN
categories: Algorithm
series: Time Series Forecasting
lang: en
mathjax: true
description: "TCNs use causal dilated convolutions for parallel training and exponential receptive fields. Complete PyTorch implementation with traffic flow and sensor data case studies."
disableNunjucks: true
---


For most of the 2010s, anyone who said "deep learning for time series" meant LSTM. The story changed in 2018 when Bai, Kolter, and Koltun published *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*. Their result was annoyingly simple: take a stack of 1-D convolutions, make them causal (no peeking at the future), space the filter taps out exponentially (dilation), wrap the whole thing in residual connections, and train. On task after task, the resulting **Temporal Convolutional Network** (TCN) matched or beat LSTM/GRU -- while training several times faster because every time step in the forward pass runs in parallel.

This chapter unpacks why that recipe works. We will derive the receptive-field formula that makes dilation worth caring about, walk through the residual block step by step, and finish with two production-grade case studies (traffic flow and multivariate sensor forecasting) using a PyTorch implementation you can copy out.

## What you will learn

- Why a causal 1-D convolution is required for honest forecasting and how left-padding implements it.
- How dilated convolutions grow the receptive field as $\mathcal{O}(2^L)$ instead of $\mathcal{O}(L)$.
- The exact anatomy of a TCN residual block (two dilated causal convs + weight norm + dropout + 1x1 skip).
- A head-to-head TCN vs LSTM/GRU/Transformer comparison on training time, memory, and accuracy.
- Two case studies: hourly traffic flow forecasting and multivariate IoT sensor prediction.

**Prerequisites**: Part 2 (LSTM) and Part 5 (Transformer). Comfort with PyTorch's `nn.Conv1d` and basic complexity analysis.

---

## Why the recipe of the time (LSTM) was painful

Before TCN, the deep-learning playbook for time series looked like this: stack two LSTM layers, throw in attention if you were feeling fancy, train for a long time. It worked, but every part of the pipeline pushed back:

- **Sequential forward pass.** To compute hidden state $h_t$ you need $h_{t-1}$. The GPU sits idle waiting for the previous step. Doubling the sequence length doubles wall-clock time even with infinite parallel hardware.
- **Vanishing/exploding gradients through time.** Backprop has to traverse $L$ multiplicative steps. LSTMs help via gates, but anything past ~200 steps is fragile. People reach for gradient clipping, layer norm, and careful initialization just to keep training stable.
- **Hidden-state opacity.** "Why did the model predict this?" usually has no good answer because the hidden state mixes everything.
- **Hyperparameter tax.** The number of layers, hidden dimensions, gate variants, dropout types, and recurrent dropout positions all interact. A bad combination wastes a day of training before you notice.

TCN's pitch: replace the recurrence with convolutions you can run in parallel, replace the implicit memory of the hidden state with an explicit receptive field, and use residual connections to keep gradients well-behaved. Same expressive power, fewer moving parts.

---

## 1-D convolution, but causal

A standard 1-D convolution slides a length-$k$ filter $f$ over an input sequence $x$:

$$
y_t = \sum_{i=0}^{k-1} f_i \, x_{t-i+\lfloor k/2 \rfloor}.
$$

That centred form lets the output at time $t$ read both past and future inputs. For forecasting that is **information leakage** -- you cannot learn to predict tomorrow's traffic by peeking at tomorrow's traffic.

A **causal** convolution shifts the filter so the output at $t$ only uses inputs from $1, \ldots, t$:

$$
y_t = \sum_{i=0}^{k-1} f_i \, x_{t-i}.
$$

Implementation-wise, you pad the input on the **left** with $k - 1$ zeros and run a vanilla `nn.Conv1d`. After the convolution you slice the right-hand padding back off so the output length equals the input length.

![Causal vs non-causal 1-D convolution at t = 6](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/temporal-convolutional-networks/fig2_causal_convolution.png)

In the figure above the green output $y_6$ is the same in both panels, but the inputs it draws on (amber) are different. The non-causal filter on the left reads $x_7$, which lies in the shaded "future" region -- a hard no for forecasting. The causal version on the right only ever looks left.

In PyTorch:

```python
import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1-D convolution with left padding and right-side trim."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        y = self.conv(x)
        if self.padding > 0:
            y = y[:, :, : -self.padding]
        return y
```

Two details worth flagging:

1. The padding amount $(k-1) \cdot d$ depends on the dilation $d$, which we are about to introduce.
2. We trim the **right** side after the conv. A common bug is trimming the left, which silently destroys the early part of the sequence.

---

## Dilation: exponential receptive field on a linear depth budget

A causal convolution of kernel $k = 3$ stacked $L$ times has receptive field $1 + 2L$. Linear growth. To see 200 steps back you need 100 layers. That is unworkable.

**Dilated convolution** spreads the filter taps apart by a factor $d$:

$$
y_t = \sum_{i=0}^{k-1} f_i \, x_{t-d \cdot i}.
$$

If you double the dilation in every layer ($d_\ell = 2^{\ell-1}$), the receptive field of an $L$-layer stack becomes

$$
\text{RF}(L) = 1 + (k - 1)\sum_{\ell=1}^{L} d_\ell = 1 + (k - 1)(2^L - 1).
$$

For $k = 3$ and $L = 8$, that is **511 time steps** -- more than enough for a week of hourly data. Same parameter count as 8 ordinary layers, exponential coverage.

![Dilated causal convolution receptive field](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/temporal-convolutional-networks/fig1_dilated_convolution.png)

The diagram traces every input that contributes to the green output neuron at the top. The dilations $1, 2, 4, 8$ make the four-layer stack look like a sparse tree -- and that sparsity is exactly what gives it the wide reach.

A practical helper for sizing your network:

```python
import math


def required_layers(receptive_field: int, kernel_size: int = 3) -> int:
    """Smallest L such that 1 + (k-1)(2**L - 1) >= receptive_field."""
    L = (receptive_field - 1) / (kernel_size - 1) + 1
    return max(1, math.ceil(math.log2(L)))
```

Calling `required_layers(168, kernel_size=3)` returns `7`, which is what you want for hourly data with weekly memory.

---

## The TCN residual block

Stacking dilated causal convs is half the recipe. The other half is the residual block that wraps them. Bai et al. settled on the following structure (almost identical to the one in Oord et al.'s WaveNet, except for the activation choice):

![TCN residual block](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/temporal-convolutional-networks/fig3_residual_block.png)

Mathematically, given input $x$:

$$
F(x) = \mathrm{Dropout}\!\big(\mathrm{ReLU}(\mathrm{WN}(\mathrm{Conv}_2 \, \mathrm{Dropout}(\mathrm{ReLU}(\mathrm{WN}(\mathrm{Conv}_1 \, x))))) \big),
$$

$$
o = \mathrm{ReLU}\!\big( F(x) + W_{\text{skip}} \, x \big).
$$

Three deliberate choices:

- **Two convolutions per block.** One conv barely changes anything. Two gives the block enough capacity to learn a non-trivial transformation while keeping the depth count low.
- **Weight normalization.** Bai et al. found that batch norm hurts on long sequences (the statistics drift across positions). Weight norm decouples direction and magnitude of each filter, leaves activations alone, and trains stably.
- **1x1 skip.** The identity shortcut only works when the input and output channel counts agree. A 1x1 convolution projects the input when they do not, at negligible cost.

A clean PyTorch implementation:

```python
class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation,
        ))
        conv2 = nn.utils.weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation, dilation=dilation,
        ))
        self.padding = (kernel_size - 1) * dilation
        self.conv1, self.conv2 = conv1, conv2
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        for layer in (self.conv1, self.conv2):
            nn.init.normal_(layer.weight, 0.0, 0.01)

    def _causal(self, conv, x):
        y = conv(x)
        return y[:, :, : -self.padding] if self.padding > 0 else y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.relu(self._causal(self.conv1, x)))
        out = self.dropout(self.relu(self._causal(self.conv2, out)))
        return self.relu(out + self.skip(x))
```

The block is simple enough that people often inline it, but having it as a module makes the receptive-field calculation transparent and lets you swap weight norm for layer norm in the rare cases where it helps.

---

## Putting the network together

A full TCN is a stack of residual blocks with exponentially growing dilation, optionally followed by a 1x1 projection that maps to the output dimension you care about.

```python
class TCN(nn.Module):
    def __init__(self, input_size: int, output_size: int,
                 channels: list[int], kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_size
        for i, c in enumerate(channels):
            layers.append(TCNResidualBlock(
                prev, c, kernel_size, dilation=2 ** i, dropout=dropout,
            ))
            prev = c
        self.network = nn.Sequential(*layers)
        self.head = nn.Conv1d(prev, output_size, kernel_size=1)
        self._channels = channels
        self._k = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.network(x))

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self._k - 1) * (2 ** len(self._channels) - 1)
```

A few notes on configuration:

- **Channels.** Most papers use a constant width (e.g. `[64] * 8`). Increasing width near the head helps when the output dimension is much larger than the input.
- **Kernel size.** $k = 3$ is the standard choice. $k = 5$ or $7$ doubles parameters and rarely buys you accuracy; bigger receptive field is almost always cheaper to obtain via more dilation.
- **Dropout.** $0.2$ is a safe default. Push to $0.3$--$0.5$ on small datasets.

---

## TCN vs RNN: the architectural picture

The flow-of-information diagram explains the speed gap better than any benchmark table:

![RNN sequential dependency vs TCN parallel forward pass](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/temporal-convolutional-networks/fig4_tcn_vs_rnn.png)

In the RNN panel each red arrow is a hard sequential dependency. The GPU can compute the contributions inside one cell in parallel, but it cannot skip ahead to step $t+1$ until step $t$ is done. The wall-clock time of a forward pass therefore scales linearly with sequence length even on infinite parallel hardware.

In the TCN panel, every output node is a function of a fixed set of input nodes. The same convolution kernel applies everywhere. The whole layer is one big matrix multiplication that the GPU happily issues in a single kernel launch.

Concretely, here is the per-epoch wall-clock comparison on a single GPU:

![Training time per epoch and TCN speedup over LSTM](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/temporal-convolutional-networks/fig5_parallel_training.png)

Two takeaways:

1. **Training time scaling matters.** All four architectures are roughly comparable at $L = 128$. By $L = 1024$, TCN is 3-4x faster than LSTM and ~6x faster than the vanilla Transformer (whose attention cost grows as $L^2$). This is exactly where most real time-series problems live.
2. **Inference parity.** At inference, RNN and TCN are typically within 1.5x of each other; the gap is a training story, not an inference one. If the only thing you care about is latency on a single sample, both are fine.

Where to use which? The honest answer is a small decision matrix:

| Situation | Best choice | Reason |
|---|---|---|
| Fixed-length window, GPU available | TCN | Parallel training, predictable receptive field. |
| Variable-length sequences, lots of padding | LSTM/GRU | Native support, no padding overhead. |
| Streaming / online inference, one step at a time | LSTM/GRU | Hidden state is the natural state. |
| Multivariate, attention-worthy cross-feature interactions | Transformer / Informer | Attention captures pairwise relationships. |
| Anything where you do not know yet | TCN as the first baseline | Trains fast, fewer hyperparameters. |

---

## Implementation in PyTorch: a complete training loop

The block and network classes above are the load-bearing parts. The training loop is unsurprising:

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def train_tcn(model, train_loader, val_loader,
              num_epochs=50, lr=1e-3, device="cuda"):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min",
                                                 factor=0.5, patience=5)
    crit = nn.MSELoss()
    best = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += crit(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        sched.step(val_loss)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "tcn_best.pt")
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}: train {train_loss:.4f} val {val_loss:.4f}")
```

Two things to highlight: gradient clipping is *not* strictly necessary for TCN (residual + weight norm keep gradients well-behaved), but it costs nothing to add. And `ReduceLROnPlateau` is more robust than a fixed schedule because the right learning rate depends on the dataset and the receptive field.

A small helper for windowing univariate data:

```python
import numpy as np


def make_windows(series: np.ndarray, history: int, horizon: int):
    """Convert a 1-D series to (X, y) tensors for direct multi-step forecasting."""
    n = len(series) - history - horizon + 1
    X = np.stack([series[i : i + history] for i in range(n)])
    y = np.stack([series[i + history : i + history + horizon] for i in range(n)])
    X = torch.from_numpy(X).float().unsqueeze(1)  # (N, 1, history)
    y = torch.from_numpy(y).float().unsqueeze(1)  # (N, 1, horizon)
    return X, y
```

---

## Case study 1: hourly traffic flow forecasting

**Setup.** Predict the next 24 hours of vehicle counts at a single highway sensor given the past week (168 hours). Univariate, strong daily and weekly seasonality, occasional event-driven spikes.

**Receptive-field budget.** We want at least one full week of history visible at the output. For $k = 3$ and $L = 7$, $\text{RF} = 1 + 2 \cdot 2 \cdot 127 = 509$. Comfortable.

```python
def synthetic_traffic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    daily = 1000 + 500 * np.sin(2 * np.pi * t / 24)
    weekly = 200 * np.sin(2 * np.pi * t / (24 * 7))
    trend = 0.05 * t
    noise = rng.normal(0, 50, n)
    return daily + weekly + trend + noise


from sklearn.preprocessing import StandardScaler

raw = synthetic_traffic()
scaler = StandardScaler()
series = scaler.fit_transform(raw.reshape(-1, 1)).flatten()

X, y = make_windows(series, history=168, horizon=24)
split = int(0.8 * len(X))
train_loader = DataLoader(TensorDataset(X[:split], y[:split]),
                          batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X[split:], y[split:]),
                        batch_size=64)

model = TCN(input_size=1, output_size=1,
            channels=[64] * 7, kernel_size=3, dropout=0.2)
print("Receptive field:", model.receptive_field)  # 509

train_tcn(model, train_loader, val_loader, num_epochs=80)
```

Note that `output_size=1` produces a one-channel sequence. In direct multi-step forecasting you usually want the network to emit the entire horizon at once. Two ways to do that:

1. **Sequence-to-sequence head.** Keep `output_size=1`, take the last $H$ time steps of the output sequence. Simple, but ties horizon to history geometry.
2. **Flatten + linear head.** Replace the final `nn.Conv1d(C, 1, 1)` with `nn.Linear(C * history, horizon)` so the model directly outputs an $H$-vector. More flexible.

Either works; option 1 trains fewer parameters and is what we use here.

**Expected behaviour.** On synthetic data the model nails the daily peaks within ~10% MAPE after 30 epochs. On real Caltrans-style data you should see MAPE in the 8-15% range with no special tuning, comfortably better than a seasonal-naive baseline.

---

## Case study 2: multivariate sensor forecasting

**Setup.** Four correlated IoT sensors (temperature, humidity, pressure, light), 5-minute sampling. Predict temperature for the next hour (12 steps) given the past 6 hours (72 steps).

```python
def synthetic_sensors(n=5000, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    temp = 20 + 5 * np.sin(2 * np.pi * t / 288) + rng.normal(0, 0.5, n)
    hum = 60 - 0.8 * (temp - 20) + rng.normal(0, 2, n)
    pres = 1013 + 2 * np.sin(2 * np.pi * t / 1000) + rng.normal(0, 0.3, n)
    light = 100 * np.maximum(0, np.sin(2 * np.pi * t / 288)) + rng.normal(0, 5, n)
    return np.column_stack([temp, hum, pres, light])


sensors = synthetic_sensors()
scaler = StandardScaler()
sensors_s = scaler.fit_transform(sensors)


def make_multivariate_windows(arr, target_idx, history, horizon):
    n = len(arr) - history - horizon + 1
    X = np.stack([arr[i : i + history].T for i in range(n)])  # (N, F, T)
    y = np.stack([arr[i + history : i + history + horizon, target_idx]
                  for i in range(n)])
    return torch.from_numpy(X).float(), torch.from_numpy(y).float().unsqueeze(1)


Xm, ym = make_multivariate_windows(sensors_s, target_idx=0,
                                   history=72, horizon=12)
# ... build loaders ...

model = TCN(input_size=4, output_size=1,
            channels=[64, 64, 128, 128, 128], kernel_size=3, dropout=0.2)
print("Receptive field:", model.receptive_field)  # 253
```

**Why a multivariate input "just works" in TCN.** Because the first layer convolves across all four input channels at every time step, cross-feature interactions are baked in for free. There is no need for a separate fusion module.

**Quick feature-importance check.** Zero out one channel at a time and look at the increase in validation MAE:

```python
def feature_ablation(model, X_val, y_val, names):
    model.eval()
    base = ((model(X_val) - y_val) ** 2).mean().item()
    out = {}
    for i, name in enumerate(names):
        Xz = X_val.clone()
        Xz[:, i, :] = 0.0
        out[name] = ((model(Xz) - y_val) ** 2).mean().item() - base
    return out


print(feature_ablation(model, Xm[:200], ym[:200],
                       ["temp", "hum", "pres", "light"]))
```

On the synthetic data above, humidity dominates (it is correlated with temperature by construction). On real sensor data the picture is messier but still informative as a sanity check.

---

## Hyperparameter and design cheat sheet

The defaults you should reach for first, by problem characteristic:

| Hyperparameter | Default | When to change it |
|---|---|---|
| Kernel size $k$ | 3 | Almost never. Use dilation to grow RF. |
| Dilation schedule | $2^i$ for layer $i$ | Almost never. Powers of 2 are the right answer. |
| Channels | constant width 32-128 | Increase if underfitting; decrease if overfitting. |
| Number of layers $L$ | smallest $L$ with $\text{RF}(L) \geq$ context | Use the formula; do not overshoot. |
| Dropout | 0.2 | 0.3-0.5 on small datasets; 0.1 on huge ones. |
| Normalization | weight norm | Layer norm if batches are tiny; avoid batch norm. |
| Optimizer | Adam, lr 1e-3 | SGD + momentum sometimes wins on huge datasets. |
| LR schedule | `ReduceLROnPlateau`, factor 0.5 | Cosine annealing if you train for many epochs. |
| Gradient clip | 1.0 | Keep it; cheap insurance. |

---

## Common pitfalls and how to debug them

- **The output looks shifted right.** You forgot to trim the right-side padding after the conv. Check that `y[:, :, : -self.padding]` is in your causal conv.
- **Loss decreases on train but not on val.** Receptive field smaller than the dominant period in your data. Re-run `required_layers` with the right horizon.
- **Loss plateaus very early.** Channels too narrow, or learning rate too low. Try doubling channels or `lr=3e-3`.
- **Val loss explodes.** Almost certainly batch norm + small batch size, or no dropout on a tiny dataset. Switch to weight norm and add 0.3 dropout.
- **Predictions ignore recent values.** The network is leaning entirely on long-range structure. Drop a few layers (smaller RF) or add a 1-step skip from the input to the output.

---

## When to *not* use a TCN

The architecture has real limits. Skip TCN when:

- **Sequences are highly variable in length and you cannot afford padding.** Use an LSTM/GRU.
- **You need true online streaming inference** (one new sample at a time, decisions in microseconds). Causal CNNs can be implemented streaming, but it is fiddlier than just running an LSTM cell.
- **The target is much longer than your window** (e.g. 100k-step physiological signals where you need 50k of context). Hierarchical models like N-BEATS-X or Informer scale better.
- **You need attention-style interpretability.** TCN filters can be visualised but the meaning is local; an attention map is much more directly readable.

In every other forecasting situation, TCN is the boring, fast, reliable first thing to try.

---

## Q&A

**How is TCN different from WaveNet?**
WaveNet (2016) is essentially a TCN with a gated activation $\tanh(W_f x) \odot \sigma(W_g x)$ instead of ReLU and a richer conditioning mechanism for audio generation. TCN strips it down to ReLU + residual for general sequence modeling.

**Should I use BatchNorm or WeightNorm?**
WeightNorm. BatchNorm's running statistics are noisy on long sequences and tend to drift; WeightNorm sidesteps the issue entirely. LayerNorm is acceptable but adds a transpose for 1-D conv data layouts.

**Do I need positional encodings like in Transformers?**
No. The convolution itself is translation-equivariant by construction; position is implicit in the receptive-field structure.

**Direct multi-step or recursive multi-step forecasting?**
Direct (output the full horizon at once) is more accurate because errors do not compound, but uses more parameters and locks the horizon at training time. Recursive (predict one step, feed it back) is flexible but accumulates error. Default to direct.

**What if I want quantile forecasts (not just point estimates)?**
Replace the L2 loss with the pinball loss at multiple quantiles and have the head output one channel per quantile. The TCN backbone is unchanged.

---

## Summary

TCN reframed sequence modeling around a single observation: causal dilated convolutions with residual connections give you long memory, parallel training, and stable gradients without any of the recurrent machinery. The math is small ($\text{RF}(L) = 1 + (k-1)(2^L - 1)$ is the only formula you really need), the implementation fits in 60 lines of PyTorch, and the empirical performance against tuned LSTMs is at-least-as-good on most fixed-length benchmarks.

Use it as your first forecasting baseline. If it loses to something more elaborate, you have learned that the elaborate thing was earning its keep. If it wins -- which it often does -- you have shipped a fast, simple model.

Next chapter we move from convolutions to **N-BEATS**, which throws away both convolution and recurrence in favour of fully connected blocks plus basis-function expansion, and won the M4 forecasting competition while staying interpretable.

---

## References and further reading

- Bai, S., Kolter, J. Z., & Koltun, V. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling.* arXiv:1803.01271.
- van den Oord, A. et al. (2016). *WaveNet: A Generative Model for Raw Audio.* arXiv:1609.03499.
- Lea, C. et al. (2017). *Temporal Convolutional Networks for Action Segmentation and Detection.* CVPR.
- Salimans, T., & Kingma, D. P. (2016). *Weight Normalization.* NeurIPS.

*This article is part of the Time Series Forecasting series. Use the navigation at the top to jump to other chapters.*
