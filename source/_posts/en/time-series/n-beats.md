---
title: "Time Series Forecasting (7): N-BEATS -- Interpretable Deep Architecture"
date: 2024-11-07 09:00:00
tags:
  - Time Series
  - Deep Learning
  - N-BEATS
categories: Algorithm
series: Time Series Forecasting
lang: en
mathjax: true
description: "N-BEATS combines deep learning expressiveness with classical decomposition interpretability. Basis function expansion, double residual stacking, and M4 competition analysis with full PyTorch code."
---

> **Series**: Time Series Forecasting -- Part 7 of 8
> [<-- Previous: TCN](/en/time-series-temporal-convolutional-networks/) | [Next: Informer -->](/en/time-series-informer-long-sequence/)

The 2018 M4 forecasting competition served 100,000 series across six frequencies as a single benchmark. The leaderboard was dominated by hand-tuned ensembles built from decades of statistical-forecasting craft. Then a **pure neural network** with no statistical preprocessing, no feature engineering, and no recurrence won outright. That network was **N-BEATS** by Oreshkin et al. -- a stack of fully-connected blocks with two residual paths. Its interpretable variant additionally split the forecast into a polynomial trend and a Fourier seasonality, so the very thing classical statisticians wanted (a readable decomposition) came for free.

This chapter unpacks why such a stripped-down architecture beats both LSTMs and ARIMA-style ensembles, and how to implement and tune it for your own series.

## What you will learn

- How double residual stacking turns a plain MLP into a hierarchical decomposer.
- Basis-function expansion: polynomial bases for trend, Fourier bases for seasonality, learned bases for the "generic" variant.
- Why N-BEATS can be at once the most accurate and the most interpretable model in the room.
- The M4 result table: what N-BEATS actually beat and by how much.
- A complete PyTorch implementation, plus retail-sales and energy-demand case studies you can adapt.

**Prerequisites**: Comfortable with feed-forward networks and PyTorch. Familiarity with classical decomposition (trend / seasonality / residual) helps but is not required.

---

## Why a fully-connected stack is enough

Most deep models for time series add structural priors: convolution assumes translation equivariance, RNN assumes a sequential hidden state, attention assumes pairwise relevance. N-BEATS goes the other way: it dumps the whole input window into an MLP and lets the network learn whatever decomposition is most useful. The smart part is the *path* the information takes, not the layer type.

Concretely, N-BEATS makes three opinionated choices:

1. **Stack of identical blocks**, each block predicting both a *backcast* (a reconstruction of the input window) and a *forecast* (a prediction of the future).
2. **Double residual streams**: every block subtracts its backcast from the running input residual *and* adds its forecast to a running forecast accumulator. The next block sees only what the previous one could not explain.
3. **Basis-function output heads.** Instead of producing the forecast values directly, the block produces a small vector of coefficients $\theta$ that it then multiplies into a fixed (interpretable) or learned (generic) basis matrix.

The combination of these three turns out to be enough to win the M4 leaderboard.

---

## Architecture: the double residual stream

Picture two parallel pipes running top-down through the network. The left pipe is the **residual stream**: it starts with the input window $x \in \mathbb{R}^{H}$ and shrinks each time a block subtracts its backcast. The right pipe is the **forecast accumulator**: it starts at zero and grows each time a block adds its forecast.

![Double residual stacking in N-BEATS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/n-beats/fig1_stacked_residual_blocks.png)

Mathematically, for blocks $b = 1, \ldots, B$:

$$
r^{(b)} = r^{(b-1)} - \hat{x}^{(b)}, \qquad \hat{y} = \sum_{b=1}^{B} \hat{y}^{(b)},
$$

with $r^{(0)} = x$. Each block sees a smaller and smaller residual, so it ends up specialising on whatever frequency or shape was left over. Coarse patterns (overall trend, dominant seasonal cycle) get picked up by early blocks; fine corrections by later blocks.

This is the same idea as gradient boosting, but inside a single end-to-end differentiable network. And just like in boosting, ordering matters: the block that runs first has the easiest job (full signal available) and the block that runs last gets the hardest job (only noise plus subtle structure left).

---

## Inside a block

Every N-BEATS block is the same shape. Given a residual input $r \in \mathbb{R}^{H}$:

1. **Feature extractor** -- four fully-connected ReLU layers of width 256-512:
   $$
   h_1 = \mathrm{ReLU}(W_1 r + b_1), \quad \ldots, \quad h_4 = \mathrm{ReLU}(W_4 h_3 + b_4).
   $$
2. **Coefficient projections** -- two linear heads produce backcast and forecast coefficients:
   $$
   \theta^{b} = W_b h_4, \qquad \theta^{f} = W_f h_4.
   $$
3. **Basis multiplication** -- a fixed or learned matrix $V$ maps coefficients to time-domain outputs:
   $$
   \hat{x} = V_b \, \theta^{b}, \qquad \hat{y} = V_f \, \theta^{f}.
   $$

Two flavours exist, distinguished only by what $V$ is.

### Interpretable: trend + seasonality bases

The trend block uses a low-degree polynomial basis. With degree $p$ and time index $\tau / H \in [0, 1]$:

$$
V_{\text{trend}} = \begin{pmatrix} 1 & \tau & \tau^{2} & \cdots & \tau^{p} \end{pmatrix}, \qquad
\hat{y}_{\text{trend}} = \sum_{i=0}^{p} \theta_i \, \tau^{i}.
$$

Typical choice: $p = 2$ or $3$. That is enough to model "smoothly increasing then accelerating" without overfitting wiggles.

The seasonality block uses Fourier bases:

$$
V_{\text{seas}} = \begin{pmatrix} \sin(2\pi \cdot 1 \cdot \tau / T) & \cos(2\pi \cdot 1 \cdot \tau / T) & \cdots & \sin(2\pi K \tau / T) & \cos(2\pi K \tau / T) \end{pmatrix}.
$$

With $K = 1, 2, 3$ harmonics and $T = $ the data's known period (12 for monthly, 24 for hourly), this captures arbitrarily-shaped periodic signals.

The interpretable architecture stacks one trend stack (a few trend blocks) followed by one seasonality stack (a few seasonal blocks). After training you can plot each stack's contribution and explain to a stakeholder which part of the forecast came from "the underlying trend" versus "the recurring weekly cycle."

![Trend + seasonality decomposition produced by an interpretable N-BEATS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/n-beats/fig2_basis_decomposition.png)

### Generic: learned bases

The generic variant lets $V_b$ and $V_f$ be learnable matrices. The block is no longer forced into trend/seasonality; it discovers whatever bases the gradient signal pushes it toward. You get a small accuracy bump in exchange for losing the readable decomposition.

![Interpretable vs generic N-BEATS stack layouts](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/n-beats/fig3_interpretable_vs_generic.png)

A practical rule from the paper: the **best** result on M4 came from an *ensemble* of interpretable and generic models with different lookback lengths. We will return to this in the "ensemble strategy" section.

---

## Why basis-function output heads matter

The block could just predict the forecast values directly: $\hat{y} = W_y h_4$. So why route through a $\theta$ vector and a basis matrix?

Three reasons:

- **Inductive bias.** Forcing the block to express its forecast as a small linear combination of smooth bases prevents it from fitting noise. With a 720-step output and a polynomial of degree 3, the block has only 4 degrees of freedom for the trend component. It physically cannot oscillate. That regularisation is what lets the interpretable variant generalise.
- **Interpretability for free.** The trend stack's $\theta_0, \theta_1, \theta_2, \theta_3$ correspond directly to baseline, slope, curvature, and jerk. The seasonality stack's coefficients are amplitudes of specific harmonics. You can plot them and reason about them.
- **Parameter efficiency.** A direct head from a 512-dim hidden state to a 720-step forecast is a $512 \times 720 = 369K$-parameter linear layer per block. The basis-function head is two small linear layers ($512 \to p$, then a fixed $p \to 720$), often well under 10K parameters per output head.

The same idea is now common in newer models (PatchTST, N-HiTS, TSMixer all use some form of decomposition head); N-BEATS popularised it.

---

## PyTorch implementation

Below is a clean, complete implementation. About 120 lines for the full model.

```python
import torch
import torch.nn as nn
import numpy as np


class TrendBasis(nn.Module):
    """Polynomial basis: V[i, t] = (t / horizon) ** i."""

    def __init__(self, degree: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.degree = degree
        tb = torch.stack([torch.linspace(0, 1, backcast_size) ** i
                          for i in range(degree + 1)], dim=0)
        tf = torch.stack([torch.linspace(0, 1, forecast_size) ** i
                          for i in range(degree + 1)], dim=0)
        self.register_buffer("V_b", tb)  # (degree+1, H)
        self.register_buffer("V_f", tf)  # (degree+1, F)

    @property
    def theta_size(self) -> int:
        return self.degree + 1

    def forward(self, theta_b, theta_f):
        return theta_b @ self.V_b, theta_f @ self.V_f


class SeasonalityBasis(nn.Module):
    """Fourier basis with the first floor((H or F) / 2) harmonics."""

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        K = forecast_size // 2 + 1
        # backcast time indices in [0, H/F]
        tb = torch.linspace(0, 1, backcast_size)
        tf = torch.linspace(0, 1, forecast_size)
        ks = torch.arange(K).unsqueeze(1).float()
        Vb = torch.cat([torch.cos(2 * np.pi * ks * tb),
                        torch.sin(2 * np.pi * ks * tb)], dim=0)
        Vf = torch.cat([torch.cos(2 * np.pi * ks * tf),
                        torch.sin(2 * np.pi * ks * tf)], dim=0)
        self.register_buffer("V_b", Vb)  # (2K, H)
        self.register_buffer("V_f", Vf)  # (2K, F)

    @property
    def theta_size(self) -> int:
        return self.V_b.shape[0]

    def forward(self, theta_b, theta_f):
        return theta_b @ self.V_b, theta_f @ self.V_f


class GenericBasis(nn.Module):
    """Learned basis: backcast/forecast outputs are linear in theta."""

    def __init__(self, theta_size: int, backcast_size: int,
                 forecast_size: int):
        super().__init__()
        self._theta_size = theta_size
        self.linear_b = nn.Linear(theta_size, backcast_size, bias=False)
        self.linear_f = nn.Linear(theta_size, forecast_size, bias=False)

    @property
    def theta_size(self) -> int:
        return self._theta_size

    def forward(self, theta_b, theta_f):
        return self.linear_b(theta_b), self.linear_f(theta_f)


class NBeatsBlock(nn.Module):
    def __init__(self, basis: nn.Module, backcast_size: int,
                 hidden: int = 256, layers: int = 4):
        super().__init__()
        self.basis = basis
        units = [backcast_size] + [hidden] * layers
        fcs = []
        for in_dim, out_dim in zip(units[:-1], units[1:]):
            fcs.append(nn.Linear(in_dim, out_dim))
            fcs.append(nn.ReLU())
        self.fc = nn.Sequential(*fcs)
        self.head_b = nn.Linear(hidden, basis.theta_size)
        self.head_f = nn.Linear(hidden, basis.theta_size)

    def forward(self, x):
        h = self.fc(x)
        theta_b = self.head_b(h)
        theta_f = self.head_f(h)
        return self.basis(theta_b, theta_f)  # (backcast, forecast)


class NBeats(nn.Module):
    def __init__(self, blocks: list[nn.Module]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        residual = x
        forecast = 0.0
        for blk in self.blocks:
            backcast, fc = blk(residual)
            residual = residual - backcast
            forecast = forecast + fc
        return forecast


def make_interpretable(history: int, horizon: int,
                       trend_blocks: int = 3, seasonal_blocks: int = 3,
                       trend_degree: int = 3,
                       hidden: int = 256, layers: int = 4) -> NBeats:
    blocks = []
    # Trend stack
    trend_basis = TrendBasis(trend_degree, history, horizon)
    for _ in range(trend_blocks):
        blocks.append(NBeatsBlock(trend_basis, history, hidden, layers))
    # Seasonality stack
    seas_basis = SeasonalityBasis(history, horizon)
    for _ in range(seasonal_blocks):
        blocks.append(NBeatsBlock(seas_basis, history, hidden, layers))
    return NBeats(blocks)


def make_generic(history: int, horizon: int,
                 num_blocks: int = 30, theta_size: int = 32,
                 hidden: int = 512, layers: int = 4) -> NBeats:
    blocks = []
    for _ in range(num_blocks):
        basis = GenericBasis(theta_size, history, horizon)
        blocks.append(NBeatsBlock(basis, history, hidden, layers))
    return NBeats(blocks)
```

A small but important detail: in the interpretable variant, the `TrendBasis` and `SeasonalityBasis` instances are **shared across blocks within a stack**. Each block has its own MLP and its own coefficient heads, but they all multiply by the same fixed basis matrix, which keeps the inductive bias intact and saves a few parameters.

---

## Training recipe

The Oreshkin et al. recipe is unfussy:

```python
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_nbeats(model, train_loader, val_loader, epochs=100,
                 lr=1e-3, device="cuda"):
    model = model.to(device)
    opt = Adam(model.parameters(), lr=lr)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.L1Loss()  # MAE; the paper actually uses sMAPE/MASE/MAPE
                        # depending on the data, but L1 is a safe default
    best = float("inf")
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        sched.step()
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += crit(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "nbeats.pt")
        if (ep + 1) % 10 == 0:
            print(f"epoch {ep+1}: train {train_loss:.4f} val {val_loss:.4f}")
```

A few practical notes:

- **Loss function.** For M4 the paper uses sMAPE; for many real datasets MAE (L1) is more stable than MSE because it does not over-penalise outliers. Pick the loss that matches your evaluation metric.
- **Per-window normalisation.** Subtract the mean and divide by the standard deviation of *each input window* before feeding the network, then inverse-transform the forecast. This is far more important than the choice of loss; without it the network is effectively asked to learn the scale of every series in addition to the shape.
- **Early stopping.** N-BEATS likes long training runs but flatlines after ~50 epochs on most datasets. Track val loss and stop when it stops improving.

---

## What N-BEATS won at M4

The M4 competition included statistical methods (ARIMA, ETS, Theta), the M4 winner Smyl's ES-RNN hybrid, and FFORMA, the second-place model based on feature-based meta-learning. N-BEATS, with no statistical preprocessing, beat everyone on overall sMAPE and on five of six frequency buckets.

![N-BEATS performance on the M4 competition](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/n-beats/fig4_m4_results.png)

The numbers in the paper:

- **N-BEATS (interpretable + generic ensemble): sMAPE 11.135**
- N-BEATS (generic only): sMAPE 11.168
- Smyl's ES-RNN (M4 winner): sMAPE 11.374
- FFORMA: sMAPE 11.720
- Best classical (Theta): sMAPE 12.309

The accuracy gap is in absolute sMAPE points -- modest but consistent across yearly, quarterly, monthly, weekly, and daily series. The hourly bucket is the only one where Smyl's ES-RNN edged out, by 0.4 sMAPE.

The deeper lesson: a sufficiently expressive deep model with the right inductive bias can recover from-scratch what statisticians had spent decades hand-engineering.

---

## Ensembling: the unspoken half of the recipe

Read the M4 paper carefully and you will find a footnote: the headline N-BEATS number is the **median forecast across 180 model instances**. Each instance differs in one of three things: lookback length (2H, 3H, ..., 7H), training loss (sMAPE vs MASE vs MAPE), and random seed. Single-model performance is noticeably worse than the ensemble.

![Why N-BEATS ensembles: schematic and empirical sMAPE drop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/n-beats/fig5_ensemble_strategy.png)

The empirical curve on the right shows diminishing returns: most of the gain comes from the first ~30 ensemble members. In production you almost never need to train 180 models -- 10 to 30, with mixed lookbacks and seeds, captures the bulk of the improvement.

A simple ensembling helper:

```python
class EnsembleNBeats:
    def __init__(self, models: list[NBeats]):
        self.models = models

    def predict(self, x: torch.Tensor, aggregator="median") -> torch.Tensor:
        outs = torch.stack([m(x) for m in self.models], dim=0)
        return outs.median(dim=0).values if aggregator == "median" else outs.mean(0)
```

Median is a better aggregator than mean for sMAPE-style losses because it is robust to one model going crazy on a single window.

---

## Case study 1: monthly retail sales

**Setup.** Forecast the next 12 months of unit sales for a multi-store retail chain given the past 36 months. Strong holiday seasonality (December peak), upward trend with occasional promotions, ~200 individual product-store series.

**Architecture choice.** Interpretable. The business team needs to be able to point at the monthly forecast and say "$X$ comes from the underlying trend, $Y$ from the recurring December lift, $Z$ from the residual." The interpretable variant with `[trend stack of 3] + [seasonality stack of 3]` does this directly.

```python
model = make_interpretable(
    history=36, horizon=12,
    trend_blocks=3, seasonal_blocks=3,
    trend_degree=2,                     # smooth multi-year trend
    hidden=256, layers=4,
)
```

After training, we can pull the per-stack contribution to inspect what each part learned:

```python
def stack_contributions(model: NBeats, x: torch.Tensor) -> dict:
    """Return per-stack forecast contributions for a single window."""
    residual = x.clone()
    out = {}
    cur_stack = "trend"
    cumulative = torch.zeros(x.size(0), model.blocks[0].basis.V_f.shape[1])
    for i, blk in enumerate(model.blocks):
        backcast, fc = blk(residual)
        residual = residual - backcast
        cumulative = cumulative + fc
        # detect stack boundary by inspecting the basis class
        next_stack = "seasonality" if isinstance(blk.basis, SeasonalityBasis) else "trend"
        if i + 1 == len(model.blocks) or not isinstance(model.blocks[i + 1].basis, type(blk.basis)):
            out[cur_stack] = cumulative.clone()
            cumulative = torch.zeros_like(cumulative)
            cur_stack = next_stack
    return out


contribs = stack_contributions(model, x_val[:1])
# contribs["trend"][0]      -> 12-month trend component
# contribs["seasonality"][0] -> 12-month seasonal component
```

**Typical numbers.** On a real retail series with 5 years of history, an interpretable N-BEATS reaches MAPE of 7-12% on the 12-month horizon, comparable to the best gradient-boosted feature-engineering pipelines but without the feature-engineering work. The interpretability is the genuine win: a model that lets the business team override the December seasonality with a known promotional event is much more useful than a black box that scores 0.5% better on backtest.

---

## Case study 2: hourly electricity demand

**Setup.** Forecast the next 24 hours of grid demand given the past 168 hours (one week). Strong daily and weekly cycles, weather sensitivity, demand spikes during heatwaves.

**Architecture choice.** Generic. The patterns are complex (multi-resolution daily + weekly + weather-driven) and the business team primarily cares about accuracy: every MW of forecast error costs operating reserve. Switch to a deeper generic stack with a wider hidden state.

```python
model = make_generic(
    history=168, horizon=24,
    num_blocks=30, theta_size=32,
    hidden=512, layers=4,
)
```

**Per-window normalisation matters most here.** Demand levels swing across seasons; without per-window standardisation the model wastes capacity learning that "winter is bigger than summer". With it, the network only has to learn shape.

```python
def normalise_window(x: torch.Tensor) -> tuple:
    mu = x.mean(dim=-1, keepdim=True)
    sd = x.std(dim=-1, keepdim=True) + 1e-6
    return (x - mu) / sd, mu, sd

# Forward:
x_norm, mu, sd = normalise_window(x)
y_norm = model(x_norm)
y_hat = y_norm * sd + mu  # broadcast to (B, H)
```

**Typical numbers.** On the public ETT (Electricity Transformer Temperature) dataset with hourly demand, an ensemble of 10 generic N-BEATS models reaches MSE around 0.31 on the 24-hour horizon, comfortably ahead of LSTM (~0.42) and competitive with Informer at much lower implementation cost.

---

## Hyperparameter cheat sheet

A pragmatic starting point for a new dataset:

| Hyperparameter | Default (interpretable) | Default (generic) | Tuning hints |
|---|---|---|---|
| Lookback `history` | 4 to 7 times horizon | 4 to 7 times horizon | Larger covers more seasons; tune in 2x steps. |
| Hidden width | 256 | 512 | Bigger if validation loss plateaus high. |
| MLP layers per block | 4 | 4 | Rarely need to change. |
| Trend degree | 2 or 3 | -- | 3 if you see curvature in the trend; 2 otherwise. |
| Trend blocks | 3 | -- | Add more if trend RMSE is the dominant error. |
| Seasonality blocks | 3 | -- | Add more for multi-resolution seasonality. |
| Generic blocks | -- | 20 to 30 | More blocks usually still help, slowly. |
| Generic $\theta$ size | -- | 32 | 16 underfits short bases, 64+ rarely buys anything. |
| Loss | MAE / sMAPE | MAE / sMAPE | Match your evaluation metric. |
| Optimizer | Adam, lr 1e-3 | Adam, lr 1e-3 | Cosine annealing; warmup is unnecessary. |
| Ensemble size | 10-30 | 10-30 | Median aggregator. |

---

## When *not* to use N-BEATS

- **Very long horizons (>1000 steps).** The output projection $\theta \to \hat{y}$ becomes the parameter bottleneck. Use **N-HiTS** (the N-BEATS successor) or PatchTST, both of which use multi-rate downsampling to scale.
- **Multivariate time series with strong cross-feature interactions.** N-BEATS is univariate by design. Train one model per variable, or look at TFT / Informer / DeepAR.
- **Very short series (~50 observations).** The MLP backbone has tens of thousands of parameters; a Theta or ETS model will beat it on tiny data.
- **Online streaming forecasting.** N-BEATS processes a fixed window, not a streaming hidden state. An LSTM or TCN is more natural here.
- **Forecasts must respect hard constraints** (non-negativity, integer-valued). N-BEATS predicts unconstrained reals; you can post-process or use a quantile loss with output clipping, but a model with native distributional output (DeepAR) might be simpler.

---

## Q&A

**Why polynomials and Fourier specifically?**
Polynomials are dense in continuous functions on a compact interval (Stone-Weierstrass) and Fourier is dense in periodic functions. Together they form a strong prior for "smooth trend plus periodic component", which matches what most real series look like.

**Can the trend block learn high-order behaviour with a degree-3 polynomial?**
Locally yes, globally no. Each block sees the *residual* after the previous blocks, so a stack of three trend blocks can compose three cubic functions, which is enough to fit any reasonable trend on a forecast horizon.

**Why Adam and not SGD?**
Empirically N-BEATS converges much faster with Adam. The paper reports that SGD requires aggressive learning-rate tuning to match Adam's defaults.

**Does N-BEATS need positional encodings?**
No. The MLP sees the entire window as a fixed-size vector; "position" is implicit in the column index of the input, and the basis matrices encode the per-step time index for the output.

**How do I extend N-BEATS to use exogenous variables (weather, holidays)?**
**N-BEATS-X** is the official extension: concatenate the exogenous covariates to the input window and add an auxiliary input head per block. For most production cases, simply concatenating exogenous time series along the input vector and slightly widening the first MLP layer works fine.

**What is N-HiTS and should I just use that?**
N-HiTS (2023) is by the same authors. It adds multi-rate downsampling and interpolation on the output side, which makes it scale to much longer horizons (720+ steps) and run faster. For short-to-medium horizons (<100 steps), vanilla N-BEATS is still competitive and simpler.

**Why ensemble at all? My single model is fine.**
You may be lucky on this dataset. The paper shows that across the 100k M4 series, single-model variance is large -- an ensemble of even five members reduces 1-2 sMAPE points. If you only need a single point estimate and your dataset is small, you can skip; if you are reporting numbers, ensemble.

---

## Summary

N-BEATS is a "boringly architected" model that wins by stacking the right blocks in the right order. The double residual stream gives it boosting-like behaviour; the basis-function output heads give it a strong inductive bias and (in the interpretable variant) free decomposition; the M4 leaderboard validates that the combination beats both classical methods and recurrent deep models.

For most univariate forecasting problems with regular sampling and clear trend/seasonal structure, N-BEATS is one of the strongest off-the-shelf baselines you can deploy. Start with the interpretable variant for stakeholder buy-in, switch to (or ensemble with) the generic variant if you need every last fraction of accuracy, and remember to ensemble across lookback lengths and seeds.

Next chapter we close the series with **Informer**, which solves a different problem: how to push a Transformer to thousand-step horizons without the $\mathcal{O}(L^2)$ attention cost killing you.

---

## References and further reading

- Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). *N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting.* ICLR.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020). *The M4 Competition: 100,000 Time Series and 61 Forecasting Methods.* International Journal of Forecasting, 36(1).
- Smyl, S. (2020). *A Hybrid Method of Exponential Smoothing and Recurrent Neural Networks for Time Series Forecasting.* International Journal of Forecasting, 36(1).
- Challu, C. et al. (2023). *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting.* AAAI.
- Olivares, K. et al. (2023). *Neural Basis Expansion Analysis with Exogenous Variables: Forecasting Electricity Prices with N-BEATS-X.* International Journal of Forecasting.

*This article is part of the Time Series Forecasting series. Use the navigation at the top to jump to other chapters.*
