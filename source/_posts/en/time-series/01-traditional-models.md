---
title: "Time Series Forecasting (1): Traditional Statistical Models"
date: 2024-11-15 09:00:00
tags:
  - Time Series
  - ARIMA
  - Statistical Models
categories: Algorithm
series: Time Series Forecasting
lang: en
mathjax: true
description: "ARIMA, SARIMA, VAR, GARCH, Prophet, exponential smoothing and the Kalman filter, derived from a single state-space view. With Box-Jenkins workflow, ACF/PACF identification, and runnable Python."
disableNunjucks: true
series_order: 1
---

> [Next: LSTM Deep Dive -->](/en/time-series-lstm/)

## What You Will Learn

- Why **stationarity** is the entry ticket for the whole ARIMA family, and how differencing buys it.
- How to read **ACF and PACF** plots like a Box-Jenkins practitioner: cut-off vs. tail-off as the rule for identifying $p$ and $q$.
- The full **ARIMA / SARIMA** machinery, including how seasonality is folded in via lag-$s$ operators.
- Where **VAR, GARCH, exponential smoothing, Prophet and the Kalman filter** sit on the same map -- mean dynamics vs. variance dynamics vs. state-space recursion.
- A decision rule for when a traditional model is the right answer and when to graduate to the deep models in the rest of this series.

## Prerequisites

- Basic probability and statistics (mean, variance, covariance, correlation).
- Familiarity with NumPy and `pandas` time indexes.
- A little linear algebra for the VAR / Kalman sections (matrix multiplication, eigenvalues).

---

## 1. Why traditional models still matter

Before the deep-learning era, the time-series toolbox was already remarkably complete. ARIMA captures linear autocorrelation, SARIMA adds calendar effects, VAR generalises to vectors, GARCH models the variance, and the Kalman filter unifies the lot inside a state-space recursion. They share three properties that deep models do not give for free:

1. **Interpretability.** Every parameter has a meaning -- "yesterday's level matters with weight $\phi_1$", "the shock from two months ago decays with weight $\theta_2$".
2. **Calibrated uncertainty.** Confidence intervals fall out of maximum likelihood, not from ad-hoc dropout tricks.
3. **Sample efficiency.** A few hundred observations is enough; you do not need a GPU.

If your series is short, smooth, or has clean calendar structure, a traditional model will routinely beat an LSTM and is much easier to debug. Use this article as the **baseline you must defeat** before reaching for anything fancier.

---

## 2. The decomposition view

A good mental picture before any modelling: most series can be written as

$$
y_t = T_t + S_t + R_t,
$$

a slowly moving **trend** $T_t$, a periodic **seasonal** component $S_t$ (period $s$), and a **residual** $R_t$ that should look like noise once the structure is removed. The classical additive decomposition makes this concrete:

![Classical additive decomposition of a synthetic monthly series into trend, seasonality and residual.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/01-traditional-models/fig1_components.png)
*Fig. 1 -- Classical additive decomposition. Once trend and seasonality are subtracted, the residual should be approximately stationary white noise; that is exactly the regime where ARMA-style models are well-posed.*

The whole ARIMA programme can be summarised in one sentence: **transform the data until what is left looks stationary, then fit a linear model with autocorrelated errors.**

### Stationarity, formally

A series is **(weakly) stationary** if its mean, variance and autocovariances do not depend on $t$:

$$
\mathbb{E}[y_t] = \mu, \qquad \mathrm{Var}(y_t) = \sigma^2, \qquad \mathrm{Cov}(y_t, y_{t-k}) = \gamma_k.
$$

Most real series are *not* stationary -- they trend, drift, or have variance that grows with the level. The two standard remedies are:

- **Differencing**: $\nabla y_t = y_t - y_{t-1}$ removes a linear trend; $\nabla^2 y_t$ removes a quadratic one.
- **Variance-stabilising transforms**: $\log y_t$ or a Box-Cox transform tames multiplicative growth.

The **Augmented Dickey-Fuller (ADF)** test gives a hypothesis test: $H_0$ is "unit root present", so a small $p$-value lets you treat the (transformed) series as stationary.

---

## 3. AR, MA, and ARMA -- three flavours of memory

ARIMA is built from two atomic ingredients. They look similar but encode very different ideas about *what* the series remembers.

![Sample paths from AR(1), MA(2) and ARMA(1,1) processes contrasted on the same axes.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/01-traditional-models/fig2_ar_ma_arma.png)
*Fig. 2 -- AR remembers past values, MA remembers past shocks, ARMA does both. The qualitative differences are visible in the sample paths even before any formal statistic is computed.*

### Autoregressive: AR($p$)

$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + \cdots + \phi_p y_{t-p} + \varepsilon_t.
$$

Today's value is a linear combination of the previous $p$ values plus a white-noise shock $\varepsilon_t \sim \mathcal{N}(0, \sigma^2)$. Persistence is encoded directly in the coefficients $\phi_k$. AR(1) with $\phi_1$ near 1 produces a very smooth, slowly mean-reverting path.

### Moving average: MA($q$)

$$
y_t = \mu + \varepsilon_t + \theta_1 \varepsilon_{t-1} + \theta_2 \varepsilon_{t-2} + \cdots + \theta_q \varepsilon_{t-q}.
$$

Today's value is a linear combination of the *last $q$ shocks*. The memory window is finite: an event that happened more than $q$ steps ago has zero direct effect.

### Combining: ARMA($p$, $q$)

$$
\phi(B)\, y_t = \theta(B)\, \varepsilon_t,
$$

where $B$ is the **lag operator** ($B y_t = y_{t-1}$) and

$$
\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p, \qquad
\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q.
$$

ARMA is *parsimonious*: a small $(p, q)$ can imitate the autocorrelation of an MA($\infty$) or AR($\infty$). For stationarity all roots of $\phi(B) = 0$ must lie outside the unit circle; for invertibility the same holds for $\theta(B)$.

### From ARMA to ARIMA

If the raw series is non-stationary, take $d$-th differences and fit ARMA on the result. The full notation is **ARIMA($p, d, q$)**:

$$
\phi(B)\, (1-B)^d\, y_t = \theta(B)\, \varepsilon_t.
$$

In practice $d = 1$ handles linear drift, $d = 2$ handles curvature; rarely do you need $d > 2$.

---

## 4. ACF and PACF -- the model-identification microscope

How do you pick $p$ and $q$? Two diagnostic plots almost always do the job.

- **Autocorrelation function (ACF)**: $\rho_k = \mathrm{Corr}(y_t, y_{t-k})$.
- **Partial autocorrelation (PACF)**: the correlation between $y_t$ and $y_{t-k}$ *after removing* the linear effect of $y_{t-1}, \ldots, y_{t-k+1}$.

The Box-Jenkins identification rules:

| Process | ACF | PACF |
|---------|-----|------|
| AR($p$) | tails off (decays smoothly) | **cuts off after lag $p$** |
| MA($q$) | **cuts off after lag $q$** | tails off |
| ARMA($p$, $q$) | tails off | tails off |

![ACF and PACF of an AR(2) process and an MA(2) process.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/01-traditional-models/fig3_acf_pacf.png)
*Fig. 3 -- The PACF of an AR(2) drops to zero after lag 2 (the spike pattern is the signature). Symmetrically, the ACF of an MA(2) drops to zero after lag 2. When in doubt, prefer the simpler model -- the diagnostic tests in step 5 will catch under-fitting.*

When neither plot has a clean cut-off, you are looking at an ARMA process; the cleanest path is then to grid-search $(p, q)$ and pick the pair that minimises an information criterion.

### AIC and BIC

$$
\mathrm{AIC} = -2\,\ell(\hat{\theta}) + 2k, \qquad
\mathrm{BIC} = -2\,\ell(\hat{\theta}) + k\log n,
$$

with $\ell$ the maximised log-likelihood, $k$ the number of free parameters, $n$ the sample size. BIC penalises complexity more harshly and is the safer default when $n$ is small.

---

## 5. The Box-Jenkins workflow

ARIMA is not a one-shot fit; it is an **iterative loop** that Box and Jenkins formalised in 1970. Every subsequent statistical-forecasting toolkit (including `auto.arima`) is just an automation of the same four boxes.

![Box-Jenkins methodology: identification, estimation, diagnostic checking and forecasting, with a feedback loop when residual diagnostics fail.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/01-traditional-models/fig6_box_jenkins_flow.png)
*Fig. 4 -- The point is the feedback arrow. If the fitted residuals fail a Ljung-Box test or show structure in their ACF, the model is misspecified -- go back and revise $(p, d, q)$ rather than trusting the forecast.*

1. **Identification.** Plot the series, run ADF for stationarity, study ACF/PACF, propose a candidate $(p, d, q)$.
2. **Estimation.** Fit by maximum likelihood (or conditional least squares). All major libraries do this for you.
3. **Diagnostic checking.** Are the residuals white noise?
    - The **Ljung-Box** statistic $Q = n(n+2)\sum_{k=1}^h \hat{\rho}_k^2 / (n-k)$ should fail to reject $H_0$.
    - The residual ACF should lie inside the $\pm 1.96/\sqrt{n}$ band.
    - QQ-plots and Jarque-Bera check normality (matters for prediction intervals).
4. **Forecasting.** Only after the residuals look clean do you produce point and interval forecasts.

---

## 6. ARIMA in code

A faithful, minimal walk-through using `statsmodels`:

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

# 1. Simulate a non-stationary series (random walk + drift)
rng = np.random.default_rng(0)
n = 200
eps = rng.normal(0, 1.0, n)
y = np.zeros(n)
for t in range(1, n):
    y[t] = y[t - 1] + 0.05 + 0.6 * eps[t]

series = pd.Series(y, index=pd.date_range("2018-01-01", periods=n, freq="D"))

# 2. Identification: ADF test on the level vs the first difference
print("ADF on level :", adfuller(series)[1])         # large p -> non-stationary
print("ADF on diff  :", adfuller(series.diff().dropna())[1])  # small p -> stationary

# 3. Estimation: train on the first 170 points
train, test = series.iloc[:-30], series.iloc[-30:]
model = ARIMA(train, order=(2, 1, 1)).fit()

# 4. Diagnostic checking
lb = acorr_ljungbox(model.resid, lags=[10], return_df=True)
print(lb)  # high p-value -> residuals are white noise

# 5. Forecasting with a 95% interval
fc = model.get_forecast(steps=30)
mean_fc = fc.predicted_mean
ci = fc.conf_int(alpha=0.05)
```

The forecast on a held-out tail looks like this:

![ARIMA(2,1,1) forecast with 95% confidence band on a held-out segment.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/01-traditional-models/fig4_arima_forecast.png)
*Fig. 5 -- The point forecast quickly reverts to the long-run drift, and the interval fans out at rate $\sigma\sqrt{h}$ in the horizon $h$. That widening cone is the honest representation of how little a linear model truly knows about the future.*

---

## 7. Seasonality: SARIMA

Many series have a calendar that ARIMA on its own cannot exploit -- monthly retail with a December peak, daily traffic with a weekly cycle, hourly load with a 24-hour cycle. **SARIMA** ($p, d, q$)($P, D, Q$)$_s$ folds in seasonal lags of period $s$:

$$
\Phi(B^s)\, \phi(B)\, (1-B)^d (1-B^s)^D y_t \;=\; \Theta(B^s)\, \theta(B)\, \varepsilon_t.
$$

You apply two kinds of differencing -- regular ($1-B$) for trend and seasonal ($1-B^s$) for the period -- and then attach AR/MA terms at both regular lags and lags that are multiples of $s$.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),  # P, D, Q, s
    enforce_stationarity=False,
    enforce_invertibility=False,
).fit(disp=False)

forecast = model.get_forecast(steps=24)
```

![SARIMA(1,1,1)(1,1,1,12) forecast on a series with strong yearly seasonality.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/01-traditional-models/fig5_sarima_forecast.png)
*Fig. 6 -- The seasonal cycle is reproduced cleanly out of sample because the seasonal-difference operator $(1-B^{12})$ has stripped the yearly pattern from the residual, and the seasonal AR/MA terms put it back in the forecast at the right phase.*

**Identification cheat sheet for the seasonal part**: look at the ACF/PACF of the *seasonally differenced* series at the seasonal lags $s, 2s, 3s, \ldots$. The same cut-off / tail-off rules apply as in the non-seasonal case.

---

## 8. Beyond ARIMA: the rest of the family

The ideas above generalise in four useful directions. They are not separate worlds -- each one specialises ARIMA in a single dimension.

### 8.1 VAR -- multivariate dynamics

When you have several series that influence each other (GDP and unemployment, electricity demand and temperature), promote the scalar AR to a **vector autoregression**:

$$
\mathbf{y}_t = \mathbf{c} + A_1 \mathbf{y}_{t-1} + A_2 \mathbf{y}_{t-2} + \cdots + A_p \mathbf{y}_{t-p} + \boldsymbol{\varepsilon}_t.
$$

Each entry of the matrix $A_k$ has an interpretation: $(A_k)_{ij}$ is the marginal effect of variable $j$ at lag $k$ on variable $i$ today. This makes VAR popular in macroeconomics where **Granger causality** ("does $x$ help predict $y$ beyond $y$'s own past?") is the central question.

A practical caveat: with $K$ series and lag $p$ the model has $K + pK^2$ free parameters. For $K = 10, p = 4$ that is already 410 numbers from probably a few hundred observations. Hence the regularised cousins -- Bayesian VAR, factor models -- exist for high-dimensional settings.

### 8.2 GARCH -- variance dynamics

ARIMA models the **mean**; GARCH models the **conditional variance**. The basic GARCH(1,1) is:

$$
\sigma_t^2 = \omega + \alpha\, \varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2,
\qquad \varepsilon_t = \sigma_t\, z_t,\quad z_t \sim \mathcal{N}(0, 1).
$$

The $\alpha$ term lets a large shock yesterday push variance up today (the "ARCH" effect); the $\beta$ term lets variance be persistent. Stationarity requires $\alpha + \beta < 1$, and in financial data $\alpha + \beta$ typically lands near 0.95-0.99 -- volatility is a slow-moving thing.

GARCH is *the* standard tool for risk management (VaR, options pricing) and pairs naturally with an ARIMA mean model: fit ARIMA to the returns, then GARCH to the squared residuals.

### 8.3 Exponential smoothing and Holt-Winters

If ARIMA is the "explicitly stochastic" view, exponential smoothing is the "weighted-average" view. Simple exponential smoothing assumes a level $\ell_t$ that drifts with new information:

$$
\ell_t = \alpha\, y_t + (1-\alpha)\, \ell_{t-1}.
$$

**Holt** adds a trend $b_t$; **Holt-Winters** adds a seasonal $s_t$. The additive variant is:

$$
\ell_t = \alpha (y_t - s_{t-s}) + (1-\alpha)(\ell_{t-1} + b_{t-1}),\\
b_t = \beta(\ell_t - \ell_{t-1}) + (1-\beta) b_{t-1},\\
s_t = \gamma(y_t - \ell_t) + (1-\gamma) s_{t-s}.
$$

This is the workhorse behind the **ETS** family in R / `statsmodels`, and many of the methods that won the M-competitions are sophisticated descendants of it.

![Holt-Winters additive forecast and the underlying level / trend / seasonal components.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/time-series/01-traditional-models/fig7_holt_winters.png)
*Fig. 7 -- The decomposition gives Holt-Winters its appeal: each component is intelligible on its own, the smoothing coefficients ($\alpha, \beta, \gamma$) say how quickly each one adapts, and the recursive form means it costs $O(n)$ to fit.*

### 8.4 Prophet

Prophet (Facebook, 2017) is a deliberately simple **additive** decomposition aimed at business analysts:

$$
y(t) = g(t) + s(t) + h(t) + \varepsilon_t,
$$

where $g(t)$ is a piecewise-linear or logistic trend with **changepoints**, $s(t)$ is a Fourier expansion for multiple seasonalities, and $h(t)$ encodes user-supplied holidays. It is fitted with Stan via MAP or full Bayesian sampling, exposes only a handful of tunables, and is robust to missing data and outliers. In practice it is *not* state of the art on benchmarks, but the API quality and the holiday handling are why it remains a default in product analytics.

### 8.5 The Kalman filter and the state-space view

Everything above is a special case of a **linear Gaussian state-space model**:

$$
\mathbf{x}_t = F_t \mathbf{x}_{t-1} + \mathbf{w}_t, \qquad \mathbf{w}_t \sim \mathcal{N}(0, Q_t),\\
\mathbf{y}_t = H_t \mathbf{x}_t + \mathbf{v}_t, \qquad \mathbf{v}_t \sim \mathcal{N}(0, R_t).
$$

The **Kalman filter** is the exact Bayesian recursion that maintains $p(\mathbf{x}_t \mid \mathbf{y}_{1:t}) = \mathcal{N}(\hat{\mathbf{x}}_t, P_t)$ as data arrive:

$$
\hat{\mathbf{x}}_{t|t-1} = F_t\, \hat{\mathbf{x}}_{t-1},\\
P_{t|t-1} = F_t\, P_{t-1}\, F_t^\top + Q_t,\\
K_t = P_{t|t-1} H_t^\top (H_t P_{t|t-1} H_t^\top + R_t)^{-1},\\
\hat{\mathbf{x}}_t = \hat{\mathbf{x}}_{t|t-1} + K_t (\mathbf{y}_t - H_t \hat{\mathbf{x}}_{t|t-1}),\\
P_t = (I - K_t H_t)\, P_{t|t-1}.
$$

Why this matters: ARIMA, exponential smoothing, dynamic linear regressions, structural models with seasonal dummies -- all of them can be cast in the form above and inherit the Kalman recursion for free. That is exactly how `statsmodels`'s `SARIMAX` works under the hood, and why it can handle missing observations and exogenous regressors with no extra effort.

---

## 9. Choosing a model

| Model | Best when | Avoid when |
|-------|-----------|------------|
| **ARIMA** | One series, linear autocorrelation, no calendar | Strong seasonality, multiple inputs |
| **SARIMA** | Clear single seasonal period, moderate length | Multiple overlapping seasonalities |
| **VAR** | Several stationary series with feedback | High-dimensional ($K \gtrsim 10$) |
| **GARCH** | Returns / volatility data with clustering | Slow, smooth series with constant variance |
| **Holt-Winters / ETS** | Robust seasonal baseline, fast retraining | Highly non-linear regimes |
| **Prophet** | Business series with holidays, missing data | Sub-daily high-frequency data |
| **Kalman / state-space** | Online updating, missing data, custom structure | When a black-box is acceptable and abundant data is available |

A pragmatic recipe: start with **ETS or SARIMA** as a baseline, add **GARCH** if the variance is the quantity you care about, and only move to deep models (LSTM, TCN, Transformer, N-BEATS, Informer -- the rest of this series) when the residuals show clear non-linear structure or you have many parallel series to share information across.

---

## 10. Limits, and what comes next

Traditional models reach their ceiling when:

- The dynamics are **non-linear** (regime switches, thresholds, hard saturations).
- You have **hundreds or thousands of parallel series** that should share parameters.
- The relevant context window is very long and ARMA roots cannot represent it parsimoniously.
- The signal is **non-Gaussian** in a way that even Box-Cox cannot fix.

That is the launchpad for the next seven articles. We start with **LSTM** as the canonical non-linear sequential model, then build up through GRU, attention, the Transformer, TCN, N-BEATS, and finally Informer for very long sequences. Each of them generalises one of the linear models above; treat ARIMA / SARIMA / Holt-Winters as the baselines you must beat, not as the past you can skip.

---

## References

- Box, G., Jenkins, G., Reinsel, G., & Ljung, G. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press.
- Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.). OTexts. https://otexts.com/fpp3/
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.
- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987-1007.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1), 37-45.

---

**Series Navigation**
