---
title: "Probability and Statistics (2): Random Variables and the Distributions That Matter"
date: 2024-08-20 09:00:00
tags:
  - Probability
  - Statistics
  - Distributions
  - Random Variables
categories:
  - Probability and Statistics
series: probability-statistics
lang: en
mathjax: true
description: "A rigorous tour of random variables, PMFs, PDFs, CDFs, and every distribution that matters in practice — Bernoulli, Binomial, Poisson, Gaussian, Exponential, Gamma, and Beta — with derivations, proofs, and Python visualizations."
disableNunjucks: true
series_order: 2
translationKey: "probability-statistics-2"
---

After building the axiomatic foundation in the previous article, you might feel like we spent a lot of time talking about sets and subsets. That's because we did. The machinery of events and sigma-algebras is necessary but austere — it doesn't give us a natural way to compute averages, measure spread, or fit models to data.

The bridge between abstract probability and applied statistics is the **random variable**. Once we assign numerical values to outcomes, the entire toolkit of calculus — derivatives, integrals, series — becomes available. And with calculus comes the ability to characterize randomness through a small set of named distributions, each encoding specific assumptions about how the world generates data.

This article catalogs the distributions you'll encounter most often and shows exactly where each one comes from.

## Random Variables as Functions

**Definition.** A **random variable** $X$ is a function from the sample space to the real numbers:

$$X: \Omega \to \mathbb{R}$$

such that for every real number $x$, the set $\{\omega \in \Omega : X(\omega) \leq x\}$ is an event in $\mathcal{F}$.

The measurability condition (the second part of the definition) ensures that questions like "what is the probability that $X$ is at most 3?" have well-defined answers. For finite and countable sample spaces, this condition is automatically satisfied.

**Example.** Roll two dice. Let $X$ be their sum. The sample space is $\Omega = \{(i,j) : 1 \leq i,j \leq 6\}$ with 36 equally likely outcomes. The random variable $X(i,j) = i + j$ maps each pair to a number between 2 and 12.

The key shift: instead of tracking the full outcome $\omega$, we work with the number $X(\omega)$. This loses information (knowing $X = 7$ doesn't tell us whether the roll was $(1,6)$ or $(3,4)$), but what we lose in detail we gain in mathematical power.

## Discrete Random Variables

A random variable is **discrete** if it takes values in a countable set (finite or countably infinite).

### Probability Mass Function (PMF)

The **PMF** of a discrete random variable $X$ is

$$p_X(x) = P(X = x)$$

for each value $x$ in the support of $X$. Properties:

1. $p_X(x) \geq 0$ for all $x$
2. $\sum_{x} p_X(x) = 1$ (sum over all values in the support)

### Cumulative Distribution Function (CDF)

The **CDF** of any random variable (discrete or continuous) is

$$F_X(x) = P(X \leq x) = \sum_{t \leq x} p_X(t) \quad \text{(discrete case)}.$$

The CDF is right-continuous, non-decreasing, with $\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to \infty} F(x) = 1$.

## Key Discrete Distributions

### Bernoulli Distribution

The simplest random variable: a single trial with two outcomes.

$$X \sim \text{Bernoulli}(p), \quad p_X(x) = p^x (1-p)^{1-x} \text{ for } x \in \{0, 1\}.$$

- **Mean:** $E[X] = p$
- **Variance:** $\text{Var}(X) = p(1-p)$

Every binary outcome — coin flip, click/no-click, spam/not-spam — is a Bernoulli trial.

### Binomial Distribution

The number of successes in $n$ independent Bernoulli trials.

$$X \sim \text{Binomial}(n, p), \quad p_X(k) = \binom{n}{k} p^k (1-p)^{n-k} \text{ for } k = 0, 1, \ldots, n.$$

*Derivation.* A specific sequence with exactly $k$ successes has probability $p^k(1-p)^{n-k}$. The number of such sequences is $\binom{n}{k}$.

- **Mean:** $E[X] = np$
- **Variance:** $\text{Var}(X) = np(1-p)$

*Proof of mean.* Write $X = X_1 + X_2 + \cdots + X_n$ where each $X_i \sim \text{Bernoulli}(p)$. By linearity of expectation: $E[X] = \sum E[X_i] = np$. $\blacksquare$

### Geometric Distribution

The number of trials until the first success.

$$X \sim \text{Geometric}(p), \quad p_X(k) = (1-p)^{k-1} p \text{ for } k = 1, 2, 3, \ldots$$

*Verification of normalization:*

$$\sum_{k=1}^{\infty} (1-p)^{k-1} p = p \sum_{j=0}^{\infty} (1-p)^j = p \cdot \frac{1}{1-(1-p)} = p \cdot \frac{1}{p} = 1. \quad \checkmark$$

- **Mean:** $E[X] = 1/p$
- **Variance:** $\text{Var}(X) = (1-p)/p^2$

The Geometric distribution has the **memoryless property**: $P(X > s + t \mid X > s) = P(X > t)$. Given that you've already waited $s$ trials without success, the distribution of remaining wait time is the same as if you started fresh.

*Proof.* $P(X > n) = (1-p)^n$ (all $n$ trials are failures). Then:

$$P(X > s+t \mid X > s) = \frac{P(X > s+t)}{P(X > s)} = \frac{(1-p)^{s+t}}{(1-p)^s} = (1-p)^t = P(X > t). \quad \blacksquare$$

**The Negative Binomial Distribution.** A generalization: the number of trials until the $r$-th success.

$$X \sim \text{NegBin}(r, p), \quad p_X(k) = \binom{k-1}{r-1} p^r (1-p)^{k-r} \text{ for } k = r, r+1, \ldots$$

The Geometric is the special case $r = 1$. The Negative Binomial arises naturally when modeling overdispersed count data (variance exceeds mean), making it a popular alternative to Poisson in practice.

- **Mean:** $E[X] = r/p$
- **Variance:** $\text{Var}(X) = r(1-p)/p^2$

### Poisson Distribution

The number of events occurring in a fixed interval when events happen at a constant average rate.

$$X \sim \text{Poisson}(\lambda), \quad p_X(k) = \frac{\lambda^k e^{-\lambda}}{k!} \text{ for } k = 0, 1, 2, \ldots$$

- **Mean:** $E[X] = \lambda$
- **Variance:** $\text{Var}(X) = \lambda$ (mean equals variance — a signature of Poisson)

*Proof of mean.*

$$E[X] = \sum_{k=0}^{\infty} k \frac{\lambda^k e^{-\lambda}}{k!} = \lambda e^{-\lambda} \sum_{k=1}^{\infty} \frac{\lambda^{k-1}}{(k-1)!} = \lambda e^{-\lambda} \cdot e^{\lambda} = \lambda. \quad \blacksquare$$

### Poisson Approximation to Binomial

When $n$ is large, $p$ is small, and $\lambda = np$ is moderate, $\text{Binomial}(n, p) \approx \text{Poisson}(\lambda)$.

*Proof sketch.* For fixed $k$:

$$\binom{n}{k} p^k (1-p)^{n-k} = \frac{n!}{k!(n-k)!} \left(\frac{\lambda}{n}\right)^k \left(1 - \frac{\lambda}{n}\right)^{n-k}.$$

As $n \to \infty$ with $\lambda = np$ fixed:
- $\frac{n!}{(n-k)! \cdot n^k} \to 1$
- $(1 - \lambda/n)^n \to e^{-\lambda}$
- $(1 - \lambda/n)^{-k} \to 1$

So the whole expression converges to $\frac{\lambda^k e^{-\lambda}}{k!}$. $\blacksquare$

**Rule of thumb:** The approximation is good when $n \geq 20$ and $p \leq 0.05$.

## Continuous Random Variables

A random variable is **continuous** if there exists a non-negative function $f_X$ (the **probability density function** or **PDF**) such that

$$P(a \leq X \leq b) = \int_a^b f_X(x) \, dx.$$

Properties:
1. $f_X(x) \geq 0$ for all $x$
2. $\int_{-\infty}^{\infty} f_X(x) \, dx = 1$

**Critical distinction:** For continuous random variables, $P(X = x) = 0$ for any single value $x$. This is not a contradiction — the density $f(x)$ can be positive even though the probability of any specific point is zero. Probability lives in intervals, not points.

The CDF is:

$$F_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t) \, dt$$

and the PDF is the derivative of the CDF (where it exists):

$$f_X(x) = F_X'(x).$$

## Key Continuous Distributions

### Uniform Distribution

$$X \sim \text{Uniform}(a, b), \quad f_X(x) = \frac{1}{b-a} \text{ for } x \in [a, b].$$

- **Mean:** $E[X] = (a+b)/2$
- **Variance:** $\text{Var}(X) = (b-a)^2/12$
- **CDF:** $F_X(x) = (x-a)/(b-a)$ for $x \in [a,b]$

The "maximum ignorance" distribution — every value in $[a,b]$ is equally likely.

### Exponential Distribution

$$X \sim \text{Exponential}(\lambda), \quad f_X(x) = \lambda e^{-\lambda x} \text{ for } x \geq 0.$$

- **Mean:** $E[X] = 1/\lambda$
- **Variance:** $\text{Var}(X) = 1/\lambda^2$
- **CDF:** $F_X(x) = 1 - e^{-\lambda x}$

The continuous analog of the Geometric distribution. It is the **only** continuous distribution with the memoryless property:

$$P(X > s + t \mid X > s) = P(X > t).$$

*Proof.* $P(X > x) = e^{-\lambda x}$. Then:

$$P(X > s+t \mid X > s) = \frac{P(X > s+t)}{P(X > s)} = \frac{e^{-\lambda(s+t)}}{e^{-\lambda s}} = e^{-\lambda t} = P(X > t). \quad \blacksquare$$

This makes the Exponential distribution natural for modeling waiting times in processes with no memory — radioactive decay, inter-arrival times in a Poisson process, time between server requests.

### Gaussian (Normal) Distribution

The most important distribution in all of statistics.

$$X \sim \mathcal{N}(\mu, \sigma^2), \quad f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right).$$

- **Mean:** $E[X] = \mu$
- **Variance:** $\text{Var}(X) = \sigma^2$

The **standard normal** $Z \sim \mathcal{N}(0, 1)$ has $\mu = 0$, $\sigma = 1$. Any normal can be standardized:

$$Z = \frac{X - \mu}{\sigma}.$$

**The 68-95-99.7 Rule:**

| Interval | Probability |
|---|---|
| $\mu \pm \sigma$ | 0.6827 |
| $\mu \pm 2\sigma$ | 0.9545 |
| $\mu \pm 3\sigma$ | 0.9973 |

**Why is the Normal distribution so important?** Three reasons:

1. **The Central Limit Theorem** (Article 5): sums and averages of many independent random variables converge to a Normal, regardless of the original distribution.
2. **Maximum entropy:** Among all distributions with a given mean and variance, the Normal has the highest entropy (the most "random" or "least informative"). Using a Normal when you only know the mean and variance is the most conservative choice.
3. **Mathematical convenience:** The Normal is closed under linear combinations, conditioning, and marginalization — making it the backbone of linear regression, Kalman filters, and Gaussian processes.

### The Log-Normal Distribution

If $X \sim \mathcal{N}(\mu, \sigma^2)$, then $Y = e^X$ follows a **log-normal** distribution. Its PDF is:

$$f_Y(y) = \frac{1}{y\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln y - \mu)^2}{2\sigma^2}\right) \quad \text{for } y > 0.$$

- **Mean:** $E[Y] = e^{\mu + \sigma^2/2}$
- **Variance:** $\text{Var}(Y) = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$

The log-normal models quantities that are products of many positive factors (incomes, stock prices, particle sizes), just as the Normal models sums. It is always right-skewed and positive-valued.

*Proof that the PDF integrates to 1.* Let $I = \int_{-\infty}^{\infty} e^{-x^2/2} dx$. Then:

$$I^2 = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} e^{-(x^2 + y^2)/2} dx \, dy.$$

Switch to polar coordinates: $x^2 + y^2 = r^2$, $dx \, dy = r \, dr \, d\theta$:

$$I^2 = \int_0^{2\pi} \int_0^{\infty} e^{-r^2/2} r \, dr \, d\theta = 2\pi \int_0^{\infty} r e^{-r^2/2} dr = 2\pi \left[-e^{-r^2/2}\right]_0^{\infty} = 2\pi.$$

So $I = \sqrt{2\pi}$, confirming $\frac{1}{\sqrt{2\pi}} e^{-x^2/2}$ integrates to 1. $\blacksquare$

### Gamma Distribution

A generalization of the Exponential: the sum of $\alpha$ independent $\text{Exponential}(\beta)$ random variables.

$$X \sim \text{Gamma}(\alpha, \beta), \quad f_X(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x} \text{ for } x > 0$$

where $\Gamma(\alpha) = \int_0^{\infty} t^{\alpha-1} e^{-t} dt$ is the gamma function. When $\alpha$ is a positive integer, $\Gamma(\alpha) = (\alpha - 1)!$.

- **Mean:** $E[X] = \alpha/\beta$
- **Variance:** $\text{Var}(X) = \alpha/\beta^2$

Special cases: $\text{Gamma}(1, \lambda) = \text{Exponential}(\lambda)$; $\text{Gamma}(n/2, 1/2) = \chi^2(n)$ (chi-squared with $n$ degrees of freedom).

### Beta Distribution

$$X \sim \text{Beta}(\alpha, \beta), \quad f_X(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)} \text{ for } x \in (0, 1)$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the beta function.

- **Mean:** $E[X] = \frac{\alpha}{\alpha + \beta}$
- **Variance:** $\text{Var}(X) = \frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$

The Beta distribution lives on $[0,1]$, making it natural for modeling probabilities. It is the conjugate prior for the Bernoulli and Binomial likelihoods — a fact we'll exploit heavily in Article 8 on Bayesian statistics.

Special cases: $\text{Beta}(1,1) = \text{Uniform}(0,1)$.

## Distribution Reference Table

| Distribution | Type | PMF/PDF | Mean | Variance | Typical Use |
|---|---|---|---|---|---|
| Bernoulli($p$) | Discrete | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ | Binary outcome |
| Binomial($n,p$) | Discrete | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | Count of successes |
| Geometric($p$) | Discrete | $(1-p)^{k-1}p$ | $1/p$ | $(1-p)/p^2$ | Trials until first success |
| Poisson($\lambda$) | Discrete | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | Event counts in fixed interval |
| Uniform($a,b$) | Continuous | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | Maximum ignorance |
| Exponential($\lambda$) | Continuous | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | Waiting times |
| Normal($\mu,\sigma^2$) | Continuous | $\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ | Everything (CLT) |
| Gamma($\alpha,\beta$) | Continuous | $\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ | $\alpha/\beta$ | $\alpha/\beta^2$ | Sum of waiting times |
| Beta($\alpha,\beta$) | Continuous | $\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$ | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ | Modeling probabilities |

## Python: Visualizing All Major Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# 1. Bernoulli
ax = axes[0, 0]
for p in [0.2, 0.5, 0.8]:
    ax.bar([0, 1], [1-p, p], alpha=0.5, width=0.3, label=f'p={p}')
ax.set_title('Bernoulli(p)')
ax.set_xlabel('x')
ax.set_ylabel('P(X=x)')
ax.legend()

# 2. Binomial
ax = axes[0, 1]
n = 20
for p in [0.2, 0.5, 0.8]:
    k = np.arange(0, n+1)
    ax.plot(k, stats.binom.pmf(k, n, p), 'o-', markersize=4, label=f'n={n}, p={p}')
ax.set_title('Binomial(n, p)')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
ax.legend()

# 3. Geometric
ax = axes[0, 2]
for p in [0.2, 0.5, 0.8]:
    k = np.arange(1, 15)
    ax.plot(k, stats.geom.pmf(k, p), 'o-', markersize=4, label=f'p={p}')
ax.set_title('Geometric(p)')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
ax.legend()

# 4. Poisson
ax = axes[1, 0]
for lam in [1, 4, 10]:
    k = np.arange(0, 20)
    ax.plot(k, stats.poisson.pmf(k, lam), 'o-', markersize=4, label=f'$\\lambda$={lam}')
ax.set_title('Poisson($\\lambda$)')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
ax.legend()

# 5. Uniform
ax = axes[1, 1]
x = np.linspace(-0.5, 1.5, 300)
ax.plot(x, stats.uniform.pdf(x, 0, 1), 'b-', linewidth=2, label='Uniform(0,1)')
ax.fill_between(x, stats.uniform.pdf(x, 0, 1), alpha=0.3)
ax.set_title('Uniform(a, b)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# 6. Exponential
ax = axes[1, 2]
x = np.linspace(0, 5, 300)
for lam in [0.5, 1, 2]:
    ax.plot(x, stats.expon.pdf(x, scale=1/lam), linewidth=2, label=f'$\\lambda$={lam}')
ax.set_title('Exponential($\\lambda$)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# 7. Normal
ax = axes[2, 0]
x = np.linspace(-5, 8, 300)
for mu, sigma in [(0,1), (2,0.5), (0,2)]:
    ax.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=2,
            label=f'$\\mu$={mu}, $\\sigma$={sigma}')
ax.set_title('Normal($\\mu$, $\\sigma^2$)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# 8. Gamma
ax = axes[2, 1]
x = np.linspace(0, 15, 300)
for a, b in [(1, 1), (2, 1), (5, 1), (5, 2)]:
    ax.plot(x, stats.gamma.pdf(x, a, scale=1/b), linewidth=2,
            label=f'$\\alpha$={a}, $\\beta$={b}')
ax.set_title('Gamma($\\alpha$, $\\beta$)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# 9. Beta
ax = axes[2, 2]
x = np.linspace(0.001, 0.999, 300)
for a, b in [(0.5, 0.5), (1, 1), (2, 5), (5, 2), (5, 5)]:
    ax.plot(x, stats.beta.pdf(x, a, b), linewidth=2,
            label=f'$\\alpha$={a}, $\\beta$={b}')
ax.set_title('Beta($\\alpha$, $\\beta$)')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

plt.suptitle('Gallery of Probability Distributions', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('distribution_gallery.png', dpi=150, bbox_inches='tight')
plt.show()
```

This gallery lets you see the shape of every distribution at a glance. A few patterns to notice:

- **Binomial** becomes more symmetric as $p$ approaches 0.5.
- **Poisson** shifts right and becomes more symmetric as $\lambda$ grows (approaching a Normal shape, per the CLT).
- **Exponential** is always right-skewed — most waiting times are short, but a few are very long.
- **Beta** is incredibly flexible: U-shaped, uniform, or skewed, depending on the parameters.
- **Gamma** generalizes the Exponential, adding a shape parameter that controls the "hump."

## Connections Between Distributions

The distributions above are not isolated — they form a family with deep connections:

1. **Bernoulli is Binomial with $n=1$**: $\text{Binomial}(1, p) = \text{Bernoulli}(p)$
2. **Binomial is a sum of Bernoullis**: If $X_i \sim \text{Bernoulli}(p)$ i.i.d., then $\sum X_i \sim \text{Binomial}(n, p)$
3. **Poisson approximates Binomial**: $\text{Binomial}(n, \lambda/n) \to \text{Poisson}(\lambda)$ as $n \to \infty$
4. **Geometric is discrete Exponential**: Both are memoryless
5. **Gamma is a sum of Exponentials**: If $X_i \sim \text{Exp}(\lambda)$ i.i.d., then $\sum X_i \sim \text{Gamma}(n, \lambda)$
6. **Chi-squared is a special Gamma**: $\chi^2(n) = \text{Gamma}(n/2, 1/2)$
7. **Beta(1,1) = Uniform(0,1)**: The uniform is a special case of Beta

These connections are not accidents. They reflect deep structural relationships in how random processes generate data.

## Quantile Functions and Inverse CDF

The **quantile function** (or **inverse CDF**) $F^{-1}(p)$ is defined for $p \in (0, 1)$ as:

$$F^{-1}(p) = \inf\{x : F(x) \geq p\}.$$

For continuous distributions with strictly increasing CDFs, this simplifies to: $F^{-1}(p)$ is the unique $x$ such that $F(x) = p$.

Key quantiles have special names:
- $F^{-1}(0.5)$: the **median**
- $F^{-1}(0.25)$ and $F^{-1}(0.75)$: the **quartiles**
- $F^{-1}(0.01), \ldots, F^{-1}(0.99)$: the **percentiles**

The quantile function is essential for generating random samples from any distribution. If $U \sim \text{Uniform}(0, 1)$, then $X = F^{-1}(U)$ has CDF $F$. This is the **inverse CDF method** (or **probability integral transform**).

*Proof.* $P(X \leq x) = P(F^{-1}(U) \leq x) = P(U \leq F(x)) = F(x)$, since $U$ is uniform on $(0,1)$. $\blacksquare$

**Example.** To generate Exponential($\lambda$) samples: $X = -\frac{1}{\lambda}\ln(1-U)$ where $U \sim \text{Uniform}(0,1)$.

*Verification.* $F(x) = 1 - e^{-\lambda x}$, so $F^{-1}(p) = -\frac{1}{\lambda}\ln(1-p)$. $\checkmark$

## Mixtures of Distributions

Not every distribution fits neatly into a single named family. A **mixture distribution** combines multiple components:

$$f(x) = \sum_{k=1}^{K} w_k f_k(x), \qquad \sum_{k=1}^K w_k = 1, \quad w_k \geq 0$$

where $f_k$ are the component densities and $w_k$ are the mixing weights.

**Example.** A population consists of two groups: 70% have income $\sim \mathcal{N}(50000, 10000^2)$ and 30% have income $\sim \mathcal{N}(90000, 15000^2)$. The overall income distribution is a two-component Gaussian mixture — bimodal, not Normal.

Gaussian Mixture Models (GMMs) are a workhorse of unsupervised learning: they model complex, multimodal data as weighted sums of Gaussians, with parameters fit via the Expectation-Maximization (EM) algorithm.

```python
# Visualize a Gaussian mixture
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(10000, 140000, 500)
component1 = 0.7 * stats.norm.pdf(x, 50000, 10000)
component2 = 0.3 * stats.norm.pdf(x, 90000, 15000)
mixture = component1 + component2

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, component1, 'b--', linewidth=1.5, label='Component 1 (70%)')
ax.plot(x, component2, 'r--', linewidth=1.5, label='Component 2 (30%)')
ax.plot(x, mixture, 'k-', linewidth=2.5, label='Mixture')
ax.fill_between(x, mixture, alpha=0.15, color='gray')
ax.set_xlabel('Income ($)')
ax.set_ylabel('Density')
ax.set_title('Gaussian Mixture Model: Bimodal Income Distribution')
ax.legend()
plt.tight_layout()
plt.savefig('gaussian_mixture.png', dpi=150)
plt.show()
```

## Choosing the Right Distribution: A Decision Guide

When modeling real data, choosing the right distribution is critical. Here's a practical decision tree:

**Is the variable discrete or continuous?**

If **discrete**:
- Binary outcome (yes/no): **Bernoulli**
- Count of successes in fixed $n$ trials: **Binomial**
- Count of trials until first success: **Geometric**
- Count of events in a fixed interval, rare events: **Poisson**
- Count of failures before $r$-th success: **Negative Binomial**

If **continuous**:
- All values in an interval equally likely: **Uniform**
- Waiting time, memoryless: **Exponential**
- Sum of waiting times: **Gamma**
- Symmetric bell-shaped, sum of many factors: **Normal**
- Probability/proportion (value in $[0,1]$): **Beta**
- Heavy tails, extreme events: **t-distribution** or **Cauchy**
- Positive values, right-skewed: **Log-Normal** or **Gamma**

**Rule of thumb:** Start simple. Use the Normal as your default for continuous data (CLT justifies this for averages and sums). Only reach for more exotic distributions when the data clearly violate Normality — heavy tails, skewness, bounded support, or discrete counts.

## Functions of Random Variables: A Preview

Given $X$ with known distribution, what is the distribution of $Y = g(X)$? This question arises constantly — transforming features, computing derived quantities, or propagating uncertainty through models. We'll develop the full machinery (Jacobians, convolutions) in Article 4. Here's a taste with a simple case.

**Example.** If $X \sim \mathcal{N}(\mu, \sigma^2)$, what is the distribution of $Y = aX + b$?

Since a linear transformation of a Normal is Normal (proved via MGFs in Article 3):

$$Y = aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2).$$

This is why standardization works: $Z = (X - \mu)/\sigma$ has $\mu_Z = 0$ and $\sigma_Z^2 = 1$.

**Example.** If $X \sim \text{Uniform}(0, 1)$, what is the distribution of $Y = X^2$?

Using the CDF method: $F_Y(y) = P(X^2 \leq y) = P(X \leq \sqrt{y}) = \sqrt{y}$ for $0 \leq y \leq 1$.

Differentiating: $f_Y(y) = \frac{1}{2\sqrt{y}}$ for $0 < y < 1$. This is a $\text{Beta}(1/2, 1)$ distribution.

## What's Next

We can now describe the probability distribution of a single random variable. But a distribution is a full object — it contains far more information than we can easily work with. The next article introduces the summary statistics that compress this information: expectation (the "center"), variance (the "spread"), and the moment-generating function (the "fingerprint" that uniquely identifies a distribution).
