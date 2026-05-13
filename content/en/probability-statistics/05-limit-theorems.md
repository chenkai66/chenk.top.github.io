---
title: "Probability and Statistics (5): Law of Large Numbers and the Central Limit Theorem"
date: 2024-08-24 09:00:00
tags:
  - Probability
  - Statistics
  - Central Limit Theorem
  - Convergence
categories: Probability and Statistics
series: probability-statistics
lang: en
mathjax: true
description: "The two pillars of probability: the Law of Large Numbers guarantees sample means converge, and the Central Limit Theorem explains why everything looks Gaussian — with proofs, convergence concepts, and Python simulations."
disableNunjucks: true
series_order: 5
translationKey: "probability-statistics-5"
---

If you had to choose just two theorems from all of probability theory, you'd choose these: the Law of Large Numbers (LLN) and the Central Limit Theorem (CLT). Together, they answer two fundamental questions. The LLN says: "Yes, your sample average will converge to the true mean." The CLT says: "And here's exactly what the fluctuations look like." Without these theorems, there's no justification for opinion polls, no reason to trust clinical trials, and no explanation for why stochastic gradient descent converges.

This article develops both theorems carefully, starting with the different notions of convergence that make the statements precise.

## Modes of Convergence

Before we can say "converges," we need to say what kind of convergence we mean. There are four main types, listed from weakest to strongest.

![Convergence modes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/05-convergence-modes.png)


### Convergence in Distribution

$X_n \xrightarrow{d} X$ if $F_{X_n}(x) \to F_X(x)$ for every $x$ at which $F_X$ is continuous.

This is the weakest form. We're only saying that the CDFs converge pointwise. The random variables $X_n$ don't even need to be defined on the same probability space as $X$.

### Convergence in Probability

$X_n \xrightarrow{P} X$ if for every $\varepsilon > 0$:

$$\lim_{n \to \infty} P(|X_n - X| > \varepsilon) = 0.$$

Stronger than convergence in distribution. Here the probability of $X_n$ being far from $X$ vanishes, but on any specific trial, $X_n$ might still differ from $X$.

### Convergence in Mean Square (L2)

$X_n \xrightarrow{L^2} X$ if:

$$\lim_{n \to \infty} E[(X_n - X)^2] = 0.$$

This implies convergence in probability (by Markov's inequality applied to $(X_n - X)^2$).

*Proof that L2 implies convergence in probability.* For any $\varepsilon > 0$:

$$P(|X_n - X| > \varepsilon) = P((X_n - X)^2 > \varepsilon^2) \leq \frac{E[(X_n - X)^2]}{\varepsilon^2} \to 0. \quad \blacksquare$$

### Almost Sure Convergence

$X_n \xrightarrow{a.s.} X$ if:

$$P\left(\lim_{n \to \infty} X_n = X\right) = 1.$$

The strongest form: the sequence converges for "almost every" outcome $\omega$.

### The Hierarchy

$$\text{a.s.} \implies \text{in probability} \implies \text{in distribution}$$

$$\text{L}^2 \implies \text{in probability} \implies \text{in distribution}$$

Almost sure and $L^2$ convergence are not directly comparable — neither implies the other in general.

## The Weak Law of Large Numbers

**Theorem (WLLN).** Let $X_1, X_2, \ldots$ be i.i.d. random variables with $E[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2 < \infty$. Define the sample mean:

![LLN convergence animation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/gifs/probstat-05-lln-convergence.gif)


![LLN simulation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/05-lln-simulation.png)


$$\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i.$$

Then $\bar{X}_n \xrightarrow{P} \mu$. That is, for every $\varepsilon > 0$:

$$\lim_{n \to \infty} P(|\bar{X}_n - \mu| > \varepsilon) = 0.$$

### Proof via Chebyshev

This is one of the most elegant proofs in probability — short, clean, and illuminating.

**Step 1.** Compute the mean and variance of $\bar{X}_n$.

$$E[\bar{X}_n] = \frac{1}{n} \sum_{i=1}^n E[X_i] = \frac{n\mu}{n} = \mu.$$

$$\text{Var}(\bar{X}_n) = \frac{1}{n^2} \sum_{i=1}^n \text{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}.$$

(We used independence to get $\text{Var}(\sum X_i) = \sum \text{Var}(X_i)$.)

**Step 2.** Apply Chebyshev's inequality to $\bar{X}_n$:

$$P(|\bar{X}_n - \mu| > \varepsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\varepsilon^2} = \frac{\sigma^2}{n\varepsilon^2}.$$

**Step 3.** Take $n \to \infty$:

$$\lim_{n \to \infty} P(|\bar{X}_n - \mu| > \varepsilon) \leq \lim_{n \to \infty} \frac{\sigma^2}{n\varepsilon^2} = 0. \quad \blacksquare$$

The proof also gives a convergence rate: the probability of being more than $\varepsilon$ away from $\mu$ is at most $O(1/n)$. Not the tightest bound (exponential concentration is often possible), but universally applicable.

### What the WLLN Means

With enough data, the sample mean is close to the true mean with high probability. This justifies:
- **Polling:** Ask enough people and the sample proportion approximates the population proportion.
- **Monte Carlo:** Average enough random samples and you approximate the integral.
- **Machine learning:** Average enough stochastic gradients and you approximate the true gradient.

## The Strong Law of Large Numbers


![Central limit theorem many random streams merging into a bel](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/05-central-limit-theorem-many-random-streams-merging-into-a-bel.jpg)

**Theorem (SLLN).** Under the same conditions as the WLLN (and even weaker ones — finite mean suffices):

$$\bar{X}_n \xrightarrow{a.s.} \mu.$$

That is, $P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1$.

The SLLN is strictly stronger than the WLLN: it says not just that the probability of deviating vanishes, but that along almost every sample path, the sequence of averages actually converges to $\mu$ in the ordinary calculus sense. The proof is considerably harder (typically using the Borel-Cantelli lemma or truncation arguments), so we state it without proof.

**Key difference:** The WLLN says "for any tolerance, most experiments succeed." The SLLN says "in a single infinite experiment, convergence happens."

## The Central Limit Theorem

The LLN says the sample mean converges to $\mu$. The CLT tells us **how** — by characterizing the fluctuations around $\mu$.

![CLT convergence animation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/gifs/probstat-05-clt-convergence.gif)


![CLT convergence](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/05-clt-convergence.png)


**Theorem (CLT).** Let $X_1, X_2, \ldots$ be i.i.d. with $E[X_i] = \mu$ and $\text{Var}(X_i) = \sigma^2 \in (0, \infty)$. Then:

$$\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1).$$

Equivalently, if $S_n = X_1 + \cdots + X_n$:

$$\frac{S_n - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1).$$

In words: no matter what distribution the $X_i$ come from (as long as they have finite mean and variance), the standardized sum converges in distribution to a standard Normal.

This is why the Gaussian distribution appears everywhere. Heights are the sum of many genetic and environmental factors. Measurement errors are the sum of many small disturbances. Stock returns are (loosely) the sum of many small trades. The CLT says: sums of many small independent effects look Normal.

### Proof Sketch via MGFs

We prove a slightly weaker version using moment-generating functions (requiring that the MGF exists in a neighborhood of 0).

**Step 1.** Let $Z_i = (X_i - \mu)/\sigma$ be standardized, so $E[Z_i] = 0$ and $\text{Var}(Z_i) = 1$. Define:

$$W_n = \frac{S_n - n\mu}{\sigma\sqrt{n}} = \frac{1}{\sqrt{n}} \sum_{i=1}^n Z_i.$$

**Step 2.** Compute the MGF of $W_n$. Since the $Z_i$ are i.i.d.:

$$M_{W_n}(t) = E\left[e^{tW_n}\right] = E\left[\prod_{i=1}^n e^{tZ_i/\sqrt{n}}\right] = \left[M_Z\left(\frac{t}{\sqrt{n}}\right)\right]^n.$$

**Step 3.** Expand $M_Z(s)$ around $s = 0$ using a Taylor series. Since $E[Z] = 0$ and $E[Z^2] = 1$:

$$M_Z(s) = 1 + sE[Z] + \frac{s^2}{2}E[Z^2] + O(s^3) = 1 + \frac{s^2}{2} + O(s^3).$$

Substituting $s = t/\sqrt{n}$:

$$M_Z\left(\frac{t}{\sqrt{n}}\right) = 1 + \frac{t^2}{2n} + O\left(\frac{t^3}{n^{3/2}}\right).$$

**Step 4.** Take the $n$-th power:

$$M_{W_n}(t) = \left[1 + \frac{t^2}{2n} + O(n^{-3/2})\right]^n \to e^{t^2/2} \quad \text{as } n \to \infty$$

using the limit $(1 + a/n)^n \to e^a$.

**Step 5.** The function $e^{t^2/2}$ is the MGF of $\mathcal{N}(0, 1)$. By the uniqueness theorem for MGFs:

$$W_n \xrightarrow{d} \mathcal{N}(0, 1). \quad \blacksquare$$

## Normal Approximation to the Binomial

A classic application. If $X \sim \text{Binomial}(n, p)$, then $X = \sum_{i=1}^n X_i$ where $X_i \sim \text{Bernoulli}(p)$ i.i.d. By the CLT:

![Normal approximation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/05-normal-approximation.png)


$$\frac{X - np}{\sqrt{np(1-p)}} \approx \mathcal{N}(0, 1) \quad \text{for large } n.$$

### Continuity Correction

Since $X$ is discrete but the Normal is continuous, we apply a **continuity correction** for better accuracy:

$$P(X \leq k) \approx \Phi\left(\frac{k + 0.5 - np}{\sqrt{np(1-p)}}\right)$$

where $\Phi$ is the standard normal CDF.

**Example.** A fair coin is flipped 100 times. What is the probability of getting 60 or more heads?

Exact: $P(X \geq 60)$ where $X \sim \text{Binomial}(100, 0.5)$.

Normal approximation with continuity correction:

$$P(X \geq 60) = P(X \geq 59.5) \approx 1 - \Phi\left(\frac{59.5 - 50}{5}\right) = 1 - \Phi(1.9) \approx 1 - 0.9713 = 0.0287.$$

The exact answer (computed via scipy) is $0.0284$. The approximation is excellent.

## The Berry-Esseen Theorem

How fast does the CLT convergence happen? The Berry-Esseen theorem provides a bound.

![Berry-Esseen bound](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/05-berry-esseen.png)


**Theorem (Berry-Esseen).** Under the CLT conditions, if $E[|Z_i|^3] = \rho < \infty$, then:

$$\sup_x \left|P(W_n \leq x) - \Phi(x)\right| \leq \frac{C \rho}{\sqrt{n}}$$

where $C$ is an absolute constant (the best known value is $C \leq 0.4748$).

This tells us convergence is $O(1/\sqrt{n})$ — double the sample size, and the maximum CDF error drops by about $\sqrt{2}$.

## CLT for Sums vs Averages

The CLT applies to both sums and averages, but with different normalizations:

**For sums:** $S_n = \sum X_i$ has mean $n\mu$ and standard deviation $\sigma\sqrt{n}$, growing with $n$:

$$\frac{S_n - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1).$$

**For averages:** $\bar{X}_n = S_n/n$ has mean $\mu$ and standard deviation $\sigma/\sqrt{n}$, shrinking with $n$:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1).$$

Equivalently: $\bar{X}_n \approx \mathcal{N}(\mu, \sigma^2/n)$ for large $n$.

The $\sqrt{n}$ in the denominator is the fundamental scaling: to halve the standard error, you need four times the data. This is the "diminishing returns" of sampling.

## When the CLT Fails

The CLT requires **finite variance**. Without it, the theorem can fail spectacularly.

### Heavy Tails: The Cauchy Distribution

The $\text{Cauchy}(0,1)$ distribution has PDF $f(x) = \frac{1}{\pi(1+x^2)}$. It has no mean and no variance — the integrals diverge.

For i.i.d. Cauchy random variables, the sample mean $\bar{X}_n$ has the **same** distribution as any single $X_i$: $\bar{X}_n \sim \text{Cauchy}(0, 1)$. Averaging does not help at all. No convergence, no CLT.

**How to detect heavy tails in practice:** If your data has extreme outliers, very high kurtosis ($\gamma_2 \gg 3$), or a sample mean that doesn't stabilize as you collect more data, you may be dealing with a heavy-tailed distribution. In such cases:

- Use the **median** instead of the mean (more robust)
- Use **trimmed means** (discard extreme observations)
- Apply **Winsorization** (cap extreme values)
- Consider distributions designed for heavy tails (t-distribution, Pareto, stable distributions)

The Cauchy example is extreme, but many real-world quantities (financial returns, insurance losses, social network degree distributions) have tails heavy enough to make the CLT converge very slowly or require very large samples.

### Lack of Independence

The CLT also requires independence (or at most weak dependence). For strongly dependent variables — like $X_1 = X_2 = \cdots = X_n$ — the sample mean doesn't concentrate:

$$\bar{X}_n = X_1, \quad \text{Var}(\bar{X}_n) = \text{Var}(X_1)$$

and no convergence to a point occurs.

### CLT Extensions for Dependent Data

While the classical CLT requires i.i.d. data, several extensions handle dependent observations:

**Martingale CLT.** If $\{S_n\}$ is a martingale with appropriate moment conditions, the scaled martingale converges to a Normal distribution. This is used in sequential analysis and financial mathematics.

**CLT for mixing sequences.** If the dependence between $X_i$ and $X_j$ decays sufficiently fast as $|i - j| \to \infty$ (e.g., $\alpha$-mixing or $\phi$-mixing conditions), a CLT still holds with appropriate variance scaling. This applies to many time series models.

**Lindeberg-Feller CLT.** The most general version for independent (but not necessarily identically distributed) variables. If the Lindeberg condition holds — informally, no single variable dominates the sum — then the standardized sum converges to $\mathcal{N}(0,1)$.

The key takeaway: the CLT is robust. You need independence to decay over time and no single term to dominate, but exact i.i.d. is not required.

## Glivenko-Cantelli Theorem: The LLN for CDFs

The LLN says $\bar{X}_n \to \mu$. The **Glivenko-Cantelli theorem** is a stronger result about the entire distribution function.

Define the **empirical CDF**:

$$\hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}_{X_i \leq x}.$$

**Theorem (Glivenko-Cantelli).** If $X_1, X_2, \ldots$ are i.i.d. with CDF $F$, then:

$$\sup_x |\hat{F}_n(x) - F(x)| \xrightarrow{a.s.} 0.$$

The empirical CDF converges uniformly to the true CDF. This is why histograms (empirical distributions) are reliable approximations of the underlying distribution with enough data. The Kolmogorov-Smirnov test is based on this result: it tests whether data follow a specified distribution by measuring the maximum gap between the empirical and theoretical CDFs.

The **Dvoretzky-Kiefer-Wolfowitz (DKW) inequality** gives a non-asymptotic bound:

$$P\left(\sup_x |\hat{F}_n(x) - F(x)| > \varepsilon\right) \leq 2e^{-2n\varepsilon^2}.$$

This is a Hoeffding-type exponential concentration inequality for the entire CDF, not just the mean. It gives confidence bands around the empirical CDF: with probability at least $1 - \alpha$, the true CDF lies within $\pm\sqrt{\frac{\ln(2/\alpha)}{2n}}$ of the empirical CDF everywhere.

## Python: Simulating CLT Convergence

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# Source distributions
distributions = {
    'Uniform(0,1)': lambda size: np.random.uniform(0, 1, size),
    'Exponential(1)': lambda size: np.random.exponential(1, size),
    'Bernoulli(0.3)': lambda size: np.random.binomial(1, 0.3, size),
    'Poisson(3)': lambda size: np.random.poisson(3, size),
}

sample_sizes = [1, 2, 5, 30]
n_simulations = 10000

fig, axes = plt.subplots(len(distributions), len(sample_sizes),
                         figsize=(16, 12))

for row, (dist_name, sampler) in enumerate(distributions.items()):
    for col, n in enumerate(sample_sizes):
        ax = axes[row, col]

        # Generate n_simulations sample means, each from n observations
        sample_means = np.array([
            sampler(n).mean() for _ in range(n_simulations)
        ])

        # Standardize
        mu = sample_means.mean()
        sigma = sample_means.std()
        standardized = (sample_means - mu) / sigma if sigma > 0 else sample_means

        # Histogram
        ax.hist(standardized, bins=50, density=True, alpha=0.7,
                color='steelblue', edgecolor='white', linewidth=0.5)

        # Overlay standard normal
        x = np.linspace(-4, 4, 200)
        ax.plot(x, stats.norm.pdf(x), 'r-', linewidth=2)

        ax.set_xlim(-4, 4)
        ax.set_ylim(0, 0.6)

        if row == 0:
            ax.set_title(f'n = {n}', fontsize=13)
        if col == 0:
            ax.set_ylabel(dist_name, fontsize=11)
        if row == len(distributions) - 1:
            ax.set_xlabel('Standardized mean')

plt.suptitle('Central Limit Theorem: Standardized Sample Means vs N(0,1)',
             fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('clt_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

This simulation is the CLT in action. Each row is a different source distribution — some skewed (Exponential), some discrete (Bernoulli), some symmetric (Uniform). Each column increases the sample size $n$. The red curve is the standard Normal $\mathcal{N}(0,1)$.

At $n = 1$, the histogram reflects the raw distribution shape. By $n = 5$, the distributions are already becoming bell-shaped. At $n = 30$, even the highly skewed Exponential distribution produces sample means that are nearly indistinguishable from a Gaussian. This is the CLT's universality: the source distribution doesn't matter (as long as it has finite variance).

## Delta Method: CLT for Transformed Quantities


![Law of large numbers coin flips converging to fifty percent](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/05-law-of-large-numbers-coin-flips-converging-to-fifty-percent.jpg)

The CLT tells us that $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$. But what if we care about a function of the mean, like $g(\bar{X}_n)$? The **delta method** extends the CLT.

**Theorem (Delta Method).** If $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$ and $g$ is differentiable at $\mu$ with $g'(\mu) \neq 0$, then:

$$\sqrt{n}(g(\bar{X}_n) - g(\mu)) \xrightarrow{d} \mathcal{N}(0, [g'(\mu)]^2 \sigma^2).$$

*Proof.* By a first-order Taylor expansion around $\mu$:

$$g(\bar{X}_n) \approx g(\mu) + g'(\mu)(\bar{X}_n - \mu).$$

Therefore:

$$\sqrt{n}(g(\bar{X}_n) - g(\mu)) \approx g'(\mu) \cdot \sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} g'(\mu) \cdot \mathcal{N}(0, \sigma^2) = \mathcal{N}(0, [g'(\mu)]^2\sigma^2). \quad \blacksquare$$

**Example.** Estimate $\theta = \mu^2$. By the delta method with $g(x) = x^2$, $g'(x) = 2x$:

$$\sqrt{n}(\bar{X}_n^2 - \mu^2) \xrightarrow{d} \mathcal{N}(0, 4\mu^2\sigma^2).$$

So $\text{Var}(\bar{X}_n^2) \approx 4\mu^2\sigma^2/n$.

**Example.** Estimate $\theta = \ln \mu$ where $\mu > 0$. With $g(x) = \ln x$, $g'(x) = 1/x$:

$$\sqrt{n}(\ln \bar{X}_n - \ln \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2/\mu^2).$$

The delta method is essential for constructing confidence intervals for transformed parameters — variance-stabilizing transformations, odds ratios, log-risks, and many other quantities in applied statistics.

## Concentration Inequalities Beyond Chebyshev

Chebyshev's inequality is universal but loose. For specific distribution families, much tighter bounds exist.

### Hoeffding's Inequality

**Theorem (Hoeffding).** If $X_1, \ldots, X_n$ are independent with $a_i \leq X_i \leq b_i$ almost surely, then:

$$P\left(\bar{X}_n - E[\bar{X}_n] \geq t\right) \leq \exp\left(-\frac{2n^2t^2}{\sum_{i=1}^n(b_i - a_i)^2}\right).$$

For identically distributed bounded variables ($a_i = a$, $b_i = b$):

$$P\left(|\bar{X}_n - \mu| \geq t\right) \leq 2\exp\left(-\frac{2nt^2}{(b-a)^2}\right).$$

This is **exponentially** small in $n$, much tighter than Chebyshev's $O(1/n)$.

**Example.** Flip a fair coin $n = 100$ times. What's the probability of observing 60% or more heads?

Chebyshev: $P(|\bar{X} - 0.5| \geq 0.1) \leq \frac{0.25}{100 \times 0.01} = 0.25$.

Hoeffding: $P(|\bar{X} - 0.5| \geq 0.1) \leq 2\exp(-2 \times 100 \times 0.01/1) = 2e^{-2} \approx 0.27$.

Actually for this case Hoeffding and Chebyshev give similar bounds. But as $t$ grows or for sub-Gaussian variables, Hoeffding becomes much tighter.

### Chernoff Bound

For any $t > 0$ and any random variable $X$:

$$P(X \geq a) = P(e^{tX} \geq e^{ta}) \leq \frac{E[e^{tX}]}{e^{ta}} = \frac{M_X(t)}{e^{ta}}.$$

Optimizing over $t$ gives the tightest bound. This is the **Chernoff bound**, and it's how Hoeffding's inequality is derived.

## The Continuous Mapping Theorem

**Theorem.** If $X_n \xrightarrow{d} X$ and $g$ is continuous, then $g(X_n) \xrightarrow{d} g(X)$.

This simple theorem is surprisingly powerful. Combined with the CLT, it lets us derive the asymptotic distributions of test statistics.

**Example.** If $Z_n \xrightarrow{d} \mathcal{N}(0,1)$, then $Z_n^2 \xrightarrow{d} \chi^2(1)$, since $g(x) = x^2$ is continuous.

## Slutsky's Theorem

**Theorem (Slutsky).** If $X_n \xrightarrow{d} X$ and $Y_n \xrightarrow{P} c$ (a constant), then:
- $X_n + Y_n \xrightarrow{d} X + c$
- $X_n Y_n \xrightarrow{d} cX$
- $X_n / Y_n \xrightarrow{d} X/c$ (if $c \neq 0$)

**Application to the t-statistic.** Under $H_0: \mu = \mu_0$:

$$T_n = \frac{\bar{X}_n - \mu_0}{S_n/\sqrt{n}} = \frac{\sqrt{n}(\bar{X}_n - \mu_0)/\sigma}{S_n/\sigma}.$$

The numerator converges in distribution to $\mathcal{N}(0, 1)$ (CLT). The denominator $S_n/\sigma \xrightarrow{P} 1$ (LLN applied to the sample variance). By Slutsky's theorem, $T_n \xrightarrow{d} \mathcal{N}(0, 1)$.

This justifies using the Z-test for large samples even when $\sigma$ is unknown — the t-distribution converges to the Normal.

## Why Machine Learning Works: The CLT Connection

Stochastic gradient descent (SGD) estimates the true gradient $\nabla L(\theta)$ by averaging over a mini-batch of $B$ samples:

$$\hat{g} = \frac{1}{B} \sum_{i=1}^B \nabla \ell(\theta; x_i).$$

By the LLN, $\hat{g} \to \nabla L(\theta)$ as $B \to \infty$. By the CLT:

$$\hat{g} \approx \mathcal{N}\left(\nabla L(\theta), \frac{\Sigma}{B}\right)$$

where $\Sigma$ is the covariance of individual gradients.

This Gaussian approximation has practical consequences:
- **Noise decreases as** $O(1/\sqrt{B})$ — doubling the batch size reduces noise by only $\sqrt{2}$.
- **Gradient noise acts as regularization** — the CLT-predicted Gaussian noise helps SGD escape sharp minima and find flatter, more generalizable solutions.
- **Learning rate and batch size are coupled** — the "linear scaling rule" ($\text{lr} \propto B$) follows from the CLT scaling.

The CLT doesn't just explain why sample averages work — it explains why noisy optimization algorithms converge and generalize.

## Python: Demonstrating the LLN Convergence

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# LLN convergence for different distributions
distributions = {
    'Exponential(1) [$\\mu=1$]': (np.random.exponential, {'scale': 1.0}, 1.0),
    'Bernoulli(0.7) [$\\mu=0.7$]': (np.random.binomial, {'n': 1, 'p': 0.7}, 0.7),
    'Poisson(3) [$\\mu=3$]': (np.random.poisson, {'lam': 3}, 3.0),
    'Uniform(0,1) [$\\mu=0.5$]': (np.random.uniform, {'low': 0, 'high': 1}, 0.5),
}

for idx, (name, (sampler, params, mu)) in enumerate(distributions.items()):
    ax = axes[idx // 2, idx % 2]
    n_max = 2000

    # Multiple sample paths
    for trial in range(10):
        data = sampler(size=n_max, **params)
        running_means = np.cumsum(data) / np.arange(1, n_max + 1)
        ax.plot(running_means, alpha=0.4, linewidth=0.8)

    ax.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'$\\mu = {mu}$')
    ax.set_xlabel('n (number of samples)')
    ax.set_ylabel('Running mean $\\bar{X}_n$')
    ax.set_title(name, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.suptitle('Law of Large Numbers: Running Averages Converge to True Mean',
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('lln_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

Each panel shows 10 independent sample paths of the running average $\bar{X}_n$. Early on, the paths are noisy and far from the true mean (red dashed line). As $n$ grows, all paths converge — this is the LLN in action. The Exponential distribution has the most initial volatility (due to its heavy right tail), but even it settles down by $n \approx 500$.

## Summary
| Theorem | Statement | Requirements | Rate |
|---|---|---|---|
| WLLN | $\bar{X}_n \xrightarrow{P} \mu$ | i.i.d., finite mean and variance | $O(1/n)$ via Chebyshev |
| SLLN | $\bar{X}_n \xrightarrow{a.s.} \mu$ | i.i.d., finite mean | — |
| CLT | $\sqrt{n}(\bar{X}_n - \mu)/\sigma \xrightarrow{d} \mathcal{N}(0,1)$ | i.i.d., finite variance | $O(1/\sqrt{n})$ via Berry-Esseen |
| Delta method | $\sqrt{n}(g(\bar{X}_n) - g(\mu)) \xrightarrow{d} \mathcal{N}(0, [g'(\mu)]^2\sigma^2)$ | CLT + differentiable $g$ | $O(1/\sqrt{n})$ |
| Hoeffding | $P(\|\bar{X}_n - \mu\| \geq t) \leq 2e^{-2nt^2/(b-a)^2}$ | Independent, bounded | Exponential |

## What's Next

The LLN and CLT tell us that sample averages converge to population parameters. But how do we actually estimate those parameters from data? The next article develops estimation theory — method of moments, maximum likelihood estimation, the bias-variance tradeoff — and connects these ideas to the regularization techniques used throughout machine learning.
