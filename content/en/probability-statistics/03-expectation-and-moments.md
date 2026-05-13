---
title: "Probability and Statistics (3): Expectation, Variance, and the Moment-Generating Trick"
date: 2024-08-21 09:00:00
tags:
  - Probability
  - Statistics
  - Expectation
  - Moment Generating Functions
categories: Probability and Statistics
series: probability-statistics
lang: en
mathjax: true
description: "From expectation and variance through covariance, correlation, and moment-generating functions to Chebyshev's inequality — the complete toolkit for summarizing random variables, with proofs for every result."
disableNunjucks: true
series_order: 3
translationKey: "probability-statistics-3"
---

A probability distribution is a complete description of a random variable — it tells you the probability of every possible outcome. But complete descriptions are unwieldy. When someone asks "how tall are people in this city?", you don't hand them a density function; you say "about 170 cm on average, give or take 10 cm." The average and the spread capture most of what matters in practice.

This article develops the mathematical framework for summarizing distributions. We start with expectation (the center), build up to variance (the spread), and then introduce moment-generating functions — a single formula that encodes every moment of a distribution and, remarkably, uniquely determines the distribution itself.

## Expectation


![Expectation as balance point](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/03-expectation.png)

### Definition

For a **discrete** random variable $X$ with PMF $p(x)$:
$$E[X] = \sum_{x} x \, p(x)$$
where the sum is over all values in the support of $X$, provided the sum converges absolutely.

For a **continuous** random variable $X$ with PDF $f(x)$:
$$E[X] = \int_{-\infty}^{\infty} x \, f(x) \, dx$$
again provided the integral converges absolutely.

The expectation is the "center of mass" of the distribution. If you placed weights of size $p(x)$ at each point $x$ on a number line, $E[X]$ is the balance point.

### Linearity of Expectation

This is arguably the single most useful property in all of probability.

**Theorem.** For any random variables $X$ and $Y$ (not necessarily independent) and constants $a, b, c$:
$$E[aX + bY + c] = aE[X] + bE[Y] + c.$$
*Proof (discrete case).* Let $(X, Y)$ have joint PMF $p(x, y)$.
$$E[aX + bY + c] = \sum_x \sum_y (ax + by + c) \, p(x, y)$$

$$= a \sum_x \sum_y x \, p(x, y) + b \sum_x \sum_y y \, p(x, y) + c \sum_x \sum_y p(x, y)$$

$$= a \sum_x x \sum_y p(x, y) + b \sum_y y \sum_x p(x, y) + c \cdot 1$$

$$= a \sum_x x \, p_X(x) + b \sum_y y \, p_Y(y) + c = aE[X] + bE[Y] + c. \quad \blacksquare$$
Notice: **we never assumed independence.** Linearity holds always. This is what makes it so powerful.

**Example.** Find the expected number of fixed points when shuffling $n$ cards (a fixed point is a card that ends up in its original position).

Let $X_i = 1$ if card $i$ is in position $i$, and $X_i = 0$ otherwise. Then $X = \sum_{i=1}^n X_i$ counts the total fixed points. $E[X_i] = P(\text{card } i \text{ stays}) = 1/n$. By linearity:
$$E[X] = \sum_{i=1}^n E[X_i] = n \cdot \frac{1}{n} = 1.$$
On average, exactly one card stays in place — regardless of $n$. We computed the expectation of a complicated random variable without ever finding its distribution.

**Example.** Expected number of inversions in a random permutation of $\{1, 2, \ldots, n\}$.

An **inversion** is a pair $(i, j)$ with $i < j$ but $\sigma(i) > \sigma(j)$. Let $X_{ij} = 1$ if $(i,j)$ is an inversion. For any pair, $P(X_{ij} = 1) = 1/2$ (by symmetry — the two elements are equally likely to appear in either order). There are $\binom{n}{2}$ pairs, so:
$$E[\text{inversions}] = \binom{n}{2} \cdot \frac{1}{2} = \frac{n(n-1)}{4}.$$
Again, linearity gives us the answer without analyzing the complex dependencies between inversions. Note that the $X_{ij}$ are **not** independent (swapping two elements affects multiple pairs), but linearity doesn't care.

## LOTUS: The Law of the Unconscious Statistician

To compute $E[g(X)]$ for some function $g$, you might think you need the distribution of $Y = g(X)$. You don't.

**Theorem (LOTUS).** If $X$ is discrete with PMF $p(x)$:
$$E[g(X)] = \sum_x g(x) \, p(x).$$
If $X$ is continuous with PDF $f(x)$:
$$E[g(X)] = \int_{-\infty}^{\infty} g(x) \, f(x) \, dx.$$
The name "unconscious statistician" is tongue-in-cheek: the formula is so natural that people use it without realizing they've invoked a theorem.

**Example.** Let $X \sim \text{Uniform}(0, 1)$. Find $E[X^2]$.
$$E[X^2] = \int_0^1 x^2 \cdot 1 \, dx = \frac{x^3}{3}\bigg|_0^1 = \frac{1}{3}.$$
We used LOTUS with $g(x) = x^2$ and $f(x) = 1$ on $[0,1]$. No need to derive the distribution of $X^2$.

## Variance


![Variance visualization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/03-variance.png)


![Covariance correlation dance partners moving together or ind](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/03-covariance-correlation-dance-partners-moving-together-or-ind.jpg)

### Definition

The **variance** of $X$ measures the average squared deviation from the mean:
$$\text{Var}(X) = E\left[(X - E[X])^2\right].$$
The **standard deviation** is $\text{SD}(X) = \sigma_X = \sqrt{\text{Var}(X)}$, which has the same units as $X$.

### Computational Formula

Expanding the square:
$$\text{Var}(X) = E\left[X^2 - 2XE[X] + (E[X])^2\right] = E[X^2] - 2E[X]E[X] + (E[X])^2$$

$$= E[X^2] - (E[X])^2.$$
This is the formula you'll use 90% of the time:
$$\boxed{\text{Var}(X) = E[X^2] - (E[X])^2}$$
**Example.** $X \sim \text{Uniform}(0,1)$: $E[X] = 1/2$, $E[X^2] = 1/3$ (computed above). So $\text{Var}(X) = 1/3 - 1/4 = 1/12$. This matches the formula $(b-a)^2/12 = 1/12$.

### Scaling and Shifting

**Theorem.** For constants $a$ and $b$:
$$\text{Var}(aX + b) = a^2 \text{Var}(X).$$
*Proof.*
$$\text{Var}(aX + b) = E[(aX + b)^2] - (E[aX + b])^2$$

$$= E[a^2X^2 + 2abX + b^2] - (aE[X] + b)^2$$

$$= a^2E[X^2] + 2abE[X] + b^2 - a^2(E[X])^2 - 2abE[X] - b^2$$

$$= a^2(E[X^2] - (E[X])^2) = a^2\text{Var}(X). \quad \blacksquare$$
Adding a constant shifts the distribution but doesn't change its spread. Multiplying by $a$ scales the spread by $|a|$ (and the variance by $a^2$).

### Variance of a Sum

If $X$ and $Y$ are **independent**:
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y).$$
More generally (without independence):
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$
which brings us to our next topic.

## Covariance


![Covariance scatter plots](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/03-covariance-scatter.png)

### Definition

The **covariance** of $X$ and $Y$ measures how they vary together:
$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y].$$
*Proof of the computational formula:*
$$\text{Cov}(X,Y) = E[XY - XE[Y] - YE[X] + E[X]E[Y]]$$

$$= E[XY] - E[X]E[Y] - E[Y]E[X] + E[X]E[Y] = E[XY] - E[X]E[Y]. \quad \blacksquare$$
### Properties

1. $\text{Cov}(X, X) = \text{Var}(X)$
2. $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ (symmetry)
3. $\text{Cov}(aX + b, Y) = a \, \text{Cov}(X, Y)$ (linearity in each argument)
4. $\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)$ (bilinearity)
5. If $X$ and $Y$ are independent, $\text{Cov}(X, Y) = 0$.

**Warning:** The converse of Property 5 is false. $\text{Cov}(X, Y) = 0$ does **not** imply independence.

**Counterexample.** Let $X \sim \text{Uniform}(-1, 1)$ and $Y = X^2$. Then $Y$ is completely determined by $X$ (maximally dependent), yet:
$$E[XY] = E[X^3] = \int_{-1}^{1} x^3 \cdot \frac{1}{2} dx = 0$$
because $x^3$ is an odd function on a symmetric interval. So $\text{Cov}(X, Y) = E[XY] - E[X]E[Y] = 0 - 0 \cdot E[Y] = 0$.

Covariance detects linear relationships. It misses nonlinear ones.

## Correlation

The **Pearson correlation coefficient** normalizes covariance to lie in $[-1, 1]$:
$$\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}}.$$
### Properties

- $-1 \leq \rho \leq 1$ (proved via Cauchy-Schwarz inequality)
- $\rho = 1$ if and only if $Y = aX + b$ with $a > 0$ (perfect positive linear relationship)
- $\rho = -1$ if and only if $Y = aX + b$ with $a < 0$ (perfect negative linear relationship)
- $\rho = 0$ means "uncorrelated" but **not** necessarily independent

*Proof that $|\rho| \leq 1$.* By the Cauchy-Schwarz inequality for expectations:
$$(E[UV])^2 \leq E[U^2] E[V^2].$$
Let $U = X - E[X]$ and $V = Y - E[Y]$:
$$(\text{Cov}(X,Y))^2 \leq \text{Var}(X) \text{Var}(Y)$$

$$\rho^2 = \frac{(\text{Cov}(X,Y))^2}{\text{Var}(X)\text{Var}(Y)} \leq 1. \quad \blacksquare$$
### What Correlation Does and Doesn't Mean

Correlation measures the strength and direction of the **linear** relationship between two variables. It does not capture:

- Nonlinear relationships ($X$ and $X^2$ can have $\rho = 0$)
- Causation (a third variable may drive both)
- The slope of the relationship (that's the regression coefficient, not $\rho$)

## Higher Moments

The $k$-th **moment** of $X$ about the origin is $E[X^k]$. The $k$-th **central moment** is $E[(X - \mu)^k]$.

![Moment generating function](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/03-mgf.png)


- 1st moment: $E[X] = \mu$ (location)
- 2nd central moment: $\text{Var}(X) = \sigma^2$ (spread)
- 3rd central moment (normalized): **Skewness**
$$\gamma_1 = \frac{E[(X - \mu)^3]}{\sigma^3}$$
Skewness measures asymmetry. $\gamma_1 > 0$: right tail is longer (right-skewed). $\gamma_1 < 0$: left tail is longer (left-skewed). $\gamma_1 = 0$: symmetric (like the Normal).

**Example.** The Exponential($\lambda$) distribution has $\gamma_1 = 2$ — always positively skewed. Income distributions typically have $\gamma_1 > 0$ (long right tail of high earners). Returns on financial assets often have $\gamma_1 < 0$ (the "crash" tail is heavier than the "boom" tail).

- 4th central moment (normalized): **Kurtosis**
$$\gamma_2 = \frac{E[(X - \mu)^4]}{\sigma^4}$$
The Normal distribution has $\gamma_2 = 3$. **Excess kurtosis** is $\gamma_2 - 3$, measuring how much heavier the tails are compared to a Normal. Positive excess kurtosis means heavier tails (more extreme events than a Gaussian predicts).

**Example.** The Student's t-distribution with $\nu$ degrees of freedom has excess kurtosis $6/(\nu - 4)$ (for $\nu > 4$). As $\nu \to \infty$, it approaches 0 (Normal). For small $\nu$, the excess kurtosis is large, reflecting heavy tails. This is why the t-distribution appears in robust statistics.

**Example.** The Uniform distribution has $\gamma_2 = 1.8$, so its excess kurtosis is $-1.2$ (lighter tails than Normal). It's as "platykurtic" as a continuous distribution gets on a bounded interval.

## Moment-Generating Functions

### Definition

The **moment-generating function** (MGF) of $X$ is
$$M_X(t) = E[e^{tX}]$$
defined for all $t$ in some open interval containing 0.

The name comes from the Taylor expansion $e^{tX} = 1 + tX + \frac{t^2 X^2}{2!} + \cdots$, so:
$$M_X(t) = 1 + tE[X] + \frac{t^2}{2!}E[X^2] + \frac{t^3}{3!}E[X^3] + \cdots$$
The $k$-th moment is extracted by differentiation:
$$E[X^k] = M_X^{(k)}(0) = \frac{d^k}{dt^k} M_X(t) \bigg|_{t=0}.$$
### Uniqueness Theorem

**Theorem.** If $M_X(t) = M_Y(t)$ for all $t$ in an open interval containing 0, then $X$ and $Y$ have the same distribution.

This is why MGFs are powerful: they are **fingerprints** of distributions. Identify the MGF, and you've identified the distribution.

### MGF of the Poisson Distribution

Let $X \sim \text{Poisson}(\lambda)$:
$$M_X(t) = E[e^{tX}] = \sum_{k=0}^{\infty} e^{tk} \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=0}^{\infty} \frac{(\lambda e^t)^k}{k!} = e^{-\lambda} \cdot e^{\lambda e^t} = e^{\lambda(e^t - 1)}.$$
*Verification:* $M_X'(t) = \lambda e^t \cdot e^{\lambda(e^t - 1)}$, so $M_X'(0) = \lambda \cdot 1 = \lambda = E[X]$. $\checkmark$

$M_X''(t) = (\lambda e^t)^2 e^{\lambda(e^t-1)} + \lambda e^t e^{\lambda(e^t-1)}$, so $M_X''(0) = \lambda^2 + \lambda = E[X^2]$. Then $\text{Var}(X) = \lambda^2 + \lambda - \lambda^2 = \lambda$. $\checkmark$

### MGF of the Normal Distribution

Let $Z \sim \mathcal{N}(0, 1)$:
$$M_Z(t) = E[e^{tZ}] = \int_{-\infty}^{\infty} e^{tz} \frac{1}{\sqrt{2\pi}} e^{-z^2/2} dz = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} e^{-(z^2 - 2tz)/2} dz.$$
Complete the square: $z^2 - 2tz = (z - t)^2 - t^2$.
$$M_Z(t) = e^{t^2/2} \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} e^{-(z-t)^2/2} dz = e^{t^2/2} \cdot 1 = e^{t^2/2}.$$
For $X \sim \mathcal{N}(\mu, \sigma^2)$, writing $X = \mu + \sigma Z$:
$$M_X(t) = E[e^{t(\mu + \sigma Z)}] = e^{\mu t} M_Z(\sigma t) = e^{\mu t + \sigma^2 t^2/2}.$$
### MGF of the Exponential Distribution

Let $X \sim \text{Exp}(\lambda)$:
$$M_X(t) = \int_0^{\infty} e^{tx} \lambda e^{-\lambda x} dx = \lambda \int_0^{\infty} e^{-(lambda - t)x} dx = \frac{\lambda}{\lambda - t} \quad \text{for } t < \lambda.$$
### Using MGFs to Prove Distribution Results

**Claim:** The sum of independent Poissons is Poisson. If $X \sim \text{Poisson}(\lambda)$ and $Y \sim \text{Poisson}(\mu)$ are independent, then $X + Y \sim \text{Poisson}(\lambda + \mu)$.

*Proof.* $M_{X+Y}(t) = M_X(t) M_Y(t) = e^{\lambda(e^t - 1)} e^{\mu(e^t - 1)} = e^{(\lambda + \mu)(e^t - 1)}$, which is the MGF of $\text{Poisson}(\lambda + \mu)$. By uniqueness, $X + Y \sim \text{Poisson}(\lambda + \mu)$. $\blacksquare$

## Markov's Inequality

**Theorem (Markov).** If $X \geq 0$ and $a > 0$, then
$$P(X \geq a) \leq \frac{E[X]}{a}.$$
*Proof.* Note that $a \cdot \mathbf{1}_{X \geq a} \leq X$ (since if $X \geq a$, we get $a \leq X$; if $X < a$, we get $0 \leq X$, which is true since $X \geq 0$). Taking expectations of both sides:
$$a \cdot P(X \geq a) \leq E[X].$$
Divide by $a$. $\blacksquare$

Markov's inequality is often loose, but it requires almost nothing — just $X \geq 0$ and finite mean.

## Chebyshev's Inequality

**Theorem (Chebyshev).** For any random variable $X$ with finite mean $\mu$ and variance $\sigma^2$, and for any $k > 0$:

![Chebyshev inequality](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/03-chebyshev-bound.png)
$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}.$$
*Proof.* Apply Markov's inequality to the non-negative random variable $(X - \mu)^2$ with $a = k^2 \sigma^2$:
$$P((X - \mu)^2 \geq k^2 \sigma^2) \leq \frac{E[(X - \mu)^2]}{k^2 \sigma^2} = \frac{\sigma^2}{k^2 \sigma^2} = \frac{1}{k^2}. \quad \blacksquare$$
| $k$ | Chebyshev bound $P(\|X - \mu\| \geq k\sigma) \leq$ | Normal exact $P(\|Z\| \geq k)$ |
|---|---|---|
| 1 | 1.000 | 0.317 |
| 2 | 0.250 | 0.046 |
| 3 | 0.111 | 0.003 |
| 4 | 0.063 | 0.00006 |

Chebyshev is **distribution-free** — it holds for any distribution with finite variance. The price for this generality is that the bound is loose (compare the columns above). But "loose but always valid" is often more useful than "tight but only for Gaussians."

## Python: Visualizing Moments and Chebyshev's Bound


![Expectation as center of mass balance point on probability d](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/03-expectation-as-center-of-mass-balance-point-on-probability-d.jpg)

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Expectation as center of mass
ax = axes[0]
x = np.linspace(-4, 6, 300)
for mu, sigma in [(0, 1), (2, 0.7), (-1, 1.5)]:
    y = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, y, linewidth=2, label=f'$\\mu$={mu}, $\\sigma$={sigma}')
    ax.axvline(mu, linestyle=':', alpha=0.5)
ax.set_title('Expectation as Location', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# 2. Variance as spread
ax = axes[1]
x = np.linspace(-6, 6, 300)
for sigma in [0.5, 1, 2]:
    y = stats.norm.pdf(x, 0, sigma)
    ax.plot(x, y, linewidth=2, label=f'$\\sigma$={sigma}')
    ax.axvspan(-sigma, sigma, alpha=0.05)
ax.set_title('Variance as Spread', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

# 3. Chebyshev bound vs actual
ax = axes[2]
ks = np.linspace(1, 5, 100)
chebyshev_bound = 1 / ks**2
normal_exact = 2 * (1 - stats.norm.cdf(ks))
uniform_exact = np.maximum(0, 1 - ks * np.sqrt(3) / 3)  # approximation

ax.plot(ks, chebyshev_bound, 'r-', linewidth=2, label='Chebyshev bound')
ax.plot(ks, normal_exact, 'b--', linewidth=2, label='Normal (exact)')
ax.set_title("Chebyshev's Inequality", fontsize=13)
ax.set_xlabel('k (number of std devs)')
ax.set_ylabel('P(|X - $\\mu$| $\\geq$ k$\\sigma$)')
ax.legend()
ax.set_yscale('log')
ax.set_ylim(1e-4, 1.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('moments_and_chebyshev.png', dpi=150)
plt.show()
```

The right panel shows the gap between Chebyshev's bound and the actual Normal tail probability on a log scale. For the Normal distribution, the true probability drops exponentially in $k^2$, while Chebyshev only guarantees $1/k^2$ decay. The bound is conservative, but it applies to **every** distribution — including ones with much heavier tails than the Gaussian.

## Conditional Expectation

### Definition

The **conditional expectation** of $X$ given $Y = y$ is:
$$E[X | Y = y] = \sum_x x \, p_{X|Y}(x | y) \quad \text{(discrete)}$$

$$E[X | Y = y] = \int_{-\infty}^{\infty} x \, f_{X|Y}(x | y) \, dx \quad \text{(continuous)}$$
The conditional expectation $E[X|Y]$ (viewed as a function of $Y$) is itself a random variable.

### The Tower Property (Law of Iterated Expectation)

**Theorem.** $E[E[X | Y]] = E[X]$.

*Proof (discrete).*
$$E[E[X|Y]] = \sum_y E[X|Y=y] \cdot p_Y(y) = \sum_y \left(\sum_x x \, p_{X|Y}(x|y)\right) p_Y(y)$$

$$= \sum_y \sum_x x \, p_{X,Y}(x,y) = \sum_x x \sum_y p_{X,Y}(x,y) = \sum_x x \, p_X(x) = E[X]. \quad \blacksquare$$
This is extremely useful. To compute $E[X]$, condition on something that simplifies the problem, compute $E[X|Y]$ for each value of $Y$, then average.

**Example.** Roll a die. If the result is $Y$, flip $Y$ coins. Let $X$ be the number of heads. Find $E[X]$.

$X | Y \sim \text{Binomial}(Y, 1/2)$, so $E[X|Y] = Y/2$.
$$E[X] = E[E[X|Y]] = E[Y/2] = E[Y]/2 = 3.5/2 = 1.75.$$
### Conditional Variance
$$\text{Var}(X | Y) = E[X^2 | Y] - (E[X|Y])^2.$$
**Law of Total Variance (Eve's Law):**
$$\text{Var}(X) = E[\text{Var}(X|Y)] + \text{Var}(E[X|Y]).$$
The total variance decomposes into:
- **Within-group variance:** $E[\text{Var}(X|Y)]$ — average variability within each level of $Y$.
- **Between-group variance:** $\text{Var}(E[X|Y])$ — variability of group means.

*Proof.* By the computational formula for variance:
$$\text{Var}(X) = E[X^2] - (E[X])^2.$$
Using the tower property: $E[X^2] = E[E[X^2|Y]]$ and $E[X] = E[E[X|Y]]$.
$$E[E[X^2|Y]] = E[\text{Var}(X|Y) + (E[X|Y])^2] = E[\text{Var}(X|Y)] + E[(E[X|Y])^2].$$

$$\text{Var}(X) = E[\text{Var}(X|Y)] + E[(E[X|Y])^2] - (E[E[X|Y]])^2.$$
The last two terms are $\text{Var}(E[X|Y])$. $\blacksquare$

## Jensen's Inequality

**Theorem (Jensen).** If $g$ is a convex function and $X$ is a random variable with finite $E[X]$:
$$g(E[X]) \leq E[g(X)].$$
If $g$ is concave, the inequality reverses: $g(E[X]) \geq E[g(X)]$.

*Proof sketch.* A convex function lies above its tangent lines. For any tangent at $E[X]$: $g(x) \geq g(E[X]) + g'(E[X])(x - E[X])$. Taking expectations of both sides and noting $E[x - E[X]] = 0$ gives the result. $\blacksquare$

**Applications:**

1. **Variance is non-negative:** Apply Jensen with $g(x) = x^2$ (convex): $E[X^2] \geq (E[X])^2$, so $\text{Var}(X) \geq 0$.

2. **AM-GM inequality:** $E[\ln X] \leq \ln E[X]$ for $X > 0$ (since $\ln$ is concave), which implies the arithmetic mean of positive numbers is at least the geometric mean.

3. **KL divergence is non-negative:** Jensen's inequality applied to the $\ln$ function proves that the Kullback-Leibler divergence $D_{\text{KL}}(P \| Q) \geq 0$ — a foundational result in information theory and machine learning.

## Characteristic Functions (Brief)

Not all distributions have moment-generating functions (the MGF may not exist for all $t$ near 0). The **characteristic function** always exists:
$$\varphi_X(t) = E[e^{itX}] = E[\cos(tX)] + i \, E[\sin(tX)]$$
where $i = \sqrt{-1}$. The characteristic function exists for every distribution (since $|e^{itX}| = 1$, the expectation always converges). It uniquely determines the distribution and shares the MGF's property that $\varphi_{X+Y}(t) = \varphi_X(t) \varphi_Y(t)$ for independent $X$ and $Y$.

The characteristic function is how the CLT is proved in full generality — you show $\varphi_{\bar{X}_n}(t) \to e^{-t^2/2}$, which is the characteristic function of $\mathcal{N}(0,1)$, and apply Levy's continuity theorem.

## Inequalities Beyond Chebyshev

### Chernoff Bound

For any $t > 0$:
$$P(X \geq a) = P(e^{tX} \geq e^{ta}) \leq \frac{E[e^{tX}]}{e^{ta}} = \frac{M_X(t)}{e^{ta}}$$
by Markov's inequality applied to $e^{tX}$. Optimizing over $t$ gives the tightest bound. This is the **Chernoff bound**, and it provides exponentially decaying tail probabilities for distributions with well-behaved MGFs.

**Example.** Standard Normal tail: $P(Z \geq a) \leq e^{-a^2/2}$, derived by choosing $t = a$ in the Chernoff bound with $M_Z(t) = e^{t^2/2}$.

### One-Sided Chebyshev (Cantelli's Inequality)

For a random variable with mean $\mu$ and variance $\sigma^2$:
$$P(X - \mu \geq t) \leq \frac{\sigma^2}{\sigma^2 + t^2}.$$
This is tighter than Chebyshev for one-sided deviations and requires no symmetry assumption.

## Summary
| Quantity | Formula | Notes |
|---|---|---|
| $E[X]$ | $\sum x \, p(x)$ or $\int x \, f(x) \, dx$ | Center of distribution |
| $E[g(X)]$ | $\sum g(x) \, p(x)$ or $\int g(x) \, f(x) \, dx$ | LOTUS |
| $\text{Var}(X)$ | $E[X^2] - (E[X])^2$ | Spread of distribution |
| $\text{Var}(aX + b)$ | $a^2 \text{Var}(X)$ | Shift doesn't affect spread |
| $\text{Cov}(X,Y)$ | $E[XY] - E[X]E[Y]$ | Linear co-variation |
| $\rho(X,Y)$ | $\text{Cov}(X,Y)/(\sigma_X \sigma_Y)$ | Normalized, $\in [-1, 1]$ |
| $M_X(t)$ | $E[e^{tX}]$ | Generates all moments |
| Tower property | $E[E[X\mid Y]] = E[X]$ | Iterated expectation |
| Total variance | $\text{Var}(X) = E[\text{Var}(X\lvert Y)] + \text{Var}(E[X\rvertY])$ | Within + between |
| Jensen | $g(E[X]) \leq E[g(X)]$ for convex $g$ | Fundamental inequality |
| Markov | $P(X \geq a) \leq E[X]/a$ | Requires $X \geq 0$ |
| Chebyshev | $P(\|X-\mu\| \geq k\sigma) \leq 1/k^2$ | Distribution-free |

## What's Next

So far we've worked with one random variable at a time. But real data is multivariate: height and weight are correlated, features in a dataset interact, errors propagate through functions. The next article tackles joint distributions — the mathematics of multiple random variables living together — including marginals, conditionals, transformations, and the bivariate normal.
