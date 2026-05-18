---
title: "Probability and Statistics (4): Joint Distributions, Marginalization, and Independence"
date: 2024-08-23 09:00:00
tags:
  - Probability
  - Statistics
  - Joint Distributions
  - Transformations
categories: Probability and Statistics
series: probability-statistics
lang: en
mathjax: true
description: "Joint PMFs and PDFs, marginal and conditional distributions, the bivariate normal, transformations via the Jacobian method, convolutions, and order statistics — with proofs and contour plot visualizations."
disableNunjucks: true
series_order: 4
series_total: 8
translationKey: "probability-statistics-4"
---

Until now, every distribution we've studied described a single quantity: one die roll, one waiting time, one measurement. But interesting problems involve relationships between variables. Does studying more hours predict a higher exam score? Are stock returns correlated across sectors? How does the sum of two random variables behave?

Answering these questions requires **joint distributions** — the mathematical framework for describing multiple random variables simultaneously. This is where probability theory starts connecting directly to regression, multivariate statistics, and the high-dimensional spaces of machine learning.

---

## Joint Distributions: Discrete Case

![Joint PMF table](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/04-joint-pmf.png)

### Joint PMF

If $X$ and $Y$ are discrete random variables defined on the same probability space, their **joint PMF** is
$$p_{X,Y}(x, y) = P(X = x, Y = y)$$
for all pairs $(x, y)$.

Properties:
1. $p_{X,Y}(x, y) \geq 0$ for all $(x, y)$
2. $\sum_x \sum_y p_{X,Y}(x, y) = 1$

**Example.** Roll two fair dice. Let $X$ be the first die and $Y$ the sum of both dice. The joint PMF is $p_{X,Y}(x, y) = 1/36$ for $x \in \{1,...,6\}$ and $y = x + j$ where $j \in \{1,...,6\}$.

### Marginal Distributions

The **marginal PMF** of $X$ is obtained by summing out $Y$:

![Marginal projections](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/04-marginal-projection.png)
$$p_X(x) = \sum_y p_{X,Y}(x, y).$$
Similarly:
$$p_Y(y) = \sum_x p_{X,Y}(x, y).$$
This is the discrete version of "integrating out" a variable — you collapse the joint distribution down to a single variable by summing over all possible values of the other.

**Key point:** Knowing the marginals does **not** determine the joint distribution. Many different joints can produce the same marginals. The joint contains more information than the marginals combined.

### Conditional Distributions (Discrete)

The **conditional PMF** of $X$ given $Y = y$ is
$$p_{X|Y}(x | y) = \frac{p_{X,Y}(x, y)}{p_Y(y)}, \quad \text{provided } p_Y(y) > 0.$$
This is the natural extension of conditional probability $P(A|B) = P(A \cap B)/P(B)$ to random variables.

For each fixed $y$, $p_{X|Y}(\cdot | y)$ is a valid PMF (non-negative, sums to 1).

## Joint Distributions: Continuous Case

### Joint PDF

For continuous random variables, the **joint PDF** $f_{X,Y}(x, y)$ satisfies
$$P((X, Y) \in A) = \iint_A f_{X,Y}(x, y) \, dx \, dy$$
for any "reasonable" set $A \subseteq \mathbb{R}^2$.

Properties:
1. $f_{X,Y}(x, y) \geq 0$
2. $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dx \, dy = 1$

The joint CDF is:
$$F_{X,Y}(x, y) = P(X \leq x, Y \leq y) = \int_{-\infty}^x \int_{-\infty}^y f_{X,Y}(s, t) \, dt \, ds.$$
### Marginal PDF
$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dy, \qquad f_Y(y) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dx.$$
### Conditional PDF
$$f_{X|Y}(x | y) = \frac{f_{X,Y}(x, y)}{f_Y(y)}, \quad \text{provided } f_Y(y) > 0.$$
**Example.** Let $(X, Y)$ have joint PDF $f(x, y) = 6(1 - y)$ for $0 < x < y < 1$, and $f(x,y) = 0$ otherwise.

*Verification that this is a valid PDF:*
$$\int_0^1 \int_0^y 6(1-y) \, dx \, dy = \int_0^1 6y(1-y) \, dy = 6\left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^1 = 6 \cdot \frac{1}{6} = 1. \quad \checkmark$$
*Find $f_X(x)$:*
$$f_X(x) = \int_x^1 6(1-y) \, dy = 6\left[y - \frac{y^2}{2}\right]_x^1 \cdot (-1) \text{ ... let's compute directly:}$$

$$f_X(x) = \int_x^1 6(1-y) \, dy = 6\left[(y - y^2/2)\right]_x^1 \cdot \text{... }$$

$$= 6\left[\left(1 - \frac{1}{2}\right) - \left(x - \frac{x^2}{2}\right)\right] = 6\left[\frac{1}{2} - x + \frac{x^2}{2}\right] = 3(1-x)^2 \quad \text{for } 0 < x < 1.$$
*Find $f_Y(y)$:*
$$f_Y(y) = \int_0^y 6(1 - y) \, dx = 6y(1 - y) \quad \text{for } 0 < y < 1.$$
*Verification:* $\int_0^1 6y(1-y) \, dy = 6 \left[\frac{y^2}{2} - \frac{y^3}{3}\right]_0^1 = 6 \cdot \frac{1}{6} = 1$. $\checkmark$

*Find $f_{X|Y}(x | y)$:*
$$f_{X|Y}(x | y) = \frac{6(1-y)}{6y(1-y)} = \frac{1}{y} \quad \text{for } 0 < x < y.$$
Given $Y = y$, $X$ is uniformly distributed on $(0, y)$. This makes sense: the conditional density doesn't depend on $x$ (it's flat), and it's supported on $(0, y)$.

## Independence

![Independence vs dependence linked chains vs free floating ev](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/04-independence-vs-dependence-linked-chains-vs-free-floating-ev.jpg)

Random variables $X$ and $Y$ are **independent** if and only if their joint distribution factors:
$$f_{X,Y}(x, y) = f_X(x) \, f_Y(y) \quad \text{for all } (x, y).$$
Equivalently, in the discrete case: $p_{X,Y}(x, y) = p_X(x) \, p_Y(y)$.

**Checking independence:** Factor the joint PDF/PMF. If you can write $f_{X,Y}(x, y) = g(x) h(y)$ for some functions $g$ and $h$ (and the support is a rectangle), then $X$ and $Y$ are independent.

In the example above, $f(x, y) = 6(1-y)$ on a triangular region $\{0 < x < y < 1\}$. The non-rectangular support immediately tells us $X$ and $Y$ are **not** independent: knowing $Y = 0.3$ restricts $X$ to $(0, 0.3)$, so $Y$ constrains $X$.

**A second check for independence.** Even if the support is rectangular, the joint PDF must factor as a product $g(x)h(y)$. For instance, $f(x,y) = 4xy$ on $[0,1]^2$ factors as $(2x)(2y)$, so $X$ and $Y$ are independent with $f_X(x) = 2x$ and $f_Y(y) = 2y$. But $f(x,y) = 2(x+y)$ on $[0,1]^2$ does not factor (there's a cross-term), so $X$ and $Y$ are dependent despite having rectangular support.

## The Bivariate Normal Distribution

![Joint probability distribution as 3d terrain map with margin](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/04-joint-probability-distribution-as-3d-terrain-map-with-margin.jpg)

The most important multivariate distribution. If $(X, Y)$ follows a bivariate normal, the joint PDF is:

![Bivariate normal contour](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/04-bivariate-normal.png)
$$f(x, y) = \frac{1}{2\pi \sigma_X \sigma_Y \sqrt{1 - \rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)} \left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X \sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right)$$
where $\mu_X, \mu_Y$ are the means, $\sigma_X, \sigma_Y$ the standard deviations, and $\rho = \text{Corr}(X, Y)$ is the correlation.

Key properties:
1. Both marginals are normal: $X \sim \mathcal{N}(\mu_X, \sigma_X^2)$ and $Y \sim \mathcal{N}(\mu_Y, \sigma_Y^2)$
2. All conditionals are normal: $X | Y = y \sim \mathcal{N}\left(\mu_X + \rho \frac{\sigma_X}{\sigma_Y}(y - \mu_Y), \sigma_X^2(1 - \rho^2)\right)$
3. $X$ and $Y$ are independent **if and only if** $\rho = 0$

Property 3 is special to the bivariate normal. In general, $\rho = 0$ does **not** imply independence. But for jointly normal variables, it does.

The conditional mean $E[X | Y = y] = \mu_X + \rho \frac{\sigma_X}{\sigma_Y}(y - \mu_Y)$ is a linear function of $y$. This is the **regression line** — one of the earliest connections between probability theory and what we now call machine learning.

The conditional variance $\sigma_X^2(1 - \rho^2)$ does not depend on $y$. This means the "spread" of $X$ around its conditional mean is the same for all values of $Y$ — a property called **homoscedasticity**. The factor $(1 - \rho^2)$ tells us how much knowing $Y$ reduces our uncertainty about $X$:

- $\rho = 0$: no reduction ($\text{Var}(X|Y) = \sigma_X^2$)
- $|\rho| = 0.5$: 25% reduction ($\text{Var}(X|Y) = 0.75\sigma_X^2$)
- $|\rho| = 0.9$: 81% reduction ($\text{Var}(X|Y) = 0.19\sigma_X^2$)
- $|\rho| = 1$: complete reduction ($\text{Var}(X|Y) = 0$, meaning $X$ is determined by $Y$)

The quantity $R^2 = \rho^2$ is the **coefficient of determination** — the proportion of variance in $X$ explained by $Y$. This is the same $R^2$ you compute in linear regression.

## Transformations of Random Variables

Given $X$ with known PDF $f_X(x)$, find the PDF of $Y = g(X)$.

### CDF Method

The most general approach:

1. Find $F_Y(y) = P(Y \leq y) = P(g(X) \leq y)$.
2. Express this in terms of $X$ and use $f_X$.
3. Differentiate: $f_Y(y) = F_Y'(y)$.

**Example.** Let $X \sim \text{Uniform}(0, 1)$ and $Y = -\frac{1}{\lambda} \ln(X)$ for $\lambda > 0$.
$$F_Y(y) = P\left(-\frac{1}{\lambda}\ln X \leq y\right) = P(\ln X \geq -\lambda y) = P(X \geq e^{-\lambda y}) = 1 - e^{-\lambda y}$$
for $y \geq 0$. Differentiating: $f_Y(y) = \lambda e^{-\lambda y}$, the $\text{Exponential}(\lambda)$ PDF.

This is the **inverse CDF method** for generating random samples — a cornerstone of Monte Carlo simulation.

### Jacobian Method (Change of Variables)

For one-to-one transformations $Y = g(X)$ with differentiable inverse $X = g^{-1}(Y)$:
$$f_Y(y) = f_X(g^{-1}(y)) \left|\frac{dg^{-1}}{dy}\right|.$$
The absolute value of the derivative (the **Jacobian**) accounts for stretching/compression.

*Derivation.* If $g$ is strictly increasing: $F_Y(y) = P(g(X) \leq y) = P(X \leq g^{-1}(y)) = F_X(g^{-1}(y))$. Differentiating by chain rule: $f_Y(y) = f_X(g^{-1}(y)) \cdot (g^{-1})'(y)$. If $g$ is decreasing, we get a minus sign, hence the absolute value. $\blacksquare$

### Multivariate Jacobian

For a transformation $(X_1, X_2) \to (Y_1, Y_2)$ via $y_1 = g_1(x_1, x_2)$, $y_2 = g_2(x_1, x_2)$, with inverse $(x_1, x_2) = h(y_1, y_2)$:
$$f_{Y_1, Y_2}(y_1, y_2) = f_{X_1, X_2}(h_1(y_1,y_2), h_2(y_1,y_2)) \left|J\right|$$
where the Jacobian determinant is:
$$J = \det \begin{pmatrix} \frac{\partial x_1}{\partial y_1} & \frac{\partial x_1}{\partial y_2} \\ \frac{\partial x_2}{\partial y_1} & \frac{\partial x_2}{\partial y_2} \end{pmatrix}.$$
## Sum of Random Variables: Convolution

If $X$ and $Y$ are independent with PDFs $f_X$ and $f_Y$, the PDF of $Z = X + Y$ is the **convolution**:

![Convolution of random variables](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/04-convolution.png)
$$f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z - x) \, dx = (f_X * f_Y)(z).$$
*Derivation.* Let $Z = X + Y$ and use the auxiliary variable $W = X$:
$$f_{Z,W}(z, w) = f_{X,Y}(w, z - w) |J| = f_X(w) f_Y(z - w) \cdot 1.$$
Marginalize over $W$: $f_Z(z) = \int f_X(w) f_Y(z - w) dw$. $\blacksquare$

**Example.** Sum of two independent $\text{Exponential}(\lambda)$ random variables.
$$f_Z(z) = \int_0^z \lambda e^{-\lambda x} \lambda e^{-\lambda(z-x)} dx = \lambda^2 e^{-\lambda z} \int_0^z dx = \lambda^2 z e^{-\lambda z}$$
for $z > 0$. This is $\text{Gamma}(2, \lambda)$, confirming that the sum of $n$ independent exponentials is Gamma.

### MGF Approach (Often Easier)

Since $M_{X+Y}(t) = M_X(t) M_Y(t)$ for independent $X, Y$, we can identify the sum's distribution by multiplying MGFs and recognizing the result.

**Example.** Sum of independent normals. If $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$:
$$M_{X+Y}(t) = e^{\mu_1 t + \sigma_1^2 t^2/2} \cdot e^{\mu_2 t + \sigma_2^2 t^2/2} = e^{(\mu_1 + \mu_2)t + (\sigma_1^2 + \sigma_2^2)t^2/2}$$
which is the MGF of $\mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$. The sum of independent normals is normal, with means and variances adding. $\blacksquare$

## The Multinomial Distribution

The multivariate generalization of the Binomial. In $n$ independent trials, each trial results in one of $k$ categories with probabilities $p_1, \ldots, p_k$ ($\sum p_i = 1$). The vector of counts $(X_1, \ldots, X_k)$ follows:
$$P(X_1 = n_1, \ldots, X_k = n_k) = \frac{n!}{n_1! \cdots n_k!} p_1^{n_1} \cdots p_k^{n_k}$$
where $\sum n_i = n$.

Properties:
- Each $X_i \sim \text{Binomial}(n, p_i)$ marginally
- $\text{Cov}(X_i, X_j) = -np_i p_j$ for $i \neq j$ (the counts are negatively correlated — if more trials land in category $i$, fewer are left for category $j$)

## Order Statistics

Given $n$ i.i.d. random variables $X_1, \ldots, X_n$, the **order statistics** are the sorted values $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$.

![Order statistics](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/04-order-statistics.png)

- $X_{(1)} = \min(X_1, \ldots, X_n)$
- $X_{(n)} = \max(X_1, \ldots, X_n)$
- $X_{(k)}$ is the $k$-th smallest value

### CDF of the Maximum
$$F_{X_{(n)}}(x) = P(\max_i X_i \leq x) = P(X_1 \leq x, \ldots, X_n \leq x) = [F_X(x)]^n.$$
Differentiating: $f_{X_{(n)}}(x) = n [F_X(x)]^{n-1} f_X(x)$.

### CDF of the Minimum
$$P(X_{(1)} > x) = P(\min_i X_i > x) = [1 - F_X(x)]^n.$$
So $F_{X_{(1)}}(x) = 1 - [1 - F_X(x)]^n$ and $f_{X_{(1)}}(x) = n [1 - F_X(x)]^{n-1} f_X(x)$.

**Example.** If $X_i \sim \text{Exp}(\lambda)$, the minimum of $n$ independent copies has:
$$P(X_{(1)} > x) = [e^{-\lambda x}]^n = e^{-n\lambda x}$$
So $X_{(1)} \sim \text{Exp}(n\lambda)$. The minimum of $n$ exponentials is also exponential, with rate $n\lambda$ — the wait until the first event gets shorter as we watch more independent processes.

### General $k$-th Order Statistic

The PDF of $X_{(k)}$ is:
$$f_{X_{(k)}}(x) = \frac{n!}{(k-1)!(n-k)!} [F_X(x)]^{k-1} [1 - F_X(x)]^{n-k} f_X(x).$$
This counts the number of ways to have exactly $k-1$ values below $x$, one value at $x$, and $n-k$ values above $x$.

## Python: Visualizing Joint Distributions

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

rhos = [-0.8, 0, 0.8]
for idx, rho in enumerate(rhos):
    ax = axes[0, idx]
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    # Generate samples
    np.random.seed(42)
    samples = np.random.multivariate_normal(mean, cov, 1000)

    # Scatter plot
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, c='steelblue')

    # Contour plot
    x = np.linspace(-3.5, 3.5, 100)
    y = np.linspace(-3.5, 3.5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = stats.multivariate_normal(mean, cov)
    ax.contour(X, Y, rv.pdf(pos), levels=5, colors='darkred', linewidths=1.5)

    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Bivariate Normal, $\\rho$ = {rho}', fontsize=13)
    ax.set_aspect('equal')

ax = axes[1, 0]
rho = 0.7
mean = [0, 0]
cov = [[1, rho], [rho, 1]]
np.random.seed(42)
samples = np.random.multivariate_normal(mean, cov, 2000)
ax.scatter(samples[:, 0], samples[:, 1], alpha=0.2, s=8, c='gray')

y_val = 1.0
mask = np.abs(samples[:, 1] - y_val) < 0.15
ax.scatter(samples[mask, 0], samples[mask, 1], c='red', s=15, alpha=0.7,
           label=f'X | Y $\\approx$ {y_val}')
cond_mean = rho * y_val
ax.axvline(cond_mean, color='red', linestyle='--', alpha=0.7,
           label=f'E[X|Y={y_val}] = {cond_mean:.1f}')
ax.axhline(y_val, color='orange', linestyle=':', alpha=0.5)
ax.set_title('Conditional Distribution', fontsize=13)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend(fontsize=9)

ax = axes[1, 1]
np.random.seed(42)
n_values = [2, 5, 10, 20]
x = np.linspace(0, 1, 200)
for n in n_values:
    # PDF of max of n Uniform(0,1)
    pdf_max = n * x**(n-1)
    ax.plot(x, pdf_max, linewidth=2, label=f'max of {n}')
ax.set_title('Order Statistics: Max of Uniform(0,1)', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

ax = axes[1, 2]
x = np.linspace(0, 10, 300)
lam = 1.0
for n in [1, 2, 3, 5]:
    # Gamma(n, lambda) PDF
    pdf = stats.gamma.pdf(x, n, scale=1/lam)
    ax.plot(x, pdf, linewidth=2, label=f'Sum of {n} Exp({lam})')
ax.set_title('Sum of Exponentials = Gamma', fontsize=13)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

plt.tight_layout()
plt.savefig('joint_distributions.png', dpi=150)
plt.show()
```

The top row shows how correlation $\rho$ shapes the bivariate normal: negative correlation tilts the ellipse one way, zero correlation gives a circle (independent), and positive correlation tilts the other way. The bottom row illustrates conditional distributions (the red slice), order statistics (how the maximum shifts toward 1 as $n$ grows), and convolutions (sums of exponentials becoming Gamma distributions).

## The Multivariate Normal Distribution

The bivariate normal generalizes to $d$ dimensions. A random vector $\mathbf{X} = (X_1, \ldots, X_d)^T$ has a **multivariate normal** distribution $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ if its PDF is:
$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$
where $\boldsymbol{\mu} \in \mathbb{R}^d$ is the mean vector and $\boldsymbol{\Sigma}$ is the $d \times d$ positive definite covariance matrix.

### Key Properties

1. **Marginals are normal.** Any subset of components has a multivariate normal distribution (just take the corresponding sub-vector of $\boldsymbol{\mu}$ and sub-matrix of $\boldsymbol{\Sigma}$).

2. **Linear transformations preserve normality.** If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\mathbf{Y} = A\mathbf{X} + \mathbf{b}$, then $\mathbf{Y} \sim \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b}, A\boldsymbol{\Sigma}A^T)$.

3. **Uncorrelated implies independent.** For jointly normal variables, $\text{Cov}(X_i, X_j) = 0$ implies $X_i$ and $X_j$ are independent. This is a special property of the normal — it fails for general distributions.

4. **Conditional distributions are normal.** Partition $\mathbf{X} = \begin{pmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{pmatrix}$ with corresponding mean and covariance blocks. Then:
$$\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$$
where $\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$ and $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$.

The conditional mean is a linear function of the conditioning variables — this is the foundation of linear regression. The conditional covariance does not depend on $\mathbf{x}_2$ at all — the "shape" of the uncertainty doesn't change, only the center shifts.

## Covariance Matrix Properties

The covariance matrix $\boldsymbol{\Sigma}$ is:
- **Symmetric:** $\boldsymbol{\Sigma}^T = \boldsymbol{\Sigma}$
- **Positive semi-definite:** $\mathbf{a}^T \boldsymbol{\Sigma} \mathbf{a} \geq 0$ for all vectors $\mathbf{a}$

*Proof of positive semi-definiteness.* $\mathbf{a}^T \boldsymbol{\Sigma} \mathbf{a} = \text{Var}(\mathbf{a}^T \mathbf{X}) \geq 0$ since variance is always non-negative. $\blacksquare$

The **eigendecomposition** $\boldsymbol{\Sigma} = Q\Lambda Q^T$ reveals the principal axes of the distribution. The eigenvectors give the directions of maximum and minimum variance, and the eigenvalues give the variances along those directions. This is exactly what **Principal Component Analysis (PCA)** computes — it finds the directions that capture the most variance in the data.

### The Mahalanobis Distance

The exponent of the multivariate normal PDF defines a distance metric:
$$d_M(\mathbf{x}, \boldsymbol{\mu}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}.$$
This is the **Mahalanobis distance** — it measures how many "standard deviations" a point is from the mean, accounting for correlations and different scales. Points on an ellipsoid of constant Mahalanobis distance have equal density under the multivariate normal.

The Mahalanobis distance reduces to:
- The Euclidean distance when $\boldsymbol{\Sigma} = \sigma^2 I$ (isotropic covariance)
- The standardized distance $|x - \mu|/\sigma$ in one dimension

In machine learning, the Mahalanobis distance is used for anomaly detection, clustering, and as the kernel of Gaussian discriminant analysis.

### Chi-Squared Connection

If $\mathbf{Z} \sim \mathcal{N}(\mathbf{0}, I_d)$, then $\|\mathbf{Z}\|^2 = Z_1^2 + \cdots + Z_d^2 \sim \chi^2(d)$. More generally, if $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then:
$$(\mathbf{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \sim \chi^2(d).$$
This is why the chi-squared distribution appears in goodness-of-fit tests, likelihood ratio tests, and confidence regions for multivariate parameters.

## Copulas: Separating Marginals from Dependence

**Sklar's theorem** states that any joint distribution can be decomposed into:

1. The marginal distributions of each variable
2. A **copula** that captures the dependence structure
$$F(x, y) = C(F_X(x), F_Y(y))$$
where $C: [0,1]^2 \to [0,1]$ is the copula function.

This means you can model marginals and dependence separately. Two datasets can have the same marginals (both Gaussian, say) but very different dependence structures (Gaussian copula vs heavy-tailed t-copula).

Copulas are widely used in finance (modeling correlated defaults), climate science (joint modeling of temperature and precipitation), and any field where the relationship between variables is more complex than linear correlation.

## Bayesian Networks and Conditional Independence

In high dimensions, specifying a full joint distribution is impractical — a joint PMF over $d$ binary variables has $2^d - 1$ free parameters. **Bayesian networks** (directed graphical models) exploit conditional independence to factor the joint distribution efficiently.

A Bayesian network encodes the factorization:
$$p(x_1, x_2, \ldots, x_d) = \prod_{i=1}^d p(x_i \mid \text{parents}(x_i))$$
where each variable depends only on its parents in a directed acyclic graph (DAG). This can reduce $2^d - 1$ parameters to something much smaller.

**Conditional independence** $X \perp Y \mid Z$ means $p(x, y \mid z) = p(x \mid z) p(y \mid z)$. Knowing $Z$ "screens off" the dependence between $X$ and $Y$.

**Example.** $X$ = "sprinkler on", $Y$ = "rain", $Z$ = "grass is wet". Knowing whether the grass is wet makes sprinkler and rain conditionally dependent (if the grass is wet and it's not raining, the sprinkler was probably on). But marginally, sprinkler and rain might be independent. This is an example of **explaining away** — a key concept in probabilistic reasoning.

Understanding the difference between marginal independence ($X \perp Y$) and conditional independence ($X \perp Y \mid Z$) is critical in causal inference and graphical models. Two variables can be marginally independent but conditionally dependent (collider bias), or marginally dependent but conditionally independent (mediation). The structure of the DAG determines which independencies hold.

For the mathematically curious: the **d-separation** criterion in DAGs provides a complete graphical rule for reading off conditional independencies from the network structure, avoiding the need to compute any probabilities directly. This connection between graph theory and probability is one of the deepest in modern statistics and forms the backbone of causal inference as developed by Judea Pearl.

## Functions of Multiple Random Variables

Beyond sums, we often need distributions of other functions of random variables.

### Product of Independent Random Variables

If $X$ and $Y$ are independent and positive, the PDF of $Z = XY$ is:
$$f_Z(z) = \int_0^{\infty} f_X(x) f_Y(z/x) \frac{1}{x} dx.$$
### Ratio of Independent Random Variables

The PDF of $Z = X/Y$ (for $Y > 0$) is:
$$f_Z(z) = \int_0^{\infty} y \, f_X(zy) f_Y(y) \, dy.$$
**Example.** If $X \sim \mathcal{N}(0,1)$ and $Y \sim \chi^2(n)/n$ are independent, then $X/\sqrt{Y} \sim t_n$, the Student's t-distribution with $n$ degrees of freedom. This is the distribution that arises naturally in the t-test (Article 7).

### Max and Min of Random Variables

For independent random variables, the distribution of the maximum and minimum can be derived from the CDF:

**Maximum:** $F_{\max}(z) = P(\max(X, Y) \leq z) = P(X \leq z)P(Y \leq z) = F_X(z)F_Y(z)$.

**Minimum:** $F_{\min}(z) = P(\min(X, Y) \leq z) = 1 - P(X > z)P(Y > z) = 1 - (1-F_X(z))(1-F_Y(z))$.

**Example.** Two components in series (system fails if either fails): lifetime = $\min(X, Y)$. Two components in parallel (system fails only if both fail): lifetime = $\max(X, Y)$. If both have Exponential($\lambda$) lifetimes:

- Series: $\min(X,Y) \sim \text{Exponential}(2\lambda)$ — fails twice as fast.
- Parallel: $F_{\max}(z) = (1-e^{-\lambda z})^2$, which is not exponential. Mean lifetime = $3/(2\lambda)$ — a 50% improvement over a single component.

## Summary

| Concept | Key Formula | Interpretation |
|---|---|---|
| Joint PMF/PDF | $p(x,y)$ or $f(x,y)$ | Full probabilistic description of $(X,Y)$ |
| Marginal | $f_X(x) = \int f(x,y) dy$ | "Forget" the other variable |
| Conditional | $f(x\mid y) = f(x,y)/f_Y(y)$ | Distribution of $X$ knowing $Y=y$ |
| Independence | $f(x,y) = f_X(x) f_Y(y)$ | Joint factors into marginals |
| Jacobian | $f_Y(y) = f_X(g^{-1}(y)) \lvert dg^{-1}/dy\rvert$ | Transform densities correctly |
| Convolution | $f_{X+Y} = f_X * f_Y$ | PDF of sum of independents |
| Order stats | $f_{X_{(k)}}$ involves $F^{k-1}(1-F)^{n-k}f$ | Distribution of $k$-th smallest |
| MVN conditional | $\boldsymbol{\mu}_{1\mid 2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$ | Linear regression |

## What's Next

We now have the machinery to describe how multiple random variables behave together. The next article tackles the crown jewels of probability: the Law of Large Numbers and the Central Limit Theorem. These theorems explain why sample averages are reliable and why the Normal distribution appears everywhere — and they connect directly to why algorithms like stochastic gradient descent converge.
