---
title: "Probability and Statistics (6): Estimation — MLE, MAP, and the Bias-Variance Story"
date: 2024-08-26 09:00:00
tags:
  - Probability
  - Statistics
  - Maximum Likelihood
  - Bias-Variance
categories:
  - Probability and Statistics
series: probability-statistics
lang: en
mathjax: true
description: "Point estimation from method of moments through maximum likelihood and MAP, with Fisher information, the Cramer-Rao bound, and the bias-variance decomposition that explains overfitting and underfitting."
disableNunjucks: true
series_order: 6
translationKey: "probability-statistics-6"
---

Everything we've built so far — distributions, expectations, limit theorems — assumed we knew the parameters. The Gaussian has mean $\mu$ and variance $\sigma^2$. The Binomial has $n$ trials with success probability $p$. But in practice, you don't know $\mu$ or $p$. You observe data and try to figure them out.

This is **estimation theory**: the bridge between probability (where parameters are given) and statistics (where parameters are inferred). It's also where the foundations of machine learning live. Every time you train a model, you are estimating parameters from data. The quality of that estimation determines whether your model generalizes or overfits.

## The Setup: Estimators vs Estimates

Suppose we observe data $x_1, x_2, \ldots, x_n$ drawn i.i.d. from some distribution $p(x|\theta)$, where $\theta$ is an unknown parameter (or vector of parameters).

![Confidence intervals](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/06-confidence-interval.png)


An **estimator** $\hat{\theta}$ is a function of the data: $\hat{\theta} = g(X_1, \ldots, X_n)$. It is a random variable (since the data are random). An **estimate** is the numerical value $\hat{\theta}(x_1, \ldots, x_n)$ obtained from a specific dataset.

**Notation convention:** $\hat{\theta}$ (hat) denotes an estimator/estimate. $\theta$ (no hat) denotes the true parameter.

## Properties of Estimators

### Bias

The **bias** of an estimator is

![Bias-variance tradeoff](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/06-bias-variance.png)


$$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta.$$

An estimator is **unbiased** if $\text{Bias}(\hat{\theta}) = 0$, meaning $E[\hat{\theta}] = \theta$ for all values of $\theta$.

**Example.** The sample mean $\bar{X} = \frac{1}{n}\sum X_i$ is unbiased for $\mu$:

$$E[\bar{X}] = \frac{1}{n}\sum E[X_i] = \mu. \quad \checkmark$$

The sample variance $S^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ is **biased**:

$$E\left[\frac{1}{n}\sum(X_i - \bar{X})^2\right] = \frac{n-1}{n}\sigma^2 \neq \sigma^2.$$

This is why the unbiased sample variance uses $n-1$ in the denominator:

$$S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2.$$

*Proof of the bias.* Expand:

$$\sum(X_i - \bar{X})^2 = \sum X_i^2 - n\bar{X}^2.$$

Taking expectations:

$$E\left[\sum X_i^2\right] = n(\sigma^2 + \mu^2), \quad E[n\bar{X}^2] = n\left(\frac{\sigma^2}{n} + \mu^2\right) = \sigma^2 + n\mu^2.$$

So $E\left[\sum(X_i - \bar{X})^2\right] = n\sigma^2 + n\mu^2 - \sigma^2 - n\mu^2 = (n-1)\sigma^2$. Dividing by $n$ gives $\frac{n-1}{n}\sigma^2$. Dividing by $n-1$ gives $\sigma^2$. $\blacksquare$

### Consistency

An estimator is **consistent** if $\hat{\theta}_n \xrightarrow{P} \theta$ as $n \to \infty$.

By the WLLN, the sample mean $\bar{X}$ is consistent for $\mu$. Both the biased ($1/n$) and unbiased ($1/(n-1)$) sample variances are consistent for $\sigma^2$.

### Efficiency

Among all unbiased estimators of $\theta$, the one with the smallest variance is the most **efficient**. The Cramer-Rao lower bound (derived below) tells us how small the variance can possibly be.

### Sufficiency

A statistic $T(X_1, \ldots, X_n)$ is **sufficient** for $\theta$ if the conditional distribution of the data given $T$ does not depend on $\theta$. Informally, $T$ captures all the information in the data about $\theta$ — knowing $T$, you can throw away the raw data without losing anything.

**Example.** For $X_i \sim \text{Bernoulli}(p)$, the sum $T = \sum X_i$ is sufficient for $p$. Knowing that you observed 7 successes out of 10 trials is all you need; the specific sequence (e.g., 1101110100 vs 1110101100) carries no additional information about $p$.

## Method of Moments

The simplest estimation method: match sample moments to population moments.

**Recipe:**
1. Express the parameters in terms of population moments: $\theta = h(\mu_1, \mu_2, \ldots)$ where $\mu_k = E[X^k]$.
2. Replace population moments with sample moments: $\hat{\mu}_k = \frac{1}{n}\sum X_i^k$.
3. Plug in: $\hat{\theta}_{\text{MoM}} = h(\hat{\mu}_1, \hat{\mu}_2, \ldots)$.

**Example: Gamma distribution.** $X \sim \text{Gamma}(\alpha, \beta)$ with $E[X] = \alpha/\beta$ and $E[X^2] = \alpha(\alpha+1)/\beta^2$.

From $\mu_1 = \alpha/\beta$ and $\mu_2 - \mu_1^2 = \alpha/\beta^2$ (the variance):

$$\hat{\beta}_{\text{MoM}} = \frac{\bar{X}}{S^2}, \qquad \hat{\alpha}_{\text{MoM}} = \frac{\bar{X}^2}{S^2}.$$

Method of moments estimators are easy to compute but generally not the most efficient. They serve as good starting points and as a sanity check.

## Maximum Likelihood Estimation


![Estimation methods comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/06-estimation-comparison.png)

### The Likelihood Function

Given observed data $x_1, \ldots, x_n$ drawn i.i.d. from $p(x|\theta)$, the **likelihood** is:

$$L(\theta) = \prod_{i=1}^n p(x_i | \theta).$$

The **log-likelihood** is:

$$\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln p(x_i | \theta).$$

The log-likelihood is easier to work with (sums are simpler than products), and maximizing $\ell$ is equivalent to maximizing $L$ since $\ln$ is monotonically increasing.

### The MLE

The **maximum likelihood estimator** is the value of $\theta$ that maximizes the likelihood:

![MLE likelihood surface](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/06-mle-likelihood.png)


$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta).$$

In regular cases, the MLE satisfies the **score equation**:

$$\frac{\partial \ell}{\partial \theta} = 0.$$

### Example 1: Bernoulli

$X_i \sim \text{Bernoulli}(p)$, observed $k$ successes in $n$ trials.

$$\ell(p) = \sum_{i=1}^n [x_i \ln p + (1-x_i) \ln(1-p)] = k \ln p + (n-k) \ln(1-p).$$

$$\frac{d\ell}{dp} = \frac{k}{p} - \frac{n-k}{1-p} = 0.$$

$$k(1-p) = (n-k)p \implies k = np \implies \hat{p}_{\text{MLE}} = \frac{k}{n} = \bar{X}.$$

The MLE is the sample proportion. Natural and intuitive.

### Example 2: Gaussian (Both Parameters)

$X_i \sim \mathcal{N}(\mu, \sigma^2)$.

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i - \mu)^2.$$

Setting $\partial\ell/\partial\mu = 0$:

$$\frac{1}{\sigma^2}\sum(x_i - \mu) = 0 \implies \hat{\mu}_{\text{MLE}} = \bar{X}.$$

Setting $\partial\ell/\partial\sigma^2 = 0$:

$$-\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum(x_i - \mu)^2 = 0 \implies \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(x_i - \bar{X})^2.$$

Note: the MLE for $\sigma^2$ divides by $n$, not $n-1$. It is biased (by a factor of $(n-1)/n$) but consistent and efficient. For large $n$, the difference is negligible.

### Example 3: Poisson

$X_i \sim \text{Poisson}(\lambda)$.

$$\ell(\lambda) = \sum_{i=1}^n [x_i \ln\lambda - \lambda - \ln(x_i!)] = \left(\sum x_i\right)\ln\lambda - n\lambda - \sum\ln(x_i!).$$

$$\frac{d\ell}{d\lambda} = \frac{\sum x_i}{\lambda} - n = 0 \implies \hat{\lambda}_{\text{MLE}} = \bar{X}.$$

Again, the sample mean — which makes sense since $E[X] = \lambda$ for Poisson.

### Example 4: Uniform

$X_i \sim \text{Uniform}(0, \theta)$, $\theta > 0$ unknown.

$$L(\theta) = \prod_{i=1}^n \frac{1}{\theta} \cdot \mathbf{1}_{0 \leq x_i \leq \theta} = \frac{1}{\theta^n} \cdot \mathbf{1}_{\theta \geq x_{(n)}}$$

where $x_{(n)} = \max(x_1, \ldots, x_n)$. For $\theta \geq x_{(n)}$, $L(\theta) = 1/\theta^n$ is decreasing in $\theta$, so $L$ is maximized at $\hat{\theta}_{\text{MLE}} = x_{(n)} = \max_i x_i$.

This is an interesting case where the MLE is **biased**: $E[X_{(n)}] = \frac{n}{n+1}\theta < \theta$. The MLE consistently underestimates because the maximum of $n$ samples can never exceed $\theta$ but can be much less. The unbiased estimator is $\frac{n+1}{n} X_{(n)}$.

This example also illustrates that MLE doesn't always satisfy a smooth score equation — here the likelihood is discontinuous at $\theta = x_{(n)}$.

## Properties of the MLE

Under regularity conditions (smooth, identifiable model):

1. **Consistency:** $\hat{\theta}_{\text{MLE}} \xrightarrow{P} \theta_0$ (the true parameter).
2. **Asymptotic normality:** $\sqrt{n}(\hat{\theta}_{\text{MLE}} - \theta_0) \xrightarrow{d} \mathcal{N}(0, I(\theta_0)^{-1})$ where $I(\theta_0)$ is the Fisher information.
3. **Asymptotic efficiency:** The MLE achieves the Cramer-Rao lower bound asymptotically.
4. **Invariance:** If $\hat{\theta}$ is the MLE of $\theta$, then $g(\hat{\theta})$ is the MLE of $g(\theta)$.

*Proof of invariance.* If $\hat{\theta} = \arg\max_\theta L(\theta)$, and $\phi = g(\theta)$ is a one-to-one function, then $\hat{\phi} = g(\hat{\theta}) = \arg\max_\phi L(g^{-1}(\phi))$. For non-injective $g$, define $\hat{\phi} = g(\hat{\theta})$ directly; the profile likelihood is maximized at this value. $\blacksquare$

**Example of invariance.** If $\hat{\sigma}^2_{\text{MLE}}$ is the MLE of $\sigma^2$, then $\hat{\sigma}_{\text{MLE}} = \sqrt{\hat{\sigma}^2_{\text{MLE}}}$ is the MLE of $\sigma$. You don't need to re-derive — just apply the transformation.

**Warning:** Invariance is a property of the MLE, not of unbiased estimators in general. If $\hat{\theta}$ is unbiased for $\theta$, $g(\hat{\theta})$ is generally **not** unbiased for $g(\theta)$ (by Jensen's inequality, unless $g$ is linear).

## Fisher Information and the Cramer-Rao Bound


![Fisher information](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/06-fisher-information.png)

### Fisher Information

The **score function** is $s(\theta) = \frac{\partial}{\partial\theta} \ln p(X|\theta)$.

Under regularity conditions, $E[s(\theta)] = 0$.

The **Fisher information** is the variance of the score:

$$I(\theta) = E\left[\left(\frac{\partial \ln p(X|\theta)}{\partial\theta}\right)^2\right] = -E\left[\frac{\partial^2 \ln p(X|\theta)}{\partial\theta^2}\right].$$

The second equality (derived via interchanging differentiation and expectation) gives a more convenient formula. For $n$ i.i.d. observations, $I_n(\theta) = n \cdot I_1(\theta)$.

**Example: Bernoulli.** $\ln p(x|p) = x\ln p + (1-x)\ln(1-p)$.

$$\frac{\partial^2}{\partial p^2}\ln p(x|p) = -\frac{x}{p^2} - \frac{1-x}{(1-p)^2}.$$

$$I_1(p) = -E\left[-\frac{X}{p^2} - \frac{1-X}{(1-p)^2}\right] = \frac{p}{p^2} + \frac{1-p}{(1-p)^2} = \frac{1}{p} + \frac{1}{1-p} = \frac{1}{p(1-p)}.$$

### The Cramer-Rao Lower Bound

**Theorem (Cramer-Rao).** For any unbiased estimator $\hat{\theta}$ of $\theta$:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I_n(\theta)} = \frac{1}{n \cdot I_1(\theta)}.$$

*Proof sketch.* Apply the Cauchy-Schwarz inequality to $\text{Cov}(\hat{\theta}, s(\theta))$, where $s$ is the total score. Since $E[s] = 0$, the covariance equals $E[\hat{\theta} \cdot s]$, which equals 1 (using the unbiasedness condition and differentiation under the integral). Then Cauchy-Schwarz gives $1 \leq \text{Var}(\hat{\theta}) \cdot \text{Var}(s) = \text{Var}(\hat{\theta}) \cdot I_n(\theta)$. $\blacksquare$

**Example.** For Bernoulli, the CRLB is $\frac{p(1-p)}{n}$. The MLE $\hat{p} = \bar{X}$ has $\text{Var}(\hat{p}) = \frac{p(1-p)}{n}$, achieving the bound exactly. The MLE is efficient.

## MAP Estimation

### From MLE to MAP

MLE treats $\theta$ as a fixed unknown. **Maximum a posteriori (MAP)** estimation treats $\theta$ as a random variable with a **prior distribution** $p(\theta)$.

By Bayes' theorem:

$$p(\theta | x_1, \ldots, x_n) \propto p(x_1, \ldots, x_n | \theta) \cdot p(\theta) = L(\theta) \cdot p(\theta).$$

The MAP estimate maximizes the posterior:

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \left[\ell(\theta) + \ln p(\theta)\right].$$

### Connection to Regularization

The MAP estimate is the MLE with a penalty term — the log-prior acts as a regularizer.

| Prior | Penalty | ML Equivalent |
|---|---|---|
| $\theta \sim \mathcal{N}(0, \tau^2)$ | $-\frac{\theta^2}{2\tau^2}$ | L2 regularization (Ridge) |
| $\theta \sim \text{Laplace}(0, b)$ | $-\frac{|\theta|}{b}$ | L1 regularization (Lasso) |

**Example.** Gaussian prior on the mean of a Gaussian with known variance $\sigma^2$.

Data: $X_i \sim \mathcal{N}(\mu, \sigma^2)$. Prior: $\mu \sim \mathcal{N}(\mu_0, \tau^2)$.

$$\hat{\mu}_{\text{MAP}} = \arg\max_\mu \left[-\frac{1}{2\sigma^2}\sum(x_i - \mu)^2 - \frac{(\mu - \mu_0)^2}{2\tau^2}\right].$$

Taking the derivative and setting to zero:

$$\frac{\sum(x_i - \mu)}{\sigma^2} - \frac{\mu - \mu_0}{\tau^2} = 0.$$

$$\hat{\mu}_{\text{MAP}} = \frac{\frac{n}{\sigma^2}\bar{X} + \frac{1}{\tau^2}\mu_0}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}} = \frac{n\tau^2\bar{X} + \sigma^2\mu_0}{n\tau^2 + \sigma^2}.$$

This is a **weighted average** of the sample mean $\bar{X}$ and the prior mean $\mu_0$. When $n$ is large, the data dominates and MAP approaches MLE. When $n$ is small, the prior pulls the estimate toward $\mu_0$.

## The Bias-Variance Decomposition

### The Mean Squared Error

The **mean squared error** (MSE) of an estimator is:

$$\text{MSE}(\hat{\theta}) = E[(\hat{\theta} - \theta)^2].$$

**Theorem (Bias-Variance Decomposition).**

$$\text{MSE}(\hat{\theta}) = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta}).$$

*Proof.* Let $b = E[\hat{\theta}] - \theta$ be the bias. Then:

$$\text{MSE} = E[(\hat{\theta} - \theta)^2] = E[(\hat{\theta} - E[\hat{\theta}] + E[\hat{\theta}] - \theta)^2]$$

$$= E[(\hat{\theta} - E[\hat{\theta}])^2 + 2(\hat{\theta} - E[\hat{\theta}])(E[\hat{\theta}] - \theta) + (E[\hat{\theta}] - \theta)^2]$$

$$= \text{Var}(\hat{\theta}) + 2(E[\hat{\theta}] - E[\hat{\theta}]) \cdot b + b^2 = \text{Var}(\hat{\theta}) + b^2. \quad \blacksquare$$

### Connection to Machine Learning

In supervised learning with predictor $\hat{f}$, for a test point $x$ with true value $y = f(x) + \varepsilon$ where $E[\varepsilon] = 0$ and $\text{Var}(\varepsilon) = \sigma^2$:

$$E[(\hat{f}(x) - y)^2] = \underbrace{(E[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{\text{Var}(\hat{f}(x))}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}.$$

- **High bias** = underfitting. The model is too simple to capture the true pattern.
- **High variance** = overfitting. The model fits noise in the training data.
- **The tradeoff:** Increasing model complexity reduces bias but increases variance. The optimal model balances both.

Regularization (= MAP with a prior) deliberately introduces bias to reduce variance, often lowering the total MSE.

## Python: Bias-Variance Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# True function
def f_true(x):
    return np.sin(2 * np.pi * x)

# Generate datasets and fit polynomials of different degrees
n_train = 20
n_datasets = 200
x_test = np.linspace(0, 1, 100)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

degrees = [1, 3, 5, 10, 15, 19]

for idx, degree in enumerate(degrees):
    ax = axes[idx // 3, idx % 3]
    predictions = np.zeros((n_datasets, len(x_test)))

    for d in range(n_datasets):
        x_train = np.random.uniform(0, 1, n_train)
        y_train = f_true(x_train) + np.random.normal(0, 0.3, n_train)
        coeffs = np.polyfit(x_train, y_train, degree)
        predictions[d] = np.polyval(coeffs, x_test)

    # Plot individual fits (first 20)
    for d in range(min(20, n_datasets)):
        ax.plot(x_test, predictions[d], 'b-', alpha=0.1, linewidth=0.5)

    # Plot mean prediction
    mean_pred = predictions.mean(axis=0)
    ax.plot(x_test, mean_pred, 'r-', linewidth=2, label='E[$\\hat{f}$]')

    # Plot true function
    ax.plot(x_test, f_true(x_test), 'k--', linewidth=2, label='f(x)')

    # Compute bias^2 and variance
    bias_sq = np.mean((mean_pred - f_true(x_test))**2)
    variance = np.mean(predictions.var(axis=0))

    ax.set_title(f'Degree {degree}\nBias$^2$={bias_sq:.3f}, Var={variance:.3f}',
                 fontsize=11)
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=9)

plt.suptitle('Bias-Variance Tradeoff: Polynomial Regression', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('bias_variance.png', dpi=150, bbox_inches='tight')
plt.show()
```

The visualization shows the tradeoff directly. At degree 1 (underfitting), the mean prediction (red) is far from the true function (black dashed) — high bias — but the individual fits (blue) cluster tightly — low variance. At degree 19 (overfitting), the mean prediction matches the true function well — low bias — but individual fits are wildly different — high variance. The sweet spot is somewhere in between.

```python
# Compute bias-variance tradeoff curve
degrees = range(1, 20)
biases = []
variances = []

for degree in degrees:
    predictions = np.zeros((n_datasets, len(x_test)))
    for d in range(n_datasets):
        x_train = np.random.uniform(0, 1, n_train)
        y_train = f_true(x_train) + np.random.normal(0, 0.3, n_train)
        coeffs = np.polyfit(x_train, y_train, degree)
        predictions[d] = np.polyval(coeffs, x_test)
    mean_pred = predictions.mean(axis=0)
    biases.append(np.mean((mean_pred - f_true(x_test))**2))
    variances.append(np.mean(predictions.var(axis=0)))

mse = np.array(biases) + np.array(variances) + 0.3**2  # + noise variance

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(list(degrees), biases, 'b-o', label='Bias$^2$', markersize=5)
ax.plot(list(degrees), variances, 'r-o', label='Variance', markersize=5)
ax.plot(list(degrees), mse, 'k-o', label='Total MSE', markersize=5)
ax.axhline(y=0.3**2, color='gray', linestyle=':', label='Noise ($\\sigma^2$)')
ax.set_xlabel('Polynomial Degree', fontsize=13)
ax.set_ylabel('Error', fontsize=13)
ax.set_title('Bias-Variance Tradeoff', fontsize=14)
ax.legend(fontsize=12)
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bias_variance_curve.png', dpi=150)
plt.show()
```

## Sufficient Statistics and Data Reduction

### The Factorization Theorem

A statistic $T(\mathbf{X})$ is **sufficient** for $\theta$ if and only if the likelihood factors as:

$$p(\mathbf{x} | \theta) = g(T(\mathbf{x}), \theta) \cdot h(\mathbf{x})$$

where $g$ depends on the data only through $T$, and $h$ does not depend on $\theta$.

**Example.** For $X_i \sim \text{Poisson}(\lambda)$:

$$p(\mathbf{x}|\lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} = \frac{\lambda^{\sum x_i} e^{-n\lambda}}{\prod x_i!} = \underbrace{\lambda^{\sum x_i} e^{-n\lambda}}_{g(\sum x_i, \lambda)} \cdot \underbrace{\frac{1}{\prod x_i!}}_{h(\mathbf{x})}.$$

So $T = \sum X_i$ is sufficient for $\lambda$. Once you know the total count, the individual observations add no further information about $\lambda$.

### The Rao-Blackwell Theorem

**Theorem.** If $\hat{\theta}$ is any unbiased estimator and $T$ is a sufficient statistic, then $\tilde{\theta} = E[\hat{\theta} | T]$ is also unbiased and has smaller (or equal) variance:

$$\text{Var}(\tilde{\theta}) \leq \text{Var}(\hat{\theta}).$$

*Proof.* Unbiasedness: $E[\tilde{\theta}] = E[E[\hat{\theta}|T]] = E[\hat{\theta}] = \theta$ (tower property). For variance, use the law of total variance:

$$\text{Var}(\hat{\theta}) = E[\text{Var}(\hat{\theta}|T)] + \text{Var}(E[\hat{\theta}|T]) = E[\text{Var}(\hat{\theta}|T)] + \text{Var}(\tilde{\theta}) \geq \text{Var}(\tilde{\theta}). \quad \blacksquare$$

This theorem says: **always condition on a sufficient statistic.** It can only help, never hurt.

## Cross-Validation: The Practical Bias-Variance Tool

The bias-variance decomposition is a theoretical framework. In practice, you can't compute bias or variance directly (they depend on the unknown true function). **Cross-validation** is the practical alternative.

**k-fold cross-validation:**
1. Split data into $k$ equal folds.
2. For each fold $i$: train on all other folds, evaluate on fold $i$.
3. Average the $k$ test errors.

This estimates the test error (which includes both bias and variance effects) without needing a separate test set. Common choices: $k = 5$ or $k = 10$.

**Leave-one-out cross-validation (LOOCV):** $k = n$. Each observation is held out once. Low bias (training set is almost the full dataset) but high variance (the $n$ training sets are very similar, so the $n$ error estimates are correlated).

The bias-variance tradeoff applies to cross-validation itself:
- Small $k$: high bias (training sets are smaller), low variance
- Large $k$: low bias, high variance (correlated estimates)
- $k = 5$ or $10$ is a good compromise in most settings

## Exponential Families: A Unifying Framework

Most distributions we've studied belong to the **exponential family**, which has the general form:

$$p(x|\theta) = h(x) \exp\left(\eta(\theta)^T T(x) - A(\theta)\right)$$

where $T(x)$ is the sufficient statistic, $\eta(\theta)$ is the natural parameter, and $A(\theta)$ is the log-partition function.

| Distribution | $T(x)$ | $\eta$ | $A(\eta)$ |
|---|---|---|---|
| Bernoulli($p$) | $x$ | $\ln\frac{p}{1-p}$ | $\ln(1+e^\eta)$ |
| Poisson($\lambda$) | $x$ | $\ln\lambda$ | $e^\eta$ |
| Normal($\mu$, known $\sigma^2$) | $x$ | $\mu/\sigma^2$ | $\eta^2\sigma^2/2$ |
| Exponential($\lambda$) | $x$ | $-\lambda$ | $-\ln(-\eta)$ |

The log-partition function $A(\theta)$ generates moments:

$$E[T(X)] = A'(\eta), \qquad \text{Var}(T(X)) = A''(\eta).$$

For exponential families, the MLE is uniquely determined by the sufficient statistic, and it equals the method of moments estimator based on $T(X)$. Conjugate priors also have a natural form, and the MLE always exists and is unique (under mild conditions).

## Numerical MLE: When Closed Forms Don't Exist

Many models don't have closed-form MLE solutions. In such cases, we optimize the log-likelihood numerically.

### Newton-Raphson Method

The score equation $\ell'(\theta) = 0$ can be solved iteratively:

$$\theta^{(t+1)} = \theta^{(t)} - \frac{\ell'(\theta^{(t)})}{\ell''(\theta^{(t)})}.$$

In the multivariate case:

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - [\nabla^2 \ell(\boldsymbol{\theta}^{(t)})]^{-1} \nabla \ell(\boldsymbol{\theta}^{(t)}).$$

### Fisher Scoring

Replace the observed Hessian $\nabla^2 \ell$ with its expectation $-I(\theta)$ (the Fisher information matrix):

$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} + [I(\boldsymbol{\theta}^{(t)})]^{-1} \nabla \ell(\boldsymbol{\theta}^{(t)}).$$

Fisher scoring is more stable than Newton-Raphson because the Fisher information matrix is guaranteed to be positive semi-definite (ensuring we move in an ascent direction). For exponential families, Newton-Raphson and Fisher scoring coincide.

### The EM Algorithm

For models with latent (unobserved) variables, direct MLE is often intractable. The **Expectation-Maximization (EM)** algorithm alternates between:

1. **E-step:** Compute $Q(\theta | \theta^{(t)}) = E_{\mathbf{Z}|\mathbf{X}, \theta^{(t)}}[\ell(\theta; \mathbf{X}, \mathbf{Z})]$ — the expected complete-data log-likelihood.
2. **M-step:** $\theta^{(t+1)} = \arg\max_\theta Q(\theta | \theta^{(t)})$ — maximize the expected log-likelihood.

Each iteration is guaranteed to increase (or maintain) the log-likelihood: $\ell(\theta^{(t+1)}) \geq \ell(\theta^{(t)})$. The algorithm converges to a local maximum.

EM is used for Gaussian mixture models, hidden Markov models, factor analysis, and any model where marginalizing over latent variables makes the likelihood intractable but conditioning on them makes it tractable.

### Gradient Descent for MLE

In high-dimensional models (like neural networks), computing the Hessian or Fisher information is too expensive. Instead, use first-order methods:

$$\theta^{(t+1)} = \theta^{(t)} + \eta \nabla \ell(\theta^{(t)})$$

where $\eta$ is the learning rate. This is **gradient ascent** on the log-likelihood — identical to what deep learning calls "training." The MLE framework gives the theoretical justification: we are maximizing the likelihood of the observed data under the model.

In stochastic settings (mini-batches), we use stochastic gradient ascent, and the CLT from Article 5 guarantees that the noise in the gradient estimate is approximately Gaussian with variance $O(1/B)$.

## Summary

| Method | Formula | Key Property |
|---|---|---|
| Method of Moments | Match $\hat{\mu}_k$ to theoretical moments | Simple, not always efficient |
| MLE | $\arg\max_\theta \sum \ln p(x_i|\theta)$ | Consistent, asymptotically efficient |
| MAP | $\arg\max_\theta [\ell(\theta) + \ln p(\theta)]$ | MLE + regularization |
| Fisher Information | $I(\theta) = -E[\partial^2 \ell/\partial\theta^2]$ | Measures information in data |
| CRLB | $\text{Var}(\hat{\theta}) \geq 1/(nI_1(\theta))$ | Lower bound on variance |
| Rao-Blackwell | $E[\hat{\theta}|T]$ improves $\hat{\theta}$ | Condition on sufficiency |
| Bias-Variance | $\text{MSE} = \text{Bias}^2 + \text{Var}$ | Explains overfitting/underfitting |

## What's Next

Estimation gives us point estimates — single best guesses for parameters. But how confident should we be? The next article tackles hypothesis testing and confidence intervals: the framework for quantifying uncertainty, controlling error rates, and making principled decisions from data. We'll also see why p-values are so often misunderstood and how to avoid the most common statistical pitfalls.
