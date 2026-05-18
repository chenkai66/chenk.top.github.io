---
title: "Probability and Statistics (8): Bayesian Statistics — Priors, Posteriors, and Why Frequentists Argue"
date: 2024-08-30 09:00:00
tags:
  - Probability
  - Statistics
  - Bayesian Inference
  - MCMC
  - Machine Learning
categories: Probability and Statistics
series: probability-statistics
lang: en
mathjax: true
description: "Bayesian inference from first principles: posterior distributions, conjugate priors, the Beta-Binomial and Normal-Normal models, credible intervals, predictive distributions, MCMC intuition, and deep connections to machine learning regularization."
disableNunjucks: true
series_order: 8
series_total: 8
translationKey: "probability-statistics-8"
---

Two statisticians walk into a bar. One says: "The probability of rain tomorrow is 30%." The other replies: "Probability is a long-run frequency. Since tomorrow only happens once, that statement is meaningless." The first one says: "It quantifies my uncertainty about a unique event." They proceed to argue for the rest of the evening.

This, roughly, is the Bayesian-frequentist debate. It's not about who's right — both frameworks are mathematically consistent. It's about what "probability" means and how that interpretation shapes the tools you use. Having worked through six articles of largely frequentist reasoning, we now develop the Bayesian perspective: parameters are random, data update our beliefs, and uncertainty is quantified through distributions rather than confidence intervals.

---

## Bayesian vs Frequentist: The Core Difference

![Conjugate prior families](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/08-conjugate-priors.png)

### The Frequentist View

- **Parameters** are fixed but unknown constants.
- **Probability** refers to long-run relative frequencies.
- **Inference** uses the sampling distribution of estimators (how $\hat{\theta}$ varies over repeated samples).
- A 95% confidence interval means: "If I repeated this experiment many times, 95% of the intervals I'd compute would contain $\theta$."

### The Bayesian View

- **Parameters** are random variables with probability distributions.
- **Probability** quantifies subjective uncertainty (degree of belief).
- **Inference** updates a prior distribution with observed data to produce a posterior distribution.
- A 95% credible interval means: "Given the data, there is a 95% probability that $\theta$ lies in this interval."

The Bayesian framework is arguably more intuitive for individual problems: you start with what you believe, observe data, and update. The frequentist framework is arguably better for method validation: you can guarantee error rates over repeated use without specifying priors.

In practice, the two often agree — especially with large samples, where the data dominate the prior.

## Bayes' Rule for Distributions

We've already seen Bayes' theorem for events. The Bayesian engine applies the same logic to distributions.

Given data $\mathbf{x} = (x_1, \ldots, x_n)$ and a parameter $\theta$:
$$\underbrace{p(\theta | \mathbf{x})}_{\text{posterior}} = \frac{\underbrace{p(\mathbf{x} | \theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}}{\underbrace{p(\mathbf{x})}_{\text{marginal likelihood}}}$$
where the marginal likelihood (also called the **evidence**) is:
$$p(\mathbf{x}) = \int p(\mathbf{x} | \theta) \, p(\theta) \, d\theta.$$
Since $p(\mathbf{x})$ is a constant with respect to $\theta$, we often write:
$$\boxed{p(\theta | \mathbf{x}) \propto p(\mathbf{x} | \theta) \cdot p(\theta)}$$
**Posterior is proportional to Likelihood times Prior.**

This is the fundamental equation of Bayesian statistics. Everything else is a consequence.

### Interpreting the Components

- **Prior** $p(\theta)$: What you believe about $\theta$ before seeing data. This is where domain knowledge, previous experiments, or "reasonable defaults" enter.
- **Likelihood** $p(\mathbf{x} | \theta)$: The probability of the observed data given a specific value of $\theta$. This is the same likelihood function used in MLE.
- **Posterior** $p(\theta | \mathbf{x})$: Your updated beliefs after seeing the data. This is a full probability distribution, not a single point estimate.

## Conjugate Priors

A prior is **conjugate** to a likelihood if the resulting posterior belongs to the same family as the prior. Conjugacy makes the math tractable — the posterior has a closed-form expression.

![Prior to posterior updating](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/08-prior-posterior.png)

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Bernoulli/Binomial | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (known $\sigma^2$) | Normal | Normal |
| Normal (known $\mu$) | Inverse-Gamma | Inverse-Gamma |
| Multinomial | Dirichlet | Dirichlet |
| Exponential | Gamma | Gamma |

Without conjugacy, the posterior integral $p(\mathbf{x}) = \int p(\mathbf{x}|\theta)p(\theta)d\theta$ may not have a closed form, and we need numerical methods (MCMC, variational inference).

## The Beta-Binomial Model

![Bayesian updating prior to posterior belief transformation t](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/08-bayesian-updating-prior-to-posterior-belief-transformation-t.jpg)

This is the canonical example of Bayesian inference. We'll work through it in full detail.

![Bayesian updating animation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/gifs/probstat-08-bayesian-updating.gif)

![Beta-Binomial sequential updating](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/08-beta-binomial.png)

### Setup

You're flipping a coin and want to estimate the probability of heads $\theta$.

- **Likelihood:** $X | \theta \sim \text{Binomial}(n, \theta)$, so $p(x | \theta) = \binom{n}{x} \theta^x (1-\theta)^{n-x}$.
- **Prior:** $\theta \sim \text{Beta}(\alpha, \beta)$, so $p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$.

### Deriving the Posterior
$$p(\theta | x) \propto p(x | \theta) \cdot p(\theta) = \binom{n}{x} \theta^x (1-\theta)^{n-x} \cdot \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)}$$

$$\propto \theta^{x + \alpha - 1} (1-\theta)^{n - x + \beta - 1}.$$
This is the kernel of a Beta distribution:
$$\boxed{\theta | x \sim \text{Beta}(\alpha + x, \beta + n - x)}$$
### Interpretation

The prior parameters $\alpha$ and $\beta$ act as "pseudo-counts" — as if you had already observed $\alpha - 1$ heads and $\beta - 1$ tails before the experiment. The data adds $x$ heads and $n - x$ tails.

The posterior mean is:
$$E[\theta | x] = \frac{\alpha + x}{\alpha + \beta + n}.$$
This is a weighted average of the prior mean $\frac{\alpha}{\alpha + \beta}$ and the sample proportion $\frac{x}{n}$:
$$E[\theta | x] = \frac{\alpha + \beta}{\alpha + \beta + n} \cdot \underbrace{\frac{\alpha}{\alpha + \beta}}_{\text{prior mean}} + \frac{n}{\alpha + \beta + n} \cdot \underbrace{\frac{x}{n}}_{\text{sample proportion}}.$$
As $n \to \infty$, the weight on the prior shrinks to zero, and the posterior mean converges to the MLE $x/n$. **With enough data, the prior is irrelevant.**

### Worked Example

You suspect a coin might be biased. Your prior: $\theta \sim \text{Beta}(2, 2)$ (symmetric, mildly favoring fairness). You flip 10 times and get 7 heads.

**Prior:** $\text{Beta}(2, 2)$. Mean = 0.5.

**Posterior:** $\text{Beta}(2 + 7, 2 + 3) = \text{Beta}(9, 5)$. Mean = $9/14 \approx 0.643$.

**MLE:** $7/10 = 0.7$.

The posterior mean (0.643) is pulled toward the prior mean (0.5) relative to the MLE (0.7). The prior acts as a regularizer, shrinking the estimate toward the center. With more data, this shrinkage weakens.

### Choosing the Prior

| Prior Parameters | Interpretation | Strength |
|---|---|---|
| $\text{Beta}(1, 1)$ | Uniform — no preference | Weak (equivalent to 0 pseudo-observations) |
| $\text{Beta}(0.5, 0.5)$ | Jeffreys prior — emphasizes extremes | Non-informative |
| $\text{Beta}(2, 2)$ | Slight preference for fairness | Weak |
| $\text{Beta}(10, 10)$ | Strong belief the coin is fair | Moderate |
| $\text{Beta}(100, 100)$ | Very strong belief the coin is fair | Strong |

The strength of the prior is controlled by $\alpha + \beta$. Larger values mean more "prior data" and more resistance to updating.

## The Normal-Normal Model

### Setup

Observe $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$ with $\sigma^2$ known. Prior: $\mu \sim \mathcal{N}(\mu_0, \tau^2)$.

### Posterior
$$p(\mu | \mathbf{x}) \propto \exp\left(-\frac{1}{2\sigma^2}\sum(x_i - \mu)^2\right) \cdot \exp\left(-\frac{(\mu - \mu_0)^2}{2\tau^2}\right).$$
Completing the square in $\mu$ (combining the exponents):
$$\mu | \mathbf{x} \sim \mathcal{N}\left(\mu_n, \sigma_n^2\right)$$
where:
$$\mu_n = \frac{\frac{n}{\sigma^2}\bar{x} + \frac{1}{\tau^2}\mu_0}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}}, \qquad \sigma_n^2 = \frac{1}{\frac{n}{\sigma^2} + \frac{1}{\tau^2}}.$$
*Derivation.* The log-posterior (ignoring constants in $\mu$) is:
$$-\frac{1}{2}\left[\frac{n(\mu - \bar{x})^2}{\sigma^2} + \frac{(\mu - \mu_0)^2}{\tau^2}\right]$$

$$= -\frac{1}{2}\left[\left(\frac{n}{\sigma^2} + \frac{1}{\tau^2}\right)\mu^2 - 2\left(\frac{n\bar{x}}{\sigma^2} + \frac{\mu_0}{\tau^2}\right)\mu + \text{const}\right].$$
This is a quadratic in $\mu$, so the posterior is Normal with precision (inverse variance) $\frac{n}{\sigma^2} + \frac{1}{\tau^2}$ and mean given by the precision-weighted average above. $\blacksquare$

### Precision Form

Define **precision** as the reciprocal of variance: $\lambda = 1/\sigma^2$, $\lambda_0 = 1/\tau^2$.
$$\text{Posterior precision} = n\lambda + \lambda_0 \qquad \text{(precisions add)}$$

$$\text{Posterior mean} = \frac{n\lambda \bar{x} + \lambda_0 \mu_0}{n\lambda + \lambda_0} \qquad \text{(precision-weighted average)}$$
Precision is additive, making it the natural parameterization for Bayesian updating with Gaussians.

## Credible Intervals vs Confidence Intervals

A **credible interval** is the Bayesian analog of a confidence interval.

![Credible vs confidence intervals](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/08-credible-vs-confidence.png)

**Definition.** A $100(1-\alpha)\%$ credible interval $[a, b]$ for $\theta$ satisfies:
$$P(\theta \in [a, b] | \mathbf{x}) = 1 - \alpha.$$
The **highest posterior density (HPD) interval** is the shortest such interval: it includes all points with posterior density above some threshold.

### The Crucial Difference

| | Confidence Interval | Credible Interval |
|---|---|---|
| What's random? | The interval (data-dependent) | $\theta$ (parameter is random) |
| Fixed | $\theta$ (unknown constant) | The data (observed) |
| Interpretation | 95% of intervals cover $\theta$ | 95% posterior probability $\theta$ is inside |
| Requires | Sampling distribution | Prior distribution |

For the Normal-Normal model with diffuse prior ($\tau \to \infty$), the 95% credible interval equals the 95% confidence interval. With informative priors or small samples, they differ.

## Point Estimates from the Posterior

The posterior $p(\theta | \mathbf{x})$ is a full distribution. To get a single number, choose:

| Estimator | Definition | Optimal for |
|---|---|---|
| Posterior mean | $E[\theta \mid \mathbf{x}]$ | Minimizes $E[(\hat\theta - \theta)^2 \mid \mathbf{x}]$ (squared error) |
| Posterior median | Median of $p(\theta \mid \mathbf{x})$ | Minimizes $E[\lvert \hat\theta - \theta\rvert \mid \mathbf{x}]$ (absolute error) |
| MAP | $\arg\max_\theta p(\theta \mid \mathbf{x})$ | Mode of posterior (= penalized MLE) |

For symmetric, unimodal posteriors (like the Normal), all three coincide. For skewed posteriors, they differ, and the choice depends on your loss function.

## The Predictive Distribution

Often, we don't care about $\theta$ itself — we want to predict a future observation $\tilde{X}$.

The **posterior predictive distribution** integrates out the parameter:
$$p(\tilde{x} | \mathbf{x}) = \int p(\tilde{x} | \theta) \, p(\theta | \mathbf{x}) \, d\theta.$$
This accounts for **parameter uncertainty**: we don't plug in a single estimate of $\theta$; instead, we average predictions over all possible $\theta$ values, weighted by the posterior.

### Beta-Binomial Predictive

After observing $x$ heads in $n$ flips with posterior $\theta | x \sim \text{Beta}(\alpha + x, \beta + n - x)$:
$$P(\tilde{X} = 1 | x) = E[\theta | x] = \frac{\alpha + x}{\alpha + \beta + n}.$$
This is **Laplace's rule of succession**. With a uniform prior ($\alpha = \beta = 1$) and $x$ successes in $n$ trials:
$$P(\text{next success}) = \frac{x + 1}{n + 2}.$$
If you've seen 7 heads in 10 flips, the predictive probability of heads on the next flip is $8/12 = 2/3$, not the MLE of $7/10$.

### Normal Predictive

With posterior $\mu | \mathbf{x} \sim \mathcal{N}(\mu_n, \sigma_n^2)$ and $\tilde{X} | \mu \sim \mathcal{N}(\mu, \sigma^2)$:
$$\tilde{X} | \mathbf{x} \sim \mathcal{N}(\mu_n, \sigma^2 + \sigma_n^2).$$
The predictive variance $\sigma^2 + \sigma_n^2$ is **larger** than the sampling variance $\sigma^2$ alone, because it includes uncertainty about $\mu$. Plug-in predictions (using $\hat\mu$ as if it were the true $\mu$) underestimate uncertainty.

## MCMC: When Conjugacy Isn't Enough

Most realistic models don't have conjugate priors. The posterior $p(\theta | \mathbf{x})$ is known only up to a normalizing constant: we can evaluate $p(\mathbf{x} | \theta) p(\theta)$ for any $\theta$, but computing $p(\mathbf{x}) = \int p(\mathbf{x}|\theta)p(\theta)d\theta$ is intractable.

![MCMC trace plot](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/08-mcmc-trace.png)

**Markov Chain Monte Carlo (MCMC)** methods generate samples $\theta^{(1)}, \theta^{(2)}, \ldots$ from the posterior without computing the normalizing constant.

### Metropolis-Hastings: The Idea

1. Start at some $\theta^{(0)}$.
2. **Propose** a new value $\theta^*$ from a proposal distribution $q(\theta^* | \theta^{(t)})$.
3. **Accept** with probability:
$$\alpha = \min\left(1, \frac{p(\theta^* | \mathbf{x}) \, q(\theta^{(t)} | \theta^*)}{p(\theta^{(t)} | \mathbf{x}) \, q(\theta^* | \theta^{(t)})}\right) = \min\left(1, \frac{p(\mathbf{x} | \theta^*) p(\theta^*) \, q(\theta^{(t)} | \theta^*)}{p(\mathbf{x} | \theta^{(t)}) p(\theta^{(t)}) \, q(\theta^* | \theta^{(t)})}\right).$$
4. If accepted, $\theta^{(t+1)} = \theta^*$. Otherwise, $\theta^{(t+1)} = \theta^{(t)}$.

The normalizing constant $p(\mathbf{x})$ cancels in the ratio. The resulting Markov chain has $p(\theta | \mathbf{x})$ as its stationary distribution, so after a burn-in period, the samples are (approximately) from the posterior.

With a **symmetric** proposal ($q(\theta^*|\theta) = q(\theta|\theta^*)$), the ratio simplifies to $p(\theta^* | \mathbf{x}) / p(\theta^{(t)} | \mathbf{x})$ — the algorithm simply moves toward higher-density regions while occasionally accepting downhill moves to explore.

## Bayesian Connections to Machine Learning

![Mcmc random walk exploring probability landscape pathfinding](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/08-mcmc-random-walk-exploring-probability-landscape-pathfinding.jpg)

### Regularization as Prior

We saw in Article 6 that MAP estimation with a Gaussian prior yields L2 regularization. Let's make this precise.

**Neural network weight prior:** Place $w \sim \mathcal{N}(0, \tau^2 I)$ on the weight vector. The MAP objective becomes:
$$\hat{w}_{\text{MAP}} = \arg\max_w \left[\sum_{i=1}^n \ln p(y_i | x_i, w) - \frac{\|w\|^2}{2\tau^2}\right]$$

$$= \arg\min_w \left[-\sum_{i=1}^n \ln p(y_i | x_i, w) + \frac{1}{2\tau^2}\|w\|^2\right].$$
This is exactly the loss function with L2 regularization (weight decay), where $\lambda = 1/\tau^2$.

A vague prior ($\tau \to \infty$) gives no regularization (MLE). A strong prior ($\tau$ small) gives strong regularization, pulling weights toward zero.

### Dropout as Approximate Bayesian Inference

Gal and Ghahramani (2016) showed that training with dropout is equivalent to approximate Bayesian inference in a deep Gaussian process. The distribution over predictions at test time (by running multiple forward passes with dropout enabled) approximates the posterior predictive distribution.

### Bayesian Neural Networks

Instead of finding a single weight vector $\hat{w}$, maintain a full posterior $p(w | \mathbf{x}, \mathbf{y})$ and predict by averaging:
$$p(y^* | x^*, \mathbf{x}, \mathbf{y}) = \int p(y^* | x^*, w) \, p(w | \mathbf{x}, \mathbf{y}) \, dw.$$
This is the gold standard for uncertainty quantification, but the integral is intractable for large networks. Practical approaches use variational inference or MCMC approximations.

## When Bayesian and Frequentist Agree

With large samples and well-specified models:

1. The posterior concentrates around the true $\theta$ (Bernstein-von Mises theorem).
2. The posterior mean approaches the MLE.
3. Credible intervals and confidence intervals coincide.
4. The prior becomes irrelevant.

They disagree when:
- Sample sizes are small and the prior matters.
- The parameter space is high-dimensional relative to the data.
- The question involves **current data** (Bayesian: "what should I believe now?") rather than **long-run properties** (frequentist: "how often does this procedure work?").

## Python: Bayesian Updating Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
theta = np.linspace(0, 1, 500)
alpha_prior, beta_prior = 2, 2
n_obs, x_obs = 10, 7

prior = stats.beta.pdf(theta, alpha_prior, beta_prior)
likelihood = theta**x_obs * (1-theta)**(n_obs - x_obs)
likelihood = likelihood / likelihood.max() * prior.max()  # scale for plotting
posterior = stats.beta.pdf(theta, alpha_prior + x_obs, beta_prior + n_obs - x_obs)

ax.plot(theta, prior, 'b-', linewidth=2, label=f'Prior: Beta({alpha_prior},{beta_prior})')
ax.plot(theta, likelihood, 'g--', linewidth=2, label=f'Likelihood (scaled)')
ax.plot(theta, posterior, 'r-', linewidth=2.5,
        label=f'Posterior: Beta({alpha_prior+x_obs},{beta_prior+n_obs-x_obs})')
ax.axvline(x_obs/n_obs, color='gray', linestyle=':', alpha=0.7, label=f'MLE = {x_obs/n_obs}')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('Density')
ax.set_title('Beta-Binomial: 7 heads in 10 flips', fontsize=13)
ax.legend(fontsize=9)

ax = axes[0, 1]
observations = [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]
alpha, beta_param = 1, 1  # Start with uniform prior
colors = plt.cm.viridis(np.linspace(0, 1, len(observations) + 1))

ax.plot(theta, stats.beta.pdf(theta, alpha, beta_param), color=colors[0],
        linewidth=1.5, label='Prior (n=0)')

for i, obs in enumerate(observations):
    alpha += obs
    beta_param += 1 - obs
    if i in [0, 4, 9, 19]:
        ax.plot(theta, stats.beta.pdf(theta, alpha, beta_param),
                color=colors[i+1], linewidth=1.5, label=f'n={i+1}')

ax.set_xlabel('$\\theta$')
ax.set_ylabel('Density')
ax.set_title('Sequential Bayesian Updating', fontsize=13)
ax.legend(fontsize=9)

ax = axes[1, 0]
n_obs, x_obs = 5, 4
priors = [(1, 1, 'Uniform'), (2, 2, 'Beta(2,2)'), (10, 10, 'Beta(10,10)'),
          (0.5, 0.5, 'Jeffreys')]

for a, b, name in priors:
    post = stats.beta.pdf(theta, a + x_obs, b + n_obs - x_obs)
    ax.plot(theta, post, linewidth=2, label=f'Prior: {name}')

ax.axvline(x_obs/n_obs, color='gray', linestyle=':', alpha=0.7, label='MLE')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('Posterior density')
ax.set_title(f'Prior Sensitivity (n={n_obs}, x={x_obs})', fontsize=13)
ax.legend(fontsize=9)

ax = axes[1, 1]
mu_0, tau = 0, 2  # prior
sigma = 1  # known
data = np.array([1.2, 0.8, 1.5, 0.9, 1.1, 1.3, 0.7, 1.4])
n = len(data)
xbar = data.mean()

precision_post = n/sigma**2 + 1/tau**2
sigma_post = 1/np.sqrt(precision_post)
mu_post = (n/sigma**2 * xbar + 1/tau**2 * mu_0) / precision_post

x = np.linspace(-1, 3, 300)
prior_dist = stats.norm.pdf(x, mu_0, tau)
post_dist = stats.norm.pdf(x, mu_post, sigma_post)

ax.plot(x, prior_dist, 'b--', linewidth=1.5, label=f'Prior: N({mu_0}, {tau}$^2$)')
ax.plot(x, post_dist, 'r-', linewidth=2.5,
        label=f'Posterior: N({mu_post:.2f}, {sigma_post:.3f}$^2$)')
ax.axvline(xbar, color='green', linestyle=':', label=f'$\\bar{{x}}$ = {xbar:.2f}')

ci_low = mu_post - 1.96 * sigma_post
ci_high = mu_post + 1.96 * sigma_post
ax.axvspan(ci_low, ci_high, alpha=0.15, color='red', label=f'95% CI: [{ci_low:.2f}, {ci_high:.2f}]')

ax.set_xlabel('$\\mu$')
ax.set_ylabel('Density')
ax.set_title('Normal-Normal Model', fontsize=13)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('bayesian_inference.png', dpi=150)
plt.show()
```

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

n_obs, x_obs = 20, 14
alpha_prior, beta_prior = 2, 2

def log_posterior(theta):
    """Log posterior (up to a constant) for Beta-Binomial."""
    if theta <= 0 or theta >= 1:
        return -np.inf
    log_lik = x_obs * np.log(theta) + (n_obs - x_obs) * np.log(1 - theta)
    log_prior = (alpha_prior - 1) * np.log(theta) + (beta_prior - 1) * np.log(1 - theta)
    return log_lik + log_prior

n_samples = 50000
samples = np.zeros(n_samples)
samples[0] = 0.5
accepted = 0

for i in range(1, n_samples):
    # Propose from normal centered at current value
    proposal = samples[i-1] + np.random.normal(0, 0.1)

    # Acceptance ratio
    log_alpha = log_posterior(proposal) - log_posterior(samples[i-1])

    if np.log(np.random.uniform()) < log_alpha:
        samples[i] = proposal
        accepted += 1
    else:
        samples[i] = samples[i-1]

burn_in = 5000
posterior_samples = samples[burn_in:]

print(f"Acceptance rate: {accepted/n_samples:.3f}")
print(f"MCMC posterior mean: {posterior_samples.mean():.4f}")
print(f"Exact posterior mean: {(alpha_prior + x_obs)/(alpha_prior + beta_prior + n_obs):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
ax.plot(samples[:2000], 'b-', alpha=0.7, linewidth=0.5)
ax.axhline(y=(alpha_prior + x_obs)/(alpha_prior + beta_prior + n_obs),
           color='red', linestyle='--', label='True mean')
ax.set_xlabel('Iteration')
ax.set_ylabel('$\\theta$')
ax.set_title('MCMC Trace Plot', fontsize=13)
ax.legend()

ax = axes[1]
theta = np.linspace(0, 1, 200)
exact = stats.beta.pdf(theta, alpha_prior + x_obs, beta_prior + n_obs - x_obs)
ax.hist(posterior_samples, bins=60, density=True, alpha=0.7,
        color='steelblue', edgecolor='white', label='MCMC samples')
ax.plot(theta, exact, 'r-', linewidth=2.5, label='Exact Beta posterior')
ax.set_xlabel('$\\theta$')
ax.set_ylabel('Density')
ax.set_title('MCMC vs Exact Posterior', fontsize=13)
ax.legend()

ax = axes[2]
max_lag = 50
acf = np.correlate(posterior_samples - posterior_samples.mean(),
                    posterior_samples - posterior_samples.mean(), mode='full')
acf = acf[len(acf)//2:]
acf = acf / acf[0]
ax.bar(range(max_lag), acf[:max_lag], color='steelblue', alpha=0.7)
ax.set_xlabel('Lag')
ax.set_ylabel('Autocorrelation')
ax.set_title('MCMC Autocorrelation', fontsize=13)
ax.axhline(0, color='gray', linestyle='-')

plt.tight_layout()
plt.savefig('mcmc_demo.png', dpi=150)
plt.show()
```

The MCMC histogram closely matches the exact Beta posterior, verifying that the sampler works. The trace plot shows the chain exploring the parameter space, and the autocorrelation plot reveals how quickly successive samples become independent (shorter correlation = more efficient sampling).

## The Series in Retrospect

Over eight articles, we've built probability and statistics from the ground up:

1. **Axioms** gave us a rigorous foundation for measuring uncertainty.
2. **Random variables** translated outcomes into numbers we can compute with.
3. **Expectations and moments** compressed distributions into summaries.
4. **Joint distributions** handled multiple variables and their dependencies.
5. **Limit theorems** explained why sample averages work and why the Gaussian is universal.
6. **Estimation theory** showed how to extract parameters from data, optimally.
7. **Hypothesis testing** provided tools for making decisions under uncertainty.
8. **Bayesian inference** offered a coherent framework for updating beliefs with data.

Each article built on the previous ones, and together they form the mathematical backbone of modern data science and machine learning. The distributions, theorems, and techniques covered here are not historical curiosities — they are the daily tools of anyone building systems that learn from data.

The path forward leads in many directions: multivariate analysis, time series, causal inference, information theory, statistical learning theory. But the foundations laid in this series are sufficient to read any of those topics with confidence.

## Summary
| Concept | Formula | Role |
|---|---|---|
| Bayes' rule | $p(\theta\lvert \mathbf{x}) \propto p(\mathbf{x}\rvert\theta)p(\theta)$ | Fundamental update |
| Conjugate prior | Prior and posterior in same family | Closed-form posterior |
| Beta-Binomial | $\text{Beta}(\alpha+x, \beta+n-x)$ | Estimating proportions |
| Normal-Normal | Precision-weighted average | Estimating means |
| Posterior mean | $E[\theta\mid \mathbf{x}]$ | Optimal under squared loss |
| MAP | $\arg\max p(\theta\mid \mathbf{x})$ | = MLE + regularization |
| Credible interval | $P(\theta \in [a,b]\mid \mathbf{x}) = 0.95$ | Direct probability statement |
| Predictive | $p(\tilde{x}\lvert \mathbf{x}) = \int p(\tilde{x}\rvert\theta)p(\theta\mid \mathbf{x})d\theta$ | Prediction with uncertainty |
| MCMC | Sample from posterior | When no closed form |
