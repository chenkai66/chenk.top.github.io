---
title: "ML Math Derivations (3): Probability Theory and Statistical Inference"
date: 2026-01-22 09:00:00
tags:
  - Machine Learning
  - Probability Theory
  - Statistical Inference
  - Maximum Likelihood Estimation
  - Bayesian
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 3
  total: 20
lang: en
mathjax: true
description: "Machine learning is uncertainty modeling. This article derives probability spaces, common distributions, MLE, Bayesian estimation, limit theorems and information theory -- the statistical engine behind every ML model."
disableNunjucks: true
series_order: 3
---

## What This Article Covers

In 1912, Ronald Fisher introduced **maximum likelihood estimation** in a short paper that quietly redefined statistics. His insight was almost embarrassingly simple: *if a parameter setting makes the observed data extremely likely, that parameter setting is probably right*. Almost every modern learning algorithm — from logistic regression to large language models — is a descendant of this idea.

But likelihood alone is not enough. To use it we need a vocabulary for uncertainty (probability spaces, distributions), guarantees that empirical quantities track population ones (laws of large numbers, central limit theorem), and tools for incorporating prior knowledge (Bayesian inference). This article assembles those pieces into a coherent foundation for everything that follows.

**What you will learn**

1. **Probability spaces and Bayes' theorem** — the axiomatic foundation that lets us say "with probability $p$" precisely
2. **Common distributions** (Bernoulli, Gaussian, Beta, Poisson, Dirichlet, …) — and *why* certain shapes appear over and over in ML
3. **Concentration and limit theorems** — Markov, Chebyshev, LLN, CLT — the reason finite samples can teach us about populations
4. **Maximum Likelihood Estimation** — what training a model actually optimizes
5. **Bayesian estimation** — how priors arise, why MAP is regularized MLE, and how conjugacy makes the math tractable
6. **Hypothesis testing and confidence intervals** — the geometry of $\alpha$, $\beta$, and coverage
7. **Information theory** — entropy, KL divergence, mutual information; the bridge to cross-entropy loss

**Prerequisites:** calculus (integration, Taylor series), basic probability (random variables, expectation, variance), and a little linear algebra for the multivariate Gaussian.

---

## 1. Probability Spaces

### 1.1 The Kolmogorov Axioms

Probability theory rests on a triplet $(\Omega, \mathcal{F}, P)$.

- **Sample space** $\Omega$ — the set of all possible outcomes of an experiment.
- **$\sigma$-algebra** $\mathcal{F}$ — the collection of "events" we are allowed to assign probabilities to. It is closed under complement and countable union.
- **Probability measure** $P : \mathcal{F} \to [0, 1]$ — satisfies non-negativity, normalization $P(\Omega) = 1$, and countable additivity for disjoint events.

*Why a $\sigma$-algebra and not all subsets?* Because for an uncountable $\Omega$ — say, $[0,1]$ — there are pathological subsets (Vitali sets) on which no translation-invariant probability can be defined consistently. Restricting to a $\sigma$-algebra is the cost of avoiding paradoxes.

### 1.2 Conditional Probability and Bayes' Theorem

For any event $B$ with $P(B) > 0$,

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}. \tag{1}
$$

Reading this in two directions and equating gives **Bayes' theorem**:

$$
\boxed{\, P(\theta \mid D) = \frac{P(D \mid \theta)\,P(\theta)}{P(D)} \,} \tag{2}
$$

| Term | Name | Role |
|------|------|------|
| $P(\theta)$ | Prior | What you believed about $\theta$ before seeing data |
| $P(D \mid \theta)$ | Likelihood | How well $\theta$ explains the data |
| $P(\theta \mid D)$ | Posterior | Updated belief after seeing the data |
| $P(D)$ | Evidence | Normalizing constant, $\int P(D \mid \theta)\,P(\theta)\,d\theta$ |

Bayes' theorem is the *learning rule of probability*: it tells you exactly how a rational observer should revise beliefs in light of evidence. Every Bayesian model — from spam filters to Gaussian processes — is an application of this single formula.

### 1.3 Independence

Events $A$ and $B$ are **independent** when $P(A \cap B) = P(A) P(B)$. They are **conditionally independent given $C$** when $P(A \cap B \mid C) = P(A \mid C)\,P(B \mid C)$.

A common pitfall: independence and conditional independence do not imply each other. Two coins are independent unconditionally, yet become dependent once you condition on "exactly one head". Most graphical-model intuition lives in this distinction.

---

## 2. Random Variables, Expectation, and Variance

A **random variable** $X$ is a measurable function $X : \Omega \to \mathbb{R}$. Its distribution is summarized by:

- **CDF**: $F(x) = P(X \le x)$ — non-decreasing and right-continuous.
- **PDF** (continuous): $f(x) = F'(x)$, with $P(a \le X \le b) = \int_a^b f(x)\,dx$.
- **PMF** (discrete): $p(x) = P(X = x)$.

**Expectation** is the probabilistic analog of the center of mass:

$$
\mathbb{E}[X] = \int x\,f(x)\,dx \quad\text{(continuous)}, \qquad \mathbb{E}[X] = \sum_x x\,p(x)\quad\text{(discrete)}. \tag{3}
$$

The single most useful property is **linearity**:

$$
\mathbb{E}[aX + bY] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y].
$$

It holds *always* — even when $X$ and $Y$ are dependent. This is what lets us decompose variances of binomials, derive bias-variance trade-offs, and analyze SGD updates.

**Variance** captures spread:

$$
\mathrm{Var}(X) = \mathbb{E}\!\left[(X - \mathbb{E}[X])^2\right] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2. \tag{4}
$$

For *independent* $X$ and $Y$, $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$ — the additivity that makes the Central Limit Theorem possible.

**Covariance and correlation** measure linear association:

$$
\mathrm{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)], \qquad \rho(X, Y) = \frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y}. \tag{5}
$$

Cauchy–Schwarz gives $|\rho| \le 1$. **Warning:** $\rho = 0$ means *uncorrelated*, not *independent*. The classical counter-example is $X \sim \mathcal{N}(0,1)$ and $Y = X^2$: $\mathrm{Cov}(X, Y) = \mathbb{E}[X^3] = 0$, yet $Y$ is a deterministic function of $X$. The exception is jointly Gaussian variables, where uncorrelated does imply independent.

---

## 3. Common Probability Distributions

A handful of distributions reappear constantly because they are either (a) the natural model of a physical mechanism, (b) the maximum-entropy distribution given some constraint, or (c) the conjugate prior to another well-loved distribution. The figure below shows the six families that cover most of what you meet in ML.

![Common probability distributions used throughout machine learning](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/03-Probability-Theory-and-Statistical-Inference/fig1_distributions.png)

### 3.1 Discrete Distributions

**Bernoulli** $X \sim \mathrm{Bern}(p)$ — a single binary trial:

$$
P(X = k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}. \tag{6}
$$

$\mathbb{E}[X] = p$ and $\mathrm{Var}(X) = p(1-p)$. This is the output distribution of every binary classifier — logistic regression learns its parameter $p$.

**Binomial** $X \sim \mathrm{Bin}(n, p)$ — count of successes in $n$ independent Bernoulli trials:

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}. \tag{7}
$$

Writing $X = \sum_{i=1}^n X_i$ with $X_i \sim \mathrm{Bern}(p)$ and using linearity gives $\mathbb{E}[X] = np$, $\mathrm{Var}(X) = np(1-p)$ in two lines.

**Poisson** $X \sim \mathrm{Poi}(\lambda)$ — counts of independent rare events in a fixed interval:

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \qquad \mathbb{E}[X] = \mathrm{Var}(X) = \lambda. \tag{8}
$$

It is the limit of $\mathrm{Bin}(n, \lambda / n)$ as $n \to \infty$, which is why it shows up for click counts, server hits, and photon arrivals.

### 3.2 Continuous Distributions

**Gaussian** $X \sim \mathcal{N}(\mu, \sigma^2)$ — the most important distribution in ML:

$$
f(x) = \frac{1}{\sqrt{2\pi}\,\sigma}\,\exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right). \tag{9}
$$

Why is the Gaussian everywhere?

1. **Central Limit Theorem** — sums of many small independent effects are approximately Gaussian, regardless of the underlying distribution.
2. **Maximum entropy** — among all distributions with a fixed mean and variance, the Gaussian carries the most uncertainty (assumes the least).
3. **Closure properties** — Gaussians stay Gaussian under linear transformations, marginalization, and conditioning. Every operation in a Kalman filter, linear regression posterior, or VAE relies on this.

The **multivariate Gaussian** $X \sim \mathcal{N}(\mu, \Sigma)$ has density

$$
f(x) = \frac{1}{(2\pi)^{d/2}\,|\Sigma|^{1/2}}\,\exp\!\left(-\tfrac{1}{2}(x - \mu)^\top \Sigma^{-1}(x - \mu)\right). \tag{10}
$$

Its level sets are ellipsoids whose principal axes are the eigenvectors of $\Sigma$ — exactly the geometry behind PCA.

**Exponential** $X \sim \mathrm{Exp}(\lambda)$ has density $f(x) = \lambda e^{-\lambda x}$ for $x \ge 0$ and the **memoryless property**: $P(X > s + t \mid X > s) = P(X > t)$. It models waiting times in a Poisson process.

**Beta** $X \sim \mathrm{Beta}(\alpha, \beta)$ lives on $[0, 1]$ and is the conjugate prior for Bernoulli/binomial. Its mean is $\alpha / (\alpha + \beta)$, and its shape ranges from U-shaped ($\alpha, \beta < 1$) to bell-shaped ($\alpha, \beta > 1$).

**Gamma** $X \sim \mathrm{Gamma}(k, \theta)$ generalizes the exponential ($k = 1$) and chi-squared ($k = n/2$, $\theta = 2$) distributions. It is the conjugate prior for the rate of a Poisson.

**Dirichlet** $X \sim \mathrm{Dir}(\alpha)$ generalizes the Beta to the $K$-simplex; it is the conjugate prior for the categorical/multinomial. The simplex panel in the figure above shows how concentration parameters $(3, 5, 2)$ pull mass toward the $x_2$ vertex.

---

## 4. Limit Theorems: Why ML Works at Scale

### 4.1 Concentration Inequalities

These are crude but always-true bounds — useful when you know nothing but a moment.

**Markov's inequality.** For non-negative $X$ and $a > 0$:
$$
P(X \ge a) \le \frac{\mathbb{E}[X]}{a}. \tag{11}
$$

**Chebyshev's inequality.** Apply Markov to $(X - \mu)^2$:

$$
P(|X - \mu| \ge k) \le \frac{\sigma^2}{k^2}. \tag{12}
$$

These two inequalities are the seeds of every PAC-style learning bound.

### 4.2 Law of Large Numbers

For i.i.d. $X_1, \ldots, X_n$ with mean $\mu$ and finite variance $\sigma^2$, the **weak law** states

$$
P(|\bar{X}_n - \mu| > \epsilon) \le \frac{\sigma^2}{n\epsilon^2} \;\longrightarrow\; 0. \tag{13}
$$

In words: *empirical averages concentrate around true expectations*. This is the formal reason that empirical risk minimization — picking a model that fits training data well — has any hope of generalizing.

### 4.3 Central Limit Theorem

LLN tells us the sample mean converges; CLT tells us *how fast and to what shape*.

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1). \tag{14}
$$

The convergence is striking: even a wildly skewed underlying distribution produces Gaussian-shaped averages. The figure below uses $\mathrm{Exp}(1)$ — about as asymmetric as it gets — and shows the standardized mean morphing into $\mathcal{N}(0,1)$ as $n$ grows.

![Standardized means of Exponential(1) converge to N(0,1) as n grows](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/03-Probability-Theory-and-Statistical-Inference/fig4_clt.png)

*Proof sketch* (characteristic functions): Taylor-expand the characteristic function of the standardized $X_i$ around 0: $\phi(t) = 1 - t^2/2 + o(t^2)$. The standardized mean has characteristic function $\phi(t/\sqrt{n})^n \approx (1 - t^2/(2n))^n \to e^{-t^2/2}$, which is the characteristic function of $\mathcal{N}(0, 1)$. Apply Lévy's continuity theorem.

**Why CLT matters for ML.** It justifies confidence intervals on validation loss, t-tests on A/B experiments, the Gaussian noise assumption in linear regression, and the asymptotic normality of MLE that we will derive next.

---

## 5. Parameter Estimation

### 5.1 Properties of Estimators

An **estimator** $\hat{\theta}_n$ is any function of the sample. Three properties matter most:

- **Unbiased**: $\mathbb{E}[\hat{\theta}_n] = \theta$.
- **Consistent**: $\hat{\theta}_n \xrightarrow{P} \theta$ as $n \to \infty$.
- **Mean-squared error decomposition**: $\mathrm{MSE}(\hat{\theta}) = \mathrm{Bias}(\hat{\theta})^2 + \mathrm{Var}(\hat{\theta})$.

The famous *bias–variance trade-off* in ML is precisely this decomposition applied to predictions instead of parameters.

**Why does sample variance divide by $n - 1$?** Because the naive $\frac{1}{n}\sum (X_i - \bar{X})^2$ is biased. Expanding,
$$
\mathbb{E}\!\left[\frac{1}{n}\sum (X_i - \bar{X})^2\right] = \frac{n - 1}{n}\,\sigma^2.
$$
Dividing by $n - 1$ corrects the bias. The intuition: estimating $\bar{X}$ from the same data "uses up" one degree of freedom, leaving $n - 1$ effectively independent residuals.

### 5.2 Maximum Likelihood Estimation (MLE)

Given i.i.d. observations $x_1, \ldots, x_n$ from a model $f(x; \theta)$, the **likelihood** is

$$
L(\theta) = \prod_{i=1}^n f(x_i; \theta), \qquad \ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i; \theta). \tag{15}
$$

The MLE is $\hat{\theta}_{\mathrm{MLE}} = \arg\max_\theta \ell(\theta)$.

**Example 1 — Bernoulli.** If $x_i \in \{0, 1\}$,
$$
\ell(p) = \left(\sum x_i\right) \log p + \left(n - \sum x_i\right) \log(1 - p).
$$
Setting $\ell'(p) = 0$ gives $\hat{p}_{\mathrm{MLE}} = \bar{x}$ — the empirical fraction of successes. Logistic regression is essentially this same computation, parameterized by features.

**Example 2 — Gaussian.** For $x_i \sim \mathcal{N}(\mu, \sigma^2)$,
$$
\hat{\mu}_{\mathrm{MLE}} = \bar{x}, \qquad \hat{\sigma}^2_{\mathrm{MLE}} = \frac{1}{n}\sum (x_i - \bar{x})^2. \tag{16}
$$
Note that $\hat{\sigma}^2_{\mathrm{MLE}}$ is *biased* — it divides by $n$, not $n - 1$. MLE optimizes likelihood, not unbiasedness.

**Asymptotic guarantees.** Under mild regularity conditions, the MLE is

1. **Consistent**: $\hat{\theta}_n \to \theta_0$ in probability.
2. **Asymptotically normal**: $\sqrt{n}(\hat{\theta}_n - \theta_0) \xrightarrow{d} \mathcal{N}(0,\, I(\theta_0)^{-1})$.
3. **Efficient**: its asymptotic variance achieves the Cramér–Rao lower bound.

Here $I(\theta) = -\mathbb{E}[\partial^2 \ell / \partial \theta^2]$ is the **Fisher information**: how curved (and therefore how informative) the log-likelihood is at the truth.

### 5.3 Bayesian Estimation

The Bayesian view treats $\theta$ as a random variable with prior $P(\theta)$. Bayes' theorem then defines the posterior:

$$
P(\theta \mid D) \propto P(D \mid \theta)\,P(\theta). \tag{17}
$$

The figure below walks through one such update for a Beta–Bernoulli model — prior, likelihood and posterior side by side.

![Bayesian update: prior multiplied by likelihood gives posterior](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/03-Probability-Theory-and-Statistical-Inference/fig2_bayes_update.png)

**Beta–Bernoulli conjugacy.** With prior $p \sim \mathrm{Beta}(\alpha, \beta)$ and data of $k$ heads in $n$ flips,
$$
p \mid D \sim \mathrm{Beta}(\alpha + k,\;\beta + n - k). \tag{18}
$$
The posterior mean,
$$
\hat{p}_{\mathrm{Bayes}} = \frac{\alpha + k}{\alpha + \beta + n},
$$
is the MLE plus "pseudo-observations" $\alpha$ and $\beta$ contributed by the prior. As $n \to \infty$ the posterior mean converges to the MLE — data overwhelms the prior in the limit.

**MAP (Maximum A Posteriori)** estimation takes the posterior mode:

$$
\hat{\theta}_{\mathrm{MAP}} = \arg\max_\theta \left[\log P(D \mid \theta) + \log P(\theta)\right]. \tag{19}
$$

This is exactly **regularized MLE**. With a Gaussian prior $\theta \sim \mathcal{N}(0, \tau^2 I)$, MAP for a linear regression model becomes ridge regression; with a Laplace prior, you get lasso.

When data is plentiful, MLE, MAP, and the full posterior agree. When data is scarce, they can disagree dramatically — and the figure below shows the three answers side by side for the same $4$-out-of-$5$ flips.

![MLE, MAP and full Bayesian posterior on the same small-sample data](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/03-Probability-Theory-and-Statistical-Inference/fig3_mle_vs_map.png)

| Feature | Frequentist (MLE) | Bayesian |
|---------|-------------------|----------|
| Parameter | Fixed but unknown | Random variable |
| Prior knowledge | Not used | Explicitly modeled |
| Output | Point estimate | Full posterior distribution |
| Uncertainty | Confidence interval | Credible interval |
| Computation | Usually closed form | May need MCMC or VI |

---

## 6. Hypothesis Testing and Confidence Intervals

### 6.1 Hypothesis Testing

A test pits a **null hypothesis** $H_0$ against an **alternative** $H_1$. We compute a test statistic $T$ and reject $H_0$ when $T$ falls in a pre-specified rejection region. Two kinds of errors are possible:

| Decision \ Truth | $H_0$ true | $H_0$ false |
|------------------|------------|-------------|
| Accept $H_0$ | Correct | Type II error ($\beta$) |
| Reject $H_0$ | Type I error ($\alpha$) | Correct (power $1 - \beta$) |

The **significance level** $\alpha$ is the probability of a Type I error — we choose it (typically 0.05) and design the test to keep it bounded. The **power** $1 - \beta$ depends on the alternative and on sample size.

The figure below visualizes the tradeoff. Two competing distributions (the world under $H_0$ vs $H_1$) are shown together with the threshold $c$. Moving $c$ left shrinks $\beta$ but inflates $\alpha$, and vice versa — there is no free lunch unless we collect more data.

![Type I (alpha) and Type II (beta) errors visualised on overlapping null and alternative distributions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/03-Probability-Theory-and-Statistical-Inference/fig6_hypothesis_errors.png)

**p-value.** The probability, under $H_0$, of observing data at least as extreme as what we saw. Reject $H_0$ when $p < \alpha$.

**Example — one-sample t-test.** Test $H_0 : \mu = \mu_0$ when $\sigma$ is unknown:

$$
t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n - 1} \quad\text{under } H_0. \tag{20}
$$

### 6.2 Confidence Intervals

A $(1 - \alpha)$ confidence interval for the mean (with known $\sigma$) is

$$
\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}. \tag{21}
$$

**The wording matters.** The probability $1 - \alpha$ refers to the *procedure*, not the parameter. If we repeated the experiment many times, about $(1 - \alpha)$ of the resulting intervals would cover the true $\mu$. Any single interval either contains $\mu$ or it does not.

The figure below makes this concrete: 50 simulated 95% CIs from the same data-generating process, colored by whether they happened to cover the true mean.

![Fifty simulated 95% confidence intervals -- about 95% cover the true parameter](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/03-Probability-Theory-and-Statistical-Inference/fig5_confidence_intervals.png)

If you want to say "the parameter is in this interval with probability 95%", you need a Bayesian credible interval — the quantiles of a posterior.

---

## 7. A Tour of Information Theory

Information theory is the bridge between probability and learning objectives. Three quantities matter most.

![Entropy, KL divergence and mutual information](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/03-Probability-Theory-and-Statistical-Inference/fig7_information_theory.png)

**Entropy** measures expected surprise. For a discrete distribution $p$,

$$
H(p) = -\sum_x p(x) \log p(x).
$$

The Bernoulli entropy peaks at $p = 0.5$ — a fair coin is maximally unpredictable. *Cross-entropy loss* is just $H(p, q) = -\sum p(x) \log q(x)$ between the true label distribution and the model's predicted distribution.

**KL divergence** $D_{\mathrm{KL}}(P \,\|\, Q) = \sum_x p(x) \log \tfrac{p(x)}{q(x)}$ measures how much information is lost when $Q$ is used to approximate $P$. It is non-negative, zero iff $P = Q$, and *asymmetric* — the figure shows $D_{\mathrm{KL}}(P \| Q) \neq D_{\mathrm{KL}}(Q \| P)$ for two displaced Gaussians. Variational inference and KL-regularized policies all hinge on this asymmetry.

**Mutual information** $I(X; Y) = D_{\mathrm{KL}}\bigl(P(X, Y) \,\|\, P(X) P(Y)\bigr)$ measures dependence. For a bivariate Gaussian, $I(X; Y) = -\tfrac{1}{2} \log(1 - \rho^2)$ — independence at $\rho = 0$, infinite information as $|\rho| \to 1$. Mutual information generalizes correlation to arbitrary (including non-linear) dependence.

---

## 8. Exercises

**Exercise 1 (Base rate fallacy).** A disease has prevalence 0.1%. A test has sensitivity 99% and specificity 95%. Given a positive result, what is $P(\text{disease})$?

*Solution.* By Bayes,
$$
P(D \mid +) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.05 \times 0.999} \approx 0.0194.
$$
Only about 2% — low base rate dominates. This is why screening tests for rare conditions need extremely high specificity to be useful.

**Exercise 2 (MLE for Uniform).** $X_i \sim \mathrm{Uniform}(0, \theta)$, find the MLE.

*Solution.* The likelihood $L(\theta) = \theta^{-n}$ for $\theta \ge X_{(n)}$ and 0 otherwise. So $\hat\theta_{\mathrm{MLE}} = X_{(n)} = \max_i X_i$. It is biased: $\mathbb{E}[X_{(n)}] = \tfrac{n}{n+1}\theta$. The unbiased correction is $\tfrac{n+1}{n} X_{(n)}$.

**Exercise 3 (CLT in practice).** Screws have $\mu = 10$ mm, $\sigma = 0.2$ mm. With $n = 100$, find $P(9.96 \le \bar{X} \le 10.04)$.

*Solution.* By CLT, $\bar X \approx \mathcal N(10, 0.02^2)$. Standardizing, $P(|Z| \le 2) \approx 0.9544$.

**Exercise 4 (Bayesian coin).** Prior $p \sim \mathrm{Beta}(2, 2)$. Observe 7 heads in 10 flips. Posterior?

*Solution.* By conjugacy, $p \mid D \sim \mathrm{Beta}(9, 5)$. Posterior mean $9/14 \approx 0.643$ — the prior pulls the MLE of $0.7$ toward $0.5$.

**Exercise 5 (Hypothesis test).** Factory claims $\mu = 500$ g. Sample: $n = 25$, $\bar x = 498$, $s = 10$. Test at $\alpha = 0.05$.

*Solution.* $t = (498 - 500)/(10/5) = -1.0$. Critical value $t_{0.025, 24} \approx 2.064$. Since $|t| < 2.064$, fail to reject $H_0$. *Caveat:* failing to reject is not evidence *for* $H_0$ — the test simply lacked the power to detect a 2 g effect at $n = 25$.

---

## Summary

| Concept | Key formula | ML connection |
|---------|-------------|---------------|
| Bayes' theorem | $P(\theta \mid D) \propto P(D \mid \theta)\,P(\theta)$ | Foundation of Bayesian ML |
| MLE | $\hat\theta = \arg\max \sum \log f(x_i; \theta)$ | What model training optimizes |
| MAP / regularization | MAP = MLE + log-prior | Ridge ↔ Gaussian prior, Lasso ↔ Laplace |
| CLT | $\bar X_n \approx \mathcal{N}(\mu, \sigma^2/n)$ | Confidence intervals, t-tests, asymptotic MLE |
| Conjugate priors | Beta–Bernoulli, Gamma–Poisson, Gaussian–Gaussian | Closed-form posteriors |
| Cross-entropy | $-\sum p(x) \log q(x)$ | Classification loss |
| KL divergence | $\sum p \log(p/q)$ | VI, distillation, RL regularizers |

Memorize these five and the rest is interpolation:

> **Bayes** updates beliefs (posterior ∝ likelihood × prior).  
> **LLN** says averages converge.  
> **CLT** says they converge to a Gaussian.  
> **MLE** is asymptotically efficient.  
> **MAP** is regularized MLE.

---

## References

1. Casella, G. & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.
3. Murphy, K. P. (2022). *Probabilistic Machine Learning: An Introduction.* MIT Press.
4. Wasserman, L. (2004). *All of Statistics.* Springer.
5. MacKay, D. J. C. (2003). *Information Theory, Inference, and Learning Algorithms.* Cambridge University Press.
6. Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.

---

## Series Navigation

| Part | Topic | Link |
|------|-------|------|
| 1 | Introduction and Mathematical Foundations | [<-- Read](/en/Machine-Learning-Mathematical-Derivations-1-Introduction-and-Mathematical-Foundations/) |
| 2 | Linear Algebra and Matrix Theory | [<-- Previous](/en/Machine-Learning-Mathematical-Derivations-2-Linear-Algebra-and-Matrix-Theory/) |
| **3** | **Probability Theory and Statistical Inference** | *You are here* |
| 4 | Convex Optimization Theory | [Read next -->](/en/Machine-Learning-Mathematical-Derivations-4-Convex-Optimization-Theory/) |
| 5 | Linear Regression | [Read -->](/en/Machine-Learning-Mathematical-Derivations-5-Linear-Regression/) |
