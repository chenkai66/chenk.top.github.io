---
title: "Probability and Statistics (7): Hypothesis Testing — p-Values, Confidence Intervals, and All Their Pitfalls"
date: 2024-08-28 09:00:00
tags:
  - Probability
  - Statistics
  - Hypothesis Testing
  - Confidence Intervals
  - A/B Testing
categories: Probability and Statistics
series: probability-statistics
lang: en
mathjax: true
description: "A rigorous treatment of hypothesis testing, p-values, Type I/II errors, confidence intervals, and multiple testing corrections — including the misinterpretations that trip up even experienced practitioners, with Python implementations."
disableNunjucks: true
series_order: 7
series_total: 8
translationKey: "probability-statistics-7"
---

You've estimated a parameter. You've quantified the bias-variance tradeoff. Now comes the question that drives most applied statistics: "Is this effect real, or just noise?"

Hypothesis testing is the formal framework for answering this question. It's also the most widely misunderstood part of statistics. Entire papers have been written about how researchers misinterpret p-values, how significance thresholds are arbitrary, and how the multiple testing problem inflates false discoveries. Understanding both the theory and the pitfalls is essential for anyone who works with data.

---

## The Framework

![Statistical power curve](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/07-power-curve.png)

### Setting Up the Hypotheses

A hypothesis test begins with two competing claims:

- **Null hypothesis** $H_0$: The "default" or "nothing special is happening" claim. Usually the status quo.
- **Alternative hypothesis** $H_1$ (or $H_a$): What we'd like to show is true.

**Examples:**
- Drug trial: $H_0: \mu_{\text{drug}} = \mu_{\text{placebo}}$ vs $H_1: \mu_{\text{drug}} > \mu_{\text{placebo}}$
- A/B test: $H_0: p_A = p_B$ vs $H_1: p_A \neq p_B$
- Quality control: $H_0: \mu = 10.0$ vs $H_1: \mu \neq 10.0$

The alternative can be **one-sided** ($H_1: \theta > \theta_0$) or **two-sided** ($H_1: \theta \neq \theta_0$).

### Test Statistics and Rejection Regions

A **test statistic** $T = T(X_1, \ldots, X_n)$ summarizes the data into a single number that measures evidence against $H_0$.

![Rejection regions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/07-rejection-region.png)

The **rejection region** $R$ is a set of values of $T$ for which we reject $H_0$:

- If $T \in R$: reject $H_0$
- If $T \notin R$: fail to reject $H_0$ (this is **not** the same as accepting $H_0$)

### Significance Level

The **significance level** $\alpha$ is the maximum probability of rejecting $H_0$ when it is actually true:
$$\alpha = P(\text{reject } H_0 \mid H_0 \text{ is true}) = P(T \in R \mid H_0).$$
Common choices: $\alpha = 0.05, 0.01, 0.001$. The rejection region is chosen to satisfy this constraint.

## Errors in Hypothesis Testing

| | $H_0$ is true | $H_0$ is false |
|---|---|---|
| **Reject $H_0$** | Type I error (false positive) | Correct (true positive) |
| **Fail to reject $H_0$** | Correct (true negative) | Type II error (false negative) |

![Type I and Type II errors](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/07-error-types.png)

- **Type I error rate** = $\alpha$ = P(reject $H_0$ | $H_0$ true)
- **Type II error rate** = $\beta$ = P(fail to reject $H_0$ | $H_0$ false)
- **Power** = $1 - \beta$ = P(reject $H_0$ | $H_0$ false)

We control Type I error directly (by choosing $\alpha$). Power depends on:
- The true effect size (larger effects are easier to detect)
- The sample size $n$ (more data = more power)
- The significance level $\alpha$ (more lenient threshold = more power, but also more false positives)
- The variance $\sigma^2$ (less noise = more power)

### The Tradeoff

Decreasing $\alpha$ reduces false positives but increases false negatives ($\beta$ goes up, power goes down). You can't minimize both simultaneously with a fixed sample size. The only free lunch is more data.

## The p-Value

![Ab testing laboratory two beakers being compared scientific](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/07-ab-testing-laboratory-two-beakers-being-compared-scientific-.jpg)

### Definition

The **p-value** is the probability, under $H_0$, of observing a test statistic at least as extreme as the one actually observed:
$$p\text{-value} = P(T \geq t_{\text{obs}} \mid H_0) \quad \text{(one-sided)}$$

$$p\text{-value} = P(|T| \geq |t_{\text{obs}}| \mid H_0) \quad \text{(two-sided)}$$
**Decision rule:** Reject $H_0$ if $p\text{-value} \leq \alpha$.

The p-value is a random variable (it depends on the data). Under $H_0$, the p-value has a $\text{Uniform}(0, 1)$ distribution — this is why $P(p\text{-value} \leq 0.05 \mid H_0) = 0.05$.

### What the p-Value IS

The p-value is: the probability of seeing data this extreme or more extreme, **assuming $H_0$ is true**.

### What the p-Value IS NOT

These are common and dangerous misinterpretations:

1. **NOT** the probability that $H_0$ is true. ($P(H_0 | \text{data})$ requires Bayes' theorem and a prior.)
2. **NOT** the probability that the result is due to chance. (The p-value is computed **assuming** the result is due to chance.)
3. **NOT** the probability of making an error. (The error rate is $\alpha$, not the p-value.)
4. **NOT** a measure of effect size. (A tiny, meaningless effect can have $p < 0.001$ with enough data.)
5. **NOT** a measure of replicability. (A $p = 0.04$ result does not have a 96% chance of replicating.)

## Common Tests

### The Z-Test

When $\sigma$ is known and the population is Normal (or $n$ is large enough for the CLT):
$$Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1) \quad \text{under } H_0: \mu = \mu_0.$$
**Example.** A factory claims bolts have mean length 10.0 mm with known $\sigma = 0.5$ mm. A sample of $n = 25$ gives $\bar{x} = 10.2$. Test $H_0: \mu = 10$ vs $H_1: \mu \neq 10$ at $\alpha = 0.05$.
$$Z = \frac{10.2 - 10.0}{0.5/\sqrt{25}} = \frac{0.2}{0.1} = 2.0.$$

$p\text{-value} = 2 \cdot P(Z > 2.0) = 2 \times 0.0228 = 0.0456 < 0.05$.

Reject $H_0$. There is statistically significant evidence that the mean length differs from 10.0 mm.

### The t-Test

When $\sigma$ is unknown (the usual case), replace it with the sample standard deviation $S$:
$$T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1} \quad \text{under } H_0$$
where $t_{n-1}$ is the Student's t-distribution with $n-1$ degrees of freedom.

The t-distribution has heavier tails than the Normal, reflecting the additional uncertainty from estimating $\sigma$. As $n \to \infty$, $t_{n-1} \to \mathcal{N}(0, 1)$.

### Two-Sample t-Test

Compare means of two groups. If $X_1, \ldots, X_{n_1} \sim \mathcal{N}(\mu_1, \sigma^2)$ and $Y_1, \ldots, Y_{n_2} \sim \mathcal{N}(\mu_2, \sigma^2)$ (equal variance assumed):
$$T = \frac{\bar{X} - \bar{Y}}{S_p \sqrt{1/n_1 + 1/n_2}}, \quad S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1 + n_2 - 2}$$
where $S_p$ is the pooled standard deviation. Under $H_0: \mu_1 = \mu_2$, $T \sim t_{n_1 + n_2 - 2}$.

**Welch's t-test** handles unequal variances by adjusting the degrees of freedom — use this as the default in practice.

### Paired t-Test

When observations come in natural pairs (before/after, left/right), compute the differences $D_i = X_i - Y_i$ and perform a one-sample t-test on $D_i$:
$$T = \frac{\bar{D}}{S_D / \sqrt{n}} \sim t_{n-1}.$$
### Chi-Squared Test for Independence

For categorical data in a contingency table, test whether two variables are independent.

**Test statistic:**
$$\chi^2 = \sum_{\text{cells}} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$
where $O_{ij}$ is the observed count and $E_{ij} = \frac{(\text{row } i \text{ total})(\text{col } j \text{ total})}{n}$ is the expected count under independence.

Under $H_0$, $\chi^2 \sim \chi^2_{(r-1)(c-1)}$ where $r$ and $c$ are the number of rows and columns.

## Confidence Intervals

### Construction

A **confidence interval** (CI) at level $1 - \alpha$ is a random interval $[L, U]$ (functions of the data) such that
$$P(\theta \in [L, U]) = 1 - \alpha.$$
For a Normal mean with known $\sigma$:
$$\bar{X} \pm z_{\alpha/2} \frac{\sigma}{\sqrt{n}}$$
where $z_{\alpha/2}$ is the $1 - \alpha/2$ quantile of $\mathcal{N}(0, 1)$. For 95% CI: $z_{0.025} = 1.96$.

For unknown $\sigma$ (using the t-distribution):
$$\bar{X} \pm t_{\alpha/2, n-1} \frac{S}{\sqrt{n}}.$$
### Interpretation (Frequentist)

"If we repeated the experiment many times and computed a 95% CI each time, approximately 95% of those intervals would contain the true parameter $\theta$."

**Common misinterpretation:** "There is a 95% probability that $\theta$ is in this interval." This is wrong in the frequentist framework — $\theta$ is a fixed (unknown) constant, not a random variable. It either is or isn't in the interval; we just don't know which.

### Relationship Between CI and Hypothesis Test

There is an exact duality:
$$\text{Reject } H_0: \theta = \theta_0 \text{ at level } \alpha \iff \theta_0 \notin \text{CI at level } 1-\alpha.$$
A 95% confidence interval contains exactly those values of $\theta_0$ that would **not** be rejected at $\alpha = 0.05$.

*Proof.* The CI is $\{\theta_0 : |T(\theta_0)| \leq z_{\alpha/2}\}$ and the test rejects when $|T(\theta_0)| > z_{\alpha/2}$. These are complements. $\blacksquare$

## The Multiple Testing Problem

### The Problem

If you test 20 independent hypotheses at $\alpha = 0.05$, and all null hypotheses are true, the probability of at least one false positive is:
$$P(\text{at least one Type I error}) = 1 - (1 - 0.05)^{20} = 1 - 0.95^{20} \approx 0.64.$$
A 64% chance of a false discovery. With 100 tests: $1 - 0.95^{100} \approx 0.994$.

This is the **multiple testing problem** or **look-elsewhere effect**. It explains many of the "replication crises" across scientific fields.

### Bonferroni Correction

The simplest fix: use significance level $\alpha/m$ for each of $m$ tests.

**Theorem (Bonferroni).** If each of $m$ tests uses level $\alpha/m$, then the **family-wise error rate** (FWER) — the probability of any false rejection — is at most $\alpha$.

*Proof.* By the union bound:
$$P(\text{any false rejection}) = P\left(\bigcup_{i=1}^{m_0} \{p_i \leq \alpha/m\}\right) \leq \sum_{i=1}^{m_0} P(p_i \leq \alpha/m) = m_0 \cdot \frac{\alpha}{m} \leq \alpha$$
where $m_0 \leq m$ is the number of true nulls. $\blacksquare$

Bonferroni is conservative — it controls FWER but can have very low power when $m$ is large.

### False Discovery Rate (FDR): Benjamini-Hochberg

The **false discovery rate** is $\text{FDR} = E\left[\frac{\text{false positives}}{\text{total rejections}}\right]$.

**Benjamini-Hochberg procedure:**
1. Sort the $m$ p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$.
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} \alpha$.
3. Reject hypotheses corresponding to $p_{(1)}, \ldots, p_{(k)}$.

This controls $\text{FDR} \leq \alpha$ and is much more powerful than Bonferroni when many tests are conducted.

## A/B Testing

![A/B test design](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/07-ab-test.png)

### The Setup

Compare two versions (A = control, B = treatment) of a web page, ad, or product feature. Observe conversion rates $\hat{p}_A = k_A/n_A$ and $\hat{p}_B = k_B/n_B$.

### Test Statistic
$$Z = \frac{\hat{p}_B - \hat{p}_A}{\sqrt{\hat{p}(1-\hat{p})(1/n_A + 1/n_B)}}$$
where $\hat{p} = (k_A + k_B)/(n_A + n_B)$ is the pooled proportion.

### Sample Size Calculation

To detect a minimum detectable effect $\delta = p_B - p_A$ with power $1 - \beta$ at significance level $\alpha$ (two-sided):
$$n \geq \left(\frac{z_{\alpha/2} + z_\beta}{\delta}\right)^2 \cdot 2\bar{p}(1-\bar{p})$$
where $\bar{p} = (p_A + p_B)/2$ is the average proportion.

**Example.** Baseline conversion $p_A = 0.10$, minimum detectable effect $\delta = 0.02$ (i.e., detect a lift to $p_B = 0.12$), $\alpha = 0.05$, power $= 0.80$.
$$n \geq \left(\frac{1.96 + 0.84}{0.02}\right)^2 \cdot 2 \times 0.11 \times 0.89 \approx \frac{19600}{1} \times 0.1958 \approx 3838$$
per group. You need about 3,838 users per group — roughly 7,700 total.

### Practical vs Statistical Significance

A result can be statistically significant (small p-value) but practically meaningless. If an A/B test with 1 million users shows $p_B - p_A = 0.0001$ with $p < 0.001$, the effect is real but tiny — probably not worth the engineering effort to ship.

Always report **effect size and confidence interval**, not just p-values.

## Python: Hypothesis Tests in Practice

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

data = np.random.normal(loc=10.2, scale=0.5, size=25)
t_stat, p_value = stats.ttest_1samp(data, popmean=10.0)
print(f"One-sample t-test: t = {t_stat:.4f}, p = {p_value:.4f}")

group_a = np.random.normal(loc=5.0, scale=1.0, size=30)
group_b = np.random.normal(loc=5.5, scale=1.2, size=35)
t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
print(f"Two-sample t-test: t = {t_stat:.4f}, p = {p_value:.4f}")

observed = np.array([[30, 10], [15, 25]])
chi2, p_value, dof, expected = stats.chi2_contingency(observed)
print(f"Chi-squared test: chi2 = {chi2:.4f}, p = {p_value:.4f}, dof = {dof}")

n_A, n_B = 5000, 5000
conversions_A, conversions_B = 500, 560
p_A = conversions_A / n_A
p_B = conversions_B / n_B
p_pool = (conversions_A + conversions_B) / (n_A + n_B)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
z = (p_B - p_A) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))
print(f"A/B test: z = {z:.4f}, p = {p_value:.4f}")
print(f"  Lift: {p_B - p_A:.4f} ({(p_B - p_A)/p_A*100:.1f}%)")
print(f"  95% CI for lift: [{p_B-p_A-1.96*se:.4f}, {p_B-p_A+1.96*se:.4f}]")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
x = np.linspace(-4, 4, 300)
ax.plot(x, stats.norm.pdf(x), 'b-', linewidth=2)
t_obs = 2.0
ax.fill_between(x[x >= t_obs], stats.norm.pdf(x[x >= t_obs]),
                alpha=0.4, color='red', label=f'p-value/2 (right tail)')
ax.fill_between(x[x <= -t_obs], stats.norm.pdf(x[x <= -t_obs]),
                alpha=0.4, color='red', label=f'p-value/2 (left tail)')
ax.axvline(t_obs, color='red', linestyle='--', alpha=0.7)
ax.axvline(-t_obs, color='red', linestyle='--', alpha=0.7)
ax.set_title(f'Two-sided p-value (z = {t_obs})', fontsize=13)
ax.set_xlabel('Test statistic')
ax.set_ylabel('Density under $H_0$')
ax.legend(fontsize=10)

ax = axes[1]
x = np.linspace(-4, 6, 300)
ax.plot(x, stats.norm.pdf(x, 0, 1), 'b-', linewidth=2, label='$H_0$: $\\mu = 0$')
delta = 2.0
ax.plot(x, stats.norm.pdf(x, delta, 1), 'r-', linewidth=2,
        label=f'$H_1$: $\\mu = {delta}$')
z_crit = 1.645  # one-sided alpha = 0.05
ax.axvline(z_crit, color='gray', linestyle='--')
ax.fill_between(x[x >= z_crit], stats.norm.pdf(x[x >= z_crit], delta, 1),
                alpha=0.3, color='green', label=f'Power = {1-stats.norm.cdf(z_crit-delta):.3f}')
ax.fill_between(x[x >= z_crit], stats.norm.pdf(x[x >= z_crit], 0, 1),
                alpha=0.3, color='red', label=f'$\\alpha$ = {1-stats.norm.cdf(z_crit):.3f}')
ax.set_title('Power of a Test', fontsize=13)
ax.set_xlabel('Test statistic')
ax.legend(fontsize=9)

ax = axes[2]
m_tests = np.arange(1, 101)
fwer = 1 - (1 - 0.05)**m_tests
ax.plot(m_tests, fwer, 'r-', linewidth=2, label='FWER (no correction)')
ax.axhline(0.05, color='gray', linestyle=':', label='$\\alpha$ = 0.05')
ax.set_xlabel('Number of tests', fontsize=12)
ax.set_ylabel('P(at least one false positive)', fontsize=12)
ax.set_title('Multiple Testing Problem', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hypothesis_testing.png', dpi=150)
plt.show()
```

```python
from scipy.stats import false_discovery_control

np.random.seed(42)
n_tests = 100
n_true_effects = 10
n = 30  # samples per test

p_values = []
truth = []

for i in range(n_tests):
    if i < n_true_effects:
        data = np.random.normal(loc=2.0, scale=1, size=n)
        truth.append(True)
    else:
        data = np.random.normal(loc=0.0, scale=1, size=n)
        truth.append(False)
    _, pval = stats.ttest_1samp(data, 0)
    p_values.append(pval)

p_values = np.array(p_values)
truth = np.array(truth)

bonferroni_reject = p_values < 0.05 / n_tests

sorted_idx = np.argsort(p_values)
bh_threshold = np.arange(1, n_tests + 1) / n_tests * 0.05
bh_reject_sorted = p_values[sorted_idx] <= bh_threshold
if bh_reject_sorted.any():
    k = np.max(np.where(bh_reject_sorted)[0]) + 1
    bh_reject = np.zeros(n_tests, dtype=bool)
    bh_reject[sorted_idx[:k]] = True
else:
    bh_reject = np.zeros(n_tests, dtype=bool)

print(f"{'Method':<25} {'Rejections':>10} {'True Pos':>10} {'False Pos':>10}")
print("-" * 55)
print(f"{'No correction':<25} {(p_values < 0.05).sum():>10} "
      f"{((p_values < 0.05) & truth).sum():>10} "
      f"{((p_values < 0.05) & ~truth).sum():>10}")
print(f"{'Bonferroni':<25} {bonferroni_reject.sum():>10} "
      f"{(bonferroni_reject & truth).sum():>10} "
      f"{(bonferroni_reject & ~truth).sum():>10}")
print(f"{'Benjamini-Hochberg':<25} {bh_reject.sum():>10} "
      f"{(bh_reject & truth).sum():>10} "
      f"{(bh_reject & ~truth).sum():>10}")
```

The output typically shows: without correction, about 5-6 false positives sneak in among the 90 nulls. Bonferroni eliminates all false positives but may miss some true effects. Benjamini-Hochberg finds most true effects while keeping false discoveries under control. This illustrates the power advantage of FDR control over FWER control.

## Effect Size: What p-Values Don't Tell You

![Hypothesis testing courtroom trial null hypothesis on trial](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/covers/articles/probability-statistics/07-hypothesis-testing-courtroom-trial-null-hypothesis-on-trial.jpg)

A p-value tells you whether an effect is statistically distinguishable from zero. It does **not** tell you how large the effect is or whether it matters.

![Effect size visualization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/diagrams/probability-statistics/07-effect-size.png)

### Cohen's d

For comparing two group means:
$$d = \frac{\bar{X}_1 - \bar{X}_2}{S_p}$$
where $S_p$ is the pooled standard deviation. This is a **standardized effect size** — it measures the difference in units of standard deviations.

| $d$ | Interpretation |
|---|---|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

### Why Effect Size Matters

With $n = 1{,}000{,}000$ per group, you can detect $d = 0.01$ at $p < 0.001$. The effect is "highly significant" but completely meaningless in practice — a 0.01 standard deviation difference is invisible to any individual.

The antidote: **always report confidence intervals for effect sizes**, not just p-values. A confidence interval for the difference in means tells you both whether the effect is significant (does the CI exclude zero?) and how large it could plausibly be.

## The Neyman-Pearson Lemma

**Theorem (Neyman-Pearson).** For testing $H_0: \theta = \theta_0$ vs $H_1: \theta = \theta_1$ (simple vs simple), the most powerful test at level $\alpha$ rejects $H_0$ when:
$$\Lambda(\mathbf{x}) = \frac{L(\theta_1; \mathbf{x})}{L(\theta_0; \mathbf{x})} > c$$
where $c$ is chosen so that $P(\Lambda > c | H_0) = \alpha$.

The **likelihood ratio** $\Lambda$ is the optimal test statistic: no other test at the same significance level can have higher power. This theoretical result justifies the widespread use of likelihood ratio tests in parametric statistics.

### Generalized Likelihood Ratio Test

For composite hypotheses ($H_0: \theta \in \Theta_0$ vs $H_1: \theta \notin \Theta_0$):
$$\Lambda = \frac{\max_{\theta \in \Theta_0} L(\theta)}{\max_{\theta \in \Theta} L(\theta)} = \frac{L(\hat{\theta}_0)}{L(\hat{\theta}_{\text{MLE}})}.$$
Under $H_0$ and regularity conditions:
$$-2 \ln \Lambda \xrightarrow{d} \chi^2_r$$
where $r = \dim(\Theta) - \dim(\Theta_0)$ is the difference in the number of free parameters.

This is **Wilks' theorem**, and it's the foundation for comparing nested models — from testing whether a regression coefficient is zero to comparing hierarchical Bayesian models.

## Permutation Tests: Distribution-Free Inference

When distributional assumptions (normality, equal variances) are questionable, **permutation tests** offer an exact alternative.

**Idea:** Under $H_0$ (no difference between groups), the group labels are arbitrary. Permute the labels many times, recompute the test statistic each time, and see where the observed statistic falls in this permutation distribution.

```python
import numpy as np
from scipy import stats

np.random.seed(42)

group_a = np.array([5.1, 4.8, 5.3, 4.9, 5.0, 5.2, 5.4, 4.7])
group_b = np.array([5.5, 5.8, 5.3, 5.7, 5.9, 5.6, 5.4, 5.8])

observed_diff = group_b.mean() - group_a.mean()
combined = np.concatenate([group_a, group_b])
n_a = len(group_a)
n_perms = 100_000

perm_diffs = np.zeros(n_perms)
for i in range(n_perms):
    np.random.shuffle(combined)
    perm_diffs[i] = combined[n_a:].mean() - combined[:n_a].mean()

p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
print(f"Observed difference: {observed_diff:.4f}")
print(f"Permutation p-value: {p_value:.4f}")

t_stat, t_pval = stats.ttest_ind(group_a, group_b)
print(f"t-test p-value: {t_pval:.4f}")
```

Permutation tests are exact (not approximate) for any test statistic, require no distributional assumptions, and are easy to implement. Their main limitation: they only test exchangeability (are the groups interchangeable under $H_0$?), which may not capture all null hypotheses of interest.

## The Replication Crisis and What to Do About It

Starting around 2010, systematic replication efforts revealed that many published findings — perhaps 50% or more in psychology and biomedicine — fail to replicate. Common contributing factors:

1. **p-hacking:** Trying multiple analyses until $p < 0.05$, then reporting only the "significant" one.
2. **HARKing:** Hypothesizing After Results are Known — presenting post-hoc findings as if they were pre-specified.
3. **Publication bias:** Journals preferentially publish "significant" results, creating a literature biased toward false positives.
4. **Underpowered studies:** With low power, a significant result is more likely to overestimate the true effect.

**Recommended practices:**
- Pre-register hypotheses and analysis plans
- Report effect sizes and confidence intervals, not just p-values
- Use appropriate multiple testing corrections
- Consider Bayesian approaches for accumulating evidence
- Focus on replication rather than single studies

## Quick Reference

| Situation | Test | Statistic | Distribution under $H_0$ |
|---|---|---|---|
| One mean, $\sigma$ known | Z-test | $(\bar{X}-\mu_0)/(\sigma/\sqrt{n})$ | $\mathcal{N}(0,1)$ |
| One mean, $\sigma$ unknown | t-test | $(\bar{X}-\mu_0)/(S/\sqrt{n})$ | $t_{n-1}$ |
| Two means, independent | Two-sample t | $(\bar{X}-\bar{Y})/(S_p\sqrt{1/n_1+1/n_2})$ | $t_{n_1+n_2-2}$ |
| Two means, paired | Paired t | $\bar{D}/(S_D/\sqrt{n})$ | $t_{n-1}$ |
| Categorical independence | Chi-squared | $\sum (O-E)^2/E$ | $\chi^2_{(r-1)(c-1)}$ |
| Two proportions | Z-test | $(p_B-p_A)/\text{SE}$ | $\mathcal{N}(0,1)$ |
| Model comparison | Likelihood ratio | $-2\ln\Lambda$ | $\chi^2_r$ (Wilks) |
| Non-parametric | Permutation test | Any statistic | Permutation distribution |

## What's Next

Hypothesis testing answers "is the effect real?" by controlling false positive rates. But it treats the parameter as a fixed unknown and the data as random. Bayesian statistics flips this: the data are fixed (you observed them), and the parameter is random (described by a distribution). The next article develops Bayesian thinking — priors, posteriors, conjugate families, and the philosophical debate that has shaped statistics for a century.
