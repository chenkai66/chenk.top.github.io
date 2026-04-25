---
title: "Machine Learning Mathematical Derivations (9): Naive Bayes"
date: 2026-02-16 09:00:00
tags:
  - Machine Learning
  - Naive Bayes
  - Probabilistic Models
  - Bayes Theorem
  - Text Classification
  - Mathematical Derivation
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 9
  total: 20
lang: en
mathjax: true
description: "Rigorous derivation of Naive Bayes from Bayes theorem through conditional independence, parameter estimation, Laplace smoothing, three model variants, and why it works despite violated assumptions."
disableNunjucks: true
series_order: 9
---

> **Hook:** A spam filter that trains in milliseconds, scales to a million features, has *no hyperparameters worth tuning*, and still beats much fancier models on short-text problems. Naive Bayes pulls this off by making one outrageous assumption — every feature is independent given the class — and refusing to apologise for it. The assumption is wrong on essentially every real dataset, yet the classifier works. Understanding *why* is a tour through generative modelling, MAP estimation, Dirichlet priors, and the bias–variance tradeoff. This article walks the entire path.

## What You Will Learn

- Bayes' theorem as the foundation of probabilistic classification, and why the Bayes-optimal rule is the *target* every classifier secretly imitates.
- The conditional-independence assumption — what it costs us geometrically, and what it buys us statistically.
- Maximum-likelihood estimates for priors and likelihoods, and the MAP / Dirichlet view of Laplace smoothing.
- The three workhorse variants — Multinomial, Bernoulli, Gaussian — and a decision rule for picking between them.
- Why Naive Bayes survives violated assumptions: the Domingos–Pazzani result, error cancellation, and the bias–variance tradeoff.
- A clean, NumPy-only implementation of all three variants, validated against scikit-learn.

## Prerequisites

- Probability: Bayes' theorem, conditional probability, joint vs. marginal distributions, Gaussian density.
- Calculus: logarithms, partial derivatives.
- Familiarity with [Part 8: Support Vector Machines](/en/Machine-Learning-Mathematical-Derivations-8-Support-Vector-Machines/), especially the discriminative perspective on classification.

---

## 1. Bayesian Decision Theory

Before we add the "naive" piece, we need to be honest about what an *optimal* probabilistic classifier looks like. Naive Bayes is one specific way of approximating it.

### 1.1 Bayes' Theorem, Reread as a Belief Update

For a class label $c_k \in \{c_1, \dots, c_K\}$ and a feature vector $\mathbf{x} \in \mathbb{R}^d$:

$$
P(c_k \mid \mathbf{x}) \;=\; \frac{P(\mathbf{x} \mid c_k)\, P(c_k)}{P(\mathbf{x})}.
$$

Each term has a job:

| Term | Name | What it answers |
|---|---|---|
| $P(c_k \mid \mathbf{x})$ | **Posterior** | After seeing the data, how likely is class $c_k$? |
| $P(\mathbf{x} \mid c_k)$ | **Likelihood** | If the class were $c_k$, how plausible is this $\mathbf{x}$? |
| $P(c_k)$ | **Prior** | Before any features, how common is $c_k$? |
| $P(\mathbf{x})$ | **Evidence** | How likely is $\mathbf{x}$ overall? Pure normaliser. |

**Why the prior matters.** Suppose a test for a rare disease is 99% accurate. With prevalence $0.1\%$, a positive test moves the posterior from $0.001$ to roughly $0.09$ — still ten-to-one against the disease. Bayes' theorem makes the prior do its arithmetic out loud; ignoring priors is the canonical statistics mistake.

### 1.2 The Bayes-Optimal Classifier

Pick the class with the largest posterior:

$$
\hat{y}(\mathbf{x}) \;=\; \arg\max_{c_k} P(c_k \mid \mathbf{x})
\;=\; \arg\max_{c_k} P(\mathbf{x} \mid c_k)\, P(c_k),
$$

dropping $P(\mathbf{x})$ because it does not depend on $c_k$. This rule **minimises the 0-1 loss**: no classifier — not the deepest network, not the largest ensemble — can do better in expectation. The remaining error,

$$
R^\star(\mathbf{x}) \;=\; 1 - \max_k P(c_k \mid \mathbf{x}),
$$

is the **Bayes risk**, and the data-set-level $\mathbb{E}_{\mathbf{x}}[R^\star(\mathbf{x})]$ is the **Bayes error rate** — the irreducible noise floor of the problem itself.

Every probabilistic classifier we will study is, in one way or another, an *approximation* of this rule under finite data.

### 1.3 Generative vs. Discriminative

There are two ways to approach $P(c_k \mid \mathbf{x})$:

- **Discriminative** models (logistic regression, SVM, neural nets) learn $P(y \mid \mathbf{x})$ directly. They put all their statistical power into the decision surface and ignore how features are distributed.
- **Generative** models (Naive Bayes, LDA, HMMs) model $P(\mathbf{x}, y) = P(\mathbf{x} \mid y)P(y)$ and *derive* the posterior via Bayes' theorem.

| | Discriminative | Generative |
|---|---|---|
| Learns | $P(y \mid \mathbf{x})$ | $P(\mathbf{x}, y)$ |
| Asymptotic accuracy | Often higher | Often lower |
| Sample efficiency | Worse | Better when assumptions hold |
| Missing features | Hard | Natural — marginalise them out |
| Generates new $\mathbf{x}$ | No | Yes |
| Uses unlabeled data | No | Easy |

A classic result of Ng & Jordan (2001) shows the trade-off cleanly: with infinite data the discriminative model wins, but the generative model converges faster. Naive Bayes is the generative side at its most extreme.

---

## 2. The Naive Bayes Classifier

### 2.1 Why We Need a Drastic Simplification

Estimating $P(\mathbf{x} \mid c_k)$ for $\mathbf{x} \in \{0,1\}^d$ requires $2^d - 1$ free parameters per class. For a vocabulary of $d = 10{,}000$ words this is utterly hopeless. The joint distribution is the bottleneck.

### 2.2 The Conditional-Independence Assumption

Naive Bayes cuts the Gordian knot by assuming that, *conditional on the class*, all features are independent:

$$
P(\mathbf{x} \mid c_k) \;=\; \prod_{j=1}^{d} P\!\left(x^{(j)} \mid c_k\right).
$$

Two readings of this:

1. **Causal.** Treat the class as a hidden cause that produces each feature independently. Once you know the patient has the flu, fever and cough are no longer informative about each other.
2. **Statistical.** Each feature contributes its own one-dimensional conditional density. The joint is the product. Parameter count drops from exponential in $d$ to *linear* in $d$.

The geometric cost of this assumption is illustrated below: the true joint density (left) has correlated, tilted contours; the Naive Bayes approximation (right) has the *same marginals* but axis-aligned, factorised contours.

![Independence assumption: same marginals, different joint](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/09-Naive-Bayes/fig3_independence.png)

The two densities can disagree dramatically, yet — as we will see in §5 — the *classification* often does not.

### 2.3 The Classification Rule

Combining Bayes' theorem with conditional independence and dropping the constant $P(\mathbf{x})$:

$$
\hat{y}(\mathbf{x}) \;=\; \arg\max_{c_k}\; P(c_k) \prod_{j=1}^{d} P\!\left(x^{(j)} \mid c_k\right).
$$

Multiplying $d$ small probabilities will underflow to $0$ for any non-trivial $d$. Always work in log-space:

$$
\boxed{\;\hat{y}(\mathbf{x}) \;=\; \arg\max_{c_k} \left[\, \ln P(c_k) \;+\; \sum_{j=1}^{d} \ln P\!\left(x^{(j)} \mid c_k\right) \right].\;}
$$

This is the entire algorithm. Everything else — three "variants", smoothing, calibration — is just a discussion of how to estimate the per-feature conditionals $P(x^{(j)} \mid c_k)$.

For a 2D toy problem with two Gaussian classes, the picture is concrete:

![Class-conditional Gaussians and decision boundary](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/09-Naive-Bayes/fig1_class_conditional.png)

Each class has a diagonal-covariance Gaussian (axis-aligned ellipses, the Naive Bayes hypothesis class for continuous features). The decision boundary is the locus of points where the two posteriors agree — quadratic in general, linear when the classes share a covariance.

The same picture viewed *probabilistically* (rather than geometrically) is the posterior surface:

![Posterior heatmap and 1D slice](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/09-Naive-Bayes/fig2_posterior.png)

The right panel makes the soft transition explicit: along a slice through the data, the posterior $P(c_1 \mid \mathbf{x})$ rises smoothly from $0$ to $1$, crossing $0.5$ exactly where the two scaled likelihoods cross. That crossover is the decision boundary.

### 2.4 Parameter Estimation by Maximum Likelihood

Given training data $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$ with $N_k$ samples in class $c_k$:

**Class prior (MLE).** The fraction of training samples in each class:

$$
\hat{P}(c_k) \;=\; \frac{N_k}{N}.
$$

**Discrete features (MLE).** For a feature taking values in $\{v_1, \dots, v_{S_j}\}$:

$$
\hat{P}\!\left(x^{(j)} = v \mid c_k\right) \;=\;
\frac{\#\{i : x_i^{(j)} = v \text{ and } y_i = c_k\}}{N_k}.
$$

**Continuous features (Gaussian).** Fit one Gaussian per (feature, class):

$$
P\!\left(x^{(j)} \mid c_k\right) \;=\; \frac{1}{\sqrt{2\pi}\,\sigma_{jk}} \exp\!\left(-\frac{(x^{(j)} - \mu_{jk})^2}{2\sigma_{jk}^2}\right),
$$

with the standard estimators

$$
\hat{\mu}_{jk} \;=\; \frac{1}{N_k}\!\sum_{i:\, y_i=c_k}\! x_i^{(j)},
\qquad
\hat{\sigma}_{jk}^2 \;=\; \frac{1}{N_k}\!\sum_{i:\, y_i=c_k}\! \bigl(x_i^{(j)} - \hat{\mu}_{jk}\bigr)^2.
$$

That is *all* the training. There is no iterative optimiser, no learning rate, no convergence check.

### 2.5 Laplace Smoothing — and Its Bayesian Soul

**The catastrophe of zero counts.** Suppose the word *excellent* never appears in your spam training set. Then $\hat{P}(\text{excellent} \mid \text{spam}) = 0$, and a *single* occurrence of *excellent* in a future email forces $P(\text{spam} \mid \mathbf{x}) = 0$ regardless of how spammy the other 199 words are. One missing word kills everything.

**The fix.** Add a pseudocount $\alpha > 0$ to every cell:

$$
\hat{P}\!\left(x^{(j)} = v \mid c_k\right) \;=\; \frac{\#\{i : x_i^{(j)} = v,\, y_i=c_k\} + \alpha}{N_k + \alpha\, S_j},
\qquad
\hat{P}(c_k) \;=\; \frac{N_k + \alpha}{N + \alpha K}.
$$

With $\alpha = 1$ this is **Laplace (add-one) smoothing**; smaller $\alpha$ is sometimes called Lidstone smoothing.

**Why it works.** Place a symmetric Dirichlet prior $\mathrm{Dir}(\alpha, \dots, \alpha)$ on the multinomial $P(\cdot \mid c_k)$. The Dirichlet is conjugate to the multinomial, so the posterior is also Dirichlet, with parameters $\alpha + \text{counts}$. The MAP estimate of that posterior is exactly the smoothed formula above. Laplace smoothing is **MAP estimation under a uniform Dirichlet prior** — every category has been "seen" $\alpha$ times before any data arrives.

The visual story:

![Laplace smoothing: zero-probability fix and shrinkage](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/09-Naive-Bayes/fig6_laplace_smoothing.png)

- Left: as $\alpha$ grows, common words give up probability mass to rare and unseen words.
- Right: the estimate of every word is pulled toward the uniform $1/V$. Frequent words shrink down, unseen words rise from $0$. The pull strength is proportional to $\alpha / N_k$.

Choosing $\alpha$ is a bias–variance dial: small $\alpha$ trusts the data (low bias, high variance); large $\alpha$ trusts the uniform prior (high bias, low variance). Cross-validate.

---

## 3. Three Variants — Same Bayes Rule, Different Likelihood

The classification rule never changes. Only the model for $P(x^{(j)} \mid c_k)$ changes.

![Three NB variants compared](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/09-Naive-Bayes/fig5_three_variants.png)

### 3.1 Multinomial Naive Bayes — for word counts

Use when features are non-negative integer counts: term frequencies, $n$-grams, hashed buckets.

**Generative story.** Each document of class $c_k$ is a bag of words drawn i.i.d. from a vocabulary distribution $\boldsymbol{\theta}_k = (\theta_{1k}, \dots, \theta_{dk})$ with $\sum_j \theta_{jk} = 1$. The likelihood of word counts $\mathbf{x}$ is

$$
P(\mathbf{x} \mid c_k) \;\propto\; \prod_{j=1}^{d} \theta_{jk}^{x^{(j)}}.
$$

**Smoothed estimator.** Total occurrences of word $j$ in class $c_k$, smoothed by $\alpha$:

$$
\hat{\theta}_{jk} \;=\; \frac{\sum_{i:\, y_i=c_k} x_i^{(j)} + \alpha}{\sum_{j'=1}^{d}\sum_{i:\, y_i=c_k} x_i^{(j')} + \alpha d}.
$$

In words: the fraction of all word tokens in class-$c_k$ documents that happen to be word $j$.

### 3.2 Bernoulli Naive Bayes — for word presence

Use when features are binary: word *present* (1) or *absent* (0). Especially good for short texts.

**Generative story.** For each word in the vocabulary, flip a coin with bias $p_{jk} = P(x^{(j)} = 1 \mid c_k)$:

$$
P(\mathbf{x} \mid c_k) \;=\; \prod_{j=1}^{d} p_{jk}^{x^{(j)}} (1 - p_{jk})^{1 - x^{(j)}}.
$$

**The crucial difference from Multinomial.** Bernoulli explicitly multiplies in $(1 - p_{jk})$ for every absent word. Absence is *evidence*. If "free" is missing from an email and "free" is normally present in 85% of spam, that absence is strong evidence *against* spam. Multinomial NB simply ignores absent words.

**Rule of thumb.** Short, sparse documents (tweets, log lines, short queries): Bernoulli usually wins. Long documents (news, reviews): Multinomial is typically better because frequency carries real information.

### 3.3 Gaussian Naive Bayes — for continuous features

Use when features are real-valued and roughly unimodal per class: physical measurements, embeddings, sensor readings.

**Generative story.** Each feature within each class is Gaussian:

$$
P(x^{(j)} \mid c_k) \;=\; \mathcal{N}\!\left(x^{(j)};\, \mu_{jk},\, \sigma_{jk}^2\right).
$$

The total per-class density is the product of $d$ univariate Gaussians — i.e. a multivariate Gaussian with **diagonal covariance**. This is exactly the picture in Figure 1: axis-aligned ellipses.

A surprising fact: when the two classes have the same per-feature variances ($\sigma_{jk}^2 = \sigma_j^2$ for all $k$), the log-posterior ratio becomes linear in $\mathbf{x}$, and Gaussian NB reproduces the same boundary as **logistic regression** — but estimated generatively rather than discriminatively.

---

## 4. Worked Example: Spam Classification

Bayes' theorem, conditional independence, and Laplace smoothing all collide in the canonical Naive Bayes use case: spam filtering. The pipeline is:

![Bag-of-words to per-class word probabilities](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/09-Naive-Bayes/fig4_bag_of_words.png)

1. **Tokenise** documents into words.
2. **Build a term-frequency matrix** of shape $(N \times d)$ — rows are documents, columns are vocabulary entries.
3. **Estimate** $\hat{P}(c_k)$ from class frequencies and $\hat{\theta}_{jk}$ (Multinomial) or $\hat{p}_{jk}$ (Bernoulli) with Laplace smoothing.
4. **Predict** by computing the per-class log score and taking the argmax.

The right panel shows the estimated per-class word distributions: spam concentrates probability on "free", "money", "win"; ham on "meeting", "schedule", "project". A test email's classification is just an additive accumulation of per-feature evidence:

![Per-feature log-odds and waterfall to a final decision](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/09-Naive-Bayes/fig7_spam_decision.png)

Read the waterfall left to right: start from the log prior ratio $\ln \frac{P(\mathrm{spam})}{P(\mathrm{ham})}$, then add the contribution of each word (positive bars push toward spam, negative toward ham). The final height is the total log-odds. If it is positive, predict spam.

This is, in a strong sense, the *whole* algorithm: every Naive Bayes prediction is a sum of per-feature log-likelihood ratios.

---

## 5. Why Does It Work When the Assumption Is Wrong?

The independence assumption is essentially never true. Words co-occur, symptoms cluster, sensor readings correlate. So why does Naive Bayes routinely win on text classification benchmarks against models that *know* about correlations?

Four reinforcing reasons:

1. **Classification needs ranking, not calibration.** The argmax depends only on which posterior is largest, not on its absolute value. A model can be wildly miscalibrated yet rank classes correctly.
2. **Error cancellation in ratios.** When we compute the log-posterior ratio, repeated correlated features inflate *both* class scores by similar amounts. The bias largely cancels.
3. **Bias–variance tradeoff.** The independence assumption injects bias but slashes variance: instead of $\mathcal{O}(K \cdot d^2)$ correlation parameters, we estimate $\mathcal{O}(K \cdot d)$ marginals. With limited data, lower variance often wins outright.
4. **The Domingos–Pazzani theorem (1997).** They showed Naive Bayes is **0-1-loss optimal** whenever the class with the highest *true* posterior also has the highest *estimated* posterior — which is a much weaker condition than the independence assumption itself. Naive Bayes can be globally wrong about probabilities and still locally right about every prediction.

This is not a license to skip diagnostics. Naive Bayes is *not* well calibrated and its probability outputs should not be trusted at face value (see Q&A). But for ranking and argmax decisions, it punches well above its weight.

---

## 6. Reference Implementation

A clean NumPy implementation of all three variants. The Gaussian version is validated against scikit-learn at the end.

```python
import numpy as np
from collections import defaultdict


class GaussianNB:
    """Gaussian Naive Bayes for continuous features."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        # mu, var: shape (K, d)
        self.mu_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.var_ = np.array([X[y == c].var(axis=0) for c in self.classes_]) + 1e-9
        return self

    def _log_likelihood(self, X):
        # log N(x; mu, var) summed across features for each class -> (N, K)
        out = np.empty((X.shape[0], len(self.classes_)))
        for k in range(len(self.classes_)):
            diff2 = (X - self.mu_[k]) ** 2
            ll = -0.5 * (np.log(2 * np.pi * self.var_[k]) + diff2 / self.var_[k])
            out[:, k] = ll.sum(axis=1)
        return out

    def predict_log_proba(self, X):
        log_post = np.log(self.prior_) + self._log_likelihood(X)
        # Normalise via log-sum-exp for numerical stability
        log_post -= log_post.max(axis=1, keepdims=True)
        log_post -= np.log(np.exp(log_post).sum(axis=1, keepdims=True))
        return log_post

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]


class MultinomialNB:
    """Multinomial Naive Bayes with Laplace smoothing (word counts)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        K, d = len(self.classes_), X.shape[1]
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        self.log_theta_ = np.empty((K, d))
        for k, c in enumerate(self.classes_):
            counts = X[y == c].sum(axis=0)
            self.log_theta_[k] = np.log((counts + self.alpha) /
                                        (counts.sum() + self.alpha * d))
        return self

    def predict(self, X):
        log_post = np.log(self.prior_) + X @ self.log_theta_.T
        return self.classes_[np.argmax(log_post, axis=1)]


class BernoulliNB:
    """Bernoulli Naive Bayes (binary features)."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X = (X > 0).astype(float)
        self.classes_ = np.unique(y)
        K, d = len(self.classes_), X.shape[1]
        self.prior_ = np.array([(y == c).mean() for c in self.classes_])
        self.p_ = np.empty((K, d))
        for k, c in enumerate(self.classes_):
            Xc = X[y == c]
            self.p_[k] = (Xc.sum(axis=0) + self.alpha) / (len(Xc) + 2 * self.alpha)
        return self

    def predict(self, X):
        X = (X > 0).astype(float)
        # log P(x|c) = sum [ x log p + (1-x) log (1-p) ]
        log_p = np.log(self.p_)
        log_1mp = np.log(1 - self.p_)
        log_lik = X @ log_p.T + (1 - X) @ log_1mp.T
        log_post = np.log(self.prior_) + log_lik
        return self.classes_[np.argmax(log_post, axis=1)]


# --- Validation against scikit-learn ---
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB as SkGNB

    X, y = load_iris(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    ours = GaussianNB().fit(X_tr, y_tr)
    sk = SkGNB().fit(X_tr, y_tr)

    print(f"Ours:    {(ours.predict(X_te) == y_te).mean():.4f}")
    print(f"sklearn: {sk.score(X_te, y_te):.4f}")
    # Both print 0.9778 on this seed.
```

---

## 7. Exercises

### Exercise 1 — Conditional independence, by definition

**Problem.** Given $P(c_1) = 0.6$, $P(x_1=1 \mid c_1) = 0.8$, $P(x_2=1 \mid c_1) = 0.5$, compute $P(x_1=1, x_2=1 \mid c_1)$ under the Naive Bayes assumption.

**Solution.** By conditional independence given $c_1$, joints factorise:

$$
P(x_1=1, x_2=1 \mid c_1) = P(x_1=1 \mid c_1)\, P(x_2=1 \mid c_1) = 0.8 \times 0.5 = 0.4.
$$

The prior $P(c_1)$ is irrelevant *inside the conditional* — it would only enter if we also wanted $P(x_1, x_2)$ unconditionally.

### Exercise 2 — Laplace smoothing in numbers

**Problem.** Vocabulary size $V = 1000$. In the positive class, the word *excellent* appears 5 times in a corpus of 200 word tokens. Compute (a) the MLE estimate, (b) the Laplace-smoothed estimate ($\alpha = 1$), (c) the smoothed estimate for an unseen word.

**Solution.**

(a) MLE: $\hat{P}(\text{excellent} \mid +) = 5 / 200 = 0.025$.

(b) Smoothed: $\hat{P}(\text{excellent} \mid +) = (5+1) / (200 + 1000) = 6 / 1200 = 0.005$.

(c) Unseen word: $\hat{P}(w \mid +) = (0+1) / (200 + 1000) = 1/1200 \approx 8.3 \times 10^{-4}$.

Notice how aggressively smoothing flattens the distribution when $\alpha V \gg N_k$ (here $1000 \gg 200$). This is the dial: with little class-conditional data, the prior dominates.

### Exercise 3 — Multinomial vs. Bernoulli on the same email

**Problem.** A 100-word email contains *win* 5 times and *free* 3 times. Under spam: $P(\text{win} \mid \text{spam}) = 0.05$, $P(\text{free} \mid \text{spam}) = 0.03$ (Multinomial); $P(\text{win present} \mid \text{spam}) = 0.7$, $P(\text{free present} \mid \text{spam}) = 0.6$ (Bernoulli). Compare log-likelihood contributions.

**Solution.**

Multinomial uses counts:

$$
\ln L_{\mathrm{mult}} = 5\ln(0.05) + 3\ln(0.03) = 5(-2.996) + 3(-3.507) = -25.5.
$$

Bernoulli uses presence only:

$$
\ln L_{\mathrm{bern}} = \ln(0.7) + \ln(0.6) = -0.357 + (-0.511) = -0.87.
$$

These numbers are not directly comparable (different probabilistic objects), but the *shape* tells the story: Multinomial amplifies repeated occurrences linearly with count, while Bernoulli flattens 5 occurrences and 1 occurrence into the same evidence. For long documents that dynamic range matters; for tweets it usually does not.

### Exercise 4 — Gaussian NB parameters by hand

**Problem.** Class $+1$ contains three 2D points: $(1, 2), (2, 3), (3, 1)$. Compute the Gaussian NB parameters.

**Solution.** Per-feature MLE means and variances:

$$
\hat\mu_1 = \tfrac{1+2+3}{3} = 2, \quad
\hat\sigma_1^2 = \tfrac{(1{-}2)^2 + (2{-}2)^2 + (3{-}2)^2}{3} = \tfrac{2}{3},
$$

$$
\hat\mu_2 = \tfrac{2+3+1}{3} = 2, \quad
\hat\sigma_2^2 = \tfrac{(2{-}2)^2 + (3{-}2)^2 + (1{-}2)^2}{3} = \tfrac{2}{3}.
$$

So $\hat\sigma_1 = \hat\sigma_2 = \sqrt{2/3} \approx 0.816$. Gaussian NB has *no* off-diagonal covariance term; the (1,3) sample being on the "wrong" diagonal does not enter the model.

### Exercise 5 — Why does Naive Bayes work?

**Problem.** Give four independent reasons Naive Bayes performs well despite violated independence.

**Solution.** The reasons developed in §5:

1. **Argmax-only decisions.** Classification needs the right *ranking*, not the right probability magnitudes.
2. **Bias cancellation in log-ratios.** Correlated features inflate the log-likelihoods of both classes by similar amounts, and these cancel in the posterior ratio.
3. **Bias–variance favourability.** Trading a small amount of bias for a large reduction in variance is often a net win, especially when $N$ is small relative to $d$.
4. **Domingos–Pazzani (1997).** A formal result: Naive Bayes is 0-1-loss optimal whenever its argmax matches the true posterior's argmax — a much weaker condition than the joint distribution actually factorising.

---

## Q&A

### Why "naive"?

The conditional-independence assumption is rarely literally true — words co-occur, symptoms cluster — yet the model is built as if they did. The label refers to that strong, almost ostentatiously simple, assumption.

### How is Naive Bayes related to logistic regression?

Under Gaussian features with shared per-feature variance, Gaussian NB produces a linear decision boundary $\mathbf{w}^\top\mathbf{x} + b = 0$ — the same form as logistic regression. The coefficients differ because NB estimates $\mathbf{w}$ by *generative* MLE on $P(\mathbf{x}, y)$, while LR estimates it *discriminatively* on $P(y \mid \mathbf{x})$. Asymptotically LR is at least as good; NB converges with fewer samples.

### Are Naive Bayes probability outputs trustworthy?

No. Because correlated features get their log-likelihoods double-counted, the posteriors are typically too extreme (close to 0 or 1). For ranking tasks (argmax, top-$K$, ROC) this is fine. For decisions that depend on the magnitude (cost-sensitive thresholds, risk computations) calibrate with **Platt scaling** or **isotonic regression** on a held-out set.

### What is the time complexity?

Training: $\mathcal{O}(Nd)$ — one pass through the training data. Prediction: $\mathcal{O}(Kd)$ per sample. There are no iterations and no hyperparameters whose tuning costs more than a CV sweep over $\alpha$.

### How does Naive Bayes handle missing features?

Naturally: when feature $j$ is missing for sample $i$, omit the term $\ln P(x_i^{(j)} \mid c_k)$ from the sum. This is mathematically equivalent to marginalising over the missing feature and is a real advantage of generative modelling.

### When should I *not* use Naive Bayes?

When (i) features are heavily correlated *and* the correlation differs across classes (so error cancellation fails), (ii) the application needs calibrated probabilities, or (iii) you have plenty of data and a discriminative model is feasible — logistic regression, gradient boosting, or transformers will usually pull ahead.

---

## References

- Domingos, P., & Pazzani, M. (1997). On the optimality of the simple Bayesian classifier under zero-one loss. *Machine Learning*, 29(2-3), 103-130.
- Ng, A. Y., & Jordan, M. I. (2001). On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes. *NeurIPS*.
- McCallum, A., & Nigam, K. (1998). A comparison of event models for naive Bayes text classification. *AAAI Workshop on Learning for Text Categorization*.
- Manning, C. D., Raghavan, P., & Schutze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press. Chapter 13.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Section 8.2.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 3.

---

<div class="series-nav">

**ML Mathematical Derivations Series**

[< Part 8: Support Vector Machines](/en/Machine-Learning-Mathematical-Derivations-8-Support-Vector-Machines/) | **Part 9: Naive Bayes** | [Part 10: Semi-Naive Bayes >](/en/Machine-Learning-Mathematical-Derivations-10-Semi-Naive-Bayes-and-Bayesian-Networks/)

</div>
