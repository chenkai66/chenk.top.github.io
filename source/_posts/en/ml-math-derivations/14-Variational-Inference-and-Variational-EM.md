---
title: "Machine Learning Mathematical Derivations (14): Variational Inference and Variational EM"
date: 2026-03-08 09:00:00
categories:
  - Machine Learning
tags:
  - Variational Inference
  - ELBO
  - KL Divergence
  - Mean Field
  - Variational Bayes
  - Mathematical Derivation
  - Machine Learning
series:
  name: "ML Mathematical Derivations"
  order: 14
  total: 20
lang: en
mathjax: true
description: "A first-principles derivation of variational inference. From the ELBO identity and the mean-field assumption to the CAVI updates, variational EM, and the reparameterization trick that powers VAEs."
disableNunjucks: true
series_order: 14
---

When the posterior $p(\mathbf{z}\mid\mathbf{x})$ is intractable, you have two roads. **Sampling** (MCMC) walks a Markov chain whose stationary distribution is the posterior — eventually exact, but slow and hard to diagnose. **Variational inference** (VI) instead picks a simple family $\mathcal{Q}$ of distributions and finds the member $q^\star\in\mathcal{Q}$ that lies closest to the true posterior. Inference becomes optimization, and the same machinery that fits a neural network now fits a Bayesian model.

This post derives VI from a single identity, builds the mean-field algorithm and CAVI from that identity, connects EM and variational EM as special cases, and ends with the reparameterization trick that turns the ELBO into a stochastic objective compatible with autodiff — the engine inside every VAE.

## What You Will Learn

- Why VI turns inference into optimization through the ELBO identity
- The mean-field assumption and the closed-form coordinate-ascent updates it produces
- How variational EM extends classical EM by replacing the exact E-step with optimization over $q$
- The reparameterization trick: a low-variance gradient estimator that backpropagates through samples
- When VI is the right tool — and when it silently lies to you (mode-seeking, under-dispersion)

## Prerequisites

- The EM algorithm and the ELBO from [Part 13](/en/Machine-Learning-Mathematical-Derivations-13-EM-Algorithm-and-GMM/)
- KL divergence and Jensen's inequality
- Multivariate calculus and the exponential family
- Comfort with stochastic gradient estimation

---

## 1. The Posterior Bottleneck

Bayesian inference with observations $\mathbf{x}$, latent variables $\mathbf{z}$ and parameters $\boldsymbol{\theta}$ produces the posterior

$$p(\mathbf{z}\mid\mathbf{x}) \;=\; \frac{p(\mathbf{x},\mathbf{z})}{p(\mathbf{x})},\qquad p(\mathbf{x}) \;=\; \int p(\mathbf{x},\mathbf{z})\,d\mathbf{z}.$$

The numerator is cheap — it is just the model's joint density. The denominator, the **evidence** $p(\mathbf{x})$, is the integral of the joint over the entire latent space. For anything but conjugate exponential-family pairs, that integral is intractable.

Two strategies dominate the literature:

| | MCMC | Variational Inference |
|---|------|-----------------------|
| Approach | Stochastic sampling | Deterministic optimization |
| Asymptotic behavior | Exact (in the limit) | Biased — bounded by $\mathcal{Q}$ |
| Computational cost | High; chain mixing dominates | Low; gradient steps |
| Diagnostics | Hard (convergence, autocorrelation) | Easy (ELBO is monotone) |
| Scales to large data | Poorly without subsampling | Naturally with mini-batches |

VI's bias is the price you pay for its speed. The rest of this post quantifies that trade-off and shows how to manage it.

---

## 2. The ELBO Identity

Pick **any** distribution $q(\mathbf{z})$ over the latent variables. Then

$$
\log p(\mathbf{x})
\;=\; \underbrace{\mathbb{E}_q\!\left[\log\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\right]}_{\displaystyle \mathcal{L}(q)\;\text{(ELBO)}}
\;+\; \underbrace{\mathrm{KL}\!\big(q(\mathbf{z})\,\big\|\,p(\mathbf{z}\mid\mathbf{x})\big)}_{\displaystyle \geq 0}.
$$

The derivation takes one line. Multiply and divide the joint by $q(\mathbf{z})$, take logs, and split:

$$\log p(\mathbf{x}) = \log\!\int q(\mathbf{z})\,\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\,d\mathbf{z}
= \mathbb{E}_q\!\left[\log\frac{p(\mathbf{x},\mathbf{z})}{q(\mathbf{z})}\right] + \mathrm{KL}(q\,\|\,p(\cdot\mid\mathbf{x})).$$

Because $\log p(\mathbf{x})$ does not depend on $q$, **maximizing the ELBO with respect to $q$ is exactly the same as minimizing the KL divergence to the true posterior**.

![ELBO decomposition: log-evidence splits into ELBO and KL gap](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/14-Variational-Inference-and-Variational-EM/fig1_elbo_decomposition.png)

*Figure 1.* The log-evidence is fixed once the data and model are chosen. Every increase in the ELBO is matched by an equal decrease in the KL gap; perfect inference would close the gap entirely. Because KL is non-negative, the ELBO is also a certified **lower bound** on the marginal likelihood — useful for model comparison.

It is worth pausing on what we have done. We replaced an intractable integral (the evidence) with an optimization problem over a function space (find the best $q$). All of variational inference is the engineering that follows from this swap.

---

## 3. Mean-Field Approximation

The optimization is still infinite-dimensional. The simplest restriction — and the one Jordan, Ghahramani, Jaakkola and Saul popularized in the 90s — is the **mean-field** assumption: $q$ factorizes across coordinates,

$$q(\mathbf{z}) \;=\; \prod_{j=1}^{M} q_j(z_j).$$

Each factor lives in its own family. We make no further parametric commitment.

![Mean-field collapses the joint into independent marginals](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/14-Variational-Inference-and-Variational-EM/fig2_mean_field.png)

*Figure 2.* The middle panel matches the marginals of the true posterior (left) but loses every off-diagonal entry. The right panel shows the global optimum of $\mathrm{KL}(q\|p)$: variance shrinks by a factor of $1-\rho^2$, so the surrogate is **under-dispersed** along both axes. This systematic under-estimation of uncertainty is the most common failure mode of mean-field VI.

### 3.1 The optimal factor

Plug the factorization into the ELBO and isolate one factor $q_j$. Treating the others as fixed,

$$
\mathcal{L}(q) = \int q_j(z_j) \,\mathbb{E}_{q_{-j}}\!\big[\log p(\mathbf{x},\mathbf{z})\big]\,dz_j
\;-\; \int q_j(z_j)\log q_j(z_j)\,dz_j \;+\; \text{const}.
$$

The bracketed expectation is a function of $z_j$ alone — call it $\log\tilde{p}(z_j)$. Maximizing over $q_j$ subject to $\int q_j = 1$ is a textbook variational calculus problem; the optimum is

$$\boxed{\;\log q_j^\star(z_j) \;=\; \mathbb{E}_{q_{-j}}\!\big[\log p(\mathbf{x},\mathbf{z})\big] \;+\; \text{const}.\;}$$

This is the central formula of mean-field VI. The optimal factor for coordinate $j$ is the geometric average of the joint with respect to all other factors, normalized.

### 3.2 Coordinate-Ascent Variational Inference (CAVI)

Because each factor's optimum depends on the others, we solve cyclically:

```
initialize q_1, ..., q_M
repeat
    for j = 1, ..., M:
        log q_j(z_j) <- E_{q_{-j}}[log p(x, z)] + const
        normalize q_j
until ELBO change < tolerance
```

Every update is a coordinate ascent step on a concave-in-each-coordinate objective, so the ELBO never decreases. Convergence to a local optimum is guaranteed; reaching the global optimum is not.

![CAVI ellipses tightening onto the target while ELBO climbs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/14-Variational-Inference-and-Variational-EM/fig5_cavi_iterations.png)

*Figure 5.* Eight CAVI sweeps on a correlated bivariate Gaussian. The diagonal $q$ both relocates and shrinks: its mean is dragged to the origin in two steps, and its variance collapses to $1/\text{precision}_{ii}$. The right panel shows the monotone ELBO trajectory — a useful convergence diagnostic in any VI implementation.

### 3.3 Conjugate exponential families

When the model is a **conjugate exponential family** — every conditional $p(z_j\mid \mathbf{z}_{-j},\mathbf{x})$ lies in an exponential family — the CAVI update has a closed form. The optimal $q_j$ is in the same exponential family as the conditional, and updating it amounts to averaging natural parameters under $q_{-j}$. This covers Bayesian Gaussian mixtures, LDA, Bayesian linear regression, hidden Markov models with Dirichlet priors, and many others. For non-conjugate models, we need the black-box approach of Section 6.

---

## 4. The Variational Family as Approximator

It pays to look at VI without the mean-field crutch. Pick **any** parametric family $q_\phi(\mathbf{z})$ — a Gaussian with learnable mean and covariance, a normalizing flow, an amortized inference network — and minimize $\mathrm{KL}(q_\phi\,\|\,p(\cdot\mid\mathbf{x}))$ over $\phi$.

![Reverse vs forward KL on a bimodal target](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/14-Variational-Inference-and-Variational-EM/fig3_q_approximates_post.png)

*Figure 3.* On a bimodal target, the **reverse-KL** optimum (purple) locks onto a single mode — it is **mode-seeking** because $\mathrm{KL}(q\|p) = \int q\log(q/p)$ pays an infinite price wherever $q>0$ but $p\approx 0$. The **forward-KL** optimum (amber, used by expectation propagation) instead matches moments and **covers** both modes, even at the cost of placing density in low-probability valleys. VI uses reverse KL because we can compute it from samples of $q$ alone — but the asymmetry has real consequences for downstream uncertainty estimates.

![KL asymmetry: zero-forcing vs mass-covering](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/14-Variational-Inference-and-Variational-EM/fig6_kl_asymmetry.png)

*Figure 6.* The same story on a symmetric mixture. Reverse KL has multiple local optima, each centered on one mode. Forward KL produces a single, broad solution centered between the modes. Knowing which behavior your application can tolerate is half the battle when designing a VI system.

### When mean-field bites

The under-dispersion in Figure 2 is not a bug; it follows from the geometry of reverse KL applied to a factorized family. Whenever your downstream task **uses the posterior variance** — Bayesian model averaging, calibrated prediction, decision theory — mean-field VI will under-state uncertainty. Three escape hatches:

1. **Structured VI**: keep some dependencies (e.g., a tree-structured $q$).
2. **Normalizing flows**: a sequence of invertible transformations gives $q$ enough flexibility to model correlations.
3. **Amortized inference** with a flexible neural network (the encoder in a VAE).

---

## 5. Variational EM

Section 2 of [Part 13](/en/Machine-Learning-Mathematical-Derivations-13-EM-Algorithm-and-GMM/) showed that the EM algorithm itself rests on the ELBO identity. EM alternates:

- **E-step**: choose $q(\mathbf{z}) = p(\mathbf{z}\mid\mathbf{x};\boldsymbol{\theta}^{(t)})$, the exact posterior — the KL gap goes to zero, so the ELBO touches $\log p(\mathbf{x};\boldsymbol{\theta}^{(t)})$.
- **M-step**: hold $q$ fixed, maximize the ELBO with respect to $\boldsymbol{\theta}$, which reduces to maximizing $\mathbb{E}_q[\log p(\mathbf{x},\mathbf{z};\boldsymbol{\theta})]$.

When the exact posterior is intractable, the E-step breaks. **Variational EM** simply replaces it with a VI sub-routine:

| Step | Standard EM | Variational EM | Variational Bayes EM |
|------|-------------|----------------|----------------------|
| E-step | Exact $p(\mathbf{z}\mid\mathbf{x};\boldsymbol{\theta})$ | Best $q(\mathbf{z})\in\mathcal{Q}$ | Best $q(\mathbf{z},\boldsymbol{\theta})\in\mathcal{Q}$ |
| M-step | Maximize $Q(\boldsymbol{\theta})$ | Maximize $Q(\boldsymbol{\theta})$ | Folded into VI |
| Treats $\boldsymbol{\theta}$ as | Point estimate | Point estimate | Random variable |

In Variational EM the ELBO is no longer tight after the E-step (the KL gap is non-zero), so the algorithm maximizes a lower bound on $\log p(\mathbf{x};\boldsymbol{\theta})$ rather than the likelihood itself. It still **monotonically increases** that lower bound — a useful guarantee in practice.

**Variational Bayes EM** (VBEM) goes further and treats $\boldsymbol{\theta}$ as a random variable with its own variational factor, producing a full posterior over both $\mathbf{z}$ and $\boldsymbol{\theta}$. This is what the variational Bayesian Gaussian mixture and variational LDA solve.

---

## 6. Black-Box VI and the Reparameterization Trick

Outside conjugate models, neither the closed-form CAVI updates nor the variational E-step are available. **Black-box VI (BBVI)** parameterizes $q_\phi$ with a neural network and optimizes the ELBO with stochastic gradients:

$$\nabla_\phi\,\mathcal{L}(\phi) \;=\; \nabla_\phi\,\mathbb{E}_{q_\phi(\mathbf{z})}\!\left[\log p(\mathbf{x},\mathbf{z}) - \log q_\phi(\mathbf{z})\right].$$

The expectation is over $q_\phi$, whose distribution depends on $\phi$ — the gradient does not move inside the expectation for free.

**Score-function (REINFORCE) estimator.** Differentiate the density:

$$\nabla_\phi \mathcal{L} \;=\; \mathbb{E}_{q_\phi}\!\left[\big(\log p(\mathbf{x},\mathbf{z}) - \log q_\phi(\mathbf{z})\big)\nabla_\phi \log q_\phi(\mathbf{z})\right].$$

Unbiased, but high variance — needs control variates and large sample sizes to be practical.

**Reparameterization trick.** Whenever we can write $\mathbf{z}$ as a deterministic transform of a parameter-free noise variable,

$$\mathbf{z} \;=\; g_\phi(\boldsymbol{\epsilon}),\qquad \boldsymbol{\epsilon}\sim p(\boldsymbol{\epsilon}),$$

the expectation moves to a fixed measure and the gradient slides inside:

$$\nabla_\phi \mathcal{L} \;=\; \mathbb{E}_{p(\boldsymbol{\epsilon})}\!\left[\nabla_\phi\big(\log p(\mathbf{x},g_\phi(\boldsymbol{\epsilon})) - \log q_\phi(g_\phi(\boldsymbol{\epsilon}))\big)\right].$$

For a diagonal Gaussian $q_\phi(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_\phi,\,\mathrm{diag}(\boldsymbol{\sigma}_\phi^2))$, the transform is $g_\phi(\boldsymbol{\epsilon}) = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi\odot\boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$. The whole computation graph is differentiable, autodiff handles the rest, and the resulting gradient has dramatically lower variance than REINFORCE. This is the engine inside the **variational autoencoder** and almost every modern continuous-latent variational model.

For discrete latent variables the trick fails (you cannot write $z\in\{0,1\}$ as a smooth function of $\epsilon$). The standard remedies are REINFORCE with control variates, the Gumbel-Softmax / Concrete relaxation, or a continuous relaxation followed by straight-through estimation.

### When to choose what

![VI vs MCMC speed-accuracy trade-off](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/14-Variational-Inference-and-Variational-EM/fig4_vi_vs_mcmc.png)

*Figure 4.* Wall-clock vs error on a representative benchmark. VI hits a small error fast and then plateaus at its bias floor (the gap between the true posterior and the best member of $\mathcal{Q}$). MCMC starts after a burn-in and grinds down toward zero error at the classic $1/\sqrt{T}$ rate. The decision matrix is short:

- **Use VI** for large datasets, online learning, exploratory modeling, and inside neural networks.
- **Use MCMC** when the posterior really matters — drug-dose response, scientific inference, anything published.
- **Use both**: warm-start an HMC chain from a VI mean. You get fast initialization plus eventual unbiasedness.

---

## 7. Application: VI for LDA

Latent Dirichlet Allocation (Blei, Ng, Jordan 2003) is the canonical large-scale VI success story. The model has per-document topic proportions $\theta_d$ and per-topic word distributions $\beta_k$, both Dirichlet-distributed. The posterior is intractable, but the model is conjugate-exponential, so mean-field CAVI gives closed-form updates:

$$q(\theta,\beta,z) \;=\; \prod_d q(\theta_d\mid\gamma_d) \prod_k q(\beta_k\mid\lambda_k) \prod_{d,n} q(z_{d,n}\mid\phi_{d,n}),$$

with Dirichlet variational factors for $\theta_d$ and $\beta_k$, and categorical factors for the per-word topic assignments.

![Variational LDA: per-document topics and per-topic words](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/14-Variational-Inference-and-Variational-EM/fig7_lda_topics.png)

*Figure 7.* The two outputs of variational LDA. **Left**: the posterior mean topic proportions $\mathbb{E}_q[\theta_d]$ for eight documents. Each document is a soft mixture of four topics; documents 1 and 5 are dominantly "ML", document 3 is dominantly "Finance", and so on. **Right**: the posterior mean word distributions $\mathbb{E}_q[\beta_k]$. Each topic concentrates mass on a few characteristic vocabulary terms — exactly the interpretable structure that made LDA a workhorse for the 2000s and 2010s.

Stochastic VI (Hoffman et al. 2013) scales the same updates to billions of documents by sampling mini-batches and applying natural-gradient steps to $\lambda_k$ — a clean example of the speed advantage in Figure 4.

---

## 8. Implementation: Variational Bayesian GMM

A compact CAVI implementation of the conjugate Bayesian GMM. The math behind the updates is in Bishop PRML §10.2; here we focus on a clean reading of the loop.

```python
import numpy as np
from scipy.special import digamma


class VariationalGMM:
    """Mean-field variational Bayes Gaussian mixture (CAVI)."""

    def __init__(self, K=3, max_iter=100, tol=1e-3):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        N, d = X.shape
        # Vague conjugate priors
        alpha0, beta0, nu0 = 1.0, 1.0, float(d)
        m0, W0 = X.mean(0), np.eye(d)

        # Variational parameters initialized to the prior + uniform mass
        self.alpha = np.full(self.K, alpha0 + N / self.K)
        self.beta = np.full(self.K, beta0 + N / self.K)
        self.nu = np.full(self.K, nu0 + N / self.K)
        self.m = np.array([m0 + 0.1 * np.random.randn(d) for _ in range(self.K)])
        self.W = np.array([W0.copy() for _ in range(self.K)])

        r = np.random.dirichlet([1] * self.K, N)
        for _ in range(self.max_iter):
            r_old = r.copy()

            # ---- E-step: variational responsibilities  q(z_n) ----
            r = self._update_r(X, N, d)

            # ---- M-step: closed-form updates of the Dirichlet & NW factors ----
            N_k = r.sum(0)
            x_bar = (r.T @ X) / N_k[:, None]
            self.alpha = alpha0 + N_k
            self.beta = beta0 + N_k
            self.nu = nu0 + N_k
            self.m = (beta0 * m0 + N_k[:, None] * x_bar) / self.beta[:, None]
            for k in range(self.K):
                diff = X - x_bar[k]
                S = (r[:, k, None] * diff).T @ diff / N_k[k]
                dm = x_bar[k] - m0
                self.W[k] = np.linalg.inv(
                    np.linalg.inv(W0)
                    + N_k[k] * S
                    + beta0 * N_k[k] / (beta0 + N_k[k]) * np.outer(dm, dm)
                )

            if np.max(np.abs(r - r_old)) < self.tol:
                break
        return self

    def _update_r(self, X, N, d):
        """Compute  rho_{nk} = exp( E_q[log pi_k] + 0.5 E_q[log|Lambda_k|]
                                    - 0.5 E_q[(x-mu_k)^T Lambda_k (x-mu_k)] )"""
        r = np.zeros((N, self.K))
        for k in range(self.K):
            E_lp = digamma(self.alpha[k]) - digamma(self.alpha.sum())
            E_ld = sum(digamma((self.nu[k] + 1 - i) / 2) for i in range(1, d + 1))
            E_ld += d * np.log(2) + np.log(max(np.linalg.det(self.W[k]), 1e-10))
            for n in range(N):
                diff = X[n] - self.m[k]
                r[n, k] = (E_lp
                           + 0.5 * E_ld
                           - 0.5 * (self.nu[k] * diff @ self.W[k] @ diff
                                    + d / self.beta[k]))
        r = np.exp(r - r.max(1, keepdims=True))
        return r / r.sum(1, keepdims=True)

    def predict(self, X):
        return np.argmax(self._update_r(X, len(X), X.shape[1]), axis=1)
```

The algorithmic shape mirrors the EM-GMM from Part 13 — alternate responsibilities and component statistics — but every quantity is a posterior moment under $q$, not a point estimate. A side benefit is **automatic component pruning**: $\alpha_k$ shrinks toward $\alpha_0$ for empty components, so unused topics quietly fade out instead of fitting noise.

---

## 9. Q&A

**Q1: Why reverse KL and not forward KL?**
Reverse $\mathrm{KL}(q\|p)$ only requires expectations under $q$, which we control. Forward $\mathrm{KL}(p\|q)$ requires expectations under the intractable $p$. The price is mode-seeking behavior — see Figure 6.

**Q2: My ELBO is going down — is something broken?**
Yes. CAVI's ELBO is monotone non-decreasing by construction. A drop means a bug, usually in (i) the responsibility normalization, (ii) a sign in the entropy term, or (iii) a stale variational parameter being read before it is updated.

**Q3: How tight is the ELBO bound?**
$\log p(\mathbf{x}) - \mathcal{L}(q^\star) = \mathrm{KL}(q^\star\|p(\cdot\mid\mathbf{x})) \geq 0$. The gap is exactly the modeling error of the variational family. Bayesian model comparison via ELBO is fair only when the gaps across models are comparable.

**Q4: When does mean-field fail catastrophically?**
Strong posterior correlations (Figure 2) and multimodality (Figure 6). Symptoms: drastically under-estimated posterior variance, overconfident predictions, sensitivity to initialization. Fix with structured VI, normalizing flows, or full-covariance amortized inference.

**Q5: Reparameterization vs REINFORCE — which one?**
Reparameterization for continuous, differentiable $q$ (Gaussians, Logistic, mixtures via Gumbel-Softmax). REINFORCE for discrete latent variables. Reparameterization typically has 100–1000× lower gradient variance when both apply.

**Q6: Variational EM vs full Variational Bayes — when do I need VBEM?**
If you need uncertainty in $\boldsymbol{\theta}$ (small data, model selection, decision theory), use VBEM. If you only need a good point estimate (large data, predictive accuracy), Variational EM is faster and simpler.

---

## 10. Exercises

**E1.** Prove $\mathcal{L}(q) \leq \log p(\mathbf{x})$ from scratch.
*Sketch.* $\log p(\mathbf{x}) = \mathcal{L}(q) + \mathrm{KL}(q\|p(\cdot\mid\mathbf{x}))$ and KL is non-negative by Jensen.

**E2.** For a two-variable mean-field $q(z_1,z_2) = q_1(z_1)q_2(z_2)$, write the optimal $q_1^\star$.
*Answer.* $\log q_1^\star(z_1) = \mathbb{E}_{q_2}[\log p(\mathbf{x},z_1,z_2)] + \text{const}$.

**E3.** Derive the closed-form CAVI update for a bivariate Gaussian target with precision matrix $\Lambda$ and zero mean. Show that at convergence, $\mathrm{Var}_q[z_j] = 1/\Lambda_{jj}$ — the inverse of the diagonal of the precision, **not** the diagonal of the covariance.
*Hint.* Reproduce the trajectory in Figure 5.

**E4.** Show that the Variational EM ELBO is non-decreasing across iterations even though the E-step is no longer exact.
*Sketch.* The E-step increases the ELBO (improves $q$); the M-step increases the ELBO (improves $\boldsymbol{\theta}$). Monotonicity follows from the alternating structure.

**E5.** Why does direct Monte-Carlo gradient estimation $\nabla_\phi \mathbb{E}_{q_\phi}[f(\mathbf{z})] \approx \frac{1}{S}\sum_s \nabla_\phi f(\mathbf{z}^{(s)})$ with $\mathbf{z}^{(s)}\sim q_\phi$ give the wrong answer?
*Answer.* The samples themselves depend on $\phi$; the naive gradient ignores that dependency. REINFORCE and reparameterization are the two principled fixes.

---

## References

- Jordan, M. I., Ghahramani, Z., Jaakkola, T. S., & Saul, L. K. (1999). An introduction to variational methods for graphical models. *Machine Learning*, 37(2), 183–233.
- Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). Variational inference: A review for statisticians. *JASA*, 112(518), 859–877.
- Hoffman, M. D., Blei, D. M., Wang, C., & Paisley, J. (2013). Stochastic variational inference. *JMLR*, 14, 1303–1347.
- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *ICLR*.
- Ranganath, R., Gerrish, S., & Blei, D. M. (2014). Black box variational inference. *AISTATS*.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 10. Springer.

---

## Series Navigation

- Previous: [Part 13 -- EM Algorithm and GMM](/en/Machine-Learning-Mathematical-Derivations-13-EM-Algorithm-and-GMM/)
- Next: [Part 15 -- Hidden Markov Models](/en/Machine-Learning-Mathematical-Derivations-15-Hidden-Markov-Models/)
- [View all 20 parts in this series](/tags/Machine-Learning/)
