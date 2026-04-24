---
title: "Machine Learning Mathematical Derivations (13): EM Algorithm and GMM"
date: 2026-03-04 09:00:00
categories:
  - Machine Learning
tags:
  - EM Algorithm
  - Expectation Maximization
  - Gaussian Mixture Model
  - GMM
  - Latent Variables
  - Mathematical Derivation
  - Machine Learning
series:
  name: "ML Mathematical Derivations"
  order: 13
  total: 20
lang: en
mathjax: true
description: "Derive the EM algorithm from Jensen's inequality and the ELBO, prove its monotone-ascent guarantee, and apply it to Gaussian Mixture Models with full E-step / M-step formulas, model selection via BIC/AIC, and the K-means correspondence."
disableNunjucks: true
---

When data carries hidden structure -- a cluster label you never observed, a missing feature, a topic you cannot directly see -- maximum likelihood becomes painful. The log of a sum has no closed form, and gradient methods get tangled in the latent variables. The **EM algorithm** sidesteps the difficulty with a deceptively simple idea: alternate between *guessing* the hidden variables under a posterior (E-step) and *fitting* the parameters as if those guesses were true (M-step). Each iteration is mathematically guaranteed to push the likelihood up. This post derives EM from first principles, proves the monotone-ascent property via Jensen's inequality, and works through its most famous application: **Gaussian Mixture Models (GMM)** -- the soft, elliptical generalisation of K-means.

## What you will learn

- Why latent variables make MLE hard (the log-of-a-sum problem)
- How Jensen's inequality builds the **Evidence Lower Bound (ELBO)**
- The EM algorithm as **alternating maximisation** of the ELBO in $(q, \boldsymbol{\theta})$
- A clean proof that $\ell(\boldsymbol{\theta}^{(t+1)}) \geq \ell(\boldsymbol{\theta}^{(t)})$
- Complete E-step / M-step formulas for GMM with full covariance
- How K-means is the hard, spherical limit of GMM
- Picking $K$ with **BIC** and **AIC**

## Prerequisites

- Maximum likelihood estimation
- Multivariate Gaussian density
- Jensen's inequality and KL divergence
- K-means clustering

---

## 1. Latent variables and incomplete-data likelihood

### 1.1 Setup

We model observations $\mathbf{x}_1,\dots,\mathbf{x}_N$ together with hidden variables $z_1,\dots,z_N$ via a joint $p(\mathbf{x}, z \mid \boldsymbol{\theta})$. We see $\mathbf{X}$, never $\mathbf{Z}$. The **incomplete-data log-likelihood** is

$$
\ell(\boldsymbol{\theta}) \;=\; \sum_{i=1}^{N} \log p(\mathbf{x}_i \mid \boldsymbol{\theta})
\;=\; \sum_{i=1}^{N} \log \sum_{z} p(\mathbf{x}_i, z \mid \boldsymbol{\theta}).
$$

The summation **inside** the logarithm is the source of all the trouble. The log no longer factors over components, so the gradient does not split into per-component pieces and there is no closed-form maximiser.

### 1.2 The mixture example

For a Gaussian mixture with $K$ components,

$$
p(\mathbf{x}\mid \boldsymbol{\theta}) \;=\; \sum_{k=1}^{K} \pi_k\, \mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k).
$$

If we *knew* which component each point came from, fitting would reduce to $K$ independent weighted-Gaussian MLEs -- trivial. We do not know, and that is exactly what EM patches up.

![GMM components with covariance ellipses](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/13-EM-Algorithm-and-GMM/fig1_gmm_clusters.png)

The figure above shows three Gaussian clusters fit by `sklearn.mixture.GaussianMixture`: each cross is a mean $\boldsymbol{\mu}_k$, the inner ellipse is the 1-sigma contour, the outer is 2-sigma, and $\pi_k$ is the mixing weight.

---

## 2. The ELBO and Jensen's inequality

### 2.1 Introducing an auxiliary distribution $q$

Pick **any** distribution $q(z)$ over the latent variable. Multiply and divide:

$$
\log p(\mathbf{x}\mid \boldsymbol{\theta})
= \log \sum_{z} q(z)\, \frac{p(\mathbf{x}, z\mid \boldsymbol{\theta})}{q(z)}.
$$

Because $\log$ is concave, **Jensen's inequality** gives

$$
\boxed{\;
\log p(\mathbf{x}\mid \boldsymbol{\theta})
\;\geq\;
\sum_{z} q(z)\, \log \frac{p(\mathbf{x}, z\mid \boldsymbol{\theta})}{q(z)}
\;\equiv\;
\mathcal{L}(q,\boldsymbol{\theta}).
\;}
$$

This $\mathcal{L}$ is the **Evidence Lower Bound (ELBO)**. It depends on both the variational distribution $q$ and the parameters $\boldsymbol{\theta}$.

### 2.2 The exact decomposition

A direct manipulation -- no inequality needed -- yields the *equality*

$$
\log p(\mathbf{x}\mid \boldsymbol{\theta})
\;=\;
\mathcal{L}(q,\boldsymbol{\theta})
\;+\;
\mathrm{KL}\bigl[q(z)\,\Vert\, p(z\mid \mathbf{x},\boldsymbol{\theta})\bigr].
$$

Two consequences are immediate:

1. The ELBO is **always** $\leq \log p(\mathbf{x}\mid\boldsymbol{\theta})$ because $\mathrm{KL}\geq 0$.
2. The bound becomes **tight**, $\mathcal{L} = \log p$, **iff** $q(z) = p(z \mid \mathbf{x}, \boldsymbol{\theta})$ -- the posterior.

This single identity is the entire engine of EM.

---

## 3. EM as coordinate ascent on the ELBO

EM repeatedly raises $\mathcal{L}$ by alternating in its two arguments.

### 3.1 The two steps

**E-step.** Hold $\boldsymbol{\theta}^{(t)}$ fixed. Maximise $\mathcal{L}(q, \boldsymbol{\theta}^{(t)})$ over $q$. The maximiser is the posterior:

$$
q^{(t)}(z) \;=\; p\bigl(z \mid \mathbf{x}, \boldsymbol{\theta}^{(t)}\bigr).
$$

After this step the bound is **tight**: $\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)}) = \log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t)})$.

**M-step.** Hold $q^{(t)}$ fixed. Maximise $\mathcal{L}(q^{(t)}, \boldsymbol{\theta})$ over $\boldsymbol{\theta}$. Dropping the entropy of $q^{(t)}$ (constant in $\boldsymbol{\theta}$), this is the same as maximising the **Q-function**

$$
Q(\boldsymbol{\theta}\mid \boldsymbol{\theta}^{(t)})
\;=\;
\mathbb{E}_{z\sim q^{(t)}}\!\bigl[\log p(\mathbf{x}, z\mid \boldsymbol{\theta})\bigr].
$$

### 3.2 The monotone-ascent proof

Chain three inequalities:

$$
\log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t)})
\;\overset{(a)}{=}\;
\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)})
\;\overset{(b)}{\leq}\;
\mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t+1)})
\;\overset{(c)}{\leq}\;
\log p(\mathbf{x}\mid \boldsymbol{\theta}^{(t+1)}).
$$

(a) holds because the E-step makes the bound tight; (b) by definition of the M-step; (c) because the ELBO is *always* $\leq \log p$. Therefore

$$
\boxed{\;\ell(\boldsymbol{\theta}^{(t+1)}) \;\geq\; \ell(\boldsymbol{\theta}^{(t)})\;}
$$

at every iteration, with equality only at fixed points. EM converges to a stationary point of $\ell$ -- typically a local maximum, occasionally a saddle point. **It is not guaranteed to reach the global maximum**, which is why multiple random restarts matter.

### 3.3 Visualising the two views

The **ELBO view** makes the dynamics very concrete. After every E-step the KL gap closes; the M-step then raises both the log-likelihood and the ELBO together, and the gap re-opens until the next E-step.

![ELBO and log-likelihood across EM iterations](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/13-EM-Algorithm-and-GMM/fig7_elbo_vs_loglik.png)

The shaded amber band is exactly the KL divergence $\mathrm{KL}\bigl[q\,\Vert\, p(z\mid\mathbf{x},\boldsymbol{\theta})\bigr]$. Green dots mark the post-E-step instants where the gap is zero by construction.

---

## 4. EM for Gaussian Mixture Models

### 4.1 The model

Generative process for one observation:

1. Sample a component label $z_i \sim \mathrm{Categorical}(\pi_1,\dots,\pi_K)$.
2. Sample $\mathbf{x}_i \mid z_i = k \;\sim\; \mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$.

The parameters are $\boldsymbol{\theta} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}_{k=1}^{K}$, with $\sum_k \pi_k = 1$ and each $\boldsymbol{\Sigma}_k \succ 0$.

### 4.2 The E-step: responsibilities

The latent posterior is just Bayes' rule on a finite alphabet. Define the **responsibility** of component $k$ for sample $i$:

$$
\boxed{\;
\gamma_{ik}
\;=\;
p\bigl(z_i = k \mid \mathbf{x}_i, \boldsymbol{\theta}^{(t)}\bigr)
\;=\;
\frac{\pi_k\,\mathcal{N}(\mathbf{x}_i\mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \pi_j\,\mathcal{N}(\mathbf{x}_i\mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}.
\;}
$$

Each row $(\gamma_{i1},\dots,\gamma_{iK})$ sums to 1 -- the soft cluster membership.

![E-step soft assignments](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/13-EM-Algorithm-and-GMM/fig2_e_step.png)

On the left every grid point is coloured by mixing the three component colours according to $\gamma_{ik}$: pure colour where one component dominates, blended colours along the boundaries. On the right, the responsibility matrix $\gamma_{ik}$ for twelve sample points -- rows sum to 1.

### 4.3 The M-step: weighted MLE

Plugging the Gaussian density into $Q(\boldsymbol{\theta}\mid \boldsymbol{\theta}^{(t)})$ and maximising (with a Lagrange multiplier for $\sum_k \pi_k = 1$) gives the closed-form updates. Let $N_k = \sum_{i=1}^{N} \gamma_{ik}$ be the *effective* sample size of component $k$:

$$
\boxed{\;
\pi_k = \frac{N_k}{N},\qquad
\boldsymbol{\mu}_k = \frac{1}{N_k}\sum_{i=1}^{N} \gamma_{ik}\,\mathbf{x}_i,\qquad
\boldsymbol{\Sigma}_k = \frac{1}{N_k}\sum_{i=1}^{N} \gamma_{ik}\,(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^{\!\top}.
\;}
$$

These are exactly the standard Gaussian MLE formulas, but with each sample re-weighted by its responsibility.

![One M-step update -- before vs after](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/13-EM-Algorithm-and-GMM/fig3_m_step.png)

Starting from a deliberately bad initialisation, a single M-step pulls the means (red arrows) onto the data and stretches the covariance ellipses to match the observed scatter. After only a handful of E-M cycles the fit is essentially correct.

### 4.4 Convergence in practice

Run EM for several random restarts and watch the log-likelihood:

![Log-likelihood is monotone non-decreasing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/13-EM-Algorithm-and-GMM/fig4_loglik_monotone.png)

Every restart curve is non-decreasing -- this is the algorithmic guarantee. Different restarts plateau at different basins; the dashed line is the best value found by `sklearn.mixture.GaussianMixture` with `n_init=20`. **Use multiple random or K-means initialisations and keep the best.**

---

## 5. K-means is the hard, spherical limit of GMM

Let $\boldsymbol{\Sigma}_k = \epsilon \mathbf{I}$ for all $k$ and let $\epsilon \to 0$. The Gaussian density becomes infinitely peaked; the responsibility for the **closest** mean tends to 1 and the others to 0. The E-step degenerates to *hard assignment* and the M-step to averaging the assigned points -- exactly K-means.

![K-means vs GMM on anisotropic data](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/13-EM-Algorithm-and-GMM/fig6_kmeans_vs_gmm.png)

On anisotropic data the difference is stark: K-means (left) imposes spherical Voronoi cells and slices the elongated cluster awkwardly; GMM (right) fits an ellipse along the actual axis of variation. **GMM should be your default whenever clusters are not isotropic or you want soft membership probabilities.**

---

## 6. Choosing the number of components

The likelihood always increases with $K$ (more flexibility), so $\ell$ alone cannot pick $K$. Use a complexity-penalised criterion:

$$
\mathrm{BIC}(K) = -2\,\hat{\ell}(K) + p_K\,\log N,
\qquad
\mathrm{AIC}(K) = -2\,\hat{\ell}(K) + 2\,p_K,
$$

where $p_K$ is the parameter count: for full-covariance GMM in $d$ dimensions,
$p_K = (K-1) + Kd + K\frac{d(d+1)}{2}$.

![BIC and AIC vs number of components K](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/13-EM-Algorithm-and-GMM/fig5_bic_aic.png)

Both curves drop sharply going from $K=1$ to the true $K=3$ and then flatten or rise. BIC penalises complexity more aggressively (an extra $\log N$ factor), so it tends to prefer smaller $K$ than AIC. Either gives the right answer here.

---

## 7. Reference implementation

A minimal NumPy implementation that mirrors the formulas above. The actual experiments and figures use `sklearn.mixture.GaussianMixture` for verification.

```python
import numpy as np
from scipy.stats import multivariate_normal


class GMM:
    """Full-covariance Gaussian Mixture Model trained by EM."""

    def __init__(self, n_components=3, max_iter=100, tol=1e-4, reg=1e-6):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg = reg

    # ----- E-step: responsibilities -----
    def _e_step(self, X):
        comp = np.column_stack([
            self.weights[k] * multivariate_normal.pdf(
                X, self.means[k], self.covs[k])
            for k in range(self.K)
        ])
        ll = np.log(comp.sum(axis=1) + 1e-300).sum()
        return comp / (comp.sum(axis=1, keepdims=True) + 1e-300), ll

    # ----- M-step: weighted MLE -----
    def _m_step(self, X, gamma):
        N_k = gamma.sum(axis=0)
        self.weights = N_k / X.shape[0]
        self.means = (gamma.T @ X) / N_k[:, None]
        d = X.shape[1]
        for k in range(self.K):
            diff = X - self.means[k]
            self.covs[k] = (gamma[:, k, None] * diff).T @ diff / N_k[k]
            self.covs[k] += self.reg * np.eye(d)

    def fit(self, X):
        N, d = X.shape
        rng = np.random.default_rng(0)
        idx = rng.choice(N, self.K, replace=False)
        self.means = X[idx].copy()
        self.weights = np.full(self.K, 1.0 / self.K)
        self.covs = np.array([np.cov(X.T) + self.reg * np.eye(d)] * self.K)

        prev = -np.inf
        for it in range(self.max_iter):
            gamma, ll = self._e_step(X)
            self._m_step(X, gamma)
            if abs(ll - prev) < self.tol:
                break
            prev = ll
        return self

    def predict(self, X):
        return np.argmax(self._e_step(X)[0], axis=1)
```

---

## Q&A

**Q1: Does EM reach the global optimum?** No. Monotone ascent guarantees only a stationary point -- typically a local maximum, sometimes a saddle. Defend yourself with multiple restarts (random or K-means seeded) and keep the run with the highest final log-likelihood.

**Q2: GMM vs K-means -- when does it actually matter?** Use GMM when (i) clusters are clearly elliptical / anisotropic, (ii) you want **soft** membership probabilities for downstream calibration, or (iii) you need a generative density model for sampling or anomaly scoring. K-means is faster and fine for roughly spherical, well-separated clusters.

**Q3: Singular covariance / collapsed component?** Add a ridge $\boldsymbol{\Sigma}_k + \epsilon \mathbf{I}$ (the reference implementation does this), tie covariances across components, restrict to diagonal covariance, or restart on detection.

**Q4: Why does the E-step "make the bound tight"?** Because $\log p = \mathcal{L}(q,\boldsymbol{\theta}) + \mathrm{KL}[q\Vert p(z\mid\mathbf{x},\boldsymbol{\theta})]$ and the KL is zero exactly when $q$ equals the posterior.

**Q5: What is generalised EM?** Replace the M-step's full maximisation with any update that *increases* $Q$ (e.g. one gradient step). The monotone ascent argument still goes through.

**Q6: How is EM related to variational inference?** Variational EM relaxes the E-step from the exact posterior to a tractable family $q \in \mathcal{Q}$. The decomposition $\log p = \mathcal{L} + \mathrm{KL}$ is identical; the algorithm now alternately minimises KL inside $\mathcal{Q}$ (E) and maximises $\mathcal{L}$ in $\boldsymbol{\theta}$ (M). See Part 14.

---

## Exercises

**E1 (E-step).** Consider a 1D GMM with $K=2$, equal priors $\pi_1 = \pi_2 = 1/2$, $\mu_1 = 0,\, \mu_2 = 3,\, \sigma^2 = 1$. Compute $\gamma_{i1}$ for $x_i = 1.5$.

*Solution.* By symmetry the two component densities at $x=1.5$ are equal, so $\gamma_{i1} = 1/2$.

**E2 (M-step).** Two samples $x_1 = 1,\; x_2 = 4$ with responsibilities $\gamma_{11} = 0.8,\; \gamma_{21} = 0.3$. Compute $\mu_1$ after the M-step.

*Solution.* $N_1 = 0.8 + 0.3 = 1.1$, so $\mu_1 = (0.8 \cdot 1 + 0.3 \cdot 4) / 1.1 = 2.0 / 1.1 \approx 1.82$.

**E3 (Monotonicity).** Where in the chain $\log p(\boldsymbol{\theta}^{(t)}) = \mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t)}) \leq \mathcal{L}(q^{(t)},\boldsymbol{\theta}^{(t+1)}) \leq \log p(\boldsymbol{\theta}^{(t+1)})$ does the M-step's optimality enter, and where does the ELBO inequality enter?

*Solution.* The middle $\leq$ is by the M-step (definition of $\boldsymbol{\theta}^{(t+1)}$); the right $\leq$ is the ELBO inequality applied at the new parameters.

**E4 (Singular limit).** Show that if you fix $\boldsymbol{\Sigma}_k = \epsilon \mathbf{I}$ for all $k$ and let $\epsilon \to 0^+$, the EM updates for the means converge to the K-means updates.

*Sketch.* Write $\mathcal{N}(\mathbf{x}\mid\boldsymbol{\mu}_k, \epsilon\mathbf{I}) \propto \exp(-\Vert \mathbf{x} - \boldsymbol{\mu}_k\Vert^2 / (2\epsilon))$. As $\epsilon \to 0$ the soft-max over $-\Vert\mathbf{x}-\boldsymbol{\mu}_k\Vert^2$ becomes a hard argmin, $\gamma_{ik}\in\{0,1\}$, and the M-step mean reduces to the cluster-mean update of K-means.

**E5 (Missing data).** Suppose feature $j$ of $\mathbf{x}_i$ is missing. Outline an EM scheme that treats the missing value as an additional latent variable and updates parameters and missing entries jointly.

*Sketch.* Add the missing entry $x_{ij}$ to $z$. The E-step computes $\mathbb{E}[x_{ij} \mid \text{observed},\boldsymbol{\theta}^{(t)}]$ (and second moments). The M-step uses these expectations as plug-ins in the weighted-MLE updates above.

---

## References

- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1-22.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. **Chapter 9**.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. **Chapter 11**.
- McLachlan, G., & Krishnan, T. (2007). *The EM Algorithm and Extensions* (2nd ed.). Wiley.
- Neal, R. M., & Hinton, G. E. (1998). A view of the EM algorithm that justifies incremental, sparse, and other variants. *Learning in Graphical Models*, 89, 355-368.

---

## Series Navigation

- Previous: [Part 12 -- XGBoost and LightGBM](/en/Machine-Learning-Mathematical-Derivations-12-XGBoost-and-LightGBM/)
- Next: [Part 14 -- Variational Inference](/en/Machine-Learning-Mathematical-Derivations-14-Variational-Inference-and-Variational-EM/)
- [View all 20 parts in this series](/tags/Machine-Learning/)
