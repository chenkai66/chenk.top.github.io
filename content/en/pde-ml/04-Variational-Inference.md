---
title: "PDE and ML (4): Variational Inference and the Fokker-Planck Equation"
date: 2024-06-15 09:00:00
tags:
  - PDE
  - Variational Inference
  - Fokker-Planck
  - ELBO
  - Langevin Dynamics
categories: PDE and Machine Learning
series: pde-ml
lang: en
mathjax: true
description: "Variational inference and Langevin MCMC are two faces of the same Fokker-Planck PDE. We derive the equivalence, build SVGD as an interacting-particle approximation, and quantify convergence under log-Sobolev inequalities."
disableNunjucks: true
series_order: 4
series_total: 8
translationKey: "pde-ml-4"
---
![PDE and ML (4): Variational Inference and the Fokker-Planck Equation — Chapter overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/illustration_1.png)

---

Why do variational inference (a method that looks purely optimization) and Langevin MCMC (a method that looks purely sampling) end up at the same partial differential equation?

That is the heart of this article. In continuous time, they are **two faces of the same Fokker–Planck PDE**: one face is the evolution of a density, the other is the Wasserstein gradient flow of KL divergence. Once you see this, several seemingly unrelated tools — the SVGD particle algorithm, the exponential convergence rate from a log-Sobolev inequality, the training of Bayesian neural networks — all snap onto a single picture.

If you've worked with VI or with Langevin samplers but always felt they lived in different worlds, this article is the bridge.

## Seven Dimensions of This Article

1. **Motivation**: why VI and MCMC look different but solve the same PDE.
2. **Theory**: derivation of the Fokker-Planck equation from the SDE.
3. **Geometry**: KL divergence as a Wasserstein gradient flow.
4. **Algorithms**: Langevin Monte Carlo, mean-field VI, and SVGD.
5. **Convergence**: log-Sobolev inequality and exponential KL decay.
6. **Numerical experiments**: 7 figures with reproducible code.
7. **Application**: Bayesian neural networks via posterior sampling.

## What You Will Learn

- How the Fokker-Planck equation governs probability density evolution from any It&ocirc; SDE.
- Langevin dynamics as a practical sampling algorithm and its discretization error.
- Why minimizing $\mathrm{KL}(q\|p^\star)$ in Wasserstein space *is* the Fokker-Planck PDE.
- The deep equivalence between variational inference and Langevin MCMC in continuous time.
- Stein Variational Gradient Descent (SVGD): a deterministic particle method that bridges both worlds.
- Practical posterior inference for Bayesian neural networks.

## Prerequisites

- Probability theory (Bayes' rule, KL divergence, expectations).
- Wasserstein gradient flows from [Part 3](/en/pde-ml/03-variational-principles/).
- Light stochastic calculus intuition (Brownian motion, It&ocirc; integral).
- Python / PyTorch for the experiments.

---

## The Inference Problem

Bayesian inference asks for the posterior
$$p(\theta \mid x) \;=\; \frac{p(x \mid \theta)\, p(\theta)}{\int p(x \mid \theta')\, p(\theta')\, d\theta'},$$
but the marginal likelihood in the denominator is intractable for any non-trivial model. Two large families of approximation algorithms address this:

- **Variational inference (VI)**: pick a tractable family $\{q_\phi\}$ and minimise
  
$$
\mathrm{KL}\bigl(q_\phi \,\|\, p(\cdot\mid x)\bigr) \;=\; \mathbb{E}_{q_\phi}\!\left[\log \tfrac{q_\phi(\theta)}{p(\theta\mid x)}\right],
$$
  equivalently maximising the **Evidence Lower Bound** $\mathrm{ELBO}(\phi) = \mathbb{E}_{q_\phi}[\log p(x\mid\theta)] - \mathrm{KL}(q_\phi \| p(\theta))$.

- **Markov Chain Monte Carlo (MCMC)**: build a Markov chain whose stationary distribution is exactly $p(\cdot\mid x)$. **Langevin dynamics** is the canonical gradient-based instance.

These look like very different objects: VI is a finite-dimensional optimisation over $\phi$, while MCMC is an infinite-time stochastic process. The PDE viewpoint reveals they are the **same evolution of probability measures**, only sampled differently.

## From SDE to Fokker-Planck

Consider an It&ocirc; SDE
$$dX_t = \mu(X_t, t)\, dt + \sigma(X_t, t)\, dW_t.$$
For any smooth test function $f$, It&ocirc;'s lemma plus integration by parts (assuming the density and its derivatives vanish at infinity) gives the **Fokker-Planck (FP) equation**:
$$
\boxed{\;\partial_t p \;=\; -\nabla\!\cdot\!(\mu\, p) \;+\; \tfrac{1}{2}\,\nabla\!\cdot\!\nabla\!\cdot\!(D\, p),\qquad D = \sigma\sigma^\top.\;}
$$
The first term is **drift** (transport), the second is **diffusion** (spreading). For the **overdamped Langevin SDE** with $\mu = -\nabla V$, $\sigma = \sqrt{2\tau} I$:
$$\partial_t p \;=\; \nabla\!\cdot\!\bigl(p\,\nabla V\bigr) + \tau\, \Delta p,$$
whose unique stationary solution (under mild regularity) is the **Gibbs distribution** $p_\infty \propto e^{-V/\tau}$. Setting $V = -\log p^\star$ and $\tau = 1$, the stationary distribution becomes the target $p^\star$ exactly.

![Density evolution under the Fokker-Planck equation in a double-well potential.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig1_fokker_planck_evolution.png)
*Figure 1. Solving the FP equation by finite differences. Starting from a narrow Gaussian on the left well, the density spreads, hops the barrier, and converges to the symmetric Gibbs density $p_\infty \propto e^{-V/D}$ (right panel).*

### Hamiltonian Monte Carlo: Momentum Beats Random Walks

ULA explores by random diffusion, which is painfully slow in high dimensions or across energy barriers. **Hamiltonian Monte Carlo (HMC)** introduces auxiliary momentum $v$ and simulates Hamiltonian dynamics on the joint energy $H(\theta, v) = V(\theta) + \frac{1}{2}\|v\|^2$. The momentum lets the sampler "roll" across low-density valleys instead of waiting for noise to kick it over.

The leapfrog integrator preserves volume (symplecticity) and is time-reversible, which guarantees detailed balance after a Metropolis correction:

```python
import numpy as np

def hmc_sample(V, grad_V, x0, step=0.02, L=20, n_samples=2000):
    # Hamiltonian Monte Carlo with leapfrog integration
    # grad_V: gradient of the potential energy V(x) = -log p*(x)
    # L: number of leapfrog steps per proposal
    d = x0.shape[0]
    x = x0.copy()
    samples = [x.copy()]
    accepted = 0

    for _ in range(n_samples):
        v = np.random.randn(d)  # sample momentum
        x_prop, v_prop = x.copy(), v.copy()

        # Leapfrog integration
        v_prop = v_prop - 0.5 * step * grad_V(x_prop)
        for l_step in range(L - 1):
            x_prop = x_prop + step * v_prop
            v_prop = v_prop - step * grad_V(x_prop)
        x_prop = x_prop + step * v_prop
        v_prop = v_prop - 0.5 * step * grad_V(x_prop)

        # Metropolis accept/reject
        H_current = 0.5 * np.dot(v, v) + V(x)      # current Hamiltonian
        H_proposed = 0.5 * np.dot(v_prop, v_prop) + V(x_prop)
        log_alpha = H_current - H_proposed

        if np.log(np.random.rand()) < log_alpha:
            x = x_prop
            accepted += 1

        samples.append(x.copy())

    print(f"HMC acceptance rate: {accepted / n_samples:.2%}")
    return np.array(samples)
```sql

Why does HMC beat ULA? Consider a double-well potential with a barrier of height $B$. ULA needs $O(e^B / \eta)$ steps to cross; HMC only needs enough kinetic energy, which happens with probability $\sim e^{-B}$ per sample of $v$. The leapfrog trajectory then carries the particle ballistically across the barrier in $L$ steps. This reduces mixing time from exponential-in-$B$ (diffusion) to polynomial (ballistic transport).

The PDE perspective: HMC corresponds to the **underdamped** (kinetic) Langevin equation $d\theta = v\,dt$, $dv = -\nabla V(\theta)\,dt - \gamma v\,dt + \sqrt{2\gamma}\,dW$, whose Fokker-Planck has second-order structure and faster convergence than the overdamped case.

## Langevin Dynamics: Sampling as a PDE

![Langevin dynamics: 200 particles sampling a double-well potential converging to Gibbs equilibrium](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/anim_langevin_sampling.gif)

The **overdamped Langevin equation** for sampling from $p^\star \propto e^{-V}$ is
$$dX_t = -\nabla V(X_t)\, dt + \sqrt{2\tau}\, dW_t.$$
The discrete-time **Unadjusted Langevin Algorithm (ULA)** is the Euler-Maruyama scheme
$$X_{k+1} \;=\; X_k - \eta\, \nabla V(X_k) + \sqrt{2\eta\tau}\, \xi_k, \qquad \xi_k \sim \mathcal{N}(0, I).$$
```python
import torch, numpy as np

def langevin_sample(grad_log_p, x0, step=0.01, n_steps=10_000, tau=1.0):
    """Overdamped Langevin sampler (a.k.a. ULA).

    grad_log_p : callable returning grad(log p*(x))
    x0         : (n_particles, dim) initial positions
    """
    x = x0.clone()
    for _ in range(n_steps):
        x = x + step * grad_log_p(x) + np.sqrt(2 * step * tau) * torch.randn_like(x)
    return x
```sql

ULA's bias is $O(\eta)$; **MALA** (Metropolis-Adjusted Langevin) restores exactness via an accept-reject step. **HMC** (Hamiltonian Monte Carlo) is the natural underdamped analogue with momentum.

![Langevin SDE trajectories and the empirical density they generate.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig2_langevin_sde_to_density.png)
*Figure 2. Left: 25 representative particles bouncing inside the double well; many never cross the barrier in finite time. Right: the histogram of 400 particles converges to the Gibbs target as $t$ grows — the discrete sampler is realising the FP equation in figure 1.*

### Stochastic Gradient Langevin Dynamics (SGLD)

Full-batch Langevin requires evaluating $\nabla V(\theta) = -\sum_{i=1}^N \nabla \log p(x_i \mid \theta) - \nabla \log p(\theta)$ over the entire dataset at every step. For $N = 10^6$ this is prohibitive. **SGLD** (Welling and Teh, 2011) replaces the full gradient with a mini-batch estimate and lets the injected noise serve double duty as both the SDE diffusion and a regularizer:

$$\theta_{k+1} = \theta_k + \frac{\eta_k}{2}\left(\nabla \log p(\theta_k) + \frac{N}{B}\sum_{i \in \mathcal{B}_k} \nabla \log p(x_i \mid \theta_k)\right) + \sqrt{\eta_k}\,\xi_k$$

where $\mathcal{B}_k$ is a mini-batch of size $B$ and $\xi_k \sim \mathcal{N}(0, I)$. The key insight: as the step size $\eta_k \to 0$ on a schedule, the mini-batch noise $O(\eta_k)$ dominates the injected noise $O(\sqrt{\eta_k})$, and the algorithm transitions smoothly from SGD (optimization) to Langevin (sampling).

```python
import numpy as np

def sgld(grad_log_prior, grad_log_likelihood, x0, data,
         batch_size=64, n_steps=50000, eta0=1e-3, decay=0.9999):
    # Stochastic Gradient Langevin Dynamics
    # grad_log_prior: gradient of log p(theta)
    # grad_log_likelihood: gradient of log p(x_i | theta) for one data point
    N = len(data)
    d = x0.shape[0]
    x = x0.copy()
    samples = []

    for k in range(n_steps):
        eta = eta0 * (decay ** k)
        # Mini-batch gradient estimate
        batch_idx = np.random.choice(N, batch_size, replace=False)
        grad_lik = np.zeros(d)
        for i in batch_idx:
            grad_lik += grad_log_likelihood(x, data[i])
        grad_lik *= (N / batch_size)  # scale to full dataset

        grad_total = grad_log_prior(x) + grad_lik
        noise = np.sqrt(eta) * np.random.randn(d)
        x = x + 0.5 * eta * grad_total + noise

        if k % 100 == 0:
            samples.append(x.copy())

    return np.array(samples)
```sql

In practice SGLD is the workhorse behind "Bayesian deep learning at scale" because it reuses the same mini-batch infrastructure as SGD. The cost per step is identical to standard training; the only addition is the noise injection $\sqrt{\eta_k}\,\xi_k$. The trade-off: finite step sizes introduce asymptotic bias, and diagnosing convergence requires monitoring the noise-to-signal ratio of the gradient estimator.

## KL Divergence is a Wasserstein Gradient Flow

Decompose the KL divergence relative to $p^\star \propto e^{-V}$:
$$\mathcal{F}[p] \;=\; \mathrm{KL}(p\,\|\,p^\star) \;=\; \underbrace{\int p\log p\,dx}_{\text{neg-entropy }\mathcal{H}[p]} \;+\; \underbrace{\int p\, V\,dx}_{\text{potential energy}} \;+\; \text{const}.$$
This is the **free energy functional** from [Part 3](/en/pde-ml/03-variational-principles/). The Jordan-Kinderlehrer-Otto (JKO) theorem (1998) tells us its **Wasserstein-2 gradient flow** is
$$\partial_t p \;=\; \nabla\!\cdot\!\bigl(p \nabla V\bigr) + \Delta p,$$
which is exactly the FP equation for Langevin with $\tau = 1$. Hence:

> **Equivalence**. Minimising $\mathrm{KL}(\cdot \| p^\star)$ in Wasserstein space and running Langevin dynamics targeting $p^\star$ are the **same PDE**. VI and Langevin MCMC are two algorithmic discretisations of one continuous-time gradient flow.

| Aspect | Variational Inference | Langevin MCMC |
|---|---|---|
| Objective | Minimise $\mathrm{KL}(q_\phi \mid p^\star)$ | Sample from $p^\star$ |
| State | Parameters $\phi$ | Particles $\{X^{(i)}\}$ |
| Step | Gradient step on ELBO | Euler-Maruyama on SDE |
| Continuous limit | Wasserstein gradient flow of KL | Fokker-Planck equation |
| Stationary | $q^\star = p^\star$ (if expressive) | $p_\infty = p^\star$ |
| Bias | Restricted family + Adam noise | Discretisation $O(\eta)$ |

![KL divergence as a Wasserstein gradient flow.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig3_kl_gradient_flow.png)
*Figure 3. Two initial densities (concentrated and broad) are evolved by the FP equation toward the bimodal target. The right panel shows their KL divergence to $p^\star$ decaying monotonically — the gradient-flow guarantee in action.*

### Sampler Comparison: ULA vs MALA vs HMC vs SGLD

The table below summarizes the four gradient-based MCMC algorithms we have discussed. All target $p^\star \propto e^{-V}$ in $d$ dimensions with condition number $\kappa = L/m$ (ratio of smoothness to strong convexity).

| Algorithm | Cost per step | Bias | Convergence (TV to $\varepsilon$) | Best regime |
|-----------|--------------|------|-----------------------------------|-------------|
| **ULA** (Unadjusted Langevin) | 1 gradient eval | $O(\eta d)$ asymptotic | $\tilde{O}(\kappa^2 d / \varepsilon^2)$ steps | Low-$d$, fast gradients, tolerant of bias |
| **MALA** (Metropolis-Adjusted) | 1 gradient + 1 density eval | None (exact) | $\tilde{O}(\kappa d^{1/3} / \varepsilon^{2/3})$ steps | Moderate-$d$, need unbiased samples |
| **HMC** (Hamiltonian MC) | $L$ gradient evals | None (exact) | $\tilde{O}(\kappa^{1/2} d^{1/4})$ steps | High-$d$, smooth targets, Stan/PyMC |
| **SGLD** (Stochastic Gradient) | 1 mini-batch gradient | $O(\eta + \sigma^2_B \eta)$ | No clean bound (non-stationary) | Large-$N$ datasets, Bayesian DL |

Key observations:

- **HMC is the gold standard** for moderate dimensions ($d < 10^4$) when full gradients are affordable. Its $d^{1/4}$ scaling crushes ULA's $d$ dependence.
- **SGLD wins on wall-clock time** when $N$ is large because each step costs $O(B)$ instead of $O(N)$, but the asymptotic bias never vanishes at fixed $\eta$.
- **MALA** is the natural "fix" for ULA's bias but gains relatively little in high dimensions compared to HMC.
- All four are instances of the same Fokker-Planck flow, differing only in their discretization scheme and whether momentum is included.

## VI vs MCMC in Practice

VI and MCMC may be equivalent in the continuous limit, but their finite-time behaviour differs dramatically.

- **VI minimising $\mathrm{KL}(q\|p^\star)$ is mode-seeking**: when $q$ is restricted to a simple family, the variational optimum collapses onto a single mode and underestimates uncertainty.
- **MCMC is mass-covering**: a long enough chain visits every mode in proportion to its mass, but mixing across barriers can be exponentially slow.

![VI vs MCMC.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig4_vi_vs_mcmc.png)
*Figure 4. Left: the best mean-field Gaussian (in reverse KL) under-fits one mode of a bimodal posterior. Right: 4000 Langevin samples cover both modes correctly — but only because the barrier in this 1D example is small.*

## Stein Variational Gradient Descent

SVGD (Liu and Wang, 2016) is a **deterministic** particle method that occupies the sweet spot between VI and MCMC. Maintain particles $\{x_i\}_{i=1}^n$ and update
$$x_i \;\leftarrow\; x_i + \eta\, \hat\phi^*(x_i),\qquad \hat\phi^*(x) = \tfrac{1}{n}\sum_{j=1}^n \Bigl[\,k(x_j, x)\,\nabla_{x_j}\log p^\star(x_j) \;+\; \nabla_{x_j} k(x_j, x)\,\Bigr],$$
with RBF kernel $k(x, y) = \exp(-\|x-y\|^2 / 2h^2)$ (median heuristic for $h$). The update has two terms with opposite roles:

- **Drift** $k\,\nabla\log p^\star$ pushes particles toward high probability.
- **Repulsion** $\nabla k$ pushes particles apart, preventing collapse onto a single mode.

```python
import numpy as np
from scipy.spatial.distance import cdist

def svgd_step(x, score, eta=0.05):
    n = x.shape[0]
    sq = cdist(x, x) ** 2
    h  = np.sqrt(0.5 * np.median(sq) / np.log(n + 1)) + 1e-6
    K  = np.exp(-sq / (2 * h**2))
    grad_K = -(x[:, None] - x[None, :]) / h**2 * K[..., None]   # (n, n, d)
    phi = (K @ score - grad_K.sum(axis=0)) / n
    return x + eta * phi
```text

In the infinite-particle limit SVGD obeys
$$\partial_t p \;=\; -\nabla\!\cdot\!\bigl(p\, v[p]\bigr),\qquad v[p](x) = \mathbb{E}_{y \sim p}\!\bigl[k(y,x)\nabla\log p^\star(y) + \nabla_y k(y,x)\bigr],$$
and as the bandwidth $h \to 0$ this PDE collapses to the standard Fokker-Planck equation. SVGD is therefore a **kernel-smoothed FP solver**.

![SVGD particles on a bimodal target.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig5_svgd_particles.png)
*Figure 5. Left: snapshots of 80 SVGD particles starting at the origin and splitting to populate both modes within a few hundred iterations. Right: full particle trajectories show the kernel repulsion preventing collapse, while the drift term locks particles around $\pm 2$.*

### Adaptive Bandwidth and Non-Gaussian Posteriors

The median heuristic $h = \text{med}(\|x_i - x_j\|^2) / \log n$ is simple but fragile. It fails in two common scenarios:

1. **Multimodal targets with unequal scales**: if one mode is tight and the other is diffuse, a single global bandwidth cannot simultaneously provide repulsion within the tight cluster and attraction across the gap.
2. **Banana-shaped (strongly correlated) posteriors**: the pairwise distances are dominated by the long axis, making $h$ too large for the narrow direction. Particles slide along the banana but never fill its width.

A practical fix is **per-particle adaptive bandwidth**: compute a local bandwidth $h_i$ based on the $k$-nearest-neighbor distance of particle $i$. This gives a spatially-varying kernel that adapts to the local geometry:

```python
import numpy as np
from scipy.spatial.distance import cdist

def svgd_adaptive(x, score_fn, eta=0.05, k_neighbors=5):
    # SVGD with per-particle adaptive bandwidth
    n, d = x.shape
    score = score_fn(x)  # (n, d)
    dists = cdist(x, x)  # (n, n)

    # Per-particle bandwidth from k-NN distance
    sorted_dists = np.sort(dists, axis=1)
    h_local = sorted_dists[:, k_neighbors]  # distance to k-th neighbor
    h_local = np.maximum(h_local, 1e-6)

    # Compute kernel with geometric mean of bandwidths
    h_matrix = np.sqrt(np.outer(h_local, h_local))  # (n, n)
    K = np.exp(-dists**2 / (2 * h_matrix**2))

    # Kernel gradient: d/dx_j k(x_j, x_i)
    diff = x[:, None, :] - x[None, :, :]  # (n, n, d)
    grad_K = -diff / (h_matrix[:, :, None]**2) * K[:, :, None]

    # SVGD update
    phi = (K @ score + grad_K.sum(axis=0)) / n
    return x + eta * phi
```sql

**Banana posterior example.** Consider the 2D distribution $p^\star(x_1, x_2) \propto \exp\bigl(-\frac{1}{2}(x_1^2/s_1^2 + (x_2 - x_1^2)^2/s_2^2)\bigr)$ with $s_1 = 2, s_2 = 0.5$. This creates a narrow, curved ridge that global-bandwidth SVGD struggles to fill. With adaptive bandwidth, particles spread along the entire banana within 500 iterations, whereas median-heuristic SVGD collapses to the bend at the origin.

The lesson generalizes: in real Bayesian posteriors (which are rarely Gaussian), local geometric adaptation is not optional --- it is the difference between coverage and mode collapse. Matrix-valued kernels (Wang et al., 2019) push this further by using a full $d \times d$ metric tensor per particle.

## Convergence Theory

**Definition (LSI).** $p^\star$ satisfies a **log-Sobolev inequality** with constant $\lambda > 0$ if for all smooth probability densities $p \ll p^\star$:
$$\mathrm{KL}(p \,\|\, p^\star) \;\leq\; \frac{1}{2\lambda}\, I(p \,\|\, p^\star),\qquad I(p\|p^\star) = \int p\, \bigl\|\nabla \log \tfrac{p}{p^\star}\bigr\|^2 dx.$$
The right-hand side is the **Fisher information** — which is also the dissipation rate of the FP equation:
$$\frac{d}{dt}\,\mathrm{KL}(p_t\,\|\,p^\star) \;=\; -\, I(p_t\,\|\,p^\star).$$
Combining, $\frac{d}{dt}\mathrm{KL}(p_t \| p^\star) \leq -2\lambda\, \mathrm{KL}(p_t \| p^\star)$, hence Gr&ouml;nwall:
$$
\boxed{\;\mathrm{KL}(p_t\,\|\,p^\star) \;\leq\; e^{-2\lambda t}\, \mathrm{KL}(p_0\,\|\,p^\star).\;}
$$
Strongly log-concave targets ($\nabla^2 V \succeq mI$) automatically satisfy LSI with $\lambda \geq m$ (Bakry-&Eacute;mery). Multimodal targets typically have tiny $\lambda$, predicting the exponentially slow mixing observed empirically.

![Convergence rates.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig6_convergence_analysis.png)
*Figure 6. Left: theoretical KL decay $e^{-2\lambda t}$ for three log-Sobolev constants. Right: empirical KL trajectories for VI, Langevin MCMC and SVGD on a smooth Gaussian target — all three converge, but with different rates and noise profiles.*

## Application: Bayesian Neural Networks

A Bayesian neural network places a prior $p(w)$ on the weights and seeks the posterior $p(w \mid \mathcal{D}) \propto p(\mathcal{D} \mid w)\, p(w)$. Even for tiny architectures the posterior is intractable, but Langevin dynamics on $w$ requires only
$$\nabla_w \log p(w \mid \mathcal{D}) \;=\; \nabla_w \log p(\mathcal{D} \mid w) + \nabla_w \log p(w),$$
i.e. exactly the gradient computed during normal back-propagation, plus a Gaussian prior term. **Stochastic Gradient Langevin Dynamics** (Welling and Teh, 2011) replaces the full-data gradient with a mini-batch estimate, making this practical at modern scale.

The figure below uses 24 random Fourier features as a tractable "Bayesian NN" so that the posterior over weights is well defined, and samples it with full-batch Langevin.

![Bayesian neural net posterior bands.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig7_bayesian_nn.png)
*Figure 7. Left: posterior predictive for a regression model with a gap in the training data; the 90% Langevin band widens precisely where data is missing. Right: predictive standard deviation peaks in the data gap — exactly the **epistemic uncertainty** that point-estimate networks lack.*

### Full Experiment: BNN Uncertainty on a 1D Regression Task

To make Bayesian uncertainty tangible, we train a small neural network on synthetic data with a deliberate gap, then sample the posterior with SGLD. The prediction bands should widen precisely where data is missing.

```python
import numpy as np

np.random.seed(42)
x_left = np.random.uniform(-2, 1, 40)
x_right = np.random.uniform(3, 6, 40)
x_train = np.concatenate([x_left, x_right])
y_train = np.sin(x_train) + 0.1 * np.random.randn(len(x_train))

def init_weights():
    W1 = np.random.randn(20, 1) * 0.5
    b1 = np.zeros(20)
    W2 = np.random.randn(1, 20) * 0.5
    b2 = np.zeros(1)
    return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

def forward(params, x):
    W1 = params[:20].reshape(20, 1)
    b1 = params[20:40]
    W2 = params[40:60].reshape(1, 20)
    b2 = params[60:61]
    h = np.tanh(x.reshape(-1, 1) @ W1.T + b1)  # (N, 20)
    return (h @ W2.T + b2).ravel()  # (N,)

def grad_log_posterior(params, x_batch, y_batch, N, sigma_y=0.1, sigma_w=1.0):
    # Compute gradient of log p(params | data) using finite differences
    # In practice you would use autograd; this is for clarity
    B = len(x_batch)
    pred = forward(params, x_batch)
    residual = y_batch - pred

    # Log-likelihood gradient (Gaussian noise model)
    eps = 1e-5
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        pred_plus = forward(params_plus, x_batch)
        dll_di = np.sum(residual * (pred_plus - pred) / eps) / sigma_y**2
        grad[i] = dll_di

    # Scale mini-batch to full dataset + prior
    grad = (N / B) * grad - params / sigma_w**2
    return grad

def sgld_bnn(x_train, y_train, n_steps=20000, batch_size=16, eta=1e-4):
    N = len(x_train)
    params = init_weights()
    posterior_samples = []

    for k in range(n_steps):
        idx = np.random.choice(N, batch_size, replace=False)
        grad = grad_log_posterior(params, x_train[idx], y_train[idx], N)
        noise = np.sqrt(eta) * np.random.randn(len(params))
        params = params + 0.5 * eta * grad + noise

        # Collect samples after burn-in
        if k > 10000 and k % 50 == 0:
            posterior_samples.append(params.copy())

    return np.array(posterior_samples)

samples = sgld_bnn(x_train, y_train)
x_test = np.linspace(-3, 7, 200)
predictions = np.array([forward(s, x_test) for s in samples])

mean_pred = predictions.mean(axis=0)
std_pred = predictions.std(axis=0)
```sql

The result demonstrates the core promise of Bayesian inference: **calibrated uncertainty**. In the gap region $x \in [1, 3]$, the posterior predictive standard deviation is 3-5x larger than in the data-rich regions. A point-estimate network (trained with SGD) would output a confident but arbitrary interpolation through the gap, with no indication that its prediction is unreliable.

This is not a toy property --- it is the foundation of active learning (query where uncertainty is high), safe reinforcement learning (avoid states with high epistemic uncertainty), and model selection (prefer models with tighter predictive bands on held-out data).

## Numerical Implementation: SDE Simulation You Can Actually Run

The continuous Langevin SDE $dX = -\nabla U(X)\,dt + \sqrt{2}\,dW$ becomes the discrete update
$$ X_{k+1} = X_k - \eta\,\nabla U(X_k) + \sqrt{2\eta}\,\xi_k,\quad \xi_k \sim \mathcal{N}(0, I). $$
This is **Euler-Maruyama**, and it is the entire algorithm. Python:

```python
import numpy as np
def langevin(grad_U, x0, eta=1e-3, n_steps=10000):
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    for _ in range(n_steps):
        x = x - eta*grad_U(x) + np.sqrt(2*eta)*np.random.randn(*x.shape)
        samples.append(x.copy())
    return np.array(samples)
```sql

Three things bite you in practice:

1. **Step-size bias.** EM samples a slightly different stationary distribution than the SDE. The bias is $O(\eta)$. Either (a) take $\eta \to 0$ and pay in mixing time, or (b) wrap with Metropolis-Hastings accept/reject — that gives MALA (Metropolis-Adjusted Langevin), unbiased at the cost of one extra log-density eval per step.
2. **Heavy tails kill you.** If $U$ grows slower than quadratically, EM blows up at the tail. Use higher-order schemes (Milstein) or clip the gradient. For neural-network log-densities this is mandatory.
3. **Multimodal targets.** Vanilla Langevin gets stuck in a basin. Replica exchange (parallel tempering) runs $K$ chains at temperatures $T_1 < \dots < T_K$ and swaps configurations. Cost is $K\times$ but mixing improves order-of-magnitude on bimodal posteriors.

Anywhere you read "we sample with Langevin", one of these three caveats applies. The papers usually skip them.

## SVGD in Practice: Where the Theory Hides Three Bugs

The gradient flow
$$ \dot x_i = \frac{1}{n}\sum_j \bigl[k(x_j, x_i)\nabla\log p(x_j) + \nabla_{x_j}k(x_j, x_i)\bigr] $$
is elegant. Implementing it correctly is not.

**Bug 1: bandwidth choice.** The RBF kernel $k(x, y) = \exp(-\|x-y\|^2/h)$ collapses if $h$ is wrong. The standard heuristic is the median trick: $h = \text{med}(\{\|x_i-x_j\|^2\})/\log n$. Without it, in $d > 20$ dimensions the kernel is effectively zero for all pairs and the repulsive force vanishes. Particles collapse to the mode.

**Bug 2: gradient evaluation.** $\nabla_{x_j}k(x_j, x_i) = -\frac{2}{h}(x_j - x_i)k(x_j, x_i)$. The minus sign matters; flipping it reverses the repulsion and you get clustering instead of coverage. Easy to mis-derive at midnight.

**Bug 3: $n$ too small in high $d$.** SVGD needs $n \gtrsim d$ particles to span the space. With $n = 50$ on $d = 200$ Bayesian NNs (the original paper's setup), the recovered posterior has rank-50 covariance, far from true. Recent work (Liu & Zhu, 2018; Chen & Ghattas, 2020) addresses this with random projection or matrix-valued kernels.

If you only remember one thing: **measure the mean pairwise kernel value periodically**. If it falls below $0.01$ you have lost the repulsive interaction.

## Score-Based Diffusion: Same Fokker-Planck, Reversed

Diffusion models train a network to approximate $\nabla \log p_t(x)$ at every noise level $t$. Sampling then runs a *reverse-time* SDE:
$$ dX = \bigl[-\nabla U(X) - 2\nabla\log p_t(X)\bigr]\,dt + \sqrt{2}\,d\bar W. $$
The whole pipeline is a Fokker-Planck story:

- **Forward**: pure noising. Density evolves from data $p_0$ to Gaussian $p_T$. Standard FP equation, $\sigma$ chosen so that $T$ is large enough.
- **Score matching**: train $s_\theta(x, t) \approx \nabla\log p_t(x)$ using denoising score matching (Vincent, 2011). The clean trick is $\nabla_x \log p_t(x) = \mathbb{E}[\nabla_x \log q(x|x_0)\,|\,x]$ for the conditional Gaussian $q(x|x_0)$.
- **Reverse**: run the time-reversed SDE (Anderson, 1982) using the learned score. Each step is one Langevin update with a learned drift correction.

The thing nobody states explicitly: **diffusion models are SVGD with the kernel replaced by a learned score field**. The repulsion-vs-attraction balance that SVGD does manually, diffusion learns from data. This is why both fall under the "sampling as gradient flow on densities" umbrella, and why Wasserstein geometry ([Section 4](#kl-divergence-is-a-wasserstein-gradient-flow)) is the right language for both.

The PDE-ML chapter 7 unpacks this in its own right; here I just wanted to land the Fokker-Planck connection.

## Summary

- Any It&ocirc; SDE has a Fokker-Planck PDE describing how its density evolves.
- Langevin dynamics samples from $p^\star \propto e^{-V}$; the discrete ULA / MALA / HMC algorithms are practical realisations.
- $\mathrm{KL}(\cdot \,\|\, p^\star)$ is a Wasserstein gradient-flow energy; its flow PDE *is* the Langevin FP equation. VI and MCMC are equivalent in continuous time.
- SVGD is a kernel-smoothed, deterministic particle approximation of the same flow, and avoids the random-walk inefficiency of MCMC.
- Convergence is exponential at rate $2\lambda$ where $\lambda$ is the log-Sobolev constant of $p^\star$; mixing through high barriers is the bottleneck in practice.
- Posterior sampling for Bayesian neural networks reduces to running Langevin (or SVGD) on the loss landscape.

**Series conclusion.** Across four articles we have used PDEs to unify scientific computing and machine learning — from solving PDEs with neural networks (PINNs), to learning solution operators (FNO/DeepONet), to training as gradient flows, to probabilistic inference as Fokker-Planck dynamics. The recurring theme: **discrete algorithms in machine learning are usually best understood as the time-discretisation of a continuous PDE**, and PDE theory is the language for proving convergence.

## References

- Q. Liu and D. Wang. "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm." *NeurIPS*, 2016. [arXiv:1608.04471](https://arxiv.org/abs/1608.04471)
- M. Welling and Y. W. Teh. "Bayesian Learning via Stochastic Gradient Langevin Dynamics." *ICML*, 2011.
- R. Jordan, D. Kinderlehrer, and F. Otto. "The variational formulation of the Fokker-Planck equation." *SIAM J. Math. Anal.*, 1998.
- D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. "Variational inference: A review for statisticians." *JASA*, 2017.
- L. Ambrosio, N. Gigli, and G. Savar&eacute;. *Gradient Flows in Metric Spaces and in the Space of Probability Measures.* Birkh&auml;user, 2008.
- A. Vempala and A. Wibisono. "Rapid convergence of the Unadjusted Langevin Algorithm: isoperimetry suffices." *NeurIPS*, 2019.
