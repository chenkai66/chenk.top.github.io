---
title: "PDE and Machine Learning (4): Variational Inference and the Fokker-Planck Equation"
date: 2024-10-16 09:00:00
tags:
  - PDE
  - Variational Inference
  - Fokker-Planck
  - ELBO
  - Langevin Dynamics
categories: Scientific Computing
series: PDE and Machine Learning
lang: en
mathjax: true
description: "Variational inference and Langevin MCMC are two faces of the same Fokker-Planck PDE. We derive the equivalence, build SVGD as an interacting-particle approximation, and quantify convergence under log-Sobolev inequalities."
disableNunjucks: true
---
> **Series**: PDE and Machine Learning -- Part 4 of 4
> [<-- Previous: Variational Principles](/en/PDE-and-Machine-Learning-3-Variational-Principles/)

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
- Wasserstein gradient flows from Part 3.
- Light stochastic calculus intuition (Brownian motion, It&ocirc; integral).
- Python / PyTorch for the experiments.

---

## 1. The Inference Problem

Bayesian inference asks for the posterior

$$
p(\theta \mid x) \;=\; \frac{p(x \mid \theta)\, p(\theta)}{\int p(x \mid \theta')\, p(\theta')\, d\theta'},
$$

but the marginal likelihood in the denominator is intractable for any non-trivial model. Two large families of approximation algorithms address this:

- **Variational inference (VI)**: pick a tractable family $\{q_\phi\}$ and minimise

  $$\mathrm{KL}\bigl(q_\phi \,\|\, p(\cdot\mid x)\bigr) \;=\; \mathbb{E}_{q_\phi}\!\left[\log \tfrac{q_\phi(\theta)}{p(\theta\mid x)}\right],$$

  equivalently maximising the **Evidence Lower Bound** $\mathrm{ELBO}(\phi) = \mathbb{E}_{q_\phi}[\log p(x\mid\theta)] - \mathrm{KL}(q_\phi \| p(\theta))$.

- **Markov Chain Monte Carlo (MCMC)**: build a Markov chain whose stationary distribution is exactly $p(\cdot\mid x)$. **Langevin dynamics** is the canonical gradient-based instance.

These look like very different objects: VI is a finite-dimensional optimisation over $\phi$, while MCMC is an infinite-time stochastic process. The PDE viewpoint reveals they are the **same evolution of probability measures**, only sampled differently.

## 2. From SDE to Fokker-Planck

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

## 3. Langevin Dynamics: Sampling as a PDE

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
```

ULA's bias is $O(\eta)$; **MALA** (Metropolis-Adjusted Langevin) restores exactness via an accept-reject step. **HMC** (Hamiltonian Monte Carlo) is the natural underdamped analogue with momentum.

![Langevin SDE trajectories and the empirical density they generate.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig2_langevin_sde_to_density.png)
*Figure 2. Left: 25 representative particles bouncing inside the double well; many never cross the barrier in finite time. Right: the histogram of 400 particles converges to the Gibbs target as $t$ grows -- the discrete sampler is realising the FP equation in figure 1.*

## 4. KL Divergence is a Wasserstein Gradient Flow

Decompose the KL divergence relative to $p^\star \propto e^{-V}$:

$$\mathcal{F}[p] \;=\; \mathrm{KL}(p\,\|\,p^\star) \;=\; \underbrace{\int p\log p\,dx}_{\text{neg-entropy }\mathcal{H}[p]} \;+\; \underbrace{\int p\, V\,dx}_{\text{potential energy}} \;+\; \text{const}.$$

This is the **free energy functional** from Part 3. The Jordan-Kinderlehrer-Otto (JKO) theorem (1998) tells us its **Wasserstein-2 gradient flow** is

$$\partial_t p \;=\; \nabla\!\cdot\!\bigl(p \nabla V\bigr) + \Delta p,$$

which is exactly the FP equation for Langevin with $\tau = 1$. Hence:

> **Equivalence**. Minimising $\mathrm{KL}(\cdot \| p^\star)$ in Wasserstein space and running Langevin dynamics targeting $p^\star$ are the **same PDE**. VI and Langevin MCMC are two algorithmic discretisations of one continuous-time gradient flow.

| Aspect | Variational Inference | Langevin MCMC |
|---|---|---|
| Objective | Minimise $\mathrm{KL}(q_\phi \| p^\star)$ | Sample from $p^\star$ |
| State | Parameters $\phi$ | Particles $\{X^{(i)}\}$ |
| Step | Gradient step on ELBO | Euler-Maruyama on SDE |
| Continuous limit | Wasserstein gradient flow of KL | Fokker-Planck equation |
| Stationary | $q^\star = p^\star$ (if expressive) | $p_\infty = p^\star$ |
| Bias | Restricted family + Adam noise | Discretisation $O(\eta)$ |

![KL divergence as a Wasserstein gradient flow.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig3_kl_gradient_flow.png)
*Figure 3. Two initial densities (concentrated and broad) are evolved by the FP equation toward the bimodal target. The right panel shows their KL divergence to $p^\star$ decaying monotonically -- the gradient-flow guarantee in action.*

## 5. VI vs MCMC in Practice

VI and MCMC may be equivalent in the continuous limit, but their finite-time behaviour differs dramatically.

- **VI minimising $\mathrm{KL}(q\|p^\star)$ is mode-seeking**: when $q$ is restricted to a simple family, the variational optimum collapses onto a single mode and underestimates uncertainty.
- **MCMC is mass-covering**: a long enough chain visits every mode in proportion to its mass, but mixing across barriers can be exponentially slow.

![VI vs MCMC.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig4_vi_vs_mcmc.png)
*Figure 4. Left: the best mean-field Gaussian (in reverse KL) under-fits one mode of a bimodal posterior. Right: 4000 Langevin samples cover both modes correctly -- but only because the barrier in this 1D example is small.*

## 6. Stein Variational Gradient Descent

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
```

In the infinite-particle limit SVGD obeys

$$\partial_t p \;=\; -\nabla\!\cdot\!\bigl(p\, v[p]\bigr),\qquad v[p](x) = \mathbb{E}_{y \sim p}\!\bigl[k(y,x)\nabla\log p^\star(y) + \nabla_y k(y,x)\bigr],$$

and as the bandwidth $h \to 0$ this PDE collapses to the standard Fokker-Planck equation. SVGD is therefore a **kernel-smoothed FP solver**.

![SVGD particles on a bimodal target.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig5_svgd_particles.png)
*Figure 5. Left: snapshots of 80 SVGD particles starting at the origin and splitting to populate both modes within a few hundred iterations. Right: full particle trajectories show the kernel repulsion preventing collapse, while the drift term locks particles around $\pm 2$.*

## 7. Convergence Theory

**Definition (LSI).** $p^\star$ satisfies a **log-Sobolev inequality** with constant $\lambda > 0$ if for all smooth probability densities $p \ll p^\star$:

$$\mathrm{KL}(p \,\|\, p^\star) \;\leq\; \frac{1}{2\lambda}\, I(p \,\|\, p^\star),\qquad I(p\|p^\star) = \int p\, \bigl\|\nabla \log \tfrac{p}{p^\star}\bigr\|^2 dx.$$

The right-hand side is the **Fisher information** -- which is also the dissipation rate of the FP equation:

$$\frac{d}{dt}\,\mathrm{KL}(p_t\,\|\,p^\star) \;=\; -\, I(p_t\,\|\,p^\star).$$

Combining, $\frac{d}{dt}\mathrm{KL}(p_t \| p^\star) \leq -2\lambda\, \mathrm{KL}(p_t \| p^\star)$, hence Gr&ouml;nwall:

$$
\boxed{\;\mathrm{KL}(p_t\,\|\,p^\star) \;\leq\; e^{-2\lambda t}\, \mathrm{KL}(p_0\,\|\,p^\star).\;}
$$

Strongly log-concave targets ($\nabla^2 V \succeq mI$) automatically satisfy LSI with $\lambda \geq m$ (Bakry-&Eacute;mery). Multimodal targets typically have tiny $\lambda$, predicting the exponentially slow mixing observed empirically.

![Convergence rates.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig6_convergence_analysis.png)
*Figure 6. Left: theoretical KL decay $e^{-2\lambda t}$ for three log-Sobolev constants. Right: empirical KL trajectories for VI, Langevin MCMC and SVGD on a smooth Gaussian target -- all three converge, but with different rates and noise profiles.*

## 8. Application: Bayesian Neural Networks

A Bayesian neural network places a prior $p(w)$ on the weights and seeks the posterior $p(w \mid \mathcal{D}) \propto p(\mathcal{D} \mid w)\, p(w)$. Even for tiny architectures the posterior is intractable, but Langevin dynamics on $w$ requires only

$$\nabla_w \log p(w \mid \mathcal{D}) \;=\; \nabla_w \log p(\mathcal{D} \mid w) + \nabla_w \log p(w),$$

i.e. exactly the gradient computed during normal back-propagation, plus a Gaussian prior term. **Stochastic Gradient Langevin Dynamics** (Welling and Teh, 2011) replaces the full-data gradient with a mini-batch estimate, making this practical at modern scale.

The figure below uses 24 random Fourier features as a tractable "Bayesian NN" so that the posterior over weights is well defined, and samples it with full-batch Langevin.

![Bayesian neural net posterior bands.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/04-Variational-Inference/fig7_bayesian_nn.png)
*Figure 7. Left: posterior predictive for a regression model with a gap in the training data; the 90% Langevin band widens precisely where data is missing. Right: predictive standard deviation peaks in the data gap -- exactly the **epistemic uncertainty** that point-estimate networks lack.*

## 9. Summary

- Any It&ocirc; SDE has a Fokker-Planck PDE describing how its density evolves.
- Langevin dynamics samples from $p^\star \propto e^{-V}$; the discrete ULA / MALA / HMC algorithms are practical realisations.
- $\mathrm{KL}(\cdot \,\|\, p^\star)$ is a Wasserstein gradient-flow energy; its flow PDE *is* the Langevin FP equation. VI and MCMC are equivalent in continuous time.
- SVGD is a kernel-smoothed, deterministic particle approximation of the same flow, and avoids the random-walk inefficiency of MCMC.
- Convergence is exponential at rate $2\lambda$ where $\lambda$ is the log-Sobolev constant of $p^\star$; mixing through high barriers is the bottleneck in practice.
- Posterior sampling for Bayesian neural networks reduces to running Langevin (or SVGD) on the loss landscape.

**Series conclusion.** Across four articles we have used PDEs to unify scientific computing and machine learning -- from solving PDEs with neural networks (PINNs), to learning solution operators (FNO/DeepONet), to training as gradient flows, to probabilistic inference as Fokker-Planck dynamics. The recurring theme: **discrete algorithms in machine learning are usually best understood as the time-discretisation of a continuous PDE**, and PDE theory is the language for proving convergence.

## References

- Q. Liu and D. Wang. "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm." *NeurIPS*, 2016. [arXiv:1608.04471](https://arxiv.org/abs/1608.04471)
- M. Welling and Y. W. Teh. "Bayesian Learning via Stochastic Gradient Langevin Dynamics." *ICML*, 2011.
- R. Jordan, D. Kinderlehrer, and F. Otto. "The variational formulation of the Fokker-Planck equation." *SIAM J. Math. Anal.*, 1998.
- D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. "Variational inference: A review for statisticians." *JASA*, 2017.
- L. Ambrosio, N. Gigli, and G. Savar&eacute;. *Gradient Flows in Metric Spaces and in the Space of Probability Measures.* Birkh&auml;user, 2008.
- A. Vempala and A. Wibisono. "Rapid convergence of the Unadjusted Langevin Algorithm: isoperimetry suffices." *NeurIPS*, 2019.
