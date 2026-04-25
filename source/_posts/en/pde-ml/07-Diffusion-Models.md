---
title: "PDE and Machine Learning (7): Diffusion Models and Score Matching"
date: 2024-07-30 09:00:00
tags:
  - PDE
  - Machine Learning
  - Diffusion Models
  - Score Matching
  - DDPM
  - DDIM
  - SDE
  - Generative Models
categories: PDE and Machine Learning
series:
  name: "PDE and Machine Learning"
  part: 7
  total: 8
lang: en
mathjax: true
description: "Diffusion models are PDE solvers in disguise. We derive the heat equation, Fokker-Planck, score matching, DDPM, and DDIM from a unified PDE perspective and visualise every step."
disableNunjucks: true
series_order: 7
---

## What This Article Covers

Since 2020, **diffusion models** have become the dominant paradigm in generative AI. From DALL·E 2 to Stable Diffusion to Sora, their generation quality and training stability are unmatched by GANs and VAEs. Beneath this success lies a remarkably clean mathematical structure: **diffusion models are numerical solvers for partial differential equations**.

- Adding Gaussian noise corresponds to integrating the **Fokker–Planck equation** forward in time.
- Learning to denoise is equivalent to learning the **score function** $\nabla\log p_t$.
- DDPM is a discretised **reverse SDE**; DDIM is the corresponding **probability-flow ODE**.
- Stable Diffusion is the same machinery, executed in a low-dimensional latent space.

**What you will learn**

1. The heat equation and Gaussian kernels — the mathematics of diffusion.
2. SDEs and the Fokker–Planck equation — how probability density evolves.
3. Score functions, score matching (DSM/SSM), and Langevin dynamics.
4. DDPM as a discretised reverse SDE; DDIM as a probability-flow ODE.
5. Latent diffusion (Stable Diffusion) and connections to scientific computing.

**Prerequisites:** multivariable calculus, basic probability (Gaussian, Bayes), neural network fundamentals.

---

## 1. Heat Equation and Diffusion Processes

### 1.1 Fick's Law and the Diffusion Equation

Heat flow, ink diffusing in water, particles diffusing under a concentration gradient — they all obey the same equation. **Fick's first law** says the flux is proportional to (minus) the concentration gradient,
$$
\mathbf{J} = -D\,\nabla u,
$$
where $D > 0$ is the diffusion coefficient. Combined with mass conservation $\partial_t u + \nabla\!\cdot\!\mathbf{J} = 0$ this gives the **heat equation** (a.k.a. diffusion equation):
$$
\frac{\partial u}{\partial t} = D\,\nabla^2 u. \tag{1}
$$
The Laplacian measures local "curvature" of $u$: where $u$ is concave (a hot spot), $\nabla^2 u < 0$ and $u$ decreases; where $u$ is convex (a cold spot), $u$ increases. The end state is uniform.

### 1.2 Gaussian Kernels: Fundamental Solutions

For the point-source initial condition $u(\mathbf{x},0) = \delta(\mathbf{x})$, the solution to (1) is the **heat kernel**
$$
G(\mathbf{x}, t) = \frac{1}{(4\pi D t)^{d/2}}\exp\!\left(-\frac{\|\mathbf{x}\|^2}{4Dt}\right). \tag{2}
$$
This is a Gaussian with variance $\sigma_t^2 = 2Dt$ growing linearly in time. For a general initial profile $u_0$, the solution is just a convolution with this kernel:
$$
u(\mathbf{x}, t) = (G_t * u_0)(\mathbf{x}).
$$
Diffusion = "blur with a growing Gaussian". Conceptually, that is exactly what the forward noising in a diffusion model does.

### 1.3 Fourier Perspective: Diffusion as a Low-Pass Filter

In Fourier space, $\widehat{\nabla^2 u}(\mathbf{k}) = -\|\mathbf{k}\|^2\,\hat u(\mathbf{k})$ turns (1) into an ODE for each mode:
$$
\hat u(\mathbf{k}, t) = \hat u_0(\mathbf{k})\,e^{-D\|\mathbf{k}\|^2 t}.
$$
High-frequency content (large $\|\mathbf{k}\|$) decays exponentially faster than low-frequency content. **Diffusion is a low-pass filter** — fine structure dies first, the coarse structure last. The reverse, denoising, must therefore reconstruct the high-frequency content; this is precisely what the score network does.

![Forward diffusion turns structured data into isotropic Gaussian noise; the bottom row shows the marginal density $p_t$ converging to $\mathcal{N}(0, I)$ as predicted by the Fokker–Planck equation.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig1_forward_diffusion.png)
*Forward diffusion turns structured data into isotropic Gaussian noise; the bottom row shows the marginal density $p_t$ converging to $\mathcal{N}(0, I)$ as predicted by the Fokker–Planck equation.*

---

## 2. SDEs and the Fokker–Planck Equation

The heat equation describes a **deterministic** evolution of densities. If we want to think of individual sample paths — which is what diffusion models actually generate — we need stochastic differential equations.

### 2.1 Brownian Motion and Itô SDEs

**Brownian motion** $\mathbf{B}_t$ satisfies $\mathbf{B}_0 = 0$, has independent Gaussian increments $\mathbf{B}_{t+\Delta t} - \mathbf{B}_t \sim \mathcal{N}(\mathbf{0}, \Delta t\,\mathbf{I})$, and continuous but nowhere-differentiable paths. A general Itô SDE has the form
$$
d\mathbf{X}_t = f(\mathbf{X}_t, t)\,dt + g(t)\,d\mathbf{B}_t, \tag{3}
$$
with **drift** $f$ (the deterministic pull) and **diffusion coefficient** $g$ (the noise amplitude).

The two schedules dominating the diffusion-model literature are:

| Schedule | Drift $f(\mathbf{x}, t)$ | Diffusion $g(t)$ | Stationary law |
|----------|--------------------------|-------------------|----------------|
| Variance-Preserving (VP) | $-\tfrac{1}{2}\beta(t)\,\mathbf{x}$ | $\sqrt{\beta(t)}$ | $\mathcal{N}(\mathbf{0},\,\mathbf{I})$ |
| Variance-Exploding (VE) | $0$ | $\sqrt{d\sigma^2/dt}$ | variance grows without bound |

DDPM is a discretisation of VP; the original NCSN of Song & Ermon (2019) is a discretisation of VE.

### 2.2 The Fokker–Planck Equation

If $\mathbf{X}_t$ obeys (3) and has density $p(\mathbf{x}, t)$, then $p$ satisfies the **Fokker–Planck equation** (Kolmogorov forward equation):
$$
\boxed{\;\frac{\partial p}{\partial t} \;=\; -\nabla\!\cdot\!\bigl(f\,p\bigr) \;+\; \tfrac{1}{2}\,g^2\,\nabla^2 p\;.\;} \tag{4}
$$

**Sketch of proof.** For any smooth test function $\varphi$, Itô's formula gives
$$
d\varphi(\mathbf{X}_t) = \bigl(f\!\cdot\!\nabla\varphi + \tfrac{1}{2}g^2\nabla^2\varphi\bigr)\,dt + g\,\nabla\varphi\!\cdot\!d\mathbf{B}_t.
$$
Taking expectations kills the martingale term, and writing $\mathbb{E}[\varphi(\mathbf{X}_t)] = \int \varphi\,p\,d\mathbf{x}$ then integrating by parts (using that $\varphi$ is arbitrary) yields (4). $\blacksquare$

**Sanity check.** Setting $f \equiv 0$ and $g^2/2 = D$ in (4) recovers the heat equation $\partial_t p = D\,\nabla^2 p$. The Fokker–Planck equation is exactly the heat equation plus a drift term.

### 2.3 The Kolmogorov Backward Equation

For a terminal payoff $g(\mathbf{X}_T)$, the conditional expectation $u(s, \mathbf{x}) = \mathbb{E}[g(\mathbf{X}_T)\,|\,\mathbf{X}_s = \mathbf{x}]$ satisfies the **backward** equation
$$
\partial_s u + f\!\cdot\!\nabla u + \tfrac{1}{2}g^2 \nabla^2 u = 0,
$$
with terminal condition $u(T, \mathbf{x}) = g(\mathbf{x})$. The forward equation evolves densities forward in time; the backward equation evolves expectations backward. Together they are the Feynman–Kac correspondence — and the time-reversed forward SDE is exactly what we will use to *generate* samples.

---

## 3. Score-Based Generative Models

### 3.1 The Score Function

The **score** of a density $p$ is
$$
\mathbf{s}(\mathbf{x}) \;:=\; \nabla_{\mathbf{x}}\,\log p(\mathbf{x}). \tag{5}
$$
Three useful properties:

- **Normalisation-free.** $\nabla \log(p / Z) = \nabla \log p$, so we never need the partition function.
- **Closed form for Gaussians.** If $p = \mathcal{N}(\boldsymbol\mu, \sigma^2 \mathbf{I})$, then $\mathbf{s}(\mathbf{x}) = -(\mathbf{x} - \boldsymbol\mu) / \sigma^2$.
- **Zero mean.** $\mathbb{E}_p[\mathbf{s}(\mathbf{x})] = \mathbf{0}$ (under mild regularity).

Geometrically, the score is a vector field that always points *toward* high-probability regions and grows large in low-density valleys.

![Score of a 2-mode Gaussian mixture: the field points uphill, away from low-density regions and toward the modes.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig3_score_field.png)
*Score of a 2-mode Gaussian mixture: the field points uphill, away from low-density regions and toward the modes.*

### 3.2 Score Matching

Because $p$ is unknown we cannot directly minimise $\mathbb{E}_p\,\|\mathbf{s}_\theta - \nabla\log p\|^2$. There are three workable surrogates.

**Implicit Score Matching (ISM, Hyvärinen 2005).** Integration by parts removes the unknown score:
$$
\mathcal{L}_{\text{ISM}}(\theta) = \mathbb{E}_p\Bigl[\,\tfrac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \mathrm{tr}\bigl(\nabla\mathbf{s}_\theta(\mathbf{x})\bigr)\Bigr].
$$
The trace of the Jacobian is expensive in high dimensions.

**Sliced Score Matching (SSM, Song et al. 2019).** Project onto random directions $\mathbf{v}$:
$$
\mathcal{L}_{\text{SSM}}(\theta) = \mathbb{E}_{\mathbf{v}, \mathbf{x}}\Bigl[\,\mathbf{v}^\top \nabla\mathbf{s}_\theta(\mathbf{x})\,\mathbf{v} + \tfrac{1}{2}(\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x}))^2\Bigr].
$$
One Hessian–vector product per sample suffices.

**Denoising Score Matching (DSM, Vincent 2011) — the workhorse.** Add noise $\tilde{\mathbf{x}} = \mathbf{x} + \sigma\,\boldsymbol\eta$, $\boldsymbol\eta \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, and learn to predict the noise direction:
$$
\boxed{\;\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}\Bigl[\bigl\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) + \tfrac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2}\bigr\|^2\Bigr].\;} \tag{6}
$$
Vincent showed (6) has the same minimiser as matching the *true* score of the noisy distribution $p_\sigma = p * \mathcal{N}(0, \sigma^2 \mathbf{I})$. As $\sigma \to 0$, $p_\sigma \to p$. In practice we anneal $\sigma$ during training.

![Left: DSM loss decreases monotonically and plateaus. Right: the learned score matches the true $\nabla\log p$ in high-density regions; near low-density valleys (centre) it is intentionally smoothed by the noise level $\sigma$.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig5_score_matching_loss.png)
*Left: DSM loss decreases monotonically and plateaus. Right: the learned score matches the true $\nabla\log p$ in high-density regions; near low-density valleys (centre) it is intentionally smoothed by the noise level $\sigma$.*

### 3.3 Langevin Dynamics

Once we have $\mathbf{s}_\theta$ we can sample with Langevin MCMC:
$$
\mathbf{x}_{k+1} = \mathbf{x}_k + \tfrac{\epsilon}{2}\,\mathbf{s}_\theta(\mathbf{x}_k) + \sqrt{\epsilon}\,\boldsymbol\eta_k,\qquad \boldsymbol\eta_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}). \tag{7}
$$
As $\epsilon \to 0$ and $k \to \infty$ the chain converges to $p$. The deterministic term is *exploitation* (climb the score), the noise term is *exploration* (escape local maxima).

### 3.4 Anderson's Reverse-Time SDE

Here is the keystone result that turns score matching into a generative model. **Anderson (1982)** showed that the time-reversal of (3) is itself an SDE:
$$
\boxed{\;d\mathbf{X}_t = \bigl[\,f(\mathbf{X}_t, t) - g(t)^2\,\nabla\log p_t(\mathbf{X}_t)\,\bigr]\,dt + g(t)\,d\bar{\mathbf{B}}_t,\;} \tag{8}
$$
where $\bar{\mathbf{B}}_t$ is a Brownian motion in reverse time and $p_t$ is the marginal of (3) at time $t$. **The only thing that depends on the data distribution is $\nabla\log p_t$ — exactly what the score network learns**. Run (8) backward from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and you generate a sample from (approximately) the data distribution.

![Reverse diffusion runs from $t = T$ down to $t = 0$; the score network supplies the drift correction needed to reverse the noising process.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig2_reverse_diffusion.png)
*Reverse diffusion runs from $t = T$ down to $t = 0$; the score network supplies the drift correction needed to reverse the noising process.*

---

## 4. From Continuous Theory to DDPM and DDIM

### 4.1 DDPM: Forward Process in Closed Form

Pick a noise schedule $\{\beta_t\}_{t=1}^T$. Define $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. The DDPM forward process is the discrete-time Markov chain
$$
q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\bigl(\mathbf{x}_t;\,\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\,\beta_t\mathbf{I}\bigr),
$$
which has the convenient closed form
$$
\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol\epsilon,\qquad \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}). \tag{9}
$$
This is exactly the Euler–Maruyama discretisation of the VP-SDE, so $\bar\alpha_T \to 0$ and $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$.

### 4.2 The DDPM Loss = Weighted DSM

Train a network $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)$ to predict the noise that was added:
$$
\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t,\,\mathbf{x}_0,\,\boldsymbol\epsilon}\Bigl[\,\bigl\|\boldsymbol\epsilon_\theta(\mathbf{x}_t, t) - \boldsymbol\epsilon\bigr\|^2\Bigr]. \tag{10}
$$
Why is this score matching in disguise? Because (9) implies
$$
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\boldsymbol\epsilon}{\sqrt{1 - \bar\alpha_t}},
$$
so the network is learning a scaled score: $\mathbf{s}_\theta(\mathbf{x}_t, t) = -\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)/\sqrt{1 - \bar\alpha_t}$. (10) is precisely (6) with weights $w(t) = 1$.

### 4.3 DDIM: The Probability-Flow ODE

A beautiful fact about (3): there is a **deterministic** ODE with the same one-time marginals at every $t$,
$$
\boxed{\;\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \tfrac{1}{2}\,g(t)^2\,\nabla\log p_t(\mathbf{x}).\;} \tag{11}
$$
This is the **probability-flow ODE**. Marginals match because both (8) and (11) yield the same Fokker–Planck equation (4) for $p_t$. Solving (11) backward in time with a high-order ODE solver — Heun, RK4, DPM-Solver — is the basis of DDIM and its descendants. The result is *deterministic* (same noise → same image), supports much larger step sizes (25–50 vs 1000), and is exactly invertible (one can encode an image back to its latent noise).

![DDPM (left) injects fresh noise at each reverse step; DDIM (right) follows a deterministic flow under the same learned score, reaching the modes in far fewer steps.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig4_ddpm_vs_ddim.png)
*DDPM (left) injects fresh noise at each reverse step; DDIM (right) follows a deterministic flow under the same learned score, reaching the modes in far fewer steps.*

### 4.4 A Unified View

| Method | Process | Typical steps | Deterministic? | Strength |
|--------|---------|---------------|----------------|----------|
| DDPM | Reverse SDE (8) | ~1000 | No | Diversity, simple training |
| DDIM | Probability-flow ODE (11) | ~25–50 | Yes | Speed, exact inversion |
| DPM-Solver | Higher-order ODE | ~10–20 | Yes | Even faster, same fidelity |
| EDM (Karras et al.) | Continuous, refined preconditioning | ~30 | Tunable | SOTA quality |

### 4.5 The PDE → Diffusion Model Map

Putting it all together:

![Heat equation $\to$ Fokker–Planck $\to$ forward SDE; Anderson's time-reversal needs $\nabla\log p_t$, which the score network learns by DSM; the same score then drives DDPM (SDE) or DDIM (ODE).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig6_pde_diffusion_bridge.png)
*Heat equation $\to$ Fokker–Planck $\to$ forward SDE; Anderson's time-reversal needs $\nabla\log p_t$, which the score network learns by DSM; the same score then drives DDPM (SDE) or DDIM (ODE).*

---

## 5. Latent Diffusion: Stable Diffusion in One Picture

Pixel-space diffusion on $512 \times 512$ images is expensive: every U-Net forward pass operates on ~$8\times 10^5$ floats. **Latent Diffusion** (Rombach et al., 2022) trains a VAE-like autoencoder $(\mathcal{E}, \mathcal{D})$ first to map images to an $\sim\!8\times$ smaller latent $\mathbf{z}_0 = \mathcal{E}(\mathbf{x})$, and runs the entire diffusion process *in latent space*. Decoding $\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z}_0)$ is a single feed-forward pass.

Conditioning (text, class labels, depth maps, ControlNet poses, …) enters the U-Net through cross-attention.

![Stable Diffusion = autoencoder + diffusion in latent space + cross-attention conditioning.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig7_latent_diffusion.png)
*Stable Diffusion = autoencoder + diffusion in latent space + cross-attention conditioning.*

The compute saving is roughly $f^{2d}$ where $f$ is the spatial downsampling factor (typically 8) and $d=2$, i.e. ~$64\times$. It is the architectural trick that turned diffusion models into a consumer technology.

---

## 6. Connection to Scientific Computing

Score-based diffusion is not just a generative-modelling trick — it is a tool for sampling from arbitrary, possibly intractable, probability distributions. Two application directions are particularly relevant for the PDE community:

1. **Conditional generation under PDE constraints.** Train a score model $\mathbf{s}_\theta(\mathbf{x}, t \mid \mathcal{C})$ where $\mathcal{C}$ encodes a PDE residual or boundary data. Reverse sampling produces fields that respect the constraint (in distribution). This is the basis of *diffusion posterior sampling* for inverse problems (Chung et al., 2023): tomography, super-resolution, deblurring, etc.

2. **Surrogate samplers for Bayesian inference.** Many Bayesian inverse problems require sampling from a posterior $p(\theta \mid \mathbf{y}) \propto p(\mathbf{y}\mid\theta)\,p(\theta)$. If we can train a score model on $(\theta, \mathbf{y})$ pairs from simulation, conditional reverse sampling replaces MCMC.

The unifying message: **whenever you need to sample from a high-dimensional, multimodal distribution and you can afford to simulate forward dynamics, the diffusion-model recipe applies**.

---

## 7. Exercises

**Exercise 1.** Show the heat equation is the special case $f \equiv 0$, $g^2 / 2 = D$ of Fokker–Planck.

> *Solution.* Substitute into (4): $\partial_t p = -\nabla\!\cdot\!(\mathbf{0}\cdot p) + D\nabla^2 p = D\nabla^2 p$. $\blacksquare$

**Exercise 2.** Explain why the DDPM loss equals (weighted) score matching.

> *Solution.* From (9), $q(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,\mathbf{x}_0, (1-\bar\alpha_t)\mathbf{I})$, so $\nabla_{\mathbf{x}_t}\log q = -(\mathbf{x}_t - \sqrt{\bar\alpha_t}\,\mathbf{x}_0)/(1-\bar\alpha_t) = -\boldsymbol\epsilon/\sqrt{1-\bar\alpha_t}$. Predicting $\boldsymbol\epsilon$ is therefore predicting a scaled score, and (10) is (6) with $w(t)=1$. $\blacksquare$

**Exercise 3.** Why can DDIM use far fewer steps than DDPM?

> *Solution.* DDIM solves a smooth deterministic ODE (11) and admits high-order multi-step methods (Heun, RK4, DPM-Solver). DDPM solves an SDE whose strong-order convergence is $\mathcal{O}(\sqrt{\Delta t})$, so the step size must remain small for accuracy. $\blacksquare$

**Exercise 4.** Interpret diffusion as a low-pass filter and explain why this is consistent with "noise destroys high-frequency content first".

> *Solution.* Fourier modes evolve as $\hat p(\mathbf{k}, t) \propto \exp(-D\|\mathbf{k}\|^2 t)$, so high-$\|\mathbf{k}\|$ content decays exponentially faster than low-$\|\mathbf{k}\|$ content. Since high $\|\mathbf{k}\|$ corresponds to fine detail, the fine structure is destroyed first; the coarse structure persists longer. $\blacksquare$

**Exercise 5.** Derive Anderson's reverse-time SDE drift in 1D for the OU process $dX_t = -\tfrac12 \beta X_t\,dt + \sqrt\beta\,dB_t$ with stationary $\mathcal{N}(0,1)$.

> *Solution sketch.* In stationary regime $p_t(x) = \mathcal{N}(x;0,1)$, so $\nabla\log p_t(x) = -x$. Plug $f = -\tfrac12\beta x$, $g^2 = \beta$ into (8): drift $= -\tfrac12\beta x - \beta(-x) = \tfrac12\beta x$, i.e. the reverse process has a *positive* mean reversion away from zero — exactly what is needed to "uncrunch" the OU collapse to the origin. $\blacksquare$

---

## References

[1] Song, Y., et al. (2020). *Score-based generative modeling through stochastic differential equations.* [arXiv:2011.13456](https://arxiv.org/abs/2011.13456).

[2] Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising diffusion probabilistic models.* NeurIPS. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239).

[3] Song, J., Meng, C., & Ermon, S. (2021). *Denoising diffusion implicit models.* ICLR. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502).

[4] Song, Y., & Ermon, S. (2019). *Generative modeling by estimating gradients of the data distribution.* NeurIPS. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600).

[5] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). *Elucidating the design space of diffusion-based generative models (EDM).* NeurIPS. [arXiv:2206.00364](https://arxiv.org/abs/2206.00364).

[6] Lu, C., et al. (2022). *DPM-Solver: a fast ODE solver for diffusion models.* NeurIPS. [arXiv:2206.00927](https://arxiv.org/abs/2206.00927).

[7] Rombach, R., et al. (2022). *High-resolution image synthesis with latent diffusion models.* CVPR. [arXiv:2112.10752](https://arxiv.org/abs/2112.10752).

[8] Anderson, B. D. O. (1982). *Reverse-time diffusion equation models.* Stochastic Processes and their Applications, 12(3), 313–326.

[9] Hyvärinen, A. (2005). *Estimation of non-normalized statistical models by score matching.* JMLR.

[10] Vincent, P. (2011). *A connection between score matching and denoising autoencoders.* Neural Computation.

[11] Chung, H., et al. (2023). *Diffusion posterior sampling for general noisy inverse problems.* ICLR. [arXiv:2209.14687](https://arxiv.org/abs/2209.14687).

---

*This is Part 7 of the [PDE and Machine Learning](/categories/PDE-and-Machine-Learning/) series. Next: [Part 8 — Reaction-Diffusion Systems and GNN](/en/PDE-and-Machine-Learning-8-Reaction-Diffusion-Systems/). Previous: [Part 6 — Continuous Normalizing Flows](/en/PDE-and-Machine-Learning-6-Continuous-Normalizing-Flows/).*
