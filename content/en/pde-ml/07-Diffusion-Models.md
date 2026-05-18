---
title: "PDE and ML (7): Diffusion Models and Score Matching"
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
series: pde-ml
lang: en
mathjax: true
description: "Diffusion models are PDE solvers in disguise. We derive the heat equation, Fokker-Planck, score matching, DDPM, and DDIM from a unified PDE perspective and visualise every step."
disableNunjucks: true
series_order: 7
series_total: 8
translationKey: "pde-ml-7"
---
![PDE and ML (7): Diffusion Models and Score Matching — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/illustration_1.png)

---

The output side of a diffusion model is familiar: a high-quality image. The training objective, on the other hand, looks counter-intuitive at first sight — **add noise to the data until it is fully Gaussian, then learn to denoise step by step**. Why is this detour more effective than learning the data distribution directly?

The answer is hidden in PDEs. The forward noising process is a **heat equation** (or, more generally, a Fokker–Planck equation), and it admits a reverse-time version — provided we know the score (the gradient of the log-density) at every time. **Score matching** is the standard way to learn that score. From this angle, DDPM, DDIM, and score-based SDEs are not three different algorithms but three discretizations of the same PDE story.

This article walks that thread: start from the heat equation, derive the score-matching loss, recover the DDPM training objective inside this unified framework, and discuss how DDIM and Latent Diffusion accelerate inference without breaking the theory. With visualizations.

## What You Will Learn

Since 2020, **diffusion models** have become the dominant paradigm in generative AI. From DALL·E 2 to Stable Diffusion to Sora, their generation quality and training stability surpass those of GANs and VAEs. This success is underpinned by a remarkably clean mathematical structure: **diffusion models are numerical solvers for partial differential equations**.

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

## Heat Equation and Diffusion Processes

### Fick's Law and the Diffusion Equation

Heat flow, ink diffusing in water, particles diffusing under a concentration gradient — they all obey the same equation. **Fick's first law** says the flux is proportional to (minus) the concentration gradient,
$$\mathbf{J} = -D\,\nabla u,$$
where $D > 0$ is the diffusion coefficient. Combined with mass conservation $\partial_t u + \nabla\!\cdot\!\mathbf{J} = 0$ this gives the **heat equation** (a.k.a. diffusion equation):
$$\frac{\partial u}{\partial t} = D\,\nabla^2 u. \tag{1}$$
The Laplacian measures local "curvature" of $u$: where $u$ is concave (a hot spot), $\nabla^2 u < 0$ and $u$ decreases; where $u$ is convex (a cold spot), $u$ increases. The end state is uniform.

### Gaussian Kernels: Fundamental Solutions

For the point-source initial condition $u(\mathbf{x},0) = \delta(\mathbf{x})$, the solution to (1) is the **heat kernel**
$$G(\mathbf{x}, t) = \frac{1}{(4\pi D t)^{d/2}}\exp\!\left(-\frac{\|\mathbf{x}\|^2}{4Dt}\right). \tag{2}$$
This is a Gaussian with variance $\sigma_t^2 = 2Dt$ growing linearly in time. For a general initial profile $u_0$, the solution is just a convolution with this kernel:
$$u(\mathbf{x}, t) = (G_t * u_0)(\mathbf{x}).$$
Diffusion = "blur with a growing Gaussian". Conceptually, that is exactly what the forward noising in a diffusion model does.

### Fourier Perspective: Diffusion as a Low-Pass Filter

In Fourier space, $\widehat{\nabla^2 u}(\mathbf{k}) = -\|\mathbf{k}\|^2\,\hat u(\mathbf{k})$ turns (1) into an ODE for each mode:
$$\hat u(\mathbf{k}, t) = \hat u_0(\mathbf{k})\,e^{-D\|\mathbf{k}\|^2 t}.$$
High-frequency content (large $\|\mathbf{k}\|$) decays exponentially faster than low-frequency content. **Diffusion is a low-pass filter** — fine structure dies first, the coarse structure last. The reverse, denoising, must therefore reconstruct the high-frequency content; this is precisely what the score network does.

![Forward diffusion turns structured data into isotropic Gaussian noise; the bottom row shows the marginal density $p_t$ converging to $\mathcal{N}(0, I)$ as predicted by the Fokker–Planck equation.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig1_forward_diffusion.png)
*Forward diffusion turns structured data into isotropic Gaussian noise; the bottom row shows the marginal density $p_t$ converging to $\mathcal{N}(0, I)$ as predicted by the Fokker–Planck equation.*

---

**Implementation: watching diffusion destroy structure.** We can simulate the forward process directly and verify that $p_t$ converges to Gaussian:

```python
import numpy as np

def make_two_moons(n=2000, noise=0.05):
    # Generate two-moons dataset
    t = np.linspace(0, np.pi, n // 2)
    x1 = np.column_stack([np.cos(t), np.sin(t)]) + noise * np.random.randn(n//2, 2)
    x2 = np.column_stack([1 - np.cos(t), 1 - np.sin(t) - 0.5]) + noise * np.random.randn(n//2, 2)
    return np.vstack([x1, x2])

x0 = make_two_moons(5000)
T, beta_min, beta_max = 1000, 0.0001, 0.02

betas = np.linspace(beta_min, beta_max, T)
alphas = 1 - betas
alpha_bar = np.cumprod(alphas)

for t in [0, 100, 300, 500, 999]:
    eps = np.random.randn(*x0.shape)
    xt = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * eps
    print(f"t={t:4d}: mean={xt.mean(0).round(3)}, std={xt.std(0).round(3)}, "
          f"alpha_bar={alpha_bar[t]:.4f}")
```sql

At $t=0$, the data has clear two-moon structure (low variance, off-centre mean). By $t=999$, $\bar\alpha_T \approx 10^{-4}$, so $\mathbf{x}_T$ is nearly pure Gaussian noise — the low-pass filter has killed all structure.

## SDEs and the Fokker–Planck Equation

![Ornstein-Uhlenbeck SDE sample paths converging with histogram evolution toward Gaussian](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig8_sde_particle_trajectories.png)

The heat equation describes a **deterministic** evolution of densities. If we want to think of individual sample paths — which is what diffusion models actually generate — we need stochastic differential equations.

### Brownian Motion and Itô SDEs

**Brownian motion** $\mathbf{B}_t$ satisfies $\mathbf{B}_0 = 0$, has independent Gaussian increments $\mathbf{B}_{t+\Delta t} - \mathbf{B}_t \sim \mathcal{N}(\mathbf{0}, \Delta t\,\mathbf{I})$, and continuous but nowhere-differentiable paths. A general Itô SDE has the form
$$d\mathbf{X}_t = f(\mathbf{X}_t, t)\,dt + g(t)\,d\mathbf{B}_t, \tag{3}$$
with **drift** $f$ (the deterministic pull) and **diffusion coefficient** $g$ (the noise amplitude).

The two schedules dominating the diffusion-model literature are:

| Schedule | Drift $f(\mathbf{x}, t)$ | Diffusion $g(t)$ | Stationary law |
|----------|--------------------------|-------------------|----------------|
| Variance-Preserving (VP) | $-\tfrac{1}{2}\beta(t)\,\mathbf{x}$ | $\sqrt{\beta(t)}$ | $\mathcal{N}(\mathbf{0},\,\mathbf{I})$ |
| Variance-Exploding (VE) | $0$ | $\sqrt{d\sigma^2/dt}$ | variance grows without bound |

DDPM is a discretisation of VP; the original NCSN of Song & Ermon (2019) is a discretisation of VE.

### The Fokker–Planck Equation

If $\mathbf{X}_t$ obeys (3) and has density $p(\mathbf{x}, t)$, then $p$ satisfies the **Fokker–Planck equation** (Kolmogorov forward equation):
$$\boxed{\;\frac{\partial p}{\partial t} \;=\; -\nabla\!\cdot\!\bigl(f\,p\bigr) \;+\; \tfrac{1}{2}\,g^2\,\nabla^2 p\;.\;} \tag{4}$$
**Sketch of proof.** For any smooth test function $\varphi$, Itô's formula gives
$$d\varphi(\mathbf{X}_t) = \bigl(f\!\cdot\!\nabla\varphi + \tfrac{1}{2}g^2\nabla^2\varphi\bigr)\,dt + g\,\nabla\varphi\!\cdot\!d\mathbf{B}_t.$$
Taking expectations kills the martingale term, and writing $\mathbb{E}[\varphi(\mathbf{X}_t)] = \int \varphi\,p\,d\mathbf{x}$ then integrating by parts (using that $\varphi$ is arbitrary) yields (4). $\blacksquare$

**Sanity check.** Setting $f \equiv 0$ and $g^2/2 = D$ in (4) recovers the heat equation $\partial_t p = D\,\nabla^2 p$. The Fokker–Planck equation is exactly the heat equation plus a drift term.

### The Kolmogorov Backward Equation

For a terminal payoff $g(\mathbf{X}_T)$, the conditional expectation $u(s, \mathbf{x}) = \mathbb{E}[g(\mathbf{X}_T)\,|\,\mathbf{X}_s = \mathbf{x}]$ satisfies the **backward** equation
$$\partial_s u + f\!\cdot\!\nabla u + \tfrac{1}{2}g^2 \nabla^2 u = 0,$$
with terminal condition $u(T, \mathbf{x}) = g(\mathbf{x})$. The forward equation evolves densities forward in time; the backward equation evolves expectations backward. Together they are the Feynman–Kac correspondence — and the time-reversed forward SDE is exactly what we will use to *generate* samples.

---

## Score-Based Generative Models

![PDE and ML (7): Diffusion Models and Score Matching — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/illustration_2.png)

### The Score Function

The **score** of a density $p$ is
$$\mathbf{s}(\mathbf{x}) \;:=\; \nabla_{\mathbf{x}}\,\log p(\mathbf{x}). \tag{5}$$
Three useful properties:

- **Normalisation-free.** $\nabla \log(p / Z) = \nabla \log p$, so we never need the partition function.
- **Closed form for Gaussians.** If $p = \mathcal{N}(\boldsymbol\mu, \sigma^2 \mathbf{I})$, then $\mathbf{s}(\mathbf{x}) = -(\mathbf{x} - \boldsymbol\mu) / \sigma^2$.
- **Zero mean.** $\mathbb{E}_p[\mathbf{s}(\mathbf{x})] = \mathbf{0}$ (under mild regularity).

Geometrically, the score is a vector field that always points *toward* high-probability regions and grows large in low-density valleys.

![Score of a 2-mode Gaussian mixture: the field points uphill, away from low-density regions and toward the modes.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig3_score_field.png)
*Score of a 2-mode Gaussian mixture: the field points uphill, away from low-density regions and toward the modes.*

### Score Matching

Because $p$ is unknown we cannot directly minimise $\mathbb{E}_p\,\|\mathbf{s}_\theta - \nabla\log p\|^2$. There are three workable surrogates.

**Implicit Score Matching (ISM, Hyvärinen 2005).** Integration by parts removes the unknown score:
$$\mathcal{L}_{\text{ISM}}(\theta) = \mathbb{E}_p\Bigl[\,\tfrac{1}{2}\|\mathbf{s}_\theta(\mathbf{x})\|^2 + \mathrm{tr}\bigl(\nabla\mathbf{s}_\theta(\mathbf{x})\bigr)\Bigr].$$
The trace of the Jacobian is expensive in high dimensions.

**Sliced Score Matching (SSM, Song et al. 2019).** Project onto random directions $\mathbf{v}$:
$$\mathcal{L}_{\text{SSM}}(\theta) = \mathbb{E}_{\mathbf{v}, \mathbf{x}}\Bigl[\,\mathbf{v}^\top \nabla\mathbf{s}_\theta(\mathbf{x})\,\mathbf{v} + \tfrac{1}{2}(\mathbf{v}^\top \mathbf{s}_\theta(\mathbf{x}))^2\Bigr].$$
One Hessian–vector product per sample suffices.

**Denoising Score Matching (DSM, Vincent 2011) — the workhorse.** Add noise $\tilde{\mathbf{x}} = \mathbf{x} + \sigma\,\boldsymbol\eta$, $\boldsymbol\eta \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, and learn to predict the noise direction:
$$\boxed{\;\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}\Bigl[\bigl\|\mathbf{s}_\theta(\tilde{\mathbf{x}}) + \tfrac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma^2}\bigr\|^2\Bigr].\;} \tag{6}$$
Vincent showed (6) has the same minimiser as matching the *true* score of the noisy distribution $p_\sigma = p * \mathcal{N}(0, \sigma^2 \mathbf{I})$. As $\sigma \to 0$, $p_\sigma \to p$. In practice we anneal $\sigma$ during training.

![Left: DSM loss decreases monotonically and plateaus. Right: the learned score matches the true $\nabla\log p$ in high-density regions; near low-density valleys (centre) it is intentionally smoothed by the noise level $\sigma$.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig5_score_matching_loss.png)
*Left: DSM loss decreases monotonically and plateaus. Right: the learned score matches the true $\nabla\log p$ in high-density regions; near low-density valleys (centre) it is intentionally smoothed by the noise level $\sigma$.*

**Implementation: denoising score matching.** DSM is the training objective that makes diffusion models work. Here is a minimal but complete implementation:

```python
import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    # Simple MLP score network for 2D data
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),  # input: (x, y, t)
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)  # output: score (sx, sy)
        )

    def forward(self, x, t):
        # t is scalar noise level, broadcast to batch
        t_embed = t.unsqueeze(-1) if t.dim() == 1 else t
        return self.net(torch.cat([x, t_embed], dim=-1))

def dsm_loss(model, x0, sigma):
    # Denoising Score Matching loss
    noise = torch.randn_like(x0)
    x_noisy = x0 + sigma * noise
    # True score of q(x_noisy | x0) = -noise / sigma
    target = -noise / sigma
    pred = model(x_noisy, sigma * torch.ones(x0.shape[0], 1))
    return ((pred - target)**2).mean()

sigmas = torch.logspace(start=-2, end=1, steps=10)  # 0.01 to 10

model = ScoreNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
x0 = torch.tensor(make_two_moons(5000), dtype=torch.float32)

for step in range(5000):
    idx = torch.randint(len(sigmas), (1,))
    sigma = sigmas[idx]
    loss = dsm_loss(model, x0, sigma)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")
```sql

The annealing over multiple noise levels is critical: small $\sigma$ gives accurate score estimates near the data but poor coverage in low-density regions; large $\sigma$ gives broad coverage but imprecise scores. The multi-scale approach gets both.

### Langevin Dynamics

Once we have $\mathbf{s}_\theta$ we can sample with Langevin MCMC:
$$\mathbf{x}_{k+1} = \mathbf{x}_k + \tfrac{\epsilon}{2}\,\mathbf{s}_\theta(\mathbf{x}_k) + \sqrt{\epsilon}\,\boldsymbol\eta_k,\qquad \boldsymbol\eta_k \sim \mathcal{N}(\mathbf{0}, \mathbf{I}). \tag{7}$$
As $\epsilon \to 0$ and $k \to \infty$ the chain converges to $p$. The deterministic term is *exploitation* (climb the score), the noise term is *exploration* (escape local maxima).

### Anderson's Reverse-Time SDE

Here is the keystone result that turns score matching into a generative model. **Anderson (1982)** showed that the time-reversal of (3) is itself an SDE:
$$\boxed{\;d\mathbf{X}_t = \bigl[\,f(\mathbf{X}_t, t) - g(t)^2\,\nabla\log p_t(\mathbf{X}_t)\,\bigr]\,dt + g(t)\,d\bar{\mathbf{B}}_t,\;} \tag{8}$$
where $\bar{\mathbf{B}}_t$ is a Brownian motion in reverse time and $p_t$ is the marginal of (3) at time $t$. **The only thing that depends on the data distribution is $\nabla\log p_t$ — exactly what the score network learns**. Run (8) backward from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ and you generate a sample from (approximately) the data distribution.

![Reverse diffusion runs from $t = T$ down to $t = 0$; the score network supplies the drift correction needed to reverse the noising process.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig2_reverse_diffusion.png)
*Reverse diffusion runs from $t = T$ down to $t = 0$; the score network supplies the drift correction needed to reverse the noising process.*

---

## From Continuous Theory to DDPM and DDIM

### DDPM: Forward Process in Closed Form

![Forward diffusion process: structured point cloud dissolving into Gaussian noise](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/anim_forward_diffusion.gif)

Pick a noise schedule $\{\beta_t\}_{t=1}^T$. Define $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. The DDPM forward process is the discrete-time Markov chain
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\bigl(\mathbf{x}_t;\,\sqrt{\alpha_t}\,\mathbf{x}_{t-1},\,\beta_t\mathbf{I}\bigr),$$
which has the convenient closed form
$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1 - \bar\alpha_t}\,\boldsymbol\epsilon,\qquad \boldsymbol\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I}). \tag{9}$$
This is exactly the Euler–Maruyama discretisation of the VP-SDE, so $\bar\alpha_T \to 0$ and $\mathbf{x}_T \approx \mathcal{N}(\mathbf{0}, \mathbf{I})$.

### The DDPM Loss = Weighted DSM

Train a network $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)$ to predict the noise that was added:
$$\mathcal{L}_{\text{DDPM}}(\theta) = \mathbb{E}_{t,\,\mathbf{x}_0,\,\boldsymbol\epsilon}\Bigl[\,\bigl\|\boldsymbol\epsilon_\theta(\mathbf{x}_t, t) - \boldsymbol\epsilon\bigr\|^2\Bigr]. \tag{10}$$
Why is this score matching in disguise? Because (9) implies
$$\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\boldsymbol\epsilon}{\sqrt{1 - \bar\alpha_t}},$$
so the network is learning a scaled score: $\mathbf{s}_\theta(\mathbf{x}_t, t) = -\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)/\sqrt{1 - \bar\alpha_t}$. (10) is precisely (6) with weights $w(t) = 1$.

**Implementation: a complete DDPM training loop.** Below is a minimal but runnable DDPM implementation for 2D data. The key insight: we train an $\boldsymbol\epsilon$-predictor (not a score predictor directly), and the loss is simply the MSE between predicted and actual noise.

```python
import torch
import torch.nn as nn

class EpsilonNet(nn.Module):
    # Time-conditioned noise predictor for 2D data
    def __init__(self, T=1000, hidden=256):
        super().__init__()
        self.time_embed = nn.Embedding(T, hidden)
        self.net = nn.Sequential(
            nn.Linear(2 + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2)
        )

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))

def ddpm_train_step(model, x0, alpha_bar, T):
    # Sample random timestep
    t = torch.randint(0, T, (x0.shape[0],))
    eps = torch.randn_like(x0)
    # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
    ab = alpha_bar[t].unsqueeze(-1)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
    # Predict noise
    eps_pred = model(x_t, t)
    return ((eps_pred - eps)**2).mean()

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

model = EpsilonNet(T=T)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)
x0 = torch.tensor(make_two_moons(5000), dtype=torch.float32)

for step in range(10000):
    loss = ddpm_train_step(model, x0, alpha_bar, T)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 2000 == 0:
        print(f"Step {step}: loss={loss.item():.4f}")
```text

The training loop is strikingly simple: pick a random timestep, add the corresponding noise, predict the noise, backpropagate. The complexity is in the architecture (here a simple MLP; for images, a U-Net with attention) and the noise schedule.

**Noise schedules: linear vs cosine.** The choice of $\beta_t$ matters more than you might think:

| Schedule | Formula | Behaviour | Best for |
|----------|---------|-----------|----------|
| **Linear** | $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$ | Aggressive early noise | DDPM (Ho et al. 2020) |
| **Cosine** | $\bar\alpha_t = \cos^2(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2})$ | Gentler decay | Improved DDPM (Nichol & Dhariwal 2021) |
| **Sigmoid** | $\bar\alpha_t = \sigma(-a + 2a \cdot t/T)$ | Steeper mid, flat ends | Stable Diffusion 3 |

The linear schedule wastes capacity: in the first few hundred steps, $\bar\alpha_t$ is still close to 1 (barely noised), so the model spends many timesteps learning to predict near-zero noise. The cosine schedule spreads the "information content" more evenly across timesteps.

### DDIM: The Probability-Flow ODE

A beautiful fact about (3): there is a **deterministic** ODE with the same one-time marginals at every $t$,
$$\boxed{\;\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \tfrac{1}{2}\,g(t)^2\,\nabla\log p_t(\mathbf{x}).\;} \tag{11}$$
This is the **probability-flow ODE**. Marginals match because both (8) and (11) yield the same Fokker–Planck equation (4) for $p_t$. Solving (11) backward in time with a high-order ODE solver — Heun, RK4, DPM-Solver — is the basis of DDIM and its descendants. The result is *deterministic* (same noise → same image), supports much larger step sizes (25–50 vs 1000), and is exactly invertible (one can encode an image back to its latent noise).

![DDPM (left) injects fresh noise at each reverse step; DDIM (right) follows a deterministic flow under the same learned score, reaching the modes in far fewer steps.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig4_ddpm_vs_ddim.png)
*DDPM (left) injects fresh noise at each reverse step; DDIM (right) follows a deterministic flow under the same learned score, reaching the modes in far fewer steps.*

**Implementation: DDIM sampling.** DDIM replaces the stochastic reverse step with a deterministic ODE step, enabling dramatically fewer sampling steps:

```python
@torch.no_grad()
def ddpm_sample(model, alpha_bar, betas, n=2000, T=1000):
    # Full DDPM sampling (1000 steps, stochastic)
    x = torch.randn(n, 2)
    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, dtype=torch.long)
        eps_pred = model(x, t_batch)
        ab = alpha_bar[t]
        ab_prev = alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
        beta = betas[t]
        # DDPM reverse step
        mean = (1 / torch.sqrt(1 - beta)) * (x - beta / torch.sqrt(1 - ab) * eps_pred)
        if t > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
    return x

@torch.no_grad()
def ddim_sample(model, alpha_bar, n=2000, steps=50, T=1000):
    # DDIM sampling (50 steps, deterministic)
    # Sub-sample timesteps evenly
    timesteps = torch.linspace(T-1, 0, steps).long()
    x = torch.randn(n, 2)
    for i in range(len(timesteps)):
        t = timesteps[i]
        t_batch = torch.full((n,), t, dtype=torch.long)
        eps_pred = model(x, t_batch)
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[timesteps[i+1]] if i+1 < len(timesteps) else torch.tensor(1.0)
        # DDIM deterministic step (eta=0)
        x0_pred = (x - torch.sqrt(1 - ab_t) * eps_pred) / torch.sqrt(ab_t)
        x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1 - ab_prev) * eps_pred
    return x

samples_ddpm = ddpm_sample(model, alpha_bar, betas, n=2000, T=T)
samples_ddim = ddim_sample(model, alpha_bar, n=2000, steps=50, T=T)
print(f"DDPM mean: {samples_ddpm.mean(0).numpy().round(3)}")
print(f"DDIM mean: {samples_ddim.mean(0).numpy().round(3)}")
```sql

DDIM uses $20\times$ fewer model evaluations while producing comparable quality. The key difference: DDPM injects fresh noise at each step (SDE), DDIM does not (ODE). For DDIM, the same initial noise always produces the same output — this enables latent-space interpolation and inversion.

### A Unified View

| Method | Process | Typical steps | Deterministic? | Strength |
|--------|---------|---------------|----------------|----------|
| DDPM | Reverse SDE (8) | ~1000 | No | Diversity, simple training |
| DDIM | Probability-flow ODE (11) | ~25–50 | Yes | Speed, exact inversion |
| DPM-Solver | Higher-order ODE | ~10–20 | Yes | Even faster, same fidelity |
| EDM (Karras et al.) | Continuous, refined preconditioning | ~30 | Tunable | SOTA quality |

### The PDE → Diffusion Model Map

Putting it all together:

![Heat equation $\to$ Fokker–Planck $\to$ forward SDE; Anderson's time-reversal needs $\nabla\log p_t$, which the score network learns by DSM; the same score then drives DDPM (SDE) or DDIM (ODE).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig6_pde_diffusion_bridge.png)
*Heat equation $\to$ Fokker–Planck $\to$ forward SDE; Anderson's time-reversal needs $\nabla\log p_t$, which the score network learns by DSM; the same score then drives DDPM (SDE) or DDIM (ODE).*

---

### Score Network Architecture: The U-Net

For image generation, the score network $\boldsymbol\epsilon_\theta(\mathbf{x}_t, t)$ is a **U-Net** — an encoder-decoder with skip connections. The key design choices:

- **Time conditioning:** sinusoidal positional embedding of $t$, projected through a linear layer and added to each residual block (similar to transformer positional encoding).
- **Spatial resolution:** the encoder downsamples by $2\times$ at each level (typically 4 levels: 64→32→16→8 for 64px images).
- **Self-attention:** inserted at the 16×16 and 8×8 resolution levels. Attention at higher resolution is too expensive.
- **Cross-attention (for conditioning):** text embeddings from CLIP/T5 enter via cross-attention at the same resolution levels.
- **Group normalisation + SiLU activation** throughout (not BatchNorm, which interacts poorly with noise levels).

The architecture can be summarised as:

```
Input x_t (C×H×W) + time embedding t
  ↓
ResBlock → ResBlock → Downsample (×4 levels)
  ↓     skip connections ↓
Bottleneck (self-attention + ResBlock)
  ↓     skip connections ↓
ResBlock → ResBlock → Upsample (×4 levels)
  ↓
Output eps_pred (C×H×W)
```text

The total parameter count for a typical image model is 100M–900M (vs ~1M for our 2D toy examples). The skip connections are essential: without them, fine spatial detail is lost in the bottleneck and the model cannot reconstruct high-frequency content — exactly the content that diffusion destroys first and must reconstruct last.

## Latent Diffusion: Stable Diffusion in One Picture

Pixel-space diffusion on $512 \times 512$ images is expensive: every U-Net forward pass operates on ~$8\times 10^5$ floats. **Latent Diffusion** (Rombach et al., 2022) trains a VAE-like autoencoder $(\mathcal{E}, \mathcal{D})$ first to map images to an $\sim\!8\times$ smaller latent $\mathbf{z}_0 = \mathcal{E}(\mathbf{x})$, and runs the entire diffusion process *in latent space*. Decoding $\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z}_0)$ is a single feed-forward pass.

Conditioning (text, class labels, depth maps, ControlNet poses, …) enters the U-Net through cross-attention.

![Stable Diffusion = autoencoder + diffusion in latent space + cross-attention conditioning.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/07-Diffusion-Models/fig7_latent_diffusion.png)
*Stable Diffusion = autoencoder + diffusion in latent space + cross-attention conditioning.*

The compute saving is roughly $f^{2d}$ where $f$ is the spatial downsampling factor (typically 8) and $d=2$, i.e. ~$64\times$. It is the architectural trick that turned diffusion models into a consumer technology.

---

### Classifier-Free Guidance

The most impactful practical technique in modern diffusion models is **classifier-free guidance** (Ho & Salimans, 2022). During training, randomly drop the conditioning signal (e.g., text prompt) with probability $p_{\text{uncond}} \approx 0.1$. At inference, combine the conditional and unconditional predictions:

$$\hat{\boldsymbol\epsilon}(\mathbf{x}_t, t, c) = \boldsymbol\epsilon_\theta(\mathbf{x}_t, t, \varnothing) + w\,\bigl[\boldsymbol\epsilon_\theta(\mathbf{x}_t, t, c) - \boldsymbol\epsilon_\theta(\mathbf{x}_t, t, \varnothing)\bigr], \tag{12}$$

where $w > 1$ is the **guidance scale**. Setting $w = 1$ recovers the vanilla conditional model; $w = 7$–$15$ is typical for text-to-image.

```python
@torch.no_grad()
def guided_sample(model, alpha_bar, cond, w=7.5, steps=50, T=1000):
    # Classifier-free guidance sampling
    timesteps = torch.linspace(T-1, 0, steps).long()
    x = torch.randn(cond.shape[0], 2)
    null_cond = torch.zeros_like(cond)  # unconditional embedding
    for i in range(len(timesteps)):
        t = timesteps[i]
        t_batch = torch.full((x.shape[0],), t, dtype=torch.long)
        # Two forward passes: conditional + unconditional
        eps_cond = model(x, t_batch, cond)
        eps_uncond = model(x, t_batch, null_cond)
        # Guided prediction
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
        # DDIM step with guided eps
        ab_t = alpha_bar[t]
        ab_prev = alpha_bar[timesteps[i+1]] if i+1 < len(timesteps) else torch.tensor(1.0)
        x0_pred = (x - torch.sqrt(1 - ab_t) * eps_guided) / torch.sqrt(ab_t)
        x = torch.sqrt(ab_prev) * x0_pred + torch.sqrt(1 - ab_prev) * eps_guided
    return x
```sql

**Why it works:** guidance amplifies the difference between "what the model generates given the prompt" and "what it generates unconditionally". This pushes samples toward regions that are *unusually likely under the condition* — tighter, more prompt-aligned generations at the cost of some diversity. Mathematically, it approximates sampling from $p(x|c)^w \cdot p(x)^{1-w}$, a sharpened conditional distribution.

| Guidance scale $w$ | Effect | Typical use |
|---------------------|--------|-------------|
| $w = 1$ | No guidance, vanilla conditional | Diversity-focused |
| $w = 3$–$5$ | Mild guidance | Creative exploration |
| $w = 7$–$10$ | Strong guidance | Text-to-image (DALL-E, SD) |
| $w > 15$ | Over-saturated, artefacts | Usually too high |

## Connection to Scientific Computing

Score-based diffusion is not just a generative-modelling trick — it is a tool for sampling from arbitrary, possibly intractable, probability distributions. Two application directions are particularly relevant for the PDE community:

1. **Conditional generation under PDE constraints.** Train a score model $\mathbf{s}_\theta(\mathbf{x}, t \mid \mathcal{C})$ where $\mathcal{C}$ encodes a PDE residual or boundary data. Reverse sampling produces fields that respect the constraint (in distribution). This is the basis of *diffusion posterior sampling* for inverse problems (Chung et al., 2023): tomography, super-resolution, deblurring, etc.

2. **Surrogate samplers for Bayesian inference.** Many Bayesian inverse problems require sampling from a posterior $p(\theta \mid \mathbf{y}) \propto p(\mathbf{y}\mid\theta)\,p(\theta)$. If we can train a score model on $(\theta, \mathbf{y})$ pairs from simulation, conditional reverse sampling replaces MCMC.

The unifying message: **whenever you need to sample from a high-dimensional, multimodal distribution and you can afford to simulate forward dynamics, the diffusion-model recipe applies**.

---

## Exercises

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

## What's next

The handful of core ideas in this chapter (PDE residual as loss, operators on function spaces, Wasserstein geometry, symplectic structure, scores, diffusion) recur throughout the rest of the series. If a section stalls you, jot the question down and keep reading — the next chapter usually re-explains it from a different angle.

The fastest sanity check on your own understanding is to run this chapter's equation on a minimal example: a 1-D heat equation, a single pendulum, a 2-D Gaussian mixture. The code is short, but it converts "looks right" into "it's right on my machine."

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

*This is Part 7 of the [PDE and Machine Learning](/en/categories/pde-and-machine-learning/) series. Next: [Part 8 — Reaction-Diffusion Systems and GNN](/en/pde-ml/08-reaction-diffusion-systems). Previous: [Part 6 — Continuous Normalizing Flows](/en/pde-ml/06-continuous-normalizing-flows).*
