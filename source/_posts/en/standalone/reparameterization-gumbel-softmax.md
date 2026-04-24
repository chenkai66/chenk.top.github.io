---
title: "Reparameterization Trick & Gumbel-Softmax: A Deep Dive"
date: 2025-01-08 09:00:00
tags:
  - ML
  - Deep Learning
  - Generative Models
categories: Algorithm
lang: en
mathjax: true
description: "Make sense of the reparameterization trick and Gumbel-Softmax: why gradients can flow through sampling, how temperature trades bias for variance, and the practical pitfalls of training discrete latent variables end-to-end."
---

The moment your model contains a sampling step, training hits a hard wall: **how do gradients flow through a random node?**

The reparameterization trick has a clean answer — rewrite $z\sim p_\theta(z)$ as $z=g_\theta(\epsilon)$, isolating the randomness in a parameter-free noise variable $\epsilon$, so backprop can flow through $g_\theta$. The trouble starts with discrete variables: operations like $\arg\max$ are not differentiable. **Gumbel-Softmax** (a.k.a. the Concrete distribution) replaces the discrete sample with a tempered softmax over Gumbel-perturbed logits, giving you a smooth, differentiable surrogate that you can train end-to-end.

This post walks through the derivations, the intuition, the implementation details, the bias-variance trade-off behind the temperature, and the most common training pitfalls.

## What you'll learn

- Why gradients cannot flow through $z\sim\mathcal N(\mu,\sigma^2)$, but flow freely through $z=\mu+\sigma\epsilon$.
- Where the Gumbel distribution comes from and **why** "logits + Gumbel noise + argmax = softmax sampling" is exact.
- How the Gumbel-Softmax temperature $\tau$ trades bias for variance, plus annealing in practice.
- The Straight-Through estimator (ST-GS): hard forward, soft backward.
- Variance comparison vs. REINFORCE / score-function estimators (typically 1–3 orders of magnitude lower).
- Full PyTorch implementations for both continuous and discrete VAEs, with the most common training pitfalls.

## Prerequisites

- Expectations, densities, change-of-variables.
- Basic PyTorch and the chain rule under autograd.
- Familiarity with the variational autoencoder (see the companion post [VAE: From Intuition to Implementation](../vae-guide/)).

---

# 1. Why we need reparameterization

The general training objective looks like

$$
\mathcal L(\theta) \;=\; \mathbb E_{z\sim q_\theta(z)}\,[\,f(z)\,].
$$

In a VAE, $f$ is the reconstruction log-likelihood; in RL, it's a return; in discrete structure learning, it's the downstream loss. **The trouble**: $\theta$ appears both inside $f$ and inside the distribution. Naively "sample $z$, evaluate $f(z)$, call `.backward()`" breaks because the computation graph is severed at the sampling step.

A blunt PyTorch example:

```python
# Wrong: sampling inside the graph
mu, logvar = encoder(x)
sigma = torch.exp(0.5 * logvar)
z = torch.normal(mu, sigma)        # <- not differentiable
recon = decoder(z)
loss = (recon - x).pow(2).sum()
loss.backward()                    # mu, sigma receive no gradient
```

`torch.normal(mu, sigma)` is a **stochastic op**: there is no smooth functional relationship between its output $z$ and its inputs $\mu,\sigma$. Change $\mu$ a tiny bit and the *distribution* shifts, but a particular drawn $z$ does not move smoothly with it — there is no derivative to define.

> **Core problem**: training requires $\nabla_\theta \mathbb E_{q_\theta}[f(z)]$, but the sampling step kills backprop.

Two families of fixes exist:

1. **Score-function / REINFORCE**: rewrite the gradient as $\mathbb E[f(z)\nabla_\theta\log q_\theta(z)]$. Universal, works on any random variable — **but the variance is enormous**.
2. **Reparameterization**: pull the randomness *out of the parameters*, replace it with parameter-free noise, and keep the graph differentiable. **Low variance and end-to-end trainable, but requires a distribution that admits such a rewrite**.

# 2. Reparameterizing continuous distributions

## 2.1 The general form

Express $z$ as a deterministic, differentiable function of a parameter-free noise $\epsilon$:

$$
z \;=\; g_\theta(\epsilon),\qquad \epsilon \sim p(\epsilon),
$$

where $p(\epsilon)$ is a fixed base distribution (e.g. $\mathcal N(0,I)$ or $\mathrm{Uniform}(0,1)$). Substituting back,

$$
\mathcal L(\theta) \;=\; \mathbb E_{\epsilon\sim p(\epsilon)}\,[\,f(g_\theta(\epsilon))\,].
$$

Now $\theta$ no longer appears inside the expectation, so we can swap differentiation and expectation:

$$
\nabla_\theta\,\mathcal L(\theta) \;=\; \mathbb E_{\epsilon\sim p(\epsilon)}\,[\,\nabla_\theta\, f(g_\theta(\epsilon))\,].
$$

A Monte Carlo estimate is just: draw one (or a few) $\epsilon$, run autograd through $f(g_\theta(\epsilon))$.

![Reparameterization trick](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/reparameterization-gumbel-softmax/fig1_reparam_trick.png)

> **Side-by-side**: on the left, $z\sim\mathcal N(\mu,\sigma^2)$ is a stochastic node and gradients cannot flow through it; on the right, $z=\mu+\sigma\epsilon$ is a deterministic function of $\mu,\sigma$ and gradients flow through the green path back into the encoder.

## 2.2 The Gaussian case

The textbook example: for $z\sim\mathcal N(\mu,\sigma^2)$,

$$
z \;=\; \mu \;+\; \sigma \odot \epsilon,\qquad \epsilon \sim \mathcal N(0,I).
$$

Verify on both sides: (i) $\mathbb E[z]=\mu$, $\mathrm{Var}(z)=\sigma^2$; (ii) since linear transforms preserve normality, $z$'s marginal is exactly $\mathcal N(\mu,\sigma^2)$. **Crucially**, $\partial z/\partial\mu=1$ and $\partial z/\partial\sigma=\epsilon$ — both trivial expressions, gradients are unobstructed.

In PyTorch, predict $\log\sigma^2$ rather than $\sigma$ for numerical stability:

```python
def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
    """z = mu + sigma * eps,  eps ~ N(0, I)."""
    std = torch.exp(0.5 * logvar)        # sigma >= 0 by construction
    eps = torch.randn_like(std)          # the only random source
    return mu + std * eps
```

## 2.3 In the VAE

The VAE optimizes the evidence lower bound (ELBO):

$$
\mathcal L_{\text{ELBO}}(\theta,\phi;x)
\;=\; \mathbb E_{q_\phi(z|x)}\!\bigl[\log p_\theta(x|z)\bigr]
\;-\;\mathrm{KL}\!\bigl(q_\phi(z|x)\,\Vert\,p(z)\bigr).
$$

The expectation in the first term is taken w.r.t. $q_\phi(z|x)$, which has $\phi$ inside. **Reparameterization is what makes this term differentiable**: writing $z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$,

$$
\nabla_\phi\,\mathbb E_{q_\phi(z|x)}[\log p_\theta(x|z)]
\;=\;\mathbb E_{\epsilon\sim\mathcal N(0,I)}\!\bigl[\,\nabla_\phi\log p_\theta(x\mid \mu_\phi+\sigma_\phi\epsilon)\,\bigr].
$$

The gradient w.r.t. $\phi$ flows through the decoder all the way back into the encoder. The KL term has a closed form when $q_\phi=\mathcal N(\mu,\sigma^2)$ and $p=\mathcal N(0,I)$:

$$
\mathrm{KL}=\tfrac12\sum_j\!\bigl(\mu_j^2+\sigma_j^2-1-\log\sigma_j^2\bigr),
$$

so no Monte Carlo is needed there.

## 2.4 Which continuous distributions are "naturally" reparameterizable?

Anything that admits a location-scale form, or a base-noise + differentiable transform:

| Distribution | Reparameterization |
|--------------|--------------------|
| $\mathcal N(\mu,\sigma^2)$ | $z=\mu+\sigma\epsilon$, $\epsilon\sim\mathcal N(0,1)$ |
| $\mathrm{Logistic}(\mu,s)$ | $z=\mu+s\,\log\frac{u}{1-u}$, $u\sim\mathrm U(0,1)$ |
| $\mathrm{Laplace}(\mu,b)$ | $z=\mu-b\,\mathrm{sign}(u)\log(1-2|u|)$, $u\sim\mathrm U(-\tfrac12,\tfrac12)$ |
| $\mathrm{Exp}(\lambda)$ | $z=-\tfrac1\lambda\log(1-u)$, $u\sim\mathrm U(0,1)$ |
| $\mathrm{Gumbel}(\mu,\beta)$ | $z=\mu-\beta\log(-\log u)$, $u\sim\mathrm U(0,1)$ |

**Counter-examples**: Gamma, Beta, Dirichlet, Student-t do not have a simple location-scale form. They need *implicit reparameterization gradients* or pathwise tricks (Figurnov et al., NeurIPS 2018; see §7).

# 3. Reparameterizing discrete distributions: the Gumbel-Max trick

## 3.1 The difficulty

Take a $K$-class categorical $\mathrm{Cat}(\pi_1,\dots,\pi_K)$ with $\pi_i=\mathrm{softmax}(\alpha)_i=\frac{\exp(\alpha_i)}{\sum_j\exp(\alpha_j)}$. The naive "sample" is: compute $\pi$, draw a class index $k$ from a multinomial. That step is **completely non-differentiable** — its output is a one-hot vector, with no notion of smooth variation.

The fallback is the score-function estimator:

$$
\nabla_\alpha\,\mathbb E_{k\sim\mathrm{Cat}(\pi)}[f(k)]
\;=\;\mathbb E_{k}\bigl[f(k)\,\nabla_\alpha\log\pi_k\bigr],
$$

universal but with variance large enough to wreck training stability. Can we, like in the Gaussian case, factor sampling into "noise + deterministic transform"? The answer is the Gumbel-Max trick.

## 3.2 A quick tour of the Gumbel distribution

The standard Gumbel $\mathrm{Gumbel}(0,1)$ has CDF/PDF

$$
F(g)=\exp(-e^{-g}),\qquad f(g)=\exp\!\bigl(-(g+e^{-g})\bigr).
$$

Mode at $0$, mean at the Euler–Mascheroni constant $\gamma\approx 0.5772$. Crucially, it is the **extreme-value distribution** for maxima of iid samples (under suitable scaling). Gumbel-Max exploits exactly this property.

Inverse-CDF sampling is one line:

$$
u\sim\mathrm U(0,1) \;\Rightarrow\; g=-\log(-\log u)\sim\mathrm{Gumbel}(0,1).
$$

![Gumbel distribution PDF/CDF/empirical](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/reparameterization-gumbel-softmax/fig2_gumbel_pdf.png)

> **Left**: the Gumbel(0,1) PDF; **middle**: inverse-CDF sampling visualized; **right**: 20k samples drawn via $-\log(-\log u)$, with the empirical histogram matching the analytic PDF.

## 3.3 The Gumbel-Max trick

**Claim.** Let $g_1,\dots,g_K\overset{iid}{\sim}\mathrm{Gumbel}(0,1)$ and define

$$
k^\star \;=\; \arg\max_i\,(\alpha_i + g_i).
$$

Then $k^\star\sim\mathrm{Cat}(\mathrm{softmax}(\alpha))$. In words, **adding Gumbel noise to logits and taking argmax is equivalent to sampling from the softmax distribution**.

### Proof (probability of class 1)

$\Pr[k^\star=1] = \Pr\bigl[\,\forall i\neq 1:\;\alpha_1+g_1>\alpha_i+g_i\,\bigr] = \Pr\bigl[\,\forall i\neq 1:\;g_i<\alpha_1-\alpha_i+g_1\,\bigr]$.

Conditioning on $g_1=t$:

$$
\Pr[k^\star=1\mid g_1=t]=\prod_{i\neq 1}F(\alpha_1-\alpha_i+t)
=\prod_{i\neq 1}\exp\!\bigl(-e^{-(\alpha_1-\alpha_i+t)}\bigr)
=\exp\!\Bigl(-e^{-t}\!\!\sum_{i\neq 1}e^{\alpha_i-\alpha_1}\Bigr).
$$

Let $S=\sum_{i\neq 1}e^{\alpha_i-\alpha_1}$ and integrate over $g_1$:

$$
\Pr[k^\star=1]=\int e^{-(t+e^{-t})}\cdot\exp(-S e^{-t})\,dt
=\int e^{-t}\exp\!\bigl(-(1+S)e^{-t}\bigr)\,dt.
$$

Substituting $u=(1+S)e^{-t}$, $du=-(1+S)e^{-t}dt$ gives $\frac{1}{1+S}\int_0^\infty e^{-u}du=\frac{1}{1+S}$.

But $1+S=1+\sum_{i\neq 1}e^{\alpha_i-\alpha_1}=e^{-\alpha_1}\sum_i e^{\alpha_i}$, so

$$
\Pr[k^\star=1]=\frac{e^{\alpha_1}}{\sum_i e^{\alpha_i}}=\mathrm{softmax}(\alpha)_1. \quad\blacksquare
$$

![Gumbel-Max trick](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/reparameterization-gumbel-softmax/fig3_gumbel_max_trick.png)

> **Left**: target categorical distribution; **middle**: a single draw — adding Gumbel noise to logits and taking the argmax selects class c2; **right**: 8000 draws — the empirical frequencies match the target softmax probabilities, confirming exact equivalence in expectation.

### Why this matters

- **Efficient**: $K$ uniforms + one argmax. No explicit normalization, no cumulative-distribution lookup.
- **Numerically stable**: addition followed by argmax is invariant to constant shifts in the logits.
- **Most importantly**: all the randomness now lives in $g$; the logits $\alpha$ enter through the deterministic $\alpha+g$. We are one step away from "differentiable" — just need to soften the $\arg\max$.

# 4. Gumbel-Softmax: softening the argmax

## 4.1 Definition

Replace $\arg\max$ with a temperature-scaled softmax to obtain a sample from the **Gumbel-Softmax / Concrete** distribution:

$$
y_i \;=\; \frac{\exp\!\bigl((\alpha_i + g_i)/\tau\bigr)}{\sum_{j=1}^K\exp\!\bigl((\alpha_j + g_j)/\tau\bigr)},
\qquad g_i\overset{iid}{\sim}\mathrm{Gumbel}(0,1).
$$

Here $y\in\Delta^{K-1}$, the $(K-1)$-simplex — a continuous vector. The two limits give intuition:

- $\tau\to 0^+$: softmax degenerates into the indicator of the argmax, $y$ becomes one-hot — **a true discrete sample**;
- $\tau\to\infty$: $(\alpha_i+g_i)/\tau\to 0$, $y$ becomes the uniform vector $(1/K,\dots,1/K)$.

Because $g$ is independent of $\alpha$ and softmax is everywhere differentiable, $y$ is **differentiable in $\alpha$** (and therefore in any upstream parameter $\theta$). This is the heart of Gumbel-Softmax: a smooth, differentiable proxy for the discrete one-hot.

## 4.2 The temperature bias-variance trade-off

![Gumbel-Softmax temperature effect](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/reparameterization-gumbel-softmax/fig4_gumbel_softmax_temp.png)

> **Four temperatures**: in each panel the light-blue bars show the target softmax, while the orange polylines are five independent Gumbel-Softmax draws.
>
> - $\tau=5$: very smooth, near-uniform — **low gradient variance** but **highly biased** away from a true discrete sample;
> - $\tau=1$: balanced;
> - $\tau=0.5$: starts looking one-hot;
> - $\tau=0.1$: nearly hard one-hot — **low bias** but **high variance** (and softmax internals are prone to over/underflow).

You can read this off the gradient estimator directly: smaller $\tau$ makes $y$ closer to the true one-hot $z$, reducing bias; but the softmax Jacobian $\partial y/\partial\alpha$ scales like $1/\tau$, so variance grows like $\tau^{-2}$.

**Practical annealing**:

- Start at $\tau\in[1.0, 2.0]$, end at $\tau\in[0.1, 0.5]$.
- Exponential schedule $\tau_t=\max(\tau_{\min},\tau_0\,e^{-rt})$ with a check every $\sim 1000$ steps.
- **Don't** drive $\tau\to 0$ — softmax numerics blow up and gradient variance dominates. Let the model "see" the soft distribution early, then "harden" it later.

## 4.3 Straight-Through Gumbel-Softmax (ST-GS)

Many tasks — hard attention, discrete token selection, sparse routing — **must** use a strict one-hot in the forward pass (e.g. the downstream is an embedding lookup expecting an integer index). The fix is the **Straight-Through estimator**:

$$
\boxed{
\;\;y_{\text{hard}}=\mathrm{onehot}(\arg\max_i y_i),
\qquad
\tilde y \;=\; y_{\text{hard}}\;-\;\mathrm{stop\_grad}(y_{\text{soft}})\;+\;y_{\text{soft}}.\;\;
}
$$

Forward: $\tilde y=y_{\text{hard}}$ (strict one-hot). Backward: $\partial\tilde y/\partial\alpha=\partial y_{\text{soft}}/\partial\alpha$ (the soft Jacobian). It's a **biased but low-variance** estimator: hard sample for the loss, soft sample for the gradient.

In PyTorch:

```python
def gumbel_softmax(logits: Tensor, tau: float = 1.0,
                   hard: bool = False) -> Tensor:
    """y = softmax((logits + g) / tau); optional ST-hard."""
    # 1) Gumbel noise; clamp to avoid log(0)
    u = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-9)
    g = -torch.log(-torch.log(u))
    # 2) soft sample
    y_soft = F.softmax((logits + g) / tau, dim=-1)
    if not hard:
        return y_soft
    # 3) Straight-Through: hard forward, soft backward
    idx = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
    return y_hard - y_soft.detach() + y_soft
```

PyTorch ships `torch.nn.functional.gumbel_softmax(logits, tau, hard)` with the same semantics; rolling your own is just to make every step explicit.

# 5. Comparison with REINFORCE: orders-of-magnitude variance gap

## 5.1 Score-function / REINFORCE estimator

General form:

$$
\nabla_\theta\,\mathbb E_{z\sim q_\theta}[f(z)]
=\mathbb E_{z\sim q_\theta}\!\bigl[f(z)\,\nabla_\theta\log q_\theta(z)\bigr].
$$

Its strength: it makes **no assumption** about differentiability of $z$ in $\theta$ — discrete, control-flow, even black-box external calls all work. Its weakness: very high variance. Intuitively, the entire scalar value $f(z)$ rides on the gradient with no differentiable path to cancel signs.

Standard variance reductions:

- **Baselines / control variates**: replace $f(z)$ with $f(z)-b$ for some $z$-independent $b$ (e.g. a running average).
- **Rao–Blackwellization**, antithetic sampling, importance sampling, etc.

Even with all these tricks, REINFORCE typically remains 1–3 orders of magnitude noisier than reparameterization.

## 5.2 Empirical comparison

We estimate $\nabla_{\alpha_0}\,\mathbb E_z[r^\top z]$ on a synthetic 8-class categorical (with a fixed reward vector $r$). Each curve aggregates 200 trials.

![Variance: REINFORCE vs Gumbel-Softmax](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/reparameterization-gumbel-softmax/fig5_discrete_pipeline.png)

> **Left pipeline**: logits → +Gumbel noise → /τ → softmax → $y_{\text{soft}}$; optional argmax → $y_{\text{hard}}$, combined via STE for hard-forward / soft-backward. **Right curves**: $x$-axis is the number of MC samples per gradient estimate, $y$-axis is the variance of the estimate (log-log). Gumbel-Softmax with $\tau=0.5$ is consistently more than an order of magnitude lower than REINFORCE; both decay as $1/n$ (slope $-1$).

This is exactly why end-to-end discrete training only became practical once reparameterization-based estimators were available.

# 6. Full PyTorch: continuous and discrete VAEs

## 6.1 Continuous VAE (reparameterization)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, x_dim: int = 784, h_dim: int = 400,
                 z_dim: int = 20):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_lv = nn.Linear(h_dim, z_dim)
        # decoder
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h_dim), nn.ReLU(),
            nn.Linear(h_dim, x_dim),
        )

    def encode(self, x):
        h = self.enc(x)
        return self.fc_mu(h), self.fc_lv(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar


def vae_loss(logits_x, x, mu, logvar):
    # logits + BCE-with-logits is more numerically stable than sigmoid + BCE
    recon = F.binary_cross_entropy_with_logits(
        logits_x, x, reduction="sum")
    # Closed-form KL: KL(N(mu, sigma^2) || N(0, I))
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl
```

**Implementation notes**:

- Use `binary_cross_entropy_with_logits` instead of `BCELoss + sigmoid` to avoid numerical issues in saturation regions.
- Compute the KL term in closed form (rather than via sampling) to reduce gradient variance.
- Optionally anneal KL ($\beta$ in $\beta$-VAE from 0 → 1) to mitigate posterior collapse.

## 6.2 Discrete-latent VAE (categorical + Gumbel-Softmax)

```python
class CategoricalVAE(nn.Module):
    def __init__(self, x_dim=784, h_dim=400, n_cat=10, n_dim=20):
        """n_cat independent categoricals, each with K = n_dim classes."""
        super().__init__()
        self.n_cat, self.n_dim = n_cat, n_dim
        self.enc = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU(),
                                 nn.Linear(h_dim, n_cat * n_dim))
        self.dec = nn.Sequential(nn.Linear(n_cat * n_dim, h_dim),
                                 nn.ReLU(), nn.Linear(h_dim, x_dim))

    def forward(self, x, tau: float, hard: bool = False):
        logits = self.enc(x).view(-1, self.n_cat, self.n_dim)
        # Gumbel-Softmax along the class dimension
        z = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        z_flat = z.view(-1, self.n_cat * self.n_dim)
        return self.dec(z_flat), logits


def cat_vae_loss(logits_x, x, q_logits):
    """Closed-form KL: KL(q || Uniform(K)) = log K - H(q)."""
    recon = F.binary_cross_entropy_with_logits(
        logits_x, x, reduction="sum")
    K = q_logits.size(-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    q = log_q.exp()
    kl = (q * (log_q + torch.log(torch.tensor(float(K))))).sum()
    return recon + kl


# Inside the training loop, anneal temperature:
# tau = max(tau_min, tau0 * exp(-r * step))
```

**Implementation notes**:

- Treat each latent as a $K$-way categorical; logits have shape `[B, n_cat, K]`, softmax over the **last** dimension.
- KL has the closed form $\mathrm{KL}(q\|\mathrm{Uniform})=\log K - H(q)$ — no sampling required.
- `hard=True` enables ST-GS (forces a true discrete forward pass); `hard=False` keeps gradients smoother.
- Anneal: `tau0=1.0`, `tau_min=0.5`, `r=1e-5`, checked every ~1k steps.

# 7. Common training pitfalls

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| **NaN loss** | $\tau$ annealed too low; softmax over/underflow; or $u=0$ in $\log(-\log u)$ | clamp $u\in[\epsilon, 1-\epsilon]$; keep $\tau_{\min}\ge 0.1$ |
| **Gradient variance explodes** | One MC sample + tiny $\tau$ | More MC samples per batch; slow the $\tau$ schedule |
| **Discrete structure not actually "hard"** | Using soft outputs at eval time | Set `hard=True` at inference, or `argmax`; switch to ST-GS late in training |
| **Posterior collapse** (continuous VAE) | KL too strong from the start | Anneal $\beta$ from 0 → 1; use free-bits |
| **Discrete VAE doesn't learn structure** | Decoder too powerful, KL too weak | Reduce decoder capacity; lower KL weight, then raise it back |
| **Gradients insensitive to logits** | $\tau$ too large, output near-uniform | Lower $\tau$ or raise the learning rate |
| **Forward / backward mismatch** | ST-GS line written incorrectly | Re-check `y_hard - y_soft.detach() + y_soft` |

# 8. Recent work

- **Implicit Reparameterization Gradients** (Figurnov et al., NeurIPS 2018) — uses the implicit function theorem to make Gamma, Beta, Dirichlet, Student-t reparameterizable, with low bias and clean derivations.
- **REBAR / RELAX** (Tucker et al., NeurIPS 2017; Grathwohl et al., ICLR 2018) — combine Gumbel-Softmax with REINFORCE and learn a control variate via a neural network, yielding **unbiased and lower-variance** discrete gradient estimators.
- **Hard Concrete gates** (Louizos et al., ICLR 2018) — stretch and clip Concrete/Gumbel-Softmax samples to $[0,1]$, producing differentiable gates with literal zeros for $L_0$ regularization and sparsification.
- **Top-$k$ Gumbel** and **Plackett–Luce** — extend Gumbel-Max to **without-replacement** sampling of $k$ classes; useful for sparse attention and routing.
- **Permutation-equivariant relaxations** (Mena et al., ICLR 2018) — Sinkhorn operator + Gumbel noise to differentiate through permutation matrices.

# Summary

- In continuous settings, reparameterization rewrites a random variable $z$ as a deterministic, differentiable transform $g_\theta(\epsilon)$ of a fixed noise — gradients flow through the deterministic path back to the parameters. This is what makes VAEs trainable with SGD.
- In discrete settings, Gumbel-Max gives the exact equivalence "logits + Gumbel noise + argmax = softmax sample"; Gumbel-Softmax then softens the argmax into a temperature-scaled softmax, making the entire sampling step differentiable.
- The temperature $\tau$ is the central knob: small $\tau$ → low bias / high variance; large $\tau$ → the opposite. Annealing (large → small) is the key to stable training.
- When a strict discrete forward pass is required, use ST-GS: hard forward, soft backward — the most common engineering compromise.
- Versus REINFORCE, reparameterization-based estimators give 1–3 orders of magnitude lower gradient variance — without that, end-to-end discrete training would not be practical.

# References

- E. Jang, S. Gu, B. Poole. "Categorical Reparameterization with Gumbel-Softmax." *ICLR*, 2017.
- C. Maddison, A. Mnih, Y. W. Teh. "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables." *ICLR*, 2017.
- D. P. Kingma, M. Welling. "Auto-Encoding Variational Bayes." *ICLR*, 2014.
- M. Figurnov, S. Mohamed, A. Mnih. "Implicit Reparameterization Gradients." *NeurIPS*, 2018.
- G. Tucker et al. "REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models." *NeurIPS*, 2017.
- C. Louizos, M. Welling, D. P. Kingma. "Learning Sparse Neural Networks through $L_0$ Regularization." *ICLR*, 2018.
- W. Kool, H. van Hoof, M. Welling. "Stochastic Beams and Where to Find Them: The Gumbel-Top-$k$ Trick for Sampling Sequences Without Replacement." *ICML*, 2019.
