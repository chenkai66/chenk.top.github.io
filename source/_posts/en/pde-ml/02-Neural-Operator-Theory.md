---
title: "PDE and Machine Learning (2) — Neural Operator Theory"
tags:
  - PDE
  - Neural Operator
  - FNO
  - DeepONet
categories: Scientific Computing
lang: en
mathjax: true
---

A classical PDE solver — finite difference, finite element, spectral — is a function: feed it one initial condition and one set of coefficients, get back one solution. A PINN is the same kind of object dressed in neural-network clothes: each new initial condition demands a fresh round of training. Switch the inflow velocity on a wing or move a single sensor reading in a forecast and you reset the clock.

Neural operators throw that contract away. They learn the **solution map itself** — a function whose input is a function (an initial condition, a coefficient field, a forcing) and whose output is another function (the solution at some later time, on some target domain). Train the operator once on a few thousand precomputed pairs, and any new instance is one forward pass. No re-meshing, no re-optimising, no PDE solver in the loop at inference time.

This article is a deep dive into how that is possible. We start from the functional-analytic setup that makes "learning a map between function spaces" mean something precise, then unpack the two architectures that dominate the literature: **DeepONet**, which decomposes the operator into a branch and a trunk in the spirit of the Chen–Chen universal approximation theorem, and the **Fourier Neural Operator (FNO)**, which performs a learnable global convolution in the spectral domain. Along the way we look at why neural operators are *resolution invariant*, what error bounds the theory actually delivers, and where each architecture quietly fails.

![Operator learning maps an entire function space of inputs to an entire function space of outputs.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/02-Neural-Operator-Theory/fig1_operator_concept.png)
*Figure 1. Neural operators learn a map between two infinite-dimensional function spaces. Each blue curve on the left is one input (e.g. an initial condition); each green curve on the right is the corresponding output (e.g. the solution at time T). The operator $\mathcal{G}_\theta$ is trained once and reused for every new input.*

## 1. Why operators, not just bigger networks

### 1.1 The PINN ceiling

Consider the 1D viscous Burgers equation,
$$
u_t + u\,u_x = \nu\,u_{xx},\qquad x\in[0,2\pi],\ t\in[0,T],
$$
with periodic boundary conditions and an initial condition $u(\cdot,0)=u_0$. A finite-difference solver hands you $u(\cdot,T)$ in milliseconds *for one specific* $u_0$ and one specific $\nu$. A PINN replaces the discretisation with a neural network $u_\theta(x,t)$ trained to minimise the residual
$$
\mathcal{L}_{\mathrm{PINN}}(\theta) = \big\| \partial_t u_\theta + u_\theta\,\partial_x u_\theta - \nu\,\partial_{xx} u_\theta \big\|^2 + \big\|u_\theta(\cdot,0) - u_0 \big\|^2 .
$$
Notice how $u_0$ appears explicitly inside the loss. **Change $u_0$ and the loss landscape changes; the optimiser has to start over.** For a design study with a thousand candidate inflow profiles, that is a thousand training runs.

### 1.2 Operator learning, formally

Let $\mathcal{A}$ be the space of admissible inputs (e.g. initial conditions in some Sobolev class) and $\mathcal{U}$ the space of solutions. The PDE defines a *solution operator*
$$
\mathcal{G} : \mathcal{A} \to \mathcal{U},\qquad \mathcal{G}(u_0) = u(\cdot, T).
$$
Both $\mathcal{A}$ and $\mathcal{U}$ are infinite dimensional. The goal of operator learning is to fit a parametric surrogate $\mathcal{G}_\theta \approx \mathcal{G}$ from a finite training set $\{(a^{(i)}, \mathcal{G}(a^{(i)}))\}_{i=1}^N$, so that for every new input $a$,
$$
u(\cdot, T) \approx \mathcal{G}_\theta(a) \quad \text{(one forward pass).}
$$
The training cost is amortised over the entire family of instances. This is the trade we want.

## 2. Function-space foundations (the bare minimum)

### 2.1 Banach, Hilbert, Sobolev — what each one buys you

A **Banach space** is a normed vector space in which every Cauchy sequence converges. The norm gives us a notion of "size of a function," and completeness gives us limits — without it we cannot even talk about convergence of a learning algorithm. The two everyday examples are
$$
C(K) \;=\; \{u:K\to\mathbb{R} \text{ continuous}\}, \quad \|u\|_\infty = \sup_{x\in K} |u(x)|,
$$
and
$$
L^p(\Omega) \;=\; \Big\{u:\Omega\to\mathbb{R} \;:\; \int_\Omega |u|^p < \infty\Big\}, \quad \|u\|_{L^p} = \Big(\int |u|^p\Big)^{1/p}.
$$

A **Hilbert space** is a Banach space whose norm comes from an inner product. The inner product gives us **angles, orthogonality and projections** — the entire machinery of Fourier series and spectral decomposition. The canonical example is $L^2(\Omega)$ with $\langle u, v\rangle = \int u\bar v$.

A **Sobolev space** $H^s(\Omega)$ controls not just the function but its derivatives up to order $s$ in $L^2$. PDE solutions usually live in some $H^s$, and the parameter $s$ measures *smoothness*. This is where the resolution-invariance argument will land: smoother functions have spectra that decay quickly, so we can throw away high-frequency modes without paying much.

### 2.2 Why the input dimension is "infinite"

When a CNN ingests a $256\times 256$ image, it really sees a vector in $\mathbb{R}^{65536}$. Doubling the resolution to $512\times 512$ changes the input dimension and breaks the model. A neural operator instead ingests a *function* and only samples it for numerical purposes. The architecture must therefore be designed so that the prediction at a query location $y$ is **independent of how the input was sampled**. This is the design constraint that distinguishes operator learning from "a CNN that happens to read a PDE solution."

## 3. The Chen–Chen theorem: an operator can be a two-layer net

The theoretical seed of operator learning was planted in 1995 by Chen and Chen.

**Theorem (Chen–Chen, 1995).** *Let $K_1\subset C(D_1)$ and $K_2\subset D_2$ be compact, and let $\mathcal{G}: K_1 \to C(K_2)$ be continuous. For every $\varepsilon>0$ there exist a non-polynomial continuous activation $\sigma$, an integer $p$, sensor points $\{x_j\}_{j=1}^m\subset D_1$, and parameters $\{c_{ij}, w_{kj}, \zeta_k, \theta_i\}$ such that*
$$
\sup_{u\in K_1,\,y\in K_2}\;\bigg|\,\mathcal{G}(u)(y) \;-\; \sum_{k=1}^{p} \underbrace{\sigma\!\Big(\textstyle\sum_{j=1}^m c_{kj}\,u(x_j) + \theta_k\Big)}_{=\,b_k(u)} \cdot \underbrace{\sigma(w_k\cdot y + \zeta_k)}_{=\,t_k(y)} \,\bigg| \;<\;\varepsilon.
$$

Two things are remarkable:

1. **Inputs are functions, but the network sees only a finite sample $u(x_1),\dots,u(x_m)$.** Compactness of $K_1$ in the uniform topology means a finite mesh suffices to capture every function in the class to whatever fidelity you need.
2. **The output factorises** as a sum of products $b_k(u)\,t_k(y)$. The first factor depends only on $u$, the second only on $y$. This is exactly DeepONet.

![Chen-Chen theorem: the existence guarantee that drives DeepONet and bounds the achievable error.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/02-Neural-Operator-Theory/fig7_universal_approx.png)
*Figure 2. Left: the Chen–Chen statement says any continuous operator $\mathcal{G}: C(K_1)\to C(K_2)$ can be approximated to arbitrary precision by a sum of $p$ rank-one terms $b_k(u)\,t_k(y)$. Right: an illustrative error decomposition. The total test error of a neural operator is the sum of an approximation term that decays with the network width $p$, a statistical term that decays with the sample size $N$, and an irreducible noise floor.*

The theorem only gives existence: it says nothing about how $p$ scales with $\varepsilon$, with the smoothness of $\mathcal{G}$, or with the input dimension. Sharper rates have been proven for restricted operator classes (Lanthaler et al., 2022; Kovachki et al., 2023), and the asymptotic picture is the familiar bias–variance–noise decomposition of the right-hand panel.

## 4. DeepONet: branch and trunk in practice

DeepONet (Lu et al., 2019) is the direct architectural realisation of Chen–Chen. Two networks compute the two factors:

- **Branch network** $b_\phi : \mathbb{R}^m \to \mathbb{R}^p$ ingests the input function sampled at $m$ fixed sensor locations and emits a coefficient vector $b(u) = b_\phi(u(x_1),\dots,u(x_m))$.
- **Trunk network** $t_\psi : \mathbb{R}^d \to \mathbb{R}^p$ ingests a query coordinate $y$ and emits a basis vector $t(y)$.

The prediction is their inner product (plus an optional bias):
$$
\mathcal{G}_\theta(u)(y) \;=\; \sum_{k=1}^{p} b_k(u)\, t_k(y) \;+\; b_0.
$$

![DeepONet architecture: branch encodes the input function, trunk encodes the query location, the inner product produces the answer at that location.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/02-Neural-Operator-Theory/fig2_deeponet_architecture.png)
*Figure 3. DeepONet decomposes the operator into a branch network (top, encodes the input function from $m$ sensor samples) and a trunk network (bottom, encodes the query coordinate). Their inner product reconstructs $\mathcal{G}(u)(y)$. The trunk plays the role of a learned basis $\{t_k\}$ and the branch produces the coefficients $\{b_k(u)\}$ in that basis.*

There are several lenses for this decomposition that all illuminate the same structure:

- **Low-rank operator factorisation.** Comparing with the singular-value decomposition $\mathcal{G} = \sum_k \sigma_k \langle\cdot, e_k\rangle f_k$, the trunk plays the role of the right singular system and the branch absorbs both the singular value and the inner product with the input.
- **Learned basis functions.** The trunk outputs $\{t_k(\cdot)\}$ form a *data-driven basis* over the output domain. Unlike Fourier or Chebyshev bases they are not orthogonal but they are tailored to the operator at hand.
- **Conditional linear model.** For fixed $u$, the prediction is linear in the trunk outputs; the entire nonlinearity in $u$ is hidden inside the branch.

A few practical wrinkles matter when you implement it:

| Concern | Lever |
|---|---|
| Sensor placement | $\{x_j\}$ should resolve the smallest scale you care about; uniform grids are fine for periodic problems, Latin hypercube or quasi-MC for higher dimensions. |
| Query strategy | Train with a random subset of $y$ per input to keep memory bounded; evaluate on a dense grid. |
| Trunk regularisation | Initialise the last trunk layer small (e.g. $\mathcal{N}(0, 0.1)$) so that early training does not produce wildly oscillatory bases. |
| Multiple outputs | For vector-valued $\mathcal{G}(u)\in\mathbb{R}^c$, share the trunk and use $c$ branch heads — this enforces a common spatial basis across components. |

**Variants worth knowing.** *POD-DeepONet* replaces the learned trunk with a precomputed POD basis from the training data and only learns the branch; it converges faster but inherits all the limitations of a fixed basis. *Physics-informed DeepONet* (Wang et al., 2021) adds a PDE-residual term to the loss, which is invaluable when labelled data is scarce.

## 5. The Fourier Neural Operator

Where DeepONet decomposes the operator as a sum of rank-one terms in the spatial domain, FNO (Li et al., 2020) operates in the *spectral* domain. The motivation is the convolution theorem.

### 5.1 Why spectral

For a translation-invariant linear operator $K$, the Schwartz kernel theorem gives
$$
(K v)(x) = \int \kappa(x - x')\,v(x')\,\mathrm{d}x',
$$
i.e. a convolution. The convolution theorem turns this into pointwise multiplication after Fourier transform:
$$
\widehat{K v}(k) \;=\; \widehat{\kappa}(k)\cdot \widehat{v}(k).
$$
So instead of learning the kernel $\kappa$ in space — costly, with a large support — we learn $\widehat{\kappa}(k)$ directly in frequency. Each frequency mode is just one complex multiplication.

For nonlinear PDEs the operator is no longer a global convolution, but the **local-nonlinearity / global-linearity** decomposition is extremely common: dissipation, dispersion and propagation are all linear and translation-invariant; reaction and advection terms are pointwise nonlinear. FNO bakes this split into the architecture.

### 5.2 The Fourier layer

A single FNO block computes
$$
v_{\ell+1}(x) \;=\; \sigma\!\Big(\;W\,v_\ell(x) \;+\; \mathcal{F}^{-1}\!\big( R_\theta \cdot \mathcal{F}(v_\ell)\big)(x) \;\Big),
$$
where $\mathcal{F}$ is the FFT, $R_\theta$ is a learnable complex tensor that multiplies each retained Fourier mode, $W$ is a $1\times 1$ convolution that handles the residual contribution from truncated and aliased modes, and $\sigma$ is a pointwise nonlinearity (typically GELU). The lifting layer $P:\mathbb{R}^{d_{\mathrm{in}}}\to\mathbb{R}^{d_v}$ embeds the input into a higher-dimensional channel space and a final projection $Q:\mathbb{R}^{d_v}\to\mathbb{R}^{d_{\mathrm{out}}}$ recovers the answer.

![FNO spectral convolution: take the FFT of the channel, keep the lowest k_max modes, multiply by a learnable filter, inverse-FFT, and add the local residual W v.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/02-Neural-Operator-Theory/fig3_fno_spectral_layer.png)
*Figure 4. Anatomy of one Fourier layer. Top: the schematic flow — FFT, learnable spectral multiplier $R_\theta$ truncated to the lowest $k_{\max}$ modes, inverse FFT, plus a $1\times 1$ residual $Wv$ and a pointwise nonlinearity. Bottom (left to right): a sample input $v(x)$ with a mix of low- and high-frequency components; its full power spectrum; the spectrum after multiplication by the learnable filter $R_\theta$ and truncation at $k_{\max}=12$; the reconstructed signal in physical space.*

Three design choices are worth defending:

**Mode truncation.** We keep only the first $k_{\max}$ Fourier modes; weights for higher modes are forced to zero. The justification is the **spectral decay of Sobolev functions**: if $u \in H^s$ then $|\hat u_k| \lesssim \|u\|_{H^s}\,|k|^{-s}$, so for any $s>1/2$ the tail energy decays faster than any sampling error you would have introduced anyway. Truncation acts as an implicit regulariser.

**Aliasing residual.** FFT assumes periodicity. If the data is not periodic, high-frequency content gets folded back onto low-frequency modes (aliasing). The $W v$ branch is a $1\times 1$ convolution that can learn to absorb this aliasing error rather than letting it pollute the spectral path.

**Channel lifting.** The FFT mixes spatial information across one dimension, but a single channel is too narrow a bottleneck. By first lifting to $d_v\in\{32,64,128\}$ channels, the network gets enough room to encode multiple "modes" of physical behaviour (e.g. transport vs. diffusion vs. shock formation).

### 5.3 What the cost looks like

For an $n$-point grid in $d$ spatial dimensions the dominant costs per Fourier layer are:

- FFT: $\mathcal{O}(n^d \log n)$ — sublinear in spatial extent.
- Spectral multiplication: $\mathcal{O}(k_{\max}^d \cdot d_v^2)$ — independent of the spatial grid.
- Pointwise residual $Wv$ and activation: $\mathcal{O}(n^d \cdot d_v^2)$ — linear in grid size.

Compare with a CNN that needs depth proportional to $n$ to achieve a global receptive field: FNO trades that depth for one FFT and one inverse FFT per layer.

## 6. Resolution invariance: train at 64, test at 256

Here is the property that finally separates neural operators from "a CNN with extra steps." Both DeepONet and FNO, when implemented carefully, are **discretisation invariant**: the same trained weights produce a coherent answer regardless of the resolution at which you sample the input or query the output.

For DeepONet this is built into the architecture: the trunk takes coordinates $y$ as input and so can be queried at any density. The branch needs sensors at fixed locations during training, but POD-style or attention-based branches relax this requirement.

For FNO the argument is sharper. The spectral multiplier $R_\theta$ acts on Fourier modes, not on grid points. A finer spatial grid simply gives access to more modes; the modes you have already learned still mean the same thing. Concretely, if your training grid resolves modes up to $k_{\max}=16$ and your test grid resolves modes up to $k_{\max}=64$, the modes from $16$ to $64$ are zeroed by the truncation — you do not get a wrong answer, you only forfeit the resolution gain there.

![Resolution invariance: one trained network evaluated on three grid resolutions produces consistent fields.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/02-Neural-Operator-Theory/fig4_resolution_invariance.png)
*Figure 5. The same operator weights, trained on a coarse $32\times 32$ grid (left), evaluated at $64\times 64$ (centre) and $256\times 256$ (right). The coarse and fine evaluations agree on the resolved scales; the fine-grid prediction simply contains more degrees of freedom for downstream consumers.*

A subtler point: **the error does not decay arbitrarily as resolution increases**, because the model never learned the higher modes. In practice, evaluation at up to $\sim 4\times$ the training resolution is reliable; beyond that the high-frequency content is "best guess" interpolation and you should retrain.

## 7. Three ways to attack a PDE: PINN, FNO, DeepONet

It is worth zooming out before diving into implementation. PINNs, FNOs and DeepONets are not interchangeable; they cover different parts of the design space.

![Capability profile and cost-vs-reuse landscape for PINN, FNO and DeepONet against a classical solver baseline.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/02-Neural-Operator-Theory/fig5_method_comparison.png)
*Figure 6. Left: a qualitative capability matrix. PINN scores high on geometry flexibility (it does not need a structured grid) but cannot reuse training across instances. FNO is the fastest at inference and most robust under resolution change but assumes a structured grid. DeepONet is in between: it generalises across instances and across geometry but is slightly slower than FNO and slightly less accurate on translation-invariant problems. Right: the cost vs. reusability landscape — both neural operators occupy the "train once, reuse forever" corner that classical solvers and PINNs miss.*

| Aspect | Classical solver | PINN | DeepONet | FNO |
|---|---|---|---|---|
| Scope | one instance | one instance | family | family |
| Inference per new instance | full solve | full retrain | one forward pass | one forward pass |
| Geometry | mesh-based, flexible | mesh-free, flexible | branch + trunk, very flexible | structured grid (Cartesian / torus) |
| Resolution invariance | mesh-dependent | network-fixed | natively invariant | natively invariant up to $k_{\max}$ |
| Strength | accuracy, guarantees | physics-only training | irregular geometries, multi-physics | translation-invariant problems, speed |
| Weakness | curse of dimensionality | per-instance cost | needs paired data | periodicity assumption |

A useful rule of thumb: **if your problem has a structured grid and translation invariance** (turbulence in a periodic box, weather on a torus-like global grid, Burgers and Navier–Stokes on a square), reach for FNO. **If your geometry is complex or you need pointwise queries on irregular meshes**, reach for DeepONet. **If you have one instance and physics but no data**, reach for PINN. Neural operators do not deprecate PINNs; they slot in where PINNs were always inappropriate.

![A single trained operator solves an entire PDE family: four different coefficient fields produce four different solutions in one forward pass each.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/02-Neural-Operator-Theory/fig6_geometry_application.png)
*Figure 7. The payoff. A neural operator trained on (coefficient field, solution) pairs for the steady Darcy equation $-\nabla\!\cdot\!(a(x)\nabla u) = f$ generalises across the entire input distribution. Top row: four sample coefficient fields $a(x)$ (high values mean low permeability). Bottom row: the corresponding pressure solutions $u(x)$ produced in one forward pass per instance. Replacing each forward pass with a finite-element solve would cost orders of magnitude more.*

## 8. Implementation: a 1D FNO from scratch

The minimum runnable FNO in PyTorch fits in a single file. The two non-obvious pieces are (a) the spectral multiplication done with `torch.einsum` so the code generalises to multiple channels, and (b) the residual `W v` branch implemented as a $1\times 1$ convolution.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    """Multiply the lowest ``modes`` Fourier coefficients by a learnable complex tensor."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weight = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, N]
        B, _, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)                  # [B, C, N//2+1]
        out_ft = torch.zeros(
            B, self.out_channels, N // 2 + 1,
            device=x.device, dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes] = torch.einsum(
            "bik,iok->bok", x_ft[:, :, : self.modes], self.weight
        )
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FNO1d(nn.Module):
    """Four Fourier layers + lift + project. Input shape [B, N], output shape [B, N]."""

    def __init__(self, modes: int = 16, width: int = 64, n_layers: int = 4):
        super().__init__()
        self.lift = nn.Linear(2, width)                   # u(x) and grid coord
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)]
        )
        self.local = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        self.proj = nn.Sequential(nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, u: torch.Tensor) -> torch.Tensor:   # u: [B, N]
        B, N = u.shape
        grid = torch.linspace(0.0, 1.0, N, device=u.device).repeat(B, 1)
        x = torch.stack([u, grid], dim=-1)                 # [B, N, 2]
        x = self.lift(x).transpose(1, 2)                   # [B, width, N]
        for spec, loc in zip(self.spectral, self.local):
            x = F.gelu(spec(x) + loc(x))
        return self.proj(x.transpose(1, 2)).squeeze(-1)    # [B, N]
```

The training loop is a standard supervised regression on $(u_0, u_T)$ pairs generated offline by any reliable solver:

```python
model = FNO1d(modes=16, width=64).cuda()
opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

for epoch in range(epochs):
    for u0, uT in loader:                                  # both [B, N]
        opt.zero_grad()
        loss = F.mse_loss(model(u0.cuda()), uT.cuda())
        loss.backward()
        opt.step()
    sched.step()
```

**Reference performance on Burgers ($\nu=10^{-2}$, $N=256$, $1{,}000$ training instances):** validation relative $L^2$ error around $1\%$, training time on a single A100 in the order of minutes. Compare with a PINN that takes minutes *per* instance and you immediately see the operator's economic advantage.

## 9. What the theory actually guarantees (and where it doesn't)

The cleanest results are about FNO and were established by Kovachki and collaborators.

**Approximation (Kovachki et al., 2021).** For any continuous operator $\mathcal{G}: H^s(\mathbb{T}^d)\to H^{s'}(\mathbb{T}^d)$ on a torus and any compact $K\subset H^s$, and any $\varepsilon>0$, there exists an FNO $\mathcal{G}_\theta$ with finite width and depth such that $\sup_{u\in K}\|\mathcal{G}(u)-\mathcal{G}_\theta(u)\|_{H^{s'}} < \varepsilon$.

**Statistical (Lanthaler et al., 2022 for DeepONet).** With $N$ i.i.d. training pairs and a $p$-dimensional latent, the expected test error decomposes as
$$
\mathbb{E}\,\|\mathcal{G}-\mathcal{G}_\theta\|^2 \;\lesssim\; \underbrace{p^{-2\alpha}}_{\text{approx}} \;+\; \underbrace{\frac{p}{N}}_{\text{stat}} \;+\; \underbrace{\sigma^2}_{\text{noise}},
$$
for an operator with smoothness exponent $\alpha$. Choosing $p\sim N^{1/(2\alpha+1)}$ balances approximation against statistics.

What is **not** guaranteed in the same generality:

- Convergence in the *strong* operator norm uniformly over unbounded sets.
- Stability under distribution shift in the input (extrapolation in $\nu$, in geometry, in forcing amplitude).
- Behaviour at sharp fronts and shocks where the Sobolev embedding fails.

Each of these is an active research front. In practice you should always treat a deployed neural operator as a *fast surrogate* whose extrapolation must be sanity-checked against an occasional reference solve.

## 10. Limits, failure modes, and what to do about them

**Aliasing on non-periodic domains.** FNO assumes periodicity. On non-periodic problems either pad and window the input, or switch to a non-Fourier basis (Spectral Neural Operator with Chebyshev or Legendre, Tran et al. 2022).

**Long-horizon roll-outs.** Auto-regressive prediction $u_{t+\Delta t} = \mathcal{G}_\theta(u_t)$ accumulates error. Mitigations: train on multi-step roll-outs from the start, add a stability penalty, or interleave with a coarse PDE step (hybrid solvers).

**Out-of-distribution coefficients.** A model trained on $\nu \in [10^{-2}, 10^{-1}]$ extrapolates poorly to $\nu = 10^{-4}$. Fix: include a parameter token in the input (as in conditional FNO / Unisolver) and sample $\nu$ over a wide log-uniform range during training.

**Shocks and discontinuities.** Sobolev-style smoothness fails. Use entropy-stable formulations of the loss, residual training in conservative variables, or hybrid solvers that fall back to a Godunov step near detected fronts.

**Complex geometry.** FNO is grid-bound. Options: Geo-FNO (learn a deformation to a reference torus), graph neural operators (Brandstetter et al., 2022) for unstructured meshes, or DeepONet with mesh-free trunks.

## 11. Summary

A neural operator is the right tool when the same PDE has to be solved many times across a family of inputs. Two architectures cover most of the design space:

- **DeepONet** factorises the operator into a branch (input function) and a trunk (query coordinate). It inherits the Chen–Chen universal approximation theorem and is naturally flexible across geometries and output queries. Its weakness is that paired training data is mandatory and its accuracy is slightly behind FNO on translation-invariant problems.
- **FNO** does a learnable global convolution in the Fourier domain. It is the fastest at inference, the most natural fit to translation-invariant PDEs, and the cleanest theoretical object — at the cost of needing a structured grid and a (usually periodic) domain.

Both architectures are **resolution invariant** in the sense that the same trained weights evaluate at any grid density up to the modes they have seen. Both train once and amortise the cost over an entire family of PDE instances. Both push the modelling burden away from per-instance optimisation and toward a single careful training curriculum.

The honest summary of the field today is that neural operators have moved from "interesting research idea" to "a credible component in a scientific computing stack" — particularly for forecasting (FourCastNet, GraphCast), uncertainty quantification, and design optimisation, where the same governing PDE is queried thousands of times. Their open frontiers are extrapolation, complex geometry, multi-physics coupling, and rigorous error control under distribution shift.

## References

- Li, Z., Kovachki, N., Azizzadenesheli, K., Liu, B., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2020). [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895). *ICLR 2021*.
- Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2019). [Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators](https://arxiv.org/abs/1910.03193). *Nature Machine Intelligence* 3(3), 218–229.
- Chen, T., & Chen, H. (1995). [Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems](https://ieeexplore.ieee.org/document/376006). *IEEE Transactions on Neural Networks* 6(4), 911–917.
- Kovachki, N., Li, Z., Liu, B., Azizzadenesheli, K., Bhattacharya, K., Stuart, A., & Anandkumar, A. (2023). [Neural Operator: Learning Maps Between Function Spaces with Applications to PDEs](https://arxiv.org/abs/2108.08481). *Journal of Machine Learning Research* 24, 1–97.
- Lanthaler, S., Mishra, S., & Karniadakis, G. E. (2022). [Error estimates for DeepONets: A deep learning framework in infinite dimensions](https://arxiv.org/abs/2102.09618). *Transactions of Mathematics and Its Applications*.
- Wang, S., Wang, H., & Perdikaris, P. (2021). [Learning the solution operator of parametric partial differential equations with physics-informed DeepONets](https://www.science.org/doi/10.1126/sciadv.abi8604). *Science Advances* 7(40).
- Pathak, J., Subramanian, S., Harrington, P., et al. (2022). [FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators](https://arxiv.org/abs/2202.11214). *arXiv:2202.11214*.
- Bonev, B., Kurth, T., Hundt, C., Pathak, J., Baust, M., Kashinath, K., & Anandkumar, A. (2023). [Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere](https://arxiv.org/abs/2306.03838). *ICML 2023*.
- Tran, A., Mathews, A., Xie, L., & Ong, C. S. (2022). [Spectral Neural Operators](https://arxiv.org/abs/2205.10573). *arXiv:2205.10573*.
- Brandstetter, J., Worrall, D., & Welling, M. (2022). [Message Passing Neural PDE Solvers](https://arxiv.org/abs/2202.03376). *ICLR 2022*.
- Wen, G., Li, Z., Long, Q., Azizzadenesheli, K., Anandkumar, A., & Benson, S. M. (2022). [U-FNO: An enhanced Fourier neural operator-based deep-learning model for multiphase flow](https://arxiv.org/abs/2109.03697). *Advances in Water Resources*.
