---
title: "PDE and Machine Learning (1): Physics-Informed Neural Networks"
date: 2024-05-01 09:00:00
tags:
  - PINN
  - Scientific Computing
  - Neural Networks
  - PDE
  - Automatic Differentiation
categories:
  - PDE and Machine Learning
series:
  name: "PDE and Machine Learning"
  part: 1
  total: 8
lang: en
mathjax: true
description: "From finite differences to PINNs: automatic differentiation, PDE residual losses, NTK-based training pathologies, Burgers inverse problems, and a side-by-side comparison with FEM and neural operators. Seven figures included."
disableNunjucks: true
series_order: 1
---

> **Series chapter 1 — about a 35-minute read.** This is the foundation of the entire series. Neural operators, variational principles, score matching — every later chapter is, at heart, *the same idea*: how do we encode physical or mathematical constraints directly into the optimisation objective of a neural network? Get PINNs right and the rest is "swap one constraint for another".

---

## 1 Prologue: a metal rod

Suppose you want the temperature distribution $u(x,t)$ along a metal rod. Half a century of numerical analysis offers two standard answers:

1. **Finite differences (FDM).** Slice $[0,L]$ into $N$ pieces and $[0,T]$ into $M$ pieces, replace the second derivative by a three-point stencil, march forward in time.
2. **Finite elements (FEM).** Triangulate the domain, approximate the solution by a linear polynomial inside each triangle, and require the weak-form residual to be orthogonal to a set of test functions.

Both routes are mature beyond reproach but share one painful prerequisite — **first you must build a mesh.** A 1-D rod is fine; an aircraft wing is annoying; a 10-dimensional state space is a death sentence (the curse of dimensionality: $N\propto h^{-d}$ explodes).

In 2019, Raissi, Perdikaris and Karniadakis [^raissi2019] proposed a third route in *Journal of Computational Physics*:

> **Skip the mesh. Let a neural network $u_\theta(x,t)$ approximate the solution directly, and write "satisfies the PDE" as a loss function.**

The seed of the idea goes back to Lagaris (1998) [^lagaris1998] and even further to Walther Ritz (1908), whose variational method recast solving a PDE as *minimising a functional over a finite-dimensional function space*. Ritz used piecewise polynomials; PINNs use neural networks. The killer move PINNs add is **automatic differentiation**: high-order derivatives are evaluated to machine precision in one line of `torch.autograd.grad`, no truncation error.

![PINN architecture: MLP + automatic differentiation + physics-informed loss.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/01-Physics-Informed-Neural-Networks/fig1_architecture.png)
*Figure 1: PINN architecture. Inputs $(x,t)$ feed an ordinary MLP that outputs $\hat u$; autodiff extracts $\partial_t\hat u$ and $\partial_x^2\hat u$, which assemble the PDE residual. Together with boundary and initial / data residuals it forms the training objective $\mathcal L=\lambda_r\mathcal L_r+\lambda_b\mathcal L_b+\lambda_i\mathcal L_i$.*

This chapter proceeds as follows. §2 quantifies the pain points of classical methods. §3 gives the minimal complete definition of a PINN and shows its equivalence with Ritz–Galerkin. §4 is the heart of the chapter: a Neural Tangent Kernel (NTK) view of *why* PINNs are so often hard to train, and three working remedies. §5 walks through a complete Burgers experiment plus an inverse-problem demonstration. §6 enumerates failure modes and limits. §7 places PINNs on the wider SciML map next to FEM and neural operators.

---

## 2 Classical numerical methods: mature, with edges

### 2.1 Finite differences — intuition at the price of stability

Consider the 1-D heat equation

$$
u_t=\nu u_{xx},\qquad x\in(0,1),\ t>0,
$$

with $u(0,t)=u(1,t)=0$ and $u(x,0)=\sin(\pi x)$. Let $h$ be the spatial step and $\tau$ the time step. Forward Euler gives

$$
\frac{u_i^{n+1}-u_i^n}{\tau}=\nu\frac{u_{i-1}^n-2u_i^n+u_{i+1}^n}{h^2}.
$$

A **Von Neumann analysis** yields the stability condition $\tau\le h^2/(2\nu)$ — the time step must scale like the *square* of the space step. Refining $h$ tenfold demands $\tau$ refined a hundredfold; total cost grows by a factor of one thousand. The implicit Crank–Nicolson scheme is unconditionally stable but pays for it with a tridiagonal solve at every step; in higher dimensions you are at the mercy of sparse direct solvers or multigrid.

**Bottom line.** FDM has a clean global error of $O(\tau+h^2)$ guaranteed by Lax equivalence. It is unbeatable on structured grids and **completely helpless on irregular geometries.**

### 2.2 Finite elements — weak forms and the Ritz functional

The weak form of $-\Delta u=f$: find $u\in H_0^1(\Omega)$ such that

$$
\underbrace{\int_\Omega\nabla u\cdot\nabla v\,\mathrm dx}_{a(u,v)}
=\underbrace{\int_\Omega fv\,\mathrm dx}_{\ell(v)},\qquad\forall v\in H_0^1(\Omega).
$$

This is **equivalent** to minimising the Dirichlet energy $J(u)=\tfrac12 a(u,u)-\ell(u)$. In a piecewise-linear subspace $V_h\subset H_0^1$, write $u_h=\sum c_j\phi_j$ and solve $Kc=f$ with $K_{ij}=a(\phi_i,\phi_j)$ — a sparse symmetric positive-definite stiffness matrix.

Céa's lemma supplies the optimal error bound $\|u-u_h\|_{H^1}\le Ch^k\|u\|_{H^{k+1}}$. **FEM's strengths** are textbook: convergence proofs, error control, adaptive mesh refinement. **The weakness**, again, is the mesh — moving boundaries, porous media and high-dimensional parameter spaces are all hard.

### 2.3 What PINNs are trying to disrupt

Stitching §2.1 and §2.2 together, the shared cost of classical methods is:

| dimension | FDM | FEM |
|---|---|---|
| mesh | required, structured | required, unstructured |
| high-order derivatives | discrete stencil, truncation $O(h^p)$ | weakened to first order via test functions |
| high dimensions | catastrophic | catastrophic |
| complex geometry | hard | moderate (mesh generation expensive) |
| inverse problems | nested optimisation | nested optimisation |
| change boundary / geometry / parameters | recompute everything | recompute everything |

PINNs aim at the last three rows simultaneously: **mesh-free, dimension-friendly, and forward + inverse unified into a single optimisation.** The price is the loss of classical convergence guarantees — those have to be replaced by training tricks.

---

## 3 The minimal complete definition of a PINN

### 3.1 The mathematical statement

Consider a generic PDE

$$
\mathcal N[u](x,t)=0,\quad(x,t)\in\Omega\times(0,T],\qquad
\mathcal B[u]=g\ \text{on}\ \partial\Omega,\qquad
u(x,0)=u_0(x).
$$

A parameterised network $u_\theta:\mathbb R^{d+1}\to\mathbb R$ (typically a tanh-MLP) approximates $u$. Define the composite loss

$$
\boxed{\;
\mathcal L(\theta)=\lambda_r\underbrace{\frac1{N_r}\sum_{i=1}^{N_r}|\mathcal N[u_\theta](x_i^r,t_i^r)|^2}_{\mathcal L_r:\,\text{PDE residual}}
+\lambda_b\underbrace{\frac1{N_b}\sum|\mathcal B[u_\theta]-g|^2}_{\mathcal L_b}
+\lambda_i\underbrace{\frac1{N_i}\sum|u_\theta-u_0|^2}_{\mathcal L_i}
+\lambda_d\underbrace{\frac1{N_d}\sum|u_\theta-u^{\mathrm{obs}}|^2}_{\mathcal L_d}\;}
$$

The last term $\mathcal L_d$ is absent in **forward problems** but central to **inverse problems**. Training is $\theta^\star=\arg\min_\theta\mathcal L(\theta)$ via Adam or L-BFGS.

![Three loss components and a balanced-weighting comparison.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/01-Physics-Informed-Neural-Networks/fig2_loss_decomposition.png)
*Figure 2. Left: with naive equal weights the PDE residual decays quickly while the boundary loss stalls — the network produces what aerodynamicists jokingly call "physics-respecting noise". Right: after NTK-balanced adaptive weighting the three curves descend together — this is what healthy PINN training looks like.*

### 3.2 Why automatic differentiation matters

Naive numerical differentiation,

$$
\partial_x u\approx\frac{u(x+\varepsilon)-u(x-\varepsilon)}{2\varepsilon},
$$

has two killers: $\varepsilon$ too small drowns in floating-point round-off, and high-order derivatives compound the error. Reverse-mode autodiff is symbolic-exact: every elementary operation has a known derivative, the chain rule is automatically composed, and the **result equals the analytic derivative to machine precision.**

```python
# 1-D heat equation residual: u_t - nu * u_xx
def heat_residual(model, x, t, nu=0.1):
    x.requires_grad_(True); t.requires_grad_(True)
    u = model(torch.cat([x, t], dim=1))                # forward pass
    u_t  = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x  = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t - nu * u_xx                             # residual, target -> 0
```

The flag `create_graph=True` keeps the derivative itself in the computation graph, so that `loss = (residual**2).mean()` propagates back to $\nabla_\theta$ correctly.

### 3.3 Isomorphism with the Ritz method

Lifting to the abstract level:

- **Ritz**: minimise $J(u)=\tfrac12 a(u,u)-\ell(u)$ inside a subspace $V_h=\mathrm{span}\{\phi_1,\dots,\phi_n\}$.
- **PINN**: minimise $\mathcal L(\theta)$ inside a subspace $V_\theta=\{u_\theta:\theta\in\mathbb R^p\}$.

Two differences:

1. **Basis functions** — piecewise polynomials vs. neural networks. The latter are nonlinear, smooth, and have a tunable spectrum.
2. **Testing mechanism** — Galerkin uses inner products on test functions; PINNs use collocation + Monte-Carlo approximation of the integral.

Read this way, PINNs are not exotic: they are **"Ritz with the finite-dimensional subspace replaced by a neural network."** The Deep Ritz method [^deepritz] makes this explicit by minimising the energy functional directly rather than the squared residual — better behaved on elliptic problems.

### 3.4 Convergence: yes, but weak

Shin, Darbon and Karniadakis (2020) [^shin2020] proved asymptotic convergence for linear second-order elliptic PDEs: as $N_r\to\infty$, network width $\to\infty$, and $\mathcal L\to 0$, $u_\theta\to u^\star$ in $L^2$. **There is no quantitative rate** like FEM's $O(h^k)$ — the most honest gap between PINNs and classical methods. Subsequent work has supplied Sobolev-norm bounds under restrictive assumptions, but engineering-grade *a priori* convergence orders remain out of reach.

---

## 4 Training pathologies: the part that's actually hard

Anyone who has run a PINN has seen the loss drop from 1 to 0.01 and then refuse to move, or seen boundary conditions fail outright. Three diagnoses follow, with engineering fixes for each.

### 4.1 Pathology A: imbalanced loss terms (gradient pathology)

Wang & Perdikaris (2021) [^wang2021] used backprop gradient statistics to expose a universal phenomenon: **$\nabla_\theta\mathcal L_r$ is several orders of magnitude larger than $\nabla_\theta\mathcal L_b$.** Plain Adam follows the dominant gradient, the boundary loss is drowned out, and the network "satisfies the PDE wonderfully in the interior but has no idea what the boundary looks like."

![Gradient pathology and spectral bias.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/01-Physics-Informed-Neural-Networks/fig6_failure_modes.png)
*Figure 6, left: per-layer norms of $\nabla_\theta\mathcal L_r$ and $\nabla_\theta\mathcal L_b$ on a logarithmic scale, separated by 3–4 orders of magnitude — typical for a vanilla PINN. Right: the same network approximating a multi-scale signal $u=\sin(\pi x)+0.5\sin(8\pi x)+0.3\sin(20\pi x)$. The low-frequency mode is essentially perfect; the $20\pi$ mode has all but vanished — this is "spectral bias".*

**Fix 1: adaptive weights.** Reset every few hundred steps:

$$
\hat\lambda_b=\frac{\max_\theta|\nabla_\theta\mathcal L_r|}{\overline{|\nabla_\theta\mathcal L_b|}},\qquad
\lambda_b\leftarrow(1-\alpha)\lambda_b+\alpha\hat\lambda_b.
$$

This is Wang's *learning-rate annealing* recipe; a few lines of code turn an unconvergent PINN into a healthy one.

**Fix 2: hard constraints.** Bake the boundary directly into the architecture. For 1-D Dirichlet zero conditions, use

$$
u_\theta(x,t)=x(1-x)\,\tilde u_\theta(x,t)+B(x,t),
$$

where $B$ satisfies the boundary by construction and $\tilde u_\theta$ is unconstrained. Then $\mathcal L_b\equiv 0$ and there is nothing to balance.

**Fix 3: NTK balancing.** Wang–Yu–Perdikaris (2022) [^wang2022ntk] proved PINN training dynamics are governed by three Neural Tangent Kernels; weighting by the trace of each NTK is the principled choice.

### 4.2 Pathology B: spectral bias

Neural network training has a well-known bias: **low frequencies are learned first, high frequencies last** (Rahaman 2019; Tancik 2020). For PINNs the impact is especially severe because the PDE residual involves second derivatives, which amplify high-frequency error by $k^2$ — the worse the network is at high frequencies, the larger the residual, in a vicious circle.

**Fixes**:

- **Fourier features**: map $x$ first to $[\sin(2\pi Bx),\cos(2\pi Bx)]$ with a Gaussian random matrix $B$; this flattens the NTK spectrum.
- **Sine activations** (SIREN [^siren]): naturally distribute energy across the frequency domain, but require careful initialisation.

### 4.3 Pathology C: violation of causality

Time-dependent PDEs respect "the past determines the future". But PINNs sample $\Omega\times[0,T]$ *all at once*, asking the network to fit $t=T$ before $t<T$ has been learned correctly. Krishnapriyan et al. (2021) [^krish2021] coined this *failure mode* on the convection equation.

**Fix**: causal training (Wang 2024) weights the residual by time-respecting factors,

$$
w_n=\exp\bigl(-\varepsilon\sum_{k<n}\mathcal L_r(t_k)\bigr).
$$

Late-time residuals are admitted into the loss only after early-time residuals have decayed.

### 4.4 Convergence comparison

Combining the fixes on a Burgers experiment:

![Whether the PDE residual breaks the data bottleneck.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/01-Physics-Informed-Neural-Networks/fig4_convergence.png)
*Figure 4. With only 50 noisy labels, pure supervised training (red) is quickly trapped at a 4% noise plateau. Adding the PDE residual (blue) keeps the physics constraint pushing the error down to 0.3%. This is the core value PINNs add over plain regression.*

---

## 5 Experiment: Burgers and an inverse problem

### 5.1 Forward problem: the Burgers shock

Consider

$$
u_t+uu_x=\nu u_{xx},\quad x\in[-1,1],\ t\in[0,1],\quad
u(\pm 1,t)=0,\ u(x,0)=-\sin(\pi x),\ \nu=\frac{0.01}{\pi}.
$$

The reference solution is obtained via the Cole–Hopf transform $u=-2\nu(\ln\phi)_x$ which reduces Burgers to the heat equation (the script's `burgers_cole_hopf` does exactly this). With $\nu$ this small, a near-discontinuous shock forms near $x=0$ from $t\approx 0.4$ onward.

![PINN solving Burgers' equation: prediction vs. exact solution.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/01-Physics-Informed-Neural-Networks/fig3_burgers_pred_vs_exact.png)
*Figure 3. Left: the Cole–Hopf reference $u(x,t)$ as a colour map; the red/blue interface is the shock. Right three panels: slices at $t=0.25/0.5/0.75$ — the PINN (dashed blue) captures both the location and the width of the shock. Look closely and the vanilla PINN still has roughly 5% overshoot on either side of the shock; this is what residual adaptive refinement, causal training and Sobolev training are designed to squash further.*

**Practical recipe**:

1. An MLP of 8 hidden layers $\times$ 20 units with tanh activations is plenty — do not blindly stack depth.
2. Residual collocation $N_r\sim 10^4$. Use **residual adaptive refinement (RAR)**: every few hundred steps, evaluate $|\mathcal N[u_\theta]|$ on a candidate pool and add the top-$k$ points to the training set.
3. Run Adam at 1e-3 for 20k steps to settle, then switch to L-BFGS for 5k more to polish.
4. **Normalise.** Map $(x,t)$ to $[-1,1]$ — otherwise the NTK is biased.

### 5.2 Inverse problem: parameter discovery

Forward problems are unremarkable; inverse problems are where PINNs shine. Append $\mathcal L_d$ to fit sparse observations and treat the unknown PDE parameter $\nu$ as a learnable scalar that joins the gradient descent.

![Recovering the diffusivity from sparse noisy data.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/01-Physics-Informed-Neural-Networks/fig5_inverse_problem.png)
*Figure 5. Left: only 30 noisy observations at $t=0.15$ (red dots). Right: treating $\nu$ as a learnable parameter alongside $\theta$, the trajectory converges from initial guess 0.45 to the true value 0.10 with relative error below 1%. This unification of forward and inverse problems within a single loss function is something FEM essentially cannot do.*

**Why is PINN so good at inverse problems?** Classical inverse problems require nested optimisation — outer loop on parameters, inner loop on the PDE solver; every parameter change triggers a full forward solve. PINNs put "satisfies the PDE" *into* the loss, so parameter and $\theta$ live on the same gradient — no outer loop. The price is uncertainty quantification: ensembles or Bayesian PINNs, not a textbook MCMC + FEM stack.

---

## 6 Failure modes and limits

PINNs are not silver bullets. Common industrial pitfalls:

| failure mode | cause | state of the art |
|---|---|---|
| high-frequency / multi-scale (turbulence) | spectral bias + 2nd-derivative amplification | partially mitigated by Fourier features and cPINN domain decomposition |
| near-discontinuities (shocks, phase transitions) | NN bias toward continuous functions | conservative PINN, weak-form PINN |
| long-time integration | causality violation, error accumulation | causal training, time partitioning |
| ill-conditioning (non-unique weak solutions) | non-convex loss landscape | priors, ensembles |
| coupled multi-physics | scale mismatch across loss terms | adaptive weighting, multi-task learning |
| high dimensions ($d>10$) | residual sampling curse | progressive sampling, quasi-MC |

A practitioner's rule of thumb: **complex geometry, high dimensions, parameter inversion required → PINN; rigorous accuracy guarantees, regular geometry, fixed parameters → FEM / spectral; many online queries of the same parametric PDE family → neural operator.**

---

## 7 PINNs on the SciML map

![PINN vs. FEM vs. neural operators: capability radar and cost-accuracy trade-off.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/01-Physics-Informed-Neural-Networks/fig7_pinn_vs_fem_vs_no.png)
*Figure 7. Left: a six-axis radar (mesh-free, high-d, complex geometry, single-solve speed, cross-PDE generalisation, accuracy guarantees). Right: a "solve time vs. error" scatter for the same PDE under different methods. FEM has the highest attainable accuracy at the cost of being purpose-built for one geometry; PINNs are versatile but expensive per solve; neural operators (DeepONet/FNO) cost ~1 second per inference once trained, making them the right choice for parametric PDEs — but their training data has to come from FEM or PINN solves.*

**Selection mnemonic:**

- **FEM** — classical, accuracy-guaranteed, regular geometry. Industrial-scale simulation still rests on FEM.
- **PINN** — a Swiss-army knife, especially for **inverse problems** and **prior-fusion** research questions.
- **Neural operators** (chapter 2) — for "same PDE family, varying input function, repeatedly queried" applications: weather, semiconductor simulation, financial pricing.

PINNs do not aim to replace FEM. Their real role is to make **prior physics a first-class citizen of deep-learning model design.** That theme will run through every subsequent chapter.

---

## 8 Handing off to the next chapters

Reading PINN as "constraint embedded in the loss function" makes the rest of the series fall into place:

- **Chapter 2 — Neural operators.** From "learn one solution" to "learn the solution operator $\mathcal G:f\mapsto u$".
- **Chapter 3 — Variational principles.** Replace the residual loss with an energy functional — more stable, closer to the FEM theory.
- **Chapter 4 — Variational inference.** Push the Fokker–Planck equation into the loss for consistent probabilistic inference.
- **Chapter 5 — Symplectic geometry.** Encode Hamiltonian conservation directly in the architecture — the extreme case of hard constraints.
- **Chapter 6 — Neural ODE / CNF.** Replace "residual = 0" with "divergence of the flow = 0".
- **Chapter 7 — Diffusion models / score matching.** Run PINN backwards: given data, recover the stochastic process whose Fokker–Planck equation it satisfies.
- **Chapter 8 — Reaction-diffusion + GNN.** Swap the MLP for a graph network to handle mesh-shaped geometry.

After this chapter, you should be able to give the one-sentence answer:

> **A PINN is a mesh-free PDE solver that uses neural networks as universal basis functions, treats the PDE residual as a loss function, and uses automatic differentiation to evaluate high-order derivatives to machine precision; its bottleneck is training dynamics — the gradient pathology and spectral bias revealed by NTK analysis — not expressive power.**

---

## ✅ Checkpoint

1. Write down what each of the three terms in the PINN loss represents.
2. Explain why NTK scale mismatches cause boundary conditions to be ignored during training.
3. Name at least two engineering fixes for spectral bias.
4. Given an inverse problem, sketch the code that puts the unknown parameter into the computation graph.
5. When should you choose PINN over FEM? Over a neural operator?

---

## References

[^raissi2019]: M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations.* J. Comput. Phys., 378:686–707, 2019. [doi:10.1016/j.jcp.2018.10.045](https://doi.org/10.1016/j.jcp.2018.10.045)
[^lagaris1998]: I. E. Lagaris, A. Likas, D. I. Fotiadis. *Artificial Neural Networks for Solving Ordinary and Partial Differential Equations.* IEEE TNN, 9(5):987–1000, 1998.
[^deepritz]: W. E, B. Yu. *The Deep Ritz Method.* Commun. Math. Stat., 6(1):1–12, 2018. [arXiv:1710.00211](https://arxiv.org/abs/1710.00211)
[^shin2020]: Y. Shin, J. Darbon, G. E. Karniadakis. *On the Convergence of Physics-Informed Neural Networks for Linear Second-Order Elliptic and Parabolic Type PDEs.* Commun. Comput. Phys., 28(5):2042–2074, 2020. [arXiv:2004.01806](https://arxiv.org/abs/2004.01806)
[^wang2021]: S. Wang, Y. Teng, P. Perdikaris. *Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks.* SIAM J. Sci. Comput., 43(5):A3055–A3081, 2021. [arXiv:2001.04536](https://arxiv.org/abs/2001.04536)
[^wang2022ntk]: S. Wang, X. Yu, P. Perdikaris. *When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective.* J. Comput. Phys., 449:110768, 2022. [arXiv:2007.14527](https://arxiv.org/abs/2007.14527)
[^krish2021]: A. Krishnapriyan et al. *Characterizing Possible Failure Modes in Physics-Informed Neural Networks.* NeurIPS 2021. [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)
[^xpinn]: A. D. Jagtap, G. E. Karniadakis. *Extended Physics-Informed Neural Networks (XPINNs).* Commun. Comput. Phys., 28(5):2002–2041, 2020. [arXiv:2104.10013](https://arxiv.org/abs/2104.10013)
[^deepxde]: L. Lu, X. Meng, Z. Mao, G. E. Karniadakis. *DeepXDE: A Deep Learning Library for Solving Differential Equations.* SIAM Rev., 63(1):208–228, 2021. [arXiv:1907.04502](https://arxiv.org/abs/1907.04502)
[^siren]: V. Sitzmann et al. *Implicit Neural Representations with Periodic Activation Functions (SIREN).* NeurIPS 2020. [arXiv:2006.09661](https://arxiv.org/abs/2006.09661)
[^pikan]: Z. Liu et al. *From PINNs to PIKANs: Physics-Informed Kolmogorov-Arnold Networks.* arXiv:2410.13228, 2024. [arXiv:2410.13228](https://arxiv.org/abs/2410.13228)
[^cuomo2022]: S. Cuomo et al. *Scientific Machine Learning Through Physics-Informed Neural Networks: Where We Are and What's Next.* J. Sci. Comput., 92(3):88, 2022. [arXiv:2201.05624](https://arxiv.org/abs/2201.05624)

---

## Series navigation

| part | topic |
|------|------|
| **1** | **Physics-Informed Neural Networks (this article)** |
| 2 | Neural Operator Theory |
| 3 | Variational Principles and Optimisation |
| 4 | Variational Inference and the Fokker–Planck Equation |
| 5 | Symplectic Geometry and Structure-Preserving Networks |
| 6 | Continuous Normalising Flows and Neural ODE |
| 7 | Diffusion Models and Score Matching |
| 8 | Reaction-Diffusion Systems and GNN |
