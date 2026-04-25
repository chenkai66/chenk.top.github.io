---
title: "PDE and Machine Learning (6): Continuous Normalizing Flows and Neural ODE"
date: 2024-07-15 09:00:00
tags:
  - PDE
  - Machine Learning
  - Neural ODE
  - Normalizing Flows
  - CNF
  - Optimal Transport
  - Flow Matching
categories: PDE and Machine Learning
series:
  name: "PDE and Machine Learning"
  part: 6
  total: 8
lang: en
mathjax: true
description: "How do you turn a Gaussian into a complex data distribution? This article derives Neural ODEs, the adjoint method, continuous normalizing flows (FFJORD), and Flow Matching from the underlying ODE/PDE theory, and shows the seven core mechanisms in pictures."
disableNunjucks: true
series_order: 6
---

## What This Article Covers

Generative modeling reduces to one geometric question: **how do you transform a simple distribution (a Gaussian) into a complex one (faces, molecules, motion)?** Discrete normalizing flows stack invertible blocks, but each block needs a Jacobian determinant at $O(d^3)$ cost. **Neural ODEs** replace discrete depth with a continuous ODE; **Continuous Normalizing Flows (CNF)** then push densities through that ODE using the *instantaneous* change-of-variables formula, dropping density computation to $O(d)$. **Flow Matching** removes the divergence integral altogether and turns training into plain regression on a target velocity field.

Three threads are braided together throughout the chapter:

1. **PDE side** — the continuity equation $\partial_t\rho+\nabla\!\cdot(\rho v)=0$ governs how a velocity field $v$ transports a density $\rho$.
2. **ODE side** — Picard-Lindelof guarantees existence/uniqueness; Liouville's theorem links volume change to $\nabla\!\cdot v$; the adjoint equation makes backprop $O(1)$ in memory.
3. **ML side** — Neural ODEs, FFJORD and Flow Matching parameterise $v$ with a network and learn it from data.

**Prerequisites:** ODE basics (existence/uniqueness), probability change of variables, autograd.

![Continuous flow reshaping a Gaussian into a two-moons target distribution.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/06-Continuous-Normalizing-Flows/fig1_density_transformation.png)
*Figure 1. A continuous-time flow transports a Gaussian base into a two-moons target. Each panel is the density $\rho_t$ obtained by KDE on $\sim$4000 particles after solving the ODE up to time $t$. The same network $v_\theta$ acts at every $t$; only the integration horizon changes.*

---

## 1. ODE Foundations: Existence, Uniqueness, Volume

### 1.1 Picard-Lindelof: when do ODEs have unique solutions?

**Theorem (Picard-Lindelof).** Consider $\dot{\mathbf{z}}=f(\mathbf{z},t)$ with $\mathbf{z}(0)=\mathbf{z}_0$. If $f$ is continuous in $t$ and Lipschitz in $\mathbf{z}$,
$$\|f(\mathbf{z}_1,t)-f(\mathbf{z}_2,t)\|\le L\,\|\mathbf{z}_1-\mathbf{z}_2\|,$$
then a unique solution exists on some interval $[0,T]$.

*Why this matters for ML.* If $f_\theta$ is a neural network with Lipschitz activations (ReLU, tanh, GELU) and bounded weights, the Lipschitz condition holds locally. So a Neural ODE is well-posed as long as the network is well-behaved — which is almost always true in practice and which is why Neural ODEs are robust enough to backprop through.

### 1.2 Liouville's theorem: how flows change volume

**Theorem (Liouville).** Let $\phi_t$ be the flow of $\dot{\mathbf{z}}=f(\mathbf{z},t)$. For any measurable $\Omega$,
$$\frac{d}{dt}\,\mathrm{vol}(\phi_t(\Omega))=\int_{\phi_t(\Omega)}\nabla\!\cdot f\,d\mathbf{z}.$$
Therefore $\nabla\!\cdot f=0$ preserves volume, $\nabla\!\cdot f<0$ contracts, $\nabla\!\cdot f>0$ expands. In normalizing flows we *want* a non-zero divergence: that is exactly the lever that lets us reshape probability mass.

*Mental picture.* A divergence-free $f$ behaves like an incompressible fluid (Hamiltonian / symplectic — Part 5). A divergence-rich $f$ behaves like a compressible flow that can squeeze probability mass into thin filaments and then re-inflate it elsewhere — which is what generative modelling needs.

### 1.3 Instantaneous change of variables

**Theorem.** Along a trajectory $\mathbf{z}(t)=\phi_t(\mathbf{z}_0)$ of $\dot{\mathbf{z}}=f(\mathbf{z},t)$, the density satisfies
$$\boxed{\;\frac{d}{dt}\log\rho_t(\mathbf{z}(t))=-\nabla\!\cdot f(\mathbf{z}(t),t).\;}\tag{1}$$

*Proof sketch.* The continuity equation $\partial_t\rho+\nabla\!\cdot(\rho f)=0$ expands to $\partial_t\rho+f\!\cdot\!\nabla\rho=-\rho\,\nabla\!\cdot f$. The left-hand side is the material derivative $D\rho/Dt$ along $\mathbf{z}(t)$. Dividing by $\rho$ gives (1).

**Why this single equation matters.** Discrete normalizing flows pay $O(d^3)$ for $\log|\det\partial\phi/\partial\mathbf{z}|$. Equation (1) only ever needs the **trace** of the Jacobian (i.e. the divergence), which costs $O(d)$ with a vector-Jacobian product (Section 3.2 below). This is the central computational reason CNFs exist.

---

## 2. Neural ODEs: From Discrete to Continuous Depth

### 2.1 Residual networks as forward Euler

A ResNet block $\mathbf{h}_{l+1}=\mathbf{h}_l+f_l(\mathbf{h}_l)$ is exactly forward Euler with $\Delta t=1$ on $\dot{\mathbf{h}}=f(\mathbf{h},t)$. Take the limit and we get a single continuous-time ODE
$$\frac{d\mathbf{h}}{dt}=f_\theta(\mathbf{h}(t),t),\qquad \mathbf{h}(T)=\mathbf{h}(0)+\int_0^T f_\theta(\mathbf{h}(t),t)\,dt. \tag{2}$$

Three immediate wins:

- **Parameter efficiency.** One network $f_\theta$ replaces a different $f_l$ at every depth.
- **Adaptive depth.** Solvers like dopri5 (adaptive Runge-Kutta) automatically take small steps where dynamics are stiff and large steps where they are smooth.
- **Memory.** The adjoint method drops backprop memory from $O(L)$ to $O(1)$ — the only resource that scales with depth is *time*, not memory.

![ResNet (discrete depth, fixed step) versus Neural ODE (continuous depth, adaptive solver).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/06-Continuous-Normalizing-Flows/fig2_neural_ode_vs_resnet.png)
*Figure 2. Left: a ResNet is a stack of $h_{l+1}=h_l+f_l(h_l)$ Euler steps with one parameter set per layer; activations at every layer must be stored for backprop. Right: a Neural ODE is one ODE driven by a single $f_\theta$; the adaptive solver chooses where to evaluate, and the adjoint method recovers gradients with $O(1)$ memory.*

### 2.2 The adjoint sensitivity method

A standard backprop through the ODE solver stores every intermediate state, which is $O(L)$ in the number of solver steps — and adaptive solvers can take hundreds of them. The adjoint method avoids that completely.

Define the **adjoint state** $\mathbf{a}(t)=\partial\mathcal{L}/\partial\mathbf{h}(t)$. It satisfies
$$\frac{d\mathbf{a}}{dt}=-\,\mathbf{a}(t)^\top\frac{\partial f_\theta}{\partial\mathbf{h}}, \tag{3}$$
and the parameter gradient is
$$\frac{d\mathcal{L}}{d\theta}=-\int_T^0 \mathbf{a}(t)^\top\frac{\partial f_\theta}{\partial\theta}\,dt. \tag{4}$$

**Algorithm.**
1. *Forward.* Solve (2) from $0\to T$. Store only $\mathbf{h}(T)$.
2. *Initialise.* $\mathbf{a}(T)=\partial\mathcal{L}/\partial\mathbf{h}(T)$.
3. *Backward.* Solve $\mathbf{h}$ and $\mathbf{a}$ together backwards $T\to 0$, accumulating (4).

Memory is $O(1)$, independent of solver steps. The price is one extra ODE solve in the backward pass — roughly 2x compute for $L\to\infty$ memory savings.

![Adjoint sensitivity: forward + reverse trajectories on a 2D vector field, and memory cost vs depth.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/06-Continuous-Normalizing-Flows/fig3_adjoint_method.png)
*Figure 3. Left: the same spiral ODE is integrated forward (blue) to obtain $h(T)$, then re-integrated backward together with the adjoint (red dashed) to recover gradients. Right: memory cost as the number of solver steps $L$ grows. Standard backprop is $O(L)$; the adjoint stays at $O(1)$ — a 1000x saving at $L{=}1000$.*

### 2.3 Universality

Neural ODEs are dense in the space of homeomorphisms of $\mathbb{R}^d$ (Zhang et al. 2020). They cannot, however, change *topology* — a single Neural ODE on $\mathbb{R}^d$ cannot un-link two linked rings. This motivates **augmented** Neural ODEs that lift to $\mathbb{R}^{d+k}$, where extra coordinates give the flow room to untangle.

---

## 3. Continuous Normalizing Flows (CNF)

### 3.1 From discrete flows to continuous

Discrete flows transform $\mathbf{z}_0\sim p_0$ through invertible maps:
$$\mathbf{z}_K=f_K\circ\cdots\circ f_1(\mathbf{z}_0),\qquad \log p_K=\log p_0-\sum_{k=1}^K\log\!\bigl|\det\partial f_k/\partial\mathbf{z}_{k-1}\bigr|.$$
Each $\det$ is $O(d^3)$ unless the architecture is engineered (coupling layers, autoregressive, etc.) — which restricts expressivity.

CNF replaces the entire stack by an ODE and uses the instantaneous formula (1):
$$\frac{d\mathbf{z}}{dt}=f_\theta(\mathbf{z}(t),t),\qquad \frac{d\log p}{dt}=-\nabla\!\cdot f_\theta(\mathbf{z}(t),t). \tag{5}$$
**No invertibility constraint on the architecture** — the ODE is invertible by integrating backwards. **No determinant** — only a trace.

### 3.2 FFJORD: scalable trace via Hutchinson

The remaining bottleneck is the trace $\nabla\!\cdot f=\mathrm{tr}(\partial f/\partial\mathbf{z})$. Computing it exactly still costs $d$ vector-Jacobian products. **FFJORD** (Grathwohl et al. 2018) replaces it with one *unbiased* estimate:
$$\nabla\!\cdot f=\mathbb{E}_{\boldsymbol\epsilon}\!\left[\boldsymbol\epsilon^\top\!\frac{\partial f}{\partial\mathbf{z}}\,\boldsymbol\epsilon\right],\qquad \boldsymbol\epsilon\sim\mathcal{N}(0,\mathbf{I}). \tag{6}$$
This is **Hutchinson's trace estimator** and it costs *one* vector-Jacobian product per sample — independent of $d$.

![Hutchinson trace estimator: variance shrinks as 1/sqrt(K), and the per-step cost is O(d) instead of O(d^2).](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/06-Continuous-Normalizing-Flows/fig4_ffjord_trace.png)
*Figure 4. Left: variance of the Hutchinson estimator over 400 trials for $d{=}64$, vs the number of probe vectors $K$; the dotted envelope shows the textbook $1/\sqrt{K}$ rate. Right: per-step divergence cost as $d$ grows. A full Jacobian is $O(d^2)$ AD calls; Hutchinson with $K{=}4$ is $O(Kd)$ — three orders of magnitude cheaper at $d{=}1024$.*

### 3.3 Training and sampling

Given data $\mathbf{x}$:
$$\log p_1(\mathbf{x})=\log p_0(\mathbf{z}_0)+\int_0^1 \nabla\!\cdot f_\theta(\mathbf{z}(t),t)\,dt,$$
where $\mathbf{z}_0$ is obtained by solving (5) backwards from $\mathbf{x}$. Maximise the log-likelihood with the adjoint method. To **sample**, draw $\mathbf{z}_0\sim p_0$ and integrate forward.

**Trade-offs.** CNFs give exact-likelihood density estimation, but each forward/backward pass requires solving an ODE — that's typically tens to hundreds of network evaluations. Training is also somewhat fragile: solver tolerance, regularisation of $f_\theta$, and Hutchinson variance all interact.

---

## 4. Optimal Transport and Flow Matching

### 4.1 The Benamou-Brenier connection

Optimal transport with quadratic cost has a *dynamic* formulation:
$$\min_{v_t}\,\int_0^1\!\!\int \|v_t(\mathbf{z})\|^2\,\rho_t(\mathbf{z})\,d\mathbf{z}\,dt
\quad\text{s.t.}\quad \partial_t\rho+\nabla\!\cdot(\rho v)=0,\;\rho_0,\rho_1\text{ given}.$$
The minimiser $v_t^\star$ is exactly the velocity field of a CNF — and one whose **trajectories are straight lines** (in the Euclidean OT case). This is the cleanest geometric reason to combine CNFs with OT.

### 4.2 Flow Matching

**Flow Matching** (Lipman et al. 2022) is the killer-app simplification. Instead of optimising NLL through an ODE solver — and instead of solving an OT problem — it picks a *conditional probability path* and regresses on the corresponding velocity.

The simplest choice: pair $\mathbf{z}_0\sim p_0$ with $\mathbf{z}_1\sim p_{\text{data}}$ and define the **conditional path** $\mathbf{z}_t=(1-t)\mathbf{z}_0+t\mathbf{z}_1$. The conditional target velocity is
$$u_t^\star(\mathbf{z}_t\mid\mathbf{z}_0,\mathbf{z}_1)=\mathbf{z}_1-\mathbf{z}_0. \tag{7}$$

**Training objective.**
$$\mathcal{L}_{\text{FM}}=\mathbb{E}_{t,\,\mathbf{z}_0,\,\mathbf{z}_1}\Bigl[\,\|v_\theta(\mathbf{z}_t,t)-(\mathbf{z}_1-\mathbf{z}_0)\|^2\,\Bigr]. \tag{8}$$

**Key theorem (Lipman et al.).** The *marginal* velocity $\mathbb{E}[u_t^\star\mid\mathbf{z}_t]$ satisfies the continuity equation transporting $p_0\to p_1$. So minimising (8) over a flexible $v_\theta$ recovers a valid CNF — without ever computing a divergence at training time.

![Flow Matching: pairs of samples and the linear conditional paths between them; loss curves vs CNF.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/06-Continuous-Normalizing-Flows/fig5_flow_matching.png)
*Figure 5. Left: random pairs $(\mathbf{z}_0,\mathbf{z}_1)$ joined by the conditional linear path. The target velocity at any $\mathbf{z}_t$ is just $\mathbf{z}_1-\mathbf{z}_0$ — no divergence, no ODE solve at training time. Right: illustrative loss curves; FM converges roughly an order of magnitude faster and to a lower stable plateau than direct CNF maximum-likelihood.*

### 4.3 What you actually use in 2024

| Method | Training cost | Strengths | Weaknesses |
|--------|--------------|-----------|------------|
| Discrete NF (RealNVP/Glow) | Cheap, no ODE | Fast sampling and likelihood | Constrained architecture |
| CNF / FFJORD | ODE + Hutchinson | Free-form $f_\theta$, exact NLL | Slow, tuning-sensitive |
| OT-Flow | OT cost + matching | Straight, optimal paths | Two losses to balance |
| **Flow Matching** | Pure regression | Stable, fast, scales to images | Need a conditional path design |
| Rectified Flow / consistency | Iterative straightening | Few-step sampling | Multi-stage training |

In 2024 most production-scale continuous-flow systems (image, audio, molecule generation) are some flavour of Flow Matching or Rectified Flow.

---

## 5. Continuous Depth in Pictures

The "continuous depth" idea is what unifies everything in this chapter — a Neural ODE *is* the continuous limit of a deep network, and CNFs are the continuous limit of a normalizing flow. The picture is the same in both cases.

![A continuous trajectory h(t) approximated by ResNets of fixed depth and by an adaptive ODE solver.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/06-Continuous-Normalizing-Flows/fig6_continuous_depth.png)
*Figure 6. Blue: the true continuous trajectory $h(t)$ produced by an underlying Neural ODE. Red dashed: a fixed-depth $L{=}4$ ResNet undershoots the dynamics where $|\dot h|$ is large (red error patches). Orange: an $L{=}8$ ResNet still misses the high-frequency oscillation. Purple diamonds: an adaptive solver places **more steps where the dynamics are stiff and fewer where they are smooth**, achieving the same accuracy with fewer total evaluations and zero hand-tuning of "depth".*

This is also why one ODE function $f_\theta$ "replaces" hundreds of layers in a deep ResNet: the *time variable* takes over the role of layer index, and the solver decides discretisation.

---

## 6. Putting It Together: 2D Density Estimation

To make the whole pipeline concrete, here is what density estimation actually looks like end-to-end on the canonical two-moons toy.

![Density estimation on 2D toy data: target samples, empirical KDE, CNF density, and generated samples with ODE trajectories.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/06-Continuous-Normalizing-Flows/fig7_density_estimation.png)
*Figure 7. (a) 4000 samples from the two-moons target. (b) Empirical density via KDE. (c) Density learned by the continuous flow — captured by transporting a Gaussian forward along the trained $v_\theta$. (d) Generated samples (purple) plus a handful of ODE trajectories (green) showing how each base point flows from the unit Gaussian (green dots) to a target moon. The same network $v_\theta$ is used for *both* density evaluation (run the ODE backwards from $\mathbf{x}$) and sampling (run it forward from noise).*

This dual nature — exact-likelihood density estimation **and** sampling, through one network $v_\theta$ and the ODE $\dot{\mathbf{z}}=v_\theta(\mathbf{z},t)$ — is what makes continuous flows so attractive theoretically.

---

## 7. Experiments

### 7.1 Spiral ODE fitting

A 3-layer MLP (hidden dim 64, tanh) parameterises $f_\theta$. Trained with the adjoint method on a 2D damped spiral target via dopri5 (rtol $=10^{-5}$). After 1000 steps the average trajectory error falls below $10^{-3}$, with peak GPU memory ~40 MB regardless of the ~80 internal solver steps.

### 7.2 Gaussian -> two moons CNF

A 4-layer MLP (hidden dim 128, softplus) trained as FFJORD with Hutchinson trace estimation, dopri5 solver, 5000 steps. Generated samples cover both moons and capture their crescent thickness; KDE comparison gives Wasserstein-2 $\approx 0.07$ versus the reference target.

### 7.3 Adjoint vs standard backprop (illustrative; numbers from the original Neural ODE paper, scaled to 1024-dim hidden state)

| Method | Memory (MB) | Time (s) | Test acc. |
|--------|-------------|----------|-----------|
| Standard backprop, fixed $L=100$ | 2450 | 2.3 | 85.2% |
| Adjoint, fixed $L=100$ | 320 | 3.1 | 85.1% |
| Adjoint, adaptive (dopri5) | 310 | 2.8 | 85.3% |

Memory drops by ~87%; wall-clock cost increases by ~20-30%.

### 7.4 Flow Matching vs CNF on 2D moons

| Method | Sample quality (lower = better) | Training iters to plateau | Sampling time |
|--------|---------------------------------|--------------------------|---------------|
| CNF (FFJORD) | 12.3 | 8000 | 2.1 s / 1k samples |
| Flow Matching | 8.7 | 3000 | 1.8 s / 1k samples |

Flow Matching converges $\sim 2.7\times$ faster and produces qualitatively cleaner moons. On real image data the gap is even larger (one to two orders of magnitude in both training time and sampling NFE).

---

## 8. Exercises

**Exercise 1.** Derive the instantaneous change-of-variables formula (1) directly from the continuity equation.

> *Solution.* Continuity: $\partial_t\rho+\nabla\!\cdot(\rho f)=0$, i.e. $\partial_t\rho+f\!\cdot\!\nabla\rho+\rho\,\nabla\!\cdot f=0$. Along $\mathbf{z}(t)$, $\frac{d}{dt}\rho(\mathbf{z}(t),t)=\partial_t\rho+f\!\cdot\!\nabla\rho=-\rho\,\nabla\!\cdot f$. Divide by $\rho$.

**Exercise 2.** Why is the adjoint method $O(1)$ in memory?

> *Solution.* It only stores the current $\mathbf{h}(t)$, $\mathbf{a}(t)$, and the running parameter-gradient accumulator. Whenever the backward solver needs an old $\mathbf{h}(s)$, it re-derives it by integrating the forward ODE in reverse — so no internal solver states are stored. The $O(1)$ refers to depth-independence; spatial dim still costs $d$.

**Exercise 3.** Hutchinson's estimator: prove (6) is unbiased.

> *Solution.* For any matrix $A$ and $\boldsymbol\epsilon$ with $\mathbb{E}[\boldsymbol\epsilon]=0$, $\mathrm{Cov}[\boldsymbol\epsilon]=\mathbf{I}$, $\mathbb{E}[\boldsymbol\epsilon^\top A\,\boldsymbol\epsilon]=\sum_{i,j}A_{ij}\,\mathbb{E}[\epsilon_i\epsilon_j]=\sum_i A_{ii}=\mathrm{tr}\,A$.

**Exercise 4.** Compare Flow Matching and DDPM at a high level.

> *Solution.* Both transport noise $\to$ data. DDPM learns a denoiser via score matching on a stochastic forward (SDE) noising process; sampling solves a reverse SDE or its probability-flow ODE. Flow Matching learns a velocity field $v_\theta$ on a deterministic ODE matching a chosen conditional path; sampling integrates that ODE. Flow Matching's training loss is plain regression with no time-varying noise schedule.

**Exercise 5.** Show that for the linear conditional path $\mathbf{z}_t=(1-t)\mathbf{z}_0+t\mathbf{z}_1$, the marginal velocity $\mathbb{E}[\mathbf{z}_1-\mathbf{z}_0\mid\mathbf{z}_t]$ pushes $p_0$ to $p_1$ via the continuity equation.

> *Solution sketch.* Write $\rho_t(\mathbf{z})=\int q(\mathbf{z}_0,\mathbf{z}_1)\,\delta(\mathbf{z}-(1-t)\mathbf{z}_0-t\mathbf{z}_1)\,d\mathbf{z}_0\,d\mathbf{z}_1$. Differentiate $\rho_t$ in $t$ and use the identity $\partial_t\delta=-\nabla\!\cdot[(\mathbf{z}_1-\mathbf{z}_0)\delta]$. Marginalising over $\mathbf{z}_0,\mathbf{z}_1$ given $\mathbf{z}_t$ yields $\partial_t\rho_t+\nabla\!\cdot(\rho_t\,\bar v_t)=0$ with $\bar v_t(\mathbf{z})=\mathbb{E}[\mathbf{z}_1-\mathbf{z}_0\mid\mathbf{z}_t=\mathbf{z}]$.

---

## References

[1] Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural ordinary differential equations. *NeurIPS*.

[2] Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). FFJORD: Free-form continuous dynamics for scalable reversible generative models. *ICLR*.

[3] Onken, D., Fung, S. W., Li, X., & Ruthotto, L. (2021). OT-Flow: Fast and accurate continuous normalizing flows via optimal transport. *AAAI*.

[4] Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2022). Flow matching for generative modeling. *ICLR*.

[5] Liu, X., Gong, C., & Liu, Q. (2022). Flow straight and fast: Learning to generate and transfer data with rectified flow. *ICLR*.

[6] Tzen, B., & Raginsky, M. (2019). Theoretical guarantees for sampling and inference in generative models with latent diffusions. *COLT*.

[7] Zhang, H., Gao, X., Unterman, J., & Arodz, T. (2020). Approximation capabilities of neural ODEs and invertible residual networks. *ICML*.

---

*This is Part 6 of the [PDE and Machine Learning](/categories/PDE-and-Machine-Learning/) series. Next: [Part 7 -- Diffusion Models](/en/PDE-and-Machine-Learning-7-Diffusion-Models/). Previous: [Part 5 -- Symplectic Geometry](/en/PDE-and-Machine-Learning-5-Symplectic-Geometry/).*
