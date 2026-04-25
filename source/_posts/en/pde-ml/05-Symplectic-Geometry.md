---
title: "PDE and Machine Learning (5): Symplectic Geometry and Structure-Preserving Networks"
date: 2024-06-30 09:00:00
tags:
  - PDE
  - Machine Learning
  - Symplectic Geometry
  - Hamiltonian
  - Structure-Preserving
  - Neural ODE
categories: PDE and Machine Learning
series:
  name: "PDE and Machine Learning"
  part: 5
  total: 8
lang: en
mathjax: true
description: "Standard neural networks violate conservation laws. This article derives Hamiltonian mechanics, symplectic integrators, HNNs, LNNs, and SympNets from the geometry of phase space."
disableNunjucks: true
series_order: 5
---

## What this article covers

Train an unconstrained neural network on pendulum data and ask it to extrapolate. After a few seconds of integration the prediction is fine; after a minute the pendulum has either crept to a halt or, more often, accelerated to escape velocity. Energy was supposed to be conserved, but the network has no idea what energy is. The bug is not in the data, the optimizer, or the depth of the network. **The bug is in the architecture.** A standard MLP can represent any vector field, including unphysical ones, and a tiny systematic bias in that vector field is amplified into macroscopic energy drift over a long rollout.

The fix is not to train harder. The fix is to **build the conservation law into the model**. This is what Hamiltonian Neural Networks (HNNs), Lagrangian Neural Networks (LNNs), and Symplectic Networks (SympNets) do. They learn a *scalar potential* (energy or action) and recover the dynamics by differentiating it; or they parameterise the discrete-time map directly out of symplectic building blocks. Either way, the resulting predictor is a Hamiltonian system by construction, and the long-term behaviour is qualitatively right whether the model is trained to one decimal place or four.

**You will learn:**

1. Why phase space and the symplectic 2-form are the right home for classical mechanics
2. Hamilton's equations as a single tensor identity $\dot{\mathbf{z}} = J\,\nabla H$
3. Liouville's theorem, Poincare's recurrence, and what "preserving structure" really means numerically
4. Why every Runge-Kutta method drifts in energy and why symplectic integrators (Verlet, leapfrog, symplectic Euler) do not
5. HNN, LNN, and SympNet: three answers to "how do I bake symplecticity into a neural network?"
6. A clean reproduction of the Greydanus et al. pendulum result with controlled experiments

**Prerequisites:** vector calculus (chain rule, gradients), an ODE solver from any numerics class, and one prior exposure to neural ODEs (covered in [Part 6](/en/PDE-and-Machine-Learning-6-Continuous-Normalizing-Flows/)).

---

![Phase-space orbits of the pendulum on the left, and a transported phase-space blob on the right showing Liouville's volume preservation.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/05-Symplectic-Geometry/fig1_hamiltonian_flow.png)
*Figure 1. Hamiltonian flow in phase space. Left: orbits of the simple pendulum are the level sets of $H(q,p) = \tfrac{1}{2}p^2 + (1-\cos q)$; arrows show the direction of the flow $J\nabla H$. Right: a small blob of initial conditions is carried by the flow of the harmonic oscillator. The shape twists but the **area is exactly preserved** -- this is Liouville's theorem, the geometric fingerprint of Hamiltonian mechanics.*

## 1. Hamiltonian mechanics: physics in phase space

### 1.1 From Newton to Hamilton

Newton's $F = ma$ is a second-order ODE in position. Hamilton's reformulation introduces the **conjugate momentum** $\mathbf{p} = \partial L / \partial \dot{\mathbf{q}}$ and writes the dynamics as a *first-order* system on the **phase space** $(\mathbf{q}, \mathbf{p}) \in \mathbb{R}^{2n}$:

$$
H(\mathbf{q}, \mathbf{p}) \;=\; \underbrace{\tfrac{1}{2}\mathbf{p}^\top M^{-1} \mathbf{p}}_{\text{kinetic}} \;+\; \underbrace{V(\mathbf{q})}_{\text{potential}}\,. \tag{1}
$$

The price you pay (doubling the dimension) is repaid many times over: Hamiltonian dynamics has a *geometric* structure that Newtonian dynamics hides.

### 1.2 Hamilton's equations as a single tensor identity

The equations of motion are

$$
\dot{\mathbf{q}} \;=\; \frac{\partial H}{\partial \mathbf{p}}, \qquad \dot{\mathbf{p}} \;=\; -\frac{\partial H}{\partial \mathbf{q}}. \tag{2}
$$

Letting $\mathbf{z} = (\mathbf{q}, \mathbf{p})^\top$ and packaging the antisymmetry into the **symplectic matrix**

$$
J \;=\; \begin{pmatrix} \mathbf{0} & I_n \\ -I_n & \mathbf{0} \end{pmatrix}, \qquad J^\top = -J, \qquad J^2 = -I_{2n},
$$

equation (2) collapses to

$$
\boxed{\;\dot{\mathbf{z}} \;=\; J\,\nabla H(\mathbf{z}).\;} \tag{3}
$$

This single line is the entire content of classical mechanics for conservative systems. Every theorem we prove from here on -- energy conservation, Liouville's theorem, Noether's theorem, KAM stability -- follows from the algebraic structure of $J$.

### 1.3 Energy conservation is automatic

Differentiate $H$ along a trajectory:

$$
\frac{dH}{dt} \;=\; (\nabla H)^\top \dot{\mathbf{z}} \;=\; (\nabla H)^\top J\,(\nabla H) \;=\; 0,
$$

because $J^\top = -J$ implies $\mathbf{v}^\top J \mathbf{v} = 0$ for every $\mathbf{v}$. **Energy conservation is a one-line corollary of antisymmetry.** This is the punchline that HNNs will exploit: write the dynamics as $\dot{\mathbf{z}} = J\,\nabla H_\theta$ and energy conservation comes for free, regardless of how badly $H_\theta$ approximates the truth.

### 1.4 Worked example: the pendulum

For the unit-mass pendulum, $H = \tfrac{1}{2}p^2 + (1 - \cos q)$. Hamilton's equations give $\dot q = p,\ \dot p = -\sin q$. Trajectories are *level sets of $H$* (Figure 1, left): closed loops for $H<2$ (libration), open curves for $H>2$ (rotation), and a separatrix at $H=2$. The dynamics never leave a level set -- a fact your neural network must respect if it hopes to extrapolate.

---

## 2. The symplectic 2-form: geometry of phase space

### 2.1 What "structure" means

Phase space is not just a vector space; it carries a closed, non-degenerate 2-form

$$
\omega \;=\; \sum_{i=1}^{n} dq_i \wedge dp_i.
$$

Concretely, $\omega$ assigns an **oriented area** to every infinitesimal parallelogram in phase space. The Hamiltonian flow $\phi_t$ is special because it preserves $\omega$:

$$
\phi_t^* \omega \;=\; \omega \qquad\text{(symplecticity).} \tag{4}
$$

### 2.2 The Jacobian condition

A diffeomorphism $\phi : \mathbb{R}^{2n} \to \mathbb{R}^{2n}$ is **symplectic** iff its Jacobian $M = \partial \phi / \partial \mathbf{z}$ satisfies

$$
\boxed{\;M^\top J\,M \;=\; J.\;} \tag{5}
$$

This is the discrete-time analogue of (3). For $n=1$ it reduces to $\det M = 1$ (area preservation in the $(q,p)$ plane). For $n>1$ it is *strictly stronger* than $\det M = 1$; symplecticity also constrains all $2k\times 2k$ minors of $M$ for every $k$.

### 2.3 Liouville's theorem

Taking determinants in (5) gives $\det(M)^2 = 1$, so $|\det M| = 1$ everywhere along the flow. Phase-space volume is preserved (Figure 1, right):

$$
\frac{d}{dt}\operatorname{vol}(\phi_t(U)) = 0 \qquad \text{for every region } U \subset \mathbb{R}^{2n}.
$$

This is the foundation of statistical mechanics: the Gibbs ensemble would not be well-defined without it. It also has a startling consequence -- **Poincare's recurrence**: any bounded Hamiltonian trajectory returns arbitrarily close to its starting point infinitely often. Your simulator must reproduce this; an integrator that lets phase volume contract to a point cannot.

### 2.4 What goes wrong without symplecticity

For *every* explicit Runge-Kutta method, $|\det M_h| = 1 + c h^{p+1} + \mathcal{O}(h^{p+2})$ with $c \neq 0$ in general, where $p$ is the order. Phase volume changes by a tiny amount each step. Multiply by $10^6$ steps and the simulated trajectory has either spiralled inward (energy lost) or outward (energy gained) by a macroscopic amount. Section 3 shows how this looks in pictures.

---

## 3. Symplectic integrators

### 3.1 Why ordinary integrators drift

Apply explicit Euler to (3):

$$
\mathbf{z}_{k+1} = \mathbf{z}_k + h\,J\,\nabla H(\mathbf{z}_k), \qquad M_h = I + h J \nabla^2 H.
$$

Then $M_h^\top J M_h = J + h^2 (\nabla^2 H)^\top J\,J\,J\,(\nabla^2 H) + \cdots = J + \mathcal{O}(h^2)$, which fails (5). Energy drifts as $\sim h\,t$. Even RK4, with its $\mathcal{O}(h^4)$ local error, fails (5) and shows secular drift over long horizons (Figure 4, right).

### 3.2 Symplectic Euler

Update $\mathbf{p}$ using the *new* position (or vice versa):

$$
\mathbf{p}_{k+1} = \mathbf{p}_k - h\,\partial_q H(\mathbf{q}_k, \mathbf{p}_{k+1}), \qquad
\mathbf{q}_{k+1} = \mathbf{q}_k + h\,\partial_p H(\mathbf{q}_k, \mathbf{p}_{k+1}). \tag{6}
$$

For separable Hamiltonians $H(q,p) = T(p) + V(q)$, this is *explicit*: $\mathbf{p}_{k+1} = \mathbf{p}_k - h\,V'(\mathbf{q}_k)$, then $\mathbf{q}_{k+1} = \mathbf{q}_k + h\,T'(\mathbf{p}_{k+1})$. A direct calculation gives $M_h^\top J M_h = J$ exactly.

### 3.3 Stormer-Verlet (leapfrog)

The workhorse of molecular dynamics, planetary integration, and Hamiltonian Monte Carlo. Half-kick, full drift, half-kick:

$$
\begin{aligned}
\mathbf{p}_{k+1/2} &= \mathbf{p}_k - \tfrac{h}{2}\,V'(\mathbf{q}_k), \\
\mathbf{q}_{k+1} &= \mathbf{q}_k + h\,\mathbf{p}_{k+1/2}, \\
\mathbf{p}_{k+1} &= \mathbf{p}_{k+1/2} - \tfrac{h}{2}\,V'(\mathbf{q}_{k+1}).
\end{aligned} 
$$

Second-order accurate, *symmetric* in time, and symplectic. Figure 4 shows the staggered grid: positions live at integer steps and momenta at half-integer steps. The structure ensures that there exists a **modified Hamiltonian** $\tilde H_h = H + h^2 H_2 + h^4 H_4 + \cdots$ that is *exactly* conserved by the discrete map. The energy you measure does not drift; it oscillates with bounded amplitude $\mathcal{O}(h^2)$ around the true value forever.

![Two-panel comparison: phase-space trajectory of the pendulum integrated by explicit Euler, symplectic Euler, and leapfrog; relative energy error versus time on a symlog axis showing Euler diverging while symplectic methods stay bounded.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/05-Symplectic-Geometry/fig2_integrator_comparison.png)
*Figure 2. Long-term behaviour of the pendulum integrated for 80 seconds at $h = 0.05$. Left: explicit Euler (red) spirals outward to escape; symplectic Euler (green) and leapfrog (blue) trace nearly the reference orbit. Right: relative energy error $(H-H_0)/H_0$ on a symlog axis. The symplectic methods oscillate inside a tiny envelope; explicit Euler drifts unboundedly. This is not a quantitative point about accuracy -- it is a **qualitative point about geometry**.*

### 3.4 Symplectic integrators in code

```python
def leapfrog(q, p, dVdq, h, n_steps):
    """Stormer-Verlet leapfrog for separable H = p^2/2 + V(q)."""
    for _ in range(n_steps):
        p = p - 0.5 * h * dVdq(q)   # half kick
        q = q + h * p               # full drift
        p = p - 0.5 * h * dVdq(q)   # half kick
    return q, p
```

That is the entire symplectic story for separable Hamiltonians. Six lines, no dependencies, qualitatively correct for every conservative system. **Now we will teach a neural network to do this.**

![Leapfrog stencil showing positions q at integer steps and momenta p at half-integer steps connected by alternating kick and drift arrows; energy error comparison of leapfrog, RK4, and explicit Euler on the harmonic oscillator.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/05-Symplectic-Geometry/fig4_leapfrog.png)
*Figure 3. The leapfrog stencil. Left: positions and momenta live on staggered time grids. The "kick" updates $p$ using $-V'(q)$ and the "drift" updates $q$ using the current $p$. Each individual step is a shear -- and shears are symplectic -- so the composition is too. Right: even RK4 (orange), which is locally fourth-order accurate, drifts on the harmonic oscillator while leapfrog (blue) stays bounded. **Order is not the same as structure preservation.***

---

## 4. Hamiltonian Neural Networks (HNN)

### 4.1 The problem with naive neural ODEs

A neural ODE learns $\dot{\mathbf{z}} = f_\theta(\mathbf{z})$ with no constraint on $f_\theta$. Generically $f_\theta \neq J\,\nabla H$ for any $H$, so the learned dynamics is *not* Hamiltonian, and energy drifts. The drift is invisible at training time (small loss) and catastrophic at inference time (long rollouts diverge).

### 4.2 The HNN trick (Greydanus, Dzamba, Yosinski, 2019)

Don't learn the vector field. Learn the **scalar Hamiltonian** $H_\theta : \mathbb{R}^{2n} \to \mathbb{R}$ and recover the dynamics by autodiff:

$$
\dot{\mathbf{q}} = \frac{\partial H_\theta}{\partial \mathbf{p}}, \qquad \dot{\mathbf{p}} = -\frac{\partial H_\theta}{\partial \mathbf{q}}. \tag{8}
$$

Two consequences are immediate:

1. **Energy is conserved by construction.** $\frac{dH_\theta}{dt} = (\nabla H_\theta)^\top J\,(\nabla H_\theta) = 0$ regardless of how good $H_\theta$ is.
2. **Symmetries become learnable.** If you give the network $H_\theta$ that depends on $|\mathbf{q}|$ alone, angular momentum is conserved too. Noether's theorem holds verbatim.

![Block diagram showing phase-space input (q,p) flowing through an MLP that outputs scalar H_theta, then through autodifferentiation to gradients, finally producing dot-q and dot-p via Hamilton's equations, with a training loss feedback path.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/05-Symplectic-Geometry/fig3_hnn_architecture.png)
*Figure 4. HNN architecture. The network maps the phase-space state $(q,p)$ to a single scalar $H_\theta$. Autodiff produces $\partial_q H_\theta$ and $\partial_p H_\theta$, and Hamilton's equations turn those gradients into the time derivatives $(\dot q, \dot p)$. The training loss compares predicted derivatives to observed ones. Energy conservation does not appear as a soft penalty -- it is **baked into the forward pass**.*

### 4.3 Training loss

If you have phase-space samples with derivatives $\{(\mathbf{q}_t, \mathbf{p}_t, \dot{\mathbf{q}}_t, \dot{\mathbf{p}}_t)\}$:

$$
\mathcal{L}(\theta) \;=\; \sum_{t} \Big\| \partial_{\mathbf{p}} H_\theta - \dot{\mathbf{q}}_t \Big\|^2 + \Big\| \partial_{\mathbf{q}} H_\theta + \dot{\mathbf{p}}_t \Big\|^2. \tag{9}
$$

If only state samples $(\mathbf{q}_t, \mathbf{p}_t)$ are available, integrate (8) inside the loss with an adjoint-friendly ODE solver (Neural-ODE style) and compare integrated trajectories.

### 4.4 Reference implementation

```python
import torch
import torch.nn as nn

class HNN(nn.Module):
    """Hamiltonian Neural Network. Learns H(q,p); dynamics come from autodiff."""

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.dim = dim                             # phase-space dim = 2n
        self.H = nn.Sequential(
            nn.Linear(dim, hidden), nn.Softplus(),
            nn.Linear(hidden, hidden), nn.Softplus(),
            nn.Linear(hidden, 1),
        )

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        return self.H(z).squeeze(-1)               # (batch,)

    def vector_field(self, z: torch.Tensor) -> torch.Tensor:
        z = z.requires_grad_(True)
        H = self.hamiltonian(z).sum()
        grad_H = torch.autograd.grad(H, z, create_graph=True)[0]   # (B, 2n)
        n = self.dim // 2
        dHdq, dHdp = grad_H[:, :n], grad_H[:, n:]
        return torch.cat([dHdp, -dHdq], dim=-1)    # J grad H
```

Two design choices matter:

* **Smooth activations** (`Softplus`, `Tanh`). Hamilton's equations need the *gradient* of $H_\theta$; a `ReLU` would give a piecewise-constant vector field with kinks.
* **No bias on the last layer is fine.** $H$ is defined up to an additive constant, which has no dynamical effect.

### 4.5 Pendulum benchmark

Train an HNN and a vanilla MLP on $T = 4$ s of pendulum derivatives. Roll both forward to $T = 25$ s.

![Three-panel pendulum experiment: angle q(t) versus time, phase portrait, and relative energy error showing the HNN tracking the reference and the vanilla MLP drifting away.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/05-Symplectic-Geometry/fig5_pendulum_hnn_vs_nn.png)
*Figure 5. Pendulum extrapolation. Left: angle versus time -- the vanilla MLP slowly accumulates phase error and amplitude drift; the HNN stays locked on. Centre: phase portrait -- the MLP orbit either spirals outward or collapses inward; the HNN orbit is a slightly displaced closed loop. Right: relative energy error -- the MLP drifts secularly; the HNN error is **bounded**, oscillating around a small bias proportional to how well $H_\theta$ approximates $H_{\text{true}}$. This is the headline result of Greydanus et al. (2019).*

The relative energy error of the HNN is $< 10^{-2}$ throughout; the vanilla MLP grows past $10^{-1}$ within tens of seconds. The MLP is not "wrong" in the supervised loss -- it has comparable training error -- but it is wrong in the geometric sense that matters at deployment.

### 4.6 What HNNs do *not* do

* **They do not solve Hamilton's equations symplectically.** The HNN forward pass produces a vector field; you still need to integrate it. If you integrate it with explicit Euler, you will see energy drift (smaller than the vanilla NN, but still present). For best results, integrate the HNN's vector field with a symplectic integrator. A few works (e.g. `SRNN`, Chen et al. 2020) bake symplectic integration into training too.
* **They cannot model dissipation or driving.** $\dot z = J\nabla H$ is conservative by definition. For dissipative systems, see Port-Hamiltonian NNs (Desai et al., 2021) or GENERIC-style decompositions.

---

## 5. Lagrangian Neural Networks (LNN)

### 5.1 Why a Lagrangian flavour

HNNs need both position **and** momentum. In many practical settings (a video of a robot arm, a ball-and-spring lab demo) you can measure positions and *velocities* but not momenta, and computing momenta requires knowing the mass matrix $M(q)$. Lagrangian Neural Networks (Cranmer et al., 2020) sidestep this by working in the configuration space $(\mathbf{q}, \dot{\mathbf{q}})$.

### 5.2 The Euler-Lagrange equations as a learnable closure

Given a learnable scalar $L_\theta(\mathbf{q}, \dot{\mathbf{q}})$, the equations of motion are

$$
\frac{d}{dt}\frac{\partial L_\theta}{\partial \dot{\mathbf{q}}} \;-\; \frac{\partial L_\theta}{\partial \mathbf{q}} \;=\; 0,
$$

which after one application of the chain rule gives

$$
\boxed{\;\ddot{\mathbf{q}} \;=\; \big(\nabla_{\dot q} \nabla_{\dot q} L_\theta\big)^{-1} \!\Big[\nabla_q L_\theta - \big(\nabla_{\dot q} \nabla_{q} L_\theta\big)\dot{\mathbf{q}}\Big].\;} \tag{10}
$$

The right-hand side is computed by a single forward pass plus one Hessian via autodiff, and the matrix inverse is $\mathcal{O}(n^3)$ in the configuration dimension (cheap for $n \lesssim 100$). Because $L_\theta$ is a scalar that defines a Hamiltonian system via the Legendre transform, the resulting dynamics is symplectic.

### 5.3 LNN vs HNN at a glance

| | HNN | LNN |
|---|---|---|
| Learns | $H(q, p)$ | $L(q, \dot q)$ |
| Needs | momenta $p$ | only $\dot q$ |
| Forward cost | one autodiff gradient | one Hessian + one inverse |
| Best for | systems with known mass matrix | systems where mass is unknown / state-dependent |

---

## 6. Symplectic Networks (SympNets)

HNNs and LNNs guarantee symplecticity at the *continuous* level. After you discretise with an ODE solver, the discrete map is only approximately symplectic. **SympNets** (Jin et al., 2020) parameterise the discrete-time map $\phi_\theta$ directly out of building blocks that are *exactly* symplectic, then compose them:

$$
\phi_\theta \;=\; \phi_K \circ \phi_{K-1} \circ \cdots \circ \phi_1.
$$

The two canonical building blocks are

* **Up-shear** $(q, p) \mapsto (q + \nabla S_\theta(p),\; p)$ for any scalar $S_\theta$,
* **Low-shear** $(q, p) \mapsto (q,\; p + \nabla T_\theta(q))$ for any scalar $T_\theta$.

A direct calculation shows each shear has a triangular Jacobian with unit diagonal, so $M^\top J M = J$ exactly. The parameter functions $S_\theta, T_\theta$ are MLPs. Composing $K$ such layers gives a universal approximator for symplectic maps.

![Three side-by-side architecture diagrams comparing HNN learning a Hamiltonian, LNN learning a Lagrangian, and SympNet learning a symplectic map directly.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/05-Symplectic-Geometry/fig6_hnn_lnn_sympnet.png)
*Figure 6. The three families. **HNN** learns the energy and recovers the flow by autodiff -- exact continuous-time symplecticity. **LNN** learns the action and pays for one Hessian inversion in the forward pass -- works with $(q,\dot q)$ data. **SympNet** parameterises the discrete-time map directly out of shears -- no ODE solver in the loop, exact discrete-time symplecticity. Choice of family depends on what you can measure and what you need to guarantee.*

---

## 7. Where this matters

![Hub-and-spoke diagram with Hamiltonian / Symplectic deep learning at the centre and six application areas around it: molecular dynamics, robotics, celestial mechanics, plasma physics, Hamiltonian Monte Carlo, and fluid / climate.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/pde-ml/05-Symplectic-Geometry/fig7_applications.png)
*Figure 7. Application landscape for structure-preserving deep learning. Anywhere a long, conservative simulation matters -- molecular dynamics ($10^9$ time steps), orbital mechanics ($10^6$ years), plasma physics, fluid reduced-order models, Hamiltonian Monte Carlo proposals -- the energy drift of vanilla integrators (and vanilla networks) is the limiting factor.*

* **Molecular dynamics.** All-atom simulations run for $10^8$-$10^9$ steps. With a non-symplectic integrator the temperature would slowly cook off; leapfrog is the only reason MD works at all. HNN-flavoured surrogates inherit this stability.
* **Orbital mechanics and the n-body problem.** The JPL ephemerides use symplectic integrators because they have to keep planetary orbits stable for $10^4$-$10^6$ years.
* **Hamiltonian Monte Carlo.** The leapfrog step is at the heart of NUTS and modern HMC. A learned Hamiltonian (Levy et al., 2018) can produce dramatically faster mixing in high-dimensional posteriors.
* **Robotics and control.** LNNs and DeLaNs (Lutter et al., 2019) learn rigid-body dynamics from video and use the learned $L$ inside a model-based RL or trajectory-optimisation loop.
* **Plasma physics, accelerator design, climate.** Anywhere you need a reduced-order model that respects energy and momentum balance over long horizons, structure-preserving networks are the right hammer.

---

## 8. Common pitfalls

* **Forgetting `create_graph=True`.** The HNN gradient is itself differentiated during backprop -- without `create_graph=True` PyTorch will silently detach the graph and your gradient w.r.t. $\theta$ will be wrong.
* **Choosing `ReLU` activations.** $H_\theta$ must be twice differentiable for the HNN gradient to be smooth. Use `Softplus`, `Tanh`, `SiLU`, or `GELU`.
* **Integrating an HNN with explicit Euler.** The continuous-time dynamics is symplectic but Euler discretisation breaks it. Use leapfrog at inference time.
* **Confusing accuracy and structure.** RK4 is *more accurate per step* than leapfrog, and *less stable per long rollout*. Order $\neq$ structure preservation.
* **Asking an HNN to model friction.** $\dot z = J\nabla H$ is conservative. Add a damping term (Port-Hamiltonian or GENERIC) for dissipative systems.

---

## 9. Exercises

**Exercise 1.** Show that any symplectic map $\phi : \mathbb{R}^{2} \to \mathbb{R}^2$ has Jacobian determinant $1$. *Hint:* take determinants in (5).

> **Solution.** From $M^\top J M = J$, $\det(M)^2 \det(J) = \det(J)$. Since $\det(J) = 1$ for $n=1$, $\det(M)^2 = 1$, hence $\det M = \pm 1$. Continuity from the identity (which has $\det = +1$) forces $\det M = +1$ along any flow.

**Exercise 2.** Verify symplectic Euler preserves area for the harmonic oscillator $H = (p^2 + q^2)/2$.

> **Solution.** The map $(q_k, p_k) \mapsto (q_{k+1}, p_{k+1})$ is $p_{k+1} = p_k - h q_k$, $q_{k+1} = q_k + h p_{k+1} = q_k + h p_k - h^2 q_k$. The Jacobian is $\begin{pmatrix} 1 - h^2 & h \\ -h & 1 \end{pmatrix}$ with determinant $(1-h^2)\cdot 1 - h\cdot(-h) = 1$. Area preserved exactly, for every $h$.

**Exercise 3.** Why does an unconstrained neural ODE generically violate energy conservation?

> **Solution.** $f_\theta : \mathbb{R}^{2n} \to \mathbb{R}^{2n}$ has $4n^2$ free Jacobian entries; the symplecticity constraint $M^\top J M = J$ pins down $n(2n+1)$ of them. The complementary $\binom{2n}{2}$-dimensional subspace of *non-Hamiltonian* perturbations is generic. Almost every $f_\theta$ that fits the training data lies off the symplectic submanifold.

**Exercise 4.** Show that the leapfrog step can be written as $L_h = L_{h/2}^{\text{kick}} \circ L_h^{\text{drift}} \circ L_{h/2}^{\text{kick}}$, where each piece is exactly symplectic.

> **Solution.** Each kick is the time-$h/2$ flow of the Hamiltonian $V(q)$; each drift is the time-$h$ flow of $T(p) = p^2/2$. Both are exact Hamiltonian flows, hence symplectic. Composition of symplectic maps is symplectic.

---

## 10. References

[1] Greydanus, S., Dzamba, M., & Yosinski, J. (2019). **Hamiltonian Neural Networks.** *NeurIPS.* [arXiv:1906.01563](https://arxiv.org/abs/1906.01563).

[2] Cranmer, M., Greydanus, S., Hoyer, S., Battaglia, P., Spergel, D., & Ho, S. (2020). **Lagrangian Neural Networks.** *ICLR DeepDiffEq workshop.* [arXiv:2003.04630](https://arxiv.org/abs/2003.04630).

[3] Jin, P., Zhang, Z., Zhu, A., Tang, Y., & Karniadakis, G. E. (2020). **SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems.** *Neural Networks*, 132, 166-179.

[4] Chen, Z., Zhang, J., Arjovsky, M., & Bottou, L. (2020). **Symplectic Recurrent Neural Networks.** *ICLR.* [arXiv:1909.13334](https://arxiv.org/abs/1909.13334).

[5] Hairer, E., Lubich, C., & Wanner, G. (2006). **Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations.** Springer (2nd ed.). The standard reference.

[6] Toth, P., Rezende, D. J., Jaegle, A., Racaniere, S., Botev, A., & Higgins, I. (2020). **Hamiltonian Generative Networks.** *ICLR.* [arXiv:1909.13789](https://arxiv.org/abs/1909.13789).

[7] Lutter, M., Ritter, C., & Peters, J. (2019). **Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning.** *ICLR.* [arXiv:1907.04490](https://arxiv.org/abs/1907.04490).

[8] Desai, S., Mattheakis, M., Joy, H., Protopapas, P., & Roberts, S. (2021). **Port-Hamiltonian Neural Networks for Learning Explicit Time-Dependent Dynamical Systems.** *Phys. Rev. E* 104, 034312. [arXiv:2107.08024](https://arxiv.org/abs/2107.08024).

[9] Levy, D., Hoffman, M. D., & Sohl-Dickstein, J. (2018). **Generalizing Hamiltonian Monte Carlo with Neural Networks.** *ICLR.* [arXiv:1711.09268](https://arxiv.org/abs/1711.09268).

[10] Arnold, V. I. (1989). **Mathematical Methods of Classical Mechanics.** Springer (2nd ed.). The geometric foundations.

---

*This is Part 5 of the [PDE and Machine Learning](/categories/PDE-and-Machine-Learning/) series. Next: [Part 6 -- Continuous Normalizing Flows and Neural ODEs](/en/PDE-and-Machine-Learning-6-Continuous-Normalizing-Flows/). Previous: [Part 4 -- Variational Inference and the Fokker-Planck Equation](/en/PDE-and-Machine-Learning-4-Variational-Inference/).*
