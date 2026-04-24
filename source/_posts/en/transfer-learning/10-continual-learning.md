---
title: "Transfer Learning (10): Continual Learning"
date: 2024-08-10 09:00:00
categories:
  - Transfer Learning
  - Machine Learning
tags:
  - Continual Learning
  - Catastrophic Forgetting
  - Incremental Learning
  - Lifelong Learning
  - Transfer Learning
series:
  name: "Transfer Learning"
  order: 10
  total: 12
lang: en
mathjax: true
description: "Derive catastrophic forgetting from gradient interference and the Fisher information matrix. Covers EWC, MAS, LwF, replay (ER/A-GEM), dynamic architectures, the three CL scenarios, FWT/BWT metrics, and a from-scratch EWC implementation."
disableNunjucks: true
---

You can teach yourself to play guitar this year and you will still remember how to ride a bike. A neural network cannot. Fine-tune a vision model on CIFAR-10 then on SVHN, evaluate it on CIFAR-10 again, and accuracy collapses to barely above chance. The phenomenon is called **catastrophic forgetting**, and overcoming it is the central problem of **continual learning (CL)**: a learner that absorbs a stream of tasks $\mathcal{T}_1, \mathcal{T}_2, \ldots$ without re-accessing past data and without losing what it already knew.

This post derives why forgetting happens (it is not a bug, it is the structure of SGD on overparameterised networks), then walks through the four families of solutions -- regularisation, replay, dynamic architectures, meta-learning -- with the math, the intuition, and a from-scratch EWC implementation.

## What You Will Learn

- The CL problem statement and the three scenarios (Task-IL, Domain-IL, Class-IL)
- Why SGD on a new task destroys old-task knowledge: gradient interference and the loss-landscape view
- Fisher information as a principled measure of parameter importance
- Regularisation methods -- EWC, MAS, SI, LwF -- and how they differ
- Replay methods -- Experience Replay, GEM, A-GEM -- and the projection geometry of A-GEM
- Dynamic architectures -- Progressive Networks, PackNet -- and their trade-offs
- The standard metrics: average accuracy, average forgetting, and forward/backward transfer
- A self-contained EWC implementation evaluated on Permuted MNIST

## Prerequisites

- Neural network training, gradients, the cross-entropy loss
- Basic familiarity with the Fisher information matrix
- Transfer-learning fundamentals (Parts 1-6 of this series)

---

## Problem Setup

Tasks arrive sequentially: $\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_T$. When the learner trains on $\mathcal{T}_t$ it sees $\mathcal{D}_t = \{(x_i, y_i)\}$, but $\mathcal{D}_{<t}$ is **not** available. After all $T$ tasks the model is tested on every task it has ever seen.

Three scenarios make the difficulty concrete (van de Ven & Tolias, 2019):

![Three CL Scenarios: Task-IL, Domain-IL, Class-IL](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/10-continual-learning/fig4_cl_scenarios.png)

- **Task-IL.** The task identity is known at test time. The model can use a per-task head -- only the shared trunk competes for capacity.
- **Domain-IL.** The label space is fixed but the input distribution shifts (clean -> rotated -> noisy MNIST). One head; no test-time task ID.
- **Class-IL.** Each task introduces *new classes* and the learner must classify across **all** classes seen so far without knowing which task a sample came from. This is the hardest setting and the one most relevant to deployment.

**Standard metrics.** Let $R_{i,j}$ be the accuracy on task $j$ after training on task $i$. After all $T$ tasks:

$$
\mathrm{Avg} \;=\; \frac{1}{T}\sum_{j=1}^{T} R_{T,j}, \qquad
\mathrm{Forgetting} \;=\; \frac{1}{T-1}\sum_{j=1}^{T-1}\!\left(\max_{t \le T} R_{t,j} - R_{T,j}\right).
$$

Lopez-Paz & Ranzato (2017) add two more, made for measuring transfer rather than retention:

$$
\mathrm{BWT} \;=\; \frac{1}{T-1}\sum_{j=1}^{T-1} (R_{T,j} - R_{j,j}),
\qquad
\mathrm{FWT} \;=\; \frac{1}{T-1}\sum_{j=2}^{T} (R_{j-1,j} - b_j),
$$

where $b_j$ is a random/untrained baseline on task $j$. **BWT < 0** is forgetting; **BWT > 0** is the rare and desirable phenomenon of *positive backward transfer* (learning later tasks helps earlier ones). **FWT > 0** means earlier tasks pre-shape representations that help future tasks zero-shot.

![Transfer matrix R[i,j] with FWT and BWT regions](./10-continual-learning/fig6_transfer_matrix.png)

---

## Why Forgetting Happens

### Gradient interference

For two tasks, write the gradients $\mathbf{g}_1 = \nabla_\theta \mathcal{L}_1$ and $\mathbf{g}_2 = \nabla_\theta \mathcal{L}_2$. A single SGD step on task 2 changes $\mathcal{L}_1$ to first order by

$$
\Delta \mathcal{L}_1 \approx -\eta\, \mathbf{g}_1 \cdot \mathbf{g}_2.
$$

If $\mathbf{g}_1 \cdot \mathbf{g}_2 < 0$, every step on task 2 *increases* the task-1 loss. In high-dimensional networks gradients of unrelated tasks are typically nearly orthogonal but the negative-cosine fraction is large enough to be devastating after thousands of steps.

### Loss-landscape view

The optima $\theta_1^{*}$ and $\theta_2^{*}$ live in different low-loss basins. SGD on task 2 starting from $\theta_1^{*}$ walks out of basin 1 unless something pulls it back. The figure below shows a vanilla baseline doing exactly that, alongside two repairs we will derive in the next sections.

![Catastrophic forgetting on a 5-task sequence: baseline vs EWC vs replay](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/10-continual-learning/fig1_catastrophic_forgetting.png)

### Fisher information = parameter importance

The Fisher information matrix of the model's predictive distribution $p_\theta(y \mid x)$ is

$$
F(\theta) \;=\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim p_\theta(\cdot \mid x)}\!\left[\nabla_\theta \log p_\theta(y \mid x)\, \nabla_\theta \log p_\theta(y \mid x)^{\top}\right].
$$

At a local optimum the Fisher equals the (positive semi-definite) Hessian of the negative log-likelihood, so the diagonal $F_i$ measures how steeply the loss rises when $\theta_i$ is perturbed. A large $F_i$ means $\theta_i$ is *load-bearing* for the task -- protect it. A small $F_i$ means the loss is flat in that direction -- it is safe to repurpose the parameter for a new task. Every regularisation method below is a specific answer to "how should we pick which parameters to protect?".

---

## Regularisation Methods

### Elastic Weight Consolidation (EWC)

Kirkpatrick et al. (2017) approximate the posterior over $\theta$ after task A by a Gaussian centred at $\theta_A^{*}$ with precision proportional to the Fisher diagonal. A second-order Taylor expansion of the task-A negative log-likelihood around $\theta_A^{*}$ then gives

$$
\mathcal{L}_A(\theta) \;\approx\; \mathcal{L}_A(\theta_A^{*}) + \tfrac{1}{2} (\theta - \theta_A^{*})^{\top} F_A\, (\theta - \theta_A^{*}).
$$

Adding this as a penalty when learning task B yields the EWC objective:

$$
\boxed{\;\mathcal{L}(\theta) \;=\; \mathcal{L}_B(\theta) \;+\; \frac{\lambda}{2} \sum_i F_{A,i}\, (\theta_i - \theta_{A,i}^{*})^{2}\;}
$$

Geometrically EWC anchors a quadratic well at the old optimum whose curvature matches the *true* curvature of the old loss. Updates are cheap in directions where $F_i$ is small (the loss was flat anyway) and expensive where $F_i$ is large.

![EWC penalty as a quadratic well in parameter space](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/10-continual-learning/fig2_ewc_penalty.png)

For a sequence of tasks one can either accumulate Fisher matrices, $F_{1:t} = \sum_{k \le t} F_k$, or use **Online EWC** (Schwarz et al., 2018) with a discount $\gamma \in (0, 1)$:

$$
\tilde F_t \;=\; \gamma\, \tilde F_{t-1} + F_t, \qquad \theta^{*}_{1:t} = \theta^{*}_t.
$$

Picking $\lambda$ matters. Too small and forgetting wins; too large and the model becomes plastic-blind ("rigidity"). Typical ranges are $\lambda \in [10^2, 10^4]$ for Permuted MNIST and $\lambda \in [1, 10]$ for Split CIFAR.

### Memory Aware Synapses (MAS)

EWC needs labels (the log-likelihood). Aljundi et al. (2018) replace it with the gradient of the squared output norm:

$$
\Omega_i \;=\; \mathbb{E}_{x}\!\left[\, \left| \frac{\partial \, \tfrac{1}{2}\|f(x;\theta)\|_2^{2}}{\partial \theta_i} \right| \, \right].
$$

This is **unsupervised** -- you can compute it on unlabelled data, even on the test stream -- which is a real advantage in deployed settings.

### Synaptic Intelligence (SI)

Zenke et al. (2017) compute importance *online* during training as the path integral of $-g_i \cdot \dot\theta_i$ along the SGD trajectory. No second pass over data is needed; the cost is folded into the optimiser.

### Learning without Forgetting (LwF)

Li & Hoiem (2017) take a different angle: instead of penalising parameter drift, penalise *output* drift. Snapshot the old model $f_{\text{old}}$ before training on the new task. For each new-task input $x$, compute soft targets $\sigma(z^{\text{old}}/T)$ and distill them into the new model on the *old* output heads, while training the new heads on the new task's labels:

$$
\mathcal{L} \;=\; \underbrace{\mathcal{L}_{\text{CE}}\bigl(y,\, z^{\text{new}}_{\text{new heads}}\bigr)}_{\text{learn new task}} \;+\; \alpha\, \underbrace{T^{2}\, \mathrm{KL}\!\bigl(\sigma(z^{\text{old}}/T)\,\Vert\,\sigma(z^{\text{new}}_{\text{old heads}}/T)\bigr)}_{\text{don't move old outputs}}.
$$

LwF needs no old data and no Fisher matrix -- only the old model. The temperature $T$ (typically 2-4) softens the distributions so the distillation signal carries shape information beyond the argmax.

![LwF: knowledge distillation from frozen old model](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/10-continual-learning/fig5_lwf_distillation.png)

---

## Replay Methods

A different philosophy: *keep* a small slice of the past around. With even a tiny memory buffer, mixing old samples into each mini-batch is by far the strongest baseline known.

![Experience replay pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/10-continual-learning/fig3_replay_buffer.png)

### Experience Replay (ER)

Maintain a memory $\mathcal{M}$ of size $N$. At every step sample $B_{\text{new}}$ from the new stream and $B_{\text{mem}}$ from $\mathcal{M}$, optimise

$$
\mathcal{L} \;=\; \mathcal{L}_{\text{new}}(B_{\text{new}}) \;+\; \alpha\, \mathcal{L}_{\text{mem}}(B_{\text{mem}}),
$$

then write some new samples back into $\mathcal{M}$. **Reservoir sampling** keeps a uniform sample over the entire past stream with a fixed-size buffer (Vitter, 1985); **class-balanced** sampling guarantees coverage of every class. Empirically $|B_{\text{mem}}| = |B_{\text{new}}|$ already recovers most of the joint-training accuracy on Split-CIFAR-style benchmarks.

### GEM and A-GEM

Lopez-Paz & Ranzato (2017) frame the gradient step itself as a constrained optimisation: pick the new gradient $\tilde{\mathbf{g}}$ closest to $\mathbf{g}_{\text{new}}$ that does not increase the loss on any past task represented in memory:

$$
\min_{\tilde{\mathbf{g}}} \tfrac{1}{2}\|\tilde{\mathbf{g}} - \mathbf{g}_{\text{new}}\|^{2}
\quad \text{s.t.} \quad \tilde{\mathbf{g}} \cdot \mathbf{g}_{k} \;\ge\; 0 \quad \forall k = 1, \ldots, t-1.
$$

This is a quadratic program with one constraint per past task -- it scales poorly. **A-GEM** (Chaudhry et al., 2019) keeps a single reference gradient $\mathbf{g}_{\text{ref}}$ averaged over a random batch from $\mathcal{M}$ and projects only when the cosine is negative:

$$
\tilde{\mathbf{g}} \;=\; \mathbf{g}_{\text{new}} \;-\; \frac{\mathbf{g}_{\text{new}} \cdot \mathbf{g}_{\text{ref}}}{\|\mathbf{g}_{\text{ref}}\|^{2}}\, \mathbf{g}_{\text{ref}} \quad \text{if } \mathbf{g}_{\text{new}} \cdot \mathbf{g}_{\text{ref}} < 0,
$$

otherwise $\tilde{\mathbf{g}} = \mathbf{g}_{\text{new}}$. The cost is one extra forward/backward on the reference batch and a single dot product -- a thousand times cheaper than GEM and almost as accurate.

### DER and DER++

Buzzega et al. (2020) store both the input *and* the model's logits at the time the sample was added. The replay loss becomes a logit-matching MSE, optionally combined with the original label cross-entropy. DER++ is currently among the strongest single-model baselines on most CL benchmarks.

---

## Dynamic Architectures

Instead of squeezing all tasks into a fixed parameter budget, grow the model.

- **Progressive Networks** (Rusu et al., 2016): freeze the network after each task and add a new column for the next task, with lateral connections from frozen columns into the new one. Forgetting becomes *zero by construction*, but parameters and inference cost grow linearly with $T$.
- **PackNet** (Mallya & Lazebnik, 2018): after each task, prune to a sparse subset of weights and freeze them; future tasks reuse the unpruned mask. Model size is fixed but available capacity shrinks each task -- after enough tasks, performance collapses.
- **Supermasks in Superposition** (Wortsman et al., 2020): keep parameters random and frozen; learn a binary mask per task. Storage per task is one bit per parameter, and surprisingly, performance rivals trained baselines.

The trade-off is universal: zero forgetting either costs growing parameters or shrinking capacity. Hybrid approaches -- a fixed trunk with lightweight per-task adapters (cf. Part 9) -- are how this technology actually ships.

---

## Implementation: EWC From Scratch

Below is a clean PyTorch implementation that (i) computes the empirical diagonal Fisher after each task, (ii) stores it together with $\theta^{*}$, and (iii) adds the EWC penalty to the next task's loss. It runs on Permuted MNIST out of the box.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class EWC:
    """Elastic Weight Consolidation.

    After each task, call `consolidate(dataloader)` to snapshot theta*
    and the empirical Fisher diagonal. During subsequent training, add
    `lambda * ewc.penalty()` to the loss.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.fisher: list[dict[str, torch.Tensor]] = []
        self.opt_params: list[dict[str, torch.Tensor]] = []

    @torch.enable_grad()
    def _empirical_fisher(self, dataloader, n_samples: int = 1024
                          ) -> dict[str, torch.Tensor]:
        """Diagonal Fisher: E[(d log p(y|x; theta) / d theta)^2]."""
        self.model.eval()
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad}

        seen = 0
        for x, _ in dataloader:
            x = x.to(self.device)
            self.model.zero_grad()
            logits = self.model(x)
            # Sample y from the model's predictive distribution -- this is the
            # *true* Fisher; using the labels gives the empirical Fisher.
            probs = F.softmax(logits, dim=-1)
            y = torch.multinomial(probs, 1).squeeze(-1)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2 * x.size(0)
            seen += x.size(0)
            if seen >= n_samples:
                break

        for n in fisher:
            fisher[n] /= max(seen, 1)
        return fisher

    def consolidate(self, dataloader, n_samples: int = 1024) -> None:
        """Call at the END of training a task."""
        self.fisher.append(self._empirical_fisher(dataloader, n_samples))
        self.opt_params.append(
            {n: p.detach().clone() for n, p in self.model.named_parameters()
             if p.requires_grad}
        )

    def penalty(self) -> torch.Tensor:
        """Sum_t Sum_i F_{t,i} * (theta_i - theta*_{t,i})^2."""
        if not self.fisher:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for F_t, theta_t in zip(self.fisher, self.opt_params):
            for n, p in self.model.named_parameters():
                if n in F_t:
                    loss = loss + (F_t[n] * (p - theta_t[n]) ** 2).sum()
        return 0.5 * loss


def train_task(model, ewc, loader, optimiser, *, ewc_lambda: float,
               epochs: int, device: str) -> None:
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(model(x), y) + ewc_lambda * ewc.penalty()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()


@torch.no_grad()
def evaluate(model, loader, device: str) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total
```

**Usage sketch** (Permuted MNIST):

```python
ewc = EWC(model, device=device)
for t, (train_loader, test_loader) in enumerate(tasks):
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    train_task(model, ewc, train_loader, opt,
               ewc_lambda=400.0 if t > 0 else 0.0,
               epochs=5, device=device)
    ewc.consolidate(train_loader)               # snapshot theta* and F
    accs = [evaluate(model, t_loader, device)   # check all tasks so far
            for _, t_loader in tasks[:t + 1]]
    print(f"After task {t + 1}: {accs}")
```

Two implementation details that matter:

1. **True vs empirical Fisher.** Sampling $y$ from $p_\theta(\cdot \mid x)$ (as above) gives the true Fisher and is theoretically what the EWC derivation uses. Plugging in the dataset labels gives the *empirical* Fisher; in practice both work and the empirical version is slightly stronger when labels are clean.
2. **Where to compute Fisher.** Compute it *after* you finish training the task -- that is when $\theta \approx \theta_t^{*}$ and the quadratic approximation is tight.

---

## Empirical Comparison

The figure below shows representative numbers on the two canonical CL benchmarks. Replay dominates regularisation when memory is allowed; with no memory, EWC and LwF still substantially beat naive SGD; nothing yet matches the joint upper bound.

![CL benchmarks: Permuted MNIST and Split CIFAR](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/10-continual-learning/fig7_benchmarks.png)

Three takeaways:

- **Replay is the strongest single trick.** Even 200 stored samples per task usually beats every regularisation-only method on class-IL.
- **Class-IL is much harder than task-IL.** Methods that score 80% on Permuted MNIST routinely drop to 40-50% on Split CIFAR-100.
- **Combine, don't choose.** Production systems use ER + LwF (or ER + DER) and a small EWC term. Each addresses a different failure mode.

---

## Q&A

**Q1. How should I pick EWC's $\lambda$?**
Start at 100 for MNIST-scale problems and 1-10 for CIFAR-scale. Run a sweep on the average accuracy *and* forgetting metric; the right $\lambda$ is the one that maximises Avg subject to Forgetting under your tolerance.

**Q2. Why does my EWC degenerate to "freeze everything" after many tasks?**
Accumulated Fisher matrices keep growing -- every parameter eventually gets a large $\sum_t F_{t,i}$. Use **Online EWC** with $\gamma \approx 0.95$ to forget old Fisher contributions exponentially.

**Q3. EWC vs MAS vs SI -- which one in practice?**
EWC for clean supervised tasks. MAS when you have unlabelled streams (it does not need labels). SI when you cannot afford a second pass over data after each task -- it is the cheapest because it is computed online.

**Q4. How big should the replay buffer be?**
On Split-CIFAR-style benchmarks the curve typically saturates around 200-500 samples per task. The interesting regime is "as small as you can afford" -- if you can afford more, replay just keeps winning.

**Q5. Continual learning vs multi-task learning -- aren't they the same?**
No. Multi-task has *all* data simultaneously, so you optimise a fixed objective; the only challenge is task balancing. CL has tasks one at a time and forbids re-access to past data; the challenge is forgetting. CL with infinite memory and no order constraint reduces to multi-task -- this is exactly the joint upper bound in the benchmark figure.

**Q6. Does replay leak data?**
Yes -- the buffer is literal training data. In privacy-sensitive deployments use **generative replay** (train a generator on past data, then sample from it for replay) or **dark experience** (store only logits, not inputs).

**Q7. Why is Class-IL so much harder than Task-IL?**
Class-IL requires *cross-task* discrimination at inference time. Even with perfect retention on each task, the per-task softmax heads have not seen each other's classes during training, so their logits are not calibrated against one another -- new-class outputs typically swamp old-class ones. iCaRL (Rebuffi et al., 2017) addresses exactly this with a nearest-class-mean classifier on top of the learned features.

---

## Summary

Catastrophic forgetting is a structural property of SGD on a single shared parameter vector, derivable from gradient interference and the geometry of high-dimensional loss landscapes. Solutions cluster into four families:

| Family | Mechanism | Typical example | Trade-off |
|---|---|---|---|
| Regularisation | Anchor important parameters | EWC, MAS, SI, LwF | No memory cost, weakest on class-IL |
| Replay | Re-train on past samples | ER, A-GEM, DER++ | Strongest in practice; needs storage |
| Dynamic architectures | Add capacity per task | Progressive Nets, PackNet, SupSup | Zero forgetting; growing model |
| Meta-learning | Learn *how* to continue learning | OML, MER | Powerful but costly to meta-train |

The takeaway for practitioners is direct: if you can store any data at all, run a small reservoir buffer with experience replay; layer in LwF for free regularisation through the old model snapshot; only reach for EWC/MAS when memory is impossible. The next part picks up cross-lingual transfer, where the "tasks" are languages and the same machinery -- careful sharing, careful protection -- carries over.

---

## References

- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.
- Schwarz, J., et al. (2018). Progress & Compress: A scalable framework for continual learning. *ICML*.
- Aljundi, R., et al. (2018). Memory Aware Synapses: Learning what (not) to forget. *ECCV*.
- Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence. *ICML*.
- Li, Z., & Hoiem, D. (2017). Learning without forgetting. *TPAMI*, 40(12), 2935-2947.
- Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *NeurIPS*.
- Chaudhry, A., et al. (2019). Efficient lifelong learning with A-GEM. *ICLR*.
- Buzzega, P., et al. (2020). Dark experience for general continual learning. *NeurIPS*.
- Rusu, A. A., et al. (2016). Progressive neural networks. *arXiv:1606.04671*.
- Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network by iterative pruning. *CVPR*.
- Wortsman, M., et al. (2020). Supermasks in superposition. *NeurIPS*.
- Rebuffi, S.-A., et al. (2017). iCaRL: Incremental classifier and representation learning. *CVPR*.
- van de Ven, G. M., & Tolias, A. S. (2019). Three scenarios for continual learning. *arXiv:1904.07734*.
- Vitter, J. S. (1985). Random sampling with a reservoir. *ACM TOMS*, 11(1), 37-57.

---

## Series Navigation

- Previous: [Part 9 -- Parameter-Efficient Fine-Tuning](/en/transfer-learning-9-parameter-efficient-fine-tuning/)
- Next: [Part 11 -- Cross-Lingual Transfer](/en/transfer-learning-11-cross-lingual-transfer/)
- [View all 12 parts in this series](/tags/Transfer-Learning/)
