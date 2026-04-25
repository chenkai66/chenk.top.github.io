---
title: "MoSLoRA: Mixture-of-Subspaces in Low-Rank Adaptation"
date: 2024-10-12 09:00:00
tags:
  - LLM
  - PEFT
categories: Paper
lang: en
mathjax: true
description: "MoSLoRA boosts LoRA expressivity by mixing multiple low-rank subspaces with a lightweight mixer. Covers when vanilla LoRA fails, mixer design choices, and tuning tips."
disableNunjucks: true
---

LoRA is the default tool for adapting a frozen base model: cheap, stable, mergeable, and good enough for most single-task settings. But the moment your fine-tuning data is genuinely heterogeneous -- code mixed with math, instruction following mixed with creative writing, several domains in one adapter -- a single low-rank subspace starts to feel cramped. You can grow $r$, but cost grows with it and you still get *one* subspace, just a fatter one.

[**MoSLoRA**](https://arxiv.org/abs/2406.11909) (Wu, Huang and Wei, 2024) takes a different turn. Instead of one rank-$r$ pair $(B, A)$ it uses $k$ rank-$r$ pairs and lets a tiny **mixer matrix** decide how to combine them. The decomposition rewrites cleanly as a single $B\, W\, A$ product, so the mergeability that made LoRA deployable is preserved, and the extra parameter cost is essentially the $k\times k$ mixer. This post walks through why a single subspace is a real bottleneck, how the mixer changes the geometry of the update, where MoSLoRA actually moves the needle, and how to tune it without overfitting the mixer.

## What you will learn

- Why "just increase $r$" is *not* the right fix when adaptations are heterogeneous
- MoSLoRA's core idea: mixture of low-rank subspaces with a $k\times k$ mixer
- Mixer design choices: global weights, input-dependent gating, structured variants
- Parameter-count and inference-overhead trade-offs vs LoRA / LoRA-MoE / Full FT
- When MoSLoRA pays off (and when vanilla LoRA is enough)
- Practical tuning tips and a clean PyTorch sketch

## Prerequisites

- Comfortable with LoRA (low-rank adaptation) and the standard Transformer block
- Basic PEFT vocabulary (Adapters, Prefix-Tuning, BitFit)
- Light familiarity with Mixture-of-Experts routing

---

## 1. LoRA recap: why low-rank updates work, and where they don't

Take any linear projection $W \in \mathbb{R}^{d_{out}\times d_{in}}$ inside a Transformer (Q/K/V/O or up/down/gate of the MLP). LoRA freezes $W_0$ and learns the update $\Delta W$ in factorised form:

$$
W \;=\; W_0 \;+\; \Delta W,
\qquad
\Delta W \;=\; \frac{\alpha}{r}\, B\, A,
\qquad
B \in \mathbb{R}^{d_{out}\times r},\;
A \in \mathbb{R}^{r\times d_{in}}.
$$

With $r \ll \min(d_{in}, d_{out})$ the trainable cost drops from $d_{in}\!\cdot\!d_{out}$ to $r(d_{in}+d_{out})$, often $0.1\%$--$1\%$ of full FT. Initialising $B = 0$ ensures the adapted model starts identical to the base, and at inference the merged $W_0 + \frac{\alpha}{r} B A$ has the same shape as $W_0$ -- so deployment is exactly as cheap as the base model.

![LoRA recap: one frozen base + one low-rank update](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/moslora/fig1_lora_recap.png)

The strong assumption baked into LoRA is that **one** rank-$r$ subspace is the right shape for the update. For narrow tasks this is empirically fine. For broad ones it is not, because:

- The optimal $\Delta W$ may be *low-rank within each task* but *high-rank across tasks* -- different sub-skills push the weights in genuinely different directions.
- A single subspace forces all those directions to share one $A$ and one $B$. Gradients from one task tend to overwrite useful directions learned for another.
- The fix of "just increase $r$" gives you a fatter subspace, but the optimum may still want a **structured** combination of several thin subspaces, not one thick one.

This is the gap MoSLoRA targets.

## 2. Why "just increase $r$" is not always the right fix

Increasing $r$ does increase capacity, but it has costs that compound at scale:

- **Parameters and memory**: $r(d_{in}+d_{out})$ is linear in $r$; doubling $r$ doubles the adapter footprint per layer.
- **Diminishing returns**: empirically, accuracy curves flatten quickly past $r=8$--$16$ on many tasks.
- **Single-direction bias**: a fatter subspace is still one subspace. If the loss surface is *piecewise low-rank* (different inputs want different directions), you waste capacity on dimensions that one direction needs but another doesn't.

What you actually want is **structured capacity**: a small set of distinct rank-$r$ subspaces that the model can selectively combine. That is exactly what MoSLoRA provides, and at marginal extra cost.

## 3. Core idea of MoSLoRA: mixture of subspaces, written as $B\, W\, A$

MoSLoRA factorises the update through a learnable **mixer**:

$$
\Delta W
\;=\;
\sum_{i=1}^{k} W_{ii}\, B_i\, A_i
\;=\;
B\, W\, A,
$$

where $A \in \mathbb{R}^{kr\times d_{in}}$ stacks the $k$ "down" maps, $B \in \mathbb{R}^{d_{out}\times kr}$ stacks the $k$ "up" maps, and $W \in \mathbb{R}^{kr\times kr}$ is the **mixer**. The diagonal of $W$ recovers a sum of independent LoRAs; the off-diagonal entries let the model **mix** subspaces -- the up-projection of subspace $i$ can be paired with the down-projection of subspace $j$. This is the source of the extra expressivity.

![MoSLoRA architecture: k low-rank subspaces with a learnable mixer](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/moslora/fig2_subspace_mixing.png)

A few properties drop out of this form:

- **No router, no top-$k$, no load balancing**: every subspace contributes on every token. Training is as smooth as plain LoRA.
- **Mergeable at inference**: $B W A$ is a single matrix product, so $W_0 + \frac{\alpha}{r} B W A$ collapses back into a single dense weight at deployment time -- exactly like LoRA. This is the property that LoRA-MoE designs typically lose.
- **Cheap extra cost**: with $k$ subspaces of rank $r$, parameters become $kr(d_{in}+d_{out}) + (kr)^2$. The $(kr)^2$ mixer is the only new term, and for typical $k=4, r=8$ it is $1024$ scalars per adapted projection -- negligible.

Think of each $(B_i, A_i)$ as a "dial" pointing in some direction of weight space. The mixer learns *how loud each dial should be*, and crucially also *how dials can interact*. With $k=1$ MoSLoRA reduces to LoRA. With $k>1$ and a non-diagonal $W$ you get a strictly richer family of updates at the same per-token compute.

### What "subspace" means here

Each pair $(B_i, A_i)$ defines a rank-$r$ subspace of update directions in $\mathbb{R}^{d_{out}\times d_{in}}$. The collection $\{(B_i, A_i)\}_{i=1}^{k}$ spans a union of $k$ such subspaces; the mixer turns that union into a parameterised manifold of rank-$kr$ updates with structure. The structure is what matters: an unstructured rank-$kr$ LoRA would have $kr(d_{in}+d_{out})$ parameters but no inductive bias toward "few interpretable directions". MoSLoRA keeps the inductive bias *and* adds the manifold.

## 4. How this differs from LoRA-MoE and classical MoE routing

It is tempting to read MoSLoRA as "LoRA-MoE without the gate". That comparison is useful but slightly off. Classical MoE adds:

- An explicit **router** producing per-token expert assignments
- **Top-$k$** sparsity to keep compute bounded
- **Load-balancing losses** so experts don't collapse
- A non-trivial inference graph -- the active experts depend on the token, so you cannot pre-merge weights

LoRA-MoE inherits most of these constraints: experts (the LoRAs) are routed, only a subset is active per token, and the merged-weight optimisation that makes LoRA deployable is gone.

MoSLoRA flips the design choice. There is no routing; all $k$ subspaces always contribute; the mixer is a *learned linear combiner*, not a *discrete selector*. The trade-off is intentional:

- You give up the conditional sparsity that lets MoE scale to dozens of experts.
- You get back smooth optimisation, mergeable weights, and a model that behaves operationally like LoRA.

For the regime LoRA is used in -- one base model, a handful of adapters, deployable at base-model cost -- this is the right trade.

## 5. Mixer design choices

The matrix $W$ has several useful variants. The paper's default is the simplest one; the others are worth knowing because they cover most of the design space you'll encounter in follow-up work.

### Global mixer (default)

A single $W \in \mathbb{R}^{kr\times kr}$ per adapted projection, learned end-to-end. No input dependency. Cheap, stable, mergeable. Initialise $W$ near the identity so MoSLoRA at step 0 behaves like a sum of independent LoRAs.

### Input-dependent gating

Replace $W$ with $W(x) = g(x) \in \mathbb{R}^{k\times k}$ produced by a small projection of the pooled hidden state. Same idea as soft-MoE: every input gets its own combination weights. More expressive but:

- the merged inference shortcut is lost (since $W$ now depends on $x$)
- the gate can overfit on small datasets

Use it when the task distribution is genuinely multi-modal *and* you have enough data.

### Layer- or projection-conditioned mixers

Use a different $W$ per Transformer layer or per projection (Q/K/V/O/MLP). The cost is still tiny but the inductive bias is stronger: low layers can learn one mixing pattern, late layers another. This often gives the best stability/expressivity trade in practice.

### Structured / low-rank mixers

For very large $k$, replace the dense $W$ by a low-rank or block-diagonal matrix to keep $(kr)^2$ from growing. Rarely needed at the $k\le 8$ scale most papers operate in.

## 6. Parameter count and compute overhead

Per adapted projection of shape $d_{out}\times d_{in}$ with $k$ subspaces of rank $r$:

| Quantity                | LoRA                  | MoSLoRA                                       |
|-------------------------|-----------------------|-----------------------------------------------|
| Trainable params        | $r(d_{in}+d_{out})$   | $k r(d_{in}+d_{out}) + (kr)^2$                |
| Forward FLOPs (per tok) | $r(d_{in}+d_{out})$   | $kr(d_{in}+d_{out}) + (kr)^2$                 |
| Mergeable at inference  | yes                   | yes (global mixer); no (input-dependent gate) |
| Routing / load balance  | none                  | none                                          |

For $d_{in}=d_{out}=4096$, $r=8$, $k=4$: LoRA is $\approx 65$K params per projection; MoSLoRA is $\approx 263$K + $1024$ mixer scalars. Across all attention + MLP projections of an 8B model this is still under $1\%$ of total parameters.

The point is that the mixer cost is asymptotically free: $(kr)^2$ is dwarfed by $kr(d_{in}+d_{out})$ as soon as $kr \ll d_{in}$, which is the regime you want to be in anyway.

## 7. Empirical behaviour: where the gap actually shows up

The headline finding in the paper, and the consistent pattern in follow-up work, is that MoSLoRA gives a small-to-moderate but *consistent* lift over LoRA across heterogeneous benchmarks, while staying inside the LoRA cost envelope.

![Downstream accuracy: MoSLoRA vs LoRA across heterogeneous tasks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/moslora/fig3_downstream_perf.png)

Two patterns are worth internalising:

1. **The gap widens as the task distribution gets more heterogeneous.** Single-domain reasoning benchmarks see modest gains. Instruction-tuning mixtures and multi-skill suites see larger gains. This matches the intuition: one subspace is enough for one direction; many subspaces start to pay when there are many directions.
2. **Increasing $k$ helps more than increasing $r$ when you're already at $r=8$ or higher.** The Pareto frontier in the parameter-vs-accuracy plot bends -- adding subspaces moves you up faster than adding rank.

![Parameter efficiency: MoSLoRA shifts the Pareto frontier upward](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/moslora/fig4_param_efficiency.png)

The geometric picture for why this works is the cleanest mental model:

![One fat subspace vs many slim subspaces](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/moslora/fig5_subspace_visualisation.png)

One LoRA is a single elongated direction; targets that lie off that direction can only be approximated, with residual error proportional to their angle off the subspace. MoSLoRA places several slim subspaces at different angles and lets the mixer combine them; the reachable set is a structured *union* of subspaces, not a single fat one.

## 8. When MoSLoRA helps -- and when LoRA is enough

MoSLoRA pays off when at least one of the following is true:

- The fine-tuning data spans multiple sub-skills or domains (instruction tuning, multi-domain adaptation, multimodal instruction following).
- You have already tried larger $r$ on LoRA and seen accuracy plateau.
- You need the deployment properties of LoRA (mergeable, no routing) but want more capacity than vanilla LoRA can give without going to LoRA-MoE.

It is *not* worth the extra moving parts when:

- The task is narrow and homogeneous and LoRA at moderate $r$ already matches full FT.
- Your dataset is small relative to the adapter size; the mixer can overfit.
- You need the maximum capacity possible and can afford a real MoE; LoRA-MoE or full sparse MoE will give you more headroom.

## 9. Practical tuning tips

A short list of high-signal knobs from the paper and from practice:

- **Start small.** $k=2$--$4$, $r=8$ is a strong default. Going to $k=8$ helps only on very heterogeneous data.
- **Initialise the mixer near identity.** $W \leftarrow I + \varepsilon \cdot \text{Gaussian}$ keeps the model close to a sum-of-LoRAs at step 0 and lets the off-diagonal terms grow only when they help. Random Gaussian init for $W$ tends to slow convergence.
- **Attach to attention first, then MLP.** Q/K/V/O is the highest-leverage placement. Add the MLP up/gate/down projections only if you still see headroom.
- **Use a slightly smaller $\alpha/r$ scale than for LoRA.** With $k$ subspaces summing into the update, the effective gain is $k\times$ larger; halving $\alpha$ is a safe starting point.
- **Watch the mixer's spectrum.** If $W$'s singular values collapse to one direction, the model is effectively running as plain LoRA -- you've over-regularised or given the mixer too little learning rate.
- **Keep the global mixer unless you really need input-dependent gating.** The static mixer is mergeable and rarely the bottleneck; the dynamic gate is harder to train and breaks mergeability.

## 10. Implementation sketch (PyTorch)

A minimal, faithful implementation -- a drop-in for `nn.Linear` that adds the MoSLoRA update on the forward pass:

```python
import torch
import torch.nn as nn

class MoSLoRALinear(nn.Module):
    """y = x W0^T + (alpha / r) * x A^T W^T B^T."""

    def __init__(self, base: nn.Linear, r: int = 8, k: int = 4,
                 alpha: float = 16.0):
        super().__init__()
        self.base = base                # frozen
        for p in self.base.parameters():
            p.requires_grad = False

        d_in  = base.in_features
        d_out = base.out_features
        self.r, self.k = r, k
        self.scale = alpha / r

        # Stacked low-rank factors:  A in (k*r, d_in),  B in (d_out, k*r)
        self.A = nn.Parameter(torch.empty(k * r, d_in))
        self.B = nn.Parameter(torch.zeros(d_out, k * r))   # init zero
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

        # Mixer:  W in (k*r, k*r), initialised near identity
        self.W = nn.Parameter(torch.eye(k * r) + 0.01 * torch.randn(k * r, k * r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base path
        y = self.base(x)
        # adapter path: x -> A^T -> W^T -> B^T
        h = x @ self.A.T          # (..., k*r)
        h = h @ self.W.T          # (..., k*r)
        h = h @ self.B.T          # (..., d_out)
        return y + self.scale * h

    @torch.no_grad()
    def merge(self) -> None:
        """Fold MoSLoRA into the base weight for inference."""
        delta = self.scale * (self.B @ self.W @ self.A)   # (d_out, d_in)
        self.base.weight.add_(delta)
        # zero out the adapter so further forward passes don't double-apply
        self.B.zero_()
        nn.init.eye_(self.W)
```

Three details worth noting:

1. `B` is initialised to zero so the model starts identical to the base, exactly as in LoRA.
2. `W` starts near identity so MoSLoRA at step 0 behaves like a sum of $k$ independent LoRAs; off-diagonal mixing emerges during training.
3. `merge()` collapses everything into a single dense `Linear`. This is the property that makes MoSLoRA deployable in production with zero inference overhead.

## 11. Comparison: LoRA vs MoSLoRA vs LoRA-MoE vs Full FT

| Method        | Trainable params | Expressivity         | Inference cost          | Best for                                                        |
|---------------|------------------|----------------------|-------------------------|-----------------------------------------------------------------|
| **Full FT**   | 100%             | Highest              | Baseline (1x)           | Single homogeneous task, no deployment constraints              |
| **LoRA**      | $\sim 0.1$--$1$% | Medium               | $\approx 1$x (mergeable)| Single or narrow task distribution                              |
| **MoSLoRA**   | $\sim 0.5$--$3$% | High                 | $\approx 1$x (mergeable)| Heterogeneous multi-task / multi-domain, LoRA-style deployment  |
| **LoRA-MoE**  | $\sim 1$--$5$%   | High (sparse)        | $> 1$x, not mergeable   | Maximum capacity, willing to give up mergeable inference        |

The key insight is that MoSLoRA sits between LoRA and full FT *on the same axis as LoRA*: same deployment story, more capacity. LoRA-MoE moves you onto a different axis with different operational properties.

## 12. When MoSLoRA matters most

- **Instruction tuning across diverse task families.** Code, math, reasoning, creativity push the weights in different directions; one subspace cannot serve all of them well.
- **Multi-domain adaptation (finance + medical + legal).** Each domain wants its own small set of update directions; the mixer effectively learns a soft per-domain combination without an explicit router.
- **Continual / additive adaptation.** New subspaces can be introduced for new tasks without touching the existing ones, giving a modular path to capacity expansion. (This goes beyond the original paper but is a natural extension that several follow-ups have explored.)

## Takeaway

MoSLoRA is best read as a **structured capacity upgrade** for LoRA, not as a slimmer MoE:

- LoRA: one rank-$r$ subspace, one direction in weight space.
- MoSLoRA: $k$ rank-$r$ subspaces $+$ a tiny $kr \times kr$ mixer, mergeable into a single dense weight at inference.

The mixer is what makes the design work: it gives you a structured manifold of rank-$kr$ updates with the same operational footprint as LoRA, and avoids every hard part of MoE -- no router, no top-$k$, no load balancing, no broken mergeability. For practitioners running heterogeneous fine-tuning workloads who don't want to take the operational hit of LoRA-MoE, it is currently the most pragmatic capacity-vs-deployability trade in the LoRA family.

## References

- Wu, T., Huang, S. and Wei, F., 2024. **Mixture-of-Subspaces in Low-Rank Adaptation**. arXiv:2406.11909. [[paper]](https://arxiv.org/abs/2406.11909) [[code]](https://github.com/wutaiqiang/MoSLoRA)
- Hu, E.J., Shen, Y., Wallis, P., et al., 2022. **LoRA: Low-Rank Adaptation of Large Language Models**. ICLR 2022. [[paper]](https://arxiv.org/abs/2106.09685)
- Fedus, W., Zoph, B. and Shazeer, N., 2022. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity**. JMLR. [[paper]](https://arxiv.org/abs/2101.03961)
- Zadouri, T., Ustun, A., Ahmadian, A., et al., 2024. **Pushing Mixture of Experts to the Limit: Extremely Parameter Efficient MoE for Instruction Tuning**. arXiv:2309.05444. [[paper]](https://arxiv.org/abs/2309.05444)
- Liu, S.Y., Wang, C.Y., Yin, H., et al., 2024. **DoRA: Weight-Decomposed Low-Rank Adaptation**. ICML 2024. [[paper]](https://arxiv.org/abs/2402.09353)
