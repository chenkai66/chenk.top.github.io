---
title: "Transfer Learning (6): Multi-Task Learning"
date: 2025-05-05 09:00:00
tags:
  - Deep Learning
  - Transfer Learning
  - Multi-Task Learning
  - Parameter Sharing
  - Gradient Conflict
  - PCGrad
  - GradNorm
categories:
  - Transfer Learning
series:
  name: "Transfer Learning"
  part: 6
  total: 12
lang: en
mathjax: true
description: "Train one model on multiple tasks simultaneously. Covers hard vs. soft parameter sharing, gradient conflicts (PCGrad, GradNorm, CAGrad), auxiliary task design, and a complete multi-task framework with dynamic weight balancing."
disableNunjucks: true
series_order: 6
---

A self-driving car looking through a single camera needs to do three things at once: detect cars and pedestrians, segment lanes and free space, and estimate how far away each pixel is. You could train three separate networks. You would burn 3x the parameters, run 3x the forward passes at inference, and ignore the obvious fact that all three tasks need the same kind of low-level features (edges, surfaces, occlusion cues).

Multi-task learning (MTL) is the alternative: one shared backbone, one task-specific head per output, all trained jointly. Done well, you cut parameters by 60% **and** lift accuracy on every task because each task acts as a regularizer for the others. Done badly, two of your three tasks regress and you waste a week wondering why.

This article is about doing it well. The hard parts are not the architecture — that is one diagram. The hard parts are (1) the loss-scale mismatch between a cross-entropy term and an L2 depth term, (2) the gradient conflicts that arise 30-50% of the time when two tasks pull in different directions, and (3) figuring out which tasks even belong in the same model. We will cover the architectures (hard vs soft sharing, cross-stitch, MTAN), the optimizers that survive contact with real loss landscapes (Uncertainty Weighting, GradNorm, PCGrad, CAGrad), and a runnable PyTorch framework that ties it all together.

## What You Will Learn

- Why MTL works at all — the regularization, data-augmentation and efficiency views
- Hard vs soft parameter sharing, and the cross-stitch / MTAN middle ground
- How to measure task affinity *before* committing to a multi-task design
- Gradient conflicts: what they are, how often they happen, how PCGrad and CAGrad fix them
- Loss-scale balancing with Uncertainty Weighting (Kendall et al.) and GradNorm
- A complete PyTorch implementation of all three balancing methods

**Prerequisites:** Parts 1-2 of this series, comfort training neural networks in PyTorch.

---

## Why Multi-Task Learning?

### Shared structure: tasks that need the same features

The clearest case for MTL is when several tasks demand the same low-level representations:

| Task                  | What the features must encode                |
| --------------------- | -------------------------------------------- |
| Object detection      | Spatial layout, object boundaries, textures  |
| Semantic segmentation | Spatial layout, object boundaries, textures  |
| Depth estimation      | Spatial layout, textures, geometric cues     |

Three tasks, one set of underlying features. Training three encoders separately means each one rediscovers edges, surfaces, and shape priors from scratch. Sharing the encoder forces those features to be learned **once**, with the supervisory signal of all three tasks pushing in the same direction.

### Regularization view

Given $T$ tasks with losses $\mathcal{L}_1, \ldots, \mathcal{L}_T$, shared parameters $\theta_{\text{sh}}$, and task-specific parameters $\theta_t$, the joint objective is

$$
\mathcal{L}_{\text{MTL}} \;=\; \sum_{t=1}^{T} w_t \cdot \mathcal{L}_t(\theta_{\text{sh}}, \theta_t).
$$

The shared parameters $\theta_{\text{sh}}$ must lie in the intersection of the "good for task $t$" regions for *every* $t$. That intersection is much smaller than any single task's region, which acts as an **implicit prior** on $\theta_{\text{sh}}$. Empirically the model overfits less to any one task's noise — exactly the kind of regularization Caruana (1997) showed in the original MTL paper.

### Data-augmentation view

When the main task is data-starved, related auxiliary tasks supply additional supervisory signal through the shared parameters.

**Concrete:** low-resource MT from English to Swahili (~100K parallel pairs). Add an auxiliary English-to-French task (~10M pairs). The shared English encoder now sees 100x more English sentences. The Swahili side gains nothing directly, but the encoder it depends on becomes much better — typical reported gains are 5-20% BLEU on the low-resource task.

### Compute efficiency

| Setup                            | Parameters | Forward passes |
| -------------------------------- | ---------- | -------------- |
| 3 separate ResNet-50 models      | 75 M       | 3              |
| 1 shared encoder + 3 lightweight heads | 31 M  | 1              |

About 60% fewer parameters and a single forward pass for all three predictions — visualised on the right of Figure 6 below. For real-time systems (autonomous driving, AR, on-device) this is often the *primary* reason to do MTL.

### The risk: negative transfer

MTL is not free. When tasks demand genuinely different features, joint training underperforms separate training. The classic counter-example:

- **Face recognition** needs fine-grained textural detail of the inner face.
- **Scene classification** needs coarse global layout.

Forcing both through the same backbone gives you a backbone that is mediocre at both. The cure is not to give up on MTL — it is to either (a) measure the conflict before training (next section), (b) use soft sharing so the tasks can diverge, or (c) use a gradient-surgery method that prevents one task from actively harming the other.

---

## Parameter Sharing Strategies

![Hard vs soft parameter sharing](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/06-multi-task-learning/fig1_hard_vs_soft_sharing.png)

### Hard parameter sharing

The default and still the strongest baseline. All tasks share the backbone; each gets its own head:

$$
\text{features} \;=\; G_{\text{shared}}(x), \qquad
\hat{y}_t \;=\; G_t^{\text{head}}(\text{features}) \quad \forall\, t.
$$

**Design rules of thumb:**

- Share the first 70-80% of layers (general features).
- Keep the last 20-30% task-specific (each head can be 1-3 layers).
- Make heads wide enough that task-specific patterns have room to live.

Hard sharing gives you the strongest regularization, the smallest parameter count, and is impossible to misconfigure. **Always start here.**

### Soft parameter sharing

Each task has its own backbone, but a coupling term encourages the parallel parameters to stay similar:

$$
\mathcal{L} \;=\; \sum_t \mathcal{L}_t(\theta_t) \;+\; \lambda \!\!\sum_{i \neq j} \! \lVert \theta_i - \theta_j \rVert^2.
$$

The dashed yellow lines on the right of Figure 1 show those coupling terms. The model can break the symmetry layer by layer — useful when tasks need similar but not identical features.

### Cross-Stitch Networks

![Cross-stitch networks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/06-multi-task-learning/fig2_cross_stitch.png)

Cross-stitch networks (Misra et al., 2016) sit between hard and soft sharing. Each task has its own column of layers, but at every layer a small **cross-stitch unit** lets the two activations mix:

$$
\tilde{x}_A^{\,l} \;=\; \alpha_{AA}\, x_A^{\,l} + \alpha_{AB}\, x_B^{\,l},
\qquad
\tilde{x}_B^{\,l} \;=\; \alpha_{BA}\, x_A^{\,l} + \alpha_{BB}\, x_B^{\,l}.
$$

The four scalars $\alpha_{\bullet\bullet}$ per layer are learned. Their values are interpretable: large $\alpha_{AB}$ at a given layer means task $A$ leans heavily on task $B$'s features at that depth — useful as a diagnostic, not just an architectural trick.

### Multi-Task Attention Network (MTAN)

MTAN (Liu et al., 2019) shares a single backbone but lets each task carve out its own subset of the shared features through a sigmoid attention mask:

$$
\text{mask}_t = \sigma(W_t \cdot F_{\text{shared}} + b_t), \qquad
F_t = \text{mask}_t \odot F_{\text{shared}}.
$$

The mask is per-task and per-layer, so each task can "tune in" to different channels at different depths. MTAN is usually the strongest soft-sharing variant for vision MTL.

### Choosing between them

- **Hard sharing.** Default. Tasks closely related (same modality, same level of abstraction).
- **Cross-stitch / MTAN.** When you observe negative transfer with hard sharing, but the tasks still share a lot.
- **Fully soft sharing or completely separate models.** When tasks share little — at this point ask whether MTL is even the right tool.

---

## Measuring Task Affinity *Before* You Commit

![Task affinity matrix and grouping](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/06-multi-task-learning/fig7_task_affinity.png)

You should not pick the tasks for an MTL model based on intuition alone. Three quantitative tests are cheap and worth doing first.

1. **Transfer-experiment affinity** (Taskonomy-style). Train on task $A$, fine-tune on task $B$, compare against a from-scratch baseline. Improvement = positive affinity.
2. **Gradient cosine similarity.** Train a small joint model for one epoch, log $\cos(\nabla_\theta \mathcal{L}_A, \nabla_\theta \mathcal{L}_B)$ at each step. Consistently negative = the tasks are fighting.
3. **Feature similarity (CKA).** Compare learned representations across tasks. High CKA = the same backbone can serve both.

Figure 7 (left) shows a typical affinity matrix for seven vision tasks. Notice the tight cluster around `Detect / Segment / Edges` (all 0.78+) and the much looser ties to `Caption`. The dendrogram on the right turns those numbers into a concrete grouping recommendation: rather than one giant shared encoder, train

```
Group 1: Detect + Segment + Edges       -> shared encoder A
Group 2: Depth + Normals                -> shared encoder B
Group 3: Pose + Caption                 -> shared encoder C
```

Standley et al. (2020) showed that automated task grouping found this way (RL or hierarchical clustering) consistently beats both hand-grouping and a single global encoder.

---

## Gradient Conflicts and Task Balancing

![Gradient conflict and PCGrad](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/06-multi-task-learning/fig3_gradient_conflict.png)

This is where most MTL projects bleed performance. The architecture is fine; the optimizer is silently wrecking one task to help another.

### What "conflict" means precisely

Two tasks' gradients on the shared parameters conflict when

$$
\nabla \mathcal{L}_1 \cdot \nabla \mathcal{L}_2 \;<\; 0.
$$

Figure 3 (left) makes the geometry obvious. Task 1's gradient $g_1$ points right-and-up; task 2's $g_2$ points left-and-up. Their average $\bar{g}$ has nearly zero component along $g_1$ — i.e. **the naive sum-of-losses gradient barely helps task 1 at all**. The cosine similarity between $g_1$ and $g_2$ is $-0.43$ in this example.

How often does this happen in practice? Figure 3 (right) shows a representative empirical distribution of $\cos$ values during a multi-task training run: roughly **45% of updates conflict**, often persistently across epochs.

### Static weights (the baselines)

**Uniform** ($w_t = 1$): simple, occasionally fine, but breaks badly when loss scales differ. A classification cross-entropy of $\sim 1$ averaged with a depth MSE of $\sim 100$ means the optimizer is essentially doing single-task depth learning.

**Hand-tuned weights**: works if you have one or two tasks and a week to spare. Does not scale.

### Uncertainty Weighting (Kendall et al., 2018)

![Uncertainty weighting](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/06-multi-task-learning/fig5_uncertainty_weighting.png)

Treat each task's output as a Gaussian with learned noise $\sigma_t$. The negative log-likelihood becomes

$$
\mathcal{L} \;=\; \sum_t \frac{1}{2\sigma_t^2}\, \mathcal{L}_t \;+\; \log \sigma_t.
$$

Two things to notice from Figure 5:

- **Left panel.** For a fixed task loss $\mathcal{L}$, the combined objective has a unique minimum at $\sigma^* = \sqrt{\mathcal{L}}$. Without the $\log \sigma$ term the optimizer would push $\sigma_t \to \infty$ to drive the weighted loss to zero — the regularizer is what keeps the trick honest.
- **Right panel.** Tasks with raw losses spanning two orders of magnitude (`1`, `50`, `0.3`) end up contributing comparable weighted losses (`1.4`, `2.8`, `0.15`) once the learned $\sigma_t$ kicks in.

Typical gain over uniform weights: 2-5%. Cost: $T$ extra scalar parameters. Cheap and almost always worth turning on.

### GradNorm: balancing gradient magnitudes by training speed

![GradNorm dynamics](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/06-multi-task-learning/fig4_gradnorm.png)

Uncertainty weighting balances *loss magnitudes*. GradNorm (Chen et al., 2018) balances *gradient magnitudes*, conditioned on each task's training progress.

Define each task's relative inverse training rate

$$
\tilde{r}_t \;=\; \frac{\mathcal{L}_t(t)\,/\,\mathcal{L}_t(0)}{\overline{\mathcal{L}(t)\,/\,\mathcal{L}(0)}}.
$$

A value of $\tilde{r}_t > 1$ means task $t$ has shrunk its loss less than average — it is *falling behind*. GradNorm then nudges $w_t$ so that

$$
\lVert w_t \nabla \mathcal{L}_t \rVert \;\approx\; \overline{G}\cdot \tilde{r}_t^{\,\alpha},
$$

where $\overline{G}$ is the mean shared-parameter gradient norm and $\alpha \!\approx\! 1.5$ controls aggressiveness.

Figure 4 walks through a 60-epoch simulation:

- **Left.** Three tasks with very different loss scales and convergence rates.
- **Middle.** Their relative training rates $\tilde{r}_t$. The slow regression task drifts above 1; the fast auxiliary task drops below.
- **Right.** GradNorm responds by *raising* the lagging task's weight and *lowering* the leading task's weight, all without manual intervention.

Reported gains over uniform weights: 3-8% across multiple benchmarks.

### PCGrad: project away the conflicting component

PCGrad (Yu et al., 2020) tackles direction, not magnitude. Whenever two task gradients conflict, project one onto the *normal plane* of the other, removing the part that actively hurts:

$$
g_i' \;=\; g_i \;-\; \frac{g_i \cdot g_j}{\lVert g_j \rVert^2}\, g_j
\qquad \text{when } g_i \cdot g_j < 0.
$$

After projection $g_i' \cdot g_j = 0$ — no remaining conflict. The green arrow $g_1^{PC}$ in Figure 3 (left) shows the result: it preserves the part of $g_1$ that does not hurt $g_2$.

Pseudocode:

```
for each task i:
    g_i = backward pass on loss_i
    for each other task j:
        if g_i . g_j < 0:                     # conflict
            g_i = g_i - proj(g_i, g_j)        # remove conflicting component
final_gradient = mean of all modified gradients
```

**Theoretical guarantee:** the final gradient does not increase any task's loss to first order.

Reported on NYUv2 (segmentation + depth + normals):

- Uniform weights: mIoU 40.2%, depth error 0.61
- PCGrad:         mIoU 42.7%, depth error 0.58

### CAGrad: globally optimal conflict resolution

CAGrad (Liu et al., 2021) generalises PCGrad by solving a single QP that finds the smallest update vector still aligned with every task gradient:

$$
g^{*} \;=\; \arg\min_g \lVert g \rVert^2 \quad \text{s.t.}\quad g \cdot g_t \geq 0 \;\; \forall t.
$$

This is the *Pareto-optimal* descent direction — guaranteed not to harm any task, and globally rather than pairwise. Cost is $\mathcal{O}(T^2)$ per step instead of pairwise PCGrad. For $T \leq 5$ tasks, just use CAGrad.

### Comparison and combinability

| Method                | What it controls       | Cost                  |
| --------------------- | ---------------------- | --------------------- |
| Uniform               | Nothing                | Free                  |
| Uncertainty weighting | Loss magnitudes        | $T$ params            |
| GradNorm              | Gradient magnitudes    | $T$ params + 1 backward |
| PCGrad                | Gradient directions    | $T$ backwards         |
| CAGrad                | Gradient directions (global) | $T$ backwards + QP |

GradNorm (magnitude) and PCGrad/CAGrad (direction) are **orthogonal**. Combining GradNorm + PCGrad is a strong default for $T \geq 3$ tasks with mismatched scales.

---

## How Much Does Any of This Buy You?

![MTL vs single-task performance and cost](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/06-multi-task-learning/fig6_mtl_vs_single_task.png)

Figure 6 puts the methods next to each other on a NYUv2-style benchmark with three tasks:

- **Left.** Per-task improvement vs single-task baselines. Uniform MTL already wins on every task (the regularization effect is real). Uncertainty Weighting adds ~1-2 points. GradNorm and PCGrad add another 1-2 points each. CAGrad sits at the top.
- **Right.** The cost/efficiency win is independent of the balancing method: shared encoder + 3 heads cuts parameters from 75M to 31M and forward passes from 3 to 1.

Practical takeaway: a competent MTL setup with PCGrad or CAGrad regularly produces a model that is **smaller, faster, and more accurate** than the single-task ensemble it replaces. That is the unusual case where engineering and accuracy point in the same direction.

---

## Auxiliary Task Design

When your real interest is one main task and you are using MTL purely as a regularizer, the design question becomes: which auxiliary tasks to add?

### Self-supervised auxiliaries (free supervision)

- **Rotation prediction.** Rotate inputs by 0 / 90 / 180 / 270, predict the angle. Teaches orientation and object structure.
- **Jigsaw puzzles.** Shuffle image patches, predict the permutation. Teaches spatial layout.
- **Contrastive (SimCLR / MoCo).** Pull together two augmentations of the same input; push apart different inputs. Teaches augmentation-invariant features.

### Domain-specific auxiliaries

| Main task                  | Useful auxiliaries                              |
| -------------------------- | ----------------------------------------------- |
| Object detection           | Edge detection, depth estimation                |
| Named entity recognition   | POS tagging, dependency parsing                 |
| CTR prediction             | Conversion rate, dwell time, follow probability |
| Speech recognition         | Speaker ID, voice activity, noise classification |

### How many auxiliaries?

- Start with 1-2 of the most plausibly related ones.
- Add more only while validation on the *main* task keeps improving.
- 2-4 auxiliaries is typical; beyond ~10 you should be clustering tasks (Figure 7) instead of stacking them.

---

## Complete Implementation

A self-contained PyTorch framework supporting hard parameter sharing with three balancing options: uniform, PCGrad, and GradNorm.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import numpy as np
from typing import List, Dict, Optional


# ===== Network Architecture =====

class SharedEncoder(nn.Module):
    """Shared backbone: first 3 blocks of ResNet-18."""
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=False)
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class TaskHead(nn.Module):
    """Task-specific classification or regression head."""
    def __init__(self, in_channels, num_outputs, task_type='classification'):
        super().__init__()
        self.task_type = task_type
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_outputs))

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))


class MultiTaskNet(nn.Module):
    """Hard parameter sharing: shared encoder + task-specific heads."""
    def __init__(self, task_configs):
        super().__init__()
        self.encoder = SharedEncoder()
        self.heads = nn.ModuleDict({
            cfg['name']: TaskHead(256, cfg['num_classes'], cfg['type'])
            for cfg in task_configs
        })

    def forward(self, x):
        features = self.encoder(x)
        return {name: head(features) for name, head in self.heads.items()}


# ===== PCGrad =====

class PCGrad:
    """Projecting Conflicting Gradients (Yu et al., NeurIPS 2020)."""
    def __init__(self, optimizer, task_names):
        self.optimizer = optimizer
        self.task_names = task_names

    @staticmethod
    def _project(g_i, g_j):
        dot = torch.dot(g_i, g_j)
        if dot < 0:
            g_i = g_i - (dot / (g_j.norm() ** 2 + 1e-8)) * g_j
        return g_i

    def step(self, losses):
        # 1. Per-task flattened gradients on the shared parameters.
        grads = {}
        for name in self.task_names:
            self.optimizer.zero_grad()
            losses[name].backward(retain_graph=True)
            grads[name] = torch.cat([
                p.grad.flatten() for p in self.optimizer.param_groups[0]['params']
                if p.grad is not None
            ]).clone()

        # 2. Project away conflicting components pairwise.
        modified = {}
        for i, ni in enumerate(self.task_names):
            g = grads[ni].clone()
            for j, nj in enumerate(self.task_names):
                if i != j:
                    g = self._project(g, grads[nj])
            modified[ni] = g

        # 3. Use the average modified gradient for the optimizer step.
        avg_grad = sum(modified.values()) / len(modified)
        self.optimizer.zero_grad()
        idx = 0
        for p in self.optimizer.param_groups[0]['params']:
            if p.grad is not None:
                n = p.numel()
                p.grad = avg_grad[idx:idx + n].view_as(p)
                idx += n
        self.optimizer.step()


# ===== GradNorm =====

class GradNorm:
    """Gradient normalization for adaptive loss balancing
    (Chen et al., ICML 2018)."""
    def __init__(self, model, task_names, alpha=1.5, lr=0.025):
        self.model = model
        self.task_names = task_names
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(len(task_names)))
        self.weight_optim = optim.Adam([self.weights], lr=lr)
        self.initial_losses = None

    def step(self, losses):
        if self.initial_losses is None:
            self.initial_losses = {n: l.item() for n, l in losses.items()}

        weighted = [self.weights[i] * losses[n]
                    for i, n in enumerate(self.task_names)]
        total = sum(weighted)

        # Per-task gradient norms on the shared encoder only.
        shared_params = list(self.model.encoder.parameters())
        grad_norms = []
        for wl in weighted:
            grads = torch.autograd.grad(
                wl, shared_params, retain_graph=True, create_graph=True)
            grad_norms.append(
                torch.norm(torch.cat([g.flatten() for g in grads])))

        avg_norm = sum(grad_norms) / len(grad_norms)
        avg_ratio = sum(
            losses[n].item() / (self.initial_losses[n] + 1e-8)
            for n in self.task_names) / len(self.task_names)

        # GradNorm loss: drive ||w_t * grad_t|| toward avg_norm * r_t^alpha.
        gn_loss = sum(
            torch.abs(grad_norms[i] - avg_norm * (
                (losses[n].item() / (self.initial_losses[n] + 1e-8))
                / (avg_ratio + 1e-8)) ** self.alpha)
            for i, n in enumerate(self.task_names))

        self.weight_optim.zero_grad()
        gn_loss.backward()
        self.weight_optim.step()
        # Renormalize so that sum of weights stays equal to T.
        with torch.no_grad():
            self.weights.data *= len(self.task_names) / self.weights.sum()

        return total, {n: self.weights[i].item()
                       for i, n in enumerate(self.task_names)}


# ===== Trainer =====

class MTLTrainer:
    """Multi-task trainer supporting uniform, PCGrad, and GradNorm."""
    def __init__(self, model, task_configs, device='cpu', method='uniform'):
        self.model = model.to(device)
        self.device = device
        self.task_configs = {c['name']: c for c in task_configs}
        self.task_names = [c['name'] for c in task_configs]
        self.method = method
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)

        if method == 'pcgrad':
            self.pcgrad = PCGrad(self.optimizer, self.task_names)
        elif method == 'gradnorm':
            self.gradnorm = GradNorm(model, self.task_names)

    def _losses(self, outputs, targets):
        losses = {}
        for n in self.task_names:
            if self.task_configs[n]['type'] == 'classification':
                losses[n] = F.cross_entropy(outputs[n], targets[n])
            else:
                losses[n] = F.mse_loss(outputs[n], targets[n])
        return losses

    def train_epoch(self, loader, epoch):
        self.model.train()
        stats = {n: 0.0 for n in self.task_names + ['total']}
        for batch in loader:
            inputs = batch['input'].to(self.device)
            targets = {n: batch[n].to(self.device) for n in self.task_names}
            outputs = self.model(inputs)
            losses = self._losses(outputs, targets)

            if self.method == 'uniform':
                total = sum(losses.values())
                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()
            elif self.method == 'pcgrad':
                self.pcgrad.step(losses)
                total = sum(l.item() for l in losses.values())
            elif self.method == 'gradnorm':
                total, _ = self.gradnorm.step(losses)
                self.optimizer.zero_grad()
                total.backward()
                self.optimizer.step()

            for n in self.task_names:
                stats[n] += (losses[n].item()
                             if isinstance(losses[n], torch.Tensor)
                             else losses[n])
            stats['total'] += (total.item()
                               if isinstance(total, torch.Tensor) else total)
        nb = len(loader)
        return {k: v / nb for k, v in stats.items()}

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        correct = {n: 0 for n in self.task_names}
        total = 0
        for batch in loader:
            inputs = batch['input'].to(self.device)
            targets = {n: batch[n].to(self.device) for n in self.task_names}
            outputs = self.model(inputs)
            total += inputs.size(0)
            for n in self.task_names:
                if self.task_configs[n]['type'] == 'classification':
                    correct[n] += (outputs[n].argmax(1) == targets[n]).sum().item()
        return {n: 100.0 * correct[n] / total for n in self.task_names}


# ===== Demo =====

class DummyMTLDataset(Dataset):
    def __init__(self, n=1000):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        return {
            'input': torch.randn(3, 32, 32),
            'task1': torch.randint(0, 10, ()).item(),
            'task2': torch.randint(0, 5, ()).item(),
        }


def main():
    configs = [
        {'name': 'task1', 'num_classes': 10, 'type': 'classification'},
        {'name': 'task2', 'num_classes': 5,  'type': 'classification'},
    ]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loader = DataLoader(DummyMTLDataset(1000), batch_size=32, shuffle=True)
    test_loader = DataLoader(DummyMTLDataset(200), batch_size=32)

    for method in ['uniform', 'pcgrad', 'gradnorm']:
        print(f"\n{'=' * 50}\nMethod: {method}\n{'=' * 50}")
        model = MultiTaskNet(configs)
        trainer = MTLTrainer(model, configs, device, method=method)
        for epoch in range(10):
            stats = trainer.train_epoch(loader, epoch)
            metrics = trainer.evaluate(test_loader)
            print(f"Epoch {epoch+1}: "
                  + " ".join(f"{k}={v:.4f}" for k, v in stats.items())
                  + " | "
                  + " ".join(f"{k}={v:.1f}%" for k, v in metrics.items()))


if __name__ == '__main__':
    main()
```

### Code architecture

| Component        | Role                                                              |
| ---------------- | ----------------------------------------------------------------- |
| `SharedEncoder`  | First 3 ResNet-18 blocks as the shared feature extractor.         |
| `TaskHead`       | Per-task classification or regression head.                       |
| `MultiTaskNet`   | Hard parameter sharing: encoder + `ModuleDict` of heads.          |
| `PCGrad`         | Projects pairwise-conflicting gradients before averaging.         |
| `GradNorm`       | Learns per-task weights so gradient magnitudes track $\tilde{r}_t^\alpha$. |
| `MTLTrainer`     | Single interface wrapping uniform, PCGrad, and GradNorm methods.  |

---

## FAQ

**Q: When should I reach for MTL in the first place?**
Three honest reasons: (1) tasks share low-level features and you want regularization, (2) the main task is data-starved and an auxiliary task can lend supervision through the shared encoder, (3) you need to serve multiple predictions cheaply at inference. If none of those apply, MTL is the wrong tool.

**Q: How do I diagnose whether tasks are conflicting?**
Two cheap checks. (a) Log $\cos(\nabla \mathcal{L}_A, \nabla \mathcal{L}_B)$ on the shared parameters — persistently negative values mean conflict (Figure 3 right). (b) Compare per-task accuracy in the multi-task model to single-task baselines. If any task drops, you have negative transfer.

**Q: Hard sharing or soft sharing?**
Start with hard sharing — it is simpler, gives stronger regularization, and uses fewer parameters. Move to cross-stitch / MTAN only after you observe negative transfer that survives PCGrad and GradNorm.

**Q: My loss scales differ by 100x. What do I do?**
Don't hand-tune weights — it never converges. Use Uncertainty Weighting (Figure 5) as the lowest-effort fix; switch to GradNorm if you also need to track training-speed differences across tasks.

**Q: Can I combine PCGrad with GradNorm?**
Yes — they are orthogonal. GradNorm controls magnitudes, PCGrad controls directions. The standard combination is: (1) use GradNorm to compute $w_t$, (2) form weighted per-task gradients $w_t g_t$, (3) apply PCGrad to those. For 3+ tasks with mismatched scales this is the sane default.

**Q: How many auxiliary tasks should I add?**
1-2 to start, never more than 4 without first checking the affinity matrix. Beyond ~10 tasks, cluster them (Figure 7) and train one shared encoder per cluster.

---

## Summary

Multi-task learning lets you train one model that does several jobs while being smaller and often more accurate than the single-task models it replaces. The architecture is the easy part — hard parameter sharing with task-specific heads is rarely beaten. The hard part is keeping the training loop honest:

- **Loss-scale balancing** via Uncertainty Weighting or GradNorm is essentially mandatory whenever your tasks span different output types or magnitudes.
- **Gradient conflicts** affect 30-50% of updates — PCGrad (cheap) or CAGrad (better) prevent them silently degrading individual tasks.
- **Task selection** matters more than any optimizer trick — measure affinity with gradient cosine or transfer experiments before committing to a multi-task design.
- **Hard sharing first**, soft / cross-stitch only when measurements say you need it.

Next up in the series is **zero-shot learning** — classifying categories the model has never seen during training, using attribute or language descriptions to bridge the gap.

---

## References

1. Caruana, R. (1997). Multitask Learning. *Machine Learning*.
2. Misra et al. (2016). Cross-Stitch Networks for Multi-task Learning. CVPR. [arXiv:1604.03539](https://arxiv.org/abs/1604.03539)
3. Kendall et al. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses. CVPR. [arXiv:1705.07115](https://arxiv.org/abs/1705.07115)
4. Chen et al. (2018). GradNorm: Gradient Normalization for Adaptive Loss Balancing. ICML. [arXiv:1711.02257](https://arxiv.org/abs/1711.02257)
5. Liu et al. (2019). End-to-End Multi-Task Learning with Attention (MTAN). CVPR. [arXiv:1803.10704](https://arxiv.org/abs/1803.10704)
6. Standley et al. (2020). Which Tasks Should Be Learned Together in Multi-task Learning? ICML. [arXiv:1905.07553](https://arxiv.org/abs/1905.07553)
7. Yu et al. (2020). Gradient Surgery for Multi-Task Learning (PCGrad). NeurIPS. [arXiv:2001.06782](https://arxiv.org/abs/2001.06782)
8. Liu et al. (2021). Conflict-Averse Gradient Descent (CAGrad). NeurIPS. [arXiv:2110.14048](https://arxiv.org/abs/2110.14048)

---

## Series Navigation

| Part | Topic |
|------|-------|
| [1](/en/transfer-learning-1-fundamentals-and-core-concepts/) | Fundamentals and Core Concepts |
| [2](/en/transfer-learning-2-pre-training-and-fine-tuning/) | Pre-training and Fine-tuning |
| [3](/en/transfer-learning-3-domain-adaptation/) | Domain Adaptation |
| [4](/en/transfer-learning-4-few-shot-learning/) | Few-Shot Learning |
| [5](/en/transfer-learning-5-knowledge-distillation/) | Knowledge Distillation |
| **6** | **Multi-Task Learning** (you are here) |
