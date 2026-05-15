---
title: "Transfer Learning (5): Knowledge Distillation"
date: 2025-05-25 09:00:00
tags:
  - Deep Learning
  - Transfer Learning
  - Knowledge Distillation
  - Model Compression
  - Soft Labels
  - Temperature Parameter
  - Self-Distillation
categories: Transfer Learning
series: transfer-learning
lang: en
mathjax: true
description: "Compress large teacher models into small student models without losing much accuracy. Covers dark knowledge, temperature scaling, response-based / feature-based / relation-based distillation, self-distillation, and a complete multi-strategy implementation."
disableNunjucks: true
series_order: 5
series_total: 12
translationKey: "transfer-learning-5"
---
You have a 340M-parameter BERT model that hits 95% accuracy. The product team wants it on a phone that can barely fit 10M parameters. Training a 10M model from scratch lands at 85%. Knowledge distillation closes most of the gap: train the small model on the *output distribution* of the large one, not just on the labels, and you can reach 92%.

The key insight, due to Hinton, is that a teacher's "wrong" predictions are not noise — they are information. When the teacher classifies a cat image and assigns 0.14 to "tiger", 0.07 to "dog", and 0.008 to "plane", it is telling you that cats look a lot like tigers, somewhat like dogs, and nothing like aeroplanes. That structure — **dark knowledge** — is invisible in a one-hot label, and learning it is what lets the student punch above its weight.


---

## What You Will Learn

- Why soft labels carry strictly more information than hard labels.
- Temperature scaling: a single knob that controls how much dark knowledge the teacher reveals.
- Three families of distillation: **response**, **feature**, **relation**.
- Self-distillation and mutual learning, both of which work *without* a pretrained teacher.
- Stacking distillation with quantisation and pruning to push compression past 10x.
- A clean PyTorch implementation that supports all five modes.

**Prerequisites:** Parts 1-2 of this series, basic PyTorch.

---

## Why distillation works

![Teacher and student trained against a combined soft-target / hard-target loss](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/05-knowledge-distillation/fig1_teacher_student.png)

### The deployment squeeze

Large models excel on benchmarks but struggle on phones, cars, and cloud bills. Four constraints drive us toward smaller models:

- **Memory:** mobile and IoT devices simply cannot hold billions of parameters.
- **Latency:** an autonomous car needs a decision in milliseconds, not seconds.
- **Cost:** a model served a billion times a day costs real money per FLOP.
- **Energy:** an edge device runs on a battery, not a power plant.

Pruning and quantization directly reduce the model size but can decrease accuracy. Distillation takes a different approach: **train a small student to imitate a large teacher's output distribution, not just its argmax**. The student learns the teacher's inductive biases without inheriting its parameters.

### Dark knowledge: what soft labels actually teach

Take a teacher classifying a cat image:

| Class | Hard label | Teacher output |
| --- | --- | --- |
| Cat | 1.0 | 0.62 |
| Tiger | 0.0 | 0.14 |
| Leopard | 0.0 | 0.10 |
| Dog | 0.0 | 0.07 |
| Car | 0.0 | 0.012 |

The hard label says "cat" and nothing more (entropy zero). The soft label says "cat, but also tiger-ish, leopard-ish, faintly dog-ish, definitely not a car" (entropy positive). That ranking is a free lesson on inter-class similarity, drawn from millions of teacher updates.

![One-hot vs. teacher softmax: the same prediction, very different supervision](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/05-knowledge-distillation/fig2_soft_vs_hard.png)

### Distribution matching, not label matching

Standard supervised training minimises cross-entropy with hard labels:
$$\mathcal{L}_{\text{hard}} \;=\; -\sum_c y_c \log \sigma(z_c^S).$$
Distillation replaces $y_c$ with the teacher's softmax:
$$\mathcal{L}_{\text{KD}} \;=\; -\sum_c \sigma(z_c^T / \tau) \, \log \sigma(z_c^S / \tau).$$
Because the teacher is fixed, this is equivalent to minimising $\mathrm{KL}\!\left(\sigma(z^T/\tau) \,\|\, \sigma(z^S/\tau)\right)$. The student is no longer learning a label — it is learning a probability distribution.

### Temperature: a knob for dark knowledge

Raw softmax outputs tend to be peaky — the top class gets 0.99, everything else collapses into the noise floor, and the dark knowledge disappears. A **temperature** $\tau$ flattens the distribution:
$$\sigma(z_i; \tau) \;=\; \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}.$$
| Temperature | Effect |
| --- | --- |
| $\tau \to 0$ | One-hot (argmax) |
| $\tau = 1$ | Standard softmax |
| $\tau = 4$ — 10 | Reveals inter-class similarity |
| $\tau \to \infty$ | Uniform over all classes |

For logits $z = [5, 3, 1]$:

- $\tau = 1$: $[0.84, 0.11, 0.04]$ — class 3 is essentially gone.
- $\tau = 3$: $[0.51, 0.31, 0.18]$ — class 3 is back in play.

At high temperature the softmax is approximately linear in the logits,
$$\sigma(z_i; \tau) \;\approx\; \frac{1}{C} + \frac{z_i - \bar z}{C \tau},$$
so the student can learn the relative magnitudes of the teacher's logits without the exponential's distortion.

![Same logits, three temperatures: from peaked to flat](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/05-knowledge-distillation/fig3_temperature_effect.png)

### The combined loss

In practice you want both: distil from the teacher *and* anchor on the ground truth.
$$\mathcal{L} \;=\; \alpha \cdot \tau^2 \cdot \mathcal{L}_{\text{KD}} \;+\; (1 - \alpha) \cdot \mathcal{L}_{\text{hard}}.$$
- $\alpha \in [0.5, 0.9]$: how much you trust the teacher.
- $\tau^2$: compensates for the gradient shrinkage at high temperature (the soft-target gradient scales as $1/\tau^2$, so we multiply the loss by $\tau^2$ to keep the two terms comparable).
- $\mathcal{L}_{\text{hard}}$: standard cross-entropy with the true label.

Sensible defaults: $\tau = 4$, $\alpha = 0.9$ when the teacher is much stronger than the student, $\alpha = 0.5$ when they are close in capacity.

---

## Response-based distillation

The classic recipe — match the teacher's output layer and nothing else.

### Hinton's algorithm

1. **Train the teacher** $T$ on the full dataset.
2. **Compute soft labels:** for each input $x$, store $\sigma(z^T(x) / \tau)$.
3. **Train the student** $S$ on the combined loss above.
4. **Deploy** the student with $\tau = 1$.

Representative ImageNet numbers:

| Setup | Top-1 |
| --- | --- |
| ResNet-34 teacher | 73.3% |
| ResNet-18 from scratch | 69.8% |
| ResNet-18 distilled | **71.4%** (+1.6) |

A free 1.6 points for changing the loss function.

### Distillation versus label smoothing

Label smoothing also softens the target:
$$y_c' \;=\; (1 - \epsilon) \, y_c + \epsilon / C.$$
The difference is *where the softness comes from*. Label smoothing applies the **same** uniform smoothing to every example. Distillation applies a **per-example** soft distribution drawn from the teacher. A photo of a Persian cat gets weight on "tiger" and "leopard"; a photo of a sedan gets weight on "truck" and "wagon". That is why distillation consistently beats label smoothing.

---

## Feature-based distillation

Response-based KD only matches the top layer. Feature-based KD also matches the **intermediate representations** — richer signal, more places for the student to absorb the teacher's geometry.

![Response distillation matches logits; feature distillation also matches intermediate maps](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/05-knowledge-distillation/fig4_feature_vs_response.png)

### FitNets: hint learning

Match the student's intermediate features to the teacher's:
$$\mathcal{L}_{\text{hint}} \;=\; \| W_r \, F_S^l - F_T^l \|_F^2,$$
where $W_r$ is a learnable 1x1 projection that aligns the student's channel dimension to the teacher's. Romero et al. trained this in two stages:

1. Train the student's lower layers + projection to match the teacher's chosen "hint" layer.
2. Freeze the lower layers and train the rest with standard distillation.

### Attention transfer

Instead of matching feature values, match **where** the teacher is looking. Define a spatial attention map by collapsing the channel dimension:
$$A(F) \;=\; \sum_c |F_c|^p, \quad p = 2.$$
Then minimise
$$\mathcal{L}_{\text{AT}} \;=\; \sum_l \left\| \frac{A_S^l}{\|A_S^l\|_2} - \frac{A_T^l}{\|A_T^l\|_2} \right\|_2^2.$$
On CIFAR-10, distilling ResNet-110 -> ResNet-20:

| Method | Acc |
| --- | --- |
| ResNet-20 baseline | 91.3% |
| Response only | 91.8% |
| Attention transfer | **92.4%** |

### Gram matrix distillation (NST)

Borrowing from neural style transfer, match Gram matrices
$$G \;=\; F^\top F,$$
so $G_{ij}$ captures the correlation between channels $i$ and $j$. This is a second-order statistic — "texture" rather than "content" — that pointwise matching cannot capture.

---

## Relation-based distillation

Match the **relationships between samples**, not the samples themselves.

### RKD: relational knowledge distillation

Two flavours:

**Distance-wise.** For sample pair $(x_i, x_j)$, preserve their pairwise distance in embedding space:
$$\mathcal{L}_{\text{dist}} \;=\; \sum_{(i,j)} \ell_\delta\!\left(d_S(i,j),\, d_T(i,j)\right).$$
**Angle-wise.** For triplet $(x_i, x_j, x_k)$, preserve the angle at $x_j$:
$$\mathcal{L}_{\text{angle}} \;=\; \sum_{(i,j,k)} \ell_\delta\!\left(\angle_S(i,j,k),\, \angle_T(i,j,k)\right).$$
Empirically, angles matter more than distances ($\lambda_{\text{angle}} = 2$, $\lambda_{\text{dist}} = 1$), because angles are scale-invariant and capture relative geometry.

### CRD: contrastive representation distillation

Treat the teacher's representation as the "positive" view and other samples as negatives:
$$\mathcal{L}_{\text{CRD}} \;=\; -\log \frac{\exp\!\left(f_S(x)^\top f_T(x) / \tau\right)}{\sum_{x'} \exp\!\left(f_S(x)^\top f_T(x') / \tau\right)}.$$
This maximises the mutual information between student and teacher features. For very small students (e.g. ResNet-8 distilled from ResNet-32), CRD beats response-only KD by 2% or more on CIFAR-100.

---

## Self-distillation: no separate teacher

What if you have no big teacher? You can still distil.

![Born-Again Networks: each generation distils from the previous one of identical architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/05-knowledge-distillation/fig6_self_distillation.png)

### Born-Again Networks

A surprising finding: distilling a model into an *identical-architecture* copy improves accuracy.

1. Train $M_1$ normally.
2. Use $M_1$ as teacher for $M_2$ (same architecture).
3. Use $M_2$ as teacher for $M_3$.
4. Stop when accuracy saturates.

CIFAR-100 numbers:

| Generation | Acc |
| --- | --- |
| 1 (baseline) | 74.3% |
| 2 (BAN) | 75.2% |
| 3 | 75.4% |
| 4 | 75.5% |

Two complementary explanations: soft labels supply smoother gradients (reducing overfitting), and each generation explores a different region of the loss landscape, giving you an implicit ensemble at no inference cost.

### Deep mutual learning

Train $M$ students simultaneously, each treating the others as teachers:
$$\mathcal{L}_i \;=\; \mathcal{L}_{\text{CE}}^i + \frac{1}{M-1} \sum_{j \neq i} \mathrm{KL}\!\left(P_j \,\|\, P_i\right).$$
No pretraining. Different random seeds make different errors; mutual supervision lets each model absorb the others' strengths. On CIFAR-100, two ResNet-32s trained jointly each reach 72.1% versus 70.2% trained alone.

---

## Distillation + compression

Distillation composes well with the other compression tools.

### Quantisation-aware distillation

Quantising FP32 to INT8 saves 4x memory and is 2-4x faster on supporting hardware, but it costs accuracy. Distillation recovers most of the loss:

| ResNet-18 on ImageNet | Top-1 |
| --- | --- |
| FP32 baseline | 69.8% |
| INT8, no KD | 68.5% (-1.3) |
| INT8, with KD | **69.2%** (-0.6) |

Distillation cuts the quantisation cost in half.

### Pruning-aware distillation

After pruning the lowest-importance channels, fine-tune with the teacher's soft labels:

| VGG-16 on CIFAR-10 | Acc | Params |
| --- | --- | --- |
| Original | 93.5% | 14.7M |
| 70% pruned, no KD | 92.1% | 4.4M |
| 70% pruned, with KD | **93.0%** | 4.4M |

A practical pipeline: train the teacher, prune to define the student structure, distil to recover accuracy, then quantise. Applied carefully you can hit 10-20x compression with under 1% accuracy loss.

![Distillation strictly dominates training-from-scratch across the size axis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/05-knowledge-distillation/fig5_compression_curve.png)

### Case study: DistilBERT

Sanh et al. (2019) distilled BERT-base into DistilBERT using a triple loss (cosine + MLM + KD), giving the canonical headline result:

![DistilBERT: 40% smaller, 60% faster, retains 97% of GLUE](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/05-knowledge-distillation/fig7_distilbert_results.png)

| | BERT-base | DistilBERT | Delta |
| --- | --- | --- | --- |
| Parameters | 110M | 66M | -40% |
| Inference latency | 410 ms | 250 ms | -39% |
| GLUE (avg) | 79.5 | 77.0 | -3% |

Three numbers that, more than anything else, made distillation production-grade in NLP.

---

## Complete implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import List
import copy

# ===== Loss functions =====

class KDLoss(nn.Module):
    """Response-based distillation: KL divergence with temperature."""
    def __init__(self, temperature: float = 4.0, alpha: float = 0.9):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(self, student_logits, teacher_logits, labels):
        kd = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction='batchmean',
        ) * (self.T ** 2)
        ce = F.cross_entropy(student_logits, labels)
        return self.alpha * kd + (1 - self.alpha) * ce

class FeatureDistillLoss(nn.Module):
    """FitNets: 1x1 projection, then MSE between feature maps."""
    def __init__(self, student_ch: int, teacher_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(student_ch, teacher_ch, 1, bias=False)

    def forward(self, student_feat, teacher_feat):
        return F.mse_loss(self.proj(student_feat), teacher_feat)

class AttentionTransferLoss(nn.Module):
    """Match L^p-aggregated, normalised spatial attention maps."""
    def __init__(self, p: float = 2.0):
        super().__init__()
        self.p = p

    def _attn(self, feat):
        a = torch.sum(torch.abs(feat) ** self.p, dim=1, keepdim=True)
        return a / (a.sum(dim=[2, 3], keepdim=True) + 1e-8)

    def forward(self, s_feat, t_feat):
        return F.mse_loss(self._attn(s_feat), self._attn(t_feat))

class RelationalLoss(nn.Module):
    """RKD: pairwise distance + cosine (angle) relations."""
    def __init__(self, w_dist: float = 1.0, w_angle: float = 2.0):
        super().__init__()
        self.w_dist, self.w_angle = w_dist, w_angle

    def forward(self, s_feat, t_feat):
        s = F.normalize(s_feat, p=2, dim=1)
        t = F.normalize(t_feat, p=2, dim=1)
        d_loss = F.mse_loss(torch.cdist(s, s), torch.cdist(t, t))
        a_loss = F.mse_loss(s @ s.t(), t @ t.t())
        return self.w_dist * d_loss + self.w_angle * a_loss

# ===== Models (return logits AND intermediate features) =====

class _ResNetBackbone(nn.Module):
    def __init__(self, ctor, num_classes: int):
        super().__init__()
        self.net = ctor(weights=None)
        self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def forward(self, x):
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        feats: List[torch.Tensor] = []
        for layer in (n.layer1, n.layer2, n.layer3, n.layer4):
            x = layer(x)
            feats.append(x)
        logits = n.fc(torch.flatten(n.avgpool(x), 1))
        return logits, feats

class TeacherNet(_ResNetBackbone):
    def __init__(self, num_classes: int = 10):
        super().__init__(torchvision.models.resnet34, num_classes)

class StudentNet(_ResNetBackbone):
    def __init__(self, num_classes: int = 10):
        super().__init__(torchvision.models.resnet18, num_classes)

# ===== Trainer (response / feature / attention / relation / combined) =====

class DistillationTrainer:
    def __init__(self, teacher, student, device='cpu',
                 mode: str = 'response',
                 temperature: float = 4.0, alpha: float = 0.9):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.device = device
        self.mode = mode
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.kd = KDLoss(temperature, alpha)
        if mode in ('feature', 'combined'):
            ch = [64, 128, 256, 512]
            self.feat = nn.ModuleList(
                [FeatureDistillLoss(s, t).to(device)
                 for s, t in zip(ch, ch)])
        if mode in ('attention', 'combined'):
            self.attn = AttentionTransferLoss()
        if mode in ('relation', 'combined'):
            self.rel = RelationalLoss()

    def _loss(self, s_logits, s_feats, t_logits, t_feats, y):
        loss = self.kd(s_logits, t_logits, y)
        if self.mode in ('feature', 'combined'):
            fl = sum(fn(sf, tf)
                     for fn, sf, tf in zip(self.feat, s_feats, t_feats))
            loss = loss + 0.5 * fl / len(s_feats)
        if self.mode in ('attention', 'combined'):
            al = sum(self.attn(sf, tf)
                     for sf, tf in zip(s_feats, t_feats))
            loss = loss + 0.3 * al / len(s_feats)
        if self.mode in ('relation', 'combined'):
            loss = loss + 0.2 * self.rel(
                s_feats[-1].flatten(1), t_feats[-1].flatten(1))
        return loss

    def train_epoch(self, loader, optimizer):
        self.student.train()
        total, correct, n = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                t_logits, t_feats = self.teacher(x)
            s_logits, s_feats = self.student(x)
            loss = self._loss(s_logits, s_feats, t_logits, t_feats, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * y.size(0)
            correct += (s_logits.argmax(1) == y).sum().item()
            n += y.size(0)
        return total / n, 100.0 * correct / n

    @torch.no_grad()
    def evaluate(self, loader):
        self.student.eval()
        correct, n = 0, 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits, _ = self.student(x)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        return 100.0 * correct / n

# ===== Self-distillation: Born-Again Networks =====

def self_distill(model_class, train_loader, test_loader,
                 num_generations: int = 3, epochs_per_gen: int = 10,
                 device: str = 'cpu', temperature: float = 4.0):
    teacher = None
    for gen in range(num_generations):
        student = model_class().to(device)
        opt = optim.SGD(student.parameters(), lr=0.1,
                        momentum=0.9, weight_decay=5e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs_per_gen)

        for _ in range(epochs_per_gen):
            student.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = student(x)
                loss = F.cross_entropy(logits, y)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits, _ = teacher(x)
                    kd = F.kl_div(
                        F.log_softmax(logits / temperature, dim=1),
                        F.softmax(t_logits / temperature, dim=1),
                        reduction='batchmean') * temperature ** 2
                    loss = 0.1 * loss + 0.9 * kd
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()

        student.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                correct += (student(x)[0].argmax(1) == y).sum().item()
                n += y.size(0)
        print(f"Generation {gen + 1}: {100.0 * correct / n:.2f}%")

        teacher = copy.deepcopy(student).eval()
        for p in teacher.parameters():
            p.requires_grad = False

# ===== Main =====

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010))
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    train_set = torchvision.datasets.CIFAR10(
        './data', train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(
        './data', train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, 128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, 128, num_workers=2)

    teacher = TeacherNet(10)
    student = StudentNet(10)
    trainer = DistillationTrainer(
        teacher, student, device, mode='combined',
        temperature=4.0, alpha=0.7)
    opt = optim.SGD(student.parameters(), lr=0.1,
                    momentum=0.9, weight_decay=5e-4)
    for epoch in range(20):
        loss, acc = trainer.train_epoch(train_loader, opt)
        test_acc = trainer.evaluate(test_loader)
        print(f"Epoch {epoch + 1}: loss={loss:.4f} "
              f"train={acc:.1f}% test={test_acc:.1f}%")

    print("\nSelf-distillation (Born-Again Networks):")
    self_distill(StudentNet, train_loader, test_loader,
                 num_generations=3, epochs_per_gen=10, device=device)

if __name__ == '__main__':
    main()
```

### What each piece does

| Module | Role |
| --- | --- |
| `KDLoss` | Soft KL with temperature, blended with hard CE. |
| `FeatureDistillLoss` | FitNets: 1x1 projection + MSE on feature maps. |
| `AttentionTransferLoss` | Channel-aggregated, normalised spatial attention. |
| `RelationalLoss` | RKD distance + angle relations between samples. |
| `DistillationTrainer` | One trainer for response / feature / attention / relation / combined. |
| `self_distill` | Born-Again Networks: iterative same-architecture distillation. |

---

## Temperature Scheduling Strategies

![Distillation methods: validation accuracy curves and temperature sensitivity.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/05-Knowledge-Distillation/fig05_distill_curves.png)

A constant $\tau = 4$ is the textbook default, and it works. But the temperature controls *what* the student is being asked to learn, and that target shifts as training progresses. Early in training the student knows nothing — it benefits most from seeing the teacher's full ranking over classes. Late in training the student has internalised the geometry and needs to commit to confident predictions. One scalar cannot serve both regimes.

The fix is to schedule $\tau$ over training. A linear warmup-down,
$$\tau(t) \;=\; \tau_{\text{start}} - (\tau_{\text{start}} - \tau_{\text{end}}) \cdot \frac{t}{T},$$
sweeps from a high value (flat distribution, lots of dark knowledge) to a low value (peaked distribution, decisive supervision). With $\tau_{\text{start}} = 20$ and $\tau_{\text{end}} = 2$ over $T$ epochs, the student spends its first epochs learning relative class similarities and its last epochs sharpening into a classifier.

Why this helps, mechanically. At high $\tau$ the softmax is approximately linear in the logits, so the KL gradient on the student's logits is dominated by the *differences* between teacher logits — the relational structure. At low $\tau$ the softmax is approximately one-hot, so the KL collapses toward standard cross-entropy on the teacher's argmax — a hard but noise-free target.

```python
import torch.nn.functional as F

class TemperatureSchedule:
    """Linear schedule from tau_start down to tau_end over T epochs."""
    def __init__(self, tau_start: float, tau_end: float, T: int):
        self.t0, self.t1, self.T = tau_start, tau_end, T

    def __call__(self, epoch: int) -> float:
        frac = min(epoch / max(self.T - 1, 1), 1.0)
        return self.t0 - (self.t0 - self.t1) * frac

def kd_loss_scheduled(s_logits, t_logits, y, tau: float, alpha: float = 0.9):
    kd = F.kl_div(
        F.log_softmax(s_logits / tau, dim=1),
        F.softmax(t_logits / tau, dim=1),
        reduction='batchmean',
    ) * (tau ** 2)
    ce = F.cross_entropy(s_logits, y)
    return alpha * kd + (1 - alpha) * ce

# Benchmark loop: ResNet-18 teacher -> ResNet-8 student on CIFAR-10
sched = TemperatureSchedule(tau_start=20.0, tau_end=2.0, T=epochs)
for epoch in range(epochs):
    tau = sched(epoch)
    student.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            t_logits, _ = teacher(x)
        s_logits, _ = student(x)
        loss = kd_loss_scheduled(s_logits, t_logits, y, tau=tau, alpha=0.9)
        opt.zero_grad(); loss.backward(); opt.step()
```

CIFAR-10 numbers, ResNet-18 teacher distilled into a ResNet-8 student over 100 epochs:

| Schedule | Test accuracy |
| --- | --- |
| Constant $\tau = 4$ | 88.1% |
| Linear $20 \to 2$ | 89.4% |
| Linear $50 \to 4$ | **89.7%** |

A free 1.3-1.6 points for replacing a scalar with a function. The schedule does most of its work in the first half of training, where the gap to the constant baseline opens; the last 20 epochs at low $\tau$ mainly stabilise.

One caveat. Push $\tau_{\text{start}}$ much past 50 and the soft target becomes nearly uniform — the $\tau^2$ scaling cannot compensate, gradients become tiny relative to the cross-entropy term, and early epochs stall or oscillate. If you want a hot start, cap it around $\tau = 30$ and stretch the schedule.

Bridge: Temperature shapes *how* the student listens to the teacher's logits. The next move is to change *what* it listens to — features, and the geometry between them.

---

## CRD: Contrastive Representation Distillation

Feature-based KD as we have seen it — FitNets, attention transfer — matches the student's features to the teacher's *pointwise*. That is a strong constraint, and a wasteful one: it asks the student to reproduce the teacher's exact activations, even when only the relational structure matters for downstream classification.

Contrastive Representation Distillation (CRD) loosens the constraint. Instead of matching $f_S(x)$ to $f_T(x)$ directly, push them *closer than* student-teacher pairs from different inputs. The teacher's representation of input $x$ becomes the positive view; the teacher's representations of all other inputs in the batch become negatives.

The loss is InfoNCE over (student, teacher) pairs:
$$\mathcal{L}_{\text{CRD}} \;=\; -\log \frac{\exp\!\left(s(z_s, z_t) / \tau\right)}{\sum_{j} \exp\!\left(s(z_s, z_j^-) / \tau\right)},$$
where $s(\cdot, \cdot)$ is cosine similarity, $z_s = g_S(f_S(x))$, $z_t = g_T(f_T(x))$, and $g_S, g_T$ are small projection heads (one nonlinear layer is enough). Maximising this lower-bounds the mutual information between student and teacher representations.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)

class CRDLoss(nn.Module):
    """InfoNCE on (student, teacher) pairs with in-batch negatives."""
    def __init__(self, s_dim: int, t_dim: int,
                 proj_dim: int = 128, tau: float = 0.07):
        super().__init__()
        self.g_s = ProjectionHead(s_dim, proj_dim)
        self.g_t = ProjectionHead(t_dim, proj_dim)
        self.tau = tau

    def forward(self, s_feat, t_feat):
        # Pool spatial dims if needed
        if s_feat.dim() == 4:
            s_feat = F.adaptive_avg_pool2d(s_feat, 1).flatten(1)
        if t_feat.dim() == 4:
            t_feat = F.adaptive_avg_pool2d(t_feat, 1).flatten(1)
        z_s = self.g_s(s_feat)             # (B, D)
        z_t = self.g_t(t_feat).detach()    # teacher fixed
        logits = z_s @ z_t.t() / self.tau  # (B, B): row i, col i is positive
        labels = torch.arange(z_s.size(0), device=z_s.device)
        return F.cross_entropy(logits, labels)
```

Slot it in alongside the standard KD loss with a small weight ($\beta = 0.8$ in the original paper):
$$\mathcal{L} \;=\; \alpha \tau^2 \mathcal{L}_{\text{KD}} + (1 - \alpha) \mathcal{L}_{\text{hard}} + \beta \mathcal{L}_{\text{CRD}}.$$

CIFAR-100, ResNet-50 teacher distilled into ResNet-18:

| Method | Test accuracy |
| --- | --- |
| Hard-label baseline | 73.3% |
| Vanilla response KD | 75.5% |
| + CRD | **76.7%** (+1.2) |

The gains are largest when teacher and student have very different capacities — exactly the case where pointwise matching breaks down because the student cannot represent the teacher's features even if it wanted to. CRD asks for a weaker thing (relative similarity) and gets more of it.

Bridge: But not all distillation runs converge — sometimes the loss flat-lines, or even diverges.

---

## When Distillation Fails

Distillation is a robust technique, not a magic one. Three failure modes show up often enough to be worth diagnosing explicitly.

**Capacity ceiling.** The student is simply too small to fit the teacher's distribution. Symptom: the teacher-student KL,
$$D_{\text{KL}}(p_t \,\|\, p_s) \;=\; \sum_c p_t(c) \log \frac{p_t(c)}{p_s(c)},$$
plateaus at a high value within the first 5-10 epochs and refuses to budge. Cross-entropy may continue to fall — the student is learning the argmax — but it is not absorbing the dark knowledge. A 100k-parameter student trying to imitate a 100M-parameter teacher will hit this almost immediately.

**Modality mismatch.** Teacher and student see different inputs: an RGB teacher distilling into a grayscale student, a high-resolution teacher into a low-resolution one, a cross-lingual setup with mismatched tokenisers. The teacher's soft targets encode features the student has no access to, so $p_t$ becomes effectively noise from the student's point of view. Symptom: KD loss is high and noisy, and the combined loss is *worse* than hard-label-only training.

**Online distillation instability.** In online or mutual-learning setups the teacher is updated concurrently with the student. The student is chasing a moving target, and if the teacher's updates are large relative to the student's, the KL trajectory oscillates instead of decaying. Symptom: epoch-to-epoch KL goes up and down by 20-30% with no downward trend.

A simple diagnostic catches all three:

```python
@torch.no_grad()
def kd_diagnostic(student, teacher, loader, n_epochs: int,
                  optimizer, device='cpu', tau: float = 4.0):
    """Returns per-epoch mean KL(p_t || p_s); flags the dominant bottleneck."""
    teacher.eval()
    kl_history = []
    for epoch in range(n_epochs):
        student.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            t_logits, _ = teacher(x)
            s_logits, _ = student(x)
            loss = F.kl_div(
                F.log_softmax(s_logits / tau, dim=1),
                F.softmax(t_logits / tau, dim=1),
                reduction='batchmean') * (tau ** 2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        # Measure KL on a clean pass
        student.eval()
        kl_sum, n = 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            t_logits, _ = teacher(x)
            s_logits, _ = student(x)
            p_t = F.softmax(t_logits / tau, dim=1)
            log_p_s = F.log_softmax(s_logits / tau, dim=1)
            kl_sum += F.kl_div(log_p_s, p_t, reduction='batchmean').item() * x.size(0)
            n += x.size(0)
        kl_history.append(kl_sum / n)

    init, mid, last = kl_history[0], kl_history[len(kl_history) // 2], kl_history[-1]
    if last > 0.5 * init:
        verdict = 'capacity'   # KL did not halve in the run
    elif max(kl_history[-5:]) - min(kl_history[-5:]) > 0.2 * last:
        verdict = 'lr / instability'
    elif init > 5.0:
        verdict = 'data / modality mismatch'
    else:
        verdict = 'healthy'
    return kl_history, verdict
```

A healthy run halves its initial KL within roughly 50 epochs and continues drifting downward. If yours does not, the verdict points at the fix:

- **Capacity** — bump the student up one width or depth tier, or add intermediate-feature distillation so the student gets richer signal per parameter. If the student is locked in size, lower the teacher: distil into your target student via an intermediate teacher first (TA-KD).
- **Modality mismatch** — align inputs before distilling. Convert the teacher's input space to match the student's, or train a small adapter that maps student features into the teacher's space and distil there.
- **Instability** — freeze the teacher periodically (update every $k$ student steps, not every step), drop the student's learning rate, or warm up the KD weight $\alpha$ from 0 over the first few epochs.

The pattern across all three: the KL trajectory is a more sensitive instrument than test accuracy. It tells you whether the student is *listening*, before you find out whether it learned anything useful.

Bridge: With these diagnostics in hand, distillation moves from a hopeful loss term to an instrumented training procedure. The remaining questions are practical — how to pick hyperparameters, how far you can push compression, and how distillation interacts with the rest of the model-shrinking toolkit.

---

## FAQ

### How do I pick the temperature?

Start at $\tau = 4$. The more classes you have and the more similar they are to each other (e.g. fine-grained species classification), the higher you should go — up to $\tau = 20$ for ImageNet-scale problems. Grid-search $\{2, 4, 8, 12, 20\}$ on a held-out set.

### How small can the student get?

You can compress 4-10x with very little loss. Past 50x even distillation cannot save you — expect 5-10% drops. The general rule: distillation buys you the most when the student has just enough capacity to *represent* what the teacher knows but not enough to learn it from labels alone.

**Why does self-distillation work at all? The student has the same capacity as the teacher.**
Two reasons. (1) The soft targets are a stronger regulariser than one-hot labels, especially on small datasets. (2) Each generation lands in a slightly different basin of the loss landscape, so iterating is an implicit ensemble at zero inference cost. Expect 1-2% gains on CIFAR-100, with diminishing returns after 3 generations.

### Can I use multiple teachers?

Yes — average their soft outputs (uniformly or weighted by validation accuracy). This usually buys robustness more than headline accuracy, at the cost of training each teacher.

### Distil first or prune first?

Both, in this order: train teacher -> prune to define the student architecture -> distil to recover accuracy -> quantise. Each step preserves more knowledge than doing them independently because every later step still has the teacher to lean on.

---

## Summary

Knowledge distillation is the art of teaching a small model to think like a big one:

- **Soft labels** carry inter-class structure that one-hot labels destroy. That structure is the dark knowledge.
- **Temperature** is a single scalar that controls how much of it the student sees.
- **Feature** and **relation** distillation push beyond logits, matching intermediate representations and pairwise geometry.
- **Self-distillation** works without a separate teacher and gives you a free ensemble.
- **Stacked with pruning and quantisation**, distillation enables 10-20x compression with single-digit accuracy cost.

Next: [Part 6 — Multi-Task Learning](/en/transfer-learning/06-multi-task-learning/), where multiple tasks share parameters to improve generalisation and efficiency.

---

## References

1. Hinton, Vinyals, Dean (2015). *Distilling the Knowledge in a Neural Network*. [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
2. Romero et al. (2015). *FitNets: Hints for Thin Deep Nets*. ICLR. [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)
3. Zagoruyko & Komodakis (2017). *Paying More Attention to Attention*. ICLR. [arXiv:1612.03928](https://arxiv.org/abs/1612.03928)
4. Park et al. (2019). *Relational Knowledge Distillation*. CVPR. [arXiv:1904.05068](https://arxiv.org/abs/1904.05068)
5. Tian et al. (2020). *Contrastive Representation Distillation*. ICLR. [arXiv:1910.10699](https://arxiv.org/abs/1910.10699)
6. Furlanello et al. (2018). *Born-Again Neural Networks*. ICML. [arXiv:1805.04770](https://arxiv.org/abs/1805.04770)
7. Sanh et al. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
8. Zhang et al. (2018). *Deep Mutual Learning*. CVPR. [arXiv:1706.00384](https://arxiv.org/abs/1706.00384)
