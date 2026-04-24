---
title: "Transfer Learning (3): Domain Adaptation"
date: 2024-08-03 09:00:00
tags:
  - Deep Learning
  - Transfer Learning
  - Domain Adaptation
  - MMD
  - DANN
  - Distribution Shift
  - Adversarial Learning
categories:
  - Transfer Learning
series:
  name: "Transfer Learning"
  part: 3
  total: 12
lang: en
mathjax: true
description: "A practical guide to domain adaptation: covariate shift, label shift, DANN with gradient reversal, MMD alignment, CORAL, self-training, AdaBN, and a complete DANN implementation."
---

Your autonomous-driving stack works perfectly on sunny California freeways. Then it rains in Seattle. Top-1 accuracy drops from 95% to 70%. The model did not get worse — the *data distribution shifted*, and your training set never told it what wet asphalt looks like at dusk.

This is the everyday problem of **domain adaptation**: you have abundant labelled data in one distribution (the *source*) and unlabelled data in another (the *target*), and you need the model to perform on the target. This article shows you how, from first-principles theory to a working DANN implementation.

## What you will learn

- Three flavours of distribution shift — covariate, label, concept — and how each is fixed
- The Ben-David bound: why adaptation is possible, and the precise quantity it lets you reduce
- DANN: adversarial alignment with the gradient reversal layer, in one backward pass
- MMD and CORAL: explicit, non-adversarial distribution-matching losses
- Self-training, AdaBN, CycleGAN, ADDA — the rest of the modern toolbox
- A complete DANN implementation in PyTorch
- A decision tree for picking a method, plus benchmark numbers on Office-31 and DomainNet

**Prerequisites:** Parts 1–2 of this series, basic familiarity with GAN-style adversarial training.

---

## 1. Three Faces of Distribution Shift

A **domain** is a feature space $\mathcal{X}$ with a marginal distribution $P(X)$. A **task** is a label space $\mathcal{Y}$ with a conditional distribution $P(Y \mid X)$. Domain adaptation studies what happens when the source and target disagree on one of these.

| Setting | Source | Target | Goal |
|---|---|---|---|
| **Source domain** $\mathcal{D}_S$ | many labelled $(x_i, y_i)$ | — | — |
| **Target domain** $\mathcal{D}_T$ | — | mostly unlabelled $x_j$ | learn $f: \mathcal{X} \to \mathcal{Y}$ that works on $\mathcal{D}_T$ |

![Source vs target distribution alignment](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/03-domain-adaptation/fig1_distribution_shift.png)

The figure is the entire game in one picture: before adaptation, the source-trained boundary slices through empty target space; after adaptation, both domains share a feature manifold and the same boundary works.

### 1.1 Covariate shift — the input distribution moved

$$P_S(X) \neq P_T(X), \qquad P_S(Y \mid X) = P_T(Y \mid X)$$

The *labelling rule* is unchanged; only what you observe is different. Examples:

- A spam filter trained on 2020 email and deployed in 2026: topics drift, but spam is still spam.
- CT scans from a Siemens scanner used to evaluate scans from a GE machine: the imaging characteristics differ, but radiologists score them the same way.

**Standard fix — importance weighting.** Reweight every source sample by the density ratio $w(x) = P_T(x) / P_S(x)$. The weighted source ERM then estimates the *target* risk:

$$\mathbb{E}_{P_T}[\ell(f(X), Y)] = \mathbb{E}_{P_S}\!\left[\frac{P_T(X)}{P_S(X)}\,\ell(f(X), Y)\right].$$

Estimating densities in high dimensions is hopeless, so practitioners estimate the *ratio* directly with KLIEP, uLSIF, or a probabilistic classifier (Bayes-optimal classifier between source and target gives you the ratio for free).

### 1.2 Label shift — the prevalence moved

$$P_S(Y) \neq P_T(Y), \qquad P_S(X \mid Y) = P_T(X \mid Y)$$

Class-conditional appearance is unchanged; only base rates differ. Examples:

- An ICU model deployed in outpatient clinics where disease prevalence is much lower.
- A recommender trained on a young-skewing pilot, deployed across all age cohorts.

**Standard fix.** Estimate the target prior $P_T(Y)$ by EM on unlabelled target data (BBSE / RLLS work well), then rescale each source-trained probability by $P_T(y) / P_S(y)$ and renormalise.

### 1.3 Concept shift — the rule itself moved

$$P_S(Y \mid X) \neq P_T(Y \mid X)$$

This is the hard case. "Sick" is positive in a music review and negative in a product review even though the *word* is identical. With no target labels at all, no method can untangle this — concept shift demands at least a few labelled target examples (the *semi-supervised* DA setting).

---

## 2. Theory: the Ben-David Bound

Why is adaptation possible at all? The classical answer is the bound of Ben-David et al. (2010). For any hypothesis $h$ in class $\mathcal{H}$:

$$
\epsilon_T(h) \;\leq\; \epsilon_S(h) \;+\; \tfrac{1}{2}\, d_{\mathcal{H}\Delta\mathcal{H}}(\mathcal{D}_S, \mathcal{D}_T) \;+\; \lambda^{*}.
$$

| Term | Meaning | What you can do about it |
|---|---|---|
| $\epsilon_S(h)$ | source-domain error | train better on the source |
| $d_{\mathcal{H}\Delta\mathcal{H}}$ | symmetric-difference divergence between domains | **this is what domain adaptation reduces** |
| $\lambda^{*}$ | error of the best joint predictor | irreducible — if it is large, no method will save you |

Two takeaways:

1. **Adaptation is bounded by an oracle.** If source and target tasks are fundamentally different ($\lambda^*$ large), you are out of luck — you need new labels, not a fancier loss.
2. **Domain divergence has a tractable proxy.** Train a binary classifier to distinguish source from target features. If it gets near 50% accuracy, your features are domain-invariant. *This is exactly the mechanism DANN automates.*

---

## 3. DANN — Adversarial Alignment in One Backward Pass

**Domain-Adversarial Neural Network** (Ganin et al., 2016) is the most influential adversarial method, and the cleanest implementation of "minimise the domain divergence proxy".

![DANN architecture with Gradient Reversal Layer](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/03-domain-adaptation/fig2_dann_architecture.png)

### 3.1 Three subnetworks, one shared trunk

| Subnet | Role | Trained on |
|---|---|---|
| **Feature extractor** $G_f$ | maps $x$ to $f = G_f(x)$ | both domains |
| **Label predictor** $G_y$ | classifies $f \to \hat{y}$ | source labels |
| **Domain discriminator** $G_d$ | classifies $f \to$ source/target | both domains |

The objective is a minimax:

$$
\min_{G_f,\, G_y}\; \max_{G_d}\quad \mathcal{L}_y(G_y \circ G_f) \;-\; \lambda\, \mathcal{L}_d(G_d \circ G_f).
$$

$G_d$ wants to tell the domains apart; $G_f$ wants to fool $G_d$ while still letting $G_y$ classify the source correctly.

### 3.2 The Gradient Reversal Layer (GRL)

A naive minimax requires alternating optimisation, which is fragile (think early GANs). DANN's contribution is to make the whole system trainable in **one** backward pass via the Gradient Reversal Layer:

$$
\text{forward: }\; \text{GRL}(x) = x, \qquad
\text{backward: }\; \frac{\partial\,\text{GRL}}{\partial x} = -\lambda\, I.
$$

GRL sits on the path from features to the domain head. During backprop, the discriminator's gradient flips sign before reaching $G_f$, so the same `loss.backward()` call:

- updates $G_y$ to classify better (normal gradients),
- updates $G_d$ to discriminate better (normal gradients),
- updates $G_f$ to *confuse* $G_d$ (reversed gradients on the domain term) while still helping $G_y$.

No alternating training, no separate optimisers, no manual freezing.

### 3.3 The adversarial weight schedule

DANN does not turn $\lambda$ on at full strength — that destroys early learning. Instead it follows a sigmoid ramp:

$$\lambda_p = \frac{2}{1 + \exp(-\gamma p)} - 1, \qquad \gamma \approx 10,$$

where $p \in [0, 1]$ is training progress. Early on ($\lambda \approx 0$), the network just learns good source features. As training proceeds ($\lambda \to 1$), domain alignment kicks in. Skipping this schedule is the single most common cause of "DANN trains but does worse than source-only".

---

## 4. MMD — Matching Means in an RKHS

Adversarial alignment is powerful but unstable. The non-adversarial alternative is to define an explicit distance between distributions and minimise it directly. **Maximum Mean Discrepancy** (Gretton et al., 2012) is the standard choice.

![Maximum Mean Discrepancy: kernel mean embeddings](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/03-domain-adaptation/fig3_mmd_kernel.png)

### 4.1 The idea

A kernel $k(x, y) = \langle \phi(x), \phi(y) \rangle_{\mathcal{H}}$ implicitly maps each sample into a (possibly infinite-dimensional) RKHS $\mathcal{H}$. The **kernel mean embedding** of a distribution $P$ is the average feature

$$\mu_P = \mathbb{E}_{X \sim P}[\phi(X)] \;\in\; \mathcal{H}.$$

For *characteristic* kernels (the Gaussian RBF being the canonical example), the map $P \mapsto \mu_P$ is injective — two distributions are equal iff their kernel means are. So we can measure how different two distributions are by the RKHS distance of their means:

$$\text{MMD}^2(P_S, P_T) = \|\mu_{P_S} - \mu_{P_T}\|_{\mathcal{H}}^2.$$

The figure shows this graphically: even when raw histograms overlap a little, the kernel mean embeddings make the gap explicit, and the shaded area is exactly $\text{MMD}^2$.

### 4.2 The estimator you actually compute

Because the embedding is implicit, expand the squared norm and the inner product becomes a kernel evaluation:

$$
\widehat{\text{MMD}}^2 = \frac{1}{n_s^2}\sum_{i,j} k(x_i^s, x_j^s) + \frac{1}{n_t^2}\sum_{i,j} k(x_i^t, x_j^t) - \frac{2}{n_s n_t}\sum_{i,j} k(x_i^s, x_j^t).
$$

This is differentiable in the features, so you can drop it straight into a deep network as an extra loss:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \widehat{\text{MMD}}^2\!\big(G_f(X_S),\, G_f(X_T)\big).$$

This is **DAN / DDC** (Long et al., 2015; Tzeng et al., 2014).

### 4.3 Practical tips

- **Use multi-kernel MMD.** A mixture $k = \sum_u \beta_u k_{\sigma_u}$ of Gaussian RBFs at several bandwidths is robust to bandwidth misspecification.
- **Median heuristic for $\sigma$.** Set the bandwidth to the median pairwise distance in the batch — cheap, robust, almost always good enough.
- **Apply MMD to deeper layers.** Lower layers carry domain-specific texture; the abstraction at the top is what you want aligned.

### 4.4 MMD vs DANN at a glance

| | MMD | DANN |
|---|---|---|
| Distance | Kernel-based RKHS norm | Jensen–Shannon (via discriminator) |
| Optimisation | Direct minimisation | Adversarial minimax (GRL) |
| Stability | Very stable | Sometimes oscillates |
| Expressiveness | Tied to kernel choice | More flexible |
| Best when | Small/medium gap, less data | Large gap, abundant data |

A reasonable default workflow: try MMD first; switch to DANN if MMD plateaus.

---

## 5. CORAL — Aligning Second-Order Statistics

If matching means is good, matching means and *covariances* is often better. **CORAL** (Sun & Saenko, 2016) does exactly this.

![CORAL covariance alignment](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/03-domain-adaptation/fig4_coral_covariance.png)

Let $C_S$ and $C_T$ be the feature covariance matrices in source and target. The CORAL loss is

$$\mathcal{L}_{\text{CORAL}} = \frac{1}{4 d^2} \|C_S - C_T\|_F^2.$$

**Intuition — whitening + recolouring.** Multiplying the source features by $C_S^{-1/2} C_T^{1/2}$ first removes the source's covariance fingerprint, then paints on the target's. Deep CORAL just adds the loss above to a deep network and lets the gradients do the same job implicitly.

CORAL is dirt cheap (one matrix and one Frobenius norm per batch), entirely deterministic, and surprisingly competitive on mild shifts. It is a great baseline before reaching for MMD or DANN.

---

## 6. AdaBN — The Free Lunch You Should Always Try First

The simplest domain adaptation trick of all: **recompute batch-norm statistics on the target.**

Standard BN at test time uses the running mean and variance accumulated during source training. If the target has a different distribution, those statistics are wrong, and they sit between every conv layer and the next non-linearity. AdaBN (Li et al., 2017):

1. Train normally on source.
2. With weights frozen, run forward passes over unlabelled target data and recompute $\mu_T, \sigma_T^2$ for every BN layer.
3. At deployment, swap source statistics for target ones.

Cost: minutes. Code change: replacing a few `BatchNorm` running stats. Effect: routinely reclaims 2–10 points of accuracy under covariate shift. Always try this *first* before any fancier method.

---

## 7. GAN-Based and Pixel-Level Adaptation

Sometimes the gap is so visual — synthetic to real, day to night — that aligning *features* is too late. You want to translate the inputs themselves.

- **CycleGAN** learns two generators $G: \mathcal{X}_S \to \mathcal{X}_T$ and $F: \mathcal{X}_T \to \mathcal{X}_S$ subject to cycle consistency $F(G(x)) \approx x$. Translate source images into target style, then train your classifier on the translated images with the original source labels. Beware: cycle consistency does *not* guarantee semantic preservation; combine with a perceptual or identity loss for safety.
- **ADDA** decouples the source and target encoders. Stage 1: train a source encoder + classifier normally. Stage 2: initialise a *target* encoder from the source, then adapt it adversarially against a domain discriminator while keeping the classifier frozen. Stage 3: at test time, route target inputs through the *target* encoder and the *source* classifier. This asymmetry gives ADDA more capacity than DANN at the cost of an extra training stage.

---

## 8. Self-Training — Bootstrapping Labels on the Target

Adversarial and statistical alignment treat the target as one undifferentiated cloud. **Self-training** (also called pseudo-labelling) goes further: it uses your current model to produce target labels and then trains on them.

![Self-training / pseudo-labelling](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/03-domain-adaptation/fig6_self_training.png)

The loop is:

1. Train $f$ on the source.
2. Predict on every target sample; keep only those where $\max_y f(x)_y > \tau$ (a high confidence threshold).
3. Treat the kept (input, prediction) pairs as new labelled data and retrain.
4. Iterate.

Self-training is powerful and underestimated, but it has one infamous failure mode: **confirmation bias**. Wrong but confident predictions get re-fed into training and amplified. The standard mitigations are:

- a high threshold $\tau$ (typically 0.9+),
- class-balanced selection (cap the number kept per class),
- consistency regularisation under augmentations (FixMatch-style),
- restarting from the source model at each round rather than from the previous self-trained one.

---

## 9. Decision Tree — Which Method, When?

```
1. Do you have any target-domain labels?
   ├─ yes → semi-supervised DA: fine-tune + importance weighting
   └─ no  → step 2

2. Where is the shift?
   ├─ P(X) differs only          → step 3
   ├─ P(Y) differs only          → label-shift correction (BBSE / EM)
   └─ P(Y|X) differs (concept)   → you need some target labels

3. How big is the visual / feature gap?
   ├─ tiny   → AdaBN (always try first)
   ├─ small  → AdaBN + Deep CORAL
   ├─ medium → MMD (DAN) or DANN
   ├─ large  → DANN / CDAN, or pixel-level (CycleGAN, ADDA)
   └─ enormous (sim → real) → CycleGAN + ADDA + self-training
```

In practice a strong pipeline often *combines* methods: AdaBN for the easy gains, MMD or DANN for feature alignment, then a self-training round for the last few points.

---

## 10. Benchmarks — How Much Does This Actually Help?

![Office-31 and DomainNet benchmark](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/03-domain-adaptation/fig7_office31_benchmark.png)

The numbers are representative literature averages with a ResNet-50 backbone. Two things worth noticing:

- **The biggest jump is from "nothing" to "anything".** Even AdaBN closes a meaningful chunk of the gap. Doing *something* matters far more than choosing the perfect method.
- **DomainNet is genuinely harder than Office-31.** A 40% accuracy on DomainNet still represents a strong method — the dataset has 345 classes across 6 wildly different visual styles. Always interpret DA accuracies relative to a source-only baseline, not in absolute terms.

---

## 11. Where Domain Adaptation Earns Its Keep

- **Medical imaging** — Siemens vs GE scanners, 1.5T vs 3T MRI, hospital A vs hospital B.
- **Autonomous driving** — sunny to rainy, city A to city B, simulation to real.
- **Recommendation** — country to country, year to year, web to mobile.
- **NLP** — movie reviews to product reviews, news to social, formal to informal.
- **Sim-to-real** — synthetic data to real sensor data in robotics and self-driving.

The common pattern: **source labels are abundant, target labels are expensive or impossible, and the model has to ship anyway.**

---

## 12. Visualising the Effect — t-SNE Before and After

A standard sanity check after training a DA model: project source and target features through t-SNE. Before adaptation, samples cluster by *domain*; after, they cluster by *class*.

![t-SNE before vs after domain adaptation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/03-domain-adaptation/fig5_tsne_before_after.png)

If your "after" plot still shows two domain blobs, alignment failed. If it shows one blob with class structure, alignment worked. This single picture is more diagnostic than any single number.

---

## 13. Complete Implementation: DANN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Function
import numpy as np
from sklearn.metrics import accuracy_score


class GradientReversalFunction(Function):
    """Identity in the forward pass, negates the gradient in the backward pass."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0

    def set_lambda(self, val):
        self.lambda_ = val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=28 * 28, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class LabelPredictor(nn.Module):
    def __init__(self, feature_dim=256, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class DomainDiscriminator(nn.Module):
    def __init__(self, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class DANN(nn.Module):
    """Domain-Adversarial Neural Network."""

    def __init__(self, input_dim=28 * 28, hidden_dim=256, num_classes=10):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim)
        self.label_predictor = LabelPredictor(hidden_dim, num_classes)
        self.domain_discriminator = DomainDiscriminator(hidden_dim)
        self.grl = GradientReversalLayer()

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_logits = self.label_predictor(features)
        self.grl.set_lambda(alpha)
        domain_logits = self.domain_discriminator(self.grl(features))
        return class_logits, domain_logits


class DANNTrainer:
    def __init__(self, model, source_loader, target_loader, test_loader,
                 num_epochs=100, lr=1e-3, device="cpu", gamma=10.0):
        self.model = model.to(device)
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.device = device
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCELoss()

    def _adaptive_lambda(self, epoch):
        # Sigmoid ramp from 0 -> 1 across training.
        p = epoch / self.num_epochs
        return 2.0 / (1.0 + np.exp(-self.gamma * p)) - 1.0

    def train_epoch(self, epoch):
        self.model.train()
        source_iter = iter(self.source_loader)
        target_iter = iter(self.target_loader)
        n_batches = min(len(self.source_loader), len(self.target_loader))
        total_loss = 0.0
        lambda_p = self._adaptive_lambda(epoch)

        for _ in range(n_batches):
            try:
                src_x, src_y = next(source_iter)
            except StopIteration:
                source_iter = iter(self.source_loader)
                src_x, src_y = next(source_iter)
            try:
                tgt_x, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_loader)
                tgt_x, _ = next(target_iter)

            src_x = src_x.to(self.device)
            src_y = src_y.to(self.device)
            tgt_x = tgt_x.to(self.device)

            # Forward — both heads, both domains.
            src_class_logits, src_dom_logits = self.model(src_x, lambda_p)
            _, tgt_dom_logits = self.model(tgt_x, lambda_p)

            # Source classification loss.
            class_loss = self.class_criterion(src_class_logits, src_y)
            # Domain discrimination loss (source = 1, target = 0).
            d_loss_s = self.domain_criterion(
                src_dom_logits, torch.ones_like(src_dom_logits))
            d_loss_t = self.domain_criterion(
                tgt_dom_logits, torch.zeros_like(tgt_dom_logits))
            domain_loss = d_loss_s + d_loss_t

            loss = class_loss + domain_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        preds, labels = [], []
        for x, y in self.test_loader:
            x = x.to(self.device)
            logits, _ = self.model(x, alpha=0.0)
            preds.extend(logits.argmax(dim=1).cpu().numpy())
            labels.extend(y.numpy())
        return accuracy_score(labels, preds)

    def train(self):
        best = 0.0
        for epoch in range(self.num_epochs):
            loss = self.train_epoch(epoch)
            acc = self.evaluate()
            if (epoch + 1) % 10 == 0:
                lam = self._adaptive_lambda(epoch)
                print(f"epoch {epoch + 1:3d}  loss={loss:.4f}  "
                      f"target_acc={acc:.4f}  lambda={lam:.3f}")
            best = max(best, acc)
        print(f"best target accuracy: {best:.4f}")


def main():
    N, D, C = 10000, 28 * 28, 10
    # Simulated source and target with distribution shift.
    src_x = torch.randn(N, 1, 28, 28)
    src_y = torch.randint(0, C, (N,))
    tgt_x = torch.randn(N, 1, 28, 28) + 0.5     # shifted
    tgt_y = torch.randint(0, C, (N,))           # not used in training
    test_x = torch.randn(2000, 1, 28, 28) + 0.5
    test_y = torch.randint(0, C, (2000,))

    BS = 128
    src_loader = DataLoader(TensorDataset(src_x, src_y), BS, shuffle=True)
    tgt_loader = DataLoader(TensorDataset(tgt_x, tgt_y), BS, shuffle=True)
    test_loader = DataLoader(TensorDataset(test_x, test_y), BS)

    model = DANN(D, 256, C)
    trainer = DANNTrainer(model, src_loader, tgt_loader, test_loader,
                          num_epochs=100, lr=1e-3)
    trainer.train()


if __name__ == "__main__":
    main()
```

### How this code works

| Component | Role |
|---|---|
| `GradientReversalLayer` | Identity forward, negated-gradient backward — turns the minimax into a single backward pass. |
| `_adaptive_lambda` | Sigmoid ramp $\frac{2}{1 + e^{-\gamma p}} - 1$ — start small so the network learns features first. |
| `class_loss` | Standard cross-entropy on source labels only (no target labels used). |
| `domain_loss` | BCE: source = 1, target = 0 — trains the discriminator. |
| GRL + domain head | Reversed gradients flow back to $G_f$ → it learns to *hide* the domain. |
| `evaluate(alpha=0)` | At test time we set $\lambda = 0$; the GRL is irrelevant — only the classification head is used. |

---

## Summary

Domain adaptation tackles the most practical problem in transfer learning: training data and deployment data come from different distributions. The toolkit, in roughly increasing order of effort:

- **AdaBN** — recompute batch-norm statistics on target; free, no retraining, always try first.
- **CORAL** — match source and target *covariance* matrices; cheap, deterministic.
- **MMD (DAN)** — match kernel mean embeddings; stable, principled, multi-kernel default.
- **DANN** — adversarial domain alignment via the gradient reversal layer; one backward pass.
- **CDAN / ADDA** — more flexible variants for larger gaps.
- **CycleGAN** — pixel-level translation when feature alignment is not enough.
- **Self-training** — pseudo-labels with a confidence gate; the last few points of accuracy.

The Ben-David bound tells you what is possible: shrink the source error and the domain divergence, and target error follows — *as long as* the joint optimal error is small. If it is not, no amount of alignment will help; you need labels.

Next: [Part 4 — Few-Shot Learning](/en/transfer-learning-4-few-shot-learning/), where we drop the assumption of abundant source data altogether and learn from a handful of examples per class.

---

## References

1. Ganin et al. (2016). Domain-Adversarial Training of Neural Networks. JMLR. [arXiv:1505.07818](https://arxiv.org/abs/1505.07818)
2. Long et al. (2015). Learning Transferable Features with Deep Adaptation Networks. ICML. [arXiv:1502.02791](https://arxiv.org/abs/1502.02791)
3. Sun & Saenko (2016). Deep CORAL: Correlation Alignment for Deep Domain Adaptation. ECCV. [arXiv:1607.01719](https://arxiv.org/abs/1607.01719)
4. Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN). ICCV. [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
5. Tzeng et al. (2017). Adversarial Discriminative Domain Adaptation (ADDA). CVPR. [arXiv:1702.05464](https://arxiv.org/abs/1702.05464)
6. Long et al. (2018). Conditional Adversarial Domain Adaptation (CDAN). NeurIPS. [arXiv:1705.10667](https://arxiv.org/abs/1705.10667)
7. Ben-David et al. (2010). A Theory of Learning from Different Domains. *Machine Learning*.
8. Li et al. (2017). Revisiting Batch Normalization for Practical Domain Adaptation (AdaBN). [arXiv:1603.04779](https://arxiv.org/abs/1603.04779)
9. Gretton et al. (2012). A Kernel Two-Sample Test (MMD). JMLR. [paper](https://jmlr.org/papers/v13/gretton12a.html)
10. Lipton et al. (2018). Detecting and Correcting for Label Shift with Black Box Predictors. ICML. [arXiv:1802.03916](https://arxiv.org/abs/1802.03916)
11. Sohn et al. (2020). FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence. NeurIPS. [arXiv:2001.07685](https://arxiv.org/abs/2001.07685)

---

## Series Navigation

| Part | Topic |
|------|-------|
| [1](/en/transfer-learning-1-fundamentals-and-core-concepts/) | Fundamentals and Core Concepts |
| [2](/en/transfer-learning-2-pre-training-and-fine-tuning/) | Pre-training and Fine-tuning |
| **3** | **Domain Adaptation** (you are here) |
| [4](/en/transfer-learning-4-few-shot-learning/) | Few-Shot Learning |
| [5](/en/transfer-learning-5-knowledge-distillation/) | Knowledge Distillation |
| [6](/en/transfer-learning-6-multi-task-learning/) | Multi-Task Learning |
