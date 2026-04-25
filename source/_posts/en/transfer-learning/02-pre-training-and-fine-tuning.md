---
title: "Transfer Learning (2): Pre-training and Fine-tuning"
date: 2025-04-19 09:00:00
tags:
  - Deep Learning
  - Transfer Learning
  - BERT
  - Fine-tuning
  - Pre-training
  - Self-Supervised Learning
  - GPT
  - LoRA
categories:
  - Transfer Learning
series:
  name: "Transfer Learning"
  part: 2
  total: 12
lang: en
mathjax: true
description: "Why pre-training learns a powerful prior from unlabeled data and how fine-tuning adapts it to your task. Covers contrastive learning, masked language models, discriminative learning rates, layer freezing, catastrophic forgetting, LoRA, and a production-ready BERT fine-tuning implementation."
disableNunjucks: true
series_order: 2
---

BERT changed NLP overnight. A model pre-trained on Wikipedia and BookCorpus could be fine-tuned on a few thousand labelled examples and beat task-specific architectures that researchers had spent years hand-crafting. The same pattern repeated in vision (ImageNet pre-training, then SimCLR, MAE), in speech (wav2vec 2.0), and in code (Codex). Today, "pre-train once, fine-tune everywhere" is the default recipe of modern deep learning.

But *why* does pre-training work? When should you freeze layers, when should you LoRA, and how small does your learning rate need to be? This article unpacks both the theory and the engineering practice behind the most successful transfer paradigm we have.

## What you will learn

- Why pre-training works, viewed through both a Bayesian and an information-theoretic lens
- Self-supervised pretext tasks: contrastive learning (SimCLR, MoCo) and masked language modelling (BERT MLM)
- Fine-tuning strategies: full fine-tuning, layer freezing, gradual unfreezing, linear probing, discriminative learning rates
- Catastrophic forgetting and how to keep source-task knowledge alive
- Parameter-efficient adaptation: Adapters and LoRA
- A complete BERT fine-tuning implementation with discriminative LR, gradient accumulation, mixed precision, and warmup scheduling

**Prerequisites:** Part 1 of this series (or equivalent transfer-learning intuition), basic familiarity with the Transformer architecture.

---

## Why pre-train?

![Pre-train then fine-tune pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/02-pre-training-and-fine-tuning/fig1_pretrain_finetune_pipeline.png)

The pipeline above captures the entire idea in one picture. Stage 1 burns enormous compute *once* on a generic, unlabelled corpus to produce a base model $\theta_{\mathrm{pre}}$. Stage 2 takes that base model and bends it to a specific task with a small labelled dataset, cheaply, and as many times as you want.

### The data asymmetry

Labels are expensive:

- **Medical imaging:** \$100--500 per CT scan for a radiologist's annotation.
- **Legal text:** lawyers reviewing each document at billable rates.
- **Low-resource languages:** parallel corpora barely exist at all.

Meanwhile, the internet stores petabytes of *unlabelled* text, images, and video. Pre-training exploits this asymmetry directly: **learn general representations from what is abundant; specialise on what is scarce.**

### A Bayesian view: pre-training is a prior

Let $\theta$ be the model parameters, $\mathcal{D}_{\mathrm{pre}}$ the pre-training corpus, and $\mathcal{D}_{\mathrm{task}}$ the labelled task data. Standard supervised training looks for

$$
\theta^{*} = \arg\max_{\theta} \log P(\mathcal{D}_{\mathrm{task}} \mid \theta).
$$

Pre-training plus fine-tuning instead does two stages of inference:

1. **Pre-train:** estimate the prior $P(\theta \mid \mathcal{D}_{\mathrm{pre}})$.
2. **Fine-tune:** Bayesian update with the task likelihood,

$$
P(\theta \mid \mathcal{D}_{\mathrm{task}}, \mathcal{D}_{\mathrm{pre}}) \;\propto\; P(\mathcal{D}_{\mathrm{task}} \mid \theta) \cdot P(\theta \mid \mathcal{D}_{\mathrm{pre}}).
$$

When $\mathcal{D}_{\mathrm{task}}$ is small, the likelihood is noisy and the posterior is dominated by the prior. A *good* prior - one that already concentrates probability in plausible regions of parameter space - dramatically improves the posterior estimate. That is exactly what a pre-trained checkpoint gives you.

### An information-theoretic view: features that transfer

Define a feature extractor $f_{\theta}: \mathcal{X} \to \mathbb{R}^{d}$. Pre-training searches for a $\theta$ such that the mutual information

$$
I(f_{\theta}(X); Y_{i})
$$

is high for many downstream label spaces $Y_{1}, Y_{2}, \dots$. ImageNet edges, textures, and object parts are useful far beyond classification - they help detection, segmentation, and even medical imaging. BERT's syntactic and semantic representations help across NLP. **A pre-trained model is, in effect, a compression of the data into features that retain transferable information.**

### Faster convergence, better minima

Pre-trained parameters land in a low-loss basin of the landscape. Fine-tuning only needs *local* adjustments - so it converges faster and tends to find flatter, better-generalising minima than random initialisation.

---

## Self-supervised pretext tasks

The trick of self-supervision is to design a task whose labels can be generated *automatically* from the data. The model learns by predicting parts of its input from other parts.

### Contrastive learning (vision)

Core idea: **pull representations of similar samples together and push dissimilar samples apart.**

#### SimCLR

For each image $x$ in a batch, apply two random augmentations (crop, colour jitter, blur) to get a positive pair $(x_{i}, x_{i'})$. Let $f$ be the encoder and $g$ a small projection head; write $z = g(f(x))$. The NT-Xent loss for one positive pair is

$$
\mathcal{L}_{i} = -\log \frac{\exp(\operatorname{sim}(z_{i}, z_{i'}) / \tau)}{\sum_{k \neq i} \exp(\operatorname{sim}(z_{i}, z_{k}) / \tau)},
$$

where $\operatorname{sim}$ is cosine similarity and $\tau$ is a temperature.

- **Numerator:** wants the positive pair to be similar.
- **Denominator:** normalises against every other sample in the batch as a negative.
- **Temperature $\tau$:** small $\tau$ sharpens the softmax and focuses the loss on the *hardest* negatives.

The catch: you need lots of negatives. SimCLR uses batch sizes of 4096--8192 and a TPU pod to keep them all in memory.

#### MoCo: momentum-updated queue

MoCo decouples the negative count from the batch size. Two encoders are kept: a *query* encoder $f_{q}$ updated by gradients, and a *key* encoder $f_{k}$ updated by exponential moving average,

$$
\theta_{k} \leftarrow m \cdot \theta_{k} + (1 - m) \cdot \theta_{q}, \qquad m \approx 0.999.
$$

Old keys are stashed in a **queue** of size 65 536. The dictionary is large *and* the encoder is consistent (because $f_{k}$ moves slowly). You get SimCLR-quality contrastive learning on a single GPU.

### Masked language modelling (NLP)

#### BERT's MLM objective

Take a sentence, replace 15 % of the tokens with a special `[MASK]`, and ask the model to recover the originals from the surrounding context:

$$
\mathcal{L}_{\mathrm{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_{i} \mid x_{\setminus \mathcal{M}}).
$$

The 15 % is split as **80 % `[MASK]`, 10 % random token, 10 % unchanged**. That mixture is not aesthetic; it closes the train-test distribution gap. At fine-tuning time there are no `[MASK]` tokens, so a model that has only ever seen `[MASK]` would underperform. The 10 % random and 10 % unchanged force the model to build a representation that is robust whether the position is masked or not.

**Why 15 % and not 5 % or 50 %?** Too few masks gives a weak learning signal per sentence; too many destroys the context the model needs to predict from. Empirically, 15 % is the sweet spot, although recent work (Wettig et al., 2023) shows you can push to 40 % with the right architecture.

#### Next-Sentence Prediction (NSP) and its replacement

BERT also pre-trained on NSP: given sentences A and B, predict whether B actually follows A. Later work (RoBERTa) showed NSP adds little - the model can solve it via topic similarity rather than discourse reasoning. ALBERT replaced NSP with **Sentence Order Prediction (SOP)**, where the negative is the same two sentences in reversed order. SOP forces real inter-sentence reasoning.

---

## Fine-tuning: why it converges so fast

![Loss curves: from scratch vs. fine-tuning](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/02-pre-training-and-fine-tuning/fig2_loss_curves_comparison.png)

The plot above is the empirical evidence behind the Bayesian argument. With identical model, dataset, and optimiser, the pre-trained model

- starts at a much lower loss (its prior is already good),
- reaches the from-scratch model's *best* validation loss in a handful of epochs, and
- ends at a noticeably lower floor.

Now the practical question: how exactly do you adapt hundreds of millions of pre-trained parameters without trampling them?

### Full fine-tuning

The simplest recipe: unfreeze everything and train end-to-end on the downstream task. To stop the parameters from drifting too far from the pre-trained checkpoint, you can add an L2 anchor:

$$
\theta^{*} = \arg\min_{\theta} \; \mathcal{L}_{\mathrm{task}}(\theta) + \lambda \lVert \theta - \theta_{\mathrm{pre}} \rVert^{2}.
$$

This is a simplified Elastic Weight Consolidation. In practice the implicit regularisation from a small learning rate plus early stopping is usually enough.

### Discriminative learning rates

Different layers carry different kinds of knowledge:

- **Bottom layers** (embeddings, early Transformer blocks) capture nearly universal features - tokenisation, basic syntax, low-level vision primitives. Touch them gently.
- **Top layers** (the classifier, the last block or two) are task-specific. Train them aggressively.

ULMFiT formalised this as **discriminative fine-tuning**. For an $L$-layer model, layer $\ell$ gets

$$
\eta_{\ell} = \frac{\eta_{L}}{\xi^{\,L - \ell}}, \qquad \xi \approx 2.6.
$$

The bottom layer ends up roughly $\xi^{L}$ times smaller than the top.

### Warmup, then decay

A schedule that just works for fine-tuning:

1. **Warmup** for the first $T_{w}$ steps - linearly grow the LR from $0$ to $\eta_{\max}$.
2. **Decay** the LR with either a cosine curve or linear ramp down to (near) zero.

Warmup matters because the freshly-attached classifier head emits noisy gradients in the first few steps. A large LR at that moment will smash through the pre-trained weights. Warmup gives the head time to settle before the rest of the network starts to move.

**Rule of thumb:** fine-tuning LR is usually 1--2 orders of magnitude smaller than pre-training LR. Pre-training BERT ran at about $10^{-4}$; fine-tuning typically lives at $2 \times 10^{-5}$.

![LR schedules and discriminative LR](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/02-pre-training-and-fine-tuning/fig6_lr_schedules.png)

### Layer freezing

![Layer freezing strategies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/02-pre-training-and-fine-tuning/fig3_layer_freezing_strategies.png)

Freezing means setting `requires_grad = False` on a layer's parameters. The layer still computes a forward pass, but it is invisible to the optimiser. Mathematically, freezing is the limit of an infinitely strong L2 anchor: $\lambda \to \infty$ on the frozen subset.

**Four common patterns:**

1. **Full fine-tune.** Everything trainable. Best when you have plenty of labels.
2. **Freeze bottom, train top.** Common default for small datasets that are similar to the pre-training domain.
3. **Gradual unfreezing (ULMFiT).** Start with only the head trainable, then unfreeze one layer per epoch from the top down. Robust against catastrophic forgetting.
4. **Linear probe.** Head only, backbone frozen. Cheapest possible adapter; often a strong baseline.

A quick decision matrix:

| Task vs. pre-training similarity | Labelled examples | Recommended |
|---|---|---|
| High | Few (\< 1 k / class) | Freeze bottom, fine-tune top |
| High | Many | Full fine-tuning |
| Low | Few | Freeze middle, fine-tune bottom and top, or use LoRA |
| Low | Many | Full fine-tuning + discriminative LR |

### Linear probing vs. full fine-tuning

![Linear probing vs. full fine-tuning](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/02-pre-training-and-fine-tuning/fig4_linear_probe_vs_full.png)

Linear probing trains *only* a linear classifier on top of frozen features. It is the purest test of representation quality - and in the very-low-data regime it often beats full fine-tuning, because there are simply not enough labels to safely move 110 M parameters. As the dataset grows, full fine-tuning crosses over and pulls away. Knowing where the crossover sits for your domain is worth a few quick experiments before you commit to a serving strategy.

### Catastrophic forgetting

![Catastrophic forgetting during fine-tuning](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/02-pre-training-and-fine-tuning/fig5_catastrophic_forgetting.png)

Aggressive fine-tuning is a one-way street: gains on the target task come at the cost of competence on the source. McCloskey and Cohen (1989) named this *catastrophic forgetting*; it is the central pain point of sequential transfer.

Three families of remedies:

- **Regularisation** (Elastic Weight Consolidation): penalise changes to parameters that mattered most for the source task, weighted by the diagonal Fisher information.
- **Replay:** keep a small buffer of source-task examples and mix them into each fine-tuning batch.
- **Architectural isolation:** keep the backbone frozen and add per-task adapters or LoRA modules - the source weights cannot be overwritten because they are never touched.

### Adapters: parameter-efficient fine-tuning

Insert a small bottleneck module into each Transformer block:

$$
\operatorname{Adapter}(h) = h + W_{\mathrm{up}}\, \sigma\big(W_{\mathrm{down}}\, h\big),
$$

with $W_{\mathrm{down}} \in \mathbb{R}^{m \times d}$, $W_{\mathrm{up}} \in \mathbb{R}^{d \times m}$, and $m \ll d$ (typically $m = 64$, $d = 768$). The residual connection means an untrained adapter is the identity, so you start exactly where the pre-trained model left off. You only train the adapter - per task you store $\sim 1\%$ of the full model's parameters.

### LoRA: low-rank weight updates

LoRA goes one step further. Instead of inserting a non-linear bottleneck, it directly decomposes the *update* to a frozen weight matrix:

$$
W' = W_{0} + \Delta W = W_{0} + B A, \qquad A \in \mathbb{R}^{r \times d}, \; B \in \mathbb{R}^{d \times r}, \; r \ll d.
$$

During fine-tuning $W_{0}$ is frozen and only $A, B$ are trained. Three properties make LoRA the de-facto PEFT method today:

- **Tiny.** Only $2dr$ parameters per adapted matrix - usually $r = 4$--$16$.
- **Zero inference overhead.** At deployment you can fold $BA$ into $W_{0}$ and serve a single matrix.
- **Trivially composable.** Swapping tasks is swapping a 10 MB delta, not loading a 1 GB checkpoint.

The implicit assumption - and it is often true - is that **task adaptation lives in a low-dimensional subspace of weight space.**

---

## How much labelled data do you actually need?

![Performance vs. target dataset size](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/transfer-learning/02-pre-training-and-fine-tuning/fig7_data_size_scaling.png)

The figure consolidates the practical choices. On the same target task, with the same backbone:

- **From scratch** is a flat line until you have many thousands of labels. Below that point the model essentially memorises noise.
- **Linear probing** wins immediately - frozen pre-trained features already separate the classes, and the only thing being learned is a hyperplane.
- **LoRA** tracks full fine-tuning closely with a small handicap and a fraction of the parameter budget.
- **Full fine-tuning** has the highest ceiling, but only justifies its cost once you have enough labels to safely move every parameter.

The horizontal arrow is the headline number: pre-training shifts the data-efficiency curve up and to the *left* by roughly an order of magnitude - the same accuracy with 10x fewer labels.

---

## BERT in practice

### Architecture recap

BERT is a stack of bidirectional Transformer encoder blocks. Given input tokens, each block produces contextual representations through multi-head self-attention plus a feed-forward network, with residuals and LayerNorm.

### Adapting to different tasks

| Task | How BERT handles it |
|---|---|
| Text classification | Take the `[CLS]` representation, feed it to a linear head. |
| Sequence labelling (NER) | Per-token linear head over the final layer. |
| Question answering (SQuAD) | Two heads predicting the start and end positions of the answer span. |
| Sentence-pair tasks (NLI) | Concatenate sentences with `[SEP]`, classify on `[CLS]`. |

### GPT: the autoregressive cousin

GPT pre-trains by left-to-right next-token prediction:

$$
\mathcal{L}_{\mathrm{GPT}} = -\sum_{t} \log P(x_{t} \mid x_{< t}).
$$

For *understanding* tasks BERT's bidirectional context wins. For *generation* tasks GPT's autoregressive structure is the natural fit. Modern decoder-only LLMs (LLaMA, Qwen, GPT-4) are GPT's lineage - and the same fine-tuning principles transfer almost entirely, just with smaller LRs and heavier reliance on LoRA.

---

## Complete implementation: BERT fine-tuning

A production-ready trainer with discriminative learning rates, gradient accumulation, mixed precision, and warmup + linear decay scheduling.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup,
)
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


class BERTClassifier(nn.Module):
    """BERT backbone + a linear classification head on the [CLS] token."""

    def __init__(self, bert_model_name="bert-base-uncased",
                 num_classes=2, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        return self.classifier(pooled)


class TextDataset(Dataset):
    """Tokenises and pads texts for BERT."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BERTFineTuner:
    """Trainer with discriminative LR, gradient accumulation and AMP."""

    def __init__(self, model, train_loader, val_loader,
                 num_epochs=3, learning_rate=2e-5, warmup_ratio=0.1,
                 gradient_accumulation_steps=1, max_grad_norm=1.0,
                 device="cuda", use_amp=True,
                 discriminative_lr=False, lr_decay=2.6):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.use_amp = use_amp
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * warmup_ratio)

        if discriminative_lr:
            self.optimizer = self._discriminative_optimizer(learning_rate, lr_decay)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps,
        )
        self.scaler = GradScaler() if use_amp else None
        self.criterion = nn.CrossEntropyLoss()

    def _discriminative_optimizer(self, lr, decay):
        """Bottom layers get exponentially smaller LRs than the top."""
        num_layers = len(self.model.bert.encoder.layer)
        groups = []
        # Embeddings -- smallest LR.
        groups.append({
            "params": self.model.bert.embeddings.parameters(),
            "lr": lr / (decay ** num_layers),
        })
        # Each Transformer layer.
        for i in range(num_layers):
            groups.append({
                "params": self.model.bert.encoder.layer[i].parameters(),
                "lr": lr / (decay ** (num_layers - i - 1)),
            })
        # Pooler + classifier -- highest LR.
        groups.append({
            "params": list(self.model.bert.pooler.parameters())
                      + list(self.model.classifier.parameters()),
            "lr": lr,
        })
        return AdamW(groups, eps=1e-8)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        for step, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            if self.use_amp:
                with autocast():
                    loss = self.criterion(self.model(input_ids, mask), labels)
                loss = loss / self.gradient_accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                loss = self.criterion(self.model(input_ids, mask), labels)
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.max_grad_norm)
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_preds, all_labels, total_loss = [], [], 0.0
        for batch in tqdm(self.val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)
            logits = self.model(input_ids, mask)
            total_loss += self.criterion(logits, labels).item()
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        return (
            total_loss / len(self.val_loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average="weighted"),
        )

    def train(self):
        best_val_loss = float("inf")
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc, val_f1 = self.evaluate()
            print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  acc={val_acc:.4f}  f1={val_f1:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pt")
                print("  -> saved best model")


def main():
    BERT_MODEL = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    # Dummy data; replace with your real dataset.
    texts = ["This movie is great!"] * 500 + ["This movie is terrible!"] * 500
    labels = [1] * 500 + [0] * 500

    dataset = TextDataset(texts, labels, tokenizer, max_length=128)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=16)

    model = BERTClassifier(BERT_MODEL, num_classes=2)
    trainer = BERTFineTuner(
        model, train_loader, val_loader,
        num_epochs=3, learning_rate=2e-5,
        gradient_accumulation_steps=2,
        discriminative_lr=True,
        use_amp=True,
    )
    trainer.train()


if __name__ == "__main__":
    main()
```

### Why these knobs matter

| Technique | What it does | When to keep it on |
|---|---|---|
| Discriminative LR | Embeddings see LR / $2.6^{12}$, classifier sees full LR. | Always for BERT-style stacks; the bottom rarely needs to move. |
| Gradient accumulation | Simulates a larger effective batch on a small GPU. | Whenever a single batch would OOM. |
| Mixed precision (AMP) | FP16 forward, FP32 master weights. ~2x speed and ~50 % memory. | Always on modern GPUs (Volta+). |
| Warmup + decay | Stabilises the first hundred steps, anneals at the end. | Always for fine-tuning. |
| Gradient clipping | Caps the global norm to prevent rare gradient blow-ups. | Always; cost is negligible. |

---

## FAQ

### Why does warmup help so much during fine-tuning?

The freshly-attached head produces noisy gradients in the first few steps. Pushing those gradients into the pre-trained backbone at full LR partially overwrites useful weights before the head has stabilised. Warmup keeps the LR small while the head settles, then ramps up.

### How much pre-training data do I need?

At minimum: hundreds of MB of text for NLP, or millions of images for vision. But *diversity* matters more than raw size. 10 M images of one species is worse than 1 M images covering many. Scaling laws (Kaplan et al., 2020) say performance grows roughly as a power of corpus size, but only when the data stays diverse and the model grows alongside.

### How do I detect overfitting during fine-tuning?

Three signals: (1) train loss falls while validation loss rises; (2) train accuracy is high but validation accuracy plateaus; (3) the model becomes overconfident, with predicted probabilities clustering near 0 or 1. Remedies: more dropout, early stopping, data augmentation, freeze more layers, or switch to LoRA.

### LoRA vs. full fine-tuning - when to use which?

Use LoRA when you serve many tasks from one base model (swap a small delta per task), when GPU memory is tight, or when you want to iterate quickly. Use full fine-tuning when you have a single target task, plenty of labels, and you care about the last point of accuracy.

### Why do bottom layers get such a small LR under discriminative fine-tuning?

Bottom layers encode features that are nearly task-independent - subword statistics, low-level grammar, simple visual primitives. Moving them costs you transferability with little gain on the target task. The top, where task-specific decisions live, is where most of the learning should happen.

---

## Summary

Pre-training compresses the world's unlabelled data into a strong prior. Fine-tuning is the Bayesian update with whatever labelled data you can afford. Together they turned deep learning from "needs millions of labels per task" into "a single base model serves a thousand specialisations."

Key takeaways:

- **Self-supervised objectives** (contrastive learning, MLM) generate supervision automatically - that is what unlocks the unlabelled web.
- **Discriminative LRs** and **gradual unfreezing** adapt different layers at appropriate speeds.
- **Linear probing** is a strong, cheap baseline; **full fine-tuning** wins when you have the data.
- **LoRA and Adapters** make fine-tuning parameter-efficient and trivially composable.
- **Warmup, decay, and gradient clipping** keep training stable. Almost every fine-tuning recipe uses all three.
- **Catastrophic forgetting** is real - protect the source task with regularisation, replay, or isolation.

Next up: [Part 3 - Domain Adaptation](/en/transfer-learning-3-domain-adaptation/), where the pre-training data and the deployment data live in different distributions and your fine-tuning recipes alone are not enough.

---

## References

1. Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
2. Chen et al. (2020). *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR).* ICML. [arXiv:2002.05709](https://arxiv.org/abs/2002.05709)
3. He et al. (2020). *Momentum Contrast for Unsupervised Visual Representation Learning (MoCo).* CVPR. [arXiv:1911.05722](https://arxiv.org/abs/1911.05722)
4. Howard and Ruder (2018). *Universal Language Model Fine-tuning for Text Classification (ULMFiT).* ACL. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)
5. Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
6. Liu et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)
7. Houlsby et al. (2019). *Parameter-Efficient Transfer Learning for NLP (Adapters).* ICML. [arXiv:1902.00751](https://arxiv.org/abs/1902.00751)
8. Kirkpatrick et al. (2017). *Overcoming Catastrophic Forgetting in Neural Networks (EWC).* PNAS. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)
9. Kaplan et al. (2020). *Scaling Laws for Neural Language Models.* [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
10. Wettig et al. (2023). *Should You Mask 15 % in Masked Language Modeling?* EACL. [arXiv:2202.08005](https://arxiv.org/abs/2202.08005)

---

## Series Navigation

| Part | Topic |
|------|-------|
| [1](/en/transfer-learning-1-fundamentals-and-core-concepts/) | Fundamentals and Core Concepts |
| **2** | **Pre-training and Fine-tuning** (you are here) |
| [3](/en/transfer-learning-3-domain-adaptation/) | Domain Adaptation |
| [4](/en/transfer-learning-4-few-shot-learning/) | Few-Shot Learning |
| [5](/en/transfer-learning-5-knowledge-distillation/) | Knowledge Distillation |
| [6](/en/transfer-learning-6-multi-task-learning/) | Multi-Task Learning |
