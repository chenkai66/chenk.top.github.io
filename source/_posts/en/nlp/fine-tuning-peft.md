---
title: "NLP (8): Model Fine-tuning and PEFT"
date: 2025-11-05 09:00:00
tags:
  - NLP
  - PEFT
  - LoRA
  - LLM
  - Fine-tuning
categories: Natural Language Processing
series: NLP
part: 8
total_parts: 12
lang: en
mathjax: true
description: "A deep dive into Parameter-Efficient Fine-Tuning. Why LoRA's low-rank update works, the math and memory accounting behind QLoRA, how Adapters and Prefix-Tuning differ, and how to choose between them in production."
disableNunjucks: true
series_order: 8
---

In 2020, fine-tuning a 7-billion-parameter language model was a project budget item: eight A100s, several days, and an engineer who knew how to babysit gradient checkpointing. In 2024, a graduate student does it on a laptop. The distance between those two worlds is almost entirely covered by one paper — Hu et al.'s LoRA (ICLR 2022) — and one follow-up — Dettmers et al.'s QLoRA (NeurIPS 2023).

The shift is not just engineering. Parameter-Efficient Fine-Tuning (PEFT) reframes what it means to "have a model." Instead of one binary blob per task, you keep a single frozen base model and a directory of small adapter files, each a few tens of megabytes. Switching tasks becomes loading a new adapter; serving N domains becomes O(1) base + N · ε.

This article reconstructs PEFT from first principles. We start with the question full fine-tuning answers — and the questions it does not — then derive LoRA's low-rank assumption, walk through the memory math that makes QLoRA fit a 7B model in 6 GB, and finish with practical choices: which method, which rank, which modules.

## What you will learn

- **Why** full fine-tuning is wasteful in the LLM era — overparameterization and the intrinsic-rank hypothesis
- **LoRA mechanics**: the decomposition $\Delta W = BA$, why $B$ is initialized to zero, how $\alpha/r$ scaling changes effective learning rate
- **QLoRA**: NF4 quantization, double quantization, and paged optimizers — and the exact memory accounting
- **Adapters and Prefix-Tuning**: where they sit in the Transformer block, when they win, when they lose
- **Production choices**: rank selection, target modules, multi-LoRA serving, alignment via instruction tuning and RLHF

## Prerequisites

- Transformer architecture ([Part 4](/en/nlp-attention-transformer/))
- GPT-style decoders ([Part 6](/en/nlp-gpt-generative-models/))
- PyTorch and basic GPU memory intuition (optimizer state, activations, gradients)

---

## 1. Why not just fine-tune everything?

### The cost ledger

Full fine-tuning means every parameter is unfrozen, every gradient is stored, and the optimizer (typically AdamW) keeps two extra fp32 buffers per parameter. For a 7B model in mixed precision, the per-step VRAM bill looks like this:

| Component | Bytes per parameter | 7B model |
|-----------|--------------------|----------|
| Weights (fp16) | 2 | 14 GB |
| Gradients (fp16) | 2 | 14 GB |
| AdamW states (fp32 m + v) | 8 | 56 GB |
| Activations (varies with seq, batch) | — | 8–20 GB |
| **Total** | — | **~95 GB** |

That is two A100-80GB cards minimum, before you have written a single line of training code. The ledger explains why PEFT is not a "nice optimization" — it is the *only* way most practitioners can touch a 7B+ model at all.

![Trainable parameters across PEFT methods on a 7B base, and disk cost when serving N tasks.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig1_full_vs_peft.png)

### The intrinsic-rank hypothesis

There is also a deeper reason to prefer PEFT, articulated by Aghajanyan et al. (2020) and the LoRA paper: **fine-tuning updates have very low intrinsic rank**. If the change you need to make to a pre-trained model lives in a low-dimensional subspace, then training a full $d \times k$ matrix is not just expensive — it is the wrong hypothesis class. You should be searching the low-rank submanifold directly.

Empirically, fine-tuning a 175B model on a downstream task can be matched by training as few as $\sim$200 directions in parameter space (Aghajanyan et al., 2020). This is the conceptual key that unlocks LoRA.

### Frozen fine-tuning — the weak baseline

Before PEFT, the simplest cost-saver was to freeze the body and train only the head, or unfreeze the top few layers:

```python
for param in model.parameters():
    param.requires_grad = False
for param in model.transformer.h[-2:].parameters():
    param.requires_grad = True
```

This works for classification on top of BERT-class encoders, but fails on generative tasks: the trained layers cannot redirect the upstream representations enough. PEFT methods, in contrast, inject trainable parameters *throughout* the network while keeping the count tiny.

---

## 2. LoRA: low-rank adaptation

### The decomposition

For a frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA learns an additive update parameterized as a product of two thin matrices:

$$
W = W_0 + \Delta W, \qquad \Delta W = \frac{\alpha}{r}\, B A,
\qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k).
$$

The forward pass becomes $h = x W_0^\top + \frac{\alpha}{r} (x A^\top) B^\top$. The original computation $x W_0^\top$ is unchanged; the LoRA branch adds a rank-$r$ correction.

![LoRA decomposes the weight update into two thin matrices, $B$ initialized to zero and $A$ to small random values.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig2_lora_decomp.png)

### Counting parameters

For $d = k = 4096$ (LLaMA-7B's hidden size) and $r = 8$:

- Original $W_0$: $d \cdot k = 16{,}777{,}216$ parameters
- LoRA $A, B$: $r \cdot (d + k) = 65{,}536$ parameters
- Reduction: **256× fewer trainable parameters per matrix**

Apply LoRA to all four attention projections in 32 layers and you are training roughly 8 M parameters out of 7 B — about 0.12%.

### Two design choices that matter

**Initialization asymmetry.** $B$ is initialized to zero, $A$ to a small Gaussian. Therefore $BA = 0$ at step zero, and the model behaves *exactly* like the pre-trained version on its first forward pass. Training begins from a known-good operating point — no warmup gymnastics needed.

**The $\alpha/r$ scale.** This is not a cosmetic constant. Increasing $r$ at fixed $\alpha$ would otherwise increase the update magnitude (more directions, larger combined norm), forcing you to retune learning rate. The $\alpha/r$ factor decouples them: pick $\alpha$ once (typical $\alpha = 2r$), sweep $r$ for capacity, the effective step size stays roughly constant.

### A minimal implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        # A: (r, in_features) — small Gaussian init
        self.lora_A = nn.Parameter(torch.empty(r, in_f))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        # B: (out_features, r) — zero init so the branch starts inert
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original frozen path
        out = self.base(x)
        # low-rank branch: x @ A^T @ B^T, scaled
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + self.scaling * lora_out

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the LoRA delta into the base weight (zero inference overhead)."""
        delta = self.scaling * (self.lora_B @ self.lora_A)
        self.base.weight.data += delta
        self.lora_B.zero_()
```

The `merge()` method is the trick that makes LoRA cost nothing at inference: once trained, fold $\Delta W$ into $W_0$ and the model becomes architecturally identical to the original — no extra matmul, no extra latency.

### Where to apply LoRA

Hu et al. found that applying LoRA to **query and value projections** (`q_proj`, `v_proj`) recovers most of the gain. The current production default is broader — all four attention projections, sometimes the FFN as well:

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 8,388,608 || all params: 6,746,804,224 || trainable%: 0.124
```

Adding `gate_proj`, `up_proj`, `down_proj` (the LLaMA FFN trio) typically buys a fraction of a point on benchmarks at 3× the trainable parameter count. Worth it for code generation; rarely worth it for classification.

---

## 3. QLoRA: pushing the base into 4 bits

LoRA solved the *trainable* parameter problem. The *frozen* parameters still sit in fp16 and dominate VRAM. QLoRA attacks that.

![QLoRA combines a 4-bit quantized base with bf16 LoRA adapters; gradients flow only through the small green path.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig5_qlora.png)

### The three QLoRA innovations

**1. NF4 (NormalFloat 4-bit) quantization.** Standard 4-bit integer quantization wastes resolution because LLM weights are not uniformly distributed — they are approximately Gaussian. NF4 picks 16 quantization levels that are equiprobable under a standard normal distribution, giving information-theoretically optimal coverage. The block size is typically 64 weights, with one fp16 scale per block.

**2. Double quantization.** Each block needs a scale constant; for a 7B model with block size 64, that is $\sim$110 M scale constants × 4 bytes = 0.44 GB of overhead. QLoRA quantizes the *constants themselves* to 8-bit, recovering ~0.4 bit per parameter — small but meaningful at 7B scale.

**3. Paged optimizers.** Long sequences cause activation memory to spike. Paged AdamW offloads optimizer state to CPU RAM via NVIDIA Unified Memory, then pages it back in. You trade some throughput for the ability to train sequences that would otherwise OOM.

### The setup

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NormalFloat 4-bit
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # quantize the quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)   # cast LayerNorms to fp32, etc.

lora = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora)
```

### What it costs you, what it saves

QLoRA gives back about 1–2 percentage points on most benchmarks compared with fp16 LoRA, and recovers most of full fine-tuning. The win is dramatic VRAM:

![Training memory by component, and peak VRAM across model scales — QLoRA brings 70B inside an A100.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig6_memory.png)

For a 70B model, full fine-tuning needs roughly a terabyte of VRAM (and is essentially never done outside frontier labs). QLoRA brings it inside a single 80 GB GPU. That is the difference between "research-only" and "your team can do this."

---

## 4. Adapters: the OG PEFT method

Houlsby et al. (2019) proposed adapters two years before LoRA. The idea: insert a small bottleneck module after each sub-layer of the Transformer.

$$
\text{Adapter}(x) = x + W_{\text{up}}\, \sigma(W_{\text{down}}\, x), \qquad
W_{\text{down}} \in \mathbb{R}^{m \times d},\ W_{\text{up}} \in \mathbb{R}^{d \times m},\ m \ll d.
$$

$W_{\text{up}}$ is initialized to zero so the adapter is initially the identity — same trick as LoRA's $B$.

![Adapters are inserted after attention and FFN sub-layers; each is a down-project / nonlinearity / up-project bottleneck with a residual.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig3_adapter.png)

```python
class Adapter(nn.Module):
    def __init__(self, d_model: int, bottleneck: int = 64):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))
```

### Adapter vs LoRA — when each wins

| | **Adapter** | **LoRA** |
|---|---|---|
| Where it sits | *Inside* the residual path | *Beside* a weight matrix |
| Inference overhead | Extra matmul per block (~5–10% latency) | None after `merge()` |
| Composability | Stack adapters | Add LoRA deltas |
| Multi-task serving | One adapter per task | One LoRA per task; supports batched mixing |
| Typical size | ~0.5–3% of base | ~0.1–1% |
| Best for | Settings with stable inference graph | LLM serving with strict latency budgets |

LoRA has eaten most of adapters' market share because of the zero-overhead inference property. Adapters remain attractive when you want very explicit modular composition — multilingual setups, for instance, where AdapterFusion (Pfeiffer et al., 2021) combines language-specific adapters at inference.

---

## 5. Prompt-based PEFT: tuning the input, not the weights

A radically different idea: leave the model entirely frozen and learn *what to feed it*.

![Three flavors of prompt-based PEFT — soft prompt at the input, KV prefix per layer, and deep prompts everywhere.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig4_prefix_prompt.png)

### Prompt-Tuning (Lester et al., 2021)

Prepend $n$ trainable embedding vectors to the input sequence. That is the entire method. With a 11B T5 model, prompt-tuning matches full fine-tuning on SuperGLUE — but the result *only holds at scale*. For sub-1B models prompt-tuning lags badly.

### Prefix-Tuning (Li & Liang, 2021)

Same idea but applied at every Transformer layer's attention: learn key and value prefix matrices $P_K, P_V$ that get prepended before attention computation:

$$
\text{Attention}(Q,\ [P_K; K],\ [P_V; V]).
$$

Because each layer has its own prefix, the model can rewrite its computation more deeply than a single input prompt allows.

### P-Tuning v2 (Liu et al., 2022)

Drops the reparameterization MLP that Prefix-Tuning used (it was unstable across model scales) and applies deep prompts uniformly. P-Tuning v2 is the first prompt-based method to match full fine-tuning on **NLU tasks at all scales**, including small models.

**When to use them.** Prompt methods shine when (a) you need extreme parameter efficiency — a few KB per task — or (b) you want to ship many prompts behind a single served model. They tend to underperform LoRA on generative tasks.

---

## 6. Choosing among methods

![PEFT methods on the parameter-efficiency / quality plane, and how LoRA rank affects different task types.](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig7_perf_vs_params.png)

A practical decision tree:

```
                     ┌─ Need the absolute best score? → Full FT (if you can afford it)
                     │
Have GPU memory?  ───┤
                     │  Yes → LoRA r=16 on q/k/v/o
                     └─ No  → QLoRA r=16 (NF4 + paged AdamW)

Need many tasks served from one base? → LoRA (mergeable, swappable)

Generative task with strict latency? → LoRA after merge
Encoder NLU task, want minimal params? → P-Tuning v2 or BitFit
Modular / compositional setup?         → Adapter (with AdapterFusion)
```

### Choosing the rank

Start with $r = 16$. From the right panel of the figure above, you can see the diminishing-returns inflection sits between $r = 8$ and $r = 32$ for most tasks:

- **Simple classification** saturates by $r = 8$
- **Code generation, reasoning** benefits up to $r = 32$ or $r = 64$
- **Domain adaptation** (medical, legal) typically wants $r = 16$–$32$

If you are unsure, sweep $r \in \{8, 16, 32\}$ and pick the smallest that closes the gap to your held-out target metric.

---

## 7. Alignment: instruction tuning and RLHF

PEFT is the lever, alignment is what you usually pull it for. Two stages dominate modern LLM post-training:

**Supervised instruction tuning.** Fine-tune the base model on `(instruction, response)` pairs so it learns to follow human-written prompts. Quality dominates quantity: 1K–10K carefully curated examples often beat 100K crowd-sourced ones (the LIMA paper, Zhou et al., 2023, made 1K examples work surprisingly well on a 65B model).

```python
def format_example(ex):
    if ex["input"]:
        return (f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n{ex['output']}")
    return (f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Response:\n{ex['output']}")
```

**RLHF** (Ouyang et al., 2022, the InstructGPT paper). After SFT, train a reward model on human preference pairs and optimize the policy with PPO against that reward. The Bradley–Terry preference loss:

$$
\mathcal{L}_{\text{RM}} = -\log \sigma\bigl(r_\theta(x, y_{\text{chosen}}) - r_\theta(x, y_{\text{rejected}})\bigr).
$$

```python
class RewardModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        out = self.base(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state[:, -1, :]
        return self.head(last).squeeze(-1)
```

For the RL side of the story, see [RL Part 12: RLHF and LLM Applications](/en/reinforcement-learning-12-rlhf-and-llm-applications/). DPO (Rafailov et al., 2023) is now a popular simpler alternative that skips the explicit reward model.

---

## 8. End-to-end recipe

```python
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          TrainingArguments, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import torch

MODEL = "meta-llama/Llama-2-7b-hf"

# 1. 4-bit base
bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL, quantization_config=bnb, device_map="auto",
)
model = prepare_model_for_kbit_training(model)

# 2. LoRA on attention projections
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none", task_type="CAUSAL_LM",
))
model.print_trainable_parameters()
# trainable params: 8,388,608 || all params: 3,508,801,536 || trainable%: 0.239

# 3. Train with SFTTrainer (handles formatting + collation)
ds = load_dataset("yahma/alpaca-cleaned", split="train")
args = TrainingArguments(
    output_dir="./llama-qlora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    bf16=True,
    optim="paged_adamw_8bit",     # the QLoRA paged optimizer
    logging_steps=10, save_steps=500,
)
trainer = SFTTrainer(
    model=model, args=args, train_dataset=ds,
    tokenizer=tokenizer, max_seq_length=1024,
    dataset_text_field="text",
)
trainer.train()
model.save_pretrained("./llama-qlora-adapter")  # ~80 MB on disk
```

### Multi-LoRA serving

Because LoRA adapters are tiny and additive, you can keep dozens of them in memory and switch per request:

```python
# load multiple adapters into the same base model
model.load_adapter("./adapter-medical", adapter_name="med")
model.load_adapter("./adapter-legal",   adapter_name="legal")

model.set_adapter("med")
out_a = model.generate(...)

model.set_adapter("legal")
out_b = model.generate(...)
```

Frameworks like vLLM and S-LoRA push this further: batch requests targeting different adapters in a single forward pass by stacking the LoRA deltas. One 7B base in VRAM, hundreds of fine-tuned models served.

---

## FAQ

### When is full fine-tuning worth it?

When you have abundant compute, $\geq$100K high-quality examples, and need every fraction of a point — e.g. base-model providers shipping a flagship instruct model. For everyone else: LoRA or QLoRA.

### LoRA rank — how to pick?

Start at 16. If it underfits (training loss plateaus high), bump to 32 or 64. If overfits and you have little data, drop to 4–8. For classification tasks 8 is usually plenty.

### Which target modules?

`q_proj` and `v_proj` give you 80% of the gain. Add `k_proj` and `o_proj` for the rest. Add the FFN trio (`gate/up/down_proj`) only if you have a generation-heavy task and the budget for 3× more trainable parameters.

### Does LoRA cost inference?

No, after merging. Before merging there is one tiny extra matmul per LoRA-wrapped layer — negligible but nonzero.

### QLoRA quality drop?

Typically 1–2 points on standard benchmarks compared with fp16 LoRA, often within noise. The memory savings are worth it for almost every practitioner.

### How much instruction data?

LIMA showed 1K hand-curated examples can produce a coherent assistant from a strong base model. Practical floor: 1K–10K high-quality examples; quality matters far more than count.

### Can I combine PEFT methods?

Yes — LoRA + prompt tuning is a documented combination, and QLoRA is itself a stack (4-bit base + LoRA + paged optimizer). Adapter + LoRA in the same model is unusual.

---

## Series Navigation

- **Previous**: [Part 7 — Prompt Engineering and In-Context Learning](/en/nlp-prompt-engineering-icl/)
- **Next**: [Part 9 — Deep Dive into LLM Architecture](/en/nlp-llm-architecture-deep-dive/)
- [View all 12 parts in the NLP series](/tags/NLP/)

## References

1. Hu, E. J. et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR.
2. Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS.
3. Houlsby, N. et al. (2019). *Parameter-Efficient Transfer Learning for NLP*. ICML.
4. Li, X. L. & Liang, P. (2021). *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. ACL.
5. Lester, B. et al. (2021). *The Power of Scale for Parameter-Efficient Prompt Tuning*. EMNLP.
6. Liu, X. et al. (2022). *P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally*. ACL.
7. Aghajanyan, A. et al. (2020). *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. ACL 2021.
8. Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT)*. NeurIPS.
9. Rafailov, R. et al. (2023). *Direct Preference Optimization*. NeurIPS.
10. Zhou, C. et al. (2023). *LIMA: Less Is More for Alignment*. NeurIPS.
