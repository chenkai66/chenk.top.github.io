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
series: nlp
part: 8
total_parts: 12
lang: en
mathjax: true
description: "A deep dive into Parameter-Efficient Fine-Tuning. Why LoRA's low-rank update works, the math and memory accounting behind QLoRA, how Adapters and Prefix-Tuning differ, and how to choose between them in production."
disableNunjucks: true
series_order: 8
series_total: 12
translationKey: "nlp-8"
---
In 2020, fine-tuning a 7-billion-parameter language model was a project budget item: eight A100s, several days, and an engineer who knew how to babysit gradient checkpointing. In 2024, a graduate student does it on a laptop. The distance between those two worlds is almost entirely covered by one paper — Hu et al.'s LoRA (ICLR 2022) — and one follow-up — Dettmers et al.'s QLoRA (NeurIPS 2023).

The shift is not just engineering. Parameter-Efficient Fine-Tuning (PEFT) reframes what it means to "have a model." Instead of one binary blob per task, you keep a single frozen base model and a directory of small adapter files, each a few tens of megabytes. Switching tasks becomes loading a new adapter; serving N domains becomes O(1) base + N · ε.

This article reconstructs PEFT from first principles. We start with the question full fine-tuning answers — and the questions it does not — then derive LoRA's low-rank assumption, walk through the memory math that makes QLoRA fit a 7B model in 6 GB, and finish with practical choices: which method, which rank, which modules.


<!-- wanx-hero -->
![NLP (8): Model Fine-tuning and PEFT — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/illustration_1.png)


---

## What You Will Learn

- **Why** full fine-tuning is wasteful in the LLM era — overparameterization and the intrinsic-rank hypothesis
- **LoRA mechanics**: the decomposition $\Delta W = BA$, why $B$ is initialized to zero, how $\alpha/r$ scaling changes effective learning rate
- **QLoRA**: NF4 quantization, double quantization, and paged optimizers — and the exact memory accounting
- **Adapters and Prefix-Tuning**: where they sit in the Transformer block, when they win, when they lose
- **Production choices**: rank selection, target modules, multi-LoRA serving, alignment via instruction tuning and RLHF

## Prerequisites

<!-- wanx-mid -->
![NLP (8): Model Fine-tuning and PEFT — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/illustration_2.png)


- Transformer architecture ([Part 4](/en/nlp/attention-transformer/))
- GPT-style decoders ([Part 6](/en/nlp/gpt-generative-models/))
- PyTorch and basic GPU memory intuition (optimizer state, activations, gradients)

---

## Why not just fine-tune everything?

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

![Fine-Tuning Strategy Decision Tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/fine-tuning-peft/fig_finetuning_decision_en.png)

### Choosing the rank

Start with $r = 16$. From the right panel of the figure above, you can see the diminishing-returns inflection sits between $r = 8$ and $r = 32$ for most tasks:

- **Simple classification** saturates by $r = 8$
- **Code generation, reasoning** benefits up to $r = 32$ or $r = 64$
- **Domain adaptation** (medical, legal) typically wants $r = 16$–$32$

If you are unsure, sweep $r \in \{8, 16, 32\}$ and pick the smallest that closes the gap to your held-out target metric.

---

## Alignment: instruction tuning and RLHF

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
$$\mathcal{L}_{\text{RM}} = -\log \sigma\bigl(r_\theta(x, y_{\text{chosen}}) - r_\theta(x, y_{\text{rejected}})\bigr).$$
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

For the RL side of the story, see [RL Part 12: RLHF and LLM Applications](/en/reinforcement-learning/12-rlhf-and-llm-applications/). DPO (Rafailov et al., 2023) is now a popular simpler alternative that skips the explicit reward model.

---

## End-to-end recipe

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

## Concrete fine-tuning recipe (LoRA on a 7B)

The hyperparameters that actually matter, with numbers I've used and shipped.

**Data sizing.** For LoRA on a 7B base, 1,000--5,000 high-quality instruction-response pairs gets you most of the gain. Below 500 examples, you're better off with a few-shot prompt. Above 20,000, you're spending money on diminishing returns unless your task is genuinely complex.

**Rank `r`.** Default `r=8`, but for harder tasks (code generation, multi-step reasoning) bump to `r=16` or `r=32`. Doubling `r` doubles parameter count but not training time noticeably. Going above `r=64` rarely helps and starts overfitting.

**Alpha.** Set `alpha = 2 * r`. So `r=8` → `alpha=16`. The ratio `alpha/r` is the effective learning rate scaling on the LoRA path. Keeping it at 2 is a reliable default.

**Target modules.** Apply LoRA to `q_proj`, `k_proj`, `v_proj`, `o_proj` *and* `gate_proj`, `up_proj`, `down_proj` if you can afford the memory. Attention-only LoRA (the original paper) leaves gains on the table for instruction tuning. Empirically, attention-plus-MLP is +2--4 points on Open LLM Leaderboard tasks.

**Learning rate.** `2e-4` for LoRA is the right starting point. Full fine-tune would use `2e-5`; LoRA tolerates 10× higher because only a small fraction of params are updating. Use cosine schedule with 3% warmup.

**Batch size and gradient accumulation.** Effective batch size 32--64 sequences. On a single A100 80GB with sequence length 2048, you'll fit micro-batch 2--4, so accumulate to reach the effective batch.

**Epochs.** 1--3. Watch the eval loss; it usually bottoms out in epoch 2. Going to 5+ epochs reliably overfits on small instruction datasets.

**Sequence packing.** Pack multiple short examples into one 2048-token sequence with proper attention masking. 2--3× throughput improvement vs padding. `transformers` SFTTrainer supports this natively now.

A run on 5K examples, 7B base, single A100, takes about 2--4 hours and costs ~$8--12 on a rented GPU.

## Where LoRA quietly fails

Three failure modes that I've seen burn teams.

**Failure 1: catastrophic forgetting in the rare-token tail.** LoRA preserves the base model better than full fine-tuning, but it still nudges output distributions. After instruction-tuning a code model, I've seen perplexity on rare programming languages (e.g., Erlang, Haskell) jump 30%+ even though English instruction-following improved. Fix: include a small (5--10%) replay of base-model-style data in your fine-tuning mix.

**Failure 2: alignment tax in non-English outputs.** LoRA + instruction tuning on an English-heavy SFT dataset degrades Chinese/Korean/Japanese output quality even when the base model was multilingual. The LoRA adapter learns English-shaped responses. Fix: include 20%+ target-language examples in your SFT data.

**Failure 3: merged LoRA != applied LoRA, sometimes.** When you merge LoRA weights into the base model for deployment (`model.merge_and_unload()`), in 4-bit quantised setups the merged model can perform measurably worse than the unmerged one because the quantisation re-rounds the now-perturbed base weights. Fix: either keep LoRA separate at inference (small latency cost), or do post-merge quantisation calibration on a held-out set.

---

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
