---
title: "NLP Part 6: GPT and Generative Language Models"
date: 2025-08-31 09:00:00
tags:
  - NLP
  - GPT
  - Deep Learning
  - Language Models
categories: Natural Language Processing
series:
  name: "Natural Language Processing"
  part: 6
  total: 12
lang: en
mathjax: true
description: "From GPT-1 to GPT-4: understand autoregressive language modeling, decoding strategies (greedy, beam search, top-k, top-p), in-context learning, and build a chatbot with HuggingFace."
disableNunjucks: true
---

When you ask ChatGPT a question and a fluent multi-paragraph answer streams back token by token, you are watching a single deceptively simple loop: feed everything-so-far into a Transformer decoder, look at the probability distribution it produces over the vocabulary, pick one token, append it, repeat. That is *all* an autoregressive language model does. The miracle is not the loop -- it is what happens when you scale the network behind the loop to hundreds of billions of parameters and train it on most of the internet.

If BERT (Part 5) is the king of *understanding*, GPT is the king of *generation*. This article walks the full GPT lineage, opens up the mechanics of autoregressive decoding, makes the choice of decoding strategy visceral, and ends with a working chatbot you can run on your laptop.

## What you will learn

- How a decoder-only Transformer turns "predict the next token" into a general-purpose AI
- The role of **causal (masked) self-attention** -- the one design choice that separates GPT from BERT
- The four canonical decoding strategies (greedy, beam search, top-k, top-p) and the role of **temperature**
- Why **in-context learning** (zero / few-shot) works *without any gradient updates*
- The empirical **scaling laws** and the strange phenomenon of **emergent capabilities**
- How to evaluate generated text (BLEU, ROUGE, perplexity) and where each metric breaks
- A working multi-turn chatbot built on GPT-2 with HuggingFace

**Prerequisites**: Part 4 (Transformer architecture), Part 5 (pretraining and BERT).

---

## 1. The decoder-only Transformer

![Decoder-only architecture and the causal mask](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/fig1_decoder_only_arch.png)

The original Transformer (Vaswani et al. 2017) had two halves: an encoder that read the source sentence and a decoder that wrote the target sentence. BERT kept only the encoder. GPT keeps only the *decoder*, with one critical modification: every self-attention layer uses a **causal mask** so that position $i$ can attend to positions $1, 2, \ldots, i$ but never to anything in the future.

Why does this single design choice matter so much? Because it makes training and inference *identical*. During training the model sees the whole sentence at once, but the mask hides the future, so each position is forced to predict its successor using only the past -- exactly the situation it will face at generation time. There is no train/inference mismatch, no separate "generation mode": the same forward pass that computes the loss during training also computes the next-token distribution at inference.

### The forward pass, in one breath

Given input tokens $x_1, \ldots, x_t$:

1. **Embed**: $h^0_i = E_{\text{tok}}(x_i) + E_{\text{pos}}(i)$
2. **Repeat $L$ times** (one Transformer block):

$$
\tilde h = h + \text{MaskedMHA}(\text{LN}(h)), \quad h \leftarrow \tilde h + \text{FFN}(\text{LN}(\tilde h))
$$

3. **Project**: logits $z_i = W_o\,h^L_i$, then $P(x_{i+1}\mid x_{\le i}) = \text{softmax}(z_i)$.

### Causal mask, made concrete

The masked attention is the same scaled dot-product as before, with a $-\infty$ added to forbidden positions before the softmax:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}} + M\right) V,
\quad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

The $-\infty$ becomes $0$ after softmax, so future tokens contribute nothing. The right panel above visualises this: each row is a query position, each column a key, and only the lower triangle (the past) is alive.

### Training objective

Maximise the log-likelihood of the corpus -- which is just per-position cross-entropy summed over the sequence:

$$
\mathcal{L} = -\sum_{i=1}^{n} \log P(x_i \mid x_1, \ldots, x_{i-1})
$$

That single loss, applied to terabytes of text, is the entire training story.

---

## 2. Autoregressive generation, step by step

![Autoregressive generation: one token at a time](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/fig2_autoregressive_step.png)

At inference time the model produces one token at a time. Three things happen at every step:

1. **Forward pass** the current sequence to obtain logits at the *last* position only.
2. **Convert logits to a probability distribution** over the vocabulary (with optional temperature).
3. **Choose** the next token (deterministically or by sampling), append it, and repeat until you hit `<eos>` or a length limit.

The bottom panel above shows the next-token distribution after the prompt `"The cat sat on the"`: `mat` wins with ~42% probability, but plenty of probability mass is sprinkled across plausible alternatives (`floor`, `couch`, `sofa`, ...). Whether your model writes the same thing every time, or surprises you, depends entirely on **how you sample from this distribution** -- which we get to in Section 4.

> **A practical note on speed**. Naïvely, generating $T$ tokens means running $T$ forward passes whose cost grows quadratically with sequence length -- prohibitive. In practice every implementation uses a **KV cache**: keys and values from past tokens are stored once and reused, so each new token only triggers attention against cached vectors. This turns generation from $O(T^3)$ into $O(T^2)$ aggregate FLOPs and is the reason real systems are usable at all.

---

## 3. The GPT family: 5 years, $\sim$10,000$\times$ growth

![GPT-1 to GPT-4 evolution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/fig4_gpt_evolution.png)

The GPT series tells the story of what happens when you take a small idea and crank the scale knob.

### GPT-1 (Jun 2018) -- proof of concept

- 12-layer decoder, **117 M** parameters, BooksCorpus (~4.5 GB).
- Recipe: pretrain with the language-modelling objective, then **fine-tune** a small task head per downstream task.
- Lesson: a generic Transformer decoder, pretrained on raw text, beats hand-engineered task-specific architectures across a wide benchmark suite.

### GPT-2 (Feb 2019) -- zero-shot emerges

- **1.5 B** parameters, WebText (~40 GB scraped from outbound Reddit links).
- New idea: skip fine-tuning entirely. If you simply *describe* a task in the prompt, the model often does it.

```text
Translate to French:
English: The cat sat on the mat.
French:
```

GPT-2 will produce a passable translation despite never having been trained on parallel data. It saw enough bilingual passages on the web to absorb the *pattern* "English: ... / French: ...".

### GPT-3 (May 2020) -- scale changes the game

- **175 B** parameters (~$100\times$ GPT-2), ~570 GB of curated text, ~3.14$\times 10^{23}$ FLOPs of training compute.
- The headline discovery: **few-shot in-context learning**. Drop a handful of input-output examples into the prompt and the model picks up the task on the fly, *with no gradient updates*.
- Many capabilities (3-digit arithmetic, code completion, multi-step reasoning) were essentially absent at GPT-2 scale and suddenly present at GPT-3 scale -- the **emergence** phenomenon we revisit in Section 7.

### GPT-4 (Mar 2023) -- multimodal and instruction-tuned

- Architecture and parameter count never officially disclosed (rumoured to be a Mixture-of-Experts in the trillion-parameter range).
- Accepts both text and images.
- Heavily refined with **RLHF** (Reinforcement Learning from Human Feedback) for safety, helpfulness, and instruction following.
- Reaches the 90th percentile on the bar exam, scores 5 on AP exams, and handles long-horizon reasoning that earlier models could not.

| Model | Parameters | Training data | Headline capability |
|-------|-----------|---------------|---------------------|
| GPT-1 | 117 M     | 4.5 GB        | Pretrain + fine-tune works |
| GPT-2 | 1.5 B     | 40 GB         | Zero-shot from prompts |
| GPT-3 | 175 B     | 570 GB        | Few-shot in-context learning |
| GPT-4 | (undisclosed) | (undisclosed) | Multimodal, strong reasoning, RLHF |

---

## 4. Decoding strategies

The choice of decoding strategy can change a generation from *repetitive and lifeless* to *surprising and on-topic* without retraining a single weight. Below we visualise the four canonical strategies on the **same** next-token distribution from Section 2.

![Greedy, top-k, top-p, and temperature on the same distribution](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/fig5_sampling_strategies.png)

### 4.1 Greedy decoding

Always pick the highest-probability token: $x_t = \arg\max_w P(w \mid x_{<t})$.

```python
def greedy_decode(model, tokenizer, prompt, max_new=100):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :]
        nxt = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)
```

**Pros**: deterministic, fast. **Cons**: drifts into degenerate loops (`"the the the"`) and produces dull, predictable text. Use only when you need reproducibility for testing.

### 4.2 Beam search

Maintain the top-$k$ partial sequences at every step, ranked by *cumulative* log-probability (with an optional length penalty to avoid favouring short sequences).

```python
def beam_search(model, tokenizer, prompt, beam=5, max_new=100, lp=0.6):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    beams = [(ids, 0.0)]
    for _ in range(max_new):
        cands = []
        for seq, score in beams:
            if seq[0, -1].item() == tokenizer.eos_token_id:
                cands.append((seq, score)); continue
            logp = torch.log_softmax(model(seq).logits[:, -1, :], dim=-1)
            top_lp, top_id = logp.topk(beam, dim=-1)
            for p, idx in zip(top_lp[0], top_id[0]):
                new_seq = torch.cat([seq, idx.view(1, 1)], dim=1)
                cands.append((new_seq, score + p.item()))
        cands.sort(key=lambda x: x[1] / (x[0].size(1) ** lp), reverse=True)
        beams = cands[:beam]
    return tokenizer.decode(beams[0][0][0], skip_special_tokens=True)
```

**Pros**: higher likelihood than greedy, the workhorse of machine translation and summarisation. **Cons**: tends to produce *bland*, generic outputs in open-ended generation -- a phenomenon called **beam-search curse** (the highest-likelihood sentences are often the most boring).

### 4.3 Top-$k$ sampling

Sample only from the $k$ most probable tokens (re-normalising their probabilities to sum to 1):

```python
def top_k_sample(model, tokenizer, prompt, k=50, T=1.0, max_new=100):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :] / T
        top_logits, top_ids = logits.topk(k, dim=-1)
        probs = torch.softmax(top_logits, dim=-1)
        idx = torch.multinomial(probs, 1)
        nxt = top_ids.gather(-1, idx)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)
```

**Caveat**: $k$ is fixed. If the model is very confident (one token has 95% mass), top-$k$ still considers $k$ alternatives and may pick something silly. If the model is genuinely uncertain across hundreds of tokens, $k=50$ is too restrictive.

### 4.4 Top-$p$ (nucleus) sampling

Pick the smallest set of tokens whose cumulative probability $\ge p$ -- the **nucleus** -- and sample from it. The size of the nucleus *adapts* to the model's confidence.

```python
def top_p_sample(model, tokenizer, prompt, p=0.9, T=1.0, max_new=100):
    ids = tokenizer(prompt, return_tensors="pt").input_ids
    for _ in range(max_new):
        logits = model(ids).logits[:, -1, :] / T
        sorted_logits, sorted_ids = logits.sort(descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # keep tokens up to and including the one that crosses p
        keep = cum <= p
        keep[..., 0] = True               # always keep the top token
        keep[..., 1:] = keep[..., :-1].clone() | (cum[..., :-1] <= p)
        sorted_logits[~keep] = -float("inf")
        nucleus_probs = torch.softmax(sorted_logits, dim=-1)
        idx = torch.multinomial(nucleus_probs, 1)
        nxt = sorted_ids.gather(-1, idx)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == tokenizer.eos_token_id:
            break
    return tokenizer.decode(ids[0], skip_special_tokens=True)
```

In panel (c) above the nucleus has 5 tokens because the top 5 already cover 85% of the mass. On a different distribution it could be 1 token (very confident model) or 200 (very flat distribution). This adaptivity is why top-$p$ has become the default for open-ended generation.

### 4.5 Temperature

Temperature $T$ rescales the logits *before* the softmax:

$$
P_T(w) = \text{softmax}(z / T)
$$

Panel (d) above shows the same logits at $T = 0.5$ and $T = 1.5$. Intuitively, $T$ controls how *peaky* the distribution is:

- $T \to 0$: distribution collapses to a one-hot at the argmax (equivalent to greedy).
- $T = 1$: original distribution (no change).
- $T \to \infty$: distribution flattens to uniform.

A practical rule of thumb: **top-$p$ = 0.9 with $T$ = 0.7-0.9** is the default for most chat / creative-writing applications.

### Strategy cheat sheet

| Strategy   | Diversity | Quality        | Speed  | Best for                      |
|------------|-----------|----------------|--------|-------------------------------|
| Greedy     | None      | Low            | Fast   | Reproducible debugging        |
| Beam       | Low       | High likelihood| Slow   | Translation, summarisation    |
| Top-$k$    | Medium    | Medium-high    | Medium | General-purpose generation    |
| Top-$p$    | Medium-high | High         | Medium | Chat, story-writing, dialogue |

---

## 5. Scaling laws: predictability before emergence

![Scaling laws: loss decreases as a power law](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/fig3_scaling_laws.png)

In 2020 Kaplan et al. discovered that test loss decreases as a *clean power law* in three quantities -- model parameters $N$, dataset size $D$, and training compute $C$ -- as long as none of the three is the bottleneck:

$$
L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N},\quad L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}
$$

with empirically tiny exponents ($\alpha_N \approx 0.076$, $\alpha_C \approx 0.050$). On log-log axes these become straight lines, which is why both panels above are linear. Two consequences:

1. **You can predict** the loss of a 175 B-parameter model from runs at $\le 1$ B. This is what made the leap to GPT-3 economically defensible -- the team knew, before spending millions of dollars on GPUs, roughly where the loss curve would land.
2. **There is an irreducible loss floor** (the dashed line) -- the entropy of the data itself. No amount of scaling crosses it.

The 2022 **Chinchilla** paper later refined the picture: for a fixed compute budget, GPT-3 was substantially *under-trained*. The compute-optimal recipe is roughly **20 tokens per parameter** -- so a 70 B model trained on 1.4 T tokens (Chinchilla) outperforms a 175 B model trained on 300 B tokens (GPT-3) at the same compute. Modern open-weights models (LLaMA, Mistral, Qwen) all follow Chinchilla-style scaling.

---

## 6. Emergent capabilities

![Emergent capabilities: sharp phase transitions with scale](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/fig6_emergent_capabilities.png)

Scaling laws say the *aggregate loss* decreases smoothly. But many *individual tasks* do not improve smoothly -- they sit at chance for orders of magnitude of scale and then suddenly snap to high accuracy.

The figure shows the qualitative shape: sentiment classification (blue) improves smoothly from small scale; few-shot in-context learning, 3-digit arithmetic, and chain-of-thought reasoning (purple, amber, green) stay near random until a critical model size, then shoot up. Wei et al. (2022) catalogued 137 such tasks in the **BIG-Bench** benchmark.

Whether emergence is "real" or an artefact of how we measure (using exact-match accuracy on a discrete task) is still debated -- Schaeffer et al. (2023) showed that switching to a continuous metric makes some emergence curves look smooth. But the operational fact remains: at small scale the model *cannot* do the task; somewhere between $10^{10}$ and $10^{11}$ parameters it can. Plan your roadmap accordingly.

---

## 7. In-context learning

![Few-shot in-context learning](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/gpt-generative-models/fig7_in_context_learning.png)

Possibly the most surprising property of large GPT-style models: you can teach them new tasks *just by showing examples in the prompt*. No backprop, no optimiser, no parameter updates -- the model adapts purely through what it reads.

### Zero-shot vs. few-shot

```text
# Zero-shot
Translate the following English sentence to French:
The bird flew in the sky.
French:

# Few-shot
Translate English to French.
English: The cat sat on the mat. -> French: Le chat s'est assis sur le tapis.
English: The dog ran in the park. -> French: Le chien a couru dans le parc.
English: The bird flew in the sky. -> French:
```

The few-shot prompt gives the model a *format* to imitate. Empirically, 2-5 examples are enough for most tasks; returns flatten beyond about 8.

### Why does this work?

The mechanism is still an active research area, but the leading explanations are:

- **Pattern matching at scale.** Pretraining on the web exposes the model to countless input-output formats; the prompt activates a matching template.
- **Implicit gradient descent.** Garg et al. (2022) and others show that, in toy linear-regression settings, a Transformer can implement a *single step of gradient descent* in its forward pass over the in-context examples. Larger models, more steps.
- **Bayesian latent-task inference.** Xie et al. (2022) frame ICL as inferring a hidden "task" variable from the examples and then conditioning generation on it.
- **Phase transition with scale.** ICL only works above a few-billion parameter threshold -- consistent with the emergence story.

### Practical prompt-engineering recipe

```python
def few_shot_prompt(task, examples, query):
    parts = [task, ""]
    for inp, out in examples:
        parts.append(f"Input: {inp}\nOutput: {out}\n")
    parts.append(f"Input: {query}\nOutput:")
    return "\n".join(parts)
```

A few rules of thumb that consistently help:

1. **Be explicit.** State the task in plain language at the top.
2. **Pick representative examples.** Cover the variation you expect at test time.
3. **Keep formatting consistent.** Same delimiters, same casing, same labels.
4. **For reasoning tasks, use chain-of-thought.** Ask the model to "think step by step" or include worked-through reasoning in your few-shot examples. This unlocks substantial accuracy gains on math and logic benchmarks.

---

## 8. Evaluating generated text

There is no perfect metric -- but knowing which approximate metric to use, and what it misses, is essential.

### BLEU (translation)

Measures n-gram precision of the generation against one or more references, with a brevity penalty:

$$
\text{BLEU} = \text{BP}\cdot\exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)
$$

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
def bleu(generated: str, reference: str) -> float:
    return sentence_bleu([reference.split()], generated.split(),
                         smoothing_function=SmoothingFunction().method1)
```

Strong correlation with human judgement *for translation*; weak for open-ended generation.

### ROUGE (summarisation)

Measures n-gram **recall** -- how much of the reference appears in the generation:

```python
from rouge_score import rouge_scorer
def rouge(generated: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(reference, generated)
    return {k: v.fmeasure for k, v in s.items()}
```

### Perplexity (intrinsic)

How "surprised" the model is by held-out text. Lower is better:

$$
\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log P(w_i\mid w_{<i})\right)
$$

```python
def perplexity(model, tokenizer, text: str) -> float:
    enc = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**enc, labels=enc.input_ids).loss
    return torch.exp(loss).item()
```

Perplexity is a useful *intrinsic* signal during training, but **a low perplexity does not mean a good chatbot** -- the model could be excellent at predicting common patterns and still hallucinate.

### Which metric for which task?

| Task                  | Recommended                     |
|-----------------------|---------------------------------|
| Machine translation   | BLEU, chrF, COMET (learned)     |
| Summarisation         | ROUGE, BERTScore                |
| Open-ended dialogue   | Human eval + diversity (distinct-n) |
| Instruction following | LLM-as-judge + human eval       |
| Language modelling    | Perplexity                      |

For anything user-facing, **human evaluation remains the gold standard**.

---

## 9. Build a chatbot with GPT-2

GPT-2 is small enough to run on CPU and large enough to be interesting. The HuggingFace `transformers` library hides almost all of the boilerplate.

### Single-turn

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class ChatBot:
    def __init__(self, name: str = "gpt2") -> None:
        self.tokenizer = GPT2Tokenizer.from_pretrained(name)
        self.model = GPT2LMHeadModel.from_pretrained(name).eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def respond(self, user: str, max_new: int = 100,
                temperature: float = 0.8, top_p: float = 0.9) -> str:
        prompt = f"User: {user}\nAssistant:"
        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        return text.split("Assistant:")[-1].strip()


print(ChatBot().respond("What is machine learning?"))
```

### Multi-turn (with rolling history)

```python
class MultiTurnBot(ChatBot):
    def __init__(self, name: str = "gpt2") -> None:
        super().__init__(name)
        self.history: list[str] = []

    def respond(self, user: str, max_turns: int = 5, **gen_kwargs) -> str:
        # Keep only the last N turns to fit in the 1024-token context window
        recent = self.history[-2 * max_turns:]
        prompt = "\n".join(recent + [f"User: {user}", "Assistant:"])
        ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                ids,
                max_new_tokens=gen_kwargs.get("max_new", 100),
                do_sample=True,
                temperature=gen_kwargs.get("temperature", 0.8),
                top_p=gen_kwargs.get("top_p", 0.9),
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        reply = self.tokenizer.decode(out[0], skip_special_tokens=True)
        reply = reply.split("Assistant:")[-1].split("User:")[0].strip()
        self.history += [f"User: {user}", f"Assistant: {reply}"]
        return reply
```

### One-liner with the pipeline API

```python
from transformers import pipeline
gen = pipeline("text-generation", model="gpt2")
print(gen("User: What is deep learning?\nAssistant:",
          max_new_tokens=80, do_sample=True, temperature=0.8, top_p=0.9)[0]["generated_text"])
```

> GPT-2 will *not* sound like ChatGPT -- it has neither instruction-tuning nor RLHF. To get ChatGPT-like behaviour you need either an instruction-tuned open model (e.g. `meta-llama/Llama-3-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`) or to fine-tune GPT-2 yourself on an instruction dataset.

---

## 10. GPT vs. BERT, in one table

| Aspect        | BERT                                      | GPT                                         |
|---------------|-------------------------------------------|---------------------------------------------|
| Architecture  | Encoder, bidirectional self-attention     | Decoder, **causal** self-attention          |
| Pretraining   | Masked LM + Next-Sentence Prediction      | Causal language modelling                   |
| Context       | Sees both left and right                  | Sees only the left (past)                   |
| Sweet spot    | Classification, NER, QA, retrieval        | Generation, dialogue, code, creative text   |
| Adaptation    | Fine-tune a task head per task            | Prompt (zero/few-shot) or fine-tune         |
| Scaling story | Returns diminish above ~1 B params        | Returns continue; emergent abilities appear |

A useful mental model: **BERT is a search engine**, **GPT is a writer**. Use BERT-family models when you need a vector representation of an input; use GPT-family models when you need to produce output text.

---

## 11. Limitations to be honest about

- **Hallucination.** Producing fluent but factually wrong text is the model's default mode for anything it does not know. Mitigations: retrieval-augmentation (Part 10), tool use, calibrated refusal training.
- **Context length.** GPT-2 had 1024 tokens; GPT-3 had 2048; GPT-4 went to 32 K and then 128 K; today 1 M is becoming common. But long contexts are expensive and recall in the *middle* of a long context degrades ("lost in the middle").
- **Compute cost.** A single forward pass of a 175 B-parameter model needs hundreds of GB of GPU memory. Inference cost dominates training cost over a model's lifetime.
- **Training-data bias.** The model reproduces whatever statistical patterns -- including stereotypes -- exist in its training data.
- **Limited controllability.** Hard guardrails on style, format, or factuality require additional machinery (system prompts, constrained decoding, RLHF, post-hoc filters).

---

## 12. Key takeaways

- A GPT model is just a **decoder-only Transformer with a causal mask** trained to predict the next token. The simplicity is the point.
- **Scale is the multiplier**: parameters, data, and compute together produce a clean power-law improvement in loss, with sudden capability jumps on top.
- **In-context learning** lets a single fixed model handle thousands of tasks via prompts -- no fine-tuning required for most use-cases.
- **Decoding strategy** is a free lever: prefer **top-$p$ + temperature 0.7-0.9** for chat and creative writing; greedy or beam for translation and reproducibility.
- **BERT understands; GPT generates.** Together they cover the full spectrum of modern NLP, and Parts 7-12 will dive into how to wield GPT-style models effectively (prompting, fine-tuning, RAG, multimodal, ...).

---

## Series navigation

| Part | Topic | Link |
|------|-------|------|
| 1 | Introduction and Text Preprocessing | [Read](/en/nlp-introduction-and-preprocessing/) |
| 2 | Word Embeddings and Language Models | [Read](/en/nlp-word-embeddings-lm/) |
| 3 | RNN and Sequence Modeling | [Read](/en/nlp-rnn-sequence-modeling/) |
| 4 | Attention and Transformer | [Read](/en/nlp-attention-transformer/) |
| 5 | BERT and Pretrained Models | [Previous](/en/nlp-bert-pretrained-models/) |
| **6** | **GPT and Generative Models (this article)** | |
| 7 | Prompt Engineering and In-Context Learning | [Next](/en/nlp-prompt-engineering-icl/) |
| 8 | Fine-tuning and PEFT | [Read](/en/nlp-fine-tuning-peft/) |
