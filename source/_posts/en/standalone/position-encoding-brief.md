---
title: "Position Encoding Brief: From Sinusoidal to RoPE and ALiBi"
date: 2024-12-30 09:00:00
tags:
  - NLP
  - Transformer
  - Deep Learning
  - LLM
categories: Algorithm
lang: en
mathjax: true
description: "A practitioner's tour of Transformer position encoding: why attention needs it at all, how sinusoidal/learned/relative/RoPE/ALiBi schemes differ, and which one to pick when long-context extrapolation matters."
disableNunjucks: true
---

Self-attention has a strange property that surprises most people the first time they compute it by hand: it does not know the order of its inputs. Permute the tokens and every attention score is permuted along with them — the function is exactly equivariant. So before we can do anything useful with a Transformer, we have to inject position information from the outside.

That single design decision — *how* to inject it — has spawned a remarkable amount of research. Sinusoidal, learned, relative, T5-style buckets, RoPE, ALiBi, NoPE, and more. This post is a practitioner's brief: enough math to know why each scheme works, enough comparison to choose one, and a clear focus on the property that matters most in the LLM era — **length extrapolation**, the ability to handle sequences longer than anything seen in training.

## What You Will Learn

- Why attention is permutation-equivariant and what that means for design
- The two big families: **absolute** vs. **relative** position encoding
- Sinusoidal, learned, T5 buckets, RoPE, ALiBi — what they actually compute
- Which schemes extrapolate to longer contexts and which collapse
- A short selection guide for new architectures

## Prerequisites

- Familiarity with self-attention ($Q$, $K$, $V$, softmax)
- Comfort with basic linear algebra (matrices, dot products, rotations)

---

# Why Attention Needs Position At All

Given input embeddings $X = [x_1, x_2, \ldots, x_n] \in \mathbb{R}^{n \times d}$, scaled dot-product attention computes

$$
\text{Attn}(X) = \text{softmax}\!\left(\tfrac{QK^\top}{\sqrt{d_k}}\right) V,
\quad Q = XW_Q,\; K = XW_K,\; V = XW_V.
$$

If we permute the rows of $X$ by any permutation matrix $P$, then $Q$, $K$, and $V$ permute the same way, and the attention output is also permuted by $P$. In symbols, $\text{Attn}(PX) = P\,\text{Attn}(X)$.

That's what "the model has no sense of order" means precisely: the function is **equivariant** to permutations of the input. It treats "the cat sat on the mat" the same as "the mat sat on the cat" — same multiset of token embeddings in, same multiset of vectors out, just shuffled. We must inject order somehow, or the model can never tell the two sentences apart.

The space of solutions divides into two clean families.

---

# Family 1: Absolute Position Encoding

The idea is the simplest one possible: assign each position $p$ its own vector $e_p \in \mathbb{R}^d$, and add it to the token embedding before the first attention layer:

$$
\tilde{x}_p = x_p + e_p.
$$

Now the input at position 1 is genuinely different from the input at position 2, even if both are the word "the". Attention can read these distinct vectors and pick up order.

The question is: where do the $e_p$ come from?

## 1.1 Learned Position Embeddings

The most direct answer: make $E \in \mathbb{R}^{L_{\max} \times d}$ a trainable parameter matrix and let SGD figure it out. This is what the original BERT and GPT-2 use.

**Pros.**

- Maximum flexibility: the model learns whatever positional structure the data actually needs.
- Very simple to implement: it is just an `nn.Embedding(L_max, d)`.

**Cons.**

- **Hard cap on sequence length.** You picked $L_{\max}$ at training time, and rows beyond it simply do not exist. To go longer, you must reinitialize and continue training.
- **No extrapolation.** Even within $L_{\max}$, positions that were rare in training get rare gradients and often look noisy. Push past $L_{\max}$ and the model has nothing to rely on.

![Learned positional encoding: random initialization sharpens into smooth structure during training, but performance falls off a cliff past the training context length](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/position-encoding-brief/fig2_learned.png)

The right panel above is the punchline: an absolute learned PE is a good in-distribution function and an *undefined* function past $L_{\max}$. For a chat model that needs to handle long documents, this is fatal.

## 1.2 Sinusoidal Position Encoding

The "Attention Is All You Need" paper proposes a hand-designed alternative. For position $p$ and dimension index $i$ (with $d$ even), define

$$
\text{PE}(p, 2i)   = \sin\!\bigl(p / 10000^{2i/d}\bigr), \qquad
\text{PE}(p, 2i+1) = \cos\!\bigl(p / 10000^{2i/d}\bigr).
$$

Each pair of dimensions $(2i, 2i+1)$ is a sine/cosine at a frequency that decreases geometrically with $i$. The lowest dimensions wiggle fast (period $2\pi$); the highest dimensions barely move over the entire sequence (period $\sim 2\pi \cdot 10000$). Each position thus gets a *unique multi-scale fingerprint*.

![Sinusoidal positional encoding: heatmap of dim x position (left), and the per-dimension fingerprint of a single position (right)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/position-encoding-brief/fig1_sinusoidal.png)

**Why these specific frequencies?** Two reasons.

1. **Linear-relative property.** Because $\sin(p+k)$ and $\cos(p+k)$ are linear combinations of $\sin(p)$, $\cos(p)$ with coefficients that depend only on $k$, the encoding for $p+k$ is a fixed linear transform of the encoding for $p$. Attention, which applies linear projections to $Q$ and $K$, can in principle learn to read off relative offsets. This is the "free" relative information that everybody talks about.

2. **No length cap.** The formula evaluates at any $p \in \mathbb{R}$. You can ask for position 100,000 even though training only saw position 512.

**Pros.**

- Parameter-free; no embedding table to size.
- Defined for all positions, with a smooth, principled extrapolation.
- The linear-relative trick gives attention a head start on relative reasoning.

**Cons.**

- **The "free" relative information is only theoretical.** Modern Transformers don't actually exploit it after the first layer because LayerNorm and FFN destroy the linear structure.
- **Empirical extrapolation is poor.** Despite the elegant formula, perplexity rises sharply past the training length — often more sharply than for ALiBi. The reason is that *attention weights*, not the encoding itself, are what need to remain sensible at long range, and there is no mechanism keeping them so.

In practice: sinusoidal PE was a beautiful idea that was overtaken by what came next.

## 1.3 Other Absolute Variants (Brief)

Two more schemes are worth mentioning quickly so you recognize them.

- **Recurrent / FLOATER (Liu et al., 2020).** Replace the lookup table with a learned ODE: $e_{p+1} = e_p + f_\theta(e_p, p)$. The recurrence gives extrapolation but breaks parallelism — exactly the trade-off Transformers were designed to avoid.

- **Multiplicative encoding.** Instead of $\tilde{x}_p = x_p + e_p$, use $\tilde{x}_p = x_p \odot e_p$. Mostly an academic curiosity; in practice it is hard to train stably.

The ceiling of pure absolute encoding is set by the same fact for all of these: position is glued onto the token *before* any attention, and after a few layers the link to "this token is at offset $k$ from that token" is gone.

---

# Family 2: Relative Position Encoding

The shift in viewpoint that powers everything modern: in language, **what matters is usually the *gap* between two tokens, not where either of them sits in the document.** "The cat sat on the mat" means the same thing whether it appears on page 1 or page 47. So inject the relative offset $i - j$ directly into attention, where it is actually used.

## 2.1 The Original Shaw-Style Formulation

Shaw et al. (2018) introduced relative position into the attention dot product as a learned bias on the keys:

$$
\text{score}(i, j) = \frac{q_i^\top \bigl(k_j + r_{i-j}\bigr)}{\sqrt{d_k}},
$$

where $r_{i-j} \in \mathbb{R}^{d_k}$ is a learned vector indexed by the *signed offset* $i - j$. Offsets larger than some clipping radius $K$ collapse to a single bucket, so a finite vocabulary of offsets covers any sequence length.

**Why this works.** The model can directly learn things like "tokens 1 step away matter more than tokens 5 steps away" without ever needing the absolute position. And because offsets repeat across the sequence, training data for "offset = 3" comes from every position pair separated by 3, not just one.

**Limitation.** Each layer adds a $|R| \times d_k$ table, and the attention computation now includes a per-pair lookup, which is awkward to fuse into the matmul.

## 2.2 T5 Bias: As Simple As It Gets

T5 (Raffel et al., 2020) takes the simplification one step further. Drop the vector $r_{i-j}$ and use a *scalar* bias added directly to the score:

$$
\text{score}(i, j) = \frac{q_i^\top k_j}{\sqrt{d_k}} + b_{B(i-j)},
$$

where $B(\cdot)$ is a "bucketing" function that maps small offsets to themselves and large offsets logarithmically into a small number of bins (typically 32). Each head learns its own table of $\sim 32$ bias scalars. That's it.

T5 also drops the position bias on the value vectors entirely. Empirically nothing breaks, and the math is dramatically simpler.

## 2.3 DeBERTa: Disentangled Attention

DeBERTa (He et al., 2021) goes the other direction and *expands* the relative formulation. If you expand $(x_i + e_i)^\top (x_j + e_j)$ you get four terms: content–content, content–position, position–content, position–position. T5 keeps content–content plus a position–position scalar; DeBERTa keeps the cross terms (content–position and position–content) and drops the position–position one. The intuition is that "what does this token attend to at relative distance $k$?" is genuinely informative, and a content-aware version of that should be even better.

DeBERTa held the SuperGLUE crown briefly. The cost is more parameters and more attention complexity.

---

# RoPE: Rotation in Disguise

RoPE (Su et al., 2021) is the position encoding that powers most modern open-source LLMs — LLaMA, Qwen, Mistral, Yi, DeepSeek, Gemma, all RoPE. It deserves its own section because it manages a magic trick: it looks like an *absolute* encoding (apply a transform to each token's $q$ and $k$ based on its absolute position) but the resulting attention score depends only on the *relative* offset. Best of both worlds.

## 3.1 The Construction

Treat each pair of dimensions in the query/key vectors as a 2-D plane. For position $m$, rotate that plane by an angle $m \theta_g$, where $\theta_g$ is a fixed frequency for the $g$-th pair (chosen on a geometric schedule, just like sinusoidal):

$$
R_m^{(g)} =
\begin{pmatrix}
  \cos m\theta_g & -\sin m\theta_g \\
  \sin m\theta_g &  \cos m\theta_g
\end{pmatrix}.
$$

Apply these block-diagonal rotations to $q$ and $k$:

$$
\tilde q_m = R_m\, q_m, \qquad \tilde k_n = R_n\, k_n.
$$

Now compute the inner product:

$$
\tilde q_m^\top \tilde k_n
= q_m^\top R_m^\top R_n\, k_n
= q_m^\top R_{n-m}\, k_n.
$$

The two absolute rotations collapse into a *single* rotation by the relative offset $n - m$. The attention score now depends only on the gap.

![RoPE rotates each (q, k) pair by an angle proportional to its absolute position; the resulting inner product depends only on the relative offset](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/position-encoding-brief/fig3_rope.png)

## 3.2 Why It Works So Well

Three properties make RoPE the default for modern LLMs.

1. **Relative by construction.** Unlike sinusoidal, where the relative information has to be recovered through linear projections that LayerNorm later destroys, RoPE bakes the relative dependence into the *score itself* at every layer.

2. **Length-agnostic.** The rotation matrix $R_m$ is defined for any real $m$, just like $\sin(m\theta)$. No embedding table to grow.

3. **Cheap.** Block rotations cost the same as a single complex multiplication per dimension pair; in code it's a few `cos`/`sin` ops and elementwise products. Negligible on top of the attention matmul.

## 3.3 Extending RoPE to Longer Context

RoPE doesn't extrapolate perfectly past the training length either — perplexity rises, just less catastrophically than sinusoidal. The community has invested heavily in fixes:

- **Position interpolation (PI, Chen et al., 2023).** Scale every position by $L_{\text{train}} / L_{\text{test}}$ before applying RoPE so that, e.g., position 4096 uses the angles that training saw at position 2048. Cheap, requires a short fine-tune.
- **NTK-aware scaling and YaRN (Peng et al., 2023).** Rescale only the *low-frequency* dimensions (the ones whose angles barely change across the training range), leaving high-frequency dimensions alone. Better long-context fidelity.
- **LongRoPE (Ding et al., 2024).** Search for non-uniform per-dimension scaling factors. Pushes useful context to 2 M tokens with minimal fine-tuning.

These are a major reason every new LLM ships with "RoPE base = 10000" in its config and "RoPE scaling = ..." in the inference setup.

---

# ALiBi: The Most Pragmatic Choice for Extrapolation

ALiBi (Press et al., 2022) might be the most surprising entry in the list because it is so simple. No embeddings. No rotations. Just add a fixed, head-specific *linear penalty* to the attention scores:

$$
\text{score}(i, j) = \frac{q_i^\top k_j}{\sqrt{d_k}} - m_h \cdot |i - j|.
$$

The slope $m_h$ is fixed per head on a geometric schedule (head 1: $m_1 = 1/2$; head 2: $m_2 = 1/4$; ...); nothing is learned.

![ALiBi: per-head linear distance bias added to attention scores; different heads decay at different rates](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/position-encoding-brief/fig4_alibi.png)

That's it. The right panel above shows the design intent: head 1 attends locally (steep slope, "what just happened"), while head 8 attends almost globally (gentle slope, "the whole document").

**Why it extrapolates.** The bias is a closed-form function of the offset. Test time can hand the model a 32 K context even though training capped at 2 K, and the bias still does the right thing — it just keeps decaying linearly. No magic, no fine-tune.

**Trade-off.** Empirically, ALiBi tends to be slightly worse than RoPE *at* the training length on standard language modeling, and dramatically better past it. Several papers (BLOOM, MPT) chose it specifically for its extrapolation behavior.

---

# Putting It All Together: The Length Extrapolation Picture

The main practical question in 2026 is: **when context length at inference exceeds training length, what happens to perplexity?**

![Length extrapolation: sinusoidal collapses, RoPE degrades smoothly, ALiBi extrapolates almost for free](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/position-encoding-brief/fig5_compare.png)

The qualitative ordering — absolute PE collapses, RoPE degrades gradually, ALiBi extrapolates almost for free — is consistent across the literature (Press et al. 2022; the RoFormer paper; the various RoPE-scaling reports). The figure is illustrative; exact numbers depend heavily on model, data, and decoding setup.

A few practitioner-level takeaways from this picture.

- **If you cap context at training length** and don't care about going beyond, learned absolute PE is fine and dirt simple. This is what classic BERT does.
- **If you want extrapolation but also want competitive in-domain quality**, RoPE plus a length-extension trick (PI, NTK-aware, YaRN) is the modern default. This is essentially every open LLM today.
- **If you want extrapolation with zero fuss and you can tolerate slightly weaker in-domain numbers**, ALiBi is the most predictable choice. BLOOM, MPT, and Falcon-RW use it.
- **Hybrid schemes** (RoPE plus a small learned bias, or NoPE-like schemes that drop position entirely in some layers) are an active research area; results are promising but unstable across model scales.

---

# Selection Guide

A short decision tree for picking a scheme:

1. **Maximum context $\leq$ training context, encoder-style model (classification/QA on bounded text)**: learned absolute PE. Stop here.
2. **Decoder-style LM, you control training length, want best in-domain quality**: RoPE. Add YaRN/PI at inference if you need to go longer.
3. **Decoder-style LM, you must serve much longer contexts than you trained on, no time to fine-tune**: ALiBi.
4. **You are training a small T5-style encoder-decoder**: T5 bucket bias. It is honestly fine.
5. **You want to publish a paper**: try a new combination, run it past 64K, report perplexity.

---

# Common Misconceptions

A few things that confused me, and seem to confuse a lot of others:

- **"Sinusoidal PE is relative."** It contains relative *information* in a linear-algebraic sense, but the model has to learn to extract it through projections that LayerNorm destroys. Empirically it acts like an absolute encoding.

- **"RoPE has no length limit."** The math has no length limit. The model still has a length limit, because attention scores at offsets it never trained on are essentially out-of-distribution. That's why we need PI/YaRN.

- **"ALiBi is just a heuristic."** It is a heuristic, but the heuristic encodes a strong inductive bias that *closer tokens matter more*, which happens to be true of natural language. That single prior is enough to get strong extrapolation.

- **"You can swap PE schemes after training."** No. The whole network — including LayerNorm gains, attention head specialization, and FFN biases — has co-adapted to the position signal. Swapping it requires re-training or at least a careful long fine-tune.

---

# References

- Vaswani et al., 2017. *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Shaw et al., 2018. *Self-Attention with Relative Position Representations.* [arXiv:1803.02155](https://arxiv.org/abs/1803.02155)
- Dai et al., 2019. *Transformer-XL.* [arXiv:1901.02860](https://arxiv.org/abs/1901.02860)
- Raffel et al., 2020. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).* [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)
- Liu et al., 2020. *Learning to Encode Position for Transformer with Continuous Dynamical Model (FLOATER).* [arXiv:2003.09229](https://arxiv.org/abs/2003.09229)
- He et al., 2021. *DeBERTa: Decoding-enhanced BERT with Disentangled Attention.* [arXiv:2006.03654](https://arxiv.org/abs/2006.03654)
- Su et al., 2021. *RoFormer: Enhanced Transformer with Rotary Position Embedding.* [arXiv:2104.09864](https://arxiv.org/abs/2104.09864) — and the author's blog post [《让研究人员绞尽脑汁的 Transformer 位置编码》](https://spaces.ac.cn/archives/8130).
- Press et al., 2022. *Train Short, Test Long: Attention with Linear Biases (ALiBi).* [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Chen et al., 2023. *Extending Context Window of Large Language Models via Positional Interpolation.* [arXiv:2306.15595](https://arxiv.org/abs/2306.15595)
- Peng et al., 2023. *YaRN: Efficient Context Window Extension of Large Language Models.* [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- Ding et al., 2024. *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.* [arXiv:2402.13753](https://arxiv.org/abs/2402.13753)
