---
title: "NLP Part 4: Attention Mechanism and Transformer"
date: 2025-10-16 09:00:00
tags:
  - NLP
  - Attention
  - Transformer
  - Deep Learning
categories: Natural Language Processing
series:
  name: "Natural Language Processing"
  part: 4
  total: 12
lang: en
mathjax: true
description: "From the bottleneck of Seq2Seq to Attention Is All You Need. Build intuition for scaled dot-product attention, multi-head attention, positional encoding, masking, and assemble a complete Transformer in PyTorch."
disableNunjucks: true
series_order: 4
---

In June 2017, eight researchers at Google Brain and Google Research published a paper with a deliberately bold title: *Attention Is All You Need*. The architecture it introduced, the **Transformer**, threw away recurrence entirely. There were no LSTMs, no GRUs, no left-to-right scanning of a sentence. Instead, every token in a sequence could look at every other token directly through a single mathematical operation: scaled dot-product attention.

That one design decision unlocked massive parallelism on GPUs, eliminated the long-range dependency problems that had plagued RNNs for decades, and became the substrate on which BERT, GPT, T5, LLaMA, Claude, and essentially every modern large language model is built. If you understand this article well, the rest of the series is mostly variations on a theme.

The road from "RNN with attention" to the full Transformer is not long, but every step matters. We will walk it carefully.

## What you will learn

- Why fixed-size context vectors broke vanilla Seq2Seq, and how attention rescued it
- Bahdanau and Luong attention as the conceptual bridge to self-attention
- The Query / Key / Value abstraction, scaled dot-product attention, and the role of the $\sqrt{d_k}$ scale factor
- Multi-head attention and the intuition behind running many "views" in parallel
- Sinusoidal vs. learned positional encoding
- Causal masking, residual connections, and LayerNorm
- A complete from-scratch Transformer in PyTorch you can run on your laptop
- How BERT, GPT, and T5 specialise the same building blocks for different tasks

**Prerequisites**: Part 3 (RNN, Seq2Seq), basic linear algebra (matrix multiplication, softmax), and working PyTorch familiarity.

---

## 1. The bottleneck that motivated attention

Recall the vanilla encoder-decoder from Part 3. An RNN reads the source sentence one token at a time and squashes everything into a single fixed-size vector $c = h_T^{\text{enc}}$. The decoder then generates the target sequence using only that vector.

Imagine translating *"The cat that chased the mouse that ate the cheese was very tired"* into French. The encoder must compress the cat, the mouse, the cheese, the chase, the eating, the tiredness, and the grammatical relationships among them into 512 numbers. The decoder then has to reconstruct everything from those 512 numbers without ever looking at the source again.

This breaks down for two distinct reasons:

- **Information capacity.** A single vector cannot losslessly hold an arbitrarily long sequence. Empirically, BLEU scores for vanilla Seq2Seq drop sharply once sentences exceed roughly 30 tokens.
- **No selective focus.** When the decoder generates the French word for "cat", it should be looking at "cat" in the source, not at "cheese". A static context vector gives equal weight to everything.

A useful analogy: imagine memorising a paragraph, putting it away, then reciting it from memory while typing the translation. Compare that with a human translator who keeps the source in front of them and re-reads the relevant phrase before writing each output word. **Attention is the second strategy implemented inside a neural network.**

---

## 2. Bahdanau attention (2015): looking back at every step

![Cross-attention alignment heatmap](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/attention-transformer/fig1_attention_heatmap.png)

Bahdanau, Cho, and Bengio introduced the first widely used attention mechanism in *Neural Machine Translation by Jointly Learning to Align and Translate*. The idea: instead of one fixed context vector, **compute a different weighted combination of all encoder hidden states at every decoding step**.

At decoder step $t$, with previous decoder state $s_{t-1}$ and encoder states $h_1, \ldots, h_n$:

**Step 1 — score.** A small feed-forward network rates how relevant each encoder state is to the current decoder state.

$$e_{tj} = \mathbf{v}^\top \tanh(W_s s_{t-1} + W_h h_j)$$

**Step 2 — normalise.** A softmax over $j$ turns scores into a probability distribution that sums to 1.

$$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{k=1}^{n} \exp(e_{tk})}$$

**Step 3 — combine.** The context vector is a convex combination of encoder states.

$$c_t = \sum_{j=1}^{n} \alpha_{tj}\, h_j$$

**Step 4 — decode.** The RNN consumes the context vector together with the previous output.

$$s_t = \text{RNN}(s_{t-1}, [c_t; y_{t-1}])$$

The figure above shows what these $\alpha_{tj}$ look like for a small English-to-French example. Each row sums to 1 and the bright cells follow the linguistic alignment (Le ↔ The, chat ↔ cat, tapis ↔ mat). Crucially, **nobody told the model these alignments**. They emerged from training to minimise translation loss. Suddenly, attention was not just a performance trick; it was an interpretable, linguistically meaningful object.

---

## 3. Luong attention: simpler scoring functions

A few months later, Luong, Pham, and Manning proposed simpler alternatives to Bahdanau's small feed-forward scorer:

| Variant   | Score function                                  | Notes                                       |
|-----------|-------------------------------------------------|---------------------------------------------|
| Dot       | $e_{tj} = s_t^\top h_j$                          | Fastest. Requires matching dimensions.      |
| General   | $e_{tj} = s_t^\top W h_j$                        | Learnable bilinear, handles dim mismatch.   |
| Concat    | $e_{tj} = \mathbf{v}^\top \tanh(W [s_t; h_j])$  | Essentially Bahdanau's form.                |

The dot-product variant is the conceptual ancestor of what the Transformer would adopt two years later. Luong also introduced **local attention**, attending only to a window of size $2D+1$ around an alignment point, which reduced cost on long inputs.

---

## 4. The leap: self-attention without recurrence

Bahdanau and Luong attention sit on top of an RNN. They speed up convergence and improve quality, but the RNN still serialises computation: token $t$ must be processed before token $t+1$. On a GPU with thousands of cores, this is a tragedy.

Vaswani et al. asked the obvious question. *What if attention is the only thing we need?* If every token can attend to every other token directly, we get two enormous benefits:

1. **Parallelism.** All positions can be computed simultaneously in a matrix multiplication.
2. **Constant path length.** Any two tokens are exactly one operation apart, regardless of how far they sit in the sequence. No vanishing gradients across 100 timesteps.

The mechanism that makes this work is **self-attention**: instead of a decoder attending to encoder states, every token in a sequence attends to every other token in the *same* sequence.

### A motivating example

Consider: *"The animal didn't cross the street because **it** was too tired."*

To represent **it** correctly, the model must know **it** refers to the animal, not the street. With self-attention, the representation of **it** is built as a weighted sum of all other token representations in the sentence. A well-trained head will assign high weight to **animal**, low weight to **street**, and use that weighted sum to refine the meaning of **it**.

### The Query, Key, Value abstraction

For each token embedding $x_i$ we learn three linear projections:

$$q_i = W_Q\, x_i, \qquad k_i = W_K\, x_i, \qquad v_i = W_V\, x_i$$

The roles parallel a dictionary lookup:

- **Query** $q_i$: "What am I looking for?" — the question this token is asking.
- **Key** $k_i$: "What do I contain?" — used to decide whether this token is a relevant match.
- **Value** $v_i$: "What information do I provide if you do attend to me?"

When we compute attention for position $i$, we score $q_i$ against every key $k_j$ to decide *how much* to read from each value $v_j$.

### Scaled dot-product attention, step by step

![Scaled dot-product attention pipeline](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/attention-transformer/fig2_qkv_computation.png)

Stack all queries, keys, and values into matrices $Q, K, V \in \mathbb{R}^{n \times d_k}$. Then:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Walk through the four panels above:

1. **Compatibility scores** $Q K^\top$ produce an $n \times n$ matrix where entry $(i, j)$ is the dot product $q_i \cdot k_j$. Larger means "more similar".
2. **Scaling** divides every entry by $\sqrt{d_k}$.
3. **Softmax** along each row turns the scaled scores into a probability distribution that sums to 1. Row $i$ tells you how to mix the values.
4. **Weighted sum** with $V$ gives the new representation for each position.

### Why divide by $\sqrt{d_k}$?

This is the most-asked detail in interviews. Suppose the components of $q$ and $k$ are independent with mean 0 and variance 1. Then:

$$\text{Var}(q \cdot k) = \text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$

For $d_k = 64$, dot products end up with standard deviation 8. Pumping numbers of that scale into a softmax pushes the distribution toward a one-hot vector, where the gradient with respect to all but one position is essentially zero. Training stalls.

Dividing by $\sqrt{d_k}$ rescales the variance back to 1, keeping softmax in the regime where it has useful gradients. It is one line of code with an outsized effect on training stability.

---

## 5. Multi-head attention

![Multi-head attention architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/attention-transformer/fig3_multihead_attention.png)

A single attention operation produces one weighted view of the sequence. But language has many simultaneous structures: subject-verb agreement, coreference, dependency syntax, semantic similarity, positional adjacency. One head cannot serve all of them.

Multi-head attention runs $h$ attention operations in parallel, each with its own learned projections:

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

The outputs are concatenated and passed through a final linear layer:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O$$

In the original paper, $d_{\text{model}} = 512$ and $h = 8$, so each head has $d_k = d_v = 512 / 8 = 64$. The total compute per layer is the same as one big head, but the model can specialise: probing studies have shown different heads end up tracking different relationships, including syntactic dependencies and coreference.

### Causal masking

![Causal mask visualisation](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/attention-transformer/fig5_causal_mask.png)

In an autoregressive decoder we cannot let position $i$ attend to positions $j > i$ during training, or the model would simply copy the right answer. We enforce this with an **additive mask**:

$$\text{MaskedAttention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

with $M_{ij} = 0$ when $j \le i$ (allowed) and $M_{ij} = -\infty$ when $j > i$ (blocked). Adding $-\infty$ before softmax sends those entries to exactly zero in the resulting weights — the right column of the figure shows the upper triangle wiped out perfectly.

This single trick is what lets GPT-style models train on the entire sequence in one forward pass while behaving, at inference time, exactly as if they were generating left-to-right.

---

## 6. Positional encoding: putting order back in

![Sinusoidal positional encoding](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/attention-transformer/fig4_positional_encoding.png)

Self-attention is **permutation invariant**. Shuffle the input tokens and you shuffle the outputs identically — the attention weights themselves are unchanged. *"cat eats fish"* and *"fish eats cat"* would produce the same internal representations, which is clearly wrong.

We fix this by adding a **positional encoding** $\text{PE}(\text{pos}) \in \mathbb{R}^{d_{\text{model}}}$ to each token embedding before it enters the first layer.

### Sinusoidal encoding

The original Transformer uses a fixed (non-learned) sinusoidal scheme:

$$PE_{(\text{pos},\, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i / d_{\text{model}}}}\right), \qquad PE_{(\text{pos},\, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i / d_{\text{model}}}}\right)$$

The left panel above shows the encoding matrix as a heatmap. The right panel plots a few individual dimensions: low-index dimensions oscillate fast (fine-grained position), high-index dimensions oscillate very slowly (coarse position). The combination gives every position a unique fingerprint, and because $\sin$ and $\cos$ obey simple linear identities, the model can in principle learn relative offsets like "three positions ahead".

### Learned positional embeddings

An equally valid approach is to treat positions as a standard embedding table of shape $(\text{max\_len}, d_{\text{model}})$. BERT and GPT-2 both do this. It is simpler and trains slightly better in practice, with one drawback: you cannot extrapolate beyond the maximum length seen in training.

### Why add and not concatenate?

Adding preserves the full $d_{\text{model}}$ dimensions for both content and position; subsequent linear projections ($W_Q, W_K, W_V$) can disentangle them as needed. Concatenating would either inflate the dimensionality or steal capacity from the content embedding.

Modern alternatives — **RoPE** (rotary position embedding), **ALiBi** (linear bias on attention scores), and **NoPE** — all attack the length-extrapolation problem and are now standard in production LLMs. The original sinusoidal scheme remains a useful baseline.

---

## 7. The full Transformer architecture

![Transformer encoder and decoder blocks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/attention-transformer/fig6_transformer_block.png)

A Transformer is a stack of encoder layers feeding a stack of decoder layers. Both stacks share three structural ideas: **sublayers wrapped by residual connections, LayerNorm, and dropout**.

### Encoder layer

Each of the $N$ encoder layers contains two sublayers:

1. **Multi-head self-attention** over the source sequence.
2. **Position-wise feed-forward network** applied independently to every position.

$$\text{FFN}(x) = \max(0,\, x W_1 + b_1)\, W_2 + b_2$$

The FFN expands to $d_{\text{ff}} = 4 \cdot d_{\text{model}}$ (typically 2048) and projects back. It is where most parameters live and where token-level non-linear processing happens. Each sublayer is wrapped as:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

The residual connection lets gradients flow directly from any layer to any earlier layer, which is critical at depth 6, 12, 24, 96.

### Decoder layer

Each of the $N$ decoder layers contains three sublayers:

1. **Masked multi-head self-attention** over the target tokens generated so far (causal).
2. **Cross-attention** where queries come from the decoder and keys/values come from the final encoder output. This is the bridge between source and target.
3. **Feed-forward network**, identical in shape to the encoder's.

The figure above pictures both blocks side-by-side, with the cross-attention connection drawn explicitly.

### Putting it all together

```
src tokens -> embed + PE -> [encoder layer x N] -> encoder output
                                                        |
                                                  K, V  |
                                                        v
tgt tokens -> embed + PE -> [decoder layer x N] -> linear -> softmax -> probs
```

For a base Transformer ($N = 6$, $d_{\text{model}} = 512$, $h = 8$, $d_{\text{ff}} = 2048$) this is about 65M parameters. GPT-3 simply makes $N$, $d_{\text{model}}$, and $h$ much larger, drops the encoder, and trains on the internet.

---

## 8. PyTorch implementation from scratch

The following implementation is intentionally minimal so each piece maps to the equations above. Run it on CPU; this is for understanding, not training a real model.

### Scaled dot-product attention

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(query, key, value, mask=None):
    """Equation: softmax(QK^T / sqrt(d_k)) V.

    Shapes:
        query: (batch, heads, seq_q, d_k)
        key:   (batch, heads, seq_k, d_k)
        value: (batch, heads, seq_k, d_v)
        mask:  broadcastable to (batch, heads, seq_q, seq_k); 0 -> blocked
    Returns:
        output:  (batch, heads, seq_q, d_v)
        weights: (batch, heads, seq_q, seq_k)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, value), weights
```

### Multi-head attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # One big matrix per role, reshaped into heads later.
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x):
        b, t, _ = x.size()
        return x.view(b, t, self.num_heads, self.d_k).transpose(1, 2)

    def _merge(self, x):
        b, _, t, _ = x.size()
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def forward(self, q, k, v, mask=None):
        Q, K, V = self._split(self.W_q(q)), self._split(self.W_k(k)), self._split(self.W_v(v))
        out, weights = scaled_dot_product_attention(Q, K, V, mask)
        return self.W_o(self.dropout(self._merge(out))), weights
```

### Sinusoidal positional encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]
```

### Position-wise feed-forward

```python
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
```

### Encoder and decoder layers

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.drop2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        sa, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.drop1(sa))
        ca, _ = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.drop2(ca))
        x = self.norm3(x + self.drop3(self.ffn(x)))
        return x
```

### Full Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.proj = nn.Linear(d_model, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def make_pad_mask(seq, pad_idx=0):
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def make_causal_mask(size):
        return ~torch.triu(torch.ones(size, size, dtype=torch.bool), diagonal=1)

    def encode(self, src, src_mask=None):
        x = self.dropout(self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model)))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, src_mask=None, tgt_mask=None):
        x = self.dropout(self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model)))
        for layer in self.decoder:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.proj(dec_out)
```

### Smoke test

```python
torch.manual_seed(0)
model = Transformer(src_vocab=10000, tgt_vocab=10000, num_layers=2)
src = torch.randint(1, 10000, (2, 20))
tgt = torch.randint(1, 10000, (2, 25))

src_mask = Transformer.make_pad_mask(src)
tgt_mask = Transformer.make_causal_mask(tgt.size(1) - 1)

logits = model(src, tgt[:, :-1], src_mask, tgt_mask)
print(logits.shape)  # torch.Size([2, 24, 10000])
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

About 24M parameters at $N=2$. The full base Transformer at $N=6$ is roughly 65M.

---

## 9. Self-attention vs. RNN vs. CNN

![Receptive field comparison](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/nlp/attention-transformer/fig7_receptive_field.png)

Why did Transformers replace RNNs and CNNs for sequence modelling so completely? The answer is in three numbers per layer:

| Architecture        | Compute per layer    | Sequential ops | Max path length |
|---------------------|---------------------|----------------|-----------------|
| **Self-attention**  | $O(n^2 \cdot d)$    | $O(1)$         | $O(1)$          |
| **RNN (LSTM/GRU)**  | $O(n \cdot d^2)$    | $O(n)$         | $O(n)$          |
| **CNN (kernel $k$)**| $O(k \cdot n \cdot d^2)$ | $O(1)$    | $O(\log_k n)$   |

Self-attention pays $O(n^2)$ in compute but gives you constant path length and full parallelism. For typical sequence lengths ($n < 1000$) and modern hardware that loves matrix multiplies, this trade is overwhelmingly favourable. For very long sequences (tens of thousands of tokens), efficient variants like **FlashAttention**, **Longformer**, **Performer**, and **Mamba**-style state-space models become attractive.

---

## 10. Three flavours of Transformer in production

The original Transformer is encoder-decoder. Two widely-used variants drop one half:

| Family             | Architecture     | Pre-training objective         | Best at                                  | Examples                |
|--------------------|------------------|--------------------------------|------------------------------------------|-------------------------|
| **Encoder-only**   | encoder stack    | masked language modelling      | classification, NER, retrieval, QA       | BERT, RoBERTa, DeBERTa  |
| **Decoder-only**   | decoder stack    | next-token prediction (causal) | generation, dialogue, code, reasoning    | GPT, LLaMA, Claude      |
| **Encoder-decoder**| both             | span corruption / seq2seq      | translation, summarisation, structured tasks | T5, BART, mT5       |

Decoder-only models have largely won the LLM race because next-token prediction on raw web text scales beautifully and unifies almost every task as text generation.

### Quick recipes with HuggingFace

For real applications you almost never train from scratch. These three snippets show the flavour of each family.

```python
# Encoder-only: classification with BERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
inputs = tok("This movie was fantastic!", return_tensors="pt")
logits = model(**inputs).logits
print("positive" if logits.argmax() == 1 else "negative")
```

```python
# Decoder-only: text generation with GPT-2
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
inputs = tok("Once upon a time", return_tensors="pt")
out = model.generate(
    inputs.input_ids, max_new_tokens=40, top_p=0.95, temperature=0.8, do_sample=True
)
print(tok.decode(out[0], skip_special_tokens=True))
```

```python
# Encoder-decoder: translation with T5
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
inputs = tok(
    "translate English to French: The cat is sleeping on the mat.",
    return_tensors="pt",
)
out = model.generate(inputs.input_ids, max_new_tokens=40, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

---

## 11. Frequently asked questions

**Why do we need masking only in the decoder, not the encoder?**
The encoder sees the entire source sentence and is supposed to look at everything bidirectionally. The decoder generates one token at a time and would otherwise cheat by attending to future ground-truth tokens during training. Encoder-only models like BERT are bidirectional precisely because they have no causal mask.

**Where does the quadratic memory cost come from?**
The $n \times n$ attention score matrix, before any reduction. For $n = 4{,}000$ and $h = 16$ heads in float16, that single buffer is already over 500 MB per layer. FlashAttention works around this by never materialising the full matrix and instead streaming the softmax tile-by-tile.

**Why $d_{\text{ff}} = 4 \cdot d_{\text{model}}$?**
Empirical. The 4x ratio was a reasonable choice in 2017 and has stuck. Recent work (PaLM, LLaMA) sometimes uses different ratios or replaces ReLU with SwiGLU/GeGLU for marginal gains.

**Pre-LN or post-LN?**
The original paper used post-LN (LayerNorm *after* the residual). Modern implementations almost always use pre-LN (LayerNorm *before* the sublayer), which trains more stably at depth and reduces the need for elaborate learning-rate warmup.

**Are positional encodings still sinusoidal in modern LLMs?**
No. Most production LLMs now use **RoPE** (LLaMA, GPT-NeoX) or **ALiBi** (BLOOM) because they extrapolate to longer contexts and integrate naturally with multi-head attention via rotation or additive bias.

---

## 12. Key takeaways

- Vanilla Seq2Seq fails on long inputs because a single context vector is too small. **Attention** lets the decoder dynamically access every encoder state.
- **Self-attention** drops recurrence: every position sees every other position in $O(1)$ steps.
- **Scaled dot-product attention** is just $\text{softmax}(QK^\top / \sqrt{d_k}) V$. The $\sqrt{d_k}$ is the difference between training stability and gradient collapse.
- **Multi-head attention** runs many smaller attention operations in parallel, letting different heads specialise.
- **Positional encoding** restores the order information that pure attention discards.
- The **Transformer block** = (multi-head attention + FFN), each wrapped in residual + LayerNorm. Stack $N$ of them. That is all there is.
- **BERT, GPT, T5** are encoder-only, decoder-only, and encoder-decoder specialisations of exactly this template.

The next two articles dive into BERT and GPT in depth — once the architecture clicks, the rest is mostly clever pre-training objectives and scale.

---

## Series Navigation

| Part | Topic | Link |
|------|-------|------|
| 1 | Introduction and Text Preprocessing | [<-- Read](/en/nlp-introduction-and-preprocessing/) |
| 2 | Word Embeddings and Language Models | [<-- Read](/en/nlp-word-embeddings-lm/) |
| 3 | RNN and Sequence Modeling | [<-- Previous](/en/nlp-rnn-sequence-modeling/) |
| **4** | **Attention Mechanism and Transformer (this article)** | |
| 5 | BERT and Pretrained Models | [Read next -->](/en/nlp-bert-pretrained-models/) |
| 6 | GPT and Generative Models | [Read -->](/en/nlp-gpt-generative-models/) |
