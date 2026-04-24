---
title: "Graph Contextualized Self-Attention Network (GC-SAN) for Session-based Recommendation"
date: 2024-12-15 09:00:00
tags:
  - Attention
  - Recommender Systems
  - GNN
categories: Paper
lang: en
mathjax: true
description: "GC-SAN combines a session-graph GGNN (local transitions) with multi-layer self-attention (global dependencies) for session-based recommendation. Covers graph construction, message passing, attention fusion, and where the design wins or breaks."
disableNunjucks: true
---

In session-based recommendation you only see a short anonymous click sequence -- no user profile, no long history, no demographics. Every signal you have lives inside that single window. **GC-SAN** (IJCAI 2019) takes the strongest two ideas of the time -- SR-GNN's session graph and the Transformer's self-attention -- and stacks them: a *graph* view captures local transition patterns and loops, a *sequence* view captures long-range intent, and a tiny weighted sum decides how much of each to trust. The result is a clean "best of both worlds" baseline that is genuinely hard to beat at its parameter budget.

## What you will learn

- Why session recommendation is structurally harder than classical CF
- How a click sequence becomes a directed weighted graph
- The GGNN cell: in/out aggregation plus GRU gates
- Self-attention as a global encoder on top of graph-contextualised embeddings
- The fusion weight $w$ between last-click and global intent
- When GC-SAN is the right baseline and when it is not

## Prerequisites

- Graph neural network basics (message passing, GRU-style updates)
- Self-attention (queries, keys, values, scaled dot product)
- Recommendation evaluation metrics (Recall@K, MRR@K)

---

## 1. Problem setup and why this is hard

Let $V = \{v_1, v_2, \dots, v_{|V|}\}$ be the item universe and let a session be an ordered click sequence $s = (v_{s,1}, v_{s,2}, \dots, v_{s,n})$. The task is to predict the next item $v_{s,n+1}$, usually by ranking all candidates and reporting Recall@K and MRR@K.

What makes this harder than classical collaborative filtering:

- **No long-term profile.** You cannot lean on stable user embeddings or demographic features.
- **Short, noisy behaviour.** A session may contain exploratory clicks, mis-clicks, or back-and-forth navigation.
- **Long-range dependencies.** Early clicks often still matter -- click "camera" first, click "memory card" twenty steps later.
- **Repeated transitions.** Users bounce between a few related items; a strict sequence model can underuse this structure.

These four pressures pull in different directions. Sequence-only models (RNN/Transformer) handle order well but treat each step as a fresh token. Graph-only models (SR-GNN) capture loops and repeats but need many hops to reach distant clicks. GC-SAN's design pitch is exactly to combine the two without paying the cost of either alone.

## 2. Where GC-SAN sits among prior work

Before GC-SAN the standard baselines were:

- **Markov chains.** Strong local signal, no global understanding.
- **GRU4Rec.** Sequential RNN; captures order but struggles past a few steps.
- **NARM, STAMP.** Attention-based sequential models; better long range, but ignore the explicit transition graph that emerges within a session.
- **SR-GNN.** Build a per-session directed graph and run a gated GNN; rich local structure, but multi-hop intent needs many propagation steps and can oversmooth.

GC-SAN's contribution is **architectural rather than algorithmic**: keep SR-GNN's gated-GNN as the local encoder, then stack a Transformer-style self-attention block on top so global dependencies do not need to be reached by graph hops. Figure 1 shows the full pipeline.

![GC-SAN pipeline overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/gcsan/fig1_architecture.png)

The pipeline is strictly serial: clicks $\to$ graph build $\to$ GGNN propagation $\to$ self-attention stack $\to$ fuse last-click and global $\to$ score every item.

## 3. Session graph construction

For each session $s$ build a directed graph $G_s = (V_s, E_s)$:

- **Nodes** are the unique items that appear in the session.
- For each adjacent pair $(v_{s,i}, v_{s,i+1})$, add a directed edge $v_{s,i} \to v_{s,i+1}$.
- If the same transition repeats, accumulate its weight (or treat as a multi-edge and normalise later).

This step is where you trade richness for compactness. The same item appearing twice in the session collapses into one node; only the *transitions* between them carry repetition. Loops -- click A then B then A -- show up as cycles in the graph, which a pure sequence model only sees as "two separate occurrences of A".

Two adjacency matrices are then built and row-normalised:

$$
A^{out}_{ij} = \frac{w(v_i \to v_j)}{\sum_k w(v_i \to v_k)}, \quad
A^{in}_{ij}  = \frac{w(v_j \to v_i)}{\sum_k w(v_k \to v_i)}.
$$

Figure 2 walks through one example end-to-end.

![Session graph construction from clicks](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/gcsan/fig2_session_graph.png)

Note that the click sequence `v1 v2 v3 v2 v4` produces a 4-node graph with a `v2 <-> v3` loop and a fan-out from `v2` to both `v3` and `v4`. The pure sequence has length 5; the graph has 4 nodes and 4 distinct edges. That compactness is the whole point.

## 4. Local encoder: GGNN over the session graph

GC-SAN reuses the SR-GNN gated graph neural network cell. Each node carries a $d$-dim embedding $h_i$. One propagation step has three parts.

**(i) Aggregate in/out neighbours.** For node $v_i$ at step $t$:

$$
a_i^{(t)} \;=\; A^{in}_{i,:} \, H^{(t-1)} W^{in} \;+\; A^{out}_{i,:} \, H^{(t-1)} W^{out} \;+\; b,
$$

where $H^{(t-1)} = [h_1^{(t-1)}, \dots, h_{|V_s|}^{(t-1)}]^\top$ stacks all node embeddings for the session and $W^{in}, W^{out} \in \mathbb{R}^{d \times d}$ are learnable. The two terms separate "evidence flowing into me" from "evidence flowing out of me", which matters for directional transitions.

**(ii) GRU gates.** Combine the aggregated message with the previous state:

$$
\begin{aligned}
z_i^{(t)} &= \sigma(W_z a_i^{(t)} + U_z h_i^{(t-1)}), \\
r_i^{(t)} &= \sigma(W_r a_i^{(t)} + U_r h_i^{(t-1)}), \\
\tilde h_i^{(t)} &= \tanh\!\bigl(W_h a_i^{(t)} + U_h (r_i^{(t)} \odot h_i^{(t-1)})\bigr), \\
h_i^{(t)} &= (1 - z_i^{(t)}) \odot h_i^{(t-1)} \;+\; z_i^{(t)} \odot \tilde h_i^{(t)}.
\end{aligned}
$$

The update gate $z$ decides how much of the new graph signal to write in; the reset gate $r$ decides how much of the old state to forget when forming the candidate. Figure 3 shows this on a single node.

![GGNN message passing on one node](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/gcsan/fig3_ggnn_message_passing.png)

After $T$ propagation steps (the paper uses $T=1$, sometimes 2), each node embedding has absorbed its local neighbourhood. **Crucially**, propagation is performed on the per-session graph -- not a global item graph -- so the embeddings are session-conditional.

> **Practical note.** Because sessions can repeat items, the implementation maintains an *alias* mapping from each sequence position to its node index in the unique-item graph. After GGNN you "scatter" node states back to sequence positions before applying self-attention. This is what `seq_hidden = hidden[alias_inputs]` does in any standard implementation.

## 5. Global encoder: self-attention over the session

GGNN is strong locally, but reaching a distant click requires many hops, and stacking too many GGNN layers leads to oversmoothing -- node embeddings collapse toward each other. Self-attention sidesteps this entirely: every position can attend to every other position in one step.

Let $E^{(0)} \in \mathbb{R}^{n \times d}$ be the per-position representation after scattering GGNN node states back to the sequence. One self-attention layer computes:

$$
F = \mathrm{softmax}\!\left(\frac{(E W^Q)(E W^K)^\top}{\sqrt{d}}\right)(E W^V),
$$

followed by a position-wise feed-forward block with a residual connection:

$$
E^{(1)} = \mathrm{ReLU}(F W_1 + b_1) W_2 + b_2 + F.
$$

Stacking $k$ such blocks produces $E^{(k)}$, the **graph-contextualised** sequence representation. Figure 4 shows what attention typically learns and why GGNN alone struggles to reach the same connections.

![Self-attention captures long-range dependencies](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/gcsan/fig4_self_attention.png)

The heatmap is illustrative but the pattern is real: the first click ("camera") attends strongly to thematically related items deep in the session (memory card, battery, bag) -- a connection that would take a 4-hop GNN traversal to reach, by which point the signal has been smoothed.

## 6. Fusion: last-click vs global intent

Session recommendation almost always benefits from explicitly mixing two signals:

- **Current interest** $h_t$: the embedding of the last clicked item (often the strongest short-term predictor).
- **Global intent** $a_t$: the last-position output of the self-attention stack, which has integrated the whole session.

GC-SAN combines them with a single scalar weight:

$$
s_f \;=\; w \cdot a_t \;+\; (1 - w) \cdot h_t,
$$

then scores every candidate by dot product with the item embedding table and normalises:

$$
\hat y \;=\; \mathrm{softmax}\!\bigl(s_f \, V^\top\bigr).
$$

The weight $w$ is a hyperparameter (typical sweet spot $w \in [0.4, 0.6]$). Set $w = 0$ and you get last-click only, set $w = 1$ and you discard the strong short-term signal. Figure 5(b) shows the typical sweep -- forgiving in the middle, painful at the extremes.

## 7. Training and evaluation

Most session recommenders train with cross-entropy over the next-item softmax:

$$
\mathcal{L} \;=\; -\sum_{i=1}^{|V|} y_i \log \hat y_i \;+\; \lambda \|\Theta\|^2,
$$

where $y$ is the one-hot label and $\Theta$ are all learnable parameters. When $|V|$ is large, BPR with negative sampling is a common alternative:

$$
\mathcal{L}_{\text{BPR}} \;=\; -\sum_{(u, i, j)} \log \sigma(\hat r_{ui} - \hat r_{uj}) + \lambda \|\Theta\|^2,
$$

with $i$ a positive (clicked) item and $j$ sampled negatives.

Standard benchmarks are **Yoochoose1/64** and **Diginetica**, reported with **Recall@20** and **MRR@20**. Figure 5 summarises the gap pattern reported in the paper.

![Performance vs SR-GNN and other baselines](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/gcsan/fig5_perf_vs_baselines.png)

Two things to read off the chart:

1. The **GC-SAN over SR-GNN** gap is consistent but not enormous (roughly +1 to +2 points on Recall and MRR). The marginal value of self-attention on top of a graph encoder is real but bounded.
2. The fusion-weight curve is **flat in the middle and steep at the edges**, which is exactly what you want from a hyperparameter: easy to tune, hard to break.

## 8. Implementation notes (what matters in practice)

**Alias mapping and batching.** Sessions repeat items, so you map each sequence position to a node index in the unique-item graph. Batching multiple session graphs together is non-trivial: either build a block-diagonal adjacency or use a library that supports batched graph operations (PyG, RecBole-GNN).

**Complexity.**
- GGNN: $\mathcal{O}(T \cdot |E_s| \cdot d)$ per session, where $T$ is propagation steps and $|E_s|$ is the number of edges in the session graph.
- Self-attention: $\mathcal{O}(n^2 d + n d^2)$ per session, quadratic in session length $n$. In session data $n$ is usually short (often $< 50$), so the quadratic cost is fine.

**Hyperparameters that change behaviour.**
- **Propagation steps $T$.** Too small misses multi-hop transitions; too large oversmooths. $T = 1$ is the paper default.
- **Self-attention layers/heads.** More layers add capacity but overfit on small datasets like Diginetica.
- **Fusion weight $w$.** Controls global-vs-local emphasis. Sweep on validation; expect a flat optimum near $0.5$.

**Padding and masking.** Self-attention must mask padded positions, otherwise gradient mass leaks into nonsense tokens. This is a frequent source of silent regressions when porting code.

## 9. Reference implementation sketch

A minimal RecBole-style implementation. The GGNN cell is reused from SR-GNN; the rest of the wiring is GC-SAN's contribution.

```python
import torch
from torch import nn
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss
from recbole.model.abstract_recommender import SequentialRecommender
from recbole_gnn.model.layers import SRGNNCell


class GCSAN(SequentialRecommender):
    """GGNN local encoder + self-attention global encoder + last/global fusion."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # Hyperparameters
        self.n_layers = config["n_layers"]                    # SA depth
        self.n_heads = config["n_heads"]                      # attention heads
        self.hidden_size = config["hidden_size"]              # d
        self.inner_size = config["inner_size"]                # FFN dim
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.step = config["step"]                            # GGNN propagation steps
        self.weight = config["weight"]                        # fusion weight w

        # Layers
        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size,
                                           padding_idx=0)
        self.gnncell = SRGNNCell(self.hidden_size)            # SR-GNN gated GGNN
        self.self_attention = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
        )
        self.loss_fct = BPRLoss()

    def forward(self, x, edge_index, alias_inputs, item_seq_len):
        # 1. Embed unique items in the session graph.
        hidden = self.item_embedding(x)

        # 2. GGNN propagation (T steps over per-session in/out adjacency).
        for _ in range(self.step):
            hidden = self.gnncell(hidden, edge_index)

        # 3. Scatter node states back to sequence positions.
        seq_hidden = hidden[alias_inputs]
        ht = self.gather_indexes(seq_hidden, item_seq_len - 1)   # last-click h_t

        # 4. Self-attention stack on the graph-contextualised sequence.
        outputs = self.self_attention(seq_hidden, output_all_encoded_layers=True)
        at = self.gather_indexes(outputs[-1], item_seq_len - 1)  # global a_t

        # 5. Linear fusion s_f = w * a_t + (1 - w) * h_t.
        return self.weight * at + (1 - self.weight) * ht

    def calculate_loss(self, interaction):
        seq_output = self.forward(
            interaction["x"],
            interaction["edge_index"],
            interaction["alias_inputs"],
            interaction[self.ITEM_SEQ_LEN],
        )
        pos = self.item_embedding(interaction[self.POS_ITEM_ID])
        neg = self.item_embedding(interaction[self.NEG_ITEM_ID])
        return self.loss_fct(
            torch.sum(seq_output * pos, dim=-1),
            torch.sum(seq_output * neg, dim=-1),
        )
```

A few things worth noticing about this reference:

- The **GGNN cell is imported, not reimplemented** -- GC-SAN is a wiring story.
- `gather_indexes` extracts the embedding at position `item_seq_len - 1`, which is the actual last click (not the padded tail).
- The fusion weight `self.weight` is a fixed scalar from config. A natural extension is to make it input-dependent (a small gating MLP on $h_t \oplus a_t$), but the paper does not explore this.
- Loss can be swapped between BPR and full-softmax cross-entropy depending on $|V|$.

## 10. When GC-SAN is a good choice (and when it is not)

**Good fit:**

- Sessions have meaningful transition structure (loops, repeats, related-item bouncing).
- Sessions are short to moderate length, so the $\mathcal{O}(n^2)$ attention cost is fine.
- You want a strong baseline that combines graph and sequence signals without exotic infrastructure.

**Limitations and risks:**

- Attention cost grows quadratically with session length. For very long sessions, switch to a linear attention variant or chunk the session.
- Graph construction choices (edge weighting, normalisation) can quietly change results. Validate on a small held-out slice when changing them.
- If item metadata is critical (text, image, category hierarchy), pure-ID GC-SAN underuses your features. Consider concatenating side-information embeddings before the GGNN, or move to a content-aware variant.
- The improvement over SR-GNN is real but modest. If you cannot afford a self-attention stack, SR-GNN alone is a respectable baseline.

## 11. Practical takeaway

GC-SAN reads as a sober "do both" recipe rather than a clever new mechanism:

- **GGNN** captures local transition patterns and repeats efficiently -- but cannot reach far without oversmoothing.
- **Self-attention** captures long-range dependencies in one step -- but treats the input as a flat sequence and ignores graph structure.
- A single scalar **fusion weight** $w$ between last-click and global intent ties them together, with a forgiving sweet spot.

For session-based recommendation in 2024, GC-SAN remains a clean baseline to beat: it tells you whether your fancy new model actually exploits session structure better than a graph encoder plus a Transformer block. If it does not, you have learned something useful before spending more compute.

## References

- Xu et al., "Graph Contextualized Self-Attention Network for Session-based Recommendation," IJCAI 2019. [Paper PDF](https://www.ijcai.org/proceedings/2019/0547.pdf)
- Wu et al., "Session-based Recommendation with Graph Neural Networks (SR-GNN)," AAAI 2019.
- Hidasi et al., "Session-based Recommendations with Recurrent Neural Networks (GRU4Rec)," ICLR 2016.
- Li et al., "Neural Attentive Session-based Recommendation (NARM)," CIKM 2017.
- Liu et al., "STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation," KDD 2018.
