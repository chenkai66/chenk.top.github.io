---
title: "Session-based Recommendation with Graph Neural Networks (SR-GNN)"
date: 2025-04-08 09:00:00
tags:
  - GNN
  - Recommender Systems
categories: Paper
lang: en
mathjax: true
description: "SR-GNN turns a click session into a directed weighted graph and runs a gated GNN to predict the next item. Covers session-graph construction, GGNN updates, attention-based session pooling, training, benchmarks, and the failure modes that decide whether you should reach for it."
disableNunjucks: true
---

A user clicks **A, B, C, B, D**. A sequence model reads this as five tokens and folds them into a hidden state. **SR-GNN** sees a *graph* in which the edge `B -> C` survives even after the user returns to `B`, the node `B` is reused (so its in/out neighbours both inform its embedding), and the geometry of the click stream is preserved as adjacency. That structural insight is why [SR-GNN (Wu et al., AAAI 2019)](https://arxiv.org/abs/1811.00855) outperforms purely sequential baselines such as GRU4Rec and NARM on standard session-based recommendation (SBR) benchmarks.

This note unpacks SR-GNN end to end: how the session graph is built, how the **gated GNN** (GGNN) propagates information over it, how a session vector is assembled from a *local* (last click) and *global* (attention) view, how scoring and training work, and where the model breaks. The aim is to leave you in a position to either drop SR-GNN into your stack or, more usefully, to know exactly when *not* to.

## What you will learn

- How the click stream is converted into a directed weighted session graph (and how the in/out adjacency rows feed the GGNN)
- The gated GNN update unpacked as a GRU cell over an aggregated message
- Session pooling: why **local + global** beats either alone, and how attention is anchored on the last click
- The training objective, the realistic hyperparameters, and what BPTT actually means for a graph this small
- Why session graphs beat RNN/GRU baselines on multi-step click patterns
- Concrete failure modes (short sessions, popularity collapse, cold-start items, large catalogs) and the standard fixes
- Reasonable variants (attention-weighted GGNN, time-gap edges, multi-task heads) and when they help

## Prerequisites

- Comfortable with message passing and basic GNN vocabulary (adjacency, propagation steps)
- Familiar with GRU / LSTM gates
- Working knowledge of recommendation metrics: Recall@K, MRR@K, sampled softmax

---

## 1. Background: what makes session-based recommendation different

In **session-based recommendation** there is no stable long-term user profile to lean on. We see only the current short sequence of clicks $s = [v_{s,1}, v_{s,2}, \dots, v_{s,n}]$ over an item catalog $V = \{v_1, \dots, v_{|V|}\}$, and we have to predict the next item $v_{s,n+1}$. The model outputs a score vector $\hat z \in \mathbb{R}^{|V|}$ over the catalog and the top-$K$ items are recommended.

Two properties make this regime awkward for classical CF and for plain RNNs:

- **Short context**: a typical session is 2--10 clicks. There is no signal beyond the session itself, so the model must extract *intent* from very little.
- **Repeated items and non-monotone intent**: users wander, double back, compare. The same item can show up multiple times in one session, and a "later" click is not necessarily a "better" preference signal than an earlier one.

Pure sequence models compress these clicks into a single hidden state and inevitably lose the relational structure between revisited items. SR-GNN's contribution is to keep that structure explicit -- as a graph -- and let message passing handle the rest.

## 2. Session graph construction

For each session $s$, SR-GNN builds a directed graph $G_s = (V_s, E_s)$ where $V_s$ are the unique items clicked in this session and $E_s$ contains a directed edge $u \to v$ for every observed transition. **Repeated nodes are deduplicated** (each item appears once as a node, no matter how many times it is clicked), but every transition is preserved as an edge. Edge weights are normalised by the source's out-degree:

$$
w_{u \to v} \;=\; \frac{\#(u \to v)}{\mathrm{outdeg}(u)} \, .
$$

Two adjacency matrices fall out of this construction: the **incoming** adjacency $A^{(\text{in})}$ (row $i$ tells node $i$ which other nodes feed it) and the **outgoing** adjacency $A^{(\text{out})}$ (row $i$ tells node $i$ which nodes it feeds). They are concatenated into a single $|V_s| \times 2|V_s|$ matrix $A_s$ that the GGNN uses for message passing.

![Session graph construction: click stream A,B,C,B,D becomes a directed weighted graph; transitions persist even after a revisit](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/session-based-recommendation-with-graph-neural-networks/fig1_session_graph.png)

Concretely, for the click stream `A, B, C, B, D`:

| edge        | count | source out-deg | weight |
| ----------- | ----- | -------------- | ------ |
| $A \to B$   | 1     | 1              | 1.00   |
| $B \to C$   | 1     | 2              | 0.50   |
| $C \to B$   | 1     | 1              | 1.00   |
| $B \to D$   | 1     | 2              | 0.50   |

A pure GRU over the same stream would compress the second visit to `B` *into* the hidden state and effectively forget the `B -> C` transition once it absorbs `C -> B`. The graph view loses none of this.

## 3. Gated GNN propagation

Each item $v$ in the session has a $d$-dimensional embedding $h_v$ pulled from a global **item table** $V \in \mathbb{R}^{|V|\times d}$. SR-GNN runs $T$ rounds of message passing over the session graph using a **Gated Graph Neural Network** (Li et al., 2016), which is essentially a GRU cell driven by an aggregated message rather than a sequential input.

For node $i$ at step $t$ the message is

$$
a_t^{(i)} \;=\; A_{s,i:}\, \big[h_1^{(t-1)}, \dots, h_n^{(t-1)}\big]^\top\, W_a \;+\; b,
$$

where $A_{s,i:}$ is row $i$ of the concatenated $[A^{(\text{in})}\,|\,A^{(\text{out})}]$ adjacency. The aggregated message then drives a GRU-style update:

$$
\begin{aligned}
z_t &\;=\; \sigma\!\big(W_z\, a_t + U_z\, h_{t-1}\big), \\
r_t &\;=\; \sigma\!\big(W_r\, a_t + U_r\, h_{t-1}\big), \\
\tilde h_t &\;=\; \tanh\!\big(W\, a_t + U\, (r_t \odot h_{t-1})\big), \\
h_t &\;=\; (1 - z_t)\, h_{t-1} \;+\; z_t\, \tilde h_t \, .
\end{aligned}
$$

The reset gate $r_t$ controls how much of the previous state contributes to the candidate; the update gate $z_t$ then blends old and new. The intuition is the same as in a GRU sequence model, but the "next input" is now the *graph* message $a_t$, not the next item in a list.

![SR-GNN end-to-end: session graph in, GGNN propagation, per-item embeddings, attention pooling, softmax over the catalog](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/session-based-recommendation-with-graph-neural-networks/fig2_architecture.png)

After $T$ steps each node carries a context-aware embedding $h_v$ that depends on its in/out neighbours, on its position in any cycles within the session, and on how often each transition fires.

![One step of the gated GNN: the in/out adjacency rows produce the message $a_t$, which then drives a GRU-style state update](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/session-based-recommendation-with-graph-neural-networks/fig3_ggnn_update.png)

A few practical notes on the propagation:

- **Number of steps $T$**: the original paper uses $T = 1$ on Yoochoose and $T = 1$ on Diginetica. Increasing $T$ rarely helps because session graphs are tiny (rarely more than 10 nodes) and signal already saturates.
- **Parameter sharing**: the GRU cell parameters $(W_z, U_z, W_r, U_r, W, U, W_a, b)$ are shared across all nodes and across all sessions -- the model is **transductive over the session catalog only at the embedding table level**, not at the cell level.
- **Repeated visits**: because deduplicated nodes appear exactly once in the graph, both visits to `B` share the same embedding throughout propagation. The model recovers ordering information later, in the pooling step.

## 4. Building the session representation

After propagation, SR-GNN turns the per-item embeddings $\{h_1, \dots, h_n\}$ into a single session vector $s_h$. A naive choice -- "use the last $h_n$" -- works surprisingly well on short sessions but throws away everything else the graph learned. The paper's design uses **two views**, fused linearly.

**Local intent.** The embedding of the last clicked item:

$$
s_l \;=\; h_n \, .
$$

This is the strongest single signal of the user's *current* mood in the session.

**Global context.** A soft-attention sum over all per-item embeddings, where the attention is *anchored on $h_n$*:

$$
\alpha_i \;=\; q^\top\, \sigma\!\big(W_1\, h_n \;+\; W_2\, h_i \;+\; c\big), \qquad
s_g \;=\; \sum_{i=1}^{n} \alpha_i\, h_i \, .
$$

Here $q \in \mathbb{R}^d$ is a learnable query and $W_1, W_2 \in \mathbb{R}^{d\times d}$ project both the last click and each item into a shared scoring space. Items that look "relevant to where the user just was" earn more weight; items that have been overshadowed earn less.

**Final session vector.** Concatenate and project:

$$
s_h \;=\; W_3\, [\,s_l \,;\, s_g\,], \qquad W_3 \in \mathbb{R}^{d \times 2d} \, .
$$

![Session pooling: per-item attention weights anchored on the last click; local + global fused into the session vector](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/session-based-recommendation-with-graph-neural-networks/fig4_attention_pooling.png)

The local + global split is the same idea you see in NARM and STAMP, but with one important difference: SR-GNN's per-item embeddings already encode *graph structure*, so the global term aggregates structurally aware vectors rather than raw item embeddings. This is most of the empirical win.

## 5. Scoring and training

Given $s_h$ and the item table $V \in \mathbb{R}^{|V|\times d}$, candidate scores are dot products:

$$
\hat z_i \;=\; s_h^\top\, v_i, \qquad
\hat y \;=\; \mathrm{softmax}(\hat z) \, .
$$

Training minimises cross-entropy against the one-hot ground-truth next item:

$$
\mathcal{L} \;=\; -\sum_{i=1}^{|V|} y_i \log \hat y_i \, .
$$

A few details that matter in practice:

- **BPTT, but for a tiny graph**: gradients flow through $T$ GGNN steps. Because $T$ is typically 1 and graphs have at most a dozen nodes, this is cheap -- nothing like sequence-model BPTT over hundreds of tokens.
- **Optimiser**: Adam with $\eta = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$. L2 weight decay $10^{-5}$ on all matrices.
- **Embedding size**: $d = 100$ is the standard. Going larger (256, 512) overfits Yoochoose 1/64 and Diginetica without lifting Recall@20.
- **Batching**: sessions vary in length, so the implementation pads each batch to the max session size and masks accordingly. The official repo at <https://github.com/CRIPAC-DIG/SR-GNN/tree/master> handles this carefully -- if you reimplement, copy that masking logic.
- **Sampled softmax for huge catalogs**: with $|V| > 10^5$ the full softmax becomes the bottleneck. Replace it with sampled softmax or a two-tower retrieval head; SR-GNN itself stays unchanged.

## 6. Why session graphs outperform sequential baselines

The pure-sequence formulation is $h_t = \mathrm{GRU}(h_{t-1}, v_t)$. It has three weaknesses that the graph view fixes by construction:

- **Lost transitions on revisits.** When a user clicks `A -> B -> C -> B`, the GRU's hidden state at the second `B` overwrites information about the `B -> C` step. The session graph keeps both edges $B \to C$ and $C \to B$ explicit; the GGNN sees both as part of node $B$'s neighbourhood.
- **Implicit relational learning.** A sequence model has to *learn* that "two clicks in different positions on the same item refer to the same item" through gradient signal alone. The session graph encodes that fact in the adjacency.
- **Single direction of information flow.** RNNs are left-to-right. The graph propagates in both directions through the in/out adjacency split, so `D` can pull information from `B` (its predecessor) without waiting for a backward pass.

Empirically these add up. On the standard SR-GNN evaluation -- Yoochoose 1/64, Yoochoose 1/4, Diginetica -- the model beats POP, Item-KNN, FPMC, GRU4Rec, NARM and STAMP on both Recall@20 and MRR@20:

![Benchmark comparison: SR-GNN vs prior session-based baselines on Yoochoose 1/64, Yoochoose 1/4 and Diginetica (Recall@20 and MRR@20)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/session-based-recommendation-with-graph-neural-networks/fig5_benchmark_perf.png)

The lifts are biggest on Diginetica, where sessions are longer and have more revisits -- exactly the regime where sequence models lose the most transition information.

## 7. Hyperparameters and training recipe

The paper's defaults are unusually well-tuned and tend to transfer to new SBR datasets with little change. Use them as a starting point:

| Hyperparameter            | Value             | Notes                                                                        |
| ------------------------- | ----------------- | ---------------------------------------------------------------------------- |
| Embedding dim $d$         | 100               | 64--128 is the sweet spot; 256+ overfits typical SBR data                    |
| GGNN propagation steps $T$| 1                 | 2 helps marginally on Diginetica, hurts on Yoochoose                         |
| Optimiser                 | Adam              | $\eta = 10^{-3}$, $(\beta_1, \beta_2) = (0.9, 0.999)$                        |
| LR schedule               | Decay by 0.1 / 3 ep | Apply after epoch 3; decay improves Recall@20 by ~0.5--1.0 pt              |
| Batch size                | 100               | 50--200 all work; not very sensitive                                         |
| L2 weight decay           | $10^{-5}$         | Apply to all $W_*$ matrices and the item table                               |
| Dropout                   | None on GGNN, 0.5 on item table during eval | Item-table dropout regularises the long tail              |
| Early stopping            | Patience 5 on Recall@20 | Most runs converge in 8--12 epochs                                       |

A subtle gotcha: the official preprocessing **filters sessions of length 1** and **filters items appearing fewer than 5 times**. If you reuse the published numbers without these filters your Recall@20 will look ~3--5 points worse, and you will spend a week debugging the model rather than the data.

## 8. Failure modes and how to fix them

SR-GNN is not a universal SBR solution. The four modes below show up reliably enough that they belong in any production checklist.

### 8.1 Popularity collapse

**Symptom.** Recall@20 looks fine, but the top-$K$ list is dominated by 5--10 globally popular items regardless of session. Diversity (intra-list distance, coverage@K) is low.

**Cause.** Cross-entropy with a global softmax is biased toward popular items: they contribute to most positive examples. The model learns "predict popular items" because it is the lowest-loss strategy on average.

**Fix.**

- **Popularity penalty** in the score: $\hat z_i \mathrel{-}= \lambda \log \mathrm{freq}(i)$. Tune $\lambda \in [0.1, 1.0]$ on a diversity-vs-recall trade-off.
- **Inverse-propensity-weighted softmax**: down-weight popular positives during training.
- **Negative sampling** with popularity-proportional sampling, which forces the model to discriminate against popular items it would otherwise default to.

### 8.2 Poor performance on very short sessions ($n \le 3$)

**Symptom.** The model is excellent on sessions of length 5+ but loses to a co-click baseline on length-2 and length-3 sessions.

**Cause.** A length-2 session graph has 2 nodes and 1 edge; there is essentially no graph structure for the GGNN to exploit, and pooling reduces to "use $h_n$".

**Fix.**

- **Hybrid serving**: route sessions of length $\le 3$ to an **item-KNN** or co-click model, and only use SR-GNN for length $\ge 4$. The blend usually wins on every length bucket.
- **Graph augmentation**: attach the last item to its top-$k$ co-clicked neighbours from the global click graph. This "borrows" structure when the session itself has none.
- **Pretrain item embeddings** on the global co-click graph (DeepWalk / node2vec) and initialise the SR-GNN item table from them. Short sessions then start with informative embeddings rather than random ones.

### 8.3 Overfitting on small datasets

**Symptom.** Training Recall@20 climbs steadily; validation Recall@20 plateaus by epoch 4 and starts to drop.

**Cause.** The item table $V \in \mathbb{R}^{|V|\times d}$ is by far the largest parameter block; on small datasets it memorises long-tail item identities.

**Fix.**

- Drop $d$ from 100 to 50.
- Add **dropout 0.5--0.7 on the item table** during training (fix the same dropout mask per session).
- L2 weight decay up to $10^{-4}$.
- Earlier early-stopping: patience 3 instead of 5.

### 8.4 Cold-start items

**Symptom.** Items that appear fewer than ~5 times in training are almost never recommended; their dot product with any $s_h$ stays small.

**Cause.** Their rows in the item table $V$ have near-zero gradient signal and stay close to initialisation.

**Fix.**

- Add **content features** (title text embedding, category, brand) as a side-channel and learn $v_i = V[i] + g(\text{content}_i)$. The cold-start row inherits a prior from $g$.
- Use **two-tower retrieval** for the candidate generation step and reserve SR-GNN for ranking on a smaller candidate set.

## 9. Variants and useful extensions

A few extensions are worth knowing because they recur in follow-up SBR papers and in production systems.

### 9.1 Attention-weighted message passing

Replace the fixed $A_s$ with attention weights computed per-edge:

$$
\alpha_{ij} \;=\; \mathrm{softmax}_j\!\big(\mathrm{LeakyReLU}(a^\top [W h_i \,||\, W h_j])\big),
\qquad a_t^{(i)} \;=\; \sum_{j \in \mathcal N(i)} \alpha_{ij}\, W h_j \, .
$$

This is essentially GAT inside the GGNN cell. Helps when transitions are not equally informative.

### 9.2 Time-gap edges

Sessions span seconds to minutes; a click that happens 2 seconds after the previous one is a stronger signal than one 5 minutes later. Encode the time gap $\Delta t_{u \to v}$ into the edge weight:

$$
w_{u \to v} \;=\; \exp\!\big(-\beta \cdot \Delta t_{u \to v}\big) \cdot \frac{\#(u \to v)}{\mathrm{outdeg}(u)} \, .
$$

A learnable $\beta$ usually settles around $0.05$--$0.2$ when $\Delta t$ is in seconds.

### 9.3 Multi-task heads

Add auxiliary losses on the same $s_h$:

- **Session length prediction** (regression on $\log n$).
- **Will-the-user-return** (binary classification within the next 24 h).
- **Category prediction** for the next click.

These regularise the session vector and tend to help when the next-click loss alone is noisy. Keep the auxiliary loss weights small ($0.05$--$0.2$) -- they are guides, not objectives.

## 10. When to use SR-GNN vs alternatives

| Scenario                                  | Recommendation                                                       |
| ----------------------------------------- | -------------------------------------------------------------------- |
| Long sessions with revisits ($n \ge 5$)   | SR-GNN -- this is its sweet spot                                     |
| Very short sessions ($n \le 3$)           | Item-KNN or co-click; SR-GNN has no graph to exploit                 |
| Heavy cold-start                          | Two-tower with content features; SR-GNN as a re-ranker only          |
| Real-time latency budget $< 5$ ms         | Cache per-item neighbour reps; consider a distilled MLP head         |
| Catalog $|V| > 10^6$                      | SR-GNN body + sampled softmax / two-tower retrieval                  |
| Data with strong long-term user history   | Look at sequential models with user embeddings (SASRec, BERT4Rec)    |

## Summary: SR-GNN in five points

- **Session as a graph.** Click streams become directed weighted graphs; revisits and cycles are preserved as adjacency rather than overwritten in a hidden state.
- **GGNN = GRU on graph messages.** A single message $a_t$ aggregated over the in/out adjacency drives reset, update and candidate gates. One propagation step is usually enough.
- **Local + global pooling.** The session vector fuses the last-click embedding (short-term intent) with an attention sum over all item embeddings (global context anchored on the last click).
- **Cross-entropy training, dot-product scoring.** The setup is standard; the win comes from what the embeddings encode, not from a fancier loss.
- **Sweet spot is medium-length sessions with revisits.** Outside that regime -- length $\le 3$, cold-start items, very large catalogs -- pair SR-GNN with the right complement (KNN, content tower, sampled softmax) instead of trying to fix it from inside.

The deeper takeaway is structural. Session-based recommendation is *not* a sequence problem dressed up; it is a graph problem with a sequential prior. Once you commit to the graph view, every later improvement in this line of work -- attention-weighted GNNs (GC-SAN), hyperbolic embeddings (HCGR), LLM-augmented session models (LLMGR) -- becomes much easier to read.
