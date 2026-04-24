---
title: "paper2repo: GitHub Repository Recommendation for Academic Papers"
date: 2024-11-29 09:00:00
tags:
  - Recommender Systems
  - GNN
categories: Paper
lang: en
mathjax: true
description: "paper2repo aligns academic papers with GitHub repositories in a shared embedding space using a constrained GCN. Covers the joint heterogeneous graph, the WARP ranking loss, the cosine alignment constraint, and the full inference path."
disableNunjucks: true
---

You read a paper, want the code, and the "code available at" link is dead, missing, or points to a stub. Search engines fall back to keyword matching over the README, which works for popular repos with descriptive names and dies on everything else. paper2repo (WWW 2020) frames this as a cross-platform recommendation problem: learn one embedding space in which a paper abstract and a GitHub repository are directly comparable by dot product, then rank.

The trick is not the text encoder - any reasonable CNN or transformer would do. The trick is that papers and repositories sit on two different graphs (citations on the paper side, co-star and tag overlap on the repo side), and naively running a GCN on each yields embeddings in two unrelated spaces. paper2repo bridges them with a small set of paper-repo pairs that we already know match, and a constraint that pulls those bridged embeddings together. Everything else in the model exists to make that constraint trainable.

## What you will learn

- How the joint heterogeneous graph is assembled from three asymmetric signals (citations, co-stars, tag overlap)
- Why two GCNs that never see each other's data still need a shared metric, and how the cosine constraint enforces one
- The WARP ranking loss and why it pairs naturally with normalized embeddings
- The Lagrangian-to-multiplicative trick that removes the constraint hyperparameter
- How the model is queried at inference time and what the reported HR@10 / MAP@10 / MRR@10 numbers actually mean

## Prerequisites

- GCN basics: layer-wise propagation $H^{(l+1)} = \sigma(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}H^{(l)}W^{(l)})$
- Standard recommendation metrics: HR@K, MAP@K, MRR@K
- Cosine similarity, hinge loss, and the idea behind margin-based ranking losses

## System overview

![paper2repo system architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/paper2repo-github-repository-recommendation/fig1_system_architecture.png)

Read Figure 1 left-to-right and top-to-bottom. There are two parallel towers. The left tower turns paper abstracts into vectors, runs a GCN over the citation graph, and emits paper embeddings $h^p$. The right tower does the same with repository descriptions and tags, runs a GCN over the repository association graph, and emits $h^r$. The towers never share weights, but they are tied at training time by two cross-tower forces: the cosine alignment constraint on bridged pairs, and the WARP ranking loss that uses paper-repo pairs as positives. At inference time the towers run independently; ranking is just a dot product against a precomputed repo index.

## The joint heterogeneous graph

The first non-trivial design choice is what graph you actually feed into each tower. Papers have a clean signal - citations - that is directional, sparse, and semantic. Repositories have nothing comparable. paper2repo manufactures a repo graph from two implicit signals.

![Heterogeneous graph: papers, repos, users, bridged pairs](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/paper2repo-github-repository-recommendation/fig2_heterogeneous_graph.png)

**Paper citation graph.** Nodes are papers; edges are citations, treated as undirected. Node features come from a CNN over the abstract.

**Repo association graph.** Nodes are repositories; edges come from two sources:

- *Co-star edges.* Two repos starred by the same user are connected. This is a noisy but cheap "people who liked X also liked Y" signal. Users do not appear as nodes in the repo graph itself - they exist only to induce co-star edges (the dotted amber lines in Figure 2).
- *Tag-overlap edges.* Two repos that share a tag whose TF-IDF score is above a threshold (the paper uses 0.3) are connected. This filters out generic tags like "python" and keeps the topical ones.

Node features on the repo side fuse the CNN-encoded description with an averaged tag embedding, projected to the same dimension.

**Bridged pairs.** A small set of paper-repo pairs is already labeled - papers that explicitly cite their own GitHub URL. These are the only supervision the alignment ever sees. Of 7,571 repos in the dataset, 2,107 are bridged. Everything else has to be reached transitively through the two graphs.

## Text encoders

Both towers use the same CNN-over-words recipe.

1. **Tokenize** the description, abstract, or tag list to a sequence $\{x_1, \dots, x_n\}$, then map each token to a $d$-dimensional pretrained vector (GloVe in the paper).
2. **Convolve** with multiple window sizes $h \in \{2, 3, 4\}$ and $k$ filters per width:
   $$c_i = \sigma(W \cdot x_{i:i+h-1} + b)$$
3. **Max-over-time pool** each filter to a scalar, concatenate to a fixed-size feature vector.
4. **Tags** are unordered, so average their word vectors and project through a fully-connected layer to the same dimension as the description features.
5. **Fuse** the description and tag features (sum or concatenate) to obtain the repository representation.

Paper abstracts go through the same encoder with the tag branch disabled. Both encoders' outputs become the input node features $H^{(0)}$ to their respective GCNs.

## Constrained GCN

A standard GCN layer is
$$H^{(l+1)} = \sigma\!\left(\tilde D^{-1/2}\tilde A\tilde D^{-1/2}\,H^{(l)}\,W^{(l)}\right)$$
where $\tilde A = A + I$ adds self-loops and $\tilde D$ is its degree matrix. Run this $L$ times on each graph and you get paper embeddings $h^p$ and repo embeddings $h^r$. They are useful for in-graph tasks (citation prediction on the paper side, co-star prediction on the repo side), but they live in two unrelated $d$-dimensional spaces - dot products across towers are meaningless.

The "constrained" in constrained GCN is what fixes that. For every bridged pair $(i, i)$ we require
$$h^{p}_{i}\!\cdot h^{r}_{i} \;\geq\; 1 - \delta, \qquad \delta \approx 10^{-3}.$$
With both embeddings $\ell_2$-normalized, the dot product is the cosine similarity, and the constraint says: bridged paper-repo pairs should be near-identical in direction. Since the two towers share no parameters, the only way to satisfy this is for the GCNs and encoders to learn aligned axes.

## Two training forces

![WARP ranking + cosine alignment constraint](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/paper2repo-github-repository-recommendation/fig3_embedding_objectives.png)

Figure 3(a) shows the ranking force. Embeddings live on the unit sphere. For a paper-positive-repo pair $(p, r^+)$ the loss pulls $h^{r^+}$ toward $h^p$ and pushes any negative $r^-$ that lands within margin $\gamma$ of the positive away. Figure 3(b) shows the alignment force. Without the cross-tower constraint the two GCNs would produce ellipsoidal "paper" and "repo" clouds floating in unrelated regions of $\mathbb{R}^d$. The constraint ties bridged pairs together, dragging the whole repo cloud into the paper cloud's frame.

### WARP ranking loss

WARP (Weighted Approximate-Rank Pairwise) is a margin-based ranking loss with a rank-aware weight. For a positive pair $(p, r^+)$ and a sampled negative $r^-$,
$$\ell(p, r^+, r^-) = L\!\left(\mathrm{rank}(p, r^+)\right) \cdot \big[\gamma - h^p\!\cdot h^{r^+} + h^p\!\cdot h^{r^-}\big]_+$$
where $[\cdot]_+ = \max(0, \cdot)$ is the hinge and $L(k) = \sum_{j=1}^{k} 1/j$ is a non-decreasing weight that grows when the positive is ranked far down. The rank is estimated by sampling negatives until one violates the margin and counting how many were tried.

Two practical points. First, WARP needs many negatives per positive - the original paper uses sampling-without-replacement until a margin violator is found. Second, on normalized embeddings the dot product is cosine in $[-1, 1]$, so the margin $\gamma$ is naturally interpretable (typical values are around 0.1).

### Removing the Lagrangian hyperparameter

The full objective is the WARP loss plus the alignment constraint. A textbook Lagrangian formulation would give
$$\min \sum_{(p,r^+,r^-)} \ell(p, r^+, r^-) \;+\; \lambda \cdot C_e$$
where $C_e$ is the mean alignment error
$$C_e = \frac{1}{|B|}\sum_{i \in B} \big[(1 - \delta) - h^{p}_{i}\!\cdot h^{r}_{i}\big]_+,$$
$|B|$ is the number of bridged pairs, and $\lambda$ trades off the two terms. The problem is that as training progresses the magnitude of both terms drifts, so any fixed $\lambda$ ends up either drowning the ranking loss or letting the constraint go slack. paper2repo replaces the additive Lagrangian with a multiplicative one:
$$\mathcal{L} = \left(\sum_{(p,r^+,r^-)} \ell(p, r^+, r^-)\right) \cdot (1 + C_e).$$
Because $h^p$ and $h^r$ are normalized, $C_e \in [0, 2]$ is bounded and on the same scale as a multiplicative factor. When the constraint is satisfied ($C_e \to 0$) the multiplier collapses to 1 and the loss is pure WARP. When the constraint is violated the entire ranking loss is amplified, which forces the optimizer to fix alignment first. There is no $\lambda$ to tune.

## Training procedure

**Positives.** Each bridged paper-repo pair is a positive. To grow the positive set, the paper also adds repo pairs $(r, r')$ that are co-starred frequently by the same users - if many users star both, treat $r'$ as a "related" positive of $r$. This expansion is symmetric and feeds into the repo-side ranking loss.

**Negatives.** Sample uniformly from the full repository pool. WARP samples until a margin violation is found, so the effective negative count adapts to how hard the positive currently is.

**Optimizer.** Adam over the union of CNN encoder weights, GCN weights, and the projection layers. The GCN is shallow (2 layers in the paper) - depth tends to over-smooth on these graphs.

**Outputs.** Aligned paper and repo embeddings, plus a precomputed repo index for fast inference.

## Inference

![Recommendation flow at query time](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/paper2repo-github-repository-recommendation/fig4_recommendation_flow.png)

At query time the paper-side tower runs once on the new abstract: CNN encoder, then a GCN propagation step using the citation neighborhood of the new node (or just the encoder output if it has no citations yet). The result is a single vector $h^p$. Ranking is then a dense matrix-vector product against the precomputed repo embedding matrix; top-K is an argpartition. There is no reranker, no second-stage filter - a deliberate choice that keeps the system serving-friendly and makes ablations clean.

The illustrative shortlist in Figure 4 shows the typical pattern: the top of the list is dominated by repos that are both topically close (high text similarity) and well-connected in the association graph (high GCN smoothness). The two signals reinforce each other when the paper has a clear topic and a popular reference implementation; they disagree on niche papers, where the repo association graph dominates and the model degrades to "what does the community usually star alongside this kind of work."

## Experiments

**Data.** 32,029 papers from Microsoft Academic (top venues, 2010-2018) and 7,571 GitHub repositories, of which 2,107 are bridged. The paper splits bridged pairs into train / validation / test for the alignment supervision and uses the rest of the graph for unsupervised structure.

**Metrics.** HR@K (does any relevant repo appear in the top-K), MAP@K (precision averaged over ranks of relevant repos), MRR@K (reciprocal rank of the first relevant repo). All evaluated at K = 10.

**Baselines.** Seven cross-domain or graph-aware recommenders: BPR, MF, LINE, NCF, CDL, KGCN, NSCR. None of them have an explicit cross-tower alignment objective.

![paper2repo vs seven baselines on HR@10, MAP@10, MRR@10](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/standalone/paper2repo-github-repository-recommendation/fig5_evaluation_results.png)

The pattern in Figure 5 is what you would expect if the alignment constraint is doing real work. Methods that ignore graph structure (BPR, MF) sit at the bottom. Methods that use one side's structure but not the other (LINE, NCF) move up. Methods that use both sides plus an external knowledge graph (CDL, KGCN, NSCR) get within striking distance. paper2repo opens a clear gap, and the gap is largest on HR@10 - the metric that rewards "the right repo is somewhere in the shortlist" rather than "the right repo is at the very top." That is consistent with the alignment constraint pulling bridged repos into the right region of the embedding space without necessarily resolving fine-grained ranking inside that region.

## Limitations

**Cold-start repos.** A new repo with no stars and one or two tags has almost no edges in the association graph. The GCN smooths it toward isolated text features, which is roughly what TF-IDF would give you. The constraint cannot help unless someone hand-labels a bridged pair for it.

**Bridged-pair scarcity.** The whole alignment depends on a thin layer of supervised pairs. In domains where paper-to-code linkage is rare (e.g., older papers, applied ML in industry) there is simply not enough supervision to align the two towers.

**GCN cost on the full graph.** Two-layer GCNs are tractable at this scale (~40K nodes total) but do not obviously extend to the millions of papers and tens of millions of repos that exist in the wild. Sampling-based GNNs (GraphSAGE, ClusterGCN) would be the natural next step.

**Static snapshot.** The graph is treated as static. In reality citations appear monthly and stars appear daily; a temporal GNN or a streaming index would be a better match for the deployed setting.

## Conclusion

The interesting idea in paper2repo is not the GCN, the WARP loss, or the CNN encoder - all of which were standard by 2020. It is the *constrained* GCN: the recognition that two independent towers can be tied together by a small set of bridged pairs and a single multiplicative term in the loss. That same recipe transfers cleanly to other cross-platform problems where you have abundant structure on each side and a small bridge between them: paper-to-dataset, paper-to-author, product-to-review, query-to-document with click logs as the bridge. The text encoders and graph types change; the constraint stays.

Three concrete extensions would matter most. Add author / venue / institution nodes to the paper side and contributor / organization nodes to the repo side, turning each tower's graph into a richer heterogeneous one. Replace the full-batch GCN with a sampled GNN to scale past the 40K-node regime. And replace the static bridged set with a self-training loop that mines new bridged pairs from high-confidence top-1 predictions.

## References

[1] Shao, H., Sun, D., Wu, J., Zhang, Z., Zhang, A., Yao, S., Liu, S., Wang, T., Zhang, C., & Abdelzaher, T. (2020). paper2repo: GitHub Repository Recommendation for Academic Papers. *Proceedings of The Web Conference 2020*, 580-590.

[2] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*.

[3] Weston, J., Bengio, S., & Usunier, N. (2011). WSABIE: Scaling Up to Large Vocabulary Image Annotation. *Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI)*, 2764-2770.

[4] Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*.
