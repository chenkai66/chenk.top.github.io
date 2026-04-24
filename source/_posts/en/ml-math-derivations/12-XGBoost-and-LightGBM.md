---
title: "Machine Learning Mathematical Derivations (12): XGBoost and LightGBM"
date: 2026-02-28 09:00:00
categories:
  - Machine Learning
tags:
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - Regularization
  - Histogram Algorithm
  - Mathematical Derivation
  - Machine Learning
series:
  name: "ML Mathematical Derivations"
  order: 12
  total: 20
lang: en
mathjax: true
description: "Derive XGBoost's second-order Taylor expansion, regularised objective and split-gain formula, then explore LightGBM's histogram algorithm, GOSS sampling and EFB bundling for industrial-scale gradient boosting."
disableNunjucks: true
series_order: 12
---

XGBoost and LightGBM are the two libraries that quietly win most tabular-data battles --- on Kaggle leaderboards, in fraud-detection pipelines, in ad ranking, in churn models. They share the same backbone (gradient-boosted trees, Part 11) but make very different engineering bets:

- **XGBoost** sharpens the *math*: it brings the second derivative of the loss into the objective, regularises the tree itself, and turns split selection into a closed-form score.
- **LightGBM** sharpens the *systems*: it bins features into a small histogram, grows trees leaf-by-leaf, throws away uninformative samples (GOSS) and bundles mutually exclusive sparse features (EFB).

The result is two tools that look interchangeable from the API but behave very differently when $N$ or $d$ becomes large. This post derives every formula behind those choices so you can read a tuning guide and know *why* each knob exists.

## What you will learn

- How XGBoost's second-order Taylor expansion produces a closed-form optimal leaf weight and a structure score for any tree.
- Why the split-gain formula carries a built-in pruning penalty $\gamma$.
- How LightGBM's histogram algorithm cuts the per-feature split cost from $O(N)$ to $O(K)$ and unlocks the histogram-subtraction trick.
- The exact statistical claim of GOSS and the constructive procedure of EFB.
- When level-wise growth dominates leaf-wise growth, and the opposite.

## Prerequisites

- GBDT fundamentals (Part 11 of this series).
- First- and second-order Taylor expansion.
- Decision-tree splitting via impurity / gain.

---

## Boosting in one picture

Before any formulas, look at what gradient boosting actually does. We start from a constant prediction (the mean of $y$) and ask each new tree to model the *current residual*. Early trees absorb the gross structure; later trees clean up local mistakes.

![Gradient boosting as iterative residual fitting](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/12-XGBoost-and-LightGBM/fig1_boosting_iterations.png)

After one tree the fit is a coarse step function and residuals are huge; after a hundred small-step trees ($\eta = 0.1$) the ensemble has captured the underlying $\sin(1.5x) + 0.35x$ trend and the residuals are essentially noise. XGBoost and LightGBM differ only in *how* they fit each tree --- the iterative skeleton above is identical.

---

## XGBoost: extreme gradient boosting

### Regularised objective

Plain GBDT minimises empirical loss only. XGBoost adds a tree-complexity term so that the optimiser knows what a "good" tree looks like *as a function*:

$$
\mathcal{L}^{(t)} \;=\; \sum_{i=1}^N L\bigl(y_i,\; \hat y_i^{(t-1)} + f_t(\mathbf{x}_i)\bigr) \;+\; \Omega(f_t),
$$

where $\hat y_i^{(t-1)}$ is the prediction from the first $t-1$ rounds, $f_t$ is the new tree to be fitted, and

$$
\Omega(f_t) \;=\; \gamma\, T \;+\; \tfrac{1}{2}\lambda \sum_{j=1}^T w_j^2.
$$

- $T$ is the number of leaves; the $\gamma T$ term is a **per-leaf cost** that acts as a soft pruning threshold.
- $w_j$ is the prediction stored at leaf $j$; the $\tfrac{1}{2}\lambda \sum w_j^2$ term is **L2 shrinkage on leaf weights**.

### Second-order Taylor expansion

Expand the per-sample loss around the current prediction $\hat y_i^{(t-1)}$:

$$
L\bigl(y_i,\; \hat y_i^{(t-1)} + f_t(\mathbf{x}_i)\bigr)
\;\approx\; L\bigl(y_i,\; \hat y_i^{(t-1)}\bigr) + g_i\, f_t(\mathbf{x}_i) + \tfrac{1}{2}h_i\, f_t(\mathbf{x}_i)^2,
$$

with the gradient and Hessian

$$
g_i \;=\; \partial L / \partial \hat y_i^{(t-1)}, \qquad h_i \;=\; \partial^2 L / \partial (\hat y_i^{(t-1)})^2.
$$

Dropping the constant zeroth-order term, the surrogate objective for round $t$ is purely quadratic in $f_t$:

$$
\widetilde{\mathcal{L}}^{(t)} \;=\; \sum_{i=1}^N \Bigl[g_i\, f_t(\mathbf{x}_i) + \tfrac{1}{2}h_i\, f_t(\mathbf{x}_i)^2\Bigr] + \Omega(f_t).
$$

This is the single design choice that separates XGBoost from classic GBDT: the second derivative $h_i$ enters the objective, giving Newton-style curvature information *for free*.

### Optimal leaf weight and structure score

Represent the tree by a leaf-assignment $q : \mathbb{R}^d \to \{1,\ldots,T\}$ and weights $\mathbf{w}$. Group samples by their leaf, $I_j = \{ i : q(\mathbf{x}_i) = j \}$, and let

$$
G_j = \sum_{i \in I_j} g_i, \qquad H_j = \sum_{i \in I_j} h_i.
$$

The objective decouples across leaves:

$$
\widetilde{\mathcal{L}}^{(t)} \;=\; \sum_{j=1}^T \Bigl[G_j\, w_j + \tfrac{1}{2}(H_j + \lambda)\, w_j^2\Bigr] + \gamma T.
$$

Each leaf is now an independent quadratic in $w_j$. Setting $\partial / \partial w_j = 0$:

$$
\boxed{\,w_j^{*} \;=\; -\frac{G_j}{H_j + \lambda}\,}
$$

Substituting back gives the **structure score** --- the best loss this tree shape can possibly achieve:

$$
\widetilde{\mathcal{L}}^{*}(q) \;=\; -\frac{1}{2}\sum_{j=1}^{T}\frac{G_j^2}{H_j + \lambda} \;+\; \gamma T.
$$

Lower is better. The score is a property of the *structure* $q$ alone --- weights are already optimised away. This converts tree learning into a structure search.

### Split gain

When considering whether to split a leaf into a left ($I_L$) and right ($I_R$) child, the change in structure score is

$$
\boxed{\;\text{Gain} \;=\; \frac{1}{2}\!\left[\frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\right] - \gamma\;}
$$

Two consequences worth pausing on:

1. The bracket is the **drop in the structure-score sum**; it is always $\ge 0$ (Cauchy--Schwarz / variance reduction) before subtracting $\gamma$.
2. The constant $\gamma$ enters as a **threshold**: if the structural improvement does not exceed $\gamma$, the split is rejected. There is no separate post-hoc pruning step --- pruning is built into the gain.

### Gradients for the standard losses

| Loss | $g_i$ | $h_i$ |
|---|---|---|
| Squared, $\tfrac{1}{2}(y-\hat y)^2$ | $\hat y_i - y_i$ | $1$ |
| Logistic, $-y\log p - (1-y)\log(1-p)$, $p = \sigma(\hat y)$ | $p_i - y_i$ | $p_i(1-p_i)$ |
| Softmax (class $c$) | $p_{ic} - \mathbb{1}[y_i = c]$ | $p_{ic}(1-p_{ic})$ |

For squared loss $h_i = 1$, so the second-order objective collapses to ordinary residual fitting --- XGBoost reduces to GBDT. For logistic and softmax the Hessian carries real information ($p(1-p)$ vanishes near saturation), and the Newton step pays off.

### Split-finding algorithms

Once the gain formula is in hand, the only remaining question is *how* to enumerate candidate splits. Pre-sorting every feature gives the exact algorithm and an $O(N)$ cost per feature per node; the *approximate* algorithm proposes a small number of candidate quantiles (weighted by $h_i$, since $h_i$ acts as the per-sample importance in the quadratic). The **sparsity-aware** variant learns a default direction at each split for missing values and only iterates over non-missing entries --- a quiet but huge win on sparse one-hot data.

![XGBoost exact pre-sorted scan vs LightGBM histogram scan](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/12-XGBoost-and-LightGBM/fig2_split_finding.png)

The left panel scans every distinct value ($N-1$ candidates). The right panel buckets the same data into $K = 32$ bins and only evaluates $K-1$ candidate splits --- yet picks essentially the same threshold and the same gain. That is the core LightGBM bet: most of the resolution in $N$ is wasted.

---

## LightGBM: efficient gradient boosting

### Histogram algorithm

LightGBM discretises every feature into $K$ integer bins (default 255) once, before training. For each leaf and each feature it then builds a histogram of $(G_b, H_b)$:

$$
G_b \;=\; \sum_{i:\, \text{bin}(x_{ij}) = b} g_i, \qquad H_b \;=\; \sum_{i:\, \text{bin}(x_{ij}) = b} h_i.
$$

Per-feature complexity collapses from $O(N)$ to $O(K)$ for both memory and split search:

| Aspect | Exact (XGBoost) | Histogram (LightGBM) |
|---|---|---|
| Memory per feature | $O(N)$ | $O(K)$ |
| Split search per feature | $O(N)$ | $O(K)$ |
| Cache-friendliness | poor (random access into pre-sort) | excellent (contiguous bins) |

There is also a **histogram subtraction** trick: once the histogram of one child is known, the sibling's histogram is just *parent minus child*. The cost of building both children is the cost of building one. In a deep tree this halves the work at every level.

### Leaf-wise vs level-wise growth

XGBoost's default growth is **level-wise** (BFS): split every node at the current depth before going deeper. LightGBM is **leaf-wise** (best-first): always split the single leaf with the highest gain.

![Leaf-wise vs level-wise tree growth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/12-XGBoost-and-LightGBM/fig5_growth.png)

For the same node budget, level-wise produces a balanced tree of depth $\log_2 T$ while leaf-wise can produce a long thin chain that sinks deep into one region of the input space. Leaf-wise usually achieves lower training loss with the same number of leaves --- but it overfits aggressively unless you cap `max_depth` and `min_data_in_leaf`.

### GOSS: gradient-based one-side sampling

The Hessian-weighted gradient $g_i$ tells you exactly how much sample $i$ contributes to the next split. Most samples in a well-fit ensemble have $|g_i| \approx 0$ (they are already well predicted). Throwing them away costs almost nothing --- *if* you compensate so that the $G$ and $H$ statistics remain unbiased.

GOSS does this in three steps with constants $a, b \in (0, 1)$:

1. Sort by $|g_i|$ descending; keep the top $a\cdot N$ samples (call this set $A$).
2. Randomly sample $b \cdot N$ samples from the remaining $(1-a) N$ (call this set $B$).
3. When computing $G_L, H_L, G_R, H_R$, weight every sample in $B$ by $\dfrac{1-a}{b}$ to undo the subsampling.

The effective gain estimator becomes

$$
\widetilde{\text{Gain}} \;=\; \frac{1}{2}\!\left[\frac{(G_L^A + \tfrac{1-a}{b}G_L^B)^2}{H_L^A + \tfrac{1-a}{b}H_L^B + \lambda} + \frac{(G_R^A + \tfrac{1-a}{b}G_R^B)^2}{H_R^A + \tfrac{1-a}{b}H_R^B + \lambda} - \cdots \right].
$$

The LightGBM paper proves that the variance introduced by this subsampling is $O(1/\sqrt{n_l})$ for a leaf with $n_l$ samples --- it shrinks faster than the natural sampling noise of the tree itself.

![GOSS keeps the informative tail and reweights the rest](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/12-XGBoost-and-LightGBM/fig3_goss.png)

The right panel is the punchline: with $a = 0.20$ and $b = 0.10$, only **30% of samples** are touched per iteration, yet the reweighted total $G$ matches the full-data $G$ in expectation.

### EFB: exclusive feature bundling

High-dimensional sparse features (one-hot category dummies, bag-of-words, click flags) rarely fire together --- if a row has `country = JP` it cannot also have `country = US`. EFB exploits this **mutual exclusivity** to compress the feature axis.

Concretely, EFB constructs a *conflict graph*: features are nodes, and an edge between $f_i$ and $f_j$ is weighted by the number of rows where they are *both* non-zero. Bundling features with no edges between them is a graph-colouring problem, solved greedily: visit features in descending degree order and place each one in the first existing bundle that contains no conflicting neighbour.

Once a bundle $\{f_{i_1}, f_{i_2}, \ldots\}$ is chosen, its members are merged into one column $\tilde f$ via integer offsets so their bins remain disjoint:

$$
\tilde f_n \;=\; \begin{cases} \text{bin}(x_{n, i_k}) + o_k & \text{if } x_{n, i_k} \neq 0 \\ 0 & \text{otherwise} \end{cases}, \qquad o_k = \sum_{r < k} K_{i_r}.
$$

![EFB merges sparse exclusive features into one bundle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/12-XGBoost-and-LightGBM/fig4_efb.png)

Panel A shows a near-exclusive sparse block (think: 6 one-hot dummies). Panel B's red edges mark features that violate exclusivity; nodes are coloured by the bundle the greedy colourer assigned. Panel C shows the merged column: each original feature occupies a disjoint slice of the bin space, so the histogram for $\tilde f$ recovers exactly the per-feature statistics --- but now there is *one* feature instead of three or four.

In high-dimensional sparse problems EFB routinely cuts the effective $d$ by 5--10$\times$, which directly multiplies the histogram-construction speedup.

---

## Two definitions of "important"

XGBoost and LightGBM both expose feature importances, but they answer different questions.

![Feature importance: XGBoost gain vs LightGBM split count](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/12-XGBoost-and-LightGBM/fig6_feature_importance.png)

- **XGBoost (gain)** sums the gain term every time a feature is used for a split. It rewards features that produce *one or two huge splits*.
- **LightGBM (split count)** counts how many splits use each feature. It rewards features that are picked *often*, even if each split is modest.

The two rankings disagree --- and both are correct, just for different questions. Use gain when you ask "which features moved the loss?", split count when you ask "which features did the model rely on across iterations?". For a model-agnostic alternative, SHAP values measure the marginal contribution to each prediction.

---

## XGBoost vs LightGBM at a glance

| Dimension | XGBoost | LightGBM |
|---|---|---|
| Split strategy | Level-wise (BFS) | Leaf-wise (best-first) |
| Base algorithm | Pre-sorted exact / quantile sketch | Histogram, $K$ bins |
| Memory per feature | $O(N)$ | $O(K)$ |
| Sample efficiency | Row + column subsampling | GOSS |
| Sparse / categorical | Sparsity-aware default direction | EFB + native categorical |
| Failure mode | Slower on huge $N$ | Overfits without `max_depth` |
| Sweet spot | Small / medium $N$, careful tuning | Large $N$, large $d$, sparse features |

### Training cost in practice

The histogram + GOSS + EFB stack is what LightGBM trades into raw throughput:

![Training time vs dataset size and Pareto view](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/12-XGBoost-and-LightGBM/fig7_time_vs_accuracy.png)

At $N = 10^4$ all three libraries are within seconds of each other --- the choice doesn't matter. At $N = 10^6$ LightGBM is roughly $5\times$ faster than XGBoost for indistinguishable test accuracy. CatBoost lands between them on speed but owns the categorical-heavy regime through ordered boosting.

---

## A minimal XGBoost in NumPy

The simplest implementation that captures the second-order objective, the closed-form leaf weight, the gain-based split and the $\gamma$-pruning behaviour:

```python
import numpy as np


class XGBoostTree:
    """A single XGBoost tree: closed-form leaf weights + gain-based splits."""

    def __init__(self, max_depth=6, min_child_weight=1, gamma=0, lambda_=1):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.lambda_ = lambda_
        self.tree = None

    def fit(self, X, g, h):
        self.tree = self._build(X, g, h, depth=0)
        return self

    def _gain(self, GL, HL, GR, HR, G, H):
        return 0.5 * (
            GL**2 / (HL + self.lambda_)
            + GR**2 / (HR + self.lambda_)
            - G**2 / (H + self.lambda_)
        ) - self.gamma

    def _best_split(self, X, g, h):
        N, d = X.shape
        G, H = g.sum(), h.sum()
        best_gain, best = 0.0, None
        for j in range(d):
            order = np.argsort(X[:, j])
            xs, gs, hs = X[order, j], g[order], h[order]
            GL, HL = 0.0, 0.0
            for i in range(N - 1):
                GL += gs[i]; HL += hs[i]
                GR, HR = G - GL, H - HL
                if xs[i] == xs[i + 1]:
                    continue
                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue
                gain = self._gain(GL, HL, GR, HR, G, H)
                if gain > best_gain:
                    best_gain = gain
                    best = (j, 0.5 * (xs[i] + xs[i + 1]))
        return best, best_gain

    def _build(self, X, g, h, depth):
        G, H = g.sum(), h.sum()
        leaf_weight = -G / (H + self.lambda_)            # closed form
        if depth >= self.max_depth or len(X) < 2:
            return {"w": leaf_weight}
        split, gain = self._best_split(X, g, h)
        if split is None or gain <= 0:                    # gamma-pruning here
            return {"w": leaf_weight}
        j, thr = split
        m = X[:, j] <= thr
        return {
            "f": j, "t": thr,
            "L": self._build(X[m], g[m], h[m], depth + 1),
            "R": self._build(X[~m], g[~m], h[~m], depth + 1),
        }

    def predict(self, X):
        return np.array([self._pred(x, self.tree) for x in X])

    def _pred(self, x, n):
        if "w" in n:
            return n["w"]
        return self._pred(x, n["L"] if x[n["f"]] <= n["t"] else n["R"])


class XGBoost:
    """Additive ensemble with second-order gradient/Hessian boosting."""

    def __init__(self, n_estimators=100, lr=0.1, max_depth=6,
                 gamma=0, lambda_=1, objective="mse"):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.gamma = gamma
        self.lambda_ = lambda_
        self.objective = objective
        self.trees, self.base = [], None

    def _gh(self, y, p):
        if self.objective == "mse":
            return p - y, np.ones_like(y)
        s = 1.0 / (1.0 + np.exp(-p))                     # logistic
        return s - y, s * (1.0 - s)

    def fit(self, X, y):
        self.base = y.mean() if self.objective == "mse" else 0.0
        pred = np.full(len(y), self.base)
        for _ in range(self.n_estimators):
            g, h = self._gh(y, pred)
            tree = XGBoostTree(self.max_depth, gamma=self.gamma,
                               lambda_=self.lambda_).fit(X, g, h)
            self.trees.append(tree)
            pred += self.lr * tree.predict(X)
        return self

    def predict(self, X):
        p = np.full(len(X), self.base)
        for t in self.trees:
            p += self.lr * t.predict(X)
        return p
```

The whole second-order machinery fits in fewer than 100 lines because once you have the gain formula, the algorithm is just "find the best split, recurse, repeat".

---

## Q&A highlights

**Q1: Why second-order information at all?**
The first-order term tells you the direction; the second-order term tells you the curvature, i.e. how big a step is safe. Newton's method converges quadratically near the optimum exactly because of this. On a per-leaf basis it gives a *closed-form* optimal weight $-G/(H+\lambda)$ instead of a learning-rate-sensitive guess.

**Q2: Why does XGBoost prune through $\gamma$ rather than after the fact?**
Because the structure score is exactly the loss with $\gamma T$ subtracted. A split that fails to beat $\gamma$ is a split that *increases* the regularised loss. Refusing it is not a heuristic --- it is the correct decision under the objective.

**Q3: When does leaf-wise hurt you?**
On small datasets where the highest-gain leaf can chase noise. The fix is `max_depth` plus `min_data_in_leaf` (sometimes 100--1000 on small data). On large datasets leaf-wise is almost free lunch.

**Q4: Tuning order?**
1. Set `learning_rate = 0.05`--`0.1` and use early stopping to pick `n_estimators`.
2. Tune tree shape: `max_depth` (or `num_leaves` for LightGBM), `min_child_weight` / `min_data_in_leaf`.
3. Tune regularisation: `gamma`, `lambda`, plus subsampling (`subsample`, `colsample_bytree`).
4. If still under-fitting, drop `learning_rate` and grow `n_estimators`.

**Q5: GBDT vs deep learning?**
Tabular: GBDT wins almost always. Unstructured (image, text, audio): deep learning wins. The crossover lives somewhere around "you have a learned representation" --- once features are dense and continuous, deep nets become competitive.

---

## Exercises

**Exercise 1 -- second-order gradients.**
For squared loss $L = \tfrac{1}{2}(y - \hat y)^2$ with $y = 5$ and $\hat y = 3$: $g = \hat y - y = -2$, $h = 1$. The Newton step at a single leaf with $\lambda = 0$ is $w^* = -g/h = 2$, exactly the residual.

**Exercise 2 -- split gain.**
A leaf has $G = -2$, $H = 10$. A candidate split sends $G_L = -1.5, H_L = 6$ left and $G_R = -0.5, H_R = 4$ right. With $\lambda = 1, \gamma = 0.5$:

$$
\text{Gain} = \tfrac{1}{2}\!\left[\tfrac{2.25}{7} + \tfrac{0.25}{5} - \tfrac{4}{11}\right] - 0.5 \approx -0.50.
$$

The structural improvement does not pay for $\gamma$ --- skip the split. Increasing $\gamma$ tightens the threshold; decreasing $\lambda$ relaxes it.

**Exercise 3 -- GOSS sample budget.**
With $N = 1000$, $a = 0.2$, $b = 0.1$: keep $200$ large-gradient samples plus $0.1 \cdot 800 = 80$ small-gradient samples. Effective sample size $= 280$ ($28\%$). Reweight factor for the small set: $(1 - a)/b = 8$.

---

## References

- Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
- Ke, G., Meng, Q., Finley, T., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
- Prokhorenkova, L., Gusev, G., Vorobev, A., et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features*. NeurIPS.
- Friedman, J. H. (2001). *Greedy function approximation: a gradient boosting machine*. Annals of Statistics.

---

## Series Navigation

- Previous: [Part 11 -- Ensemble Learning](/en/Machine-Learning-Mathematical-Derivations-11-Ensemble-Learning/)
- Next: [Part 13 -- EM Algorithm and GMM](/en/Machine-Learning-Mathematical-Derivations-13-EM-Algorithm-and-GMM/)
- [View all 20 parts in this series](/tags/Machine-Learning/)
