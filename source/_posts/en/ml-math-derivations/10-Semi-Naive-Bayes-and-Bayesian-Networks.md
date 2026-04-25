---
title: "Machine Learning Mathematical Derivations (10): Semi-Naive Bayes and Bayesian Networks"
date: 2026-01-29 09:00:00
tags:
  - Machine Learning
  - Semi-Naive Bayes
  - Bayesian Networks
  - Probabilistic Graphical Models
  - TAN
  - AODE
  - Mathematical Derivation
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 10
  total: 20
lang: en
mathjax: true
description: "From SPODE, TAN and AODE to full Bayesian networks: how relaxing the conditional-independence assumption -- through one-dependence trees, ensembles of super-parents and graphical structure learning -- closes the gap between Naive Bayes and the full joint distribution."
disableNunjucks: true
series_order: 10
---

> **Hook.** Naive Bayes assumes every feature is conditionally independent given the class. It is a convenient lie -- one that lets us train in a single pass over the data, but one that classifiers based on tree structures and small graphs can systematically beat by a few accuracy points on virtually every UCI benchmark. This part walks the spectrum from "no dependencies" (Naive Bayes) to "all dependencies" (full joint), showing the three sweet spots that practitioners actually use: SPODE, TAN and AODE. The same factorisation idea, taken to its general form, is the Bayesian network.

## What you will learn

- Why the full conditional-independence assumption fails, and how the parameter count grows when we relax it.
- **SPODE**: every feature gets a single shared "super-parent".
- **TAN**: each feature gets its *own* parent, chosen by a maximum-spanning-tree on conditional mutual information (Chow-Liu).
- **AODE**: averaging over all eligible super-parents to remove model-selection variance.
- **Bayesian networks**: DAG factorisation, d-separation, the Markov blanket, variable elimination and the junction tree.
- A practical decision rule: NB vs TAN vs AODE vs full BN.

## Prerequisites

- Joint distributions, conditional independence, mutual information.
- Spanning trees, dynamic programming on trees.
- Familiarity with [Part 9: Naive Bayes](/en/Machine-Learning-Mathematical-Derivations-9-Naive-Bayes/).

---

## 1. Why relax the independence assumption?

### 1.1 The convenient lie

Recall the Naive Bayes factorisation:

$$
P(\mathbf{x}\mid c_k) \;=\; \prod_{j=1}^{d} P(x^{(j)}\mid c_k).
$$

The lie is that, conditioned on the class, features are unrelated. Three places it almost always breaks:

- **Text.** The bigram "machine learning" is far more frequent than $P(\text{machine})\,P(\text{learning})$ would suggest, even within the "ML article" class.
- **Medicine.** Within "flu", *fever* and *cough* remain strongly correlated -- they share a downstream physiological pathway.
- **Vision.** Adjacent pixels are nearly identical regardless of the class label.

When the conditional-independence assumption is severely violated, the posterior $P(c\mid\mathbf{x})$ is *miscalibrated*: the "winning" class still wins, but with wildly inflated confidence, and the ranking among close runners-up degrades.

### 1.2 The spectrum

| Model | Allowed dependencies | # parameters | Accuracy |
|---|---|---|---|
| Naive Bayes | none | $O(K d)$ | baseline |
| 1-dependence (SPODE / TAN / AODE) | each feature has 1 parent | $O(K d S)$ | better |
| $k$-dependence | each feature has $k$ parents | $O(K d S^{k})$ | even better |
| Full joint | everything | $O(K S^{d})$ | best, but intractable |

Here $S$ is the average number of values per feature and $K$ the number of classes. **The empirical sweet spot is $k=1$**: the cost is one extra factor of $S$, and the accuracy gain over Naive Bayes is consistently the largest single jump.

---

## 2. One-dependence estimators (ODE)

### 2.1 SPODE: a single super-parent

Pick one feature $x^{(p)}$ -- the *super-parent* -- and let every other feature condition on both the class and $x^{(p)}$:

$$
P(\mathbf{x}\mid c_k) \;=\; P(x^{(p)}\mid c_k)\prod_{j\neq p} P(x^{(j)}\mid c_k,\,x^{(p)}).
$$

**Reading the model.** Instead of "all features are independent given the class", SPODE says "all features share *one* hidden moderator beyond the class". For spam detection that moderator might be the token *free*: knowing whether *free* appears changes the probabilities of *click*, *offer*, *winner*.

**Maximum-likelihood estimate** (with Laplace smoothing $\alpha$):

$$
\hat{P}(x^{(j)}{=}v\mid c_k,\,x^{(p)}{=}u)
\;=\;
\frac{\#\{x^{(j)}{=}v,\,c_k,\,x^{(p)}{=}u\}+\alpha}
     {\#\{c_k,\,x^{(p)}{=}u\}+\alpha\,S_j}.
$$

**Choosing the super-parent.** Two standard criteria:

$$
p^{*} \;=\; \arg\max_{p}\, I(X^{(p)};Y),
\qquad
p^{*} \;=\; \arg\max_{p}\,\sum_{j\neq p} I(X^{(j)};Y\mid X^{(p)}).
$$

The first picks the feature most informative about $Y$; the second picks the feature whose conditioning maximally explains the rest. Both are heuristics -- and the next two models exist precisely because this single choice is fragile.

### 2.2 AODE: average all eligible super-parents

![AODE: each super-parent yields a SPODE; AODE averages them](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/10-Semi-Naive-Bayes-and-Bayesian-Networks/fig4_aode.png)

**Idea.** Rather than betting on a single super-parent, AODE averages over *all* of them whose support is large enough to estimate reliably:

$$
\hat{P}(c_k\mid\mathbf{x})\;\propto\;
\sum_{i:\;n_i\geq m} P(c_k)\,P(x^{(i)}\mid c_k)\prod_{j\neq i} P(x^{(j)}\mid c_k,\,x^{(i)}).
$$

Here $n_i$ is the number of training samples with attribute value $x^{(i)}$ and $m$ (typically 30) is the minimum support for a super-parent to be trusted.

**Why it works.**

1. *No structure search.* The model is fixed; only counts are estimated.
2. *Variance reduction.* Averaging $d$ correlated SPODE estimates is a variance-shrinker, just like bagging.
3. *Trivial to ship.* Same counters as SPODE, summed over all $i$.

```python
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score

# Three binary features where x1, x2 BOTH depend on x0 given the class.
np.random.seed(42)
N = 500
y  = np.random.binomial(1, 0.5, N)
x0 = np.random.binomial(1, 0.8 * y + 0.2 * (1 - y), N)
x1 = np.random.binomial(1, 0.7 * x0 + 0.1 * (1 - x0), N)  # depends on x0
x2 = np.random.binomial(1, 0.6 * x0 + 0.2 * (1 - x0), N)  # depends on x0
X  = np.column_stack([x0, x1, x2])

print("Naive Bayes 5-fold CV:",
      cross_val_score(BernoulliNB(), X, y, cv=5).mean().round(4))
# AODE would consistently beat this here because conditional independence
# between x1 and x2 given y is violated -- they share x0 as a mediator.
```

---

## 3. TAN: tree-augmented Naive Bayes

### 3.1 Each feature picks its own parent

![Naive Bayes (star) vs TAN (star + augmenting tree)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/10-Semi-Naive-Bayes-and-Bayesian-Networks/fig3_nb_vs_tan.png)

Where SPODE forces every feature to share one super-parent, TAN lets each feature choose its own *single* extra parent -- a feature-level tree:

$$
P(\mathbf{x}\mid c_k) \;=\; \prod_{j=1}^{d} P(x^{(j)}\mid c_k,\,x^{(\pi_j)}),
$$

where $\pi_j$ is the unique parent of feature $j$ in the augmenting tree (the root has no parent, reducing to $P(x^{(j)}\mid c_k)$).

### 3.2 Learning the tree -- Chow-Liu

**Theorem (Chow & Liu 1968; Friedman, Geiger, Goldszmidt 1997).** Among all tree-shaped augmenting structures, the one that maximises the conditional log-likelihood is the **maximum spanning tree** with edge weights

$$
w_{jk} \;=\; I(X^{(j)};X^{(k)}\mid Y)
\;=\;
\sum_{y,x_j,x_k} P(x_j,x_k,y)\,
\log\frac{P(x_j,x_k\mid y)}{P(x_j\mid y)\,P(x_k\mid y)}.
$$

**Reading the weight.** $I(X^{(j)};X^{(k)}\mid Y)$ measures the residual correlation between $X^{(j)}$ and $X^{(k)}$ that the class has *not* already explained. Edges with high weight are exactly the dependencies most worth modelling.

**Algorithm.**

1. Build a complete graph over the $d$ features.
2. Set edge weight $(j,k)$ to $I(X^{(j)};X^{(k)}\mid Y)$.
3. Run Kruskal or Prim to obtain the maximum spanning tree -- $O(d^{2}\log d)$.
4. Pick a root (commonly $\arg\max_j I(X^{(j)};Y)$).
5. Orient edges away from the root.

```python
import numpy as np
from itertools import combinations

def conditional_mutual_info(X, y, j, k):
    """Compute I(X_j ; X_k | Y) from data, base e."""
    cmi, N = 0.0, len(y)
    for c in np.unique(y):
        mask = (y == c); p_c = mask.mean()
        Xc = X[mask]
        for vj in np.unique(X[:, j]):
            for vk in np.unique(X[:, k]):
                p_jk = ((Xc[:, j] == vj) & (Xc[:, k] == vk)).mean()
                p_j  = (Xc[:, j] == vj).mean()
                p_k  = (Xc[:, k] == vk).mean()
                if min(p_jk, p_j, p_k) > 0:
                    cmi += p_c * p_jk * np.log(p_jk / (p_j * p_k))
    return cmi
```

The CMI for the dependent pair $(x_0,x_1)$ in the synthetic example above will be visibly larger than for the independent pair $(x_0,x_2)$ -- that ordering is exactly what Kruskal will exploit.

---

## 4. Bayesian networks

Both NB and TAN are special cases of a general object: a Bayesian network. We now zoom out to the full graphical-models picture.

### 4.1 DAG factorisation

![Toy Bayesian network: Cloudy / Sprinkler / Rain / WetGrass](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/10-Semi-Naive-Bayes-and-Bayesian-Networks/fig1_dag.png)

A **Bayesian network** is a directed acyclic graph $\mathcal{G}=(V,E)$ in which each node carries a *conditional probability table* (CPT). The joint distribution factors as

$$
P(X_1,\dots,X_n) \;=\; \prod_{i=1}^{n} P\bigl(X_i\mid \mathrm{Pa}(X_i)\bigr).
$$

**Why factorisation matters.** In the four-variable network above, the joint over four binary variables would have $2^4 - 1 = 15$ free parameters. The factored form needs $1+2+2+4 = 9$. With $n=20$ binary variables and average parent count 3, the joint has $\approx 10^{6}$ entries while the factored form has $\sim 20\cdot 2^{3}=160$.

**Special cases we have already met:**

- *Naive Bayes* is the star graph: $Y$ is the parent of every $X_j$, no edges among the $X_j$.
- *TAN* is a star plus a tree on the $X_j$.

### 4.2 d-separation

![The three patterns: chain, fork, collider](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/10-Semi-Naive-Bayes-and-Bayesian-Networks/fig2_d_separation.png)

Conditional independence in a DAG is decided by the graphical criterion **d-separation**. Three structures generate every case:

- **Chain $A\to B\to C$.** With $B$ unobserved, $A\not\!\perp C$. Conditioning on $B$ blocks the path: $A\perp C\mid B$.
- **Fork $A\leftarrow B\to C$.** Same behaviour as the chain: a common cause makes $A,C$ correlated, observing it removes the correlation.
- **Collider $A\to B\leftarrow C$.** Behaves *oppositely*. Without $B$, $A\perp C$. *Observing* $B$ activates the path, coupling its two causes -- the famous **explaining-away** effect.

A path is **blocked** if any chain or fork node on it is observed, *or* any collider on it is *not* observed (and none of its descendants are). Two sets are d-separated by $Z$ iff every path between them is blocked by $Z$. This is the engine that drives every conditional-independence claim in graphical models.

### 4.3 Markov blanket

![Markov blanket = parents + children + co-parents](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/10-Semi-Naive-Bayes-and-Bayesian-Networks/fig6_markov_blanket.png)

The **Markov blanket** $\mathrm{MB}(X)$ is the *smallest* set such that $X$ is independent of everything outside $\mathrm{MB}(X)$ given $\mathrm{MB}(X)$. In a Bayesian network it is exactly:

$$
\mathrm{MB}(X) \;=\; \mathrm{Pa}(X)\;\cup\;\mathrm{Ch}(X)\;\cup\;\bigl\{\text{other parents of }X\text{'s children}\bigr\}.
$$

The co-parents are easy to forget but essential -- they appear because, by the collider rule, conditioning on a child *activates* the path through that child to its other parents. The blanket is the minimal set of features a Gibbs sampler needs to resample $X$, and in causal terms it is the smallest sufficient statistic for predicting $X$ from the rest of the network.

### 4.4 Inference: variable elimination

![Variable elimination on a chain](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/10-Semi-Naive-Bayes-and-Bayesian-Networks/fig5_variable_elimination.png)

To compute a posterior such as $P(D\mid \text{evidence})$, the brute-force sum is exponential in $n$. **Variable elimination** sums variables out one at a time, reusing intermediate factors:

1. Pick an elimination order, e.g. $A,B,C$ for a query about $D$.
2. Multiply all factors that mention $A$, then $\sum_A$ to obtain a new factor $\tau_1(B)$.
3. Repeat: multiply factors that mention $B$, sum to get $\tau_2(C)$, and so on.

For a chain this is $O(nS^{2})$ instead of $O(S^{n})$. For a general graph the complexity is governed by the **treewidth** of the graph -- equivalently, by the largest intermediate factor produced -- which is why dense graphs remain hard even after elimination.

### 4.5 The junction tree

![Junction tree pipeline: triangulate -> max cliques -> separator-labelled tree](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/10-Semi-Naive-Bayes-and-Bayesian-Networks/fig7_junction_tree.png)

The **junction tree** algorithm globalises variable elimination so that *all* marginals can be computed by two message-passing sweeps over a single auxiliary structure:

1. **Moralise.** Add an undirected edge between every pair of co-parents and drop edge directions.
2. **Triangulate.** Add chords until every cycle of length $\geq 4$ has a chord. This step controls treewidth and is NP-hard in the optimal case -- heuristics such as *minimum fill-in* work well in practice.
3. **Extract maximal cliques** of the triangulated graph.
4. **Build a tree of cliques** in which adjacent cliques share a separator set, satisfying the *running-intersection property*: any variable appears in a connected sub-tree.
5. **Pass messages** along the tree (collect-then-distribute). After two sweeps, every clique holds the correct marginal over its variables.

The cost is exponential in the largest clique size, i.e. treewidth $+\,1$ -- the same complexity bound as variable elimination, but amortised across all queries.

### 4.6 Structure learning

The DAG itself can be learned. Two main families:

- **Score + search.** Hill-climb over DAGs by adding/deleting/reversing edges; score each candidate with a likelihood-plus-penalty criterion. The standard one is **BIC**:

$$
\mathrm{BIC}(\mathcal{G}) \;=\; \log p(\mathcal{D}\mid\hat{\theta},\mathcal{G}) \;-\; \tfrac{|\theta|}{2}\log N.
$$

The first term rewards fit, the second penalises parameter count -- BIC is consistent (selects the true graph as $N\to\infty$) under standard assumptions.

- **Constraint-based** (PC algorithm, FCI). Run conditional-independence tests, build the skeleton, then orient v-structures using the collider rule.

A hard fact: *exact* structure learning is NP-hard (Chickering, Heckerman & Meek 2004). All practical algorithms cap the number of parents per node, use random restarts, or exploit prior knowledge about edge directions.

---

## 5. Worked exercises

### Exercise 1 -- SPODE parameter estimate

100 samples; class $c_1$ has 60. In $c_1$, 30 samples have $x^{(p)}{=}1$, of which 18 have $x^{(j)}{=}1$. With Laplace smoothing $\alpha=1$ and $|x^{(j)}|=3$,

$$
\hat{P}\bigl(x^{(j)}{=}1\mid c_1, x^{(p)}{=}1\bigr) \;=\; \frac{18+1}{30+1\cdot 3} \;=\; \frac{19}{33} \;\approx\; 0.576.
$$

### Exercise 2 -- TAN parameter count

For binary features and $K$ classes:

- Naive Bayes: $K\cdot d$ parameters (one per class-feature combination).
- TAN: each non-root feature has the class **and** one binary parent, hence $2$ extra entries per class-feature pair -- total $2Kd$.

TAN doubles the parameter budget and gains the ability to model pairwise dependencies. With $S$-valued features the multiplier is $S$ instead of $2$.

### Exercise 3 -- Chow-Liu edge selection

Suppose $I(X_1;X_2\mid Y)=0.8$ and $I(X_1;X_3\mid Y)=0.5$. The Chow-Liu MST greedily picks the heavier edge first, so $X_1\!-\!X_2$ enters the tree before $X_1\!-\!X_3$. The lighter edge enters only if it does not close a cycle.

### Exercise 4 -- d-separation

In $A\to C\leftarrow B$:

- (a) $A\perp B$? **Yes.** $C$ is a collider, and *unobserved* colliders block the path.
- (b) $A\perp B\mid D$ where $D$ is unrelated? **Yes.** $D$ is not a descendant of $C$, so the collider stays blocked.
- (c) $A\perp B\mid C$? **No.** Conditioning on a collider activates the path -- explaining away couples $A$ and $B$.

### Exercise 5 -- AODE vs TAN with limited data

- Large $d$, plenty of data: TAN's $O(d)$ prediction beats AODE's $O(d^{2})$.
- Small $N$: AODE's averaging acts as regulariser, often more robust than TAN's structure search which can overfit MST edges from noisy CMI estimates.

A reasonable default policy: start with AODE; switch to TAN once $N$ comfortably exceeds the number of CMI cells you need to estimate.

---

## Q&A

### Why not always use a full Bayesian network?

Structure learning is NP-hard and the search space of DAGs is super-exponential. TAN and AODE *constrain* the structure to keep learning tractable while capturing the most informative dependencies.

### Do directed edges encode causation?

No. Two DAGs can encode the same conditional-independence structure (an *equivalence class*) and be statistically indistinguishable. Causal claims require interventional or counterfactual assumptions -- Pearl's do-calculus, instrumental variables, etc.

### Continuous features in TAN?

Three options: (i) discretise (equal-width or equal-frequency bins), (ii) assume conditional Gaussians (a *Conditional Gaussian Bayesian network*), (iii) use kernel density estimates for the local CPDs.

### AODE or TAN -- which?

AODE: simpler, more robust on small samples, no structure-learning step. TAN: faster prediction at high $d$, captures heterogeneous parent choices. In production text classifiers AODE is the more common default; in dense numeric tables, TAN often wins.

### Where do deep generative models fit?

Modern latent-variable models (VAE, normalising flows, autoregressive models like PixelCNN) are continuous Bayesian networks where the CPTs are neural conditional densities. The factorisation idea is identical; only the parametric form of $P(X_i\mid \mathrm{Pa}(X_i))$ has changed.

---

## References

- Friedman, N., Geiger, D. & Goldszmidt, M. (1997). *Bayesian network classifiers.* Machine Learning, 29(2-3), 131-163.
- Webb, G. I., Boughton, J. R. & Wang, Z. (2005). *Not so naive Bayes: Aggregating one-dependence estimators.* Machine Learning, 58(1), 5-24.
- Chow, C. & Liu, C. (1968). *Approximating discrete probability distributions with dependence trees.* IEEE Transactions on Information Theory, 14(3), 462-467.
- Chickering, D. M., Heckerman, D. & Meek, C. (2004). *Large-sample learning of Bayesian networks is NP-hard.* JMLR, 5, 1287-1330.
- Koller, D. & Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques.* MIT Press.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems.* Morgan Kaufmann.

---

<div class="series-nav">

**ML Mathematical Derivations Series**

[< Part 9: Naive Bayes](/en/Machine-Learning-Mathematical-Derivations-9-Naive-Bayes/) | **Part 10: Semi-Naive Bayes & Bayesian Networks** | [Part 11: Ensemble Learning >](/en/Machine-Learning-Mathematical-Derivations-11-Ensemble-Learning/)

</div>
