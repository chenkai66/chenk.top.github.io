---
title: "Machine Learning Mathematical Derivations (8): Support Vector Machines"
date: 2026-01-27 09:00:00
tags:
  - Machine Learning
  - Support Vector Machines
  - SVM
  - Kernel Methods
  - KKT Conditions
  - Dual Problem
  - Mathematical Derivation
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 8
  total: 20
lang: en
mathjax: true
description: "Complete SVM derivation from maximum margin to Lagrangian duality, KKT conditions, soft margin, kernel trick, and SMO algorithm with step-by-step proofs and Python code."
disableNunjucks: true
series_order: 8
---

> **Hook.** You have two clouds of points and infinitely many lines that separate them. Which line is "best"? SVM gives a startlingly geometric answer: the line that sits in the middle of the *widest empty corridor* between the two classes. Push that single idea through Lagrangian duality and it produces a sparse model (only the points on the corridor wall matter), a quadratic program with a global optimum, and -- almost as a free gift -- the kernel trick that lets the same linear machinery carve curved boundaries in infinite-dimensional spaces.

## What you will learn

- How "widest corridor" becomes a convex quadratic program with linear constraints
- Why the dual problem is more useful than the primal, and how KKT conditions force sparsity
- The soft-margin variant and what the regularizer $C$ really controls
- The kernel trick: replacing $\phi(x)^\top \phi(z)$ with $K(x, z)$ and never building $\phi$
- Mercer's condition and the standard kernel zoo (linear / polynomial / RBF / sigmoid)
- The SMO algorithm: why two coordinates at a time, how clipping works, why it converges
- Hinge loss, the bridge from SVM to the rest of supervised learning

## Prerequisites

- Linear algebra: inner products, hyperplanes, projections
- Calculus: Lagrange multipliers, partial derivatives
- A glance at convex duality is helpful but not required
- Familiarity with [Part 7: Decision Trees](/en/Machine-Learning-Mathematical-Derivations-7-Decision-Trees/)

---

## 1. Hard-margin SVM

### 1.1 Functional and geometric margin

Take binary labels $y_i \in \{-1, +1\}$ and a linear decision rule $\hat{y} = \operatorname{sign}(w^\top x + b)$. Two notions of "how far" a point sits from the boundary:

$$
\hat{\gamma}_i \;=\; y_i\,(w^\top x_i + b)
\qquad\text{(functional margin)}
$$

$$
\gamma_i \;=\; \frac{y_i\,(w^\top x_i + b)}{\lVert w \rVert}
\qquad\text{(geometric margin)}
$$

The functional margin is positive on correctly classified points but is *not* scale-invariant: doubling $(w, b)$ doubles it. The geometric margin is the actual Euclidean distance from $x_i$ to the hyperplane, signed by the label, and is invariant to rescaling. That invariance is what makes the optimization well-posed -- without it, "make the margin bigger" has no fixed-point answer, you can always shrink $\lVert w \rVert$.

*Why this formula?* For any point $x_0$, the closest point on $w^\top x + b = 0$ is its orthogonal projection, and the displacement is $-(w^\top x_0 + b)/\lVert w \rVert^2 \cdot w$. Its norm is $\lvert w^\top x_0 + b\rvert / \lVert w \rVert$. Multiplying by $y_i$ keeps it positive whenever the prediction is correct.

### 1.2 The maximum-margin program

![Hard-margin SVM: the maximum-margin hyperplane and its support vectors](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/08-Support-Vector-Machines/fig1_max_margin.png)

We want the hyperplane that maximises the *worst-case* geometric margin:

$$
\max_{w, b}\; \min_i\; \frac{y_i\,(w^\top x_i + b)}{\lVert w \rVert}
$$

This looks awkward because $(w, b)$ is only defined up to a positive rescaling. We pin the scale down by demanding that the closest points have functional margin exactly $1$. Equivalent program:

$$
\boxed{\;\min_{w, b}\; \tfrac{1}{2}\lVert w \rVert^2
\quad \text{s.t.} \quad y_i(w^\top x_i + b) \;\ge\; 1, \quad i = 1, \dots, N.\;}
$$

This is a **convex quadratic program** with linear constraints. Strict convexity of the objective gives a unique optimal $w^\*$; the bias $b^\*$ is unique whenever the data span both classes.

The points where the constraint is tight, $y_i(w^\top x_i + b) = 1$, are the **support vectors**. Geometrically they sit on the two parallel margin lines flanking the boundary; algebraically, they are the only points that determine $w^\*$.

### 1.3 The dual via Lagrange

Attach multipliers $\alpha_i \ge 0$ to each constraint:

$$
L(w, b, \alpha) \;=\; \tfrac{1}{2}\lVert w \rVert^2 \;-\; \sum_{i=1}^N \alpha_i \bigl[\,y_i(w^\top x_i + b) - 1\,\bigr].
$$

**Step 1 -- minimise over $w$ and $b$.** Setting $\partial_w L = 0$ and $\partial_b L = 0$:

$$
w^\* \;=\; \sum_{i=1}^N \alpha_i y_i x_i,
\qquad
\sum_{i=1}^N \alpha_i y_i \;=\; 0.
$$

**Step 2 -- substitute back.** Using $w^\* = \sum_i \alpha_i y_i x_i$ inside $L$:

$$
W(\alpha) \;=\; \sum_{i=1}^N \alpha_i \;-\; \tfrac{1}{2} \sum_{i, j} \alpha_i \alpha_j y_i y_j\, x_i^\top x_j.
$$

The dual program:

$$
\boxed{\;\max_{\alpha}\; \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j\, x_i^\top x_j
\quad \text{s.t.} \quad \alpha_i \ge 0, \;\; \sum_i \alpha_i y_i = 0.\;}
$$

Two consequences are doing all the heavy lifting here:

1. The data only enter through inner products $x_i^\top x_j$. Replace this with $K(x_i, x_j)$ and the entire derivation goes through unchanged -- this is the kernel trick before we even introduce it.
2. The dual has $N$ scalar variables and a single equality constraint, so it is much cheaper than the primal whenever the feature dimension $d \gg N$.

### 1.4 KKT conditions and sparsity

The primal is convex with affine constraints, so Slater's condition is satisfied and **strong duality** holds. Optimal $(w^\*, b^\*, \alpha^\*)$ must obey:

- **Primal feasibility:** $y_i(w^{\*\top} x_i + b^\*) \ge 1$.
- **Dual feasibility:** $\alpha_i^\* \ge 0$.
- **Stationarity:** $w^\* = \sum_i \alpha_i^\* y_i x_i$ and $\sum_i \alpha_i^\* y_i = 0$.
- **Complementary slackness:** $\alpha_i^\* \cdot \bigl[\, y_i(w^{\*\top} x_i + b^\*) - 1 \,\bigr] = 0$.

The last line is the punchline. For each $i$ exactly one of these holds:

- $\alpha_i^\* = 0$ -- the point is *strictly* outside the margin band, irrelevant to $w^\*$.
- $y_i(w^{\*\top} x_i + b^\*) = 1$ -- the point sits *on* the margin and can carry $\alpha_i^\* > 0$.

Therefore the optimal classifier is supported by only the second group:

$$
f(x) \;=\; \sum_{i \in \mathrm{SV}} \alpha_i^\* y_i\, x_i^\top x \;+\; b^\*.
$$

For prediction, you can throw away every non-SV training point. That is the model's defining sparsity.

```python
import numpy as np
from sklearn.svm import SVC

# A tiny separable problem
X = np.array([[1, 2], [2, 3], [3, 3],
              [1, 0], [2, 1], [0, 1]], dtype=float)
y = np.array([1, 1, 1, -1, -1, -1])

clf = SVC(kernel="linear", C=1e6).fit(X, y)   # C huge ~ hard margin
w, b = clf.coef_[0], clf.intercept_[0]

print(f"w = {w},  b = {b:.4f}")
print(f"#SV = {len(clf.support_)} of {len(X)} samples")
for i in clf.support_:
    print(f"  SV idx {i}: y(w.x+b) = {y[i] * (X[i] @ w + b):.4f}  "
          f"(should be ~1)")
```

---

## 2. Soft-margin SVM

### 2.1 Slack variables and the $C$ knob

Real data overlap. We let each point misbehave by a non-negative amount $\xi_i$:

$$
\min_{w, b, \xi}\; \tfrac{1}{2}\lVert w \rVert^2 \;+\; C \sum_i \xi_i
\quad \text{s.t.} \quad y_i(w^\top x_i + b) \ge 1 - \xi_i,\; \xi_i \ge 0.
$$

Reading $\xi_i$:

- $\xi_i = 0$: outside the margin, classified correctly.
- $0 < \xi_i \le 1$: inside the margin band, still on the right side.
- $\xi_i > 1$: misclassified.

$C > 0$ trades two evils. Big $C$ punishes slack heavily, narrows the margin and approaches the hard-margin model. Small $C$ pays slack cheaply, widens the margin and tolerates more noise.

![Soft-margin SVM at three values of C: wider margin pays in slack, narrower margin pays in $\lVert w \rVert$](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/08-Support-Vector-Machines/fig2_soft_margin_C.png)

### 2.2 Dual: the box constraint

Repeat the Lagrangian recipe, this time with multipliers $\alpha_i \ge 0$ for the margin constraints and $\mu_i \ge 0$ for $\xi_i \ge 0$. Setting $\partial_\xi L = 0$ gives $\alpha_i + \mu_i = C$, hence $0 \le \alpha_i \le C$. The dual:

$$
\boxed{\;\max_{\alpha}\; \sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j\, x_i^\top x_j
\quad \text{s.t.} \quad 0 \le \alpha_i \le C, \;\; \sum_i \alpha_i y_i = 0.\;}
$$

Only difference from hard margin: an upper bound $\alpha_i \le C$. The KKT conditions now define **three regimes**:

| Regime | Conditions | Interpretation |
|---|---|---|
| Inactive | $\alpha_i = 0$, $\xi_i = 0$ | safely outside the margin |
| Boundary SV | $0 < \alpha_i < C$, $\xi_i = 0$ | exactly on the margin; usable to recover $b^\*$ |
| Bound SV | $\alpha_i = C$, $\xi_i > 0$ | inside the margin or misclassified |

![KKT geometry: complementary slackness partitions the data into three regimes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/08-Support-Vector-Machines/fig6_kkt_geometry.png)

The bias is recovered from any boundary support vector: $b^\* = y_i - \sum_j \alpha_j^\* y_j x_j^\top x_i$. For numerical stability, average this over all boundary SVs.

### 2.3 Hinge loss view

Eliminate $\xi_i$ from the primal by noting that the optimal slack is $\xi_i^\* = \max(0,\, 1 - y_i(w^\top x_i + b))$. Substituting:

$$
\min_{w, b}\; \tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \max\bigl(0,\, 1 - y_i(w^\top x_i + b)\bigr).
$$

The right-hand sum is the **hinge loss**. So soft-margin SVM is exactly *L2-regularised empirical risk minimisation with hinge loss*. This view makes SVM look like every other linear classifier you know -- only the loss function differs.

![Surrogate losses on the margin axis: hinge upper-bounds 0/1 loss and is convex; squared loss penalises confident-correct examples](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/08-Support-Vector-Machines/fig5_loss_comparison.png)

The hinge has a kink at $m = 1$ and is exactly zero beyond, which is what creates the SV sparsity: confidently correct points contribute zero gradient and zero $\alpha$.

---

## 3. Kernels

### 3.1 The kernel trick

Map inputs to some feature space $\phi: \mathbb{R}^d \to \mathcal{H}$. Run linear SVM in $\mathcal{H}$. The dual objective becomes

$$
W(\alpha) \;=\; \sum_i \alpha_i \;-\; \tfrac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \;\phi(x_i)^\top \phi(x_j).
$$

We never used $\phi$ outside an inner product. Define a **kernel** as that inner product:

$$
K(x, z) \;=\; \phi(x)^\top \phi(z).
$$

Anywhere $x_i^\top x_j$ appeared in the dual or in the prediction, write $K(x_i, x_j)$. We never construct $\phi$, never store features, never even need $\dim \mathcal{H}$ to be finite.

![The kernel trick: rings that no line can separate become flat-plane separable in $(x_1, x_2, x_1^2 + x_2^2)$](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/08-Support-Vector-Machines/fig3_kernel_trick_3d.png)

The lifted picture shows *why* this works. The inner two-class data on the left admits no separating line. Lifting $\phi(x) = (x_1, x_2, x_1^2 + x_2^2)$ pushes the outer ring upward in the third coordinate, and the horizontal plane $x_1^2 + x_2^2 = c$ separates the classes. Computing inner products in $\mathbb{R}^3$ via $K(x, z) = (1 + x^\top z)^2$ achieves the same effect without ever forming $\phi$.

### 3.2 The kernel zoo

| Kernel | Formula | Feature space |
|---|---|---|
| Linear | $K(x, z) = x^\top z$ | the input space itself |
| Polynomial | $K(x, z) = (\gamma\, x^\top z + r)^p$ | all monomials up to degree $p$ |
| RBF (Gaussian) | $K(x, z) = \exp(-\gamma\,\lVert x - z\rVert^2)$ | infinite-dimensional |
| Sigmoid | $K(x, z) = \tanh(\gamma\, x^\top z + r)$ | only PSD for some parameters |

**RBF intuition.** Each training point lays down a bump of radius $1/\sqrt{\gamma}$. Large $\gamma$ -- narrow bumps, decision boundary wraps tightly around individual points (overfit risk). Small $\gamma$ -- wide bumps, smoother boundary, possible underfit.

![RBF SVM at three values of $\gamma$: bandwidth controls how locally each support vector influences the boundary](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/08-Support-Vector-Machines/fig4_rbf_boundary.png)

### 3.3 Mercer's condition

**Theorem.** A symmetric function $K(x, z)$ corresponds to some inner product $\phi(x)^\top \phi(z)$ in a Hilbert space if and only if for every finite point set $\{x_i\}_{i=1}^N$, the kernel matrix $\mathbf{K}_{ij} = K(x_i, x_j)$ is positive semi-definite, i.e.

$$
\sum_{i, j} c_i c_j\, K(x_i, x_j) \;\ge\; 0 \quad \text{for all } c \in \mathbb{R}^N.
$$

This is what licenses the trick: PSD kernels *are* inner products, by construction of the reproducing-kernel Hilbert space (RKHS).

**Useful closure rules.** If $K_1$ and $K_2$ are kernels, so are: $K_1 + K_2$, $\lambda K_1$ ($\lambda \ge 0$), $K_1 \cdot K_2$, $f(x) K_1(x, z) f(z)$, polynomials of $K_1$ with non-negative coefficients, and $\exp(K_1)$. RBF arises from these closure rules applied to the linear kernel.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# XOR: linearly inseparable, RBF solves it instantly
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])

print("linear:", accuracy_score(y, SVC(kernel="linear").fit(X, y).predict(X)))
print("rbf   :", accuracy_score(y, SVC(kernel="rbf", gamma=5).fit(X, y).predict(X)))

# Verify the Gram matrix is PSD
gamma = 5.0
K = np.exp(-gamma * np.sum((X[:, None] - X[None, :]) ** 2, axis=-1))
eig = np.linalg.eigvalsh(K)
print(f"eigenvalues: {eig.round(4)} -- all >= 0: {np.all(eig >= -1e-10)}")
```

---

## 4. The SMO algorithm

### 4.1 Why two coordinates at a time

The dual is a QP in $N$ variables with a single equality constraint $\sum_i \alpha_i y_i = 0$. General QP solvers run in $O(N^3)$ time and $O(N^2)$ memory -- prohibitive for tens of thousands of points.

A naive coordinate-descent strategy (fix all but one $\alpha_i$ and optimise) is illegal: the equality constraint forces *any* legal move to change at least two coordinates. The minimum number of variables to update while staying feasible is therefore **two**. That is the entire premise of Sequential Minimal Optimization (Platt, 1998): pick a pair, solve the two-variable QP analytically, repeat.

### 4.2 The two-variable sub-problem

Pick indices $1, 2$ and freeze the others. The equality constraint reduces to

$$
\alpha_1 y_1 + \alpha_2 y_2 \;=\; -\sum_{i \ge 3} \alpha_i y_i \;=:\; \zeta \quad (\text{constant}).
$$

So $\alpha_1$ is determined by $\alpha_2$, and the dual objective collapses to a one-variable quadratic in $\alpha_2$. Differentiating and setting to zero gives the **unconstrained update**

$$
\alpha_2^{\text{new, unc}} \;=\; \alpha_2^{\text{old}} \;+\; \frac{y_2(E_1 - E_2)}{\eta},
$$

where

$$
E_i \;=\; f(x_i) - y_i, \qquad
\eta \;=\; K(x_1, x_1) + K(x_2, x_2) - 2K(x_1, x_2) \;\ge\; 0.
$$

$\eta$ is the squared distance $\lVert \phi(x_1) - \phi(x_2) \rVert^2$ in feature space, hence non-negative.

### 4.3 Clipping to $[L, H]$

The pair must respect $0 \le \alpha_1, \alpha_2 \le C$ *and* the equality constraint. Combining them restricts $\alpha_2$ to an interval $[L, H]$ that depends on the sign agreement of the two labels:

$$
\begin{cases}
y_1 \neq y_2: & L = \max(0,\, \alpha_2^{\text{old}} - \alpha_1^{\text{old}}), \quad H = \min(C,\, C + \alpha_2^{\text{old}} - \alpha_1^{\text{old}}). \\[2pt]
y_1 = y_2: & L = \max(0,\, \alpha_1^{\text{old}} + \alpha_2^{\text{old}} - C), \quad H = \min(C,\, \alpha_1^{\text{old}} + \alpha_2^{\text{old}}).
\end{cases}
$$

Clip and back-substitute:

$$
\alpha_2^{\text{new}} \;=\; \operatorname{clip}(\alpha_2^{\text{new, unc}},\, L,\, H), \qquad
\alpha_1^{\text{new}} \;=\; \alpha_1^{\text{old}} + y_1 y_2\,(\alpha_2^{\text{old}} - \alpha_2^{\text{new}}).
$$

![One step of SMO in the $(\alpha_1, \alpha_2)$ plane: the feasibility line, the box $[0,C]^2$, the unconstrained optimum and the clipped result](./08-Support-Vector-Machines/fig7_smo_step.png)

### 4.4 Heuristics and convergence

- **Outer loop:** alternate between (a) sweeping all examples and (b) sweeping non-bound SVs only ($0 < \alpha_i < C$), since bound coordinates rarely move.
- **First variable:** any $\alpha_i$ that violates the KKT conditions by more than a tolerance.
- **Second variable:** the one that maximises $\lvert E_1 - E_2 \rvert$, which approximates the largest-step heuristic.

Each step strictly improves the dual objective unless the chosen pair is already at the optimum. Combined with bounded objective and finite step set, SMO converges. In practice it runs orders of magnitude faster than off-the-shelf QP, and updating the error cache $E_i$ incrementally is the engineering detail that keeps each step at $O(N)$ instead of $O(N^2)$.

---

## 5. Practical notes

- **Always standardise features.** SVM uses Euclidean geometry; mismatched scales let one coordinate dominate the margin.
- **Choose $C$ and $\gamma$ together.** Their product loosely controls model capacity. Grid-search $C \in \{10^{-2}, \dots, 10^3\}$ and $\gamma \in \{10^{-3}, \dots, 10^1\}$ on a log scale with cross-validation.
- **For $N \gg 10^5$, prefer linear SVM (LIBLINEAR) or stochastic methods.** Kernel SVM is roughly $O(N^2)$ in training time.
- **Multi-class.** Standard SVM is binary. Wrap it as **One-vs-Rest** ($K$ models) or **One-vs-One** ($K(K-1)/2$ models with voting). scikit-learn's `SVC` defaults to OvO.
- **Probabilities.** SVM outputs distances, not calibrated probabilities. Use Platt scaling (logistic regression on decision values) or isotonic regression if you need probabilities.

---

## 6. Exercises

### Exercise 1 -- Geometric margin

**Problem.** Hyperplane $w = (3, 4)^\top$, $b = -1$. Point $x_0 = (1, 1)^\top$ with label $y = +1$. Compute the geometric margin.

**Solution.**

$$
\gamma = \frac{y\,(w^\top x_0 + b)}{\lVert w \rVert} = \frac{1 \cdot (3 + 4 - 1)}{\sqrt{9 + 16}} = \frac{6}{5} = 1.2.
$$

```python
import numpy as np
w, b, x, y = np.array([3, 4]), -1, np.array([1, 1]), 1
print(y * (w @ x + b) / np.linalg.norm(w))   # 1.2
```

### Exercise 2 -- Closure of kernels under sum

**Problem.** Show that if $K_1, K_2$ are valid kernels, so is $K = K_1 + K_2$.

**Solution.** For any finite $\{x_i\}$ and any $c \in \mathbb{R}^N$,

$$
\sum_{i, j} c_i c_j K(x_i, x_j) = \sum_{i, j} c_i c_j K_1(x_i, x_j) + \sum_{i, j} c_i c_j K_2(x_i, x_j) \ge 0,
$$

since both terms are $\ge 0$ by Mercer applied to $K_1$ and $K_2$. Hence $K$ is PSD and therefore a valid kernel.

### Exercise 3 -- Reading $C$ from the SV count

**Problem.** With 100 training samples: $C = 0.1$ yields 30 SVs; $C = 100$ yields 15 SVs. Explain.

**Solution.** Smaller $C$ pays slack cheaply, so the optimiser widens the margin band; many more points then fall inside it and become $\alpha = C$ bound SVs. Larger $C$ punishes slack heavily, the margin shrinks, fewer points sit inside, fewer SVs. The general rule: more regularisation (smaller $C$) -> wider margin -> more SVs.

### Exercise 4 -- Diagnosing $\gamma$

**Problem.** With RBF: $\gamma = 0.01$ gives train 95 % / test 60 %; $\gamma = 100$ gives train 100 % / test 70 %. Which is better and what should you try?

**Solution.** Both are bad. $\gamma = 0.01$ has wide bumps and underfits weakly (train decent, test poor). $\gamma = 100$ has needle-thin bumps memorising every training point and severely overfits. Cross-validate over $\gamma \in \{0.1, 0.3, 1, 3, 10, 30\}$ jointly with $C$; the best pair will close the train/test gap.

### Exercise 5 -- SVM vs logistic regression

**Problem.** When prefer one over the other?

**Solution.**

| Need | Better choice | Why |
|---|---|---|
| Calibrated probabilities | logistic regression | SVM gives signed distances |
| Small data, nonlinear boundary | RBF SVM | kernel trick, sparse model |
| Millions of samples | linear logistic / linear SVM (SGD) | kernel SVM is $O(N^2)$-ish |
| Sparse feature selection | logistic + L1 | natural sparsity in $w$, not in samples |
| Robust to outliers far from boundary | SVM | hinge ignores deeply-correct examples |

---

## Q&A

### Why maximise the margin?

Geometrically, a wide margin tolerates noise: any perturbation smaller than the margin keeps the prediction unchanged. Statistically, margin-based bounds (e.g. via Rademacher complexity) show generalisation error decreases with the margin, so SVM is implicitly regularising model complexity.

### Why work with the dual?

Three reasons. (1) Data only enter as inner products, enabling kernels. (2) For $d \gg N$ the dual is much smaller than the primal. (3) The KKT conditions reveal sparsity: most $\alpha_i = 0$, so prediction is cheap.

### What if $\eta = 0$ in SMO?

$\eta = \lVert \phi(x_1) - \phi(x_2) \rVert^2$, so $\eta = 0$ means the two points coincide in feature space. The objective is then linear in $\alpha_2$ on the feasibility segment; the optimum is whichever endpoint $L$ or $H$ gives a larger objective.

### Does SVM extend to regression?

Yes -- **support vector regression** uses the $\varepsilon$-insensitive loss $\max(0, |y - f(x)| - \varepsilon)$. Errors smaller than $\varepsilon$ cost nothing, errors larger become support vectors. Same dual machinery, two multipliers per point ($\alpha_i, \alpha_i^\*$ for the upper and lower side).

### Why standardise inputs?

$\eta$ and the kernel matrix involve $\lVert x \rVert$. A feature with large units overwhelms the others, pulling the boundary toward axis alignment. Standardising (zero mean, unit variance per feature) is a one-line fix that often improves both speed and accuracy.

---

## References

- Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.
- Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.
- Platt, J. C. (1998). *Sequential Minimal Optimization: A Fast Algorithm for Training Support Vector Machines*. Tech. Rep. MSR-TR-98-14, Microsoft Research.
- Scholkopf, B., & Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 12.

---

<div class="series-nav">

**ML Mathematical Derivations Series**

[< Part 7: Decision Trees](/en/Machine-Learning-Mathematical-Derivations-7-Decision-Trees/) | **Part 8: Support Vector Machines** | [Part 9: Naive Bayes >](/en/Machine-Learning-Mathematical-Derivations-9-Naive-Bayes/)

</div>
