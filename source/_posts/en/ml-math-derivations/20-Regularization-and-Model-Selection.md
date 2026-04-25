---
title: "ML Math Derivations (20): Regularization and Model Selection"
date: 2026-02-08 09:00:00
tags:
  - Machine Learning
  - Regularization
  - L1 Regularization
  - L2 Regularization
  - Dropout
  - Cross-Validation
  - VC Dimension
  - PAC Learning
  - Mathematical Derivations
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 20
  total: 20
lang: en
mathjax: true
description: "The series finale: from the bias-variance decomposition to L1/L2 geometry, dropout as a sub-network sampler, k-fold CV, AIC/BIC, VC bounds, and the modern double-descent phenomenon that broke classical theory."
disableNunjucks: true
series_order: 20
---

## What This Article Covers

A 100-million-parameter network trained on 50,000 images *should* overfit catastrophically. Modern deep networks generalise anyway. **Why?** Two ingredients: *regularisation* (techniques that constrain capacity) and *generalisation theory* (mathematics that says when learning works at all). This article is the closing chapter of the series, and we use it to gather every tool we have built — least squares, MAP estimation, optimisation, EM, neural networks — and turn them on the deepest open question in the field: *why does learning generalise?*

**Roadmap.**

1. The bias-variance decomposition and the generalisation gap.
2. L2 / L1 / elastic-net penalties — three views: optimisation, geometry, Bayesian prior.
3. Dropout as a $2^M$-network ensemble and as adaptive L2.
4. Early stopping = adaptive ridge with $\lambda_{\text{eff}} \propto 1/t$.
5. Model selection: K-fold CV, AIC, BIC.
6. VC dimension, PAC learning, and the modern double-descent puzzle.

**Prerequisites.** Calculus, probability (expectation, variance), linear algebra, gradient descent. Parts 1-4 of this series cover the math; Part 5 (linear regression), Part 6 (logistic regression), and Part 19 (neural networks) provide the models we will regularise.

---

## 1. Overfitting and the Bias-Variance Decomposition

### 1.1 Empirical vs Expected Risk

The training error and the *true* error you actually care about are not the same object:

$$
\hat{R}(f) = \frac{1}{N}\sum_{i=1}^{N} \ell(f(\mathbf{x}_i), y_i),
\qquad
R(f) = \mathbb{E}_{(\mathbf{x},y)\sim\mathcal D}\bigl[\ell(f(\mathbf{x}), y)\bigr]. \tag{1}
$$

The **generalisation gap** is $R(f) - \hat{R}(f)$. Overfitting means the gap is large; underfitting means even $\hat R$ is large.

### 1.2 Bias-Variance Decomposition

Pick a fixed test point $\mathbf{x}$, draw a fresh training set $S$, fit a regressor $f_S$, and look at the expected squared error. Define the average prediction $\bar f(\mathbf{x}) = \mathbb{E}_S[f_S(\mathbf{x})]$. Adding and subtracting $\bar f$:

$$
\mathbb{E}_S\bigl[(f_S(\mathbf{x}) - y)^2\bigr]
= \underbrace{(\bar f(\mathbf{x}) - f^\star(\mathbf{x}))^2}_{\text{Bias}^2}
+ \underbrace{\mathbb{E}_S\bigl[(f_S(\mathbf{x}) - \bar f(\mathbf{x}))^2\bigr]}_{\text{Variance}}
+ \underbrace{\sigma^2}_{\text{Noise}}. \tag{2}
$$

The cross terms vanish because $\mathbb{E}_S[f_S - \bar f] = 0$ and the noise is independent of $f_S$.

| Term | Driven by | High when... |
|------|-----------|--------------|
| Bias$^2$ | Misspecification | Model too simple (underfitting) |
| Variance | Estimation noise | Model too complex (overfitting) |
| $\sigma^2$ | Data | Always present (irreducible) |

Increasing capacity trades bias for variance. The optimum is where their derivatives meet.

![Bias-variance tradeoff: training, test, bias-squared, and variance vs model complexity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/20-Regularization-and-Model-Selection/fig4_complexity_curves.png)

The U-shape on the test curve is the entire story of classical model selection: every regulariser, every CV procedure, every information criterion is some way of locating that minimum without ever measuring $R(f)$ directly.

---

## 2. L2 Regularisation (Ridge Regression)

### 2.1 Three Equivalent Views

**Penalised loss.** Add $\tfrac{\lambda}{2}\|\mathbf{w}\|_2^2$ to the empirical risk. For squared loss this gives the closed form

$$
\hat{\mathbf{w}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}. \tag{3}
$$

The added $\lambda\mathbf{I}$ guarantees invertibility even when features are colinear.

**Constrained form.** By Lagrange duality, equivalently solve $\min \hat R(\mathbf{w})\ \text{s.t.}\ \|\mathbf{w}\|_2 \le t$. The feasible region is a Euclidean ball.

**Bayesian (MAP).** With Gaussian likelihood $y\mid\mathbf{w} \sim \mathcal N(\mathbf{x}^\top\mathbf{w}, \sigma^2)$ and Gaussian prior $\mathbf{w} \sim \mathcal N(\mathbf 0, \tau^2 \mathbf I)$,

$$
\hat{\mathbf{w}}_{\text{MAP}} = \arg\min_{\mathbf{w}}\Bigl[\hat R(\mathbf{w}) + \tfrac{\sigma^2}{2\tau^2}\|\mathbf{w}\|^2\Bigr],
\qquad \lambda = \tfrac{\sigma^2}{\tau^2}.
$$

Larger prior variance ($\tau^2 \uparrow$) means weaker regularisation.

### 2.2 SVD View: Shrinking Small Singular Directions

Decompose $\mathbf{X} = \mathbf{U}\boldsymbol\Sigma\mathbf{V}^\top$. Then

$$
\hat{\mathbf{w}}_{\text{ridge}} = \sum_j \frac{\sigma_j^2}{\sigma_j^2 + \lambda}\,\frac{\mathbf{u}_j^\top\mathbf{y}}{\sigma_j}\,\mathbf{v}_j. \tag{4}
$$

The shrinkage factor $\sigma_j^2/(\sigma_j^2+\lambda)$ is $\approx 1$ for large $\sigma_j$ (signal) and $\approx 0$ for small $\sigma_j$ (noise direction). Ridge is **soft principal-component truncation**.

### 2.3 Weight Decay

The gradient step is $\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \eta\nabla\hat R(\mathbf{w})$: each step starts by *shrinking* the weights. This is why deep-learning libraries call L2 "weight decay."

---

## 3. L1 Regularisation (Lasso) and Sparsity

### 3.1 The Geometry of Sparsity

Replace $\|\mathbf{w}\|_2^2$ with $\|\mathbf{w}\|_1 = \sum_j |w_j|$. The constraint region $\|\mathbf{w}\|_1 \le t$ is a diamond (in 2D) or a cross-polytope (in higher dim). Its **corners lie on the coordinate axes**, and the elliptical loss contours of the unregularised problem will generically meet the constraint at one of those corners — driving some $w_j$ to *exactly* zero.

![L1 vs L2 constraint regions: the diamond corner gives sparsity, the smooth ball gives a generic interior point](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/20-Regularization-and-Model-Selection/fig1_l1_vs_l2_geometry.png)

This is *not* a numerical artefact. The L1 sub-differential at zero is the interval $[-1, 1]$, so zero is a **stable** stationary point: small perturbations of the data do not move the optimal coefficient away from the axis. L2 has gradient $w_j$ at $w_j = 0$, so the coefficient *immediately* slides off the axis under any signal.

### 3.2 Soft-Thresholding

The proximal operator for $\lambda\|\mathbf{w}\|_1$ has a beautifully simple form, the **soft-threshold**:

$$
\hat w_j = \mathrm{sign}(w_j^\star)\,\max\bigl(|w_j^\star| - \lambda,\ 0\bigr). \tag{5}
$$

Coordinate descent applies (5) coordinate-by-coordinate; this is essentially how `glmnet` works.

### 3.3 Lasso as Embedded Feature Selection

Tracing the LASSO solution as $\lambda$ decreases from $\infty$ to $0$ yields the **regularisation path**. Relevant features switch on one at a time; irrelevant ones stay glued to zero.

![LASSO coefficient path: relevant features enter the model as lambda shrinks; irrelevant ones stay at zero](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/20-Regularization-and-Model-Selection/fig2_lasso_path.png)

The path itself is piecewise linear (LARS algorithm of Efron et al.) — a non-trivial geometric fact that follows from the polyhedral structure of the L1 ball.

### 3.4 Bayesian Reading and Elastic Net

L1 = MAP under a **Laplace prior** $p(w_j) \propto \exp(-|w_j|/b)$. The Laplace distribution has a *cusp* at zero, putting more prior mass exactly there than the Gaussian — hence sparsity.

When features are strongly correlated, pure LASSO behaves erratically (it picks one of a correlated group at random). The **elastic net** mixes both penalties:

$$
\mathcal L_{\text{enet}} = \hat R(\mathbf{w}) + \lambda_1\|\mathbf{w}\|_1 + \lambda_2\|\mathbf{w}\|_2^2,
$$

inheriting the sparsity of L1 and the grouping effect of L2.

---

## 4. Dropout

### 4.1 The Mechanism

At every training step, each hidden unit is independently zeroed with probability $p$ and the survivors are scaled by $1/(1-p)$:

$$
\tilde h_j = \frac{m_j}{1-p}\,h_j,\qquad m_j \sim \mathrm{Bernoulli}(1-p).
$$

The scaling keeps the expected activation unchanged: $\mathbb E[\tilde h_j] = h_j$. At test time we use the full network with no mask (this is *inverted dropout*, the modern convention).

### 4.2 Two Mathematical Stories

**Story 1 — Implicit ensemble.** A network with $M$ Bernoulli-droppable units defines $2^M$ thinned sub-networks that all share the same weights. Dropout SGD samples a uniformly random sub-network at every mini-batch. At test time, evaluating the full network with the inverted-dropout scaling approximates the geometric mean of the sub-network predictions — a free ensemble of exponentially many models.

![Dropout as random sub-network sampling: the full MLP plus three thinned samples sharing weights](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/20-Regularization-and-Model-Selection/fig6_dropout_concept.png)

**Story 2 — Adaptive L2.** Apply dropout to the *input* of a linear regressor. The expected loss is

$$
\mathbb E_{\mathbf m}\bigl[\|y - (\mathbf m \odot \mathbf x)^\top\mathbf w\|^2\bigr]
= \|y - \mathbf x^\top\mathbf w\|^2 + \frac{p}{1-p}\sum_j w_j^2 x_j^2. \tag{6}
$$

The extra term is an L2 penalty *weighted by feature variance*: large-magnitude inputs are regularised more aggressively. Wager, Wang & Liang (2013) extended this to GLMs and showed dropout is approximately *adaptive ridge*.

### 4.3 Variants

* **DropConnect.** Drop weights, not activations.
* **Spatial dropout.** For CNNs, drop entire feature maps (channels), not individual pixels — pixels are too correlated for elementwise dropout to do much.
* **Variational dropout.** RNNs use the *same* mask across all time steps so the recurrent dynamics remain coherent.

---

## 5. Early Stopping as Implicit Regularisation

### 5.1 The Strategy

Hold out a validation set. Stop when validation error has not improved for $P$ epochs (the *patience*). Return the best-so-far model.

### 5.2 Why It Equals Ridge (in the Quadratic Case)

For least squares with gradient descent from $\mathbf{w}_0 = \mathbf 0$, expand in the eigenbasis of $\mathbf{X}^\top\mathbf{X}$. After $t$ steps,

$$
\hat{\mathbf{w}}_t = \sum_j \bigl[1 - (1 - \eta\lambda_j)^t\bigr]\,\frac{\mathbf{u}_j^\top \mathbf{X}^\top\mathbf{y}}{\lambda_j}\,\mathbf{u}_j, \tag{7}
$$

while ridge gives $\sum_j \frac{\lambda_j}{\lambda_j + \alpha}\cdot(\cdot)$. Comparing the two shrinkage factors,

$$
1 - (1-\eta\lambda_j)^t \ \approx\ \frac{\lambda_j}{\lambda_j + 1/(\eta t)},
$$

so **early stopping at iteration $t$ is approximately ridge with $\alpha_{\text{eff}} \sim 1/(\eta t)$**: longer training $\Leftrightarrow$ weaker regularisation. Intuitively, gradient descent fits high-eigenvalue (low-frequency, "signal") directions first; the noise lives in the small-eigenvalue tail and only gets fitted late. Stop early, lose only the noise.

---

## 6. Model Selection: CV, AIC, BIC

### 6.1 K-Fold Cross-Validation

Partition the data into $K$ folds; for each $k$, train on the other $K-1$ and validate on fold $k$:

$$
\hat R_{\text{CV}} = \frac{1}{K}\sum_{k=1}^K \hat R_{\text{val}, k}. \tag{8}
$$

![5-fold cross-validation: each row is a fold, the coloured tile is the validation set, every sample validates exactly once](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/20-Regularization-and-Model-Selection/fig3_kfold_cv.png)

$K = 5$ or $10$ is the practical sweet spot (small bias, manageable variance, $K$ training runs). $K = N$ is **leave-one-out** — unbiased but expensive (and surprisingly *high-variance* in small samples). For time series, never shuffle; use blocked, expanding-window CV that respects causality.

### 6.2 Information Criteria

Both criteria penalise the negative log-likelihood by a function of the parameter count $p$:

$$
\mathrm{AIC} = -2\log p(\mathcal D \mid \hat{\mathbf w}) + 2p,
\qquad
\mathrm{BIC} = -2\log p(\mathcal D \mid \hat{\mathbf w}) + p\log N. \tag{9}
$$

For $N \ge 8$ we have $\log N > 2$, so BIC penalises complexity more strongly. Statistically:

* **BIC is consistent**: as $N \to \infty$, BIC selects the true model with probability one (when it is in the candidate set).
* **AIC is asymptotically efficient for prediction**: it minimises expected out-of-sample loss, but it does not necessarily recover the true model.

Use BIC when you trust your candidate set and want interpretability. Use AIC (or CV) when you only care about predictive accuracy.

![AIC, BIC and 5-fold CV scores selecting polynomial degree on a degree-3 truth, with the BIC fit overlaid on the data](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/20-Regularization-and-Model-Selection/fig5_aic_bic_vs_cv.png)

In small samples, the three criteria can disagree by 1-2 degrees. In large samples they typically converge. *When in doubt, use CV* — it is the only one that does not assume the model is correctly specified.

---

## 7. Generalisation Theory: VC Dimension, PAC, and Beyond

### 7.1 VC Dimension

A hypothesis class $\mathcal H$ **shatters** a set of $m$ points if it can realise all $2^m$ binary labelings. The **VC dimension** is the largest such $m$:

$$
\mathrm{VC}(\mathcal H) = \max\{m : \exists\, S\ \text{of size}\ m\ \text{shattered by}\ \mathcal H\}.
$$

For linear classifiers in $\mathbb R^d$, $\mathrm{VC} = d + 1$.

### 7.2 The VC Generalisation Bound

With probability $\ge 1 - \delta$, simultaneously for every $h \in \mathcal H$,

$$
R(h) \le \hat R(h) + O\!\left(\sqrt{\frac{\mathrm{VC}(\mathcal H)\log\!\bigl(N/\mathrm{VC}(\mathcal H)\bigr) + \log(1/\delta)}{N}}\right). \tag{10}
$$

Higher VC dimension demands more data; more data tightens the bound. This bound is **distribution-free** — it holds for *any* data distribution, which is also why it tends to be pessimistic.

### 7.3 PAC Learnability

A class is **PAC** (probably approximately correct) learnable if, for every $\epsilon, \delta > 0$, there is an algorithm that with probability $\ge 1 - \delta$ outputs a hypothesis with error $\le \epsilon$, using at most polynomially many samples. **Theorem (Blumer et al., 1989):** finite VC dimension $\Leftrightarrow$ PAC learnable.

### 7.4 The Deep-Learning Mystery and Double Descent

Modern neural networks have $p \gg N$. Their VC dimension is enormous (proportional to $p$), so (10) is *vacuous* — the bound exceeds 1. Yet they generalise. Why?

Several non-classical phenomena are at work:

* **Implicit regularisation by SGD.** SGD, especially with small batches and large learning rates, prefers flat minima of the loss landscape. Flat minima generalise better than sharp ones (a connection going back to Hochreiter & Schmidhuber, 1997).
* **Norm-based bounds.** Even though the *parameter count* is huge, the *norm* of the trained weights is small. PAC-Bayes and Rademacher bounds in terms of weight norms can be tight enough to be predictive.
* **Implicit minimum-norm interpolation.** For overparameterised linear models, gradient descent from zero converges to the *minimum-norm* interpolating solution — the same solution as ridge regression in the $\lambda \to 0$ limit.
* **Double descent.** As model capacity grows past the interpolation threshold $p = N$, the test error first follows the classical U, *spikes* at $p = N$ (the pseudo-inverse becomes ill-conditioned), then **decreases again** in the over-parameterised regime.

![Double descent: the classical U-shape, an interpolation peak at p = N, then a second descent in the modern regime](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/20-Regularization-and-Model-Selection/fig7_double_descent.png)

The second descent is real, robust, and breaks every classical intuition: for over-parameterised networks, *bigger really is better*. A satisfying theory of why this happens is one of the central open problems in machine learning.

---

## 8. The Practitioner's Cheat Sheet

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Train ≪ val | Overfitting | More data, stronger reg., dropout, early stop |
| Train ≈ val, both high | Underfitting | Bigger model, weaker reg., more features |
| Train good, val unstable | Small validation set | Use CV, average over seeds |
| Loss diverges | LR too large, no normalisation | Lower LR, add LayerNorm/BN, gradient clip |
| Sparse solution wanted | Many irrelevant features | LASSO or elastic net |
| Correlated features | Pure LASSO wobbles | Elastic net, group LASSO |

**Default starting recipe** for a moderate-size deep model: AdamW with weight decay $10^{-2}$, dropout $0.1$ on FC layers, cosine LR schedule, and *early stopping on a held-out 10%*. Tune via 5-fold CV on a small grid of learning rates and weight decays.

---

## 9. Exercises

**Exercise 1 (Ridge gradient).** Show $\nabla_{\mathbf w}\bigl[\tfrac12\|\mathbf y - \mathbf{X}\mathbf w\|^2 + \tfrac\lambda 2\|\mathbf w\|^2\bigr] = (\mathbf X^\top\mathbf X + \lambda\mathbf I)\mathbf w - \mathbf X^\top\mathbf y$ and recover (3).

**Exercise 2 (Sparsity proof).** Show that the optimal $w_j$ for $\min_w \tfrac12 (w - w^\star)^2 + \lambda |w|$ is the soft-threshold (5). *Hint:* differentiate the smooth part and use the sub-differential $\partial|w| = [-1,1]$ at $w = 0$.

**Exercise 3 (Dropout = ridge).** Verify (6) for a 1-d linear regressor by direct expectation over $m \sim \mathrm{Bernoulli}(1-p)$.

**Exercise 4 (Early stopping ↔ ridge).** From (7), set $\eta\lambda_j \ll 1$ and use $1 - (1-x)^t \approx 1 - e^{-tx}$ to derive $\alpha_{\text{eff}} \approx 1/(\eta t)$ in the small-eigenvalue regime.

**Exercise 5 (BIC threshold).** Solve $p\log N > 2p$ to confirm BIC penalises more than AIC for $N \ge 8$.

**Exercise 6 (VC of axis-aligned rectangles in $\mathbb R^2$).** Show it equals 4. *Hint:* place four points along the axes; prove no five points can be shattered.

---

## 10. Series Wrap-Up

This series began with calculus and probability and ends with the question that motivates the whole field: *why does empirical risk minimisation work?* Looking back across the twenty parts:

* **Parts 1-4 — Foundations.** Linear algebra, probability, and convex optimisation gave us the language.
* **Parts 5-9 — Classical supervised learning.** Linear / logistic regression, decision trees, SVMs, naive Bayes — the bread-and-butter models, each with its own clean derivation.
* **Parts 10-12 — Bayesian networks and ensembles.** From graphical models to XGBoost: how to combine weak structure into strong predictions.
* **Parts 13-15 — Latent variables.** EM, variational inference, HMMs — what to do when the data does not tell you everything.
* **Parts 16-18 — Beyond labels.** CRFs, dimensionality reduction, clustering — structured prediction and unsupervised learning.
* **Part 19 — Neural networks.** Backpropagation as the chain rule, and the reason everything since 2012 happened.
* **Part 20 — This article.** The meta-question: when does any of it *generalise*?

The honest answer to that meta-question, today, is: *we are still figuring it out*. Classical theory (VC, Rademacher) gives a lower-bound story that under-explains the modern over-parameterised regime. Newer ideas — implicit bias, flat minima, neural tangent kernels, PAC-Bayes, scaling laws — are pieces of a picture that has not yet snapped into place. If the next decade of theory is as productive as the last decade of practice, the sequel to this series will be a different book.

Until then: regularise, cross-validate, and trust the validation set.

---

## References

[1] Tikhonov, A. N. (1963). Solution of incorrectly formulated problems and the regularization method. *Soviet Math. Doklady*, 5, 1035-1038.

[2] Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS-B*, 58(1), 267-288.

[3] Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS-B*, 67(2), 301-320.

[4] Efron, B., Hastie, T., Johnstone, I., & Tibshirani, R. (2004). Least angle regression. *Annals of Statistics*, 32(2), 407-499.

[5] Srivastava, N., et al. (2014). Dropout: A simple way to prevent neural networks from overfitting. *JMLR*, 15(1), 1929-1958.

[6] Wager, S., Wang, S., & Liang, P. (2013). Dropout training as adaptive regularization. *NeurIPS*.

[7] Vapnik, V. N., & Chervonenkis, A. Y. (1971). On the uniform convergence of relative frequencies. *Theory Prob. & Appl.*, 16(2), 264-280.

[8] Blumer, A., Ehrenfeucht, A., Haussler, D., & Warmuth, M. K. (1989). Learnability and the Vapnik-Chervonenkis dimension. *JACM*, 36(4), 929-965.

[9] Hochreiter, S., & Schmidhuber, J. (1997). Flat minima. *Neural Computation*, 9(1), 1-42.

[10] Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). Understanding deep learning requires rethinking generalization. *ICLR*.

[11] Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine-learning practice and the classical bias-variance trade-off. *PNAS*, 116(32), 15849-15854.

[12] Nakkiran, P., Kaplun, G., Bansal, Y., Yang, T., Barak, B., & Sutskever, I. (2020). Deep double descent. *ICLR*.

---

*This is the final part of the [ML Mathematical Derivations](/tags/Mathematical-Derivations/) series. Previous: [Part 19 -- Neural Networks and Backpropagation](/en/Machine-Learning-Mathematical-Derivations-19-Neural-Networks-and-Backpropagation/). Start from the beginning: [Part 1 -- Introduction](/en/Machine-Learning-Mathematical-Derivations-1-Introduction-and-Mathematical-Foundations/).*
