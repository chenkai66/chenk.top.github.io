---
title: "Mathematical Derivation of Machine Learning (5): Linear Regression"
date: 2026-01-24 09:00:00
tags:
  - Machine Learning
  - Linear Regression
  - Least Squares
  - Gradient Descent
  - Mathematical Derivation
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 5
  total: 20
lang: en
mathjax: true
description: "A complete derivation of linear regression from three perspectives -- algebra (the normal equation), geometry (orthogonal projection), and probability (maximum likelihood) -- followed by Ridge, Lasso, gradient methods, and diagnostics, with every claim verified against scikit-learn."
disableNunjucks: true
series_order: 5
---

> **Hook.** In 1886 Francis Galton noticed something strange about heredity: children of unusually tall (or short) parents tended to be closer to the average than their parents were. He called this drift toward the mean *regression*, and the name stuck. The statistical curiosity grew up into the most consequential model in machine learning -- not because linear regression is powerful on its own, but because almost every other algorithm (logistic regression, neural networks, kernel methods) is some twist on the same idea: **fit a line, but in the right space.**

This chapter develops linear regression from three independent starting points -- algebra, geometry, and probability -- and shows that they all land on the same equation. Then we look at what happens when the assumptions break, and how Ridge, Lasso, and robust losses repair them.

## What This Article Covers

1. **Setup** -- how to write the model in matrix form so the math becomes one line.
2. **Algebra** -- minimize a quadratic, get the *normal equation* $w^\* = (X^\top X)^{-1} X^\top y$.
3. **Geometry** -- the same answer falls out of orthogonally projecting $y$ onto $\operatorname{Col}(X)$.
4. **Probability** -- under Gaussian noise, least squares is exactly maximum likelihood.
5. **Regularization** -- Ridge (L2) makes the system stable, Lasso (L1) makes it sparse.
6. **Optimization** -- when $X^\top X$ is too big to invert, gradient descent steps in.
7. **Diagnostics** -- residuals, multicollinearity, outliers, and how to fix each.

# Setup: Putting Linear Regression in Matrix Form

## Problem Statement

Given a training set $\{(x_i, y_i)\}_{i=1}^{m}$ where $x_i \in \mathbb{R}^d$ and $y_i \in \mathbb{R}$, we want a function

$$f(x) = w^\top x + b$$

whose predictions are as close as possible (in a sense we'll make precise) to the observed targets.

**Notation trick.** Carrying the bias $b$ around separately is annoying. We absorb it into the weights by appending a constant 1 to every input:

$$\tilde x = \begin{pmatrix} x \\\\ 1 \end{pmatrix}, \qquad \tilde w = \begin{pmatrix} w \\\\ b \end{pmatrix}, \qquad f(x) = \tilde w^\top \tilde x.$$

From now on we drop the tilde and just write $w^\top x$, with the understanding that the last component of $x$ is 1.

## Matrix Form

Stack everything:

$$X = \begin{bmatrix} x_1^\top \\\\ x_2^\top \\\\ \vdots \\\\ x_m^\top \end{bmatrix} \in \mathbb{R}^{m \times (d+1)}, \qquad y = \begin{bmatrix} y_1 \\\\ y_2 \\\\ \vdots \\\\ y_m \end{bmatrix} \in \mathbb{R}^{m}.$$

The vector of all predictions is then simply $\hat y = Xw$. Our job is to choose $w$ so that $\hat y$ is as close to $y$ as possible.

# Perspective 1 -- Algebra: The Normal Equation

## The Squared-Error Loss

Pick the squared-error loss:

$$L(w) = \tfrac{1}{2}\,\|y - Xw\|_2^2 = \tfrac{1}{2}\,(y - Xw)^\top (y - Xw).$$

The factor $\tfrac{1}{2}$ exists only to cancel a 2 when we differentiate; it changes nothing else.

The picture below shows what the loss is actually summing: the squared lengths of the vertical lines from each data point to the fitted line.

![Linear regression minimises the sum of squared vertical residuals](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/05-Linear-Regression/fig1_scatter_residuals.png)

## Computing the Gradient

Expand the loss:

$$L(w) = \tfrac{1}{2}\bigl(y^\top y - 2\,y^\top X w + w^\top X^\top X\,w\bigr).$$

Each term is easy to differentiate (matrix calculus identities $\nabla_w (a^\top w) = a$ and $\nabla_w (w^\top A w) = 2 A w$ for symmetric $A$):

$$\nabla_w L = -X^\top y + X^\top X\, w = X^\top (Xw - y).$$

## The Normal Equation

Setting $\nabla_w L = 0$:

$$\boxed{\,X^\top X\, w = X^\top y\,}.$$

This is the **normal equation** -- so named because, geometrically, it asserts that the residual is *normal* (perpendicular) to every column of $X$. We'll see that geometry in a moment.

**Theorem 1 (Closed-form least-squares solution).** *If $X^\top X$ is invertible, the unique minimiser of $L$ is*

$$w^\* = (X^\top X)^{-1} X^\top y.$$

*Proof.* The first-order condition $\nabla L = 0$ yields the normal equation. The Hessian is $\nabla^2 L = X^\top X$, which is positive definite whenever $X$ has full column rank: for any $v \neq 0$,

$$v^\top X^\top X\, v = \|Xv\|_2^2 > 0$$

(strict because full column rank means $Xv \neq 0$). A convex quadratic with positive-definite Hessian and zero gradient is at its global minimum. $\square$

**When does $X^\top X$ fail to be invertible?**

- $m < d+1$ -- fewer samples than features (the system is underdetermined),
- columns of $X$ are linearly dependent (perfect multicollinearity).

In both cases the *pseudoinverse* gives the minimum-norm solution: $w^\* = X^{+} y$, computed via SVD ($X = U \Sigma V^\top \Rightarrow X^+ = V \Sigma^+ U^\top$, where $\Sigma^+$ inverts the nonzero singular values and leaves zeros alone).

# Perspective 2 -- Geometry: Orthogonal Projection

## The Same Answer, A Different Story

Forget calculus for a moment. The columns of $X$ span a subspace $\operatorname{Col}(X) \subseteq \mathbb{R}^m$. The vector $\hat y = Xw$ lives somewhere inside that subspace, no matter what $w$ is. We want to find the point in $\operatorname{Col}(X)$ closest to $y$ (in Euclidean distance).

Geometry tells us that point exactly: it is the **orthogonal projection** of $y$ onto $\operatorname{Col}(X)$, and it is characterised by the residual $y - \hat y$ being perpendicular to every column of $X$:

$$X^\top (y - X w^\*) = 0 \quad\Longleftrightarrow\quad X^\top X\, w^\* = X^\top y.$$

That's the normal equation again -- this time derived without taking a single derivative.

![OLS as orthogonal projection of y onto the column space of X](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/05-Linear-Regression/fig2_projection_geometry.png)

The figure makes three things visible simultaneously: the columns of $X$ spanning a plane (purple/blue), the target vector $y$ floating above that plane, the projection $\hat y$ on the plane, and the residual $y - \hat y$ shooting straight up at a right angle to the plane. **Right angle = optimum.**

## The Projection Matrix

The projection itself is a linear map -- the **hat matrix**:

$$H = X(X^\top X)^{-1} X^\top, \qquad \hat y = H y.$$

It satisfies the two defining properties of an orthogonal projector:

- **Idempotent**: $H^2 = H$. Projecting twice does nothing extra.
- **Symmetric**: $H^\top = H$. Equivalent to "the projection is *orthogonal*."

The *residual maker* $M = I - H$ projects onto the orthogonal complement; $M y$ is the residual vector. It satisfies $MX = 0$, formalising "residuals are orthogonal to every feature."

# Perspective 3 -- Probability: Maximum Likelihood

## The Linear-Gaussian Model

Assume the data is generated by

$$y_i = w^\top x_i + \epsilon_i, \qquad \epsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2).$$

Equivalently, conditional on $x_i$, the target $y_i$ is Gaussian with mean $w^\top x_i$ and variance $\sigma^2$.

## Likelihood and Log-Likelihood

The joint likelihood factorises by independence:

$$p(y \mid X, w, \sigma^2) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi}\,\sigma}\,\exp\!\left(-\frac{(y_i - w^\top x_i)^2}{2\sigma^2}\right).$$

Take logs:

$$\log p = -\frac{m}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{m}(y_i - w^\top x_i)^2.$$

Maximising over $w$ (with $\sigma^2$ fixed) is the same as **minimising the sum of squared residuals**. So:

**Theorem 2.** *Under the Gaussian-noise model, the MLE for $w$ is exactly the OLS solution:* $\hat w_{\text{MLE}} = (X^\top X)^{-1} X^\top y$.

Differentiating with respect to $\sigma^2$ yields $\hat\sigma^2_{\text{MLE}} = \tfrac{1}{m}\|y - X\hat w\|_2^2$ -- the mean residual sum of squares. (This estimator is biased; the unbiased version divides by $m - d - 1$.)

## Bayesian Step: Where Ridge Comes From

Now place a prior $w \sim \mathcal{N}(0, \tau^2 I)$. The MAP estimate maximises the posterior, equivalently minimises

$$\sum_i (y_i - w^\top x_i)^2 + \frac{\sigma^2}{\tau^2}\|w\|_2^2.$$

That is *Ridge regression*, and we just identified its regularisation strength: $\lambda = \sigma^2 / \tau^2$. A weaker prior (large $\tau$) means smaller $\lambda$.

# Regularization: Ridge, Lasso, Elastic Net

## Why Regularize at All

When features are nearly collinear, $X^\top X$ becomes nearly singular and $(X^\top X)^{-1}$ contains huge entries. A tiny perturbation in $y$ then sends $\hat w$ flying. Regularization tames this by penalising large $w$:

$$L_{\text{reg}}(w) = \tfrac{1}{2}\|y - Xw\|_2^2 + \lambda \cdot \mathcal{P}(w).$$

## Ridge Regression (L2)

Take $\mathcal{P}(w) = \tfrac{1}{2}\|w\|_2^2$. The closed form drops out by the same gradient calculation as before:

$$\boxed{\;\hat w_{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y.\;}$$

The added $\lambda I$ shifts every eigenvalue of $X^\top X$ up by $\lambda$, so the matrix is *always* invertible for $\lambda > 0$ -- even when $X^\top X$ alone is singular. This is what people mean by "Ridge stabilises the inverse."

## Lasso Regression (L1)

Take $\mathcal{P}(w) = \|w\|_1$. There is no closed form -- $|\cdot|$ is not differentiable at zero -- but the optimisation is still convex and is solved efficiently by *coordinate descent* or *proximal gradient*. The remarkable property:

> **Lasso produces exactly-zero coefficients**, performing variable selection automatically.

The figure below shows the coefficient *paths* -- how each $w_j$ evolves as $\lambda$ slides from very small (left, equals OLS) to very large (right, every coefficient crushed to zero):

![Coefficient paths: Ridge shrinks smoothly, Lasso clips to zero](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/05-Linear-Regression/fig4_regularization_paths.png)

In Ridge, each curve approaches zero asymptotically -- it gets *small*, but never *zero*. In Lasso, curves **hit the axis and stay there**. That binary "in or out" behaviour is what makes Lasso a feature selector.

## Why Lasso is Sparse: A Geometric Argument

Rewrite both penalties in *constrained* form:

- Ridge: minimise $\|y - Xw\|^2$ subject to $\|w\|_2 \le t$.
- Lasso: minimise $\|y - Xw\|^2$ subject to $\|w\|_1 \le t$.

The level sets of the loss are ellipsoids around $\hat w_{\text{OLS}}$. The constraint regions are very different shapes:

![L1 vs L2 geometry: corners on axes give sparsity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/05-Linear-Regression/fig5_l1_vs_l2_geometry.png)

The L2 disk has a smooth boundary, so the loss ellipse generally touches it at an *interior* point of the boundary -- both coordinates non-zero. The L1 diamond has *corners on the axes*, and ellipses tilted in generic directions almost always touch a corner first -- giving sparse solutions. The higher the dimension, the more corners and edges the diamond grows, and the more aggressively Lasso zeroes things out.

## Elastic Net

A convex combination of both:

$$\mathcal{P}_{\text{EN}}(w) = \alpha \|w\|_1 + \tfrac{1-\alpha}{2}\|w\|_2^2.$$

Keeps the sparsity from L1 and the stability from L2, particularly useful when groups of correlated features would otherwise cause Lasso to pick one arbitrarily.

# Optimization: When Closed Form Isn't Practical

Computing $(X^\top X)^{-1}$ costs $\mathcal{O}(d^3)$ time and $\mathcal{O}(d^2)$ memory. For $d = 10^6$, that's a non-starter. Iterative methods are then the only option.

## Batch Gradient Descent (BGD)

$$w^{(t+1)} = w^{(t)} - \eta \cdot \frac{1}{m} X^\top (X w^{(t)} - y).$$

Per-iteration cost: $\mathcal{O}(md)$. Converges *linearly* for strongly convex $L$:

$$L(w^{(t)}) - L(w^\*) \le \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2t} \bigl(L(w^{(0)}) - L(w^\*)\bigr),$$

where $\kappa = \lambda_{\max}(X^\top X) / \lambda_{\min}(X^\top X)$ is the condition number. Ill-conditioned problems (large $\kappa$) crawl; this is why people standardise features before running gradient descent.

## Stochastic Gradient Descent (SGD)

Each step uses a single sample $(x_i, y_i)$:

$$w^{(t+1)} = w^{(t)} - \eta \cdot (w^{(t)\top} x_i - y_i)\, x_i.$$

- **Pros**: $\mathcal{O}(d)$ per step, supports streaming/online learning, mild noise can help escape bad regions in non-convex settings.
- **Cons**: noisy updates require a decreasing learning rate $\eta_t$ (e.g. $\eta_t = \eta_0 / (1 + t)$) for almost-sure convergence.

## Mini-batch GD

The pragmatic middle ground: average the gradient over a batch of $B \in [32, 512]$ samples. Combines vectorised speed with SGD's variance reduction. This is the version used in essentially every deep-learning library.

# Model Evaluation and Diagnostics

## Metrics

For predictions $\hat y_i$ on a test set:

- **MSE** $= \frac{1}{n}\sum (y_i - \hat y_i)^2$ -- penalises large errors quadratically.
- **RMSE** $= \sqrt{\text{MSE}}$ -- in the original units of $y$.
- **MAE** $= \frac{1}{n}\sum |y_i - \hat y_i|$ -- more robust to outliers.
- **$R^2$** $= 1 - \dfrac{\sum(y_i - \hat y_i)^2}{\sum(y_i - \bar y)^2}$ -- fraction of variance explained.

A subtle warning about $R^2$: it never decreases when you add features. Use **adjusted $R^2$** when comparing models of different sizes:

$$R^2_{\text{adj}} = 1 - (1 - R^2)\,\frac{n - 1}{n - d - 1}.$$

## Cross-Validation: Picking Model Capacity

There is no single "best" polynomial degree, no single "best" $\lambda$. Cross-validation finds the choice that generalises best by holding out folds of data:

![5-fold cross-validation reveals the bias-variance sweet spot](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/05-Linear-Regression/fig6_cv_curve.png)

The training error keeps falling as we add capacity (more polynomial terms, smaller $\lambda$). The CV error follows a U-shape: too little capacity is *biased* (underfit), too much is *variance-dominated* (overfit). The minimum of the CV curve is your model.

## Polynomial Regression: Bias and Variance Made Visible

Linear regression is "linear in the parameters", *not* "linear in the inputs". Adding polynomial features lets us fit curves while still solving a linear system. The risk is overfitting:

![Underfit vs right fit vs overfit on a sine curve](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/05-Linear-Regression/fig3_polynomial_fits.png)

A degree-1 model is too rigid to capture the sine wave (high bias). A degree-3 polynomial nails it. A degree-15 polynomial passes through almost every point but oscillates wildly between them -- a classic overfit. Compare the *training* MSE in each panel: it monotonically decreases. The training error tells you nothing about overfitting; only held-out data does.

# Outlier Sensitivity and Robust Alternatives

Squared loss has a known weakness: a single outlier with a large residual contributes its squared error to the total, so the model bends to accommodate it. The picture is striking:

![OLS gets dragged by outliers; Huber regression resists](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/05-Linear-Regression/fig7_outlier_robustness.png)

The dashed grey line is the truth. OLS (purple) tilts visibly toward the three outliers. The **Huber loss** -- quadratic for small residuals, linear for large ones -- treats outliers gently:

$$\rho_\delta(r) = \begin{cases} \tfrac{1}{2} r^2, & |r| \le \delta, \\\\ \delta\,(|r| - \tfrac{1}{2}\delta), & |r| > \delta. \end{cases}$$

The Huber fit (green) almost overlaps the truth despite the same outliers. Other options exist (RANSAC, M-estimators, quantile regression), but Huber is the workhorse: differentiable everywhere, convex, robust.

# Q&A: Common Sticking Points

### Q1: Why squared loss instead of absolute loss?

| Loss | Differentiable? | Closed form? | Robust to outliers? | Implied noise |
|---|---|---|---|---|
| Squared (L2) | everywhere | yes | weak | Gaussian |
| Absolute (L1) | not at 0 | no (LP) | strong | Laplace |
| Huber | everywhere | no | medium | mixed |

Three reasons squared loss dominates: (i) it's smooth, (ii) it has a closed-form solution, (iii) it corresponds to MLE under the most common noise model. Use Huber when robustness matters.

### Q2: Normal equation or gradient descent?

**Use the normal equation when** $d \lesssim 10^4$ and $X^\top X$ fits in memory -- it's exact and one-shot.

**Use gradient descent when** $d$ is large, when you need online updates, or when you've added a non-smooth penalty (Lasso) that has no closed form.

### Q3: Why does Ridge always have a solution?

Because $X^\top X + \lambda I$ is positive definite for any $\lambda > 0$:

$$v^\top (X^\top X + \lambda I) v = \|Xv\|^2 + \lambda\|v\|^2 \ge \lambda\|v\|^2 > 0 \quad (v \ne 0).$$

Even if $X v = 0$, the $\lambda \|v\|^2$ term keeps the form strictly positive. Geometrically: Ridge says "in directions the data can't pin down, prefer 0."

### Q4: How do I pick $\lambda$?

Cross-validation. Almost always cross-validation. `RidgeCV` and `LassoCV` in scikit-learn do this in one line. Search log-uniformly in $[10^{-3}, 10^{3}]$, then refine.

### Q5: Why standardise features?

Two reasons:

1. **Conditioning.** Different scales make $X^\top X$ ill-conditioned (huge $\kappa$), which slows gradient descent dramatically.
2. **Fair regularization.** $\|w\|_2^2$ penalises every coordinate the same way; a feature measured in millimetres has weight 1000x larger than the same feature in metres, and Ridge would crush it harder for no good reason.

Use `StandardScaler`: subtract the (training) mean, divide by the (training) std.

### Q6: How does multicollinearity affect things?

Two correlated features $x_1 \approx x_2$ make $X^\top X$ nearly singular. Symptoms: huge standard errors on $\hat w_1$ and $\hat w_2$, signs that flip on tiny perturbations, but the *predictions* $\hat y = X \hat w$ remain accurate.

**Diagnose** with the Variance Inflation Factor: $\text{VIF}_j = 1 / (1 - R_j^2)$ where $R_j^2$ is from regressing feature $j$ on the others. Rule of thumb: VIF $> 10$ is severe.

**Fix** with Ridge regression, by dropping redundant features, or by replacing them with PCA components.

### Q7: What are the classical assumptions and what breaks if they fail?

| Assumption | Diagnostic | Fix when violated |
|---|---|---|
| Linearity in $w$ | residual plot has no pattern | polynomial features, basis expansion, kernel methods |
| Independent errors | Durbin–Watson $\approx 2$ | autoregressive models, GLS |
| Homoscedasticity | residuals uniform across $\hat y$ | weighted LS, log-transform $y$ |
| Normal errors | Q–Q plot near diagonal | Box–Cox, GLM, robust loss |

For inference (p-values, confidence intervals) all four matter. For *prediction* you can usually get away with violating normality.

### Q8: How do I encode categorical features?

Never as integers (that fakes an ordering). Use **one-hot encoding** (`pd.get_dummies` or `OneHotEncoder`). For a $k$-level category, create $k$ indicators -- and drop one to avoid the multicollinearity caused by the columns summing to 1 (`drop='first'`).

### Q9: Can linear regression model nonlinear relationships?

Yes -- *linear* refers to the parameters, not the inputs. Replace $x$ with $\phi(x)$ (polynomial, log, sin, ReLU, you name it), then fit a linear model in $\phi$. This is exactly what kernel methods and neural networks generalise.

### Q10: How do I interpret coefficients?

After standardising features, $\hat w_j$ is the change in $\hat y$ (in standard deviations) per one-standard-deviation increase in $x_j$, *holding the other features fixed*. Two large warnings:

- **Multicollinearity** makes individual $\hat w_j$ unstable even when predictions are fine -- don't read a single $\hat w_j$ in isolation.
- **Correlation is not causation.** A regression coefficient is a conditional association in your data, full stop. Causal claims need experimental design or special identification strategies.

# Verifying the Code

Every figure in this article was generated by `scripts/figures/ml-math-derivations/05-linear-regression.py`, which cross-checks each claim against scikit-learn (e.g. the manually computed slope and intercept in fig1 are asserted to match `LinearRegression`'s fit exactly, the projection in fig2 verifies orthogonality of the residual numerically). The file is the single source of truth for the visuals; reproducing them takes one command.

# Summary

We approached linear regression from three independent directions and arrived at the same equation:

- **Algebra.** Minimise a quadratic. Set the gradient to zero. Get $w^\* = (X^\top X)^{-1} X^\top y$.
- **Geometry.** Project $y$ orthogonally onto $\operatorname{Col}(X)$. The condition that the residual is perpendicular to the columns *is* the normal equation.
- **Probability.** Under Gaussian noise, MLE for $w$ minimises squared residuals. Under a Gaussian prior on $w$, MAP gives Ridge.

Then we addressed three failures of vanilla OLS: instability under collinearity (Ridge), failure to select features (Lasso), and oversensitivity to outliers (Huber). The optimisation toolkit -- BGD, SGD, mini-batch -- handles the case where $X^\top X$ is too big to invert. And cross-validation tells us, empirically, what model capacity to choose.

**Next chapter.** We'll generalise to *classification* -- the output space becomes discrete. The sigmoid function, cross-entropy loss, and the geometric meaning of decision boundaries all fall out of asking "what's the linear-Gaussian model for binary outputs?"

# References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.
- Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *JRSS-B*, 58(1), 267–288.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *JRSS-B*, 67(2), 301–320.
- Huber, P. J. (1964). Robust estimation of a location parameter. *Annals of Math. Stat.*, 35(1), 73–101.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Shalev-Shwartz, S., & Ben-David, S. (2014). *Understanding Machine Learning*. Cambridge University Press.
