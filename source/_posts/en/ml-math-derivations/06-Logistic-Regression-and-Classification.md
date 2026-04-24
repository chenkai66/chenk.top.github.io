---
title: "Machine Learning Mathematical Derivations (6): Logistic Regression and Classification"
date: 2024-03-11 09:00:00
tags:
  - Machine Learning
  - Logistic Regression
  - Classification
  - Maximum Likelihood Estimation
  - Gradient Descent
  - Mathematical Derivation
categories: Machine Learning
series:
  name: "ML Mathematical Derivations"
  part: 6
  total: 20
lang: en
mathjax: true
description: "Complete derivation of logistic regression from sigmoid to softmax, cross-entropy loss, gradient computation, regularization, and multi-class extension with Python verification."
disableNunjucks: true
---

> **Hook.** Linear regression maps inputs to any real number — but what if the output has to be a probability between 0 and 1? Logistic regression solves this with one elegant trick: a sigmoid squashing function. Despite its name, logistic regression is a *classification* algorithm, and its math underpins every neuron in every modern neural network.

## What You Will Learn

- Why sigmoid is the natural way to turn a real-valued score into a probability, and why its derivative is so clean.
- How cross-entropy loss falls out of maximum likelihood estimation in two lines.
- Why cross-entropy beats MSE for classification — a vanishing-gradient argument made visible.
- The full gradient and Hessian for both binary and multi-class (softmax) cases, and why the loss is convex.
- L1, L2 and elastic-net regularization, and the Bayesian priors hiding behind them.
- Decision-boundary geometry and the threshold-free metrics (ROC / PR / AUC) that you actually need under class imbalance.

## Prerequisites

- Calculus: chain rule, partial derivatives.
- Linear algebra: matrix multiplication, transpose.
- Probability: Bernoulli and categorical distributions, likelihood.
- Familiarity with [Part 5: Linear Regression](/en/Machine-Learning-Mathematical-Derivations-5-Linear-Regression/).

---

## 1. From Linear Models to Probabilistic Classification

### 1.1 The Problem with Raw Linear Output

Linear regression gives us $\hat y = \mathbf{w}^\top \mathbf{x}$, which is unbounded. For classification, two things go wrong:

1. **Unconstrained range.** $\mathbf{w}^\top \mathbf{x} \in (-\infty, +\infty)$, but a class label lives in a finite set.
2. **No probability semantics.** "How sure are you this email is spam?" has no answer in $\mathbb{R}$.

The fix is a **link function** that squashes the linear score into $[0, 1]$. The canonical choice is the sigmoid.

### 1.2 The Sigmoid Function

![Sigmoid function with tangent at z=0 and its derivative](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/06-Logistic-Regression-and-Classification/fig1_sigmoid.png)

The sigmoid (logistic) function is

$$
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

Think of it as a "soft switch": it is essentially $0$ for very negative $z$, essentially $1$ for very positive $z$, and crosses $0.5$ at the origin. The picture above also shows the tangent at $z = 0$, whose slope is exactly $1/4$ — this is the *steepest* the sigmoid can ever be, a fact we will use over and over.

Three properties make the sigmoid mathematically delightful.

**Property 1 — Range.** For all $z \in \mathbb{R}$, $0 < \sigma(z) < 1$, so the output is a valid probability.

**Property 2 — Symmetry.** $\sigma(-z) = 1 - \sigma(z)$. This is what lets us write $P(y=0\mid \mathbf{x})$ in the same form as $P(y=1\mid \mathbf{x})$.

*Proof.*

$$
\sigma(-z) = \frac{1}{1 + e^{z}} = \frac{e^{-z}}{1 + e^{-z}} = 1 - \sigma(z). \quad\square
$$

**Property 3 — Self-expressing derivative.** $\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)$. This is the property that makes the cross-entropy gradient collapse to one line.

*Proof.*

$$
\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2}
= \frac{1}{1 + e^{-z}} \cdot \frac{e^{-z}}{1 + e^{-z}}
= \sigma(z)\bigl(1 - \sigma(z)\bigr). \quad\square
$$

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

z = np.linspace(-6, 6, 1000)
sig = sigmoid(z)

# Property 1 — range is (0, 1)
print(f"min={sig.min():.6f}  max={sig.max():.6f}")

# Property 2 — symmetry
print(f"symmetry err: {np.max(np.abs(sigmoid(-z) - (1 - sigmoid(z)))):.2e}")

# Property 3 — derivative against finite differences
num = np.gradient(sig, z)
ana = sig * (1 - sig)
print(f"derivative err: {np.max(np.abs(num - ana)):.2e}")
```

### 1.3 Logistic Regression Model

![Decision boundary on 2D classification data with probability contours](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/06-Logistic-Regression-and-Classification/fig2_decision_boundary.png)

For binary classification ($y \in \{0, 1\}$) we model

$$
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x}), \qquad
P(y = 0 \mid \mathbf{x}) = 1 - \sigma(\mathbf{w}^\top \mathbf{x}).
$$

A single compact form for both cases (Bernoulli pmf):

$$
P(y \mid \mathbf{x}) = \hat y^{\,y} (1 - \hat y)^{1 - y},
\qquad \hat y = \sigma(\mathbf{w}^\top \mathbf{x}).
$$

When $y = 1$ this returns $\hat y$; when $y = 0$ it returns $1 - \hat y$. The figure above shows what this model looks like geometrically: the boundary $\mathbf{w}^\top\mathbf{x} = 0$ is a hyperplane, and the orange arrow $\mathbf{w}$ is its normal — moving along $\mathbf{w}$ pushes the predicted probability from $0.5$ toward $1$.

---

## 2. Maximum Likelihood and the Cross-Entropy Loss

### 2.1 Building the Likelihood

For an i.i.d. training set $\{(\mathbf{x}_i, y_i)\}_{i=1}^N$, the joint likelihood is

$$
L(\mathbf{w}) = \prod_{i=1}^N P(y_i \mid \mathbf{x}_i; \mathbf{w})
              = \prod_{i=1}^N \hat y_i^{\,y_i}(1 - \hat y_i)^{1 - y_i}.
$$

We want the $\mathbf{w}$ that makes the observed labels most probable.

### 2.2 From Log-Likelihood to Cross-Entropy

Take the log (monotone, same optimum):

$$
\ell(\mathbf{w}) = \sum_{i=1}^N \bigl[\, y_i \ln \hat y_i + (1 - y_i)\ln(1 - \hat y_i) \,\bigr].
$$

Maximising $\ell$ is the same as minimising the **negative** average log-likelihood, also known as the **binary cross-entropy loss**:

$$
\boxed{\;\mathcal{L}(\mathbf{w}) = -\frac{1}{N} \sum_{i=1}^N \bigl[\, y_i \ln \hat y_i + (1 - y_i)\ln(1 - \hat y_i) \,\bigr].\;}
$$

**Information-theoretic view.** Cross-entropy $H(p, q) = -\sum_x p(x) \ln q(x)$ measures the extra bits needed to encode samples from $p$ using a code optimised for $q$. Here $p$ is the hard one-hot label and $q$ is the sigmoid output, so minimising $\mathcal{L}$ is literally pulling our predicted distribution toward the data distribution.

### 2.3 Why Not MSE?

![Cross-entropy vs MSE: loss curve and gradient magnitude when y=1](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/06-Logistic-Regression-and-Classification/fig3_loss_landscape.png)

Suppose we naively used mean squared error $\mathcal{L}_{\text{MSE}} = \tfrac12(\hat y - y)^2$. The gradient w.r.t. the logit $z = \mathbf{w}^\top\mathbf{x}$ is

$$
\frac{\partial \mathcal{L}_{\text{MSE}}}{\partial z} = (\hat y - y)\,\sigma'(z) = (\hat y - y)\,\hat y(1 - \hat y).
$$

The extra factor $\hat y(1-\hat y)$ is bounded by $1/4$ and **vanishes** when $\hat y$ is near $0$ or $1$. So if the model is *confidently wrong* ($\hat y \approx 0$ but $y = 1$), MSE produces almost no gradient and learning stalls.

Cross-entropy has no such factor:

$$
\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z} = \hat y - y.
$$

The right panel above makes this concrete: when $y = 1$ and the model predicts $\hat y \approx 0$, the CE gradient is near its maximum (push hard!) while the MSE gradient is essentially zero (give up).

```python
# Compare gradient magnitude wrt the logit z when y = 1
y_hat = np.linspace(1e-3, 1 - 1e-3, 500)
grad_mse = np.abs((y_hat - 1) * y_hat * (1 - y_hat))
grad_ce  = np.abs(y_hat - 1)
# CE dominates exactly where it matters: confidently wrong predictions.
```

---

## 3. Gradient Derivation and Optimisation

### 3.1 The Key Cancellation

For one sample, $\mathcal{L} = -\bigl[y \ln \hat y + (1 - y)\ln(1 - \hat y)\bigr]$ with $\hat y = \sigma(z)$, $z = \mathbf{w}^\top\mathbf{x}$. Chain rule:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}}
= \frac{\partial \mathcal{L}}{\partial \hat y} \cdot \frac{\partial \hat y}{\partial z} \cdot \frac{\partial z}{\partial \mathbf{w}}.
$$

- **Loss w.r.t. prediction:** $\dfrac{\partial \mathcal{L}}{\partial \hat y} = -\dfrac{y}{\hat y} + \dfrac{1 - y}{1 - \hat y}$.
- **Sigmoid derivative (Property 3):** $\dfrac{\partial \hat y}{\partial z} = \hat y(1 - \hat y)$.
- **Linear part:** $\dfrac{\partial z}{\partial \mathbf{w}} = \mathbf{x}$.

Multiply them:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{w}}
= \left(-\frac{y}{\hat y} + \frac{1 - y}{1 - \hat y}\right) \cdot \hat y(1 - \hat y) \cdot \mathbf{x}
= \bigl[-y(1 - \hat y) + (1 - y)\hat y\bigr]\mathbf{x}
= (\hat y - y)\,\mathbf{x}.
$$

The sigmoid derivative cancels with the $1/\hat y$ and $1/(1-\hat y)$ poles in $\partial \mathcal{L}/\partial \hat y$, leaving the famously clean result

$$
\boxed{\;\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = (\hat y - y)\,\mathbf{x}.\;}
$$

### 3.2 Full Batch Gradient

Averaging over $N$ samples and stacking:

$$
\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N}\sum_{i=1}^N (\hat y_i - y_i)\,\mathbf{x}_i = \frac{1}{N}\,\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y}),
$$

where $\mathbf{X} \in \mathbb{R}^{N \times d}$ is the data matrix.

### 3.3 Hessian and Convexity

Differentiating $(\hat y_i - y_i)\mathbf{x}_i$ once more:

$$
\nabla^2 \mathcal{L} = \frac{1}{N}\sum_{i=1}^N \hat y_i(1 - \hat y_i)\,\mathbf{x}_i \mathbf{x}_i^\top = \frac{1}{N}\,\mathbf{X}^\top \mathbf{S}\, \mathbf{X},
$$

with $\mathbf{S} = \operatorname{diag}\bigl(\hat y_i(1-\hat y_i)\bigr)$. For any vector $\mathbf{v} \neq \mathbf{0}$,

$$
\mathbf{v}^\top \nabla^2 \mathcal{L}\, \mathbf{v} = \frac{1}{N} \sum_i \hat y_i(1 - \hat y_i)(\mathbf{v}^\top \mathbf{x}_i)^2 \geq 0,
$$

so the Hessian is positive semi-definite and **the loss is convex**. There is a single global optimum and any reasonable optimiser will find it.

```python
# Numerical sanity check of the gradient formula
np.random.seed(42)
N, d = 50, 3
X = np.random.randn(N, d)
y = (sigmoid(X @ np.array([1.0, -0.5, 0.3])) > 0.5).astype(float)
w = np.zeros(d)

grad_ana = X.T @ (sigmoid(X @ w) - y) / N

eps, grad_num = 1e-5, np.zeros(d)
def loss(w_):
    p = sigmoid(X @ w_)
    return -np.mean(y*np.log(p+1e-15) + (1-y)*np.log(1-p+1e-15))
for j in range(d):
    e = np.zeros(d); e[j] = eps
    grad_num[j] = (loss(w + e) - loss(w - e)) / (2 * eps)

print(f"max diff: {np.max(np.abs(grad_ana - grad_num)):.2e}")
```

### 3.4 Optimisation Variants

- **Batch GD:** $\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \tfrac{1}{N}\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y})$. Stable, slow per epoch.
- **SGD:** $\mathbf{w} \leftarrow \mathbf{w} - \eta(\hat y_i - y_i)\mathbf{x}_i$ for one random $i$. Noisy, scales to massive data.
- **Mini-batch GD:** average the gradient over a batch of size $b$. The default in practice.
- **Newton / IRLS:** use $\nabla^2 \mathcal{L}$ for quadratic convergence — feasible because the Hessian is cheap and PSD.

---

## 4. Multi-Class Extension: Softmax Regression

### 4.1 From Binary to $K$ Classes

For $K \geq 3$ we learn one weight vector $\mathbf{w}_k$ per class. The class-$k$ score is $z_k = \mathbf{w}_k^\top \mathbf{x}$, and the softmax turns scores into a probability distribution:

$$
P(y = k \mid \mathbf{x}) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}.
$$

Softmax is a *soft argmax*: it exponentiates each score (so they are positive) and normalises (so they sum to one). The biggest score wins the most mass, but every class still gets a non-zero share.

![Softmax probability simplex for K=3 with sample logits](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/06-Logistic-Regression-and-Classification/fig4_softmax_simplex.png)

Geometrically, every softmax output is a point in the **probability simplex** — the triangle above for $K = 3$. Vertices are deterministic predictions; the centre is maximal uncertainty $(1/3, 1/3, 1/3)$; example logits are projected to show how concentrated mass corresponds to clearer decisions.

### 4.2 Cross-Entropy with One-Hot Labels

If the true class is $c$, encode it as a one-hot vector $\mathbf{t}$ with $t_k = \mathbb{1}[k = c]$. The loss is

$$
\mathcal{L} = -\sum_{k=1}^K t_k \ln P(y = k \mid \mathbf{x}) = -\ln P(y = c \mid \mathbf{x})
= -z_c + \ln \sum_{j=1}^K e^{z_j}.
$$

This is the multi-class **negative log-likelihood**.

### 4.3 The Softmax Gradient — Same Form Again

Differentiating $\mathcal{L}$ w.r.t. $z_k$:

- If $k = c$: $\dfrac{\partial \mathcal{L}}{\partial z_c} = -1 + P_c$.
- If $k \neq c$: $\dfrac{\partial \mathcal{L}}{\partial z_k} = P_k$.

Both cases combine into a single, beautiful expression:

$$
\boxed{\;\frac{\partial \mathcal{L}}{\partial z_k} = P_k - t_k.\;}
$$

This is structurally identical to the binary case — predicted probability minus true label. Pulling weights through $z_k = \mathbf{w}_k^\top \mathbf{x}$:

$$
\nabla_{\mathbf{W}} \mathcal{L} = \frac{1}{N}\,\mathbf{X}^\top(\hat{\mathbf{Y}} - \mathbf{T}),
$$

where $\mathbf{W} \in \mathbb{R}^{d \times K}$ stacks the per-class weight vectors.

```python
def stable_softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

K, c = 4, 2
z = np.random.randn(K)
t = np.eye(K)[c]
grad_ana = stable_softmax(z) - t

eps, grad_num = 1e-5, np.zeros(K)
for k in range(K):
    e = np.zeros(K); e[k] = eps
    grad_num[k] = (-np.log(stable_softmax(z + e)[c])
                   + np.log(stable_softmax(z - e)[c])) / (2 * eps)

print(f"max diff: {np.max(np.abs(grad_ana - grad_num)):.2e}")
```

---

## 5. Regularisation

### 5.1 L2 (Ridge)

$$
\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2}\|\mathbf{w}\|_2^2.
$$

The gradient picks up a $\lambda \mathbf{w}$ term, and the SGD update becomes a *weight-decay* update:

$$
\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \frac{\eta}{N}\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y}).
$$

**Bayesian view.** L2 is MAP estimation under a Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tfrac{1}{\lambda}\mathbf{I})$.

### 5.2 L1 (Lasso)

$$
\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda \|\mathbf{w}\|_1.
$$

L1 is non-differentiable at zero; using the subgradient $\partial_{w_j}\|\mathbf{w}\|_1 = \operatorname{sign}(w_j)$ gives the proximal / soft-thresholding update. The penalty's sharp corner at the origin pushes many coefficients **exactly** to zero, producing automatic feature selection.

**Bayesian view.** L1 corresponds to a Laplace prior $p(w_j) \propto e^{-\lambda |w_j|}$, whose peak at zero is the source of sparsity.

### 5.3 Elastic Net

$$
\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda_1 \|\mathbf{w}\|_1 + \frac{\lambda_2}{2}\|\mathbf{w}\|_2^2.
$$

Combines L1's sparsity with L2's stability when features are correlated.

---

## 6. Decision Boundary and Geometry

### 6.1 Binary Boundary

Logistic regression predicts class 1 when $\hat y \geq 0.5$, i.e. when $\mathbf{w}^\top \mathbf{x} \geq 0$. So the decision boundary is the hyperplane

$$
\mathbf{w}^\top \mathbf{x} + b = 0.
$$

The signed distance from a point $\mathbf{x}_0$ to the boundary is

$$
d = \frac{\mathbf{w}^\top \mathbf{x}_0 + b}{\|\mathbf{w}\|},
$$

and $|d|$ is exactly how confidently the model classifies $\mathbf{x}_0$.

![Logistic regression as a linear classifier with weight vector and signed distance](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/06-Logistic-Regression-and-Classification/fig7_feature_space.png)

The figure makes three things explicit:
1. The boundary is a hyperplane (a line in 2D).
2. The weight vector $\mathbf{w}$ is the **normal** to that hyperplane.
3. The norm $\|\mathbf{w}\|$ controls the *steepness* of the probability transition: a larger $\|\mathbf{w}\|$ collapses the $\hat y \approx 0.27 \to 0.73$ band into a thin strip.

### 6.2 Multi-Class Regions

For softmax regression, the decision rule "argmax over $z_k$" partitions feature space into $K$ convex regions. The boundary between class $j$ and class $k$ is the hyperplane

$$
(\mathbf{w}_j - \mathbf{w}_k)^\top \mathbf{x} + (b_j - b_k) = 0,
$$

so all pairwise boundaries are still linear.

---

## 7. Model Evaluation

### 7.1 Confusion Matrix and Headline Metrics

For binary classification:

|                   | Predicted Positive | Predicted Negative |
|-------------------|--------------------|--------------------|
| Actually Positive | TP                 | FN                 |
| Actually Negative | FP                 | TN                 |

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \quad
\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}, \quad
F_1 = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}.
$$

**Read precision as** "of those I flagged, how many were real?" and **recall as** "of the real ones, how many did I catch?" Optimising one without the other is almost always wrong.

### 7.2 Class Imbalance: Why Accuracy Lies

![Confusion matrices for an imbalanced problem: naive vs trained model](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/06-Logistic-Regression-and-Classification/fig6_confusion_imbalance.png)

When the positive class is rare (95% negative, 5% positive), a *trivial* "always negative" classifier scores 95% accuracy and is completely useless: it never finds a single positive case. The trained model in the right panel sacrifices a little accuracy to actually solve the problem — its $F_1$ is dramatically higher even though its accuracy is lower. Lesson: **always pair accuracy with precision, recall, and $F_1$ when classes are imbalanced.**

### 7.3 ROC, PR, and AUC

Sweeping the threshold $\tau$ ("predict positive when $\hat y \geq \tau$") traces out two curves:

- **ROC**: TPR ($= \text{Recall}$) vs FPR ($= FP / (FP + TN)$).
- **PR**: precision vs recall.

![ROC and PR curves with shaded AUC region](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/06-Logistic-Regression-and-Classification/fig5_roc_pr.png)

**AUC** = area under the ROC curve. AUC = 1 is perfect ranking, AUC = 0.5 is random. There is a clean probabilistic reading: AUC equals the probability that a uniformly drawn positive scores higher than a uniformly drawn negative.

When positives are very rare, the **PR curve and average precision (AP)** are usually more informative than ROC, because the FPR denominator $FP + TN$ is dominated by the (huge) negative class and masks model differences.

---

## 8. Implementation

### 8.1 Numerically Stable Sigmoid

When $z$ is very negative, $e^{-z}$ overflows. Use a branched form:

```python
def stable_sigmoid(z):
    return np.where(
        z >= 0,
        1 / (1 + np.exp(-z)),
        np.exp(z) / (1 + np.exp(z)),
    )
```

### 8.2 Numerically Stable Softmax

Direct $e^{z_k}$ overflows for large logits. Use the shift-invariance $\operatorname{softmax}(z) = \operatorname{softmax}(z - \max_j z_j)$:

```python
def stable_softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)
```

### 8.3 Complete Binary Classifier

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 regularization='l2', lambda_reg=0.01):
        self.lr, self.n_iter = learning_rate, n_iterations
        self.reg, self.lambda_reg = regularization, lambda_reg
        self.w = None

    def _sigmoid(self, z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def fit(self, X, y):
        N, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.n_iter):
            y_hat = self._sigmoid(X @ self.w)
            grad = X.T @ (y_hat - y) / N
            if self.reg == 'l2':
                grad += self.lambda_reg * self.w
            elif self.reg == 'l1':
                grad += self.lambda_reg * np.sign(self.w)
            self.w -= self.lr * grad

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

### 8.4 Complete Multi-Class Classifier

```python
class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000,
                 lambda_reg=0.01):
        self.lr, self.n_iter = learning_rate, n_iterations
        self.lambda_reg = lambda_reg
        self.W = None

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)
        e = np.exp(Z)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        N, d = X.shape
        K = int(y.max()) + 1
        self.W = np.zeros((d, K))
        T = np.zeros((N, K)); T[np.arange(N), y] = 1
        for _ in range(self.n_iter):
            Y_hat = self._softmax(X @ self.W)
            grad = X.T @ (Y_hat - T) / N + self.lambda_reg * self.W
            self.W -= self.lr * grad

    def predict_proba(self, X):
        return self._softmax(X @ self.W)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)
```

---

## 9. Exercises

### Exercise 1 — Sigmoid Properties

**Problem.** Prove $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ and $\sigma(-z) = 1 - \sigma(z)$.

*Solution.*

$$
\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2} = \sigma(z)\bigl(1-\sigma(z)\bigr),
\qquad
\sigma(-z) = \frac{1}{1+e^{z}} = 1 - \sigma(z). \quad\square
$$

### Exercise 2 — Cross-Entropy from MLE

**Problem.** Derive binary cross-entropy from maximum likelihood estimation.

*Solution.* Likelihood $L = \prod_i \hat y_i^{y_i}(1-\hat y_i)^{1-y_i}$. Take logs, negate, average:

$$
\mathcal{L} = -\frac{1}{N}\sum_i \bigl[y_i \ln \hat y_i + (1 - y_i)\ln(1 - \hat y_i)\bigr]. \quad\square
$$

### Exercise 3 — Softmax Gradient

**Problem.** Derive $\partial \mathcal{L} / \partial z_k$ for softmax cross-entropy.

*Solution.* With $P_k = e^{z_k}/\sum_j e^{z_j}$ and $\mathcal{L} = -\ln P_c$,

$$
\frac{\partial \mathcal{L}}{\partial z_k} = P_k - \mathbb{1}[k = c] = P_k - t_k,
$$

structurally identical to the binary gradient.

### Exercise 4 — Regularisation as a Bayesian Prior

**Problem.** What priors do L2 and L1 correspond to?

*Solution.* L2 ↔ Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \lambda^{-1}\mathbf{I})$, since $-\ln p(\mathbf{w}) \propto \tfrac{\lambda}{2}\|\mathbf{w}\|^2$. L1 ↔ Laplace prior $p(w_j) \propto e^{-\lambda|w_j|}$, whose sharp peak at $0$ is what produces sparse MAP solutions.

### Exercise 5 — Decision Boundary

**Problem.** Show that $\mathbf{w}^\top\mathbf{x} + b = 0$ is a hyperplane and explain the role of $\mathbf{w}$ and $\|\mathbf{w}\|$.

*Solution.* The set is an affine hyperplane with **normal vector** $\mathbf{w}$ and offset controlled by $b$. The direction of $\mathbf{w}$ orients the boundary; the magnitude $\|\mathbf{w}\|$ controls the *steepness* of the sigmoid transition (large $\|\mathbf{w}\|$ ⇒ sharp jump from $\hat y \approx 0$ to $\hat y \approx 1$ near the boundary). Signed distance from $\mathbf{x}_0$ to the boundary is $d = (\mathbf{w}^\top\mathbf{x}_0 + b)/\|\mathbf{w}\|$.

---

## Q&A

**Q: Why is it called "logistic regression" if it does classification?**
A: Historical accident. The logistic function was first used to *regress* a probability; classification was a later use case. The name stuck.

**Q: What is the essential difference from linear regression?**
A: Output space and likelihood. Linear regression assumes Gaussian noise on a continuous target (MSE = $-\ln$ Gaussian likelihood). Logistic regression assumes Bernoulli labels (CE = $-\ln$ Bernoulli likelihood). Both are special cases of **generalised linear models**, differing only in their link function and noise distribution.

**Q: Can logistic regression handle nonlinear boundaries?**
A: Not by itself — its boundary is a hyperplane in input space. But (a) polynomial features, (b) kernels, or (c) stacking inside a neural network give you arbitrarily nonlinear boundaries while keeping the cross-entropy loss intact.

**Q: Softmax vs. independent sigmoids?**
A: Softmax enforces $\sum_k P_k = 1$ — use it for **mutually exclusive** classes (single-label). Independent sigmoids let labels coexist — use them for **multi-label** problems (an image being both "outdoor" and "sunny").

**Q: How do I pick the regularisation strength $\lambda$?**
A: Cross-validation. Sweep $\lambda \in \{10^{-4}, 10^{-3}, \dots, 10^{2}\}$ and pick the one with the lowest validation loss (or best validation AUC for imbalanced problems).

**Q: Why is logistic regression convex?**
A: The Hessian $\nabla^2 \mathcal{L} = \tfrac{1}{N}\mathbf{X}^\top \mathbf{S}\,\mathbf{X}$ is PSD because each diagonal entry $\hat y_i(1 - \hat y_i) \in (0, 1/4]$ is non-negative. Convexity ⇒ any local minimum is global.

**Q: Logistic regression vs. SVM?**

| Aspect       | Logistic Regression | SVM                   |
|--------------|---------------------|-----------------------|
| Loss         | Cross-entropy       | Hinge loss            |
| Output       | Calibrated probability | Decision value     |
| Sparsity     | Every sample contributes a gradient | Only support vectors do |
| Kernel trick | Needs adaptation    | Native                |
| Best for     | Probability estimates, downstream calibration | Hard classification, complex nonlinear boundaries |

---

## References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. Chapter 4.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer. Chapter 4.
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. Chapter 8.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press. Chapter 5.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

---

<div class="series-nav">

**ML Mathematical Derivations Series**

[< Part 5: Linear Regression](/en/Machine-Learning-Mathematical-Derivations-5-Linear-Regression/) | **Part 6: Logistic Regression** | [Part 7: Decision Trees >](/en/Machine-Learning-Mathematical-Derivations-7-Decision-Trees/)

</div>
