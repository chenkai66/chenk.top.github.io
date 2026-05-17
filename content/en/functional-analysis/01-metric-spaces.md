---
title: "Metric Spaces: Distance, Convergence, and Completeness"
date: 2021-10-01 09:00:00
tags:
  - functional-analysis
  - metric-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 1
series_total: 12
translationKey: "functional-analysis-1"
description: "From the real line to infinite-dimensional function spaces: why completeness is the dividing line."
---

# Metric Spaces: Distance, Convergence, and Completeness

## Why Infinite-Dimensional Spaces Need New Tools

Infinite-dimensional spaces, such as function spaces, arise naturally in many areas of mathematics and its applications. These spaces are more complex than their finite-dimensional counterparts, and the tools and techniques used to study them need to be adapted accordingly. In this section, we will explore why infinite-dimensional spaces require new tools and how metric spaces provide a suitable framework for their study.

### Motivation

![Unit balls in l1, l2, and l-infinity norms](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/01-metric-spaces/fa_fig1_unit_balls.png)


Consider the space of all continuous functions on the interval $[a, b]$, denoted by $C[a, b]$. This space is infinite-dimensional because it contains an infinite number of linearly independent functions. For example, the set $\{1, x, x^2, x^3, \ldots\}$ is linearly independent in $C[a, b]$.

In finite-dimensional spaces, such as $\mathbb{R}^n$, we can use the Euclidean norm to measure distances and define convergence. However, in infinite-dimensional spaces, the Euclidean norm is not always well-defined or practical. For instance, in $C[a, b]$, the Euclidean norm would involve summing an infinite series, which may not converge. Therefore, we need a more general notion of distance that can be applied to both finite and infinite-dimensional spaces.

### Definitions and Theorems

A **metric space** is a set $X$ together with a function $d: X \times X \to \mathbb{R}$, called a **metric**, that satisfies the following properties for all $x, y, z \in X$:
1. **Non-negativity**: $d(x, y) \geq 0$, and $d(x, y) = 0$ if and only if $x = y$.
2. **Symmetry**: $d(x, y) = d(y, x)$.
3. **Triangle Inequality**: $d(x, z) \leq d(x, y) + d(y, z)$.

These properties ensure that $d$ behaves like a distance function, allowing us to measure the "distance" between any two points in $X$.

### Proof Sketches and Examples

To illustrate the need for metric spaces, consider the space $C[a, b]$ with the **supremum norm** (or **uniform norm**):
$$
\|f\|_\infty = \sup_{x \in [a, b]} |f(x)|.
$$
This norm induces a metric on $C[a, b]$ given by:
$$
d(f, g) = \|f - g\|_\infty = \sup_{x \in [a, b]} |f(x) - g(x)|.
$$
The supremum norm is well-defined and finite for all continuous functions on $[a, b]$, making it a suitable choice for measuring distances in $C[a, b]$.

### Worked Example

Let $f(x) = x$ and $g(x) = x^2$ on the interval $[0, 1]$. Compute the distance between $f$ and $g$ using the supremum norm.

**Solution:**
\[
\|f - g\|_\infty = \sup_{x \in [0, 1]} |x - x^2|.
\]
To find the supremum, we first find the maximum value of the function $h(x) = x - x^2$ on $[0, 1]$. The derivative of $h(x)$ is:
\[
h'(x) = 1 - 2x.
\]
Setting $h'(x) = 0$ gives $x = \frac{1}{2}$. Evaluating $h(x)$ at the critical point and the endpoints of the interval, we get:
\[
h(0) = 0, \quad h(1) = 0, \quad h\left(\frac{1}{2}\right) = \frac{1}{4}.
\]
Thus, the maximum value of $|x - x^2|$ on $[0, 1]$ is $\frac{1}{4}$. Therefore,
\[
\|f - g\|_\infty = \frac{1}{4}.
\]

In summary, the need for new tools in infinite-dimensional spaces arises from the limitations of the Euclidean norm. Metric spaces provide a flexible and powerful framework for studying these spaces, allowing us to define and analyze concepts such as distance, convergence, and continuity in a general setting.

## Metric Spaces: Definition and Examples

In the previous section, we introduced the concept of a metric space and discussed why it is necessary for studying infinite-dimensional spaces. In this section, we will provide a formal definition of metric spaces and explore several important examples, including $\mathbb{R}^n$, discrete spaces, $C[a, b]$, and $l^p$ spaces.

### Motivation

Metric spaces are fundamental in analysis and topology because they allow us to generalize the notion of distance and convergence. By understanding different types of metric spaces, we can apply these concepts to a wide range of mathematical objects and problems.

### Definitions and Theorems

A **metric space** is a pair $(X, d)$, where $X$ is a set and $d: X \times X \to \mathbb{R}$ is a metric, satisfying the properties of non-negativity, symmetry, and the triangle inequality.

#### Examples of Metric Spaces

1. **Euclidean Space $\mathbb{R}^n$**:
   - **Set**: $X = \mathbb{R}^n$
   - **Metric**: The Euclidean metric is given by
     \[
     d(x, y) = \sqrt{\sum_{i=1}^n (x_i - y_i)^2},
     \]
     where $x = (x_1, x_2, \ldots, x_n)$ and $y = (y_1, y_2, \ldots, y_n)$.

2. **Discrete Space**:
   - **Set**: Any non-empty set $X$
   - **Metric**: The discrete metric is defined by
     \[
     d(x, y) = \begin{cases}
     0 & \text{if } x = y, \\
     1 & \text{if } x \neq y.
     \end{cases}
     \]

3. **Space of Continuous Functions $C[a, b]$**:
   - **Set**: $X = C[a, b]$, the set of all continuous functions on the interval $[a, b]$
   - **Metric**: The supremum metric is given by
     \[
     d(f, g) = \|f - g\|_\infty = \sup_{x \in [a, b]} |f(x) - g(x)|.
     \]

4. **Sequence Space $l^p$**:
   - **Set**: $X = l^p$, the set of all sequences $(x_n)$ such that $\sum_{n=1}^\infty |x_n|^p < \infty$
   - **Metric**: The $l^p$ metric is defined by
     \[
     d(x, y) = \|x - y\|_p = \left( \sum_{n=1}^\infty |x_n - y_n|^p \right)^{1/p},
     \]
     where $1 \leq p < \infty$.

### Proof Sketches and Examples

We will now verify that each of these metrics satisfies the properties of a metric.

#### Euclidean Space $\mathbb{R}^n$

- **Non-negativity**: $d(x, y) \geq 0$ and $d(x, y) = 0$ if and only if $x = y$.
- **Symmetry**: $d(x, y) = d(y, x)$.
- **Triangle Inequality**: $d(x, z) \leq d(x, y) + d(y, z)$ follows from the Minkowski inequality.

#### Discrete Space

- **Non-negativity**: $d(x, y) \geq 0$ and $d(x, y) = 0$ if and only if $x = y$.
- **Symmetry**: $d(x, y) = d(y, x)$.
- **Triangle Inequality**: If $x = y$ or $y = z$, then $d(x, z) \leq d(x, y) + d(y, z)$. If $x \neq y$ and $y \neq z$, then $d(x, z) = 1 \leq 1 + 1 = d(x, y) + d(y, z)$.

#### Space of Continuous Functions $C[a, b]$

- **Non-negativity**: $d(f, g) \geq 0$ and $d(f, g) = 0$ if and only if $f = g$.
- **Symmetry**: $d(f, g) = d(g, f)$.
- **Triangle Inequality**: $d(f, h) \leq d(f, g) + d(g, h)$ follows from the properties of the supremum.

#### Sequence Space $l^p$

- **Non-negativity**: $d(x, y) \geq 0$ and $d(x, y) = 0$ if and only if $x = y$.
- **Symmetry**: $d(x, y) = d(y, x)$.
- **Triangle Inequality**: $d(x, z) \leq d(x, y) + d(y, z)$ follows from the Minkowski inequality for sequences.

### Worked Example

Consider the sequence space $l^2$ with the metric
\[
d(x, y) = \|x - y\|_2 = \left( \sum_{n=1}^\infty |x_n - y_n|^2 \right)^{1/2}.
\]
Let $x = (1, \frac{1}{2}, \frac{1}{3}, \ldots)$ and $y = (0, 0, 0, \ldots)$. Compute the distance $d(x, y)$.

**Solution:**
\[
d(x, y) = \left( \sum_{n=1}^\infty \left| \frac{1}{n} - 0 \right|^2 \right)^{1/2} = \left( \sum_{n=1}^\infty \frac{1}{n^2} \right)^{1/2}.
\]
The series $\sum_{n=1}^\infty \frac{1}{n^2}$ converges to $\frac{\pi^2}{6}$. Therefore,
\[
d(x, y) = \left( \frac{\pi^2}{6} \right)^{1/2} = \frac{\pi}{\sqrt{6}}.
\]

In conclusion, metric spaces provide a versatile framework for studying various mathematical objects. By understanding the properties and examples of metric spaces, we can apply these concepts to a wide range of problems in analysis and topology.

## Convergence, Open/Closed Sets, and Continuity

In the previous sections, we introduced the concept of metric spaces and provided several important examples. In this section, we will delve into the fundamental concepts of convergence, open and closed sets, and continuity in metric spaces. These concepts are essential for understanding the behavior of sequences and functions in a metric space.

### Motivation

Convergence, open and closed sets, and continuity are central to the study of metric spaces. They allow us to describe the behavior of sequences and functions, and they provide a rigorous framework for analyzing the topological properties of a space.

### Definitions and Theorems

#### Convergence

A sequence $(x_n)$ in a metric space $(X, d)$ **converges** to a point $x \in X$ if for every $\epsilon > 0$, there exists an integer $N$ such that for all $n \geq N$, $d(x_n, x) < \epsilon$. We write this as
\[
\lim_{n \to \infty} x_n = x \quad \text{or} \quad x_n \to x.
\]

#### Open and Closed Sets

- A set $U \subseteq X$ is **open** if for every $x \in U$, there exists an $\epsilon > 0$ such that the ball $B(x, \epsilon) = \{y \in X : d(x, y) < \epsilon\}$ is contained in $U$.
- A set $F \subseteq X$ is **closed** if its complement $X \setminus F$ is open.

#### Continuity

A function $f: (X, d_X) \to (Y, d_Y)$ between two metric spaces is **continuous** at a point $x \in X$ if for every $\epsilon > 0$, there exists a $\delta > 0$ such that for all $y \in X$ with $d_X(x, y) < \delta$, we have $d_Y(f(x), f(y)) < \epsilon$. The function $f$ is **continuous** if it is continuous at every point in $X$.

### Proof Sketches and Examples

We will now provide some examples and proofs to illustrate these concepts.

#### Convergence

**Example 1: Convergence in $\mathbb{R}$**

Consider the sequence $(x_n) = \left( \frac{1}{n} \right)$ in $\mathbb{R}$ with the Euclidean metric. We claim that $x_n \to 0$.

**Proof:**
For any $\epsilon > 0$, choose $N$ such that $N > \frac{1}{\epsilon}$. Then for all $n \geq N$,
\[
|x_n - 0| = \left| \frac{1}{n} \right| < \epsilon.
\]
Thus, $x_n \to 0$.

#### Open and Closed Sets

**Example 2: Open and Closed Sets in $\mathbb{R}$**

- The interval $(0, 1)$ is an open set in $\mathbb{R}$ because for any $x \in (0, 1)$, we can choose $\epsilon = \min\{x, 1 - x\}$, and the ball $B(x, \epsilon) = (x - \epsilon, x + \epsilon)$ is contained in $(0, 1)$.
- The interval $[0, 1]$ is a closed set in $\mathbb{R}$ because its complement, $(-\infty, 0) \cup (1, \infty)$, is open.

#### Continuity

**Example 3: Continuity of a Function in $\mathbb{R}$**

Consider the function $f: \mathbb{R} \to \mathbb{R}$ defined by $f(x) = x^2$. We claim that $f$ is continuous.

**Proof:**
Let $x \in \mathbb{R}$ and $\epsilon > 0$. Choose $\delta = \min\{1, \frac{\epsilon}{2|x| + 1}\}$. For any $y \in \mathbb{R}$ with $|x - y| < \delta$,
\[
|f(x) - f(y)| = |x^2 - y^2| = |x - y||x + y| < \delta (|x| + |y|).
\]
Since $|y| \leq |x| + \delta \leq |x| + 1$, we have
\[
|f(x) - f(y)| < \delta (2|x| + 1) < \epsilon.
\]
Thus, $f$ is continuous at $x$.

### Worked Example

Consider the sequence $(x_n) = \left( \frac{1}{n} \right)$ in the space of continuous functions $C[0, 1]$ with the supremum metric. Determine whether $(x_n)$ converges to the zero function $f(x) = 0$.

**Solution:**
We need to show that for every $\epsilon > 0$, there exists an integer $N$ such that for all $n \geq N$,
\[
\|x_n - 0\|_\infty = \sup_{x \in [0, 1]} \left| \frac{1}{n} - 0 \right| < \epsilon.
\]
Since $\left| \frac{1}{n} \right| = \frac{1}{n}$ for all $x \in [0, 1]$, we have
\[
\|x_n - 0\|_\infty = \frac{1}{n}.
\]
For any $\epsilon > 0$, choose $N$ such that $N > \frac{1}{\epsilon}$. Then for all $n \geq N$,
\[
\|x_n - 0\|_\infty = \frac{1}{n} < \epsilon.
\]
Thus, $x_n \to 0$ in $C[0, 1]$.

In summary, convergence, open and closed sets, and continuity are fundamental concepts in the study of metric spaces. By understanding these concepts, we can analyze the behavior of sequences and functions in a rigorous and systematic manner.

## Completeness and Cauchy Sequences

In the previous sections, we explored the basic concepts of metric spaces, including convergence, open and closed sets, and continuity. In this section, we will delve into the concept of completeness, which is a crucial property of metric spaces. We will also discuss Cauchy sequences and prove the completion theorem, which shows that every metric space can be embedded in a complete metric space.

### Motivation

Completeness is a fundamental property that ensures the existence of limits for certain sequences. In many applications, it is essential to work in a complete metric space to guarantee the convergence of iterative processes and the existence of solutions to equations. Cauchy sequences are a key tool in the study of completeness, as they capture the idea of sequences that should converge but may not in an incomplete space.

### Definitions and Theorems

#### Cauchy Sequences

A sequence $(x_n)$ in a metric space $(X, d)$ is a **Cauchy sequence** if for every $\epsilon > 0$, there exists an integer $N$ such that for all $m, n \geq N$, $d(x_m, x_n) < \epsilon$.

#### Complete Metric Spaces

A metric space $(X, d)$ is **complete** if every Cauchy sequence in $X$ converges to a point in $X$.

#### Completion Theorem

Every metric space $(X, d)$ can be embedded in a complete metric space $(\overline{X}, \overline{d})$ such that $X$ is dense in $\overline{X}$ and $\overline{d}(x, y) = d(x, y)$ for all $x, y \in X$.

### Proof Sketches and Examples

#### Cauchy Sequences

**Example 1: Cauchy Sequence in $\mathbb{R}$**

Consider the sequence $(x_n) = \left( 1 + \frac{1}{2} + \frac{1}{3} + \cdots + \frac{1}{n} \right)$ in $\mathbb{R}$. This sequence is known as the harmonic series, and it is not a Cauchy sequence.

**Proof:**
To show that $(x_n)$ is not a Cauchy sequence, we need to find an $\epsilon > 0$ such that for any $N$, there exist $m, n \geq N$ with $|x_m - x_n| \geq \epsilon$.

Consider the partial sums of the harmonic series. For any $k \geq 1$,
\[
x_{2k} - x_k = \left( 1 + \frac{1}{2} + \cdots + \frac{1}{2k} \right) - \left( 1 + \frac{1}{2} + \cdots + \frac{1}{k} \right) = \frac{1}{k+1} + \frac{1}{k+2} + \cdots + \frac{1}{2k}.
\]
Each term in the sum $\frac{1}{k+1} + \frac{1}{k+2} + \cdots + \frac{1}{2k}$ is at least $\frac{1}{2k}$, and there are $k$ terms. Therefore,
\[
x_{2k} - x_k \geq k \cdot \frac{1}{2k} = \frac{1}{2}.
\]
Thus, for $\epsilon = \frac{1}{2}$, no matter how large $N$ is, we can always find $m = 2k$ and $n = k$ with $m, n \geq N$ such that $|x_m - x_n| \geq \epsilon$. Hence, $(x_n)$ is not a Cauchy sequence.

#### Complete Metric Spaces

**Example 2: Completeness of $\mathbb{R}$**

The real numbers $\mathbb{R}$ with the Euclidean metric are a complete metric space.

**Proof:**
Let $(x_n)$ be a Cauchy sequence in $\mathbb{R}$. We need to show that $(x_n)$ converges to a limit in $\mathbb{R}$.

Since $(x_n)$ is a Cauchy sequence, it is bounded. By the Bolzano-Weierstrass theorem, every bounded sequence in $\mathbb{R}$ has a convergent subsequence. Let $(x_{n_k})$ be a convergent subsequence of $(x_n)$, and let $L$ be its limit. We will show that $(x_n)$ converges to $L$.

Given $\epsilon > 0$, there exists an integer $K$ such that for all $k \geq K$, $|x_{n_k} - L| < \frac{\epsilon}{2}$. Since $(x_n)$ is a Cauchy sequence, there exists an integer $N$ such that for all $m, n \geq N$, $|x_m - x_n| < \frac{\epsilon}{2}$.

Choose $M = \max\{N, n_K\}$. For any $n \geq M$, there exists $k \geq K$ such that $n_k \geq M$. Then,
\[
|x_n - L| \leq |x_n - x_{n_k}| + |x_{n_k} - L| < \frac{\epsilon}{2} + \frac{\epsilon}{2} = \epsilon.
\]
Thus, $(x_n)$ converges to $L$.

#### Completion Theorem

**Proof of the Completion Theorem:**

Let $(X, d)$ be a metric space. Define the set $\overline{X}$ to be the set of all equivalence classes of Cauchy sequences in $X$, where two Cauchy sequences $(x_n)$ and $(y_n)$ are equivalent if
\[
\lim_{n \to \infty} d(x_n, y_n) = 0.
\]
Define a metric $\overline{d}$ on $\overline{X}$ by
\[
\overline{d}([(x_n)], [(y_n)]) = \lim_{n \to \infty} d(x_n, y_n),
\]
where $[(x_n)]$ denotes the equivalence class of the Cauchy sequence $(x_n)$.

1. **Well-definedness of $\overline{d}$**: If $(x_n) \sim (x_n')$ and $(y_n) \sim (y_n')$, then
   \[
   \lim_{n \to \infty} d(x_n, x_n') = 0 \quad \text{and} \quad \lim_{n \to \infty} d(y_n, y_n') = 0.
   \]
   By the triangle inequality,
   \[
   |d(x_n, y_n) - d(x_n', y_n')| \leq d(x_n, x_n') + d(y_n, y_n').
   \]
   Taking the limit as $n \to \infty$, we get
   \[
   \lim_{n \to \infty} |d(x_n, y_n) - d(x_n', y_n')| = 0,
   \]
   so $\lim_{n \to \infty} d(x_n, y_n) = \lim_{n \to \infty} d(x_n', y_n')$.

2. **Properties of $\overline{d}$**: It can be verified that $\overline{d}$ satisfies the properties of a metric.

3. **Embedding of $X$ in $\overline{X}$**: Define a map $i: X \to \overline{X}$ by $i(x) = [(x, x, x, \ldots)]$. This map is an isometry, i.e., $\overline{d}(i(x), i(y)) = d(x, y)$ for all $x, y \in X$.

4. **Density of $X$ in $\overline{X}$**: For any equivalence class $[(x_n)] \in \overline{X}$ and any $\epsilon > 0$, there exists an integer $N$ such that for all $m, n \geq N$, $d(x_m, x_n) < \epsilon$. Thus, $d(x_N, x_n) < \epsilon$ for all $n \geq N$, and
   \[
   \overline{d}([(x_n)], i(x_N)) = \lim_{n \to \infty} d(x_n, x_N) \leq \epsilon.
   \]
   Therefore, $i(X)$ is dense in $\overline{X}$.

5. **Completeness of $\overline{X}$**: Let $([(x_n^k)])$ be a Cauchy sequence in $\overline{X}$. For each $k$, choose a representative $(x_n^k)$ of the equivalence class $[(x_n^k)]$. Since $([(x_n^k)])$ is a Cauchy sequence, for any $\epsilon > 0$, there exists an integer $K$ such that for all $j, k \geq K$,
   \[
   \overline{d}([(x_n^j)], [(x_n^k)]) = \lim_{n \to \infty} d(x_n^j, x_n^k) < \epsilon.
   \]
   In particular, for each fixed $n$, $(x_n^k)$ is a Cauchy sequence in $X$. Since $X$ is a metric space, we can choose a subsequence $(x_n^{k_n})$ such that $(x_n^{k_n})$ converges to some $x_n \in X$ for each $n$. The sequence $(x_n)$ is a Cauchy sequence in $X$, and $[(x_n)] \in \overline{X}$. Moreover,
   \[
   \overline{d}([(x_n^k)], [(x_n)]) = \lim_{n \to \infty} d(x_n^k, x_n) \to 0 \quad \text{as} \quad k \to \infty.
   \]
   Therefore, $([(x_n^k)])$ converges to $[(x_n)]$ in $\overline{X}$, and $\overline{X}$ is complete.

### Worked Example

Consider the space of rational numbers $\mathbb{Q}$ with the Euclidean metric. Show that the sequence $(x_n) = \left( 1 + \frac{1}{2!} + \frac{1}{3!} + \cdots + \frac{1}{n!} \right)$ is a Cauchy sequence in $\mathbb{Q}$, but it does not converge to a point in $\mathbb{Q}$.

**Solution:**

1. **Cauchy Sequence**:
   To show that $(x_n)$ is a Cauchy sequence, we need to show that for every $\epsilon > 0$, there exists an integer $N$ such that for all $m, n \geq N$, $|x_m - x_n| < \epsilon$.

   Without loss of generality, assume $m > n$. Then,
   \[
   |x_m - x_n| = \left| \frac{1}{(n+1)!} + \frac{1}{(n+2)!} + \cdots + \frac{1}{m!} \right|.
   \]
   Using the fact that $\frac{1}{k!} \leq \frac{1}{2^{k-1}}$ for $k \geq 2$, we get
   \[
   |x_m - x_n| \leq \frac{1}{(n+1)!} + \frac{1}{(n+2)!} + \cdots + \frac{1}{m!} \leq \frac{1}{(n+1)!} \left( 1 + \frac{1}{2} + \frac{1}{2^2} + \cdots \right) = \frac{1}{(n+1)!} \cdot 2.
   \]
   For any $\epsilon > 0$, choose $N$ such that $\frac{2}{(N+1)!} < \epsilon$. Then for all $m, n \geq N$,
   \[
   |x_m - x_n| < \epsilon.
   \]
   Therefore, $(x_n)$ is a Cauchy sequence in $\mathbb{Q}$.

2. **Non-convergence in $\mathbb{Q}$**:
   The sequence $(x_n)$ converges to $e$ in $\mathbb{R}$, where $e = \sum_{k=0}^\infty \frac{1}{k!}$. However, $e$ is an irrational number, so it does not belong to $\mathbb{Q}$. Therefore, $(x_n)$ does not converge to a point in $\mathbb{Q}$.

In conclusion, completeness and Cauchy sequences are essential concepts in the study of metric spaces. The completion theorem provides a way to embed any metric space in a complete metric space, ensuring the existence of limits for Cauchy sequences. By understanding these concepts, we can analyze the topological and analytical properties of metric spaces in a more comprehensive manner.

## Baire Category Theorem

The Baire Category Theorem is a fundamental result in topology and functional analysis. It provides a powerful tool for understanding the structure of complete metric spaces and has numerous applications in various areas of mathematics. In this section, we will state and prove the Baire Category Theorem and discuss some of its important consequences.

### Motivation

The Baire Category Theorem is motivated by the need to understand the "size" and "density" of subsets in a complete metric space. It asserts that a complete metric space cannot be written as a countable union of nowhere dense sets, which has profound implications for the structure of the space.

### Definitions and Theorems

#### Nowhere Dense Sets

A subset $A$ of a metric space $(X, d)$ is **nowhere dense** if the interior of its closure is empty, i.e., $\text{int}(\overline{A}) = \emptyset$.

#### Baire Category Theorem

**Theorem (Baire Category Theorem):** Let $(X, d)$ be a complete metric space. If $\{A_n\}_{n=1}^\infty$ is a countable collection of nowhere dense subsets of $X$, then the union $\bigcup_{n=1}^\infty A_n$ is not dense in $X$. Equivalently, the complement $X \setminus \bigcup_{n=1}^\infty A_n$ is dense in $X$.

### Proof of the Baire Category Theorem

**Proof:**

Assume, for contradiction, that $\bigcup_{n=1}^\infty A_n$ is dense in $X$. This means that for every non-empty open set $U \subseteq X$, $U \cap \bigcup_{n=1}^\infty A_n \neq \emptyset$. We will construct a nested sequence of closed balls whose intersection is empty, leading to a contradiction.

1. **Step 1: Constructing the Nested Sequence of Closed Balls**

   - Start with a non-empty open set $U_1 \subseteq X$. Since $\bigcup_{n=1}^\infty A_n$ is dense in $X$, there exists an integer $n_1$ such that $U_1 \cap A_{n_1} \neq \emptyset$.
   - Since $A_{n_1}$ is nowhere dense, $\text{int}(\overline{A_{n_1}}) = \emptyset$. Therefore, $U_1 \setminus \overline{A_{n_1}}$ is a non-empty open set.
   - Choose a closed ball $B_1 = \overline{B(x_1, r_1)} \subseteq U_1 \setminus \overline{A_{n_1}}$ with radius $r_1 < 1$.

   - Inductively, suppose we have constructed a closed ball $B_k = \overline{B(x_k, r_k)} \subseteq U_k \setminus \overline{A_{n_k}}$ with radius $r_k < \frac{1}{k}$.
   - Since $\bigcup_{n=1}^\infty A_n$ is dense in $X$, there exists an integer $n_{k+1}$ such that $B_k \cap A_{n_{k+1}} \neq \emptyset$.
   - Since $A_{n_{k+1}}$ is nowhere dense, $\text{int}(\overline{A_{n_{k+1}}}) = \emptyset$. Therefore, $B_k \setminus \overline{A_{n_{k+1}}}$ is a non-empty open set.
   - Choose a closed ball $B_{k+1} = \overline{B(x_{k+1}, r_{k+1})} \subseteq B_k \setminus \overline{A_{n_{k+1}}}$ with radius $r_{k+1} < \frac{1}{k+1}$.

2. **Step 2: Intersection of the Nested Sequence**

   - The sequence of closed balls $\{B_k\}_{k=1}^\infty$ is nested, i.e., $B_1 \supseteq B_2 \supseteq B_3 \supseteq \cdots$.
   - The radii of the balls satisfy $r_k < \frac{1}{k}$, so the diameters of the balls tend to zero.
   - Since $(X, d)$ is a complete metric space, the intersection of the nested sequence of closed balls is non-empty. Let $x \in \bigcap_{k=1}^\infty B_k$.

3. **Step 3: Contradiction**

   - For each $k$, $x \in B_k \subseteq U_k \setminus \overline{A_{n_k}}$.
   - Therefore, $x \notin \overline{A_{n_k}}$ for any $k$.
   - Since $\{A_{n_k}\}_{k=1}^\infty$ is a subsequence of $\{A_n\}_{n=1}^\infty$, $x \notin \bigcup_{n=1}^\infty A_n$.
   - This contradicts the assumption that $\bigcup_{n=1}^\infty A_n$ is dense in $X$.

Thus, the union $\bigcup_{n=1}^\infty A_n$ is not dense in $X$, and the complement $X \setminus \bigcup_{n=1}^\infty A_n$ is dense in $X$.

### Consequences of the Baire Category Theorem

The Baire Category Theorem has several important consequences, including:

1. **Non-emptiness of Residual Sets**: A **residual set** in a complete metric space is a countable intersection of dense open sets. The Baire Category Theorem implies that residual sets are dense in the space. This is often used to show the existence of "generic" elements with certain properties.

2. **Existence of Nowhere Differentiable Continuous Functions**: The Baire Category Theorem can be used to show that the set of continuous functions on $[0, 1]$ that are nowhere differentiable is residual in $C[0, 1]$. This implies that most continuous functions are nowhere differentiable in a topological sense.

3. **Banach-Steinhaus Theorem (Uniform Boundedness Principle)**: The Baire Category Theorem is a key ingredient in the proof of the Banach-Steinhaus Theorem, which states that a family of continuous linear operators between Banach spaces is uniformly bounded if it is pointwise bounded.

### Worked Example

Consider the space of continuous functions $C[0, 1]$ with the supremum metric. Show that the set of continuous functions that are nowhere differentiable is residual in $C[0, 1]$.

**Solution:**

1. **Nowhere Differentiable Functions**:
   A function $f \in C

---

*This is Part 1 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Next: [Part 2 — Normed and Banach Spaces](/en/functional-analysis/02-normed-and-banach/)*

## What's next

With metric spaces providing the foundation of distance and convergence, we are ready to add algebraic structure. In the next article, we equip our spaces with a norm — a single function that simultaneously gives us distance and a vector space structure — and discover Banach spaces, where completeness and linearity combine to produce the powerful theorems that make functional analysis tick.

