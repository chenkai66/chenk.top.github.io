---
title: "Normed Spaces and Banach Spaces: The Right Framework"
date: 2021-10-03 09:00:00
tags:
  - functional-analysis
  - banach-spaces
  - normed-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 2
series_total: 12
translationKey: "functional-analysis-2"
description: "Adding linear structure to metrics: norms, Banach spaces, and why finite dimensions are special."
---

# Normed Spaces and Banach Spaces

In the realm of functional analysis, normed spaces and Banach spaces play a fundamental role. These structures generalize the concept of Euclidean space by introducing a notion of "size" or "length" for vectors, which is crucial for many applications in mathematics and beyond. This article will delve into the definitions, properties, and examples of normed spaces and Banach spaces, providing rigorous proofs and worked examples to illustrate key concepts.

## From Metrics to Norms (Adding Linear Structure)

### Motivation
In metric spaces, we have a notion of distance between points, but no inherent structure that allows us to add or scale these points. By adding a linear structure, we can define a norm, which not only measures the distance from a point to the origin but also respects the operations of vector addition and scalar multiplication. This leads to the concept of normed spaces, which are more structured and powerful than mere metric spaces.


![Unit balls in different norms on R^2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/02-normed-and-banach/fa_fig1_unit_balls.png)

### Definitions and Theorems
A **metric space** \((X, d)\) is a set \(X\) equipped with a metric \(d: X \times X \to \mathbb{R}\) that satisfies:
1. \(d(x, y) \geq 0\) for all \(x, y \in X\), and \(d(x, y) = 0\) if and only if \(x = y\).
2. \(d(x, y) = d(y, x)\) for all \(x, y \in X\).
3. \(d(x, z) \leq d(x, y) + d(y, z)\) for all \(x, y, z \in X\).

A **vector space** \(V\) over a field \(\mathbb{F}\) (typically \(\mathbb{R}\) or \(\mathbb{C}\)) is a set equipped with two operations: vector addition and scalar multiplication, satisfying certain axioms.

A **norm** on a vector space \(V\) is a function \(\|\cdot\|: V \to \mathbb{R}\) that satisfies:
1. \(\|x\| \geq 0\) for all \(x \in V\), and \(\|x\| = 0\) if and only if \(x = 0\).
2. \(\|\alpha x\| = |\alpha| \|x\|\) for all \(\alpha \in \mathbb{F}\) and \(x \in V\).
3. \(\|x + y\| \leq \|x\| + \|y\|\) for all \(x, y \in V\).

The norm induces a metric \(d(x, y) = \|x - y\|\).

### Proof Sketches
To show that a norm induces a metric, we need to verify the three properties of a metric:
1. **Non-negativity and identity of indiscernibles**: \(\|x - y\| \geq 0\) and \(\|x - y\| = 0\) if and only if \(x = y\). This follows directly from the properties of the norm.
2. **Symmetry**: \(\|x - y\| = \|-(x - y)\| = \|y - x\|\). This follows from the homogeneity property of the norm.
3. **Triangle inequality**: \(\|x - z\| \leq \|x - y\| + \|y - z\|\). This follows from the triangle inequality property of the norm.

### Worked Example
Consider the vector space \(\mathbb{R}^2\) with the Euclidean norm \(\|(x, y)\| = \sqrt{x^2 + y^2}\). We will show that this norm induces a metric.

1. **Non-negativity and identity of indiscernibles**: \(\|(x, y) - (a, b)\| = \sqrt{(x - a)^2 + (y - b)^2} \geq 0\). If \(\|(x, y) - (a, b)\| = 0\), then \((x - a)^2 + (y - b)^2 = 0\), which implies \(x = a\) and \(y = b\).
2. **Symmetry**: \(\|(x, y) - (a, b)\| = \sqrt{(x - a)^2 + (y - b)^2} = \sqrt{(a - x)^2 + (b - y)^2} = \|(a, b) - (x, y)\|\).
3. **Triangle inequality**: \(\|(x, y) - (c, d)\| = \sqrt{(x - c)^2 + (y - d)^2} \leq \sqrt{(x - a)^2 + (y - b)^2} + \sqrt{(a - c)^2 + (b - d)^2} = \|(x, y) - (a, b)\| + \|(a, b) - (c, d)\|\).

Thus, the Euclidean norm on \(\mathbb{R}^2\) induces a metric.

## Normed Spaces Definition, Induced Metric, Examples

### Motivation
Normed spaces are vector spaces equipped with a norm, which provides a way to measure the "size" of vectors. This structure is essential for many areas of mathematics, including functional analysis, differential equations, and optimization. Understanding the definition and properties of normed spaces is crucial for working with these spaces.

### Definitions and Theorems
A **normed space** is a pair \((V, \|\cdot\|)\) where \(V\) is a vector space and \(\|\cdot\|\) is a norm on \(V\). The norm induces a metric \(d(x, y) = \|x - y\|\), making \((V, d)\) a metric space.

### Examples
1. **Euclidean space \(\mathbb{R}^n\)**: The Euclidean norm \(\|x\|_2 = \sqrt{\sum_{i=1}^n x_i^2}\) makes \(\mathbb{R}^n\) a normed space.
2. **Sequence spaces \(\ell^p\)**: For \(1 \leq p < \infty\), the space \(\ell^p\) consists of all sequences \((x_n)\) such that \(\sum_{n=1}^\infty |x_n|^p < \infty\). The norm is \(\|x\|_p = \left(\sum_{n=1}^\infty |x_n|^p\right)^{1/p}\).
3. **Function spaces \(C[a, b]\)**: The space of continuous functions on \([a, b]\) with the sup norm \(\|f\|_\infty = \sup_{t \in [a, b]} |f(t)|\).

### Proof Sketches
To show that \(\ell^p\) is a normed space, we need to verify the properties of the norm:
1. **Non-negativity and identity of indiscernibles**: \(\|x\|_p \geq 0\) and \(\|x\|_p = 0\) if and only if \(x = 0\). This follows from the fact that the sum of non-negative terms is zero if and only if each term is zero.
2. **Homogeneity**: \(\|\alpha x\|_p = \left(\sum_{n=1}^\infty |\alpha x_n|^p\right)^{1/p} = |\alpha| \left(\sum_{n=1}^\infty |x_n|^p\right)^{1/p} = |\alpha| \|x\|_p\).
3. **Triangle inequality**: \(\|x + y\|_p = \left(\sum_{n=1}^\infty |x_n + y_n|^p\right)^{1/p} \leq \left(\sum_{n=1}^\infty (|x_n| + |y_n|)^p\right)^{1/p} \leq \left(\sum_{n=1}^\infty |x_n|^p\right)^{1/p} + \left(\sum_{n=1}^\infty |y_n|^p\right)^{1/p} = \|x\|_p + \|y\|_p\).

### Worked Example
Consider the sequence space \(\ell^2\) with the norm \(\|x\|_2 = \left(\sum_{n=1}^\infty |x_n|^2\right)^{1/2}\). We will show that this norm satisfies the triangle inequality.

Let \(x = (x_n)\) and \(y = (y_n)\) be sequences in \(\ell^2\). Then,
\[
\|x + y\|_2 = \left(\sum_{n=1}^\infty |x_n + y_n|^2\right)^{1/2}.
\]
Using the Minkowski inequality for sums, we have
\[
\left(\sum_{n=1}^\infty |x_n + y_n|^2\right)^{1/2} \leq \left(\sum_{n=1}^\infty |x_n|^2\right)^{1/2} + \left(\sum_{n=1}^\infty |y_n|^2\right)^{1/2} = \|x\|_2 + \|y\|_2.
\]
Thus, the norm \(\|\cdot\|_2\) on \(\ell^2\) satisfies the triangle inequality.

## \(l^p\) Spaces (Completeness Proof, Unit Ball Shapes)

### Motivation
The \(l^p\) spaces are a fundamental class of normed spaces that arise naturally in various areas of mathematics. They are particularly important because they are complete, meaning every Cauchy sequence converges. This completeness property makes them Banach spaces, which are central to functional analysis.

### Definitions and Theorems
For \(1 \leq p < \infty\), the space \(l^p\) consists of all sequences \((x_n)\) such that \(\sum_{n=1}^\infty |x_n|^p < \infty\). The norm is given by
\[
\|x\|_p = \left(\sum_{n=1}^\infty |x_n|^p\right)^{1/p}.
\]
The unit ball in \(l^p\) is the set \(\{x \in l^p : \|x\|_p \leq 1\}\).

### Completeness Proof
To show that \(l^p\) is complete, we need to prove that every Cauchy sequence in \(l^p\) converges to a limit in \(l^p\).

**Proof:**
Let \((x^{(k)})\) be a Cauchy sequence in \(l^p\), where \(x^{(k)} = (x_n^{(k)})\). For any \(\epsilon > 0\), there exists \(N \in \mathbb{N}\) such that for all \(m, n \geq N\),
\[
\|x^{(m)} - x^{(n)}\|_p < \epsilon.
\]
This implies that for each fixed \(n\), the sequence \((x_n^{(k)})\) is a Cauchy sequence in \(\mathbb{R}\) (or \(\mathbb{C}\)). Since \(\mathbb{R}\) (or \(\mathbb{C}\)) is complete, \((x_n^{(k)})\) converges to some \(x_n \in \mathbb{R}\) (or \(\mathbb{C}\)). Define \(x = (x_n)\).

We need to show that \(x \in l^p\) and that \(x^{(k)} \to x\) in \(l^p\). First, we show that \(x \in l^p\). Since \((x^{(k)})\) is Cauchy, there exists \(M \in \mathbb{N}\) such that for all \(m, n \geq M\),
\[
\|x^{(m)} - x^{(n)}\|_p < 1.
\]
Fix \(m \geq M\). Then for all \(n \geq M\),
\[
\left(\sum_{n=1}^\infty |x_n^{(m)} - x_n^{(n)}|^p\right)^{1/p} < 1.
\]
Taking the limit as \(n \to \infty\), we get
\[
\left(\sum_{n=1}^\infty |x_n^{(m)} - x_n|^p\right)^{1/p} \leq 1.
\]
Thus,
\[
\sum_{n=1}^\infty |x_n^{(m)} - x_n|^p < \infty.
\]
Since \(x^{(m)} \in l^p\), we have
\[
\sum_{n=1}^\infty |x_n^{(m)}|^p < \infty.
\]
Therefore,
\[
\sum_{n=1}^\infty |x_n|^p \leq \sum_{n=1}^\infty (|x_n^{(m)}| + |x_n^{(m)} - x_n|)^p < \infty.
\]
This shows that \(x \in l^p\).

Next, we show that \(x^{(k)} \to x\) in \(l^p\). Given \(\epsilon > 0\), choose \(N \in \mathbb{N}\) such that for all \(m, n \geq N\),
\[
\|x^{(m)} - x^{(n)}\|_p < \frac{\epsilon}{2}.
\]
For \(k \geq N\),
\[
\|x^{(k)} - x\|_p = \left(\sum_{n=1}^\infty |x_n^{(k)} - x_n|^p\right)^{1/p} \leq \frac{\epsilon}{2} < \epsilon.
\]
Thus, \(x^{(k)} \to x\) in \(l^p\).

### Unit Ball Shapes
The unit ball in \(l^p\) has different shapes depending on \(p\):
- For \(p = 1\), the unit ball is a diamond.
- For \(p = 2\), the unit ball is a sphere.
- For \(p = \infty\), the unit ball is a cube.

### Worked Example
Consider the sequence space \(l^2\) with the norm \(\|x\|_2 = \left(\sum_{n=1}^\infty |x_n|^2\right)^{1/2}\). We will show that the sequence \((x^{(k)})\) defined by \(x_n^{(k)} = \frac{1}{k}\) for \(n \leq k\) and \(x_n^{(k)} = 0\) for \(n > k\) is a Cauchy sequence in \(l^2\).

For \(m, n \geq N\),
\[
\|x^{(m)} - x^{(n)}\|_2 = \left(\sum_{k=\min(m, n)+1}^{\max(m, n)} \left(\frac{1}{k}\right)^2\right)^{1/2}.
\]
Since \(\sum_{k=1}^\infty \frac{1}{k^2}\) converges, for any \(\epsilon > 0\), there exists \(N \in \mathbb{N}\) such that for all \(m, n \geq N\),
\[
\left(\sum_{k=\min(m, n)+1}^{\max(m, n)} \left(\frac{1}{k}\right)^2\right)^{1/2} < \epsilon.
\]
Thus, \((x^{(k)})\) is a Cauchy sequence in \(l^2\).

## \(C[a, b]\) with Sup Norm (Completeness, Weierstrass Connection)

### Motivation
The space \(C[a, b]\) of continuous functions on a closed interval \([a, b]\) with the sup norm is a classic example of a Banach space. This space is important in many areas of analysis, including approximation theory and the study of differential equations. The completeness of \(C[a, b]\) and its connection to the Weierstrass approximation theorem make it a fundamental object in functional analysis.

### Definitions and Theorems
The space \(C[a, b]\) consists of all continuous functions \(f: [a, b] \to \mathbb{R}\) (or \(\mathbb{C}\)). The sup norm is given by
\[
\|f\|_\infty = \sup_{t \in [a, b]} |f(t)|.
\]

### Completeness Proof
To show that \(C[a, b]\) is complete, we need to prove that every Cauchy sequence in \(C[a, b]\) converges to a limit in \(C[a, b]\).

**Proof:**
Let \((f_n)\) be a Cauchy sequence in \(C[a, b]\). For any \(\epsilon > 0\), there exists \(N \in \mathbb{N}\) such that for all \(m, n \geq N\),
\[
\|f_m - f_n\|_\infty < \epsilon.
\]
This implies that for each fixed \(t \in [a, b]\), the sequence \((f_n(t))\) is a Cauchys sequence in \(\mathbb{R}\) (or \(\mathbb{C}\)). Since \(\mathbb{R}\) (or \(\mathbb{C}\)) is complete, \((f_n(t))\) converges to some \(f(t) \in \mathbb{R}\) (or \(\mathbb{C}\)). Define \(f: [a, b] \to \mathbb{R}\) (or \(\mathbb{C}\)) by \(f(t) = \lim_{n \to \infty} f_n(t)\).

We need to show that \(f \in C[a, b]\) and that \(f_n \to f\) in \(C[a, b]\). First, we show that \(f\) is continuous. Fix \(t_0 \in [a, b]\) and \(\epsilon > 0\). Choose \(N \in \mathbb{N}\) such that for all \(n \geq N\),
\[
\|f_n - f_N\|_\infty < \frac{\epsilon}{3}.
\]
Since \(f_N\) is continuous at \(t_0\), there exists \(\delta > 0\) such that for all \(t \in [a, b]\) with \(|t - t_0| < \delta\),
\[
|f_N(t) - f_N(t_0)| < \frac{\epsilon}{3}.
\]
For such \(t\),
\[
|f(t) - f(t_0)| \leq |f(t) - f_N(t)| + |f_N(t) - f_N(t_0)| + |f_N(t_0) - f(t_0)| < \frac{\epsilon}{3} + \frac{\epsilon}{3} + \frac{\epsilon}{3} = \epsilon.
\]
Thus, \(f\) is continuous at \(t_0\).

Next, we show that \(f_n \to f\) in \(C[a, b]\). Given \(\epsilon > 0\), choose \(N \in \mathbb{N}\) such that for all \(m, n \geq N\),
\[
\|f_m - f_n\|_\infty < \frac{\epsilon}{2}.
\]
For \(n \geq N\),
\[
\|f_n - f\|_\infty = \sup_{t \in [a, b]} |f_n(t) - f(t)| \leq \frac{\epsilon}{2} < \epsilon.
\]
Thus, \(f_n \to f\) in \(C[a, b]\).

### Weierstrass Approximation Theorem
The Weierstrass approximation theorem states that for any continuous function \(f \in C[a, b]\) and any \(\epsilon > 0\), there exists a polynomial \(P\) such that
\[
\|f - P\|_\infty < \epsilon.
\]
This theorem shows that polynomials are dense in \(C[a, b]\) with respect to the sup norm.

### Worked Example
Consider the sequence of functions \(f_n(t) = t^n\) on \([0, 1]\). We will show that \((f_n)\) is a Cauchy sequence in \(C[0, 1]\) with the sup norm.

For \(m, n \geq N\),
\[
\|f_m - f_n\|_\infty = \sup_{t \in [0, 1]} |t^m - t^n|.
\]
Since \(t^m\) and \(t^n\) are both continuous and \(t \in [0, 1]\), the maximum value of \(|t^m - t^n|\) occurs at \(t = 1\). Thus,
\[
\|f_m - f_n\|_\infty = |1^m - 1^n| = 0.
\]
This shows that \((f_n)\) is a Cauchy sequence in \(C[0, 1]\).

## Finite vs Infinite Dimensions (Equivalence of Norms Proof, Riesz Lemma)

### Motivation
The distinction between finite-dimensional and infinite-dimensional normed spaces is a fundamental one in functional analysis. In finite dimensions, all norms are equivalent, and the unit ball is compact. In infinite dimensions, these properties do not hold, and the geometry of the space can be much more complex. Understanding these differences is crucial for working with normed spaces.

### Definitions and Theorems
A normed space \(V\) is **finite-dimensional** if it has a finite basis. Otherwise, it is **infinite-dimensional**.

**Theorem (Equivalence of Norms in Finite Dimensions):** All norms on a finite-dimensional normed space are equivalent. That is, if \(\|\cdot\|_1\) and \(\|\cdot\|_2\) are norms on a finite-dimensional space \(V\), there exist constants \(c, C > 0\) such that for all \(x \in V\),
\[
c \|x\|_1 \leq \|x\|_2 \leq C \|x\|_1.
\]

**Riesz Lemma:** Let \(V\) be an infinite-dimensional normed space and \(Y \subset V\) a proper closed subspace. For any \(0 < \theta < 1\), there exists \(x \in V\) with \(\|x\| = 1\) such that \(\|x - y\| \geq \theta\) for all \(y \in Y\).

### Proof Sketches
**Equivalence of Norms in Finite Dimensions:**
Let \(\{e_1, e_2, \ldots, e_n\}\) be a basis for \(V\). Any \(x \in V\) can be written as \(x = \sum_{i=1}^n \alpha_i e_i\). Define
\[
\|x\|_1 = \sum_{i=1}^n |\alpha_i|, \quad \|x\|_2 = \left(\sum_{i=1}^n |\alpha_i|^2\right)^{1/2}.
\]
By the Cauchy-Schwarz inequality,
\[
\|x\|_1 \leq \sqrt{n} \|x\|_2.
\]
Also,
\[
\|x\|_2 \leq \sqrt{n} \|x\|_1.
\]
Thus, \(\|x\|_1\) and \(\|x\|_2\) are equivalent. By a similar argument, any two norms on \(V\) are equivalent.

**Riesz Lemma:**
Let \(Y\) be a proper closed subspace of \(V\). Choose \(z \in V \setminus Y\) and let \(d = \inf_{y \in Y} \|z - y\|\). Since \(Y\) is closed, \(d > 0\). For any \(0 < \theta < 1\), choose \(y_0 \in Y\) such that \(\|z - y_0\| < \frac{d}{\theta}\). Let \(x = \frac{z - y_0}{\|z - y_0\|}\). Then \(\|x\| = 1\) and for any \(y \in Y\),
\[
\|x - y\| = \left\|\frac{z - y_0}{\|z - y_0\|} - y\right\| = \frac{1}{\|z - y_0\|} \|z - (y_0 + \|z - y_0\| y)\| \geq \frac{d}{\|z - y_0\|} > \theta.
\]

### Worked Example
Consider the finite-dimensional space \(\mathbb{R}^2\) with the Euclidean norm \(\|(x, y)\|_2 = \sqrt{x^2 + y^2}\) and the taxicab norm \(\|(x, y)\|_1 = |x| + |y|\). We will show that these norms are equivalent.

First, note that
\[
\|(x, y)\|_1 = |x| + |y| \leq \sqrt{2} \sqrt{x^2 + y^2} = \sqrt{2} \|(x, y)\|_2.
\]
Also,
\[
\|(x, y)\|_2 = \sqrt{x^2 + y^2} \leq \sqrt{2} \max(|x|, |y|) \leq \sqrt{2} (|x| + |y|) = \sqrt{2} \|(x, y)\|_1.
\]
Thus, the norms \(\|\cdot\|_1\) and \(\|\cdot\|_2\) are equivalent with constants \(c = \frac{1}{\sqrt{2}}\) and \(C = \sqrt{2}\).

## Series in Banach Spaces (Absolute Convergence Implies Convergence)

### Motivation
In Banach spaces, the concept of series is a natural extension of the familiar notion of series in \(\mathbb{R}\) or \(\mathbb{C}\). One of the key results in Banach spaces is that absolute convergence implies convergence. This property is crucial for many applications in functional analysis, including the study of Fourier series and the solution of differential equations.

### Definitions and Theorems
A **Banach space** is a complete normed space. A series \(\sum_{n=1}^\infty x_n\) in a Banach space \(X\) is said to be **absolutely convergent** if \(\sum_{n=1}^\infty \|x_n\|\) converges in \(\mathbb{R}\).

**Theorem (Absolute Convergence Implies Convergence):** If \(\sum_{n=1}^\infty x_n\) is absolutely convergent in a Banach space \(X\), then \(\sum_{n=1}^\infty x_n\) converges in \(X\).

### Proof Sketch
Let \(\sum_{n=1}^\infty x_n\) be an absolutely convergent series in a Banach space \(X\). Define the partial sums \(S_N = \sum_{n=1}^N x_n\). We need to show that \((S_N)\) is a Cauchy sequence in \(X\).

Given \(\epsilon > 0\), since \(\sum_{n=1}^\infty \|x_n\|\) converges, there exists \(N \in \mathbb{N}\) such that for all \(m > n \geq N\),
\[
\sum_{k=n+1}^m \|x_k\| < \epsilon.
\]
Then,
\[
\|S_m - S_n\| = \left\|\sum_{k=n+1}^m x_k\right\| \leq \sum_{k=n+1}^m \|x_k\| < \epsilon.
\]
Thus, \((S_N)\) is a Cauchy sequence in \(X\). Since \(X\) is complete, \((S_N)\) converges to some \(S \in X\). Therefore, \(\sum_{n=1}^\infty x_n\) converges to \(S\).

### Worked Example
Consider the Banach space \(\ell^2\) with the norm \(\|x\|_2 = \left(\sum_{n=1}^\infty |x_n|^2\right)^{1/2}\). Let \((x_n)\) be a sequence in \(\ell^2\) defined by \(x_n = \frac{1}{n^2}\). We will show that the series \(\sum_{n=1}^\infty x_n\) is absolutely convergent and hence convergent in \(\ell^2\).

First, note that
\[
\sum_{n=1}^\infty \|x_n\|_2 = \sum_{n=1}^\infty \left(\sum_{k=1}^\infty \left|\frac{1}{n^2}\right|^2\right)^{1/2} = \sum_{n=1}^\infty \left(\frac{1}{n^4}\right)^{1/2} = \sum_{n=1}^\infty \frac{1}{n^2}.
\]
The series \(\sum_{n=1}^\infty \frac{1}{n^2}\) converges (it is a p-series with \(p = 2 > 1\)). Therefore, \(\sum_{n=1}^\infty \|x_n\|_2\) converges, and \(\sum_{n=1}^\infty x_n\) is absolutely convergent in \(\ell^2\).

By the theorem, \(\sum_{n=1}^\infty x_n\) converges in \(\ell^2\).

## What's Next

Having explored the fundamental concepts of normed spaces and Banach spaces, the next steps in your journey through functional analysis might include:

1. **Hilbert Spaces:** These are Banach spaces with an additional inner product structure, allowing for a geometric interpretation and the use of orthogonal projections. Key topics include the Riesz representation theorem, orthonormal bases, and the spectral theorem for self-adjoint operators.

2. **Operator Theory:** This branch of functional analysis studies linear operators between Banach spaces. Important concepts include bounded and unbounded operators, the spectrum of an operator, and the study of compact and Fredholm operators.

3. **Functional Analysis on Locally Convex Spaces:** These are generalizations of Banach spaces where the topology is defined by a family of seminorms. Key topics include the Hahn-Banach theorem, the Krein-Milman theorem, and the theory of distributions.

4. **Applications in Partial Differential Equations (PDEs):** Functional analysis provides a powerful framework for studying PDEs. Topics include the Lax-Milgram theorem, Sobolev spaces, and the theory of weak solutions.

5. **Measure Theory and Integration:** A deeper understanding of measure theory and integration is essential for advanced topics in functional analysis, including the study of \(L^p\) spaces and the Radon-Nikodym theorem.

By delving into these topics, you will gain a comprehensive understanding of the rich and diverse landscape of functional analysis, equipping you with the tools to tackle a wide range of mathematical problems.

---

*This is Part 2 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 1 — Metric Spaces](/en/functional-analysis/01-metric-spaces/)*

*Next: [Part 3 — Hilbert Spaces](/en/functional-analysis/03-hilbert-spaces/)*
