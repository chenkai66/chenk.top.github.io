---
title: "Inner Product Spaces and Hilbert Spaces"
date: 2021-03-08 09:00:00
tags:
  - functional-analysis
  - hilbert-spaces
  - inner-products
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
disableNunjucks: true
series_order: 2
series_total: 6
translationKey: "functional-analysis-2"
description: "Hilbert spaces bring geometry to infinite dimensions — orthogonality, projections, and Fourier series all live here."
---

## Geometry in infinite dimensions

Banach spaces have distance and linear structure, but no notion of angle. You can't talk about orthogonality in $\ell^1$ or $C[0,1]$ (with sup norm) in any natural way. To get geometry — perpendicularity, projections, least-squares approximation — you need an inner product.

## Inner product spaces

An **inner product** on a vector space $X$ over $\mathbb{C}$ is a map $\langle \cdot, \cdot \rangle: X \times X \to \mathbb{C}$ satisfying:

$$\langle x, x \rangle \ge 0, \quad \langle x, x \rangle = 0 \iff x = 0,$$
$$\langle x, y \rangle = \overline{\langle y, x \rangle},$$
$$\langle \alpha x + \beta y, z \rangle = \alpha \langle x, z \rangle + \beta \langle y, z \rangle.$$

Every inner product induces a norm: $\|x\| = \sqrt{\langle x, x \rangle}$. But not every norm comes from an inner product.

**The parallelogram law test.** A norm $\|\cdot\|$ comes from an inner product if and only if:

$$\|x + y\|^2 + \|x - y\|^2 = 2\|x\|^2 + 2\|y\|^2 \quad \forall\, x, y.$$

The $\ell^2$ norm satisfies this. The $\ell^1$ and $\ell^\infty$ norms don't. So among the $\ell^p$ spaces, only $\ell^2$ is an inner product space. This is why $\ell^2$ is special.

**Example 1.** The space $\ell^2$ with:

$$\langle x, y \rangle = \sum_{n=1}^\infty x_n \overline{y_n}.$$

**Example 2.** The space $L^2[0,1]$ with:

$$\langle f, g \rangle = \int_0^1 f(t)\overline{g(t)}\, dt.$$

## The Cauchy-Schwarz inequality

The single most important inequality in all of analysis:

$$|\langle x, y \rangle| \le \|x\|\,\|y\|,$$

with equality iff $x$ and $y$ are linearly dependent.

*Proof.* For $y \ne 0$, set $\alpha = \langle x, y \rangle / \|y\|^2$. Then $0 \le \|x - \alpha y\|^2 = \|x\|^2 - |\langle x, y \rangle|^2 / \|y\|^2$. Rearrange. $\square$

This implies the triangle inequality for the induced norm (so the norm is actually a norm), and it gives us angles: define $\cos\theta = \text{Re}\,\langle x, y \rangle / (\|x\|\,\|y\|)$.

## Hilbert spaces

A **Hilbert space** is a complete inner product space:

$$\text{Hilbert space} = \text{Banach space with inner product}.$$

Both $\ell^2$ and $L^2[0,1]$ are Hilbert spaces. The space of finite sequences with the $\ell^2$ norm is an inner product space but not Hilbert (not complete).

## Orthogonality and projections

Vectors $x, y$ are **orthogonal** (written $x \perp y$) if $\langle x, y \rangle = 0$. A set $S$ is **orthonormal** if its elements are pairwise orthogonal unit vectors.

**Theorem (Orthogonal projection).** Let $M$ be a closed subspace of a Hilbert space $H$. For every $x \in H$, there exists a unique $m_0 \in M$ such that:

$$\|x - m_0\| = \inf_{m \in M} \|x - m\|.$$

Moreover, $m_0$ is characterized by $(x - m_0) \perp M$.

*Proof sketch.* Let $d = \inf_{m \in M} \|x - m\|$. Take a minimizing sequence $(m_n)$. The parallelogram law gives:

$$\|m_n - m_k\|^2 = 2\|m_n - x\|^2 + 2\|m_k - x\|^2 - 4\left\|\frac{m_n + m_k}{2} - x\right\|^2 \le 2\|m_n - x\|^2 + 2\|m_k - x\|^2 - 4d^2.$$

Since both terms on the right approach $2d^2$, the sequence is Cauchy. Completeness gives the limit $m_0 \in M$. Uniqueness and the orthogonality characterization follow from expanding $\|x - m_0 - t\,v\|^2 \ge \|x - m_0\|^2$ for $v \in M$. $\square$

This fails in general Banach spaces. In $\ell^1$, closest points in subspaces need not be unique. The inner product is essential.

**Corollary (Orthogonal decomposition).** $H = M \oplus M^\perp$ where $M^\perp = \{x \in H : x \perp m \ \forall m \in M\}$.

Every $x$ splits uniquely as $x = m + m^\perp$. This is the infinite-dimensional generalization of projecting onto a subspace in $\mathbb{R}^n$.

## Orthonormal bases and Fourier series

An orthonormal set $\{e_n\}$ in $H$ is an **orthonormal basis** (or complete orthonormal system) if the only vector orthogonal to all $e_n$ is zero.

**Theorem (Parseval).** If $\{e_n\}_{n=1}^\infty$ is an orthonormal basis of $H$, then for every $x \in H$:

$$x = \sum_{n=1}^\infty \langle x, e_n \rangle\, e_n, \quad \|x\|^2 = \sum_{n=1}^\infty |\langle x, e_n \rangle|^2.$$

The coefficients $\langle x, e_n \rangle$ are the **Fourier coefficients** of $x$ with respect to the basis.

**Example 3.** In $L^2[0, 2\pi]$, the functions $e_n(t) = \frac{1}{\sqrt{2\pi}} e^{int}$ form an orthonormal basis. Parseval's identity becomes:

$$\frac{1}{2\pi}\int_0^{2\pi} |f(t)|^2\, dt = \sum_{n=-\infty}^{\infty} |\hat{f}(n)|^2,$$

which is the classical Parseval theorem from Fourier analysis.

**Example 4.** In $\ell^2$, the standard basis $e_n = (0, \ldots, 0, 1, 0, \ldots)$ is orthonormal, and the Fourier expansion is trivial: $x = \sum x_n e_n$. The abstract theory generalizes this to arbitrary Hilbert spaces.

## The Riesz representation theorem

**Theorem (Riesz-Frechet).** Every continuous linear functional $f: H \to \mathbb{C}$ on a Hilbert space has the form $f(x) = \langle x, y \rangle$ for a unique $y \in H$, with $\|f\| = \|y\|$.

*Proof sketch.* If $f = 0$, take $y = 0$. Otherwise, $\ker f$ is a closed subspace of codimension 1. Pick $z \perp \ker f$ with $f(z) = 1$. Set $y = z/\|z\|^2$. Then $f(x) = f(x - f(x)z + f(x)z) = f(x)\langle z, z \rangle \cdot \langle x, y\rangle / \langle z, y \rangle$... more cleanly: for any $x$, write $x = (x - f(x)z) + f(x)z$. The first part is in $\ker f$, so $\langle x, z \rangle = f(x)\|z\|^2$, giving $f(x) = \langle x, z/\|z\|^2 \rangle$. $\square$

This identifies $H^* \cong H$ (antilinearly). No other Banach space has this self-duality. It's why Hilbert spaces are the "nicest" infinite-dimensional spaces.

## Separability

A Hilbert space is **separable** if it has a countable orthonormal basis. All separable infinite-dimensional Hilbert spaces are isometrically isomorphic to $\ell^2$. There's essentially one separable Hilbert space up to isomorphism.

This is astonishing. $L^2[0,1]$, $L^2(\mathbb{R})$ (with appropriate measure), the Sobolev space $H^1[0,1]$ — they all look like $\ell^2$ as Hilbert spaces. The differences show up in which operators act on them and which subspaces are natural.

## What's next

With Hilbert space geometry in hand, the next step is to study the maps between spaces: bounded linear operators. These are the infinite-dimensional analogues of matrices, and they behave in ways that would horrify anyone used to finite-dimensional linear algebra.
