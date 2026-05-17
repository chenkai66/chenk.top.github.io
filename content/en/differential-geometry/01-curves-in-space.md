---
title: "Curves in Space — Curvature, Torsion, and the Frenet Frame"
date: 2021-05-01 09:00:00
tags:
  - differential-geometry
  - curves
  - curvature
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
disableNunjucks: true
series_order: 1
series_total: 6
translationKey: "differential-geometry-1"
description: "How curves bend and twist in three-dimensional space, captured by curvature and torsion."
---

A curve is the simplest object in differential geometry: a one-dimensional thing living in three-dimensional space. Yet even here the mathematics is rich enough to occupy us for a while.

## Parametrized curves

A smooth curve in $\mathbb{R}^3$ is a map $\alpha: I \to \mathbb{R}^3$ where $I$ is an open interval and $\alpha$ is infinitely differentiable. We write $\alpha(t) = (x(t), y(t), z(t))$.

The velocity vector is $\alpha'(t)$. A curve is **regular** if $\alpha'(t) \neq 0$ for all $t$. Regularity means the curve never stops moving — no cusps, no backtracking.

**Example.** The circular helix $\alpha(t) = (a\cos t,\, a\sin t,\, bt)$ with $a > 0$ has velocity $\alpha'(t) = (-a\sin t,\, a\cos t,\, b)$ and speed $|\alpha'(t)| = \sqrt{a^2 + b^2}$. It is regular for all $t$.

## Arc length and reparametrization

The arc length from $t_0$ to $t$ is

$$s(t) = \int_{t_0}^{t} |\alpha'(u)|\, du.$$

Since $ds/dt = |\alpha'(t)| > 0$ for a regular curve, we can invert $s(t)$ and reparametrize by arc length. A curve parametrized by arc length satisfies $|\alpha'(s)| = 1$ — the unit-speed condition.

Arc-length parametrization strips away the "how fast" and leaves only the "which way." All intrinsic geometry of the curve depends only on this reparametrized version.

## Curvature

For a unit-speed curve $\alpha(s)$, define the **curvature**:

$$\kappa(s) = |\alpha''(s)|.$$

Curvature measures how sharply the curve bends. A straight line has $\kappa = 0$ everywhere. A circle of radius $r$ has $\kappa = 1/r$ — smaller circles bend harder.

For a curve not parametrized by arc length, the formula becomes:

$$\kappa = \frac{|\alpha' \times \alpha''|}{|\alpha'|^3}.$$

**Example.** For the helix $\alpha(t) = (a\cos t, a\sin t, bt)$: we compute $\alpha' \times \alpha'' = (b\sin t,\, -b\cos t,\, a)$ (after working through the cross product), giving $|\alpha' \times \alpha''| = \sqrt{a^2 + b^2}$. Since $|\alpha'|^3 = (a^2+b^2)^{3/2}$, the curvature is

$$\kappa = \frac{a}{a^2 + b^2}.$$

Constant curvature. The helix bends the same amount everywhere.

## The Frenet frame

Assume $\kappa(s) > 0$. Define three orthonormal vectors at each point:

$$\mathbf{T}(s) = \alpha'(s), \quad \mathbf{N}(s) = \frac{\alpha''(s)}{|\alpha''(s)|}, \quad \mathbf{B}(s) = \mathbf{T}(s) \times \mathbf{N}(s).$$

Here $\mathbf{T}$ is the unit tangent, $\mathbf{N}$ is the principal normal (pointing toward the center of curvature), and $\mathbf{B}$ is the binormal. Together $\{\mathbf{T}, \mathbf{N}, \mathbf{B}\}$ form the **Frenet frame** — a moving orthonormal basis attached to the curve.

The Frenet frame is a coordinate system that rides along with the curve, rotating as the curve bends and twists.

## Torsion

The **torsion** $\tau(s)$ measures how the osculating plane rotates about the tangent:

$$\tau(s) = -\mathbf{B}'(s) \cdot \mathbf{N}(s).$$

A plane curve has $\tau = 0$: it never leaves its osculating plane. Nonzero torsion means the curve is genuinely three-dimensional.

For the helix: $\tau = b/(a^2 + b^2)$. Constant torsion, just like its constant curvature.

## The Frenet-Serret formulas

The derivatives of the frame vectors are governed by:

$$\begin{aligned}
\mathbf{T}' &= \kappa \mathbf{N}, \\
\mathbf{N}' &= -\kappa \mathbf{T} + \tau \mathbf{B}, \\
\mathbf{B}' &= -\tau \mathbf{N}.
\end{aligned}$$

These three equations encode everything about the local geometry of the curve. They say: the tangent turns toward the normal (curvature), the normal is pulled back by the tangent and pushed by the binormal (both curvature and torsion), and the binormal drifts toward the normal (torsion).

**Theorem (Fundamental theorem of curves).** Given smooth functions $\kappa(s) > 0$ and $\tau(s)$, there exists a unit-speed curve $\alpha(s)$ in $\mathbb{R}^3$ with curvature $\kappa$ and torsion $\tau$. This curve is unique up to rigid motion (rotation and translation).

*Proof sketch.* The Frenet-Serret equations form a linear ODE system for the frame $\{\mathbf{T}, \mathbf{N}, \mathbf{B}\}$. By the existence-uniqueness theorem for ODEs, given initial conditions (a starting point and initial frame), there is a unique solution. The curve is recovered by integrating $\mathbf{T}$.

This is remarkable: two scalar functions completely determine the shape of a space curve.

## Worked example: characterizing helices

**Theorem (Lancret).** A curve with $\kappa > 0$ is a generalized helix (its tangent makes a constant angle with a fixed direction) if and only if $\tau/\kappa$ is constant.

*Proof sketch.* If $\mathbf{T} \cdot \mathbf{v} = \cos\theta$ for a fixed unit vector $\mathbf{v}$, differentiate: $\kappa \mathbf{N} \cdot \mathbf{v} = 0$, so $\mathbf{v}$ lies in the $\mathbf{T}$-$\mathbf{B}$ plane. Write $\mathbf{v} = \cos\theta\, \mathbf{T} + \sin\theta\, \mathbf{B}$. Differentiate again and use the Frenet equations to get $\kappa\cos\theta - \tau\sin\theta = 0$, hence $\tau/\kappa = \cot\theta = \text{const}$.

Our circular helix satisfies $\tau/\kappa = b/a$, confirming Lancret's theorem.

## What's next

We move from one-dimensional curves to two-dimensional surfaces. The first fundamental form will tell us how to measure lengths and angles on a surface — the beginning of intrinsic geometry.
