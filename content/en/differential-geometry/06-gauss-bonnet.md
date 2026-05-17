---
title: "The Gauss-Bonnet Theorem — Where Geometry Meets Topology"
date: 2021-06-05 09:00:00
tags:
  - differential-geometry
  - gauss-bonnet
  - topology
  - euler-characteristic
  - mathematics
categories: Mathematics
series: differential-geometry
lang: en
mathjax: true
disableNunjucks: true
series_order: 6
series_total: 6
translationKey: "differential-geometry-6"
description: "The Gauss-Bonnet theorem connects total curvature to topology via the Euler characteristic."
---

The Gauss-Bonnet theorem is one of the most beautiful results in all of mathematics. It says: integrate the Gaussian curvature over a closed surface, and you get a topological invariant — a number that depends only on the "shape" of the surface in the rubber-sheet sense, not on its specific geometry.

## The local Gauss-Bonnet theorem

Let $R$ be a region on a surface $S$ bounded by a simple closed piecewise-smooth curve $\partial R$. Suppose $\partial R$ consists of smooth arcs meeting at exterior angles $\theta_1, \ldots, \theta_k$. Then:

$$\iint_R K\,dA + \int_{\partial R} \kappa_g\,ds + \sum_{i=1}^k \theta_i = 2\pi$$

where $\kappa_g$ is the geodesic curvature of the boundary and the integral is taken with the region on the left.

**Example.** A geodesic triangle ($\kappa_g = 0$ on each side, three vertices with exterior angles $\theta_i = \pi - \alpha_i$):

$$\iint_T K\,dA + 0 + \sum(\pi - \alpha_i) = 2\pi$$

which gives $\alpha_1 + \alpha_2 + \alpha_3 = \pi + \iint_T K\,dA$. On a sphere, angles sum to more than $\pi$. On a saddle, less.

## The global Gauss-Bonnet theorem

**Theorem.** Let $S$ be a compact oriented surface without boundary. Then:

$$\iint_S K\,dA = 2\pi\chi(S)$$

where $\chi(S)$ is the **Euler characteristic** of $S$.

The Euler characteristic is a topological invariant: $\chi = 2 - 2g$ where $g$ is the genus (number of handles). So:
- Sphere ($g = 0$): $\chi = 2$, total curvature $= 4\pi$.
- Torus ($g = 1$): $\chi = 0$, total curvature $= 0$.
- Double torus ($g = 2$): $\chi = -2$, total curvature $= -4\pi$.

*Proof sketch.* Triangulate $S$ into geodesic triangles. For each triangle $T_j$, the local Gauss-Bonnet gives $\iint_{T_j}K\,dA = (\alpha_1^j + \alpha_2^j + \alpha_3^j) - \pi$. Sum over all triangles. Interior edges are shared and their geodesic curvature contributions cancel. At each interior vertex, angles sum to $2\pi$. Careful bookkeeping gives:

$$\iint_S K\,dA = 2\pi V - \pi F - \sum(\text{edge cancellations}) = 2\pi(V - E + F) = 2\pi\chi(S)$$

where $V$, $E$, $F$ are vertices, edges, faces of the triangulation.

## Verification: the sphere

For a sphere of radius $R$: $K = 1/R^2$ everywhere, $A = 4\pi R^2$. So $\iint K\,dA = 4\pi R^2 \cdot (1/R^2) = 4\pi = 2\pi \cdot 2 = 2\pi\chi(S^2)$. Checks out.

## Verification: the torus

The torus has regions of positive curvature (the outside) and negative curvature (the inside). Gauss-Bonnet demands they exactly cancel: $\iint K\,dA = 0 = 2\pi\chi(T^2)$. We verified this directly in Chapter 3 — the total curvature of the torus is zero regardless of its specific proportions $R$ and $r$.

## Why this is remarkable

The left side — $\iint K\,dA$ — depends on the Riemannian metric. Change the metric (stretch, compress, bend the surface) and $K$ changes at every point.

The right side — $2\pi\chi(S)$ — is purely topological. It does not change under any smooth deformation.

Gauss-Bonnet says: no matter how you deform the metric, the total curvature is locked in by topology. Positive curvature in one region forces negative curvature elsewhere (unless the surface is a sphere). The integral is rigid.

## Applications

### Topology constrains geometry

A compact surface with $K > 0$ everywhere must have $\chi > 0$, hence must be a sphere (or $\mathbb{RP}^2$ if non-orientable). There is no closed surface of genus $\geq 1$ with everywhere-positive Gaussian curvature.

### The hairy ball theorem

On $S^2$, every smooth vector field has a zero. This follows from the Poincare-Hopf theorem (a generalization of Gauss-Bonnet for vector fields): the sum of indices of zeros of any vector field equals $\chi(S^2) = 2 \neq 0$.

### Geometry constrains topology

If a surface admits a flat metric ($K = 0$ everywhere), then $\chi = 0$, so it must be a torus or Klein bottle. Negative curvature forces $\chi < 0$, meaning genus $\geq 2$.

## The Chern-Gauss-Bonnet theorem

The Gauss-Bonnet theorem generalizes to higher dimensions. For a compact oriented Riemannian manifold $M$ of even dimension $2n$:

$$\int_M \text{Pf}(\Omega) = (2\pi)^n\chi(M)$$

where $\text{Pf}(\Omega)$ is the Pfaffian of the curvature form. In dimension 2, the Pfaffian reduces to $K\,dA/(2\pi)$ and we recover the classical theorem.

This is a cornerstone of the Atiyah-Singer index theorem — one of the deepest results of 20th-century mathematics, connecting analysis, geometry, and topology.

## A worked example: the Mobius strip

The Mobius strip is non-orientable, so the global Gauss-Bonnet for closed surfaces does not directly apply. But the version with boundary does. A Mobius strip has $\chi = 0$ (it deformation-retracts to a circle). For a flat Mobius strip (made from a paper strip), $K = 0$, so

$$0 + \int_{\partial M}\kappa_g\,ds + 0 = 2\pi\chi(M) = 0.$$

The boundary of the Mobius strip is a single closed curve whose total geodesic curvature is zero — it winds around twice, and the contributions cancel.

## Where to go from here

This series covered the classical arc: curves, surfaces, intrinsic geometry, and the bridge to topology. The natural continuation is Riemannian geometry proper (curvature tensors, comparison theorems, Ricci flow), Lie groups and symmetric spaces, or the gauge theory and fiber bundles that underpin modern physics.

The Gauss-Bonnet theorem is not an endpoint. It is a prototype — the first in a family of index theorems that connect local differential data to global topological invariants. That pattern — local analysis constraining global structure — is perhaps the deepest theme in all of geometry.
