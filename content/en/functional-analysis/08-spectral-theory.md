---
title: "Functional Analysis (8): Spectral Theory — Decomposing Operators"
date: 2021-10-15 09:00:00
tags:
  - functional-analysis
  - spectral-theory
  - operators
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
description: "The spectrum generalizes eigenvalues to infinite dimensions — the spectral theorem for bounded self-adjoint operators and continuous functional calculus give us a complete decomposition."
disableNunjucks: true
series_order: 8
series_total: 12
translationKey: "functional-analysis-8"
---

In finite dimensions, every Hermitian matrix can be diagonalized: there exists an orthonormal basis of eigenvectors, and the matrix becomes a diagonal matrix of eigenvalues. This is the spectral theorem for matrices, and it is the foundation of principal component analysis, quantum mechanics, and half of applied mathematics.

In infinite dimensions, the story is more subtle. A bounded self-adjoint operator on a Hilbert space may have no eigenvalues at all — consider the multiplication operator $(Mf)(x) = xf(x)$ on $L^2([0,1])$, which has no eigenfunctions but clearly has "spectral content" spread over $[0,1]$. The spectrum $\sigma(T)$ generalizes the set of eigenvalues, the resolvent $(T - \lambda I)^{-1}$ generalizes the inverse, and the spectral theorem replaces diagonalization with a **multiplication operator representation** or, equivalently, a **projection-valued measure**.

This article develops the general spectral theory for bounded operators on Banach and Hilbert spaces, culminating in the spectral theorem for bounded self-adjoint operators and the continuous functional calculus.

---

## Beyond Eigenvalues: The Spectrum $\sigma(T)$

**Definition.** Let $X$ be a complex Banach space and $T \in B(X)$. The **resolvent set** of $T$ is

$$\rho(T) = \{\lambda \in \mathbb{C} : (\lambda I - T) \text{ is bijective with bounded inverse}\}.$$

![Decomposition of the spectrum of an operator](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/08-spectral-theory/fa_fig5_spectrum.png)


The **spectrum** of $T$ is $\sigma(T) = \mathbb{C} \setminus \rho(T)$.

For $\lambda \in \rho(T)$, the **resolvent operator** is $R(\lambda, T) = (\lambda I - T)^{-1} \in B(X)$.

Note that bijectivity of a bounded operator between Banach spaces automatically gives a bounded inverse (by the Bounded Inverse Theorem from Article 6). So $\lambda \in \rho(T)$ if and only if $\lambda I - T$ is bijective.

**First properties of the spectrum:**

1. **$\sigma(T)$ is nonempty** (for complex Banach spaces). This is a consequence of the Liouville theorem for vector-valued analytic functions — if $\sigma(T)$ were empty, $R(\cdot, T)$ would be an entire $B(X)$-valued function vanishing at infinity, hence identically zero, which is impossible.

2. **$\sigma(T)$ is compact**, contained in the closed disk $\{|\lambda| \leq \|T\|\}$. For $|\lambda| > \|T\|$, the Neumann series $(\lambda I - T)^{-1} = \frac{1}{\lambda}\sum_{n=0}^\infty (T/\lambda)^n$ converges in operator norm.

3. **$\sigma(T)$ is closed** (equivalently, $\rho(T)$ is open). If $\lambda_0 \in \rho(T)$ and $|\lambda - \lambda_0| < \|R(\lambda_0, T)\|^{-1}$, then $\lambda \in \rho(T)$, via the Neumann series expansion $R(\lambda, T) = R(\lambda_0, T) \sum_{n=0}^\infty [(\lambda_0 - \lambda)R(\lambda_0, T)]^n$.

**Example 1 (Spectrum of a diagonal operator).** Let $T: \ell^2 \to \ell^2$ be defined by $T(e_n) = \alpha_n e_n$ where $(\alpha_n)$ is a bounded sequence. Then $\sigma(T) = \overline{\{\alpha_n : n \geq 1\}}$ (the closure of the set of diagonal entries). The eigenvalues are exactly the $\alpha_n$, and if $\lambda$ is an accumulation point of the $\alpha_n$ but not equal to any $\alpha_n$, then $\lambda \in \sigma(T)$ but $\lambda$ is not an eigenvalue — it belongs to the continuous spectrum.

**Example 2 (Spectrum of the shift operator).** The right shift $S: \ell^2 \to \ell^2$, $S(x_1, x_2, \ldots) = (0, x_1, x_2, \ldots)$, has $\sigma(S) = \overline{\mathbb{D}} = \{|\lambda| \leq 1\}$ (the closed unit disk). There are no eigenvalues (if $Sx = \lambda x$, then $0 = \lambda x_1$, $x_1 = \lambda x_2$, etc., forcing $x = 0$ for $|\lambda| \leq 1$). The adjoint $S^*$ (left shift) has $\sigma(S^*) = \overline{\mathbb{D}}$ as well, but every $\lambda$ with $|\lambda| < 1$ **is** an eigenvalue: $S^* v_\lambda = \lambda v_\lambda$ where $v_\lambda = (1, \lambda, \lambda^2, \ldots) \in \ell^2$.

---

## Point, Continuous, and Residual Spectrum

The spectrum decomposes into three disjoint parts based on the behavior of $\lambda I - T$.

**Definition.** For $T \in B(X)$ and $\lambda \in \sigma(T)$:

- **Point spectrum** $\sigma_p(T)$: $\lambda I - T$ is not injective (i.e., $\lambda$ is an eigenvalue).
- **Continuous spectrum** $\sigma_c(T)$: $\lambda I - T$ is injective, has dense range, but the range is not all of $X$ (equivalently, the inverse exists on a dense domain but is unbounded).
- **Residual spectrum** $\sigma_r(T)$: $\lambda I - T$ is injective, but the range is not dense.

**Properties for specific classes:**

| Operator class | $\sigma_r(T)$ |
|---|---|
| Self-adjoint ($T = T^*$) | Empty |
| Normal ($TT^* = T^*T$) | Empty |
| Unitary | Empty |
| General bounded | Can be nonempty |

For self-adjoint operators, the absence of residual spectrum follows from the fact that $\overline{\text{range}(\lambda I - T)} = \ker(\overline{\lambda} I - T^*)^\perp = \ker(\lambda I - T)^\perp$ (using $T = T^*$ and $\lambda \in \mathbb{R}$). So if $\lambda I - T$ is injective, the range is dense.

**Example 3 (All three parts present).** Consider the operator on $\ell^2(\mathbb{Z})$ defined by $(Tx)_n = a_n x_n$ where $a_n = 1/n$ for $n \geq 1$, $a_0 = 0$, and $a_n = i$ for $n \leq -1$. Then:
- $\sigma_p(T) = \{0, i\} \cup \{1/n : n \geq 1\}$ (eigenvalues are the $a_n$).
- $0$ is an eigenvalue (with eigenvector $e_0$) and also an accumulation point of eigenvalues $1/n$.
- More exotic decompositions arise for operators on non-separable spaces.

For a cleaner example: the right shift on $\ell^2(\mathbb{N})$ has $\sigma_p(S) = \emptyset$, $\sigma_r(S) = \{|\lambda| < 1\}$ (the range of $\lambda I - S$ misses certain vectors), and $\sigma_c(S) = \{|\lambda| = 1\}$.

**Why the residual spectrum vanishes for self-adjoint operators.** This is worth proving carefully, as it is fundamental. Let $T = T^*$ and suppose $\lambda \in \sigma(T)$ but $\lambda \notin \sigma_p(T)$, i.e., $\lambda I - T$ is injective. We claim the range is dense (so $\lambda \in \sigma_c(T)$, not $\sigma_r(T)$).

Since $T = T^*$, all spectral values are real ($\lambda \in \mathbb{R}$, proven in Article 7, Step 1 of the spectral theorem proof). Now, $\overline{\text{range}(\lambda I - T)}^\perp = \ker((\lambda I - T)^*) = \ker(\lambda I - T^*) = \ker(\lambda I - T) = \{0\}$ (since $\lambda I - T$ is injective). Therefore the range is dense.

This means the spectrum of a self-adjoint operator has a clean dichotomy: every spectral point is either an eigenvalue (point spectrum) or belongs to the continuous spectrum. There is no residual spectrum, and this greatly simplifies the spectral analysis.

**Example 4 (Spectrum of a weighted shift).** Define $T: \ell^2 \to \ell^2$ by $T(e_n) = w_n e_{n+1}$ where $(w_n)$ is a bounded sequence of positive weights. This is a weighted right shift. The spectral properties depend on the weights:

- If $w_n = 1$ for all $n$ (unweighted shift), then $\sigma(T) = \overline{\mathbb{D}}$, $\sigma_p(T) = \emptyset$.
- If $w_n \to 0$, then $T$ is compact, and $\sigma(T) = \{0\}$ (since $r(T) = \lim \|T^n\|^{1/n} = \lim (\sup_k \prod_{j=k}^{k+n-1} w_j)^{1/n}$, which tends to $0$ when the $w_n$ decay).
- If $w_n = 1/n$, the operator is compact and quasinilpotent (like the Volterra operator).

---

## The Resolvent and Spectral Radius Formula

The resolvent operator $R(\lambda, T) = (\lambda I - T)^{-1}$, defined for $\lambda \in \rho(T)$, is the fundamental analytic tool of spectral theory.

**Resolvent identity.** For $\lambda, \mu \in \rho(T)$:

$$R(\lambda, T) - R(\mu, T) = (\mu - \lambda) R(\lambda, T) R(\mu, T).$$

**Proof.** $R(\lambda, T) - R(\mu, T) = R(\lambda,T)[(\mu I - T) - (\lambda I - T)]R(\mu, T) = (\mu - \lambda)R(\lambda,T)R(\mu,T)$. $\square$

This shows that $R(\lambda, T)$ and $R(\mu, T)$ commute, and that $\lambda \mapsto R(\lambda, T)$ is a $B(X)$-valued analytic function on $\rho(T)$.

**Theorem (Spectral Radius Formula / Gelfand's Formula).** For $T \in B(X)$, the **spectral radius**

$$r(T) = \sup\{|\lambda| : \lambda \in \sigma(T)\}$$

satisfies

$$r(T) = \lim_{n \to \infty} \|T^n\|^{1/n} = \inf_{n \geq 1} \|T^n\|^{1/n}.$$

**Proof.** We establish both inequalities.

*Step 1: $r(T) \leq \inf_n \|T^n\|^{1/n}$.* If $\lambda \in \sigma(T)$, then $\lambda^n \in \sigma(T^n)$ (since $\lambda^n I - T^n = (\lambda I - T)\sum_{k=0}^{n-1} \lambda^{n-1-k} T^k$; if $\lambda^n I - T^n$ were invertible, so would $\lambda I - T$ be). Therefore $|\lambda|^n \leq \|T^n\|$, giving $|\lambda| \leq \|T^n\|^{1/n}$ for all $n$.

*Step 2: $\limsup_n \|T^n\|^{1/n} \leq r(T)$.* For $|\lambda| > r(T)$, we have $\lambda \in \rho(T)$, and the Neumann series $R(\lambda, T) = \sum_{n=0}^\infty T^n / \lambda^{n+1}$ converges. For any $f \in X^*$, the function $\lambda \mapsto f(R(\lambda, T))$ is analytic for $|\lambda| > r(T)$ with Laurent expansion $\sum_{n=0}^\infty f(T^n)/\lambda^{n+1}$. By the root test for Laurent series, $\limsup_n |f(T^n)|^{1/n} \leq r(T)$ for each $f \in X^*$.

Now apply the Uniform Boundedness Principle: the family of operators $\{T^n / r^n\}_{n \geq 1}$ (for any $r > r(T)$) is pointwise bounded in $B(X)$ (since for each $x$, $f(T^n x) / r^n \to 0$ for all $f$, giving $\|T^n x\|/r^n$ bounded). By UBP, $\sup_n \|T^n\|/r^n < \infty$, so $\limsup_n \|T^n\|^{1/n} \leq r$. Since $r > r(T)$ was arbitrary, $\limsup \|T^n\|^{1/n} \leq r(T)$.

Combining with Step 1: $r(T) \leq \inf_n \|T^n\|^{1/n} \leq \liminf_n \|T^n\|^{1/n} \leq \limsup_n \|T^n\|^{1/n} \leq r(T)$, so equality holds throughout and the limit exists. $\square$

**Example 4 (Spectral radius computation).** For the Volterra operator on $L^2([0,1])$ defined by $(Vf)(x) = \int_0^x f(t)\,dt$, one can compute $\|V^n\| = 1/n!$ (by induction: the $n$-fold iterated integral has kernel $(x-t)^{n-1}/(n-1)!$ on the triangle $0 \leq t \leq x \leq 1$). Therefore $r(V) = \lim (1/n!)^{1/n} = 0$. The spectrum is $\sigma(V) = \{0\}$ — the Volterra operator is **quasinilpotent**. Despite having spectral radius zero, $V$ is not nilpotent ($V^n \neq 0$ for any $n$) and not even compact (wait — actually it is compact; the point is that $\sigma(V) = \{0\}$ with $0$ not being an eigenvalue, so $\sigma(V) = \sigma_c(V) = \{0\}$).

Actually, let us verify: $(Vf)(x) = 0$ for all $x$ implies $\int_0^x f(t)\,dt = 0$ for all $x$, so $f = 0$ a.e. Hence $0$ is not an eigenvalue. The Volterra operator has purely continuous spectrum at the single point $\{0\}$.

**Example 6 (Spectral radius vs. operator norm).** The spectral radius can be strictly smaller than the operator norm. Consider the nilpotent matrix $T = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$, which has $\sigma(T) = \{0\}$, so $r(T) = 0$, but $\|T\| = 1$.

For a more dramatic infinite-dimensional example, consider the Volterra operator: $r(V) = 0$ but $\|V\| = 2/\pi$ (this can be computed by finding the norm of the self-adjoint operator $V^*V$).

The gap between $r(T)$ and $\|T\|$ measures the "non-normality" of $T$. For normal operators ($TT^* = T^*T$), we will prove that $r(T) = \|T\|$ always, so there is no gap. The spectral radius formula $r(T) = \lim \|T^n\|^{1/n}$ shows that powers of $T$ eventually "feel" only the spectral radius, regardless of the operator norm.

**Functional calculus for polynomials and rational functions.** Before developing the full continuous functional calculus, note that we can already define $p(T)$ for any polynomial $p$ and $r(T) = p(T)/q(T)$ for rational functions whose poles lie outside $\sigma(T)$. The key facts are:

- **Spectral mapping theorem for polynomials:** $\sigma(p(T)) = p(\sigma(T)) = \{p(\lambda) : \lambda \in \sigma(T)\}$.
- **Proof:** $\mu \in \sigma(p(T))$ iff $p(T) - \mu I$ is not invertible. Factor $p(\lambda) - \mu = c\prod_j (\lambda - \alpha_j)$, so $p(T) - \mu I = c\prod_j (T - \alpha_j I)$. This is non-invertible iff some $T - \alpha_j I$ is non-invertible, iff some $\alpha_j \in \sigma(T)$, iff $\mu = p(\alpha_j) \in p(\sigma(T))$.

---

## The Spectral Theorem for Bounded Self-Adjoint Operators

For compact self-adjoint operators (Article 7), we had a diagonalization with respect to an orthonormal eigenbasis. For general bounded self-adjoint operators, eigenvalues may not exist, so we need a more sophisticated decomposition.

**Theorem (Spectral Theorem, Multiplication Operator Form).** Let $T$ be a bounded self-adjoint operator on a separable Hilbert space $H$. Then there exist a measure space $(\Omega, \mu)$, a bounded measurable function $\varphi: \Omega \to \mathbb{R}$, and a unitary operator $U: H \to L^2(\Omega, \mu)$ such that

$$UTU^{-1} = M_\varphi,$$

where $M_\varphi$ is the multiplication operator $(M_\varphi g)(\omega) = \varphi(\omega)g(\omega)$.

Moreover, $\sigma(T) = \text{ess-range}(\varphi)$, and $\lambda$ is an eigenvalue of $T$ if and only if $\mu(\{\varphi = \lambda\}) > 0$.

This theorem says that every bounded self-adjoint operator is **unitarily equivalent to a multiplication operator**. Multiplication operators are the infinite-dimensional analogs of diagonal matrices, and this result is the proper generalization of matrix diagonalization.

Rather than proving this in its most general form (which requires the theory of C*-algebras or direct integral decompositions), we develop it through the continuous functional calculus.

---

## The Continuous Functional Calculus

The key idea is: if $T$ is self-adjoint with $\sigma(T) \subset \mathbb{R}$, we want to define $f(T)$ for continuous functions $f: \sigma(T) \to \mathbb{C}$, extending the obvious definition for polynomials.

**Theorem (Continuous Functional Calculus).** Let $T$ be a bounded self-adjoint operator on a Hilbert space $H$. There exists a unique isometric $*$-homomorphism

$$\Phi: C(\sigma(T)) \to B(H)$$

such that $\Phi(1) = I$ and $\Phi(\text{id}) = T$ (where $\text{id}(\lambda) = \lambda$). We write $f(T) = \Phi(f)$.

The map $\Phi$ satisfies:
1. **Isometry:** $\|f(T)\| = \|f\|_{\infty, \sigma(T)}$.
2. **$*$-homomorphism:** $(\alpha f + \beta g)(T) = \alpha f(T) + \beta g(T)$, $(fg)(T) = f(T)g(T)$, and $\overline{f}(T) = f(T)^*$.
3. **Spectral mapping:** $\sigma(f(T)) = f(\sigma(T)) = \{f(\lambda) : \lambda \in \sigma(T)\}$.
4. **Positivity:** if $f \geq 0$ on $\sigma(T)$, then $f(T) \geq 0$ (i.e., $\langle f(T)x, x \rangle \geq 0$ for all $x$).

**Construction.** For polynomials $p(\lambda) = \sum a_k \lambda^k$, define $p(T) = \sum a_k T^k$. The key estimate is:

$$\|p(T)\| = r(p(T)) = \sup\{|\mu| : \mu \in \sigma(p(T))\} = \sup\{|p(\lambda)| : \lambda \in \sigma(T)\} = \|p\|_{\infty, \sigma(T)}.$$

The first equality uses the fact that $p(T)$ is normal (actually self-adjoint if $p$ has real coefficients), so $\|p(T)\| = r(p(T))$. The second equality is the spectral mapping theorem for polynomials: $\sigma(p(T)) = p(\sigma(T))$.

This shows the map $p \mapsto p(T)$ is an isometry from polynomials (with the sup-norm on $\sigma(T)$) to $B(H)$. By the Stone-Weierstrass theorem, polynomials are dense in $C(\sigma(T))$ (since $\sigma(T) \subset \mathbb{R}$ is compact). The isometry extends uniquely to all of $C(\sigma(T))$. $\square$

**Example 5 (Square root of a positive operator).** If $T \geq 0$ (self-adjoint with $\sigma(T) \subset [0, \infty)$), then $f(\lambda) = \sqrt{\lambda}$ is continuous on $\sigma(T)$, and $T^{1/2} = f(T)$ is the unique positive square root: $(T^{1/2})^2 = (\sqrt{\cdot})^2(T) = \text{id}(T) = T$, and $T^{1/2} \geq 0$.

**Example 6 (Spectral projections via approximation).** For $\lambda_0 \in \sigma(T)$ isolated (e.g., an eigenvalue of a compact self-adjoint operator), the characteristic function $\chi_{\{\lambda_0\}}$ is continuous on $\sigma(T)$ (since $\lambda_0$ is isolated). Then $P_{\lambda_0} = \chi_{\{\lambda_0\}}(T)$ is the orthogonal projection onto the eigenspace $\ker(T - \lambda_0 I)$. For non-isolated spectral points, we approximate $\chi$ by continuous functions, leading to the spectral measure.

---

## Spectral Measures and the Projection-Valued Measure Approach

The continuous functional calculus extends to a **Borel functional calculus** by passing from $C(\sigma(T))$ to the space of bounded Borel measurable functions on $\sigma(T)$. This extension is what produces spectral projections for arbitrary Borel sets.

**Definition.** A **projection-valued measure** (PVM) on a measurable space $(\Omega, \mathcal{B})$ with values in $B(H)$ is a map $E: \mathcal{B} \to B(H)$ such that:

1. Each $E(B)$ is an orthogonal projection.
2. $E(\emptyset) = 0$, $E(\Omega) = I$.
3. $E(B_1 \cap B_2) = E(B_1)E(B_2)$.
4. For disjoint $(B_n)$, $E(\bigcup B_n) = \sum E(B_n)$ (convergence in the strong operator topology).

**Theorem (Spectral Theorem, PVM Form).** For every bounded self-adjoint operator $T$ on $H$, there exists a unique projection-valued measure $E$ on $(\sigma(T), \text{Borel})$ such that

$$T = \int_{\sigma(T)} \lambda \, dE(\lambda).$$

This means: for all $x, y \in H$, $\langle Tx, y \rangle = \int_{\sigma(T)} \lambda \, d\mu_{x,y}(\lambda)$, where $\mu_{x,y}(B) = \langle E(B)x, y \rangle$ is a complex measure. The scalar measures $\mu_{x,y}$ encode all the information about the operator in terms of integration theory.

**How to think about the spectral measure.** The spectral measure $E$ decomposes the Hilbert space $H$ into a "continuous family" of eigenspaces. For a compact self-adjoint operator, $E$ is a sum of rank-one projections concentrated at the eigenvalues. For a general self-adjoint operator, $E$ may assign nonzero projections to intervals rather than points — this corresponds to the continuous spectrum. The spectral measure is the bridge between the discrete eigenvalue picture (familiar from linear algebra) and the continuous spectral picture (needed for operators like multiplication by $x$ on $L^2([0,1])$).

More generally, for any bounded Borel function $f$:

$$f(T) = \int_{\sigma(T)} f(\lambda) \, dE(\lambda).$$

**Connecting the two forms.** The multiplication operator form and the PVM form are two perspectives on the same object:

- In the multiplication form, $T$ acts as multiplication by $\varphi$ on $L^2(\Omega, \mu)$. The spectral measure is $E(B) = M_{\chi_{\varphi^{-1}(B)}}$ (multiplication by the indicator function of $\varphi^{-1}(B)$).
- In the PVM form, the spectral projections $E(B)$ decompose $H$ into invariant subspaces where $T$ behaves like a scalar in $B$.

**Example 7 (Spectral measure of a multiplication operator).** Let $T = M_x$ on $L^2([0,1])$, $(Tf)(x) = xf(x)$. Then $\sigma(T) = [0,1]$ (essential range of $x \mapsto x$), and $E([a,b])f = \chi_{[a,b]} \cdot f$ (multiplication by the indicator). There are no eigenvalues ($\sigma_p = \emptyset$), and the entire spectrum is continuous.

For any $f \in L^2([0,1])$, the scalar spectral measure is $\mu_{f,f}(B) = \int_B |f(x)|^2 \, dx$ — absolutely continuous with respect to Lebesgue measure, with density $|f|^2$. This captures how the "energy" of $f$ is distributed across the spectrum.

**Example 8 (Spectral decomposition of a compact self-adjoint operator revisited).** If $T$ is compact and self-adjoint with eigenvalues $\lambda_n$ and orthonormal eigenvectors $e_n$, the spectral measure is purely atomic:

$$E(B) = \sum_{n: \lambda_n \in B} \langle \cdot, e_n \rangle e_n + \chi_{\{0 \in B\}} P_{\ker T}.$$

The integral $\int \lambda \, dE(\lambda) = \sum_n \lambda_n \langle \cdot, e_n \rangle e_n$ recovers the spectral decomposition from Article 7.

---

## The Spectral Radius and Normal Operators

For normal operators ($TT^* = T^*T$), the spectral theory extends cleanly from the self-adjoint case.

**Theorem.** If $T$ is normal, then $\|T\| = r(T)$.

**Proof for self-adjoint $T$.** We have $\|T^2\| = \|T^*T\| = \|T\|^2$ (the C*-identity). By induction, $\|T^{2^n}\| = \|T\|^{2^n}$, so $\|T^{2^n}\|^{1/2^n} = \|T\|$. The Gelfand formula gives $r(T) = \lim_n \|T^n\|^{1/n} = \|T\|$.

For general normal $T$: $\|T^2\| = r(T^2) = r(T)^2$ (by the spectral mapping theorem), and $\|T^2\| = \|T^*T\| = \|T\|^2$, so $r(T) = \|T\|$. $\square$

This has a striking consequence: for normal operators, the operator norm is entirely determined by the spectrum. There is no "non-spectral" part of the norm, unlike for general operators (where $\|T\|$ can far exceed $r(T)$ — think of nilpotent matrices).

**Continuous functional calculus for normal operators.** The construction extends: for $T$ normal (not necessarily self-adjoint), there is an isometric $*$-homomorphism $C(\sigma(T)) \to B(H)$ sending the identity function to $T$ and the conjugate function $\overline{z}$ to $T^*$. This uses the fact that $\sigma(T) \subset \mathbb{C}$ is compact, and polynomials in $z$ and $\overline{z}$ are dense in $C(\sigma(T))$ by Stone-Weierstrass.

**The Gelfand representation and commutative Banach algebras.** The continuous functional calculus is a special case of a more general theory. If $\mathcal{A}$ is a commutative unital Banach algebra, the **Gelfand transform** maps $\mathcal{A}$ to $C(\Delta(\mathcal{A}))$, where $\Delta(\mathcal{A})$ is the space of characters (multiplicative linear functionals) equipped with the weak-* topology. For $\mathcal{A} = $ the closed subalgebra of $B(H)$ generated by $T$ and $I$ (where $T$ is normal), the Gelfand space is naturally identified with $\sigma(T)$, and the Gelfand transform coincides with the continuous functional calculus.

The **Gelfand-Naimark theorem** extends this: every commutative C*-algebra is isometrically $*$-isomorphic to $C(X)$ for some compact Hausdorff space $X$. This is the ultimate abstraction of the spectral theorem — it says that commutative C*-algebras "are" function algebras, and their elements "are" multiplication operators on some space. The spectral theorem for a single normal operator is the special case where the C*-algebra is generated by one element.

**Functional calculus and operator inequalities.** The continuous functional calculus preserves order: if $f \leq g$ on $\sigma(T)$, then $f(T) \leq g(T)$ (in the sense of the operator ordering: $g(T) - f(T) \geq 0$, i.e., $\langle (g(T) - f(T))x, x \rangle \geq 0$ for all $x$). This leads to powerful operator inequalities. For example:

- **Operator monotonicity:** if $T \leq S$ (i.e., $S - T \geq 0$) and $f$ is operator monotone (e.g., $f(t) = t^\alpha$ for $0 < \alpha \leq 1$, or $f(t) = \log t$), then $f(T) \leq f(S)$.
- **Jensen's inequality:** for a convex function $f$ and self-adjoint $T$, $f(\langle Tx, x \rangle) \leq \langle f(T)x, x \rangle$ for unit vectors $x$.
- **Lowner-Heinz inequality:** if $0 \leq T \leq S$, then $T^\alpha \leq S^\alpha$ for $0 \leq \alpha \leq 1$. (Note: this fails for $\alpha > 1$ in general.)

---

## Applications and Perspective

**Application to quantum mechanics.** In quantum mechanics, observables are self-adjoint operators on a Hilbert space $H$. The spectral theorem says every observable $A$ has a spectral decomposition $A = \int \lambda \, dE(\lambda)$. The probability of measuring a value in the set $B \subset \mathbb{R}$ when the system is in state $\psi$ (unit vector) is:

$$\text{Prob}(A \in B) = \langle E(B)\psi, \psi \rangle = \|\chi_B(A)\psi\|^2.$$

The expected value is $\langle A\psi, \psi \rangle = \int \lambda \, d\langle E(\lambda)\psi, \psi \rangle$. The spectral theorem is not just a mathematical convenience — it **is** the mathematical formulation of the measurement postulate.

**Application to operator semigroups and evolution equations.** For a self-adjoint operator $A$, the spectral theorem lets us define $e^{itA} = \int e^{it\lambda}\,dE(\lambda)$, which is a unitary operator (since $|e^{it\lambda}| = 1$). The family $\{e^{itA}\}_{t \in \mathbb{R}}$ is a strongly continuous one-parameter unitary group, and Stone's theorem says **every** such group arises this way. The Schrodinger equation $i\frac{d\psi}{dt} = A\psi$ has the solution $\psi(t) = e^{-itA}\psi(0)$, computed directly via the spectral measure.

**Application to the heat equation.** For the Laplacian $A = -\Delta$ on a bounded domain (a positive self-adjoint operator with compact resolvent), the spectral decomposition $A = \sum_n \lambda_n P_n$ (where $P_n$ are projections onto eigenspaces with eigenvalues $0 < \lambda_1 \leq \lambda_2 \leq \ldots \to \infty$) gives the solution of the heat equation $\partial_t u = -Au$ as:

$$u(t) = e^{-tA}u_0 = \sum_n e^{-\lambda_n t} P_n u_0.$$

Each mode decays exponentially at rate $\lambda_n$, so high-frequency modes (large $\lambda_n$) die out rapidly, and the solution smooths out over time. The long-time behavior is dominated by the smallest eigenvalue $\lambda_1$ (the fundamental mode): $u(t) \approx e^{-\lambda_1 t} P_1 u_0$ for large $t$. This spectral perspective on parabolic equations is fundamental in mathematical physics and numerical analysis.

**Spectral characterization of operator properties.** The spectrum encodes many properties of the operator in a compact form:

| Property of $T$ | Spectral condition |
|---|---|
| $T$ is invertible | $0 \notin \sigma(T)$ |
| $T$ is positive ($\langle Tx, x \rangle \geq 0$) | $\sigma(T) \subset [0, \infty)$ (for $T$ self-adjoint) |
| $T$ is a projection ($T^2 = T$, $T = T^*$) | $\sigma(T) \subset \{0, 1\}$ |
| $T$ is unitary ($T^*T = TT^* = I$) | $\sigma(T) \subset \{|\lambda| = 1\}$ |
| $T$ is nilpotent ($T^n = 0$ for some $n$) | $\sigma(T) = \{0\}$ and $T^n = 0$ |
| $T$ is quasinilpotent ($r(T) = 0$) | $\sigma(T) = \{0\}$ |

---

## What's Next

We have now covered the spectral theory of bounded operators, completing the core analytical toolkit. The next articles in this series will move in two directions: **unbounded operators** (essential for differential operators and quantum mechanics, where the operators of interest — Laplacians, Schrodinger operators, Dirac operators — are never bounded) and **Banach algebras and C*-algebras** (the algebraic framework that unifies and extends everything we have done). The Gelfand-Naimark theorem will reveal that every commutative C*-algebra is isometrically $*$-isomorphic to $C(X)$ for some compact Hausdorff space $X$, providing the ultimate generalization of the spectral theorem.

The development from compact operators (Article 7) through spectral theory (this article) to C*-algebras represents the natural progression of abstraction in functional analysis: first we understand the simplest operators (compact), then the general bounded case, then we recognize the algebraic framework (C*-algebras) that unifies everything. The spectral theorem, in its various forms, is the single most important theorem in operator theory, with applications ranging from quantum mechanics to number theory (via the spectral theory of automorphic forms) to data science (via principal component analysis, which is just the spectral theorem for finite-dimensional self-adjoint operators applied to covariance matrices).

---

*This is Part 8 of the [Functional Analysis](/en/series/functional-analysis/) series (12 articles).*

*Previous: [Part 7 — Compact Operators](/en/functional-analysis/07-compact-operators/)*

*Next: [Part 9 — Unbounded Operators](/en/functional-analysis/09-unbounded-operators/)*
