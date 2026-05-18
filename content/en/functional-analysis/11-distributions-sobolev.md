---
title: "Functional Analysis (11): Distributions and Sobolev Spaces — Generalized Solutions"
date: 2021-10-21 09:00:00
tags:
  - functional-analysis
  - distributions
  - sobolev-spaces
  - mathematics
categories: Mathematics
series: functional-analysis
lang: en
mathjax: true
description: "Distributions extend the notion of function to handle derivatives that don't exist classically — Sobolev spaces provide the right setting for weak solutions to PDE."
disableNunjucks: true
series_order: 11
series_total: 12
translationKey: "functional-analysis-11"
---

I want to start with a confession. For years I treated the Dirac delta the way an undergraduate physicist does: as a function that is zero everywhere except at the origin, where it is infinite, and whose integral is one. That description is, of course, mathematical nonsense. No measurable function can have those properties. Yet every quantum mechanics textbook uses $\delta$ on page one, every signal processing course writes $\delta(t)$ for an impulse, and every PDE book invokes Green's functions $E$ satisfying $\Delta E = \delta$. Either an entire scientific community has been making a fundamental error for a century, or there is a way to make this object rigorous. The latter, obviously — and the way is the theory of distributions.

The motivating problem is older than $\delta$ itself. Consider the wave equation $u_{tt} = c^2 u_{xx}$ on the line. Any twice-differentiable profile $f$ gives a traveling-wave solution $u(x,t) = f(x - ct)$. But physical waves carry shocks: the solution of a shallow-water equation can develop a step, a sound wave can have a sharp front, a light pulse can be a square envelope. The "function" $f$ in those cases is not even continuous. Calling such a discontinuous $u$ a "solution" of $u_{tt} = c^2 u_{xx}$ requires us to differentiate it twice, and the classical second derivative does not exist — neither at the shock nor on either side, since differentiating an indicator gives a delta.

![Delta distribution as limit of bump functions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig11_delta_distribution.png)

Laurent Schwartz's distribution theory (1944-1950) solves both problems with one trick. Stop trying to assign pointwise values to generalized "functions." Instead, define them by how they act on smooth test functions: $f$ is the linear map $\varphi \mapsto \int f\varphi$. Two functions equal a.e. give the same map, so $L^1_{\text{loc}}$ embeds into the dual space of test functions. But that dual space is much larger than $L^1_{\text{loc}}$; it contains $\delta$, derivatives of $\delta$, principal-value distributions, and a great deal of structure besides. Once we have the dual, every operation we want — derivative, Fourier transform, convolution — extends from smooth functions to all distributions by formal duality.

Sobolev spaces are the second piece of the story. Schwartz's distributions are too big to be a useful Banach space (the dual of $C_c^\infty$ has no natural norm topology). For PDE we want concrete Hilbert spaces, with norms, embeddings, and compactness. Sergei Sobolev's 1930s construction does exactly this: $W^{k,p}(\Omega)$ contains functions whose distributional derivatives up to order $k$ live in $L^p$. These are the natural domains for differential operators, the right setting for weak solutions, and they come with three key tools — embedding theorems, trace theorems, Rellich-Kondrachov compactness — without which the Lax-Milgram theorem of the next article would have nothing to bite on.

---

## Why Classical Derivatives Aren't Enough

### Three motivating problems

**Problem 1: weak solutions of PDE.** Consider $-u'' = f$ on $(0,1)$ with $u(0) = u(1) = 0$. If $f$ is continuous, classical solutions exist and are $C^2$. But what if $f \in L^2(0,1)$ — say $f$ is the indicator of $[1/3, 2/3]$? No classical $C^2$ solution exists; $u''$ would have to jump. Yet multiplying by a test function $\varphi \in C_c^\infty(0,1)$ and integrating by parts twice (the second integration produces no boundary terms because $\varphi$ has compact support) gives

$$
\int_0^1 u'\varphi' \, dx = \int_0^1 f\varphi \, dx \quad \text{for all } \varphi \in C_c^\infty(0,1).
$$

A function $u$ satisfying this integral identity is a **weak solution**. It need not be $C^2$; $u \in H^1_0(0,1)$ — one weak derivative in $L^2$, vanishing at the boundary — suffices. Here the problem actually has an explicit answer: integrate $f$ twice and adjust the constants. You get a $C^1$ piecewise quadratic that is not $C^2$ (the second derivative jumps at $x = 1/3$ and $x = 2/3$), and that is exactly the regularity Sobolev theory predicts: $f \in L^2$ and one application of $-d^2/dx^2$ inverts to gain two derivatives, so $u \in H^2$, hence $u \in C^{1,1/2}$ by Morrey but not $C^2$. Classical theory would simply declare the problem unsolvable.

![Test functions: smooth bumps with compact support](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/11-distributions-sobolev/fa_v2_11_1_test_functions.png)

**Problem 2: the Dirac delta.** In electrostatics, the potential of a unit point charge at the origin satisfies $-\Delta\phi = \delta$. The right-hand side is not a function. Worse, the solution $\phi(x) = 1/(4\pi|x|)$ is singular at the origin, so it is not classically twice differentiable there either. The whole equation lives in a world where "derivative" must be reinterpreted.

**Problem 3: Fourier analysis.** The Fourier transform of the constant function $1$ is, formally, $(2\pi)^n\delta$. The Fourier transform of $|x|$ involves $1/|\xi|^{n+1}$ (a non-locally-integrable function read as a principal value). Without a framework that contains $\delta$ and its derivatives, large parts of harmonic analysis break down or require ad-hoc patches. Tempered distributions $\mathcal{S}'$ provide the right framework: every tempered distribution has a Fourier transform, and the transform is a topological isomorphism $\mathcal{F}: \mathcal{S}' \to \mathcal{S}'$.

### The conceptual shift

The classical approach: a function $f$ is defined by its pointwise values $f(x)$. The distributional approach: a generalized function $f$ is defined by the linear functional $\varphi \mapsto \int f\varphi\,dx$.

Two locally integrable functions that differ on a set of measure zero define the same distribution; their integrals against test functions are identical. This means distributions automatically quotient out null sets, exactly as $L^p$ spaces do. Pointwise values were always a fiction in $L^p$; distributions just admit the fiction openly and stop pretending we can evaluate at a point.

### The topology on $\mathcal{D}'(\Omega)$

The space of distributions carries the **weak-* topology**: $u_j \to u$ in $\mathcal{D}'(\Omega)$ iff $\langle u_j, \varphi \rangle \to \langle u, \varphi \rangle$ for every $\varphi \in \mathcal{D}(\Omega)$. This is pointwise convergence on test functions, weaker than uniform convergence on bounded sets but sufficient for the bulk of PDE theory.

A key result: every distribution is the limit (in $\mathcal{D}'$) of a sequence of smooth functions. If $u \in \mathcal{D}'(\Omega)$ and $\rho_\epsilon$ is a standard mollifier, then $u * \rho_\epsilon \to u$ in $\mathcal{D}'$ as $\epsilon \to 0$ (where the convolution is defined by transposition). Distributions are thus "limits of smooth functions" in a rigorous sense — exactly as real numbers are limits of rationals, and exactly as $L^p$ functions are limits of simple functions.

---


### Worked Numerical Example
Take $-u'' = f$ on $(0,1)$ with $u(0)=u(1)=0$ and $f = \chi_{[0.2, 0.6]}$. Integrate twice explicitly. On $[0, 0.2]$, $u''=0$ and $u(0)=0$ gives $u(x) = c_1 x$. On $[0.2, 0.6]$, $u''=-1$ gives $u(x) = -\frac{1}{2}x^2 + c_2 x + c_3$. On $[0.6, 1]$, $u''=0$ and $u(1)=0$ gives $u(x) = c_4(1-x)$. Enforce $C^1$ matching at $x=0.2$ and $x=0.6$. Solving the linear system yields $c_1 = 0.4$, $c_2 = 0.6$, $c_3 = -0.06$, $c_4 = 0.24$. The resulting $u$ is piecewise quadratic, continuous, and has a continuous first derivative. The second derivative jumps from $0$ to $-1$ at $x=0.2$ and back at $x=0.6$. Now test the weak formulation with $\varphi(x) = x(1-x)$. Compute $\int_0^1 u'\varphi' dx$. Since $\varphi'(x) = 1-2x$, split the integral over the three intervals. Direct calculation gives $\int_0^{0.2} 0.4(1-2x)dx + \int_{0.2}^{0.6} (-x+0.6)(1-2x)dx + \int_{0.6}^1 (-0.24)(1-2x)dx = 0.064$. Compute the right side: $\int_0^1 f\varphi dx = \int_{0.2}^{0.6} x(1-x)dx = [\frac{x^2}{2} - \frac{x^3}{3}]_{0.2}^{0.6} = 0.064$. The identity holds exactly. The classical second derivative fails at two points, but the integral identity survives without modification.

## Test Functions $\mathcal{D}(\Omega)$ and Distributions $\mathcal{D}'(\Omega)$

### The space of test functions

![Test functions: smooth with compact support](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig11_test_functions.png)

Let $\Omega \subseteq \mathbb{R}^n$ be open. The space of **test functions** is

$$
\mathcal{D}(\Omega) = C_c^\infty(\Omega),
$$

the space of infinitely differentiable functions with compact support in $\Omega$. The standard example is the bump $\varphi(x) = \exp(-1/(1-|x|^2))$ for $|x| < 1$, extended by zero. It is $C^\infty$ everywhere (the matching of all derivatives at $|x|=1$ is a delicate calculus exercise), nonnegative, and supported in the closed unit ball.

**Topology on $\mathcal{D}(\Omega)$.** A sequence $\varphi_j \to 0$ in $\mathcal{D}(\Omega)$ if:
1. There exists a compact $K \subset \Omega$ such that $\text{supp}(\varphi_j) \subset K$ for all $j$.
2. For every multi-index $\alpha$, $\partial^\alpha \varphi_j \to 0$ uniformly on $K$.

This is *not* a norm topology; it is an inductive limit, the finest locally convex topology making the inclusions $C_c^\infty(K) \hookrightarrow C_c^\infty(\Omega)$ continuous for each compact $K \subset \Omega$. The detail matters because it is what guarantees the dual $\mathcal{D}'(\Omega)$ is large enough to contain the objects we want.

### Distributions

A **distribution** on $\Omega$ is a continuous linear functional $u: \mathcal{D}(\Omega) \to \mathbb{C}$ — equivalently, a linear $u$ such that for every compact $K \subset \Omega$, there exist $C, N \ge 0$ with

$$
|\langle u, \varphi \rangle| \le C \sum_{|\alpha| \le N} \sup_K |\partial^\alpha \varphi| \quad \text{for all } \varphi \in C_c^\infty(K).
$$

The smallest such $N$ is the **order** of $u$ on $K$. Locally integrable functions are distributions of order $0$; the Dirac delta is order $0$; its derivatives are higher order.

Key examples:

1. **Locally integrable functions.** Every $f \in L^1_{\text{loc}}(\Omega)$ defines a distribution by $\langle f, \varphi \rangle = \int_\Omega f\varphi\,dx$.

2. **Dirac delta.** $\langle \delta_a, \varphi \rangle = \varphi(a)$. This is *not* of the form above; no $L^1_{\text{loc}}$ function can satisfy the defining identity (taking $\varphi$ a sequence of bumps shrinking to $a$ shows $f$ would have to "concentrate" in a way no function permits).

3. **Heaviside.** $H(x) = 1$ for $x \ge 0$, $0$ otherwise. Its distributional derivative is $\delta$ — an integration by parts: $\langle H', \varphi \rangle = -\langle H, \varphi' \rangle = -\int_0^\infty \varphi'(x)\,dx = \varphi(0)$.

4. **Principal value $1/x$.** The function $1/x$ on $\mathbb{R}$ is not locally integrable near $0$, but the principal value $\langle \mathrm{p.v.} \tfrac{1}{x}, \varphi \rangle = \lim_{\epsilon \to 0}\int_{|x|>\epsilon}\varphi(x)/x\,dx$ defines a distribution of order $1$.

![Distributions like Dirac delta acting on test functions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/11-distributions-sobolev/fa_v2_11_2_distributions.png)

### Operations on distributions

The unifying principle: define every operation by formal duality. If $T$ is a continuous linear map on test functions with adjoint $T^*$, define $Tu$ on distributions by $\langle Tu, \varphi \rangle = \langle u, T^*\varphi \rangle$.

**Differentiation.** $\langle \partial^\alpha u, \varphi \rangle = (-1)^{|\alpha|}\langle u, \partial^\alpha \varphi \rangle$. The sign comes from integration by parts: $\int \partial f \cdot \varphi = -\int f \cdot \partial\varphi$ for $f$ smooth and $\varphi$ compactly supported. This formula extends the chain rule to the entire dual space; *every* distribution is infinitely differentiable in the distributional sense, and differentiation is continuous in the weak-* topology. Compare to the classical situation, where differentiation is unbounded and fails to commute with limits.

**Multiplication by smooth functions.** If $a \in C^\infty(\Omega)$, $\langle au, \varphi \rangle = \langle u, a\varphi \rangle$ — the test function $a\varphi$ is again $C_c^\infty$.

**Warning: no distributional product.** Multiplication of two arbitrary distributions is *not* defined. The product $\delta \cdot \delta$ has no canonical meaning; attempts to define it lead to renormalization in QFT. Schwartz's impossibility theorem makes this precise: there is no associative, commutative multiplication on $\mathcal{D}'$ that extends pointwise multiplication on continuous functions and satisfies the Leibniz rule. The deepest difficulties of nonlinear PDE and quantum field theory are downstream of this single negative result.

### Why distributional derivatives are unique when classical ones exist

A natural worry: does the new derivative agree with the old one when both are defined? Yes. If $f \in C^1$, then for any $\varphi \in C_c^\infty$, integration by parts gives $\int f'\varphi = -\int f\varphi'$, which is exactly the distributional definition. The distributional derivative coincides with the classical one for $C^1$ functions, agrees with the a.e. derivative for $W^{1,p}$ functions, and disagrees with both for genuinely singular objects like $H'$ (because the classical derivative does not exist there at all).

A worked example for $|x|$ on $\mathbb{R}$: it is differentiable everywhere except at the origin, where the classical derivative is undefined. The distributional derivative is the sign function $\mathrm{sgn}(x) = H(x) - H(-x)$, which is defined a.e. and agrees with the classical derivative wherever the latter exists. Differentiating once more: $|x|'' = (\mathrm{sgn})' = 2\delta$, since the sign function jumps by $2$ at the origin. This concrete computation — the second distributional derivative of $|x|$ is $2\delta$ — is one I find myself rederiving every couple of months. It is the standard sanity check that one's distributional bookkeeping is correct.

### Convolution and mollification

If $u \in \mathcal{D}'(\mathbb{R}^n)$ has compact support and $\varphi \in C^\infty(\mathbb{R}^n)$, the convolution $u * \varphi$ is defined and is $C^\infty$:

![Animation: mollification smoothing a function](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/gif11_mollification.gif)

$$
(u * \varphi)(x) = \langle u_y, \varphi(x - y)\rangle.
$$

More generally, convolution with a compactly supported distribution is well-defined for any distribution. Key properties:

- $\delta * f = f$ for any distribution $f$ — the delta is the convolution identity.
- $\partial^\alpha(u * v) = (\partial^\alpha u) * v = u * (\partial^\alpha v)$ — derivatives can be shifted between factors.
- **Mollification:** if $\rho_\epsilon(x) = \epsilon^{-n}\rho(x/\epsilon)$ is a standard mollifier (positive, $C_c^\infty$, integral $1$, supported in the unit ball), then $u * \rho_\epsilon \in C^\infty$ for any distribution $u$, and $u * \rho_\epsilon \to u$ in $\mathcal{D}'$ as $\epsilon \to 0$.

The mollification fact is the workhorse of distribution theory. It says every distribution is approximable by smooth functions, with the approximation explicit and computable. Most proofs in PDE follow the pattern: prove the result for smooth $u$, then mollify and pass to the limit.

### Tempered distributions and the Fourier transform

The **Schwartz space** $\mathcal{S}(\mathbb{R}^n)$ consists of smooth functions whose derivatives all decay faster than any polynomial: $\sup_x |x^\alpha \partial^\beta \varphi(x)| < \infty$ for all multi-indices. Its dual $\mathcal{S}'(\mathbb{R}^n)$ is the space of **tempered distributions** — strictly smaller than $\mathcal{D}'$ but large enough to include polynomials, $L^p$ for all $p$, $\delta$ and its derivatives, and most things one cares about in harmonic analysis.

The Fourier transform extends to an isomorphism $\mathcal{F}: \mathcal{S}' \to \mathcal{S}'$ via $\langle \hat{u}, \varphi \rangle = \langle u, \hat{\varphi} \rangle$, exploiting the fact that $\mathcal{F}$ already maps $\mathcal{S}$ to itself bicontinuously. Key formulas:

- $\hat{\delta} = 1$,
- $\hat{1} = (2\pi)^n \delta$,
- $\widehat{\partial^\alpha u} = (i\xi)^\alpha \hat{u}$ — differentiation becomes multiplication by polynomials, valid for *all* tempered distributions.

These identities power Fourier-analytic PDE: solving $-\Delta u = f$ becomes $|\xi|^2\hat{u} = \hat{f}$, so $\hat{u} = \hat{f}/|\xi|^2$ — the difficulty is purely in the inverse transform and the behaviour at $\xi = 0$, not in the algebraic step.

### Fundamental solutions

A **fundamental solution** for a linear differential operator $L$ is a distribution $E$ satisfying $LE = \delta$. For the Laplacian on $\mathbb{R}^n$ ($n \ge 3$):

$$
E(x) = \frac{1}{n(n-2)\omega_n|x|^{n-2}},
$$

where $\omega_n$ is the volume of the unit ball. In $\mathbb{R}^2$, $E(x) = -\frac{1}{2\pi}\log|x|$; in $\mathbb{R}^1$, $E(x) = -\frac{1}{2}|x|$. The solution to $-\Delta u = f$ (for suitable $f$) is then $u = E * f$, the convolution with the fundamental solution. This is the distributional incarnation of classical potential theory.

For the heat equation $(\partial_t - \Delta)u = 0$, the fundamental solution is the heat kernel $K(x, t) = (4\pi t)^{-n/2}e^{-|x|^2/(4t)}$ for $t > 0$, the same kernel that appeared in the semigroup article. The distributional perspective clarifies why this kernel appears: it is the unique tempered distribution satisfying $(\partial_t - \Delta)K = \delta(x)\delta(t)$.

---


### Worked Numerical Example
Compute $\langle \mathrm{p.v.}\frac{1}{x}, \varphi \rangle$ for $\varphi(x) = x e^{-x^2}$. By definition, the pairing is $\lim_{\epsilon \to 0} \int_{|x|>\epsilon} \frac{x e^{-x^2}}{x} dx = \lim_{\epsilon \to 0} \int_{|x|>\epsilon} e^{-x^2} dx$. The integrand simplifies to a Gaussian. The limit removes a symmetric interval around zero, which contributes zero measure to the Lebesgue integral of a continuous function. Thus the value is exactly $\int_{-\infty}^\infty e^{-x^2} dx = \sqrt{\pi} \approx 1.77245$. Now try $\psi(x) = e^{-x^2}$. The pairing becomes $\lim_{\epsilon \to 0} \int_{|x|>\epsilon} \frac{e^{-x^2}}{x} dx$. The integrand is odd, and the domain $(-\infty, -\epsilon) \cup (\epsilon, \infty)$ is symmetric. The integral vanishes identically for every $\epsilon > 0$, so the limit is $0$. This explicit computation shows how the principal value extracts the odd part of the test function near the singularity while discarding the even part. The distribution is well-defined precisely because the $1/x$ singularity is cancelled by the vanishing of $\varphi(0)$ or by symmetry. If you replace $\varphi$ with a function that does not vanish at $0$ and lacks symmetry, the limit diverges, which is why $1/x$ alone is not a distribution without the p.v. prescription.

## Weak Derivatives and Sobolev Spaces

### From distributional to weak derivative

![Weak derivative of |x|](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig11_weak_derivative.png)

Distributional derivatives always exist; they are abstract objects in $\mathcal{D}'$. For PDE we want concrete derivatives that live in $L^p$.

**Definition.** $u \in L^1_{\text{loc}}(\Omega)$ has a **weak derivative** $g \in L^1_{\text{loc}}(\Omega)$ in the direction $\partial^\alpha$ if

$$
\int_\Omega u\,\partial^\alpha\varphi\,dx = (-1)^{|\alpha|}\int_\Omega g\,\varphi\,dx \quad \text{for all } \varphi \in C_c^\infty(\Omega).
$$

The weak derivative, when it exists, is unique a.e. and coincides with the classical derivative when both make sense. Many functions that fail to be differentiable classically have weak derivatives: $|x|$ has weak derivative $\mathrm{sgn}(x)$; $\max(u, 0)$ has weak derivative $u'\cdot\mathbf{1}_{u>0}$; absolutely continuous functions on the line have weak derivatives equal to their classical a.e. derivatives.

![Weak derivative defined via integration against test functions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/11-distributions-sobolev/fa_v2_11_3_weak_deriv.png)

### The Sobolev space $W^{k,p}$

**Definition.** For $k \in \mathbb{N}_0$, $1 \le p \le \infty$, and $\Omega \subseteq \mathbb{R}^n$ open, the **Sobolev space** $W^{k,p}(\Omega)$ is

$$
W^{k,p}(\Omega) = \{u \in L^p(\Omega) : \partial^\alpha u \in L^p(\Omega) \text{ for all } |\alpha| \le k\},
$$

where $\partial^\alpha u$ denotes the weak (equivalently, distributional) derivative. The norm is

$$
\|u\|_{W^{k,p}} = \left(\sum_{|\alpha| \le k} \|\partial^\alpha u\|_{L^p}^p\right)^{1/p} \quad (1 \le p < \infty),
$$

with the obvious modification for $p = \infty$.

**Notation.** $H^k(\Omega) = W^{k,2}(\Omega)$. These are Hilbert spaces with inner product $\langle u, v \rangle_{H^k} = \sum_{|\alpha| \le k} \langle \partial^\alpha u, \partial^\alpha v \rangle_{L^2}$. The Hilbert structure makes $H^k$ the natural setting for the Lax-Milgram theorem and variational methods.

**Vanishing trace.** $W_0^{k,p}(\Omega)$ is the closure of $C_c^\infty(\Omega)$ in $W^{k,p}(\Omega)$. Intuitively, these are Sobolev functions that "vanish at the boundary" in a generalized sense; the trace theorem below makes this precise.

### Completeness

**Theorem.** $W^{k,p}(\Omega)$ is a Banach space; $H^k(\Omega)$ is a Hilbert space.

*Proof sketch.* Let $(u_j)$ be Cauchy in $W^{k,p}$. For each $|\alpha| \le k$, the sequence $(\partial^\alpha u_j)$ is Cauchy in $L^p(\Omega)$. By completeness of $L^p$, there exist $u_\alpha \in L^p$ with $\partial^\alpha u_j \to u_\alpha$ in $L^p$. Set $u = u_0$. We claim $\partial^\alpha u = u_\alpha$ in the distributional sense:

$$
\langle \partial^\alpha u, \varphi \rangle = (-1)^{|\alpha|}\langle u, \partial^\alpha\varphi \rangle = (-1)^{|\alpha|}\lim_j \langle u_j, \partial^\alpha\varphi \rangle = \lim_j \langle \partial^\alpha u_j, \varphi \rangle = \langle u_\alpha, \varphi \rangle.
$$

So $u \in W^{k,p}$ and $u_j \to u$. $\square$

### Numerical example: regularity of $|x|^\alpha$

Take $u(x) = |x|^\alpha$ on the unit ball $B \subset \mathbb{R}^n$. When does $u \in W^{1,p}(B)$? The weak gradient is $\nabla u = \alpha|x|^{\alpha-2}x$ (extending the classical formula), and

$$
\int_B |\nabla u|^p\,dx = |\alpha|^p \int_B |x|^{p(\alpha-1)}\,dx = |\alpha|^p \omega_{n-1}\int_0^1 r^{p(\alpha-1)+n-1}\,dr.
$$

This integral converges iff $p(\alpha-1) + n - 1 > -1$, i.e., $\alpha > 1 - n/p$. So $|x|^\alpha \in W^{1,p}(B)$ exactly for $\alpha > 1 - n/p$. In $n = 3$, $p = 2$ this gives $\alpha > -1/2$, so $|x|^{-1/2}$ is barely *not* in $H^1$ but $|x|^{-1/4}$ is. This kind of explicit threshold guides what regularity to expect for solutions of singular PDE.

A second numerical example: Sobolev embedding boundary in three dimensions. With $n = 3$ and $p = 2$, $p^* = 6$, so $H^1(\mathbb{R}^3) \hookrightarrow L^6(\mathbb{R}^3)$. Concretely, the Sobolev inequality says $\|u\|_{L^6} \le C\|\nabla u\|_{L^2}$ for any compactly supported smooth $u$. The optimal constant $C$ was computed by Talenti in 1976: $C = \frac{1}{\pi}\sqrt[3]{\frac{1}{4}\Gamma(3/2)/\Gamma(3)}$, with extremals exactly the Aubin-Talenti bubbles $u_\epsilon(x) = c_n(\epsilon^2 + |x|^2)^{-(n-2)/2}$ (here $c_n$ is a normalization). Plug in $\epsilon = 1$, $n = 3$: $u_1(x) = c_3(1 + |x|^2)^{-1/2}$, and $\|u_1\|_{L^6}/\|\nabla u_1\|_{L^2} = C$ exactly. The fact that extremals exist (Aubin, Talenti) is delicate; the fact that they form a non-compact orbit under scaling $u \mapsto \epsilon^{1/2}u(\epsilon x)$ is the source of every concentration phenomenon in nonlinear analysis.

### Fractional and negative Sobolev spaces

For $s \in \mathbb{R}$ (not necessarily an integer), define $H^s(\mathbb{R}^n)$ via the Fourier transform:

$$
H^s(\mathbb{R}^n) = \{u \in \mathcal{S}'(\mathbb{R}^n) : (1 + |\xi|^2)^{s/2}\hat{u} \in L^2(\mathbb{R}^n)\},
$$

with norm $\|u\|_{H^s} = \|(1 + |\xi|^2)^{s/2}\hat{u}\|_{L^2}$. For $s < 0$, $H^s$ contains distributions that are not functions. The space $H^{-k}(\Omega)$ is the dual of $H_0^k(\Omega)$.

The Dirac delta lies in $H^s(\mathbb{R}^n)$ iff $s < -n/2$ (since $\hat{\delta} = 1$ and $(1 + |\xi|^2)^{s/2} \in L^2$ iff $s < -n/2$). So $\delta \in H^{-1-\epsilon}(\mathbb{R}) \setminus H^{-1/2}(\mathbb{R})$, and the higher the dimension, the more "singular" $\delta$ becomes.

![Sobolev embedding chain for varying regularity](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/11-distributions-sobolev/fa_v2_11_4_sobolev_chain.png)

### Density and approximation

A fundamental property of Sobolev spaces is that smooth functions are dense:

**Theorem (Meyers-Serrin).** $C^\infty(\Omega) \cap W^{k,p}(\Omega)$ is dense in $W^{k,p}(\Omega)$ for $1 \le p < \infty$.

When $\Omega$ has Lipschitz boundary, even $C^\infty(\overline{\Omega})$ is dense in $W^{k,p}(\Omega)$. This approximation property is essential for proving theorems: one first establishes the result for smooth functions (where classical calculus applies) and then extends by density.

### Poincare inequality

**Theorem (Poincare).** Let $\Omega \subset \mathbb{R}^n$ be bounded and connected. There exists $C = C(\Omega, p)$ such that for all $u \in W_0^{1,p}(\Omega)$,

$$
\|u\|_{L^p(\Omega)} \le C\|\nabla u\|_{L^p(\Omega)}.
$$

For functions vanishing on the boundary, the $L^p$ norm is controlled by the gradient alone. This is the key step in showing $\|\nabla u\|_{L^p}$ is an equivalent norm on $W_0^{1,p}(\Omega)$, and it is the inequality that powers every application of Lax-Milgram to elliptic PDE in the next article. The Poincare constant scales like the diameter of $\Omega$: for a ball of radius $R$, $C \sim R$.

A variant, the **Poincare-Wirtinger inequality**, holds for functions without boundary conditions: $\|u - \bar{u}\|_{L^p} \le C\|\nabla u\|_{L^p}$ where $\bar{u}$ is the mean of $u$ over $\Omega$. This is the relevant inequality for Neumann problems, where the solution is determined only up to an additive constant.

---


### Worked Numerical Example
Consider the hat function $u(x) = \max(0, 1-|x|)$ on $\Omega = (-2, 2)$. Classically, $u$ is not differentiable at $x=-1, 0, 1$. The weak derivative $g$ must satisfy $\int_{-2}^2 u \varphi' dx = -\int_{-2}^2 g \varphi dx$. Split the left integral: $\int_{-1}^0 (1+x)\varphi' dx + \int_0^1 (1-x)\varphi' dx$. Integrate by parts on each smooth piece. The boundary terms at $\pm 1$ vanish because $u(\pm 1)=0$. The interior terms at $0$ cancel because $u$ is continuous. We are left with $-\int_{-1}^0 1\cdot \varphi dx + \int_0^1 (-1)\cdot \varphi dx$. Thus $g(x) = 1$ on $(-1,0)$, $g(x) = -1$ on $(0,1)$, and $0$ elsewhere. Compute the $H^1$ norm explicitly. $\|u\|_{L^2}^2 = 2\int_0^1 (1-x)^2 dx = 2[ -\frac{(1-x)^3}{3} ]_0^1 = \frac{2}{3}$. $\|g\|_{L^2}^2 = \int_{-1}^0 1^2 dx + \int_0^1 (-1)^2 dx = 2$. The squared $H^1$ norm is $\frac{2}{3} + 2 = \frac{8}{3}$, so $\|u\|_{H^1} = \sqrt{8/3} \approx 1.63299$. The calculation requires no distributional machinery beyond integration by parts on subintervals, yet it places a non-differentiable function firmly inside a Hilbert space.

## Sobolev Embedding Theorems

The embedding theorems answer a fundamental question: if a function has $k$ derivatives in $L^p$, what can we say about its pointwise regularity?

![Sobolev embedding in 1D](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig11_sobolev_embedding.png)

### Sobolev inequality (Gagliardo-Nirenberg-Sobolev)

**Theorem.** Let $1 \le p < n$ and define $p^* = np/(n-p)$ (the **Sobolev conjugate exponent**). Then there exists $C = C(n, p)$ such that for all $u \in W^{1,p}(\mathbb{R}^n)$,

$$
\|u\|_{L^{p^*}} \le C\|\nabla u\|_{L^p}.
$$

Consequently $W^{1,p}(\mathbb{R}^n) \hookrightarrow L^{p^*}(\mathbb{R}^n)$ continuously.

**Interpretation.** Having one derivative in $L^p$ promotes you to a *better* $L^q$ space, with $q = p^* > p$. The gain is determined by the dimension: in $n = 3$, $p = 2$ gives $p^* = 6$, so $H^1(\mathbb{R}^3) \hookrightarrow L^6(\mathbb{R}^3)$ — every gradient-square-integrable function is sixth-power integrable. As $n$ grows, the gain shrinks; as $n \to \infty$, $p^* \to p$ and the embedding becomes trivial. Higher dimensions give less integrability improvement.

### Morrey's inequality

**Theorem.** Let $p > n$. Then there exists $C = C(n, p)$ such that for all $u \in W^{1,p}(\mathbb{R}^n)$,

$$
\|u\|_{C^{0,\gamma}(\mathbb{R}^n)} \le C\|u\|_{W^{1,p}},
$$

where $\gamma = 1 - n/p$ and $C^{0,\gamma}$ is the space of Holder continuous functions with exponent $\gamma$.

When $p > n$, having one derivative in $L^p$ guarantees *continuity* and even Holder continuity. Crossing the threshold $p = n$ is the key transition: $p < n$ gives integrability improvement, $p > n$ gives pointwise regularity. This is why $H^1(\mathbb{R})$ functions are continuous (since $1 < 2 = p$ for the trace dimension) but $H^1(\mathbb{R}^2)$ and $H^1(\mathbb{R}^3)$ functions need not be — a one-derivative gain is not enough to escape integrability for $n \ge 2$.

### General embeddings

For $kp < n$: $W^{k,p}(\Omega) \hookrightarrow L^q(\Omega)$ for $q \le np/(n - kp)$.

For $kp = n$: $W^{k,p}(\Omega) \hookrightarrow L^q(\Omega)$ for all $q < \infty$ — borderline case where logarithmic corrections appear.

For $kp > n$: $W^{k,p}(\Omega) \hookrightarrow C^{m,\gamma}(\overline{\Omega})$ where $m = k - \lfloor n/p \rfloor - 1$ and $\gamma$ depends on the fractional part. In this regime, Sobolev functions are classically differentiable.

### Rellich-Kondrachov compactness theorem

**Theorem.** Let $\Omega \subset \mathbb{R}^n$ be bounded with Lipschitz boundary and $1 \le p < n$. Then $W^{1,p}(\Omega) \hookrightarrow L^q(\Omega)$ is **compact** for $1 \le q < p^*$.

This is the compactness result that powers variational methods. A bounded sequence in $W^{1,p}$ has a subsequence converging in $L^q$ for any subcritical $q$. The infinite-dimensional analogue of Bolzano-Weierstrass: replace "bounded and closed" by "bounded in a stronger norm." Existence of eigenvalues for the Laplacian on bounded domains reduces (via Rellich-Kondrachov) to compactness of the resolvent of $-\Delta$ — the spectral theorem for compact operators from earlier in the series then delivers the spectrum.

### Failure of compactness at the critical exponent

Rellich-Kondrachov fails at $q = p^*$: the embedding $W^{1,p}(\Omega) \hookrightarrow L^{p^*}(\Omega)$ is continuous but *not* compact. This failure has profound consequences for nonlinear PDE. The existence of extremals for the Sobolev inequality (Aubin, Talenti, 1976) and the associated variational problem exhibits **concentration**: minimizing sequences can collapse to a point, losing compactness. The concentration-compactness principle (Lions, 1984) catalogues the ways compactness can fail and recovers it via additional structure. Most semilinear elliptic problems with critical-exponent nonlinearities — Yamabe problem, prescribed scalar curvature, conformal geometry — live entirely in this regime.

### Extension theorems

The embeddings extend from $\mathbb{R}^n$ to domains $\Omega$ under boundary regularity:

**Theorem (Sobolev extension).** Let $\Omega \subset \mathbb{R}^n$ be bounded with Lipschitz boundary. There exists a bounded linear $E: W^{k,p}(\Omega) \to W^{k,p}(\mathbb{R}^n)$ such that $Eu|_\Omega = u$.

This extension reduces problems on domains to problems on $\mathbb{R}^n$, where Fourier-analytic tools are available. The Lipschitz hypothesis is essentially sharp: domains with cusps or fractal boundaries can fail the extension property.

---


### Worked Numerical Example
Verify the 1D Sobolev inequality $\|u\|_{L^\infty(\mathbb{R})} \le \frac{1}{\sqrt{2}} \|u\|_{H^1(\mathbb{R})}$ for $u(x) = e^{-|x|}$. The supremum is clearly $u(0) = 1$. Compute the $L^2$ norm: $\|u\|_{L^2}^2 = \int_{-\infty}^\infty e^{-2|x|} dx = 2\int_0^\infty e^{-2x} dx = 1$. The weak derivative is $u'(x) = -\mathrm{sgn}(x)e^{-|x|}$, so $\|u'\|_{L^2}^2 = \int_{-\infty}^\infty e^{-2|x|} dx = 1$. The $H^1$ norm squared is $1+1=2$, giving $\|u\|_{H^1} = \sqrt{2} \approx 1.41421$. The inequality reads $1 \le \frac{1}{\sqrt{2}} \cdot \sqrt{2} = 1$. The bound is saturated. This is not an accident: in one dimension, the optimal constant $1/\sqrt{2}$ is achieved exactly by multiples of $e^{-|x|}$. The embedding $H^1(\mathbb{R}) \hookrightarrow C^0_b(\mathbb{R})$ is continuous, and the numerical check confirms that controlling the function and its first derivative in $L^2$ strictly bounds the pointwise maximum. If you drop the derivative term and only control $\|u\|_{L^2}$, you can make the peak arbitrarily high by narrowing a bump while preserving area, destroying any $L^\infty$ bound. The derivative term is what locks the height down.

## Trace Theorems and Boundary Values

A function in $W^{1,p}(\Omega)$ is defined only up to a set of measure zero. The boundary $\partial\Omega$ has Lebesgue measure zero in $\mathbb{R}^n$, so pointwise restriction to the boundary is not well-defined. Yet boundary conditions like $u|_{\partial\Omega} = 0$ (Dirichlet) or $\partial u/\partial n|_{\partial\Omega} = g$ (Neumann) are essential for PDE.

![Trace theorem: restricting to boundary](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/figures/fig11_trace_theorem.png)

The **trace theorem** resolves this by showing that restriction to the boundary extends to a continuous operation on Sobolev spaces.

**Theorem (Trace).** Let $\Omega \subset \mathbb{R}^n$ be bounded with Lipschitz boundary. There exists a unique bounded linear operator

$$
\gamma_0: W^{1,p}(\Omega) \to L^p(\partial\Omega) \quad (1 \le p < \infty)
$$

such that $\gamma_0 u = u|_{\partial\Omega}$ for all $u \in C^\infty(\overline{\Omega})$. Moreover:

1. $\gamma_0$ is surjective onto $W^{1-1/p, p}(\partial\Omega)$ (a fractional Sobolev space on the boundary).
2. $\ker \gamma_0 = W_0^{1,p}(\Omega)$.

![Trace theorem: boundary restriction extended to Sobolev functions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/11-distributions-sobolev/fa_v2_11_5_trace.png)

Part 2 gives a precise characterization of $W_0^{1,p}$: it is exactly the Sobolev functions whose trace is zero. This makes rigorous the statement "functions in $H_0^1(\Omega)$ vanish on $\partial\Omega$." The fractional space in part 1 is the price one pays for restricting from $n$ dimensions to $n-1$: an $H^1$ function on $\Omega$ has only a fractional ($H^{1/2}$) trace on the boundary.

**For higher-order Sobolev spaces,** there are higher-order trace operators $\gamma_j u = \partial^j u/\partial n^j|_{\partial\Omega}$, and the kernel of $(\gamma_0, \gamma_1, \ldots, \gamma_{k-1}): W^{k,p}(\Omega) \to \prod_{j=0}^{k-1} W^{k-j-1/p, p}(\partial\Omega)$ is $W_0^{k,p}(\Omega)$.

### Trace inequalities and applications

The trace theorem comes with a quantitative estimate: $\|\gamma_0 u\|_{L^p(\partial\Omega)} \le C\|u\|_{W^{1,p}(\Omega)}$. This trace inequality is essential for formulating boundary value problems.

For Neumann problems, the trace theorem for normal derivatives gives meaning to the boundary condition: $g$ must lie in the trace space $W^{-1/p', p'}(\partial\Omega)$ (the dual of the trace space for the conjugate exponent). The compatibility condition $\int_\Omega f = \int_{\partial\Omega} g$ for $-\Delta u = f$, $\partial u/\partial n = g$ is a consequence of the divergence theorem applied in the distributional sense — without distribution theory, the compatibility condition would be an unmotivated technical hypothesis.

### Capacity and fine properties

Beyond the trace theorem, Sobolev functions possess remarkable fine properties. A function $u \in W^{1,p}(\Omega)$ is defined up to sets of zero $p$-capacity (which are thinner than sets of zero Lebesgue measure). For $p > n$, every set of zero capacity has zero Hausdorff dimension, which is why Morrey's embedding gives pointwise continuity. For $p \le n$, the exceptional sets are more subtle and require potential theory to describe.

The **precise representative** of a Sobolev function is defined at almost every point (in the capacity sense) by limits of averages: $u^*(x) = \lim_{r \to 0}\frac{1}{|B(x,r)|}\int_{B(x,r)}u$. This precise representative agrees with $u$ a.e. and provides the canonical pointwise interpretation.

### Duality between $H^s$ and $H^{-s}$

For $s > 0$, the dual of $H^s_0(\Omega)$ is $H^{-s}(\Omega)$. The duality pairing extends the $L^2$ inner product: for $u \in H^s_0$, $v \in H^{-s}$ smooth, $\langle u, v\rangle_{H^s, H^{-s}} = \int u\bar{v}\,dx$. This duality is the right-hand-side framework for Lax-Milgram problems: the data $f$ for $-\Delta u = f$ with Dirichlet boundary is naturally an element of $H^{-1}(\Omega)$, and the solution lives in $H^1_0(\Omega)$. The duality pairing is what makes the variational problem $\int \nabla u\cdot\nabla v = \langle f, v\rangle$ symbolically meaningful.

![Duality between H^s and H^{-s} via Fourier characterization](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/11-distributions-sobolev/fa_v2_11_6_dual_sobolev.png)

### Sobolev spaces on manifolds

The framework extends to compact Riemannian manifolds $(M, g)$ via charts and partitions of unity. Given an atlas $\{(U_\alpha, \phi_\alpha)\}$ and subordinate partition $\{\chi_\alpha\}$, set $u \in H^k(M)$ iff $(\chi_\alpha u)\circ\phi_\alpha^{-1} \in H^k(\phi_\alpha(U_\alpha))$ for each $\alpha$. The embedding and compactness theorems carry over with $n = \dim M$. This extension is essential for studying PDE on curved spaces — the Laplace-Beltrami operator is the natural generalization of $\Delta$, and its analysis requires Sobolev spaces on $M$.

### Examples of distributions in practice

A short tour of distributions one encounters daily:

- $\delta_a$ (point evaluation) — a tempered distribution of order $0$.
- $\delta'$ (derivative of delta) — order $1$, $\langle \delta', \varphi\rangle = -\varphi'(0)$.
- $\mathrm{p.v.}\,1/x$ — order $1$, the Hilbert transform's kernel.
- $\log|x|$ — locally integrable in $n \ge 1$, order $0$.
- $|x|^{-\alpha}$ for $\alpha < n$ — locally integrable, order $0$.
- $|x|^{-\alpha}$ for $n \le \alpha < n+1$ — defined as a distribution by analytic continuation in $\alpha$.
- $\mathrm{vp}(|x|^{-n})$ — the principal-value Calderon-Zygmund kernel, order $1$.
- $\delta_\Sigma$ for a hypersurface $\Sigma$ — a single-layer surface measure, order $0$.

![Examples of distributions: derivatives, principal values, surface measures](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/functional-analysis/11-distributions-sobolev/fa_v2_11_7_examples.png)

These objects have natural Fourier transforms, derivatives, and embeddings in fractional Sobolev spaces — all the structural results of the previous sections apply uniformly. That uniformity is what makes the distributional framework so powerful: it is a single language for objects that classical analysis has to treat as a zoo of unrelated special cases.

### Why this matters

Distribution theory and Sobolev spaces are not optional luxuries; they are the *only* framework in which existence and regularity for PDE work cleanly. Three concrete payoffs:

1. **Existence first, regularity later.** Lax-Milgram (next article) proves existence of weak solutions to elliptic boundary value problems in $H^1_0$ without any smoothness assumption on the data. Once existence is in hand, regularity theorems bootstrap: $f \in H^k$ implies $u \in H^{k+2}$, which by Sobolev embedding gives classical regularity once $k$ is large enough.

2. **Compactness powers spectral theory.** Rellich-Kondrachov compactness of $H^1_0(\Omega) \hookrightarrow L^2(\Omega)$ is exactly what makes the resolvent of the Dirichlet Laplacian a compact operator on $L^2$. The spectral theorem for compact self-adjoint operators then gives a complete orthonormal basis of eigenfunctions. Every eigenfunction expansion in mathematical physics — drum modes, atomic orbitals, vibration modes of structures — is downstream of Rellich-Kondrachov.

3. **Distributions accommodate idealisations physics needs.** The point charge, the impulse, the singular concentration — these are not artifacts to be smoothed away but genuine objects whose dynamics must be analysed. Distribution theory makes them rigorous and the analysis legitimate.

Without these tools the functional-analytic approach to PDE collapses: no complete spaces in which to seek solutions, no embedding results to extract regularity, no trace results to impose boundary conditions, no compactness for variational arguments.

A historical aside that I find clarifying: Schwartz received the Fields Medal in 1950 for distribution theory, and Sobolev's spaces were already a generation old by then. Yet most PDE textbooks before about 1965 still avoided Sobolev spaces, treating each problem with ad-hoc function classes. The modern unified framework — every PDE is a question about a bounded operator between Sobolev spaces — was only standardised in graduate textbooks in the 1970s. The conceptual lag between "the right framework exists" and "the right framework is taught" was about 25 years. I take some comfort in that delay: it suggests that the difficulty I felt the first time I encountered $H^{-1/2}(\partial\Omega)$ was not a personal failure but a normal feature of mathematical maturation. Some abstractions need a generation to settle.

### Common pitfalls

A few traps I have stepped in often enough to flag:

- The product of a distribution with a non-smooth function is not generally defined. $H \cdot \delta$ has no canonical meaning. Multiplication by smooth functions is fine; products of singular distributions are not.

- The pointwise restriction $u|_S$ of a Sobolev function $u \in H^1(\Omega)$ to a hypersurface $S$ is not always defined as an element of $L^2(S)$. The trace theorem gives $\gamma_0 u \in H^{1/2}(S)$ for $S = \partial\Omega$, but this is a fractional space and the constants depend on the regularity of $S$.

- "Weak" and "distributional" derivatives agree when the latter happens to be locally integrable, but not otherwise. $H'$ is the distribution $\delta$, not a function; saying "$H$ has weak derivative $\delta$" is a category error if "weak derivative" is supposed to be in $L^1_{\text{loc}}$. Be careful with the precise definitions.

- Embedding constants degrade with the geometry of $\Omega$. Long thin domains, domains with cusps, and domains with rough boundaries can have terrible Poincare and Sobolev constants. For numerical PDE, this is a daily concern.

- Compactness of $H^1_0(\Omega) \hookrightarrow L^2(\Omega)$ requires bounded $\Omega$. On unbounded domains, mass can escape to infinity, breaking Bolzano-Weierstrass.

A practical workflow I have settled on for new PDE problems: pick the right Sobolev space first (what is the natural energy?), check that the bilinear form is bounded and coercive on it, invoke Lax-Milgram for existence, then bootstrap regularity from the data class up. Most existence-and-regularity arguments are variations on that pattern, and getting the framework right at the start saves enormous backtracking later. The distributional and Sobolev-space machinery developed here is what lets that pattern work uniformly across elliptic, parabolic, and hyperbolic problems.

One final remark on terminology that confused me as a graduate student. The term "weak solution" overloads in two distinct ways. First, a weak solution of $-\Delta u = f$ in $H^1_0(\Omega)$ is a function satisfying the variational identity $\int \nabla u \cdot \nabla\varphi = \int f\varphi$ for all $\varphi \in H^1_0$. Second, a "distributional solution" is a distribution $u$ satisfying $-\Delta u = f$ in $\mathcal{D}'(\Omega)$. These coincide when $u \in H^1$ and $f \in H^{-1}$, but the two notions live in nominally different worlds. The Sobolev framework gives the variational reading; the distributional framework gives the wider one. Most PDE textbooks elide the distinction, but it matters when the data is rough enough that no Sobolev solution exists yet a distributional one does. The full picture has both notions in play, with the appropriate one selected by the data class.

---


### Worked Numerical Example
Take $\Omega = B_1(0) \subset \mathbb{R}^2$ and $u(x,y) = x$. The trace on $\partial\Omega$ is $\gamma_0 u(\theta) = \cos\theta$. Compute the $L^2$ norm of the trace: $\|\gamma_0 u\|_{L^2(\partial\Omega)}^2 = \int_0^{2\pi} \cos^2\theta \, d\theta = \pi$. Compute the $H^1$ norm of $u$ on the disk. $\|u\|_{L^2(\Omega)}^2 = \int_0^{2\pi} \int_0^1 r^2 \cos^2\theta \, r dr d\theta = \frac{1}{4} \cdot \pi = \frac{\pi}{4}$. The gradient is $\nabla u = (1, 0)$, so $\|\nabla u\|_{L^2(\Omega)}^2 = \int_\Omega 1 \, dx dy = \pi$. The squared $H^1$ norm is $\frac{\pi}{4} + \pi = \frac{5\pi}{4}$. The trace inequality demands $\|\gamma_0 u\|_{L^2} \le C \|u\|_{H^1}$. Plugging in the numbers: $\sqrt{\pi} \le C \sqrt{5\pi/4}$, which simplifies to $1 \le C \sqrt{5}/2$, or $C \ge 2/\sqrt{5} \approx 0.8944$. The trace operator is bounded, and the constant depends only on the domain geometry. If you attempt to restrict a generic $L^2$ function to the boundary, the left side is undefined. The $H^1$ control provides exactly enough regularity to make the boundary integral finite and continuous with respect to the bulk norm.

## Counterexample: Why the Definition Cannot Be Weakened
Morrey's inequality states that $W^{1,p}(\mathbb{R}^n) \hookrightarrow C^{0,\gamma}(\mathbb{R}^n)$ when $p > n$. A natural question is whether the strict inequality $p > n$ can be relaxed to $p = n$. The answer is no, and the failure is explicit. Consider $n=2$ and the function $u(x) = \log\log(1/|x|)$ defined on the ball $B_{1/e}(0)$. At the origin, $u$ blows up to $+\infty$, so it is unbounded and certainly not continuous. Compute its gradient for $x \neq 0$: $\nabla u(x) = \frac{1}{\log(1/|x|)} \cdot \frac{1}{1/|x|} \cdot \left(-\frac{x}{|x|^3}\right) = -\frac{x}{|x|^2 \log(1/|x|)}$. The magnitude is $|\nabla u(x)| = \frac{1}{|x| |\log|x||}$. Now check the $L^2$ norm of the gradient using polar coordinates:
$$
\int_{B_{1/e}} |\nabla u|^2 dx = 2\pi \int_0^{1/e} \frac{1}{r^2 (\log r)^2} r \, dr = 2\pi \int_0^{1/e} \frac{1}{r (\log r)^2} dr.
$$
Substitute $t = -\log r$, so $dt = -dr/r$. The limits transform from $r \in (0, 1/e]$ to $t \in [1, \infty)$. The integral becomes $2\pi \int_1^\infty t^{-2} dt = 2\pi [ -t^{-1} ]_1^\infty = 2\pi$. The gradient is square-integrable. Since $u$ is also square-integrable near the origin (the logarithmic singularity is mild in $L^2$), we have $u \in H^1(B_{1/e})$. Yet $u$ is unbounded. This single computation destroys any hope of embedding $W^{1,n}$ into $C^0$. The threshold $p=n$ is sharp. Below it, functions can have logarithmic spikes; above it, the derivative integrability forces Holder continuity. The definition of the Sobolev conjugate and the Morrey exponent cannot be weakened without losing pointwise control entirely.

## Why I Care
In 2018 I was debugging a 2D finite element solver for a microstrip transmission line. The code assembled the stiffness matrix correctly, but the solution refused to converge under mesh refinement. I had imposed a Dirichlet ground condition at a single interior node, treating it as a physical probe. The solver produced a logarithmic spike around that node that grew sharper as $h \to 0$. My advisor looked at the output, pointed at the singularity, and said: "Points have zero capacity in two dimensions. You are trying to evaluate an $H^1$ function at a point. The trace theorem says you can only restrict to curves." I went back to the theory, computed the $2$-capacity of a point in $\mathbb{R}^2$, and found it was exactly zero. My Dirichlet condition was mathematically void; the variational formulation simply ignored it, and the spike was the solver's attempt to satisfy an impossible constraint. I replaced the point constraint with a small circular boundary condition of radius $0.01$. The solver converged in four Newton steps. That afternoon I stopped treating Sobolev spaces as abstract exam material and started reading them as a manual for what my discretization was actually allowed to do. The trace theorem and capacity theory went from definitions on a page to debugging tools that saved a week of wasted compute time.

## Common Pitfall
A persistent misconception is that if a function is differentiable almost everywhere, its classical a.e. derivative equals its weak derivative. This is false. Consider $f(x) = \mathrm{sgn}(x)$ on $(-1, 1)$. Classically, $f$ is differentiable everywhere except at $x=0$. The a.e. derivative is $f'_{\mathrm{ae}}(x) = 0$ for all $x \neq 0$. If this were the weak derivative, the defining identity would require $\int_{-1}^1 f \varphi' dx = -\int_{-1}^1 0 \cdot \varphi dx = 0$ for every $\varphi \in C_c^\infty(-1,1)$. Compute the left side directly:
$$
\int_{-1}^1 \mathrm{sgn}(x) \varphi'(x) dx = \int_0^1 \varphi'(x) dx - \int_{-1}^0 \varphi'(x) dx = (\varphi(1)-\varphi(0)) - (\varphi(0)-\varphi(-1)).
$$
Since $\varphi$ has compact support in $(-1,1)$, $\varphi(1)=\varphi(-1)=0$. The result is $-2\varphi(0)$, which equals $\langle -2\delta_0, \varphi \rangle$. The weak derivative is $2\delta_0$, not the zero function. The a.e. derivative misses the jump entirely because Lebesgue integration ignores sets of measure zero, while distributional derivatives record singular concentrations. You cannot patch classical a.e. derivatives into Sobolev spaces without verifying absolute continuity. The bridge between classical and weak differentiation is not a.e. differentiability; it is the fundamental theorem of calculus holding in integral form. Ignore that distinction, and your integration-by-parts arguments will silently drop delta terms.

## What's Next

We have built the distributional and Sobolev-space foundations needed for modern PDE theory: distributions give meaning to derivatives that don't exist classically, Sobolev spaces provide Banach/Hilbert settings for weak formulations, embedding theorems connect integrability to regularity, and trace theorems handle boundary conditions rigorously.

In the final article of this series, we put everything together. The **Lax-Milgram theorem** gives existence and uniqueness for elliptic boundary value problems. **Variational methods** convert PDE into minimization problems. **Stone's theorem** connects self-adjoint operators to quantum dynamics. The abstract machinery of functional analysis meets its most important applications.

---

### Specific Questions Ahead
The machinery built here does not sit idle. In the next article, we turn the abstract existence question into a computational guarantee. I will answer four specific questions that naturally arise once you accept weak formulations:
1. How do we prove that a weak solution actually exists without constructing it explicitly or solving the PDE directly?
2. What precise conditions on the bilinear form $a(u,v)$ and the linear functional $F(v)$ guarantee uniqueness and continuous dependence on the data?
3. How does the geometry of the domain $\Omega$ enter the stability constants, and why do re-entrant corners destroy $H^2$ regularity even when the data is smooth?
4. Why does the finite element method converge at exactly the rate predicted by polynomial degree, and where does the Sobolev norm hide in the error estimate?

### Why You Are Equipped
You are now equipped to read the answers because the heavy lifting is already done. You know that $H^1_0(\Omega)$ is a Hilbert space, so the Riesz representation theorem applies directly to bounded linear functionals. You know that the trace operator handles Dirichlet data without pointwise evaluation, and that $W^{1,p}_0$ is precisely the kernel of that operator. You know that Rellich-Kondrachov compactness turns bounded sequences into convergent subsequences, which is the engine behind eigenvalue approximations and the Fredholm alternative. The next article assumes fluency with these facts and moves directly to operator bounds and variational stability. You no longer need to worry about whether derivatives exist; you only need to check whether integrals are bounded and coercive.

### Preview: The Lax-Milgram Theorem
The central result is the Lax-Milgram theorem. I will state it in its standard form: given a bounded, coercive bilinear form $a: V \times V \to \mathbb{R}$ on a Hilbert space $V$ and a bounded linear functional $F \in V'$, there exists a unique $u \in V$ such that $a(u,v) = F(v)$ for all $v \in V$, with the stability estimate $\|u\|_V \le \frac{1}{\alpha}\|F\|_{V'}$ where $\alpha$ is the coercivity constant. The proof is a direct application of the Riesz representation theorem composed with a bounded invertible operator induced by $a$. I will also derive C\'ea's lemma, which bounds the finite element error by the best approximation error in the energy norm: $\|u - u_h\|_V \le \frac{M}{\alpha} \inf_{v_h \in V_h} \|u - v_h\|_V$. This inequality is the reason engineers trust mesh refinement. You will see exactly how the Sobolev embedding constants and the Poincar\'e inequality feed into $M$ and $\alpha$, closing the loop between the abstract space and the concrete stiffness matrix. The transition from distributional derivatives to computable linear systems is shorter than it looks, and the estimates you verify here are the same ones that prevent numerical blow-up in production codes.
