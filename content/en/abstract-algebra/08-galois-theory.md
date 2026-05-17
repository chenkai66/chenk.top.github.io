---
title: "Abstract Algebra (8): Galois Theory — The Bridge Between Fields and Groups"
date: 2021-09-15 09:00:00
tags:
  - abstract-algebra
  - galois-theory
  - field-theory
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
description: "The Fundamental Theorem of Galois Theory establishes a perfect correspondence between intermediate fields and subgroups — and settles the ancient question of solvability by radicals."
disableNunjucks: true
series_order: 8
series_total: 12
translationKey: "abstract-algebra-8"
---

In 1832, a twenty-year-old mathematician named Evariste Galois, on the eve of a duel he would not survive, wrote down the final versions of his mathematical ideas in a letter to a friend. Those ideas — connecting the symmetries of polynomial roots to the structure of groups — would take more than a decade to be understood and published, but they would reshape algebra forever. Galois theory, as we now call it, establishes a precise dictionary between intermediate fields of a field extension and subgroups of a group of symmetries. It explains, in one elegant framework, why the quadratic formula exists, why there is no analogous formula for degree five, and what "solvability" really means.

What I find astonishing about Galois's idea is the *direction* of the abstraction. He did not study polynomials by computing their roots more cleverly. He studied them by ignoring the roots entirely and instead analyzing the *permutations* of those roots that preserve all algebraic relations among them. The set of such permutations forms a group, and the structure of that group tells you everything you wanted to know about the original polynomial. It is a complete change of subject — from numbers to groups — that nevertheless answers the original question. This article walks through that change of subject in detail.

---

## The Galois Group: Automorphisms Fixing the Base Field

Given a field extension $L/K$, an *automorphism of $L$ over $K$* is a field isomorphism $\sigma : L \to L$ such that $\sigma(a) = a$ for every $a \in K$. The collection of all such automorphisms forms a group under composition, called the *Galois group* of $L/K$:

$$\mathrm{Gal}(L/K) = \mathrm{Aut}_K(L) = \{\sigma : L \to L \mid \sigma \text{ is a field automorphism},\ \sigma|_K = \mathrm{id}\}.$$

The key insight is that automorphisms of $L$ that fix $K$ permute the roots of any polynomial in $K[x]$. If $f(x) \in K[x]$ and $f(\alpha) = 0$ for some $\alpha \in L$, then applying $\sigma \in \mathrm{Gal}(L/K)$ to both sides:

$$0 = \sigma(0) = \sigma(f(\alpha)) = f(\sigma(\alpha)),$$

since $\sigma$ fixes the coefficients of $f$. So $\sigma(\alpha)$ is also a root of $f$. The Galois group acts on the roots of $f$ by permutation, and that action is faithful when $L$ is a splitting field — once you know what $\sigma$ does to the roots, you know $\sigma$ everywhere.

![The Galois group of Q(√2)/Q acting by sign flips](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/08-galois-theory/aa_v2_08_1_galois_group.png)

**Example ($\mathbb{Q}(\sqrt{2})/\mathbb{Q}$).** Any $\sigma \in \mathrm{Gal}(\mathbb{Q}(\sqrt{2})/\mathbb{Q})$ must send $\sqrt{2}$ to a root of $x^2 - 2$, namely $\pm\sqrt{2}$. So there are exactly two automorphisms: the identity and $\sigma(\sqrt{2}) = -\sqrt{2}$. The Galois group is $\mathbb{Z}/2\mathbb{Z}$.

Note: $\sigma$ is determined by its action on $\sqrt{2}$, since every element of $\mathbb{Q}(\sqrt{2})$ has the form $a + b\sqrt{2}$ with $a, b \in \mathbb{Q}$, and $\sigma(a + b\sqrt{2}) = a + b\sigma(\sqrt{2})$. The arithmetic identity that survives sign-flip is exactly $(a+b\sqrt{2})(a-b\sqrt{2}) = a^2 - 2b^2$ — the *norm* — which is automatically $\mathbb{Q}$-valued and Galois-invariant.

**Example ($\mathbb{Q}(\sqrt{2}, \sqrt{3})/\mathbb{Q}$).** Let $L = \mathbb{Q}(\sqrt{2}, \sqrt{3})$. An automorphism $\sigma$ fixing $\mathbb{Q}$ sends $\sqrt{2} \mapsto \pm\sqrt{2}$ and $\sqrt{3} \mapsto \pm\sqrt{3}$ independently, giving four automorphisms: identity, $\sqrt{2} \mapsto -\sqrt{2}$, $\sqrt{3} \mapsto -\sqrt{3}$, and the product. The Galois group is $\mathbb{Z}/2 \times \mathbb{Z}/2$, the Klein four-group $V_4$. And $|\mathrm{Gal}(L/\mathbb{Q})| = 4 = [L:\mathbb{Q}]$, matching the degree.

**Example ($\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$).** The minimal polynomial $x^3 - 2$ has only one real root, $\sqrt[3]{2}$, while the other two are complex. Any $\sigma : \mathbb{Q}(\sqrt[3]{2}) \to \mathbb{Q}(\sqrt[3]{2})$ must send $\sqrt[3]{2}$ to a root of $x^3 - 2$ inside $\mathbb{Q}(\sqrt[3]{2}) \subset \mathbb{R}$. Only one such root exists. So $\sigma = \mathrm{id}$ and $\mathrm{Gal}(\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q})$ is trivial. Yet $[\mathbb{Q}(\sqrt[3]{2}):\mathbb{Q}] = 3 \neq 1$.

This is the punchline of separability and normality from the previous article: the equality $|\mathrm{Gal}(L/K)| = [L:K]$ holds *exactly* for Galois extensions. Here $\mathbb{Q}(\sqrt[3]{2})/\mathbb{Q}$ fails normality (the extension is not a splitting field), and the group collapses.

**Theorem.** A finite extension $L/K$ is Galois iff $|\mathrm{Gal}(L/K)| = [L:K]$.

This is one of those equivalences whose two halves are both useful: in one direction it lets us check the Galois condition by counting automorphisms, in the other it lets us verify that we have found *all* automorphisms by checking the count against the degree.

It also lines up with the abstract definition of "Galois" via separability + normality. A separable extension has at most $[L:K]$ embeddings into the algebraic closure (by counting roots of minimal polynomials, all distinct); a normal extension has those embeddings actually land back in $L$ (rather than escaping to the algebraic closure); so a Galois extension has exactly $[L:K]$ automorphisms — none missing, none extra. The "embedding count = degree" identity is exactly the equality $|\mathrm{Gal}(L/K)| = [L:K]$ in disguise.

**Why this matters.** The Galois group is a small, finite, computable object that nevertheless captures the entire content of a field extension. Group theory has a 200-year head start of structural results — Sylow theorems, simple groups classification, representation theory — and Galois theory lets us pull all of it into the study of polynomial roots. That trade is worth essentially everything we paid in Part 7 to set up the machinery.

There is a useful conceptual reframing here. The Galois group is the *automorphism group of the polynomial $f$*, in a sense made precise by the action on the roots. We could equivalently define $\mathrm{Gal}(f)$ as the subgroup of $S_n$ (where $n = \deg f$) consisting of those permutations of the roots that respect every algebraic relation among them. The "respect every algebraic relation" clause is what makes the definition non-trivial — without it, every permutation would be an automorphism, and the Galois group would always be $S_n$. The fact that some permutations *do not* extend to field automorphisms is exactly the information you exploit when computing the group.

A small worked check. For $f(x) = x^4 + 1$ (the eighth cyclotomic polynomial), the four roots are $\zeta_8, \zeta_8^3, \zeta_8^5, \zeta_8^7$, the primitive eighth roots of unity. There is an algebraic relation $\zeta_8^3 = \zeta_8 \cdot \zeta_8^2 = \zeta_8 \cdot i$, and another $\zeta_8 + \zeta_8^7 = \sqrt{2}$. Permutations of the four roots that respect all such identities form a group of order 4 — namely $(\mathbb{Z}/8)^\times$ — even though there are $4! = 24$ permutations on the underlying set. Most permutations break some relation; the surviving four are exactly the Galois group.

---

## Fixed Fields and the Galois Correspondence

In one direction, given an extension $L/K$, we get a group $\mathrm{Gal}(L/K)$. In the other direction, given a subgroup $H \leq \mathrm{Gal}(L/K)$, we get a field:

$$L^H = \{\alpha \in L : \sigma(\alpha) = \alpha \text{ for all } \sigma \in H\}.$$

This is the *fixed field* of $H$. It is a subfield of $L$ containing $K$ (since elements of $K$ are fixed by everything in $\mathrm{Gal}(L/K)$).

We thus have two maps:

- $\Phi$: subgroups of $\mathrm{Gal}(L/K)$ → intermediate fields, $H \mapsto L^H$.
- $\Psi$: intermediate fields → subgroups, $M \mapsto \mathrm{Gal}(L/M)$.

**Theorem (Galois Correspondence).** If $L/K$ is a finite Galois extension, then $\Phi$ and $\Psi$ are mutually inverse, order-reversing bijections between the set of subgroups of $\mathrm{Gal}(L/K)$ and the set of intermediate fields $K \subseteq M \subseteq L$.

![Galois correspondence: subgroups of the Galois group match intermediate fields](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/08-galois-theory/aa_v2_08_2_correspondence.png)

The order-reversing aspect is the part that surprises people on a first encounter, but it is forced: the larger the subgroup, the more constraints we impose, the smaller the fixed field. Symbolically, $H_1 \leq H_2 \implies L^{H_1} \supseteq L^{H_2}$.

Two formulas pin down the dictionary:

- $[L : L^H] = |H|$ (the degree equals the size of the fixing group).
- $[L^H : K] = [\mathrm{Gal}(L/K) : H]$ (the degree from the bottom equals the index in the full group).

Multiplying gives $|\mathrm{Gal}(L/K)| = [L:K]$, which is the Galois condition.

The proof of the correspondence rests on two lemmas:

**Artin's Lemma.** If $G$ is a finite group of automorphisms of $L$, then $[L : L^G] = |G|$. So *every* finite group acting faithfully on $L$ realizes $L$ as a Galois extension of its fixed field.

**Galois's Theorem.** If $L/K$ is finite Galois, then $L^{\mathrm{Gal}(L/K)} = K$. So no element of $L$ outside $K$ is fixed by *every* automorphism — automorphisms see everything.

Combine the two: starting with $L/K$ Galois, the subgroup $\mathrm{Gal}(L/K)$ has fixed field $K$, and applying Artin to subgroups $H$ gives the equality $[L : L^H] = |H|$ that drives the bijection.

**Why this matters.** This is the bridge from field-theoretic questions to group-theoretic answers. Want to know how many subfields lie strictly between $K$ and $L$? Count proper non-trivial subgroups of $\mathrm{Gal}(L/K)$. Want to know which subfields are themselves Galois over $K$? Look at *normal* subgroups. Every structural question about the extension has a group-theoretic shadow that is, in practice, much easier to compute.

A practical lemma worth carrying around: if $H_1, H_2 \leq G$ are subgroups with fixed fields $M_1, M_2$, then:

- $L^{H_1 \cap H_2} = M_1 \cdot M_2$ (the compositum, the smallest field containing both).
- $L^{\langle H_1, H_2 \rangle} = M_1 \cap M_2$.

So the lattice operations on subgroups (intersection, generated subgroup) translate to lattice operations on subfields (compositum, intersection), with the order reversed. Once you internalize this, drawing the subfield lattice of any concrete extension becomes a purely group-theoretic exercise.

There is also a strong uniqueness statement hiding in the correspondence. If $L/K$ is Galois and $\sigma \in \mathrm{Gal}(L/K)$ acts trivially on every intermediate field, then $\sigma = \mathrm{id}$. Equivalently, the only element fixing all subfields is the identity. This is a way of saying that the lattice of subfields, together with the Galois group action, contains all the information of the extension.

---

## The Fundamental Theorem of Galois Theory

Putting the pieces together, we get the central result:

**Fundamental Theorem of Galois Theory.** Let $L/K$ be a finite Galois extension with Galois group $G = \mathrm{Gal}(L/K)$. Then:

1. *(Bijection)* The maps $H \mapsto L^H$ and $M \mapsto \mathrm{Gal}(L/M)$ are mutually inverse, order-reversing bijections between subgroups of $G$ and intermediate fields $K \subseteq M \subseteq L$.

2. *(Degrees match)* For every subgroup $H \leq G$, $[L : L^H] = |H|$ and $[L^H : K] = [G : H]$.

3. *(Normality)* An intermediate extension $M/K$ is Galois (equivalently, normal) iff $\mathrm{Gal}(L/M)$ is a *normal subgroup* of $G$. In that case, $\mathrm{Gal}(M/K) \cong G/\mathrm{Gal}(L/M)$.

The third part is the most striking. The word "normal" appears on both sides — for fields it means "splitting field of some polynomial," for groups it means "closed under conjugation" — and the theorem says these are the same condition viewed from opposite ends of the dictionary. That coincidence of terminology is no coincidence at all; "normal" was originally defined to make the correspondence work out.

![Normal subgroups correspond to normal field extensions](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/08-galois-theory/aa_v2_08_7_normal_field.png)

**Sketch of part (3).** If $H \trianglelefteq G$, define a homomorphism $G \to \mathrm{Aut}_K(L^H)$ by restricting each $\sigma \in G$ to $L^H$. (Restriction makes sense because $L^H$ is mapped to itself by $\sigma$: if $\alpha \in L^H$ and $\tau \in H$, then $\tau\sigma(\alpha) = \sigma\sigma^{-1}\tau\sigma(\alpha) = \sigma(\alpha)$ since $\sigma^{-1}\tau\sigma \in H$ by normality.) The kernel is exactly $H$, so $G/H$ embeds into $\mathrm{Aut}_K(L^H)$. Counting: $|G/H| = [L^H : K]$, so by the Galois condition $G/H \cong \mathrm{Gal}(L^H/K)$ and $L^H/K$ is Galois.

Conversely, if $M/K$ is Galois, then every $\sigma \in G$ maps $M$ to itself (because $M$ is a splitting field, hence stable under all $K$-automorphisms of $L$), so the restriction map $G \to \mathrm{Aut}_K(M)$ is well-defined, and its kernel is $\mathrm{Gal}(L/M)$, which is therefore normal.

The "stable under $\sigma$" step in the converse is worth a closer look. Suppose $M = K(\alpha_1, \ldots, \alpha_r)$ is the splitting field of $f \in K[x]$ inside $L$. For any $\sigma \in G$, $\sigma$ sends each root $\alpha_i$ of $f$ to another root $\alpha_j$ of $f$ (because $\sigma$ fixes the coefficients of $f$). The other roots are also in $M$ (that is what splitting field means), so $\sigma(\alpha_i) \in M$. Therefore $\sigma(M) \subseteq M$. By dimension counting (or by injectivity of $\sigma$), $\sigma(M) = M$. So restriction is well-defined.

**Why this matters.** The Fundamental Theorem is the engine that powers every concrete computation in Galois theory. If you want to find the subfields of $L$, draw the subgroup lattice of $G$ — those are the same lattice. If you want to know which subfield is Galois over $K$, mark the normal subgroups. The whole question moves from a subject (fields) where you have to reason about elements to a subject (groups) where you can reason combinatorially.

Three additional structural payoffs of the FTGT that are worth being explicit about:

1. *Counting intermediate fields.* The number of subfields of $L$ containing $K$ equals the number of subgroups of $G$. For $G \cong S_3$ this is $1 + 3 + 1 + 1 = 6$ subgroups, hence 6 intermediate fields. For $G \cong (\mathbb{Z}/p)^n$ this is the number of subspaces of $(\mathbb{F}_p)^n$, given by Gaussian binomial coefficients.

2. *Detecting Galois closure.* The smallest Galois extension of $K$ containing a given $M \subseteq L$ is the fixed field of the largest normal subgroup of $G$ contained in $\mathrm{Gal}(L/M)$. So Galois closures correspond to normal cores. This shows up constantly when you start with a non-Galois extension and want to do Galois theory anyway.

3. *Coupling with quotient groups.* Whenever $H \trianglelefteq G$, there is an induced action of $G/H$ on $L^H$, and that action is exactly $\mathrm{Gal}(L^H/K)$. This often lets you reduce computations in a big Galois group to two smaller ones — one for $H$, one for $G/H$.

---

## Computing Galois Groups: Concrete Examples

The theory is elegant, but examples drive the intuition. Let us actually compute some Galois groups.

### Example 1: $\mathbb{Q}(\sqrt{2}, \sqrt{3})/\mathbb{Q}$

We already saw $G \cong V_4 = \mathbb{Z}/2 \times \mathbb{Z}/2$. The subgroups are:

- $\{e\}$, fixed field $L = \mathbb{Q}(\sqrt{2}, \sqrt{3})$, degree 4 over $\mathbb{Q}$.
- $\langle \sigma_2 \rangle$ (where $\sigma_2 : \sqrt{2} \mapsto -\sqrt{2}$, $\sqrt{3} \mapsto \sqrt{3}$), fixed field $\mathbb{Q}(\sqrt{3})$, degree 2.
- $\langle \sigma_3 \rangle$ (where $\sigma_3 : \sqrt{2} \mapsto \sqrt{2}$, $\sqrt{3} \mapsto -\sqrt{3}$), fixed field $\mathbb{Q}(\sqrt{2})$, degree 2.
- $\langle \sigma_2 \sigma_3 \rangle$ (sends $\sqrt{2} \mapsto -\sqrt{2}$, $\sqrt{3} \mapsto -\sqrt{3}$, hence $\sqrt{6} \mapsto \sqrt{6}$), fixed field $\mathbb{Q}(\sqrt{6})$, degree 2.
- $G$, fixed field $\mathbb{Q}$, degree 1.

Five subgroups, five intermediate fields, all matching by degree. Since $V_4$ is abelian, every subgroup is normal, so every intermediate field is Galois over $\mathbb{Q}$ — easy to verify directly since each is a splitting field of a quadratic.

This is the prototypical "biquadratic" extension. It also illustrates how the apparently mysterious element $\sqrt{6} = \sqrt{2}\sqrt{3}$ ends up fixed by the diagonal automorphism: since both square roots flip sign together, their product is invariant. This is the same trick that produces the Pell-equation lattice $\mathbb{Z}[\sqrt{6}]$ as the ring of integers of $\mathbb{Q}(\sqrt{6})$, sitting cleanly inside $\mathbb{Z}[\sqrt{2}, \sqrt{3}]$.

### Example 2: Splitting Field of $x^3 - 2$ over $\mathbb{Q}$

Let $L = \mathbb{Q}(\sqrt[3]{2}, \omega)$ where $\omega = e^{2\pi i/3}$. From Part 7, $[L:\mathbb{Q}] = 6$.

The three roots of $x^3 - 2$ are $\sqrt[3]{2}$, $\sqrt[3]{2}\omega$, $\sqrt[3]{2}\omega^2$. The Galois group permutes these three roots, embedding into $S_3$. Since the order is 6, $G \cong S_3$. We can write generators:

- $\sigma$ (order 3): $\sqrt[3]{2} \mapsto \sqrt[3]{2}\omega$, $\omega \mapsto \omega$.
- $\tau$ (order 2): $\sqrt[3]{2} \mapsto \sqrt[3]{2}$, $\omega \mapsto \omega^2$ (complex conjugation).

Subgroup lattice of $S_3$:

- $\{e\}$ — fixed field $L$.
- $\langle \tau \rangle$, $\langle \sigma\tau \rangle$, $\langle \sigma^2\tau \rangle$ — three order-2 subgroups, fixed fields $\mathbb{Q}(\sqrt[3]{2})$, $\mathbb{Q}(\sqrt[3]{2}\omega^2)$, $\mathbb{Q}(\sqrt[3]{2}\omega)$ (each is one of the three real or complex cube roots).
- $\langle \sigma \rangle = A_3$ — fixed field $\mathbb{Q}(\omega)$, degree 2.
- $S_3$ — fixed field $\mathbb{Q}$.

The three order-2 subgroups are *not* normal in $S_3$ (they are conjugate to each other), and correspondingly the three fields $\mathbb{Q}(\sqrt[3]{2}\omega^k)$ are not Galois over $\mathbb{Q}$ — none of them is the splitting field of anything, and none is closed under all the automorphisms. The subgroup $A_3$ *is* normal (index 2 in $S_3$), and correspondingly $\mathbb{Q}(\omega)/\mathbb{Q}$ *is* Galois (it is the splitting field of $x^2 + x + 1$, with Galois group $\mathbb{Z}/2$).

This is the canonical example of how non-normality on the field side appears as non-normality on the group side.

A short numerical check on the FTGT here. Take the subgroup $\langle \tau \rangle$ of order 2 with fixed field $\mathbb{Q}(\sqrt[3]{2})$, of degree 3 over $\mathbb{Q}$. The FTGT predicts $|G| / |H| = 6/2 = 3$, matching. Now take $A_3$, order 3, fixed field $\mathbb{Q}(\omega)$, of degree 2: ratio $6/3 = 2$, matching. The numerics are pedestrian; what is striking is that *every* subgroup of $S_3$ has a unique field that knows about it, and vice versa.

### Example 3: Splitting Field of $x^4 - 2$ over $\mathbb{Q}$

Let $L = \mathbb{Q}(\sqrt[4]{2}, i)$. From Part 7, $[L:\mathbb{Q}] = 8$. The Galois group acts on the four roots $\pm\sqrt[4]{2}, \pm i\sqrt[4]{2}$. Generators:

- $r$ (order 4): $\sqrt[4]{2} \mapsto i\sqrt[4]{2}$, $i \mapsto i$. (Cycles the four roots.)
- $s$ (order 2): $\sqrt[4]{2} \mapsto \sqrt[4]{2}$, $i \mapsto -i$. (Complex conjugation.)

These satisfy $r^4 = s^2 = 1$, $srs = r^{-1}$. So $G \cong D_4$, the dihedral group of order 8.

![Full Galois correspondence for the splitting field of x^4 - 2](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/08-galois-theory/aa_v2_08_3_x4_minus_2.png)

$D_4$ has 10 subgroups: $\{e\}$, three subgroups of order 2 ($\langle r^2 \rangle$, $\langle s \rangle$, $\langle rs \rangle$, $\langle r^2 s \rangle$, $\langle r^3 s \rangle$ — five total), three subgroups of order 4 ($\langle r \rangle$, $\langle r^2, s \rangle$, $\langle r^2, rs \rangle$), and $D_4$. So there are 10 intermediate fields.

The center of $D_4$ is $\langle r^2 \rangle$, normal. The three index-2 subgroups (of order 4) are all normal, giving three Galois sub-extensions of $\mathbb{Q}$. Working out the fixed fields explicitly is a satisfying exercise; among them you find $\mathbb{Q}(i)$, $\mathbb{Q}(\sqrt{2})$, $\mathbb{Q}(i, \sqrt{2})$, and a few less obvious ones like $\mathbb{Q}(\sqrt{2}\cdot i)$.

Two non-normal subgroups of order 2 are conjugate, namely $\langle s \rangle$ and $\langle r^2 s \rangle$ (and likewise $\langle rs \rangle, \langle r^3 s \rangle$). Their fixed fields $\mathbb{Q}(\sqrt[4]{2})$ and $\mathbb{Q}(i\sqrt[4]{2})$ are not Galois over $\mathbb{Q}$, but they are isomorphic to each other (one is the conjugate of the other). The reflection of group-theoretic conjugacy as field-isomorphism-but-not-equality is something I find satisfying every time I see it: the FTGT does not just match subgroups to subfields, it matches conjugacy classes to isomorphism classes of "the same" field embedded differently.

### Example 4: Cyclotomic Fields

The $n$-th cyclotomic field is $\mathbb{Q}(\zeta_n)$, where $\zeta_n = e^{2\pi i/n}$. The minimal polynomial of $\zeta_n$ over $\mathbb{Q}$ is the $n$-th cyclotomic polynomial $\Phi_n(x)$, of degree $\varphi(n)$.

**Theorem.** $\mathrm{Gal}(\mathbb{Q}(\zeta_n)/\mathbb{Q}) \cong (\mathbb{Z}/n\mathbb{Z})^\times$.

The isomorphism sends $\sigma \in \mathrm{Gal}$ to the unique $a \in (\mathbb{Z}/n\mathbb{Z})^\times$ such that $\sigma(\zeta_n) = \zeta_n^a$. The proof comes down to (i) showing $\Phi_n$ is irreducible over $\mathbb{Q}$ — a classical theorem of Gauss — and (ii) noting that the $\varphi(n)$ primitive $n$-th roots of unity are exactly $\zeta_n^a$ for $\gcd(a, n) = 1$, so each Galois automorphism corresponds to a choice of such $a$.

![Cyclotomic extensions and their abelian Galois groups](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/08-galois-theory/aa_v2_08_6_cyclotomic.png)

Since $(\mathbb{Z}/n\mathbb{Z})^\times$ is abelian, all cyclotomic extensions of $\mathbb{Q}$ are abelian. The Kronecker-Weber theorem (1853, completed 1886) says the *converse* holds: every finite abelian extension of $\mathbb{Q}$ is contained in some cyclotomic field. So abelian extensions of $\mathbb{Q}$ are *exactly* the subfields of cyclotomic fields. This is the prototype of class field theory.

For example, $\mathrm{Gal}(\mathbb{Q}(\zeta_8)/\mathbb{Q}) \cong (\mathbb{Z}/8)^\times \cong \mathbb{Z}/2 \times \mathbb{Z}/2$. The intermediate fields are $\mathbb{Q}(\sqrt{2})$, $\mathbb{Q}(i)$, $\mathbb{Q}(i\sqrt{2})$ — all real or imaginary quadratics, sitting inside the eighth cyclotomic field. In particular, the quadratic Gauss sum identity
$$\zeta_8 + \zeta_8^{-1} = \sqrt{2}$$
is no longer mysterious: it is the trace from $\mathbb{Q}(\zeta_8)$ down to its index-2 subfield $\mathbb{Q}(\sqrt{2})$, fixed by complex conjugation.

### Example 5: Finite Fields

The Galois group $\mathrm{Gal}(\mathbb{F}_{p^n}/\mathbb{F}_p)$ is cyclic of order $n$, generated by the Frobenius automorphism $\mathrm{Frob}_p : x \mapsto x^p$. So every finite extension of a finite field is Galois with cyclic Galois group, and the entire subgroup lattice is the divisor lattice of $n$. This is the cleanest possible Galois theory: $\mathbb{F}_{p^d}$ sits inside $\mathbb{F}_{p^n}$ iff $d \mid n$, and the Galois group of $\mathbb{F}_{p^n}/\mathbb{F}_{p^d}$ is cyclic of order $n/d$, generated by $\mathrm{Frob}_p^d$.

**Why this matters.** Galois group computations are not just sport. They are the algorithmic step in a great deal of algebraic number theory: identifying when a polynomial's roots admit closed-form expressions, computing class numbers, building reciprocity laws, doing modern cryptography. The number-field package in any computer algebra system is essentially a Galois-group calculator wrapped in a database of factorizations.

Some practical computational notes:

- For a degree-$n$ irreducible $f \in \mathbb{Q}[x]$, the Galois group is a transitive subgroup of $S_n$. There are 5 transitive subgroups of $S_4$, 5 of $S_5$, 16 of $S_6$, 7 of $S_7$, 50 of $S_8$, and so on. Identifying which one a given $f$ produces is a finite (but sometimes annoying) check. PARI/GP and SageMath both ship with `polgalois` / `f.galois_group()` for this.

- The factorization pattern of $f \bmod p$ for various primes $p$ tells you about the cycle structure of Frobenius in $\mathrm{Gal}(f)$. Chebotarev's density theorem says that as $p$ varies, every cycle structure appearing in $\mathrm{Gal}(f)$ shows up with the right density. So you can guess the Galois group by factoring $f$ mod many primes and matching cycle types — a delightfully effective heuristic.

- The discriminant of $f$ is a square in $\mathbb{Q}$ iff $\mathrm{Gal}(f) \subseteq A_n$. So $\mathrm{disc}(f)$ being or not being a square cuts the candidate Galois groups roughly in half.

---

## Solvability by Radicals and Solvable Groups

We now translate the question "can this polynomial be solved by radicals?" into group theory.

**Definition.** A polynomial $f(x) \in K[x]$ is *solvable by radicals* over $K$ if there is a tower
$$K = K_0 \subseteq K_1 \subseteq \cdots \subseteq K_r$$
in which each step is obtained by adjoining a radical (i.e., $K_{i+1} = K_i(\sqrt[n_i]{a_i})$ for some $n_i \geq 1$ and $a_i \in K_i$), and the splitting field of $f$ is contained in $K_r$.

In words: you can write the roots of $f$ using $+, -, \times, \div$ and $n$-th roots, starting from $K$.

**Definition.** A group $G$ is *solvable* if it has a chain of subgroups
$$\{e\} = G_0 \trianglelefteq G_1 \trianglelefteq \cdots \trianglelefteq G_n = G$$
such that each quotient $G_{i+1}/G_i$ is abelian.

![Solvable group: chain of normal subgroups with abelian quotients](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/08-galois-theory/aa_v2_08_4_solvable_chain.png)

**Examples.**

- All abelian groups (chain of length 1).
- All groups of order $< 60$.
- $S_n$ for $n \leq 4$ ($S_3$ has $\{e\} \trianglelefteq A_3 \trianglelefteq S_3$ with abelian quotients $\mathbb{Z}/3, \mathbb{Z}/2$; $S_4$ has $\{e\} \trianglelefteq V_4 \trianglelefteq A_4 \trianglelefteq S_4$).
- Dihedral groups $D_n$.
- Every $p$-group is solvable.

**Non-examples.** $S_n$ for $n \geq 5$ — because $A_5$ is simple and non-abelian, so it has no nontrivial normal subgroups, and any solvable chain through $S_5$ would have to break at $A_5$.

A few useful structural facts about solvable groups:

- Subgroups and quotients of solvable groups are solvable.
- An extension of a solvable group by a solvable group is solvable. (So solvability is "closed under group extensions.")
- Burnside (1904): every group of order $p^a q^b$ is solvable. The proof uses character theory and was the first major application of representation theory to abstract group theory.
- Feit-Thompson (1963): every group of odd order is solvable. The proof runs over 250 pages and was the spark that started the classification of finite simple groups.

The two cited theorems show how much group theory has to say about *which* groups are solvable. For our purposes, the relevant fact is that solvability is a local property of finite groups that you can check by chasing normal subgroups — exactly the kind of question that responds to the Sylow theorems.

**Theorem (Galois).** $f(x) \in K[x]$ is solvable by radicals over $K$ iff its Galois group $\mathrm{Gal}(L/K)$ (where $L$ is the splitting field of $f$) is a solvable group.

The forward direction is the hard part. The idea: a tower of radical extensions has Galois group built up from cyclic pieces (each $\sqrt[n]{a}$ generates a cyclic-Galois extension once you have enough roots of unity), and "built up from abelian pieces" is exactly the definition of solvable. So if $f$ is solvable by radicals, $\mathrm{Gal}(L/K)$ embeds in a solvable group, hence is solvable. The converse is also true and slightly more constructive: a solvable Galois group can be unwound into radical extensions.

**Why this matters.** The statement turns the question "is there a closed-form solution?" into a finite combinatorial check on a finite group. Once you know the Galois group, you know whether radicals suffice — and if they suffice, you know the chain of radicals to use.

A useful slogan: *"radical extensions are exactly the cyclic-step Galois extensions, after enough roots of unity have been adjoined."* The "enough roots of unity" caveat is real — over $\mathbb{Q}$, the extension generated by one $n$-th root is not Galois until you also throw in $\zeta_n$ — but it is a clean, separable adjustment, and the resulting picture is that "radical" means "tower of cyclic." Hence solvability of the Galois group is exactly what makes the radical tower possible.

The proof of solvability $\Leftrightarrow$ radicals also gives a recipe. If $G$ is solvable with derived series $G = G_0 \supseteq G_1 \supseteq \cdots \supseteq G_n = 1$, the corresponding tower of fixed fields gives you the chain of radicals: each $G_i / G_{i+1}$ is abelian, hence (after adjoining roots of unity) cyclic, and a cyclic extension is generated by a single radical (Hilbert 90). So solvability is not just a yes/no answer; it is a recipe for the formula.

---

## The Insolvability of the General Quintic

For the *general* polynomial of degree $n$ — the one with indeterminate coefficients, $f(x) = x^n + a_{n-1}x^{n-1} + \cdots + a_0$ over $K = \mathbb{Q}(a_0, \ldots, a_{n-1})$ — the Galois group is $S_n$. (This is essentially because no algebraic relations hold among the roots beyond what is forced by the symmetric functions.)

**Corollary.** The general polynomial of degree $n$ is solvable by radicals iff $S_n$ is solvable iff $n \leq 4$.

For $n = 2$: $S_2 = \mathbb{Z}/2$ is abelian. Quadratic formula.
For $n = 3$: $S_3$ has solvable chain $\{e\} \trianglelefteq A_3 \trianglelefteq S_3$. Cardano's formula.
For $n = 4$: $S_4$ has solvable chain $\{e\} \trianglelefteq V_4 \trianglelefteq A_4 \trianglelefteq S_4$. Ferrari's method.
For $n \geq 5$: $A_n$ is simple non-abelian (Galois proved this for $n = 5$; the general fact is due to Jordan), so $S_n$ is *not* solvable. No formula in radicals exists.

There is a beautiful conceptual reason the cutoff is $n = 4$: the symmetric group $S_n$ acts on a 4-element set canonically when $n = 4$ (namely, on the four cosets of $V_4$ in $S_4$), giving a homomorphism $S_4 \to S_4/V_4 \cong S_3$. This "outer" reduction does not exist for $n \geq 5$ because the alternating group $A_n$ for $n \geq 5$ is simple, blocking any quotient. The classical degree-3-and-4 formulas exploit this reduction explicitly: to solve a quartic, you reduce it to a "resolvent cubic," whose roots correspond to the three pairs of conjugate roots; you solve the cubic by Cardano, and lift back. The whole apparatus depends on the existence of the resolvent, which depends on the existence of a normal subgroup of index 3 in $A_4$, which depends on $V_4$ being normal in $S_4$ — and this last fact is exactly what fails when $n \geq 5$.

![A_5 is simple and non-abelian — the obstruction to solving the quintic](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/08-galois-theory/aa_v2_08_5_quintic.png)

The simplicity of $A_5$ is the obstruction. It is remarkable that the impossibility of a 19th-century calculator's dream — a quintic formula — comes down to the absence of a normal subgroup in a 60-element group. That is the kind of leverage Galois theory provides.

**A specific quintic.** $f(x) = x^5 - 4x + 2 \in \mathbb{Q}[x]$ has Galois group $S_5$ (irreducible by Eisenstein at $p=2$, three real roots and two complex by analyzing $f'$, hence the Galois group contains a transposition and a 5-cycle, which generate $S_5$). So this specific polynomial is not solvable by radicals — its roots cannot be expressed using $+, -, \times, \div, \sqrt[n]{\cdot}$.

The argument that "transposition + 5-cycle generates $S_5$" is a small group-theoretic fact worth knowing: any 5-cycle and any transposition in $S_5$ generate the whole symmetric group, because the 5-cycle has order 5 (prime), so its powers fill out a subgroup that, combined with one transposition, lifts to $S_5$ via standard Cayley arguments. Concretely: given $(1\ 2\ 3\ 4\ 5)$ and any transposition, you can manufacture all transpositions $(i\ i+1)$ by conjugating, and adjacent transpositions generate $S_n$ for any $n$.

The "three real, two complex" part is where calculus sneaks back in. $f'(x) = 5x^4 - 4$ has real roots $\pm(4/5)^{1/4}$, so $f$ has exactly three real critical-point-bracketed roots, leaving two complex conjugate roots. Complex conjugation is a transposition of those two roots, fixing the three real ones. The Galois group of $f$ contains complex conjugation, which is a transposition.

However: many specific quintics *are* solvable by radicals. $x^5 - 1$ is solvable (Galois group $(\mathbb{Z}/5)^\times \cong \mathbb{Z}/4$, abelian). $x^5 - 2$ is solvable (Galois group of order 20, isomorphic to $\mathbb{Z}/5 \rtimes \mathbb{Z}/4$, which contains $\mathbb{Z}/5$ as a normal subgroup with cyclic quotient). The unsolvable quintics are the "generic" ones whose Galois group is the full $S_5$.

A delightful subplot: the *icosahedral* extensions over $\mathbb{Q}$ — those with Galois group $A_5$ — are not solvable by radicals, but Hermite (1858) showed they can be solved using elliptic *modular* functions. So if you allow yourself slightly more exotic functions than $n$-th roots, the quintic is back on the table. This is the spiritual ancestor of the modern theory of "solvability by special functions" and shows how the Galois-theoretic obstruction shifts when you change the toolkit.

**Why this matters.** This is one of the great theorems of mathematics. A question — "is there a formula for the roots of a polynomial?" — that occupied mathematicians for centuries, that was answered for degrees 2, 3, 4 by ingenious case analysis, is settled in full generality by an abstract structural argument about groups. The bridge from "find a formula" to "the alternating group is simple" is exactly the Galois correspondence.

It also illustrates the broader pattern of 19th-century algebra: hard concrete problems get reduced to abstract structural ones, and the abstraction lets you handle infinitely many cases at once. Galois did not solve the quintic; he proved that no one ever will, and explained why. That is a different kind of mathematical achievement, and it set the template for much of what came after.

---

## What's Next

Galois theory completes our tour of fields. We have seen how the structure of polynomials, fields, and groups intertwine to give one of the deepest results in classical algebra. In the next article, we leave fields and groups behind to introduce a new perspective: *modules*, the natural generalization of vector spaces to arbitrary rings. Modules unify abelian groups, vector spaces, and ideals into a single framework — and they are the first step toward the homological methods that dominate modern algebra.

Before moving on, three takeaways worth keeping in your back pocket:

1. *The Galois group sees everything algebraic.* Two elements of a Galois extension are conjugate over $K$ iff some Galois-group element swaps them. Every algebraic property that survives field automorphisms can be read off the group action.
2. *Normal subgroups encode normal sub-extensions.* The terminology was chosen exactly so that the FTGT statement is uniform on both sides. When you see "normal" in either context, the other context's "normal" is what makes it work.
3. *Solvability is about chains of cyclic steps.* The general philosophy — break a problem into iterated cyclic (or abelian) pieces — recurs throughout algebra: composition series for groups, refinements for representations, derived functors in homological algebra. Galois theory is the original example, and it is the one whose connection to a concrete classical question is most striking.

---

*This is Part 8 of the [Abstract Algebra](/en/series/abstract-algebra/) series (12 articles).*

*Previous: [Part 7 — Field Extensions](/en/abstract-algebra/07-field-extensions/)*

*Next: [Part 9 — Modules](/en/abstract-algebra/09-modules/)*
