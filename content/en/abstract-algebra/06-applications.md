---
title: "Applications — Coding Theory, Cryptography, and Physics"
date: 2021-02-14 09:00:00
tags:
  - abstract-algebra
  - applications
  - mathematics
categories: Mathematics
series: abstract-algebra
lang: en
mathjax: true
disableNunjucks: true
series_order: 6
series_total: 6
translationKey: "abstract-algebra-6"
description: "Abstract algebra is not abstract: it underlies error correction, RSA, elliptic curves, and gauge theory in physics."
---

## Algebra in the wild

Five articles of definitions, theorems, proof sketches. Now the payoff. Abstract algebra is not a self-contained intellectual game — it is the language behind some of the most consequential engineering and physics of the past century. Error correction that makes deep-space communication possible, cryptography that secures the internet, and the gauge symmetries that structure fundamental physics all rest on the algebraic machinery we have built in [Parts 1-5](/en/series/abstract-algebra/).

I remember being skeptical as an undergraduate: "When will I ever use the Fundamental Theorem of Galois Theory?" The answer, it turns out, is: not directly. But the habits of thought — quotient structures, group actions, working over finite fields — show up constantly in applied work. The specific theorems become invisible scaffolding.

## Error-correcting codes

Digital communication is noisy. Bits flip. You need redundancy to detect and correct errors. The algebra of finite fields (from [Part 4](/en/abstract-algebra/04-fields-and-extensions/)) gives you that redundancy in an optimal, systematic way — not by naive repetition, but by encoding messages as evaluations of polynomials.

A **linear code** $C$ over $\mathbb{F}_q$ is a subspace $C \leq \mathbb{F}_q^n$ of dimension $k$. It has parameters $[n, k, d]$ where $d$ is the minimum Hamming distance between distinct codewords. Such a code can detect up to $d - 1$ errors and correct up to $\lfloor (d-1)/2 \rfloor$ errors.

![Applications overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/abstract-algebra/06-applications/fig_concept.png)

### The Hamming code

The $(7,4)$ Hamming code over $\mathbb{F}_2$ encodes 4 bits into 7 bits and corrects any single-bit error ($[n=7, k=4, d=3]$). The parity check matrix:

$$
H = \begin{pmatrix} 1 & 0 & 1 & 0 & 1 & 0 & 1 \\ 0 & 1 & 1 & 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 1 & 1 & 1 & 1 \end{pmatrix}
$$

The columns of $H$ are all nonzero vectors in $\mathbb{F}_2^3$. If position $j$ is corrupted ($y = c + e_j$), the syndrome $s = Hy^T = He_j^T$ gives the binary representation of $j$. You locate and fix the error with a single matrix-vector multiply over $\mathbb{F}_2$. The elegance is purely algebraic: the code's structure comes from the vector space $\mathbb{F}_2^3$.

**Non-example (why linearity matters).** A "random" subset of $\mathbb{F}_2^7$ with $2^4 = 16$ elements might not have any decoding structure — you would need a lookup table of size 16 for every possible received word. Linearity gives you the syndrome: a homomorphism $\mathbb{F}_2^7 \to \mathbb{F}_2^3$ whose kernel is the code. The quotient structure from [Part 2](/en/abstract-algebra/02-homomorphisms-and-quotients/) is doing the work.

### Reed-Solomon codes

A Reed-Solomon code over $\mathbb{F}_q$ works as follows. Fix distinct evaluation points $\alpha_1, \ldots, \alpha_n \in \mathbb{F}_q$ (typically $n = q - 1$, using all nonzero elements) and a message length $k < n$. Encode a message $(m_0, \ldots, m_{k-1})$ as the polynomial $f(x) = m_0 + m_1 x + \cdots + m_{k-1}x^{k-1}$, then transmit the evaluations:

$$
(f(\alpha_1), f(\alpha_2), \ldots, f(\alpha_n))
$$

Since a degree-$(k-1)$ polynomial is uniquely determined by $k$ points, any $k$ correct evaluations suffice to recover $f$ (via Lagrange interpolation). A nonzero polynomial of degree $< k$ has at most $k - 1$ roots, so any two distinct codewords differ in at least $n - (k-1) = n - k + 1$ positions. The minimum distance $d = n - k + 1$ achieves the **Singleton bound** — Reed-Solomon codes are MDS (maximum distance separable).

**Worked example.** Over $\mathbb{F}_7$ with $n = 6$ evaluation points $\{1, 2, 3, 4, 5, 6\}$ and $k = 3$. Encode message $(2, 1, 3)$ as $f(x) = 2 + x + 3x^2$. Evaluations: $f(1) = 6$, $f(2) = 6 \cdot 2 = 16 \equiv 2$, $f(3) = 2 + 3 + 27 = 32 \equiv 4$, $f(4) = 2 + 4 + 48 = 54 \equiv 5$, $f(5) = 2 + 5 + 75 = 82 \equiv 5$, $f(6) = 2 + 6 + 108 = 116 \equiv 4 \pmod{7}$. Transmit $(6, 2, 4, 5, 5, 4)$. This code corrects up to $\lfloor(6-3)/2\rfloor = 1$ symbol error.

**Decoding.** Suppose we receive $(6, 2, 4, 5, 0, 4)$ — position 5 corrupted. The Berlekamp-Massey algorithm computes an "error locator polynomial" $\Lambda(x)$ whose roots point to error positions. In $O(n^2)$ operations over $\mathbb{F}_q$, the original message is recovered. All of this is polynomial arithmetic over a finite field — the ring theory of [Part 3](/en/abstract-algebra/03-rings-and-ideals/) in action.

**Real-world use:** CDs use RS codes over $\mathbb{F}_{2^8}$ (the field with 256 elements). A scratched CD still plays because RS codes can correct burst errors spanning hundreds of bits. QR codes, deep-space communication (Voyager, Mars rovers), and RAID storage all use Reed-Solomon as their error-correction backbone.

## Public-key cryptography: RSA

RSA relies on the multiplicative group $(\mathbb{Z}/n\mathbb{Z})^*$ where $n = pq$ is a product of two large primes. The key asymmetry: computing $\varphi(n) = (p-1)(q-1)$ requires knowing $p$ and $q$, but factoring $n$ into $pq$ is (believed to be) computationally hard.

**Setup:** Choose large primes $p, q$ (2048 bits each). Let $n = pq$. Pick $e$ with $\gcd(e, \varphi(n)) = 1$ (commonly $e = 65537$). Compute $d \equiv e^{-1} \pmod{\varphi(n)}$ using the extended Euclidean algorithm. Public key: $(n, e)$. Private key: $d$.

**Encryption and decryption:**

$$
\text{Encrypt: } c \equiv m^e \pmod{n}, \qquad \text{Decrypt: } m \equiv c^d \pmod{n}
$$

**Correctness proof.** We have $ed \equiv 1 \pmod{\varphi(n)}$, so $ed = 1 + k\varphi(n)$ for some integer $k$. Then:

$$
c^d = (m^e)^d = m^{ed} = m^{1 + k\varphi(n)} = m \cdot (m^{\varphi(n)})^k
$$

By Euler's theorem (itself a consequence of Lagrange's theorem from [Part 1](/en/abstract-algebra/01-groups-and-subgroups/) applied to the group $(\mathbb{Z}/n\mathbb{Z})^*$ of order $\varphi(n)$): $m^{\varphi(n)} \equiv 1 \pmod{n}$ when $\gcd(m, n) = 1$. So $c^d \equiv m \pmod{n}$. $\square$

**Worked example (toy size).** Let $p = 61$, $q = 53$. Then $n = 3233$ and $\varphi(n) = 60 \times 52 = 3120$. Choose $e = 17$ ($\gcd(17, 3120) = 1$). Compute $d$: we need $17d \equiv 1 \pmod{3120}$. Extended Euclidean: $d = 2753$ (check: $17 \times 2753 = 46801 = 15 \times 3120 + 1$).

Encrypt $m = 65$: $c = 65^{17} \bmod 3233 = 2790$.
Decrypt: $m = 2790^{2753} \bmod 3233 = 65$. (Use fast modular exponentiation.)

**The CRT speedup.** Since the holder of the private key knows $p$ and $q$, they can compute $c^d \bmod p$ and $c^d \bmod q$ separately (with smaller exponents $d_p = d \bmod (p-1)$ and $d_q = d \bmod (q-1)$), then combine via the Chinese Remainder Theorem. This is roughly 4x faster than working mod $n$ directly. The CRT from [Part 3](/en/abstract-algebra/03-rings-and-ideals/) is not just a theoretical curiosity — it is a standard optimization in every RSA implementation.

## Elliptic curve cryptography

An **elliptic curve** over a field $F$ (with $\text{char}(F) \neq 2, 3$) is defined by:

$$
E: y^2 = x^3 + ax + b, \quad \Delta = -16(4a^3 + 27b^2) \neq 0
$$

together with a point at infinity $\mathcal{O}$. The remarkable fact: these points form an **abelian group** under a geometrically-defined addition law. Given $P, Q \in E$: draw the line through $P$ and $Q$, find the third intersection with the curve, and reflect across the $x$-axis. The point at infinity serves as the identity element.

The explicit formulas: if $P = (x_1, y_1)$ and $Q = (x_2, y_2)$ with $P \neq \pm Q$:

$$
\lambda = \frac{y_2 - y_1}{x_2 - x_1}, \quad x_3 = \lambda^2 - x_1 - x_2, \quad y_3 = \lambda(x_1 - x_3) - y_1
$$

Over a finite field $\mathbb{F}_p$, the group $E(\mathbb{F}_p)$ has order roughly $p$ (Hasse's theorem: $||E(\mathbb{F}_p)| - p - 1| \leq 2\sqrt{p}$).

**The ECDLP.** Given points $P$ and $Q = nP$ (where $nP$ means adding $P$ to itself $n$ times), find the integer $n$. The best known algorithms for ECDLP are fully exponential ($O(\sqrt{p})$ via Pollard's rho), unlike the sub-exponential index calculus for $(\mathbb{Z}/p\mathbb{Z})^*$. This means 256-bit ECC provides security comparable to 3072-bit RSA — explaining why TLS 1.3, Bitcoin (secp256k1), and Signal all use elliptic curves.

## Symmetry in physics

### Gauge symmetry and Lie groups

The Standard Model of particle physics is built on the gauge group:

$$
G = SU(3) \times SU(2) \times U(1)
$$

Each factor governs a fundamental interaction: $SU(3)$ for the strong force (quantum chromodynamics), $SU(2) \times U(1)$ for the electroweak force. These are **Lie groups** — groups that are simultaneously smooth manifolds — with the algebraic structure (Lie algebras, root systems, representation theory) providing the classification.

The principle of **gauge invariance** states: the laws of physics must be unchanged under local transformations by the gauge group. This single requirement *determines* the form of all fundamental interactions. The gauge fields (gluons, $W^\pm/Z$ bosons, photons) are connections on principal $G$-bundles, and their dynamics follow from demanding the Lagrangian be gauge-invariant.

This is abstract algebra at its most powerful: the group does not merely describe the symmetry — it *dictates* which physical theories are possible.

### Representation theory and particle classification

A **representation** of a group $G$ on a vector space $V$ is a homomorphism $\rho: G \to GL(V)$ — the same concept of homomorphism from [Part 2](/en/abstract-algebra/02-homomorphisms-and-quotients/), now applied to classify particles.

Quarks transform in the **fundamental representation** of $SU(3)$ (dimension 3 — hence "three colors": red, green, blue). Antiquarks transform in the conjugate representation $\bar{3}$. Gluons transform in the adjoint representation (dimension 8 — hence eight gluon types). Leptons are $SU(3)$-singlets (dimension 1 — don't feel the strong force). The Higgs field transforms as a doublet under $SU(2)$.

The entire zoo of elementary particles is organized by irreducible representations of the gauge group. Group theory tells you which particles *must* exist and constrains their interactions. The prediction of the $\Omega^-$ baryon (1964) from $SU(3)$ representation theory, confirmed experimentally within two years, was one of the great triumphs of algebraic methods in physics.

## Algebraic geometry (a glimpse)

The polynomial ring $k[x_1, \ldots, x_n]$ and its ideals from [Part 3](/en/abstract-algebra/03-rings-and-ideals/) parametrize geometric objects. The **variety** $V(I) = \{p \in k^n : f(p) = 0 \;\forall f \in I\}$ is the zero set of an ideal — the geometric shadow of algebraic data. Hilbert's Nullstellensatz provides the bridge:

$$
I(V(J)) = \sqrt{J} \quad \text{(the radical of } J\text{)}
$$

This establishes a dictionary: prime ideals correspond to irreducible varieties, maximal ideals to points, and quotient rings $k[x_1, \ldots, x_n]/I$ to coordinate rings of varieties. Modern algebraic geometry (Grothendieck's scheme theory) generalizes this to arbitrary commutative rings, unifying number theory and geometry under one conceptual framework. The proof of Fermat's Last Theorem (Wiles, 1995) relied on this unification — specifically, on the theory of elliptic curves over number fields and their associated Galois representations.

## Where to go from here

This series covered groups, homomorphisms, rings, fields, Galois theory, and applications. For further study:

- **Representation theory:** Serre's *Linear Representations of Finite Groups* — compact, complete, and demanding. Extends the group action ideas from Part 1 to vector spaces.
- **Commutative algebra:** Atiyah-Macdonald for the algebraic foundations of ring/ideal theory, then Hartshorne or Vakil for algebraic geometry.
- **Number theory:** Galois theory over $\mathbb{Q}$ leads to algebraic number theory and class field theory (Neukirch, *Algebraic Number Theory*). The absolute Galois group $\text{Gal}(\overline{\mathbb{Q}}/\mathbb{Q})$ from Part 5 is the central object.
- **Category theory:** the language that reveals the structural patterns shared across all branches of algebra. Mac Lane, *Categories for the Working Mathematician*.
- **Computational algebra:** Sage, Magma, or GAP for computing Galois groups, factoring polynomials over finite fields, and exploring group structure experimentally. Nothing builds intuition like computing the Sylow subgroups of $S_6$ by hand and then watching a computer do it in milliseconds.

Abstract algebra started as a theory of polynomial equations in the early 19th century. It became the structural backbone of modern mathematics — and of much of theoretical computer science, signal processing, and fundamental physics. Every structure you encounter — topological spaces, vector bundles, type systems in programming languages — is algebraic underneath, whether or not the word "group" appears in its description.

---

*This is Part 6 of [Abstract Algebra](/en/series/abstract-algebra/) (6 parts).
Previous: [Part 5 — Galois Theory](/en/abstract-algebra/05-galois-theory/)*
