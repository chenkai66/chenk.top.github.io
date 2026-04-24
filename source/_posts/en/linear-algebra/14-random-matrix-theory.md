---
title: "Essence of Linear Algebra (14): Random Matrix Theory"
date: 2025-03-08 09:00:00
tags:
  - Linear Algebra
  - random matrices
  - Wigner semicircle law
  - Marchenko-Pastur distribution
categories:
  - Linear Algebra
series:
  name: "Linear Algebra"
  part: 14
  total: 18
lang: en
mathjax: true
description: "Fill a huge matrix with random numbers, compute its eigenvalues, and watch stunning regularity emerge from chaos. Learn the Wigner semicircle law, Marchenko-Pastur distribution, and Tracy-Widom limit -- with applications to MIMO, finance, and PCA in ML."
disableNunjucks: true
series_order: 14
---

A million i.i.d. coin flips, arranged into a thousand-by-thousand symmetric matrix, somehow produce eigenvalues that fill a perfect semicircle. A noisy sample covariance matrix that should be the identity instead spreads its eigenvalues across an interval whose width you can predict before seeing a single number. The largest eigenvalue of a Wigner matrix has a tail distribution that turns up everywhere -- in growing crystals, in the longest increasing subsequence of a random permutation, in the energy levels of heavy nuclei. **Random matrix theory** (RMT) is the study of why these regularities appear, and how to use them.

> **What you will learn**
> - Wigner and Wishart matrices: the two model families that drive almost everything
> - The semicircle law and the Marchenko-Pastur law as "central limit theorems" for spectra
> - Eigenvalue repulsion, the Wigner surmise, and the Tracy-Widom edge
> - A working understanding of free probability and the Stieltjes transform
> - Applications: MIMO capacity, covariance cleaning, PCA thresholding, the spiked-covariance / BBP phase transition
>
> **Prerequisites:** eigendecomposition (Chapter 6), SVD (Chapter 9), basic probability (mean, variance, i.i.d., the classical CLT)

---

## 1. The Big Surprise: Why Random Matrices Are Not "Random"

Take a $1000 \times 1000$ matrix, fill its entries with independent standard Gaussians, symmetrise it, and divide by $\sqrt{n}$. Compute the eigenvalues, draw a histogram. The histogram does not depend on the random seed -- it always looks like the same semicircle on $[-2, 2]$. Repeat with uniform $\{-1, +1\}$ entries instead of Gaussians: same semicircle. Replace symmetric Gaussians with complex Hermitian Gaussians: still the same semicircle.

This is not magic; it is the same phenomenon that makes the classical central limit theorem work. When you average over a million weakly correlated random variables (and an eigenvalue is a complicated average over the matrix entries), the microscopic distribution gets washed away and only a few coarse statistics -- mean, variance, symmetry -- determine the limit. The semicircle law is the **central limit theorem of spectra**.

What changes in matrix-land is the limiting object: instead of a single Gaussian on the line, we get a whole **density of eigenvalues**, plus precise statements about the **gaps** between them and the **fluctuations** at the edge. RMT is the calculus of all three.

![Wigner semicircle law: empirical density of normalised GOE eigenvalues vs the theoretical semicircle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/14-random-matrix-theory/fig1_wigner_semicircle.png)

---

## 2. The Two Model Families

### 2.1 Wigner matrices: symmetric noise

A **Wigner matrix** $\mathbf{W} \in \mathbb{R}^{n\times n}$ has

- diagonal entries $w_{ii}$ i.i.d. with mean $0$ and finite variance $\sigma_d^2$,
- upper-triangular entries $w_{ij}$ ($i < j$) i.i.d. with mean $0$ and variance $\sigma^2$,
- $w_{ji} = w_{ij}$ enforced by symmetry.

The **Gaussian Orthogonal Ensemble** (GOE) is the special case where all entries are Gaussian. The "orthogonal" name comes from the fact that the distribution is invariant under conjugation by any orthogonal matrix $\mathbf{O}\mathbf{W}\mathbf{O}^\top$. For complex Hermitian entries you get the **GUE** (unitary invariance); for quaternionic entries the **GSE** (symplectic). The trio GOE / GUE / GSE corresponds to whether time-reversal symmetry is present, broken by a magnetic field, or broken with half-integer spin -- which is why the same matrices keep appearing in physics.

**Mental picture.** Imagine an Erdős-Rényi social graph where edge weight $w_{ij}$ is a random "closeness" score. The eigenvalues of that closeness matrix tell you about its global community structure -- and for purely random closenesses, the structure is the universal one we are about to study.

### 2.2 Wishart matrices: sample covariance

Let $\mathbf{X} \in \mathbb{R}^{n \times p}$ have i.i.d. entries with mean $0$ and variance $1$. The **Wishart matrix** (or sample covariance matrix) is

$$
\mathbf{S} \;=\; \frac{1}{n}\,\mathbf{X}^\top \mathbf{X} \;\in\; \mathbb{R}^{p\times p}.
$$

If $n \gg p$, $\mathbf{S}$ is a good estimator of the true covariance. If $n$ and $p$ are *both* large with their ratio fixed, $\mathbf{S}$ is wildly off -- in a structured, predictable way that the Marchenko-Pastur law describes.

**Mental picture.** Track the daily returns of $p = 500$ stocks for one trading year ($n \approx 252$). The "covariance matrix" you compute has $125{,}000$ free parameters but only $\approx 126{,}000$ data points; the resulting estimate is essentially a random matrix and its eigenvalues are spread out by RMT laws even when the true covariance is the identity.

---

## 3. The Wigner Semicircle Law

### 3.1 Statement

Let $\mathbf{W}$ be an $n \times n$ Wigner matrix with off-diagonal variance $\sigma^2$. Form the normalised matrix

$$
\hat{\mathbf{W}} \;=\; \frac{\mathbf{W}}{\sigma\sqrt{n}}.
$$

As $n \to \infty$, the **empirical spectral distribution** $\frac{1}{n}\sum_i \delta_{\lambda_i(\hat{\mathbf{W}})}$ converges almost surely (in the weak sense) to the **semicircle density**

$$
f(x) \;=\; \frac{1}{2\pi}\sqrt{4 - x^2}, \qquad x \in [-2, 2],
$$

and $f(x) = 0$ outside $[-2, 2]$.

### 3.2 Three ways to see why it has to be a semicircle

**1) Method of moments (the rigorous route).** Compute $m_k = \mathbb{E}[\frac{1}{n}\operatorname{tr}\hat{\mathbf{W}}^k]$. Each term in the trace is a closed walk on $n$ vertices using $k$ steps, weighted by the corresponding product of Gaussian moments. Independence and zero mean force the only surviving walks to be **pair-matched non-crossing walks** -- exactly the structure counted by Catalan numbers $C_{k/2}$. The Catalan numbers are precisely the moments of the semicircle, so the limits match.

**2) Coulomb gas (the physicist's route).** The joint density of GOE eigenvalues is
$$
\rho(\lambda_1, \dots, \lambda_n) \;\propto\; \prod_{i<j} |\lambda_i - \lambda_j|\;\exp\!\Big(-\tfrac{n}{4}\sum_i \lambda_i^2\Big),
$$
which describes $n$ charged particles on a line with logarithmic repulsion (the Vandermonde factor) confined by a harmonic potential. The equilibrium density that balances repulsion against confinement is the semicircle.

**3) Free CLT (the algebraic route).** A symmetric random matrix can be written as a sum of many "free" rank-one perturbations. In free probability the analogue of "sum of independent variables" is the **free additive convolution**, and its central limit theorem yields the semicircle distribution -- not the Gaussian. Section 7 expands on this.

### 3.3 Why the underlying distribution does not matter

The semicircle law is **universal**: change the entry distribution from Gaussian to uniform, to $\pm 1$, to anything with mean zero and finite variance, and the limit is the same. This mirrors the classical CLT, where the limit of the normalised sum is Gaussian regardless of the summands' distribution.

The hidden mechanism is identical too: in both cases, only the second moment survives the limit; everything else gets averaged away by the scaling.

### 3.4 Code: verify it yourself

```python
import numpy as np
import matplotlib.pyplot as plt

n, repeats = 1500, 30
all_eigs = []
for _ in range(repeats):
    a = np.random.randn(n, n)
    w = (a + a.T) / np.sqrt(2 * n)        # variance scaling: std semicircle
    all_eigs.append(np.linalg.eigvalsh(w))
all_eigs = np.concatenate(all_eigs)

x = np.linspace(-2, 2, 600)
plt.hist(all_eigs, bins=90, density=True, alpha=0.55, label="Empirical")
plt.plot(x, np.sqrt(np.maximum(4 - x**2, 0)) / (2 * np.pi),
         lw=2.5, label="Semicircle")
plt.legend(); plt.show()
```

The fit is essentially perfect already at $n = 200$.

---

## 4. The Marchenko-Pastur Law

### 4.1 Statement

Let $\mathbf{X}$ be $n \times p$ with i.i.d. entries of mean $0$ and variance $1$, and set $\gamma = p/n$ (the **aspect ratio**). As $n, p \to \infty$ with $\gamma$ fixed, the empirical spectral distribution of $\mathbf{S} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$ converges to the **Marchenko-Pastur density**

$$
f(\lambda) \;=\; \frac{1}{2\pi\gamma\,\lambda}\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)},\qquad \lambda \in [\lambda_-, \lambda_+],
$$

with edges

$$
\lambda_\pm \;=\; (1 \pm \sqrt{\gamma})^2.
$$

If $\gamma > 1$ the matrix has rank $n < p$ and there are $p - n$ exact zero eigenvalues in addition to the bulk on $[\lambda_-, \lambda_+]$.

### 4.2 What the density tells you

Even when the population covariance is the identity, finite samples spread the eigenvalues out:

- $\gamma = 0.1$: edges $[0.47, 1.69]$ -- mild widening.
- $\gamma = 0.5$: edges $[0.09, 2.91]$ -- the largest sample eigenvalue is **three times** what it should be.
- $\gamma = 1.0$: edges $[0, 4]$ -- the spectrum touches zero; the matrix is on the verge of singularity.

This single density underwrites every quantitative use of RMT in statistics: any sample eigenvalue that lives strictly above $\lambda_+$ is *significantly* large; anything inside $[\lambda_-, \lambda_+]$ is consistent with pure noise.

![Marchenko-Pastur law for four aspect ratios. As gamma grows, the bulk widens; at gamma=1 the support reaches zero](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/14-random-matrix-theory/fig2_marchenko_pastur.png)

### 4.3 Code

```python
import numpy as np, matplotlib.pyplot as plt

n, p, repeats = 1000, 500, 30
gamma = p / n
all_eigs = []
for _ in range(repeats):
    x = np.random.randn(n, p)
    all_eigs.append(np.linalg.eigvalsh(x.T @ x / n))
all_eigs = np.concatenate(all_eigs)

lam_minus, lam_plus = (1 - np.sqrt(gamma))**2, (1 + np.sqrt(gamma))**2
xs = np.linspace(lam_minus + 1e-3, lam_plus - 1e-3, 500)
mp = np.sqrt((lam_plus - xs) * (xs - lam_minus)) / (2 * np.pi * gamma * xs)

plt.hist(all_eigs, bins=100, density=True, alpha=0.5, label="Empirical")
plt.plot(xs, mp, "r", lw=2, label="MP theory")
for e in (lam_minus, lam_plus):
    plt.axvline(e, ls="--", color="g")
plt.legend(); plt.show()
```

---

## 5. The Fine Structure: Repulsion and the Edge

The semicircle and MP densities describe the *bulk* -- the macroscopic shape. RMT also gives precise answers on two finer scales.

### 5.1 Spacings: eigenvalues repel

Fix a generic point in the bulk and look at the distribution of the gap to the nearest neighbour, normalised by the local mean spacing. For GOE matrices the gap distribution is closely approximated by the **Wigner surmise**

$$
p(s) \;=\; \frac{\pi s}{2}\,\exp\!\Big(-\frac{\pi s^2}{4}\Big).
$$

The crucial feature is $p(0) = 0$: eigenvalues do not coincide. They actively *repel*, with quadratic vanishing at $s=0$ for GOE (and cubic for GUE -- different symmetry classes have different repulsion exponents).

Compare this to **independent** levels: their spacings would be exponential, $p(s) = e^{-s}$, with **maximum** at $s = 0$. The clustering you would expect from independence is exactly what eigenvalues refuse to do.

![Eigenvalue spacings vs Poisson: the GOE histogram vanishes at zero gap; the Poisson curve peaks there](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/14-random-matrix-theory/fig3_eigenvalue_repulsion.png)

### 5.2 The edge: Tracy-Widom

The largest eigenvalue $\lambda_{\max}$ of an $n \times n$ GOE sits near $2$ for large $n$, but its fluctuations are tiny -- of order $n^{-2/3}$ rather than the $n^{-1/2}$ you would naively guess. Specifically,

$$
n^{2/3}\big(\lambda_{\max} - 2\big) \;\xrightarrow{d}\; \mathrm{TW}_1,
$$

where $\mathrm{TW}_1$ is the **Tracy-Widom distribution** for $\beta = 1$. It is highly asymmetric: the left tail decays super-exponentially (you almost never see $\lambda_{\max}$ much smaller than $2$), the right tail decays like $\exp(-\frac{2}{3} t^{3/2})$ (large outliers do occur, but rarely). The same TW law governs the longest increasing subsequence of a random permutation, the height of growing crystals, and the largest singular value of large random matrices used in modern statistics.

### 5.3 Random vs deterministic spectra

A picture summarises everything: a random Wigner spectrum is a *smooth bulk* with strongly correlated points, while a deterministic spectrum is a *handful of clusters* with independent sampling noise inside each cluster.

![A random Wigner spectrum spreads smoothly over the semicircle; a deterministic diagonal matrix produces a few sharp clusters](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/14-random-matrix-theory/fig4_random_vs_diagonal.png)

---

## 6. Applications

### 6.1 Wireless: MIMO capacity scales linearly in antennas

A MIMO channel with $n_t$ transmit and $n_r$ receive antennas has capacity

$$
C \;=\; \sum_i \log_2\!\Big(1 + \frac{\mathrm{SNR}}{n_t}\,\lambda_i\Big),
$$

where $\lambda_i$ are eigenvalues of $\mathbf{H}\mathbf{H}^\dagger$ and $\mathbf{H}$ is the random channel matrix. The MP law tells you the limiting density of $\lambda_i$, and integrating against $\log_2(1 + \mathrm{SNR}\cdot\lambda)$ gives the ergodic capacity. The conclusion -- that capacity scales **linearly** with $\min(n_t, n_r)$ -- is the theoretical reason every modern phone, base station and Wi-Fi router uses multiple antennas.

```python
import numpy as np

def mimo_capacity(n_r, n_t, snr_db, trials=500):
    snr = 10 ** (snr_db / 10)
    caps = []
    for _ in range(trials):
        H = (np.random.randn(n_r, n_t)
             + 1j * np.random.randn(n_r, n_t)) / np.sqrt(2)
        eigs = np.linalg.eigvalsh(H @ H.conj().T)
        caps.append(np.sum(np.log2(1 + snr / n_t * eigs)))
    return float(np.mean(caps))

print(f"4x4 @ 10 dB: {mimo_capacity(4, 4, 10):.2f} bits/s/Hz")
print(f"8x8 @ 10 dB: {mimo_capacity(8, 8, 10):.2f} bits/s/Hz")
```

### 6.2 Finance: cleaning sample covariance matrices

Track $p$ stocks over $n$ trading days; for any realistic $p, n$ the aspect ratio $\gamma = p/n$ is far from zero. The MP edges then tell you exactly which sample eigenvalues are "noise". A standard recipe:

1. Diagonalise: $\mathbf{S} = \mathbf{U}\,\mathrm{diag}(\lambda_1, \dots, \lambda_p)\,\mathbf{U}^\top$.
2. Estimate the noise variance $\sigma^2$ (e.g. by averaging eigenvalues inside the MP support).
3. Compute $\lambda_\pm = \sigma^2(1 \pm \sqrt{\gamma})^2$.
4. **Replace** all eigenvalues in $[\lambda_-, \lambda_+]$ by their mean (or by a more sophisticated shrinkage).
5. Reconstruct $\tilde{\mathbf{S}} = \mathbf{U}\,\mathrm{diag}(\tilde\lambda_i)\,\mathbf{U}^\top$.

Portfolios built on $\tilde{\mathbf{S}}$ typically improve out-of-sample Sharpe ratios by $10$-$30\%$ and cut turnover.

![Sample covariance, RMT-cleaned covariance, and the true identity covariance shown as heatmaps](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/14-random-matrix-theory/fig6_covariance_estimation.png)

```python
import numpy as np

def clean_covariance(returns):
    """Replace the MP bulk with its mean."""
    n, p = returns.shape
    gamma = p / n
    S = np.cov(returns, rowvar=False)
    eigs, vecs = np.linalg.eigh(S)
    sigma2 = np.mean(eigs)
    lo, hi = sigma2 * (1 - np.sqrt(gamma))**2, sigma2 * (1 + np.sqrt(gamma))**2
    bulk = (eigs >= lo) & (eigs <= hi)
    eigs[bulk] = eigs[bulk].mean()
    return vecs @ np.diag(eigs) @ vecs.T
```

### 6.3 PCA in high dimensions: how many components to keep?

In the PCA workflow, the question "how many principal components are signal?" usually has no good classical answer when $p \approx n$. RMT gives a sharp one: **count the eigenvalues that exceed $\lambda_+ = (1 + \sqrt{\gamma})^2$.** Anything below the MP edge is statistically indistinguishable from i.i.d. noise.

```python
import numpy as np

def pca_signal_count(X):
    n, p = X.shape
    gamma = p / n
    Xs = (X - X.mean(0)) / X.std(0)
    eigs = np.linalg.eigvalsh(Xs.T @ Xs / n)
    return int(np.sum(eigs > (1 + np.sqrt(gamma))**2))
```

### 6.4 Spiked covariance and the BBP phase transition

The PCA criterion above is too crude when the signal is small. A more honest model is the **spiked covariance**: the population covariance is $\Sigma = \mathbf{I} + \sum_{k=1}^{r} (s_k - 1) \mathbf{v}_k\mathbf{v}_k^\top$, i.e. identity plus $r$ "spikes". The **Baik-Ben Arous-Péché (BBP) transition** says:

- if a population spike $s_k > 1 + \sqrt{\gamma}$, the corresponding sample eigenvalue separates from the MP bulk and lands at $s_k + \gamma s_k/(s_k - 1)$, *and* the sample eigenvector aligns with $\mathbf{v}_k$ with positive cosine;
- if $s_k \le 1 + \sqrt{\gamma}$, the spike is **invisible**: it is buried inside the MP bulk and PCA cannot recover it, no matter how clever your algorithm.

This is a hard impossibility theorem for high-dim PCA. It tells you exactly when more samples are necessary and when a problem is statistically unsolvable at the current $n, p$.

![Spiked covariance: bulk follows MP, super-critical spikes appear as outliers shifted from their population value, sub-critical spikes vanish into the bulk](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/14-random-matrix-theory/fig7_spiked_covariance.png)

### 6.5 Neural network initialisation

Initialising weights so that $\mathbf{W}^\top\mathbf{W}$ has eigenvalues concentrated near $1$ keeps the variance of activations stable layer by layer. The Xavier/Glorot rule $\mathrm{Var}(w_{ij}) = 2 / (n_\text{in} + n_\text{out})$ is exactly the variance that puts the singular values of $\mathbf{W}$ on the MP support around $1$. Orthogonal initialisation goes further -- it eliminates singular-value spread completely and gives provably better signal propagation in deep linear networks.

---

## 7. Tools of the Trade

### 7.1 Stieltjes transform

Every probability measure $\mu$ on $\mathbb{R}$ has a **Stieltjes transform**

$$
m_\mu(z) \;=\; \int \frac{d\mu(\lambda)}{\lambda - z}, \qquad z \in \mathbb{C}^+.
$$

You recover the density via the inversion formula
$$
f(\lambda) \;=\; -\frac{1}{\pi}\lim_{\eta \to 0^+}\operatorname{Im}\,m_\mu(\lambda + i\eta).
$$
Why bother? Because the Stieltjes transform of an empirical spectral distribution equals $\frac{1}{n}\operatorname{tr}(\mathbf{M} - z\mathbf{I})^{-1}$, and resolvents are easy to manipulate algebraically. Most modern proofs in RMT happen entirely on the level of the Stieltjes transform: write down a self-consistent equation for $m(z)$, solve it, invert.

For the semicircle, the equation is $m(z)^2 + zm(z) + 1 = 0$, solving to $m(z) = (-z + \sqrt{z^2 - 4})/2$. For Marchenko-Pastur the equation is similarly small.

### 7.2 Free probability in one paragraph

In free probability, "non-commutative random variables" $a, b$ are **freely independent** if their alternating mixed moments factor in a particular tracial way. Voiculescu's theorem says **large independent random matrices are asymptotically free**. The free analogue of convolution -- *free additive convolution* $\boxplus$ -- takes the spectral distribution of $\mathbf{A}$ and that of $\mathbf{B}$ and returns the spectral distribution of $\mathbf{A} + \mathbf{B}$. The associated central limit theorem gives the semicircle distribution. Practical takeaway: anything you used to know how to do for sums of independent scalar variables (mean, variance, CLT, Berry-Esseen), there is a free analogue for spectra of sums of independent matrices.

![Free additive convolution: sum of two free semicircles is again a semicircle, with summed variance](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/linear-algebra/14-random-matrix-theory/fig5_free_convolution.png)

### 7.3 Sketch of the semicircle proof (method of moments)

Compute $m_k = \mathbb{E}[\frac{1}{n}\operatorname{tr}\hat{\mathbf{W}}^k]$. Expanding the trace gives a sum over closed length-$k$ walks on $\{1, \dots, n\}$, weighted by $\prod_e \mathbb{E}[w_e]$. Independence + zero mean kill any walk that uses an edge an odd number of times. For even $k$, the dominant surviving walks are *non-crossing pair-matched* walks -- counted by the Catalan number $C_{k/2}$ -- and they come with a weight of $1$ after normalisation. Catalan numbers are exactly the moments of the semicircle, so the limit must be the semicircle.

---

## 8. Exercises

### Basics

1. Write down $\mathbb{E}[\mathbf{W}]$ and $\mathrm{Cov}(w_{ij}, w_{kl})$ for a $3 \times 3$ GOE with diagonal variance $2$ and off-diagonal variance $1$.
2. Why is the $1/\sqrt{n}$ normalisation necessary? What happens to $\lambda_\max$ without it?
3. For $\gamma = 0.5$, compute $\lambda_\pm$ and sketch the MP density. Where is its maximum?

### Computation and proof

4. Show $\int_{-2}^{2}\frac{1}{2\pi}\sqrt{4 - x^2}\,dx = 1$ using $x = 2\sin\theta$.
5. Compute the second and fourth semicircle moments and check they equal the Catalan numbers $C_1$ and $C_2$.
6. For $\mathbf{X}$ with i.i.d. $\mathcal{N}(0, 1)$ entries, prove $\mathbb{E}[\mathbf{X}^\top\mathbf{X}] = n\mathbf{I}_p$.
7. Derive the joint density of the two eigenvalues of a $2 \times 2$ symmetric Gaussian matrix; identify the repulsion factor.

### Programming

8. Verify the Wigner surmise empirically: sample $200$ GOE matrices of size $n = 500$, compute *bulk* nearest-neighbour spacings, normalise by the local mean, and compare to $p(s)$ and to $e^{-s}$.
9. Plot the MP histogram for $\gamma \in \{0.1, 0.5, 1.0, 2.0\}$. For $\gamma > 1$, separately count zero eigenvalues.
10. Plot ergodic MIMO capacity vs SNR for $2\times 2, 4\times 4, 8\times 8, 16\times 16$. Confirm the "double the antennas, double the capacity" rule at high SNR.
11. Reproduce the BBP transition: fix $\gamma = 0.5$, sweep the spike strength $s$ from $1$ to $3$, and plot the largest sample eigenvalue against $s$. Mark the predicted critical point $s = 1 + \sqrt{\gamma}$.

### Applications

12. A quant tracks $100$ stocks for $200$ days. Compute $\gamma$, find $[\lambda_-, \lambda_+]$, and decide whether a sample eigenvalue of $3.5$ is signal or noise.
13. For an $8 \times 4$ MIMO channel with complex Gaussian entries, write the capacity expression, estimate it at $30$ dB, and predict the gain from upgrading to $16 \times 8$.
14. Given $1000$ samples and $500$ features, compute the MP threshold and explain how to use it to choose the number of PCA components.

### Advanced

15. Look up the Tracy-Widom density and its role in the Roy largest-root test for high-dimensional MANOVA. Why is the classical chi-squared approximation wrong when $p/n$ is not small?
16. State and explain the BBP phase transition. Why is no estimator able to detect a sub-critical spike?
17. Why do energy-level spacings of complex nuclei follow GOE statistics? What is the role of time-reversal symmetry in the choice of GOE vs GUE?

---

## 9. Chapter Summary

**Models:** Wigner matrices model symmetric noise; Wishart matrices model sample covariance. The Gaussian variants -- GOE, GUE, GSE -- correspond to different symmetry classes.

**Bulk laws:** the **semicircle** for Wigner spectra and the **Marchenko-Pastur** density for Wishart spectra. Both are universal: the entry distribution does not matter as long as it has finite variance.

**Fine structure:** eigenvalues **repel** with a known law (Wigner surmise); the largest eigenvalue fluctuates on scale $n^{-2/3}$ according to the **Tracy-Widom** distribution.

**Toolkit:** the Stieltjes transform turns spectral problems into algebraic equations; **free probability** lifts classical CLT-style reasoning to spectra of independent matrices.

**Why it matters:** RMT gives sharp, often parameter-free predictions for MIMO capacity, covariance cleaning, PCA thresholding, the BBP phase transition for spiked models, and neural-network initialisation. In every case, the lesson is the same: high-dimensional randomness has a hidden, deterministic skeleton, and you can use it.

---

## References

- Bai, Z., & Silverstein, J. W. *Spectral Analysis of Large Dimensional Random Matrices.* Springer, 2010.
- Anderson, G. W., Guionnet, A., & Zeitouni, O. *An Introduction to Random Matrices.* Cambridge University Press, 2010.
- Mehta, M. L. *Random Matrices.* Academic Press, 2004.
- Tao, T. *Topics in Random Matrix Theory.* AMS, 2012.
- Tulino, A. M., & Verdú, S. *Random Matrix Theory and Wireless Communications.* Foundations and Trends, 2004.
- Bouchaud, J.-P., & Potters, M. *Financial Applications of Random Matrix Theory.* arXiv:0910.1205, 2009.
- Couillet, R., & Debbah, M. *Random Matrix Methods for Wireless Communications.* Cambridge University Press, 2011.
- Baik, J., Ben Arous, G., & Péché, S. "Phase transition of the largest eigenvalue for nonnull complex sample covariance matrices." *Annals of Probability*, 2005.

---

## Series Navigation

- **Previous:** [Chapter 13: Tensors and Multilinear Algebra](/en/chapter-13-tensors-and-multilinear-algebra/)
- **Next:** [Chapter 15: Linear Algebra in Machine Learning](/en/chapter-15-linear-algebra-in-machine-learning/)
- **Full Series:** Essence of Linear Algebra (1--18)
