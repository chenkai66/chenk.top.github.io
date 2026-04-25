---
title: "Machine Learning Mathematical Derivations (15): Hidden Markov Models"
date: 2026-02-03 09:00:00
categories:
  - Machine Learning
tags:
  - Hidden Markov Model
  - HMM
  - Viterbi Algorithm
  - Forward Algorithm
  - Baum-Welch Algorithm
  - Mathematical Derivation
  - Machine Learning
series:
  name: "ML Mathematical Derivations"
  order: 15
  total: 20
lang: en
mathjax: true
description: "Derive the three classical HMM algorithms from one principle (factorising the joint, then sharing sub-computations across time): Forward-Backward for evaluation and smoothing, Viterbi for MAP decoding, and Baum-Welch (EM) for parameter learning."
disableNunjucks: true
series_order: 15
---

You hear footsteps behind you in a fog. You cannot see the walker, only the sounds. From the rhythm and pitch -- short, soft, hurried -- can you guess whether they are walking, running, or limping? And if you observed an entire sequence, which gait sequence is most likely? How likely is *any* sequence of sounds under your model of how walking works?

These are the **three problems of HMMs**, and the surprise is that all three reduce to one trick: write the joint $P(\mathbf{O}, \mathbf{I})$ as a product of local factors along time, then **share sub-computations across time** with dynamic programming. Brute force costs $O(N^T)$. Forward-Backward, Viterbi, and Baum-Welch all cost $O(N^2 T)$. The exponent collapses because the Markov assumption makes the future conditionally independent of the past given the present.

## What You Will Learn

- The HMM joint distribution and its **two conditional-independence assumptions**
- **Forward / Backward**: $P(\mathbf{O}\mid\lambda)$ and posterior smoothing $\gamma_t(i), \xi_t(i,j)$
- **Viterbi**: MAP decoding via DP, with the single change of `sum` -> `max`
- **Baum-Welch**: EM specialised to HMMs; why the M-step formulae are *expected counts*
- Numerical pitfalls (underflow, scaling) and modern descendants (CRF, RNN, CTC)

## Prerequisites

- Markov chains and stochastic matrices
- Dynamic programming / memoisation
- The EM algorithm and the ELBO ([Part 13](/en/Machine-Learning-Mathematical-Derivations-13-EM-Algorithm-and-GMM/))

---

## 1. The Model

![HMM as a graphical model: a Markov chain over hidden states emits independent observations](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/15-Hidden-Markov-Models/fig1_graphical_model.png)

An HMM is the simplest non-trivial latent-variable model for sequences. Two parallel chains run in time:

- a **hidden** state chain $\mathbf{I}=(i_1,\dots,i_T)$ with $i_t\in\{1,\dots,N\}$, and
- an **observed** emission chain $\mathbf{O}=(o_1,\dots,o_T)$ with $o_t\in\{v_1,\dots,v_M\}$.

The model is fully described by three parameter blocks $\lambda=(\boldsymbol{\pi}, \mathbf{A}, \mathbf{B})$:

| Block | Symbol | Meaning |
| --- | --- | --- |
| Initial distribution | $\pi_i = P(i_1 = i)$ | how the chain is born |
| Transition matrix | $a_{ij} = P(i_{t+1}=j \mid i_t=i)$ | how state evolves |
| Emission matrix | $b_j(k) = P(o_t = v_k \mid i_t = j)$ | how state radiates evidence |

**Two conditional-independence assumptions** make everything tractable:

1. **First-order Markov on states** -- $P(i_{t+1}\mid i_{1:t}) = P(i_{t+1}\mid i_t)$.
2. **Observation independence** -- $o_t$ depends only on $i_t$, not on other states or observations.

These two together let us factorise the full joint along time:

$$P(\mathbf{O}, \mathbf{I} \mid \lambda) = \pi_{i_1}\,b_{i_1}(o_1) \prod_{t=2}^{T} a_{i_{t-1},i_t}\,b_{i_t}(o_t).$$

Every algorithm in this article is a clever way to **sum or maximise** this product without enumerating the $N^T$ hidden paths.

### The 3-State Weather Toy

![A three-state Markov chain with transition probabilities (rows of A sum to 1)](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/15-Hidden-Markov-Models/fig2_transition_diagram.png)

Throughout the article we lean on a 3-state weather example -- *Sunny, Rainy, Cloudy* -- with the transitions shown above. Sunny is sticky ($a_{\text{SS}}=0.70$), and Cloudy is the hub that connects everything else. This is small enough to inspect by hand and rich enough to expose every algorithmic subtlety.

### Three Problems, One Joint

Given $\lambda$, three questions exhaust what we can ask:

| # | Problem | Input | Output | Algorithm |
| --- | --- | --- | --- | --- |
| 1 | **Evaluation** | $\lambda, \mathbf{O}$ | $P(\mathbf{O}\mid\lambda)$ | Forward / Backward |
| 2 | **Decoding** | $\lambda, \mathbf{O}$ | $\arg\max_{\mathbf{I}} P(\mathbf{I}\mid\mathbf{O},\lambda)$ | Viterbi |
| 3 | **Learning** | $\mathbf{O}$ (and $N$) | $\hat\lambda = \arg\max_\lambda P(\mathbf{O}\mid\lambda)$ | Baum-Welch (EM) |

A naive approach to (1) sums the joint over all $N^T$ hidden sequences -- already $\approx 10^{170}$ for $N{=}50, T{=}100$. The next three sections make all three problems polynomial.

---

## 2. Forward Algorithm: Evaluation by Left-to-Right Sweep

![Forward algorithm trellis: alpha values flow left-to-right; each node sums incoming paths and multiplies by the local emission probability](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/15-Hidden-Markov-Models/fig3_forward_trellis.png)

Define the **forward variable**

$$\alpha_t(i) = P(o_1, o_2, \dots, o_t,\; i_t = i \mid \lambda).$$

It is the joint probability of having generated the observations seen *so far* and ending up in state $i$ at time $t$.

**Initialisation.** At $t=1$ there is no transition yet, only the initial distribution and the first emission:

$$\alpha_1(i) = \pi_i\, b_i(o_1).$$

**Recursion.** To extend $\alpha$ from $t-1$ to $t$, condition on the previous state and marginalise:

$$\begin{aligned}
\alpha_t(j) &= P(o_1,\dots,o_t,\, i_t=j) \\
&= \sum_{i=1}^N P(o_1,\dots,o_{t-1},\, i_{t-1}=i)\,P(i_t=j\mid i_{t-1}=i)\,P(o_t\mid i_t=j)\\
&= \left[\sum_{i=1}^N \alpha_{t-1}(i)\, a_{ij}\right] b_j(o_t).
\end{aligned}$$

The bracketed sum is the **only** computation that crosses the time boundary; everything else is local. This is dynamic programming in the cleanest possible form.

**Termination.** Sum out the final hidden state:

$$P(\mathbf{O}\mid\lambda) = \sum_{i=1}^N \alpha_T(i).$$

**Cost.** Each of $T$ steps has $N$ targets, each requires summing over $N$ predecessors: $O(N^2 T)$. For $N{=}50, T{=}100$ this is $2.5\times 10^{5}$ operations -- about $10^{165}\times$ faster than the brute-force sum.

**Underflow.** Probabilities multiply geometrically, so $\alpha_t$ eventually underflows. Two standard fixes:

- **Log-space with `logsumexp`**: $\log\alpha_t(j) = \mathrm{lse}_i\!\big(\log\alpha_{t-1}(i) + \log a_{ij}\big) + \log b_j(o_t)$.
- **Scaled forward**: at each $t$ divide $\alpha_t$ by $c_t = \sum_j \alpha_t(j)$ and accumulate $\log P(\mathbf{O}) = \sum_t \log c_t$.

---

## 3. Backward Algorithm and Posterior Smoothing

The forward sweep alone evaluates $P(\mathbf{O})$. To compute the **posterior over hidden states** at every $t$, we also need a right-to-left sweep.

![Backward algorithm trellis: beta flows right-to-left from the boundary beta_T(i) = 1](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/15-Hidden-Markov-Models/fig4_backward_trellis.png)

Define the **backward variable**

$$\beta_t(i) = P(o_{t+1}, o_{t+2}, \dots, o_T \mid i_t = i, \lambda).$$

Note the conditioning: $\beta$ is a *conditional* probability of the *future*, while $\alpha$ is a *joint* probability of the *past*. They mirror each other.

**Boundary.** Past time $T$ there are no observations, so $\beta_T(i) = 1$ for all $i$.

**Recursion.** Step from $t+1$ back to $t$ by summing over the next state:

$$\beta_t(i) = \sum_{j=1}^N a_{ij}\, b_j(o_{t+1})\, \beta_{t+1}(j).$$

**Sanity check.** Combining both sweeps must recover the marginal:

$$P(\mathbf{O}\mid\lambda) = \sum_i \pi_i\,b_i(o_1)\,\beta_1(i) = \sum_i \alpha_T(i).$$

### Two Posteriors That Drive Learning

Once both $\alpha$ and $\beta$ are tabulated we can read off, in $O(1)$ extra work per cell:

**State posterior (smoothing):**

$$\gamma_t(i) = P(i_t=i \mid \mathbf{O},\lambda) = \frac{\alpha_t(i)\,\beta_t(i)}{P(\mathbf{O}\mid\lambda)}.$$

**Pairwise posterior (transition responsibility):**

$$\xi_t(i,j) = P(i_t=i, i_{t+1}=j \mid \mathbf{O},\lambda) = \frac{\alpha_t(i)\,a_{ij}\,b_j(o_{t+1})\,\beta_{t+1}(j)}{P(\mathbf{O}\mid\lambda)}.$$

These are the two statistics Baum-Welch needs in its E-step.

---

## 4. Viterbi: From Sum to Max

![Viterbi trellis: the max-product DP highlights the single most-likely state path through time](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/15-Hidden-Markov-Models/fig5_viterbi_path.png)

Decoding asks for the **most likely state path** rather than the total probability:

$$\mathbf{I}^* = \arg\max_{\mathbf{I}}\, P(\mathbf{I}\mid\mathbf{O},\lambda) = \arg\max_{\mathbf{I}}\, P(\mathbf{O},\mathbf{I}\mid\lambda).$$

(The denominator $P(\mathbf{O})$ does not depend on $\mathbf{I}$, so it can be dropped.)

Define

$$\delta_t(j) = \max_{i_1,\dots,i_{t-1}} P(i_1,\dots,i_{t-1}, i_t=j,\, o_1,\dots,o_t\mid \lambda),$$

the probability of the **best** path of length $t$ ending in state $j$. The recursion mirrors Forward, with one operator swapped:

$$\boxed{\;\delta_t(j) = \max_{i}\big[\delta_{t-1}(i)\, a_{ij}\big]\, b_j(o_t).\;}$$

To recover the path itself, store back-pointers

$$\psi_t(j) = \arg\max_i\big[\delta_{t-1}(i)\, a_{ij}\big].$$

After the forward pass, terminate at $i_T^* = \arg\max_i \delta_T(i)$ and **backtrack**: $i_t^* = \psi_{t+1}(i_{t+1}^*)$.

**Why does swapping `sum` for `max` work?** Both operators distribute over the time-factorised joint -- "max-times" forms a commutative semiring, just like "sum-times". The dynamic-programming bookkeeping is identical; only the local reduction changes.

**Numerical form.** In practice always run Viterbi in log-space:

$$\log\delta_t(j) = \max_i\big[\log\delta_{t-1}(i) + \log a_{ij}\big] + \log b_j(o_t).$$

Now everything is an addition, so underflow is impossible.

---

## 5. Baum-Welch: EM for HMMs

When $\lambda$ is unknown, learn it by maximising $\log P(\mathbf{O}\mid\lambda)$. The hidden states $\mathbf{I}$ are latent, so apply EM. The result -- the **Baum-Welch algorithm** -- predates the general EM framework by several years (Baum & Petrie, 1966) and is one of the prettiest examples of EM in the wild.

![Baum-Welch monotonically improves the log-likelihood; recovered transition matrix tracks the truth](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/15-Hidden-Markov-Models/fig6_baum_welch.png)

### E-step: Posterior Statistics

Given the current $\lambda^{(k)}$, run **forward-backward** to obtain $\gamma_t(i)$ and $\xi_t(i,j)$. These are the *expected* values, under the current model, of the indicator variables $\mathbb{1}[i_t=i]$ and $\mathbb{1}[i_t=i, i_{t+1}=j]$.

The complete-data log-likelihood factorises as

$$\log P(\mathbf{O},\mathbf{I}\mid\lambda) = \log\pi_{i_1} + \sum_{t=1}^{T-1}\log a_{i_t i_{t+1}} + \sum_{t=1}^{T}\log b_{i_t}(o_t).$$

Taking the expectation under $P(\mathbf{I}\mid \mathbf{O},\lambda^{(k)})$ replaces each $\mathbb{1}[\cdot]$ by its posterior probability:

$$Q(\lambda;\lambda^{(k)}) = \sum_i \gamma_1(i)\log\pi_i + \sum_{t=1}^{T-1}\sum_{i,j}\xi_t(i,j)\log a_{ij} + \sum_{t=1}^{T}\sum_{j,k}\gamma_t(j)\,\mathbb{1}[o_t=v_k]\log b_j(k).$$

The three terms are **decoupled** in $\boldsymbol{\pi}, \mathbf{A}, \mathbf{B}$ -- the M-step solves three independent constrained maximisations.

### M-step: Three Closed-Form Updates

Each block has a normalisation constraint (rows sum to 1). With Lagrange multipliers, every update reduces to

$$\text{parameter} = \frac{\text{expected count of the event}}{\text{expected count of its conditioning context}}.$$

Concretely:

$$\boxed{\;
\hat\pi_i = \gamma_1(i),\qquad
\hat a_{ij} = \frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\gamma_t(i)},\qquad
\hat b_j(k) = \frac{\sum_{t:\,o_t=v_k}\gamma_t(j)}{\sum_{t=1}^{T}\gamma_t(j)}.
\;}$$

Read each ratio as expected-transitions-from-$i$-to-$j$ over expected-departures-from-$i$, and similarly for emissions. Identical in spirit to MLE on observed counts; only "observed" is replaced by "expected under the posterior".

### Convergence

EM guarantees $P(\mathbf{O}\mid\lambda^{(k+1)}) \geq P(\mathbf{O}\mid\lambda^{(k)})$ -- the curve in the figure is provably monotonic. The **catch**: the surface is multi-modal, so Baum-Welch finds a local maximum. Standard remedies are random restarts, k-means / segmental k-means initialisation, or informative priors.

---

## 6. Application: POS Tagging

![POS tagging: Viterbi picks the most-likely tag sequence; emission heatmap shows per-tag word probabilities](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/ml-math-derivations/15-Hidden-Markov-Models/fig7_pos_tagging.png)

Part-of-speech (POS) tagging is the textbook HMM application. Hidden states are tags (PRON, VERB, ADJ, NOUN, ...) and observations are words. Transitions encode grammar (a determiner is usually followed by an adjective or noun); emissions encode the lexicon (the word *love* is most often a verb, sometimes a noun).

For "I love natural language processing" the Viterbi path lights up as **PRON / VERB / ADJ / NOUN / NOUN** -- correct, despite *processing* being lexically ambiguous, because the ADJ -> NOUN transition is highly probable.

The same engine appears in **speech recognition** (states = phonemes, observations = MFCC frames), **bioinformatics** profile HMMs (states = match/insert/delete columns), and **gesture recognition** -- whenever a discrete latent process generates a noisy observable stream.

---

## Q&A

### Forward vs. Viterbi -- why does swapping operators matter?

Forward returns the marginal $P(\mathbf{O}\mid\lambda) = \sum_{\mathbf{I}} P(\mathbf{O},\mathbf{I})$; Viterbi returns $\max_{\mathbf{I}} P(\mathbf{O},\mathbf{I})$. Same DP skeleton, different semiring (sum-product vs. max-product). Forward answers "*how plausible is this evidence?*"; Viterbi answers "*what story best explains it?*"

### Why does Viterbi maximise the joint instead of the posterior?

Because $P(\mathbf{I}\mid\mathbf{O}) = P(\mathbf{O},\mathbf{I})/P(\mathbf{O})$ and the denominator is a constant in $\mathbf{I}$. Maximising the joint is therefore equivalent and avoids one normalisation.

### When does Baum-Welch fail?

Three classic failure modes: (a) bad initialisation -- it lands in a flat or trivial local optimum; (b) **label switching** -- states are only identified up to permutation; (c) **observation collapse** -- a state's emission concentrates on observed symbols only, leaving zero probability for unseen ones. Smooth with a small Dirichlet prior to fix (c).

### Why CRFs over HMMs for sequence labelling?

CRFs are *discriminative*: they model $P(\mathbf{I}\mid\mathbf{O})$ directly and can exploit overlapping global features of $\mathbf{O}$ (capitalisation, suffix templates, surrounding words) without the conditional-independence straitjacket. HMMs are still preferred when you need to *generate* sequences or when training data is scarce.

### Are HMMs obsolete in the deep-learning era?

As stand-alone end-to-end models, mostly yes -- RNN/Transformer encoders dominate. But the *inference algorithms* live on. CTC decoding in modern speech systems is essentially Forward over an alignment lattice; sequence-level distillation uses Viterbi; structured-output Transformers borrow from CRFs which borrow from HMMs.

### How do I choose $N$?

Start small, then use information criteria ($\text{AIC} = -2\log L + 2|\lambda|$, $\text{BIC} = -2\log L + |\lambda|\log T$) or held-out likelihood. Bayesian non-parametrics (the iHMM with a hierarchical Dirichlet process prior) place a prior over $N$ and let the data decide.

### How do I handle continuous observations?

Replace the emission matrix with a density: a single Gaussian, a Gaussian mixture (the classical GMM-HMM in speech), or a neural density (Mixture Density Network -> "neural HMM"). The forward-backward recursions are identical; only $b_j(o_t)$ becomes a likelihood evaluation.

---

## Exercises

**E1.** With $N=2$, $\boldsymbol{\pi}=(0.6, 0.4)$, $\mathbf{B} = \begin{pmatrix}0.5 & 0.5 \\ 0.4 & 0.6\end{pmatrix}$, and $o_1 = v_1$, compute $\alpha_1$.
<details><summary>Solution</summary>$\alpha_1(1) = 0.6 \cdot 0.5 = 0.30$, $\alpha_1(2) = 0.4 \cdot 0.4 = 0.16$.</details>

**E2.** For $N{=}50, T{=}100$, compare Forward to brute-force enumeration.
<details><summary>Solution</summary>Forward: $N^2 T = 2.5\times 10^{5}$ multiplications. Brute force: $N^T \approx 10^{170}$ paths -- intractable.</details>

**E3.** Interpret $\hat a_{ij} = \frac{\sum_t \xi_t(i,j)}{\sum_t \gamma_t(i)}$.
<details><summary>Solution</summary>Expected number of $i\to j$ transitions divided by expected number of departures from state $i$. Soft generalisation of MLE counts.</details>

**E4.** Show that running Forward then Backward gives the same $P(\mathbf{O}\mid\lambda)$ as Forward alone.
<details><summary>Solution</summary>$\sum_i \pi_i b_i(o_1)\beta_1(i) = \sum_i \alpha_1(i)\beta_1(i)/1 = \sum_i \alpha_t(i)\beta_t(i)$ for any $t$ (a consequence of the chain rule); at $t = T$, $\beta_T \equiv 1$ gives $\sum_i \alpha_T(i)$.</details>

**E5.** Explain why Viterbi requires only $O(N^2 T)$ time but $O(NT)$ memory for back-pointers.
<details><summary>Solution</summary>The same trellis as Forward (cost $O(N^2 T)$) plus an integer back-pointer per (cell, time) -- $NT$ slots -- consulted during the linear-time backtrack.</details>

---

## References

1. Rabiner, L. R. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proc. IEEE 77(2), 257-286.
2. Baum, L. E., & Petrie, T. (1966). *Statistical Inference for Probabilistic Functions of Finite State Markov Chains*. Annals of Math. Statist. 37(6).
3. Viterbi, A. J. (1967). *Error Bounds for Convolutional Codes and an Asymptotically Optimum Decoding Algorithm*. IEEE Trans. IT 13(2).
4. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Ch. 17. MIT Press.
5. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Ch. 13. Springer.
6. Jelinek, F. (1997). *Statistical Methods for Speech Recognition*. MIT Press.
7. Lafferty, J., McCallum, A., & Pereira, F. (2001). *Conditional Random Fields*. ICML.
8. Eddy, S. R. (1998). *Profile Hidden Markov Models*. Bioinformatics 14(9).

---

## Series Navigation

- Previous: [Part 14 -- Variational Inference](/en/Machine-Learning-Mathematical-Derivations-14-Variational-Inference-and-Variational-EM/)
- Next: [Part 16 -- Conditional Random Fields](/en/Machine-Learning-Mathematical-Derivations-16-Conditional-Random-Fields/)
- [View all 20 parts in this series](/tags/Machine-Learning/)
