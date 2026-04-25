---
title: "Recommendation Systems (2): Collaborative Filtering and Matrix Factorization"
date: 2025-12-04 09:00:00
tags:
  - Recommendation Systems
  - Collaborative Filtering
  - Matrix Factorization
categories: Recommendation Systems
lang: en
mathjax: true
series: recommendation-systems
series_title: "Recommendation Systems Series"
series_order: 2
description: "An in-depth tour of collaborative filtering and matrix factorization: User-CF and Item-CF, similarity metrics, latent-factor models, SVD++, ALS, BPR, and factorization machines — with intuitions, derivations, and runnable Python."
disableNunjucks: true
---

You finish *The Shawshank Redemption* and want something with the same feeling. A genre filter would surface every prison drama ever made, most of them awful. Collaborative filtering takes a different route: it never looks at the movie itself. It looks at *people who watched what you watched* and asks what else they loved.

That single idea — let the crowd's behaviour speak — powers Amazon, YouTube, Spotify and every modern feed. This article unpacks the algorithms behind it, from the neighbourhood methods of the 1990s to the matrix-factorization models that won the Netflix Prize.

> **What you will learn:**
> - The two flavours of neighbourhood CF (User-CF and Item-CF) and when each one wins
> - How to measure "similar taste" with cosine, Pearson, and adjusted cosine
> - Why matrix factorization beats neighbourhood methods on sparse data
> - SGD vs. ALS — same model, very different training stories
> - BPR for ranking, FM for side features, and weighted MF for implicit feedback
> - Working Python for every algorithm, plus a final Q&A on the tricky bits

---

## 1 · The core idea of collaborative filtering

Collaborative filtering (CF) rests on one assumption:

> **Users with similar histories want similar things; items liked by the same people are similar to each other.**

That is enough. CF needs no genre tags, no actor lists, no demographics — only a log of who interacted with what. Two paradigms fall straight out of the assumption:

- **User-Based CF (User-CF).** Find users whose past ratings look like yours, then recommend what *they* liked.
- **Item-Based CF (Item-CF).** Find items whose rating patterns look like the ones you already enjoyed, then recommend those.

![Two views of collaborative filtering: User-CF connects users with shared taste; Item-CF connects items with shared audiences](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/02-collaborative-filtering/fig1_user_user_vs_item_item.png)

### A toy rating matrix

Five users, five movies, a 1–5 scale, missing entries marked `?`:

| User  | Shawshank | Forrest | Pursuit | Titanic | Godfather |
|-------|:---------:|:-------:|:-------:|:-------:|:---------:|
| Alice | 5 | 5 | ? | 3 | 4 |
| Bob   | 5 | 5 | 4 | 2 | ? |
| Carol | 4 | 4 | 3 | 5 | 5 |
| David | 2 | 1 | ? | 4 | 3 |
| Eve   | ? | ? | 2 | 3 | 4 |

**User-CF** notices Alice and Bob agree on Shawshank and Forrest. Bob enjoyed *Pursuit* (4), so we recommend *Pursuit* to Alice.

**Item-CF** notices that Shawshank and Forrest collect almost identical ratings across users. If you liked one, you should like the other.

### Strengths and limits

CF wins on **simplicity** (no feature engineering) and **serendipity** (it surfaces non-obvious links — the famous "diapers and beer" pattern). It also recommends across categories: a book reader can be pointed at a podcast.

But it pays for that with three classic pains:

- **Cold start.** A brand-new user or item has no history.
- **Sparsity.** Real rating matrices are 99 %+ empty.
- **Popularity bias.** Hits crowd out the long tail.

Matrix factorization, which we meet later, attacks the second problem head-on.

---

## 2 · User-Based CF

### The recipe

1. Pick a similarity metric and compute it for every pair of users.
2. For each prediction, take the $k$ most similar users to the target — its **neighbourhood**.
3. Predict the missing rating as a similarity-weighted average of what the neighbours gave.
4. Sort, slice, recommend.

### Measuring "similar taste"

Let $I_{uv}$ be the items both users $u$ and $v$ have rated. Three metrics dominate the literature.

**Cosine similarity** treats each user as a vector and measures the angle between them:

$$
\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui}\, r_{vi}}{\sqrt{\sum_{i \in I_{uv}} r_{ui}^2}\;\sqrt{\sum_{i \in I_{uv}} r_{vi}^2}}
$$

It only cares about *direction*. If User A rates everything 4–5 and User B rates everything 1–2, but in the same shape, cosine still says they agree.

**Pearson correlation** subtracts each user's mean rating first, so "I always rate high" can no longer fake agreement:

$$
\text{sim}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{ui} - \bar{r}_u)^2}\;\sqrt{\sum_{i \in I_{uv}} (r_{vi} - \bar{r}_v)^2}}
$$

**Adjusted cosine** does the same de-meaning trick; mathematically it is Pearson under another name when the same items are aligned.

The picture below makes the difference visceral. User A is a generous rater; User B is strict. Their shapes are identical, only the level differs.

![Two users with the same shape but different levels: cosine says 0.99, Pearson says 1.00 — but cosine confuses level with taste](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/02-collaborative-filtering/fig3_cosine_vs_pearson.png)

Cosine returns ~0.99 because the vectors point almost the same way. Pearson returns exactly 1.00 — once we subtract each user's mean, the two centred vectors are identical. In real systems where rating habits vary, Pearson is the safer default.

Running Pearson over the toy matrix gives the heatmap below. Alice and Bob are tightly correlated; David is the contrarian.

![Pearson correlation between every pair of users in the toy rating matrix](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/02-collaborative-filtering/fig2_user_similarity_heatmap.png)

### Predicting a rating

Once we have the $k$ nearest neighbours $N_k(u)$ of user $u$:

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N_k(u)} \text{sim}(u, v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N_k(u)} |\text{sim}(u, v)|}
$$

Read it as: *"start at my average rating, then nudge up or down based on how my neighbours felt about this item, weighted by how much I trust each neighbour."* The $r_{vi} - \bar{r}_v$ term is what each neighbour does *relative to their own baseline* — exactly the same de-meaning idea as Pearson.

### A working implementation

```python
import numpy as np
from scipy.stats import pearsonr


class UserBasedCF:
    """Neighbourhood collaborative filtering with Pearson / cosine / Euclidean similarity."""

    def __init__(self, k: int = 20, similarity: str = "pearson"):
        self.k = k
        self.similarity = similarity

    def fit(self, ratings: dict[str, dict[str, float]]) -> "UserBasedCF":
        self.ratings = ratings
        self.user_mean = {u: np.mean(list(r.values())) for u, r in ratings.items()}
        self.users = list(ratings)
        self.user_idx = {u: i for i, u in enumerate(self.users)}
        n = len(self.users)
        self.sim = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                s = self._pair_sim(self.users[i], self.users[j])
                self.sim[i, j] = self.sim[j, i] = s
        return self

    def _pair_sim(self, u: str, v: str) -> float:
        common = set(self.ratings[u]) & set(self.ratings[v])
        if len(common) < 2:
            return 0.0
        ru = np.array([self.ratings[u][i] for i in common])
        rv = np.array([self.ratings[v][i] for i in common])
        if self.similarity == "pearson":
            r, _ = pearsonr(ru, rv)
            return 0.0 if np.isnan(r) else float(r)
        if self.similarity == "cosine":
            return float(ru @ rv / (np.linalg.norm(ru) * np.linalg.norm(rv)))
        # Euclidean → similarity in (0, 1]
        return 1.0 / (1.0 + np.linalg.norm(ru - rv))

    def predict(self, user: str, item: str) -> float:
        if user not in self.ratings:
            return 3.0
        idx = self.user_idx[user]
        neighbours = [
            (v, self.sim[idx, self.user_idx[v]], self.ratings[v][item])
            for v in self.ratings
            if v != user and item in self.ratings[v] and self.sim[idx, self.user_idx[v]] > 0
        ]
        if not neighbours:
            return self.user_mean[user]
        neighbours.sort(key=lambda t: t[1], reverse=True)
        neighbours = neighbours[: self.k]
        num = sum(s * (r - self.user_mean[v]) for v, s, r in neighbours)
        den = sum(abs(s) for _, s, _ in neighbours)
        return self.user_mean[user] + num / den if den else self.user_mean[user]

    def recommend(self, user: str, n: int = 10) -> list[tuple[str, float]]:
        seen = set(self.ratings.get(user, {}))
        candidates = {i for r in self.ratings.values() for i in r} - seen
        scored = [(i, self.predict(user, i)) for i in candidates]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:n]


if __name__ == "__main__":
    ratings = {
        "Alice": {"Shawshank": 5, "Forrest": 5, "Titanic": 3, "Godfather": 4},
        "Bob":   {"Shawshank": 5, "Forrest": 5, "Pursuit": 4, "Titanic": 2},
        "Carol": {"Shawshank": 4, "Forrest": 4, "Pursuit": 3, "Titanic": 5, "Godfather": 5},
        "David": {"Shawshank": 2, "Forrest": 1, "Titanic": 4, "Godfather": 3},
        "Eve":   {"Pursuit": 2, "Titanic": 3, "Godfather": 4},
    }
    model = UserBasedCF(k=2).fit(ratings)
    print(f"Alice's predicted rating for Pursuit: {model.predict('Alice', 'Pursuit'):.2f}")
    for item, score in model.recommend("Alice", n=3):
        print(f"  {item}: {score:.2f}")
```

### Where User-CF struggles

User-CF is intuitive and explainable, but it scales as $O(m^2 \cdot n)$ in the offline step — quadratic in the user count. It also struggles to find common ratings on sparse data, and is hit hardest by user cold start. The next section flips the problem on its head.

---

## 3 · Item-Based CF

### Why Item-CF wins in production

Amazon's 2003 paper made Item-CF the industry default, for three boring but decisive reasons:

- **There are usually fewer items than users**, so the similarity matrix is smaller and cheaper.
- **Item similarity is stable.** A movie's audience profile barely changes overnight; a user's mood does. We can precompute item similarities once and cache them.
- **Recommendations are explainable.** *"Because you liked A, here is similar B"* is a sentence users actually understand.

### Adjusted cosine: the metric of choice

For items $i$ and $j$ rated by the shared user set $U_{ij}$:

$$
\text{sim}(i, j) = \frac{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_{ij}} (r_{ui} - \bar{r}_u)^2}\;\sqrt{\sum_{u \in U_{ij}} (r_{uj} - \bar{r}_u)^2}}
$$

Critical detail: we subtract the **user** mean, not the item mean. That removes the noise of "who tends to rate high"; what is left is whether two items provoke the same reaction in the same person.

### Predicting

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{j \in N_k(i)} \text{sim}(i, j) \cdot (r_{uj} - \bar{r}_u)}{\sum_{j \in N_k(i)} |\text{sim}(i, j)|}
$$

The $k$ nearest neighbours of *item* $i$ now do the voting.

### Implementation

```python
import numpy as np
from collections import defaultdict


class ItemBasedCF:
    """Item-based CF with adjusted-cosine similarity."""

    def __init__(self, k: int = 20):
        self.k = k

    def fit(self, ratings: dict[str, dict[str, float]]) -> "ItemBasedCF":
        self.ratings = ratings
        self.user_mean = {u: np.mean(list(r.values())) for u, r in ratings.items()}
        self.item_users: dict[str, dict[str, float]] = defaultdict(dict)
        for u, items in ratings.items():
            for i, r in items.items():
                self.item_users[i][u] = r
        self.sim: dict[str, dict[str, float]] = defaultdict(dict)
        items = list(self.item_users)
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                s = self._adjusted_cosine(items[a], items[b])
                if s > 0:
                    self.sim[items[a]][items[b]] = s
                    self.sim[items[b]][items[a]] = s
        return self

    def _adjusted_cosine(self, i: str, j: str) -> float:
        common = set(self.item_users[i]) & set(self.item_users[j])
        if not common:
            return 0.0
        num = den_i = den_j = 0.0
        for u in common:
            di = self.item_users[i][u] - self.user_mean[u]
            dj = self.item_users[j][u] - self.user_mean[u]
            num += di * dj
            den_i += di * di
            den_j += dj * dj
        return num / np.sqrt(den_i * den_j) if den_i and den_j else 0.0

    def predict(self, user: str, item: str) -> float:
        if user not in self.ratings or item not in self.sim:
            return self.user_mean.get(user, 3.0)
        seen = self.ratings[user]
        rated_neighbours = [(j, s) for j, s in self.sim[item].items() if j in seen]
        if not rated_neighbours:
            return self.user_mean[user]
        rated_neighbours.sort(key=lambda t: t[1], reverse=True)
        rated_neighbours = rated_neighbours[: self.k]
        mean = self.user_mean[user]
        num = sum(s * (seen[j] - mean) for j, s in rated_neighbours)
        den = sum(abs(s) for _, s in rated_neighbours)
        return mean + num / den if den else mean
```

Item-CF is the workhorse, but on million-item catalogues even the precomputed similarity matrix becomes a memory hog. That motivates a fundamentally different approach: stop comparing rows and columns directly, and instead *compress* them.

---

## 4 · Matrix factorization

### From neighbourhoods to latent factors

The rating matrix $R \in \mathbb{R}^{m \times n}$ is huge and almost empty. Yet the *signal* is low-rank: a handful of underlying "taste dimensions" explain most of the variance. Matrix factorization captures that intuition explicitly.

We approximate

$$
R \approx P \cdot Q^{\!T}
$$

where $P \in \mathbb{R}^{m \times k}$ is a row per user and $Q \in \mathbb{R}^{n \times k}$ is a row per item. The dimension $k$ — typically 20–200 — is the number of latent factors. The predicted rating is just an inner product:

$$
\hat{r}_{ui} = \mathbf{p}_u \cdot \mathbf{q}_i = \sum_{f=1}^{k} p_{uf}\, q_{if}
$$

![A sparse rating matrix R approximated by the product of two thin matrices P and Qᵀ; the question marks are the missing entries we want to fill](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/02-collaborative-filtering/fig4_matrix_factorization.png)

### Why this is so much better than neighbourhoods

Two reasons:

1. **Storage shrinks from $O(mn)$ to $O((m+n)k)$.** A million users × a million items × 8 bytes is 8 TB of dense matrix, or about 0.8 GB once factorized at $k=50$.
2. **Missing entries become first-class citizens.** Neighbourhood methods need shared ratings to compare two rows; factorization happily predicts every cell from learned factors.

### Geometric reading

Think of the $k$ axes as discovered taste dimensions — *gritty drama vs. light comedy*, *modern blockbuster vs. arthouse classic*, and so on. The model learns these axes from data; nobody hand-labels them.

![Users (●) and items (▲) embedded in a 2-D latent space; Alice and Shawshank land in the same corner so their dot product is high](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/02-collaborative-filtering/fig6_latent_space.png)

A user vector $\mathbf{p}_u$ records *how much you care* about each axis; an item vector $\mathbf{q}_i$ records *how much it has* of each. Their dot product is the match score.

### The objective

We fit $P$ and $Q$ to minimise squared error on observed ratings, with L2 regularization to stop the factors blowing up:

$$
\mathcal{L} = \sum_{(u,i) \in \mathcal{R}} (r_{ui} - \mathbf{p}_u \cdot \mathbf{q}_i)^2 + \lambda \big(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2\big)
$$

Two algorithms dominate the optimisation: SGD and ALS.

### Optimisation 1 — Stochastic gradient descent

For one observed rating $(u, i)$ define the error $e_{ui} = r_{ui} - \hat{r}_{ui}$, then take a small step against the gradient:

$$
\mathbf{p}_u \leftarrow \mathbf{p}_u + \alpha\, (e_{ui}\, \mathbf{q}_i - \lambda\, \mathbf{p}_u)
$$

$$
\mathbf{q}_i \leftarrow \mathbf{q}_i + \alpha\, (e_{ui}\, \mathbf{p}_u - \lambda\, \mathbf{q}_i)
$$

SGD is dead simple, easy to make online, and tiny in memory. Its weakness is that each update touches one cell at a time, so it converges slowly.

```python
import numpy as np
import random


class MatrixFactorization:
    """Plain matrix factorization trained with stochastic gradient descent."""

    def __init__(self, k=50, lr=0.01, reg=0.01, epochs=20, seed=0):
        self.k, self.lr, self.reg, self.epochs = k, lr, reg, epochs
        self.rng = np.random.default_rng(seed)

    def fit(self, ratings: dict[str, dict[str, float]]):
        users = list(ratings)
        items = list({i for r in ratings.values() for i in r})
        self.u_idx = {u: i for i, u in enumerate(users)}
        self.i_idx = {i: j for j, i in enumerate(items)}
        self.items = items
        self.P = self.rng.normal(0, 0.1, (len(users), self.k))
        self.Q = self.rng.normal(0, 0.1, (len(items), self.k))

        samples = [(self.u_idx[u], self.i_idx[i], r)
                   for u, row in ratings.items() for i, r in row.items()]

        for epoch in range(1, self.epochs + 1):
            random.shuffle(samples)
            sse = 0.0
            for u, i, r in samples:
                err = r - self.P[u] @ self.Q[i]
                p, q = self.P[u].copy(), self.Q[i].copy()
                self.P[u] += self.lr * (err * q - self.reg * p)
                self.Q[i] += self.lr * (err * p - self.reg * q)
                sse += err * err
            if epoch % 5 == 0:
                print(f"epoch {epoch}: RMSE={np.sqrt(sse / len(samples)):.4f}")
        return self

    def predict(self, user: str, item: str) -> float:
        if user not in self.u_idx or item not in self.i_idx:
            return 0.0
        return float(self.P[self.u_idx[user]] @ self.Q[self.i_idx[item]])
```

### Optimisation 2 — Alternating least squares

ALS exploits a beautiful fact: **fix one of $P$ or $Q$ and the loss becomes a regularized linear regression with a closed-form solution.** So we alternate.

Fixing $Q$, the optimal user vector is

$$
\mathbf{p}_u = (Q_u^{\!T} Q_u + \lambda I)^{-1} Q_u^{\!T} \mathbf{r}_u
$$

where $Q_u$ stacks the item vectors that user $u$ rated. By symmetry, fixing $P$:

$$
\mathbf{q}_i = (P_i^{\!T} P_i + \lambda I)^{-1} P_i^{\!T} \mathbf{r}_i
$$

Each half-step is a small linear solve, and crucially **the per-user updates are independent of each other** — perfect for distributed engines like Spark MLlib.

### SGD vs. ALS in practice

The chart compares both on a synthetic 80×60 matrix. ALS plummets in the first few epochs because each step is a closed-form solve; SGD takes longer to wander down to the same neighbourhood.

![Training RMSE over 40 epochs: ALS drops in a handful of iterations, SGD takes longer but ends up close](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/02-collaborative-filtering/fig5_sgd_vs_als_convergence.png)

That headline ("ALS is faster") needs a footnote. ALS pays in memory, because each update assembles a $k \times k$ Gram matrix per user/item; SGD walks the data one observation at a time. Rough rules of thumb:

- **Small data, online updates needed:** SGD.
- **Large batch, distributed cluster:** ALS.
- **Best of both worlds:** start with ALS to bootstrap, refine with SGD for online updates.

---

## 5 · Bias terms — the cheapest accuracy boost you'll ever get

Plain factorization assumes ratings come purely from the user–item interaction. Reality is messier:

- Some users rate everything 4–5 (generous), others 1–2 (strict).
- Some items are universally adored or panned.
- The whole platform has an average rating level (~3.5 on a 1–5 scale is typical).

Adding bias terms costs almost nothing and usually drops RMSE by 10–20 %:

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u \cdot \mathbf{q}_i
$$

where $\mu$ is the global mean, $b_u$ is the user offset, and $b_i$ is the item offset. The factor vectors are then free to model only the *interaction* — the part of the rating that is genuinely about taste.

Initialise the biases from data, then learn them jointly with the factors:

$$
b_u^{(0)} = \frac{1}{|I_u|}\sum_{i \in I_u}(r_{ui} - \mu), \qquad
b_i^{(0)} = \frac{1}{|U_i|}\sum_{u \in U_i}(r_{ui} - \mu - b_u^{(0)})
$$

The SGD update gains two extra lines:

```python
self.b_u[u] += self.lr * (err - self.reg_b * self.b_u[u])
self.b_i[i] += self.lr * (err - self.reg_b * self.b_i[i])
```

That is the "SVD" of the Netflix Prize era — not the textbook singular value decomposition, but biased matrix factorization marketed under a familiar name.

---

## 6 · SVD++ — folding in implicit feedback

Most user behaviour is *implicit*: clicks, scrolls, dwell time, purchases without rating. SVD++ uses these silent signals to build a richer user vector.

Let $I_u$ be the items user $u$ has *touched in any way* (rated or not). For each such item we learn an additional vector $\mathbf{y}_j$. The user's representation becomes

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{q}_i^{\!T} \left(\mathbf{p}_u + |I_u|^{-1/2}\sum_{j \in I_u} \mathbf{y}_j\right)
$$

The bracketed term reads: *"start from your explicit factors, then add the average of all the implicit-feedback embeddings of items you've interacted with."* The $|I_u|^{-1/2}$ keeps power users from dominating.

```python
class SVDpp:
    """SVD++ factor model with implicit feedback."""

    def __init__(self, k=50, lr=0.005, reg=0.02, epochs=20):
        self.k, self.lr, self.reg, self.epochs = k, lr, reg, epochs

    def fit(self, ratings, implicit=None):
        # ... index users / items, init P, Q, Y, biases, mu ...
        # In each step:
        #   N = implicit_items[u]
        #   y_sum = Y[N].sum(0) / sqrt(len(N))
        #   pred  = mu + b_u + b_i + (P[u] + y_sum) @ Q[i]
        #   gradients flow into P[u], Q[i], Y[N], b_u, b_i
        ...
```

(The full ~120-line implementation is in [the original article version](https://github.com/koren-svd-pp), but the recipe above captures the idea.)

In the Netflix Prize, going from biased MF → SVD++ shaved another ~2 % off RMSE. Tiny in absolute terms, decisive in a leaderboard.

---

## 7 · BPR — when ranking matters more than rating

Squared error is the wrong loss if your real metric is *Did the user click the top item?* RMSE penalises a 4.6 prediction for a true 5 just as much as it penalises ranking *Pursuit* above *Shawshank*.

Bayesian Personalized Ranking (BPR) reframes the problem. Build triples $(u, i, j)$ where user $u$ has interacted with item $i$ but not item $j$. Then maximise the probability that the model scores $i$ above $j$:

$$
\mathcal{L} = \sum_{(u, i, j) \in D_S} \ln \sigma(\hat{r}_{ui} - \hat{r}_{uj}) - \lambda \|\Theta\|^2
$$

where $\sigma$ is the logistic function. Because BPR only uses the *order* of pairs, it shrugs off the absence of negative samples — exactly the situation in implicit-feedback data.

The SGD updates fall out of the gradient. For one triple $(u, i, j)$ with $x_{uij} = \sigma(\hat{r}_{uj} - \hat{r}_{ui})$:

```python
P[u] += lr * (x_uij * (Q[i] - Q[j]) - reg * P[u])
Q[i] += lr * ( x_uij *  P[u]        - reg * Q[i])
Q[j] += lr * (-x_uij *  P[u]        - reg * Q[j])
```

Empirically, BPR pushes AUC 5–10 % above plain MF on implicit data. It is the natural fit when you have clicks, not stars.

---

## 8 · Factorization Machines — when you have side features

Pure MF only knows about user IDs and item IDs. But you usually have more: the user's age, the item's category, the time of day. Factorization Machines (Rendle, 2010) generalise MF to any sparse feature vector $\mathbf{x}$:

$$
\hat{y}(\mathbf{x}) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n}\sum_{j=i+1}^{n} \langle \mathbf{v}_i, \mathbf{v}_j \rangle\, x_i\, x_j
$$

Every feature gets a latent vector $\mathbf{v}_i$, and pairwise interactions are scored by inner products. The clever bit is that the seemingly $O(n^2)$ interaction term collapses to $O(kn)$:

$$
\sum_{i<j} \langle \mathbf{v}_i, \mathbf{v}_j \rangle x_i x_j = \tfrac{1}{2}\sum_{f=1}^{k}\!\left[\Big(\sum_i v_{i,f} x_i\Big)^2 - \sum_i v_{i,f}^2 x_i^2\right]
$$

Linear in features and factors, so FM scales to industrial CTR datasets with millions of one-hot features.

If you set $\mathbf{x}$ to be just `[user_one_hot ; item_one_hot]`, FM reduces exactly to matrix factorization — MF is the special case where features are only IDs.

---

## 9 · Implicit feedback in one paragraph

Real platforms rarely have ratings. They have *behaviour*: views, clicks, listens, purchases. Three tactics handle this:

- **Weighted MF** treats every observed interaction as a 1 with a *confidence* $c_{ui} = 1 + \alpha \log(1 + \text{count})$, and missing entries as 0s with low confidence. The loss becomes $\sum c_{ui} (1 - \hat{r}_{ui})^2 + \lambda \dots$
- **Negative sampling** pairs each observed positive with random unobserved items, then trains as if they were 0s.
- **BPR** as above — uses ranking pairs and avoids the question entirely.

```python
def confidence(count: float, alpha: float = 10.0) -> float:
    """Hu/Koren/Volinsky confidence weighting for implicit feedback."""
    return 1.0 + alpha * np.log1p(count)
```

---

## 10 · Q&A — the parts that bite in production

**User-CF or Item-CF?** Default to Item-CF. Items are usually fewer and more stable, so the similarity matrix is smaller and longer-lived. User-CF is worth a look only when users are rare and active.

**How to pick $k$?** Start at 20–50 for under 100 K users, 50–100 in the millions, 100–200 above that. Then sweep with cross-validation; the sweet spot is the smallest $k$ where validation RMSE stops improving.

**Cold start.** New users → seed factors from registration features (age, locale) or fall back to popular items. New items → seed from content features (category, tags) or use Item-CF on item-side features. The hybrid playbook is "content first, switch to CF once you have ~10–20 interactions."

**SGD or ALS?** SGD wins on memory and online updates. ALS wins on convergence speed and parallelism (Spark, distributed clusters). Many teams bootstrap with ALS overnight, fine-tune with SGD throughout the day.

**Why does BPR beat RMSE-trained MF on implicit data?** Because RMSE optimises the wrong thing. The downstream metric (CTR, NDCG) is a function of ranks, not absolute scores; BPR optimises ranks directly.

**Sparsity tactics.** Factorization itself, plus regularization, plus implicit-feedback signals, plus side features through FM. There is no single fix; it is a stack.

**Why bias terms matter so much.** They isolate "structural" effects (this user is generous, that item is universally loved) from genuine *interaction* signal, freeing the latent factors to model real taste. Expect a 10–20 % RMSE drop the moment you add them.

**Initialisation.** $\mathcal{N}(0, 0.1)$ for factors, zero for biases (they get initialised from data on the first epoch). Avoid all-zero factors — gradients get stuck.

**Out-of-range predictions.** Just clip to $[r_{\min}, r_{\max}]$. A sigmoid scaling is mathematically prettier but rarely beats clipping in offline evaluations.

**How do you serve real-time?** Cache the user vector per user; precompute the top-$N$ for power users; recompute on the fly only for tail users. For continuous learning, use SGD on a stream and version the embedding tables.

**Avoiding overfitting.** L2 regularization with $\lambda \in [0.01, 0.1]$, early stopping on a held-out validation set, and shrinking $k$ before reaching for fancier tricks.

---

## Summary

Collaborative filtering is the foundation that everything else in modern recommendation builds on. The arc is short:

- **Neighbourhood methods** turn "people like you also liked…" into similarity-weighted averages. Item-CF is the production default.
- **Matrix factorization** compresses the sparse rating matrix into low-rank user and item embeddings, predicting ratings as inner products.
- **Bias terms** strip out structural noise; **SVD++** folds in implicit feedback.
- **ALS** and **SGD** are the two optimisation stories — closed-form vs. streaming, parallel vs. online.
- **BPR** swaps RMSE for a ranking loss; **FM** generalises MF to arbitrary sparse features.

These ideas reappear, in disguise, throughout deep recommendation models. Embedding layers in DeepFM are factorization. Two-tower retrieval is matrix factorization with neural encoders. Sequential models still rely on item embeddings learned the way SVD++ learned them. Master this chapter and the rest of the series will feel like variations on a theme.

## References

- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30–37. [IEEE](https://ieeexplore.ieee.org/document/5197422)
- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. *UAI*. [arXiv:1205.2618](https://arxiv.org/abs/1205.2618)
- Rendle, S. (2010). Factorization machines. *ICDM*. [DOI](https://doi.org/10.1109/ICDM.2010.127)
- Koren, Y. (2008). Factorization meets the neighborhood: a multifaceted collaborative filtering model. *KDD*. [DOI](https://doi.org/10.1145/1401890.1401944)
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. *ICDM*. [DOI](https://doi.org/10.1109/ICDM.2008.22)
