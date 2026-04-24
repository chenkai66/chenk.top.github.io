---
title: "Recommendation Systems (1): Fundamentals and Core Concepts"
date: 2025-10-15 09:00:00
tags:
  - Recommendation Systems
  - Collaborative Filtering
  - Introduction
categories: Recommendation Systems
lang: en
mathjax: true
series: recommendation-systems
series_title: "Recommendation Systems Series"
series_order: 1
description: "A beginner-friendly guide to recommendation systems: the three core paradigms (collaborative filtering, content-based, hybrid), evaluation metrics, the multi-stage funnel architecture used in production, and the open challenges of cold-start, sparsity, and the long tail. With working Python implementations."
disableNunjucks: true
---

Open Netflix and the homepage somehow knows you. Scroll TikTok and the next video is the one you didn't realise you wanted. Drop into Spotify on a Monday morning and *Discover Weekly* serves up thirty songs you've never heard of, and you save half of them.

None of this is magic. It is one of the most commercially successful applications of machine learning, quietly running behind almost every consumer product you use: the **recommendation system**.

The numbers explain why every major platform invests so heavily here. McKinsey reports that **35% of Amazon's purchases** come from its recommendation engine. YouTube engineers, in their RecSys 2016 paper, attribute about **70% of watch time** to recommendations. Netflix's product leads have publicly stated that around **75% of what people stream** is driven by their recommender.

This article is the first in a 16-part series. Its job is to give you a complete and honest mental model of how these systems work — enough to understand any modern recommender paper or production architecture you encounter later.

> **What you will learn**
> - The three foundational paradigms — collaborative filtering, content-based, hybrid — and when each wins
> - Matrix factorization, the workhorse that won the Netflix Prize, with full math and code
> - The evaluation metrics that actually matter: Precision, Recall, MAP, NDCG, plus diversity and coverage
> - The multi-stage *funnel* architecture every production system uses to go from millions of items to a top-10 list in 100 ms
> - The five open challenges every recommender engineer wrestles with: cold-start, sparsity, long tail, temporal drift, scale
> - Working Python implementations of User-CF, Item-CF and matrix factorization that you can run today
>
> **Prerequisites:** comfort with Python, basic linear algebra (dot products, matrices), and willingness to read a few formulas. No prior machine-learning experience required.

---

## 1. Why Recommendation Systems Matter

Before any algorithm, it helps to be clear about what problem we are solving.

### The shape of the problem

Every modern catalog is too large for a human to browse:

- Netflix carries **~17,000 titles** worldwide
- YouTube ingests **500 hours of video per minute**
- Spotify hosts **over 100 million tracks**
- Amazon lists **hundreds of millions of products**

Without filtering, a user spends more time searching than consuming, and the *paradox of choice* kicks in — more options, worse decisions, less satisfaction. A recommender's job is to act as a personalised lens that surfaces the tiny fraction of content that matters to *you*.

### The business case

The dollars are large enough to drive entire org charts.

![Cited business-impact figures across major platforms — recommendations now drive a majority of activity at Amazon, Netflix and YouTube](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/01-fundamentals/fig7_business_impact.png)

A few representative numbers, with sources:

| Platform | Impact | Source |
|---|---|---|
| Amazon | ~35% of revenue from recommendations | MGI / McKinsey 2013 |
| Netflix | ~75% of streams driven by the recommender | Gomez-Uribe & Hunt, ACM TMIS 2015 |
| YouTube | ~70% of watch time from recommendations | Covington et al., RecSys 2016 |
| Spotify | *Discover Weekly* surpassed 2.3 B streams within a year | Spotify Newsroom |

These are not marginal lifts. For a platform at scale, a 1% relative improvement in click-through rate is often worth eight or nine figures a year, which is why a recommender team can easily justify hundreds of engineers.

### Where you encounter them daily

Different surfaces demand different recipes:

- **E-commerce** — "Customers who bought this also bought…", personalised home rows, complete-the-look bundles.
- **Streaming video** — taste-clustered home rows, "Because you watched…", auto-play the next episode.
- **Music** — *Discover Weekly*, *Daily Mix*, radio stations seeded from a single track.
- **Social feeds** — engagement-optimised ranking, "People you may know", *For You* pages.

All of these are powered, at the bottom, by a small set of ideas. Let us look at them.

---

## 2. The Three Paradigms

Every recommender, from a 50-line script to YouTube's stack, builds on one of three philosophies:

1. **Collaborative filtering** — learn from *who liked what*.
2. **Content-based filtering** — learn from *what items are made of*.
3. **Hybrid** — combine both, because each has blind spots.

The data we start from looks like this:

![A small user-item rating matrix where most cells are missing — the central data structure of collaborative filtering](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/01-fundamentals/fig1_user_item_matrix.png)

Most entries are unknown. The job of any recommender is, in essence, to fill in the blanks.

### 2.1 Collaborative Filtering — *"users like you liked this"*

Collaborative filtering (CF) rests on a beautifully simple intuition: **if two users agreed in the past, they are likely to agree in the future**. The algorithm needs to know nothing about what an item *is*. It learns purely from the pattern of who interacted with what.

**The taste-twin analogy.** You and a friend have rated 50 movies almost identically — she is your *taste twin*. She watches a new film and loves it. You haven't seen it. The reasonable bet is that you will love it too, even if neither of you can articulate *why*.

**Two flavours.** CF comes in two practical variants:

- **User-based CF** — find users similar to *you*, recommend what they liked.
- **Item-based CF** — find items similar to ones *you* already liked.

Item-based CF tends to win in production, for two reasons. Items move slowly (a movie's neighbours barely change), so similarities can be precomputed and cached. And on most platforms there are far fewer items than users, making item–item more scalable.

**The math.** Let:

- $U = \{u_1, \dots, u_m\}$ be the set of $m$ users
- $I = \{i_1, \dots, i_n\}$ be the set of $n$ items
- $R \in \mathbb{R}^{m \times n}$ be the user-item rating matrix, with $r_{ui}$ user $u$'s rating for item $i$

$R$ is extremely **sparse** — typically less than 0.1% of entries are observed. The goal is to predict the missing ones. For user-based CF:

$$
\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in N(u)} \text{sim}(u, v) \cdot (r_{vi} - \bar{r}_v)}{\sum_{v \in N(u)} |\text{sim}(u, v)|}
$$

In words: start from user $u$'s personal baseline $\bar r_u$, then nudge it by what similar users thought, weighted by *how* similar.

**Choosing a similarity.** Two metrics dominate:

$$
\text{sim}_{\cos}(u, v) = \frac{\sum_{i \in I_{uv}} r_{ui} r_{vi}}{\sqrt{\sum r_{ui}^2}\sqrt{\sum r_{vi}^2}}
$$

$$
\text{sim}_{\text{pearson}}(u, v) = \frac{\sum_{i \in I_{uv}} (r_{ui} - \bar r_u)(r_{vi} - \bar r_v)}{\sqrt{\sum (r_{ui} - \bar r_u)^2}\sqrt{\sum (r_{vi} - \bar r_v)^2}}
$$

The crucial difference: Alice rates everything 4–5 and Bob rates everything 2–3. Cosine sees them as different. Pearson centres each user first, recognises that their *relative* preferences may be identical, and rates them as twins.

**Strengths and weaknesses.**

| Strengths | Weaknesses |
|---|---|
| Domain-agnostic — no item features needed | Cold start: useless for brand-new users or items |
| Captures serendipity | Suffers when data is sparse |
| Improves automatically with more data | Popularity bias — popular items get recommended more |

### 2.2 Content-Based Filtering — *"you'll like this because it's similar"*

Where CF asks "who else liked this?", content-based filtering asks "what *is* this thing, and have you liked similar things before?"

The algorithm builds a profile of *your* taste from the **attributes** of items you've engaged with, then ranks unseen items by how well they match that profile. It does not care what other users think.

**Representing items.** The choice of feature representation depends entirely on the medium:

- **Text** (news, articles): TF-IDF vectors, sentence embeddings, topic models
- **Images** (products, art): CNN embeddings, colour histograms, visual attributes
- **Audio** (music): MFCCs, learned audio embeddings, metadata (tempo, key, genre)
- **Structured** (movies, products): one-hot genres, prices, knowledge-graph features

**The math.** Let $\mathbf{x}_i \in \mathbb{R}^d$ be the feature vector for item $i$ and $\mathbf{w}_u \in \mathbb{R}^d$ be user $u$'s learned preference weights. Then:

$$
\hat{r}_{ui} = \mathbf{w}_u^\top \mathbf{x}_i + b_u
$$

Each item is a list of feature scores; each user has a weight per feature; multiply, sum, done. We learn $\mathbf{w}_u$ by minimising squared error on the items that user has rated:

$$
\min_{\mathbf{w}_u, b_u} \sum_{i \in I_u} (r_{ui} - \mathbf{w}_u^\top \mathbf{x}_i - b_u)^2 + \lambda \|\mathbf{w}_u\|^2
$$

The $\lambda \|\mathbf{w}_u\|^2$ term is **L2 regularisation** — a tax on large weights that prevents the model from over-fitting on a handful of ratings.

**Strengths and weaknesses.**

| Strengths | Weaknesses |
|---|---|
| Handles new items as soon as features exist | Requires good features, which means feature engineering |
| Recommendations are easy to explain | Tends to over-specialise — keeps recommending the same kind of thing |
| Works with a single user's history | New users still cold-start (no profile yet) |

### 2.3 Hybrid Methods — *combining what each does best*

Pure CF and pure content-based each have predictable failure modes. Hybrids combine them so that when one approach falters, the other takes over.

![Side-by-side comparison of the three paradigms across cold-start handling, serendipity, explainability, sparsity tolerance and feature requirements — higher is better on every axis](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/01-fundamentals/fig2_paradigm_comparison.png)

There are five common ways to combine them:

**1. Weighted.** Linear blend of two scores:

$$
\hat{r}_{ui} = \alpha \cdot \hat{r}_{ui}^{\text{CF}} + (1 - \alpha) \cdot \hat{r}_{ui}^{\text{CB}}
$$

The weight $\alpha$ can be fixed, learned on a validation set, or made adaptive (more CF for power users, more content for newcomers).

**2. Switching.** Pick a method based on context.

```python
def predict(user, item):
    if item.rating_count < MIN_RATINGS:
        return content_based_predict(user, item)
    if user.rating_count < MIN_USER_RATINGS:
        return content_based_predict(user, item)
    return collaborative_filtering_predict(user, item)
```

**3. Feature combination.** Treat content features as additional signals inside a single CF model.

**4. Cascade.** Use a cheap method to generate candidates, then a precise method to re-rank. This is the dominant pattern in industry — and it generalises into the funnel architecture we'll see in §5.

**5. Meta-level.** Use one model's outputs as another's inputs. Modern deep recommenders that learn item embeddings from content and then apply CF on those embeddings fall into this bucket.

**A real example.** Netflix's production recommender is a sophisticated hybrid: matrix factorization on viewing history, deep nets that fuse text/image/audio features, context signals (time of day, device), business constraints (recency, diversity), and a final ensemble across dozens of models. Almost no major platform runs a single algorithm any more.

---

## 3. Matrix Factorization: The Workhorse

Neighbourhood methods like User-CF and Item-CF are intuitive, but **matrix factorization (MF)** is the technique that won the Netflix Prize and underpins much of what came after, including modern two-tower deep models.

### 3.1 The geometric idea

MF assumes user preferences and item characteristics can be summarised by a small number of **latent factors** — hidden axes that emerge from the data rather than being labelled by hand.

![Users and items embedded in the same 2D latent space; a high dot product means the item is a good match for the user](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/01-fundamentals/fig4_embedding_space.png)

The picture says it best: every user and every item lives at a point in a $k$-dimensional space (typically $k = 20$–$200$ in practice). For movies, the axes might end up looking like *blockbuster ↔ art-house* or *light ↔ serious*, but you never tell the model that — it discovers them.

The predicted rating is a **dot product**:

$$
\hat{r}_{ui} = \mathbf{p}_u^\top \mathbf{q}_i
$$

If user and item vectors point in similar directions, the dot product is large, and the model predicts a high rating.

### 3.2 The optimisation problem

We factor $R \approx P Q^\top$ with $P \in \mathbb{R}^{m \times k}$ holding user vectors and $Q \in \mathbb{R}^{n \times k}$ holding item vectors. Train by minimising squared error on observed ratings, with L2 regularisation:

$$
\min_{P, Q} \sum_{(u, i) \in \Omega} (r_{ui} - \mathbf{p}_u^\top \mathbf{q}_i)^2 + \lambda (\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2)
$$

In production we add bias terms to capture systematic effects:

$$
\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^\top \mathbf{q}_i
$$

where $\mu$ is the global mean, $b_u$ is "this user rates everything high", and $b_i$ is "this item is universally loved (or hated)". With those biases pulled out, the latent factors only have to model the *interaction*, which is what we actually want them to learn.

### 3.3 Two ways to optimise

**Stochastic Gradient Descent (SGD).** For each observed rating, compute the error $e_{ui} = r_{ui} - \hat{r}_{ui}$ and step:

$$
\mathbf{p}_u \leftarrow \mathbf{p}_u + \eta (e_{ui} \mathbf{q}_i - \lambda \mathbf{p}_u), \quad \mathbf{q}_i \leftarrow \mathbf{q}_i + \eta (e_{ui} \mathbf{p}_u - \lambda \mathbf{q}_i)
$$

**Alternating Least Squares (ALS).** Fix $Q$, solve a closed-form least-squares problem for each $\mathbf{p}_u$, then swap. ALS parallelises trivially across users (and items), which is why Spark MLlib's recommender ships with it.

### 3.4 Implementation

Here is a minimal but complete MF with biases, trained by SGD:

```python
import numpy as np
from typing import List, Tuple


class MatrixFactorization:
    """Matrix factorization with user/item biases, trained via SGD."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int = 20,
        learning_rate: float = 0.01,
        regularization: float = 0.02,
        n_epochs: int = 20,
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.n_epochs = n_epochs

        # Latent factors initialised with small random values.
        rng = np.random.default_rng(0)
        self.P = rng.normal(0, 0.1, (n_users, n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, n_factors))

        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = 0.0

    def fit(self, ratings: List[Tuple[int, int, float]]) -> "MatrixFactorization":
        self.global_mean = float(np.mean([r for _, _, r in ratings]))
        for epoch in range(self.n_epochs):
            np.random.shuffle(ratings)
            sse = 0.0
            for u, i, r in ratings:
                pred = self._predict(u, i)
                err = r - pred
                sse += err ** 2

                p_u, q_i = self.P[u].copy(), self.Q[i].copy()
                self.P[u] += self.lr * (err * q_i - self.reg * p_u)
                self.Q[i] += self.lr * (err * p_u - self.reg * q_i)
                self.user_bias[u] += self.lr * (err - self.reg * self.user_bias[u])
                self.item_bias[i] += self.lr * (err - self.reg * self.item_bias[i])

            if (epoch + 1) % 5 == 0:
                rmse = np.sqrt(sse / len(ratings))
                print(f"epoch {epoch + 1:3d}  RMSE = {rmse:.4f}")
        return self

    def _predict(self, u: int, i: int) -> float:
        return (
            self.global_mean
            + self.user_bias[u]
            + self.item_bias[i]
            + self.P[u] @ self.Q[i]
        )

    def recommend(
        self, u: int, top_n: int = 10, exclude: set | None = None
    ) -> List[Tuple[int, float]]:
        exclude = exclude or set()
        scores = [(i, self._predict(u, i)) for i in range(self.n_items) if i not in exclude]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]


if __name__ == "__main__":
    ratings = [
        (0, 0, 5.0), (0, 1, 3.0), (0, 2, 2.5),
        (1, 0, 4.0), (1, 2, 4.5), (1, 3, 3.0),
        (2, 1, 3.5), (2, 2, 4.0), (2, 4, 2.0),
        (3, 0, 3.0), (3, 3, 4.0), (3, 4, 4.5),
        (4, 1, 4.0), (4, 2, 3.5), (4, 3, 5.0),
    ]
    model = MatrixFactorization(n_users=5, n_items=5, n_factors=10, n_epochs=50)
    model.fit(ratings)
    print("\ntop-3 for user 0:", model.recommend(0, top_n=3, exclude={0, 1, 2}))
```

> **Try it.** Sweep `n_factors` from 2 to 50 and watch RMSE: too few and the model under-fits, too many and it over-fits this tiny dataset. The right value on real data is almost always in the 20–200 range.

### 3.5 Implicit feedback

In production you rarely have explicit 1-to-5 ratings. You have **implicit feedback**: clicks, watch time, dwell time, purchases. The trouble is that *no interaction* doesn't mean *dislike* — the user may simply never have seen the item.

The standard fix is **weighted matrix factorization** (Hu, Koren & Volinsky 2008):

$$
\min_{P, Q} \sum_{u, i} c_{ui} (p_{ui} - \mathbf{p}_u^\top \mathbf{q}_i)^2 + \lambda(\|\mathbf{p}_u\|^2 + \|\mathbf{q}_i\|^2)
$$

with $p_{ui} \in \{0, 1\}$ (interacted or not) and $c_{ui} = 1 + \alpha f_{ui}$ a confidence that grows with interaction frequency $f_{ui}$. We optimise over *all* user-item pairs, but observed interactions get high confidence ("definitely a positive") and unobserved ones get low confidence ("probably negative, but we are not sure").

---

## 4. Evaluation: Measuring What Matters

Building a recommender is half the battle. Knowing whether it is actually good is the other half — and it is genuinely harder than people assume.

### 4.1 Why a single accuracy number lies

If your model has 95% accuracy at predicting ratings, is it good? Maybe — but consider:

- Users only rate things they like. Predicting "high" for every item can score well and recommend nothing useful.
- If your test set is dominated by popular items, you've measured the easy cases.
- If recommendations lack diversity, users may be satisfied today and churn next month.

Honest evaluation needs **multiple metrics** measuring different things.

### 4.2 Classification metrics

For each user $u$, let $R_u$ be the items we recommended (top-$K$) and $T_u$ the items they actually liked.

$$
\text{Precision@}K = \frac{|R_u \cap T_u|}{|R_u|}, \quad \text{Recall@}K = \frac{|R_u \cap T_u|}{|T_u|}
$$

**Precision@K** asks: of what we recommended, how much was relevant?

**Recall@K** asks: of what was relevant, how much did we surface?

These are intuitive but they ignore *order* within the top-$K$ list — a relevant item at position 1 counts the same as at position 10.

### 4.3 Ranking metrics — position matters

In real interfaces, position 1 is worth dramatically more than position 10. Two metrics dominate.

**Mean Average Precision (MAP).** For one user:

$$
\text{AP@}K = \frac{1}{|T_u|} \sum_{k=1}^{K} \text{Precision@}k \cdot \text{rel}(k)
$$

where $\text{rel}(k) = 1$ when item at rank $k$ is relevant. MAP is the mean over users. It rewards putting hits early.

**Normalised Discounted Cumulative Gain (NDCG).** The gold standard when you have graded relevance:

$$
\text{DCG@}K = \sum_{k=1}^{K} \frac{2^{\text{rel}_k} - 1}{\log_2(k + 1)}, \quad \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}
$$

The $\log_2(k+1)$ term *discounts* lower positions — rank 1 gets full credit, rank 10 gets about a third. IDCG is the DCG of the perfectly sorted list, so NDCG always lands in $[0, 1]$.

The same ranked list can look very different through these three lenses:

![Worked example of Precision@k, Recall@k and NDCG@k on a single ranked list of 10 items, 4 of which are relevant](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/01-fundamentals/fig5_evaluation_metrics.png)

A reference implementation:

```python
import numpy as np
from typing import List


def precision_at_k(ranked: List[int], relevant: set, k: int) -> float:
    if k == 0:
        return 0.0
    hits = sum(1 for i in ranked[:k] if i in relevant)
    return hits / k


def recall_at_k(ranked: List[int], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for i in ranked[:k] if i in relevant)
    return hits / len(relevant)


def average_precision(ranked: List[int], relevant: set, k: int) -> float:
    """AP@K — rewards relevant items appearing earlier."""
    if not relevant:
        return 0.0
    score, hits = 0.0, 0
    for rank, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def ndcg_at_k(true_relevances: List[float], k: int) -> float:
    """NDCG@K — works with graded relevance (ratings)."""
    rel = np.asarray(true_relevances[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rel.size + 2))
    dcg = np.sum((2 ** rel - 1) / discounts)
    ideal = np.sort(np.asarray(true_relevances, dtype=float))[::-1][:k]
    idcg = np.sum((2 ** ideal - 1) / discounts[: ideal.size])
    return float(dcg / idcg) if idcg else 0.0


if __name__ == "__main__":
    ranked = [5, 2, 8, 1, 9, 3, 7, 4, 6, 10]
    relevant = {1, 2, 5, 7}
    print(f"Precision@5  = {precision_at_k(ranked, relevant, 5):.3f}")
    print(f"Recall@5     = {recall_at_k(ranked, relevant, 5):.3f}")
    print(f"AP@10        = {average_precision(ranked, relevant, 10):.3f}")
    print(f"NDCG@10      = {ndcg_at_k([5,5,2,5,1,3,4,2,1,1], 10):.3f}")
```

### 4.4 Beyond accuracy: coverage, diversity, serendipity

A recommender that always serves the global top 10 will score well on precision and ruin your product. Three accuracy-orthogonal metrics keep you honest:

- **Catalog coverage** — fraction of items the system *ever* recommends. If you only ever surface 1% of your catalog, the other 99% might as well not exist.
- **Intra-list diversity** — average pairwise dissimilarity inside a recommendation list. Ten near-duplicate sequels are not a great list.
- **Serendipity** — items that are both relevant and unexpected. The "wow, how did it know I'd like this?" effect.

These objectives often *conflict* with raw accuracy. Picking the trade-off is a product decision, not an algorithmic one.

### 4.5 Online metrics — what really decides

Offline metrics guide development. Online metrics decide promotions. The ones that matter in production:

- **CTR** — clicks / impressions
- **Conversion rate** — purchases / clicks
- **Engagement** — watch time, session length, return rate
- **Business** — revenue per user, LTV, churn

The tool is **A/B testing**: split users randomly, run the new system against the old, ship the winner if (and only if) the lift survives statistical scrutiny.

---

## 5. System Architecture: From Millions to Top-10

A real recommender has a daunting job: pick 10 items from a catalog of 10⁷, in <100 ms, hundreds of thousands of times per second. No single model can do that. The answer everyone has converged on is a **multi-stage funnel** that progressively narrows the candidate set while applying increasingly expensive models.

![Recommendation funnel: catalog → recall → coarse rank → fine rank → rerank, with item counts and per-stage latency budgets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/01-fundamentals/fig3_funnel_architecture.png)

### Stage 1 — Recall (a.k.a. candidate generation)

Goal: collapse 10⁷ items down to a few thousand likely-relevant candidates, in well under 10 ms.

Multiple cheap *recall channels* run in parallel — Item-CF, embedding ANN search (FAISS, ScaNN, HNSW), trending content, social graph, content match — and their outputs are merged. This stage is engineered for **recall**: it is fine to pass through many false positives, the next stages will weed them out. Missing a true positive here is fatal — no later stage can recover an item that was never proposed.

```python
class CandidateGenerator:
    def __init__(self):
        self.user_cf = UserCFRecall()
        self.item_cf = ItemCFRecall()
        self.mf      = MatrixFactorizationRecall()
        self.popular = PopularItemsRecall()
        self.content = ContentBasedRecall()

    def generate(self, user_id: int, n: int = 2000) -> list[int]:
        candidates: set[int] = set()
        candidates.update(self.user_cf.recall(user_id, n=500))
        candidates.update(self.item_cf.recall(user_id, n=500))
        candidates.update(self.mf.recall(user_id, n=500))
        candidates.update(self.popular.recall(user_id, n=200))
        candidates.update(self.content.recall(user_id, n=300))
        candidates -= self.history(user_id)
        return list(candidates)[:n]
```

### Stage 2 — Coarse ranking

Goal: cut a few thousand candidates to a few hundred, with a *cheap-but-not-tiny* model.

Typical choices: logistic regression, small MLPs, or a distilled deep model. The features are coarse — user demographics, item popularity, recall scores, the recall channel itself — because we cannot afford rich features at this scale.

### Stage 3 — Fine ranking

Goal: rank the remaining few hundred candidates with the **best, most expensive** model in the stack.

This is where the deep nets live: Wide & Deep, DCN, DIN, transformer-based sequence models, two-tower architectures. They consume rich features (long user history, real-time session, item embeddings, cross features) and often optimise multiple objectives at once — click *and* conversion *and* watch time.

```python
import torch
import torch.nn as nn


class WideAndDeep(nn.Module):
    """Wide & Deep ranker — linear memorisation + MLP generalisation."""

    def __init__(
        self,
        n_wide_features: int,
        embedding_sizes: list[int],
        embedding_dim: int = 16,
        deep_layers: tuple[int, ...] = (128, 64, 32),
    ):
        super().__init__()
        self.wide = nn.Linear(n_wide_features, 1)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, embedding_dim) for n in embedding_sizes]
        )

        layers, in_dim = [], len(embedding_sizes) * embedding_dim
        for hidden in deep_layers:
            layers += [nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.3)]
            in_dim = hidden
        self.deep = nn.Sequential(*layers)
        self.out = nn.Linear(deep_layers[-1] + 1, 1)

    def forward(self, wide_x, emb_idx):
        wide = self.wide(wide_x)
        deep_in = torch.cat(
            [emb(emb_idx[:, i]) for i, emb in enumerate(self.embeddings)], dim=1
        )
        deep = self.deep(deep_in)
        return torch.sigmoid(self.out(torch.cat([wide, deep], dim=1))).squeeze()
```

### Stage 4 — Rerank and policy

Goal: turn a relevance-sorted list into a *good* final list.

This is where business logic lives: enforce diversity (often via Maximal Marginal Relevance), inject freshness, demote items the user just saw, apply content-policy filters, honour sponsored placements, satisfy fairness constraints.

```python
import numpy as np


def maximal_marginal_relevance(
    candidates: list,
    scores: np.ndarray,
    similarity: np.ndarray,
    top_n: int,
    lam: float = 0.5,
) -> list:
    """MMR: balance relevance and diversity. lam=1 ⇒ pure relevance."""
    selected = [int(np.argmax(scores))]
    remaining = [i for i in range(len(candidates)) if i != selected[0]]
    while len(selected) < top_n and remaining:
        mmr = [
            lam * scores[i] - (1 - lam) * max(similarity[i, j] for j in selected)
            for i in remaining
        ]
        pick = remaining[int(np.argmax(mmr))]
        selected.append(pick)
        remaining.remove(pick)
    return [candidates[i] for i in selected]
```

The funnel's discipline is what makes it possible to combine "deep learning quality" with "100 ms latency". Cheap models cast a wide net; expensive models do the careful work on a small set.

---

## 6. The Five Open Challenges

Decades of research later, every recommender team still wrestles with the same five problems. There is no clean solution to any of them.

### Cold start

A new user has no history. A new item has no interactions. Both look invisible to CF.

For **new users** — onboard with a few quick taste choices, fall back to demographics, lean on popular and trending until a few signals accumulate.

For **new items** — use content features (embedding the item from text/image/audio puts it near similar items), and use *exploration*: deliberately surface new items with an upper-confidence-bound bonus

$$
\text{score}(i) = \hat{r}_i + \beta \sqrt{\frac{\log N}{n_i}}
$$

so items with few impressions get a chance to prove themselves.

```python
import numpy as np


def epsilon_greedy_recommend(
    candidates, scores, item_counts, *, epsilon=0.1, top_n=10, threshold=10
):
    """ε-greedy: spend most of top-N on exploitation, a slice on exploration."""
    candidates = np.asarray(candidates)
    scores = np.asarray(scores)
    cold = np.array([item_counts.get(c, 0) < threshold for c in candidates])

    n_explore = int(round(top_n * epsilon))
    n_exploit = top_n - n_explore

    exploit = candidates[~cold][np.argsort(-scores[~cold])][:n_exploit]
    explore_pool = candidates[cold]
    explore = np.random.choice(
        explore_pool, size=min(n_explore, len(explore_pool)), replace=False
    ) if len(explore_pool) else np.array([], dtype=candidates.dtype)

    out = np.concatenate([exploit, explore])
    np.random.shuffle(out)
    return out[:top_n].tolist()
```

### Sparsity

Netflix has on the order of 200M users and 17K titles — about 3.4 × 10⁹ possible interactions. The average user rates a few dozen movies. Density is well below 0.001%.

Tools that help: matrix factorization (shares statistical strength via the latent space), implicit feedback (clicks and dwell time densify the matrix dramatically), cross-domain transfer (use music taste to seed movie taste), and aggressive regularisation.

### The long tail

Item popularity follows a power law. The picture is always the same:

![Long-tail distribution: a tiny head absorbs most interactions, while half the catalog falls into a sparsely-populated cold-start zone](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/01-fundamentals/fig6_cold_start_longtail.png)

The top few percent of items absorb the majority of interactions. The bottom half lives in the *cold-start zone* — too few signals for CF to do anything.

This causes three real harms: popularity bias (rich-get-richer feedback loops), creator unfairness (niche creators stay invisible), and user dissatisfaction (niche tastes are poorly served). Mitigations include **inverse-propensity weighting** (down-weight popular items in training), explicit diversity constraints in reranking, and contextual bandits that *learn* which tail items resonate with which user clusters.

### Temporal drift

Tastes evolve. Items age. Trends explode and burn out. A model trained on last quarter's behaviour can be confidently wrong about today.

Standard remedies: time-decayed training weights $w(t) = e^{-\lambda(t_{\text{now}} - t)}$, online learning with small incremental updates, time-of-day and freshness features, and session-based sequence models that condition on the last few minutes of behaviour rather than the last few months.

### Scale

Pinterest, ByteDance, Meta — the largest systems serve millions of QPS at sub-100 ms p99. That is a brutal engineering constraint, and it shapes algorithm choices as much as accuracy does.

The standard toolkit: aggressive caching, ANN libraries (FAISS, ScaNN, HNSW), model quantisation and distillation, sharding by user, and a strict offline/online split — pre-compute everything you can the night before, and only do request-specific work at serve time.

---

## 7. Implementations from Scratch

Reading code teaches things that prose does not. Two reference implementations follow: User-CF and Item-CF, both fully runnable.

### 7.1 User-Based Collaborative Filtering

```python
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple


class UserBasedCF:
    """User-based CF with Pearson similarity and mean-centred prediction."""

    def __init__(self, k_neighbors: int = 50, min_common_items: int = 3,
                 similarity: str = "pearson"):
        self.k = k_neighbors
        self.min_common = min_common_items
        self.similarity = similarity

        self.user_ratings: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.item_users:   Dict[int, set]              = defaultdict(set)
        self.user_means:   Dict[int, float]            = {}
        self.global_mean = 0.0

    def fit(self, ratings: List[Tuple[int, int, float]]) -> "UserBasedCF":
        all_r = []
        for u, i, r in ratings:
            self.user_ratings[u][i] = r
            self.item_users[i].add(u)
            all_r.append(r)
        self.global_mean = float(np.mean(all_r))
        for u, items in self.user_ratings.items():
            self.user_means[u] = float(np.mean(list(items.values())))
        return self

    def _sim(self, u: int, v: int) -> float:
        common = set(self.user_ratings[u]) & set(self.user_ratings[v])
        if len(common) < self.min_common:
            return 0.0
        r1 = np.array([self.user_ratings[u][i] for i in common])
        r2 = np.array([self.user_ratings[v][i] for i in common])
        if self.similarity == "pearson":
            r1 = r1 - self.user_means[u]
            r2 = r2 - self.user_means[v]
        n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
        return float(r1 @ r2 / (n1 * n2)) if n1 and n2 else 0.0

    def predict(self, u: int, i: int) -> float:
        if u not in self.user_ratings:
            return self.global_mean
        if i not in self.item_users:
            return self.user_means.get(u, self.global_mean)
        if i in self.user_ratings[u]:
            return self.user_ratings[u][i]

        sims = [(v, self._sim(u, v)) for v in self.item_users[i] if v != u]
        sims = sorted((s for s in sims if s[1] > 0), key=lambda x: -x[1])[: self.k]
        if not sims:
            return self.user_means[u]

        weighted = sum(s * (self.user_ratings[v][i] - self.user_means[v]) for v, s in sims)
        denom    = sum(abs(s) for _, s in sims)
        return self.user_means[u] + weighted / denom if denom else self.user_means[u]

    def recommend(self, u: int, n: int = 10, exclude_rated: bool = True
                  ) -> List[Tuple[int, float]]:
        if u not in self.user_ratings:
            popular = sorted(self.item_users.items(), key=lambda x: -len(x[1]))
            return [(i, self.global_mean) for i, _ in popular[:n]]
        rated = set(self.user_ratings[u]) if exclude_rated else set()
        scores = [(i, self.predict(u, i)) for i in self.item_users if i not in rated]
        return sorted(scores, key=lambda x: -x[1])[:n]
```

### 7.2 Item-Based Collaborative Filtering

The architecture mirrors User-CF but flips the question: instead of "which users are like me?", we ask "which items are like the ones I already liked?" The standard similarity is **adjusted cosine** — cosine after subtracting each *user's* mean, which removes their personal rating scale.

```python
class ItemBasedCF:
    """Item-based CF with adjusted-cosine similarity and a similarity cache."""

    def __init__(self, k_neighbors: int = 50, min_common_users: int = 3):
        self.k = k_neighbors
        self.min_common = min_common_users
        self.item_ratings: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.user_items:   Dict[int, set]              = defaultdict(set)
        self.user_means:   Dict[int, float]            = {}
        self.sim_cache:    Dict[Tuple[int, int], float] = {}
        self.global_mean = 0.0

    def fit(self, ratings: List[Tuple[int, int, float]]) -> "ItemBasedCF":
        all_r = []
        for u, i, r in ratings:
            self.item_ratings[i][u] = r
            self.user_items[u].add(i)
            all_r.append(r)
        self.global_mean = float(np.mean(all_r))
        for u, items in self.user_items.items():
            self.user_means[u] = float(np.mean([self.item_ratings[i][u] for i in items]))
        return self

    def _sim(self, a: int, b: int) -> float:
        key = (min(a, b), max(a, b))
        if key in self.sim_cache:
            return self.sim_cache[key]
        common = set(self.item_ratings[a]) & set(self.item_ratings[b])
        if len(common) < self.min_common:
            self.sim_cache[key] = 0.0
            return 0.0
        r1 = np.array([self.item_ratings[a][u] - self.user_means[u] for u in common])
        r2 = np.array([self.item_ratings[b][u] - self.user_means[u] for u in common])
        n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
        s = float(r1 @ r2 / (n1 * n2)) if n1 and n2 else 0.0
        self.sim_cache[key] = s
        return s

    def predict(self, u: int, i: int) -> float:
        if u not in self.user_items:
            return self.global_mean
        if i not in self.item_ratings:
            return self.user_means.get(u, self.global_mean)
        sims = [(j, self._sim(i, j)) for j in self.user_items[u] if j != i]
        sims = sorted((s for s in sims if s[1] > 0), key=lambda x: -x[1])[: self.k]
        if not sims:
            return self.user_means.get(u, self.global_mean)
        num = sum(s * self.item_ratings[j][u] for j, s in sims)
        den = sum(abs(s) for _, s in sims)
        return num / den if den else self.user_means.get(u, self.global_mean)

    def similar_items(self, i: int, n: int = 10) -> List[Tuple[int, float]]:
        others = [j for j in self.item_ratings if j != i]
        sims = [(j, self._sim(i, j)) for j in others]
        return sorted(sims, key=lambda x: -x[1])[:n]
```

---

## 8. Frequently Asked Questions

**When should I use CF vs. content-based?**

Use **CF** when you have abundant interaction data, items are hard to describe with features (what makes a song "good"?), and you want serendipity. Use **content-based** when item metadata is rich but interactions are sparse, when items are constantly new, when you need explainable recommendations, or when privacy precludes broad behavioural data. In production, ship a hybrid.

**How do I handle implicit feedback (clicks, views) instead of explicit ratings?**

Implicit data is more abundant but noisier — absence of interaction is *not* dislike. Three standard approaches: weighted matrix factorization (treat observed as positive with high confidence, unobserved as negative with low confidence), negative sampling (sample a few unobserved items per positive), or pairwise ranking losses such as BPR.

**What dimensionality should I use for matrix factorization?**

Typical range: 20–200. Below 10 you usually under-fit. Above 500 you get diminishing returns, longer training, and overfitting risk. Start at 50–100 and tune on validation.

**How often should I retrain?**

Depends on data velocity. TikTok-like systems do online learning with sub-second updates. Netflix and Spotify retrain daily or weekly. Long-lived catalogs like Amazon products can get away with weekly to monthly retraining for the heavy components, with online updates for user state.

**How do I balance exploration vs. exploitation?**

Pure exploitation creates filter bubbles; pure exploration annoys users. Practical recipes: ε-greedy with $\epsilon \in [0.05, 0.15]$, Thompson sampling for a Bayesian flavour that adapts automatically, or simply reserving 1–2 slots in the top-10 for exploratory items chosen by an upper-confidence-bound rule.

---

## 9. Takeaways and What's Next

Five things to remember from this article:

1. **No single algorithm wins.** The right approach depends on data, scale, and product goals. Hybrids dominate in practice.
2. **Architecture beats algorithms at scale.** The funnel — recall → coarse rank → fine rank → rerank — is what lets a deep model serve in 100 ms.
3. **Evaluation is multi-objective.** Accuracy, ranking quality, diversity, coverage, and online business metrics all matter, and they often disagree.
4. **The hard problems are perennial.** Cold-start, sparsity, the long tail, temporal drift, scale — every team works on these forever.
5. **Read code.** The intuition lives in the implementations as much as in the equations.

In the rest of this series we go deep on each piece:

- **Part 2** — collaborative filtering and matrix factorization in full detail
- **Part 3** — deep-learning building blocks for recommenders
- **Part 4** — CTR prediction (Wide & Deep, DCN, DeepFM)
- **Part 5** — embedding techniques (Word2Vec, item2vec, graph embeddings)
- **Parts 6–10** — sequential models, GNNs, knowledge graphs, multi-task, DIN
- **Parts 11–16** — contrastive learning, LLM-based recommenders, fairness, cross-domain, real-time, industrial practice

**Further reading**

- Aggarwal, *Recommender Systems: The Textbook* — the most comprehensive single reference
- Koren, Bell & Volinsky, *Matrix Factorization Techniques for Recommender Systems* (IEEE Computer 2009) — the Netflix Prize paper
- Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets* (ICDM 2008)
- Cheng et al., *Wide & Deep Learning for Recommender Systems* (DLRS 2016)
- Covington, Adams & Sargin, *Deep Neural Networks for YouTube Recommendations* (RecSys 2016)
- He et al., *Neural Collaborative Filtering* (WWW 2017)

**Open-source libraries to play with**

- [Surprise](https://surpriselib.com/) — small, classroom-friendly
- [implicit](https://github.com/benfred/implicit) — fast ALS / BPR for implicit feedback
- [LightFM](https://github.com/lyst/lightfm) — clean hybrid models
- [RecBole](https://github.com/RUCAIBox/RecBole) — comprehensive research toolkit
- [TensorFlow Recommenders](https://www.tensorflow.org/recommenders) — production-grade

---

> **Series Navigation**
>
> You are reading **Part 1** of 16 in the Recommendation Systems series.
>
> - **Part 1: Fundamentals and Core Concepts** (you are here)
> - [Part 2: Collaborative Filtering and Matrix Factorization](/en/recommendation-systems-2-collaborative-filtering/)
> - [Part 3: Deep Learning Foundation Models](/en/recommendation-systems-3-deep-learning-basics/)
> - [Part 4: CTR Prediction Models](/en/recommendation-systems-4-ctr-prediction/)
> - [Part 5: Embedding Techniques](/en/recommendation-systems-5-embedding-techniques/)
> - [Part 6: Sequential Recommendation](/en/recommendation-systems-6-sequential-recommendation/)
> - [Part 7: Graph Neural Networks](/en/recommendation-systems-7-graph-neural-networks/)
> - [Part 8: Knowledge Graph Integration](/en/recommendation-systems-8-knowledge-graph/)
> - [Part 9: Multi-Task Learning](/en/recommendation-systems-9-multi-task-learning/)
> - [Part 10: Deep Interest Networks](/en/recommendation-systems-10-deep-interest-networks/)
> - [Part 11: Contrastive Learning](/en/recommendation-systems-11-contrastive-learning/)
> - [Part 12: LLM-Based Recommendation](/en/recommendation-systems-12-llm-recommendation/)
> - [Part 13: Fairness and Explainability](/en/recommendation-systems-13-fairness-explainability/)
> - [Part 14: Cross-Domain and Cold Start](/en/recommendation-systems-14-cross-domain-cold-start/)
> - [Part 15: Real-Time and Online Learning](/en/recommendation-systems-15-real-time-online/)
> - [Part 16: Industrial Practice](/en/recommendation-systems-16-industrial-practice/)
