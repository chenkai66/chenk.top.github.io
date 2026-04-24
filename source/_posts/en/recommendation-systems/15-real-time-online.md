---
title: "Recommendation Systems (15): Real-Time Recommendation and Online Learning"
date: 2025-12-10 09:00:00
tags:
  - Recommendation Systems
  - Real-Time
  - Online Learning
categories: Recommendation Systems
series:
  name: "Recommendation Systems"
  part: 15
  total: 16
lang: en
mathjax: true
description: "A practitioner's guide to real-time recommendation: streaming pipelines (Kafka + Flink), online learning (SGD, FTRL, AdaGrad), bandits (UCB, Thompson Sampling, LinUCB), latency budgets, feature freshness, concept drift, and the cache-vs-compute trade-off you actually tune in production."
disableNunjucks: true
series_order: 15
---

> A user opens your app at 14:02 and searches for "trail running shoes". By 15:30 they have moved on and are reading kitchen reviews. A model that retrains nightly is still showing them Salomon ads at 16:00 — and that gap is exactly the bug a real-time system fixes. The interesting part is not "make it faster" but "what *should* be fast" — most features add nothing to AUC even when made real-time, and the wrong design point burns money for no lift.

## What you will learn

- **The two paths**: a real-time recommender is two pipelines glued together — an *asynchronous write-path* (events → state → model) and a *synchronous read-path* (request → recall → rank → response).
- **Where the milliseconds go**: latency is not an average; the p99 tail is what users feel. We break down the 100 ms budget by stage and by percentile.
- **Online learning in practice**: SGD, AdaGrad, and the production workhorse — FTRL-Proximal — that powers Google's Ad Click Prediction.
- **Bandits**: UCB1, Thompson Sampling, and contextual LinUCB — what their regret bounds actually mean when items churn daily.
- **Streaming architecture**: a concrete Kafka + Flink + KV-store layout, with checkpointing and exactly-once semantics.
- **Concept drift**: how to detect it (ADWIN, DDM, page-Hinkley) and what to do once you have.
- **Cache vs compute**: the trade-off you actually tune — and why the answer is almost always *hybrid*.

## Prerequisites

- Python and NumPy (Parts 1-2)
- SGD and loss functions (Part 7)
- The recommendation pipeline overview (Part 11)

---

## 1. Why real-time, and what "real-time" actually means

Three realities push toward real time:

1. **Sessions are short.** Median session length on a feed app is 3-7 minutes. A model that updates daily literally never sees most sessions before they end.
2. **Trends are short-lived.** A meme video can hit 80 % of its lifetime engagement in the first 6 hours. Yesterday's batch model has nothing to recommend it with.
3. **The feedback loop is the model.** Once you serve a recommendation, the click that comes back is the *next* training example. Closing that loop in seconds vs days is the difference between a learning system and a stale one.

But "real-time" is not a single thing — it is a spectrum:

| Tier | Update cadence | Typical use |
|------|----------------|-------------|
| Real-time | < 1 second | session intent, in-feed dedup, abuse signals |
| Near-real-time | 1 second – 1 hour | recent-click sequences, per-creator CTR |
| Hourly | 1 – 24 hours | trending topics, item popularity decay |
| Batch | 1+ days | user demographics, item embeddings, retrieval index |

The mistake is to make *everything* real-time. As Figure 4 shows, demographics and item metadata gain almost nothing from a real-time pipeline — but recent click sequences gain 2-3 AUC points. **Real-time is a budget you spend on the features that move the metric.**

---

## 2. The two-path architecture

![Real-time recommendation system: an asynchronous write-path keeps state fresh, while a synchronous read-path serves under a hard latency SLO](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/15-real-time-online/fig1_pipeline.png)

A real-time recommender decomposes cleanly into two paths:

**Write-path (asynchronous, throughput-bound).** Events leave the client, hit a Kafka topic partitioned by `user_id`, get aggregated by a Flink job into rolling windows (last-N clicks, 10-minute CTR, etc.), and land in two places: a *feature store* (Redis or RocksDB) for the read-path to consume, and an *online learner* that updates model weights. A new model snapshot is pushed to a registry every few minutes.

**Read-path (synchronous, latency-bound).** A serving request arrives. We do recall (ANN over embeddings, plus inverted indexes for fresh items), then a *single round-trip* feature fetch from the store, then ranking, then re-ranking for diversity / business rules, then return. Total budget: < 100 ms.

The discipline is keeping these two paths decoupled. The serving path **never** writes to the model, never trains, never blocks on stream processing. If the streaming side falls behind, serving keeps working with slightly stale features — degraded, not down.

---

## 3. The latency budget — where every millisecond goes

![Latency budget broken down by stage and percentile, showing p50/p95/p99 across recall, feature fetch, ranker, and re-rank](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/15-real-time-online/fig2_latency_budget.png)

A 100 ms end-to-end SLO is the industry norm for feed-style products (perception research puts the "feels instant" threshold around 100-200 ms for visual updates). Inside that budget, the split looks roughly like this:

| Stage | p50 | p95 | p99 | Notes |
|-------|-----|-----|-----|-------|
| Network in | 4 | 7 | 12 | TLS handshake amortised by keep-alive |
| Recall (ANN) | 10 | 18 | 28 | HNSW or ScaNN over 100 M items |
| Feature fetch | 6 | 14 | 30 | Redis pipeline; tail = GC / network |
| Ranker (DNN) | 18 | 32 | 55 | Batched scoring over ~500 candidates |
| Re-rank + logging | 4 | 9 | 18 | Diversity, biz rules, async log |
| Network out | 3 | 6 | 11 |  |
| **End-to-end** | **45** | **86** | **154** | p99 blows the SLO — that's normal |

Two practical lessons:

1. **Average latency lies.** A p50 of 45 ms looks like room to spare. The p99 of 154 ms means 1 % of requests miss the SLO — that's millions per day on a billion-request platform.
2. **The ranker dominates.** Batching candidates, model distillation, and TensorRT/ONNX-Runtime quantization buy more than any single optimization elsewhere. Pinterest reported a 30 % p99 reduction by moving from a 6-layer DNN to a distilled 2-layer student plus feature crosses.

---

## 4. Streaming: Kafka + Flink in production

![Streaming reference architecture: clients write to Kafka topics, Flink performs stateful aggregation and online learning, output is fanned out to feature store, model registry, and metrics](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/15-real-time-online/fig5_streaming_arch.png)

### 4.1 Kafka — the durable transport

Kafka's role is narrow but essential: a durable, partitioned, replayable log. Three properties matter:

- **Partitioning by `user_id`** keeps a single user's events on a single partition, which preserves causal order — critical for stateful joins like "did the click come before or after the impression?".
- **Replication** (typically `replication-factor=3`) means a broker can die without data loss.
- **Retention** lets you replay the last 7 days for backfilling a new model — the same code path serves online and recovery.

What Kafka does *not* do: aggregation, joins, machine learning. It is a postal service.

```python
from kafka import KafkaProducer
import json, time

producer = KafkaProducer(
    bootstrap_servers=["kafka-1:9092", "kafka-2:9092", "kafka-3:9092"],
    value_serializer=lambda v: json.dumps(v).encode(),
    key_serializer=lambda k: k.encode(),
    acks="all",        # wait for all in-sync replicas
    enable_idempotence=True,  # exactly-once producer semantics
    compression_type="lz4",
    linger_ms=5,       # tiny batching window — trades 5ms latency for 10x throughput
)

def emit_click(user_id: str, item_id: str, position: int) -> None:
    """Emit one click event. Same partition for the same user."""
    producer.send(
        topic="clicks",
        key=user_id,                # partition key
        value={
            "user_id": user_id,
            "item_id": item_id,
            "position": position,
            "ts": int(time.time() * 1000),
        },
    )
```

### 4.2 Flink — the stateful compute

Where Kafka is a transport, Flink is a *stateful* stream processor. The killer feature is **exactly-once processing under failure**, achieved by Chandy-Lamport-style distributed snapshots: every checkpoint interval (default 60 s), Flink atomically captures the state of every operator and writes it to durable storage (S3). On failure, it rewinds Kafka offsets to the last checkpoint and replays — the externally visible effect is as if no failure occurred.

A canonical Flink job for click attribution:

```python
# Flink SQL — last-10-minute CTR per user, written to feature store
from pyflink.table import EnvironmentSettings, StreamTableEnvironment

t_env = StreamTableEnvironment.create(
    environment_settings=EnvironmentSettings.in_streaming_mode()
)

t_env.execute_sql("""
CREATE TABLE clicks (
    user_id STRING,
    item_id STRING,
    ts BIGINT,
    event_time AS TO_TIMESTAMP_LTZ(ts, 3),
    WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'clicks',
    'properties.bootstrap.servers' = 'kafka-1:9092',
    'format' = 'json',
    'scan.startup.mode' = 'latest-offset'
)
""")

t_env.execute_sql("""
CREATE TABLE user_features (
    user_id STRING,
    window_end TIMESTAMP(3),
    clicks_10m BIGINT,
    PRIMARY KEY (user_id) NOT ENFORCED
) WITH (
    'connector' = 'redis',
    'host' = 'redis.prod',
    'ttl-sec' = '900'
)
""")

# 10-minute hopping window, slide 1 minute → emit fresh CTR every minute
t_env.execute_sql("""
INSERT INTO user_features
SELECT
    user_id,
    HOP_END(event_time, INTERVAL '1' MINUTE, INTERVAL '10' MINUTE) AS window_end,
    COUNT(*) AS clicks_10m
FROM clicks
GROUP BY
    user_id,
    HOP(event_time, INTERVAL '1' MINUTE, INTERVAL '10' MINUTE)
""")
```

The watermark is the part most people get wrong. It says "I will not see any event with `event_time < watermark`." That gives Flink permission to *close* and emit a window. Set it too tight and you drop late events; set it too loose and your features are systematically delayed.

---

## 5. Online learning: from SGD to FTRL

![Online learning vs batch retraining — left: convergence on a stationary task; right: behavior under concept drift](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/15-real-time-online/fig3_online_vs_batch.png)

### 5.1 The core update

For a logistic ranker with parameters $\theta$ and a single example $(x_t, y_t)$:

$$\theta_{t+1} = \theta_t - \eta_t \, \nabla_\theta \mathcal{L}(\sigma(\theta_t^\top x_t), y_t)$$

That is plain SGD. It works, but on web-scale CTR data — millions of sparse features, each appearing in a small fraction of examples — it has two problems:

1. **No per-feature learning rate**: a feature that fires 1 in 10 000 needs bigger steps than one that fires every example.
2. **No sparsity**: weights drift away from zero on noise. A model with 10⁹ parameters that never zeros any of them is unservable.

### 5.2 AdaGrad — adaptive per-feature step

AdaGrad fixes the first problem by accumulating squared gradients per feature:

$$\theta_{t+1, i} = \theta_{t, i} - \frac{\eta}{\sqrt{G_{t, i} + \varepsilon}} g_{t, i}, \quad G_{t, i} = \sum_{s=1}^{t} g_{s, i}^2$$

A rare feature has small $G$, so it gets a big step the few times it does fire. A common feature has large $G$ and is updated cautiously.

### 5.3 FTRL-Proximal — the production workhorse

FTRL-Proximal (McMahan et al., *Ad Click Prediction: a View from the Trenches*, KDD 2013 — the paper Google published describing the algorithm running their ad system) combines AdaGrad's per-feature scaling with $L_1$ regularization that produces *exact* zeros, not just small weights. The per-coordinate update:

$$
z_{t,i} \leftarrow z_{t-1,i} + g_{t,i} - \frac{\sigma_{t,i}}{\eta} \theta_{t-1,i},
\qquad
\theta_{t,i} =
\begin{cases}
0 & \text{if } |z_{t,i}| \le \lambda_1 \\
-\frac{1}{\eta_{t,i}} \big(z_{t,i} - \mathrm{sign}(z_{t,i}) \lambda_1\big) & \text{otherwise}
\end{cases}
$$

where $\sigma_{t,i} = \frac{1}{\eta_{t,i}} - \frac{1}{\eta_{t-1,i}}$ and $\eta_{t,i} = \alpha / (\beta + \sqrt{\sum_s g_{s,i}^2})$.

The crucial property: the model is genuinely sparse. Google reported FTRL-Proximal cutting model size by an order of magnitude versus naive SGD with $L_2$ — at the same AUC.

```python
import numpy as np

class FTRLProximal:
    """FTRL-Proximal for online logistic regression.

    Reference: McMahan et al., "Ad Click Prediction: a View from the
    Trenches", KDD 2013. The actual algorithm in Google ads at the time.
    """
    def __init__(self, n_features: int, alpha=0.1, beta=1.0,
                 l1=1.0, l2=1.0):
        self.n = n_features
        self.alpha, self.beta, self.l1, self.l2 = alpha, beta, l1, l2
        # z and n are the only persisted state — w is recomputed lazily
        self.z = np.zeros(n_features)
        self.n_sq = np.zeros(n_features)

    def _weight(self, i: int) -> float:
        """Lazy materialization of w_i from (z_i, n_sq_i)."""
        z, n_sq = self.z[i], self.n_sq[i]
        if abs(z) <= self.l1:
            return 0.0
        sign = 1.0 if z > 0 else -1.0
        return -(z - sign * self.l1) / (
            (self.beta + np.sqrt(n_sq)) / self.alpha + self.l2
        )

    def predict_proba(self, x_indices: np.ndarray, x_values: np.ndarray) -> float:
        """Sparse prediction; only iterate over non-zero features."""
        s = sum(self._weight(i) * v for i, v in zip(x_indices, x_values))
        return 1.0 / (1.0 + np.exp(-max(min(s, 35.0), -35.0)))

    def update(self, x_indices: np.ndarray, x_values: np.ndarray, y: int) -> float:
        """Single-example update. Returns the pre-update probability."""
        p = self.predict_proba(x_indices, x_values)
        for i, v in zip(x_indices, x_values):
            g = (p - y) * v                       # gradient on this feature
            sigma = (np.sqrt(self.n_sq[i] + g * g) - np.sqrt(self.n_sq[i])) / self.alpha
            self.z[i]   += g - sigma * self._weight(i)
            self.n_sq[i] += g * g
        return p
```

### 5.4 Online vs batch — the picture

The left panel of Figure 3 shows the stationary case: online learning converges smoothly while batch retraining produces a staircase — a fresh snapshot every 200 events, plateaus in between. On a stable task the gap closes in expectation, but online wins on integral over time.

The right panel is where it matters: a sudden distribution shift (a viral event, a new product launch, a holiday). The batch model, blind to anything outside its training window, takes a full window to recover. Online learning starts adjusting on the very first new example and is back to near-peak AUC in roughly 100 events. **In production, drift is the rule, not the exception** — and that is what makes online learning a real lever, not just an academic preference.

---

## 6. Feature freshness — how much does it actually matter?

![Feature staleness vs AUC: AUC degrades roughly linearly in log-staleness, and the loss is almost entirely in the behavioral feature family](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/15-real-time-online/fig4_freshness_auc.png)

The freshness question is what a serious team argues about, because making something real-time is *expensive*. The empirical pattern from public reports (Meta's deep learning recommendation models, Pinterest's PinnerSAGE, ByteDance's Monolith) is consistent:

- **AUC drops roughly linearly in log-staleness.** Going from 1 second to 1 minute: tiny loss. From 1 minute to 1 hour: meaningful (~0.005 AUC). From 1 hour to 1 day: substantial (~0.015-0.020 AUC).
- **The loss is concentrated in behavioral features.** Recent click sequences and session intent are responsible for ~80 % of the freshness premium. Demographics and item metadata are essentially flat — you could update them weekly with no measurable impact.

This shapes the architecture: do not pay the streaming cost for features that don't pay back. A typical feed system runs three pipelines side by side — real-time for behavioral, hourly for popularity-style aggregates, daily for embeddings and demographics.

---

## 7. Bandits — the principled exploration story

### 7.1 The problem in one sentence

You have $K$ items to choose from. Each item has an unknown click probability $\mu_i$. Each round you pick one, observe its click, and update. Over $T$ rounds, **regret** is what you lost relative to always picking the best:

$$R_T = T \mu^* - \mathbb{E}\!\left[\sum_{t=1}^T \mu_{a_t}\right]$$

A "good" algorithm has *sublinear regret*: $R_T = o(T)$, i.e. average per-round loss → 0.

### 7.2 UCB1 — be optimistic in the face of uncertainty

UCB1 (Auer et al., 2002) picks the arm with the highest *upper confidence bound*:

$$a_t = \arg\max_i \left( \hat\mu_i + \sqrt{\frac{2 \ln t}{n_i}} \right)$$

where $n_i$ is how many times arm $i$ has been pulled. The bonus shrinks as $n_i$ grows — explore until you're sure, then exploit. UCB1 achieves $O(\log T)$ regret, which is provably optimal up to constants for stationary bandits (Lai-Robbins lower bound).

### 7.3 Thompson Sampling — the Bayesian way

Maintain a posterior over each arm's success rate. For Bernoulli rewards, the conjugate prior is Beta:

$$\theta_i \sim \text{Beta}(\alpha_i, \beta_i), \quad \text{update: } (\alpha_i, \beta_i) \leftarrow (\alpha_i + r, \beta_i + 1 - r)$$

Each round, *sample* one $\theta_i$ from each arm's posterior and pick the highest sample. Wide posteriors get explored (their samples are noisy and occasionally land high); narrow posteriors get exploited. Thompson Sampling matches UCB1's $O(\log T)$ asymptotically (Agrawal & Goyal, 2012) and beats it in practice on most benchmarks, while being simpler to extend to delayed feedback, batched updates, and complex reward models.

```python
class ThompsonSampling:
    """Bernoulli Thompson Sampling. Production-grade implementation
    is the same — the only addition is decay (alpha, beta *= 0.99 daily)
    to handle non-stationary item populations."""

    def __init__(self, n_arms: int):
        self.alpha = np.ones(n_arms)   # successes + 1
        self.beta  = np.ones(n_arms)   # failures + 1

    def select(self) -> int:
        return int(np.argmax(np.random.beta(self.alpha, self.beta)))

    def update(self, arm: int, reward: int) -> None:
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
```

### 7.4 LinUCB — context that actually matters

Plain bandits learn a global ranking. The whole point of recommendation is *personalization*. LinUCB (Li et al., *A Contextual-Bandit Approach to Personalized News Article Recommendation*, WWW 2010 — the algorithm Yahoo used for front-page personalization) assumes the expected reward is linear in a context vector $x_t$:

$$\mathbb{E}[r_a \mid x_t] = x_t^\top \theta_a$$

Each arm maintains a ridge regression model $(A_a, b_a)$ where $A_a = I + \sum X X^\top$ and $b_a = \sum r X$. The selection rule:

$$a_t = \arg\max_a \left( x_t^\top \hat\theta_a + \alpha \sqrt{x_t^\top A_a^{-1} x_t} \right), \quad \hat\theta_a = A_a^{-1} b_a$$

The bonus term $\sqrt{x_t^\top A_a^{-1} x_t}$ is the predictive standard deviation of ridge regression — large when the current context is unlike anything we've seen for this arm. Regret is $\tilde O(\sqrt{d T})$ where $d$ is the context dimension.

```python
class LinUCB:
    """LinUCB — disjoint model variant.
    Reference: Li, Chu, Langford, Schapire, WWW 2010."""

    def __init__(self, n_arms: int, n_features: int, alpha: float = 1.0):
        self.alpha = alpha
        self.A = [np.eye(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def select(self, x: np.ndarray) -> int:
        ucbs = []
        for A_a, b_a in zip(self.A, self.b):
            A_inv = np.linalg.inv(A_a)
            theta = A_inv @ b_a
            ucbs.append(x @ theta + self.alpha * np.sqrt(x @ A_inv @ x))
        return int(np.argmax(ucbs))

    def update(self, arm: int, x: np.ndarray, reward: float) -> None:
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
```

In practice you almost never use a "pure" LinUCB. The serving system runs a deep ranker; bandits live one layer above, deciding *which strategy* (creator-boost, fresh-item-boost, exploitation) to allocate to a given request. The arms are policies, not items.

---

## 8. Concept drift — detect, don't pretend it isn't there

![Concept drift detection: top — observed CTR with rolling mean and reference window; bottom — z-score detector firing on gradual and abrupt drift](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/15-real-time-online/fig6_drift_detection.png)

Online learning adapts to drift only if the learner *can* adapt — if the learning rate has decayed to nothing, it can't. Production systems explicitly detect drift and react.

### 8.1 Three classical detectors

- **Page-Hinkley test**: cumulative sum of deviations from the mean; alarm when CUSUM exceeds a threshold. Good for monotonic drift.
- **DDM (Drift Detection Method, Gama et al. 2004)**: tracks the binomial error rate $p_t$ and its standard deviation. Warning at $p_t + s_t \ge p_{\min} + 2 s_{\min}$, alarm at $\ge p_{\min} + 3 s_{\min}$.
- **ADWIN (Bifet & Gavaldà 2007)**: maintains an adaptive window; whenever the mean of two sub-windows differs by more than a Hoeffding bound, the older half is dropped. Provably bounded false-alarm rate.

The simple z-score variant in Figure 6 is what you actually deploy for a v1: keep a reference window of "known-good" performance, compute the z-score of recent rolling-mean CTR vs that reference, alarm at $|z| > 3$.

### 8.2 Reacting to drift

Detection without a response is just an alert. Practical reactions, in order of cost:

1. **Bump the learning rate** (cheap, reversible). Multiply $\eta$ by 2-5x for a fixed window.
2. **Reset bandit posteriors** (medium). Halve $\alpha, \beta$ for affected arms — keeps the prior shape but doubles uncertainty.
3. **Force a checkpoint reload** (expensive). Roll back to the last known-good snapshot, replay from Kafka, retrain on the post-drift window.

```python
class DriftAdaptiveLearner:
    def __init__(self, base_learner, base_lr=0.1, window=200, z_thresh=3.0):
        self.learner = base_learner
        self.base_lr = base_lr
        self.window = window
        self.z_thresh = z_thresh
        self.recent = []        # rolling errors
        self.ref_mean = None
        self.ref_std = None

    def update(self, x, y):
        pred = self.learner.predict(x)
        err = abs(pred - y)
        self.recent.append(err)
        if len(self.recent) > self.window:
            self.recent.pop(0)

        # Establish reference once, after first window has stabilized
        if self.ref_mean is None and len(self.recent) == self.window:
            self.ref_mean = np.mean(self.recent)
            self.ref_std = np.std(self.recent) + 1e-6

        # Detect drift
        if self.ref_mean is not None:
            z = (np.mean(self.recent[-50:]) - self.ref_mean) / self.ref_std
            if abs(z) > self.z_thresh:
                self.learner.learning_rate = self.base_lr * 4   # bump
                # Reset reference after handling — avoids permanent over-eager state
                self.recent = []
                self.ref_mean = None

        self.learner.update(x, y)
```

---

## 9. Cache vs compute — the trade-off you actually tune

![Cache vs compute: left — latency, freshness, and cost as a function of cache TTL; right — Pareto front of strategies, with hybrid winning](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/15-real-time-online/fig7_cache_vs_compute.png)

Every read-path decision lands somewhere on this trade-off:

- **Always recompute** is correct but expensive — every request hits the model, every feature is computed fresh.
- **Aggressive caching** is cheap and fast but stale — minutes-old features for a session that just changed intent.
- **The hybrid pattern** is what production systems converge to:
  - **Hot path** (top users / active sessions / recent items): compute live, no cache.
  - **Cold path** (everyone else): serve from cache with a 30-60 s TTL.
  - **Negative cache**: explicitly cache "nothing changed" responses to avoid repeated computation on idle users.

The right panel of Figure 7 shows why: a 60-second TTL captures most of the latency win at a tiny freshness cost; pure recompute spends 5x the compute for ~0.005 AUC. The "always recompute" point is on the front but rarely worth it.

---

## 10. Putting it together — a minimal but realistic system

```python
class RealtimeRanker:
    """End-to-end real-time ranker. Combines:
      - Feature store (fresh features from Flink)
      - FTRL-Proximal online learner (per-event weight update)
      - LinUCB on top of the learner's score (exploration layer)
      - Drift detector (escalates learning rate on regime change)
      - Snapshot rollback (production safety)
    """

    def __init__(self, n_features: int, n_strategies: int = 4):
        self.scorer = FTRLProximal(n_features=n_features, alpha=0.1,
                                    beta=1.0, l1=1.0, l2=1.0)
        # Bandit chooses among ranking strategies, not items
        self.bandit = LinUCB(n_arms=n_strategies, n_features=n_features, alpha=0.5)
        self.drift  = DriftDetector(z_thresh=3.0)
        self.snapshots = []  # circular buffer of (ts, scorer)

    def rank(self, ctx: np.ndarray, candidates: list[dict]) -> list[str]:
        # 1. Bandit picks the strategy
        strategy = self.bandit.select(ctx)
        # 2. Score every candidate with FTRL — sparse vectors
        scored = [
            (c["id"], self.scorer.predict_proba(c["idx"], c["val"]), c)
            for c in candidates
        ]
        # 3. Strategy modifies the scoring (e.g. fresh-boost, creator-boost)
        scored = apply_strategy(strategy, scored)
        scored.sort(key=lambda r: -r[1])
        return [r[0] for r in scored[:20]]

    def observe(self, ctx, x_idx, x_val, y, strategy):
        """One feedback loop iteration."""
        # Update online learner
        p = self.scorer.update(x_idx, x_val, y)
        # Update bandit on the strategy that was actually played
        self.bandit.update(strategy, ctx, reward=float(y))
        # Feed drift detector with calibration error
        if self.drift.update(abs(p - y)):
            self.scorer.alpha *= 4              # bump learning rate
        # Periodic safety snapshot
        if len(self.snapshots) == 0 or time.time() - self.snapshots[-1][0] > 600:
            self.snapshots.append((time.time(), copy.deepcopy(self.scorer)))
            self.snapshots = self.snapshots[-12:]  # keep 2 hours
```

A few things that look like trivia but matter in production:

- The bandit operates on *strategies*, not items. The action space of "pick one item out of 100 M" is too large for any contextual bandit to converge — but "pick one of 4 ranking heuristics" is exactly its sweet spot.
- Snapshots are a circular buffer of the *online learner state*, not the predictions. If a bad batch of training data corrupts weights, you roll back the state and replay clean events from Kafka.
- `DriftDetector` reads calibration error (`|p - y|`), not raw CTR. Calibration drift catches more failure modes than rate drift alone.

---

## Q & A

**Q: How do I A/B test an online learning system?**
Standard A/B tests assume independent observations, but online learners *learn from their treatment group*. The fix is to use **interleaved comparisons** (Chapelle et al., 2012) or to hold a separate "frozen" model in one arm and the live online model in the other, then compare *cumulative* reward, not per-request CTR.

**Q: What is the actual difference between Flink and Spark Streaming?**
Flink processes events one at a time with millisecond latency; Spark Streaming processes them in micro-batches with seconds of latency. For recommendations, Flink's lower latency *and* its more mature exactly-once state management both matter — virtually every recently-built feed system at scale uses Flink (or proprietary equivalents like Twitter's Heron, ByteDance's Aiops).

**Q: Is online learning unstable?**
It can be, and that is the failure mode you must engineer against. Three protections: (1) bound the learning rate; (2) clip gradients; (3) keep snapshots and a rollback path. Google's FTRL paper devotes a full section to "tricks of the trade" — saturation guards, per-coordinate learning rate floors, calibration loss monitoring — that exist solely to make production online learning behave.

**Q: How do bandits handle delayed reward?**
Two patterns. **Batched updates**: collect $k$ observations, update with all of them, repeat. Both UCB and Thompson Sampling have proofs that batching multiplies regret by a factor of at most $\log k$. **Optimistic counters**: when an arm is pulled, increment $n_i$ immediately and assume reward = mean (or 0 for safety) — correct it once the real reward arrives. The optimistic variant is what production systems use because it keeps the system from over-pulling a single arm during a delay window.

**Q: When should I *not* use a real-time pipeline?**
When the feature is slow-moving (demographics, long-term taste embeddings), when the cost of a stale recommendation is small (warehouse search, browsing taxonomies), or when you can't measure improvement (a freshness bump from 1 h to 1 s on a low-traffic surface is invisible in the noise). A real-time pipeline costs ~10-100x of a daily batch — earn it.

---

## Summary

- A real-time recommender is **two pipelines glued together** by a feature store and a model registry: an asynchronous write-path that keeps state fresh, and a synchronous read-path that serves under a hard SLO.
- **Latency is a tail problem.** Optimise for p99, not p50 — the ranker dominates and is where distillation and quantisation pay back.
- **Online learning is FTRL-Proximal in production.** It is the algorithm Google published for ad CTR; SGD/AdaGrad are stepping stones.
- **Streaming is Kafka + Flink.** Kafka is durable transport; Flink is stateful compute with exactly-once semantics via distributed snapshots.
- **Freshness has a price curve.** Behavioral features pay back the streaming cost; demographics and item metadata don't.
- **Bandits work above the ranker, not at the item level.** Their action space is "which policy", not "which item".
- **Drift will happen.** Detect it (z-score, ADWIN, DDM), react cheaply (bump $\eta$), keep a rollback path.
- **Cache vs compute is not a binary.** The production answer is hybrid — hot path live, cold path cached, with a 30-60 s TTL covering 95 % of traffic at a fraction of the cost.

The rule of thumb that has held across every system I have seen at scale: *make real-time only what moves AUC, and budget the rest by how often it actually changes*.

---

## Series Navigation

This is **Part 15 of 16** in the Recommendation Systems series.

**Previous**: [Part 14 -- Cross-Domain Recommendation and Cold-Start Solutions](/en/recommendation-systems-14-cross-domain-cold-start/)
**Next**: [Part 16 -- Industrial Architecture and Best Practices](/en/recommendation-systems-16-industrial-practice/)

[View all parts in the Recommendation Systems series](/categories/Recommendation-Systems/)
