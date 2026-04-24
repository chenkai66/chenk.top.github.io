---
title: "Recommendation Systems (16): Industrial Architecture and Best Practices"
date: 2024-05-17 09:00:00
tags:
  - Recommendation Systems
  - Industrial Practice
  - System Architecture
categories: Recommendation Systems
series:
  name: "Recommendation Systems"
  part: 16
  total: 16
lang: en
mathjax: true
description: "Production recommendation systems serve hundreds of millions of users with sub-100ms latency. This final article covers the industrial multi-stage pipeline (recall, coarse ranking, fine ranking, reranking), feature stores, A/B testing, model optimization, deployment, and team responsibilities -- drawing on patterns from YouTube, TikTok, Taobao, and ByteDance."
disableNunjucks: true
---
> The hardest part of a production recommendation system is not the model. It is the **system around the model**: the feature store that prevents training/serving skew, the canary deployment that catches a regression before it hits 100M users, the orchestration that meets a 100ms p95 latency budget while running four ML models in sequence. This final article describes the architecture that every major tech company has converged on -- and the trade-offs hiding inside each layer.

## What You Will Learn

- **Multi-stage pipeline** -- recall, coarse ranking, fine ranking, and reranking, with the constraints that determine each stage's design
- **Multi-channel recall** -- combining collaborative filtering, two-tower deep learning, graph traversal, and real-time behaviour signals
- **Ranking models in production** -- Wide & Deep, DeepFM, and DIN, with concrete code
- **Reranking strategies** -- diversity (MMR), business rules, and freshness boosts
- **Feature store** -- the offline + online architecture that decouples training from serving
- **A/B testing** -- consistent assignment, z-test for proportions, and how long to run
- **Performance optimisation** -- quantisation, distillation, and prediction caching
- **Deployment and monitoring** -- canary rollouts, drift detection, and auto-rollback
- **Team responsibilities** -- who owns recall, ranking, the feature store, and serving

## Prerequisites

- All previous parts of this series (especially Parts 7, 11, 15)
- Basic familiarity with distributed systems (load balancers, message queues)
- Comfortable with Python, PyTorch, and REST APIs

---

## The Industrial Recommendation Pipeline

### Architecture Overview

![Full industrial recommendation pipeline showing data, training, and serving planes with the recall, coarse ranking, fine ranking, reranking funnel](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/16-industrial-practice/fig1_industrial_pipeline.png)

Every major tech company -- Google, Amazon, Alibaba, ByteDance -- has converged on the same three-plane architecture:

1. **Data plane** generates samples and features from logs and content. This is where Hive, Spark, Flink, and Kafka live.
2. **Training plane** turns samples into models, validates them offline, and writes the result to a model registry.
3. **Serving plane** is the real-time funnel that the user actually waits on. It is the only plane with a strict latency budget.

The serving plane itself is a **funnel that progressively narrows the candidate set while increasing scoring precision**:

```
User request → Recall (10⁶ → 2K) → Coarse Rank (2K → 200) → Fine Rank (200 → 50) → Re-rank (50 → 20) → Response
```

| Stage | Input → Output | Model class | Latency budget |
|-------|---------------|-------------|----------------|
| Recall | 10⁶ → ~2,000 | Two-tower DNN, ANN, simple CF | 20-30 ms |
| Coarse ranking | ~2,000 → ~200 | Shallow DNN or XGBoost | 10-20 ms |
| Fine ranking | ~200 → ~50 | Wide & Deep, DeepFM, DIN | 30-50 ms |
| Reranking | ~50 → ~20 | Rules + lightweight ML | 10-20 ms |
| **Total** | | | **< 100 ms p95** |

Think of it as a hiring funnel: recall is a resume screen (fast, wide net), coarse ranking is the phone screen, fine ranking is the on-site interview, and reranking is the hiring committee that makes the final adjustments for diversity and team fit.

### Why a Funnel Instead of One Big Model?

The brute-force alternative -- score every item with one heavy model -- would take seconds per request. The funnel buys orders of magnitude of speed because each stage uses a model that is appropriate to its candidate count: cheap models on many items, expensive models on few. The recall stage typically spends ~5 microseconds per item; the fine ranker spends ~250 microseconds. The product makes the budget work.

### Key Design Principles

**Stateless services.** Every service must be horizontally scalable. State (user embeddings, recent behaviour) lives in Redis, KV stores, or feature stores -- never in process memory.

```python
class RankingService:
    """Stateless ranking service -- replicate freely."""

    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.feature_extractor = FeatureExtractor()

    def rank(self, user_id, candidates, context):
        features = self.feature_extractor.extract(user_id, candidates, context)
        scores = self.model.predict(features)
        return sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

**Graceful degradation.** Every component must have a fallback. If the deep recall channel times out, the system falls back to collaborative filtering. If everything fails, it serves popular items. The user must never see an empty page.

```python
class FaultTolerantRecall:
    def __init__(self, channels):
        self.channels = channels
        self.fallback = PopularItemsRecall()

    def recall(self, user_id, context):
        results = []
        for channel in self.channels:
            try:
                results.extend(channel.recall(user_id, context, timeout=20))
            except Exception as exc:
                logger.warning("channel %s failed: %s", channel.name, exc)

        if not results:
            return self.fallback.recall(user_id, context)
        return deduplicate(results)
```

**Latency budget enforcement.** Every call has an aggressive timeout, and the orchestrator enforces it. A slow recall channel does not delay the whole pipeline -- it is dropped on the floor.

---

## The Funnel in Detail

![Recall, coarse ranking, fine ranking, and reranking funnel with item counts and per-stage latency budgets](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/16-industrial-practice/fig2_funnel.png)

The funnel above shows the order-of-magnitude reduction at each stage. Two design rules are worth memorising:

- **The recall stage sets the upper bound on quality.** If a great item never enters the funnel, no amount of fine ranking can save it. This is why production systems run multiple recall channels in parallel.
- **The narrower the stage, the heavier the model can be.** Fine ranking on 200 items can afford a 100M-parameter DIN. Recall on a million items cannot afford anything heavier than two embedding lookups and a dot product.

---

## Multi-Channel Recall

A single recall strategy will always miss something. Collaborative filtering misses cold items. Content recall misses serendipitous discoveries. Real-time signals miss the user's longer-term interests. So production systems run **3-5 recall channels in parallel** and merge the results.

### Channel 1: Two-Tower Deep Recall

The two-tower architecture is the workhorse of modern recall. The user tower runs at request time; the item tower runs offline and its embeddings are loaded into an ANN index (Faiss or HNSW). At serving time, recall is a single ANN query in 5-10 ms.

```python
import torch
import torch.nn as nn


class TwoTowerRecall(nn.Module):
    """Two-tower model with offline-indexed item embeddings."""

    def __init__(self, user_dim: int, item_dim: int, hidden=(256, 128)):
        super().__init__()
        self.user_tower = self._tower(user_dim, hidden)
        self.item_tower = self._tower(item_dim, hidden)

    @staticmethod
    def _tower(in_dim, hidden):
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)]
            prev = h
        return nn.Sequential(*layers)

    def recall(self, user_features, ann_index, top_k=1000):
        with torch.no_grad():
            user_emb = self.user_tower(user_features)
        # ann_index is e.g. a Faiss IVF-PQ index of pre-computed item embeddings
        _, top_ids = ann_index.search(user_emb.cpu().numpy(), top_k)
        return top_ids
```

Two practical notes. First, the loss matters: in-batch sampled softmax with **logQ correction** for popularity bias is now standard (Yi et al., RecSys 2019, the YouTube paper). Second, the item index needs to be rebuilt when item embeddings drift -- typically hourly for fast-moving catalogues, daily otherwise.

### Channel 2: Graph-Based Recall

Graph recall finds items through multi-hop traversal: user A liked items X and Y; user B liked Y and Z; therefore Z is a candidate for A. This catches discoveries that pure embedding similarity misses.

```python
from collections import defaultdict


class GraphRecall:
    """Item-to-item recall via Jaccard similarity on shared users."""

    def __init__(self, interaction_graph):
        self.graph = interaction_graph
        self.item_similarity = self._compute_similarity()

    def _compute_similarity(self):
        sim = defaultdict(dict)
        items = [n for n in self.graph.nodes() if self.graph.nodes[n]["type"] == "item"]
        for a in items:
            users_a = set(self.graph.neighbors(a))
            for b in items:
                if a == b:
                    continue
                users_b = set(self.graph.neighbors(b))
                inter, union = len(users_a & users_b), len(users_a | users_b)
                if union:
                    sim[a][b] = inter / union
        return sim

    def recall(self, user_id, top_k=1000):
        seen = [n for n in self.graph.neighbors(user_id)
                if self.graph.nodes[n]["type"] == "item"]
        scores = defaultdict(float)
        for item in seen:
            for neighbour, w in self.item_similarity.get(item, {}).items():
                scores[neighbour] += w
        return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]
```

### Channel 3: Real-Time Behaviour Recall

A recall channel that captures what the user is doing **right now**. If a user just clicked three items in the same category, the next recommendation should reflect that within seconds, not days.

```python
from collections import defaultdict, deque
from datetime import datetime, timedelta

import numpy as np


class RealTimeBehaviorRecall:
    """Recall from the last 30 minutes of activity, with exponential decay
    on age and weighted by action type."""

    def __init__(self, window_minutes=30):
        self.window = timedelta(minutes=window_minutes)
        self.behaviours = defaultdict(deque)

    def add_behavior(self, user_id, item_id, action_type, ts):
        self.behaviours[user_id].append({"item": item_id, "action": action_type, "ts": ts})
        cutoff = ts - self.window
        while self.behaviours[user_id] and self.behaviours[user_id][0]["ts"] < cutoff:
            self.behaviours[user_id].popleft()

    def recall(self, user_id, top_k=500):
        weights = {"view": 1.0, "click": 2.0, "purchase": 5.0}
        scores = defaultdict(float)
        now = datetime.now()
        for b in self.behaviours.get(user_id, []):
            age_min = (now - b["ts"]).total_seconds() / 60
            recency = np.exp(-age_min / 10)
            scores[b["item"]] += recency * weights.get(b["action"], 1.0)
        return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]
```

### Channel Fusion

Each channel returns a ranked list. The merger uses **rank-based fusion** rather than raw scores (scores from different channels are not comparable):

```python
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


class MultiChannelRecall:
    def __init__(self, channels, weights=None):
        self.channels = channels
        self.weights = weights or {c.name: 1.0 for c in channels}

    def recall(self, user_id, context, target=2000):
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.channels)) as pool:
            futures = {pool.submit(c.recall, user_id, context): c.name for c in self.channels}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    results[name] = fut.result(timeout=25)
                except Exception as exc:
                    logger.error("channel %s failed: %s", name, exc)

        scores = defaultdict(float)
        for name, items in results.items():
            w = self.weights.get(name, 1.0)
            for rank, item in enumerate(items):
                scores[item] += w / (rank + 1)  # reciprocal rank fusion
        return [i for i, _ in sorted(scores.items(), key=lambda x: -x[1])[:target]]
```

This is **reciprocal rank fusion** -- robust to scale differences and well known from search engines. The 25 ms per-channel timeout is non-negotiable: a slow channel is dropped, never blocked on.

---

## Ranking: Coarse and Fine

### Coarse Ranking

Coarse ranking trims thousands of candidates down to hundreds with a fast, lightweight model. The point is **eliminating obviously bad candidates cheaply** -- not perfect ranking. Two patterns dominate:

- A **shallow two-tower** model whose item-side runs offline (similar to recall but with richer features).
- An **XGBoost ranker** on simple features (popularity, CTR, basic user/item stats).

```python
import xgboost as xgb


class CoarseRanker:
    def __init__(self):
        self.model = xgb.XGBRanker(
            objective="rank:pairwise",
            tree_method="hist",
            max_depth=4,
            n_estimators=50,
        )

    def fit(self, X, y, group):
        self.model.fit(X, y, group=group)

    def predict(self, X):
        return self.model.predict(X)
```

A common mistake is making coarse ranking **too good**. If its top-200 already matches what the fine ranker would choose, the fine ranker adds no value. Aim for the coarse ranker's recall@200 vs. fine ranking to be around 0.7 -- enough to filter, not enough to dominate.

### Fine Ranking: Wide & Deep, DeepFM, DIN

Fine ranking runs heavy models on the reduced candidate set. Three architectures dominate production CTR prediction.

**Wide & Deep** (Google, 2016) combines memorisation (wide linear model on cross features) with generalisation (deep MLP on embeddings):

```python
class WideDeepRanking(nn.Module):
    """Google's Wide & Deep: linear memorisation + deep generalisation."""

    def __init__(self, wide_dim, embed_dims, deep_hidden):
        super().__init__()
        self.wide = nn.Linear(wide_dim, 1)
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(vocab, dim) for name, (vocab, dim) in embed_dims.items()}
        )
        deep_in = sum(dim for _, dim in embed_dims.values())
        layers, prev = [], deep_in
        for h in deep_hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)

    def forward(self, wide_feats, sparse_ids, dense_feats):
        wide_out = self.wide(wide_feats)
        emb = [self.embeddings[name](ids) for name, ids in sparse_ids.items()]
        deep_out = self.deep(torch.cat(emb + [dense_feats], dim=1))
        return wide_out + deep_out
```

**DeepFM** (Huawei, 2017) replaces the hand-crafted wide cross features with a factorisation machine that learns pairwise interactions automatically. This is the right default if you do not want to hand-curate cross features.

**DIN -- Deep Interest Network** (Alibaba, 2018) adds an attention mechanism over the user's behaviour sequence. Instead of averaging the embeddings of all past items, DIN attends to the past items most similar to the current candidate:

```python
class DIN(nn.Module):
    """Deep Interest Network: attention over user behaviour history."""

    def __init__(self, item_dim, user_dim, hidden=(128, 64)):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(item_dim * 4, 36), nn.ReLU(), nn.Linear(36, 1)
        )
        in_dim = item_dim + user_dim + item_dim
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, candidate, history, user_profile):
        cand_exp = candidate.unsqueeze(1).expand_as(history)
        attn_in = torch.cat(
            [history, cand_exp, history - cand_exp, history * cand_exp], dim=2
        )
        weights = torch.softmax(self.attention(attn_in).squeeze(-1), dim=1)
        weighted = (history * weights.unsqueeze(-1)).sum(dim=1)
        return self.mlp(torch.cat([candidate, weighted, user_profile], dim=1))
```

The attention trick matters: a user who has bought 50 books in 5 categories does not have a single "average interest" -- they have category-specific interests, and DIN unlocks them per candidate.

---

## Reranking

Reranking is where business logic meets algorithmic output. Three patterns appear in almost every production system.

### Diversity (MMR)

Pure CTR optimisation produces a list that all looks the same -- the user clicks the first item, then drops off. **Maximal Marginal Relevance** greedily picks items that balance relevance with novelty against already-selected items:

```python
import numpy as np


class DiversityReranker:
    def __init__(self, lambda_div=0.3):
        self.lam = lambda_div

    def rerank(self, items, scores, features, top_k=20):
        selected, remaining = [], list(zip(items, scores, features))
        while len(selected) < top_k and remaining:
            best_idx, best_obj = None, -np.inf
            for idx, (it, sc, ft) in enumerate(remaining):
                if selected:
                    diversity = min(self._dist(ft, s_ft) for _, _, s_ft in selected)
                else:
                    diversity = 1.0
                obj = (1 - self.lam) * sc + self.lam * diversity
                if obj > best_obj:
                    best_obj, best_idx = obj, idx
            selected.append(remaining.pop(best_idx))
        return [it for it, _, _ in selected]

    @staticmethod
    def _dist(a, b):
        va, vb = np.array(list(a.values())), np.array(list(b.values()))
        return 1 - va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
```

The diversity weight (typically 0.2-0.3) is itself an A/B test parameter. Too low and the feed becomes monotone; too high and CTR drops because relevance is sacrificed.

### Business Rules

Hard constraints live here, not in the ML model. Out-of-stock filtering, regulatory compliance, promoted-item boosting -- these are deterministic rules, easier to reason about as code than as features.

```python
class BusinessRulesReranker:
    def __init__(self, rules):
        self.rules = rules

    def rerank(self, items, scores, metadata):
        out = []
        for item, score in zip(items, scores):
            meta, adj, ok = metadata.get(item, {}), score, True
            for rule in self.rules:
                if not rule.check(item, meta):
                    ok = False
                    break
                adj += rule.score_adjustment(item, meta)
            if ok:
                out.append((item, adj))
        out.sort(key=lambda x: -x[1])
        return [i for i, _ in out]
```

### Freshness Boost

For news, video, and short-form content, recency is a feature in itself. An exponential decay gives recent items a bounded boost without dominating the list:

```python
from datetime import datetime
import numpy as np


class FreshnessReranker:
    def __init__(self, decay_hours=24, max_boost=0.3):
        self.decay, self.max_boost = decay_hours, max_boost

    def rerank(self, items, scores, timestamps):
        now, out = datetime.now(), []
        for item, score in zip(items, scores):
            ts = timestamps.get(item)
            if ts:
                age_h = (now - ts).total_seconds() / 3600
                boost = self.max_boost * np.exp(-age_h / self.decay)
                out.append((item, score * (1 + boost)))
            else:
                out.append((item, score))
        out.sort(key=lambda x: -x[1])
        return [i for i, _ in out]
```

---

## The Feature Store

![Feature store architecture showing offline batch path and online realtime path sharing a single feature definition](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/16-industrial-practice/fig3_feature_store.png)

The feature store is the single most important piece of infrastructure in a mature recommendation system, and the one most often built last. Its job is to **eliminate training/serving skew**: the guarantee that a feature computed offline for training has exactly the same definition as the feature computed online at serving time.

The architecture has two paths sharing one feature definition:

- **Offline path** runs Spark/Flink jobs over the data lake, materialises features into Parquet, and feeds the training pipeline.
- **Online path** consumes Kafka events with Flink, writes aggregated features to Redis, and serves them at p99 < 5 ms.

Both paths execute the same feature definition (typically a SQL or YAML spec). When a feature is changed, both pipelines change together. Without this discipline you will, eventually, train a model on a feature that means one thing offline and a different thing online -- and the AUC drop will be silent and brutal.

```python
import json


class FeatureStore:
    """Redis-backed online store with batch retrieval."""

    def __init__(self, redis_client, ttl=3600):
        self.redis, self.ttl = redis_client, ttl

    def set(self, entity, eid, name, value):
        self.redis.setex(f"{entity}:{eid}:{name}", self.ttl, json.dumps(value))

    def batch_get(self, entity, eids, names):
        keys = [f"{entity}:{e}:{n}" for e in eids for n in names]
        values = self.redis.mget(keys)
        n = len(names)
        return [
            [json.loads(v) if v else None for v in values[i:i + n]]
            for i in range(0, len(values), n)
        ]
```

Open-source feature stores worth knowing: **Feast** (most popular open source), **Tecton** (commercial), and Alibaba's internal **Feathr**. They all follow the same offline+online pattern.

### Cyclical Encoding for Time Features

A small but important detail. Hour 23 and hour 0 are adjacent in time, but a linear model treats them as 23 units apart. Encode them as `(sin, cos)` so the model sees them as neighbours:

```python
def temporal_features(ts):
    dt = datetime.fromtimestamp(ts)
    return {
        "hour_sin": np.sin(2 * np.pi * dt.hour / 24),
        "hour_cos": np.cos(2 * np.pi * dt.hour / 24),
        "dow_sin":  np.sin(2 * np.pi * dt.weekday() / 7),
        "dow_cos":  np.cos(2 * np.pi * dt.weekday() / 7),
        "is_weekend": int(dt.weekday() >= 5),
    }
```

---

## A/B Testing Framework

![A/B test results showing 14-day CTR trends for control vs treatment, with confidence bands and significance marker, plus per-metric lift breakdown](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/16-industrial-practice/fig4_ab_test_results.png)

A/B testing is how you discover that the model that beat baseline by 3% offline actually loses 1.5% online -- which happens more often than people admit. Three properties matter:

- **Consistent hashing** for assignment, so a user always sees the same variant. Flickering between variants destroys both UX and statistical validity.
- **Pre-registered metrics**, including guardrail metrics (latency, error rate, revenue) that block a launch even if the primary metric wins.
- **Power analysis up front** to know how many samples you need before the experiment starts. Stopping early because the result "looks good" inflates false positives dramatically.

```python
import numpy as np
from collections import defaultdict
from scipy.stats import norm


class ABTestFramework:
    """Production A/B framework with consistent assignment and z-test."""

    def __init__(self):
        self.experiments, self.events = {}, defaultdict(list)

    def create(self, exp_id, variants_split):  # e.g. {"control": 50, "v1": 50}
        self.experiments[exp_id] = variants_split

    def assign(self, user_id, exp_id):
        h = hash(f"{user_id}:{exp_id}") % 100
        cum = 0
        for variant, split in self.experiments[exp_id].items():
            cum += split
            if h < cum:
                return variant
        return "control"

    def log(self, user_id, exp_id, variant, event):
        self.events[exp_id].append({"user": user_id, "variant": variant, "event": event})

    def analyse(self, exp_id):
        stats = defaultdict(lambda: {"impr": 0, "click": 0})
        for ev in self.events[exp_id]:
            if ev["event"] == "impression":
                stats[ev["variant"]]["impr"] += 1
            elif ev["event"] == "click":
                stats[ev["variant"]]["click"] += 1

        out, ctrl = {}, stats.get("control")
        for v, s in stats.items():
            ctr = s["click"] / max(s["impr"], 1)
            out[v] = {"ctr": ctr, **s}

        if ctrl:
            for v, s in out.items():
                if v == "control":
                    continue
                p_pool = (ctrl["click"] + s["click"]) / (ctrl["impr"] + s["impr"])
                se = np.sqrt(p_pool * (1 - p_pool) * (1 / ctrl["impr"] + 1 / s["impr"]))
                z = (s["ctr"] - out["control"]["ctr"]) / (se + 1e-8)
                s["lift"] = (s["ctr"] - out["control"]["ctr"]) / (out["control"]["ctr"] + 1e-8)
                s["p_value"] = 2 * (1 - norm.cdf(abs(z)))
                s["significant"] = s["p_value"] < 0.05
        return out
```

**How long should an A/B test run?** Long enough to: (1) capture at least one weekly cycle (usually 2 weeks), (2) reach the sample size given by power analysis, and (3) outlast novelty effects (users sometimes click new things just because they are new). Two weeks is the modal answer. Anything less than one week is suspect.

**Common pitfalls.** Network effects between variants (treatment users influencing control users via shared content); SUTVA violations; heterogeneous treatment effects across user segments; cumulative effects (the treatment helps long-term retention but hurts short-term CTR). The cure for most of these is layered experimentation infrastructure -- which is why Google, Facebook and ByteDance all built their own.

---

## Continuous Training

![Continuous training pipeline with retrain triggers, deployment stages, and monitoring feedback loop](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/16-industrial-practice/fig5_training_pipeline.png)

Models decay. User behaviour drifts, the catalogue changes, seasonality shifts. A model that was state-of-the-art last month will be a liability next month if it is not retrained. The training pipeline must run automatically, triggered by:

- **Schedule** -- daily for fine ranking, hourly for incremental updates, near-real-time for online learning on the most volatile features.
- **Drift detection** -- PSI (Population Stability Index) > 0.2 on an important feature triggers a retrain even if the schedule has not fired.
- **Metric decay** -- offline AUC drops by more than 2% between checkpoints.
- **Code change** -- new feature definition or model architecture.

The output of training is not a deployed model. It is an **artifact in a model registry** with metadata: version, training data window, offline metrics, lineage. Deployment is a separate, gated step.

### Deployment: Shadow → Canary → A/B → Full Rollout

A new model never goes from registry to 100% traffic in one step. The standard staircase:

1. **Shadow** -- 0% traffic, but the model runs in parallel and predictions are logged. This catches latency regressions, schema mismatches, and serving bugs without risking users.
2. **Canary** -- 1-10% traffic for 1-24 hours. Auto-rollback if guardrail metrics breach.
3. **A/B test** -- 50% traffic for 1-2 weeks for proper statistical validation.
4. **Full rollout** -- 100% traffic.

Auto-rollback is non-negotiable. The criteria are blunt: if p95 latency exceeds the SLO, error rate exceeds 1%, or CTR drops more than 5%, roll back automatically and page a human.

```python
class CanaryDeployment:
    def deploy(self, version, initial_pct=10):
        if not self.validate_offline(version):
            raise ValueError("offline validation failed")
        canary = self.deploy_canary(version, traffic=initial_pct)
        metrics = self.monitor(canary, minutes=60)
        if self.healthy(metrics):
            self.rollout(canary, target=100)
        else:
            self.rollback(canary)
            self.alert("canary failed", metrics)

    @staticmethod
    def healthy(m):
        return (m["p95_latency"] < 100
                and m["error_rate"] < 0.01
                and m.get("ctr_delta", 0) > -0.05)
```

---

## Serving Infrastructure

![Serving infrastructure showing API gateway, load balancer, recommendation orchestrator, and downstream recall, ranking, feature, and cache services](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/16-industrial-practice/fig6_serving_infra.png)

The serving stack has four layers, all stateless and horizontally scaled:

- **API gateway + load balancer** (Nginx, Envoy, or a cloud LB). Handles TLS, auth, rate limiting, and routing.
- **Recommendation orchestrator** -- the stateless service that runs the funnel. It calls recall, ranking, and reranking in sequence and merges the results.
- **Backend services** for each stage: a recall service backed by Faiss/HNSW; a ranking service running TensorFlow Serving or Triton on GPU; a feature service backed by Redis (hot) and HBase (cold).
- **Caches** at multiple levels: feature cache, embedding cache, and full-prediction cache. Hit rates of 30-50% on the prediction cache are typical and cut compute cost roughly proportionally.

### Performance Optimisation

Three techniques compound to give 5-10x speedups without measurable quality loss:

**Quantisation** (INT8 from FP32) gives 2-4x inference speedup on CPU and modest GPU gains.

```python
import torch.quantization as quant

def quantize(model, calibration_loader):
    model.eval()
    model.qconfig = quant.get_default_qconfig("fbgemm")
    quant.prepare(model, inplace=True)
    with torch.no_grad():
        for batch in calibration_loader:
            model(batch)
    return quant.convert(model, inplace=False)
```

**Knowledge distillation** trains a small student model to mimic a large teacher. The student learns from soft probabilities (not just hard labels), which carry information about relative item quality.

```python
import torch.nn.functional as F

def distill_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.7):
    s_soft = F.log_softmax(student_logits / T, dim=1)
    t_soft = F.softmax(teacher_logits / T, dim=1)
    soft = F.kl_div(s_soft, t_soft, reduction="batchmean") * (T ** 2)
    hard = F.cross_entropy(student_logits, labels)
    return alpha * soft + (1 - alpha) * hard
```

**Prediction caching** for the long tail of repeated requests. A 5-minute TTL is a good default -- long enough to amortise compute, short enough to reflect new behaviour.

The standard recipe is: distill first, then prune, then quantise. In that order each step preserves the quality gains of the previous one.

### Monitoring

Three categories of metrics, all alerting:

```python
class RecommendationMonitor:
    def __init__(self, metrics):
        self.metrics = metrics

    def log(self, user_id, scores, latency_ms):
        self.metrics.histogram("pred.latency", latency_ms)
        self.metrics.histogram("pred.score_mean", float(np.mean(scores)))

    def check(self):
        recent = self.metrics.recent("pred.latency", minutes=5)
        if recent["p95"] > 150:
            self.alert("high_latency", recent)
        if self.metrics.error_rate(minutes=5) > 0.05:
            self.alert("high_error_rate")
        scores = self.metrics.recent("pred.score_mean", minutes=30)
        baseline = self.metrics.baseline("pred.score_mean")
        if abs(scores["mean"] - baseline["mean"]) > 2 * baseline["std"]:
            self.alert("distribution_shift", scores)
```

The most subtle alert is the third one -- **prediction distribution shift**. If the average predicted CTR jumps by two standard deviations, something is broken upstream: a corrupted feature, a stale embedding index, or a model that is silently serving the wrong version. By the time business metrics move, you have lost an hour. Distribution monitoring catches it in minutes.

---

## Team Responsibilities

![Roles in a production recommendation team and primary owners by stage](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/recommendation-systems/16-industrial-practice/fig7_org_roles.png)

A production recommendation system is too big for one person and too coupled for fully independent teams. The role boundaries that work in practice:

- **Algorithm engineers** own model code, feature design, and A/B experiments. They write the recall, ranking, and reranking models.
- **Data engineers** own ETL pipelines, sample generation, and the offline side of the feature store. They are the firewall against data quality bugs.
- **MLOps / platform engineers** own training infrastructure, the model registry, CI/CD, and the serving runtime. They make it possible to ship a new model in a day rather than a month.
- **SRE / infra** own latency SLOs, capacity planning, and incident response. They are the ones paged at 3 a.m.
- **Analysts / research** own long-horizon evaluation, causal inference, and ranking diagnostics. They catch the metrics-look-good-but-revenue-is-flat problem.
- **Product** owns business KPIs and content policy.

The matrix on the right side of the figure shows primary owners by pipeline stage. The pattern: every stage has at least two owners, because every stage has both a model-quality dimension and an operational dimension.

---

## Industrial Frameworks

**Alibaba EasyRec** (open source). End-to-end framework with feature engineering, pre-built models (Wide & Deep, DeepFM, DIN, MMoE), training on PAI/MaxCompute, and PAI-EAS for serving. The fastest path to a production-quality baseline if you are on Alibaba Cloud.

**Meta's Looper / TorchRec.** TorchRec is the open-source library powering Meta's internal recommendation stack. Strong support for sharded embedding tables, which is the hard distributed-systems problem of recommendation training.

**ByteDance Monolith** (open source). Designed for online learning at billion-parameter scale. Built around collisionless embedding hash tables and asynchronous training that updates the model from production logs in near-real-time. Powers parts of TikTok's recommendation stack.

**YouTube's two-stage system** is described in the classic 2016 paper (Covington et al.) -- a two-tower deep candidate generator plus a deep ranker. The architecture has evolved since (the 2019 sampled-softmax paper is the most influential follow-up), but the two-stage skeleton remains the template most teams copy.

---

## Q&A: Common Questions

**Q: How many recall channels should we use?**

Start with three: collaborative filtering, two-tower deep, and real-time behaviour. Add specialised channels (graph, content, geo, social) only when an offline gap analysis shows they would catch items the existing channels miss. Beyond ten channels you spend more on plumbing than on quality.

**Q: What is the right coarse-to-fine reduction ratio?**

Typically 10:1 (2,000 → 200 → 20). Monitor recall@K end-to-end: too aggressive a coarse stage drops good candidates that the fine ranker would have surfaced; too lax wastes the fine ranker's compute budget.

**Q: How complex should the fine ranker be?**

Start with Wide & Deep or DeepFM. Add DIN-style attention over user history only after you have measured that the existing model under-uses sequential information (look for users with rich history but flat predictions). Each step up in complexity needs to justify its serving cost.

**Q: Model-based or rule-based reranking?**

Hybrid. Hard constraints (compliance, stock, blocklists) belong in deterministic rules where you can audit them. Soft optimisation (diversity, freshness, exploration) is where learned rerankers add value. Mixing them is normal.

**Q: How to choose between quantisation, pruning, and distillation?**

Quantisation gives the biggest speed-up per unit of effort (2-4x on CPU). Distillation is the right tool when you also need a smaller model footprint, not just a faster one. Pruning is the most fragile -- it works but needs careful retraining. The recommended sequence is distill → prune → quantise.

**Q: How do you handle new users?**

Multi-stage fallback: (1) popular and trending items for truly new users; (2) demographics-based content recommendations once minimal profile is known; (3) bandit-style exploration to bootstrap signal in 3-10 interactions; (4) the full personalised model after ~50 interactions. See Part 14 for the meta-learning angle.

**Q: How do you decide to retire a model?**

A model is retired when (a) a successor wins an A/B test on the primary metric *and* does not regress any guardrail, *and* (b) the operational cost of the new model is acceptable. Always keep the previous model deployable for 30 days in case of a delayed regression.

---

## Summary

This article assembled the complete industrial recommendation stack:

- **Three planes** -- data, training, serving -- with clear interfaces between them
- **A four-stage funnel** -- recall, coarse rank, fine rank, rerank -- that fits hundreds of millions of users into a 100ms budget
- **Multi-channel recall** with reciprocal rank fusion, because no single channel covers all of quality
- **Wide & Deep, DeepFM, and DIN** as the production-grade ranking architectures
- **A feature store** that eliminates training/serving skew by sharing one feature definition between offline and online paths
- **A/B testing** with consistent hashing, z-tests, and pre-registered guardrails
- **Continuous training** triggered by schedule, drift, and metric decay
- **Canary deployment** with auto-rollback on latency, error rate, and CTR
- **Serving infrastructure** -- gateway, orchestrator, GPU model servers, Redis feature store, prediction cache
- **Team responsibilities** clearly mapped to pipeline stages

The single most valuable lesson from industrial practice is also the simplest: **start small, measure everything, and let A/B tests decide what stays.** A pipeline with popular items, simple two-tower recall, a DeepFM ranker, and disciplined experimentation will beat an exotic GNN that is launched without an A/B framework. The frameworks in this article are not the source of competitive advantage -- the loops they enable are.

---

## Series Navigation

This is **Part 16 of 16** -- the final article in the Recommendation Systems series. Thank you for reading.

**Previous**: [Part 15 -- Real-Time Recommendation and Online Learning](/en/recommendation-systems-15-real-time-online/)

[View all parts in the Recommendation Systems series](/categories/Recommendation-Systems/)

---

## References

- Covington, P., Adams, J., Sargin, E. "Deep Neural Networks for YouTube Recommendations." RecSys 2016. [paper](https://research.google/pubs/pub45530/)
- Yi, X., et al. "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations." RecSys 2019. (the "two-tower with logQ correction" paper)
- Cheng, H., et al. "Wide & Deep Learning for Recommender Systems." DLRS 2016. [arXiv:1606.07792](https://arxiv.org/abs/1606.07792)
- Guo, H., et al. "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction." IJCAI 2017. [arXiv:1703.04247](https://arxiv.org/abs/1703.04247)
- Zhou, G., et al. "Deep Interest Network for Click-Through Rate Prediction." KDD 2018. [arXiv:1706.06978](https://arxiv.org/abs/1706.06978)
- Liu, Z., et al. "Monolith: Real Time Recommendation System With Collisionless Embedding Table." 2022. [arXiv:2209.07663](https://arxiv.org/abs/2209.07663)
- Alibaba EasyRec: [github.com/alibaba/EasyRec](https://github.com/alibaba/EasyRec)
- TorchRec: [github.com/pytorch/torchrec](https://github.com/pytorch/torchrec)
- Feast (open-source feature store): [feast.dev](https://feast.dev)
