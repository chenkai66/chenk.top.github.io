---
title: "Aliyun PAI (4): PAI-EAS — Model Serving, Cold Starts, and the TPS Lie"
date: 2026-03-08 09:00:00
tags:
  - Aliyun PAI
  - PAI-EAS
  - Model Serving
  - Inference
  - Auto Scaling
categories: Aliyun PAI
lang: en
mathjax: false
series: aliyun-pai
series_title: "Aliyun PAI Practical Guide"
series_order: 4
description: "End-to-end PAI-EAS for production: image-based deploy from OSS-mounted weights, the three inference modes, an autoscaler that doesn't blow your budget, and canary releases via service groups. Includes a working vLLM Qwen3 deployment from the official Quick Start."
disableNunjucks: true
translationKey: "aliyun-pai-4"
---

EAS is where the money goes. DSW costs a few hundred RMB a month for development. DLC costs spike. EAS bills 24/7 because someone might call your endpoint, and the "minimum replica count" in the autoscaler config is the most critical setting in the entire platform. This article covers what I wish I'd known before shipping our first production endpoint.

![Aliyun PAI (4): PAI-EAS — Model Serving, Cold Starts, and the TPS Lie — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/illustration_1.png)

---

## What EAS is, per the docs

The official "EAS overview" describes it as: "deploy trained models as online inference services or AI web applications, with heterogeneous resources, automatic scaling, one-click stress testing, canary releases, and real-time monitoring". The two key points are:

- It's a **container-runtime serving layer** — your model is in OSS, your code is in a container image, EAS pulls the image, mounts OSS at startup, runs your start command, and listens on a port.
- It's **autoscaled by replica count** — not a serverless function model (with one important exception, see below). Replicas are real GPU pods that take 30-120 seconds to start. Plan accordingly.

## The request path

![EAS request path](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/fig1_eas_request_path.png)

The four components the docs highlight for runtime-image deployment:

1. **Runtime image** — read-only template with OS, CUDA, Python, deps. Use an official one (`vllm:0.11.2-mows0.5.1`, `pytorch:...`) or push your own to ACR.
2. **Code and model** — *not in the image*. They live in OSS/NAS. Decoupling them allows you to update weights without rebuilding the image.
3. **Storage mounting** — at startup, EAS FUSE-mounts the OSS path you specified to a directory inside the container, e.g. `/mnt/data/`.
4. **Run command** — the first command after the container starts. Typically launches your HTTP server (`vllm serve /mnt/data/Qwen/Qwen3-0.6B`).

> **Real-world tip:** Bake `/mnt/data/` into your code paths from day one. Do not let model paths get hardcoded to `/workspace/models/`. Switching from local-dev to EAS becomes a one-line config change instead of a code refactor.

## Three inference modes

The docs list three. Choose carefully — the wrong mode wastes either money or latency.

![EAS inference modes](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/fig2_eas_inference_modes.png)

A practical guideline:

- **Real-time sync** — chatbots, RAG retrieval, ad ranking, search. You care about p99 latency.
- **Async** — anything that takes 5+ seconds: image-gen, video-gen, OCR-on-PDF batches. The built-in queue scales replicas based on backlog, which is the right approach for these workloads.
- **Batch** — anything you can wait minutes for: nightly embeddings, voice transcription. Use preemptible instances and halve the cost.

## The Quick Start, in real config

The official Quick Start deploys Qwen3-0.6B with vLLM. The console process is:

1. **Method:** Image-based deployment.
2. **Image:** `vllm:0.11.2-mows0.5.1` (official EAS image — vLLM ≥ 0.8.5 is required for OpenAI-compatible chat).
3. **Model:** OSS, `oss://your-bucket/models/`, mount path `/mnt/data/`.
4. **Command:** `vllm serve /mnt/data/Qwen/Qwen3-0___6B`.
5. **Resource:** `ecs.gn7i-c16g1.4xlarge` (1 × A10).
6. **Click Deploy.** ~5 minutes to `Running`.

You then get an OpenAI-compatible endpoint at the URL the console provides. Call it:

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["EAS_TOKEN"],          # the token from "View Call Information"
    base_url="https://YOUR-ENDPOINT.cn-shanghai.pai-eas.aliyuncs.com/v1",
)

resp = client.chat.completions.create(
    model="Qwen3-0.6B",
    messages=[{"role": "user", "content": "What is EAS in one sentence?"}],
)
print(resp.choices[0].message.content)
```

If that returns a sentence, your endpoint is live, and you can return to your colleagues looking like a wizard.

## Auto-scaling done right

This is the part the docs don't emphasize. Default autoscaler behavior (scaling on request rate with a minimum of 1 replica) can lead to cold-start latency issues or unexpected bills.

![EAS auto-scaling — replicas track QPS](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/fig3_eas_autoscaling.png)

The three settings that matter:

- **`min_replicas`** — never set to zero in production. A cold start on a 7B vLLM container is 60-120 seconds; the user gives up at 5. I default to 2 (one for HA, one for redundancy). For asynchronous services you can do 0 and rely on the queue.
- **`max_replicas`** — the budget brake. Calculate as: `(p99_qps_per_replica) * 2`. If you don't know your per-replica QPS, run the one-click stress test. The docs cover this under "Service stress testing".
- **Scaling metric** — by default it's `qps`. For LLM serving, switch to `concurrent_requests` (or vLLM's `running` metric). QPS is misleading because long generations don't register as additional requests.

> **Real-world tip:** The single biggest wasted spend I have ever seen on PAI was a `max_replicas=50` autoscaler with `min_replicas=10` on a service that got 0.5 QPS off-peak. 5 idle A10s, 24/7, for two months. Always look at the Saturday-night dashboard before you go on holiday.

## Canary, blue/green, and traffic mirroring

EAS does this with **service groups**: a routing front-end that points to multiple service versions and splits traffic by percentage. The same primitive supports **traffic mirroring** — a copy of real traffic is sent to a candidate version, but the response is discarded so users see no impact. This is the safest way to test a new model on production traffic.

![EAS service groups — canary and mirror](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/fig4_eas_canary_release.png)

I use a 90/10 split for the first 24 hours of any model swap, then 50/50, and finally 0/100. If any step shows degradation in success rate or p99 metrics, rollback is immediate — service groups change traffic weights in seconds.

## Stress testing — actually do this

The docs have a whole section on the one-click stress tester. Use it. It auto-ramps QPS, charts replica scale-out, and tells you the per-replica saturation point. That number is what you build your autoscaler around. Deploying without one is the most common cause of "the model fell over at the 3pm peak" tickets.

## The 180-day gotcha

Buried in the docs: "If an EAS service remains in a non-Running state for 180 consecutive days, the system automatically deletes the service." Set a calendar reminder. I lost a service config once because the team that owned it dissolved and no one paid the bill. Restoring took an afternoon of re-bisecting which `vllm` version was on which weights.

## Cold start mitigation, in order of effectiveness

![Aliyun PAI (4): PAI-EAS — Model Serving, Cold Starts, and the TPS Lie — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/04-pai-eas-model-serving/illustration_2.png)

Cold start is EAS's biggest practical issue. A vLLM Qwen3-7B container takes 60-120 seconds from scheduler pick to first token served — model load alone takes 30-60 seconds. If your autoscaler needs to add a replica under load, the first user requests in that window will time out.

The mitigations I've actually shipped, ranked by impact:

**1. Pre-built container with weights baked in (saves 30-60 s).** The definitionault flow downloads the model from OSS at container start. Bake the model into the image instead — 14 GB for Qwen3-7B added to the layer is fine because EAS caches images per-node. First start on a fresh node is the same; second and subsequent starts on the same node skip the OSS pull entirely. Trade-off: image rebuild on every weight change (a Dockerfile + CI job, maybe 10 min build, 5 min push).

**2. Warmup pings (saves 5-15 s of CUDA / kernel init).** A vLLM container is "running" the moment the HTTP server is up, but the first real request triggers JIT compilation of CUDA kernels for that batch shape. Pre-warm with a synthetic request:

```python
# In your container start script, after vllm serve is healthy:
import requests, time
def warmup():
    for _ in range(3):
        requests.post("http://localhost:8000/v1/chat/completions", json={
            "model": "Qwen3-7B",
            "messages": [{"role": "user", "content": "warmup"}],
            "max_tokens": 4,
        })
    print("warmup done")

# Run once before EAS health check passes
threading.Thread(target=warmup, daemon=True).start()
```

EAS won't route real traffic to the replica until the health check returns 200, so structure your health check to return only after warmup completes. This adds 5-15 s to the *visible* cold start but eliminates the 5-15 s latency penalty on the first 1-3 real requests.

**3. Pre-loaded weights via shared NAS (saves 20-40 s).** Instead of downloading from OSS, mount a NAS volume that already has the weights. NAS read bandwidth is steadier than OSS-FUSE and the model load step ("loading model weights from /mnt/models/...") drops from 30-60 s to 10-20 s. Only worth it if you're managing many small replicas — for one big replica, the OSS-FUSE path is fine.

**4. Always-warm minimum replicas (saves 100% of cold start, costs replica × 24 h).** The bluntest tool. `min_replicas=2` and your first user never sees a cold start. The math at A10 prices: ~5 RMB/h × 24 × 30 = ~3600 RMB/month per always-warm replica. For a service with revenue impact above that line, obvious. For a low-traffic internal tool, painful — use the next item.

**5. Scheduled scaling (saves cold-start cost during predictable troughs).** EAS supports time-window scaling rules:

```yaml
autoscaling:
  rules:
    - name: business-hours
      cron: "0 9 * * MON-FRI"
      min_replicas: 3
      max_replicas: 20
    - name: off-hours
      cron: "0 19 * * MON-FRI"
      min_replicas: 1
      max_replicas: 5
    - name: weekends
      cron: "0 0 * * SAT,SUN"
      min_replicas: 0
      max_replicas: 3
```

I use this pattern for B2B services with predictable traffic. Saves roughly 40% of replica-hours on a typical mainland China business-hours pattern, no user-visible latency impact.

**6. Async inference mode (sidesteps the question).** For workloads tolerant of seconds of queue time (image gen, long-form generation), use the async mode. The queue scales replicas based on backlog instead of QPS, so a cold start that takes 90 s during a surge just means the user sees a 90-s queue delay instead of a 5-s timeout. Same money, very different user experience.

What I actually deploy for production LLM serving: pre-built container with weights, warmup ping in the start script, `min_replicas=2` during business hours dropping to `1` overnight, async mode for any inference that can take >3 s.

## Auto-scaling policies: CPU vs request rate vs custom metric

The default autoscaler scales on QPS. For LLM serving, that's the wrong metric, and the docs don't really explain why. A short detour through the math.

A vLLM replica processes requests with paged-attention batching. Throughput depends on the *concurrent* requests being served (more concurrent = better GPU utilization, up to the batch limit) and the *generation length* of each request (longer = each request occupies the GPU longer). QPS — requests started per second — is a poor proxy for either.

Three scaling metric options EAS exposes, and when each is right:

**`qps` (default).** Scales on request arrival rate. Right for: synchronous, fixed-cost endpoints (image classification, embedding). Wrong for: anything with variable generation length.

**`concurrent_requests`.** Scales on the number of in-flight requests at any moment. Right for: LLM chat, RAG endpoints, anything where you can specify a target concurrency per replica. The number to use: run the one-click stress test, find the concurrency level where p99 starts climbing, set the target to 70% of that.

**Custom metric (CloudMonitor).** Scales on whatever metric you publish. The two I've used:
- `vllm_running_requests_avg` — vLLM's internal "actively decoding" count, more accurate than EAS-side `concurrent_requests` because it excludes queued-but-not-yet-decoding requests.
- `gpu_memory_pct` — when KV cache pressure is the bottleneck (long-context workloads). Scale up at 75% memory utilization.

A worked example for a Qwen3-7B chat service:

```yaml
autoscaling:
  metric: concurrent_requests
  target: 12              # found via stress test: p99 spikes above 16
  min_replicas: 2
  max_replicas: 10
  scale_up_stabilization_window: 60s
  scale_down_stabilization_window: 600s   # slow scale-down
```

The asymmetric stabilization windows matter. Scale up fast (within 60 s of crossing the threshold) so you don't queue users; scale down slow (10 min of sustained low load before removing a replica) so you don't flap during traffic dips. The default is symmetric and produces too much churn.

**The metric I do not recommend: CPU.** EAS supports `cpu` as a metric, but vLLM is GPU-bound and CPU sits at 5-15% regardless of load. Scaling on CPU will either never trigger or trigger on a memory-allocation spike that has nothing to do with serving capacity.

## Blue-green deployment + traffic shaping that actually works

Service groups give you the primitives; using them right takes some discipline. The pattern I run for any model swap:

**Step 0: Deploy the candidate as a new service with `min_replicas=2`.** Same image, same hardware, same OSS path but pointing at the new weights. Don't put it in the service group yet.

**Step 1: Sanity check with private traffic.** Hit the candidate's direct endpoint (not via the service group) with a fixed eval set — 50-200 prompts you've golden-labeled. If this fails, you don't waste service-group routing churn on a bad model.

**Step 2: Mirror 5% of production traffic for 1 hour.** Mirror copies real user requests to the candidate, discards the response, lets you compare candidate's responses to baseline's offline. EAS does this with a `mirror_weight: 5` field on the service group route. Watch p99 latency, error rate, and (if you log responses) qualitative diff against baseline.

```python
# Service group config (illustrative):
service_group = {
    "name": "qwen-chat-prod",
    "routes": [
        {"service": "qwen3-7b-v23", "weight": 100, "mirror_weight": 0},
        {"service": "qwen3-7b-v24", "weight": 0,   "mirror_weight": 5},
    ],
}
```

**Step 3: Live shift, 5% / 25% / 50% / 100% over 24 h.** Each step holds for at least 1 h with monitoring alerts on success rate, p99, and a per-route qualitative check. If anything wobbles, drop weight back to 0 — service groups update in <10 s.

**Step 4: Decommission the old service.** Don't delete it for at least 48 h after 100% migration. If you need to roll back at hour 36, "set weight back to 100 on the old service" is the fastest possible recovery — much faster than re-deploying from OSS.

The traffic-shaping primitive can also do more interesting splits: route by user-agent (test on a single client first), by region (canary in cn-shanghai before cn-hangzhou), by request-header value (internal-tester vs public). All configured in the same service group routing rules. I've used the user-agent split to launch a new model to my team's Cherry Studio sessions before any external user, which catches bugs the eval set misses.

## The cost arithmetic, per inference

The single most useful spreadsheet I keep for any EAS service. Per-inference cost is dominated by replica-hours, not per-request fees. A worked example for a Qwen3-7B chat endpoint serving roughly 5 QPS during business hours:

| Component | Number | Cost/month |
|---|---|---|
| Min-replica baseline (2 × A10, 24/7) | 2 × 5 RMB/h × 720 h | ~7,200 RMB |
| Burst replicas (avg 3 extra during 9 h × 22 days) | 3 × 5 × 198 h | ~2,970 RMB |
| Per-request fee (5 QPS × 86400 × 22) | 9.5 M req × 0.0001 RMB | ~950 RMB |
| OSS bandwidth (model loads on cold start) | 14 GB × 30 cold starts × 0.5 RMB/GB | ~210 RMB |
| **Total** | | **~11,330 RMB** |

Three observations from this table that took me a while to internalize:

1. **The min-replica line dominates.** Cutting `min_replicas` from 2 to 1 in off-hours saves ~3,600 RMB/month. Going to 0 saves another ~3,600 RMB/month *but* introduces 60-120 s cold start. Pick the trade based on your SLA, not based on what feels safe.
2. **Per-request fees are negligible for LLM workloads.** They matter for high-QPS classification (1000+ QPS), where the per-million fee can dominate. For chat / generation, ignore.
3. **Cold-start bandwidth is non-trivial at scale.** 30 cold starts at 14 GB each is 420 GB of OSS read traffic — at 0.5 RMB/GB inter-region, that's a real number. Bake weights into the image (see Cold Start section above) and this line goes to near zero.

The formula I use for any new service:

```text
monthly_cost = min_replicas × replica_price_per_hour × 720
             + avg_burst_replicas × replica_price_per_hour × business_hours_month
             + total_requests × per_request_fee
             + cold_starts × model_size_gb × oss_read_price
```

Plug in numbers *before* deploying. The number of times a "small" deploy turned out to cost an unexpected $2k/month because someone defaulted `min_replicas=10` is non-zero. The console shows you a price estimate when you click Deploy — read it.

## What's next

Article 5 closes the series with the honest pitch for **Designer** and **Model Gallery** — the two zero/low-code surfaces. They are not what most engineers reach for, but they earn their keep when used right, and there is a specific set of jobs where they are obviously the correct answer.
