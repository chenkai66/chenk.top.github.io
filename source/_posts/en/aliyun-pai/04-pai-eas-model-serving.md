---
title: "Aliyun PAI (4): PAI-EAS — Model Serving, Cold Starts, and the TPS Lie"
date: 2026-03-08 09:00:00
tags:
  - Aliyun PAI
  - Machine Learning
  - PAI-EAS
  - Model Serving
  - Inference
categories: Aliyun PAI
lang: en
mathjax: false
series: aliyun-pai
series_title: "Aliyun PAI Practical Guide"
series_order: 4
description: "Deploying a fine-tuned Qwen-7B as a production EAS endpoint, the difference between Image mode and Processor mode, why your TPS dashboard is lying to you, and how to size warm pools so your p99 doesn't spike at 3am."
disableNunjucks: true
---

EAS is where the PAI bill stops being theoretical. A DSW notebook costs you a few yuan an hour while you're at your desk; a DLC job is a one-time spend; an EAS endpoint sits there 24×7 burning money whether anyone's calling it or not. The good news is EAS is also the most production-shaped piece of PAI — autoscaling, blue/green, traffic splitting, health checks, all the things you'd build yourself eventually.

This article ships a real endpoint: a fine-tuned Qwen-7B (the one we trained in article 3) served as `qwen-7b-prod`, called from a Python client. Along the way we'll cover the two deployment modes, the cold-start trap that has cost me at least one bad demo, and why the TPS metric on the dashboard does not mean what you think it means.

## What EAS actually is

PAI-EAS (Elastic Algorithm Service) is a model-serving platform. You give it a model artifact and a way to load + invoke it; EAS gives you back an HTTPS endpoint with autoscaling, metrics, and traffic routing. Under the hood it's running pods on a GPU/CPU cluster behind an internal load balancer.

Two deployment modes you have to choose between up front:

- **Image mode** — you push a Docker image with whatever HTTP server you want (FastAPI, Triton, vLLM, your hand-rolled thing). EAS just runs the container, routes traffic, scales replicas. You own everything inside the container.
- **Processor mode** — you write a Python class with `initialize()` and `process()` methods (and a YAML descriptor); EAS provides the HTTP server, the request routing, the model loading hooks. Less code, less control.

For PyTorch / Hugging Face / vLLM workloads I default to **Image mode**. Processor mode is fine for scikit-learn-shaped models and for legacy stuff. For LLMs you want vLLM or SGLang inside an Image, period.

## Deploying our Qwen-7B fine-tune

The artifact from article 3 lives at `oss://my-bucket/checkpoints/qwen25-7b-sft-001/checkpoint-final/`. We'll wrap it in a vLLM-based image and deploy as an endpoint named `qwen-7b-prod`.

### Step 1 — pick or build the image

PAI publishes a vLLM image: `pai-image-vllm:0.6-cu124-py310`. It exposes the OpenAI-compatible API at `/v1/chat/completions` if you set the entrypoint correctly. For most LLM serving you don't need a custom image at all — the prebuilt one plus the right command and a few env vars is enough.

### Step 2 — write the deployment with the SDK

```python
# deploy_qwen.py
import os
from pai.session import setup_default_session
from pai.predictor import Predictor
from pai.model import Model, InferenceSpec, container_serving_spec

setup_default_session(region_id="cn-shanghai")

inf_spec = container_serving_spec(
    image_uri="pai-image-vllm:0.6-cu124-py310",
    command=(
        "python -m vllm.entrypoints.openai.api_server "
        "--model /mnt/model "                # OSS mount path
        "--served-model-name qwen-7b-prod "
        "--port 8000 --host 0.0.0.0 "
        "--max-model-len 8192 "
        "--gpu-memory-utilization 0.9 "
        "--enable-prefix-caching"
    ),
    port=8000,
    health_check={"path": "/health", "initial_delay_seconds": 120},
)

m = Model(
    model_data="oss://my-bucket/checkpoints/qwen25-7b-sft-001/checkpoint-final/",
    inference_spec=inf_spec,
)

predictor: Predictor = m.deploy(
    service_name="qwen-7b-prod",
    instance_type="ecs.gn7e-c12g1.3xlarge",   # 1 x A100-40GB
    instance_count=1,
    options={
        "metadata.rpc.batching": True,
        "metadata.rpc.keepalive": 60000,
    },
    autoscaler={
        "min_replicas": 1,
        "max_replicas": 4,
        "metric": "QPS",
        "threshold": 8,            # scale up when each replica > 8 QPS
        "scale_in_cooldown": 600,  # don't scale down for 10 min
    },
)

print("Endpoint:", predictor.endpoint)
print("Token  :", predictor.access_token)   # KEEP SECRET
```

### Step 3 — call it from a client

EAS endpoints sit behind both an internal VPC URL and (optionally) a public URL. Both require an `Authorization` header with the access token.

```python
# client.py
import os, requests, json

URL   = os.environ["EAS_ENDPOINT"]   # e.g. https://1234567890.cn-shanghai.pai-eas.aliyuncs.com/api/predict/qwen-7b-prod
TOKEN = os.environ["EAS_TOKEN"]

resp = requests.post(
    f"{URL}/v1/chat/completions",
    headers={"Authorization": TOKEN, "Content-Type": "application/json"},
    json={
        "model": "qwen-7b-prod",
        "messages": [
            {"role": "system", "content": "You are a senior backend engineer."},
            {"role": "user",   "content": "Explain idempotency keys in two sentences."},
        ],
        "max_tokens": 256,
        "temperature": 0.2,
    },
    timeout=60,
)
print(resp.json()["choices"][0]["message"]["content"])
```

That's it. You have a production-shaped, autoscaling LLM endpoint.

## The cold-start trap

Here is the part the docs do not adequately warn you about.

When EAS scales from 1 replica to 2, the new replica has to:

1. Schedule onto a GPU node (10–60s, depends on cluster availability)
2. Pull the container image (30–120s, depends on image size and cache state)
3. Mount the OSS model directory and load the weights into GPU RAM (30–180s for a 7B, 5–15 minutes for a 70B)
4. Pass the health check
5. Start receiving traffic

Total cold-start for a 7B can be 2–5 minutes. For a 70B it can be 10–20 minutes. **During this entire window the existing replica is taking 100% of the increased traffic.** If your scale trigger fires at 8 QPS/replica and traffic is climbing fast, your single replica is now serving 24 QPS while the second replica boots, and your p99 latency is in the toilet.

Three mitigations, in order of how much I trust them:

1. **Set `min_replicas` higher than 1.** For anything customer-facing I set min ≥ 2 even if it's "wasteful" off-peak. The cost of one extra replica is much less than the cost of a 5-minute outage.
2. **Pre-warm by calling `/health` on a schedule.** Doesn't help cold-start latency itself but prevents idle GPU memory from getting evicted.
3. **Use prefix caching and persistent KV cache** (vLLM `--enable-prefix-caching`). Doesn't help cold start but cuts steady-state latency by 30–50% on chatbot-shaped traffic.

> **Real-world tip:** Test your cold-start by manually scaling to 0 and back to 1 in the console. Time it with a stopwatch. Whatever you measure, that's your worst-case "we got slammed" recovery time. Plan capacity accordingly.

## The TPS metric is a lie (or at least misleading)

The EAS dashboard shows a "TPS" / "QPS" graph. For a chat LLM endpoint, **this number tells you almost nothing useful** because:

- A request that returns 5 tokens and a request that returns 500 tokens count the same
- A request that's 100% in the prefill (long prompt, short reply) is GPU-bound; a request that's 100% in decode is bandwidth-bound; they look identical on the QPS graph
- "Average latency" is dominated by the longest replies, not the typical user experience

What to actually monitor:

| Metric | Where | Why |
|---|---|---|
| `vllm:time_to_first_token_seconds` | vLLM Prometheus exporter | This is what your user *feels* |
| `vllm:time_per_output_token_seconds` | vLLM Prometheus exporter | Streaming throughput perception |
| `vllm:gpu_cache_usage_perc` | vLLM Prometheus exporter | Are you about to start evicting KV cache? |
| EAS replica count | EAS dashboard | Is autoscaler doing what you expect? |
| EAS pending requests | EAS dashboard | Are requests queueing? |

You can scrape vLLM's `/metrics` endpoint and pipe it to ARMS or Prometheus. PAI also has a built-in metrics view but it doesn't expose the LLM-specific counters by default.

## Image vs Processor, in one table

| | Image mode | Processor mode |
|---|---|---|
| Code shape | Whatever HTTP server you want | Subclass + YAML |
| LLM serving (vLLM/SGLang) | Yes, native | Painful |
| Streaming responses | Native | Possible but awkward |
| Custom dependencies | `pip install` in your Dockerfile | Pre-baked image only |
| Time to first deploy | 30 min if you're new to Docker | 5 min |
| Best for | LLMs, vision models, anything modern | Tabular models, sklearn, legacy code |

I have not deployed a Processor-mode service in over a year.

## Blue/green and A/B routing

EAS supports two replicas of a service under one logical endpoint, with a percentage traffic split. The pattern I use for LLM rollouts:

1. Deploy `qwen-7b-prod` v1 — 100% traffic
2. Deploy `qwen-7b-prod` v2 alongside — 0% traffic, warm
3. Shift 5% → 25% → 50% → 100% over a few hours, watching error rate and latency
4. Tear down v1 once v2 has been at 100% for a day

The SDK call is `predictor.update_traffic_split(...)`. Console works too. The important thing is **never push v2 directly to 100%** even if you "tested it locally" — production traffic always finds a prompt your eval set didn't.

## What's next

Article 5 — the last in this series — is the honest comparison of **PAI-Designer** (drag-and-drop pipeline builder) and **PAI-QuickStart** (one-click model deploy from the model hub). Both have legitimate uses, both also tend to get oversold; I'll cover when each actually beats the DSW + DLC + EAS path you've seen so far.
