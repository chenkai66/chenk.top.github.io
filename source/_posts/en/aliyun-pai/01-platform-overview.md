---
title: "Aliyun PAI (1): Platform Overview and the Product Family Map"
date: 2026-03-05 09:00:00
tags:
  - Aliyun PAI
  - Machine Learning
  - DSW
  - DLC
  - EAS
categories: Aliyun PAI
lang: en
mathjax: false
series: aliyun-pai
series_title: "Aliyun PAI Practical Guide"
series_order: 1
description: "What Aliyun PAI actually is in 2026, the five sub-products you'll touch (DSW, DLC, EAS, Designer, QuickStart), how they relate to ECS and OSS, and a sane account setup so the rest of the series can skip the boilerplate."
disableNunjucks: true
---

If your team trains or serves any model on Alibaba Cloud, sooner or later you will end up in the PAI console. PAI is the umbrella; underneath it sit the actual workhorses — a notebook product, a distributed training service, a model-serving service, and a couple of GUI/quick-deploy layers on top. After about eighteen months of running real LLM workloads on it for an AI marketing platform, this series is the field guide I wish someone had handed me before I shipped my first endpoint.

This first article is the lay of the land. It's deliberately short on code — articles 2–5 are the deep dives. The goal here is so that when I say "DLC job" or "EAS endpoint" later you already know what bucket they fall into.

## What PAI is, and what it isn't

PAI (Platform for AI, 阿里云机器学习平台 PAI) is **a family of related products** that share a console, an account model, an OSS-backed storage layer, and a single Python SDK. It is not one monolithic service. The console at `pai.console.aliyun.com` is just a launcher; each click takes you into a different sub-product with its own quotas, pricing, and resource model.

The mental model that has worked best for me:

- **PAI is the shop**.
- **DSW, DLC, EAS, Designer, QuickStart, Studio** are the workbenches inside.
- **ECS, OSS, NAS** are where the actual silicon and bytes live. PAI just orchestrates them on your behalf.

Treat PAI as an opinionated layer over GPU ECS instances. When a DSW notebook starts, a real `ecs.gn7i-c8g1.2xlarge` (or whatever you picked) boots somewhere. When an EAS endpoint scales out, real GPU pods come up. The reason to use PAI instead of raw ECS is that it pre-bakes the CUDA/PyTorch images, mounts your OSS bucket, gives you a metrics dashboard, and bills you per second.

## The five sub-products you actually touch

After a year and a half of production work I have only ever paid for these five:

| Product | What it solves | When to reach for it |
|---|---|---|
| **PAI-DSW** (Data Science Workshop) | Cloud Jupyter with GPU, mounted OSS, pre-baked images | Interactive dev, debugging, small-scale training |
| **PAI-DLC** (Deep Learning Container) | Submitted distributed training jobs on a managed cluster | Multi-GPU / multi-node SFT, pretraining, large eval |
| **PAI-EAS** (Elastic Algorithm Service) | Model serving with autoscaling, blue/green, traffic splitting | Production inference endpoints |
| **PAI-Designer** | Drag-and-drop pipeline composer (formerly PAI-Studio) | ETL → Train → Eval flows handed off to non-coders |
| **PAI-QuickStart** | One-click deploy of catalogued open-source models | Evaluating a new HF model in 10 minutes |

There is also **PAI-Studio**, **PAI-Lingjun (灵骏)** for very large clusters, **PAI-Blade** for inference optimization, and a few others, but unless you're doing >1000-GPU pretraining or ASIC-targeted optimization, you can ignore them on day one.

The product split maps neatly onto the ML lifecycle:

```
        DSW          DLC           EAS
         |            |             |
    [explore]   [train at scale] [serve]
         \           |            /
          \          |           /
           +---- OSS / NAS -----+
                    |
                  ECS GPUs
```

Designer and QuickStart are orthogonal — they sit on top, generating jobs that ultimately run on the same DLC/EAS substrate.

## How PAI relates to ECS and OSS

This trips up everyone who comes from a pure cloud-VM background. Three rules:

1. **PAI never owns your data.** Datasets, checkpoints, and model artifacts all live in **OSS** (or **NAS** for POSIX semantics). When a DSW or DLC instance dies, anything you didn't write to OSS is gone. There is a "system disk" but treat it as `/tmp`.
2. **PAI does own the compute.** You do not provision GPU ECS instances yourself for PAI workloads. PAI manages a pool, you ask for `1 * ecs.gn7i-c8g1.2xlarge` and you get billed per second of allocation.
3. **PAI shares your account but uses its own RAM roles.** When you grant PAI access to OSS, you're attaching a service-linked role (`AliyunPAIAccessingOSSRole`) so PAI's compute can read your bucket without a long-lived AK pair. Do not skip this — without it your DLC jobs will fail at `data_loader` time with a 403.

> **Real-world tip:** The single most common "PAI is broken" ticket is a permission issue between PAI and OSS. Before debugging your training script, `oss ls oss://your-bucket/` from inside a DSW terminal. If that fails, fix the role, not the code.

## Account, region, workspace

To get started you need three things in this order:

1. **An aliyun.com account** with real-name verification (实名认证) — required for any GPU resource. International accounts work for most regions but Hangzhou/Shanghai have the best GPU stock.
2. **A region.** Pick one and stick to it. PAI resources, OSS buckets, and ECS GPUs are all region-scoped, and cross-region traffic costs money and adds latency. For mainland production I default to `cn-shanghai`; for international, `ap-southeast-1` (Singapore).
3. **A workspace.** Workspaces are PAI's tenancy primitive — they hold quotas, datasets, model registries, and IAM bindings. You almost always want at least two: a `dev` workspace where humans poke around in DSW, and a `prod` workspace where DLC jobs and EAS endpoints live. Cross-workspace permissioning is fiddly, but the isolation pays for itself the first time an intern accidentally restarts a serving endpoint.

```bash
# All of this is region-scoped
aliyun configure set --region cn-shanghai
```

## Two paths: console vs SDK

Like Bailian, PAI gives you two ways to do everything. The console is good for one-offs and inspecting state; the SDK is what you ship in CI.

The Python SDK is one package:

```bash
pip install alibabacloud-pai-python-sdk
```

A "hello PAI" — list your workspaces:

```python
import os
from pai.session import setup_default_session

sess = setup_default_session(
    access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
    access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
    region_id="cn-shanghai",
)

for ws in sess.workspace_api.list().items:
    print(ws.id, ws.name)
```

If that prints at least one workspace ID, your account, region, and credentials are wired correctly and you can move on to article 2.

> **Real-world tip:** Use a sub-account with a scoped RAM policy for SDK work. Never use the root account access key — and if your AK pair shows up in any git history, rotate immediately. Aliyun's leaked-key detection is OK but it's not GitHub-grade fast.

## Pricing model in one paragraph

DSW is billed **per-second-of-instance** while running, plus the underlying disk if you use persistent storage. DLC is also per-second, with a separate quota for spot/preemptible GPUs that's roughly 30–50% cheaper if your job can checkpoint. EAS is per-second-of-replica plus a small per-million-requests charge; auto-scaled minimum replicas dominate the cost, not the request volume. Designer and QuickStart have no charge themselves — they spawn DLC/EAS resources that bill normally. There's a small free tier for new accounts (a few hundred yuan) that's enough to follow this whole series end-to-end.

## What's next

Article 2 is **PAI-DSW** end-to-end: picking the right GPU instance, the image catalog, OSS-FUSE mounting, and a working CIFAR-10 ResNet notebook. Article 3 is **PAI-DLC** distributed training — a real 8-GPU LLM SFT job. Article 4 is **PAI-EAS** model serving, including the cold-start trap that has bitten me more than once. Article 5 is the honest comparison of **Designer vs QuickStart** for the "I just want to ship something" cases.

If you only read one, read article 4 — EAS is where most of the production money is spent and where the docs are thinnest.
