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
description: "What Aliyun PAI actually is in 2026, the four-layer architecture from the official docs, the five sub-products you'll touch, and a sane account/workspace setup so the rest of the series can skip the boilerplate."
disableNunjucks: true
translationKey: "aliyun-pai-1"
---

If your team trains or serves any model on Alibaba Cloud, sooner or later you will end up in the PAI console. PAI is the umbrella; underneath it sit the actual workhorses — a notebook product, a distributed training service, a model-serving service, plus a couple of GUI/quick-deploy layers on top. After about eighteen months of running real LLM workloads on it for an AI marketing platform, this series is the field guide I wish someone had handed me before I shipped my first endpoint.

This first article is the lay of the land. It is deliberately short on code — articles 2 to 5 are the deep dives. The goal here is so that when I say "DLC job" or "EAS endpoint" later you already know what bucket they fall into.

![Aliyun PAI (1): Platform Overview and the Product Family Map — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/01-platform-overview/illustration_1.png)

## What PAI is, and what it isn't

Per the official docs, **Platform for AI (PAI)** is "Alibaba Cloud's AI development platform covering the full lifecycle: data annotation, model development, training, and deployment". The console at `pai.console.aliyun.com` is one entry point, but PAI itself is a *family* of related products that share an account model, an OSS-backed storage layer, and a single Python SDK.

The mental model that has worked best for me:

- **PAI is the shop.**
- **DSW, DLC, EAS, Designer, Model Gallery** are the workbenches inside.
- **ECS, OSS, NAS, CPFS** are where the actual silicon and bytes live. PAI just orchestrates them on your behalf.

The official "Service architecture" topic spells it out as a four-layer stack:

![PAI four-layer service architecture](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/01-platform-overview/fig1_pai_4layer_architecture.png)

Read that bottom-up. The infrastructure layer is the silicon — CPUs, GPUs, RDMA fabric, and ACK Kubernetes underneath. On top of that, *Lingjun* (灵骏) gives you very-high-density AI compute and *general-purpose compute* gives you everyday ECS-backed GPU pools. The platform-and-tools layer is where you spend your day: PyTorch / Megatron / DeepSpeed, plus PAI's optimization toys (TorchAcc, BladeLLM, EasyCkpt, AIMaster), plus the visible products (DSW, DLC, EAS, Designer, FeatureStore, iTAG). The application layer is how PAI plugs into the rest of Alibaba's MaaS world (ModelScope, Bailian/DashScope, Model Studio). The business layer is the marketing slide for industry use cases.

The reason to use PAI instead of raw ECS is that it pre-bakes the CUDA / PyTorch images, mounts your OSS bucket for you, gives you a metrics dashboard, and bills per second.

## The five sub-products you actually touch

After a year and a half of production work I have only ever paid for these, drawn straight from the official "Core components" table:

| Component | Per the docs | When to reach for it |
|---|---|---|
| **DSW** (Data Science Workshop) | Cloud-based IDE with Jupyter / VSCode / terminal, pre-configured PyTorch and TensorFlow images, GPU instances | Interactive dev, debugging, small-scale training |
| **DLC** (Deep Learning Containers) | Kubernetes-based training with Megatron, DeepSpeed, PyTorch, TF, Slurm, Ray, MPI, XGBoost — no cluster setup | Multi-GPU / multi-node SFT, pretraining, large eval |
| **EAS** (Elastic Algorithm Service) | Online inference with auto-scaling, canary release, traffic splitting, mirroring | Production inference endpoints |
| **Designer** | 140+ built-in algorithm components, drag-and-drop pipelines, exportable JSON, schedulable in DataWorks | ETL → train → eval flows handed off to non-coders |
| **Model Gallery** | Wraps DLC + EAS for zero-code deploy and fine-tune of catalogued open-source models | Evaluating a Qwen / DeepSeek / Llama model in 10 minutes |

There's also **iTAG** (data annotation), **PAI-Lingjun** for very large clusters, **PAI-Blade / BladeLLM** for inference optimization, and **FeatureStore**, but unless you're doing >1000-GPU pretraining or building a recommender system, you can ignore them on day one.

The product split maps cleanly onto the ML lifecycle:

![PAI sub-products on the ML lifecycle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/01-platform-overview/fig2_pai_subproducts_lifecycle.png)

Designer and Model Gallery are orthogonal — they sit on top, generating jobs that ultimately run on the same DLC / EAS substrate.

## How PAI relates to ECS and OSS

This trips up everyone who comes from a pure cloud-VM background. Three rules:

1. **PAI never owns your data.** Datasets, checkpoints, and model artifacts all live in **OSS** (or **NAS** for POSIX semantics, or **CPFS** for HPC-style throughput). When a DSW or DLC instance dies, anything you didn't write to OSS is gone. There is a "system disk" but treat it as `/tmp`.
2. **PAI does own the compute.** You do not provision GPU ECS instances yourself for PAI workloads. PAI manages a pool, you ask for `1 * ecs.gn7i-c8g1.2xlarge` and you get billed per second of allocation.
3. **PAI shares your account but uses its own RAM roles.** When you grant PAI access to OSS, you're attaching a service-linked role (`AliyunPAIAccessingOSSRole`) so PAI's compute can read your bucket without a long-lived AK pair. Do not skip this — without it your DLC jobs will fail at `data_loader` time with a 403.

> **Real-world tip:** The single most common "PAI is broken" ticket is a permission issue between PAI and OSS. Before debugging your training script, run `oss ls oss://your-bucket/` from inside a DSW terminal. If that fails, fix the role, not the code.

## Account, region, workspace

To get started you need three things in this order:

1. **An aliyun.com account** with real-name verification (实名认证) — required for any GPU resource. International accounts work for most regions but Hangzhou, Shanghai, and Beijing have the best GPU stock.
2. **A region.** Pick one and stick to it. PAI resources, OSS buckets, and ECS GPUs are all region-scoped, and cross-region traffic costs money and adds latency. For mainland production I default to `cn-shanghai`; for international, `ap-southeast-1` (Singapore).
3. **A workspace.** Per the docs, the workspace is PAI's tenancy primitive — it holds quotas, datasets, model registries, and IAM bindings. You almost always want at least two: a `dev` workspace where humans poke around in DSW, and a `prod` workspace where DLC jobs and EAS endpoints live. Cross-workspace permissioning is fiddly, but the isolation pays for itself the first time an intern accidentally restarts a serving endpoint.

![PAI tenancy: account, region, workspace](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/01-platform-overview/fig3_pai_account_workspace.png)

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

The docs list five billing methods: **pay-as-you-go**, **subscription** (monthly/yearly prepaid), **resource plan** (DSW prepaid quota), **savings plan** (commit for discount), and **pay-by-inference-duration** (EAS serverless — no idle replica cost). DSW is per-second-of-instance while running, DLC is per-second with a separate quota for spot/preemptible GPUs that's roughly 30-50% cheaper if your job can checkpoint, EAS is per-second-of-replica plus a small per-million-requests charge with auto-scaled minimum replicas dominating the cost. Designer and Model Gallery have no charge themselves — they spawn DLC/EAS resources that bill normally. There's a small free tier for new accounts that's enough to follow this whole series end-to-end.

## What a Designer workflow really looks like under the hood

![Aliyun PAI (1): Platform Overview and the Product Family Map — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/01-platform-overview/illustration_2.png)

Designer feels like Lego on the canvas, but the export is a flat JSON document the platform can replay. The first time I had to debug a stuck pipeline I exported the JSON from the canvas (`...` menu → *Export workflow*) and it stopped looking like magic. Roughly the structure:

```json
{
  "workflowName": "user-segmentation-weekly",
  "workspaceId": "ws-1xxxx",
  "globalParams": {
    "outputProject": "marketing_dwd",
    "samplingRatio": 0.05
  },
  "nodes": [
    {
      "id": "n1",
      "type": "ReadODPSTable",
      "params": {"projectName": "marketing_dwd", "tableName": "user_features_p"},
      "position": {"x": 60, "y": 80}
    },
    {
      "id": "n2",
      "type": "Split",
      "params": {"trainRatio": 0.8},
      "inputs": [{"from": "n1", "port": "out"}]
    },
    {
      "id": "n3",
      "type": "KMeans",
      "params": {"k": 12, "maxIter": 50, "seed": 42},
      "inputs": [{"from": "n2", "port": "train"}],
      "compute": {"engine": "MaxCompute", "instanceQuota": "general"}
    }
  ],
  "edges": [
    {"from": "n1.out", "to": "n2.in"},
    {"from": "n2.train", "to": "n3.in"}
  ],
  "schedule": {"cron": "0 3 * * 1", "retries": 2, "timeoutMin": 60}
}
```

Three things matter here. First, every node has an explicit `compute.engine` — `MaxCompute` for SQL-shaped components, `DLC` for the Python / PyAlink ones, `Flink` if you're streaming. Designer does not magically pick the cheapest one; it picks based on the component type, and any custom Python node defaults to a DLC pod that costs you whether the upstream MaxCompute step took 30 s or 30 min. Second, the JSON is *idempotent* and *diffable* — I keep ours in git next to the SQL it depends on, and code-review changes the same way I review a Terraform plan. Third, the same JSON can be submitted via the SDK (`pai.designer.submit_workflow(json_path)`), which is how I run pipelines from CI without ever opening the console.

The hidden upgrade path: once you have the JSON, "I want this in code" stops being a rewrite. You can generate it from a Python DAG, version it, lint it, and even unit-test individual nodes by mocking inputs. The canvas is a UI on top of that file, not a separate tool.

## When Designer is actually faster than custom code, and when it isn't

I've watched teams burn a week building a custom DLC + PySpark pipeline for a job that would have been a 90-minute Designer canvas, and I've watched the opposite — a senior engineer try to do real-time feature engineering inside Designer and fight the platform for two weeks. The split is sharper than the marketing suggests:

| Signal | Designer wins | Custom code wins |
|---|---|---|
| Source data lives in MaxCompute? | Yes (saves cross-system pulls) | No (Designer's strongest binding is wasted) |
| Pipeline mostly tabular ETL + classical ML? | Yes | No (LLM, RL, custom CUDA → DLC) |
| Owner is a data analyst / BI engineer? | Yes | No (engineers will resent the canvas) |
| Cron + retries enough? | Yes | No (custom triggering, multi-stage approvals → Airflow / DataWorks code) |
| Custom Python > 100 lines per node? | No (the canvas becomes a wrapper around a `.py`) | Yes (just write the script) |
| Need streaming / sub-second SLA? | No (Designer is batch-shaped) | Yes |

A specific number from production: a recommendation feature pipeline I migrated *to* Designer ran in 22 min on MaxCompute spot and cost ~6 RMB per daily run; the previous PySpark-on-DLC version took 41 min on a 4-node cluster and cost ~38 RMB. Six times cheaper because the data never had to leave MaxCompute. The opposite migration also happened: a "use Designer for our Qwen LoRA" attempt that needed a custom training loop ended up as 600 lines of glue inside one Python node and was painful to debug because the canvas hides stack traces under a tiny `view logs` link. We rewrote it as a normal DLC submission in an afternoon and the team was happier.

The trap is treating Designer as either *always* the right answer (PMs do this) or *always* wrong (senior engineers do this). It is a tool for a specific class of problems — tabular ML at MaxCompute scale with a non-coder owner — and it is genuinely the best option in that class.

## The DLC backend that Designer hides from you

This took me longer to figure out than it should have. Click *Run* on a Designer canvas with a Python node and the platform spins up a DLC job behind the scenes — same one you'd see if you submitted via the SDK in chapter 3. The canvas does not show this clearly; you have to dig into *Operations → Recent runs → View backend job* to find it. Implications:

- **You pay DLC prices for Python nodes**, even if your canvas otherwise looks "MaxCompute-only". One stray `PyAlink` step on a 1-node `ecs.gn6i-c8g1.2xlarge` is roughly 4-5 RMB/h while the canvas is running.
- **Container image versions matter.** Designer pins its Python nodes to a specific image (typically `pai-designer:latest` or a version tag in the workspace settings). If your custom Python imports `vllm==0.8.2` but the image ships `0.6.0`, the run fails at import time with a stack trace that doesn't surface in the canvas — only in the underlying DLC log. I burned half a day on this once.
- **Quotas leak across products.** If your DLC quota is tight, a Designer run that needs a Python pod can starve a real DLC training job of resources, and vice versa. They share the same workspace quota pool. The fix: a separate "designer-only" resource quota in the workspace, capped at maybe 4 CPU + 16 GB so a runaway canvas can't eat your training budget.

Once you know Designer is just a UI for emitting MaxCompute SQL + DLC pod submissions, the entire product becomes much less mysterious — and you can route around it cleanly when needed.

## What's next

Article 2 is **PAI-DSW** end-to-end: picking the right GPU instance, the image catalog, OSS-FUSE mounting, and a working MNIST notebook (the one straight out of the official Quick Start). Article 3 is **PAI-DLC** distributed training — a real multi-GPU job with AIMaster fault tolerance. Article 4 is **PAI-EAS** model serving, including the cold-start trap that has bitten me more than once. Article 5 is the honest comparison of **Designer vs Model Gallery** for the "I just want to ship something" cases.

If you only read one, read article 4 — EAS is where most of the production money is spent and where the docs are thinnest.
