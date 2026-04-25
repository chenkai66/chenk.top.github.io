---
title: "Aliyun PAI (3): PAI-DLC — Distributed Training Without the Yak Shaving"
date: 2026-04-27 09:00:00
tags:
  - Aliyun PAI
  - Machine Learning
  - PAI-DLC
  - Distributed Training
  - PyTorch DDP
categories: Aliyun PAI
lang: en
mathjax: false
series: aliyun-pai
series_title: "Aliyun PAI Practical Guide"
series_order: 3
description: "How to submit real PyTorch DDP jobs to PAI-DLC, mount data efficiently, survive spot preemption, and a working 8-GPU LLM SFT example you can adapt to your own model."
disableNunjucks: true
---

The first time I ran a multi-node training job on raw GPU ECS instances I spent two days debugging NCCL connectivity, another half day on `torchrun`'s rendezvous, and a full day on permissions for the OSS mount. PAI-DLC is the product I wish I'd used from the start. Submit a job spec, get a cluster, get logs, get a checkpoint in OSS, get a bill. Done.

This article walks through DLC end-to-end with a concrete 8-GPU LLM supervised fine-tune. By the end you'll have a working SDK script, an understanding of when DLC is worth it over DSW, and a few hard-won opinions about data loading.

## What DLC actually is

PAI-DLC (Deep Learning Container) is a managed batch-job system for distributed training. You hand it:

1. A **container image** (one of PAI's, or your own from ACR)
2. A **command** to run inside the image (`torchrun ...`, `python train.py`, etc.)
3. A **resource spec** (worker count, GPUs per worker, CPU/RAM, instance SKU)
4. **Mounts** (OSS bucket, NAS volume)
5. **Hyperparameters** (passed as env vars)

It schedules onto an internal GPU cluster, sets up the inter-node networking (RDMA where available), runs your job, streams logs, and tears down when you're done. You pay per GPU-second. There is a spot-instance pool that's roughly 40% cheaper if your job can survive preemption — and after this article, yours will be able to.

## DSW vs DLC, decided once

Use DSW when you're iterating on the script. Move to DLC when **any** of these is true:

- More than one GPU
- Run time > 4 hours unattended
- Need RDMA / NCCL across nodes
- Want to fan out a hyperparameter sweep
- Want to use spot GPUs for cost

The transition is usually painless because DLC accepts the same image you used in DSW.

## Job framework: PyTorchJob, the only one that matters

DLC supports several "job frameworks" — TensorFlowJob, PyTorchJob, MPIJob, ElasticBatchJob. For modern work you want **PyTorchJob**: it sets `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, and `LOCAL_RANK` env vars on each container so `torchrun` and `torch.distributed.init_process_group` Just Work.

Concretely, this is what `torchrun` sees inside a 2-node × 4-GPU PyTorchJob:

```
WORLD_SIZE=8       RANK=0..7      LOCAL_RANK=0..3
MASTER_ADDR=<rank-0 hostname>   MASTER_PORT=23456
```

Your training script doesn't need to care about DLC at all. That's the whole point.

## A real 8-GPU LLM SFT job

Suppose we want to SFT a 7B model (Qwen2.5-7B, but pick whichever) on a few thousand instruction examples. The training script is plain Hugging Face — nothing PAI-specific:

```python
# train_sft.py — runs identically on a laptop, in DSW, and in DLC
import os, torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)

MODEL  = os.environ.get("MODEL_PATH", "/mnt/models/Qwen2.5-7B")
DATA   = os.environ.get("DATA_PATH",  "/mnt/data/sft_train")
OUTPUT = os.environ.get("OUTPUT_DIR", "/mnt/output/run-001")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ds  = load_from_disk(DATA)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

args = TrainingArguments(
    output_dir=OUTPUT,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    logging_steps=10,
    deepspeed=os.environ.get("DEEPSPEED_CONFIG"),  # optional
    ddp_find_unused_parameters=False,
    report_to="none",
)

Trainer(model=model, tokenizer=tok, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
```

Now submit it via the PAI Python SDK:

```python
# submit_dlc.py
import os
from pai.session import setup_default_session
from pai.job import TrainingJob, TrainingJobSpec
from pai.image import retrieve

setup_default_session(region_id="cn-shanghai")

image = retrieve(
    framework_name="PyTorch",
    framework_version="2.4",
    accelerator_type="GPU",
).image_uri

spec = TrainingJobSpec(
    display_name="qwen25-7b-sft-001",
    framework="PyTorchJob",
    image_uri=image,
    instance_type="ecs.gn7e-c12g1.3xlarge",   # 1 x A100-40GB per worker
    instance_count=2,                          # 2 workers
    accelerator_count_per_node=4,              # 4 GPUs each => 8 GPUs total
    code_dir="./code",                         # uploaded to OSS, mounted at /ml/code
    command=(
        "cd /ml/code && "
        "torchrun --nproc_per_node=4 "
        "--nnodes=$WORLD_SIZE_NODES --node_rank=$NODE_RANK "
        "--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT "
        "train_sft.py"
    ),
    environment_variables={
        "MODEL_PATH": "/mnt/data/models/Qwen2.5-7B",
        "DATA_PATH":  "/mnt/data/sft_train",
        "OUTPUT_DIR": "/mnt/output/qwen25-7b-sft-001",
    },
    inputs={
        "data":   "oss://my-bucket/datasets/sft_train/",
        "models": "oss://my-bucket/models/Qwen2.5-7B/",
    },
    outputs={
        "output": "oss://my-bucket/checkpoints/qwen25-7b-sft-001/",
    },
)

job = TrainingJob.create(spec)
print("Job ID:", job.job_id)
print("Console:", job.console_url)
job.wait_for_completion(show_logs=True)
```

When this runs, DLC:

1. Pulls the PyTorch 2.4 image onto each worker
2. Mounts your OSS dataset and model directories into `/mnt/data/...`
3. Sets `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, `NODE_RANK` etc.
4. Runs your `torchrun` command on each worker simultaneously
5. Streams stdout/stderr back to the console (and to your terminal via `wait_for_completion(show_logs=True)`)
6. Syncs `/mnt/output/...` back to OSS on completion

Cost on the day I'm writing this: 2 × 4 × A100 ≈ ¥48/hour for the on-demand pool. A 3-epoch SFT on 50k examples takes about 4 hours, so call it ¥200. Spot is around ¥120 for the same job if you don't get preempted.

## Data loading: the choice that determines whether you're idle

The single biggest lever for DLC throughput is how you read data. Three options, ranked by how often I use them:

1. **Stream from OSS via OSS-FUSE mount** — the default the SDK gives you. Easy, no code changes. Fine for big sequential reads (model weights, sharded parquet). **Bad** for millions of small files.
2. **Pre-shard into webdataset/litdata `.tar` files on OSS** — DataLoader reads tars sequentially, decodes on-the-fly. This is what I use for image/audio at scale.
3. **Sync to local NVMe on job start** — first thing the script does is `ossutil cp -r oss://... /local/data/`, then trains from `/local/data/`. Adds 2–10 minutes of startup but the per-step overhead disappears. Worth it for runs > 2 hours.

> **Real-world tip:** Watch GPU utilization in the DLC dashboard. If `nvidia-smi` shows your A100s sitting at 30%, the bottleneck is data loading, not compute. Switching from OSS-FUSE random-access to local NVMe took one of my LLaVA jobs from 36 GPU-hours to 14.

## Surviving spot preemption

Spot pricing is roughly 40% cheaper but the cluster can yank your nodes back with ~30 seconds warning. To make a job preemption-safe:

1. **Checkpoint frequently to OSS.** `save_steps=200` in the snippet above is fine for a 3-epoch SFT.
2. **Make the script resumable.** Hugging Face `Trainer` does this automatically if you pass `resume_from_checkpoint=True` and `output_dir` already contains a `checkpoint-*`.
3. **Set `auto_recovery=True` in the job spec.** DLC will re-launch the failed worker(s); your script reattaches via the resume logic.

```python
spec = TrainingJobSpec(
    ...,
    spot_spec={"enable": True, "max_price_ratio": 0.6},
    fault_tolerance={"auto_recovery": True, "max_retries": 5},
)
```

The catch: every `auto_recovery` retry restarts the *whole* job from the latest checkpoint, not from where the failed worker died. If your checkpoint cadence is too coarse you can lose hours. Tune `save_steps` so a single retry costs < 10% of total runtime.

## Logs, metrics, debugging

DLC streams stdout/stderr to the console under "Logs" per worker. The console search is OK for grep-style lookups. For anything more interesting, the SDK exposes:

```python
for line in job.stream_logs(worker_index=0):
    print(line)
```

Hardware metrics (GPU util, GPU mem, network) are visualized per worker under "Monitoring". Two things to actually look at:

- **GPU utilization < 70%** → data-loading bottleneck
- **GPU memory close to 100%** → reduce batch size or enable gradient checkpointing before you OOM at step 5000

For deeper debugging, attach a TensorBoard log directory in OSS and add `report_to="tensorboard"` to TrainingArguments. The PAI console has a built-in TensorBoard viewer that points at any OSS path.

## What's next

Article 4 picks up the checkpoint produced by this DLC job and serves it through PAI-EAS — including the cold-start trap that has bitten me on three separate launches and the difference between Image mode and Processor mode for PyTorch models.
