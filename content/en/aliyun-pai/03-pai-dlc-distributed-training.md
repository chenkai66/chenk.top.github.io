---
title: "Aliyun PAI (3): PAI-DLC — Distributed Training Without the Cluster Pain"
date: 2026-03-07 09:00:00
tags:
  - Aliyun PAI
  - PAI-DLC
  - Distributed Training
  - LLM
  - SFT
categories: Aliyun PAI
lang: en
mathjax: false
series: aliyun-pai
series_title: "Aliyun PAI Practical Guide"
series_order: 3
description: "Submit a real multi-GPU training job on PAI-DLC, understand the resource pools (Lingjun vs general vs preemptible), and use AIMaster + EasyCKPT so a flaky node doesn't cost you a day."
disableNunjucks: true
translationKey: "aliyun-pai-3"
---

A DSW notebook is for one engineer on one GPU. The moment you need eight GPUs across two nodes, or the moment training runs longer than the eight hours you'll keep the tab open, you switch to **DLC**. DLC is PAI's job-submission front-end for a managed Kubernetes cluster: you describe what you want (image, command, resources, data mounts), DLC schedules pods, runs them to completion, persists logs, and tells you what happened. The docs call this *Deep Learning Containers*; we just say "DLC job".

![Aliyun PAI (3): PAI-DLC — Distributed Training Without the Cluster Pain — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/03-pai-dlc-distributed-training/illustration_1.jpg)

## What the docs actually claim

The official DLC overview lists four bullets I want to highlight, because they matter:

- **Diverse compute** — Lingjun AI computing service, ECS, ECI, Shenlong bare metal, Lingjun bare metal. Hybrid scheduling.
- **Multiple distributed job types** — pre-built support for Megatron, DeepSpeed, PyTorch DDP, TensorFlow PS/Worker, Slurm, Ray, MPI, XGBoost. No need to build your own cluster.
- **Fault tolerance** — AIMaster (the watchdog), EasyCKPT (the async checkpointer), SanityCheck (pre-flight node health), node self-healing.
- **Training acceleration** — built-in framework with data parallelism, pipeline parallelism, operator splitting, automatic parallel-strategy exploration, topology-aware scheduling, optimized communication.

The first and third bullets are what make DLC interesting compared to renting GPU ECS yourself.

## The job lifecycle

A DLC job goes through six phases between submit and done:

![DLC job lifecycle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/03-pai-dlc-distributed-training/fig1_dlc_job_lifecycle.png)

Two of those phases — **scheduler places pods** and **mount OSS / NAS** — are where almost all of the "my job is stuck in PENDING" tickets get filed. Stuck on schedule means your resource group is full; stuck on mount means your storage RAM role is wrong. Same as DSW, the diagnostic move is to spin up a tiny DSW with the same OSS mount and confirm `oss ls` works.

## Picking a resource pool

You submit to one of three pools. The docs mostly talk about quotas and bills; the practical decision is about how tolerant your job is.

![DLC resource pools](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/03-pai-dlc-distributed-training/fig3_dlc_resource_pools.png)

For most teams the answer is general-purpose, pay-as-you-go. Lingjun makes sense once you're training above 8 GPUs and need RDMA between nodes — the docs note RDMA is configurable on Lingjun and gives "accelerated inter-node communication" (which is a polite way of saying NCCL AllReduce will be 5-10x faster than over standard ethernet). Preemptible is a cost saver for jobs that checkpoint cleanly, which thanks to EasyCKPT is most jobs.

## A real distributed job

Here is the topology you build with a four-node, 2-GPU-per-node DLC job:

![DLC distributed training topology](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/03-pai-dlc-distributed-training/fig2_dlc_distributed_topology.png)

A minimal `PyTorchJob`-style submission via the SDK, scaled out from the MNIST notebook:

```python
from pai.job import TrainingJob

job = TrainingJob(
    name="mnist-ddp",
    image_uri="dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai/"
              "modelscope:1.28.0-pytorch2.3.1tensorflow2.16.1-gpu-py311-cu121-ubuntu22.04",
    command=(
        "torchrun --nproc_per_node=2 --nnodes=$WORLD_SIZE "
        "--node_rank=$RANK --master_addr=$MASTER_ADDR "
        "--master_port=$MASTER_PORT /mnt/data/code/train_ddp.py"
    ),
    job_type="PyTorchJob",
    instance_count=4,                # 4 worker nodes
    instance_type="ecs.gn7i-c16g1.4xlarge",  # 2 x A10 each
    datasets={"train": "oss://your-bucket/datasets/mnist"},
    code_uri="oss://your-bucket/code/mnist-ddp.zip",
    output_uri="oss://your-bucket/runs/mnist-ddp/",
    fault_tolerance=True,            # turns on AIMaster
    enable_easyckpt=True,            # async checkpoint
)
job.submit(wait=False)
print(job.id, job.status)
```

A few things worth noting that are not obvious from a quick read of the docs:

- **`$WORLD_SIZE`, `$RANK`, `$MASTER_ADDR`, `$MASTER_PORT`** are injected by DLC. You do not have to discover peers — DLC handles peer discovery and writes those env vars before your container starts. (See "Built-in environment variables" in the User Guide.)
- **`fault_tolerance=True`** spins up an AIMaster sidecar that watches every worker. If a worker pod dies, AIMaster marks it, requests a replacement, and the surviving workers wait for it instead of crashing the whole job. This is the *single most important toggle* for jobs longer than a few hours.
- **`enable_easyckpt=True`** swaps `torch.save` for an async path that writes to OSS without blocking the training step. On a 70B model this turns checkpointing from a 3-minute stall into about 10 seconds of overlap.
- **Image URL is region-specific.** The `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com` prefix only works inside the Shanghai VPC; use the matching one for your region or pulls will time out.

## Watching it run

The console "Training Jobs" view gives you logs, GPU utilization, network throughput, and AIMaster events. From the SDK you can stream logs:

```python
for line in job.tail_logs(follow=True):
    print(line)
```

For longer jobs I forward logs to SLS (Log Service) and set CloudMonitor alerts on `gpu_util < 0.3` for 15 minutes — that is the canonical signal that something is wedged on data loading or distributed init.

## Common failures and what they actually mean

| Symptom | Real cause |
|---|---|
| Job stuck in `Pending` for >5 min | Resource group full, or your quota is exhausted. Switch pool or reduce `instance_count`. |
| `cannot mount oss://...` at startup | RAM role missing the `AliyunPAIAccessingOSSRole` attachment. Re-attach in workspace settings. |
| NCCL hangs at start of step 1 | RDMA misconfig on Lingjun, or a flaky node. Enable `SanityCheck` to isolate before the run. |
| Loss explodes on resume from checkpoint | EasyCKPT saved optimizer state but you did not load it. Read the EasyCKPT load helper, not `torch.load`. |
| Job finishes but `output_uri` is empty | Your training script wrote to `/root` instead of the mounted OSS path. Recheck `OUTPUT_DIR`. |

## Cost reality

For a typical 7B SFT (4 x A10, 6 hours) on general-purpose pay-as-you-go you're looking at roughly the cost of an OK dinner in Shanghai. A 70B QLoRA (8 x A100 80 GB, 12 hours) is closer to a long weekend in Hangzhou. Preemptible cuts that 30-50% if your job can survive being killed every few hours — with EasyCKPT it can.

## AIMaster fault tolerance, what actually gets restarted

The docs say AIMaster does "fault tolerance"; they don't say what that means at the pod level, which is the actual question. Here's what I've reverse-engineered from running it across roughly 200 jobs.

AIMaster runs as a sidecar pod alongside your training pods, with its own ServiceAccount that has the right K8s permissions to mark, evict, and recreate worker pods. It does three things:

1. **Liveness probing.** Every 5-10 s it checks each worker's health endpoint (a small HTTP server AIMaster injects into your container via a side-mount). The check is shallow — process alive, GPU visible, no NCCL deadlock signature in `/proc/[pid]/stack`. It is *not* a deep check on training progress; if your loss is NaN, AIMaster will not notice.
2. **Restart policy.** When a worker fails the liveness check three times in a row, AIMaster marks the pod failed, asks the scheduler for a replacement on a healthy node (avoiding the one that just failed), and waits for the new pod to come up. The other workers are blocked at the next collective op (an `all_reduce` or `barrier`) — they don't crash, they wait. Once the replacement is in, AIMaster signals "resume" and the workers re-init the process group.
3. **Checkpoint coordination.** This is the bit most people miss. AIMaster doesn't restore your model state — your training script does that. AIMaster just guarantees the new worker can find the latest checkpoint. If you used `enable_easyckpt`, it knows which checkpoint is the most recent durable one and exposes it via env var `EASYCKPT_RESUME_PATH`. Your script should `if os.path.exists(EASYCKPT_RESUME_PATH): load_from(EASYCKPT_RESUME_PATH)` at start.

What does *not* get restarted automatically:

- **NaN loss / diverging training.** AIMaster doesn't watch your loss curve. Your script needs to detect this and exit non-zero so AIMaster sees the worker as failed.
- **OOM in the middle of a step.** If a single worker OOMs, AIMaster restarts it, but the underlying cause (a single oversized batch) will repeat unless you wrap your forward pass in a try/except and skip the bad batch. Otherwise AIMaster will restart, OOM, restart, OOM in a tight loop until your max retry budget is hit.
- **Hung NCCL collectives that don't trip the liveness check.** Some NCCL bugs leave the process technically alive but stuck in `cudaStreamSynchronize` forever. AIMaster catches some of these via the `/proc` stack check, but not all. The mitigation is `NCCL_TIMEOUT=1800` (30 min) so a true hang eventually crashes the process and AIMaster sees it as failed.
- **The AIMaster sidecar itself.** If AIMaster crashes (rare, but I've seen it once when a workspace had stale credentials), no fault tolerance. The DLC scheduler will restart AIMaster, but workers are unsupervised in the meantime.

Practical recommendation: set `max_retries=3` per worker and a job-level `max_runtime` cap. AIMaster + EasyCKPT will handle most flaky-node failures; the budget caps protect you from infinite-loop failures that should be human-investigated.

## Multi-node NCCL: RDMA vs TCP, ring vs tree

![Aliyun PAI (3): PAI-DLC — Distributed Training Without the Cluster Pain — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/03-pai-dlc-distributed-training/illustration_2.jpg)

The single biggest performance lever in multi-node training, and one that's barely documented in PAI's own pages. The defaults work; tuning them halves your per-step time on jobs with heavy gradient communication.

**RDMA vs TCP.** On Lingjun nodes you get RDMA over Converged Ethernet (RoCE) by default; on general-purpose GPU nodes you get TCP. The difference at AllReduce time, measured on a real 8-node × 8-A100 job training a 70B model with full data parallelism:

| Communication | All-reduce 1 GB | Per-step bubble | Tokens/sec/GPU |
|---|---|---|---|
| TCP (general nodes) | 380-450 ms | ~30% | ~85 |
| RoCE (Lingjun) | 35-50 ms | ~4% | ~160 |

The cost of Lingjun is roughly 1.5x general for the same GPU type, but you get nearly 2x throughput on comm-heavy jobs. For data parallelism on >8 GPUs, Lingjun pays for itself. For pipeline parallelism with sparse cross-stage comm, the math is closer.

Enabling RDMA in PAI is a checkbox at job submission (`enable_rdma=True` in the SDK, "RDMA acceleration" in the console). On the wire you also need:

```bash
# In your training command, before torchrun:
export NCCL_IB_HCA=mlx5_0          # use the mellanox HCA
export NCCL_IB_GID_INDEX=3         # RoCE v2 default
export NCCL_SOCKET_IFNAME=eth0     # the data plane interface
export NCCL_DEBUG=INFO             # one-time debug; remove for prod
```

If `nccl_debug=INFO` shows `via NET/IB` you're on RDMA; `via NET/Socket` means it fell back to TCP and you're paying full price for half the perf. I've seen this happen when a node was provisioned outside the RDMA fabric — file a ticket and switch nodes.

**Ring vs tree allreduce.** NCCL picks automatically, but you can override:

```bash
export NCCL_ALGO=Ring   # or Tree, or Auto
```

Ring is bandwidth-optimal for medium-sized buffers (10 MB - 1 GB), tree is latency-optimal for tiny buffers (<1 MB) like gradient sync on a small model. For 70B-class data parallel, ring wins (and is the default). For an 8-node job that does many small all_reduces (e.g., a Megatron pipeline with frequent micro-batch syncs), tree can be 30% faster. The honest answer: try both, time 100 steps each, keep the winner. The difference shows up in `nccl-tests` if you want to benchmark in isolation.

**Topology awareness.** Lingjun nodes know their own NVSwitch topology and PAI's scheduler tries to place worker pods on adjacent nodes (same rack, same spine). If the scheduler can't (resource pressure), you get cross-spine traffic and AllReduce latency goes up. The job-level `topology_constraint=spine` config forces same-spine placement at the cost of possible scheduling delays — worth it for jobs >4 nodes.

## Spot / preemptible quirks: cadence and notice

Preemptible saves 30-50%, but the operational details aren't in the marketing material. After running about 40 preemptible jobs, here's what you actually need to know.

**Preemption notice window.** PAI gives 30 seconds of advance notice via SIGTERM to the worker pod before it's force-killed. That's enough for an emergency `torch.save` if you're already at a checkpoint boundary, not enough for a clean checkpoint mid-step. The signal handler needs to:

```python
import signal, os, threading

_preempt = threading.Event()

def _on_term(signum, frame):
    _preempt.set()
    print("[preempt] SIGTERM received, will checkpoint at next step boundary")

signal.signal(signal.SIGTERM, _on_term)

# In your training loop:
for step, batch in enumerate(loader):
    train_step(model, batch)
    if _preempt.is_set() or step % CKPT_EVERY == 0:
        save_easyckpt(model, optimizer, step)
        if _preempt.is_set():
            sys.exit(0)   # clean exit; AIMaster will not retry within 30 s
```

**Checkpoint cadence.** Without preemption the rule is "checkpoint every N steps where reloading + losing N steps is cheaper than the checkpoint cost". With preemption, that becomes "checkpoint every M steps where M minutes is the average preemption interval". On a 70B SFT in cn-shanghai during 2026 Q1 the average preemption interval was ~6 hours; one step at full sequence length was ~12 s; checkpoint cost was ~10 s with EasyCKPT. So checkpointing every 100 steps (20 min of work) means losing at most 20 min on preemption — acceptable. Without EasyCKPT the same 100-step interval would block the GPU for ~3 min per save, dropping effective throughput by 15%.

**Spot eviction patterns.** Preemption is not random — it's correlated with time of day (more frequent during business hours when on-demand demand is high) and with GPU type (A100 80 GB gets preempted more aggressively than A10). A100s I've seen evicted at 15-30% per day rate during work hours, A10s closer to 5%. For long training runs (>24 h) on A100 spot, expect the job to be killed and restarted at least 3-5 times.

**The retry trap.** Set a job-level `max_retries` (I use 5). Without it, a misbehaving spot job can eat its entire budget bouncing between pods because `instance_count * preemption_rate * retry_cost > original_savings`. The math: 4 × A100 spot at 50% off vs on-demand breaks even at ~3 preemptions per day per pod. More than that, and you're paying *more* than on-demand for less actual training time.

## Dataset sharding patterns for distributed training

The default is `DistributedSampler`, which shards by `(rank, world_size)` and works fine for in-memory datasets. Once your data is on OSS or larger than memory, the patterns matter.

**Pattern 1: Index-shard a manifest file.** Write a single text file with one path per line, sorted, into OSS once. Each worker reads its own range:

```python
class ShardedManifestDataset(torch.utils.data.IterableDataset):
    def __init__(self, manifest_path, rank, world_size):
        self.manifest_path = manifest_path
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        with open(self.manifest_path) as f:
            for i, line in enumerate(f):
                if i % self.world_size == self.rank:
                    yield self._load_one(line.strip())
```

Cheap, no coordination needed, scales to billions of samples. The shard boundary is per-line so workers see disjoint subsets.

**Pattern 2: WebDataset shards on OSS.** Pre-shard your dataset into 1-2 GB `.tar` files (e.g., `images-{00000..00099}.tar`) and let WebDataset handle sharding:

```python
import webdataset as wds
url = "oss://your-bucket/datasets/coco-shards/images-{00000..00099}.tar"
ds = (wds.WebDataset(url, shardshuffle=True, nodesplitter=wds.split_by_node)
        .decode("pil").to_tuple("jpg", "json"))
```

`split_by_node` does the per-worker shard assignment. Throughput: ~600 MB/s per worker on OSS-FUSE, vs ~80 MB/s for individual-file reads. Worth the pre-sharding step for any dataset you'll iterate more than 10 times.

**Pattern 3: CPFS for many-small-files.** If your data is millions of tiny files (genomics, time series), neither manifest nor WebDataset really wins — listing and small reads dominate. CPFS at $0.5-2/GB/month is expensive but its `listdir` and small-file random read are an order of magnitude faster than OSS-FUSE. For datasets <500 GB, often worth it.

**Pattern 4: Dynamic re-shuffling on resume.** The subtle bug: if a worker dies at step 10000 and restarts, `DistributedSampler(seed=42)` will give it the same shuffle order, but you've lost track of *where in the epoch* it was. The fix is a stateful sampler that persists `(epoch, sample_idx)` to the checkpoint:

```python
class ResumableSampler:
    def __init__(self, dataset, rank, world_size, seed=42):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.start_idx = 0   # set on resume

    def state_dict(self):
        return {"epoch": self.epoch, "start_idx": self.start_idx}
```

Without this, every restart wastes the partial epoch the worker had completed. On a 7B SFT with a 1 M-sample dataset, that's ~5% wasted compute per restart.

## What's next

Article 4 is **EAS** — taking whatever you trained and putting it behind an HTTP endpoint that auto-scales, mirrors traffic, and does not fall over at 3am. EAS is where most of your monthly Aliyun bill will live; it is worth getting right.
