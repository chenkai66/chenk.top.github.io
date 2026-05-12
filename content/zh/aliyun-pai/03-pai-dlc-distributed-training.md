---
title: "阿里云 PAI（三）：DLC 分布式训练"
date: 2026-03-07 09:00:00
tags:
  - Aliyun PAI
  - PAI-DLC
  - Distributed Training
  - LLM
  - SFT
categories: 阿里云 PAI
lang: zh
mathjax: false
series: aliyun-pai
series_title: "阿里云 PAI 实战手册"
series_order: 3
description: "在 PAI-DLC 上提交真实多卡训练任务、看懂三种资源池（灵骏、通用、抢占）、用好 AIMaster + EasyCKPT，让一台抽风节点不会让你白干一天。"
disableNunjucks: true
translationKey: "aliyun-pai-3"
---
DSW 笔记本适合单人单卡场景；一旦需要八卡、跨两节点训练，或训练时长超过你愿意为一个浏览器标签页持续守候的八小时，就该切换到 **DLC**。 DLC 是 PAI 面向托管 Kubernetes 集群的作业提交入口——你声明需求（镜像、命令、资源、数据挂载），它会自动调度 Pod、运行至完成、持久化日志并返回结果。文档称其为 *Deep Learning Containers*，日常交流中则统一简称为“DLC 任务”。

![Aliyun PAI (3): PAI-DLC — Distributed Training Without the Cluster Pain — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/03-pai-dlc-distributed-training/illustration_1.png)

## 文档到底说了啥

官方 DLC 概览列了四点，我特意挑出来讲，因为它们确实有用。

- **多样化算力** — 灵骏 AI 计算服务、ECS、ECI、神龙裸金属、灵骏裸金属，支持混合调度。
- **多种分布式任务类型** — 内置支持 Megatron、DeepSpeed、PyTorch DDP、TensorFlow PS/Worker、Slurm、Ray、MPI、XGBoost，无需自行搭建集群。
- **容错能力** — AIMaster（看门狗）、EasyCKPT（异步检查点）、SanityCheck（跑前节点健康检查）、节点自愈。
- **训练加速** — 内置框架支持数据并行、流水线并行、算子拆分、自动并行策略探索、拓扑感知调度和通信优化。

多样化算力和容错能力是 DLC 相较于自行租用 GPU ECS 的核心优势。

## 任务生命周期

一个 DLC 任务从提交到完成会经历六个阶段。

![DLC job lifecycle](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/03-pai-dlc-distributed-training/fig1_dlc_job_lifecycle.png)

其中两个阶段——**调度器放置 Pod** 和 **挂载 OSS/NAS**——是大多数“任务卡在 PENDING”工单的根源。卡在调度阶段通常是由于资源组配额耗尽；卡在挂载阶段则常因 OSS/NAS 的 RAM 角色权限配置错误。跟 DSW 一样，排查手段就是起一个带同样 OSS 挂载的微型 DSW，确认 `oss ls` 能通。

## 选资源池

你可以提交到三个池子中的一个。文档主要讲解配额和账单，实际决策取决于你的任务耐受度。

![DLC resource pools](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/03-pai-dlc-distributed-training/fig3_dlc_resource_pools.png)

对大多数团队而言，通用型按量付费已足够；但当训练规模超过 8 卡且依赖节点间 RDMA 加速时，灵骏更具成本效益。 文档称灵骏支持 RDMA 配置，即“加速节点间通信”，实际上指 NCCL AllReduce 性能可达标准以太网的 5-10 倍。抢占式实例能节省成本，前提是任务能够干净地打检查点。得益于 EasyCKPT，大多数任务都能做到这一点。

## 真实的分布式任务

下面是用四节点、每节点双卡 DLC 任务构建的拓扑：

![DLC distributed training topology](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/03-pai-dlc-distributed-training/fig2_dlc_distributed_topology.png)

这是一个最小化的 `PyTorchJob` 风格提交示例，基于 MNIST 笔记本扩展而来：

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

有几件事文档里没明说，但值得注意：

- **`$WORLD_SIZE`, `$RANK`, `$MASTER_ADDR`, `$MASTER_PORT`** 由 DLC 注入。你不需要自己发现 peer — DLC 处理 peer 发现并在容器启动前写入这些环境变量。（参见用户指南中的 "Built-in environment variables"。）
- **`fault_tolerance=True`** 会启动一个 AIMaster sidecar 监控每个 worker。如果 worker Pod 挂了， AIMaster 会标记它、请求替换，存活的 worker 会等待替换完成而不是让整个任务崩溃。这是保障持续数小时以上训练任务稳定运行的关键机制。
- **`enable_easyckpt=True`** 把 `torch.save` 换成异步路径，写入 OSS 时不阻塞训练步骤。在 70B 模型上，这能把检查点从 3 分钟卡顿变成约 10 秒的重叠开销。
- **镜像 URL 是区域特定的。** `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com` 前缀只在上海 VPC 内有效；用你区域对应的前缀，否则拉取会超时。

## 盯着它跑

控制台的“Training Jobs”页面提供任务日志、GPU利用率、网络吞吐量以及AIMaster事件。通过 SDK 你可以流式获取日志：

```python
for line in job.tail_logs(follow=True):
    print(line)
```

对于长任务，我把日志转发到 SLS （日志服务），并在 CloudMonitor 上设置 `gpu_util < 0.3` 持续 15 分钟的告警 — 这通常是数据加载阻塞或分布式初始化失败的典型表现。

## 常见故障及其真实含义

| 症状 | 真实原因 |
|---|---|
| 任务卡在 `Pending` 超过 5 分钟 | 资源组满了，或者配额耗尽。切换资源池或减少 `instance_count`。 |
| 启动时 `cannot mount oss://...` | RAM 角色缺少 `AliyunPAIAccessingOSSRole` 附件。在工作区设置里重新挂载。 |
| NCCL 在第一步开始时就挂起 | 灵骏上 RDMA 配置错误，或者节点不稳定。运行前启用 `SanityCheck` 隔离问题。 |
| 从检查点恢复后 Loss 爆炸 | EasyCKPT 保存了优化器状态但你没加载它。去读 EasyCKPT 加载助手，别用 `torch.load`。 |
| 任务结束但 `output_uri` 为空 | 训练脚本写到了 `/root` 而不是挂载的 OSS 路径。复查 `OUTPUT_DIR`。 |

## 成本现实

典型的 7B SFT（4 x A10，6小时）在通用型按量付费上，花费大概相当于上海一顿还不错的饭。 70B QLoRA （8 x A100 80 GB， 12 小时）接近杭州一个长周末的开销。如果任务能承受每隔几小时被杀掉一次，抢占式实例能省 30-50%——有了 EasyCKPT 就能做到。

## AIMaster 容错，到底重启了什么

文档说 AIMaster 做 "容错"；没说 Pod 层面具体意味着什么，这才是实际问题。这是我跑了大概 200 个任务逆向摸索出来的。

AIMaster 以 sidecar Pod 形式与训练 Pod 同驻运行，拥有独立的 ServiceAccount，具备标记失败 Pod、触发驱逐及重建 worker Pod 所需的 Kubernetes 权限。它做三件事：

1. **存活探测。** 每 5-10 秒检查每个 worker 的健康端点（AIMaster 通过 side-mount 注入容器的小型 HTTP 服务器）。检查很浅 — 进程活着、 GPU 可见、`/proc/[pid]/stack` 里没有 NCCL 死锁特征。它 *不* 深入检查训练进度；如果你的 loss 是 NaN， AIMaster 不会察觉。
2. **重启策略。** 当 worker 连续三次失败存活检查， AIMaster 标记 Pod 失败，向调度器请求在健康节点上替换（避开刚失败的那个），并等待新 Pod 启动。其他 worker 阻塞在下一次集合操作（`all_reduce` 或 `barrier`）— 它们不崩溃，只是等待。替换到位后， AIMaster 信号 "resume"， worker 重新初始化进程组。
3. **检查点协调。** 这是大多数人忽略的部分。 AIMaster 不恢复你的模型状态 — 训练脚本自己做。 AIMaster 只保证新 worker 能找到最新的检查点。如果你用了 `enable_easyckpt`，它知道哪个检查点是最新的持久化版本，并通过环境变量 `EASYCKPT_RESUME_PATH` 暴露。你的脚本应该在启动时 `if os.path.exists(EASYCKPT_RESUME_PATH): load_from(EASYCKPT_RESUME_PATH)`。

什么 *不会* 自动重启：

- **NaN loss / 训练发散。** AIMaster 不监控 loss 曲线。你的脚本需要检测这个并非零退出，这样 AIMaster 才能看到 worker 失败。
- **步骤中间的 OOM。** 如果单个 worker OOM， AIMaster 会重启它，但根本原因（单个过大的 batch）会重复，除非你在 forward pass 外包裹 try/except 并跳过坏 batch。否则 AIMaster 会重启、 OOM、重启、 OOM 进入死循环，直到 hitting 最大重试预算。
- **未触发存活检查的 hung NCCL 集合操作。** 有些 NCCL bug 让进程技术上活着但永远卡在 `cudaStreamSynchronize`。 AIMaster 通过 `/proc` 栈检查能捕获部分，但不是全部。缓解措施是设置 `NCCL_TIMEOUT=1800`（30 分钟），这样真正的挂起最终会崩溃进程， AIMaster 就能视为失败。
- **AIMaster sidecar 本身。** 如果 AIMaster 崩溃（罕见，但我见过一次工作区凭证过期的情况），就没有容错。 DLC 调度器会重启 AIMaster，但期间 worker 无人监管。

实际建议：每个 worker 设置 `max_retries=3` 并设置任务级 `max_runtime` 上限。 AIMaster + EasyCKPT 能处理大多数节点抖动失败；预算上限保护你免受需要人工介入的无限循环失败。
## Multi-node NCCL: RDMA vs TCP, ring vs tree

![Aliyun PAI (3): PAI-DLC — Distributed Training Without the Cluster Pain — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/03-pai-dlc-distributed-training/illustration_2.png)

多机训练里最大的性能杠杆，偏偏 PAI 文档里几乎没提。默认配置能跑，但调优一下，梯度通信重的任务每步耗时能减半。

**RDMA vs TCP.** 灵骏节点默认走 RDMA over Converged Ethernet (RoCE)，通用 GPU 节点走 TCP。全数据并行训练 70B 模型，实打实 8-node × 8-A100 任务上测得的 AllReduce 耗时差异：

| Communication | All-reduce 1 GB | Per-step bubble | Tokens/sec/GPU |
|---|---|---|---|
| TCP (general nodes) | 380-450 ms | ~30% | ~85 |
| RoCE (Lingjun) | 35-50 ms | ~4% | ~160 |

灵骏成本大概贵 1.5 倍，但通信密集型任务吞吐量接近翻倍。 8 卡以上数据并行，灵骏能回本。流水线并行跨阶段通信少，账得细算。

PAI 里开启 RDMA 就是个勾选框（SDK 里 `enable_rdma=True`，控制台叫"RDMA 加速"）。底层还得配几个环境变量：

```bash
# In your training command, before torchrun:
export NCCL_IB_HCA=mlx5_0          # use the mellanox HCA
export NCCL_IB_GID_INDEX=3         # RoCE v2 default
export NCCL_SOCKET_IFNAME=eth0     # the data plane interface
export NCCL_DEBUG=INFO             # one-time debug; remove for prod
```

如果 `nccl_debug=INFO` 显示 `via NET/IB` 就是 RDMA 成了，`via NET/Socket` 就是降级 TCP 了，花钱买半残性能。我遇到过节点没划进 RDMA fabric 的情况，提工单换节点。

**Ring vs tree allreduce.** NCCL 会自动选，也能强制指定：

```bash
export NCCL_ALGO=Ring   # or Tree, or Auto
```

中等 buffer （10 MB - 1 GB） Ring 带宽最优，小 buffer （<1 MB） Tree 延迟更低，比如小模型梯度同步。 70B 级别数据并行， Ring 胜出（也是默认值）。 8 节点任务如果频繁小 all_reduce （比如 Megatron 流水线微批次同步）， Tree 能快 30%。老实说：都试试，跑 100 步计时，谁快用谁。想单独基准测试，`nccl-tests` 里能看到差异。

**Topology awareness.** 灵骏节点清楚自己的 NVSwitch 拓扑， PAI 调度器尽量把 worker pod 放在相邻节点（同机架同 spine）。资源紧张调度不了时，跨 spine 通信会让 AllReduce 延迟上去。 job 级配置 `topology_constraint=spine` 强制同 spine 放置，可能调度慢点，但 4 节点以上任务值得。

## Spot / preemptible quirks: cadence and notice

抢占式实例省 30-50%，但运营细节营销材料里不讲。跑了大概 40 个抢占式任务，这几条是实战经验。

**Preemption notice window.** PAI 强杀 worker pod 前会给 30 秒 SIGTERM 通知。刚好在 checkpoint 边界能紧急 `torch.save`，步中间来不及。信号处理得这么写：

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

**Checkpoint cadence.** 没抢占时规则是“重加载 + 丢失 N 步成本 < checkpoint 成本”。有抢占时变成"M 分钟约等于平均抢占间隔”。 2026 Q1 上海 70B SFT 平均抢占间隔约 6 小时；全序列长度一步约 12 秒； EasyCKPT 存一次约 10 秒。每 100 步（20 分钟工作量）存一次，被抢最多丢 20 分钟，能接受。不用 EasyCKPT 存 100 步得堵 GPU 约 3 分钟，有效吞吐掉 15%。

**Spot eviction patterns.** 抢占不是随机的，跟时间段（业务高峰抢得狠）和 GPU 型号有关（A100 80 GB 比 A10 抢得凶）。工作时间 A100 每天 eviction 率 15-30%， A10 大概 5%。长任务（>24 h）用 A100 spot，做好被杀 3-5 次的准备。

**The retry trap.** 设个 job 级 `max_retries`（我设 5）。不然故障 spot 任务会在 pod 间反复横跳吃光预算，因为 `instance_count * preemption_rate * retry_cost > original_savings`。算笔账： 4 × A100 spot 打五折，每天每 pod 被抢 3 次就跟按量付费持平了。超过这个数，你花钱更多还训得更少。

## Dataset sharding patterns for distributed training

默认是 `DistributedSampler`，按 `(rank, world_size)` 分片，内存数据集没问题。一旦数据上 OSS 或者超内存，模式就重要了。

**Pattern 1: Index-shard a manifest file.** 写一个 manifest 文本文件，一行一个路径，排好序放 OSS。每个 worker 读自己的范围：

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

便宜，无需协调，能扩到十亿样本。分片边界按行， worker 之间数据不重叠。

**Pattern 2: WebDataset shards on OSS.** 预分片你的数据集成 1-2 GB `.tar` 包（比如 `images-{00000..00099}.tar`），让 WebDataset 处理分片：

```python
import webdataset as wds
url = "oss://your-bucket/datasets/coco-shards/images-{00000..00099}.tar"
ds = (wds.WebDataset(url, shardshuffle=True, nodesplitter=wds.split_by_node)
        .decode("pil").to_tuple("jpg", "json"))
```

`split_by_node` 负责分配 worker 分片。吞吐： OSS-FUSE 上每 worker 约 600 MB/s，单文件读只有 80 MB/s。任何要迭代 10 次以上的数据集，值得做预分片。

**Pattern 3: CPFS for many-small-files.** 如果数据是百万级小文件（基因组、时间序列）， manifest 和 WebDataset 都不占优——listing 和小读主导。 CPFS 贵（$0.5-2/GB/month），但 `listdir` 和小文件随机读比 OSS-FUSE 快一个数量级。 500 GB 以下数据集往往值得。

**Pattern 4: Dynamic re-shuffling on resume.** 有个隐蔽 bug： worker 在第 10000 步挂了重启，`DistributedSampler(seed=42)` 给的 shuffle 顺序一样，但*epoch 里走到哪了*丢了。修复方案是状态化 sampler，把 `(epoch, sample_idx)` 持久化到 checkpoint：

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

不然每次重启都浪费 worker 已完成的部分 epoch。 7B SFT 加 100 万样本数据集，每次重启浪费约 5% 算力。

## What's next

第四篇是 **EAS** —— 把训好的模型塞进 HTTP 端点，自动扩缩容，流量镜像，凌晨 3 点不崩。 EAS 是你阿里云月度账单的大头，值得搞定。