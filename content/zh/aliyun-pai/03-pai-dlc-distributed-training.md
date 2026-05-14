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
DSW 笔记本适合单人单卡的场景；一旦你需要八张 GPU 跨两个节点训练，或者训练时长超过八小时（也就是你愿意为一个浏览器标签页持续守候的极限），就该切换到 **DLC**。DLC 是 PAI 面向托管 Kubernetes 集群的作业提交入口：你只需声明需求（镜像、命令、资源规格、数据挂载路径），它就会自动调度 Pod、运行至完成、持久化日志并返回结果。官方文档称其为 *Deep Learning Containers*，但我们日常交流中统一简称为“DLC 任务”。

![阿里云PAI (3): PAI-DLC — 无需集群烦恼的分布式训练 — 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/03-pai-dlc-distributed-training/illustration_1.png)

---

## 文档到底说了啥

官方 DLC 概览列出了四点核心能力，我特意挑出来讲，因为它们确实有用：

- **多样化算力** — 支持灵骏 AI 计算服务、ECS、ECI、神龙裸金属、灵骏裸金属，并可混合调度。
- **多种分布式任务类型** — 内置支持 Megatron、DeepSpeed、PyTorch DDP、TensorFlow PS/Worker、Slurm、Ray、MPI、XGBoost，无需自行搭建和维护集群。
- **容错能力** — 包含 AIMaster（看门狗）、EasyCKPT（异步检查点）、SanityCheck（预检节点健康状态）以及节点自愈机制。
- **训练加速** — 内置框架支持数据并行、流水线并行、算子拆分、自动并行策略探索、拓扑感知调度和通信优化。

其中，“多样化算力”和“容错能力”是 DLC 相较于直接租用 GPU ECS 的真正优势所在。

## 任务生命周期

一个 DLC 任务从提交到完成会经历六个阶段。

![DLC作业生命周期](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/03-pai-dlc-distributed-training/fig1_dlc_job_lifecycle.png)

其中有两个阶段——**调度器放置 Pod** 和 **挂载 OSS / NAS**——几乎囊括了所有“任务卡在 PENDING”工单的根源。如果卡在调度阶段，通常是因为资源组已满或配额耗尽；如果卡在挂载阶段，则多半是存储所用的 RAM 角色权限配置有误。排查方法和 DSW 一样：启动一个带相同 OSS 挂载的微型 DSW 实例，确认 `oss ls` 能正常执行即可。

## 选资源池

你可以将任务提交到三个资源池之一。文档主要围绕配额和计费展开，但实际决策应基于你的任务对中断的容忍度。

![DLC资源池](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/03-pai-dlc-distributed-training/fig3_dlc_resource_pools.png)

对大多数团队而言，通用型按量付费池已是最佳选择；但当你训练规模超过 8 张 GPU 且依赖节点间 RDMA 加速时，灵骏池更具性价比。文档提到灵骏支持 RDMA 配置，所谓“加速节点间通信”，实际上是指 NCCL AllReduce 性能可达标准以太网的 5–10 倍。抢占式实例则适合能干净打检查点的任务——得益于 EasyCKPT，大多数任务都满足这一条件，因此能节省 30%–50% 的成本。

## 真实的分布式任务

下面是一个四节点、每节点双 GPU 的 DLC 任务所构建的拓扑结构：

![DLC分布式训练拓扑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/03-pai-dlc-distributed-training/fig2_dlc_distributed_topology.png)

这是一个最小化的 `PyTorchJob` 风格提交示例，由 MNIST 笔记本扩展而来：

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

有几点关键信息文档并未明确说明，但实际使用中至关重要：

- **`$WORLD_SIZE`, `$RANK`, `$MASTER_ADDR`, `$MASTER_PORT`** 由 DLC 自动注入。你无需手动实现 peer 发现——DLC 会在容器启动前完成 peer 发现，并将这些环境变量写入容器。（详见用户指南中的 “Built-in environment variables” 章节。）
- **`fault_tolerance=True`** 会启动一个 AIMaster sidecar 来监控每个 worker。若某个 worker Pod 崩溃，AIMaster 会将其标记为失败、请求调度器替换，并让其余 worker 在下一次集体通信处等待新 Pod 上线，而非直接终止整个任务。这是保障数小时以上长任务稳定运行的**最关键开关**。
- **`enable_easyckpt=True`** 会将 `torch.save` 替换为异步写入 OSS 的路径，避免阻塞训练步骤。在 70B 模型上，这能将原本长达 3 分钟的检查点卡顿压缩至约 10 秒的重叠开销。
- **镜像 URL 是区域特定的。** 例如 `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com` 前缀仅在上海 VPC 内有效；请务必使用与你所在地域匹配的前缀，否则镜像拉取会超时。

## 盯着它跑

控制台的 “Training Jobs” 页面提供日志、GPU 利用率、网络吞吐量以及 AIMaster 事件。你也可以通过 SDK 流式读取日志：

```python
for line in job.tail_logs(follow=True):
    print(line)
```

对于长时间运行的任务，我会将日志转发至 SLS（日志服务），并在 CloudMonitor 中设置告警规则：当 `gpu_util < 0.3` 持续 15 分钟时触发。这通常是数据加载阻塞或分布式初始化失败的典型信号。

## 常见故障及其真实含义

| 症状 | 真实原因 |
|---|---|
| 任务卡在 `Pending` 超过 5 分钟 | 资源组已满，或配额耗尽。建议切换资源池或减少 `instance_count`。 |
| 启动时报 `cannot mount oss://...` | RAM 角色未附加 `AliyunPAIAccessingOSSRole`。请在工作区设置中重新绑定该角色。 |
| NCCL 在第一步开始时就挂起 | 灵骏节点上 RDMA 配置错误，或节点本身不稳定。建议启用 `SanityCheck` 在任务启动前隔离问题节点。 |
| 从检查点恢复后 Loss 爆炸 | EasyCKPT 已保存优化器状态，但你的脚本未正确加载。请使用 EasyCKPT 提供的加载辅助函数，而非直接调用 `torch.load`。 |
| 任务结束但 `output_uri` 为空 | 训练脚本将输出写入了 `/root`，而非挂载的 OSS 路径。请检查 `OUTPUT_DIR` 是否配置正确。 |

## 成本现实

一次典型的 7B SFT 训练（4 × A10，6 小时）在通用型按量付费池上的花费，大约相当于在上海吃一顿不错的晚餐。而一次 70B QLoRA 训练（8 × A100 80 GB，12 小时）则接近在杭州度过一个长周末的开销。如果任务能承受每隔几小时被中断一次，抢占式实例可节省 30%–50% 的费用——有了 EasyCKPT，大多数任务都能做到这一点。

## AIMaster 容错，到底重启了什么

文档只说 AIMaster 提供“容错”，却没说明在 Pod 层面具体意味着什么——这才是工程师真正关心的问题。以下是我通过运行约 200 个任务逆向总结出的行为逻辑。

AIMaster 以 sidecar Pod 的形式与训练 Pod 共存，拥有独立的 ServiceAccount，并具备 Kubernetes 中标记、驱逐和重建 worker Pod 所需的权限。它主要执行三项任务：

1. **存活探测**：每 5–10 秒检查每个 worker 的健康端点（一个由 AIMaster 通过 side-mount 注入容器的小型 HTTP 服务器）。检查非常轻量——仅验证进程是否存活、GPU 是否可见，以及 `/proc/[pid]/stack` 中是否存在 NCCL 死锁特征。它**不会**深入监控训练进度；如果你的 loss 变成 NaN，AIMaster 并不会察觉。
2. **重启策略**：当某个 worker 连续三次未能通过存活检查，AIMaster 会将其标记为失败，向调度器申请在健康节点上创建替代 Pod（避开刚失败的节点），并等待新 Pod 启动。其余 worker 会在下一次集体操作（如 `all_reduce` 或 `barrier`）处阻塞等待，而非直接崩溃。一旦新 Pod 就绪，AIMaster 会发送“resume”信号，所有 worker 重新初始化进程组。
3. **检查点协调**：这是最容易被忽视的一环。AIMaster **不会**恢复你的模型状态——那是训练脚本的职责。它只确保新 worker 能找到最新的检查点。如果你启用了 `enable_easyckpt`，它会识别出最新且已持久化的检查点，并通过环境变量 `EASYCKPT_RESUME_PATH` 暴露路径。你的脚本应在启动时加入类似 `if os.path.exists(EASYCKPT_RESUME_PATH): load_from(EASYCKPT_RESUME_PATH)` 的逻辑。

以下情况**不会**被自动处理：

- **NaN loss 或训练发散**：AIMaster 不监控 loss 曲线。你的脚本需自行检测异常并以非零状态退出，这样 AIMaster 才能识别 worker 失败。
- **步骤中途 OOM**：若单个 worker 因 batch 过大而 OOM，AIMaster 会重启它，但根本原因仍会复现。除非你在 forward pass 外包裹 try/except 并跳过异常 batch，否则会陷入“重启 → OOM → 重启 → OOM”的死循环，直至耗尽最大重试次数。
- **未触发存活检查的 NCCL 挂起**：某些 NCCL bug 会导致进程看似存活，实则永久卡在 `cudaStreamSynchronize`。AIMaster 能通过 `/proc` 栈检查捕获部分此类问题，但并非全部。建议设置 `NCCL_TIMEOUT=1800`（30 分钟），让真正的挂起最终导致进程崩溃，从而被 AIMaster 识别为失败。
- **AIMaster sidecar 自身崩溃**：虽然罕见（我曾因工作区凭证过期遇到一次），但若 AIMaster 崩溃，容错机制将失效。DLC 调度器会尝试重启它，但在此期间 worker 处于无监管状态。

**实用建议**：为每个 worker 设置 `max_retries=3`，并为整个任务设置 `max_runtime` 上限。AIMaster + EasyCKPT 能应对大多数节点抖动问题，而重试预算则能防止无限循环类故障消耗过多资源。

## 多节点 NCCL：RDMA 与 TCP，环形与树形

![阿里云PAI (3): PAI-DLC — 无需集群烦恼的分布式训练 — 示意图](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/03-pai-dlc-distributed-training/illustration_2.png)

这是多机训练中最大的性能杠杆，偏偏 PAI 官方文档几乎只字未提。默认配置虽能运行，但稍作调优，通信密集型任务的每步耗时就能减半。

**RDMA vs TCP**：灵骏节点默认启用 RDMA over Converged Ethernet（RoCE），而通用 GPU 节点则使用 TCP。在真实 8 节点 × 8-A100、全数据并行训练 70B 模型的场景下，AllReduce 性能差异显著：

| 通信方式 | All-reduce 1 GB 耗时 | 每步空闲时间占比 | Tokens/sec/GPU |
|---|---|---|---|
| TCP（通用节点） | 380–450 ms | ~30% | ~85 |
| RoCE（灵骏） | 35–50 ms | ~4% | ~160 |

灵骏的成本约为同型号 GPU 通用节点的 1.5 倍，但在通信密集型任务中吞吐量接近翻倍。对于超过 8 卡的数据并行训练，灵骏基本能回本；而对于跨阶段通信稀疏的流水线并行，收益则需仔细核算。

在 PAI 中启用 RDMA 只需勾选一个选项（SDK 中设为 `enable_rdma=True`，控制台中名为“RDMA 加速”）。底层还需配置以下环境变量：

```bash
# In your training command, before torchrun:
export NCCL_IB_HCA=mlx5_0          # use the mellanox HCA
export NCCL_IB_GID_INDEX=3         # RoCE v2 default
export NCCL_SOCKET_IFNAME=eth0     # the data plane interface
export NCCL_DEBUG=INFO             # one-time debug; remove for prod
```

若 `nccl_debug=INFO` 输出显示 `via NET/IB`，说明已成功使用 RDMA；若显示 `via NET/Socket`，则已降级为 TCP，等于花高价买半残性能。我曾遇到节点未被正确分配至 RDMA fabric 的情况，此时需提交工单并更换节点。

**Ring vs Tree AllReduce**：NCCL 会自动选择策略，但也可手动覆盖：

```bash
export NCCL_ALGO=Ring   # or Tree, or Auto
```

对于中等大小缓冲区（10 MB – 1 GB），Ring 策略带宽效率更高；对于小缓冲区（<1 MB，如小模型梯度同步），Tree 策略延迟更低。在 70B 级别的数据并行训练中，Ring 是默认且更优的选择。但在 8 节点 Megatron 流水线任务中，若频繁进行微批次同步（即多次小规模 all_reduce），Tree 可能快 30%。最稳妥的做法是：两种都试，各跑 100 步计时，保留更快的那个。若想单独基准测试，`nccl-tests` 能清晰反映差异。

**拓扑感知调度**：灵骏节点了解自身的 NVSwitch 拓扑，PAI 调度器会尽量将 worker Pod 放置在同一机架、同一 spine 的相邻节点上。若因资源紧张无法满足，跨 spine 通信将导致 AllReduce 延迟上升。通过设置任务级参数 `topology_constraint=spine`，可强制同 spine 放置（可能延长调度等待时间），对于超过 4 节点的任务，此举通常值得。

## Spot / 抢占式实例的特性：节奏和通知

抢占式实例能节省 30%–50% 的成本，但运营细节并未出现在营销材料中。在运行约 40 个抢占式任务后，我总结出以下实战经验。

**抢占通知窗口**：PAI 会在强制终止 worker Pod 前发送 SIGTERM 信号，预留 30 秒窗口。这足够在检查点边界执行紧急 `torch.save`，但不足以在训练步中途完成完整检查点。信号处理逻辑应如下所示：

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

**检查点频率**：无抢占时，检查点间隔 N 应满足“重加载 + 丢失 N 步的成本 < 检查点开销”。有抢占时，规则变为“M 分钟 ≈ 平均抢占间隔”。以 2026 年 Q1 上海区域的 70B SFT 为例，平均抢占间隔约为 6 小时；单步训练（全序列长度）耗时约 12 秒；使用 EasyCKPT 的检查点开销约 10 秒。因此，每 100 步（约 20 分钟工作量）保存一次，最多损失 20 分钟训练进度——这是可接受的。若未使用 EasyCKPT，同样频率的检查点会阻塞 GPU 约 3 分钟，导致有效吞吐下降 15%。

**抢占规律**：抢占并非随机事件，而是与时间段和 GPU 型号强相关。业务高峰期（白天）抢占更频繁；A100 80 GB 比 A10 更容易被抢占。实测数据显示，工作时段 A100 日均抢占率为 15%–30%，而 A10 仅为 5% 左右。对于超过 24 小时的长任务，若使用 A100 抢占实例，应预期任务会被中断并重启 3–5 次。

**重试陷阱**：务必设置任务级 `max_retries`（我通常设为 5）。否则，一个行为异常的抢占任务可能在多个 Pod 间反复重启，最终消耗的总成本反而超过按量付费。简单估算：4 × A100 抢占实例（5 折）与按量付费的成本平衡点约为每天每 Pod 被抢占 3 次。超过此阈值，你不仅花得更多，实际训练时间还更少。

## 分布式训练的数据集分片模式

默认的 `DistributedSampler` 按 `(rank, world_size)` 分片，适用于内存内数据集。但当数据存储在 OSS 上或超出内存容量时，分片策略就变得至关重要。

**Pattern 1: 基于索引分片的 manifest 文件**  
将所有样本路径写入一个排序后的文本文件（每行一个路径），一次性上传至 OSS。每个 worker 读取其专属范围：

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

该方案成本低、无需协调，可轻松扩展至十亿级样本。由于分片边界按行划分，各 worker 获取的数据互不重叠。

**Pattern 2: OSS 上的 WebDataset 分片**  
将数据集预分片为 1–2 GB 的 `.tar` 文件（如 `images-{00000..00099}.tar`），并交由 WebDataset 处理分片逻辑：

```python
import webdataset as wds
url = "oss://your-bucket/datasets/coco-shards/images-{00000..00099}.tar"
ds = (wds.WebDataset(url, shardshuffle=True, nodesplitter=wds.split_by_node)
        .decode("pil").to_tuple("jpg", "json"))
```

`split_by_node` 负责为每个 worker 分配分片。吞吐表现：在 OSS-FUSE 上，单 worker 可达约 600 MB/s，而逐文件读取仅约 80 MB/s。对于需迭代 10 次以上的数据集，预分片带来的性能提升完全值得投入。

**Pattern 3: CPFS 应对海量小文件**  
若数据由百万级小文件组成（如基因组或时间序列数据），manifest 和 WebDataset 均难奏效——目录遍历和小文件读取成为瓶颈。CPFS 虽贵（约 $0.5–2/GB/月），但其 `listdir` 和小文件随机读性能比 OSS-FUSE 快一个数量级。对于小于 500 GB 的数据集，通常值得选用。

**Pattern 4: 恢复时的动态重洗牌**  
这里有个隐蔽 bug：若 worker 在第 10000 步崩溃并重启，`DistributedSampler(seed=42)` 会生成相同的 shuffle 顺序，但脚本已丢失“当前 epoch 进度”的信息。修复方法是使用状态化 sampler，将 `(epoch, sample_idx)` 持久化至检查点：

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

否则，每次重启都会浪费该 worker 已完成的部分 epoch。以 7B SFT 配合 100 万样本数据集为例，每次重启约浪费 5% 的计算资源。

## 接下来

下一篇是 **EAS** —— 把你训练好的模型部署为 HTTP 端点，支持自动扩缩容、流量镜像，并确保凌晨 3 点也不会崩。EAS 将构成你阿里云月度账单的主要部分，值得认真对待。
