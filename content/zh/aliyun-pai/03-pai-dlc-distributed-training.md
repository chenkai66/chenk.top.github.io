---
title: "阿里云 PAI 实战（三）：PAI-DLC——不用通宵刨坑的分布式训练"
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
DSW Notebook 更适合单个工程师使用单张 GPU 的场景。如果你需要在两台机器上调度 8 张 GPU，或者训练时间超过你愿意开着浏览器等待的 8 小时，那么就该切换到 **DLC** 了。DLC 是 PAI 提供的一个任务提交前端，基于托管的 Kubernetes 集群：你只需要描述清楚需求（比如镜像、命令、资源、数据挂载等），DLC 就会负责调度 Pod，执行任务直到完成，并保存日志，最后向你汇报运行结果。官方文档中称其为 *Deep Learning Containers*，但日常交流中我们更习惯叫它“DLC 任务”。

![阿里云 PAI 实战（三）：PAI-DLC——分布式训练不再头疼 —— 视觉化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/03-pai-dlc-distributed-training/illustration_1.png)
## 文档到底说了什么

在官方的 DLC 概览中，有几个关键特性值得特别关注，因为它们直接影响使用体验：

- **丰富的计算资源** — 提供灵骏 AI 智算服务、ECS、ECI、神龙裸金属以及灵骏裸金属等多种选择，并支持异构混合调度，灵活满足不同需求。
- **多样的分布式任务支持** — 内置对 Megatron、DeepSpeed、PyTorch DDP、TensorFlow PS/Worker、Slurm、Ray、MPI 和 XGBoost 等框架的支持，省去了自建集群的麻烦。
- **强大的容错能力** — 包括 AIMaster（看门狗机制）、EasyCKPT（异步检查点功能）、SanityCheck（节点健康预检）以及节点自愈能力，确保任务稳定运行。
- **高效的训练加速** — 内置优化框架，支持数据并行、流水线并行、算子拆分、自动并行策略探索、拓扑感知调度以及高性能通信优化。

其中，第一点“丰富的计算资源”和第三点“强大的容错能力”，正是 DLC 相较于直接租用 GPU ECS 的核心优势所在。
## 任务生命周期

一个 DLC 任务从提交到完成会经历六个阶段：

![DLC 任务生命周期](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/03-pai-dlc-distributed-training/fig1_dlc_job_lifecycle.png)

其中有两个阶段——**调度 Pod 分配** 和 **挂载 OSS/NAS**——是绝大多数“我的任务卡在 PENDING 状态”问题的根源。如果任务卡在调度阶段，通常是因为资源组已满；如果卡在挂载阶段，则可能是存储相关的 RAM 角色配置有问题。与 DSW 类似，排查这类问题的方法是启动一个小型的 DSW 实例，使用相同的 OSS 挂载配置，然后运行 `oss ls` 命令确认是否能够正常访问。
## 选择资源池

在提交任务时，你需要从三种资源池中选择一个。虽然文档主要讨论的是配额和账单，但实际的选择标准更多取决于你的任务对资源波动的容忍度。

![DLC 资源池](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/03-pai-dlc-distributed-training/fig3_dlc_resource_pools.png)

对于大多数团队来说，通用算力搭配按量付费是最合适的选择。如果你的任务需要使用 8 张以上的 GPU，并且节点间需要 RDMA（远程直接内存访问）支持，那么灵骏会是更好的选择。文档提到，灵骏支持配置 RDMA，能够“加速节点间通信”，简单来说就是 NCCL AllReduce 的性能比普通以太网快 5 到 10 倍。而抢占式实例则适合那些可以定期保存检查点（checkpoint）的任务，得益于 EasyCKPT 的支持，绝大多数任务都能很好地适配这种模式，从而有效降低成本。
## 一个真实的分布式任务

以下是你在四节点、每节点 2 张 GPU 的 DLC 分布式任务中构建的拓扑结构：

![DLC 分布式训练拓扑](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/03-pai-dlc-distributed-training/fig2_dlc_distributed_topology.png)

通过 SDK 提交的一个最小化 `PyTorchJob` 示例，基于 MNIST Notebook 横向扩展而来：

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
    instance_count=4,                # 4 个 worker 节点
    instance_type="ecs.gn7i-c16g1.4xlarge",  # 每节点 2 × A10
    datasets={"train": "oss://your-bucket/datasets/mnist"},
    code_uri="oss://your-bucket/code/mnist-ddp.zip",
    output_uri="oss://your-bucket/runs/mnist-ddp/",
    fault_tolerance=True,            # 开启 AIMaster
    enable_easyckpt=True,            # 启用异步 checkpoint
)
job.submit(wait=False)
print(job.id, job.status)
```

以下是几个从文档中不容易一眼看出的关键细节：

- **`$WORLD_SIZE`、`$RANK`、`$MASTER_ADDR`、`$MASTER_PORT`** 这些环境变量由 DLC 自动注入。你无需手动进行节点发现——DLC 会在容器启动前完成节点发现并设置好这些变量。（更多内容可参考用户指南中的“内置环境变量”部分。）
- **`fault_tolerance=True`** 会启动一个 AIMaster 辅助容器，用于监控每个 worker 的状态。如果某个 worker pod 意外退出，AIMaster 会标记该节点并申请替换，同时其他存活的 worker 会等待新节点加入，而不会导致整个任务崩溃。对于运行时间超过数小时的任务来说，这是**最重要的配置开关**。
- **`enable_easyckpt=True`** 将默认的 `torch.save` 替换为异步存储路径，避免阻塞训练步骤，直接将 checkpoint 写入 OSS。在 70B 参数量的模型上，这一优化将 checkpoint 的耗时从原来的 3 分钟卡顿缩短到仅约 10 秒的重叠时间。
- **镜像 URL 是区域相关的。** `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com` 前缀仅适用于上海 VPC 区域。如果你在其他区域运行任务，请确保使用对应区域的镜像地址，否则镜像拉取会超时。
## 查看任务运行情况

在控制台的 "Training Jobs" 界面中，你可以查看日志、GPU 使用率、网络吞吐量以及 AIMaster 事件。如果需要实时获取日志，可以通过 SDK 实现流式输出：

```python
for line in job.tail_logs(follow=True):
    print(line)
```

对于耗时较长的任务，我通常会将日志转发到 SLS（日志服务），并在 CloudMonitor 中设置告警规则：当 `gpu_util < 0.3` 持续 15 分钟时触发告警。这通常是数据加载或分布式初始化出现问题的典型信号。
## 常见问题及其背后的原因

| 现象 | 可能原因与解决办法 |
|---|---|
| 任务一直处于 `Pending` 状态超过 5 分钟 | 资源池可能已满，或者您的配额已经用完。可以尝试切换资源池，或者减少 `instance_count` 的值。 |
| 启动时提示 `cannot mount oss://...` | RAM 角色未绑定 `AliyunPAIAccessingOSSRole` 权限。请前往工作空间设置中重新绑定该角色。 |
| 第一步训练时 NCCL 卡住 | 可能是灵积（Lingjun）的 RDMA 配置有问题，或者是节点不稳定。建议启用 `SanityCheck` 功能，在运行前进行问题排查和隔离。 |
| 从 checkpoint 恢复后 loss 值突然飙升 | EasyCKPT 已保存优化器状态（optimizer state），但您未正确加载。请使用 EasyCKPT 提供的加载工具，而不是直接调用 `torch.load`。 |
| 任务完成后 `output_uri` 为空 | 训练脚本可能将输出写入了 `/root` 目录，而非挂载的 OSS 路径。请检查并确保 `OUTPUT_DIR` 配置正确。 |
## 成本现状

如果你用通用型按量付费的资源来运行一个典型的 7B SFT（4 × A10，6 小时），花费大概相当于在上海吃顿不错的晚餐。而如果是 70B QLoRA（8 × A100 80G，12 小时），成本则接近在杭州度过一个长周末的花销。如果任务能够容忍每几个小时被中断一次，使用抢占式实例可以帮你节省 30%-50% 的成本——有了 EasyCKPT 的加持，大多数任务都能顺利应对这种中断。
## AIMaster 的容错机制：到底重启了什么？

文档里提到 AIMaster 提供“容错”功能，但并没有具体说明在 pod 层面它究竟做了什么——而这才是大家真正关心的问题。以下是我在运行了大约 200 个任务后总结出的细节。

AIMaster 以 sidecar pod 的形式运行在你的训练 pod 旁边，拥有自己的 ServiceAccount，并具备足够的 K8s 权限来标记、驱逐和重建 worker pod。它的主要职责可以分为以下三部分：

1. **存活探测。**  
   每隔 5 到 10 秒，AIMaster 会检查每个 worker 的健康状态端点（这是一个通过 side-mount 注入到容器中的小型 HTTP 服务）。检查的内容很简单：进程是否存活、GPU 是否可见、`/proc/[pid]/stack` 中是否存在 NCCL 死锁的特征。需要注意的是，这种检查并不深入——如果你的 loss 变成 NaN，AIMaster 是不会察觉的。

2. **重启策略。**  
   如果某个 worker 连续三次未能通过存活检查，AIMaster 会将该 pod 标记为失败，向调度器申请在另一台健康的节点上创建替补 pod（避开刚刚发生故障的节点），并等待新 pod 启动完成。在此期间，其他 worker 会在下一个集合通信操作（如 `all_reduce` 或 `barrier`）处阻塞——它们不会崩溃，而是耐心等待。当替补 pod 就位后，AIMaster 会发出“恢复”信号，worker 们重新初始化进程组。

3. **Checkpoint 协调。**  
   这一点很容易被忽略。AIMaster 并不负责恢复模型的状态——这是你的训练脚本的任务。AIMaster 的作用是确保新启动的 worker 能找到最新的 checkpoint。如果你启用了 `enable_easyckpt`，它会记录最新的持久化 checkpoint，并通过环境变量 `EASYCKPT_RESUME_PATH` 暴露出来。因此，你的脚本需要在启动时加入如下逻辑：  
   ```python
   if os.path.exists(EASYCKPT_RESUME_PATH):
       load_from(EASYCKPT_RESUME_PATH)
   ```

### 哪些情况不会自动重启？

- **NaN loss 或训练发散。**  
  AIMaster 不会监控你的 loss 曲线。如果出现 NaN 或训练发散的情况，你的脚本需要自行检测，并以非零退出码终止，这样 AIMaster 才能识别该 worker 为失败状态。

- **训练中途 OOM（内存不足）。**  
  如果某个 worker 因为 OOM 挂掉，AIMaster 会尝试重启它。然而，如果根本原因（例如一个超大的 batch）没有解决，问题会反复出现。建议在 forward 过程中加入 try/except 逻辑，跳过导致问题的 batch。否则，AIMaster 会陷入“重启 → OOM → 重启 → OOM”的死循环，直到达到最大重试次数。

- **挂起但未触发存活检查的 NCCL 集合通信。**  
  某些 NCCL bug 会导致进程虽然技术上仍然存活，但却永远卡在 `cudaStreamSynchronize` 中。AIMaster 通过 `/proc` 栈检查能够捕获部分此类问题，但并非全部。缓解方法是设置 `NCCL_TIMEOUT=1800`（30 分钟），这样真正的挂起最终会导致进程崩溃，AIMaster 也能识别为失败。

- **AIMaster sidecar 自身崩溃。**  
  这种情况很少见，但我曾遇到过一次——某个工作空间的凭证过期导致 AIMaster 崩溃。此时，容错机制会失效。虽然 DLC 调度器会重启 AIMaster，但在重启完成之前，worker 处于无人监管的状态。

### 实践建议

- 为每个 worker 设置 `max_retries=3`，并在任务级别设置 `max_runtime` 上限。  
- AIMaster + EasyCKPT 能够处理大多数因节点不稳定导致的故障；而预算上限则可以保护你免受无限循环类问题的影响——这类问题通常需要人工介入排查。
## 多机 NCCL：RDMA 与 TCP，Ring 与 Tree 的选择

![阿里云 PAI 实战（三）：PAI-DLC——分布式训练的轻松之道 — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/03-pai-dlc-distributed-training/illustration_2.png)

在多机训练中，有一个性能优化的关键点，但 PAI 的官方文档对此几乎没有提及。默认配置虽然能用，但如果针对通信密集型任务进行调优，每步耗时可以减少一半。

### RDMA 与 TCP 的对比

灵骏节点默认使用 RDMA over Converged Ethernet（RoCE），而通用 GPU 节点则使用 TCP。在一个实际的 8 节点 × 8 张 A100 训练 70B 模型的全数据并行任务中，AllReduce 阶段的性能差异如下：

| 通信方式       | AllReduce 1 GB | 每步 Bubble | Tokens/秒/GPU |
|----------------|----------------|-------------|---------------|
| TCP（通用节点） | 380-450 ms     | ~30%        | ~85           |
| RoCE（灵骏）   | 35-50 ms       | ~4%         | ~160          |

灵骏节点的价格大约是通用节点的 1.5 倍（同 GPU 型号），但对于通信密集型任务，吞吐量几乎翻倍。如果任务涉及超过 8 卡的数据并行，灵骏的性价比显而易见；而对于流水线并行这种跨阶段通信稀疏的场景，性价比则相对接近。

在 PAI 中启用 RDMA 很简单，只需在提交任务时勾选（SDK 中设置 `enable_rdma=True`，控制台中选择“RDMA 加速”）。此外，还需要在容器内配置以下环境变量：

```bash
# 在训练命令中、torchrun 之前：
export NCCL_IB_HCA=mlx5_0          # 使用 Mellanox HCA
export NCCL_IB_GID_INDEX=3         # RoCE v2 默认值
export NCCL_SOCKET_IFNAME=eth0     # 数据平面网卡
export NCCL_DEBUG=INFO             # 一次性调试，生产环境请移除
```

如果 `nccl_debug=INFO` 输出显示 `via NET/IB`，说明成功启用了 RDMA；如果显示 `via NET/Socket`，则说明回退到了 TCP——这意味着你付了全价却只得到了一半的性能。这种情况通常是因为某些节点未分配到 RDMA 网络中，建议提交工单更换节点。

### Ring 与 Tree AllReduce 的选择

NCCL 会自动选择算法，但你也可以手动指定：

```bash
export NCCL_ALGO=Ring   # 或 Tree、Auto
```

对于中等大小的缓冲区（10 MB - 1 GB），Ring 算法带宽最优；而对于小缓冲区（<1 MB，例如小模型的梯度同步），Tree 算法延迟更低。在 70B 级别的数据并行任务中，Ring 是默认且更优的选择。然而，对于 8 节点多频次小规模 AllReduce 的任务（如 Megatron 流水线中频繁的微批次同步），Tree 可能快 30%。最实在的建议是：两种算法各跑 100 步计时，择优选用。如果想单独测试，可以运行 `nccl-tests`。

### 拓扑感知的重要性

灵骏节点能够感知自身的 NVSwitch 拓扑，PAI 的调度器也会尽量将 worker pod 分配到相邻的节点上（如同一机架或同一 spine）。然而，在资源紧张的情况下，调度器可能无法满足这一条件，导致跨 spine 流量增加，AllReduce 延迟上升。此时，可以通过任务级配置 `topology_constraint=spine` 强制将任务限制在同一 spine 内，但这可能会带来一定的调度延迟。对于超过 4 节点的任务来说，这种权衡通常是值得的。
## 抢占式实例（Spot）的使用细节：抢占节奏与通知机制

抢占式实例虽然能节省 30%-50% 的成本，但其运维细节往往不会出现在官方文档或宣传材料中。在实际运行了大约 40 个抢占式任务后，我总结了一些关键点，供资深工程师参考。

---

### 抢占通知窗口

PAI 平台会在强制终止实例前通过 `SIGTERM` 信号提前 30 秒通知 worker pod。这段时间足够在 checkpoint 边界完成一次紧急的 `torch.save` 操作，但如果需要在中途保存一个完整的 checkpoint，则时间远远不够。以下是一个简单的信号处理实现：

```python
import signal, os, threading

_preempt = threading.Event()

def _on_term(signum, frame):
    _preempt.set()
    print("[抢占通知] 收到 SIGTERM 信号，将在下一个步边界保存 checkpoint")

signal.signal(signal.SIGTERM, _on_term)

# 在训练循环中：
for step, batch in enumerate(loader):
    train_step(model, batch)
    if _preempt.is_set() or step % CKPT_EVERY == 0:
        save_easyckpt(model, optimizer, step)
        if _preempt.is_set():
            sys.exit(0)   # 干净退出；AIMaster 在 30 秒内不会重试
```

---

### Checkpoint 的频率设计

在没有抢占的情况下，checkpoint 的频率通常遵循“每 N 步保存一次，确保重新加载并丢失 N 步的成本低于保存 checkpoint 的开销”。但在抢占式实例中，这一规则需要调整为“每 M 步保存一次，其中 M 分钟接近平均的抢占间隔”。

以 2026 年第一季度在 cn-shanghai 区域运行 70B 参数规模的 SFT 训练为例，平均抢占间隔约为 6 小时；每步（满序列长度）耗时约 12 秒；使用 EasyCKPT 保存一次 checkpoint 需要约 10 秒。因此，设置每 100 步保存一次 checkpoint（相当于 20 分钟的工作量），意味着即使发生抢占，最多只会丢失 20 分钟的进度——这是一个可以接受的损失。

如果不使用 EasyCKPT，同样的 100 步保存频率会导致每次保存阻塞 GPU 约 3 分钟，从而使得有效吞吐量下降 15%。

---

### 抢占模式的规律

抢占并不是随机发生的，而是与时间和 GPU 类型密切相关：

- **时间因素**：工作时段（如白天）的抢占频率更高，因为此时按需实例的需求量较大。
- **GPU 类型**：A100 80 GB 的抢占频率显著高于 A10。根据我的观察，在工作时段，A100 的日均抢占率约为 15%-30%，而 A10 的抢占率则接近 5%。

对于在 A100 Spot 实例上运行的长时间任务（超过 24 小时），预计任务会被中断并重启 3-5 次。

---

### 重试陷阱

务必为任务设置一个合理的 `max_retries` 参数（我个人建议设为 5）。如果没有设置，表现不佳的 Spot 任务可能会在多个 Pod 之间反复重启，最终耗尽预算。这种情况的发生可以用以下公式解释：

$$
\text{instance\_count} \times \text{preemption\_rate} \times \text{retry\_cost} > \text{original\_savings}
$$

以 4 张 A100 Spot 实例为例，假设折扣为 50%，那么每张卡每天被抢占 3 次时刚好达到盈亏平衡点。如果超出这个频率，不仅实际训练时间减少，总成本甚至会**高于按需实例**。

---

以上内容希望能帮助你在使用抢占式实例时更好地规划任务，避免不必要的资源浪费和性能损失。
## 分布式训练中的数据分片模式

默认情况下，`DistributedSampler` 会根据 `(rank, world_size)` 对数据进行分片，这对内存中的数据集来说完全够用。但当数据存储在 OSS 上或者数据量超出内存容量时，选择合适的分片模式就变得至关重要。

**模式 1：基于清单文件的索引分片**  
首先生成一个排序好的文本文件，每行记录一个数据路径，并将其存储到 OSS 中。每个 worker 只需读取属于自己的那一部分数据范围即可：

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

这种方式成本低、无需额外协调，且能够轻松扩展到数十亿样本规模。由于分片粒度是按行划分的，各个 worker 的数据子集互不重叠。

**模式 2：OSS 上的 WebDataset 分片**  
如果数据集较大，可以预先将其切分为 1-2 GB 的 `.tar` 文件（例如 `images-{00000..00099}.tar`），然后交给 WebDataset 来处理分片逻辑：

```python
import webdataset as wds
url = "oss://your-bucket/datasets/coco-shards/images-{00000..00099}.tar"
ds = (wds.WebDataset(url, shardshuffle=True, nodesplitter=wds.split_by_node)
        .decode("pil").to_tuple("jpg", "json"))
```

其中，`split_by_node` 负责为每个 worker 分配对应的分片。性能方面，通过 OSS-FUSE 每个 worker 的吞吐量可达 ~600 MB/s，而单文件读取仅 ~80 MB/s。对于需要迭代 10 次以上的数据集，这种预分片的方式非常值得推荐。

**模式 3：海量小文件场景下的 CPFS**  
如果你的数据集由数百万个小文件组成（如基因组数据或时间序列数据），无论是清单文件还是 WebDataset 都难以胜任——因为此时的主要开销集中在文件列表查询和小文件读取上。虽然 CPFS 的存储成本较高（$0.5-2/GB/月），但它的 `listdir` 和小文件随机读取性能比 OSS-FUSE 快一个数量级。对于小于 500 GB 的数据集，使用 CPFS 往往是划算的。

**模式 4：恢复训练时的动态重洗牌**  
这里有一个容易被忽视的问题：假设某个 worker 在第 10000 步时崩溃并重启，尽管 `DistributedSampler(seed=42)` 会为其提供相同的洗牌顺序，但你已经丢失了该 worker 在当前 epoch 中的进度信息。解决方法是实现一个有状态的采样器，将 `(epoch, sample_idx)` 保存到 checkpoint 中：

```python
class ResumableSampler:
    def __init__(self, dataset, rank, world_size, seed=42):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.epoch = 0
        self.start_idx = 0   # 恢复时设置

    def state_dict(self):
        return {"epoch": self.epoch, "start_idx": self.start_idx}
```

如果不解决这个问题，每次重启都会浪费 worker 已经完成的部分 epoch。以一个包含 100 万样本的数据集进行 7B 参数的 SFT 训练为例，每次重启大约会浪费 5% 的计算资源。
## 下一篇

第四篇是 **EAS** —— 把训完的东西塞到一个会自动扩缩、能镜像流量、半夜不会挂的 HTTP 端点后面。EAS 是你阿里云月账单的大头，值得花时间做对。
