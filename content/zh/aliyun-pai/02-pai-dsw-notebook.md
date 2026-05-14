---
title: "阿里云 PAI（二）：DSW Notebook 避坑指南"
date: 2026-03-06 09:00:00
tags:
  - Aliyun PAI
  - Machine Learning
  - PAI-DSW
  - Jupyter
  - GPU
categories: 阿里云 PAI
lang: zh
mathjax: false
series: aliyun-pai
series_title: "阿里云 PAI 实战手册"
series_order: 2
description: "PAI-DSW 实战：选对 GPU 镜像、把 OSS 挂好不丢权重、把官方 Quick Start 的 MNIST 完整跑通。再附上一些只在淘宝场景里踩过才知道的坑。"
disableNunjucks: true
translationKey: "aliyun-pai-2"
---
每次带新人上手 PAI，第一天的剧本都差不多：启动 DSW 实例，`pip install` 一通依赖，训练一小时，不知为何重启了 kernel，然后一脸茫然地问我模型文件去哪了。实话实说——“在 `/root` 下，但那台节点已经没了”——这种教训一次就够了。这篇文章就是让你提前避坑的版本。

![阿里云PAI (2): PAI-DSW — 不会吞噬你的权重的笔记本 — 视觉展示](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/02-pai-dsw-notebook/illustration_1.png)

---

## DSW 到底是什么

官方文档称 DSW 是**面向 AI 开发的云端 IDE**，集成了 JupyterLab、VSCode 和终端，预置了 PyTorch 和 TensorFlow 容器镜像，支持异构计算（CPU / GPU / 灵骏），并能挂载 OSS、NAS 和 CPFS 数据集。实际体验非常直接：点一下“打开”，一分钟内你就能获得一个运行在真实 GPU 上的完整 Jupyter 环境，`nvidia-smi` 正常工作，PyTorch 也能直接 import。

真正值得注意的是**镜像里其实什么都没多装**。DSW 容器的系统盘生命周期与实例完全绑定——你 `pip install` 的包能扛过 kernel 重启，但扛不住实例重启。除非你主动将 conda 环境持久化到 OSS，或通过快照功能保存到 ACR，否则一切都会随实例销毁而消失。

![DSW实例的结构](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/02-pai-dsw-notebook/fig1_dsw_anatomy.png)

## 怎么选实例类型

根据文档，DSW 资源分为两类：**公共资源**（按量付费）和**专属资源**（包年包月或灵骏）。日常开发强烈推荐公共资源——按秒计费，用多少付多少，哪怕只跑 10 分钟的实验也毫无压力。

我自己常用的配置如下：

- **微小实验 / 调试** — `ecs.gn7i-c8g1.2xlarge`（1 × A10，24 GB）。价格便宜，足够用于 4-bit 量化微调 7B 模型，或运行 512×512 分辨率的扩散模型。
- **小模型正式训练** — `ecs.gn7i-c16g1.4xlarge` 或 `ecs.gn7e-c12g1.3xlarge`（A10 / A100 40 GB）。轻松应对 CIFAR-10 上的 ResNet、ImageNet-tiny，或使用 QLoRA 微调 7B 模型。
- **LLM 开发** — `ecs.gn7e-c12g1.6xlarge` 或更高（A100 80 GB）。若需在不启用 offloading 的前提下直接加载 13–30B 模型的 BF16 权重，此配置必不可少。

> **实战建议**：如果控制台显示目标 GPU 类型“缺货”，不妨切换可用区（AZ）。库存是按 AZ 统计的，而非整个 Region。我曾亲眼见过同一分钟内，`cn-shanghai-h` 的 80 GB A100 显示无货，而 `cn-shanghai-l` 却有充足资源。

## 镜像目录

DSW 镜像均由官方维护，带版本号和清晰标签。快速入门使用的镜像是 `modelscope:1.26.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04`——这个字符串从左到右明确告诉你里面有什么：ModelScope SDK 1.26、PyTorch 2.6、GPU 版本、Python 3.11、CUDA 12.4、Ubuntu 22.04。

我几乎总是选择 `pytorch` 或 `modelscope` 镜像。TensorFlow 镜像虽然可用，但通常落后一个大版本。此外还有 `dsw-stable` 系列，专为稳定性设计，适合那些绝不希望在训练中途遭遇 CUDA 升级的生产级任务。

你也可以自行构建镜像并推送到 ACR。对于依赖复杂（如 `vllm`、`flash-attn`、自定义 CUDA kernel）的项目，这么做能省下每次新实例启动时约四分钟的 `pip install` 时间。

## 不丢数据的标准工作流

控制台操作流程如下：

![标准DSW工作流程](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/02-pai-dsw-notebook/fig2_dsw_workflow.png)

生命周期钩子很容易被忽略，但一旦忘记，代价可能很高。我默认开启 **30 分钟空闲自动关机**；同时设置 **晚上 11 点定时关机**，以防周末忘记关闭实例。每台空闲的 DSW GPU 实例每小时约 5 元，若整个周末持续运行，周一账单可能多出近 100 元。

## 数据存在哪

整个 DSW 文档中最关键的一张图：

![DSW存储布局](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/02-pai-dsw-notebook/fig3_dsw_storage_layout.png)

这是我通用的挂载路径：

```text
/mnt/data/
├── datasets/      # 只读 OSS 挂载（Bucket 永久存在）
├── checkpoints/   # 可写 OSS 前缀（每 N 步保存一次）
└── code/          # git  repo，也放在 OSS 上，新实例挂载即用
```

OSS 挂载需在创建实例时配置，文档中称为“配置存储”。选择目标 Bucket 和前缀，挂载路径设为 `/mnt/data/`，访问模式保持默认（基于 FUSE）。实例启动后，在终端执行 `oss ls oss://your-bucket/` 应能正常列出内容——这是验证“PAI ↔ OSS RAM 角色”权限是否正常的快速检查。

## 能跑的 MNIST 笔记本（直接来自快速入门）

官方快速入门使用 MNIST 手写数字识别任务。以下是简化后的最小可行训练单元格——完整版 `mnist.ipynb` 可直接从文档链接下载并上传：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

tx = transforms.Compose([transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))])
train = datasets.MNIST("/mnt/data/datasets", train=True,  download=True, transform=tx)
val   = datasets.MNIST("/mnt/data/datasets", train=False, download=True, transform=tx)

train_loader = DataLoader(train, batch_size=128, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val,   batch_size=512, shuffle=False, num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(1, 32, 3, 1), nn.Conv2d(32, 64, 3, 1)
        self.fc1, self.fc2 = nn.Linear(9216, 128), nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.c1(x)); x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

model, opt = Net().to(device), torch.optim.AdamW(Net().parameters(), lr=1e-3)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(3):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); F.cross_entropy(model(xb), yb).backward(); opt.step()
    # checkpoint to OSS, not /root
    torch.save(model.state_dict(), f"/mnt/data/checkpoints/mnist_e{epoch}.pt")
    print(f"epoch {epoch} done")
```

快速入门预期在单张 A10 上训练 3 个 epoch 后达到约 98% 的验证准确率。如果结果明显偏低，大概率是 OSS 挂载路径错误，导致读取了错误目录——并非模型本身有问题。

## 内嵌 TensorBoard

DSW 内置了 TensorBoard 扩展，文档会引导你从菜单启用。我通常直接在 notebook 中运行一个 cell：

```python
%load_ext tensorboard
%tensorboard --logdir /mnt/data/checkpoints/runs --port 6006
```

文档提示你点击的链接是 `http://localhost:6006/`——DSW 会自动代理该端口，因此你可以通过 DSW 的 URL 在浏览器中正常访问。如果提示端口“被占用”，说明同一实例中的另一个 notebook 正在使用它；此时只需重启那个 notebook 的 kernel，无需重启整个实例。

## 会话间保存环境

DSW 提供两种机制来保留环境状态，都值得掌握：

1. **实例镜像快照**：将当前容器的完整状态（包括已安装包和系统文件）打包存入 ACR。下次启动时选择该镜像，即可完全还原上次的环境。过程较慢（需几分钟），但精确无误。
2. **OSS 上的 Conda 环境**：将所有 `pip` 依赖安装到 `/mnt/data/envs/myenv/` 并激活。即使实例销毁，环境依然存在，无需重新构建。速度快，但无法保留系统级变更（如 `apt install` 安装的包）。

我做项目时默认采用 conda-on-OSS 方案；只有面对“半年后仍需原样展示的冻结 demo”时，才会使用快照机制。

## 镜像目录，别靠猜

每次启动 DSW 实例都要选择镜像，而大多数团队至少要踩一个季度的坑才能摸清门道。目前镜像目录大致分为四类：

| 家族 | 标签模式 | 最适合 | 何时避免 |
|---|---|---|---|
| `modelscope:*` | `modelscope:1.28.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04` | LLM 开发、从 ModelScope Hub 下载模型、已预装 `transformers` 和 `vllm` 的场景 | 需要比 ModelScope 提供更新的 CUDA 版本 |
| `pytorch:*` | `pytorch:2.6.0-gpu-py311-cu124-ubuntu22.04` | 原生 PyTorch 开发、自定义训练循环、厌恶依赖冲突的场景 | 需要开箱即用的 LLM 工具链 |
| `tensorflow:*` | `tensorflow:2.16.1-gpu-py311-cu123-ubuntu22.04` | TF / Keras 代码库、TFRecord 数据流水线 | 2026 年还从零开始新项目（别这么干） |
| `dsw-stable:*` | `dsw-stable:1.10-pytorch2.4-cu121` | 长期运行的 notebook，不希望季度中途 CUDA 升级 | 需要最新框架特性 |

镜像命名规则从左到右依次为：SDK 版本 → 框架版本 → CPU/GPU 标识 → Python 版本 → CUDA 版本 → 操作系统。建议像读菜谱一样解读标签。我见过最多的两类错误：

- **想要裸 PyTorch 却选了 ModelScope 镜像**：该镜像约 14 GB，首次启动需拉取 600 多个 Python 包。如果你根本用不到 `modelscope` SDK 或 `vllm`，不如省下 90 秒冷启动时间和磁盘压力。
- **想要 ModelScope 却选了 PyTorch 镜像**：结果每次新实例都要手动 `pip install vllm flash-attn modelscope`。虽然能跑通，但浪费四分钟，还可能因 `flash-attn` 编译时匹配了错误的 nvcc 而引发 CUDA 版本冲突。

对于重度依赖 `vllm` 的 Qwen3 工作流，`modelscope:1.28.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04` 是阻力最小的选择——`vllm`、`flash-attn`、`xformers`、`transformers` 均已预装且版本兼容。若是纯自定义 CUDA 研究，则建议从 `pytorch:2.6.0-gpu-py311-cu124-ubuntu22.04` 起步逐步构建。对于任何计划六个月后回溯的项目，务必在 README 中冻结镜像标签——即便标签相同，`modelscope:1.28.0-...` 六个月后也可能因阿里云定期重建和依赖修补而不再是同一个 artifact。

> **实战建议**：启动前务必在控制台查看镜像构建日期。若超过 60 天，很可能缺少至少一个 CVE 安全补丁；对于生产相关任务，请选择同一家族中的最新标签。

## OSS-FUSE 挂载、延迟 profile 以及何时该拷贝 instead

![阿里云PAI (2): PAI-DSW — 不会吞噬你的权重的笔记本 — 可视化](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/02-pai-dsw-notebook/illustration_2.png)

OSS-FUSE 是默认挂载方式，适用于 90% 的场景，但其失败模式相当隐蔽。心理模型应是：对挂载路径的每次 `read()` 都会触发一次 OSS HTTP 请求，而每次 `write()` 会先缓存在本地 buffer，直到 `close()` 或累积数 MB 后才真正刷写。这带来若干影响：

| 操作 | 本地 SSD | OSS-FUSE | NAS | CPFS |
|---|---|---|---|---|
| Sequential read 1 GB | ~250 ms | 600–1200 ms | 400–700 ms | 80–120 ms |
| Random read 4 KB (cold) | ~0.1 ms | 30–80 ms | 5–15 ms | 1–3 ms |
| `os.listdir(10000 files)` | ~10 ms | 800–2000 ms | 200–500 ms | 50–100 ms |
| Append 10 MB log | <5 ms | 200–400 ms | 50–100 ms | 10–30 ms |

具体数值会因 Region 和 Bucket 类型略有波动，但趋势一致。我总结出以下几条经验：

- **切勿对含 1 万以上文件的 OSS 挂载目录执行 `os.listdir()`**：这会触发一次 HTTP `ListObjects` 请求，而 Python 的懒迭代会让训练脚本在每个 epoch 开始时“假死”1–2 秒。建议预先生成 manifest 文件。
- **训练中不要直接向 OSS-FUSE 写 checkpoint，除非启用了 `enable_easyckpt`**：一个 7B 参数模型的 `state_dict` 约 14 GB，FUSE 会阻塞训练进程 10–30 秒，导致 GPU 空转。要么先拷贝到本地 SSD 再异步上传，要么使用 EasyCKPT（见[第 3 章](/zh/aliyun-pai/03-pai-dlc-distributed-training/)）。
- **训练数据超过 100 GB 时，不要直接从 OSS-FUSE 读取**：单个 FUSE 挂载点的带宽上限约为 200–400 MB/s，极易成为数据加载瓶颈。建议启动时用 `ossutil cp -r --jobs 8` 将数据拷至本地 SSD，再从本地训练。若数据集大于本地 SSD 容量，则改用 NAS 或 CPFS。
- **代码、配置文件和输出目录可放心使用 OSS-FUSE**：偶尔读取的延迟可以接受，只要不在热循环中频繁写入，开销并不显著。

其他挂载方式也值得关注：

```python
# When you create a DSW instance, "Configure storage" lets you pick:
mount_modes = {
    "oss-fuse":   "default — POSIX-ish, lazy fetch, fine for code/config",
    "oss-direct": "skip FUSE; use ossutil/oss SDK from your code",
    "nas":        "real POSIX, paid per-GB-month, good for shared scratch",
    "cpfs":       "HPC throughput, expensive, reach for it on >50 GB/s aggregate",
}
```

我的默认策略是统一使用 OSS-FUSE，但在任何训练任务开始前加入一个 `local_copy_step()`，将热点数据复制到 `/root/data/`（系统盘，高速 SSD，虽为临时存储但无妨——因为 OSS 才是唯一真相源）。这一招能将 50 GB 图像数据集的加载时间从流式读取的约 15 分钟缩短至拷贝后的约 3 分钟。

## 空闲 shutdown、自动扩容和 GPU 共享

这三个生命周期功能，只要你忘过一次，它们省下的钱就值回票价。

**空闲 shutdown**：可按实例配置。平台监控 CPU、GPU 和网络活动；若三项指标持续 N 分钟低于阈值，实例将自动停止。我的默认设置是：个人开发设 30 分钟，与队友共享的实例设 15 分钟，长期训练任务则关闭空闲 shutdown（改用定时 shutdown）。算笔账：一台 A10 实例约 5 元/小时，若长周末忘记关闭，账单可能增加 360 元；而设 30 分钟空闲 shutdown 仅损失 2.5 元，其余全部节省。注意一个陷阱：屏幕共享时若 TensorBoard 图表每分钟更新一次，可能骗过活跃检测。如需强制保活，可在终端运行 `nvidia-smi -l 1`。

**自动扩容**：知道的人不多。你可以停止实例，更换 GPU 类型后重启，而 `/mnt/data/` 挂载和持久化到 OSS 的 conda 环境不会丢失。例如，我会在半小时负载测试时升配到 A100 80 GB，次日开发再降回 A10。虽然是手动“停止 → 修改 → 启动”，但因状态已持久化，整个过程只需 2 分钟，无需重新配置环境。

**GPU 共享（cGPU）**：最新功能，容易被忽略。PAI 支持通过 cGPU 虚拟化将一块物理 GPU 分配给两个 DSW 实例，按需划分显存和算力（例如将 A10 的 24 GB 拆为 16 GB + 8 GB）。适合初级工程师仅需 <8 GB 显存做推理的场景，无需单独为其分配整卡。代价是 cGPU 会引入 5–15% 的性能开销，且租户间隔离仅为“尽力而为”——切勿与不可信用户共享。该功能需在工作空间的 *Resource sharing* 中启用，之后会在 DSW 实例类型选择器中出现新选项。

这三项功能组合使用，曾让我上一个团队的开发环境 GPU 账单直接减半。任何新 workspace 都应在第一天就配置妥当。

## Snapshot vs 自定义镜像 vs git pull——选对方案

有三种方式能让“我起了个新 DSW 实例”感觉像“我回到了上次离开的地方”，每种都有其适用场景。

**从 `/mnt/data/code/` 执行 `git pull`** 是管理源代码的唯一正确方式。永远如此。代码存于 OSS，首次实例克隆仓库，后续每次启动只需 `git pull`。这种方式不受实例销毁影响，是唯一能与代码审查流程集成的方案，且完全免费。如果你的工作流尚未引入 Git，请先解决这个问题再继续阅读。

**OSS 上的 Conda 环境** 是管理 Python 依赖的最佳实践。在 `/mnt/data/envs/myenv/` 下创建环境并激活，所有 `pip install` 的包都会持久化到 OSS。实例销毁后环境依然可用。启动时比本地环境慢约 30 秒（因 FUSE 导致 conda activate 扫描变慢），但无需重建，也无缓存失效问题。限制在于无法通过此方式安装系统级包（如 `apt install`），这些仍会随实例消失。

**实例镜像 snapshot** 适用于包含非 Python 状态的场景，例如系统包（`libnuma`、自定义 CUDA 库、特殊 C++ 依赖）、ACR 管理的内核模块，或 `/etc/` 下的任何配置。Snapshot 会将整个容器文件系统冻结到 ACR；下次启动时选择该镜像，即可完全复现上次状态。过程较慢（制作需 3–8 分钟，拉取额外 1–2 分钟），但精确可靠。我仅在两种狭窄场景使用：(a) 半年后需原样展示的冻结 demo；(b) 花了一整天调试才达成兼容的 CUDA 栈，绝不想重来。

**带 Dockerfile 的自定义 ACR 镜像** 是团队级可复现性的首选。在 CI 中构建，打上日期标签，推送到 ACR，确保每位成员使用完全相同的环境。对于超过两名贡献者的项目，我默认采用此方案——snapshot 路径会让“到底装了什么”变得不透明，而 Dockerfile 可审查、可版本控制。代价是每次变更需额外 5 分钟 CI 时间，并需维护 Dockerfile，但绝对值得。

决策树如下：

- 仅有代码变更？→ `git pull`  
- 代码 + 少量仅供个人使用的 `pip install`？→ Conda on OSS  
- 重做整个环境会让你崩溃？→ Snapshot  
- 多人共用该环境？→ Custom ACR image  

最常见的错误是：默认所有内容都用 snapshot，包括本该纳入 Git 的代码。结果得到一个冻结镜像，其中硬编码了 `/root/notebooks/foo.ipynb`，却无法追溯自上个季度以来的任何变更。

## 下一步

第 3 篇文章将使用同一个 MNIST 任务，展示通过 DLC 跨多 GPU 和多节点扩展时会发生哪些变化——包括文档中提及但未深入解释的 AIMaster 容错机制。
