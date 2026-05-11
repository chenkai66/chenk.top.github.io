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
每次带新人上手 PAI，第一天的剧本都差不多。启动 DSW 实例，`pip install` 一通，训练一小时，不知为何重启了 kernel，然后问我模型文件去哪了。实话实说——“在 `/root` 下，但那台节点已经没了”——这种教训一次就够了。这篇文章就是让你提前避坑的版本。

![Aliyun PAI (2): PAI-DSW — Notebooks That Don't Eat Your Weights — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/02-pai-dsw-notebook/illustration_1.png)

## DSW 到底是什么

官方文档说 DSW 是**面向 AI 开发的云端 IDE**，集成了 JupyterLab、VSCode 和终端，预配了 PyTorch 和 TensorFlow 容器镜像，支持异构计算（CPU / GPU / 灵骏），还能挂载 OSS、NAS 和 CPFS 数据集。实际操作起来，就是点一下“打开”，一分钟内你就能拿到一个真正的 Jupyter，跑在真正的 GPU 上，`nvidia-smi` 正常，PyTorch 直接 import。

有意思的是**镜像里其实什么都没多装**。DSW 容器的系统盘生命周期与实例一致。你 `pip install` 的东西能扛过 kernel 重启，但扛不住实例重启，除非你把 conda 环境持久化到 OSS，或者通过快照功能存到 ACR。

![Anatomy of a DSW instance](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/02-pai-dsw-notebook/fig1_dsw_anatomy.png)

## 怎么选实例类型

文档里 DSW 资源分两类：**公共资源**（按量付费）和**专属资源**（包年包月或灵骏）。日常开发选用公共资源即可：按实际使用的 GPU 秒数计费，用多少付多少；运行 10 分钟的实验，可随时启停，毫无压力。

我自己怎么选：

- **微小实验 / 调试** — `ecs.gn7i-c8g1.2xlarge` (1 × A10, 24 GB)。便宜，微调 7B 模型开 4-bit 量化或者跑 512×512 的扩散模型绰绰有余。
- **小模型正式训练** — `ecs.gn7i-c16g1.4xlarge` 或 `ecs.gn7e-c12g1.3xlarge` (A10 / A100 40 GB)。运行 CIFAR-10 上的 ResNet、ImageNet-tiny 或 7B 模型的 SFT + QLoRA 训练都很流畅。
- **LLM 开发** — `ecs.gn7e-c12g1.6xlarge` 或更高 (A100 80 GB)。若需在不启用 offloading 的前提下直接加载 13–30B 的 BF16 模型，此配置为必需。

> **实战建议：** 如果控制台显示想要的 GPU 类型“缺货”，换个可用区（AZ）。库存是按 AZ 算的，不是按 Region。我见过同一分钟内 `cn-shanghai-h` 的 80 GB A100 没货，但 `cn-shanghai-l` 随便用。

## 镜像目录

DSW 镜像都是官方维护、带版本号和标签的。快速入门用的是 `modelscope:1.26.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04`——该镜像标签直观体现了其技术栈组成。从左往右读：ModelScope SDK 1.26，PyTorch 2.6，GPU 版，Python 3.11，CUDA 12.4，Ubuntu 22.04。

我基本只选 `pytorch` 或 `modelscope` 镜像。TensorFlow 镜像没问题，但版本总慢半拍。还有个 `dsw-stable` 系列，设计上保持版本稳定，适用于不希望在训练过程中遭遇 CUDA 升级的生产环境任务。

你也可以自己打包镜像推到 ACR。对于那些依赖树特别重的项目（`vllm`, `flash-attn`, 自定义 CUDA kernel），这么做每次新实例启动能省四分钟 `pip install`。

## 不丢数据的标准工作流

控制台操作流程如下：

![Standard DSW workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/02-pai-dsw-notebook/fig2_dsw_workflow.png)

生命周期钩子容易忽略，但若忽略此项，将带来显著成本损失。**空闲 shutdown** 我默认设 30 分钟；**定时 shutdown** 设晚上 11 点，防止周末忘了关笔记本。每台空闲的 DSW GPU 实例每小时计费约 5 元，若周末全程未关闭，周一可能产生约 100 元费用。

## 数据存在哪

整个 DSW 文档里最重要的一张图：

![DSW storage layout](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/02-pai-dsw-notebook/fig3_dsw_storage_layout.png)

这是我通用的挂载路径：

```
/mnt/data/
├── datasets/      # 只读 OSS 挂载（Bucket 永久存在）
├── checkpoints/   # 可写 OSS 前缀（每 N 步保存一次）
└── code/          # git  repo，也放在 OSS 上，新实例挂载即用
```

挂载 OSS 在创建实例时配置，文档里叫“配置存储”。选 Bucket 和前缀，挂载路径选 `/mnt/data/`，访问模式默认（基于 FUSE）。启动后，终端里跑 `oss ls oss://your-bucket/` 应该能通——这是检查"PAI ↔ OSS RAM role"权限的健康测试。

## 能跑的 MNIST 笔记本（直接来自快速入门）

官方快速入门用的是 MNIST 手写数字识别。这是最小可行训练单元格，为了文章简化过——文档链接里有完整的 `mnist.ipynb` 可以直接上传：

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

快速入门预期在单张 A10 上跑 3 个 epoch 后验证准确率大概 98%。如果看到数字低得离谱，大概率是 OSS 挂载错了，读错了目录——不是模型 bug。

## 内嵌 TensorBoard

DSW 内置了 TensorBoard 扩展；文档会教你从菜单启用。我通常直接跑个 cell：

```python
%load_ext tensorboard
%tensorboard --logdir /mnt/data/checkpoints/runs --port 6006
```

文档让你点的链接是 `http://localhost:6006/`——DSW 会代理这个端口，所以你在浏览器通过 DSW URL 就能访问。如果提示端口“被占用”，说明同一个实例里的另一个笔记本占着它；重启那个肇事者的 kernel，不用重启实例。

## 会话间保存环境

DSW 这里有两个机制，都值得了解：

1. **实例镜像快照**——把当前容器状态（已安装包、系统文件）打包到 ACR。下次启动实例选这个镜像，就能回到之前的状态。慢（几分钟）但精确。
2. **OSS 上的 Conda 环境**——把所有 `pip` 依赖装到 `/mnt/data/envs/myenv/` 然后激活。实例没了环境还在，不用重新打包。快但抓不到系统级变更（`apt install` 等）。

项目干活我默认用 conda-on-OSS，如果是“半年后还要展示的死固定 demo"就用快照机制。

## 镜像目录，别靠猜

选 DSW 镜像是每次启动实例都要做的决定，大多数团队至少要踩坑一个季度才能摸清规律。目录里大概有四大家族：

| 家族 | 标签模式 | 最适合 | 何时避免 |
|---|---|---|---|
| `modelscope:*` | `modelscope:1.28.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04` | LLM 开发，从 ModelScope hub 下载，任何已 pinned `transformers` 和 `vllm` 的场景 | 你需要比 ModelScope 发布的更新版本的 CUDA |
| `pytorch:*` | `pytorch:2.6.0-gpu-py311-cu124-ubuntu22.04` | 原生 PyTorch 工作，自定义训练循环，任何讨厌依赖 soup 的场景 | 你想要开箱即用的 LLM 栈 |
| `tensorflow:*` | `tensorflow:2.16.1-gpu-py311-cu123-ubuntu22.04` | TF / Keras 代码库，TFRecord 流水线 | 你在 2026 年从头开始（别这么做） |
| `dsw-stable:*` | `dsw-stable:1.10-pytorch2.4-cu121` | 长期运行的笔记本，不想季度中遇到 CUDA 升级 | 你需要最新特性 |

命名规则从左到右依次为：SDK 版本、框架版本、CPU/GPU 类型、Python 版本、CUDA 版本、操作系统。建议按此顺序解读镜像标签。我遇到最多的两类典型问题：

- **想要 bare PyTorch 却选了 ModelScope。** 镜像大概 14 GB，首次启动要拉 600+ 个 Python 包。如果不需要 `modelscope` SDK 或 `vllm`，省掉 90 秒冷启动和磁盘压力。
- **想要 ModelScope 却选了 PyTorch。** 然后每次新实例都要 `pip install vllm flash-attn modelscope`。能用，但浪费四分钟，而且偶尔会遇到 CUDA 版本不匹配，比如 `flash-attn` 决定跟错误的 nvcc 编译。

对于基于 Qwen3 的重 `vllm` 工作流，`modelscope:1.28.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04` 镜像阻力最小——`vllm`, `flash-attn`, `xformers`, `transformers` 全都预装好了兼容版本。纯自定义 CUDA 研究，从 `pytorch:2.6.0-gpu-py311-cu124-ubuntu22.04` 开始 build。任何你要在 6 个月后回来看的项目，在项目 README 里冻结镜像标签——`modelscope:1.28.0-...` 即使标签相同，六个月后也不是同一个 artifact，因为阿里云偶尔会重建并修补依赖。

> **实战建议：** 启动前在控制台检查镜像构建日期。如果超过 60 天，预计至少缺一个 CVE 补丁；对于生产级任务，选同家族里的最新标签。
## OSS-FUSE 挂载、延迟 profile 以及何时该拷贝 instead

![Aliyun PAI (2): PAI-DSW — Notebooks That Don't Eat Your Weights — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/02-pai-dsw-notebook/illustration_2.png)

OSS-FUSE 是默认的挂载方案，90% 的场景都够用，但它踩坑的方式比较隐蔽。脑子里的模型应该是：对挂载路径的每次 `read()` 都是一次 OSS HTTP 请求，每次 `write()` 先攒在本地 buffer，等到 `close()` 或者攒够几 MB 才刷写。这带来几个影响：

| Operation | Local SSD | OSS-FUSE | NAS | CPFS |
|---|---|---|---|---|
| Sequential read 1 GB | ~250 ms | 600-1200 ms | 400-700 ms | 80-120 ms |
| Random read 4 KB (cold) | ~0.1 ms | 30-80 ms | 5-15 ms | 1-3 ms |
| `os.listdir(10000 files)` | ~10 ms | 800-2000 ms | 200-500 ms | 50-100 ms |
| Append 10 MB log | <5 ms | 200-400 ms | 50-100 ms | 10-30 ms |

数据会随 Region 和 Bucket 类型波动，但规律不变。我踩坑总结出来的几条原则：

- **别对挂载了 OSS 的目录跑 `os.listdir()`，尤其是文件数超过 10k+。** 这是一次 HTTP `ListObjects` 往返，Python 还是懒迭代——你的训练脚本每个 epoch 都会假死 1-2 s。写个 manifest 文件一次性生成好。
- **训练过程中别直接往 OSS-FUSE 存 checkpoint，除非开了 `enable_easyckpt`。** 一个 7B-param `state_dict` 大概 14 GB，FUSE 每次保存会阻塞训练进程 10-30 s，GPU 只能干等着。要么先拷到本地 SSD 再异步上传，要么走 EasyCKPT 路径（见第 3 章）。
- **训练数据超过 100 GB 就别直接从 OSS-FUSE 读了。** 每个 FUSE 挂载点的带宽上限大概 200-400 MB/s，数据加载会成瓶颈。启动时用 `ossutil cp -r --jobs 8` 拷到本地 SSD，从本地读。如果数据集比本地 SSD 还大，改用 NAS 或 CPFS。
- **代码、配置文件和输出目录放心用 OSS-FUSE。** 偶尔读读的延迟能接受，只要不在热循环里写，开销也不大。

其他挂载模式也值得了解一下：

```python
# When you create a DSW instance, "Configure storage" lets you pick:
mount_modes = {
    "oss-fuse":   "default — POSIX-ish, lazy fetch, fine for code/config",
    "oss-direct": "skip FUSE; use ossutil/oss SDK from your code",
    "nas":        "real POSIX, paid per-GB-month, good for shared scratch",
    "cpfs":       "HPC throughput, expensive, reach for it on >50 GB/s aggregate",
}
```

我的默认策略是全员 OSS-FUSE，但在任何训练任务开始前加一个 `local_copy_step()`，把热点数据拉到 `/root/data/`（系统盘，快 SSD，虽说是临时的但没关系——反正 OSS 才是源）。这样能把 50 GB 图片数据集的数据加载时间从流式读取的 ~15 分钟降到拷贝后的 ~3 分钟。

## 空闲 shutdown、自动扩容和 GPU 共享

这三个生命周期功能，只要你忘过一次，它们省下的钱就值回票价了。

**空闲 shutdown。** 按实例配置。平台监控 CPU + GPU + 网络活动；三项低于阈值持续 N 分钟，实例就停止。我的默认设置：开发工作 30 min，和队友共用的实例 15 min，长周期训练 notebook 不设空闲 shutdown（改用定时 shutdown）。算笔账：忘关的 A10 实例大概 ~5 RMB/h，放个长周末就是 360 RMB；设了 30 min 空闲 shutdown 只亏 2.5 RMB，剩下的都省下了。有个坑：如果你开着屏幕共享盯着 TensorBoard 看，plot 每分钟更新一次，会骗过检测器以为实例还“活”着。如果真需要保活，在终端跑个 `nvidia-smi -l 1`。

**自动扩容。** 知道的人少一点。你可以停止实例，换个 GPU 类型重启，`/mnt/data/` 挂载和 conda 环境（如果持久化到 OSS）都不会丢。我会在半小时负载测试时升到 A100 80 GB，第二天开发再降回 A10。这是手动 停止/变更/启动，不是原地扩容——但因为状态持久化了，操作只要 2 分钟，不用重新配环境。

**GPU 共享 (cGPU)。** 最新的功能，容易漏看。PAI 允许通过 cGPU 虚拟化在两个 DSW 实例间共享一块物理 GPU，分配 fractional memory + compute（比如把 A10 的 24 GB 拆成 16 GB / 8 GB 给两个人用）。适合手头有个初级工程师做推理，需求 <8 GB，没必要单独给他配一台 A10 的情况。代价：cGPU 有 5-15% 开销，租户间隔离是“尽力而为”——别跟不可信的人共享。在工作空间的 *Resource sharing* 下配置，然后在 DSW 选择器里会显示为新的实例类型。

这三个旋钮组合起来，把我上一个团队的开发环境 GPU 账单砍掉了一半。任何新 workspace 第一天就值得配好。

## Snapshot vs 自定义镜像 vs git pull——选对方案

有三种方式能让“我起了个新 DSW 实例”感觉像“我回到了上次离开的地方”。每种都有适用场景。

**从 `/mnt/data/code/` git pull** 是源代码的正解。永远都是。代码放 OSS，首次实例 clone repo，后续每次启动 `git pull`。实例挂了也不怕，是唯一能跟 code review 集成的机制，而且免费。如果你的工作流还没上 git，先别往下读了，先把这个解决了。

**OSS 上的 Conda 环境** 是 Python 依赖的正解。在 `/mnt/data/envs/myenv/` 下创建环境，activate 它，`pip install` 所有东西。因为环境文件在 OSS 上，实例挂了也能活。启动比本地环境慢 ~30 s（conda activate 扫描有 FUSE 开销），但不用重建，没有缓存失效问题。限制：没法用这种方式 `apt install` 系统包，那些会随实例消失。

**实例镜像 snapshot** 适合你有非 Python 状态的时候。系统包（`libnuma`、自定义 `cuda` 库、奇怪的 C++ 依赖）、ACR 管理的内核模块、`/etc/` 里的任何东西。Snapshot 把整个容器文件系统冻结到 ACR；下次启动实例选这个镜像，就跟上次会话一模一样。慢（制作 3-8 min，拉取额外 1-2 min）但精确。我只在两种窄场景用 snapshot：(a) 半年后要展示的冻结 demo，(b) 花了一天调试才兼容的 CUDA 栈，再也不想重来。

**带 Dockerfile 的自定义 ACR 镜像** 适合团队级复现。CI 里构建，打上日期 tag，推送到 ACR，团队每个人的 DSW 都拉同一个镜像。任何超过 2 个贡献者的项目我都默认用这个——snapshot 路径让“到底装了什么”变得不透明，而 Dockerfile 可审查。代价：每次变更多 5 min 的 CI，得维护 Dockerfile。但值。

决策树：

- 只是代码变了？`git pull`。
- 代码 + 几个 `pip install` 仅供我自己？Conda on OSS。
- 重做整个环境会让我很痛苦？Snapshot。
- 超过一个人用这个环境？Custom ACR image。

我最常见到的错误：默认所有东西都用 snapshot，包括本该进 git 的东西。结果就是一个冻结镜像，里面硬编码了 `/root/notebooks/foo.ipynb`，还没法 diff 上个季度以来变了啥。

## 接下来是什么

第 3 篇文章会拿同一个 MNIST 任务，展示通过 DLC 跨多 GPU 和多节点扩展时会发生什么变化——包括文档里提了但没真正解释清楚的 AIMaster 容错机制。