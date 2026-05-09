---
title: "阿里云 PAI 实战（二）：PAI-DSW——不会吃掉你权重的 Notebook"
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
每次带新人加入 PAI 团队，他们的第一天几乎如出一辙：先启动一个 DSW 实例，接着用 `pip install` 安装一堆依赖，跑一个小时的模型训练，中途因为某些原因重启了 kernel，然后就会跑来问我：“我的模型文件怎么不见了？” 老实说，答案是这样的——“在 `/root` 目录下，但那台实例早就没了。” 这种教训，踩一次坑就够了。而这篇文章的目的，就是让你在掉进坑之前就能学到这一课。

![阿里云 PAI 实战（二）：PAI-DSW——不会吃掉你权重的 Notebook — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/02-pai-dsw-notebook/illustration_1.jpg)
## DSW 到底是什么？

根据官方文档《DSW Overview》的描述，DSW 是一款**云端 AI 开发集成环境（IDE）**，集成了 JupyterLab、VSCode 和终端功能。它预装了 PyTorch 和 TensorFlow 的容器镜像，支持异构计算资源（如 CPU、GPU 以及灵骏加速器），并且能够挂载来自 OSS、NAS 和 CPFS 的数据集。实际使用时，只需点击“打开”按钮，不到一分钟你就能获得一个运行在真实 GPU 上的 Jupyter 环境，`nvidia-smi` 可用，PyTorch 也能直接导入。

有趣的是，文档里没提到的*盒子里缺了什么*。DSW 容器自带一块系统盘，但这块盘的生命周期与实例绑定——实例销毁，盘也随之消失。通过 `pip install` 安装的依赖包可以撑过 kernel 重启，但无法在实例重启后幸存。如果需要持久化环境，可以将 conda 环境保存到 OSS，或者利用快照功能将其存储到 ACR。

![DSW 实例结构解析](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/02-pai-dsw-notebook/fig1_dsw_anatomy.png)
## 如何选择实例规格

根据文档，DSW 的资源类型分为两类：**公共资源**（按量付费）和**专有资源**（基于通用计算或灵骏的包年包月）。对于日常开发工作，公共资源是更合适的选择——它按实际使用的 GPU 时间计费，并且支持秒级计费精度。即使只开 10 分钟跑个实验，也不会觉得浪费。

我的实际选择经验如下：

- **小规模实验或调试** — `ecs.gn7i-c8g1.2xlarge`（1 × A10，24 GB）。价格实惠，适合用来做 4-bit 量化的 7B 模型微调，或者运行 512×512 分辨率的扩散模型。
- **小型模型的正式训练** — `ecs.gn7i-c16g1.4xlarge` 或 `ecs.gn7e-c12g1.3xlarge`（A10 / A100 40 GB）。无论是 CIFAR-10 的 ResNet、ImageNet-tiny，还是使用 QLoRA 微调 7B 模型，这些配置都能轻松应对。
- **大语言模型（LLM）开发** — 至少选择 `ecs.gn7e-c12g1.6xlarge` 或更高配置（A100 80 GB）。如果需要在 BF16 精度下加载 13B 到 30B 的模型，且不依赖卸载技术，这一档是最低要求。

> **实战小贴士：** 如果控制台显示目标 GPU 类型“无库存”，可以尝试切换可用区（AZ）。库存是按可用区分的，而不是按区域（region）。我曾经遇到过这种情况：同一分钟内，`cn-shanghai-h` 的 A100 80G 显示无货，而 `cn-shanghai-l` 却有空闲资源。
## 镜像目录

DSW 提供的镜像是官方维护的，带有明确的版本号和标签。比如，在 Quick Start 中使用的 `modelscope:1.26.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04`，这个名称本身就清楚地描述了镜像的内容：从左到右依次是 ModelScope SDK 1.26、PyTorch 2.6、支持 GPU 的构建、Python 3.11、CUDA 12.4，以及基于 Ubuntu 22.04 的环境。

在实际使用中，我通常会选择 `pytorch` 或 `modelscope` 系列的镜像。虽然 TensorFlow 镜像也不错，但它的版本更新往往会滞后一个大版本。此外，还有 `dsw-stable` 系列镜像，这类镜像是有意保持版本滞后的——如果你在生产环境中进行临时训练任务，建议选择它，避免在训练过程中因为 CUDA 版本升级而导致中断。

当然，你也可以根据需求自定义镜像并推送到 ACR（阿里云容器镜像服务）。对于那些依赖复杂或较重的项目（比如 `vllm`、`flash-attn` 或自定义 CUDA 内核），我会选择这种方式。这样一来，每次启动新实例时，可以省去大约 4 分钟的 `pip install` 时间，效率提升非常明显。
## 不丢失数据的标准工作流程

控制台的操作流程大致如下：

![标准 DSW 工作流](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/02-pai-dsw-notebook/fig2_dsw_workflow.png)

生命周期钩子虽然容易被忽略，但一旦忘记，代价可不小。我通常会将**空闲自动关闭**设置为默认的 30 分钟，而**定时关闭**则设在晚上 11 点，这样可以避免周末忘记关闭运行中的 Notebook。每块闲置的 GPU 按 5 元/小时计算，如果从周五晚上挂到周一，差不多就得花掉 100 块。
## 数据存储的位置

在整个 DSW 文档中，有一张图堪称最关键：

![DSW 存储布局](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/02-pai-dsw-notebook/fig3_dsw_storage_layout.png)

这是我常用的数据挂载结构：

```
/mnt/data/
├── datasets/      # OSS 只读挂载（这个存储桶是长期有效的）
├── checkpoints/   # OSS 可写路径（每 N 步保存一次）
└── code/          # Git 仓库，同样存放在 OSS 上，新实例启动后直接挂载即可使用
```

在创建实例时需要配置 OSS 挂载，官方文档中称之为“配置存储”。选择对应的存储桶和路径前缀，设置挂载点为 `/mnt/data/`，访问模式默认选择基于 FUSE 的方式即可。实例启动后，你可以在终端运行 `oss ls oss://your-bucket/` 来验证是否正常工作——这一步实际上也是对 PAI ↔ OSS RAM 角色的健康检查。
## 跑通 MNIST 示例（直接参考 Quick Start）

官方 Quick Start 使用了经典的 MNIST 手写数字识别任务。以下是经过简化的最小训练代码片段，适合快速上手——如果需要完整版本，文档中提供了一个可以直接上传的 `mnist.ipynb` 文件。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备:", device)

# 数据预处理：转换为 Tensor 并标准化
tx = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练集和验证集
train = datasets.MNIST("/mnt/data/datasets", train=True,  download=True, transform=tx)
val   = datasets.MNIST("/mnt/data/datasets", train=False, download=True, transform=tx)

# 创建数据加载器
train_loader = DataLoader(train, batch_size=128, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val,   batch_size=512, shuffle=False, num_workers=2)

# 定义卷积神经网络模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 3, 1)  # 第一层卷积
        self.c2 = nn.Conv2d(32, 64, 3, 1) # 第二层卷积
        self.fc1 = nn.Linear(9216, 128)   # 全连接层 1
        self.fc2 = nn.Linear(128, 10)     # 全连接层 2

    def forward(self, x):
        x = F.relu(self.c1(x))            # 激活函数
        x = F.max_pool2d(F.relu(self.c2(x)), 2) # 最大池化
        x = torch.flatten(x, 1)           # 展平
        x = F.relu(self.fc1(x))           # 激活函数
        return self.fc2(x)                # 输出层

# 初始化模型和优化器
model = Net().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 训练模型
for epoch in range(3):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)  # 将数据移动到指定设备
        opt.zero_grad()                        # 清空梯度
        loss = F.cross_entropy(model(xb), yb)  # 计算损失
        loss.backward()                        # 反向传播
        opt.step()                             # 更新参数
    
    # 保存模型检查点到 OSS，而不是本地 /root 目录
    torch.save(model.state_dict(), f"/mnt/data/checkpoints/mnist_e{epoch}.pt")
    print(f"第 {epoch} 轮训练完成")
```

根据 Quick Start 的预期，在单张 A10 显卡上跑完 3 轮训练后，验证集的准确率应该接近 98%。如果你的结果明显低于这个数值，大概率是因为 OSS 挂载有问题，导致读取的数据目录不正确——这并不是模型本身的 bug。
## 内嵌 TensorBoard

DSW 已经将 TensorBoard 集成为了内置扩展，官方文档中详细介绍了如何通过菜单启用它。不过，我通常会直接在代码单元中运行以下命令来启动：

```python
%load_ext tensorboard
%tensorboard --logdir /mnt/data/checkpoints/runs --port 6006
```

按照文档的说明，你需要点击的链接是 `http://localhost:6006/`。不过，DSW 会自动代理这个端口，因此你可以直接通过 DSW 提供的 URL 在浏览器中访问 TensorBoard。如果遇到端口被占用的情况，大概率是因为同一个实例中的其他 notebook 正在使用该端口。这时只需重启占用端口的 notebook 的内核即可，无需重启整个实例。
## 跨会话保存环境

DSW 提供了两种机制来实现环境的持久化，各有特点，建议都了解一下：

1. **实例镜像快照** — 将当前容器的状态（包括已安装的软件包、系统文件等）完整保存到 ACR。下次启动新实例时，选择这个镜像即可恢复到上次的工作状态。这种方式速度较慢（通常需要几分钟），但能够精确还原所有改动。
2. **基于 OSS 的 Conda 环境** — 将所有 `pip` 依赖安装到 `/mnt/data/envs/myenv/` 目录下，并激活该环境使用。即使实例意外终止，环境也能保留，无需重新生成镜像。这种方式速度快，但无法捕获系统级别的变更（例如通过 `apt install` 安装的软件）。

在实际工作中，我通常会根据场景选择不同的方式：对于日常项目开发，更倾向于使用基于 OSS 的 Conda 环境；而对于需要长期保存的演示环境（比如“半年后还要展示的固化 demo”），则会选择实例镜像快照。
## 镜像目录，选对不靠猜

每次启动 DSW 实例时，选择一个合适的镜像是必经之路，但很多团队在最初的几个月里都会踩坑选错。镜像目录大致分为四个家族，各有其适用场景和限制：

| 家族 | 标签模式 | 适合 | 不适合 |
|---|---|---|---|
| `modelscope:*` | `modelscope:1.28.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04` | LLM 开发、从 ModelScope Hub 下载模型、依赖 `transformers` 和 `vllm` 的项目 | 需要比 ModelScope 默认提供的更新版本的 CUDA |
| `pytorch:*` | `pytorch:2.6.0-gpu-py311-cu124-ubuntu22.04` | 纯 PyTorch 项目、自定义训练逻辑、避免依赖冲突的场景 | 需要开箱即用的 LLM 工具栈 |
| `tensorflow:*` | `tensorflow:2.16.1-gpu-py311-cu123-ubuntu22.04` | TF/Keras 老项目、TFRecord 数据流水线 | 2026 年新开项目（尽量别这么做） |
| `dsw-stable:*` | `dsw-stable:1.10-pytorch2.4-cu121` | 长期运行的 Notebook，避免季度内 CUDA 版本变动 | 需要最新功能 |

镜像命名遵循一致的规则，从左到右依次是：SDK 版本 → 框架版本 → CPU/GPU → Python 版本 → CUDA 版本 → 操作系统。可以把它当作一份“配方”来读。以下是两种最常见的翻车场景：

- **需要纯 PyTorch 却误选了 ModelScope。** 这个镜像体积约 14 GB，首次启动会拉取 600 多个 Python 包。如果你不需要 `modelscope` SDK 或 `vllm`，这不仅浪费 90 秒冷启动时间，还会增加磁盘压力。
- **需要 ModelScope 却误选了 PyTorch。** 结果每次启动新实例时都得手动安装 `pip install vllm flash-attn modelscope`。虽然能跑起来，但每次多花 4 分钟，而且偶尔还会因为 `flash-attn` 编译时绑定错误的 nvcc 导致 CUDA 版本不匹配。

如果工作流重度依赖 `vllm`，比如在 Qwen3 上开发，推荐使用 `modelscope:1.28.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04`，这是阻力最小的选择——`vllm`、`flash-attn`、`xformers` 和 `transformers` 都预装了兼容版本。如果是纯自定义 CUDA 研究，则可以从 `pytorch:2.6.0-gpu-py311-cu124-ubuntu22.04` 开始逐步搭建环境。对于那些半年后还需要复现的项目，务必在项目 README 中固定镜像标签——即使 `modelscope:1.28.0-...` 的标签不变，半年后的实际内容可能已经不同，因为阿里云会不定期重建镜像并修补依赖。

> **实战建议：** 启动前先检查镜像的构建日期。如果超过 60 天，大概率会缺失至少一个 CVE 补丁；对于接近生产环境的工作负载，建议在同一家族中选择最新的镜像标签。
## OSS-FUSE 挂载：延迟分析与拷贝策略

![阿里云 PAI 实战（二）：PAI-DSW——不会吃掉你权重的 Notebook — visual](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/zh/aliyun-pai/02-pai-dsw-notebook/illustration_2.jpg)

OSS-FUSE 是默认的挂载方式，能满足 90% 的使用场景，但它的性能瓶颈往往隐藏在细节中。简单来说，每次对挂载路径执行 `read()` 操作都会触发一次 HTTP 请求到 OSS，而 `write()` 则会先写入本地缓冲区，等到 `close()` 或者缓冲区达到几 MB 时才会真正上传。这种机制带来了以下影响：

| 操作 | 本地 SSD | OSS-FUSE | NAS | CPFS |
|---|---|---|---|---|
| 顺序读取 1 GB | ~250 ms | 600-1200 ms | 400-700 ms | 80-120 ms |
| 随机读取 4 KB（冷数据） | ~0.1 ms | 30-80 ms | 5-15 ms | 1-3 ms |
| `os.listdir(10000 文件)` | ~10 ms | 800-2000 ms | 200-500 ms | 50-100 ms |
| 追加写入 10 MB 日志 | <5 ms | 200-400 ms | 50-100 ms | 10-30 ms |

这些数值会因地域和存储桶类型有所不同，但整体趋势是一致的。以下是我在实际工作中总结的经验：

- **避免对包含 1 万+ 文件的 OSS 挂载目录调用 `os.listdir()`**。这相当于发起了一次 `ListObjects` HTTP 请求，而 Python 的迭代器是惰性的——你的训练脚本会在每个 epoch 开始时卡住 1-2 秒。建议提前生成一个 manifest 文件来替代。
- **训练过程中不要直接将 checkpoint 写入 OSS-FUSE，除非启用了 `enable_easyckpt`**。一个 7B 参数的 `state_dict` 大约 14 GB，FUSE 会让训练进程在每次保存时阻塞 10-30 秒，导致 GPU 空转。推荐的做法是先存到本地 SSD，再异步上传；或者直接使用 EasyCKPT 路径（详见第三章）。
- **如果训练数据超过 100 GB，不要直接从 OSS-FUSE 读取**。每个 FUSE 挂载点的带宽上限约为 200-400 MB/s，容易成为数据加载的瓶颈。可以在启动时使用 `ossutil cp -r --jobs 8` 将数据拷贝到本地 SSD，然后从本地进行训练。对于超出本地 SSD 容量的数据集，可以选择 NAS 或 CPFS。
- **代码、配置文件和输出目录完全可以使用 OSS-FUSE**。偶尔的读取操作延迟是可以接受的，写入操作也不算昂贵，只要不在高频循环中频繁写入即可。

其他挂载模式也值得了解：

```python
# 创建 DSW 实例时，“配置存储” 提供以下选项：
mount_modes = {
    "oss-fuse":   "默认模式 — 类 POSIX，按需加载，适合代码和配置",
    "oss-direct": "跳过 FUSE，直接使用 ossutil 或 OSS SDK",
    "nas":        "真正的 POSIX 文件系统，按 GB-月计费，适合共享暂存",
    "cpfs":       "高性能计算专用，吞吐量高，适合 >50 GB/s 的聚合带宽场景",
}
```

我的习惯是默认使用 OSS-FUSE，并在训练开始时通过 `local_copy_step()` 将热点数据复制到 `/root/data/`（系统盘，高速 SSD，临时存储——但这没关系，因为 OSS 上有真实的数据源）。以一个 50 GB 的图像数据集为例，直接流式读取可能需要 15 分钟，而先拷贝到本地再训练只需 3 分钟，彻底解决了数据加载的瓶颈问题。
## 空闲自动关机、动态调整规格与 GPU 共享

这三个功能，哪怕你第一次用的时候忘了配置，它们的价值也会立刻让你意识到。

**空闲自动关机。** 每个实例都可以单独设置。平台会监控 CPU、GPU 和网络活动，如果三项指标都低于设定的阈值，并持续 N 分钟，实例就会自动停止运行。我的习惯是：开发环境设为 30 分钟，和同事共享的实例设为 15 分钟，而长时间运行的训练 notebook 则关闭此功能（改用定时关机）。举个例子，如果你忘记关掉一台 A10 实例（约 5 RMB/小时），一个长周末下来可能就要花掉 360 RMB；但如果设置了 30 分钟空闲关机，最多只消耗 2.5 RMB，剩下的钱都省下了。不过要小心的是，比如你在屏幕共享时盯着每分钟刷新一次的 TensorBoard 图表，系统可能会误判为“活跃”。如果确实需要保持实例运行，可以在终端执行 `nvidia-smi -l 1` 来避免被误停。

**动态调整规格。** 这个功能知道的人不多。你可以随时停止实例，切换到另一种 GPU 类型后再启动，而且不会丢失 `/mnt/data/` 挂载的数据或 OSS 上保存的 conda 环境。比如，我有时会临时将实例升配到 A100 80 GB 跑半小时的压力测试，第二天早上再降回 A10 继续开发工作。虽然这不是热切换——需要手动停止、修改配置、再启动——但由于状态都持久化了，整个过程只需 2 分钟，完全不需要重新搭建环境。

**GPU 共享（cGPU）。** 这是最新增加的功能，很容易被忽略。PAI 平台通过 cGPU 虚拟化技术，可以将一块物理 GPU 分配给两个 DSW 实例使用，按需分配显存和计算资源。例如，可以把一张 A10 的 24 GB 显存切分为 16 GB 和 8 GB，供两位用户分别使用。这种方案非常适合初级工程师做推理任务时显存需求小于 8 GB 的场景，不必单独给他们分配整张 A10。但需要注意的是，启用 cGPU 会带来 5%-15% 的性能开销，且租户间的隔离是“尽力而为”的——不要和不信任的用户共享资源。这个功能需要在工作空间的 *资源共享* 设置中开启，然后在 DSW 实例类型选择界面会多出一个新的选项。

合理配置这三个功能后，团队的开发环境 GPU 成本差不多能砍掉一半。因此，无论什么时候创建新的工作空间，第一天就应该把这些功能配置好。
## 快照、自定义镜像还是 git pull？选对工具事半功倍

在阿里云 DSW 中，有三种方法可以让“新开一个实例”感觉像是“我从未离开过”。每种方法都有其适用场景，关键是根据需求选择合适的方案。

**代码管理：始终用 `git pull`**  
如果你的工作流涉及源码管理，那么答案很简单：把代码存放在 OSS 中，首次启动实例时克隆仓库，之后每次启动实例时执行 `git pull`。这种方法不仅能确保代码安全（即使实例挂掉也不受影响），还能无缝对接代码审查流程，并且完全免费。如果你的团队还没有使用 git，请先解决这个问题再继续。

**Python 依赖：OSS 上的 Conda 环境**  
对于 Python 的依赖管理，最佳实践是将 Conda 环境存储在 OSS 中。具体来说，在 `/mnt/data/envs/myenv/` 下创建环境，激活后通过 `pip install` 安装所需依赖。由于环境文件存储在 OSS 中，即使实例被销毁，依赖也不会丢失。唯一的代价是启动时会比本地环境慢约 30 秒（因为 FUSE 在 Conda 激活时需要扫描文件系统），但避免了重建环境和缓存失效的问题。需要注意的是，这种方法不适用于通过 `apt install` 安装的系统级包，这些包仍然会随着实例的销毁而消失。

**非 Python 状态：实例快照**  
如果你的环境中包含非 Python 的状态（例如系统库 `libnuma`、自定义的 CUDA 库、奇怪的 C++ 依赖，或者 ACR 管理的内核模块等），那么实例快照是最合适的选择。快照会将整个容器文件系统冻结并存储到 ACR 中，下次启动实例时选择该镜像即可恢复到上次的状态。虽然这种方法速度较慢（制作快照需要 3-8 分钟，拉取镜像额外需要 1-2 分钟），但它能精确还原环境。我个人仅在两种情况下使用快照：(a) 半年后需要展示的固定演示环境；(b) 调试了一天才搞定兼容性的 CUDA 栈，再也不想重复配置。

**团队协作：自定义 ACR 镜像 + Dockerfile**  
如果需要确保团队级别的环境一致性，最佳选择是基于 Dockerfile 构建自定义 ACR 镜像。通过 CI 流水线构建镜像，按日期打标签并推送到 ACR，团队成员的 DSW 实例可以直接拉取相同的镜像。对于任何超过两名开发者的项目，我都推荐这种方式——快照路径会让“到底安装了什么”变得不透明，而 Dockerfile 是可审查的，便于维护。代价是每次改动需要额外 5 分钟的 CI 时间，并且需要维护 Dockerfile，但从长期来看这是值得的。

**决策树：快速定位你的需求**  
- 只修改了代码？用 `git pull`。  
- 修改了代码并添加了一些个人使用的 Python 包？用 OSS 上的 Conda 环境。  
- 如果重新配置环境会让你头疼？用快照。  
- 多人共享同一环境？用自定义 ACR 镜像。

**常见误区：滥用快照**  
我见过最常见的错误是，无论什么情况都默认使用快照，甚至包括那些本该放进 git 的内容。结果就是，最终得到一个冻结的镜像，里面硬编码了类似 `/root/notebooks/foo.ipynb` 的路径，完全无法追踪上个季度做了哪些修改。
## 下一篇

第三篇把同一个 MNIST 任务推到多卡多机的 DLC 上，重点讲 AIMaster 容错——文档里提了但没真正讲清楚的那一块。
