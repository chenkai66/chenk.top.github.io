---
title: "阿里云 PAI 实战（二）：PAI-DSW——不会吃掉权重的 Notebook"
date: 2026-04-26 09:00:00
tags:
  - 阿里云 PAI
  - 机器学习
  - PAI-DSW
  - Jupyter
  - GPU
categories: 阿里云 PAI
lang: zh-CN
mathjax: false
series: aliyun-pai
series_title: "阿里云 PAI 实战手册"
series_order: 2
description: "PAI-DSW 的真实使用方式：选对 GPU 镜像、把 OSS 挂上去这样实例重启不会丢 checkpoint，以及一个可以直接复制到新 Notebook 的 CIFAR-10 ResNet 训练例子。"
disableNunjucks: true
---

每次有新 ML 同学接入 PAI，第一天的剧本都一模一样。开 DSW 实例、`pip install` 一通、训练一个小时、不知道为什么重启了内核，然后过来问我："师兄，模型文件去哪了？" 标准答案是——"在一台已经不存在的机器上的 `/root` 里"。这种课，听一次就够了。这篇文章就是让你提前听到的那一次。

PAI-DSW（Data Science Workshop）是 PAI 的 Notebook 产品。本质就是 JupyterLab + 浏览器版 VSCode，跑在 PAI 给你管好的 GPU ECS 上。卖点是你不用再装 CUDA/cuDNN/PyTorch，90 秒拿到一台能用的 GPU 机器。

## DSW 什么时候该用、什么时候别用

适合用 DSW 的：

- 交互式探索、EDA、画图
- 用 `pdb` 加真实 GPU 调试模型
- 小规模训练（单卡，几个小时之内）
- 写训练脚本、写完之后丢给 DLC
- 把推理代码调通、之后包成 EAS 服务

**不**适合 DSW 的：

- 多卡/多机训练——那是 DLC 的事
- 任何需要无人值守跑 8 小时以上的任务——DSW 的空闲关机会把你干掉
- 生产推理——那是 EAS

心智模型：**DSW 是开发机，DLC 是批量系统，EAS 是生产**。代码从左往右逐步成熟。

## 选机型

机型选择器有几十个 SKU。日常默认值：

| 工作类型 | 机型 | 选它的理由 |
|---|---|---|
| EDA、轻量开发，不需要 GPU | `ecs.g7.xlarge`（4c/16G） | 便宜、启动快 |
| 单卡开发，模型 < 24GB | `ecs.gn7i-c8g1.2xlarge`（1×A10，24GB） | 原型阶段最佳性价比 |
| 单卡 LLM 7B/13B 工作 | `ecs.gn7e-c12g1.3xlarge`（1×A100 40GB） | fp16 装得下 7B，int8 装得下 13B |
| 70B 推理快速验证 | `ecs.gn8v-c8g1.2xlarge`（1×H800 80GB） | 装得下 70B int4，贵——记得停 |

我自己的套路：**先小后大**。PAI 允许在数据落到持久化磁盘或 OSS 的前提下，停下 DSW 实例并以不同 SKU 重启。EDA 跑在 4 核 CPU 机器上 2 块钱一小时，真要 `.cuda()` 时再切 A10。

> **真实经验：** 一定要设自动关机时间。每个 DSW 实例都有 "空闲关机" 的开关，默认 1 小时，开发时拉到 30 分钟。我自己有过周一到公司发现一个被遗忘的 A100 实例从周五开始计费整个周末——这种事不止一次，丢人到我都不好意思公开说几次。

## 镜像目录

DSW 提供预烤好的镜像，省得你第一万次 `pip install torch`。目前真正有用的：

- `pai-image-pytorch:2.4-gpu-py310-cu124-ubuntu22.04` —— 现代 PyTorch，新项目我都用这个
- `pai-image-pytorch:2.1-gpu-py310-cu118-ubuntu22.04` —— 老一些，但和很多公开训练脚本对得上
- `pai-image-tensorflow:2.15-gpu-py310-cu121-ubuntu22.04` —— 必须用 TF 的话
- `modelscope:1.18.0-pytorch2.4-gpu-py310-cu124-ubuntu22.04` —— 自带 ModelScope 和能用的 transformers
- `pai-image-vllm:0.6-cu124-py310` —— 推理实验用

**按 CUDA 版本选，不要按 PyTorch 版本选。** CUDA 不匹配是 `RuntimeError: CUDA error: no kernel image is available for execution on the device` 的头号成因。A10/A100 系列要 CUDA ≥ 11.8，H800 要 ≥ 12.1。

你可以在镜像之上自己装包，这些 `pip install` 的东西落在用户级磁盘上，**只要不换 SKU 就能跨重启保留**。换 SKU 可能会拿到一个干净的根目录，所以 `requirements.txt` 一定要提交。

## 存储：每个人都会被坑的部分

DSW 实例有三层存储，你必须三层都搞清楚：

1. **系统盘**（`/`、`/root`、`/tmp`）——临时存储。停启会清空。当作 scratch 盘。
2. **用户持久化盘**（默认挂在 `/mnt/workspace`）——跨停启保留，按 GB-月计费。Notebook 文件就放在这里。
3. **OSS / NAS 挂载**——你的 bucket 显示成一个目录。数据集、checkpoint、所有需要长期保留的东西都放这里。

不那么直观的一点是：**重启内核不保留 RAM**（这很显然），**但停止实例会保留 `/mnt/workspace`**（不那么显然）。真正危险的操作是 "换实例 SKU"——这可能会重建系统盘。重要工件务必写到 OSS。

OSS 在创建实例时挂上。控制台 UI 有 "存储" 区，选 "OSS"，指向你的 bucket，挂载路径填 `/mnt/oss-data` 之类。背后 PAI 替你跑 `ossfs`。

```python
# Notebook 里 OSS 就是一个普通路径
import os, glob
files = glob.glob("/mnt/oss-data/datasets/cifar-10-batches-py/*")
print(len(files), "个文件可见")
```

> **真实经验：** OSS-FUSE 适合大文件的顺序读、适合写 checkpoint。它**完全不适合**对几百万个小文件做随机访问（比如 100 万张 JPEG）。这种工作负载，正确做法是把数据集 `tar` 成一个大 shard，实例启动时把 tar 拷到本地盘，然后在那儿解开。我亲眼见过这个改动把单 epoch 从 6 小时降到 25 分钟。

## 完整的 CIFAR-10 ResNet Notebook

把下面几段代码放到 DSW Notebook 的 cell 里，文件存为 `cifar_resnet.ipynb`。假设你已经把 OSS 挂在 `/mnt/oss-data`，模型也写到那儿。

```python
# Cell 1 —— 体检
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

```python
# Cell 2 —— 数据
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader

tfm = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                 T.ToTensor(),
                 T.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))])

train = torchvision.datasets.CIFAR10(
    root="/mnt/oss-data/datasets/cifar10", train=True,
    download=True, transform=tfm)
loader = DataLoader(train, batch_size=256, shuffle=True,
                    num_workers=4, pin_memory=True)
```

```python
# Cell 3 —— 模型（小 ResNet，不能直接用 torchvision.resnet18，
# 32×32 输入和 7×7 卷积不搭）
import torch.nn as nn, torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        self.short = (nn.Sequential() if stride == 1 and c_in == c_out
                      else nn.Sequential(
                          nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                          nn.BatchNorm2d(c_out)))
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.short(x))

class TinyResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem  = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False),
                                   nn.BatchNorm2d(64), nn.ReLU(True))
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.head   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(256, num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.head(x)
```

```python
# Cell 4 —— 训练，每个 epoch 把 checkpoint 写到 OSS
import os, time
device = "cuda"
model  = TinyResNet().to(device)
opt    = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                         weight_decay=5e-4, nesterov=True)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
loss_f = nn.CrossEntropyLoss()

CKPT_DIR = "/mnt/oss-data/checkpoints/cifar_resnet"
os.makedirs(CKPT_DIR, exist_ok=True)

for epoch in range(20):
    t0, total, correct, loss_sum = time.time(), 0, 0, 0.0
    model.train()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out = model(x); loss = loss_f(out, y)
        loss.backward(); opt.step()
        loss_sum += loss.item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item(); total += x.size(0)
    sched.step()
    torch.save({"model": model.state_dict(), "epoch": epoch},
               f"{CKPT_DIR}/epoch_{epoch:02d}.pt")
    print(f"epoch {epoch:02d}  loss={loss_sum/total:.4f}  "
          f"acc={correct/total:.4f}  {time.time()-t0:.1f}s")
```

单张 A10 上一个 epoch 大概 90 秒，20 个 epoch 测试集准确率 92% 左右。重点不是准确率——重点是**每个 epoch 的 checkpoint 都落到 OSS 上**，实例哪怕崩了，新的 DSW 也能从任意 epoch 续训。

## 快照：好用，但不能替代 OSS

DSW 有实例快照功能：暂停实例、对磁盘做快照、之后用别的 SKU 恢复。在 "我想要原封不动的环境，包括所有装好的包" 这种交接场景下确实有用。但它**不是备份策略**——快照绑定在实例和区域上，工作空间一删就一起没了。OSS 才是唯一可靠的持久化层。

## 下一篇

第三篇会把这套训练循环从 DSW 单卡扩展到 PAI-DLC 的 8 卡跨节点——包括如何让 checkpoint 逻辑在 DDP 下真的能用。如果你 Google 过 "PyTorch DDP NCCL hang"，那一篇你会想看。
