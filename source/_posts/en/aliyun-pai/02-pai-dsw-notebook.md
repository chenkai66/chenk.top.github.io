---
title: "Aliyun PAI (2): PAI-DSW — Notebooks That Don't Eat Your Weights"
date: 2026-03-06 09:00:00
tags:
  - Aliyun PAI
  - Machine Learning
  - PAI-DSW
  - Jupyter
  - GPU
categories: Aliyun PAI
lang: en
mathjax: false
series: aliyun-pai
series_title: "Aliyun PAI Practical Guide"
series_order: 2
description: "Working with PAI-DSW for real: choosing the right GPU image, mounting OSS so you don't lose checkpoints when the instance restarts, and a CIFAR-10 ResNet that you can copy-paste into a fresh notebook."
disableNunjucks: true
---

Every time I onboard a new ML engineer to PAI the first day looks the same. They start a DSW instance, `pip install` their world, train for an hour, restart the kernel for some reason, and then ask me where their model file went. The honest answer — "in `/root` on a node that no longer exists" — is the kind of lesson you only need to learn once. This article is the version of that lesson you read in advance.

PAI-DSW (Data Science Workshop) is PAI's notebook product. It's JupyterLab + VSCode-in-browser running on a GPU ECS instance that PAI manages for you. The pitch is that you skip the CUDA/cuDNN/PyTorch install dance and get a usable GPU box in about 90 seconds.

## When to use DSW (and when not to)

Reach for DSW for:

- Interactive exploration, EDA, plotting
- Debugging a model with `pdb` and a real GPU
- Small-scale training (single-GPU, < a few hours)
- Writing the training script you will eventually submit to DLC
- Iterating on inference code before wrapping it for EAS

Do **not** use DSW for:

- Multi-GPU / multi-node training — that's DLC
- Anything that needs to run unattended for > 8 hours — DSW idle-shutdown will kill it
- Production inference — that's EAS

The mental model: **DSW is your dev box. DLC is your batch system. EAS is your prod.** Code matures left to right.

## Picking an instance type

The instance picker has dozens of SKUs. Practical defaults:

| Workload | SKU | Why |
|---|---|---|
| EDA, light dev, no GPU needed | `ecs.g7.xlarge` (4c/16G) | Cheap, fast boot |
| Single-GPU dev, < 24 GB models | `ecs.gn7i-c8g1.2xlarge` (1×A10, 24 GB) | Best $/hr for prototyping |
| Single-GPU LLM 7B/13B work | `ecs.gn7e-c12g1.3xlarge` (1×A100 40 GB) | Fits 7B in fp16, 13B in int8 |
| Quick test of 70B inference | `ecs.gn8v-c8g1.2xlarge` (1×H800 80 GB) | Fits 70B int4, expensive — stop it |

The pattern I follow: **start small, then resize**. PAI lets you stop a DSW instance and restart it with a different SKU as long as your data is on a persistent disk or OSS. Run EDA on a 4-core CPU box for 2 yuan/hour, switch to A10 only when you actually call `.cuda()`.

> **Real-world tip:** Set the auto-shutdown timer. Every DSW instance has an "idle shutdown" knob — default 1 hour. Push it to 30 minutes for dev. The number of times I've come in on Monday to find a forgotten A100 instance billing all weekend is too high to admit publicly.

## The image catalog

DSW ships pre-baked images so you don't `pip install torch` for the millionth time. The current useful ones:

- `pai-image-pytorch:2.4-gpu-py310-cu124-ubuntu22.04` — modern PyTorch, what I use for everything new
- `pai-image-pytorch:2.1-gpu-py310-cu118-ubuntu22.04` — older but matches a lot of public training scripts
- `pai-image-tensorflow:2.15-gpu-py310-cu121-ubuntu22.04` — TF if you must
- `modelscope:1.18.0-pytorch2.4-gpu-py310-cu124-ubuntu22.04` — comes with ModelScope and a working transformers stack
- `pai-image-vllm:0.6-cu124-py310` — for serving experiments

**Pick by CUDA version, not by PyTorch version.** Mismatched CUDA is the #1 cause of `RuntimeError: CUDA error: no kernel image is available for execution on the device`. The A10/A100 series wants CUDA ≥ 11.8; H800 wants ≥ 12.1.

You can install your own packages on top, but everything you `pip install` lives in the user-level disk and **survives instance restart** as long as you didn't change SKU. If you change SKU, you may get a fresh root, so always commit a `requirements.txt`.

## Storage: the part that bites everyone

A DSW instance has three storage tiers and you must understand all three:

1. **System disk** (`/`, `/root`, `/tmp`) — ephemeral. Wiped on stop/start. Treat as scratch.
2. **User persistent disk** (`/mnt/workspace` by default) — survives stop/start, charged per GB-month. This is where notebooks live.
3. **OSS / NAS mount** — your bucket appears as a directory. Datasets, checkpoints, anything you care about long-term.

The non-obvious detail: **the kernel restart button does not preserve RAM** (obvious) **but the instance stop button does preserve `/mnt/workspace`** (less obvious). The risky operation is "change instance SKU" — that can recreate the root disk. Always write important artifacts to OSS.

Mount OSS at instance creation time. The console UI has a "Storage" section; pick "OSS", point at your bucket, choose a mount path like `/mnt/oss-data`. Behind the scenes PAI runs `ossfs` for you.

```python
# Inside the notebook, OSS is just a path
import os, glob
files = glob.glob("/mnt/oss-data/datasets/cifar-10-batches-py/*")
print(len(files), "files visible from OSS")
```

> **Real-world tip:** OSS-FUSE is fine for sequential reads of large files and for writing checkpoints. It is **terrible** for random access on millions of small files (think 1M JPEG images). For that workload, `tar` your dataset into one big shard, copy the tar to local disk on instance start, and untar there. I have seen this turn a 6-hour epoch into a 25-minute one.

## A complete CIFAR-10 ResNet notebook

Save this as `cifar_resnet.ipynb` cells in DSW. It assumes you mounted an OSS path at `/mnt/oss-data` and that you'll write the model out there.

```python
# Cell 1 — sanity
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
```

```python
# Cell 2 — data
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
# Cell 3 — model (a small ResNet, not torchvision.resnet18 because
# 32x32 inputs and 7x7 conv don't mix)
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
# Cell 4 — train, save checkpoint to OSS every epoch
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

On a single A10 this runs in about 90 seconds per epoch and gets you to ~92% test accuracy in 20 epochs. The point isn't the accuracy — it's that **every checkpoint goes to OSS**, so if the instance crashes you can resume from any epoch in a fresh DSW.

## Snapshots: nice but not a substitute for OSS

DSW has an instance-snapshot feature: pause an instance, snapshot the disk, restore later on a different SKU. It's genuinely useful for "I want this exact environment with all my installed packages" handoffs. It is **not** a backup strategy — snapshots are tied to the instance and the region, and if you delete the workspace they go with it. OSS is the only durable layer.

## What's next

Article 3 takes the same training loop and shows how to scale it from one GPU in DSW to eight GPUs across two nodes via PAI-DLC, including how to make the checkpoint logic actually work under DDP. If you've ever Googled "PyTorch DDP NCCL hang" you'll want that one.
