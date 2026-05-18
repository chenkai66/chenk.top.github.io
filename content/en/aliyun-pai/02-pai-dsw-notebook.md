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
series_total: 5
description: "Working with PAI-DSW for real: choosing the right GPU image, mounting OSS so you don't lose checkpoints when the instance restarts, and an MNIST notebook drawn from the official Quick Start that you can copy-paste."
disableNunjucks: true
translationKey: "aliyun-pai-2"
---

Every time I onboard a new ML engineer to PAI the first day looks the same. They start a DSW instance, `pip install` their world, train for an hour, restart the kernel for some reason, and then ask me where their model file went. The honest answer — "in `/root` on a node that no longer exists" — is the kind of lesson you only need to learn once. This article is the version of that lesson you read in advance.

![Aliyun PAI (2): PAI-DSW — Notebooks That Don't Eat Your Weights — Chapter overview](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/02-pai-dsw-notebook/illustration_1.png)

---

## What DSW actually is

Per the official "DSW Overview", DSW is a **cloud-based IDE for AI development** that integrates JupyterLab, VSCode, and a terminal, with pre-configured container images for PyTorch and TensorFlow, heterogeneous compute (CPU / GPU / Lingjun), and the ability to mount datasets from OSS, NAS, and CPFS. In practice that means you click "Open" and within a minute you have a real Jupyter on a real GPU with `nvidia-smi` working and PyTorch already importable.

What's interesting is *what's not in the box*. The DSW container has a system disk that lives only as long as the instance does. Anything you `pip install` survives a kernel restart but does not survive an instance restart unless you persist the conda env to OSS or save it to ACR via the snapshot feature.

![Anatomy of a DSW instance](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/02-pai-dsw-notebook/fig1_dsw_anatomy.png)

## Picking an instance type

According to the docs, DSW resource types come in two flavors: **public resources** (pay-as-you-go) and **dedicated resources** (subscription on general-purpose compute or Lingjun). For day-to-day work, public resources are the best choice — you pay for the GPU minutes you use, and the per-second metering lets you spin one up for a 10-minute experiment without worry.

What I actually pick:

- **Tiny experiment / debugging** — `ecs.gn7i-c8g1.2xlarge` (1 × A10, 24 GB). Cheap, plenty for fine-tuning a 7B in 4-bit quant or running diffusion at 512×512.
- **Real training of a small model** — `ecs.gn7i-c16g1.4xlarge` or `ecs.gn7e-c12g1.3xlarge` (A10 / A100 40 GB). Comfortable for a CIFAR-10 ResNet, ImageNet-tiny, or a 7B SFT with QLoRA.
- **LLM dev** — `ecs.gn7e-c12g1.6xlarge` or higher (A100 80 GB). Required if you want to load a 13-30B in BF16 without offloading.

> **Real-world tip:** If the GPU type you want is "out of stock" in the console, switch the AZ. Stock is per-AZ, not per-region. I have seen 80 GB A100 unavailable in `cn-shanghai-h` and free in `cn-shanghai-l` in the same minute.

## The image catalog

DSW images are official, versioned, and tagged. The Quick Start uses `modelscope:1.26.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04` — that string tells you exactly what is inside. Read it left to right: ModelScope SDK 1.26, PyTorch 2.6, GPU build, Python 3.11, CUDA 12.4, Ubuntu 22.04.

I almost always pick a `pytorch` or `modelscope` image. The TensorFlow images are fine but lag a major release behind. There is also a `dsw-stable` family that lags by design — pick it for production-adjacent work where you do not want a CUDA bump in the middle of a training run.

You can also bake your own image and push it to ACR. I do this for projects with a heavy dependency tree (`vllm`, `flash-attn`, custom CUDA kernels) — saves four minutes of `pip install` every time someone starts a fresh instance.

## A standard workflow that does not lose data

The console flow looks like this:

![Standard DSW workflow](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/02-pai-dsw-notebook/fig2_dsw_workflow.png)

The lifecycle hooks are easy to ignore and expensive to forget. **Idle shutdown** at 30 minutes is my default; **scheduled shutdown** at 11pm catches the case where I leave a notebook running over the weekend. Every DSW idle GPU at 5 RMB an hour is roughly 100 RMB you owe Aliyun by Monday.

## Where your data lives

The single most important diagram in the entire DSW docs:

![DSW storage layout](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/02-pai-dsw-notebook/fig3_dsw_storage_layout.png)

The mount path I use everywhere:

```text
/mnt/data/
├── datasets/      # read-only OSS mount (the bucket lives forever)
├── checkpoints/   # writeable OSS prefix (save every N steps)
└── code/          # git repo, also on OSS so a new instance is one mount away
```

Mounting OSS is configured at instance create time; the docs call it "Configure storage". Pick the bucket and the prefix, choose mount path `/mnt/data/`, accept the default access mode (FUSE-backed). After launch, `oss ls oss://your-bucket/` should work from the terminal — that is your "PAI ↔ OSS RAM role" health check.

## A working MNIST notebook (straight from the Quick Start)

The official Quick Start uses MNIST handwritten digit recognition. Here is the minimum viable training cell, simplified for the article — the docs link to a full `mnist.ipynb` you can upload as-is:

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

The Quick Start expects about 98% validation accuracy after 3 epochs on a single A10. If you see significantly lower results, you've likely mounted OSS incorrectly and are reading from the wrong directory — not a model bug.

## TensorBoard inline

DSW ships TensorBoard as a built-in extension; the docs walk through enabling it from the menu. I usually just run it as a cell:

```python
%load_ext tensorboard
%tensorboard --logdir /mnt/data/checkpoints/runs --port 6006
```

The link the docs tell you to click is `http://localhost:6006/` — DSW proxies the port so it works in your browser through the DSW URL. If the port is "in use", another notebook in the same instance is holding it; restart the kernel of the offender, not the instance.

## Saving the env between sessions

DSW has two mechanisms here, both worth knowing:

1. **Instance image snapshot** — bakes your current container state (installed packages, system files) to ACR. Next instance you start, pick that image and you are back where you left off. Slow (a few minutes) but exact.
2. **Conda env on OSS** — install all your `pip` deps under `/mnt/data/envs/myenv/` and activate it. Survives instance death without rebaking. Faster but does not capture system-level changes (`apt install` etc).

I default to the conda-on-OSS approach for project work and the snapshot mechanism for "frozen demo I want to show in 6 months".

## The image catalog, choosing without guessing

Picking a DSW image is a choice you make every time you start an instance, and most teams pick the wrong one for at least a quarter before they figure it out. The catalog has roughly four families:

| Family | Tag pattern | Best for | Avoid when |
|---|---|---|---|
| `modelscope:*` | `modelscope:1.28.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04` | LLM dev, downloading from ModelScope hub, anything with `transformers` and `vllm` already pinned | You need a CUDA version newer than what ModelScope ships |
| `pytorch:*` | `pytorch:2.6.0-gpu-py311-cu124-ubuntu22.04` | Vanilla PyTorch work, custom training loops, anything that hates dependency soup | You want batteries-included LLM stack |
| `tensorflow:*` | `tensorflow:2.16.1-gpu-py311-cu123-ubuntu22.04` | TF / Keras codebases, TFRecord pipelines | You're starting fresh in 2026 (just don't) |
| `dsw-stable:*` | `dsw-stable:1.10-pytorch2.4-cu121` | Long-running notebooks where you don't want CUDA bumps mid-quarter | You need the latest features |

The naming is consistent left-to-right: SDK version → framework version → CPU/GPU → Python version → CUDA version → OS. Read it like a recipe. The two failure modes I've seen most:

- **Picking ModelScope when you wanted bare PyTorch.** The image is ~14 GB and pulls 600+ Python packages on first start. If you don't need `modelscope` SDK or `vllm`, save yourself 90 s of cold start and the disk pressure.
- **Picking PyTorch when you wanted ModelScope.** Then `pip install vllm flash-attn modelscope` on every fresh instance. It works; it just costs you four minutes and occasional CUDA-version mismatches when `flash-attn` decides to compile against the wrong nvcc.

For a `vllm`-heavy workflow on Qwen3, the `modelscope:1.28.0-pytorch2.6.0-gpu-py311-cu124-ubuntu22.04` image is the path of least resistance — `vllm`, `flash-attn`, `xformers`, `transformers` all preinstalled at compatible versions. For pure custom-CUDA research, start from `pytorch:2.6.0-gpu-py311-cu124-ubuntu22.04` and build up. For anything you'll come back to in 6 months, freeze the image tag in your project README — `modelscope:1.28.0-...` is not the same artifact six months from now even with the same tag, because Aliyun rebuilds occasionally and patches dependencies.

> **Real-world tip:** Check the image build date in the console before launching. If it's older than 60 days, expect at least one CVE patch is missing; for production-adjacent work pick the latest tag in the same family.

## OSS-FUSE mount, latency profile, and when to copy instead

![Aliyun PAI (2): PAI-DSW — Notebooks That Don't Eat Your Weights — Chapter summary](https://blog-pic-ck.oss-cn-beijing.aliyuncs.com/posts/en/aliyun-pai/02-pai-dsw-notebook/illustration_2.png)

OSS-FUSE is the default mount mechanism, and it's fine for 90% of work, but the ways it fails are subtle. The mental model: every `read()` on a mounted path is an HTTP request to OSS, every `write()` accumulates in a local buffer and flushes on `close()` or every few MB. That has implications:

| Operation | Local SSD | OSS-FUSE | NAS | CPFS |
|---|---|---|---|---|
| Sequential read 1 GB | ~250 ms | 600-1200 ms | 400-700 ms | 80-120 ms |
| Random read 4 KB (cold) | ~0.1 ms | 30-80 ms | 5-15 ms | 1-3 ms |
| `os.listdir(10000 files)` | ~10 ms | 800-2000 ms | 200-500 ms | 50-100 ms |
| Append 10 MB log | <5 ms | 200-400 ms | 50-100 ms | 10-30 ms |

The numbers vary by region and bucket type, but the pattern holds. The takeaways I've internalized:

- **Don't `os.listdir()` an OSS-mounted directory with 10k+ files.** That's a single HTTP `ListObjects` round trip, and Python iterates it lazily — your training script will appear to hang for 1-2 s on every epoch. Use a manifest file you write once.
- **Don't checkpoint to OSS-FUSE during training without `enable_easyckpt`.** A 7B-param `state_dict` is ~14 GB; FUSE will block the training process for 10-30 s on every save, and your GPU sits idle. Either copy to local SSD then async-upload, or use the EasyCKPT path (see chapter 3).
- **Don't read training data directly from OSS-FUSE if it's >100 GB.** The bandwidth cap per FUSE mount is roughly 200-400 MB/s and you'll bottleneck on data loading. Copy to local SSD at startup with `ossutil cp -r --jobs 8`, train from local. For datasets bigger than the local SSD, use NAS or CPFS instead.
- **Do use OSS-FUSE for code, configs, and the output directory.** Latency is fine for occasional reads, and writes are cheap as long as you're not doing them in the hot loop.

The other mount modes are worth knowing:

```python
# When you create a DSW instance, "Configure storage" lets you pick:
mount_modes = {
    "oss-fuse":   "default — POSIX-ish, lazy fetch, fine for code/config",
    "oss-direct": "skip FUSE; use ossutil/oss SDK from your code",
    "nas":        "real POSIX, paid per-GB-month, good for shared scratch",
    "cpfs":       "HPC throughput, expensive, reach for it on >50 GB/s aggregate",
}
```

I default to OSS-FUSE for everything plus a `local_copy_step()` at the start of any training run that pulls hot data to `/root/data/` (system disk, fast SSD, ephemeral but that's fine — it's a copy of an OSS source-of-truth). Cuts data-loading time on a 50 GB image dataset from ~15 minutes streaming to ~3 minutes after the initial copy.

## Idle shutdown, auto-resize, and GPU sharing

The three lifecycle features that pay for themselves the first time you forget them.

**Idle shutdown.** Configurable per instance. The platform watches CPU + GPU + network activity; below all three thresholds for N minutes, instance is stopped. My defaults: 30 min for dev work, 15 min for an instance I share with a teammate, no idle shutdown for a long-running training notebook (use scheduled shutdown instead). The save: a forgotten A10 instance at ~5 RMB/h for a long weekend is roughly 360 RMB; idle shutdown at 30 min costs you 2.5 RMB and saves the rest. The trap: a screen-share where you're watching a TensorBoard plot that updates every minute can fool the detector into "active". Run `nvidia-smi -l 1` in a terminal if you genuinely need it kept alive.

**Auto-resize.** Less well-known. You can stop an instance and restart it on a different GPU type without losing your `/mnt/data/` mounts or your conda env (if persisted to OSS). I scale up to A100 80 GB for a half-hour load test, then back down to A10 for development the next morning. This is a manual stop / change / start, not in-place — but the persisted state means it's a 2-minute operation, not a re-setup.

**GPU sharing (cGPU).** The newest knob and easy to miss. PAI lets you share a single physical GPU across two DSW instances using cGPU virtualization, allocating fractional memory + compute (e.g., split an A10's 24 GB into 16 GB / 8 GB for two users). Useful when you have a junior engineer doing inference work that needs <8 GB and you don't want to provision them their own A10. The catch: cGPU adds 5-15% overhead and isolation between tenants is "best effort" — don't share with a non-trusted user. Configured in the workspace under *Resource sharing*, then exposed as a new instance type in the DSW selector.

Combined, these three knobs roughly halved the dev-environment GPU bill on the team I last ran. Worth setting up properly on day one of any new workspace.

## Snapshot vs custom image vs git pull — pick the right one

Three ways to make "I started a new DSW instance" feel like "I'm right back where I left off". Each has a sweet spot.

**Git pull from `/mnt/data/code/`** is the right answer for source code. Always. Code in OSS, repo cloned on first instance, `git pull` on every subsequent boot. Survives instance death, is the only mechanism that integrates with code review, and costs nothing. If your workflow doesn't include git, fix that before reading further.

**Conda env on OSS** is the right answer for Python deps. Create the env under `/mnt/data/envs/myenv/`, activate it, `pip install` everything. Survives instance death because the env files live in OSS. ~30 s slower to start than a local env (FUSE overhead on the conda activate scan) but no rebuild, no cache invalidation. The constraint: you can't `apt install` system packages this way, those die with the instance.

**Instance image snapshot** is the right answer when you have non-Python state. System packages (`libnuma`, custom `cuda` libs, weird C++ deps), ACR-managed kernel modules, anything in `/etc/`. The snapshot freezes the entire container filesystem to ACR; next time you start an instance, pick that image and you're identical to last session. Slow (3-8 min to bake, 1-2 min extra to pull) but exact. I use snapshots for two narrow cases: (a) a frozen demo I'll show in 6 months, (b) a CUDA stack that took me a day to debug into compatibility and I never want to redo.

**Custom ACR image with a Dockerfile** is the right answer for team-wide reproducibility. Build it in CI, tag it with a date, push to ACR, every team member's DSW pulls the same image. This is what I default to for any project with >2 contributors — the snapshot path makes "what's actually installed" opaque, and a Dockerfile is reviewable. Cost: an extra 5 min of CI per change, and you have to maintain the Dockerfile. Worth it.

The decision tree:

- Just code changes? `git pull`.
- Code + a couple of `pip install` for me only? Conda on OSS.
- I'd be sad if I had to redo this whole environment? Snapshot.
- More than one person uses this environment? Custom ACR image.

The mistake I see most: defaulting to snapshot for everything, including things that should be in git. Then you have a frozen image with a hard-coded `/root/notebooks/foo.ipynb` and no way to diff what changed since last quarter.

## What's Next

Article 3 takes the same MNIST job and shows what changes when you scale it across multiple GPUs and multiple nodes via DLC — including the AIMaster fault tolerance that the docs mention but do not really explain.
