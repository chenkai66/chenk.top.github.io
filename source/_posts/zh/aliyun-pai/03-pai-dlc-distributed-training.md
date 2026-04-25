---
title: "阿里云 PAI 实战（三）：PAI-DLC——不用通宵刨坑的分布式训练"
date: 2026-03-07 09:00:00
tags:
  - 阿里云 PAI
  - 机器学习
  - PAI-DLC
  - 分布式训练
  - PyTorch DDP
categories: 阿里云 PAI
lang: zh-CN
mathjax: false
series: aliyun-pai
series_title: "阿里云 PAI 实战手册"
series_order: 3
description: "如何用 PAI-DLC 提交真实的 PyTorch DDP 任务、高效挂载数据、扛住抢占式实例被回收，以及一个可以套到自己模型上的 8 卡 LLM SFT 完整示例。"
disableNunjucks: true
---

第一次在裸 GPU ECS 上跑多机训练，我花了两天调 NCCL 互联，再半天调 `torchrun` 的 rendezvous，再一整天搞 OSS 挂载的权限。后来才知道，PAI-DLC 就是为了让你不用经历这些。提交一份任务规范，拿到一个集群、看到日志、checkpoint 自动落 OSS、月底收账单。完事。

这篇文章端到端走一遍 DLC，配一个具体的 8 卡 LLM SFT。读完你会有一份能跑通的 SDK 提交脚本、对 "什么时候该从 DSW 切到 DLC" 的判断、以及关于数据加载的几个用血换的观点。

## DLC 到底是什么

PAI-DLC（Deep Learning Container）是托管的分布式训练任务系统。你交给它：

1. 一个**容器镜像**（PAI 官方的，或你自己 ACR 的）
2. 一条**启动命令**（`torchrun ...`、`python train.py` 之类）
3. 一份**资源规格**（worker 数、每 worker GPU 数、CPU/内存、机型）
4. **挂载**（OSS bucket、NAS volume）
5. **超参**（以环境变量传入）

它会在内部 GPU 集群上调度、配好节点间网络（条件允许走 RDMA）、跑你的任务、推流日志、跑完销毁。按 GPU 秒计费。还有一个比常规便宜约 40% 的抢占式池子——只要你的任务能扛住被回收。读完这一篇，你的任务就能扛住。

## DSW vs DLC，一次定下来

迭代脚本时用 DSW。下面任何一条命中，就切到 DLC：

- 超过一张 GPU
- 无人值守超过 4 小时
- 需要跨节点 RDMA / NCCL
- 想 fan-out 一组超参扫描
- 想用抢占式 GPU 省钱

切换通常无痛——DLC 接受你在 DSW 用的同一个镜像。

## Job framework：PyTorchJob，唯一你需要关心的

DLC 支持几种 "Job framework"——TensorFlowJob、PyTorchJob、MPIJob、ElasticBatchJob。现代工作选 **PyTorchJob**：它会在每个容器里设好 `MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`、`RANK`、`LOCAL_RANK`，`torchrun` 和 `torch.distributed.init_process_group` 直接就能用。

具体来说，2 节点 × 4 GPU 的 PyTorchJob 里，`torchrun` 看到的环境是：

```
WORLD_SIZE=8       RANK=0..7      LOCAL_RANK=0..3
MASTER_ADDR=<rank-0 主机名>     MASTER_PORT=23456
```

你的训练脚本完全不用关心 DLC，这才是关键。

## 一个真实的 8 卡 LLM SFT 任务

假设我们要给 7B 模型（这里用 Qwen2.5-7B，换别的都行）在几千条指令样本上做 SFT。训练脚本是纯 Hugging Face，不带任何 PAI 依赖：

```python
# train_sft.py —— 在笔记本、DSW、DLC 上跑都一样
import os, torch
from datasets import load_from_disk
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)

MODEL  = os.environ.get("MODEL_PATH", "/mnt/models/Qwen2.5-7B")
DATA   = os.environ.get("DATA_PATH",  "/mnt/data/sft_train")
OUTPUT = os.environ.get("OUTPUT_DIR", "/mnt/output/run-001")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
ds  = load_from_disk(DATA)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

args = TrainingArguments(
    output_dir=OUTPUT,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    logging_steps=10,
    deepspeed=os.environ.get("DEEPSPEED_CONFIG"),  # 可选
    ddp_find_unused_parameters=False,
    report_to="none",
)

Trainer(model=model, tokenizer=tok, args=args, train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False)).train()
```

用 PAI Python SDK 提交：

```python
# submit_dlc.py
import os
from pai.session import setup_default_session
from pai.job import TrainingJob, TrainingJobSpec
from pai.image import retrieve

setup_default_session(region_id="cn-shanghai")

image = retrieve(
    framework_name="PyTorch",
    framework_version="2.4",
    accelerator_type="GPU",
).image_uri

spec = TrainingJobSpec(
    display_name="qwen25-7b-sft-001",
    framework="PyTorchJob",
    image_uri=image,
    instance_type="ecs.gn7e-c12g1.3xlarge",   # 1 x A100-40GB / worker
    instance_count=2,                          # 2 个 worker
    accelerator_count_per_node=4,              # 每个 4 卡 => 共 8 卡
    code_dir="./code",                         # 自动上传到 OSS，挂在 /ml/code
    command=(
        "cd /ml/code && "
        "torchrun --nproc_per_node=4 "
        "--nnodes=$WORLD_SIZE_NODES --node_rank=$NODE_RANK "
        "--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT "
        "train_sft.py"
    ),
    environment_variables={
        "MODEL_PATH": "/mnt/data/models/Qwen2.5-7B",
        "DATA_PATH":  "/mnt/data/sft_train",
        "OUTPUT_DIR": "/mnt/output/qwen25-7b-sft-001",
    },
    inputs={
        "data":   "oss://my-bucket/datasets/sft_train/",
        "models": "oss://my-bucket/models/Qwen2.5-7B/",
    },
    outputs={
        "output": "oss://my-bucket/checkpoints/qwen25-7b-sft-001/",
    },
)

job = TrainingJob.create(spec)
print("Job ID:", job.job_id)
print("控制台:", job.console_url)
job.wait_for_completion(show_logs=True)
```

跑起来时，DLC 做的事：

1. 在每个 worker 上拉 PyTorch 2.4 镜像
2. 把 OSS 上的数据集和模型目录挂到 `/mnt/data/...`
3. 设好 `MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`、`RANK`、`LOCAL_RANK`、`NODE_RANK` 等
4. 在每个 worker 上同时执行 `torchrun` 命令
5. 把 stdout/stderr 推流到控制台（`wait_for_completion(show_logs=True)` 也会推到本地终端）
6. 任务完成时把 `/mnt/output/...` 同步回 OSS

写这篇时的成本：2 × 4 × A100 大约 ¥48/小时（按需池）。3 epoch、5 万条样本的 SFT 大约 4 小时，约 ¥200。抢占式不被回收的话约 ¥120。

## 数据加载：决定你 GPU 是不是空转的关键

DLC 训练吞吐最大的杠杆是你怎么读数据。三种方案，按我自己的使用频率排序：

1. **OSS-FUSE 流式读** —— SDK 给你的默认方式。简单、零代码改动。适合大文件顺序读（模型权重、parquet 分片）。**不适合**几百万个小文件。
2. **预先打成 webdataset/litdata `.tar` 写到 OSS** —— DataLoader 顺序读 tar，边读边解码。规模化的图像/音频我都用这套。
3. **任务启动时同步到本地 NVMe** —— 脚本第一步 `ossutil cp -r oss://... /local/data/`，然后从 `/local/data/` 训。多花 2-10 分钟启动时间，每步开销消失。> 2 小时的任务都值得。

> **真实经验：** 盯着 DLC dashboard 上的 GPU 利用率。如果 `nvidia-smi` 显示 A100 只有 30% 占用，瓶颈是数据加载，不是计算。我之前一个 LLaVA 任务，从 OSS-FUSE 随机访问换到本地 NVMe，36 GPU-小时直接降到 14。

## 抗住抢占

抢占式 GPU 大约便宜 40%，但集群可能在 30 秒预警之后把你的节点收回去。让任务能扛住抢占需要：

1. **频繁 checkpoint 到 OSS。** 上面例子里 `save_steps=200` 对 3 epoch SFT 够用。
2. **脚本要能续训。** Hugging Face `Trainer` 自带：传 `resume_from_checkpoint=True`，且 `output_dir` 里有 `checkpoint-*` 就行。
3. **任务规范里设 `auto_recovery=True`。** DLC 会重新拉起被回收的 worker；你的脚本通过续训逻辑自动接上。

```python
spec = TrainingJobSpec(
    ...,
    spot_spec={"enable": True, "max_price_ratio": 0.6},
    fault_tolerance={"auto_recovery": True, "max_retries": 5},
)
```

陷阱：每次 `auto_recovery` 重试是从最近 checkpoint 重启**整个任务**，不是从死掉的 worker 那一步。Checkpoint 间隔太大，单次重试可能丢几小时。`save_steps` 调到让一次重试损失 < 总时长 10%。

## 日志、指标、调试

DLC 把 stdout/stderr 推到控制台 "日志" 标签下，按 worker 分。控制台搜索做 grep 凑合。要更精细的，SDK 提供：

```python
for line in job.stream_logs(worker_index=0):
    print(line)
```

硬件指标（GPU 利用率、显存、网络）在 "监控" 标签里按 worker 可视化。两个真正要看的：

- **GPU 利用率 < 70%** → 数据加载瓶颈
- **GPU 显存接近 100%** → 在第 5000 步 OOM 之前先减 batch size 或者开 gradient checkpointing

更深的调试，在 OSS 上挂一个 TensorBoard 日志目录，TrainingArguments 加 `report_to="tensorboard"`。PAI 控制台自带 TensorBoard 查看器，可以指向任意 OSS 路径。

## 下一篇

第四篇会拿这个 DLC 任务产出的 checkpoint，通过 PAI-EAS 部署上线——包括我在三次发布里都踩到的冷启动陷阱，以及 PyTorch 模型在 Image 模式 vs Processor 模式下的取舍。
