---
title: "阿里云 PAI 实战（四）：PAI-EAS——模型部署、冷启动、以及 TPS 谎言"
date: 2026-03-08 09:00:00
tags:
  - 阿里云 PAI
  - 机器学习
  - PAI-EAS
  - 模型部署
  - 推理服务
categories: 阿里云 PAI
lang: zh-CN
mathjax: false
series: aliyun-pai
series_title: "阿里云 PAI 实战手册"
series_order: 4
description: "把第三篇训出来的 Qwen-7B 部署成生产级 EAS endpoint，对比 Image 模式和 Processor 模式，解释为什么 dashboard 上的 TPS 在骗你，以及 warm pool 怎么设才能让你凌晨三点的 p99 不爆。"
disableNunjucks: true
---

EAS 是 PAI 账单从 "理论" 变成 "实打实" 的地方。DSW 在你工位前一小时几块钱；DLC 任务是一次性支出；EAS 只要在那挂着，不管有没有人调，就在 7×24 烧钱。好消息是 EAS 也是 PAI 里最像生产系统的那一块——自动扩缩、蓝绿、流量切分、健康检查，所有你迟早要自己造轮子的东西它都给你了。

这一篇上线一个真实 endpoint：把第三篇训出来的 Qwen-7B 部署成 `qwen-7b-prod`，再用 Python 客户端调起来。途中会讲两种部署模式、至少让我搞砸过一次商务 demo 的冷启动陷阱，以及为什么 dashboard 上那个 TPS 数字根本不是你想的那个意思。

## EAS 到底是什么

PAI-EAS（Elastic Algorithm Service）是模型推理服务平台。你给它一个模型工件 + 一个加载/调用方式，它给你 HTTPS endpoint，外加自动扩缩、监控、流量路由。底层是在 GPU/CPU 集群上跑 Pod，前面挂内部负载均衡。

一上来要选定的两种部署模式：

- **Image 模式** —— 你推一个 Docker 镜像，里面塞任何 HTTP 服务（FastAPI、Triton、vLLM、自己撸的也行）。EAS 只负责跑容器、路由、扩副本。容器里的一切由你掌控。
- **Processor 模式** —— 你写一个 Python 类，实现 `initialize()` 和 `process()` 方法（外加一份 YAML）；EAS 提供 HTTP 服务、请求路由、模型加载钩子。代码少，但控制力也少。

PyTorch / Hugging Face / vLLM 类负载我默认用 **Image 模式**。Processor 模式适合 sklearn 那种形状的模型和遗留代码。LLM 一律 Image + vLLM 或 SGLang，没得商量。

## 部署 Qwen-7B SFT 模型

第三篇的 checkpoint 在 `oss://my-bucket/checkpoints/qwen25-7b-sft-001/checkpoint-final/`。我们用 vLLM 镜像把它包起来，部署成名为 `qwen-7b-prod` 的 endpoint。

### 第 1 步：选或构建镜像

PAI 官方有 vLLM 镜像：`pai-image-vllm:0.6-cu124-py310`。entrypoint 设对了就在 `/v1/chat/completions` 暴露 OpenAI 兼容 API。大多数 LLM 部署根本不需要自定义镜像——预置镜像 + 正确启动命令 + 几个环境变量就够了。

### 第 2 步：用 SDK 写部署脚本

```python
# deploy_qwen.py
import os
from pai.session import setup_default_session
from pai.predictor import Predictor
from pai.model import Model, InferenceSpec, container_serving_spec

setup_default_session(region_id="cn-shanghai")

inf_spec = container_serving_spec(
    image_uri="pai-image-vllm:0.6-cu124-py310",
    command=(
        "python -m vllm.entrypoints.openai.api_server "
        "--model /mnt/model "                # OSS 挂载路径
        "--served-model-name qwen-7b-prod "
        "--port 8000 --host 0.0.0.0 "
        "--max-model-len 8192 "
        "--gpu-memory-utilization 0.9 "
        "--enable-prefix-caching"
    ),
    port=8000,
    health_check={"path": "/health", "initial_delay_seconds": 120},
)

m = Model(
    model_data="oss://my-bucket/checkpoints/qwen25-7b-sft-001/checkpoint-final/",
    inference_spec=inf_spec,
)

predictor: Predictor = m.deploy(
    service_name="qwen-7b-prod",
    instance_type="ecs.gn7e-c12g1.3xlarge",   # 1 x A100-40GB
    instance_count=1,
    options={
        "metadata.rpc.batching": True,
        "metadata.rpc.keepalive": 60000,
    },
    autoscaler={
        "min_replicas": 1,
        "max_replicas": 4,
        "metric": "QPS",
        "threshold": 8,            # 单副本 > 8 QPS 就扩容
        "scale_in_cooldown": 600,  # 缩容前等 10 分钟
    },
)

print("Endpoint:", predictor.endpoint)
print("Token  :", predictor.access_token)   # 保密
```

### 第 3 步：客户端调用

EAS endpoint 既有 VPC 内网地址，也可以选公网地址。两者都需要 `Authorization` header。

```python
# client.py
import os, requests, json

URL   = os.environ["EAS_ENDPOINT"]   # 形如 https://1234567890.cn-shanghai.pai-eas.aliyuncs.com/api/predict/qwen-7b-prod
TOKEN = os.environ["EAS_TOKEN"]

resp = requests.post(
    f"{URL}/v1/chat/completions",
    headers={"Authorization": TOKEN, "Content-Type": "application/json"},
    json={
        "model": "qwen-7b-prod",
        "messages": [
            {"role": "system", "content": "你是一名资深后端工程师。"},
            {"role": "user",   "content": "用两句话解释幂等键。"},
        ],
        "max_tokens": 256,
        "temperature": 0.2,
    },
    timeout=60,
)
print(resp.json()["choices"][0]["message"]["content"])
```

完事。一个生产形态、自动扩缩的 LLM endpoint 就上线了。

## 冷启动陷阱

这一段是文档没把你警告到位的部分。

EAS 从 1 副本扩到 2 副本时，新副本必须经历：

1. 调度到 GPU 节点（10-60 秒，取决于库存）
2. 拉镜像（30-120 秒，取决于镜像大小和缓存命中）
3. 挂 OSS 模型目录、把权重 load 到 GPU 显存（7B 需要 30-180 秒，70B 需要 5-15 分钟）
4. 通过健康检查
5. 开始接流量

7B 完整冷启动 2-5 分钟，70B 可能 10-20 分钟。**这整段窗口里，原副本要扛 100% 增量流量。** 如果你的扩容触发是 8 QPS/副本、流量正在快速上涨，那么单副本现在要顶 24 QPS 等第二个副本起来，p99 就直接糊脸。

三种缓解，按我对它们的信任度排序：

1. **`min_replicas` 设大于 1。** 任何面向 C 端的服务我都设 ≥ 2，哪怕低峰期 "浪费"。多一个副本的成本远低于 5 分钟故障的成本。
2. **定时打 `/health` 预热。** 不能加快冷启动本身，但能避免空闲 GPU 显存被回收。
3. **开 prefix caching 和持久化 KV 缓存**（vLLM `--enable-prefix-caching`）。也不影响冷启动，但聊天形态流量稳态延迟能降 30-50%。

> **真实经验：** 自己测一下冷启动——控制台手动从 0 扩到 1，掐表。测出来多少，那就是 "被流量打爆" 时的最坏恢复时间。容量按这个数字规划。

## TPS 指标在骗你（至少在误导你）

EAS dashboard 有一张 "TPS" / "QPS" 图。对一个聊天 LLM endpoint，**这个数字几乎没用**：

- 返回 5 个 token 的请求和返回 500 个 token 的请求计数一样
- 100% prefill（长 prompt 短回复）是 GPU bound；100% decode 是带宽 bound；它们在 QPS 图上长得一模一样
- "平均延迟" 由最长回复主导，不反映典型用户体验

应该看的指标：

| 指标 | 来源 | 为什么 |
|---|---|---|
| `vllm:time_to_first_token_seconds` | vLLM Prometheus exporter | 用户**感受到**的就是这个 |
| `vllm:time_per_output_token_seconds` | vLLM Prometheus exporter | 流式吞吐感受 |
| `vllm:gpu_cache_usage_perc` | vLLM Prometheus exporter | 还有多久开始驱逐 KV 缓存 |
| EAS 副本数 | EAS dashboard | 自动扩缩有没有按预期工作 |
| EAS pending 请求数 | EAS dashboard | 请求是不是在排队 |

vLLM 的 `/metrics` endpoint 抓到 ARMS 或 Prometheus 即可。PAI 也有自带监控视图，但默认不暴露 LLM 专用计数器。

## Image vs Processor，一张表

| | Image 模式 | Processor 模式 |
|---|---|---|
| 代码形态 | 任何 HTTP 服务 | 子类 + YAML |
| LLM 部署（vLLM/SGLang） | 原生支持 | 难搞 |
| 流式响应 | 原生 | 能但别扭 |
| 自定义依赖 | Dockerfile 里 `pip install` | 只能用预置镜像 |
| 第一次部署耗时 | 30 分钟（如果你 Docker 不熟） | 5 分钟 |
| 适用场景 | LLM、视觉模型、所有现代东西 | 表格模型、sklearn、遗留代码 |

我已经一年多没部署过 Processor 模式的服务了。

## 蓝绿和 A/B 路由

EAS 同一个逻辑 endpoint 下可以挂两个版本，按比例分流。LLM 滚动发布我用的套路：

1. `qwen-7b-prod` v1 部署，100% 流量
2. `qwen-7b-prod` v2 并行部署，0% 流量但已预热
3. 几个小时内把 v2 从 5% → 25% → 50% → 100%，盯着错误率和延迟
4. v2 在 100% 稳定一天后销毁 v1

SDK 调用是 `predictor.update_traffic_split(...)`。控制台也行。关键是**永远别把 v2 直接推到 100%**，哪怕你 "本地测过了"——线上流量永远能找到你 eval 集没覆盖到的 prompt。

## 下一篇

第五篇是这个系列的收尾，老老实实对比 **PAI-Designer**（拖拽式流程构建）和 **PAI-QuickStart**（模型库一键部署）。两个都有正经用法，也都有被过度推销的倾向；我会讲清楚什么时候它们真的比 DSW + DLC + EAS 这条路更香。
